from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.electrical.calculators import calculate_ohms_law
from app.providers.base import ProviderError, create_provider
from app.rag.core import VectorStore, hash_embedding
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Session,
    SourceCitation,
    ToolCall,
)
from app.session_store import SessionStore
from app.vision.api import create_vision_router
from app.vision.detector import YoloDetector

settings = get_settings()
store = SessionStore(settings.sqlite_path)
vector_store = VectorStore(settings.vector_db_path)
app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(create_vision_router(YoloDetector(settings.yolo_model_path or "missing.pt")))

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


def _calculation(message: str) -> ToolCall | None:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(k?Ω|ohm).*?(\d+(?:\.\d+)?)\s*V|"
        r"(\d+(?:\.\d+)?)\s*V.*?(\d+(?:\.\d+)?)\s*(k?Ω|ohm)",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    if match.group(1) is not None:
        resistance_value, resistance_unit, voltage_value = match.group(1), match.group(2), match.group(3)
    else:
        voltage_value, resistance_value, resistance_unit = match.group(4), match.group(5), match.group(6)
    voltage = float(voltage_value)
    resistance = float(resistance_value) * (1000 if resistance_unit.lower().startswith("k") else 1)
    result = calculate_ohms_law(voltage=voltage, resistance=resistance)
    return ToolCall(
        tool="calculate_ohms_law",
        arguments={"voltage": voltage, "resistance": resistance},
        result=result,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or store.create()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    store.append(session_id, ChatMessage(role="user", content=request.message))
    tool_calls: list[ToolCall] = []
    calculation = _calculation(request.message)
    if calculation:
        tool_calls.append(calculation)

    sources: list[SourceCitation] = []
    retrieved = vector_store.search(hash_embedding(request.message), settings.rag_top_k)
    context = ""
    if retrieved and retrieved[0].score > 0.15:
        context = "\n\n".join(
            f"[{item.source}{f', page {item.page}' if item.page else ''}]\n{item.content}"
            for item in retrieved
        )
        sources = [
            SourceCitation(
                document_id=item.document_id,
                source=item.source,
                page=item.page,
                score=item.score,
            )
            for item in retrieved
        ]

    messages = [message.model_dump() for message in store.messages(session_id)]
    if context:
        messages.insert(
            0,
            {
                "role": "system",
                "content": "Use the following untrusted reference context only when relevant. Cite the provided source names and do not invent citations.\n\n"
                + context,
            },
        )
    if calculation:
        messages.append(
            {
                "role": "system",
                "content": f"A deterministic calculation tool returned: {calculation.result}. Explain it clearly and do not redo the arithmetic.",
            }
        )
    try:
        answer = create_provider(settings.llm_provider, settings).chat(messages)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Configured LLM provider request failed") from exc
    store.append(session_id, ChatMessage(role="assistant", content=answer))
    return ChatResponse(session_id=session_id, answer=answer, tool_calls=tool_calls, sources=sources)


@app.get("/api/sessions/{session_id}", response_model=Session)
def get_session(session_id: str) -> Session:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/chat", response_model=ChatResponse, include_in_schema=False)
def legacy_chat(request: ChatRequest) -> ChatResponse:
    return chat(request)
