from __future__ import annotations

import logging
import re
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.prompts import image_system_prompt, text_system_prompt
from app.agent.vision import decide_image_route
from app.config import get_settings
from app.electrical.calculators import calculate_ohms_law
from app.providers.base import ProviderError, create_provider
from app.rag.core import VectorStore, hash_embedding
from app.rag.dense import search_dense
from app.rag.kuphaldt import DocumentChunk
from app.rag.rerank import rerank_chunks
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Session,
    SessionSummary,
    SourceCitation,
    ToolCall,
)
from app.session_store import SessionStore
from app.vision.api import create_vision_router
from app.vision.detector import YoloDetector

logger = logging.getLogger(__name__)
settings = get_settings()
store = SessionStore(settings.sqlite_path)
vector_store = VectorStore(settings.vector_db_path)
app = FastAPI(title=settings.app_name, debug=settings.debug)
detector = YoloDetector(
    settings.yolo_model_path,
    model_repo=settings.yolo_model_repo,
    model_filename=settings.yolo_model_filename,
    runtime=settings.yolo_model_runtime,
    device=settings.yolo_device,
)
app.include_router(
    create_vision_router(
        detector,
        upload_dir=settings.upload_dir,
        max_upload_bytes=settings.max_upload_bytes,
    )
)

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


def _dense_retrieve(message: str) -> list[tuple[DocumentChunk, float]]:
    index_path = Path(settings.dense_index_path)
    if not (index_path / "chunks.faiss").is_file():
        return []
    try:
        candidate_k = settings.reranker_candidate_k if settings.reranker_enabled else settings.rag_top_k
        candidates = search_dense(index_path, message, candidate_k)
        if not settings.reranker_enabled:
            return candidates
        try:
            return rerank_chunks(
                message,
                candidates,
                settings.rag_top_k,
                model_name=settings.reranker_model,
                batch_size=settings.reranker_batch_size,
            )
        except (OSError, RuntimeError, ValueError):
            return candidates[: settings.rag_top_k]
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return []


def _load_uploaded_image(image_id: str) -> tuple[bytes, str]:
    safe_name = Path(image_id).name
    if safe_name != image_id:
        raise HTTPException(status_code=400, detail="invalid image_id")
    image_path = Path(settings.upload_dir) / safe_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    content = image_path.read_bytes()
    content_type = guess_type(image_path.name)[0] or "image/png"
    return content, content_type


def _fallback_session_title(message: str) -> str:
    words = re.findall(r"\w[\w'/-]*", message, re.UNICODE)
    ignored = {"a", "an", "and", "are", "can", "do", "does", "how", "i", "in", "is", "me", "my", "please", "the", "to", "what", "why"}
    useful = [word for word in words if word.lower() not in ignored]
    selected = (useful or words)[:4]
    if len(selected) >= 3:
        return " ".join(selected).title()
    if selected:
        return " ".join(selected).title() + " Question"
    return "New Electrical Question"


def _session_title(message: str) -> str:
    fallback = _fallback_session_title(message)
    prompt = (
        "Create a concise session title from the user's first question. "
        "Return only 3 to 5 words, in title case, with no punctuation, explanation, or quotation marks."
    )
    try:
        provider = create_provider(settings.llm_provider, settings)
        raw = provider.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ])
        title = re.sub(r"[^\w'/-]+", " ", raw, flags=re.UNICODE).strip()
        words = title.split()
        if 3 <= len(words) <= 5:
            return " ".join(words).title()
    except Exception as exc:  # noqa: BLE001 - title generation must not block chat
        logger.debug("Session title generation failed: %s", exc)
    return fallback


def _vision_messages(
    session_id: str,
    image_bytes: bytes,
    content_type: str,
    system_prompt: str | None = None,
) -> list[dict]:
    import base64

    messages = [{"role": "system", "content": system_prompt or image_system_prompt()}]
    messages.extend(message.model_dump() for message in store.messages(session_id))
    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] = [
                {"type": "text", "text": message["content"]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
                    },
                },
            ]
            break
    return messages


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    created_session = request.session_id is None
    session_id = request.session_id or store.create()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    first_message = not store.messages(session_id)
    store.append(session_id, ChatMessage(role="user", content=request.message))
    if first_message:
        store.set_title(session_id, _session_title(request.message))
    session_title = store.get(session_id).title
    tool_calls: list[ToolCall] = []
    calculation = _calculation(request.message)
    if calculation:
        tool_calls.append(calculation)

    if request.image_id:
        image_bytes, content_type = _load_uploaded_image(request.image_id)
        logger.info(
            "image_processing image_id=%s content_type=%s size_bytes=%d session_id=%s",
            request.image_id,
            content_type,
            len(image_bytes),
            session_id,
        )
        provider = create_provider(settings.llm_provider, settings)
        try:
            decision = decide_image_route(
                provider,
                request.message,
                image_bytes,
                content_type,
                yolo_available=detector.available,
            )
        except ProviderError as exc:
            if created_session:
                store.delete(session_id)
            raise HTTPException(status_code=503, detail=f"image routing provider unavailable: {exc}") from exc
        except (ValueError, OSError) as exc:
            if created_session:
                store.delete(session_id)
            raise HTTPException(status_code=502, detail=f"image routing decision failed: {exc}") from exc

        tool_calls.append(
            ToolCall(
                tool="route_image",
                arguments={"image_id": request.image_id},
                result={"route": decision.route, "rationale": decision.rationale},
            )
        )
        if decision.route == "yolo":
            detections = detector.detect(image_bytes)
            detection_result = {
                "available": detector.available,
                "error": detector.error,
                "detections": [d.__dict__ for d in detections],
            }
            tool_calls.append(
                ToolCall(
                    tool="detect_circuit_components",
                    arguments={"image_id": request.image_id},
                    result=detection_result,
                )
            )
            tool_calls.append(
                ToolCall(
                    tool="verify_with_vision",
                    arguments={"image_id": request.image_id, "mode": "hybrid"},
                    result={
                        "detector_detections": len(detections),
                        "vision_cross_check": True,
                    },
                )
            )
            messages = _vision_messages(
                session_id,
                image_bytes,
                content_type,
                system_prompt=image_system_prompt(detection_result),
            )
        else:
            messages = _vision_messages(session_id, image_bytes, content_type)
        try:
            answer = provider.chat(messages)
        except ProviderError as exc:
            if created_session:
                store.delete(session_id)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            if created_session:
                store.delete(session_id)
            raise HTTPException(status_code=502, detail="Configured LLM provider request failed") from exc
        store.append(session_id, ChatMessage(role="assistant", content=answer))
        return ChatResponse(session_id=session_id, session_title=session_title, answer=answer, tool_calls=tool_calls)

    sources: list[SourceCitation] = []
    dense_retrieved = _dense_retrieve(request.message)
    context = ""
    if dense_retrieved:
        context = "\n\n".join(
            f"SOURCE: {chunk.source_path}\n{chunk.content}" for chunk, _score in dense_retrieved
        )
        sources = [
            SourceCitation(
                document_id=chunk.document_id,
                source=chunk.source_path,
                page=None,
                score=score,
            )
            for chunk, score in dense_retrieved
        ]
    else:
        retrieved = vector_store.search(hash_embedding(request.message), settings.rag_top_k)
        if not retrieved or retrieved[0].score <= 0.15:
            retrieved = []
        if retrieved:
            context = "\n\n".join(
                f"SOURCE: {item.source}{f', page {item.page}' if item.page else ''}\n{item.content}"
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

    messages = [{
        "role": "system",
        "content": text_system_prompt(
            context,
            calculation.result if calculation else None,
        ),
    }]
    messages.extend(message.model_dump() for message in store.messages(session_id))
    try:
        answer = create_provider(settings.llm_provider, settings).chat(messages)
    except ProviderError as exc:
        if created_session:
            store.delete(session_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if created_session:
            store.delete(session_id)
        raise HTTPException(status_code=502, detail="Configured LLM provider request failed") from exc
    store.append(session_id, ChatMessage(role="assistant", content=answer))
    return ChatResponse(session_id=session_id, session_title=session_title, answer=answer, tool_calls=tool_calls, sources=sources)


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return store.list_sessions()


@app.get("/api/sessions/{session_id}", response_model=Session)
def get_session(session_id: str) -> Session:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    store.delete(session_id)


@app.post("/chat", response_model=ChatResponse, include_in_schema=False)
def legacy_chat(request: ChatRequest) -> ChatResponse:
    return chat(request)
