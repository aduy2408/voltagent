from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class SourceCitation(BaseModel):
    document_id: str
    source: str
    page: int | None = None
    score: float | None = None


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None
    image_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    session_title: str | None = None
    answer: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)


class Session(BaseModel):
    id: str
    title: str = "New session"
    created_at: datetime
    messages: list[ChatMessage] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    message_count: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
