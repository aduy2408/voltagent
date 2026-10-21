from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

@dataclass(slots=True)
class AgentMessage:
    role: Role
    content: str
    name: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass(slots=True)
class Decision:
    action: Literal["respond", "use_tool", "delegate"]
    rationale: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AgentState:
    session_id: str
    messages: list[AgentMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    pending_decision: Decision | None = None

    def add(self, role: Role, content: str, *, name: str | None = None) -> AgentMessage:
        message = AgentMessage(role=role, content=content, name=name)
        self.messages.append(message)
        return message

    @property
    def last_user_message(self) -> AgentMessage | None:
        return next((message for message in reversed(self.messages) if message.role == "user"), None)
