from __future__ import annotations

from typing import Protocol

from .decision import DecisionProvider
from .registry import ToolRegistry
from .state import AgentState, Decision


class AgentRouter(Protocol):
    def route(self, state: AgentState) -> Decision: ...

class JevFallbackRouter:
    """Optional adapter point for a Jev-like router, with local fallback."""
    def __init__(self, provider: DecisionProvider, registry: ToolRegistry) -> None:
        self.provider = provider
        self.registry = registry

    def route(self, state: AgentState) -> Decision:
        decision = self.provider.decide(state, available_tools=[tool.name for tool in self.registry.list()])
        state.pending_decision = decision
        return decision
