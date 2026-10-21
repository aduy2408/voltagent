from __future__ import annotations

from abc import ABC, abstractmethod

from .state import AgentState, Decision


class DecisionProvider(ABC):
    @abstractmethod
    def decide(self, state: AgentState, *, available_tools: list[str] | None = None) -> Decision:
        """Choose the next action without committing to an LLM vendor."""

class HeuristicDecisionProvider(DecisionProvider):
    def decide(self, state: AgentState, *, available_tools: list[str] | None = None) -> Decision:
        message = (state.last_user_message.content if state.last_user_message else "").lower()
        tools = available_tools or []
        if "weather" in message and "weather" in tools:
            return Decision("use_tool", "Weather request detected", "weather", {})
        return Decision("respond", "No registered heuristic matched")
