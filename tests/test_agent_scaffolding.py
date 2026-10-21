from app.agent import AgentState, HeuristicDecisionProvider, ToolDefinition, ToolRegistry
from app.agent.router import JevFallbackRouter


def test_registry_register_and_execute():
    registry = ToolRegistry()
    registry.register(ToolDefinition("echo", "Echo input", lambda args: args["value"]))
    assert registry.execute("echo", {"value": "ok"}) == "ok"

def test_registry_rejects_duplicates():
    registry = ToolRegistry()
    tool = ToolDefinition("echo", "Echo", lambda _: None)
    registry.register(tool)
    try:
        registry.register(tool)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate tools must be rejected")

def test_heuristic_router_uses_weather_tool():
    registry = ToolRegistry()
    registry.register(ToolDefinition("weather", "Weather", lambda _: None))
    state = AgentState("test")
    state.add("user", "What is the weather?")
    decision = JevFallbackRouter(HeuristicDecisionProvider(), registry).route(state)
    assert decision.action == "use_tool"
    assert decision.tool_name == "weather"
    assert state.pending_decision is decision
