from app.agent.vision import decide_image_route
from app.providers.base import Provider


class FakeVisionProvider(Provider):
    def __init__(self, response: str):
        self.response = response
        self.messages = []

    def chat(self, messages):
        self.messages = list(messages)
        return self.response


def test_image_route_uses_structured_llm_decision():
    provider = FakeVisionProvider('{"route":"yolo","rationale":"hand-drawn schematic"}')

    decision = decide_image_route(
        provider,
        "Find the components",
        b"image-bytes",
        "image/png",
        yolo_available=True,
    )

    assert decision.route == "yolo"
    assert decision.rationale == "hand-drawn schematic"
    assert isinstance(provider.messages[-1]["content"], list)
    assert provider.messages[-1]["content"][1]["type"] == "image_url"


def test_image_route_downgrades_yolo_when_detector_is_unavailable():
    provider = FakeVisionProvider('{"route":"yolo","rationale":"looks like a schematic"}')

    decision = decide_image_route(
        provider,
        "Inspect it",
        b"image-bytes",
        "image/png",
        yolo_available=False,
    )

    assert decision.route == "vision"
    assert "unavailable" in decision.rationale
