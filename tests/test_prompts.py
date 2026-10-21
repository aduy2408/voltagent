from app.agent.prompts import (
    GLOBAL_SYSTEM_PROMPT,
    image_system_prompt,
    routing_system_prompt,
    text_system_prompt,
)


def test_global_prompt_defines_role_format_and_citation_policy():
    assert "VoltAgent" in GLOBAL_SYSTEM_PROMPT
    assert "electrical-engineering assistant" in GLOBAL_SYSTEM_PROMPT
    assert "[Source: <source path>]" in GLOBAL_SYSTEM_PROMPT
    assert "Answer" in GLOBAL_SYSTEM_PROMPT
    assert "Caveats" in GLOBAL_SYSTEM_PROMPT


def test_text_prompt_includes_reference_context_and_calculation():
    prompt = text_system_prompt(
        "SOURCE: kuphaldt/DC/DC_2.html\nOhm's law content",
        {"current": 2.0, "unit": "A"},
    )
    assert "SOURCE: kuphaldt/DC/DC_2.html" in prompt
    assert "[Source: kuphaldt/DC/DC_2.html]" in prompt
    assert '"current": 2.0' in prompt
    assert "do not redo or contradict the arithmetic" in prompt


def test_image_prompts_keep_detector_output_untrusted():
    prompt = image_system_prompt({"detections": [], "available": True})
    assert "untrusted evidence" in prompt
    assert "DETECTOR_OUTPUT" in prompt
    assert "Do not invent component values" in prompt


def test_routing_prompt_requires_machine_readable_json():
    prompt = routing_system_prompt(True)
    assert "YOLO available: True" in prompt
    assert '"route": "yolo" or "vision"' in prompt
    assert '"rationale"' in prompt
