from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from app.agent.prompts import routing_system_prompt
from app.providers.base import Provider

ImageRoute = Literal["yolo", "vision"]


@dataclass(frozen=True, slots=True)
class ImageRouteDecision:
    route: ImageRoute
    rationale: str
    raw: str


def _parse_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise TypeError("routing decision was not a JSON object")
    return value


def decide_image_route(
    provider: Provider,
    message: str,
    image_bytes: bytes,
    content_type: str,
    *,
    yolo_available: bool,
) -> ImageRouteDecision:
    """Ask the configured LLM to choose the image tool, not to answer the user yet."""
    data_url = f"data:{content_type};base64,{__import__('base64').b64encode(image_bytes).decode()}"
    prompt = routing_system_prompt(yolo_available)
    raw = provider.chat(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message or "Classify this image for the next tool."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
    )
    parsed = _parse_json(raw)
    route = parsed.get("route")
    if route not in {"yolo", "vision"}:
        raise ValueError(f"unsupported image route: {route!r}")
    if route == "yolo" and not yolo_available:
        return ImageRouteDecision("vision", "YOLO is unavailable, so native vision is safer.", raw)
    return ImageRouteDecision(route, str(parsed.get("rationale", "")), raw)
