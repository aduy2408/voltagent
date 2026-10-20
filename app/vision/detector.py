"""Optional YOLO adapter. Importing this module never requires ultralytics."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


class YoloDetector:
    def __init__(self, checkpoint: str | Path = "yolov8n.pt", confidence: float = 0.25):
        self.checkpoint = Path(checkpoint)
        self.confidence = confidence
        self._model: Any = None
        self.error: str | None = None
        try:
            from ultralytics import YOLO
            if not self.checkpoint.exists():
                self.error = f"checkpoint not found: {self.checkpoint}"
            else:
                self._model = YOLO(str(self.checkpoint))
        except ImportError:
            self.error = "ultralytics is not installed"
        except Exception as exc:  # noqa: BLE001 - optional model loading must fail safely
            self.error = f"could not load detector: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, image: Any) -> list[Detection]:
        if not self.available:
            return []
        try:
            results = self._model(image, conf=self.confidence, verbose=False)
            detections: list[Detection] = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is None:
                    continue
                for box, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                    detections.append(Detection(str(names.get(int(class_id), int(class_id))), float(confidence), tuple(map(float, box))))
            return detections
        except Exception as exc:  # noqa: BLE001 - inference errors are returned as service state
            self.error = f"inference failed: {exc}"
            return []
