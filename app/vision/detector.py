"""Optional YOLO adapter. Importing this module never requires ultralytics."""
import importlib.util
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_INFERENCE_LOCKS_LOCK = threading.Lock()


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None and _cuda_available()


def choose_runtime(runtime: str = "auto") -> str:
    """Return the preferred artifact format for this machine.

    TensorRT is selected only when both CUDA and the TensorRT Python bindings are
    present. ONNX is the portable accelerated fallback. The plain PyTorch
    checkpoint remains the final fallback when no exported artifact is present.
    """
    normalized = runtime.strip().lower()
    if normalized in {"tensorrt", "engine"}:
        return "tensorrt"
    if normalized in {"onnx"}:
        return "onnx"
    if normalized in {"pytorch", "torch", "pt"}:
        return "pytorch"
    if normalized != "auto":
        raise ValueError("YOLO runtime must be auto, tensorrt, onnx, or pytorch")
    if _tensorrt_available():
        return "tensorrt"
    if _cuda_available() or importlib.util.find_spec("onnxruntime") is not None:
        return "onnx"
    return "pytorch"


def resolve_model_artifact(checkpoint: Path, runtime: str) -> tuple[Path, str]:
    """Resolve an exported sibling artifact, falling back to the source checkpoint."""
    suffixes = {
        "tensorrt": (".engine",),
        "onnx": (".onnx",),
        "pytorch": (".pt",),
    }
    selected = choose_runtime(runtime)
    candidates = [checkpoint.with_suffix(suffix) for suffix in suffixes[selected]]
    if selected == "pytorch":
        candidates.insert(0, checkpoint)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, selected
    if selected != "pytorch" and checkpoint.is_file():
        return checkpoint, "pytorch"
    return candidates[0], selected


class YoloDetector:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        confidence: float = 0.25,
        *,
        model_repo: str | None = None,
        model_filename: str = "yolo26m_obb_16class_aug.pt",
        runtime: str = "auto",
        device: str = "auto",
    ):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.runtime = runtime
        self.device = device
        self.selected_runtime: str | None = None
        self.selected_model: Path | None = None
        self.confidence = confidence
        self._model: Any = None
        self._cache_key: tuple[str, str] | None = None
        self._inference_lock: threading.Lock | None = None
        self.error: str | None = None
        if self.checkpoint is not None and not self.checkpoint.exists():
            self.error = f"checkpoint not found: {self.checkpoint}"

    @property
    def available(self) -> bool:
        return self._model is not None or self.error is None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self.error is not None:
            return False
        try:
            from ultralytics import YOLO

            checkpoint = self.checkpoint
            if checkpoint is not None and not checkpoint.exists():
                self.error = f"checkpoint not found: {checkpoint}"
                return False
            if checkpoint is None:
                if not self.model_repo:
                    self.error = "no YOLO checkpoint or Hugging Face model configured"
                    return False
                from huggingface_hub import hf_hub_download

                checkpoint = Path(
                    hf_hub_download(repo_id=self.model_repo, filename=self.model_filename)
                )
            self.checkpoint = checkpoint
            if checkpoint.suffix.lower() == ".pt":
                checkpoint, selected_runtime = resolve_model_artifact(checkpoint, self.runtime)
                if not checkpoint.exists():
                    self.error = (
                        f"no {selected_runtime} artifact found beside checkpoint: {checkpoint}"
                    )
                    return False
            else:
                selected_runtime = {
                    ".engine": "tensorrt",
                    ".onnx": "onnx",
                }.get(checkpoint.suffix.lower(), "pytorch")
            self.selected_model = checkpoint
            self.selected_runtime = selected_runtime
            cache_key = (str(checkpoint.resolve()), selected_runtime)
            with _MODEL_CACHE_LOCK:
                self._model = _MODEL_CACHE.get(cache_key)
                if self._model is None:
                    self._model = YOLO(str(checkpoint))
                    _MODEL_CACHE[cache_key] = self._model
            with _INFERENCE_LOCKS_LOCK:
                self._inference_lock = _INFERENCE_LOCKS.setdefault(cache_key, threading.Lock())
            self._cache_key = cache_key
            return True
        except ImportError as exc:
            self.error = f"optional vision dependency is not installed: {exc}"
        except Exception as exc:  # noqa: BLE001 - optional model loading must fail safely
            self.error = f"could not load detector: {exc}"
        return False

    def detect(self, image: Any) -> list[Detection]:
        if not self._load():
            return []
        try:
            source = image
            if isinstance(image, (bytes, bytearray)):
                from PIL import Image

                source = Image.open(BytesIO(image)).convert("RGB")
            kwargs: dict[str, Any] = {"conf": self.confidence, "verbose": False}
            if self.device != "auto":
                kwargs["device"] = self.device
            elif self.selected_runtime == "tensorrt" or _cuda_available():
                kwargs["device"] = 0
            if self._inference_lock is None:
                return []
            with self._inference_lock:
                results = self._model(source, **kwargs)
            detections: list[Detection] = []
            for result in results:
                boxes = getattr(result, "obb", None)
                if boxes is None:
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
