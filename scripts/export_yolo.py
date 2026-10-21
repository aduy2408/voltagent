"""Export the circuit YOLO checkpoint for the current machine.

Use ``--format auto`` in deployment scripts. It selects TensorRT when CUDA and
TensorRT are installed, otherwise ONNX. Exported files are written next to the
source checkpoint and the detector discovers them automatically.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from app.vision.detector import choose_runtime


def export_model(
    checkpoint: str | Path,
    *,
    output_format: str = "auto",
    imgsz: int = 640,
    half: bool = False,
    opset: int | None = None,
    workspace: float | None = None,
) -> Path:
    from ultralytics import YOLO

    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    target = choose_runtime(output_format)
    if target == "pytorch":
        raise ValueError("export format must be onnx or tensorrt, not pytorch")
    if target == "tensorrt" and importlib.util.find_spec("tensorrt") is None:
        raise RuntimeError("TensorRT bindings are not installed on this machine")

    model = YOLO(str(source))
    kwargs: dict[str, object] = {"format": "engine" if target == "tensorrt" else "onnx", "imgsz": imgsz}
    if half:
        kwargs["half"] = True
    if opset is not None and target == "onnx":
        kwargs["opset"] = opset
    if workspace is not None and target == "tensorrt":
        kwargs["workspace"] = workspace
    if target == "tensorrt":
        kwargs["device"] = 0
    exported = model.export(**kwargs)
    return Path(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="data/models/circuit-component-detector/yolo26m_obb_16class_aug.pt")
    parser.add_argument("--format", choices=("auto", "onnx", "tensorrt"), default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--opset", type=int)
    parser.add_argument("--workspace", type=float)
    args = parser.parse_args()
    selected = choose_runtime(args.format)
    print(f"YOLO_EXPORT_RUNTIME {selected}")
    output = export_model(
        args.checkpoint,
        output_format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        opset=args.opset,
        workspace=args.workspace,
    )
    print(f"YOLO_EXPORT_READY {output}")


if __name__ == "__main__":
    main()
