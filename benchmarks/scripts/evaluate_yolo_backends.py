"""Evaluate YOLO PyTorch, ONNX, and TensorRT artifacts on Pascal VOC XML labels."""
from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from ultralytics import YOLO

CLASS_NAMES = [
    "resistor", "terminal", "crossover", "capacitor", "diode", "transistor",
    "switch", "inductor", "voltage_source", "text", "junction", "gnd", "vss",
    "integrated_circuit", "operational_amplifier", "other",
]
MERGE_MAP = {
    "resistor": {"resistor", "resistor.adjustable", "resistor.photo"},
    "terminal": {"terminal"},
    "crossover": {"crossover"},
    "capacitor": {"capacitor.adjustable", "capacitor.polarized", "capacitor.unpolarized"},
    "diode": {"diode", "diode.light_emitting", "diode.thyrector", "diode.zener"},
    "transistor": {"transistor.bjt", "transistor.fet", "transistor.photo"},
    "switch": {"switch"},
    "inductor": {"inductor", "inductor.coupled", "inductor.ferrite"},
    "voltage_source": {"voltage.ac", "voltage.battery", "voltage.dc"},
    "text": {"text"},
    "junction": {"junction"},
    "gnd": {"gnd"},
    "vss": {"vss"},
    "integrated_circuit": {"integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator"},
    "operational_amplifier": {"operational_amplifier", "operational_amplifier.schmitt_trigger"},
}
OTHER = {name for names in MERGE_MAP.values() for name in names}
OTHER |= {
    "and", "antenna", "block", "crystal", "diac", "explanatory", "fuse", "lamp", "magnetic",
    "mechanical", "microphone", "motor", "nand", "nor", "not", "optical", "optocoupler", "or",
    "probe", "probe.current", "probe.voltage", "relay", "socket", "speaker", "thyristor", "transformer",
    "triac", "unknown", "varistor", "xor",
}


def map_class(name: str) -> int:
    name = name.strip().lower()
    for index, target in enumerate(CLASS_NAMES[:-1]):
        if name in MERGE_MAP[target]:
            return index
    if name in OTHER:
        return len(CLASS_NAMES) - 1
    return len(CLASS_NAMES) - 1


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def parse_xml(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    output = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "other")
        box = obj.find("bndbox")
        if box is None:
            continue
        values = tuple(float(box.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax"))
        output.append((map_class(name), values))
    return output


def predictions(model: YOLO, image: Path, imgsz: int, conf: float, device: str) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    results = model(str(image), imgsz=imgsz, conf=conf, verbose=False, device=device)
    output = []
    for result in results:
        boxes = getattr(result, "obb", None)
        if boxes is None:
            boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        names = getattr(result, "names", {})
        for box, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            name = str(names.get(int(cls), int(cls))).strip().lower()
            output.append((map_class(name), float(score), tuple(map(float, box))))
    return output


def ap50(records: list[tuple[int, float, tuple[float, float, float, float]]], ground_truth: list[list[tuple[int, tuple[float, float, float, float]]]], class_id: int, threshold: float) -> float:
    gt_by_image = {i: [box for cls, box in gt if cls == class_id] for i, gt in enumerate(ground_truth)}
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return math.nan
    matched = {i: set() for i in gt_by_image}
    ranked = sorted(records, key=lambda item: item[1], reverse=True)
    tp, fp = [], []
    for image_id, _score, box in ranked:
        candidates = [(iou(box, gt_box), j) for j, gt_box in enumerate(gt_by_image.get(image_id, []))]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= threshold and best[1] not in matched.setdefault(image_id, set()):
            matched[image_id].add(best[1])
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    cumulative_tp = 0
    cumulative_fp = 0
    precision = []
    recall = []
    for true_positive, false_positive in zip(tp, fp):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        precision.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
        recall.append(cumulative_tp / total_gt)
    return sum(max((p for p, r in zip(precision, recall) if r >= level), default=0.0) for level in [i / 100 for i in range(101)]) / 101


def match_count(left, right, threshold: float) -> int:
    pairs = sorted((iou(a[2], b[2]), i, j) for i, a in enumerate(left) for j, b in enumerate(right) if a[0] == b[0])
    used_left, used_right = set(), set()
    count = 0
    for score, i, j in reversed(pairs):
        if score < threshold or i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        count += 1
    return count


def detection_metrics(predictions_by_image, ground_truth, threshold: float) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    for predictions_for_image, gt_for_image in zip(predictions_by_image, ground_truth):
        pairs = sorted(
            (iou(prediction[2], target[1]), pi, gi)
            for pi, prediction in enumerate(predictions_for_image)
            for gi, target in enumerate(gt_for_image)
            if prediction[0] == target[0]
        )
        used_predictions, used_gt = set(), set()
        matched = 0
        for score, pi, gi in reversed(pairs):
            if score < threshold or pi in used_predictions or gi in used_gt:
                continue
            used_predictions.add(pi)
            used_gt.add(gi)
            matched += 1
        true_positive += matched
        false_positive += len(predictions_for_image) - matched
        false_negative += len(gt_for_image) - matched
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="data/evaluation/cghd/subset")
    parser.add_argument("--output", default="benchmarks/results/yolo_accuracy_subset_2026-10-21.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--agreement-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    subset = Path(args.subset)
    images = sorted(subset.glob("drafter_*/images/*"))
    pairs = [(image, image.parents[1] / "annotations" / f"{image.stem}.xml") for image in images]
    pairs = [(image, xml) for image, xml in pairs if xml.is_file()]
    if not pairs:
        raise SystemExit("No paired JPG/XML evaluation examples found")
    ground_truth = [parse_xml(xml) for _image, xml in pairs]
    model_paths = {
        "pytorch": "data/models/circuit-component-detector/yolo26m_obb_16class_aug.pt",
        "onnx": "data/models/circuit-component-detector/yolo26m_obb_16class_aug.onnx",
        "tensorrt": "data/models/circuit-component-detector/yolo26m_obb_16class_aug.engine",
    }
    result: dict[str, Any] = {
        "dataset": "CGHD subset",
        "images": len(pairs),
        "drafters": sorted({image.parent.parent.name for image, _xml in pairs}),
        "imgsz": args.imgsz,
        "confidence_for_ap": args.conf,
        "agreement_confidence": args.agreement_conf,
        "iou_threshold": args.iou,
        "models": model_paths,
        "backends": {},
        "agreement": {},
        "limitations": [
            "This is a 50-image subset, not the model card's full 468-image validation split.",
            "Ground-truth Pascal VOC boxes are compared as axis-aligned boxes to YOLO OBB xyxy boxes.",
            "Class mapping merges the 61 CGHD labels into the model's 16 classes.",
        ],
    }
    all_predictions = {}
    for backend, path in model_paths.items():
        model = YOLO(path)
        started = time.perf_counter()
        per_image = [predictions(model, image, args.imgsz, args.conf, args.device) for image, _xml in pairs]
        elapsed = time.perf_counter() - started
        all_predictions[backend] = per_image
        records_by_class = defaultdict(list)
        for image_id, preds in enumerate(per_image):
            for cls, score, box in preds:
                records_by_class[cls].append((image_id, score, box))
        aps = [ap50(records_by_class[cls], ground_truth, cls, args.iou) for cls in range(len(CLASS_NAMES))]
        result["backends"][backend] = {
            "images": len(per_image),
            "detections": sum(len(preds) for preds in per_image),
            "mean_detections_per_image": sum(len(preds) for preds in per_image) / len(per_image),
            "map50_subset": sum(x for x in aps if not math.isnan(x)) / sum(not math.isnan(x) for x in aps),
            "per_class_ap50": {CLASS_NAMES[i]: round(ap, 6) for i, ap in enumerate(aps) if not math.isnan(ap)},
            "elapsed_seconds": round(elapsed, 3),
        }
        threshold_predictions = []
        for preds in per_image:
            threshold_predictions.append([p for p in preds if p[1] >= args.agreement_conf])
        result["backends"][backend]["metrics_at_agreement_conf"] = detection_metrics(
            threshold_predictions, ground_truth, args.iou
        )
        all_predictions[backend + "_agreement"] = threshold_predictions
    for left, right in (("pytorch", "onnx"), ("pytorch", "tensorrt"), ("onnx", "tensorrt")):
        matched = sum(match_count(a, b, args.iou) for a, b in zip(all_predictions[left + "_agreement"], all_predictions[right + "_agreement"]))
        left_count = sum(len(x) for x in all_predictions[left + "_agreement"])
        right_count = sum(len(x) for x in all_predictions[right + "_agreement"])
        result["agreement"][f"{left}_vs_{right}"] = {
            "matched_detections_iou50_class_aware": matched,
            "left_detections": left_count,
            "right_detections": right_count,
            "matched_over_left": matched / max(1, left_count),
            "matched_over_right": matched / max(1, right_count),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
