"""Evaluate YOLO, YOLO-plus-DeepSeek, and DeepSeek-only on a CGHD image subset.

The LLM modes are evaluated as structured component inventory extraction. This avoids
pretending that a conversational answer can be scored as detector boxes while still
measuring whether the model identifies the annotated component classes and counts.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.providers.base import ProviderError, create_provider
from benchmarks.scripts.evaluate_yolo_backends import CLASS_NAMES, map_class, predictions


def parse_xml(path: Path) -> list[int]:
    root = ET.parse(path).getroot()
    labels: list[int] = []
    for obj in root.findall("object"):
        labels.append(map_class(obj.findtext("name", "other")))
    return labels


def pairs_for_subset(subset: Path, limit: int) -> list[tuple[Path, Path]]:
    images = sorted(subset.glob("drafter_*/images/*"))
    pairs = [(image, image.parents[1] / "annotations" / f"{image.stem}.xml") for image in images]
    pairs = [(image, xml) for image, xml in pairs if xml.is_file()]
    if len(pairs) < limit:
        raise SystemExit(f"Expected at least {limit} paired CGHD images, found {len(pairs)}")
    return pairs[:limit]


def counts(labels: list[int]) -> Counter[int]:
    return Counter(labels)


def parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_inventory(value: dict[str, Any]) -> Counter[int]:
    output: Counter[int] = Counter()
    components = value.get("components", [])
    if not isinstance(components, list):
        return output
    for component in components:
        if isinstance(component, str):
            label, count = component, 1
        elif isinstance(component, dict):
            label = component.get("label", component.get("name", "other"))
            count = component.get("count", 1)
        else:
            continue
        try:
            count_int = max(0, int(count))
        except (TypeError, ValueError):
            count_int = 1
        if count_int:
            output[map_class(str(label))] += count_int
    return output


def inventory_prompt(mode: str, detector_output: list[dict[str, Any]] | None = None) -> str:
    evidence = ""
    if detector_output is not None:
        evidence = (
            "\nA local detector produced the following untrusted evidence. Use it as a hint, "
            "but verify it against the image and correct obvious errors:\n"
            + json.dumps(detector_output, sort_keys=True)
        )
    return f"""You are evaluating a circuit-component image for a benchmark. Evaluation mode: {mode}.
Inspect the attached image and return JSON only. Do not include Markdown or commentary.
Identify the visible component classes and the number of instances of each class.
Use only these canonical labels:
{", ".join(CLASS_NAMES)}
Count only visible symbols/components, not wires or empty regions. If uncertain, omit the
component rather than inventing it. Return exactly:
{{"components":[{{"label":"resistor","count":1}}],"confidence":"high|medium|low"}}
{evidence}""".strip()


def image_message(image: Path, prompt: str) -> list[dict[str, Any]]:
    content_type = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return [
        {"role": "system", "content": "Return valid JSON only. Do not reveal hidden reasoning."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{encoded}"}},
            ],
        },
    ]


def inventory_metrics(predicted: list[Counter[int]], truth: list[Counter[int]]) -> dict[str, Any]:
    tp = fp = fn = 0
    exact = 0
    count_abs_error = 0
    for prediction, target in zip(predicted, truth):
        if prediction == target:
            exact += 1
        keys = set(prediction) | set(target)
        for key in keys:
            actual, expected = prediction.get(key, 0), target.get(key, 0)
            tp += min(actual, expected)
            fp += max(0, actual - expected)
            fn += max(0, expected - actual)
            count_abs_error += abs(actual - expected)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "images": len(truth),
        "exact_inventory_match": exact / max(1, len(truth)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_absolute_count_error": count_abs_error / max(1, len(truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="data/evaluation/cghd/subset")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="benchmarks/results/cghd_vision_modes_2026-10-21.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new", type=int, default=0, help="Stop after this many new API requests; 0 means no limit")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("deepseek_only", "yolo_plus_deepseek"),
        default=["deepseek_only", "yolo_plus_deepseek"],
    )
    args = parser.parse_args()

    subset = Path(args.subset)
    pairs = pairs_for_subset(subset, args.limit)
    truth_labels = [parse_xml(xml) for _image, xml in pairs]
    truth_counts = [counts(labels) for labels in truth_labels]

    from ultralytics import YOLO

    model_path = Path("data/models/circuit-component-detector/yolo26m_obb_16class_aug.engine")
    if not model_path.is_file():
        model_path = Path("data/models/circuit-component-detector/yolo26m_obb_16class_aug.pt")
    existing: dict[str, Any] = {}
    if args.resume and Path(args.output).is_file():
        existing = json.loads(Path(args.output).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = existing.get("records", [])
    if len(records) != len(pairs):
        records = []
    if records and all("yolo_detections" in record for record in records):
        yolo_predictions = [
            [
                (
                    map_class(item["label"]),
                    float(item["confidence"]),
                    tuple(map(float, item["bbox"])),
                )
                for item in record["yolo_detections"]
            ]
            for record in records
        ]
    else:
        model = YOLO(str(model_path))
        yolo_predictions = []
        records = []
        for image, _xml in pairs:
            pred = predictions(model, image, args.imgsz, args.conf, args.device)
            yolo_predictions.append(pred)
            records.append({
                "image": str(image),
                "ground_truth_counts": {CLASS_NAMES[k]: v for k, v in truth_counts[len(records)].items()},
                "yolo_counts": {CLASS_NAMES[k]: v for k, v in counts([item[0] for item in pred]).items()},
                "yolo_detections": [
                    {"label": CLASS_NAMES[label], "confidence": score, "bbox": list(box)}
                    for label, score, box in pred
                ],
            })

    # The full YOLO box metrics are already produced by evaluate_yolo_backends. Keep this
    # runner focused on the three-mode comparison and report inventory metrics here.
    yolo_inventory = inventory_metrics(
        [counts([item[0] for item in pred]) for pred in yolo_predictions], truth_counts
    )

    llm_results: dict[str, Any] = {}
    if not args.skip_llm:
        provider = create_provider(get_settings().llm_provider, get_settings())
        new_requests = 0
        for mode in args.modes:
            predictions_for_mode: list[Counter[int]] = []
            parsed = 0
            failures = 0
            for index, (image, _xml) in enumerate(pairs):
                previous = records[index].get(mode)
                if previous is not None:
                    value = previous.get("parsed")
                    prediction = normalize_inventory(value) if isinstance(value, dict) else Counter()
                    parsed += int(value is not None)
                    failures += int(value is None)
                    predictions_for_mode.append(prediction)
                    continue
                if args.max_new and new_requests >= args.max_new:
                    break
                detector_output = None
                if mode == "yolo_plus_deepseek":
                    detector_output = [
                        {
                            "label": label,
                            "count": count,
                            "mean_confidence": round(
                                sum(item["confidence"] for item in records[index]["yolo_detections"] if item["label"] == label)
                                / max(1, count),
                                4,
                            ),
                        }
                        for label, count in records[index]["yolo_counts"].items()
                    ]
                try:
                    raw = provider.chat(image_message(image, inventory_prompt(mode, detector_output)))
                    value = parse_json(raw)
                    if value is None:
                        failures += 1
                        prediction = Counter()
                    else:
                        parsed += 1
                        prediction = normalize_inventory(value)
                    records[index][mode] = {"raw": raw, "parsed": value}
                except (ProviderError, OSError, RuntimeError, ValueError) as exc:
                    failures += 1
                    prediction = Counter()
                    records[index][mode] = {"error": str(exc)}
                predictions_for_mode.append(prediction)
                new_requests += 1
                print(
                    f"VISION_EVAL mode={mode} image={index + 1}/{len(pairs)} parsed={parsed} failures={failures}",
                    flush=True,
                )
                partial = {
                    "benchmark": "CGHD 50-image vision mode comparison",
                    "date": "2026-10-21",
                    "dataset": "data/evaluation/cghd/subset",
                    "images": len(pairs),
                    "model": str(model_path),
                    "mode": mode,
                    "completed_images": index + 1,
                    "parsed": parsed,
                    "failures": failures,
                    "records": records,
                }
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(partial, indent=2), encoding="utf-8")
                if args.sleep:
                    time.sleep(args.sleep)
            llm_results[mode] = {
                "parse_success_rate": parsed / max(1, len(pairs)),
                "request_failures": failures,
                "inventory_metrics": inventory_metrics(predictions_for_mode, truth_counts),
            }
            checkpoint = {
                "benchmark": "CGHD 50-image vision mode comparison",
                "date": "2026-10-21",
                "dataset": "data/evaluation/cghd/subset",
                "images": len(pairs),
                "drafters": sorted({image.parent.parent.name for image, _xml in pairs}),
                "model": str(model_path),
                "settings": {"imgsz": args.imgsz, "confidence": args.conf, "device": args.device},
                "deepseek_modes": llm_results,
                "records": records,
                "status": f"completed_{mode}",
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            if args.max_new and new_requests >= args.max_new:
                break

    result = {
        "benchmark": "CGHD 50-image vision mode comparison",
        "date": "2026-10-21",
        "dataset": "data/evaluation/cghd/subset",
        "images": len(pairs),
        "drafters": sorted({image.parent.parent.name for image, _xml in pairs}),
        "model": str(model_path),
        "settings": {"imgsz": args.imgsz, "confidence": args.conf, "device": args.device},
        "metrics_definition": {
            "yolo": "Instance detection metrics are in yolo_accuracy_subset_2026-10-21.json; inventory metrics below are class-count comparison.",
            "deepseek_modes": "Structured visible-component inventory compared with CGHD mapped class counts; no LLM bounding-box claim is made.",
            "class_mapping": "The existing 61-to-16 CGHD mapping from evaluate_yolo_backends.py.",
        },
        "yolo_only": {"inventory_metrics": yolo_inventory, "detection_metric_note": "See existing YOLO result artifact for mAP50 and IoU-based detection precision/recall/F1."},
        "deepseek_modes": llm_results,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("benchmark", "images", "yolo_only", "deepseek_modes")}, indent=2))


if __name__ == "__main__":
    main()
