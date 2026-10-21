from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.huggingface import HF_INDEX_NAME, HF_INDEX_REPO, download_published_index

YOLO_REPO = "boscochanam/circuit-component-detector"
YOLO_FILENAME = "yolo26m_obb_16class_aug.pt"


def download_yolo(output_dir: str | Path) -> Path:
    from huggingface_hub import hf_hub_download

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cached = Path(hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILENAME))
    target = destination / YOLO_FILENAME
    if target.exists():
        target.chmod(0o644)
    if cached.resolve() != target.resolve():
        target.write_bytes(cached.read_bytes())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the local dense index and optional YOLO checkpoint before Docker setup."
    )
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--index-output", default=f"data/indexes/{HF_INDEX_NAME}")
    parser.add_argument("--yolo-output", default="data/models/circuit-component-detector")
    args = parser.parse_args()

    if not args.skip_index:
        output = download_published_index(args.index_output, HF_INDEX_REPO, HF_INDEX_NAME)
        print(f"DENSE_INDEX_READY {output}")
    if not args.skip_yolo:
        output = download_yolo(args.yolo_output)
        print(f"YOLO_MODEL_READY {output} {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
