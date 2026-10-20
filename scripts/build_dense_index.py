from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag.dense import DEFAULT_MODEL, build_dense_index
from app.rag.kuphaldt import SUPPORTED_VOLUMES, KuphaldtParser, chunk_sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent FAISS dense index for Kuphaldt chunks.")
    parser.add_argument("--root", default="data/documents/kuphaldt")
    parser.add_argument("--output", default="data/indexes/kuphaldt-bge-base-en-v1.5")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="Embedding device: auto, cuda, or cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--volumes", nargs="+", choices=SUPPORTED_VOLUMES, default=list(SUPPORTED_VOLUMES))
    args = parser.parse_args()

    sections, report = KuphaldtParser(args.root).parse(args.volumes)
    chunks = chunk_sections(sections, args.chunk_size, args.chunk_overlap)
    metadata = build_dense_index(chunks, args.output, args.model, args.batch_size, args.device)
    metadata.update(
        {
            "root": str(Path(args.root).resolve()),
            "volumes": args.volumes,
            "parsed_sections": len(sections),
            "parsing_errors": len(report.errors),
        }
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
