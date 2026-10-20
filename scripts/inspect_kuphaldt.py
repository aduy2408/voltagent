from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.rag.kuphaldt import SUPPORTED_VOLUMES, inspect_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and chunk the Kuphaldt HTML corpus.")
    parser.add_argument(
        "--root",
        default="data/documents/kuphaldt",
        help="Kuphaldt corpus root directory.",
    )
    parser.add_argument(
        "--volumes",
        nargs="+",
        choices=SUPPORTED_VOLUMES,
        default=list(SUPPORTED_VOLUMES),
        help="Volumes to inspect. Defaults to all supported volumes.",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Maximum token-like words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap token-like words between chunks.")
    parser.add_argument("--sample-sections", type=int, default=3, help="Number of sections to print.")
    parser.add_argument("--sample-chunks", type=int, default=3, help="Number of chunks to print.")
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print each skipped HTML file and its reason.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sections, chunks, report = inspect_corpus(
        args.root,
        args.volumes,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    skipped_by_reason = Counter(item.reason for item in report.skipped_files)
    summary = {
        "root": str(Path(args.root).resolve()),
        "volumes": args.volumes,
        "eligible_chapter_files": report.eligible_chapter_files,
        "parsed_chapters": report.parsed_chapters,
        "parsed_sections": report.parsed_sections,
        "total_chunks": len(chunks),
        "total_image_references": report.image_references,
        "skipped_files": len(report.skipped_files),
        "skipped_by_reason": dict(skipped_by_reason),
        "skipped_file_details": [item.__dict__ for item in report.skipped_files],
        "parsing_errors": [error.__dict__ for error in report.errors],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Kuphaldt corpus inspection")
    for key, value in summary.items():
        if key not in {"skipped_by_reason", "skipped_file_details", "parsing_errors"}:
            print(f"{key}: {value}")
    print(f"skipped_by_reason: {dict(skipped_by_reason)}")
    if args.show_skipped and report.skipped_files:
        print("skipped_file_details:")
        for item in report.skipped_files:
            print(f"  - {item.path}: {item.reason}")
    if report.errors:
        print("parsing_errors:")
        for error in report.errors:
            print(f"  - {error.path}: {error.error}")

    if args.sample_sections:
        print("\nRepresentative sections:")
        for section in sections[: args.sample_sections]:
            print(f"\n[{section.volume} | Chapter {section.chapter_number} | {section.section_title}]")
            print(section.content[:800])
            if section.image_references:
                print(f"images: {len(section.image_references)}")

    if args.sample_chunks:
        print("\nRepresentative chunks:")
        for chunk in chunks[: args.sample_chunks]:
            print(f"\n[{chunk.chunk_id[:12]} | {chunk.volume} | {chunk.section_title}]")
            print(chunk.content[:800])


if __name__ == "__main__":
    main()
