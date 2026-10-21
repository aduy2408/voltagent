from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "benchmarks" / "questions"
OUTPUT = EVALUATION_DIR / "kuphaldt_questions_expanded.json"


def main() -> None:
    source_files = [
        EVALUATION_DIR / "kuphaldt_questions.json",
        *sorted(EVALUATION_DIR.glob("agent_*.json")),
    ]
    merged: list[dict[str, object]] = []
    seen: set[str] = set()

    for path in source_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError(f"Expected a JSON list in {path}")
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f"Expected question objects in {path}")
            question = str(record["question"]).strip()
            key = question.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized = dict(record)
            normalized.setdefault("difficulty", "medium")
            normalized.setdefault("intent_clarity", "clear")
            normalized.setdefault("scenario_type", "conceptual")
            normalized.setdefault("expected_retrieval_hops", "1")
            merged.append(normalized)

    OUTPUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(merged)} unique questions to {OUTPUT}")


if __name__ == "__main__":
    main()
