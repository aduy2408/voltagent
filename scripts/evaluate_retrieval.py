from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.rag.evaluation import EvaluationQuery, evaluate_retrieval
from app.rag.kuphaldt import SUPPORTED_VOLUMES, KuphaldtParser, chunk_sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical retrieval on Kuphaldt chunks.")
    parser.add_argument("--root", default="data/documents/kuphaldt")
    parser.add_argument("--questions", default="data/evaluation/kuphaldt_questions.json")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    sections, report = KuphaldtParser(args.root).parse(SUPPORTED_VOLUMES)
    chunks = chunk_sections(sections)
    questions = [EvaluationQuery.model_validate(item) for item in json.loads(Path(args.questions).read_text())]
    query_started = time.perf_counter()
    evaluations, metrics = evaluate_retrieval(chunks, questions, args.candidate_k)
    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "chunks": float(len(chunks)),
            "sections": float(len(sections)),
            "index_build_seconds": query_started - started,
            "evaluation_seconds": time.perf_counter() - query_started,
            "total_seconds": elapsed,
            "parsing_errors": float(len(report.errors)),
        }
    )
    output = {
        "method": "lexical_tfidf_baseline",
        "metrics": metrics,
        "queries": [
            {
                "question": item.question,
                "first_relevant_rank": next(
                    (index + 1 for index, chunk_id in enumerate(item.retrieved_chunk_ids) if chunk_id in item.relevant_chunk_ids),
                    None,
                ),
                "recall_at_5": item.recall_at_5,
                "recall_at_10": item.recall_at_10,
            }
            for item in evaluations
        ],
    }
    if args.json:
        print(json.dumps(output, indent=2))
        return
    print("Kuphaldt retrieval evaluation")
    print("method: lexical_tfidf_baseline")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nPer-query results:")
    for item in output["queries"]:
        print(
            f"- rank={item['first_relevant_rank']} "
            f"recall@5={item['recall_at_5']:.0f} recall@10={item['recall_at_10']:.0f}: "
            f"{item['question']}"
        )


if __name__ == "__main__":
    main()
