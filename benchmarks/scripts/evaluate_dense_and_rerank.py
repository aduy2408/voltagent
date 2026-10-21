from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from app.rag.dense import build_dense_index, resolve_device
from app.rag.kuphaldt import SUPPORTED_VOLUMES, KuphaldtParser, chunk_sections
from app.rag.rerank import rerank_chunks

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "benchmarks" / "questions" / "kuphaldt_questions_expanded.json"
INDEX_ROOT = ROOT / "data" / "indexes" / "benchmark"
OUTPUT = ROOT / "benchmarks" / "results" / "dense_rerank_benchmark.json"

DEFAULT_MODELS = (
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "disham993/electrical-embeddinggemma-ir_finetune_16bit",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "intfloat/e5-base-v2",
)


def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"Expected a JSON list in {path}")
    required = {"question", "relevant_source_sections"}
    queries = []
    for item in raw:
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError(f"Invalid evaluation item in {path}")
        queries.append(item)
    return queries


def _relevant_ids(chunks: list[Any], query: dict[str, Any]) -> set[str]:
    sections = set(query["relevant_source_sections"])
    return {
        chunk.chunk_id
        for chunk in chunks
        if f"{chunk.volume}|{chunk.chapter_number}|{chunk.section_title}" in sections
    }


def _metrics(results: list[tuple[list[Any], set[str]]]) -> dict[str, float]:
    count = len(results) or 1
    recalls = {5: 0.0, 10: 0.0}
    reciprocal_rank = 0.0
    for ranked, relevant in results:
        ids = [chunk.chunk_id for chunk, _score in ranked]
        for k in recalls:
            recalls[k] += float(bool(set(ids[:k]) & relevant))
        rank = next((index + 1 for index, chunk_id in enumerate(ids[:10]) if chunk_id in relevant), None)
        reciprocal_rank += 1.0 / rank if rank else 0.0
    return {
        "queries": float(len(results)),
        "recall_at_5": recalls[5] / count,
        "recall_at_10": recalls[10] / count,
        "mrr_at_10": reciprocal_rank / count,
    }


def evaluate_model(
    model: str,
    chunks: list[Any],
    queries: list[dict[str, Any]],
    index_dir: Path,
    device: str,
    candidate_k: int,
    reranker_model: str | None,
    reranker_candidate_k: int,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    build_dense_index(chunks, index_dir, model_name=model, batch_size=batch_size, device=device)
    from app.rag.dense import _load_embedding_model, load_dense_index

    index, indexed_chunks, metadata = load_dense_index(index_dir)
    encoder = _load_embedding_model(model, resolve_device(device))
    dense_results = []
    reranked_results = []
    rerank_started = time.perf_counter()
    for query in queries:
        vector = encoder.encode([query["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, positions = index.search(vector, min(candidate_k, len(indexed_chunks)))
        candidates = [
            (indexed_chunks[position], float(score))
            for position, score in zip(positions[0], scores[0])
            if position >= 0
        ]
        relevant = _relevant_ids(chunks, query)
        dense_results.append((candidates, relevant))
        if reranker_model:
            rerank_candidates = candidates[:reranker_candidate_k]
            reranked = rerank_chunks(
                query["question"],
                rerank_candidates,
                top_k=10,
                model_name=reranker_model,
                device=device,
                batch_size=batch_size,
            )
            reranked_results.append((reranked, relevant))
    result: dict[str, Any] = {
        "model": model,
        "metadata": metadata,
        "dense": _metrics(dense_results),
        "seconds": time.perf_counter() - started,
    }
    if reranker_model:
        result["reranked"] = _metrics(reranked_results)
        result["rerank_seconds"] = time.perf_counter() - rerank_started
        result["reranker_model"] = reranker_model
        result["reranker_candidate_k"] = reranker_candidate_k
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense embedding models with optional cross-encoder reranking.")
    parser.add_argument("--root", default="data/documents/kuphaldt")
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--index-root", default=str(INDEX_ROOT))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--reranker-candidate-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    sections, report = KuphaldtParser(args.root).parse(SUPPORTED_VOLUMES)
    chunks = chunk_sections(sections)
    queries = _load_queries(Path(args.questions))
    reranker = None if args.no_rerank else args.reranker_model
    results = []
    for model in args.models:
        print(f"Evaluating {model}", flush=True)
        results.append(
            evaluate_model(
                model,
                chunks,
                queries,
                Path(args.index_root) / model.replace("/", "__"),
                args.device,
                args.candidate_k,
                reranker,
                args.reranker_candidate_k,
                args.batch_size,
            )
        )
    payload = {
        "questions": len(queries),
        "chunks": len(chunks),
        "sections": len(sections),
        "parse_errors": len(report.errors),
        "device": resolve_device(args.device),
        "reranker": reranker,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
