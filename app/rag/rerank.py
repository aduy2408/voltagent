from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.rag.dense import embedding_text, resolve_device
from app.rag.kuphaldt import DocumentChunk

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _require_reranker_dependencies() -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranking requires the 'dense' optional dependencies. "
            "Install with: uv sync --extra dense"
        ) from exc
    return CrossEncoder


@lru_cache(maxsize=4)
def _load_reranker(model_name: str, device: str) -> Any:
    cross_encoder = _require_reranker_dependencies()
    return cross_encoder(model_name, device=device)


def rerank_chunks(
    query: str,
    candidates: list[tuple[DocumentChunk, float]],
    top_k: int,
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str = "auto",
    batch_size: int = 16,
) -> list[tuple[DocumentChunk, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not candidates:
        return []

    reranker = _load_reranker(model_name, resolve_device(device))
    pairs = [(query, embedding_text(chunk)) for chunk, _dense_score in candidates]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(
        ((chunk, float(score)) for (chunk, _dense_score), score in zip(candidates, scores)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[: min(top_k, len(ranked))]
