from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.rag.kuphaldt import DocumentChunk

DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"


def _require_dense_dependencies() -> tuple[Any, Any, Any]:
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires the 'dense' optional dependencies. "
            "Install with: uv sync --extra dense"
        ) from exc
    return faiss, np, SentenceTransformer


def embedding_text(chunk: DocumentChunk) -> str:
    return (
        f"Volume: {chunk.volume}\n"
        f"Chapter {chunk.chapter_number}: {chunk.chapter_title}\n"
        f"Section: {chunk.section_title}\n\n"
        f"{chunk.content}"
    )


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_embedding_model(model_name: str, device: str) -> Any:
    _faiss, _np, sentence_transformer = _require_dense_dependencies()
    return sentence_transformer(model_name, device=device)


def build_dense_index(
    chunks: list[DocumentChunk],
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, object]:
    if not chunks:
        raise ValueError("Cannot build a dense index with no chunks")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    faiss, _np, _sentence_transformer = _require_dense_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    model = _load_embedding_model(model_name, resolved_device)
    texts = [embedding_text(chunk) for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output / "chunks.faiss"))
    (output / "chunks.json").write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2),
        encoding="utf-8",
    )
    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "metric": "inner_product_on_normalized_embeddings",
        "device": resolved_device,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_dense_index(index_dir: str | Path) -> tuple[Any, list[DocumentChunk], dict[str, object]]:
    faiss, _, _ = _require_dense_dependencies()
    directory = Path(index_dir)
    index = faiss.read_index(str(directory / "chunks.faiss"))
    raw_chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return index, [DocumentChunk.model_validate(item) for item in raw_chunks], metadata


def search_dense(
    index_dir: str | Path,
    query: str,
    top_k: int = 5,
    device: str = "auto",
) -> list[tuple[DocumentChunk, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    _, _np, _sentence_transformer = _require_dense_dependencies()
    index, chunks, metadata = load_dense_index(index_dir)
    resolved_device = resolve_device(device)
    model = _load_embedding_model(str(metadata["model_name"]), resolved_device)
    # Electrical EmbeddingGemma IR is trained as a Sentence Transformers
    # feature-extraction model and is used with the raw query text.
    vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, positions = index.search(vector, min(top_k, len(chunks)))
    return [(chunks[position], float(score)) for position, score in zip(positions[0], scores[0]) if position >= 0]
