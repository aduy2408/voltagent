from app.rag.kuphaldt import DocumentChunk
from app.rag.rerank import rerank_chunks


def _chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        volume="Direct Current",
        chapter_number=1,
        chapter_title="Chapter",
        section_title="Section",
        content=content,
        processed_content=content,
        source_path=f"{chunk_id}.html",
        source_url=f"file:///{chunk_id}.html",
        token_count=3,
        chunk_index=0,
        parent_section_index=0,
    )


def test_rerank_chunks_sorts_by_cross_encoder_score(monkeypatch):
    class FakeReranker:
        def predict(self, pairs, batch_size, show_progress_bar):
            assert len(pairs) == 2
            assert batch_size == 4
            assert show_progress_bar is False
            return [0.2, 0.9]

    monkeypatch.setattr("app.rag.rerank._load_reranker", lambda model, device: FakeReranker())
    candidates = [(_chunk("first", "first content"), 0.8), (_chunk("second", "second content"), 0.7)]

    results = rerank_chunks("query", candidates, top_k=1, batch_size=4)

    assert [chunk.chunk_id for chunk, _score in results] == ["second"]
    assert results[0][1] == 0.9


def test_rerank_chunks_rejects_invalid_parameters():
    try:
        rerank_chunks("query", [], top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected top_k validation error")
