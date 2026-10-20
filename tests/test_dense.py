import pytest

from app.rag.dense import DEFAULT_MODEL, embedding_text, resolve_device
from app.rag.kuphaldt import DocumentChunk


def test_embedding_text_includes_structural_metadata():
    chunk = DocumentChunk(
        chunk_id="chunk",
        document_id="document",
        volume="Direct Current",
        chapter_number=2,
        chapter_title="Ohm's Law",
        section_title="Voltage and current",
        content="Voltage is potential energy per unit charge.",
        processed_content="Voltage is potential energy per unit charge.",
        source_path="DC_2.html",
        source_url="file:///DC_2.html",
        token_count=8,
        chunk_index=0,
        parent_section_index=0,
    )

    text = embedding_text(chunk)

    assert DEFAULT_MODEL == "BAAI/bge-base-en-v1.5"
    assert "Volume: Direct Current" in text
    assert "Chapter 2: Ohm's Law" in text
    assert "Section: Voltage and current" in text
    assert chunk.content in text


def test_dense_index_rejects_empty_chunks_before_optional_import():
    from app.rag.dense import build_dense_index

    with pytest.raises(ValueError, match="no chunks"):
        build_dense_index([], "/tmp/unused-dense-index")


def test_device_can_be_explicitly_pinned():
    assert resolve_device("cpu") == "cpu"
