from app.rag.core import VectorStore, chunk_text, hash_embedding


def test_chunking_and_embedding():
    chunks = chunk_text("one two three four", chunk_size=2, overlap=0)
    assert chunks == ["one two", "three four"]
    assert abs(sum(x*x for x in hash_embedding("hello")) - 1) < 1e-6

def test_vector_store(tmp_path):
    store = VectorStore(str(tmp_path / "v.db"))
    store.add("a.txt", "hello world", hash_embedding("hello world"))
    assert store.search(hash_embedding("hello"), 1)[0][1] == "a.txt"
