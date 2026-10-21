import sys
from pathlib import Path
from types import SimpleNamespace

from app.rag.huggingface import INDEX_FILES, download_published_index


def test_download_published_index_copies_all_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()

    def fake_download(repo_id, filename):
        source = cache / Path(filename).name
        source.write_text(f"{repo_id}:{filename}", encoding="utf-8")
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    output = download_published_index(tmp_path / "index", "repo", "index-name")

    assert output == tmp_path / "index"
    assert {path.name for path in output.iterdir()} == set(INDEX_FILES)
