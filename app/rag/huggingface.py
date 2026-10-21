from __future__ import annotations

import shutil
from pathlib import Path

HF_INDEX_REPO = "duyle2408/kuphaldt-retrieval-indexes"
HF_INDEX_NAME = "kuphaldt-arctic-embed-l-v2"
INDEX_FILES = ("chunks.faiss", "chunks.json", "metadata.json")


def download_published_index(
    output_dir: str | Path = f"data/indexes/{HF_INDEX_NAME}",
    repo_id: str = HF_INDEX_REPO,
    index_name: str = HF_INDEX_NAME,
) -> Path:
    """Download a published FAISS index and its chunk metadata from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading published indexes requires huggingface-hub. "
            "Install with: uv sync --extra dense"
        ) from exc

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in INDEX_FILES:
        cached_file = hf_hub_download(repo_id=repo_id, filename=f"{index_name}/{filename}")
        target = destination / filename
        if target.exists():
            target.chmod(0o644)
        shutil.copy2(cached_file, target)
        downloaded.append(target)

    missing = [path.name for path in downloaded if not path.is_file()]
    if missing:
        raise RuntimeError(f"Hugging Face index download is incomplete: {', '.join(missing)}")
    return destination
