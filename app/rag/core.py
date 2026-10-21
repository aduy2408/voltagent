from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalResult:
    document_id: str
    source: str
    content: str
    score: float
    page: int | None = None

    def __getitem__(self, index: int):
        return (self.score, self.source, self.content)[index]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def hash_embedding(text: str, dimensions: int = 256) -> list[float]:
    values = [0.0] * dimensions
    for token in re.findall(r"\w+", text.lower()):
        index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions
        values[index] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class VectorStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT PRIMARY KEY, source TEXT NOT NULL, page INTEGER,
                content TEXT NOT NULL, embedding TEXT NOT NULL)""")

    def add(self, source: str, content: str, embedding: list[float], page: int | None = None) -> str:
        document_id = hashlib.sha256(f"{source}\n{page}\n{content}".encode()).hexdigest()
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)", (document_id, source, page, content, json.dumps(embedding)))
        return document_id

    def search(self, query: list[float], top_k: int = 5) -> list[RetrievalResult]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT document_id, source, page, content, embedding FROM chunks").fetchall()
        scored = []
        for document_id, source, page, content, embedding in rows:
            score = sum(a * b for a, b in zip(query, json.loads(embedding)))
            scored.append(RetrievalResult(document_id, source, content, score, page))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def _read_file(path: Path) -> list[tuple[str, int | None]]:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires the optional pypdf dependency") from exc
        return [(page.extract_text() or "", index + 1) for index, page in enumerate(PdfReader(str(path)).pages)]
    return [(path.read_text(encoding="utf-8"), None)]


def ingest_path(path: str, store: VectorStore, chunk_size: int = 800, overlap: int = 120) -> int:
    root = Path(path)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    count = 0
    for file in files:
        for text, page in _read_file(file):
            for chunk in chunk_text(text, chunk_size, overlap):
                store.add(str(file), chunk, hash_embedding(chunk), page)
                count += 1
    return count
