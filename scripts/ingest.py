import argparse

from app.config import get_settings
from app.rag.core import VectorStore, ingest_path

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--db", default=None)
args = parser.parse_args()
settings = get_settings()
store = VectorStore(args.db or settings.vector_db_path)
print(f"Ingested {ingest_path(args.path, store, settings.chunk_size, settings.chunk_overlap)} chunks")
