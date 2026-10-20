# VoltAgent

VoltAgent is a modular FastAPI assistant for electrical engineering. It uses remote cloud APIs for conversational LLM calls, while deterministic electrical calculations, document retrieval, and optional YOLO inference run locally.

## Implemented phases

- **Phase 1:** FastAPI app, OpenAI-compatible and Gemini provider abstraction, SQLite sessions, vanilla JavaScript chat UI.
- **Phase 2:** Markdown/TXT/PDF ingestion, stable chunk IDs, persistent SQLite vector store, deterministic fallback embeddings, retrieval citations.
- **Phase 3:** Ohm's law, series/parallel resistance, dividers, power, RC time constants, and three-phase power helpers.
- **Phase 4:** Configurable YOLO detector and validated image upload route. YOLO dependencies and checkpoints are optional.
- **Phase 5-6 foundation:** Typed agent state, registered tools, replaceable decision provider interface, heuristic fallback router.

The application remains usable without Jev, YOLO, SPICE, or optional RAG packages. A configured conversational provider is required for live answers.

## Install and run

```bash
uv sync --extra test
cp .env.example .env
# edit .env with LLM_PROVIDER, LLM_MODEL, and the matching API key
uv run uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`. API endpoints include:

- `POST /api/chat` with `{"message": "Explain Ohm's law", "session_id": null}`
- `GET /api/sessions/{session_id}`
- `GET /health`

## RAG ingestion

```bash
uv sync --extra rag
uv run python scripts/ingest.py data/documents
```

Supported formats are `.md`, `.txt`, and `.pdf`. Configure `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RAG_TOP_K`, and `VECTOR_DB_PATH` in `.env`.

### Kuphaldt textbook parsing foundation

The Stage 1-2 Kuphaldt pipeline parses only numeric chapter pages from the six supported textbook volumes under `data/documents/kuphaldt/`. It excludes navigation pages, root assets, appendices, and contributor-only sections. Images are preserved as resolved references and are not interpreted as circuit topology.

Inspect the corpus and print representative sections and chunks:

```bash
uv run python -m scripts.inspect_kuphaldt
uv run python -m scripts.inspect_kuphaldt --volumes DC AC --chunk-size 512 --chunk-overlap 64
uv run python -m scripts.inspect_kuphaldt --volumes DC --json
uv run python -m scripts.inspect_kuphaldt --volumes DC --show-skipped --sample-sections 0 --sample-chunks 0
```

The parser is available through `app.rag.kuphaldt.KuphaldtParser`, and `chunk_sections` creates stable, section-aware chunks. This is the independently testable foundation for later dense, BM25, hybrid, contextual, and reranking experiments. Dense retrieval, LightRAG, and cloud contextualization are intentionally not part of this stage.

### Initial retrieval quality baseline

An initial 36-question, manually reviewed evaluation set is stored at `data/evaluation/kuphaldt_questions.json`. It currently evaluates a deterministic lexical TF-IDF-style baseline over the real Kuphaldt chunks:

```bash
uv run python -m scripts.evaluate_retrieval
uv run python -m scripts.evaluate_retrieval --json
```

This reports Recall@5, Recall@10, MRR@10, and local parsing/evaluation timing. It is a baseline for comparison, not evidence that lexical retrieval is superior to dense, hybrid, or reranked retrieval.

### Dense embedding setup

The dense baseline uses `BAAI/bge-base-en-v1.5`, Sentence Transformers, normalized embeddings, and a persistent FAISS inner-product index. The embedding model uses `--device auto` by default and selects CUDA when a CUDA-enabled PyTorch installation is available. FAISS remains CPU-backed for this small 1,893-vector corpus because index search is negligible compared with model encoding; the device is configurable for larger deployments.

```bash
uv sync --extra dense
uv run python -m scripts.build_dense_index --device auto
```

The command writes `data/indexes/kuphaldt-bge-base-en-v1.5/` with `chunks.faiss`, `chunks.json`, and `metadata.json`. The generated index is ignored by Git because it is a derived artifact. Query-side dense search is available through `app.rag.dense.search_dense`; the index records the embedding model name so queries use the same model as documents.

## YOLO detection

Install optional vision dependencies and provide a compatible trained component checkpoint:

```bash
uv sync --extra vision
export YOLO_MODEL_PATH=/path/to/electrical-components.pt
```

The detector does not assume generic YOLO weights recognize electrical components. It returns a clear unavailable/error state when no compatible checkpoint is configured. Bounding boxes do not establish schematic connectivity.

## Tests and lint

```bash
uv run pytest -q
uv run ruff check app scripts tests
```

The standard test suite uses mocks and synthetic data. It does not require cloud API credentials, a YOLO checkpoint, or locally hosted LLM inference.

## Architecture

- `app/main.py`: lightweight HTTP boundary and composition root.
- `app/providers/`: cloud LLM adapters.
- `app/session_store.py`: SQLite conversation persistence.
- `app/rag/`: chunking, embedding fallback, ingestion, and retrieval.
- `app/electrical/`: pure deterministic calculations and schemas.
- `app/vision/`: upload validation and optional YOLO inference.
- `app/agent/`: typed state, tool registry, and decision interfaces.
- `frontend/`: framework-free HTML, CSS, and JavaScript.

## Limitations

- The current baseline uses a deterministic hash embedding fallback so it can run without downloading an embedding model. Sentence Transformers and Chroma are optional extension dependencies.
- The chat route includes a small deterministic calculation path for common Ohm's law questions. Full native multi-step LLM tool calling and circuit topology extraction are intentionally future extensions.
- The frontend displays plain text safely. It does not yet render full Markdown or detection overlays.
