# Tensor Serve Source

This directory contains the Python package for Tensor Serve. The root
`README.md` covers the app's user-facing functionality, REST API, presets,
configuration, workflow, and operational notes. This file is for code-level
orientation inside `src/`.

## Entry Points

| File | Purpose |
|---|---|
| `main.py` | FastAPI application, lifespan setup, app state, and API routes |
| `manage_zim.py` | CLI for downloading, listing, uninstalling, and cleaning ZIM-related working files |

Run these entry points from the repository root:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
python -m src.manage_zim status
```

## Module Map

| File | Purpose |
|---|---|
| `ai_client.py` | HTTP client for the configured local or OpenAI-compatible upstream LLM endpoint |
| `bm25_index.py` | BM25 keyword index wrapper, including save/load/search helpers |
| `cache.py` | In-memory LRU cache for query embeddings and search results |
| `chunker.py` | 500-word overlapping text chunking |
| `config.py` | Persistent JSON configuration helpers |
| `conversations.py` | SQLite-backed conversation history |
| `embedder.py` | SentenceTransformer embedding wrapper, preferring locally cached model files |
| `hybrid_search.py` | Reciprocal Rank Fusion for merging FAISS and BM25 results |
| `ingest.py` | Single-ZIM ingestion pipeline |
| `multi_ingest.py` | Multi-ZIM ingestion pipeline for combined databases and presets |
| `presets.py` | Built-in/custom preset definitions and active-preset persistence |
| `query_analyzer.py` | RAG/no-RAG decisions and search-mode selection |
| `reranker.py` | Optional cross-encoder reranker for second-stage result ordering |
| `utils.py` | ZIM article iteration and HTML/text cleanup helpers |
| `vectordb.py` | FAISS index wrapper, including save/load/search helpers |
| `zim_downloader.py` | Kiwix OPDS catalog interface, download engine, install status, and manifest handling |

## Internal Architecture

- **Embedding**: `embedder.py` wraps `sentence-transformers` with `all-MiniLM-L6-v2`.
- **Vector search**: `vectordb.py` stores FAISS indexes and chunk text stores.
- **Keyword search**: `bm25_index.py` builds a `rank-bm25` index from the same chunks.
- **Hybrid retrieval**: `hybrid_search.py` merges FAISS and BM25 rankings with Reciprocal Rank Fusion.
- **Query routing**: `query_analyzer.py` decides whether retrieval is needed and chooses `hybrid`, `faiss`, or `bm25`.
- **Optional reranking**: `reranker.py` can run a cross-encoder pass over retrieved chunks.
- **LLM forwarding**: `ai_client.py` handles native chat calls, while `main.py` proxies OpenAI-compatible `/v1/*` traffic.
- **State**: `config.py`, `presets.py`, `conversations.py`, and `zim_downloader.py` manage JSON, SQLite, and manifest-backed state.
