# Tensor Serve Source

This directory contains the Python package for Tensor Serve. The root
`README.md` covers the app's user-facing functionality, REST API, collections,
configuration, workflow, and operational notes. This file is for code-level
orientation inside `src/`.

## Entry Points

| File | Purpose |
|---|---|
| `../main.py` | FastAPI application, lifespan setup, app state, and API routes |
| `manage_zim.py` | CLI for downloading, listing, uninstalling, and cleaning ZIM-related working files |

Run these entry points from the repository root:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
python -m src.manage_zim status
```

## Module Map

| File | Purpose |
|---|---|
| `ai_client.py` | HTTP client for any OpenAI-compatible upstream endpoint (local or cloud) with optional API key authentication |
| `bm25_index.py` | BM25 keyword index wrapper, including save/load/search helpers |
| `cache.py` | In-memory LRU cache for query embeddings and search results |
| `chunker.py` | 500-word overlapping text chunking |
| `config.py` | Persistent JSON configuration helpers, including optional `zim_source_folder` |
| `embedder.py` | SentenceTransformer embedding wrapper, preferring locally cached model files |
| `hybrid_search.py` | Reciprocal Rank Fusion for merging FAISS and BM25 results |
| `ingest.py` | Single-ZIM ingestion pipeline |
| `multi_ingest.py` | Multi-ZIM ingestion pipeline for combined databases from files or directories |
| `collections.py` | Collection folder discovery, metadata, and active-collection persistence |
| `query_analyzer.py` | RAG/no-RAG decisions and search-mode selection |
| `reranker.py` | Optional cross-encoder reranker for second-stage result ordering |
| `utils.py` | ZIM article iteration and HTML/text cleanup helpers |
| `vectordb.py` | FAISS index wrapper, including save/load/search helpers |
| `web_search.py` | Web search integration with pluggable providers (DuckDuckGo, Brave, Google Custom Search), caching, and time-sensitive query detection |
| `zim_downloader.py` | Kiwix OPDS catalog interface, download engine, custom ZIM source-folder support, install status, and manifest handling |

## Internal Architecture

- **Embedding**: `embedder.py` wraps `sentence-transformers` with `all-MiniLM-L6-v2`.
- **Vector search**: `vectordb.py` stores FAISS indexes and chunk text stores.
- **Keyword search**: `bm25_index.py` builds a `rank-bm25` index from the same chunks.
- **Hybrid retrieval**: `hybrid_search.py` merges FAISS and BM25 rankings (plus optional web results) with Reciprocal Rank Fusion.
- **Query routing**: `query_analyzer.py` decides whether retrieval is needed, detects time-sensitive queries, and chooses `hybrid`, `faiss`, or `bm25`.
- **Web search**: `web_search.py` provides pluggable search providers (DuckDuckGo, Brave, Google Custom Search) for time-sensitive queries; results are cached and merged into local results via RRF.
- **Optional reranking**: `reranker.py` can run a cross-encoder pass over retrieved chunks.
- **LLM forwarding**: `main.py` proxies OpenAI-compatible `/v1/*` traffic to any configured upstream endpoint (local runtime, cloud API, or gateway). API keys enable authentication with services that require them.
- **State**: `config.py`, `collections.py`, and `zim_downloader.py` manage JSON and manifest-backed state. Collection metadata is stored in `collections.json`; collection folders contain lightweight links to ZIM files in the active source folder.
