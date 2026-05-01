# Tensor Serve Test Guide

This directory contains the automated unit tests and a small manual smoke test
for the OpenAI-compatible API.

## Automated tests

Run the unit suite from the repository root:

```bash
.venv/bin/python -m pytest -q
```

These tests are designed to avoid live network calls, real LLM servers, and
large ZIM ingestion. They cover the core pieces that can be checked quickly:

| Test file | What it covers |
|---|---|
| `test_chunker.py` | Word chunking and overlap behavior |
| `test_utils.py` | HTML stripping and whitespace cleanup |
| `test_bm25.py` | BM25 index build, search, and text lookup |
| `test_reranker_and_hybrid.py` | RRF merging, hybrid search, and reranker fallback behavior |
| `test_query_analyzer.py` | RAG/no-RAG decisions and automatic search-mode selection |
| `test_cache.py` | LRU caching, TTL expiry, and cached search-result keys |
| `test_config.py` | Persistent JSON config load/save helpers |
| `test_conversations.py` | SQLite conversation storage and chronological history retrieval |
| `test_vectordb.py` | FAISS index save/load/search behavior |
| `test_ai_client.py` | Prompt context deduplication and model discovery parsing |
| `test_proxy_api.py` | `/v1/*` proxy forwarding, chat context injection, and upstream response passthrough |
| `test_presets_and_zim.py` | Custom preset lifecycle and local ZIM manifest scanning |

## Manual API smoke test

`test_openai_api.py` is intentionally a manual smoke script, not a pytest unit
test. It expects a running Tensor Serve instance, a configured AI endpoint, and
a loaded vector database.

Start the server:

```bash
uvicorn src.main:app --reload
```

Configure an AI endpoint:

```bash
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{"ai_endpoint": "http://localhost:1234", "ai_model": "local-model"}'
```

Load an ingested database:

```bash
curl "http://localhost:8000/load?name=coding_db"
```

Run the smoke script:

```bash
.venv/bin/python test/test_openai_api.py
```

Use the smoke script when you want to verify the full running service path:
HTTP server, configured upstream LLM endpoint, loaded retrieval index, context
injection, and proxied OpenAI-compatible chat responses.
