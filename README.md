Tensor Serve

Tensor Serve is an **offline-first AI backend**. It downloads documentation and knowledge bases as ZIM files, builds a local semantic vector database from them, and uses that database to give an AI model relevant context when answering questions — while keeping your data private.

---

## 1. ZIM File Manager CLI (`manage_zim.py`)

The command-line tool for managing your local knowledge base files.

```/dev/null/shell.sh#L1-7
python manage_zim.py list                       # List all available preset files and their sizes
python manage_zim.py status                     # Show all installed ZIM files
python manage_zim.py status <preset>            # Show install status for one preset (research, learn, literature, coding)
python manage_zim.py install <file_id>          # Download a specific file by its Kiwix ID
python manage_zim.py uninstall <file_id>        # Remove a file and its manifest entry
python manage_zim.py install-preset <preset>    # Interactive checkbox picker for a preset's files
python manage_zim.py install-devdocs            # Interactive checkbox picker for all 231 DevDocs entries
```

- Downloads come from the **live Kiwix OPDS catalog** — always up to date
- Prefers **text-only (`nopic`) flavours** automatically to keep sizes small
- Downloads show a **live progress bar** (`████░░ 45.2%  210 MB / 465 MB`)
- Tracks all installed files in `zim_manifest.json`

---

## 2. REST API (`main.py`)

Start the server with: `uvicorn main:app --host 0.0.0.0 --port 8000`

Interactive docs available at `http://localhost:8000/docs`

### Health & Configuration
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Server status, whether DB is loaded, active preset |
| `GET` | `/config` | View current AI endpoint, model, and settings |
| `GET` | `/config/models` | List models available at the configured endpoint (or `?endpoint=<url>` to probe any URL) |
| `POST` | `/config/set-ai-endpoint` | Set `ai_endpoint` and optionally `ai_model` (auto-detected if omitted) |
| `PATCH` | `/config` | Update any combination of the four settings (all fields optional) |

### Presets
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/presets` | List all presets (built-in + custom) and which is active |
| `GET` | `/presets/{preset_id}` | Details and file installation status for one preset |
| `POST` | `/presets/{preset_id}/ingest` | Process a preset's ZIM files into a vector database |
| `POST` | `/presets/custom/create` | Create a custom preset from your own ZIM file paths |
| `DELETE` | `/presets/custom/{preset_id}` | Delete a custom preset |

### ZIM File Management
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/zim/available` | List all preset files with installed/not-installed status |
| `GET` | `/zim/installed` | List every installed ZIM file with path and size (scans disk, auto-registers untracked files) |
| `GET` | `/zim/status/{preset_id}` | Per-file install status for a preset |
| `POST` | `/zim/install` | Queue one or more ZIM files for background download by Kiwix ID |
| `POST` | `/zim/install-preset` | Queue all (or selected) files from a preset for background download |
| `DELETE` | `/zim/uninstall/{file_id}` | Remove a ZIM file from disk and the manifest |
| `GET` | `/zim/devdocs` | Live catalog of all 231 DevDocs entries from Kiwix |
| `POST` | `/zim/devdocs/install` | Queue DevDocs downloads in the background (specific IDs or all) |
| `GET` | `/zim/progress` | Poll aggregate progress for all active and recent downloads |
| `GET` | `/zim/progress/{file_id}` | Poll download progress for a single file by Kiwix ID |

### Vector Database
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/ingest` | Ingest a single ZIM file into a named vector DB |
| `POST` | `/ingest-multiple` | Ingest multiple ZIM files into one combined vector DB |
| `GET` | `/load?name=<db>` | Load a previously ingested database into memory |
| `POST` | `/search` | Hybrid search (FAISS + BM25 via RRF) — returns top-k chunks; response includes `search_mode` |

### Chat & Conversations
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/chat` | Send a message; get a context-grounded AI response |
| `GET` | `/conversation/{id}` | Retrieve the full history of a conversation |

#### Download progress fields

When a file is actively downloading, `GET /zim/progress/{file_id}` returns:

| Field | Description |
|---|---|
| `status` | `downloading` \| `completed` \| `partial` \| `error` \| `already_installed` |
| `percent` | `0.0` – `100.0` |
| `downloaded` | Human-readable bytes received (e.g. `"210.3 MB"`) |
| `total` | Human-readable total size |
| `downloaded_bytes` | Raw bytes received |
| `total_bytes` | Raw total bytes (0 if server did not send `Content-Length`) |

The `devdocs_all` bundle entry additionally includes `completed_files` and `total_files` so a GUI can show per-entry progress.

### Cleanup
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/clean` | Delete all vector DB index files, text stores, conversation history, and `__pycache__/` — preserves presets, config, and ZIM files |

---

## 3. Built-in Presets

Four curated knowledge bases, each automatically selecting text-only ZIM variants:

| Preset | Contents | Approx. size |
|---|---|---|
| `research` | Wikipedia, Wikisource, Wikinews | ~65 GB |
| `learn` | Wikiversity, Wikibooks | ~5 GB |
| `literature` | Project Gutenberg, Wikibooks | ~63 GB |
| `coding` | Stack Overflow, All DevDocs (231 entries) | ~80 GB + ~588 MB |

---

## 4. How the AI pipeline works

1. **Download** — ZIM files fetched from Kiwix and stored in `zim_files/`
2. **Ingest** — Articles extracted, HTML stripped, split into 500-word overlapping chunks, embedded with `sentence-transformers`, indexed in FAISS **and** BM25
3. **Auto-load** — On server startup, the last active preset's FAISS and BM25 indexes are loaded automatically
4. **Chat** — User message is embedded → hybrid search (FAISS + BM25 via RRF) retrieves top-k chunks → chunks + message sent to the local LLM → response returned with source context included

### Hybrid search (FAISS + BM25 via Reciprocal Rank Fusion)

Every search and chat request runs **two retrievals in parallel** and merges them:

| | FAISS (semantic) | BM25 (keyword) |
|---|---|---|
| Finds | Conceptually related chunks | Exact term / token matches |
| Good for | *"How does backpressure work?"* | *"asyncio.gather"*, error codes, API names |
| Index file | `{name}.index` + `{name}.pkl` | `{name}.bm25` |

Results are merged with **Reciprocal Rank Fusion** (`score = Σ 1 / (60 + rank)`). Chunks that rank well in both lists float to the top. The pipeline degrades gracefully — if only one index is available it is used alone.

---

## 5. Key files

| File | Purpose |
|---|---|
| `main.py` | FastAPI application and all API routes |
| `manage_zim.py` | CLI for downloading and managing ZIM files |
| `presets.py` | Preset definitions and configuration persistence |
| `zim_downloader.py` | Kiwix OPDS catalog interface and download engine |
| `ingest.py` / `multi_ingest.py` | ZIM → vector database pipeline (FAISS + BM25) |
| `embedder.py` | Sentence-transformer embeddings |
| `vectordb.py` | FAISS index wrapper (save/load/search/search_indices) |
| `bm25_index.py` | BM25 keyword index wrapper (save/load/search_indices) |
| `hybrid_search.py` | Reciprocal Rank Fusion — merges FAISS and BM25 results |
| `chunker.py` | 500-word overlapping text chunker |
| `utils.py` | ZIM article iterator and HTML cleaner |
| `ai_client.py` | HTTP client for the local LLM endpoint |
| `conversations.py` | SQLite-backed conversation history |
| `config.py` | Persistent JSON configuration |
| `presets.json` | Saved preset state and active preset |
| `zim_manifest.json` | Record of all installed ZIM files |

---

### Settings

All four settings are readable via `GET /config` and writable via `PATCH /config`.
`ai_endpoint` and `ai_model` can also be set together with `POST /config/set-ai-endpoint`.

| Setting | Default | What it controls |
|---|---|---|
| `ai_endpoint` | `null` | URL of local LLM server (required for `/chat`) |
| `ai_model` | `null` | Model name passed to the LLM (required for `/chat`) |
| `context_size` | `3` | Number of vector DB chunks retrieved as context per chat message |
| `max_conversation_history` | `20` | Maximum messages returned by `GET /conversation/{id}` |

#### Model auto-detection

`ai_model` does not need to be set manually. The server can discover available models by querying the endpoint directly — it tries the OpenAI-compatible `GET /v1/models` route first (vLLM, LM Studio, LocalAI, etc.), then falls back to Ollama's `GET /api/tags`.

**Example — list models before configuring:**
```bash
curl "http://localhost:8000/config/models?endpoint=http://localhost:11434"
```

**Example — set endpoint and auto-detect model:**
```bash
# Omit ai_model entirely — the first available model is selected automatically.
# If multiple models are found, the response lists all of them so you can switch with PATCH /config.
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{"ai_endpoint": "http://localhost:11434"}'
```

**Example — change only context size:**
```bash
curl -X PATCH http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"context_size": 5}'
```

**Example — change all settings at once:**
```bash
curl -X PATCH http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "llama3",
    "context_size": 5,
    "max_conversation_history": 50
  }'
```

---

## Architecture

- **Embedder**: Uses `sentence-transformers` (all-MiniLM-L6-v2) for semantic embeddings
- **Vector DB**: FAISS index for efficient similarity search
- **BM25 Index**: `rank-bm25` keyword index built from the same text chunks as FAISS
- **Hybrid Search**: Reciprocal Rank Fusion merges FAISS and BM25 ranked lists — no tuning required
- **AI Client**: HTTP client for communicating with local LLM endpoint
- **Conversations**: SQLite database for tracking message history
- **Config**: JSON file for persistent settings

## Error Handling

- **400**: Bad request (DB not loaded, AI not configured, invalid input)
- **404**: Resource not found (database files missing)
- **500**: Server error
- **502**: AI endpoint unreachable or error

## Performance Notes

- Large ZIM files (>1GB) may take 10-30 minutes to ingest
- Both FAISS (`.index` + `.pkl`) and BM25 (`.bm25`) indexes are saved to disk and reloaded on startup — no re-ingestion needed
- FAISS similarity search is O(1); BM25 scoring is O(n) but extremely fast in practice
- Hybrid RRF adds negligible overhead — both searches run in milliseconds
- Chat responses depend on AI endpoint response time
- Existing databases ingested before hybrid search was added will use semantic-only search until re-ingested (no `.bm25` file present → graceful fallback)

---

## Workflow

### Complete Example

```bash
# 1. Start the server
uvicorn main:app --reload

# 2. Configure AI endpoint (assuming Ollama running on port 11434)
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "mistral"
  }'

# 3. Check health
curl http://localhost:8000/health

# 4. Ingest ZIM file
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "zim_path": "/data/wikipedia.zim",
    "output_name": "wiki"
  }'

# 5. Load the database
curl http://localhost:8000/load?name=wiki

# 6. Start chatting
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who invented the telephone?"}'

# 7. Get conversation history
curl http://localhost:8000/conversation/your-conversation-id
```