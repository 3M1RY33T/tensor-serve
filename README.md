# Tensor Serve

Tensor Serve is an **offline-first AI backend**. It downloads documentation and knowledge bases as ZIM files, builds a local semantic vector database from them, and uses that database to give an AI model relevant context when answering questions — while keeping your data private.

---

## 1. How the AI pipeline works

1. **Download** — ZIM files fetched from Kiwix and stored in `zim_files/`
2. **Ingest** — Articles extracted, HTML stripped, split into 500-word overlapping chunks, embedded with `sentence-transformers`, indexed in FAISS **and** BM25
3. **Auto-load** — On server startup, the last active preset's FAISS and BM25 indexes are loaded automatically
4. **Analyze** — Simple queries can skip retrieval; domain-specific queries use the query analyzer to choose the best search mode (`hybrid`, `faiss`, or `bm25`)
5. **Chat / Proxy** — User message is embedded (or served from cache) → hybrid search retrieves top-k chunks → optional cross-encoder reranking improves result order → chunks are sent to the local LLM. Native `/chat` builds its own request; `/v1/chat/completions` injects context and proxies to the upstream AI server.

### Hybrid search (FAISS + BM25 via Reciprocal Rank Fusion)

Search and chat requests can run **two retrievals in parallel** and merge them:

| | FAISS (semantic) | BM25 (keyword) |
|---|---|---|
| Finds | Conceptually related chunks | Exact term / token matches |
| Good for | *"How does backpressure work?"* | *"asyncio.gather"*, error codes, API names |
| Index file | `{name}.index` + `{name}.pkl` | `{name}.bm25` |

Results are merged with **Reciprocal Rank Fusion** (`score = Σ 1 / (60 + rank)`). Chunks that rank well in both lists float to the top. The pipeline degrades gracefully — if only one index is available it is used alone.

The query analyzer automatically selects the search strategy:

| Mode | When it is used |
|---|---|
| `hybrid` | Mixed or general queries where semantic and keyword signals both help |
| `faiss` | Conceptual queries such as explanations, architecture, patterns, and design questions |
| `bm25` | Keyword-heavy queries such as API names, code symbols, methods, classes, errors, and short exact searches |

Query embeddings and search results are cached with an in-memory LRU cache to reduce repeated embedding and retrieval work. If enabled, the optional cross-encoder reranker performs a second-stage pass over retrieved chunks before context is sent to the model.

---

## 2. ZIM File Manager CLI (`src/manage_zim.py`)

The command-line tool for managing your local knowledge base files.

```/dev/null/shell.sh#L1-7
python -m src.manage_zim list                       # List all available preset files and their sizes
python -m src.manage_zim status                     # Show all installed ZIM files
python -m src.manage_zim status <preset>            # Show install status for one preset (research, learn, literature, coding)
python -m src.manage_zim install <file_id>          # Download a specific file by its Kiwix ID
python -m src.manage_zim uninstall <file_id>        # Remove a file and its manifest entry
python -m src.manage_zim install-preset <preset>    # Interactive checkbox picker for a preset's files
python -m src.manage_zim install-devdocs            # Interactive checkbox picker for all 231 DevDocs entries
```

- Downloads come from the **live Kiwix OPDS catalog** — always up to date
- Prefers **text-only (`nopic`) flavours** automatically to keep sizes small
- Downloads show a **live progress bar** (`████░░ 45.2%  210 MB / 465 MB`)
- Tracks all installed files in `zim_manifest.json`

---

## 3. REST API (`src/main.py`)

Start the server with: `uvicorn src.main:app --host 0.0.0.0 --port 8000`

Interactive docs available at `http://localhost:8000/docs`

### Health & Configuration
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Server status, whether DB is loaded, active preset |
| `GET` | `/config` | View current AI endpoint, model, and settings |
| `GET` | `/config/models` | List models available at the configured endpoint (or `?endpoint=<url>` to probe any URL) |
| `POST` | `/config/set-ai-endpoint` | Set `ai_endpoint` and optionally `ai_model` (auto-detected if omitted) |
| `PATCH` | `/config` | Update any combination of the four settings (all fields optional) |

### Cache
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/cache/stats` | View query embedding and search-result cache statistics |
| `POST` | `/cache/clear` | Clear all cached query embeddings and search results |

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
| `POST` | `/search` | Auto-selected retrieval (`hybrid`, `faiss`, or `bm25`) — returns top-k chunks; response includes `search_mode` |

### Chat & Conversations
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/chat` | Send a message; get a context-grounded AI response with a `Sources:` block and a `sources` field in the JSON |
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

### OpenAI-Compatible API
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/v1/models` | Proxies model discovery to the configured upstream AI server |
| `POST` | `/v1/chat/completions` | RAG proxy for OpenAI-compatible chat; injects retrieved context when available, then forwards to the upstream AI server |
| `*` | `/v1/{path}` | Pass-through proxy for other OpenAI-compatible endpoints such as embeddings, responses, audio, or provider-specific routes |

**How the RAG proxy works**: When you send a message to `/v1/chat/completions`, Tensor Serve automatically:
1. Reads the original request body without converting it into a local response schema
2. Extracts the last user message
3. Analyzes whether retrieval is needed and selects `hybrid`, `faiss`, or `bm25`
4. Uses cached embeddings/results when available, otherwise retrieves relevant context chunks
5. Optionally reranks retrieved chunks, then prepends a context system message
6. Forwards the modified request to the configured upstream endpoint's `/v1/chat/completions`
7. Returns the upstream response body, status code, and content type directly to the client

All other `/v1/*` routes are forwarded unchanged. If no vector database is loaded, or if query analysis decides retrieval is unnecessary, chat requests are forwarded without context injection. This keeps Tensor Serve focused on offline context while the local AI server remains responsible for its own API surface.

**Point any OpenAI-compatible tool at `http://localhost:8000/v1`** (or `http://localhost:8000` for tools that auto-discover models):

| Tool | Configuration |
|---|---|
| **Zed** | Settings: `assistant.openai_api_url` = `http://localhost:8000` |
| **Cursor** | Settings → Models → OpenAI Base URL = `http://localhost:8000` |
| **Continue (VS Code)** | `~/.continue/config.json` → `models` → `apiBase` = `http://localhost:8000/v1` |
| **Aider** | `--openai-api-base http://localhost:8000/v1` |
| **Open WebUI** | Admin → Connections → OpenAI API → Base URL = `http://localhost:8000/v1` |
| **OpenAI SDKs** | `client = OpenAI(base_url="http://localhost:8000/v1")` |

---

## 4. Built-in Presets

Four curated knowledge bases, each automatically selecting text-only ZIM variants:

| Preset | Contents | Approx. size |
|---|---|---|
| `research` | Wikipedia, Wikisource, Wikinews | ~65 GB |
| `learn` | Wikiversity, Wikibooks | ~5 GB |
| `literature` | Project Gutenberg, Wikibooks | ~63 GB |
| `coding` | Stack Overflow, All DevDocs (231 entries) | ~80 GB + ~588 MB |

---

## 5. Runtime Files

Generated runtime files live at the repository root:

| File or directory | Purpose |
|---|---|
| `config.json` | Current AI endpoint, model, and retrieval settings |
| `presets.json` | Saved preset state and active preset |
| `zim_manifest.json` | Record of installed ZIM files |
| `zim_files/` | Downloaded ZIM files |
| `*.index`, `*.pkl`, `*.bm25` | Generated FAISS, text-store, and BM25 database artifacts |
| `conversations.db` | SQLite conversation history, when created |

## 6. Source Code

Application source lives in `src/`. See [`src/README.md`](src/README.md) for the module map and code-level architecture notes.

---

### Settings

All four settings are readable via `GET /config` and writable via `PATCH /config`.
`ai_endpoint` and `ai_model` can also be set together with `POST /config/set-ai-endpoint`.

| Setting | Default | What it controls |
|---|---|---|
| `ai_endpoint` | `null` | URL of the upstream local AI server (required for `/chat` and `/v1/*` proxying) |
| `ai_model` | `null` | Default model name used by native `/chat`; `/v1/*` proxy requests keep the client's requested model |
| `context_size` | `3` | Number of vector DB chunks retrieved as context per chat message |
| `max_conversation_history` | `20` | Maximum messages returned by `GET /conversation/{id}` |

Advanced retrieval settings are read from `config.json`:

| Setting | Default | What it controls |
|---|---|---|
| `relevance_threshold` | `0.05` | Minimum hybrid-search relevance score required for a chunk to be included |
| `query_analysis_enabled` | `true` | Whether simple queries can skip RAG and domain queries can auto-select search mode |
| `reranker_enabled` | `false` | Whether retrieved chunks are passed through the optional cross-encoder reranker |

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
uvicorn src.main:app --reload

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
