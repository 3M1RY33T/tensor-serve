# Tensor Serve

Tensor Serve is an **offline-first AI backend**. It downloads documentation and knowledge bases as ZIM files, builds a local semantic vector database from them, and uses that database to give an AI model relevant context when answering questions — while keeping your data private.

---

## 1. How the AI pipeline works

1. **Download** — ZIM files fetched from Kiwix and stored in the configured ZIM source folder (`zim_files/` by default)
2. **Ingest** — Articles extracted, HTML stripped, split into 500-word overlapping chunks, embedded with `sentence-transformers`, indexed in FAISS **and** BM25
3. **Auto-load** — On server startup, the last active collection's FAISS and BM25 indexes are loaded automatically
4. **Analyze** — Simple queries can skip retrieval; domain-specific queries use the query analyzer to choose the best search mode (`hybrid`, `faiss`, or `bm25`)
5. **OpenAI-compatible proxy** — For `/v1/chat/completions`, the user message is embedded (or served from cache) → hybrid search retrieves top-k chunks → optional cross-encoder reranking improves result order → retrieved context is injected into the request before it is forwarded to the upstream AI server.

### Hybrid search (FAISS + BM25 via Reciprocal Rank Fusion)

Search requests and OpenAI-compatible chat requests can run **two retrievals in parallel** and merge them:

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
python -m src.manage_zim list                       # List all available category files and their sizes
python -m src.manage_zim status                     # Show all installed ZIM files
python -m src.manage_zim status <category>          # Show install status for one category
python -m src.manage_zim install <file_id>          # Download a specific file by its Kiwix ID
python -m src.manage_zim uninstall <file_id>        # Remove a file and its manifest entry
python -m src.manage_zim install-category <category> # Interactive checkbox picker for a category's files
python -m src.manage_zim install-devdocs            # Interactive checkbox picker for all 231 DevDocs entries
```

- Downloads are resolved through the **live Kiwix OPDS catalog**; archive freshness depends on what Kiwix publishes
- Prefers **text-only (`nopic`) flavours** automatically to keep sizes small
- Downloads show a **live progress bar** (`████░░ 45.2%  210 MB / 465 MB`)
- Tracks all installed files in `zim_manifest.json`
- Use `PUT /zim/source-folder` to point Tensor Serve at an existing folder of ZIM files, `zim_files/` will be created in the project root if not configured.

If you already have `.zim` files elsewhere, point Tensor Serve at that folder before listing or installing files:

```bash
curl -X PUT http://localhost:8000/zim/source-folder \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/zim"}'
```

You can also register a single existing ZIM file without changing the source folder:

```bash
curl -X POST http://localhost:8000/zim/register \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/zim/stackoverflow.com_en_all.zim",
    "file_id": "stackoverflow.com_en_all",
    "title": "Stack Overflow"
  }'
```

---

## 3. REST API (`main.py`)

Start the server with: `uvicorn main:app --host 0.0.0.0 --port 8000`

Interactive docs available at `http://localhost:8000/docs`

### Health & Configuration
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Server status, whether DB is loaded, active collection |
| `GET` | `/config` | View current AI endpoint, model, and settings |
| `GET` | `/config/models` | List models available at the configured endpoint (or `?endpoint=<url>` to probe any URL) |
| `POST` | `/config/set-ai-endpoint` | Set `ai_endpoint` and optionally `ai_model` (auto-detected if omitted) |
| `PATCH` | `/config` | Update any combination of the core settings (all fields optional) |

### Cache
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/cache/stats` | View query embedding and search-result cache statistics |
| `POST` | `/cache/clear` | Clear all cached query embeddings and search results |

### Collections
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/collections` | List collection folders in the active ZIM source folder |
| `GET` | `/collections/{collection_id}` | Details and ZIM files for one collection folder |
| `GET` | `/collections/{collection_id}/files` | List the ZIM files referenced by one collection folder |
| `POST` | `/collections` | Create a collection folder, optionally adding selected ZIM files |
| `PATCH` | `/collections/{collection_id}` | Rename a collection or update its description |
| `POST` | `/collections/{collection_id}/files` | Add additional `.zim` files to a collection without duplicating archives |
| `DELETE` | `/collections/{collection_id}/files` | Delete selected `.zim` files from a collection folder |
| `POST` | `/collections/{collection_id}/ingest` | Process all ZIM files in a collection folder into a vector database |
| `DELETE` | `/collections/{collection_id}` | Delete collection metadata and the collection folder |
| `POST` | `/collections/custom/create` | Legacy alias for `POST /collections` |
| `DELETE` | `/collections/custom/{collection_id}` | Legacy alias for `DELETE /collections/{collection_id}` |

### ZIM File Management
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/zim/available` | List curated category files with installed/not-installed status |
| `GET` | `/zim/installed` | List every installed ZIM file with path and size (scans disk, auto-registers untracked files) |
| `GET` | `/zim/source-folder` | Show the active ZIM source folder |
| `PUT` | `/zim/source-folder` | Point Tensor Serve at an existing folder of `.zim` files |
| `DELETE` | `/zim/source-folder` | Reset ZIM storage to the default `zim_files/` folder |
| `POST` | `/zim/register` | Register one existing `.zim` file path in the manifest without downloading it |
| `POST` | `/zim/source-folder/ingest` | Ingest every `.zim` file under the active source folder |
| `GET` | `/zim/status/{collection_id}` | Per-file status for a collection folder |
| `POST` | `/zim/install` | Queue one or more ZIM files for background download by Kiwix ID |
| `POST` | `/zim/install-category` | Queue all or selected files from a curated category |
| `DELETE` | `/zim/uninstall/{file_id}` | Remove a ZIM file from disk and the manifest |
| `GET` | `/zim/devdocs` | Live catalog of all 231 DevDocs entries from Kiwix |
| `POST` | `/zim/devdocs/install` | Queue DevDocs downloads in the background (specific IDs or all) |
| `GET` | `/zim/progress` | Poll aggregate progress for all active and recent downloads |
| `GET` | `/zim/progress/{file_id}` | Poll download progress for a single file by Kiwix ID |

### Vector Database
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/ingest` | Ingest a single ZIM file or a directory of ZIM files into a named vector DB |
| `POST` | `/ingest-multiple` | Ingest multiple ZIM files and/or directories into one combined vector DB |
| `GET` | `/load?name=<db>` | Load a previously ingested database into memory |
| `POST` | `/search` | Auto-selected retrieval (`hybrid`, `faiss`, or `bm25`) — returns top-k chunks; response includes `search_mode` |

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
| `POST` | `/clean` | Delete all vector DB index files, text stores, BM25 indexes, and `__pycache__/` — preserves collections, config, manifest, and ZIM files |

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

## 4. Curated Categories

Four curated categories are available for convenience. They help queue downloads, but they do not define what gets ingested. Ingestion reads actual `.zim` files from source-folder paths and collection folders.

| Category | Contents | Approx. size |
|---|---|---|
| Research | Wikipedia, Wikisource, Wikinews | ~65 GB |
| Learning | Wikiversity, Wikibooks | ~5 GB |
| Literature | Project Gutenberg, Wikibooks | ~63 GB |
| Coding | Stack Overflow, All DevDocs (231 entries) | ~80 GB + ~588 MB |

---

## 5. Runtime Files

Generated runtime files live at the repository root:

| File or directory | Purpose |
|---|---|
| `config.json` | Current AI endpoint, model, and retrieval settings |
| `collections.json` | Saved collection metadata and active collection |
| `zim_manifest.json` | Record of installed ZIM files |
| `zim_files/` | Default downloaded ZIM folder, created only when no custom `zim_source_folder` is configured |
| `*.index`, `*.pkl`, `*.bm25` | Generated FAISS, text-store, and BM25 database artifacts |

## 6. Source Code

Application source lives in `src/`. See [`src/README.md`](src/README.md) for the module map and code-level architecture notes.

---

### Settings

Core settings are readable via `GET /config` and writable via `PATCH /config`.
`ai_endpoint` and `ai_model` can also be set together with `POST /config/set-ai-endpoint`.

| Setting | Default | What it controls |
|---|---|---|
| `ai_endpoint` | `null` | URL of the upstream local AI server required for `/v1/*` proxying |
| `ai_model` | `null` | Saved model selection from endpoint auto-detection; `/v1/*` proxy requests keep the client's requested model |
| `context_size` | `3` | Number of vector DB chunks retrieved as context per OpenAI-compatible chat request |
| `zim_source_folder` | `null` | Optional folder to scan for existing `.zim` files and use for future downloads |

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
    "zim_source_folder": "/data/zim"
  }'
```

**Example — reset ZIM storage to the default `zim_files/` folder:**
```bash
curl -X DELETE http://localhost:8000/zim/source-folder
```

**Example — create and ingest a collection folder from local ZIM paths:**
```bash
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{
    "collection_id": "local_docs",
    "name": "Local Docs",
    "description": "My local documentation set",
    "zim_paths": ["/data/zim/docs.zim"]
  }'

curl -X POST http://localhost:8000/collections/local_docs/ingest
```

Selected ZIM files are referenced from the collection folder with lightweight
filesystem links. Files already inside the active ZIM source folder are linked
in place; files outside it are copied once into the source folder root and then
linked from collections. To create an empty collection, send an empty
`zim_paths` list.

**Example — rename a collection and add/remove files:**
```bash
curl -X PATCH http://localhost:8000/collections/local_docs \
  -H "Content-Type: application/json" \
  -d '{"name": "Reference Docs"}'

curl -X POST http://localhost:8000/collections/local_docs/files \
  -H "Content-Type: application/json" \
  -d '{"zim_paths": ["/data/zim/python.zim", "/data/zim/sqlite.zim"]}'

curl -X DELETE http://localhost:8000/collections/local_docs/files \
  -H "Content-Type: application/json" \
  -d '{"file_names": ["sqlite.zim"]}'
```

**Example — ingest an entire collection folder by path:**
```bash
curl -X POST http://localhost:8000/ingest-multiple \
  -H "Content-Type: application/json" \
  -d '{
    "zim_paths": ["/data/zim/local_docs"],
    "output_name": "local_docs_db"
  }'
```

**Example — ingest the entire active ZIM source folder:**
```bash
curl -X POST http://localhost:8000/zim/source-folder/ingest \
  -H "Content-Type: application/json" \
  -d '{"output_name": "all_zims_db"}'
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

# 6. Start chatting through the OpenAI-compatible proxy
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "Who invented the telephone?"}
    ]
  }'
```
