## REST API (`api/main.py`)

Start the server with: `python -m tensor_serve start` (recommended) or `uvicorn api.main:app --host 0.0.0.0 --port 8000`

Interactive docs available at `http://localhost:8000/docs`

### Health & Configuration
| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Server status, whether DB is loaded, active collection |
| `GET` | `/config` | View current AI endpoint, model, and settings |
| `GET` | `/config/models` | List models available at the configured endpoint (or `?endpoint=<url>` to probe any URL) |
| `GET` | `/config/local-ai/detect` | Probe common local AI runtimes such as Ollama and LM Studio |
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
7. Appends a short resource footer when local context was used, such as `Read from Stack Overflow` and `Enhanced by coding_db`
8. Returns the upstream status code and content type to the client

All other `/v1/*` routes are forwarded unchanged. If no vector database is loaded, or if query analysis decides retrieval is unnecessary, chat requests are forwarded without context injection. This keeps Tensor Serve focused on offline context while the local AI server remains responsible for its own API surface.

To suppress the resource footer for a request, add `"tensor_show_resources": false`
to the `/v1/chat/completions` JSON body. Tensor removes that field before
forwarding the request upstream.

**Point any OpenAI-compatible tool at `http://localhost:8000/v1`** (or `http://localhost:8000` for tools that auto-discover models):

| Tool | Configuration |
|---|---|
| **Zed** | Settings: `assistant.openai_api_url` = `http://localhost:8000` |
| **Cursor** | Settings → Models → OpenAI Base URL = `http://localhost:8000` |
| **Continue (VS Code)** | `~/.continue/config.json` → `models` → `apiBase` = `http://localhost:8000/v1` |
| **Aider** | `--openai-api-base http://localhost:8000/v1` |
| **Open WebUI** | Admin → Connections → OpenAI API → Base URL = `http://localhost:8000/v1` |
| **OpenAI SDKs** | `client = OpenAI(base_url="http://localhost:8000/v1")` |

### Settings

Core settings are readable via `GET /config` and writable via `PATCH /config`.
`ai_endpoint` and `ai_model` can also be set together with `POST /config/set-ai-endpoint`.

| Setting | Default | What it controls |
|---|---|---|
| `ai_provider` | `openai-compatible` | Label for the upstream provider or gateway |
| `ai_endpoint` | `null` | URL of the upstream local AI server, cloud API, or LLM gateway required for `/v1/*` proxying |
| `ai_model` | `null` | Saved model selected for Tensor-enhanced chat; `/v1/chat/completions` requests are forwarded using this model when configured |
| `ai_api_key` | `null` | Optional provider API key; `GET /config` only reports whether one is configured |
| `ai_api_key_header` | `Authorization` | Header used for the API key |
| `ai_api_key_prefix` | `Bearer` | Optional prefix prepended to the API key; use an empty string for raw-key headers |
| `ai_extra_headers` | `{}` | Optional additional upstream headers for gateways or provider-specific requirements |
| `context_size` | `3` | Number of vector DB chunks retrieved as context per OpenAI-compatible chat request |
| `zim_source_folder` | `null` | Optional folder to scan for existing `.zim` files and use for future downloads |

Advanced retrieval settings are read from `config.json`:

| Setting | Default | What it controls |
|---|---|---|
| `relevance_threshold` | `0.05` | Minimum hybrid-search relevance score required for a chunk to be included |
| `query_analysis_enabled` | `true` | Whether simple queries can skip RAG and domain queries can auto-select search mode |
| `reranker_enabled` | `false` | Whether retrieved chunks are passed through the optional cross-encoder reranker |
| `web_search_enabled` | `false` | Whether web search is enabled for time-sensitive queries |
| `web_search_provider` | `duckduckgo` | Which search provider to use (`duckduckgo`, `brave`, or `google`) |
| `web_search_api_key` | `null` | API key for web search provider (if required) |
| `web_search_engine_id` | `null` | Google Custom Search engine ID (only for Google provider) |
| `web_search_results` | `3` | Number of web search results to retrieve per query |

### Web Search for Time-Sensitive Information

Web search is **disabled by default** and **only triggered for time-sensitive queries** that mention keywords like:
- Recent time: "latest", "today", "yesterday", "this week", "current", "breaking", "trending", "recently"
- Specific years: "2024", "2025", "2026"
- Real-time info: "live", "now", "upcoming", "what's new", "real-time"
- Current events: "news", "stock", "price", "weather", "election", "pandemic", "outbreak", etc.

When enabled, time-sensitive queries retrieve results from both offline ZIM indexes and web search, then merge them via Reciprocal Rank Fusion. Local knowledge is prioritized.

**Enable web search with DuckDuckGo** (no API key required):
```bash
curl -X POST http://localhost:8000/config/web-search/enable \
  -H "Content-Type: application/json" \
  -d '{"provider": "duckduckgo"}'
```

**Enable web search with Brave Search** (requires API key):
```bash
# 1. Get a free Brave Search API key from https://api.search.brave.com
# 2. Configure it:
curl -X POST http://localhost:8000/config/web-search/set-provider \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "brave",
    "api_key": "your-brave-api-key"
  }'

# 3. Enable web search:
curl -X POST http://localhost:8000/config/web-search/enable \
  -H "Content-Type: application/json" \
  -d '{"provider": "brave"}'
```

**Enable web search with Google Custom Search** (requires credentials):
```bash
# 1. Set up a custom search engine at https://programmablesearchengine.google.com
# 2. Configure it:
curl -X POST http://localhost:8000/config/web-search/set-provider \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "google",
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id"
  }'

# 3. Enable web search:
curl -X POST http://localhost:8000/config/web-search/enable \
  -H "Content-Type: application/json" \
  -d '{"provider": "google"}'
```

**Check web search status**:
```bash
curl http://localhost:8000/config/web-search/status
```

**Disable web search**:
```bash
curl -X POST http://localhost:8000/config/web-search/disable
```

### Search Mode Customization

Control which search methods are available for different queries:

**Keyword Search Modes** — how to handle term-based and API name lookups:
- `auto` (default): Automatically select based on query characteristics
- `web`: Use web search only for keyword matching (requires web search enabled)
- `zim`: Use local ZIM indexes only (BM25)
- `off`: Disable keyword search entirely

**Semantic Search Modes** — how to handle conceptual and explanation queries:
- `auto` (default): Automatically decide based on query characteristics
- `on`: Always use semantic search (FAISS) when available
- `off`: Disable semantic search entirely

**View current search modes**:
```bash
curl http://localhost:8000/config/search-modes
```

**Update search modes**:
```bash
curl -X PATCH http://localhost:8000/config/search-modes \
  -H "Content-Type: application/json" \
  -d '{
    "keyword_search_mode": "zim",     # Optional: auto|web|zim|off
    "semantic_search_mode": "auto"    # Optional: auto|on|off
  }'
```

**Examples**:
- **Only semantic search** (for conceptual questions):
  ```bash
  curl -X PATCH http://localhost:8000/config/search-modes \
    -d '{"keyword_search_mode": "off", "semantic_search_mode": "on"}'
  ```

- **Only keyword search** (for code lookups and APIs):
  ```bash
  curl -X PATCH http://localhost:8000/config/search-modes \
    -d '{"keyword_search_mode": "zim", "semantic_search_mode": "off"}'
  ```

- **Web-first for keyword searches** (useful if offline knowledge base is incomplete):
  ```bash
  curl -X PATCH http://localhost:8000/config/search-modes \
    -d '{"keyword_search_mode": "web"}'
  ```

- **Disable all search** (use only model knowledge):
  ```bash
  curl -X PATCH http://localhost:8000/config/search-modes \
    -d '{"keyword_search_mode": "off", "semantic_search_mode": "off"}'
  ```

#### Model auto-detection

`ai_model` does not need to be set manually. The server can discover available models by querying the endpoint directly — it tries the OpenAI-compatible `GET /v1/models` route first (LM Studio, Ollama's OpenAI-compatible API, vLLM, LocalAI, LiteLLM, OpenAI-compatible cloud APIs, etc.), then falls back to Ollama's `GET /api/tags`.

**Example — detect local runtimes:**
```bash
curl http://localhost:8000/config/local-ai/detect
```

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

**Example — configure LM Studio:**
```bash
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{"ai_provider": "lm-studio", "ai_endpoint": "http://localhost:1234"}'
```

**Example — configure an OpenAI-compatible cloud API with a key:**
```bash
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_provider": "openai",
    "ai_endpoint": "https://api.openai.com/v1",
    "ai_model": "gpt-4.1-mini",
    "ai_api_key": "sk-..."
  }'
```

**Example — configure a LiteLLM gateway:**
```bash
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_provider": "litellm",
    "ai_endpoint": "http://localhost:4000/v1",
    "ai_model": "anthropic/claude-sonnet-4-5"
  }'
```

Provider secrets are stored in `config.json` by default. For a packaged app,
store API keys in the OS keychain and pass them to Tensor at runtime instead of
writing long-lived secrets to disk.

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
    "ai_api_key": null,
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