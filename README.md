What Tensor Serve does

Tensor Serve is an **offline-first AI backend**. It downloads documentation and knowledge bases as ZIM files, builds a local semantic vector database from them, and uses that database to give an AI model relevant context when answering questions — all without sending your data to the internet.

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
| `POST` | `/config/set-ai-endpoint` | Point the server at a local LLM (Ollama, vLLM, etc.) |

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
| `GET` | `/zim/installed` | List every installed ZIM file with path and size |
| `GET` | `/zim/status/{preset_id}` | Per-file install status for a preset |
| `GET` | `/zim/devdocs` | Live catalog of all 231 DevDocs entries from Kiwix |
| `POST` | `/zim/devdocs/install` | Queue DevDocs downloads in the background (specific IDs or all) |

### Vector Database
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/ingest` | Ingest a single ZIM file into a named vector DB |
| `POST` | `/ingest-multiple` | Ingest multiple ZIM files into one combined vector DB |
| `GET` | `/load?name=<db>` | Load a previously ingested database into memory |
| `POST` | `/search` | Semantic search — returns the top-k most relevant text chunks |

### Chat & Conversations
| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/chat` | Send a message; get a context-grounded AI response |
| `GET` | `/conversation/{id}` | Retrieve the full history of a conversation |

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
2. **Ingest** — Articles extracted, HTML stripped, split into 500-word overlapping chunks, embedded with `sentence-transformers`, indexed in FAISS
3. **Auto-load** — On server startup, the last active preset's database is loaded automatically
4. **Chat** — User message is embedded → top-k similar chunks retrieved → chunks + message sent to the local LLM → response returned with the source context included

---

## 5. Key files

| File | Purpose |
|---|---|
| `main.py` | FastAPI application and all API routes |
| `manage_zim.py` | CLI for downloading and managing ZIM files |
| `presets.py` | Preset definitions and configuration persistence |
| `zim_downloader.py` | Kiwix OPDS catalog interface and download engine |
| `ingest.py` / `multi_ingest.py` | ZIM → vector database pipeline |
| `embedder.py` | Sentence-transformer embeddings |
| `vectordb.py` | FAISS index wrapper (save/load/search) |
| `chunker.py` | 500-word overlapping text chunker |
| `utils.py` | ZIM article iterator and HTML cleaner |
| `ai_client.py` | HTTP client for the local LLM endpoint |
| `conversations.py` | SQLite-backed conversation history |
| `config.py` | Persistent JSON configuration |
| `presets.json` | Saved preset state and active preset |
| `zim_manifest.json` | Record of all installed ZIM files
