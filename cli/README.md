## ZIM File Manager CLI (`tensor-serve zim`)

The command-line tool for managing your local knowledge base files.

```/dev/null/shell.sh#L1-7
python -m tensor_serve zim list                       # List all available category files and their sizes
python -m tensor_serve zim status                     # Show all installed ZIM files
python -m tensor_serve zim status <category>          # Show install status for one category
python -m tensor_serve zim install <file_id>          # Download a specific file by its Kiwix ID
python -m tensor_serve zim uninstall <file_id>        # Remove a file and its manifest entry
python -m tensor_serve zim install-category <category> # Interactive checkbox picker for a category's files
python -m tensor_serve zim install-devdocs            # Interactive checkbox picker for all 231 DevDocs entries
```

- Downloads are resolved through the **live Kiwix OPDS catalog**; archive freshness depends on what Kiwix publishes
- Prefers **text-only (`nopic`) flavours** automatically to keep sizes small
- Downloads can be tracked via **live progress bar**
- Tracks all installed files in `zim_manifest.json`
- Use `PUT /zim/source-folder` to point Tensor Serve at an existing folder of ZIM files, `zim_files/` will be created in the project root if not configured.

### Curated Categories

Four curated categories are available for convenience. They help queue downloads, but they do not define what gets ingested. Ingestion reads actual `.zim` files from source-folder paths, directories, or metadata-backed collections.

| Category | Contents | Approx. size |
|---|---|---|
| Research | Wikipedia, Wikisource, Wikinews | ~65 GB |
| Learning | Wikiversity, Wikibooks | ~5 GB |
| Literature | Project Gutenberg, Wikibooks | ~63 GB |
| Coding | Stack Overflow, All DevDocs (231 entries) | ~80 GB + ~588 MB |

## Configuration CLI (`tensor-serve config`)

Configure the local AI endpoint, ZIM file source, search behavior, and web search settings from the terminal.

```bash
tensor-serve config show
tensor-serve config set-ai-endpoint --endpoint http://localhost:11434 --model gpt-4o-mini --api-key "MY_API_KEY"
tensor-serve config set-zim-source ~/data/zim
tensor-serve config clear-zim-source
tensor-serve config set-search-modes --keyword-mode auto --semantic-mode on
tensor-serve config set-context-size 3
tensor-serve config enable-web-search --provider duckduckgo
tensor-serve config disable-web-search
tensor-serve config reset
tensor-serve config list-models --endpoint http://localhost:11434
tensor-serve config detect-local-ai
```

Available configuration commands:

- `show` — Print the current configuration from `config.json` with secrets masked
- `set-ai-endpoint` — Set the AI endpoint, optional model and API key
- `set-zim-source` — Configure a custom ZIM source folder
- `clear-zim-source` — Reset ZIM storage back to the default `zim_files/` folder
- `set-search-modes` — Update keyword and semantic search mode behavior
- `set-context-size` — Change how many context documents are used
- `enable-web-search` / `disable-web-search` — Toggle web search for time-sensitive queries
- `reset` — Reset `config.json` to default settings; add `--server http://localhost:8000` to reset a running server through `POST /config/reset`
- `list-models` — Probe an endpoint and list available models
- `detect-local-ai` — Discover local runtimes such as Ollama, LM Studio, or LocalAI

API keys and extra upstream headers are encrypted at rest before they are written to `config.json`. Tensor Serve creates `.tensor_config.key` by default; use `TENSOR_CONFIG_KEY` or `TENSOR_CONFIG_KEY_FILE` when you want to manage the encryption key outside the project folder.

## Health CLI (`tensor-serve health`)

Check the status of a running Tensor Serve server from the terminal.

```bash
tensor-serve health
tensor-serve health --server http://localhost:8000
```

This calls `GET /health` and reports whether the server is up, whether FAISS/BM25 indexes are loaded, whether an AI endpoint is configured, and which collection is active.

## Cache CLI (`tensor-serve cache`)

Inspect or clear the running server's in-memory query and embedding cache.

```bash
tensor-serve cache stats
tensor-serve cache clear
tensor-serve cache stats --server http://localhost:8000
```

Available cache commands:

- `stats` — Call `GET /cache/stats` and show embedding/search cache sizes, max size, and TTL
- `clear` — Call `POST /cache/clear` and remove cached embeddings and search results from the running server

Cache state lives inside the server process, so these commands require Tensor Serve to be running.

## Cleanup CLI (`tensor-serve clean`)

Remove generated working files through the running Tensor Serve server.

```bash
tensor-serve clean
tensor-serve cleanup --server http://localhost:8000
tensor-serve reset
tensor-serve clean-all --keep-collection-folders
```

This calls `POST /clean`, which deletes generated vector database files (`*.index`, `*.pkl`, `*.bm25`) and `__pycache__/` from the server's working directory, then resets the server's loaded database state. It preserves `collections.json`, `config.json`, `zim_manifest.json`, and ZIM files. `tensor-serve cleanup` is accepted as an alias.

Use `tensor-serve reset` when you want the full server-side cleanup bundle. It calls `POST /clean/all`, clears caches, removes generated vector databases, resets `collections.json`, removes matching legacy collection folders by default, and resets `config.json`. ZIM archives are preserved. `tensor-serve clean-all` is accepted as an alias.

## Ingestion CLI (`tensor-serve ingest`)

Create FAISS and BM25 vector databases from local ZIM files without making REST calls.

```bash
tensor-serve ingest /data/zim/python.zim --output-name python_docs
tensor-serve ingest /data/zim/python.zim /data/zim/sqlite.zim --output-name coding_docs
tensor-serve ingest /data/zim --output-name local_docs
tensor-serve ingest --source-folder --output-name all_zims
tensor-serve ingest --collection local_docs --output-name local_docs_db
```

The CLI accepts individual `.zim` files, directories, the active ZIM source folder, or a named collection. Directories are expanded recursively to every `.zim` file they contain. `--source-folder` ingests the active ZIM source folder configured with `tensor-serve config set-zim-source`; `--collection <collection_id>` ingests every `.zim` file referenced by that collection and marks it active for startup auto-load. Always pass `--output-name` so the generated vector database has an explicit name.

## Vector Database CLI (`tensor-serve db`)

List local vector databases and switch the database loaded by a running Tensor Serve server.

```bash
tensor-serve db list
tensor-serve db show local_docs
tensor-serve db load local_docs
tensor-serve db use local_docs --server http://localhost:8000
tensor-serve db status
```

Available database commands:

- `list` — Show local databases with matching `.index` and `.pkl` files
- `show <name>` — Inspect the files for one local database
- `load <name>` / `use <name>` — Call the running server's `/load?name=<db>` endpoint to switch the in-memory database
- `status` — Call the running server's `/health` endpoint to show loaded database status

Loading or switching a database changes state inside the running server process, so `db load` and `db status` require Tensor Serve to be running.

## Collections CLI (`tensor-serve collections`)

Create, inspect, update, delete, select, and ingest ZIM collections from the terminal.

```bash
tensor-serve collections list
tensor-serve collections create local_docs --name "Local Docs" --description "Project documentation"
tensor-serve collections add-files local_docs /data/zim/python.zim /data/zim/sqlite.zim
tensor-serve collections add-files local_docs /data/zim
tensor-serve collections files local_docs
tensor-serve collections update local_docs --description "Python and SQLite docs"
tensor-serve collections ingest local_docs --output-name local_docs_db
tensor-serve collections use local_docs
tensor-serve collections active
tensor-serve collections remove-files local_docs sqlite.zim
tensor-serve collections delete local_docs
tensor-serve collections reset
```

Available collection commands:

- `list` — List metadata-backed collections and any legacy collection folders
- `active` — Show the collection selected for server startup auto-load
- `show <collection_id>` — Show collection metadata and files
- `files <collection_id>` — List files referenced by a collection
- `create <collection_id>` — Create a metadata-backed collection; add `--zim-path <path>` one or more times to seed it from files or directories without copying or linking archives
- `update <collection_id>` — Update `--name` and/or `--description`
- `delete <collection_id>` — Delete collection metadata and any matching legacy collection folder
- `add-files <collection_id> <paths...>` — Add references to existing `.zim` files or every `.zim` file under one or more directories without copying or linking archives
- `remove-files <collection_id> <file_names...>` — Remove selected `.zim` references from the collection
- `ingest <collection_id> --output-name <db>` — Build the named vector database and make the collection active
- `use <collection_id>` — Select the active collection for startup auto-load
- `reset` — Reset `collections.json` and remove matching legacy collection folders; add `--keep-folders` to reset metadata only, or `--server http://localhost:8000` to call `POST /collections/reset`

`tensor-serve collection ...` is accepted as an alias for `tensor-serve collections ...`.

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
