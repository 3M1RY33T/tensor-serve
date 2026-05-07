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

Four curated categories are available for convenience. They help queue downloads, but they do not define what gets ingested. Ingestion reads actual `.zim` files from source-folder paths and collection folders.

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
tensor-serve config set-search-modes --keyword auto --semantic on
tensor-serve config set-context-size 3
tensor-serve config enable-web-search --provider duckduckgo
tensor-serve config disable-web-search
tensor-serve config list-models --endpoint http://localhost:11434
tensor-serve config detect-local-ai
```

Available configuration commands:

- `show` — Print the current configuration from `config.json`
- `set-ai-endpoint` — Set the AI endpoint, optional model and API key
- `set-zim-source` — Configure a custom ZIM source folder
- `clear-zim-source` — Reset ZIM storage back to the default `zim_files/` folder
- `set-search-modes` — Update keyword and semantic search mode behavior
- `set-context-size` — Change how many context documents are used
- `enable-web-search` / `disable-web-search` — Toggle web search for time-sensitive queries
- `list-models` — Probe an endpoint and list available models
- `detect-local-ai` — Discover local runtimes such as Ollama, LM Studio, or LocalAI

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