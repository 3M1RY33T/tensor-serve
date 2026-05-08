# <img align="left" width="50" height="50" src="assets/tensor.png"> Tensor (Serve)

`tensor-serve` is a ZIM-based retrieval augmented proxy for any OpenAI-compatible AI. This program lets you download ZIM documentation from the **live Kiwix OPDS catalog**, builds a local semantic vector database from it, and uses that database to provide an AI model relevant context when answering questions.

The purpose of this program is to provide the service for customizing your AI for your specific needs seamlessly.

Combining `keyword search` and `semantic search`, Tensor helps produce more accurate responses for the data you have included in a ZIM database.

---

## 1. How the AI pipeline works

1. **Download** — ZIM files fetched from Kiwix and stored in the configured ZIM source folder (`zim_files/` by default)
2. **Ingest** — Articles extracted, HTML stripped, split into 500-word overlapping chunks, embedded with `sentence-transformers`, indexed in FAISS **and** BM25
3. **Auto-load** — On server startup, the last active collection's FAISS and BM25 indexes are loaded automatically
4. **Analyze** — Simple queries can skip retrieval; domain-specific queries use the query analyzer to choose the best search mode (`hybrid`, `faiss`, or `bm25`); time-sensitive queries optionally trigger web search
5. **OpenAI-compatible proxy** — For `/v1/chat/completions`, the user message is embedded (or served from cache) → hybrid search retrieves top-k chunks (optionally merged with web results) → optional cross-encoder reranking improves result order → retrieved context is injected into the request before it is forwarded to the upstream AI server.

### Hybrid search (FAISS + BM25 + optional Web Search w/ Reciprocal Rank Fusion)

Search requests and OpenAI-compatible chat requests can run **up to three retrievals in parallel** and merge them:

| | FAISS (semantic) | BM25 (keyword) | Web Search |
|---|---|---|---|
| Finds | Conceptually related chunks | Exact term / token matches | Current / recent information |
| Good for | *"How does backpressure work?"* | *"asyncio.gather"*, error codes, API names | *"latest news"*, *"today's events"*, time-sensitive queries |
| Requires setup | Automatic | Automatic | Optional; disabled by default |

Results are merged with **Reciprocal Rank Fusion** (`score = Σ 1 / (60 + rank)`). Chunks that rank well in multiple result sets float to the top. The pipeline degrades gracefully — if one index is unavailable it is skipped.

The query analyzer automatically selects the search strategy:

| Mode | When it is used |
|---|---|
| `hybrid` | Mixed or general queries where semantic and keyword signals both help |
| `faiss` | Conceptual queries such as explanations, architecture, patterns, and design questions |
| `bm25` | Keyword-heavy queries such as API names, code symbols, methods, classes, errors, and short exact searches |

Query embeddings and search results are cached with an in-memory LRU cache to reduce repeated embedding and retrieval work. If enabled, the optional cross-encoder reranker performs a second-stage pass over retrieved chunks before context is sent to the model.

Detailed information about the RAG proxy implementation can be found [here](api/README.md).

---

## 3. CLI Reference

[CLI Reference](cli/README.md) can be found here. It covers ZIM downloads, configuration, health, cache, cleanup, ingestion, vector databases, and collections.

---

## 5. REST API (`api/main.py`)

[API Reference](api/README.md) can be found here. Contains `Health & Configuration`, `Cache`, `Collections`, `ZIM File Management`, `Vector Database`, `Download progress fields`, `Cleanup`, `OpenAI-Compatible API`, `Settings`, `Web Search for Time-Sensitive Information`, `Search Mode Customization`, `Model auto-detection`.

---

### Using With OpenAI-compatible Tools (Code Editors, etc..)

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

## Setup

### Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **pip** (Python package manager, usually bundled with Python)
- **An OpenAI-compatible AI endpoint** (examples: Ollama, LM Studio, OpenAI API, Anthropic, LiteLLM gateway) — optional for basic setup, required for chat functionality

### Setup Example

```bash
# 1. Clone and enter the project
git clone https://github.com/3M1RY33T/tensor-serve.git
cd tensor-serve

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install Tensor Serve and its dependencies
pip install -r requirements.txt
pip install -e .

# 4. Configure the upstream OpenAI-compatible AI endpoint
tensor-serve config detect-local-ai
tensor-serve config set-ai-endpoint \
  --endpoint http://localhost:11434 \
  --model mistral

# Optional: inspect models exposed by the configured endpoint
tensor-serve config list-models

# 5. Choose where ZIM files are stored
tensor-serve config set-zim-source ./zim_files

# 6. Browse and download ZIM content from Kiwix
tensor-serve zim list
tensor-serve zim install wikivoyage_en_europe

# Optional: use an interactive category downloader instead
# tensor-serve zim install-category coding

# 7. Review the saved configuration and installed ZIM files
tensor-serve config show
tensor-serve zim status

# 8. Start the server
tensor-serve start

# Custom port
tensor-serve start --port 3000

# Auto-select available port if 8000 is in use
tensor-serve start --auto-port

# Development mode with auto-reload
tensor-serve start --reload
```

For cloud or gateway providers, include an API key and provider-specific endpoint:

```bash
tensor-serve config set-ai-endpoint \
  --endpoint https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY"
```

API keys are encrypted before they are written to `config.json`. Tensor Serve uses a local `.tensor_config.key` file by default, or you can provide `TENSOR_CONFIG_KEY` / `TENSOR_CONFIG_KEY_FILE` for deployments that manage secrets externally.

### Supported Environments

**Local AI Runtimes** (no API key needed):
- [Ollama](https://ollama.ai) — easy single-command setup
- [LM Studio](https://lmstudio.ai/) — GUI-based model management
- [vLLM](https://github.com/vllm-project/vllm) — high-performance serving

**Cloud APIs** (API key required):
- OpenAI (`https://api.openai.com/v1`)
- Anthropic Claude
- Other OpenAI-compatible endpoints

**Gateways**:
- LiteLLM — unified interface for multiple providers

---

## Workflow

### Complete Example

**Prerequisites:**
- Tensor Serve is installed (see [Setup](#setup) above)
- An OpenAI-compatible AI endpoint is running locally or accessible via API (e.g., Ollama on `http://localhost:11434`)

**Steps:**

```bash
# 1. Start the server
tensor-serve start

# 2. Leave the server running. In another terminal, check health
tensor-serve health

# 3. Ingest all files from the configured ZIM source folder into a vector database
tensor-serve ingest --source-folder --output-name travel

# 4. Load the database into memory
tensor-serve db load travel

# 5. Optional: enable web search for time-sensitive queries with the configuration CLI
tensor-serve config enable-web-search --provider duckduckgo
tensor-serve config set-search-modes --keyword-mode auto --semantic-mode on

# 6. Start chatting through the OpenAI-compatible proxy
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "tensor_show_resources": false,
    "messages": [
      {"role": "user", "content": "Who invented the telephone?"}
    ]
  }'

# 7. Time-sensitive query (if web search is enabled, Tensor Serve can search web + ZIM)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {"role": "user", "content": "What is the latest news about AI?"}
    ]
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
- Hybrid RRF adds negligible overhead — both searches run in milliseconds (or up to 3 sources with web search)
- Chat responses depend on AI endpoint response time
- Existing databases ingested before hybrid search was added will use semantic-only search until re-ingested (no `.bm25` file present → graceful fallback)
- **Web search** (when enabled): adds 1-3 seconds per time-sensitive query; cached results are instant; disabled by default (zero overhead)

---
