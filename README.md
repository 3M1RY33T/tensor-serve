# Tensor (Serve)

`tensor-serve` is a ZIM-based retrieval augmented proxy for any OpenAI-compatible AI. This program (optionally) lets you download ZIM documentation from the **live Kiwix OPDS catalog**, builds a local semantic vector database from them, and uses that database to provide an AI model relevany context when answering questions.

The purpose of this program is to provide the service for customizing your AI for your specific needs seamlessly.

Tensor works with any OpenAI-compatible endpoint — local runtimes (Ollama, LM Studio), cloud APIs (OpenAI, Anthropic), or LLM gateways (LiteLLM, vLLM). API keys enable authentication with services that require them.

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

---

## 3. CLI Reference

[CLI Reference](cli/README.md) can be found here. Contains `ZIM File Manager` and `Configuration CLI`.

---

## 5. REST API (`api/main.py`)

[API Reference](api/README.md) can be found here. Contains `Health & Configuration`, `Cache`, `Collections`, `ZIM File Management`, `Vector Database`, `Download progress fields`, `Cleanup`, `OpenAI-Compatible API`, `Settings`, `Web Search for Time-Sensitive Information`, `Search Mode Customization`, `Model auto-detection`.

---

## Setup

### Prerequisites

- **Python 3.8+** (check with `python3 --version`)
- **pip** (Python package manager, usually bundled with Python)
- **An OpenAI-compatible AI endpoint** (examples: Ollama, LM Studio, OpenAI API, Anthropic, LiteLLM gateway) — optional for basic setup, required for chat functionality

### Installation

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/yourusername/tensor-serve.git
   cd tensor-serve
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation** (optional):
   ```bash
   python -m tensor_serve zim list
   ```

5. **Start the server**:
   ```bash
   # Basic startup (port 8000)
   python -m tensor_serve start
   
   # Custom port
   python -m tensor_serve start --port 3000
   
   # Auto-select available port if 8000 is in use
   python -m tensor_serve start --auto-port
   
   # Development mode with auto-reload
   python -m tensor_serve start --reload
   ```

   **Server Options**:
   - `--port, -p`: Specify port (default: 8000, or `TENSOR_PORT` env var)
   - `--host`: Host to bind to (default: 0.0.0.0, or `TENSOR_HOST` env var)
   - `--auto-port`: Automatically find available port if specified port is in use
   - `--reload`: Enable auto-reload for development

   **Environment Variables**:
   - `TENSOR_PORT`: Default port
   - `TENSOR_HOST`: Default host
   - `TENSOR_AUTO_PORT=true`: Enable auto-port selection by default

6. **Manage ZIM files**:
   ```bash
   # List available ZIM files
   python -m tensor_serve zim list
   
   # Show installed ZIM files
   python -m tensor_serve zim status
   
   # Install a ZIM file
   python -m tensor_serve zim install wikipedia_en_all
   
   # Interactive category installation
   python -m tensor_serve zim install-category coding
   ```

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
- A ZIM file is available (download via `python -m tensor_serve zim install <file_id>` or place one in `zim_files/`)

**Steps:**

```bash
# 1. Activate the virtual environment (if you created one)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Start the server
python -m tensor_serve start

# 3. In another terminal, configure AI endpoint (assuming Ollama running on port 11434)
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "mistral"
  }'

# 4. Check health
curl http://localhost:8000/health

# 5. Download a ZIM file (or use an existing one)
python -m tensor_serve zim install wikipedia_en_all  # Downloads Wikipedia

# 6. Ingest ZIM file(s) into a vector database
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "zim_path": "zim_files/wikipedia_en_all.zim",
    "output_name": "wiki"
  }'

# 7. Load the database into memory
curl http://localhost:8000/load?name=wiki

# 8. (Optional) Enable web search for time-sensitive queries
curl -X POST http://localhost:8000/config/web-search/enable

# 9. Start chatting through the OpenAI-compatible proxy
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "tensor_show_resources": false,
    "messages": [
      {"role": "user", "content": "Who invented the telephone?"}
    ]
  }'

# 10. Time-sensitive query (if web search enabled, will search web + ZIM)
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
