# Tensor Serve

A backend service that enables locally-run AI models to work with offline web content. Tensor Serve processes ZIM files (from Kiwix), creates semantic embeddings, and provides an AI-powered interface for retrieving context and generating responses.

## Features

- 🔗 **Independent Backend**: Works standalone without requiring the web GUI
- 📚 **ZIM File Processing**: Ingest offline Wikipedia and other Kiwix content
- 🧠 **Semantic Search**: Uses embeddings to find relevant context
- 💬 **Conversational AI**: Chat interface with context-aware responses
- 💾 **Conversation History**: Track and retrieve past conversations
- ⚙️ **Persistent Configuration**: Save AI endpoint settings
- 🚀 **Production Ready**: FastAPI with proper error handling and validation

## Architecture

```
User/Frontend
    ↓
[Tensor Serve API]
    ├─ Config Manager (config.json)
    ├─ Vector DB (FAISS + embeddings)
    ├─ Embedder (sentence-transformers)
    ├─ Conversation DB (SQLite)
    └─ AI Client (calls local LLM)
```

## Installation

### Requirements
- Python 3.8+
- Local LLM endpoint (e.g., Ollama, vLLM, text-generation-webui)
- ZIM files from Kiwix (optional)

### Setup

```bash
# Clone and navigate to directory
cd tensor-serve

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Start Tensor Serve
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Configure AI Endpoint
```bash
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "mistral"
  }'
```

### 3. Ingest Content
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "zim_path": "/path/to/wikipedia.zim",
    "output_name": "zim_db"
  }'
```

### 4. Load Database
```bash
curl -X GET "http://localhost:8000/load?name=zim_db"
```

### 5. Start Chatting
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?"
  }'
```

See [API.md](./API.md) for complete API documentation.

## Project Structure

```
tensor-serve/
├── main.py              # FastAPI application and routes
├── config.py            # Configuration management
├── conversations.py     # Conversation history (SQLite)
├── ai_client.py         # AI endpoint communication
├── embedder.py          # Embedding model (sentence-transformers)
├── vectordb.py          # Vector database (FAISS)
├── ingest.py            # ZIM file processing pipeline
├── chunker.py           # Text chunking algorithm
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── API.md              # API documentation
└── README.md           # This file
```

## Configuration

Settings are stored in `config.json`:

```json
{
  "ai_endpoint": "http://localhost:11434",
  "ai_model": "mistral",
  "context_size": 3,
  "max_conversation_history": 20
}
```

### Settings Explained
- **ai_endpoint**: URL where local LLM is running
- **ai_model**: Name of the model to use
- **context_size**: Number of relevant text chunks to retrieve
- **max_conversation_history**: Maximum messages per conversation

## Data Flow

### Ingestion Pipeline
1. **Extract**: Read articles from ZIM file
2. **Clean**: Remove HTML tags and normalize text
3. **Chunk**: Split text into overlapping chunks
4. **Embed**: Convert chunks to semantic embeddings
5. **Store**: Save FAISS index + text data

### Chat Pipeline
1. **Encode**: Convert user message to embedding
2. **Search**: Find k most similar text chunks
3. **Retrieve**: Get context from vector DB
4. **Prompt**: Build prompt with context + message
5. **Call**: Send to local AI endpoint
6. **Store**: Save message and response to history
7. **Return**: Send response to user

## Performance Tips

- **First load is slow**: Model downloads (~200MB) on first run
- **Large ZIM files**: Can take 10-30 minutes to ingest depending on size
- **Batch processing**: Uses 100-chunk batches to minimize memory usage
- **FAISS indexing**: Creates `.index` and `.pkl` files for fast reloading
- **Context window**: Adjust `context_size` based on your LLM's capabilities

## Common Setups

### With Ollama
```bash
# Install Ollama: https://ollama.ai
# Run model
ollama run mistral

# Configure Tensor Serve
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "mistral"
  }'
```

### With vLLM
```bash
# Install vLLM
pip install vllm

# Run server
python -m vllm.entrypoints.openai.api_server --model mistral-7b-instruct-v0.1

# Configure
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:8000",
    "ai_model": "mistral-7b-instruct-v0.1"
  }'
```

## Troubleshooting

### "DB not loaded" error
- Call `/load` endpoint first: `GET /load?name=zim_db`
- Check that `zim_db.index` and `zim_db.pkl` exist

### "AI endpoint not configured" error
- Call `/config/set-ai-endpoint` to configure
- Verify AI endpoint is running and accessible

### Ingest taking too long
- Large ZIM files (>1GB) are normal
- Check system resources (CPU, RAM, disk)
- Process runs in batches, so you can check progress

### Embedding model not found
- First run downloads the embedding model (~200MB)
- Requires internet connection for download
- Model is cached locally after first run

## API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check server status |
| GET | `/config` | Get current configuration |
| POST | `/config/set-ai-endpoint` | Configure AI model |
| POST | `/ingest` | Process ZIM file |
| GET | `/load` | Load vector database |
| POST | `/search` | Search for context |
| POST | `/chat` | Chat with AI |
| GET | `/conversation/{id}` | Get conversation history |

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Python code passes syntax checks
- API changes documented in API.md
- No breaking changes to existing endpoints
