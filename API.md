# Tensor Serve API Documentation

Tensor Serve is a backend service that enables locally-run AI models to work with offline web content (ZIM files from Kiwix). It operates independently from the web GUI and provides a complete API for content ingestion, context retrieval, and AI-powered conversations.

## Quick Start

### 1. Configure AI Endpoint
```bash
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:8000",  # Your local LLM endpoint
    "ai_model": "mistral"                     # Model name
  }'
```

### 2. Ingest ZIM File
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "zim_path": "/path/to/wikipedia.zim",
    "output_name": "zim_db"
  }'
```

### 3. Load Vector Database
```bash
curl -X GET "http://localhost:8000/load?name=zim_db"
```

### 4. Chat with AI
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is photosynthesis?"
  }'
```

## API Endpoints

### Health & Configuration

#### GET `/health`
Check server status and component availability.

**Response:**
```json
{
  "status": "ok",
  "db_loaded": true,
  "ai_configured": true
}
```

#### GET `/config`
Get current configuration.

**Response:**
```json
{
  "ai_endpoint": "http://localhost:8000",
  "ai_model": "mistral",
  "context_size": 3,
  "max_conversation_history": 20
}
```

#### POST `/config/set-ai-endpoint`
Set the local AI model endpoint.

**Request:**
```json
{
  "ai_endpoint": "http://localhost:8000",
  "ai_model": "mistral"
}
```

**Response:**
```json
{
  "status": "configured",
  "ai_endpoint": "http://localhost:8000",
  "ai_model": "mistral"
}
```

### Content Ingestion

#### POST `/ingest`
Process a ZIM file, split into chunks, and create embeddings.

**Request:**
```json
{
  "zim_path": "/path/to/file.zim",
  "output_name": "zim_db"
}
```

**Response:**
```json
{
  "status": "completed",
  "output": "zim_db"
}
```

**Notes:**
- Processes articles in batches for memory efficiency
- Creates `.index` and `.pkl` files with embeddings
- Can take several minutes for large ZIM files

#### GET `/load`
Load a pre-processed vector database.

**Query Parameters:**
- `name` (string): Database name (default: "zim_db")

**Response:**
```json
{
  "status": "loaded",
  "db": "zim_db"
}
```

### Search & Retrieval

#### POST `/search`
Search for relevant context from the vector database.

**Request:**
```json
{
  "query": "photosynthesis",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "photosynthesis",
  "results": [
    "Photosynthesis is a process by which...",
    "The light-dependent reactions occur in...",
    "Chlorophyll absorbs light energy..."
  ]
}
```

### Conversation & Chat

#### POST `/chat`
Send a message and get an AI response with relevant context.

**Request:**
```json
{
  "message": "Explain photosynthesis",
  "conversation_id": "uuid-string"  # Optional, auto-generated if omitted
}
```

**Response:**
```json
{
  "conversation_id": "a1b2c3d4-...",
  "user_message": "Explain photosynthesis",
  "ai_response": "Photosynthesis is a biological process...",
  "context": [
    "Photosynthesis is a process...",
    "The light-dependent reactions...",
    "Chlorophyll absorbs light energy..."
  ]
}
```

#### GET `/conversation/{conversation_id}`
Retrieve full conversation history.

**Response:**
```json
{
  "conversation_id": "a1b2c3d4-...",
  "messages": [
    {
      "role": "user",
      "content": "What is photosynthesis?",
      "context": "...relevant chunks...",
      "created_at": "2024-01-15T10:30:00"
    },
    {
      "role": "assistant",
      "content": "Photosynthesis is...",
      "created_at": "2024-01-15T10:30:05"
    }
  ]
}
```

## Configuration

Configuration is stored in `config.json`:
```json
{
  "ai_endpoint": "http://localhost:8000",
  "ai_model": "mistral",
  "context_size": 3,
  "max_conversation_history": 20
}
```

### Settings

- **ai_endpoint**: URL of local LLM server (required for chat)
- **ai_model**: Model name to use (required for chat)
- **context_size**: Number of context chunks to retrieve (default: 3)
- **max_conversation_history**: Maximum messages to track per conversation (default: 20)

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

## Architecture

- **Embedder**: Uses `sentence-transformers` (all-MiniLM-L6-v2) for semantic embeddings
- **Vector DB**: FAISS index for efficient similarity search
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
- Embeddings are cached in `.index` and `.pkl` files for fast reloading
- Context retrieval is O(1) for similarity search
- Chat responses depend on AI endpoint response time
