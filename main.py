from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from uuid import uuid4
from ingest import run_ingestion
from embedder import Embedder
from vectordb import VectorDB
from config import load_config, set_config_value, get_config_value
from conversations import create_conversation, add_message, get_conversation_history
from ai_client import AIClient

# -------- Global State --------
class AppState:
    def __init__(self):
        self.embedder = None
        self.db = None
        self.db_loaded = False
        self.ai_client = AIClient()

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize embedder model
    app_state.embedder = Embedder()
    yield
    # Shutdown: cleanup
    app_state.embedder = None
    app_state.db = None


app = FastAPI(
    lifespan=lifespan,
    title="Tensor Serve",
    description="Backend service for local AI with offline content",
)


# -------- Request Models --------
class IngestRequest(BaseModel):
    zim_path: str
    output_name: str = "zim_db"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ConfigRequest(BaseModel):
    ai_endpoint: str
    ai_model: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None


class ChatResponse(BaseModel):
    conversation_id: str
    user_message: str
    ai_response: str
    context: list


# -------- GET: Health Check --------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "db_loaded": app_state.db_loaded,
        "ai_configured": app_state.ai_client.is_configured(),
    }


# -------- GET: Config Status --------
@app.get("/config")
def get_config():
    config = load_config()
    return {
        "ai_endpoint": config.get("ai_endpoint"),
        "ai_model": config.get("ai_model"),
        "context_size": config.get("context_size", 3),
        "max_conversation_history": config.get("max_conversation_history", 20),
    }


# -------- POST: Set AI Endpoint --------
@app.post("/config/set-ai-endpoint")
def set_ai_endpoint(req: ConfigRequest):
    try:
        set_config_value("ai_endpoint", req.ai_endpoint)
        set_config_value("ai_model", req.ai_model)
        app_state.ai_client.update_config(req.ai_endpoint, req.ai_model)

        return {
            "status": "configured",
            "ai_endpoint": req.ai_endpoint,
            "ai_model": req.ai_model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- POST: Ingest ZIM File --------
@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        result = run_ingestion(req.zim_path, req.output_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- GET: Load Vector Database --------
@app.get("/load")
def load_db(name: str = "zim_db"):
    try:
        app_state.db = VectorDB(dim=384)
        app_state.db.load(name)
        app_state.db_loaded = True
        return {"status": "loaded", "db": name}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- POST: Search Context --------
@app.post("/search")
def search(req: SearchRequest):
    if not app_state.db_loaded or app_state.db is None:
        raise HTTPException(status_code=400, detail="DB not loaded. Call /load first.")

    try:
        query_embedding = app_state.embedder.encode([req.query])[0]
        results = app_state.db.search(query_embedding, req.top_k)

        return {"query": req.query, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- POST: Chat with AI --------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not app_state.ai_client.is_configured():
        raise HTTPException(
            status_code=400,
            detail="AI endpoint not configured. Call /config/set-ai-endpoint first.",
        )

    if not app_state.db_loaded or app_state.db is None:
        raise HTTPException(status_code=400, detail="DB not loaded. Call /load first.")

    try:
        # Generate conversation ID if not provided
        conversation_id = req.conversation_id or str(uuid4())

        # Ensure conversation exists
        try:
            get_conversation_history(conversation_id)
        except Exception:
            create_conversation(conversation_id)

        # Get context from vector database
        context_size = get_config_value("context_size")
        query_embedding = app_state.embedder.encode([req.message])[0]
        context = app_state.db.search(query_embedding, context_size)

        # Get AI response with context
        ai_response = app_state.ai_client.chat(req.message, context)

        # Store messages in conversation history
        context_str = "\n".join(context) if context else None
        add_message(conversation_id, "user", req.message, context_str)
        add_message(conversation_id, "assistant", ai_response)

        return ChatResponse(
            conversation_id=conversation_id,
            user_message=req.message,
            ai_response=ai_response,
            context=context,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- GET: Conversation History --------
@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    try:
        history = get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))