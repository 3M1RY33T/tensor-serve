import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from ai_client import AIClient
from config import get_config_value, load_config, set_config_value
from conversations import add_message, create_conversation, get_conversation_history
from embedder import Embedder
from ingest import run_ingestion
from multi_ingest import run_multi_ingest
from presets import (
    create_custom_preset,
    delete_custom_preset,
    get_active_preset,
    get_all_presets,
    get_preset,
    get_preset_with_installation_status,
    init_presets,
    set_active_preset,
)
from vectordb import VectorDB
from zim_downloader import (
    bytes_to_human,
    get_preset_installation_status,
    is_file_installed,
    list_installed_files,
)


class AppState:
    def __init__(self):
        self.embedder = None
        self.db = None
        self.db_loaded = False
        self.ai_client = AIClient()
        self.active_preset = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.embedder = Embedder()
    init_presets()
    app_state.active_preset = get_active_preset()

    # Auto-load the active preset's database if it was already ingested
    if app_state.active_preset:
        preset_id = app_state.active_preset["id"]
        db_name = f"{preset_id}_db"
        if os.path.exists(f"{db_name}.index") and os.path.exists(f"{db_name}.pkl"):
            try:
                app_state.db = VectorDB(dim=384)
                app_state.db.load(db_name)
                app_state.db_loaded = True
                print(f"[startup] Auto-loaded database: {db_name}")
            except Exception as e:
                print(f"[startup] Warning: could not auto-load '{db_name}': {e}")

    yield
    app_state.embedder = None
    app_state.db = None


app = FastAPI(
    lifespan=lifespan,
    title="Tensor Serve",
    description="Backend service for local AI with offline content",
)


class IngestRequest(BaseModel):
    zim_path: str
    output_name: str = "zim_db"


class MultiIngestRequest(BaseModel):
    zim_paths: list
    output_name: str = "combined_db"


class CustomPresetRequest(BaseModel):
    preset_id: str
    name: str
    description: str
    zim_paths: list


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ConfigRequest(BaseModel):
    ai_endpoint: str
    ai_model: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None


class DevdocsInstallRequest(BaseModel):
    file_ids: list = []  # if empty, all uninstalled devdocs entries are queued


class ChatResponse(BaseModel):
    conversation_id: str
    user_message: str
    ai_response: str
    context: list


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "db_loaded": app_state.db_loaded,
        "ai_configured": app_state.ai_client.is_configured(),
        "active_preset": app_state.active_preset["id"]
        if app_state.active_preset
        else None,
    }


@app.get("/config")
def get_config():
    config = load_config()
    return {
        "ai_endpoint": config.get("ai_endpoint"),
        "ai_model": config.get("ai_model"),
        "context_size": config.get("context_size", 3),
        "max_conversation_history": config.get("max_conversation_history", 20),
    }


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


@app.get("/presets")
def list_presets():
    all_presets = get_all_presets()
    return {
        "presets": {
            preset_id: {
                "name": preset["name"],
                "description": preset["description"],
                "category": preset["category"],
                "file_count": len(preset.get("zim_files", [])),
            }
            for preset_id, preset in all_presets.items()
        },
        "active": app_state.active_preset["id"] if app_state.active_preset else None,
    }


@app.get("/presets/{preset_id}")
def get_preset_details(preset_id: str):
    preset = get_preset_with_installation_status(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    response = {
        "id": preset_id,
        "name": preset["name"],
        "description": preset["description"],
        "category": preset["category"],
    }

    if preset.get("category") == "preset":
        response["files"] = preset.get("files_with_status", preset.get("zim_files", []))
    else:
        response["zim_files"] = preset.get("zim_files", [])

    return response


@app.post("/presets/{preset_id}/ingest")
def ingest_preset(preset_id: str, zim_file_indices: list = None):
    preset = get_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    try:
        if preset.get("category") == "preset":
            # For built-in presets, resolve paths from the manifest via
            # zim_downloader so that updated IDs (e.g. devdocs_all) are
            # honoured instead of the stale IDs stored in presets.py.
            from zim_downloader import get_installed_files_for_preset as _get_paths

            zim_paths = _get_paths(preset_id)
        else:
            # Custom presets store absolute paths directly in the definition.
            zim_files = preset.get("zim_files", [])
            if zim_file_indices:
                zim_files = [
                    zim_files[i] for i in zim_file_indices if i < len(zim_files)
                ]
            zim_paths = [f["path"] for f in zim_files if "path" in f]

        if not zim_paths:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No installed ZIM files found for preset '{preset_id}'. "
                    "Install files first with: python manage_zim.py install-preset "
                    f"{preset_id}"
                ),
            )

        db_name = f"{preset_id}_db"
        result = run_multi_ingest(zim_paths, db_name)

        set_active_preset(preset_id)
        app_state.active_preset = {"id": preset_id, "preset": preset}

        return {**result, "preset_id": preset_id, "db_name": db_name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/presets/custom/create")
def create_preset(req: CustomPresetRequest):
    try:
        create_custom_preset(req.preset_id, req.name, req.description, req.zim_paths)
        return {
            "status": "created",
            "preset_id": req.preset_id,
            "name": req.name,
            "file_count": len(req.zim_paths),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/presets/custom/{preset_id}")
def delete_preset(preset_id: str):
    if delete_custom_preset(preset_id):
        if app_state.active_preset and app_state.active_preset["id"] == preset_id:
            app_state.active_preset = None
        return {"status": "deleted", "preset_id": preset_id}
    else:
        raise HTTPException(
            status_code=404, detail="Preset not found or is a built-in preset"
        )


@app.get("/zim/installed")
def list_installed_zim():
    """List all installed ZIM files."""
    installed = list_installed_files()
    return {
        "installed_files": {
            file_id: {
                "title": info["title"],
                "size": info["size"],
                "path": info["path"],
            }
            for file_id, info in installed.items()
        },
        "count": len(installed),
    }


@app.get("/zim/status/{preset_id}")
def get_zim_installation_status(preset_id: str):
    """Get ZIM installation status for a preset."""
    status = get_preset_installation_status(preset_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@app.get("/zim/available")
def list_available_zim():
    """List all available ZIM files for download."""
    from zim_downloader import list_available_files

    available = list_available_files()
    result = {}

    for preset_id, files in available.items():
        result[preset_id] = {
            "files": [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "description": f["description"],
                    "size": f["size"],
                    "installed": is_file_installed(f["id"]),
                }
                for f in files
            ]
        }

    return result


@app.get("/zim/devdocs")
def list_devdocs():
    """
    List all available devdocs entries from the live Kiwix OPDS catalog.

    Each entry includes its installed status. Note that this endpoint makes
    a live network request to Kiwix and may take a moment to respond.
    """
    from zim_downloader import list_devdocs_catalog

    catalog = list_devdocs_catalog()
    total_bytes = sum(e["size_bytes"] for e in catalog)
    installed_count = sum(1 for e in catalog if is_file_installed(e["id"]))
    uninstalled_bytes = sum(
        e["size_bytes"] for e in catalog if not is_file_installed(e["id"])
    )

    return {
        "entries": [
            {
                "id": e["id"],
                "name": e["name"],
                "description": e["description"],
                "size": e["size"],
                "size_bytes": e["size_bytes"],
                "installed": is_file_installed(e["id"]),
            }
            for e in catalog
        ],
        "count": len(catalog),
        "installed_count": installed_count,
        "total_size": bytes_to_human(total_bytes),
        "total_size_bytes": total_bytes,
        "remaining_size": bytes_to_human(uninstalled_bytes),
        "remaining_size_bytes": uninstalled_bytes,
    }


@app.post("/zim/devdocs/install")
def install_devdocs(req: DevdocsInstallRequest, background_tasks: BackgroundTasks):
    """
    Queue devdocs ZIM files for background download.

    - Provide specific ``file_ids`` to install selected entries.
    - Pass an empty ``file_ids`` list to install every uninstalled devdocs entry.

    Downloads run in the background; the response returns immediately with a
    list of queued file IDs so the caller can poll ``GET /zim/installed`` to
    track progress.
    """
    from zim_downloader import download_file as _download_file
    from zim_downloader import list_devdocs_catalog

    catalog = list_devdocs_catalog()
    catalog_map = {e["id"]: e for e in catalog}

    if req.file_ids:
        invalid = [fid for fid in req.file_ids if fid not in catalog_map]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown devdocs file ID(s): {invalid}",
            )
        to_install = [fid for fid in req.file_ids if not is_file_installed(fid)]
        already_installed = [fid for fid in req.file_ids if is_file_installed(fid)]
    else:
        to_install = [e["id"] for e in catalog if not is_file_installed(e["id"])]
        already_installed = [e["id"] for e in catalog if is_file_installed(e["id"])]

    def _bulk_download(file_ids: list):
        for fid in file_ids:
            _download_file(fid, show_progress=False)

    if to_install:
        background_tasks.add_task(_bulk_download, to_install)

    queued_size = sum(
        catalog_map[fid]["size_bytes"] for fid in to_install if fid in catalog_map
    )

    return {
        "status": "started" if to_install else "nothing_to_do",
        "queued": len(to_install),
        "queued_files": to_install,
        "queued_size": bytes_to_human(queued_size),
        "already_installed": len(already_installed),
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        result = run_ingestion(req.zim_path, req.output_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-multiple")
def ingest_multiple(req: MultiIngestRequest):
    try:
        result = run_multi_ingest(req.zim_paths, req.output_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        conversation_id = req.conversation_id or str(uuid4())

        try:
            get_conversation_history(conversation_id)
        except Exception:
            create_conversation(conversation_id)

        context_size = get_config_value("context_size")
        query_embedding = app_state.embedder.encode([req.message])[0]
        context = app_state.db.search(query_embedding, context_size)

        ai_response = app_state.ai_client.chat(req.message, context)

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


@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    try:
        history = get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
