import glob
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional
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
        self.bm25 = None
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
                print(f"[startup] Auto-loaded FAISS database: {db_name}")
            except Exception as e:
                print(f"[startup] Warning: could not auto-load '{db_name}': {e}")

            # Also load BM25 index if present (graceful — not required)
            bm25_path = f"{db_name}.bm25"
            if os.path.exists(bm25_path):
                try:
                    from bm25_index import BM25Index

                    app_state.bm25 = BM25Index()
                    app_state.bm25.load(db_name)
                    print(f"[startup] Auto-loaded BM25 index: {bm25_path}")
                except Exception as e:
                    print(f"[startup] Warning: could not load BM25 index: {e}")

    yield
    app_state.embedder = None
    app_state.db = None
    app_state.bm25 = None


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
    ai_model: Optional[str] = None


class UpdateConfigRequest(BaseModel):
    ai_endpoint: Optional[str] = None
    ai_model: Optional[str] = None
    context_size: Optional[int] = None
    max_conversation_history: Optional[int] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None


class DevdocsInstallRequest(BaseModel):
    file_ids: list = []  # if empty, all uninstalled devdocs entries are queued


class ZimInstallRequest(BaseModel):
    file_ids: list  # one or more Kiwix file IDs to download


class ZimInstallPresetRequest(BaseModel):
    preset_id: str
    file_ids: list = []  # specific IDs from the preset; empty = all uninstalled


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
        "bm25_loaded": app_state.bm25 is not None,
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
        ai_model = req.ai_model
        auto_detected = False
        all_models = []

        if not ai_model:
            from ai_client import AIClient as _AIClient

            all_models = _AIClient.list_models(req.ai_endpoint)
            if not all_models:
                raise HTTPException(
                    status_code=502,
                    detail="Endpoint reachable but returned no models.",
                )
            ai_model = all_models[0]["id"]
            auto_detected = True

        set_config_value("ai_endpoint", req.ai_endpoint)
        set_config_value("ai_model", ai_model)
        app_state.ai_client.update_config(req.ai_endpoint, ai_model)

        response = {
            "status": "configured",
            "ai_endpoint": req.ai_endpoint,
            "ai_model": ai_model,
            "auto_detected": auto_detected,
        }
        if auto_detected and len(all_models) > 1:
            response["available_models"] = [m["id"] for m in all_models]
            response["note"] = (
                "Multiple models found — first one was selected. "
                "Use PATCH /config to switch to a different model."
            )
        return response
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/models")
def list_available_models(endpoint: Optional[str] = None):
    """
    List models available at the AI endpoint.

    Uses the currently configured endpoint by default. Pass an optional
    ``?endpoint=<url>`` query parameter to probe a different URL without
    saving it — useful for previewing models before calling
    ``POST /config/set-ai-endpoint``.
    """
    from ai_client import AIClient as _AIClient

    target = endpoint or app_state.ai_client.endpoint
    if not target:
        raise HTTPException(
            status_code=400,
            detail="No endpoint configured. Provide ?endpoint=<url> or call /config/set-ai-endpoint first.",
        )

    try:
        models = _AIClient.list_models(target)
        return {
            "endpoint": target,
            "models": [m["id"] for m in models],
            "count": len(models),
            "source": models[0]["source"] if models else None,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.patch("/config")
def update_config(req: UpdateConfigRequest):
    """
    Update one or more configuration settings.

    All fields are optional — only the fields you provide will be changed.
    If ai_endpoint or ai_model are updated the live AI client is refreshed
    immediately without requiring a server restart.
    """
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    for key, value in updates.items():
        set_config_value(key, value)

    # Keep the live AI client in sync if connection settings changed
    if "ai_endpoint" in updates or "ai_model" in updates:
        config = load_config()
        app_state.ai_client.update_config(
            config.get("ai_endpoint"),
            config.get("ai_model"),
        )

    config = load_config()
    return {
        "status": "updated",
        "updated_fields": list(updates.keys()),
        "config": {
            "ai_endpoint": config.get("ai_endpoint"),
            "ai_model": config.get("ai_model"),
            "context_size": config.get("context_size", 3),
            "max_conversation_history": config.get("max_conversation_history", 20),
        },
    }


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

        # Load BM25 index if available (enables hybrid search)
        bm25_path = f"{name}.bm25"
        bm25_loaded = False
        if os.path.exists(bm25_path):
            try:
                from bm25_index import BM25Index

                app_state.bm25 = BM25Index()
                app_state.bm25.load(name)
                bm25_loaded = True
            except Exception as e:
                print(f"Warning: could not load BM25 index: {e}")
                app_state.bm25 = None

        return {
            "status": "loaded",
            "db": name,
            "bm25_loaded": bm25_loaded,
            "search_mode": "hybrid" if bm25_loaded else "semantic",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest):
    if not app_state.db_loaded or app_state.db is None:
        raise HTTPException(status_code=400, detail="DB not loaded. Call /load first.")

    try:
        from hybrid_search import hybrid_search

        query_embedding = app_state.embedder.encode([req.query])[0]
        results = hybrid_search(
            query=req.query,
            query_embedding=query_embedding,
            vectordb=app_state.db,
            bm25_index=app_state.bm25,
            top_k=req.top_k,
        )
        return {
            "query": req.query,
            "results": results,
            "search_mode": "hybrid" if app_state.bm25 is not None else "semantic",
        }
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

        from hybrid_search import hybrid_search

        context_size = get_config_value("context_size") or 3
        query_embedding = app_state.embedder.encode([req.message])[0]
        context = hybrid_search(
            query=req.message,
            query_embedding=query_embedding,
            vectordb=app_state.db,
            bm25_index=app_state.bm25,
            top_k=context_size,
        )

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
        limit = get_config_value("max_conversation_history") or 20
        history = get_conversation_history(conversation_id, limit=limit)
        return {"conversation_id": conversation_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clean")
def clean_working_files():
    """
    Remove all working files generated by ingestion and chat sessions.

    Deletes:
    - Vector database index files (*.index)
    - Vector database text stores (*.pkl)
    - BM25 keyword index files (*.bm25)
    - Conversation history (conversations.db)
    - Python bytecode cache (__pycache__/)

    Preserves:
    - presets.json  (preset configuration)
    - config.json   (AI endpoint settings)
    - zim_manifest.json (installed ZIM record)
    - zim_files/    (downloaded ZIM files)
    """
    removed = []
    errors = []

    # Remove vector DB index and text-store files
    for pattern in ("*.index", "*.pkl", "*.bm25"):
        for path in sorted(glob.glob(pattern)):
            try:
                os.remove(path)
                removed.append(path)
            except Exception as e:
                errors.append({"file": path, "error": str(e)})

    # Remove conversation history database
    conv_db = "conversations.db"
    if os.path.exists(conv_db):
        try:
            os.remove(conv_db)
            removed.append(conv_db)
        except Exception as e:
            errors.append({"file": conv_db, "error": str(e)})

    # Remove __pycache__ directory
    if os.path.isdir("__pycache__"):
        try:
            shutil.rmtree("__pycache__")
            removed.append("__pycache__/")
        except Exception as e:
            errors.append({"file": "__pycache__/", "error": str(e)})

    # Reset in-memory vector DB state so stale handles are not used
    app_state.db = None
    app_state.db_loaded = False
    app_state.bm25 = None

    return {
        "status": "cleaned",
        "removed_files": removed,
        "removed_count": len(removed),
        "errors": errors,
    }


@app.post("/zim/install")
def install_zim(req: ZimInstallRequest, background_tasks: BackgroundTasks):
    """
    Queue one or more ZIM files for background download.

    Provide a list of Kiwix file IDs (e.g. ``["devdocs_en_python", "devdocs_en_rust"]``).
    Use ``GET /zim/progress`` to track download status after queuing.
    """
    from zim_downloader import download_file as _dl

    queued = []
    already_installed = []

    for file_id in req.file_ids:
        if is_file_installed(file_id):
            already_installed.append(file_id)
        else:
            background_tasks.add_task(_dl, file_id, False)
            queued.append(file_id)

    return {
        "status": "queued" if queued else "nothing_to_do",
        "queued": queued,
        "queued_count": len(queued),
        "already_installed": already_installed,
    }


@app.post("/zim/install-preset")
def install_preset_zim(req: ZimInstallPresetRequest, background_tasks: BackgroundTasks):
    """
    Queue ZIM downloads for a preset's files.

    Pass specific ``file_ids`` to install only a subset of the preset's files.
    Leave ``file_ids`` empty to queue every uninstalled file in the preset.

    This is the API equivalent of:
    ``python manage_zim.py install-preset <preset>``
    """
    from zim_downloader import PRESET_FILES
    from zim_downloader import download_file as _dl

    if req.preset_id not in PRESET_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{req.preset_id}' not found. "
            f"Valid options: {', '.join(PRESET_FILES)}",
        )

    preset_files = PRESET_FILES[req.preset_id]

    if req.file_ids:
        valid_ids = {f["id"] for f in preset_files}
        invalid = [fid for fid in req.file_ids if fid not in valid_ids]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"File IDs not part of preset '{req.preset_id}': {invalid}",
            )
        to_check = [f for f in preset_files if f["id"] in req.file_ids]
    else:
        to_check = preset_files

    queued = []
    already_installed = []

    for f in to_check:
        if is_file_installed(f["id"]):
            already_installed.append(f["id"])
        else:
            background_tasks.add_task(_dl, f["id"], False)
            queued.append(f["id"])

    return {
        "status": "queued" if queued else "nothing_to_do",
        "preset_id": req.preset_id,
        "queued": queued,
        "queued_count": len(queued),
        "already_installed": already_installed,
    }


@app.delete("/zim/uninstall/{file_id}")
def uninstall_zim(file_id: str):
    """
    Remove an installed ZIM file from disk and from the manifest.

    This is the API equivalent of:
    ``python manage_zim.py uninstall <file_id>``
    """
    from zim_downloader import uninstall_file as _uninstall

    if not is_file_installed(file_id):
        raise HTTPException(
            status_code=404, detail=f"File '{file_id}' is not installed"
        )

    success = _uninstall(file_id)
    if success:
        return {"status": "uninstalled", "file_id": file_id}
    raise HTTPException(status_code=500, detail=f"Failed to uninstall '{file_id}'")


@app.get("/zim/progress")
def get_zim_progress():
    """
    Get download progress for all active and recently completed downloads.

    Poll this endpoint while downloads are running to show a progress bar
    in a web GUI. Each entry contains at minimum a ``status`` field
    (``downloading`` | ``completed`` | ``partial`` | ``error`` | ``already_installed``).

    Active file downloads also include:
    - ``percent``          – 0.0–100.0
    - ``downloaded``       – human-readable bytes received (e.g. ``"210.3 MB"``)
    - ``total``            – human-readable total size
    - ``downloaded_bytes`` – raw bytes received
    - ``total_bytes``      – raw total bytes

    The ``devdocs_all`` bundle entry additionally includes:
    - ``completed_files``  – number of individual entries finished
    - ``total_files``      – total number of entries being installed
    """
    from zim_downloader import get_download_progress

    progress = get_download_progress()
    active = sum(1 for p in progress.values() if p.get("status") == "downloading")
    return {
        "active_downloads": active,
        "total_tracked": len(progress),
        "downloads": progress,
    }


@app.get("/zim/progress/{file_id}")
def get_zim_file_progress(file_id: str):
    """
    Get download progress for a specific file by its Kiwix ID.

    Returns 404 if no download has been started for this file in the
    current server session.
    """
    from zim_downloader import get_file_progress

    progress = get_file_progress(file_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"No download record for '{file_id}' in this session. "
            "Start a download with POST /zim/install first.",
        )
    return {"file_id": file_id, **progress}
