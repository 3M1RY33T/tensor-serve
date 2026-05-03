import glob
import json
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from src.ai_client import AIClient
from src.cache import query_cache
from src.config import get_config_value, load_config, set_config_value
from src.embedder import Embedder
from src.ingest import run_ingestion
from src.multi_ingest import run_multi_ingest
from src.collections import (
    add_files_to_collection,
    create_custom_collection,
    delete_custom_collection,
    get_active_collection,
    get_all_collections,
    get_collection,
    get_collection_path,
    get_collection_with_installation_status,
    init_collections,
    remove_files_from_collection,
    set_active_collection,
    update_collection,
)
from src.vectordb import VectorDB
from src.zim_downloader import (
    ZIM_FOLDER,
    bytes_to_human,
    clear_zim_source_folder,
    get_zim_source_folder,
    has_custom_zim_source_folder,
    is_file_installed,
    list_installed_files,
    normalize_category_id,
    register_zim_file,
    resolve_zim_inputs,
    set_zim_source_folder,
)


class AppState:
    def __init__(self):
        self.embedder = None
        self.db = None
        self.bm25 = None
        self.db_loaded = False
        self.db_name = None
        self.ai_client = AIClient()
        self.active_collection = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.embedder = Embedder()
    init_collections()
    app_state.active_collection = get_active_collection()

    # Auto-load the active collection's database if it was already ingested
    if app_state.active_collection:
        category_id = app_state.active_collection["id"]
        db_name = f"{category_id}_db"
        if os.path.exists(f"{db_name}.index") and os.path.exists(f"{db_name}.pkl"):
            try:
                app_state.db = VectorDB(dim=384)
                app_state.db.load(db_name)
                app_state.db_loaded = True
                app_state.db_name = db_name
                print(f"[startup] Auto-loaded FAISS database: {db_name}")
            except Exception as e:
                print(f"[startup] Warning: could not auto-load '{db_name}': {e}")

            # Also load BM25 index if present (graceful — not required)
            bm25_path = f"{db_name}.bm25"
            if os.path.exists(bm25_path):
                try:
                    from src.bm25_index import BM25Index

                    app_state.bm25 = BM25Index()
                    app_state.bm25.load(db_name)
                    print(f"[startup] Auto-loaded BM25 index: {bm25_path}")
                except Exception as e:
                    print(f"[startup] Warning: could not load BM25 index: {e}")

    yield
    app_state.embedder = None
    app_state.db = None
    app_state.bm25 = None
    app_state.db_loaded = False
    app_state.db_name = None


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


class CustomCollectionRequest(BaseModel):
    collection_id: str
    name: Optional[str] = None
    description: str = ""
    zim_paths: list = []


class CollectionUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CollectionFilesRequest(BaseModel):
    zim_paths: list = []
    file_names: list = []


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ConfigRequest(BaseModel):
    ai_endpoint: str
    ai_model: Optional[str] = None
    ai_provider: Optional[str] = None
    ai_api_key: Optional[str] = None
    ai_api_key_header: Optional[str] = None
    ai_api_key_prefix: Optional[str] = None
    ai_extra_headers: Optional[dict] = None


class UpdateConfigRequest(BaseModel):
    ai_provider: Optional[str] = None
    ai_endpoint: Optional[str] = None
    ai_model: Optional[str] = None
    ai_api_key: Optional[str] = None
    ai_api_key_header: Optional[str] = None
    ai_api_key_prefix: Optional[str] = None
    ai_extra_headers: Optional[dict] = None
    context_size: Optional[int] = None
    zim_source_folder: Optional[str] = None


class DevdocsInstallRequest(BaseModel):
    file_ids: list = []  # if empty, all uninstalled devdocs entries are queued


class ZimInstallRequest(BaseModel):
    file_ids: list  # one or more Kiwix file IDs to download


class ZimInstallCategoryRequest(BaseModel):
    category_id: str
    file_ids: list = []  # specific IDs from the category; empty = all uninstalled


class ZimSourceFolderRequest(BaseModel):
    path: str


class ZimRegisterRequest(BaseModel):
    path: str
    file_id: Optional[str] = None
    title: Optional[str] = None


class ZimSourceIngestRequest(BaseModel):
    output_name: str = "zim_source_db"


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "db_loaded": app_state.db_loaded,
        "bm25_loaded": app_state.bm25 is not None,
        "ai_configured": app_state.ai_client.is_configured(),
        "active_collection": app_state.active_collection["id"]
        if app_state.active_collection
        else None,
    }


@app.get("/cache/stats")
def cache_stats():
    """Get query cache statistics."""
    return query_cache.get_stats()


@app.post("/cache/clear")
def clear_cache():
    """Clear all cached queries and embeddings."""
    query_cache.clear()
    return {"status": "cleared"}


@app.get("/config")
def get_config():
    config = load_config()
    return {
        "ai_provider": config.get("ai_provider", "openai-compatible"),
        "ai_endpoint": config.get("ai_endpoint"),
        "ai_model": config.get("ai_model"),
        "ai_api_key_configured": bool(config.get("ai_api_key")),
        "ai_api_key_header": config.get("ai_api_key_header", "Authorization"),
        "ai_api_key_prefix": config.get("ai_api_key_prefix", "Bearer"),
        "ai_extra_headers": {
            key: "<configured>" for key in (config.get("ai_extra_headers") or {})
        },
        "context_size": config.get("context_size", 3),
        "zim_source_folder": config.get("zim_source_folder"),
    }


@app.post("/config/set-ai-endpoint")
def set_ai_endpoint(req: ConfigRequest):
    try:
        ai_model = req.ai_model
        auto_detected = False
        all_models = []
        provider = req.ai_provider or "openai-compatible"
        api_key_header = req.ai_api_key_header or "Authorization"
        api_key_prefix = (
            "Bearer" if req.ai_api_key_prefix is None else req.ai_api_key_prefix
        )
        extra_headers = req.ai_extra_headers or {}

        if not ai_model:
            from src.ai_client import AIClient as _AIClient

            all_models = _AIClient.list_models(
                req.ai_endpoint,
                req.ai_api_key,
                api_key_header,
                api_key_prefix,
                extra_headers,
            )
            if not all_models:
                raise HTTPException(
                    status_code=502,
                    detail="Endpoint reachable but returned no models.",
                )
            ai_model = all_models[0]["id"]
            auto_detected = True

        set_config_value("ai_provider", provider)
        set_config_value("ai_endpoint", req.ai_endpoint)
        set_config_value("ai_model", ai_model)
        set_config_value("ai_api_key", req.ai_api_key)
        set_config_value("ai_api_key_header", api_key_header)
        set_config_value("ai_api_key_prefix", api_key_prefix)
        set_config_value("ai_extra_headers", extra_headers)
        app_state.ai_client.update_config(
            req.ai_endpoint,
            ai_model,
            provider,
            req.ai_api_key,
            api_key_header,
            api_key_prefix,
            extra_headers,
        )

        response = {
            "status": "configured",
            "ai_provider": provider,
            "ai_endpoint": req.ai_endpoint,
            "ai_model": ai_model,
            "ai_api_key_configured": bool(req.ai_api_key),
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
    from src.ai_client import AIClient as _AIClient

    config = load_config()
    target = endpoint or app_state.ai_client.endpoint
    if not target:
        raise HTTPException(
            status_code=400,
            detail="No endpoint configured. Provide ?endpoint=<url> or call /config/set-ai-endpoint first.",
        )

    try:
        use_configured_auth = not endpoint or target == config.get("ai_endpoint")
        models = _AIClient.list_models(
            target,
            config.get("ai_api_key") if use_configured_auth else None,
            config.get("ai_api_key_header", "Authorization"),
            config.get("ai_api_key_prefix", "Bearer"),
            config.get("ai_extra_headers", {}) if use_configured_auth else {},
        )
        return {
            "endpoint": target,
            "models": [m["id"] for m in models],
            "count": len(models),
            "source": models[0]["source"] if models else None,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/config/local-ai/detect")
def detect_local_ai_endpoints():
    """Detect common local OpenAI-compatible AI runtimes."""
    from src.ai_client import AIClient as _AIClient

    return {"endpoints": _AIClient.detect_local_endpoints()}


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
        if key == "zim_source_folder":
            try:
                if value:
                    set_zim_source_folder(value)
                else:
                    clear_zim_source_folder()
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            set_config_value(key, value)

    # Keep the live AI client in sync if connection settings changed
    if any(key.startswith("ai_") for key in updates):
        config = load_config()
        app_state.ai_client.update_config(
            config.get("ai_endpoint"),
            config.get("ai_model"),
            config.get("ai_provider", "openai-compatible"),
            config.get("ai_api_key"),
            config.get("ai_api_key_header", "Authorization"),
            config.get("ai_api_key_prefix", "Bearer"),
            config.get("ai_extra_headers", {}),
        )

    config = load_config()
    return {
        "status": "updated",
        "updated_fields": list(updates.keys()),
        "config": {
            "ai_provider": config.get("ai_provider", "openai-compatible"),
            "ai_endpoint": config.get("ai_endpoint"),
            "ai_model": config.get("ai_model"),
            "ai_api_key_configured": bool(config.get("ai_api_key")),
            "ai_api_key_header": config.get("ai_api_key_header", "Authorization"),
            "ai_api_key_prefix": config.get("ai_api_key_prefix", "Bearer"),
            "ai_extra_headers": {
                key: "<configured>" for key in (config.get("ai_extra_headers") or {})
            },
            "context_size": config.get("context_size", 3),
            "zim_source_folder": config.get("zim_source_folder"),
        },
    }


@app.get("/collections")
def list_collections():
    all_collections = get_all_collections()
    return {
        "collections": {
            collection_id: {
                "name": collection["name"],
                "description": collection["description"],
                "category": collection["category"],
                "path": collection["path"],
                "file_count": len(collection.get("zim_files", [])),
            }
            for collection_id, collection in all_collections.items()
        },
        "active": (
            app_state.active_collection["id"] if app_state.active_collection else None
        ),
    }


@app.get("/collections/{collection_id}")
def get_collection_details(collection_id: str):
    collection = get_collection_with_installation_status(collection_id)
    if not collection:
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection_id}' not found"
        )

    response = {
        "id": collection_id,
        "name": collection["name"],
        "description": collection["description"],
        "category": collection["category"],
        "path": collection["path"],
        "file_count": len(collection.get("zim_files", [])),
    }

    if collection.get("category") == "collection":
        response["files"] = collection.get("zim_files", [])
    else:
        response["zim_files"] = collection.get("zim_files", [])

    return response


@app.get("/collections/{collection_id}/files")
def get_collection_files(collection_id: str):
    collection = get_collection(collection_id)
    if not collection:
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection_id}' not found"
        )
    return {"collection_id": collection_id, "files": collection.get("zim_files", [])}


@app.post("/collections/{collection_id}/ingest")
def ingest_collection(collection_id: str, zim_file_indices: list = None):
    collection = get_collection(collection_id)
    if not collection:
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection_id}' not found"
        )

    try:
        zim_files = collection.get("zim_files", [])
        if zim_file_indices:
            zim_files = [zim_files[i] for i in zim_file_indices if i < len(zim_files)]
            zim_paths = [f["path"] for f in zim_files if "path" in f]
        else:
            zim_paths = resolve_zim_inputs([get_collection_path(collection_id)])

        if not zim_paths:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No ZIM files found in collection '{collection_id}'. "
                    f"Add .zim files to: {get_collection_path(collection_id)}"
                ),
            )

        db_name = f"{collection_id}_db"
        result = run_multi_ingest(zim_paths, db_name)

        set_active_collection(collection_id)
        app_state.active_collection = {"id": collection_id, "collection": collection}

        return {**result, "collection_id": collection_id, "db_name": db_name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
def create_collection(req: CustomCollectionRequest):
    try:
        collection = create_custom_collection(
            req.collection_id, req.name, req.description, req.zim_paths
        )
        return {
            "status": "created",
            "collection_id": req.collection_id,
            "name": collection["name"],
            "path": collection["path"],
            "file_count": len(collection.get("zim_files", [])),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/custom/create")
def create_collection_legacy(req: CustomCollectionRequest):
    return create_collection(req)


@app.patch("/collections/{collection_id}")
def update_collection_details(collection_id: str, req: CollectionUpdateRequest):
    if req.name is None and req.description is None:
        raise HTTPException(
            status_code=400, detail="Provide at least one field to update"
        )
    try:
        collection = update_collection(collection_id, req.name, req.description)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection_id}' not found"
            )
        return {
            "status": "updated",
            "collection_id": collection_id,
            "name": collection["name"],
            "description": collection["description"],
            "path": collection["path"],
            "file_count": len(collection.get("zim_files", [])),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/{collection_id}/files")
def add_collection_files(collection_id: str, req: CollectionFilesRequest):
    if not req.zim_paths:
        raise HTTPException(status_code=400, detail="Provide at least one zim_path")
    try:
        collection = add_files_to_collection(collection_id, req.zim_paths)
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection_id}' not found"
            )
        return {
            "status": "updated",
            "collection_id": collection_id,
            "files": collection.get("zim_files", []),
            "file_count": len(collection.get("zim_files", [])),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_id}/files")
def delete_collection_files(collection_id: str, req: CollectionFilesRequest):
    if not req.file_names and not req.zim_paths:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one file_name or zim_path to delete",
        )
    try:
        collection = remove_files_from_collection(
            collection_id, req.file_names, req.zim_paths
        )
        if not collection:
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection_id}' not found"
            )
        return {
            "status": "updated",
            "collection_id": collection_id,
            "files": collection.get("zim_files", []),
            "file_count": len(collection.get("zim_files", [])),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_id}")
def delete_collection(collection_id: str):
    try:
        deleted = delete_custom_collection(collection_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if deleted:
        if (
            app_state.active_collection
            and app_state.active_collection["id"] == collection_id
        ):
            app_state.active_collection = None
        return {"status": "deleted", "collection_id": collection_id}
    else:
        raise HTTPException(status_code=404, detail="Collection not found")


@app.delete("/collections/custom/{collection_id}")
def delete_collection_legacy(collection_id: str):
    return delete_collection(collection_id)


@app.get("/zim/source-folder")
def get_zim_source_folder_details():
    """Show the folder used for ZIM downloads and scans."""
    return {
        "path": get_zim_source_folder(),
        "custom": has_custom_zim_source_folder(),
        "default": os.path.abspath(ZIM_FOLDER),
    }


@app.put("/zim/source-folder")
def update_zim_source_folder(req: ZimSourceFolderRequest):
    """Point Tensor Serve at a folder that already contains local ZIM files."""
    try:
        path = set_zim_source_folder(req.path)
        return {
            "status": "updated",
            "path": path,
            "custom": True,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/zim/source-folder")
def reset_zim_source_folder():
    """Reset ZIM storage to the default repository-local zim_files folder."""
    return {
        "status": "reset",
        "path": clear_zim_source_folder(),
        "custom": False,
    }


@app.post("/zim/register")
def register_existing_zim(req: ZimRegisterRequest):
    """Register one existing .zim path in the manifest without downloading it."""
    try:
        info = register_zim_file(req.path, req.file_id, req.title)
        return {
            "status": "registered",
            "file_id": info["id"],
            "file": info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/zim/source-folder/ingest")
def ingest_zim_source_folder(req: ZimSourceIngestRequest):
    """Ingest every .zim file under the active ZIM source folder."""
    zim_paths = resolve_zim_inputs([get_zim_source_folder()])
    if not zim_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No ZIM files found in source folder: {get_zim_source_folder()}",
        )
    try:
        result = run_multi_ingest(zim_paths, req.output_name)
        return {**result, "db_name": req.output_name, "file_count": len(zim_paths)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/zim/status/{collection_id}")
def get_zim_installation_status(collection_id: str):
    """Get ZIM file status for a collection folder."""
    collection = get_collection_with_installation_status(collection_id)
    if not collection:
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection_id}' not found"
        )
    return {"collection": collection_id, "files": collection.get("zim_files", [])}


@app.get("/zim/available")
def list_available_zim():
    """List all available ZIM files for download."""
    from src.zim_downloader import list_available_files

    available = list_available_files()
    result = {}

    for category_id, files in available.items():
        result[category_id] = {
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
    from src.zim_downloader import list_devdocs_catalog

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
    from src.zim_downloader import download_file as _download_file
    from src.zim_downloader import list_devdocs_catalog

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
        zim_paths = resolve_zim_inputs([req.zim_path])
        if not zim_paths:
            raise HTTPException(
                status_code=400,
                detail=f"No ZIM files found for path: {req.zim_path}",
            )
        if len(zim_paths) == 1:
            return run_ingestion(zim_paths[0], req.output_name)
        return run_multi_ingest(zim_paths, req.output_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-multiple")
def ingest_multiple(req: MultiIngestRequest):
    try:
        zim_paths = resolve_zim_inputs(req.zim_paths)
        if not zim_paths:
            raise HTTPException(
                status_code=400,
                detail="No ZIM files found for the provided paths",
            )
        result = run_multi_ingest(zim_paths, req.output_name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load")
def load_db(name: str = "zim_db"):
    try:
        app_state.db = VectorDB(dim=384)
        app_state.db.load(name)
        app_state.db_loaded = True
        app_state.db_name = name

        # Load BM25 index if available (enables hybrid search)
        bm25_path = f"{name}.bm25"
        bm25_loaded = False
        if os.path.exists(bm25_path):
            try:
                from src.bm25_index import BM25Index

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
        from src.hybrid_search import hybrid_search
        from src.query_analyzer import QueryAnalyzer

        relevance_threshold = get_config_value("relevance_threshold") or 0.05
        reranker_enabled = get_config_value("reranker_enabled")
        if reranker_enabled is None:
            reranker_enabled = False
        
        search_mode = QueryAnalyzer.select_search_mode(req.query)
        
        # Check cache first
        cached_results = query_cache.get_search_result(req.query, search_mode, req.top_k)
        if cached_results is not None:
            results = cached_results
        else:
            cached_embedding = query_cache.get_embedding(req.query)
            if cached_embedding is not None:
                query_embedding = cached_embedding
            else:
                query_embedding = app_state.embedder.encode([req.query])[0]
                query_cache.cache_embedding(req.query, query_embedding)
            
            results = hybrid_search(
                query=req.query,
                query_embedding=query_embedding,
                vectordb=app_state.db,
                bm25_index=app_state.bm25,
                top_k=req.top_k,
                candidate_k=max(req.top_k * 3, 10),
                relevance_threshold=relevance_threshold,
                search_mode=search_mode,
            )
            
            # Re-rank results if enabled
            if reranker_enabled and results:
                from src.reranker import rerank_results
                results = rerank_results(
                    query=req.query,
                    documents=results,
                    top_k=req.top_k,
                    reranker_enabled=True,
                )
            
            # Cache search results
            query_cache.cache_search_result(req.query, search_mode, req.top_k, results)
        
        return {
            "query": req.query,
            "results": results,
            "search_mode": search_mode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _require_ai_endpoint() -> str:
    """Return configured upstream AI endpoint or raise a client-facing error."""
    if not app_state.ai_client.is_configured():
        raise HTTPException(
            status_code=400,
            detail="AI endpoint not configured. Call /config/set-ai-endpoint first.",
        )
    return app_state.ai_client.endpoint.rstrip("/")


def _forward_request_headers(request: Request) -> dict:
    """Forward useful client headers while letting requests recalculate transport headers."""
    excluded = {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "accept-encoding",
    }
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in excluded
    }


def _forward_response_headers(response: requests.Response) -> dict:
    """Forward safe upstream response headers through FastAPI."""
    excluded = {
        "content-length",
        "connection",
        "transfer-encoding",
        "content-encoding",
        "content-type",
    }
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in excluded
    }


def _upstream_url(path: str, request: Request) -> str:
    from src.ai_client import AIClient as _AIClient

    upstream = _AIClient.endpoint_url(_require_ai_endpoint(), path)
    if request.url.query:
        upstream = f"{upstream}?{request.url.query}"
    return upstream


async def _proxy_ai_request(
    request: Request,
    path: str,
    json_payload: Optional[dict] = None,
    stream: bool = False,
    response_suffix: Optional[str] = None,
):
    """
    Forward an HTTP request to the configured upstream AI server.

    If json_payload is provided, it replaces the original body. Otherwise the
    request body is forwarded unchanged.
    """
    url = _upstream_url(path, request)
    headers = _forward_request_headers(request)
    headers.update(app_state.ai_client.auth_headers())

    try:
        if stream:
            upstream = requests.request(
                request.method,
                url,
                headers=headers,
                json=json_payload,
                data=None if json_payload is not None else await request.body(),
                stream=True,
                timeout=None,
            )

            def body_stream():
                try:
                    for chunk in upstream.iter_content(chunk_size=None):
                        if chunk:
                            if response_suffix and chunk.strip() == b"data: [DONE]":
                                yield _streaming_chat_suffix_chunk(response_suffix)
                            yield chunk
                finally:
                    upstream.close()

            return StreamingResponse(
                body_stream(),
                status_code=upstream.status_code,
                headers=_forward_response_headers(upstream),
                media_type=upstream.headers.get("content-type"),
            )

        upstream = requests.request(
            request.method,
            url,
            headers=headers,
            json=json_payload,
            data=None if json_payload is not None else await request.body(),
            timeout=60,
        )
        content = upstream.content
        content_type = upstream.headers.get("content-type", "")
        if response_suffix and "application/json" in content_type:
            content = _chat_response_with_suffix(content, response_suffix)

        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=_forward_response_headers(upstream),
            media_type=upstream.headers.get("content-type"),
        )

    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"AI endpoint error: {str(e)}")


def _last_user_text(messages: list) -> Optional[str]:
    """Return the last string user message content, if one is present."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _context_for_query(query: str) -> list:
    """Retrieve optional RAG context for an OpenAI-compatible chat request."""
    if not app_state.db_loaded or app_state.db is None:
        return []

    from src.hybrid_search import hybrid_search
    from src.query_analyzer import QueryAnalyzer

    query_analysis_enabled = get_config_value("query_analysis_enabled")
    if query_analysis_enabled is None:
        query_analysis_enabled = True

    if query_analysis_enabled:
        needs_rag, reason = QueryAnalyzer.needs_rag(query)
    else:
        needs_rag = True
        reason = "query_analysis_disabled"

    if not needs_rag:
        return []

    context_size = get_config_value("context_size") or 3
    relevance_threshold = get_config_value("relevance_threshold") or 0.05
    reranker_enabled = get_config_value("reranker_enabled")
    if reranker_enabled is None:
        reranker_enabled = False

    search_mode = QueryAnalyzer.select_search_mode(query)
    cached_chunks = query_cache.get_search_result(query, search_mode, context_size)
    if cached_chunks is not None:
        return cached_chunks

    cached_embedding = query_cache.get_embedding(query)
    if cached_embedding is not None:
        query_embedding = cached_embedding
    else:
        query_embedding = app_state.embedder.encode([query])[0]
        query_cache.cache_embedding(query, query_embedding)

    context_chunks = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        vectordb=app_state.db,
        bm25_index=app_state.bm25,
        top_k=context_size,
        candidate_k=max(context_size * 3, 10),
        relevance_threshold=relevance_threshold,
        search_mode=search_mode,
    )

    if reranker_enabled and context_chunks:
        from src.reranker import rerank_results

        context_chunks = rerank_results(
            query=query,
            documents=context_chunks,
            top_k=context_size,
            reranker_enabled=True,
        )

    query_cache.cache_search_result(query, search_mode, context_size, context_chunks)
    return context_chunks


def _payload_with_context(payload: dict, context_chunks: list) -> dict:
    """Return a chat payload with retrieved context injected as a system message."""
    if not context_chunks:
        return payload

    context_text = "\n\n".join([f"- {chunk}" for chunk in context_chunks])
    system_message = {
        "role": "system",
        "content": f"Use the following context to answer questions:\n\n{context_text}",
    }

    updated = dict(payload)
    updated["messages"] = [system_message, *payload.get("messages", [])]
    return updated


def _pop_show_resources(payload: dict) -> bool:
    value = payload.pop("tensor_show_resources", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _metadata_for_chunks(context_chunks: list) -> list:
    """Return chunk metadata for retrieved chunks when the loaded DB has it."""
    db = app_state.db
    if db is None or not getattr(db, "metadata", None):
        return []

    index_by_text = {}
    for idx, text in enumerate(getattr(db, "texts", [])):
        index_by_text.setdefault(text, idx)

    metadata = []
    for chunk in context_chunks:
        idx = index_by_text.get(chunk)
        if idx is None or idx >= len(db.metadata):
            continue
        metadata.append(db.metadata[idx] or {})
    return metadata


def _readable_source_title(metadata: dict) -> Optional[str]:
    title = metadata.get("zim_title")
    if title:
        return str(title)

    zim_path = metadata.get("zim_path")
    if zim_path:
        return os.path.splitext(os.path.basename(str(zim_path)))[0]
    return None


def _resource_attribution(context_chunks: list) -> str:
    """Build the response footer that names retrieved ZIM sources and local DB."""
    if not context_chunks:
        return ""

    lines = []
    source_titles = []
    seen_sources = set()
    for metadata in _metadata_for_chunks(context_chunks):
        title = _readable_source_title(metadata)
        if title and title not in seen_sources:
            seen_sources.add(title)
            source_titles.append(title)

    if source_titles:
        lines.append(f"Read from {', '.join(source_titles)}")
    if app_state.db_name:
        lines.append(f"Enhanced by {app_state.db_name}")

    return "\n".join(lines)


def _chat_response_with_suffix(content: bytes, suffix: str) -> bytes:
    """Append attribution text to non-streaming OpenAI chat responses."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return content

    choices = data.get("choices")
    if not isinstance(choices, list):
        return content

    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        text = message.get("content")
        if isinstance(text, str):
            message["content"] = f"{text}\n\n{suffix}"

    return json.dumps(data).encode("utf-8")


def _streaming_chat_suffix_chunk(suffix: str) -> bytes:
    """Return an OpenAI-compatible streaming delta for attribution text."""
    data = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": f"\n\n{suffix}"}}],
    }
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


@app.get("/v1/models")
async def list_v1_models(request: Request):
    """
    Proxy model discovery to the configured upstream AI server.
    """
    return await _proxy_ai_request(request, "v1/models")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy OpenAI-compatible chat completions, injecting ZIM context when available.
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="JSON body must be an object",
        )

    messages = payload.get("messages", [])
    if messages is not None and not isinstance(messages, list):
        raise HTTPException(
            status_code=400, detail="'messages' must be a list when provided"
        )

    user_message = _last_user_text(messages)
    context_chunks = _context_for_query(user_message) if user_message else []
    show_resources = _pop_show_resources(payload)
    if app_state.ai_client.model:
        payload["model"] = app_state.ai_client.model
    payload = _payload_with_context(payload, context_chunks)
    response_suffix = (
        _resource_attribution(context_chunks) if show_resources else ""
    )

    return await _proxy_ai_request(
        request,
        "v1/chat/completions",
        json_payload=payload,
        stream=bool(payload.get("stream")),
        response_suffix=response_suffix or None,
    )


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_v1(path: str, request: Request):
    """Proxy any other OpenAI-compatible endpoint to the upstream AI server."""
    return await _proxy_ai_request(request, f"v1/{path}")


@app.post("/clean")
def clean_working_files():
    """
    Remove all working files generated by ingestion.

    Deletes:
    - Vector database index files (*.index)
    - Vector database text stores (*.pkl)
    - BM25 keyword index files (*.bm25)
    - Python bytecode cache (__pycache__/)

    Preserves:
    - collections.json  (collection configuration)
    - config.json   (AI endpoint settings)
    - zim_manifest.json (installed ZIM record)
    - configured ZIM source folder files
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
    app_state.db_name = None
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
    from src.zim_downloader import download_file as _dl

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


@app.post("/zim/install-category")
def install_category_zim(
    req: ZimInstallCategoryRequest, background_tasks: BackgroundTasks
):
    """
    Queue ZIM downloads for a curated category's files.

    Pass specific ``file_ids`` to install only a subset of the category's files.
    Leave ``file_ids`` empty to queue every uninstalled file in the category.

    This is the API equivalent of:
    ``python -m src.manage_zim install-category <category>``
    """
    from src.zim_downloader import CATEGORY_FILES
    from src.zim_downloader import download_file as _dl

    category_id = normalize_category_id(req.category_id)
    if not category_id:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{req.category_id}' not found. "
            f"Valid options: {', '.join(CATEGORY_FILES)}",
        )

    category_files = CATEGORY_FILES[category_id]

    if req.file_ids:
        valid_ids = {f["id"] for f in category_files}
        invalid = [fid for fid in req.file_ids if fid not in valid_ids]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"File IDs not part of category '{category_id}': {invalid}",
            )
        to_check = [f for f in category_files if f["id"] in req.file_ids]
    else:
        to_check = category_files

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
        "category": category_id,
        "queued": queued,
        "queued_count": len(queued),
        "already_installed": already_installed,
    }


@app.delete("/zim/uninstall/{file_id}")
def uninstall_zim(file_id: str):
    """
    Remove an installed ZIM file from disk and from the manifest.

    This is the API equivalent of:
    ``python -m src.manage_zim uninstall <file_id>``
    """
    from src.zim_downloader import uninstall_file as _uninstall

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
    from src.zim_downloader import get_download_progress

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
    from src.zim_downloader import get_file_progress

    progress = get_file_progress(file_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"No download record for '{file_id}' in this session. "
            "Start a download with POST /zim/install first.",
        )
    return {"file_id": file_id, **progress}
