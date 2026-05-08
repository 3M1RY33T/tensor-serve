import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from api.zim_downloader import (
    bytes_to_human,
    get_zim_source_folder,
    init_zim_folder,
    resolve_zim_inputs,
)

COLLECTIONS_FILE = "collections.json"
LEGACY_COLLECTIONS_FILE = "pre" + "sets.json"
LEGACY_COLLECTIONS_KEY = "pre" + "sets"
LEGACY_CUSTOM_KEY = "custom"
LEGACY_COLLECTION_CATEGORY = "pre" + "set"


def _default_state() -> Dict:
    return {"active": None, "collections": {}}


def _normalize_state(state: Dict) -> Dict:
    """Return collection metadata using the current on-disk schema."""
    if LEGACY_COLLECTIONS_KEY in state and "collections" not in state:
        state["collections"] = state.pop(LEGACY_COLLECTIONS_KEY)

    state.setdefault("active", None)
    state.setdefault("collections", {})

    custom = state.pop(LEGACY_CUSTOM_KEY, {})
    for collection_id, collection in custom.items():
        state["collections"].setdefault(collection_id, collection)

    for collection in state["collections"].values():
        if collection.get("category") == LEGACY_COLLECTION_CATEGORY:
            collection["category"] = "collection"
        collection.setdefault("zim_paths", [])

    return {
        "active": state.get("active"),
        "collections": state.get("collections", {}),
    }


def init_collections():
    """Initialize collection metadata if it doesn't exist."""
    if not os.path.exists(COLLECTIONS_FILE) and os.path.exists(LEGACY_COLLECTIONS_FILE):
        with open(LEGACY_COLLECTIONS_FILE, "r") as f:
            save_collections(_normalize_state(json.load(f)))
    if not os.path.exists(COLLECTIONS_FILE) and os.path.exists("tunings.json"):
        os.rename("tunings.json", COLLECTIONS_FILE)
    if not os.path.exists(COLLECTIONS_FILE):
        save_collections(_default_state())
    return load_collections()


def load_collections() -> Dict:
    """Load collection metadata."""
    if os.path.exists(COLLECTIONS_FILE):
        try:
            with open(COLLECTIONS_FILE, "r") as f:
                return _normalize_state(json.load(f))
        except Exception:
            return _default_state()
    return _default_state()


def save_collections(collections: Dict):
    """Save collection metadata."""
    with open(COLLECTIONS_FILE, "w") as f:
        json.dump(_normalize_state(collections), f, indent=2)


def reset_collections(delete_folders: bool = True) -> Dict:
    """Reset collection metadata and optionally remove legacy collection folders."""
    init_zim_folder()
    state = load_collections()
    source = Path(get_zim_source_folder()).resolve()
    removed_folders = []
    errors = []

    if delete_folders and source.is_dir():
        for collection_id in sorted(state.get("collections", {})):
            metadata = state.get("collections", {}).get(collection_id, {})
            if not _uses_legacy_folder(metadata):
                continue
            try:
                _validate_collection_id(collection_id)
            except ValueError:
                continue
            folder = source / collection_id
            if not folder.exists():
                continue
            try:
                folder.resolve().relative_to(source)
                if folder.is_symlink():
                    folder.unlink()
                else:
                    shutil.rmtree(folder)
                removed_folders.append(str(folder))
            except Exception as exc:
                errors.append({"folder": str(folder), "error": str(exc)})

    save_collections(_default_state())
    return {
        "status": "reset",
        "collections_file": COLLECTIONS_FILE,
        "folders_removed": removed_folders,
        "folders_removed_count": len(removed_folders),
        "errors": errors,
    }


def _validate_collection_id(collection_id: str):
    path = Path(collection_id)
    if (
        not collection_id
        or path.is_absolute()
        or len(path.parts) != 1
        or collection_id in {".", ".."}
    ):
        raise ValueError("Collection ID must be a single path segment")


def get_collection_path(collection_id: str) -> str:
    """Return the legacy folder path for a collection."""
    _validate_collection_id(collection_id)
    return str(Path(get_zim_source_folder()) / collection_id)


def _collection_metadata(collection_id: str) -> Dict:
    state = load_collections()
    return state.get("collections", {}).get(collection_id, {})


def _title_from_id(collection_id: str) -> str:
    return collection_id.replace("_", " ").replace("-", " ").title()


def _zim_file_entry(path: str) -> Dict:
    zim_path = Path(path)
    installed = zim_path.is_file()
    return {
        "name": zim_path.name,
        "path": str(zim_path),
        "size": bytes_to_human(zim_path.stat().st_size) if installed else None,
        "installed": installed,
        "needs_download": not installed,
    }


def _normalized_zim_path(path: str) -> str:
    zim_path = Path(path).expanduser()
    if not zim_path.is_absolute():
        zim_path = Path.cwd() / zim_path
    return str(zim_path.resolve(strict=False))


def _dedupe_zim_paths(paths: List[str]) -> List[str]:
    resolved = []
    seen = set()
    for path in paths:
        zim_path = Path(path).expanduser()
        if zim_path.suffix != ".zim":
            continue
        normalized = _normalized_zim_path(str(zim_path))
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(normalized)
    return resolved


def _merge_zim_paths(existing_paths: List[str], added_paths: List[str]) -> List[str]:
    return _dedupe_zim_paths(list(existing_paths or []) + list(added_paths or []))


def _uses_legacy_folder(metadata: Dict) -> bool:
    return metadata.get("storage") != "metadata"


def _legacy_collection_folder(collection_id: str) -> Path:
    return Path(get_collection_path(collection_id))


def list_collection_zim_paths(collection_id: str) -> List[str]:
    """Return all existing .zim files referenced by a collection."""
    metadata = _collection_metadata(collection_id)
    zim_paths = [
        path
        for path in _dedupe_zim_paths(metadata.get("zim_paths", []))
        if Path(path).is_file()
    ]

    legacy_folder = _legacy_collection_folder(collection_id)
    if _uses_legacy_folder(metadata) and legacy_folder.is_dir():
        zim_paths = _merge_zim_paths(
            zim_paths,
            sorted(
                str(candidate.resolve())
                for candidate in legacy_folder.rglob("*.zim")
                if candidate.is_file()
            ),
        )

    return zim_paths


def _build_collection(collection_id: str, folder: Optional[Path], metadata: Dict) -> Dict:
    referenced_paths = _dedupe_zim_paths(metadata.get("zim_paths", []))
    uses_legacy_folder = _uses_legacy_folder(metadata)
    if uses_legacy_folder and folder and folder.is_dir():
        referenced_paths = _merge_zim_paths(
            referenced_paths,
            sorted(
                str(candidate.resolve())
                for candidate in folder.rglob("*.zim")
                if candidate.is_file()
            ),
        )
    zim_files = [_zim_file_entry(path) for path in referenced_paths]
    return {
        "name": metadata.get("name") or _title_from_id(collection_id),
        "description": metadata.get("description", ""),
        "category": "collection",
        "path": str(folder) if uses_legacy_folder and folder and folder.is_dir() else None,
        "storage": "folder" if uses_legacy_folder and folder and folder.is_dir() else "metadata",
        "zim_files": zim_files,
        "created_at": metadata.get("created_at"),
    }


def get_all_collections() -> Dict:
    """Get all metadata collections, plus legacy folder-backed collections."""
    init_zim_folder()
    source = Path(get_zim_source_folder())
    state = load_collections()
    metadata = state.get("collections", {})
    collections = {
        collection_id: _build_collection(
            collection_id,
            _legacy_collection_folder(collection_id),
            collection_metadata,
        )
        for collection_id, collection_metadata in metadata.items()
    }

    if source.is_dir():
        for folder in sorted(path for path in source.iterdir() if path.is_dir()):
            if folder.name in collections:
                continue
            collections[folder.name] = _build_collection(
                folder.name, folder, metadata.get(folder.name, {})
            )

    return collections


def get_collection(collection_id: str) -> Optional[Dict]:
    """Get a collection by ID."""
    try:
        folder = _legacy_collection_folder(collection_id)
    except ValueError:
        return None
    metadata = _collection_metadata(collection_id)
    if not metadata and not folder.is_dir():
        return None
    return _build_collection(collection_id, folder, metadata)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collection_folder(collection_id: str) -> Path:
    return Path(get_collection_path(collection_id))


def create_custom_collection(
    collection_id: str,
    name: Optional[str] = None,
    description: str = "",
    zim_paths: Optional[List[str]] = None,
) -> Dict:
    """Create a metadata-backed collection without copying or linking files."""
    _validate_collection_id(collection_id)
    init_zim_folder()
    resolved_zim_paths = resolve_zim_inputs(zim_paths or [])
    if zim_paths and not resolved_zim_paths:
        raise ValueError("No .zim files found for the provided paths")

    state = load_collections()
    state.setdefault("collections", {})
    existing = state["collections"].get(collection_id, {})
    existing_paths = existing.get("zim_paths", []) or list_collection_zim_paths(
        collection_id
    )
    state["collections"][collection_id] = {
        "name": name or existing.get("name") or _title_from_id(collection_id),
        "description": description,
        "category": "collection",
        "storage": "metadata",
        "zim_paths": _merge_zim_paths(existing_paths, resolved_zim_paths),
        "created_at": existing.get("created_at") or _now(),
        "updated_at": _now(),
    }
    save_collections(state)
    return get_collection(collection_id)


def update_collection(
    collection_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Dict]:
    """Update collection metadata without changing the collection ID."""
    collection = get_collection(collection_id)
    if not collection:
        return None

    state = load_collections()
    state.setdefault("collections", {})
    metadata = state["collections"].setdefault(
        collection_id,
        {
            "name": collection["name"],
            "description": collection["description"],
            "category": "collection",
            "created_at": collection.get("created_at") or _now(),
        },
    )

    if name is not None:
        metadata["name"] = name
    if description is not None:
        metadata["description"] = description
    if not metadata.get("zim_paths"):
        metadata["zim_paths"] = list_collection_zim_paths(collection_id)
    metadata["category"] = "collection"
    metadata["storage"] = "metadata"
    metadata["updated_at"] = _now()

    save_collections(state)
    return get_collection(collection_id)


def add_files_to_collection(collection_id: str, zim_paths: List[str]) -> Optional[Dict]:
    """Add .zim file references or directories of .zim file references."""
    collection = get_collection(collection_id)
    if not collection:
        return None

    resolved_zim_paths = resolve_zim_inputs(zim_paths)
    if not resolved_zim_paths:
        raise ValueError("No .zim files found for the provided paths")

    state = load_collections()
    state.setdefault("collections", {})
    metadata = state["collections"].setdefault(
        collection_id,
        {
            "name": collection["name"],
            "description": collection["description"],
            "category": "collection",
            "storage": "metadata",
            "created_at": collection.get("created_at") or _now(),
            "zim_paths": list_collection_zim_paths(collection_id),
        },
    )
    metadata["zim_paths"] = _merge_zim_paths(
        metadata.get("zim_paths", []), resolved_zim_paths
    )
    metadata["category"] = "collection"
    metadata["storage"] = "metadata"
    metadata["updated_at"] = _now()
    save_collections(state)
    return get_collection(collection_id)


def remove_files_from_collection(
    collection_id: str,
    file_names: Optional[List[str]] = None,
    zim_paths: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Remove ZIM file references from a collection."""
    collection = get_collection(collection_id)
    if not collection:
        return None

    requested = list(file_names or []) + list(zim_paths or [])
    requested_names = {Path(raw).name for raw in requested if raw}
    requested_paths = {_normalized_zim_path(raw) for raw in requested if raw}

    state = load_collections()
    state.setdefault("collections", {})
    metadata = state["collections"].setdefault(
        collection_id,
        {
            "name": collection["name"],
            "description": collection["description"],
            "category": "collection",
            "storage": "metadata",
            "created_at": collection.get("created_at") or _now(),
            "zim_paths": list_collection_zim_paths(collection_id),
        },
    )

    existing_paths = _dedupe_zim_paths(metadata.get("zim_paths", []))
    remaining_paths = [
        path
        for path in existing_paths
        if Path(path).name not in requested_names and path not in requested_paths
    ]

    if len(remaining_paths) != len(existing_paths):
        metadata["zim_paths"] = remaining_paths
        metadata["category"] = "collection"
        metadata["storage"] = "metadata"
        metadata["updated_at"] = _now()
        save_collections(state)

    return get_collection(collection_id)


def set_active_collection(collection_id: str) -> bool:
    """Set the active collection."""
    if get_collection(collection_id):
        state = load_collections()
        state["active"] = collection_id
        save_collections(state)
        return True
    return False


def get_active_collection() -> Optional[Dict]:
    """Get the currently active collection."""
    state = load_collections()
    active_id = state.get("active")
    collection = get_collection(active_id) if active_id else None
    if collection:
        return {"id": active_id, "collection": collection}
    return None


def delete_custom_collection(collection_id: str) -> bool:
    """Delete collection metadata and any legacy folder for that collection."""
    state = load_collections()
    metadata = state.get("collections", {}).get(collection_id, {})
    existed = collection_id in state.get("collections", {})
    state.get("collections", {}).pop(collection_id, None)

    if state.get("active") == collection_id:
        state["active"] = None

    try:
        folder = _collection_folder(collection_id)
    except ValueError:
        folder = None

    if folder and folder.exists() and _uses_legacy_folder(metadata):
        existed = True
        source = Path(get_zim_source_folder()).resolve()
        try:
            folder.resolve().relative_to(source)
        except ValueError:
            raise ValueError("Collection folder is outside the active ZIM source folder")
        shutil.rmtree(folder)

    save_collections(state)
    return existed


def list_collection_files(collection_id: str) -> Optional[List[Dict]]:
    """List all ZIM files for a collection."""
    collection = get_collection(collection_id)
    return collection.get("zim_files", []) if collection else None


def get_installed_paths_for_collection(collection_id: str) -> List[str]:
    """Get local file paths for files referenced by a collection."""
    return list_collection_zim_paths(collection_id)


def get_collection_with_installation_status(collection_id: str) -> Optional[Dict]:
    """Get collection details with file status for each referenced ZIM."""
    return get_collection(collection_id)
