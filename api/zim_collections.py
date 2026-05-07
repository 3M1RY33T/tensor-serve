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


def _validate_collection_id(collection_id: str):
    path = Path(collection_id)
    if (
        not collection_id
        or path.is_absolute()
        or len(path.parts) != 1
        or collection_id in {".", ".."}
    ):
        raise ValueError("Collection ID must be a single folder name")


def get_collection_path(collection_id: str) -> str:
    """Return the absolute folder path for a collection."""
    _validate_collection_id(collection_id)
    return str(Path(get_zim_source_folder()) / collection_id)


def _collection_metadata(collection_id: str) -> Dict:
    state = load_collections()
    return state.get("collections", {}).get(collection_id, {})


def _title_from_id(collection_id: str) -> str:
    return collection_id.replace("_", " ").replace("-", " ").title()


def _zim_file_entry(path: str) -> Dict:
    zim_path = Path(path)
    return {
        "name": zim_path.name,
        "path": str(zim_path),
        "size": bytes_to_human(zim_path.stat().st_size),
        "installed": True,
        "needs_download": False,
    }


def list_collection_zim_paths(collection_id: str) -> List[str]:
    """Return all .zim files in a collection folder."""
    folder = Path(get_collection_path(collection_id))
    if not folder.is_dir():
        return []
    return sorted(
        str(candidate.absolute())
        for candidate in folder.rglob("*.zim")
        if candidate.is_file()
    )


def _build_collection(collection_id: str, folder: Path, metadata: Dict) -> Dict:
    zim_paths = list_collection_zim_paths(collection_id)
    zim_files = [_zim_file_entry(path) for path in zim_paths]
    return {
        "name": metadata.get("name") or _title_from_id(collection_id),
        "description": metadata.get("description", ""),
        "category": "collection",
        "path": str(folder),
        "zim_files": zim_files,
        "created_at": metadata.get("created_at"),
    }


def get_all_collections() -> Dict:
    """Get all collection folders in the active ZIM source folder."""
    init_zim_folder()
    source = Path(get_zim_source_folder())
    state = load_collections()
    metadata = state.get("collections", {})
    collections = {}

    if source.is_dir():
        for folder in sorted(path for path in source.iterdir() if path.is_dir()):
            collections[folder.name] = _build_collection(
                folder.name, folder, metadata.get(folder.name, {})
            )

    return collections


def get_collection(collection_id: str) -> Optional[Dict]:
    """Get a specific collection folder by ID."""
    try:
        folder = Path(get_collection_path(collection_id))
    except ValueError:
        return None
    if not folder.is_dir():
        return None
    return _build_collection(collection_id, folder, _collection_metadata(collection_id))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collection_folder(collection_id: str) -> Path:
    return Path(get_collection_path(collection_id))


def _source_folder() -> Path:
    return Path(get_zim_source_folder()).resolve()


def _is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def _same_file(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.samefile(path_b)
    except OSError:
        return False


def _unique_source_target(zim_path: Path) -> Path:
    source = _source_folder()
    target = source / zim_path.name
    if not target.exists():
        return target

    if _same_file(target, zim_path):
        return target

    index = 2
    while True:
        candidate = source / f"{zim_path.stem}-{index}{zim_path.suffix}"
        if not candidate.exists():
            return candidate
        if _same_file(candidate, zim_path):
            return candidate
        index += 1


def _canonical_zim_path(zim_path: Path) -> Path:
    """Return a ZIM path under the source folder, copying external files once."""
    if not zim_path.is_file() or zim_path.suffix != ".zim":
        raise ValueError(f"Path must point to an existing .zim file: {zim_path}")

    zim_path = zim_path.resolve()
    if _is_inside(zim_path, _source_folder()):
        return zim_path

    source = _source_folder()
    source.mkdir(parents=True, exist_ok=True)
    target = _unique_source_target(zim_path)
    if not target.exists():
        shutil.copy2(zim_path, target)
    return target.resolve()


def _relative_link_target(source: Path, target_folder: Path) -> str:
    return os.path.relpath(source, start=target_folder)


def _link_zim_into_collection(collection_folder: Path, zim_path: Path) -> str:
    """Reference a ZIM from a collection without duplicating archive bytes."""
    source = _canonical_zim_path(zim_path)
    target = collection_folder / source.name

    if target.resolve(strict=False) == source.resolve(strict=False):
        return str(target)

    if target.exists() or target.is_symlink():
        if _same_file(target, source):
            return str(target)
        raise ValueError(
            f"Collection already contains a different file named {target.name}"
        )

    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(_relative_link_target(source, collection_folder))
        except OSError as e:
            raise ValueError(
                "Could not create a filesystem link for the ZIM file"
            ) from e
    return str(target.resolve())


def create_custom_collection(
    collection_id: str,
    name: Optional[str] = None,
    description: str = "",
    zim_paths: Optional[List[str]] = None,
) -> Dict:
    """Create a collection folder in the active ZIM source folder."""
    _validate_collection_id(collection_id)
    init_zim_folder()
    folder = _collection_folder(collection_id)
    folder.mkdir(parents=True, exist_ok=True)

    for path in zim_paths or []:
        _link_zim_into_collection(folder, Path(path).expanduser().resolve())

    state = load_collections()
    state.setdefault("collections", {})
    existing = state["collections"].get(collection_id, {})
    state["collections"][collection_id] = {
        "name": name or existing.get("name") or _title_from_id(collection_id),
        "description": description,
        "category": "collection",
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
    """Update collection metadata without changing the collection folder name."""
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
    metadata["category"] = "collection"
    metadata["updated_at"] = _now()

    save_collections(state)
    return get_collection(collection_id)


def add_files_to_collection(collection_id: str, zim_paths: List[str]) -> Optional[Dict]:
    """Add one or more .zim files to a collection without duplicating archives."""
    collection = get_collection(collection_id)
    if not collection:
        return None

    folder = _collection_folder(collection_id)
    for path in zim_paths:
        _link_zim_into_collection(folder, Path(path).expanduser().resolve())

    state = load_collections()
    if collection_id in state.get("collections", {}):
        state["collections"][collection_id]["updated_at"] = _now()
        save_collections(state)
    return get_collection(collection_id)


def _candidate_in_collection(collection_folder: Path, candidate: Path) -> Optional[Path]:
    if not candidate.is_absolute():
        candidate = collection_folder / candidate
    candidate = candidate.expanduser().absolute()
    try:
        candidate.relative_to(collection_folder.absolute())
    except ValueError:
        return None
    return candidate


def remove_files_from_collection(
    collection_id: str,
    file_names: Optional[List[str]] = None,
    zim_paths: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Remove ZIM files from a collection folder."""
    collection = get_collection(collection_id)
    if not collection:
        return None

    folder = _collection_folder(collection_id)
    requested = list(file_names or []) + list(zim_paths or [])
    existing_by_name = {
        Path(path).name: Path(path)
        for path in list_collection_zim_paths(collection_id)
    }

    removed = False
    for raw in requested:
        path = existing_by_name.get(raw)
        if path is None:
            path = _candidate_in_collection(folder, Path(raw))
        if path and path.exists() and path.is_file() and path.suffix == ".zim":
            path.unlink()
            removed = True

    if removed:
        state = load_collections()
        if collection_id in state.get("collections", {}):
            state["collections"][collection_id]["updated_at"] = _now()
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
    """Delete collection metadata and its collection folder."""
    state = load_collections()
    existed = collection_id in state.get("collections", {})
    state.get("collections", {}).pop(collection_id, None)

    if state.get("active") == collection_id:
        state["active"] = None

    try:
        folder = _collection_folder(collection_id)
    except ValueError:
        folder = None

    if folder and folder.exists():
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
    """Get local file paths for files in a collection folder."""
    return list_collection_zim_paths(collection_id)


def get_collection_with_installation_status(collection_id: str) -> Optional[Dict]:
    """Get collection details with file status for each ZIM in its folder."""
    return get_collection(collection_id)
