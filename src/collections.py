import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.zim_downloader import get_installed_files_for_collection, is_file_installed

COLLECTIONS_FILE = "collections.json"
LEGACY_COLLECTIONS_FILE = "pre" + "sets.json"
LEGACY_COLLECTIONS_KEY = "pre" + "sets"
LEGACY_COLLECTION_CATEGORY = "pre" + "set"

# Built-in collections with their ZIM file identifiers
COLLECTIONS = {
    "research": {
        "name": "Research",
        "description": "Academic and encyclopedic content for research",
        "category": "collection",
        "file_ids": [
            "wikipedia_en_all",
            "wikisource_en_all",
            "wikinews_en_all",
            "mdwiki_en_all",
            "wikivoyage_en_europe",
        ],
        "zim_files": [
            {
                "id": "wikipedia_en_all",
                "name": "Wikipedia (English)",
                "description": "The free encyclopedia – text only, no images (nopic flavour)",
                "size": "~52GB",
            },
            {
                "id": "wikisource_en_all",
                "name": "Wikisource",
                "description": "Free library of source texts and original documents – text only, no images",
                "size": "~12GB",
            },
            {
                "id": "wikinews_en_all",
                "name": "Wikinews",
                "description": "Free-content news source – text only, no images",
                "size": "~100MB",
            },
            {
                "id": "mdwiki_en_all",
                "name": "MDWiki Medical Encyclopedia",
                "description": "Healthcare articles curated by WikiProjectMed (maxi – no text-only version available)",
                "size": "~2.1GB",
            },
            {
                "id": "wikivoyage_en_europe",
                "name": "Wikivoyage – Europe",
                "description": "Travel guide for European destinations – text only, no images (nopic flavour)",
                "size": "~67MB",
            },
        ],
    },
    "learn": {
        "name": "Learn",
        "description": "Educational and textbook content",
        "category": "collection",
        "file_ids": ["wikiversity_en_all", "wikibooks_en_all"],
        "zim_files": [
            {
                "id": "wikiversity_en_all",
                "name": "Wikiversity",
                "description": "Free learning resources, courses and research – text only, no images",
                "size": "~2GB",
            },
            {
                "id": "wikibooks_en_all",
                "name": "Wikibooks",
                "description": "Free open-content textbooks and manuals – text only, no images",
                "size": "~3GB",
            },
        ],
    },
    "literature": {
        "name": "Literature",
        "description": "Books and literary works",
        "category": "collection",
        "file_ids": ["gutenberg_en_all", "wikibooks_en_all", "gutenberg_en_lcc-pk"],
        "zim_files": [
            {
                "id": "gutenberg_en_all",
                "name": "Project Gutenberg",
                "description": "70,000+ free public-domain ebooks",
                "size": "~60GB",
            },
            {
                "id": "wikibooks_en_all",
                "name": "Wikibooks",
                "description": "Free textbooks and manuals – text only, no images",
                "size": "~3GB",
            },
            {
                "id": "gutenberg_en_lcc-pk",
                "name": "Project Gutenberg – Indo-Iranian Languages",
                "description": "Public-domain texts in Indo-Iranian languages and literatures",
                "size": "~80MB",
            },
        ],
    },
    "coding": {
        "name": "Coding",
        "description": "Developer documentation and resources",
        "category": "collection",
        "file_ids": [
            "stackoverflow.com_en_all",
            "robotics.stackexchange.com_en_all",
            "devdocs_all",
        ],
        "zim_files": [
            {
                "id": "stackoverflow.com_en_all",
                "name": "Stack Overflow",
                "description": "Q&A for professional and enthusiast programmers (no text-only version available)",
                "size": "~80GB",
            },
            {
                "id": "robotics.stackexchange.com_en_all",
                "name": "Robotics Stack Exchange",
                "description": "Q&A for everything related to robotics",
                "size": "~233MB",
            },
            {
                "id": "devdocs_all",
                "name": "All DevDocs (231+ entries)",
                "description": "Complete DevDocs collection – every available API doc pooled into one install (~588 MB total)",
                "size": "~588MB",
            },
        ],
    },
}


def _default_state() -> Dict:
    return {"active": None, "collections": COLLECTIONS, "custom": {}}


def _normalize_state(state: Dict) -> Dict:
    """Return collection state using the current on-disk schema."""
    if LEGACY_COLLECTIONS_KEY in state and "collections" not in state:
        state["collections"] = state.pop(LEGACY_COLLECTIONS_KEY)

    state.setdefault("active", None)
    state.setdefault("collections", COLLECTIONS)
    state.setdefault("custom", {})

    for group in ("collections", "custom"):
        for collection in state[group].values():
            if collection.get("category") == LEGACY_COLLECTION_CATEGORY:
                collection["category"] = "collection"

    return state


def init_collections():
    """Initialize collections file with built-in collections if it doesn't exist."""
    # Migrate old runtime files to collections.json if needed.
    if not os.path.exists(COLLECTIONS_FILE) and os.path.exists(LEGACY_COLLECTIONS_FILE):
        with open(LEGACY_COLLECTIONS_FILE, "r") as f:
            save_collections(_normalize_state(json.load(f)))
    if not os.path.exists(COLLECTIONS_FILE) and os.path.exists("tunings.json"):
        os.rename("tunings.json", COLLECTIONS_FILE)
    if not os.path.exists(COLLECTIONS_FILE):
        save_collections(_default_state())
    return load_collections()


def load_collections() -> Dict:
    """Load collections configuration."""
    if os.path.exists(COLLECTIONS_FILE):
        try:
            with open(COLLECTIONS_FILE, "r") as f:
                return _normalize_state(json.load(f))
        except Exception:
            return _default_state()
    return _default_state()


def save_collections(collections: Dict):
    """Save collections configuration."""
    with open(COLLECTIONS_FILE, "w") as f:
        json.dump(collections, f, indent=2)


def get_all_collections() -> Dict:
    """Get all available collections (built-in + custom)."""
    collections = load_collections()
    return {**collections["collections"], **collections["custom"]}


def get_collection(collection_id: str) -> Optional[Dict]:
    """Get a specific collection by ID."""
    return get_all_collections().get(collection_id)


def create_custom_collection(
    collection_id: str, name: str, description: str, zim_paths: List[str]
) -> Dict:
    """Create a custom collection."""
    collections = load_collections()
    custom_collection = {
        "name": name,
        "description": description,
        "category": "custom",
        "zim_files": [
            {"path": path, "name": os.path.basename(path)} for path in zim_paths
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    collections["custom"][collection_id] = custom_collection
    save_collections(collections)
    return custom_collection


def set_active_collection(collection_id: str) -> bool:
    """Set the active collection."""
    collections = load_collections()
    if collection_id in get_all_collections():
        collections["active"] = collection_id
        save_collections(collections)
        return True
    return False


def get_active_collection() -> Optional[Dict]:
    """Get the currently active collection."""
    collections = load_collections()
    active_id = collections.get("active")
    if active_id:
        return {"id": active_id, "collection": get_collection(active_id)}
    return None


def delete_custom_collection(collection_id: str) -> bool:
    """Delete a custom collection."""
    collections = load_collections()
    if collection_id in collections["custom"]:
        del collections["custom"][collection_id]
        if collections.get("active") == collection_id:
            collections["active"] = None
        save_collections(collections)
        return True
    return False


def list_collection_files(collection_id: str) -> Optional[List[Dict]]:
    """List all ZIM files for a collection."""
    collection = get_collection(collection_id)
    return collection.get("zim_files", []) if collection else None


def get_installed_paths_for_collection(collection_id: str) -> List[str]:
    """Get local file paths for installed files in a collection."""
    return get_installed_files_for_collection(collection_id)


def get_collection_with_installation_status(collection_id: str) -> Optional[Dict]:
    """Get collection details with installation status for each file."""
    collection = get_collection(collection_id)
    if not collection:
        return None

    if collection.get("category") == "collection":
        result = collection.copy()
        result["files_with_status"] = []
        for file_info in collection.get("zim_files", []):
            file_id = file_info.get("id")
            installed = is_file_installed(file_id)
            result["files_with_status"].append(
                {**file_info, "installed": installed, "needs_download": not installed}
            )
        return result

    return collection
