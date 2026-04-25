import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from zim_downloader import get_installed_files_for_tuning

TUNINGS_FILE = "tunings.json"

# Preset tunings with their ZIM file identifiers
PRESET_TUNINGS = {
    "research": {
        "name": "Research",
        "description": "Academic and encyclopedic content for research",
        "category": "preset",
        "file_ids": ["wikipedia_en", "wikisource_en", "wikinews_en"],
        "zim_files": [
            {
                "id": "wikipedia_en",
                "name": "Wikipedia",
                "description": "The free encyclopedia",
                "size": "~20GB",
            },
            {
                "id": "wikisource_en",
                "name": "Wikisource",
                "description": "Free library of texts",
                "size": "~5GB",
            },
            {
                "id": "wikinews_en",
                "name": "Wikinews",
                "description": "Free news source",
                "size": "~2GB",
            },
        ],
    },
    "learn": {
        "name": "Learn",
        "description": "Educational and textbook content",
        "category": "preset",
        "file_ids": ["libretexts", "wikiversity_en"],
        "zim_files": [
            {
                "id": "libretexts",
                "name": "LibreTexts",
                "description": "Open-source textbooks and courses",
                "size": "~15GB",
            },
            {
                "id": "wikiversity_en",
                "name": "Wikiversity",
                "description": "Free learning environment",
                "size": "~3GB",
            },
        ],
    },
    "literature": {
        "name": "Literature",
        "description": "Books and literary works",
        "category": "preset",
        "file_ids": ["gutenberg", "wikibooks_en"],
        "zim_files": [
            {
                "id": "gutenberg",
                "name": "Project Gutenberg",
                "description": "Over 70,000 free ebooks",
                "size": "~25GB",
            },
            {
                "id": "wikibooks_en",
                "name": "Wikibooks",
                "description": "Free textbooks and manuals",
                "size": "~3GB",
            },
        ],
    },
    "coding": {
        "name": "Coding",
        "description": "Developer documentation and resources",
        "category": "preset",
        "file_ids": ["devdocs", "stack_exchange"],
        "zim_files": [
            {
                "id": "devdocs",
                "name": "DevDocs",
                "description": "API documentation aggregator",
                "size": "~5GB",
            },
            {
                "id": "stack_exchange",
                "name": "Stack Exchange",
                "description": "Q&A from Stack Overflow and related",
                "size": "~10GB",
            },
        ],
    },
}


def init_tunings():
    """Initialize tunings file with presets if it doesn't exist."""
    if not os.path.exists(TUNINGS_FILE):
        tunings = {
            "active": None,
            "presets": PRESET_TUNINGS,
            "custom": {},
        }
        save_tunings(tunings)
    return load_tunings()


def load_tunings() -> Dict:
    """Load tunings configuration."""
    if os.path.exists(TUNINGS_FILE):
        try:
            with open(TUNINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "active": None,
                "presets": PRESET_TUNINGS,
                "custom": {},
            }
    return {
        "active": None,
        "presets": PRESET_TUNINGS,
        "custom": {},
    }


def save_tunings(tunings: Dict):
    """Save tunings configuration."""
    with open(TUNINGS_FILE, "w") as f:
        json.dump(tunings, f, indent=2)


def get_all_tunings() -> Dict:
    """Get all available tunings (presets + custom)."""
    tunings = load_tunings()
    return {
        **tunings["presets"],
        **tunings["custom"],
    }


def get_tuning(tuning_id: str) -> Optional[Dict]:
    """Get a specific tuning by ID."""
    all_tunings = get_all_tunings()
    return all_tunings.get(tuning_id)


def create_custom_tuning(
    tuning_id: str, name: str, description: str, zim_paths: List[str]
) -> Dict:
    """Create a custom tuning."""
    tunings = load_tunings()

    custom_tuning = {
        "name": name,
        "description": description,
        "category": "custom",
        "zim_files": [
            {"path": path, "name": os.path.basename(path)} for path in zim_paths
        ],
        "created_at": datetime.utcnow().isoformat(),
    }

    tunings["custom"][tuning_id] = custom_tuning
    save_tunings(tunings)

    return custom_tuning


def set_active_tuning(tuning_id: str) -> bool:
    """Set the active tuning."""
    tunings = load_tunings()
    if tuning_id in get_all_tunings():
        tunings["active"] = tuning_id
        save_tunings(tunings)
        return True
    return False


def get_active_tuning() -> Optional[Dict]:
    """Get the currently active tuning."""
    tunings = load_tunings()
    active_id = tunings.get("active")
    if active_id:
        return {
            "id": active_id,
            "tuning": get_tuning(active_id),
        }
    return None


def delete_custom_tuning(tuning_id: str) -> bool:
    """Delete a custom tuning."""
    tunings = load_tunings()
    if tuning_id in tunings["custom"]:
        del tunings["custom"][tuning_id]
        if tunings.get("active") == tuning_id:
            tunings["active"] = None
        save_tunings(tunings)
        return True
    return False


def list_tuning_files(tuning_id: str) -> Optional[List[Dict]]:
    """List all ZIM files for a tuning."""
    tuning = get_tuning(tuning_id)
    if tuning:
        return tuning.get("zim_files", [])
    return None


def get_installed_paths_for_tuning(tuning_id: str) -> List[str]:
    """Get local file paths for installed files in a tuning."""
    return get_installed_files_for_tuning(tuning_id)


def get_tuning_with_installation_status(tuning_id: str) -> Optional[Dict]:
    """Get tuning details with installation status for each file."""
    tuning = get_tuning(tuning_id)
    if not tuning:
        return None

    if tuning.get("category") == "preset":
        installed_paths = get_installed_files_for_tuning(tuning_id)
        
        # Enhance with installation status
        result = tuning.copy()
        result["files_with_status"] = []
        
        for file_info in tuning.get("zim_files", []):
            file_id = file_info.get("id")
            installed = any(path.endswith(f"{file_id}.zim") for path in installed_paths)
            result["files_with_status"].append({
                **file_info,
                "installed": installed,
                "needs_download": not installed,
            })
        
        return result
    
    return tuning
