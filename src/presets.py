import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from src.zim_downloader import get_installed_files_for_preset

PRESETS_FILE = "presets.json"

# Built-in presets with their ZIM file identifiers
PRESETS = {
    "research": {
        "name": "Research",
        "description": "Academic and encyclopedic content for research",
        "category": "preset",
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
        "category": "preset",
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
        "category": "preset",
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
        "category": "preset",
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


def init_presets():
    """Initialize presets file with built-in presets if it doesn't exist."""
    # Migrate old tunings.json → presets.json if needed
    if not os.path.exists(PRESETS_FILE) and os.path.exists("tunings.json"):
        os.rename("tunings.json", PRESETS_FILE)
    if not os.path.exists(PRESETS_FILE):
        save_presets({"active": None, "presets": PRESETS, "custom": {}})
    return load_presets()


def load_presets() -> Dict:
    """Load presets configuration."""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"active": None, "presets": PRESETS, "custom": {}}
    return {"active": None, "presets": PRESETS, "custom": {}}


def save_presets(presets: Dict):
    """Save presets configuration."""
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def get_all_presets() -> Dict:
    """Get all available presets (built-in + custom)."""
    presets = load_presets()
    return {**presets["presets"], **presets["custom"]}


def get_preset(preset_id: str) -> Optional[Dict]:
    """Get a specific preset by ID."""
    return get_all_presets().get(preset_id)


def create_custom_preset(
    preset_id: str, name: str, description: str, zim_paths: List[str]
) -> Dict:
    """Create a custom preset."""
    presets = load_presets()
    custom_preset = {
        "name": name,
        "description": description,
        "category": "custom",
        "zim_files": [
            {"path": path, "name": os.path.basename(path)} for path in zim_paths
        ],
        "created_at": datetime.utcnow().isoformat(),
    }
    presets["custom"][preset_id] = custom_preset
    save_presets(presets)
    return custom_preset


def set_active_preset(preset_id: str) -> bool:
    """Set the active preset."""
    presets = load_presets()
    if preset_id in get_all_presets():
        presets["active"] = preset_id
        save_presets(presets)
        return True
    return False


def get_active_preset() -> Optional[Dict]:
    """Get the currently active preset."""
    presets = load_presets()
    active_id = presets.get("active")
    if active_id:
        return {"id": active_id, "preset": get_preset(active_id)}
    return None


def delete_custom_preset(preset_id: str) -> bool:
    """Delete a custom preset."""
    presets = load_presets()
    if preset_id in presets["custom"]:
        del presets["custom"][preset_id]
        if presets.get("active") == preset_id:
            presets["active"] = None
        save_presets(presets)
        return True
    return False


def list_preset_files(preset_id: str) -> Optional[List[Dict]]:
    """List all ZIM files for a preset."""
    preset = get_preset(preset_id)
    return preset.get("zim_files", []) if preset else None


def get_installed_paths_for_preset(preset_id: str) -> List[str]:
    """Get local file paths for installed files in a preset."""
    return get_installed_files_for_preset(preset_id)


def get_preset_with_installation_status(preset_id: str) -> Optional[Dict]:
    """Get preset details with installation status for each file."""
    preset = get_preset(preset_id)
    if not preset:
        return None

    if preset.get("category") == "preset":
        installed_paths = get_installed_files_for_preset(preset_id)
        result = preset.copy()
        result["files_with_status"] = []
        for file_info in preset.get("zim_files", []):
            file_id = file_info.get("id")
            installed = any(path.endswith(f"{file_id}.zim") for path in installed_paths)
            result["files_with_status"].append(
                {**file_info, "installed": installed, "needs_download": not installed}
            )
        return result

    return preset
