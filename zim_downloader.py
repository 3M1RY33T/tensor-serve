import requests
import os
import json
from typing import List, Dict, Optional
from pathlib import Path

ZIM_FOLDER = "zim_files"
MANIFEST_FILE = "zim_manifest.json"

KIWIX_API = "https://library.kiwix.org/catalog/v2/entries"

# Mapping of preset tuning files to Kiwix identifiers
PRESET_FILES = {
    "research": [
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
    "learn": [
        {
            "id": "libretexts",
            "name": "LibreTexts",
            "description": "Open-source textbooks",
            "size": "~15GB",
        },
        {
            "id": "wikiversity_en",
            "name": "Wikiversity",
            "description": "Free learning environment",
            "size": "~3GB",
        },
    ],
    "literature": [
        {
            "id": "gutenberg",
            "name": "Project Gutenberg",
            "description": "Free ebooks",
            "size": "~25GB",
        },
        {
            "id": "wikibooks_en",
            "name": "Wikibooks",
            "description": "Free textbooks and manuals",
            "size": "~3GB",
        },
    ],
    "coding": [
        {
            "id": "devdocs",
            "name": "DevDocs",
            "description": "API documentation",
            "size": "~5GB",
        },
        {
            "id": "stack_exchange",
            "name": "Stack Exchange",
            "description": "Q&A from Stack Overflow",
            "size": "~10GB",
        },
    ],
}


def init_zim_folder():
    """Create ZIM files folder if it doesn't exist."""
    os.makedirs(ZIM_FOLDER, exist_ok=True)


def init_manifest():
    """Initialize manifest file if it doesn't exist."""
    if not os.path.exists(MANIFEST_FILE):
        manifest = {
            "installed": {},
            "downloading": {},
        }
        save_manifest(manifest)


def load_manifest() -> Dict:
    """Load manifest of installed ZIM files."""
    init_manifest()
    with open(MANIFEST_FILE, "r") as f:
        return json.load(f)


def save_manifest(manifest: Dict):
    """Save manifest of installed ZIM files."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def get_kiwix_file_info(file_id: str) -> Optional[Dict]:
    """Get file information from Kiwix library."""
    try:
        response = requests.get(KIWIX_API, timeout=10)
        response.raise_for_status()
        data = response.json()

        for entry in data.get("entries", []):
            if entry.get("id") == file_id:
                return {
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "size": entry.get("size"),
                    "url": entry.get("downloadUrl"),
                    "md5": entry.get("md5Hash"),
                }
        return None
    except Exception as e:
        print(f"Error fetching Kiwix info: {e}")
        return None


def list_available_files() -> Dict[str, List[Dict]]:
    """List all available files from preset tunings."""
    return PRESET_FILES


def list_installed_files() -> Dict:
    """List all installed ZIM files."""
    manifest = load_manifest()
    return manifest.get("installed", {})


def is_file_installed(file_id: str) -> bool:
    """Check if a file is installed."""
    installed = list_installed_files()
    return file_id in installed and os.path.exists(installed[file_id]["path"])


def get_installed_files_for_tuning(tuning_id: str) -> List[str]:
    """Get installed files for a specific tuning."""
    if tuning_id not in PRESET_FILES:
        return []

    installed = list_installed_files()
    result = []

    for file_info in PRESET_FILES[tuning_id]:
        file_id = file_info["id"]
        if is_file_installed(file_id):
            result.append(installed[file_id]["path"])

    return result


def download_file(file_id: str, show_progress: bool = True) -> bool:
    """
    Download a ZIM file from Kiwix.
    
    Args:
        file_id: Kiwix file identifier
        show_progress: Show download progress
        
    Returns:
        True if successful, False otherwise
    """
    init_zim_folder()
    init_manifest()

    # Check if already installed
    if is_file_installed(file_id):
        print(f"✓ File '{file_id}' is already installed")
        return True

    # Get file info from Kiwix
    print(f"Fetching information for '{file_id}'...")
    file_info = get_kiwix_file_info(file_id)

    if not file_info:
        print(f"✗ Could not find file '{file_id}' on Kiwix")
        return False

    url = file_info.get("url")
    if not url:
        print(f"✗ No download URL available for '{file_id}'")
        return False

    # Download file
    filename = os.path.basename(url).split("?")[0]
    filepath = os.path.join(ZIM_FOLDER, filename)

    print(f"\nDownloading: {file_info.get('title')}")
    print(f"Size: {file_info.get('size')}")
    print(f"URL: {url}\n")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")

        # Update manifest
        manifest = load_manifest()
        manifest["installed"][file_id] = {
            "path": filepath,
            "title": file_info.get("title"),
            "size": file_info.get("size"),
            "md5": file_info.get("md5"),
            "installed_at": Path(filepath).stat().st_mtime,
        }
        save_manifest(manifest)

        print(f"✓ Successfully installed: {filepath}")
        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def uninstall_file(file_id: str) -> bool:
    """Remove an installed ZIM file."""
    manifest = load_manifest()
    installed = manifest.get("installed", {})

    if file_id not in installed:
        print(f"✗ File '{file_id}' is not installed")
        return False

    filepath = installed[file_id]["path"]

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
        del manifest["installed"][file_id]
        save_manifest(manifest)
        print(f"✓ Uninstalled: {file_id}")
        return True
    except Exception as e:
        print(f"✗ Failed to uninstall: {e}")
        return False


def get_tuning_installation_status(tuning_id: str) -> Dict:
    """Get installation status for all files in a tuning."""
    if tuning_id not in PRESET_FILES:
        return {"error": f"Unknown tuning: {tuning_id}"}

    status = {"tuning": tuning_id, "files": []}

    for file_info in PRESET_FILES[tuning_id]:
        file_id = file_info["id"]
        installed = is_file_installed(file_id)

        status["files"].append(
            {
                "id": file_id,
                "name": file_info["name"],
                "description": file_info["description"],
                "size": file_info["size"],
                "installed": installed,
                "path": list_installed_files().get(file_id, {}).get("path")
                if installed
                else None,
            }
        )

    return status
