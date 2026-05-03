import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.config import get_config_value, set_config_value

ZIM_FOLDER = "zim_files"
MANIFEST_FILE = "zim_manifest.json"

KIWIX_API = "https://library.kiwix.org/catalog/v2/entries"
_ATOM_NS = "http://www.w3.org/2005/Atom"

# Mapping of collection IDs to their Kiwix <name> identifiers.
# Each "id" must exactly match the <name> field returned by the Kiwix OPDS catalog.
COLLECTION_FILES = {
    "research": [
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
    "learn": [
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
    "literature": [
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
    "coding": [
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
}


# ---------------------------------------------------------------------------
# Download progress tracking (in-memory, polled by the REST API)
# ---------------------------------------------------------------------------

# Maps file_id -> progress dict. Written by download_file(), read by the API.
_download_progress: Dict[str, Dict] = {}


def get_download_progress() -> Dict:
    """Return a snapshot of all tracked downloads (active and recently finished)."""
    return dict(_download_progress)


def get_file_progress(file_id: str) -> Optional[Dict]:
    """Return progress info for a single file_id, or None if not tracked."""
    return _download_progress.get(file_id)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _normalize_folder(path: str) -> str:
    """Return a stable absolute folder path for user-provided ZIM storage."""
    return os.path.abspath(os.path.expanduser(path))


def has_custom_zim_source_folder() -> bool:
    """Return True when the user has configured a non-default ZIM source folder."""
    return bool(get_config_value("zim_source_folder"))


def get_zim_source_folder() -> str:
    """Return the folder used for ZIM downloads and disk scans."""
    configured = get_config_value("zim_source_folder")
    if configured:
        return _normalize_folder(configured)
    return os.path.abspath(ZIM_FOLDER)


def set_zim_source_folder(path: str) -> str:
    """Save a user-provided folder for existing and future ZIM files."""
    folder = _normalize_folder(path)
    if not os.path.isdir(folder):
        raise ValueError(f"ZIM source folder does not exist: {folder}")
    set_config_value("zim_source_folder", folder)
    return folder


def clear_zim_source_folder() -> str:
    """Reset ZIM storage back to the default local folder."""
    set_config_value("zim_source_folder", None)
    return get_zim_source_folder()


def init_zim_folder():
    """Create the active ZIM folder if it doesn't exist."""
    os.makedirs(get_zim_source_folder(), exist_ok=True)


def init_manifest():
    """Initialize manifest file if it doesn't exist."""
    if not os.path.exists(MANIFEST_FILE):
        save_manifest({"installed": {}, "downloading": {}})


def load_manifest() -> Dict:
    """Load manifest of installed ZIM files."""
    init_manifest()
    with open(MANIFEST_FILE, "r") as f:
        return json.load(f)


def save_manifest(manifest: Dict):
    """Save manifest of installed ZIM files."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Kiwix OPDS catalog helpers
# ---------------------------------------------------------------------------


def bytes_to_human(num_bytes: int) -> str:
    """Convert a byte count to a human-readable size string."""
    value: float = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _find_text(element: ET.Element, tag: str) -> str:
    """Return the text of a child element in the Atom namespace, or ''."""
    child = element.find(f"{{{_ATOM_NS}}}{tag}")
    return (child.text or "").strip() if child is not None else ""


def _flavour_score(entry: ET.Element) -> int:
    """
    Score an OPDS entry by how text-friendly its flavour is.
    Lower score = more preferred.

      0  nopic  – full article text, no images (best for low file size)
      1  mini   – no images, but also drops article details
      2  (none) – only one version exists (e.g. devdocs, Stack Overflow)
      3  maxi   – full images; last resort when no text-only build exists
      9  other  – unknown flavour; deprioritise
    """
    flavour = _find_text(entry, "flavour")
    return {"nopic": 0, "mini": 1, "": 2, "maxi": 3}.get(flavour, 9)


def get_kiwix_file_info(file_id: str) -> Optional[Dict]:
    """
    Query the Kiwix OPDS catalog for a file by its <name> identifier.

    The catalog returns an Atom/XML feed (not JSON).  When multiple flavours
    exist for the same name (e.g. Wikipedia maxi / mini / nopic) the text-only
    'nopic' flavour is preferred to minimise file size, falling back to 'mini',
    then any unflavoured entry, and finally 'maxi' as a last resort.

    Returns a dict with keys: id, title, size, url  – or None on failure.
    """
    try:
        response = requests.get(KIWIX_API, params={"name": file_id}, timeout=15)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        entries = root.findall(f"{{{_ATOM_NS}}}entry")

        if not entries:
            return None

        # Pick the most text-friendly flavour available (nopic > mini > none > maxi).
        chosen = min(entries, key=_flavour_score)

        flavour = _find_text(chosen, "flavour")
        base_title = _find_text(chosen, "title") or file_id
        title = f"{base_title} ({flavour})" if flavour else base_title

        # Locate the open-access acquisition link that carries the download URL.
        url = None
        size_bytes = 0
        acquisition_rel = "http://opds-spec.org/acquisition/open-access"
        for link in chosen.findall(f"{{{_ATOM_NS}}}link"):
            if link.get("rel") == acquisition_rel:
                href = link.get("href", "")
                # The catalog serves .zim.meta4 metalink files.
                # Strip the .meta4 suffix to get the direct ZIM download URL.
                url = href.removesuffix(".meta4") if href.endswith(".meta4") else href
                try:
                    size_bytes = int(link.get("length", 0))
                except (ValueError, TypeError):
                    size_bytes = 0
                break

        return {
            "id": file_id,
            "title": title,
            "size": bytes_to_human(size_bytes) if size_bytes else "Unknown",
            "url": url,
        }

    except ET.ParseError as e:
        print(f"Error parsing Kiwix catalog XML: {e}")
        return None
    except Exception as e:
        print(f"Error fetching Kiwix info: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_available_files() -> Dict[str, List[Dict]]:
    """Return all available files grouped by collection."""
    return COLLECTION_FILES


def scan_zim_folder() -> Dict:
    """
    Reconcile the manifest with what is actually on disk.

    * Removes manifest entries whose .zim file no longer exists on disk.
    * Auto-registers any .zim files found in the active ZIM folder that are not yet
      tracked (e.g. files placed there manually or downloaded outside the tool).

    All paths are stored and compared as absolute paths so the function works
    correctly regardless of the process working directory.

    Returns the up-to-date installed dict (also saved to the manifest).
    """
    zim_dir = get_zim_source_folder()
    if not os.path.isdir(zim_dir):
        return load_manifest().get("installed", {})

    manifest = load_manifest()
    installed = manifest.get("installed", {})
    changed = False

    # Normalise every existing manifest path to absolute for consistent comparison
    tracked_paths = {os.path.abspath(v["path"]) for v in installed.values()}

    # 1. Purge stale entries — file was deleted from disk
    stale = [
        fid
        for fid, info in list(installed.items())
        if not os.path.exists(os.path.abspath(info["path"]))
    ]
    for fid in stale:
        del installed[fid]
        changed = True

    # 2. Register untracked .zim files found on disk
    for fname in os.listdir(zim_dir):
        if not fname.endswith(".zim"):
            continue
        fpath = os.path.join(zim_dir, fname)  # always absolute
        if fpath in tracked_paths:
            continue
        stem = fname[:-4]  # strip .zim
        size_bytes = os.path.getsize(fpath)
        installed[stem] = {
            "path": fpath,
            "title": stem,
            "size": bytes_to_human(size_bytes),
            "installed_at": os.path.getmtime(fpath),
            "untracked": True,
        }
        tracked_paths.add(fpath)
        changed = True

    if changed:
        manifest["installed"] = installed
        save_manifest(manifest)

    return installed


def list_installed_files() -> Dict:
    """Return all installed ZIM files, including any untracked files found on disk."""
    return scan_zim_folder()


def register_zim_file(
    path: str, file_id: Optional[str] = None, title: Optional[str] = None
) -> Dict:
    """Record an existing local ZIM file in the manifest without downloading it."""
    zim_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(zim_path):
        raise ValueError(f"ZIM file does not exist: {zim_path}")
    if not zim_path.endswith(".zim"):
        raise ValueError(f"Path must point to a .zim file: {zim_path}")

    zim_id = file_id or Path(zim_path).stem
    zim_title = title or zim_id
    size_bytes = os.path.getsize(zim_path)

    manifest = load_manifest()
    manifest.setdefault("installed", {})
    manifest["installed"][zim_id] = {
        "id": zim_id,
        "path": zim_path,
        "title": zim_title,
        "size": bytes_to_human(size_bytes),
        "installed_at": os.path.getmtime(zim_path),
        "registered": True,
    }
    save_manifest(manifest)
    return manifest["installed"][zim_id]


def is_file_installed(file_id: str) -> bool:
    """Return True if the file is recorded in the manifest and exists on disk.

    For the virtual 'devdocs_all' bundle, returns True as soon as at least one
    individual devdocs entry has been downloaded (no network call required).
    """
    installed = list_installed_files()
    if file_id == "devdocs_all":
        return any(k.startswith("devdocs_en_") for k in installed)
    return file_id in installed and os.path.exists(installed[file_id]["path"])


def _count_installed_devdocs() -> int:
    """Return the number of devdocs_en_* entries present in the manifest."""
    return sum(1 for k in list_installed_files() if k.startswith("devdocs_en_"))


def get_installed_files_for_collection(collection_id: str) -> List[str]:
    """Return disk paths of installed files for a specific collection.

    For the virtual 'devdocs_all' bundle, expands to the individual paths of
    every devdocs_en_* entry present in the manifest.
    """
    if collection_id not in COLLECTION_FILES:
        return []

    installed = list_installed_files()
    result = []
    for file_info in COLLECTION_FILES[collection_id]:
        file_id = file_info["id"]
        if file_id == "devdocs_all":
            for k, v in installed.items():
                if k.startswith("devdocs_en_") and os.path.exists(v["path"]):
                    result.append(v["path"])
        elif is_file_installed(file_id):
            result.append(installed[file_id]["path"])
    return result


def download_file(file_id: str, show_progress: bool = True) -> bool:
    """
    Download a ZIM file from Kiwix by its catalog <name> identifier.

    Pass ``file_id='devdocs_all'`` to bulk-install every entry in the live
    DevDocs catalog (equivalent to running ``install-devdocs`` from the CLI).

    Args:
        file_id:       Kiwix <name> value (e.g. 'devdocs_en_python') or the
                       special value 'devdocs_all' to install the full collection.
        show_progress: Print a progress indicator while downloading.

    Returns:
        True on success (or partial success), False if nothing could be installed.
    """
    # ── special case: bulk-install the entire DevDocs collection ─────────────
    if file_id == "devdocs_all":
        print("Fetching DevDocs catalog from Kiwix...")
        catalog = list_devdocs_catalog()
        if not catalog:
            print("✗ Could not fetch DevDocs catalog")
            return False

        to_install = [e for e in catalog if not is_file_installed(e["id"])]
        if not to_install:
            print("✓ All DevDocs entries are already installed")
            return True

        total_bytes = sum(e["size_bytes"] for e in to_install)
        print(
            f"Installing {len(to_install)} DevDocs entries ({bytes_to_human(total_bytes)})...\n"
        )
        _download_progress["devdocs_all"] = {
            "status": "downloading",
            "title": "All DevDocs",
            "total_files": len(to_install),
            "completed_files": 0,
            "failed_files": 0,
            "total_bytes": total_bytes,
            "total": bytes_to_human(total_bytes),
        }

        success = True
        for i, entry in enumerate(to_install, 1):
            print(f"[{i}/{len(to_install)}] {entry['name']}")
            ok = download_file(entry["id"], show_progress=show_progress)
            if ok:
                _download_progress["devdocs_all"]["completed_files"] = i
            else:
                success = False
                _download_progress["devdocs_all"]["failed_files"] = (
                    _download_progress["devdocs_all"].get("failed_files", 0) + 1
                )
            print()
        _download_progress["devdocs_all"]["status"] = (
            "completed" if success else "partial"
        )
        return success
    # ─────────────────────────────────────────────────────────────────────────

    init_zim_folder()
    init_manifest()

    if is_file_installed(file_id):
        print(f"✓ File '{file_id}' is already installed")
        return True

    print(f"Fetching catalog info for '{file_id}'...")
    file_info = get_kiwix_file_info(file_id)

    if not file_info:
        print(f"✗ Could not find '{file_id}' in the Kiwix catalog")
        return False

    url = file_info.get("url")
    if not url:
        print(f"✗ No download URL available for '{file_id}'")
        return False

    filename = os.path.basename(url.split("?")[0])
    filepath = os.path.join(get_zim_source_folder(), filename)

    print(f"\nDownloading: {file_info['title']}")
    print(f"Size:        {file_info['size']}")
    print(f"URL:         {url}\n")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        _download_progress[file_id] = {
            "status": "downloading",
            "title": file_info["title"],
            "downloaded_bytes": 0,
            "total_bytes": total_size,
            "percent": 0.0,
            "downloaded": "0 B",
            "total": file_info["size"],
        }

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    _pct = (downloaded / total_size * 100) if total_size > 0 else 0.0
                    _download_progress[file_id].update(
                        {
                            "downloaded_bytes": downloaded,
                            "percent": round(_pct, 1),
                            "downloaded": bytes_to_human(downloaded),
                        }
                    )
                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        downloaded_str = bytes_to_human(downloaded)
                        total_str = bytes_to_human(total_size)
                        print(
                            f"\r[{bar}] {percent:5.1f}%  {downloaded_str} / {total_str}",
                            end="",
                            flush=True,
                        )

        if show_progress and total_size > 0:
            print()  # newline after progress bar

        # Record in manifest
        manifest = load_manifest()
        manifest["installed"][file_id] = {
            "path": os.path.abspath(filepath),
            "title": file_info["title"],
            "size": file_info["size"],
            "installed_at": Path(filepath).stat().st_mtime,
        }
        save_manifest(manifest)
        _download_progress[file_id]["status"] = "completed"

        print(f"✓ Successfully installed: {filepath}")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        _download_progress[file_id] = {
            "status": "error",
            "error": str(e),
            "file_id": file_id,
        }
        return False


def uninstall_file(file_id: str) -> bool:
    """Remove an installed ZIM file from disk and from the manifest."""
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


def get_collection_installation_status(collection_id: str) -> Dict:
    """Return installation status for every file in a collection."""
    if collection_id not in COLLECTION_FILES:
        return {
            "error": f"Unknown collection: '{collection_id}'. "
            f"Valid options: {', '.join(COLLECTION_FILES)}"
        }

    installed_map = list_installed_files()
    files_status = []

    for file_info in COLLECTION_FILES[collection_id]:
        file_id = file_info["id"]
        installed = is_file_installed(file_id)

        entry = {
            "id": file_id,
            "name": file_info["name"],
            "description": file_info["description"],
            "size": file_info["size"],
            "installed": installed,
            "path": installed_map.get(file_id, {}).get("path") if installed else None,
        }

        if file_id == "devdocs_all":
            count = _count_installed_devdocs()
            entry["installed_count"] = count
            entry["path"] = None

        files_status.append(entry)

    return {"collection": collection_id, "files": files_status}


def list_devdocs_catalog() -> List[Dict]:
    """
    Fetch every English devdocs entry from the Kiwix OPDS catalog dynamically.

    Uses the ``tag=devdocs`` filter so the list always reflects what Kiwix
    currently publishes (231+ entries as of 2026).

    Returns a list of dicts sorted alphabetically by title, each containing:
        id          – Kiwix <name> value (e.g. 'devdocs_en_python')
        name        – human-readable title  (e.g. 'Python Docs')
        description – one-line summary
        size        – human-readable size string
        size_bytes  – raw byte count (useful for computing totals)
        url         – direct .zim download URL (meta4 suffix stripped)
    """
    acquisition_rel = "http://opds-spec.org/acquisition/open-access"
    try:
        response = requests.get(
            KIWIX_API,
            params={"tag": "devdocs", "lang": "eng", "count": 500},
            timeout=30,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        entries = root.findall(f"{{{_ATOM_NS}}}entry")

        result = []
        for entry in entries:
            name = _find_text(entry, "name")
            if not name:
                continue

            title = _find_text(entry, "title")
            summary = _find_text(entry, "summary")

            url = None
            size_bytes = 0
            for link in entry.findall(f"{{{_ATOM_NS}}}link"):
                if link.get("rel") == acquisition_rel:
                    href = link.get("href", "")
                    url = (
                        href.removesuffix(".meta4") if href.endswith(".meta4") else href
                    )
                    try:
                        size_bytes = int(link.get("length", 0))
                    except (ValueError, TypeError):
                        size_bytes = 0
                    break

            result.append(
                {
                    "id": name,
                    "name": title,
                    "description": summary,
                    "size": bytes_to_human(size_bytes) if size_bytes else "Unknown",
                    "size_bytes": size_bytes,
                    "url": url,
                }
            )

        return sorted(result, key=lambda x: x["name"].lower())

    except ET.ParseError as e:
        print(f"Error parsing Kiwix catalog XML: {e}")
        return []
    except Exception as e:
        print(f"Error fetching devdocs catalog: {e}")
        return []
