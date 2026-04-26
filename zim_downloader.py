import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests

ZIM_FOLDER = "zim_files"
MANIFEST_FILE = "zim_manifest.json"

KIWIX_API = "https://library.kiwix.org/catalog/v2/entries"
_ATOM_NS = "http://www.w3.org/2005/Atom"

# Mapping of preset tuning IDs to their Kiwix <name> identifiers.
# Each "id" must exactly match the <name> field returned by the Kiwix OPDS catalog.
PRESET_FILES = {
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
    ],
    "coding": [
        {
            "id": "stackoverflow.com_en_all",
            "name": "Stack Overflow",
            "description": "Q&A for professional and enthusiast programmers (no text-only version available)",
            "size": "~80GB",
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
# Manifest helpers
# ---------------------------------------------------------------------------


def init_zim_folder():
    """Create ZIM files folder if it doesn't exist."""
    os.makedirs(ZIM_FOLDER, exist_ok=True)


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
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


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
    """Return all available files grouped by tuning preset."""
    return PRESET_FILES


def list_installed_files() -> Dict:
    """Return all installed ZIM files from the manifest."""
    return load_manifest().get("installed", {})


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


def get_installed_files_for_preset(preset_id: str) -> List[str]:
    """Return disk paths of installed files for a specific preset.

    For the virtual 'devdocs_all' bundle, expands to the individual paths of
    every devdocs_en_* entry present in the manifest.
    """
    if preset_id not in PRESET_FILES:
        return []

    installed = list_installed_files()
    result = []
    for file_info in PRESET_FILES[preset_id]:
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

        success = True
        for i, entry in enumerate(to_install, 1):
            print(f"[{i}/{len(to_install)}] {entry['name']}")
            ok = download_file(entry["id"], show_progress=show_progress)
            if not ok:
                success = False
            print()
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
    filepath = os.path.join(ZIM_FOLDER, filename)

    print(f"\nDownloading: {file_info['title']}")
    print(f"Size:        {file_info['size']}")
    print(f"URL:         {url}\n")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
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
            "path": filepath,
            "title": file_info["title"],
            "size": file_info["size"],
            "installed_at": Path(filepath).stat().st_mtime,
        }
        save_manifest(manifest)

        print(f"✓ Successfully installed: {filepath}")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
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


def get_preset_installation_status(preset_id: str) -> Dict:
    """Return installation status for every file in a preset."""
    if preset_id not in PRESET_FILES:
        return {
            "error": f"Unknown preset: '{preset_id}'. "
            f"Valid options: {', '.join(PRESET_FILES)}"
        }

    installed_map = list_installed_files()
    files_status = []

    for file_info in PRESET_FILES[preset_id]:
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

    return {"preset": preset_id, "files": files_status}


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
