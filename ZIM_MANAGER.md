# ZIM File Manager

Tensor Serve includes a **ZIM File Manager** that allows you to download and manage ZIM files from the Kiwix library directly from the terminal. This enables users without pre-downloaded ZIM files to easily set up their preferred knowledge bases.

## Quick Start

### 1. Check Available Files
```bash
python manage_zim.py list
```

Shows all available ZIM files organized by tuning (Research, Learn, Literature, Coding).

### 2. Check Installation Status
```bash
python manage_zim.py status              # Overall status
python manage_zim.py status research     # Status for Research tuning
```

Shows which files are installed and their local paths.

### 3. Install Files for a Tuning
```bash
python manage_zim.py install-tuning research
```

Interactive installation wizard:
- Lists uninstalled files for the tuning
- Let's you select which files to install
- Supports "all" to install everything
- Downloads files to `./zim_files/` folder

### 4. Install Individual Files
```bash
python manage_zim.py install wikipedia_en
python manage_zim.py install stack_exchange
```

### 5. Uninstall Files
```bash
python manage_zim.py uninstall wikipedia_en
```

## CLI Commands Reference

### list
List all available ZIM files by tuning.

```bash
python manage_zim.py list
```

**Output Example:**
```
================================================================================
AVAILABLE ZIM FILES FOR TENSOR SERVE
================================================================================

📚 RESEARCH
────────────────────────────────────────────────────────────────────────────────
  ID:          wikipedia_en
  Name:        Wikipedia
  Description: The free encyclopedia
  Size:        ~20GB

  ID:          wikisource_en
  Name:        Wikisource
  Description: Free library of texts
  Size:        ~5GB
```

### status
Show installation status.

```bash
# Overall status
python manage_zim.py status

# Status for specific tuning
python manage_zim.py status research
python manage_zim.py status learn
python manage_zim.py status literature
python manage_zim.py status coding
```

**Output Example:**
```
================================================================================
Installation Status: RESEARCH
================================================================================

✓ Wikipedia (wikipedia_en)
  Size: ~20GB
  Path: /path/to/zim_files/wikipedia_en.zim

○ Wikisource (wikisource_en)
  Size: ~5GB
```

### install
Install a specific ZIM file by ID.

```bash
python manage_zim.py install wikipedia_en
python manage_zim.py install coding devdocs
```

Downloads file to `./zim_files/` and tracks installation in `zim_manifest.json`.

### uninstall
Uninstall a ZIM file.

```bash
python manage_zim.py uninstall wikipedia_en
```

Removes the file and updates the manifest.

### install-tuning
Interactive installation for a complete tuning.

```bash
python manage_zim.py install-tuning research
python manage_zim.py install-tuning coding
python manage_zim.py install-tuning literature
python manage_zim.py install-tuning learn
```

**Interactive Example:**
```
================================================================================
INSTALL RESEARCH TUNING
================================================================================

Available files to install:

1. Wikipedia (wikipedia_en)
   Description: The free encyclopedia
   Size: ~20GB

2. Wikisource (wikisource_en)
   Description: Free library of texts
   Size: ~5GB

3. Wikinews (wikinews_en)
   Description: Free news source
   Size: ~2GB

Select files to install (comma-separated numbers, or 'all'):
> 1,2

Installing 2 file(s)...

Fetching information for 'wikipedia_en'...
Downloading: Wikipedia
Size: ~20GB
URL: https://download.kiwix.org/...

Progress: 45.3% (9234567890/20000000000 bytes)
```

## Complete Workflow

### Setup from Scratch

```bash
# 1. Start tensor-serve
uvicorn main:app --reload

# 2. Check what's available
python manage_zim.py list

# 3. Install files for a tuning
python manage_zim.py install-tuning research

# 4. Verify installation
python manage_zim.py status research

# 5. Start tensor-serve and configure
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "mistral"
  }'

# 6. Ingest the research tuning
curl -X POST http://localhost:8000/tunings/research/ingest

# 7. Load and chat
curl http://localhost:8000/load?name=research_db
curl -X POST http://localhost:8000/chat \
  -d '{"message": "What is photosynthesis?"}'
```

## API Endpoints for ZIM Management

### GET /zim/available
List all available ZIM files for download.

```bash
curl http://localhost:8000/zim/available
```

**Response:**
```json
{
  "research": {
    "files": [
      {
        "id": "wikipedia_en",
        "name": "Wikipedia",
        "description": "The free encyclopedia",
        "size": "~20GB",
        "installed": false
      }
    ]
  }
}
```

### GET /zim/installed
List all installed ZIM files.

```bash
curl http://localhost:8000/zim/installed
```

**Response:**
```json
{
  "installed_files": {
    "wikipedia_en": {
      "title": "Wikipedia",
      "size": "20GB",
      "path": "/path/to/zim_files/wikipedia_en.zim"
    }
  },
  "count": 1
}
```

### GET /zim/status/{tuning_id}
Get ZIM installation status for a tuning.

```bash
curl http://localhost:8000/zim/status/research
```

**Response:**
```json
{
  "tuning": "research",
  "files": [
    {
      "id": "wikipedia_en",
      "name": "Wikipedia",
      "description": "The free encyclopedia",
      "size": "~20GB",
      "installed": true,
      "path": "/path/to/zim_files/wikipedia_en.zim"
    }
  ]
}
```

## Files and Storage

### Manifest File: `zim_manifest.json`
Tracks all installed ZIM files:

```json
{
  "installed": {
    "wikipedia_en": {
      "path": "/path/to/zim_files/wikipedia_en.zim",
      "title": "Wikipedia",
      "size": "20GB",
      "md5": "abc123...",
      "installed_at": 1234567890
    }
  },
  "downloading": {}
}
```

### Storage Directory: `zim_files/`
Where downloaded ZIM files are stored:
```
zim_files/
├── wikipedia_en.zim
├── wikisource_en.zim
├── stack_exchange.zim
└── devdocs.zim
```

## Tuning File Requirements

### Research (3 files, ~27GB)
- ✓ Wikipedia - ~20GB
- ○ Wikisource - ~5GB
- ○ Wikinews - ~2GB

### Learn (2 files, ~18GB)
- ○ LibreTexts - ~15GB
- ○ Wikiversity - ~3GB

### Literature (2 files, ~28GB)
- ○ Project Gutenberg - ~25GB
- ○ Wikibooks - ~3GB

### Coding (2 files, ~15GB)
- ○ DevDocs - ~5GB
- ○ Stack Exchange - ~10GB

## Troubleshooting

### "Could not find file on Kiwix"
- The file ID might be incorrect
- Check available files with `python manage_zim.py list`
- Verify internet connection

### Download interrupted
- Re-run the install command (will resume if possible)
- Check disk space availability
- Verify network stability

### File not found after installation
- Check `python manage_zim.py status`
- Verify file exists in `zim_files/` folder
- Check `zim_manifest.json` for path

### "No local ZIM files found" when ingesting
- Install files first with CLI: `python manage_zim.py install-tuning <tuning>`
- Verify files installed: `python manage_zim.py status <tuning>`
- Check API: `GET /zim/installed`

## Performance

- **Download speed**: Depends on internet connection (typically 1-10 MB/s)
- **Installation time**: See file sizes (e.g., Wikipedia ~2-5 hours on average connection)
- **Storage**: Ensure sufficient disk space (multiple large files)
- **Compression**: ZIM files are pre-compressed, no additional compression needed

## Network Usage

- Downloads come from official Kiwix mirrors
- Typical download sizes: 2-25GB per file
- Can pause and resume downloads
- Recommended to use on fast, stable connection

## Advanced Usage

### Download without Tensor Serve running
```bash
# Works independently of the API server
python manage_zim.py install wikipedia_en
```

### Programmatic Usage
```python
from zim_downloader import download_file, list_installed_files

# Download a file
download_file("wikipedia_en", show_progress=True)

# Check what's installed
installed = list_installed_files()
for file_id, info in installed.items():
    print(f"{info['title']}: {info['path']}")
```

### Check Kiwix API directly
```python
from zim_downloader import get_kiwix_file_info

info = get_kiwix_file_info("wikipedia_en")
print(info["url"])  # Get download URL
```
