# Tunings Feature Guide

Tensor Serve includes a **Tunings** system that lets you manage curated collections of ZIM files and AI model configurations. Each tuning provides a tailored experience for different use cases.

## Available Preset Tunings

### Research
- **Description**: Academic and encyclopedic content
- **Files**: Wikipedia, Wikisource, Wikinews
- **Best for**: Academic research, fact-checking, broad knowledge
- **Load time**: ~30 minutes
- **Size**: ~27GB

### Learn
- **Description**: Educational and textbook content
- **Files**: LibreTexts, Wikiversity
- **Best for**: Studying, learning new subjects, educational content
- **Load time**: ~20 minutes
- **Size**: ~18GB

### Literature
- **Description**: Books and literary works
- **Files**: Project Gutenberg, Wikibooks
- **Best for**: Reading, writing, literary analysis
- **Load time**: ~35 minutes
- **Size**: ~28GB

### Coding
- **Description**: Developer documentation and resources
- **Files**: DevDocs, Stack Exchange
- **Best for**: Programming, debugging, technical questions
- **Load time**: ~15 minutes
- **Size**: ~15GB

### Custom
- **Description**: User-defined collection
- **Files**: Any ZIM files you choose
- **Best for**: Specialized domains, combined content
- **Load time**: Depends on files

## API Usage

### List Available Tunings
```bash
curl http://localhost:8000/tunings
```

**Response**:
```json
{
  "tunings": {
    "research": {
      "name": "Research",
      "description": "Academic and encyclopedic content for research",
      "category": "preset",
      "file_count": 3
    },
    "learn": { ... },
    "literature": { ... },
    "coding": { ... }
  },
  "active": null
}
```

### Get Tuning Details
```bash
curl http://localhost:8000/tunings/research
```

**Response**:
```json
{
  "id": "research",
  "name": "Research",
  "description": "Academic and encyclopedic content for research",
  "category": "preset",
  "zim_files": [
    {
      "name": "Wikipedia",
      "url": "https://wiki.kiwix.org/wiki/Wikipedia",
      "description": "The free encyclopedia",
      "size": "~20GB"
    },
    ...
  ]
}
```

### Ingest Tuning (All Files)
```bash
curl -X POST http://localhost:8000/tunings/research/ingest
```

**Response**:
```json
{
  "status": "completed",
  "output": "research_db",
  "tuning_id": "research",
  "files_processed": 3,
  "total_articles": 6789123,
  "total_chunks": 50000000
}
```

### Ingest Tuning (Selected Files)
```bash
# Ingest only files at indices 0 and 2
curl -X POST "http://localhost:8000/tunings/research/ingest?zim_file_indices=0&zim_file_indices=2"
```

### Load Tuning Database
```bash
curl http://localhost:8000/load?name=research_db
```

### Chat with Active Tuning
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain photosynthesis"
  }'
```

## Custom Tunings

### Create Custom Tuning
```bash
curl -X POST http://localhost:8000/tunings/custom/create \
  -H "Content-Type: application/json" \
  -d '{
    "tuning_id": "my_science",
    "name": "Science Collection",
    "description": "Biology, Chemistry, and Physics",
    "zim_paths": [
      "/path/to/biology.zim",
      "/path/to/chemistry.zim",
      "/path/to/physics.zim"
    ]
  }'
```

**Response**:
```json
{
  "status": "created",
  "tuning_id": "my_science",
  "name": "Science Collection",
  "file_count": 3
}
```

### Ingest Custom Tuning
```bash
curl -X POST http://localhost:8000/tunings/my_science/ingest
```

### Delete Custom Tuning
```bash
curl -X DELETE http://localhost:8000/tunings/custom/my_science
```

**Response**:
```json
{
  "status": "deleted",
  "tuning_id": "my_science"
}
```

## Complete Workflow

### Example 1: Use Research Tuning

```bash
# 1. Check available tunings
curl http://localhost:8000/tunings

# 2. View details of Research tuning
curl http://localhost:8000/tunings/research

# 3. Configure AI endpoint (if not already done)
curl -X POST http://localhost:8000/config/set-ai-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "ai_endpoint": "http://localhost:11434",
    "ai_model": "mistral"
  }'

# 4. Ingest the tuning (takes ~30 minutes)
curl -X POST http://localhost:8000/tunings/research/ingest

# 5. Load the database
curl http://localhost:8000/load?name=research_db

# 6. Start chatting
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who won the Nobel Prize in Physics in 2023?"}'
```

### Example 2: Create Custom Programming Tuning

```bash
# 1. Create custom tuning with your downloaded ZIM files
curl -X POST http://localhost:8000/tunings/custom/create \
  -H "Content-Type: application/json" \
  -d '{
    "tuning_id": "python_dev",
    "name": "Python Development",
    "description": "Python docs, Django, FastAPI, etc",
    "zim_paths": [
      "/data/python.zim",
      "/data/django.zim",
      "/data/fastapi.zim"
    ]
  }'

# 2. Ingest the custom tuning
curl -X POST http://localhost:8000/tunings/python_dev/ingest

# 3. Load the database
curl http://localhost:8000/load?name=python_dev_db

# 4. Ask Python questions
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I create a FastAPI route?"}'
```

## Configuration File

Tunings are stored in `tunings.json`:

```json
{
  "active": "research",
  "presets": {
    "research": { ... },
    "learn": { ... },
    "literature": { ... },
    "coding": { ... }
  },
  "custom": {
    "my_science": { ... },
    "python_dev": { ... }
  }
}
```

## Performance Notes

- **First ingest is slow**: Creates embeddings for all chunks
- **Subsequent loads are fast**: Reads cached FAISS index
- **Larger tunings**: Use more disk space but provide broader context
- **Smaller selections**: Use less disk space but narrower context

## Memory Usage

| Tuning | Files | Chunks | Index Size | Load Time |
|--------|-------|--------|------------|-----------|
| Research | 3 | 50M | ~15GB | 2-5 sec |
| Learn | 2 | 30M | ~9GB | 1-3 sec |
| Literature | 2 | 35M | ~10GB | 2-4 sec |
| Coding | 2 | 25M | ~7GB | 1-3 sec |

## Tips

1. **Start small**: Begin with the Coding tuning (smallest, fastest)
2. **Use selective ingestion**: Ingest only the files you need
3. **Multiple tunings**: Create different custom tunings for different projects
4. **Switch tunings**: Delete old databases and load new ones as needed
5. **Monitor disk space**: Large tunings take 10-30GB per database

## Troubleshooting

**"Remote file not supported"**
- Download ZIM files locally first
- Preset tunings show URLs where to download from Kiwix

**"No local ZIM files found"**
- Ensure ZIM file paths are absolute paths
- Check files actually exist

**"DB not found"**
- Ingest the tuning first before loading
- Default database name is `{tuning_id}_db`

**Slow ingestion**
- Large files take time (normal)
- Check disk space availability
- Monitor system resources
