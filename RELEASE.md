# Release Guide

Use this checklist before publishing `tensor-serve` to PyPI.

## 1. Prepare a Clean Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## 2. Validate

```bash
python -m compileall api cli tensor_serve main.py
pytest
python -m tensor_serve --help
python -m tensor_serve config --help
python -m tensor_serve zim --help
python -m tensor_serve ingest --help
python -m tensor_serve db --help
python -m tensor_serve collections --help
```

## 3. Build

```bash
rm -rf build dist *.egg-info
python -m build
twine check dist/*
```

Inspect the generated archive before upload:

```bash
tar -tzf dist/tensor_serve-*.tar.gz | less
unzip -l dist/tensor_serve-*.whl | less
```

Make sure generated runtime files are not present, especially:

- `config.json`
- `.tensor_config.key`
- `collections.json`
- `zim_manifest.json`
- `zim_files/`
- `*.index`, `*.pkl`, `*.bm25`

## 4. Publish

Prefer PyPI Trusted Publishing from GitHub Actions. The included publish
workflow runs when a GitHub release is published, and can also be started with
`workflow_dispatch`. Configure a PyPI Trusted Publisher for:

- Repository: `3M1RY33T/tensor-serve`
- Workflow: `publish.yml`
- Environment: `pypi`

For a manual upload:

```bash
twine upload dist/*
```

After publishing:

```bash
python3 -m venv /tmp/tensor-serve-smoke
source /tmp/tensor-serve-smoke/bin/activate
pip install tensor-serve
tensor-serve --help
```
