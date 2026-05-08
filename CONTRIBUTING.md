# Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. First, fork the this repository on GitHub. This will create your own copy of the code that you can modify.
2. Next, clone the forked repository to your local machine. This will allow you to make changes to the code and test them.
3. Create a new branch for your changes. This helps to keep your changes separate from the main codebase until they are ready to be merged.
4. Make the necessary changes to the code and test them to ensure that they work as expected.
5. When you're happy with your changes, commit them to your forked repository and push them to GitHub.
6. Finally, submit a pull request to the this repository on GitHub. 

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request:

```bash
python -m compileall api cli tensor_serve main.py
python -m pytest
python -m tensor_serve --help
```

For packaging changes:

```bash
python -m build
python -m twine check dist/*
```

For Docker changes:

```bash
docker build -t tensor-serve:local .
docker run --rm -p 8000:8000 tensor-serve:local
```

## Pull Requests

- Keep changes focused.
- Include tests for behavior changes.
- Update `README.md`, `cli/README.md`, or `api/README.md` when user-facing behavior changes.
- Do not commit generated runtime state such as `config.json`, `collections.json`, `zim_manifest.json`, `zim_files/`, vector DB files, or `.tensor_config.key`.
