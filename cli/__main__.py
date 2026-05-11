#!/usr/bin/env python3
"""
Tensor Serve - CLI interface for managing ZIM files and running the server
"""

import argparse
import json
from pathlib import Path
import sys


def start_server(args):
    """Start the Tensor Serve server."""
    from cli.run_server import main as run_server_main

    # Convert args to sys.argv format for the existing script
    original_argv = sys.argv[:]
    try:
        # Build argv for the server command parser.
        new_argv = ["tensor-serve start"]
        if args.port:
            new_argv.extend(['--port', str(args.port)])
        if args.host:
            new_argv.extend(['--host', args.host])
        if args.auto_port:
            new_argv.append('--auto-port')
        if args.reload:
            new_argv.append('--reload')

        sys.argv = new_argv
        run_server_main()
    finally:
        sys.argv = original_argv

def zim_command(args):
    """Handle ZIM file management commands."""
    from cli.manage_zim import main as manage_zim_main

    # Convert args to sys.argv format for the existing script
    original_argv = sys.argv[:]
    try:
        # Build argv for the ZIM command parser.
        new_argv = ["tensor-serve zim", args.subcommand]

        # Add any additional arguments based on subcommand
        if hasattr(args, 'category') and args.category:
            new_argv.append(args.category)
        if hasattr(args, 'file_id') and args.file_id:
            new_argv.append(args.file_id)

        sys.argv = new_argv
        manage_zim_main()
    finally:
        sys.argv = original_argv


def _print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _existing_vector_db(name):
    index_path = Path(f"{name}.index")
    text_path = Path(f"{name}.pkl")
    bm25_path = Path(f"{name}.bm25")
    return {
        "name": name,
        "index": str(index_path),
        "texts": str(text_path),
        "bm25": str(bm25_path) if bm25_path.exists() else None,
        "complete": index_path.exists() and text_path.exists(),
    }


def _require_vector_db(name):
    info = _existing_vector_db(name)
    if not info["complete"]:
        missing = []
        if not Path(info["index"]).exists():
            missing.append(info["index"])
        if not Path(info["texts"]).exists():
            missing.append(info["texts"])
        raise SystemExit(
            f"Vector database '{name}' is incomplete. Missing: {', '.join(missing)}"
        )
    return info


def _human_file_size(path):
    from api.zim_downloader import bytes_to_human

    return bytes_to_human(Path(path).stat().st_size)


def _server_request(method, server, path, params=None, json_body=None, timeout=60):
    import requests

    base = server.rstrip("/")
    try:
        response = requests.request(
            method,
            f"{base}{path}",
            params=params,
            json=json_body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"Could not reach Tensor Serve at {base}: {exc}")
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = payload.get("detail", payload)
        raise SystemExit(f"Server returned {response.status_code}: {detail}")
    return payload


def _server_get(server, path, params=None, timeout=60):
    return _server_request("GET", server, path, params=params, timeout=timeout)


def _server_post(server, path, params=None, json_body=None, timeout=60):
    return _server_request(
        "POST",
        server,
        path,
        params=params,
        json_body=json_body,
        timeout=timeout,
    )


def _parse_json_object(value, option_name):
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise SystemExit(f"Invalid JSON for {option_name}: {exc}")
    if not isinstance(parsed, dict):
        raise SystemExit(f"{option_name} must be a JSON object.")
    return parsed


def health_command(args):
    result = _server_get(args.server, "/health", timeout=args.timeout)
    _print_json(result)


def cache_stats(args):
    result = _server_get(args.server, "/cache/stats", timeout=args.timeout)
    _print_json(result)


def cache_clear(args):
    result = _server_post(args.server, "/cache/clear", timeout=args.timeout)
    _print_json(result)


def cache_command(args):
    if args.cache_command == "stats":
        cache_stats(args)
    elif args.cache_command == "clear":
        cache_clear(args)
    else:
        raise SystemExit("Unknown cache command")


def clean_command(args):
    result = _server_post(args.server, "/clean", timeout=args.timeout)
    _print_json(result)


def reset_all_command(args):
    result = _server_post(
        args.server,
        "/clean/all",
        params={"delete_collection_folders": not args.keep_collection_folders},
        timeout=args.timeout,
    )
    _print_json(result)


def ingest_command(args):
    from api.ingest import run_ingestion
    from api.multi_ingest import run_multi_ingest
    from api.zim_collections import (
        get_collection,
        list_collection_zim_paths,
        set_active_collection,
    )
    from api.zim_downloader import get_zim_source_folder, resolve_zim_inputs

    if not args.output_name:
        raise SystemExit("Provide --output-name to name the vector database.")

    source_modes = sum(
        [
            bool(args.paths),
            bool(args.source_folder),
            bool(args.collection),
        ]
    )
    if source_modes != 1:
        raise SystemExit("Use exactly one of paths, --source-folder, or --collection.")

    collection_id = None
    if args.collection:
        collection = get_collection(args.collection)
        if not collection:
            raise SystemExit(f"Collection '{args.collection}' not found")
        collection_id = args.collection
        zim_paths = list_collection_zim_paths(args.collection)
    elif args.source_folder:
        inputs = [get_zim_source_folder()]
        zim_paths = resolve_zim_inputs(inputs)
    else:
        inputs = args.paths
        zim_paths = resolve_zim_inputs(inputs)
    if not zim_paths:
        if collection_id:
            raise SystemExit(f"No ZIM files found in collection '{collection_id}'.")
        raise SystemExit("No .zim files found for the provided input.")

    if len(zim_paths) == 1:
        result = run_ingestion(zim_paths[0], args.output_name)
    else:
        result = run_multi_ingest(zim_paths, args.output_name)

    output = {**result, "db_name": args.output_name, "file_count": len(zim_paths)}
    if collection_id:
        set_active_collection(collection_id)
        output["collection_id"] = collection_id
    _print_json(output)


def db_list(args):
    dbs = []
    for index_path in sorted(Path(".").glob("*.index")):
        name = index_path.with_suffix("").name
        info = _existing_vector_db(name)
        info["index_size"] = _human_file_size(info["index"])
        if Path(info["texts"]).exists():
            info["texts_size"] = _human_file_size(info["texts"])
        if info["bm25"]:
            info["bm25_size"] = _human_file_size(info["bm25"])
        dbs.append(info)
    _print_json({"databases": dbs, "count": len(dbs)})


def db_show(args):
    info = _require_vector_db(args.name)
    info["index_size"] = _human_file_size(info["index"])
    info["texts_size"] = _human_file_size(info["texts"])
    if info["bm25"]:
        info["bm25_size"] = _human_file_size(info["bm25"])
    _print_json(info)


def db_load(args):
    _require_vector_db(args.name)
    result = _server_get(
        args.server,
        "/load",
        params={"name": args.name},
        timeout=args.timeout,
    )
    _print_json(result)


def db_status(args):
    result = _server_get(args.server, "/health", timeout=args.timeout)
    _print_json(result)


def db_command(args):
    if args.db_command == "list":
        db_list(args)
    elif args.db_command == "show":
        db_show(args)
    elif args.db_command in ("load", "use"):
        db_load(args)
    elif args.db_command == "status":
        db_status(args)
    else:
        raise SystemExit("Unknown db command")


def config_show(args):
    from api.config import load_config, mask_config
    from api.zim_downloader import get_zim_source_folder, has_custom_zim_source_folder

    config = mask_config(load_config())
    config["zim_source_folder"] = get_zim_source_folder()
    config["custom_zim_source_folder"] = has_custom_zim_source_folder()
    _print_json(config)


def config_set_ai_endpoint(args):
    from api.config import set_config_value

    extra_headers = {}
    if args.extra_headers:
        try:
            extra_headers = json.loads(args.extra_headers)
        except Exception as exc:
            raise SystemExit(
                f'Invalid JSON for --extra-headers: {exc}. Use \'{{"X-Header": "value"}}\' format.'
            )

    set_config_value("ai_provider", args.provider or "openai-compatible")
    set_config_value("ai_endpoint", args.endpoint)
    if args.model is not None:
        set_config_value("ai_model", args.model)
    if args.api_key is not None:
        set_config_value("ai_api_key", args.api_key)
    set_config_value("ai_api_key_header", args.api_key_header or "Authorization")
    set_config_value("ai_api_key_prefix", args.api_key_prefix or "Bearer")
    set_config_value("ai_extra_headers", extra_headers)

    print("Saved AI endpoint configuration to config.json")
    config_show(args)


def config_set_zim_source(args):
    from api.zim_downloader import set_zim_source_folder

    folder = set_zim_source_folder(args.path)
    print(f"ZIM source folder set to: {folder}")


def config_clear_zim_source(args):
    from api.zim_downloader import clear_zim_source_folder

    folder = clear_zim_source_folder()
    print(f"Cleared custom ZIM source. Using default folder: {folder}")


def config_set_search_modes(args):
    from api.config import set_config_value

    if args.keyword_mode:
        if args.keyword_mode not in ("auto", "web", "zim", "off"):
            raise SystemExit("keyword_search_mode must be one of auto, web, zim, off")
        set_config_value("keyword_search_mode", args.keyword_mode)
    if args.semantic_mode:
        if args.semantic_mode not in ("auto", "on", "off"):
            raise SystemExit("semantic_search_mode must be one of auto, on, off")
        set_config_value("semantic_search_mode", args.semantic_mode)

    print("Saved search mode settings to config.json")
    config_show(args)


def _search_profile_overrides_from_args(args):
    overrides = _parse_json_object(args.overrides, "--overrides")

    flag_values = {
        "keyword_backend": args.keyword_backend,
        "semantic_backend": args.semantic_backend,
        "max_search_candidates": args.max_candidates,
        "reranker_model": args.reranker_model,
    }
    if args.max_candidates is not None and args.max_candidates < 1:
        raise SystemExit("--max-candidates must be a positive integer.")

    for key, value in flag_values.items():
        if value is not None:
            overrides[key] = value

    if args.query_expansion_type is not None:
        overrides["query_expansion_type"] = args.query_expansion_type
        if args.query_expansion_type == "none":
            overrides["query_expansion_enabled"] = False
        elif "query_expansion_enabled" not in overrides:
            overrides["query_expansion_enabled"] = True

    if args.enable_query_expansion:
        overrides["query_expansion_enabled"] = True
    if args.disable_query_expansion:
        overrides["query_expansion_enabled"] = False

    if args.enable_reranker:
        overrides["reranker_enabled"] = True
    if args.disable_reranker:
        overrides["reranker_enabled"] = False

    return overrides


def _local_search_profiles_payload():
    from api.config import load_config
    from api.search_backends import KEYWORD_BACKENDS, SEMANTIC_BACKENDS
    from api.search_profiles import PROFILES, RERANKER_MODELS

    config = load_config()
    return {
        "current_profile": config.get("search_profile", "balanced"),
        "profiles": PROFILES,
        "available_backends": {
            "keyword": list(KEYWORD_BACKENDS.keys()),
            "semantic": list(SEMANTIC_BACKENDS.keys()),
        },
        "reranker_models": RERANKER_MODELS,
    }


def config_search_profiles(args):
    if args.server:
        result = _server_get(
            args.server,
            "/config/search-profiles",
            timeout=args.timeout,
        )
    else:
        result = _local_search_profiles_payload()
    _print_json(result)


def config_set_search_profile(args):
    from api.config import load_config, set_config_value
    from api.search_profiles import (
        list_profiles,
        merge_profile_with_overrides,
        validate_manual_config,
    )

    overrides = _search_profile_overrides_from_args(args)

    if args.server:
        json_body = overrides or None
        result = _server_post(
            args.server,
            f"/config/search-profiles/{args.profile}",
            json_body=json_body,
            timeout=args.timeout,
        )
        _print_json(result)
        return

    if args.profile not in list_profiles() and args.profile != "manual":
        available = ", ".join([*list_profiles().keys(), "manual"])
        raise SystemExit(f"Unknown profile: {args.profile}. Available: {available}")

    if args.profile == "manual":
        if not overrides:
            raise SystemExit(
                "Manual profile requires overrides with at least "
                "--keyword-backend and --semantic-backend."
            )
        if not validate_manual_config(overrides):
            raise SystemExit("Invalid manual configuration. Check backend names.")
        config_to_apply = overrides
    else:
        config_to_apply = merge_profile_with_overrides(args.profile, overrides)

    for key, value in config_to_apply.items():
        set_config_value(key, value)
    set_config_value("search_profile", args.profile)

    config = load_config()
    _print_json(
        {
            "status": "updated",
            "profile": args.profile,
            "applied_config": {k: v for k, v in config_to_apply.items()},
            "full_config": {
                "search_profile": config.get("search_profile"),
                "keyword_backend": config.get("keyword_backend"),
                "semantic_backend": config.get("semantic_backend"),
                "max_search_candidates": config.get("max_search_candidates"),
                "query_expansion_enabled": config.get("query_expansion_enabled"),
                "query_expansion_type": config.get("query_expansion_type"),
                "reranker_enabled": config.get("reranker_enabled"),
                "reranker_model": config.get("reranker_model"),
            },
        }
    )


def config_set_context_size(args):
    from api.config import set_config_value

    if args.size < 1:
        raise SystemExit("context size must be a positive integer")
    set_config_value("context_size", args.size)
    print(f"Set context_size = {args.size}")
    config_show(args)


def config_enable_web_search(args):
    from api.config import set_config_value

    provider = args.provider or "duckduckgo"
    if provider not in ("duckduckgo", "brave", "google"):
        raise SystemExit("Web search provider must be duckduckgo, brave, or google")
    set_config_value("web_search_enabled", True)
    set_config_value("web_search_provider", provider)
    if args.api_key is not None:
        set_config_value("web_search_api_key", args.api_key)
    if args.search_engine_id is not None:
        set_config_value("web_search_engine_id", args.search_engine_id)

    print(f"Enabled web search with provider: {provider}")
    config_show(args)


def config_disable_web_search(args):
    from api.config import set_config_value

    set_config_value("web_search_enabled", False)
    print("Web search disabled")
    config_show(args)


def config_reset(args):
    from api.config import reset_config

    if args.server:
        result = _server_post(args.server, "/config/reset", timeout=args.timeout)
        _print_json(result)
        return

    reset_config()
    print("Reset config.json to default settings")
    config_show(args)


def config_list_models(args):
    from api.ai_client import AIClient
    from api.config import load_config

    config = load_config()
    endpoint = args.endpoint or config.get("ai_endpoint")
    if not endpoint:
        raise SystemExit("No AI endpoint configured. Provide --endpoint or configure ai_endpoint first.")

    api_key = config.get("ai_api_key")
    api_key_header = config.get("ai_api_key_header", "Authorization")
    api_key_prefix = config.get("ai_api_key_prefix", "Bearer")
    extra_headers = config.get("ai_extra_headers", {})

    models = AIClient.list_models(endpoint, api_key, api_key_header, api_key_prefix, extra_headers)
    _print_json({"endpoint": endpoint, "models": [m["id"] for m in models]})


def config_detect_local_ai(args):
    from api.ai_client import AIClient

    endpoints = AIClient.detect_local_endpoints()
    _print_json({"detected_endpoints": endpoints})


def config_command(args):
    if args.config_command == "show":
        config_show(args)
    elif args.config_command == "set-ai-endpoint":
        config_set_ai_endpoint(args)
    elif args.config_command == "set-zim-source":
        config_set_zim_source(args)
    elif args.config_command == "clear-zim-source":
        config_clear_zim_source(args)
    elif args.config_command == "set-search-modes":
        config_set_search_modes(args)
    elif args.config_command == "search-profiles":
        config_search_profiles(args)
    elif args.config_command == "set-search-profile":
        config_set_search_profile(args)
    elif args.config_command == "set-context-size":
        config_set_context_size(args)
    elif args.config_command == "enable-web-search":
        config_enable_web_search(args)
    elif args.config_command == "disable-web-search":
        config_disable_web_search(args)
    elif args.config_command == "reset":
        config_reset(args)
    elif args.config_command == "list-models":
        config_list_models(args)
    elif args.config_command == "detect-local-ai":
        config_detect_local_ai(args)
    else:
        raise SystemExit("Unknown config command")


def collections_list(args):
    from api.zim_collections import (
        get_active_collection,
        get_all_collections,
        init_collections,
    )

    init_collections()
    collections = get_all_collections()
    active = get_active_collection()
    _print_json(
        {
            "active": active["id"] if active else None,
            "collections": {
                collection_id: {
                    "name": collection["name"],
                    "description": collection["description"],
                    "category": collection["category"],
                    "path": collection["path"],
                    "file_count": len(collection.get("zim_files", [])),
                }
                for collection_id, collection in collections.items()
            },
        }
    )


def collections_show(args):
    from api.zim_collections import get_collection_with_installation_status

    collection = get_collection_with_installation_status(args.collection_id)
    if not collection:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    _print_json({"id": args.collection_id, **collection})


def collections_files(args):
    from api.zim_collections import list_collection_files

    files = list_collection_files(args.collection_id)
    if files is None:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    _print_json(
        {"collection_id": args.collection_id, "files": files, "file_count": len(files)}
    )


def collections_create(args):
    from api.zim_collections import create_custom_collection

    try:
        collection = create_custom_collection(
            args.collection_id,
            name=args.name,
            description=args.description or "",
            zim_paths=args.zim_paths or [],
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    _print_json(
        {
            "status": "created",
            "collection_id": args.collection_id,
            "name": collection["name"],
            "path": collection["path"],
            "file_count": len(collection.get("zim_files", [])),
        }
    )


def collections_update(args):
    from api.zim_collections import update_collection

    if args.name is None and args.description is None:
        raise SystemExit("Provide --name and/or --description.")
    try:
        collection = update_collection(args.collection_id, args.name, args.description)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not collection:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    _print_json(
        {
            "status": "updated",
            "collection_id": args.collection_id,
            "name": collection["name"],
            "description": collection["description"],
            "path": collection["path"],
            "file_count": len(collection.get("zim_files", [])),
        }
    )


def collections_delete(args):
    from api.zim_collections import delete_custom_collection

    try:
        deleted = delete_custom_collection(args.collection_id)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not deleted:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    _print_json({"status": "deleted", "collection_id": args.collection_id})


def collections_add_files(args):
    from api.zim_collections import add_files_to_collection

    try:
        collection = add_files_to_collection(args.collection_id, args.zim_paths)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not collection:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    _print_json(
        {
            "status": "updated",
            "collection_id": args.collection_id,
            "files": collection.get("zim_files", []),
            "file_count": len(collection.get("zim_files", [])),
        }
    )


def collections_remove_files(args):
    from api.zim_collections import remove_files_from_collection

    try:
        collection = remove_files_from_collection(
            args.collection_id, file_names=args.files
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not collection:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    _print_json(
        {
            "status": "updated",
            "collection_id": args.collection_id,
            "files": collection.get("zim_files", []),
            "file_count": len(collection.get("zim_files", [])),
        }
    )


def collections_ingest(args):
    from api.multi_ingest import run_multi_ingest
    from api.zim_collections import (
        get_collection,
        list_collection_zim_paths,
        set_active_collection,
    )

    collection = get_collection(args.collection_id)
    if not collection:
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    if not args.output_name:
        raise SystemExit("Provide --output-name to name the vector database.")

    zim_files = collection.get("zim_files", [])
    if args.index:
        selected = []
        for index in args.index:
            if index < 0 or index >= len(zim_files):
                raise SystemExit(f"File index out of range: {index}")
            selected.append(zim_files[index])
        zim_paths = [entry["path"] for entry in selected if "path" in entry]
    else:
        zim_paths = list_collection_zim_paths(args.collection_id)

    if not zim_paths:
        raise SystemExit(f"No ZIM files found in collection '{args.collection_id}'.")

    result = run_multi_ingest(zim_paths, args.output_name)
    set_active_collection(args.collection_id)
    _print_json(
        {**result, "collection_id": args.collection_id, "db_name": args.output_name}
    )


def collections_use(args):
    from api.zim_collections import get_active_collection, set_active_collection

    if not set_active_collection(args.collection_id):
        raise SystemExit(f"Collection '{args.collection_id}' not found")
    active = get_active_collection()
    _print_json({"status": "active", "active": active["id"] if active else None})


def collections_active(args):
    from api.zim_collections import get_active_collection

    active = get_active_collection()
    _print_json(
        {
            "active": active["id"] if active else None,
            "collection": active["collection"] if active else None,
        }
    )


def collections_reset(args):
    from api.zim_collections import reset_collections

    if args.server:
        result = _server_post(
            args.server,
            "/collections/reset",
            params={"delete_folders": not args.keep_folders},
            timeout=args.timeout,
        )
        _print_json(result)
        return

    result = reset_collections(delete_folders=not args.keep_folders)
    _print_json(result)


def collections_command(args):
    if args.collections_command == "list":
        collections_list(args)
    elif args.collections_command == "show":
        collections_show(args)
    elif args.collections_command == "files":
        collections_files(args)
    elif args.collections_command == "create":
        collections_create(args)
    elif args.collections_command == "update":
        collections_update(args)
    elif args.collections_command == "delete":
        collections_delete(args)
    elif args.collections_command == "add-files":
        collections_add_files(args)
    elif args.collections_command == "remove-files":
        collections_remove_files(args)
    elif args.collections_command == "ingest":
        collections_ingest(args)
    elif args.collections_command == "use":
        collections_use(args)
    elif args.collections_command == "active":
        collections_active(args)
    elif args.collections_command == "reset":
        collections_reset(args)
    else:
        raise SystemExit("Unknown collections command")


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Serve - ZIM-based retrieval augmented proxy for OpenAI-compatible AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start server command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the Tensor Serve server"
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    start_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    start_parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Automatically find an available port if the specified port is in use"
    )
    start_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    # ZIM management command
    zim_parser = subparsers.add_parser(
        "zim",
        help="Manage ZIM files"
    )
    zim_subparsers = zim_parser.add_subparsers(dest="subcommand", help="ZIM subcommands")

    # ZIM list
    zim_subparsers.add_parser("list", help="List available category ZIM files")

    # ZIM status
    zim_status_parser = zim_subparsers.add_parser("status", help="Show installation status")
    zim_status_parser.add_argument(
        "category",
        nargs="?",
        help="Specific category to check (Research, Learning, Literature, Coding)"
    )

    # ZIM install
    zim_install_parser = zim_subparsers.add_parser("install", help="Install a ZIM file by ID")
    zim_install_parser.add_argument("file_id", help="File ID to install")

    # ZIM uninstall
    zim_uninstall_parser = zim_subparsers.add_parser("uninstall", help="Uninstall a ZIM file")
    zim_uninstall_parser.add_argument("file_id", help="File ID to uninstall")

    # ZIM install-category
    zim_install_cat_parser = zim_subparsers.add_parser(
        "install-category", help="Interactively install files for a category"
    )
    zim_install_cat_parser.add_argument(
        "category",
        help="Category ID (Research, Learning, Literature, Coding)"
    )

    # ZIM install-devdocs
    zim_subparsers.add_parser(
        "install-devdocs",
        help="Browse and install devdocs entries from the full Kiwix catalog"
    )

    # ZIM clean
    zim_subparsers.add_parser(
        "clean",
        help="Remove working files (*.index, *.pkl, *.bm25, __pycache__)"
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="View and update Tensor Serve configuration"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Configuration commands")

    config_subparsers.add_parser("show", help="Show current configuration")

    ai_parser = config_subparsers.add_parser(
        "set-ai-endpoint",
        help="Configure the local AI endpoint and optional model/API key"
    )
    ai_parser.add_argument("--endpoint", required=True, help="AI endpoint URL, e.g. http://localhost:11434")
    ai_parser.add_argument("--model", help="AI model name to use")
    ai_parser.add_argument("--provider", help="Provider name (default: openai-compatible)")
    ai_parser.add_argument("--api-key", help="API key for the AI endpoint")
    ai_parser.add_argument("--api-key-header", help="Header name for API key (default Authorization)")
    ai_parser.add_argument("--api-key-prefix", help="Header prefix for API key (default Bearer)")
    ai_parser.add_argument(
        "--extra-headers",
        help="JSON object of additional headers, e.g. '{\"X-Api-Key\": \"value\"}'",
    )

    zim_source_parser = config_subparsers.add_parser(
        "set-zim-source",
        help="Set a custom folder for local ZIM source files"
    )
    zim_source_parser.add_argument("path", help="Path to the ZIM source folder")

    config_subparsers.add_parser(
        "clear-zim-source",
        help="Reset ZIM file source back to the default local zim_files folder"
    )

    search_parser = config_subparsers.add_parser(
        "set-search-modes",
        help="Configure keyword and semantic search modes"
    )
    search_parser.add_argument(
        "--keyword-mode",
        choices=["auto", "web", "zim", "off"],
        help="Keyword search mode"
    )
    search_parser.add_argument(
        "--semantic-mode",
        choices=["auto", "on", "off"],
        help="Semantic search mode"
    )

    search_profiles_parser = config_subparsers.add_parser(
        "search-profiles",
        help="List available search profiles and backend options"
    )
    search_profiles_parser.add_argument(
        "--server",
        help="Read profiles from a running Tensor Serve server instead of local files"
    )
    search_profiles_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    set_profile_parser = config_subparsers.add_parser(
        "set-search-profile",
        help="Apply a search profile with optional backend/query/reranker overrides"
    )
    set_profile_parser.add_argument(
        "profile",
        choices=["lightweight", "balanced", "production", "manual"],
        help="Search profile to apply"
    )
    set_profile_parser.add_argument(
        "--server",
        help="Apply profile through a running Tensor Serve server instead of local files"
    )
    set_profile_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )
    set_profile_parser.add_argument(
        "--overrides",
        help="JSON object of raw profile overrides, e.g. '{\"max_search_candidates\": 200}'"
    )
    set_profile_parser.add_argument(
        "--keyword-backend",
        choices=["bm25_okapi", "bm25_plus"],
        help="Keyword backend override"
    )
    set_profile_parser.add_argument(
        "--semantic-backend",
        choices=["faiss_flat", "faiss_ivf"],
        help="Semantic backend override"
    )
    set_profile_parser.add_argument(
        "--max-candidates",
        type=int,
        help="Maximum candidate documents considered before final top-k selection"
    )
    set_profile_parser.add_argument(
        "--query-expansion",
        "--query-expansion-type",
        dest="query_expansion_type",
        choices=["none", "prf", "entity"],
        help="Query expansion strategy; non-none values enable query expansion"
    )
    query_expansion_group = set_profile_parser.add_mutually_exclusive_group()
    query_expansion_group.add_argument(
        "--enable-query-expansion",
        action="store_true",
        help="Enable query expansion without changing the configured strategy"
    )
    query_expansion_group.add_argument(
        "--disable-query-expansion",
        action="store_true",
        help="Disable query expansion"
    )
    reranker_group = set_profile_parser.add_mutually_exclusive_group()
    reranker_group.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Enable reranking"
    )
    reranker_group.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranking"
    )
    set_profile_parser.add_argument(
        "--reranker-model",
        choices=["lightweight", "balanced"],
        help="Reranker model variant"
    )

    context_parser = config_subparsers.add_parser(
        "set-context-size",
        help="Set the number of context documents to include"
    )
    context_parser.add_argument("size", type=int, help="Number of context documents")

    web_search_parser = config_subparsers.add_parser(
        "enable-web-search",
        help="Enable web search for time-sensitive queries"
    )
    web_search_parser.add_argument(
        "--provider",
        choices=["duckduckgo", "brave", "google"],
        help="Web search provider"
    )
    web_search_parser.add_argument("--api-key", help="API key for Brave or Google search")
    web_search_parser.add_argument("--search-engine-id", help="Google Custom Search engine ID")

    config_subparsers.add_parser(
        "disable-web-search",
        help="Disable web search and use offline ZIM content only"
    )

    config_reset_parser = config_subparsers.add_parser(
        "reset",
        help="Reset config.json to default settings"
    )
    config_reset_parser.add_argument(
        "--server",
        help="Reset configuration through a running Tensor Serve server instead of local files"
    )
    config_reset_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    models_parser = config_subparsers.add_parser(
        "list-models",
        help="Probe the configured AI endpoint and list available models"
    )
    models_parser.add_argument("--endpoint", help="Optional endpoint URL to probe")

    config_subparsers.add_parser(
        "detect-local-ai",
        help="Detect common local AI runtimes such as Ollama or LM Studio"
    )

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Show health for a running Tensor Serve server"
    )
    health_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    health_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    # Cache command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Inspect and clear the running server's query cache"
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command", help="Cache commands"
    )
    cache_stats_parser = cache_subparsers.add_parser(
        "stats", help="Show query and embedding cache statistics"
    )
    cache_stats_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    cache_stats_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )
    cache_clear_parser = cache_subparsers.add_parser(
        "clear", help="Clear query and embedding caches"
    )
    cache_clear_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    cache_clear_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    # Cleanup command
    clean_parser = subparsers.add_parser(
        "clean",
        aliases=["cleanup"],
        help="Clean generated vector DB files through the running server"
    )
    clean_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    clean_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    # Full reset command
    reset_parser = subparsers.add_parser(
        "reset",
        aliases=["clean-all"],
        help="Reset generated DB files, caches, collections, and configuration"
    )
    reset_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    reset_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )
    reset_parser.add_argument(
        "--keep-collection-folders",
        action="store_true",
        help="Reset collection metadata but keep matching legacy collection folders"
    )

    # Ingestion command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest ZIM files or folders into a vector database"
    )
    ingest_parser.add_argument(
        "paths",
        nargs="*",
        help="One or more .zim files or directories containing .zim files"
    )
    ingest_parser.add_argument(
        "--output-name", "-o",
        required=True,
        help="Vector database name to write"
    )
    ingest_parser.add_argument(
        "--source-folder",
        action="store_true",
        help="Ingest every .zim file from the configured ZIM source folder"
    )
    ingest_parser.add_argument(
        "--collection",
        help="Ingest every .zim file from a named collection"
    )

    # Vector database command
    db_parser = subparsers.add_parser(
        "db",
        help="List, inspect, and load vector databases"
    )
    db_subparsers = db_parser.add_subparsers(
        dest="db_command", help="Vector database commands"
    )

    db_subparsers.add_parser("list", help="List local vector databases")

    db_show_parser = db_subparsers.add_parser(
        "show", help="Show local vector database files"
    )
    db_show_parser.add_argument("name", help="Vector database name")

    db_load_parser = db_subparsers.add_parser(
        "load",
        aliases=["use"],
        help="Load a vector database into a running Tensor Serve server"
    )
    db_load_parser.add_argument("name", help="Vector database name")
    db_load_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    db_load_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    db_status_parser = db_subparsers.add_parser(
        "status",
        help="Show the running server's loaded database status"
    )
    db_status_parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Tensor Serve server URL (default: http://localhost:8000)"
    )
    db_status_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    # Collections command
    collections_parser = subparsers.add_parser(
        "collections",
        aliases=["collection"],
        help="Create, update, delete, and ingest ZIM collections"
    )
    collections_subparsers = collections_parser.add_subparsers(
        dest="collections_command", help="Collection commands"
    )

    collections_subparsers.add_parser("list", help="List collections")
    collections_subparsers.add_parser("active", help="Show the active collection")

    collections_reset_parser = collections_subparsers.add_parser(
        "reset",
        help="Reset collection metadata and remove matching legacy collection folders"
    )
    collections_reset_parser.add_argument(
        "--keep-folders",
        action="store_true",
        help="Reset metadata but keep matching legacy collection folders"
    )
    collections_reset_parser.add_argument(
        "--server",
        help="Reset collections through a running Tensor Serve server instead of local files"
    )
    collections_reset_parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds"
    )

    collections_show_parser = collections_subparsers.add_parser(
        "show", help="Show one collection"
    )
    collections_show_parser.add_argument("collection_id", help="Collection ID")

    collections_files_parser = collections_subparsers.add_parser(
        "files", help="List files in a collection"
    )
    collections_files_parser.add_argument("collection_id", help="Collection ID")

    collections_create_parser = collections_subparsers.add_parser(
        "create", help="Create a collection"
    )
    collections_create_parser.add_argument("collection_id", help="Collection ID")
    collections_create_parser.add_argument("--name", help="Display name")
    collections_create_parser.add_argument("--description", help="Description")
    collections_create_parser.add_argument(
        "--zim-path",
        dest="zim_paths",
        action="append",
        help="Existing .zim file or directory to reference; repeat for multiple inputs"
    )

    collections_update_parser = collections_subparsers.add_parser(
        "update", help="Update collection metadata"
    )
    collections_update_parser.add_argument("collection_id", help="Collection ID")
    collections_update_parser.add_argument("--name", help="Display name")
    collections_update_parser.add_argument("--description", help="Description")

    collections_delete_parser = collections_subparsers.add_parser(
        "delete", help="Delete a collection"
    )
    collections_delete_parser.add_argument("collection_id", help="Collection ID")

    collections_add_parser = collections_subparsers.add_parser(
        "add-files", help="Add .zim files to a collection"
    )
    collections_add_parser.add_argument("collection_id", help="Collection ID")
    collections_add_parser.add_argument(
        "zim_paths",
        nargs="+",
        help="One or more existing .zim files or directories containing .zim files",
    )

    collections_remove_parser = collections_subparsers.add_parser(
        "remove-files", help="Remove .zim files from a collection by file name"
    )
    collections_remove_parser.add_argument("collection_id", help="Collection ID")
    collections_remove_parser.add_argument("files", nargs="+", help="One or more file names in the collection")

    collections_ingest_parser = collections_subparsers.add_parser(
        "ingest", help="Ingest a collection into a vector database"
    )
    collections_ingest_parser.add_argument("collection_id", help="Collection ID")
    collections_ingest_parser.add_argument(
        "--output-name", "-o",
        required=True,
        help="Vector database name to write"
    )
    collections_ingest_parser.add_argument(
        "--index",
        action="append",
        type=int,
        help="Only ingest a file by its zero-based index in the collection; repeat for multiple files"
    )

    collections_use_parser = collections_subparsers.add_parser(
        "use",
        help="Set the active collection for startup auto-load"
    )
    collections_use_parser.add_argument("collection_id", help="Collection ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "start":
        start_server(args)
    elif args.command == "zim":
        if not args.subcommand:
            zim_parser.print_help()
            return
        zim_command(args)
    elif args.command == "config":
        if not args.config_command:
            config_parser.print_help()
            return
        config_command(args)
    elif args.command == "health":
        health_command(args)
    elif args.command == "cache":
        if not args.cache_command:
            cache_parser.print_help()
            return
        cache_command(args)
    elif args.command in ("clean", "cleanup"):
        clean_command(args)
    elif args.command in ("reset", "clean-all"):
        reset_all_command(args)
    elif args.command == "ingest":
        ingest_command(args)
    elif args.command == "db":
        if not args.db_command:
            db_parser.print_help()
            return
        db_command(args)
    elif args.command in ("collections", "collection"):
        if not args.collections_command:
            collections_parser.print_help()
            return
        collections_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
