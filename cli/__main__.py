#!/usr/bin/env python3
"""
Tensor Serve - CLI interface for managing ZIM files and running the server
"""

import argparse
import json
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


def config_show(args):
    from api.config import load_config
    from api.zim_downloader import get_zim_source_folder, has_custom_zim_source_folder

    config = load_config()
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
    elif args.config_command == "set-context-size":
        config_set_context_size(args)
    elif args.config_command == "enable-web-search":
        config_enable_web_search(args)
    elif args.config_command == "disable-web-search":
        config_disable_web_search(args)
    elif args.config_command == "list-models":
        config_list_models(args)
    elif args.config_command == "detect-local-ai":
        config_detect_local_ai(args)
    else:
        raise SystemExit("Unknown config command")


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

    models_parser = config_subparsers.add_parser(
        "list-models",
        help="Probe the configured AI endpoint and list available models"
    )
    models_parser.add_argument("--endpoint", help="Optional endpoint URL to probe")

    config_subparsers.add_parser(
        "detect-local-ai",
        help="Detect common local AI runtimes such as Ollama or LM Studio"
    )

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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
