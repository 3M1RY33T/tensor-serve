#!/usr/bin/env python3
"""
Tensor Serve ZIM File Manager CLI

Usage:
    python -m src.manage_zim list                    # List available collection files
    python -m src.manage_zim status                  # Show installation status
    python -m src.manage_zim status <collection>         # Status for specific collection
    python -m src.manage_zim install <file_id>       # Install a file by ID
    python -m src.manage_zim uninstall <file_id>     # Uninstall a file
    python -m src.manage_zim install-collection <collection> # Interactive install for collection
    python -m src.manage_zim install-devdocs         # Install devdocs from full catalog
    python -m src.manage_zim clean                   # Remove generated index files and caches
"""

import argparse
import glob
import os
import shutil
import sys

import questionary

from src.zim_downloader import (
    bytes_to_human,
    download_file,
    get_collection_installation_status,
    get_zim_source_folder,
    is_file_installed,
    list_available_files,
    list_devdocs_catalog,
    list_installed_files,
    uninstall_file,
)


def print_available():
    """Print available collection files grouped by collection."""
    available = list_available_files()

    print("\n" + "=" * 80)
    print("AVAILABLE ZIM FILES FOR TENSOR SERVE".center(80))
    print("=" * 80)

    for collection_id, files in available.items():
        print(f"\n📚 {collection_id.upper()}")
        print("-" * 80)
        for file_info in files:
            print(f"  ID:          {file_info['id']}")
            print(f"  Name:        {file_info['name']}")
            print(f"  Description: {file_info['description']}")
            print(f"  Size:        {file_info['size']}")
            print()


def print_status(collection_id=None):
    """Print installation status."""
    if collection_id:
        status = get_collection_installation_status(collection_id)
        if "error" in status:
            print(f"✗ {status['error']}")
            return

        print(f"\n{'=' * 80}")
        print(f"Installation Status: {collection_id.upper()}".center(80))
        print("=" * 80)

        for file in status["files"]:
            icon = "✓" if file["installed"] else "○"
            print(f"\n{icon} {file['name']} ({file['id']})")
            print(f"  Size: {file['size']}")
            if file["installed"]:
                if "installed_count" in file:
                    count = file["installed_count"]
                    print(f"  Installed: {count} entr{'y' if count == 1 else 'ies'}")
                elif file.get("path"):
                    print(f"  Path: {file['path']}")
    else:
        installed = list_installed_files()
        print(f"\n{'=' * 80}")
        print("INSTALLED ZIM FILES".center(80))
        print("=" * 80)

        if not installed:
            print(
                "\nNo files installed yet. Use 'python -m src.manage_zim list' to see available files."
            )
            return

        for file_id, info in installed.items():
            print(f"\n✓ {info['title']} ({file_id})")
            print(f"  Size: {info['size']}")
            print(f"  Path: {info['path']}")


def interactive_install_collection(collection_id):
    """Interactively install files for a collection."""
    status = get_collection_installation_status(collection_id)

    if "error" in status:
        print(f"✗ {status['error']}")
        return

    print(f"\n{'=' * 80}")
    print(f"INSTALL {collection_id.upper()} COLLECTION".center(80))
    print("=" * 80)

    uninstalled = [f for f in status["files"] if not f["installed"]]

    if not uninstalled:
        print(f"\n✓ All files for '{collection_id}' are already installed!")
        return

    choices = [
        questionary.Choice(
            title=f"{f['name']}  –  {f['description']}  ({f['size']})",
            value=f,
        )
        for f in uninstalled
    ]

    selected = questionary.checkbox(
        "Select files to install  (Space to toggle, ↑↓ to navigate, Enter to confirm):",
        choices=choices,
    ).ask()

    if not selected:
        print("No files selected.")
        return

    print(f"\nInstalling {len(selected)} file(s)...\n")

    for file in selected:
        download_file(file["id"], show_progress=True)
        print()


def interactive_install_devdocs():
    """Fetch the full devdocs catalog from Kiwix and interactively install entries."""
    print("\n" + "=" * 80)
    print("INSTALL DEVDOCS".center(80))
    print("=" * 80)

    print("\nFetching devdocs catalog from Kiwix (this may take a moment)...")
    catalog = list_devdocs_catalog()

    if not catalog:
        print("✗ Could not fetch devdocs catalog. Check your internet connection.")
        return

    for entry in catalog:
        entry["installed"] = is_file_installed(entry["id"])

    installed = [e for e in catalog if e["installed"]]
    uninstalled = [e for e in catalog if not e["installed"]]

    total_bytes = sum(e["size_bytes"] for e in uninstalled)
    print(f"\n  Total entries : {len(catalog)}")
    print(f"  Installed     : {len(installed)}")
    print(
        f"  Not installed : {len(uninstalled)}  ({bytes_to_human(total_bytes)} remaining)"
    )

    if not uninstalled:
        print("\n✓ All devdocs entries are already installed!")
        return

    choices = [
        questionary.Choice(
            title=f"{e['name']:<40}  {e['size']:>10}",
            value=e,
        )
        for e in uninstalled
    ]

    selected = questionary.checkbox(
        "Select devdocs to install  (Space to toggle, ↑↓ to navigate, Enter to confirm):",
        choices=choices,
    ).ask()

    if not selected:
        print("No files selected.")
        return

    selected_bytes = sum(e["size_bytes"] for e in selected)
    print(
        f"\nInstalling {len(selected)} devdocs file(s)  ({bytes_to_human(selected_bytes)} total)...\n"
    )

    failed = []
    for i, entry in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {entry['name']}")
        success = download_file(entry["id"], show_progress=True)
        if not success:
            failed.append(entry["name"])
        print()

    installed_count = len(selected) - len(failed)
    print("─" * 60)
    print(f"✓ Installed : {installed_count}")
    if failed:
        print(f"✗ Failed    : {len(failed)}")
        for name in failed:
            print(f"    • {name}")


def clean_working_files():
    """Remove generated working files, preserving collections, config, and ZIM files."""
    print("\n" + "=" * 80)
    print("CLEAN WORKING FILES".center(80))
    print("=" * 80)
    print("\nThis will remove:")
    print("  • Vector DB index files (*.index)")
    print("  • Vector DB text stores (*.pkl)")
    print("  • BM25 keyword index files (*.bm25)")
    print("  • Python bytecode cache (__pycache__/)")
    print("\nThis will NOT remove:")
    print("  • collections.json, config.json, zim_manifest.json")
    print(f"  • ZIM files in {get_zim_source_folder()}/\n")

    confirm = questionary.confirm("Proceed with cleanup?").ask()
    if not confirm:
        print("Cancelled.")
        return

    removed = []

    for pattern in ("*.index", "*.pkl", "*.bm25"):
        for path in sorted(glob.glob(pattern)):
            os.remove(path)
            removed.append(path)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")
        removed.append("__pycache__/")

    if removed:
        print(f"\n✓ Removed {len(removed)} item(s):")
        for r in removed:
            print(f"  • {r}")
    else:
        print("\n✓ Nothing to clean.")


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Serve ZIM File Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List available collection ZIM files")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show installation status")
    status_parser.add_argument(
        "collection",
        nargs="?",
        help="Specific collection to check (research, learn, literature, coding)",
    )

    # Install command
    install_parser = subparsers.add_parser("install", help="Install a ZIM file by ID")
    install_parser.add_argument("file_id", help="File ID to install")

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall a ZIM file")
    uninstall_parser.add_argument("file_id", help="File ID to uninstall")

    # Install collection command
    install_collection_parser = subparsers.add_parser(
        "install-collection", help="Interactively install files for a collection"
    )
    install_collection_parser.add_argument(
        "collection",
        help="Collection ID (research, learn, literature, coding)",
    )

    # Install devdocs command
    subparsers.add_parser(
        "install-devdocs",
        help="Browse and install devdocs entries from the full Kiwix catalog",
    )

    # Clean command
    subparsers.add_parser(
        "clean",
        help="Remove working files (*.index, *.pkl, *.bm25, __pycache__/)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        print_available()

    elif args.command == "status":
        print_status(args.collection)

    elif args.command == "install":
        if download_file(args.file_id):
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.command == "uninstall":
        if uninstall_file(args.file_id):
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.command == "install-collection":
        interactive_install_collection(args.collection)

    elif args.command == "install-devdocs":
        interactive_install_devdocs()

    elif args.command == "clean":
        clean_working_files()


if __name__ == "__main__":
    main()
