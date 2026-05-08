#!/usr/bin/env python3
"""
Tensor Serve ZIM File Manager CLI

Usage:
    python -m tensor_serve zim list                    # List available category files
    python -m tensor_serve zim status                  # Show installation status
    python -m tensor_serve zim status <category>       # Status for a category
    python -m tensor_serve zim install <file_id>       # Install a file by ID
    python -m tensor_serve zim uninstall <file_id>     # Uninstall a file
    python -m tensor_serve zim install-category <category>  # Interactive category install
    python -m tensor_serve zim install-devdocs         # Install devdocs from full catalog
    python -m tensor_serve zim clean                   # Remove generated index files and caches
"""

import argparse
import glob
import os
import shutil
import sys

import questionary

from api.zim_downloader import (
    bytes_to_human,
    download_file,
    get_category_installation_status,
    get_zim_source_folder,
    is_file_installed,
    list_available_files,
    list_devdocs_catalog,
    list_installed_files,
    uninstall_file,
)


def print_available():
    """Print available files grouped by category."""
    available = list_available_files()

    print("\n" + "=" * 80)
    print("AVAILABLE ZIM FILES FOR TENSOR SERVE".center(80))
    print("=" * 80)

    for category_id, files in available.items():
        print(f"\n📚 {category_id}")
        print("-" * 80)
        for file_info in files:
            print(f"  ID:          {file_info['id']}")
            print(f"  Name:        {file_info['name']}")
            print(f"  Description: {file_info['description']}")
            print(f"  Size:        {file_info['size']}")
            print()


def print_status(category_id=None):
    """Print installation status."""
    if category_id:
        status = get_category_installation_status(category_id)
        if "error" in status:
            print(f"✗ {status['error']}")
            return

        print(f"\n{'=' * 80}")
        print(f"Installation Status: {category_id}".center(80))
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
                "\nNo files installed yet. Use 'python -m tensor_serve zim list' to see available files."
            )
            return

        for file_id, info in installed.items():
            print(f"\n✓ {info['title']} ({file_id})")
            print(f"  Size: {info['size']}")
            print(f"  Path: {info['path']}")


def interactive_install_category(category_id):
    """Interactively install files for a curated category."""
    status = get_category_installation_status(category_id)

    if "error" in status:
        print(f"✗ {status['error']}")
        return

    print(f"\n{'=' * 80}")
    print(f"INSTALL {category_id} CATEGORY".center(80))
    print("=" * 80)

    uninstalled = [f for f in status["files"] if not f["installed"]]

    if not uninstalled:
        print(f"\n✓ All files for '{category_id}' are already installed!")
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
    print("  • Build artifacts (build/, dist/, *.egg-info/)")
    print("  • Configuration file (config.json - auto-generated on startup)")
    print("  • Collection metadata (collections.json - auto-generated on startup)")
    print("  • ZIM manifest (zim_manifest.json - auto-rebuilt on next scan)")
    print("\nThis will NOT remove:")
    print("  • ZIM files in the ZIM source folder")
    print(f"  • (Located in: {get_zim_source_folder()}/)\n")

    confirm = questionary.confirm("Proceed with cleanup?").ask()
    if not confirm:
        print("Cancelled.")
        return

    removed = []

    for pattern in ("*.index", "*.pkl", "*.bm25"):
        for path in sorted(glob.glob(pattern)):
            try:
                os.remove(path)
                removed.append(path)
            except Exception as e:
                print(f"  ✗ Error removing {path}: {e}")

    if os.path.isdir("__pycache__"):
        try:
            shutil.rmtree("__pycache__")
            removed.append("__pycache__/")
        except Exception as e:
            print(f"  ✗ Error removing __pycache__/: {e}")

    # Remove build artifacts
    for directory in ("build", "dist"):
        if os.path.isdir(directory):
            try:
                shutil.rmtree(directory)
                removed.append(f"{directory}/")
            except Exception as e:
                print(f"  ✗ Error removing {directory}/: {e}")

    # Remove .egg-info directories
    for egg_info_dir in sorted(glob.glob("*.egg-info")):
        try:
            shutil.rmtree(egg_info_dir)
            removed.append(f"{egg_info_dir}/")
        except Exception as e:
            print(f"  ✗ Error removing {egg_info_dir}/: {e}")

    # Remove auto-generated configuration and metadata files
    for filename in ("zim_manifest.json", "config.json", "collections.json"):
        if os.path.exists(filename):
            try:
                os.remove(filename)
                removed.append(filename)
            except Exception as e:
                print(f"  ✗ Error removing {filename}: {e}")

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
    subparsers.add_parser("list", help="List available category ZIM files")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show installation status")
    status_parser.add_argument(
        "category",
        nargs="?",
        help="Specific category to check (Research, Learning, Literature, Coding)",
    )

    # Install command
    install_parser = subparsers.add_parser("install", help="Install a ZIM file by ID")
    install_parser.add_argument("file_id", help="File ID to install")

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall a ZIM file")
    uninstall_parser.add_argument("file_id", help="File ID to uninstall")

    # Install category command
    install_category_parser = subparsers.add_parser(
        "install-category", help="Interactively install files for a category"
    )
    install_category_parser.add_argument(
        "category",
        help="Category ID (Research, Learning, Literature, Coding)",
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
        print_status(args.category)

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

    elif args.command == "install-category":
        interactive_install_category(args.category)

    elif args.command == "install-devdocs":
        interactive_install_devdocs()

    elif args.command == "clean":
        clean_working_files()


if __name__ == "__main__":
    main()
