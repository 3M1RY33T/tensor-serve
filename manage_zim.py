#!/usr/bin/env python3
"""
Tensor Serve ZIM File Manager CLI

Usage:
    python manage_zim.py list                    # List available preset files
    python manage_zim.py status                  # Show installation status
    python manage_zim.py status <tuning>         # Status for specific tuning
    python manage_zim.py install <file_id>       # Install a file by ID
    python manage_zim.py uninstall <file_id>     # Uninstall a file
    python manage_zim.py install-tuning <tuning> # Interactive install for tuning
    python manage_zim.py install-devdocs         # Install devdocs from full catalog
"""

import argparse
import sys

import questionary

from zim_downloader import (
    bytes_to_human,
    download_file,
    get_installed_files_for_tuning,
    get_tuning_installation_status,
    is_file_installed,
    list_available_files,
    list_devdocs_catalog,
    list_installed_files,
    uninstall_file,
)


def print_available():
    """Print available preset files grouped by tuning."""
    available = list_available_files()

    print("\n" + "=" * 80)
    print("AVAILABLE ZIM FILES FOR TENSOR SERVE".center(80))
    print("=" * 80)

    for tuning_id, files in available.items():
        print(f"\n📚 {tuning_id.upper()}")
        print("-" * 80)
        for file_info in files:
            print(f"  ID:          {file_info['id']}")
            print(f"  Name:        {file_info['name']}")
            print(f"  Description: {file_info['description']}")
            print(f"  Size:        {file_info['size']}")
            print()


def print_status(tuning_id=None):
    """Print installation status."""
    if tuning_id:
        status = get_tuning_installation_status(tuning_id)
        if "error" in status:
            print(f"✗ {status['error']}")
            return

        print(f"\n{'=' * 80}")
        print(f"Installation Status: {tuning_id.upper()}".center(80))
        print("=" * 80)

        for file in status["files"]:
            icon = "✓" if file["installed"] else "○"
            print(f"\n{icon} {file['name']} ({file['id']})")
            print(f"  Size: {file['size']}")
            if file["installed"]:
                print(f"  Path: {file['path']}")
    else:
        installed = list_installed_files()
        print(f"\n{'=' * 80}")
        print("INSTALLED ZIM FILES".center(80))
        print("=" * 80)

        if not installed:
            print(
                "\nNo files installed yet. Use 'python manage_zim.py list' to see available files."
            )
            return

        for file_id, info in installed.items():
            print(f"\n✓ {info['title']} ({file_id})")
            print(f"  Size: {info['size']}")
            print(f"  Path: {info['path']}")


def interactive_install_tuning(tuning_id):
    """Interactively install files for a tuning."""
    status = get_tuning_installation_status(tuning_id)

    if "error" in status:
        print(f"✗ {status['error']}")
        return

    print(f"\n{'=' * 80}")
    print(f"INSTALL {tuning_id.upper()} TUNING".center(80))
    print("=" * 80)

    uninstalled = [f for f in status["files"] if not f["installed"]]

    if not uninstalled:
        print(f"\n✓ All files for '{tuning_id}' are already installed!")
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

    # Split into installed / uninstalled
    for entry in catalog:
        entry["installed"] = is_file_installed(entry["id"])

    installed = [e for e in catalog if e["installed"]]
    uninstalled = [e for e in catalog if not e["installed"]]

    # ── summary ──────────────────────────────────────────────────────────────
    total_bytes = sum(e["size_bytes"] for e in uninstalled)
    print(f"\n  Total entries : {len(catalog)}")
    print(f"  Installed     : {len(installed)}")
    print(
        f"  Not installed : {len(uninstalled)}  ({bytes_to_human(total_bytes)} remaining)"
    )

    if not uninstalled:
        print("\n✓ All devdocs entries are already installed!")
        return

    # ── checkbox selection ────────────────────────────────────────────────────
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

    # ── download ─────────────────────────────────────────────────────────────
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

    # ── final report ─────────────────────────────────────────────────────────
    installed_count = len(selected) - len(failed)
    print("─" * 60)
    print(f"✓ Installed : {installed_count}")
    if failed:
        print(f"✗ Failed    : {len(failed)}")
        for name in failed:
            print(f"    • {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Serve ZIM File Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List available preset ZIM files")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show installation status")
    status_parser.add_argument(
        "tuning",
        nargs="?",
        help="Specific tuning to check (research, learn, literature, coding)",
    )

    # Install command
    install_parser = subparsers.add_parser("install", help="Install a ZIM file by ID")
    install_parser.add_argument("file_id", help="File ID to install")

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall a ZIM file")
    uninstall_parser.add_argument("file_id", help="File ID to uninstall")

    # Install tuning command
    install_tuning_parser = subparsers.add_parser(
        "install-tuning", help="Interactively install preset files for a tuning"
    )
    install_tuning_parser.add_argument(
        "tuning",
        help="Tuning ID (research, learn, literature, coding)",
    )

    # Install devdocs command
    subparsers.add_parser(
        "install-devdocs",
        help="Browse and install devdocs entries from the full Kiwix catalog",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        print_available()

    elif args.command == "status":
        print_status(args.tuning)

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

    elif args.command == "install-tuning":
        interactive_install_tuning(args.tuning)

    elif args.command == "install-devdocs":
        interactive_install_devdocs()


if __name__ == "__main__":
    main()
