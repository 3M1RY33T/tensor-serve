#!/usr/bin/env python3
"""
Tensor Serve ZIM File Manager CLI

Usage:
    python manage_zim.py list                    # List available files
    python manage_zim.py status                  # Show installation status
    python manage_zim.py status <tuning>         # Status for specific tuning
    python manage_zim.py install <file_id>       # Install a file
    python manage_zim.py uninstall <file_id>     # Uninstall a file
    python manage_zim.py install-tuning <tuning> # Interactive install for tuning
"""

import sys
import argparse
from zim_downloader import (
    list_available_files,
    list_installed_files,
    download_file,
    uninstall_file,
    get_tuning_installation_status,
    get_installed_files_for_tuning,
    is_file_installed,
)


def print_available():
    """Print available files grouped by tuning."""
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
            print("\nNo files installed yet. Use 'python manage_zim.py list' to see available files.")
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

    print(f"\nAvailable files to install:")
    for i, file in enumerate(uninstalled, 1):
        print(f"\n{i}. {file['name']} ({file['id']})")
        print(f"   Description: {file['description']}")
        print(f"   Size: {file['size']}")

    print(f"\nSelect files to install (comma-separated numbers, or 'all'):")
    user_input = input("> ").strip().lower()

    if user_input == "all":
        selected = uninstalled
    else:
        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            selected = [uninstalled[i] for i in indices if 0 <= i < len(uninstalled)]
        except (ValueError, IndexError):
            print("✗ Invalid selection")
            return

    if not selected:
        print("No files selected")
        return

    print(f"\nInstalling {len(selected)} file(s)...\n")

    for file in selected:
        download_file(file["id"], show_progress=True)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Serve ZIM File Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List available ZIM files")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show installation status")
    status_parser.add_argument(
        "tuning",
        nargs="?",
        help="Specific tuning to check (research, learn, literature, coding)",
    )

    # Install command
    install_parser = subparsers.add_parser("install", help="Install a ZIM file")
    install_parser.add_argument("file_id", help="File ID to install")

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall a ZIM file")
    uninstall_parser.add_argument("file_id", help="File ID to uninstall")

    # Install tuning command
    install_tuning_parser = subparsers.add_parser(
        "install-tuning", help="Interactively install files for a tuning"
    )
    install_tuning_parser.add_argument(
        "tuning",
        help="Tuning ID (research, learn, literature, coding)",
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


if __name__ == "__main__":
    main()
