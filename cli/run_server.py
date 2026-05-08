#!/usr/bin/env python3
"""
Tensor Serve - Server startup script with configurable port and auto-selection
"""

import argparse
import os
import socket
import sys
from contextlib import closing


def find_free_port(start_port=8000, max_attempts=100):
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port after checking {max_attempts} ports starting from {start_port}")


def main():
    parser = argparse.ArgumentParser(description='Start Tensor Serve server')
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=int(os.environ.get('TENSOR_PORT', 8000)),
        help='Port to run the server on (default: 8000 or TENSOR_PORT env var)'
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('TENSOR_HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0 or TENSOR_HOST env var)'
    )
    parser.add_argument(
        '--auto-port',
        action='store_true',
        default=os.environ.get('TENSOR_AUTO_PORT', '').lower() == 'true',
        help='Automatically find an available port if the specified port is in use'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )

    args = parser.parse_args()

    # Determine the port to use
    port = args.port
    if args.auto_port:
        try:
            # Check if the requested port is available
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('', port))
            print(f"Port {port} is available.")
        except OSError:
            print(f"Port {port} is in use. Finding an available port...")
            try:
                port = find_free_port(port + 1)
                print(f"Using port {port} instead.")
            except RuntimeError as e:
                print(f"Error: {e}")
                sys.exit(1)
    else:
        # Check if port is available without auto-selection
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('', port))
        except OSError:
            print(f"Error: Port {port} is already in use. Use --auto-port to automatically find an available port.")
            sys.exit(1)

    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Please run 'pip install uvicorn' first.")
        sys.exit(1)

    print(f"Starting Tensor Serve on {args.host}:{port}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"API docs available at http://localhost:{port}/docs")

    # Start the server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
