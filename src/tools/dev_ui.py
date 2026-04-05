"""
Developer launcher for the Heimdall app server.

Finds an available local port (starting at 8000 by default) and starts Uvicorn.
"""
from __future__ import annotations

import argparse
import socket
from pathlib import Path

import uvicorn


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _find_open_port(host: str, start_port: int, end_port: int) -> int:
    for port in range(start_port, end_port + 1):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{end_port}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Heimdall app server on an open port.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="First port to try (default: 8000)",
    )
    parser.add_argument(
        "--end-port",
        type=int,
        default=8100,
        help="Last port to try (default: 8100)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.end_port < args.start_port:
        raise SystemExit("--end-port must be >= --start-port")

    port = _find_open_port(args.host, args.start_port, args.end_port)
    use_reload = not args.no_reload
    print(f"Heimdall App: http://{args.host}:{port}/analysis/")
    kwargs = {
        "host": args.host,
        "port": port,
        "reload": use_reload,
    }
    if use_reload:
        project_root = Path(__file__).resolve().parents[2]
        kwargs["reload_dirs"] = [str(project_root / "src")]

    uvicorn.run(
        "src.tools.ui_server:app",
        **kwargs,
    )


if __name__ == "__main__":
    main()
