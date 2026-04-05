"""
Developer launcher for the Heimdall app server (API + dashboard + analysis UI).

Automatically picks an available port and starts Uvicorn with reload enabled.
"""
from __future__ import annotations

from src.tools.dev_ui import main


if __name__ == "__main__":
    main()

