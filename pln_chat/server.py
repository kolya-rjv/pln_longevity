"""Importable combined ASGI application for production-style launchers.

Examples:
    uvicorn server:app --host 127.0.0.1 --port 7860
    python server.py
"""
from __future__ import annotations

import uvicorn

from app import create_combined_app
from config import PLN_SERVER_HOST, PLN_SERVER_PORT


app = create_combined_app()


if __name__ == "__main__":
    uvicorn.run(app, host=PLN_SERVER_HOST, port=PLN_SERVER_PORT)
