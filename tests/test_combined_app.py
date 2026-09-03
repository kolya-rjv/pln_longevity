"""Integration contract for serving Gradio and REST from one ASGI origin."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

gradio = pytest.importorskip("gradio")
httpx = pytest.importorskip("httpx")

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

import app as ui_module  # noqa: E402
import fastapi.dependencies.utils as dependency_utils  # noqa: E402
import fastapi.routing as fastapi_routing  # noqa: E402


def test_ui_and_rest_api_share_one_origin(monkeypatch):
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(fastapi_routing, "run_in_threadpool", run_inline)
    monkeypatch.setattr(dependency_utils, "run_in_threadpool", run_inline)
    combined = ui_module.create_combined_app()

    paths = [route.path for route in combined.routes]
    assert paths.index("/health") < paths.index("")

    async def fetch_both():
        transport = httpx.ASGITransport(app=combined)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get("/health"), await client.get("/")

    health, ui = asyncio.run(fetch_both())
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ui.status_code == 200
    assert "PLN Natural Language Query Interface" in ui.text
