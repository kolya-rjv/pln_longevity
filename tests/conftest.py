"""Shared pytest configuration for the suite.

Marker registration, and one guard that applies to every module: no test may call
OpenAI to read a patient's text. The layers under test (MeTTa, HTTP, ETL) otherwise
want very different setups, so there are no other suite-wide fixtures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PLN_CHAT = Path(__file__).resolve().parent.parent / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: spawns a worker process or runs a real MeTTa query; still fast "
        "(a few seconds) but heavier than a pure unit test.",
    )


@pytest.fixture(autouse=True)
def _no_live_patient_extraction(monkeypatch):
    """The model reader (core.patient_extract) never reaches OpenAI from a test, even
    when a developer's .env holds a key: its one network call raises instead. Tests
    use recorded extractions. PLN_LIVE_EXTRACT=1 lifts the guard (the live eval,
    scripts/eval_patient_extraction.py, is a script, not a test)."""
    if os.environ.get("PLN_LIVE_EXTRACT") == "1":
        yield
        return
    from core import patient_extract

    def _refuse(self, text):
        raise RuntimeError("a test tried to call OpenAI to read a patient's text; use a recorded "
                           "extraction, or set PLN_LIVE_EXTRACT=1")

    monkeypatch.setattr(patient_extract.OpenAIExtractor, "_request", _refuse)
    yield
