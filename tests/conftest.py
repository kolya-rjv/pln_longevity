"""Shared pytest configuration for the suite.

Only marker registration for now — the suite deliberately has no fixtures that
apply to every module, because the layers under test (MeTTa, HTTP, ETL) want
very different setups.
"""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: spawns a worker process or runs a real MeTTa query; still fast "
        "(a few seconds) but heavier than a pure unit test.",
    )
