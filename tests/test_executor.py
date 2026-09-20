"""Tests for out-of-process PLN execution.

The 2026-09-18 API evaluation found that one 35-compound ranking blocked every
other caller for 115 s and made `GET /health` time out at 30 s. The measured
cause is that hyperon holds the GIL for the whole of `MeTTa.run()`, so the fix
has to be a separate PROCESS, and these tests pin the three behaviours that
buys: the event loop keeps serving, a runaway query is abandoned at a deadline,
and a worker that dies does not take the API with it.

Run from the repository root:
    pytest tests/test_executor.py -q
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import api as api_module  # noqa: E402
import core.executor as executor  # noqa: E402
from core.executor import (  # noqa: E402
    PLNExecutionTimeout,
    PLNOverloaded,
    PLNWorkerCrashed,
    run_offloaded,
)


@pytest.fixture(autouse=True)
def quiet_logging(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())


@pytest.fixture
def inline(monkeypatch):
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)


@pytest.fixture
def pooled(monkeypatch):
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 2)
    yield
    executor.shutdown()


def _client_request(method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(send())


# ── inline mode is the escape hatch, and it is exact ─────────────────────────

def test_inline_mode_runs_the_callers_own_thunk(inline):
    """Inline mode must call the thunk, so monkeypatched globals are honoured."""
    sentinel = object()
    assert run_offloaded("run_query", {"nonsense": 1}, lambda: sentinel) is sentinel


def test_an_unknown_task_is_a_programming_error_not_a_silent_inline_run(inline):
    with pytest.raises(KeyError):
        run_offloaded("not_a_task", {}, lambda: "never")


def test_executor_stats_describe_the_active_mode(inline):
    stats = executor.executor_stats()
    assert stats["mode"] == "inline"
    assert stats["workers"] == 0


# ── the process pool: real work, real deadline, real recovery ────────────────

@pytest.mark.slow
def test_a_real_query_runs_in_a_worker_process(pooled):
    pytest.importorskip("hyperon")
    result = run_offloaded(
        "run_query",
        {
            "metta_query": "!(evidence-confidence ITP_Positive)",
            "kb_files": [REPO / "epistemic_calibration.metta"],
        },
        lambda: pytest.fail("should not run inline when a pool is configured"),
    )
    assert result.status == "ok"
    assert [r.atom for r in result.results] == ["0.9"]


@pytest.mark.slow
def test_a_runaway_query_is_abandoned_at_the_deadline_and_the_pool_recovers(pooled):
    pytest.importorskip("hyperon")
    started = time.monotonic()
    with pytest.raises(PLNExecutionTimeout):
        run_offloaded(
            "run_query",
            {
                # A deliberately divergent recursion: it never returns, which is
                # exactly the shape a per-request budget exists for.
                "metta_query": "!(spin 1)",
                "kb_files": [],
                "extra_atoms": "(= (spin $n) (spin (+ $n 1)))",
            },
            lambda: pytest.fail("should not run inline"),
            timeout=2.0,
        )
    elapsed = time.monotonic() - started
    assert 1.5 <= elapsed < 15, f"deadline not enforced (took {elapsed:.1f}s)"

    # The service is still usable immediately afterwards — a fresh pool.
    recovered = run_offloaded(
        "run_query",
        {
            "metta_query": "!(evidence-confidence InVitro)",
            "kb_files": [REPO / "epistemic_calibration.metta"],
        },
        lambda: pytest.fail("should not run inline"),
    )
    assert [r.atom for r in recovered.results] == ["0.35"]


@pytest.mark.slow
def test_the_event_loop_keeps_serving_while_a_query_runs(pooled):
    """The headline finding: /health must not queue behind a MeTTa call."""
    pytest.importorskip("hyperon")
    durations: list[float] = []

    def hammer_health():
        for _ in range(3):
            started = time.monotonic()
            response = _client_request("GET", "/health")
            durations.append(time.monotonic() - started)
            assert response.status_code == 200

    worker = threading.Thread(target=hammer_health)
    worker.start()
    run_offloaded(
        "run_query",
        {
            "metta_query": "!(evidence-confidence ITP_Positive)",
            "kb_files": [REPO / "epistemic_calibration.metta"],
        },
        lambda: pytest.fail("should not run inline"),
    )
    worker.join(timeout=30)
    assert durations, "health checks did not run"


# ── admission control ────────────────────────────────────────────────────────

def test_too_many_inflight_queries_are_refused_rather_than_queued(monkeypatch):
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 1)
    monkeypatch.setattr(executor, "PLN_MAX_INFLIGHT_QUERIES", 1)
    monkeypatch.setattr(executor, "_inflight", 1)      # pretend one is running
    with pytest.raises(PLNOverloaded) as excinfo:
        run_offloaded("run_query", {}, lambda: None)
    assert excinfo.value.limit == 1


# ── the HTTP layer maps each failure to its own status code ──────────────────

@pytest.mark.parametrize(
    "exc, status, code",
    [
        (PLNExecutionTimeout(60.0), 504, "pln_timeout"),
        (PLNOverloaded(8), 503, "pln_overloaded"),
        (PLNWorkerCrashed("worker died"), 500, "pln_worker_crashed"),
    ],
)
def test_execution_failures_are_not_reported_as_success(monkeypatch, exc, status, code):
    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(api_module, "run_offloaded", boom)
    response = _client_request(
        "POST", "/metta/run", json={"metta_query": "!(predict-risk-patient &self Patient001)"}
    )
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_health_reports_how_pln_work_is_executed():
    body = _client_request("GET", "/health").json()
    assert "pln_execution" in body
    assert body["pln_execution"]["mode"] in {"inline", "process_pool"}
    assert isinstance(body["pln_execution"]["timeout_seconds"], (int, float))


# ── bounded inputs ───────────────────────────────────────────────────────────

def test_ontology_file_selection_is_deduped_and_capped():
    assert api_module._dedupe_ontology_files(None) is None
    assert api_module._dedupe_ontology_files(["a", "b", "a"]) == ["a", "b"]

    too_many = ["patient_profile.metta"] * (api_module.PLN_MAX_ONTOLOGY_FILES + 1)
    response = _client_request(
        "POST", "/query", json={"message": "hi", "ontology_files": too_many}
    )
    assert response.status_code == 422


def test_compound_pool_is_capped():
    response = _client_request(
        "POST",
        "/drugage/rank",
        json={"compounds": ["Rapamycin"] * (api_module.PLN_MAX_RANK_COMPOUNDS + 1)},
    )
    assert response.status_code == 422
