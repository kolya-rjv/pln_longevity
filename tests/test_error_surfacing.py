"""An upstream or runtime failure must not look like a successful empty answer.

The 2026-09-18 API evaluation sent two questions whose ontology selection
overflowed the model's context window. Both came back HTTP 200 with
`intent: "clarification"`, `validation_valid: true`, `pln_status: "empty"` and
the OpenAI error text buried in `answer` — a shape indistinguishable from "the
KB has nothing to say about this", so "a client filtering on status would never
notice".

Run from the repository root:
    pytest tests/test_error_surfacing.py -q
"""
from __future__ import annotations

import asyncio
import sys
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
import core.executor as executor_module  # noqa: E402
import core.llm_translator as translator_module  # noqa: E402
from core.llm_translator import ERROR_CODES, TranslationResult, _classify_api_error  # noqa: E402
from core.pln_runner import PLNRunResult  # noqa: E402
from ontology.registry import OntologyRegistry  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_api(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)
    monkeypatch.setattr(
        api_module, "_build_context", lambda selected: (OntologyRegistry(), {})
    )
    monkeypatch.setattr(api_module, "build_system_prompt", lambda registry, raw, inventory=None: "prompt")


def _post(path: str, payload: dict):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post(path, json=payload)

    return asyncio.run(send())


def _failed(code: str, message: str = "upstream said no") -> TranslationResult:
    return TranslationResult(
        metta_query="",
        explanation="",
        intent="error",
        requires_pln_inference=False,
        confidence_filter=0.0,
        error=message,
        error_code=code,
    )


# ── the classifier ───────────────────────────────────────────────────────────

def test_a_context_length_400_is_told_apart_from_any_other_400():
    """The caller can fix a too-big prompt; they cannot fix an upstream outage."""
    import openai

    overflow = openai.BadRequestError(
        "too long",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body={"error": {"code": "context_length_exceeded", "message": "..."}},
    )
    other = openai.BadRequestError(
        "nope",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body={"error": {"code": "invalid_value", "message": "..."}},
    )
    assert _classify_api_error(overflow) == "context_length_exceeded"
    assert _classify_api_error(other) == "upstream_error"


def test_a_failed_translation_is_never_labelled_clarification(monkeypatch):
    """`clarification` is a real answer; a failure must not borrow it."""
    monkeypatch.setattr(translator_module, "OPENAI_API_KEY", "")
    result = translator_module.translate("q", "prompt", [], "gpt-4o", 0.0)
    assert not result.ok
    assert result.intent == "error"
    assert result.error_code == "missing_api_key"
    assert result.error_code in ERROR_CODES


# ── each failure class gets its own status ───────────────────────────────────

@pytest.mark.parametrize(
    "code, status",
    [
        ("missing_api_key", 503),
        ("auth", 503),
        ("rate_limit", 429),
        ("context_length_exceeded", 413),
        ("timeout", 504),
        ("connection", 502),
        ("upstream_error", 502),
        ("bad_json", 502),
    ],
)
def test_translation_failures_are_not_http_200(monkeypatch, code, status):
    monkeypatch.setattr(api_module, "translate", Mock(return_value=_failed(code)))
    run_query = Mock()
    monkeypatch.setattr(api_module, "run_query", run_query)

    response = _post("/query", {"message": "does metformin extend human lifespan?"})

    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["stage"] == "translation"
    run_query.assert_not_called()      # no PLN work on a dead translation


def test_rate_limiting_tells_the_caller_when_to_retry(monkeypatch):
    monkeypatch.setattr(api_module, "translate", Mock(return_value=_failed("rate_limit")))
    response = _post("/query", {"message": "hello"})
    assert response.status_code == 429
    assert response.headers.get("retry-after")


# ── PLN execution failures too ───────────────────────────────────────────────

def test_a_runtime_error_is_a_502_not_an_empty_result(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "run_query",
        Mock(return_value=PLNRunResult(
            status="error", mode="runtime",
            error="hyperon exploded", error_code="runtime_error",
        )),
    )
    response = _post("/metta/run", {"metta_query": "!(predict-risk-patient &self Patient001)"})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "runtime_error"


def test_a_missing_drugage_build_is_503_service_not_ready(monkeypatch, tmp_path):
    from core import drugage_router

    monkeypatch.setattr(drugage_router, "BUILD_DRUGAGE", tmp_path / "absent.metta")
    response = _post("/drugage/rank", {"compounds": ["Rapamycin"]})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "drugage_build_missing"
    assert "run_etl.sh" in detail["message"]


def test_a_translation_failure_keeps_its_payload_for_debugging(monkeypatch):
    failure = _failed("upstream_error", "OpenAI API error: Error code: 500")
    failure.usage = {"prompt_tokens": 61_000, "completion_tokens": 0, "total_tokens": 61_000}
    monkeypatch.setattr(api_module, "translate", Mock(return_value=failure))

    detail = _post("/query", {"message": "hi"}).json()["detail"]
    assert detail["message"] == "OpenAI API error: Error code: 500"
    assert detail["usage"]["prompt_tokens"] == 61_000


# ── and the prompt that would not have fitted is never sent ──────────────────

def test_an_oversized_prompt_is_refused_before_the_api_call(monkeypatch):
    """The two context-length failures cost a round trip and a bill each."""
    translate = Mock()
    monkeypatch.setattr(api_module, "translate", translate)
    monkeypatch.setattr(
        api_module, "build_system_prompt", lambda registry, raw, inventory=None: "x" * 2_000_000
    )

    response = _post("/query", {"message": "which genes drive senescence?"})

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "prompt_too_large"
    assert detail["estimated_tokens"] > detail["limit_tokens"]
    translate.assert_not_called()


def test_the_size_guard_names_the_files_responsible(monkeypatch):
    monkeypatch.setattr(api_module, "translate", Mock())
    monkeypatch.setattr(
        api_module, "build_system_prompt", lambda registry, raw, inventory=None: "x" * 2_000_000
    )
    response = _post(
        "/query",
        {"message": "q", "ontology_files": ["patient_profile.metta", "hallmarks_core.metta"]},
    )
    named = {entry["file"] for entry in response.json()["detail"]["largest_selected_files"]}
    assert named == {"patient_profile.metta", "hallmarks_core.metta"}


def test_a_normal_prompt_reports_its_estimated_size(monkeypatch):
    from core.pln_runner import PLNAtomResult

    monkeypatch.setattr(api_module, "translate", Mock(return_value=TranslationResult(
        metta_query="!(match &self $x $x)", explanation="", intent="inference",
        requires_pln_inference=True, confidence_filter=0.0,
    )))
    monkeypatch.setattr(api_module, "run_query", Mock(return_value=PLNRunResult(
        status="ok", results=[PLNAtomResult("(answer)")], mode="runtime",
    )))
    monkeypatch.setattr(api_module, "log_turn", Mock())

    body = _post("/query", {"message": "hello"}).json()
    assert body["prompt_tokens_estimate"] > 0
    assert body["error"] is None and body["error_code"] is None
