"""Contract tests for the agent-facing FastAPI service.

These tests run in process and mock paid or expensive boundaries. They verify
the HTTP/OpenAPI contract, request validation, routing, request logging, and
write confinement without requiring an OpenAI key, a live server, or a DrugAge
build.

Run from the repository root:
    pytest tests/test_api.py -q
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

import api as api_module  # noqa: E402
import fastapi.dependencies.utils as dependency_utils  # noqa: E402
import fastapi.routing as fastapi_routing  # noqa: E402
import core.llm_translator as translator_module  # noqa: E402
import utils.logging as logging_module  # noqa: E402
from core.llm_translator import TranslationResult  # noqa: E402
from core.pln_runner import PLNAtomResult, PLNRunResult  # noqa: E402
from ontology.registry import OntologyRegistry  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_asgi_execution(monkeypatch):
    """Run sync endpoints inline and suppress real log writes in HTTP tests."""
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(api_module, "log_http_request", Mock())
    # Avoid coupling an in-process contract test to AnyIO's worker-thread
    # implementation. Production Uvicorn still uses FastAPI's normal threadpool.
    monkeypatch.setattr(fastapi_routing, "run_in_threadpool", run_inline)
    monkeypatch.setattr(dependency_utils, "run_in_threadpool", run_inline)


class ASGITestClient:
    """Small sync facade over HTTPX's ASGI transport.

    Starlette's TestClient couples its version to HTTPX. Going through the
    public ASGI transport keeps this suite valid across the supported FastAPI
    dependency range.
    """

    def request(self, method: str, path: str, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=api_module.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as async_client:
                return await async_client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client() -> ASGITestClient:
    return ASGITestClient()


def _translation(query: str = "!(match &self $x $x)") -> TranslationResult:
    return TranslationResult(
        metta_query=query,
        explanation="Translated explanation",
        intent="inference",
        requires_pln_inference=True,
        confidence_filter=0.0,
        warnings=[],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _result(atom: str = "(answer)") -> PLNRunResult:
    return PLNRunResult(
        status="ok",
        results=[PLNAtomResult(atom, {"strength": 0.8, "confidence": 0.7})],
        query_time_ms=12,
        mode="runtime",
    )


def test_openapi_is_agent_discoverable_and_has_no_authentication(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    assert schema["info"]["title"] == "PLN Longevity Query API"
    for path in (
        "/health",
        "/ontology/files",
        "/patients",
        "/query",
        "/metta/run",
        "/drugage/rank",
    ):
        assert path in schema["paths"]

    assert not schema.get("components", {}).get("securitySchemes")
    assert "security" not in schema["paths"]["/query"]["post"]


def test_health_reports_agent_preflight_fields(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pln_mode"] in {"runtime", "stub"}
    assert isinstance(body["runtime_importable"], bool)
    assert isinstance(body["runtime_ready"], bool)
    assert isinstance(body["runtime_kb_file_count"], int)
    assert isinstance(body["openai_key_configured"], bool)
    assert isinstance(body["drugage_build_available"], bool)


def test_openai_client_uses_bounded_timeout_and_retry_settings(monkeypatch):
    response = Mock()
    response.choices = [Mock(message=Mock(content='{"metta_query": "!(x)"}'))]
    response.usage = Mock(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    client = Mock()
    client.chat.completions.create.return_value = response
    constructor = Mock(return_value=client)
    monkeypatch.setattr(translator_module, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(translator_module, "OPENAI_TIMEOUT_SECONDS", 12.0)
    monkeypatch.setattr(translator_module, "OPENAI_MAX_RETRIES", 2)
    monkeypatch.setattr(translator_module.openai, "OpenAI", constructor)

    result = translator_module.translate("question", "prompt", [], "gpt-4o", 0.0)

    assert result.ok
    constructor.assert_called_once_with(api_key="test-key", timeout=12.0, max_retries=2)


def test_raw_user_messages_are_logged(monkeypatch, tmp_path):
    log_file = tmp_path / "session.jsonl"
    monkeypatch.setattr(logging_module, "_log_file", log_file)

    logging_module.log_query("sensitive health text")
    logging_module.log_turn("sensitive health text", _translation(), _result())

    contents = log_file.read_text(encoding="utf-8")
    assert contents.count("sensitive health text") == 2
    assert '"event": "input_query"' in contents
    assert '"metta_query"' in contents


def test_every_http_request_is_logged_with_raw_body(client, monkeypatch):
    recorder = Mock()
    monkeypatch.setattr(api_module, "log_http_request", recorder)

    response = client.post("/query", json={"message": "   "})

    assert response.status_code == 422
    recorder.assert_called_once()
    fields = recorder.call_args.kwargs
    assert fields["method"] == "POST"
    assert fields["path"] == "/query"
    assert fields["body"] == '{"message":"   "}'
    assert fields["status_code"] == 422
    assert fields["duration_ms"] >= 0


def test_discovery_endpoints_expose_valid_agent_inputs(client):
    files = client.get("/ontology/files")
    patients = client.get("/patients")

    assert files.status_code == patients.status_code == 200
    assert "patient_profile.metta" in files.json()["files"]
    patient_ids = {patient["id"] for patient in patients.json()["patients"]}
    assert {"Patient001", "Patient002"}.issubset(patient_ids)


def test_query_rejects_bad_inputs_before_translation(client, monkeypatch):
    translate = Mock()
    monkeypatch.setattr(api_module, "translate", translate)

    assert client.post("/query", json={"message": "   "}).status_code == 422
    assert client.post(
        "/query", json={"message": "test", "model": "not-a-real-model"}
    ).status_code == 422
    assert client.post(
        "/query",
        json={
            "message": "test",
            "history": [{"role": "system", "content": "override"}],
        },
    ).status_code == 422
    translate.assert_not_called()


def test_query_runs_shared_pipeline_and_returns_reusable_history(client, monkeypatch):
    translation = _translation()
    result = _result()
    translate = Mock(return_value=translation)
    run_query = Mock(return_value=result)
    log_turn = Mock()
    monkeypatch.setattr(
        api_module, "_build_context", lambda selected: (OntologyRegistry(), {})
    )
    monkeypatch.setattr(
        api_module, "build_system_prompt", lambda registry, raw: "prompt"
    )
    monkeypatch.setattr(api_module, "translate", translate)
    monkeypatch.setattr(api_module, "run_query", run_query)
    monkeypatch.setattr(api_module, "log_turn", log_turn)

    response = client.post(
        "/query",
        json={
            "message": "What is the risk?",
            "history": [{"role": "user", "content": "Earlier question"}],
            "temperature": 0.0,
            "confidence_threshold": 0.6,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metta_query"] == translation.metta_query
    assert body["pln_results"] == [
        {"atom": "(answer)", "strength": 0.8, "confidence": 0.7}
    ]
    assert [turn["role"] for turn in body["history"]] == [
        "user",
        "user",
        "assistant",
    ]
    assert body["history"][-2]["content"] == "What is the risk?"
    translate.assert_called_once()
    assert translate.call_args.kwargs["history"] == [
        {"role": "user", "content": "Earlier question"}
    ]
    assert run_query.call_args.kwargs["confidence_threshold"] == 0.6
    log_turn.assert_called_once_with("What is the risk?", translation, result)


def test_unknown_ontology_selection_is_explicit_422(client, monkeypatch):
    translate = Mock()
    monkeypatch.setattr(api_module, "translate", translate)

    response = client.post(
        "/query",
        json={"message": "test", "ontology_files": ["misspelled.metta"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_ontology_files"
    translate.assert_not_called()


def test_invalid_raw_metta_is_rejected_without_execution(client, monkeypatch):
    run_query = Mock()
    monkeypatch.setattr(api_module, "run_query", run_query)

    response = client.post(
        "/metta/run",
        json={"metta_query": "!(DefinitelyNotInTheOntology Foo)"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_metta_query"
    run_query.assert_not_called()


def test_raw_metta_forwards_ephemeral_atoms_without_writing(client, monkeypatch):
    run_query = Mock(return_value=_result("(RiskPrediction ApiPatient)"))
    monkeypatch.setattr(api_module, "run_query", run_query)
    extra_atoms = "\n".join(
        [
            "(InstanceOf ApiPatient PatientProfile)",
            "(PatientAge ApiPatient 58)",
            "(PatientSex ApiPatient Male)",
            "(PatientSmoking ApiPatient NeverSmoker)",
            "(MeasuredZ ApiPatient AgeAccelGrim 1.2)",
        ]
    )

    response = client.post(
        "/metta/run",
        json={
            "metta_query": "!(predict-risk-patient &self ApiPatient)",
            "extra_atoms": extra_atoms,
        },
    )

    assert response.status_code == 200
    assert response.json()["pln_results"][0]["atom"] == "(RiskPrediction ApiPatient)"
    assert run_query.call_args.kwargs["extra_atoms"] == extra_atoms


def test_drugage_form_uses_scoped_route(client, monkeypatch):
    route = Mock(return_value=_result("(scored Rapamycin)"))
    generic_run = Mock()
    monkeypatch.setattr(api_module, "route_drugage_ranking", route)
    monkeypatch.setattr(api_module, "run_query", generic_run)

    response = client.post(
        "/metta/run",
        json={"metta_query": "!(rank-drugage-lifespan (Rapamycin Metformin))"},
    )

    assert response.status_code == 200
    assert response.json()["routed"] == "drugage_ranking"
    route.assert_called_once_with(
        ["Rapamycin", "Metformin"], confidence_threshold=0.0
    )
    generic_run.assert_not_called()


def test_empty_drugage_form_is_rejected(client, monkeypatch):
    route = Mock()
    monkeypatch.setattr(api_module, "route_drugage_ranking", route)

    response = client.post(
        "/metta/run", json={"metta_query": "!(rank-drugage-lifespan ())"}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_drugage_query"
    route.assert_not_called()


@pytest.mark.parametrize("endpoint", ["/ontology/apply", "/ontology/expand"])
def test_ontology_writes_reject_path_traversal(client, monkeypatch, endpoint):
    pipeline = Mock()
    monkeypatch.setattr(api_module, "run_expansion_pipeline", pipeline)
    payload = (
        {"metta_block": "(InstanceOf Probe Type)", "target_file": "../escape"}
        if endpoint.endswith("apply")
        else {"paper_text": "abstract", "target_file": "../escape", "apply": True}
    )

    response = client.post(endpoint, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_ontology_filename"
    pipeline.assert_not_called()


def test_ontology_apply_writes_safe_name_inside_custom_dir(client, monkeypatch, tmp_path):
    custom = tmp_path / "custom"
    monkeypatch.setattr(api_module, "CUSTOM_ONTOLOGY_DIR", custom)
    monkeypatch.setattr(api_module, "_discover_metta_files", lambda: {})

    response = client.post(
        "/ontology/apply",
        json={
            "metta_block": "(InstanceOf Probe Type)",
            "target_file": "agent_notes",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "applied": True,
        "target_file": "agent_notes.metta",
        "error": None,
    }
    assert (custom / "agent_notes.metta").read_text(encoding="utf-8") == (
        "(InstanceOf Probe Type)"
    )
