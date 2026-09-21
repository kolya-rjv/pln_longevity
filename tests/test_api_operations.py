"""Contract tests for the operational basics: keys, rate limits, versioning.

The 2026-09-18 evaluation, recommendation 12: "Operational basics: API keys,
rate limits, prompt caching of the fixed system prompt, streaming or batch for
/query, versioned responses. Needed before anything beyond an ngrok demo." Its
§Performance finding was blunter: "There is no authentication, rate limit,
request timeout or cap on list sizes, so any caller can stall the service with
one large ranking."

Timeouts and list caps landed earlier (core/executor.py, PLN_MAX_* in
config.py). What these tests pin down is the rest:

* the key check and the rate limit are BOTH OFF by default — the single most
  important property here, because the Gradio UI, the acceptance runner and
  every other test in this suite send no credential;
* when switched on they refuse correctly, in the same structured `detail`
  shape as every other failure this API reports;
* /health, /docs, /redoc, /openapi.json and CORS preflights stay reachable
  without a key, so an agent can always find out what it is missing;
* the Gradio mount at `/` is not metered as if it were an API route;
* every response carries the version the caller is talking to;
* and the system prompt keeps its static content in front, which is what makes
  the fixed prefix cacheable upstream.

Run from the repository root:
    pytest tests/test_api_operations.py -q
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
import core.executor as executor_module  # noqa: E402
from core.llm_translator import TranslationResult  # noqa: E402
from core.pln_runner import PLNAtomResult, PLNRunResult  # noqa: E402
from core.rate_limit import MAX_TRACKED_KEYS, TokenBucketLimiter  # noqa: E402
from ontology.registry import OntologyRegistry  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_asgi_execution(monkeypatch):
    """Same setup as tests/test_api.py: run endpoints inline, write no logs."""
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)
    monkeypatch.setattr(fastapi_routing, "run_in_threadpool", run_inline)
    monkeypatch.setattr(dependency_utils, "run_in_threadpool", run_inline)


class ASGITestClient:
    """Small sync facade over HTTPX's ASGI transport (mirrors tests/test_api.py)."""

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

    def options(self, path: str, **kwargs):
        return self.request("OPTIONS", path, **kwargs)


@pytest.fixture
def client() -> ASGITestClient:
    return ASGITestClient()


@pytest.fixture
def protected(monkeypatch):
    """Turn the key check on for one test."""
    monkeypatch.setattr(api_module, "PLN_API_KEYS", ["shared-secret", "second-key"])


@pytest.fixture
def metered(monkeypatch):
    """Turn the rate limit on for one test, at two requests per minute."""
    limiter = TokenBucketLimiter(per_minute=2)
    monkeypatch.setattr(api_module, "_RATE_LIMITER", limiter)
    return limiter


# ── The default is open, and stays open ──────────────────────────────────────

def test_the_service_is_unauthenticated_and_unmetered_unless_configured(client):
    assert api_module.PLN_API_KEYS == []
    assert api_module._RATE_LIMITER.enabled is False

    response = client.get("/patients")

    assert response.status_code == 200
    assert response.headers["X-API-Version"] == api_module.PLN_API_VERSION


def test_health_reports_which_controls_are_switched_on(client, protected, metered):
    body = client.get("/health").json()

    assert body["api_key_required"] is True
    assert body["rate_limit_per_minute"] == 2
    assert body["version"] == api_module.PLN_API_VERSION


# ── Optional API key ─────────────────────────────────────────────────────────

def test_a_protected_deployment_refuses_a_request_with_no_key(client, protected):
    response = client.get("/patients")

    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["code"] == "api_key_required"
    assert detail["header"] == "X-API-Key"
    assert "/health" in detail["open_paths"]


def test_a_wrong_key_is_distinguishable_from_a_missing_one(client, protected):
    response = client.get("/patients", headers={"X-API-Key": "not-the-secret"})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "api_key_invalid"


def test_a_non_ascii_key_is_refused_rather_than_crashing(client, protected):
    """Starlette decodes headers as latin-1 and `compare_digest` rejects a
    non-ASCII str, so comparing as bytes is what keeps this a 401."""
    # Sent as raw bytes: an HTTP client cannot encode a non-ASCII header value
    # as str, but a hand-rolled caller or a proxy can put one on the wire.
    response = client.get("/patients", headers={"X-API-Key": b"s\xe9cret"})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "api_key_invalid"


@pytest.mark.parametrize("key", ["shared-secret", "second-key"])
def test_any_configured_key_is_accepted(client, protected, key):
    """Several keys so one consumer can be revoked without cutting off the rest."""
    response = client.get("/patients", headers={"X-API-Key": key})

    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/health", "/openapi.json", "/docs"])
def test_discovery_and_readiness_never_need_a_key(client, protected, path):
    """A probe that needs a secret reports the wrong thing when the secret is wrong."""
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/health", "/openapi.json", "/docs"])
def test_discovery_and_readiness_are_not_metered_either(client, metered, path):
    """...and one that can be rate-limited reports the service as down when it
    is merely busy. Observed against a live listener with the limit at 3/min: a
    six-call smoke test left `GET /health` answering 429."""
    statuses = [client.get(path).status_code for _ in range(6)]

    assert statuses == [200] * 6


def test_a_cors_preflight_is_not_answered_with_401(client, protected):
    """A preflight carries no credentials by construction; refusing it makes a
    browser client see an opaque failure instead of the real request's 401."""
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_the_key_check_runs_before_the_body_is_interpreted(client, protected, monkeypatch):
    """A refused caller must not be able to spend an OpenAI call or a worker."""
    translate = Mock()
    monkeypatch.setattr(api_module, "translate", translate)

    response = client.post("/query", json={"message": "what should I take?"})

    assert response.status_code == 401
    translate.assert_not_called()


# ── Rate limiting: the token bucket itself ───────────────────────────────────

def test_a_disabled_bucket_keeps_no_state_at_all():
    limiter = TokenBucketLimiter(per_minute=0)

    assert limiter.enabled is False
    assert [limiter.check("1.2.3.4") for _ in range(1000)] == [None] * 1000
    assert limiter._buckets == {}


def test_a_full_minute_of_allowance_may_arrive_as_one_burst():
    limiter = TokenBucketLimiter(per_minute=60)

    allowed = [limiter.check("1.2.3.4", now=100.0) for _ in range(60)]

    assert allowed == [None] * 60
    assert limiter.check("1.2.3.4", now=100.0) is not None


def test_retry_after_never_rounds_down_to_a_moment_the_token_is_absent():
    limiter = TokenBucketLimiter(per_minute=60)   # one token per second
    limiter.check("1.2.3.4", now=0.0)
    for _ in range(59):
        limiter.check("1.2.3.4", now=0.0)

    retry_after = limiter.check("1.2.3.4", now=0.0)

    assert retry_after == 1
    assert limiter.check("1.2.3.4", now=retry_after) is None


def test_the_bucket_refills_continuously_rather_than_on_a_minute_boundary():
    limiter = TokenBucketLimiter(per_minute=60)
    for _ in range(60):
        limiter.check("1.2.3.4", now=0.0)

    assert limiter.check("1.2.3.4", now=0.5) is not None    # half a token
    assert limiter.check("1.2.3.4", now=1.0) is None        # one token
    assert limiter.check("1.2.3.4", now=1.0) is not None    # and no more


def test_each_address_gets_its_own_allowance():
    limiter = TokenBucketLimiter(per_minute=1)

    assert limiter.check("1.2.3.4", now=0.0) is None
    assert limiter.check("5.6.7.8", now=0.0) is None
    assert limiter.check("1.2.3.4", now=0.0) is not None


def test_idle_buckets_are_evicted_so_the_key_space_cannot_grow_without_bound():
    """The key is remote-controlled, so an unbounded dict is a slow leak."""
    limiter = TokenBucketLimiter(per_minute=10)
    for index in range(MAX_TRACKED_KEYS):
        limiter.check(f"10.0.{index // 256}.{index % 256}", now=0.0)

    assert len(limiter._buckets) == MAX_TRACKED_KEYS

    limiter.check("172.16.0.1", now=3600.0)   # an hour later: all refilled

    assert len(limiter._buckets) == 1


# ── Rate limiting: through HTTP ──────────────────────────────────────────────

def test_a_metered_caller_is_refused_with_429_and_a_retry_after(client, metered):
    assert client.get("/patients").status_code == 200
    assert client.get("/patients").status_code == 200

    third = client.get("/patients")

    assert third.status_code == 429
    assert third.headers["Retry-After"] == "30"     # 2/min -> one token per 30 s
    detail = third.json()["detail"]
    assert detail["code"] == "rate_limited"
    assert detail["limit_per_minute"] == 2
    assert detail["retry_after_seconds"] == 30
    assert third.headers["X-API-Version"] == api_module.PLN_API_VERSION


def test_a_readiness_probe_cannot_exhaust_another_callers_allowance(client, metered):
    """The open paths spend no tokens, so a probe loop leaves the budget intact."""
    for _ in range(10):
        client.get("/health")

    assert client.get("/patients").status_code == 200
    assert client.get("/patients").status_code == 200
    assert client.get("/patients").status_code == 429


def test_only_this_modules_routes_are_metered():
    """Gradio is mounted at `/` on the same app and issues its own traffic.

    Its mount is a Starlette `Mount`, never an `APIRoute`, so a UI session's
    static assets and queue polls are not spent against an API rate limit.
    """
    assert api_module._is_api_path("/query") is True
    assert api_module._is_api_path("/patients/markers") is True
    assert api_module._is_api_path("/genes/TP53") is True        # path parameter
    assert api_module._is_api_path("/") is False
    assert api_module._is_api_path("/gradio_api/queue/join") is False
    assert api_module._is_api_path("/theme.css") is False


# ── Versioned responses ──────────────────────────────────────────────────────

def test_one_version_string_is_reported_three_ways(client):
    """A caller that pins a contract should not have to diff an OpenAPI doc."""
    header = client.get("/patients").headers["X-API-Version"]
    health = client.get("/health").json()["version"]
    schema = client.get("/openapi.json").json()["info"]["version"]

    assert header == health == schema == api_module.PLN_API_VERSION
    assert api_module.PLN_API_VERSION.startswith("2."), (
        "the 1.x contract answered upstream and runtime failures with HTTP 200; "
        "they are 4xx/5xx now, which is a breaking change for any client that "
        "branched on the status code"
    )


def test_a_refusal_is_versioned_too(client, protected):
    response = client.get("/patients")

    assert response.status_code == 401
    assert response.headers["X-API-Version"] == api_module.PLN_API_VERSION


# ── Prompt caching: keep the fixed prefix fixed ──────────────────────────────
#
# The evaluation wrote: "with no prompt caching, cost and latency scale with
# that number on every question". There is nothing to implement here, and a
# hand-rolled cache would be the wrong answer: OpenAI caches a stable prompt
# PREFIX automatically, so the fixed system prompt is already cacheable. The
# lever that actually mattered was prompt SIZE, and that is already pulled —
# measured on this checkout, the default prompt is 292,550 characters (~73,100
# estimated tokens) and an oversized ETL selection is replaced by a schema card
# rather than pasted (drugage_etl_short.metta: ~26,900 estimated tokens
# verbatim -> ~340 as a card).
#
# What CAN regress silently is the ORDERING: move one per-request string in
# front of the static ontology block and the cacheable prefix drops to nothing
# while every number above stays the same. These two tests are that guard.

def _capture_system_prompt(client, monkeypatch, payload: dict) -> str:
    translation = TranslationResult(
        metta_query="!(match &self $x $x)",
        explanation="e",
        intent="inference",
        requires_pln_inference=True,
        confidence_filter=0.0,
    )
    translate = Mock(return_value=translation)
    monkeypatch.setattr(api_module, "translate", translate)
    monkeypatch.setattr(api_module, "log_turn", Mock())
    monkeypatch.setattr(
        api_module,
        "run_query",
        Mock(return_value=PLNRunResult(
            status="ok",
            results=[PLNAtomResult("(answer)", {"strength": 0.8, "confidence": 0.7})],
            query_time_ms=1,
            mode="runtime",
        )),
    )
    monkeypatch.setattr(
        api_module, "_build_context", lambda selected: (OntologyRegistry(), {})
    )

    assert client.post("/query", json=payload).status_code == 200
    return translate.call_args.kwargs["system_prompt"]


def test_the_system_prompt_is_identical_for_two_different_questions(
    client, monkeypatch
):
    first = _capture_system_prompt(client, monkeypatch, {"message": "what is NMN?"})
    second = _capture_system_prompt(
        client, monkeypatch, {"message": "rank rapamycin and metformin"}
    )

    assert first == second
    assert "what is NMN?" not in first     # the question is a later message
    assert len(first) > 1000               # and it is a prefix worth caching


def test_a_callers_patient_is_appended_after_the_static_prompt_not_in_front(
    client, monkeypatch
):
    static = _capture_system_prompt(client, monkeypatch, {"message": "q"})
    personalized = _capture_system_prompt(client, monkeypatch, {
        "message": "q",
        "patient": {"age": 45, "sex": "Female", "markers": {"CRP": 1.2}},
    })

    assert personalized.startswith(static), (
        "the per-request patient block must be APPENDED; moving it in front of "
        "the ontology snapshot would make every personalized request a cache miss"
    )
    assert "THIS REQUEST'S PATIENT" in personalized[len(static):]


# ── the preflight an agent is told to call first is typed ────────────────────

def test_health_is_described_in_the_openapi_schema(client):
    """It was the one route with no `response_model`.

    API.md tells integrators to hand an agent the base URL plus
    `/openapi.json`, names `/health` the first call, and directs callers to
    branch on `api_key_required` and `pln_execution` — none of which an
    untyped `dict` return puts in the schema.
    """
    schema = client.get("/openapi.json").json()
    ref = (schema["paths"]["/health"]["get"]["responses"]["200"]
           ["content"]["application/json"]["schema"])
    assert "$ref" in ref, "/health still returns an untyped object"

    model = schema["components"]["schemas"][ref["$ref"].rsplit("/", 1)[-1]]
    assert {"api_key_required", "pln_execution", "version", "runtime_ready"} \
        <= set(model["properties"])

    # …and the nested execution block, which is what tells a caller whether the
    # deadline and the admission limit are actually being applied.
    execution_ref = model["properties"]["pln_execution"]
    name = execution_ref.get("$ref", "").rsplit("/", 1)[-1]
    execution = schema["components"]["schemas"][name]
    assert {"mode", "timeout_enforced", "admission_control", "timeout_seconds"} \
        <= set(execution["properties"])

    # The live response must satisfy the schema it advertises.
    body = client.get("/health").json()
    assert set(model["properties"]) <= set(body)
