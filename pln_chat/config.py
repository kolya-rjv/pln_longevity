import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# Primary ontology dir: the pln_longevity repo root (parent of pln_chat)
ONTOLOGY_DIR = BASE_DIR.parent
# Drop additional .metta files here to extend the knowledge base
CUSTOM_ONTOLOGY_DIR = BASE_DIR / "ontology" / "metta_files"
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL: str = os.getenv("PLN_MODEL", "gpt-5.4-mini")
AVAILABLE_MODELS: list[str] = ["gpt-5.4-mini", "gpt-5.4", "gpt-4o", "gpt-4-turbo"]
DEFAULT_TEMPERATURE: float = 0.2
OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_RETRIES: int = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "1")))
# The model that reads the My Patient text (core/patient_extract.py) has its own
# settings: it is called while a person waits on the Read button, so its timeout is
# short and it never retries. It must support strict structured outputs; the name is
# checked against core.patient_extract.SUPPORTED_MODELS, and an unsupported one is a
# configuration error, never a silent fall-back to the rules.
PLN_EXTRACT_MODEL: str = os.getenv("PLN_EXTRACT_MODEL", "gpt-6-luna")
PLN_EXTRACT_TIMEOUT_SECONDS: float = float(os.getenv("PLN_EXTRACT_TIMEOUT_SECONDS", "20"))
# GPT-5-family and o-series models take a reasoning effort instead of a temperature.
PLN_EXTRACT_REASONING_EFFORT: str = os.getenv("PLN_EXTRACT_REASONING_EFFORT", "low")
# Longest text the model reader takes; longer text is read by the rules only (the UI)
# or refused with 413 (the API, reader='model').
PLN_EXTRACT_MAX_CHARS: int = int(os.getenv("PLN_EXTRACT_MAX_CHARS", "4000"))
# Largest prompt (system + history + question) the API will SEND, in estimated
# tokens. The default /query prompt pastes the curated .metta files verbatim and
# already measures ~62k tokens; selecting a gene ETL file pushed it to 417k and
# came back as an OpenAI 400 AFTER the request was billed and the caller waited.
# Checking the estimate first turns that into an immediate, actionable 413 that
# names the files responsible. Raise it for a larger-context model.
PLN_MAX_PROMPT_TOKENS: int = int(os.getenv("PLN_MAX_PROMPT_TOKENS", "200000"))
# Characters per token used by that estimate. 4.0 is the usual English rule of
# thumb; MeTTa's punctuation density makes it conservative (i.e. it slightly
# OVER-counts), which is the safe direction for a pre-flight guard.
PLN_CHARS_PER_TOKEN: float = float(os.getenv("PLN_CHARS_PER_TOKEN", "4.0"))
# A selected .metta file larger than this is SUMMARISED into a schema card
# (predicates, arities, fact counts) instead of being pasted into the prompt
# verbatim. Every hand-written layer in this repo is under 20 KB; the files that
# exceed it are ETL dumps, whose row-by-row text is the least useful thing per
# token the translator could be shown. 0 disables summarisation.
PLN_PROMPT_FILE_MAX_BYTES: int = int(os.getenv("PLN_PROMPT_FILE_MAX_BYTES", "25000"))

# ── PLN runtime ────────────────────────────────────────────────────────────────
# Auto-detected: true when the `hyperon` package is importable.
# Override with PLN_RUNTIME_AVAILABLE=false in .env to force stub mode.
def _detect_hyperon() -> bool:
    _env = os.getenv("PLN_RUNTIME_AVAILABLE", "").lower()
    if _env == "false":
        return False
    if _env == "true":
        return True
    try:
        import importlib.util  # NB: submodule must be imported explicitly
        return importlib.util.find_spec("hyperon") is not None
    except Exception:
        return False

PLN_RUNTIME_AVAILABLE: bool = _detect_hyperon()

# Max size (bytes) of a .metta file loaded into the hyperon runtime space.
# Files larger than this are SKIPPED at execution time: hyperon 0.2.10 panics
# (hyperon-space trie: "Option::unwrap() on None") — or silently mis-matches —
# once a space grows past a few thousand atoms, and the ~107 KB
# drugage_etl_short.metta dump trips it. A size cap excludes that (and any future
# oversized ETL output) generically while keeping every hand-written ontology /
# inference file (all < 10 KB). Such bulk data stays queryable in stub mode.
# Raise this (or set PLN_MAX_KB_FILE_BYTES) once the runtime handles larger spaces.
PLN_MAX_KB_FILE_BYTES: int = int(os.getenv("PLN_MAX_KB_FILE_BYTES", "60000"))

# ── DrugAge ranking ────────────────────────────────────────────────────────────
# Hard cap on the compound pool one ranking request may ask for. The MeTTa
# insertion sort behind `rank-interventions` is O(n^2) with a large constant
# (measured: n=10 -> 6.1 s), which is how one 35-compound request blocked the
# API for 115 s. The default `linear` strategy is ~70 ms per compound, so this
# cap bounds a single request at a few seconds rather than minutes.
PLN_MAX_RANK_COMPOUNDS: int = int(os.getenv("PLN_MAX_RANK_COMPOUNDS", "60"))
# The SAME cap does not fit both strategies, and one number for both was a
# quiet promise the service could not keep: `metta_sort` is the O(n^2) MeTTa
# insertion sort, measured at 6.1 s for n=10, 52.6 s for n=20 and past the 60 s
# deadline by n=40 — so every request between 40 and the 60 published here was
# advertised as legal and answered with a 504. `linear` is ~70 ms per compound
# and is the default. metta_sort is kept because it is the reference
# implementation the parity test scores against, not because it scales.
PLN_MAX_METTA_SORT_COMPOUNDS: int = int(
    os.getenv("PLN_MAX_METTA_SORT_COMPOUNDS", "10")
)
# Max DrugAge source rows echoed back in a ranking response (0 disables the
# per-row listing). Rapamycin alone has 37 rows, so an unbounded list makes a
# big response out of a small question.
PLN_MAX_RANK_ROWS: int = int(os.getenv("PLN_MAX_RANK_ROWS", "120"))
# Max entries in a request's `ontology_files` selection. The list is read and
# re-parsed as given, so repeats used to amplify work with no ceiling; the API
# now also de-duplicates it. There are ~30 .metta files in total, so 64 is
# generous.
PLN_MAX_ONTOLOGY_FILES: int = int(os.getenv("PLN_MAX_ONTOLOGY_FILES", "64"))

# ── Ontology writes ────────────────────────────────────────────────────────────
# The curated .metta files ARE the service's reasoning: its rules, its evidence
# tiers, its calibration constants. `POST /ontology/apply` (and the Gradio
# "Apply to Ontology" button, which is mounted on the same app) appended a
# caller-supplied block to any of them. Verified on this checkout: one
# unauthenticated request appending `(= (evidence-confidence RCT_Human) 0.05)`
# to epistemic_calibration.metta silently re-tiers every human trial for every
# subsequent caller. Caller-generated entries belong in CUSTOM_ONTOLOGY_DIR;
# writing a curated file is an operator decision, made here.
PLN_ALLOW_CURATED_WRITES: bool = os.getenv(
    "PLN_ALLOW_CURATED_WRITES", "0"
).strip().lower() in {"1", "true", "yes", "on"}
# Largest block a single write may append. Generous for an extracted paper
# (the canonical taurine block is ~2 KB) and far below PLN_MAX_KB_FILE_BYTES,
# so no one write can push a file past the size at which the runtime stops
# loading it.
PLN_MAX_APPLY_BYTES: int = int(os.getenv("PLN_MAX_APPLY_BYTES", "32000"))

# ── PLN execution workers ──────────────────────────────────────────────────────
# hyperon 0.2.10 HOLDS THE GIL for the whole of MeTTa.run() (measured: two runs
# in two threads take exactly as long as two runs in sequence, ratio 0.998), so
# a MeTTa query starves the event loop and every other request with it — one
# 35-compound ranking blocked the whole API for 115 s. A thread pool cannot fix
# that; a process pool can, and it also makes a per-request timeout enforceable
# and contains hyperon's non-unwinding Rust abort (which kills the interpreter
# in-process but only a worker out-of-process).
# The value is the number of MeTTa queries that may run AT ONCE — each one gets
# its own forked process, because a shared pool cannot kill a single task and a
# timeout there takes other callers' healthy requests down with it (measured).
#   0 = run inline, in the request thread (the pre-2026-09 behaviour; used by
#       the in-process contract tests, which monkeypatch module globals a child
#       process could never see). Inline mode CANNOT enforce the timeout or the
#       admission limit — GET /health says so under pln_execution.
PLN_WORKER_POOL_SIZE: int = max(0, int(os.getenv("PLN_WORKER_POOL_SIZE", "2")))
# Per-request PLN budget in seconds. 0 disables. 60 s is deliberately generous:
# a legitimate 20-compound ranking is a few seconds, so this only catches the
# pathological shapes.
PLN_QUERY_TIMEOUT_SECONDS: float = float(os.getenv("PLN_QUERY_TIMEOUT_SECONDS", "60"))
# Max PLN tasks queued or running before new ones are refused with 503.
# 0 = derive it (4x the worker count).
PLN_MAX_INFLIGHT_QUERIES: int = max(0, int(os.getenv("PLN_MAX_INFLIGHT_QUERIES", "0")))
# (PLN_WORKER_MAX_TASKS was removed: every query now runs in its own process,
#  which exits when the query does, so there is nothing left to recycle.)

# ── UI defaults ────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.0
SHOW_METTA_DEFAULT: bool = True
SHOW_EXPLANATION_DEFAULT: bool = True
SHOW_DEBUG_DEFAULT: bool = False

# ── HTTP API (api.py) ────────────────────────────────────────────────────────
# Standalone API-only listener (`python api.py`). The normal combined listener
# below serves both the Gradio UI and these routes from one origin.
PLN_API_HOST: str = os.getenv("PLN_API_HOST", "0.0.0.0")
PLN_API_PORT: int = int(os.getenv("PLN_API_PORT", "8000"))
# Combined Gradio + API server used by `python app.py`. Defaults preserve the
# original Gradio listener so an existing ngrok tunnel to port 7860 keeps working.
PLN_SERVER_HOST: str = os.getenv("PLN_SERVER_HOST", "127.0.0.1")
PLN_SERVER_PORT: int = int(os.getenv("PLN_SERVER_PORT", "7860"))
PLN_CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("PLN_CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]

# ── HTTP API: version ────────────────────────────────────────────────────────
# Reported three ways: the FastAPI `info.version` in /openapi.json, a `version`
# field on GET /health, and an `X-API-Version` response header on every reply.
# A client that caches a schema or hard-codes a response shape can watch one
# string instead of diffing the OpenAPI document.
#
# 2.0.0 is a deliberate MAJOR bump, not a courtesy one. 1.1.0 answered an
# upstream outage, a hyperon exception and an oversized prompt with HTTP 200 and
# `intent: "clarification"`; those are now 4xx/5xx with a machine-readable
# `code` (see API.md "Failures are HTTP failures"). Any client that branched on
# `status == 200` is broken by that, which is exactly what a major version is
# for. The same release adds the discovery endpoints (/drugage/top,
# /interventions, /hallmarks, /kb/schema, /patients/markers), caller-supplied
# patients, and the optional auth + rate limiting below.
#
# 2.1.0 was additive: POST /patients/from-text took `reader: "rules" | "model"`.
#
# 3.0.0 breaks POST /patients/from-text: a model reads the text and code checks it
# (core/patient_read.py). `reader` is "auto" (the model when configured, else canonical
# lines) | "model" | "lines" ("rules" is the old name of "lines"); the default was
# "rules". The response loses `read_as_text`, `suggestions` and the statements'
# `source`/`typed` (there are no rewrites any more) and gains `discarded`.
PLN_API_VERSION: str = "3.0.0"

# ── HTTP API: optional access control ────────────────────────────────────────
# BOTH CONTROLS ARE OFF BY DEFAULT. Unset, the service behaves exactly as it
# did before this file grew these lines — that is the shape the Gradio UI, the
# contract tests and every existing agent integration expect.
#
# `.env.example` has documented `PLN_API_KEY` since the API was first split out
# of app.py; until now nothing read it. It does now. Set it (or the plural
# `PLN_API_KEYS`, comma-separated, to issue one key per consumer and revoke
# them independently) and every request must carry `X-API-Key: <value>`.
# /health, /docs, /redoc, /openapi.json and CORS preflights stay open: a
# readiness probe that needs a secret is a readiness probe that reports the
# wrong thing, and an agent that cannot read the schema cannot find the header
# it is missing.
def _collect_api_keys() -> list[str]:
    keys: list[str] = []
    for raw in (os.getenv("PLN_API_KEY", ""), os.getenv("PLN_API_KEYS", "")):
        for candidate in raw.split(","):
            value = candidate.strip()
            if value and value not in keys:
                keys.append(value)
    return keys

PLN_API_KEYS: list[str] = _collect_api_keys()
# Requests per minute per client address, 0 = off (the default). This is a
# per-process courtesy limit keyed on `request.client.host`; see the HONESTY
# CONTRACT in core/rate_limit.py for what it does and does not defend against.
PLN_RATE_LIMIT_PER_MINUTE: int = max(0, int(os.getenv("PLN_RATE_LIMIT_PER_MINUTE", "0")))
