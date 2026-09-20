"""PLN Natural Language Query Interface — plain HTTP/JSON API.

This is the REST portion of the Gradio UI in app.py, for scripts and agents
that want to call the query pipeline directly instead of driving a browser.
It runs the exact same core pipeline as the chat handler in app.py
(translate -> validate -> run_query -> format_bot_response) and exposes it
as JSON endpoints with auto-generated OpenAPI docs. Callers that already
know the MeTTa they want can skip translation via POST /metta/run instead
of POST /query.

The KB now includes a curated inference stack (calibration, deduction,
abductive diagnosis, intervention ranking, patient grounding, counterfactual
analysis, risk prediction, supplement recommendations — see app.py's
_INFERENCE_STACK) plus a scoped DrugAge lifespan-ranking engine reachable
via POST /drugage/rank (or a `(rank-drugage-lifespan (...))` form through
/query or /metta/run). See API.md for the full picture.

Run standalone:
    python api.py
    # -> http://0.0.0.0:8000  (interactive docs at /docs, schema at /openapi.json)

Or with uvicorn directly (e.g. for --reload during development):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

When `app.py` starts, it mounts Gradio at `/` on this FastAPI application, so
the UI and these routes share one port and ngrok origin. This module can still
be run on its own when an API-only process is useful.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path

# Ensure the pln_chat package root is on sys.path so submodule imports work
# whether the file is run directly or via `python -m`, matching app.py.
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import (
    AVAILABLE_MODELS,
    CUSTOM_ONTOLOGY_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ONTOLOGY_DIR,
    OPENAI_API_KEY,
    PLN_CORS_ORIGINS,
    PLN_CHARS_PER_TOKEN,
    PLN_MAX_KB_FILE_BYTES,
    PLN_MAX_ONTOLOGY_FILES,
    PLN_MAX_PROMPT_TOKENS,
    PLN_PROMPT_FILE_MAX_BYTES,
    PLN_MAX_RANK_COMPOUNDS,
    PLN_MAX_RANK_ROWS,
    PLN_RUNTIME_AVAILABLE,
)
from ontology.hallmarks import hallmark_index
from ontology.inventory import inventory_for, schema_card, summarise_oversized
from ontology.loader import load_specific_files, parse_metta_text
from ontology.registry import BUILTIN_REGISTRY, OntologyRegistry
from ontology.expander import run_expansion_pipeline
from ontology.drugage_scoring import load_knobs
from ontology.drugage_selector import BUILD_DRUGAGE
from core.context_builder import build_system_prompt
from core.patient_builder import (
    MARKERS,
    BuiltPatient,
    PatientSpecError,
    build_patient,
    marker_catalog,
)
from core.executor import (
    PLNExecutionTimeout,
    PLNOverloaded,
    PLNWorkerCrashed,
    executor_stats,
    run_offloaded,
)
from core.drugage_router import (
    SCORE_SEMANTICS,
    _row_out,
    drugage_top,
    parse_drugage_query,
    rank_drugage,
    resolve_compounds,
    route_drugage_ranking,
)
from core.llm_translator import translate
from core.metta_validator import ValidationResult, validate
from core.pln_runner import run_query
from utils.formatting import format_bot_response
from utils.logging import log_http_request, log_turn


# ── Ontology file discovery / resolution (mirrors app.py) ──────────────────

def _discover_metta_files() -> dict[str, Path]:
    """Return {filename: absolute_path} for every .metta file in the project."""
    files: dict[str, Path] = {}
    for base in (ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR):
        if base.exists():
            for p in sorted(base.glob("*.metta")):
                files[p.name] = p
    return files


# The coherent inference stack the LLM translator needs to SEE (as system-prompt
# context) to emit calls into the demo functions — calibrate-tv / infer / explain
# / rank-interventions / diagnose-patient / decompose-grimage / counterfactual /
# predict-risk / recommend-supplements — and for the symbol validator to
# recognise them. Mirrors app.py's _INFERENCE_STACK verbatim; keep in sync.
_INFERENCE_STACK: list[str] = [
    "system_types.metta",
    "logical_predicates.metta",
    "measurement_types.metta",
    "epistemic_calibration.metta",
    "species_taxonomy.metta",

    "drugage_entries.metta",
    "cellage_metadata.metta",

    "grim_age_core.metta",
    "grim_age_lu2019_evidence.metta",
    "evidence_calibration.metta",

    "hallmarks_core.metta",
    "hallmarks_lopezotin2023_anchors.metta",
    "hallmarks_lopezotin2023_intervention_evidence.metta",

    "mechanistic_bridges.metta",
    "pln_deduction.metta",
    "pln_intervention_ranking.metta",
    "pln_abductive_diagnosis.metta",

    "drugage_calibration.metta",

    "patient_profile.metta",
    "pln_counterfactual.metta",
    "pln_risk_prediction.metta",

    "supplement_evidence.metta",
    "pln_supplement_recommendation.metta",
]


def _default_selection(choices: list[str]) -> list[str]:
    """Mirrors app.py's _DEFAULT_SELECTION: the curated inference stack (so the
    LLM translator sees the demo functions), falling back to an 'epistemic' file
    or the first available file if the stack isn't present."""
    return (
        [f for f in _INFERENCE_STACK if f in choices]
        or [k for k in choices if "epistemic" in k.lower()]
        or choices[:1]
    )


# KB files actually usable at EXECUTION time: every discovered file MINUS any
# over PLN_MAX_KB_FILE_BYTES. hyperon 0.2.10 panics (or silently mis-matches)
# once a space exceeds a few thousand atoms, and e.g. the ~107 KB
# drugage_etl_short.metta dump trips it. Mirrors app.py's _runtime_kb_paths();
# excluded files stay queryable in stub mode and are listed by GET /ontology/files.
def _runtime_kb_paths() -> list[Path]:
    kept: list[Path] = []
    for path in _discover_metta_files().values():
        try:
            too_big = path.stat().st_size > PLN_MAX_KB_FILE_BYTES
        except OSError:
            too_big = False
        if not too_big:
            kept.append(path)
    return kept


def _build_context(selected_files: list[str]) -> tuple[OntologyRegistry, dict[str, str]]:
    """Load selected .metta files for the LLM system-prompt context.

    Mirrors app.py: the selection only controls what the LLM sees. Execution
    against the KB always uses the runtime-safe file set (_runtime_kb_paths),
    regardless of this selection — see /query and /metta/run.

    A selected file over PLN_PROMPT_FILE_MAX_BYTES is replaced by its schema
    card rather than pasted verbatim: a 525 KB gene dump is what took one
    prompt to 417,000 tokens and an upstream 400.
    """
    metta_files = _discover_metta_files()
    paths = [metta_files[f] for f in selected_files if f in metta_files]
    if not paths:
        return BUILTIN_REGISTRY, {}
    registry, raw_contents = load_specific_files(paths)
    raw_contents, _ = summarise_oversized(
        raw_contents, paths, max_bytes=PLN_PROMPT_FILE_MAX_BYTES
    )
    return registry, raw_contents


def _runtime_inventory():
    """What the EXECUTION KB actually holds (ground facts, not declarations)."""
    return inventory_for(_runtime_kb_paths())


def _dedupe_ontology_files(value: Optional[list[str]]) -> Optional[list[str]]:
    """Collapse repeats while preserving order.

    `load_specific_files` reads and re-parses the list as given, so a caller
    passing the same filename N times paid N times for it — inside the
    GIL-holding request thread, with no ceiling. Order is preserved because the
    selection order is the order the files appear in the LLM's context.
    """
    if value is None:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for name in value:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _validate_ontology_files(selected_files: list[str]) -> None:
    """Reject misspelled file selections instead of silently using less context."""
    known = _discover_metta_files()
    unknown = sorted(set(selected_files) - set(known))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unknown_ontology_files",
                "message": "Unknown ontology file selection.",
                "files": unknown,
            },
        )


def _runtime_registry() -> OntologyRegistry:
    """Registry built from every file actually used at execution time
    (_runtime_kb_paths) — the default for validating a raw MeTTa query in
    /metta/run when the caller hasn't scoped `ontology_files`. Deliberately NOT
    "every discovered file": a file excluded from execution (oversized) would
    otherwise validate symbols that then silently fail to resolve at runtime.
    """
    paths = _runtime_kb_paths()
    if not paths:
        return BUILTIN_REGISTRY
    registry, _ = load_specific_files(paths)
    return registry


_SAFE_METTA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.metta$")


def _normalise_metta_name(value: str, *, field: str) -> str:
    """Return a safe basename ending in .metta, or reject the request."""
    name = value.strip()
    if not name.endswith(".metta"):
        name = f"{name}.metta"
    if not _SAFE_METTA_NAME.fullmatch(name) or Path(name).name != name:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_ontology_filename",
                "message": f"{field} must be a plain .metta filename without path separators.",
            },
        )
    return name


def _ensure_allowed_target(path: Path) -> Path:
    """Confine ontology writes to direct children of the two ontology roots."""
    resolved = path.resolve(strict=False)
    allowed_parents = {
        ONTOLOGY_DIR.resolve(strict=False),
        CUSTOM_ONTOLOGY_DIR.resolve(strict=False),
    }
    if resolved.parent not in allowed_parents:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_ontology_filename",
                "message": "Ontology target resolves outside an allowed ontology directory.",
            },
        )
    return path


def _resolve_target_path(target_file: Optional[str], new_filename: Optional[str]) -> Path:
    """Return an absolute Path for a .metta file to append extracted entries to.

    If `target_file` names an existing .metta file, append to it in place.
    Otherwise treat it (or `new_filename`) as the stem of a new file created
    under CUSTOM_ONTOLOGY_DIR — same behaviour as the "create new file…"
    option in the Gradio Ontology Expander tab.
    """
    metta_files = _discover_metta_files()
    if target_file and target_file in metta_files:
        return _ensure_allowed_target(metta_files[target_file])

    raw_name = target_file or new_filename or "expanded_ontology"
    name = _normalise_metta_name(raw_name, field="target_file/new_filename")
    CUSTOM_ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    return _ensure_allowed_target(CUSTOM_ONTOLOGY_DIR / name)


# ── Patient profile discovery ───────────────────────────────────────────────
# Patients are hardcoded facts in patient_profile.metta (part of the runtime KB
# set), not something a caller submits — a query just names one (e.g.
# "Patient001") for the dedicated <Patient> query forms documented in API.md /
# the system prompt. This just makes the known names (+ a few headline facts)
# discoverable over HTTP instead of requiring a caller to read the .metta file.
_PATIENT_AGE_RE = re.compile(r"\(PatientAge\s+(\S+)\s+([\d.]+)\)")
_PATIENT_SEX_RE = re.compile(r"\(PatientSex\s+(\S+)\s+(\S+)\)")
_PATIENT_SMOKING_RE = re.compile(r"\(PatientSmoking\s+(\S+)\s+(\S+)\)")


def _patient_summaries() -> list[dict]:
    registry, raw_contents = load_specific_files(_runtime_kb_paths())
    patient_ids = sorted(
        name for name, entry in registry.entries.items()
        if entry.type_signature == "PatientProfile"
    )
    text = "\n".join(raw_contents.values())
    ages = dict(_PATIENT_AGE_RE.findall(text))
    sexes = dict(_PATIENT_SEX_RE.findall(text))
    smoking = dict(_PATIENT_SMOKING_RE.findall(text))
    return [
        {
            "id": pid,
            "age": float(ages[pid]) if pid in ages else None,
            "sex": sexes.get(pid),
            "smoking": smoking.get(pid),
        }
        for pid in patient_ids
    ]


# ── Caller-supplied patients ────────────────────────────────────────────────
# Patients used to be KB facts only, so the whole personalized stack — risk,
# decomposition, counterfactuals, ranking, supplements — worked for exactly two
# people. The inference never needed that: it reads PatientAge / PatientSex /
# MeasuredZ and nothing else, and run_query already injects caller-supplied
# atoms into the same space. What was missing is a typed surface, a z-scoring
# policy, and sanitisation — see core/patient_builder.py for why the last one is
# load-bearing.

_SD_TO_YEARS_RE = re.compile(r"\(=\s*\(grimaccel-sd-to-years\)\s*([\d.]+)\s*\)")
_ELEVATED_Z_RE = re.compile(r"\(=\s*\(elevated-z-threshold\)\s*([\d.]+)\s*\)")


def _patient_knobs() -> tuple[float, float]:
    """`grimaccel-sd-to-years` and `elevated-z-threshold`, read off the KB.

    Both are documented tunables of the MeTTa layer. Reading them rather than
    copying them keeps a caller's years->z conversion and the Elevated/Low
    labels in lockstep with the engine that will consume them.
    """
    sd_to_years, elevated = 4.2, 1.0
    try:
        risk = (ONTOLOGY_DIR / "pln_risk_prediction.metta").read_text(encoding="utf-8")
        m = _SD_TO_YEARS_RE.search(risk)
        if m:
            sd_to_years = float(m.group(1))
    except OSError:
        pass
    try:
        profile = (ONTOLOGY_DIR / "patient_profile.metta").read_text(encoding="utf-8")
        m = _ELEVATED_Z_RE.search(profile)
        if m:
            elevated = float(m.group(1))
    except OSError:
        pass
    return sd_to_years, elevated


def _known_patient_ids() -> set[str]:
    return {p["id"] for p in _patient_summaries()}


def _build_caller_patient(payload: Optional[dict]) -> Optional[BuiltPatient]:
    """Validate and render a caller's patient, or raise a 422 explaining why."""
    if payload is None:
        return None
    sd_to_years, elevated = _patient_knobs()
    try:
        return build_patient(
            payload,
            existing_ids=_known_patient_ids(),
            sd_to_years=sd_to_years,
            elevated_threshold=elevated,
        )
    except PatientSpecError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": exc.message, **exc.extra},
        ) from None


# Rule and constant REDEFINITIONS in caller-supplied atoms. `(= (f …) …)` does
# not shadow the KB's definition — hyperon keeps both and every call becomes
# non-deterministic — so a payload carrying one silently corrupts OTHER
# patients' answers in the same request. Verified: injecting
# `(= (baseline-risk-chd $a $s) 0.999)` made `patient-baseline` return two
# values and a risk query return 256 atoms, one of them 5.4e11.
_DEFINITION_RE = re.compile(r"\(\s*=\s*\(")


def _guard_extra_atoms(text: Optional[str], *, allow_definitions: bool) -> None:
    if not text or allow_definitions:
        return
    if _DEFINITION_RE.search(text):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "definition_in_extra_atoms",
                "message": (
                    "`extra_atoms` contains a rule or constant definition "
                    "`(= (…) …)`. A definition does not replace the knowledge "
                    "base's own — the engine keeps both and every affected "
                    "answer, including other patients', becomes "
                    "non-deterministic. Send facts only, or set "
                    "`allow_definitions: true` if you are deliberately testing "
                    "a redefinition."
                ),
            },
        )


_PATIENT_MENTION_RE = re.compile(r"\b((?:Patient|Caller_)[A-Za-z0-9_]*)\b")


def _unknown_patient_warning(query: str, known: set[str]) -> Optional[str]:
    """Catch a query about a patient nobody defined.

    `rank-interventions-for-patient` returns a full, plausible ranking for an id
    that does not exist: the personal term degenerates to zero and the
    population ranking survives, so a typo produces a confident wrong-looking-
    right answer instead of an error.
    """
    mentioned = {m for m in _PATIENT_MENTION_RE.findall(query)}
    unknown = sorted(mentioned - known)
    if not unknown:
        return None
    return (
        "Query mentions patient id(s) the knowledge base does not hold: "
        + ", ".join(unknown)
        + ". Some personalized forms still return a population-level answer for "
        "an unknown patient, so treat this result as unpersonalized. Known ids: "
        + (", ".join(sorted(known)) or "none")
        + ". Submit your own patient with the `patient` field."
    )


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PLN Longevity Query API",
    description=(
        "Programmatic JSON API over the PLN natural-language query pipeline — "
        "the same logic behind the Gradio chat UI (app.py), for scripts and agents. "
        "See /docs for interactive testing."
    ),
    version="1.1.0",
)

# Permissive by default so a local agent/script can call this without CORS
# friction during experimentation. Set PLN_CORS_ORIGINS to a comma-separated
# allowlist when a browser client reaches this beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=PLN_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_api_request(request: Request, call_next):
    """Record every API request, including raw JSON/text bodies and failures."""
    started = time.monotonic()
    raw_body = await request.body()
    body = raw_body.decode("utf-8", errors="replace")
    status_code = 500
    error: Optional[str] = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        client = request.client.host if request.client else None
        log_http_request(
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            body=body,
            status_code=status_code,
            duration_ms=int((time.monotonic() - started) * 1000),
            client=client,
            content_type=request.headers.get("content-type"),
            user_agent=request.headers.get("user-agent"),
            error=error,
        )


#: Which HTTP status each translator failure class deserves. The evaluation's
#: complaint was not that failures happen — it is that they arrived as HTTP 200
#: with intent "clarification", indistinguishable from a successful-but-empty
#: answer, so "a client filtering on status would never notice".
_TRANSLATION_ERROR_STATUS: dict[str, int] = {
    "missing_api_key":          503,   # the service is not configured to answer
    "auth":                     503,   # ...and its credential is rejected
    "rate_limit":               429,   # retry later
    "context_length_exceeded":  413,   # the CALLER can fix this one
    "timeout":                  504,
    "connection":               502,
    "upstream_error":           502,
    "bad_json":                 502,
}


def _estimate_tokens(text: str) -> int:
    return int(len(text) / max(1.0, PLN_CHARS_PER_TOKEN))


def _guard_prompt_size(system_prompt: str, message: str, history: list[dict],
                       selected: list[str]) -> int:
    """Refuse an over-long prompt BEFORE spending an OpenAI call on it.

    Selecting a gene ETL file as `ontology_files` produced a 417,000-token
    prompt, which upstream rejected — after the round trip, and in a shape that
    read like an ordinary empty result. The estimate is deliberately simple
    (characters / PLN_CHARS_PER_TOKEN) so it needs no tokenizer dependency and
    cannot itself fail; it over-counts slightly, which is the safe direction.
    """
    total = _estimate_tokens(system_prompt) + _estimate_tokens(message)
    total += sum(_estimate_tokens(turn.get("content", "")) for turn in history)
    if PLN_MAX_PROMPT_TOKENS and total > PLN_MAX_PROMPT_TOKENS:
        files = _discover_metta_files()
        biggest = sorted(
            ((name, files[name].stat().st_size) for name in selected if name in files),
            key=lambda pair: -pair[1],
        )[:5]
        raise HTTPException(
            status_code=413,
            detail={
                "code": "prompt_too_large",
                "message": (
                    f"The assembled prompt is about {total:,} tokens, over the "
                    f"{PLN_MAX_PROMPT_TOKENS:,}-token limit, so it was not sent. "
                    f"Narrow `ontology_files` (see GET /ontology/files) or shorten "
                    f"`history`."
                ),
                "estimated_tokens": total,
                "limit_tokens": PLN_MAX_PROMPT_TOKENS,
                "largest_selected_files": [
                    {"file": name, "bytes": size} for name, size in biggest
                ],
            },
        )
    return total


#: The same idea for PLN execution failures: a hyperon exception or a missing
#: DrugAge build are not "a successful query that happened to be empty".
_PLN_ERROR_STATUS: dict[str, int] = {
    "runtime_error":         502,
    "drugage_build_missing": 503,   # the service is not ready, not the caller's fault
}


def _raise_pln_failure(pln_result, *, stage: str, extra: Optional[dict] = None) -> None:
    """Turn a PLN execution failure into a real HTTP status."""
    code = pln_result.error_code or "runtime_error"
    detail = {
        "code": code,
        "message": pln_result.error,
        "stage": stage,
        "pln_mode": pln_result.mode,
    }
    if extra:
        detail.update(extra)
    raise HTTPException(
        status_code=_PLN_ERROR_STATUS.get(code, 502), detail=detail
    )


def _raise_translation_failure(translation) -> None:
    """Turn a failed translation into a real HTTP status."""
    status = _TRANSLATION_ERROR_STATUS.get(translation.error_code or "", 502)
    headers = {"Retry-After": "10"} if status == 429 else None
    raise HTTPException(
        status_code=status,
        detail={
            "code": translation.error_code or "upstream_error",
            "message": translation.error,
            "stage": "translation",
            "usage": translation.usage,
        },
        headers=headers,
    )


# ── Failure handling ─────────────────────────────────────────────────────────
# PLN execution runs in a worker process (core/executor.py). Its three failure
# modes are real, distinguishable HTTP conditions — not 200s with a note.

@app.exception_handler(PLNExecutionTimeout)
async def _pln_timeout_handler(request: Request, exc: PLNExecutionTimeout):
    return JSONResponse(
        status_code=504,
        content={"detail": {
            "code": "pln_timeout",
            "message": str(exc),
            "timeout_seconds": exc.timeout_s,
        }},
    )


@app.exception_handler(PLNOverloaded)
async def _pln_overloaded_handler(request: Request, exc: PLNOverloaded):
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "5"},
        content={"detail": {
            "code": "pln_overloaded",
            "message": str(exc),
            "max_inflight": exc.limit,
        }},
    )


@app.exception_handler(PLNWorkerCrashed)
async def _pln_worker_crashed_handler(request: Request, exc: PLNWorkerCrashed):
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "pln_worker_crashed", "message": str(exc)}},
    )


# ── Schemas ──────────────────────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)


class PatientIn(BaseModel):
    """A patient the CALLER supplies, scored for this request only.

    Nothing is written to disk: the atoms live in the query's hyperon space and
    disappear with it. The id is namespaced `Caller_…` so it can never collide
    with a curated patient — submitting a second `Patient001` does not replace
    the first, it unions both and makes every answer non-deterministic.
    """
    id: Optional[str] = Field(
        default=None, max_length=48,
        description="Letters, digits and underscores. Prefixed with 'Caller_'.",
    )
    age: Optional[float] = Field(
        default=None, ge=0, le=130,
        description="Required for an ABSOLUTE risk — it selects the baseline.",
    )
    sex: Optional[str] = Field(
        default=None,
        description="Male | Female. Required for an absolute risk. The baseline "
                    "CHD table is stratified by exactly these two; a third branch "
                    "would be an invented number.",
    )
    smoking: Optional[str] = Field(
        default=None, description="NeverSmoker | FormerSmoker | CurrentSmoker.",
    )
    markers: dict[str, object] = Field(
        default_factory=dict,
        description="Biomarker -> a z-score (a bare number), or an object with "
                    "`z`, or `value` (+ optional `unit`) to be standardised "
                    "server-side. See GET /patients/markers for what is supported "
                    "and which conversions are curated priors.",
    )


class QueryRequest(BaseModel):
    message: str = Field(
        ..., max_length=50_000, description="Natural-language question for the KB."
    )
    history: list[HistoryTurn] = Field(
        default_factory=list,
        max_length=100,
        description="Prior turns (oldest first) for multi-turn context. "
                    "Pass back the `history` from the previous response to continue a conversation.",
    )
    ontology_files: Optional[list[str]] = Field(
        default=None,
        max_length=PLN_MAX_ONTOLOGY_FILES,
        description="Which .metta files to inject into the LLM's system-prompt context "
                    "(see GET /ontology/files for choices). Defaults to the curated inference "
                    "stack (calibration, deduction, diagnosis, ranking, patient grounding, "
                    "counterfactual, risk, supplement layers) so the LLM knows about the demo "
                    "functions. PLN execution always runs against every runtime-safe .metta "
                    "file (GET /ontology/files -> excluded_from_runtime) regardless of this.",
    )
    model: str = Field(default=DEFAULT_MODEL, description=f"One of {AVAILABLE_MODELS}.")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0,
        description="PLN results below this confidence are filtered out.",
    )
    patient: Optional[PatientIn] = Field(
        default=None,
        description="Ask about YOUR patient instead of a built-in one. The atoms "
                    "are injected into this request's space only. Mention the "
                    "returned id (or just say 'my patient') in `message`.",
    )
    show_metta: bool = Field(default=True, description="Include the generated MeTTa query in `answer`.")
    show_explanation: bool = Field(default=True, description="Include the NL explanation in `answer`.")
    show_debug: bool = Field(default=False, description="Include token usage / raw LLM response in `answer`.")

    @field_validator("model")
    @classmethod
    def model_must_be_supported(cls, value: str) -> str:
        if value not in AVAILABLE_MODELS:
            raise ValueError(f"model must be one of: {', '.join(AVAILABLE_MODELS)}")
        return value

    @field_validator("ontology_files")
    @classmethod
    def ontology_files_are_deduped(cls, value):
        return _dedupe_ontology_files(value)


class PLNAtomOut(BaseModel):
    atom: str
    strength: Optional[float] = None
    confidence: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str = Field(description="Fully formatted response, identical to what the Gradio chatbot shows.")
    metta_query: str
    explanation: str
    intent: str
    requires_pln_inference: bool
    confidence_filter: float = Field(
        description="The threshold the LLM TRANSLATOR suggested for this question — "
                    "informational only, it is not applied. The threshold that was "
                    "actually applied is `confidence_threshold_applied`.",
    )
    confidence_threshold_applied: float = Field(
        description="The `confidence_threshold` from the request, as applied to "
                    "`pln_results`. A collapsed result (a ranking, a diagnosis, a "
                    "tiered recommendation) is filtered ENTRY BY ENTRY, not as a "
                    "whole — before this it was kept or dropped on its leading "
                    "entry's confidence, which made the parameter look inert.",
    )
    warnings: list[str]
    validation_valid: bool
    validation_issues: list[str]
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal observations about the generated query — chiefly, "
                    "that it calls a predicate the runtime KB has no facts for.",
    )
    ungrounded_predicates: list[str] = Field(
        default_factory=list,
        description="Predicates the query uses that are DECLARED in the ontology but "
                    "hold zero ground atoms, so the query is well-formed and returns "
                    "nothing. An empty `pln_results` alongside a non-empty list here "
                    "means 'the KB cannot answer this', not 'the answer is no'. "
                    "See GET /kb/schema.",
    )
    pln_status: str
    pln_mode: str
    pln_query_time_ms: int
    pln_results: list[PLNAtomOut]
    pln_error: Optional[str] = Field(
        default=None,
        description="Set when pln_status is 'error' (the `answer` text also embeds this).",
    )
    routed: Optional[str] = Field(
        default=None,
        description="Set to 'drugage_ranking' when the generated MeTTa was a "
                    "`(rank-drugage-lifespan ...)` form and got dispatched to the scoped "
                    "DrugAge engine instead of the generic KB (see POST /drugage/rank).",
    )
    usage: Optional[dict] = None
    patient: Optional["PatientPreviewResponse"] = Field(
        default=None,
        description="Set when the request carried a `patient`: the id it was given, "
                    "the atoms injected, and how each marker was standardised.",
    )
    prompt_tokens_estimate: int = Field(
        default=0,
        description="Estimated prompt size (characters / PLN_CHARS_PER_TOKEN) checked "
                    "BEFORE the call. A request over PLN_MAX_PROMPT_TOKENS is refused "
                    "with 413 `prompt_too_large` instead of being billed and rejected "
                    "upstream.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Kept for compatibility. A translation failure is now an HTTP "
                    "error (413/429/502/503/504) carrying the same message in "
                    "`detail`, so this is null on every 2xx response.",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable failure class; see core/llm_translator.ERROR_CODES.",
    )
    history: list[HistoryTurn] = Field(description="Updated history — pass back verbatim for the next turn.")


class MettaRunRequest(BaseModel):
    metta_query: str = Field(
        ...,
        max_length=200_000,
        description="Raw MeTTa expression(s) to validate and execute directly — "
                    "skips the LLM translator entirely (no OpenAI call). A "
                    "`(rank-drugage-lifespan (Compound1 Compound2 ...))` form is "
                    "detected and dispatched to the scoped DrugAge engine, same as /query.",
    )
    ontology_files: Optional[list[str]] = Field(
        default=None,
        max_length=PLN_MAX_ONTOLOGY_FILES,
        description="Which .metta files to check symbols against for validation "
                    "(see GET /ontology/files). Defaults to every runtime-safe file — "
                    "unlike /query, there's no LLM context window to economize here, "
                    "but an oversized file excluded from execution is still excluded "
                    "from validation too, so a 'valid' query is one that will actually "
                    "find data. Execution always runs against the same runtime-safe set.",
    )
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0,
        description="PLN results below this confidence are filtered out.",
    )
    patient: Optional[PatientIn] = Field(
        default=None,
        description="A caller-supplied patient, validated and rendered to atoms "
                    "before the query runs. Safer than hand-writing the same "
                    "atoms into `extra_atoms`: ids are namespaced, markers are "
                    "checked against what the KB can reason about, and raw "
                    "values are standardised with the KB's own knobs.",
    )
    allow_definitions: bool = Field(
        default=False,
        description="Permit `(= (…) …)` rule/constant definitions in `extra_atoms`. "
                    "Off by default: a definition does not replace the KB's own, so "
                    "it makes every affected answer — including other patients' — "
                    "non-deterministic.",
    )
    extra_atoms: Optional[str] = Field(
        default=None,
        max_length=500_000,
        description="Optional raw MeTTa text injected into the same space after the KB "
                    "files and before the query runs — e.g. a scratch fact to test a "
                    "hypothetical without writing it to a .metta file.",
    )


    @field_validator("ontology_files")
    @classmethod
    def ontology_files_are_deduped(cls, value):
        return _dedupe_ontology_files(value)


class MettaRunResponse(BaseModel):
    metta_query: str
    patient_id: Optional[str] = Field(
        default=None,
        description="The id given to a caller-supplied `patient` for this request.",
    )
    warnings: list[str] = Field(default_factory=list)
    confidence_threshold_applied: float = Field(
        default=0.0,
        description="The `confidence_threshold` from the request, as applied to "
                    "`pln_results` (entry by entry inside a collapsed result).",
    )
    validation_valid: bool
    validation_issues: list[str]
    validation_warnings: list[str] = Field(default_factory=list)
    ungrounded_predicates: list[str] = Field(
        default_factory=list,
        description="Predicates this query uses that hold zero ground atoms. The "
                    "query still RUNS — it is valid MeTTa — but it cannot match "
                    "anything. See GET /kb/schema.",
    )
    pln_status: str
    pln_mode: str
    pln_query_time_ms: int
    pln_results: list[PLNAtomOut]
    pln_error: Optional[str] = Field(
        default=None,
        description="Set when pln_status is 'error' — e.g. the DrugAge ETL "
                    "output hasn't been generated yet, or the hyperon runtime raised.",
    )
    routed: Optional[str] = Field(
        default=None,
        description="Set to 'drugage_ranking' when metta_query was a "
                    "`(rank-drugage-lifespan ...)` form (see POST /drugage/rank).",
    )


class OntologyFilesResponse(BaseModel):
    files: list[str]
    default_selection: list[str]
    excluded_from_runtime: list[str] = Field(
        default_factory=list,
        description="Discovered files over PLN_MAX_KB_FILE_BYTES, skipped at PLN execution "
                    "time to avoid a hyperon panic. Still queryable in stub mode.",
    )


class PredicateOut(BaseModel):
    name: str
    arity: Optional[int] = None
    fact_count: int
    sources: list[str] = Field(default_factory=list)
    sample_arguments: list[str] = Field(default_factory=list)


class KbSchemaResponse(BaseModel):
    """What the execution KB actually holds — counted, not declared."""
    files: list[str]
    ground_facts: int
    grounded_predicates: list[PredicateOut]
    declared_but_empty_predicates: list[str] = Field(
        description="Declared in the ontology's vocabulary, zero ground atoms. A "
                    "query using one of these is valid MeTTa and returns nothing.",
    )
    entity_count: int
    function_count: int
    type_count: int
    facts_by_file: dict[str, int]
    schema_card: str = Field(
        description="The same information as compact text — the block injected "
                    "into the LLM translator's system prompt.",
    )


class PatientOut(BaseModel):
    id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    smoking: Optional[str] = None


class PatientsResponse(BaseModel):
    patients: list[PatientOut]


class ResolvedMarkerOut(BaseModel):
    marker: str
    z: float
    derived: bool = Field(description="True when the z was computed from a raw value here.")
    raw_value: Optional[float] = None
    unit: Optional[str] = None
    formula: Optional[str] = Field(
        default=None, description="Exactly how a derived z was computed.",
    )
    status: str = Field(description="Elevated | Normal | Low, at the KB's own threshold.")
    note: Optional[str] = None


class PatientPreviewResponse(BaseModel):
    patient_id: str
    atoms: str = Field(description="The MeTTa facts this patient becomes.")
    markers: list[ResolvedMarkerOut]
    warnings: list[str]
    can_predict_risk: bool = Field(
        description="False when age, sex or the AgeAccelGrim clock is missing — "
                    "the risk model returns nothing rather than guessing.",
    )
    age: Optional[float] = None
    sex: Optional[str] = None
    smoking: Optional[str] = None


class MarkerCatalogResponse(BaseModel):
    markers: list[dict]
    z_convention: str
    elevated_threshold: float
    raw_value_note: str


class DrugAgeRankRequest(BaseModel):
    compounds: list[str] = Field(
        ...,
        min_length=1,
        max_length=PLN_MAX_RANK_COMPOUNDS,
        description="DrugAge intervention names, e.g. ['Rapamycin', 'Metformin', 'Resveratrol']. "
                    "Synonyms, abbreviations and Greek letters are resolved (see `resolutions` "
                    f"in the response). At most {PLN_MAX_RANK_COMPOUNDS} per request.",
    )
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0,
        description="Compounds whose calibrated STV confidence is below this are moved to "
                    "`filtered_out` instead of being ranked. NB the confidence you see is "
                    "always 0.9 x the row's evidence tier (the Lifespan -> Mortality chain "
                    "discount): an ITP row reads 0.81, a non-ITP mouse row 0.45. See "
                    "`semantics.confidence_tiers`.",
    )
    strategy: Literal["linear", "metta_sort"] = Field(
        default="linear",
        description="'linear' scores each compound separately and sorts in Python — "
                    "~70 ms per compound, and each compound carries its own truth value "
                    "so confidence_threshold can filter per compound. 'metta_sort' is the "
                    "original single rank-interventions call whose MeTTa insertion sort "
                    "is O(n^2) (n=10 takes ~6 s); it is the reference implementation, "
                    "kept for parity checking.",
    )
    include_rows: bool = Field(
        default=True,
        description="Return every matching DrugAge row (species, sex, significance, "
                    "change percent, PMID), not just the representative one the score "
                    "was computed from.",
    )


class CompoundResolutionOut(BaseModel):
    """How ONE requested compound name was matched against the DrugAge vocabulary."""
    query: str = Field(description="The name exactly as the caller sent it.")
    matched: Optional[str] = Field(
        default=None,
        description="The DrugAge intervention symbol used, or null when the name "
                    "could not be resolved (nothing is ranked for it).",
    )
    method: str = Field(
        description="Which rung matched: exact | normalized | synonym | etl_artifact | "
                    "fuzzy | ambiguous | unmatched. Anything other than exact/normalized "
                    "means the endpoint made a judgement worth reading.",
    )
    score: float = Field(description="Similarity for a `fuzzy` match, else 1.0 / 0.0.")
    note: Optional[str] = Field(default=None, description="Why a synonym/artefact rung fired.")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Nearest DrugAge names for an unresolved or ambiguous request.",
    )


class ScoredCompoundOut(BaseModel):
    """One compound's calibrated, signed effect on mortality."""
    compound: str
    score: float = Field(description="strength x confidence, signed so higher is better.")
    sign: str = Field(description="'Neg' = protective (lowers mortality) | 'Pos' = harmful.")
    direction: str = Field(description="'protective' | 'harmful' — the sign in words.")
    strength: float
    confidence: float
    atom: str = Field(description="The MeTTa (scored ...) tuple this row was parsed from.")


class DrugAgeRowOut(BaseModel):
    """One raw DrugAge experiment row behind a ranked compound."""
    row_id: str
    compound: str
    species: Optional[str] = None
    sex: Optional[str] = None
    is_itp: bool = False
    significance: Optional[str] = None
    avg_lifespan_change_percent: Optional[float] = None
    pmid: Optional[str] = None
    scorable: bool = True


class DrugAgeRankResponse(BaseModel):
    status: str
    mode: str
    query_time_ms: int
    results: list[PLNAtomOut] = Field(
        description="Ranked (scored ...) tuples first, then one provenance line per ranked "
                    "compound (PMID + evidence), then an 'Omitted' note for any requested "
                    "compound with no matching DrugAge row, then one note per name that "
                    "did not match literally.",
    )
    resolutions: list[CompoundResolutionOut] = Field(
        default_factory=list,
        description="One entry per requested compound saying which DrugAge symbol it "
                    "was matched to and how. Read this before trusting an omission: "
                    "'sirolimus' is Rapamycin, 'NMN' is Nicotinamide_mononucleotide.",
    )
    ranked: list[ScoredCompoundOut] = Field(
        default_factory=list,
        description="The ranking as data, most protective first — no atom parsing needed.",
    )
    rows: list[DrugAgeRowOut] = Field(
        default_factory=list,
        description="Every DrugAge row behind the ranked compounds, with species and SEX. "
                    "The score uses ONE representative row per compound (see "
                    "`semantics.representative_row_policy`); this is how you audit that "
                    "choice — e.g. astaxanthin's ITP study reports +12% in males "
                    "(significant) and +3% in females (not significant).",
    )
    rows_truncated: bool = Field(
        default=False,
        description=f"True when more than {PLN_MAX_RANK_ROWS} rows matched and the list was cut.",
    )
    unscorable: list[str] = Field(
        default_factory=list,
        description="Compounds with a matching DrugAge row that reports no average "
                    "lifespan change, so no Effect can be lifted and no score exists. "
                    "Previously these came back as an empty score with no explanation.",
    )
    filtered_out: list[ScoredCompoundOut] = Field(
        default_factory=list,
        description="Compounds that scored but fell below `confidence_threshold`. "
                    "Reported rather than silently dropped.",
    )
    strategy: str = Field(default="linear", description="Which scoring strategy ran.")
    source: str = Field(
        default="",
        description="The DrugAge file the rows came from — the full ETL build or the "
                    "committed 201-row sample are very different datasets.",
    )
    semantics: dict = Field(
        default_factory=dict,
        description="What the numbers mean: sign convention, the strength transform, the "
                    "confidence tiers and the representative-row policy, read off the "
                    ".metta calibration knobs.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Set if the DrugAge ETL output hasn't been generated yet "
                    "(run scripts/run_etl.sh) or the engine raised.",
    )


class ExpandRequest(BaseModel):
    paper_text: str = Field(
        ..., max_length=2_000_000,
        description="Paper text or abstract to extract new ontology entries from.",
    )
    filename: str = Field(
        default="pasted_text.txt",
        description="Source label recorded in the generated MeTTa block's header comment.",
    )
    target_file: Optional[str] = Field(
        default=None,
        description="Existing .metta filename to append to (see GET /ontology/files). "
                    "If it doesn't match an existing file, a new one is created from it "
                    "(or from new_filename) instead.",
    )
    new_filename: Optional[str] = Field(
        default=None,
        description="Stem for a new .metta file, used when target_file isn't an existing file.",
    )
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    apply: bool = Field(
        default=False,
        description="If true, write the extracted entries to disk immediately. "
                    "If false (default), only preview them — call POST /ontology/apply to write.",
    )

    @field_validator("model")
    @classmethod
    def model_must_be_supported(cls, value: str) -> str:
        if value not in AVAILABLE_MODELS:
            raise ValueError(f"model must be one of: {', '.join(AVAILABLE_MODELS)}")
        return value


class ExtractedEntryOut(BaseModel):
    kind: str
    name: str
    metta: str
    description: str


class ExpandResponse(BaseModel):
    paper_title: str
    paper_summary: str
    target_file: str
    new_entries: list[ExtractedEntryOut]
    duplicate_entries: list[ExtractedEntryOut]
    metta_block: str = Field(description="Generated MeTTa block for `new_entries`; pass to POST /ontology/apply.")
    applied: bool
    error: Optional[str] = None


class ApplyRequest(BaseModel):
    metta_block: str = Field(..., description="Typically the `metta_block` from a prior POST /ontology/expand.")
    target_file: str = Field(..., description="Filename to append to, e.g. 'expanded_ontology.metta'.")


class ApplyResponse(BaseModel):
    applied: bool
    target_file: str
    error: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Liveness + config check — confirm the server is reachable before querying."""
    runtime_importable = importlib.util.find_spec("hyperon") is not None
    runtime_ready = PLN_RUNTIME_AVAILABLE and runtime_importable and bool(_runtime_kb_paths())
    return {
        "status": "ok",
        "pln_mode": "runtime" if PLN_RUNTIME_AVAILABLE else "stub",
        "runtime_importable": runtime_importable,
        "runtime_ready": runtime_ready,
        "runtime_kb_file_count": len(_runtime_kb_paths()),
        "openai_key_configured": bool(OPENAI_API_KEY),
        "available_models": AVAILABLE_MODELS,
        "drugage_build_available": BUILD_DRUGAGE.exists(),
        "pln_execution": executor_stats(),
    }


@app.get("/ontology/files", response_model=OntologyFilesResponse)
def ontology_files() -> OntologyFilesResponse:
    """List discovered .metta files, for populating `ontology_files` / `target_file`."""
    all_files = _discover_metta_files()
    choices = list(all_files.keys())
    runtime_names = {p.name for p in _runtime_kb_paths()}
    excluded = sorted(name for name in choices if name not in runtime_names)
    return OntologyFilesResponse(
        files=choices,
        default_selection=_default_selection(choices),
        excluded_from_runtime=excluded,
    )


class DrugAgeTopEntry(BaseModel):
    rank: int
    compound: str
    score: float
    sign: str
    direction: str
    strength: float
    confidence: float
    evidence_tier: str = Field(description="The EvidenceCategory the confidence came from.")
    species: Optional[str] = None
    sex: Optional[str] = None
    is_itp: bool = False
    significance: Optional[str] = None
    avg_lifespan_change_percent: Optional[float] = None
    pmid: Optional[str] = None
    row_id: str


class DrugAgeTopResponse(BaseModel):
    entries: list[DrugAgeTopEntry]
    total_compounds: int = Field(description="Distinct compounds matching the filters.")
    total_rows: int
    scored_rows: int
    unscorable_rows: int = Field(
        description="Rows with no reported average lifespan change, which cannot "
                    "be lifted into an Effect link and therefore have no score.",
    )
    source: str
    filters: dict
    semantics: dict


class HallmarkEvidenceOut(BaseModel):
    record_id: str
    intervention: Optional[str] = None
    hallmark: Optional[str] = None
    species_model: Optional[str] = None
    outcome_text: Optional[str] = None
    reference_number: Optional[int] = None
    publication: Optional[str] = None
    source_file: Optional[str] = None


class InterventionsResponse(BaseModel):
    hallmark: Optional[str] = None
    intervention: Optional[str] = None
    evidence: list[HallmarkEvidenceOut]
    covered_interventions: list[str] = Field(
        description="Every intervention with at least one hallmark evidence record. "
                    "The curated layer is a REVIEW TABLE, not a census: an absence "
                    "here means no record was curated, not that no link exists.",
    )
    covered_hallmarks: list[str]
    note: Optional[str] = None


class HallmarkOut(BaseModel):
    name: str
    components: list[str] = Field(default_factory=list)
    intervention_count: int = 0
    interventions: list[str] = Field(default_factory=list)


class HallmarksResponse(BaseModel):
    hallmarks: list[HallmarkOut]
    evidence_records: int


@app.get("/drugage/top", response_model=DrugAgeTopResponse)
def drugage_top_endpoint(
    n: int = 20,
    species: Optional[str] = None,
    clade: Optional[str] = None,
    min_confidence: float = 0.0,
    itp_only: bool = False,
    significant_only: bool = False,
    direction: Literal["protective", "harmful", "any"] = "protective",
) -> DrugAgeTopResponse:
    """Rank the WHOLE DrugAge build by calibrated effect on mortality. No LLM.

    "Which drugs extend lifespan in mice with the strongest evidence?" — the
    question there was no way to ask. POST /drugage/rank scores a pool the
    caller already knows; this scores all 1,043 compounds and returns the top n.

    Filters compose: `species=Mus_musculus&itp_only=true&min_confidence=0.5`
    is "gold-standard replicated mouse evidence only". `clade` takes
    Vertebrate / Invertebrate / Fungi / Protozoa. `direction=harmful` ranks the
    other end — compounds that SHORTENED lifespan — most harmful first.

    Scoring is done in Python because it cannot be done in the engine: 1,043
    compounds would be 1,043 MeTTa calls, and loading the rows to rank them in
    one space aborts hyperon. It uses the calibration layer's own constants,
    parsed from the .metta files, and a test asserts it agrees with the engine
    bit for bit.
    """
    if n < 1 or n > 500:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_n", "message": "n must be between 1 and 500."},
        )
    top = run_offloaded(
        "drugage_top",
        {"n": n, "species": species, "clade": clade,
         "min_confidence": min_confidence, "itp_only": itp_only,
         "significant_only": significant_only, "direction": direction},
        lambda: drugage_top(
            n=n, species=species, clade=clade, min_confidence=min_confidence,
            itp_only=itp_only, significant_only=significant_only,
            direction=direction,
        ),
    )
    return DrugAgeTopResponse(
        entries=[
            DrugAgeTopEntry(
                rank=i,
                compound=e.row.compound,
                score=e.score,
                sign=e.sign,
                direction="protective" if e.protective else "harmful",
                strength=e.strength,
                confidence=e.confidence,
                evidence_tier=e.tier_category,
                species=e.row.species,
                sex=e.row.sex,
                is_itp=e.row.is_itp,
                significance=e.row.significance,
                avg_lifespan_change_percent=e.row.avg_change,
                pmid=e.row.pmid,
                row_id=e.row.row_id,
            )
            for i, e in enumerate(top.entries, 1)
        ],
        total_compounds=top.total_compounds,
        total_rows=top.total_rows,
        scored_rows=top.scored_rows,
        unscorable_rows=top.unscorable_rows,
        source=top.source,
        filters=top.filters,
        semantics=SCORE_SEMANTICS,
    )


@app.get("/interventions", response_model=InterventionsResponse)
def interventions(
    hallmark: Optional[str] = None,
    intervention: Optional[str] = None,
) -> InterventionsResponse:
    """Which interventions target a hallmark of aging, and vice versa. No LLM.

    Both directions of the question that kept returning empty. The translator
    reached for `TargetsHallmark`, a predicate the ontology declares and nothing
    populates; the relation actually lives in the López-Otín 2023 review-level
    evidence records, which carry the species model, the reported outcome text
    and the reference number.

    Pass `hallmark=MitochondrialDysfunction` or `intervention=Fisetin`, or
    neither to list the whole curated table. An intervention with no record
    comes back as an explicit `note`, not as silence.
    """
    index = hallmark_index(_runtime_kb_paths())
    if hallmark and intervention:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_filter",
                "message": "Pass hallmark OR intervention, not both.",
            },
        )
    if hallmark:
        records = index.for_hallmark(hallmark)
    elif intervention:
        records = index.for_intervention(intervention)
    else:
        records = index.records()

    note = None
    if hallmark and not records:
        note = (
            f"No curated intervention-evidence record names the hallmark "
            f"'{hallmark}'. Known hallmarks with records: "
            f"{', '.join(index.covered_hallmarks())}."
        )
    elif intervention and not records:
        note = (
            f"'{intervention}' has no curated hallmark-evidence record. The "
            f"hallmark layer is a review TABLE (López-Otín 2023, Table 1), not a "
            f"census, so this means no record was curated — not that the "
            f"intervention has no mechanism. Covered interventions: "
            f"{', '.join(index.interventions())}."
        )

    return InterventionsResponse(
        hallmark=hallmark,
        intervention=intervention,
        evidence=[HallmarkEvidenceOut(**r.as_dict()) for r in records],
        covered_interventions=index.interventions(),
        covered_hallmarks=index.covered_hallmarks(),
        note=note,
    )


@app.get("/hallmarks", response_model=HallmarksResponse)
def hallmarks() -> HallmarksResponse:
    """Every hallmark of aging in the KB, with its anchors and interventions."""
    index = hallmark_index(_runtime_kb_paths())
    names = sorted(index.hallmarks | set(index.components))
    out: list[HallmarkOut] = []
    for name in names:
        records = index.for_hallmark(name)
        out.append(HallmarkOut(
            name=name,
            components=sorted(index.components.get(name, [])),
            intervention_count=len({r.intervention for r in records if r.intervention}),
            interventions=sorted({r.intervention for r in records if r.intervention}),
        ))
    return HallmarksResponse(hallmarks=out, evidence_records=len(index.records()))


@app.get("/kb/schema", response_model=KbSchemaResponse)
def kb_schema() -> KbSchemaResponse:
    """What the knowledge base actually holds: predicates, arities, fact counts.

    The answer to "list your data sources and counts", and the reference a
    caller needs before trusting an empty result. The ontology DECLARES a larger
    vocabulary than it populates — `TargetsHallmark`, `Causes`, `Predicts`,
    `Extends`, the gene predicates and the DrugAge row predicates are all
    declared with zero facts in the generic runtime — so a query over one of
    them is well-formed and returns nothing. Those are listed separately here
    (and flagged per-query as `ungrounded_predicates`), because "the KB has no
    such relation" and "the answer is no" are very different statements.

    Counted from the ground atoms of the runtime-safe file set, not from type
    declarations.
    """
    paths = _runtime_kb_paths()
    inv = inventory_for(paths)
    by_file: dict[str, int] = {}
    for path in paths:
        by_file[path.name] = inventory_for([path]).fact_total()
    return KbSchemaResponse(
        files=inv.files,
        ground_facts=inv.fact_total(),
        grounded_predicates=[
            PredicateOut(
                name=p.name,
                arity=p.arity,
                fact_count=p.fact_count,
                sources=sorted(p.sources),
                sample_arguments=list(p.sample_args),
            )
            for p in inv.grounded_predicates()
        ],
        declared_but_empty_predicates=sorted(inv.declared_only),
        entity_count=len(inv.entities),
        function_count=len(inv.functions),
        type_count=len(inv.types),
        facts_by_file=by_file,
        schema_card=schema_card(inv, max_predicates=200),
    )


@app.get("/patients", response_model=PatientsResponse)
def patients() -> PatientsResponse:
    """Known patient profiles (from patient_profile.metta).

    Patients are static KB facts, not something you submit — use one of these
    IDs as the <Patient> argument in a /query question ("what's Patient001's
    10-year CHD risk?") or a dedicated /metta/run form
    (`(predict-risk-patient &self Patient001)`).
    """
    return PatientsResponse(patients=[PatientOut(**p) for p in _patient_summaries()])


@app.get("/patients/markers", response_model=MarkerCatalogResponse)
def patient_markers() -> MarkerCatalogResponse:
    """Which biomarkers a caller-supplied patient may carry, and in what units.

    Read this before POSTing a patient. It also says which raw-value
    conversions exist and flags them as provisional — there is no calibrated
    age/sex-stratified reference table in this knowledge base, so a conversion
    from mg/L or mg/dL uses a documented coarse prior. Sending `z` directly
    bypasses that entirely and is always exact.
    """
    _, elevated = _patient_knobs()
    return MarkerCatalogResponse(
        markers=marker_catalog(),
        z_convention=(
            "z = standard deviations from the AGE- AND SEX-ADJUSTED population "
            "mean for that marker. Positive is above the norm. Because the "
            "adjustment is already baked into z, age and sex enter the risk "
            "model only through the baseline table."
        ),
        elevated_threshold=elevated,
        raw_value_note=(
            "A `value` is standardised server-side with the reference shown here "
            "and reported back with `derived: true` and the exact formula. Those "
            "references are CURATED PRIORS, not calibrated cohort statistics — "
            "this repository contains no reference table. Send `z` when you have "
            "a properly standardised measurement."
        ),
    )


@app.post("/patients/preview", response_model=PatientPreviewResponse)
def patients_preview(patient: PatientIn) -> PatientPreviewResponse:
    """Validate a patient and show exactly what it becomes — no inference, no LLM.

    Use it to check a payload before spending a query on it: it returns the
    generated atoms, each marker's z (and how it was derived), the
    Elevated/Normal/Low status at the KB's own threshold, and whether the
    patient carries enough to get an absolute risk.
    """
    built = _build_caller_patient(patient.model_dump())
    assert built is not None
    return PatientPreviewResponse(
        patient_id=built.patient_id,
        atoms=built.atoms,
        markers=[ResolvedMarkerOut(**m.as_dict()) for m in built.markers],
        warnings=built.warnings,
        can_predict_risk=built.can_predict_risk,
        age=built.age,
        sex=built.sex,
        smoking=built.smoking,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Ask a natural-language question of the PLN knowledge base.

    Equivalent to typing into the "PLN Query" tab and clicking Send — runs
    translate -> validate -> run_query -> format_bot_response and returns
    every intermediate result as structured JSON (not just the rendered text).
    A translated `(rank-drugage-lifespan ...)` query is dispatched to the
    scoped DrugAge engine instead (see `routed` in the response).
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty.")

    selected = req.ontology_files
    if selected is None:
        selected = _default_selection(list(_discover_metta_files().keys()))
    else:
        _validate_ontology_files(selected)
    registry, raw_contents = _build_context(selected)
    system_prompt = build_system_prompt(registry, raw_contents, _runtime_inventory())

    patient = _build_caller_patient(req.patient.model_dump() if req.patient else None)
    if patient is not None:
        # The translator has to know the id exists, or it will answer "I cannot
        # compute a personalized risk from the current KB" — which is what the
        # evaluation saw for a 45-year-old woman with LDL 130 and CRP 4.
        system_prompt += (
            "\n\n--- THIS REQUEST'S PATIENT ---\n"
            f"The caller submitted a patient, loaded for this request only:\n"
            f"{patient.atoms}\n"
            f"Treat `{patient.patient_id}` as a valid <Patient> for every "
            f"dedicated patient form. When the question says 'me', 'my', 'this "
            f"patient' or gives no id, it means {patient.patient_id}.\n"
        )

    history_msgs = [turn.model_dump() for turn in req.history]
    prompt_tokens_estimate = _guard_prompt_size(
        system_prompt, req.message, history_msgs, selected
    )

    translation = translate(
        user_message=req.message,
        system_prompt=system_prompt,
        history=history_msgs,
        model=req.model,
        temperature=req.temperature,
    )

    if not translation.ok:
        _raise_translation_failure(translation)

    routed: Optional[str] = None
    drugage_compounds = parse_drugage_query(translation.metta_query)
    if drugage_compounds is not None:
        if not drugage_compounds:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_drugage_query",
                    "message": "rank-drugage-lifespan requires at least one compound.",
                },
            )
        routed = "drugage_ranking"
        validation = ValidationResult(valid=True)
        pln_result = run_offloaded(
            "route_drugage_ranking",
            {"compounds": drugage_compounds,
             "confidence_threshold": req.confidence_threshold},
            lambda: route_drugage_ranking(
                drugage_compounds,
                confidence_threshold=req.confidence_threshold,
            ),
        )
    else:
        validation = validate(
            translation.metta_query, registry, _runtime_inventory()
        )
        kb_files = _runtime_kb_paths()
        extra_atoms = patient.atoms if patient is not None else None
        pln_result = run_offloaded(
            "run_query",
            {"metta_query": translation.metta_query,
             "confidence_threshold": req.confidence_threshold,
             "kb_files": kb_files,
             "extra_atoms": extra_atoms},
            lambda: run_query(
                metta_query=translation.metta_query,
                confidence_threshold=req.confidence_threshold,
                kb_files=kb_files,
                extra_atoms=extra_atoms,
            ),
        )

    if pln_result.status == "error":
        _raise_pln_failure(
            pln_result,
            stage="pln_execution",
            extra={
                "metta_query": translation.metta_query,
                "explanation": translation.explanation,
            },
        )

    answer = format_bot_response(
        translation=translation,
        validation=validation,
        pln_result=pln_result,
        show_metta=req.show_metta,
        show_explanation=req.show_explanation,
        show_debug=req.show_debug,
    )

    known_ids = _known_patient_ids() | (
        {patient.patient_id} if patient is not None else set()
    )
    patient_warning = _unknown_patient_warning(translation.metta_query, known_ids)
    warnings = list(translation.warnings)
    if patient_warning:
        warnings.append(patient_warning)
    if patient is not None:
        warnings.extend(patient.warnings)

    log_turn(req.message, translation, pln_result)

    updated_history = history_msgs + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": answer},
    ]

    return QueryResponse(
        answer=answer,
        metta_query=translation.metta_query,
        explanation=translation.explanation,
        intent=translation.intent,
        requires_pln_inference=translation.requires_pln_inference,
        confidence_filter=translation.confidence_filter,
        confidence_threshold_applied=req.confidence_threshold,
        warnings=warnings,
        patient=(
            PatientPreviewResponse(
                patient_id=patient.patient_id,
                atoms=patient.atoms,
                markers=[ResolvedMarkerOut(**m.as_dict()) for m in patient.markers],
                warnings=patient.warnings,
                can_predict_risk=patient.can_predict_risk,
                age=patient.age, sex=patient.sex, smoking=patient.smoking,
            )
            if patient is not None else None
        ),
        validation_valid=validation.valid,
        validation_issues=validation.issues,
        validation_warnings=validation.warnings,
        ungrounded_predicates=validation.ungrounded_predicates,
        pln_status=pln_result.status,
        pln_mode=pln_result.mode,
        pln_query_time_ms=pln_result.query_time_ms,
        pln_results=[
            PLNAtomOut(
                atom=r.atom,
                strength=(r.stv or {}).get("strength"),
                confidence=(r.stv or {}).get("confidence"),
            )
            for r in pln_result.results
        ],
        pln_error=pln_result.error,
        routed=routed,
        usage=translation.usage,
        error=translation.error,
        error_code=translation.error_code,
        prompt_tokens_estimate=prompt_tokens_estimate,
        history=[HistoryTurn(**m) for m in updated_history],
    )


@app.post("/metta/run", response_model=MettaRunResponse)
def metta_run(req: MettaRunRequest) -> MettaRunResponse:
    """Validate and execute a raw MeTTa query directly, bypassing the LLM translator.

    For callers that already know the MeTTa they want to run (e.g. an agent
    iterating on queries) — no OpenAI call, and no risk of the translator
    reinterpreting a query you already wrote correctly. Equivalent to the
    validate -> run_query half of /query, skipping the translate step. A
    `(rank-drugage-lifespan ...)` form is dispatched to the scoped DrugAge
    engine, same as /query (see `routed` in the response) — writing that form
    by hand and expecting the generic path to see DrugAge data will not work,
    since that data is deliberately excluded from the generic runtime KB.
    """
    if not req.metta_query.strip():
        raise HTTPException(status_code=422, detail="metta_query must not be empty.")

    _guard_extra_atoms(req.extra_atoms, allow_definitions=req.allow_definitions)
    patient = _build_caller_patient(req.patient.model_dump() if req.patient else None)
    injected = "\n".join(
        part for part in (patient.atoms if patient else None, req.extra_atoms) if part
    ) or None

    routed: Optional[str] = None
    drugage_compounds = parse_drugage_query(req.metta_query)
    if drugage_compounds is not None:
        if not drugage_compounds:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_drugage_query",
                    "message": "rank-drugage-lifespan requires at least one compound.",
                },
            )
        routed = "drugage_ranking"
        validation = ValidationResult(valid=True)
        pln_result = run_offloaded(
            "route_drugage_ranking",
            {"compounds": drugage_compounds,
             "confidence_threshold": req.confidence_threshold},
            lambda: route_drugage_ranking(
                drugage_compounds,
                confidence_threshold=req.confidence_threshold,
            ),
        )
    else:
        if req.ontology_files is not None:
            _validate_ontology_files(req.ontology_files)
        registry = (
            _runtime_registry()
            if req.ontology_files is None
            else _build_context(req.ontology_files)[0]
        )
        if injected:
            registry.merge(parse_metta_text(injected, source_name="<api-extra-atoms>"))
        validation = validate(
            "\n".join(part for part in (injected, req.metta_query) if part),
            registry,
            _runtime_inventory(),
        )
        if not validation.valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_metta_query",
                    "message": "MeTTa validation failed; the query was not executed.",
                    "issues": validation.issues,
                },
            )
        kb_files = _runtime_kb_paths()
        pln_result = run_offloaded(
            "run_query",
            {"metta_query": req.metta_query,
             "confidence_threshold": req.confidence_threshold,
             "kb_files": kb_files,
             "extra_atoms": injected},
            lambda: run_query(
                metta_query=req.metta_query,
                confidence_threshold=req.confidence_threshold,
                kb_files=kb_files,
                extra_atoms=injected,
            ),
        )

    if pln_result.status == "error":
        _raise_pln_failure(
            pln_result, stage="pln_execution", extra={"metta_query": req.metta_query}
        )

    known_ids = _known_patient_ids() | (
        {patient.patient_id} if patient is not None else set()
    )
    patient_warning = _unknown_patient_warning(req.metta_query, known_ids)

    return MettaRunResponse(
        metta_query=req.metta_query,
        patient_id=patient.patient_id if patient is not None else None,
        warnings=([patient_warning] if patient_warning else [])
                 + (patient.warnings if patient is not None else []),
        confidence_threshold_applied=req.confidence_threshold,
        validation_valid=validation.valid,
        validation_issues=validation.issues,
        validation_warnings=validation.warnings,
        ungrounded_predicates=validation.ungrounded_predicates,
        pln_status=pln_result.status,
        pln_mode=pln_result.mode,
        pln_query_time_ms=pln_result.query_time_ms,
        pln_results=[
            PLNAtomOut(
                atom=r.atom,
                strength=(r.stv or {}).get("strength"),
                confidence=(r.stv or {}).get("confidence"),
            )
            for r in pln_result.results
        ],
        pln_error=pln_result.error,
        routed=routed,
    )


@app.post("/drugage/rank", response_model=DrugAgeRankResponse)
def drugage_rank(req: DrugAgeRankRequest) -> DrugAgeRankResponse:
    """Rank real DrugAge compounds by calibrated, signed effect on lifespan/mortality.

    Direct structured entry point to the same scoped engine /query and
    /metta/run dispatch to for a `(rank-drugage-lifespan ...)` form — skip the
    LLM translator (and MeTTa syntax) entirely when you already know the
    compound names. Requires build/drugage_etl.metta (run scripts/run_etl.sh
    first) — as of this writing the engine does NOT fall back to the
    committed 201-row sample despite drugage_etl_short.metta existing in the
    repo; see GET /health's drugage_build_available before calling this.
    """
    ranking = run_offloaded(
        "rank_drugage",
        {"compounds": req.compounds,
         "confidence_threshold": req.confidence_threshold,
         "strategy": req.strategy,
         "include_all_rows": req.include_rows},
        lambda: rank_drugage(
            req.compounds,
            confidence_threshold=req.confidence_threshold,
            strategy=req.strategy,
            include_all_rows=req.include_rows,
        ),
    )
    result = ranking.result
    if result.status == "error":
        _raise_pln_failure(
            result, stage="drugage_ranking", extra={"compounds": req.compounds}
        )
    rows = [_row_out(r) for r in ranking.rows]
    truncated = len(rows) > PLN_MAX_RANK_ROWS
    if truncated:
        rows = rows[:PLN_MAX_RANK_ROWS]

    return DrugAgeRankResponse(
        status=result.status,
        mode=result.mode,
        query_time_ms=result.query_time_ms,
        resolutions=[CompoundResolutionOut(**r.as_dict()) for r in ranking.resolutions],
        ranked=[ScoredCompoundOut(**s.as_dict()) for s in ranking.ranked],
        rows=[DrugAgeRowOut(**r) for r in rows],
        rows_truncated=truncated,
        unscorable=ranking.unscorable,
        filtered_out=[ScoredCompoundOut(**s.as_dict()) for s in ranking.filtered_out],
        strategy=req.strategy,
        source=ranking.source,
        semantics=SCORE_SEMANTICS,
        results=[
            PLNAtomOut(
                atom=r.atom,
                strength=(r.stv or {}).get("strength"),
                confidence=(r.stv or {}).get("confidence"),
            )
            for r in result.results
        ],
        error=result.error,
    )


@app.post("/ontology/expand", response_model=ExpandResponse)
def ontology_expand(req: ExpandRequest) -> ExpandResponse:
    """Extract new PLN ontology entries from pasted paper text.

    Equivalent to the "Ontology Expander" tab's Extract step (and, if
    `apply=true`, the Apply step too).
    """
    if not req.paper_text.strip():
        raise HTTPException(status_code=422, detail="paper_text must not be empty.")

    target_path = _resolve_target_path(req.target_file, req.new_filename)

    result = run_expansion_pipeline(
        paper_data=req.paper_text.encode("utf-8"),
        filename=req.filename,
        target_file_path=target_path,
        model=req.model,
        temperature=req.temperature,
        apply=req.apply,
    )

    if not result.ok:
        raise HTTPException(status_code=400, detail=result.error)

    return ExpandResponse(
        paper_title=result.paper_title,
        paper_summary=result.paper_summary,
        target_file=result.target_file,
        new_entries=[
            ExtractedEntryOut(kind=e.kind, name=e.name, metta=e.metta, description=e.description)
            for e in result.new_entries
        ],
        duplicate_entries=[
            ExtractedEntryOut(kind=e.kind, name=e.name, metta=e.metta, description=e.description)
            for e in result.duplicate_entries
        ],
        metta_block=result.metta_block,
        applied=result.applied,
        error=result.error,
    )


@app.post("/ontology/apply", response_model=ApplyResponse)
def ontology_apply(req: ApplyRequest) -> ApplyResponse:
    """Write a previously-previewed MeTTa block to disk.

    Equivalent to the "Apply to Ontology" button — pairs with a prior
    POST /ontology/expand call made with apply=false.
    """
    if not req.metta_block.strip():
        raise HTTPException(status_code=422, detail="metta_block must not be empty.")

    metta_files = _discover_metta_files()
    name = _normalise_metta_name(req.target_file, field="target_file")
    target_path = metta_files.get(name) or (CUSTOM_ONTOLOGY_DIR / name)
    target_path = _ensure_allowed_target(target_path)

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as fh:
            if target_path.exists() and target_path.stat().st_size > 0:
                fh.write("\n\n")
            fh.write(req.metta_block)
    except OSError as exc:
        return ApplyResponse(applied=False, target_file=target_path.name, error=str(exc))

    return ApplyResponse(applied=True, target_file=target_path.name)


if __name__ == "__main__":
    import uvicorn

    from config import PLN_API_HOST, PLN_API_PORT

    uvicorn.run(app, host=PLN_API_HOST, port=PLN_API_PORT)
