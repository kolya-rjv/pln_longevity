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
    PLN_MAX_KB_FILE_BYTES,
    PLN_RUNTIME_AVAILABLE,
)
from ontology.loader import load_specific_files, parse_metta_text
from ontology.registry import BUILTIN_REGISTRY, OntologyRegistry
from ontology.expander import run_expansion_pipeline
from ontology.drugage_selector import BUILD_DRUGAGE
from core.context_builder import build_system_prompt
from core.drugage_router import parse_drugage_query, route_drugage_ranking
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
    """
    metta_files = _discover_metta_files()
    paths = [metta_files[f] for f in selected_files if f in metta_files]
    if paths:
        return load_specific_files(paths)
    return BUILTIN_REGISTRY, {}


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


# ── Schemas ──────────────────────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)


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
    show_metta: bool = Field(default=True, description="Include the generated MeTTa query in `answer`.")
    show_explanation: bool = Field(default=True, description="Include the NL explanation in `answer`.")
    show_debug: bool = Field(default=False, description="Include token usage / raw LLM response in `answer`.")

    @field_validator("model")
    @classmethod
    def model_must_be_supported(cls, value: str) -> str:
        if value not in AVAILABLE_MODELS:
            raise ValueError(f"model must be one of: {', '.join(AVAILABLE_MODELS)}")
        return value


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
    confidence_filter: float
    warnings: list[str]
    validation_valid: bool
    validation_issues: list[str]
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
    error: Optional[str] = Field(default=None, description="Set when the LLM translation step failed.")
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
    extra_atoms: Optional[str] = Field(
        default=None,
        max_length=500_000,
        description="Optional raw MeTTa text injected into the same space after the KB "
                    "files and before the query runs — e.g. a scratch fact to test a "
                    "hypothetical without writing it to a .metta file.",
    )


class MettaRunResponse(BaseModel):
    metta_query: str
    validation_valid: bool
    validation_issues: list[str]
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


class PatientOut(BaseModel):
    id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    smoking: Optional[str] = None


class PatientsResponse(BaseModel):
    patients: list[PatientOut]


class DrugAgeRankRequest(BaseModel):
    compounds: list[str] = Field(
        ...,
        min_length=1,
        description="DrugAge intervention names, e.g. ['Rapamycin', 'Metformin', 'Resveratrol']. "
                    "Matched case-/separator-insensitively against DrugAge rows. Keep the pool "
                    "to a handful — the underlying ranking is ~O(n^2).",
    )
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0,
        description="Ranked results below this STV confidence are filtered out.",
    )


class DrugAgeRankResponse(BaseModel):
    status: str
    mode: str
    query_time_ms: int
    results: list[PLNAtomOut] = Field(
        description="Ranked (scored ...) tuples first, then one provenance line per ranked "
                    "compound (PMID + evidence), then an 'Omitted' note for any requested "
                    "compound with no matching DrugAge row.",
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


@app.get("/patients", response_model=PatientsResponse)
def patients() -> PatientsResponse:
    """Known patient profiles (from patient_profile.metta).

    Patients are static KB facts, not something you submit — use one of these
    IDs as the <Patient> argument in a /query question ("what's Patient001's
    10-year CHD risk?") or a dedicated /metta/run form
    (`(predict-risk-patient &self Patient001)`).
    """
    return PatientsResponse(patients=[PatientOut(**p) for p in _patient_summaries()])


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
    system_prompt = build_system_prompt(registry, raw_contents)

    history_msgs = [turn.model_dump() for turn in req.history]

    translation = translate(
        user_message=req.message,
        system_prompt=system_prompt,
        history=history_msgs,
        model=req.model,
        temperature=req.temperature,
    )

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
        pln_result = route_drugage_ranking(
            drugage_compounds,
            confidence_threshold=req.confidence_threshold,
        )
    else:
        validation = validate(translation.metta_query, registry)
        pln_result = run_query(
            metta_query=translation.metta_query,
            confidence_threshold=req.confidence_threshold,
            kb_files=_runtime_kb_paths(),
        )

    answer = format_bot_response(
        translation=translation,
        validation=validation,
        pln_result=pln_result,
        show_metta=req.show_metta,
        show_explanation=req.show_explanation,
        show_debug=req.show_debug,
    )

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
        warnings=translation.warnings,
        validation_valid=validation.valid,
        validation_issues=validation.issues,
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
        pln_result = route_drugage_ranking(
            drugage_compounds,
            confidence_threshold=req.confidence_threshold,
        )
    else:
        if req.ontology_files is not None:
            _validate_ontology_files(req.ontology_files)
        registry = (
            _runtime_registry()
            if req.ontology_files is None
            else _build_context(req.ontology_files)[0]
        )
        if req.extra_atoms:
            registry.merge(parse_metta_text(req.extra_atoms, source_name="<api-extra-atoms>"))
        validation = validate(
            "\n".join(part for part in (req.extra_atoms, req.metta_query) if part),
            registry,
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
        pln_result = run_query(
            metta_query=req.metta_query,
            confidence_threshold=req.confidence_threshold,
            kb_files=_runtime_kb_paths(),
            extra_atoms=req.extra_atoms,
        )

    return MettaRunResponse(
        metta_query=req.metta_query,
        validation_valid=validation.valid,
        validation_issues=validation.issues,
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
    result = route_drugage_ranking(req.compounds, confidence_threshold=req.confidence_threshold)

    return DrugAgeRankResponse(
        status=result.status,
        mode=result.mode,
        query_time_ms=result.query_time_ms,
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
