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
from functools import lru_cache
import re
import secrets
import sys
import time
from pathlib import Path

# Ensure the pln_chat package root is on sys.path so submodule imports work
# whether the file is run directly or via `python -m`, matching app.py.
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import (
    AVAILABLE_MODELS,
    CUSTOM_ONTOLOGY_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ONTOLOGY_DIR,
    OPENAI_API_KEY,
    PLN_API_KEYS,
    PLN_API_VERSION,
    PLN_CORS_ORIGINS,
    PLN_RATE_LIMIT_PER_MINUTE,
    PLN_CHARS_PER_TOKEN,
    PLN_MAX_KB_FILE_BYTES,
    PLN_MAX_ONTOLOGY_FILES,
    PLN_MAX_PROMPT_TOKENS,
    PLN_PROMPT_FILE_MAX_BYTES,
    PLN_MAX_METTA_SORT_COMPOUNDS,
    PLN_MAX_RANK_COMPOUNDS,
    PLN_MAX_RANK_ROWS,
    PLN_RUNTIME_AVAILABLE,
)
from ontology.hallmarks import hallmark_index
from ontology.human_evidence import human_evidence_index
from ontology.lever_gate import lever_warnings
from ontology.scoped_forms import scoped_form_warnings
from ontology.write_gate import OntologyWriteRefused, guard_ontology_write
from ontology.inventory import (
    inventory_for,
    schema_card,
    summarise_oversized,
)
from ontology.loader import load_specific_files
from ontology.registry import BUILTIN_REGISTRY, OntologyRegistry
from ontology.expander import run_expansion_pipeline
from ontology.drugage_scoring import load_knobs
from ontology.drugage_selector import BUILD_DRUGAGE
from ontology.gene_index import (
    MAX_LIMIT as GENE_MAX_LIMIT,
    NON_HUMAN_SOURCES,
    SOURCE_KEYS,
    gene_index,
)
from core.context_builder import build_system_prompt
from core.patient_builder import (
    MARKERS,
    BuiltPatient,
    PatientSpecError,
    marker_catalog,
)
from core.rate_limit import TokenBucketLimiter
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
    DrugAgePoolTooLarge,
    guard_compound_pool,
    parse_drugage_query,
    rank_drugage,
    resolve_compounds,
    route_drugage_ranking,
)
from core.llm_translator import translate
from core.metta_validator import ValidationResult, merge_validation_results, validate
from core.pln_runner import (
    CELLAGE_STACK,
    LINAGE2_GENERATED_BASELINE,
    PLNRunResult,
    linage2_patient_kb,
    merge_run_results,
    patient_stack,
    run_cellage_effects,
    run_query,
    run_query_parts,
)
from core.linage2_builder import feature_listing as linage2_feature_listing
from core.patient_extract import OpenAIExtractor
from core.patient_read import read_patient
from core.patient_context import (
    build_caller_patient,
    names_a_patient,
    patient_knobs,
    patient_prompt_section,
    validation_text,
    with_injected,
)
from core.linage2_router import (
    DEFAULT_LEVERS as LINAGE2_DEFAULT_LEVERS,
    LINAGE2_FORMS,
    analysis_program as linage2_analysis_program,
    collect_analysis as linage2_collect_analysis,
    SplitProgram,
    linage2_form_warnings,
    nesting_warnings,
    split_linage2_program,
)
from ontology.inventory import inventory_for as _inventory_for_paths
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
    "hallmark_targeting.metta",

    "mechanistic_bridges.metta",
    "pln_deduction.metta",
    "pln_intervention_ranking.metta",
    "pln_abductive_diagnosis.metta",

    "drugage_calibration.metta",

    "patient_profile.metta",
    # The NHANES grounding and baseline layers are NOT here. They run in their own
    # query-scoped space (core.pln_runner.NHANES_PATIENT_STACK), for the reason measured
    # below _runtime_kb_paths: this shared space is saturated on DISTINCT HEAD SYMBOLS,
    # 201 of them, with no margin — adding the ~9 those layers introduce aborts the
    # process. Narrowing execution to this stack bought thousands of rule definitions of
    # headroom but no head-symbol headroom, which is a different budget.
    "pln_counterfactual.metta",
    "pln_risk_prediction.metta",

    "lifestyle_evidence.metta",

    "supplement_evidence.metta",
    "pln_supplement_recommendation.metta",

    "human_evidence.metta",
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
# EXECUTION RUNS THE CURATED STACK, not every .metta in the repo root.
#
# This used to load every discovered file under PLN_MAX_KB_FILE_BYTES, which left the
# engine almost no room: hyperon 0.2.10 aborts the process — a non-unwinding Rust panic in
# its space trie, uncatchable from Python — once one space holds too many rule
# definitions. Measured on this KB, appending trivial definitions:
#
#     full repo root (26 files)   -> aborts with NO padding at all
#     curated stack  (25 files)   -> still answering at +4,096 definitions
#
# Three orders of magnitude, from ONE file: cellage_calibration.metta was in the root set
# but not in the curated stack, so execution paid for it while the translator never saw
# it. It does not need to be here — the CellAge feature builds its own query-scoped space
# (core.pln_runner.CELLAGE_STACK) and loads that file itself, which is the pattern that
# makes this safe, and is why /genes inference keeps working.
#
# A per-file byte limit cannot express this constraint, because the failure belongs to the
# whole space rather than to any one file. Scoping execution to the stack the translator
# is shown is both the smaller space and the more honest one: what runs is now what the
# LLM was told exists. The byte filter is kept as a second line of defence.
#
# Mirrors app.py; keep in sync. Files outside the stack stay discoverable and queryable in
# stub mode, and a feature needing one at runtime should scope its own space.
def _runtime_kb_paths() -> list[Path]:
    discovered = _discover_metta_files()
    kept: list[Path] = []
    for name in _INFERENCE_STACK:
        path = discovered.get(name)
        if path is None:
            continue                      # a stack entry that is not on disk (generated)
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
        # A COPY: /metta/run merges caller-supplied atoms into whatever registry
        # it gets, and handing out the process-global singleton would let one
        # request's scratch facts leak into every later request's symbol table.
        fresh = OntologyRegistry()
        fresh.merge(BUILTIN_REGISTRY)
        return fresh, {}
    registry, raw_contents = load_specific_files(paths)
    raw_contents, _ = summarise_oversized(
        raw_contents, paths, max_bytes=PLN_PROMPT_FILE_MAX_BYTES
    )
    return registry, raw_contents


def _runtime_inventory():
    """What the EXECUTION KB actually holds (ground facts, not declarations)."""
    return inventory_for(_runtime_kb_paths())


def _guard_drugage_pool(compounds: list[str]) -> list[str]:
    """Apply the ranking's per-request cap to a pool that arrived as MeTTa.

    `DrugAgeRankRequest.compounds` carries `max_length=PLN_MAX_RANK_COMPOUNDS`,
    but that only guards `/drugage/rank`. A `(rank-drugage-lifespan (…))` form
    reaching `/query` or `/metta/run` supplies its pool as a MeTTa list, which
    pydantic never sees — so the same engine was reachable uncapped. Same limit,
    same 422 shape, one message that says which door it came through.
    """
    try:
        return guard_compound_pool(compounds)
    except DrugAgePoolTooLarge as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "too_many_compounds",
                "message": str(exc),
                "requested": exc.count,
                "limit": exc.limit,
            },
        ) from exc


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


def _guard_ontology_write(target_path: Path, block: str, *, schema_checked: bool) -> None:
    """Refuse an unsafe ontology append as HTTP, not as a silent success."""
    try:
        guard_ontology_write(
            target_path, block,
            inventory=_runtime_inventory(),
            schema_checked=schema_checked,
        )
    except OntologyWriteRefused as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.as_detail()) from exc


def _append_metta_block(target_path: Path, block: str) -> Optional[str]:
    """Append `block` to `target_path`; return an error message, or None.

    The single place either ontology endpoint touches the disk, so the guard
    above cannot be bypassed by adding a third caller that forgets it.
    """
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        separator = "\n\n" if target_path.exists() and target_path.stat().st_size else ""
        with open(target_path, "a", encoding="utf-8") as fh:
            fh.write(separator + block)
    except OSError as exc:
        return str(exc)
    return None


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

def _patient_knobs() -> tuple[float, float]:
    """`grimaccel-sd-to-years` and `elevated-z-threshold`, read off the KB
    (core.patient_context.patient_knobs — shared with the UI's My Patient tab)."""
    return patient_knobs()


def _known_patient_ids() -> set[str]:
    return {p["id"] for p in _patient_summaries()}


def _build_caller_patient(payload: Optional[dict]) -> Optional[BuiltPatient]:
    """Validate and render a caller's patient, or raise a 422 explaining why."""
    if payload is None:
        return None
    try:
        return build_caller_patient(payload, _known_patient_ids())
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


def _strip_metta_comments(text: str) -> str:
    """Blank out `;` comments and string literals before a structural check.

    MeTTa treats everything after an unquoted `;` as a comment, so
    `(;\n= (baseline-risk-chd $a $s) 0.999)` is a rule definition that no regex
    over the raw text can see — the comment sits between the `(` and the `=`.
    Blanking comments first is what makes the guard below structural rather
    than textual.
    """
    out: list[str] = []
    in_string = False
    in_comment = False
    for ch in text:
        if in_comment:
            out.append("\n" if ch == "\n" else " ")
            if ch == "\n":
                in_comment = False
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if ch == ";" and not in_string:
            in_comment = True
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _guard_extra_atoms(text: Optional[str], *, allow_definitions: bool) -> None:
    if not text or allow_definitions:
        return
    if _DEFINITION_RE.search(_strip_metta_comments(text)):
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
    inventory = _runtime_inventory()
    mentioned = {m for m in _PATIENT_MENTION_RE.findall(query)}
    # `PatientAge`, `PatientSex`, `PatientSmoking` and `PatientProfile` all start
    # with "Patient" and are predicates and types, not patients. Anything the KB
    # knows as a predicate, function or type is not a missing patient id.
    mentioned -= set(inventory.predicates) | set(inventory.functions) | inventory.types
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
    version=PLN_API_VERSION,
)

# Permissive by default so a local agent/script can call this without CORS
# friction during experimentation. Set PLN_CORS_ORIGINS to a comma-separated
# allowlist when a browser client reaches this beyond localhost.
# `expose_headers` matters for the two headers a browser client has to act on:
# the version it is coded against, and how long to wait after a 429.
app.add_middleware(
    CORSMiddleware,
    allow_origins=PLN_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-API-Version", "Retry-After"],
)


# ── Operational guards: optional key auth, optional rate limit ───────────────
# The evaluation's recommendation 12 asked for "API keys, rate limits ...
# versioned responses. Needed before anything beyond an ngrok demo." Both
# controls below are OFF unless an operator configures them, because the
# service IS still an ngrok demo and every existing caller — the Gradio UI, the
# contract tests, the acceptance runner — sends no credential today.
#
# They live in the logging middleware rather than in a FastAPI dependency for
# two reasons: the Gradio UI is mounted on this same app at `/` and is NOT a
# FastAPI route (a dependency could not see it), and a refusal recorded here is
# still recorded by `log_http_request`, so a 401 storm is visible in the log
# like everything else.

#: The header name has been promised by `pln_chat/.env.example` since the API
#: was split out of app.py ("# PLN_API_KEY=some-shared-secret   # if set,
#: requests need header: X-API-Key: <value>"). Nothing read it until now.
API_KEY_HEADER_NAME = "X-API-Key"
_API_KEY_SCHEME_NAME = "PLNApiKey"
_API_KEY_HEADER = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description=(
        "Shared secret, required only when the deployment sets PLN_API_KEY / "
        "PLN_API_KEYS. When it is unset this scheme is absent from the schema "
        "and every endpoint is open."
    ),
)

#: Reachable without a key AND never metered, in every configuration.
#: `/health` is a readiness probe — one that needs a secret reports the wrong
#: thing when the secret is wrong, and one that can be rate-limited reports the
#: service as down when it is merely busy (observed: with the limit at 3/min, a
#: six-call smoke test left `GET /health` answering 429). The schema and docs
#: routes are how an agent DISCOVERS that it needs a key at all. None of the
#: five touch the LLM, the knowledge base or the disk, so exempting them costs
#: nothing a cap would have protected.
_OPEN_PATHS = frozenset({
    "/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect",
})

_HTTP_OPERATIONS = frozenset({
    "get", "put", "post", "delete", "options", "head", "patch", "trace",
})

#: Per-process token bucket; see the HONESTY CONTRACT in core/rate_limit.py.
#: Rebuilt only at import, so changing PLN_RATE_LIMIT_PER_MINUTE needs a
#: restart (tests replace this object directly).
_RATE_LIMITER = TokenBucketLimiter(per_minute=PLN_RATE_LIMIT_PER_MINUTE)


def _is_api_path(path: str) -> bool:
    """True when `path` is one of THIS module's routes.

    Gradio is mounted at `/` on the same application (see app.create_combined_app),
    so "every request" and "every API request" are different sets: the UI issues
    a stream of static-asset and queue-poll requests that must not be metered by
    an API rate limit. Gradio's mount is a Starlette `Mount`, never an
    `APIRoute`, so asking the router which routes are APIRoutes separates the
    two without hard-coding a path list that would rot as endpoints are added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path_regex.match(path):
            return True
    return False


def _api_key_accepted(presented: Optional[str]) -> bool:
    if not presented:
        return False
    # Constant-time compare: the keys are shared secrets, and `==` on a str
    # leaks a prefix-length oracle to anyone who can time the response.
    # Compared as BYTES, not str: Starlette decodes headers as latin-1, and
    # `compare_digest` raises TypeError on a non-ASCII str — which would turn a
    # header containing one high byte into a 500 instead of a 401.
    offered = presented.encode("utf-8", "surrogateescape")
    return any(
        secrets.compare_digest(offered, key.encode("utf-8")) for key in PLN_API_KEYS
    )


def _guard_request(request: Request) -> Optional[JSONResponse]:
    """Refuse a request before it reaches a route, or return None to let it pass.

    Returns a fully-formed JSONResponse in the same `{"detail": {"code", ...}}`
    shape as every other failure in this API (see API.md "Failures are HTTP
    failures"), so a caller needs no new parsing for these two.
    """
    path = request.url.path
    # A CORS preflight carries no credentials by construction and must reach
    # CORSMiddleware, or a browser client sees an opaque failure instead of the
    # 401/429 the real request would get.
    if request.method == "OPTIONS":
        return None

    if PLN_API_KEYS and path not in _OPEN_PATHS:
        presented = request.headers.get(API_KEY_HEADER_NAME)
        if not _api_key_accepted(presented):
            code = "api_key_invalid" if presented else "api_key_required"
            return JSONResponse(
                status_code=401,
                content={"detail": {
                    "code": code,
                    "message": (
                        f"This deployment requires a shared secret in the "
                        f"{API_KEY_HEADER_NAME} header."
                        if code == "api_key_required"
                        else f"The {API_KEY_HEADER_NAME} header was not recognised."
                    ),
                    "header": API_KEY_HEADER_NAME,
                    "open_paths": sorted(_OPEN_PATHS),
                }},
            )

    if _RATE_LIMITER.enabled and path not in _OPEN_PATHS and _is_api_path(path):
        client = request.client.host if request.client else "unknown"
        retry_after = _RATE_LIMITER.check(client)
        if retry_after is not None:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": {
                    "code": "rate_limited",
                    "message": (
                        f"More than {_RATE_LIMITER.per_minute} requests per "
                        f"minute from this address; retry in {retry_after}s."
                    ),
                    "limit_per_minute": _RATE_LIMITER.per_minute,
                    "retry_after_seconds": retry_after,
                }},
            )
    return None


#: FastAPI caches `app.openapi_schema` after the first build. The schema has to
#: track whether auth is configured, so the cache is keyed on that flag instead
#: of being a one-shot: a deployment that sets PLN_API_KEY publishes a document
#: in which every non-open operation carries `security`, and one that does not
#: publishes a document with no `securitySchemes` at all. Both are true
#: statements about that deployment, which is the point — an agent pointed at
#: /openapi.json discovers the header it needs, or discovers it needs none.
_AUTH_FLAG = "x-pln-api-key-required"
_default_openapi = app.openapi


def _openapi_with_optional_auth() -> dict:
    enabled = bool(PLN_API_KEYS)
    cached = app.openapi_schema
    if cached is not None and cached.get(_AUTH_FLAG) == enabled:
        return cached
    app.openapi_schema = None
    schema = _default_openapi()
    schema[_AUTH_FLAG] = enabled
    if enabled:
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})[_API_KEY_SCHEME_NAME] = (
            _API_KEY_HEADER.model.model_dump(
                by_alias=True, exclude_none=True, mode="json"
            )
        )
        requirement = [{_API_KEY_SCHEME_NAME: []}]
        for path, item in schema.get("paths", {}).items():
            if path in _OPEN_PATHS:
                continue
            for method, operation in item.items():
                if method.lower() in _HTTP_OPERATIONS and isinstance(operation, dict):
                    operation["security"] = requirement
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_optional_auth


@app.middleware("http")
async def log_api_request(request: Request, call_next):
    """Record every API request, including raw JSON/text bodies and failures.

    Also the place the two operational guards run, and where every response —
    including Gradio's — is stamped with `X-API-Version`.
    """
    started = time.monotonic()
    # The guards run FIRST, before the body is read or written to the log. An
    # unauthenticated or rate-limited caller should not be able to put a
    # megabyte of their choosing into this service's session log, and a refused
    # request has no business being buffered into memory at all.
    refusal = _guard_request(request)
    if refusal is not None:
        refusal.headers["X-API-Version"] = PLN_API_VERSION
        log_http_request(
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            body="<not read: request refused before the body was consumed>",
            status_code=refusal.status_code,
            duration_ms=int((time.monotonic() - started) * 1000),
            client=request.client.host if request.client else None,
            content_type=request.headers.get("content-type"),
            user_agent=request.headers.get("user-agent"),
            error=None,
        )
        return refusal

    raw_body = await request.body()
    body = raw_body.decode("utf-8", errors="replace")
    status_code = 500
    error: Optional[str] = None
    try:
        response = await call_next(request)
        response.headers["X-API-Version"] = PLN_API_VERSION
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


class LinAge2ContributionIn(BaseModel):
    """One entry of the LinAge2 service's `feature_contributions`."""
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(description="NHANES variable code, e.g. LBXCRP (GET /linage2/features).")
    contribution_years: float = Field(description="This input's share of the BA-CA delta, in years.")
    is_imputed: Optional[bool] = Field(
        default=None,
        description="True when the service filled the value in from its reference cohort. "
                    "Falls back to membership in `imputed_features` when omitted.",
    )


class LinAge2MetadataIn(BaseModel):
    """The `metadata` object of a LinAge2 `/predict` response."""
    model_config = ConfigDict(extra="forbid")

    chronological_age: float
    delta_ba_ca: float
    feature_contributions: list[LinAge2ContributionIn] = Field(min_length=1, max_length=64)
    imputed_features: Optional[list[str]] = None
    features_used: Optional[int] = None
    total_features: Optional[int] = None
    warnings: Optional[list[str]] = None


class LinAge2In(BaseModel):
    """A LinAge2 service response, passed through as the client received it.

    Either the `/predict` response body (`biological_age` + `metadata`), or the
    flattened shape with `chronological_age`, `delta_ba_ca` and
    `feature_contributions` at the top level. `code` and `message` are accepted so
    the body can be forwarded verbatim; they are ignored.
    """
    model_config = ConfigDict(extra="forbid")

    code: Optional[int] = None
    message: Optional[str] = None
    biological_age: float
    metadata: Optional[LinAge2MetadataIn] = None
    chronological_age: Optional[float] = None
    delta_ba_ca: Optional[float] = None
    feature_contributions: Optional[list[LinAge2ContributionIn]] = Field(default=None, max_length=64)
    imputed_features: Optional[list[str]] = None
    features_used: Optional[int] = None
    total_features: Optional[int] = None
    warnings: Optional[list[str]] = None


class PatientIn(BaseModel):
    """A patient the CALLER supplies, scored for this request only.

    Nothing is written to disk: the atoms live in the query's hyperon space(s) and
    disappear with them. (A LinAge2 block's atoms go only to the LinAge2 scoped
    space; the shared space gets the rest — BuiltPatient.shared_atoms.) The id is namespaced `Caller_…` so it can never collide
    with a curated patient — submitting a second `Patient001` does not replace
    the first, it unions both and makes every answer non-deterministic.

    `extra="forbid"` is load-bearing. `/query` and `/metta/run` take this object
    NESTED, under a `patient` key; `/patients/preview` takes it at the top
    level. With extras ignored, posting the nested shape to /patients/preview
    returned 200 and a confident, well-formed description of an EMPTY patient —
    the whole payload silently discarded, every documented refusal
    (`unknown_marker`, `invalid_sex`) bypassed because nothing was ever read.
    An endpoint whose stated job is "check a payload before spending a query on
    it" must not answer for a payload it did not look at.
    """
    model_config = ConfigDict(extra="forbid")

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
    linage2: Optional[LinAge2In] = Field(
        default=None,
        description="The LinAge2 service's /predict response for THIS patient, "
                    "forwarded as received. Becomes the LinAgeAccel clock marker plus "
                    "one LinAgeContribution atom per model input, for this request "
                    "only, and unlocks the `linage-*` forms (GET /linage2/features, "
                    "POST /linage2/analyze). Send the patient's own CRP / HbA1c / "
                    "FastingGlucose z (or value) and smoking status alongside: the "
                    "engine credits a cause to a LinAge2 contribution only when the "
                    "patient's own value witnesses the direction.",
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
                    "are injected into this request's space(s) only — a LinAge2 "
                    "block's into the LinAge2 scoped space, the rest into the "
                    "shared one. Mention the returned id (or just say 'my "
                    "patient') in `message`.",
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
                    "DrugAge engine instead of the generic KB (see POST /drugage/rank); "
                    "'linage2' when it called a `(linage-… &self <Patient> …)` form and "
                    "ran in the LinAge2 scoped space (see GET /linage2/features); "
                    "'linage2+generic' when it mixed such forms with others — the "
                    "LinAge2 forms ran in the scoped space, the rest in the generic "
                    "one, and the results are joined in program order.",
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
                    "`(rank-drugage-lifespan ...)` form (see POST /drugage/rank); "
                    "'linage2' when it called a `linage-*` form, which runs in the "
                    "LinAge2 query-scoped space rather than the generic one; "
                    "'linage2+generic' when it mixed `linage-*` forms with others, "
                    "each part running in its own space.",
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
    has_linage2: bool = Field(
        default=False,
        description="True when a LinAge2 delta is present (a `linage2` block or a bare "
                    "LinAgeAccel marker), so the `linage-*` forms have input.",
    )
    linage2: Optional[dict] = Field(
        default=None,
        description="Set when the request carried a `linage2` block: the delta, its z "
                    "and status, and every contribution (measured first, by |years|) "
                    "with the KB biomarker it reads out, if any.",
    )


class MarkerCatalogResponse(BaseModel):
    markers: list[dict]
    z_convention: str
    elevated_threshold: float
    raw_value_note: str
    linage2: dict = Field(
        default_factory=dict,
        description="How to send a LinAge2 clinical-clock result: the `linage2` block, "
                    "what it unlocks, and where the feature vocabulary is published.",
    )


class LinAge2FeaturesResponse(BaseModel):
    features: list[dict] = Field(
        description="Every LinAge2 model input the KB declares: NHANES code, KB symbol, "
                    "description, the KB biomarker it reads out (4 of 59), and how a "
                    "cause gets credited to it.",
    )
    clock: dict
    forms: list[str] = Field(description="The MeTTa forms the LinAge2 layer defines.")
    stack: list[str] = Field(description="The query-scoped files those forms run in.")
    baseline_available: bool = Field(
        description="True when the generated NHANES all-cause baseline is present, so "
                    "linage-risk-patient returns an absolute risk; otherwise only the "
                    "relative hazard is computable and the risk form yields nothing.",
    )


class LinAge2AnalyzeRequest(PatientIn):
    """`/patients/preview`'s top-level patient shape, with a `linage2` block REQUIRED,
    plus the levers to run counterfactuals for."""
    levers: Optional[list[str]] = Field(
        default=None, max_length=8,
        description="Levers for the counterfactuals (a cause, an intervention or a "
                    "marker the KB knows). Defaults to ChronicInflammation, "
                    "CellularSenescence, InsulinResistance, SmokingCessation.",
    )


class LinAge2AnalyzeResponse(BaseModel):
    patient_id: str
    atoms: str
    linage2: dict = Field(description="The validated block, as /patients/preview reports it.")
    decomposition: Optional[dict] = Field(
        default=None,
        description="Every input's years, measured and imputed apart, with the KB "
                    "biomarker it reads out, whether the patient's own value witnesses "
                    "its direction, and the hallmark causes credited (only under a "
                    "witness); plus the totals and the explicit age-term residual.",
    )
    hazard: Optional[dict] = Field(
        default=None,
        description="The relative all-cause-mortality hazard, HR^delta, with confidence. "
                    "Always computable from a delta.",
    )
    risk: Optional[dict] = Field(
        default=None,
        description="An ABSOLUTE ten-year all-cause-mortality risk — only when the "
                    "generated NHANES baseline is loaded (see `risk_note`).",
    )
    risk_note: str
    counterfactuals: list[dict] = Field(
        description="Per lever: the expected change in the LinAge2 delta, in years, "
                    "through the causal graph and the lever's own evidence edge; the "
                    "inputs credited (`via`); 0 with an empty `via` when the lever "
                    "reaches no witnessed input.",
    )
    projected_risks: list[dict] = Field(
        description="The counterfactuals in absolute-risk terms; empty without a baseline.",
    )
    warnings: list[str]
    metta_query: str
    pln_status: str
    pln_query_time_ms: int
    unparsed: list[str] = Field(
        default_factory=list,
        description="Result atoms this endpoint could not read; should be empty.",
    )


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
                    "is O(n^2) (n=10 takes ~6 s, n=20 ~53 s); it is the reference "
                    "implementation, kept for parity checking, and is capped at "
                    f"{PLN_MAX_METTA_SORT_COMPOUNDS} compounds per request because "
                    "beyond that it cannot finish inside the query deadline.",
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
    direction: str = Field(
        description="'protective' | 'harmful' | 'no_effect' — the sign in words. "
                    "'no_effect' is a row that reported 0.0% lifespan change: a "
                    "measured null, which `sign` cannot express (the Effect "
                    "convention has only Pos and Neg). See `semantics.zero_score`.",
    )
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
    identifier: Optional[str] = Field(
        default=None,
        description="PMID_<digits> or DOI the entry rests on. Never null for an "
                    "accepted entry — an entry with neither is rejected.",
    )
    evidence_tier: Optional[str] = Field(
        default=None,
        description="The EvidenceCategory the extractor read off the paper. Its "
                    "CONFIDENCE is not taken from the extractor: the emitted atom "
                    "carries `(evidence-confidence <tier>)`, unevaluated.",
    )
    effect_size_pct: Optional[float] = Field(
        default=None,
        description="Reported percent lifespan change. Present means the Effect "
                    "link's strength was DERIVED from it; absent means the "
                    "strength is a curated prior.",
    )
    provisional: bool = Field(
        default=False,
        description="True when any value in this entry was proposed by the model "
                    "rather than derived — see `provisional_fields`.",
    )
    provisional_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list,
        description="Where each number came from; the same lines appear as `;;` "
                    "comments above the entry in `metta_block`.",
    )


class RejectedEntryOut(BaseModel):
    kind: str
    name: str
    metta: str
    description: str
    codes: list[str] = Field(
        description="Machine-readable refusal codes: unknown_predicate, "
                    "invented_truth_value, invented_confidence_constant, "
                    "redefines_calibration, unknown_evidence_category, "
                    "missing_identifier.",
    )
    reasons: list[str] = Field(description="The same refusals in prose.")


class ExpandResponse(BaseModel):
    paper_title: str
    paper_summary: str
    target_file: str
    new_entries: list[ExtractedEntryOut]
    duplicate_entries: list[ExtractedEntryOut]
    rejected_entries: list[RejectedEntryOut] = Field(
        default_factory=list,
        description="Entries the schema gate refused, with the reason for each. "
                    "Reported rather than silently dropped.",
    )
    metta_block: str = Field(description="Generated MeTTa block for `new_entries`; pass to POST /ontology/apply.")
    unconsumed_predicates: list[str] = Field(
        default_factory=list,
        description="Predicates in `metta_block` the runtime KB grounds nowhere "
                    "else — the block would land them with zero facts to join.",
    )
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

class PLNExecutionOut(BaseModel):
    """How this deployment executes MeTTa, and what it can therefore enforce."""
    mode: str = Field(description="'process_per_query' | 'inline'.")
    max_concurrent: int = Field(
        description="Query processes allowed at once (PLN_WORKER_POOL_SIZE). "
                    "0 means inline, in the request thread.",
    )
    timeout_seconds: Optional[float] = Field(
        description="The per-request PLN budget, or null when it cannot be "
                    "enforced. Inline mode CANNOT enforce it — you cannot "
                    "preempt a GIL-holding Rust call — so it reports null "
                    "rather than advertising a deadline it will not apply.",
    )
    timeout_enforced: bool
    max_inflight: Optional[int] = Field(
        description="Admission limit, or null when there is none.",
    )
    admission_control: bool
    inflight: int = Field(description="Queries running right now.")


class HealthResponse(BaseModel):
    """The preflight an agent is told to call first — so it is typed.

    `/health` was the one route with no `response_model`, which made it an
    untyped object in `/openapi.json` — while API.md tells integrators to hand
    an agent the base URL plus `/openapi.json` and to read `api_key_required`
    and `pln_execution` from here. An agent reading the schema could not see
    the fields it is instructed to branch on.
    """
    status: str
    version: str = Field(
        description="Same string as `info.version` and the X-API-Version "
                    "header: one unauthenticated call tells you which contract "
                    "you are talking to.",
    )
    api_key_required: bool = Field(
        description="True when PLN_API_KEY/PLN_API_KEYS is set and every other "
                    "route needs an `X-API-Key` header.",
    )
    rate_limit_per_minute: int = Field(description="0 when off (the default).")
    pln_mode: str = Field(description="'runtime' | 'stub'.")
    runtime_importable: bool = Field(description="Is `hyperon` importable here?")
    runtime_ready: bool = Field(
        description="Importable AND enabled AND at least one KB file loadable.",
    )
    runtime_kb_file_count: int
    openai_key_configured: bool = Field(
        description="False means /query returns 503; the LLM-free endpoints "
                    "still work.",
    )
    available_models: list[str]
    drugage_build_available: bool = Field(
        description="False means the DrugAge rankings answer 503 — run "
                    "`bash scripts/run_etl.sh`.",
    )
    pln_execution: PLNExecutionOut


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """Liveness + config check — confirm the server is reachable before querying."""
    runtime_importable = importlib.util.find_spec("hyperon") is not None
    runtime_ready = PLN_RUNTIME_AVAILABLE and runtime_importable and bool(_runtime_kb_paths())
    return {
        "status": "ok",
        # Same string as `info.version` in /openapi.json and the X-API-Version
        # header, so one unauthenticated call tells a caller which contract it
        # is talking to — and whether it has to send a key to talk further.
        "version": PLN_API_VERSION,
        "api_key_required": bool(PLN_API_KEYS),
        "rate_limit_per_minute": _RATE_LIMITER.per_minute,
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
    direction: str = Field(
        description="'protective' | 'harmful' | 'no_effect'. See "
                    "ScoredCompoundOut.direction.",
    )
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
    total_compounds: int = Field(
        description="Compounds matching the filters that have at least one "
                    "SCORABLE row — the size of the ranking's real universe. "
                    "Over the whole build that is 1,035, not the 1,043 distinct "
                    "compounds DrugAge lists: 8 have no reported lifespan "
                    "change on any row and cannot be scored at all.",
    )
    total_compounds_in_source: int = Field(
        default=0,
        description="Distinct compounds in the filtered rows, scorable or not.",
    )
    unscorable_compounds: int = Field(
        default=0,
        description="Compounds dropped because no row of theirs reports a "
                    "lifespan change. Reported rather than silently missing.",
    )
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


class HallmarkTargetingOut(BaseModel):
    intervention: str
    hallmark: str
    source_file: Optional[str] = None
    publications: list[str] = Field(default_factory=list)
    provenance: Literal["targeting_fact"] = "targeting_fact"


class InterventionsResponse(BaseModel):
    hallmark: Optional[str] = None
    intervention: Optional[str] = None
    evidence: list[HallmarkEvidenceOut]
    targeting: list[HallmarkTargetingOut] = Field(
        default_factory=list,
        description="`(TargetsHallmark …)` facts with no review record behind them "
                    "— currently rapamycin and metformin. A targeting fact says "
                    "WHAT the intervention acts on and nothing more: no species "
                    "model, no reported outcome, no effect size. It is listed "
                    "separately rather than padded into `evidence`, because "
                    "inventing those fields is exactly the failure this API "
                    "exists to avoid. A link already carried by a record is not "
                    "repeated here.",
    )
    covered_interventions: list[str] = Field(
        description="Every intervention with a hallmark link of either kind. "
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


class HumanPublicationOut(BaseModel):
    symbol: str
    title: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None


class HumanStudyOut(BaseModel):
    record_id: str
    intervention: Optional[str] = None
    outcome: Optional[str] = None
    design: Optional[str] = Field(
        default=None,
        description="RandomizedControlledTrial | ObservationalCohort | "
                    "OpenLabelPilot | PlannedTrial. Never flattened to \"a study\".",
    )
    n: Optional[int] = None
    measured: Optional[str] = None
    finding: Optional[str] = None
    result: Optional[str] = Field(
        default=None,
        description="ReportedBenefit | ReportedNull | ReportedHarm | "
                    "NotYetReported. `ReportedNull` means somebody measured it "
                    "in people and found nothing — which is not the same as "
                    "this KB holding no record.",
    )
    evidence_tier: Optional[str] = Field(
        default=None,
        description="An EvidenceCategory from epistemic_calibration.metta, or "
                    "null. Null is meaningful: a PlannedTrial that has not "
                    "reported has no tier, and inventing one for it would be "
                    "the most consequential fabrication available here.",
    )
    caveat: Optional[str] = None
    pmid: Optional[str] = Field(
        default=None,
        description="The PMID written in the record itself; `publication` "
                    "carries the full citation it must agree with.",
    )
    publication: Optional[HumanPublicationOut] = None
    source_file: Optional[str] = None


class HumanCrossReferenceOut(BaseModel):
    intervention: str
    evidence_tier: str
    where: str
    source_file: Optional[str] = None
    provenance: Literal["cross_reference"] = "cross_reference"


class HumanEvidenceResponse(BaseModel):
    intervention: Optional[str] = None
    studies: list[HumanStudyOut]
    cross_references: list[HumanCrossReferenceOut] = Field(
        default_factory=list,
        description="Human evidence this KB holds somewhere else, pointed at "
                    "rather than copied — omega-3's tier lives in "
                    "supplement_evidence.metta. One body of evidence, one home.",
    )
    covered_interventions: list[str] = Field(
        description="Every intervention with a human record or cross-reference. "
                    "This layer is a small hand-curated table, not a census: an "
                    "absence here means no record was curated, not that no human "
                    "study exists.",
    )
    covered_outcomes: list[str]
    note: Optional[str] = None


@app.get("/drugage/top", response_model=DrugAgeTopResponse)
def drugage_top_endpoint(
    n: int = 20,
    species: Optional[str] = None,
    clade: Optional[str] = None,
    min_confidence: float = 0.0,
    itp_only: bool = False,
    significant_only: bool = False,
    direction: Literal["protective", "harmful", "none", "any"] = "protective",
) -> DrugAgeTopResponse:
    """Rank the WHOLE DrugAge build by calibrated effect on mortality. No LLM.

    "Which drugs extend lifespan in mice with the strongest evidence?" — the
    question there was no way to ask. POST /drugage/rank scores a pool the
    caller already knows; this scores every scorable compound in the build (1,035 of the 1,043 DrugAge lists — 8 report no lifespan change anywhere)
    and returns the top n.

    Filters compose: `species=Mus_musculus&itp_only=true&min_confidence=0.5`
    is "gold-standard replicated mouse evidence only". `clade` takes
    Vertebrate / Invertebrate / Fungi / Protozoa. `direction=harmful` ranks the
    other end — compounds that SHORTENED lifespan — most harmful first, and
    `direction=none` returns the measured NULLS (0.0% change), which used to be
    counted as protective because the MeTTa sign convention has no zero.

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
                direction=e.direction,
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
        total_compounds_in_source=top.total_compounds_in_source,
        unscorable_compounds=top.unscorable_compounds,
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

    Both directions of the question that kept returning empty. The answer now
    comes from two shapes, reported separately:

    * `evidence` — the López-Otín 2023 review-level records, which carry the
      species model, the reported outcome text and the reference number;
    * `targeting` — plain `(TargetsHallmark …)` facts from
      `hallmark_targeting.metta`, which carry a target and a publication and
      nothing else. Rapamycin and metformin, the two drugs the KB held no
      hallmark link for at all, arrive this way.

    Pass `hallmark=MitochondrialDysfunction` or `intervention=Rapamycin`, or
    neither to list the whole curated table. An intervention with no link of
    either kind comes back as an explicit `note`, not as silence.
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
        links = index.targeting_for_hallmark(hallmark)
    elif intervention:
        records = index.for_intervention(intervention)
        links = index.targeting_for_intervention(intervention)
    else:
        records = index.records()
        # The whole table: every targeting fact the records do not already say.
        by_record = {
            ((r.intervention or "").lower(), (r.hallmark or "").lower())
            for r in records
        }
        links = [
            t for t in index.targeting
            if (t.intervention.lower(), t.hallmark.lower()) not in by_record
        ]

    subject = hallmark or intervention
    note = None
    if subject and not records and not links:
        if hallmark:
            note = (
                f"No curated intervention-evidence record and no TargetsHallmark "
                f"fact names the hallmark '{hallmark}'. Known hallmarks with "
                f"links: {', '.join(index.covered_hallmarks())}."
            )
        else:
            note = (
                f"'{intervention}' has no curated hallmark-evidence record. The "
                f"hallmark layer is a review TABLE (López-Otín 2023, Table 1), not a "
                f"census, so this means no record was curated — not that the "
                f"intervention has no mechanism. Covered interventions: "
                f"{', '.join(index.interventions())}."
            )
    elif subject and not records and links:
        note = (
            f"'{subject}' is linked by TargetsHallmark facts only — no review "
            f"record backs it, so there is no species model, reported outcome or "
            f"reference number to show. See `targeting` and its `publications`."
        )

    return InterventionsResponse(
        hallmark=hallmark,
        intervention=intervention,
        evidence=[HallmarkEvidenceOut(**r.as_dict()) for r in records],
        targeting=[HallmarkTargetingOut(**t.as_dict()) for t in links],
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
        # Both shapes, deduplicated: a hallmark whose only link to rapamycin is
        # a TargetsHallmark fact still lists rapamycin here.
        linked = index.interventions_for(name)
        out.append(HallmarkOut(
            name=name,
            components=sorted(index.components.get(name, [])),
            intervention_count=len(linked),
            interventions=linked,
        ))
    return HallmarksResponse(hallmarks=out, evidence_records=len(index.records()))


@app.get("/evidence/human", response_model=HumanEvidenceResponse)
def human_evidence(intervention: Optional[str] = None) -> HumanEvidenceResponse:
    """What HUMAN evidence the KB holds for an intervention. No LLM.

    "What does the evidence say about metformin in humans?" used to be answered
    in prose, because the answer was true but un-grounded: the bulk data loaded
    here (DrugAge, GenAge, CellAge) is model-organism and cell evidence, and
    there was nothing to point at. `human_evidence.metta` is the record set;
    this is the way to read it without an LLM in the loop.

    Each study carries its design, its n, what was measured, what was found, the
    PMID and an evidence tier, and the endpoint keeps apart three states that a
    summary sentence destroys:

    * a study with `result="ReportedNull"` — measured in people, nothing found
      (dasatinib + quercetin did not change pulmonary function in 14 people);
    * a study with `evidence_tier=null` and `result="NotYetReported"` — TAME is
      planned and has not reported, so there is no tier to give;
    * no record at all — an explicit `note`, never an unexplained empty list.

    Pass `intervention=Metformin`, or nothing to list the whole table.
    """
    index = human_evidence_index(_runtime_kb_paths())
    if intervention:
        studies = index.for_intervention(intervention)
        crossrefs = index.cross_references_for(intervention)
    else:
        studies = index.records()
        crossrefs = list(index.cross_references)

    note = None
    if intervention and not studies and not crossrefs:
        note = (
            f"No curated human study names '{intervention}'. This layer is a "
            f"small hand-built table, so that is an absence of a RECORD, not "
            f"evidence of absence — and it is emphatically not a null result. "
            f"Interventions with human evidence here: "
            f"{', '.join(index.interventions())}."
        )
    elif intervention and not studies and crossrefs:
        note = (
            f"'{intervention}' has no study record in human_evidence.metta: its "
            f"human evidence is recorded elsewhere in the knowledge base and is "
            f"cross-referenced rather than duplicated. See `cross_references`."
        )
    elif studies and all(s.result == "NotYetReported" for s in studies):
        note = (
            f"Every human record for '{intervention}' is a trial that has not "
            f"reported. There is no result to quote and no evidence tier to "
            f"attach."
        )

    return HumanEvidenceResponse(
        intervention=intervention,
        studies=[HumanStudyOut(**s.as_dict()) for s in studies],
        cross_references=[HumanCrossReferenceOut(**x.as_dict()) for x in crossrefs],
        covered_interventions=index.interventions(),
        covered_outcomes=index.covered_outcomes(),
        note=note,
    )


# ── Gene queries: CellAge + GenAge ───────────────────────────────────────────
# Evaluation §4: the gene data was on disk and unreachable. "Which genes drive
# cellular senescence?" returned four hallmark components; "GenAge human genes"
# was empty; "CellAge ∩ GenAge" could not be expressed; and selecting a gene ETL
# as `ontology_files` blew the prompt past OpenAI's limit. The prompt half is
# fixed (an oversized file becomes a schema card). These routes are the other
# half: the data, answered directly, with no LLM and no hyperon in the default
# path. See ontology/gene_index.py for why the CSVs and not the generated MeTTa.

#: Which columns each source contributes to a gene record. Returned by
#: GET /genes/sources because the record objects are SPARSE — a CellAge row has
#: no `organism` key at all rather than `organism: null`, since a null there
#: would read as "looked and found nothing" instead of "that table has no such
#: column". This map is how an agent learns the shape instead.
GENE_FIELDS_BY_SOURCE: dict[str, list[str]] = {
    "cellage_curated": [
        "senescence_effect", "senescence_direction", "senescence_type",
        "cell_context", "pmids",
    ],
    "cellage_expression": [
        "expression_direction", "expression_samples", "p_value",
    ],
    "genage_human": ["uniprot", "selection_basis"],
    "genage_models": [
        "organism", "lifespan_effect", "longevity_influence",
        "avg_lifespan_change_percent",
    ],
}

#: Every record also carries these.
GENE_COMMON_FIELDS: list[str] = [
    "source", "row_index", "symbol", "entrez", "gene_name", "metta_row_id",
]

#: What a lifted CellAge Effect link means, returned beside the links so the
#: numbers can never travel without their provenance.
CELLAGE_EFFECT_SEMANTICS: dict = {
    "link": "(Effect <gene> CellularSenescence <sign> (stv <strength> <confidence>))",
    "sign": "Pos = the gene INDUCES cellular senescence; Neg = it INHIBITS it.",
    "strength": "A curated prior from cellage_calibration.metta §1, IDENTICAL for "
                "every curated row. CellAge records a direction and no effect "
                "size, so there is no magnitude to report and none is invented.",
    "confidence": "(evidence-confidence InVitro) from epistemic_calibration.metta "
                  "— CellAge curation is cell-line experimental evidence.",
    "not_the_etl_numbers": "build/cellage_genes.metta also carries "
                           "(Causes <gene> (Increases CellularSenescence) "
                           "(stv 0.82 0.70)). Those are computed by "
                           "cellage_etl.calibrated_stv from the senescence type "
                           "and the cancer-cell flag; they are NOT calibrated and "
                           "this layer ignores them.",
    "one_link_per_row": "A gene with three curated rows yields three links, not a "
                        "merged verdict. Averaging them would report a number no "
                        "row states.",
    "no_mortality_bridge": "CellularSenescence is a hallmark, not an outcome. This "
                           "layer does NOT chain senescence to mortality: "
                           "senescence is tumour-suppressive in a cancer cell and "
                           "damaging in an ageing tissue, and the KB holds no "
                           "evidence that fixes that sign.",
}


class GeneSourceOut(BaseModel):
    source: str
    label: str
    available: bool = Field(
        description="False means the table could not be read at all — an absence "
                    "of DATA, never an assertion about a gene.",
    )
    rows: int
    file: Optional[str] = None
    origin: Optional[str] = Field(
        default=None,
        description="\"csv\" (the unpacked table under data/, gitignored) or "
                    "\"zip\" (the committed archive, read in memory).",
    )
    note: Optional[str] = None
    fields: list[str] = Field(
        default_factory=list,
        description="The source-specific keys a record from this source carries, "
                    "on top of the common ones.",
    )


class GeneSourcesResponse(BaseModel):
    sources: list[GeneSourceOut]
    common_fields: list[str]
    total_records: int
    distinct_entrez: int
    distinct_symbols: int
    vocabularies: dict = Field(
        description="The exact values the /genes filters accept, read off the "
                    "data rather than hard-coded.",
    )
    note: str


class GeneLookupResponse(BaseModel):
    query: str
    resolved_as: Optional[str] = Field(
        default=None,
        description="\"entrez\" when the key was all digits, else \"symbol\".",
    )
    entrez: Optional[int] = None
    symbols: list[str] = Field(default_factory=list)
    gene_name: Optional[str] = None
    sources: list[str] = Field(
        default_factory=list,
        description="Every source holding this gene. Two entries that include "
                    "both cellage_curated and genage_human ARE the intersection.",
    )
    records: list[dict] = Field(
        default_factory=list,
        description="One entry per (source, row). Sparse: a record carries only "
                    "the keys its own table has — see GET /genes/sources.",
    )
    inference: Optional[dict] = Field(
        default=None,
        description="Present only with `infer=true`: the CellAge rows lifted into "
                    "calibrated (Effect … CellularSenescence …) links by hyperon.",
    )
    unavailable_sources: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class GeneListResponse(BaseModel):
    records: list[dict]
    total: int = Field(description="Records matching the filters, before paging.")
    returned: int
    truncated: bool = Field(
        description="True when `total` exceeds what was returned, so an empty "
                    "tail is never mistaken for the end of the data.",
    )
    limit: int
    offset: int
    filters: dict
    unavailable_sources: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class GeneIntersectionResponse(BaseModel):
    a: str
    b: str
    key: str
    genes: list[dict]
    total: int
    returned: int
    truncated: bool
    a_genes: int = Field(description="Distinct genes in `a` under this join key.")
    b_genes: int
    warnings: list[str] = Field(
        default_factory=list,
        description="Reasons this particular join may not mean what it looks like.",
    )
    note: Optional[str] = None


def _gene_source_error(value: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "code": "unknown_source",
            "message": f"Unknown gene source {value!r}.",
            "known_sources": list(SOURCE_KEYS),
        },
    )


def _unavailable(index) -> list[str]:
    return [k for k, s in index.status.items() if not s.available]


def _gene_limit(limit: int) -> int:
    if limit < 1 or limit > GENE_MAX_LIMIT:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_limit",
                "message": f"limit must be between 1 and {GENE_MAX_LIMIT}.",
            },
        )
    return limit


@app.get("/genes/sources", response_model=GeneSourcesResponse)
def gene_sources() -> GeneSourcesResponse:
    """What gene data this instance can actually answer from. No LLM.

    Four tables — CellAge curated (927 senescence genes after the ETL's Unclear
    filter, 949 rows in the source), CellAge expression signatures (1,259),
    GenAge human (307) and GenAge model organisms (2,205) — read from
    `data/`, or from the committed `.zip` archives when the unpacked tables are
    absent, because `data/**/*.csv` is gitignored.

    A source that cannot be read at all comes back `available: false` with a
    note. Every other gene route then reports it in `unavailable_sources`
    instead of returning a silently short answer.
    """
    index = gene_index()
    return GeneSourcesResponse(
        sources=[
            GeneSourceOut(
                **index.status[k].as_dict(),
                fields=GENE_FIELDS_BY_SOURCE.get(k, []),
            )
            for k in SOURCE_KEYS if k in index.status
        ],
        common_fields=GENE_COMMON_FIELDS,
        total_records=len(index.records),
        distinct_entrez=len(index.by_entrez),
        distinct_symbols=len(index.by_symbol),
        vocabularies={
            "effect": index.vocabulary("senescence_effect"),
            "senescence_type": index.vocabulary("senescence_type"),
            "cell_context": index.vocabulary("cell_context"),
            "organism": index.vocabulary("organism"),
            "lifespan_effect": index.vocabulary("lifespan_effect"),
            "longevity_influence": index.vocabulary("longevity_influence"),
            "expression_direction": index.vocabulary("expression_direction"),
        },
        note=(
            "These are CURATED DATABASE ANNOTATIONS, not inference. A CellAge "
            "effect label means a curator read a paper reporting that gene "
            "inducing or inhibiting senescence in a cell line. Nothing here is "
            "derived, ranked or weighted unless you ask for it with infer=true."
        ),
    )


@app.get("/genes/intersection", response_model=GeneIntersectionResponse)
def gene_intersection(
    a: str = "cellage_curated",
    b: str = "genage_human",
    key: Literal["entrez", "symbol"] = "entrez",
    limit: int = 100,
    offset: int = 0,
) -> GeneIntersectionResponse:
    """Genes held by two sources at once — the "CellAge ∩ GenAge" answer. No LLM.

    The evaluation could not express this question at all. The default pair is
    the one it asked for, and the answer is **113 genes** joined on entrez id.

    `key=entrez` is the default because it is the join that works. The symbol
    join finds 112 for the same pair — close enough to look interchangeable, and
    it is not: it loses one gene and would silently pick up aliases.

    Against `genage_models` the difference stops being cosmetic. That table is
    worm, yeast, fly and mouse, so entrez finds **1** shared gene (the id spaces
    are per-species) while symbols find 67 — and those 67 are ORTHOLOGUES with
    matching names (`ATM`, `AKT1`, `BRCA1`), not the same gene measured twice.
    A symbol join involving GenAge models therefore always comes back with a
    `warnings` entry saying so, and `genage_models_parser.py` destroys the real
    symbols in its MeTTa output anyway (`aak-2` is emitted as `aak_2`), which is
    the second reason this index is built from the CSVs.
    """
    limit = _gene_limit(limit)
    if offset < 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_offset", "message": "offset must be >= 0."},
        )
    for value in (a, b):
        if value not in SOURCE_KEYS:
            raise _gene_source_error(value)
    if a == b:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_filter",
                "message": "a and b must name two different sources.",
            },
        )

    index = gene_index()
    genes = index.intersect(a, b, key=key)
    table = index.sources_by_entrez if key == "entrez" else index.sources_by_symbol
    a_genes = sum(1 for srcs in table.values() if a in srcs)
    b_genes = sum(1 for srcs in table.values() if b in srcs)

    warnings: list[str] = []
    if key == "symbol" and NON_HUMAN_SOURCES & {a, b}:
        warnings.append(
            "A SYMBOL-KEYED JOIN AGAINST GenAge models IS AN ORTHOLOGY CLAIM, not "
            "an identity. That table is Caenorhabditis elegans, Saccharomyces "
            "cerevisiae, Drosophila melanogaster and Mus musculus; a shared "
            "symbol means the two organisms' genes were given the same name, not "
            "that the same gene appears twice. Joining the same pair on entrez "
            "gives 1 gene, not 67."
        )
    elif key == "entrez" and NON_HUMAN_SOURCES & {a, b}:
        warnings.append(
            "Entrez ids are species-specific, so a human source and GenAge "
            "models barely overlap by construction. A near-empty result here is "
            "correct, and is not evidence that the two sets are unrelated."
        )
    for source in (a, b):
        if not index.status.get(source) or not index.status[source].available:
            warnings.append(
                f"Source '{source}' could not be read, so this intersection is "
                f"empty for want of data, not for want of shared genes."
            )

    page = genes[offset:offset + limit]
    return GeneIntersectionResponse(
        a=a, b=b, key=key,
        genes=page,
        total=len(genes),
        returned=len(page),
        truncated=len(genes) > offset + len(page),
        a_genes=a_genes,
        b_genes=b_genes,
        warnings=warnings,
        note=(
            f"{len(genes)} genes are in both '{a}' and '{b}' by {key}. Each entry "
            f"lists every symbol the two sides use, so a disagreement is visible "
            f"rather than resolved silently."
        ),
    )


@app.get("/genes", response_model=GeneListResponse)
def gene_list(
    source: Optional[str] = None,
    effect: Optional[str] = None,
    senescence_type: Optional[str] = None,
    cell_context: Optional[str] = None,
    organism: Optional[str] = None,
    in_sources: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> GeneListResponse:
    """Filtered listing across CellAge and GenAge. No LLM, no hyperon.

    **"Which genes drive cellular senescence?"** is
    `?source=cellage_curated&effect=Induces` — 417 curated rows, each with the
    PMID it was curated from, the senescence type, the cell context and the
    `CellAgeRow_…` atom it becomes in the ETL. `effect=Inhibits` is the other
    510. These are CURATED EXPERIMENTAL ANNOTATIONS, not inferred: a curator
    read a paper reporting that gene inducing or inhibiting senescence in a cell
    line, and that is the entire claim. Nothing here is ranked or weighted.

    **"GenAge human genes"** is `?source=genage_human` — 307 rows with their
    uniprot id and the curators' selection basis (`human`, `mammal`,
    `functional`, …), the same vocabulary `epistemic_calibration.metta` maps to a
    confidence.

    `in_sources` is the cross-source filter, comma-separated: a record is kept
    when its gene appears in ALL the named sources, joined on entrez.
    `?source=cellage_curated&in_sources=cellage_curated,genage_human` is the 113
    shared genes as CellAge rows.

    Every listing is capped and carries `total` with an explicit `truncated`
    flag, so a page is never mistaken for the whole answer.
    """
    limit = _gene_limit(limit)
    if offset < 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_offset", "message": "offset must be >= 0."},
        )
    if source is not None and source not in SOURCE_KEYS:
        raise _gene_source_error(source)
    wanted = [s.strip() for s in in_sources.split(",")] if in_sources else None
    for value in (wanted or []):
        if value and value not in SOURCE_KEYS:
            raise _gene_source_error(value)

    index = gene_index()
    matched = index.select(
        source=source,
        effect=effect,
        senescence_type=senescence_type,
        cell_context=cell_context,
        organism=organism,
        in_sources=wanted,
    )
    page = matched[offset:offset + limit]

    note = None
    if not matched:
        note = (
            "No record matches these filters. Filter values are case-insensitive "
            "but must come from the vocabularies in GET /genes/sources — they are "
            "read off the data, so a value that is not there matches nothing by "
            "construction."
        )
    return GeneListResponse(
        records=[r.as_dict() for r in page],
        total=len(matched),
        returned=len(page),
        truncated=len(matched) > offset + len(page),
        limit=limit,
        offset=offset,
        filters={
            "source": source, "effect": effect,
            "senescence_type": senescence_type, "cell_context": cell_context,
            "organism": organism, "in_sources": wanted,
        },
        unavailable_sources=_unavailable(index),
        note=note,
    )


@app.get("/genes/{symbol_or_entrez}", response_model=GeneLookupResponse)
def gene_lookup(symbol_or_entrez: str, infer: bool = False) -> GeneLookupResponse:
    """Everything the KB knows about one gene, across all four sources. No LLM.

    The key is a symbol (`TP53`, case-insensitive) or an entrez id (`7157`). The
    response carries one record per (source, row) with its provenance — the PMID
    CellAge curated it from, the senescence type and cell context it was seen in,
    the organism and reported lifespan change for a GenAge models row, the
    selection basis for a GenAge human row — plus `sources`, which is this gene's
    membership across the four tables and therefore its own intersection answer.

    A gene appears once per ROW, not once per gene: CellAge annotates TP53 three
    times, one per senescence type, and collapsing those would pick a winner no
    source picked.

    `infer=true` additionally runs the ONE piece of real inference available
    here: it selects this gene's CellAge rows, injects them into a query-scoped
    hyperon space with `cellage_calibration.metta`, and lifts each into
    `(Effect <gene> CellularSenescence <sign> (stv s c))`. The strength is a
    curated prior identical for every row (CellAge reports a direction and no
    effect size) and the confidence is `(evidence-confidence InVitro)` — the
    returned `semantics` block says both, every time, so the numbers cannot
    travel without their provenance. The ETL's own `(Causes … (stv 0.82 0.70))`
    atoms are NOT that, and are not used.
    """
    index = gene_index()
    records, resolved_as = index.lookup(symbol_or_entrez)

    entrez = next((r.entrez for r in records if r.entrez is not None), None)
    symbols = sorted({r.symbol for r in records})
    sources = sorted({r.source for r in records})

    note = None
    if not records:
        note = (
            f"No record for '{symbol_or_entrez}' in any of "
            f"{', '.join(SOURCE_KEYS)}. These are four curated tables, not a "
            f"census of the genome, so this means the gene was not curated into "
            f"any of them — it is not a statement about the gene. Try an entrez "
            f"id if you passed an alias."
        )

    inference = None
    if infer:
        inference = _cellage_inference(symbols or [symbol_or_entrez], entrez, records)

    return GeneLookupResponse(
        query=symbol_or_entrez,
        resolved_as=resolved_as if records else None,
        entrez=entrez,
        symbols=symbols,
        gene_name=next((r.gene_name for r in records if r.gene_name), None),
        sources=sources,
        records=[r.as_dict() for r in records],
        inference=inference,
        unavailable_sources=_unavailable(index),
        note=note,
    )


def _cellage_inference(symbols: list[str], entrez: Optional[int], records) -> dict:
    """The `infer=true` block of GET /genes/{key}: lift this gene's CellAge rows.

    Kept apart from the route so the honest failure cases stay readable. There
    are three, and none of them is an exception:

    * no CellAge curated record for this gene -> nothing to lift;
    * `build/cellage_genes.metta` absent -> the slice comes off the committed
      25-row fixture, which is said so in `source`, or off nothing at all;
    * the hyperon runtime disabled -> reported, not faked.
    """
    from ontology.cellage_selector import MAX_ROWS, build_available, resolved_source

    keys: list[str] = list(symbols)
    if entrez is not None:
        keys.append(str(entrez))
    source = resolved_source()

    if not any(r.source == "cellage_curated" for r in records):
        return {
            "requested": True,
            "available": False,
            "effects": [],
            "rows_injected": 0,
            "source": source,
            "note": "No CellAge curated record names this gene, so there is no "
                    "row to lift. CellAge is the only source this layer lifts: "
                    "GenAge rows are lifespan phenotypes in model organisms, not "
                    "senescence effects, and the expression signatures are "
                    "correlations that must not become causal links.",
        }
    if source is None:
        return {
            "requested": True,
            "available": False,
            "effects": [],
            "rows_injected": 0,
            "source": None,
            "note": "build/cellage_genes.metta has not been generated (run "
                    "scripts/run_etl.sh) and no committed fixture is present, so "
                    "there are no row blocks to inject.",
        }

    result, rows, effects = run_offloaded(
        "cellage_effects",
        {"genes": keys, "limit": MAX_ROWS},
        lambda: run_cellage_effects(keys, limit=MAX_ROWS),
    )
    if result.status == "error":
        _raise_pln_failure(result, stage="cellage_inference",
                           extra={"genes": keys, "rows": len(rows)})

    note = None
    if not build_available():
        note = (
            f"build/cellage_genes.metta is absent, so the rows came from the "
            f"committed sample at {source} — 25 rows of a 927-row table. Run "
            f"scripts/run_etl.sh for the full set."
        )
    elif not effects:
        note = ("CellAge holds rows for this gene but none carries a stated "
                "direction, so no Effect link could be lifted. An `Unclear` "
                "curation asserts nothing, and neither does the engine.")
    return {
        "requested": True,
        "available": True,
        "effects": [e.as_dict() for e in effects],
        "rows_injected": len(rows),
        "rows_cap": MAX_ROWS,
        "source": source,
        "mode": result.mode,
        "query_time_ms": result.query_time_ms,
        "stack": [p.name for p in CELLAGE_STACK],
        "semantics": CELLAGE_EFFECT_SEMANTICS,
        "note": note,
    }


@app.get("/kb/schema", response_model=KbSchemaResponse)
def kb_schema() -> KbSchemaResponse:
    """What the knowledge base actually holds: predicates, arities, fact counts.

    The answer to "list your data sources and counts", and the reference a
    caller needs before trusting an empty result. The ontology DECLARES a larger
    vocabulary than it populates — `Predicts`, `Extends`, `HazardRatio`, the
    gene predicates and the DrugAge row predicates are all declared with zero
    facts in the generic runtime — so a query over one of them is well-formed
    and returns nothing. (`TargetsHallmark` and `Causes` were on that list until
    `hallmark_targeting.metta` populated them; the lists below are computed from
    the ground atoms every time, so they never go stale the way this sentence
    would.) Those are listed separately here
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
            "model only through the baseline table. THAT HOLDS FOR A z YOU "
            "SEND. A z the server DERIVES from a raw `value` is standardised "
            "against a single pooled mean and sd — there is no age/sex-"
            "stratified reference table in this repository — so it is NOT "
            "adjusted, and for a marker that drifts with age the difference is "
            "not small. Every derived marker comes back with `derived: true`, "
            "the formula that produced it, and a warning on the patient."
        ),
        elevated_threshold=elevated,
        raw_value_note=(
            "A `value` is standardised server-side with the reference shown here "
            "and reported back with `derived: true` and the exact formula. Those "
            "references are CURATED PRIORS, not calibrated cohort statistics — "
            "this repository contains no reference table. Send `z` when you have "
            "a properly standardised measurement."
        ),
        linage2=_linage2_block_description(),
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
        has_linage2=built.has_linage2,
        linage2=built.linage2.as_dict() if built.linage2 is not None else None,
    )


class PatientTextIn(BaseModel):
    """A few lines of plain text about a person (the "My Patient" tab's input)."""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ..., max_length=20_000,
        description="Age, sex, smoking, lab values with units, diagnoses — one per line "
                    "(e.g. '58 year old male, current smoker', 'albumin 4.1 g/dL', "
                    "'diagnoses: hypertension'). Read by fixed rules; see `reader`.",
    )
    id: str = Field(default="Me", max_length=48, pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
                    description="Letters, digits and underscores; becomes `Caller_<id>`.")
    reader: Literal["rules", "model"] = Field(
        default="rules",
        description="'rules' (default): fixed rules only, no LLM. 'model': the rules, then a "
                    "language model (PLN_EXTRACT_MODEL) rewrites the statements the rules did not "
                    "understand into the rules' own wording, which the rules then read — it never "
                    "supplies a value (numbers and units are copied from the text), never replaces "
                    "a refusal (that becomes a `suggestion`), and never sets a smoking status on its "
                    "own. Needs the server's OPENAI_API_KEY (503 if not configured) and text of at "
                    "most PLN_EXTRACT_MAX_CHARS characters (413). The text is sent to OpenAI.")


class PatientTextResponse(BaseModel):
    ok: bool = Field(description="Everything read is usable and LinAge2 could be scored.")
    read: dict = Field(description="What was read: every value as typed, in LinAge2's unit, "
                                   "with a status (ok / unit assumed / needs a unit / unknown "
                                   "unit / out of range / duplicate), plus the questionnaire "
                                   "answers and whatever was not understood.")
    problems: list[str] = Field(description="Why `ok` is false — fix the text and resend.")
    reader_used: str = Field(default="rules", description="'rules', or 'rules+model' when the model "
                                                         "read the text.")
    read_as_text: Optional[str] = Field(
        default=None, description="What the rules read: the text with the model's rewrites in. Equal "
                                  "to `text` for reader='rules'. Sending it with reader='rules' gives "
                                  "the same patient, without the model.")
    model_error: Optional[dict] = Field(
        default=None, description="reader='model' only: why the text was read by the rules alone "
                                  "({code, message, configuration}); null when the model read it.")
    suggestions: list[dict] = Field(
        default_factory=list,
        description="Wordings the model offers for statements the rules refused, or that the two "
                    "read differently: {line, start, end, original, wordings, reason, blocking}. "
                    "Replace `original` (at line/start/end of `text`) with a wording and resend.")
    statements: list[dict] = Field(
        default_factory=list,
        description="Every statement as read: {index, line, start, end, text, outcome, source, "
                    "typed}; source 'model' marks a rewrite, `typed` what the person wrote there.")
    patient: Optional[dict] = Field(
        default=None,
        description="The patient, LinAge2 scored in-process: send it as `patient` to /query "
                    "or /metta/run, or as the whole body of /patients/preview and "
                    "/linage2/analyze. Null when `ok` is false.")
    preview: Optional[PatientPreviewResponse] = None


@app.post("/patients/from-text", response_model=PatientTextResponse)
def patients_from_text(req: PatientTextIn) -> PatientTextResponse:
    """Plain text -> a patient, as the UI's "My Patient" tab does it.

    reader='rules' (the default) uses fixed rules only — no LLM. reader='model' adds
    the tab's model reader (core/patient_read.py): it may rewrite statements the
    rules did not understand, never a value, never a refusal. A value whose unit
    cannot be pinned down is not guessed: `ok` is false and `read` says which value
    and why (core/patient_text.py). Otherwise LinAge2 is scored in-process
    (core/linage2_model.py) and the result is an ordinary caller-supplied patient,
    previewed exactly as POST /patients/preview would.
    """
    extractor = None
    if req.reader == "model":
        extractor = OpenAIExtractor()
        if extractor.error is not None:
            raise HTTPException(status_code=503, detail={
                "code": "model_reader_not_configured", "message": extractor.error.message})
        from config import PLN_EXTRACT_MAX_CHARS
        if len(req.text) > PLN_EXTRACT_MAX_CHARS:
            raise HTTPException(status_code=413, detail={
                "code": "text_too_long_for_model_reader",
                "message": f"reader='model' takes at most {PLN_EXTRACT_MAX_CHARS} characters; "
                           f"send reader='rules' or shorten the text"})
    result = read_patient(req.text, extractor)
    parsed = result.parsed
    extra = dict(reader_used=result.reader, read_as_text=result.read_as,
                 model_error=None if result.model_error is None else
                 {"code": result.model_error.code, "message": result.model_error.message,
                  "configuration": result.model_error.is_config},
                 suggestions=[s.as_dict() for s in result.suggestions],
                 statements=result.as_dict()["statements"])
    if not parsed.ok:
        return PatientTextResponse(ok=False, read=parsed.as_dict(), problems=parsed.all_problems(), **extra)
    try:
        payload, _ = parsed.to_patient(req.id)
    except PatientSpecError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message,
                                                      **exc.extra})
    preview = patients_preview(PatientIn.model_validate(payload))
    return PatientTextResponse(ok=True, read=parsed.as_dict(), problems=[], patient=payload,
                               preview=preview, **extra)


# ── LinAge2: the clinical clock ─────────────────────────────────────────────
# A caller's LinAge2 /predict response arrives under `patient.linage2` and is
# rendered to request-scoped atoms by core.linage2_builder. The forms that read
# them live in pln_linage2.metta and run in core.pln_runner.LINAGE2_PATIENT_STACK,
# a query-scoped space — the shared space cannot take the layer's head symbols
# (linage2_core.metta header). core/linage2_router.py recognises the forms.

_LINAGE2_BASELINE_NOTE_PRESENT = (
    "A generated NHANES all-cause-mortality baseline (build/nhanes_mortality_baseline"
    ".metta) is loaded, so `risk` is an absolute ten-year risk: 1 - (1 - baseline)^"
    "(HR^delta), with the baseline's own age band, sex and horizon."
)
_LINAGE2_BASELINE_NOTE_ABSENT = (
    "No absolute risk: the knowledge base holds no all-cause-mortality baseline. It "
    "deliberately ships no curated one (docs/nhanes_integration.md §5), and the "
    "survey-weighted NHANES baseline is an ETL output — run scripts/run_etl.sh with "
    "the NHANES mortality linkage present and `risk` becomes available. `hazard` is "
    "the relative hazard versus a same-age, same-sex person with a delta of 0 and "
    "needs no baseline."
)


def _linage2_block_description() -> dict:
    return {
        "field": "patient.linage2",
        "what": "the LinAge2 service's POST /predict response body, forwarded as "
                "received (or its flattened metadata)",
        "becomes": "the LinAgeAccel clock marker (z = delta / linage-sd-to-years) plus "
                   "one (LinAgeContribution <Patient> <Input> <years> Measured|Imputed) "
                   "atom per model input — this request only, nothing stored",
        "unlocks": [f"({form} &self <Patient> …)" for form in LINAGE2_FORMS[:7]],
        "features": "GET /linage2/features",
        "no_llm": "POST /linage2/analyze",
        "send_alongside": "the patient's own CRP / HbA1c / FastingGlucose (z or value) "
                          "and smoking status — a LinAge2 contribution is credited to "
                          "a cause only when the patient's own value witnesses the "
                          "direction; a contribution's sign is not a lab's direction",
        "excludes": "markers.LinAgeAccel (one clock, one z)",
    }


@lru_cache(maxsize=1)
def _linage2_context():
    """Registry + inventory over the LinAge2 scoped stack (static files; cached)."""
    paths = linage2_patient_kb()
    registry, _ = load_specific_files(paths)
    return registry, _inventory_for_paths(paths)


def _validate_linage2_query(metta_query: str, injected: Optional[str]) -> ValidationResult:
    """Validate a LinAge2 form against the space it will actually run in."""
    base_registry, base_inventory = _linage2_context()
    registry, inventory = with_injected(base_registry, base_inventory, injected)
    return validate(validation_text(injected, metta_query), registry, inventory)


def _offloaded_run(
    metta_query: str, confidence_threshold: float, kb_files: list[Path],
    extra_atoms: Optional[str],
) -> PLNRunResult:
    return run_offloaded(
        "run_query",
        {"metta_query": metta_query,
         "confidence_threshold": confidence_threshold,
         "kb_files": kb_files,
         "extra_atoms": extra_atoms},
        lambda: run_query(
            metta_query=metta_query,
            confidence_threshold=confidence_threshold,
            kb_files=kb_files,
            extra_atoms=extra_atoms,
        ),
    )


def _generic_kb(metta_query: str) -> list[Path]:
    """The shared runtime stack — or, for a program that names a patient, the
    patient stack (core.pln_runner.patient_stack): the full space aborts on the
    patient forms at its head-symbol edge, built-in patients included."""
    runtime = _runtime_kb_paths()
    return patient_stack(runtime) if names_a_patient(metta_query) else runtime


def _validate_generic_with(
    metta_query: str, registry: OntologyRegistry, injected: Optional[str]
) -> ValidationResult:
    """Validate against the shared space as it will actually run: its KB PLUS the
    atoms injected into it. Without them a caller's own patient id (Caller_W58) is
    "not found in loaded ontology", and every answer about that patient carried a
    validation issue on /query."""
    merged, inventory = with_injected(registry, _runtime_inventory(), injected)
    return validate(validation_text(injected, metta_query), merged, inventory)


_SPACE_LABELS = ("LinAge2 space", "shared space")


def _run_linage2_program(
    metta_query: str,
    split: SplitProgram,
    *,
    linage_atoms: Optional[str],
    shared_atoms: Optional[str],
    confidence_threshold: float,
    validate_generic: Callable[[str], ValidationResult],
    strict: bool,
) -> tuple[str, ValidationResult, PLNRunResult]:
    """Run a program that calls at least one LinAge2 form.

    A pure LinAge2 program runs in the LinAge2 scoped space with every patient
    atom. A MIXED one ("my LinAge2 drivers and my supplement plan") is split per
    top-level expression (`split_linage2_program`): the LinAge2 forms run there, the
    rest in the shared space with `shared_atoms` — the patient minus its LinAge2
    atoms, which that space cannot hold (BuiltPatient.shared_atoms). Both halves
    run in ONE offloaded task (one worker, one deadline, one admission) and the
    answers come back in program order. `strict` refuses an invalid program with a
    422 before anything runs (/metta/run); otherwise the verdict travels with the
    answer (/query).
    """
    if not split.mixed:
        validation = _validate_linage2_query(metta_query, linage_atoms)
    else:
        validation = merge_validation_results(
            _validate_linage2_query(split.linage2, linage_atoms),
            validate_generic(split.generic),
            labels=_SPACE_LABELS,
        )
    validation.warnings.extend(nesting_warnings(split))
    if strict and not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_metta_query",
                "message": (
                    "MeTTa validation failed against the LinAge2 scoped space; the "
                    "query was not executed." if not split.mixed else
                    "MeTTa validation failed (the program mixes LinAge2 forms, validated "
                    "against the LinAge2 scoped space, with forms validated against the "
                    "shared one — each issue names its space); the query was not executed."
                ),
                "issues": validation.issues,
            },
        )
    if not split.mixed:
        result = _offloaded_run(
            metta_query, confidence_threshold, linage2_patient_kb(), linage_atoms
        )
        return "linage2", validation, result

    parts = [
        {"metta_query": split.linage2, "kb_files": linage2_patient_kb(),
         "extra_atoms": linage_atoms},
        {"metta_query": split.generic, "kb_files": _generic_kb(split.generic),
         "extra_atoms": shared_atoms},
    ]
    results = run_offloaded(
        "run_query_parts",
        {"parts": parts, "confidence_threshold": confidence_threshold},
        lambda: run_query_parts(parts, confidence_threshold=confidence_threshold),
    )
    return "linage2+generic", validation, merge_run_results(results, split.positions())


@app.get("/linage2/features", response_model=LinAge2FeaturesResponse)
def linage2_features() -> LinAge2FeaturesResponse:
    """What a LinAge2 result may contain, and what the knowledge base does with it.

    Read this before sending `patient.linage2`. Every NHANES code the LinAge2
    service emits is declared in linage2_core.metta with its KB symbol; four of
    the 59 inputs read out a biomarker the causal graph reaches (CRP, HbA1c,
    glucose, cotinine) and can therefore be credited to a cause — the rest are
    carried as years and left unexplained, never guessed at.
    """
    return LinAge2FeaturesResponse(
        features=linage2_feature_listing(),
        clock={
            "symbol": "LinAge2",
            "acceleration": "LinAgeAccel (BA - CA, years; AccelDefinition Difference)",
            "outcome": "AllCauseMortality",
            "hazard_per_year": "read off Fong2025_LinAgeAccel_AllCauseMortality "
                               "(linage2_fong2025_evidence.metta): 1.093, derived from "
                               "the reported null-model mortality rate doubling time of "
                               "~7.8 years",
            "publication": "Fong et al. 2025, npj Aging 11:29, PMID 40268972",
        },
        forms=list(LINAGE2_FORMS[:7]),
        stack=[p.name for p in linage2_patient_kb()],
        baseline_available=LINAGE2_GENERATED_BASELINE.exists(),
    )


@app.post("/linage2/analyze", response_model=LinAge2AnalyzeResponse)
def linage2_analyze(req: LinAge2AnalyzeRequest) -> LinAge2AnalyzeResponse:
    """The whole LinAge2 analysis for one caller-supplied patient, no LLM.

    Takes the patient at the TOP LEVEL like /patients/preview, with the `linage2`
    block required. Runs the decomposition, the hazard, the absolute risk (when a
    baseline is loaded) and one counterfactual per lever in the LinAge2 scoped
    space, in one MeTTa program, and returns them as JSON rather than atoms.
    """
    payload = req.model_dump(exclude={"levers"})
    if payload.get("linage2") is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "linage2_required",
                    "message": "POST /linage2/analyze needs a `linage2` block (the LinAge2 "
                               "/predict response). For markers alone use /patients/preview."},
        )
    built = _build_caller_patient(payload)
    assert built is not None and built.linage2 is not None

    levers = tuple(req.levers) if req.levers else LINAGE2_DEFAULT_LEVERS
    _, inventory = _linage2_context()
    unknown = [lv for lv in levers if not inventory.knows_symbol(lv)]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"code": "unknown_lever",
                    "message": f"Lever(s) {', '.join(unknown)} are not symbols the LinAge2 "
                               f"space holds. A lever is a cause (ChronicInflammation, "
                               f"CellularSenescence, InsulinResistance), an intervention "
                               f"(Metformin, DasatinibPlusQuercetin, SmokingCessation) or a "
                               f"marker (CRP).",
                    "levers": unknown},
        )
    try:
        program = linage2_analysis_program(
            built.patient_id, levers, with_projections=LINAGE2_GENERATED_BASELINE.exists()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_lever", "message": str(exc)})

    kb_files = linage2_patient_kb()
    atoms = built.atoms
    pln_result = run_offloaded(
        "run_query",
        {"metta_query": program, "confidence_threshold": 0.0,
         "kb_files": kb_files, "extra_atoms": atoms},
        lambda: run_query(metta_query=program, confidence_threshold=0.0,
                          kb_files=kb_files, extra_atoms=atoms),
    )
    if pln_result.status == "error":
        _raise_pln_failure(pln_result, stage="pln_execution", extra={"metta_query": program})
    analysis = linage2_collect_analysis(program, pln_result)

    warnings = list(built.warnings)
    warnings.extend(lever_warnings(program, extra_atoms=atoms, known_patients={built.patient_id}))
    return LinAge2AnalyzeResponse(
        patient_id=built.patient_id,
        atoms=atoms,
        linage2=built.linage2.as_dict(),
        decomposition=analysis.decomposition,
        hazard=analysis.hazard,
        risk=analysis.risk,
        risk_note=(_LINAGE2_BASELINE_NOTE_PRESENT if LINAGE2_GENERATED_BASELINE.exists()
                   else _LINAGE2_BASELINE_NOTE_ABSENT),
        counterfactuals=analysis.counterfactuals,
        projected_risks=analysis.projected_risks,
        warnings=warnings,
        metta_query=program,
        pln_status=pln_result.status,
        pln_query_time_ms=pln_result.query_time_ms,
        unparsed=analysis.unparsed,
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
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_message",
                    "message": "message must not be empty."},
        )

    selected = req.ontology_files
    if selected is None:
        selected = _default_selection(list(_discover_metta_files().keys()))
    else:
        _validate_ontology_files(selected)
    registry, raw_contents = _build_context(selected)
    system_prompt = build_system_prompt(registry, raw_contents, _runtime_inventory())

    patient = _build_caller_patient(req.patient.model_dump() if req.patient else None)
    if patient is not None:
        # Shared with the Gradio chat (core.patient_context): same text, same place.
        system_prompt += patient_prompt_section(patient)

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
        drugage_compounds = _guard_drugage_pool(drugage_compounds)
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
    elif (split := split_linage2_program(translation.metta_query)).linage2:
        # A LinAge2 form. Its layer cannot live in the shared space (head-symbol
        # budget — linage2_core.metta header), so it is validated against and run
        # in the LinAge2 scoped stack, with the caller's atoms; anything else the
        # program asks for runs in the shared space beside it. Same worker pool,
        # same deadline, same admission limit as everything else.
        routed, validation, pln_result = _run_linage2_program(
            translation.metta_query,
            split,
            linage_atoms=patient.atoms if patient is not None else None,
            shared_atoms=patient.shared_atoms if patient is not None else None,
            confidence_threshold=req.confidence_threshold,
            validate_generic=lambda q: _validate_generic_with(
                q, registry, patient.shared_atoms if patient is not None else None),
            strict=False,
        )
    else:
        # The shared space gets the patient WITHOUT its LinAge2 atoms: with them
        # it aborts on the patient forms (BuiltPatient.shared_atoms).
        shared = patient.shared_atoms if patient is not None else None
        validation = _validate_generic_with(translation.metta_query, registry, shared)
        pln_result = _offloaded_run(
            translation.metta_query, req.confidence_threshold,
            _generic_kb(translation.metta_query), shared,
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
    # A lever the named patient cannot pull. The engine cannot check this
    # (pln_counterfactual.metta §3b: the KB is at hyperon's ceiling), so the
    # number comes back unqualified and the qualification is attached here.
    warnings.extend(lever_warnings(
        translation.metta_query,
        extra_atoms=patient.atoms if patient is not None else None,
        known_patients=known_ids,
    ))
    # A rule whose data lives in a scoped space. Without this the response is
    # a well-formed, validated, structurally empty answer that reads as "no".
    warnings.extend(scoped_form_warnings(
        translation.metta_query, _runtime_kb_paths(), _runtime_inventory()
    ))
    # A LinAge2 form for a patient with no LinAge2 result: valid, empty, and
    # misleading unless it says so.
    warnings.extend(linage2_form_warnings(
        translation.metta_query, patient.atoms if patient is not None else None
    ))

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
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_metta_query",
                    "message": "metta_query must not be empty."},
        )

    _guard_extra_atoms(req.extra_atoms, allow_definitions=req.allow_definitions)
    patient = _build_caller_patient(req.patient.model_dump() if req.patient else None)
    injected = "\n".join(
        part for part in (patient.atoms if patient else None, req.extra_atoms) if part
    ) or None
    # What the SHARED space gets: the patient without its LinAge2 atoms, which
    # abort that space on the patient forms (BuiltPatient.shared_atoms).
    injected_shared = "\n".join(
        part for part in (patient.shared_atoms if patient else None, req.extra_atoms)
        if part
    ) or None

    def validate_generic(metta_query: str) -> ValidationResult:
        if req.ontology_files is not None:
            _validate_ontology_files(req.ontology_files)
        registry = (
            _runtime_registry()
            if req.ontology_files is None
            else _build_context(req.ontology_files)[0]
        )
        return _validate_generic_with(metta_query, registry, injected_shared)

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
        drugage_compounds = _guard_drugage_pool(drugage_compounds)
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
    elif (split := split_linage2_program(req.metta_query)).linage2:
        # A LinAge2 form runs in its own scoped space (see /query for why); the
        # rest of a mixed program runs in the shared space beside it.
        routed, validation, pln_result = _run_linage2_program(
            req.metta_query,
            split,
            linage_atoms=injected,
            shared_atoms=injected_shared,
            confidence_threshold=req.confidence_threshold,
            validate_generic=validate_generic,
            strict=True,
        )
    else:
        validation = validate_generic(req.metta_query)
        if not validation.valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_metta_query",
                    "message": "MeTTa validation failed; the query was not executed.",
                    "issues": validation.issues,
                },
            )
        pln_result = _offloaded_run(
            req.metta_query, req.confidence_threshold, _generic_kb(req.metta_query),
            injected_shared,
        )

    if pln_result.status == "error":
        _raise_pln_failure(
            pln_result, stage="pln_execution", extra={"metta_query": req.metta_query}
        )

    known_ids = _known_patient_ids() | (
        {patient.patient_id} if patient is not None else set()
    )
    patient_warning = _unknown_patient_warning(req.metta_query, known_ids)
    run_warnings = (
        ([patient_warning] if patient_warning else [])
        + (patient.warnings if patient is not None else [])
        + lever_warnings(
            req.metta_query,
            extra_atoms=injected,
            known_patients=known_ids,
        )
        + scoped_form_warnings(
            req.metta_query, _runtime_kb_paths(), _runtime_inventory()
        )
        + linage2_form_warnings(req.metta_query, injected)
    )

    return MettaRunResponse(
        metta_query=req.metta_query,
        patient_id=patient.patient_id if patient is not None else None,
        warnings=run_warnings,
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
    if req.strategy == "metta_sort" and len(req.compounds) > PLN_MAX_METTA_SORT_COMPOUNDS:
        # The reference implementation's insertion sort is O(n^2) in MeTTa:
        # 6.1 s at n=10, 52.6 s at n=20, past the 60 s deadline by n=40. The
        # published 60-compound cap is the `linear` cap, so every request
        # between 40 and 60 on this strategy was advertised as legal and
        # answered with a 504.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "too_many_compounds_for_strategy",
                "message": (
                    f"strategy='metta_sort' ranks at most "
                    f"{PLN_MAX_METTA_SORT_COMPOUNDS} compounds: its MeTTa "
                    f"insertion sort is O(n^2) and exceeds the query deadline "
                    f"well inside the {PLN_MAX_RANK_COMPOUNDS}-compound cap that "
                    f"applies to strategy='linear'. Use 'linear' (the default) "
                    f"for a pool this size — the two agree bit for bit."
                ),
                "requested": len(req.compounds),
                "limit": PLN_MAX_METTA_SORT_COMPOUNDS,
                "strategy": req.strategy,
            },
        )

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


def _entry_out(entry) -> ExtractedEntryOut:
    """One accepted entry, with the provenance the evaluation found missing."""
    return ExtractedEntryOut(
        kind=entry.kind,
        name=entry.name,
        metta=entry.metta,
        description=entry.description,
        identifier=entry.identifier,
        evidence_tier=entry.evidence_tier,
        effect_size_pct=entry.effect_size_pct,
        provisional=entry.provisional,
        provisional_fields=entry.provisional_fields,
        notes=entry.notes,
    )


@app.post("/ontology/expand", response_model=ExpandResponse)
def ontology_expand(req: ExpandRequest) -> ExpandResponse:
    """Extract new PLN ontology entries from pasted paper text.

    Equivalent to the "Ontology Expander" tab's Extract step (and, if
    `apply=true`, the Apply step too).

    Every entry passes a schema gate before it can reach `metta_block`: it must
    use predicates the rules actually read, must not carry a two-float truth
    value or mint a confidence constant, must name an EvidenceCategory the
    calibration table scores, and must carry a PMID or a DOI. Refusals come back
    in `rejected_entries` with their reasons — they are never dropped quietly.
    """
    if not req.paper_text.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_paper_text",
                    "message": "paper_text must not be empty."},
        )

    target_path = _resolve_target_path(req.target_file, req.new_filename)

    # Extraction never writes. The write is done HERE, after the same gate
    # /ontology/apply passes, so both doors refuse the same things — and so an
    # `apply=true` request that is going to be refused is refused after the
    # caller can see what was extracted, not silently applied.
    result = run_expansion_pipeline(
        paper_data=req.paper_text.encode("utf-8"),
        filename=req.filename,
        target_file_path=target_path,
        model=req.model,
        temperature=req.temperature,
        apply=False,
    )

    if not result.ok:
        raise HTTPException(
            status_code=400,
            detail={"code": "extraction_failed", "message": result.error},
        )

    if req.apply and result.metta_block.strip():
        # schema_checked: every entry in this block already passed check_entry
        # inside the pipeline; what is re-checked is the target and the size.
        _guard_ontology_write(target_path, result.metta_block, schema_checked=True)
        write_error = _append_metta_block(target_path, result.metta_block)
        result.applied = write_error is None
        if write_error is not None:
            result.error = f"Failed to write to {target_path.name}: {write_error}"

    return ExpandResponse(
        paper_title=result.paper_title,
        paper_summary=result.paper_summary,
        target_file=result.target_file,
        new_entries=[_entry_out(e) for e in result.new_entries],
        duplicate_entries=[_entry_out(e) for e in result.duplicate_entries],
        rejected_entries=[
            RejectedEntryOut(
                kind=r.kind, name=r.name, metta=r.metta,
                description=r.description, codes=r.codes, reasons=r.reasons,
            )
            for r in result.rejected_entries
        ],
        metta_block=result.metta_block,
        unconsumed_predicates=result.unconsumed_predicates,
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
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_metta_block",
                    "message": "metta_block must not be empty."},
        )

    metta_files = _discover_metta_files()
    name = _normalise_metta_name(req.target_file, field="target_file")
    target_path = metta_files.get(name) or (CUSTOM_ONTOLOGY_DIR / name)
    target_path = _ensure_allowed_target(target_path)
    # The block arrived as text. Nothing so far has required it to be the one
    # /ontology/expand produced, so it faces that endpoint's schema gate here.
    _guard_ontology_write(target_path, req.metta_block, schema_checked=False)

    error = _append_metta_block(target_path, req.metta_block)
    if error is not None:
        return ApplyResponse(applied=False, target_file=target_path.name, error=error)

    return ApplyResponse(applied=True, target_file=target_path.name)


if __name__ == "__main__":
    import uvicorn

    from config import PLN_API_HOST, PLN_API_PORT

    uvicorn.run(app, host=PLN_API_HOST, port=PLN_API_PORT)
