"""PLN Natural Language Query Interface — plain HTTP/JSON API.

This is a REST sibling of the Gradio UI in app.py, for scripts and agents
that want to call the query pipeline directly instead of driving a browser.
It runs the exact same core pipeline as the chat handler in app.py
(translate -> validate -> run_query -> format_bot_response) and exposes it
as JSON endpoints with auto-generated OpenAPI docs.

Run standalone:
    python api.py
    # -> http://0.0.0.0:8000  (interactive docs at /docs, schema at /openapi.json)

Or with uvicorn directly (e.g. for --reload during development):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

This process is independent of app.py — run it alongside the Gradio UI
(different port) or on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the pln_chat package root is on sys.path so submodule imports work
# whether the file is run directly or via `python -m`, matching app.py.
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    AVAILABLE_MODELS,
    CUSTOM_ONTOLOGY_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ONTOLOGY_DIR,
    OPENAI_API_KEY,
    PLN_API_KEY,
    PLN_RUNTIME_AVAILABLE,
)
from ontology.loader import load_specific_files
from ontology.registry import BUILTIN_REGISTRY, OntologyRegistry
from ontology.expander import run_expansion_pipeline
from core.context_builder import build_system_prompt
from core.llm_translator import translate
from core.metta_validator import validate
from core.pln_runner import run_query
from utils.formatting import format_bot_response
from utils.logging import log_turn


# ── Ontology file discovery / resolution (mirrors app.py) ──────────────────

def _discover_metta_files() -> dict[str, Path]:
    """Return {filename: absolute_path} for every .metta file in the project."""
    files: dict[str, Path] = {}
    for base in (ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR):
        if base.exists():
            for p in sorted(base.glob("*.metta")):
                files[p.name] = p
    return files


def _default_selection(choices: list[str]) -> list[str]:
    return [k for k in choices if "epistemic" in k.lower()] or choices[:1]


def _build_context(selected_files: list[str]) -> tuple[OntologyRegistry, dict[str, str]]:
    """Load selected .metta files for the LLM system-prompt context.

    Mirrors app.py: the selection only controls what the LLM sees. Execution
    against the KB always uses every discovered .metta file (see /query).
    """
    metta_files = _discover_metta_files()
    paths = [metta_files[f] for f in selected_files if f in metta_files]
    if paths:
        return load_specific_files(paths)
    return BUILTIN_REGISTRY, {}


def _resolve_target_path(target_file: Optional[str], new_filename: Optional[str]) -> Path:
    """Return an absolute Path for a .metta file to append extracted entries to.

    If `target_file` names an existing .metta file, append to it in place.
    Otherwise treat it (or `new_filename`) as the stem of a new file created
    under CUSTOM_ONTOLOGY_DIR — same behaviour as the "create new file…"
    option in the Gradio Ontology Expander tab.
    """
    metta_files = _discover_metta_files()
    if target_file and target_file in metta_files:
        return metta_files[target_file]

    stem = (target_file or new_filename or "expanded_ontology").strip()
    if stem.endswith(".metta"):
        stem = stem[: -len(".metta")]
    stem = stem or "expanded_ontology"
    CUSTOM_ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    return CUSTOM_ONTOLOGY_DIR / f"{stem}.metta"


# ── Optional auth ────────────────────────────────────────────────────────────

def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """No-op unless PLN_API_KEY is set; then require a matching X-API-Key header.

    Every call here costs an OpenAI request (and /ontology/* can write to
    disk), so set PLN_API_KEY before exposing this beyond localhost.
    """
    if PLN_API_KEY and x_api_key != PLN_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PLN Longevity Query API",
    description=(
        "Programmatic JSON API over the PLN natural-language query pipeline — "
        "the same logic behind the Gradio chat UI (app.py), for scripts and agents. "
        "See /docs for interactive testing."
    ),
    version="1.0.0",
)

# Permissive by default so a local agent/script can call this without CORS
# friction during experimentation. Tighten allow_origins if this is ever
# exposed beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    message: str = Field(..., description="Natural-language question for the KB.")
    history: list[HistoryTurn] = Field(
        default_factory=list,
        description="Prior turns (oldest first) for multi-turn context. "
                    "Pass back the `history` from the previous response to continue a conversation.",
    )
    ontology_files: Optional[list[str]] = Field(
        default=None,
        description="Which .metta files to inject into the LLM's system-prompt context "
                    "(see GET /ontology/files for choices). Defaults to the same "
                    "selection the UI starts with. PLN execution always runs against "
                    "every .metta file regardless of this selection.",
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
    usage: Optional[dict] = None
    error: Optional[str] = Field(default=None, description="Set when the LLM translation step failed.")
    history: list[HistoryTurn] = Field(description="Updated history — pass back verbatim for the next turn.")


class OntologyFilesResponse(BaseModel):
    files: list[str]
    default_selection: list[str]


class ExpandRequest(BaseModel):
    paper_text: str = Field(..., description="Paper text or abstract to extract new ontology entries from.")
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
    return {
        "status": "ok",
        "pln_mode": "runtime" if PLN_RUNTIME_AVAILABLE else "stub",
        "openai_key_configured": bool(OPENAI_API_KEY),
        "api_key_required": bool(PLN_API_KEY),
        "available_models": AVAILABLE_MODELS,
    }


@app.get("/ontology/files", response_model=OntologyFilesResponse)
def ontology_files() -> OntologyFilesResponse:
    """List discovered .metta files, for populating `ontology_files` / `target_file`."""
    choices = list(_discover_metta_files().keys())
    return OntologyFilesResponse(files=choices, default_selection=_default_selection(choices))


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    """Ask a natural-language question of the PLN knowledge base.

    Equivalent to typing into the "PLN Query" tab and clicking Send — runs
    translate -> validate -> run_query -> format_bot_response and returns
    every intermediate result as structured JSON (not just the rendered text).
    """
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty.")

    all_kb_paths = list(_discover_metta_files().values())
    selected = req.ontology_files
    if selected is None:
        selected = _default_selection(list(_discover_metta_files().keys()))
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

    validation = validate(translation.metta_query, registry)

    pln_result = run_query(
        metta_query=translation.metta_query,
        confidence_threshold=req.confidence_threshold,
        kb_files=all_kb_paths,
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
        usage=translation.usage,
        error=translation.error,
        history=[HistoryTurn(**m) for m in updated_history],
    )


@app.post("/ontology/expand", response_model=ExpandResponse, dependencies=[Depends(_require_api_key)])
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


@app.post("/ontology/apply", response_model=ApplyResponse, dependencies=[Depends(_require_api_key)])
def ontology_apply(req: ApplyRequest) -> ApplyResponse:
    """Write a previously-previewed MeTTa block to disk.

    Equivalent to the "Apply to Ontology" button — pairs with a prior
    POST /ontology/expand call made with apply=false.
    """
    if not req.metta_block.strip():
        raise HTTPException(status_code=422, detail="metta_block must not be empty.")

    metta_files = _discover_metta_files()
    name = req.target_file if req.target_file.endswith(".metta") else f"{req.target_file}.metta"
    target_path = metta_files.get(req.target_file) or (CUSTOM_ONTOLOGY_DIR / name)

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
