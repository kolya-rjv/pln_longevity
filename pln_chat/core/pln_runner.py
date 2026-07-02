"""Execute MeTTa queries against the PLN knowledge base.

Two modes:
  stub    — returns pattern-matched mock results; no runtime dependency.
  runtime — delegates to the Hyperon MeTTa interpreter (requires `hyperon`
            package and knowledge base files to be loaded).

To enable runtime mode, set PLN_RUNTIME_AVAILABLE=true in your .env and
ensure `hyperon` is installed.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import ONTOLOGY_DIR, PLN_RUNTIME_AVAILABLE

# ── Scoped DrugAge inference stack ───────────────────────────────────────────────
# The MINIMAL set of hand-written layers needed to lift + rank DrugAge rows,
# loaded into a QUERY-SCOPED space alongside a filtered row slice (see
# run_drugage_ranking). It deliberately EXCLUDES grim_age / hallmarks /
# mechanistic_bridges: those are the CHD axis, add atoms, and would let compound
# names collide with curated Effect bridges. Keeping the stack minimal is what
# keeps the space under the hyperon panic threshold.
DRUGAGE_STACK: list[Path] = [
    ONTOLOGY_DIR / f for f in (
        "system_types.metta",
        "logical_predicates.metta",
        "epistemic_calibration.metta",
        "species_taxonomy.metta",
        "evidence_calibration.metta",
        "pln_deduction.metta",
        "pln_intervention_ranking.metta",
        "drugage_calibration.metta",
    )
]


@dataclass
class PLNAtomResult:
    atom: str
    stv: Optional[dict] = None   # {"strength": float, "confidence": float}


@dataclass
class PLNRunResult:
    status: str                              # "ok" | "empty" | "error"
    results: list[PLNAtomResult] = field(default_factory=list)
    query_time_ms: int = 0
    mode: str = "stub"                       # "stub" | "runtime"
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


# ── Stub mode ──────────────────────────────────────────────────────────────────

_STUB_DATA: list[PLNAtomResult] = [
    PLNAtomResult("RCT_Human",                {"strength": 1.00, "confidence": 1.00}),
    PLNAtomResult("ITP_Positive",             {"strength": 0.90, "confidence": 0.90}),
    PLNAtomResult("ITP_Negative",             {"strength": 0.90, "confidence": 0.90}),
    PLNAtomResult("MultipleHumanTrials",      {"strength": 0.85, "confidence": 0.85}),
    PLNAtomResult("SingleHumanTrial",         {"strength": 0.70, "confidence": 0.70}),
    PLNAtomResult("AnimalStudies_Replicated", {"strength": 0.65, "confidence": 0.65}),
    PLNAtomResult("Epidemiological",          {"strength": 0.60, "confidence": 0.60}),
    PLNAtomResult("AnimalStudies_Single",     {"strength": 0.50, "confidence": 0.50}),
    PLNAtomResult("Preprint",                 {"strength": 0.40, "confidence": 0.40}),
    PLNAtomResult("InVitro",                  {"strength": 0.35, "confidence": 0.35}),
    PLNAtomResult("TraditionalUse",           {"strength": 0.20, "confidence": 0.20}),
]

_STUB_DRUGS: list[PLNAtomResult] = [
    PLNAtomResult("Rapamycin",   {"strength": 0.90, "confidence": 0.90}),
    PLNAtomResult("Metformin",   {"strength": 0.75, "confidence": 0.70}),
    PLNAtomResult("Resveratrol", {"strength": 0.55, "confidence": 0.50}),
    PLNAtomResult("Acarbose",    {"strength": 0.65, "confidence": 0.65}),
]


def _apply_threshold(results: list[PLNAtomResult], threshold: float) -> list[PLNAtomResult]:
    if threshold <= 0:
        return results
    return [r for r in results if r.stv is None or r.stv.get("confidence", 1.0) >= threshold]


def _stub_run(metta_query: str, confidence_threshold: float) -> PLNRunResult:
    """Return plausible mock results based on simple keyword matching."""
    start = time.monotonic()
    time.sleep(0.05)   # simulate slight latency
    q = metta_query.lower()

    if "evidence-confidence" in q:
        # Try to match a specific constant first
        for atom in _STUB_DATA:
            if atom.atom.lower() in q:
                results = [PLNAtomResult(str(atom.stv["confidence"] if atom.stv else "?"))]
                break
        else:
            results = list(_STUB_DATA)
    elif "apply-tv" in q:
        # Extract strength from query text
        m = re.search(r"apply-tv\s+([\d.]+)", metta_query)
        strength = float(m.group(1)) if m else 0.8
        # Find evidence type
        evidence = next(
            (a.atom for a in _STUB_DATA if a.atom in metta_query),
            "AnimalStudies_Replicated",
        )
        conf = next((a.stv["confidence"] for a in _STUB_DATA if a.atom == evidence), 0.65)
        results = [PLNAtomResult(f"(stv {strength:.2f} {conf:.2f})")]
    elif any(kw in q for kw in ("lifespan", "lifespanextender", "extends-lifespan")):
        results = list(_STUB_DRUGS)
    else:
        results = [PLNAtomResult("(stub-result)", {"strength": 0.50, "confidence": 0.30})]

    results = _apply_threshold(results, confidence_threshold)
    elapsed = int((time.monotonic() - start) * 1000)
    return PLNRunResult(
        status="ok" if results else "empty",
        results=results,
        query_time_ms=elapsed,
        mode="stub",
    )


# ── Runtime mode (Hyperon) ─────────────────────────────────────────────────────

def _stv_from_atom(atom_str: str) -> Optional[dict]:
    m = re.search(r"\(stv\s+([\d.]+)\s+([\d.]+)\)", atom_str)
    if m:
        return {"strength": float(m.group(1)), "confidence": float(m.group(2))}
    return None


def _iter_top_level_exprs(text: str):
    """Yield each top-level S-expression from a MeTTa text block."""
    buf: list[str] = []
    depth = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Strip comment-only lines
        if line.startswith(";;") or line.startswith(";"):
            continue
        for ch in line:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        buf.append(line)
        if depth <= 0 and buf:
            expr = " ".join(buf).strip()
            if expr:
                yield expr
            buf = []
            depth = 0
    if buf:
        expr = " ".join(buf).strip()
        if expr:
            yield expr


def _normalize_query(metta_query: str) -> str:
    """Prefix each top-level expression with ! so MeTTa evaluates it."""
    parts: list[str] = []
    for expr in _iter_top_level_exprs(metta_query):
        parts.append(expr if expr.startswith("!") else "!" + expr)
    return "\n".join(parts)


def _hyperon_run(
    metta_query: str,
    confidence_threshold: float,
    kb_files: Optional[list[Path]],
    extra_atoms: Optional[str] = None,
) -> PLNRunResult:
    """Execute query using the real Hyperon MeTTa interpreter.

    The knowledge base is loaded by concatenating every KB file's content into
    one shared space, then the query is run separately against it:

        <contents of kb_file_1>
        <contents of kb_file_2>
        ...
        !(match &self (, ...) $template)

    This is deliberately NOT `!(import! &self <stem>)` per file. Per-file imports
    put each module in its own space, so a `(match &self ...)` inside one file
    cannot see atoms defined in another (it returns empty silently), and module
    name resolution is order-dependent. One shared space avoids both problems.

    Parameters
    ----------
    metta_query:
        One or more top-level MeTTa expressions (``!`` prefix optional —
        added automatically if absent).
    confidence_threshold:
        Filter out results whose STV confidence is below this value.
    kb_files:
        Ordered list of .metta knowledge-base files whose contents are loaded
        into the space before the query runs.
    """
    try:
        from hyperon import MeTTa  # type: ignore

        normalized = _normalize_query(metta_query)
        if not normalized:
            return PLNRunResult(status="empty", mode="runtime")

        # Concatenate every KB file's content into one block. A missing or
        # unreadable file is skipped rather than aborting the whole query.
        kb_blocks: list[str] = []
        for path in (kb_files or []):
            try:
                kb_blocks.append(path.read_text(encoding="utf-8"))
            except OSError:
                pass
        # A caller-supplied slice (e.g. a filtered set of DrugAge rows selected
        # per-query) is injected into the SAME space after the files. This is how
        # a query-scoped space is assembled without loading a whole ETL dump.
        if extra_atoms:
            kb_blocks.append(extra_atoms)
        kb_text = "\n".join(kb_blocks)

        # Log the query (and which KB files were loaded) for debugging. The KB
        # bodies are large and unchanging, so only their names are recorded.
        try:
            from config import LOGS_DIR
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            loaded = "\n".join(f";; loaded: {p.name}" for p in (kb_files or []))
            (LOGS_DIR / "last_query.metta").write_text(
                f"{loaded}\n\n{normalized}", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass  # logging must never break query execution

        try:
            metta = MeTTa()
            start = time.monotonic()
            if kb_text.strip():
                metta.run(kb_text)                   # populate &self; result ignored
            raw: list[list] = metta.run(normalized)  # every group is a query result
            elapsed = int((time.monotonic() - start) * 1000)
        except Exception as run_exc:
            return PLNRunResult(status="error", mode="runtime", error=str(run_exc))

        results: list[PLNAtomResult] = []
        for result_group in raw:
            for atom in result_group:
                atom_str = str(atom)
                results.append(PLNAtomResult(atom=atom_str, stv=_stv_from_atom(atom_str)))

        results = _apply_threshold(results, confidence_threshold)
        return PLNRunResult(
            status="ok" if results else "empty",
            results=results,
            query_time_ms=elapsed,
            mode="runtime",
        )
    except Exception as exc:   # noqa: BLE001
        return PLNRunResult(status="error", mode="runtime", error=str(exc))


# ── Public API ─────────────────────────────────────────────────────────────────

def run_query(
    metta_query: str,
    confidence_threshold: float = 0.0,
    kb_files: Optional[list[Path]] = None,
    extra_atoms: Optional[str] = None,
) -> PLNRunResult:
    """Execute a MeTTa query, using stub or runtime mode as configured.

    Parameters
    ----------
    metta_query:
        MeTTa expression(s) to execute.
    confidence_threshold:
        Minimum STV confidence for results to be included.
    kb_files:
        .metta KB files to load into the Hyperon space (runtime mode only).
    extra_atoms:
        Optional raw MeTTa text injected into the SAME space after the files —
        used to scope a query to a selected data slice (e.g. DrugAge rows).
    """
    if not metta_query.strip():
        return PLNRunResult(status="empty", mode="stub" if not PLN_RUNTIME_AVAILABLE else "runtime")
    if PLN_RUNTIME_AVAILABLE:
        return _hyperon_run(metta_query, confidence_threshold, kb_files, extra_atoms)
    return _stub_run(metta_query, confidence_threshold)


def run_drugage_ranking(
    compounds: list[str],
    *,
    outcome: str = "Mortality",
    best_per_compound: bool = True,
    limit: Optional[int] = None,
    source: Optional[Path] = None,
    confidence_threshold: float = 0.0,
) -> tuple[PLNRunResult, list]:
    """Rank real DrugAge compounds by calibrated, signed effect on lifespan.

    Assembles a QUERY-SCOPED hyperon space = the DrugAge inference stack
    (DRUGAGE_STACK) + only the DrugAge rows matching `compounds` (selected by
    ontology.drugage_selector, capped under the panic threshold), then runs
    `rank-interventions` against `outcome` (default Mortality — the compound ->
    Lifespan -> Mortality chain keeps the Neg=beneficial convention; see
    docs/etl_inference_wiring.md).

    Returns (PLNRunResult, selected_rows). The result's atoms are the ranked,
    signed, uncertainty-quantified `(scored ...)` tuples; selected_rows carries
    the provenance of exactly which rows were injected.

    Keep the compound pool to a handful (Demo-2 scale): the underlying MeTTa
    insertion sort in pln_intervention_ranking is ~O(n^2) with a high constant
    (docs/etl_inference_wiring.md §7). best_per_compound (default) keeps the pool
    at one entry per compound.
    """
    from ontology.drugage_selector import MAX_ROWS, build_drugage_slice

    slice_text, rows = build_drugage_slice(
        compounds,
        best_per_compound=best_per_compound,
        limit=limit if limit is not None else MAX_ROWS,
        source=source,
    )
    # Candidate atoms = the compounds actually present in the slice (a requested
    # compound with no matching row simply drops out — no false ranking).
    cands = " ".join(sorted({r.compound for r in rows}))
    if not cands:
        return PLNRunResult(status="empty", mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub"), rows

    query = f"!(rank-interventions &self ({cands}) {outcome})"
    result = run_query(
        query,
        confidence_threshold=confidence_threshold,
        kb_files=DRUGAGE_STACK,
        extra_atoms=slice_text,
    )
    return result, rows
