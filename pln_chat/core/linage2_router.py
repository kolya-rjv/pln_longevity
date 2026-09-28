"""Route a LinAge2 question to the query-scoped LinAge2 stack, and read its answers back.

Why a router, again
-------------------
`core/drugage_router.py` explains the pattern: the generic chat path runs a
translated query against the shared execution space, and some layers cannot live
there. The LinAge2 layer is one of them — not for size, but for hyperon 0.2.10's
head-symbol budget (docs/nhanes_integration.md §8, linage2_core.metta header):
its four new head symbols abort the shared space, and were measured to. So the
LinAge2 forms run in `core.pln_runner.LINAGE2_PATIENT_STACK`, the NHANES patient
stack plus the smoking lever plus the three LinAge2 files, with the caller's
patient atoms injected — exactly as `rank-drugage-lifespan` runs in DRUGAGE_STACK
with its row slice.

This module is the glue:

* `parse_linage2_query` recognises a translated (or hand-written) query that
  calls one of the `linage-*` forms, so `/query`, `/metta/run` and the Gradio
  chat can dispatch it before the generic path validates it against a space
  that does not define those forms;
* `route_linage2_query` runs it in the scoped space;
* `analyze_patient` is the LLM-free entry point behind `POST /linage2/analyze`:
  one MeTTa program (decomposition, hazard, risk, the standing counterfactuals),
  parsed into JSON so an app never has to read atoms.

Kept free of any FastAPI or Gradio import so it is unit-testable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.pln_runner import (
    LINAGE2_GENERATED_BASELINE,
    PLNRunResult,
    linage2_patient_kb,
    run_query,
)


def _baseline_present() -> bool:
    return LINAGE2_GENERATED_BASELINE.exists()

#: Every form pln_linage2.metta defines that a caller may reasonably name. The
#: `-patient` drivers are what the translator is taught; the bare forms take an
#: explicit outcome and are here so a hand-written /metta/run query routes too.
LINAGE2_FORMS: tuple[str, ...] = (
    "linage-decomposition-patient",
    "linage-drivers-patient",
    "linage-hazard-patient",
    "linage-risk-patient",
    "linage-counterfactual-patient",
    "linage-project-risk-patient",
    "linage-scenarios-patient",
    "linage-decomposition",
    "linage-drivers",
    "linage-hazard",
    "linage-risk",
    "linage-counterfactual",
    "linage-project-risk",
    "linage-witnessed",
    "linage-biomarker-causes",
    "linage-lever-drivers",
    "linage-delta",
)

#: The four standing scenarios pln_linage2.metta's `linage-scenarios-patient` runs.
DEFAULT_LEVERS: tuple[str, ...] = (
    "ChronicInflammation", "CellularSenescence", "InsulinResistance", "SmokingCessation",
)

_FORM_RE = re.compile(r"\(\s*(linage-[a-z][a-z0-9\-]*)\b")
_LEVER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def parse_linage2_query(metta_query: str) -> Optional[list[str]]:
    """The LinAge2 forms a query calls, in order of appearance, or None.

    None means "not a LinAge2 query — take the generic path". An empty list is
    never returned: a query either names a form or it does not.
    """
    found = [m.group(1) for m in _FORM_RE.finditer(metta_query or "")]
    known = [f for f in found if f in LINAGE2_FORMS]
    return known or None


_DELTA_RE = re.compile(r"\(LinAgeDelta\s+(\S+)\s+[-\d.eE]+\)")
_PATIENT_ARG_RE = re.compile(r"\(\s*linage-[a-z][a-z0-9\-]*\s+&self\s+([A-Za-z][A-Za-z0-9_]*)")


def linage2_form_warnings(metta_query: str, extra_atoms: Optional[str] = None) -> list[str]:
    """Say why a LinAge2 form is about to come back empty, before it does.

    Every LinAge2 form reads (LinAgeDelta <Patient> …), and that atom exists only
    for a patient whose request carried a `linage2` block (or a LinAgeAccel marker,
    which gives the hazard alone). The curated patients have none. So a form naming
    a patient with no delta is valid, runs, and returns nothing — the shape API.md
    tells a caller to read as "the KB cannot express this", which here would be
    wrong twice over. Name the cause instead.
    """
    forms = parse_linage2_query(metta_query)
    if not forms:
        return []
    with_delta = set(_DELTA_RE.findall(extra_atoms or ""))
    named = {m.group(1) for m in _PATIENT_ARG_RE.finditer(metta_query)}
    missing = sorted(p for p in named if p not in with_delta)
    if not missing:
        return []
    return [
        f"LinAge2 form(s) {', '.join(forms)} name patient(s) {', '.join(missing)} "
        f"that carry no LinAge2 result in this request. The forms read "
        f"(LinAgeDelta <Patient> …), which only a caller-supplied `patient.linage2` "
        f"block (the LinAge2 /predict response) creates — the built-in patients have "
        f"none. An empty result here means 'no LinAge2 input', not 'no effect'."
    ]


def route_linage2_query(
    metta_query: str,
    *,
    extra_atoms: Optional[str] = None,
    confidence_threshold: float = 0.0,
) -> PLNRunResult:
    """Run `metta_query` in the LinAge2 scoped space with the caller's atoms."""
    return run_query(
        metta_query,
        confidence_threshold=confidence_threshold,
        kb_files=linage2_patient_kb(),
        extra_atoms=extra_atoms,
    )


# ── a small s-expression reader for the answers ──────────────────────────────
# `_hyperon_run` hands back every result atom as a STRING. The atoms this layer
# produces are plain nested expressions of symbols and numbers, so a ~30-line
# reader is enough to turn them into JSON — the same job `parse_scored` does for
# the DrugAge ranking, generalised.

_TOKEN_RE = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()]+')
_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def parse_sexpr(text: str) -> Any:
    """One expression -> nested lists; numbers become floats, symbols stay strings."""
    tokens = _TOKEN_RE.findall(text)
    pos = 0

    def read() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of expression")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items = []
            while pos < len(tokens) and tokens[pos] != ")":
                items.append(read())
            if pos >= len(tokens):
                raise ValueError("unbalanced expression")
            pos += 1  # the ')'
            return items
        if tok == ")":
            raise ValueError("unexpected ')'")
        if _NUM_RE.match(tok):
            return float(tok)
        if tok.startswith('"') and tok.endswith('"'):
            return tok[1:-1]
        return tok

    value = read()
    if pos != len(tokens):
        raise ValueError("trailing tokens after expression")
    return value


def _fields(items: list) -> dict[str, Any]:
    """`[(name v…) …]` -> {name: v or [v…]} for the tagged fields an atom carries."""
    out: dict[str, Any] = {}
    for item in items:
        if isinstance(item, list) and item and isinstance(item[0], str):
            out[item[0]] = item[1] if len(item) == 2 else item[1:]
    return out


def _contribution(sx: list) -> dict:
    """(Contribution <F> (years y) <Prov> (ReadsOut …) (DrivenBy (…)))"""
    _, symbol, years, prov, reads, driven = sx
    entry: dict[str, Any] = {
        "symbol": symbol,
        "years": years[1] if isinstance(years, list) else years,
        "provenance": prov,
        "reads_out": None,
        "witnessed": None,
        "driven_by": [],
    }
    if isinstance(reads, list) and len(reads) >= 2 and reads[1] != "None":
        entry["reads_out"] = reads[1]
        if len(reads) >= 3 and isinstance(reads[2], list) and len(reads[2]) == 2:
            entry["witnessed"] = reads[2][1] == "True"
    if isinstance(driven, list) and len(driven) == 2 and isinstance(driven[1], list):
        entry["driven_by"] = [str(c) for c in driven[1]]
    return entry


def decomposition_to_dict(sx: list) -> dict:
    f = _fields(sx[2:])
    measured = [_contribution(c) for c in (f.get("Measured") or [])]
    imputed = [_contribution(c) for c in (f.get("Imputed") or [])]
    key = lambda c: -abs(c["years"])  # noqa: E731
    return {
        "patient": sx[1],
        "delta_years": f.get("delta-years"),
        "attributed_measured_years": f.get("attributed-measured"),
        "attributed_imputed_years": f.get("attributed-imputed"),
        "age_term_residual_years": f.get("age-term-residual"),
        "measured": sorted(measured, key=key),
        "imputed": sorted(imputed, key=key),
    }


def hazard_to_dict(sx: list) -> dict:
    f = _fields(sx[3:])
    return {
        "patient": sx[1], "outcome": sx[2],
        "delta_years": f.get("delta-years"),
        "hazard_multiplier": f.get("hazard-multiplier"),
        "confidence": f.get("confidence"),
    }


def risk_to_dict(sx: list) -> dict:
    f = _fields(sx[3:])
    ci = f.get("ci")
    return {
        "patient": sx[1], "outcome": sx[2],
        "point": f.get("point"),
        "ci_low": ci[0] if isinstance(ci, list) else None,
        "ci_high": ci[1] if isinstance(ci, list) else None,
        "confidence": f.get("confidence"),
        "baseline": f.get("baseline"),
        "multiplier": f.get("multiplier"),
        "clock": f.get("clock"),
    }


def counterfactual_to_dict(sx: list) -> dict:
    """(LinAgeCounterfactual <P> <lever> (expected-delta-years d) (signed Neg (stv s c)) (Via (…)))"""
    f = _fields(sx[3:])
    signed = f.get("signed")
    stv = signed[1] if isinstance(signed, list) and len(signed) == 2 else None
    via = f.get("Via")
    return {
        "patient": sx[1], "lever": sx[2],
        "expected_delta_years": f.get("expected-delta-years"),
        "strength": stv[1] if isinstance(stv, list) and len(stv) == 3 else None,
        "confidence": stv[2] if isinstance(stv, list) and len(stv) == 3 else None,
        "via": [str(v) for v in via] if isinstance(via, list) else [],
    }


def projected_to_dict(sx: list) -> dict:
    """(ProjectedRisk <lever> <outcome> (point p') (reduction r) (delta-years d) (confidence c) (Via (…)) (clock LinAge2))"""
    f = _fields(sx[3:])
    via = f.get("Via")
    return {
        "lever": sx[1], "outcome": sx[2],
        "point": f.get("point"), "reduction": f.get("reduction"),
        "delta_years": f.get("delta-years"), "confidence": f.get("confidence"),
        "via": [str(v) for v in via] if isinstance(via, list) else [],
        "clock": f.get("clock"),
    }


@dataclass
class LinAge2Analysis:
    metta_query: str
    decomposition: Optional[dict] = None
    hazard: Optional[dict] = None
    risk: Optional[dict] = None
    counterfactuals: list[dict] = field(default_factory=list)
    projected_risks: list[dict] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    query_time_ms: int = 0
    pln_status: str = "empty"
    pln_error: Optional[str] = None
    pln_error_code: Optional[str] = None


def analysis_program(
    patient_id: str,
    levers: tuple[str, ...] = DEFAULT_LEVERS,
    *,
    with_projections: bool = True,
) -> str:
    """The MeTTa program /linage2/analyze runs: one `!` per deliverable.

    `with_projections=False` leaves out `linage-risk-patient` and the per-lever
    `linage-project-risk-patient`: without a generated baseline both yield nothing,
    and each projection re-runs its counterfactual, so skipping them halves the
    program's wall time for an answer that was going to be empty anyway.
    """
    for lever in levers:
        if not _LEVER_RE.match(lever):
            raise ValueError(f"not a lever symbol: {lever!r}")
    lines = [
        f"!(linage-decomposition-patient &self {patient_id})",
        f"!(linage-hazard-patient &self {patient_id})",
    ]
    if with_projections:
        lines.append(f"!(linage-risk-patient &self {patient_id})")
    for lever in levers:
        lines.append(f"!(linage-counterfactual-patient &self {patient_id} {lever})")
        if with_projections:
            lines.append(f"!(linage-project-risk-patient &self {patient_id} {lever})")
    return "\n".join(lines)


def collect_analysis(program: str, result: PLNRunResult) -> LinAge2Analysis:
    """Dispatch the flattened result atoms by head symbol."""
    out = LinAge2Analysis(
        metta_query=program,
        query_time_ms=result.query_time_ms,
        pln_status=result.status,
        pln_error=result.error,
        pln_error_code=getattr(result, "error_code", None),
    )
    for atom in result.results:
        text = atom.atom
        try:
            sx = parse_sexpr(text)
        except ValueError:
            out.unparsed.append(text)
            continue
        head = sx[0] if isinstance(sx, list) and sx else None
        try:
            if head == "LinAgeDecomposition":
                out.decomposition = decomposition_to_dict(sx)
            elif head == "LinAgeHazard":
                out.hazard = hazard_to_dict(sx)
            elif head == "RiskPrediction":
                out.risk = risk_to_dict(sx)
            elif head == "LinAgeCounterfactual":
                out.counterfactuals.append(counterfactual_to_dict(sx))
            elif head == "ProjectedRisk":
                out.projected_risks.append(projected_to_dict(sx))
            else:
                out.unparsed.append(text)
        except (IndexError, TypeError, ValueError):
            out.unparsed.append(text)
    return out


def analyze_patient(
    patient_id: str,
    patient_atoms: str,
    *,
    levers: tuple[str, ...] = DEFAULT_LEVERS,
    generated: tuple[Path, ...] = (),
) -> LinAge2Analysis:
    """Run the whole LinAge2 analysis for one caller-supplied patient, in process.

    `generated` lets a caller add ETL-produced NHANES record files (the all-cause
    baseline that turns a hazard into an absolute risk); absent files are skipped.
    """
    program = analysis_program(
        patient_id, levers,
        with_projections=any(p.exists() for p in generated) or _baseline_present(),
    )
    result = run_query(
        program,
        confidence_threshold=0.0,
        kb_files=linage2_patient_kb(*generated),
        extra_atoms=patient_atoms,
    )
    return collect_analysis(program, result)
