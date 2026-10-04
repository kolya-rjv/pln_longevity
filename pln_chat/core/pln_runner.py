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


# ── Scoped CellAge inference stack ───────────────────────────────────────────────
# The MINIMAL set of layers needed to lift a CellAge curated senescence row into
# an (Effect <gene> CellularSenescence <sign> (stv s c)) link, loaded into a
# QUERY-SCOPED space alongside a filtered row slice (see run_cellage_effects).
# It is even smaller than DRUGAGE_STACK: no species taxonomy (CellAge is human
# cell lines only) and no intervention ranking (a gene is not an intervention,
# and this layer deliberately does NOT bridge CellularSenescence to mortality —
# see cellage_calibration.metta §2). Keeping the stack minimal is what keeps the
# space under the hyperon abort boundary, which for this query shape is between
# 125 and 150 data rows (measured; ontology.cellage_selector.MAX_ROWS).
CELLAGE_STACK: list[Path] = [
    ONTOLOGY_DIR / f for f in (
        "system_types.metta",
        "logical_predicates.metta",
        "epistemic_calibration.metta",
        "evidence_calibration.metta",
        "pln_deduction.metta",
        "cellage_calibration.metta",
    )
]


# ── NHANES patient-grounding stack ────────────────────────────────────────────
#
# A QUERY-SCOPED space, for the same reason CELLAGE_STACK is one, and measured rather
# than assumed. hyperon 0.2.10 aborts the process — a non-unwinding Rust panic in its
# space trie, uncatchable from Python — once one space carries too many DISTINCT HEAD
# SYMBOLS. The app's shared execution space already holds 201 of them with no margin:
# adding the ~9 the NHANES layers introduce aborts it at once, while thousands of extra
# rule DEFINITIONS in that same space are harmless. Head symbols and definitions are
# different budgets, and only the first is exhausted.
#
# So the NHANES grounding does not join the shared space. This list is the minimal set
# that answers a patient question, and with both NHANES layers in it the space runs
# patient-z, diagnose-patient and predict-risk-patient cleanly (verified in
# tests/test_nhanes_integration.py).
#
# Add a file here only if a patient query needs it: the whole point is that this space
# stays small enough for the engine.
NHANES_PATIENT_STACK: list[Path] = [
    ONTOLOGY_DIR / f for f in (
        "system_types.metta",
        "logical_predicates.metta",
        "epistemic_calibration.metta",
        "grim_age_core.metta",
        "grim_age_lu2019_evidence.metta",
        "evidence_calibration.metta",
        "hallmarks_core.metta",
        "hallmarks_lopezotin2023_intervention_evidence.metta",
        "mechanistic_bridges.metta",
        "pln_deduction.metta",
        "pln_intervention_ranking.metta",
        "pln_abductive_diagnosis.metta",
        "patient_profile.metta",
        "nhanes_reference.metta",
        "pln_counterfactual.metta",
        "pln_risk_prediction.metta",
        "nhanes_baseline.metta",
    )
]


def nhanes_patient_kb(*generated: Path) -> list[Path]:
    """NHANES_PATIENT_STACK plus any ETL-generated record files that exist.

    The generated files (build/nhanes_reference.metta, build/nhanes_mortality_baseline
    .metta, build/nhanes_dnam_clocks.metta) are what carry the actual numbers; without
    them the layers are inert and every lookup correctly yields nothing. Missing paths
    are skipped rather than raising, so a partial ETL run still works for what it did
    produce.
    """
    return list(NHANES_PATIENT_STACK) + [p for p in generated if p.exists()]


# ── Patient stack ─────────────────────────────────────────────────────────────
#
# Where every question that NAMES A PATIENT runs: the shared runtime stack minus the
# files no patient form reads. Measured, like the stacks above, and for the same
# reason: the full shared space carries its 201 distinct head symbols with no
# margin, and at that edge whether a query aborts hyperon 0.2.10 (a panic in its
# space index, hyperon-space/src/index/trie.rs:179, `unwrap()` on a missing hashed
# atom) depends on details as small as one float. Observed in the full space:
#   * built-in patients: rank-interventions-for-patient (Patient001, Patient002),
#     recommend-supplements-patient (Patient001, Patient002), supplement-for-patient
#     (Patient001) all abort;
#   * a caller-supplied patient: diagnose-patient, recommend-supplements-patient and
#     rank-interventions-for-patient abort for CRP/HbA1c z in {0.31, 1.2, 2.0, 0.73,
#     2.6} — 15 of 15 — while z = 0.3 or 0.5 happen to run; dropping any one of a
#     dozen unrelated files also makes it run.
# In this stack all of those answer, and wherever the full stack answers, this one
# gives a BYTE-IDENTICAL answer (33 patient-form runs over the three built-in
# patients), with at least 64 head symbols of margin on top of a caller patient
# (tests/test_patient_stack.py). What it leaves out: the DrugAge species taxonomy and
# short entries (the DrugAge ranking runs in its own DRUGAGE_STACK anyway), the
# DrugAge calibration, the human-evidence and hallmark-targeting layers, the López-
# Otín anchors and the measurement-type vocabulary — none of which a patient form reads.
PATIENT_STACK_EXCLUDED: tuple[str, ...] = (
    "measurement_types.metta",
    "species_taxonomy.metta",
    "drugage_entries.metta",
    "hallmarks_lopezotin2023_anchors.metta",
    "hallmark_targeting.metta",
    "drugage_calibration.metta",
    "human_evidence.metta",
)


def patient_stack(runtime_paths: list[Path]) -> list[Path]:
    """The runtime stack, in its order, minus PATIENT_STACK_EXCLUDED."""
    return [p for p in runtime_paths if p.name not in PATIENT_STACK_EXCLUDED]


# ── LinAge2 clinical-clock stack ──────────────────────────────────────────────
#
# A QUERY-SCOPED space for the LinAge2 forms (pln_linage2.metta), routed to by
# core/linage2_router.py. Scoped for the same measured reason as NHANES_PATIENT_STACK:
# the shared space is saturated on distinct head symbols, and the LinAge2 layer adds
# three (MeasuresBiomarker, and the per-request LinAgeDelta / LinAgeContribution).
#
# It is NOT "the NHANES patient stack plus LinAge2". Measured with the whole
# NHANES_PATIENT_STACK underneath, the LinAge2 layer loaded with ZERO head-symbol
# margin — two more heads and any query aborted — which also means a generated
# NHANES baseline file (12 field heads) could never join it, and an absolute risk
# would have been unreachable forever. Dropping what no LinAge2 form reads
# (pln_intervention_ranking, pln_abductive_diagnosis, the López-Otín intervention
# records, nhanes_reference) bought a margin of 32+ heads, enough for the generated
# all-cause baseline with room to spare (tests/test_linage2.py asserts both the
# canary queries and the margin). So this list is minimal on purpose; add a file
# only if a LinAge2 form needs it, and re-run the margin test when you do.
#
# What each entry is for:
#   types / predicates / evidence tiers ............ system_types, logical_predicates,
#                                                    epistemic_calibration
#   ExposureBiomarker, SmokingPackYears, Outcomes .. grim_age_core, grim_age_lu2019_evidence
#   calibrate-tv ................................... evidence_calibration
#   the hallmark list the causes are drawn from ..... hallmarks_core
#   the causal graph (CRP / glucose / HbA1c axes) ... mechanistic_bridges
#   infer, chain-discount .......................... pln_deduction
#   patient-z, unique-tuple, elevated-z-threshold .. patient_profile
#   pos-transmission, cf-tv-s / cf-tv-c ............ pln_counterfactual
#   patient-baseline, risk-conf-discount, risk-ci-k  pln_risk_prediction
#   the outcome-keyed data baseline lookup ......... nhanes_baseline
#   SmokingCessation, SmokingPackYears edges ....... lifestyle_evidence
#   the clock, its inputs, the hazard record, rules  linage2_core, linage2_fong2025_evidence,
#                                                    pln_linage2
LINAGE2_PATIENT_STACK: list[Path] = [
    ONTOLOGY_DIR / f for f in (
        "system_types.metta",
        "logical_predicates.metta",
        "epistemic_calibration.metta",
        "grim_age_core.metta",
        "grim_age_lu2019_evidence.metta",
        "evidence_calibration.metta",
        "hallmarks_core.metta",
        "mechanistic_bridges.metta",
        "pln_deduction.metta",
        "patient_profile.metta",
        "pln_counterfactual.metta",
        "pln_risk_prediction.metta",
        "nhanes_baseline.metta",
        "lifestyle_evidence.metta",
        "linage2_core.metta",
        "linage2_fong2025_evidence.metta",
        "pln_linage2.metta",
    )
]

#: The three files that ARE the LinAge2 layer. Never in _INFERENCE_STACK (api.py /
#: app.py) — the shared space cannot take their head symbols — and asserted absent
#: from it by tests/test_linage2.py.
LINAGE2_LAYER_FILES: tuple[str, ...] = (
    "linage2_core.metta", "linage2_fong2025_evidence.metta", "pln_linage2.metta",
)

#: The one ETL output the LinAge2 stack picks up when present: the survey-weighted
#: all-cause-mortality baseline (scripts/run_etl.sh -> nhanes_mortality_etl.py). It
#: is what turns `linage-hazard-patient` (always available) into `linage-risk-patient`
#: (an absolute ten-year risk). The reference-distribution and DNAm-clock outputs are
#: deliberately NOT here: no LinAge2 form reads them, nhanes_reference.metta is not in
#: this stack, and their 12-21 head symbols each would spend the margin for nothing.
LINAGE2_GENERATED_BASELINE: Path = ONTOLOGY_DIR / "build" / "nhanes_mortality_baseline.metta"


def linage2_patient_kb(*generated: Path) -> list[Path]:
    """LINAGE2_PATIENT_STACK plus the generated all-cause baseline, when it exists.

    Extra `generated` paths are appended if present, the same contract as
    `nhanes_patient_kb`. With no baseline file the stack is complete and every
    LinAge2 form except the absolute risk answers; `linage-risk-patient` then
    yields nothing, which is the honest answer to "what is my risk" without a
    baseline to multiply.
    """
    extras = [LINAGE2_GENERATED_BASELINE, *generated]
    seen: set[Path] = set()
    out = list(LINAGE2_PATIENT_STACK)
    for p in extras:
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


@dataclass
class PLNAtomResult:
    atom: str
    stv: Optional[dict] = None   # {"strength": float, "confidence": float}
    #: Which `!` expression of the program produced this atom (0-based), when the
    #: runtime knows. Lets a program run in pieces be put back in program order.
    expr_index: Optional[int] = None


@dataclass
class PLNRunResult:
    status: str                              # "ok" | "empty" | "error"
    results: list[PLNAtomResult] = field(default_factory=list)
    query_time_ms: int = 0
    mode: str = "stub"                       # "stub" | "runtime"
    error: Optional[str] = None
    #: Machine-readable failure class when status == "error", so the HTTP layer
    #: can pick a status code without grepping the message.
    #:   runtime_error       — the hyperon interpreter raised
    #:   drugage_build_missing — build/drugage_etl.metta has not been generated
    error_code: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


@dataclass
class ScoredCompound:
    """One parsed `(scored <compound> <score> (signed <sign> (stv s c)))` tuple.

    The MeTTa ranking returns its whole sorted pool as ONE atom, which is why
    `confidence_threshold` never filtered anything: `_apply_threshold` reads the
    FIRST `(stv …)` in that string and then keeps or drops the entire ranking on
    that one number. Scoring each compound separately (see
    `run_drugage_ranking`) gives one atom per compound, and this dataclass is
    the structured form the HTTP layer returns.
    """
    compound: str
    score: float
    sign: str          # "Neg" = protective (lowers mortality) | "Pos" = harmful
    strength: float
    confidence: float
    atom: str

    @property
    def protective(self) -> bool:
        return self.sign == "Neg"

    @property
    def direction(self) -> str:
        """'protective' | 'harmful' | 'no_effect' — the sign in words.

        `sign` is the MeTTa Effect convention and has no zero: a row reporting
        0.0% lifespan change is `Neg` because it is not negative. Reading that
        straight out as "protective" put a contradiction inside one response —
        `direction: "protective"` next to a `score` of exactly 0.0 and a
        `semantics.zero_score` note saying that a 0.0 is a REPORTED NULL. The
        most consequential rows in the build are exactly these: the ITP nulls
        for metformin and resveratrol, at the highest confidence the KB gives.
        """
        if self.strength == 0.0:
            return "no_effect"
        return "protective" if self.protective else "harmful"

    def as_dict(self) -> dict:
        return {
            "compound": self.compound,
            "score": self.score,
            "sign": self.sign,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "atom": self.atom,
        }


_SCORED_RE = re.compile(
    r"\(scored\s+(?P<compound>\S+)\s+(?P<score>[-+\d.eE]+)\s+"
    r"\(signed\s+(?P<sign>Pos|Neg)\s+"
    r"\(stv\s+(?P<strength>[-+\d.eE]+)\s+(?P<confidence>[-+\d.eE]+)\)\)\)"
)


def parse_scored(atom: str) -> list[ScoredCompound]:
    """Every `(scored …)` tuple inside `atom` (one, or a whole ranked tuple)."""
    out: list[ScoredCompound] = []
    for m in _SCORED_RE.finditer(atom):
        try:
            out.append(ScoredCompound(
                compound=m.group("compound"),
                score=float(m.group("score")),
                sign=m.group("sign"),
                strength=float(m.group("strength")),
                confidence=float(m.group("confidence")),
                atom=m.group(0),
            ))
        except ValueError:   # a malformed number never breaks a ranking
            continue
    return out


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


# A collapsed MeTTa result (a ranking, a diagnosis, a tiered recommendation)
# arrives as ONE atom holding a TUPLE of sub-expressions, each with its own
# `(stv s c)`. `_stv_from_atom` reads only the FIRST one, so an atom-level
# filter keeps or drops the whole tuple on the leading entry's confidence —
# which is why `confidence_threshold` looked like a no-op on /drugage/rank and
# /query. When a threshold is actually requested we therefore descend one level
# and filter the ENTRIES, re-assembling the tuple so the atom's shape (and every
# caller that parses it) is unchanged.
def _split_top_level(atom: str) -> Optional[list[str]]:
    """Split `(a…) (b…) (c…)` wrapped in one outer pair of parens, else None."""
    text = atom.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    inner = text[1:-1].strip()
    if not inner.startswith("("):
        return None
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                parts.append(inner[start:i + 1])
            elif depth < 0:
                return None
    if depth != 0:
        return None
    # Only a genuine tuple (>1 element, nothing but sub-expressions) qualifies.
    rebuilt = " ".join(parts)
    if len(parts) < 2 or rebuilt != " ".join(inner.split()):
        return None
    return parts


def _filter_tuple_entries(atom: str, threshold: float) -> Optional[str]:
    """Drop the sub-expressions below `threshold`.

    None means "nothing here survives" — either the atom is not a tuple (so its
    own failing STV is the verdict) or every entry failed.
    """
    parts = _split_top_level(atom)
    if parts is None:
        return None
    kept = []
    for part in parts:
        stv = _stv_from_atom(part)
        if stv is None or stv.get("confidence", 1.0) >= threshold:
            kept.append(part)
    if not kept:
        return None
    return "(" + " ".join(kept) + ")"


def _apply_threshold(results: list[PLNAtomResult], threshold: float) -> list[PLNAtomResult]:
    if threshold <= 0:
        return results
    kept: list[PLNAtomResult] = []
    for r in results:
        if r.stv is not None and r.stv.get("confidence", 1.0) >= threshold:
            kept.append(r)
            continue
        if r.stv is None:
            kept.append(r)
            continue
        # The atom's leading STV failed. Before discarding it, check whether it
        # is a collapsed tuple whose OTHER entries pass.
        filtered = _filter_tuple_entries(r.atom, threshold)
        if filtered is not None:
            kept.append(PLNAtomResult(atom=filtered, stv=_stv_from_atom(filtered)))
    return kept


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

#: hyperon prints a small magnitude in scientific notation (0.00001 -> 1e-05),
#: and `[\d.]+` cannot match an exponent or a sign. An unparsed truth value
#: reads as "no STV", which `_apply_threshold` treats as "keep unconditionally"
#: — so the lowest-confidence results were exactly the ones a confidence filter
#: could not remove.
_STV_RE = re.compile(
    r"\(stv\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)"
)


def _stv_from_atom(atom_str: str) -> Optional[dict]:
    m = _STV_RE.search(atom_str)
    if m:
        return {"strength": float(m.group(1)), "confidence": float(m.group(2))}
    return None


def _iter_top_level_exprs(text: str):
    """Yield each top-level S-expression of a MeTTa program, in order.

    Character-level. It used to cut only at the end of a LINE whose parentheses
    balanced, so `!(a) !(b)` on one line was one "expression": `_normalize_query`
    put a `!` on the first only (the second was silently added to the space instead
    of evaluated), and the LinAge2 splitter could not separate the two. Now: several
    expressions on a line are several expressions; a `;` comment runs to the end of
    its line (outside a string) and is dropped; a parenthesis inside a "string" does
    not count; a `!` stays with the expression it prefixes; a bare top-level symbol
    is an expression of its own. Whitespace outside strings collapses to one space.
    """
    out: list[str] = []
    depth = 0
    in_str = escaped = False
    i, n = 0, len(text)

    def pending() -> str:
        return "".join(out).strip()

    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch in " \t\r\n":
            if depth > 0:
                if out and out[-1] != " ":
                    out.append(" ")
            elif pending() and pending() != "!":
                yield pending()
                out = []
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch == "(":
            if depth == 0 and pending() and pending() != "!":
                yield pending()                      # a bare token glued to "("
                out = []
            depth += 1
            out.append(ch)
        elif ch == ")":
            out.append(ch)
            depth -= 1
            if depth <= 0:
                depth = 0
                yield pending()
                out = []
        else:
            out.append(ch)
        i += 1
    if pending():
        yield pending()


def split_top_level_exprs(text: str) -> list[str]:
    """Every top-level S-expression of a MeTTa program, in order (comments dropped)."""
    return list(_iter_top_level_exprs(text or ""))


def merge_run_results(
    parts: list[PLNRunResult],
    positions: Optional[list[list[int]]] = None,
) -> PLNRunResult:
    """One result for a program that ran as several pieces in different spaces.

    An error in any piece is the result: a half-answered program must not read as
    a complete one. Otherwise the atoms are joined in PROGRAM order when
    `positions` says where each piece's expressions sat in the original program
    (`positions[k][j]` = original index of piece k's j-th expression) and the
    runtime tagged its atoms with `expr_index`; atoms without a tag keep piece order.
    """
    if not parts:
        return PLNRunResult(status="empty", mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub")
    for part in parts:
        if part.status == "error":
            return part
    keyed = []
    for k, part in enumerate(parts):
        for seq, r in enumerate(part.results):
            where = None
            if positions is not None and r.expr_index is not None \
                    and 0 <= r.expr_index < len(positions[k]):
                where = positions[k][r.expr_index]
            keyed.append(((where if where is not None else 10 ** 9 + k), k, seq, r))
    keyed.sort(key=lambda t: t[:3])
    results = [t[3] for t in keyed]
    return PLNRunResult(
        status="ok" if results else "empty",
        results=results,
        query_time_ms=sum(part.query_time_ms for part in parts),
        mode=parts[0].mode,
    )


def run_query_parts(parts: list[dict], confidence_threshold: float = 0.0) -> list[PLNRunResult]:
    """Run several programs, each against its own KB, one after another in THIS
    process — so a program split across spaces costs one worker, one deadline and
    one admission, like any other request. Each part: {metta_query, kb_files,
    extra_atoms}. (Measured: the LinAge2 scoped stack and the shared stack run back
    to back in one process in either order.)"""
    return [
        run_query(
            part["metta_query"],
            confidence_threshold=confidence_threshold,
            kb_files=part.get("kb_files"),
            extra_atoms=part.get("extra_atoms"),
        )
        for part in parts
    ]


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
            return PLNRunResult(
                status="error", mode="runtime", error=str(run_exc),
                error_code="runtime_error",
            )

        results: list[PLNAtomResult] = []
        for index, result_group in enumerate(raw):
            for atom in result_group:
                atom_str = str(atom)
                results.append(PLNAtomResult(atom=atom_str, stv=_stv_from_atom(atom_str),
                                             expr_index=index))

        results = _apply_threshold(results, confidence_threshold)
        return PLNRunResult(
            status="ok" if results else "empty",
            results=results,
            query_time_ms=elapsed,
            mode="runtime",
        )
    except Exception as exc:   # noqa: BLE001
        return PLNRunResult(
            status="error", mode="runtime", error=str(exc), error_code="runtime_error",
        )


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
    strategy: str = "linear",
) -> tuple[PLNRunResult, list]:
    """Rank real DrugAge compounds by calibrated, signed effect on lifespan.

    Assembles a QUERY-SCOPED hyperon space = the DrugAge inference stack
    (DRUGAGE_STACK) + only the DrugAge rows matching `compounds` (selected by
    ontology.drugage_selector, capped under the panic threshold), then scores
    each compound against `outcome` (default Mortality — the compound ->
    Lifespan -> Mortality chain keeps the Neg=beneficial convention; see
    docs/etl_inference_wiring.md).

    Two strategies, same arithmetic:

    ``linear`` (default)
        One `(score-candidate &self <C> <outcome>)` expression per compound, all
        in ONE space, then sorted in Python. Cost is linear in the pool size and
        each compound comes back as its OWN atom with its OWN truth value — so
        `confidence_threshold` can finally filter per compound.
    ``metta_sort``
        The original single `(rank-interventions &self (…) <outcome>)` call,
        which sorts inside MeTTa. Kept because it is the reference
        implementation the ordering is defined by; `tests/test_drugage_ranking_
        strategies.py` asserts the two agree. Its MeTTa insertion sort is O(n^2)
        with a very large constant (measured: n=5 1.1 s, n=8 3.0 s, n=10 6.1 s
        against the full build), which is what froze the API for 115 s on a
        35-compound request.

    Returns (PLNRunResult, selected_rows). The result's FIRST atom is the ranked
    tuple (the long-standing contract the formatter and tests parse), followed by
    one atom per scored compound. selected_rows carries the provenance of exactly
    which rows were injected.
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
    cands = sorted({r.compound for r in rows})
    if not cands:
        return PLNRunResult(status="empty", mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub"), rows

    if strategy == "metta_sort":
        query = f"!(rank-interventions &self ({' '.join(cands)}) {outcome})"
        result = run_query(
            query,
            confidence_threshold=confidence_threshold,
            kb_files=DRUGAGE_STACK,
            extra_atoms=slice_text,
        )
        return result, rows

    if strategy != "linear":
        raise ValueError(f"unknown ranking strategy: {strategy!r}")

    # One expression per compound. `_normalize_query` splits them and hyperon
    # returns one result group each, so every compound arrives as its own atom.
    query = "\n".join(f"!(score-candidate &self {c} {outcome})" for c in cands)
    raw = run_query(
        query,
        confidence_threshold=0.0,          # filtered per compound below
        kb_files=DRUGAGE_STACK,
        extra_atoms=slice_text,
    )
    if raw.status == "error":
        return raw, rows

    scored: list[ScoredCompound] = []
    for atom in raw.results:
        scored.extend(parse_scored(atom.atom))
    # `infer` yields one derivation per matching row; best_per_compound keeps
    # that at one, but de-duplicate defensively so a widened slice cannot
    # produce two entries for the same compound.
    by_compound: dict[str, ScoredCompound] = {}
    for sc in scored:
        best = by_compound.get(sc.compound)
        if best is None or sc.score > best.score:
            by_compound[sc.compound] = sc
    kept = [
        sc for sc in by_compound.values()
        if confidence_threshold <= 0 or sc.confidence >= confidence_threshold
    ]
    # Descending by score = most protective first, identical to sort-scored.
    kept.sort(key=lambda sc: (-sc.score, sc.compound))

    if not kept:
        return PLNRunResult(
            status="empty",
            mode=raw.mode,
            query_time_ms=raw.query_time_ms,
        ), rows

    ranked_tuple = "(" + " ".join(sc.atom for sc in kept) + ")"
    results = [PLNAtomResult(ranked_tuple, _stv_from_atom(ranked_tuple))]
    results.extend(PLNAtomResult(sc.atom, {"strength": sc.strength, "confidence": sc.confidence})
                   for sc in kept)
    return PLNRunResult(
        status="ok",
        results=results,
        query_time_ms=raw.query_time_ms,
        mode=raw.mode,
    ), rows


@dataclass
class CellAgeEffect:
    """One `(Effect <gene> CellularSenescence <sign> (stv s c))` the engine lifted."""
    gene_atom: str
    sign: str            # "Pos" = induces senescence | "Neg" = inhibits it
    strength: float
    confidence: float
    atom: str
    #: The CellAge row this link was lifted from, and the qualifiers that row
    #: carries. The LINK itself is identical for every curated row of a gene (the
    #: strength is a single curated prior), so without this a gene with three
    #: rows comes back as three indistinguishable objects and the caller cannot
    #: tell whether that is three findings or one repeated. None when the
    #: engine's results could not be aligned to the injected rows one-to-one.
    row_id: Optional[str] = None
    senescence_type: Optional[str] = None
    cell_context: Optional[str] = None
    pmid: Optional[str] = None

    @property
    def direction(self) -> str:
        return "induces_senescence" if self.sign == "Pos" else "inhibits_senescence"

    def as_dict(self) -> dict:
        return {
            "gene_atom": self.gene_atom,
            "sign": self.sign,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "atom": self.atom,
            "row_id": self.row_id,
            "senescence_type": self.senescence_type,
            "cell_context": self.cell_context,
            "pmid": self.pmid,
            # Named so a caller can never mistake this for the ETL's own
            # `(Causes … (stv 0.82 0.70))` numbers, which are not calibrated.
            "confidence_source": "epistemic_calibration.metta: "
                                 "(evidence-confidence InVitro)",
            "strength_source": "cellage_calibration.metta §1: curated prior "
                               "(CellAge records a direction, not a magnitude)",
        }


_CELLAGE_EFFECT_RE = re.compile(
    r"\(Effect\s+(?P<gene>\S+)\s+CellularSenescence\s+(?P<sign>Pos|Neg)\s+"
    r"\(stv\s+(?P<strength>[-+\d.eE]+)\s+(?P<confidence>[-+\d.eE]+)\)\)"
)


def parse_cellage_effects(atom: str) -> list[CellAgeEffect]:
    """Every `(Effect … CellularSenescence …)` link inside `atom`."""
    out: list[CellAgeEffect] = []
    for m in _CELLAGE_EFFECT_RE.finditer(atom):
        try:
            out.append(CellAgeEffect(
                gene_atom=m.group("gene"),
                sign=m.group("sign"),
                strength=float(m.group("strength")),
                confidence=float(m.group("confidence")),
                atom=m.group(0),
            ))
        except ValueError:   # a malformed number never breaks a lookup
            continue
    return out


def run_cellage_effects(
    genes: list[str],
    *,
    limit: Optional[int] = None,
    source: Optional[Path] = None,
    confidence_threshold: float = 0.0,
) -> tuple[PLNRunResult, list, list[CellAgeEffect]]:
    """Lift the CellAge rows for `genes` into calibrated senescence Effect links.

    Assembles a QUERY-SCOPED hyperon space = CELLAGE_STACK + only the CellAge
    rows naming one of `genes` (selected by ontology.cellage_selector, capped
    under the abort boundary), then evaluates `(cellage-effect &self <row>)` once
    per selected row — one expression per row, so each link comes back as its own
    atom with its own truth value, exactly as the `linear` DrugAge strategy does.

    Per-row rather than per-gene on purpose: a gene with three curated rows (TP53
    has three, one per senescence type) yields three links, and merging them into
    one verdict would report a number no row states.

    Returns (PLNRunResult, selected_rows, parsed_effects). An empty selection is
    reported as `status="empty"` — an absence of CURATED ROWS, never an assertion
    that the gene does not affect senescence.
    """
    from ontology.cellage_selector import MAX_ROWS, build_cellage_slice

    slice_text, rows = build_cellage_slice(
        genes,
        limit=limit if limit is not None else MAX_ROWS,
        source=source,
    )
    if not rows:
        return (
            PLNRunResult(
                status="empty",
                mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
            ),
            rows,
            [],
        )

    query = "\n".join(f"!(cellage-effect &self {r.row_id})" for r in rows)
    raw = run_query(
        query,
        confidence_threshold=confidence_threshold,
        kb_files=CELLAGE_STACK,
        extra_atoms=slice_text,
    )
    effects: list[CellAgeEffect] = []
    for atom in raw.results:
        effects.extend(parse_cellage_effects(atom.atom))

    # Align each link to the row it came from. One expression per row, each
    # yielding exactly one link (every selected row is `liftable`), so the
    # engine's result order is the row order. The equal-length check is the
    # guard: if that assumption ever breaks the links come back WITHOUT row
    # provenance rather than with someone else's.
    if len(effects) == len(rows):
        for effect, row in zip(effects, rows):
            effect.row_id = row.row_id
            effect.senescence_type = row.senescence_type
            effect.cell_context = row.cell_context
            effect.pmid = row.pmid
    return raw, rows, effects
