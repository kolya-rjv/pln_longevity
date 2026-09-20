"""Route a `rank-drugage-lifespan` intent to the scoped DrugAge ranking engine.

Why this module exists
----------------------
The generic chat path is `translate(NL) -> metta_query -> run_query(…,
kb_files=_ALL_KB_PATHS)`. That path is WRONG for ranking real compounds by
lifespan/mortality effect:

  * `_ALL_KB_PATHS` does NOT contain the DrugAge `build/` rows (they are excluded
    for being too big — hyperon 0.2.10 panics on a multi-thousand-atom space), so
    a generic `rank-interventions … Mortality` sees no DrugAge evidence; and
  * it DOES contain the grim_age / hallmarks / mechanistic_bridges curated Effect
    links the DrugAge stack deliberately excludes (name collisions + panic risk).

`core.pln_runner.run_drugage_ranking` already does the right thing — a SCOPED
space (DRUGAGE_STACK + a small filtered DrugAge slice) ranked by protective
effect on Mortality. This module is the thin glue that lets the chat app REACH
it: the LLM translator emits a dedicated `(rank-drugage-lifespan (C1 C2 …))`
form, `parse_drugage_query` recognises it, and `route_drugage_ranking` dispatches
to `run_drugage_ranking` and packages the result (ranking + provenance) as a
`PLNRunResult` the existing `format_bot_response` renders unchanged.

Caller spellings are resolved to DrugAge symbols FIRST, by the shared
`ontology.compound_names.CompoundResolver`, so `sirolimus`, `NMN`, `EGCG`,
`NAC` and `17-alpha-estradiol` reach the rows they name instead of silently
dropping out of the ranking (the single most misleading behaviour the
2026-09-18 API evaluation found).

Kept free of any Gradio import so it is unit-testable without the UI stack.

See docs/etl_inference_wiring.md §8.5.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import PLN_RUNTIME_AVAILABLE
from core.pln_runner import (
    PLNAtomResult,
    PLNRunResult,
    ScoredCompound,
    parse_scored,
    run_drugage_ranking,
)
from ontology.compound_names import Resolution
from ontology.drugage_scoring import RowScore, load_knobs, score_rows
from ontology.drugage_selector import (
    BUILD_DRUGAGE,
    REPRESENTATIVE_POLICY,
    SAMPLE_DRUGAGE,
    DrugAgeRow,
    _norm,
    build_resolver,
    load_rows,
    select_rows,
)

# The dedicated NL-facing symbol the translator emits for this intent. A DISTINCT
# symbol (not the generic `rank-interventions`) is what lets the app route
# unambiguously to the scoped DrugAge engine — see the module docstring.
DRUGAGE_RANK_SYMBOL = "rank-drugage-lifespan"

# `(rank-drugage-lifespan (C1 C2 …))` — capture the FIRST parenthesised group
# after the symbol as the compound list. `[^()]*` deliberately stops at the first
# close-paren so a stray trailing outcome token (if the LLM adds one) is ignored;
# the outcome is fixed to Mortality by run_drugage_ranking (the sign convention).
_FORM_RE = re.compile(DRUGAGE_RANK_SYMBOL + r"\s*\(\s*([^()]*?)\s*\)")


def parse_drugage_query(metta_query: str) -> Optional[list[str]]:
    """Recognise a DrugAge-lifespan ranking request in a translated MeTTa query.

    Returns
    -------
    None
        The query is NOT a `rank-drugage-lifespan` form — the caller should fall
        through to the generic `run_query` path.
    list[str]
        The parsed compound tokens (possibly empty if the symbol appears with no
        parseable list). Tokens are returned verbatim; the selector matches them
        case-/separator-insensitively, so `rapamycin` still hits `Rapamycin`.
    """
    if DRUGAGE_RANK_SYMBOL not in metta_query:
        return None
    m = _FORM_RE.search(metta_query)
    if not m:
        return []
    return m.group(1).split()


def resolve_compounds(
    compounds: list[str],
    *,
    source: Optional[Path] = None,
) -> list[Resolution]:
    """Resolve caller-typed compound names against the DrugAge vocabulary.

    The SAME call the ranking makes, exposed so the HTTP layer can report what
    each requested name resolved to without repeating the logic (the resolver
    is cached per source file, so calling it twice costs a dict lookup).
    """
    resolver = build_resolver(source or BUILD_DRUGAGE)
    return resolver.resolve_all(compounds)


def _missing_build_result() -> PLNRunResult:
    """Graceful degradation when the DrugAge ETL output has not been generated."""
    return PLNRunResult(
        status="error",
        mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
        error=(
            "DrugAge data not found at `build/drugage_etl.metta`. This ranking "
            "reads the regenerated ETL rows — run `bash scripts/run_etl.sh` to "
            "generate them, then ask again."
        ),
        error_code="drugage_build_missing",
    )


def _provenance_line(row) -> str:
    """One human-readable audit-trail bullet for a ranked compound's source row."""
    bits: list[str] = [f"{row.compound} ← {row.pmid or 'PMID unreported'}"]
    detail: list[str] = []
    if row.species:
        detail.append(row.species.replace("_", " "))
    if row.sex and row.sex not in ("Unknown",):
        detail.append(row.sex)
    if row.avg_change is not None:
        detail.append(f"{row.avg_change:+.4g}% avg lifespan")
    else:
        detail.append("no avg lifespan change reported")
    if row.is_itp:
        detail.append("ITP")
    if row.significance:
        detail.append(row.significance)
    if detail:
        bits.append(" · ".join(detail))
    return " — ".join(bits)


# ── What a DrugAge score MEANS ───────────────────────────────────────────────
# Returned verbatim with every ranking. The 2026-09-18 evaluation could not tell
# from the response that `signed Neg` is the GOOD direction, where the strength
# transform came from, or how one row per compound was chosen — so it read a
# protective ranking as if the sign were arbitrary. Every number below is read
# off the .metta knobs, not restated from memory: drugage_calibration.metta
# §1/§3/§5/§6, epistemic_calibration.metta §1, pln_deduction.metta §5.
SCORE_SEMANTICS: dict = {
    "score": "strength x confidence, signed so that HIGHER IS BETTER: a "
             "protective effect scores positive, a harmful one negative.",
    "sign": {
        "Neg": "protective — the compound LOWERS mortality (it extended lifespan)",
        "Pos": "harmful — the compound RAISES mortality (it shortened lifespan)",
    },
    "sign_convention": "The lift is on the Lifespan axis (extending lifespan is "
                       "Pos), then chained through the curated "
                       "(Effect Lifespan Mortality Neg) adapter. Pos x Neg = Neg, "
                       "so a life-extender reads as Neg = protective on the "
                       "mortality axis every other ranking in this KB uses.",
    "strength": "|AvgLifespanChangePercent| / (|AvgLifespanChangePercent| + 20). "
                "Saturating, so +20% reads 0.50 and +80% reads 0.80; the "
                "half-saturation constant is (lifespan-halfsat) in "
                "drugage_calibration.metta.",
    "confidence": "min(evidence tier, significance gate) x 0.9. The 0.9 is the "
                  "per-hop chain discount for the Lifespan -> Mortality step, so "
                  "the confidence you see is always 0.9 x the row's tier.",
    "confidence_tiers": {
        "0.81": "ITP row (replicated NIA Interventions Testing Program mouse "
                "study) — 0.90 x 0.9. Applies to a negative ITP result too: a "
                "well-run null is high-confidence evidence of ~no effect.",
        "0.45": "non-ITP vertebrate (mouse, rat, fish), reported Significant — "
                "0.50 x 0.9",
        "0.315": "invertebrate (worm, fly) or an untaxonomised species — "
                 "0.35 x 0.9",
        "0.18": "fungi or protozoa (yeast) — 0.20 x 0.9",
        "note": "A non-ITP row is additionally CAPPED by its significance: "
                "Significant 1.0, Unreported 0.6, NotSignificant 0.4.",
    },
    "representative_row_policy": list(REPRESENTATIVE_POLICY),
    "zero_score": "A score of exactly 0.0 is a reported null, not a missing "
                  "value — e.g. metformin and resveratrol at confidence 0.81 are "
                  "ITP negatives. A compound with no reported change percent is "
                  "reported under `unscorable`, never as 0.0.",
}


@dataclass
class DrugAgeRanking:
    """Everything POST /drugage/rank needs, already structured.

    `result` keeps the atom shape the chat formatter and the existing tests
    expect; the other fields are the same information without string parsing.
    """
    result: PLNRunResult
    resolutions: list[Resolution] = field(default_factory=list)
    ranked: list[ScoredCompound] = field(default_factory=list)
    rows: list[DrugAgeRow] = field(default_factory=list)
    unscorable: list[str] = field(default_factory=list)
    filtered_out: list[ScoredCompound] = field(default_factory=list)
    source: str = ""


def _row_out(row: DrugAgeRow) -> dict:
    """One DrugAge row as data — sex and species included.

    The evaluation reported astaxanthin as "+3% avg lifespan, not significant"
    while the cited paper headlines +12% (p = 0.003). Both are real rows of the
    same ITP study, one per sex; collapsing to a single representative hid that.
    Every matching row is now returned so the collapse is auditable.
    """
    return {
        "row_id": row.row_id,
        "compound": row.compound,
        "species": row.species,
        "sex": row.sex,
        "is_itp": row.is_itp,
        "significance": row.significance,
        "avg_lifespan_change_percent": row.avg_change,
        "pmid": row.pmid,
        "scorable": row.scorable,
    }


def rank_drugage(
    compounds: list[str],
    *,
    confidence_threshold: float = 0.0,
    source: Optional[Path] = None,
    strategy: str = "linear",
    include_all_rows: bool = True,
) -> DrugAgeRanking:
    """The structured ranking: resolve -> select -> score -> report.

    `route_drugage_ranking` is the thin wrapper that returns just the
    `PLNRunResult` for the chat path.
    """
    src = source or BUILD_DRUGAGE
    if not src.exists():
        return DrugAgeRanking(result=_missing_build_result(), source=str(src))

    if not compounds:
        return DrugAgeRanking(
            result=PLNRunResult(
                status="empty",
                mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
            ),
            source=str(src),
        )

    # Resolve caller spellings to DrugAge symbols BEFORE any row selection, so
    # the ranking, the provenance bullets and the omitted note all speak about
    # the same compounds (see ontology/compound_names.py for the ladder).
    resolutions = resolve_compounds(compounds, source=src)
    canonical = [r.matched for r in resolutions if r.matched is not None]

    if not canonical:
        return DrugAgeRanking(
            result=PLNRunResult(
                status="empty",
                mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
                results=[
                    PLNAtomResult(note)
                    for note in (r.warning for r in resolutions) if note
                ],
            ),
            resolutions=resolutions,
            source=str(src),
        )

    try:
        # Score with NO threshold so the response can say what was filtered out
        # rather than silently shortening the list.
        result, rows = run_drugage_ranking(
            canonical,
            source=src,
            confidence_threshold=0.0,
            strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001 — a DrugAge query must never crash chat
        return DrugAgeRanking(
            result=PLNRunResult(
                status="error",
                mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
                error=f"DrugAge ranking failed: {exc}",
            ),
            resolutions=resolutions,
            source=str(src),
        )

    if result.status == "error":
        return DrugAgeRanking(result=result, resolutions=resolutions, source=str(src))

    scored = parse_scored(result.results[0].atom) if result.results else []
    kept = [s for s in scored
            if confidence_threshold <= 0 or s.confidence >= confidence_threshold]
    dropped = [s for s in scored if s not in kept]

    # A compound whose representative row carries no AvgLifespanChangePercent
    # cannot be lifted into an Effect link, so it produces no score at all. That
    # used to look identical to "ranked last".
    ranked_names = {s.compound for s in scored}
    unscorable = sorted({r.compound for r in rows if r.compound not in ranked_names})

    all_rows = rows
    if include_all_rows:
        selected = {r.compound for r in rows}
        all_rows = select_rows(load_rows(src), compounds=selected, limit=10_000)

    return DrugAgeRanking(
        result=_render(result, kept, dropped, rows, unscorable, resolutions,
                       confidence_threshold),
        resolutions=resolutions,
        ranked=kept,
        rows=all_rows,
        unscorable=unscorable,
        filtered_out=dropped,
        source=str(src),
    )


def _render(
    result: PLNRunResult,
    kept: list[ScoredCompound],
    dropped: list[ScoredCompound],
    rows: list[DrugAgeRow],
    unscorable: list[str],
    resolutions: list[Resolution],
    confidence_threshold: float,
) -> PLNRunResult:
    """Re-assemble the atom view (ranking tuple first) after filtering."""
    atoms: list[PLNAtomResult] = []
    if kept:
        tuple_atom = "(" + " ".join(s.atom for s in kept) + ")"
        atoms.append(PLNAtomResult(tuple_atom))
        atoms.extend(
            PLNAtomResult(s.atom, {"strength": s.strength, "confidence": s.confidence})
            for s in kept
        )
    for row in sorted(rows, key=lambda r: r.compound):
        atoms.append(PLNAtomResult(_provenance_line(row)))

    if unscorable:
        atoms.append(PLNAtomResult(
            "Unscorable (a matching DrugAge row exists but reports no average "
            f"lifespan change, so no effect can be lifted): {', '.join(unscorable)}"
        ))
    if dropped:
        atoms.append(PLNAtomResult(
            f"Filtered out below confidence_threshold={confidence_threshold}: "
            + ", ".join(f"{s.compound} ({s.confidence:.3g})" for s in dropped)
        ))

    matched = {_norm(r.compound) for r in rows}
    omitted: list[str] = []
    for res in resolutions:
        if res.matched is None or _norm(res.matched) not in matched:
            omitted.append(res.query)
    if omitted:
        atoms.append(PLNAtomResult(
            f"Omitted (no DrugAge lifespan rows matched): {', '.join(omitted)}"
        ))
    for res in resolutions:
        note = res.warning
        if note:
            atoms.append(PLNAtomResult(note))

    return PLNRunResult(
        status="ok" if atoms else "empty",
        results=atoms,
        query_time_ms=result.query_time_ms,
        mode=result.mode,
    )


def route_drugage_ranking(
    compounds: list[str],
    *,
    confidence_threshold: float = 0.0,
    source: Optional[Path] = None,
    strategy: str = "linear",
) -> PLNRunResult:
    """Run the scoped DrugAge ranking for `compounds` and package it for display.

    The atom view of `rank_drugage`, for the chat path (`/query`, `/metta/run`)
    whose formatter renders `PLNRunResult` atoms. Atoms are, in order:

      1. the ranked, signed, uncertainty-quantified `(scored …)` tuple, then one
         atom per ranked compound (so a confidence filter can act per compound),
      2. one provenance bullet per selected row (the backing PMID + evidence),
      3. an "unscorable" note for a compound whose row reports no lifespan
         change, and a note for anything the confidence threshold removed,
      4. an "omitted" note for any requested compound with no matching DrugAge
         row (omitted rather than mis-ranked — docs/etl_inference_wiring.md §5),
      5. one note per requested name that did NOT match literally — a synonym
         (`sirolimus` -> `Rapamycin`), an ETL symbol artefact, an accepted typo
         correction, an ambiguity or a miss with suggestions.

    Degrades gracefully (a clear message, never a crash) when `build/` is missing
    or the engine raises.
    """
    return rank_drugage(
        compounds,
        confidence_threshold=confidence_threshold,
        source=source,
        strategy=strategy,
        include_all_rows=False,
    ).result


# ── Whole-KB discovery (no LLM, no MeTTa) ────────────────────────────────────

@dataclass
class DrugAgeTop:
    """A ranking over the WHOLE DrugAge build, not a caller-supplied pool."""
    entries: list[RowScore] = field(default_factory=list)
    total_compounds: int = 0
    total_rows: int = 0
    scored_rows: int = 0
    unscorable_rows: int = 0
    source: str = ""
    filters: dict = field(default_factory=dict)


def _representative(scores: list[RowScore]) -> RowScore:
    """The same representative-row policy the pooled ranking uses (§ selector)."""
    best = max(s.row.evidence_rank for s in scores)
    tier = [s for s in scores if s.row.evidence_rank == best]
    top_sig = max(
        _SIG_RANK.get(s.row.significance or "Unreported", 1) for s in tier
    )
    tier = [
        s for s in tier
        if _SIG_RANK.get(s.row.significance or "Unreported", 1) == top_sig
    ]
    tier.sort(key=lambda s: (s.row.avg_change if s.row.avg_change is not None else 0.0,
                             s.row.row_id))
    return tier[(len(tier) - 1) // 2]


_SIG_RANK = {"Significant": 2, "Unreported": 1, "NotSignificant": 0}


def drugage_top(
    *,
    n: int = 20,
    species: Optional[str] = None,
    clade: Optional[str] = None,
    min_confidence: float = 0.0,
    itp_only: bool = False,
    significant_only: bool = False,
    direction: str = "protective",
    source: Optional[Path] = None,
) -> DrugAgeTop:
    """Rank the whole DrugAge build by calibrated, signed effect on mortality.

    This is the "strongest evidence overall" question the evaluation called the
    one people ask first and found unanswerable: the translator invented a
    four-compound pool and ranked only those, and a generic MeTTa match returned
    nothing because DrugAge rows are excluded from the runtime space.

    It cannot go through the engine. Ranking 1,043 compounds would be 1,043
    MeTTa calls, and loading the rows to do it in one space aborts the
    interpreter (hyperon 0.2.10 panics on a variable-slot match past a few
    hundred rows). The arithmetic is therefore applied in Python, using the
    calibration layer's OWN constants — see ontology/drugage_scoring.py, and the
    equality test against the engine in tests/test_drugage_discovery.py.
    """
    src = _resolve_top_source(source)
    rows = load_rows(src)
    knobs = load_knobs()

    if species:
        rows = [r for r in rows if r.species and _norm(r.species) == _norm(species)]
    if clade:
        rows = [
            r for r in rows
            if knobs.species_clade.get(r.species or "", "").lower() == clade.lower()
        ]
    if itp_only:
        rows = [r for r in rows if r.is_itp]
    if significant_only:
        rows = [r for r in rows if r.significance == "Significant"]

    scored = score_rows(rows, knobs=knobs)
    unscorable = len(rows) - len(scored)

    by_compound: dict[str, list[RowScore]] = {}
    for s in scored:
        by_compound.setdefault(s.row.compound, []).append(s)
    representatives = [_representative(group) for group in by_compound.values()]

    if min_confidence > 0:
        representatives = [s for s in representatives if s.confidence >= min_confidence]
    if direction == "protective":
        representatives = [s for s in representatives if s.protective]
    elif direction == "harmful":
        representatives = [s for s in representatives if not s.protective]

    # Most protective first; for `direction=harmful` the interesting end is the
    # other one, so sort ascending there rather than showing the least harmful.
    if direction == "harmful":
        representatives.sort(key=lambda s: (s.score, s.row.compound))
    else:
        representatives.sort(key=lambda s: (-s.score, s.row.compound))
    return DrugAgeTop(
        entries=representatives[: max(0, n)],
        total_compounds=len(by_compound),
        total_rows=len(rows),
        scored_rows=len(scored),
        unscorable_rows=unscorable,
        source=str(src),
        filters={
            "species": species,
            "clade": clade,
            "min_confidence": min_confidence,
            "itp_only": itp_only,
            "significant_only": significant_only,
            "direction": direction,
            "n": n,
        },
    )


def _resolve_top_source(source: Optional[Path]) -> Path:
    """The build if it exists, else the committed sample — and say which."""
    if source is not None and source.exists():
        return source
    return BUILD_DRUGAGE if BUILD_DRUGAGE.exists() else SAMPLE_DRUGAGE
