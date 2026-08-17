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

Kept free of any Gradio import so it is unit-testable without the UI stack.

See docs/etl_inference_wiring.md §8.5.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from config import PLN_RUNTIME_AVAILABLE
from core.pln_runner import PLNAtomResult, PLNRunResult, run_drugage_ranking
from ontology.drugage_selector import BUILD_DRUGAGE, _norm

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
    )


def _provenance_line(row) -> str:
    """One human-readable audit-trail bullet for a ranked compound's source row."""
    bits: list[str] = [f"{row.compound} ← {row.pmid or 'PMID unreported'}"]
    detail: list[str] = []
    if row.species:
        detail.append(row.species.replace("_", " "))
    if row.avg_change is not None:
        detail.append(f"{row.avg_change:+.0f}% avg lifespan")
    if row.is_itp:
        detail.append("ITP")
    if row.significance:
        detail.append(row.significance)
    if detail:
        bits.append(" · ".join(detail))
    return " — ".join(bits)


def route_drugage_ranking(
    compounds: list[str],
    *,
    confidence_threshold: float = 0.0,
    source: Optional[Path] = None,
) -> PLNRunResult:
    """Run the scoped DrugAge ranking for `compounds` and package it for display.

    Dispatches to `run_drugage_ranking` (SCOPED DrugAge stack + a filtered row
    slice from `source`, default `build/drugage_etl.metta`) and returns a
    `PLNRunResult` whose atoms are:

      1. the ranked, signed, uncertainty-quantified `(scored …)` tuple, followed by
      2. one provenance bullet per ranked compound (the backing PMID + evidence),
      3. an "omitted" note for any requested compound with no matching DrugAge row
         (omitted rather than mis-ranked — docs/etl_inference_wiring.md §5).

    Degrades gracefully (a clear message, never a crash) when `build/` is missing
    or the engine raises.
    """
    src = source or BUILD_DRUGAGE
    if not src.exists():
        return _missing_build_result()

    if not compounds:
        return PLNRunResult(
            status="empty",
            mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
            error=None,
        )

    try:
        result, rows = run_drugage_ranking(
            compounds,
            source=src,
            confidence_threshold=confidence_threshold,
        )
    except Exception as exc:  # noqa: BLE001 — a DrugAge query must never crash chat
        return PLNRunResult(
            status="error",
            mode="runtime" if PLN_RUNTIME_AVAILABLE else "stub",
            error=f"DrugAge ranking failed: {exc}",
        )

    if result.status == "error":
        return result

    # Enrich: keep the ranking atom(s) first (the test/format contract), then the
    # provenance audit trail, then a note for compounds nothing matched.
    enriched: list[PLNAtomResult] = list(result.results)
    for row in sorted(rows, key=lambda r: r.compound):
        enriched.append(PLNAtomResult(_provenance_line(row)))

    matched = {_norm(r.compound) for r in rows}
    omitted = [c for c in compounds if _norm(c) not in matched]
    if omitted:
        enriched.append(PLNAtomResult(
            f"Omitted (no DrugAge lifespan rows matched): {', '.join(omitted)}"
        ))

    return PLNRunResult(
        status="ok" if enriched else "empty",
        results=enriched,
        query_time_ms=result.query_time_ms,
        mode=result.mode,
    )
