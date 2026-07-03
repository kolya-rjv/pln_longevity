"""Tests for the chat-app routing of the DrugAge lifespan/mortality ranking.

The engine (`run_drugage_ranking`) is covered by test_drugage_calibration.py.
This module covers the GLUE that lets the chat app reach it:

  * `parse_drugage_query` recognises the dedicated `(rank-drugage-lifespan …)`
    form the translator emits and leaves every other query for the generic path;
  * `route_drugage_ranking` dispatches to `run_drugage_ranking` (the scoped
    space — NOT the generic _ALL_KB_PATHS), pushes an ITP-negative compound to
    ~0, omits an absent compound instead of mis-ranking it, and degrades
    gracefully when the DrugAge build/ output is missing.

Driven from the committed fixture (tests/fixtures/drugage_real_rows.metta) so the
assertions are deterministic without regenerating the full ETL.

Skipped automatically if `hyperon` is not installed.  Run:  pytest tests/ -v
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

hyperon = pytest.importorskip("hyperon")

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_ROWS = FIXTURES / "drugage_real_rows.metta"

os.environ["PLN_RUNTIME_AVAILABLE"] = "true"
sys.path.insert(0, str(REPO / "pln_chat"))

from core.drugage_router import (  # noqa: E402
    DRUGAGE_RANK_SYMBOL,
    parse_drugage_query,
    route_drugage_ranking,
)

_SCORED = re.compile(r"\(scored\s+(\S+)\s+([-\d.eE]+)")


def _scored(result) -> list[tuple[str, float]]:
    """Parse the ranked (scored <name> <score> …) tuples out of a routed result."""
    assert result.results, f"no result atoms: {result.error}"
    return [(m.group(1), float(m.group(2)))
            for m in _SCORED.finditer(result.results[0].atom)]


# ── parse: recognise the dedicated form, leave everything else alone ────────────

def test_parse_recognizes_the_dedicated_form():
    assert parse_drugage_query(
        "(rank-drugage-lifespan (Rapamycin Resveratrol Metformin))"
    ) == ["Rapamycin", "Resveratrol", "Metformin"]


def test_parse_returns_none_for_generic_queries():
    # These must fall through to the generic run_query path (None, not []).
    assert parse_drugage_query(
        "(rank-interventions &self (Fisetin Spermidine) CoronaryHeartDisease)") is None
    assert parse_drugage_query(
        "(match &self (UsesIntervention $e Rapamycin) $e)") is None
    assert parse_drugage_query("") is None


def test_parse_preserves_casing_for_the_loose_selector_and_tolerates_empty():
    # Casing is preserved verbatim; the selector matches case-/separator-insensitively.
    assert parse_drugage_query(f"({DRUGAGE_RANK_SYMBOL} (rapamycin))") == ["rapamycin"]
    # Symbol present but no parseable compound list -> empty (handled, not a crash).
    assert parse_drugage_query(f"({DRUGAGE_RANK_SYMBOL} ())") == []


# ── the acceptance test: NL routes to run_drugage_ranking, ITP-negative -> ~0 ────

def test_nl_query_routes_to_run_drugage_ranking(monkeypatch):
    import core.drugage_router as router

    seen: dict = {}
    real = router.run_drugage_ranking

    def spy(compounds, **kw):
        seen["compounds"] = compounds          # prove the dispatch happened
        return real(compounds, **kw)

    monkeypatch.setattr(router, "run_drugage_ranking", spy)

    metta_query = "(rank-drugage-lifespan (Rapamycin Resveratrol Metformin))"
    compounds = parse_drugage_query(metta_query)
    result = router.route_drugage_ranking(compounds, source=REAL_ROWS)

    assert seen["compounds"] == ["Rapamycin", "Resveratrol", "Metformin"]
    assert result.status == "ok", result.error

    order = _scored(result)
    names = [n for n, _ in order]
    scores = dict(order)
    assert names[0] == "Rapamycin"                       # replicated ITP positive on top
    assert scores["Resveratrol"] == pytest.approx(0.0, abs=1e-9)  # ITP-negative sinks to ~0
    assert scores["Metformin"] == pytest.approx(0.0, abs=1e-9)
    assert scores["Rapamycin"] > scores["Resveratrol"]   # same tier — strength decides


# ── absent compound is omitted, not mis-ranked ─────────────────────────────────

def test_absent_compound_is_omitted_not_misranked():
    compounds = parse_drugage_query("(rank-drugage-lifespan (Rapamycin Dasatinib))")
    result = route_drugage_ranking(compounds, source=REAL_ROWS)
    assert result.status == "ok", result.error

    names = [n for n, _ in _scored(result)]
    assert "Rapamycin" in names
    assert "Dasatinib" not in names                      # no row -> not ranked

    joined = " ".join(r.atom for r in result.results)
    assert "Omitted" in joined and "Dasatinib" in joined  # surfaced, not silently dropped


# ── provenance: the backing PMID rides along with the ranking ──────────────────

def test_ranking_carries_pmid_provenance():
    compounds = parse_drugage_query("(rank-drugage-lifespan (Rapamycin))")
    result = route_drugage_ranking(compounds, source=REAL_ROWS)
    assert result.status == "ok", result.error
    joined = " ".join(r.atom for r in result.results)
    assert "PMID_24341993" in joined                     # Rapamycin's ITP study (fixture)


# ── graceful degradation when the DrugAge ETL output is missing ────────────────

def test_missing_build_degrades_gracefully(tmp_path):
    result = route_drugage_ranking(["Rapamycin"], source=tmp_path / "nope.metta")
    assert result.status == "error"
    assert "scripts/run_etl.sh" in (result.error or "")   # tells the user how to fix it
