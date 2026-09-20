"""Whole-KB discovery, and proof that the Python scorer is the MeTTa scorer.

"Which drugs extend lifespan in mice with the strongest evidence?" is, per the
2026-09-18 evaluation, "the one people ask first and it is unanswerable now":
the translator invented a four-compound pool and ranked only those, and a
generic MeTTa match returned nothing because DrugAge rows are excluded from the
runtime space.

Answering it needs the arithmetic in Python (1,043 compounds is 1,043 MeTTa
calls, and loading the rows to rank them in one space aborts hyperon), which
makes `test_python_scoring_agrees_with_the_metta_engine` the load-bearing test
in this file: it is what stops the second implementation from drifting.

Run from the repository root:
    pytest tests/test_drugage_discovery.py -q
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import api as api_module  # noqa: E402
import core.executor as executor_module  # noqa: E402
from core.drugage_router import drugage_top  # noqa: E402
from core.pln_runner import parse_scored, run_drugage_ranking  # noqa: E402
from ontology.drugage_scoring import load_knobs, score_row, score_rows  # noqa: E402
from ontology.drugage_selector import load_rows  # noqa: E402
from ontology.hallmarks import hallmark_index  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_ROWS = FIXTURES / "drugage_real_rows.metta"
SAMPLE = REPO / "drugage_etl_short.metta"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)


def _get(path: str):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(path)

    return asyncio.run(send())


# ── the second implementation must equal the first ───────────────────────────

@pytest.mark.slow
def test_python_scoring_agrees_with_the_metta_engine():
    """Bit-for-bit, on every compound of the fixture. Drift fails here."""
    pytest.importorskip("hyperon")
    rows = load_rows(REAL_ROWS)
    compounds = sorted({r.compound for r in rows})

    engine, selected = run_drugage_ranking(compounds, source=REAL_ROWS, strategy="linear")
    from_engine = {s.compound: (s.score, s.strength, s.confidence, s.sign)
                   for s in parse_scored(engine.results[0].atom)}

    from_python = {}
    for scored in score_rows(selected):
        from_python[scored.row.compound] = (
            scored.score, scored.strength, scored.confidence, scored.sign
        )

    assert set(from_python) == set(from_engine)
    for compound, values in from_engine.items():
        for got, want in zip(from_python[compound], values):
            if isinstance(want, float):
                assert got == pytest.approx(want, abs=1e-12), compound
            else:
                assert got == want, compound


def test_the_knobs_are_read_from_the_metta_sources_not_hard_coded():
    knobs = load_knobs(force=True)
    calibration = (REPO / "drugage_calibration.metta").read_text(encoding="utf-8")
    assert f"(= (lifespan-halfsat) {knobs.halfsat})" in calibration
    assert knobs.sig_gates == {"Significant": 1.0, "Unreported": 0.6, "NotSignificant": 0.4}
    assert knobs.evidence_confidence["ITP_Positive"] == 0.9
    assert knobs.clade_categories["Vertebrate"] == "AnimalStudies_Single"
    assert knobs.chain_discount == 0.9


def test_a_row_without_a_change_percent_has_no_score():
    rows = load_rows(SAMPLE)
    scoreless = [r for r in rows if r.avg_change is None]
    assert scoreless, "the sample should contain the known scoreless row"
    assert score_row(scoreless[0]) is None


def test_every_drugage_species_has_a_taxonomy_entry():
    """A mouse row must not be scored at the invertebrate tier."""
    knobs = load_knobs(force=True)
    rows = load_rows(SAMPLE)
    missing = {r.species for r in rows if r.species and r.species not in knobs.species_clade}
    assert not missing, f"species with no clade: {sorted(missing)}"


# ── the ranking itself ───────────────────────────────────────────────────────

def test_top_n_ranks_the_whole_build_not_a_caller_supplied_pool():
    top = drugage_top(n=5, source=SAMPLE)
    assert top.total_compounds > 5
    assert len(top.entries) == 5
    scores = [e.score for e in top.entries]
    assert scores == sorted(scores, reverse=True)
    assert all(e.protective for e in top.entries)


def test_filters_narrow_the_evidence_rather_than_the_ranking():
    everything = drugage_top(n=500, source=SAMPLE)
    flies = drugage_top(n=500, species="Drosophila_melanogaster", source=SAMPLE)
    assert 0 < flies.total_compounds < everything.total_compounds
    assert all(e.row.species == "Drosophila_melanogaster" for e in flies.entries)

    strong = drugage_top(n=500, min_confidence=0.4, source=SAMPLE)
    assert all(e.confidence >= 0.4 for e in strong.entries)

    significant = drugage_top(n=500, significant_only=True, source=SAMPLE)
    assert all(e.row.significance == "Significant" for e in significant.entries)


def test_the_harmful_end_is_ranked_most_harmful_first():
    harmful = drugage_top(n=5, direction="harmful", source=SAMPLE)
    assert harmful.entries
    assert all(not e.protective for e in harmful.entries)
    scores = [e.score for e in harmful.entries]
    assert scores == sorted(scores)          # most negative first


def test_top_n_reports_its_source_and_what_it_could_not_score():
    top = drugage_top(n=3, source=SAMPLE)
    assert top.source.endswith("drugage_etl_short.metta")
    assert top.unscorable_rows >= 1
    assert top.scored_rows + top.unscorable_rows == top.total_rows


# ── hallmark lookups, both directions ────────────────────────────────────────

def test_interventions_for_a_hallmark_come_back_with_provenance():
    index = hallmark_index(api_module._runtime_kb_paths())
    records = index.for_hallmark("CellularSenescence")
    names = {r.intervention for r in records}
    assert {"Fisetin", "DasatinibPlusQuercetin"} <= names
    for record in records:
        assert record.species_model and record.outcome_text
        assert record.publication


def test_hallmarks_for_an_intervention_is_the_same_relation_backwards():
    index = hallmark_index(api_module._runtime_kb_paths())
    assert [r.hallmark for r in index.for_intervention("Fisetin")] == ["CellularSenescence"]


def test_an_intervention_with_no_record_says_so_explicitly():
    body = _get("/interventions?intervention=Rapamycin").json()
    assert body["evidence"] == []
    assert "no curated hallmark-evidence record" in body["note"]
    assert "Fisetin" in body["covered_interventions"]


def test_an_unknown_hallmark_lists_the_ones_that_exist():
    body = _get("/interventions?hallmark=NotAHallmark").json()
    assert body["evidence"] == []
    assert "CellularSenescence" in body["note"]


def test_hallmarks_endpoint_lists_anchors_and_interventions():
    body = _get("/hallmarks").json()
    by_name = {h["name"]: h for h in body["hallmarks"]}
    assert "CellularSenescence" in by_name
    senescence = by_name["CellularSenescence"]
    assert "SASP" in senescence["components"]
    assert senescence["intervention_count"] >= 2
    assert body["evidence_records"] >= 14


# ── over HTTP ────────────────────────────────────────────────────────────────

def test_top_endpoint_returns_ranked_rows_with_provenance():
    body = _get("/drugage/top?n=3").json()
    assert len(body["entries"]) <= 3
    for entry in body["entries"]:
        assert entry["compound"] and entry["row_id"]
        assert entry["direction"] in {"protective", "harmful"}
        assert 0.0 <= entry["confidence"] <= 1.0
    assert body["semantics"]["sign"]["Neg"].startswith("protective")
    assert "source" in body


def test_top_endpoint_rejects_an_absurd_n():
    assert _get("/drugage/top?n=0").status_code == 422
    assert _get("/drugage/top?n=100000").status_code == 422


def test_interventions_endpoint_rejects_both_filters_at_once():
    response = _get("/interventions?hallmark=CellularSenescence&intervention=Fisetin")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_filter"
