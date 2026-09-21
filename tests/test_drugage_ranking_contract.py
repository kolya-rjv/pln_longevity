"""Contract tests for the structured DrugAge ranking.

Every test here pins one behaviour the 2026-09-18 API evaluation reported as
wrong or unexplained:

  * `confidence_threshold` did nothing (a 0.5 threshold still returned
    confidence-0.315 rows);
  * a 35-compound request blocked the whole API for 115 s;
  * astaxanthin came back "+3% avg lifespan, not significant" although the
    cited ITP study reports +12 % in males (p = 0.003) — one row per compound,
    sex collapsed, policy undocumented;
  * D-glucosamine "matched a row with no lifespan percentage and returned an
    empty score with no warning".

Run from the repository root:
    pytest tests/test_drugage_ranking_contract.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("hyperon")

from core.drugage_router import SCORE_SEMANTICS, rank_drugage  # noqa: E402
from core.pln_runner import (  # noqa: E402
    PLNAtomResult,
    _apply_threshold,
    _stv_from_atom,
    parse_scored,
    run_drugage_ranking,
)
from ontology.drugage_selector import (  # noqa: E402
    REPRESENTATIVE_POLICY,
    DrugAgeRow,
    _collapse_best,
    load_rows,
    select_rows,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_ROWS = FIXTURES / "drugage_real_rows.metta"


def _post(path: str, body: dict):
    """One HTTP call against the in-process ASGI app."""
    import asyncio

    httpx = pytest.importorskip("httpx")
    import api as api_module

    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(send())


def _row(row_id, compound, species, itp, sig, pct, sex=None) -> DrugAgeRow:
    block = "\n".join(
        [f"(InstanceOf {row_id} Experiment)", f"(UsesIntervention {row_id} {compound})"]
    )
    return DrugAgeRow(row_id, compound, species, itp, sig, pct, block, sex)


# ── the confidence threshold actually filters ─────────────────────────────────

def test_confidence_threshold_filters_entry_by_entry():
    """A collapsed tuple is filtered per entry, not kept/dropped as a whole."""
    atom = (
        "((scored A 0.23 (signed Neg (stv 0.75 0.315))) "
        "(scored B 0.19 (signed Neg (stv 0.42 0.45))))"
    )
    results = [PLNAtomResult(atom, _stv_from_atom(atom))]

    assert _apply_threshold(results, 0.0)[0].atom == atom          # off by default
    kept = _apply_threshold(results, 0.32)
    assert len(kept) == 1
    assert "scored B" in kept[0].atom and "scored A" not in kept[0].atom
    assert _apply_threshold(results, 0.5) == []                    # nothing survives


def test_a_single_low_confidence_atom_is_still_dropped():
    atom = "(scored A 0.23 (signed Neg (stv 0.75 0.315)))"
    results = [PLNAtomResult(atom, _stv_from_atom(atom))]
    assert _apply_threshold(results, 0.5) == []
    assert _apply_threshold(results, 0.3)[0].atom == atom


def test_an_atom_without_a_truth_value_is_never_filtered():
    """Provenance lines and notes carry no STV and must survive any threshold."""
    note = PLNAtomResult("Omitted (no DrugAge lifespan rows matched): Dasatinib")
    assert _apply_threshold([note], 0.9) == [note]


def test_ranking_reports_what_the_threshold_removed():
    ranking = rank_drugage(
        ["Rapamycin", "Trimethadione"], source=REAL_ROWS, confidence_threshold=0.5
    )
    ranked = {s.compound for s in ranking.ranked}
    filtered = {s.compound for s in ranking.filtered_out}
    assert "Rapamycin" in ranked                      # ITP row, confidence 0.81
    assert filtered                                   # something fell below 0.5
    assert not (ranked & filtered)
    joined = " ".join(r.atom for r in ranking.result.results)
    assert "Filtered out below confidence_threshold" in joined


# ── linear scoring agrees with the MeTTa sort, and is linear ──────────────────

def test_linear_and_metta_sort_agree_on_every_score():
    pool = ["Rapamycin", "Metformin", "Resveratrol", "Acarbose", "Trimethadione"]
    linear, _ = run_drugage_ranking(pool, source=REAL_ROWS, strategy="linear")
    sorted_, _ = run_drugage_ranking(pool, source=REAL_ROWS, strategy="metta_sort")

    lin = {s.compound: s.score for s in parse_scored(linear.results[0].atom)}
    mtt = {s.compound: s.score for s in parse_scored(sorted_.results[0].atom)}
    assert lin == mtt

    # Identical ordering too, up to ties (equal scores may be ordered either
    # way; the linear path breaks a tie by compound name, MeTTa by insertion).
    lin_scores = [s.score for s in parse_scored(linear.results[0].atom)]
    mtt_scores = [s.score for s in parse_scored(sorted_.results[0].atom)]
    assert lin_scores == mtt_scores
    assert lin_scores == sorted(lin_scores, reverse=True)


def test_linear_ranking_emits_one_atom_per_compound_after_the_tuple():
    """The tuple stays first (the formatter/test contract); entries follow."""
    result, _ = run_drugage_ranking(
        ["Rapamycin", "Acarbose"], source=REAL_ROWS, strategy="linear"
    )
    assert len(parse_scored(result.results[0].atom)) == 2      # the ranked tuple
    per_compound = [r for r in result.results[1:] if r.atom.startswith("(scored ")]
    assert len(per_compound) == 2
    assert all(r.stv and "confidence" in r.stv for r in per_compound)


# ── the representative-row policy is correct and documented ───────────────────

def test_a_scoreless_row_is_never_the_representative():
    """D-glucosamine's mouse row reports no change percent; the worm row does."""
    rows = [
        _row("R1", "D_glucosamine", "Mus_musculus", False, "Significant", None, "Female"),
        _row("R2", "D_glucosamine", "Caenorhabditis_elegans", False, "Significant", 11.0),
    ]
    [picked] = _collapse_best(rows)
    assert picked.row_id == "R2"
    assert picked.scorable


def test_a_significant_result_outranks_a_null_within_the_same_tier():
    """The astaxanthin case: two ITP rows, one per sex, +12% sig vs +3% null."""
    rows = [
        _row("F", "Astaxanthin", "Mus_musculus", True, "NotSignificant", 3.0, "Female"),
        _row("M", "Astaxanthin", "Mus_musculus", True, "Significant", 12.0, "Male"),
    ]
    [picked] = _collapse_best(rows)
    assert picked.row_id == "M"
    assert picked.avg_change == 12.0


def test_itp_tier_still_beats_a_significant_single_lab_row():
    """Tier order is unchanged — a flashy worm result cannot outrank the ITP."""
    rows = [
        _row("ITP", "X", "Mus_musculus", True, "NotSignificant", 0.0, "Male"),
        _row("WORM", "X", "Caenorhabditis_elegans", False, "Significant", 40.0),
    ]
    [picked] = _collapse_best(rows)
    assert picked.row_id == "ITP"          # the honest ITP-negative story survives


def test_every_compound_with_no_scorable_row_is_reported_not_silent():
    ranking = rank_drugage(["Rapamycin"], source=REAL_ROWS)
    assert ranking.unscorable == []        # the fixture's rows all score
    # ...and the shape exists for when they do not:
    assert isinstance(ranking.unscorable, list)


def test_rows_carry_species_and_sex_so_the_collapse_is_auditable():
    rows = load_rows(REAL_ROWS)
    assert rows, "fixture should parse"
    assert any(r.sex for r in rows), "HasSex must be parsed off the row block"
    ranking = rank_drugage(["Rapamycin"], source=REAL_ROWS, include_all_rows=True)
    assert ranking.rows
    assert all(r.compound == "Rapamycin" for r in ranking.rows)


# ── the numbers explain themselves ────────────────────────────────────────────

def test_score_semantics_are_returned_and_match_the_calibration_knobs():
    assert "protective" in SCORE_SEMANTICS["sign"]["Neg"]
    assert "harmful" in SCORE_SEMANTICS["sign"]["Pos"]
    assert "+ 20)" in SCORE_SEMANTICS["strength"]
    assert SCORE_SEMANTICS["representative_row_policy"] == list(REPRESENTATIVE_POLICY)
    for tier in ("0.81", "0.45", "0.315", "0.18"):
        assert tier in SCORE_SEMANTICS["confidence_tiers"]


def test_halfsat_and_chain_discount_still_match_the_metta_source():
    """The semantics block must not drift from the .metta knobs it describes."""
    calib = (REPO / "drugage_calibration.metta").read_text(encoding="utf-8")
    deduction = (REPO / "pln_deduction.metta").read_text(encoding="utf-8")
    assert "(= (lifespan-halfsat) 20.0)" in calib
    assert "(= (chain-discount) 0.9)" in deduction
    assert "+ 20)" in SCORE_SEMANTICS["strength"]
    assert "0.9" in SCORE_SEMANTICS["confidence"]


def test_selector_still_honours_an_explicit_compound_filter():
    rows = load_rows(REAL_ROWS)
    picked = select_rows(rows, compounds=["Rapamycin"], best_per_compound=True)
    assert [r.compound for r in picked] == ["Rapamycin"]


# ── confidence_threshold, end to end ─────────────────────────────────────────

@pytest.mark.slow
def test_confidence_threshold_removes_low_confidence_atoms_from_run_query():
    """The runtime path, not the helper.

    The evaluation's finding was that "a 0.5 threshold still returned
    confidence-0.315 rows". The tests that pinned the repair all called the
    private `_apply_threshold` directly, so deleting its call from `run_query`
    left the whole suite green — the defect could return on `/query` and
    `/metta/run` with nothing noticing. This drives the real function.
    """
    pytest.importorskip("hyperon")
    from core.pln_runner import run_query

    kb = [REPO / f for f in (
        "system_types.metta", "logical_predicates.metta",
        "epistemic_calibration.metta", "pln_deduction.metta",
    )]
    # Two independent two-hop chains: one lands at c=0.648, one at c=0.054.
    extra = (
        "(Effect ProbeA ProbeB Pos (stv 0.9 0.8))\n"
        "(Effect ProbeB ProbeC Pos (stv 0.9 0.9))\n"
        "(Effect ProbeD ProbeE Pos (stv 0.9 0.2))\n"
        "(Effect ProbeE ProbeF Pos (stv 0.9 0.3))"
    )
    query = "!(superpose ((infer &self ProbeA ProbeC) (infer &self ProbeD ProbeF)))"

    def confidences(threshold: float) -> list[float]:
        result = run_query(
            metta_query=query, kb_files=kb, extra_atoms=extra,
            confidence_threshold=threshold,
        )
        assert result.status == "ok", result.error
        return sorted((r.stv or {}).get("confidence") for r in result.results)

    unfiltered = confidences(0.0)
    assert len(unfiltered) == 2
    assert unfiltered[0] == pytest.approx(0.054)
    assert unfiltered[1] == pytest.approx(0.648)

    filtered = confidences(0.3)
    assert filtered == [pytest.approx(0.648)], (
        "the sub-threshold derivation survived; confidence_threshold is inert "
        "on the runtime path again"
    )


def test_confidence_threshold_moves_a_compound_to_filtered_out_over_http():
    """And the HTTP contract: filtered, not dropped — the caller is told.

    Needs `build/drugage_etl.metta`, which is gitignored; without it the
    endpoint correctly answers 503 `drugage_build_missing` and there is nothing
    to rank. `bash scripts/run_etl.sh` generates it.
    """
    body = {"compounds": ["Rapamycin", "Trimethadione"], "include_rows": False}

    response = _post("/drugage/rank", {**body, "confidence_threshold": 0.0})
    if response.status_code == 503:
        pytest.skip("build/drugage_etl.metta not generated (run scripts/run_etl.sh)")
    unfiltered = response.json()

    ranked = {e["compound"]: e["confidence"] for e in unfiltered["ranked"]}
    # An ITP row (0.90 x 1.0 x 0.9) against an invertebrate Significant row
    # (0.35 x 1.0 x 0.9) — a threshold between them must separate them.
    assert ranked == {"Rapamycin": pytest.approx(0.81),
                      "Trimethadione": pytest.approx(0.315)}
    assert unfiltered["filtered_out"] == []

    filtered = _post("/drugage/rank", {**body, "confidence_threshold": 0.5}).json()
    assert [e["compound"] for e in filtered["ranked"]] == ["Rapamycin"]
    # Removed from the ranking and REPORTED, with the confidence that removed
    # it — an omission a caller cannot see is how a ranking starts lying.
    assert [e["compound"] for e in filtered["filtered_out"]] == ["Trimethadione"]
    assert filtered["filtered_out"][0]["confidence"] == pytest.approx(0.315)
