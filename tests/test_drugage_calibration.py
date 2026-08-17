"""Tests for the DrugAge ETL -> inference wiring.

Covers the two halves of the vertical slice:

  * drugage_calibration.metta — lifting ONE raw DrugAge row into a signed,
    calibrated (Effect <compound> Lifespan <sign> (stv s c)) link, with
    confidence scaled by species tier / ITP / significance, and the
    Lifespan -> Mortality sign convention;
  * the scoped-space selector + runner (pln_chat) — pulling only the relevant
    rows so the space never approaches the hyperon panic threshold, and ranking
    real compounds through the existing rank-interventions / rank-score.

Same one-space strategy as tests/test_pln_deduction.py: build a MeTTa space from
an explicit file list + a small row slice, assert on collapsed results.

Skipped automatically if `hyperon` is not installed.
Run:  pytest tests/ -v
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

hyperon = pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_ROWS = FIXTURES / "drugage_real_rows.metta"

# Make the pln_chat selector/runner importable and force runtime mode on.
os.environ["PLN_RUNTIME_AVAILABLE"] = "true"
sys.path.insert(0, str(REPO / "pln_chat"))

from core.pln_runner import DRUGAGE_STACK, run_drugage_ranking  # noqa: E402
from ontology.drugage_selector import (  # noqa: E402
    BUILD_DRUGAGE,
    build_drugage_slice,
    load_rows,
    select_rows,
)

_STV = r"\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)"
_SIGNED = re.compile(r"\(signed\s+(Pos|Neg)\s+" + _STV + r"\)")
_EFFECT = re.compile(r"\(Effect\s+(\S+)\s+Lifespan\s+(Pos|Neg)\s+" + _STV + r"\)")

# Per-hop discount combine-tv applies through the Lifespan -> Mortality bridge.
HOP = 0.9

# ── Synthetic fixture: one row per calibration axis, exact values ────────────────
# Distinct compound names so a single space serves every unit query.
SYNTH = """
; DrugA: ITP mouse, +13%, Significant  -> Pos, strength 13/33, c 0.90
(InstanceOf E1 Experiment)
(UsesIntervention E1 DrugA)
(UsesSpecies E1 Mus_musculus)
(IsITPStudy E1)
(AvgLifespanChangePercent E1 13.0)
(AvgLifespanSignificance E1 Significant)
(ReportedIn E1 PMID_1)

; MouseSig: non-ITP mouse, +20%, Significant   -> c 0.50 (Vertebrate)
(InstanceOf E2 Experiment)
(UsesIntervention E2 MouseSig)
(UsesSpecies E2 Mus_musculus)
(AvgLifespanChangePercent E2 20.0)
(AvgLifespanSignificance E2 Significant)

; WormSig: non-ITP worm, +20%, Significant     -> c 0.35 (Invertebrate)
(InstanceOf E3 Experiment)
(UsesIntervention E3 WormSig)
(UsesSpecies E3 Caenorhabditis_elegans)
(AvgLifespanChangePercent E3 20.0)
(AvgLifespanSignificance E3 Significant)

; ItpMouse: ITP mouse, +20%, Significant        -> c 0.90
(InstanceOf E4 Experiment)
(UsesIntervention E4 ItpMouse)
(UsesSpecies E4 Mus_musculus)
(IsITPStudy E4)
(AvgLifespanChangePercent E4 20.0)
(AvgLifespanSignificance E4 Significant)

; MouseNotSig: non-ITP mouse, +20%, NotSignificant -> c 0.40 (gate)
(InstanceOf E5 Experiment)
(UsesIntervention E5 MouseNotSig)
(UsesSpecies E5 Mus_musculus)
(AvgLifespanChangePercent E5 20.0)
(AvgLifespanSignificance E5 NotSignificant)

; ExtDrug: extends lifespan (+30%)  -> net Neg (protective) at Mortality
(InstanceOf E6 Experiment)
(UsesIntervention E6 ExtDrug)
(UsesSpecies E6 Mus_musculus)
(IsITPStudy E6)
(AvgLifespanChangePercent E6 30.0)
(AvgLifespanSignificance E6 Significant)

; ToxDrug: SHORTENS lifespan (-30%) -> net Pos (harmful) at Mortality
(InstanceOf E7 Experiment)
(UsesIntervention E7 ToxDrug)
(UsesSpecies E7 Mus_musculus)
(AvgLifespanChangePercent E7 -30.0)
(AvgLifespanSignificance E7 Significant)

; Ranking pool: GOOD (ITP +20), WEAK (worm +40), NIL (ITP null), BAD (-30)
(InstanceOf G1 Experiment)
(UsesIntervention G1 GoodDrug)
(UsesSpecies G1 Mus_musculus)
(IsITPStudy G1)
(AvgLifespanChangePercent G1 20.0)
(AvgLifespanSignificance G1 Significant)

(InstanceOf G2 Experiment)
(UsesIntervention G2 WeakDrug)
(UsesSpecies G2 Caenorhabditis_elegans)
(AvgLifespanChangePercent G2 40.0)
(AvgLifespanSignificance G2 Significant)

(InstanceOf G3 Experiment)
(UsesIntervention G3 NilDrug)
(UsesSpecies G3 Mus_musculus)
(IsITPStudy G3)
(AvgLifespanChangePercent G3 0.0)
(AvgLifespanSignificance G3 NotSignificant)

(InstanceOf G4 Experiment)
(UsesIntervention G4 BadDrug)
(UsesSpecies G4 Mus_musculus)
(AvgLifespanChangePercent G4 -30.0)
(AvgLifespanSignificance G4 Significant)
"""


def _stack_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in DRUGAGE_STACK)


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """DrugAge inference stack + the synthetic rows, in one shared space."""
    m = MeTTa()
    m.run(_stack_text() + "\n" + SYNTH)
    return m


def _num(kb: MeTTa, query: str) -> float:
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    return float(str(res[0][0]))


def _effect(kb: MeTTa, query: str):
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    m = _EFFECT.search(str(res[0][0]))
    assert m, f"no Effect link in {query}: {res}"
    return m.group(1), m.group(2), float(m.group(3)), float(m.group(4))


def _signed(kb: MeTTa, query: str):
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    m = _SIGNED.search(str(res[0][0]))
    assert m, f"no signed-stv in {query}: {res}"
    return m.group(1), float(m.group(2)), float(m.group(3))


def _ranked(kb: MeTTa, cands: str, outcome: str = "Mortality") -> list[tuple[str, float]]:
    res = kb.run(f"!(rank-interventions &self ({cands}) {outcome})")
    assert res and res[0], f"no ranking: {res}"
    out = str(res[0][0])
    return [(m.group(1), float(m.group(2)))
            for m in re.finditer(r"\(scored\s+(\S+)\s+([-\d.eE]+)", out)]


# ── The deliverable: one row -> one signed, calibrated Effect ────────────────────

def test_lift_one_row_to_signed_effect(kb):
    comp, sign, s, c = _effect(kb, "!(drugage-effect &self E1)")
    assert comp == "DrugA"
    assert sign == "Pos"                       # +13% extends lifespan
    assert s == pytest.approx(13.0 / 33.0)     # saturating: |13| / (|13| + 20)
    assert c == pytest.approx(0.90)            # ITP significant -> ITP_Positive


# ── Confidence: species tier, ITP, significance each move it as designed ─────────

def test_species_tier_lowers_confidence(kb):
    mouse = _num(kb, "!(drugage-confidence &self E2)")  # Vertebrate
    worm = _num(kb, "!(drugage-confidence &self E3)")   # Invertebrate
    assert mouse == pytest.approx(0.50)
    assert worm == pytest.approx(0.35)
    assert worm < mouse                                 # "shown only in a worm is weaker"


def test_itp_raises_confidence_over_single_study(kb):
    itp = _num(kb, "!(drugage-confidence &self E4)")     # ITP mouse
    single = _num(kb, "!(drugage-confidence &self E2)")  # non-ITP mouse
    assert itp == pytest.approx(0.90)
    assert itp > single                                  # ITP is gold-standard


def test_significance_gate_caps_confidence(kb):
    sig = _num(kb, "!(drugage-confidence &self E2)")     # Significant
    nonsig = _num(kb, "!(drugage-confidence &self E5)")  # NotSignificant
    assert sig == pytest.approx(0.50)
    assert nonsig == pytest.approx(0.40)                 # gated below the tier
    assert nonsig < sig


def test_strength_saturates_monotonically(kb):
    # bigger reported change -> bigger strength, but bounded in [0,1)
    s13 = _num(kb, "!(drugage-row-strength &self E1)")   # +13%
    s20 = _num(kb, "!(drugage-row-strength &self E2)")   # +20%
    assert s13 == pytest.approx(13.0 / 33.0)
    assert s20 == pytest.approx(20.0 / 40.0)
    assert 0.0 < s13 < s20 < 1.0


# ── The SIGN TRAP guard: beneficial Lifespan, but Neg=beneficial preserved ───────

def test_extending_lifespan_is_protective_at_mortality(kb):
    # +30% extends lifespan (Pos) -> reduces mortality (Neg): net protective.
    sign, s, c = _signed(kb, "!(infer &self ExtDrug Mortality)")
    assert sign == "Neg"
    assert s == pytest.approx(30.0 / 50.0)
    assert c == pytest.approx(0.90 * HOP)


def test_shortening_lifespan_is_harmful_at_mortality(kb):
    # -30% shortens lifespan (Neg) -> raises mortality (Pos): net harmful.
    # Getting this wrong silently inverts the whole ranking (the documented trap).
    sign, s, c = _signed(kb, "!(infer &self ToxDrug Mortality)")
    assert sign == "Pos"
    assert s == pytest.approx(30.0 / 50.0)


# ── Ranking over a slice: order + signs + the beneficial/harmful split ───────────

def test_ranking_orders_by_protective_effect(kb):
    order = _ranked(kb, "GoodDrug WeakDrug NilDrug BadDrug")
    names = [n for n, _ in order]
    scores = dict(order)
    # GOOD (ITP, +20) > WEAK (worm, +40 but low confidence) > NIL (~0) > BAD (<0)
    assert names == ["GoodDrug", "WeakDrug", "NilDrug", "BadDrug"]
    assert scores["GoodDrug"] == pytest.approx(0.5 * 0.90 * HOP)      # 0.405
    assert scores["WeakDrug"] == pytest.approx((40 / 60) * 0.35 * HOP)  # ~0.21
    assert scores["NilDrug"] == pytest.approx(0.0)                    # null: high conf, ~0 strength
    assert scores["BadDrug"] < 0.0                                    # harmful ranks below zero


def test_species_can_trade_off_against_magnitude(kb):
    # A big worm effect (WEAK) outscoring a small ITP effect is legitimate: the
    # ranking weighs magnitude AND confidence, never sign alone.
    _, weak = next(x for x in _ranked(kb, "WeakDrug") if x[0] == "WeakDrug")
    assert weak == pytest.approx((40 / 60) * 0.35 * HOP)


# ══════════════════════════ Selector (pure Python) ══════════════════════════════

def test_selector_parses_and_filters_by_compound():
    rows = load_rows(REAL_ROWS)
    assert {r.compound for r in rows} == {
        "Rapamycin", "Acarbose", "Resveratrol", "Metformin", "Trimethadione"}
    rap = select_rows(rows, compounds=["rapamycin"])   # loose, case-insensitive
    assert rap and all(r.compound == "Rapamycin" for r in rap)


def test_selector_reads_itp_and_significance():
    rows = {r.compound: r for r in load_rows(REAL_ROWS)}
    assert rows["Resveratrol"].is_itp and rows["Resveratrol"].significance == "NotSignificant"
    assert rows["Trimethadione"].species == "Caenorhabditis_elegans"
    assert not rows["Trimethadione"].is_itp


def test_best_per_compound_prefers_itp_tier():
    # Two rows for one compound (ITP + non-ITP); best_per_compound keeps the ITP one.
    rows = load_rows(REAL_ROWS)
    itp = next(r for r in rows if r.compound == "Rapamycin")
    from ontology.drugage_selector import DrugAgeRow
    weaker = DrugAgeRow("X", "Rapamycin", "Caenorhabditis_elegans", False,
                        "NotSignificant", 2.0, "(InstanceOf X Experiment)")
    picked = select_rows([itp, weaker], best_per_compound=True)
    assert len(picked) == 1 and picked[0].is_itp


# ══════════════════ Scoped space: small, no panic, real ranking ═════════════════

def test_scoped_stack_excludes_the_full_dump():
    # The panic fix: the scoped stack is the small hand-written layers only —
    # never the multi-thousand-row ETL dumps.
    names = {p.name for p in DRUGAGE_STACK}
    assert "drugage_etl.metta" not in names
    assert "drugage_etl_short.metta" not in names
    assert all(p.stat().st_size < 60_000 for p in DRUGAGE_STACK)


def test_scoped_space_stays_small():
    # stack + a 2-compound slice must stay far under the ~4.8k-atom panic boundary.
    slice_text, rows = build_drugage_slice(
        ["Rapamycin", "Resveratrol"], source=REAL_ROWS, best_per_compound=True)
    text = _stack_text() + "\n" + slice_text
    atoms = len(re.findall(r"^\s*\(", text, flags=re.MULTILINE))
    assert atoms < 3000, f"scoped space too big: {atoms} atoms"
    assert len(rows) <= 2


def test_ranking_real_rows_downweights_itp_negative():
    # THE BONUS (pipeline.md App. B.3 / Demo 6): resveratrol failed the ITP, so
    # despite HIGH (ITP) confidence its near-zero strength sinks it far below a
    # replicated ITP positive (rapamycin). Runs through the shipped Python path
    # (selector -> scoped space -> rank-interventions), so completing at all also
    # proves the scoped space does not trip the hyperon panic.
    result, rows = run_drugage_ranking(
        ["Rapamycin", "Resveratrol", "Acarbose", "Metformin", "Trimethadione"],
        source=REAL_ROWS,
    )
    assert result.status == "ok", result.error
    order = [(m.group(1), float(m.group(2)))
             for m in re.finditer(r"\(scored\s+(\S+)\s+([-\d.eE]+)",
                                  result.results[0].atom)]
    names = [n for n, _ in order]
    scores = dict(order)
    assert names[0] == "Rapamycin"                       # replicated ITP positive on top
    assert scores["Resveratrol"] == pytest.approx(0.0, abs=1e-9)
    assert scores["Metformin"] == pytest.approx(0.0, abs=1e-9)
    assert scores["Rapamycin"] > scores["Resveratrol"]   # same tier, strength decides


@pytest.mark.skipif(not BUILD_DRUGAGE.exists(),
                    reason="full ETL output (build/drugage_etl.metta) not generated")
def test_full_dataset_selector_and_scoped_ranking():
    # On the FULL 3,423-row dump the selector still returns only a handful of
    # rows, and ranking them runs cleanly (no panic) end-to-end.
    all_rows = load_rows(BUILD_DRUGAGE)
    assert len(all_rows) > 3000
    result, rows = run_drugage_ranking(["Rapamycin", "Resveratrol"], source=BUILD_DRUGAGE)
    assert len(rows) <= 10           # scoped, not the whole dump
    assert result.status == "ok", result.error
    scores = {m.group(1): float(m.group(2))
              for m in re.finditer(r"\(scored\s+(\S+)\s+([-\d.eE]+)",
                                   result.results[0].atom)}
    assert scores["Rapamycin"] > scores["Resveratrol"]
