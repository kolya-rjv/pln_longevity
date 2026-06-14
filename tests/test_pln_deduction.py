"""Smoke tests for the calibration + PLN deduction layers against live hyperon.

These load the real .metta knowledge-base files into one shared MeTTa space
(the same one-space strategy pln_chat/core/pln_runner.py uses) and assert that:

  * the evidence -> truth-value calibration reproduces its documented values;
  * transitive deduction propagates (stv s c) and attenuates confidence;
  * SIGN propagates — a harmful axis is Pos, a senolytic intervention into that
    axis flips the net effect to Neg (protective);
  * a novel cross-source inference is derived with an auditable path;
  * unconnected concepts yield NO chain (no false positives).

Skipped automatically if `hyperon` is not installed.

Run:  pytest tests/ -v       (from the repo root)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

hyperon = pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Dependency-ordered load (matches the headers in each .metta file).
KB_FILES = [
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
]

_STV = r"\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)"
_SIGNED = re.compile(r"\(signed\s+(Pos|Neg)\s+" + _STV + r"\)")


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """A MeTTa interpreter with the full KB loaded into one shared space."""
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES)
    m = MeTTa()
    m.run(text)
    return m


def _stv(kb: MeTTa, query: str) -> tuple[float, float]:
    """Run a query expected to return exactly one (stv s c); return (s, c)."""
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    m = re.search(_STV, str(res[0][0]))
    assert m, f"no stv in result for {query}: {res}"
    return float(m.group(1)), float(m.group(2))


def _signed(kb: MeTTa, query: str) -> tuple[str, float, float]:
    """Run a query expected to return one (signed <Sign> (stv s c))."""
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    m = _SIGNED.search(str(res[0][0]))
    assert m, f"no signed-stv in result for {query}: {res}"
    return m.group(1), float(m.group(2)), float(m.group(3))


# ── Calibration baseline (regression-guards the documented values) ──────────────

@pytest.mark.parametrize("record, exp_s, exp_c", [
    ("Lu2019_AgeAccelGrim_AllCauseMortality", 0.5, 0.85),  # HR 1.10
    ("Lu2019_AgeAccelGrim_CHD",               0.5, 0.85),  # HR 1.07
    ("Lu2019_DNAmPAI1_CHD",                   0.6, 0.85),  # HR 1.31
])
def test_calibration_values(kb, record, exp_s, exp_c):
    s, c = _stv(kb, f"!(calibrate-tv &self {record})")
    assert s == pytest.approx(exp_s)
    assert c == pytest.approx(exp_c)


# ── Sign algebra ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a, b, exp", [
    ("Pos", "Pos", "Pos"),
    ("Pos", "Neg", "Neg"),
    ("Neg", "Pos", "Neg"),
    ("Neg", "Neg", "Pos"),   # reducing a reducer raises
])
def test_sign_product(kb, a, b, exp):
    res = kb.run(f"!(sign-product {a} {b})")
    assert str(res[0][0]) == exp


# ── Deduction: a direct empirical link lifts to its calibrated, signed TV ────────

def test_direct_link_is_calibrated_and_signed(kb):
    sign, s, c = _signed(kb, "!(link-effect &self DNAmPAI1 CoronaryHeartDisease)")
    assert sign == "Pos"                       # HR 1.31 > 1 → harmful
    assert (s, c) == pytest.approx((0.6, 0.85))


# ── Deduction: confidence attenuates with each hop ─────────────────────────────

def test_one_hop_propagation(kb):
    # SASP -> DNAmPAI1 (0.70, 0.65) chained into DNAmPAI1 -> CHD (0.6, 0.85)
    sign, s, c = _signed(kb, "!(infer &self SASP CoronaryHeartDisease)")
    assert sign == "Pos"
    assert s == pytest.approx(0.70 * 0.6)                    # 0.42
    assert c == pytest.approx(0.65 * 0.85 * 0.9)             # 0.49725


def test_novel_cross_source_inference_harmful(kb):
    # CellularSenescence -> CHD is stated by NO source; derived across the
    # Hallmarks concept, the curated SASP/PAI-1 bridges, and Lu 2019. Harmful.
    sign, s, c = _signed(kb, "!(infer &self CellularSenescence CoronaryHeartDisease)")
    assert sign == "Pos"
    assert s == pytest.approx(0.85 * 0.70 * 0.6)             # 0.357
    assert c == pytest.approx(0.65 * 0.65 * 0.85 * 0.9 * 0.9)  # ~0.29089
    assert c < 0.65                                          # below any single hop


# ── The headline: a protective intervention flips the net sign ─────────────────

def test_protective_intervention_flips_sign(kb):
    # D+Q clears senescence (Neg) -> ... -> CHD (Pos axis): net Neg = REDUCES risk.
    sign, s, c = _signed(kb, "!(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)")
    assert sign == "Neg"                                     # protective
    assert s == pytest.approx(0.70 * 0.85 * 0.70 * 0.6)      # 0.2499
    assert c == pytest.approx(0.65 ** 3 * 0.85 * 0.9 ** 3)   # ~0.17017


# ── Provenance: the derivation path is auditable ───────────────────────────────

def test_explain_returns_signed_path(kb):
    res = kb.run("!(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)")
    out = str(res[0][0])
    assert ("(Chain (DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1 "
            "CoronaryHeartDisease)") in out
    assert "(signed Neg" in out


# ── No false chains between unconnected concepts ───────────────────────────────

def test_no_false_chain(kb):
    res = kb.run("!(infer &self DasatinibPlusQuercetin AllCauseMortality)")
    # MeTTa returns a single empty result group when nothing matches.
    assert res == [[]] or all(len(g) == 0 for g in res)


# ── Demo 2: intervention ranking with uncertainty propagation ──────────────────
# The added Fisetin / Spermidine bridges give a candidate pool that all reaches
# CHD through the same senescence axis, with chains of differing length.

def test_fisetin_protective_weaker_than_dq(kb):
    # Fisetin is a senolytic like D+Q but with single-study evidence, so it is
    # protective (Neg) with a smaller magnitude AND lower confidence than D+Q.
    sign, s, c = _signed(kb, "!(infer &self Fisetin CoronaryHeartDisease)")
    assert sign == "Neg"
    assert s == pytest.approx(0.65 * 0.85 * 0.70 * 0.6)          # 0.23205
    assert c == pytest.approx(0.50 * 0.65 * 0.65 * 0.85 * 0.9 ** 3)  # ~0.13091


def test_spermidine_longer_chain_lower_confidence(kb):
    # Spermidine reaches CHD one hop further upstream (via Autophagy), so the
    # net effect stays protective but confidence is lower than the 4-hop
    # senolytics — "uncertainty increases with chain length".
    sign, s, c = _signed(kb, "!(infer &self Spermidine CoronaryHeartDisease)")
    assert sign == "Neg"
    assert s == pytest.approx(0.80 * 0.60 * 0.85 * 0.70 * 0.6)   # 0.17136
    assert c == pytest.approx(0.50 * 0.65 ** 3 * 0.85 * 0.9 ** 4)  # ~0.076577


@pytest.mark.parametrize("query, expect_sign", [
    ("(signed Neg (stv 0.4 0.5))", 1),    # protective -> positive score
    ("(signed Pos (stv 0.4 0.5))", -1),   # harmful   -> negative score
])
def test_rank_score_signs(kb, query, expect_sign):
    res = kb.run(f"!(rank-score {query})")
    score = float(str(res[0][0]))
    assert score == pytest.approx(expect_sign * 0.2)


def _ranked_order(kb: MeTTa, cands: str, outcome: str) -> list[str]:
    """Return the intervention names in ranked order from rank-interventions."""
    res = kb.run(f"!(rank-interventions &self ({cands}) {outcome})")
    assert res and res[0], f"no ranking returned: {res}"
    return re.findall(r"\(scored\s+(\S+)", str(res[0][0]))


def test_rank_interventions_orders_by_protection(kb):
    order = _ranked_order(
        kb,
        "DasatinibPlusQuercetin Fisetin Spermidine Elamipretide",
        "CoronaryHeartDisease",
    )
    # Most protective first; Elamipretide has no chain to CHD -> omitted entirely.
    assert order == ["DasatinibPlusQuercetin", "Fisetin", "Spermidine"]


def test_rank_interventions_is_order_independent(kb):
    # Identical inputs -> identical output regardless of candidate listing order
    # (a headline PLN-over-LLM property).
    a = _ranked_order(kb, "Spermidine Fisetin DasatinibPlusQuercetin", "CoronaryHeartDisease")
    b = _ranked_order(kb, "Fisetin DasatinibPlusQuercetin Spermidine", "CoronaryHeartDisease")
    assert a == b == ["DasatinibPlusQuercetin", "Fisetin", "Spermidine"]
