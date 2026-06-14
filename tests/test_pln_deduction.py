"""Smoke tests for the calibration + PLN deduction layers against live hyperon.

These load the real .metta knowledge-base files into one shared MeTTa space
(the same one-space strategy pln_chat/core/pln_runner.py uses) and assert that:

  * the evidence -> truth-value calibration reproduces its documented values;
  * transitive deduction propagates (stv s c) and attenuates confidence;
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
    "mechanistic_bridges.metta",
    "pln_deduction.metta",
]


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
    m = re.search(r"\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)", str(res[0][0]))
    assert m, f"no stv in result for {query}: {res}"
    return float(m.group(1)), float(m.group(2))


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


# ── Deduction: a direct empirical link lifts to its calibrated TV ───────────────

def test_direct_link_is_calibrated(kb):
    s, c = _stv(kb, "!(link-tv &self DNAmPAI1 CoronaryHeartDisease)")
    assert (s, c) == pytest.approx((0.6, 0.85))


# ── Deduction: confidence attenuates with each hop ─────────────────────────────

def test_one_hop_propagation(kb):
    # SASP -> DNAmPAI1 (0.70, 0.65) chained into DNAmPAI1 -> CHD (0.6, 0.85)
    s, c = _stv(kb, "!(infer &self SASP CoronaryHeartDisease)")
    assert s == pytest.approx(0.70 * 0.6)                    # 0.42
    assert c == pytest.approx(0.65 * 0.85 * 0.9)             # 0.49725


def test_novel_cross_source_inference(kb):
    # CellularSenescence -> CHD is stated by NO source; it is derived across
    # the Hallmarks concept, the curated SASP/PAI-1 bridges, and Lu 2019.
    s, c = _stv(kb, "!(infer &self CellularSenescence CoronaryHeartDisease)")
    assert s == pytest.approx(0.85 * 0.70 * 0.6)             # 0.357
    assert c == pytest.approx(0.65 * 0.65 * 0.85 * 0.9 * 0.9)  # ~0.29089
    # Confidence must be strictly lower than any single hop's confidence.
    assert c < 0.65


# ── Provenance: the derivation path is auditable ───────────────────────────────

def test_explain_returns_path(kb):
    res = kb.run("!(explain &self CellularSenescence CoronaryHeartDisease)")
    out = str(res[0][0])
    assert "(Chain (CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease)" in out
    assert "stv" in out


# ── No false chains between unconnected concepts ───────────────────────────────

def test_no_false_chain(kb):
    res = kb.run("!(infer &self CellularSenescence AllCauseMortality)")
    # MeTTa returns a single empty result group when nothing matches.
    assert res == [[]] or all(len(g) == 0 for g in res)
