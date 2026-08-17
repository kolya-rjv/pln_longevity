"""Smoke tests for the cardiovascular risk-prediction layer (pln_risk_prediction.metta).

These load the full inference + patient + counterfactual stack PLUS
pln_risk_prediction.metta into one shared MeTTa space (the same one-space strategy
pln_chat/core/pln_runner.py uses) and assert Demo 5's headline properties:

  * predict-risk returns an ABSOLUTE point estimate = baseline × HR^(z·sd->years),
    read from the Lu 2019 AgeAccelGrim -> CHD hazard ratio already in the KB;
  * a CONFIDENCE INTERVAL brackets the point estimate and WIDENS as confidence
    falls (uncertainty is first-class);
  * risk-decomposition splits the EXCESS over baseline into attributable component
    factors that SUM (with an explicit residual) back to the excess — crediting the
    two elevated GrimAge surrogates (DNAmPAI1, DNAmGDF15), never inventing one;
  * projected risk under the three scenarios reproduces the Demo-4 ordering in
    absolute-risk terms — clearing senescence (B) beats reducing inflammation (A)
    beats the metabolic lever (C ≈ 0), which reaches no clock surrogate;
  * an edge-less lever (Elamipretide) is omitted, not given a fabricated number;
  * identical inputs -> identical output (determinism).

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

# Dependency-ordered load (matches the headers in each .metta file); the
# risk-prediction layer loads last, on top of the whole inference + patient +
# counterfactual stack (it reuses decompose-grimage / counterfactual-patient).
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
    "pln_abductive_diagnosis.metta",
    "patient_profile.metta",
    "pln_counterfactual.metta",
    "pln_risk_prediction.metta",
]

_NUM = r"([-\d.eE]+)"


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """A MeTTa interpreter with the full KB + risk layer in one space."""
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES)
    m = MeTTa()
    m.run(text)
    return m


def _is_empty(res) -> bool:
    """MeTTa returns a single empty group when nothing matches."""
    return res == [[]] or all(len(g) == 0 for g in res)


def _one(kb: MeTTa, query: str) -> str:
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    return str(res[0][0])


def _field(out: str, name: str) -> float:
    m = re.search(r"\(" + re.escape(name) + r"\s+" + _NUM + r"\)", out)
    assert m, f"field {name} not in {out}"
    return float(m.group(1))


def _prediction(kb: MeTTa, patient: str = "Patient001") -> dict:
    out = _one(kb, f"!(predict-risk-patient &self {patient})")
    ci = re.search(r"\(ci\s+" + _NUM + r"\s+" + _NUM + r"\)", out)
    assert ci, f"no ci in {out}"
    return {
        "point": _field(out, "point"),
        "ci_lo": float(ci.group(1)),
        "ci_hi": float(ci.group(2)),
        "confidence": _field(out, "confidence"),
        "baseline": _field(out, "baseline"),
        "multiplier": _field(out, "multiplier"),
    }


def _projection(kb: MeTTa, lever: str) -> dict:
    out = _one(kb, f"!(project-risk-patient &self Patient001 {lever})")
    via = re.search(r"\(Via\s+\(([^)]*)\)\)", out)
    return {
        "point": _field(out, "point"),
        "reduction": _field(out, "reduction"),
        "delta_clock": _field(out, "delta-clock"),
        "confidence": _field(out, "confidence"),
        "via": set(via.group(1).split()) if via else set(),
    }


# ── (i) The absolute-risk point estimate ─────────────────────────────────────────

def test_point_estimate_is_baseline_times_hazard(kb):
    p = _prediction(kb, "Patient001")
    # Patient001: 58yo male -> baseline 0.08; AgeAccelGrim z 1.6, HR 1.07/yr,
    # sd->years 4.2 -> exponent 6.72 -> multiplier 1.07**6.72.
    assert p["baseline"] == pytest.approx(0.08)
    assert p["multiplier"] == pytest.approx(1.07 ** (1.6 * 4.2))
    assert p["point"] == pytest.approx(0.08 * 1.07 ** (1.6 * 4.2))
    # Sanity: a real elevated risk, well above baseline and still a probability.
    assert p["baseline"] < p["point"] < 1.0


def test_baseline_band_differs_by_age(kb):
    # Patient002 is 61yo male -> the 60-69 baseline band (0.15), not 0.08.
    p2 = _prediction(kb, "Patient002")
    assert p2["baseline"] == pytest.approx(0.15)
    assert p2["point"] == pytest.approx(0.15 * 1.07 ** (1.3 * 4.2))


# ── (ii) The confidence interval brackets the point and widens as c falls ────────

def test_confidence_interval_brackets_point_estimate(kb):
    p = _prediction(kb, "Patient001")
    assert p["ci_lo"] < p["point"] < p["ci_hi"]
    # confidence = evidence 0.85 × one modeling discount 0.9 = 0.765.
    assert p["confidence"] == pytest.approx(0.85 * 0.9)
    # Relative half-width = (1 − c) × k (k = 1.0), symmetric around the point.
    rel = 1.0 - p["confidence"]
    assert p["ci_lo"] == pytest.approx(p["point"] * (1.0 - rel))
    assert p["ci_hi"] == pytest.approx(p["point"] * (1.0 + rel))


def test_interval_widens_as_confidence_drops(kb):
    # Lowering the modeling-confidence knob must WIDEN the interval — uncertainty is
    # first-class, not cosmetic. Re-derive the CI at a lower discount and compare
    # relative widths (the point estimate itself is unchanged).
    p = _prediction(kb, "Patient001")
    width_hi = (p["ci_hi"] - p["ci_lo"]) / p["point"]
    lower_c = _one(
        kb,
        "!(let* (($p (absolute-risk &self Patient001 CoronaryHeartDisease)) "
        "        ($c (* (clock-conf &self CoronaryHeartDisease) 0.5)) "
        "        ($rel (* (- 1.0 $c) (risk-ci-k)))) "
        "   (- (* $p (+ 1.0 $rel)) (* $p (- 1.0 $rel))))",
    )
    width_lo_c = float(lower_c) / p["point"]
    assert width_lo_c > width_hi


# ── (iii) Decomposition of the excess into contributing factors ──────────────────

def test_decomposition_credits_elevated_components(kb):
    out = _one(kb, "!(risk-decomposition-patient &self Patient001)")
    comps = set(re.findall(r"\(RiskFactor\s+(\S+)", out))
    assert comps == {"DNAmPAI1", "DNAmGDF15"}      # the two elevated surrogates
    # Low / unmeasured surrogates are never invented as factors.
    assert "DNAmLeptin" not in comps
    assert "DNAmTIMP1" not in comps


def test_decomposition_sums_to_the_excess(kb):
    out = _one(kb, "!(risk-decomposition-patient &self Patient001)")
    baseline = _field(out, "baseline")
    total = _field(out, "total")
    excess = _field(out, "excess")
    attributed = _field(out, "attributed-excess")
    residual = _field(out, "residual-excess")
    # excess == total − baseline, and attributed + residual == excess (nothing lost).
    assert excess == pytest.approx(total - baseline)
    assert attributed + residual == pytest.approx(excess)
    # The coarse v1 weights + unmeasured components leave an honest positive residual.
    assert residual > 0.0
    # Per-factor attributable-risk fields also sum to the attributed excess.
    per_factor = [float(x) for x in re.findall(
        r"\(attributable-risk\s+" + _NUM + r"\)", out)]
    assert sum(per_factor) == pytest.approx(attributed)


def test_decomposition_shares_track_component_elevation(kb):
    out = _one(kb, "!(risk-decomposition-patient &self Patient001)")
    # DNAmPAI1 (z 1.8) carries a larger share of the excess than DNAmGDF15 (z 1.5).
    pai1 = re.search(r"\(RiskFactor DNAmPAI1\b.*?\(attributable-risk\s+" + _NUM + r"\)", out)
    gdf15 = re.search(r"\(RiskFactor DNAmGDF15\b.*?\(attributable-risk\s+" + _NUM + r"\)", out)
    assert float(pai1.group(1)) > float(gdf15.group(1))


# ── (iv) Projected risk under interventions: B > A > C ≈ 0 ────────────────────────

def test_projection_scenario_a_reduces_risk(kb):
    a = _projection(kb, "ChronicInflammation")
    assert a["via"] == {"DNAmGDF15"}               # inflammation reaches only GDF15
    assert a["delta_clock"] < 0.0                  # lowers the clock
    assert a["reduction"] > 0.0                    # a real absolute-risk reduction
    assert a["point"] < _prediction(kb)["point"]


def test_projection_b_beats_a_beats_c(kb):
    a = _projection(kb, "ChronicInflammation")     # scenario A
    b = _projection(kb, "CellularSenescence")      # scenario B
    c = _projection(kb, "InsulinResistance")       # scenario C
    # THE headline, in absolute-risk terms: clearing senescence (upstream, reaches
    # both clock surrogates) buys a BIGGER CHD-risk reduction than inflammation alone,
    # and the metabolic lever moves the clock — hence the risk — by ~0.
    assert b["reduction"] > a["reduction"] > c["reduction"]
    assert c["reduction"] == pytest.approx(0.0)
    assert b["via"] == {"DNAmPAI1", "DNAmGDF15"}


def test_projection_confidence_follows_route_length(kb):
    # The bigger-effect scenario (B) is held with LOWER confidence — its clock
    # surrogates sit further downstream of the lever (uncertainty grows with route).
    a = _projection(kb, "ChronicInflammation")
    b = _projection(kb, "CellularSenescence")
    assert b["confidence"] < a["confidence"]


def test_metabolic_lever_projects_zero_change(kb):
    # The metabolic axis reaches no clock surrogate, so its projected CHD-risk change
    # is honestly ~0 (empty Via) — not a fabricated number.
    c = _projection(kb, "InsulinResistance")
    assert c["reduction"] == pytest.approx(0.0)
    assert c["delta_clock"] == pytest.approx(0.0)
    assert c["via"] == set()


# ── (v) An edge-less lever is omitted, not placed ────────────────────────────────

def test_unconnected_lever_is_omitted(kb):
    # Elamipretide has no Effect edge -> counterfactual yields nothing -> the risk
    # projection yields nothing (no fabricated placement).
    assert _is_empty(kb.run("!(project-risk-patient &self Patient001 Elamipretide)"))


# ── The scenario runner returns all three, ordered by reduction magnitude ─────────

def test_scenario_runner_returns_three(kb):
    out = _one(kb, "!(risk-scenarios &self Patient001)")
    levers = set(re.findall(r"\(ProjectedRisk\s+(\S+)", out))
    assert levers == {"ChronicInflammation", "CellularSenescence", "InsulinResistance"}


# ── Determinism: identical inputs -> identical output (headline PLN property) ─────

def test_prediction_is_deterministic(kb):
    assert _prediction(kb, "Patient001") == _prediction(kb, "Patient001")
