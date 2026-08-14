"""Smoke tests for the counterfactual GrimAge layer (pln_counterfactual.metta).

These load the full inference + patient stack PLUS pln_counterfactual.metta into
one shared MeTTa space (the same one-space strategy pln_chat/core/pln_runner.py
uses) and assert Demo 4's headline properties:

  * decompose-grimage splits a patient's ELEVATED GrimAge into per-component
    attributable shares, each traced to the upstream cause(s) infer credits, and
    reports the honest unattributed residual (never inventing a component);
  * the counterfactual do-operator normalizes a lever and propagates the reduction
    through the GrimAge composition edge, returning a signed, uncertainty-
    quantified change with (Via …) provenance;
  * clearing senescence (B) beats reducing inflammation (A) — senescence is
    UPSTREAM, so it reaches BOTH DNAmPAI1 and DNAmGDF15 — a bigger reduction but,
    because those components sit further downstream, at LOWER confidence;
  * a marker lever (CRP) is routed THROUGH its shared cause (ChronicInflammation),
    never via a fictitious CRP -> GrimAge edge; a metabolic lever reaches no clock
    surrogate (honest ~0); an edge-less lever (Elamipretide) is omitted, not placed.

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
# counterfactual layer loads last, on top of the whole inference + patient stack.
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
]

_NUM = r"([-\d.eE]+)"
# (Counterfactual <lever> <outcome> (expected-delta d) (signed Neg (stv s c)) (Via (…)))
_CF_RE = re.compile(
    r"\(Counterfactual\s+(\S+)\s+(\S+)\s+"
    r"\(expected-delta\s+" + _NUM + r"\)\s+"
    r"\(signed\s+(Pos|Neg)\s+\(stv\s+" + _NUM + r"\s+" + _NUM + r"\)\)\s+"
    r"\(Via\s+\(([^)]*)\)\)"
)


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """A MeTTa interpreter with the full KB + counterfactual layer in one space."""
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


def _counterfactual(kb: MeTTa, lever: str, outcome: str = "AgeAccelGrim") -> dict:
    """Run counterfactual-patient and parse the single result into a dict."""
    out = _one(kb, f"!(counterfactual-patient &self Patient001 {lever})")
    m = _CF_RE.search(out)
    assert m, f"unparseable counterfactual for {lever}: {out}"
    return {
        "lever": m.group(1),
        "outcome": m.group(2),
        "delta": float(m.group(3)),
        "sign": m.group(4),
        "s": float(m.group(5)),
        "c": float(m.group(6)),
        "via": set(m.group(7).split()),
    }


# ── Lever resolution — route to the harmful driver (the CRP subtlety) ────────────

@pytest.mark.parametrize("lever, driver", [
    ("ChronicInflammation", "ChronicInflammation"),   # a driver -> itself
    ("CellularSenescence",  "CellularSenescence"),
    ("InsulinResistance",   "InsulinResistance"),
    ("CRP",                 "ChronicInflammation"),    # leaf marker -> its shared cause
    ("DasatinibPlusQuercetin", "CellularSenescence"),  # intervention -> node it reduces
    ("Metformin",           "InsulinResistance"),
])
def test_resolve_lever_routes_to_driver(kb, lever, driver):
    assert _one(kb, f"!(resolve-lever &self {lever})") == driver


def test_resolve_lever_omits_edgeless_node(kb):
    # Elamipretide has only a HallmarkInterventionEvidence record, no Effect edge,
    # so it cannot be resolved to a driver -> nothing (never mis-placed).
    assert _is_empty(kb.run("!(resolve-lever &self Elamipretide)"))


# ── (i) Causal decomposition of the elevated GrimAge ─────────────────────────────

def test_decompose_credits_the_elevated_components(kb):
    out = _one(kb, "!(decompose-grimage &self Patient001)")
    comps = dict(re.findall(r"\(Component\s+(\S+)\s+\(z\s+" + _NUM + r"\)", out))
    # The two elevated GrimAge surrogates are credited with their measured z…
    assert set(comps) == {"DNAmPAI1", "DNAmGDF15"}
    assert float(comps["DNAmPAI1"]) == pytest.approx(1.8)
    assert float(comps["DNAmGDF15"]) == pytest.approx(1.5)
    # …and the Low / unmeasured surrogates are NOT invented as components.
    assert "DNAmLeptin" not in comps            # measured but Low
    assert "DNAmTIMP1" not in comps             # unmeasured


def test_decompose_contributions_and_residual(kb):
    out = _one(kb, "!(decompose-grimage &self Patient001)")
    # contribution = grimage-weight (0.125) × z
    contribs = dict(
        (name, float(c))
        for name, c in re.findall(
            r"\(Component\s+(\S+).*?\(contribution\s+" + _NUM + r"\)", out)
    )
    assert contribs["DNAmPAI1"] == pytest.approx(0.125 * 1.8)    # 0.225
    assert contribs["DNAmGDF15"] == pytest.approx(0.125 * 1.5)   # 0.1875
    attributed = float(re.search(r"\(attributed\s+" + _NUM + r"\)", out).group(1))
    residual = float(re.search(r"\(residual\s+" + _NUM + r"\)", out).group(1))
    assert attributed == pytest.approx(0.4125)
    # AgeAccelGrim measured at 1.6; the coarse v1 weights + unmeasured components
    # leave an honest positive residual, never forced to zero.
    assert residual == pytest.approx(1.6 - 0.4125)
    assert residual > 0.0


def test_decompose_traces_components_to_upstream_causes(kb):
    out = _one(kb, "!(decompose-grimage &self Patient001)")
    # DNAmPAI1's driver is the senescence axis; DNAmGDF15 is reached by three
    # hallmark causes (senescence, inflammation, mito) — the "attributable to"
    # provenance an LLM cannot quantify.
    pai1 = re.search(r"\(Component\s+DNAmPAI1\b.*?\(DrivenBy\s+\(([^)]*)\)\)", out)
    gdf15 = re.search(r"\(Component\s+DNAmGDF15\b.*?\(DrivenBy\s+\(([^)]*)\)\)", out)
    assert set(pai1.group(1).split()) == {"CellularSenescence"}
    assert set(gdf15.group(1).split()) == {
        "CellularSenescence", "ChronicInflammation", "MitochondrialDysfunction"}


def test_grimage_share_single_component(kb):
    out = _one(kb, "!(grimage-share &self Patient001 DNAmPAI1)")
    assert "(Component DNAmPAI1" in out
    assert "(contribution 0.225" in out
    # A Low component has no share (nothing returned), never a fabricated one.
    assert _is_empty(kb.run("!(grimage-share &self Patient001 DNAmLeptin)"))


# ── (ii) Scenario B (clear senescence) > Scenario A (reduce inflammation) ─────────

def test_scenario_a_reaches_only_gdf15(kb):
    a = _counterfactual(kb, "ChronicInflammation")
    assert a["sign"] == "Neg"                       # protective (reduces the clock)
    assert a["via"] == {"DNAmGDF15"}                # inflammation reaches only GDF15
    assert a["delta"] == pytest.approx(-0.121875)   # -(0.125 × 0.65 × 1.5)
    assert a["c"] == pytest.approx(0.65)            # single direct hop


def test_scenario_b_reaches_both_components(kb):
    b = _counterfactual(kb, "CellularSenescence")
    assert b["sign"] == "Neg"
    assert b["via"] == {"DNAmPAI1", "DNAmGDF15"}    # senescence reaches BOTH
    # -(0.125 × 0.595 × 1.8  +  0.125 × 0.469625 × 1.5)
    assert b["delta"] == pytest.approx(-0.2219296875)


def test_scenario_b_beats_scenario_a(kb):
    # THE headline: clearing senescence reduces GrimAge MORE than inflammation
    # alone, because senescence is upstream and reaches both clock surrogates.
    a = _counterfactual(kb, "ChronicInflammation")
    b = _counterfactual(kb, "CellularSenescence")
    assert abs(b["delta"]) > abs(a["delta"])
    assert b["s"] > a["s"]                          # strength = reduction magnitude


# ── (v) Confidence falls with route length ───────────────────────────────────────

def test_confidence_decreases_with_route_length(kb):
    # B's components sit 2–3 hops downstream of the lever; A's is a single hop.
    # So the bigger-magnitude estimate is held with LOWER confidence.
    a = _counterfactual(kb, "ChronicInflammation")
    b = _counterfactual(kb, "CellularSenescence")
    assert b["sign"] == "Neg"
    assert b["c"] < a["c"]                          # 0.318 < 0.65
    # And below the single-hop transmission confidence it aggregates from.
    single_hop = kb.run(
        "!(let (signed Pos (stv $s $c)) "
        "(infer &self ChronicInflammation DNAmGDF15) $c)")
    assert b["c"] < float(str(single_hop[0][0]))    # < 0.65


# ── (iii) The CRP subtlety + the metabolic ~0 ────────────────────────────────────

def test_crp_lever_routes_through_shared_cause_not_a_direct_edge(kb):
    # "If my CRP were normal?" — CRP is a leaf co-effect, so it is routed through
    # ChronicInflammation and moves GrimAge via DNAmGDF15 — identical to scenario
    # A, NOT via a fabricated CRP -> GrimAge edge.
    crp = _counterfactual(kb, "CRP")
    a = _counterfactual(kb, "ChronicInflammation")
    assert crp["via"] == {"DNAmGDF15"}
    assert crp["delta"] == pytest.approx(a["delta"])
    assert crp["delta"] < 0.0                        # a real, non-zero reduction
    # There is NO direct CRP -> GrimAge / CRP -> DNAm* mechanism in the graph.
    assert _is_empty(kb.run("!(pos-transmission &self CRP DNAmGDF15)"))
    assert _is_empty(kb.run("!(match &self (Effect CRP $x $s $tv) $x)"))


def test_metabolic_lever_moves_the_clock_by_zero(kb):
    # The metabolic axis terminates at FastingGlucose -> CHD and reaches NO GrimAge
    # surrogate, so a metabolic lever honestly moves the clock by ~0 (empty Via) —
    # NOT a fabricated number. Both the pathway node and the drug resolve the same.
    for lever in ("InsulinResistance", "Metformin"):
        c = _counterfactual(kb, lever)
        assert c["delta"] == pytest.approx(0.0)
        assert c["s"] == pytest.approx(0.0)
        assert c["via"] == set()


# ── (iv) An edge-less lever is omitted, not placed ───────────────────────────────

def test_unconnected_lever_is_omitted(kb):
    # Elamipretide has no Effect edge at all -> unresolvable -> the counterfactual
    # returns nothing (the restraint of intervention ranking's Elamipretide-omission,
    # carried into counterfactual reasoning).
    assert _is_empty(kb.run("!(counterfactual-patient &self Patient001 Elamipretide)"))


# ── Intervention levers resolve to the driver they target ────────────────────────

def test_intervention_lever_matches_its_target_driver(kb):
    # D+Q normalizes CellularSenescence, so as a lever it reproduces scenario B;
    # Metformin normalizes InsulinResistance, reproducing the metabolic ~0.
    dq = _counterfactual(kb, "DasatinibPlusQuercetin")
    b = _counterfactual(kb, "CellularSenescence")
    assert dq["delta"] == pytest.approx(b["delta"])
    assert dq["via"] == b["via"]


# ── The scenario runner returns all three, in the expected magnitude order ───────

def test_scenario_runner_returns_three_ordered_by_magnitude(kb):
    out = _one(kb, "!(counterfactual-scenarios &self Patient001)")
    by_lever = {m.group(1): float(m.group(3)) for m in _CF_RE.finditer(out)}
    assert set(by_lever) == {
        "ChronicInflammation", "CellularSenescence", "InsulinResistance"}
    # |Δ_B| > |Δ_A| > |Δ_C| == 0
    assert abs(by_lever["CellularSenescence"]) > abs(by_lever["ChronicInflammation"])
    assert abs(by_lever["ChronicInflammation"]) > abs(by_lever["InsulinResistance"])
    assert by_lever["InsulinResistance"] == pytest.approx(0.0)


# ── Determinism: identical inputs -> identical output (headline PLN property) ─────

def test_counterfactual_is_deterministic(kb):
    a1 = _counterfactual(kb, "CellularSenescence")
    a2 = _counterfactual(kb, "CellularSenescence")
    assert a1 == a2
