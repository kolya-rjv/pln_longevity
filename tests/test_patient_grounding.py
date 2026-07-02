"""Smoke tests for the patient-grounding layer (patient_profile.metta).

These load the full inference stack PLUS patient_profile.metta into one shared
MeTTa space (the same one-space strategy pln_chat/core/pln_runner.py uses) and
assert that:

  * raw standardized values are GROUNDED into Elevated / Normal / Low findings
    at the documented threshold (and unmeasured markers invent nothing);
  * a patient's ELEVATED markers become the observation set that drives Demo 1;
  * `diagnose-patient` reproduces the population abductive ranking from the
    patient's measured values, carrying an un-explainable finding (AgeAccelGrim)
    without crediting it to any hypothesis;
  * `patient-relevance` scores an intervention by how much it reduces the markers
    THIS patient presents elevated, and the personalized ranking folds that in
    additively on top of the population protective score (Demo 2 f(patient
    factors)).

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

# Dependency-ordered load (matches the headers in each .metta file); the patient
# layer loads last, on top of the whole inference stack.
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
]

_STV = r"\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)"
_HYP_RE = re.compile(
    r"\(Hypothesis\s+(\S+)\s+" + _STV + r"\s+([-\d.eE]+)\s+([-\d.eE]+)\s+"
    r"\(SupportedBy\s+\(([^)]*)\)\)"
)


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """A MeTTa interpreter with the full KB + patient layer in one shared space."""
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES)
    m = MeTTa()
    m.run(text)
    return m


def _one(kb: MeTTa, query: str) -> str:
    """Run a query expected to return exactly one atom; return it as a string."""
    res = kb.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    return str(res[0][0])


def _num(kb: MeTTa, query: str) -> float:
    return float(_one(kb, query))


def _is_empty(res) -> bool:
    """MeTTa returns a single empty group when nothing matches."""
    return res == [[]] or all(len(g) == 0 for g in res)


# ── Grounding: standardized value -> qualitative status ─────────────────────────

@pytest.mark.parametrize("z, status", [
    ("1.6", "Elevated"),
    ("1.01", "Elevated"),
    ("1.0", "Normal"),      # boundary: strictly-greater threshold -> Normal
    ("0.0", "Normal"),
    ("-1.0", "Normal"),     # boundary: strictly-less threshold -> Normal
    ("-1.5", "Low"),
])
def test_z_to_status(kb, z, status):
    assert _one(kb, f"!(z->status {z})") == status


@pytest.mark.parametrize("marker, status", [
    ("DNAmPAI1", "Elevated"),
    ("DNAmGDF15", "Elevated"),
    ("CRP", "Elevated"),
    ("AgeAccelGrim", "Elevated"),
    ("HorvathAgeAccel", "Normal"),   # discordance: first-gen clock is normal
    ("DNAmLeptin", "Low"),
])
def test_patient_status(kb, marker, status):
    assert _one(kb, f"!(patient-status &self Patient001 {marker})") == status


def test_unmeasured_marker_invents_nothing(kb):
    # A marker with no MeasuredZ atom yields NO status (not a default Normal).
    assert _is_empty(kb.run("!(patient-status &self Patient001 DNAmTIMP1)"))


# ── Observation set: only ELEVATED markers, derived from measured values ─────────

def test_patient_observations_are_the_elevated_markers(kb):
    obs = set(re.findall(r"\w+", _one(kb, "!(patient-observations &self Patient001)")))
    assert obs == {"DNAmPAI1", "DNAmGDF15", "CRP", "AgeAccelGrim"}
    # Normal / Low markers are excluded — grounding, not a raw dump.
    assert "HorvathAgeAccel" not in obs
    assert "DNAmLeptin" not in obs


# ── Demo 1 grounded: diagnose straight from the patient profile ──────────────────

def _diagnose_patient(kb: MeTTa) -> list[dict]:
    out = _one(
        kb,
        "!(diagnose-patient &self Patient001 "
        "(CellularSenescence MitochondrialDysfunction ChronicInflammation))",
    )
    parsed = []
    for m in _HYP_RE.finditer(out):
        parsed.append({
            "name": m.group(1),
            "s": float(m.group(2)), "c": float(m.group(3)),
            "coverage": int(float(m.group(4))), "mass": float(m.group(5)),
            "markers": set(m.group(6).split()),
        })
    return parsed


def test_diagnose_patient_reproduces_population_ranking(kb):
    dx = _diagnose_patient(kb)
    assert [h["name"] for h in dx] == [
        "CellularSenescence", "ChronicInflammation", "MitochondrialDysfunction"]
    assert [h["coverage"] for h in dx] == [3, 2, 1]


def test_diagnose_patient_provenance_and_tv(kb):
    dx = {h["name"]: h for h in _diagnose_patient(kb)}
    # CellularSenescence accounts for exactly the three explainable findings.
    assert dx["CellularSenescence"]["markers"] == {"DNAmPAI1", "DNAmGDF15", "CRP"}
    assert dx["CellularSenescence"]["s"] == pytest.approx(0.595)
    assert dx["CellularSenescence"]["c"] == pytest.approx(0.625, abs=1e-2)


def test_unexplainable_finding_is_carried_not_credited(kb):
    # AgeAccelGrim is an elevated finding but no bridge reaches it, so it appears
    # in the observation set yet in NO hypothesis's SupportedBy — never invented
    # into an explanation.
    dx = _diagnose_patient(kb)
    for h in dx:
        assert "AgeAccelGrim" not in h["markers"]


# ── Demo 2 grounded: patient factors in intervention ranking ─────────────────────

def test_patient_relevance_ranks_by_targeting_the_patients_markers(kb):
    dq = _num(kb, "!(patient-relevance &self Patient001 DasatinibPlusQuercetin)")
    fis = _num(kb, "!(patient-relevance &self Patient001 Fisetin)")
    spr = _num(kb, "!(patient-relevance &self Patient001 Spermidine)")
    ela = _num(kb, "!(patient-relevance &self Patient001 Elamipretide)")
    # Senolytics reduce this patient's elevated senescence-axis markers -> > 0,
    # ordered by mechanism strength / chain length; Elamipretide reaches none -> 0.
    assert dq > fis > spr > 0.0
    assert ela == pytest.approx(0.0)


def _ranked(kb: MeTTa, query: str) -> list[tuple[str, float]]:
    out = _one(kb, query)
    return [(m.group(1), float(m.group(2)))
            for m in re.finditer(r"\(scored\s+(\S+)\s+([-\d.eE]+)", out)]


def test_personalized_ranking_order_and_omission(kb):
    order = _ranked(
        kb,
        "!(rank-interventions-for-patient &self Patient001 "
        "(DasatinibPlusQuercetin Fisetin Spermidine Elamipretide) CoronaryHeartDisease)",
    )
    names = [n for n, _ in order]
    # Most relevant/protective first; Elamipretide has no chain to CHD -> omitted.
    assert names == ["DasatinibPlusQuercetin", "Fisetin", "Spermidine"]


def test_personalized_score_is_population_base_plus_relevance(kb):
    # The personalized score is the population protective score plus the
    # patient-relevance bonus — additive and auditable.
    base = _num(kb, "!(rank-score (infer &self DasatinibPlusQuercetin CoronaryHeartDisease))")
    rel = _num(kb, "!(patient-relevance &self Patient001 DasatinibPlusQuercetin)")
    order = dict(_ranked(
        kb,
        "!(rank-interventions-for-patient &self Patient001 "
        "(DasatinibPlusQuercetin Fisetin Spermidine) CoronaryHeartDisease)",
    ))
    assert order["DasatinibPlusQuercetin"] == pytest.approx(base + rel)
