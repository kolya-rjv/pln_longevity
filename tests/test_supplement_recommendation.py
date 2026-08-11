"""Smoke tests for the supplement-recommendation layer
(supplement_evidence.metta + pln_supplement_recommendation.metta, pipeline.md Demo 6).

These load the full inference + patient + counterfactual + risk stack PLUS the two
Demo-6 files into one shared MeTTa space (the same one-space strategy
pln_chat/core/pln_runner.py uses) and assert Demo 6's headline properties — the five
things pipeline.md §4.6 / Table 2 say an LLM cannot do:

  (i)   TIERING by evidence quality — Omega3 (extensive human) -> Tier1; Fisetin / NMN
        (animal / preliminary) -> Tier2; supplement confidence stays below RCT-grade;
  (ii)  the NEGATIVE-EVIDENCE VETO — Resveratrol is NotRecommended DESPITE a relevant
        senescence mechanism (an LLM recommends it on that mechanism);
  (iii) PERSONALIZATION reorders — the SAME pool recommends Berberine for the metabolic
        Patient002 but OMITS it for the senescence-dominant Patient001;
  (iv)  INTERACTION flagging — the Berberine<->Metformin interaction fires for Patient002
        (on metformin), not for the med-free Patient001;
  (v)   PROVENANCE + DETERMINISM — each recommendation carries the elevated markers it
        targets, and identical inputs give identical output.

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

# Dependency-ordered load (matches the headers in each .metta file); the two Demo-6
# files load last, on top of the whole inference + patient + counterfactual + risk stack.
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
    "supplement_evidence.metta",
    "pln_supplement_recommendation.metta",
]

_NUM = r"([-\d.eE]+)"
# (SuppRec <supp> (tier <T>) (score <s>) (evidence <cat> <conf>) (safety <lbl> <f>) (Targets (…)))
_REC_RE = re.compile(
    r"\(SuppRec\s+(\S+)\s+"
    r"\(tier\s+(\S+)\)\s+"
    r"\(score\s+" + _NUM + r"\)\s+"
    r"\(evidence\s+(\S+)\s+" + _NUM + r"\)\s+"
    r"\(safety\s+(\S+)\s+" + _NUM + r"\)\s+"
    r"\(Targets\s+\(([^)]*)\)\)"
)


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """A MeTTa interpreter with the full KB + Demo-6 layer in one space."""
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


def _rec(supp: str, out: str) -> dict | None:
    """Parse the SuppRec for one supplement out of a recommendation blob."""
    for m in _REC_RE.finditer(out):
        if m.group(1) == supp:
            return {
                "supp": m.group(1),
                "tier": m.group(2),
                "score": float(m.group(3)),
                "evidence": m.group(4),
                "conf": float(m.group(5)),
                "safety_label": m.group(6),
                "safety_factor": float(m.group(7)),
                "targets": set(m.group(8).split()),
            }
    return None


@pytest.fixture(scope="module")
def rec001(kb) -> str:
    return _one(kb, "!(recommend-supplements-patient &self Patient001)")


@pytest.fixture(scope="module")
def rec002(kb) -> str:
    return _one(kb, "!(recommend-supplements-patient &self Patient002)")


# ── (i) Tiering by evidence quality ──────────────────────────────────────────────

def test_patient001_tier1_is_omega3(rec001):
    # The Tier1 block holds exactly Omega3 (extensive human evidence), tiered Tier1.
    tier1 = re.search(r"\(Tier1HighConfidence\s+\((.*?)\)\)\s+\(Tier2Promising", rec001, re.DOTALL)
    assert tier1 and "Omega3" in tier1.group(1)
    o = _rec("Omega3", rec001)
    assert o["tier"] == "Tier1HighConfidence"
    assert o["evidence"] == "MultipleHumanTrials"


def test_patient001_tier2_holds_fisetin_and_nmn(rec001):
    # Both Tier-2 supplements are present, tiered Tier2, sorted by score (NMN ≻ Fisetin).
    for supp, cat in (("NMN", "SingleHumanTrial"), ("Fisetin", "AnimalStudies_Single")):
        r = _rec(supp, rec001)
        assert r is not None and r["tier"] == "Tier2Promising"
        assert r["evidence"] == cat
    assert _rec("NMN", rec001)["score"] > _rec("Fisetin", rec001)["score"]


def test_supplement_confidence_below_pharmaceutical(kb, rec001):
    # Layer-4 honesty: every supplement recommendation carries a confidence at or below
    # the strongest supplement tier (0.85) — strictly below a pharmaceutical RCT (1.0).
    for supp in ("Omega3", "NMN", "Fisetin"):
        assert _rec(supp, rec001)["conf"] <= 0.85
    # A pharmaceutical mechanism on the same axis carries higher evidence confidence
    # (Metformin -> InsulinResistance is MultipleHumanTrials 0.85 vs e.g. Fisetin 0.50);
    # the point is supplements are NOT conflated with RCT-grade evidence.
    assert _rec("Fisetin", rec001)["conf"] < 1.0


# ── (ii) The negative-evidence veto (the killer) ─────────────────────────────────

def test_resveratrol_is_vetoed_despite_a_relevant_mechanism(kb, rec001):
    # Resveratrol HAS a senescence-clearing mechanism that reaches Patient001's markers
    # (relevance > 0) — an LLM recommends it on exactly that. PLN puts it in
    # NotRecommended on the ITP-negative trial, overriding the mechanism.
    r = _rec("Resveratrol", rec001)
    assert r is not None
    assert r["tier"] == "NotRecommended"
    assert r["evidence"] == "ITP_Negative"
    # The mechanism really is relevant (non-empty targets, positive relevance) — the veto
    # is doing work, not just filtering an irrelevant compound.
    assert r["targets"]                       # reaches real markers
    rel = float(_one(kb, "!(patient-relevance &self Patient001 Resveratrol)"))
    assert rel > 0.0
    # And it lands in the NotRecommended block, not a recommended tier.
    notrec = re.search(r"\(NotRecommended\s+\((.*?)\)\)\s+\(Interactions", rec001, re.DOTALL)
    assert notrec and "Resveratrol" in notrec.group(1)


# ── (iii) Personalization reorders across patients ───────────────────────────────

def test_berberine_omitted_for_senescence_patient(kb, rec001):
    # Berberine acts on the metabolic axis (InsulinResistance -> glucose/HbA1c); Patient001
    # presents NONE of those elevated, so Berberine is irrelevant and OMITTED entirely.
    assert _rec("Berberine", rec001) is None
    assert _is_empty(kb.run("!(supplement-for-patient &self Patient001 Berberine)"))


def test_berberine_recommended_for_metabolic_patient(rec002):
    # Same pool, different patient: Berberine IS relevant to the metabolic Patient002
    # (reaches their elevated FastingGlucose / HbA1c) and lands in Tier1.
    b = _rec("Berberine", rec002)
    assert b is not None
    assert b["tier"] == "Tier1HighConfidence"
    assert b["targets"] == {"FastingGlucose", "HbA1c"}


def test_senescence_supplements_omitted_for_metabolic_patient(rec002):
    # The senescence/inflammation supplements reach none of Patient002's elevated markers.
    for supp in ("Omega3", "Fisetin", "NMN", "Resveratrol"):
        assert _rec(supp, rec002) is None


# ── (iv) Interaction flagging against current medications ────────────────────────

def test_interaction_flagged_for_patient_on_metformin(rec002):
    # Patient002 is on metformin; the recommended Berberine shares its AMPK mechanism, so
    # the Berberine<->Metformin interaction is flagged.
    flag = re.search(
        r"\(InteractionFlag\s+(\S+)\s+(\S+)\s+\"", rec002)
    assert flag is not None
    assert {flag.group(1), flag.group(2)} == {"Berberine", "Metformin"}


def test_no_interaction_flagged_for_med_free_patient(rec001):
    # Patient001 has no CurrentMedication, so the Interactions block is empty — no
    # spurious flag even though supplements are recommended.
    interactions = re.search(r"\(Interactions\s+\((.*?)\)\)\s*\)\s*$", rec001, re.DOTALL)
    assert interactions is not None
    assert "InteractionFlag" not in interactions.group(1)


# ── (v) Provenance + determinism ─────────────────────────────────────────────────

def test_recommendations_carry_targeted_markers(rec001):
    # Provenance: each recommendation names the patient's elevated markers it targets.
    assert _rec("Omega3", rec001)["targets"] == {"CRP", "DNAmGDF15"}
    # Fisetin, a senolytic, reaches all three senescence-axis markers Patient001 presents.
    assert _rec("Fisetin", rec001)["targets"] == {"CRP", "DNAmGDF15", "DNAmPAI1"}


def test_omega3_score_is_relevance_times_safety(kb, rec001):
    # score = patient-relevance × safety-factor (GenerallyWellTolerated = 1.0 here).
    rel = float(_one(kb, "!(patient-relevance &self Patient001 Omega3)"))
    o = _rec("Omega3", rec001)
    assert o["safety_factor"] == pytest.approx(1.0)
    assert o["score"] == pytest.approx(rel * o["safety_factor"])


def test_safety_derates_the_score(kb, rec002):
    # Berberine's UseWithCaution safety label de-rates its score below its raw relevance
    # (0.8 factor) — the safety axis orders within a tier without changing the tier.
    rel = float(_one(kb, "!(patient-relevance &self Patient002 Berberine)"))
    b = _rec("Berberine", rec002)
    assert b["safety_label"] == "UseWithCaution"
    assert b["safety_factor"] == pytest.approx(0.8)
    assert b["score"] == pytest.approx(rel * 0.8)
    assert b["score"] < rel


def test_evidence_tier_map_and_veto(kb):
    # The tier authority: strong human evidence -> Tier1; weaker non-negative -> Tier2;
    # ITP_Negative -> the NotRecommended veto.
    assert _one(kb, "!(evidence-tier MultipleHumanTrials)") == "Tier1HighConfidence"
    assert _one(kb, "!(evidence-tier SingleHumanTrial)") == "Tier2Promising"
    assert _one(kb, "!(evidence-tier InVitro)") == "Tier2Promising"
    assert _one(kb, "!(evidence-tier ITP_Negative)") == "NotRecommended"


def test_recommendation_is_deterministic(kb):
    a = _one(kb, "!(recommend-supplements-patient &self Patient001)")
    b = _one(kb, "!(recommend-supplements-patient &self Patient001)")
    assert a == b
