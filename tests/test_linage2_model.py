"""The in-process LinAge2 (core/linage2_model.py) against the service it replaces.

The golden cases (tests/fixtures/linage2_golden.json) were scored by
Rejuve/LinAge2-Python's own `process_payload` — scripts/extract_linage2_model.py
runs it — on random partial panels, both sexes, ages 25-84, a third of them with a
full questionnaire. Matching them to 1e-9 years is the claim that the port IS the
model, not an approximation of it.

The rest pins what the port does differently on purpose: honest provenance for
derived and questionnaire inputs, cotinine on the training scale, refusals instead
of NaN — and that its output flows into the knowledge base unchanged.

    pytest tests/test_linage2_model.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.linage2_builder import build_linage2, feature_catalog  # noqa: E402
from core.linage2_model import (  # noqa: E402
    ASSUMED,
    DERIVED_FROM_IMPUTED,
    IMPUTED,
    MEASURED,
    MODEL_PATH,
    compute_linage2,
    load_model,
)
from core.patient_builder import PatientSpecError, build_patient  # noqa: E402

GOLDEN = json.loads((REPO / "tests" / "fixtures" / "linage2_golden.json").read_text(encoding="utf-8"))
TOL = 1e-9


def _years(result) -> dict[str, float]:
    return {fc["feature"]: fc["contribution_years"]
            for fc in result.response["metadata"]["feature_contributions"]}


# ═══════════════════════════ it is the model ═══════════════════════════════════

@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"{c['sex'][0]}{c['age']}")
def test_every_golden_case_matches_the_service(case):
    r = compute_linage2(sex=case["sex"], age=case["age"], labs=case["labs"],
                        questionnaire=case["questionnaire"])
    svc = case["service"]
    assert r.delta_years == pytest.approx(svc["delta_ba_ca"], abs=TOL)
    assert r.biological_age == pytest.approx(svc["biological_age"], abs=TOL)
    mine = _years(r)
    assert set(mine) == set(svc["feature_contributions"])
    for code, years in svc["feature_contributions"].items():
        assert mine[code] == pytest.approx(years, abs=TOL), code
    # the service's imputed list is lab inputs only; ours names the same lab inputs
    assert set(r.imputed_inputs) == set(svc["imputed_features"])


def test_the_golden_set_covers_what_it_claims():
    cases = GOLDEN["cases"]
    assert len(cases) >= 40
    assert {c["sex"] for c in cases} == {"Male", "Female"}
    assert any(c["questionnaire"] for c in cases) and any(not c["questionnaire"] for c in cases)
    assert any(c["age"] < 40 for c in cases)                       # the extrapolated range too
    assert {c["labs"]["LBXCOT"] for c in cases} == {0.0, 1.0, 2.0, 3.0}
    assert all(len(c["service"]["imputed_features"]) > 5 for c in cases)   # partial panels


def test_the_model_file_says_where_it_came_from():
    raw = load_model().raw
    assert raw["source"]["repository"] == "Rejuve/LinAge2-Python"
    assert raw["source"]["commit"] and len(raw["source"]["artifact_sha256"]) >= 12
    assert GOLDEN["source"]["commit"] == raw["source"]["commit"]
    assert len(raw["features"]) == 59 and len(raw["lab_inputs"]) == 57
    assert MODEL_PATH.stat().st_size < 200_000                    # numbers, not a dataset


def test_the_feature_set_is_exactly_the_knowledge_bases():
    """The KB's catalog (linage2_core.metta) and the model are one vocabulary."""
    assert set(load_model().features) == set(feature_catalog())


# ═══════════════════════════ linearity ════════════════════════════════════════

def test_each_input_moves_only_its_own_years():
    """Why a partial panel is still informative: an input's years depend on that
    input alone, so the years of what WAS measured are exact."""
    base = {"LBXGH": 5.4, "LBDSALSI": 44.0, "LBXCOT": 0.0, "LBXCRP": 0.2}
    a = _years(compute_linage2(sex="Male", age=58, labs=base))
    b = _years(compute_linage2(sex="Male", age=58, labs={**base, "LBXGH": 7.5}))
    changed = {k for k in a if abs(a[k] - b[k]) > 1e-12}
    assert changed == {"LBXGH"}
    assert b["LBXGH"] > a["LBXGH"]                                # higher HbA1c, older


def test_the_years_add_up_to_the_delta_with_the_age_term():
    r = compute_linage2(sex="Female", age=63, labs={"LBXCOT": 3.0, "LBDSALSI": 40.0})
    sexb = load_model().raw["sex"]["female"]
    age_term = (63 * 12 - sexb["mu_age_months"]) * sexb["w_age"] / 12
    assert sum(_years(r).values()) + age_term == pytest.approx(r.delta_years, abs=1e-9)


# ═══════════════════════════ honest provenance ════════════════════════════════

def test_derived_inputs_are_flagged_when_any_part_was_imputed():
    full = compute_linage2(sex="Male", age=50, labs={"LBDTCSI": 5.2, "LBDSTRSI": 1.4, "LBDHDLSI": 1.3,
                                                     "URXUMASI": 9.0, "URXUCRSI": 11000.0})
    assert full.provenance["LDLV"] == MEASURED and full.provenance["crAlbRat"] == MEASURED
    part = compute_linage2(sex="Male", age=50, labs={"LBDTCSI": 5.2, "URXUMASI": 9.0})
    assert part.provenance["LDLV"] == DERIVED_FROM_IMPUTED
    assert part.provenance["crAlbRat"] == DERIVED_FROM_IMPUTED
    none = compute_linage2(sex="Male", age=50, labs={})
    assert none.provenance["LDLV"] == IMPUTED
    flags = {fc["feature"]: fc["is_imputed"] for fc in part.response["metadata"]["feature_contributions"]}
    assert flags["LDLV"] is True and flags["crAlbRat"] is True


def test_an_ldl_can_be_given_directly():
    r = compute_linage2(sex="Male", age=50, labs={"LDLV": 3.4})
    assert r.provenance["LDLV"] == MEASURED


def test_unanswered_questionnaire_scores_are_assumed_and_said_so():
    r = compute_linage2(sex="Female", age=55, labs={})
    assert {r.provenance[s] for s in ("fs1Score", "fs2Score", "fs3Score")} == {ASSUMED}
    assert any("Questionnaire not answered" in w for w in r.warnings)
    answered = compute_linage2(sex="Female", age=55, labs={},
                               questionnaire={"BPQ020": 1, "HUQ010": 4})
    assert answered.provenance["fs1Score"] == MEASURED and answered.provenance["fs2Score"] == MEASURED
    assert answered.provenance["fs3Score"] == ASSUMED
    assert _years(answered)["fs1Score"] != _years(r)["fs1Score"]


def test_imputed_cotinine_is_on_the_training_scale():
    """The service imputed RAW cotinine (~0.1 ng/mL) as if it were a level; the
    model was trained on digitized levels, whose cohort median is 0."""
    model = load_model()
    assert all(v == 0.0 for v in model.raw["imputation"]["male"]["LBXCOT"])
    assert all(v == 0.0 for v in model.raw["imputation"]["female"]["LBXCOT"])
    r = compute_linage2(sex="Male", age=58, labs={})
    assert r.inputs_used["LBXCOT"] == 0.0 and r.provenance["LBXCOT"] == IMPUTED


def test_the_extrapolated_age_range_is_warned():
    young = compute_linage2(sex="Male", age=30, labs={})
    assert any("outside the ages LinAge2 was fitted on" in w for w in young.warnings)
    mid = compute_linage2(sex="Male", age=58, labs={})
    assert not any("outside the ages" in w for w in mid.warnings)


# ═══════════════════════════ refusals, not NaN ═════════════════════════════════

@pytest.mark.parametrize("kwargs, code", [
    (dict(sex=None, age=50, labs={}), "linage2_sex_required"),
    (dict(sex="Other", age=50, labs={}), "linage2_sex_required"),
    (dict(sex="Male", age=None, labs={}), "linage2_age_required"),
    (dict(sex="Male", age=15, labs={}), "linage2_age_out_of_range"),
    (dict(sex="Male", age=95, labs={}), "linage2_age_out_of_range"),
    (dict(sex="Male", age=50, labs={"LBXFOO": 1}), "linage2_unknown_input"),
    (dict(sex="Male", age=50, labs={"LBXGH": "high"}), "linage2_invalid_value"),
    (dict(sex="Male", age=50, labs={"LBXGH": float("nan")}), "linage2_invalid_value"),
    (dict(sex="Male", age=50, labs={"LBXSATSI": -3}), "linage2_invalid_value"),
    (dict(sex="Male", age=50, labs={"LBXCOT": 1.5}), "linage2_invalid_cotinine_level"),
    (dict(sex="Male", age=50, labs={}, questionnaire={"HUQ010": 9}), "linage2_invalid_questionnaire"),
    (dict(sex="Male", age=50, labs={}, questionnaire={"XYZ": 1}), "linage2_invalid_questionnaire"),
])
def test_bad_input_is_a_coded_refusal(kwargs, code):
    with pytest.raises(PatientSpecError) as excinfo:
        compute_linage2(**kwargs)
    assert excinfo.value.code == code


def test_a_zero_under_a_logarithm_folds_like_the_service():
    """ALT is log-transformed; 0 gives -inf, which the service folds to -z_max."""
    r = compute_linage2(sex="Male", age=50, labs={"LBXSATSI": 0.0})
    assar = _years(r)["LBXSATSI"]
    sexb = load_model().raw["sex"]["male"]
    i = load_model().features.index("LBXSATSI")
    assert assar == pytest.approx((-6.0 - sexb["mu_z"][i]) * sexb["w_months_per_sd"][i] / 12)


# ═══════════════════════════ into the knowledge base ═══════════════════════════

def test_the_response_is_what_build_linage2_already_reads():
    r = compute_linage2(sex="Male", age=58, labs={"LBXCOT": 3.0, "LBXGH": 6.4, "LBDSALSI": 41.0})
    built = build_linage2(r.response, "Caller_T", age=58)
    assert built.delta_years == pytest.approx(r.delta_years, abs=1e-6)
    assert len(built.contributions) == 59
    measured = {c.code for c in built.contributions if not c.imputed}
    assert measured == {"LBXCOT", "LBXGH", "LBDSALSI"}
    # the KB's own identity holds on the port's numbers
    total = sum(c.years for c in built.contributions)
    sexb = load_model().raw["sex"]["male"]
    assert total + (58 * 12 - sexb["mu_age_months"]) * sexb["w_age"] / 12 == pytest.approx(built.delta_years, abs=1e-5)


def test_a_patient_built_from_the_port_is_a_normal_linage2_patient():
    r = compute_linage2(sex="Male", age=58, labs={"LBXCOT": 3.0, "LBXGH": 6.4})
    built = build_patient({"id": "Port", "age": 58, "sex": "Male", "smoking": "CurrentSmoker",
                           "markers": {"HbA1c": 1.8}, "linage2": r.response})
    assert built.has_linage2 and "(LinAgeDelta Caller_Port " in built.atoms
    assert "LinAgeContribution" not in built.shared_atoms
