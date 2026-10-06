"""What was read -> a patient (core/patient_text.py): the units, the domain rules, and
the canonical-line reader.

The rules' job is to refuse rather than guess: a unit they cannot pin down is a question
back to the person, because a wrong unit is the commonest way to get a confident, wrong
biological age. These tests pin that contract, the unit table against the reference
cohort it serves, and the hand-off into LinAge2 and the KB — through `read_lines`, the
reader of canonical lines. Free text is the model's (tests/test_patient_read.py).

    pytest tests/test_patient_text.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.linage2_model import load_model  # noqa: E402
from core.patient_builder import PatientSpecError, build_patient  # noqa: E402
from core.patient_text import (  # noqa: E402
    ASSUMED_UNIT,
    DUPLICATE,
    EXAMPLES,
    LABS,
    NEEDS_UNIT,
    OK,
    OUT_OF_RANGE,
    SPECS,
    UNKNOWN_UNIT,
    normalise_unit,
    read_lines,
)


def _one(text: str):
    p = read_lines(text)
    assert len(p.readings) == 1, (text, p.readings, p.not_understood)
    return p.readings[0]


# ═══════════════════════════ the unit table serves the model ══════════════════

def test_every_lab_input_can_be_typed():
    model_inputs = set(load_model().lab_inputs) - {"LBXCOT"}          # cotinine: smoking words
    assert model_inputs <= set(SPECS)
    assert set(SPECS) - model_inputs == {"LDLV"}


def test_each_canonical_unit_is_the_reference_cohorts_unit():
    """The adult median must sit inside the typical range in the model's unit — the
    check that would have caught CRP documented as mg/L when NHANES 1999-2002 is mg/dL."""
    central = load_model().raw["nhanes_central"]
    for spec in LABS:
        if spec.code == "LDLV":
            continue
        lo, med, hi = central[spec.code]
        assert lo <= med <= hi, spec.code
        assert any(scale == 1 and offset == 0 for scale, offset in spec.units.values()), spec.code
    assert SPECS["LBXCRP"].unit == "mg/dL" and central["LBXCRP"][1] < 1.0      # mg/dL, not mg/L
    assert SPECS["LBDSALSI"].unit == "g/L" and 35 < central["LBDSALSI"][1] < 50


@pytest.mark.parametrize("typed, expected", [
    ("µmol/L", "umol/l"), ("μmol/L", "umol/l"), ("mcg/dL", "ug/dl"), ("x10^9/L", "e9/l"),
    ("10^9/L", "e9/l"), ("×10⁹/L", "e⁹/l") , ("10e3/uL", "e3/ul"), ("kg/m²", "kg/m2"), ("g/dL", "g/dl"),
])
def test_units_are_normalised(typed, expected):
    if "⁹" in expected:                       # superscripts are not digits; still accepted below
        assert normalise_unit(typed).startswith("e") or normalise_unit(typed).startswith("10")
    else:
        assert normalise_unit(typed) == expected


# ═══════════════════════════ refuse rather than guess ═════════════════════════

@pytest.mark.parametrize("text, code, value", [
    ("albumin 4.1 g/dL", "LBDSALSI", 41.0),
    ("Albumin: 41 g/L", "LBDSALSI", 41.0),
    ("CRP 3.1 mg/L", "LBXCRP", 0.31),
    ("hs-CRP 0.3 mg/dL", "LBXCRP", 0.3),
    ("HbA1c 6.4%", "LBXGH", 6.4),
    ("HbA1c 46 mmol/mol", "LBXGH", 46 * 0.09148 + 2.152),
    ("fasting glucose 5.6 mmol/L", "LBDSGLSI", 5.6),
    ("creatinine 88 µmol/L", "LBDSCRSI", 88.0),
    ("WBC 6.1 x10^9/L", "LBXWBCSI", 6.1),
    ("platelets 250 K/uL", "LBXPLTSI", 250.0),
    ("total cholesterol 200 mg/dL", "LBDTCSI", 200 * 0.02586),
    ("NT-proBNP 120 pg/mL", "SSBNP", 120.0),
    ("vitamin B12 400 pg/mL", "LBDB12SI", 400 * 0.738),
    ("LDL 130 mg/dL", "LDLV", 130 * 0.02586),
])
def test_a_typed_unit_is_converted(text, code, value):
    r = _one(text)
    assert (r.code, r.status) == (code, OK)
    assert r.value == pytest.approx(value, rel=1e-9)


def test_a_value_without_a_unit_is_taken_only_when_one_unit_fits():
    assert _one("HbA1c 6.4").status == OK                     # % is the only plausible reading
    r = _one("albumin 4.2")
    assert r.status == ASSUMED_UNIT and r.value == pytest.approx(42.0) and "g/dL" in r.note
    r = _one("glucose 105")
    assert r.status == ASSUMED_UNIT and r.value == pytest.approx(105 * 0.0555)
    r = _one("CRP 3.1")                                       # mg/L or mg/dL: both plausible
    assert r.status == NEEDS_UNIT and r.value is None and "mg/L" in r.note and "mg/dL" in r.note


def test_a_percentage_and_a_count_are_told_apart_by_value():
    assert _one("lymphocytes 30").code == "LBXLYPCT"
    assert _one("lymphocytes 1.9").code == "LBDLYMNO"
    assert _one("neutrophils 4.1 x10^9/L").code == "LBDNENO"
    assert _one("neutrophils 58 %").code == "LBXNEPCT"


def test_an_impossible_value_is_refused_with_the_unit_that_would_fit():
    r = _one("albumin 4.2 g/L")
    assert r.status == OUT_OF_RANGE and r.value == pytest.approx(4.2)
    assert "did you mean g/dL" in r.note
    r = _one("albumin 4.2 furlongs")
    assert r.status == UNKNOWN_UNIT and "g/dL" in r.note


def test_unknown_names_are_not_matched_loosely():
    p = read_lines("frobnicate 3\nalbuminuria 30 mg/g\nmy mood is great")
    assert p.readings == [] and len(p.not_understood) == 3


def test_a_value_given_twice_differently_is_refused():
    p = read_lines("albumin 4.1 g/dL\nalbumin 38 g/L")
    assert [r.status for r in p.readings] == [OK, DUPLICATE]
    assert not p.ok


def test_short_aliases_do_not_swallow_longer_words():
    p = read_lines("k 4.1 mmol/L\ndiagnoses: kidney disease")
    assert [r.code for r in p.readings] == ["LBXSKSI"]
    assert p.questionnaire["KIQ020"] == 1


# ═══════════════════════════ who the person is ════════════════════════════════

@pytest.mark.parametrize("text, age, sex, smoking, level", [
    ("58 year old male, current smoker", 58, "Male", "CurrentSmoker", 3),
    ("58 year old woman\nnever smoked", 58, "Female", "NeverSmoker", 0),
    ("age 66, male, former smoker", 66, "Male", "FormerSmoker", 0),
    ("41 year old female; occasional smoker", 41, "Female", "CurrentSmoker", 1),
    ("50 year old man, moderate smoker", 50, "Male", "CurrentSmoker", 2),
])
def test_demographics_and_smoking(text, age, sex, smoking, level):
    p = read_lines(text)
    assert (p.age, p.sex, p.smoking, p.cotinine_level) == (age, sex, smoking, level)
    assert not p.not_understood


def test_a_measured_cotinine_uses_the_training_bins():
    for ng, level in ((3, 0), (45, 1), (150, 2), (250, 3)):
        assert read_lines(f"cotinine {ng} ng/mL").cotinine_level == level


def test_age_and_sex_are_required_to_score():
    p = read_lines("albumin 4.1 g/dL")
    assert not p.ok and any("no age" in x for x in p.all_problems())
    assert any("no sex" in x for x in p.all_problems())
    with pytest.raises(PatientSpecError) as excinfo:
        p.to_patient()
    assert excinfo.value.code == "patient_text_unusable"


# ═══════════════════════════ the questionnaire ════════════════════════════════

def test_diagnoses_answer_the_whole_list():
    p = read_lines("diagnoses: hypertension, diabetes, osteoporosis")
    q = p.questionnaire
    assert (q["BPQ020"], q["DIQ010"], q["OSQ060"]) == (1, 1, 1)
    assert q["MCQ160F"] == 2                                   # not listed: No
    assert read_lines("diagnoses: prediabetes").questionnaire["DIQ010"] == 3
    assert read_lines("no known conditions").questionnaire["MCQ220"] == 2
    assert read_lines("no diabetes").questionnaire["DIQ010"] == 2


@pytest.mark.parametrize("text, item, answer", [
    ("self-rated health: fair", "HUQ010", 4),
    ("self-rated health: excellent", "HUQ010", 1),
    ("health compared to a year ago: worse", "HUQ020", 2),
    ("healthcare visits in the past year: 6", "HUQ050", 3),
    ("healthcare visits in the past year: 1", "HUQ050", 1),
    ("healthcare visits in the past year: 15", "HUQ050", 5),
])
def test_health_questions(text, item, answer):
    assert read_lines(text).questionnaire[item] == answer


def test_weight_and_height_give_a_bmi():
    p = read_lines("weight 180 lb\nheight 5'10\"")
    assert p.labs()["BMXBMI"] == pytest.approx(25.8, abs=0.1)
    assert read_lines("weight 70 kg\nheight 1.75 m").labs()["BMXBMI"] == pytest.approx(22.86, abs=0.01)


# ═══════════════════════════ into LinAge2 and the KB ══════════════════════════

@pytest.mark.parametrize("name", list(EXAMPLES))
def test_every_example_reads_cleanly_and_scores(name):
    p = read_lines(EXAMPLES[name])
    assert p.ok, p.all_problems()
    assert not p.not_understood
    payload, result = p.to_patient()
    built = build_patient(payload)
    assert built.has_linage2 and built.patient_id == "Caller_Me"
    measured = {c.code for c in built.linage2.contributions if not c.imputed}
    assert {"LBDSALSI", "LBXGH", "LBXCRP"} <= measured


def test_one_value_feeds_both_the_clock_and_the_witnesses():
    """CRP typed once is LinAge2's LBXCRP in mg/dL AND the KB's CRP marker in mg/L;
    a fasting glucose is the FastingGlucose witness, a plain glucose is not."""
    p = read_lines("50 year old man\nCRP 3.1 mg/L\nHbA1c 6.4 %\nfasting glucose 112 mg/dL")
    assert p.labs()["LBXCRP"] == pytest.approx(0.31)
    assert p.kb_markers() == {"CRP": {"value": 3.1, "unit": "mg/L"},
                              "HbA1c": {"value": 6.4, "unit": "%"},
                              "FastingGlucose": {"value": pytest.approx(112.0), "unit": "mg/dL"}}
    assert "FastingGlucose" not in read_lines("glucose 112 mg/dL").kb_markers()


def test_the_smoker_example_credits_cotinine_to_smoking():
    payload, result = read_lines(EXAMPLES["58-year-old smoker"]).to_patient()
    built = build_patient(payload)
    assert built.smoking == "CurrentSmoker"
    cot = next(c for c in built.linage2.contributions if c.code == "LBXCOT")
    assert not cot.imputed and cot.years > 5                 # level 3 on the training scale
    assert result.provenance["fs1Score"] == "measured"       # "diagnoses: hypertension"


def test_a_grimage_result_becomes_the_kb_clock_marker_not_a_linage2_input():
    p = read_lines("58 year old male\nGrimAge acceleration +4.5 years\nalbumin 4.1 g/dL")
    assert p.kb_markers()["AgeAccelGrim"] == {"value": 4.5, "unit": "years"}
    assert "AgeAccelGrim" not in p.labs() and not p.not_understood
    payload, _ = p.to_patient()
    built = build_patient(payload)
    assert built.can_predict_risk and built.has_linage2


# ═══════════════════════ contradictions and questions, never a last-wins ═══════

def test_two_different_ages_are_a_problem_not_a_last_wins():
    p = read_lines("58 year old male\nage 61")
    assert not p.ok and any("two different ages" in x for x in p.all_problems())


@pytest.mark.parametrize("text", [
    "58 year old male\ncotinine 5 ng/mL\ncurrent smoker",
    "58 year old male\ncurrent smoker\ncotinine 5 ng/mL",
])
def test_a_measured_cotinine_is_never_replaced_by_words_whatever_the_order(text):
    p = read_lines(text)
    assert (p.smoking, p.cotinine_level, p.ok) == ("CurrentSmoker", 0, True)
    assert any("measured value" in n for n in p.notes)


def test_a_list_of_diagnoses_answers_the_rest_no_and_one_note_says_so():
    p = read_lines("62 year old male\ndiagnoses: hypertension, diabetes\nno other conditions")
    assert (p.questionnaire["BPQ020"], p.questionnaire["DIQ010"], p.questionnaire["MCQ220"]) == (1, 1, 2)
    assert p.questionnaire_notes == ["diagnoses: hypertension, diabetes — every diagnosis not "
                                     "listed is answered No"]


def test_a_diagnosis_and_its_denial_contradict():
    p = read_lines("62 year old male\ndiagnoses: diabetes\nno diabetes")
    assert not p.ok and any("differently" in x for x in p.all_problems())
    assert read_lines("62 year old male\nno diabetes\ndiagnoses: prediabetes").questionnaire["DIQ010"] == 3


def test_a_glucose_not_marked_fasting_is_said_not_to_be_a_witness():
    p = read_lines("60 year old female\nglucose 140 mg/dL")
    assert "FastingGlucose" not in p.kb_markers() and any("not marked fasting" in n for n in p.notes)
    assert not any("not marked fasting" in n
                   for n in read_lines("60 year old female\nfasting glucose 140 mg/dL").notes)


@pytest.mark.parametrize("text", ["weight 150", "height 165", "height 1.75"])
def test_weight_and_height_need_their_unit(text):
    p = read_lines("40 year old female\n" + text)
    assert any("give the" in x and "unit" in x for x in p.all_problems())


def test_an_impossible_bmi_from_weight_and_height_is_refused():
    p = read_lines("58 year old male\nweight 30 kg\nheight 220 cm")
    assert any("BMI" in x for x in p.all_problems())


def test_cotinine_needs_ng_per_ml_or_the_word_level():
    assert any("cotinine" in x for x in read_lines("58 year old male\ncotinine 3").all_problems())
    assert read_lines("cotinine level 3").cotinine_level == 3
    assert read_lines("cotinine 3 ng/mL").cotinine_level == 0


def test_a_grimage_clock_age_is_not_taken_as_an_acceleration():
    p = read_lines("42 year old female\nGrimAge 46.3")
    assert "AgeAccelGrim" not in p.kb_markers() and any("ACCELERATION" in x for x in p.all_problems())
    assert read_lines("GrimAge acceleration -3 years").kb_markers()["AgeAccelGrim"]["value"] == -3
    assert any("±30" in x for x in read_lines("GrimAge acceleration +46 years").all_problems())


def test_urea_is_not_read_with_the_urea_nitrogen_factor():
    assert _one("urea 40 mg/dL").value == pytest.approx(40 * 0.1665)
    assert _one("BUN 18 mg/dL").value == pytest.approx(18 * 0.357)


def test_an_abnormal_value_is_never_quietly_read_as_normal_in_another_unit():
    """Hemoglobin 9.5 is anaemic in g/dL and normal-looking in mmol/L: ask."""
    r = _one("hemoglobin 9.5")
    assert r.status == NEEDS_UNIT and r.value is None
    assert _one("hemoglobin 9.5 g/dL").value == pytest.approx(9.5)


@pytest.mark.parametrize("text", ["lymphocytes 30 percent", "HbA1c 6.3 percent", "WBC 7.2 10⁹/L",
                                  "RBC 4.8 10¹²/L", "platelets 250 10³/µL"])
def test_units_the_reader_displays_are_units_it_accepts(text):
    assert _one(text).status == OK


@pytest.mark.parametrize("line", ["HbA1c 12 %", "fasting glucose 250 mg/dL"])
def test_a_real_but_extreme_value_still_builds_and_witnesses(line):
    p = read_lines("58 year old male\n" + line)
    payload, _ = p.to_patient()
    built = build_patient(payload)
    marker = next(m for m in built.markers if m.name in ("HbA1c", "FastingGlucose"))
    assert marker.status == "Elevated" and p.witness_notes


def test_an_unknown_lab_next_to_diagnoses_is_only_not_understood():
    p = read_lines("58 year old male\nvitamin D 30 ng/mL\ndiagnoses: hypertension")
    assert p.ok and p.not_understood == ["vitamin D 30 ng/mL"]


# ═══════════════════════════ statements, spans and typed problems ═══════════════
# Every statement and every problem says where it is and what kind it is, so the tab and
# the API can point at what was typed. The strings a caller sees are plain: a Problem is
# still a str.

def test_every_statement_points_at_its_text():
    text = "58 year old male, current smoker\n  - albumin 4.1 g/dL; CRP 3.1 mg/L\ndiagnoses: asthma,  arthritis"
    p = read_lines(text)
    lines = text.splitlines()
    for st in p.statements:
        typed = lines[st.line][st.start:st.end]
        assert typed == st.text or typed.replace(",  ", ", ") == st.text, (st, typed)
    assert [st.text for st in p.statements] == [
        "58 year old male", "current smoker", "albumin 4.1 g/dL", "CRP 3.1 mg/L",
        "diagnoses: asthma,  arthritis"]
    assert p.statements[0].facts == {"age": 58.0, "sex": "Male"}
    assert p.statements[1].facts["smoking"] == ("CurrentSmoker", 3)
    assert p.statements[2].facts["labs"] == {"LBDSALSI": pytest.approx(41.0)}
    assert p.readings[0].statement == 2
    assert p.statements[4].facts["conditions"] == {"MCQ010": 1, "MCQ160A": 1}
    assert {st.outcome for st in p.statements} == {"read"}


def test_problems_are_still_strings_with_a_kind_and_a_statement():
    import copy
    import json
    import pickle

    from core.patient_text import PROBLEM_KINDS, Problem

    p = read_lines("58 year old\nCRP 3.1\nfoo bar\ndiagnoses: diabetes\nno diabetes")
    problems = p.all_problems()
    assert all(isinstance(x, Problem) and isinstance(x, str) for x in problems)
    by_kind = {x.kind: x for x in problems}
    assert set(by_kind) <= set(PROBLEM_KINDS)
    assert by_kind["missing"] == "no sex found ('male' or 'female'): LinAge2 has a separate model for each"
    assert by_kind["unit"].topic == "lab" and p.statements[by_kind["unit"].statements[0]].text == "CRP 3.1"
    contradiction = by_kind["contradiction"]
    assert contradiction.topic == "condition" and p.statements[contradiction.statements[0]].text == "no diabetes"
    assert p.not_understood == ["foo bar"] and p.not_understood[0].statement == 2
    assert [st.outcome for st in p.statements] == ["read", "refused", "not_understood", "read", "refused"]
    # what the API and the tab do with them: serialise, copy (gr.State), pickle
    assert json.loads(json.dumps(problems)) == [str(x) for x in problems]
    for clone in (copy.deepcopy(p), pickle.loads(pickle.dumps(p))):
        assert [(x.kind, x.statements) for x in clone.all_problems()] == \
            [(x.kind, x.statements) for x in problems]


# ═══════════════ the 10-year CHD risk is a first-event model ═════════════════════
#
# Someone who reports coronary heart disease, a heart attack or angina gets the same incident-CHD
# number as someone who does not (the history is not an input), so the reader flags it for a note
# beside that number. Heart failure is a different event and is not flagged.

def test_reported_chd_a_heart_attack_and_angina_are_flagged_and_heart_failure_is_not():
    from core.patient_builder import PREVALENT_CHD
    base = "58 year old male\nCRP 3.1 mg/L\n"
    payload, _ = read_lines(base + "diagnoses: heart attack, angina, coronary heart disease").to_patient("X")
    assert payload["prevalent_chd"] == ["coronary heart disease", "angina", "heart attack"]   # item order, not typed order
    assert set(payload["prevalent_chd"]) <= set(PREVALENT_CHD)             # the builder's allow-list
    for text in ("diagnoses: heart failure", "diagnoses: hypertension", "no heart attack",
                 "diagnoses: hypertension\nno angina"):
        payload, _ = read_lines(base + text).to_patient("X")
        assert "prevalent_chd" not in payload, text
    assert "prevalent_chd" not in read_lines(base).to_patient("X")[0]


# ═══════════ RDW and albumin as a z against LinAge2's young reference (#7) ═══════════

def test_young_reference_z_puts_the_cut_points_where_the_report_said():
    """z = 1 at RDW 13.09 % (men) / 13.27 % (women) and at albumin 43 g/L (men) / 41 g/L (women)."""
    from core.linage2_model import young_reference_z
    assert young_reference_z("LBXRDW", 13.09, "Male") == pytest.approx(1.0, abs=0.01)
    assert young_reference_z("LBXRDW", 13.27, "Female") == pytest.approx(1.0, abs=0.01)
    assert -young_reference_z("LBDSALSI", 43.0, "Male") == pytest.approx(1.0, abs=0.02)
    assert -young_reference_z("LBDSALSI", 41.0, "Female") == pytest.approx(1.0, abs=0.02)
    assert young_reference_z("LBXRDW", 13.0, "Male") < 1.0 < young_reference_z("LBXRDW", 13.2, "Male")
    with pytest.raises(KeyError):
        young_reference_z("LBXCRP_NOT_A_CODE", 1.0, "Male")


def test_an_elevated_rdw_and_a_low_albumin_are_passed_as_z_with_the_note_that_says_what_that_means():
    p = read_lines(EXAMPLES["58-year-old smoker"])
    m = p.kb_markers()
    assert m["RDW"]["z"] == pytest.approx(2.70, abs=0.01) and m["LowSerumAlbumin"]["z"] == pytest.approx(1.69, abs=0.01)
    assert set(m["RDW"]) == {"z"}                                    # z only: the KB has no raw reference for them
    notes = " ".join(p.witness_notes)
    assert "'RDW 14.1 %' counts as high here" in notes and "'albumin 4.1 g/dL' counts as low here" in notes
    assert "stricter than a laboratory range" in notes and "not age-adjusted" in notes
    assert "not a finding" in notes and "Low protein intake and a recent meal also lower albumin" in notes


def test_a_value_that_is_not_beyond_the_reference_is_no_witness_and_says_nothing():
    p = read_lines(EXAMPLES["healthy 45-year-old woman"])
    assert "RDW" not in p.kb_markers() and "LowSerumAlbumin" not in p.kb_markers() and p.witness_notes == []


@pytest.mark.parametrize("sex, rdw, expected", [("male", 13.0, False), ("male", 13.2, True), ("female", 13.2, False),
                                                ("female", 13.4, True)])
def test_rdw_is_a_witness_just_above_the_sex_specific_reference(sex, rdw, expected):
    p = read_lines(f"58 year old {sex}\nRDW {rdw} %")
    assert ("RDW" in p.kb_markers()) is expected


@pytest.mark.parametrize("sex, g_per_dl, expected", [("male", 4.4, False), ("male", 4.2, True), ("female", 4.2, False),
                                                     ("female", 4.0, True)])
def test_albumin_is_a_witness_just_below_the_sex_specific_reference(sex, g_per_dl, expected):
    p = read_lines(f"58 year old {sex}\nalbumin {g_per_dl} g/dL")
    assert ("LowSerumAlbumin" in p.kb_markers()) is expected


@pytest.mark.parametrize("extra, withheld", [
    ("hemoglobin 11.5 g/dL", True), ("hemoglobin 14.5 g/dL", False), ("ferritin 20 ug/L", True), ("ferritin 80 ug/L", False),
    ("vitamin b12 120 pmol/L", True), ("vitamin b12 300 pmol/L", False), ("folate 8 nmol/L", True), ("folate 25 nmol/L", False),
])
def test_a_raised_rdw_is_withheld_when_an_anaemia_or_a_deficiency_explains_it(extra, withheld):
    p = read_lines(f"58 year old male\nRDW 14.5 %\n{extra}")
    m = p.kb_markers()
    assert ("RDW" not in m) is withheld
    assert any("was not passed on as a sign of inflammation" in n for n in p.witness_notes) is withheld
    assert p.labs()["LBXRDW"] == 14.5                                # LinAge2 still has it as typed


def test_the_hemoglobin_limit_is_sex_specific():
    assert "RDW" not in read_lines("58 year old male\nRDW 14.5 %\nhemoglobin 12.5 g/dL").kb_markers()
    assert "RDW" in read_lines("58 year old female\nRDW 14.5 %\nhemoglobin 12.5 g/dL").kb_markers()


def test_albumin_is_never_gated_on_a_deficiency():
    assert "LowSerumAlbumin" in read_lines("58 year old male\nalbumin 4.0 g/dL\nferritin 20 ug/L").kb_markers()


# ═══════════ fasting triglycerides as a witness (#12) ═══════════

@pytest.mark.parametrize("typed, elevated", [
    ("fasting triglycerides 149 mg/dL", False), ("fasting triglycerides 150 mg/dL", True), ("fasting tg 190 mg/dL", True),
    ("fasting triglyceride 1.6 mmol/L", False), ("fasting triglycerides 1.7 mmol/L", True),
])
def test_a_fasting_triglyceride_of_150_mg_dl_or_more_is_elevated(typed, elevated):
    from core.patient_builder import build_patient
    p = read_lines(f"58 year old male\n{typed}")
    m = p.kb_markers()
    assert set(m["Triglycerides"]) == {"value", "unit"} and m["Triglycerides"]["unit"] == "mg/dL"
    built = build_patient({"age": 58, "sex": "Male", "markers": {"Triglycerides": m["Triglycerides"]}})
    assert (built.witnesses == ["Triglycerides"]) is elevated, (typed, m)


def test_a_triglyceride_not_marked_fasting_is_no_witness_and_says_why():
    p = read_lines("58 year old male\ntriglycerides 190 mg/dL")
    assert "Triglycerides" not in p.kb_markers()
    assert any("triglycerides were not marked fasting" in n and "fasting triglycerides" in n for n in p.notes)
    assert p.labs()["LBDSTRSI"] == pytest.approx(190 * 0.01129)       # LinAge2 still has it, as typed
    assert not any("triglycerides were not marked fasting" in n
                   for n in read_lines("58 year old male\nfasting triglycerides 190 mg/dL").notes)


def test_a_triglyceride_typed_twice_is_a_witness_whichever_line_came_first():
    for text in ("58 year old male\nfasting triglycerides 190 mg/dL\ntriglycerides 190 mg/dL",
                 "58 year old male\ntriglycerides 190 mg/dL\nfasting triglycerides 190 mg/dL"):
        p = read_lines(text)
        assert p.kb_markers()["Triglycerides"] == {"value": 190.0, "unit": "mg/dL"}, text
        assert not any("not marked fasting" in n for n in p.notes), (text, p.notes)


def test_the_not_fasting_note_does_not_claim_linage2_uses_a_triglyceride_it_does_not():
    plain = read_lines("58 year old male\ntriglycerides 190 mg/dL")
    assert any("in the calculated LDL" in n for n in plain.notes)
    with_ldl = read_lines("58 year old male\ntriglycerides 190 mg/dL\nLDL 130 mg/dL")
    note = next(n for n in with_ldl.notes if "not marked fasting" in n)
    assert "in the calculated LDL" not in note and "does not use them (you gave an LDL)" in note


@pytest.mark.parametrize("sex, rdw, capped", [("male", 20, True), ("male", 25, True), ("female", 25, True), ("female", 30, True),
                                              ("female", 22, False)])
def test_an_rdw_beyond_the_references_range_is_passed_on_as_the_largest_z_the_kb_accepts(sex, rdw, capped):
    from core.patient_builder import Z_LIMIT
    from core.patient_context import build_caller_patient
    p = read_lines(f"58 year old {sex}\nRDW {rdw} %")
    z = p.kb_markers()["RDW"]["z"]
    assert (z == Z_LIMIT) is capped, (sex, rdw, z)
    built = build_caller_patient(p.to_patient("Me")[0], ())          # the build does not refuse it
    assert built.witnesses == ["RDW"]
    assert any("passed on as z 12" in n for n in p.witness_notes) is capped


@pytest.mark.parametrize("sex, lab, below, at_or_above", [
    ("male", "hemoglobin {} g/dL", 12.9, 13.0), ("female", "hemoglobin {} g/dL", 11.9, 12.0),
    ("male", "ferritin {} ug/L", 29, 30), ("female", "ferritin {} ug/L", 29, 30),
    ("male", "vitamin b12 {} pmol/L", 147, 148), ("female", "vitamin b12 {} pmol/L", 147, 148),
    ("male", "folate {} nmol/L", 9.5, 10), ("female", "folate {} nmol/L", 9.5, 10),
])
def test_each_rdw_gate_limit_withholds_just_below_it_and_not_at_it(sex, lab, below, at_or_above):
    def kept(value):
        return "RDW" in read_lines(f"58 year old {sex}\nRDW 14.5 %\n{lab.format(value)}").kb_markers()
    assert not kept(below) and kept(at_or_above), (sex, lab)


def test_an_rdw_just_above_the_z_one_cut_point_is_a_witness_and_just_below_is_not():
    assert "RDW" in read_lines("58 year old male\nRDW 13.12 %").kb_markers()
    assert "RDW" not in read_lines("58 year old male\nRDW 13.07 %").kb_markers()
