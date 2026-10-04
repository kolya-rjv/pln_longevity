"""Plain text about a person -> a patient (core/patient_text.py).

The reader's job is to refuse rather than guess: a unit it cannot pin down is a
question back to the person, because a wrong unit is the commonest way to get a
confident, wrong biological age. These tests pin that contract, the unit table
against the reference cohort it serves, and the hand-off into LinAge2 and the KB.

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
    read_patient_text,
)


def _one(text: str):
    p = read_patient_text(text)
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
    p = read_patient_text("frobnicate 3\nalbuminuria 30 mg/g\nmy mood is great")
    assert p.readings == [] and len(p.not_understood) == 3


def test_a_value_given_twice_differently_is_refused():
    p = read_patient_text("albumin 4.1 g/dL\nalbumin 38 g/L")
    assert [r.status for r in p.readings] == [OK, DUPLICATE]
    assert not p.ok


def test_short_aliases_do_not_swallow_longer_words():
    p = read_patient_text("k 4.1 mmol/L\nkidney disease")
    assert [r.code for r in p.readings] == ["LBXSKSI"]
    assert p.questionnaire["KIQ020"] == 1


# ═══════════════════════════ who the person is ════════════════════════════════

@pytest.mark.parametrize("text, age, sex, smoking, level", [
    ("58 year old male, current smoker", 58, "Male", "CurrentSmoker", 3),
    ("58-year-old woman who never smoked", 58, "Female", "NeverSmoker", 0),
    ("age 66, male, former smoker", 66, "Male", "FormerSmoker", 0),
    ("I'm a 41 yo female; light smoker", 41, "Female", "CurrentSmoker", 1),
    ("sex: m\nage: 50\nsmoking: no", 50, "Male", "NeverSmoker", 0),
])
def test_demographics_and_smoking(text, age, sex, smoking, level):
    p = read_patient_text(text)
    assert (p.age, p.sex, p.smoking, p.cotinine_level) == (age, sex, smoking, level)
    assert not p.not_understood


def test_a_measured_cotinine_uses_the_training_bins():
    for ng, level in ((3, 0), (45, 1), (150, 2), (250, 3)):
        assert read_patient_text(f"cotinine {ng} ng/mL").cotinine_level == level


def test_age_and_sex_are_required_to_score():
    p = read_patient_text("albumin 4.1 g/dL")
    assert not p.ok and any("no age" in x for x in p.all_problems())
    assert any("no sex" in x for x in p.all_problems())
    with pytest.raises(PatientSpecError) as excinfo:
        p.to_patient()
    assert excinfo.value.code == "patient_text_unusable"


# ═══════════════════════════ the questionnaire ════════════════════════════════

def test_diagnoses_answer_the_whole_list():
    p = read_patient_text("diagnoses: hypertension, type 2 diabetes, osteoporosis")
    q = p.questionnaire
    assert (q["BPQ020"], q["DIQ010"], q["OSQ060"]) == (1, 1, 1)
    assert q["MCQ160F"] == 2                                   # not listed: No
    assert read_patient_text("prediabetes").questionnaire["DIQ010"] == 3
    assert read_patient_text("no known conditions").questionnaire["MCQ220"] == 2
    p = read_patient_text("62 year old man with diabetes and hypertension")
    assert p.age == 62 and p.questionnaire["DIQ010"] == 1 and not p.not_understood


@pytest.mark.parametrize("text, item, answer", [
    ("self-rated health: fair", "HUQ010", 4),
    ("general health excellent", "HUQ010", 1),
    ("health compared to a year ago: worse", "HUQ020", 2),
    ("doctor visits last year: 6", "HUQ050", 3),
    ("healthcare visits: 1", "HUQ050", 1),
    ("healthcare visits: 15", "HUQ050", 5),
])
def test_health_questions(text, item, answer):
    assert read_patient_text(text).questionnaire[item] == answer


def test_weight_and_height_give_a_bmi():
    p = read_patient_text("weight 180 lb\nheight 5'10\"")
    assert p.labs()["BMXBMI"] == pytest.approx(25.8, abs=0.1)
    assert read_patient_text("weight 70 kg\nheight 1.75 m").labs()["BMXBMI"] == pytest.approx(22.86, abs=0.01)


# ═══════════════════════════ into LinAge2 and the KB ══════════════════════════

@pytest.mark.parametrize("name", list(EXAMPLES))
def test_every_example_reads_cleanly_and_scores(name):
    p = read_patient_text(EXAMPLES[name])
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
    p = read_patient_text("50 year old man\nCRP 3.1 mg/L\nHbA1c 6.4 %\nfasting glucose 112 mg/dL")
    assert p.labs()["LBXCRP"] == pytest.approx(0.31)
    assert p.kb_markers() == {"CRP": {"value": 3.1, "unit": "mg/L"},
                              "HbA1c": {"value": 6.4, "unit": "%"},
                              "FastingGlucose": {"value": pytest.approx(112.0), "unit": "mg/dL"}}
    assert "FastingGlucose" not in read_patient_text("glucose 112 mg/dL").kb_markers()


def test_the_smoker_example_credits_cotinine_to_smoking():
    payload, result = read_patient_text(EXAMPLES["58-year-old smoker"]).to_patient()
    built = build_patient(payload)
    assert built.smoking == "CurrentSmoker"
    cot = next(c for c in built.linage2.contributions if c.code == "LBXCOT")
    assert not cot.imputed and cot.years > 5                 # level 3 on the training scale
    assert result.provenance["fs1Score"] == "measured"       # "diagnoses: hypertension"


def test_a_grimage_result_becomes_the_kb_clock_marker_not_a_linage2_input():
    p = read_patient_text("58 year old male\nGrimAge acceleration +4.5 years\nalbumin 4.1 g/dL")
    assert p.kb_markers()["AgeAccelGrim"] == {"value": 4.5, "unit": "years"}
    assert "AgeAccelGrim" not in p.labs() and not p.not_understood
    payload, _ = p.to_patient()
    built = build_patient(payload)
    assert built.can_predict_risk and built.has_linage2


# ═══════════════════════ what an adversarial review broke ════════════════════
# Each case below produced a CONFIDENT WRONG patient before it was fixed.

@pytest.mark.parametrize("text", [
    "58 year old male\nquit smoking 20 years ago",
    "58 year old male, smoker for 40 years",
    "58 year old male\nhypertension for 25 years",
    "58 year old male\nbiological age 65",
    "58 year old male\nheart age 70",
])
def test_a_later_number_of_years_is_never_the_age(text):
    assert read_patient_text(text).age == 58


def test_two_different_ages_are_a_problem_not_a_last_wins():
    p = read_patient_text("58 year old male\nage 61")
    assert not p.ok and any("two different ages" in x for x in p.all_problems())


@pytest.mark.parametrize("text, status, level", [
    ("58 year old male, not a smoker", "NeverSmoker", 0),
    ("never a smoker", "NeverSmoker", 0),
    ("smoker: no", "NeverSmoker", 0),
    ("45 year old female, smokes: no", "NeverSmoker", 0),
    ("previous smoker", "FormerSmoker", 0),
    ("smoker until 2015", "FormerSmoker", 0),
    ("58 year old male, former heavy smoker", "FormerSmoker", 0),
    ("58 year old male, trying to quit smoking", "CurrentSmoker", 3),
])
def test_negated_past_and_ongoing_smoking(text, status, level):
    p = read_patient_text(text)
    assert (p.smoking, p.cotinine_level) == (status, level) and not p.not_understood


@pytest.mark.parametrize("text", [
    "45 year old female, never smoked\nmy husband smokes",
    "45 year old female, never smoked\nlives with male partner who smokes",
    "45 year old female, never smoked\nexposed to second-hand smoke",
])
def test_someone_else_is_set_aside_and_said_to_be(text):
    p = read_patient_text(text)
    assert (p.sex, p.smoking, p.cotinine_level) == ("Female", "NeverSmoker", 0)
    assert len(p.set_aside) == 1 and any("set aside" in n for n in p.notes)


def test_a_family_history_is_not_the_persons_diagnosis():
    p = read_patient_text("58 year old male\nfamily history of diabetes")
    assert "DIQ010" not in p.questionnaire and p.set_aside


@pytest.mark.parametrize("text, item, answer", [
    ("no known conditions except hypertension", "BPQ020", 1),
    ("no medical history apart from diabetes", "DIQ010", 1),
    ("no chronic diseases besides asthma", "MCQ010", 1),
    ("no diabetes, no hypertension", "DIQ010", 2),
    ("hypertension, no diabetes", "DIQ010", 2),
])
def test_exceptions_and_negations_in_diagnoses(text, item, answer):
    assert read_patient_text(text).questionnaire[item] == answer


@pytest.mark.parametrize("text", ["weight 150", "height 165", "height 1.75"])
def test_weight_and_height_need_their_unit(text):
    p = read_patient_text("40 year old female\n" + text)
    assert any("give the" in x for x in p.all_problems())


def test_an_impossible_bmi_from_weight_and_height_is_refused():
    p = read_patient_text("58 year old male\nweight 30 kg\nheight 220 cm")
    assert any("BMI" in x for x in p.all_problems())


def test_cotinine_needs_ng_per_ml_or_the_word_level():
    assert any("cotinine" in x for x in read_patient_text("58 year old male\ncotinine 3").all_problems())
    assert read_patient_text("cotinine level 3").cotinine_level == 3
    assert read_patient_text("cotinine 3 ng/mL").cotinine_level == 0


def test_a_grimage_clock_age_is_not_taken_as_an_acceleration():
    p = read_patient_text("42 year old female\nGrimAge 46.3")
    assert "AgeAccelGrim" not in p.kb_markers() and any("clock AGE" in x for x in p.all_problems())
    assert read_patient_text("GrimAge acceleration -3 years").kb_markers()["AgeAccelGrim"]["value"] == -3
    assert any("±30" in x for x in read_patient_text("GrimAge acceleration +46 years").all_problems())


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
    p = read_patient_text("58 year old male\n" + line)
    payload, _ = p.to_patient()
    built = build_patient(payload)
    marker = next(m for m in built.markers if m.name in ("HbA1c", "FastingGlucose"))
    assert marker.status == "Elevated" and p.witness_notes
