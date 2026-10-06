"""Reading the My Patient text with a model (core/patient_read.py) — with recorded model
outputs, never a live call.

The model reads the text; code checks every item against it before anything is used:
the quote is in the text, the number and the unit are in the quote, nothing inside a
statement about someone else is the person's. What passes goes through the same domain
rules as canonical lines (core.patient_text.assemble), so a unit is still never guessed.
The items below are what a model returns — or what a hostile one could.

    pytest tests/test_patient_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.patient_extract import ExtractError, Extraction, check_output  # noqa: E402
from core.patient_read import read_patient  # noqa: E402
from core.patient_text import read_lines  # noqa: E402


class Recorded:
    """An extractor that returns what a model once returned (or could)."""
    model = "recorded"

    def __init__(self, *items):
        self.items = [dict(i) for i in items]
        out = {"items": self.items}
        assert check_output(out) is None, check_output(out)       # schema-valid, as strict mode makes it

    def __call__(self, text):
        return Extraction(list(self.items), "recorded", 1.0, {"prompt_tokens": 1})


def age(quote, n):
    return {"quote": quote, "kind": "age", "number": n}


def sex(quote, s):
    return {"quote": quote, "kind": "sex", "sex": s}


def smoking(quote, status, amount="unstated", other="none"):
    return {"quote": quote, "kind": "smoking", "status": status, "amount": amount, "other_nicotine": other}


def lab(quote, group, number, unit=""):
    return {"quote": quote, "kind": "lab", "lab": group, "number": number, "unit": unit}


def weight(quote, number, unit):
    return {"quote": quote, "kind": "weight", "number": number, "unit": unit}


def height(quote, number, unit):
    return {"quote": quote, "kind": "height", "number": number, "unit": unit}


def condition(quote, key, answer="yes"):
    return {"quote": quote, "kind": "condition", "condition": key, "answer": answer}


def med(quote, taking="now"):
    return {"quote": quote, "kind": "medication", "drug": "Metformin", "taking": taking}


def k(quote, kind, **kw):
    return {"quote": quote, "kind": kind, **kw}


def read(text, *items):
    return read_patient(text, Recorded(*items))


# ═══════════════════════════ without the model ═════════════════════════════════

def test_without_an_extractor_the_text_is_read_as_canonical_lines():
    text = "58 year old male, current smoker\nalbumin 4.1 g/dL"
    r = read_patient(text)
    assert r.reader == "lines" and "one fact per line" in r.header()
    assert r.parsed.as_dict() == read_lines(text).as_dict()


@pytest.mark.parametrize("error", [ExtractError("no_key", "no OPENAI_API_KEY"), ExtractError("timeout", "slow"),
                                   ExtractError("refused", "declined"), ExtractError("config", "bad request")])
def test_when_the_model_cannot_read_the_text_is_read_as_lines_and_says_why(error):
    class Failing:
        model = "x"

        def __call__(self, text):
            raise error

    r = read_patient("58 year old male\nalbumin 4.1 g/dL", Failing())
    assert r.reader == "lines" and r.model_error is error and error.message in r.header()
    assert r.parsed.ok and r.parsed.age == 58


def test_an_unexpected_extractor_failure_is_an_upstream_error_not_a_crash():
    class Broken:
        model = "x"

        def __call__(self, text):
            raise ValueError("boom")

    r = read_patient("58 year old male", Broken())
    assert r.reader == "lines" and r.model_error.code == "upstream" and "boom" in r.model_error.message


# ═══════════════════════════ how people type it ════════════════════════════════

HER_TEXT = "A 35 year old female  height 5'3'' inch weight 135 lbs "


def test_height_and_weight_on_the_age_line_are_read():
    """The text that started this: one line, no commas, the inch marks and 'inch' both. The
    model quoted the height without its label (live), which leaves 'height' alone: not 'not used'."""
    r = read(HER_TEXT, age("35 year old female", "35"), sex("35 year old female", "female"),
             height("5'3'' inch", "5'3", "ft-in"), weight("weight 135 lbs", "135", "lb"))
    p = r.parsed
    assert (p.age, p.sex) == (35, "Female") and p.ok and not r.discarded
    assert p.height_cm == pytest.approx(160.02) and p.weight_kg == pytest.approx(61.235, abs=1e-3)
    assert "BMI 23.9 from weight and height" in p.notes and not p.not_understood


@pytest.mark.parametrize("typed, number", [
    ("5'3''", "5'3"), ("5'3\"", "5'3''"), ("5 ft 3 in", "5'3"), ("5 feet 3 inches", "5'3"), ("5′3″", "5'3\""),
])
def test_feet_and_inches_however_they_are_written(typed, number):
    text = f"35 year old female, {typed}, 135lbs"
    p = read(text, age("35 year old female", "35"), sex("35 year old female", "female"),
             height(typed, number, "ft-in"), weight("135lbs", "135", "lb")).parsed
    assert p.ok and p.height_cm == pytest.approx(160.02) and p.weight_kg == pytest.approx(61.235, abs=1e-3)


def test_what_is_not_about_the_person_is_listed_as_not_used():
    text = "A 35 year old female 5'3'' 135lbs wants to build muscle and bone density. what supplements?"
    r = read(text, age("35 year old female", "35"), sex("35 year old female", "female"),
             height("5'3''", "5'3", "ft-in"), weight("135lbs", "135", "lb"))
    p = r.parsed
    assert p.ok and p.not_understood == ["wants to build muscle and bone density. what supplements?"]
    assert p.statements[p.not_understood[0].statement].outcome == "not_understood"


def test_two_facts_from_one_quote_are_one_statement():
    p = read("58 yo M", age("58 yo M", "58"), sex("58 yo M", "male")).parsed
    assert (p.age, p.sex) == (58, "Male") and len(p.statements) == 1
    assert p.statements[0].facts == {"age": 58.0, "sex": "Male"}


def test_blood_pressure_is_two_items_on_one_quote():
    p = read("58 year old male, BP 142/88", age("58 year old male", "58"), sex("58 year old male", "male"),
             lab("BP 142/88", "systolic blood pressure", "142"),
             lab("BP 142/88", "diastolic blood pressure", "88")).parsed
    assert p.ok and p.labs()["BPXSAR"] == 142 and p.labs()["BPXDAR"] == 88


# ═══════════════════════════ the checks against the text ══════════════════════

def test_a_quote_that_is_not_in_the_text_is_dropped():
    r = read("58 year old male", age("58 year old male", "58"), sex("58 year old male", "male"),
             lab("albumin 4.1 g/dL", "albumin", "4.1", "g/dL"))
    assert r.parsed.readings == [] and r.discarded == [("albumin 4.1 g/dL", "lab", "the quote is not in the text")]


@pytest.mark.parametrize("text, item", [
    ("albumin 4.1 g/dL", lab("albumin 4.1 g/dL", "albumin", "41", "g/dL")),           # converted by the model
    ("glucose 105 mg/dL", lab("glucose 105 mg/dL", "glucose", "10", "mg/dL")),        # cut from 105
    ("LDL 3.45 mmol/L", lab("LDL 3.45 mmol/L", "ldl cholesterol", "3.4", "mmol/L")),   # cut from 3.45
    ("height 5'11\"", height("height 5'11\"", "5'1", "ft-in")),                          # cut from 5'11
    ("weight 80 kg", weight("weight 80 kg", "176", "lb")),                              # the model's conversion
    ("age 61", age("age 61", "16")),
])
def test_a_number_that_is_not_in_its_quote_is_dropped(text, item):
    r = read("58 year old male\n" + text, age("58 year old male", "58"), sex("58 year old male", "male"), item)
    p = r.parsed
    assert [d[0] for d in r.discarded] == [text], r.discarded
    assert not p.readings and p.weight_kg is None and p.height_cm is None and p.age == 58


def test_a_lab_unit_the_text_does_not_give_is_never_used():
    r = read("58 year old male\nCRP 3.1", age("58 year old male", "58"), sex("58 year old male", "male"),
             lab("CRP 3.1", "c-reactive protein", "3.1", "mg/L"))
    assert r.discarded == [("CRP 3.1", "lab", "the unit is not in the quote")]
    r = read("58 year old male\nCRP 3.1", age("58 year old male", "58"), sex("58 year old male", "male"),
             lab("CRP 3.1", "c-reactive protein", "3.1", ""))
    (reading,) = r.parsed.readings
    assert reading.status == "needs a unit" and not r.parsed.ok          # the unit rule asks: mg/L or mg/dL?


@pytest.mark.parametrize("item, what", [(weight("weight 150", "150", "lb"), "weight"),
                                        (height("height 165", "165", "cm"), "height")])
def test_a_body_unit_the_text_does_not_give_is_asked_not_guessed(item, what):
    p = read("40 year old female\n" + item["quote"], age("40 year old female", "40"),
             sex("40 year old female", "female"), item).parsed
    assert not p.ok and any(f"give the {what}'s unit" in x for x in p.all_problems())
    assert p.weight_kg is None and p.height_cm is None


def test_typography_case_and_spacing_do_not_stop_a_quote():
    text = "58  Year Old MALE\nAlbumin   4.1 g/dL"
    p = read(text, age("58 year old male", "58"), sex("58 year old male", "male"),
             lab("albumin 4.1 g/dL", "albumin", "4.1", "g/dL")).parsed
    assert p.ok and p.labs()["LBDSALSI"] == pytest.approx(41)
    assert p.readings[0].typed == "Albumin   4.1 g/dL"                         # what was typed, shown


def test_a_decimal_comma_and_a_thousands_comma():
    p = read("58 year old male\nalbumin 4,1 g/dL", age("58 year old male", "58"), sex("58 year old male", "male"),
             lab("albumin 4,1 g/dL", "albumin", "4,1", "g/dL")).parsed
    assert p.ok and p.labs()["LBDSALSI"] == pytest.approx(41)
    p = read("58 year old male\nlymphocytes 1,900 cells/uL", age("58 year old male", "58"),
             sex("58 year old male", "male"), lab("lymphocytes 1,900 cells/uL", "lymphocytes", "1,900", "cells/uL")).parsed
    assert not p.ok and any("thousands comma" in x for x in p.all_problems())


def test_nothing_inside_a_statement_about_someone_else_is_the_persons():
    text = "45 year old female, never smoked\nmy husband smokes and has diabetes"
    r = read(text, age("45 year old female", "45"), sex("45 year old female", "female"),
             smoking("never smoked", "never"), k("my husband smokes and has diabetes", "someone_else"),
             smoking("smokes", "current"), condition("has diabetes", "diabetes"))
    p = r.parsed
    assert (p.smoking, p.cotinine_level) == ("NeverSmoker", 0) and "DIQ010" not in p.questionnaire
    assert p.set_aside == ["my husband smokes and has diabetes"] and len(r.discarded) == 2
    assert p.statements[p.set_aside[0].statement].outcome == "set_aside"
    # someone else's diabetes is dropped quietly; a smoking status found there is asked about,
    # because it may be the person's own ("my wife and I smoke")
    (asked,) = p.all_problems()
    assert asked.topic == "smoking" and "could not be checked" in asked and not p.ok


def test_my_partner_and_i_smoke_is_asked_not_lost():
    r = read(T + "my wife and I smoke", *WHO, smoking("my wife and I smoke", "current"),
             k("my wife and I smoke", "someone_else"))
    assert r.parsed.smoking is None and not r.parsed.ok
    assert any(x.topic == "smoking" and "on its own line" in x for x in r.parsed.all_problems())


def test_a_condition_the_checks_drop_is_asked_about_never_answered_no():
    """The model joined two places into one quote; the check drops it — and the list must not
    then answer asthma No (seen live: 'no known conditions except hypertension, asthma')."""
    text = T + "no known conditions except hypertension, asthma"
    r = read(text, *WHO, condition("no known conditions except hypertension", "hypertension"),
             condition("no known conditions except asthma", "asthma"))
    p = r.parsed
    assert r.discarded == [("no known conditions except asthma", "condition", "the quote is not in the text")]
    assert not p.ok and any(x.kind == "lost_condition" for x in p.all_problems())
    assert p.not_understood == ["asthma"]                         # and the words show as not used


# ═══════════════════════════ the domain rules, on the model's items ═══════════

T = "58 year old male\n"
WHO = (age("58 year old male", "58"), sex("58 year old male", "male"))


@pytest.mark.parametrize("line, item, expect", [
    ("quit smoking 20 years ago", smoking("quit smoking 20 years ago", "former"), ("FormerSmoker", 0)),
    ("smoker, a pack a day", smoking("smoker, a pack a day", "current"), ("CurrentSmoker", 3)),
    ("moderate smoker", smoking("moderate smoker", "current", "moderate"), ("CurrentSmoker", 2)),
    ("smokes at weekends", smoking("smokes at weekends", "current", "occasional"), ("CurrentSmoker", 1)),
    ("never smoked", smoking("never smoked", "never", "occasional"), ("NeverSmoker", 0)),   # amount ignored
])
def test_smoking_status_and_its_cotinine_level(line, item, expect):
    p = read(T + line, *WHO, item).parsed
    assert p.ok and (p.smoking, p.cotinine_level) == expect


def test_a_smoking_status_the_model_cannot_tell_is_asked():
    p = read(T + "smoked a pack a day for 40 years", *WHO,
             smoking("smoked a pack a day for 40 years", "unclear")).parsed
    assert p.smoking is None and not p.ok
    assert any(x.kind == "ambiguous" and x.topic == "smoking" for x in p.all_problems())


def test_vaping_needs_a_measured_cotinine():
    p = read(T + "I vape daily", *WHO, smoking("I vape daily", "unclear", other="vaping")).parsed
    assert not p.ok and any(x.kind == "vaping" for x in p.all_problems())
    p = read(T + "I vape daily\ncotinine 150 ng/mL", *WHO, smoking("I vape daily", "unclear", other="vaping"),
             {"quote": "cotinine 150 ng/mL", "kind": "cotinine", "number": "150", "unit": "ng/mL"}).parsed
    assert p.ok and p.cotinine_level == 2 and any("measured cotinine is used" in n for n in p.notes)


def test_cannabis_is_asked_and_second_hand_smoke_is_not_counted():
    p = read(T + "I smoke weed", *WHO, smoking("I smoke weed", "unclear", other="cannabis")).parsed
    assert not p.ok and any("cannabis is not tobacco" in x for x in p.all_problems())
    p = read(T + "never smoked, but around second-hand smoke at work", *WHO,
             smoking("never smoked, but around second-hand smoke at work", "never", other="secondhand")).parsed
    assert p.ok and p.smoking == "NeverSmoker" and any("second-hand smoke is not smoking" in n for n in p.notes)


def test_a_measured_cotinine_beats_the_words():
    p = read(T + "current smoker\ncotinine 5 ng/mL", *WHO, smoking("current smoker", "current"),
             {"quote": "cotinine 5 ng/mL", "kind": "cotinine", "number": "5", "unit": "ng/mL"}).parsed
    assert (p.smoking, p.cotinine_level, p.ok) == ("CurrentSmoker", 0, True)


def test_two_smoking_statuses_contradict():
    p = read(T + "never smoked\ncurrent smoker", *WHO, smoking("never smoked", "never"),
             smoking("current smoker", "current")).parsed
    assert not p.ok and any("two different smoking statuss" in x or "smoking status" in x for x in p.all_problems())


@pytest.mark.parametrize("line, taking, current", [
    ("on metformin since 2019", "now", ["Metformin"]),
    ("my doctor wants me to start metformin", "not_now", []),
    ("stopped metformin last year", "not_now", []),
])
def test_a_medication_counts_only_when_taken_now(line, taking, current):
    p = read(T + line, *WHO, med(line, taking)).parsed
    assert p.ok and p.medications == current


def test_taking_and_not_taking_the_same_drug_contradict():
    p = read(T + "on metformin\nstopped metformin", *WHO, med("on metformin"), med("stopped metformin", "not_now")).parsed
    assert not p.ok and any(x.kind == "contradiction" and x.topic == "medication" for x in p.all_problems())


def test_conditions_answer_their_items_and_no_other_conditions_the_rest():
    p = read(T + "diabetic, high blood pressure, nothing else", *WHO,
             condition("diabetic", "diabetes"), condition("high blood pressure", "hypertension"),
             k("nothing else", "no_other_conditions")).parsed
    q = p.questionnaire
    assert p.ok and (q["DIQ010"], q["BPQ020"], q["MCQ220"]) == (1, 1, 2)
    p = read(T + "no diabetes, prediabetes", *WHO, condition("no diabetes", "diabetes", "no"),
             condition("prediabetes", "diabetes", "borderline")).parsed
    assert p.ok and p.questionnaire["DIQ010"] == 3


def test_a_borderline_answer_only_diabetes_has():
    p = read(T + "borderline hypertension", *WHO, condition("borderline hypertension", "hypertension", "borderline")).parsed
    assert not p.ok and any("no borderline answer" in x for x in p.all_problems())


def test_what_the_model_cannot_tell_about_a_condition_is_asked_and_about_a_lab_is_a_note():
    p = read(T + "heart trouble", *WHO, k("heart trouble", "unclear", topic="condition", why="vague")).parsed
    assert not p.ok and any(x.topic == "condition" and "vague" in x for x in p.all_problems())
    p = read(T + "some liver numbers were off", *WHO,
             k("some liver numbers were off", "unclear", topic="lab", why="no values")).parsed
    assert p.ok and any("could not read" in n for n in p.notes)


@pytest.mark.parametrize("line, item, value", [
    ("GrimAge says I'm 4 years older than my age",
     k("GrimAge says I'm 4 years older", "grimage", number="4", direction="older", wording="acceleration"), 4.0),
    ("GrimAge acceleration -2.5 years",
     k("GrimAge acceleration -2.5 years", "grimage", number="-2.5", direction="signed", wording="acceleration"), -2.5),
    ("GrimAge 3 years younger", k("GrimAge 3 years younger", "grimage", number="3", direction="younger",
                                  wording="acceleration"), -3.0),
])
def test_a_grimage_acceleration_with_its_sign(line, item, value):
    p = read(T + line, *WHO, item).parsed
    assert p.ok and p.extra_markers["AgeAccelGrim"]["value"] == value


def test_a_grimage_clock_age_is_asked_about():
    p = read("42 year old female\nGrimAge 46.3", age("42 year old female", "42"), sex("42 year old female", "female"),
             k("GrimAge 46.3", "grimage", number="46.3", direction="unstated", wording="clock_age")).parsed
    assert not p.ok and "AgeAccelGrim" not in p.extra_markers and any("ACCELERATION" in x for x in p.all_problems())


def test_visits_per_month_are_counted_per_year():
    p = read(T + "I see a doctor twice a month", *WHO,
             k("see a doctor twice a month", "healthcare_visits", number="2", period="month")).parsed
    assert p.ok and p.questionnaire["HUQ050"] == 5                          # 24 a year: 13+


def test_two_ages_contradict():
    p = read("58 year old male\nI'm 61", *WHO, age("I'm 61", "61")).parsed
    assert not p.ok and any("two different ages" in x for x in p.all_problems())


def test_the_reading_builds_the_same_patient_as_its_canonical_lines():
    """A model reading and the canonical lines that say the same facts are one patient."""
    text = ("62 yo woman. Smoker. Diabetic, on metformin since 2019.\n"
            "hs-CRP 2.4 mg/L, fasting glucose 6.1 mmol/L, albumin 3.9 g/dL")
    r = read(text, age("62 yo woman", "62"), sex("62 yo woman", "female"), smoking("Smoker", "current"),
             condition("Diabetic", "diabetes"), med("on metformin since 2019"),
             lab("hs-CRP 2.4 mg/L", "c-reactive protein", "2.4", "mg/L"),
             lab("fasting glucose 6.1 mmol/L", "fasting glucose", "6.1", "mmol/L"),
             lab("albumin 3.9 g/dL", "albumin", "3.9", "g/dL"))
    lines = read_lines("62 year old female, current smoker\ndiagnoses: diabetes\nmedications: metformin\n"
                       "CRP 2.4 mg/L\nfasting glucose 6.1 mmol/L\nalbumin 3.9 g/dL")
    assert r.parsed.ok and lines.ok
    assert r.parsed.to_patient("Me")[0] == lines.to_patient("Me")[0]


def test_a_quote_must_be_whole_words_in_the_text():
    """'male' is not in 'female'; a sex the quote does not state is not taken."""
    r = read("58 year old female", age("58 year old female", "58"), sex("male", "male"))
    assert r.parsed.sex is None and r.discarded == [("male", "sex", "the quote is not in the text")]
    r = read("58 year old female", age("58 year old female", "58"), sex("58 year old", "male"))
    assert r.parsed.sex is None and r.discarded == [("58 year old", "sex", "the quote does not say male")]
    p = read("58M, albumin 4.1 g/dL", age("58M", "58"), sex("58M", "male")).parsed
    assert (p.age, p.sex) == (58, "Male")


def test_the_reading_shows_what_was_typed_for_each_diagnosis():
    p = read(T + "high blood pressure, T2D", *WHO, condition("high blood pressure", "hypertension"),
             condition("T2D", "diabetes")).parsed
    assert p.questionnaire_notes[0].startswith(
        "diagnoses: hypertension ('high blood pressure'), diabetes ('T2D') — every diagnosis not listed")


# ═══════════════════════════ replay: real model outputs, offline ═══════════════
# scripts/eval_patient_extraction.py --record keeps what the model returned for every text
# of the evaluation. Replayed here through the checks and the rules: a text with an
# expected reading must never come back usable with a different value, and no reading may
# hold a value its text does not.

import gzip  # noqa: E402
import json  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "patient_extractions.json.gz"
CORPUS = REPO / "docs" / "patient_extraction" / "eval_corpus.json"


def _recorded() -> dict:
    from core.patient_extract import schema_hash
    if not FIXTURE.exists():
        return {}
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_hash") != schema_hash():
        return {}                                   # recorded against another schema: re-record
    return {e["text"]: e["items"] for e in data["extractions"].values()}


RECORDED = _recorded()


def _expected() -> list:
    d = json.loads(CORPUS.read_text(encoding="utf-8"))
    return [(section, e) for section in ("smoking", "diagnoses", "body") for e in d[section]
            if e["expect"] is not None and e["text"] in RECORDED]


@pytest.mark.skipif(not RECORDED, reason="no recorded extractions for this schema")
def test_no_recorded_reading_is_a_confident_wrong_patient():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", REPO / "scripts" / "eval_patient_extraction.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    wrong = []
    for section, e in _expected():
        p = read_patient(e["text"], Recorded(*RECORDED[e["text"]])).parsed
        different = [m for m in ev._mismatches({**e, "source": f"corpus:{section}"}, p) if " None (" not in m]
        if p.ok and different:                     # a value missing is listed as not used; a different one is wrong
            wrong.append((e["text"], different))
    assert not wrong, wrong


@pytest.mark.skipif(not RECORDED, reason="no recorded extractions for this schema")
def test_every_recorded_reading_holds_only_what_its_text_says():
    from core.patient_read import _fold
    for text, items in RECORDED.items():
        r = read_patient(text, Recorded(*items))
        folded = _fold(text)[0]
        for reading in r.parsed.readings:
            assert _fold(reading.typed)[0] in folded, (text, reading.typed)
        for st in r.parsed.statements:
            assert text.splitlines()[st.line][st.start:st.end] == st.text, (text, st)
