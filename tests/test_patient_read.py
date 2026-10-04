"""Reading the My Patient text with a model, without giving up "refuse rather than guess"
(core/patient_read.py) — all with recorded extractions, never a live call.

The model may rewrite only what the rules did not understand; what the rules refused
it may only suggest; a smoking status from it is never taken without a click; where
it reads a statement the rules read differently, smoking, age and sex block. The
property tests at the bottom hold that against a hostile model: every schema-valid
output whose quotes are in the text, over every corpus entry.

    pytest tests/test_patient_read.py -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
for p in (str(PLN_CHAT), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.patient_extract import ExtractError, Extraction, check_output  # noqa: E402
from core.patient_read import read_patient  # noqa: E402
from core.patient_text import read_patient_text  # noqa: E402
from core.patient_vocabulary import vocabulary  # noqa: E402

V = vocabulary()


class Recorded:
    """An extractor that returns what a model once returned (or would)."""
    model = "recorded"

    def __init__(self, *items):
        self.items = [dict(i) for i in items]
        out = {"items": self.items}
        assert check_output(out) is None, check_output(out)
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return Extraction(list(self.items), "recorded", 1.0, {"prompt_tokens": 1})


def smoking(quote, status, occasional=False, other="none"):
    return {"quote": quote, "kind": "smoking", "status": status, "occasional": occasional,
            "other_nicotine": other}


def condition(quote, key, answer="yes"):
    return {"quote": quote, "kind": "condition", "condition": key, "answer": answer}


def lab(quote, group):
    return {"quote": quote, "kind": "lab", "lab": group}


def k(quote, kind, **kw):
    return {"quote": quote, "kind": kind, **kw}


# ═══════════════════════════ without a model ═══════════════════════════════════

def test_without_an_extractor_it_is_the_rules_alone():
    text = "58 year old male, current smoker\nalbumin 4.1 g/dL"
    r = read_patient(text)
    assert r.reader == "rules" and r.read_as == text and r.header() == "Read by rules only"
    assert r.parsed.as_dict() == read_patient_text(text).as_dict()


@pytest.mark.parametrize("error, header", [
    (ExtractError("no_key", "no OPENAI_API_KEY"), "Read by rules only — no OPENAI_API_KEY"),
    (ExtractError("timeout", "x"), "Read by rules only — the model did not answer in time"),
    (ExtractError("refused", "x"), "Read by rules only — the model declined"),
    (ExtractError("config", "OpenAI rejected the request"),
     "Read by rules only — model reader misconfigured: OpenAI rejected the request"),
])
def test_a_failed_model_leaves_the_rules_reading_and_says_why(error, header):
    def extractor(text):
        raise error
    text = "58 yo M\nnonsmoker"
    r = read_patient(text, extractor)
    assert r.reader == "rules" and r.read_as == text and r.header() == header
    assert r.parsed.as_dict() == read_patient_text(text).as_dict()


def test_an_unexpected_extractor_crash_is_a_model_error_not_a_crash():
    def extractor(text):
        raise RuntimeError("boom")
    r = read_patient("58 year old male", extractor)
    assert r.model_error.code == "upstream" and r.parsed.ok


# ═══════════════════════════ rewriting what the rules did not understand ═══════

def test_an_unread_header_is_rewritten_and_marked():
    r = read_patient("58 yo M\nalbumin 4.1 g/dL",
                     Recorded(k("58 yo M", "age"), k("58 yo M", "sex", sex="male"),
                              lab("albumin 4.1 g/dL", "albumin")))
    assert read_patient_text("58 yo M\nalbumin 4.1 g/dL").sex is None
    assert r.read_as == "58 year old; male\nalbumin 4.1 g/dL"
    assert r.parsed.ok and (r.parsed.age, r.parsed.sex) == (58, "Male")
    assert r.header() == "Read by rules + recorded"
    assert r.source(0) == r.source(1) == "model" and r.model_quotes[0] == "58 yo M"
    assert r.source(2) == "rules"


def test_an_unread_lab_line_is_rewritten_with_its_own_number_and_unit():
    text = "58 year old male\nmy albumin was 4.1 g/dL\nmy last HbA1c: 6.1 % (high)"
    r = read_patient(text, Recorded(lab("albumin was 4.1 g/dL", "albumin"),
                                    lab("HbA1c: 6.1 % (high)", "hba1c")))
    assert r.read_as == "58 year old male\nAlbumin 4.1 g/dL\nHbA1c 6.1 %"
    assert {x.code: x.value for x in r.parsed.readings} == {"LBDSALSI": pytest.approx(41.0),
                                                            "LBXGH": pytest.approx(6.1)}
    assert r.parsed.ok and not r.parsed.not_understood


def test_an_unread_condition_is_only_ever_suggested():
    """An unread line that names one of the 23 conditions is a refusal of the rules (it
    would be answered No); the model's reading of it is a wording, never a rewrite."""
    text = "58 year old male\nI was told I have T2D\nmy doctor worries about my heart"
    r = read_patient(text, Recorded(condition("T2D", "diabetes"),
                                    condition("my heart", "coronary heart disease")))
    assert not r.parsed.ok and r.read_as == text and "DIQ010" not in r.parsed.questionnaire
    assert [s.wordings for s in r.suggestions] == [["diagnoses: diabetes"]]
    assert any("does not say it" in n for n in r.notes)                 # inferred: a note


def test_an_unread_age_and_lab_are_rewritten_beside_a_refused_line():
    text = "58 yo M\nmy albumin was 4.1 g/dL\nI was told I have T2D"
    r = read_patient(text, Recorded(k("58 yo M", "age"), k("58 yo M", "sex", sex="male"),
                                    lab("albumin was 4.1 g/dL", "albumin"), condition("T2D", "diabetes")))
    assert r.read_as == "58 year old; male\nAlbumin 4.1 g/dL\nI was told I have T2D"
    assert (r.parsed.age, r.parsed.sex) == (58, "Male") and r.parsed.readings[0].code == "LBDSALSI"
    assert not r.parsed.ok and r.suggestions[0].wordings == ["diagnoses: diabetes"]


def test_a_model_number_never_replaces_the_typed_one():
    """The model has no number to give; a quote with a different number is not in the text."""
    r = read_patient("58 year old male\nmy albumin was 4.1 g/dL",
                     Recorded(lab("albumin was 3.1 g/dL", "albumin")))
    assert r.discarded and r.parsed.readings == []


# ═══════════════════════════ never without a click ═════════════════════════════

def test_a_refused_statement_is_only_ever_a_suggestion():
    text = "58 year old male\ncurrent smoker, quit 2015"
    r = read_patient(text, Recorded(smoking("current smoker, quit 2015", "former")))
    assert not r.parsed.ok and r.read_as == text
    assert r.parsed.all_problems() == read_patient_text(text).all_problems()
    (s,) = r.suggestions
    assert s.wordings == ["former smoker"] and s.original == "current smoker, quit 2015"
    assert s.apply(text, s.wordings[0]) == "58 year old male\nformer smoker"


def test_a_unit_refusal_is_a_suggestion_too():
    text = "58 year old male\nAlbumin: 4.1 g/dL (3.5-5.0) H"
    assert not read_patient_text(text).ok
    r = read_patient(text, Recorded(lab("Albumin: 4.1 g/dL (3.5-5.0) H", "albumin")))
    assert not r.parsed.ok and [s.wordings for s in r.suggestions] == [["Albumin 4.1 g/dL"]]


def test_a_smoking_status_only_the_model_read_blocks_until_clicked():
    text = "58 year old male\nquit the pipe in 2010"
    assert read_patient_text(text).ok and read_patient_text(text).smoking is None   # not read
    r = read_patient(text, Recorded(smoking("quit the pipe in 2010", "former")))
    assert not r.parsed.ok and r.parsed.smoking is None and r.read_as == text
    (s,) = r.suggestions
    assert s.blocking and s.wordings == ["former smoker"]
    after = read_patient(s.apply(text, s.wordings[0]), Recorded(smoking("former smoker", "former")))
    assert after.parsed.ok and after.parsed.smoking == "FormerSmoker"


# ═══════════════════════════ disagreements ══════════════════════════════════════

def test_a_smoking_disagreement_blocks_with_both_wordings():
    text = "58 year old male\nex-smoker"
    assert read_patient_text(text).ok and read_patient_text(text).smoking == "FormerSmoker"
    r = read_patient(text, Recorded(smoking("ex-smoker", "current")))
    assert not r.parsed.ok
    (p,) = [x for x in r.parsed.all_problems() if x.kind == "disagreement"]
    assert "the rules read former smoker, the model reads current smoker" in p
    (s,) = r.suggestions
    assert s.blocking and s.wordings == ["current smoker", "former smoker"]


def test_a_refusal_of_the_rules_keeps_its_own_message_and_gets_the_models_wording():
    text = "58 year old male\nI never quit smoking"                    # round 4: was read as former
    r = read_patient(text, Recorded(smoking("I never quit smoking", "current")))
    assert not r.parsed.ok and r.parsed.all_problems() == read_patient_text(text).all_problems()
    assert [s.wordings for s in r.suggestions] == [["current smoker"]]


def test_an_intensity_disagreement_blocks_too():
    r = read_patient("58 year old male\nlight smoker", Recorded(smoking("light smoker", "current")))
    assert read_patient_text("58 year old male\nlight smoker").cotinine_level == 1
    assert not r.parsed.ok and r.suggestions[0].wordings == ["current smoker", "occasional smoker"]


def test_occasional_needs_its_word_in_the_quote():
    r = read_patient("58 year old male\ncurrent smoker",
                     Recorded(smoking("current smoker", "current", occasional=True)))
    assert r.parsed.ok and r.parsed.cotinine_level == 3


def test_a_canonical_statement_always_wins():
    r = read_patient("58 year old male\nformer smoker", Recorded(smoking("former smoker", "current")))
    assert r.parsed.ok and r.parsed.smoking == "FormerSmoker" and not r.suggestions


def test_age_and_sex_disagreements_block():
    r = read_patient("58 year old male\nalbumin 4.1 g/dL",
                     Recorded(k("58 year old male", "sex", sex="female")))
    assert r.discarded                                  # no 'female' in the quote: not grounded
    r = read_patient("58 yo male, 85 kg", Recorded(k("58 yo male", "age")))
    assert r.parsed.age == 58


def test_a_lab_disagreement_is_a_note_and_the_rules_value_stands():
    text = "58 year old male\nalbumin 4.1 g/dL"
    r = read_patient(text, Recorded(lab("albumin 4.1 g/dL", "total protein")))
    assert r.parsed.ok and r.parsed.readings[0].code == "LBDSALSI"
    assert ("albumin 4.1 g/dL", "lab", "the quote does not name total protein exactly once") in r.discarded


def test_a_refused_list_gets_the_models_wording():
    text = "58 year old male\ndiagnoses: hypertension, T2D"
    assert not read_patient_text(text).ok                              # round 4: T2D was dropped as No
    r = read_patient(text, Recorded(condition("hypertension", "hypertension"), condition("T2D", "diabetes")))
    assert not r.parsed.ok
    (s,) = r.suggestions
    assert not s.blocking and s.wordings == ["diagnoses: hypertension; diagnoses: diabetes"]
    fixed = s.apply(text, s.wordings[0])
    assert read_patient_text(fixed).questionnaire["DIQ010"] == 1


def test_a_lab_or_condition_the_model_reads_differently_is_a_note():
    text = "58 year old male\nCRP 3.1 mg/L"
    r = read_patient(text, Recorded(lab("CRP 3.1 mg/L", "c-reactive protein")))
    assert r.parsed.ok and not r.notes and not r.suggestions             # agreement: nothing to say


# ═══════════════════════════ what code checks in a claim ══════════════════════

def test_vaping_or_nicotine_replacement_next_to_never_smoked_blocks():
    text = "58 year old male, never smoked\nuses pouches daily"     # the rules read it as nothing
    assert read_patient_text(text).ok
    r = read_patient(text, Recorded(smoking("uses pouches daily", "unclear", other="smokeless")))
    assert not r.parsed.ok and any(x.kind == "vaping" for x in r.parsed.all_problems())
    measured = read_patient(text + "\ncotinine 250 ng/mL",
                            Recorded(smoking("uses pouches daily", "unclear", other="smokeless")))
    assert not any(x.kind == "vaping" for x in measured.parsed.all_problems())
    assert any("measured cotinine" in n for n in measured.notes)


def test_cannabis_is_not_tobacco():
    text = "58 year old male\nI smoke weed"
    r = read_patient(text, Recorded(smoking("I smoke weed", "current", other="cannabis")))
    assert not r.parsed.ok and any("cannabis is not tobacco" in x for x in r.parsed.all_problems())


def test_secondhand_smoke_is_a_note():
    text = "58 year old male\nnon-smoker, but exposed to secondhand smoke"
    r = read_patient(text, Recorded(smoking("non-smoker", "never"),
                                    k("exposed to secondhand smoke", "someone_else")))
    assert r.parsed.ok and r.parsed.smoking == "NeverSmoker"


def test_an_unclear_smoking_statement_blocks_unless_canonical():
    r = read_patient("58 year old male\nsmoker-ish", Recorded(k("smoker-ish", "unclear", topic="smoking", why="?")))
    assert not r.parsed.ok
    r = read_patient("58 year old male\ncurrent smoker", Recorded(k("current smoker", "unclear", topic="smoking",
                                                                    why="?")))
    assert r.parsed.ok


def test_an_unclear_non_medical_statement_is_a_note():
    r = read_patient("58 year old male\nlikes walking", Recorded(k("likes walking", "unclear", topic="other",
                                                                   why="not health")))
    assert r.parsed.ok and any("could not read" in n for n in r.notes)


def test_a_dated_hospital_stay_is_outside_the_window_and_blocks():
    from core.patient_read import Item, _check_condition

    text = "58 year old male\ndiagnoses: hypertension\nhospitalized in 2015"
    assert not read_patient_text(text).ok                                # round 4: was a Yes
    r = read_patient(text, Recorded(condition("hospitalized in 2015", "overnight hospital stay")))
    assert not r.parsed.ok and any("past 12 months" in n for n in r.notes)
    it = Item({"quote": "hospitalized in 2015", "kind": "condition", "condition": "overnight hospital stay",
               "answer": "yes"}, "condition", 0, 0, 20, "hospitalized in 2015")
    assert _check_condition(it) == "" and it.fact is None and "past 12 months" in it.problem[0]


def test_an_excluded_wording_blocks():
    text = "58 year old male\nkidney disease"
    r = read_patient(text, Recorded(condition("kidney disease", "kidney disease")))
    assert r.parsed.ok
    text = "58 year old male\nosteoporosis\nno other conditions"
    r = read_patient(text, Recorded(condition("osteoporosis", "osteoporosis")))
    assert r.parsed.ok


@pytest.mark.parametrize("quote, key, answer, ok", [
    ("no diabetes or hypertension", "hypertension", "no", True),
    ("no diabetes or hypertension", "hypertension", "yes", False),       # the wording says no
    ("hypertension, no diabetes", "diabetes", "no", True),
    ("hypertension, no diabetes", "hypertension", "yes", True),
    ("hypertension: no", "hypertension", "no", True),
    ("hypertension ruled out", "hypertension", "yes", False),
    ("prediabetes", "diabetes", "borderline", True),
    ("prediabetes", "diabetes", "yes", False),
    ("diabetes", "diabetes", "borderline", False),
])
def test_condition_polarity_is_computed_by_code(quote, key, answer, ok):
    from core.patient_read import Item, _check_condition

    it = Item({"quote": quote, "kind": "condition", "condition": key, "answer": answer},
              "condition", 0, 0, len(quote), quote)
    assert (_check_condition(it) == "" and it.fact is not None) is ok


@pytest.mark.parametrize("quote, sex, ok", [
    ("58M", "male", True), ("58 y/o F", "female", True), ("Sex: F", "female", True),
    ("male", "male", True), ("Gender: M", "male", True), ("height 1.78 m", "male", False),
    ("my husband", "female", False), ("PSA 1.2", "male", False), ("58M", "female", False),
])
def test_sex_needs_an_explicit_token(quote, sex, ok):
    from core.patient_read import Item, _check_sex

    it = Item({"quote": quote, "kind": "sex", "sex": sex}, "sex", 0, 0, len(quote), quote)
    assert (_check_sex(it) == "") is ok


@pytest.mark.parametrize("quote, ok, number", [
    ("58 yo", True, "58"), ("58-year-old", True, "58"), ("aged 58", True, "58"), ("58M", True, "58"),
    ("Male, 58", True, "58"), ("quit at age 40", False, None), ("biological age 62", False, None),
    ("diabetic for 5 years", False, None), ("58 years", False, None),
])
def test_age_needs_an_age_phrase(quote, ok, number):
    from core.patient_read import Item, _check_age

    it = Item({"quote": quote, "kind": "age"}, "age", 0, 0, len(quote), quote)
    assert (_check_age(it) == "") is ok and (it.fact.number if it.fact else None) == number


@pytest.mark.parametrize("quote, group, why_or_line", [
    ("albumin 4.1 g/dL", "albumin", "Albumin 4.1 g/dL"),
    ("Albumin (g/dL) 4.1", "albumin", "Albumin 4.1 g/dL"),
    ("albumin is 41 g/L.", "albumin", "Albumin 41 g/L"),
    ("albumin 4.1 g/dL (ref 3.5-5.0)", "albumin", "Albumin 4.1 g/dL"),
    ("albumin 4.1 gm/dl", "albumin", "Albumin 4.1 gm/dl"),               # unknown: the rules refuse it
    ("albumin 4.1", "albumin", "Albumin 4.1"),                           # no unit: the rules decide
    ("albumin 4,1 g/dL", "albumin", "the number continues"),
    ("albumin <3.5 g/dL", "albumin", "a censored value"),
    ("albumin 4.1 then 3.9 g/dL", "albumin", "a second number"),
    ("non-HDL cholesterol 4.1 mmol/L", "hdl cholesterol", "other words name the analyte"),
    ("albumin/creatinine ratio 30", "albumin", "no number right after the name"),
    ("lymphocytes 30 %", "lymphocytes", "lymphocytes 30 %"),
    ("glucose (fasting) 98 mg/dL", "fasting glucose", "the quote does not say fasting"),
    ("fasting glucose 98 mg/dL", "glucose", "Glucose 98 mg/dL"),        # fasting is never granted
    ("fasting glucose 98 mg/dL", "fasting glucose", "fasting glucose 98 mg/dL"),
    ("BUN 15 mg/dL", "urea", "the quote does not name urea"),            # 2.1x apart in mg/dL
    ("urea 15 mg/dL", "urea nitrogen (bun)", "the quote does not name urea nitrogen"),
])
def test_a_lab_claim_copies_name_number_and_unit_from_the_quote(quote, group, why_or_line):
    from core.patient_canonical import render
    from core.patient_read import Item, _check_lab

    it = Item({"quote": quote, "kind": "lab", "lab": group}, "lab", 0, 0, len(quote), quote)
    why = _check_lab(it)
    if why:
        assert why.startswith(why_or_line), why
    else:
        assert render(it.fact) == why_or_line


@pytest.mark.parametrize("quote, line", [
    ("cotinine 250 ng/mL", "cotinine 250 ng/mL"), ("cotinine level 2", "cotinine level 2"),
    ("cotinine <10 ng/mL", "cotinine 0 ng/mL"), ("cotinine was 250 µg/L", "cotinine 250 ng/mL"),
    ("cotinine <50 ng/mL", None), ("cotinine undetectable", None), ("cotinine 3", None),
])
def test_a_cotinine_claim(quote, line):
    from core.patient_canonical import render
    from core.patient_read import Item, _check_cotinine

    it = Item({"quote": quote, "kind": "cotinine"}, "cotinine", 0, 0, len(quote), quote)
    why = _check_cotinine(it)
    assert (render(it.fact) if not why else None) == line


@pytest.mark.parametrize("quote, period, line", [
    ("doctor visits: 2 per month", "month", "healthcare visits in the past year: 24"),
    ("saw my GP twice last year", "year", "healthcare visits in the past year: 2"),
    ("doctor visits: 3", "unstated", "healthcare visits in the past year: 3"),
    ("doctor visits: 3-5", "unstated", None), ("doctor visits: 3 (last 5 years)", "unstated", None),
    ("doctor visits: 2 per month", "year", None),
])
def test_visits_are_converted_to_a_year(quote, period, line):
    from core.patient_canonical import render
    from core.patient_read import Item, _check_visits

    it = Item({"quote": quote, "kind": "healthcare_visits", "period": period}, "healthcare_visits", 0, 0,
              len(quote), quote)
    why = _check_visits(it)
    assert (render(it.fact) if not why else None) == line


# ═══════════════════════════ grounding ═════════════════════════════════════════

@pytest.mark.parametrize("quote, why", [
    ("current smoker", "the quote is not in the text"),
    ("smoker", "the quote occurs more than once"),
    ("58", "the quote is too short"),
])
def test_a_quote_must_occur_exactly_once_on_one_line(quote, why):
    text = "58 year old male, former smoker\nnot a smoker now"
    r = read_patient(text, Recorded(smoking(quote, "current")))
    assert (quote, "smoking", why) in r.discarded


def test_typography_does_not_stop_grounding():
    text = "58 year old male\nnon–smoker — never"
    r = read_patient(text, Recorded(smoking("non-smoker - never", "never")))
    assert not any(d[2].startswith("the quote is not") for d in r.discarded)


def test_a_quote_about_someone_else_is_dropped():
    text = "58 year old male\nmy father had a stroke"
    r = read_patient(text, Recorded(condition("had a stroke", "stroke")))
    assert r.parsed.questionnaire.get("MCQ160F") is None and r.discarded
    r = read_patient("58 year old male\nI had a stroke, my wife smokes",
                     Recorded(smoking("my wife smokes", "current"), k("my wife smokes", "someone_else")))
    assert r.parsed.smoking is None


def test_a_rewrite_that_would_change_another_line_is_withdrawn():
    from core.patient_read import Substitution, _apply, _check

    rules = read_patient_text("58 year old male\nfoo bar")
    sub = Substitution(1, 0, 7, "foo bar", ["current smoker, quit 2015"], (1,))
    read_as, expected = _apply(rules_lines := "58 year old male\nfoo bar".splitlines(), rules, [sub])
    assert rules_lines and _check(rules, read_patient_text(read_as), expected, [sub]) == {-1}


# ═══════════════════════════ a hostile model ════════════════════════════════════
# For every corpus entry, outputs a model could give whose quotes are in the text:
# every smoking status (occasional or not), sex, age, someone else, no other
# conditions, every condition whose words the statement holds, every lab group it
# names — one at a time and all at once. Expected-refused texts must stay refused;
# usable texts may gain a note or a block, never a changed value.

def _corpus():
    from test_patient_text_corpus import DIAGNOSES, SMOKING, _STATUS, _text

    for text, expect in SMOKING:
        yield _text(text), expect == "X", ("smoking", _STATUS.get(expect))
    for text, ok, items in DIAGNOSES:
        yield _text(text), not ok, ("conditions", items)


CORPUS = list(_corpus())


def _hostile_outputs(text: str):
    rules = read_patient_text(text)
    lines = text.splitlines()
    quotes = set()
    for st in rules.statements:
        quotes.add(lines[st.line][st.start:st.end])
        quotes.add(lines[st.line])
    singles = []
    for q in sorted(quotes):
        if len(q.replace(" ", "")) < 3:
            continue
        low = q.lower()
        for status, occ in itertools.product(("never", "former", "current"), (False, True)):
            singles.append(smoking(q, status, occ))
        singles += [k(q, "sex", sex="male"), k(q, "sex", sex="female"), k(q, "age"), k(q, "someone_else"),
                    k(q, "no_other_conditions"), k(q, "cotinine"),
                    k(q, "unclear", topic="other", why="x")]
        for c in V.condition_info.values():
            import re
            if re.search(rf"\b(?:{c.terms})\b", low):
                for a in ("yes", "no", "borderline"):
                    singles.append(condition(q, c.key, a))
        for key, g in V.labs.items():
            if any(a in low for a in g.aliases):
                singles.append(lab(q, key))
    yield from ([s] for s in singles)
    yield singles                                         # everything at once
    yield [s for s in singles if s["kind"] != "someone_else"]


def _values(p):
    """What the rules READ: a status, a measured cotinine, age, sex, the questionnaire
    answers some statement gave (not the No that 'no other conditions' fills in for the
    rest), and the labs. A model may add to these; it may not change one."""
    answered: dict = {}
    for st in p.statements:
        answered.update(st.facts.get("conditions", {}))
        answered.update(st.facts.get("questionnaire", {}))
    return {"smoking": p.smoking, "cotinine": p.cotinine_level if p.cotinine_measured else None,
            "age": p.age, "sex": p.sex, "q": {i: p.questionnaire[i] for i in answered if i in p.questionnaire},
            "labs": {r.code: r.value for r in p.readings if not r.blocking}}


@pytest.mark.parametrize("text, refused, expect", CORPUS, ids=[t.replace("\n", " | ") for t, _, _ in CORPUS])
def test_no_grounded_output_makes_a_refused_text_usable_or_flips_a_read_value(text, refused, expect):
    rules = read_patient_text(text)
    before = _values(rules)
    kept = {str(x) for x in rules.all_problems() if x.kind != "missing"}
    for items in _hostile_outputs(text):
        r = read_patient(text, Recorded(*items))
        final = r.parsed
        assert kept <= {str(x) for x in final.all_problems()}, (items, final.all_problems())
        if refused:
            assert not final.ok, (items, r.read_as)
            continue
        if not final.ok:
            continue                                      # a block is always allowed
        after = _values(final)
        for key in ("smoking", "cotinine", "age", "sex"):
            if before[key] is not None:
                assert after[key] == before[key], (key, items, r.read_as)
        for item, val in before["q"].items():
            assert after["q"].get(item) == val, (item, items, r.read_as)
        for code, val in before["labs"].items():
            assert after["labs"].get(code) == pytest.approx(val), (code, items)


#: the first line of a round-4 reproduction that is itself an age/sex header
HEADER = r"""(?i)^(?:\d{2}\s*(?:year|yr|y|,|m\b|f\b)|age\b|aged\b|(?:male|female|man|woman|sex|gender)\b|i'?m \d|[mf]\s*,?\s*\d)"""


def _round4_texts():
    """Every quoted reproduction in docs/patient_extraction/review_round4.json, as typed."""
    import json
    import re

    d = json.loads((REPO / "docs" / "patient_extraction" / "review_round4.json").read_text(encoding="utf-8"))
    out = set()
    for f in d["confirmed"] + d["critic_unverified"]:
        for a, b in re.findall(r"'([^'\n]{3,200})'|\"([^\"\n]{3,200})\"", f["reproduction"]):
            t = (a or b).replace("\\n", "\n")
            header = re.match(HEADER, t)                # a header test brings its own age and sex
            out.add(t if header else "58 year old male\n" + t)
    return sorted(out)


ROUND4 = _round4_texts()


def test_round4_texts_are_many():
    assert len(ROUND4) > 400


@pytest.mark.parametrize("chunk", range(8))
def test_no_grounded_output_flips_a_value_on_the_round4_texts(chunk):
    """The same property on every round-4 reproduction: refusals stay, read values stay
    unless the read is blocked. These texts have many unread statements, so the rewrite
    path is exercised far more than by the corpus."""
    rewrites = 0
    for text in ROUND4[chunk::8]:
        rules = read_patient_text(text)
        before = _values(rules)
        kept = {str(x) for x in rules.all_problems() if x.kind != "missing"}
        for items in _hostile_outputs(text):
            r = read_patient(text, Recorded(*items))
            rewrites += bool(r.substitutions)
            assert kept <= {str(x) for x in r.parsed.all_problems()}, (text, items)
            if any(x.kind != "missing" for x in rules.all_problems()):
                assert not r.parsed.ok, (text, items, r.read_as)
                continue
            # refused only for a missing age or sex ("58F"): a checked age/sex may complete it
            if not r.parsed.ok:
                continue
            after = _values(r.parsed)
            for key in ("smoking", "cotinine", "age", "sex"):
                if before[key] is not None:
                    assert after[key] == before[key], (text, key, items, r.read_as)
            for item, val in before["q"].items():
                assert after["q"].get(item) == val, (text, item, items, r.read_as)
            for code, val in before["labs"].items():
                assert after["labs"].get(code) == pytest.approx(val), (text, code, items)
    assert rewrites > 0


# ═══════════════════════════ replaying the live model ══════════════════════════
# scripts/eval_patient_extraction.py --record keeps what the live model returned for
# every corpus entry and round-4 reproduction. Replayed here, offline, through the
# current code: the same safety properties, and the corpus still mostly usable.

FIXTURE = REPO / "tests" / "fixtures" / "patient_extractions.json.gz"


def _recorded():
    import gzip
    import json

    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.skipif(not FIXTURE.exists(), reason="no recorded live extractions")
def test_replayed_live_extractions_keep_refusals_and_values():
    data = _recorded()
    assert len(data["extractions"]) > 1000
    for rec in data["extractions"].values():
        text = rec["text"]
        rules = read_patient_text(text)
        before = _values(rules)
        r = read_patient(text, Recorded(*rec["items"]))
        kept = {str(x) for x in rules.all_problems() if x.kind != "missing"}
        assert kept <= {str(x) for x in r.parsed.all_problems()}, text
        if any(x.kind != "missing" for x in rules.all_problems()):
            assert not r.parsed.ok, (text, r.read_as)
            continue
        if not r.parsed.ok:
            continue
        after = _values(r.parsed)
        for key in ("smoking", "cotinine", "age", "sex"):
            if before[key] is not None:
                assert after[key] == before[key], (text, key)
        for item, val in before["q"].items():
            assert after["q"].get(item) == val, (text, item)


@pytest.mark.skipif(not FIXTURE.exists(), reason="no recorded live extractions")
def test_replayed_live_extractions_leave_the_corpus_usable():
    """A check that over-blocks clear text shows up here before it reaches anyone."""
    by_text = {rec["text"]: rec["items"] for rec in _recorded()["extractions"].values()}
    usable = [(t, e) for t, refused, e in CORPUS if not refused and t in by_text]
    agree = 0
    for text, (what, expect) in usable:
        p = read_patient(text, Recorded(*by_text[text])).parsed
        if not p.ok:
            continue
        if what == "smoking":
            agree += (p.smoking, p.cotinine_level) == tuple(expect)
        else:
            agree += all(p.questionnaire.get(k) == v for k, v in expect.items())
    assert len(usable) > 150 and agree >= 0.95 * len(usable), (agree, len(usable))
