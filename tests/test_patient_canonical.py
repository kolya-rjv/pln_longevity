"""The round trip: every canonical line means exactly the Fact it was made from.

Canonical lines are the format a person can type without the model, and the one a reading
is written in (core.patient_canonical.render). These tests render every Fact the
vocabulary allows — every lab name group in every unit it accepts, every condition and
answer, every smoking status and amount, sex, age, weight, height, medication, health
answer, visit count, GrimAge acceleration and cotinine form — and check that
`parse(render(fact)) == [fact]` and that read_lines(render(fact)) reads back exactly that
fact, and nothing else.

    pytest tests/test_patient_canonical.py -q
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
from core.patient_canonical import (  # noqa: E402
    HEALTH_WORDS,
    SMOKING_PHRASES,
    TREND_WORDS,
    Fact,
    parse,
    render,
)
from core.patient_text import (  # noqa: E402
    _HEALTH,
    _TREND,
    LABS,
    OK,
    _cotinine_level,
    _visits_category,
    display_unit,
    normalise_unit,
    read_lines,
)
from core.patient_vocabulary import vocabulary  # noqa: E402

HEADER = "58 year old male\n"
V = vocabulary()


def _number(x: float) -> str:
    """A value as a person types it: three significant figures, no exponent."""
    x = float(f"{x:.3g}")
    return f"{x:f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))


def _the_line(text: str):
    """Read HEADER + one line; return the parse and the line's own statement(s)."""
    p = read_lines(HEADER + text)
    return p, [st for st in p.statements if st.line == 1]


def _round_trip(fact: Fact) -> str:
    line = render(fact)
    assert parse(line) == [fact], (fact, line, parse(line))
    return line


def _lab_cases():
    model = load_model()
    for key, group in V.labs.items():
        for spec in LABS:
            if spec.code not in group.codes or not set(spec.aliases) & set(group.aliases):
                continue
            median = 3.0 if spec.code == "LDLV" else model.raw["nhanes_central"][spec.code][1]
            for unit_key, (scale, offset) in spec.units.items():
                yield key, spec.code, _number((median - offset) / scale), display_unit(unit_key)
            yield key, spec.code, _number(median), ""          # no unit: the rules decide


LAB_CASES = list(_lab_cases())


@pytest.mark.parametrize("key, code, number, unit", LAB_CASES,
                         ids=[f"{k}|{n} {u}" for k, _, n, u in LAB_CASES])
def test_a_lab_line_reads_back_as_its_group_number_and_unit(key, code, number, unit):
    group = V.labs[key]
    line = _round_trip(Fact("lab", key, number, unit))
    p, (st,) = _the_line(line)
    (r,) = p.readings
    assert r.code in group.codes and r.value_typed == float(number), (line, r)
    assert r.fasting == group.fasting and st.outcome in ("read", "refused") and not p.not_understood
    if unit:
        # a unit is read as typed, and picks the input that takes it (percent vs count)
        assert r.code == code and normalise_unit(r.unit_typed) == normalise_unit(unit), (line, r)
        assert r.status == OK, (line, r.status, r.note)
    else:
        assert r.unit_typed is None


def test_every_lab_group_and_unit_is_covered():
    covered = {(k, normalise_unit(u)) for k, _, _, u in LAB_CASES if u}
    for key, group in V.labs.items():
        for u in group.units:
            assert (key, normalise_unit(u)) in covered, (key, u)


CONDITION_CASES = [(q, a) for q in V.conditions for a in ("yes", "no")] + [("DIQ010", "borderline")]


@pytest.mark.parametrize("item, answer", CONDITION_CASES, ids=[f"{q}-{a}" for q, a in CONDITION_CASES])
def test_a_condition_line_answers_its_item_and_only_it(item, answer):
    line = _round_trip(Fact("condition", item, answer=answer))
    p, (st,) = _the_line(line)
    expect = {"yes": 1, "no": 2, "borderline": 3}[answer]
    assert p.ok and st.outcome == "read", (line, p.all_problems(), p.not_understood)
    assert st.facts["conditions"] == {item: expect}, (line, st.facts)


def test_no_other_conditions_answers_the_rest_no_and_keeps_the_list():
    p = read_lines(HEADER + render(Fact("condition", "BPQ020", answer="yes")) + "\n"
                          + render(Fact("no_other_conditions")))
    assert p.ok and p.questionnaire["BPQ020"] == 1 and p.questionnaire["MCQ220"] == 2


@pytest.mark.parametrize("status, amount, expect", [
    ("NeverSmoker", "", ("NeverSmoker", 0)), ("FormerSmoker", "", ("FormerSmoker", 0)),
    ("CurrentSmoker", "", ("CurrentSmoker", 3)), ("CurrentSmoker", "moderate", ("CurrentSmoker", 2)),
    ("CurrentSmoker", "occasional", ("CurrentSmoker", 1)),
])
def test_a_smoking_line_reads_back_as_its_status_and_level(status, amount, expect):
    line = _round_trip(Fact("smoking", status, amount=amount))
    p, (st,) = _the_line(line)
    assert p.ok and (p.smoking, p.cotinine_level) == expect and st.facts == {"smoking": expect}


def test_every_smoking_status_of_the_kb_has_a_line():
    assert set(SMOKING_PHRASES) == set(V.smoking_statuses)


@pytest.mark.parametrize("sex", ["Male", "Female"])
def test_a_sex_line(sex):
    p = read_lines("58 year old\n" + _round_trip(Fact("sex", sex)))
    assert p.ok and p.sex == sex and p.statements[1].facts == {"sex": sex}


@pytest.mark.parametrize("age", range(20, 91))
def test_an_age_line(age):
    p = read_lines("male\n" + _round_trip(Fact("age", number=str(age))))
    assert p.ok and p.age == age and p.statements[1].facts == {"age": float(age)}


@pytest.mark.parametrize("number, unit, kg", [
    ("40", "kg", 40), ("72.5", "kg", 72.5), ("150", "kg", 150), ("90", "lb", 90 * 0.45359237),
    ("176", "lb", 176 * 0.45359237), ("330", "lb", 330 * 0.45359237)])
def test_a_weight_line(number, unit, kg):
    p, (st,) = _the_line(_round_trip(Fact("weight", number=number, unit=unit)))
    assert p.ok and p.weight_kg == pytest.approx(kg) and st.facts == {"weight_kg": pytest.approx(kg)}


@pytest.mark.parametrize("number, unit, cm", [
    ("150", "cm", 150), ("178", "cm", 178), ("1.78", "m", 178), ("70", "in", 177.8),
    ("5'10", "ft-in", 177.8), ("6'0", "ft-in", 182.88), ("4'11", "ft-in", 149.86)])
def test_a_height_line(number, unit, cm):
    p, (st,) = _the_line(_round_trip(Fact("height", number=number, unit=unit)))
    assert p.ok and p.height_cm == pytest.approx(cm) and st.facts == {"height_cm": pytest.approx(cm)}


@pytest.mark.parametrize("word", HEALTH_WORDS)
def test_a_self_rated_health_line(word):
    p, (st,) = _the_line(_round_trip(Fact("self_rated_health", word)))
    assert p.ok and st.facts == {"questionnaire": {"HUQ010": _HEALTH[word]}}


@pytest.mark.parametrize("word", TREND_WORDS)
def test_a_health_trend_line(word):
    p, (st,) = _the_line(_round_trip(Fact("health_vs_year_ago", word)))
    assert p.ok and st.facts == {"questionnaire": {"HUQ020": _TREND[word]}}


@pytest.mark.parametrize("n", range(0, 61))
def test_a_healthcare_visits_line(n):
    p, (st,) = _the_line(_round_trip(Fact("healthcare_visits", "year", number=str(n))))
    assert p.ok and st.facts == {"questionnaire": {"HUQ050": _visits_category(n)}}


@pytest.mark.parametrize("number", ["+4.5", "-2", "+0", "-12.3", "3"])
def test_a_grimage_line(number):
    p, (st,) = _the_line(_round_trip(Fact("grimage", number=number)))
    assert p.ok and st.facts == {"grimage": float(number)}


@pytest.mark.parametrize("number, unit, level", [
    ("0", "ng/mL", 0), ("9.9", "ng/mL", 0), ("10", "ng/mL", 1), ("99", "ng/mL", 1), ("150", "ng/mL", 2),
    ("200", "ng/mL", 3), ("450", "ng/mL", 3), ("0", "level", 0), ("1", "level", 1), ("2", "level", 2),
    ("3", "level", 3)])
def test_a_cotinine_line(number, unit, level):
    p, (st,) = _the_line(_round_trip(Fact("cotinine", number=number, unit=unit)))
    assert p.ok and p.cotinine_level == level and p.cotinine_measured and st.facts == {"cotinine": level}
    if unit != "level":
        assert _cotinine_level(float(number)) == level


@pytest.mark.parametrize("answer, expect", [("now", ["Metformin"]), ("not_now", [])])
def test_a_medication_line(answer, expect):
    p, (st,) = _the_line(_round_trip(Fact("medication", "Metformin", answer=answer)))
    assert p.ok and p.medications == expect and st.facts == {"medications": {
        "Metformin": "current" if answer == "now" else "not_current"}}


@pytest.mark.parametrize("piece", ["58 yo M", "non-smoker", "current smoker since 1990", "diagnoses: high bp",
                                   "medications: lisinopril", "albumin four g/dL", "", "weight: about 80 kg"])
def test_anything_else_is_not_canonical(piece):
    assert parse(piece) is None


def test_render_refuses_what_the_vocabulary_does_not_have():
    for fact in (Fact("smoking", "Vaper"), Fact("condition", "DIQ010", answer="maybe"),
                 Fact("condition", "BPQ020", answer="borderline"), Fact("weight", number="80", unit="st"),
                 Fact("height", number="70", unit="ft"), Fact("self_rated_health", "great"),
                 Fact("health_vs_year_ago", "same-ish"), Fact("unclear"), Fact("someone_else"),
                 Fact("smoking", "CurrentSmoker", detail="vaping"), Fact("healthcare_visits", "month", "2")):
        with pytest.raises((ValueError, KeyError)):
            render(fact)
