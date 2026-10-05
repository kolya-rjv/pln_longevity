"""The My Patient reader's medications (core/patient_medications.py).

The knowledge base has one medication fact that matters to a patient: a supplement plan flags a
supplement that interacts with a drug the person takes. So the reader reads a drug only from a small
table of identities, only from a statement that is about the person taking it now, and reads "not
taking" from a statement that says so. The cost of a missed reading is a missing flag (what every patient
had before); the cost of a wrong one is a flag for a drug the person does not take.

The first version read "<anything> on metformin" with a deny-list of heads. An adversarial review
(docs/kb_quick_wins/REVIEW_ROUND1.md) confirmed 40 ways it read a drug the person does not take, three
of them high (a medication tail swallowed the unread rest of a smoking clause and let "Smoker: 0 takes
metformin" build; "isn't on metformin" was read as TAKING it; a whitespace run made one regex quadratic:
19,000 characters stalled the process for 10 s). Every input the review found is pinned below
(NEVER_CURRENT), so the long tail stays closed.

    pytest tests/test_patient_medications.py -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.patient_builder import kb_interaction_drugs  # noqa: E402
from core.patient_medications import (  # noqa: E402
    _BLOCK, _OTHERS, MAX_STATEMENT_CHARS, MEDICATIONS, MedContext, MedRead, heading_kind, line_about_other,
    line_blocked, normalise, read_medication, starts_with_stop)
from core.patient_text import _head_understood, read_patient_text  # noqa: E402

M = ("Metformin",)
BASE = "58 year old male\n"


def _heads(text: str) -> bool:
    return text in {"type 2 diabetes", "diabetes", "has diabetes", "hba1c 6.1 %", "diabetes since 2015", "hypertension"}


def _read(text: str, **ctx):
    return read_medication(text, MedContext(**ctx), _heads)


# ═══════════════════════════ the table is the KB's ═══════════════════════════════

def test_every_drug_the_reader_can_read_is_a_pharmaceutical_the_kb_has_an_interaction_fact_for():
    """A drug the KB has no Interaction fact for would be an inert atom and a promise of a flag that can
    never come; a SUPPLEMENT (Berberine) is the other side of that fact and is never the drug. The set is
    read off the KB, so a new fact widens what is allowed."""
    assert kb_interaction_drugs() == frozenset({"Metformin"})
    assert set(MEDICATIONS.values()) <= kb_interaction_drugs()


def test_the_alias_table_is_exactly_identities_of_metformin():
    names = {"metformin", "metformin hcl", "metformin hydrochloride", "metformin er", "metformin xr",
             "metformin sr", "metformin ir", "metformin extended release", "metformin immediate release",
             "metformin slow release", "glucophage", "glucophage xr", "glumetza", "fortamet", "riomet"}
    assert set(MEDICATIONS) == names             # equal, not a superset: a drug that is not metformin fails here
    assert set(MEDICATIONS.values()) == {"Metformin"}


# ═══════════════════════════ the grammar ═════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "takes metformin", "Takes Metformin.", "I take metformin 500 mg twice daily", "I'm on metformin hcl 1000 mg",
    "we are taking metformin", "currently taking Glucophage XR", "on metformin", "I've been on metformin",
    "using metformin ER 750 mg daily", "medications: metformin", "Meds - metformin 500mg", "current medications: metformin",
    "my medications: metformin", "metformin 500 mg", "metformin 500 mg twice daily", "Glumetza 500 mg", "I take my metformin",
    "I take 500 mg of metformin", "I have been taking metformin for years", "takes metformin since 2015", "on metformin since 2015",
    "takes Metformin-XR", "takes metformin 1,000 mg", "takes metformin 500 mg/day", "taking metformin for 5 years",
    "I take metformin as prescribed", "now I take metformin", "I now take metformin 1000 mg", "metformin extended-release 500 mg",
])
def test_a_statement_about_taking_the_drug_is_a_current_medication(text):
    got = _read(text)
    assert got is not None and got.kind == "current" and got.symbols == M and got.head == ""


@pytest.mark.parametrize("text", [
    "no metformin", "not on metformin", "not taking metformin", "not currently on metformin", "stopped metformin",
    "stopped taking metformin", "I stopped taking metformin", "discontinued metformin", "never took metformin",
    "never been on metformin", "off metformin", "no longer takes metformin", "no longer on metformin",
    "doesn't take metformin", "do not take metformin", "I don't take metformin", "used to take metformin",
    "previously on metformin", "came off metformin", "isn't on metformin", "wasn't on metformin", "I am not on metformin",
])
def test_a_statement_that_says_the_person_is_not_taking_it_is_not_a_medication(text):
    assert _read(text) == MedRead("stopped", M)


@pytest.mark.parametrize("text, head", [
    ("type 2 diabetes on metformin", "type 2 diabetes"), ("diabetes (on metformin)", "diabetes"),
    ("has diabetes and takes metformin", "has diabetes"), ("HbA1c 6.1 % on metformin", "hba1c 6.1 %"),
    ("diabetes, taking metformin", "diabetes"), ("diabetes treated with metformin", "diabetes"),
    ("I have diabetes and take metformin", "diabetes"), ("diabetes since 2015 on metformin", "diabetes since 2015"),
])
def test_an_understood_head_in_front_of_on_a_drug_is_left_for_the_rest_of_the_reader(text, head):
    got = _read(text)
    assert got == MedRead("current", M, head=head)


@pytest.mark.parametrize("text", [
    "take metformin", "Take metformin 500 mg twice daily", "use metformin daily", "TAKE METFORMIN",     # the imperative
    "metformin", "metformin?", "metformin for diabetes", "metformin 500 mg made me nauseous",
    "he takes metformin", "she is on metformin", "John takes metformin", "my doctor takes metformin",   # not a subject we read
    "research on metformin", "should be on metformin", "advice on metformin", "years ago on metformin", "since 2015 on metformin",
    "I want to be on metformin", "doctor recommends taking metformin", "maybe on metformin",           # a head the reader does not understand
    "no medications", "medications: none", "takes no medications", "on a diet", "on and off",
    "metformin 0 mg", "metformin 0.0 mg twice daily",
])
def test_a_statement_that_is_not_a_closed_medication_form_is_not_read(text):
    assert _read(text) is None


def test_the_rest_of_a_statement_goes_on_to_be_read_as_it_would_have_been_alone():
    assert _read("takes metformin and lisinopril") == MedRead("current", M, rest="lisinopril")
    assert _read("takes metformin and has diabetes") == MedRead("current", M, rest="has diabetes")
    assert _read("takes metformin 500 mg and glucophage") == MedRead("current", M)       # one drug, once
    assert _read("takes lisinopril") is None and _read("on insulin") is None                # no symbol: not consumed


def test_a_head_that_the_reader_does_not_fully_understand_is_not_a_head():
    assert read_medication("hypertension on metformin", MedContext(), lambda text: False) is None
    assert _head_understood("type 2 diabetes") and _head_understood("hba1c 6.1 %")
    assert not _head_understood("no diabetes") and not _head_understood("stroke: - -")
    assert not _head_understood("cancer: 0 -") and not _head_understood("tummy trouble")
    assert not _head_understood("asthma, high cholesterol")           # something in it is dropped


def test_the_surroundings_of_a_statement_can_withdraw_it():
    assert _read("takes metformin", blocked=True) is None                    # a word of the line says not-now
    assert _read("takes metformin", next_stops=True) is None                 # the next line takes it back
    assert _read("metformin 500 mg", heading="other") is None                # "Allergies:" above
    assert _read("metformin 500 mg", heading="current") == MedRead("current", M)
    assert _read("takes metformin", after_other=True) is None                # the line above is about someone else
    assert _read("I take metformin", after_other=True) == MedRead("current", M)     # but "I" is the person
    assert _read("metformin") is None and _read("metformin", list_kind="current") == MedRead("current", M)
    assert _read("no metformin", blocked=True) == MedRead("stopped", M)      # a stop is still a stop


def test_a_long_statement_is_not_read_at_all():
    assert _read("takes metformin " + "and " * (MAX_STATEMENT_CHARS // 4)) is None


# ═══════════════════════════ the line decides ═════════════════════════════════════

def test_every_blocking_word_blocks_the_line_and_every_relative_is_someone_else():
    """Pins each word: delete one from the set and its own case fails."""
    for word in sorted(_BLOCK | _OTHERS):
        assert line_blocked(f"takes metformin, {word}"), word
        got = read_patient_text(f"{BASE}takes metformin, {word}")
        assert got.medications == [], word


@pytest.mark.parametrize("line", [
    "I do not, in fact, take metformin", "I take metformin, but not anymore", "In 2019, on metformin", "2015-2020 on metformin",
    "takes metformin; I don't take it", "Should I be on metformin", "on metformin?", "I wasn\u2019t on metformin",
    "I wasn\u00b4t on metformin", "I wasn`t on metformin", "I havent been on metformin", "I wasnt on metformin",
])
def test_negations_in_every_spelling_a_year_and_a_question_block_the_line(line):
    assert line_blocked(line)
    assert read_patient_text(BASE + line).medications == []


def test_only_since_a_year_keeps_a_year_current():
    assert not line_blocked("takes metformin since 2015") and line_blocked("takes metformin 2015-2020")
    assert not line_blocked("takes metformin") and line_blocked("takes metformin 3 years ago")


def test_headings_and_the_lines_around_them():
    assert heading_kind("Medications:") == "current" and heading_kind("Current medications -") == "current"
    for other in ("Allergies:", "Past medications:", "Stopped:", "Not taking:", "Plan:", "Discontinued medications", "Wife"):
        assert heading_kind(other) == "other", other
    assert heading_kind("albumin 4.1 g/dL") is None and heading_kind("takes metformin") is None
    assert starts_with_stop("stopped 2021") and starts_with_stop("I quit it in June") and starts_with_stop("Status: stopped")
    assert not starts_with_stop("albumin 4.1 g/dL")
    assert line_about_other("My mother has diabetes") and not line_about_other("I have diabetes")
    assert normalise("I wasn\u2019t") == "i wasnt" and normalise("1,000mg") == "1000 mg"


# ═══════════════════════════ through the reader ══════════════════════════════════

@pytest.mark.parametrize("text, current, stopped", [
    ("takes metformin", ["Metformin"], []),
    ("diabetes on metformin", ["Metformin"], []),
    ("has diabetes and takes metformin", ["Metformin"], []),
    ("I have diabetes and take metformin", ["Metformin"], []),
    ("diagnoses: type 2 diabetes\nmedications: metformin", ["Metformin"], []),
    ("diagnoses: hypertension, on metformin", ["Metformin"], []),
    ("medications: lisinopril, metformin", ["Metformin"], []),
    ("Medications: lisinopril, aspirin, metformin", ["Metformin"], []),
    ("medications:\nmetformin 500 mg", ["Metformin"], []),
    ("takes metformin and lisinopril", ["Metformin"], []),
    ("takes metformin and has diabetes", ["Metformin"], []),
    ("albumin 4.1 g/dL\nI take metformin 500 mg twice daily", ["Metformin"], []),
    ("stopped metformin", [], ["Metformin"]),
    ("no metformin", [], ["Metformin"]),
    ("takes lisinopril", [], []),
    ("no medications", [], []),
    ("my HbA1c was 9 % before metformin", [], []),
])
def test_the_reader_records_what_the_person_takes_and_says_they_do_not(text, current, stopped):
    p = read_patient_text(BASE + text)
    assert p.ok, p.all_problems()
    assert (p.medications, p.medications_stopped) == (current, stopped)


def test_the_condition_in_front_of_a_medication_is_still_read_as_before():
    p = read_patient_text(BASE + "type 2 diabetes on metformin")
    assert p.questionnaire["DIQ010"] == 1 and p.medications == ["Metformin"]
    p = read_patient_text(BASE + "HbA1c 6.1 % on metformin")
    assert [r.code for r in p.readings] == ["LBXGH"] and p.medications == ["Metformin"]
    p = read_patient_text(BASE + "takes metformin and has hypertension")
    assert p.questionnaire["BPQ020"] == 1 and p.medications == ["Metformin"]
    p = read_patient_text(BASE + "takes metformin and asthma")        # a condition after the drug is not lost
    assert p.questionnaire["MCQ010"] == 1 and p.medications == ["Metformin"]


NEVER_CURRENT = [
    '2015-2020 on metformin',
    '2019: on metformin',
    '58 year old male, CurrentSmoker ... + doctor recommends taking metformin',
    '<condition>: 0 - on metformin',
    'Allergies:\nmetformin 500 mg',
    'Am I on metformin',
    'Current smoker: 0 who takes metformin',
    'Discontinued medications\nmetformin 500 mg twice daily',
    'Doctor says, take metformin',
    'Doctor says: take metformin',
    'Family history: mother diabetes, takes metformin',
    'He should take metformin',
    "I ain't on metformin",
    'I am seldom on metformin',
    'I am unable to be on metformin',
    "I can't be on metformin",
    "I can't take metformin",
    'I cannot be on metformin',
    "I couldn't be on metformin",
    'I do not take metformin',
    'I do not, in fact, take metformin',
    'I do not. Take metformin',
    "I don't take metformin with meals",
    'I dont keep taking metformin',
    'I dont take metformin',
    'I forgot to take metformin',
    'I havent been on metformin',
    'I haven´t been on metformin',
    'I hope to be on metformin',
    'I isnt on metformin',
    'I might be on metformin',
    'I might, if my doctor agrees, take metformin',
    "I mustn't be on metformin",
    'I never quit smoking on metformin',
    'I never stopped smoking and take metformin',
    'I never took lisinopril, metformin 500 mg',
    'I never, ever, take metformin',
    'I no longer take lisinopril and metformin 500 mg',
    'I no longer take lisinopril, metformin',
    'I no longer take lisinopril, metformin 500 mg',
    'I no longer take metformin twice daily',
    'I plan to start taking metformin',
    'I refuse to be on metformin',
    'I should be on metformin',
    'I should not, according to my doctor, take metformin',
    "I shouldn't be on metformin",
    'I stopped, last year, taking metformin',
    'I switched to metformin',
    'I take 500 mg metformin twice daily',
    'I take lisinopril, atorvastatin and metformin',
    'I take metformin\nI quit it in June',
    'I take metformin 500 mg but stopped last week',
    "I take metformin 500 mg with dinner. I don't take metformin with breakfast.",
    'I take metformin as prescribed',
    "I take metformin every evening; I don't take metformin in the morning",
    'I take metformin!',
    'I take metformin, but not anymore',
    'I take metformin; I am no longer on it',
    'I take walks, metformin',
    'I use alcohol, metformin',
    'I used to take metformin 500 mg, I now take metformin 1000 mg',
    'I used to take metformin 500 mg, now I take metformin 1000 mg',
    'I used to, years ago, take metformin',
    'I want to be on metformin',
    'I want to start metformin',
    'I wasn`t on metformin',
    'I wasnt on metformin',
    'I wasn´t on metformin',
    'I wasn’t on metformin',
    'I will start taking metformin',
    "I won't be taking metformin",
    "I'll start taking metformin",
    "I'm not on metformin",
    "I'm on metformin, but stopped last week",
    'If ..., I would take metformin',
    'If HbA1c is above 7, take metformin',
    'In 2019, on metformin',
    'John takes metformin',
    'Medication: metformin 500 mg\nStatus: stopped',
    'Medications I used to take:\n- metformin 500 mg',
    'Medications: metformin, stopped last week',
    'Mr. Smith takes metformin',
    'My doctor told me, take metformin daily',
    'My friend, who is on metformin, says I should try it',
    'My mother has diabetes\ntakes metformin',
    'My mother, on metformin',
    'My wife has diabetes, takes metformin',
    'My wife takes metformin',
    'No. Metformin',
    'Not taking:\nmetformin 500 mg',
    'On metformin; no',
    'Past medications:\nmetformin 500 mg',
    'Past medications: metformin 500 mg',
    'Plan, take metformin',
    'Should I be on metformin, or not?',
    'Should I be on metformin?',
    'Should I take metformin?',
    'Smoker: - and I take metformin',
    'Smoker: - takes metformin',
    'Smoker: 0 takes metformin',
    'Stopped:\n- metformin 500 mg',
    'Stopped: lisinopril, metformin 500 mg',
    'TAKE METFORMIN',
    'Take Glucophage XR 500 mg',
    'Take metformin',
    'Take metformin 500 mg twice daily',
    'The label says, take metformin 500 mg twice daily',
    'Tomorrow, take metformin',
    'Wife\nmetformin 500 mg',
    'X, take metformin ...',
    'allergic to penicillin, metformin 500 mg',
    'allergies: sulfa, metformin 500 mg',
    'asthma - 0 - on metformin 500 mg',
    'be on / taking metformin',
    'cancer: 0 - on metformin',
    'cancer: 0 -- on metformin',
    'cancer: 0 – on metformin',
    'cancer: no - on metformin',
    'considering berberine, metformin 500 mg daily',
    'current smoker: - takes metformin',
    'd like to start taking metformin',
    "diabetes and I'd been on metformin",
    'diabetes and discontinued taking metformin',
    'diabetes, treated with metformin',
    'diagnoses: diabetes, I take metformin 500 mg twice daily',
    'diagnoses: diabetes, medications: metformin',
    'diagnoses: diabetes, metformin 500 mg',
    'discontinued atorvastatin 20 mg, metformin 500 mg',
    'doctor recommends taking metformin',
    'doctor wants me on metformin',
    'doctor wants me to start taking metformin',
    'does not take lisinopril, metformin 500 mg',
    'father is diabetic, on metformin',
    'glucophage er',
    'he takes metformin',
    'hypertension: 0 - 0 - on metformin',
    'in 2019 on metformin',
    "isn't on metformin",
    'isnt on metformin',
    'looking into taking metformin',
    'maybe on metformin',
    'metformin',
    'metformin ',
    'metformin 0 mg',
    'metformin 0.0 mg',
    'metformin 500 mg (stopped)',
    'metformin 500 mg - stopped',
    'metformin 500 mg, discontinued',
    'metformin 500 mg, never filled',
    'metformin 500 mg, stopped in 2020',
    'metformin 500 mg, stopped last month',
    'metformin 500 mg; stopped',
    'metformin hcl er',
    'mom: diabetes; on metformin',
    'my doctor is taking metformin',
    'my father takes metformin',
    'my grandma takes metformin',
    'my nan takes metformin 500 mg twice daily',
    'my neighbour takes metformin',
    'my stepmother is on metformin',
    'no allergies and metformin',
    'no allergies, metformin',
    'no allergies, metformin\ntakes metformin',
    'no longer takes lisinopril, metformin 500 mg',
    'no pain and metformin 500 mg',
    'no pain, metformin',
    'no problems, metformin',
    'nobody on metformin',
    'none on metformin',
    'not on insulin, metformin',
    'not on lisinopril, metformin 1000 mg daily',
    'not sure, metformin',
    'not taking aspirin, metformin 500 mg',
    'off lisinopril, metformin 500 mg',
    'off work, metformin',
    'on disability, metformin',
    'on holiday, metformin',
    'on metformin currently',
    'on metformin during 2019',
    'on metformin for diabetes',
    'on metformin until 2019',
    'on metformin, no longer',
    'on vacation, metformin',
    'possibly on metformin',
    'rarely takes metformin',
    'relatives on metformin',
    'reluctant to start taking metformin',
    'research on metformin',
    'she is on metformin',
    'should I be on metformin',
    'should be on metformin',
    'since 2015 on metformin',
    'smoker (-) takes metformin',
    'someone on metformin',
    'sometimes takes metformin',
    'started/restarted/resumed metformin',
    'stopped lisinopril, metformin 500 mg',
    'stopped metformin',
    'stopped metformin 500 mg twice daily',
    'stopped metformin, now taking it again',
    'stopped metformin, restarted',
    'stroke: - - on metformin',
    'take metformin',
    'take metformin daily',
    'take/use metformin ...',
    'takes lisinopril, aspirin, metformin',
    'takes metformin 500 mg ER',
    'takes metformin 500 mg per day',
    'takes metformin regularly',
    'takes metformin sometimes',
    'takes metformin, sometimes',
    'takes metformin, stopped in 2020',
    'thinking of starting metformin',
    'unless on metformin',
    'use metformin daily',
    'was on lisinopril, metformin 500 mg',
    'was on metformin in 2019',
    'who is not on metformin',
    'would it help to be on metformin',
    'years ago on metformin',
]


@pytest.mark.parametrize("text", NEVER_CURRENT, ids=[t.replace("\n", " / ")[:60] for t in NEVER_CURRENT])
def test_what_the_adversarial_review_found_is_never_read_as_a_current_medication(text):
    p = read_patient_text(BASE + text)
    assert p.medications == [], (text, p.medications, p.notes)


@pytest.mark.parametrize("text", [
    "58 year old male isn't on metformin", "58 year old male who isn't on metformin", "58 year old male on metformin until 2019",
    "58 year old male on metformin 5 years ago", "58 year old male metformin 500 mg until 2020", "58 year old male on metformin in 2019",
    "58 year old male who hasn't been on metformin", "58 year old male isn't taking metformin",
])
def test_a_statement_that_also_carries_age_and_sex_is_not_read_for_a_medication(text):
    """The reader rewrites such a statement (drops 'until 2019', turns "isn't" into "isn t") before a medication
    rule could see it; the medication goes on its own line."""
    assert read_patient_text(text).medications == []


SMOKING_BASES = ["I never quit smoking", "I never stopped smoking", "Smoker: 0", "Smoker: -", "smoker (-)", "current smoker: -",
                 "Current smoker: 0"]


@pytest.mark.parametrize("base", SMOKING_BASES)
@pytest.mark.parametrize("tail", [" takes metformin", " and I take metformin", " who takes metformin", " on metformin"])
def test_a_medication_never_makes_the_reader_accept_a_smoking_clause_it_refused(base, tail):
    """The medication tail used to swallow the unread rest of the smoking clause, bypassing the guard that
    refuses it: 'Smoker: 0 takes metformin' built a current smoker with cotinine level 3 (+8.8 years)."""
    assert not read_patient_text(BASE + base).ok
    p = read_patient_text(BASE + base + tail)
    assert not p.ok and p.medications == [] and p.medications_stopped == []


def test_a_drug_mentioned_where_it_could_not_be_read_is_not_counted_and_the_text_says_so():
    p = read_patient_text(BASE + "takes metformin\nmy HbA1c was 9 % before metformin")
    assert p.medications == []
    assert any("also mentions Metformin but could not be read" in n for n in p.notes)
    p = read_patient_text(BASE + "my mother takes metformin\nI take metformin")          # a set-aside mention is not a conflict
    assert p.medications == ["Metformin"] or p.medications == []                         # (the line above is about someone else)
    p = read_patient_text(BASE + "takes metformin\nHbA1c 6.1 %\nthinking of stopping metformin")
    assert p.medications == []


def test_saying_both_that_you_take_it_and_that_you_do_not_is_a_contradiction_to_fix_in_either_order():
    for text in ("takes metformin\nHbA1c 6.1 %\nstopped metformin", "stopped metformin\nHbA1c 6.1 %\ntakes metformin"):
        p = read_patient_text(BASE + text)
        assert not p.ok and any("keep one" in str(x) and x.topic == "medication" for x in p.all_problems()), text
    adjacent = read_patient_text(BASE + "takes metformin\nstopped metformin")        # the next line takes it back
    assert adjacent.medications == [] and adjacent.medications_stopped == ["Metformin"]


def test_a_heading_a_stop_line_or_a_relative_above_or_below_withdraws_the_entry():
    for text in ("Past medications:\nmetformin 500 mg", "Allergies:\nmetformin 500 mg", "Discontinued medications\nmetformin 500 mg twice daily",
                 "metformin 500 mg\nstopped 2021", "takes metformin\nI quit it in June", "My mother has diabetes\ntakes metformin",
                 "Wife\nmetformin 500 mg", "stopped lisinopril, metformin 500 mg", "allergies: sulfa, metformin 500 mg"):
        assert read_patient_text(BASE + text).medications == [], text
    assert read_patient_text(BASE + "Medications:\nmetformin 500 mg").medications == ["Metformin"]


def test_the_reader_says_what_it_did_with_the_medication():
    notes = " | ".join(read_patient_text(BASE + "takes metformin and lisinopril").notes)
    assert "medication: Metformin read as a current medication" in notes and "no ranking" in notes
    notes = " | ".join(read_patient_text(BASE + "stopped metformin").notes)
    assert "not counted as a current medication" in notes


def test_the_payload_carries_the_medication_as_a_kb_symbol_list():
    payload, _ = read_patient_text(BASE + "CRP 3.1 mg/L\nHbA1c 6.4 %\ntakes metformin").to_patient("X")
    assert payload["medications"] == ["Metformin"]
    payload, _ = read_patient_text(BASE + "CRP 3.1 mg/L\nstopped metformin").to_patient("X")
    assert "medications" not in payload


# ═══════════════════════════ it cannot be made to stall ═══════════════════════════

HOSTILE = {
    "a whitespace run": BASE + "x" + " " * 19_000 + "!",
    "spaces around a parenthesis": BASE + "x" + " " * 9_000 + "(" + " " * 9_000 + "!",
    "x on metformin + spaces": BASE + "x on metformin" + " " * 19_000 + "!",
    "an adverb run": BASE + "x " + "now " * 4_990,
    "repeated timing": BASE + "takes metformin " + "twice daily " * 20 + "!",
    "repeated timing, bare": BASE + "metformin " + "twice daily " * 20 + "t",
    "a drug and letters": BASE + "metformin " + "t" * 22,
    "'and' run": BASE + "takes metformin " + "and " * 5_000,
    "a long line of entries": BASE + "metformin 500 mg " * 1_250,
    "twelve hundred lines": BASE + "takes metformin\n" * 1_200,
    "nested parentheses": BASE + "diabetes " + "(" * 5_000 + " on metformin",
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_no_text_the_review_threw_at_the_reader_stalls_it(name):
    """Before: 19,000 characters took 10 s, and the regex held the GIL, so it stalled the whole server."""
    started = time.perf_counter()
    read_patient_text(HOSTILE[name])
    assert time.perf_counter() - started < 0.5, name
