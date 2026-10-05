"""The My Patient reader's medications (core/patient_medications.py).

The knowledge base has one medication fact that matters to a patient: a supplement plan flags a
supplement that interacts with a drug the person takes. So the reader reads a drug only from a small
table of identities, only from a statement that is about the person taking it, and reads "not taking"
from a statement that says so. Everything else that merely mentions the drug is left unread — the cost
of a missed reading is a missing flag (what every patient had before), the cost of a wrong one is a flag
for a drug the person does not take.

    pytest tests/test_patient_medications.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.patient_builder import kb_interaction_drugs  # noqa: E402
from core.patient_medications import MEDICATIONS, MedRead, read_medication  # noqa: E402
from core.patient_text import _names_condition, read_patient_text  # noqa: E402

M = ("Metformin",)


def _read(text: str, listed=None):
    return read_medication(text.lower(), listed, _names_condition)


# ═══════════════════════════ the table is the KB's ═══════════════════════════════

def test_every_drug_the_reader_can_read_is_one_the_kb_has_an_interaction_fact_for():
    """A drug the KB has no Interaction fact for would be an inert atom and a promise of a flag
    that can never come. The set is read off the KB, so a new fact widens what is allowed."""
    assert kb_interaction_drugs() == frozenset({"Berberine", "Metformin"})
    assert set(MEDICATIONS.values()) <= kb_interaction_drugs()


def test_the_aliases_are_identities_not_similarities():
    assert {n for n, s in MEDICATIONS.items() if s == "Metformin"} >= {
        "metformin", "metformin hcl", "metformin hydrochloride", "metformin er", "metformin xr",
        "glucophage", "glucophage xr", "glumetza", "fortamet", "riomet"}
    # combination products are different products: deliberately absent
    assert not [n for n in MEDICATIONS if "janumet" in n or "sitagliptin" in n or "glyburide" in n]


# ═══════════════════════════ what is read ════════════════════════════════════════

@pytest.mark.parametrize("text", [
    "takes metformin", "Takes Metformin.", "I take metformin 500 mg twice daily", "I'm on metformin hcl 1000 mg",
    "we are taking metformin", "currently taking Glucophage XR", "on metformin", "I've been on metformin",
    "using metformin ER 750 mg daily", "medications: metformin", "Meds - metformin 500mg", "current medications: metformin",
    "metformin 500 mg", "metformin 500 mg twice daily", "Glumetza 500 mg",
])
def test_a_statement_about_taking_the_drug_is_a_current_medication(text):
    assert _read(text) == MedRead("current", M)


@pytest.mark.parametrize("text", [
    "no metformin", "not on metformin", "not currently taking metformin", "stopped metformin",
    "I stopped taking metformin", "discontinued metformin", "never took metformin", "off metformin",
    "no longer takes metformin", "doesn't take metformin", "do not take metformin", "used to take metformin",
    "previously on metformin", "came off metformin",
])
def test_a_statement_that_says_the_person_is_not_taking_it_is_not_a_medication(text):
    assert _read(text) == MedRead("stopped", M)


@pytest.mark.parametrize("text, head", [
    ("type 2 diabetes on metformin", "type 2 diabetes"), ("diabetes (on metformin)", "diabetes"),
    ("type 2 diabetes (on metformin 500 mg twice daily)", "type 2 diabetes"),
    ("has diabetes and takes metformin", "has diabetes"), ("HbA1c 6.1 % on metformin", "hba1c 6.1 %"),
    ("diabetes, taking metformin", "diabetes"), ("diabetes treated with metformin", "diabetes"),
])
def test_a_statement_that_ends_in_taking_it_leaves_the_rest_for_the_rest_of_the_reader(text, head):
    assert _read(text) == MedRead("current", M, head=head)


def test_a_bare_name_is_a_list_item_only_inside_a_list_and_otherwise_not_read():
    assert _read("metformin") is None
    assert _read("metformin", listed="current") == MedRead("current", M)
    assert _read("metformin", listed="stopped") == MedRead("stopped", M)
    assert _read("lisinopril, metformin", listed="current") == MedRead("current", M, others=("lisinopril",))
    assert _read("metformin 500 mg") == MedRead("current", M)          # a dose says it is an entry


def test_other_drugs_beside_it_are_noted_not_used_and_a_drug_alone_is_not_consumed():
    assert _read("takes metformin and lisinopril") == MedRead("current", M, others=("lisinopril",))
    assert _read("taking metformin, berberine and aspirin").others == ("berberine", "aspirin")
    assert _read("takes metformin 500 mg and glucophage") == MedRead("current", M)       # one drug, once
    only_others = _read("takes lisinopril")
    assert only_others == MedRead("current", (), others=("lisinopril",))        # no symbol: not consumed
    assert _read("on insulin").symbols == ()


# ═══════════════════════════ what is not read ════════════════════════════════════

@pytest.mark.parametrize("text", [
    # a past or a plan is not "takes"
    "was on metformin in 2019", "my HbA1c was 9 % before metformin", "HbA1c was 9 % on metformin",
    "thinking of starting metformin", "wants to try metformin", "recently started metformin",
    "I might take metformin", "plans to start metformin", "will take metformin",
    # only the name, with a hedge or a reason
    "allergic to metformin", "intolerant of metformin", "takes metformin sometimes",
    "takes metformin as needed", "metformin for diabetes", "metformin", "metformin?",
    # a negation of the condition in front of the drug, a condition after it
    "no diabetes on metformin", "takes metformin and has diabetes",
    # not a drug at all
    "no medications", "medications: none", "takes no medications", "on a diet", "on and off",
])
def test_a_statement_that_only_mentions_the_drug_is_not_read(text):
    assert _read(text) is None


@pytest.mark.parametrize("text", ["takes lisinopril", "on insulin", "medications: lisinopril", "no lisinopril"])
def test_a_drug_the_kb_has_nothing_on_never_yields_a_symbol(text):
    got = _read(text)
    assert got is None or got.symbols == ()


# ═══════════════════════════ through the reader ══════════════════════════════════

BASE = "58 year old male\n"


@pytest.mark.parametrize("text, current, stopped", [
    ("takes metformin", ["Metformin"], []),
    ("diabetes on metformin", ["Metformin"], []),
    ("has diabetes and takes metformin", ["Metformin"], []),
    ("diagnoses: type 2 diabetes\nmedications: metformin", ["Metformin"], []),
    ("medications: lisinopril, metformin", ["Metformin"], []),
    ("takes lisinopril, metformin", ["Metformin"], []),
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


def test_saying_both_that_you_take_it_and_that_you_do_not_is_a_contradiction_to_fix():
    p = read_patient_text(BASE + "takes metformin\nstopped metformin")
    assert not p.ok and any("keep one" in str(x) and "medication" in x.topic for x in p.all_problems())
    assert p.medications == ["Metformin"] and p.medications_stopped == []


def test_the_reader_says_what_it_did_with_the_medication():
    notes = " | ".join(read_patient_text(BASE + "takes metformin and lisinopril").notes)
    assert "medication: Metformin read as a current medication" in notes and "no ranking" in notes
    assert "has no interaction fact for (not used): lisinopril" in notes
    notes = " | ".join(read_patient_text(BASE + "stopped metformin").notes)
    assert "not counted as a current medication" in notes


def test_someone_elses_medication_is_set_aside_not_read():
    p = read_patient_text(BASE + "my father takes metformin")
    assert p.medications == [] and p.set_aside


def test_the_payload_carries_the_medication_as_a_kb_symbol_list():
    payload, _ = read_patient_text(BASE + "CRP 3.1 mg/L\nHbA1c 6.4 %\ntakes metformin").to_patient("X")
    assert payload["medications"] == ["Metformin"]
    payload, _ = read_patient_text(BASE + "CRP 3.1 mg/L\nstopped metformin").to_patient("X")
    assert "medications" not in payload
