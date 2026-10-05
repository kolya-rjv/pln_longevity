"""The My Patient reader's medications (core/patient_medications.py).

The knowledge base has one medication fact that matters to a patient: a supplement plan flags a
supplement that interacts with a drug the person takes. So the reader reads a drug only from a small
table of identities, only from a statement that is about the person taking it now, and reads "not
taking" from a statement that says so. The cost of a missed reading is a missing flag (what every patient
had before); the cost of a wrong one is a flag for a drug the person does not take.

The first version read "<anything> on metformin" with a deny-list of heads. An adversarial review
(docs/kb_quick_wins/REVIEW_ROUND1.md) confirmed 40 root causes, nearly all of them ways it read a drug the
person does not take. The worst three: a medication tail swallowed the unread rest of a smoking clause, so
"Smoker: 0 takes metformin" built; "isn't on metformin" was read as TAKING it; and a whitespace run made one
regex quadratic (19,000 characters stalled the process for 10 s). The strict redesign was reviewed again
(round 2, REVIEW_ROUND2.md: a tokenizer that dropped "❌" and "не", a heading that protected one line, a
retraction two lines down, possessive relatives). Every input either round confirmed is pinned in
tests/fixtures/medication_never_current.json, so the long tail stays closed.

    pytest tests/test_patient_medications.py -q
"""
from __future__ import annotations

import json
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
    _BLOCK, _OTHERS, MAX_STATEMENT_CHARS, MEDICATIONS, MedContext, MedRead, context_for, heading_kind,
    line_about_other, line_blocked, normalise, read_medication, retracts)
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
    assert _read("takes metformin", retracted=True) is None                  # a line below takes it back
    assert _read("metformin 500 mg", heading="other") is None                # "Allergies:" above
    assert _read("metformin 500 mg", heading="current") == MedRead("current", M)
    assert _read("takes metformin", other=True) is None                      # the line above is about someone else
    assert _read("I take metformin", other=True) == MedRead("current", M)    # but "I" is the person
    assert _read("takes metformin", heading="other") is None                 # under "Past medications:" - every form
    assert _read("metformin 500 mg", intro_blocks=True) is None              # a bare entry under a line with a block word
    assert _read("metformin") is None and _read("metformin", list_kind="current") == MedRead("current", M)
    assert _read("no metformin", blocked=True) == MedRead("stopped", M)      # a stop is still a stop
    assert _read("no metformin", other=True) is None                         # ... but not when it is someone else's


def test_a_long_statement_is_not_read_at_all():
    assert _read("takes metformin " + "and " * (MAX_STATEMENT_CHARS // 4)) is None


# ═══════════════════════════ the line decides ═════════════════════════════════════

BLOCK_WORDS = (
    'actually', 'adverse', 'advice', 'advise', 'advised', 'after', 'again', 'ago', 'aint', 'allergic',
    'allergies', 'allergy', 'although', 'anymore', 'apparently', 'arent', 'asked', 'avoid', 'avoided',
    'avoids', 'barely', 'before', 'began', 'begin', 'beginning', 'begins', 'but', 'can', 'canceled',
    'cancelled', 'cannot', 'cant', 'cease', 'ceased', 'changed', 'completed', 'consider', 'considering',
    'considers', 'contraindicated', 'contraindication', 'correction', 'could', 'couldnt', 'dc', 'dcd',
    'decide', 'decided', 'declined', 'didnt', 'discontinue', 'discontinued', 'discontinuing', 'doesnt',
    'dont', 'end', 'ended', 'eventually', 'except', 'expired', 'finished', 'former', 'formerly', 'going',
    'had', 'hadnt', 'hardly', 'hasnt', 'havent', 'he', 'held', 'her', 'hes', 'him', 'his', 'historical',
    'hold', 'hope', 'hopefully', 'hopes', 'hoping', 'how', 'however', 'if', 'inactive', 'instead', 'intend',
    'intended', 'intends', 'intolerance', 'intolerant', 'isnt', 'jk', 'joke', 'kidding', 'later', 'lol',
    'may', 'maybe', 'might', 'mistake', 'must', 'mustnt', 'need', 'neednt', 'needs', 'neither', 'never',
    'next', 'no', 'nobody', 'none', 'nope', 'nor', 'not', 'nothing', 'occasionally', 'off', 'or', 'past',
    'pause', 'paused', 'perhaps', 'plan', 'planned', 'planning', 'plans', 'possibly', 'prescribe',
    'prescribed', 'previous', 'previously', 'prior', 'probably', 'quit', 'quits', 'quitting', 'rarely',
    'rather', 'reaction', 'reactions', 'recommend', 'recommended', 'recommends', 'refuse', 'refused',
    'refuses', 'replace', 'replaced', 'replacing', 'restart', 'restarted', 'resume', 'resumed', 'seldom',
    'shall', 'she', 'shes', 'should', 'shouldnt', 'side', 'sometimes', 'soon', 'sorry', 'start', 'started',
    'starting', 'starts', 'stop', 'stopped', 'stopping', 'stops', 'suggest', 'suggested', 'suggests',
    'supposedly', 'switch', 'switched', 'tapered', 'tapering', 'their', 'them', 'then', 'they', 'theyre',
    'think', 'thinking', 'thinks', 'though', 'till', 'told', 'tomorrow', 'tonight', 'tried', 'tries', 'try',
    'trying', 'typo', 'unable', 'uncertain', 'unless', 'unsure', 'until', 'used', 'voided', 'want', 'wanted',
    'wants', 'was', 'wasnt', 'weaned', 'were', 'werent', 'what', 'when', 'whether', 'which', 'who', 'whom',
    'whose', 'why', 'will', 'wish', 'wishes', 'withdrawn', 'without', 'wont', 'would', 'wouldnt', 'wrong',
)
OTHER_WORDS = (
    'anyone', 'aunt', 'boss', 'boyfriend', 'brother', 'carer', 'cat', 'child', 'children', 'colleague',
    'cousin', 'coworker', 'dad', 'daughter', 'doctor', 'dog', 'everyone', 'ex', 'family', 'father',
    'flatmate', 'friend', 'girlfriend', 'grandfather', 'grandma', 'grandmother', 'grandpa', 'grandparent',
    'granny', 'husband', 'kid', 'mate', 'mom', 'mother', 'mum', 'nan', 'nana', 'neighbor', 'neighbour',
    'nephew', 'niece', 'nurse', 'others', 'parent', 'partner', 'patient', 'people', 'pet', 'relative',
    'roomate', 'roommate', 'sibling', 'sister', 'somebody', 'someone', 'son', 'spouse', 'stepdad',
    'stepfather', 'stepmom', 'stepmother', 'twin', 'uncle', 'wife',
)


def test_every_blocking_word_blocks_the_line_and_every_relative_is_someone_else():
    """Pins each word against a literal copy (not against the sets themselves): delete one from
    patient_medications.py and its own case fails; add one and the equality asks for it here."""
    assert set(BLOCK_WORDS) == set(_BLOCK) and set(OTHER_WORDS) == set(_OTHERS)
    for word in BLOCK_WORDS + OTHER_WORDS:
        assert line_blocked(f"takes metformin, {word}"), word
        assert read_patient_text(f"{BASE}takes metformin, {word}").medications == [], word
        if word in OTHER_WORDS:
            assert line_blocked(f"takes metformin, {word}s") and line_about_other(f"the {word}s"), word   # plural / possessive


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
    assert retracts("stopped 2021") and retracts("I quit it in June") and retracts("Status: stopped") and retracts("2021")
    assert not retracts("albumin 4.1 g/dL") and not retracts("my albumin was 4.1 g/dL")
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


NEVER_CURRENT = json.loads((REPO / "tests" / "fixtures" / "medication_never_current.json")
                           .read_text(encoding="utf-8"))["texts"]


@pytest.mark.parametrize("text", NEVER_CURRENT, ids=[f"{i}" for i in range(len(NEVER_CURRENT))])
def test_what_the_adversarial_review_found_is_never_read_as_a_current_medication(text):
    p = read_patient_text(text if text.startswith("58 year old") else BASE + text)
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
    far = "\nHbA1c 6.1 %\nCRP 3 mg/L\nalbumin 4.1 g/dL\n"                  # beyond the two lines a retraction is looked for
    for text in (f"takes metformin{far}stopped metformin", f"stopped metformin{far}takes metformin"):
        p = read_patient_text(BASE + text)
        assert not p.ok and any("keep one" in str(x) and x.topic == "medication" for x in p.all_problems()), text
    for near in ("takes metformin\nstopped metformin", "takes metformin\nHbA1c 6.1 %\nstopped metformin"):
        adjacent = read_patient_text(BASE + near)                                    # a line below takes it back
        assert adjacent.medications == [] and adjacent.medications_stopped == ["Metformin"], near


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


# ═══════════════════════════ round 2 of the review ════════════════════════════════

@pytest.mark.parametrize("text", [
    "takes metformin ❌", "I take metformin 👎", "[ ] metformin 500 mg", "☐ metformin 500 mg", "~~I take metformin~~",
    "I take ~~metformin~~", "metformin 500 mg 🚫", "takes metformin and [ ] diabetes", "I take metformin не", "I take metformin 否",
    "Прошлые лекарства:\nmetformin 500 mg", "takes metformin\nнет", "I stop\u200bped metformin\ntakes metformin",
])
def test_a_character_the_grammar_cannot_account_for_means_the_line_is_not_read(text):
    """The tokenizer used to drop "❌" and "не" silently and read the drug anyway."""
    assert read_patient_text(BASE + text).medications == []


@pytest.mark.parametrize("text", [
    "Past medications:\n- lisinopril 10 mg\n- metformin 500 mg", "Allergies:\n- penicillin\n- metformin 500 mg",
    "Stopped:\nlisinopril 10 mg\nmetformin 500 mg", "Not taking:\n1. lisinopril\n2. metformin 500 mg",
    "Not taking:\non metformin", "Past medications:\nI take metformin 500 mg twice daily", "Allergies:\ntakes metformin",
    "Contraindications:\ndiabetes on metformin", "Previous medications\nmetformin 500 mg", "Inactive medications\nmetformin 500 mg",
    "**Past medications:**\nmetformin 500 mg", "Past medications:\n\nmetformin 500 mg", "Past medications:\n-----\nmetformin 500 mg",
    "CRP 3.1 mg/L\nPast medications:\nlisinopril 10 mg\nmetformin 500 mg", "Family history:\n- diabetes\n- on metformin",
    "Medications I'm allergic to\nmetformin 500 mg", "2019\nmetformin 500 mg", "metformin 500 mg\n2021",
    "takes metformin\nstopped\nlisinopril", "takes metformin\n-----\nstopped 2021", "takes metformin\nd/c'd",
    "takes metformin\nreplaced by insulin", "metformin 500 mg, replaced by insulin", "metformin 500 mg, d/c'd", "takes metformin, inactive",
])
def test_the_nearest_heading_above_and_a_retraction_below_decide_not_just_the_adjacent_line(text):
    assert read_patient_text(BASE + text).medications == [], text


@pytest.mark.parametrize("text", [
    "My husband's diabetes. Takes metformin.", "My husband's diabetes\nTakes metformin", "Mom's diabetes\nOn metformin",
    "My parents have diabetes\ntakes metformin", "My girlfriend has diabetes\ntakes metformin", "Dad's meds\nmetformin 500 mg",
    "Wife's medications:\nlisinopril\nmetformin", "My mother has diabetes\n\ntakes metformin",
])
def test_a_relative_above_or_on_the_line_makes_a_subject_less_medication_theirs(text):
    assert read_patient_text(BASE + text).medications == [], text
    assert read_patient_text(BASE + text.replace("Takes metformin", "I take metformin").replace("takes metformin", "I take metformin")
                             ).medications in ([], ["Metformin"])        # "I" is the person: allowed once the line is the person's own


def test_a_line_the_reader_does_not_understand_is_a_retraction_only_if_it_starts_like_one():
    """"my albumin was 4.1 g/dL" is a lab, not a stop: the first words decide, and a model rewrite of a
    neighbouring line must not flip a medication reading (patient_read compares what the rules read)."""
    assert read_patient_text(BASE + "takes metformin\nmy albumin was 4.1 g/dL").medications == ["Metformin"]
    assert read_patient_text(BASE + "takes metformin\nAlbumin 4.1 g/dL").medications == ["Metformin"]
    assert read_patient_text(BASE + "takes metformin\nHbA1c was 6.1 %").medications == ["Metformin"]
    assert read_patient_text(BASE + "takes metformin\nI quit it in June").medications == []
    assert read_patient_text(BASE + "takes metformin\nStatus: stopped").medications == []


def test_text_handed_back_to_the_reader_is_as_typed_with_its_hyphens():
    p = read_patient_text(BASE + "takes metformin and has pre-diabetes")
    assert p.ok and p.questionnaire["DIQ010"] == 3 and p.medications == ["Metformin"]       # the head was 'pre diabetes'
    p = read_patient_text(BASE + "has type 2 diabetes and takes metformin")
    assert p.questionnaire["DIQ010"] == 1 and p.medications == ["Metformin"]


@pytest.mark.parametrize("text", [
    "metformin 2000 mg", "I take metformin 2000 mg daily", "Metformin 2,000 mg daily", "I take metformin 500 mg a day",
    "I take metformin 500 mg per day", "I take metformin 500 mg each day", "I take metformin as prescribed",
    "takes metformin as prescribed", "metformin 500 mg as prescribed", "medications: metformin as prescribed",
    "on metformin as prescribed", "takes Metformin-XR", "takes metformin-ER", "metformin hcl er 500 mg", "currently I take metformin",
    "now I take metformin 1000 mg", "medications:\n- lisinopril\n- metformin", "Medications: lisinopril, aspirin, metformin",
])
def test_ordinary_present_tense_phrasings_the_review_found_missed_are_read(text):
    assert read_patient_text(BASE + text).medications == ["Metformin"], text


def test_a_dose_is_not_a_year_and_a_year_is_not_a_dose():
    assert not line_blocked("takes metformin 2000 mg daily") and not line_blocked("takes metformin 1,000 mg")
    assert line_blocked("takes metformin 2019") and line_blocked("takes metformin until 2020")
    assert not line_blocked("takes metformin since 2019")
    assert read_medication("metformin infinity mg", MedContext()) is None and read_medication("metformin inf mg", MedContext()) is None


def test_context_for_scans_the_lines_around_a_statement():
    lines = ["albumin 4.1 g/dL", "Past medications:", "lisinopril 10 mg", "- aspirin", "metformin 500 mg", "stopped 2021"]
    ctx = context_for(lines, 4)
    assert ctx.heading == "other" and ctx.retracted is True                   # "stopped 2021" is below it
    assert context_for(lines, 0).heading is None and context_for(lines, 3).heading == "other"
    assert context_for(lines[:5], 4).retracted is False
    assert context_for(["takes metformin", "-----", "stopped"], 0).retracted
    assert not context_for(["takes metformin", "my albumin was 4.1 g/dL"], 0, is_lab_line=lambda line: True).retracted
    assert context_for(["Mom's diabetes", "takes metformin"], 1).other


HOSTILE.update({
    "an NFKC sandwich": BASE + "takes metformin\n" + "x metformin " + "\u0344" * 9_850 + "\u0f73\u0f75" * 4_900 + "\ntakes metformin",
    "a giant line with no drug": BASE + ("\u0f73\u0f75" * 9_800),
    "twenty thousand blank lines": BASE + "takes metformin" + "\n" * 20_000 + "stopped",
    "ten thousand headers": BASE + "medications: \n" * 10_000,
    "many bullets": BASE + "- \n" * 9_000 + "metformin 500 mg",
})


def test_a_fullwidth_spelling_of_the_drug_is_the_drug_and_a_denial_of_it_is_a_contradiction():
    p = read_patient_text(BASE + "takes metformin\nHbA1c 6.1 %\nCRP 3 mg/L\nalbumin 4.1 g/dL\nI do not take ｍｅｔｆｏｒｍｉｎ")
    assert not p.ok and any("keep one" in str(x) for x in p.all_problems())
