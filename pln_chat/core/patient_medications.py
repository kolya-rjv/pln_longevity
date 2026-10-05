"""Medications the My Patient reader understands: only a drug the knowledge base can act on.

The knowledge base has ONE medication fact that matters to a patient: a supplement plan flags a
supplement that interacts with a drug the person takes (`(Interaction Berberine Metformin …)`,
supplement_evidence.metta), against `(CurrentMedication <Patient> <Drug>)`. So the reader reads a
drug ONLY from the table below — names that are identities (the same substance: generic, salt,
formulation, brand), never similarity — and leaves every other drug exactly where it was, in "not
understood". It reads a medication only from a statement that is *about* the person taking it:

    takes metformin · I am taking Glucophage 500 mg twice daily · on metformin
    medications: metformin · metformin 500 mg · type 2 diabetes on metformin · HbA1c 6.1 % on metformin

and it reads "not taking" from a statement that says so, which is NOT a medication:

    no metformin · not on metformin · stopped metformin · never took metformin · off metformin

Anything else that merely mentions the drug ("my HbA1c was 9 % before metformin", "allergic to
metformin", "thinking of starting metformin", "was on metformin in 2019") is not read at all. The
cost of a missed reading is a missing flag, which is what every patient had before; the cost of a
wrong one is a flag for a drug the person does not take, so the grammar is strict and a drug that
is only suggested is not read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

# ── the drugs ────────────────────────────────────────────────────────────────

#: spelled form -> KB symbol. A symbol is listed only if supplement_evidence.metta holds an
#: (Interaction …) fact for it (tests/test_patient_medications.py checks that against the KB).
#: Combination products (sitagliptin/metformin, glyburide/metformin …) are deliberately absent.
_BASES = {
    "metformin": ("Metformin", ("", "hcl", "hydrochloride", "er", "xr", "sr", "ir",
                                "extended release", "extended-release", "immediate release",
                                "immediate-release", "slow release", "slow-release")),
    "glucophage": ("Metformin", ("", "xr")),
    "glumetza": ("Metformin", ("",)),
    "fortamet": ("Metformin", ("",)),
    "riomet": ("Metformin", ("",)),
}
MEDICATIONS: dict[str, str] = {
    (f"{base} {suffix}".strip()): symbol
    for base, (symbol, suffixes) in _BASES.items() for suffix in suffixes
}

_DRUG = "(?:" + "|".join(re.escape(n) for n in sorted(MEDICATIONS, key=len, reverse=True)) + ")"
_DOSE = r"\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg)(?:\s*/\s*(?:day|d))?"
_WHEN = (r"(?:once|twice|two times|three times|\d\s*(?:x|times?))\s*(?:a|per|/)?\s*(?:day|daily|week)\b"
         r"|daily|nightly|bid|tid|qd|qhs|every (?:day|morning|evening|night)"
         r"|(?:in the )?(?:morning|evening)|with (?:meals|food|breakfast|lunch|dinner)")
_EXTRA = rf"(?:{_DOSE}|{_WHEN})"
_DRUG_PHRASE = re.compile(rf"(?P<drug>{_DRUG})(?P<extras>(?:\s+{_EXTRA})*)")
#: a drug word we do not know: ONE plain word, then dose / timing
_OTHER_DRUG = re.compile(rf"(?P<word>[a-z][a-z\-]{{3,}})(?P<extras>(?:\s+{_EXTRA})*)")
#: words that are not a drug name: a list item made of one of them is not a medication list
_NOT_A_DRUG = re.compile(
    r"^(?:medications?|meds?|pills?|tablets?|drugs?|treatments?|therapy|diet|exercise|lifestyle|nothing|"
    r"none|supplements?|vitamins?|herbal|injections?|shots?|something|anything|stuff|them|any|other|"
    r"others?|regular|daily|usual|prescribed|prescriptions?|with|have|has|had|because|after|before|"
    r"since|until|never|stopped|quit|started|every|each|when|off|also)$")

# ── the frames ───────────────────────────────────────────────────────────────

_SUBJECT = r"(?:(?:i'm|i am|i've been|i have been|i|we're|we are|we)\s+)?"
_NOW = r"(?:(?:currently|now|also|still|regularly|presently|daily|just)\s+)*"
_CURRENT = re.compile(
    rf"^{_SUBJECT}(?:(?:am|are|have been|been)\s+)?{_NOW}"
    r"(?:taking|take|takes|on|using|use|uses)\s+(?:the\s+)?")
_HEADER = re.compile(r"^(?:(?:current|regular|daily)\s+)?(?:medications?|meds|medication list|rx|drugs)"
                     r"\s*(?:[:=]|-)\s*")
_STOPPED = re.compile(
    rf"^{_SUBJECT}(?:(?:am|are|was|were|have|has|had|do|does|did)\s+)?"
    r"(?:not|never|no longer|stopped|quit|discontinued|ceased|(?:have\s+)?come off|came off|off|"
    r"used to (?:take|be on|use)|previously (?:on|took|taking|taken|used)|formerly (?:on|taking|took)|no)\b\s*"
    r"(?:(?:currently|now|any)\s+)*(?:(?:on|taking|take|takes|took|taken|using|use|used|been on|having)\s+)?"
    r"(?:the\s+)?")
#: "<anything> on metformin", "has diabetes and takes metformin", "diabetes (on metformin)"
_TAIL = re.compile(
    r"^(?P<head>.+?)(?:\s*[,;(]\s*|\s+and\s+|\s+)"
    r"(?P<frame>(?:(?:currently|now|also|still)\s+)*(?:on|taking|takes|using|treated with|managed with|"
    r"controlled (?:with|on)))\s+(?P<list>.+?)\s*\)?$")
#: what must not be in the part before a tail: a past, a negation, a hedge or a plan
#: a head that is only a subject or an auxiliary is no head: "i'm on metformin"
_PRONOUN_HEAD = re.compile(r"^(?:i'm|i am|i've been|i have been|i|we're|we are|we|am|are|been|have been)$")
_BAD_HEAD = re.compile(
    r"\b(?:was|were|had|used to|previously|formerly|before|after|since|until|stopped|quit|no longer|not|"
    r"never|no|without|denies|denied|might|may|plan\w*|will|going to|considering|thinking|if|but|"
    r"allergic|intolerant)\b")
_SPLIT = re.compile(r"\s*(?:,|\band\b|&|\+|/)\s*")
_CONTRACTION = re.compile(r"\b(do|does|did|is|are|was|were|have|has|had)n't\b")


@dataclass(frozen=True)
class MedRead:
    kind: str                        # "current" | "stopped" (said not taken, or no longer)
    symbols: tuple[str, ...]         # KB symbols read, in the order given
    others: tuple[str, ...] = ()     # drug words the KB has no interaction fact for (not used)
    head: str = ""                   # what is left of the statement for the rest of the reader


def _parse_list(body: str, is_condition: Callable[[str], bool]
                ) -> Optional[tuple[tuple[str, ...], tuple[str, ...]]]:
    """'metformin 500 mg and lisinopril' -> (('Metformin',), ('lisinopril',)); None when any item
    is neither a known drug nor a plain unknown drug word, or when it names a condition ('takes
    metformin and has diabetes' must not lose the diabetes as an unknown drug)."""
    symbols: list[str] = []
    others: list[str] = []
    for part in _SPLIT.split(body.strip(" .")):
        part = part.strip()
        if not part:
            continue
        m = _DRUG_PHRASE.fullmatch(part)
        if m:
            symbols.append(MEDICATIONS[m.group("drug")])
            continue
        m = _OTHER_DRUG.fullmatch(part)
        if m and not _NOT_A_DRUG.match(m.group("word")) and not is_condition(part):
            others.append(m.group("word"))
            continue
        return None
    return tuple(dict.fromkeys(symbols)), tuple(dict.fromkeys(others))


def read_medication(low: str, listed: Optional[str] = None,
                    is_condition: Callable[[str], bool] = lambda text: False) -> Optional[MedRead]:
    """Read one statement (lowercase) as a medication statement, or return None.

    `listed` is "current" or "stopped" when the previous statement on the same line was a
    medication frame of that kind: a bare drug after it ('takes lisinopril, metformin') is a
    list item. A MedRead with no `symbols` is a medication frame whose drugs the knowledge base
    has nothing on: the caller does not consume the statement, but may keep `kind` as the list's."""
    low = _CONTRACTION.sub(r"\1 not", low.replace("’", "'")).strip()
    for pattern, kind in ((_STOPPED, "stopped"), (_CURRENT, "current"), (_HEADER, "current")):
        m = pattern.match(low)
        if m:
            parsed = _parse_list(low[m.end():], is_condition)
            if parsed is not None:
                return MedRead(kind, *parsed)
            if pattern is not _CURRENT:
                return None
    bare = _DRUG_PHRASE.fullmatch(low.strip(" ."))
    if bare is not None:
        # a dose or a timing says it is a medication entry; a bare name only inside a list
        kind = "current" if bare.group("extras").strip() else listed
        return MedRead(kind, (MEDICATIONS[bare.group("drug")],)) if kind else None
    if listed is not None:
        parsed = _parse_list(low, is_condition)
        if parsed is not None and parsed[0]:
            return MedRead(listed, *parsed)
    m = _TAIL.match(low)
    if m and not _BAD_HEAD.search(m.group("head")) and not _PRONOUN_HEAD.match(m.group("head").strip()):
        parsed = _parse_list(m.group("list"), is_condition)
        if parsed is not None and parsed[0]:
            return MedRead("current", *parsed, head=m.group("head").strip(" ,;("))
    return None
