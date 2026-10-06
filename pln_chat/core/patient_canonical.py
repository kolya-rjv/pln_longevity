"""Facts, and the canonical lines that say them.

A Fact is one thing read about the person: an age, a lab value as typed, a condition with
its answer. Two readers produce Facts — the model (core.patient_read, from free text) and
`parse` here (from canonical lines, with no model) — and core.patient_text.assemble turns
either list into the ParsedPatient that the rest of the system builds from.

The canonical lines are the format a person can type without a model, one fact per piece:

    58 year old male, current smoker
    albumin 4.1 g/dL
    blood pressure 142/88
    weight 61 kg
    height 5'3"
    diagnoses: hypertension, prediabetes
    medications: metformin

`render(fact)` writes a Fact as its line and `parse(piece)` reads a line back into Facts —
exactly these forms and nothing else: a piece that is not one of them is None, and the
caller lists it as not understood. tests/test_patient_canonical.py checks
`parse(render(f)) == [f]` over the whole vocabulary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from core.patient_vocabulary import BORDERLINE, LabGroup, vocabulary

#: kinds a Fact can have. The last two are not facts about the person: what the model says
#: is about someone else, and what it says it cannot tell (key = topic, detail = why).
KINDS = ("lab", "cotinine", "smoking", "condition", "no_other_conditions", "sex", "age", "weight",
         "height", "medication", "self_rated_health", "health_vs_year_ago", "healthcare_visits", "grimage",
         "someone_else", "unclear")

SMOKING_PHRASES = {"NeverSmoker": "never smoked", "FormerSmoker": "former smoker",
                   "CurrentSmoker": "current smoker"}
#: a current smoker's amount -> its line (cotinine level 1 and 2; unstated is "current smoker", level 3)
AMOUNT_PHRASES = {"occasional": "occasional smoker", "moderate": "moderate smoker"}
SEX_PHRASES = {"Male": "male", "Female": "female"}
HEALTH_WORDS = ("excellent", "very good", "good", "fair", "poor")
TREND_WORDS = ("better", "worse", "about the same")
NO_OTHER_CONDITIONS = "no other conditions"


@dataclass(frozen=True)
class Fact:
    """One thing read about the person. `number` and `unit` are as typed (a lab's unit is
    the person's own text, "" if none); the other units are kg | lb, cm | m | in | ft-in
    (number "5'3"), ng/mL | level. `answer` is yes | no | borderline for a condition and
    now | not_now for a medication; `key` is the lab group, condition item, smoking status
    (or "unclear"), sex, rating, trend, drug, visits period or unclear topic; `amount` is a
    current smoker's "occasional" or "moderate"; `detail` is a smoking item's other nicotine,
    or an unclear item's reason."""
    kind: str
    key: str = ""
    number: str = ""
    unit: str = ""
    answer: str = ""
    amount: str = ""
    detail: str = ""


def canonical_alias(group: LabGroup) -> str:
    """The name a canonical lab line uses: the reader's label when it is itself a name the
    reader knows ("Albumin", "HbA1c"), else the group's first name ("BUN", "lymphocytes")."""
    if group.fasting:
        return "fasting " + group.labels[0].lower()
    if len(group.labels) == 1 and group.labels[0].lower() in group.aliases:
        return group.labels[0]
    first = group.aliases[0]
    return first.upper() if len(first) <= 4 and first.isalpha() else first


def render(f: Fact) -> str:
    """The canonical line for `f`. Raises ValueError for a Fact that has none."""
    v = vocabulary()
    if f.kind == "lab":
        group = v.labs[f.key]
        return f"{canonical_alias(group)} {f.number} {f.unit}".strip()
    if f.kind == "cotinine":
        if f.unit == "level":
            return f"cotinine level {f.number}"
        return f"cotinine {f.number} ng/mL"
    if f.kind == "smoking":
        if f.key not in SMOKING_PHRASES or f.detail:
            raise ValueError(f"no canonical line for smoking {f.key!r} ({f.detail!r})")
        if f.key == "CurrentSmoker" and f.amount in AMOUNT_PHRASES:
            return AMOUNT_PHRASES[f.amount]
        return SMOKING_PHRASES[f.key]
    if f.kind == "condition":
        c = v.condition_info[f.key]
        if f.answer == "yes":
            return f"diagnoses: {c.yes}"
        if f.answer == "no":
            return c.no
        if f.answer == "borderline" and f.key == "DIQ010":
            return f"diagnoses: {BORDERLINE}"
        raise ValueError(f"no answer {f.answer!r} for {f.key}")
    if f.kind == "no_other_conditions":
        return NO_OTHER_CONDITIONS
    if f.kind == "sex":
        return SEX_PHRASES[f.key]
    if f.kind == "age":
        return f"{f.number} year old"
    if f.kind == "weight":
        if f.unit not in ("kg", "lb"):
            raise ValueError(f"no weight unit {f.unit!r}")
        return f"weight {f.number} {f.unit}"
    if f.kind == "height":
        if f.unit == "ft-in":
            feet, inches = f.number.split("'")
            return f"height {feet}'{inches}\""
        if f.unit not in ("cm", "m", "in"):
            raise ValueError(f"no height unit {f.unit!r}")
        return f"height {f.number} {f.unit}"
    if f.kind == "medication":
        if f.answer == "now":
            return f"medications: {f.key.lower()}"
        if f.answer == "not_now":
            return f"not taking {f.key.lower()}"
        raise ValueError(f"no medication answer {f.answer!r}")
    if f.kind == "self_rated_health":
        if f.key not in HEALTH_WORDS:
            raise ValueError(f"no health rating {f.key!r}")
        return f"self-rated health: {f.key}"
    if f.kind == "health_vs_year_ago":
        if f.key not in TREND_WORDS:
            raise ValueError(f"no trend {f.key!r}")
        return f"health compared to a year ago: {f.key}"
    if f.kind == "healthcare_visits":
        if f.key != "year":
            raise ValueError(f"a canonical visits line is per year, not per {f.key!r}")
        return f"healthcare visits in the past year: {f.number}"
    if f.kind == "grimage":
        return f"GrimAge acceleration {f.number} years"
    raise ValueError(f"no canonical line for a {f.kind!r}")


# ═══════════════════════════ parse: a canonical piece -> Facts ══════════════════

_N = r"(\d+(?:\.\d+)?)"
_SEX_WORDS = {"male": "Male", "man": "Male", "female": "Female", "woman": "Female"}
_SMOKING_LINES = {phrase: status for status, phrase in SMOKING_PHRASES.items()}
_SIMPLE = (
    (re.compile(r"^(\d{1,3}) years? old(?: (male|female|man|woman))?$"), "age_sex"),
    (re.compile(r"^age:? (\d{1,3})$"), "age"),
    (re.compile(r"^(male|female|man|woman)$"), "sex"),
    (re.compile(r"^cotinine:? (\d+(?:\.\d+)?) ng/ml$"), "cotinine"),
    (re.compile(r"^cotinine level:? ([0-3])$"), "cotinine_level"),
    (re.compile(rf"^weight:? {_N} ?(kg|lb|lbs)$"), "weight"),
    (re.compile(rf"^height:? {_N} ?(cm|m|in)$"), "height"),
    (re.compile(r"^height:? (\d)'(\d{1,2})\"$"), "height_ftin"),
    (re.compile(r"^grimage acceleration:? ([+-]?\d+(?:\.\d+)?) years?$"), "grimage"),
    (re.compile(r"^self-rated health: (excellent|very good|good|fair|poor)$"), "health"),
    (re.compile(r"^health compared to a year ago: (better|worse|about the same)$"), "trend"),
    (re.compile(r"^healthcare visits in the past year: (\d+)$"), "visits"),
    (re.compile(r"^(?:blood pressure|bp):? (\d{2,3})/(\d{2,3})(?: (mmhg))?$"), "bp"),
    (re.compile(r"^not taking (\w+)$"), "not_taking"),
    # a number whose unit (or meaning) is missing: asked about, never guessed
    (re.compile(r"^(weight|height|cotinine):? \d+(?:\.\d+)?$"), "no_unit"),
    (re.compile(r"^grimage(?: age)?:? \d+(?:\.\d+)?(?: years?)?$"), "grim_clock"),
)
LIST_HEAD = re.compile(r"^(diagnoses|diagnosis|medications|medication):\s*", re.I)


@lru_cache(maxsize=1)
def _lab_names() -> tuple[re.Pattern, dict]:
    """Every lab name the vocabulary has, longest first, and the group each one names."""
    group_of = {alias: key for key, g in vocabulary().labs.items() for alias in g.aliases}
    names = sorted(group_of, key=len, reverse=True)
    rx = re.compile(r"^(" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])\s*[:=]?\s*"
                    r"(\d+(?:[.,]\d+)?)\s*(.*)$", re.I)
    return rx, group_of


@lru_cache(maxsize=1)
def _condition_names() -> tuple[dict, dict]:
    """A condition's canonical Yes and No phrase -> (item, answer)."""
    yes = {c.yes.lower(): c.item for c in vocabulary().condition_info.values()}
    yes[BORDERLINE] = "DIQ010"
    no = {c.no.lower(): c.item for c in vocabulary().condition_info.values()}
    return yes, no


def _drug(word: str) -> Optional[str]:
    from core.patient_builder import kb_interaction_drugs
    return next((d for d in kb_interaction_drugs() if d.lower() == word.strip().lower()), None)


def _list(head: str, body: str) -> Optional[list[Fact]]:
    items = [x.strip() for x in re.split(r"[,;]", body) if x.strip()]
    if not items:
        return None
    out: list[Fact] = []
    if head.startswith("diagnos"):
        yes, _ = _condition_names()
        for item in items:
            key = yes.get(item.lower())
            if key is None:
                return None
            out.append(Fact("condition", key, answer="borderline" if item.lower() == BORDERLINE else "yes"))
    else:
        for item in items:
            drug = _drug(item)
            if drug is None:
                return None
            out.append(Fact("medication", drug, answer="now"))
    return out


def parse(piece: str) -> Optional[list[Fact]]:
    """The Facts a canonical piece says, or None when it is not one of the canonical forms."""
    text = re.sub(r"\s+", " ", piece.strip().rstrip(".")).lower()
    head = LIST_HEAD.match(text)
    if head:
        return _list(head.group(1), text[head.end():])
    if text in _SMOKING_LINES:
        return [Fact("smoking", _SMOKING_LINES[text])]
    for amount, phrase in AMOUNT_PHRASES.items():
        if text == phrase:
            return [Fact("smoking", "CurrentSmoker", amount=amount)]
    if text in ("no other conditions", "no known conditions"):
        return [Fact("no_other_conditions")]
    _, no = _condition_names()
    if text in no:
        return [Fact("condition", no[text], answer="no")]
    for rx, what in _SIMPLE:
        m = rx.match(text)
        if not m:
            continue
        if what == "age_sex":
            out = [Fact("age", number=m.group(1))]
            return out + ([Fact("sex", _SEX_WORDS[m.group(2)])] if m.group(2) else [])
        if what == "age":
            return [Fact("age", number=m.group(1))]
        if what == "sex":
            return [Fact("sex", _SEX_WORDS[m.group(1)])]
        if what == "cotinine":
            return [Fact("cotinine", number=m.group(1), unit="ng/mL")]
        if what == "cotinine_level":
            return [Fact("cotinine", number=m.group(1), unit="level")]
        if what == "weight":
            return [Fact("weight", number=m.group(1), unit="lb" if m.group(2).startswith("lb") else "kg")]
        if what == "height":
            return [Fact("height", number=m.group(1), unit=m.group(2))]
        if what == "height_ftin":
            return [Fact("height", number=f"{m.group(1)}'{m.group(2)}", unit="ft-in")]
        if what == "grimage":
            return [Fact("grimage", number=m.group(1))]
        if what == "health":
            return [Fact("self_rated_health", m.group(1))]
        if what == "trend":
            return [Fact("health_vs_year_ago", m.group(1))]
        if what == "visits":
            return [Fact("healthcare_visits", "year", number=m.group(1))]
        if what == "bp":
            unit = "mmHg" if m.group(3) else ""
            return [Fact("lab", _lab_names()[1]["systolic"], number=m.group(1), unit=unit),
                    Fact("lab", _lab_names()[1]["diastolic"], number=m.group(2), unit=unit)]
        if what == "no_unit":
            return [Fact("unclear", m.group(1), detail="no unit")]
        if what == "grim_clock":
            return [Fact("unclear", "grimage", detail="a clock age, not an acceleration")]
        if what == "not_taking":
            drug = _drug(m.group(1))
            return [Fact("medication", drug, answer="not_now")] if drug else None
    rx, group_of = _lab_names()
    m = rx.match(re.sub(r"\s+", " ", piece.strip().rstrip(".")))      # the unit as typed: "g/dL"
    if m:
        return [Fact("lab", group_of[m.group(1).lower()], number=m.group(2), unit=m.group(3).strip())]
    return None
