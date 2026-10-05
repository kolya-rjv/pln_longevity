"""Canonical lines: what a model's reading of the My Patient text becomes before the
rules read it.

A model never produces a value. It names enum keys from core.patient_vocabulary and
quotes the text; code copies the number and unit out of the quote and writes ONE line
in the reader's own grammar — "albumin 4.1 g/dL", "former smoker", "diagnoses:
hypertension", "58 year old", "female" — which core.patient_text.read_patient_text then
reads like anything the person typed. `render` is that step. tests/test_patient_canonical.py
reads every line `render` can write back through the rules and checks it means exactly
the Fact it came from (the round trip), so a canonical line can never mean something
else to the reader than what the model chose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.patient_vocabulary import BORDERLINE, LabGroup, vocabulary

#: kinds a Fact can have, i.e. what a canonical line can say
KINDS = ("lab", "cotinine", "smoking", "condition", "no_other_conditions", "sex", "age", "weight",
         "height", "self_rated_health", "health_vs_year_ago", "healthcare_visits", "grimage")

SMOKING_PHRASES = {"NeverSmoker": "never smoked", "FormerSmoker": "former smoker",
                   "CurrentSmoker": "current smoker"}
OCCASIONAL_PHRASE = "occasional smoker"            # CurrentSmoker at cotinine level 1
SEX_PHRASES = {"Male": "male", "Female": "female"}
HEALTH_WORDS = ("excellent", "very good", "good", "fair", "poor")
TREND_WORDS = ("better", "worse", "about the same")
NO_OTHER_CONDITIONS = "no other conditions"


@dataclass(frozen=True)
class Fact:
    """One thing a canonical line says. `number` and `unit` are text copied from what the
    person typed (or a unit word the code chose: kg, lb, cm, m, in, ng/mL, level)."""
    kind: str
    key: str = ""          # lab group | condition item | smoking status | sex | health/trend word
    number: str = ""
    unit: str = ""
    answer: str = ""       # condition: yes | no | borderline
    occasional: bool = False


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
    """The canonical line for `f`. Raises ValueError for a Fact the vocabulary does not have."""
    v = vocabulary()
    if f.kind == "lab":
        group = v.labs[f.key]
        return f"{canonical_alias(group)} {f.number} {f.unit}".strip()
    if f.kind == "cotinine":
        if f.unit == "level":
            return f"cotinine level {f.number}"
        return f"cotinine {f.number} ng/mL"
    if f.kind == "smoking":
        if f.key not in SMOKING_PHRASES:
            raise ValueError(f"no smoking status {f.key!r}")
        if f.key == "CurrentSmoker" and f.occasional:
            return OCCASIONAL_PHRASE
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
    if f.kind == "self_rated_health":
        if f.key not in HEALTH_WORDS:
            raise ValueError(f"no health rating {f.key!r}")
        return f"self-rated health: {f.key}"
    if f.kind == "health_vs_year_ago":
        if f.key not in TREND_WORDS:
            raise ValueError(f"no trend {f.key!r}")
        return f"health compared to a year ago: {f.key}"
    if f.kind == "healthcare_visits":
        return f"healthcare visits in the past year: {f.number}"
    if f.kind == "grimage":
        return f"GrimAge acceleration {f.number} years"
    raise ValueError(f"no canonical line for a {f.kind!r}")


#: A statement already in canonical form for age, sex or smoking: typing it always
#: settles a disagreement with the model ("58 year old male" counts: two phrases).
_CANONICAL_WHO = re.compile(
    r"^(?:(?:\d{1,3} year old)?\s*(?:male|female)?|never smoked|former smoker|current smoker|"
    r"occasional smoker)$")


def is_canonical(statement: str) -> bool:
    """True when the statement is, word for word, a canonical age/sex/smoking phrase."""
    text = re.sub(r"\s+", " ", statement.strip().lower().rstrip("."))
    return bool(text) and bool(_CANONICAL_WHO.match(text))
