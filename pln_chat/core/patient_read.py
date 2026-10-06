"""Read the My Patient text: a model reads it, code verifies what the model says, the
person confirms what was read.

    read_patient(text, extractor)  -> PatientRead

With an extractor (core.patient_extract.OpenAIExtractor) the model reads the whole text and
returns items — a kind, enum keys, the number and unit as written, and the QUOTE that holds
them. Each item is checked against the text before it is used, by checks that do not depend
on how the person phrased anything:

* the quote is in the text (typography, case and spacing aside);
* every number the item carries is in its quote as a whole number ("5'1" is not in
  "5'11"; "10" is not in "105"), and so is a lab's unit, a weight's or a height's;
* an item inside a statement the model says is about someone else is not the person's.

An item that fails is dropped and listed with the reason. What passes becomes Facts
(core.patient_canonical), and core.patient_text.assemble applies the domain rules — units
and the NHANES ranges, contradictions, the smoking conventions, the questionnaire — exactly
as for canonical lines. Text that no item quotes is listed as not used, so the person sees
everything that was left out. The person then sees the reading and presses Build.

Without an extractor, or when the model cannot be reached, the text is read as canonical
lines (core.patient_text.read_lines) and the reading says why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from core.patient_canonical import Fact
from core.patient_extract import SEX, STATUS, ExtractError
from core.patient_text import ParsedPatient, Span, assemble, feet_inches, number, read_lines, unify

#: words that carry nothing on their own — filler, and a measurement's label left beside the
#: value the model quoted ("height" in "height 5'3''"): a piece made only of these is not "not used"
_FILLER = {"a", "an", "the", "i", "i'm", "im", "am", "is", "are", "and", "or", "but", "also", "my", "me",
           "with", "of", "to", "in", "on", "at", "for", "so", "very", "really", "just", "hi", "hello",
           "thanks", "please", "patient", "pt", "height", "weight", "tall", "age", "aged", "sex", "gender",
           "year", "years", "old", "yo", "inch", "inches", "ft", "feet", "foot", "lb", "lbs", "pounds", "kg",
           "cm"}
_WORD = re.compile(r"[^\W\d_]{2,}")
_OTHERS = "in what the model says is about someone else"
#: the words that state a sex (the prompt's own list): a sex item's quote must hold one
_SEX_WORDS = {"male": {"male", "man", "m", "gentleman", "masculine"},
              "female": {"female", "woman", "f", "lady", "feminine"}}
_UNIT_WORDS = {
    "kg": r"kg|kgs|kilo|kilos|kilogram|kilograms",
    "lb": r"lb|lbs|pound|pounds|#",
    "cm": r"cm|centimet(?:er|re)s?",
    "m": r"m|met(?:er|re)s?",
    "in": r"in|inch|inches|\"|''",
}


@dataclass
class PatientRead:
    """What was read from `text`, and how: `reader` is "model" or "lines" (canonical
    lines, no model). `discarded` holds the model's items that failed a check, as
    (quote, kind, why)."""
    text: str
    parsed: ParsedPatient
    reader: str = "lines"
    model: Optional[str] = None
    model_error: Optional[ExtractError] = None
    discarded: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False

    def header(self) -> str:
        if self.reader == "model":
            return (f"Read by {self.model}; every value was checked against your text"
                    + (" (cached)" if self.cached else ""))
        if self.model_error is not None:
            return (f"Read as one fact per line, without the model ({self.model_error.message}) — "
                    f"write lines like the examples")
        return "Read as one fact per line, without the model — write lines like the examples"

    def as_dict(self) -> dict:
        return {
            "reader": self.reader, "model": self.model,
            "model_error": None if self.model_error is None else
            {"code": self.model_error.code, "message": self.model_error.message,
             "configuration": self.model_error.is_config},
            "statements": [st.as_dict() for st in self.parsed.statements],
            "discarded": [{"quote": q, "kind": k, "why": w} for q, k, w in self.discarded],
        }


def read_patient(text: str, extractor=None) -> PatientRead:
    """Read `text`: with an extractor by the model, else as canonical lines."""
    text = text or ""
    if extractor is None:
        return PatientRead(text, read_lines(text))
    model = getattr(extractor, "model", None)
    try:
        ex = extractor(text)
    except ExtractError as exc:
        return PatientRead(text, read_lines(text), model=model, model_error=exc)
    except Exception as exc:                    # noqa: BLE001 — Read must still answer
        return PatientRead(text, read_lines(text), model=model,
                           model_error=ExtractError("upstream", f"{type(exc).__name__}: {exc}"[:300]))
    spans, discarded = verify(text, ex.items)
    return PatientRead(text, assemble(spans, lost=_lost(discarded)), reader="model", model=ex.model,
                       discarded=discarded, usage=dict(ex.usage), latency_s=ex.latency_s, cached=ex.cached)


#: kinds whose loss would not be a missing value but a wrong one: a dropped condition would be
#: answered No beside a list, a dropped smoking status would leave a smoker's cotinine to imputation
_ASK_IF_LOST = ("condition", "no_other_conditions", "smoking")


def _lost(discarded: list) -> list:
    """The dropped items the person must be asked about. A condition inside what the model says
    is about someone else is that person's, and dropping it is right; a smoking status there may
    be the person's own ("my wife and I smoke"), so it is asked about."""
    return [(q, kind, why) for q, kind, why in discarded if kind in _ASK_IF_LOST
            and not (kind != "smoking" and why == _OTHERS)]


# ═══════════════════════════ the checks ════════════════════════════════════════

def _fold(text: str) -> tuple[str, list[int]]:
    """`text` with typography unified, case folded and every run of whitespace one space —
    and, for each character kept, its offset in `text`."""
    out, where = [], []
    t = unify(text)
    for i, c in enumerate(t):
        if c.isspace():
            if out and out[-1] == " ":
                continue
            c = " "
        out.append(c.lower())
        where.append(i)
    return "".join(out), where


def locate(quote: str, lines: list[str]) -> Optional[tuple[int, int, int]]:
    """(line, start, end) of the first place `quote` is in the text as whole words ("male" is
    not in "female", "5'1" is not in "5'11"), or None."""
    q, _ = _fold(quote.strip())
    if len(q) < 3:
        return None
    for n, line in enumerate(lines):
        folded, where = _fold(line)
        i = folded.find(q)
        while i >= 0:
            j = i + len(q)
            if (i == 0 or not (folded[i - 1].isalnum() and q[0].isalnum())) and \
                    (j == len(folded) or not (folded[j].isalnum() and q[-1].isalnum())):
                return n, where[i], where[j - 1] + 1
            i = folded.find(q, i + 1)
    return None


#: a small count written as a word: "twice a month", "three visits"
_COUNT_WORDS = {0: "zero", 1: "one|once", 2: "two|twice", 3: "three|thrice", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _has_number(quote: str, num: str) -> bool:
    """`num` is in `quote` as a whole number — no digit, decimal point or comma-digit runs on
    — or, for a count up to twelve, as its word."""
    n = unify(num).strip().lstrip("+-")
    if not re.fullmatch(r"\d+(?:[.,]\d+)?", n):
        return False
    body = "[.,]".join(re.escape(part) for part in re.split(r"[.,]", n))   # 4.1 or 4,1
    if re.search(rf"(?<![\d.,]){body}(?![\d]|[.,]\d)", unify(quote)):
        return True
    words = _COUNT_WORDS.get(int(n)) if n.isdigit() else None
    return bool(words and re.search(rf"\b(?:{words})\b", unify(quote).lower()))


def _has_unit(quote: str, unit: str) -> bool:
    """A lab's unit as written is in the quote (spacing, case and µ/u aside)."""
    from core.patient_text import normalise_unit
    u = normalise_unit(unit)
    return bool(u) and u in normalise_unit(unify(quote))


def _has_unit_word(quote: str, unit: str) -> bool:
    return bool(re.search(rf"(?<![a-z])(?:{_UNIT_WORDS[unit]})(?![a-z])", unify(quote).lower()))


def _fact(raw: dict, quote: str) -> tuple[Optional[Fact], str]:
    """The Fact an item says, after the checks that need only its quote; or why not."""
    kind = raw.get("kind", "")
    if kind == "lab":
        if not _has_number(quote, raw["number"]):
            return None, "the number is not in the quote"
        unit = raw["unit"].strip()
        if unit and not _has_unit(quote, unit):
            return None, "the unit is not in the quote"
        return Fact("lab", raw["lab"], number=raw["number"].strip(), unit=unit), ""
    if kind == "cotinine":
        if not _has_number(quote, raw["number"]):
            return None, "the number is not in the quote"
        if not (_has_unit(quote, "ng/mL") if raw["unit"] == "ng/mL" else "level" in quote.lower()):
            return Fact("unclear", "cotinine", detail="no unit"), ""
        return Fact("cotinine", number=raw["number"].strip(), unit=raw["unit"]), ""
    if kind == "smoking":
        status = STATUS.get(raw["status"], "unclear")
        amount = raw["amount"] if status == "CurrentSmoker" and raw["amount"] != "unstated" else ""
        return Fact("smoking", status, amount=amount,
                    detail="" if raw["other_nicotine"] == "none" else raw["other_nicotine"]), ""
    if kind == "condition":
        from core.patient_vocabulary import vocabulary
        item = next(q for q, c in vocabulary().condition_info.items() if c.key == raw["condition"])
        return Fact("condition", item, answer=raw["answer"]), ""
    if kind == "no_other_conditions":
        return Fact("no_other_conditions"), ""
    if kind == "sex":
        words = set(re.findall(r"[a-z]+", re.sub(r"(?<=\d)(?=[a-z])", " ", unify(quote).lower())))  # 58M
        if not words & _SEX_WORDS[raw["sex"]]:
            return None, f"the quote does not say {raw['sex']}"
        return Fact("sex", SEX[raw["sex"]]), ""
    if kind == "age":
        if not _has_number(quote, raw["number"]):
            return None, "the number is not in the quote"
        return Fact("age", number=raw["number"].strip()), ""
    if kind in ("weight", "height"):
        unit, num = raw["unit"], raw["number"].strip()
        if unit == "ft-in":
            fi = feet_inches(num)
            if fi is None:
                return None, "not feet and inches"
            if not re.search(rf"(?<!\d){fi[0]}\s*(?:'|ft\b|feet|foot)\s*{fi[1]}(?!\d)", unify(quote).lower()):
                return None, "the height is not in the quote"
        else:
            if not _has_number(quote, num):
                return None, "the number is not in the quote"
            if not _has_unit_word(quote, unit):
                return Fact("unclear", kind, detail="no unit"), ""     # never the model's guess
        return Fact(kind, number=num, unit=unit), ""
    if kind == "medication":
        return Fact("medication", raw["drug"], answer=raw["taking"]), ""
    if kind == "self_rated_health":
        return Fact("self_rated_health", raw["rating"]), ""
    if kind == "health_vs_year_ago":
        return Fact("health_vs_year_ago", raw["trend"]), ""
    if kind == "healthcare_visits":
        if not _has_number(quote, raw["number"]):
            return None, "the number is not in the quote"
        return Fact("healthcare_visits", raw["period"], number=raw["number"].strip()), ""
    if kind == "grimage":
        if not _has_number(quote, raw["number"]):
            return None, "the number is not in the quote"
        if raw["wording"] != "acceleration":
            return Fact("unclear", "grimage", detail="a clock age, or not said to be an acceleration"), ""
        size = number(raw["number"])
        if size is None:
            return None, "the number is not a number"
        written = unify(raw["number"]).strip()
        if raw["direction"] == "signed" and written[:1] in "+-":
            value = written
        elif raw["direction"] == "older":
            value = f"+{abs(size):g}"
        elif raw["direction"] == "younger":
            value = f"-{abs(size):g}"
        elif raw["direction"] == "unstated" and written[:1] not in "+-":
            value = written
        else:
            return Fact("unclear", "grimage", detail="its sign is not clear"), ""
        return Fact("grimage", number=value), ""
    if kind == "someone_else":
        return Fact("someone_else"), ""
    if kind == "unclear":
        why = re.sub(r"[`*_\[\]<>]", "", str(raw.get("why", "")))[:160]
        return Fact("unclear", raw["topic"], detail=why), ""
    return None, f"no such kind {kind!r}"


def verify(text: str, items: list) -> tuple[list[Span], list]:
    """The model's items -> Spans of verified Facts (plus a Span for each stretch of text no
    item quotes), and the items that failed, as (quote, kind, why)."""
    lines = text.splitlines()
    discarded: list = []
    placed: list[tuple[tuple[int, int, int], Fact]] = []
    others: list[tuple[int, int, int]] = []
    covered: list[tuple[int, int, int]] = []
    for raw in items:
        quote, kind = str(raw.get("quote", "")), str(raw.get("kind", ""))
        where = locate(quote, lines)
        if where is None:
            discarded.append((quote[:120], kind, "the quote is not in the text"))
            continue
        covered.append(where)
        line, start, end = where
        typed = lines[line][start:end]
        fact, why = _fact(raw, typed)
        if fact is None:
            discarded.append((typed, kind, why))
            continue
        if fact.kind == "someone_else":
            others.append(where)
        placed.append((where, fact))

    def inside_other(w: tuple[int, int, int]) -> bool:
        return any(o[0] == w[0] and o[1] < w[2] and w[1] < o[2] for o in others)

    by_span: dict[tuple[int, int, int], list[Fact]] = {}
    for where, fact in placed:
        if fact.kind != "someone_else" and inside_other(where):
            discarded.append((lines[where[0]][where[1]:where[2]], fact.kind, _OTHERS))
            continue
        facts = by_span.setdefault(where, [])
        if fact not in facts:
            facts.append(fact)
    spans = [Span(line, start, end, lines[line][start:end], facts)
             for (line, start, end), facts in by_span.items()]
    spans += [Span(line, a, b, lines[line][a:b], unread=True) for line, a, b in _unused(lines, covered)]
    return spans, discarded


def _unused(lines: list[str], covered: list[tuple[int, int, int]]):
    """(line, start, end) of each stretch of text no item quotes that says something."""
    for n, line in enumerate(lines):
        mask = [False] * len(line)
        for ln, a, b in covered:
            if ln == n:
                for i in range(a, b):
                    mask[i] = True
        i = 0
        while i < len(line):
            if mask[i]:
                i += 1
                continue
            j = i
            while j < len(line) and not mask[j]:
                j += 1
            piece = line[i:j]
            words = [w for w in _WORD.findall(piece.lower()) if w not in _FILLER]
            if words:
                lead = len(piece) - len(piece.lstrip(" \t,;:.-–—•*"))
                trail = len(piece.rstrip(" \t,;:.-–—•*"))
                yield n, i + lead, i + max(trail, lead)
            i = j
