"""Read the My Patient text with the rules, and — only when asked — with a model too.

    read_patient(text)              the rules alone: core.patient_text.read_patient_text
    read_patient(text, extractor)   the rules, then a model (core.patient_extract), then
                                    the rules again on what the model rewrote

The model never produces a value. Its items are checked here, in code, against the
text: the quote must occur exactly once, inside one line; numbers and units are copied
out of the quote; every claim needs its own words in the quote (an alias of the lab, a
term of the condition, a smoking word, an explicit sex token, an age phrase). What
passes becomes a canonical line (core.patient_canonical.render) — and then:

* a statement the rules did NOT understand is replaced by its canonical line(s) in the
  "read as" text, which the rules read like anything typed ("· model" in the table);
* a statement the rules REFUSED (ambiguous, contradictory, unit, vaping, someone
  else, a condition that would be lost) is never replaced: the canonical wording is
  offered as a suggestion the person can click, and the refusal stands;
* a smoking status from the model alone is a suggestion too, never a substitution;
* a statement the rules read, which the model reads differently: smoking, age or sex
  blocks the build with both wordings as buttons (unless the statement is already in
  canonical form, which always wins); labs, conditions and the rest give a note.

The rules then read the "read as" text, and code checks that the rewrite added only
what it meant to: every statement it did not touch reads exactly as before, every
problem on those statements is still there, and every canonical line reads exactly as
it does alone. Otherwise the rewrite is withdrawn and offered as a suggestion.

Build never calls the model: it builds from the stored reading (patient_tab.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from core.patient_canonical import Fact, is_canonical, render
from core.patient_extract import SEX, STATUS, ExtractError, Extraction
from core.patient_text import (
    _ALIAS_INDEX,
    _MEDICAL_WORDS,
    _SOMEONE_ELSE,
    ParsedPatient,
    Problem,
    _outcomes,
    normalise_unit,
    read_patient_text,
)
from core.patient_vocabulary import vocabulary

Extractor = Callable[[str], Extraction]

# ═══════════════════════════ text normalisation ════════════════════════════════

_DASHES = set("‐‑‒–—―−﹘﹣－")
_SQUOTES = set("‘’‚‛′ʼ`´")
_DQUOTES = set("“”„‟″")


def _unify_char(c: str) -> str:
    if c in _DASHES:
        return "-"
    if c in _SQUOTES:
        return "'"
    if c in _DQUOTES:
        return '"'
    if c.isspace():
        return " "
    low = c.lower()
    return low if len(low) == 1 else c


def unify(s: str) -> str:
    """One character for one: dashes, quotes and spaces unified, lower case. Offsets
    into the result are offsets into `s`."""
    return "".join(_unify_char(c) for c in s)


def _collapse(s: str, loose: bool = False) -> tuple[str, list[int]]:
    """`s` unified, runs of spaces collapsed (and, if loose, punctuation dropped), with
    the offset in `s` of every character kept."""
    u = unify(s)
    out: list[str] = []
    idx: list[int] = []
    for i, c in enumerate(u):
        if loose and c in ",;:()[]{}!?\"*" :
            c = " "
        if loose and c == "." and not (0 < i < len(u) - 1 and u[i - 1].isdigit() and u[i + 1].isdigit()):
            c = " "
        if c == " " and (not out or out[-1] == " "):
            continue
        out.append(c)
        idx.append(i)
    return "".join(out), idx


_EDGE = " .,;:!?\"'"


def _locate(quote: str, lines: list[str]) -> tuple[Optional[tuple[int, int, int]], str]:
    """(line, start, end) of the one place `quote` occurs, or None and why not."""
    for loose in (False, True):
        q, _ = _collapse(quote, loose)
        q = q.strip(_EDGE)
        if len(q.replace(" ", "")) < 3:
            return None, "the quote is too short"
        if "\n" in quote.strip():
            return None, "the quote spans two lines"
        hits = []
        for n, line in enumerate(lines):
            c, idx = _collapse(line, loose)
            start = c.find(q)
            while start >= 0:
                hits.append((n, idx[start], idx[start + len(q) - 1] + 1))
                start = c.find(q, start + 1)
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            return None, "the quote occurs more than once"
    return None, "the quote is not in the text"


# ═══════════════════════════ what the model's items become ═════════════════════

@dataclass
class Item:
    """One model item, grounded: where its quote is, and what it was checked to mean."""
    raw: dict
    kind: str
    line: int
    start: int
    end: int
    quote: str                                # the text there, as typed
    statements: tuple = ()                    # the rules' statements it overlaps
    fact: Optional[Fact] = None               # what it says, if it can be written as a line
    problem: Optional[tuple] = None           # (text, kind, topic, wordings) — blocks
    note: str = ""


@dataclass
class Suggestion:
    """A wording the person can click to put in place of what they typed."""
    line: int
    start: int
    end: int
    original: str
    wordings: list                            # first: the model's reading; then the rules'
    reason: str
    blocking: bool = False                    # the build waits for it (a problem says why)

    def apply(self, text: str, wording: str) -> str:
        """`text` with this suggestion's span replaced by `wording`; unchanged if the
        span no longer holds what it held at Read."""
        lines = text.splitlines(keepends=True)            # numbered as the reader numbers them
        if self.line >= len(lines) or lines[self.line][self.start:self.end] != self.original:
            return text
        lines[self.line] = lines[self.line][:self.start] + wording + lines[self.line][self.end:]
        return "".join(lines)

    def as_dict(self) -> dict:
        return {"line": self.line, "start": self.start, "end": self.end, "original": self.original,
                "wordings": list(self.wordings), "reason": self.reason, "blocking": self.blocking}


@dataclass
class Substitution:
    line: int
    start: int
    end: int
    original: str
    lines: list
    statements: tuple


@dataclass
class PatientRead:
    """Everything Read produced: what Build builds from, and what the tab shows."""
    text: str                                 # as typed
    read_as: str                              # what the rules read (the model's rewrites in)
    parsed: ParsedPatient                     # the rules' reading of `read_as`, model problems added
    reader: str = "rules"                     # "rules" | "rules+model"
    model: Optional[str] = None
    model_error: Optional[ExtractError] = None
    substitutions: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    #: (quote, kind, why) of every model item that did not pass the checks
    discarded: list = field(default_factory=list)
    #: parsed.statements index -> the words the person typed, for rows from a model rewrite
    model_quotes: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False

    def source(self, statement: int) -> str:
        return "model" if statement in self.model_quotes else "rules"

    def header(self) -> str:
        if self.reader == "rules+model":
            return f"Read by rules + {self.model}"
        if self.model_error is None:
            return "Read by rules only"
        e = self.model_error
        why = {"no_key": "no OPENAI_API_KEY", "timeout": "the model did not answer in time",
               "refused": "the model declined", "truncated": "the model's answer was cut off",
               "too_long": e.message}.get(e.code, e.message)
        if e.is_config and e.code != "no_key":
            return f"Read by rules only — model reader misconfigured: {why}"
        return f"Read by rules only — {why}"

    def as_dict(self) -> dict:
        return {
            "reader_used": self.reader, "model": self.model,
            "model_error": None if self.model_error is None else
            {"code": self.model_error.code, "message": self.model_error.message,
             "configuration": self.model_error.is_config},
            "read_as_text": self.read_as,
            "suggestions": [s.as_dict() for s in self.suggestions],
            "model_notes": list(self.notes),
            "statements": [{**st.as_dict(), "source": self.source(st.index),
                            "typed": self.model_quotes.get(st.index, st.text)}
                           for st in self.parsed.statements],
        }


# ═══════════════════════════ checks per kind ═══════════════════════════════════

_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
_SMOKE_WORD = re.compile(r"smok|cig|tobacco|nicotin|\bpacks?\b|pack[- ]?years?|cigar|\bpipe|vap|e-?cig|snus|"
                         r"\bchew|zyn|juul|pouch|patch|nicotine gum|lozenge|weed|cannabis|marijuana|\bpot\b|joint")
_TOBACCO_WORD = re.compile(r"cig|tobacco|cigar|\bpacks?\b|\bpipe")
_NICOTINE_WORD = re.compile(r"vap|e-?cig|juul|snus|\bchew|\bdip\b|zyn|pouch|patch|nicotine gum|lozenge|nrt")
_OCCASIONAL = re.compile(r"\b(?:occasional(?:ly)?|social(?:ly)?|rarely|light(?:ly)?|some ?days?|weekends?|"
                         r"part(?:y|ies)|now and then|once in a while|sometimes|seldom|infrequent(?:ly)?)\b")
_FLAG = re.compile(r"^(?:h|l|hh|ll|high|low|normal|abnormal|elevated|raised|ok|borderline|critical|wnl|"
                   r"\*+|↑|↓|\(h\)|\(l\)|\(high\)|\(low\)|\(normal\))$")
_CONTEXT_WORD = re.compile(r"^(?:today|yesterday|on|in|at|last|this|from|taken|measured|done|recently|"
                           r"fasting|non-fasting|nonfasting|random|ref|reference|range|nr|normal)\b")
_LEAD_FILLER = re.compile(r"^(?:(?:my|the|a|his|her|latest|last|recent|most recent|today'?s|current|"
                          r"repeat|baseline)\s+)*$")
_RANGE = re.compile(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?")
_DATE = re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_NEG = r"(?:no|not|never|denies|denied|deny|negative for|without|free of|nor|none of|n't|no history of|no hx of)"
_HEADER = re.compile(r"^\W*(?:(?:past )?medical history|pmh|history|hx|diagnos[ie]s|dx|known conditions|"
                     r"conditions?|comorbidities|problems?(?: list)?)\s*[:\-=]?\s*")
_TRAIL_NEG = re.compile(r"^\s*(?:[:=\-]\s*)?(?:no|n|none|0|negative|neg|denies|denied|false|absent|"
                        r"ruled out|excluded|resolved)\b|\b(?:ruled out|excluded|resolved|negative)\b|\?")
_EMPTY_ANSWER = re.compile(r"^\s*[:=]\s*-*\s*$")


def _alias_spans(low: str) -> list[tuple[int, int, str]]:
    """Every lab name in `low`, longest first, never overlapping — as the reader picks them."""
    out: list[tuple[int, int, str]] = []
    for alias, _ in _ALIAS_INDEX:
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", low):
            if not any(m.start() < e and s < m.end() for s, e, _ in out):
                out.append((m.start(), m.end(), alias))
    return sorted(out)


def _group_of(alias: str):
    for g in vocabulary().labs.values():
        if alias in g.aliases:
            return g
    return None


def _check_lab(it: Item) -> str:
    v = vocabulary()
    chosen = v.labs[it.raw["lab"]]
    q = it.quote
    low = unify(q)
    names = _alias_spans(low)
    # the chosen group's names; a fasting and a plain glucose name are one analyte (fasting is
    # then never granted below), but urea and urea nitrogen are not: their mg/dL differ 2.1x
    mine = [n for n in names if (g := _group_of(n[2])) is not None
            and (g.key == chosen.key or (g.codes == chosen.codes and (g.fasting or chosen.fasting)))]
    if len(mine) != 1:
        return f"the quote does not name {chosen.key} exactly once"
    s, e, alias = mine[0]
    prefix = low[:s].strip(" -•*:").strip()
    if any(n[0] < s for n in names) or (prefix and not _LEAD_FILLER.match(prefix + " ")):
        return "other words name the analyte before it"
    group = _group_of(alias)
    if group.fasting and not chosen.fasting:
        group = chosen                         # never grant 'fasting' the model did not choose
    elif chosen.fasting and not group.fasting:
        return "the quote does not say fasting"
    accepted = {k for code in group.codes for spec in _specs(code, group) for k in spec.units}
    m = re.match(rf"\s*(?:\((?P<bu>[^()]{{1,15}})\)\s*)?(?:level|value|result|count)?\s*"
                 rf"(?:[:=]|\bis\b|\bwas\b|\bof\b|\bat\b|-)?\s*(?:about|around|approx\.?|approximately|~)?\s*"
                 rf"(?P<cmp>(?:<|>|≤|≥|less than|greater than|under|over|below|above)\s*)?(?P<num>{_NUM})",
                 low[e:])
    if not m:
        return "no number right after the name"
    if any(n[0] >= e and n[0] < e + m.start("num") for n in names):
        return "another lab name between the name and the number"
    if m.group("cmp"):
        return "a censored value (< or >)"
    num_end = e + m.end("num")
    number = q[e + m.start("num"):num_end]
    tail = q[num_end:]
    if re.match(r"\s*(?:,\d|/\s*\d|-\s*\d|\.\d)", tail):
        return "the number continues (a decimal comma, a ratio or a range)"
    unit = ""
    bu = m.group("bu")
    if bu and normalise_unit(bu) in accepted:
        unit = q[e + m.start("bu"):e + m.end("bu")]
    rest = tail.strip()
    if not unit:
        bracket = re.match(r"^[(\[]\s*([^()\[\]]{1,15}?)\s*[)\]]", rest)
        if bracket and normalise_unit(bracket.group(1)) in accepted:
            unit, rest = bracket.group(1), rest[bracket.end():]
        else:
            tokens = re.findall(r"\S+", rest)
            for k in range(min(4, len(tokens)), 0, -1):
                cand = re.match(r"^\S+(?:\s+\S+){%d}" % (k - 1), rest).group(0)
                cand_clean = cand.rstrip(".,;:")
                if normalise_unit(cand_clean) in accepted:
                    unit, rest = cand_clean, rest[len(cand_clean):]
                    break
            else:
                first = tokens[0].rstrip(".,;:") if tokens else ""
                if first and not _FLAG.match(first.lower()) and not _CONTEXT_WORD.match(first.lower()) \
                        and not first.startswith(("(", "[")) and not re.match(r"^[<>≤≥\d]", first):
                    unit, rest = first, rest[len(first):]          # unknown: the rules will refuse it
    left = _DATE.sub(" ", _RANGE.sub(" ", unify(rest)))
    left = re.sub(r"(?:<|>|≤|≥)\s*\d+(?:\.\d+)?", " ", left)      # a reference limit "(<5)"
    if re.search(r"\d", left):
        return "a second number for the analyte"
    it.fact = Fact("lab", chosen.key if not group.fasting else group.key, number, unit)
    return ""


def _specs(code: str, group):
    from core.patient_text import LABS
    return [s_ for s_ in LABS if s_.code == code and set(s_.aliases) & set(group.aliases)]


def _check_cotinine(it: Item) -> str:
    low = unify(it.quote)
    m = re.search(r"\bcotinine\b", low)
    if not m:
        return "the quote does not say cotinine"
    after = low[m.end():]
    if re.search(r"\b(?:undetectable|not detected|negative|nd)\b", after):
        return "a censored cotinine value"
    c = re.match(rf"\s*(?P<lvl>level)?\s*(?:value|result)?\s*(?:[:=]|\bis\b|\bwas\b|\bof\b)?\s*"
                 rf"(?P<cmp>(?:<|>|≤|≥|less than|greater than|under|over|below|above)\s*)?(?P<num>{_NUM})"
                 rf"\s*(?P<unit>ng\s*/\s*ml|[uµμ]g\s*/\s*l)?\b", after)
    if not c:
        return "no cotinine value"
    tail = _RANGE.sub(" ", after[c.end():])
    if re.search(r"\d", tail):
        return "a second number"
    number = it.quote[m.end() + c.start("num"):m.end() + c.end("num")]
    if c.group("cmp"):
        if c.group("cmp").strip() in ("<", "less than", "under", "below", "≤") and c.group("unit") \
                and float(number) <= 10:
            it.fact = Fact("cotinine", number="0", unit="ng/mL")   # below 10 ng/mL: level 0
            return ""
        return "a censored cotinine value"
    if c.group("unit"):
        it.fact = Fact("cotinine", number=number, unit="ng/mL")
        return ""
    if c.group("lvl") and number in ("0", "1", "2", "3"):
        it.fact = Fact("cotinine", number=number, unit="level")
        return ""
    return "cotinine without ng/mL or a level 0-3"


def _check_smoking(it: Item, measured_cotinine: bool) -> str:
    low = unify(it.quote)
    if not _SMOKE_WORD.search(low):
        return "the quote has no smoking word"
    status, other = it.raw["status"], it.raw["other_nicotine"]
    nicotine = _NICOTINE_WORD.search(low)
    if other in ("vaping", "nicotine_replacement", "smokeless") and nicotine \
            and re.search(r"\b(?:never|no|not|nor|without|don'?t|doesn'?t|didn'?t)\b|n't\b|\bor\s*$",
                          re.split(r"[,;.]|\bbut\b", low[:nicotine.start()])[-1]):
        other = "none"                           # "never smoked or vaped", "doesn't vape"
    if status == "unclear" and other == "none" and not _TOBACCO_WORD.search(low) and not re.search(r"smok", low):
        return "says nothing about smoking"      # "never vaped", "20 pack years" alone
    if other == "cannabis" and not _TOBACCO_WORD.search(low):
        it.problem = (f"'{it.quote}': cannabis is not tobacco, and LinAge2 reads tobacco exposure "
                      f"(cotinine); say your tobacco smoking on its own ('never smoked', 'former "
                      f"smoker', 'current smoker')", "ambiguous", "smoking", [])
        return ""
    if other in ("vaping", "nicotine_replacement", "smokeless"):
        what = {"vaping": "vaping", "nicotine_replacement": "nicotine replacement",
                "smokeless": "smokeless tobacco"}[other]
        text = (f"'{it.quote}': {what} raises cotinine, which LinAge2 reads, but is not smoking to "
                f"the knowledge base; give 'cotinine N ng/mL' from a test, or remove the {what} to "
                f"be read without it")
        if measured_cotinine:
            it.note = f"'{it.quote}': {what} — the measured cotinine is used for it"
        else:
            it.problem = (text, "vaping", "smoking", [])
    elif other == "secondhand":
        it.note = f"'{it.quote}': second-hand smoke is not smoking; it is not counted"
    if status == "unclear":
        why = re.sub(r"[`*_\[\]<>]", "", str(it.raw.get("why", "")))[:160]
        if it.problem is None:
            it.problem = (f"'{it.quote}': the model cannot tell whether you smoke now, used to, or never "
                          f"did{f' ({why})' if why else ''}; write 'current smoker', 'former smoker' or "
                          f"'never smoked'", "ambiguous", "smoking", [])
        return ""
    if other == "cannabis":
        return ""                                # tobacco words too: the rules decide
    it.fact = Fact("smoking", STATUS[status],
                   occasional=bool(it.raw["occasional"] and status == "current" and _OCCASIONAL.search(low)))
    return ""


def _negated(low: str, ts: int, te: int) -> bool:
    """Is the condition at low[ts:te] denied? A negation heading a list covers the whole
    list ("no diabetes or hypertension"); one heading the nearest clause covers it
    ("hypertension, no diabetes"); a trailing "ruled out", "?" or ": no" too."""
    before = low[:ts]
    head = _HEADER.sub("", before)
    if re.match(rf"^\W*(?:i\s+)?(?:have\s+|had\s+|has\s+)?{_NEG}\b", head):
        return True
    clause = re.split(r"[,;.]|\band\b|\bbut\b|\bwith\b", before)[-1]
    if re.match(rf"^\s*(?:i\s+)?(?:have\s+|had\s+|has\s+)?{_NEG}\b", clause):
        return True
    after = re.split(r"[,;.]", low[te:])[0]
    return bool(_TRAIL_NEG.search(after) or _EMPTY_ANSWER.match(after))


def _dated_past(low: str, window: str) -> bool:
    """A time reference that puts the event outside the item's window."""
    this_year = date.today().year
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", low)]
    recent = re.search(r"\b(?:past|last) (?:12|twelve) months\b|\bpast year\b|\blast year\b|\bthis year\b"
                       r"|\bthis month\b|\blast month\b|\b(?:past|last) (?:3|three) months\b|\bcurrently\b|\bnow\b",
                       low)
    old = re.search(r"\b(?:history of|hx of|as a (?:child|kid|teen\w*)|in my (?:teens|twenties|thirties|20s|30s|40s)"
                    r"|years ago|long ago|previous(?:ly)?|prior|in the past|had|once)\b", low)
    if window == "12 months":
        if any(y < this_year - 1 for y in years):
            return True
        if re.search(r"\b(?:\d+|two|three|four|five|several|many)\s+years?\s+ago\b", low):
            return True
        return bool(old and not recent)
    if window == "3 months":
        if any(y < this_year for y in years):
            return True
        m = re.search(r"\b(\d+)\s+months?\s+ago\b", low)
        if m and int(m.group(1)) > 3:
            return True
        return bool(old and not recent)
    return False


def _is_lab_or_vital(low: str) -> bool:
    names = _alias_spans(low)
    if names and re.match(rf"\s*(?:level|value)?\s*(?:[:=]|is|was|of)?\s*{_NUM}", low[names[0][1]:]):
        return True
    return bool(re.search(r"\b(?:bp|blood pressure)\s*[:=]?\s*\d{2,3}\s*/\s*\d{2,3}", low)
                or re.search(rf"{_NUM}\s*(?:%|mg/dl|mmol/l|mmol/mol|g/dl|g/l|u/l|ng/ml)", low))


def _check_condition(it: Item) -> str:
    v = vocabulary()
    by_key = {c.key: c for c in v.condition_info.values()}
    c = by_key[it.raw["condition"]]
    low = unify(it.quote)
    answer = it.raw["answer"]
    yes_line, no_line = render(Fact("condition", c.item, answer="yes")), render(Fact("condition", c.item, answer="no"))
    term = re.search(rf"\b(?:{c.terms})\b", low)
    if not term or _is_lab_or_vital(low):
        if answer != "no":
            it.note = (f"the model reads {c.key} from '{it.quote}', which does not say it; write "
                       f"'{yes_line}' if a doctor told you so")
        return ""
    if c.exclusions and re.search(rf"\b(?:{c.exclusions})\b", low):
        it.problem = (f"'{it.quote}': LinAge2's {c.key} item is NHANES's question \"{c.question}\" — "
                      f"this wording may be one it leaves out; write '{yes_line}' or '{no_line}'",
                      "ambiguous", "condition", [yes_line, no_line])
        return ""
    if c.window and _dated_past(low, c.window):
        it.problem = (f"'{it.quote}': LinAge2's {c.key} item covers only the past {c.window} "
                      f"(\"{c.question}\"); write '{yes_line}' or '{no_line}'",
                      "ambiguous", "condition", [yes_line, no_line])
        return ""
    borderline_words = re.search(r"pre-?diabet|borderline diabet", low)
    if answer == "borderline" and not (c.item == "DIQ010" and borderline_words):
        return "borderline without 'prediabetes' or 'borderline diabetes'"
    if answer == "yes" and c.item == "DIQ010" and borderline_words:
        return "prediabetes read as diabetes"
    computed = "no" if _negated(low, term.start(), term.end()) else "yes"
    if computed != ("no" if answer == "no" else "yes"):
        return f"the wording reads {computed}, the model says {answer}"
    it.fact = Fact("condition", c.item, answer=answer)
    return ""


_NO_OTHER = re.compile(
    r"\bno\b.*\b(?:conditions?|diagnos[ie]s|diseases?|illness(?:es)?|medical (?:history|problems?|issues?)|"
    r"health (?:problems?|issues?|conditions?)|problems?|issues?|comorbidit(?:y|ies)|pmh|history)\b"
    r"|\botherwise (?:healthy|well|fit)\b|\bnothing else\b|\bhealthy otherwise\b|^\W*none\W*$")


def _check_no_other(it: Item) -> str:
    low = unify(it.quote)
    if not _NO_OTHER.search(low):
        return "the quote does not say there are no (other) conditions"
    if re.search(r"\b(?:except|apart from|other than|besides|aside from|but)\b", low):
        return "the quote names exceptions"
    it.fact = Fact("no_other_conditions")
    return ""


_SEX_WORDS = {"male": "Male", "man": "Male", "gentleman": "Male", "m": "Male",
              "female": "Female", "woman": "Female", "lady": "Female", "f": "Female"}


def _sex_tokens(low: str) -> set:
    found = set()
    for m in re.finditer(r"\b(male|female|man|woman|gentleman|lady)\b", low):
        found.add(_SEX_WORDS[m.group(1)])
    for m in re.finditer(r"\b(?:sex|gender)\s*[:=]?\s*(male|female|m|f)\b", low):
        found.add(_SEX_WORDS[m.group(1)])
    if not re.search(r"height|weight|tall|\bcm\b|\bkg\b|\blbs?\b|bmi|\bmm\b|met(?:er|re)|\bmg\b", low):
        for m in re.finditer(r"(?<![\d.])[1-9]\d\s*(?:y/?o|yo|y\.o\.?|yrs?|years?(?:\s*old)?|-year-old)?\s*[,/]?\s*"
                             r"([mf])\b(?!\s*[/\d])", low):
            found.add(_SEX_WORDS[m.group(1)])
        for m in re.finditer(r"(?<![a-z])([mf])\s*[,/]?\s*(?:aged?\s*)?[1-9]\d\b(?![./\d])", low):
            found.add(_SEX_WORDS[m.group(1)])
    return found


def _check_sex(it: Item) -> str:
    found = _sex_tokens(unify(it.quote))
    want = SEX[it.raw["sex"]]
    if found != {want}:
        return "no explicit sex in the quote" if not found else "the quote's sex is not the model's"
    it.fact = Fact("sex", want)
    return ""


_AGE_PHRASES = (
    re.compile(r"(?<![\d.])(\d{1,3})\s*(?:-|\s)?(?:years?|yrs?)(?:\s*|-)old\b"),
    re.compile(r"(?<![\d.])(\d{1,3})\s*(?:-|\s)?(?:y/?o|yo|y\.o\.?|yr-old|year-old)(?![a-z])"),
    re.compile(r"\b(?:age|aged)\s*(?:is|:|=)?\s*(\d{1,3})\b"),
    re.compile(r"(?<![\d.])(\d{2})\s*,?\s*(?:m|f|male|female)\b"),
    re.compile(r"\b(?:m|f|male|female)\s*,?\s*(?:aged?\s*)?(\d{2})\b(?![./\d])"),
)
_NOT_AN_AGE = re.compile(r"\b(?:biological|bio|metabolic|epigenetic|grim\w*|lin\w*|pheno\w*|clock|"
                         r"heart age|lung age|brain age|since|until|when|at age|ago|onset|diagnos\w*|"
                         r"retire\w*|menopaus\w*|quit|started|stopped|child|son|daughter|husband|wife|"
                         r"partner|mother|father|died|death|married|born)\b")


def _check_age(it: Item) -> str:
    low = unify(it.quote)
    if _NOT_AN_AGE.search(low):
        return "the quote is about another age"
    numbers = {m.group(1) for rx in _AGE_PHRASES for m in rx.finditer(low)}
    if len(numbers) != 1:
        return "no single age phrase in the quote"
    (number,) = numbers
    if len(re.findall(r"\d+", low)) != 1:
        return "other numbers in the quote"
    it.fact = Fact("age", number=str(int(number)))
    return ""


def _check_weight(it: Item) -> str:
    low = unify(it.quote)
    if not re.search(r"\b(?:weight|weigh(?:s|ed)?|wt)\b", low):
        return "the quote does not say weight"
    if re.search(r"\b(?:lost|lose|losing|gain\w*|down|up|target|goal|ideal|want|less|more|by)\b", low):
        return "a change or a goal, not a weight"
    m = re.findall(r"(\d+(?:\.\d+)?)\s*(kgs?|kilos?|kilograms?|lbs?|pounds?)\b", low)
    if len(m) != 1 or len(re.findall(r"\d+(?:\.\d+)?", low)) != 1:
        return "not one weight with a unit"
    number, unit = m[0]
    it.fact = Fact("weight", number=number, unit="lb" if unit.startswith(("lb", "pound")) else "kg")
    return ""


def _check_height(it: Item) -> str:
    low = unify(it.quote)
    if not re.search(r"\b(?:height|tall|ht)\b", low):
        return "the quote does not say height"
    ftin = re.findall(r"\b(\d)\s*(?:'|ft|feet|foot)\s*(\d{1,2})\s*(?:\"|''|in\b|inch(?:es)?)?", low)
    if ftin:
        if len(ftin) != 1 or len(re.findall(r"\d+", low)) != 2:
            return "not one height"
        it.fact = Fact("height", number=f"{ftin[0][0]}'{ftin[0][1]}", unit="ft-in")
        return ""
    m = re.findall(r"(\d+(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?|in|inch(?:es)?)\b", low)
    if len(m) != 1 or len(re.findall(r"\d+(?:\.\d+)?", low)) != 1:
        return "not one height with a unit"
    number, unit = m[0]
    unit = "cm" if unit.startswith(("cm", "centi")) else "in" if unit.startswith("in") else "m"
    it.fact = Fact("height", number=number, unit=unit)
    return ""


def _check_health(it: Item) -> str:
    low = unify(it.quote)
    rating = it.raw["rating"]
    found = set(re.findall(r"\bvery good\b|\bexcellent\b|\bfair\b|\bpoor\b", low))
    if re.search(r"(?<!very )\bgood\b", low):
        found.add("good")
    if found != {rating}:
        return "the rating word is not alone in the quote" if found else "no rating word in the quote"
    if not re.search(r"\b(?:health|healthy|feel|self[- ]rated|overall|general)\b", low):
        return "the quote is not about health"
    if re.search(r"\b(?:not|n't|isn'?t|never|hardly)\b", low):
        return "a negated rating"
    it.fact = Fact("self_rated_health", rating)
    return ""


def _check_trend(it: Item) -> str:
    low = unify(it.quote)
    trend = it.raw["trend"]
    found = set()
    for word, key in (("better", "better"), ("improved", "better"), ("worse", "worse"), ("same", "about the same"),
                      ("unchanged", "about the same"), ("similar", "about the same")):
        if re.search(rf"\b{word}\b", low):
            found.add(key)
    if found != {trend}:
        return "the trend word is not alone in the quote" if found else "no trend word"
    if not re.search(r"year ago|last year|past year|12 months|a year back|than a year|this time last year", low):
        return "not a comparison with a year ago"
    if re.search(r"\b(?:not|n't|never)\b", low):
        return "a negated trend"
    it.fact = Fact("health_vs_year_ago", trend)
    return ""


_COUNT_WORDS = {"once": 1, "twice": 2, "none": 0, "no": 0, "zero": 0, "never": 0, "one": 1, "two": 2,
                "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _check_visits(it: Item) -> str:
    low = unify(it.quote)
    if not re.search(r"visit|appointment|doctor|\bgp\b|physician|clinic|check-?up|\bseen\b|\bsaw\b", low):
        return "the quote is not about healthcare visits"
    if re.search(r"\d+\s*(?:-|to)\s*\d+", low):
        return "a range of visits"
    nums = re.findall(r"\b\d+\b", low)
    words = [w for w in re.findall(r"\b[a-z]+\b", low) if w in _COUNT_WORDS]
    if len(nums) + len(words) != 1:
        nums = [n for n in nums if n != "12" or not re.search(r"\b12 months\b", low)]
        if len(nums) + len(words) != 1:
            return "not one count"
    n = int(nums[0]) if nums else _COUNT_WORDS[words[0]]
    if re.search(r"\bper month\b|\ba month\b|/\s*month|\bmonthly\b|\beach month\b|\bevery month\b", low):
        period, factor = "month", 12
    elif re.search(r"\bper week\b|\ba week\b|/\s*week|\bweekly\b|\beach week\b|\bevery week\b", low):
        period, factor = "week", 52
    elif re.search(r"\b(?:\d+|two|three|five|several)\s+(?:years|months|weeks)\b|\bdecade\b|\bever\b|"
                   r"\blifetime\b", low.replace("12 months", "")):
        return "a period other than a year, a month or a week"
    elif re.search(r"\bper year\b|\ba year\b|/\s*year|\byearly\b|\bannual(?:ly)?\b|\blast year\b|\bpast year\b|"
                   r"\b12 months\b|\bthis year\b", low):
        period, factor = "year", 1
    else:
        period, factor = "unstated", 1
    if it.raw["period"] != period:
        return f"the period reads {period}, the model says {it.raw['period']}"
    it.fact = Fact("healthcare_visits", number=str(n * factor))
    return ""


def _check_grimage(it: Item) -> str:
    low = unify(it.quote)
    if not re.search(r"grim|ageaccelgrim", low):
        return "the quote does not say GrimAge"
    if it.raw["wording"] == "clock_age":
        return "a clock age, not an acceleration"
    m = re.findall(r"([+-]?)\s*(\d+(?:\.\d+)?)", low)
    if len(m) != 1:
        return "not one number"
    sign, number = m[0]
    direction = it.raw["direction"]
    older, younger = bool(re.search(r"\bolder\b", low)), bool(re.search(r"\byounger\b", low))
    if direction == "signed" and sign:
        value = f"{sign}{number}"
    elif direction == "older" and older and not younger and not sign:
        value = f"+{number}"
    elif direction == "younger" and younger and not older and not sign:
        value = f"-{number}"
    elif direction == "unstated" and not sign and not older and not younger \
            and re.search(r"accel|ageaccelgrim", low):
        value = number
    else:
        return "the sign is not written in the quote"
    it.fact = Fact("grimage", number=value)
    return ""


def _check_unclear(it: Item) -> str:
    low = unify(it.quote)
    topic = it.raw["topic"]
    why = re.sub(r"[`*_\[\]<>]", "", str(it.raw.get("why", "")))[:160]
    v = vocabulary()
    if topic == "smoking" and _SMOKE_WORD.search(low):
        it.problem = (f"'{it.quote}': the model cannot tell what this says about your smoking"
                      f"{f' ({why})' if why else ''}; write 'current smoker', 'former smoker' or "
                      f"'never smoked'", "ambiguous", "smoking", [])
    elif topic == "condition" and (_MEDICAL_WORDS.search(low) or any(
            re.search(rf"\b(?:{c.terms})\b", low) for c in v.condition_info.values())):
        it.problem = (f"'{it.quote}': the model cannot tell whether this is one of the conditions "
                      f"LinAge2 counts{f' ({why})' if why else ''}; write it as 'diagnoses: …' or "
                      f"'no …' with the condition's name", "ambiguous", "condition", [])
    else:
        it.note = f"the model could not read '{it.quote}'{f' ({why})' if why else ''}"
    return ""


# ═══════════════════════════ reading ═══════════════════════════════════════════

def read_patient(text: str, extractor: Optional[Extractor] = None) -> PatientRead:
    """The rules' reading of `text`, and with an extractor the model's rewrites too."""
    text = text or ""
    rules = read_patient_text(text)
    if extractor is None:
        return PatientRead(text, text, rules)
    model = getattr(extractor, "model", None)
    try:
        ex = extractor(text)
    except ExtractError as exc:
        return PatientRead(text, text, rules, model=model, model_error=exc)
    except Exception as exc:                    # noqa: BLE001 — Read must still answer
        return PatientRead(text, text, rules, model=model,
                           model_error=ExtractError("upstream", f"{type(exc).__name__}: {exc}"[:300]))
    out = _with_model(text, rules, ex)
    out.model, out.usage, out.latency_s, out.cached = ex.model, dict(ex.usage), ex.latency_s, ex.cached
    return out


_CHECKS = {"lab": _check_lab, "cotinine": _check_cotinine, "condition": _check_condition,
           "no_other_conditions": _check_no_other, "sex": _check_sex, "age": _check_age,
           "weight": _check_weight, "height": _check_height, "self_rated_health": _check_health,
           "health_vs_year_ago": _check_trend, "healthcare_visits": _check_visits,
           "grimage": _check_grimage, "unclear": _check_unclear}


def _ground(raw: dict, lines: list[str], rules: ParsedPatient) -> tuple[Optional[Item], str]:
    if not isinstance(raw, dict) or not isinstance(raw.get("quote"), str):
        return None, "no quote"
    where, why = _locate(raw["quote"], lines)
    if where is None:
        return None, why
    line, start, end = where
    sts = tuple(st.index for st in rules.statements
                if st.line == line and st.start < end and start < st.end)
    if not sts:
        return None, "the quote is in no statement"
    return Item(raw, raw.get("kind", ""), line, start, end, lines[line][start:end], sts), ""


def _merge(facts_list: list[dict]) -> dict:
    out: dict = {}
    for facts in facts_list:
        for k, val in facts.items():
            if isinstance(val, dict):
                out.setdefault(k, {}).update(val)
            else:
                out[k] = val
    return out


_LINE_FACTS: dict = {}


def _line_facts(line: str) -> dict:
    """What the rules read from one canonical line on its own."""
    if line not in _LINE_FACTS:
        p = read_patient_text(line)
        _LINE_FACTS[line] = _merge([st.facts for st in p.statements])
    return _LINE_FACTS[line]


_WHO_PHRASE = {("NeverSmoker", 0): "never smoked", ("FormerSmoker", 0): "former smoker",
               ("CurrentSmoker", 3): "current smoker", ("CurrentSmoker", 1): "occasional smoker",
               ("CurrentSmoker", 2): "moderate smoker"}


def _rules_wording(facts: dict) -> Optional[str]:
    """The rules' reading of a statement as canonical lines, if it is only who-facts."""
    lines = []
    for k, val in facts.items():
        if k == "age":
            lines.append(f"{val:g} year old")
        elif k == "sex":
            lines.append(val.lower())
        elif k == "smoking":
            lines.append(_WHO_PHRASE.get(tuple(val), ""))
        elif k == "conditions":
            for item, a in val.items():
                lines.append(render(Fact("condition", item, answer={1: "yes", 2: "no", 3: "borderline"}[a])))
        else:
            return None
    return "; ".join(x for x in lines if x) or None


def _describe(key: str, val) -> str:
    if key == "smoking":
        return _WHO_PHRASE.get(tuple(val), f"{val[0]} (level {val[1]})")
    if key == "age":
        return f"age {val:g}"
    return str(val).lower()


def _with_model(text: str, rules: ParsedPatient, ex: Extraction) -> PatientRead:
    lines = text.splitlines()                  # as the reader numbers them
    out = PatientRead(text, text, rules, reader="rules+model")

    grounded: list[Item] = []
    for raw in ex.items:
        it, why = _ground(raw, lines, rules)
        if it is None:
            out.discarded.append((str(raw.get("quote", ""))[:120], str(raw.get("kind", "")), why))
        else:
            grounded.append(it)
    others = [it for it in grounded if it.kind == "someone_else"]
    items: list[Item] = []
    for it in grounded:
        if it.kind == "someone_else":
            continue
        why = ""
        if any(o.line == it.line and o.start < it.end and it.start < o.end for o in others):
            why = "in what the model says is about someone else"
        elif any(rules.statements[i].outcome == "set_aside" for i in it.statements):
            why = "in a statement the rules set aside as about someone else"
        elif _SOMEONE_ELSE.search(unify(it.quote)):
            why = "the quote is about someone else"
        elif it.kind == "smoking":
            why = _check_smoking(it, rules.cotinine_measured)
        elif it.kind in _CHECKS:
            why = _CHECKS[it.kind](it)
        else:
            why = f"no such kind {it.kind!r}"
        if why:
            out.discarded.append((it.quote, it.kind, why))
        else:
            items.append(it)

    model_problems: list[tuple] = []           # (text, kind, topic, statements, wordings, span)

    # a rules reading inside what the model calls someone else's
    for o in others:
        for i in o.statements:
            st = rules.statements[i]
            if st.outcome in ("set_aside", "refused") or not st.facts:
                continue
            if "smoking" in st.facts and not is_canonical(st.text):
                model_problems.append((
                    f"'{st.text}': the model reads this as about someone else, but your smoking was read "
                    f"from it; put your own smoking on its own line ('never smoked', 'former smoker', "
                    f"'current smoker')", "disagreement", "smoking", (i,), [], None))
            else:
                out.notes.append(f"the model reads '{o.quote}' as about someone else; it was read as yours")

    # components: statements joined by the items that quote them
    parent = {st.index: st.index for st in rules.statements}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for it in items:
        for i in it.statements[1:]:
            parent[find(i)] = find(it.statements[0])
    groups: dict[int, list[Item]] = {}
    for it in items:
        groups.setdefault(find(it.statements[0]), []).append(it)

    subs: list[Substitution] = []
    suggestions: list[Suggestion] = []
    for root, its in groups.items():
        sts = sorted(i for i in parent if find(i) == root)
        first, last = rules.statements[sts[0]], rules.statements[sts[-1]]
        line = first.line
        span = (line, first.start, last.end)
        original = lines[line][first.start:last.end]
        facts = [it.fact for it in its if it.fact is not None]
        wording = "; ".join(dict.fromkeys(render(f) for f in facts))
        outcomes = {rules.statements[i].outcome for i in sts}
        canonical = is_canonical(original)

        for it in its:
            if it.note:
                out.notes.append(it.note)
            if it.problem is not None:
                ptext, pkind, ptopic, pwords = it.problem
                if canonical and ptopic == "smoking":
                    out.notes.append(f"the model was unsure about '{it.quote}'; it is in canonical form, "
                                     f"so it stands")
                elif "refused" in outcomes:
                    out.notes.append(ptext)
                else:
                    model_problems.append((ptext, pkind, ptopic, tuple(sts), pwords,
                                           (line, it.start, it.end, it.quote)))
        if not wording:
            continue

        if "refused" in outcomes:
            suggestions.append(Suggestion(*span, original, [wording],
                                          "the rules could not use this as typed; the model reads it as",
                                          blocking=False))
            continue
        rules_facts = _merge([rules.statements[i].facts for i in sts])
        model_facts = _merge([_line_facts(render(f)) for f in facts])
        model_smoking = any(f.kind == "smoking" for f in facts)
        read = outcomes <= {"read"}
        blocking = []
        for key in ("smoking", "age", "sex"):
            if key in rules_facts and key in model_facts and rules_facts[key] != model_facts[key]:
                r, m = rules_facts[key], model_facts[key]
                if key == "smoking" and r[0] == m[0] == "CurrentSmoker" and r[1] == 2 and m[1] == 3:
                    continue                    # "moderate": a current smoker the model has no level for
                blocking.append((key, r, m))
        if read and model_smoking and "smoking" not in rules_facts:
            blocking.append(("smoking", None, model_facts.get("smoking")))
        if blocking and not canonical:
            said = "; ".join(f"the rules read {_describe(k, r) if r is not None else 'no smoking status'}, "
                             f"the model reads {_describe(k, m)}" for k, r, m in blocking)
            rules_way = _rules_wording(rules_facts)
            choices = [wording] + ([rules_way] if rules_way and rules_way != wording else [])
            model_problems.append((
                f"'{original}': {said}; choose a wording below, or rewrite it",
                "disagreement", blocking[0][0], tuple(sts), choices, span + (original,)))
            suggestions.append(Suggestion(*span, original, choices, "the rules and the model differ",
                                          blocking=True))
            continue
        if read:
            notes = []
            for key in ("labs", "conditions", "questionnaire"):
                for k, val in model_facts.get(key, {}).items():
                    r = rules_facts.get(key, {}).get(k)
                    if r is None or (r != val and not (isinstance(r, float) and isinstance(val, float)
                                                       and abs(r - val) <= 1e-6 * max(1.0, abs(r)))):
                        notes.append(k)
            for key in ("cotinine", "weight_kg", "height_cm", "grimage", "no_conditions"):
                if key in model_facts and rules_facts.get(key) != model_facts[key]:
                    notes.append(key)
            if notes and not canonical:
                out.notes.append(f"the model reads '{original}' differently from the rules ({', '.join(notes)}); "
                                 f"the rules' reading is used — the wording below gives the model's")
                suggestions.append(Suggestion(*span, original, [wording], "the model reads it as",
                                              blocking=False))
            continue

        # not (fully) understood by the rules: rewrite, unless the model alone says how the
        # person smokes (the same status the rules read from it may be rewritten)
        if model_smoking and "smoking" not in rules_facts:
            model_problems.append((
                f"'{original}' was not understood by the rules; the model reads it as '{wording}' — use "
                f"that wording, or rewrite it", "not_understood", "smoking", tuple(sts), [wording],
                span + (original,)))
            suggestions.append(Suggestion(*span, original, [wording],
                                          "a smoking status is never taken from the model without you",
                                          blocking=True))
            continue
        lost = [k for k, val in rules_facts.items() if k != "ignored" and not _covers(model_facts, k, val)]
        if lost:
            suggestions.append(Suggestion(*span, original, [wording],
                                          "the model reads it as (it leaves out part of what the rules read)",
                                          blocking=False))
            continue
        subs.append(Substitution(*span, original, list(dict.fromkeys(render(f) for f in facts)), tuple(sts)))

    # rewrite, read again, and keep only rewrites that add exactly what they say
    while True:
        read_as, expected = _apply(lines, rules, subs)
        final = read_patient_text(read_as)
        bad = _check(rules, final, expected, subs)
        if not bad:
            break
        for s in [s for s in subs if s.line in bad or -1 in bad]:
            subs.remove(s)
            suggestions.append(Suggestion(s.line, s.start, s.end, s.original, ["; ".join(s.lines)],
                                          "the model reads it as (not applied: it would change other lines)",
                                          blocking=False))

    index = {orig: n for n, (orig, _, _) in enumerate(expected) if orig is not None}
    for n, (orig, _, sub) in enumerate(expected):
        if sub is not None:
            out.model_quotes[n] = sub.original
    for ptext, pkind, ptopic, sts, _, _ in model_problems:
        final.problems.append(Problem(ptext, pkind, ptopic, tuple(index[i] for i in sts if i in index)))
    _outcomes(final)
    out.read_as, out.parsed, out.substitutions = read_as, final, subs
    out.suggestions = sorted(suggestions, key=lambda s: (not s.blocking, s.line, s.start))
    out.notes = list(dict.fromkeys(out.notes))
    return out


def _covers(model_facts: dict, key: str, val) -> bool:
    if isinstance(val, dict):
        got = model_facts.get(key, {})
        return all(k in got and (got[k] == x or (isinstance(x, float) and isinstance(got[k], float)
                                                  and abs(got[k] - x) <= 1e-6 * max(1.0, abs(x))))
                   for k, x in val.items())
    return model_facts.get(key) == val


def _apply(lines: list[str], rules: ParsedPatient, subs: list[Substitution]):
    """The read-as text, and the statements it should split into: (original index or
    None, text, the substitution it came from or None)."""
    new_lines = list(lines)
    by_line: dict[int, list[Substitution]] = {}
    for s in subs:
        by_line.setdefault(s.line, []).append(s)
    for n, ss in by_line.items():
        line = new_lines[n]
        for s in sorted(ss, key=lambda s: -s.start):
            line = line[:s.start] + "; ".join(s.lines) + line[s.end:]
        new_lines[n] = line
    covered = {i: s for s in subs for i in s.statements}
    expected: list = []
    done = set()
    for st in rules.statements:
        s = covered.get(st.index)
        if s is None:
            expected.append((st.index, st.text, None))
        elif id(s) not in done:
            done.add(id(s))
            expected.extend((None, x, s) for x in s.lines)
    return "\n".join(new_lines), expected


def _check(rules: ParsedPatient, final: ParsedPatient, expected: list, subs: list) -> set:
    """Lines whose rewrite did more than add its own lines (-1: withdraw them all)."""
    if not subs:
        return set()
    got = [st.text for st in final.statements]
    if got != [t for _, t, _ in expected]:
        return {-1}
    bad = set()
    for st, (orig, text_, sub) in zip(final.statements, expected):
        if sub is None:
            if st.facts != rules.statements[orig].facts:
                bad.add(rules.statements[orig].line)
        elif st.facts != _line_facts(text_):
            bad.add(sub.line)
    if bad:
        return bad
    touched = {i for s in subs for i in s.statements}
    before = {str(p) for p in rules.all_problems() if p.kind != "missing"
              and set(getattr(p, "statements", ())) and not set(p.statements) & touched}
    after = {str(p) for p in final.all_problems()}
    if not before <= after:
        return {-1}
    return set()
