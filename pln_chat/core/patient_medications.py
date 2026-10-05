"""Medications the My Patient reader understands: only a drug the knowledge base can act on.

The knowledge base has ONE medication fact that matters to a patient: a supplement plan flags a
supplement that interacts with a drug the person takes (`(Interaction Berberine Metformin …)`,
supplement_evidence.metta), against `(CurrentMedication <Patient> <Drug>)`. So the reader reads a
drug ONLY from the table below — names that are identities (the same substance: generic, salt,
formulation, brand), never similarity — and leaves every other drug exactly where it was, in "not
understood".

It is written to be strict, not clever. The first version read a medication from "<anything> on
metformin" with a deny-list of heads, and an adversarial review (docs/kb_quick_wins/REVIEW_ROUND1.md)
found 40 ways it read a drug the person does not take: a plan ("I want to be on metformin"), a
question, a past, a negation spelled "isn't" or "wasnt", someone else ("my grandma takes it"), a
qualifier cut off by the statement splitter ("I take metformin, but not anymore"), a heading above
("Past medications:") — the long tail the smoking reader had already taught. So now:

* a medication is read only from a statement that is, WHOLE, a closed grammar over tokens
      [I | we | I am | I have been] [currently | now | still …] (take | takes | taking | on | using)
          [the | my] <drug> [dose · timing · since YEAR · for N years · as prescribed]
      medications: <drug> …          <drug> <dose> …          <condition or lab> on <drug>
  A head in front of "on <drug>" must be fully understood by the rest of the reader as a condition or a
  lab; any other head is not a head. No free-text subject, no imperative ("take metformin"), no question;
* it is read only if no word of the WHOLE LINE means not-now, not-me or not-sure: negations in every
  spelling, a past, a plan, a wish, advice, a hedge, a question, a contrast, someone else. The statement
  splitter cuts "I do not, in fact, take metformin" and "I take metformin, but not anymore" into
  statements that each look fine alone. Nor is it read if the line above is a heading that is not a
  current-medication heading ("Allergies:", "Past medications:"), or the next line starts with a stop word;
* a statement that also carries age, sex or smoking is not read: the reader rewrites it (drops "until
  2019", turns "isn't" into "isn t") before a medication rule could see it;
* "not taking" is read only from a closed set of whole-statement forms (no / not on / stopped /
  discontinued / never took / off / no longer / used to take), and records nothing: a drug said not to be
  taken is not a medication. Saying both is a contradiction to fix;
* nothing is guessed about other drugs: the rest of a statement ("… and lisinopril", "… and has
  diabetes") goes on to be read as it would have been alone;
* no regular expression here can backtrack: the grammar is matched on tokens, in time linear in the
  statement, and a statement over MAX_STATEMENT_CHARS is not read at all.

The cost of a missed reading is a missing flag, which is what every patient had before; the cost of a
wrong one is a flag for a drug the person does not take.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

#: A medication statement is short; a longer one is not read (no cost, and no work for a hostile one).
MAX_STATEMENT_CHARS = 240
MAX_LINE_CHARS = 600

# ── the drugs ────────────────────────────────────────────────────────────────

#: first word -> (KB symbol, the words that may follow it as part of the name). A symbol is listed only
#: if supplement_evidence.metta holds an (Interaction …) fact for it and the KB types it a
#: Pharmaceutical (tests/test_patient_medications.py checks both). Combination products
#: (sitagliptin/metformin, glyburide/metformin …) are different products and are deliberately absent.
_DRUGS: dict[str, tuple[str, frozenset[str]]] = {
    "metformin": ("Metformin", frozenset({"hcl", "hydrochloride", "er", "xr", "sr", "ir"})),
    "glucophage": ("Metformin", frozenset({"xr"})),
    "glumetza": ("Metformin", frozenset()),
    "fortamet": ("Metformin", frozenset()),
    "riomet": ("Metformin", frozenset()),
}
#: two-word formulation phrases after "metformin"
_FORMULATIONS = frozenset({("extended", "release"), ("immediate", "release"), ("slow", "release")})

#: every spelled form -> symbol (for the tests and for anyone who needs the table)
MEDICATIONS: dict[str, str] = {}
for _first, (_symbol, _suffixes) in _DRUGS.items():
    MEDICATIONS[_first] = _symbol
    for _s in sorted(_suffixes):
        MEDICATIONS[f"{_first} {_s}"] = _symbol
for _a, _b in sorted(_FORMULATIONS):
    MEDICATIONS[f"metformin {_a} {_b}"] = "Metformin"
del _first, _symbol, _suffixes, _s, _a, _b

# ── words that mean "not now", "not me" or "not sure" ──────────────────────────

#: Any of these anywhere on the LINE means a medication on it is not read as current. Contractions are
#: normalised first (isn't, isnt, isn t -> isnt), so each spelling is one entry.
_BLOCK = frozenset("""
no not never none nor neither nothing nobody without cannot nope aint isnt arent wasnt werent hasnt
havent hadnt dont doesnt didnt cant couldnt wont wouldnt shouldnt mustnt neednt
stopped stop stops stopping quit quits quitting discontinued discontinue discontinuing ceased cease
ended end finished completed tapered tapering weaned held hold paused pause dc off ago until till
was were had used previously formerly before after
will would shall should could can might may must need needs want wants wanted hope hopes hoping wish
wishes plan plans planning planned going intend intends intended decided decide consider considers
considering think thinks thinking advised advise advice recommend recommends recommended suggest
suggests suggested told asked prescribed prescribe start starts started starting begin begins began
beginning try tries tried trying switch switched
maybe perhaps possibly probably sometimes occasionally rarely seldom unless when whether unsure
uncertain unable refuse refuses refused declined hardly barely if or
but however although though except then instead rather anymore again restart restarted resume resumed
he she they his her their him them hes shes theyre
what which how why who whom whose
allergic allergy allergies intolerant intolerance reaction reactions adverse side avoid avoids avoided
contraindicated contraindication
""".split())
#: …and these name someone other than the person. The reader's own set-aside rule knows a closed list of
#: relatives; "my doctor", "John" and "he" are caught by the grammar, which allows no subject but I / we.
_OTHERS = frozenset("""
wife husband partner spouse mother mom mum father dad son daughter brother sister aunt uncle cousin
grandmother grandfather grandma grandpa granny nan nana stepmother stepfather stepmom stepdad nephew
niece friend neighbour neighbor boss colleague doctor nurse patient relatives family someone somebody
anyone everyone people others
""".split())

_APOSTROPHES = re.compile("[´`’‘′ʼ＇]")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[%/+&,:;=()?!]|[-–—]")
_NUM_UNIT = re.compile(r"(\d)(mg|g|mcg|µg|μg|ug)\b")
_YEAR = re.compile(r"(?<![0-9.])(?:19|20)\d\d(?![0-9])")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_EXPAND = (
    (re.compile(r"\bi'm\b"), "i am"), (re.compile(r"\bwe're\b"), "we are"),
    (re.compile(r"\bi've\b"), "i have"), (re.compile(r"\bwe've\b"), "we have"),
    (re.compile(r"\b(?:i'll|we'll|i'd|we'd)\b"), "will"),     # a plan or a past: both are blocked
)
_CONTRACTION = re.compile(r"\b(\w+)n['\s]t\b")


def normalise(text: str) -> str:
    """Lower case, NFKC, one apostrophe, I'm / we're expanded, other contractions glued (isn't, isn t
    -> isnt), digits and units apart ('500mg' -> '500 mg'), '1,000' -> '1000'."""
    t = _APOSTROPHES.sub("'", text or "")          # before NFKC: an acute accent becomes a space and a mark
    t = unicodedata.normalize("NFKC", t).lower()
    for rx, repl in _EXPAND:
        t = rx.sub(repl, t)
    t = _CONTRACTION.sub(r"\1nt", t)
    t = t.replace("'", "")
    t = _NUM_UNIT.sub(r"\1 \2", t)
    return _THOUSANDS.sub("", t)


def _tokens(norm: str) -> tuple[list[tuple[str, int]], str]:
    """([(token, offset in the returned text)], the text): hyphens between letters read as spaces."""
    norm = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", norm)
    return [(m.group(0), m.start()) for m in _TOKEN.finditer(norm)], norm


def line_blocked(line: str) -> bool:
    """A word on this line means a medication on it is not read as current: not-now, not-me, not-sure.
    Judged on the whole LINE, because the statement splitter cuts a qualifier loose."""
    if len(line) > MAX_LINE_CHARS:
        return True
    norm = normalise(line)
    if "?" in norm:
        return True
    if any(w in _BLOCK or w in _OTHERS for w in re.findall(r"[a-z]+", norm)):
        return True
    # a year says WHEN ("In 2019, on metformin", "2015-2020"): only "since 2015" keeps it current
    return any(not re.search(r"\bsince\s*$", norm[:m.start()]) for m in _YEAR.finditer(norm))


#: words of a heading that introduces a CURRENT medication list
_CURRENT_HEADING = frozenset({
    "medications", "medication", "meds", "med", "medicines", "medicine", "rx", "drugs", "list",
    "current", "regular", "daily", "my", "usual",
})


def heading_kind(line: str) -> Optional[str]:
    """'current' | 'other' when `line` is a heading (it ends in ':' or '-'): 'current' only for the plain
    medication headings; 'Allergies:', 'Past medications:', 'Plan:' … are 'other'. None for any other line."""
    norm = normalise(line).strip()
    words = re.findall(r"[a-z]+", norm)
    if not words:
        return None
    if norm.endswith((":", "-", "–", "—")):
        return "current" if all(w in _CURRENT_HEADING for w in words) else "other"
    # a short line with no digits that names a past, a negation, an allergy or a person is a heading too
    # ("Discontinued medications", "Allergies", "Wife")
    if len(words) <= 4 and not re.search(r"\d", norm) and any(w in _BLOCK or w in _OTHERS for w in words):
        return "other"
    return "current" if len(words) <= 3 and all(w in _CURRENT_HEADING for w in words) else None


def line_about_other(line: str) -> bool:
    """The line names someone other than the person ("My mother has diabetes", "Wife")."""
    return any(w in _OTHERS for w in re.findall(r"[a-z]+", normalise(line)))


def mentioned_symbols(text: str) -> tuple[str, ...]:
    """The KB symbols of the drugs `text` names anywhere (a cheap pre-check, then the tokens)."""
    low = (text or "").lower()
    if not any(d in low for d in _DRUGS):
        return ()
    toks, _ = _tokens(normalise(text))
    t = [w for w, _ in toks]
    return tuple(dict.fromkeys(_DRUGS[w][0] for w in t if w in _DRUGS))


def starts_current_header(statement: str) -> bool:
    """'medications: …' — whatever follows the colon: a list of drugs that starts a medication list."""
    toks, _ = _tokens(normalise(statement))
    t = [w for w, _ in toks]
    j = 0
    while j < len(t) and t[j] in _CURRENT_HEADING:
        j += 1
    return bool(j) and j < len(t) and t[j] in {":", "=", "-", "–", "—"}


_STOP_START = frozenset("""
stopped stop discontinued ceased ended finished tapered weaned held paused off no not never none until
dc status quit used previously formerly was were had
""".split())


def starts_with_stop(line: str) -> bool:
    """The line begins with a word that takes a medication on the line above back."""
    words = re.findall(r"[a-z]+", normalise(line))
    return any(w in _STOP_START for w in words[:3])


# ── the token grammar ────────────────────────────────────────────────────────

_UNITS = frozenset({"mg", "g", "mcg", "ug", "μg", "µg"})
_PER = frozenset({"a", "per"})
_PERIODS = frozenset({"day", "daily", "week", "weekly"})
_YEARS = frozenset({"year", "years", "yr", "yrs", "month", "months", "week", "weeks"})
_ADVERBS = frozenset({"currently", "now", "still", "also", "regularly", "presently", "usually", "always"})
_SUBJECT_VERBS = frozenset({"take", "taking", "use", "using", "on"})          # after I / we
_NO_SUBJECT_VERBS = frozenset({"takes", "taking", "using", "uses", "on"})     # no subject: never the imperative
_AUX_VERBS = frozenset({"taking", "using", "on"})                               # after "have been" / "am"
_TAIL_VERBS = frozenset({"on", "taking", "takes", "take", "using", "uses"})
_FREQ_ONE = frozenset({"daily", "nightly", "bid", "tid", "qd", "qhs", "weekly"})
_LIST_JOINERS = frozenset({"and", ",", "&", "+"})
_SUBJECT_PREFIXES = (("i", "have", "been"), ("we", "have", "been"), ("i", "am"), ("we", "are"),
                     ("have", "been"), ("i",), ("we",), ("am",), ("are",), ("been",))
_HEAD_STRIP = frozenset({"and", "&", "+", ",", "who", "which", "that"})


def _num(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except ValueError:
        return None


def _per_period(t: list[str], j: int) -> Optional[int]:
    if j < len(t) and t[j] in _PER:
        j += 1
    return j + 1 if j < len(t) and t[j] in _PERIODS else None


def _extra(t: list[str], i: int) -> Optional[int]:
    """The dose / timing phrase that starts at t[i], as the index after it, or None."""
    n, w = len(t), t[i]
    v = _num(w)
    if v is not None:
        if v <= 0:
            return None                                        # "0 mg" is not a medication
        if i + 1 < n and t[i + 1] in _UNITS:
            j = i + 2
            if j + 1 < n and t[j] == "/" and t[j + 1] in {"day", "d"}:
                j += 2
            return j
        if i + 1 < n and t[i + 1] in {"times", "x"}:           # "2 times a day"
            return _per_period(t, i + 2)
        return None
    if w in {"once", "twice", "thrice"}:
        return _per_period(t, i + 1)                           # "twice daily", "once a day"
    if w in {"two", "three", "four"} and i + 1 < n and t[i + 1] in {"times", "x"}:
        return _per_period(t, i + 2)
    if w in _FREQ_ONE:
        return i + 1
    if w == "every" and i + 1 < n and t[i + 1] in {"day", "morning", "evening", "night"}:
        return i + 2
    if w in {"in", "at"}:
        j = i + 1
        if j < n and t[j] == "the":
            j += 1
        return j + 1 if j < n and t[j] in {"morning", "evening", "night", "bedtime"} else None
    if w == "with" and i + 1 < n and t[i + 1] in {"meals", "food", "breakfast", "lunch", "dinner"}:
        return i + 2
    if w == "as" and i + 1 < n and t[i + 1] == "prescribed":
        return i + 2
    if w == "since" and i + 1 < n and re.fullmatch(r"(?:19|20)\d\d", t[i + 1]):
        return i + 2
    if w == "for":
        if i + 1 < n and t[i + 1] == "years":
            return i + 2
        if i + 2 < n and _num(t[i + 1]) is not None and t[i + 2] in _YEARS:
            return i + 3
    return None


def _extras(t: list[str], i: int) -> tuple[int, int]:
    """Consume dose / timing phrases greedily: (index after, how many)."""
    count = 0
    while i < len(t):
        j = _extra(t, i)
        if j is None:
            break
        i, count = j, count + 1
    return i, count


def _drug(t: list[str], i: int) -> Optional[tuple[str, int]]:
    """A drug name at t[i]: (symbol, index after it)."""
    if i >= len(t) or t[i] not in _DRUGS:
        return None
    symbol, suffixes = _DRUGS[t[i]]
    j = i + 1
    if t[i] == "metformin" and j + 1 < len(t) and (t[j], t[j + 1]) in _FORMULATIONS:
        j += 2
    while j < len(t) and t[j] in suffixes:
        j += 1
    return symbol, j


@dataclass(frozen=True)
class MedContext:
    """What a statement's surroundings say, computed by the caller from the raw lines."""
    blocked: bool = False            # a word of the line means not-now / not-me / not-sure
    heading: Optional[str] = None    # the line above is a heading: 'current' | 'other' | None
    next_stops: bool = False         # the next line starts with a stop word
    list_kind: Optional[str] = None  # a medication header earlier on THIS line, directly before: 'current'
    after_other: bool = False        # the line above is about someone else: a subject-less statement is theirs


@dataclass(frozen=True)
class MedRead:
    kind: str                        # 'current' | 'stopped' (said not taken, or no longer)
    symbols: tuple[str, ...]
    head: str = ""                   # the text before "on <drug>", for the rest of the reader
    rest: str = ""                   # the text after the drug ("… and has diabetes"), likewise


def _read_list(norm: str, toks: list[tuple[str, int]], i: int, *, need_extra: bool = False
               ) -> Optional[tuple[tuple[str, ...], str]]:
    """<drug> [extras] {and <drug> [extras]}* [and <rest>]  ->  (symbols, rest); None if no drug."""
    t = [w for w, _ in toks]
    symbols: list[str] = []
    while True:
        d = _drug(t, i)
        if d is None:
            break
        symbols.append(d[0])
        i, count = _extras(t, d[1])
        if need_extra and count == 0 and len(symbols) == 1:
            return None
        if i < len(t) and t[i] in _LIST_JOINERS and _drug(t, i + 1) is not None:
            i += 1
            continue
        break
    if not symbols:
        return None
    if i >= len(t):
        return tuple(dict.fromkeys(symbols)), ""
    if t[i] in _LIST_JOINERS and i + 1 < len(t):
        return tuple(dict.fromkeys(symbols)), norm[toks[i + 1][1]:].strip()
    return None                                               # something else follows the drug


#: whole-statement "not taking" forms: the words before the drug, which is then the whole rest
_STOPPED_FRAMES = tuple(tuple(f.split()) for f in (
    "no", "not on", "not taking", "not currently on", "not currently taking", "no longer on",
    "no longer taking", "no longer takes", "no longer take", "stopped", "stopped taking", "discontinued",
    "discontinued taking", "never took", "never taken", "never been on", "never taking", "never on", "off",
    "came off", "come off", "used to take", "used to be on", "previously on", "previously took",
    "formerly on", "formerly took", "do not take", "does not take", "did not take", "dont take",
    "doesnt take", "didnt take", "isnt on", "isnt taking", "arent on", "wasnt on",
))


def _read_stopped(toks: list[tuple[str, int]]) -> Optional[MedRead]:
    t = [w for w, _ in toks]
    for first in ((), ("i",), ("we",), ("i", "have"), ("i", "am")):
        if tuple(t[:len(first)]) != first:
            continue
        for frame in _STOPPED_FRAMES:
            i = len(first) + len(frame)
            if tuple(t[len(first):i]) != frame:
                continue
            if i < len(t) and t[i] in {"the", "my"}:
                i += 1
            d = _drug(t, i)
            if d is not None and d[1] == len(t):               # a drug and NOTHING else
                return MedRead("stopped", (d[0],))
    return None


def read_medication(statement: str, ctx: MedContext = MedContext(),
                    head_ok: Callable[[str], bool] = lambda text: False) -> Optional[MedRead]:
    """Read one statement as a medication statement, or return None (it is then read as before).

    `head_ok(text)` says whether the rest of the reader fully understands `text` as a condition or a
    lab: the only kind of text allowed in front of 'on <drug>'."""
    if not statement or len(statement) > MAX_STATEMENT_CHARS:
        return None
    toks, norm = _tokens(normalise(statement).strip())
    if toks and toks[-1][0] == ")" and any(w == "(" for w, _ in toks):
        toks = toks[:-1]                                       # "diabetes (on metformin)"
    t = [w for w, _ in toks]
    if not any(w in _DRUGS for w in t):
        return None
    stopped = _read_stopped(toks)
    if stopped is not None:
        return stopped
    if ctx.blocked or ctx.next_stops:
        return None
    first_person = bool(t) and t[0] in {"i", "we"}
    if ctx.after_other and not first_person:
        return None

    # [adverbs] [subject] [adverbs] verb [the|my] [dose of] <drug> …
    i, subject = 0, ()
    while i < len(t) and t[i] in _ADVERBS:
        i += 1
    start = i
    for prefix in _SUBJECT_PREFIXES:
        if tuple(t[start:start + len(prefix)]) == prefix:
            subject, i = prefix, start + len(prefix)
            break
    while i < len(t) and t[i] in _ADVERBS:
        i += 1
    verbs = (_SUBJECT_VERBS if subject and subject[0] in {"i", "we"}
             else _AUX_VERBS if subject else _NO_SUBJECT_VERBS)
    if i < len(t) and t[i] in verbs:
        i += 1
        if i < len(t) and t[i] in {"the", "my"}:
            i += 1
        j, _ = _extras(t, i)                                   # "I take 500 mg of metformin"
        if j > i and j < len(t) and t[j] == "of":
            i = j + 1
        got = _read_list(norm, toks, i)
        if got is not None:
            return MedRead("current", got[0], rest=got[1])

    # medications: <drug> …
    j = 0
    while j < len(t) and t[j] in _CURRENT_HEADING:
        j += 1
    if j and j < len(t) and t[j] in {":", "=", "-", "–", "—"}:
        got = _read_list(norm, toks, j + 1)
        if got is not None:
            return MedRead("current", got[0], rest=got[1])

    # <drug> <dose> …: a dose or a timing says it is a medication entry; the line above must not be a
    # heading that is not a medication heading
    if ctx.heading != "other":
        got = _read_list(norm, toks, 0, need_extra=True)
        if got is not None:
            return MedRead("current", got[0], rest=got[1])
    # a bare <drug> in a list that a medication header started
    if ctx.list_kind == "current" or ctx.heading == "current":
        got = _read_list(norm, toks, 0)
        if got is not None:
            return MedRead("current", got[0], rest=got[1])

    # <condition or lab> on <drug>: the head must be understood by the rest of the reader
    for k in range(len(t) - 1, 0, -1):
        if t[k] in {"treated", "managed", "controlled"} and k + 1 < len(t) and t[k + 1] in {"with", "on"}:
            m = k + 2
        elif t[k] in _TAIL_VERBS:
            m = k + 1
        else:
            continue
        if m < len(t) and t[m] in {"the", "my"}:
            m += 1
        got = _read_list(norm, toks, m)
        if got is None or got[1]:
            continue
        head_toks = t[:k]
        while head_toks and head_toks[-1] in _HEAD_STRIP:
            head_toks.pop()
        head = norm[:toks[len(head_toks) - 1][1] + len(head_toks[-1])].strip(" (") if head_toks else ""
        head = re.sub(r"^(?:i|we)\s+(?:have|am|are)\s+", "", head)         # "I have diabetes and take …"
        if head and head_ok(head):
            return MedRead("current", got[0], head=head)
    return None
