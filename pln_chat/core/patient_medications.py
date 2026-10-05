"""Medications the My Patient reader understands: only a drug the knowledge base can act on.

The knowledge base has ONE medication fact that matters to a patient: a supplement plan flags a
supplement that interacts with a drug the person takes (`(Interaction Berberine Metformin …)`,
supplement_evidence.metta), against `(CurrentMedication <Patient> <Drug>)`. So the reader reads a
drug ONLY from the table below — names that are identities (the same substance: generic, salt,
formulation, brand), never similarity — and leaves every other drug exactly where it was, in "not
understood".

It is written to be strict, not clever. Round 1 of the adversarial review
(docs/kb_quick_wins/REVIEW_ROUND1.md) found that the first version, which read "<anything> on
metformin" with a deny-list of heads, read a drug the person does not take in dozens of ways: a plan
("I want to be on metformin"), a question, a past, a negation spelled "isn't" or "wasnt", someone else
("my grandma takes it"), a qualifier cut off by the statement splitter ("I take metformin, but not
anymore"), a heading above ("Past medications:") — the long tail the smoking reader had already taught.
Round 2 (docs/kb_quick_wins/REVIEW_ROUND2.md) found the redesign's own tails: a tokenizer that silently
dropped "❌" and "не", a heading that protected only the line directly below it, a retraction two lines
down, a relative's possessive ("my husband's diabetes. Takes metformin."). So now:

* a medication is read only from a statement that is, WHOLE, a closed grammar over tokens
      [I | we | I am | I have been] [currently | now | still …] (take | takes | taking | on | using)
          [the | my] <drug> [dose · timing · since YEAR · for N years · as prescribed]
      medications: <drug> …          <drug> <dose> …          <condition or lab> on <drug>
  A head in front of "on <drug>" must be fully understood by the rest of the reader as a condition or a
  lab; any other head is not a head. No free-text subject, no imperative ("take metformin"), no question;
* it is read only if no word of the WHOLE LINE means not-now, not-me or not-sure: negations in every
  spelling, a past, a plan, a wish, advice, a hedge, a correction, a question, a contrast, a relative or
  other person (possessives and plurals included), an allergy, a year other than "since YEAR". The
  statement splitter cuts "I do not, in fact, take metformin" and "I take metformin, but not anymore"
  into statements that each look fine alone;
* a statement or a line with a character this grammar cannot account for (an emoji, "[ ]", "~~", a
  non-Latin word, a zero-width character) is not read. Only ASCII letters, digits, spaces and
  `.,;:!?()%/+&="-–—°µ` are accounted for;
* the context is scanned, not sampled: the nearest HEADING above (through the list items under it, up to
  six lines) decides whether an entry is a current medication; the next two lines are checked for a
  retraction (a line the reader understands as a lab is never one); someone else on the lines above, a
  year on a neighbouring line and a blocking word in the line above a bare entry are decided in one place
  (context_for);
* a statement that also carries age, sex or smoking is not read: the reader rewrites it (drops "until
  2019", turns "isn't" into "isn t") before a medication rule could see it;
* "not taking" is read only from a closed set of whole-statement forms (no / not on / stopped /
  discontinued / never took / off / no longer / used to take), and records nothing: a drug said not to be
  taken is not a medication. Saying both is a contradiction to fix;
* nothing is guessed about other drugs: the rest of a statement ("… and lisinopril", "… and has
  diabetes") goes on to be read as it would have been alone, from the text as typed (hyphens intact);
* no regular expression here can backtrack: the grammar is matched on tokens, in time linear in the
  statement, and nothing longer than MAX_STATEMENT_CHARS / MAX_LINE_CHARS is normalised or read.

The cost of a missed reading is a missing flag, which is what every patient had before; the cost of a
wrong one is a flag for a drug the person does not take.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

#: A medication statement is short; a longer one is not read (no cost, and no work for a hostile one).
MAX_STATEMENT_CHARS = 240
MAX_LINE_CHARS = 600
#: how many non-empty lines above a statement are scanned for its heading, and below for a retraction
HEADING_LOOKBACK = 6
RETRACTION_LOOKAHEAD = 2

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
ended end finished completed tapered tapering weaned held hold paused pause dc dcd off ago until till
was were had used previously formerly before after
past previous prior former inactive historical expired withdrawn voided cancelled canceled replaced
replace replacing switched switch changed
will would shall should could can might may must need needs want wants wanted hope hopes hoping hopefully
wish wishes plan plans planning planned going intend intends intended decided decide consider considers
considering think thinks thinking advised advise advice recommend recommends recommended suggest
suggests suggested told asked prescribed prescribe start starts started starting begin begins began
beginning try tries tried trying
maybe perhaps possibly probably sometimes occasionally rarely seldom unless when whether unsure
uncertain unable refuse refuses refused declined hardly barely if or
tomorrow tonight soon later next eventually supposedly apparently actually correction sorry typo mistake
wrong kidding joke jk lol
but however although though except then instead rather anymore again restart restarted resume resumed
he she they his her their him them hes shes theyre
what which how why who whom whose
allergic allergy allergies intolerant intolerance reaction reactions adverse side avoid avoids avoided
contraindicated contraindication
dropped drop dropping gave give giving swapped swap swapping ran run error placebo ordered order pending
guess guessing believe idk scratch ignore hypothetically last
monday tuesday wednesday thursday friday saturday sunday
""".split())
#: …and these name someone other than the person. A word with a trailing "s" counts too (husband's,
#: parents, friends). The reader's own set-aside rule (patient_text._SOMEONE_ELSE) is consulted as well.
_OTHERS = frozenset("""
wife husband partner spouse boyfriend girlfriend mother mom mum father dad parent son daughter child
children kid brother sister sibling aunt uncle cousin grandmother grandfather grandparent grandma grandpa
granny nan nana stepmother stepfather stepmom stepdad nephew niece friend neighbour neighbor boss
colleague coworker roommate roomate flatmate doctor nurse patient relative family someone somebody anyone
everyone people others twin carer mate ex pet dog cat
""".split())
#: words of a heading that makes what is under it NOT a current-medication list
_HEADING_OTHER = frozenset("""
allergies allergy allergic intolerance intolerances reactions adverse avoid plan recommendations
recommendation options treatment treatments history past previous prior former inactive historical
discontinued stopped family relatives notes note contraindications contraindicated
old hx visit discharge pre preop
""".split())
#: words of a heading that introduces a CURRENT medication list
_CURRENT_HEADING = frozenset({
    "medications", "medication", "meds", "med", "medicines", "medicine", "rx", "drugs", "list",
    "current", "regular", "daily", "my", "usual",
})
_NEGATORS = frozenset("no not never none nothing nobody without denies denied".split())
#: words that are headings ONLY on a line of nothing but heading vocabulary (see heading_kind)
_HEADING_ONLY_WORDS = frozenset("old hx visit discharge pre preop dropped gave swapped ran reason active".split())
_STOP_START = frozenset("""
stopped stop stops discontinued discontinue ceased ended finished completed tapered weaned held hold paused
off no not never none until till status quit used previously formerly was were had replaced replace
cancelled canceled inactive historical past expired withdrawn voided dc dcd entered history prior
previous former switched changed dropped gave swapped ran reason active
""".split())
#: a next line that has one of these in its first four words takes the drug above back ("I do not take it", "I don't
#: take it anymore", "I haven't taken it since June", "I have recently stopped"). Narrower than _BLOCK on purpose: a
#: neighbour such as "my albumin was 4.1" has to keep reading.
_RETRACT_ANY = frozenset("""
no not never nor dont doesnt didnt havent hasnt hadnt isnt arent wasnt werent cant wont wouldnt couldnt
stopped stop stops quit quits discontinued ceased ended finished tapered dropped gave swapped recently
""".split())

_HEADING_VOCABULARY = (_HEADING_ONLY_WORDS | _HEADING_OTHER | _STOP_START | _CURRENT_HEADING
                       | frozenset("of the at last op my for".split()))

_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789 \t.,;:!?()%/+&=\"-–—°µμ")
_APOSTROPHES = re.compile("[´`’‘′ʼ＇]")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[%/+&,:;=()?!\"]|[-–—]")
_NUM_UNIT = re.compile(r"(\d)(mg|g|mcg|µg|μg|ug)\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
#: a year: 1900-2099 that is not a dose ("2000 mg") and is not preceded by "since"
_YEAR = re.compile(r"(?<![0-9.])(?:19|20)\d\d(?![0-9])(?!\s*(?:mg|g|mcg|ug|µg|μg)\b)")
_DC = re.compile(r"\bd\s*/\s*cd?\b|\bdc'?d\b")
_EXPAND = (
    (re.compile(r"\bi'm\b"), "i am"), (re.compile(r"\bwe're\b"), "we are"),
    (re.compile(r"\bi've\b"), "i have"), (re.compile(r"\bwe've\b"), "we have"),
    (re.compile(r"\b(?:i'll|we'll|i'd|we'd)\b"), "will"),     # a plan or a past: both are blocked
)
_CONTRACTION = re.compile(r"\b(\w+)n['\s]t\b")
_MARKDOWN = re.compile(r"[*_#>~`|\[\]]")
_BULLET = re.compile(r"^[\s\-•*·]*(?:\d{1,2}[.)]\s+)?")
_HAS_CONTENT = re.compile(r"[^\W_]")


def normalise(text: str) -> str:
    """Lower case, NFKC, one apostrophe, I'm / we're expanded, other contractions glued (isn't, isn t
    -> isnt), digits and units apart ('500mg' -> '500 mg'), '1,000' -> '1000'. Callers cap the length
    first: nothing here is meant for a long text."""
    t = _APOSTROPHES.sub("'", text or "")          # before NFKC: an acute accent becomes a space and a mark
    t = unicodedata.normalize("NFKC", t).lower()
    for rx, repl in _EXPAND:
        t = rx.sub(repl, t)
    t = _CONTRACTION.sub(r"\1nt", t)
    t = t.replace("'", "")
    t = _NUM_UNIT.sub(r"\1 \2", t)
    return _THOUSANDS.sub("", t)


def _foreign(norm: str) -> bool:
    """A character the grammar cannot account for: an emoji, a bracket, a non-Latin letter, a zero-width mark."""
    return any(ch not in _ALLOWED for ch in norm)


def _is_other(word: str) -> bool:
    return word in _OTHERS or (word.endswith("s") and word[:-1] in _OTHERS)


def _tokens(norm: str) -> list[tuple[str, int]]:
    """[(token, offset in `norm`)]; hyphens are tokens of their own, so `norm` slices keep them."""
    return [(m.group(0), m.start()) for m in _TOKEN.finditer(norm)]


def _blocking(norm: str) -> bool:
    """`norm` (a normalised line) carries a word that means not-now / not-me / not-sure."""
    if "?" in norm or _foreign(norm) or _DC.search(norm):
        return True
    norm = norm.replace("as prescribed", " ")        # "I take metformin as prescribed": not a doctor's order
    if any(w in _BLOCK or _is_other(w) for w in re.findall(r"[a-z]+", norm)):
        return True
    # a year says WHEN ("In 2019, on metformin", "2015-2020"): only "since 2015" keeps it current
    return any(not re.search(r"\bsince\s*$", norm[:m.start()]) for m in _YEAR.finditer(norm))


def line_blocked(line: str) -> bool:
    """A word on this line means a medication on it is not read as current: not-now, not-me, not-sure.
    Judged on the whole LINE, because the statement splitter cuts a qualifier loose."""
    if len(line) > MAX_LINE_CHARS:
        return True
    return _blocking(normalise(line))


def line_about_other(line: str) -> bool:
    """The line names someone other than the person ("My mother has diabetes", "Wife", "Mom's meds")."""
    if len(line) > MAX_LINE_CHARS:
        return False
    return any(_is_other(w) for w in re.findall(r"[a-z]+", normalise(line)))


def heading_kind(line: str) -> Optional[str]:
    """'current' | 'other' when `line` is a heading, None for any other line. A heading ends in ':' or
    '-', or is a short line with no digits that names a past, a negation, an allergy or a person
    ("Discontinued medications", "Allergies", "Wife"); markdown and bullets are ignored. 'current' only
    for the plain medication headings ("Medications:", "Current medications -")."""
    if len(line) > MAX_LINE_CHARS:
        return None
    text = _BULLET.sub("", _MARKDOWN.sub(" ", line)).strip()
    norm = normalise(text).strip()
    words = re.findall(r"[a-z]+", norm)
    ends = norm.endswith((":", "-", "–", "—"))
    if not words:                                       # no Latin letters at all: unreadable, so not a current list
        return "other" if ends and _HAS_CONTENT.search(text) else None
    if _foreign(norm):
        return "other" if ends else None
    if ends:
        return "current" if all(w in _CURRENT_HEADING for w in words) else "other"
    if len(words) <= 4 and not re.search(r"\d", norm) and words[0] not in _NEGATORS \
            and not any(w in _DRUGS for w in words):         # "stopped metformin" is a statement, not a heading
        if any((w in _HEADING_OTHER or w in _STOP_START) and w not in _HEADING_ONLY_WORDS or _is_other(w) for w in words):
            return "other"
        # the words the round-2 closure added ("old", "hx", "visit", "pre", "active", "reason", "gave"…) mean a
        # heading only on a line made of heading vocabulary ("Old meds", "Last visit", "Pre-op"): "I have
        # pre-diabetes" and "I am active" are sentences
        if any(w in _HEADING_ONLY_WORDS for w in words) and all(w in _HEADING_VOCABULARY for w in words):
            return "other"
        if all(w in _CURRENT_HEADING for w in words):
            return "current"
    return None


def starts_current_header(statement: str) -> bool:
    """'medications: …' — whatever follows the colon: a list of drugs that starts a medication list."""
    if len(statement) > MAX_STATEMENT_CHARS:
        return False
    t = [w for w, _ in _tokens(normalise(statement))]
    j = 0
    while j < len(t) and t[j] in _CURRENT_HEADING:
        j += 1
    return bool(j) and j < len(t) and t[j] in {":", "=", "-", "–", "—"}


#: What may follow a current medication on its line without the reader understanding it: more of the same list.
#: A CLOSED allow-list on purpose — a deny-list of qualifiers ("dropped it", "ran out", "entered in error",
#: "I guess" …) never ends, and the round-2 red team found a new one each time. Anything else makes the
#: medication unread (a missed flag, said aloud) instead of guessing it still holds.
_LIST_FILLER = frozenset("""
and with also currently now still regularly usually always daily nightly weekly twice once thrice a an the of to my
plus day week morning evening night bedtime breakfast lunch dinner meal meals food at in per each as needed prn
yes hi hello hey well so ok okay honestly personally present moment right baby low dose lowdose
bid tid qd qhs mg mcg ug g iu units unit ml tablet tablets tab tabs pill pills capsule capsules cap caps er xr sr dr
extended release on taking takes take using uses
""".split())
#: drugs and supplements a person lists beside a prescription, and the stems of the generic names
_COMMON_DRUGS = frozenset("""
aspirin insulin warfarin levothyroxine synthroid liothyronine ibuprofen paracetamol acetaminophen tylenol naproxen
allopurinol amlodipine lisinopril losartan valsartan atorvastatin simvastatin rosuvastatin pravastatin metoprolol
carvedilol bisoprolol propranolol atenolol hydrochlorothiazide furosemide spironolactone clopidogrel apixaban
rivaroxaban digoxin omeprazole pantoprazole esomeprazole sertraline fluoxetine citalopram escitalopram venlafaxine
bupropion amitriptyline gabapentin pregabalin prednisone prednisolone dexamethasone methotrexate hydroxychloroquine
tamsulosin finasteride sildenafil tadalafil montelukast cetirizine loratadine fexofenadine albuterol salbutamol
ezetimibe fenofibrate gemfibrozil glipizide glimepiride gliclazide sitagliptin linagliptin empagliflozin
dapagliflozin canagliflozin liraglutide semaglutide tirzepatide pioglitazone vitamin vitamins d c e k b b6 b12
multivitamin magnesium zinc calcium iron potassium fish oil omega 3 supplement supplements probiotic coq10 creatine
statin statins ppi ppis ace inhibitor inhibitors blocker blockers lipitor zocor crestor norvasc zestril ozempic wegovy
mounjaro melatonin berberine resveratrol nmn nad nicotinamide quercetin curcumin turmeric glucosamine collagen d3 k2 b2
antihistamine antacid
""".split())
_DRUG_STEM = re.compile(r"[a-z]{3,}(?:pril|sartan|statin|olol|dipine|prazole|thiazide|tidine|formin|gliptin|"
                        r"gliflozin|glutide|glitazone|semide|cillin|mycin|floxacin|oxetine|pram|triptyline|"
                        r"setron|zepam|zolam|fibrate|solone|terol)")


def _list_word(w: str) -> bool:
    return (w in _LIST_FILLER or w in _CURRENT_HEADING or w in _COMMON_DRUGS or w in _DRUGS or bool(_NUMBER.fullmatch(w))
            or bool(_DRUG_STEM.fullmatch(w)))


def benign_sibling(text: str) -> bool:
    """`text` is something the reader did NOT understand, next to a current medication on the same line.
    True only when it is more of a medication list ("lisinopril 10 mg", "and atorvastatin", "twice daily
    with food"): every word a dose, a timing, a drug or supplement the allow-list knows. "dropped it", "I
    guess", "entered in error" and anything unknown are not."""
    if len(text) > MAX_STATEMENT_CHARS:
        return False
    norm = normalise(text)
    if "?" in norm or _foreign(norm):
        return False
    return all(_list_word(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", norm))


def may_name_a_medication(text: str) -> bool:
    """A cheap pre-check for the reader: the text names a drug the KB knows, or has a word a medication
    clause hangs on ("on", "taking", "takes" …), which an unknown drug ("hypertension on lisinopril") needs."""
    return bool(mentioned_symbols(text)) or bool(_TAIL_WORD.search(text or ""))


def mentioned_symbols(text: str) -> tuple[str, ...]:
    """The KB symbols of the drugs `text` names anywhere (a cheap pre-check, then the tokens). A very long
    text is only scanned as typed: nothing long is normalised."""
    text = text or ""
    low = normalise(text) if len(text) <= 2 * MAX_STATEMENT_CHARS else text.lower()
    if not any(d in low for d in _DRUGS):
        return ()
    return tuple(dict.fromkeys(_DRUGS[w][0] for w in re.findall(r"[a-z]+", low) if w in _DRUGS))


def retracts(line: str) -> bool:
    """The line takes a medication on the line above back: it starts with a stop / negation / past word
    ("stopped 2021", "I quit it in June", "Status: stopped", "d/c'd"), is a bare date, or is unreadable."""
    if len(line) > MAX_LINE_CHARS:
        return True
    norm = normalise(_BULLET.sub("", line)).strip()
    if _foreign(norm) or _DC.search(norm):
        return True
    words = re.findall(r"[a-z]+", norm)
    if any(w in _STOP_START for w in words[:2]) or any(w in _RETRACT_ANY for w in words[:4]):
        return True
    return len(words) <= 3 and any(not re.search(r"\bsince\s*$", norm[:m.start()]) for m in _YEAR.finditer(norm))


# ── the context of a statement ───────────────────────────────────────────────

@dataclass(frozen=True)
class MedContext:
    """What a statement's surroundings say, computed by `context_for` from the raw lines."""
    blocked: bool = False            # a word of the line means not-now / not-me / not-sure / unreadable
    heading: Optional[str] = None    # the nearest heading above: 'current' | 'other' | None
    intro_blocks: bool = False       # the line directly above carries a blocking word (a bare entry is then not read)
    retracted: bool = False          # a line below takes the drug back
    other: bool = False              # this line, or one just above, is about someone else
    list_kind: Optional[str] = None  # a medication header earlier on THIS line: 'current'


def _nonempty(lines: Sequence[str], n: int, step: int, limit: int) -> list[str]:
    out, k = [], n + step
    while 0 <= k < len(lines) and len(out) < limit:
        if _HAS_CONTENT.search(lines[k]):                # a '-----' rule, a bullet, a zero-width line is no line
            out.append(lines[k][:MAX_LINE_CHARS + 1])
        k += step
    return out


def context_for(lines: Sequence[str], n: int, *, is_lab_line: Callable[[str], bool] = lambda line: False,
                about_other: Callable[[str], bool] = lambda line: False) -> MedContext:
    """The context of a statement on line `n` of `lines`. `is_lab_line` says whether the rest of the reader
    understands a line as a lab (never a retraction); `about_other` is its own someone-else rule."""
    line = lines[n]
    above = _nonempty(lines, n, -1, HEADING_LOOKBACK)
    below = _nonempty(lines, n, +1, RETRACTION_LOOKAHEAD)
    heading = next((k for k in map(heading_kind, above) if k), None)
    if above and len(re.findall(r"[a-z]+", normalise(above[0])[:MAX_LINE_CHARS])) <= 4 and _YEAR.search(normalise(above[0])):
        heading = "other"                                # a date line above: the entry below is dated
    return MedContext(
        blocked=line_blocked(line) or about_other(line),
        heading=heading,
        intro_blocks=bool(above) and line_blocked(above[0]),
        retracted=any(retracts(b) and not is_lab_line(b) for b in below),
        other=line_about_other(line) or about_other(line)
        or any(line_about_other(a) or about_other(a) for a in above[:3]),
    )


# ── the token grammar ────────────────────────────────────────────────────────

_UNITS = frozenset({"mg", "g", "mcg", "ug", "μg", "µg"})
_PER = frozenset({"a", "per", "each"})
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
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PLAIN_WORD = re.compile(r"[a-z]{4,}")
_TAIL_WORD = re.compile(r"\b(?:on|taking|takes|take|using|uses)\b", re.I)
#: a word after "on" that is not the name of a drug: "hypertension on medication", "diabetes on a diet"
_NOT_A_DRUG = frozenset("""
medication medications medicine medicines meds pills pill tablets tablet drugs treatment treatments therapy
diet exercise lifestyle nothing none supplements supplement vitamins vitamin herbs vacation holiday
leave duty call track board
""".split())


def _num(tok: str) -> Optional[float]:
    return float(tok) if _NUMBER.fullmatch(tok) else None


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
            elif j + 1 < n and t[j] in _PER and t[j + 1] in {"day", "week"}:
                j += 2                                         # "500 mg a day", "per day", "each day"
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
    """A drug name at t[i]: (symbol, index after it). Hyphens are tokens: 'metformin-xr' is three."""
    if i >= len(t) or t[i] not in _DRUGS:
        return None
    symbol, suffixes = _DRUGS[t[i]]
    j = i + 1
    while j < len(t):
        k = j + 1 if t[j] == "-" else j                        # an optional hyphen
        if k >= len(t):
            break
        if t[i] == "metformin" and k + 1 < len(t) and (t[k], t[k + 1]) in _FORMULATIONS:
            j = k + 2                                          # "extended release"
        elif t[i] == "metformin" and k + 2 < len(t) and t[k + 1] == "-" and (t[k], t[k + 2]) in _FORMULATIONS:
            j = k + 3                                          # "extended-release"
        elif t[k] in suffixes:
            j = k + 1                                          # "hcl", "xr", "er" …
        else:
            break
    return symbol, j


@dataclass(frozen=True)
class MedRead:
    kind: str                        # 'current' | 'stopped' (said not taken, or no longer)
    symbols: tuple[str, ...]
    head: str = ""                   # the text before "on <drug>", as typed, for the rest of the reader
    rest: str = ""                   # the text after the drug ("… and has diabetes"), as typed, likewise
    tail: str = ""                   # kind "other": "on lisinopril" — a drug the KB has nothing on, left unread


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


def _read_stopped(t: list[str]) -> Optional[MedRead]:
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


#: text that normalise() rewrites (5'1 -> 51, 245,000 -> 245000): a head or rest cut out of the rewritten
#: text would no longer be what was typed, so such a statement is left to the reader as it was
_DOSE_COMMA = re.compile(r"\d,\d{3}(?=\s*(?:mg|mcg|ug|µg|μg|g|iu|units?)\b)", re.I)
_LOSSY = re.compile(r"\d['´`’‘′ʼ＇]\d|\d,\d{3}(?!\d)")


def read_medication(statement: str, ctx: MedContext = MedContext(),
                    head_ok: Callable[[str], bool] = lambda text: False) -> Optional[MedRead]:
    got = _read_medication(statement, ctx, head_ok)
    # a thousands comma INSIDE a dose ("metformin 1,000 mg") is read as the dose it is and never ends up in a slice
    if got is not None and (got.head or got.rest or got.tail) and _LOSSY.search(_DOSE_COMMA.sub("", statement)):
        return None
    return got


def _read_medication(statement: str, ctx: MedContext = MedContext(),
                     head_ok: Callable[[str], bool] = lambda text: False) -> Optional[MedRead]:
    """Read one statement as a medication statement, or return None (it is then read as before).

    `head_ok(text)` says whether the rest of the reader fully understands `text` as a condition or a
    lab: the only kind of text allowed in front of 'on <drug>'."""
    if not statement or len(statement) > MAX_STATEMENT_CHARS:
        return None
    norm = normalise(statement).strip()
    if _foreign(norm):
        return None
    toks = _tokens(norm)
    if toks and toks[-1][0] == ")" and any(w == "(" for w, _ in toks):
        toks = toks[:-1]                                       # "diabetes (on metformin)"
    t = [w for w, _ in toks]
    if not any(w in _DRUGS or w in _TAIL_VERBS for w in t):
        return None
    first_person = bool(t) and t[0] in {"i", "we"}
    if not (ctx.other and not first_person):
        stopped = _read_stopped(t)
        if stopped is not None:
            return stopped
    if ctx.blocked or ctx.retracted or ctx.heading == "other":
        return None
    if ctx.other and not first_person:
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

    # <drug> <dose> …: a dose or a timing says it is a medication entry — unless the line above
    # carries a blocking word, which a bare entry (no verb of its own) cannot answer
    if not ctx.intro_blocks:
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

    # <condition or lab> on <a drug the KB has nothing on>: "hypertension on lisinopril". The head is read as it
    # would have been alone and the drug stays unread; a head the reader does not understand is no head.
    for k in range(len(t) - 1, 0, -1):
        if t[k] not in _TAIL_VERBS:
            continue
        m = k + 1
        if m < len(t) and t[m] in {"the", "my"}:
            m += 1
        if m >= len(t) or not _PLAIN_WORD.fullmatch(t[m]) or t[m] in _DRUGS or t[m] in _NOT_A_DRUG \
                or t[m] in _BLOCK or _is_other(t[m]) or t[m] in _CURRENT_HEADING or head_ok(t[m]):
            continue
        if _extras(t, m + 1)[0] != len(t):
            continue
        head_toks = t[:k]
        while head_toks and head_toks[-1] in _HEAD_STRIP:
            head_toks.pop()
        head = norm[:toks[len(head_toks) - 1][1] + len(head_toks[-1])].strip(" (") if head_toks else ""
        head = re.sub(r"^(?:i|we)\s+(?:have|am|are)\s+", "", head)
        if head and head_ok(head):
            return MedRead("other", (), head=head, tail=norm[toks[k][1]:].strip())
    return None
