"""The MeTTa runtime seam: run one MeTTa program on hyperon or on PeTTa.

`core.pln_runner` is the only place the app executes MeTTa, and it does so through this module:

    runtime = get_runtime()                                   # PLN_RUNTIME = hyperon | petta | auto
    groups = runtime.run_program(kb_text, ["!(f 1)", "!(g)"])   # -> [["(res a)"], []]

`run_program` loads `kb_text` into a FRESH interpreter, evaluates each `!` expression and returns, for each
expression, the result atoms as strings. Every consumer downstream parses those strings, so the two runtimes
have to agree on how an answer reads; the places where they do not are listed under "What PeTTa changes".

Two runtimes
------------
HyperonRuntime    exactly what `pln_runner._hyperon_run` did before this module existed: `MeTTa()`, one
                  `run(kb_text)`, one `run("\\n".join(expressions))`, `str(atom)` for each result. In process.
                  `deadline_s` is accepted and ignored: a GIL-holding Rust call cannot be preempted, so the
                  deadline belongs to the executor, which kills the whole worker (core/executor.py).
PettaRuntime      MeTTa compiled to SWI-Prolog (trueagi-io/PeTTa), driven through janus. Every `run_program` forks
                  a fresh child process; see below for why.

Why PeTTa runs in a forked child, once per program
--------------------------------------------------
* PeTTa's space and its compiled clauses are process-global with no reset. A second KB load in one process
  duplicates every clause and multiplies recursive answers, so there is exactly ONE load per process and a
  new program needs a new process. The child imports janus and PeTTa, loads the KB with one
  `process_metta_string`, evaluates each expression with its own call (PeTTa hands Python one flat list per
  call, so a call per `!` keeps the per-expression grouping `expr_index` depends on) and sends the strings
  back over a pipe.
* The parent never imports janus or petta: `available()` only inspects `importlib.util.find_spec` and a few
  paths. Forking after janus has started SWI-Prolog is unsafe, so only children ever start it.
* A runaway program is stopped by SIGKILL at the deadline. A janus call cannot be interrupted politely.
* The fork is `os.fork()`, not `multiprocessing.Process`: the executor's workers are daemonic and a daemonic
  process may not have `multiprocessing` children.
* The child is also bounded without the parent. It lowers SWI-Prolog's stack limit
  (`PETTA_STACK_LIMIT_MB`), sets an RLIMIT_CPU backstop a little past the deadline, and on Linux asks to be
  SIGKILLed when its parent dies. The executor SIGKILLs a worker at its deadline, and the worker's PeTTa
  child would otherwise run on, unreaped and unbounded, with nobody to read its answer.
* Its stdin, stdout and stderr are /dev/null: PeTTa prints traces (`println!`, warnings) at the file
  descriptor level, below Python's `sys.stdout`, and they must reach neither the server's log nor a test's
  captured output. Errors travel through the pipe, not through stderr.

What PeTTa changes (and what this module does about it)
-------------------------------------------------------
1. A call stored inside a FACT is not evaluated: `(Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))`
   keeps a list where hyperon computes a number, and the first arithmetic on it raises. `reduce_stored_calls`
   rewrites such facts in the KB text, before the load, using the tier table found in the same text. Only
   the KB text is rewritten, never an expression, and only for PeTTa. The other three incompatibilities the
   study found (`(superpose ())`, the nested conjunction, ...) were fixed in the sources.
2. PeTTa prints `true` and `false`; hyperon prints `True` and `False`. `normalise_result` is the one function
   that maps the first onto the second, outside strings. Numbers are NOT rewritten: PeTTa keeps integers
   as integers (`3` where hyperon sometimes prints `3.0`) and prints `1.0e-05`, and every consumer in this
   repo accepts both. The raw strings stay available as `PettaRuntime.last_raw`.
3. A form that fails to compile raises through janus, but the forms before it stay loaded. Any exception
   during the load is therefore a failed load, and since the process is the unit of state it is simply
   discarded. `_PettaEngine.load` also treats a load goal that FAILS WITHOUT raising as a failed load (by
   calling the janus goal directly instead of PeTTa's own wrapper, which turns that into `[]`, the same as
   a good load of a KB with no `!` forms) — defensively: no malformed KB tried against this PeTTa commit
   (13 probed, from unbalanced parentheses to a bad `superpose`) actually fails this way, every one raises
   instead, so this branch has no known trigger and is untested. Kept because a goal failing silently and
   being read as "loaded cleanly" would be a worse failure mode than a branch that never fires.
4. A query that fails without raising comes back as `[]`: an empty answer, as on hyperon.
5. Facts and solutions come back in assertion order where hyperon often returns the reverse.
6. `assert` and `test` call `halt(1)`; `add-atom` and `remove-atom` change the process-global space
   (`remove-atom` retracts every identical copy); a runtime-added `(= ...)` becomes a global clause;
   `import!`, `py-call`, `shell`, `callPredicate` (and friends — PeTTa's own "call any Prolog predicate by
   name" doors), ... reach the host. `core.metta_validator` and `PettaRuntime`/`PettaSession` (via
   `petta_refused_forms`/`petta_refused_forms_in_kb`) all refuse the LITERAL spelling, in `kb_text`, an
   expression and a session's text alike — a fast, friendly 422 for the common case, and nothing more: a
   symbol PeTTa builds at run time (`atom_concat`) never appears as a token in any of that text, so none
   of these scans can see it. `_install_petta_sandbox` is the actual boundary: it redefines the dangerous
   predicates inside the Prolog engine itself, so the restriction holds regardless of how the program's
   text spells or constructs the call that reaches them.

Sessions
--------
`runtime.session(kb_text)` is for tests and tools, not for production requests: a stateful interpreter that
loads the KB once and then answers `run(text)` calls, like a hyperon `MeTTa` object. Text without a `!`
adds facts to that interpreter's space. Under petta a session is one long-lived forked child; `close()`
kills and reaps it.

Nothing in this module imports hyperon, janus or petta at import time.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from multiprocessing.connection import Connection, Pipe
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

__all__ = [
    "MettaRuntimeError", "Runtime", "Session", "HyperonRuntime", "PettaRuntime",
    "get_runtime", "reset_runtime_cache", "selected_runtime_name", "runtime_available_for",
    "normalise_result", "reduce_stored_calls", "read_tier_table", "Reduction",
    "split_top_level_forms", "petta_refused_forms", "petta_refused_forms_in_kb", "PETTA_REFUSED_FORMS",
    "RUNTIME_NAMES",
]

_LOG = logging.getLogger(__name__)

#: the values of PLN_RUNTIME
RUNTIME_NAMES: tuple[str, ...] = ("hyperon", "petta", "auto")

#: How far past the parent's deadline a PeTTa child may keep computing if nobody kills it (CPU seconds).
BACKSTOP_GRACE_S = 10.0
#: How long a finished child is given to exit by itself before it is killed.
_EXIT_WAIT_S = 5.0
#: Longest piece of an underlying error text kept in a MettaRuntimeError message.
_MAX_DETAIL_CHARS = 1500


class MettaRuntimeError(RuntimeError):
    """A MeTTa program could not be run to an answer.

    `str(error)` is safe to show a caller: it says what happened and keeps the underlying engine text
    (clipped). `kind` is one of: `unavailable` (the runtime cannot start here), `load` (the knowledge base
    did not load), `expression` (an expression raised), `timeout` (the deadline killed the run), `crash`
    (the interpreter process died without an answer), `refused` (the program uses a form the PeTTa runtime
    does not run), `session` (a session was closed or lost), `runtime` (anything else, notably every
    hyperon exception). `detail` is the underlying text on its own.
    """

    def __init__(self, message: str, *, kind: str = "runtime", detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail


@runtime_checkable
class Session(Protocol):
    """A stateful interpreter: the KB is loaded once, `run` can be called repeatedly."""

    #: raw result strings of the most recent `run`, before normalisation (petta); equal to the result on hyperon
    last_raw: list[list[str]]

    def run(self, text: str, deadline_s: Optional[float] = None) -> list[list[str]]:
        """Evaluate `text`; one list of result strings per `!` form in it (forms without `!` add facts)."""

    def close(self) -> None:
        """Release the interpreter. Idempotent."""


@runtime_checkable
class Runtime(Protocol):
    name: str
    #: Can the runtime stop a program at `deadline_s` itself? PeTTa can (it kills its child process); hyperon
    #: cannot, since a GIL-holding Rust call is not preemptible in process: its deadline is the executor's kill.
    enforces_deadline: bool

    def available(self) -> bool:
        """Can this runtime run here? Must not import the engine into the calling process."""

    def run_program(self, kb_text: str, expressions: Sequence[str],
                    deadline_s: Optional[float] = None) -> list[list[str]]:
        """Load `kb_text` into a fresh interpreter, evaluate each `!` expression, return the result atoms
        of each as strings. Raises MettaRuntimeError."""

    def session(self, kb_text: str) -> Session:
        """A stateful interpreter with `kb_text` loaded (tests and tools)."""


# ── text scanning ─────────────────────────────────────────────────────────────

#: a "string", a ;comment, a parenthesis, or a bare token
_SCAN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|;[^\n]*|[()]|[^\s()";]+')
#: a "string", a parenthesis, or a bare token: the grammar of a RESULT, which has no comments
_RESULT_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[()]|[^\s()"]+')
_NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def split_top_level_forms(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level parenthesised form of `text`, in order.

    A `!` directly in front of the opening parenthesis belongs to the span. Strings (a `;` or a parenthesis
    inside "..." is text) and `;` comments are skipped. Bare top-level tokens are not forms and are ignored,
    as PeTTa's own reader would reject them. An unbalanced tail is dropped.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    bang_start = bang_end = -2
    for m in _SCAN_RE.finditer(text):
        tok = m.group()
        first = tok[0]
        if first == ";" or first == '"':
            continue
        if tok == "(":
            if depth == 0:
                start = bang_start if bang_end == m.start() else m.start()
            depth += 1
        elif tok == ")":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    spans.append((start, m.end()))
        elif depth == 0 and tok == "!":
            bang_start, bang_end = m.start(), m.end()
    return spans


def _form_tokens(form: str) -> list[tuple[str, int, int]]:
    """The non-comment tokens of one form with their offsets."""
    return [(m.group(), m.start(), m.end()) for m in _SCAN_RE.finditer(form) if m.group()[0] != ";"]


# ── the result normaliser ─────────────────────────────────────────────────────

_BOOLEANS = {"true": "True", "false": "False"}


def normalise_result(text: str) -> str:
    """Spell a PeTTa result the way hyperon does: standalone `true` / `false` tokens outside strings
    become `True` / `False`. Nothing else changes (numbers keep PeTTa's rendering; every consumer accepts it).

    The only repo parser that compares a boolean spelling is `linage2_router._contribution`
    (`reads[2][1] == "True"`); every other parser reads numbers, which both runtimes print parseably.
    A symbol literally named `true` used as DATA cannot be told from the boolean and is rewritten too.
    """
    if "true" not in text and "false" not in text:
        return text
    return _RESULT_TOKEN_RE.sub(lambda m: _BOOLEANS.get(m.group(), m.group()), text)


# ── the stored-call reducer (PeTTa only, KB text only) ────────────────────────

@dataclass(frozen=True)
class Reduction:
    """What `reduce_stored_calls` did."""
    text: str                       #: the rewritten text (the same object when nothing changed)
    reduced: int = 0                #: stored calls replaced by their number
    unresolved: tuple[str, ...] = ()  #: tiers a stored call named that the table does not define (left as they were)


def _tier_equation(tokens: list[tuple[str, int, int]]) -> Optional[tuple[str, str]]:
    """('RCT_Human', '1.0') for the tokens of `(= (evidence-confidence RCT_Human) 1.0)`, else None."""
    texts = [t for t, _, _ in tokens]
    if (len(texts) == 8 and texts[:4] == ["(", "=", "(", "evidence-confidence"] and texts[5] == ")"
            and texts[7] == ")" and _NUMBER_TOKEN_RE.fullmatch(texts[6])
            and texts[4] not in ("(", ")") and texts[4][0] not in '"$'):
        return texts[4], texts[6]
    return None


def read_tier_table(kb_text: str) -> dict[str, str]:
    """{tier: number as written} from the `(= (evidence-confidence <Tier>) <number>)` equations of `kb_text`.

    The number stays text, so the substituted fact parses to the very float the equation holds. A tier
    defined twice with different numbers is ambiguous (the engine would answer with both) and is left out.
    """
    if "evidence-confidence" not in kb_text:
        return {}
    table: dict[str, str] = {}
    ambiguous: set[str] = set()
    for s, e in split_top_level_forms(kb_text):
        form = kb_text[s:e]
        if form[0] == "!" or "evidence-confidence" not in form:
            continue
        eq = _tier_equation(_form_tokens(form))
        if eq is None:
            continue
        tier, number = eq
        if tier in table and float(table[tier]) != float(number):
            ambiguous.add(tier)
        table.setdefault(tier, number)
    for tier in ambiguous:
        del table[tier]
    return table


def reduce_stored_calls(kb_text: str, tiers: Optional[Mapping[str, str]] = None) -> Reduction:
    """Replace `(evidence-confidence <Tier>)` stored inside a FACT by the tier's number.

    PeTTa does not evaluate a call stored in a fact ("Match does not automatically interpret the returned
    pattern as code"), so `(stv 0.7 (evidence-confidence RCT_Human))` would reach `combine-tv` as a list. The
    number comes from the `(= (evidence-confidence <Tier>) <number>)` equations in `kb_text` itself (plus
    `tiers`, for text added to a KB whose table was read earlier), so the tier table stays the single source
    of truth. A call whose tier is unknown or ambiguous is left alone and reported in `unresolved`.

    Only facts are rewritten. Rule bodies and constants `(= ...)`, declarations `(: ...)` and `!` expressions
    evaluate normally and are left untouched, as are comments and strings. Text without a stored call comes
    back unchanged (the same object).
    """
    if "evidence-confidence" not in kb_text:
        return Reduction(kb_text)
    table = dict(tiers or {})
    table.update(read_tier_table(kb_text))
    pieces: list[str] = []
    last = 0
    reduced = 0
    unresolved: list[str] = []
    for s, e in split_top_level_forms(kb_text):
        form = kb_text[s:e]
        if form[0] == "!" or "evidence-confidence" not in form:
            continue
        tokens = _form_tokens(form)
        head = tokens[1][0] if len(tokens) > 1 else ""
        if head in ("=", ":"):
            continue
        edits: list[tuple[int, int, str]] = []
        for i in range(len(tokens) - 3):
            if (tokens[i][0] == "(" and tokens[i + 1][0] == "evidence-confidence"
                    and tokens[i + 3][0] == ")" and tokens[i + 2][0] not in ("(", ")")):
                tier = tokens[i + 2][0]
                number = table.get(tier)
                if number is None:
                    if tier[0] != "$":          # a variable is no tier the table could ever name
                        unresolved.append(tier)
                    continue
                edits.append((tokens[i][1], tokens[i + 3][2], number))
        if not edits:
            continue
        rebuilt: list[str] = []
        at = 0
        for a, b, number in edits:
            rebuilt.append(form[at:a])
            rebuilt.append(number)
            at = b
        rebuilt.append(form[at:])
        pieces.append(kb_text[last:s])
        pieces.append("".join(rebuilt))
        last = e
        reduced += len(edits)
    if not reduced:
        return Reduction(kb_text, 0, tuple(dict.fromkeys(unresolved)))
    pieces.append(kb_text[last:])
    return Reduction("".join(pieces), reduced, tuple(dict.fromkeys(unresolved)))


# ── what the PeTTa runtime does not run ───────────────────────────────────────

#: Forms refused under PLN_RUNTIME=petta. `add-atom`, `remove-atom` and a `(= ...)` definition change the
#: process-global space (`remove-atom` retracts every identical copy; a runtime-added `=` becomes a global
#: clause); `assert` and `test` call halt(1); the rest reach the host or load code. `cached-import!` is
#: `import!`'s cached variant. `callPredicate`/`assertaPredicate`/`assertzPredicate`/`retractPredicate`/
#: `import_prolog_function`/`call` are PeTTa's own "call any Prolog predicate by name" doors (documented in
#: metta.pl's "Prolog interop" section) — the same shape of escape as `py-call`, just without the name.
#: This text scan is a friendly, fast 422 for the literal spelling; it is NOT the security boundary for any
#: of these forms — `_install_petta_sandbox` is (a symbol built at runtime by `atom_concat` never appears in
#: the text this function reads, so it cannot be caught here; see that function's docstring).
PETTA_REFUSED_FORMS: tuple[str, ...] = (
    "add-atom", "remove-atom", "=", "assert", "test", "halt", "import!", "git-import!", "cached-import!",
    "include", "py-call", "py-atom", "py-dot", "shell", "load_metta_file", "load_metta_file_cached",
    "process_metta_string", "new-space", "bind!", "call", "callPredicate", "assertaPredicate",
    "assertzPredicate", "retractPredicate", "import_prolog_function",
)
_REFUSED_ANYWHERE = frozenset(PETTA_REFUSED_FORMS) - {"="}


def petta_refused_forms(text: str) -> list[str]:
    """The refused forms `text` uses, sorted, comments and strings aside.

    A name other than `=` is refused wherever it stands as a symbol, not only as a head: `(let $f add-atom
    ($f &self x))` reaches `add-atom` without ever writing it as a head. `=` is refused only as the head of a
    TOP-LEVEL form, which is a definition; inside a pattern `(match &self (= (f $x) $v) ...)` reads the
    equations and defines nothing (two taught few-shot examples do that).
    """
    found: set[str] = set()
    depth = 0
    previous = ""
    for m in _SCAN_RE.finditer(text):
        tok = m.group()
        first = tok[0]
        if first == ";" or first == '"':
            continue
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        else:
            if tok in _REFUSED_ANYWHERE:
                found.add(tok)
            elif tok == "=" and previous == "(" and depth == 1:
                found.add("=")
        previous = tok
    return sorted(found)


def petta_refused_forms_in_kb(text: str) -> list[str]:
    """Like `petta_refused_forms`, for text that may hold an ordinary KB: every form in
    `_REFUSED_ANYWHERE` (add-atom, remove-atom, assert, test, halt, import!, py-call, shell, ...) is
    refused wherever it appears, but a top-level `(= ...)` is NOT, because that is how an ordinary KB
    defines every one of its functions (every `.metta` file in this repo has many) and how a tier table's
    own equations read (`read_tier_table`) — flagging it here would refuse every legitimate load.
    A caller-supplied `=` REDEFINITION is `core.metta_validator.validate`'s job: it checks the caller's own
    query/`extra_atoms` text specifically, before it is concatenated into a `kb_text` that also holds the
    trusted ontology files, where `petta_refused_forms`'s top-level check still applies."""
    found: set[str] = set()
    for m in _SCAN_RE.finditer(text):
        tok = m.group()
        if tok[0] in ';"':
            continue
        if tok in _REFUSED_ANYWHERE:
            found.add(tok)
    return sorted(found)


def _refuse_if_forbidden(text: str, where: str, *, scan: Callable[[str], list[str]] = petta_refused_forms) -> None:
    """Raise `MettaRuntimeError(kind="refused")` if `scan(text)` finds a refused form.

    Applied to every expression and (with `scan=petta_refused_forms_in_kb`) to `kb_text` and a session's
    text: `kb_text` is not attacker-free (it carries the caller's own `extra_atoms`, `core/pln_runner.py`),
    and PeTTa executes any top-level `!` form found anywhere in it during the load, the same as an
    expression — this closes that for the literal spelling. The engine-level sandbox
    (`_install_petta_sandbox`) is what enforces the restriction against a dynamically constructed symbol,
    which no text scan can see."""
    refused = scan(text)
    if refused:
        raise MettaRuntimeError(
            f"{where} uses form(s) the PeTTa runtime does not run: {', '.join(refused)}",
            kind="refused", detail=", ".join(refused))


# ── the engine-level sandbox (closes what the text scan above cannot) ─────────

#: `add-atom`/`remove-atom` materialise every ordinary ground fact while an ordinary KB loads
#: (`filereader.pl`'s `process_form` calls them directly for every non-`!` top-level form), so they keep
#: their real behaviour there and are refused only while `petta_sandbox_on` is in effect: around evaluating
#: a `!` form, never around `engine.load` or a session's plain-fact text. See `_install_petta_sandbox`.
_SANDBOX_DELEGATING_PREDICATES: tuple[tuple[str, int], ...] = (
    ("add-atom", 3), ("remove-atom", 3),
)

#: No ordinary KB load or any taught few-shot query uses any of these (checked against every tracked
#: `.metta` file and `prompts/few_shot_examples.json`), so they are refused unconditionally, for the whole
#: life of the child — during `engine.load` too, which closes this for `kb_text` the same as for an
#: expression or a session's text. `shell`/`callPredicate`/`assertaPredicate`/`assertzPredicate`/
#: `retractPredicate`/`import_prolog_function` are PeTTa's generic "call any Prolog predicate by name"
#: doors (`metta.pl`'s "Prolog interop" section): reachable even when the MeTTa-level name the caller wrote
#: is not `py-call`/`shell` at all (`(callPredicate (Predicate (shell $Cmd 0)))` never writes the word
#: `shell` next to `call`), so they are refused here too, not only `py-call`/`shell` themselves.
_SANDBOX_ALWAYS_PREDICATES: tuple[tuple[str, int], ...] = (
    ("py-call", 2), ("py-call", 3),
    ("shell", 1), ("shell", 2), ("shell", 3),
    ("halt", 1),
    ("import!", 3), ("library-import!", 3), ("cached-import!", 3),
    ("bind!", 3),
    ("assert", 2), ("test", 3),
    ("callPredicate", 2), ("assertaPredicate", 2), ("assertzPredicate", 2), ("retractPredicate", 2),
    ("import_prolog_function", 2),
)

#: Prolog consulted once per child, right after `PeTTa()` has loaded `main.pl`/`helper.pl`. Defines the
#: sandbox flag and the two helpers `_install_petta_sandbox` drives; does not touch any predicate itself
#: (that happens per-entry below, so one failing entry never takes the whole setup down with it).
_SANDBOX_SETUP_SRC = r"""
:- nb_setval(petta_sandbox_blocking, false).
petta_sandbox_on  :- nb_setval(petta_sandbox_blocking, true).
petta_sandbox_off :- nb_setval(petta_sandbox_blocking, false).
petta_sandbox_blocked :- nb_current(petta_sandbox_blocking, true).

%Copies Name/Arity's existing clauses under a private name, then replaces Name/Arity with a dispatcher that
%throws while the sandbox flag is set and otherwise calls the private copy - the real predicate, unchanged.
petta_sandbox_guard_delegating(Name, Arity) :-
    functor(Head, Name, Arity),
    atom_concat('$petta_orig$', Name, OrigName),
    functor(OrigHead, OrigName, Arity),
    OrigHead =.. [OrigName|Args],
    Head =.. [Name|Args],
    findall(clause(OrigHead, Body), clause(Head, Body), Clauses),
    Clauses \== [],
    redefine_system_predicate(OrigName/Arity),
    forall(member(clause(OH, Body), Clauses), assertz((OH :- Body))),
    redefine_system_predicate(Name/Arity),
    abolish(Name/Arity),
    assertz((Head :- ( petta_sandbox_blocked
                       -> throw(error(permission_error(call, Name), petta_sandbox))
                        ; call(OrigHead) ))).

%Replaces Name/Arity with a predicate that always throws - no delegation, because none of these have any
%legitimate caller in this repo's knowledge bases (see _SANDBOX_ALWAYS_PREDICATES above).
petta_sandbox_guard_always(Name, Arity) :-
    functor(Head, Name, Arity),
    redefine_system_predicate(Name/Arity),
    ignore(catch(abolish(Name/Arity), _, true)),
    assertz((Head :- throw(error(permission_error(call, Name), petta_sandbox)))).
"""


def _install_petta_sandbox(janus: Any) -> None:
    """Close the hole the text scan above cannot: a symbol built at runtime (`atom_concat("add","-atom")`,
    bound to a variable, then applied as `($f &self (Hacked yes))`) never appears as the token `add-atom` in
    the program's TEXT, which is all `petta_refused_forms` can read — but PeTTa's own dispatcher
    (`translator.pl`'s `reduce/2`) treats a function's identity as an ordinary runtime value, so that
    symbol runs all the same. The restriction has to live inside the engine, not on the caller's text.

    Redefines the predicates in `_SANDBOX_DELEGATING_PREDICATES` and `_SANDBOX_ALWAYS_PREDICATES` directly,
    in Prolog, so the block holds no matter how the program's text spells or constructs the call that
    reaches them — in `kb_text`, in an expression, or inside a `PettaSession`, uniformly, independent of
    which Python code path concatenated the caller's bytes into which string. Called once per forked child,
    right after `PeTTa()` consults `main.pl`/`helper.pl` and before anything is loaded or run.

    Raises if any predicate cannot be found or redefined (the child then reports a `kind="start"` error and
    is discarded, by the same path as any other construction failure): refusing to run a program is safer
    than running one with this boundary silently missing, for instance after an upstream PeTTa update
    renames or re-arities one of these predicates.

    `halt/0` is the one predicate SWI itself will not let any program redefine (confirmed empirically
    against this PeTTa checkout: `redefine_system_predicate(halt/0)` followed by `asserta/1` still raises
    `permission_error(modify, static_procedure, halt/0)`), so it is not in either list above. Reaching it
    (`(call (halt))`) ends the child with exit code 0; `_Child._died()` already reports that as a
    `kind="crash"` result, the same bounded, already-handled outcome as `halt/1` before this module existed
    (a halt only ever ends the one process it runs in) — not a new failure mode and not a host escape.
    """
    janus.consult("petta_sandbox_setup", data=_SANDBOX_SETUP_SRC)
    for name, arity in _SANDBOX_DELEGATING_PREDICATES:
        if not janus.query_once("petta_sandbox_guard_delegating(N, A)", {"N": name, "A": arity}):
            raise RuntimeError(f"could not sandbox the PeTTa predicate '{name}'/{arity}")
    for name, arity in _SANDBOX_ALWAYS_PREDICATES:
        if not janus.query_once("petta_sandbox_guard_always(N, A)", {"N": name, "A": arity}):
            raise RuntimeError(f"could not sandbox the PeTTa predicate '{name}'/{arity}")


# ── forking ───────────────────────────────────────────────────────────────────

#: Serialises "create the pipe, fork, close the child's end" so that two threads forking at once cannot hand
#: each other's pipe ends to their children (a copy of a write end held by a stranger delays EOF).
_FORK_LOCK = threading.Lock()


def _describe_status(status: Optional[int]) -> str:
    if status is None or status < 0:
        return "an unknown status"
    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        return f"signal {sig} ({name})"
    if os.WIFEXITED(status):
        return f"exit code {os.WEXITSTATUS(status)}"
    return f"status {status}"


class _Child:
    """A forked worker process and the parent's end of its pipe. `close()` always reaps it."""

    def __init__(self, pid: int, conn: Connection) -> None:
        self.pid = pid
        self.conn = conn
        self.status: Optional[int] = None      # waitpid status once reaped; -1 when it was reaped elsewhere

    # -- the process
    def exited(self) -> bool:
        if self.status is not None:
            return True
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:               # reaped elsewhere (SIGCHLD ignored)
            self.status = -1
            return True
        if pid:
            self.status = status
            return True
        return False

    def kill(self) -> None:
        if self.status is None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def reap(self, timeout: float) -> bool:
        """Wait up to `timeout` s for the child to be gone and collect it. True when it is reaped."""
        end = time.monotonic() + timeout
        delay = 0.0005
        while not self.exited():
            if time.monotonic() >= end:
                return False
            time.sleep(delay)
            delay = min(delay * 2, 0.02)
        return True

    def close(self) -> None:
        """Kill if still running, reap, close the pipe. Safe to call twice, and from a `finally`."""
        if not self.exited():
            self.kill()
            self.reap(10.0)
        try:
            self.conn.close()
        except Exception:                       # pragma: no cover - closing must never mask the real error
            pass

    # -- the pipe
    def send(self, message: dict) -> None:
        self.conn.send_bytes(json.dumps(message).encode("utf-8"))

    def wait_message(self, deadline: Optional[float]) -> Optional[dict]:
        """The child's next message; None at the deadline; MettaRuntimeError if the child died first."""
        while True:
            if deadline is None:
                slice_s = 0.25
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                slice_s = min(remaining, 0.25)
            if self.conn.poll(slice_s):
                try:
                    return json.loads(self.conn.recv_bytes().decode("utf-8"))
                except (EOFError, OSError):
                    raise self._died() from None
            if self.exited() and not self.conn.poll(0):
                raise self._died()

    def _died(self) -> MettaRuntimeError:
        self.reap(2.0)
        how = _describe_status(self.status)
        hint = ""
        if self.status is not None and self.status >= 0 and os.WIFEXITED(self.status) and os.WEXITSTATUS(self.status) == 1:
            hint = " (PeTTa's `test` and `assert` end the process with code 1)"
        return MettaRuntimeError(f"the PeTTa process ended with {how} before it answered{hint}", kind="crash")


def _fork_child(target: Callable[..., None], *args: Any) -> _Child:
    """Fork a child that runs `target(conn, *args)` and never returns into the caller's code."""
    with _FORK_LOCK:
        parent_conn, child_conn = Pipe(duplex=True)
        try:
            with warnings.catch_warnings():
                # Python 3.12 warns about fork() in a multi-threaded process. The executor's workers are
                # single-threaded; inline mode and tests may not be, and the warning would only be noise.
                warnings.simplefilter("ignore", DeprecationWarning)
                pid = os.fork()
        except BaseException:
            parent_conn.close()
            child_conn.close()
            raise
        if pid == 0:                            # the child
            code = 70
            try:
                parent_conn.close()
                target(child_conn, *args)
                code = 0
            except BaseException:               # noqa: BLE001 - nothing may escape into the caller's stack
                pass
            finally:
                os._exit(code)
        child_conn.close()
    return _Child(pid, parent_conn)


def _child_prepare(deadline_s: Optional[float], *, parent_pid: Optional[int]) -> None:
    """First thing in a forked PeTTa child: cut it off from the parent's terminal and bound it."""
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    if devnull > 2:
        os.close(devnull)
    sys.stdout = sys.stderr = open(os.devnull, "w")
    sys.stdin = open(os.devnull, "r")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM, signal.SIGXCPU):
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError):           # pragma: no cover
            pass
    _set_cpu_backstop(deadline_s)
    if parent_pid is not None and sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)          # PR_SET_PDEATHSIG
            if os.getppid() != parent_pid:                        # the parent died before the call took effect
                os._exit(71)
        except Exception:                       # pragma: no cover
            pass


def _set_cpu_backstop(deadline_s: Optional[float]) -> None:
    """Kernel-enforced: SIGXCPU (default action: terminate) once the child has used `deadline_s` +
    BACKSTOP_GRACE_S CPU seconds from now. The parent's SIGKILL at the deadline normally comes first; this is
    for the child whose parent is gone."""
    try:
        import resource
        if deadline_s is None:
            limit = resource.RLIM_INFINITY
        else:
            limit = int(math.ceil(time.process_time() + deadline_s + BACKSTOP_GRACE_S))
        resource.setrlimit(resource.RLIMIT_CPU, (limit, resource.RLIM_INFINITY))
    except Exception:                           # pragma: no cover - a platform without RLIMIT_CPU
        pass


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(value / 1e6 if sys.platform == "darwin" else value / 1024.0, 1)
    except Exception:                           # pragma: no cover
        return None


def _clip(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit // 2] + " ... " + text[-limit // 2:]


# ── the real PeTTa engine (constructed in the child only) ─────────────────────

class _PettaEngine:
    """PeTTa through janus. Constructed in the forked child: this is the only code that imports them."""

    def __init__(self, petta_home: str, stack_limit_mb: int) -> None:
        sys.dont_write_bytecode = True                       # never write .pyc into the PeTTa checkout
        pydir = str(Path(petta_home) / "python")
        if pydir not in sys.path:
            sys.path.insert(0, pydir)
        import janus_swi as janus                            # starts the embedded SWI-Prolog
        from petta import PeTTa
        import petta as petta_module
        where = Path(getattr(petta_module, "__file__", "") or "").resolve()
        if Path(petta_home).resolve() not in where.parents:
            raise ImportError(f"imported a `petta` that is not the one under {petta_home}: {where}")
        PeTTa(verbose=False, petta_path=str(petta_home))     # consults main.pl and helper.pl
        _install_petta_sandbox(janus)                        # before anything is loaded or run
        janus.query_once(f"set_prolog_flag(stack_limit, {int(stack_limit_mb) * 1024 * 1024})")
        self._janus = janus

    def _helper(self, text: str, *, sandboxed: bool) -> Optional[list[str]]:
        """PeTTa's own `_run_helper`, except that a goal that FAILS is returned as None instead of [].

        `sandboxed` toggles the engine-level guard (`_install_petta_sandbox`) around this one call: on
        while evaluating a `!` form, off while loading a KB or adding a session's plain facts, so
        `add-atom`/`remove-atom` keep materialising ordinary ground facts there."""
        if sandboxed:
            self._janus.query_once("petta_sandbox_on")
        try:
            answer = self._janus.query_once(
                "run_metta_helper(Verbose, HelperName, Argument, Results)",
                {"Verbose": "false", "HelperName": "process_metta_string", "Argument": text},
            )
        finally:
            if sandboxed:
                self._janus.query_once("petta_sandbox_off")
        if answer is None:
            return None
        return [str(x) for x in answer.get("Results", [])]

    def load(self, kb_text: str) -> None:
        if self._helper(kb_text, sandboxed=False) is None:
            raise RuntimeError("the load goal failed without raising an error")

    def run(self, text: str, *, sandboxed: bool = True) -> list[str]:
        out = self._helper(text, sandboxed=sandboxed)
        return [] if out is None else out


EngineFactory = Callable[[str, int], Any]


# ── the child programs ────────────────────────────────────────────────────────

def _send(conn: Connection, message: dict) -> None:
    conn.send_bytes(json.dumps(message).encode("utf-8"))


def _error_message(stage: str, exc: BaseException, index: Optional[int] = None) -> dict:
    return {"kind": "error", "stage": stage, "index": index,
            "type": type(exc).__name__, "message": _clip(f"{exc}")}


def _program_child(conn: Connection, factory: EngineFactory, petta_home: str, stack_limit_mb: int,
                   kb_text: str, expressions: list[str], deadline_s: Optional[float],
                   parent_pid: int) -> None:
    """The forked child of `PettaRuntime.run_program`: one load, one call per expression, one message."""
    try:
        _child_prepare(deadline_s, parent_pid=parent_pid)
    except BaseException as exc:                # noqa: BLE001
        _send(conn, _error_message("start", exc))
        return
    t0 = time.perf_counter()
    try:
        engine = factory(petta_home, stack_limit_mb)
    except BaseException as exc:                # noqa: BLE001
        _send(conn, _error_message("start", exc))
        return
    t1 = time.perf_counter()
    if kb_text.strip():
        try:
            engine.load(kb_text)
        except BaseException as exc:            # noqa: BLE001 - any exception during a load is a failed load
            _send(conn, _error_message("load", exc))
            return
    t2 = time.perf_counter()
    groups: list[list[str]] = []
    for index, expression in enumerate(expressions):
        try:
            groups.append([str(x) for x in engine.run(expression)])
        except BaseException as exc:            # noqa: BLE001
            _send(conn, _error_message("expression", exc, index))
            return
    t3 = time.perf_counter()
    _send(conn, {"kind": "result", "groups": groups,
                 "stats": {"engine_s": round(t1 - t0, 4), "load_s": round(t2 - t1, 4),
                           "queries_s": round(t3 - t2, 4), "peak_rss_mb": _peak_rss_mb()}})


def _run_text(engine: Any, text: str, tiers: Mapping[str, str]) -> tuple[list[list[str]], int]:
    """Evaluate `text` form by form: one group per `!` form; consecutive facts go in as one chunk, with their
    stored calls reduced. Returns (groups, number of calls reduced)."""
    groups: list[list[str]] = []
    reduced = 0
    pending: Optional[list[int]] = None          # [start, end] of the facts waiting to be added

    def flush() -> None:
        nonlocal pending, reduced
        if pending is not None:
            reduction = reduce_stored_calls(text[pending[0]:pending[1]], tiers)
            reduced += reduction.reduced
            engine.run(reduction.text, sandboxed=False)   # plain facts: add-atom must keep its real behaviour
            pending = None

    for s, e in split_top_level_forms(text):
        if text[s] == "!":
            flush()
            groups.append([str(x) for x in engine.run(text[s:e])])   # sandboxed: default True
        elif pending is None:
            pending = [s, e]
        else:
            pending[1] = e
    flush()
    return groups, reduced


def _session_child(conn: Connection, factory: EngineFactory, petta_home: str, stack_limit_mb: int,
                   kb_text: str, tiers: dict[str, str], deadline_s: Optional[float]) -> None:
    """The forked child of a PettaSession: load once, then answer requests until told to stop or orphaned."""
    try:
        _child_prepare(deadline_s, parent_pid=None)
        engine = factory(petta_home, stack_limit_mb)
    except BaseException as exc:                # noqa: BLE001
        _send(conn, _error_message("start", exc))
        return
    if kb_text.strip():
        try:
            engine.load(kb_text)
        except BaseException as exc:            # noqa: BLE001 - any exception during a load is a failed load
            _send(conn, _error_message("load", exc))
            return
    _send(conn, {"kind": "ready", "stats": {"peak_rss_mb": _peak_rss_mb()}})
    while True:
        try:
            request = json.loads(conn.recv_bytes().decode("utf-8"))
        except (EOFError, OSError):             # the parent is gone
            return
        if request.get("op") == "close":
            return
        _set_cpu_backstop(request.get("deadline_s"))
        try:
            groups, reduced = _run_text(engine, request["text"], tiers)
        except BaseException as exc:            # noqa: BLE001 - the session survives an expression that raised
            _send(conn, _error_message("expression", exc))
            continue
        _send(conn, {"kind": "result", "groups": groups, "reduced": reduced,
                     "stats": {"peak_rss_mb": _peak_rss_mb()}})


# ── runtimes ──────────────────────────────────────────────────────────────────

def _bang(expression: str) -> str:
    """`pln_runner._normalize_query`'s rule: an expression is evaluated, so it carries a `!`."""
    expression = expression.strip()
    return expression if expression.startswith("!") else "!" + expression


class _HyperonSession:
    def __init__(self, kb_text: str) -> None:
        try:
            from hyperon import MeTTa  # type: ignore
            self._metta = MeTTa()
            if kb_text.strip():
                self._metta.run(kb_text)
        except Exception as exc:
            raise MettaRuntimeError(str(exc), kind="runtime", detail=str(exc)) from exc
        self.last_raw: list[list[str]] = []

    def run(self, text: str, deadline_s: Optional[float] = None) -> list[list[str]]:
        if self._metta is None:
            raise MettaRuntimeError("the session is closed", kind="session")
        try:
            groups = [[str(atom) for atom in group] for group in self._metta.run(text)]
        except Exception as exc:
            raise MettaRuntimeError(str(exc), kind="runtime", detail=str(exc)) from exc
        self.last_raw = groups
        return groups

    def close(self) -> None:
        self._metta = None

    def __enter__(self) -> "_HyperonSession":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class HyperonRuntime:
    """hyperon-experimental 0.2.10, in process. The behaviour `pln_runner._hyperon_run` always had."""

    name = "hyperon"
    enforces_deadline = False

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("hyperon") is not None
        except Exception:
            return False

    def run_program(self, kb_text: str, expressions: Sequence[str],
                    deadline_s: Optional[float] = None) -> list[list[str]]:
        """`deadline_s` is ignored: a GIL-holding Rust call cannot be preempted in process. The executor
        enforces the request deadline by killing the worker."""
        try:
            from hyperon import MeTTa  # type: ignore
            normalized = "\n".join(_bang(e) for e in expressions)
            metta = MeTTa()
            if kb_text.strip():
                metta.run(kb_text)                   # populate &self; result ignored
            raw = metta.run(normalized)              # every group is a query result
            return [[str(atom) for atom in group] for group in raw]
        except Exception as exc:
            # The text is what pln_runner always reported: str(the exception), nothing added.
            raise MettaRuntimeError(str(exc), kind="runtime", detail=str(exc)) from exc

    def session(self, kb_text: str) -> Session:
        return _HyperonSession(kb_text)


class PettaSession:
    """One long-lived PeTTa child holding one loaded KB. Create it with `PettaRuntime.session`."""

    def __init__(self, runtime: "PettaRuntime", kb_text: str, deadline_s: Optional[float] = None) -> None:
        self._runtime = runtime
        self._child: Optional[_Child] = None
        self._closed = False
        self.last_raw: list[list[str]] = []
        _refuse_if_forbidden(kb_text, "the knowledge base text", scan=petta_refused_forms_in_kb)
        reduction = reduce_stored_calls(kb_text)
        self.reduced_calls = reduction.reduced
        self.unresolved_calls = reduction.unresolved
        self._tiers = read_tier_table(kb_text)
        budget = runtime._budget(deadline_s)
        self._child = runtime._fork(_session_child, runtime._engine_factory, str(runtime.petta_home),
                                    runtime.stack_limit_mb, reduction.text, self._tiers, budget)
        try:
            message = self._child.wait_message(None if budget is None else time.monotonic() + budget)
            if message is None:
                raise MettaRuntimeError(
                    f"the PeTTa session did not start within {budget:g} s and was killed", kind="timeout")
            if message.get("kind") != "ready":
                raise runtime._error_from(message)
        except BaseException:
            self.close()
            raise

    def run(self, text: str, deadline_s: Optional[float] = None) -> list[list[str]]:
        child = self._child
        if self._closed or child is None:
            raise MettaRuntimeError("the session is closed", kind="session")
        _refuse_if_forbidden(text, "the session text", scan=petta_refused_forms_in_kb)
        budget = self._runtime._budget(deadline_s)
        try:
            child.send({"op": "run", "text": text, "deadline_s": budget})
            message = child.wait_message(None if budget is None else time.monotonic() + budget)
        except (BrokenPipeError, OSError):
            self.close()
            raise MettaRuntimeError("the PeTTa session process is gone", kind="session") from None
        except MettaRuntimeError:
            self.close()
            raise
        if message is None:
            self.close()
            raise MettaRuntimeError(
                f"the PeTTa session exceeded the {budget:g} s deadline and was killed", kind="timeout")
        if message.get("kind") != "result":
            raise self._runtime._error_from(message)
        self.reduced_calls += int(message.get("reduced", 0))
        self.last_raw = message["groups"]
        return [[normalise_result(s) for s in group] for group in self.last_raw]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        child, self._child = self._child, None
        if child is None:
            return
        try:
            if not child.exited():
                try:
                    child.send({"op": "close"})
                except Exception:
                    pass
                if not child.reap(0.5):
                    child.kill()
        finally:
            child.close()

    def __enter__(self) -> "PettaSession":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:                  # a session that was never closed must not leak its child
        try:
            self.close()
        except Exception:                       # pragma: no cover
            pass


class PettaRuntime:
    """PeTTa (MeTTa compiled to SWI-Prolog) through janus, in a forked child per program.

    Diagnostics of the most recent `run_program`, readable by tests and tools:
      reduced_calls   how many stored `(evidence-confidence T)` calls the KB text had reduced
      unresolved_calls  tiers a stored call named that no equation defines (those calls were left alone)
      last_raw        the raw result strings, before `normalise_result`
      last_run        {"total_s", "fork_s", "engine_s" (import + consult), "load_s", "queries_s", "peak_rss_mb"}
      last_pid        the child's pid (gone by the time `run_program` returns)
    """

    name = "petta"
    enforces_deadline = True

    def __init__(self, petta_home: Optional[os.PathLike | str] = None, stack_limit_mb: Optional[int] = None,
                 engine_factory: Optional[EngineFactory] = None) -> None:
        import config
        self.petta_home = Path(petta_home if petta_home is not None else config.PETTA_HOME)
        self.stack_limit_mb = int(stack_limit_mb if stack_limit_mb is not None else config.PETTA_STACK_LIMIT_MB)
        #: how the child builds its engine; tests inject a double (the child is forked, so a closure is fine)
        self._engine_factory: EngineFactory = engine_factory or _PettaEngine
        self.reduced_calls = 0
        self.unresolved_calls: tuple[str, ...] = ()
        self.last_raw: list[list[str]] = []
        self.last_run: dict = {}
        self.last_pid: Optional[int] = None

    # -- availability (imports nothing)
    def available(self) -> bool:
        return _petta_available(self.petta_home)

    # -- helpers
    def _budget(self, deadline_s: Optional[float]) -> Optional[float]:
        """Seconds, or None for no deadline. None means the configured default; 0 or less means none."""
        if deadline_s is None:
            import config
            deadline_s = config.PLN_QUERY_TIMEOUT_SECONDS
        return float(deadline_s) if deadline_s and deadline_s > 0 else None

    def _fork(self, target: Callable[..., None], *args: Any) -> _Child:
        try:
            return _fork_child(target, *args)
        except OSError as exc:
            raise MettaRuntimeError(f"could not start a PeTTa process: {exc}", kind="unavailable",
                                    detail=str(exc)) from exc

    def _error_from(self, message: dict) -> MettaRuntimeError:
        stage, text = message.get("stage"), message.get("message", "")
        kind_of = f"{message.get('type', 'Error')}: {text}"
        if stage == "start":
            # The server's own filesystem path (and, in it, the owner's home directory and username) is
            # not for an HTTP caller: an `os.fork()` refused under load (too many processes, out of memory)
            # raises this same kind="unavailable" error, and a caller can provoke that just by sending
            # enough concurrent requests. Log the path server-side; the caller gets the exception type only
            # (which `detail`, kept below, also carries — `str(exc)` is the only part any caller ever sees).
            _LOG.warning("PeTTa could not be started (PETTA_HOME=%s): %s", self.petta_home, kind_of)
            return MettaRuntimeError(f"PeTTa could not be started ({message.get('type', 'Error')})",
                                     kind="unavailable", detail=text)
        if stage == "load":
            return MettaRuntimeError(f"PeTTa could not load the knowledge base: {kind_of}", kind="load", detail=text)
        index = message.get("index")
        where = f"expression {index + 1}" if isinstance(index, int) else "the program"
        return MettaRuntimeError(f"PeTTa raised while evaluating {where}: {kind_of}", kind="expression", detail=text)

    # -- the seam
    def run_program(self, kb_text: str, expressions: Sequence[str],
                    deadline_s: Optional[float] = None) -> list[list[str]]:
        exprs = [_bang(e) for e in expressions]
        for index, expr in enumerate(exprs):
            _refuse_if_forbidden(expr, f"expression {index + 1}")
        _refuse_if_forbidden(kb_text, "the knowledge base text", scan=petta_refused_forms_in_kb)
        reduction = reduce_stored_calls(kb_text)
        self.reduced_calls = reduction.reduced
        self.unresolved_calls = reduction.unresolved
        if not exprs:
            self.last_raw, self.last_run, self.last_pid = [], {}, None
            return []
        budget = self._budget(deadline_s)
        started = time.monotonic()
        deadline = None if budget is None else started + budget
        child = self._fork(_program_child, self._engine_factory, str(self.petta_home), self.stack_limit_mb,
                           reduction.text, exprs, budget, os.getpid())
        self.last_pid = child.pid
        forked = time.monotonic()
        try:
            message = child.wait_message(deadline)
            if message is None:
                child.kill()
                raise MettaRuntimeError(
                    f"PeTTa exceeded the {budget:g} s deadline and was killed", kind="timeout")
            if message.get("kind") != "result":
                raise self._error_from(message)
            child.reap(_EXIT_WAIT_S)
        finally:
            child.close()
        raw = message["groups"]
        self.last_raw = raw
        self.last_run = {"total_s": round(time.monotonic() - started, 4), "fork_s": round(forked - started, 4),
                         **message.get("stats", {})}
        return [[normalise_result(s) for s in group] for group in raw]

    def session(self, kb_text: str, deadline_s: Optional[float] = None) -> PettaSession:
        return PettaSession(self, kb_text, deadline_s)


# ── availability without importing anything ───────────────────────────────────

def _petta_available(petta_home: os.PathLike | str) -> bool:
    """Is PeTTa runnable here? Looks, never imports: `janus_swi` is found with `find_spec` and the checkout
    by its files, so the calling process never loads SWI-Prolog (forking after that is unsafe)."""
    try:
        if not hasattr(os, "fork"):
            return False
        home = Path(petta_home)
        if not ((home / "src" / "main.pl").is_file() and (home / "python" / "helper.pl").is_file()
                and (home / "python" / "petta" / "__init__.py").is_file()):
            return False
        return importlib.util.find_spec("janus_swi") is not None
    except Exception:
        return False


def _hyperon_available() -> bool:
    try:
        return importlib.util.find_spec("hyperon") is not None
    except Exception:
        return False


def selected_runtime_name(name: Optional[str] = None) -> str:
    """'hyperon' or 'petta': PLN_RUNTIME (or `name`) with `auto` resolved. `auto` is hyperon when it is
    importable, else petta; with neither importable it is hyperon, and the app then runs in stub mode."""
    if name is None:
        import config
        name = config.PLN_RUNTIME
    name = (name or "hyperon").strip().lower()
    if name not in RUNTIME_NAMES:
        raise ValueError(f"PLN_RUNTIME must be one of {', '.join(RUNTIME_NAMES)}; got {name!r}")
    if name == "auto":
        return "hyperon" if _hyperon_available() or not _petta_available(_configured_petta_home()) else "petta"
    return name


def _configured_petta_home() -> Path:
    import config
    return Path(config.PETTA_HOME)


def runtime_available_for(name: str, petta_home: Optional[os.PathLike | str] = None) -> bool:
    """Is the runtime `name` (hyperon | petta | auto) runnable here? What config.PLN_RUNTIME_AVAILABLE is
    made of. Imports neither engine."""
    name = (name or "hyperon").strip().lower()
    if name == "hyperon":
        return _hyperon_available()
    if name == "petta":
        return _petta_available(petta_home if petta_home is not None else _configured_petta_home())
    if name == "auto":
        return _hyperon_available() or _petta_available(
            petta_home if petta_home is not None else _configured_petta_home())
    raise ValueError(f"PLN_RUNTIME must be one of {', '.join(RUNTIME_NAMES)}; got {name!r}")


# ── selection ─────────────────────────────────────────────────────────────────

_RUNTIMES: dict[str, Runtime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_runtime(name: Optional[str] = None) -> Runtime:
    """The runtime PLN_RUNTIME selects (or `name`: hyperon | petta | auto), one instance per process."""
    resolved = selected_runtime_name(name)
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(resolved)
        if runtime is None:
            runtime = HyperonRuntime() if resolved == "hyperon" else PettaRuntime()
            _RUNTIMES[resolved] = runtime
        return runtime


def reset_runtime_cache() -> None:
    """Forget the cached instances (tests that change PETTA_HOME or the stack limit)."""
    with _RUNTIMES_LOCK:
        _RUNTIMES.clear()
