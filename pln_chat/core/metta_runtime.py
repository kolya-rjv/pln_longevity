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
* Memory is bounded by `PETTA_MEMORY_LIMIT_MB` (default 512), in two ways, and `PETTA_STACK_LIMIT_MB` is NOT
  one of them: it caps SWI-Prolog's own stacks, and a string or an atom lives on the heap, outside them. A
  ~900-byte program of `let*`/`atom_concat` doubles a string 22 times and takes 1 GB in half a second under a
  64 MB stack limit, with every builtin it uses allowed. (1) A watchdog thread in the child samples its
  peak resident set every 5 ms and ends the process (exit code 73, which the parent reports as
  `kind="memory"`) once it has grown by more than the limit since `_child_prepare`. It runs on every
  platform, needs no parent, and binds an orphaned child too; janus releases the GIL while Prolog runs, which
  is what lets a thread see a Prolog program growing (checked). The child calls `gc.freeze()` first: on macOS a
  fork maps the server's heap lazily, and one garbage collection in the child would otherwise count all of it as
  growth. It is a sample, not a wall: a single call that
  allocates more than the limit between two samples (about 5 ms) is not stopped by it. (2) On Linux the
  kernel is asked for the wall: `RLIMIT_AS` at the address space the child had when it was prepared, plus the
  limit, plus 1 GB for what the libraries and the allocator reserve and never use. macOS refuses a
  finite `RLIMIT_AS` (EINVAL, checked), so there only the watchdog applies. Neither replaces a memory limit
  on the whole worker pool (a cgroup, `docker --memory`, systemd `MemoryMax=`): N children each at the limit
  are still N times the limit.
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
   a good load of a KB with no `!` forms). No malformed KB tried against this PeTTa commit (13 probed, from
   unbalanced parentheses to a bad `superpose`) fails this way — every one raises — but a goal failing silently
   and being read as "loaded cleanly" is the worse failure mode, and the branch is tested against a double of
   janus. (janus answers a failed goal with `{"truth": False}`, a non-empty dict: every check of an answer
   here reads `["truth"]`, never the dict's own truthiness.)
4. A query that fails without raising comes back as `[]`: an empty answer, as on hyperon.
5. Facts and solutions come back in assertion order where hyperon often returns the reverse.
6. A program can make PeTTa call a Prolog predicate by a name the program chooses (`maplist getenv`,
   `(match read_file_to_string ...)`, `call`, `translatePredicate`, `callPredicate`, ...), change the process-global
   space (`add-atom`; `remove-atom` retracts every identical copy), halt the process (`assert`, `test`) or reach
   the host (`import!`, `py-call`, `shell`). `_install_petta_sandbox` closes each of those routes inside the engine,
   once per forked child, whatever the text spells and however it arrived; `_child_prepare` takes from the child
   what a program has no use for (the server's environment, descriptors and working directory). The literal
   spelling is also refused up front by `core.metta_validator` and by `PettaRuntime`/`PettaSession`
   (`petta_refused_forms`/`petta_refused_forms_in_kb`), reading symbols the way PeTTa's own reader does: a fast,
   friendly 422 and nothing more — a name PeTTa builds at run time (`atom_concat`) never appears in any of that
   text. The table of routes, and the one place each is closed, is under "the engine-level sandbox" below.

The chainer hook (PLN_CHAINER=pln)
----------------------------------
`run_program(..., after_load=step)` lets ONE caller run a step inside the PeTTa child, after the KB is loaded and
before the space is locked. `core.pln_chainer` is that caller: its step loads the vendored PLN library, derives the
`infer`/`explain` answers for every source in the KB with patham9's `PLN.Derive`, and adds them to the space as
facts (the space accepts no `add-atom` once locked, and consumers bind the source of an `infer` while they
evaluate, so the answers have to be there before the first query). The step receives the engine (`.run(text)`,
`.load(text)`), runs inside the sandbox like any program, and may return a small JSON-able summary, which comes
back as `last_run["chain"]`. If it raises, the request fails with `MettaRuntimeError(kind="chain")`. Nothing
else uses the hook; without it (every request under PLN_CHAINER=legacy) this module behaves exactly as before.
HyperonRuntime and sessions have no hook: a request that asks for one on hyperon is refused.

Sessions
--------
`runtime.session(kb_text)` is for tests and tools, not for production requests: a stateful interpreter that
loads the KB once and then answers `run(text)` calls, like a hyperon `MeTTa` object. Text without a `!`
adds facts to that interpreter's space. Under petta a session is one long-lived forked child; `close()`
kills and reaps it.

Nothing in this module imports hyperon, janus or petta at import time.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import sys
import tempfile
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
    "PETTA_KNOWN_FUN_NAMES", "PETTA_ALLOWED_FUN_NAMES", "PETTA_ENGINE_INTERNAL_FUN_NAMES",
    "petta_strip_comments", "petta_tokens", "petta_symbols",
    "resolve_memory_limit_mb", "DEFAULT_MEMORY_LIMIT_MB", "MIN_MEMORY_LIMIT_MB",
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
#: How much a PeTTa child's peak resident set may grow after it is prepared, in MB (`PETTA_MEMORY_LIMIT_MB`).
#: Measured: every PeTTa test in the suite, the PLN chainer lane included, passes with 128; all 33 `.metta` files
#: of the repository loaded together peak at 116 MB in all. This is four times the larger figure. A deployment
#: that loads much bigger knowledge bases (ETL output) raises it: a KB costs about 0.2 MB per KB of text.
DEFAULT_MEMORY_LIMIT_MB = 512
#: The smallest limit that is honoured: a child imports janus and consults PeTTa (30-60 MB) before it runs
#: anything, so a smaller number would stop every program. A smaller value is raised to this one.
MIN_MEMORY_LIMIT_MB = 128
#: The child's exit code when its own watchdog ended it for using too much memory (70 and 71 are taken).
_EXIT_MEMORY = 73
#: How often the child looks at its own memory.
_MEMORY_POLL_S = 0.005
#: On Linux the address-space limit is the child's size when prepared + the memory limit + this. The libraries a
#: child maps after it is prepared (SWI-Prolog, PeTTa's) and the 64 MB glibc reserves per allocation arena are
#: address space nobody uses; the limit is a backstop behind the watchdog and must not stop an honest program,
#: so it is generous.
_ADDRESS_SPACE_SLACK_MB = 1024


def resolve_memory_limit_mb(value: Optional[int] = None) -> int:
    """The memory limit of a PeTTa child, in MB; 0 means none.

    `value` (a constructor argument) wins; otherwise `PETTA_MEMORY_LIMIT_MB` from the server's environment,
    otherwise `DEFAULT_MEMORY_LIMIT_MB`. A value below `MIN_MEMORY_LIMIT_MB` is raised to it, 0 switches the limit
    off (for a deployment that bounds memory some other way), and an environment value that is not a whole
    number, or is negative, is ignored with a warning: an unreadable setting falls back to the protected
    default, never to no protection. A negative argument is a programming error and raises."""
    if value is None:
        raw = os.environ.get("PETTA_MEMORY_LIMIT_MB", "").strip()
        if not raw:
            return DEFAULT_MEMORY_LIMIT_MB
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value < 0:
            _LOG.warning("PETTA_MEMORY_LIMIT_MB=%r is not a whole number of megabytes, 0 or more; using %d",
                         raw, DEFAULT_MEMORY_LIMIT_MB)
            return DEFAULT_MEMORY_LIMIT_MB
    value = int(value)
    if value < 0:
        raise ValueError(f"the memory limit of a PeTTa child is 0 (none) or more megabytes; got {value}")
    return 0 if value == 0 else max(MIN_MEMORY_LIMIT_MB, value)


class MettaRuntimeError(RuntimeError):
    """A MeTTa program could not be run to an answer.

    `str(error)` is safe to show a caller: it says what happened and keeps the underlying engine text
    (clipped). `kind` is one of: `unavailable` (the runtime cannot start here), `load` (the knowledge base
    did not load), `expression` (an expression raised), `timeout` (the deadline killed the run), `crash`
    (the interpreter process died without an answer), `memory` (the process grew past
    `PETTA_MEMORY_LIMIT_MB` and was stopped), `refused` (the program uses a form the PeTTa runtime
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
                    deadline_s: Optional[float] = None, *,
                    after_load: Optional[Callable[[Any], Optional[dict]]] = None) -> list[list[str]]:
        """Load `kb_text` into a fresh interpreter, evaluate each `!` expression, return the result atoms
        of each as strings. Raises MettaRuntimeError. `after_load` is the chainer hook (see "The chainer hook"
        in the module docstring): PeTTa only, and a runtime that cannot honour it refuses it."""

    def session(self, kb_text: str) -> Session:
        """A stateful interpreter with `kb_text` loaded (tests and tools)."""


# ── text scanning ─────────────────────────────────────────────────────────────

#: a "string", a ;comment, a parenthesis, or a bare token
_SCAN_RE = re.compile(r'"(?:[^"\\]|\\[\s\S])*"|;[^\n]*|[()]|[^\s()";]+')
#: a "string", a parenthesis, or a bare token: the grammar of a RESULT, which has no comments
_RESULT_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\[\s\S])*"|[()]|[^\s()"]+')
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


# ── what the PeTTa runtime does not run: an ALLOWLIST, not a denylist ────────

#: Every name `src/metta.pl` registers as `fun/1` in the pinned PeTTa checkout (HEAD 93fb137's
#: `maplist(register_fun, [...])`) — PeTTa's own notion of "callable from MeTTa". A denylist of
#: individually-named "dangerous" forms misses whatever it does not know to name: an independent review
#: found `add-translator-rule!` reachable this way (it is `fun/1`-registered, lets a program register an
#: arbitrary symbol as a translator hook — `metta.pl:433-437` — so that every LATER occurrence of that
#: symbol as a head calls it directly as a Prolog goal during translation) and reproduced it with a
#: harmless canary through `PettaRuntime` and over HTTP `/metta/run`; it was on no refusal list anywhere
#: in this module. `_install_petta_sandbox` does not trust this frozen snapshot at run time — it enumerates
#: the LIVE `fun/1` facts of the engine it just constructed and guards every one not in
#: `PETTA_ALLOWED_FUN_NAMES`, so a name this constant has never seen (an upstream PeTTa version that
#: renames, re-arities or ADDS a `fun/1` entry) is refused the same way as any other non-allowed name,
#: automatically, rather than needing this list updated first or being silently trusted meanwhile; see
#: that function's docstring. `PETTA_KNOWN_FUN_NAMES` itself is this checkout's complete list (148 names),
#: kept for documentation and tests: a test asserts it still matches the live registry, so a version drift is
#: caught and reviewed rather than only silently guarded. Six of the names have no backing Prolog predicate at
#: all — `let`, `let*`, `concat`, `get-mettatype`, `mm2-exec`, `set_hook` are `fun/1`-registered but clauseless
#: in this checkout, confirmed against a live, unsandboxed dump: nothing to guard, nothing a program can call.
PETTA_KNOWN_FUN_NAMES: frozenset[str] = frozenset({
    "!=", "#*", "#+", "#-", "#//", "#<", "#=", "#>", "#\\=", "#div", "#max", "#min", "#mod", "%", "*", "+",
    "-", "/", "<", "<=", "=", "==", "=?", "=alpha", ">", ">=", "@<", "@<=", "@>", "@>=", "Predicate",
    "abs-math", "acos-math", "add-atom", "add-translator-rule!", "alpha-unique-atom", "and", "append",
    "argv", "asin-math", "assert", "assertaPredicate", "assertzPredicate", "atan-math", "atom_chars",
    "atom_concat", "bind!", "cached-import!", "callPredicate", "car-atom", "cdr-atom", "ceil-math",
    "change-state!", "concat", "cons", "cons-atom", "copy_term", "cos", "cos-math", "current-time",
    "decons", "decons-atom", "empty", "eval", "exclude-item", "exists_file", "exp", "filter-atom", "first",
    "first-from-pair", "floor-math", "foldl", "foldl-atom", "format-time", "get-atoms", "get-metatype",
    "get-mettatype", "get-state", "get-type", "id", "implies", "import!", "import_prolog_function",
    "index-atom", "intersection-atom", "is-alpha-member", "is-expr", "is-ground", "is-member", "is-space",
    "is-var", "isinf-math", "isnan-math", "last", "length", "let", "let*", "library", "library-import!",
    "list_to_set", "log", "log-math", "map-atom", "maplist", "match", "max", "max-atom", "member", "min",
    "min-atom", "mm2-exec", "msort", "not", "or", "parse", "pow-math", "println!", "py-call",
    "random-float", "random-int", "readln!", "reduce", "remove-atom", "remove-translator-rule!", "repr",
    "repra", "retractPredicate", "reverse", "round-math", "second-from-pair", "set_hook", "sin",
    "sin-math", "size-atom", "sort", "sort-atom", "sqrt", "sqrt-math", "sread", "subtraction-atom",
    "superpose", "tan-math", "term_hash", "test", "trunc-math", "union-atom", "unique-atom", "xor",
})

#: The subset of `PETTA_KNOWN_FUN_NAMES` a program may call under `PLN_RUNTIME=petta`, exactly as PeTTa defines
#: them: pure arithmetic, comparison, logic, list/tuple, string/repr and type-introspection builtins, `eval`
#: and `reduce`, and the two space-READ forms (`match`, `get-atoms`) every taught few-shot query, every tracked
#: `.metta` file's rules and the ontology/router generators are built from. None of these mutate `&self`,
#: touch the filesystem, the network, a process, or any process-global state — and `match`, `get-atoms`,
#: `eval` and `reduce` are the ones that turn a program's data into a call, which is why the sandbox also
#: constrains what they are given (`_install_petta_sandbox`).
#: `add-atom`/`remove-atom` are NOT here — they stay on the flagged path (`_SANDBOX_FLAGGED_PREDICATES`):
#: needed verbatim while a KB loads, refused while a `!` form runs. `PETTA_ENGINE_INTERNAL_FUN_NAMES` below
#: are not here either. Everything else `PETTA_KNOWN_FUN_NAMES` holds — `change-state!`, `get-state`,
#: `bind!`, `assert`, `test` (halt the process), `add-translator-rule!`/`remove-translator-rule!` (registers a
#: symbol PeTTa then calls directly, by name, the next time it is a head), `callPredicate`/`assertaPredicate`/
#: `assertzPredicate`/`retractPredicate`/`import_prolog_function`/`Predicate` (PeTTa's "call any Prolog
#: predicate by name" doors), `import!`/`cached-import!`/`library-import!`/`library`, `py-call`, and every
#: name with no legitimate caller anywhere in this repo (`argv`, `repra`, ...) — is refused by
#: `_install_petta_sandbox`, by construction, because it is simply not in this set; nothing had to be
#: named as "known-dangerous" for that to happen.
PETTA_ALLOWED_FUN_NAMES: frozenset[str] = frozenset({
    "!=", "#*", "#+", "#-", "#//", "#<", "#=", "#>", "#\\=", "#div", "#max", "#min", "#mod", "%", "*", "+",
    "-", "/", "<", "<=", "=", "==", "=?", "=alpha", ">", ">=", "@<", "@<=", "@>", "@>=", "abs-math",
    "acos-math", "alpha-unique-atom", "and", "append", "asin-math", "atan-math", "atom_chars",
    "atom_concat", "car-atom", "cdr-atom", "ceil-math", "cons", "cons-atom", "copy_term", "cos",
    "cos-math", "current-time", "decons", "decons-atom", "empty", "eval", "exclude-item",
    "exp", "filter-atom", "first", "first-from-pair", "floor-math", "foldl-atom", "format-time",
    "get-atoms", "get-type", "id", "implies", "index-atom", "intersection-atom", "is-alpha-member",
    "is-expr", "is-ground", "is-member", "is-space", "is-var", "isinf-math", "isnan-math", "last",
    "length", "list_to_set", "log", "log-math", "map-atom", "match", "max", "max-atom",
    "member", "min", "min-atom", "msort", "not", "or", "parse", "pow-math", "println!", "random-float",
    "random-int", "readln!", "reduce", "repr", "reverse", "round-math", "second-from-pair",
    "sin", "sin-math", "size-atom", "sort", "sort-atom", "sqrt", "sqrt-math", "subtraction-atom",
    "superpose", "tan-math", "term_hash", "trunc-math", "union-atom", "unique-atom", "xor", "get-metatype",
})

#: `fun/1`-registered names PeTTa's OWN code calls as plain Prolog predicates on every load and every query,
#: by the same name and a matching arity, so the predicates cannot be redefined, and which are nonetheless
#: not for a program: `maplist` and `foldl` call whatever predicate their first argument names (with
#: `getenv` or `directory_files` there, a program reads the server's environment and directories — the
#: independent review reproduced both over HTTP), `sread` parses text and `exists_file` is a file-existence
#: oracle. They are therefore not in `PETTA_ALLOWED_FUN_NAMES`, and `_install_petta_sandbox` does not redefine
#: them either: it takes them off the `fun/1` list, the one place the translator and `reduce/2` look to decide
#: that a symbol is a function. After that `(maplist f $xs)` is data, as it is for any symbol that names
#: no function, and the engine's own Prolog calls to `maplist/3`, `foldl/4`, `sread/2` and `exists_file/1`
#: are untouched. (The translator's compiled `map`/`foldl` pseudo-lambdas call them directly, not through
#: `fun/1`, and the full suite runs on this.)
PETTA_ENGINE_INTERNAL_FUN_NAMES: frozenset[str] = frozenset({"maplist", "foldl", "sread", "exists_file"})

#: `fun/1`-registered names with NO backing Prolog predicate in this checkout at all (confirmed against a
#: live, unsandboxed dump: `arity/2` holds no fact for any of them) — pure compiler forms (`let`, `let*`)
#: inlined into the translator, or legacy/optional registrations nothing in this repo ever defines
#: (`concat`, `get-mettatype`, `mm2-exec`, `set_hook`). `_install_petta_sandbox`'s engine-level guard
#: already skips these (nothing to redefine), and they are excluded here too: `let`/`let*` are ordinary,
#: pervasive MeTTa syntax (every KB file and most taught few-shot queries use one or both), so putting
#: them in the TEXT-scan refusal list below would 422 nearly every real program, for a name that was never
#: callable as a predicate in the first place.
_PETTA_CLAUSELESS_FUN_NAMES: frozenset[str] = frozenset({
    "let", "let*", "concat", "get-mettatype", "mm2-exec", "set_hook",
})

#: Forms refused under PLN_RUNTIME=petta, for the TEXT scan (`petta_refused_forms`/`_refuse_if_forbidden`):
#: every `PETTA_KNOWN_FUN_NAMES` name NOT in `PETTA_ALLOWED_FUN_NAMES` and not clauseless (this is what
#: makes `add-translator-rule!`/`remove-translator-rule!` — and anything else not explicitly allowed —
#: refused at the fast, pre-engine 422 too, not only at `_install_petta_sandbox`'s engine-level boundary),
#: plus a handful of forms that are NOT `fun/1`-registered at all (reachable only through the
#: generic-dispatch primitives — `call`, `callPredicate`+`Predicate`, `match`'s own Space-as-predicate-name
#: dispatch — kept here for the same fast-path friendliness the previous list already gave `shell`/`halt`/
#: `py-atom`/`py-dot`/`git-import!`/`include`/`load_metta_file`/`load_metta_file_cached`/
#: `process_metta_string`/`new-space`; none of these tokens collide with ordinary KB prose — checked
#: against every tracked `.metta` file, both prompt files and every test fixture). `=` keeps its existing,
#: narrower meaning (a TOP-LEVEL `(= ...)` only — see `petta_refused_forms`'s docstring); it is not a
#: `fun/1` name at all, it is MeTTa's definition syntax.
#: This text scan is a friendly, fast 422 for the literal spelling; it is NOT the security boundary for any
#: of these forms — `_install_petta_sandbox` is (a symbol built at runtime by `atom_concat` never appears in
#: the text this function reads, so it cannot be caught here; see that function's docstring).
PETTA_REFUSED_FORMS: tuple[str, ...] = tuple(sorted(
    (PETTA_KNOWN_FUN_NAMES - PETTA_ALLOWED_FUN_NAMES - _PETTA_CLAUSELESS_FUN_NAMES) | {
        "=", "halt", "git-import!", "include", "py-atom", "py-dot", "shell", "load_metta_file",
        "load_metta_file_cached", "process_metta_string", "new-space", "call", "translatePredicate",
    }
))
_REFUSED_ANYWHERE = frozenset(PETTA_REFUSED_FORMS) - {"="}


#: PeTTa's own reading of program text, as far as a symbol scan needs it. Two stages decide what a symbol IS:
#:   `filereader.pl`'s `strip/3` removes `;` comments (and the newline that ends one, so a comment can join the
#:   text on either side of it into one token) while tracking "strings" — a `"` ANYWHERE toggles, a backslash
#:   escapes only inside one; then `parser.pl`'s `sexpr//3` reads a form: a token is a run of anything but
#:   whitespace and parentheses, so a `"` in the MIDDLE of a token is an ordinary character, and a string
#:   literal can only START an element. A scan that tokenises the way a programmer reads (a quote starts a
#:   string wherever it stands) sees `a"b add-atom c"d` as `a`, one string and `d`; PeTTa reads the three atoms
#:   `a"b`, `add-atom` and `c"d`. These two expressions are that reading; `tests/test_metta_runtime.py` checks
#:   them against the engine's own reader on a corpus of nasty texts.
_PETTA_STRIP_RE = re.compile(r'"(?:[^"\\]|\\[\s\S])*"|;[^\n]*\n?')
#: ... and which characters end a token, and which are skipped BEFORE one, are two different sets. `parser.pl`'s
#: `token//1` ends a token at a space, tab, newline, carriage return or parenthesis and at nothing else, so
#: `b"<U+2029>"` is ONE token whatever the Unicode says about U+2029, and the quote inside it starts no string.
#: The whitespace skipped between elements, `blanks//0` (SWI-Prolog's `code_type(C, space)`), is wider and not
#: the same everywhere: the vertical tab and form feed on every platform, then (build and platform dependent)
#: U+001C-U+001F, U+0085, U+00A0, U+1680, U+2000-U+200A, U+2028, U+2029, U+202F, U+205F and U+3000 (SWI-Prolog 10
#: on macOS skips all of them but U+001C-U+001F and U+0085). It matters in one place: a `"` after such a
#: character starts a string if the character is a blank and is part of a token if it is not, and a string hides
#: whatever it holds. `tests/test_petta_sandbox.py` checks both lists against the engine it runs on.
_PETTA_SURE_BLANKS = "\x0b\x0c"
_PETTA_MAYBE_BLANKS = "\x1c-\x1f\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"


def _petta_token_re(skipped: str) -> "re.Pattern[str]":
    """The tokens of a form when `blanks//0` skips `skipped` (a regex character-class body) before an element:
    a string, a parenthesis, or a symbol that does not START with such a character and then runs on to the next
    space, tab, newline, carriage return or parenthesis."""
    return re.compile(r'(?P<string>"(?:[^"\\]|\\[\s\S])*")|(?P<paren>[()])|(?P<symbol>[^ \t\r\n()' + skipped
                      + r'][^ \t\r\n()]*)')


#: the widest reading of `blanks//0` (every character that might be one is one) and the narrowest (only those
#: that always are)
_PETTA_TOKEN_RE = _petta_token_re(_PETTA_SURE_BLANKS + _PETTA_MAYBE_BLANKS)
_PETTA_TOKEN_RE_NARROW = _petta_token_re(_PETTA_SURE_BLANKS)
_PETTA_MAYBE_BLANK_RE = re.compile("[" + _PETTA_MAYBE_BLANKS + "]")
_PETTA_ODD_SPACE_RE = re.compile("[" + _PETTA_SURE_BLANKS + _PETTA_MAYBE_BLANKS + "]")


def petta_strip_comments(text: str) -> str:
    """`text` as PeTTa's `strip/3` leaves it: every `;` comment gone together with its newline, strings intact."""
    if ";" not in text:
        return text
    return _PETTA_STRIP_RE.sub(lambda m: m.group() if m.group()[0] == '"' else "", text)


def petta_tokens(text: str) -> list[tuple[str, str]]:
    """('string' | 'paren' | 'symbol', text) for each token PeTTa's reader sees in `text`, comments aside.

    One reading: the widest one of `blanks//0` (see `_PETTA_MAYBE_BLANKS`). A name check must not rest on it
    alone; `petta_symbols` and everything built on it read the text every way the engine might."""
    return [(m.lastgroup, m.group()) for m in _PETTA_TOKEN_RE.finditer(petta_strip_comments(text))]


def _petta_token_readings(text: str) -> list[list[tuple[str, str]]]:
    """The token sequences of `text` under every reading of the whitespace PeTTa's reader might skip, each with
    a symbol that holds odd whitespace also split at it (`a<vertical tab>add-atom` is one atom for the reader;
    the scan reports `add-atom` too, the safe way round). A text with no such character has one reading."""
    stripped = petta_strip_comments(text)
    if not _PETTA_ODD_SPACE_RE.search(stripped):
        return [[(m.lastgroup, m.group()) for m in _PETTA_TOKEN_RE.finditer(stripped)]]
    patterns = [_PETTA_TOKEN_RE]
    if _PETTA_MAYBE_BLANK_RE.search(stripped):
        patterns.append(_PETTA_TOKEN_RE_NARROW)
    readings = []
    for pattern in patterns:
        tokens: list[tuple[str, str]] = []
        for m in pattern.finditer(stripped):
            if m.lastgroup == "symbol" and _PETTA_ODD_SPACE_RE.search(m.group()):
                tokens.extend(("symbol", piece) for piece in m.group().split())
            else:
                tokens.append((m.lastgroup, m.group()))
        readings.append(tokens)
    return readings


def petta_symbols(text: str) -> frozenset[str]:
    """Every symbol PeTTa's reader could see in `text` (comments and strings aside) under any reading of the
    whitespace its `blanks//0` skips; a symbol that holds odd whitespace stands for the pieces it splits into. A
    name the reader finds is never missing from this set; it can hold a few names the reader does not see (a
    piece of such a symbol, or text with an odd space character before a quote, where whether the quote opens a
    string depends on the SWI-Prolog build), which is the safe direction for a refusal."""
    return frozenset(tok for tokens in _petta_token_readings(text) for kind, tok in tokens if kind == "symbol")


def petta_refused_forms(text: str) -> list[str]:
    """The refused forms `text` uses, sorted, comments and strings aside.

    A name other than `=` is refused wherever it stands as a symbol, not only as a head: `(let $f add-atom
    ($f &self x))` reaches `add-atom` without ever writing it as a head. `=` is refused only as the head of a
    TOP-LEVEL form, which is a definition; inside a pattern `(match &self (= (f $x) $v) ...)` reads the
    equations and defines nothing (two taught few-shot examples do that).

    Symbols are the ones PeTTa's own reader finds (see `petta_tokens`). This scan is a fast, friendly refusal
    for the literal spelling and nothing more: a symbol built at run time never appears in the text at all,
    and the engine (`_install_petta_sandbox`) refuses what the program does, whatever it is spelled like.
    """
    found: set[str] = set()
    for tokens in _petta_token_readings(text):
        depth = 0
        previous = ""
        for kind, tok in tokens:
            if kind == "string":
                continue
            if kind == "paren":
                depth = depth + 1 if tok == "(" else max(0, depth - 1)
            elif tok in _REFUSED_ANYWHERE:
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
    return sorted(petta_symbols(text) & _REFUSED_ANYWHERE)


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
#
# What a program can make PeTTa call, and what stops it. Every way a MeTTa program reaches a Prolog predicate
# by a name IT chooses, read off the pinned checkout (`src/translator.pl`, `src/spaces.pl`, `src/metta.pl`,
# `src/typecheck/builtin_registry.pl`), and the single place each is closed. The text scan is not on this list:
# it only gives the literal spelling a friendly 422, and a program that builds a name at run time never
# contains the literal.
#
#   dynamic dispatch   `($f ...)` with `$f` bound at run time: `translator.pl`'s `reduce/2` calls it only when
#   `reduce/2`,        `fun($f)` holds — the one fact the translator and `reduce/2` both ask. Every `fun/1`
#   `apply_fn*`        name that is not allowed is redefined to throw; `maplist`, `foldl`, `sread` and
#                      `exists_file` (PeTTa calls them itself) are taken OFF `fun/1`, so they are data.
#   compiler forms     `call` and `translatePredicate` build `Goal =.. [F|Args]` from the program's own data
#   `manual_call`,     and run it with no `fun/1` check; they can also be reached without ever writing the
#   `translate_`       name (`atom_concat`, `cons-atom`, then `eval` of the list, which compiles it). They
#   `predicate`        are chosen by `builtin_registry:special_builtin_form/3`, which is wrapped so the form
#                      is refused unless the target it names is a `fun/1` function (PeTTa's own rewrites of
#                      `unique`, `union`, ... produce `(call (superpose ...))` and keep working).
#   spaces             `match` and `get-atoms` turn the program's Space argument into a predicate name:
#                      `Term =.. [Space, Rel|Args], call(Term)`, so `(match getenv (HOME $v) $v)` is a call
#                      of `getenv/2` (and `read_file_to_string`, `directory_files`, ... likewise).
#                      `space_call/1` and `get-atoms/2` accept only the name of a space, an atom that begins
#                      with `&` — PeTTa's own `is-space` test.
#   space mutation     `add-atom`, `remove-atom` and the `-proven` pair the translator emits for a typed
#                      space refuse while a `!` form runs (`process_form/4` is wrapped to say so); the
#                      engine's own `add_sexp/2` and `remove_sexp/2` stay open, because PeTTa's runtime
#                      specialiser calls them while it compiles a query.
#   Prolog by name     `callPredicate`, `Predicate`, `assertzPredicate`, ... and `shell`/`halt`: refused.
#
# Not closed, and why: an OS-level confinement (a seccomp or sandbox_init profile) differs per platform and
# cannot be tested on more than one here; `_child_prepare` removes what a process holds that a program should
# not reach (the environment, the parent's descriptors, the working directory), but a host predicate that
# some future PeTTa version lets a program name is stopped by this list only if it is on it.

#: `add-atom`/`remove-atom` materialise every ordinary ground fact while a KB loads (`filereader.pl`'s
#: `process_form` calls them for every non-`!` form), so they keep their real behaviour there and refuse only
#: while the flag is set. The flag is set around each `!` form by the `process_form/4` wrapper (so a `!` form
#: hidden in `kb_text`/`extra_atoms` is covered the same as an expression), and set for good once a
#: `run_program` child has loaded its KB.
_SANDBOX_FLAGGED_PREDICATES: tuple[tuple[str, int], ...] = (
    ("add-atom", 3), ("remove-atom", 3), ("add-atom-proven", 3), ("remove-atom-proven", 3),
)

#: Non-`fun/1` Prolog predicates guarded unconditionally: `shell` and `halt` are real SWI predicates, never
#: registered as a MeTTa builtin, so enumerating `fun/1` never surfaces them. With the doors above closed
#: they are unreachable from a program; they stay as a second line. Deliberately NOT extended to every
#: dangerous predicate SWI ships (`open`, `delete_file`, `process_create`, `getenv`, ...): PeTTa and janus call
#: several of them on every load, so redefining them breaks the engine, and a predicate no program can name is
#: not made safer by it.
_SANDBOX_NONFUN_ALWAYS_PREDICATES: tuple[tuple[str, int], ...] = (
    ("shell", 1), ("shell", 2), ("shell", 3), ("halt", 1),
)

#: The closure-call shortcuts of `translator.pl`: they call the function a `partial(F, Args)` closure names.
_SANDBOX_CLOSURE_PREDICATES: tuple[tuple[str, int], ...] = (
    ("apply_fn1", 3), ("apply_fn2", 4), ("apply_fn3", 5), ("apply_fnN", 3),
)

#: Prolog consulted once per child, right after `PeTTa()` has loaded `main.pl`/`helper.pl`. `@FLAG@` and
#: `@NOTE@` are replaced by per-child random names, and so are the private names the original predicates are
#: copied to (`_install_petta_sandbox`), so none of the sandbox's state has a name to guess.
_SANDBOX_SETUP_SRC = r"""
%The state lives in a flag and a dynamic predicate, not in global variables: nb_setval/nb_getval are
%thread-local in SWI-Prolog, and PeTTa's `hyperpose` runs its branches in threads of their own, which would
%see the sandbox switched off (an `add-atom` in a branch mutated the space) and a refusal noted nowhere
%the main thread can read it. Flags and the clauses of a dynamic predicate are shared by every thread.
:- dynamic '@NOTE@'/2.
petta_sandbox_blocked :- flag('@FLAG@', N, N), N > 0.
petta_sandbox_lock :- flag('@FLAG@', _, 1000000).
petta_sandbox_clear_refusal :- retractall('@NOTE@'(_, _)).
petta_sandbox_last_refusal(Name, Why) :- '@NOTE@'(Name, Why), !.

%Records the refused name and the reason (builtin | mutation | space | target) where neither a catch/3 nor
%backtracking can unwind it, then throws. translator.pl's reduce/2 (a dynamically dispatched head) wraps its call in
%catch(_, _, fail), which turns the throw into an ordinary failure: the containing expression has no solution
%and would read as "no answer". The note survives that, and _PettaEngine._raise_if_refused reads it after
%every call.
petta_sandbox_refuse(Name, Why) :-
    ignore(retractall('@NOTE@'(_, _))),
    assertz('@NOTE@'(Name, Why)),
    throw(error(permission_error(call, Name), petta_sandbox)).

%Runs Goal once with the flag raised, and lowers it again, however Goal ends.
petta_sandbox_blocking_call(Goal) :-
    setup_call_cleanup(flag('@FLAG@', N, N + 1), once(Goal), flag('@FLAG@', M, M - 1)).

%Every name PeTTa itself treats as a MeTTa builtin - read once per child to compute the allowlist complement
%from the LIVE registry, not a frozen Python guess at it.
petta_sandbox_fun_names(Out) :- findall(N, user:fun(N), Out).
%Every arity Name is actually CALLABLE at, not just the ones PeTTa's own `arity/2` recorded: register_fun's
%own scan (metta.pl) excludes an arity <= 2 for any name that is ALSO a Prolog OPERATOR (`current_op(_,_,N)`),
%which is why `arity('/',2)` is absent even though SWI's `/` IS callable at that arity - `reduce/2`, PeTTa's
%own dispatcher, checks `current_predicate/1` directly, not `arity/2`, so an excluded arity is still reached.
%Bounded at 9: no name in this checkout resolves above it (confirmed against the live registry by a test).
petta_sandbox_reachable_arities(Name, Out) :-
    findall(A, (between(0, 9, A), current_predicate(Name/A)), Out).
%Whether Name/Arity resolves to library(yall) - SWI's generic "apply a goal, closing over free variables"
%mechanism (`/`, `>>`), exported for lambdas and NOT limited to a fixed arity. PeTTa's own `/` (division,
%metta.pl, arity 3) and yall's `/` share one name by accident; yall's import brings every OTHER arity of it
%along, so an allowed, purely-arithmetic name can still carry a generic call-by-name door at arities nobody
%meant to allow. Checked by origin, not by arity, so this also catches an arity this checkout does not use yet.
petta_sandbox_is_generic_apply(Name, Arity) :-
    functor(Head, Name, Arity),
    predicate_property(user:Head, imported_from(yall)).

%Copies Module:Name/Arity's clauses under a private name.
petta_sandbox_copy(M, Name, Arity, OrigName) :-
    functor(Head, Name, Arity), Head =.. [Name|Args],
    OrigHead =.. [OrigName|Args],
    findall(OrigHead-Body, clause(M:Head, Body), Clauses),
    Clauses \== [],
    forall(member(OH-B, Clauses), assertz(M:(OH :- B))).

%Replaces Name/Arity with a predicate that throws while the flag is set and otherwise runs the original.
petta_sandbox_guard_flagged(M, Name, Arity, OrigName) :-
    petta_sandbox_copy(M, Name, Arity, OrigName),
    functor(Head, Name, Arity), Head =.. [Name|Args], OrigHead =.. [OrigName|Args],
    abolish(M:Name/Arity),
    assertz(M:(Head :- ( user:petta_sandbox_blocked -> user:petta_sandbox_refuse(Name, mutation) ; OrigHead ))).

%Replaces Name/Arity with a predicate that always throws: none of these has a legitimate caller in a KB.
petta_sandbox_guard_always(Name, Arity) :-
    functor(Head, Name, Arity),
    redefine_system_predicate(Name/Arity),
    ignore(catch(abolish(Name/Arity), _, true)),
    assertz((Head :- petta_sandbox_refuse(Name, builtin))).

%A program may name a space only the way PeTTa's own is-space/2 does: an atom beginning with '&'.
petta_sandbox_space_name(Space) :- atom(Space), sub_atom(Space, 0, 1, _, '&').

%spaces.pl: every `match` goes through space_call/1, which calls whatever predicate the Space argument named.
petta_sandbox_guard_space_call :-
    abolish(space_call/1),
    assertz((space_call(Term) :-
                (   callable(Term), functor(Term, Name, _), petta_sandbox_space_name(Name)
                ->  catch(Term, E, space_missing(E, Term))
                ;   petta_sandbox_refuse(match, space) ))).

%`get-atoms` lists the clauses of whatever predicate its first argument names.
petta_sandbox_guard_space_arg(Name, Arity, OrigName) :-
    petta_sandbox_copy(user, Name, Arity, OrigName),
    functor(Head, Name, Arity), Head =.. [Name, Space|Rest], OrigHead =.. [OrigName, Space|Rest],
    abolish(Name/Arity),
    assertz((Head :- ( petta_sandbox_space_name(Space) -> OrigHead ; petta_sandbox_refuse(Name, space) ))).

%Compiler forms that build a goal from the program's own data (see the table above). Allowed only when the
%target they name is a function PeTTa knows, which is what every ordinary call needs and every rewrite of
%PeTTa's own (`unique`, `union`, ...: `(call (superpose ...))`) satisfies.
petta_sandbox_goal_hook(manual_call).
petta_sandbox_goal_hook(translate_predicate).
petta_sandbox_goal_target([Expr]) :- nonvar(Expr), Expr = [Target|_], atom(Target), user:fun(Target).

petta_sandbox_guard_hooks(OrigName) :-
    petta_sandbox_copy(builtin_registry, special_builtin_form, 3, OrigName),
    abolish(builtin_registry:special_builtin_form/3),
    Orig =.. [OrigName, F, Args, Hook],
    assertz(builtin_registry:(special_builtin_form(F, Args, Hook) :-
                (   nonvar(Hook), user:petta_sandbox_goal_hook(Hook)
                ->  Orig,
                    (   user:petta_sandbox_goal_target(Args) -> true ; user:petta_sandbox_refuse(F, target) )
                ;   Orig ))).

%A closure PeTTa builds for an under-applied function is `partial(F, Args)`, and the closure-call shortcuts
%(apply_fn1/2/3/N) call F without asking whether it is a function. PeTTa only ever builds one for a function;
%nothing in the allowed set lets a program write one (repra, which parsed text into a term, is not allowed),
%but this is the sink a forged term would reach, so it asks.
petta_sandbox_closure_ok(F) :- var(F), !.
petta_sandbox_closure_ok(partial(Base, _)) :- !, atom(Base), user:fun(Base).
petta_sandbox_closure_ok(_).

petta_sandbox_guard_closure(Name, Arity, OrigName) :-
    petta_sandbox_copy(user, Name, Arity, OrigName),
    functor(Head, Name, Arity), Head =.. [Name, F|Rest], OrigHead =.. [OrigName, F|Rest],
    abolish(Name/Arity),
    assertz((Head :- ( petta_sandbox_closure_ok(F) -> OrigHead ; petta_sandbox_refuse(partial, target) ))).

%filereader.pl: every `!` form - an expression, or one inside the KB text - is run by process_form/4.
petta_sandbox_guard_process_form(OrigName) :-
    petta_sandbox_copy(user, process_form, 4, OrigName),
    abolish(process_form/4),
    Orig =.. [OrigName, Space, Mode, Form, Result],
    assertz((process_form(Space, Mode, Form, Result) :-
                (   nonvar(Form), Form = parsed(runnable, _, _, _)
                ->  petta_sandbox_blocking_call(Orig)
                ;   Orig ))).

%The top-level forms of Text as PeTTa's OWN reader splits them (comments removed): [[bang|fact, FormText], ...].
petta_sandbox_forms(Text, Pairs) :-
    metta_string_forms(Text, Forms),
    findall([Kind, Str],
            ( member(Form, Forms), Form =.. [Tag, Str, _], ( Tag == runnable -> Kind = bang ; Kind = fact ) ),
            Pairs).
"""


def _install_petta_sandbox(janus: Any) -> None:
    """Close, inside the engine, every way a program reaches a Prolog predicate by a name it chooses (the
    table above). The restriction has to live in the engine, not in the caller's text: a symbol built at run
    time (`atom_concat "add" "-atom"`, bound to a variable, applied) never appears as a token a scan can read,
    and PeTTa's own dispatcher treats a function's identity as an ordinary runtime value.

    An ALLOWLIST, not a denylist, for the names PeTTa registers: every name the LIVE `fun/1` registry holds
    (read from the engine just constructed, not assumed from `PETTA_KNOWN_FUN_NAMES`) is redefined to throw
    unless it is in `PETTA_ALLOWED_FUN_NAMES`, so a name this module has never seen — a future PeTTa builtin —
    is refused without anyone having to name it. `PETTA_ENGINE_INTERNAL_FUN_NAMES` are the exception that
    cannot be redefined, taken off `fun/1` instead. A `fun/1` name with no Prolog predicate behind it (`let`,
    `let*`, `concat`, `get-mettatype`, `mm2-exec`, `set_hook` in this checkout: `arity/2` holds nothing) is
    skipped, because there is nothing to call.

    Raises if a predicate cannot be found or redefined (the child then reports a `kind="start"` error and is
    discarded): refusing to run a program is safer than running one with this boundary silently missing.

    `halt/0` is the one predicate SWI will not let any program redefine (`redefine_system_predicate(halt/0)`
    followed by `asserta/1` still raises `permission_error(modify, static_procedure, halt/0)`). Reaching it
    (`(call (halt))`) is refused by the `call` rule; if it were reached, the child would end with exit code 0,
    which `_Child._died()` reports as a `kind="crash"`: a halt only ever ends the one process it runs in.

    Every private name and both state keys carry a fresh random token, so nothing a program could name is
    shared between children or guessable.
    """
    token = secrets.token_hex(6)
    janus.consult("petta_sandbox_setup", data=_SANDBOX_SETUP_SRC.replace("@FLAG@", f"petta_sandbox_flag_{token}")
                  .replace("@NOTE@", f"petta_sandbox_note_{token}"))

    def must(goal: str, bindings: Optional[dict] = None, what: str = "") -> None:
        # janus answers a goal that failed with {"truth": False}, which is a non-empty (truthy) dict
        if not janus.query_once(goal, bindings or {}).get("truth"):
            raise RuntimeError(f"could not sandbox the PeTTa {what or goal}")

    flagged = {name for name, _ in _SANDBOX_FLAGGED_PREDICATES}
    for name, arity in _SANDBOX_FLAGGED_PREDICATES:
        must("petta_sandbox_guard_flagged(user, N, A, O)", {"N": name, "A": arity, "O": f"petta_orig_{token}_{name}"},
             f"predicate '{name}'/{arity}")
    live_fun_names = {str(n) for n in janus.query_once("petta_sandbox_fun_names(Out)")["Out"]}
    if not live_fun_names:
        raise RuntimeError("could not read the PeTTa builtin registry (fun/1 is empty)")
    for name in sorted(live_fun_names - PETTA_ALLOWED_FUN_NAMES - PETTA_ENGINE_INTERNAL_FUN_NAMES - flagged):
        arities = janus.query_once("petta_sandbox_reachable_arities(Name, Out)", {"Name": name})["Out"]
        for raw_arity in arities:
            must("petta_sandbox_guard_always(N, A)", {"N": name, "A": int(raw_arity)},
                 f"predicate '{name}'/{int(raw_arity)}")
    # An ALLOWED name can still carry a generic call-by-name door at an arity nobody meant to allow (see
    # `petta_sandbox_is_generic_apply`'s docstring: `/` is exactly this, confirmed against the live engine).
    for name in sorted(live_fun_names & PETTA_ALLOWED_FUN_NAMES):
        for raw_arity in janus.query_once("petta_sandbox_reachable_arities(Name, Out)", {"Name": name})["Out"]:
            arity = int(raw_arity)
            if janus.query_once("petta_sandbox_is_generic_apply(N, A)", {"N": name, "A": arity}).get("truth"):
                must("petta_sandbox_guard_always(N, A)", {"N": name, "A": arity}, f"predicate '{name}'/{arity}")
    for name in sorted(live_fun_names & PETTA_ENGINE_INTERNAL_FUN_NAMES):
        janus.query_once("retractall(fun(N))", {"N": name})
    for name, arity in _SANDBOX_NONFUN_ALWAYS_PREDICATES:
        must("petta_sandbox_guard_always(N, A)", {"N": name, "A": arity}, f"predicate '{name}'/{arity}")
    must("petta_sandbox_guard_space_call", what="predicate 'space_call'/1")
    must("petta_sandbox_guard_space_arg(N, A, O)", {"N": "get-atoms", "A": 2, "O": f"petta_orig_{token}_get-atoms"},
         "predicate 'get-atoms'/2")
    must("petta_sandbox_guard_hooks(O)", {"O": f"petta_orig_{token}_special_builtin_form"},
         "predicate 'special_builtin_form'/3")
    for name, arity in _SANDBOX_CLOSURE_PREDICATES:
        must("petta_sandbox_guard_closure(N, A, O)", {"N": name, "A": arity, "O": f"petta_orig_{token}_{name}"},
             f"predicate '{name}'/{arity}")
    must("petta_sandbox_guard_process_form(O)", {"O": f"petta_orig_{token}_process_form"},
         "predicate 'process_form'/4")


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

    def __init__(self, pid: int, conn: Connection, scratch: Optional[str] = None) -> None:
        self.pid = pid
        self.conn = conn
        self.memory_limit_mb = 0                # what the runtime asked of it, for the message when it is stopped
        self.scratch = scratch                  # the child's working directory, removed once it is reaped
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
        if self.scratch is not None and self.status is not None:
            shutil.rmtree(self.scratch, ignore_errors=True)
            self.scratch = None

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
        if self.status is not None and self.status >= 0 and os.WIFEXITED(self.status) \
                and os.WEXITSTATUS(self.status) == _EXIT_MEMORY:
            limit = f"{self.memory_limit_mb} MB" if self.memory_limit_mb else "its memory limit"
            return MettaRuntimeError(f"the PeTTa process used more than {limit} of memory and was stopped",
                                     kind="memory", detail=how)
        return MettaRuntimeError(f"the PeTTa process ended with {how} before it answered{hint}", kind="crash")


#: The directory the forked child works in (set in the child by `_fork_child`, read by `_child_prepare`).
_CHILD_SCRATCH: Optional[str] = None


def _fork_child(target: Callable[..., None], *args: Any) -> _Child:
    """Fork a child that runs `target(conn, *args)` and never returns into the caller's code. The child gets a
    fresh empty directory of its own (private to this process, removed by `_Child.close`): its working
    directory, home, temp directory and cache directory."""
    global _CHILD_SCRATCH
    scratch = tempfile.mkdtemp(prefix="pln-petta-")
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
            shutil.rmtree(scratch, ignore_errors=True)
            raise
        if pid == 0:                            # the child
            code = 70
            try:
                _CHILD_SCRATCH = scratch
                parent_conn.close()
                target(child_conn, *args)
                code = 0
            except BaseException:               # noqa: BLE001 - nothing may escape into the caller's stack
                pass
            finally:
                os._exit(code)
        child_conn.close()
    return _Child(pid, parent_conn, scratch)


#: What a PeTTa child keeps of the server's environment, by exact name (and every `LC_*`): what the dynamic
#: linker and SWI-Prolog need to start. Everything else goes — the server's `.env` puts its API keys in
#: `os.environ`, and a child has no use for them. `PLN_PETTA_CHILD_ENV_KEEP` (comma-separated names, read from
#: the server's environment) adds names for a deployment whose SWI-Prolog needs one that is not listed.
PETTA_CHILD_ENV_KEEP: tuple[str, ...] = (
    "PATH", "LANG", "LANGUAGE", "TZ", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "SWI_HOME_DIR",
)
#: Soft limits set in the child on top of the CPU backstop: no core file (it would hold the server's memory),
#: a bounded number of descriptors, a bounded file size.
_CHILD_LIMITS: tuple[tuple[str, int], ...] = (
    ("RLIMIT_CORE", 0), ("RLIMIT_NOFILE", 512), ("RLIMIT_FSIZE", 64 * 1024 * 1024),
)


def _scrub_environment(scratch: Optional[str]) -> None:
    """Replace the child's environment by the allowlisted names, and point everything that names a place to
    write (`HOME`, `TMPDIR`, `XDG_CACHE_HOME`, PeTTa's own `PETTA_LOAD_CACHE`) at the child's scratch directory."""
    keep = set(PETTA_CHILD_ENV_KEEP) | {n.strip() for n in os.environ.get("PLN_PETTA_CHILD_ENV_KEEP", "").split(",")
                                        if n.strip()}
    kept = {k: v for k, v in os.environ.items() if k in keep or k.startswith("LC_")}
    os.environ.clear()                          # unsetenv for each name: the C environment goes too
    os.environ.update(kept)
    if scratch:
        os.environ.update(HOME=scratch, TMPDIR=scratch, XDG_CACHE_HOME=scratch,
                          PETTA_LOAD_CACHE=os.path.join(scratch, "load-cache"))


def _neutralise_inherited_descriptors(devnull: int, keep: tuple[int, ...]) -> None:
    """Point every inherited descriptor but 0-2, the child's pipe and `devnull` at /dev/null: the parent's
    sockets, log files and its other children's pipes are not for the child. The numbers stay taken (they are
    not closed): a Python object the fork copied may still close its descriptor when it is collected, and
    that must not close one the engine has opened since."""
    for directory in ("/proc/self/fd", "/dev/fd"):
        try:
            numbers = [int(n) for n in os.listdir(directory)]
            break
        except (OSError, ValueError):
            continue
    else:                                       # pragma: no cover - no way to list them
        numbers = list(range(3, 256))
    for fd in numbers:
        if fd > 2 and fd != devnull and fd not in keep:
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass


def _address_space_bytes() -> Optional[int]:
    """This process's virtual size, from /proc (Linux); None where there is no such file."""
    try:
        with open("/proc/self/statm", "rb") as handle:
            return int(handle.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _limit_address_space(memory_limit_mb: int) -> None:
    """Linux: have the KERNEL refuse an allocation that takes the child past its address space now + the memory
    limit + `_ADDRESS_SPACE_SLACK_MB`. This is the wall; `_start_memory_watchdog` is the sampler in front of it.
    A no-op elsewhere: macOS answers a finite RLIMIT_AS with EINVAL."""
    if memory_limit_mb <= 0 or not sys.platform.startswith("linux"):
        return
    try:
        import resource
        res = getattr(resource, "RLIMIT_AS", None)
        used = _address_space_bytes()
        if res is None or used is None:
            return
        wanted = used + (memory_limit_mb + _ADDRESS_SPACE_SLACK_MB) * 1024 * 1024
        current, hard = resource.getrlimit(res)
        if current != resource.RLIM_INFINITY:
            wanted = min(wanted, current)
        if hard != resource.RLIM_INFINITY:
            wanted = min(wanted, hard)
        resource.setrlimit(res, (wanted, hard))
    except (OSError, ValueError, ImportError):  # pragma: no cover - a kernel that refuses; the watchdog remains
        pass


def _peak_rss_bytes() -> Optional[int]:
    """This process's peak resident set in bytes (`ru_maxrss` is bytes on macOS and kilobytes elsewhere)."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except Exception:                           # pragma: no cover
        return None


def _start_memory_watchdog(memory_limit_mb: int) -> None:
    """Child only. A thread that ends the process with `_EXIT_MEMORY` once its peak resident set is more than
    `memory_limit_mb` above what it was now. `_exit` ends the process at once, whatever the main thread is
    doing, which is the point: a janus call cannot be interrupted politely, and janus lets go of the GIL while
    Prolog runs, so this thread keeps running while a program grows."""
    if memory_limit_mb <= 0:
        return
    baseline = _peak_rss_bytes()
    if baseline is None:                        # pragma: no cover - a platform without getrusage
        return
    ceiling = baseline + memory_limit_mb * 1024 * 1024

    def watch() -> None:
        while True:
            time.sleep(_MEMORY_POLL_S)
            peak = _peak_rss_bytes()
            if peak is not None and peak > ceiling:
                os._exit(_EXIT_MEMORY)

    threading.Thread(target=watch, name="petta-memory-watchdog", daemon=True).start()


def _limit_child_resources(memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB) -> None:
    try:
        import resource
    except ImportError:                         # pragma: no cover
        return
    _limit_address_space(memory_limit_mb)
    for name, soft in _CHILD_LIMITS:
        res = getattr(resource, name, None)
        if res is None:                         # pragma: no cover
            continue
        try:
            current, hard = resource.getrlimit(res)
            wanted = soft if current == resource.RLIM_INFINITY else min(soft, current)
            if hard != resource.RLIM_INFINITY:
                wanted = min(wanted, hard)
            resource.setrlimit(res, (wanted, hard))
        except (OSError, ValueError):           # pragma: no cover
            pass
    try:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)      # an oversized write fails with an error, it does not kill
    except (OSError, ValueError, AttributeError):          # pragma: no cover
        pass


def _child_prepare(deadline_s: Optional[float], *, parent_pid: Optional[int], keep_fds: tuple[int, ...] = (),
                   memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB) -> None:
    """First thing in a forked PeTTa child: cut it off from the parent's terminal, descriptors, environment and
    working directory, and bound it (CPU time, files, descriptors and memory: `memory_limit_mb`, 0 for none)."""
    # The child shares the server's heap copy-on-write. Where a fork maps pages lazily (macOS: measured, a child's
    # first full garbage collection added the whole 585 MB of a 400 MB server heap to its resident set) a
    # collection that walks every inherited object would be counted as the PROGRAM's growth and stop an honest
    # child. Frozen objects are never walked, and never copied by a collection either.
    gc.freeze()
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    _neutralise_inherited_descriptors(devnull, keep_fds)
    if devnull > 2:
        os.close(devnull)
    sys.stdout = sys.stderr = open(os.devnull, "w")
    sys.stdin = open(os.devnull, "r")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM, signal.SIGXCPU):
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError):           # pragma: no cover
            pass
    _scrub_environment(_CHILD_SCRATCH)
    if _CHILD_SCRATCH:
        try:
            os.chdir(_CHILD_SCRATCH)
        except OSError:                         # pragma: no cover
            pass
    _limit_child_resources(memory_limit_mb)
    _set_cpu_backstop(deadline_s)
    _start_memory_watchdog(memory_limit_mb)
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
    peak = _peak_rss_bytes()
    return None if peak is None else round(peak / 1e6, 1)


def _clip(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit // 2] + " ... " + text[-limit // 2:]


# ── the real PeTTa engine (constructed in the child only) ─────────────────────

_REFUSAL_REASONS = {
    "builtin": "that form is not in the allowed set (core.metta_runtime.PETTA_ALLOWED_FUN_NAMES)",
    "mutation": "a query may not change the space it reads",
    "space": "the space it was given is not a space (the name of one begins with '&')",
    "target": "it may only call a function PeTTa knows, and the one it names is not",
}


class _RefusedBuiltin(RuntimeError):
    """A sandboxed call reached something a program may not do; raised inside `_PettaEngine._helper`.

    The message names the refused head and the reason and nothing else — no filesystem path, no Prolog
    internals — so it is safe to show a caller verbatim. `_error_message` (below) recognises this type and
    reports it under `stage="refused"` regardless of whether it happened during a load or an expression,
    which `PettaRuntime._error_from`/`PettaSession` map to `MettaRuntimeError(kind="refused")`.
    """

    def __init__(self, name: str, why: str = "builtin") -> None:
        self.name = name
        self.why = why
        super().__init__(f"the PeTTa runtime refused to run '{name}': {_REFUSAL_REASONS.get(why, why)}")


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

    def _raise_if_refused(self) -> None:
        """Turn a recorded sandbox refusal into `_RefusedBuiltin`, clearing the side channel.

        A LITERAL head call to a guarded predicate raises normally: the `permission_error`
        `_install_petta_sandbox` throws propagates through `janus.query_once` as a `PrologError`. A
        DYNAMICALLY-DISPATCHED head (`($f ...)`, `$f` bound to a runtime-built symbol) does not:
        `translator.pl`'s `reduce/2` — PeTTa's own dynamic dispatcher — calls the resolved goal inside
        `catch(call(Goal), _, fail)`, a catch-all that turns ANY exception, including this one, into an
        ordinary Prolog FAILURE. The containing expression then has no solution, `run_metta_helper` returns
        `[]` the same as a query that legitimately found nothing, and a caller cannot tell a blocked form from
        a true empty answer. `petta_sandbox_refuse/1` records the refused name in a side channel, `nb_setval`,
        that `catch/3` cannot unwind — checked here after every `_helper` call, success or failure, so a
        swallowed refusal still surfaces as `_RefusedBuiltin` instead of a silent `[]`. (A program that
        catches the refusal itself with MeTTa's `catch` is refused the same way: the note is still there.)"""
        row = self._janus.query_once("petta_sandbox_last_refusal(N, W)")
        if row.get("truth"):
            self._janus.query_once("petta_sandbox_clear_refusal")
            raise _RefusedBuiltin(str(row["N"]), str(row["W"]))

    def _helper(self, text: str) -> Optional[list[str]]:
        """PeTTa's own `_run_helper`, except that a goal that FAILS is returned as None instead of [].

        Whether a `!` form in `text` runs sandboxed is not decided here: `process_form/4` is wrapped inside
        the engine (`_install_petta_sandbox`), so every `!` form PeTTa evaluates is, wherever it came from."""
        self._janus.query_once("petta_sandbox_clear_refusal")
        try:
            answer = self._janus.query_once(
                "run_metta_helper(Verbose, HelperName, Argument, Results)",
                {"Verbose": "false", "HelperName": "process_metta_string", "Argument": text},
            )
        finally:
            # A refusal recorded during the call above wins over whatever `answer` says, including a
            # PrologError already propagating out of the `try` — see `_raise_if_refused`'s docstring.
            self._raise_if_refused()
        if not answer.get("truth"):                 # janus answers {"truth": False} to a goal that failed
            return None
        return [str(x) for x in answer.get("Results", [])]

    def load(self, kb_text: str) -> None:
        """Load `kb_text` in ONE `process_metta_string` call, so PeTTa's own declaration and clause-body
        prepass ("every function type declaration in the file is visible to every definition in it,
        independent of order") sees the whole text. A top-level `!` form in it — a caller's `extra_atoms` can
        hold one, and PeTTa evaluates it during the load like any expression — is run sandboxed by the
        engine's `process_form/4` wrapper, not by anything this method does: it neither splits the text nor
        drops a fragment it cannot read (an unbalanced tail is PeTTa's own syntax error, as at HEAD)."""
        if self._helper(kb_text) is None:
            raise RuntimeError("the load goal failed without raising an error")

    def lock(self) -> None:
        """Keep `add-atom` and its siblings refused for the rest of this child's life. Called by a
        `run_program` child once its KB is loaded: nothing it evaluates from then on may change the space."""
        if not self._janus.query_once("petta_sandbox_lock").get("truth"):
            raise RuntimeError("could not lock the PeTTa space")

    def split_forms(self, text: str) -> list[tuple[str, str]]:
        """('bang' | 'fact', form text) for each top-level form of `text`, as PeTTa's OWN reader splits it
        (comments removed; a 'bang' form's text lacks its `!`). A text the reader cannot split raises its
        syntax error instead of losing the part it could not read."""
        answer = self._janus.query_once("petta_sandbox_forms(Text, Pairs)", {"Text": text})
        if not answer.get("truth"):
            raise RuntimeError("PeTTa's reader could not split the text")
        return [(str(kind), str(form)) for kind, form in answer["Pairs"]]

    def run(self, text: str) -> list[str]:
        out = self._helper(text)
        return [] if out is None else out


EngineFactory = Callable[[str, int], Any]


# ── the child programs ────────────────────────────────────────────────────────

def _send(conn: Connection, message: dict) -> None:
    conn.send_bytes(json.dumps(message).encode("utf-8"))


def _error_message(stage: str, exc: BaseException, index: Optional[int] = None) -> dict:
    """One `stage` always wins over whatever stage the caller passed: `_RefusedBuiltin` is reported as
    `stage="refused"` whether it surfaced while loading the KB or while evaluating an expression, so
    `PettaRuntime._error_from`/`PettaSession` can map every refusal to `MettaRuntimeError(kind="refused")`
    the same way regardless of where in the run it was caught."""
    if isinstance(exc, _RefusedBuiltin):
        stage = "refused"
    return {"kind": "error", "stage": stage, "index": index,
            "type": type(exc).__name__, "message": _clip(f"{exc}")}


def _program_child(conn: Connection, factory: EngineFactory, petta_home: str, stack_limit_mb: int,
                   kb_text: str, expressions: list[str], deadline_s: Optional[float],
                   parent_pid: int, after_load: Optional[Callable[[Any], Optional[dict]]] = None,
                   memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB) -> None:
    """The forked child of `PettaRuntime.run_program`: one load, one call per expression, one message."""
    try:
        _child_prepare(deadline_s, parent_pid=parent_pid, keep_fds=(conn.fileno(),), memory_limit_mb=memory_limit_mb)
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
    # ---- the chainer hook (PLN_CHAINER=pln): after the load, before the lock; see "The chainer hook" above ----
    chain: Optional[dict] = None
    chain_s = 0.0
    if after_load is not None:
        tc = time.perf_counter()
        try:
            chain = after_load(engine)
        except BaseException as exc:            # noqa: BLE001 - a chainer that did not finish fails the request
            _send(conn, _error_message("chain", exc))
            return
        chain_s = time.perf_counter() - tc
    # ---- end of the chainer hook ----
    try:
        engine.lock()                           # from here on nothing a program does may change the space
    except BaseException as exc:                # noqa: BLE001
        _send(conn, _error_message("start", exc))
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
    stats = {"engine_s": round(t1 - t0, 4), "load_s": round(t2 - t1 - chain_s, 4),
             "queries_s": round(t3 - t2, 4), "peak_rss_mb": _peak_rss_mb()}
    if after_load is not None:
        stats["chain_s"] = round(chain_s, 4)
        stats["chain"] = chain
    _send(conn, {"kind": "result", "groups": groups, "stats": stats})


def _run_text(engine: Any, text: str, tiers: Mapping[str, str]) -> tuple[list[list[str]], int]:
    """Evaluate `text` form by form: one group per `!` form; consecutive facts go in as one chunk, with their
    stored calls reduced. Returns (groups, number of calls reduced).

    The forms are the ones PeTTa's own reader finds (`engine.split_forms`), not this module's: a splitter of
    its own that disagrees with the engine about where a form ends would drop or merge text without a word,
    and PeTTa raises its syntax error for text it cannot read. Whether a `!` form runs sandboxed is the
    engine's business (see `_install_petta_sandbox`), so a plain fact needs no flag from here."""
    groups: list[list[str]] = []
    reduced = 0
    pending: list[str] = []                      # the facts waiting to be added

    def flush() -> None:
        nonlocal reduced
        if pending:
            reduction = reduce_stored_calls("\n".join(pending), tiers)
            reduced += reduction.reduced
            engine.run(reduction.text)
            pending.clear()

    for kind, form in engine.split_forms(text):
        if kind == "bang":
            flush()
            groups.append([str(x) for x in engine.run("!" + form)])
        else:
            pending.append(form)
    flush()
    return groups, reduced


def _session_child(conn: Connection, factory: EngineFactory, petta_home: str, stack_limit_mb: int,
                   kb_text: str, tiers: dict[str, str], deadline_s: Optional[float],
                   memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB) -> None:
    """The forked child of a PettaSession: load once, then answer requests until told to stop or orphaned."""
    try:
        _child_prepare(deadline_s, parent_pid=None, keep_fds=(conn.fileno(),), memory_limit_mb=memory_limit_mb)
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
                    deadline_s: Optional[float] = None, *,
                    after_load: Optional[Callable[[Any], Optional[dict]]] = None) -> list[list[str]]:
        """`deadline_s` is ignored: a GIL-holding Rust call cannot be preempted in process. The executor
        enforces the request deadline by killing the worker. `after_load` (the PeTTa chainer hook) is refused."""
        if after_load is not None:
            raise MettaRuntimeError("the PLN chainer hook needs PLN_RUNTIME=petta; hyperon has none",
                                    kind="unavailable")
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
                                    runtime.stack_limit_mb, reduction.text, self._tiers, budget,
                                    runtime.memory_limit_mb)
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
                 engine_factory: Optional[EngineFactory] = None, memory_limit_mb: Optional[int] = None) -> None:
        import config
        self.petta_home = Path(petta_home if petta_home is not None else config.PETTA_HOME)
        self.stack_limit_mb = int(stack_limit_mb if stack_limit_mb is not None else config.PETTA_STACK_LIMIT_MB)
        #: how far a child's resident set may grow, in MB (0: no limit); `PETTA_MEMORY_LIMIT_MB` unless given
        self.memory_limit_mb = resolve_memory_limit_mb(memory_limit_mb)
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
            child = _fork_child(target, *args)
        except OSError as exc:
            raise MettaRuntimeError(f"could not start a PeTTa process: {exc}", kind="unavailable",
                                    detail=str(exc)) from exc
        child.memory_limit_mb = self.memory_limit_mb
        return child

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
        if stage == "chain":
            # The chainer hook's own error text is written for a caller (core.pln_chainer.PlnChainerError); anything
            # else it raised is an internal failure, named by its type. A PlnInputError is the PROGRAM's fault (a fact that
            # cannot be chained, a graph too dense to derive): kind="refused", which the runner answers as a 422.
            if message.get("type") == "PlnInputError":
                return MettaRuntimeError(f"the PLN chainer refused the program: {text}", kind="refused", detail=text)
            return MettaRuntimeError(f"the PLN chainer could not finish: {text}" if message.get("type") == "PlnChainerError"
                                     else f"the PLN chainer failed: {kind_of}", kind="chain", detail=text)
        if stage == "refused":
            # `text` is `_RefusedBuiltin`'s own message: the refused head's name and nothing else (no
            # filesystem path, no Prolog internals) — safe to show a caller verbatim, so it is used as-is
            # rather than wrapped in `kind_of`'s "<type>: <text>" framing.
            return MettaRuntimeError(text, kind="refused", detail=text)
        index = message.get("index")
        where = f"expression {index + 1}" if isinstance(index, int) else "the program"
        return MettaRuntimeError(f"PeTTa raised while evaluating {where}: {kind_of}", kind="expression", detail=text)

    # -- the seam
    def run_program(self, kb_text: str, expressions: Sequence[str],
                    deadline_s: Optional[float] = None, *,
                    after_load: Optional[Callable[[Any], Optional[dict]]] = None) -> list[list[str]]:
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
                           reduction.text, exprs, budget, os.getpid(), after_load, self.memory_limit_mb)
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
    """Forget the cached instances (tests that change PETTA_HOME, the stack limit or the memory limit)."""
    with _RUNTIMES_LOCK:
        _RUNTIMES.clear()
