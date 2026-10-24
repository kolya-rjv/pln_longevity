"""Tests for the MeTTa runtime seam (core/metta_runtime.py) and its three callers.

Three layers, cheapest first:

* the pure functions: the result normaliser, the stored-call reducer, the PeTTa refusal list, the form
  splitter, the runtime selection, `available()` (which must import neither janus nor petta);
* the PeTTa supervision logic against a TEST DOUBLE of the engine. The child is forked, so the double is
  injected through `PettaRuntime(engine_factory=...)`; nothing in production is test-aware. A scripted
  engine sleeps, spins, crashes, raises and writes to its stdout so the deadline kill, the crash report, the
  output redirection, zombie and descriptor hygiene and the orphan backstop can be checked without PeTTa;
* the REAL PeTTa (`-m petta`, skipped when SWI-Prolog, janus or the pinned checkout is missing): the five
  smoke queries of pln_deduction.metta section 5 with their written answers, the patient and LinAge2 stacks
  against hyperon, a fresh interpreter per program, `run_query_parts` over two stacks, the deadline on an
  endless recursion, sessions, and the executor's daemonic worker forking a PeTTa child.

No test calls a model.

    pytest tests/test_metta_runtime.py -q              # everything that can run here
    pytest tests/test_metta_runtime.py -q -m petta     # only the real-PeTTa tests
    pytest tests/test_metta_runtime.py -q -m "not petta"
"""
from __future__ import annotations

import gc
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

import config  # noqa: E402
import core.metta_runtime as mr  # noqa: E402
import core.pln_runner as pln_runner  # noqa: E402
from core.metta_runtime import (  # noqa: E402
    HyperonRuntime,
    MettaRuntimeError,
    PETTA_ALLOWED_FUN_NAMES,
    PETTA_KNOWN_FUN_NAMES,
    PettaRuntime,
    get_runtime,
    normalise_result,
    petta_refused_forms,
    read_tier_table,
    reduce_stored_calls,
    reset_runtime_cache,
    runtime_available_for,
    selected_runtime_name,
    split_top_level_forms,
)
from core.metta_validator import validate  # noqa: E402
from ontology.registry import OntologyRegistry  # noqa: E402


# ══════════════════════════════ helpers ═══════════════════════════════════════

def _real_petta_available() -> bool:
    """Not just `available()` (file + `find_spec` checks only): a real one-off start, so a STALE or
    PARTIAL checkout skips every `-m petta` test instead of making each one fail. `available()` alone
    cannot tell the two apart — confirmed: a scratch `PETTA_HOME` with empty `src/main.pl`,
    `python/helper.pl` and `python/petta/__init__.py` (the three marker files `available()` checks) still
    reports available, and `runtime_ready` in `/health` would read true while every real request is a
    502. One real fork+run costs about 0.15 s, paid once, here, at collection time."""
    if not PettaRuntime().available():
        return False
    try:
        return PettaRuntime().run_program("", ["!(+ 1 2)"]) == [["3"]]
    except Exception:
        return False


REAL_PETTA = _real_petta_available()
requires_petta = pytest.mark.skipif(
    not REAL_PETTA,
    reason="PeTTa is not installed here: needs SWI-Prolog, janus-swi and the checkout at "
           f"{config.PETTA_HOME} (bash scripts/setup_petta.sh; see API.md 'The PeTTa runtime')",
)


requires_hyperon = pytest.mark.skipif(not HyperonRuntime().available(), reason="hyperon is not installed")


def real_petta(fn):
    """A test that runs the real PeTTa: marked `petta`, skipped when it cannot run."""
    return pytest.mark.petta(requires_petta(fn))


_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[()]|[^\s()"]+')


def same_answer(a: str, b: str, rel: float = 1e-9) -> bool:
    """Two result strings agree token by token; numbers compare by value (3 == 3.0)."""
    ta, tb = _TOKEN.findall(a), _TOKEN.findall(b)
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if not math.isclose(fx, fy, rel_tol=rel, abs_tol=1e-12):
            return False
    return True


def _bag(atom: str) -> list[str]:
    """The tokens of an answer with numbers canonicalised, sorted: equal bags = same content in any order."""
    out = []
    for tok in _TOKEN.findall(atom):
        try:
            out.append(format(float(tok), ".9g"))
        except ValueError:
            out.append(tok)
    return sorted(out)


def assert_same_groups(petta: list[list[str]], hyperon: list[list[str]], *, ordered: bool = True) -> None:
    """Per expression, the same answers. ordered=False tolerates what the two engines legitimately differ in: PeTTa
    lists a collapse in assertion order where hyperon often lists it in reverse (`(SupportedBy (CRP DNAmGDF15))`
    against `(SupportedBy (DNAmGDF15 CRP))`), so the content is compared, not the sequence."""
    assert len(petta) == len(hyperon), (petta, hyperon)
    for p, h in zip(petta, hyperon):
        assert len(p) == len(h), (p, h)
        if ordered:
            assert all(same_answer(x, y) for x, y in zip(p, h)), (p, h)
        else:
            assert sorted(map(_bag, p)) == sorted(map(_bag, h)), (p, h)


#: Background threads of earlier tests (anyio, httpx) can open or close a descriptor while one of these tests runs.
#: A leak in the runtime is one or two descriptors PER RUN, and every test below loops, so it is far past this.
FD_SLACK = 2


def open_fds() -> int:
    for d in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    pytest.skip("no way to count open descriptors here")


def process_is_gone(pid: int) -> bool:
    """True once `pid` is neither running nor a zombie waiting to be reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():                                       # Linux: a zombie answers kill(pid, 0)
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        except (OSError, IndexError):
            return False
    return False


def process_is_reaped(pid: int) -> bool:
    """True once `pid` has actually been reaped by someone's `os.waitpid` — stricter than
    `process_is_gone`, whose Linux branch counts an UNREAPED zombie as "gone" (matching an orphan
    reparented to an init that may never get around to collecting it). Every "no zombie"/"reaped"/"closed"
    assertion wants this one instead, since an unreaped zombie left behind by our own code is exactly the
    bug such a test is checking for; the two orphan tests keep `process_is_gone`, on purpose.

    Production code always reaps through its OWN `os.waitpid` before a test ever gets here, so the
    expected, correct outcome is that THIS call finds no such child left to wait for at all
    (`ChildProcessError`). If it instead succeeds — whether because the pid is still running (`(0, 0)`) or
    because it was an unreaped zombie this very call just had to collect (`(pid, status)`) — that is the
    bug: a lingering zombie our own check would otherwise paper over by reaping it on the test's behalf.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True                      # nothing left to wait for: already reaped
    return False


def wait_until(predicate, timeout: float, step: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


# ══════════════════════ the test double of the PeTTa engine ═══════════════════

class ScriptedEngine:
    """Stands in for the PeTTa engine inside the forked child. The expression says what it does."""

    def __init__(self, options: dict) -> None:
        self.options = options
        self.kb = ""
        self.facts: list[str] = []
        #: the text of every `.run()` call, in order — lets a test check what `_run_text` sends the engine
        self.calls: list[str] = []
        self.locked = False
        if options.get("start_error"):
            raise ImportError("No module named 'janus_swi' (test double)")
        if options.get("pidfile"):
            Path(options["pidfile"]).write_text(str(os.getpid()))

    def load(self, kb_text: str) -> None:
        if self.options.get("load_error"):
            raise ValueError("Syntax error: failed to process form (test double)")
        self.kb = kb_text

    def lock(self) -> None:
        self.locked = True

    def split_forms(self, text: str) -> list[tuple[str, str]]:
        """What `_PettaEngine.split_forms` answers (PeTTa's own reader), for the forms the double understands."""
        return [("bang", text[s + 1:e]) if text[s] == "!" else ("fact", text[s:e])
                for s, e in split_top_level_forms(text)]

    def run(self, text: str) -> list[str]:
        self.calls.append(text)
        text = text.strip()
        if not text.startswith("!"):
            self.facts.append(text)
            return []
        body = text[2:-1].strip()                           # "!(echo x y)" -> "echo x y"
        verb, _, arg = body.partition(" ")
        if verb == "echo":
            return [arg]
        if verb == "kb":
            return [self.kb]
        if verb == "facts":
            return list(self.facts)
        if verb == "remember":
            self.facts.append(arg)
            return []
        if verb == "pid":
            return [str(os.getpid())]
        if verb == "sleep":
            time.sleep(float(arg))
            return []
        if verb == "spin":
            while True:
                pass
        if verb == "crash":
            os._exit(int(arg))
        if verb == "raise":
            raise RuntimeError("boom: expression failed (test double)")
        if verb == "noise":                                 # what PeTTa's traces do: write below Python
            os.write(1, b"NOISE-ON-FD-1\n")
            os.write(2, b"NOISE-ON-FD-2\n")
            print("NOISE-ON-SYS-STDOUT")
            print("NOISE-ON-SYS-STDERR", file=sys.stderr)
            return []
        if verb == "bool":
            return ["true", "false", "(f true)", '"true"', "trueish", "(x False)"]
        if verb == "many":
            return [f"(r {i})" for i in range(int(arg))]
        if verb == "grow":                                  # take `arg` MB of the heap and touch every page of it
            self.ballast = bytearray(int(arg) * 1024 * 1024)
            for offset in range(0, len(self.ballast), 1024):
                self.ballast[offset] = 1
            return [str(len(self.ballast) >> 20)]
        raise AssertionError(f"the double does not know {text!r}")


def scripted(**options):
    """An engine factory for PettaRuntime(engine_factory=...). The closure survives the fork."""
    def factory(petta_home: str, stack_limit_mb: int) -> ScriptedEngine:
        engine = ScriptedEngine(options)
        engine.petta_home, engine.stack_limit_mb = petta_home, stack_limit_mb
        return engine
    return factory


@pytest.fixture
def fake(tmp_path):
    """A PettaRuntime whose child runs the scripted double. Deadline 20 s unless a test says otherwise."""
    def make(**options) -> PettaRuntime:
        return PettaRuntime(petta_home=tmp_path, stack_limit_mb=64, engine_factory=scripted(**options))
    return make


# ═══════════════════════════ the result normaliser ════════════════════════════

@pytest.mark.parametrize("raw, expected", [
    ("true", "True"),
    ("false", "False"),
    ("(witnessed true)", "(witnessed True)"),
    ("(a true (b false))", "(a True (b False))"),
    ('(Title "true or false" true)', '(Title "true or false" True)'),     # a string is data
    ('"true"', '"true"'),
    ("trueish", "trueish"),                                               # only a standalone token
    ("(is-false x)", "(is-false x)"),
    ("True", "True"),
    ("3", "3"),                                                           # numbers are NOT rewritten
    ("1.0e-05", "1.0e-05"),
    ("(stv 0.7 1.0)", "(stv 0.7 1.0)"),
    ("()", "()"),
    ("", ""),
])
def test_the_normaliser_spells_booleans_like_hyperon_and_touches_nothing_else(raw, expected):
    assert normalise_result(raw) == expected


def test_the_normaliser_is_idempotent():
    once = normalise_result('(a true (b "false" false))')
    assert normalise_result(once) == once


# ═══════════════════════════ the stored-call reducer ══════════════════════════

TIERS = "(= (evidence-confidence RCT_Human) 1.0)\n(= (evidence-confidence Epidemiological) 0.6)\n"


def test_a_stored_call_in_a_fact_becomes_the_tiers_number():
    kb = TIERS + "(Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))\n"
    red = reduce_stored_calls(kb)
    assert red.text == TIERS + "(Effect A B Pos (stv 0.7 1.0))\n"
    assert red.reduced == 1 and red.unresolved == ()


def test_the_number_is_the_tables_own_text_so_it_parses_to_the_same_float():
    kb = "(= (evidence-confidence T1) 0.65)\n(F (stv 0.5 (evidence-confidence T1)))\n"
    assert "(F (stv 0.5 0.65))" in reduce_stored_calls(kb).text


def test_several_calls_in_one_fact_and_several_facts_are_all_reduced():
    kb = (TIERS + "(Two (evidence-confidence RCT_Human) (evidence-confidence Epidemiological))\n"
          "(One (stv 0.1 (evidence-confidence Epidemiological)))\n")
    red = reduce_stored_calls(kb)
    assert "(Two 1.0 0.6)" in red.text and "(One (stv 0.1 0.6))" in red.text
    assert red.reduced == 3


def test_the_table_is_read_from_the_whole_text_wherever_the_equation_stands():
    kb = "(F (stv 0.5 (evidence-confidence T1)))\n(= (evidence-confidence T1) 0.65)\n"     # used before defined
    assert "(F (stv 0.5 0.65))" in reduce_stored_calls(kb).text


def test_an_unknown_tier_is_left_alone_and_reported():
    kb = TIERS + "(F (stv 0.5 (evidence-confidence NoSuchTier)))\n(G (stv 0.5 (evidence-confidence RCT_Human)))\n"
    red = reduce_stored_calls(kb)
    assert "(F (stv 0.5 (evidence-confidence NoSuchTier)))" in red.text
    assert "(G (stv 0.5 1.0))" in red.text
    assert red.reduced == 1 and red.unresolved == ("NoSuchTier",)


def test_a_tier_defined_twice_with_different_numbers_is_ambiguous_and_left_alone():
    kb = ("(= (evidence-confidence T1) 0.5)\n(= (evidence-confidence T1) 0.6)\n"
          "(F (stv 1 (evidence-confidence T1)))\n")
    assert read_tier_table(kb) == {}
    red = reduce_stored_calls(kb)
    assert red.text == kb and red.reduced == 0 and red.unresolved == ("T1",)
    same = "(= (evidence-confidence T1) 0.5)\n(= (evidence-confidence T1) 0.50)\n(F (evidence-confidence T1))\n"
    assert read_tier_table(same) == {"T1": "0.5"}


def test_a_variable_or_computed_tier_is_not_a_stored_call_it_can_resolve():
    kb = TIERS + "(F (evidence-confidence $t))\n(G (evidence-confidence (tier-of X)))\n"
    red = reduce_stored_calls(kb)
    assert "(F (evidence-confidence $t))" in red.text and "(G (evidence-confidence (tier-of X)))" in red.text
    assert red.reduced == 0


def test_comments_and_strings_are_skipped():
    kb = (TIERS
          + "; (Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))\n"
          + '(Note "see (evidence-confidence RCT_Human) in the paper; really")\n'
          + "(Real (stv 0.7 (evidence-confidence RCT_Human))) ; trailing (evidence-confidence RCT_Human)\n")
    red = reduce_stored_calls(kb)
    assert "; (Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))" in red.text
    assert '(Note "see (evidence-confidence RCT_Human) in the paper; really")' in red.text
    assert "(Real (stv 0.7 1.0)) ; trailing (evidence-confidence RCT_Human)" in red.text
    assert red.reduced == 1


def test_rules_constants_declarations_and_expressions_are_left_to_the_engine():
    kb = (TIERS
          + "(= (tier-conf $t) (evidence-confidence $t))\n"
          + "(= (best-conf) (evidence-confidence RCT_Human))\n"
          + "(: evidence-confidence (-> EvidenceCategory Number))\n"
          + "!(evidence-confidence RCT_Human)\n"
          + "!(add-atom &self (F (evidence-confidence RCT_Human)))\n")
    red = reduce_stored_calls(kb)
    assert red.text is kb and red.reduced == 0          # nothing to do: the very same object


def test_text_without_a_stored_call_comes_back_unchanged():
    for kb in ("", "(Foo a b)\n", TIERS, "(= (f $x) $x)\n!(f 1)\n"):
        red = reduce_stored_calls(kb)
        assert red.text is kb and red.reduced == 0 and red.unresolved == ()


def test_a_tier_table_from_earlier_text_serves_text_added_later():
    table = read_tier_table(TIERS)
    red = reduce_stored_calls("(F (stv 0.5 (evidence-confidence Epidemiological)))\n", table)
    assert red.text == "(F (stv 0.5 0.6))\n" and red.reduced == 1


def test_the_repos_own_knowledge_base_has_no_stored_call_left_after_reduction():
    """Every runtime stack, as the app assembles it, reduces completely: no fact keeps a call and no tier is unknown."""
    pytest.importorskip("fastapi")
    import api
    names = {p.name: p for p in api._runtime_kb_paths()}
    names.update({p.name: p for p in pln_runner.linage2_patient_kb()})
    kb = "\n".join(p.read_text(encoding="utf-8") for p in names.values())
    assert "(evidence-confidence " in kb
    red = reduce_stored_calls(kb)
    assert red.reduced > 0 and red.unresolved == ()
    for s, e in split_top_level_forms(red.text):
        form = red.text[s:e]
        if form.startswith("!") or form.startswith("(=") or form.startswith("(:"):
            continue
        assert "(evidence-confidence " not in re.sub(r'"(?:[^"\\]|\\.)*"|;[^\n]*', "", form), form


# ═══════════════════════════ the form splitter ════════════════════════════════

def test_the_splitter_finds_top_level_forms_and_keeps_the_bang():
    text = '; header (not a form)\n(Foo a)\n!(bar (baz 1)) (Qux "a ) b") ; (c)\n!(last)'
    assert [text[s:e] for s, e in split_top_level_forms(text)] == [
        "(Foo a)", "!(bar (baz 1))", '(Qux "a ) b")', "!(last)"]


def test_the_splitter_ignores_bare_tokens_and_an_unbalanced_tail():
    assert split_top_level_forms("foo (a) bar (b") == [(4, 7)]


# ═══════════════════════════ what PeTTa is not given ══════════════════════════

REFUSED_PROGRAMS = {
    "add-atom": "!(add-atom &self (Foo a))",
    "remove-atom": "!(remove-atom &self (Foo a))",
    "=": "(= (f $x) $x)",
    "assert": "!(assert (== 1 1))",
    "test": "!(test 1 1)",
    "halt": "!(halt)",
    "import!": "!(import! &self mylib)",
    "git-import!": "!(git-import! \"https://example.invalid/x.git\")",
    "cached-import!": "!(cached-import! &self mylib)",
    "include": "!(include mylib)",
    "py-call": '!(py-call (os.system "ls"))',
    "py-atom": "!(py-atom os.system)",
    "py-dot": "!(py-dot os system)",
    "shell": "!(shell ls)",
    "load_metta_file": '!(load_metta_file "x.metta")',
    "load_metta_file_cached": '!(load_metta_file_cached "x.metta")',
    "process_metta_string": '!(process_metta_string "(Foo a)")',
    "new-space": "!(new-space)",
    "bind!": "!(bind! &x (new-space))",
    # PeTTa's own "call any Prolog predicate by name" doors (metta.pl's "Prolog interop" section): the same
    # shape of escape as py-call, reachable without ever writing the name of what they end up calling.
    "call": "!(call (shell ls))",
    "callPredicate": "!(callPredicate (Predicate (shell ls)))",
    "assertaPredicate": "!(assertaPredicate (Predicate (foo a)))",
    "assertzPredicate": "!(assertzPredicate (Predicate (foo a)))",
    "retractPredicate": "!(retractPredicate (Predicate (foo a)))",
    "import_prolog_function": "!(import_prolog_function foo)",
    "Predicate": "!(Predicate (foo a))",
    # The verifier's own finding: fun/1-registered, absent from every refusal list before this change.
    # Registering a symbol as a translator hook makes PeTTa call it directly, by name, during translation.
    "add-translator-rule!": "!(add-translator-rule! myHook)",
    "remove-translator-rule!": "!(remove-translator-rule! myHook)",
    "change-state!": "!(change-state! &x 1)",
    "get-state": "!(get-state &x)",
    "library": "!(library mylib)",
    "library-import!": "!(library-import! &self mylib)",
    "argv": "!(argv 0)",
    # `call`'s sibling: the compiler form that runs a goal built from the program's own data with no
    # function check (the independent review found it on no list, and the scan the only barrier to it).
    "translatePredicate": "!(translatePredicate (foo 1))",
    # PeTTa calls these four itself on every load and query; a program that names them gets a host
    # predicate called by name (`maplist getenv`, the review's reproduction), a parse or a file oracle.
    "maplist": '!(maplist getenv ("HOME"))',
    "foldl": '!(foldl getenv ("HOME") 0)',
    "sread": '!(sread "(a b)")',
    "exists_file": '!(exists_file "/etc/hosts")',
    # Parses text into a term: with the output bound first (`let`), the way to write a `partial(F, Args)`
    # closure for any F, which `apply_fn1`/`apply_fn2`/.../`apply_fnN` call without checking it is a function.
    "repra": '!(repra (a b))',
}


@pytest.mark.parametrize("name", sorted(REFUSED_PROGRAMS))
def test_every_refused_form_is_found(name):
    assert name in petta_refused_forms(REFUSED_PROGRAMS[name])


def test_the_refusal_list_is_the_documented_one():
    spec = {"add-atom", "remove-atom", "=", "assert", "test", "halt", "import!", "git-import!", "include",
            "py-call", "py-atom", "py-dot", "shell", "load_metta_file", "load_metta_file_cached",
            "process_metta_string", "new-space", "bind!"}
    assert spec <= set(mr.PETTA_REFUSED_FORMS)
    assert set(REFUSED_PROGRAMS) == set(mr.PETTA_REFUSED_FORMS)        # and the table above covers all


@pytest.mark.parametrize("program, names", [
    ("!(let $f add-atom ($f &self (Foo a)))", ["add-atom"]),               # by value, not as a head
    ("!(collapse (remove-atom &self (Foo a)))", ["remove-atom"]),         # a nested head
    ("!(superpose ((add-atom &self a) (shell ls)))", ["add-atom", "shell"]),
    ("!(= 1 1)", ["="]),                                                   # a top-level = is a definition, with or without !
])
def test_a_refused_form_is_found_wherever_it_stands(program, names):
    assert petta_refused_forms(program) == sorted(names)


@pytest.mark.parametrize("program", [
    "!(match &self (= (evidence-confidence $e) $c) (if (> $c 0.7) (pair $e $c) (empty)))",   # taught few-shot shape
    "!(match &self (Effect $a $b $s $t) $t)",
    '!(match &self (Title $p "add-atom, (= x), assert and halt in a string") $p)',
    "!(match &self (Foo $x) $x) ; add-atom in a comment\n",
    "!(infer &self CellularSenescence CoronaryHeartDisease)",
    "!(match &self (HasTest $x Test) $x)",                                 # `Test` is not `test`
    "!(match &self (Foo $test) $test)",                                    # a variable
    "!(match &self (Foo testing included) 1)",                             # only whole tokens
    "",
])
def test_a_program_that_only_reads_is_not_refused(program):
    assert petta_refused_forms(program) == []


# ─── petta_refused_forms_in_kb: the same scan, minus the top-level `=` check ───

@pytest.mark.parametrize("name", sorted(set(REFUSED_PROGRAMS) - {"="}))
def test_the_kb_scan_still_finds_every_other_refused_form(name):
    assert name in mr.petta_refused_forms_in_kb(REFUSED_PROGRAMS[name])


def test_the_kb_scan_does_not_refuse_a_top_level_equals():
    """`=` is how an ordinary KB defines every one of its functions — `petta_refused_forms` (used for an
    expression or a caller's own query text) refuses it as a top-level definition; `petta_refused_forms_in_kb`
    (used for `kb_text`/a session's kb and text — the trusted ontology files plus the caller's `extra_atoms`,
    already concatenated together by the time `PettaRuntime.run_program` sees them) must not, or every
    ordinary load would be refused."""
    assert mr.petta_refused_forms_in_kb("(= (f $x) $x)") == []
    assert mr.petta_refused_forms_in_kb("!(= 1 1)") == []


@pytest.mark.parametrize("path", sorted((REPO).glob("*.metta")))
def test_no_tracked_metta_file_is_refused_as_kb_text(path):
    """Every file this repo's stacks are assembled from must load under PLN_RUNTIME=petta: confirms the KB
    scan added for the kb_text gap (finding 2) does not turn an ordinary load into a 422/`kind="refused"`."""
    assert mr.petta_refused_forms_in_kb(path.read_text(encoding="utf-8")) == []


def test_a_literal_refused_form_hidden_among_ordinary_facts_is_still_found_in_kb_text():
    assert mr.petta_refused_forms_in_kb(
        "(Foo a)\n(= (sq $x) (* $x $x))\n!(add-atom &self (Hacked yes))\n(Bar b)"
    ) == ["add-atom"]


def _few_shot_queries() -> list[str]:
    import json
    examples = json.loads((PLN_CHAT / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8"))
    return [ex["metta_query"] for ex in examples if ex.get("metta_query")]


@pytest.mark.parametrize("query", _few_shot_queries())
def test_every_taught_few_shot_query_passes_the_petta_refusal(query):
    """The refusal check is token-level and strict (any symbol spelled exactly like a refused form is
    refused, even as plain data) — the taught examples must keep passing it as new ones are added, since
    this is exactly where a false positive would first be seen by a real user."""
    assert petta_refused_forms(query) == []


# ═══════════════════════════ the validator under both runtimes ════════════════

def _issues(program: str, runtime: str | None) -> list[str]:
    return validate(program, OntologyRegistry(), runtime=runtime).issues


@pytest.mark.parametrize("name", sorted(REFUSED_PROGRAMS))
def test_under_petta_the_validator_refuses_the_form(name):
    result = validate(REFUSED_PROGRAMS[name], OntologyRegistry(), runtime="petta")
    assert result.valid is False
    assert any(i.startswith("Form(s) not allowed under the PeTTa runtime:") and name in i for i in result.issues)


@pytest.mark.parametrize("name", sorted(REFUSED_PROGRAMS))
def test_under_hyperon_the_validator_does_not_change_at_all(name, monkeypatch):
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    program = REFUSED_PROGRAMS[name]
    explicit, implicit = validate(program, OntologyRegistry(), runtime="hyperon"), validate(program, OntologyRegistry())
    assert explicit == implicit                                    # the default runtime is hyperon
    assert explicit.valid is True and explicit.issues == []        # exactly what it said before the PeTTa check existed


def test_under_petta_a_read_only_program_validates_as_it_does_under_hyperon():
    program = "!(match &self (= (evidence-confidence $e) $c) (pair $e $c))"
    assert validate(program, OntologyRegistry(), runtime="petta") == validate(program, OntologyRegistry(), runtime="hyperon")


def test_the_validator_follows_pln_runtime_when_it_is_not_told(monkeypatch):
    program = "!(add-atom &self (Foo a))"
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    assert not validate(program, OntologyRegistry()).valid
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    assert validate(program, OntologyRegistry()).valid


def test_the_check_covers_the_atoms_injected_beside_the_query(monkeypatch):
    """/metta/run validates `validation_text(injected, query)`: a definition smuggled in extra_atoms is refused too."""
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    text = "(= (baseline-risk-chd $a $s) 0.999)\n!(patient-baseline &self P)"
    assert any("=" in i for i in _issues(text, None))


def test_the_check_covers_the_atoms_injected_beside_the_query_through_the_real_api(monkeypatch):
    """The test above calls `validate()` on a string that already contains the `(= ...)` — it would pass
    even if `/metta/run` stopped passing `extra_atoms` to that check. This calls the real handler, the way
    an HTTP request reaches it, and checks the same thing through it."""
    pytest.importorskip("fastapi")
    import api
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    with pytest.raises(api.HTTPException) as err:
        api.metta_run(api.MettaRunRequest(
            metta_query="!(+ 1 2)", extra_atoms="(= (f $x) $x)", allow_definitions=False))
    assert err.value.status_code == 422
    assert "definition" in str(err.value.detail).lower()

    # allow_definitions=True reaches the generic validator instead, which still refuses `=` under petta.
    with pytest.raises(api.HTTPException) as err:
        api.metta_run(api.MettaRunRequest(
            metta_query="!(+ 1 2)", extra_atoms="(= (f $x) $x)", allow_definitions=True))
    assert err.value.status_code == 422
    assert "Form(s) not allowed under the PeTTa runtime" in str(err.value.detail)

    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    # add-atom is refused under petta only; confirms the 422 above is PeTTa-specific, not generic rejection
    # of every definition in extra_atoms.
    assert api.metta_run(api.MettaRunRequest(
        metta_query="!(+ 1 2)", extra_atoms="(= (f $x) $x)", allow_definitions=True)).metta_query == "!(+ 1 2)"


@real_petta
def test_a_blocked_form_surfaces_as_an_error_over_http_not_an_empty_answer(monkeypatch, tmp_path):
    """The real engine, over the real HTTP path (`httpx.ASGITransport`, the pattern `tests/test_api.py`
    uses) — not a double, not an in-process call to `core.metta_runtime` directly.

    `extra_atoms` declares `atom_concat`, the stand-in head name and the constant the canary uses: exactly
    how the verifier's own reproduction got a validator-"unknown symbol" 422 out of the way, so the query
    reaches the real `PettaRuntime` — where this task's engine-level fix, not the validator, is what
    refuses it. Before that fix, this dynamically-constructed `add-atom` call returned an EMPTY, "ok"-ish
    answer indistinguishable from a query that legitimately found nothing; here it must come back as a
    clear HTTP error naming the refused head."""
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    import asyncio
    from unittest.mock import Mock
    import api
    import core.executor as executor_module

    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)   # run inline, in this test process
    monkeypatch.setattr(api, "log_http_request", Mock())
    reset_runtime_cache()

    extra_atoms = (
        "(: atom_concat (-> String String Atom))\n"
        "(: HttpRefusalCanary (-> Atom Atom))\n"
        "(: yes Atom)"
    )
    query = '!(let $f (atom_concat "add" "-atom") ($f &self (HttpRefusalCanary yes)))'

    async def post() -> "httpx.Response":
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/metta/run", json={"metta_query": query, "extra_atoms": extra_atoms})

    try:
        response = asyncio.run(post())
    finally:
        reset_runtime_cache()

    # a caller's own program asked for something the runtime does not run: 422, not the 502 of a service fault
    assert response.status_code == 422, response.text
    body = response.json()
    detail = body.get("detail", body)
    message = detail.get("message", "") if isinstance(detail, dict) else str(detail)
    assert isinstance(detail, dict) and detail.get("code") == "refused_form", body
    assert "add-atom" in message and "refus" in message.lower(), body


def test_the_other_checks_still_run_under_petta(monkeypatch):
    issues = _issues("!(add-atom &self (Foo a)", "petta")
    assert "Unbalanced parentheses in MeTTa query." in issues
    assert any("not allowed under the PeTTa runtime" in i for i in issues)


# ═══════════════════════════ runtime selection ════════════════════════════════

def test_hyperon_is_the_runtime_named_hyperon():
    assert get_runtime("hyperon").name == "hyperon" and isinstance(get_runtime("hyperon"), HyperonRuntime)


def test_get_runtime_reads_pln_runtime_and_caches_one_instance_per_process(monkeypatch):
    reset_runtime_cache()
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    first = get_runtime()
    assert isinstance(first, PettaRuntime) and first.name == "petta"
    assert get_runtime() is first and get_runtime("petta") is first
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    assert get_runtime().name == "hyperon" and get_runtime("petta") is first
    reset_runtime_cache()
    assert get_runtime("petta") is not first


def test_auto_is_hyperon_when_it_is_importable_else_petta(monkeypatch):
    monkeypatch.setattr(mr, "_hyperon_available", lambda: True)
    assert selected_runtime_name("auto") == "hyperon"
    monkeypatch.setattr(mr, "_hyperon_available", lambda: False)
    monkeypatch.setattr(mr, "_petta_available", lambda home: True)
    assert selected_runtime_name("auto") == "petta"
    monkeypatch.setattr(mr, "_petta_available", lambda home: False)
    assert selected_runtime_name("auto") == "hyperon"          # neither: hyperon, and the app runs in stub mode
    assert runtime_available_for("auto", "/nowhere") is False


def test_an_unknown_runtime_name_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="PLN_RUNTIME"):
        get_runtime("prolog")
    with pytest.raises(ValueError, match="PLN_RUNTIME"):
        runtime_available_for("prolog")


def _config_in_subprocess(env: dict, expr: str) -> subprocess.CompletedProcess:
    code = f"import sys; sys.path.insert(0, {str(PLN_CHAT)!r}); import config; {expr}"
    full = {k: v for k, v in os.environ.items() if not k.startswith(("PLN_", "PETTA_"))}
    full["PYTHON_DOTENV_DISABLED"] = "1"                    # never read a .env from a test
    full.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=full, cwd=str(REPO), timeout=60)


def test_pln_runtime_available_reflects_the_selected_runtime(tmp_path):
    out = _config_in_subprocess({"PLN_RUNTIME": "petta", "PETTA_HOME": str(tmp_path / "nope")},
                                "print(config.PLN_RUNTIME, config.PLN_RUNTIME_AVAILABLE, config.PETTA_HOME)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[:2] == ["petta", "False"]               # no checkout there
    hyperon = _config_in_subprocess({}, "print(config.PLN_RUNTIME, config.PLN_RUNTIME_AVAILABLE)")
    assert hyperon.stdout.split() == ["hyperon", str(runtime_available_for("hyperon"))]
    forced = _config_in_subprocess({"PLN_RUNTIME_AVAILABLE": "false"}, "print(config.PLN_RUNTIME_AVAILABLE)")
    assert forced.stdout.strip() == "False"


def test_config_defaults_and_overrides_for_the_petta_settings(tmp_path):
    default = _config_in_subprocess({}, "print(config.PETTA_HOME, config.PETTA_STACK_LIMIT_MB)")
    assert default.stdout.split() == [str(REPO / ".petta" / "PeTTa"), "2048"]
    custom = _config_in_subprocess({"PETTA_HOME": str(tmp_path), "PETTA_STACK_LIMIT_MB": "512"},
                                   "print(config.PETTA_HOME, config.PETTA_STACK_LIMIT_MB)")
    assert custom.stdout.split() == [str(tmp_path), "512"]


def test_a_bad_pln_runtime_stops_the_service_at_start():
    out = _config_in_subprocess({"PLN_RUNTIME": "prolog"}, "pass")
    assert out.returncode != 0 and "PLN_RUNTIME must be hyperon, petta or auto" in out.stderr


def test_available_imports_neither_janus_nor_petta_nor_hyperon():
    """The parent must never load SWI-Prolog: forking after janus has started it is unsafe."""
    code = (
        f"import sys; sys.path.insert(0, {str(PLN_CHAT)!r})\n"
        "import config\n"
        "from core.metta_runtime import PettaRuntime, get_runtime, runtime_available_for, selected_runtime_name\n"
        "PettaRuntime().available(); get_runtime('petta').available(); runtime_available_for('auto')\n"
        "selected_runtime_name('auto'); get_runtime('hyperon').available()\n"
        "bad = [m for m in ('janus_swi', 'petta', 'hyperon') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "clean", out.stderr


def test_availability_needs_the_checkout_and_janus(tmp_path, monkeypatch):
    home = tmp_path / "PeTTa"
    assert PettaRuntime(petta_home=home).available() is False                  # nothing there
    (home / "src").mkdir(parents=True)
    (home / "python" / "petta").mkdir(parents=True)
    (home / "src" / "main.pl").write_text("")
    (home / "python" / "helper.pl").write_text("")
    assert PettaRuntime(petta_home=home).available() is False                  # no wrapper yet
    (home / "python" / "petta" / "__init__.py").write_text("")
    monkeypatch.setattr(mr.importlib.util, "find_spec",
                        lambda name, package=None: object() if name == "janus_swi" else None)
    assert PettaRuntime(petta_home=home).available() is True
    monkeypatch.setattr(mr.importlib.util, "find_spec", lambda name, package=None: None)
    assert PettaRuntime(petta_home=home).available() is False                  # janus missing


@pytest.mark.skipif(mr.importlib.util.find_spec("janus_swi") is None, reason="needs the real janus_swi")
def test_available_is_true_for_a_checkout_that_cannot_actually_start(tmp_path):
    """`available()` checks three marker files and `find_spec`, nothing more — a checkout with the right
    files but no real content reports available and then fails to start, rather than being filtered out
    as "not installed here". This is exactly why `-m petta` collection (`_real_petta_available`) does a
    real one-off start instead of trusting `available()` alone."""
    home = tmp_path / "PeTTa"
    (home / "src").mkdir(parents=True)
    (home / "python" / "petta").mkdir(parents=True)
    (home / "src" / "main.pl").write_text("")
    (home / "python" / "helper.pl").write_text("")
    (home / "python" / "petta" / "__init__.py").write_text("")
    rt = PettaRuntime(petta_home=home)
    assert rt.available() is True
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("", ["!(+ 1 2)"])
    assert err.value.kind == "unavailable"


# ═══════════════════════ the supervision logic, against the double ════════════

def test_run_program_returns_one_group_per_expression_normalised_with_the_raw_kept(fake):
    rt = fake()
    out = rt.run_program("(Foo a)", ["!(echo hello world)", "!(bool)", "!(many 3)", "!(echo )"])
    assert out == [["hello world"],
                   ["True", "False", "(f True)", '"true"', "trueish", "(x False)"],
                   ["(r 0)", "(r 1)", "(r 2)"],
                   [""]]
    assert rt.last_raw[1] == ["true", "false", "(f true)", '"true"', "trueish", "(x False)"]
    assert rt.last_raw[0] == ["hello world"]


def test_an_expression_without_a_bang_is_evaluated_like_pln_runner_does(fake):
    assert fake().run_program("", ["(echo x)", "  !(echo y) "]) == [["x"], ["y"]]


def test_no_expressions_means_no_process(fake):
    rt = fake()
    assert rt.run_program("(Foo a)", []) == [] and rt.last_pid is None


def test_the_child_gets_the_checkout_and_the_stack_limit(fake, tmp_path):
    rt = fake()
    assert rt.run_program("", ["!(echo ok)"]) == [["ok"]]
    assert rt.stack_limit_mb == 64 and rt.petta_home == tmp_path


def test_stored_calls_are_reduced_in_the_kb_text_the_child_loads_and_counted(fake):
    kb = TIERS + "(Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))\n(G (evidence-confidence Nope))\n"
    rt = fake()
    (loaded,), = rt.run_program(kb, ["!(kb)"])
    assert "(Effect A B Pos (stv 0.7 1.0))" in loaded and "(G (evidence-confidence Nope))" in loaded
    assert rt.reduced_calls == 1 and rt.unresolved_calls == ("Nope",)
    rt.run_program("(Plain fact)", ["!(echo x)"])
    assert rt.reduced_calls == 0 and rt.unresolved_calls == ()


def test_the_reducer_never_touches_an_expression(fake):
    expression = "!(echo (stv 0.7 (evidence-confidence RCT_Human)))"
    assert fake().run_program(TIERS, [expression]) == [["(stv 0.7 (evidence-confidence RCT_Human))"]]


def test_caller_atoms_are_part_of_the_kb_so_they_are_reduced_too(fake):
    """pln_runner appends extra_atoms to the KB text; that is the whole text the reducer sees."""
    kb = TIERS + "\n" + "(Caller_Fact X (stv 0.9 (evidence-confidence Epidemiological)))"
    (loaded,), = fake().run_program(kb, ["!(kb)"])
    assert loaded.endswith("(Caller_Fact X (stv 0.9 0.6))")


def test_a_failed_start_is_reported_as_unavailable_without_leaking_the_checkout_path(fake, tmp_path):
    """Regression for the finding that `PETTA_HOME` (the server's own filesystem path, including the
    owner's home directory and username) reached the HTTP caller verbatim on a start failure — and a start
    failure is not only misconfiguration: `os.fork()` refused under load raises the identical error, so a
    caller provoking that leaked the path too. The exception TYPE is still shown (useful, not sensitive);
    the detailed text (which, for a real missing-checkout FileNotFoundError, names the path again) is kept
    only on `.detail`, read by the server's own log, never by `str()`."""
    with pytest.raises(MettaRuntimeError) as err:
        fake(start_error=True).run_program("", ["!(echo x)"])
    assert err.value.kind == "unavailable"
    assert "could not be started" in str(err.value) and "ImportError" in str(err.value)
    assert str(tmp_path) not in str(err.value)
    assert "janus_swi" in err.value.detail


def test_any_exception_during_the_load_is_a_failed_load(fake):
    with pytest.raises(MettaRuntimeError) as err:
        fake(load_error=True).run_program("(Foo a)", ["!(echo x)"])
    assert err.value.kind == "load" and "could not load the knowledge base" in str(err.value)
    assert "Syntax error" in err.value.detail


def test_an_empty_kb_is_not_loaded(fake):
    assert fake(load_error=True).run_program("   \n", ["!(echo ok)"]) == [["ok"]]


def test_an_expression_that_raises_fails_the_run_and_names_it(fake):
    with pytest.raises(MettaRuntimeError) as err:
        fake().run_program("", ["!(echo fine)", "!(raise)", "!(echo never)"])
    assert err.value.kind == "expression"
    assert "expression 2" in str(err.value) and "boom" in str(err.value)


def test_a_child_that_dies_without_an_answer_is_a_crash_with_its_exit_status(fake):
    rt = fake()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("", ["!(crash 3)"])
    assert err.value.kind == "crash" and "exit code 3" in str(err.value)
    with pytest.raises(MettaRuntimeError) as halted:
        rt.run_program("", ["!(crash 1)"])
    assert halted.value.kind == "crash" and "halt" not in halted.value.detail and "`assert`" in str(halted.value)


def test_a_child_killed_by_a_signal_is_a_crash_naming_the_signal(fake, tmp_path):
    pidfile = tmp_path / "pid"
    rt = fake(pidfile=str(pidfile))

    def killer():
        wait_until(pidfile.exists, 10)
        time.sleep(0.2)
        os.kill(int(pidfile.read_text()), signal.SIGKILL)

    threading.Thread(target=killer, daemon=True).start()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("", ["!(sleep 30)"], deadline_s=20)
    assert err.value.kind == "crash" and "SIGKILL" in str(err.value)


def test_the_deadline_kills_the_child_and_says_so(fake):
    rt = fake()
    started = time.monotonic()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("", ["!(sleep 30)"], deadline_s=0.6)
    elapsed = time.monotonic() - started
    assert err.value.kind == "timeout" and "0.6 s deadline" in str(err.value)
    assert 0.5 < elapsed < 10, elapsed
    assert wait_until(lambda: process_is_reaped(rt.last_pid), 3)


def test_a_cpu_bound_child_is_killed_at_the_deadline_too(fake):
    rt = fake()
    started = time.monotonic()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("", ["!(spin)"], deadline_s=0.5)
    assert err.value.kind == "timeout" and time.monotonic() - started < 10
    assert wait_until(lambda: process_is_reaped(rt.last_pid), 3)


def test_the_default_deadline_is_the_configured_query_timeout(fake, monkeypatch):
    monkeypatch.setattr(config, "PLN_QUERY_TIMEOUT_SECONDS", 0.5)
    with pytest.raises(MettaRuntimeError, match="0.5 s deadline"):
        fake().run_program("", ["!(sleep 30)"])


def test_a_zero_deadline_means_no_deadline(fake, monkeypatch):
    monkeypatch.setattr(config, "PLN_QUERY_TIMEOUT_SECONDS", 0)
    assert fake().run_program("", ["!(sleep 0.3)", "!(echo done)"])[1] == ["done"]
    assert fake().run_program("", ["!(echo done)"], deadline_s=0) == [["done"]]


def test_the_children_output_never_reaches_the_parent(fake, capfd):
    capfd.readouterr()
    fake().run_program("", ["!(noise)", "!(echo after)"])
    seen = capfd.readouterr()
    assert "NOISE" not in seen.out and "NOISE" not in seen.err


@pytest.mark.parametrize("how", ["success", "load_error", "expression_error", "crash", "timeout"])
def test_no_zombie_is_left_behind(fake, how):
    rt = fake(load_error=(how == "load_error"))
    program = {"success": "!(echo x)", "load_error": "!(echo x)", "expression_error": "!(raise)",
               "crash": "!(crash 9)", "timeout": "!(sleep 30)"}[how]
    try:
        rt.run_program("(Foo a)", [program], deadline_s=0.5 if how == "timeout" else 20)
    except MettaRuntimeError:
        assert how != "success"
    assert rt.last_pid is not None and process_is_reaped(rt.last_pid), f"pid {rt.last_pid} is still there after {how}"


def test_no_descriptor_leaks_after_success_failure_and_timeout(fake):
    rt = fake()
    rt.run_program("", ["!(echo warm)"])                    # first use may open lazily-held descriptors
    before = open_fds()
    for _ in range(4):
        rt.run_program("", ["!(echo x)"])
        for program in ("!(raise)", "!(crash 2)"):
            with pytest.raises(MettaRuntimeError):
                rt.run_program("", [program])
        with pytest.raises(MettaRuntimeError):
            rt.run_program("", ["!(sleep 30)"], deadline_s=0.3)
    assert open_fds() <= before + FD_SLACK


def test_the_parent_never_imports_janus_or_petta(fake):
    fake().run_program("", ["!(echo x)"])
    assert "janus_swi" not in sys.modules and "petta" not in sys.modules


def test_each_run_is_a_new_process(fake):
    rt = fake()
    (pid1,), = rt.run_program("", ["!(pid)"])
    (pid2,), = rt.run_program("", ["!(pid)"])
    assert pid1 != pid2 and int(pid1) != os.getpid()


def test_several_threads_can_fork_at_once_without_mixing_up_pipes(fake):
    """`_FORK_LOCK` serialises "create the pipe, fork, close the child's end" so that two threads forking
    at once cannot hand each other's pipe ends to their children (a copy of a write end held by a
    stranger delays EOF). Inline mode (`PLN_WORKER_POOL_SIZE=0`) runs this fork from a FastAPI sync route,
    which Starlette serves from its thread pool, so this is genuinely exercised in production — not
    exercised by any other test, which mutation `fork_lock_off` survives."""
    before = open_fds()
    rt = fake()
    results: list[object] = [None] * 8
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            results[i] = rt.run_program("", [f"!(echo thread-{i})"])
        except BaseException as exc:                # noqa: BLE001 - surfaced to the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors, errors
    assert results == [[[f"thread-{i}"]] for i in range(8)]
    assert open_fds() <= before + FD_SLACK


def test_a_run_state_is_not_shared_between_programs(fake):
    rt = fake()
    assert rt.run_program("", ["!(remember one)", "!(facts)"]) == [[], ["one"]]
    assert rt.run_program("", ["!(facts)"]) == [[]]


def test_a_daemonic_worker_can_have_a_petta_child(fake, tmp_path):
    """The executor's workers are daemonic multiprocessing children, which may not start multiprocessing
    children: the runtime forks with os.fork."""
    rt = fake()
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    def worker(conn):
        try:
            conn.send(("ok", rt.run_program("(Foo a)", ["!(echo from-a-daemon)"])))
        except BaseException as exc:                         # noqa: BLE001
            conn.send(("error", f"{type(exc).__name__}: {exc}"))

    proc = ctx.Process(target=worker, args=(child,), daemon=True)
    proc.start()
    child.close()
    assert parent.poll(30)
    assert parent.recv() == ("ok", [["from-a-daemon"]])
    proc.join(10)


def test_a_petta_run_refuses_the_forms_the_validator_refuses(fake):
    rt = fake()
    for name, program in REFUSED_PROGRAMS.items():
        with pytest.raises(MettaRuntimeError) as err:
            rt.run_program("(Foo a)", ["!(echo ok)", program])
        assert err.value.kind == "refused" and "expression 2" in str(err.value) and name in str(err.value)


def test_a_petta_run_also_refuses_a_refused_form_hidden_in_kb_text(fake):
    """Regression for the finding that `run_program` scanned every expression but never `kb_text` itself —
    `kb_text` is not attacker-free (it carries the caller's own `extra_atoms`, `core/pln_runner.py`), and
    PeTTa executes any top-level `!` form found anywhere in it during the load, the same as an expression.
    The scan used for `kb_text` must not flag an ordinary top-level `=` (every `.metta` file has many), so
    this checks that a `(= ...)` fact loads fine while a literal refused form next to it does not."""
    rt = fake()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("(Foo a)\n(= (sq $x) (* $x $x))\n!(add-atom &self (Hacked yes))", ["!(echo ok)"])
    assert err.value.kind == "refused"
    assert "knowledge base text" in str(err.value) and "add-atom" in str(err.value)
    assert rt.last_pid is None                                        # refused before any child was forked
    # the same kb_text, minus the smuggled form, loads and runs normally
    assert rt.run_program("(Foo a)\n(= (sq $x) (* $x $x))", ["!(echo ok)"]) == [["ok"]]


def test_an_orphaned_child_stops_by_itself(fake, tmp_path, monkeypatch):
    """The executor SIGKILLs a worker at its deadline. The worker's PeTTa child must not run on forever:
    it carries a CPU-time backstop (and on Linux asks to die with its parent)."""
    pidfile = tmp_path / "spinner.pid"
    monkeypatch.setattr(mr, "BACKSTOP_GRACE_S", 1.0)
    rt = fake(pidfile=str(pidfile))
    middle = os.fork()
    if middle == 0:                                          # the "worker"
        try:
            rt.run_program("", ["!(spin)"], deadline_s=1.0)
        finally:
            os._exit(0)
    try:
        assert wait_until(pidfile.exists, 15), "the child never started"
        spinner = int(pidfile.read_text())
        time.sleep(0.2)
        os.kill(middle, signal.SIGKILL)                      # the worker dies at its deadline: nobody reaps or kills the child
        os.waitpid(middle, 0)
        assert wait_until(lambda: process_is_gone(spinner), 15), "the orphaned child is still spinning"
    finally:
        try:
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass


# ═══════════════════ memory: a legal program must not take the host ═══════════
#
# The final security review: a ~900-byte program of `let*`/`atom_concat` doubles a string 22 times and takes 1 GB
# in half a second with every builtin it uses allowed, under a 64 MB Prolog stack limit (a string is on the heap,
# not on the stacks). The child had no memory limit of its own. The real-PeTTa version of this is in
# `tests/test_petta_sandbox.py`; these tests use the double, whose `grow` verb takes real memory.

def with_memory_limit(tmp_path, limit_mb, **options) -> PettaRuntime:
    return PettaRuntime(petta_home=tmp_path, stack_limit_mb=64, memory_limit_mb=limit_mb,
                        engine_factory=scripted(**options))


def test_a_child_that_takes_too_much_memory_is_stopped_and_reported(tmp_path):
    rt = with_memory_limit(tmp_path, 128)
    started = time.monotonic()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program("", ["!(grow 300)", "!(echo not-reached)"])
    assert err.value.kind == "memory" and "128 MB" in str(err.value), str(err.value)
    assert time.monotonic() - started < 10
    assert wait_until(lambda: process_is_reaped(rt.last_pid), 3)


def test_a_child_inside_the_limit_is_left_alone(tmp_path):
    rt = with_memory_limit(tmp_path, 400)
    assert rt.run_program("", ["!(grow 100)", "!(echo after)"]) == [["100"], ["after"]]
    assert rt.last_run["peak_rss_mb"] > 100


def test_the_limit_is_what_the_child_grows_by_not_what_the_server_already_held(tmp_path):
    """The child is a fork of a server that may hold hundreds of MB (on Linux a fork counts them in its own
    resident set): the limit is counted from the size the child has when it is prepared."""
    held = bytearray(300 * 1024 * 1024)
    for offset in range(0, len(held), 1024):
        held[offset] = 1
    assert with_memory_limit(tmp_path, 128).run_program("", ["!(grow 50)"]) == [["50"]]
    del held


def test_the_ceiling_is_the_size_at_preparation_plus_the_limit():
    """What the child already holds when `_child_prepare` runs (here 200 MB, on Linux a fork's share of the
    server) does not count against it: growing by 50 MB under a 128 MB limit is fine, growing by 200 MB is not."""
    def child_target(conn, growth_mb):
        held = bytearray(200 * 1024 * 1024)
        for offset in range(0, len(held), 1024):
            held[offset] = 1
        mr._child_prepare(None, parent_pid=None, keep_fds=(conn.fileno(),), memory_limit_mb=128)
        more = bytearray(growth_mb * 1024 * 1024)
        for offset in range(0, len(more), 1024):
            more[offset] = 1
        time.sleep(0.3)
        mr._send(conn, {"kind": "result", "growth": growth_mb})

    child = mr._fork_child(child_target, 50)
    try:
        assert child.wait_message(time.monotonic() + 20) == {"kind": "result", "growth": 50}
    finally:
        child.close()
    child = mr._fork_child(child_target, 200)
    child.memory_limit_mb = 128
    try:
        with pytest.raises(MettaRuntimeError) as err:
            child.wait_message(time.monotonic() + 20)
        assert err.value.kind == "memory"
    finally:
        child.close()


def test_a_garbage_collection_in_the_child_does_not_count_the_servers_heap_as_growth():
    """Measured on macOS: a forked child maps its parent's heap lazily, and its first full collection walks every
    inherited object, so a 400 MB server heap became 585 MB of the child's own resident set and an honest program
    was stopped under a 128 MB limit (the full suite at 128 MB failed one test that way). `gc.freeze()` in
    `_child_prepare` keeps the inherited objects out of the collector's reach. (On Linux a fork has the pages
    resident already and the baseline counts them, so this passes there either way.)"""
    heap = [(i, str(i), [i]) for i in range(1_000_000)]

    def child_target(conn):
        mr._child_prepare(None, parent_pid=None, keep_fds=(conn.fileno(),), memory_limit_mb=128)
        gc.collect()
        time.sleep(0.3)                                      # long enough for the watchdog to look many times
        mr._send(conn, {"kind": "result", "objects": len(heap)})

    child = mr._fork_child(child_target)
    try:
        assert child.wait_message(time.monotonic() + 60) == {"kind": "result", "objects": 1_000_000}
    finally:
        child.close()
        del heap


def test_a_limit_of_zero_means_no_limit(tmp_path):
    rt = with_memory_limit(tmp_path, 0)
    assert rt.memory_limit_mb == 0
    assert rt.run_program("", ["!(grow 300)"]) == [["300"]]


def test_a_session_child_is_stopped_too_and_its_session_is_closed(tmp_path):
    session = with_memory_limit(tmp_path, 128).session("")
    pid = session._child.pid
    with pytest.raises(MettaRuntimeError) as err:
        session.run("!(grow 300)")
    assert err.value.kind == "memory" and wait_until(lambda: process_is_reaped(pid), 3)
    with pytest.raises(MettaRuntimeError, match="closed"):
        session.run("!(echo x)")


def test_the_child_ends_itself_the_parent_only_reads_the_status():
    """The watchdog lives in the child (so an orphan has it too). Nothing here supervises the child but the
    pipe: it ends with the exit code the parent maps to `kind="memory"`."""
    def grower(conn, limit_mb):
        mr._child_prepare(None, parent_pid=None, keep_fds=(conn.fileno(),), memory_limit_mb=limit_mb)
        ballast = bytearray(300 * 1024 * 1024)
        for offset in range(0, len(ballast), 1024):
            ballast[offset] = 1
        time.sleep(30)

    child = mr._fork_child(grower, 128)
    child.memory_limit_mb = 128
    try:
        with pytest.raises(MettaRuntimeError) as err:
            child.wait_message(time.monotonic() + 20)
        assert err.value.kind == "memory" and "128 MB" in str(err.value)
        assert os.WIFEXITED(child.status) and os.WEXITSTATUS(child.status) == mr._EXIT_MEMORY
    finally:
        child.close()


def test_a_limit_of_zero_starts_no_watchdog():
    before = {t.name for t in threading.enumerate()}
    mr._start_memory_watchdog(0)
    assert {t.name for t in threading.enumerate()} == before


@pytest.mark.parametrize("env, expected", [
    (None, mr.DEFAULT_MEMORY_LIMIT_MB), ("", mr.DEFAULT_MEMORY_LIMIT_MB), ("2048", 2048), (" 700 ", 700),
    ("0", 0),                                               # switched off on purpose
    ("64", mr.MIN_MEMORY_LIMIT_MB),                         # too small to start a child at all: raised
    ("-5", mr.DEFAULT_MEMORY_LIMIT_MB), ("1GB", mr.DEFAULT_MEMORY_LIMIT_MB), ("1.5", mr.DEFAULT_MEMORY_LIMIT_MB),
])
def test_the_memory_limit_comes_from_PETTA_MEMORY_LIMIT_MB(monkeypatch, env, expected):
    """An unreadable setting falls back to the PROTECTED default, never to no protection."""
    if env is None:
        monkeypatch.delenv("PETTA_MEMORY_LIMIT_MB", raising=False)
    else:
        monkeypatch.setenv("PETTA_MEMORY_LIMIT_MB", env)
    assert mr.resolve_memory_limit_mb() == expected
    assert PettaRuntime().memory_limit_mb == expected


def test_an_unreadable_memory_limit_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("PETTA_MEMORY_LIMIT_MB", "1GB")
    with caplog.at_level("WARNING", logger=mr.__name__):
        mr.resolve_memory_limit_mb()
    assert "PETTA_MEMORY_LIMIT_MB" in caplog.text and "1GB" in caplog.text


def test_an_argument_wins_over_the_environment_and_a_negative_one_is_refused(monkeypatch):
    monkeypatch.setenv("PETTA_MEMORY_LIMIT_MB", "900")
    assert PettaRuntime(memory_limit_mb=300).memory_limit_mb == 300
    with pytest.raises(ValueError):
        PettaRuntime(memory_limit_mb=-1)


def test_the_default_limit_is_well_below_the_default_stack_limit_which_bounds_no_heap():
    """The stack limit (2048 MB unless the environment lowers it) does not bound the heap; the default memory limit
    is a few times the heaviest real knowledge base and ends a runaway before the stack limit would."""
    assert mr.MIN_MEMORY_LIMIT_MB <= mr.DEFAULT_MEMORY_LIMIT_MB <= 512


class _RecordedLimits:
    """`resource.setrlimit` as a recording double, for the Linux-only address-space wall: this suite also runs on
    a Mac, where the kernel refuses a finite RLIMIT_AS (EINVAL), so the calculation is checked against a double."""

    def __init__(self, monkeypatch, *, size=3 * 1024 ** 3, soft=None, hard=None, platform="linux"):
        import resource
        self.resource = resource
        self.calls = []
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(mr, "_address_space_bytes", lambda: size)
        monkeypatch.setattr(resource, "getrlimit", lambda res: (
            resource.RLIM_INFINITY if soft is None else soft, resource.RLIM_INFINITY if hard is None else hard))
        monkeypatch.setattr(resource, "setrlimit", lambda res, limits: self.calls.append((res, limits)))


def test_on_linux_the_kernel_is_asked_for_a_wall_above_the_watchdog(monkeypatch):
    recorded = _RecordedLimits(monkeypatch)
    mr._limit_address_space(256)
    wanted = 3 * 1024 ** 3 + (256 + mr._ADDRESS_SPACE_SLACK_MB) * 1024 * 1024
    assert recorded.calls == [(recorded.resource.RLIMIT_AS, (wanted, recorded.resource.RLIM_INFINITY))]
    assert wanted > 3 * 1024 ** 3 + 256 * 1024 * 1024               # the sampler's ceiling is below the wall


def test_the_address_space_wall_never_raises_a_limit_that_was_already_lower(monkeypatch):
    recorded = _RecordedLimits(monkeypatch, soft=2 * 1024 ** 3, hard=5 * 1024 ** 3)
    mr._limit_address_space(256)
    assert recorded.calls == [(recorded.resource.RLIMIT_AS, (2 * 1024 ** 3, 5 * 1024 ** 3))]
    recorded = _RecordedLimits(monkeypatch, hard=3 * 1024 ** 3 + 1)
    mr._limit_address_space(256)
    assert recorded.calls[0][1][0] == 3 * 1024 ** 3 + 1


@pytest.mark.parametrize("platform, limit, size", [
    ("darwin", 256, 3 * 1024 ** 3),                         # macOS answers a finite RLIMIT_AS with EINVAL
    ("linux", 0, 3 * 1024 ** 3),                            # no limit asked for
    ("linux", 256, None),                                   # no /proc: nothing to add the allowance to
])
def test_no_address_space_wall_where_it_cannot_be_set(monkeypatch, platform, limit, size):
    recorded = _RecordedLimits(monkeypatch, platform=platform, size=size)
    mr._limit_address_space(limit)
    assert recorded.calls == []


def test_a_kernel_that_refuses_the_address_space_wall_does_not_stop_the_child(monkeypatch):
    recorded = _RecordedLimits(monkeypatch)
    import resource

    def refuse(res, limits):
        raise ValueError("current limit exceeds maximum limit")
    monkeypatch.setattr(resource, "setrlimit", refuse)
    mr._limit_address_space(256)                            # no exception: the watchdog is still there
    assert recorded.calls == []


def test_the_child_limits_include_the_address_space_wall_and_the_documented_ones(monkeypatch):
    asked = []
    monkeypatch.setattr(mr, "_limit_address_space", lambda limit: asked.append(limit))
    import resource
    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)          # this is the test process, not a child
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    mr._limit_child_resources(300)
    assert asked == [300]
    assert dict(mr._CHILD_LIMITS)["RLIMIT_CORE"] == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/statm is Linux's")
def test_the_address_space_of_this_process_is_read_from_proc():
    assert mr._address_space_bytes() > 10 * 1024 * 1024
    assert mr._address_space_bytes() >= (mr._peak_rss_bytes() or 0) // 2


def test_the_peak_resident_set_is_a_number_of_bytes():
    peak = mr._peak_rss_bytes()
    assert peak and 5 * 1024 * 1024 < peak < 64 * 1024 ** 3
    assert abs(mr._peak_rss_mb() - peak / 1e6) < 50



# ═══════════════════════ sessions, against the double ═════════════════════════

def test_a_session_keeps_its_state_between_runs(fake):
    with fake().session("(Fact one)") as session:
        assert session.run("!(facts)") == [[]]               # the KB went through load(), not run()
        assert session.run("(Fact two)\n(Fact three)") == []
        assert session.run("!(facts)") == [["(Fact two)\n(Fact three)"]]


def test_a_session_refuses_the_forms_the_validator_refuses_before_sending_anything_to_the_child(fake):
    """Regression for the finding that `PettaSession.run()` performed no refusal check of any kind — the
    class is exported and nothing stops a future caller from constructing one with caller-controlled text.
    No message is sent to the child at all: the session is still usable afterward."""
    with fake().session("(Foo a)") as session:
        for name, program in REFUSED_PROGRAMS.items():
            if name == "=":
                continue                              # a session's own kb/text may define functions; see below
            with pytest.raises(MettaRuntimeError) as err:
                session.run(program)
            assert err.value.kind == "refused" and "session text" in str(err.value) and name in str(err.value)
        assert session.run("!(facts)") == [[]]        # the session never saw any of the refused calls


def test_a_session_may_define_a_function_with_a_top_level_equals_in_its_text(fake):
    """`=` is refused only as a CALLER-SUPPLIED redefinition risk (`core.metta_validator`); a session's own
    text defining a new function is the documented, intentional use `PettaSession` exists for ("for tests
    and tools"), so the session-text scan (`petta_refused_forms_in_kb`) must not refuse it."""
    with fake().session("") as session:
        assert session.run("(= (sq $x) (* $x $x))") == []


def test_a_session_answers_one_group_per_bang_form(fake):
    with fake().session("") as session:
        out = session.run("(Fact a)\n!(echo 1)\n(Fact b)\n!(echo 2)\n!(echo 3)")
        assert out == [["1"], ["2"], ["3"]]
        assert session.last_raw == out
        assert session.run("; nothing but a comment") == []


def test_a_session_normalises_and_keeps_the_raw_strings(fake):
    with fake().session("") as session:
        assert session.run("!(bool)")[0][:2] == ["True", "False"]
        assert session.last_raw[0][:2] == ["true", "false"]


def test_a_session_reduces_stored_calls_in_the_kb_and_in_facts_it_is_given(fake):
    with fake().session(TIERS + "(F (stv 1 (evidence-confidence RCT_Human)))") as session:
        assert session.reduced_calls == 1
        session.run("(G (stv 1 (evidence-confidence Epidemiological)))")
        assert session.reduced_calls == 2
        assert session.run("!(facts)") == [["(G (stv 1 0.6))"]]


def test_run_text_sends_each_bang_form_alone_and_the_facts_between_them_together():
    """`_run_text` is what gives a session its one-group-per-`!`-form contract (PeTTa hands back one flat list
    per call). Whether a `!` form runs sandboxed is no longer decided here — the engine's `process_form/4`
    wrapper does it for every `!` form, wherever it came from (checked against the real engine below)."""
    engine = ScriptedEngine({})
    groups, _ = mr._run_text(engine, "(Fact a)\n(Fact b)\n!(echo 1)\n(Fact c)\n!(echo 2)", {})
    assert groups == [["1"], ["2"]]
    assert engine.calls == ["(Fact a)\n(Fact b)", "!(echo 1)", "(Fact c)", "!(echo 2)"]


def test_a_session_survives_an_expression_that_raises(fake):
    with fake().session("") as session:
        with pytest.raises(MettaRuntimeError, match="boom"):
            session.run("!(raise)")
        assert session.run("!(echo still-here)") == [["still-here"]]


def test_a_session_whose_load_fails_raises_and_leaves_nothing_behind(fake):
    rt = fake(load_error=True)
    before = open_fds()
    with pytest.raises(MettaRuntimeError) as err:
        rt.session("(Foo a)")
    assert err.value.kind == "load" and open_fds() <= before + FD_SLACK


def test_a_session_deadline_kills_it(fake):
    session = fake().session("")
    pid = session._child.pid
    with pytest.raises(MettaRuntimeError) as err:
        session.run("!(sleep 30)", deadline_s=0.5)
    assert err.value.kind == "timeout" and wait_until(lambda: process_is_reaped(pid), 3)
    with pytest.raises(MettaRuntimeError, match="closed"):
        session.run("!(echo x)")


def test_a_session_that_crashes_is_closed_and_reaped(fake):
    session = fake().session("")
    pid = session._child.pid
    with pytest.raises(MettaRuntimeError) as err:
        session.run("!(crash 1)")
    assert err.value.kind == "crash" and wait_until(lambda: process_is_reaped(pid), 3)


def test_closing_a_session_reaps_its_child_and_is_idempotent(fake):
    before = open_fds()
    session = fake().session("")
    pid = session._child.pid
    session.close()
    session.close()
    assert process_is_reaped(pid) and open_fds() <= before + FD_SLACK
    with pytest.raises(MettaRuntimeError, match="closed"):
        session.run("!(echo x)")


def test_a_session_whose_engine_cannot_start_says_so(fake):
    with pytest.raises(MettaRuntimeError) as err:
        fake(start_error=True).session("")
    assert err.value.kind == "unavailable"


# ═══════════════════════ pln_runner on top of the seam ════════════════════════

class RecordingRuntime:
    """Stands in for a runtime inside pln_runner: records what it was given, answers what it was told to."""
    name = "recording"
    enforces_deadline = True

    def __init__(self, groups=None, error: Exception | None = None):
        self.groups, self.error, self.calls = groups or [], error, []

    def available(self) -> bool:
        return True

    def run_program(self, kb_text, expressions, deadline_s=None):
        self.calls.append({"kb_text": kb_text, "expressions": list(expressions), "deadline_s": deadline_s})
        if self.error:
            raise self.error
        return self.groups


@pytest.fixture
def recording(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")

    def install(**kw) -> RecordingRuntime:
        runtime = RecordingRuntime(**kw)
        monkeypatch.setattr(pln_runner, "get_runtime", lambda name=None: runtime)
        return runtime
    return install


def test_the_runner_gives_the_runtime_the_assembled_kb_and_the_banged_expressions(recording, tmp_path):
    runtime = recording(groups=[["(stv 0.9 0.8)"], []])
    one, two = tmp_path / "one.metta", tmp_path / "two.metta"
    one.write_text("(Foo a)\n", encoding="utf-8")
    two.write_text("(Bar b)", encoding="utf-8")
    result = pln_runner._runtime_run("(f 1) !(g 2)\n(h)", 0.0, [one, tmp_path / "missing.metta", two], "(Extra x)")
    (call,) = runtime.calls
    assert call["kb_text"] == "(Foo a)\n\n(Bar b)\n(Extra x)"          # the files joined by newlines, then extra_atoms
    assert call["expressions"] == ["!(f 1)", "!(g 2)", "!(h)"]
    assert (result.status, result.mode, result.runtime_name) == ("ok", "runtime", "recording")
    assert [(r.atom, r.expr_index) for r in result.results] == [("(stv 0.9 0.8)", 0)]
    assert result.results[0].stv == {"strength": 0.9, "confidence": 0.8}


def test_the_threshold_still_filters_what_the_runtime_returned(recording):
    recording(groups=[["(stv 0.9 0.8)", "(stv 0.9 0.1)"]])
    result = pln_runner._runtime_run("!(f)", 0.5, [], None)
    assert [r.atom for r in result.results] == ["(stv 0.9 0.8)"]


def test_no_answer_is_empty_not_an_error(recording):
    recording(groups=[[]])
    assert pln_runner._runtime_run("!(f)", 0.0, [], None).status == "empty"


def test_a_runtime_error_becomes_a_runtime_error_result_with_its_text(recording):
    recording(error=MettaRuntimeError("PeTTa exceeded the 60 s deadline and was killed", kind="timeout"))
    result = pln_runner._runtime_run("!(f)", 0.0, [], None)
    assert (result.status, result.error_code, result.mode) == ("error", "runtime_error", "runtime")
    assert result.error == "PeTTa exceeded the 60 s deadline and was killed"
    assert result.runtime_name == "recording" and not result.ok


def test_a_refused_program_is_a_refused_form_not_a_runtime_error(recording):
    """The program asked for something the PeTTa runtime does not run: its author's fault, a 422 at the HTTP
    layer. A crash or a deadline stays a `runtime_error` (502)."""
    recording(error=MettaRuntimeError("the PeTTa runtime refused to run 'match'", kind="refused"))
    result = pln_runner._runtime_run("!(f)", 0.0, [], None)
    assert (result.status, result.error_code) == ("error", "refused_form")
    assert result.error == "the PeTTa runtime refused to run 'match'"
    recording(error=MettaRuntimeError("the PeTTa process ended with signal 9", kind="crash"))
    assert pln_runner._runtime_run("!(f)", 0.0, [], None).error_code == "runtime_error"


def test_any_exception_the_runtime_raises_is_reported_not_propagated(recording):
    recording(error=OSError("disk on fire"))
    result = pln_runner._runtime_run("!(f)", 0.0, [], None)
    assert (result.status, result.error_code, result.error) == ("error", "runtime_error", "disk on fire")


def test_an_empty_query_never_reaches_the_runtime(recording):
    runtime = recording()
    assert pln_runner._runtime_run("  ; just a comment\n", 0.0, [], None).status == "empty"
    assert runtime.calls == []


def test_the_old_name_still_works():
    assert pln_runner._hyperon_run is pln_runner._runtime_run


def test_patching_the_old_name_actually_intercepts_run_query(monkeypatch):
    """Regression for the finding that `_hyperon_run` was kept as an alias "so a test that patches it keeps
    working", but `run_query` called `_runtime_run` directly — patching the alias silently did nothing."""
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    calls = []

    def patched(metta_query, confidence_threshold, kb_files, extra_atoms=None):
        calls.append(metta_query)
        return pln_runner.PLNRunResult(status="ok", results=[], mode="runtime")

    monkeypatch.setattr(pln_runner, "_hyperon_run", patched)
    pln_runner.run_query("!(+ 1 2)")
    assert calls == ["!(+ 1 2)"]


def test_run_query_parts_gives_every_part_the_time_that_is_left_of_one_budget(recording, monkeypatch):
    runtime = recording(groups=[["(a)"]])
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(pln_runner, "PLN_QUERY_TIMEOUT_SECONDS", 30.0)
    real_run = runtime.run_program

    def slow(kb_text, expressions, deadline_s=None):
        time.sleep(0.3)
        return real_run(kb_text, expressions, deadline_s)
    runtime.run_program = slow
    parts = [{"metta_query": "!(f)"}, {"metta_query": "!(g)"}, {"metta_query": "!(h)"}]
    results = pln_runner.run_query_parts(parts)
    budgets = [c["deadline_s"] for c in runtime.calls]
    assert len(results) == 3 and all(r.status == "ok" for r in results)
    assert budgets[0] <= 30.0 and budgets[0] > 29.0
    assert budgets[0] > budgets[1] > budgets[2]                                # one clock, not three
    assert 0.5 < budgets[0] - budgets[2] < 5                                  # about the 0.6 s the first two parts took


def test_run_query_parts_without_a_timeout_passes_no_deadline(recording, monkeypatch):
    runtime = recording(groups=[["(a)"]])
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(pln_runner, "PLN_QUERY_TIMEOUT_SECONDS", 0)
    pln_runner.run_query_parts([{"metta_query": "!(f)"}, {"metta_query": "!(g)"}])
    assert [c["deadline_s"] for c in runtime.calls] == [None, None]


def test_a_part_that_starts_after_the_budget_is_used_up_is_an_error_not_a_hang(recording, monkeypatch):
    runtime = recording(groups=[["(a)"]])
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(pln_runner, "PLN_QUERY_TIMEOUT_SECONDS", 1.0)
    real_run = runtime.run_program

    def slow(kb_text, expressions, deadline_s=None):
        time.sleep(1.3)                                                        # the first part eats the whole budget
        return real_run(kb_text, expressions, deadline_s)
    runtime.run_program = slow
    first, second = pln_runner.run_query_parts([{"metta_query": "!(f)"}, {"metta_query": "!(g)"}])
    assert first.status == "ok"
    assert second.status == "error" and second.error_code == "runtime_error" and "time budget" in second.error
    assert len(runtime.calls) == 1                                             # the second part never reached the runtime


def test_a_runtime_that_cannot_enforce_a_deadline_is_not_given_one(recording, monkeypatch):
    """hyperon keeps running a late part, as it always did: the executor's kill is its deadline."""
    runtime = recording(groups=[["(a)"]])
    runtime.enforces_deadline = False
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(pln_runner, "PLN_QUERY_TIMEOUT_SECONDS", 0.01)
    real_run = runtime.run_program

    def slow(kb_text, expressions, deadline_s=None):
        time.sleep(0.05)
        return real_run(kb_text, expressions, deadline_s)
    runtime.run_program = slow
    results = pln_runner.run_query_parts([{"metta_query": "!(f)"}, {"metta_query": "!(g)"}])
    assert [r.status for r in results] == ["ok", "ok"]
    assert [c["deadline_s"] for c in runtime.calls] == [None, None]
    assert HyperonRuntime.enforces_deadline is False and PettaRuntime.enforces_deadline is True


def test_the_shared_deadline_does_not_leak_out_of_run_query_parts(recording, monkeypatch):
    runtime = recording(groups=[["(a)"]])
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    pln_runner.run_query_parts([{"metta_query": "!(f)"}])
    pln_runner._runtime_run("!(g)", 0.0, [], None)
    assert runtime.calls[-1]["deadline_s"] is None


def test_merged_results_remember_which_runtime_answered():
    a = pln_runner.PLNRunResult(status="ok", mode="runtime", runtime_name="petta",
                                results=[pln_runner.PLNAtomResult("(a)", expr_index=0)])
    b = pln_runner.PLNRunResult(status="empty", mode="runtime", runtime_name="petta")
    assert pln_runner.merge_run_results([a, b]).runtime_name == "petta"


# ═══════════════════════ hyperon is exactly what it was ═══════════════════════

HYPERON_KB = """
(= (evidence-confidence RCT_Human) 1.0)
(= (double $x) (* 2 $x))
(Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))
(Foo a b)
(Foo c d)
"""
HYPERON_QUERIES = ["!(double 21)", "!(match &self (Foo $x $y) ($x $y))", "!(== 1 1)", "!(unknown-fn 1)",
                   "!(match &self (Effect A B Pos $t) $t)", "!(match &self (Nope $x) $x)"]


@requires_hyperon
def test_the_hyperon_runtime_returns_what_metta_run_returns():
    from hyperon import MeTTa
    metta = MeTTa()
    metta.run(HYPERON_KB)
    direct = [[str(atom) for atom in group] for group in metta.run("\n".join(HYPERON_QUERIES))]
    assert HyperonRuntime().run_program(HYPERON_KB, HYPERON_QUERIES) == direct
    assert direct[0] == ["42"] and direct[2] == ["True"]


@requires_hyperon
def test_the_hyperon_runtime_ignores_the_deadline_and_adds_a_missing_bang():
    assert HyperonRuntime().run_program("(= (f) 1)", ["(f)"], deadline_s=0.001) == [["1"]]


@requires_hyperon
def test_a_hyperon_failure_carries_the_engines_own_text():
    from hyperon import MeTTa
    with pytest.raises(Exception) as direct:
        MeTTa().run("(Foo a")
    with pytest.raises(MettaRuntimeError) as err:
        HyperonRuntime().run_program("(Foo a", ["!(Foo $x)"])
    assert str(err.value) == str(direct.value) and err.value.kind == "runtime"


@requires_hyperon
def test_a_hyperon_session_is_a_metta_object_with_string_results():
    with HyperonRuntime().session("(Foo a)") as session:
        out = session.run("(Foo b)\n!(match &self (Foo $x) $x)\n!(match &self (Nope $x) $x)")
        assert len(out) == 2 and sorted(out[0]) == ["a", "b"] and out[1] == []
        assert session.last_raw == out
        session.close()
        with pytest.raises(MettaRuntimeError, match="closed"):
            session.run("!(Foo $x)")


@requires_hyperon
def test_the_runner_on_hyperon_matches_an_inline_metta_run(tmp_path, monkeypatch):
    """_runtime_run on the default runtime == the old _hyperon_run: same atoms, same expr_index, same STV."""
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    from hyperon import MeTTa
    kb = tmp_path / "kb.metta"
    kb.write_text(HYPERON_KB, encoding="utf-8")
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    result = pln_runner._runtime_run("(match &self (Effect A B Pos $t) $t) !(double 21)", 0.0, [kb], None)
    metta = MeTTa()
    metta.run(HYPERON_KB)
    expected = [[str(a) for a in g] for g in metta.run("!(match &self (Effect A B Pos $t) $t)\n!(double 21)")]
    assert [(r.atom, r.expr_index) for r in result.results] == [
        (atom, i) for i, group in enumerate(expected) for atom in group]
    assert result.runtime_name == "hyperon" and result.status == "ok"


def test_health_reports_the_runtime_and_keeps_every_other_key(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    import api
    try:
        body = api.health()
        for key in ("status", "version", "api_key_required", "rate_limit_per_minute", "pln_mode", "runtime_importable",
                    "runtime_ready", "runtime_kb_file_count", "openai_key_configured", "available_models",
                    "drugage_build_available", "pln_execution"):
            assert key in body, key
        assert body["runtime_name"] in ("hyperon", "petta")
        assert body["runtime_name"] == get_runtime().name
        assert body["pln_mode"] == ("runtime" if api.PLN_RUNTIME_AVAILABLE else "stub")
        assert body["runtime_importable"] == get_runtime().available()
        assert "runtime_name" in api.HealthResponse.model_fields

        monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
        body = api.health()
        assert body["runtime_name"] == "petta" and body["runtime_importable"] == PettaRuntime().available()
        monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
        assert api.health()["runtime_name"] == "hyperon"

        # Both engines are available in THIS environment, so every assertion above would pass just the
        # same if `runtime_importable` were wired to hyperon's own availability regardless of the
        # selected runtime — this is the case that actually tells the two apart (finding: the test could
        # not distinguish the selected engine's availability from hyperon's).
        monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
        monkeypatch.setattr(config, "PETTA_HOME", tmp_path / "nope")
        reset_runtime_cache()
        body = api.health()
        assert body["runtime_name"] == "petta"
        assert body["runtime_importable"] is False and body["runtime_ready"] is False

        # "auto" with hyperon unavailable must report petta (and not silently read as "no runtime at all").
        monkeypatch.setattr(config, "PLN_RUNTIME", "auto")
        monkeypatch.setattr(config, "PETTA_HOME", config.ONTOLOGY_DIR / ".petta" / "PeTTa")   # the real one
        monkeypatch.setattr(mr, "_hyperon_available", lambda: False)
        reset_runtime_cache()
        body = api.health()
        assert body["runtime_name"] == "petta" and body["runtime_importable"] is True
    finally:
        reset_runtime_cache()


# ═══════════════════════ the real PeTTa ═══════════════════════════════════════

#: pln_deduction.metta section 5: the load order and the written answers
SMOKE_FILES = ("system_types", "logical_predicates", "epistemic_calibration", "grim_age_core",
               "grim_age_lu2019_evidence", "evidence_calibration", "hallmarks_core",
               "hallmarks_lopezotin2023_intervention_evidence", "mechanistic_bridges", "pln_deduction")
SMOKE = [
    ("!(link-effect &self DNAmPAI1 CoronaryHeartDisease)", ["(signed Pos (stv 0.6 0.85))"]),
    ("!(infer &self CellularSenescence CoronaryHeartDisease)", ["(signed Pos (stv 0.357 0.29089125))"]),
    ("!(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)", ["(signed Neg (stv 0.2499 0.17017138125))"]),
    ("!(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)",
     ["(Chain (DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease) "
      "(signed Neg (stv 0.2499 0.17017138125)))"]),
    ("!(infer &self DasatinibPlusQuercetin AllCauseMortality)", []),               # no false chain
]


def _kb_text(paths) -> str:
    return "\n".join(Path(p).read_text(encoding="utf-8") for p in paths)


@pytest.fixture(scope="module")
def petta():
    return PettaRuntime()


@pytest.fixture(scope="module")
def smoke_kb() -> str:
    return _kb_text(REPO / f"{name}.metta" for name in SMOKE_FILES)


@pytest.fixture(scope="module")
def runtime_paths():
    pytest.importorskip("fastapi")
    import api
    return api._runtime_kb_paths()


@real_petta
def test_petta_known_fun_names_matches_the_live_registry(petta):
    """Drift detection, not a safety gate: `_install_petta_sandbox` never trusts `PETTA_KNOWN_FUN_NAMES` at
    run time (it reads the live `fun/1` facts itself, so a name this constant has never seen is still
    guarded, by construction — see that function's docstring). This exists so an upstream PeTTa update
    that renames, drops or adds a `fun/1` name is caught here, for review, rather than only ever showing
    up as "this checkout's documented reachable-heads table is stale".

    Out of process, like every other "import janus directly" check in this file: the parent must never
    import janus or petta (`test_the_parent_never_imports_janus_or_petta`), so this starts a FRESH
    process to ask the live engine, the same way `_config_in_subprocess` asks a fresh process about
    `config` rather than importing it here."""
    code = (
        f"import sys; sys.path.insert(0, {str(petta.petta_home / 'python')!r}); "
        "import janus_swi as janus; from petta import PeTTa; "
        f"PeTTa(verbose=False, petta_path={str(petta.petta_home)!r}); "
        "janus.consult('dump_fun_names', data='dump_fun_names(Out) :- findall(N, fun(N), Out).'); "
        "import json; print(json.dumps(sorted(str(n) for n in janus.query_once('dump_fun_names(Out)')['Out'])))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    live = set(json.loads(proc.stdout))
    assert live == PETTA_KNOWN_FUN_NAMES
    assert PETTA_ALLOWED_FUN_NAMES <= PETTA_KNOWN_FUN_NAMES


@real_petta
def test_petta_answers_the_five_smoke_queries_of_pln_deduction_as_written(petta, smoke_kb):
    out = petta.run_program(smoke_kb, [q for q, _ in SMOKE])
    assert len(out) == 5
    for (query, expected), got in zip(SMOKE, out):
        assert len(got) == len(expected), (query, got)
        assert all(same_answer(g, e) for g, e in zip(got, expected)), (query, got, expected)
    assert petta.reduced_calls > 0 and petta.unresolved_calls == ()


@real_petta
@requires_hyperon
def test_petta_and_hyperon_agree_on_the_smoke_queries(petta, smoke_kb):
    queries = [q for q, _ in SMOKE]
    assert_same_groups(petta.run_program(smoke_kb, queries), HyperonRuntime().run_program(smoke_kb, queries))


@real_petta
def test_without_the_reducer_a_stored_call_breaks_the_arithmetic(petta, monkeypatch):
    """The reducer is what makes the KB run: PeTTa does not evaluate a call stored in a fact."""
    kb = TIERS + "(Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))"
    query = "!(match &self (Effect A B Pos (stv $s $c)) (* 2 $c))"
    assert petta.run_program(kb, [query]) == [["2.0"]]
    monkeypatch.setattr(mr, "reduce_stored_calls", lambda text, tiers=None: mr.Reduction(text))
    try:
        out = petta.run_program(kb, [query])
    except MettaRuntimeError as exc:
        assert exc.kind == "expression"
    else:
        assert out != [["2.0"]]


@real_petta
@requires_hyperon
def test_the_patient_stack_answers_on_petta_as_on_hyperon(petta, runtime_paths):
    kb = _kb_text(pln_runner.patient_stack(runtime_paths))
    queries = ["!(patient-z &self Patient001 CRP)", "!(diagnose-patient &self Patient001)"]
    got = petta.run_program(kb, queries)
    assert got[0] == ["1.7"]
    expected = HyperonRuntime().run_program(kb, queries)
    assert_same_groups(got[:1], expected[:1])
    assert_same_groups(got, expected, ordered=False)                    # the diagnosis lists its witnesses in another order
    assert petta.reduced_calls > 0 and petta.unresolved_calls == ()


@real_petta
@requires_hyperon
def test_the_linage2_stack_answers_on_petta_as_on_hyperon(petta):
    from core.patient_builder import build_patient
    fixture = json.loads((REPO / "tests" / "fixtures" / "linage2_response.json").read_text(encoding="utf-8"))
    built = build_patient({"id": "W58", "age": 58, "sex": "Male", "smoking": "CurrentSmoker",
                           "markers": {"HbA1c": 1.6, "CRP": 0.3}, "linage2": fixture})
    kb = _kb_text(pln_runner.linage2_patient_kb()) + "\n" + built.atoms
    queries = ["!(linage-hazard-patient &self Caller_W58)", "!(linage-decomposition-patient &self Caller_W58)"]
    got = petta.run_program(kb, queries)
    assert all(len(group) == 1 for group in got), got
    expected = HyperonRuntime().run_program(kb, queries)
    assert_same_groups(got[:1], expected[:1])                           # the hazard: the very same numbers
    assert_same_groups(got, expected, ordered=False)                    # the decomposition lists its contributions in another order


@real_petta
def test_a_second_program_in_the_same_process_starts_from_a_fresh_interpreter(petta, smoke_kb):
    """PeTTa's space is process-global: loading twice in one process would double every answer.
    Each run_program forks, so the counts are the same every time."""
    counts = "!(collapse (match &self (ZzProbe $t $c) $t))"
    program = smoke_kb + "\n(ZzProbe T1 0.5)\n(ZzProbe T2 0.6)"
    queries = [counts, "!(infer &self CellularSenescence CoronaryHeartDisease)"]
    first = petta.run_program(program, queries)
    second = petta.run_program(program, queries)
    assert first == second
    assert first[0] == ["(T1 T2)"] and len(first[1]) == 1
    assert "janus_swi" not in sys.modules and "petta" not in sys.modules


@real_petta
def test_a_stored_call_is_a_number_when_petta_reads_it_back(petta):
    kb = TIERS + "(Effect A B Pos (stv 0.7 (evidence-confidence RCT_Human)))\n(= (twice $x) (* 2 $x))"
    out = petta.run_program(kb, ["!(match &self (Effect A B Pos (stv $s $c)) (twice $c))"])
    assert out == [["2.0"]] and petta.reduced_calls == 1


@real_petta
def test_the_normaliser_gives_hyperons_booleans_on_the_real_engine(petta):
    out = petta.run_program("", ["!(== 1 1)", "!(and (== 1 1) (== 1 2))", "!(< 2 1)"])
    assert out == [["True"], ["False"], ["False"]]
    assert petta.last_raw == [["true"], ["false"], ["false"]]


@real_petta
def test_an_expression_that_fails_without_raising_is_an_empty_answer(petta):
    """A call no clause matches yields nothing in PeTTa (hyperon would return it unreduced); that is [], not an error."""
    assert petta.run_program("(= (known-f 1) one)", ["!(known-f 2)", "!(match &self (Nope $x) $x)", "!(empty)"]) == [[], [], []]


@real_petta
def test_a_form_that_does_not_compile_fails_the_load_and_the_process_is_discarded(petta):
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("(Foo a)\n(= (f) (superpose ()))", ["!(Foo $x)"])
    assert err.value.kind == "load" and "could not load the knowledge base" in str(err.value)
    assert petta.run_program("(Foo a)", ["!(match &self (Foo $x) $x)"]) == [["a"]]            # the next program is unaffected


@real_petta
def test_an_expression_that_raises_names_the_engines_error(petta):
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("", ["!(car-atom ())"])
    assert err.value.kind == "expression" and "car-atom" in str(err.value)


@real_petta
def test_petta_refuses_add_atom_before_it_reaches_the_engine(petta):
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("(Foo a)", ["!(add-atom &self (Foo b))"])
    assert err.value.kind == "refused"


@real_petta
def test_petta_also_refuses_a_literal_add_atom_hidden_in_kb_text(petta):
    """Finding 2's own reproduction, against the real engine: the text scan now covers `kb_text` (it
    carries the caller's own `extra_atoms`), not only expressions, so PeTTa never gets as far as loading
    `!(add-atom ...)` and executing it during the load."""
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("!(add-atom &self (Hacked yes))", ["!(match &self (Hacked $x) $x)"])
    assert err.value.kind == "refused" and "add-atom" in str(err.value)


@real_petta
def test_a_dynamically_constructed_add_atom_is_refused(petta):
    """Finding 1's own reproduction. `petta_refused_forms` reads the program's TEXT, so it can only ever
    see the literal token `add-atom` (asserted below: the scanner finds nothing here) — a symbol PeTTa
    builds at run time with `atom_concat`, bound to a variable, and then applied is invisible to any text
    scan. The engine-level sandbox (`_install_petta_sandbox`) is what has to stop the mutation itself.

    `reduce/2` (PeTTa's own dynamic dispatcher) calls the resolved goal inside `catch(_, _, fail)`, which
    would otherwise turn the guard's `throw` into an ordinary Prolog FAILURE — an unanswered `!` form the
    caller cannot tell from a query that legitimately found nothing. `_PettaEngine._raise_if_refused`
    reads a side channel the guard sets before it throws, so this still surfaces as a proper
    `MettaRuntimeError(kind="refused")` naming `add-atom`, not a silently empty `[[], []]` (what this
    exact reproduction returned before that fix existed).

    That the fact was NOT added is checked where a check can see it, in a session (the next test):
    `run_program` starts a new child for every call and raises at the refusal, so a second call to it
    could never see this child's space, whatever this child did."""
    expr = '!(let $f (atom_concat "add" "-atom") ($f &self (HackedA yes)))'
    assert petta_refused_forms(expr) == []
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("", [expr, "!(match &self (HackedA $x) $x)"])
    assert err.value.kind == "refused" and "add-atom" in str(err.value)


@real_petta
def test_a_dynamically_constructed_callpredicate_cannot_reach_shell(petta):
    """`callPredicate`/`Predicate` are PeTTa's own "call any Prolog predicate by name" doors (`metta.pl`'s
    "Prolog interop" section) — reachable even when neither the name the caller wrote nor the predicate it
    ends up calling (`shell`) ever appears as a literal token next to anything suspicious, so the text scan
    cannot catch this construction either. The engine-level sandbox refuses `Predicate` itself (it has no
    legitimate use outside this exact pairing — see `PETTA_ALLOWED_FUN_NAMES`), so the goal `callPredicate`
    would have invoked is never even built; `shell` and `callPredicate` are both refused unconditionally
    too, so any of the three names being reached stops the call."""
    expr = ('!(let $cp (atom_concat "callPre" "dicate") '
            '(let $pr (atom_concat "Predi" "cate") '
            '(let $sh (atom_concat "she" "ll") '
            '($cp ($pr ($sh "true" 0))))))')
    assert petta_refused_forms(expr) == []
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("", [expr])
    assert err.value.kind == "refused"
    assert any(name in str(err.value) for name in ("Predicate", "callPredicate", "shell"))


@real_petta
def test_an_endless_recursion_is_killed_at_the_deadline_and_the_parent_is_fine(petta):
    kb = "(= (spin $n) (spin (+ $n 1)))"
    started = time.monotonic()
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program(kb, ["!(spin 1)"], deadline_s=1.5)
    elapsed = time.monotonic() - started
    assert err.value.kind == "timeout" and 1.0 < elapsed < 10, elapsed
    assert wait_until(lambda: process_is_reaped(petta.last_pid), 5)
    assert petta.run_program("(Foo a)", ["!(match &self (Foo $x) $x)"]) == [["a"]]            # and the next program runs
    assert "janus_swi" not in sys.modules


@real_petta
def test_a_runaway_recursion_dies_of_the_stack_limit_as_an_exception():
    """A recursion that grows the Prolog stack ends in a resource error long before it takes the machine."""
    rt = PettaRuntime(stack_limit_mb=16)
    kb = "(= (grow $n) (cons-atom $n (grow (+ $n 1))))"
    started = time.monotonic()
    with pytest.raises(MettaRuntimeError) as err:
        rt.run_program(kb, ["!(grow 1)"], deadline_s=60)
    assert err.value.kind in ("expression", "crash") and time.monotonic() - started < 60
    assert wait_until(lambda: process_is_reaped(rt.last_pid), 5)


@real_petta
def test_a_petta_child_whose_worker_was_killed_stops_by_itself(monkeypatch):
    """The executor SIGKILLs a worker at its deadline; the PeTTa process below it must not spin on."""
    # subprocess.run of a MISSING command raises FileNotFoundError, not returncode 127 (verified) — check
    # with shutil.which before ever invoking it, or a slim image with no procps errors instead of skipping.
    if shutil.which("pgrep") is None:
        pytest.skip("pgrep is not available")
    monkeypatch.setattr(mr, "BACKSTOP_GRACE_S", 1.0)
    rt = PettaRuntime()
    middle = os.fork()
    if middle == 0:                                          # the "worker"
        try:
            rt.run_program("(= (spin $n) (spin (+ $n 1)))", ["!(spin 1)"], deadline_s=1.0)
        finally:
            os._exit(0)
    spinner = None
    try:
        def find_child():
            nonlocal spinner
            out = subprocess.run(["pgrep", "-P", str(middle)], capture_output=True, text=True).stdout.split()
            spinner = int(out[0]) if out else None
            return spinner is not None
        assert wait_until(find_child, 15), "the PeTTa child never appeared"
        time.sleep(0.8)                                      # let it get into the recursion
        os.kill(middle, signal.SIGKILL)                      # the worker dies; nobody reaps or kills its child
        os.waitpid(middle, 0)
        assert wait_until(lambda: process_is_gone(spinner), 20), "the orphaned PeTTa child is still spinning"
    finally:
        if spinner is not None and not process_is_gone(spinner):
            os.kill(spinner, signal.SIGKILL)


@real_petta
def test_no_zombie_and_no_descriptor_leak_on_the_real_engine(petta):
    petta.run_program("(Foo a)", ["!(match &self (Foo $x) $x)"])
    before = open_fds()
    for _ in range(4):
        petta.run_program("(Foo a)", ["!(match &self (Foo $x) $x)"])
        with pytest.raises(MettaRuntimeError):
            petta.run_program("", ["!(car-atom ())"])
        with pytest.raises(MettaRuntimeError):
            petta.run_program("(= (spin $n) (spin (+ $n 1)))", ["!(spin 1)"], deadline_s=0.7)
        assert process_is_reaped(petta.last_pid)
    assert open_fds() <= before + FD_SLACK


@real_petta
def test_petta_prints_nothing_into_the_parent(petta, capfd):
    capfd.readouterr()
    petta.run_program("(Foo a)", ['!(println! "TRACE-FROM-PETTA")', "!(match &self (Foo $x) $x)"])
    seen = capfd.readouterr()
    assert "TRACE-FROM-PETTA" not in seen.out + seen.err


@real_petta
@requires_hyperon
def test_run_query_parts_runs_two_stacks_under_petta_one_answer_set_per_part(runtime_paths, monkeypatch, tmp_path):
    from core.patient_builder import build_patient
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    fixture = json.loads((REPO / "tests" / "fixtures" / "linage2_response.json").read_text(encoding="utf-8"))
    built = build_patient({"id": "W58", "age": 58, "sex": "Male", "smoking": "CurrentSmoker",
                           "markers": {"HbA1c": 1.6, "CRP": 0.3}, "linage2": fixture})
    parts = [
        {"metta_query": "!(linage-hazard-patient &self Caller_W58)",
         "kb_files": pln_runner.linage2_patient_kb(), "extra_atoms": built.atoms},
        {"metta_query": "!(match &self (TargetsHallmark $i $h) (pair $i $h))",
         "kb_files": pln_runner.generic_stack(runtime_paths), "extra_atoms": None},
    ]
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    on_hyperon = pln_runner.run_query_parts(parts)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    reset_runtime_cache()
    try:
        on_petta = pln_runner.run_query_parts(parts)
    finally:
        reset_runtime_cache()
    assert len(on_petta) == 2 and [r.status for r in on_petta] == ["ok", "ok"]
    assert [r.runtime_name for r in on_petta] == ["petta", "petta"]
    assert [r.mode for r in on_petta] == ["runtime", "runtime"]
    assert len(on_petta[0].results) == 1 and len(on_petta[1].results) == len(on_hyperon[1].results) > 10
    assert same_answer(on_petta[0].results[0].atom, on_hyperon[0].results[0].atom)
    assert {r.atom for r in on_petta[1].results} == {r.atom for r in on_hyperon[1].results}
    merged = pln_runner.merge_run_results(on_petta, positions=[[0], [1]])
    assert merged.runtime_name == "petta" and merged.results[0].atom == on_petta[0].results[0].atom


@real_petta
def test_run_query_runs_on_petta_when_selected_and_reports_the_engines_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    reset_runtime_cache()
    try:
        ok = pln_runner.run_query("!(evidence-confidence ITP_Positive)", kb_files=[REPO / "epistemic_calibration.metta"])
        assert (ok.status, ok.runtime_name, [r.atom for r in ok.results]) == ("ok", "petta", ["0.9"])
        bad = pln_runner.run_query("(car-atom ())", kb_files=[])
        assert (bad.status, bad.error_code) == ("error", "runtime_error") and "car-atom" in bad.error
        refused = pln_runner.run_query("!(add-atom &self (X y))", kb_files=[])
        assert refused.status == "error" and "does not run" in refused.error
    finally:
        reset_runtime_cache()


@real_petta
def test_a_real_session_keeps_one_loaded_kb_and_closes_cleanly(petta):
    before = open_fds()
    with petta.session("(Foo a)\n(= (evidence-confidence T1) 0.5)") as session:
        pid = session._child.pid
        assert session.run("!(match &self (Foo $x) $x)") == [["a"]]
        assert session.run("(Foo b)\n(G (stv 1 (evidence-confidence T1)))") == []
        assert session.run("!(collapse (match &self (Foo $x) $x))\n!(match &self (G $x) $x)") == [["(a b)"], ["(stv 1 0.5)"]]
        assert session.reduced_calls == 1
        assert session.last_raw[0] == ["(a b)"]
        with pytest.raises(MettaRuntimeError) as err:
            session.run("!(car-atom ())")
        assert err.value.kind == "expression"
        assert session.run("!(== 1 1)") == [["True"]]                  # an expression that raised did not end the session
    assert process_is_reaped(pid) and open_fds() <= before + FD_SLACK


@real_petta
def test_assert_inside_a_real_session_is_refused_before_it_can_halt(petta):
    """PeTTa's `assert` calls `halt(1)`. Before the fix (the finding that `PettaSession.run()` performed
    no check of any kind), this reached that call and genuinely ended the child — a `kind="crash"` result,
    and the session was gone. `assert` is a refused form, so it is now caught before anything is sent to
    the child: the session survives the call entirely, which is the strictly better outcome (no crash to
    survive in the first place)."""
    with petta.session("(Foo a)") as session:
        pid = session._child.pid
        with pytest.raises(MettaRuntimeError) as err:
            session.run("!(assert (== 1 2))")
        assert err.value.kind == "refused" and "assert" in str(err.value)
        os.kill(pid, 0)                                 # still alive: no crash, nothing was sent to the child
        assert session.run("!(== 1 1)") == [["True"]]    # and the session keeps answering


@real_petta
def test_a_dynamically_constructed_symbol_cannot_mutate_a_real_session(petta):
    """Finding 4's deeper concern, through a session rather than `run_program`: the text scan added for
    finding 4 only catches the literal spelling. The engine-level sandbox is what actually has to hold for
    a session's `run()` text too, since a session can live across many calls and its space has no reset.
    The refusal surfaces as `MettaRuntimeError(kind="refused")` (see the `run_program` version of this
    test for why that needs its own fix, not just the guard itself), and the session survives it — nothing
    was ever sent to the real `add-atom/3`, so there is nothing to recover from."""
    with petta.session("(Foo a)") as session:
        expr = '!(let $f (atom_concat "add" "-atom") ($f &self (HackedA yes)))'
        assert petta_refused_forms(expr) == []
        with pytest.raises(MettaRuntimeError) as err:
            session.run(expr)
        assert err.value.kind == "refused" and "add-atom" in str(err.value)
        assert session.run("!(match &self (HackedA $x) $x)") == [[]]   # the fact was never added, session alive


# ─── every reachable head not on the allowlist, built several ways, refused ───
#
# `PETTA_ALLOWED_FUN_NAMES` is what `_install_petta_sandbox` checks; this table exists so the ENGINE-level
# refusal (not the text scan, which these constructions mostly defeat on purpose) is proven against a
# representative spread of the generic-dispatch primitives `reachable_heads.md`'s review catalogued —
# `atom_concat`+`let`, `chain` (PeTTa's own `let_bind` lowering, a different argument order than `let`),
# `let*`, `case`, `superpose` of a literal head, and `quote`/`eval` of a literal head — each against a
# head drawn from a different safety class (a space mutator, the verifier's own translator-hook finding,
# the halt-on-failure pair, process-global state, and a non-`fun/1` host predicate reachable only through
# a generic-dispatch primitive). Every canary is harmless: a `&self`-scoped fact that is never actually
# added (the whole point is that it ISN'T), or a call PeTTa itself would refuse before touching the host.
_DYNAMIC_REFUSAL_CANARIES = [
    ("atom_concat+let, add-atom (a space mutator)",
     '!(let $f (atom_concat "add" "-atom") ($f &self (DynHead1 yes)))', "add-atom"),
    ("chain (let_bind, Pat-Val-Body order), add-atom",
     '!(chain $f (atom_concat "add" "-atom") ($f &self (DynHead2 yes)))', "add-atom"),
    ("let*, change-state! (process-global state)",
     '!(let* (($f (atom_concat "change-" "state!"))) ($f &dyn3 1))', "change-state!"),
    ("case, add-translator-rule! (the verifier's finding, spelled literally in a branch)",
     "!((case 1 ((1 add-translator-rule!) (2 remove-translator-rule!))) myDynHook4)",
     "add-translator-rule!"),
    ("superpose of one literal head, assert (halts on failure)",
     '!(let $f (superpose (assert)) ($f (== 1 2)))', "assert"),
    ("quote/eval of a literal head, shell (not fun/1 — reachable only through a generic primitive)",
     '!(eval (quote (shell "true" 0)))', "shell"),
]


@pytest.mark.parametrize("label, expr, head", _DYNAMIC_REFUSAL_CANARIES, ids=[c[0] for c in _DYNAMIC_REFUSAL_CANARIES])
@real_petta
def test_every_construction_style_is_refused_as_an_expression(petta, label, expr, head):
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program("", [expr])
    assert err.value.kind == "refused" and head in str(err.value), (label, str(err.value))


@pytest.mark.parametrize("label, expr, head", _DYNAMIC_REFUSAL_CANARIES, ids=[c[0] for c in _DYNAMIC_REFUSAL_CANARIES])
@real_petta
def test_every_construction_style_is_refused_hidden_in_kb_text(petta, label, expr, head):
    """The same canaries, placed where a caller's own `extra_atoms` lands: concatenated into `kb_text`
    ahead of a harmless query, not passed as a separate expression. PeTTa evaluates any top-level `!` form
    found anywhere in `kb_text` during the load — `petta_refused_forms_in_kb`'s whole reason to exist —
    and the engine-level guard holds there exactly as it does for an expression."""
    with pytest.raises(MettaRuntimeError) as err:
        petta.run_program(expr + "\n(Harmless fact)", ["!(match &self (Harmless $x) $x)"])
    assert err.value.kind == "refused" and head in str(err.value), (label, str(err.value))


@pytest.mark.parametrize("label, expr, head", _DYNAMIC_REFUSAL_CANARIES, ids=[c[0] for c in _DYNAMIC_REFUSAL_CANARIES])
@real_petta
def test_every_construction_style_is_refused_inside_a_session(petta, label, expr, head):
    """A session's `run()` text is a THIRD place caller bytes reach the engine (`PettaRuntime.run_program`
    and `kb_text` are the other two); same guard, same construction styles, and the session survives —
    confirmed once per canary, not just for the `add-atom` case the earlier dedicated test covers."""
    with petta.session("(Foo a)") as session:
        with pytest.raises(MettaRuntimeError) as err:
            session.run(expr)
        assert err.value.kind == "refused" and head in str(err.value), (label, str(err.value))
        assert session.run("!(== 1 1)") == [["True"]]          # the session is still alive and answering



@real_petta
@pytest.mark.slow
def test_the_executors_daemonic_worker_forks_a_petta_child(monkeypatch, tmp_path):
    """core.executor runs a query in a daemonic multiprocessing child; PeTTa runs one level below it."""
    import core.executor as executor
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 2)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")   # the forked worker inherits this; keeps
                                                                  # it out of the repo's own pln_chat/logs
    reset_runtime_cache()
    try:
        result = executor.run_offloaded(
            "run_query",
            {"metta_query": "!(evidence-confidence ITP_Positive)", "kb_files": [REPO / "epistemic_calibration.metta"]},
            lambda: pytest.fail("should not run inline when a pool is configured"),
        )
    finally:
        reset_runtime_cache()
    assert result.status == "ok" and result.runtime_name == "petta"
    assert [r.atom for r in result.results] == ["0.9"]
