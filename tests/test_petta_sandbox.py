"""The PeTTa sandbox, door by door.

`tests/test_metta_runtime.py` checks the runtime seam and that the literal spelling of a refused form is
refused. This file checks what stops a program that does NOT spell it: every way MeTTa can make PeTTa call a
Prolog predicate by a name the program chooses (see "the engine-level sandbox" in `core/metta_runtime.py`),
what the forked child is cut off from, and the reader the text scan relies on.

Four layers, cheapest first:

* the text scan against the engine's own reader (`petta_tokens`, a corpus checked against `sread`);
* the child's hardening against a TEST DOUBLE of the engine (its environment, working directory, descriptors);
* the sandbox's own installation against a double of janus (a failing guard must stop the child);
* the REAL PeTTa (`-m petta`): the doors with harmless canaries, the add-atom family, the sandbox over HTTP.

Every canary is harmless to touch: a temporary file whose content must not appear, a temporary file that
must not be deleted or created, a synthetic environment variable. No test calls a model.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_metta_runtime import (  # noqa: E402  (also puts pln_chat on sys.path)
    REPO, PLN_CHAT, ScriptedEngine, real_petta, requires_petta,
)

import config  # noqa: E402
import core.metta_runtime as mr  # noqa: E402
from core.metta_runtime import (  # noqa: E402
    MettaRuntimeError, PETTA_ALLOWED_FUN_NAMES, PETTA_ENGINE_INTERNAL_FUN_NAMES, PETTA_KNOWN_FUN_NAMES,
    PettaRuntime, petta_refused_forms, petta_refused_forms_in_kb, petta_tokens, reset_runtime_cache,
    split_top_level_forms,
)

PETTA_HOME = Path(config.PETTA_HOME)


# ═══════════════════════ the text scan reads what PeTTa's reader reads ════════════════════════

@pytest.mark.parametrize("text, expected", [
    # a quote in the MIDDLE of a token is an ordinary character for PeTTa's reader: three atoms, not a string
    ('(x a"b add-atom c"d y)', ["add-atom"]),
    # a comment takes its newline with it, so it can join the text on both sides into one token
    ("(foo add-;a comment\natom)", ["add-atom"]),
    ("(foo ad;one\nd-at;two\nom)", ["add-atom"]),
    # ... but inside a string it is text
    ('(foo "add-;x\natom")', []),
    # a string that ends a line with a backslash is still one string
    ('(Note "a\\\n(") !(shell ls)', ["shell"]),
    ('(Note "add-atom\\\n)")', []),
    # the second compiler form that builds a goal from the program's own data
    ("!(translatePredicate (getenv \"X\" $v))", ["translatePredicate"]),
    ("!(call (getenv \"X\"))", ["call"]),
    # an escaped quote does not end the string
    ('(Note "a \\" add-atom")', []),
    ("(a b) ; add-atom in a comment\n(c d)", []),
])
def test_the_scan_reads_symbols_the_way_petta_reads_them(text, expected):
    assert petta_refused_forms(text) == expected
    assert petta_refused_forms_in_kb(text) == [n for n in expected if n != "="]


def test_a_symbol_is_a_symbol_even_beside_a_string():
    kinds = [kind for kind, _ in petta_tokens('(f "a b" c"d e)')]
    assert kinds == ["paren", "symbol", "string", "symbol", "symbol", "paren"]


_READER_PROLOG = r"""
petta_probe_leaf(T, _) :- var(T), !, fail.
petta_probe_leaf(T, A) :- is_list(T), !, member(E, T), petta_probe_leaf(E, A).
petta_probe_leaf(T, T) :- atom(T).
petta_probe_atoms(Text, Atoms) :-
    metta_string_forms(Text, Forms),
    findall(A, ( member(F, Forms), F =.. [_, Str, _], sread(Str, T), petta_probe_leaf(T, A) ), Atoms).
petta_probe_forms(Text, Forms) :-
    metta_string_forms(Text, Fs),
    findall(Str, ( member(F, Fs), F =.. [_, Str, _] ), Forms).
"""


def _reader_child(conn, home: str, texts: list[str]) -> None:
    mr._child_prepare(None, parent_pid=None, keep_fds=(conn.fileno(),))
    sys.path.insert(0, f"{home}/python")
    import janus_swi as janus
    from petta import PeTTa
    PeTTa(verbose=False, petta_path=home)
    janus.consult("probe", data=_READER_PROLOG)
    atoms, forms = [], []
    for text in texts:
        try:
            row = janus.query_once("petta_probe_atoms(T, A)", {"T": text})
            atoms.append(sorted({str(a) for a in row["A"]}) if row.get("truth") else None)
            row = janus.query_once("petta_probe_forms(T, F)", {"T": text})
            forms.append([str(f) for f in row["F"]] if row.get("truth") else None)
        except Exception:                                   # the reader refused the text
            atoms.append(None)
            forms.append(None)
    conn.send_bytes(json.dumps({"kind": "result", "atoms": atoms, "forms": forms}).encode("utf-8"))


def _engines_reading(texts: list[str]) -> tuple[list, list]:
    """Per text: the atoms PeTTa's own reader finds in it and the forms it splits it into (None when the
    reader raises). Out of process, like every direct use of janus in this suite."""
    child = mr._fork_child(_reader_child, str(PETTA_HOME), texts)
    try:
        message = child.wait_message(time.monotonic() + 180)
    finally:
        child.close()
    assert message and message.get("kind") == "result", message
    return message["atoms"], message["forms"]


def _reader_corpus() -> list[str]:
    """Texts built to make a hand-written scanner and PeTTa's reader disagree: quotes inside tokens, comments
    that join tokens, backslashes before newlines, vertical tabs, strings that never end. Seeded, so the same
    on every run."""
    rng = random.Random(20261024)
    pieces = ["(", ")", "!(", "!", '"', ";", "\\", " ", " ", "\n", "a", "add-atom", "call", "translatePredicate",
              "shell", "$x", "1", 'b"c', 'd"', "\x0b", "\t", "(a", "b)", '"s s"', ";c\n", "\\\n"]
    texts = ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 14))) for _ in range(900)]

    def sexp(depth: int = 0) -> str:
        if depth > 3 or rng.random() < 0.5:
            return rng.choice(["a", "add-atom", "call", "translatePredicate", "shell", "$x", "1", 'b"c', 'd"e f"g',
                               '"str"', '"a b"', '"x\\"y"', "x;y", 'q"', '"q'])
        return "(" + rng.choice(["", " ", "\n"]).join(sexp(depth + 1) for _ in range(rng.randint(0, 4))) + ")"

    for _ in range(900):
        form = sexp()
        if not form.startswith("("):
            form = "(" + form + ")"
        form = "!" * rng.randint(0, 1) + form
        if rng.random() < 0.5:
            form = form.replace(" ", " ;c" + rng.choice(["", "x"]) + "\n", 1)
        texts.append(form + rng.choice(["", "\n", " ", " ;tail"]) + (sexp() if rng.random() < 0.3 else ""))
    texts += ['(x a"b add-atom c"d y)', "(foo add-;c\natom)", '(Note "a\\\n(")\n(b)', "(a\x0badd-atom)"]
    return texts


@real_petta
def test_the_scan_finds_exactly_the_symbols_the_engines_reader_finds():
    """The scan is a friendly 422, not a boundary — but a scan that sees something other than what the engine
    sees is how `add-atom` got past it inside `(x a"b add-atom c"d y)`. Every text of a seeded corpus that
    PeTTa's reader accepts: a name the reader finds as an atom is a symbol of the scan, and a symbol of the
    scan is an atom of the reader.

    One deliberate exception, in the safe direction: PeTTa's token ends only at a space, tab, newline,
    carriage return or parenthesis, so `a<vertical tab>add-atom` is ONE atom for the reader; the scan splits at
    any whitespace and reports `add-atom` too. A name the reader finds is never missing from the scan."""
    texts = _reader_corpus()
    atoms, _ = _engines_reading(texts)
    probes = ("add-atom", "call", "translatePredicate", "shell", "a")
    parsed = 0
    holding = 0
    for text, found in zip(texts, atoms):
        if found is None:
            continue
        parsed += 1
        scan = {tok for kind, tok in petta_tokens(text) if kind == "symbol"}
        exotic_whitespace = any(ch.isspace() and ch not in " \t\r\n" for ch in text)
        for name in probes:
            if name in found:
                assert name in scan, (name, text, found, sorted(scan))
            elif not exotic_whitespace:
                assert name not in scan, (name, text, found, sorted(scan))
        holding += any(name in found for name in probes)
    assert parsed > 400 and holding > 100, (parsed, holding)           # the corpus is not vacuous


@real_petta
def test_the_form_splitter_agrees_with_the_engines_on_the_texts_it_accepts():
    """`split_top_level_forms` serves the stored-call reducer. Where PeTTa splits a text, it finds the same
    forms (PeTTa's own come with the comments removed)."""
    texts = _reader_corpus()
    _, forms = _engines_reading(texts)
    compared = 0
    for text, theirs in zip(texts, forms):
        if theirs is None:
            continue
        mine = [mr.petta_strip_comments(text[s:e]).lstrip("!") for s, e in split_top_level_forms(text)]
        assert mine == theirs, (text, mine, theirs)
        compared += 1
    assert compared > 400


def test_a_string_that_ends_a_line_with_a_backslash_is_one_string_for_the_splitter():
    text = '(Fact "a\\\n(")\n(Fact2 b)\n!(go)'
    assert [text[s:e] for s, e in split_top_level_forms(text)] == ['(Fact "a\\\n(")', "(Fact2 b)", "!(go)"]


# ═══════════════════════ the child is cut off from the server ════════════════════════

class ProbeEngine(ScriptedEngine):
    """The scripted double with verbs that report what the forked child can reach."""

    def run(self, text: str) -> list[str]:
        body = text.strip()[2:-1].strip() if text.strip().startswith("!") else ""
        verb, _, arg = body.partition(" ")
        if verb == "env":
            return [str(os.environ.get(arg))]
        if verb == "environ":
            return sorted(os.environ)
        if verb == "cwd":
            return [os.getcwd()]
        if verb == "ls":
            return sorted(os.listdir(arg or "."))
        if verb == "limits":
            import resource
            return [str(resource.getrlimit(resource.RLIMIT_CORE)[0]), str(resource.getrlimit(resource.RLIMIT_NOFILE)[0])]
        if verb == "fds":
            import stat
            devnull = os.stat(os.devnull).st_rdev
            kinds = []
            for fd in range(3, 128):
                try:
                    st = os.fstat(fd)
                except OSError:
                    continue
                is_null = stat.S_ISCHR(st.st_mode) and st.st_rdev == devnull
                kinds.append(f"{fd}:{'null' if is_null else 'other'}")
            return kinds
        return super().run(text)


@pytest.fixture
def probe(tmp_path):
    def factory(petta_home, stack_limit_mb):
        engine = ProbeEngine({})
        engine.petta_home = petta_home
        return engine
    return PettaRuntime(petta_home=tmp_path, stack_limit_mb=64, engine_factory=factory)


def test_the_child_does_not_inherit_the_servers_secrets(probe, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    monkeypatch.setenv("PLN_TEST_UNLISTED", "x")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    (names,) = probe.run_program("", ["!(environ)"])
    assert "OPENAI_API_KEY" not in names and "PLN_TEST_UNLISTED" not in names
    assert "PATH" in names and "LC_ALL" in names                          # what a process needs to start stays
    assert probe.run_program("", ["!(env OPENAI_API_KEY)"]) == [["None"]]


def test_a_deployment_can_keep_a_variable_its_swi_prolog_needs(probe, monkeypatch):
    monkeypatch.setenv("PLN_TEST_NEEDED_BY_SWI", "yes")
    monkeypatch.setenv("PLN_PETTA_CHILD_ENV_KEEP", "PLN_TEST_NEEDED_BY_SWI, SOMETHING_ELSE")
    assert probe.run_program("", ["!(env PLN_TEST_NEEDED_BY_SWI)"]) == [["yes"]]


def test_everything_that_names_a_place_to_write_points_into_the_childs_own_directory(probe):
    cwd, home, tmp, xdg, cache = probe.run_program(
        "", ["!(cwd)", "!(env HOME)", "!(env TMPDIR)", "!(env XDG_CACHE_HOME)", "!(env PETTA_LOAD_CACHE)"])
    scratch = os.path.realpath(cwd[0])
    for value in (home, tmp, xdg):
        assert os.path.realpath(value[0]) == scratch
    assert os.path.realpath(cache[0]).startswith(scratch + os.sep)


def test_the_childs_working_directory_is_empty_private_and_removed_with_it(probe):
    (listing,) = probe.run_program("", ["!(ls)"])
    assert listing == []
    (first,) = probe.run_program("", ["!(cwd)"])
    (second,) = probe.run_program("", ["!(cwd)"])
    assert first != second                                                 # one directory per child
    assert not Path(first[0]).exists() and not Path(second[0]).exists()    # and none left behind


def test_a_killed_child_leaves_no_directory_behind(probe, monkeypatch):
    made: list[str] = []
    real_mkdtemp = mr.tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs):
        made.append(real_mkdtemp(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(mr.tempfile, "mkdtemp", recording_mkdtemp)
    with pytest.raises(MettaRuntimeError) as err:
        probe.run_program("", ["!(sleep 30)"], deadline_s=0.5)
    assert err.value.kind == "timeout"
    assert len(made) == 1 and not Path(made[0]).exists()


def test_the_child_holds_none_of_the_parents_descriptors(probe):
    """A server's sockets and log files are copied into every fork; the child must not be able to reach them."""
    import socket
    a, b = socket.socketpair()
    r, w = os.pipe()
    handle = open(__file__, "rb")
    try:
        mine = {a.fileno(), b.fileno(), r, w, handle.fileno()}
        (seen,) = probe.run_program("", ["!(fds)"])
        by_fd = {int(item.split(":")[0]): item.split(":")[1] for item in seen}
        assert mine <= set(by_fd)                                  # the numbers are still taken ...
        assert {by_fd[fd] for fd in mine} == {"null"}              # ... by /dev/null
        assert sum(kind != "null" for kind in by_fd.values()) <= 1, by_fd      # only the pipe to the parent is left
    finally:
        a.close(); b.close(); os.close(r); os.close(w); handle.close()


def test_the_child_cannot_write_a_core_file_or_open_unbounded_descriptors(probe):
    ((core, nofile),) = probe.run_program("", ["!(limits)"])
    assert core == "0"
    assert 0 < int(nofile) <= 512


# ═══════════════════════ the sandbox is installed whole, or not at all ════════════════════════

class FakeJanus:
    """Records what `_install_petta_sandbox` asks of janus; `fail` makes a goal containing that text fail."""

    def __init__(self, fail: str = "", fun_names=None) -> None:
        self.fail = fail
        self.consulted: list[str] = []
        self.goals: list[tuple[str, dict]] = []
        self.fun_names = list(PETTA_KNOWN_FUN_NAMES) if fun_names is None else fun_names

    def consult(self, name: str, data: str = "") -> None:
        self.consulted.append(data)

    def query_once(self, goal: str, bindings: dict | None = None, **_):
        self.goals.append((goal, dict(bindings or {})))
        if (self.fail and self.fail in goal) or goal.startswith("petta_sandbox_last_refusal"):
            return {"truth": False}                  # what janus answers to a goal that failed
        if goal.startswith("petta_sandbox_fun_names"):
            return {"truth": True, "Out": self.fun_names}
        if goal.startswith("petta_sandbox_reachable_arities"):
            return {"truth": True, "Out": [1]}
        if goal.startswith("petta_sandbox_is_generic_apply"):
            return {"truth": False}                  # none of the double's names resolve to yall
        return {"truth": True}


@pytest.mark.parametrize("failing", [
    "petta_sandbox_guard_flagged", "petta_sandbox_guard_always", "petta_sandbox_guard_space_call",
    "petta_sandbox_guard_space_arg", "petta_sandbox_guard_hooks", "petta_sandbox_guard_closure",
    "petta_sandbox_guard_process_form",
])
def test_a_guard_that_fails_stops_the_child_instead_of_leaving_a_hole(failing):
    """janus answers a failed goal with `{"truth": False}` — a non-empty dict, so `if not answer` never fired,
    and a guard that failed was read as installed. Every guard step must stop the install."""
    with pytest.raises(RuntimeError, match="could not sandbox"):
        mr._install_petta_sandbox(FakeJanus(fail=failing))


def test_an_empty_builtin_registry_stops_the_child():
    with pytest.raises(RuntimeError, match="registry"):
        mr._install_petta_sandbox(FakeJanus(fun_names=[]))


def test_every_name_outside_the_allowed_set_is_guarded_and_the_internal_ones_are_taken_off_fun():
    janus = FakeJanus(fun_names=sorted(PETTA_KNOWN_FUN_NAMES | {"a-builtin-a-future-petta-adds"}))
    mr._install_petta_sandbox(janus)
    guarded = {b["N"] for goal, b in janus.goals if goal.startswith("petta_sandbox_guard_always")}
    flagged = {b["N"] for goal, b in janus.goals if goal.startswith("petta_sandbox_guard_flagged")}
    unregistered = {b["N"] for goal, b in janus.goals if goal.startswith("retractall(fun")}
    everything = PETTA_KNOWN_FUN_NAMES | {"a-builtin-a-future-petta-adds"}
    assert "a-builtin-a-future-petta-adds" in guarded                      # nothing had to name it
    assert guarded == everything - PETTA_ALLOWED_FUN_NAMES - PETTA_ENGINE_INTERNAL_FUN_NAMES - {"add-atom", "remove-atom"} | {"shell", "halt"}
    assert flagged == {"add-atom", "remove-atom", "add-atom-proven", "remove-atom-proven"}
    assert unregistered == set(PETTA_ENGINE_INTERNAL_FUN_NAMES)
    assert not guarded & PETTA_ALLOWED_FUN_NAMES


def test_no_name_in_the_sandbox_is_shared_between_children_or_guessable():
    first, second = FakeJanus(), FakeJanus()
    mr._install_petta_sandbox(first)
    mr._install_petta_sandbox(second)
    assert "@FLAG@" not in first.consulted[0] and "@NOTE@" not in first.consulted[0]
    assert first.consulted[0] != second.consulted[0]
    copies = lambda j: {b["O"] for goal, b in j.goals if "O" in b}
    assert copies(first) and not copies(first) & copies(second)


def test_a_goal_that_fails_without_raising_is_a_failed_load():
    """The branch the module's docs called untriggerable: janus hands back `{"truth": False}`, not None."""
    engine = mr._PettaEngine.__new__(mr._PettaEngine)
    engine._janus = FakeJanus(fail="run_metta_helper")
    with pytest.raises(RuntimeError, match="failed without raising"):
        engine.load("(Fact a)")
    assert engine.run("!(f)") == []                      # an EXPRESSION that fails is an empty answer, as ever


def test_the_engines_split_forms_say_so_when_the_reader_cannot_split():
    engine = mr._PettaEngine.__new__(mr._PettaEngine)
    engine._janus = FakeJanus(fail="petta_sandbox_forms")
    with pytest.raises(RuntimeError, match="reader"):
        engine.split_forms("(a)")


# ═══════════════════════ the doors, against the real engine ════════════════════════

ENV_CANARY = "PLN_TEST_CANARY_SECRET"
ENV_VALUE = "CANARY-ENV-VALUE"
FILE_VALUE = "CANARY-FILE-CONTENT"
DIR_ENTRY = "canary-entry.txt"


@pytest.fixture
def canaries(tmp_path, monkeypatch):
    """Things a program must not read, delete or create. The environment variable is kept through the child's
    scrub on purpose: these tests are about the DOORS, so the secret has to be there to find."""
    secret = tmp_path / "secret.txt"
    secret.write_text(FILE_VALUE)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / DIR_ENTRY).write_text("x")
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    monkeypatch.setenv(ENV_CANARY, ENV_VALUE)
    monkeypatch.setenv("PLN_PETTA_CHILD_ENV_KEEP", ENV_CANARY)
    return SimpleNamespace(secret=secret, tree=tree, victim=victim, touched=tmp_path / "touched.txt")


def _doors(c) -> list[tuple[str, str, str]]:
    """(label, expression, what it would give away). Every shape works, with a leak, against the engine
    without the fix (checked against the previous version of the module when the fix was made)."""
    s, t, v, x, e = c.secret, c.tree, c.victim, c.touched, ENV_CANARY
    return [
        # `maplist` calls whatever predicate its first argument names — the independent review's reproduction
        ("maplist+getenv", f'!(maplist getenv ("{e}"))', "env"),
        ("maplist+directory_files", f'!(maplist directory_files ("{t}"))', "dir"),
        # `match` turns its Space argument into a predicate name
        ("match+getenv", f"!(match getenv ({e} $v) $v)", "env"),
        ("match+read_file_to_string", f'!(match read_file_to_string ("{s}" $v ()) $v)', "file"),
        ("match+directory_files", f'!(match directory_files ("{t}" $v) $v)', "dir"),
        ("match+exists_file", f'!(match exists_file ("{s}") found)', "exists"),
        ("match+delete_file", f'!(match delete_file ("{v}") deleted)', "victim"),
        ("match+shell", f'!(match shell ("touch {x}" $s) $s)', "touched"),
        ("conjunctive match+getenv", f"!(match getenv (, ({e} $v)) $v)", "env"),
        ("get-atoms of PeTTa's own registry", "!(collapse (get-atoms fun))", "registry"),
        ("get-atoms of every predicate", "!(collapse (get-atoms $s))", "registry"),
        # compiler forms that build a goal from the program's own data, spelled out
        ("call+getenv", f'!(call (getenv "{e}"))', "env"),
        ("call+directory_files", f'!(call (directory_files "{t}"))', "dir"),
        ("translatePredicate+getenv", f'!(translatePredicate (getenv "{e}" $v))', "env"),
        ("translatePredicate+delete_file", f'!(translatePredicate (delete_file "{v}"))', "victim"),
        ("translatePredicate+shell", f'!(translatePredicate (shell "touch {x}" $s))', "touched"),
        ("translatePredicate+read_file_to_string", f'!(translatePredicate (read_file_to_string "{s}" $v ()))', "file"),
        # ... and never spelled: the name is built, the list is built, and `eval` compiles it
        ("built call+getenv",
         f'!(let $c (atom_concat "ca" "ll") (let $e (cons-atom $c ((getenv "{e}"))) (eval $e)))', "env"),
        ("built translatePredicate+delete_file",
         f'!(let $c (atom_concat "translate" "Predicate") (let $e (cons-atom $c ((delete_file "{v}"))) (eval $e)))', "victim"),
        ("built translatePredicate+shell",
         f'!(let $c (atom_concat "translate" "Predicate") (let $e (cons-atom $c ((shell "touch {x}" $s))) (eval $e)))', "touched"),
        # the engine's own function-call paths leave a name that is not a function as data
        ("dynamic head getenv", f'!(let $f (atom_concat "get" "env") ($f "{e}"))', "env"),
        ("superpose of a head", f'!(let $f (superpose (getenv)) ($f "{e}"))', "env"),
        ("map-atom with a host predicate", f'!(map-atom ("{e}") getenv)', "env"),
        ("foldl-atom with a host predicate", f'!(foldl-atom ("{e}") 0 getenv)', "env"),
        ("exists_file as a function", f'!(exists_file "{s}")', "exists"),
    ]


def _gave_away(kind: str, c, answer: str) -> str:
    """'' when nothing was given away, else what."""
    if kind == "env" and ENV_VALUE in answer:
        return "the environment variable"
    if kind == "file" and FILE_VALUE in answer:
        return "the file"
    if kind == "dir" and DIR_ENTRY in answer:
        return "the directory listing"
    if kind == "exists" and re.search(r"\b(found|True|true)\b", answer):
        return "whether a file exists"
    if kind == "victim" and not c.victim.exists():
        return "a deletion"
    if kind == "touched" and c.touched.exists():
        return "a process that wrote a file"
    if kind == "registry" and "add-translator-rule!" in answer:
        return "PeTTa's registry of builtins"
    return ""


@real_petta
def test_no_door_reaches_a_host_predicate(canaries, monkeypatch):
    """One session, every door, one harmless canary behind each. With the text scan off, because this is
    about what the ENGINE lets through; a door either raises `kind="refused"` or leaves the name as data,
    and in neither case does anything happen on the host."""
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)
    failures: list[str] = []
    refused = 0
    with PettaRuntime().session("(Fact a)") as session:
        for label, expression, kind in _doors(canaries):
            try:
                groups = session.run(expression)
                answer = json.dumps(groups)
            except MettaRuntimeError as err:
                if err.kind != "refused":
                    failures.append(f"{label}: {err.kind}: {str(err)[:100]}")
                answer = str(err)
                refused += 1
            gave = _gave_away(kind, canaries, answer)
            if gave:
                failures.append(f"{label}: gave away {gave}: {answer[:100]}")
        assert session.run("!(match &self (Fact $x) $x)") == [["a"]]      # the session is alive, and reads
    assert not failures, "\n".join(failures)
    assert refused >= 15                                                    # most doors are refused outright


@real_petta
def test_the_names_peTTa_calls_itself_are_data_to_a_program_and_still_work_for_the_engine():
    expression = '!(maplist f (1 2))'
    saved = mr._refuse_if_forbidden
    mr._refuse_if_forbidden = lambda *a, **k: None
    try:
        with PettaRuntime().session("") as session:
            assert session.run(expression) == [["(maplist f (1 2))"]]
            assert session.run('!(foldl f (1 2) 0)') == [["(foldl f (1 2) 0)"]]
            assert session.run('!(sread "(a b)")') == [['(sread "(a b)")']]
            # the translator's own maplist/foldl (pseudo-lambdas) and `parse` (which calls sread) are untouched
            assert session.run("!(foldl-atom (1 2 3) 0 $acc $x (+ $acc $x))") == [["6"]]
            assert session.run('!(parse "(a b)")') == [["(a b)"]]
    finally:
        mr._refuse_if_forbidden = saved


@real_petta
def test_call_still_runs_a_function_peTTa_knows_and_so_do_its_own_rewrites():
    """`unique`, `union`, ... are rewritten by the translator into `(call (superpose ...))`: the guard on
    `call` and `translatePredicate` must let a function through."""
    saved = mr._refuse_if_forbidden
    mr._refuse_if_forbidden = lambda *a, **k: None
    try:
        with PettaRuntime().session("") as session:
            assert session.run("!(unique (superpose (1 1 2)))") == [["1", "2"]]
            assert session.run("!(union (superpose (1 2)) (superpose (2 3)))") == [["1", "2", "2", "3"]]
            assert session.run("!(call (superpose (1 2)))") == [["1", "2"]]
            assert session.run("!(call (+ 1 2))") == [["3"]]
    finally:
        mr._refuse_if_forbidden = saved


@real_petta
def test_a_program_that_catches_the_refusal_is_refused_all_the_same():
    expression = '!(catch (let $f (atom_concat "add" "-atom") ($f &self (Caught yes))))'
    with PettaRuntime().session("") as session:
        with pytest.raises(MettaRuntimeError) as err:
            session.run(expression)
        assert err.value.kind == "refused" and "add-atom" in str(err.value)
        assert session.run("!(match &self (Caught $x) $x)") == [[]]


@real_petta
def test_a_refusal_names_the_head_and_the_reason_and_nothing_of_the_server(canaries, monkeypatch):
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)
    cases = {
        f'!(match getenv ({ENV_CANARY} $v) $v)': ("match", "space"),
        f'!(call (getenv "{ENV_CANARY}"))': ("call", "function PeTTa knows"),
        '!(let $f (atom_concat "add" "-atom") ($f &self (X y)))': ("add-atom", "may not change the space"),
        '!(change-state! &x 1)': ("change-state!", "allowed set"),
    }
    with PettaRuntime().session("") as session:
        for expression, (head, reason) in cases.items():
            with pytest.raises(MettaRuntimeError) as err:
                session.run(expression)
            message = str(err.value)
            assert err.value.kind == "refused" and f"'{head}'" in message and reason in message, message
            assert str(canaries.secret.parent) not in message and ENV_VALUE not in message


def _sandboxed_child(conn, home: str, goals: list[str]) -> None:
    mr._child_prepare(None, parent_pid=None, keep_fds=(conn.fileno(),))
    engine = mr._PettaEngine(home, 512)                      # installs the sandbox, as in every real child
    results = []
    for goal in goals:
        try:
            row = engine._janus.query_once(goal)
            results.append("failed" if not row.get("truth") else "ok " + json.dumps(
                {k: str(v) for k, v in row.items() if k != "truth"}, sort_keys=True))
        except Exception as exc:                                # noqa: BLE001
            results.append(f"raised {type(exc).__name__}: {str(exc)[:160]}")
    conn.send_bytes(json.dumps({"kind": "result", "results": results}).encode("utf-8"))


def _prolog_in_a_sandboxed_child(goals: list[str]) -> list[str]:
    """Run raw Prolog goals in a child whose sandbox is installed, the way a real child's is. For the sinks no
    MeTTa program can reach any more, which a test has to hit directly to show the guard is there."""
    child = mr._fork_child(_sandboxed_child, str(PETTA_HOME), goals)
    try:
        message = child.wait_message(time.monotonic() + 60)
    finally:
        child.close()
    assert message and message.get("kind") == "result", message
    return message["results"]


@real_petta
def test_a_closure_naming_something_that_is_not_a_function_is_not_called():
    """`apply_fn1/2/3/N` call the function a `partial(F, Args)` closure names without asking whether it is one.
    PeTTa only builds them for functions, and a program could write one only through `repra` (text parsed into a
    term), which is no longer allowed — so the sink is checked on its own. A real partial still works."""
    results = _prolog_in_a_sandboxed_child([
        'apply_fn1(partial(getenv, []), "HOME", Out)',
        'apply_fn2(partial(getenv, []), "HOME", x, Out)',
        'apply_fn3(partial(shell, []), a, b, c, Out)',
        'apply_fnN(partial(directory_files, []), ["."], Out)',
        "apply_fn1(partial(+, [1]), 2, Out)",                      # a closure over a real function
        "apply_fn1(+, 2, Out)",                                    # a bare name that is not complete: PeTTa's own reduce
    ])
    assert all(r.startswith("raised") and "permission_error" in r for r in results[:4]), results
    assert results[4] == 'ok {"Out": "3"}', results
    assert not results[5].startswith("raised"), results


@real_petta
def test_text_cannot_be_read_back_into_a_term_by_a_program(monkeypatch):
    """`repra` was `term_to_atom/2`: with its output already bound (which `let` does before the call) it parses
    text into a term, and `partial(getenv, [])` is a closure the shortcuts above would call. Not allowed now."""
    forged = '!(let "partial(getenv,[])" (repra $t) ((|-> ($f) ($f "PLN_TEST_CANARY_SECRET")) $t))'
    assert petta_refused_forms(forged) == ["repra"]                         # the literal spelling: the scan
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)   # the engine, on its own
    with PettaRuntime().session("") as session:
        with pytest.raises(MettaRuntimeError) as err:
            session.run(forged)
        assert err.value.kind == "refused" and "'repra'" in str(err.value)
        assert session.run('!(repr (a "b" 1))') == [['"(a \\"b\\" 1)"']]       # writing a term as text is still there


_FORGERY_AUDIT = r"""
:- use_module(library(time)).
audit_pool([foo, 'partial(getenv,[])', "partial(getenv,[])", 1, [a], _]).
%a compound that is not a list cell: nothing the pool holds, so something a builtin made out of it. Depth- and
%cycle-bounded: this is a defensive scan over adversarial inputs, so a builtin that hands back a cyclic or
%pathologically deep term (SWI's default occurs_check is off; several allowed builtins unify a fresh output
%variable without checking) must read as "not provably forged" rather than blow this process's own stack.
forged(T) :- cyclic_term(T), !, fail.
forged(T) :- forged(T, 5000).
forged(T, _) :- var(T), !, fail.
forged(T, D) :- compound(T), \+ T = [_|_], !, D > 0.
forged([H|T], D) :- D > 0, D1 is D - 1, ( forged(H, D1) ; forged(T, D1) ), !.
audit_args(0, []) :- !.
audit_args(N, [A|As]) :- N > 0, audit_pool(P), member(A, P), N1 is N - 1, audit_args(N1, As).
audit_builtin(Name, Arity, Report) :-
    functor(Goal, Name, Arity), Goal =.. [Name|Args],
    findall(Text,
            ( audit_args(Arity, Args),
              catch(call_with_time_limit(0.2, once(Goal)), _, fail),
              catch(call_with_time_limit(0.2, forged(Args)), _, fail),
              format(atom(Text), '~q', [Args]) ),
            Found),
    ( Found = [First|_] -> Report = First ; Report = none ).
"""


def _forgery_child(conn, home: str) -> None:
    mr._child_prepare(None, parent_pid=None, keep_fds=(conn.fileno(),))
    engine = mr._PettaEngine(home, 512)
    janus = engine._janus
    janus.consult("audit", data=_FORGERY_AUDIT)
    forged = {}
    for name in sorted(PETTA_ALLOWED_FUN_NAMES):
        for arity in janus.query_once("petta_sandbox_reachable_arities(N, O)", {"N": name})["O"]:
            if int(arity) > 3 or janus.query_once("petta_sandbox_is_generic_apply(N, A)",
                                                   {"N": name, "A": int(arity)}).get("truth"):
                continue                              # a generic-apply arity is already refused, not a forgery
            row = janus.query_once("audit_builtin(N, A, R)", {"N": name, "A": int(arity)})
            if row.get("truth") and str(row["R"]) != "none":
                forged[f"{name}/{arity}"] = str(row["R"])
    conn.send_bytes(json.dumps({"kind": "result", "forged": forged}).encode("utf-8"))


@real_petta
def test_no_allowed_builtin_makes_a_term_a_program_could_not_have_written():
    """A program can only write lists, atoms, numbers, strings and variables. Several sinks in PeTTa treat a
    compound term as a goal or a closure (`partial(F, Args)`), so a builtin that builds one out of what it is
    given — `repra`, which parsed text into a term — would be a way to a call by name. Every allowed builtin is
    called with every combination of adversarial arguments (text that looks like a term, in every position,
    output included); none may hand back a compound that is not a list. The audit that found `repra` ran the
    whole set at arity 4 and 11 values per position (six minutes); this is its fast cousin, and the way a
    builtin added to the allowed set later gets looked at."""
    child = mr._fork_child(_forgery_child, str(PETTA_HOME))
    try:
        message = child.wait_message(time.monotonic() + 300)
    finally:
        child.close()
    assert message and message.get("kind") == "result", message
    assert message["forged"] == {}, message["forged"]


@real_petta
def test_the_pure_arithmetic_slash_is_not_also_yalls_generic_apply(canaries, monkeypatch):
    """`/` is allowed as PeTTa's own arithmetic division (arity 3, defined in metta.pl). SWI's `library(yall)`
    — loaded for lambdas — ALSO exports `/` as a generic "apply a goal, closing over free variables" operator,
    at every OTHER arity (2, 4-9); PeTTa's own startup scan folded those into the SAME MeTTa builtin by name,
    so a name the allowlist calls pure arithmetic carried a call-by-name door at arities nobody meant to allow:
    `(/ {} getenv "SOME_VAR")` returned the server's environment variable. Division itself must still work."""
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)
    with PettaRuntime().session("") as session:
        assert session.run("!(/ 6 3)") == [["2"]]
        for expression in (f'!(/ {{}} getenv "{ENV_CANARY}")', f'!(/ {{}} directory_files "{canaries.tree}")'):
            with pytest.raises(MettaRuntimeError) as err:
                session.run(expression)
            assert err.value.kind == "refused" and "'/'" in str(err.value)
            assert ENV_VALUE not in str(err.value) and DIR_ENTRY not in str(err.value)
        assert session.run("!(/ 20 4)") == [["5"]]                        # still working afterwards


# ─── the add-atom family ───

TYPED_SPACE = "(: &jobs (SpaceOf (job Number)))\n"


@real_petta
@pytest.mark.parametrize("expressions, head", [
    (["!(add-atom &jobs (job 7))"], "add-atom-proven"),
    (["!(remove-atom &jobs (job 7))"], "remove-atom-proven"),
    (['!(let $f (atom_concat "add" "-atom") ($f &jobs (job 7)))'], "add-atom"),
])
def test_a_typed_space_is_not_a_way_around_the_add_atom_guard(monkeypatch, expressions, head):
    """For a literal `add-atom` on a space with a declared row type the translator emits
    `add-atom-proven/3`, a sibling the first guard did not cover: the fact was added and the query read it
    back. Text scan off, because the point is what the engine does with the form."""
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)
    with pytest.raises(MettaRuntimeError) as err:
        PettaRuntime().run_program(TYPED_SPACE, expressions)
    assert err.value.kind == "refused" and f"'{head}'" in str(err.value), str(err.value)


@real_petta
def test_a_typed_space_stays_empty_in_a_session_after_the_refusal(monkeypatch):
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)
    with PettaRuntime().session(TYPED_SPACE) as session:
        with pytest.raises(MettaRuntimeError):
            session.run("!(add-atom &jobs (job 7))")
        assert session.run("!(match &jobs (job $x) $x)") == [[]]


@real_petta
def test_a_function_in_the_kb_that_adds_to_a_typed_space_is_refused_when_it_runs(monkeypatch):
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)      # the KB scan refuses `add-atom` too
    kb = TYPED_SPACE + "(: add-job (-> Number Bool))\n(= (add-job $n) (add-atom &jobs (job $n)))\n"
    with pytest.raises(MettaRuntimeError) as err:
        PettaRuntime().run_program(kb, ["!(add-job 7)"])
    assert err.value.kind == "refused"


@real_petta
@pytest.mark.parametrize("branch, head", [
    ('(let $f (atom_concat "add" "-atom") ($f &self (ThreadAdded yes)))', "add-atom"),
    ("(call (getenv \"HOME\"))", "call"),
    ("(match getenv (HOME $v) $v)", "match"),
])
def test_a_thread_the_program_starts_is_sandboxed_too(monkeypatch, branch, head):
    """`hyperpose` runs its branches in threads of their own. The first version of the add-atom guard kept its
    flag in a global VARIABLE, which SWI-Prolog keeps per thread: a branch saw the guard switched off and
    `add-atom` changed the space (the previous version of this module did exactly that). The flag is a shared
    flag now, and so is the note a refusal leaves."""
    monkeypatch.setattr(mr, "_refuse_if_forbidden", lambda *a, **k: None)
    with PettaRuntime().session("(Fact a)") as session:
        with pytest.raises(MettaRuntimeError) as err:
            session.run(f"!(hyperpose ({branch} (+ 1 2)))")
        assert err.value.kind == "refused" and f"'{head}'" in str(err.value), str(err.value)
        assert session.run("!(match &self (ThreadAdded $x) $x)") == [[]]
        assert sorted(session.run("!(hyperpose ((+ 1 2) (* 2 3)))")[0]) == ["3", "6"]   # threads still work


@real_petta
def test_an_expression_cannot_add_a_fact_once_the_kb_is_loaded(monkeypatch):
    """`run_program` locks the space after the load: a plain fact riding behind an expression is refused
    instead of added (it used to be added, and read by the expressions after it)."""
    with pytest.raises(MettaRuntimeError) as err:
        PettaRuntime().run_program("(Fact a)", ["!(+ 1 2)\n(Late fact)", "!(match &self (Late $x) $x)"])
    assert err.value.kind == "refused" and "'add-atom'" in str(err.value)


@real_petta
def test_a_session_still_adds_plain_facts_and_a_kb_still_loads_its_own():
    with PettaRuntime().session("(Fact a)\n(Other b)") as session:
        assert session.run("(Fact c)\n!(collapse (match &self (Fact $x) $x))") == [["(a c)"]]
        assert session.run("!(match &self (Other $x) $x)") == [["b"]]


# ─── text the old splitter dropped ───

_BACKSLASH_KB = '!(+ 1 2)\n(Fact "a\\\n(")\n(Fact2 b)\n'


@real_petta
def test_facts_after_a_string_that_ends_a_line_with_a_backslash_are_not_lost():
    """A KB with a `!` form used to be split by a regex that did not know a backslash may escape a newline:
    the `(` inside the string unbalanced it and every fact after it vanished, without an error."""
    got = PettaRuntime().run_program(
        _BACKSLASH_KB, ["(collapse (match &self (Fact2 $x) $x))", "(collapse (match &self (Fact $x) $x))"])
    assert got == [["(b)"], ['("a\n(")']]


@real_petta
def test_a_session_text_with_such_a_string_is_not_dropped_either():
    with PettaRuntime().session("(Marker m)") as session:
        got = session.run('(Note "a\\\n(")\n!(+ 1 2)\n(Fact3 c)\n!(collapse (match &self (Fact3 $x) $x))')
    assert got == [["3"], ["(c)"]]


@real_petta
def test_text_the_engine_cannot_read_is_its_syntax_error_not_a_silent_loss():
    with pytest.raises(MettaRuntimeError) as err:
        PettaRuntime().run_program("!(+ 1 2)\n(Fact a", ["!(+ 1 2)"])
    assert err.value.kind == "load"
    with PettaRuntime().session("") as session:
        with pytest.raises(MettaRuntimeError):
            session.run("(Fact a")
        assert session.run("!(+ 1 2)") == [["3"]]


# ═══════════════════════ over HTTP, the way a caller reaches it ═══════════════════════

def _post_metta_run(monkeypatch, tmp_path, body: dict):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from unittest.mock import Mock
    import api
    import core.executor as executor_module

    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)       # inline, in this test process
    monkeypatch.setattr(api, "log_http_request", Mock())
    reset_runtime_cache()

    async def post():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/metta/run", json=body)

    try:
        return asyncio.run(post())
    finally:
        reset_runtime_cache()


def _detail(response) -> dict:
    body = response.json()
    detail = body.get("detail", body)
    return detail if isinstance(detail, dict) else {"message": str(detail)}


@real_petta
def test_the_review_reproduction_over_http_does_not_give_the_environment_away(canaries, monkeypatch, tmp_path):
    """`maplist getenv`, with the two symbols declared in `extra_atoms` so the validator calls them known —
    which returned the server's environment variable in a 200 before `maplist` stopped being a function."""
    response = _post_metta_run(monkeypatch, tmp_path, {
        "metta_query": f'!(maplist getenv ("{ENV_CANARY}"))',
        "extra_atoms": "(: getenv (-> String String))\n(: maplist (-> Atom Atom Atom))",
    })
    assert response.status_code == 422 and ENV_VALUE not in response.text, response.text
    assert "maplist" in response.text


@real_petta
@pytest.mark.parametrize("label, query, declared", [
    ("match", "!(match read_file_to_string ({secret} $v ()) $v)", "(: read_file_to_string (-> String Atom Atom Atom))"),
    ("built call", '!(let $c (atom_concat "ca" "ll") (let $e (cons-atom $c ((getenv "PLN_TEST_CANARY_SECRET"))) (eval $e)))',
     "(: atom_concat (-> String String Atom))\n(: getenv (-> String String))\n(: eval (-> Atom Atom))"),
])
def test_a_door_the_validator_cannot_see_is_a_422_from_the_engine(canaries, monkeypatch, tmp_path, label, query, declared):
    """Nothing in the text is a refused name, and declaring the symbols (`extra_atoms`, the documented way to
    introduce one) makes the validator call them known: only the engine can refuse these. The answer is the
    caller's fault, a 422 with the code `refused_form`, and it carries nothing of the host."""
    response = _post_metta_run(monkeypatch, tmp_path, {
        "metta_query": query.replace("{secret}", f'"{canaries.secret}"'),
        "extra_atoms": declared,
    })
    assert response.status_code == 422, response.text
    assert _detail(response).get("code") == "refused_form", response.text
    assert FILE_VALUE not in response.text and ENV_VALUE not in response.text
    assert str(REPO) not in str(_detail(response).get("message", ""))        # the server's own paths stay home


def test_a_refused_program_is_not_a_service_fault_in_the_documented_table():
    text = (PLN_CHAT / "API.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| `refused_form`"))
    assert "422" in row


@real_petta
def test_the_runner_both_tabs_share_reports_a_refusal_with_its_own_code(monkeypatch, tmp_path):
    """The chat tab and every route run a query through `pln_runner.run_query`: the code is decided there."""
    import core.pln_runner as pln_runner
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    reset_runtime_cache()
    try:
        literal = pln_runner.run_query("!(add-atom &self (Hacked yes))")
        built = pln_runner.run_query('!(let $f (atom_concat "add" "-atom") ($f &self (Hacked yes)))')
        fine = pln_runner.run_query("!(+ 1 2)")
    finally:
        reset_runtime_cache()
    for result in (literal, built):
        assert (result.status, result.error_code) == ("error", "refused_form"), result
        assert "add-atom" in result.error
    assert fine.status == "ok" and fine.error_code is None


@real_petta
def test_query_answers_a_refused_program_with_422_and_the_stage_it_died_in(monkeypatch, tmp_path):
    """`/query` runs whatever the model wrote, validated or not; a refused form is the model's fault, not the
    service's, and the response says where it stopped."""
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    import api
    import core.executor as executor_module
    from core.llm_translator import TranslationResult
    from ontology.registry import OntologyRegistry

    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)
    monkeypatch.setattr(api, "log_http_request", Mock())
    monkeypatch.setattr(api, "log_turn", Mock())
    monkeypatch.setattr(api, "_build_context", lambda selected: (OntologyRegistry(), {}))
    monkeypatch.setattr(api, "build_system_prompt", lambda registry, raw, inventory=None: "prompt")
    monkeypatch.setattr(api, "translate", lambda **kw: TranslationResult(
        metta_query="!(add-atom &self (Hacked yes))", explanation="", intent="inference",
        requires_pln_inference=True, confidence_filter=0.0))
    reset_runtime_cache()

    async def post():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/query", json={"message": "add a fact"})

    try:
        response = asyncio.run(post())
    finally:
        reset_runtime_cache()
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "refused_form" and detail["stage"] == "pln_execution"
    assert "add-atom" in detail["message"]


# ═══════════════════════ the validator and the documentation ═══════════════════════

def test_under_hyperon_a_head_ending_in_a_bang_reads_as_it_always_did():
    """The head pattern that keeps a trailing `!` exists for the PeTTa check; hyperon must read the heads it
    always read (`foo!` is `foo`), and PeTTa the head it now reads."""
    from core.metta_validator import validate
    from ontology.inventory import RuntimeInventory
    from ontology.registry import OntologyRegistry
    inventory = RuntimeInventory(declared_only={"foo"}, functions={"foo": 1})
    hyperon = validate("(foo! 1)", OntologyRegistry(), inventory, runtime="hyperon")
    petta = validate("(foo! 1)", OntologyRegistry(), inventory, runtime="petta")
    assert hyperon.ungrounded_predicates == ["foo"] and hyperon.valid
    assert petta.ungrounded_predicates == []


def test_the_validator_names_a_builtin_head_that_extra_atoms_declared():
    from core.metta_validator import validate
    from ontology.inventory import RuntimeInventory
    from ontology.registry import OntologyRegistry
    inventory = RuntimeInventory(functions={"maplist": 2, "getenv": 1})
    result = validate('(maplist getenv ("X"))', OntologyRegistry(), inventory, runtime="petta")
    assert not result.valid and any("maplist" in issue for issue in result.issues)


def test_the_api_doc_states_the_counts_the_module_computes():
    """API.md used to say 19 (and, in the same paragraph, 25) refused names where the module held 27: numbers
    that are right only if a test says so."""
    text = " ".join((PLN_CHAT / "API.md").read_text(encoding="utf-8").split())
    clauseless = mr._PETTA_CLAUSELESS_FUN_NAMES
    refused = PETTA_KNOWN_FUN_NAMES - PETTA_ALLOWED_FUN_NAMES - PETTA_ENGINE_INTERNAL_FUN_NAMES - clauseless
    assert PETTA_ENGINE_INTERNAL_FUN_NAMES <= PETTA_KNOWN_FUN_NAMES and clauseless <= PETTA_KNOWN_FUN_NAMES
    assert not (PETTA_ALLOWED_FUN_NAMES & (PETTA_ENGINE_INTERNAL_FUN_NAMES | clauseless))
    assert f"registers {len(PETTA_KNOWN_FUN_NAMES)} names; {len(PETTA_ALLOWED_FUN_NAMES)} of them are allowed" in text
    assert f"{len(refused)} are refused" in text
    assert f"{len(PETTA_ENGINE_INTERNAL_FUN_NAMES)} PeTTa calls itself" in text
    assert f"{len(clauseless)} have no Prolog predicate" in text
    # and the 19 that are refused for good are the ones it names
    always = refused - {"add-atom", "remove-atom"}
    assert f"the other {len(always)} always" in text
    assert all(f"`{name}`" in text for name in always)
