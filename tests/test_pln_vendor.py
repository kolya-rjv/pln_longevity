"""Tests for the vendored fork library (vendor/pln_fork) and for gate G0 on the real PeTTa.

What is vendored: patham9's PLN (patham9/PLN @ 55f1751) in the form its author ported to PeTTa, PeTTa's
`lib/lib_pln.metta` @ fea83178 (sha256 d864c3cf...), with ONE hunk (the per-step trace in PLN.Derive is removed).
`vendor/pln_fork/lib_pln_restricted.metta` is GENERATED from it by `scripts/make_pln_restricted.py`, a form-level
filter that keeps Revision, Modus Ponens and Not-elimination and copies every kept form byte for byte.

Three layers, cheapest first:

* the vendoring itself (no PeTTa): the upstream hash is recovered from the vendored file by re-applying the hunk
  backwards, both MIT licences travel with it, the restricted artifact equals a fresh build, every kept form is
  byte-identical to the vendored one, the filter refuses a form it has no rule for, and the artifact passes the
  PeTTa sandbox's text scan (the unfiltered file does not: it holds `Predicate`);
* gate G0 on the real PeTTa (`-m petta`, skipped when PeTTa cannot run here): the load-time probes compared in Python,
  `PLN.Derive` returning exactly one result, the stamp (0) and `(State X)` terms passing the library's Sentence
  patterns, `pln_tuffy` on the restricted artifact, `pln_roman` on the UNFILTERED file (via PeTTa's run.sh on a copy);
* the closure of the shipped graphs on the real library with the decision's encoding (state sentences + Modus Ponens,
  strength stored as q = e + (1 - e) p, reported as |s1 - s0| / (1 - e)^hops, confidence times 0.9^(hops - 1)):
  100 of 100 patient pairs, 86 of 86 LinAge2 pairs, 101 generic pairs, 109 pairs with the feedback edge, each equal
  to the legacy chainer in sign, strength, confidence and chain, none extra, none missing; the D+Q levels; the
  pure-Python simulator `docs/petta_migration/fork_design/sim/fork_sim.py` as an independent oracle.

No test calls a model. Nothing here runs under hyperon: the library is PeTTa-only.

    pytest tests/test_pln_vendor.py -q                 # everything that can run here
    pytest tests/test_pln_vendor.py -q -m "not petta"  # the vendoring checks only
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

VENDOR = REPO / "vendor" / "pln_fork"
FORK_DESIGN = REPO / "docs" / "petta_migration" / "fork_design"
CONTRACT = FORK_DESIGN / "contract"
LIB_FULL = VENDOR / "lib_pln_fork.metta"
LIB_RESTRICTED = VENDOR / "lib_pln_restricted.metta"
PETTA_HOME = REPO / ".petta" / "PeTTa"

#: PeTTa lib/lib_pln.metta @ fea83178af3722666d2948d074591d10a3134f56
UPSTREAM_SHA256 = "d864c3cf376ad1f9252d153306d92b7aae72a184701c2eb4805c9661eed27ece"
#: the vendored file itself (provenance header + upstream file with the one hunk)
VENDORED_SHA256 = "4b01b7c66c5294533e6a2496eb20958d6600d62fab40ce53e361e6190cfaa018"
LICENSE_SHA256 = {
    "LICENSE.NARTECH": "922e8a4a2003a4325dd44fa8e8f16cd7277c513de447ea0a3a6b0908c81bee51",
    "LICENSE.PeTTa": "451e08f082f06f100facd2ef99f3c661155a7a433af091889d4c04e264838ce5",
}
HEADER_END = "; ---- END OF PROVENANCE HEADER ----"

#: The single hunk, as the text before (upstream) and after (vendored). The vendored body is the upstream file
#: with OLD replaced by NEW; the test turns that around and checks the upstream hash.
HUNK_OLD = """\
                 (let $temp (trace! (SELECTED $steps (Sentence $x $Ev1)) 42)
                            (PLN.Derive (LimitSize (Without (Unique (TupleConcat $Tasks $derivations) ()) (Sentence $x $Ev1)) $taskqueuesize)
                                        (LimitSize (Unique (TupleConcat $Beliefs $derivations) ()) $beliefqueuesize)
                                        (+ $steps 1)
                                        $maxsteps
                                        $taskqueuesize
                                        $beliefqueuesize))))))
"""
HUNK_NEW = """\
                 (PLN.Derive (LimitSize (Without (Unique (TupleConcat $Tasks $derivations) ()) (Sentence $x $Ev1)) $taskqueuesize)
                             (LimitSize (Unique (TupleConcat $Beliefs $derivations) ()) $beliefqueuesize)
                             (+ $steps 1)
                             $maxsteps
                             $taskqueuesize
                             $beliefqueuesize)))))
"""


def _load_filter():
    spec = importlib.util.spec_from_file_location("make_pln_restricted", REPO / "scripts" / "make_pln_restricted.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_pln_restricted"] = mod      # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


FILTER = _load_filter()


def _body(path: Path) -> str:
    return FILTER.vendored_body(path.read_text(encoding="utf-8"))


# ══════════════════════════════ the vendoring (no PeTTa) ═══════════════════════

def test_the_vendored_file_is_the_upstream_file_with_one_hunk():
    assert hashlib.sha256(LIB_FULL.read_bytes()).hexdigest() == VENDORED_SHA256
    body = _body(LIB_FULL)
    assert body.count(HUNK_NEW) == 1
    assert HUNK_OLD not in body
    upstream = body.replace(HUNK_NEW, HUNK_OLD)
    assert hashlib.sha256(upstream.encode("utf-8")).hexdigest() == UPSTREAM_SHA256
    # the hunk drops exactly the trace wrapper and nothing else: no other line differs
    assert "trace!" not in body and "SELECTED" not in body
    assert len(upstream.splitlines()) - len(body.splitlines()) == 1


def test_the_vendored_file_matches_the_pinned_checkout_when_it_is_present():
    bundled = PETTA_HOME / "lib" / "lib_pln.metta"
    if not bundled.exists():
        pytest.skip("the pinned PeTTa checkout is not here")
    assert hashlib.sha256(bundled.read_bytes()).hexdigest() == UPSTREAM_SHA256
    assert _body(LIB_FULL).replace(HUNK_NEW, HUNK_OLD) == bundled.read_text(encoding="utf-8")


def test_the_header_names_the_pins_and_the_licences():
    text = LIB_FULL.read_text(encoding="utf-8")
    header = text.split(HEADER_END, 1)[0]
    for needle in ("55f1751d993f71b8a24da03e3aec94ab40789a59", "fea83178af3722666d2948d074591d10a3134f56",
                   UPSTREAM_SHA256, "NARTECH", "Patrick Hammer", "LICENSE.NARTECH", "LICENSE.PeTTa"):
        assert needle in header, needle
    assert all(line.startswith(";") for line in header.strip().splitlines()), "the header must be comments only"


def test_both_mit_licences_travel_with_the_library():
    for name, sha in LICENSE_SHA256.items():
        data = (VENDOR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, name
        assert data.startswith(b"MIT License")
    assert b"Copyright (c) 2025 NARTECH" in (VENDOR / "LICENSE.NARTECH").read_bytes()
    assert b"Copyright (c) 2025 Patrick Hammer" in (VENDOR / "LICENSE.PeTTa").read_bytes()


def test_provenance_md_records_what_a_refresh_needs():
    text = (VENDOR / "PROVENANCE.md").read_text(encoding="utf-8")
    for needle in ("55f1751d993f71b8a24da03e3aec94ab40789a59", "fea83178af3722666d2948d074591d10a3134f56", UPSTREAM_SHA256,
                   "MIT", "NARTECH", "make_pln_restricted.py", "tests/test_pln_vendor.py", "hunk"):
        assert needle in text, needle


def test_the_restricted_artifact_is_a_fresh_build():
    assert LIB_RESTRICTED.read_text(encoding="utf-8") == FILTER.build()


def test_every_kept_form_is_byte_identical_to_the_vendored_form():
    body = _body(LIB_FULL)
    vendored = FILTER.split_forms(body)
    restricted = FILTER.split_forms(LIB_RESTRICTED.read_text(encoding="utf-8"))
    kept = [f for f in vendored if FILTER.classify(f) == "keep"]
    assert [f.text for f in restricted] == [f.text for f in kept]
    # in file order, and each is a verbatim substring of the vendored body
    position = -1
    for f in restricted:
        at = body.find(f.text, position + 1)
        assert at > position, f.text[:60]
        position = at
    assert len(vendored) == len(kept) + len([f for f in vendored if FILTER.classify(f) == "drop"])
    assert len(kept) < len(vendored)


def test_the_restricted_rule_set_is_revision_modus_ponens_and_not_elimination():
    restricted = FILTER.split_forms(LIB_RESTRICTED.read_text(encoding="utf-8"))
    rules = sorted(f.rule for f in restricted if f.name == "|-")
    assert rules == sorted(FILTER.KEEP_RULES)
    assert sorted(FILTER.KEEP_RULES.values()) == ["ModusPonens", "NotElimination", "Revision"]
    text = LIB_RESTRICTED.read_text(encoding="utf-8")
    code = re.sub(r";[^\n]*", "", text)
    for gone in ("Truth_Deduction", "Truth_Induction", "Truth_Abduction", "Truth_inversion", "Truth_equivalenceToImplication",
                 "Truth_transitiveSimilarity", "Truth_evaluationImplication", "Truth_SymmetricModusPonens", "SyllogisticRuleGuard",
                 "SymmetricModusPonensRuleGuard", "STV", "Predicate", "Evaluation", "Inheritance", "Equivalence", "Similarity", "Member"):
        assert gone not in code, gone
    for there in ("Truth_ModusPonens", "Truth_Negation", "Truth_Revision", "PLN.Derive", "PLN.Query", "PLN.Config.MaxSteps",
                  "StampDisjoint", "BestCandidate", "LimitSize", "PriorityRank", "ConfidenceRank"):
        assert there in code, there


def test_every_rule_of_the_vendored_file_is_classified():
    vendored = FILTER.split_forms(_body(LIB_FULL))
    assert all(FILTER.classify(f) in ("keep", "drop") for f in vendored)
    rules = [f for f in vendored if f.name == "|-"]
    assert len(rules) == 17                      # the fork's 17 rules, 14 distinct premise patterns
    assert sum(1 for f in rules if FILTER.classify(f) == "keep") == 3


def test_the_filter_refuses_a_form_it_has_no_rule_for(tmp_path):
    src = tmp_path / "lib_pln_fork.metta"
    src.write_text(LIB_FULL.read_text(encoding="utf-8") + "\n(= (|- ((Brand $A) $T)) ($A $T))\n", encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        FILTER.build(src)
    assert "no rule for" in str(err.value)
    src.write_text(LIB_FULL.read_text(encoding="utf-8") + "\n(= (BrandNewHelper $x) $x)\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        FILTER.build(src)


def test_the_filter_refuses_when_its_tables_name_a_form_the_file_no_longer_has(tmp_path):
    src = tmp_path / "lib_pln_fork.metta"
    text = LIB_FULL.read_text(encoding="utf-8")
    start = text.index("(= (Truth_Negation")
    end = text.index("(= (Truth_inversion")
    src.write_text(text[:start] + text[end:], encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        FILTER.build(src)
    assert "no longer has" in str(err.value)


def test_the_filter_check_flag_reports_a_stale_artifact(tmp_path):
    stale = tmp_path / "lib_pln_restricted.metta"
    stale.write_text(LIB_RESTRICTED.read_text(encoding="utf-8") + "\n; edited by hand\n", encoding="utf-8")
    assert FILTER.main(["--check", "--target", str(stale)]) == 1
    assert FILTER.main(["--check", "--target", str(LIB_RESTRICTED)]) == 0


def test_the_library_defines_no_name_the_knowledge_base_defines():
    """PeTTa function names are process-global, and an extra `(= (name ..))` clause ADDS answers. The 48 functions of the vendored
    file (31 of them in the restricted artifact) must not be a function or a fact head anywhere in the repo's knowledge base."""
    defined = set()
    for path in (LIB_FULL, LIB_RESTRICTED):
        defined |= {f.name for f in FILTER.split_forms(_body(LIB_FULL) if path == LIB_FULL else path.read_text(encoding="utf-8"))
                    if f.kind == "def" and f.name != "|-"}
    assert len(defined) == 48
    taken = {}
    for path in sorted(REPO.glob("*.metta")):
        text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8"))
        for m in re.finditer(r"\(=\s+\(\s*([^\s()]+)", text):
            taken.setdefault(m.group(1), path.name)
        for m in re.finditer(r"^\(([^\s()=:][^\s()]*)", text, re.M):
            taken.setdefault(m.group(1), path.name)
    assert sorted(n for n in defined if n in taken) == []


def test_the_sandbox_text_scan_passes_the_artifact_and_refuses_the_unfiltered_file_for_Predicate_only():
    from core.metta_runtime import petta_refused_forms_in_kb
    assert petta_refused_forms_in_kb(LIB_RESTRICTED.read_text(encoding="utf-8")) == []
    assert petta_refused_forms_in_kb(LIB_FULL.read_text(encoding="utf-8")) == ["Predicate"]


# ══════════════════════════════ the real PeTTa ═════════════════════════════════

def _real_petta_available() -> bool:
    try:
        from core.metta_runtime import PettaRuntime
        if not PettaRuntime().available():
            return False
        return PettaRuntime().run_program("", ["!(+ 1 2)"]) == [["3"]]
    except Exception:
        return False


REAL_PETTA = _real_petta_available()
requires_petta = pytest.mark.skipif(not REAL_PETTA, reason="PeTTa is not installed here (SWI-Prolog, janus-swi and the "
                                                            "pinned checkout; bash scripts/setup_petta.sh)")


def real_petta(fn):
    return pytest.mark.petta(requires_petta(fn))


# ── a compact driver: s-expressions in and out, PLN.Derive in a session of the production adapter ──────────────

def _sexp(t) -> str:
    if isinstance(t, tuple):
        return "(" + " ".join(_sexp(x) for x in t) + ")"
    if isinstance(t, float):
        r = repr(t)
        assert "e" not in r and "inf" not in r and "nan" not in r, r
        return r
    return str(t)


def _load_sim():
    spec = importlib.util.spec_from_file_location("fork_sim", FORK_DESIGN / "sim" / "fork_sim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fork_sim"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse(text):
    forms = _load_sim().parse_sexprs(text)
    assert len(forms) == 1, text[:120]
    return forms[0]


def _sentence_out(p):
    assert p[0] == "Sentence", p
    (term, stv), ids = p[1], p[2]
    return (term, (stv[1], stv[2]), tuple(ids))


class Lib:
    """One PeTTa child (a session of core.metta_runtime) holding the restricted artifact, driven with data only."""

    def __init__(self, text: str):
        from core.metta_runtime import PettaRuntime
        self.session = PettaRuntime().session(text, 300)

    def close(self):
        self.session.close()

    def eval(self, expr: str) -> list[str]:
        groups = self.session.run("!" + expr, 300)
        assert len(groups) == 1
        return groups[0]

    def one(self, expr: str) -> str:
        out = self.eval(expr)
        assert len(out) == 1, out
        return out[0]

    def derive(self, tasks, beliefs, steps, maxsteps, tq, bq):
        def lit(xs):
            return _sexp(tuple(("Sentence", (t, ("stv", s, c)), tuple(ids)) for (t, (s, c), ids) in xs))
        out = self.eval("(PLN.Derive %s %s %d %d %d %d)" % (lit(tasks), lit(beliefs), steps, maxsteps, tq, bq))
        assert len(out) == 1, "PLN.Derive must return exactly one result, got %d" % len(out)
        parsed = _parse(out[0])
        assert len(parsed) == 2
        return [_sentence_out(x) for x in parsed[0]], [_sentence_out(x) for x in parsed[1]]


@pytest.fixture(scope="module")
def lib():
    if not REAL_PETTA:
        pytest.skip("PeTTa is not installed here")
    handle = Lib(LIB_RESTRICTED.read_text(encoding="utf-8"))
    yield handle
    handle.close()


def _near(a, b, tol=1e-15):
    return abs(a - b) <= tol


# ── G0 (b): the load-time probes, compared in Python (never MeTTa test/assert: they halt the process) ───────────

@real_petta
def test_modus_ponens_probe_and_the_leak(lib):
    v = _parse(lib.one("(Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8))"))
    assert v[0] == "stv" and _near(v[1], 0.26) and _near(v[2], 0.64)
    leak = _parse(lib.one("(Truth_ModusPonens (stv 0.0 1.0) (stv 0.0 1.0))"))
    assert leak[0] == "stv" and _near(leak[1], 0.02) and _near(leak[2], 1.0)


@real_petta
def test_truth_revision_with_full_confidence_gives_no_result(lib):
    for expr in ("(Truth_Revision (stv 0.5 1.0) (stv 0.5 0.5))", "(Truth_Revision (stv 0.5 0.5) (stv 0.5 1.0))",
                 "(Truth_Revision (stv 0.5 1.0) (stv 0.5 1.0))"):
        assert lib.eval(expr) == [], expr          # no result, no exception
    capped = _parse(lib.one("(Truth_Revision (stv 0.5 0.999999) (stv 0.5 0.5))"))
    assert capped[0] == "stv" and capped[2] < 1.0


@real_petta
def test_revision_fold_equals_hand_arithmetic(lib):
    """The four Neg rows of MixCompound (design 1, section 10): Neg 0.263050, native c 0.949466."""
    rows = [(0.5, 0.9), (0.6666666666666666, 0.35), (0.3333333333333333, 0.2), (0.0, 0.9)]
    got = _parse(lib.one("(foldl-atom ((stv 0.6666666666666666 0.35) (stv 0.3333333333333333 0.2) (stv 0.0 0.9)) "
                         "(stv 0.5 0.9) Truth_Revision)"))
    acc = rows[0]
    for f2, c2 in rows[1:]:
        w1, w2 = acc[1] / (1 - acc[1]), c2 / (1 - c2)
        w = w1 + w2
        acc = (min(1.0, (w1 * acc[0] + w2 * f2) / w), min(1.0, max(max(w / (w + 1), acc[1]), c2)))
    assert _near(got[1], acc[0]) and _near(got[2], acc[1])
    assert abs(got[1] - 0.263050) < 5e-7 and abs(got[2] - 0.949466) < 5e-7


@real_petta
def test_stamp_zero_and_state_terms_pass_the_library_sentence_patterns(lib):
    assert lib.one("(PriorityRank (Sentence ((State X) (stv 1.0 1.0)) (0)))") == "1.0"
    assert lib.one("(PriorityRankNeg (Sentence ((State X) (stv 1.0 1.0)) (0)))") in ("-1.0", "-1")
    assert lib.one("(PriorityRank (Sentence ((End X) (stv 0.3 0.65)) (1 2)))") == "0.65"
    assert lib.one("(ConfidenceRank ((stv 1.0 1.0) (0)))") == "1.0"
    assert lib.one("(StampDisjoint (0) (1 2))") == "True"
    assert lib.one("(StampDisjoint (0) (0))") == "False"
    assert lib.one("(StampDisjoint (0 3) (0 4))") == "False"
    best = lib.one("(BestCandidate PriorityRank () ((Sentence ((State A) (stv 1.0 1.0)) (0)) "
                   "(Sentence ((State B) (stv 0.5 0.9)) (1))))")
    assert best.startswith("(Sentence ((State A) (stv 1.0 1.0)) (0))")


@real_petta
def test_the_three_rules_fire_on_state_and_end_terms(lib):
    mp = _parse(lib.one("(|- ((State A) (stv 1.0 1.0)) ((Implication (State A) (State B)) (stv 0.706 0.65)))"))
    assert mp[0] == ("State", "B") and _near(mp[1][1], 0.706) and _near(mp[1][2], 0.65)
    neg = _parse(lib.one("(|- ((State A) (stv 1.0 1.0)) ((Implication (State A) (Not (End B))) (stv 0.706 0.65)))"))
    assert neg[0] == ("Not", ("End", "B"))
    ne = _parse(lib.one("(|- ((Not (End B)) (stv 0.706 0.65)))"))
    assert ne[0] == ("End", "B") and _near(ne[1][1], 0.294) and _near(ne[1][2], 0.65)
    # a hazard-ratio record can end a chain and never continue one
    assert lib.eval("(|- ((End B) (stv 0.5 0.9)) ((Implication (State B) (State C)) (stv 0.7 0.9)))") == []


@real_petta
def test_derive_returns_exactly_one_result(lib):
    p = (("State", "A"), (1.0, 1.0), (0,))
    k = (("Implication", ("State", "A"), ("State", "B")), (0.7, 0.9), (1,))
    for tasks, beliefs, steps, maxsteps in (([p], [k, p], 1, 100), ([], [k], 1, 100), ([p], [p], 5, 4), ([p], [k, p], 1, 1024)):
        out_tasks, out_beliefs = lib.derive(tasks, beliefs, steps, maxsteps, 256, 256)
        assert isinstance(out_tasks, list) and isinstance(out_beliefs, list)
    t, b = lib.derive([p], [k, p], 1, 100, 10, 100)
    assert t == [] and (("State", "B"), (0.7, 0.9), (0, 1)) in b


# ── G0 (a): the upstream examples ──────────────────────────────────────────────────────────────────────────────

def _run_example(tmp_path: Path, example: str, lib_path: Path):
    src = (PETTA_HOME / "examples" / ("%s.metta" % example)).read_text(encoding="utf-8")
    fixed = re.sub(r"^!\(import! &self (?:\.\./lib/lib_pln|\(library lib_pln\))\)", '!(import! &self "%s")' % lib_path, src, flags=re.M)
    assert fixed != src
    copy = tmp_path / ("%s.metta" % example)
    copy.write_text(fixed, encoding="utf-8")
    done = subprocess.run(["sh", str(PETTA_HOME / "run.sh"), str(copy)], capture_output=True, text=True, timeout=180, cwd=tmp_path)
    return done.returncode, done.stdout + done.stderr


@real_petta
@pytest.mark.skipif(not (PETTA_HOME / "run.sh").exists(), reason="PeTTa's run.sh is not here")
@pytest.mark.parametrize("example,which,passes", [
    ("pln_roman", "full", 1), ("pln_tuffy", "full", 1), ("pln_direct", "full", 3),
    ("pln_tuffy", "restricted", 1), ("pln_direct", "restricted", 3)])
def test_upstream_examples_give_their_pinned_values_on_the_vendored_library(tmp_path, example, which, passes):
    rc, text = _run_example(tmp_path, example, LIB_FULL if which == "full" else LIB_RESTRICTED)
    assert rc == 0, text[-600:]
    assert text.count("✅") == passes and "❌" not in text


@real_petta
@pytest.mark.skipif(not (PETTA_HOME / "run.sh").exists(), reason="PeTTa's run.sh is not here")
def test_pln_roman_needs_deduction_so_it_fails_on_the_restricted_artifact(tmp_path):
    rc, text = _run_example(tmp_path, "pln_roman", LIB_RESTRICTED)
    assert "❌" in text and "✅" not in text


def _example_kb(example: str) -> str:
    src = (PETTA_HOME / "examples" / ("%s.metta" % example)).read_text(encoding="utf-8")
    src = "\n".join(l for l in src.split("\n") if not l.startswith("!(import!"))
    out, k = [], 0
    while k < len(src):                         # drop the `!(test ...)` forms (balanced)
        if src.startswith("!(test", k):
            depth, j = 0, k + 1
            while True:
                depth += (src[j] == "(") - (src[j] == ")")
                if depth == 0:
                    break
                j += 1
            k = j + 1
            continue
        out.append(src[k])
        k += 1
    return "".join(out)


@real_petta
def test_pln_tuffy_on_the_restricted_artifact_through_the_adapter_session():
    handle = Lib(LIB_RESTRICTED.read_text(encoding="utf-8") + "\n" + _example_kb("pln_tuffy"))
    try:
        got = _parse(handle.one("(PLN.Query (kb) (Inheritance Edward (IntSet cancerous)))"))
    finally:
        handle.close()
    assert got == (("stv", 0.5199204545454544, 0.8290782204129108), (1, 2, 4, 9, 10))


# ── G0 (c): the closure of the shipped graphs on the real library ──────────────────────────────────────────────

SIGN = {("Pos", "Pos"): "Pos", ("Pos", "Neg"): "Neg", ("Neg", "Pos"): "Neg", ("Neg", "Neg"): "Pos"}
TOL = 1e-12
CHAIN_DISCOUNT = 0.9
SETTINGS = {"lib defaults 100/10/100": (100, 10, 100), "fork defaults 20/20/200": (20, 20, 200), "K=256": (1024, 256, 256)}


def _links(stack: str) -> list[dict]:
    data = json.loads((CONTRACT / "links.json").read_text(encoding="utf-8"))
    out = [dict(l) for l in data["links"] if stack in l["stacks"]]
    return _with_ids(out)


def _with_ids(links: list[dict]) -> list[dict]:
    order = sorted(range(len(links)), key=lambda i: (links[i]["from"], links[i]["to"], links[i]["sign"],
                                                     links[i]["terminal_only"], links[i]["strength"], links[i]["confidence"]))
    for n, i in enumerate(order, start=1):
        links[i]["id"] = n                       # 0 is the premise; one id per link, so no id repeats along a path
    assert len({l["id"] for l in links}) == len(links)
    return links


def _link_term(l):
    tgt = ("End" if l["terminal_only"] else "State", l["to"])
    return ("Implication", ("State", l["from"]), tgt if l["sign"] == "Pos" else ("Not", tgt))


def _kb(links, leak):
    """q = e + (1 - e) p: exact for every p in [0, 1] (the earlier q = min(1, p + e) shrinks p above 0.98)."""
    return {l["id"]: (_link_term(l), (leak + (1.0 - leak) * l["strength"], l["confidence"]), (l["id"],)) for l in links}


def _reachable(links, a):
    out_by = {}
    for l in links:
        out_by.setdefault(l["from"], []).append(l)
    seen, frontier, keep = {a}, [a], []
    while frontier:
        nxt = []
        for x in frontier:
            for l in out_by.get(x, []):
                keep.append(l)
                if l["to"] not in seen and not l["terminal_only"]:
                    seen.add(l["to"])
                    nxt.append(l["to"])
        frontier = nxt
    return keep


def _legacy(links, a):
    """Node-simple legacy `infer` from a: [(nodes, sign, s, c)], right-nested floats as in pln_deduction.metta."""
    out_by = {}
    for l in links:
        out_by.setdefault(l["from"], []).append(l)

    def derive(x, visited):
        res = []
        for l in out_by.get(x, []):
            b = l["to"]
            if b in visited:
                continue
            res.append(((x, b), l["sign"], l["strength"], l["confidence"]))
            if not l["terminal_only"]:
                for (nodes, sg, s, c) in derive(b, visited | {b}):
                    res.append(((x,) + nodes, SIGN[(l["sign"], sg)], l["strength"] * s, (l["confidence"] * c) * CHAIN_DISCOUNT))
        return res
    return derive(a, frozenset([a]))


def _path(by_id, a, stamp):
    used = [by_id[i] for i in stamp if i != 0]
    nxt = {}
    for l in used:
        if l["from"] in nxt:
            return None
        nxt[l["from"]] = l
    path, cur = [a], a
    while cur in nxt:
        cur = nxt[cur]["to"]
        path.append(cur)
        if len(path) > len(nxt) + 1:
            return None
    return path if len(path) == len(nxt) + 1 else None


def _routes(b1, b0, by_id, a, leak):
    """The adapter step: join the two runs on the stamp, rebuild the node-simple path, strength |s1 - s0| / (1-e)^hops,
    sign = parity of Neg links (cross-checked against the sign of s1 - s0), confidence times 0.9^(hops-1)."""
    idx0 = {(t[1], ev): tv for (t, tv, ev) in b0 if isinstance(t, tuple) and len(t) == 2 and t[0] in ("State", "End")}
    out = {}
    for (t, (s1, c1), ev) in b1:
        if not (isinstance(t, tuple) and len(t) == 2 and t[0] in ("State", "End")) or t[1] == a or c1 <= 0:
            continue
        path = _path(by_id, a, ev)
        if path is None or path[-1] != t[1] or len(set(path)) != len(path) or (t[1], ev) not in idx0:
            continue
        hops = len(path) - 1
        ate = s1 - idx0[(t[1], ev)][0]
        sign = "Neg" if sum(1 for i in ev if i and by_id[i]["sign"] == "Neg") % 2 else "Pos"
        if abs(ate) > 1e-12:
            assert ("Neg" if ate < 0 else "Pos") == sign, (path, ate, sign)
        out[tuple(path)] = (sign, abs(ate) / (1.0 - 0.02) ** hops, c1 * CHAIN_DISCOUNT ** (hops - 1))
    return out


def _run_source(handle, kb_by_id, links, leak, a, level, budget):
    prem = (("State", a), (level, 1.0), (0,))            # id 0 and c = 1.0: an empty stamp with c < 1 floods the queue
    beliefs = [kb_by_id[l["id"]] for l in _reachable(links, a)] + [prem]
    steps, tq, bq = budget
    tasks, final = handle.derive([prem], beliefs, 1, steps, tq, bq)
    return tasks, final


def _closure(handle, links, budget, leak=0.02):
    by_id = {l["id"]: l for l in links}
    kb_by_id = _kb(links, leak)
    nodes = sorted({l["from"] for l in links} | {l["to"] for l in links})
    pairs, bad, missing, extra, dev = 0, [], [], [], 0.0
    for a in nodes:
        t1, b1 = _run_source(handle, kb_by_id, links, leak, a, 1.0, budget)
        t0, b0 = _run_source(handle, kb_by_id, links, leak, a, 0.0, budget)
        assert t1 == [] and t0 == [], "the run did not finish: tasks left for %s" % a
        mine = _routes(b1, b0, by_id, a, leak)
        leg = {nodes_: (sg, s, c) for (nodes_, sg, s, c) in _legacy(links, a)}
        pairs += len({p[-1] for p in mine})
        missing += [p for p in leg if p not in mine]
        extra += [p for p in mine if p not in leg]
        for p, (sg, s, c) in mine.items():
            if p in leg:
                lg, ls, lc = leg[p]
                dev = max(dev, abs(s - ls), abs(c - lc))
                if sg != lg:
                    bad.append(p)
    return pairs, missing, extra, bad, dev


@pytest.fixture(scope="module")
def closure_lib():
    if not REAL_PETTA:
        pytest.skip("PeTTa is not installed here")
    handle = Lib(LIB_RESTRICTED.read_text(encoding="utf-8"))
    yield handle
    handle.close()


@real_petta
@pytest.mark.parametrize("stack,pairs", [("patient", 100), ("linage2", 86), ("generic", 101)])
@pytest.mark.parametrize("budget", list(SETTINGS.values()), ids=list(SETTINGS))
def test_the_closure_equals_the_legacy_chainer(closure_lib, stack, pairs, budget):
    """100 / 86 / 101 ordered pairs: sign, strength, confidence (with the 0.9 shim) and chain equal to 1e-12, none
    extra, none missing. The generic stack holds Lifespan -> Mortality at p = 1.0, where the capped mapping broke."""
    got, missing, extra, bad, dev = _closure(closure_lib, _links(stack), budget)
    assert (got, missing, extra, bad) == (pairs, [], [], [])
    assert dev <= TOL


@real_petta
def test_the_legacy_reference_in_this_file_reproduces_the_hyperon_closure():
    """The oracle itself: bit-identical to closure.json (real hyperon output) on both recorded stacks."""
    for stack in ("patient", "linage2"):
        links = _links(stack)
        ref = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"])
               for d in json.loads((CONTRACT / "closure.json").read_text(encoding="utf-8"))["stacks"][stack]["derivations"]}
        mine = {}
        for a in sorted({l["from"] for l in links} | {l["to"] for l in links}):
            for (nodes, sg, s, c) in _legacy(links, a):
                mine[nodes] = (sg, s, c)
        assert mine == ref, stack


@real_petta
def test_the_earlier_capped_mapping_is_caught_by_the_generic_stack(closure_lib):
    """Negative control: with q = min(1, p + e) the slope of Lifespan -> Mortality (p = 1.0) is 0.98, so the legacy 1.0 is reported
    as 0.98 and every chain through it loses 2 percent; with q = e + (1 - e) p the reported strength is exactly 1.0."""
    links = _links("generic")

    def mortality_difference(kb):
        _, b1 = _run_source(closure_lib, kb, links, 0.02, "Lifespan", 1.0, SETTINGS["K=256"])
        _, b0 = _run_source(closure_lib, kb, links, 0.02, "Lifespan", 0.0, SETTINGS["K=256"])
        s1 = [s for (t, (s, c), ev) in b1 if t == ("State", "Mortality")]
        s0 = [s for (t, (s, c), ev) in b0 if t == ("State", "Mortality")]
        assert len(s1) == len(s0) == 1
        return abs(s1[0] - s0[0])
    capped = {l["id"]: (_link_term(l), (min(1.0, l["strength"] + 0.02), l["confidence"]), (l["id"],)) for l in links}
    assert abs(mortality_difference(capped) - 0.98) < 1e-12          # the earlier mapping reports |s1 - s0| as is
    assert abs(mortality_difference(_kb(links, 0.02)) / (1.0 - 0.02) - 1.0) < 1e-12


@real_petta
@pytest.mark.parametrize("budget", list(SETTINGS.values()), ids=list(SETTINGS))
def test_a_feedback_edge_terminates_and_adds_nine_pairs(closure_lib, budget):
    """ChronicInflammation -> CellularSenescence closes CS -> SASP -> CI -> CS. Legacy hangs; the fork terminates
    (a derivation uses each link id once): 109 pairs, the 100 old answers unchanged, 9 new."""
    old = _links("patient")
    links = [dict(l) for l in old]
    for l in links:
        l.pop("id", None)
    links.append({"from": "ChronicInflammation", "to": "CellularSenescence", "sign": "Pos", "strength": 0.5, "confidence": 0.65,
                  "terminal_only": False, "kind": "Effect", "source": "curated"})
    links = _with_ids(links)
    got, missing, extra, bad, dev = _closure(closure_lib, links, budget)
    assert (got, missing, extra, bad) == (109, [], [], [])
    assert dev <= TOL
    recorded = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"])
                for d in json.loads((CONTRACT / "closure.json").read_text(encoding="utf-8"))["stacks"]["patient"]["derivations"]}
    by_id = {l["id"]: l for l in links}
    kb = _kb(links, 0.02)
    same, new = 0, set()
    for a in sorted({l["from"] for l in links}):
        _, b1 = _run_source(closure_lib, kb, links, 0.02, a, 1.0, budget)
        _, b0 = _run_source(closure_lib, kb, links, 0.02, a, 0.0, budget)
        for p, (sg, s, c) in _routes(b1, b0, by_id, a, 0.02).items():
            if p in recorded:
                same += (recorded[p][0] == sg and abs(recorded[p][1] - s) <= TOL and abs(recorded[p][2] - c) <= TOL)
            else:
                new.add(p)
    assert (same, len(new)) == (100, 9)


@real_petta
def test_the_dq_levels_on_the_real_library(closure_lib):
    """With q = p (the fork read literally) the D+Q chain gives the decision's hand-checked levels; the two-run
    difference is -0.222599, which the affine storage turns into the legacy -0.2499 (divided by 0.98^4)."""
    links = _links("patient")
    a = "DasatinibPlusQuercetin"
    raw = {l["id"]: (_link_term(l), (l["strength"], l["confidence"]), (l["id"],)) for l in links}
    chain = ["CellularSenescence", "SASP", "DNAmPAI1", "CoronaryHeartDisease"]
    want = {"CellularSenescence": (0.30000, 0.98000), "SASP": (0.26900, 0.83340), "DNAmPAI1": (0.20292, 0.58671),
            "CoronaryHeartDisease": (0.13769, 0.36029)}
    levels = {}
    for level in (1.0, 0.0):
        _, final = _run_source(closure_lib, raw, links, 0.02, a, level, SETTINGS["K=256"])
        for (t, (s, c), ev) in final:
            if isinstance(t, tuple) and len(t) == 2 and t[0] in ("State", "End") and t[1] in chain:
                path = _path({l["id"]: l for l in links}, a, ev)
                if path is not None and path[-1] == t[1] and len(path) == chain.index(t[1]) + 2:
                    levels.setdefault(t[1], {})[level] = s
    for node in chain:
        assert abs(levels[node][1.0] - want[node][0]) < 5e-6 and abs(levels[node][0.0] - want[node][1]) < 5e-6, node
    diff = levels["CoronaryHeartDisease"][1.0] - levels["CoronaryHeartDisease"][0.0]
    assert abs(diff + 0.222599) < 5e-7


@real_petta
def test_the_real_library_equals_the_simulator_run_for_run(closure_lib):
    """fork_sim.py (a pure-Python port, validated against every upstream pin) as an independent oracle: the final
    (tasks, beliefs) of every patient run (two per source with an out-link) are the same lists, in the same order, to 1e-12."""
    sim_mod = _load_sim()
    links = _links("patient")
    kb_by_id = _kb(links, 0.02)
    runs = 0
    sources = sorted({l["from"] for l in links})
    for a in sources:
        for level in (1.0, 0.0):
            prem = (("State", a), (level, 1.0), (0,))
            beliefs = [kb_by_id[l["id"]] for l in _reachable(links, a)] + [prem]
            tasks, final = closure_lib.derive([prem], beliefs, 1, 100, 10, 100)
            sim = sim_mod.Sim(rules="restricted", maxsteps=100, task_queue_size=10, belief_queue_size=100, variant="petta_lib")
            sim_tasks, sim_final = sim.derive([prem], beliefs, 100, 10, 100)
            assert len(final) == len(sim_final) and len(tasks) == len(sim_tasks)
            for mine, theirs in zip(final, sim_final):
                assert mine[0] == theirs[0] and mine[2] == tuple(theirs[2])
                assert abs(mine[1][0] - theirs[1][0]) <= TOL and abs(mine[1][1] - theirs[1][1]) <= TOL
            runs += 1
    assert runs == 2 * len(sources) and len(sources) == 20


@real_petta
def test_the_completeness_status_is_sound_only_when_the_task_queue_is_as_large_as_the_belief_queue(closure_lib):
    """`complete` iff the task queue is empty and fewer than K-1 beliefs (LimitSize keeps K-1). With tq = bq = K every run
    that lost a pair is flagged; with a small tq and a large bq some are not (decision correction 1)."""
    links = _links("patient")
    by_id = {l["id"]: l for l in links}
    kb_by_id = _kb(links, 0.02)
    full = {}
    for a in sorted({l["from"] for l in links}):
        _, b1 = _run_source(closure_lib, kb_by_id, links, 0.02, a, 1.0, SETTINGS["K=256"])
        _, b0 = _run_source(closure_lib, kb_by_id, links, 0.02, a, 0.0, SETTINGS["K=256"])
        full[a] = set(_routes(b1, b0, by_id, a, 0.02))

    def lost_and_flag(a, tq, bq):
        t1, b1 = _run_source(closure_lib, kb_by_id, links, 0.02, a, 1.0, (1024, tq, bq))
        t0, b0 = _run_source(closure_lib, kb_by_id, links, 0.02, a, 0.0, (1024, tq, bq))
        got = set(_routes(b1, b0, by_id, a, 0.02))
        complete = not t1 and not t0 and len(b1) < bq - 1 and len(b0) < bq - 1
        return bool(full[a] - got), complete
    undetected_eq, undetected_lt, lost_eq = 0, 0, 0
    for a in full:
        if not full[a]:
            continue
        for k in (6, 8, 10, 12, 16):
            lost, complete = lost_and_flag(a, k, k)
            lost_eq += lost
            undetected_eq += lost and complete
        for tq in (2, 3, 4):
            lost, complete = lost_and_flag(a, tq, 200)
            undetected_lt += lost and complete
    assert lost_eq > 0 and undetected_eq == 0
    assert undetected_lt > 0


@real_petta
def test_serial_links_that_share_an_evidence_id_are_refused_by_the_library(closure_lib):
    """Shared ids are for parallel evidence of ONE link only: on a path A -> B -> C the stamp rule refuses the chain."""
    def reach(ids):
        tasks, final = closure_lib.derive(
            [(("State", "A"), (1.0, 1.0), (0,))],
            [(("Implication", ("State", "A"), ("State", "B")), (0.8, 0.9), (ids[0],)),
             (("Implication", ("State", "B"), ("State", "C")), (0.7, 0.9), (ids[1],)),
             (("State", "A"), (1.0, 1.0), (0,))], 1, 1024, 256, 256)
        return any(t == ("State", "C") for (t, tv, ev) in final)
    assert reach((7, 7)) is False
    assert reach((7, 8)) is True


@real_petta
def test_an_empty_stamp_with_confidence_below_one_floods_the_queue(closure_lib):
    """Why the intervention is the task (State A) (stv 1.0 1.0) with stamp (0): an empty stamp is disjoint from itself, so
    the premise is revised with itself again and again."""
    links = _links("patient")
    a = "DasatinibPlusQuercetin"
    kb_by_id = _kb(links, 0.02)
    beliefs = [kb_by_id[l["id"]] for l in _reachable(links, a)]
    bad = (("State", a), (1.0, 0.9), ())
    _, final = closure_lib.derive([bad], beliefs + [bad], 1, 12, 256, 256)
    assert sum(1 for b in final if b[0] == ("State", a)) > 50
    good = (("State", a), (1.0, 1.0), (0,))
    tasks, final = closure_lib.derive([good], beliefs + [good], 1, 1024, 256, 256)
    assert tasks == [] and sum(1 for b in final if b[0] == ("State", a)) == 1
