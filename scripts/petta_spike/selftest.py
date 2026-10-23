#!/usr/bin/env python
"""Offline self-test of the kit. Runs NOTHING on PeTTa and never imports janus_swi or petta from the checkout.

    python selftest.py            # all offline checks (a few seconds)
    python selftest.py --hyperon  # also drives the adapter, run_shapes.py and run_loads.py on hyperon

What it covers:
  * the normaliser and the comparator (kit_common);
  * the three Phase 2 fixes (kb_fixes), on strings and on the work tree's files;
  * the adapter's process plumbing - spawn, deadlines, SIGKILL, crash, stdout capture, load verification,
    load failure without an exception, cached mode - against a FAKE janus_swi / petta written into out/fake_petta/
    (the fake is this file's own code, so running it executes nothing from the PeTTa checkout);
  * with --hyperon, the same plumbing on the hyperon engine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path

import kit_common as K

FAKE_JANUS = '''
import os, re, sys, time
class PrologError(Exception):
    pass
_consulted = []
def consult(path, *a, **k):
    _consulted.append(path)
def _helper(name, arg):
    cache = os.environ.get("PETTA_LOAD_CACHE", "")
    if name == "load_metta_file_cached":
        text = open(arg, encoding="utf-8").read()
        snap = os.path.join(cache, "fake.snapshot")
        if not os.path.exists(snap):
            time.sleep(0.15)
            open(snap, "w").write("snapshot")
        else:
            time.sleep(0.01)
        return _helper("process_metta_string", text)
    if name == "load_metta_file":
        return _helper("process_metta_string", open(arg, encoding="utf-8").read())
    if arg.lstrip().startswith("!"):
        e = arg
        if "BOOM" in e: raise PrologError("fake: boom in expression")
        if "SLEEP" in e: time.sleep(120)
        if "CRASH" in e: os._exit(3)
        if "PRINT" in e:
            print("hello from fake prolog (python stdout)")
            os.write(1, b"raw fd write\\n")
            os.write(2, b"raw stderr write\\n")
            return []
        if "NUM" in e: return ["1.0e-5", "true", "3", "(stv 0.5 1.0e-7)", "(signed Pos (stv 2.0e+22 0.25))"]
        if "NONE" in e: return None
        return ["(ok " + e + ")"]
    if "SILENTFAIL" in arg: return None
    if "RAISE" in arg: raise PrologError("fake load error: cannot compile form (superpose ())")
    return []
def query_once(goal, inputs=None, **kw):
    if goal.startswith("run_metta_helper"):
        r = _helper(inputs["HelperName"], inputs["Argument"])
        return None if r is None else {"Results": r, "truth": True}
    if goal.startswith("set_prolog_flag"): return {"truth": True}
    if goal.startswith("between("):                      # a long C-level call that HOLDS the GIL (a builtin loop never yields it)
        sum(range(20_000_000))
        return {"truth": True}
    if goal.startswith("aggregate_all(count, thread_property"): return {"N": 1, "truth": True}
    if goal.startswith("current_prolog_flag(version"): return {"V": 100002, "truth": True}
    if goal.startswith("findall(S, (member(N-A, ["):
        heads = re.findall(r"'((?:[^'\\\\]|\\\\.)*)'-(\\d+)", goal)
        rows = []
        for n, a in heads:
            c = 0 if n.startswith("missing") else (1 if n.startswith("short") else 2)
            rows.append(f"{n} {a} {c}")
        return {"Rows": rows, "truth": True}
    raise PrologError("fake: unknown goal " + goal[:80])
'''

FAKE_PETTA = '''
import os, importlib
CONSULTED = False
janus = None
class PeTTa:
    def __init__(self, verbose=False, petta_path=None):
        global janus, CONSULTED
        janus = importlib.import_module("janus_swi")
        janus.query_once("set_prolog_flag(stack_limit, 8000000000)")
        janus.consult(os.path.join(petta_path or "", "src", "main.pl"))
        CONSULTED = True
    def _run_helper(self, name, arg):
        r = janus.query_once("run_metta_helper(Verbose, HelperName, Argument, Results)",
                             {"Verbose": "false", "HelperName": name, "Argument": arg})
        return [] if r is None else r.get("Results", [])
    def load_metta_file(self, p): return self._run_helper("load_metta_file", p)
    def load_metta_file_cached(self, p): return self._run_helper("load_metta_file_cached", p)
    def process_metta_string(self, c): return self._run_helper("process_metta_string", c)
'''


def make_fake_petta() -> Path:
    home = K.out_dir() / "fake_petta"
    (home / "src").mkdir(parents=True, exist_ok=True)
    (home / "python" / "petta").mkdir(parents=True, exist_ok=True)
    (home / "FAKE_PETTA").write_text("the self-test's stand-in for a PeTTa tree; petta_adapter.petta_allowed() lets it through\n")
    (home / "src" / "main.pl").write_text("% fake\n")
    (home / "python" / "helper.pl").write_text("% fake\n")
    (home / "python" / "janus_swi.py").write_text(FAKE_JANUS)
    (home / "python" / "petta" / "__init__.py").write_text(FAKE_PETTA)
    return home


PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(("  ok    " if cond else "  FAIL  ") + name + (f"   {detail}" if (detail and not cond) else ""), flush=True)


def t_common() -> None:
    print("kit_common")
    check("bool spelling", K.normalise_answer("(a true (b false))") == "(a True (b False))")
    check("rust exponent", K.normalise_answer("(stv 1.0e-5 2.0e+22 1.5e-7)") == "(stv 1e-5 2e22 1.5e-7)", K.normalise_answer("(stv 1.0e-5 2.0e+22 1.5e-7)"))
    check("ints stay ints", K.normalise_answer("(x 3 3.0 -2)") == "(x 3 3.0 -2)")
    check("strings untouched", K.normalise_answer('(s "a (b); true 1.0e-5")') == '(s "a (b); true 1.0e-5")')
    check("inf spellings", K.normalise_answer("(1.0Inf -1.0Inf)") == "(inf -inf)")
    check("empty tuple", K.normalise_answer("()") == "()")
    check("identical", K.compare_groups(["(a 1)"], ["(a 1)"])["class"] == "identical")
    r = K.compare_groups(["(stv 0.5 3.0)", "True"], ["(stv 0.5 3)", "true"])
    check("number format only", r["class"] == "number_format_only", str(r))
    r = K.compare_groups(["(stv 0.357 0.29089125000000005)"], ["(stv 0.357 0.29089125)"])
    check("within tolerance", r["class"] == "within_tolerance", str(r))
    r = K.compare_groups(["(a 1)", "(b 2)"], ["(b 2)", "(a 1)"])
    check("ordering only (list)", r["class"] == "ordering_only", str(r))
    r = K.compare_groups(["((a 1) (b 2))"], ["((b 2) (a 1))"])
    check("ordering only (inside a tuple)", r["class"] == "ordering_only", str(r))
    check("different", K.compare_groups(["(a 1)"], ["(a 2)"])["class"] == "different")
    check("different count", K.compare_groups(["x"], [])["class"] == "different")
    check("expected smoke", K.matches_expected("(signed Pos (stv 0.357 0.29089125))", ["(signed Pos (stv 0.357 0.29089125000000005))"])["ok"])
    check("expected empty", K.matches_expected("(empty)", [])["ok"] and not K.matches_expected("(empty)", ["x"])["ok"])
    kb = '(= (f $x) 1)\n(= (f $x $y) 2)\n(= (f $z) 3) ; c\n(Effect A B Pos (stv 0.1 (evidence-confidence X)))\n(= (g) 5)\n!(f 1)\n(= (h (p $x) q) "a ; (b")'
    eq = K.kb_equations(kb)
    check("kb_equations", eq == {("f", 2): 2, ("f", 3): 1, ("g", 1): 1, ("h", 3): 1}, str(eq))


def t_fixes() -> None:
    print("kb_fixes")
    import kb_fixes as F
    t, n = F.fix_superpose_empty('(= (a) (superpose ()))\n; (superpose ()) in a comment\n(= (b) "(superpose ())")\n(= (c) (superpose  ( ) ))')
    check("superpose: code only", n == 2 and "; (superpose ())" in t and '"(superpose ())"' in t and t.count("(empty)") == 2, t)
    tiers = {"T1": "0.65"}
    t, n, un = F.fix_stored_calls("(Effect A B Pos (stv 0.8 (evidence-confidence T1)))\n(= (r) (stv 0.5 (evidence-confidence T1)))\n(Effect C D Pos (stv 0.5 (evidence-confidence ZZ)))", tiers)
    check("stored calls: facts only", n == 1 and "(stv 0.8 0.65)" in t and "(= (r) (stv 0.5 (evidence-confidence T1)))" in t and un == ["ZZ"], t)
    t, n = F.fix_nested_conjunction("(match $s (, (A $x)\n   (, (B $x)\n      (, (C $x) (D $x)))) $x)")
    check("nested conj, two deep", n == 2 and "(, (A $x)" in t and "(, (B" not in t and "(, (C" not in t, t)
    t, n = F.fix_nested_conjunction('(match $s (, (A $x) (B "(, not code")) $x)')
    check("conj: strings ignored", n == 0)
    repo = K.repo_root()
    if (repo / "epistemic_calibration.metta").exists():
        files = sorted(repo.glob("*.metta"))
        named = [(f.name, f.read_text(encoding="utf-8")) for f in files]
        _, rep = F.fix_kb_texts(named, F.ALL_FIXES)
        tot = {k: sum(v.get(k, 0) for v in rep.values()) for k in ("superpose_empty", "stored_calls", "nested_conj")}
        check("work tree: 20 superpose / 25 stored calls / 1 nested conj (LAUNCH.md Phase 2)", tot == {"superpose_empty": 20, "stored_calls": 25, "nested_conj": 1}, str(tot))


def _petta_style(answer: str) -> str:
    """What SWI-Prolog's write/1 would print for a hyperon answer: lower-case booleans, `1.0e-5` for `1e-5`."""
    import re
    def one(tok: str) -> str:
        if tok in ("True", "False"):
            return tok.lower()
        if re.fullmatch(r"[-+]?\d+e[-+]?\d+", tok):
            m, e = tok.split("e")
            return f"{m}.0e{e}"
        return tok
    return K._TOKEN_RE.sub(lambda m: one(m.group(0)), answer)


def t_parsers() -> None:
    """The normaliser against the repo's OWN parsers (imported read-only): same results from hyperon's strings and from
    PeTTa-style strings after normalisation; and the raw PeTTa style really does break one of them."""
    print("normaliser vs the repo's parsers (pln_runner, linage2_router)")
    base = K.baseline_path()
    if not base.exists():
        check("baseline exists (run record_baseline.py first)", False)
        return
    import stacks as S
    S.repo_import()
    from core import pln_runner as R
    from core import linage2_router as L
    doc = json.loads(base.read_text())
    answers = [a for sh in doc["shapes"] for p in sh["parts"] for g in p["hyperon"].get("groups", []) for a in g]
    check("baseline holds answers", len(answers) > 150, str(len(answers)))
    n_changed = sum(1 for a in answers if _petta_style(a) != a)
    check("PeTTa-style rendering differs from hyperon's on some answers", n_changed > 10, str(n_changed))
    bad_norm = [a for a in answers if K.normalise_answer(_petta_style(a)) != a]
    check("normalise(petta_style(x)) == x for every recorded answer", not bad_norm, str(bad_norm[:2]))
    stv_bad = [a for a in answers if R._stv_from_atom(K.normalise_answer(_petta_style(a))) != R._stv_from_atom(a)]
    check("pln_runner._stv_from_atom agrees on every answer", not stv_bad, str(stv_bad[:2]))
    scored_bad = [a for a in answers if R.parse_scored(K.normalise_answer(_petta_style(a))) != R.parse_scored(a)]
    check("pln_runner.parse_scored agrees", not scored_bad)
    cell_bad = [a for a in answers if R.parse_cellage_effects(K.normalise_answer(_petta_style(a))) != R.parse_cellage_effects(a)]
    check("pln_runner.parse_cellage_effects agrees", not cell_bad)
    lin = [a for a in answers if a.startswith("(LinAgeDecomposition")]
    check("a LinAge2 decomposition is among the answers", bool(lin))
    if lin:
        want = L.decomposition_to_dict(L.parse_sexpr(lin[0]))
        got = L.decomposition_to_dict(L.parse_sexpr(K.normalise_answer(_petta_style(lin[0]))))
        raw = L.decomposition_to_dict(L.parse_sexpr(_petta_style(lin[0])))
        check("linage2_router.decomposition_to_dict agrees after normalisation", want == got)
        witnessed = [c["witnessed"] for c in want["measured"] + want["imputed"] if c["witnessed"] is not None]
        check("...and the raw PeTTa style would silently flip `witnessed` (the reason the normaliser exists)",
              bool(witnessed) and any(w is True for w in witnessed) and raw != want, f"{witnessed[:4]}")


def t_gate() -> None:
    print("the default-deny gate")
    import petta_adapter as A
    os.environ.pop("KIT_ALLOW_PETTA", None)
    ok, why = A.petta_allowed(K.DEFAULT_PETTA_HOME)
    check("the real PeTTa tree is refused without KIT_ALLOW_PETTA=1", not ok and "KIT_ALLOW_PETTA" in why, why)
    saved = os.environ.get("PETTA_HOME")
    os.environ["PETTA_HOME"] = K.DEFAULT_PETTA_HOME
    try:
        A.run_program_report("(= (f) 1)", ["!(f)"], deadline_s=5)
        check("run_program_report refuses to start PeTTa", False)
    except A.PettaNotAllowed:
        check("run_program_report refuses to start PeTTa", True)
    finally:
        if saved is None:
            os.environ.pop("PETTA_HOME", None)
        else:
            os.environ["PETTA_HOME"] = saved
    os.environ["KIT_ALLOW_PETTA"] = "1"
    check("KIT_ALLOW_PETTA=1 is the go", A.petta_allowed(K.DEFAULT_PETTA_HOME)[0])
    os.environ.pop("KIT_ALLOW_PETTA", None)


def t_adapter_fake() -> None:
    print("petta_adapter against a fake janus_swi / petta")
    import petta_adapter as A
    home = make_fake_petta()
    os.environ["PETTA_HOME"] = str(home)
    os.environ["PETTA_LOAD_CACHE"] = str(K.out_dir() / "fake_load_cache")
    Path(os.environ["PETTA_LOAD_CACHE"]).mkdir(parents=True, exist_ok=True)
    for f in Path(os.environ["PETTA_LOAD_CACHE"]).glob("*"):
        f.unlink()
    kb = "(= (f $x) $x)\n(= (f $x $y) $y)\n(= (short-one $x) $x)\n(= (short-one $x) 2)\n(= (missing-one $x) $x)\n"

    r = A.run_program_report(kb, ["!(f 1)", "!(NUM)"], deadline_s=60)
    check("ok run", r.status == "ok" and r.n_expr_errors == 0, f"{r.status} {r.error}")
    check("answers carry raw next to normalised", r.answers[1][0].raw == "1.0e-5" and r.answers[1][0].norm == "1e-5"
          and r.answers[1][1].norm == "True" and r.answers[1][2].norm == "3" and r.answers[1][3].norm == "(stv 0.5 1e-7)"
          and r.answers[1][4].norm == "(signed Pos (stv 2e22 0.25))", str([(a.raw, a.norm) for a in r.answers[1]]))
    check("one answer list per expression", len(r.answers) == 2 and r.answers[0][0].raw == "(ok !(f 1))")
    check("timings recorded", all(k in r.timing for k in ("import_s", "consult_s", "load_s", "wall_s", "child_s")), str(r.timing))
    check("rss recorded", "load" in r.rss_mb and r.peak_rss_mb is not None)
    check("petta from the PETTA_HOME tree", (r.env.get("petta_file") or "").startswith(str(home)), str(r.env))
    v = r.load.get("verify", {})
    check("load verification counts heads", v.get("checked") == 4 and v.get("n_missing") == 1 and v.get("n_short") == 1
          and "missing-one/2" in v.get("missing", []) and v.get("short") == ["short-one/2: 1 of 2 clauses"], str(v))
    check("stages reported", [s["stage"] for s in r.stages][:3] == ["imported", "consulted", "loaded"], str(r.stages))
    out = A.run_program(kb, ["!(f 1)"], deadline_s=60)
    check("run_program shape", out == [["(ok !(f 1))"]], str(out))

    r = A.run_program_report(kb, ["!(BOOM)", "!(f 2)"], deadline_s=60)
    check("expression error recorded, run continues", r.status == "ok" and r.expr_errors[0] and "boom" in r.expr_errors[0]["message"]
          and r.answers[1] and not r.expr_errors[1], str(r.expr_errors))
    try:
        A.run_program(kb, ["!(BOOM)"], deadline_s=60)
        check("run_program raises on an expression error", False)
    except A.PettaRunError as e:
        check("run_program raises on an expression error", e.report.n_expr_errors == 1)

    r = A.run_program_report("(= (a) 1)\n;; SILENTFAIL", ["!(f 1)"], deadline_s=60)
    check("load goal that fails without raising is a failed load", r.status == "load_failed" and r.load.get("failed_silently") is True
          and r.answers == [[]], f"{r.status} {r.load}")
    r = A.run_program_report("(= (a) 1)\n;; SILENTFAIL", ["!(f 1)"], deadline_s=60, use_wrapper=True)
    check("the PeTTa wrapper cannot tell (documented)", r.status == "ok" and r.load.get("failed_silently") is False)

    r = A.run_program_report("(= (a) (superpose ()))\n;; RAISE", ["!(f 1)"], deadline_s=60)
    check("load exception is a failed load, verified anyway", r.status == "load_error" and "superpose" in r.load["exception"]["message"]
          and r.load.get("verify", {}).get("checked") == 1 and r.answers == [[]], f"{r.status} {r.load}")
    check("first_error names it", "fake load error" in (r.first_error() or ""))

    t = time.time()
    r = A.run_program_report(kb, ["!(f 1)", "!(SLEEP)", "!(f 3)"], deadline_s=60, expr_deadline_s=1.5)
    check("expression deadline SIGKILLs the child", r.status == "timeout" and r.timeout_scope == "expr:1" and r.signal == "SIGKILL"
          and time.time() - t < 20, f"{r.status} {r.timeout_scope} {r.signal} {time.time() - t:.1f}s")
    r = A.run_program_report(kb, ["!(SLEEP)"], deadline_s=2.0)
    check("total deadline", r.status == "timeout" and r.timeout_scope in ("total", "expr:0"), f"{r.status} {r.timeout_scope}")
    r = A.run_program_report(kb, ["!(CRASH)"], deadline_s=60)
    check("crash (os._exit(3), like halt(1)) is reported", r.status == "crash" and r.exitcode == 3, f"{r.status} {r.exitcode}")
    r = A.run_program_report(kb, ["!(PRINT)"], deadline_s=60)
    check("stdout / stderr captured at fd level", "hello from fake prolog" in r.stdout and "raw fd write" in r.stdout and "raw stderr write" in r.stderr,
          f"{r.stdout!r} {r.stderr!r}")
    r = A.run_program_report(kb, ["!(NONE)"], deadline_s=60)
    check("an expression whose goal fails is recorded", r.expr_errors[0] and r.expr_errors[0]["type"] == "GoalFailed")

    r1 = A.run_program_report(kb, ["!(f 1)"], deadline_s=60, mode="cached")
    r2 = A.run_program_report(kb, ["!(f 1)"], deadline_s=60, mode="cached")
    check("cached mode: miss then hit", r1.status == "ok" and r1.load["cache_hit_guess"] is False and r2.load["cache_hit_guess"] is True
          and r2.load["seconds"] < r1.load["seconds"], f"{r1.load.get('cache_hit_guess')} {r2.load.get('cache_hit_guess')} {r1.load.get('seconds')} {r2.load.get('seconds')}")
    check("cached mode: KB file named by content hash", Path(r1.load["kb_file"]).name == f"kb_{K.sha256_text(kb)[:16]}.metta" and Path(r1.load["kb_file"]).read_text() == kb)
    r = A.run_program_report(kb, ["!(f 1)"], deadline_s=60, mode="file")
    check("file mode", r.status == "ok" and r.load["mode"] == "file")
    r = A.run_program_report("", ["!(f 1)"], deadline_s=60)
    check("empty KB", r.status == "ok" and r.load.get("skipped"))
    r = A.run_program_report(kb, ["(f 1)"], deadline_s=60)
    check("'!' added when missing", r.answers[0][0].raw == "(ok !(f 1))")
    empty = K.out_dir() / "fake_petta_without_wrapper"
    (empty / "python").mkdir(parents=True, exist_ok=True)
    (empty / "FAKE_PETTA").write_text("marker\n")
    (empty / "python" / "janus_swi.py").write_text("raise ImportError('fake janus_swi: not installed')\n")   # never import the real one in a self-test
    os.environ["PETTA_HOME"] = str(empty)
    r = A.run_program_report(kb, ["!(f 1)"], deadline_s=60)
    check("a failing janus import is an import_error", r.status == "import_error" and "fake janus_swi" in (r.first_error() or ""), f"{r.status} {r.error}")
    os.environ["PETTA_HOME"] = str(home)


def t_measure_fake() -> None:
    """measure.py's PeTTa-only sections (consult, limits, fork, gil), control flow only, against the fake tree."""
    print("measure.py sections against the fake janus_swi / petta")
    import subprocess
    home = make_fake_petta()
    out = K.out_dir() / "selftest_measure"
    env = {k: v for k, v in os.environ.items() if k != "KIT_ALLOW_PETTA"}
    env.update(PETTA_HOME=str(home), KIT_OUT=str(out), PETTA_LOAD_CACHE=str(out / "cache"))
    p = subprocess.run([sys.executable, "-B", str(K.KIT_DIR / "measure.py"), "--sections", "consult,limits,fork,gil", "--reps", "1", "--forks", "2",
                        "--gil-loop", "1000", "--gil-count", "10", "--fork-stack", "human_evidence", "--deadline", "120", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    check("measure.py runs its PeTTa-only sections on the fake tree", p.returncode == 0, (p.stdout + p.stderr)[-1200:])
    mj = out / "measure.json"
    if mj.exists():
        d = json.loads(mj.read_text())
        check("consult section ok", d["consult"]["runs"][0].get("ok") is True, str(d["consult"]["runs"][0])[:300])
        check("fork section: forks answered", d["fork"].get("ok") and d["fork"]["after_load_summary"]["answered"] == 2 and d["fork"]["after_consult_summary"]["answered"] == 2, str(d["fork"])[:400])
        g = d["gil"]
        check("gil section: the busy fake holds the GIL", g.get("ok") and g["prolog_loop"]["tick_ratio"] is not None and g["prolog_loop"]["tick_ratio"] < 0.5, str(g)[:400])
        check("limits section ran all probes", len(d["limits"]) == 9 and all("status" in v for v in d["limits"].values()), str(list(d["limits"])))
        md = (out / "measure.md").read_text()
        check("report renders every section", all(h in md for h in ("## consult", "## limits and hazards", "## fork() after a child-only import", "## GIL")))
    else:
        check("measure.json written", False)


def t_shapes_fake() -> None:
    """run_shapes.py on a three-shape mini baseline against the fake tree: batch, crash, isolation, classes."""
    print("run_shapes.py: batching and isolation against the fake tree")
    import subprocess
    home = make_fake_petta()
    out = K.out_dir() / "selftest_shapes"
    out.mkdir(parents=True, exist_ok=True)
    kb = "(= (f) 1)"

    def shape(i, expr, hyp):
        return {"id": f"x:{i}", "source": "smoke", "nl": "", "program": expr, "route": "x", "parts": [{
            "stack": "s", "kb_files": [], "extra_atoms": kb, "expressions": [expr], "kb_sha256": K.sha256_text(kb),
            "hyperon": {"status": "ok", "groups": [hyp], "timing": {"per_expr_ms": [1.0], "init_ms": 1, "load_ms": 1}}}]}
    mini = {"meta": {"git_head": "test"}, "stacks": {}, "shapes": [
        shape(1, "!(A1)", ["(ok !(A1))"]), shape(2, "!(CRASH)", ["whatever"]), shape(3, "!(A3)", ["(ok !(A3))"]),
        shape(4, "!(NUM)", ["1e-5", "True", "3", "(stv 0.5 1e-7)", "(signed Pos (stv 2e22 0.25))"]),
        shape(5, "!(A5)", ["(different)"])]}
    (out / "mini.json").write_text(json.dumps(mini))
    env = {k: v for k, v in os.environ.items() if k != "KIT_ALLOW_PETTA"}
    env.update(PETTA_HOME=str(home), KIT_OUT=str(out), PETTA_LOAD_CACHE=str(out / "cache"))
    p = subprocess.run([sys.executable, "-B", str(K.KIT_DIR / "run_shapes.py"), "--baseline", str(out / "mini.json"), "--fixes", "none", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    rep = out / "shapes_report.json"
    check("run_shapes ran (exit 1 because a shape errored)", p.returncode == 1 and rep.exists(), (p.stdout + p.stderr)[-800:])
    if rep.exists():
        d = json.loads(rep.read_text())
        cls = {r["id"]: r["class"] for r in d["results"]}
        check("classes per shape", cls == {"x:1": "identical", "x:2": "error", "x:3": "identical", "x:4": "number_format_only", "x:5": "different"}, str(cls))
        check("the crashing batch was isolated, not lost", len(d["meta"]["batches_isolated"]) == 1 and d["meta"]["batches_isolated"][0]["status"] == "crash", str(d["meta"]["batches_isolated"]))
        x4 = next(r for r in d["results"] if r["id"] == "x:4")["parts"][0]["exprs"][0]
        check("raw and normalised kept side by side, normalised equals hyperon", x4["petta_raw"][0] == "1.0e-5" and x4["petta_norm"][0] == "1e-5" and x4["norm_equal"], str(x4)[:300])
        x2 = next(r for r in d["results"] if r["id"] == "x:2")
        check("error text recorded", "exited with code 3" in (x2["parts"][0]["first_error"] or ""), str(x2["parts"][0]["first_error"]))
        check("breakage table written", (out / "breakage_table.md").read_text().count("| `x:") >= 3)


def t_loads_fake() -> None:
    """run_loads.one_run against the fake tree: a load that raises, a load that fails silently, and the smoke verdict path."""
    print("run_loads.py: failure paths against the fake tree")
    import argparse
    import run_loads
    import stacks as S
    home = make_fake_petta()
    os.environ["PETTA_HOME"] = str(home)
    os.environ["PETTA_LOAD_CACHE"] = str(K.out_dir() / "fake_load_cache")
    ns = argparse.Namespace(engine="petta", deadline=60.0, expr_deadline=30.0, stack_limit_mb=256, strict_verify=False, smoke_everywhere=True)
    smoke = [{"expr": "!(infer &self A B)", "expected": "(signed Pos (stv 0.5 0.5))"}, {"expr": "!(infer &self C D)", "expected": "(empty)"}]
    spec = S.StackSpec("x", "fake", [], extra_atoms="(= (f $x) $x)\n;; RAISE", canary=["!(f 1)"])
    r = run_loads.one_run(spec, "asis", "string", set(), smoke, None, ns)
    check("load exception -> status load_error, text kept, no smoke verdict", r["status"] == "load_error" and "fake load error" in (r["first_error"] or "")
          and r["smoke_ok"] is None, f"{r['status']} {r['first_error']}")
    spec = S.StackSpec("x", "fake", [], extra_atoms="(= (f $x) $x)\n;; SILENTFAIL", canary=["!(f 1)"])
    r = run_loads.one_run(spec, "asis", "string", set(), smoke, None, ns)
    check("goal failure -> status load_failed", r["status"] == "load_failed" and r["failed_silently"] is True, f"{r['status']}")
    spec = S.StackSpec("x", "fake", [], extra_atoms="(= (f $x) $x)", canary=["!(f 1)"])
    r = run_loads.one_run(spec, "fixed", "string", {"superpose_empty"}, smoke, None, ns)
    check("good load: smoke verdicts recorded and judged against the expected text", r["status"] == "ok" and r["smoke_ok"] is False
          and r["smoke"][0]["ok"] is False and r["smoke"][1]["ok"] is False and r["verify"]["checked"] == 1, str(r["smoke"]))
    g = run_loads.gate([r])
    check("gate block", g["stacks_loading_after_fixes"] == ["x"] and g["smoke_answers_after_phase2_fixes_only"] is None)


def t_adapter_hyperon() -> None:
    print("petta_adapter, hyperon engine")
    import petta_adapter as A
    kb = "(= (f $x) (+ $x 1))\n(Q a)\n(Q b)"
    r = A.run_program_report(kb, ["!(f 1)", "!(collapse (match &self (Q $x) $x))"], engine="hyperon", deadline_s=60)
    check("hyperon engine ok", r.status == "ok" and r.normalised() == [["2"], ["(a b)"]], f"{r.status} {r.normalised()} {r.error}")
    r = A.run_program_report("(= (spin) (spin))", ["!(spin)"], engine="hyperon", deadline_s=4)
    check("hyperon runaway is killed at the deadline", r.status == "timeout" and r.signal == "SIGKILL", f"{r.status} {r.signal}")


def t_pipeline_hyperon(fast: bool) -> None:
    """record -> run_shapes on the hyperon engine: every answer must come back identical."""
    print("run_shapes / run_loads on the hyperon engine")
    import subprocess
    py = sys.executable
    base = K.baseline_path()
    if not base.exists():
        check("baseline exists (run record_baseline.py first)", False)
        return
    out = K.out_dir() / "selftest_hyperon"
    env = {k: v for k, v in os.environ.items() if k != "KIT_ALLOW_PETTA"}
    env["KIT_OUT"] = str(out)
    only = r"^(smoke|fs:(0[0-1]|1[6-9]|2[0-2])|fam:F0[1-4]|probe:order:collapse-match|probe:site:patient-z|probe:print:int)"
    p = subprocess.run([py, "-B", str(K.KIT_DIR / "run_shapes.py"), "--engine", "hyperon", "--only", only, "--mode", "batch", "--deadline", "120"],
                       capture_output=True, text=True, env=env, timeout=900)
    ok = p.returncode == 0
    check("run_shapes --engine hyperon exits 0", ok, (p.stdout + p.stderr)[-800:])
    rep = out / "shapes_report.json"
    if rep.exists():
        d = json.loads(rep.read_text())
        bad = [r["id"] for r in d["results"] if r["class"] not in ("identical",)]
        check("every compared shape is identical on the same engine", not bad, str(bad[:10]))
        check("classes counted", d["counts"].get("identical", 0) >= 10, str(d["counts"]))
    else:
        check("shapes_report.json written", False)
    p = subprocess.run([py, "-B", str(K.KIT_DIR / "run_loads.py"), "--engine", "hyperon", "--stacks", "smoke10,human_evidence", "--modes", "string",
                        "--deadline", "120"], capture_output=True, text=True, env=env, timeout=900)
    check("run_loads --engine hyperon exits 0", p.returncode == 0, (p.stdout + p.stderr)[-800:])
    ld = out / "loads.json"
    if ld.exists():
        d = json.loads(ld.read_text())
        smoke = [r for r in d["runs"] if r["stack"] == "smoke10"]
        check("smoke answers match the section-5 comments", smoke and all(s["smoke_ok"] for s in smoke), str(smoke[:1])[:400])
    else:
        check("loads.json written", False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyperon", action="store_true", help="also drive the adapter and the run_* scripts on hyperon")
    ap.add_argument("--only", help="comma list of: common,fixes,gate,fake,loads,shapes,measure,parsers,hyperon,pipeline")
    a = ap.parse_args(argv)
    os.environ.pop("KIT_ALLOW_PETTA", None)           # the self-test must never be able to start the real PeTTa
    want = set((a.only or "common,fixes,gate,fake,loads,shapes,measure,parsers").split(","))
    if a.hyperon:
        want |= {"hyperon", "pipeline"}
    for name, fn in (("common", t_common), ("fixes", t_fixes), ("gate", t_gate), ("fake", t_adapter_fake), ("loads", t_loads_fake), ("shapes", t_shapes_fake), ("measure", t_measure_fake), ("parsers", t_parsers), ("hyperon", t_adapter_hyperon),
                     ("pipeline", lambda: t_pipeline_hyperon(False))):
        if name in want:
            try:
                fn()
            except Exception:
                FAILED.append(name + " (exception)")
                traceback.print_exc()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print("  FAILED:", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
