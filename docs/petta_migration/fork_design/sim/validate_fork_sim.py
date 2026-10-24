"""
validate_fork_sim.py -- runs fork_sim.py against every pinned expectation in the upstream
examples/ and ruletests/ (read as DATA, never executed or imported), and against PeTTa's own
pln_*.metta / plntest*.metta (which use PeTTa's BUNDLED lib_pln, a different snapshot).

Usage:  python3 -I validate_fork_sim.py [--json out.json] [--eq-order file|reverse] [--unique fork|keep]
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fork_sim as F  # noqa: E402

FORK = "/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa/repos/PLN"
PETTA = "/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa"

args = sys.argv[1:]
EQ_ORDER = "file"
UNIQUE = None
JSON_OUT = None
for i, a in enumerate(args):
    if a == "--eq-order":
        EQ_ORDER = args[i + 1]
    if a == "--unique":
        UNIQUE = args[i + 1]
    if a == "--json":
        JSON_OUT = args[i + 1]

RESULTS = []


def fmt(r):
    if r is None:
        return "()"
    (s, c), ev = r
    return "((stv %r %r) %r)" % (s, c, tuple(ev))


def compare(case, source, expected, got, extra=""):
    """expected/got: ((s,c), stamp) or None."""
    if expected is None or got is None:
        ok = expected is None and got is None
        maxabs = None
        exact15 = ok
    else:
        (es, ec), eev = expected
        (gs, gc), gev = got
        maxabs = max(abs(es - gs), abs(ec - gc))
        exact15 = (F.round15(gs) == es and F.round15(gc) == ec) and tuple(eev) == tuple(gev)
        ok = exact15
        if not exact15 and maxabs < 1e-12 and tuple(eev) == tuple(gev):
            extra += " (values equal within 1e-12 but not equal after rounding to 15 digits)"
    RESULTS.append(
        dict(case=case, source=source, expected=fmt(expected), simulated=fmt(got), match=bool(ok), maxabs=maxabs, note=extra)
    )
    return ok


def read(path):
    with open(path) as f:
        return f.read()


# ------------------------------------------------------------------------------------
# generic extraction of  (= (name) body) / !(PLN.Init X) / !(Test (PLN.Query ...) expected)
# ------------------------------------------------------------------------------------


def is_sentence_form(f):
    return isinstance(f, tuple) and len(f) == 3 and f[0] in ("sentence", "Sentence")


def evaluate(form, env):
    """Evaluate the tiny subset of MeTTa needed to reach a knowledge base."""
    if isinstance(form, tuple):
        if len(form) == 1 and isinstance(form[0], str) and form[0] in env:
            return evaluate(env[form[0]], env)
        if form and form[0] == "TupleConcat":
            out = []
            for a in form[1:]:
                out.extend(evaluate(a, env))
            return out
        if all(is_sentence_form(f) for f in form):
            return F.kb_from_metta(form)
    return form


def run_file(path, label, eq_order=EQ_ORDER, unique=UNIQUE, variant="fork"):
    forms = F.parse_sexprs(read(path))
    env = {}
    sim = None
    n_tests = 0
    for form in forms:
        if isinstance(form, tuple) and len(form) == 3 and form[0] == "=" and isinstance(form[1], tuple) and len(form[1]) == 1:
            env[form[1][0]] = form[2]
            continue
        if isinstance(form, tuple) and len(form) == 2 and form[0] == "!":
            call = form[1]
            if call[0] == "PLN.Init":
                static = evaluate(call[1], env)
                sim = F.Sim(rules="full", static=static, eq_order=eq_order, unique_mode=unique, variant=variant)
            elif call[0] == "Test" and isinstance(call[1], tuple) and call[1][0] == "PLN.Query":
                q = call[1]
                kb = evaluate(q[1], env)
                term = q[2]
                nums = list(q[3:])
                kw = {}
                if len(nums) >= 1:
                    kw["maxsteps"] = nums[0]
                if len(nums) >= 3:
                    kw["task_queue_size"], kw["belief_queue_size"] = nums[1], nums[2]
                exp_tv, exp_ev = call[2]
                expected = ((exp_tv[1], exp_tv[2]), tuple(exp_ev))
                got = sim.query(kb, term, **kw)
                n_tests += 1
                compare("%s Test #%d" % (label, n_tests), "%s" % path.replace(FORK + "/", ""), expected, got)
    return sim, env


def main():
    # ----- examples -------------------------------------------------------------------
    sims = {}
    for name in ("Smokes", "FlyingRaven", "Toothbrush"):
        sims[name] = run_file(os.path.join(FORK, "examples", name + ".metta"), name)
    # ----- ruletests ------------------------------------------------------------------
    for name in (
        "RuleTester",
        "equivalenceToImplication",
        "evaluationImplicationRuleA",
        "evaluationWithNegationAndInheritanceInversion",
        "inversion",
        "memberDeductionA",
        "transitiveSimilarity",
    ):
        run_file(os.path.join(FORK, "ruletests", name + ".metta"), name)

    # ----- Robot.metta: the Test runs a small MeTTa program around PLN.Query ----------------
    forms = F.parse_sexprs(read(os.path.join(FORK, "examples", "Robot.metta")))
    env = {}
    static = None
    for form in forms:
        if isinstance(form, tuple) and len(form) == 3 and form[0] == "=" and form[1] == ("kbstatic",):
            static = F.kb_from_metta(form[2])
        if isinstance(form, tuple) and len(form) == 3 and form[0] == "=" and form[1] == ("kbdynamic",):
            kbdyn = F.kb_from_metta(form[2])
    sim = F.Sim(rules="full", static=static, eq_order=EQ_ORDER, unique_mode=UNIQUE)

    def categorize(label, category, **kw):
        kb = kbdyn + [F.S(("Inheritance", ("Concept", label), ("Concept", "see")), 0.99, 0.99, (99,))]
        return sim.query(kb, ("Inheritance", ("Concept", label), ("Concept", category)), **kw)

    detections = [("frisbee", "someCoords1"), ("{SELF}", "someCoords2"), ("person", "someCoords3"), ("orange", "someCoords4")]
    kept = []
    answers = {}
    for lab, coord in detections:
        r = categorize(lab, "bring")
        answers[lab] = r
        if r is not None and lab not in [k[0] for k in kept]:
            kept.append((lab, coord))
    got_objs = [("detection", l, c) for l, c in kept]
    exp_objs = [("detection", "frisbee", "someCoords1"), ("detection", "orange", "someCoords4")]
    RESULTS.append(
        dict(
            case="Robot Test #1 (objects to bring)",
            source="examples/Robot.metta:56-61",
            expected=repr(exp_objs),
            simulated=repr(got_objs) + "  answers=" + repr({k: fmt(v) for k, v in answers.items()}),
            match=got_objs == exp_objs,
            maxabs=None,
            note="",
        )
    )
    return sims


# ------------------------------------------------------------------------------------
# PeTTa's own examples.  They import PeTTa's BUNDLED lib/lib_pln.metta (a different snapshot: same
# formulas and same Deriver, but defaults 100/10/100, order-keeping list_to_set for Unique, a
# `(= (STV $stv) (empty))` fallback and no Translator), so they run with variant='petta_lib',
# rules = everything except 'Translated'.  PeTTa prints shortest-round-trip floats, so the
# comparison is EXACT (no 15-digit rounding).
# ------------------------------------------------------------------------------------


def compare_exact(case, source, expected, got, note=""):
    if expected is None or got is None:
        ok = expected is None and got is None
        maxabs = None
    else:
        (es, ec), eev = expected
        (gs, gc), gev = got
        maxabs = max(abs(es - gs), abs(ec - gc))
        ok = es == gs and ec == gc and tuple(eev) == tuple(gev)
    RESULTS.append(dict(case=case, source=source, expected=fmt(expected), simulated=fmt(got), match=bool(ok), maxabs=maxabs, note=note))
    return ok


def petta_examples():
    lib_rules = [r for r in F.PRESETS["full"] if r != "Translated"]
    # --- pln_roman.metta and pln_tuffy.metta : full PLN.Query on the bundled lib
    for name in ("pln_roman", "pln_tuffy"):
        path = os.path.join(PETTA, "examples", name + ".metta")
        forms = F.parse_sexprs(read(path))
        static, kb = [], None
        for form in forms:
            if isinstance(form, tuple) and len(form) == 3 and form[0] == "=":
                head = form[1]
                if isinstance(head, tuple) and len(head) == 2 and head[0] == "STV":
                    tv = form[2]
                    static.append((head[1], (tv[1], tv[2]), (0,)))
                elif head == ("kb",):
                    kb = F.kb_from_metta(form[2])
        sim = F.Sim(rules=lib_rules, static=static, variant="petta_lib")
        for form in forms:
            if isinstance(form, tuple) and len(form) == 2 and form[0] == "!" and form[1][0] == "test":
                call, exp = form[1][1], form[1][2]
                assert call[0] == "PLN.Query"
                expected = ((exp[0][1], exp[0][2]), tuple(exp[1]))
                got = sim.query(kb, call[2])
                compare_exact(name + " test", "PeTTa/examples/%s.metta" % name, expected, got, "bundled-lib defaults 100/10/100, Unique keeps order")

    # --- plntest.metta : a single (|- x y) with the Deduction rule
    sim = F.Sim(rules=["Deduction"], variant="petta_lib", static=[(n, (0.4, 0.9), (0,)) for n in ("a", "b", "c")])
    res = sim.infer2((("Inheritance", "a", "b"), (0.9, 0.9)), (("Inheritance", "b", "c"), (0.8, 0.9)))
    got = None if not res else ((res[0][1][0], res[0][1][1]), ())
    exp = ((0.7333333333333334, 0.9), ())
    ok = bool(res) and res[0][0] == ("Inheritance", "a", "c") and compare_exact("plntest (|- deduction)", "PeTTa/examples/plntest.metta:50-52", exp, got)

    # --- plntestdirect.metta : Truth_Deduction on two (0.9 0.9) premises with STV a,b,c = (0.4 0.9)
    tv = F.Truth_Deduction((0.4, 0.9), (0.4, 0.9), (0.4, 0.9), (0.9, 0.9), (0.9, 0.9))
    compare_exact("plntestdirect (Truth_Deduction)", "PeTTa/examples/plntestdirect.metta:46-47", ((0.8166666666666668, 0.9), ()), (tv, ()))

    # --- pln_direct.metta : formula-level (Truth_ModusPonens, Truth_Revision); no Deriver involved
    MP, REV = F.Truth_ModusPonens, F.Truth_Revision
    imp = (1.0, 0.9)
    one = (1.0, 1.0)
    # (=> (, (father $a $b) (father $b $c)) (grandfather $a $c) (stv 1.0 0.9)):  MP(foldl MP over the premises, imp)
    p1 = MP(MP(one, (1.0, 0.9)), (1.0, 0.9))            # father a b, father b c
    g1 = MP(p1, imp)
    p2 = MP(MP(one, (1.0, 0.9)), (0.0, 0.9))            # father a y, mother y c
    g2 = MP(p2, imp)
    ans = REV(REV((0.5, 0.0), g1), g2)
    compare_exact("pln_direct (grandfather a c)", "PeTTa/examples/pln_direct.metta:26-27", ((0.51, 0.8432620011567381), ()), (ans, ()), "formula-level: MP chain then Revision fold from (stv 0.5 0.0)")
    o1, o2 = MP(g1, imp), MP(g2, imp)
    ans2 = REV(REV((0.5, 0.0), o1), o2)
    compare_exact("pln_direct (old a)", "PeTTa/examples/pln_direct.metta:30-31", ((0.5198, 0.7923434575206812), ()), (ans2, ()), "formula-level")


if __name__ == "__main__":
    main()
    petta_examples()
    ok = sum(1 for r in RESULTS if r["match"])
    for r in RESULTS:
        print(("MATCH   " if r["match"] else "MISMATCH"), r["case"], "[%s]" % r["source"])
        print("          expected :", r["expected"])
        print("          simulated:", r["simulated"])
        if r["note"]:
            print("          note     :", r["note"])
    print("\n%d / %d match   (eq_order=%s unique=%s)" % (ok, len(RESULTS), EQ_ORDER, UNIQUE))
    if JSON_OUT:
        with open(JSON_OUT, "w") as f:
            json.dump(RESULTS, f, indent=1)
