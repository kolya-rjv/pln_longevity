"""G0(a) and G0(b): upstream examples on the vendored library, and the load-time probes (compared in Python)."""
from g0lib import *  # noqa: F401,F403
import subprocess

OUT = {"checks": [], "data": {}}


def check(name, expected, measured, ok):
    OUT["checks"].append(dict(check=name, expected=str(expected), measured=str(measured), ok=bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + " | expected " + str(expected) + " | measured " + str(measured), flush=True)


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_sh(example, timeout=120):
    path = os.path.join(SCRATCH, "examples", example + ".metta")
    t = time.perf_counter()
    r = subprocess.run(["sh", PETTA_HOME + "/run.sh", path], capture_output=True, text=True, timeout=timeout, cwd=SCRATCH)
    dt = time.perf_counter() - t
    text = ANSI.sub("", r.stdout + r.stderr)
    return r.returncode, text, dt


def fl(x, y, tol=1e-12):
    return abs(x - y) <= tol


def prepare_examples():
    """Copies of PeTTa's pln_roman / pln_tuffy / pln_direct with the import line edited to the vendored file (absolute path)."""
    d = os.path.join(SCRATCH, "examples")
    os.makedirs(d, exist_ok=True)
    for lib, path in (("full", LIB_FULL), ("restricted", LIB_RESTRICTED)):
        for ex in ("pln_roman", "pln_tuffy", "pln_direct"):
            src = open(PETTA_HOME + "/examples/%s.metta" % ex, encoding="utf-8").read()
            fixed = re.sub(r"^!\(import! &self (?:\.\./lib/lib_pln|\(library lib_pln\))\)", '!(import! &self "%s")' % path, src, flags=re.M)
            assert fixed != src, ex
            open(os.path.join(d, "%s_%s.metta" % (ex, lib)), "w", encoding="utf-8").write(fixed)


# ---------------------------------------------------------------------------------------------- (a) run.sh on copies
def part_a():
    expectations = {
        "pln_roman": ("((stv 0.5 0.9473684210526316) (1 2 3 4))", 1),
        "pln_tuffy": ("((stv 0.5199204545454544 0.8290782204129108) (1 2 4 9 10))", 1),
        "pln_direct": (None, 3),
    }
    for lib, examples in (("full", ("pln_roman", "pln_tuffy", "pln_direct")), ("restricted", ("pln_tuffy", "pln_direct"))):
        for ex in examples:
            rc, text, dt = run_sh("%s_%s" % (ex, lib))
            ok_n, bad_n = text.count("✅"), text.count("❌")
            want_n = expectations[ex][1]
            check("run.sh %s on the %s %s library: every pinned `test` passes" % (
                ex, "UNFILTERED vendored" if lib == "full" else "RESTRICTED artifact", "(copy, import line edited)"),
                "%d pass, 0 fail, exit 0" % want_n, "%d pass, %d fail, exit %d, %.2f s" % (ok_n, bad_n, rc, dt),
                ok_n == want_n and bad_n == 0 and rc == 0)
    # negative control: Deduction was filtered out, so pln_roman must NOT pass on the restricted artifact
    rc, text, dt = run_sh("pln_roman_restricted")
    check("negative control: pln_roman on the RESTRICTED artifact fails (no Deduction rule)", "1 fail", "%d pass, %d fail" % (
        text.count("✅"), text.count("❌")), text.count("❌") == 1 and text.count("✅") == 0)


# ---------------------------------------------------------------------------------------------- (a) in Python
def example_kb(example):
    """The example's text without its import line and its `!(test ...)` forms."""
    src = open(PETTA_HOME + "/examples/%s.metta" % example, encoding="utf-8").read()
    lines = src.split("\n")
    out, skip = [], 0
    i = 0
    text = "\n".join(l for l in lines if not l.startswith("!(import!"))
    # drop the `!(test ...)` forms (balanced)
    res, k = [], 0
    while k < len(text):
        if text.startswith("!(test", k):
            depth, j = 0, k + 1
            while True:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            k = j + 1
            continue
        res.append(text[k])
        k += 1
    return "".join(res)


def part_a_python():
    tuffy_expr = "(PLN.Query (kb) (Inheritance Edward (IntSet cancerous)))"
    roman_expr = "(PLN.Query (kb) (Inheritance A D))"
    pinned = {"tuffy": (0.5199204545454544, 0.8290782204129108, (1, 2, 4, 9, 10)),
              "roman": (0.5, 0.9473684210526316, (1, 2, 3, 4))}

    def read(res_text, pin):
        v = parse_one(res_text)  # ((stv s c) (ids))
        assert v[0][0] == "stv", v
        return v[0][1], v[0][2], tuple(v[1]), res_text

    # unfiltered library through PeTTa's Python interface (fresh process each)
    for name, expr in (("roman", roman_expr), ("tuffy", tuffy_expr)):
        r = petta_py(files=[LIB_FULL], strings=[example_kb("pln_" + name)], exprs=[expr], tag="a_" + name)
        res = r["results"][0]
        assert res["ok"] and len(res["out"]) == 1, res
        s, c, ids, txt = read(res["out"][0], pinned[name])
        p = pinned[name]
        check("pln_%s through PeTTa's Python interface on the UNFILTERED vendored library: pinned value, Python comparison" % name,
              "stv %r %r, stamp %r (exact)" % p, "%s" % txt, (s, c, ids) == p)
    # direct formulas: pln_direct needs add-atom -> PeTTa's Python interface, both libraries
    for lib, path in (("unfiltered", LIB_FULL), ("restricted", LIB_RESTRICTED)):
        r = petta_py(files=[path], strings=[example_kb("pln_direct")], exprs=["(? (grandfather a c))", "(? (grandfather $who c))", "(? (old $who))"],
                     tag="a_direct_" + lib)
        outs = [x["out"] if x["ok"] else x["error"] for x in r["results"]]
        want = ["(((grandfather a c) (stv 0.51 0.8432620011567381)))", "(((grandfather a c) (stv 0.51 0.8432620011567381)))",
                "(((old a) (stv 0.5198 0.7923434575206812)))"]
        # PeTTa returns one result (a tuple) per expression; compare numerically in Python
        ok = True
        shown = []
        for o, w in zip(outs, want):
            if not (isinstance(o, list) and len(o) == 1):
                ok = False
                shown.append(str(o))
                continue
            got, exp = parse_one(o[0].replace("(quote ", "(").replace("))))", "))))")), parse_one(w)
            shown.append(o[0])
            ok &= (str(got).replace(" ", "") == str(exp).replace(" ", "")) or _num_equal(got, exp)
        check("pln_direct formulas (Truth_ModusPonens chains + Truth_Revision fold) on the %s library, Python comparison" % lib,
              "0.51/0.8432620011567381 (x2), 0.5198/0.7923434575206812", " ; ".join(shown), ok)
    # restricted artifact through the production adapter: pln_tuffy
    with Engine("restricted", extra=example_kb("pln_tuffy")) as eng:
        txt = eng.one(tuffy_expr)
        s, c, ids, _ = read(txt, None)
        check("pln_tuffy through the production adapter session on the RESTRICTED artifact: pinned value, Python comparison",
              "stv 0.5199204545454544 0.8290782204129108, stamp (1 2 4 9 10) (exact)", txt, (s, c, ids) == pinned["tuffy"])
        # the formula level of pln_direct (what its Revision fold computes), evaluated as data on the sandboxed adapter
        v = parse_one(eng.one("(foldl-atom ((stv 1.0 0.729) (stv 0.02 0.729)) (stv 0.5 0.0) Truth_Revision)"))
        check("pln_direct revision fold of the two routes, adapter session (restricted)", "stv 0.51 0.8432620011567381 (<=1e-15)",
              v, v[0] == "stv" and fl(v[1], 0.51, 1e-15) and fl(v[2], 0.8432620011567381, 1e-15))


def _num_equal(a, b, tol=1e-15):
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(_num_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


# ---------------------------------------------------------------------------------------------- (b) probes
def probes(ev, label):
    """ev(expr) -> list of result strings (one group).  Every probe compares in Python (never MeTTa test/assert)."""
    def one(expr):
        g = ev(expr)
        return g

    # MP probe
    g = one("(Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8))")
    v = parse_one(g[0]) if len(g) == 1 else None
    check("[%s] Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8) = (0.26, 0.64)" % label, "stv 0.26 0.64 (<=1e-15)", g,
          v is not None and v[0] == "stv" and fl(v[1], 0.26, 1e-15) and fl(v[2], 0.64, 1e-15))
    g = one("(Truth_ModusPonens (stv 0.0 1.0) (stv 0.0 1.0))")
    v = parse_one(g[0]) if len(g) == 1 else None
    check("[%s] leak e = strength of Truth_ModusPonens (stv 0 1) (stv 0 1) is 0.02" % label, "0.02", g, v is not None and fl(v[1], 0.02, 1e-15) and fl(v[2], 1.0, 1e-15))
    # Revision with c = 1.0
    for expr in ("(Truth_Revision (stv 0.5 1.0) (stv 0.5 0.5))", "(Truth_Revision (stv 0.5 0.5) (stv 0.5 1.0))",
                 "(Truth_Revision (stv 0.5 1.0) (stv 0.5 1.0))"):
        try:
            g = one(expr)
            ok = (g == [])
        except Exception as exc:  # noqa: BLE001
            g, ok = "RAISED %s" % exc, False
        check("[%s] %s gives NO result (empty, no exception)" % (label, expr), "[]", g, ok)
    g = one("(Truth_Revision (stv 0.5 0.999999) (stv 0.5 0.5))")
    check("[%s] Truth_Revision with c capped at 0.999999 gives a result" % label, "one stv", g, len(g) == 1 and parse_one(g[0])[0] == "stv")
    # MixCompound fold (design 1 section 10): rows (0.5 0.9) (0.6667 0.35) (0.3333 0.2) (0.0 0.9) -> Neg 0.263050 / native c 0.949466
    g = one("(foldl-atom ((stv 0.6666666666666666 0.35) (stv 0.3333333333333333 0.2) (stv 0.0 0.9)) (stv 0.5 0.9) Truth_Revision)")
    v = parse_one(g[0])
    # independent arithmetic in Python
    acc = (0.5, 0.9)
    for (f2, c2) in ((0.6666666666666666, 0.35), (0.3333333333333333, 0.2), (0.0, 0.9)):
        w1, w2 = acc[1] / (1 - acc[1]), c2 / (1 - c2)
        w = w1 + w2
        f = (w1 * acc[0] + w2 * f2) / w
        c = w / (w + 1)
        acc = (min(1.0, f), min(1.0, max(max(c, acc[1]), c2)))
    check("[%s] Truth_Revision fold of the 4 MixCompound rows = hand-typed arithmetic (<=1e-15) = decision row 5 (0.263050 / 0.949466)" % label,
          "stv %.6f %.6f" % acc, g, fl(v[1], acc[0], 1e-15) and fl(v[2], acc[1], 1e-15) and abs(v[1] - 0.263050) < 5e-7 and abs(v[2] - 0.949466) < 5e-7)
    # Sentence / stamp patterns
    g = one("(PriorityRank (Sentence ((State X) (stv 1.0 1.0)) (0)))")
    check("[%s] PriorityRank matches (Sentence ((State X) (stv 1.0 1.0)) (0))" % label, "1.0", g, g == ["1.0"])
    g = one("(PriorityRankNeg (Sentence ((State X) (stv 1.0 1.0)) (0)))")
    check("[%s] PriorityRankNeg matches the same sentence" % label, "-1.0", g, g in (["-1.0"], ["-1"]))
    g = one("(PriorityRank (Sentence ((End X) (stv 0.3 0.65)) (1 2)))")
    check("[%s] PriorityRank matches an (End X) term with a two-id stamp" % label, "0.65", g, g == ["0.65"])
    g = one("(ConfidenceRank ((stv 1.0 1.0) (0)))")
    check("[%s] ConfidenceRank matches ((stv 1.0 1.0) (0))" % label, "1.0", g, g == ["1.0"])
    g1 = one("(StampDisjoint (0) (1 2))")
    g2 = one("(StampDisjoint (0) (0))")
    g3 = one("(StampDisjoint (0 3) (0 4))")
    check("[%s] StampDisjoint (0) (1 2) true; (0) (0) false; (0 3) (0 4) false" % label, "true/false/false", [g1, g2, g3],
          [x.lower() for x in g1] == ["true"] and [x.lower() for x in g2] == ["false"] and [x.lower() for x in g3] == ["false"])
    g = one("(BestCandidate PriorityRank () ((Sentence ((State A) (stv 1.0 1.0)) (0)) (Sentence ((State B) (stv 0.5 0.9)) (1))))")
    check("[%s] BestCandidate PriorityRank picks the premise (stamp (0), c 1.0) first" % label, "the premise sentence", g,
          len(g) == 1 and g[0].startswith("(Sentence ((State A) (stv 1.0 1.0)) (0))"))
    # the rules fire on (State X)/(End X) terms
    g = one("(|- ((State A) (stv 1.0 1.0)) ((Implication (State A) (State B)) (stv 0.706 0.65)))")
    v = parse_one(g[0]) if len(g) == 1 else None
    check("[%s] ModusPonens rule fires on (State A) with an Implication (State A) (State B): exactly one result" % label,
          "((State B) (stv 0.706 0.65))", g, v is not None and v[0] == ("State", "B") and fl(v[1][1], 0.706, 1e-15) and fl(v[1][2], 0.65, 1e-15))
    g = one("(|- ((State A) (stv 1.0 1.0)) ((Implication (State A) (Not (End B))) (stv 0.706 0.65)))")
    v = parse_one(g[0]) if len(g) == 1 else None
    check("[%s] ModusPonens derives (Not (End B)) for a Neg record link" % label, "((Not (End B)) (stv 0.706 0.65))", g,
          v is not None and v[0] == ("Not", ("End", "B")))
    g = one("(|- ((Not (End B)) (stv 0.706 0.65)))")
    v = parse_one(g[0]) if len(g) == 1 else None
    check("[%s] NotElimination derives (End B) with 1 - s, same c" % label, "((End B) (stv 0.294 0.65))", g,
          v is not None and v[0] == ("End", "B") and fl(v[1][1], 0.294, 1e-15) and fl(v[1][2], 0.65, 1e-15))
    g = one("(|- ((End B) (stv 0.5 0.9)) ((Implication (State B) (State C)) (stv 0.7 0.9)))")
    check("[%s] a record (End B) never continues a chain: no MP with Implication (State B) .." % label, "[]", g, g == [])
    # Derive returns exactly one result of shape ($Tasks $Beliefs), for assorted inputs
    prem = "(Sentence ((State A) (stv 1.0 1.0)) (0))"
    lk = "(Sentence ((Implication (State A) (State B)) (stv 0.7 0.9)) (1))"
    for tag, expr in (("normal", "(PLN.Derive (%s) (%s %s) 1 100 10 100)" % (prem, lk, prem)),
                      ("empty tasks", "(PLN.Derive () (%s) 1 100 10 100)" % lk),
                      ("steps > maxsteps", "(PLN.Derive (%s) (%s) 5 4 10 100)" % (prem, prem)),
                      ("K = 256", "(PLN.Derive (%s) (%s %s) 1 1024 256 256)" % (prem, lk, prem))):
        g = one(expr)
        v = parse_one(g[0]) if len(g) == 1 else None
        check("[%s] PLN.Derive (%s) returns exactly one result of shape ($Tasks $Beliefs)" % (label, tag), "1 result, 2-tuple", "%d result(s): %s" % (len(g), g[0][:100] if g else g),
              len(g) == 1 and isinstance(v, tuple) and len(v) == 2)


def part_b():
    with Engine("restricted") as eng:
        probes(lambda e: eng.run("!" + e)[0], "restricted/adapter")
    cache = {}
    counter = [0]

    def unfiltered(e):
        if e not in cache:
            counter[0] += 1
            r = petta_py(files=[LIB_FULL], exprs=[e], tag="b_full_%d" % counter[0])
            x = r["results"][0]
            cache[e] = x["out"] if x["ok"] else ("RAISED " + x["error"])
        return cache[e]
    probes(unfiltered, "unfiltered/PeTTa python interface")



def part_trace():
    """The one hunk: the patched libraries print nothing per step; the upstream bundled file prints one line per selection."""
    body = ("(= (kb) ((Sentence ((Implication (State A) (State B)) (stv 0.7 0.9)) (1))\n"
            "         (Sentence ((State A) (stv 1.0 1.0)) (0))))\n"
            "!(PLN.Derive ((Sentence ((State A) (stv 1.0 1.0)) (0))) (kb) 1 100 10 100)\n")
    res = {}
    for label, lib in (("vendored", LIB_FULL), ("restricted", LIB_RESTRICTED), ("upstream bundled (unpatched)", PETTA_HOME + "/lib/lib_pln.metta")):
        path = os.path.join(SCRATCH, "examples", "trace_probe.metta")
        open(path, "w").write('!(import! &self "%s")\n' % lib + body)
        rc, text, dt = run_sh("trace_probe")
        res[label] = (text.count("SELECTED"), "(Sentence ((State B) (stv 0.7 0.9)) (0 1))" in text)
    check("per-step trace: vendored and restricted print no SELECTED line, the unpatched upstream file prints one per selection",
          "0 / 0 / >=1, result still printed", "%s" % res,
          res["vendored"][0] == 0 and res["restricted"][0] == 0 and res["upstream bundled (unpatched)"][0] >= 1
          and res["vendored"][1] and res["restricted"][1] and res["upstream bundled (unpatched)"][1])



def part_hunk():
    """The one hunk is neutral: 60 patient runs give byte-identical result strings on the vendored file, the restricted artifact
    (through the adapter) and the UNPATCHED upstream file (PeTTa's own interface)."""
    links = load_links("patient")
    assign_ids(links)
    kb_all = build_kb(links, 0.02)
    by = {l["id"]: s for l, s in zip(links, kb_all)}
    exprs = []
    for A in sorted({l["from"] for l in links}):
        kb = [by[l["id"]] for l in reachable_links(links, A)]
        for level in (1.0, 0.0):
            exprs.append(derive_text(kb, A, level, SETTINGS["petta_lib_defaults"]))
    outs = {}
    for label, path in (("unpatched upstream", PETTA_HOME + "/lib/lib_pln.metta"), ("vendored", LIB_FULL)):
        r = petta_py(files=[path], exprs=exprs, tag="hunk_" + label.split()[0])
        outs[label] = [x["out"] if x["ok"] else x["error"] for x in r["results"]]
    with Engine("restricted") as eng:
        outs["restricted via adapter"] = [eng.run(e)[0] for e in exprs]

    def norm(o):
        return [[s.lower() if s in ("True", "False") else s for s in g] for g in o]
    same_v = norm(outs["unpatched upstream"]) == norm(outs["vendored"])
    same_r = norm(outs["vendored"]) == norm(outs["restricted via adapter"])
    check("the one hunk is neutral: %d PLN.Derive runs give byte-identical result strings on the unpatched upstream file and the vendored file; "
          "the restricted artifact (adapter) gives the same strings" % len(exprs), "identical, identical", "%s, %s" % (same_v, same_r), same_v and same_r)


if __name__ == "__main__":
    prepare_examples()
    part_a()
    part_a_python()
    part_b()
    part_trace()
    part_hunk()
    json.dump(OUT, open(SCRATCH + "/results_ab.json", "w"), indent=1, default=str)
    fails = [c for c in OUT["checks"] if not c["ok"]]
    print("\n%d checks, %d failed" % (len(OUT["checks"]), len(fails)))
    for c in fails:
        print("FAILED:", c["check"], "|", c["measured"])
