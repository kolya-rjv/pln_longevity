"""G0 library: the decision.md section-3 encoding ("state sentences with Modus Ponens") with the reviewers'
corrections, driven on the REAL PeTTa library through the production adapter (core.metta_runtime sessions),
plus an independent legacy reference (node-simple, right-nested arithmetic).  Scratch code, not production."""
from common import *  # noqa: F401,F403
from fork_sim import parse_sexprs  # the repo's validated simulator, used as a parser and as an oracle
import math

SIGN = {("Pos", "Pos"): "Pos", ("Pos", "Neg"): "Neg", ("Neg", "Pos"): "Neg", ("Neg", "Neg"): "Pos"}
CHAIN_DISCOUNT = 0.9
TOL = 1e-12

# --------------------------------------------------------------------------- MeTTa text <-> python tuples

def sexp(t):
    if isinstance(t, tuple):
        return "(" + " ".join(sexp(x) for x in t) + ")"
    if isinstance(t, bool):
        raise TypeError("bool")
    if isinstance(t, float):
        r = repr(t)
        assert "e" not in r and "inf" not in r and "nan" not in r, r
        return r
    if isinstance(t, int):
        return str(t)
    return str(t)


def sent(term, s, c, ids):
    """python sentence in fork_sim form: (term, (s, c), (ids))"""
    return (term, (s, c), tuple(ids))


def sent_text(x):
    term, (s, c), ids = x
    return sexp(("Sentence", (term, ("stv", s, c)), tuple(ids)))


def from_parsed(p):
    """('Sentence', (term, ('stv', s, c)), ids) -> (term, (s, c), ids)"""
    assert p[0] == "Sentence", p
    (term, stv), ids = p[1], p[2]
    assert stv[0] == "stv"
    return (term, (stv[1], stv[2]), tuple(ids))


def parse_one(text):
    forms = parse_sexprs(text)
    assert len(forms) == 1, text[:200]
    return forms[0]


# --------------------------------------------------------------------------- the engine (real PeTTa)

class Engine:
    """One PeTTa child (a PettaSession of the production adapter) holding a library; Derive calls are data-only."""

    def __init__(self, lib="restricted", extra="", deadline_s=900):
        self.rt = runtime()
        t = time.perf_counter()
        self.sess = self.rt.session(lib_text(lib) + "\n" + extra, deadline_s)
        self.start_s = time.perf_counter() - t
        self.deadline_s = deadline_s
        self.calls = 0
        self.eval_s = 0.0

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def run(self, text):
        t = time.perf_counter()
        groups = self.sess.run(text, self.deadline_s)
        dt = time.perf_counter() - t
        self.calls += 1
        self.eval_s += dt
        self.last_s = dt
        return groups

    def one(self, expr):
        """Evaluate one `!` expression; return its single result string (assert exactly one)."""
        groups = self.run("!" + expr if not expr.startswith("!") else expr)
        assert len(groups) == 1 and len(groups[0]) == 1, groups
        return groups[0][0]

    def derive(self, tasks, beliefs, steps, maxsteps, tq, bq):
        """PLN.Derive (6-argument form).  Returns (tasks, beliefs) as lists of fork_sim sentences; asserts that
        the call returns exactly ONE result of the shape ($Tasks $Beliefs)."""
        expr = "(PLN.Derive %s %s %d %d %d %d)" % (
            sexp(tuple(("Sentence", (t, ("stv", s, c)), tuple(ids)) for (t, (s, c), ids) in tasks)),
            sexp(tuple(("Sentence", (t, ("stv", s, c)), tuple(ids)) for (t, (s, c), ids) in beliefs)),
            steps, maxsteps, tq, bq)
        groups = self.run("!" + expr)
        assert len(groups) == 1, groups
        assert len(groups[0]) == 1, "PLN.Derive must return exactly one result, got %d: %r" % (len(groups[0]), groups[0][:2])
        out = parse_one(groups[0][0])
        assert isinstance(out, tuple) and len(out) == 2, out[:2]
        return [from_parsed(x) for x in out[0]], [from_parsed(x) for x in out[1]]


# --------------------------------------------------------------------------- links, ids, encoding

def load_links(stack):
    d = json.load(open(CONTRACT + "/links.json"))
    return [dict(l) for l in d["links"] if stack in l["stacks"]]


def link(frm, to, sign, p, c, terminal=False, **kw):
    d = {"from": frm, "to": to, "sign": sign, "strength": p, "confidence": c, "terminal_only": terminal,
         "kind": kw.get("kind", "Effect"), "source": kw.get("source", "curated")}
    d.update(kw)
    return d


def assign_ids(links):
    """Deterministic ids 1..n (0 is the premise): sorted by (from, to, sign, terminal, strength, confidence)."""
    order = sorted(range(len(links)), key=lambda i: (links[i]["from"], links[i]["to"], links[i]["sign"],
                                                     links[i]["terminal_only"], links[i]["strength"],
                                                     links[i]["confidence"]))
    ids = {}
    for n, i in enumerate(order, start=1):
        links[i]["id"] = n
        ids[n] = links[i]
    return ids


def assert_ids_unique_along_paths(links):
    """Reviewer correction (e): an evidence id may be shared only by parallel evidence of ONE link; two links of a
    derivable path must not share an id (StampDisjoint would refuse the chain).  Here every link has its own id,
    so the check is that the `id` field is unique over links."""
    seen = {}
    for l in links:
        ids = l["id"] if isinstance(l["id"], tuple) else (l["id"],)
        for i in ids:
            assert i not in seen or seen[i] is l, "evidence id %r is shared by two links" % (i,)
            seen[i] = l


def link_term(l):
    tgt = ("End" if l["terminal_only"] else "State", l["to"])
    return ("Implication", ("State", l["from"]), tgt if l["sign"] == "Pos" else ("Not", tgt))


def stored_strength(p, leak, mapping="affine"):
    if mapping == "affine":       # reviewer correction (a): q = e + (1-e) p, exact for every p in [0,1]
        return leak + (1.0 - leak) * p
    if mapping == "capped":       # the earlier decision text: q = min(1, p + e)
        return min(1.0, p + leak)
    if mapping == "raw":          # the fork read literally: q = p
        return p
    raise ValueError(mapping)


def build_kb(links, leak, mapping="affine"):
    return [sent(link_term(l), stored_strength(l["strength"], leak, mapping), l["confidence"], (l["id"],))
            for l in links]


def reachable_links(links, A):
    out_by = {}
    for l in links:
        out_by.setdefault(l["from"], []).append(l)
    seen, frontier, keep = {A}, [A], []
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


def premise(A, level):
    return sent(("State", A), level, 1.0, (0,))


def read_leak(engine):
    """e = strength of Truth_ModusPonens (stv 0 1) (stv 0 1), read from the library at load (never typed)."""
    r = parse_one(engine.one("(Truth_ModusPonens (stv 0.0 1.0) (stv 0.0 1.0))"))
    assert r[0] == "stv"
    return r[1]


# --------------------------------------------------------------------------- one source = two Derive runs

SETTINGS = {
    "petta_lib_defaults": dict(maxsteps=100, tq=10, bq=100),    # PeTTa lib: PLN.Config.* = 100 / 10 / 100
    "fork_defaults": dict(maxsteps=20, tq=20, bq=200),         # patham9/PLN Config.metta = 20 / 20 / 200
    "K256": dict(maxsteps=1024, tq=256, bq=256),               # the production schedule's first rung: 4K, K, K
    "K2048": dict(maxsteps=8192, tq=2048, bq=2048),
}


def run_level(engine, kb, A, level, settings, tasks="applied"):
    app = premise(A, level)
    beliefs = list(kb) + [app]
    t0 = [app] if tasks == "applied" else list(beliefs)
    return engine.derive(t0, beliefs, 1, settings["maxsteps"], settings["tq"], settings["bq"])


def count_selections(engine, kb, A, level, settings):
    """Chunked resumption with the 6-argument form: one selection per call (maxsteps = steps).  Returns
    (selections, tasks, beliefs); the final state must equal the one-shot run's."""
    app = premise(A, level)
    tasks, beliefs = [app], list(kb) + [app]
    n = 0
    while tasks and n < settings["maxsteps"]:
        tasks, beliefs = engine.derive(tasks, beliefs, n + 1, n + 1, settings["tq"], settings["bq"])
        n += 1
    return n, tasks, beliefs


def path_from_stamp(ids, A, stamp):
    used = [ids[i] for i in stamp if i != 0]
    out = {}
    for l in used:
        if l["from"] in out:
            return None
        out[l["from"]] = l
    path, cur = [A], A
    while cur in out:
        cur = out[cur]["to"]
        path.append(cur)
        if len(path) > len(out) + 1:
            return None
    return path if len(path) == len(out) + 1 else None


def is_state(t):
    return isinstance(t, tuple) and len(t) == 2 and t[0] in ("State", "End")


def routes_for_source(b1, b0, ids, A, leak, mapping="affine", shim=True):
    """The adapter step: join the level-1 and level-0 beliefs on the stamp; rebuild the node-simple path;
    strength = |s1 - s0| / (1-e)^hops (affine), sign = parity of Neg links (cross-checked against the sign of
    s1 - s0), confidence = c_native * 0.9^(hops-1) (shim).  Returns {C: [route, ...]}."""
    idx0 = {}
    for (t, (s, c), ev) in b0:
        if is_state(t):
            idx0[(t[1], ev)] = (s, c)
    out = {}
    for (t, (s1, c1), ev) in b1:
        if not is_state(t) or t[1] == A or c1 <= 0:
            continue
        C = t[1]
        path = path_from_stamp(ids, A, ev)
        if path is None or path[-1] != C or len(set(path)) != len(path):
            continue
        v0 = idx0.get((C, ev))
        if v0 is None:
            continue
        hops = len(path) - 1
        ate = s1 - v0[0]
        denom = (1.0 - leak) ** hops if mapping == "affine" else 1.0
        sign_struct = "Neg" if sum(1 for i in ev if i != 0 and ids[i]["sign"] == "Neg") % 2 else "Pos"
        sign_ate = ("Neg" if ate < 0 else "Pos") if abs(ate) > 1e-12 else None
        c = c1 * (CHAIN_DISCOUNT ** (hops - 1)) if shim else c1
        out.setdefault(C, []).append(dict(path=tuple(path), sign=sign_struct, sign_ate=sign_ate, s=abs(ate) / denom,
                                          c=c, c_native=c1, level1=s1, level0=v0[0], ate=ate, stamp=ev, hops=hops,
                                          end=t[0]))
    for C in out:
        out[C].sort(key=lambda r: (r["hops"], r["stamp"]))
    return out


# --------------------------------------------------------------------------- the legacy reference (independent)

def legacy_routes(links, A):
    """Node-simple legacy `infer` from A: [(nodes, sign, s, c)].  Right-nested floats exactly as pln_deduction.metta:
    s = s_link * s_rest, c = (c_link * c_rest) * 0.9.  Records (terminal_only) end a chain and never continue one."""
    out_by = {}
    for l in links:
        out_by.setdefault(l["from"], []).append(l)

    def derive(a, visited):
        res = []
        for l in out_by.get(a, []):
            b = l["to"]
            if b in visited:
                continue
            res.append(((a, b), l["sign"], l["strength"], l["confidence"]))
            if not l["terminal_only"]:
                for (nodes, sg2, s2, c2) in derive(b, visited | {b}):
                    res.append(((a,) + nodes, SIGN[(l["sign"], sg2)], l["strength"] * s2,
                                (l["confidence"] * c2) * CHAIN_DISCOUNT))
        return res

    return derive(A, frozenset([A]))


def compare_source(routes, legacy, A):
    """Compare PeTTa routes {C: [route]} with legacy [(nodes, sign, s, c)] for source A.
    Returns dict(missing=[...], extra=[...], mismatches=[...], matched=n)."""
    mine = {}
    for C, rs in routes.items():
        for r in rs:
            mine[r["path"]] = r
    leg = {nodes: (sg, s, c) for (nodes, sg, s, c) in legacy}
    assert len(leg) == len(legacy), "legacy reference has duplicate paths"
    missing = [list(p) for p in leg if p not in mine]
    extra = [list(p) for p in mine if p not in leg]
    mism, matched = [], 0
    for p, r in mine.items():
        if p not in leg:
            continue
        sg, s, c = leg[p]
        bad = []
        if r["sign"] != sg:
            bad.append(("sign", r["sign"], sg))
        if r["sign_ate"] is not None and r["sign_ate"] != sg:
            bad.append(("sign_ate", r["sign_ate"], sg))
        if abs(r["s"] - s) > TOL:
            bad.append(("s", r["s"], s))
        if abs(r["c"] - c) > TOL:
            bad.append(("c", r["c"], c))
        if bad:
            mism.append((list(p), bad))
        else:
            matched += 1
    return dict(missing=missing, extra=extra, mismatches=mism, matched=matched)


def closure_run(engine, links, leak, settings, mapping="affine", prefilter=True, sources=None, shim=True):
    """Run every source of `links` (two Derive runs each) and compare with the legacy reference.  Returns a summary
    dict with pair counts, per-source stats and any difference."""
    ids = {l["id"]: l for l in links}
    nodes = sorted({l["from"] for l in links} | {l["to"] for l in links})
    srcs = sources or nodes
    kb_all = build_kb(links, leak, mapping)
    by_id = {l["id"]: s for l, s in zip(links, kb_all)}
    tot = dict(pairs_pet=0, pairs_leg=0, routes_pet=0, routes_leg=0, missing=0, extra=0, mismatches=0, matched=0,
               max_dev_s=0.0, max_dev_c=0.0, sources=0, runs=0, unfinished=0, multi_route_pairs=0)
    detail, per_source = [], {}
    for A in srcs:
        sub = reachable_links(links, A) if prefilter else links
        kb = [by_id[l["id"]] for l in sub]
        t0 = time.perf_counter()
        T1, B1 = run_level(engine, kb, A, 1.0, settings)
        T0, B0 = run_level(engine, kb, A, 0.0, settings)
        dt = time.perf_counter() - t0
        tot["runs"] += 2
        K = settings["bq"]
        unfinished = bool(T1) or bool(T0) or len(B1) >= K - 1 or len(B0) >= K - 1
        tot["unfinished"] += int(unfinished)
        routes = routes_for_source(B1, B0, ids, A, leak, mapping, shim)
        leg = legacy_routes(links, A)
        cmp_ = compare_source(routes, leg, A)
        tot["sources"] += 1
        tot["pairs_pet"] += len(routes)
        tot["pairs_leg"] += len({n[-1] for (n, _, _, _) in leg})
        tot["routes_pet"] += sum(len(v) for v in routes.values())
        tot["routes_leg"] += len(leg)
        tot["multi_route_pairs"] += sum(1 for v in routes.values() if len(v) > 1)
        tot["missing"] += len(cmp_["missing"])
        tot["extra"] += len(cmp_["extra"])
        tot["mismatches"] += len(cmp_["mismatches"])
        tot["matched"] += cmp_["matched"]
        for p, r in ((tuple(r["path"]), r) for v in routes.values() for r in v):
            for (nodes_, sg, s, c) in leg:
                if nodes_ == p:
                    tot["max_dev_s"] = max(tot["max_dev_s"], abs(r["s"] - s))
                    tot["max_dev_c"] = max(tot["max_dev_c"], abs(r["c"] - c))
        per_source[A] = dict(pairs=len(routes), belief1=len(B1), belief0=len(B0), links=len(sub), s=round(dt, 4),
                             unfinished=unfinished)
        if cmp_["missing"] or cmp_["extra"] or cmp_["mismatches"]:
            detail.append((A, cmp_))
    tot["detail"] = detail
    tot["per_source"] = per_source
    return tot


def fmt_summary(name, tot):
    return ("%s: pairs %d/%d routes %d/%d missing %d extra %d mismatch %d max|ds| %.2e max|dc| %.2e multi-route pairs %d "
            "unfinished runs %d" % (name, tot["pairs_pet"], tot["pairs_leg"], tot["routes_pet"], tot["routes_leg"],
                                    tot["missing"], tot["extra"], tot["mismatches"], tot["max_dev_s"],
                                    tot["max_dev_c"], tot["multi_route_pairs"], tot["unfinished"]))


# --------------------------------------------------------------------------- PeTTa's own Python interface (unfiltered lib)

def petta_py(files=(), strings=(), exprs=(), tag="run", timeout=300):
    """Fresh process, PeTTa's python/petta interface.  Returns the child's JSON ({"loaded", "results"})."""
    import subprocess
    spec = os.path.join(SCRATCH, "spec_%s.json" % tag)
    out = os.path.join(SCRATCH, "out_%s.json" % tag)
    if os.path.exists(out):
        os.remove(out)
    json.dump({"files": list(files), "strings": list(strings), "exprs": list(exprs)}, open(spec, "w"))
    env = dict(os.environ, PLN_G0_PETTA_HOME=PETTA_HOME, PYTHON_DOTENV_DISABLED="1")
    r = subprocess.run([PY, "-B", os.path.join(str(HERE), "petta_py_child.py"), spec, out], stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=timeout, cwd=SCRATCH, text=True, env=env)
    if not os.path.exists(out):
        raise RuntimeError("petta child failed rc=%s: %s" % (r.returncode, r.stderr[-800:]))
    return json.load(open(out))


def derive_text(kb, A, level, st):
    """The `!(PLN.Derive ...)` expression for one run: task = the premise, beliefs = links + premise (data only)."""
    app = premise(A, level)
    beliefs = list(kb) + [app]
    return "!(PLN.Derive %s %s 1 %d %d %d)" % (
        sexp((("Sentence", (app[0], ("stv", app[1][0], app[1][1])), app[2]),)),
        sexp(tuple(("Sentence", (t, ("stv", s, c)), tuple(ids)) for (t, (s, c), ids) in beliefs)),
        st["maxsteps"], st["tq"], st["bq"])
