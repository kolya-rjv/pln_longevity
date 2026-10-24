"""One scenario per finding: `python repro.py <name>` prints one JSON object. Same script before and after the fixes."""
import sys, os, json, subprocess, time
from fx import *


def out(**kw):
    print(json.dumps(kw, default=str, sort_keys=True))


def sc_zero():
    from ontology.drugage_selector import load_rows
    from ontology import drugage_fold as df
    import dataclasses
    rows = load_rows(REPO / "tests/fixtures/drugage_fold/three_compounds.metta")
    k = df.load_knobs()
    res = {}
    for comp in ["Resveratrol", "Metformin", "Rapamycin", "Ethanol"]:
        rs = [r for r in rows if r.compound == comp]
        lifted = [l for l in (df.lift_row(r, k) for r in rs) if l]
        zeros = [l for l in lifted if l.row.avg_change == 0]
        o = {"rows": len(rs), "zero_rows": len(zeros)}
        _, net = df.fold_lifted(lifted, fold=True, chain_discount=k.chain_discount); o["default"] = round(net.score, 4)
        _, net = df.fold_lifted([l for l in lifted if l.row.avg_change != 0], fold=True, chain_discount=k.chain_discount); o["neither"] = round(net.score, 4)
        both = list(lifted) + [dataclasses.replace(l, sign=df.SIGN_POS) for l in zeros]
        _, net = df.fold_lifted(both, fold=True, chain_discount=k.chain_discount); o["both"] = round(net.score, 4)
        res[comp] = o
    rows = load_rows(BUILD)
    by = {}
    for r in rows:
        by.setdefault(r.compound, []).append(r)
    tot = withzero = flip_neither = flip_both = 0
    sg = lambda x: "harm" if x < 0 else ("zero" if x == 0 else "prot")
    for comp, rs in by.items():
        lifted = [l for l in (df.lift_row(r, k) for r in rs) if l]
        if not lifted:
            continue
        tot += 1
        _, net = df.fold_lifted(lifted, fold=True, chain_discount=k.chain_discount)
        zeros = [l for l in lifted if l.row.avg_change == 0]
        if not zeros:
            continue
        withzero += 1
        nz = [l for l in lifted if l.row.avg_change != 0]
        if nz:
            _, nb = df.fold_lifted(nz, fold=True, chain_discount=k.chain_discount)
            flip_neither += sg(net.score) != sg(nb.score)
        both = list(lifted) + [dataclasses.replace(l, sign=df.SIGN_POS) for l in zeros]
        _, nc = df.fold_lifted(both, fold=True, chain_discount=k.chain_discount)
        flip_both += sg(net.score) != sg(nc.score)
    out(scenario="zero", fixture=res, build={"scorable": tot, "with_zero": withzero, "sign_flips_neither": flip_neither, "sign_flips_both": flip_both})


def sc_rank():
    src = REPO / "tests/fixtures/drugage_fold/three_compounds.metta"
    o = {}
    for mode in ("legacy", "pln"):
        use(mode)
        for bpc in (True, False):
            try:
                res, rows = pln_runner.run_drugage_ranking(["Rapamycin", "Resveratrol", "Metformin", "Ethanol"], source=src,
                                                           best_per_compound=bpc, limit=(400 if not bpc else None))
                o[f"{mode}/bpc={bpc}"] = {"status": res.status, "rows": len(rows), "error": res.error,
                                          "atoms": [a.atom for a in res.results[1:]]}
            except Exception as e:
                o[f"{mode}/bpc={bpc}"] = {"exc": repr(e)[:200]}
    out(scenario="rank", **{k.replace("/", "_"): v for k, v in o.items()})


def sc_exact():
    from ontology.drugage_selector import load_rows
    from ontology import drugage_fold as df
    rows = load_rows(REPO / "tests/fixtures/drugage_fold/three_compounds.metta")
    k = df.load_knobs()

    def items_for(comp):
        o = []
        for r in rows:
            if r.compound != comp:
                continue
            l = df.lift_row(r, k)
            if not l:
                continue
            o.append(dict(compound=comp, target="Lifespan", direction="Pos" if l.sign == df.SIGN_NEG else "Neg", strength=l.strength,
                          confidence=l.native_confidence, rows=(r.row_id,), papers=(r.pmid or "",), change=r.avg_change,
                          significance=r.significance, itp=r.is_itp, species=r.species))
        return o
    res = items_for("Resveratrol") + items_for("Metformin")
    dummy = dict(compound="OtherDrug", target="Lifespan", direction="Pos", strength=0.3, confidence=0.5, rows=("DrugAgeRow_X",),
                 papers=("PMID_9",), change=10.0, significance=None, itp=False, species="Mus_musculus")
    exact, mixed = df.fold_items(res), df.fold_items(res + [dummy])
    net = lambda o, c: [round(x["net"], 4) for x in o if x["compound"] == c][0]
    out(scenario="exact", alone={c: net(exact, c) for c in ("Resveratrol", "Metformin")},
        with_unrelated_row={c: net(mixed, c) for c in ("Resveratrol", "Metformin")})


def sc_skip():
    o = {}
    cases = {"strength_1.5": "(Effect Foo Bar Pos (stv 1.5 0.5))\n(Effect Bar Baz Pos (stv 0.5 0.5))\n",
             "negative": "(Effect Foo Bar Pos (stv -0.1 0.5))\n(Effect Bar Baz Pos (stv 0.5 0.5))\n",
             "percentage": "(Effect Foo Bar Pos (stv 50 0.9))\n(Effect Bar Baz Pos (stv 0.5 0.5))\n"}
    for name, atoms in cases.items():
        for mode in ("legacy", "pln"):
            use(mode, PLN_QUERY_TIMEOUT_SECONDS=60)
            r, dt = run(MIN, atoms, ["!(infer &self Foo Baz)"])
            o[f"{name}/{mode}"] = short(r)
    out(scenario="skip", **{k.replace("/", "_"): v for k, v in o.items()})


def sc_back():
    import api
    STACK = pln_runner.patient_stack(api._runtime_kb_paths())
    extra = """
(: BackRec EmpiricalAssociation)
(HasPredictor BackRec InsulinResistance)
(HasOutcome BackRec Metformin)
(EffectMetric BackRec "hazard_ratio")
(EffectValue BackRec 1.3)
(EffectScale BackRec "x")
(PValue BackRec 0.001)
(SupportedByPublication BackRec LuEtAl2019_GrimAge)
(HasEvidenceCategory BackRec MultipleHumanTrials)
"""
    o = {}
    for mode in ("legacy", "pln"):
        use(mode)
        r, dt = run(STACK, extra, ["!(let $r (explain &self Metformin Metformin) (R $r))",
                                  "!(let $r (explain &self Metformin InsulinResistance) (R2 $r))"])
        o[mode] = short(r)
    out(scenario="back", **o)


def sc_badcfg():
    o = {}
    for var, val in (("PLN_QUEUE_SIZE", "abc"), ("PLN_QUEUE_SIZE", ""), ("PLN_MAX_STEPS", "x"), ("PLN_RULESET", "bogus"),
                     ("PLN_CHAIN_CONFIDENCE", "zzz")):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PLN_")}
        env.update({"PYTHON_DOTENV_DISABLED": "1", "PLN_RUNTIME": "hyperon", var: val})
        p = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import config; print('imported', config.PLN_CHAINER)" % str(REPO / "pln_chat")],
                           capture_output=True, text=True, env=env, timeout=120)
        o[f"{var}={val!r}"] = (p.stdout.strip() or p.stderr.strip().splitlines()[-1])[:100]
    # and with the chainer ON the strictness must stay
    for var, val in (("PLN_QUEUE_SIZE", "abc"),):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PLN_")}
        env.update({"PYTHON_DOTENV_DISABLED": "1", "PLN_RUNTIME": "petta", "PLN_CHAINER": "pln", var: val})
        p = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import config; print('imported', config.PLN_CHAINER)" % str(REPO / "pln_chat")],
                           capture_output=True, text=True, env=env, timeout=120)
        o[f"pln+{var}={val!r}"] = (p.stdout.strip() or p.stderr.strip().splitlines()[-1])[:100]
    out(scenario="badcfg", **o)


def _rows(r, comp, pct, sig="Significant", pmid="PMID_1", sp="Mus_musculus"):
    return (f"(InstanceOf {r} Experiment)\n(UsesIntervention {r} {comp})\n(UsesSpecies {r} {sp})\n(HasSex {r} Male)\n(ReportedIn {r} {pmid})\n"
            f"(AvgLifespanChangePercent {r} {pct})\n(AvgLifespanSignificance {r} {sig})\n")


def sc_dabad():
    files = list(pln_runner.DRUGAGE_STACK)
    cases = {
        "dup_change_atom": _rows("DrugAgeRow_1", "X", 10.0) + "(AvgLifespanChangePercent DrugAgeRow_1 10.0)\n",
        "dup_whole_row": _rows("DrugAgeRow_1", "X", 10.0) + _rows("DrugAgeRow_1", "X", 10.0),
        "two_changes": _rows("DrugAgeRow_1", "X", 10.0) + "(AvgLifespanChangePercent DrugAgeRow_1 -20.0)\n",
        "row_two_compounds": _rows("DrugAgeRow_1", "X", 10.0) + "(UsesIntervention DrugAgeRow_1 Y)\n",
    }
    o = {}
    for name, extra in cases.items():
        for mode in ("legacy", "pln"):
            use(mode, PLN_QUERY_TIMEOUT_SECONDS=60)
            r, dt = run(files, extra, ["!(infer &self X Mortality)", "!(infer &self Y Mortality)"])
            o[f"{name}/{mode}"] = short(r, 200)
    out(scenario="dabad", **{k.replace("/", "_"): v for k, v in o.items()})


def sc_readans():
    use("pln")
    r, dt = run(MIN, "(Effect A B Pos (stv 0.5 0.8))\n(Effect B C Pos (stv 0.5 0.9))\n", ["!(match &self ($h A $b $c $d $e $f) $h)"])
    out(scenario="readans", **short(r))


def sc_order():
    atoms = "(Effect A B Pos (stv 0.5 0.8))\n(Effect A B Neg (stv 0.3 0.6))\n(Effect A C Neg (stv 0.9 0.9))\n(Effect C B Pos (stv 0.9 0.9))\n(Effect A D Pos (stv 0.5 0.8))\n(Effect D B Neg (stv 0.9 0.9))\n"
    o = {}
    for mode in ("legacy", "pln"):
        use(mode)
        r, dt = run(MIN, atoms, ["!(explain &self A B)"])
        o[mode] = short(r, 100)
    out(scenario="order", **o)


def sc_cost(kind):
    """cost scenarios: whole request, deadline 40 s"""
    DL = 40
    t0 = time.time()
    if kind == "dag":
        names, links = layered(6, 5, 3, 11)
        extra = effects(links)
        q = [f"!(let $r (explain &self {names[0][0]} $t) (R $r))"]
        routes = pc.count_routes([pc.Link(a, b, sg, s, c) for a, b, sg, s, c in links], names[0][0], 10**7)
    elif kind == "digraph5":
        nodes = list("ABCDE")
        links = [(a, b, "Pos", 0.5, 0.8) for a in nodes for b in nodes if a != b]
        extra, q, routes = effects(links), ["!(explain &self A E)"], None
    elif kind == "chain200":
        links = [(f"C{i}", f"C{i+1}", "Pos", 0.9, 0.9) for i in range(200)]
        extra, q, routes = effects(links), ["!(explain &self C0 C200)"], 200
    elif kind == "nolookup":
        names, links = layered(4, 6, 4, 5)
        extra, q, routes = effects(links), ["!(match &self (Effect L0N0 $b $s $t) $b)"], None
    elif kind == "feedback":
        import api
        STACK = pln_runner.patient_stack(api._runtime_kb_paths())
        rev = [("CoronaryHeartDisease", "ChronicInflammation"), ("ChronicInflammation", "CellularSenescence"),
               ("AgeAccelGrim", "SASP"), ("SASP", "DNAmPAI1"), ("DNAmGDF15", "CRP"), ("LowSerumAlbumin", "ChronicInflammation"),
               ("AllCauseMortality", "CoronaryHeartDisease"), ("CRP", "CellularSenescence")][:8]
        extra = effects([(a, b, "Pos", 0.4, 0.8) for a, b in rev])
        use("pln", PLN_QUERY_TIMEOUT_SECONDS=DL)
        r, dt = run(STACK, extra, ["!(explain &self CellularSenescence CoronaryHeartDisease)"])
        out(scenario="cost_feedback8", seconds=round(dt, 1), **short(r, 120)); return
    use("pln", PLN_QUERY_TIMEOUT_SECONDS=DL)
    r, dt = run(MIN, extra, q)
    out(scenario="cost_" + kind, links=len(links), routes=routes, seconds=round(dt, 1), **short(r, 120))


if __name__ == "__main__":
    name = sys.argv[1]
    if name.startswith("cost_"):
        sc_cost(name[5:])
    else:
        globals()["sc_" + name]()
