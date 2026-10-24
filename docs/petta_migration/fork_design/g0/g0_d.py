"""G0(d): wall time of PLN.Derive on the patient graph, the 10-run rank-interventions-for-patient workload and the
60-run eager precompute, against the LEGACY chainer on PeTTa on the same shape (the increment, not the hyperon figure).

Measured: (1) warm per-run wall time inside one PeTTa session (serialise, evaluate, parse), (2) fresh-process requests:
legacy-only vs legacy + library + N Derive expressions in the same child (the increment = what the pln lane adds),
(3) the Python adapter's join (parse + path rebuild + shim) separately, (4) a scale probe on layered trees."""
from g0lib import *  # noqa: F401,F403
import statistics

OUT = {"data": {}}
CAND = ["DasatinibPlusQuercetin", "Fisetin", "Spermidine", "Metformin", "Berberine"]
SMOKE10 = ["system_types", "logical_predicates", "epistemic_calibration", "grim_age_core", "grim_age_lu2019_evidence",
           "evidence_calibration", "hallmarks_core", "hallmarks_lopezotin2023_intervention_evidence",
           "mechanistic_bridges", "pln_deduction"]


def stats(xs):
    xs = sorted(xs)
    return dict(n=len(xs), min=round(xs[0], 4), median=round(statistics.median(xs), 4), mean=round(statistics.mean(xs), 4),
                p90=round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 4), max=round(xs[-1], 4), total=round(sum(xs), 4))


def kb_text_of(names):
    return "\n".join(open(REPO + "/%s.metta" % n, encoding="utf-8").read() for n in names)


def patient_kb_text():
    import api  # noqa: WPS433 (dotenv switched off in common)
    import core.pln_runner as pr
    paths = pr.patient_stack(api._runtime_kb_paths())
    return "\n".join(p.read_text(encoding="utf-8") for p in paths), [p.name for p in paths]


def derive_expr(kb, A, level, st):
    app = premise(A, level)
    beliefs = list(kb) + [app]
    return "!(PLN.Derive %s %s 1 %d %d %d)" % (
        sexp((("Sentence", (app[0], ("stv", app[1][0], app[1][1])), app[2]),)),
        sexp(tuple(("Sentence", (t, ("stv", s, c)), tuple(ids)) for (t, (s, c), ids) in beliefs)),
        st["maxsteps"], st["tq"], st["bq"])


def main():
    st = SETTINGS["K256"]
    links = load_links("patient")
    assign_ids(links)
    ids = {l["id"]: l for l in links}
    nodes = sorted({l["from"] for l in links} | {l["to"] for l in links})
    res = OUT["data"]
    P, pnames = patient_kb_text()
    res["patient_stack_files"] = len(pnames)
    res["patient_stack_bytes"] = len(P.encode())

    # ---------------------------------------------------------------- (1) warm per-run wall time in one session
    with Engine("restricted") as eng:
        leak = read_leak(eng)
        kb_all = build_kb(links, leak)
        bymap = {l["id"]: s for l, s in zip(links, kb_all)}

        def one_run(A, level):
            sub = reachable_links(links, A)
            kb = [bymap[l["id"]] for l in sub]
            t0 = time.perf_counter()
            text = derive_expr(kb, A, level, st)
            t1 = time.perf_counter()
            groups = eng.sess.run(text, 900)
            t2 = time.perf_counter()
            assert len(groups) == 1 and len(groups[0]) == 1
            out = parse_one(groups[0][0])
            tasks = [from_parsed(x) for x in out[0]]
            beliefs = [from_parsed(x) for x in out[1]]
            t3 = time.perf_counter()
            return (t1 - t0, t2 - t1, t3 - t2, len(sub), len(beliefs)), tasks, beliefs

        # warm-up (first call pays one-off compilation of the Derive clause bodies)
        first = []
        for _ in range(3):
            t = time.perf_counter()
            one_run("DasatinibPlusQuercetin", 1.0)
            first.append(time.perf_counter() - t)
        res["warmup_first_three_runs_s"] = [round(x, 4) for x in first]

        # the 60 runs, repeated 5 times
        reps = []
        per_run_eval, per_run_total, ser, par = [], [], [], []
        by_src = {}
        for rep in range(5):
            t0 = time.perf_counter()
            join_s = 0.0
            for A in nodes:
                b = {}
                for level in (1.0, 0.0):
                    (a_ser, a_eval, a_par, nl, nb), T, B = one_run(A, level)
                    per_run_eval.append(a_eval)
                    per_run_total.append(a_ser + a_eval + a_par)
                    ser.append(a_ser)
                    par.append(a_par)
                    b[level] = B
                    by_src.setdefault(A, []).append(a_ser + a_eval + a_par)
                tj = time.perf_counter()
                routes_for_source(b[1.0], b[0.0], ids, A, leak)
                join_s += time.perf_counter() - tj
            reps.append(dict(total_s=time.perf_counter() - t0, join_s=join_s))
        res["warm_60_runs_repeats"] = [dict(total_s=round(r["total_s"], 4), join_s=round(r["join_s"], 4)) for r in reps]
        res["warm_per_run_eval_s"] = stats(per_run_eval)
        res["warm_per_run_total_s"] = stats(per_run_total)
        res["warm_per_run_serialise_s"] = stats(ser)
        res["warm_per_run_parse_s"] = stats(par)
        res["warm_slowest_sources_s"] = sorted(((round(statistics.mean(v), 4), k) for k, v in by_src.items()), reverse=True)[:5]

        # the 10-run workload
        w10 = []
        for rep in range(7):
            t0 = time.perf_counter()
            for A in CAND:
                for level in (1.0, 0.0):
                    one_run(A, level)
            w10.append(time.perf_counter() - t0)
        res["warm_10_run_workload_s"] = stats(w10)
        # selections for the 10 runs (chunked resumption), and rule-attempt scale
        sel = {}
        for A in CAND:
            kb = [bymap[l["id"]] for l in reachable_links(links, A)]
            n, _, _ = count_selections(eng, kb, A, 1.0, st)
            sel[A] = n
        res["selections_candidates"] = sel
        res["selections_10_runs_total"] = 2 * sum(sel.values())

    # ---------------------------------------------------------------- (2) fresh-process requests
    rt = runtime()
    Q = "!(rank-interventions-for-patient &self Patient001 (%s) CoronaryHeartDisease)" % " ".join(CAND)
    lib = lib_text("restricted")
    kb_vals = build_kb(links, 0.02)
    bymap2 = {l["id"]: s for l, s in zip(links, kb_vals)}

    def exprs_for(sources, st_=st):
        out = []
        for A in sources:
            kb = [bymap2[l["id"]] for l in reachable_links(links, A)]
            for level in (1.0, 0.0):
                out.append(derive_expr(kb, A, level, st_))
        return out

    def timed_requests(label, kb_text, exprs, n=7):
        runs = []
        rows = []
        answer = None
        for _ in range(n):
            t = time.perf_counter()
            out = rt.run_program(kb_text, exprs, deadline_s=600)
            runs.append(time.perf_counter() - t)
            rows.append(dict(rt.last_run))
            answer = out
        res["request_" + label] = dict(wall=stats(runs), last_run=rows[-1], expressions=len(exprs))
        return runs, answer

    L, ansL = timed_requests("legacy_petta_patient_rank5", P, [Q])
    e10 = exprs_for(CAND)
    N10, ans10 = timed_requests("pln_patient_rank5_10derive", P + "\n" + lib, e10 + [Q])
    e60 = exprs_for(nodes)
    N60, ans60 = timed_requests("pln_patient_rank5_60derive", P + "\n" + lib, e60 + [Q])

    # answer atoms the eager seam would hand to infer/explain: 186 facts (100 patient pairs + 86 more) loaded as KB text
    facts = "\n".join("(PLNAnswer Node%d Node%d Neg 0.24989999999999998 0.17017138125000003 (Chain (Node%d Node%d)))" % (i, j, i, j)
                      for i in range(14) for j in range(14))[:]
    facts = "\n".join(facts.split("\n")[:186])
    timed_requests("pln_patient_rank5_60derive_plus_186_answer_atoms", P + "\n" + lib + "\n" + facts, e60 + [Q])
    res["increment_186_answer_atoms_s"] = round(res["request_pln_patient_rank5_60derive_plus_186_answer_atoms"]["wall"]["median"]
                                                - res["request_pln_patient_rank5_60derive"]["wall"]["median"], 4)

    # the legacy answer inside the combined load is unchanged (the library collides with nothing in the stack)
    res["legacy_answer_unchanged_with_library_loaded"] = (ansL[0] == ans10[-1] == ans60[-1])

    # smoke:2 shape = (infer &self CellularSenescence CoronaryHeartDisease) on the 10-file smoke stack
    S10 = kb_text_of(SMOKE10)
    q2 = "!(infer &self CellularSenescence CoronaryHeartDisease)"
    timed_requests("legacy_petta_smoke2", S10, [q2])
    timed_requests("pln_smoke2_1source_2derive", S10 + "\n" + lib, exprs_for(["CellularSenescence"]) + [q2])
    timed_requests("pln_smoke2_all30sources_60derive", S10 + "\n" + lib, e60 + [q2])
    # the cheapest possible request: a trivial program
    timed_requests("trivial_program_fresh_process", "", ["!(+ 1 2)"])
    timed_requests("library_load_only_fresh_process", lib, ["!(+ 1 2)"])

    def inc(a, b):
        return round(statistics.median(a) - statistics.median(b), 4)

    res["increment_rank5_10derive_s"] = inc(N10, L)
    res["increment_rank5_60derive_s"] = inc(N60, L)
    res["ratio_rank5_10derive"] = round(statistics.median(N10) / statistics.median(L), 3)
    res["ratio_rank5_60derive"] = round(statistics.median(N60) / statistics.median(L), 3)
    for key in ("pln_smoke2_1source_2derive", "pln_smoke2_all30sources_60derive"):
        a, b = res["request_" + key]["wall"]["median"], res["request_legacy_petta_smoke2"]["wall"]["median"]
        res["increment_" + key + "_s"] = round(a - b, 4)
        res["ratio_" + key] = round(a / b, 3)


    # ---------------------------------------------------------------- (2b) warm legacy on PeTTa, same shapes
    rt0 = runtime()
    sess = rt0.session(P, 900)
    try:
        sess.run(Q, 900)                          # warm-up
        qt = []
        for _ in range(30):
            t = time.perf_counter()
            out = sess.run(Q, 900)
            qt.append(time.perf_counter() - t)
        res["warm_legacy_rank5_query_s"] = stats(qt)
        res["legacy_rank5_answer_len"] = len(out[0][0]) if out and out[0] else 0
        tabs = []
        for rep in range(10):
            t = time.perf_counter()
            for A in nodes:
                sess.run("!(collapse (infer &self %s $c))" % A, 900)
            tabs.append(time.perf_counter() - t)
        res["warm_legacy_infer_table_30_sources_s"] = stats(tabs)
        one = []
        for rep in range(30):
            t = time.perf_counter()
            sess.run("!(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)", 900)
            one.append(time.perf_counter() - t)
        res["warm_legacy_infer_dq_chd_s"] = stats(one)
    finally:
        sess.close()

    # ---------------------------------------------------------------- (3) scale probe: layered trees (control lane risk)
    def tree(depth, fan):
        ls, n = [], 0
        frontier = [("n0",)]
        names = {("n0",): "T0"}
        idx = 0
        cur = ["T0"]
        cnt = 1
        for d in range(depth):
            nxt = []
            for a in cur:
                for f in range(fan):
                    b = "T%d" % cnt
                    cnt += 1
                    ls.append(link(a, b, "Pos" if (cnt % 3) else "Neg", 0.6 + 0.3 * ((cnt * 7) % 10) / 10, 0.7 + 0.2 * ((cnt * 3) % 10) / 10))
                    nxt.append(b)
            cur = nxt
        return ls

    scale = {}
    with Engine("restricted") as eng:
        leak = read_leak(eng)
        for depth, fan in ((3, 3), (4, 4), (5, 4), (4, 5)):
            tl = tree(depth, fan)
            assign_ids(tl)
            kbt = build_kb(tl, leak)
            L_ = len(tl)
            K = 2048 if L_ + 4 > 255 else 256
            stt = dict(maxsteps=4 * K, tq=K, bq=K)
            t = time.perf_counter()
            try:
                T1, B1 = run_level(eng, kbt, "T0", 1.0, stt)
                dt = time.perf_counter() - t
                scale["tree d=%d fan=%d links=%d K=%d" % (depth, fan, L_, K)] = dict(
                    seconds=round(dt, 3), beliefs=len(B1), tasks_left=len(T1), complete=(not T1) and len(B1) < K - 1)
            except Exception as exc:  # noqa: BLE001
                scale["tree d=%d fan=%d links=%d K=%d" % (depth, fan, L_, K)] = dict(error=str(exc)[:200])
            print("scale", list(scale.items())[-1], flush=True)
            if time.perf_counter() - t > 120:
                break
    res["scale_probe"] = scale

    json.dump(OUT, open(SCRATCH + "/results_d.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
