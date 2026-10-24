"""G0(c): the decision.md section-3 encoding (with the reviewers' affine strength correction) on the REAL PeTTa
library, restricted artifact, through the production adapter (core.metta_runtime sessions).

Checks: legacy reference vs closure.json (bit-identical), closure of the patient (100), LinAge2 (86) and generic (101)
stacks at three budget settings, the feedback edge (109), D+Q intermediate levels, selections by chunked resumption,
the fork_sim oracle (exact belief-set equality), completeness status (tq = bq vs tq < bq), id uniqueness, the
number lanes for D+Q -> CHD.  Writes results_c.json."""
from g0lib import *  # noqa: F401,F403
import fork_sim

OUT = {"checks": [], "data": {}}


def check(name, expected, measured, ok):
    OUT["checks"].append(dict(check=name, expected=str(expected), measured=str(measured), ok=bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + " | expected " + str(expected) + " | measured " + str(measured), flush=True)


def load_closure(stack):
    return json.load(open(CONTRACT + "/closure.json"))["stacks"][stack]["derivations"]


def legacy_vs_closure_json(stack, links):
    """Does the independent legacy reference (own DFS) reproduce closure.json (real hyperon output) bit for bit?"""
    mine = {}
    for A in sorted({l["from"] for l in links} | {l["to"] for l in links}):
        for (nodes, sg, s, c) in legacy_routes(links, A):
            mine[nodes] = (sg, s, c)
    ref = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"]) for d in load_closure(stack)}
    same_keys = set(mine) == set(ref)
    exact = sum(1 for k in ref if k in mine and mine[k] == ref[k])
    return len(ref), len(mine), same_keys, exact


def main():
    res = OUT["data"]
    with Engine("restricted") as eng:
        res["engine_start_s"] = round(eng.start_s, 3)
        leak = read_leak(eng)
        check("leak e read from Truth_ModusPonens (stv 0 1) (stv 0 1)", 0.02, leak, abs(leak - 0.02) < 1e-15)

        expected_pairs = {"patient": 100, "linage2": 86, "generic": 101}
        stacks = {}
        for name in ("patient", "linage2", "generic"):
            ls = load_links(name)
            assign_ids(ls)
            assert_ids_unique_along_paths(ls)
            stacks[name] = ls
            res["links_" + name] = len(ls)

        # 1. the legacy reference is the real hyperon closure
        for name in ("patient", "linage2"):
            n_ref, n_mine, same, exact = legacy_vs_closure_json(name, stacks[name])
            check("legacy reference == closure.json (hyperon), %s: derivations, same path set, bit-identical floats" % name,
                  "%d/%d/True/%d" % (n_ref, n_ref, n_ref), "%d/%d/%s/%d" % (n_ref, n_mine, same, exact),
                  same and exact == n_ref and n_mine == n_ref)


        # 1b. the repo's own executable spec of the legacy chainer (contract/legacy_reference.py) is a second oracle
        sys.path.insert(0, CONTRACT)
        import legacy_reference as lr
        for name in ("patient", "linage2"):
            ls = stacks[name]
            effects = [l["atom"] for l in ls if not l["terminal_only"]]
            recs = []
            for k, l in enumerate(x for x in ls if x["terminal_only"]):
                hr = 2.0 if l["strength"] == 0.70 else (1.3 if l["strength"] == 0.60 else (1.05 if l["sign"] == "Pos" else 0.9))
                cat = {0.85: "MultipleHumanTrials", 0.60: "Epidemiological"}[l["confidence"]]
                recs.append(dict(id="r%d" % k, predictor=l["from"], outcome=l["to"], hr=hr, category=cat))
            ref = lr.Legacy(effects, recs)
            nodes_ = sorted({l["from"] for l in ls} | {l["to"] for l in ls})
            mine = {}
            for a in nodes_:
                for b in nodes_:
                    for (chain_, sg, s, c) in ref.derive(a, b):
                        mine[tuple(chain_)] = (sg, s, c)
            mine_dfs = {}
            for a in nodes_:
                for (nodes__, sg, s, c) in legacy_routes(ls, a):
                    mine_dfs[nodes__] = (sg, s, c)
            check("contract/legacy_reference.py (the repo's executable spec) == the independent legacy DFS used here == closure.json, %s" % name,
                  "same paths, bit-identical floats", "%d paths; identical to DFS: %s" % (len(mine), mine == mine_dfs),
                  mine == mine_dfs and len(mine) == expected_pairs[name])

        # 2. closure per stack and budget
        for name in ("patient", "linage2", "generic"):
            for sname in ("petta_lib_defaults", "fork_defaults", "K256"):
                t = time.time()
                tot = closure_run(eng, stacks[name], leak, SETTINGS[sname])
                res["closure_%s_%s" % (name, sname)] = {k: v for k, v in tot.items() if k not in ("detail", "per_source")}
                res["closure_%s_%s" % (name, sname)]["wall_s"] = round(time.time() - t, 3)
                ok = (tot["pairs_pet"] == tot["pairs_leg"] == expected_pairs[name] and tot["missing"] == 0 and tot["extra"] == 0
                      and tot["mismatches"] == 0 and tot["multi_route_pairs"] == 0)
                check("closure %s @ %s: ordered pairs equal to legacy in sign, strength, confidence, chain; none extra, none missing"
                      % (name, sname), "%d pairs, 0 missing, 0 extra, 0 mismatches (<=1e-12)" % expected_pairs[name],
                      "%d pairs (legacy %d), missing %d, extra %d, mismatches %d, max|ds| %.2e, max|dc| %.2e"
                      % (tot["pairs_pet"], tot["pairs_leg"], tot["missing"], tot["extra"], tot["mismatches"],
                         tot["max_dev_s"], tot["max_dev_c"]),
                      ok and tot["max_dev_s"] <= TOL and tot["max_dev_c"] <= TOL)
                if tot["detail"]:
                    print("   detail", tot["detail"][:3])
        # no pre-filter (whole KB as beliefs) gives the same answers on the patient stack
        for sname in ("petta_lib_defaults", "K256"):
            tot = closure_run(eng, stacks["patient"], leak, SETTINGS[sname], prefilter=False)
            check("closure patient WITHOUT the reachable pre-filter @ %s" % sname, "100 pairs, 0/0/0",
                  "%d pairs, missing %d, extra %d, mismatches %d" % (tot["pairs_pet"], tot["missing"], tot["extra"], tot["mismatches"]),
                  tot["pairs_pet"] == 100 and tot["missing"] == tot["extra"] == tot["mismatches"] == 0)


        # 2b. harness sensitivity (negative controls): the earlier capped mapping and the literal fork mapping must be CAUGHT
        gen = stacks["generic"]
        tot = closure_run(eng, gen, leak, SETTINGS["K256"], mapping="capped")
        flagged = [(a, m[0], [b[0] for b in m[1]]) for a, c in tot["detail"] for m in c["mismatches"]]
        check("negative control: the earlier q = min(1, p + e) mapping fails the generic-stack closure on exactly Lifespan -> Mortality (0.98 vs 1.0)",
              "1 mismatch: Lifespan->Mortality strength 0.98", "%d mismatch(es): %s, max|ds| %.3f" % (tot["mismatches"], flagged, tot["max_dev_s"]),
              tot["mismatches"] == 1 and flagged and flagged[0][1] == ["Lifespan", "Mortality"])
        totr = closure_run(eng, stacks["patient"], leak, SETTINGS["K256"], mapping="raw")
        check("negative control: the fork read literally (q = p) fails the patient closure (strength shrinks)", "100 of 100 pairs differ",
              "%d of %d pairs differ in strength (max %.3f)" % (totr["mismatches"], totr["pairs_pet"], totr["max_dev_s"]),
              totr["mismatches"] == 100 and totr["max_dev_s"] > 0.03)

        # 3. the lanes for D+Q -> CHD (patient stack): legacy / R0 raw+native / R1 affine+native / R2 affine+shim / capped
        A = "DasatinibPlusQuercetin"
        ids = {l["id"]: l for l in stacks["patient"]}
        lanes = {}
        for lane, (mapping, shim) in {"R2 affine+shim": ("affine", True), "R1 affine+native c": ("affine", False),
                                      "R0 raw q=p, native c": ("raw", False), "capped q=min(1,p+e), shim": ("capped", True)}.items():
            sub = reachable_links(stacks["patient"], A)
            kb = build_kb(sub, leak, mapping)
            T1, B1 = run_level(eng, kb, A, 1.0, SETTINGS["K256"])
            T0, B0 = run_level(eng, kb, A, 0.0, SETTINGS["K256"])
            r = routes_for_source(B1, B0, ids, A, leak, mapping, shim)["CoronaryHeartDisease"][0]
            lanes[lane] = (r["sign"], r["s"], r["c"])
        leg = [x for x in legacy_routes(stacks["patient"], A) if x[0][-1] == "CoronaryHeartDisease"][0]
        lanes["legacy (hyperon closure.json)"] = (leg[1], leg[2], leg[3])
        res["lanes_DQ_CHD"] = lanes
        check("D+Q -> CHD lanes: R2 equals legacy 0.2499 / 0.17017138125; R1 keeps 0.2499 with c 0.2334; R0 0.2226 / 0.2334",
              "R2 (Neg 0.2499, 0.17017138); R1 (0.2499, 0.23343125); R0 (0.2226, 0.23343125)",
              "R2 %s; R1 %s; R0 %s" % (tuple(round(v, 8) if isinstance(v, float) else v for v in lanes["R2 affine+shim"]),
                                       tuple(round(v, 8) if isinstance(v, float) else v for v in lanes["R1 affine+native c"]),
                                       tuple(round(v, 8) if isinstance(v, float) else v for v in lanes["R0 raw q=p, native c"])),
              abs(lanes["R2 affine+shim"][1] - 0.2499) < 1e-12 and abs(lanes["R2 affine+shim"][2] - 0.17017138125) < 1e-12
              and abs(lanes["R1 affine+native c"][1] - 0.2499) < 1e-12 and abs(lanes["R1 affine+native c"][2] - 0.23343125) < 1e-12
              and abs(lanes["R0 raw q=p, native c"][1] - 0.2226) < 5e-4 and lanes["R2 affine+shim"][0] == "Neg")

        # 4. D+Q intermediate levels with q = p (raw), decision section 1 row 2 / design1 section 2 table
        sub = reachable_links(stacks["patient"], A)
        kb = build_kb(sub, leak, "raw")
        T1, B1 = run_level(eng, kb, A, 1.0, SETTINGS["K256"])
        T0, B0 = run_level(eng, kb, A, 0.0, SETTINGS["K256"])
        chain = ["CellularSenescence", "SASP", "DNAmPAI1", "CoronaryHeartDisease"]
        want = {"CellularSenescence": (0.30000, 0.98000), "SASP": (0.26900, 0.83340), "DNAmPAI1": (0.20292, 0.58671),
                "CoronaryHeartDisease": (0.13769, 0.36029)}
        got = {}
        for i, node in enumerate(chain):
            prefix = tuple(sorted([0] + [l["id"] for l in sub if (l["from"], l["to"]) in
                                         set(zip(["DasatinibPlusQuercetin"] + chain[:i], chain[:i + 1]))]))
            a1 = [s for (t, (s, c), ev) in B1 if is_state(t) and t[1] == node and ev == prefix]
            a0 = [s for (t, (s, c), ev) in B0 if is_state(t) and t[1] == node and ev == prefix]
            got[node] = (a1[0], a0[0])
        res["DQ_levels_raw"] = got
        dq_ok = all(abs(got[n][0] - want[n][0]) < 5e-6 and abs(got[n][1] - want[n][1]) < 5e-6 for n in chain)
        diff_raw = got["CoronaryHeartDisease"][0] - got["CoronaryHeartDisease"][1]
        check("D+Q intermediate levels, raw q=p, level 1 / level 0 (CS, SASP, DNAmPAI1, CHD)",
              "0.30000/0.98000, 0.26900/0.83340, 0.20292/0.58671, 0.13769/0.36029; CHD difference -0.222599",
              "; ".join("%s %.5f/%.5f" % (n, got[n][0], got[n][1]) for n in chain) + "; CHD difference %.6f" % diff_raw,
              dq_ok and abs(diff_raw + 0.222599) < 5e-7)
        # and with the affine mapping the difference divided by 0.98^4 is 0.2499 (checked in the closure above)

        # 5. selections by chunked resumption (6-argument form), K256 and fork defaults
        sel = {}
        for sname in ("K256", "fork_defaults", "petta_lib_defaults"):
            mx, mxsrc, per = 0, None, {}
            consistent = True
            for src in sorted({l["from"] for l in stacks["patient"]}):
                sub = reachable_links(stacks["patient"], src)
                kb = build_kb(sub, leak)
                n, tasks, beliefs = count_selections(eng, kb, src, 1.0, SETTINGS[sname])
                T1, B1 = run_level(eng, kb, src, 1.0, SETTINGS[sname])
                consistent &= (beliefs == B1 and tasks == T1)
                per[src] = n
                if n > mx:
                    mx, mxsrc = n, src
            sel[sname] = dict(max=mx, argmax=mxsrc, per_source=per, resumption_equals_one_shot=consistent)
            check("selections (chunked resumption) patient @ %s: max over sources, final state == one-shot run" % sname,
                  "max <= 12, resumption == one-shot", "max %d (%s), D+Q %d, resumption==one-shot %s" % (
                      mx, mxsrc, per[A], consistent), mx <= 12 and consistent)
        res["selections_patient"] = sel

        # 6. the feedback edge ChronicInflammation -> CellularSenescence
        fb = [dict(l) for l in stacks["patient"]]
        for l in fb:
            l.pop("id", None)
        fb.append(link("ChronicInflammation", "CellularSenescence", "Pos", 0.5, 0.65))
        assign_ids(fb)
        assert_ids_unique_along_paths(fb)
        for sname in ("petta_lib_defaults", "fork_defaults", "K256"):
            tot = closure_run(eng, fb, leak, SETTINGS[sname])
            res["feedback_" + sname] = {k: v for k, v in tot.items() if k not in ("detail", "per_source")}
            check("feedback edge CI -> CS @ %s: terminates, 109 ordered pairs equal to the node-simple legacy reference" % sname,
                  "109 pairs, 0 missing/extra/mismatch", "%d pairs (legacy %d), missing %d, extra %d, mismatches %d, multi-route %d"
                  % (tot["pairs_pet"], tot["pairs_leg"], tot["missing"], tot["extra"], tot["mismatches"], tot["multi_route_pairs"]),
                  tot["pairs_pet"] == tot["pairs_leg"] == 109 and tot["missing"] == tot["extra"] == tot["mismatches"] == 0)
        # old answers unchanged
        old = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"]) for d in load_closure("patient")}
        idsfb = {l["id"]: l for l in fb}
        kbfb = build_kb(fb, leak)
        bymap = {l["id"]: s for l, s in zip(fb, kbfb)}
        unchanged, new_pairs, selfb = 0, set(), {}
        for src in sorted({l["from"] for l in fb}):
            sub = reachable_links(fb, src)
            kb = [bymap[l["id"]] for l in sub]
            T1, B1 = run_level(eng, kb, src, 1.0, SETTINGS["K256"])
            T0, B0 = run_level(eng, kb, src, 0.0, SETTINGS["K256"])
            rt = routes_for_source(B1, B0, idsfb, src, leak)
            n, _, _ = count_selections(eng, kb, src, 1.0, SETTINGS["K256"])
            selfb[src] = n
            for C, rs in rt.items():
                for r in rs:
                    if r["path"] in old:
                        o = old[r["path"]]
                        if o[0] == r["sign"] and abs(o[1] - r["s"]) <= TOL and abs(o[2] - r["c"]) <= TOL:
                            unchanged += 1
                    else:
                        new_pairs.add(r["path"])
        res["feedback_selections"] = selfb
        res["feedback_new_pairs"] = sorted("->".join(p) for p in new_pairs)
        check("feedback edge: the 100 old answers unchanged, 9 new pairs", "100 unchanged, 9 new", "%d unchanged, %d new" % (unchanged, len(new_pairs)),
              unchanged == 100 and len(new_pairs) == 9)
        check("feedback edge selections: D+Q 12, CellularSenescence 10, max over sources 16 (decision section 1 row 11)",
              "12 / 10 / 16", "%d / %d / %d" % (selfb[A], selfb["CellularSenescence"], max(selfb.values())),
              selfb[A] == 12 and selfb["CellularSenescence"] == 10 and max(selfb.values()) == 16)

        # 7. oracle: fork_sim (validated pure-python port) gives the SAME final (tasks, beliefs) as the real library
        exact, total, worst, order_equal = 0, 0, 0.0, 0
        for name in ("patient", "linage2", "generic"):
            ls = stacks[name]
            for sname in ("petta_lib_defaults", "K256"):
                st = SETTINGS[sname]
                kball = build_kb(ls, leak)
                bymap = {l["id"]: s for l, s in zip(ls, kball)}
                for src in sorted({l["from"] for l in ls}):
                    sub = reachable_links(ls, src)
                    kb = [bymap[l["id"]] for l in sub]
                    for level in (1.0, 0.0):
                        T, B = run_level(eng, kb, src, level, st)
                        sim = fork_sim.Sim(rules="restricted", maxsteps=st["maxsteps"], task_queue_size=st["tq"],
                                           belief_queue_size=st["bq"], variant="petta_lib")
                        app = premise(src, level)
                        Ts, Bs = sim.derive([app], list(kb) + [app], st["maxsteps"], st["tq"], st["bq"])
                        total += 1
                        same = (len(B) == len(Bs) and len(T) == len(Ts))
                        dev = 0.0
                        if same:
                            for a, b in zip(B + T, Bs + Ts):
                                if a[0] != b[0] or a[2] != tuple(b[2]):
                                    same = False
                                    break
                                dev = max(dev, abs(a[1][0] - b[1][0]), abs(a[1][1] - b[1][1]))
                        worst = max(worst, dev)
                        if same and dev <= 1e-12:
                            exact += 1
                        if same and dev == 0.0:
                            order_equal += 1
        res["oracle"] = dict(runs=total, equal=exact, bit_identical=order_equal, worst=worst)
        check("fork_sim oracle: final (tasks, beliefs) of the simulator == the real library (same order, floats <= 1e-12)",
              "%d of %d" % (total, total), "%d of %d (bit-identical %d, worst deviation %.2e)" % (exact, total, order_equal, worst),
              exact == total)

        # 8. completeness status: tq = bq = K flags every cut; tq < bq is blind
        ls = stacks["patient"]
        kball = build_kb(ls, leak)
        bymap = {l["id"]: s for l, s in zip(ls, kball)}
        full_pairs = {}
        for src in sorted({l["from"] for l in ls}):
            sub = reachable_links(ls, src)
            kb = [bymap[l["id"]] for l in sub]
            T1, B1 = run_level(eng, kb, src, 1.0, SETTINGS["K256"])
            T0, B0 = run_level(eng, kb, src, 0.0, SETTINGS["K256"])
            full_pairs[src] = (set(routes_for_source(B1, B0, {l["id"]: l for l in ls}, src, leak)), kb)

        def pairs_with(src, tq, bq, steps=1024):
            kb = full_pairs[src][1]
            T1, B1 = engine_run(src, kb, 1.0, steps, tq, bq)
            T0, B0 = engine_run(src, kb, 0.0, steps, tq, bq)
            got = set(routes_for_source(B1, B0, {l["id"]: l for l in ls}, src, leak))
            complete = (not T1) and (not T0) and len(B1) < bq - 1 and len(B0) < bq - 1
            return got, complete

        def engine_run(src, kb, level, steps, tq, bq):
            return run_level(eng, kb, src, level, dict(maxsteps=steps, tq=tq, bq=bq))

        cut_detected = cut_undetected = lost_total = 0
        blind_detected = blind_undetected = 0
        for src in full_pairs:
            want_pairs = full_pairs[src][0]
            if not want_pairs:
                continue
            for K in (6, 8, 10, 12, 16):
                got, complete = pairs_with(src, K, K)
                lost = want_pairs - got
                if lost:
                    lost_total += 1
                    if complete:
                        cut_undetected += 1
                    else:
                        cut_detected += 1
            for tq, bq in ((2, 200), (3, 200), (4, 200)):
                got, complete = pairs_with(src, tq, bq)
                lost = want_pairs - got
                if lost:
                    if complete:
                        blind_undetected += 1
                    else:
                        blind_detected += 1
        res["completeness"] = dict(tq_eq_bq_cut_runs=lost_total, detected=cut_detected, undetected=cut_undetected,
                                   tq_lt_bq_detected=blind_detected, tq_lt_bq_undetected=blind_undetected)
        check("completeness test (tasks empty and |beliefs| < K-1) with tq = bq = K: every run that lost pairs is flagged",
              "0 undetected", "%d runs lost pairs, %d flagged, %d undetected" % (lost_total, cut_detected, cut_undetected),
              lost_total > 0 and cut_undetected == 0)
        check("the same test with tq < bq (tq 2..4, bq 200) is blind (decision correction 1)", "some undetected losses",
              "%d lost-pair runs flagged, %d undetected" % (blind_detected, blind_undetected), blind_undetected > 0)

        # 9. evidence ids: shared on two SERIAL links, StampDisjoint refuses the chain
        two = [link("A", "B", "Pos", 0.8, 0.9), link("B", "C", "Pos", 0.7, 0.9)]
        assign_ids(two)
        two[1]["id"] = two[0]["id"]  # shared id along a derivable path
        kb2 = build_kb(two, leak)
        T, B = run_level(eng, kb2, "A", 1.0, SETTINGS["K256"])
        have_C = any(is_state(t) and t[1] == "C" for (t, tv, ev) in B)
        two[1]["id"] = two[0]["id"] + 1
        kb3 = build_kb(two, leak)
        T, B3 = run_level(eng, kb3, "A", 1.0, SETTINGS["K256"])
        have_C3 = any(is_state(t) and t[1] == "C" for (t, tv, ev) in B3)
        try:
            assert_ids_unique_along_paths([dict(two[0], id=7), dict(two[1], id=7)])
            guard = False
        except AssertionError:
            guard = True
        check("serial links sharing an evidence id: the library refuses the chain (A->C not derived); distinct ids derive it; the adapter guard fires",
              "C absent with a shared id, present with distinct ids, guard raises", "shared: C %s; distinct: C %s; guard %s" % (
                  "present" if have_C else "absent", "present" if have_C3 else "absent", guard),
              (not have_C) and have_C3 and guard)


        # 10. determinism: the answers do not depend on the order the links are listed in (contract MUST 5)
        import random
        rng = random.Random(20261024)
        ls = stacks["patient"]
        ids_ = {l["id"]: l for l in ls}
        kb_all2 = build_kb(ls, leak)
        by2 = {l["id"]: s for l, s in zip(ls, kb_all2)}
        diffs = 0
        runs_ = 0
        for src in sorted({l["from"] for l in ls}):
            sub = [by2[l["id"]] for l in reachable_links(ls, src)]
            base = None
            variants = [list(sub), list(reversed(sub))] + [rng.sample(sub, len(sub)) for _ in range(5)]
            for v in variants:
                T1, B1 = run_level(eng, v, src, 1.0, SETTINGS["K256"])
                T0, B0 = run_level(eng, v, src, 0.0, SETTINGS["K256"])
                r = routes_for_source(B1, B0, ids_, src, leak)
                flat = {path: (x["sign"], x["s"], x["c"]) for C, xs in r.items() for x in xs for path in [x["path"]]}
                runs_ += 1
                if base is None:
                    base = flat
                else:
                    ok_ = set(flat) == set(base) and all(flat[k][0] == base[k][0] and abs(flat[k][1] - base[k][1]) <= TOL and abs(flat[k][2] - base[k][2]) <= TOL for k in base)
                    diffs += (not ok_)
        check("determinism: 7 listing orders (as given, reversed, 5 shuffles) of every source's links give the same routes", "0 differences",
              "%d runs, %d differences" % (runs_, diffs), diffs == 0)

        # 11. why the seam never uses `PLN.Query kb term`: it seeds every sentence as a task and loses pairs (decision section 1 row 10)
        seeded = {}
        for sname, st_ in (("fork_defaults", SETTINGS["fork_defaults"]), ("petta_lib_defaults", SETTINGS["petta_lib_defaults"])):
            found = set()
            simfound = set()
            for src in sorted({l["from"] for l in ls}):
                app = premise(src, 1.0)
                beliefs = list(kb_all2) + [app]
                T, B = eng.derive(list(beliefs), list(beliefs), 1, st_["maxsteps"], st_["tq"], st_["bq"])
                for (t, tv, ev) in B:
                    if is_state(t) and t[1] != src and 0 in ev:
                        path = path_from_stamp(ids_, src, ev)
                        if path is not None and path[-1] == t[1] and len(set(path)) == len(path):
                            found.add(tuple(path))
                sim = fork_sim.Sim(rules="restricted", maxsteps=st_["maxsteps"], task_queue_size=st_["tq"], belief_queue_size=st_["bq"], variant="petta_lib")
                T2, B2 = sim.derive(list(beliefs), list(beliefs), st_["maxsteps"], st_["tq"], st_["bq"])
                for (t, tv, ev) in B2:
                    if is_state(t) and t[1] != src and 0 in ev:
                        path = path_from_stamp(ids_, src, ev)
                        if path is not None and path[-1] == t[1] and len(set(path)) == len(path):
                            simfound.add(tuple(path))
            seeded[sname] = (len(found), len(simfound))
        res["seeding_every_sentence_as_a_task"] = seeded
        check("Tasks = Beliefs (what `PLN.Query kb term` does) loses pairs on the real library; the simulator predicts the same count",
              "fewer than 100 of 100 (decision row 10: 66 at the PeTTa-lib defaults), simulator == library",
              "real library: %s; simulator: %s (pairs found of 100; order-keeping Unique)" % (
                  {k: v[0] for k, v in seeded.items()}, {k: v[1] for k, v in seeded.items()}),
              all(v[0] < 100 and v[0] == v[1] for v in seeded.values()))

        # 9b. premise handling: an empty stamp with c < 1 floods the queue; id 0 with c = 1.0 does not
        A = "DasatinibPlusQuercetin"
        kbq = build_kb(reachable_links(stacks["patient"], A), leak)
        app_bad = sent(("State", A), 1.0, 0.9, ())
        Tb, Bb = eng.derive([app_bad], list(kbq) + [app_bad], 1, 12, 256, 256)
        copies = sum(1 for b in Bb if b[0] == ("State", A))
        app_ok = premise(A, 1.0)
        To, Bo = eng.derive([app_ok], list(kbq) + [app_ok], 1, 1024, 256, 256)
        copies_ok = sum(1 for b in Bo if b[0] == ("State", A))
        check("premise with an empty stamp and c = 0.9 floods the queue (revised with itself); the id-0, c = 1.0 premise does not",
              "flood: many copies; id 0: 1 copy", "empty stamp: %d premise copies after 12 steps; id 0: %d copy, tasks left %d" % (copies, copies_ok, len(To)),
              copies > 50 and copies_ok == 1 and not To)

    json.dump(OUT, open(SCRATCH + "/results_c.json", "w"), indent=1, default=str)
    fails = [c for c in OUT["checks"] if not c["ok"]]
    print("\n%d checks, %d failed" % (len(OUT["checks"]), len(fails)))
    for c in fails:
        print("FAILED:", c["check"], "|", c["measured"])


if __name__ == "__main__":
    main()
