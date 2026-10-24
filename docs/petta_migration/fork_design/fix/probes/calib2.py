import sys, json, time
from calib_lib import *
from core.metta_runtime import PettaRuntime
CASES = [("chain", 30), ("chain", 60), ("chain", 90), ("chain", 120), ("chain", 150), ("negchain", 100),
         ("layered", 4, 5, 3), ("layered", 5, 6, 3), ("layered", 6, 6, 3), ("layered", 7, 6, 3), ("layered", 8, 6, 3), ("layered", 6, 7, 3), ("layered", 5, 8, 3),
         ("digraph", 4), ("ring_chords", 8, 6), ("ring_chords", 10, 9), ("ring_chords", 10, 12), ("ring_chords", 12, 12), ("ring_chords", 12, 16, 5), ("ring_chords", 14, 18, 7)]
rt = PettaRuntime(); sess = rt.session(pc.library_text(), 900)
ev = lambda e: sess.run("!" + e, 900)[0]
leak = pc.probe_library(ev); out = []
for case in CASES:
    links, src = graph(*case)
    L = [pc.Link(a, b, sg, s, c) for a, b, sg, s, c in links]; pc.assign_ids(L)
    sub = pc.reachable_links(L, src)
    S, tl = states(sub, src, 20000)
    if S > 20000: print(case, "states >20000 skipped", flush=True); continue
    K = 1
    while K < 2 * (len(sub) + 2 * S + 8): K *= 2
    enc = pc._Encoding(L, leak)
    t0 = time.perf_counter()
    t1, b1, _ = pc._run(ev, enc, sub, src, 1.0, 4 * K, K, K, False)
    dt = time.perf_counter() - t0
    lens = [len([i for i in ids if i]) for (_, _, _, ids) in b1]
    r = dict(case=case, L=len(sub), S=S, avg_len=round(tl / max(S, 1), 2), K=K, B=len(b1), seconds=round(dt, 3), complete=(not t1 and len(b1) < K - 1))
    out.append(r); print(json.dumps(r), flush=True)
sess.close(); json.dump(out, open(FIX / "out" / "calib2.json", "w"), indent=1)
