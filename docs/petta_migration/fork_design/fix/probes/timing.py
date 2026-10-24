"""Request cost under the legacy and the pln chainer on the same PeTTa runtime, fresh child per request, median of 9."""
import statistics, json
from fx import *
import api
STACK = pln_runner.patient_stack(api._runtime_kb_paths())
SHAPES = {
 "no infer (+ 1 2)": ["!(+ 1 2)"],
 "one source: infer Spermidine CHD": ["!(infer &self Spermidine CoronaryHeartDisease)"],
 "one source: explain D+Q $x": ["!(explain &self DasatinibPlusQuercetin $x)"],
 "every source: infer $a CHD": ["!(infer &self $a CoronaryHeartDisease)"],
 "consumer: rank-interventions-for-patient": ["!(rank-interventions-for-patient &self Patient001 (DasatinibPlusQuercetin Fisetin Metformin Berberine Spermidine) CoronaryHeartDisease)"],
}
out = {}
for name, q in SHAPES.items():
    row = {}
    for mode in ("legacy", "pln"):
        use(mode, PLN_QUERY_TIMEOUT_SECONDS=60)
        run(STACK, None, q)                      # warm the page cache / imports
        ts = []
        for _ in range(9):
            r, dt = run(STACK, None, q)
            assert r.status in ("ok", "empty"), r.error
            ts.append(dt)
        row[mode] = round(statistics.median(ts), 3)
    row["extra_s"] = round(row["pln"] - row["legacy"], 3)
    row["ratio"] = round(row["pln"] / row["legacy"], 2)
    out[name] = row
    print(name, row, flush=True)
json.dump(out, open(FIX / "out" / "timing.json", "w"), indent=1)
