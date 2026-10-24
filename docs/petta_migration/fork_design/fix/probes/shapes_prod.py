"""The 145 recorded query shapes through the PRODUCTION path (core.pln_runner._runtime_run -> PettaRuntime), legacy chainer then pln chainer.

For every part of every shape: raw PeTTa strings from the runtime (`last_raw`), compared
  (a) with the raw strings of the kit's own run (run_shapes.py's shapes_report.json): the legacy chainer must be identical, string for string;
  (b) with the hyperon baseline, classified with the kit's own comparator (identical | number_format_only | ordering_only | within_tolerance | different | error):
      class counts per chainer, so 'classes unchanged' can be read off for legacy, and what the pln chainer changes can be read off for pln.
usage: python shapes_prod.py <kit shapes_report.json> <out.json>"""
import sys, json, time, os, tempfile, pathlib
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
REPO = pathlib.Path(__file__).resolve().parents[5]
SPIKE = REPO / "scripts" / "petta_spike"
sys.path.insert(0, str(REPO / "pln_chat")); sys.path.insert(0, str(SPIKE))
os.environ.setdefault("KIT_ALLOW_PETTA", "1")
import config
import kit_common as K
from core import pln_runner
from core.metta_runtime import get_runtime

baseline = json.loads((SPIKE / "baseline_hyperon.json").read_text())
kit = {r["id"]: r for r in json.loads(pathlib.Path(sys.argv[1]).read_text())["results"]}
config.PLN_RUNTIME = "petta"
config.LOGS_DIR = pathlib.Path(tempfile.mkdtemp(prefix="shapes_logs_"))
out = {"legacy": {}, "pln": {}}
raws = {"legacy": {}, "pln": {}}
t0 = time.time()
for mode in ("legacy", "pln"):
    config.PLN_CHAINER = mode
    counts, raw_mismatch, errors, n = {}, [], [], 0
    for sh in baseline["shapes"]:
        if sh.get("skipped"):
            continue
        for pi, part in enumerate(sh["parts"]):
            exprs = part["expressions"]
            if not exprs:
                continue
            kb_files = [(REPO / f) if not pathlib.Path(f).is_absolute() else pathlib.Path(f) for f in part["kb_files"]]
            res = pln_runner._runtime_run("\n".join(exprs), 0.0, kb_files, part["extra_atoms"] or None)
            hyp = part["hyperon"]
            base_groups = hyp.get("groups") or []
            have_base = hyp.get("status") in ("ok", "empty")
            if res.status == "error":
                raw = None
                errors.append({"id": sh["id"], "part": pi, "error": (res.error or "")[:160], "code": res.error_code})
            else:
                raw = get_runtime("petta").last_raw
            kit_part = (kit.get(sh["id"]) or {}).get("parts", [None] * (pi + 1))[pi]
            for ei, e in enumerate(exprs):
                n += 1
                got = [] if raw is None or ei >= len(raw) else raw[ei]
                raws[mode][(sh["id"], pi, ei)] = None if raw is None else got
                if raw is None:
                    cls = "error"
                elif not have_base:
                    cls = "no_baseline"
                else:
                    cls = K.compare_groups(base_groups[ei] if ei < len(base_groups) else [], got)["class"]
                counts[cls] = counts.get(cls, 0) + 1
                if kit_part and ei < len(kit_part["exprs"]) and raw is not None:
                    kraw = kit_part["exprs"][ei].get("petta_raw")
                    if kraw is not None and kraw != got:
                        raw_mismatch.append({"id": sh["id"], "part": pi, "expr": e[:80]})
    out[mode] = {"expressions": n, "classes": dict(sorted(counts.items())), "raw_mismatch_vs_kit": len(raw_mismatch), "mismatches": raw_mismatch[:15],
                 "runner_errors": errors[:15], "n_runner_errors": len(errors), "seconds": round(time.time() - t0, 1)}
    print(mode, json.dumps(out[mode])[:700], flush=True)
# the pln chainer against the legacy chainer ON THE SAME RUNTIME, expression by expression, with the kit's comparator
cmp_counts, details = {}, []
for key, lg in raws["legacy"].items():
    pl = raws["pln"].get(key)
    if lg is None or pl is None:
        cls = "error_in_one" if (lg is None) != (pl is None) else "error_in_both"
    else:
        c = K.compare_groups(lg, pl)
        cls = c["class"]
        if cls not in ("identical", "within_tolerance", "number_format_only", "ordering_only"):
            details.append({"id": key[0], "part": key[1], "expr": key[2], "class": cls, "detail": str(c.get("detail", ""))[:300]})
    cmp_counts[cls] = cmp_counts.get(cls, 0) + 1
out["pln_vs_legacy_same_runtime"] = {"classes": dict(sorted(cmp_counts.items())), "other": details[:20]}
print("pln vs legacy", json.dumps(out["pln_vs_legacy_same_runtime"])[:1500], flush=True)
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
