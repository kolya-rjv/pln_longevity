#!/usr/bin/env python
"""Load every stack on PeTTa, as a string and through the cached file, and run the smoke queries.   (PeTTa: NOT run by the kit's author)

    python run_loads.py                         # smoke10 + the six stacks, as-is and with the Phase 2 fixes, string + cached x2
    python run_loads.py --stacks smoke10        # the quickest first look: does anything load at all?
    python run_loads.py --stacks all            # also the full shared stack, the NHANES stack and the fork's src/
    python run_loads.py --modes string --variants fixed

For each (stack, variant, mode) one FRESH child process loads the KB text and then runs the stack's canary queries and,
where the stack has the files for them, the five smoke queries of pln_deduction.metta section 5:

  variant  asis   the KB text exactly as pln_runner would load it       -> the FIRST ERROR PER STACK
           fixed  with the three Phase 2 fixes applied (kb_fixes.py)    -> do the smoke answers appear after only those?
  mode     string        process_metta_string(kb_text)
           cached_miss   KB written to out/kb_files/kb_<sha>.metta, load_metta_file_cached, snapshot directory emptied first
           cached_hit    the same again: the snapshot is replayed (cache_hit_guess says whether it was)

Recorded per run: status, the first error text, import / consult / load seconds, RSS, load verification
(heads missing or short), the smoke verdicts against the expected answers WRITTEN IN THE FILE (numeric tolerance
rel 1e-6), the canary answers against hyperon's, and the hyperon load time next to it.

Outputs (out/ or $KIT_OUT): loads.json, loads.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import kit_common as K
import petta_adapter as A
import shapes as SH
import stacks as S
from kb_fixes import parse_fix_spec

SMOKE_NEEDS = {"pln_deduction.metta", "mechanistic_bridges.metta", "grim_age_lu2019_evidence.metta"}


def _runs_plan(stack_names: list[str], variants: list[str], modes: list[str]):
    for st in stack_names:
        for var in variants:
            for mode in modes:
                yield st, var, mode


def one_run(spec: S.StackSpec, variant: str, mode: str, fixes: set, smoke: list[dict], baseline: Optional[dict], a) -> dict:
    part = SH.Part(spec.name, spec.files, spec.extra_atoms, [])
    text = part.kb_text(fixes=fixes if variant == "fixed" else None)
    smoke_here = bool(SMOKE_NEEDS <= {Path(f).name for f in spec.files}) or a.smoke_everywhere
    exprs = list(spec.canary) + ([q["expr"] for q in smoke] if smoke_here and spec.name != "smoke10" else [])
    if spec.name == "smoke10":
        exprs = [q["expr"] for q in smoke]
    cache_dir = K.out_dir() / "petta_load_cache" / f"{spec.name}_{variant}"
    eng_mode = "string"
    if mode.startswith("cached"):
        eng_mode = "cached"
        if mode == "cached_miss" and cache_dir.exists():
            shutil.rmtree(cache_dir)
    t = time.perf_counter()
    rep = A.run_program_report(text, exprs, engine=a.engine, mode=eng_mode, deadline_s=a.deadline, load_deadline_s=a.deadline,
                               expr_deadline_s=a.expr_deadline, stack_limit_mb=a.stack_limit_mb, strict_verify=a.strict_verify,
                               load_cache=str(cache_dir), label=f"{spec.name}/{variant}/{mode}")
    run = {
        "stack": spec.name, "variant": variant, "mode": mode, "engine": a.engine, "n_files": len(spec.files),
        "kb_bytes": rep.kb_bytes, "kb_sha256": rep.kb_sha256[:16], "status": rep.status, "first_error": rep.first_error(),
        "exception": (rep.load.get("exception") or {}),
        "import_s": rep.timing.get("import_s"), "consult_s": rep.timing.get("consult_s"), "load_s": rep.timing.get("load_s"),
        "queries_s": rep.timing.get("queries_s"), "wall_s": rep.timing.get("wall_s"), "cpu_s": rep.timing.get("cpu_s"),
        "rss_import_mb": rep.rss_mb.get("import_"), "rss_consult_mb": rep.rss_mb.get("consult"), "rss_load_mb": rep.rss_mb.get("load"),
        "rss_end_mb": rep.rss_mb.get("end"), "peak_rss_mb": rep.peak_rss_mb,
        "failed_silently": rep.load.get("failed_silently"), "cache_hit_guess": rep.load.get("cache_hit_guess"),
        "verify": {k: v for k, v in (rep.load.get("verify") or {}).items() if k in ("checked", "defined", "n_missing", "n_short", "missing", "short", "error")},
        "stdout_tail": rep.stdout[-500:], "stderr_tail": rep.stderr[-500:], "exitcode": rep.exitcode, "signal": rep.signal,
        "env": rep.env,
    }
    # smoke verdicts
    if smoke_here and rep.status == "ok":
        offset = 0 if spec.name == "smoke10" else len(spec.canary)
        verdicts = []
        for i, q in enumerate(smoke):
            gi = offset + i
            ans = rep.answers[gi] if gi < len(rep.answers) else []
            err = rep.expr_errors[gi] if gi < len(rep.expr_errors) else None
            if err:
                verdicts.append({"expr": q["expr"], "ok": False, "why": f"{err['type']}: {err['message'][:300]}"})
            else:
                m = K.matches_expected(q["expected"], [x.norm for x in ans])
                verdicts.append({"expr": q["expr"], "expected": q["expected"], "ok": m["ok"], "why": m["why"],
                                 "got": [x.raw for x in ans][:2]})
        run["smoke"] = verdicts
        run["smoke_ok"] = all(v["ok"] for v in verdicts)
    else:
        run["smoke_ok"] = None
    # canaries against hyperon's recorded answers
    if spec.canary and rep.status == "ok" and spec.name != "smoke10":
        sh = next((x for x in (baseline or {}).get("shapes", []) if x["id"] == f"canary:{spec.name}"), None)
        hyp = sh["parts"][0]["hyperon"] if sh and sh["parts"] else None
        cls = []
        for i in range(len(spec.canary)):
            ans = rep.answers[i] if i < len(rep.answers) else []
            err = rep.expr_errors[i] if i < len(rep.expr_errors) else None
            if err:
                cls.append("error")
            elif hyp is None:
                cls.append("no_baseline")
            elif hyp.get("status") not in ("ok", "empty"):
                cls.append(f"hyperon {hyp.get('status')}")
            else:
                cls.append(K.compare_groups(hyp["groups"][i], [x.raw for x in ans])["class"])
        run["canary_classes"] = cls
        run["canary_answers"] = [[x.raw[:200] for x in g][:2] for g in rep.answers[:len(spec.canary)]]
    run["expr_errors"] = [{"i": i, **e} for i, e in enumerate(rep.expr_errors) if e][:8]
    if baseline and spec.name in baseline.get("stacks", {}):
        h = baseline["stacks"][spec.name]["hyperon"]
        run["hyperon_load_ms"] = h.get("load_ms")
        run["hyperon_rss_mb"] = h.get("rss_mb")
    return run


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["petta", "hyperon"], default="petta")
    ap.add_argument("--stacks", default="default", help="comma list, 'default' (smoke10 + six stacks) or 'all'")
    ap.add_argument("--variants", default="asis,fixed")
    ap.add_argument("--modes", default="string,cached_miss,cached_hit")
    ap.add_argument("--fixes", default="all", help="what the 'fixed' variant applies")
    ap.add_argument("--stack-source", choices=["import", "parse"], default="import")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--deadline", type=float, default=900.0)
    ap.add_argument("--expr-deadline", type=float, default=180.0)
    ap.add_argument("--stack-limit-mb", type=int, default=A.DEFAULT_STACK_LIMIT_MB)
    ap.add_argument("--strict-verify", action="store_true")
    ap.add_argument("--smoke-everywhere", action="store_true", help="run the smoke queries on every stack, not only those holding their rules")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    fixes = parse_fix_spec(a.fixes)
    all_stacks = S.list_stacks(a.stack_source)
    if a.stacks == "default":
        names = [n for n in ["smoke10"] + S.CORE_STACKS if n in all_stacks]
    elif a.stacks == "all":
        names = list(all_stacks)
    else:
        names = [n.strip() for n in a.stacks.split(",")]
        missing = [n for n in names if n not in all_stacks]
        if missing:
            print(f"unknown or unavailable stacks {missing}; have {sorted(all_stacks)}", file=sys.stderr)
            return 2
    variants, modes = a.variants.split(","), a.modes.split(",")
    base_path = a.baseline or K.baseline_path()
    baseline = K.json_load(base_path) if base_path.exists() else None
    _, smoke = S.parse_smoke()
    plan = list(_runs_plan(names, variants, modes))
    print(f"{len(plan)} runs ({len(names)} stacks x {variants} x {modes}); engine={a.engine}; petta_home={K.petta_home()}; fixes for 'fixed': {sorted(fixes)}", flush=True)
    env = K.env_report()

    runs: list = [None] * len(plan)

    def work(i: int) -> None:
        st, var, mode = plan[i]
        # a cached_hit needs its cached_miss to have run first: keep them in plan order inside one worker
        runs[i] = one_run(all_stacks[st], var, mode, fixes, smoke, baseline, a)
        r = runs[i]
        print(f"[{i + 1:2d}/{len(plan)}] {st:15s} {var:5s} {mode:11s} {r['status']:12s} import {r['import_s']} consult {r['consult_s']} load {r['load_s']} s  "
              f"rss {r['rss_load_mb']} MB  smoke {r['smoke_ok']}  {str(r['first_error'] or '')[:140]}", flush=True)

    t0 = time.perf_counter()
    if a.jobs <= 1:
        for i in range(len(plan)):
            work(i)
    else:
        # runs of one (stack, variant) must stay ordered (miss, then hit); parallelise across stacks
        by_key: dict = {}
        for i, (st, var, mode) in enumerate(plan):
            by_key.setdefault((st, var), []).append(i)
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(lambda idxs: [work(i) for i in idxs], by_key.values()))

    out_dir = a.out or K.out_dir()
    doc = {"meta": {"kit_version": K.KIT_VERSION, "ran": K.now_iso(), "engine": a.engine, "petta_home": str(K.petta_home()), "fixes": sorted(fixes),
                    "elapsed_s": round(time.perf_counter() - t0, 1), "env": env, "stack_source": a.stack_source},
           "runs": [r for r in runs if r], "gate": gate(runs)}
    K.json_dump(doc, out_dir / "loads.json")
    (out_dir / "loads.md").write_text(render_markdown(doc), encoding="utf-8")
    print(json.dumps(doc["gate"], indent=1))
    print(f"wrote {out_dir / 'loads.json'} and {out_dir / 'loads.md'}")
    return 0


def gate(runs: list) -> dict:
    """The Phase 1 questions that loads can answer."""
    runs = [r for r in runs if r]

    def find(stack, variant, mode="string"):
        return next((r for r in runs if r["stack"] == stack and r["variant"] == variant and r["mode"] == mode), None)

    g: dict = {}
    r = find("smoke10", "asis") or find("smoke10", "fixed")
    g["janus_backed_petta_ran"] = None if r is None else r["status"] not in ("import_error", "consult_error", "crash", "timeout")
    rf = find("smoke10", "fixed")
    g["smoke_answers_after_phase2_fixes_only"] = None if rf is None else bool(rf.get("smoke_ok"))
    g["stacks_loading_as_is"] = sorted({r["stack"] for r in runs if r["variant"] == "asis" and r["mode"] == "string" and r["status"] == "ok"})
    g["stacks_loading_after_fixes"] = sorted({r["stack"] for r in runs if r["variant"] == "fixed" and r["mode"] == "string" and r["status"] == "ok"})
    g["stacks_failing_after_fixes"] = sorted({r["stack"] for r in runs if r["variant"] == "fixed" and r["mode"] == "string" and r["status"] != "ok"})
    loads = [r["load_s"] for r in runs if r["variant"] == "fixed" and r["mode"] == "string" and r["status"] == "ok" and r["load_s"] is not None]
    g["max_string_load_s"] = max(loads) if loads else None
    g["silent_partial_loads"] = sorted({f"{r['stack']}/{r['variant']}/{r['mode']}" for r in runs
                                         if r["status"] == "ok" and ((r.get("verify") or {}).get("n_missing") or (r.get("verify") or {}).get("n_short"))})
    hits = [r for r in runs if r["mode"] == "cached_hit" and r["status"] == "ok"]
    g["cached_replays_that_hit"] = [f"{r['stack']}/{r['variant']}" for r in hits if r.get("cache_hit_guess")]
    return g


def render_markdown(doc: dict) -> str:
    runs = doc["runs"]
    L = ["# PeTTa loads", "", f"engine `{doc['meta']['engine']}`, {len(runs)} runs, {doc['meta']['elapsed_s']} s. PeTTa @ `{doc['meta']['petta_home']}`", "",
         "| stack | variant | mode | status | import s | consult s | load s | RSS MB (load) | peak MB | missing/short heads | smoke | hyperon load ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        v = r.get("verify") or {}
        L.append(f"| {r['stack']} | {r['variant']} | {r['mode']} | {r['status']} | {r['import_s']} | {r['consult_s']} | {r['load_s']} | {r['rss_load_mb']} | "
                 f"{r['peak_rss_mb']} | {v.get('n_missing', '-')}/{v.get('n_short', '-')} | {r['smoke_ok']} | {r.get('hyperon_load_ms', '-')} |")
    L += ["", "## First error per stack (as-is, string load)", ""]
    for r in runs:
        if r["variant"] == "asis" and r["mode"] == "string":
            L.append(f"* **{r['stack']}**: {r['status']}" + (f" - {str(r['first_error'])[:400]}" if r["first_error"] else ""))
    L += ["", "## After the Phase 2 fixes (string load)", ""]
    for r in runs:
        if r["variant"] == "fixed" and r["mode"] == "string":
            L.append(f"* **{r['stack']}**: {r['status']}" + (f" - {str(r['first_error'])[:400]}" if r["first_error"] else ""))
    L += ["", "## Smoke queries", ""]
    for r in runs:
        if r.get("smoke") and r["mode"] == "string":
            L.append(f"* **{r['stack']} / {r['variant']}**: " + ("all five as written in pln_deduction.metta" if r["smoke_ok"] else "MISMATCH"))
            for v in r["smoke"]:
                if not v["ok"]:
                    L.append(f"    * `{v['expr']}` expected `{v.get('expected')}`: {v['why']} {v.get('got', '')}")
    L += ["", "## Gate", "", "```", json.dumps(doc["gate"], indent=1), "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A.PettaNotAllowed as exc:
        print(f"not started: {exc}", file=sys.stderr)
        raise SystemExit(3)
