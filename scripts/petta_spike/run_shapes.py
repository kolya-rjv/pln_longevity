#!/usr/bin/env python
"""Run every shape of baseline_hyperon.json on PeTTa and diff against hyperon.     (PeTTa: NOT run by the kit's author)

    python run_shapes.py                       # all shapes, Phase 2 fixes applied, batch per KB
    python run_shapes.py --fixes none          # the as-is KB: which shapes are blocked at load
    python run_shapes.py --mode isolate        # one process per shape: each cost includes import + consult + load
    python run_shapes.py --engine hyperon      # self-test of this script on the same engine: all identical

For every part (one KB load + its `!` expressions) the KB text is rebuilt from the baseline's recorded file list and
extra atoms (and checked against the recorded sha256, so a changed work tree is noticed), loaded in a FRESH child
process, and each expression is run with its own helper call. Answers are compared per expression with hyperon's:

    identical | number_format_only | within_tolerance | ordering_only | different | error | no_baseline

  number_format_only  the same values spelled differently: True/true, 3 vs 3.0, 1e-5 vs 1.0e-5, $x vs $_1
  within_tolerance    floats equal to 1e-9 relative / 1e-12 absolute but not bit-equal
  ordering_only       the same answers in another order (as a list, or inside one tuple answer)
  error               PeTTa raised, timed out, crashed, or the KB did not load; the first error text is recorded
  no_baseline         hyperon had no answer (abort, timeout, error): PeTTa's answer is recorded for the record

Outputs (in out/, or $KIT_OUT): shapes_report.json, breakage_table.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import kit_common as K
import petta_adapter as A
import shapes as SH
from kb_fixes import parse_fix_spec


def _task_groups(baseline: dict, only: Optional[str], sources: Optional[set]):
    """Tasks = (shape record, part index). Grouped by the KB they load."""
    tasks = []
    for sh in baseline["shapes"]:
        if sh.get("skipped"):
            continue
        if only and not re.search(only, sh["id"]):
            continue
        if sources and sh["source"] not in sources:
            continue
        for pi, part in enumerate(sh["parts"]):
            tasks.append((sh, pi))
    groups: dict = {}
    for sh, pi in tasks:
        part = sh["parts"][pi]
        key = (tuple(part["kb_files"]), part["extra_atoms"])
        groups.setdefault(key, []).append((sh, pi))
    return tasks, groups


def _run_group(members, fixes, load_mode, a) -> dict:
    """One KB load for several (shape, part) tasks; returns {'report', 'index'} with the expression index map."""
    sh0, pi0 = members[0]
    p0 = sh0["parts"][pi0]
    part = SH.Part(p0["stack"], p0["kb_files"], p0["extra_atoms"], [])
    text = part.kb_text(fixes=fixes)
    drift = K.sha256_text(part.kb_text()) != p0["kb_sha256"]
    exprs, index = [], []
    for sh, pi in members:
        for ei, e in enumerate(sh["parts"][pi]["expressions"]):
            index.append((sh["id"], pi, ei, len(exprs)))
            exprs.append(e)
    rep = A.run_program_report(text, exprs, engine=a.engine, mode=load_mode, deadline_s=a.deadline, expr_deadline_s=a.expr_deadline,
                               load_deadline_s=a.load_deadline, stack_limit_mb=a.stack_limit_mb, strict_verify=a.strict_verify,
                               label=f"{p0['stack']}[{len(members)} shapes]")
    return {"report": rep, "index": index, "drift": drift, "kb_bytes": len(text.encode("utf-8"))}


def _classify_part(sh: dict, pi: int, run: dict, exprs_offset: dict) -> dict:
    """Compare one part's expressions with hyperon's. `run` is _run_group's result."""
    rep: A.RunReport = run["report"]
    part = sh["parts"][pi]
    hyp = part["hyperon"]
    out = {"stack": part["stack"], "petta_status": rep.status, "kb_drift": run["drift"], "exprs": [], "class": "identical"}
    completed = set(rep.completed)
    classes = []
    base_groups = hyp.get("groups") or []
    have_base = hyp.get("status") in ("ok", "empty")
    for ei, expr in enumerate(part["expressions"]):
        gi = exprs_offset[(sh["id"], pi, ei)]
        base = base_groups[ei] if ei < len(base_groups) else []
        answers = rep.answers[gi] if gi < len(rep.answers) else []
        err = rep.expr_errors[gi] if gi < len(rep.expr_errors) else None
        e = {"expr": expr, "hyperon": base, "petta_raw": [x.raw for x in answers], "petta_norm": [x.norm for x in answers]}
        if gi < len(rep.expr_seconds) and gi in completed:
            e["petta_ms"] = round(rep.expr_seconds[gi] * 1000, 2)
        hms = (hyp.get("timing") or {}).get("per_expr_ms") or []
        if ei < len(hms):
            e["hyperon_ms"] = hms[ei]
        if gi not in completed:
            e["class"] = "error"
            e["detail"] = f"{rep.status}: {rep.first_error() or rep.error}"[:700]
        elif err:
            e["class"] = "error"
            e["detail"] = f"{err['type']}: {err['message']}"[:600]
        elif not have_base:
            e["class"] = "no_baseline"
            e["detail"] = f"hyperon {hyp.get('status')}: {str(hyp.get('error'))[:200]}; PeTTa answered {len(answers)}"
        else:
            # the class describes PeTTa's RAW strings against hyperon's; norm_equal says whether the adapter's normaliser hides the difference
            cmp_ = K.compare_groups(base, e["petta_raw"])
            e["class"], e["detail"], e["tags"], e["max_rel"] = cmp_["class"], cmp_.get("detail", ""), cmp_["tags"], cmp_["max_rel"]
            e["norm_equal"] = (e["petta_norm"] == base)
        classes.append(e["class"])
        out["exprs"].append(e)
    if not classes:
        out["class"] = "identical"
    elif all(c == "no_baseline" for c in classes):
        out["class"] = "no_baseline"
    elif "error" in classes:
        out["class"] = "error"
    else:
        out["class"] = K.worst_class(c for c in classes if c != "no_baseline")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--engine", choices=["petta", "hyperon"], default="petta")
    ap.add_argument("--mode", choices=["batch", "isolate", "auto"], default="auto",
                    help="batch: one child per distinct KB; isolate: one child per shape; auto: batch, then isolate the shapes of a batch that timed out or crashed")
    ap.add_argument("--load-mode", choices=["string", "cached", "file"], default="string")
    ap.add_argument("--fixes", default="all", help="Phase 2 fixes applied to the KB text: all | none | superpose_empty,stored_calls,nested_conj")
    ap.add_argument("--only", help="regex on shape ids")
    ap.add_argument("--source", help="comma list of sources (smoke,few_shot,family,rule_form,route,canary,probe_print,probe_order)")
    ap.add_argument("--deadline", type=float, default=600.0, help="whole child, seconds")
    ap.add_argument("--load-deadline", type=float, default=300.0)
    ap.add_argument("--expr-deadline", type=float, default=120.0)
    ap.add_argument("--stack-limit-mb", type=int, default=A.DEFAULT_STACK_LIMIT_MB)
    ap.add_argument("--strict-verify", action="store_true", help="treat a head with fewer clauses than equations as a failed load")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default out/)")
    a = ap.parse_args(argv)

    base_path = a.baseline or K.baseline_path()
    if not base_path.exists():
        print(f"no baseline at {base_path}: run record_baseline.py first", file=sys.stderr)
        return 2
    baseline = K.json_load(base_path)
    fixes = parse_fix_spec(a.fixes)
    sources = set(a.source.split(",")) if a.source else None
    tasks, groups = _task_groups(baseline, a.only, sources)
    out_dir = a.out or K.out_dir()
    print(f"{len(tasks)} shape parts in {len(groups)} KB groups; engine={a.engine} mode={a.mode} fixes={sorted(fixes)} petta_home={K.petta_home()}", flush=True)

    results_by_part: dict = {}
    t_all = time.perf_counter()
    group_items = list(groups.items())
    batch_info: list = []

    def do_group(item) -> None:
        key, members = item
        isolate_all = a.mode == "isolate"
        run_members = [members] if not isolate_all else [[m] for m in members]
        for chunk in run_members:
            run = _run_group(chunk, fixes, a.load_mode, a)
            rep = run["report"]
            offs = {(sid, pi, ei): gi for (sid, pi, ei, gi) in run["index"]}
            died = rep.status in ("timeout", "crash")
            loaded_ok = any(st.get("stage") == "loaded" and st.get("ok", True) for st in rep.stages)    # a KB that never loaded dies the same way alone
            if died and loaded_ok and a.mode == "auto" and len(chunk) > 1:
                batch_info.append({"stack": chunk[0][0]["parts"][chunk[0][1]]["stack"], "n_shapes": len(chunk), "status": rep.status,
                                   "scope": rep.timeout_scope, "note": "batch died; shapes re-run one per process"})
                print(f"  batch of {len(chunk)} on {chunk[0][0]['parts'][chunk[0][1]]['stack']} ended {rep.status} ({rep.timeout_scope}); isolating", flush=True)
                for m in chunk:
                    r1 = _run_group([m], fixes, a.load_mode, a)
                    offs1 = {(sid, pi, ei): gi for (sid, pi, ei, gi) in r1["index"]}
                    results_by_part[(m[0]["id"], m[1])] = (_classify_part(m[0], m[1], r1, offs1), r1["report"], True)
                continue
            for sh, pi in chunk:
                results_by_part[(sh["id"], pi)] = (_classify_part(sh, pi, run, offs), rep, len(chunk) == 1)
            first = chunk[0]
            print(f"  {first[0]['parts'][first[1]]['stack']:15s} {len(chunk):3d} shape(s)  {rep.status:12s} import {rep.timing.get('import_s', '-')}s "
                  f"consult {rep.timing.get('consult_s', '-')}s load {rep.timing.get('load_s', '-')}s  rss {rep.rss_mb.get('end') or rep.rss_mb.get('load')} MB"
                  f"{'  ' + str(rep.first_error())[:160] if rep.status != 'ok' else ''}", flush=True)

    if a.jobs <= 1:
        for it in group_items:
            do_group(it)
    else:
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(do_group, group_items))

    # assemble per-shape results
    results, counts = [], {}
    for sh in baseline["shapes"]:
        if sh.get("skipped") or not any((sh["id"], pi) in results_by_part for pi in range(len(sh["parts"]))):
            continue
        parts_out = []
        for pi in range(len(sh["parts"])):
            if (sh["id"], pi) not in results_by_part:
                continue
            cls, rep, isolated = results_by_part[(sh["id"], pi)]
            cls["load"] = {k: rep.load.get(k) for k in ("mode", "seconds", "failed_silently", "exception", "verify", "cache_hit_guess")}
            cls["timing"] = dict(rep.timing)
            cls["isolated_process"] = bool(isolated)
            cls["first_error"] = rep.first_error()
            cls["stdout"] = rep.stdout[-800:] if rep.stdout.strip() else ""
            cls["stderr"] = rep.stderr[-800:] if rep.stderr.strip() else ""
            parts_out.append(cls)
        klass = K.worst_class(p["class"] if p["class"] != "no_baseline" else "identical" for p in parts_out)
        if parts_out and all(p["class"] == "no_baseline" for p in parts_out):
            klass = "no_baseline"
        elif any(p["class"] == "error" for p in parts_out):
            klass = "error"
        first_issue = next((e for p in parts_out for e in p["exprs"] if e["class"] not in ("identical",)), None)
        results.append({"id": sh["id"], "source": sh["source"], "route": sh["route"], "nl": sh["nl"][:160], "class": klass,
                        "first_issue": ({"expr": first_issue["expr"][:200], "class": first_issue["class"], "detail": first_issue.get("detail", "")[:500]}
                                        if first_issue else None),
                        "parts": parts_out})
        counts[klass] = counts.get(klass, 0) + 1

    doc = {
        "meta": {"kit_version": K.KIT_VERSION, "ran": K.now_iso(), "engine": a.engine, "mode": a.mode, "load_mode": a.load_mode,
                 "fixes": sorted(fixes), "baseline": str(base_path), "baseline_git_head": baseline["meta"].get("git_head"),
                 "petta_home": str(K.petta_home()), "elapsed_s": round(time.perf_counter() - t_all, 1), "env": K.env_report(),
                 "batches_isolated": batch_info},
        "counts": counts,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    K.json_dump(doc, out_dir / "shapes_report.json")
    (out_dir / "breakage_table.md").write_text(render_markdown(doc), encoding="utf-8")
    print(json.dumps(counts))
    print(f"wrote {out_dir / 'shapes_report.json'} and {out_dir / 'breakage_table.md'}")
    return 0 if not counts.get("error") else 1


def render_markdown(doc: dict) -> str:
    m, counts, results = doc["meta"], doc["counts"], doc["results"]
    lines = [f"# PeTTa vs hyperon: breakage table",
             "",
             f"engine `{m['engine']}`, load mode `{m['load_mode']}`, run mode `{m['mode']}`, fixes `{', '.join(m['fixes']) or 'none'}`, "
             f"baseline @ `{m.get('baseline_git_head')}`, {len(results)} shapes, {m['elapsed_s']} s.",
             "",
             "| class | shapes |", "|---|---|"]
    for c in K.CLASS_ORDER + ["no_baseline"]:
        if c in counts:
            lines.append(f"| {c} | {counts[c]} |")
    by_src: dict = {}
    for r in results:
        by_src.setdefault(r["source"], {}).setdefault(r["class"], 0)
        by_src[r["source"]][r["class"]] += 1
    lines += ["", "## By source", "", "| source | " + " | ".join(c for c in K.CLASS_ORDER + ["no_baseline"] if c in counts) + " |",
              "|---|" + "---|" * len([c for c in K.CLASS_ORDER + ['no_baseline'] if c in counts])]
    for src, cs in by_src.items():
        lines.append(f"| {src} | " + " | ".join(str(cs.get(c, 0)) for c in K.CLASS_ORDER + ["no_baseline"] if c in counts) + " |")
    lines += ["", "## Not identical", "", "| shape | route | class | what | PeTTa error text |", "|---|---|---|---|---|"]
    for r in results:
        if r["class"] == "identical":
            continue
        fi = r["first_issue"] or {}
        err = next((p["first_error"] for p in r["parts"] if p.get("first_error")), "") or ""
        what = (fi.get("detail") or "").replace("|", "\\|").replace("\n", " ")[:260]
        lines.append(f"| `{r['id']}` | {r['route']} | {r['class']} | {what} | {str(err).replace('|', chr(92) + '|').replace(chr(10), ' ')[:300]} |")
    lines += ["", "## Latency (PeTTa per expression vs hyperon per expression, ms; load not included)", "",
              "| shape | expressions | hyperon ms | PeTTa ms | ratio |", "|---|---|---|---|---|"]
    rows = []
    for r in results:
        h = sum(e.get("hyperon_ms", 0) or 0 for p in r["parts"] for e in p["exprs"])
        pt = sum(e.get("petta_ms", 0) or 0 for p in r["parts"] for e in p["exprs"])
        n = sum(len(p["exprs"]) for p in r["parts"])
        if pt and h:
            rows.append((pt / h, r["id"], n, h, pt))
    for ratio, sid, n, h, pt in sorted(rows, reverse=True)[:25]:
        lines.append(f"| `{sid}` | {n} | {h:.1f} | {pt:.1f} | {ratio:.1f}x |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A.PettaNotAllowed as exc:
        print(f"not started: {exc}", file=sys.stderr)
        raise SystemExit(3)
