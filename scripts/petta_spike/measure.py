#!/usr/bin/env python
"""Measure PeTTa next to hyperon: start-up, load, latency, printing, answer order, limits, fork, GIL.   (PeTTa: NOT run by the kit's author)

    python measure.py                          # every section
    python measure.py --sections consult,probes    # the cheap, informative first look
    python measure.py --quick                  # one repetition, the small stacks only

sections
  consult   consult time of src/main.pl (and helper.pl), import of janus_swi, first-call cost, in fresh processes
  loads     cold string load, cached-file miss, cached-file replay, per stack: seconds and RSS next to hyperon's
  latency   per-query latency on PeTTa (first call and warm) next to hyperon's, and the Phase 1 gate: no routed probe more
            than five times slower than on hyperon WITH THE LOAD INCLUDED (fresh process per query), and the same with the
            fork-from-a-warm-template cost when the fork section has run
  probes    how ints, floats, booleans, 1e-05, (superpose ()), (car-atom ()), log of 0 ... print, side by side with hyperon;
            and the answer order at the seven first-answer sites of 01_verdict.md section 2 item 7
  limits    a runaway query (deadline + SIGKILL), a stack overflow (catchable?), the halt(1) paths of assert/test, remove-atom
            retracting identical copies, a second load duplicating clauses
  fork      fork() after a child-only import of janus/petta: does a forked child answer? after the consult only, and after
            the KB is loaded (a warm template). Latency of fork + query. Does the template survive?
  gil       does janus.query_once hold the GIL during a long Prolog call? (canary thread ticks), and do two Python threads
            run Prolog in parallel?

Outputs (out/ or $KIT_OUT): measure.json, measure.md.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import kit_common as K
import petta_adapter as A
import shapes as SH
import stacks as S
from kb_fixes import parse_fix_spec

ALL_SECTIONS = ["consult", "loads", "latency", "probes", "limits", "fork", "gil"]


# ── children that need their own targets ─────────────────────────────────────

def _consult_child(conn, spec: dict) -> None:
    """import janus_swi, then consult main.pl and helper.pl separately (what PeTTa() does in one step), then the first calls."""
    def send(*m):
        try:
            conn.send(m)
        except Exception:
            pass
    out: dict = {}
    try:
        A.redirect_output(spec)
        A.apply_env(spec)
        home = Path(spec["petta_home"])
        t = time.perf_counter()
        import janus_swi as janus
        out["import_janus_s"] = round(time.perf_counter() - t, 4)
        out["rss_import_mb"] = K.rss_mb()
        send("stage", "imported", {})
        if spec.get("stack_limit_mb"):
            janus.query_once(f"set_prolog_flag(stack_limit, {int(spec['stack_limit_mb']) * 1_000_000})")
        t = time.perf_counter()
        janus.consult(str(home / "src" / "main.pl"))
        out["consult_main_pl_s"] = round(time.perf_counter() - t, 4)
        out["rss_main_mb"] = K.rss_mb()
        t = time.perf_counter()
        janus.consult(str(home / "python" / "helper.pl"))
        out["consult_helper_pl_s"] = round(time.perf_counter() - t, 4)
        send("stage", "consulted", {})
        send("stage", "loaded", {})
        for k, expr in enumerate(["!(+ 1 2)", "!(+ 1 2)", "!(collapse (superpose (1 2 3)))"]):
            t = time.perf_counter()
            res = A.helper_call(janus, "process_metta_string", expr)
            out[f"first_calls_s_{k}"] = round(time.perf_counter() - t, 5)
            out[f"first_calls_answer_{k}"] = res
        out["rss_end_mb"] = K.rss_mb()
        out["cpu_s"] = round(time.process_time(), 3)
        out["peak_rss_mb"] = K.peak_rss_mb()
        try:
            out["swipl_version"] = str(janus.query_once("current_prolog_flag(version, V)").get("V"))
            out["prolog_threads_running"] = str(janus.query_once("aggregate_all(count, thread_property(_, status(running)), N)").get("N"))
        except Exception:
            pass
        out["ok"] = True
    except BaseException as exc:
        out["ok"] = False
        out["error"] = A.exc_info(exc)
    A.finish_child(conn, out)


def _fork_child(conn, spec: dict) -> None:
    """A spawned template process: import janus/petta, consult, then fork() grandchildren that each answer one query."""
    def send(*m):
        try:
            conn.send(m)
        except Exception:
            pass
    out: dict = {}
    try:
        A.redirect_output(spec)
        A.apply_env(spec)
        janus, petta, info = A.bootstrap_petta(spec, send)
        out["bootstrap"] = {k: info.get(k) for k in ("import_s", "consult_s", "swipl_version")}

        def forks(label: str, expr: str, n: int, deadline: float) -> list:
            rows = []
            for k in range(n):
                r, w = os.pipe()
                t = time.perf_counter()
                pid = os.fork()
                if pid == 0:                                       # grandchild
                    try:
                        os.close(r)
                        tq = time.perf_counter()
                        res = A.helper_call(janus, "process_metta_string", expr)
                        os.write(w, json.dumps({"ok": True, "answers": res, "query_s": round(time.perf_counter() - tq, 5)}).encode())
                    except BaseException as e:
                        try:
                            os.write(w, json.dumps({"ok": False, "error": repr(e)[:300]}).encode())
                        except Exception:
                            pass
                    finally:
                        os._exit(0)
                os.close(w)
                row: dict = {"label": label, "k": k}
                ready, _, _ = select.select([r], [], [], deadline)
                if ready:
                    data = b""
                    while True:
                        chunk = os.read(r, 65536)
                        if not chunk:
                            break
                        data += chunk
                    row["fork_to_answer_s"] = round(time.perf_counter() - t, 5)
                    if data:
                        try:
                            row.update(json.loads(data.decode()))
                        except ValueError:
                            row["ok"] = False
                            row["error"] = "unparsable answer"
                    else:
                        row.update(ok=False, error="grandchild closed the pipe without answering (died)")
                    _, st = os.waitpid(pid, 0)
                    row["exit_status"] = st
                else:
                    row.update(ok=False, hang=True, error=f"no answer in {deadline}s: killed")
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                os.close(r)
                rows.append(row)
            return rows

        n = int(spec.get("n_forks", 5))
        dl = float(spec.get("fork_deadline_s", 20))
        out["after_consult"] = forks("after_consult", "!(+ 1 2)", n, dl)
        t = time.perf_counter()
        res = A.helper_call(janus, "process_metta_string", spec["kb_text"]) if spec["kb_text"].strip() else []
        out["template_load_s"] = round(time.perf_counter() - t, 4)
        out["rss_after_load_mb"] = K.rss_mb()
        send("stage", "loaded", {})
        out["after_load"] = []
        for expr in spec["expressions"]:
            send("stage", "expr_start", {"i": 1})
            out["after_load"] += forks("after_load", expr, n, dl)
        # is the template itself still healthy?
        try:
            out["template_after"] = A.helper_call(janus, "process_metta_string", "!(+ 1 2)")
        except BaseException as e:
            out["template_after"] = f"raised {type(e).__name__}: {str(e)[:200]}"
        out["ok"] = True
    except BaseException as exc:
        out["ok"] = False
        out["error"] = A.exc_info(exc)
    A.finish_child(conn, out)


def _gil_child(conn, spec: dict) -> None:
    """Canary-thread ticks while janus runs a long Prolog call, then two Python threads each running one."""
    def send(*m):
        try:
            conn.send(m)
        except Exception:
            pass
    out: dict = {}
    try:
        A.redirect_output(spec)
        A.apply_env(spec)
        janus, petta, info = A.bootstrap_petta(spec, send)
        send("stage", "loaded", {})
        n_loop = int(spec.get("loop", 30_000_000))
        ticks: list = []
        stop = threading.Event()

        def canary():
            while not stop.is_set():
                time.sleep(0.005)
                ticks.append(time.perf_counter())

        th = threading.Thread(target=canary, daemon=True)
        th.start()
        time.sleep(0.3)
        n0 = len(ticks)
        time.sleep(0.6)
        base_rate = (len(ticks) - n0) / 0.6
        out["canary_ticks_per_s_idle"] = round(base_rate, 1)

        def timed(label: str, fn) -> dict:
            n_before, t = len(ticks), time.perf_counter()
            fn()
            dt = time.perf_counter() - t
            got = len(ticks) - n_before
            exp = dt * base_rate
            return {"label": label, "seconds": round(dt, 3), "ticks": got, "expected_ticks": round(exp, 1),
                    "tick_ratio": round(got / exp, 3) if exp else None}

        goal = f"between(1, {n_loop}, X), X >= {n_loop}"
        out["prolog_loop"] = timed("pure Prolog loop via janus.query_once", lambda: janus.query_once(goal))
        kb = "(= (count $n) (if (> $n 0) (count (- $n 1)) done))"
        A.helper_call(janus, "process_metta_string", kb)
        out["metta_loop"] = timed("PeTTa (count N) via the helper", lambda: A.helper_call(janus, "process_metta_string", f"!(count {int(spec.get('count', 1_000_000))})"))
        # two threads
        errs: list = []

        def run_goal():
            try:
                janus.query_once(goal)
            except BaseException as e:
                errs.append(f"{type(e).__name__}: {str(e)[:200]}")

        t = time.perf_counter()
        run_goal()
        single = time.perf_counter() - t
        ts = [threading.Thread(target=run_goal) for _ in range(2)]
        t = time.perf_counter()
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        both = time.perf_counter() - t
        out["two_threads"] = {"single_s": round(single, 3), "two_threads_wall_s": round(both, 3),
                              "ratio": round(both / single, 2) if single else None, "errors": errs,
                              "reading": "~1.0 = the two Prolog calls ran in parallel (GIL released); ~2.0 = serialised"}
        stop.set()
        out["ok"] = True
    except BaseException as exc:
        out["ok"] = False
        out["error"] = A.exc_info(exc)
    A.finish_child(conn, out)


# ── sections ─────────────────────────────────────────────────────────────────

def _spec(extra: Optional[dict] = None, kb_text: str = "", expressions: Optional[list] = None, stack_limit_mb: Optional[int] = A.DEFAULT_STACK_LIMIT_MB) -> dict:
    return {"engine": "petta", "petta_home": str(K.petta_home()), "load_cache": str(K.load_cache_dir()), "pycache_prefix": str(K.out_dir() / "pycache"),
            "stack_limit_mb": stack_limit_mb, "kb_text": kb_text, "expressions": expressions or [], **(extra or {})}


def _stats(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs)} if xs else {"n": 0}


def section_consult(a) -> dict:
    runs = []
    for i in range(a.reps):
        sup = A.spawn_supervised(_consult_child, _spec(), total_deadline_s=a.deadline, load_deadline_s=a.deadline, expr_deadline_s=a.deadline)
        d = sup.done or {"ok": False, "error": {"message": f"child ended: exit {sup.exitcode} {sup.signal} timeout={sup.timeout_scope}"}}
        d["wall_s"] = sup.wall_s
        d["stderr_tail"] = sup.stderr[-400:]
        runs.append(d)
        print(f"  consult run {i + 1}/{a.reps}: ok={d.get('ok')} import {d.get('import_janus_s')} main.pl {d.get('consult_main_pl_s')} helper.pl {d.get('consult_helper_pl_s')} wall {d['wall_s']}", flush=True)
    good = [r for r in runs if r.get("ok")]
    return {"runs": runs, "summary": {k: _stats([r.get(k) for r in good]) for k in
                                      ("import_janus_s", "consult_main_pl_s", "consult_helper_pl_s", "first_calls_s_0", "first_calls_s_1", "first_calls_s_2",
                                       "rss_import_mb", "rss_main_mb", "rss_end_mb", "peak_rss_mb", "wall_s")},
            "hyperon_reference": "import hyperon 4 ms, MeTTa() 9 ms, load of a stack 1-8 ms (baseline_hyperon.json stacks)"}


def section_loads(a, stack_specs: dict, baseline: Optional[dict], fixes: set) -> dict:
    names = [n for n in (["smoke10", "human_evidence", "drugage_slice", "cellage_slice", "generic", "patient", "linage2"] if not a.quick else ["smoke10", "human_evidence", "cellage_slice"])
             if n in stack_specs]
    if a.stacks:
        names = [n for n in a.stacks.split(",") if n in stack_specs]
    rows = []
    for name in names:
        spec = stack_specs[name]
        part = SH.Part(name, spec.files, spec.extra_atoms, [])
        text = part.kb_text(fixes=fixes)
        cache = K.out_dir() / "petta_load_cache" / f"measure_{name}"
        row: dict = {"stack": name, "kb_bytes": len(text.encode("utf-8")), "n_files": len(spec.files), "cold_string": [], "cached_miss": None, "cached_hit": []}
        for _ in range(a.reps):
            r = A.run_program_report(text, [], engine=a.engine, mode="string", deadline_s=a.deadline, load_deadline_s=a.deadline, stack_limit_mb=a.stack_limit_mb, label=f"{name}/string")
            row["cold_string"].append(_load_row(r))
        import shutil
        if cache.exists():
            shutil.rmtree(cache)
        r = A.run_program_report(text, [], engine=a.engine, mode="cached", deadline_s=a.deadline, load_deadline_s=a.deadline, stack_limit_mb=a.stack_limit_mb, load_cache=str(cache), label=f"{name}/cached_miss")
        row["cached_miss"] = _load_row(r)
        for _ in range(a.reps):
            r = A.run_program_report(text, [], engine=a.engine, mode="cached", deadline_s=a.deadline, load_deadline_s=a.deadline, stack_limit_mb=a.stack_limit_mb, load_cache=str(cache), label=f"{name}/cached_hit")
            row["cached_hit"].append(_load_row(r))
        row["snapshot_bytes"] = sum(p.stat().st_size for p in cache.glob("*.snapshot")) if cache.exists() else None
        if baseline and name in baseline.get("stacks", {}):
            row["hyperon"] = baseline["stacks"][name]["hyperon"]
        rows.append(row)
        ok = [x for x in row["cold_string"] if x["status"] == "ok"]
        print(f"  loads {name:15s} cold {_stats([x['load_s'] for x in ok]).get('min')} s, cached miss {row['cached_miss']['load_s']} s, "
              f"hit {_stats([x['load_s'] for x in row['cached_hit'] if x['status'] == 'ok']).get('min')} s, rss {ok[0]['rss_mb'] if ok else None} MB", flush=True)
    return {"fixes": sorted(fixes), "stacks": rows}


def _load_row(r: A.RunReport) -> dict:
    return {"status": r.status, "error": r.first_error(), "import_s": r.timing.get("import_s"), "consult_s": r.timing.get("consult_s"),
            "load_s": r.timing.get("load_s"), "wall_s": r.timing.get("wall_s"), "rss_mb": r.rss_mb.get("load"), "peak_mb": r.peak_rss_mb,
            "cache_hit_guess": r.load.get("cache_hit_guess"), "cpu_s": r.timing.get("cpu_s")}


def section_latency(a, baseline: dict, fixes: set, fork_result: Optional[dict]) -> dict:
    if baseline is None:
        return {"skipped": "no baseline"}
    groups: dict = {}
    for sh in baseline["shapes"]:
        if sh.get("skipped") or sh["source"] not in ("smoke", "few_shot", "family", "rule_form", "route", "canary"):
            continue
        for pi, part in enumerate(sh["parts"]):
            if part["hyperon"].get("status") not in ("ok", "empty"):
                continue
            groups.setdefault((tuple(part["kb_files"]), part["extra_atoms"]), []).append((sh, pi))
    if a.quick:
        groups = {k: v for k, v in groups.items() if len(k[0]) <= 10}
    rows, per_kb = [], []
    for (files, extra), members in groups.items():
        part0 = members[0][0]["parts"][members[0][1]]
        text = SH.Part(part0["stack"], list(files), extra, []).kb_text(fixes=fixes)
        uniq: list = []
        for sh, pi in members:
            for ei, e in enumerate(sh["parts"][pi]["expressions"]):
                if (sh["id"], pi, ei) not in [u[:3] for u in uniq]:
                    uniq.append((sh["id"], pi, ei, e, sh["parts"][pi]["hyperon"]))
        reps = max(2, a.reps)
        exprs = [u[3] for u in uniq for _ in range(reps)]
        r = A.run_program_report(text, exprs, engine=a.engine, deadline_s=a.deadline, load_deadline_s=a.deadline, expr_deadline_s=a.expr_deadline,
                                 stack_limit_mb=a.stack_limit_mb, label=f"latency/{part0['stack']}")
        fresh = sum(x for x in (r.timing.get("import_s"), r.timing.get("consult_s"), r.timing.get("load_s")) if x is not None)
        per_kb.append({"stack": part0["stack"], "status": r.status, "fresh_process_s": round(fresh, 3), "import_s": r.timing.get("import_s"),
                       "consult_s": r.timing.get("consult_s"), "load_s": r.timing.get("load_s"), "n_expr": len(uniq)})
        print(f"  latency {part0['stack']:15s} {len(uniq):3d} expressions x{reps}: {r.status}; fresh-process cost {fresh:.2f} s", flush=True)
        if r.status != "ok":
            continue
        for k, (sid, pi, ei, e, hyp) in enumerate(uniq):
            secs = r.expr_seconds[k * reps:(k + 1) * reps]
            if not secs:
                continue
            first, warm = secs[0] * 1000, min(secs[1:]) * 1000 if len(secs) > 1 else secs[0] * 1000
            t = hyp.get("timing") or {}
            hms = (t.get("per_expr_ms") or [None] * (ei + 1))[ei] if t else None
            hyper_cold = (t.get("init_ms", 0) + t.get("load_ms", 0) + (hms or 0)) if t else None
            err = r.expr_errors[k * reps]
            rows.append({"shape": sid, "stack": part0["stack"], "expr": e[:100], "petta_first_ms": round(first, 2), "petta_warm_ms": round(warm, 2),
                         "hyperon_ms": hms, "hyperon_cold_ms": round(hyper_cold, 2) if hyper_cold else None,
                         "petta_cold_ms": round(fresh * 1000 + first, 1), "error": (err or {}).get("message", "")[:120] if err else None})
    for row in rows:
        if row["hyperon_cold_ms"]:
            row["ratio_fresh_process"] = round(row["petta_cold_ms"] / row["hyperon_cold_ms"], 1)
        if row["hyperon_ms"]:
            row["ratio_warm_query"] = round(row["petta_warm_ms"] / row["hyperon_ms"], 2)
    fork_ms = None
    if fork_result and fork_result.get("after_load"):
        ok = [x["fork_to_answer_s"] for x in fork_result["after_load"] if x.get("ok") and x.get("fork_to_answer_s") is not None]
        fork_ms = round(statistics.median(ok) * 1000, 1) if ok else None
    gate_rows = [r for r in rows if r.get("ratio_fresh_process") is not None]
    over = [r for r in gate_rows if r["ratio_fresh_process"] > 5]
    return {"kb_groups": per_kb, "rows": rows, "fork_median_ms": fork_ms,
            "gate": {"rule": "no routed probe more than 5x slower than hyperon with the load included (fresh process per query)",
                     "probes": len(gate_rows), "over_5x": len(over), "worst": sorted(gate_rows, key=lambda r: -r["ratio_fresh_process"])[:5],
                     "pass": (len(over) == 0) if gate_rows else None,
                     "fork_template_note": ("warm-template fork+query median %.1f ms" % fork_ms) if fork_ms else "run the fork section to price a warm template"}}


def section_probes(a, fixes: set) -> dict:
    """The print probes and the seven first-answer sites, via run_shapes (same normaliser and comparator)."""
    import run_shapes
    out_dir = K.out_dir() / "measure_probes"
    argv = ["--engine", a.engine, "--source", "probe_print,probe_order", "--fixes", a.fixes, "--mode", "batch", "--out", str(out_dir),
            "--deadline", str(a.deadline), "--expr-deadline", str(a.expr_deadline), "--stack-limit-mb", str(a.stack_limit_mb)]
    run_shapes.main(argv)
    rep = K.json_load(out_dir / "shapes_report.json")
    rows = []
    for r in rep["results"]:
        for p in r["parts"]:
            for e in p["exprs"]:
                rows.append({"shape": r["id"], "source": r["source"], "expr": e["expr"], "hyperon": e["hyperon"], "petta_raw": e["petta_raw"],
                             "petta_norm": e["petta_norm"], "class": e["class"], "detail": e.get("detail", ""), "tags": e.get("tags", []),
                             "norm_equal": e.get("norm_equal")})
    sites = []
    for r in rep["results"]:
        if r["id"].startswith("probe:site:"):
            es = [e for p in r["parts"] for e in p["exprs"]]
            sites.append({"site": r["id"][len("probe:site:"):], "class": r["class"],
                          "raw_order": {"hyperon": es[0]["hyperon"], "petta": es[0]["petta_raw"]} if es else None,
                          "first_answer": {"hyperon": es[-1]["hyperon"] if es else None, "petta": es[-1]["petta_raw"] if es else None},
                          "same_first_answer": bool(es) and es[-1]["class"] in ("identical", "number_format_only", "within_tolerance"),
                          "same_raw_order": bool(es) and es[0]["class"] in ("identical", "number_format_only", "within_tolerance")})
    return {"rows": rows, "sites": sites, "counts": rep["counts"]}


def section_limits(a) -> dict:
    out: dict = {}

    def one(label, expect, kb, exprs, **kw):
        r = A.run_program_report(kb, exprs, deadline_s=kw.pop("deadline_s", 120), expr_deadline_s=kw.pop("expr_deadline_s", 30),
                                 stack_limit_mb=kw.pop("stack_limit_mb", 256), label=label, **kw)
        d = {"expect": expect, "status": r.status, "error": r.first_error(), "exitcode": r.exitcode, "signal": r.signal, "timeout_scope": r.timeout_scope,
             "answers_raw": r.raw_answers(), "expr_errors": [(e or {}).get("message", "")[:240] if e else None for e in r.expr_errors],
             "wall_s": r.timing.get("wall_s"), "rss_mb": r.rss_mb, "stderr_tail": r.stderr[-300:], "stdout_tail": r.stdout[-300:]}
        out[label] = d
        print(f"  limits {label:34s} {r.status:10s} {str(r.first_error() or '')[:110]}", flush=True)
        return r

    one("runaway_recursion_deadline", "timeout, scope expr:0, SIGKILL, RSS at kill recorded", "(= (spin $n) (spin (+ $n 1)))", ["!(spin 0)"],
        expr_deadline_s=6, deadline_s=60)
    one("deep_recursion_stack_overflow", "expression 0 raises a stack-limit error, the child survives, expression 1 answers 3",
        "(= (deep $n) (+ 1 (deep (+ $n 1))))", ["!(deep 0)", "!(+ 1 2)"], stack_limit_mb=128, expr_deadline_s=90)
    one("halt_via_assert", "crash with exit code 1 at expression 1 (metta.pl: assert/2 calls halt(1)); expression 2 never runs",
        "(= (f) 1)", ["!(+ 1 1)", "!(assert (== 1 2))", "!(+ 2 2)"])
    one("halt_via_test", "crash with exit code 1 (metta.pl: test/3 halts when the two sides differ)", "(= (f) 1)", ["!(test (+ 1 1) 3)"])
    one("remove_atom_retracts_every_copy", "after remove-atom the list is (b): both copies of (Fact a) are gone (LAUNCH.md: remove-atom retracts every identical copy)",
        "(Fact a)\n(Fact a)\n(Fact b)",
        ["!(collapse (match &self (Fact $x) $x))", "!(remove-atom &self (Fact a))", "!(collapse (match &self (Fact $x) $x))"])
    one("second_load_duplicates_clauses", "(collapse (dup)) is (1 1): a second load of the same equation adds a clause",
        "(= (dup) 1)", ["(= (dup) 1)", "!(collapse (dup))"], bang=False)
    one("add_atom_then_query", "(1)", "(= (f) 1)", ["!(add-atom &self (Added 1))", "!(collapse (match &self (Added $x) $x))"])
    one("unknown_function_unreduced", "(known 2): no answer (a known function with no matching clause fails); (unknown-fn 2): unreduced",
        "(= (known 1) one)", ["!(known 2)", "!(unknown-fn 2)"])
    one("error_inside_collapse", "(f ()) raises car_atom_empty; (f (a b)) answers a", "(= (f $x) (car-atom $x))", ["!(f ())", "!(f (a b))"])
    return out


def _fork_kb(stack_specs: dict, fixes: set, name: str):
    spec = stack_specs[name]
    text = SH.Part(name, spec.files, spec.extra_atoms, []).kb_text(fixes=fixes)
    return text, list(spec.canary[:1])


def section_fork(a, stack_specs: dict, fixes: set) -> dict:
    name = a.fork_stack if a.fork_stack in stack_specs else "smoke10"
    text, exprs = _fork_kb(stack_specs, fixes, name)
    spec = _spec({"n_forks": a.forks, "fork_deadline_s": 20}, kb_text=text, expressions=exprs)
    sup = A.spawn_supervised(_fork_child, spec, total_deadline_s=a.deadline, load_deadline_s=a.deadline, expr_deadline_s=max(60, a.forks * 25))
    d = sup.done or {"ok": False, "error": {"message": f"template ended: exit {sup.exitcode} {sup.signal} timeout={sup.timeout_scope}"}}
    d["stack"] = name
    d["stderr_tail"] = sup.stderr[-400:]
    for k in ("after_consult", "after_load"):
        rows = d.get(k) or []
        ok = [r for r in rows if r.get("ok")]
        d[f"{k}_summary"] = {"forks": len(rows), "answered": len(ok), "hangs": sum(1 for r in rows if r.get("hang")),
                              "died": sum(1 for r in rows if not r.get("ok") and not r.get("hang")),
                              "fork_to_answer_s": _stats([r.get("fork_to_answer_s") for r in ok])}
    print(f"  fork: after_consult {d.get('after_consult_summary')}  after_load {d.get('after_load_summary')}", flush=True)
    return d


def section_gil(a) -> dict:
    spec = _spec({"loop": a.gil_loop, "count": a.gil_count})
    sup = A.spawn_supervised(_gil_child, spec, total_deadline_s=a.deadline, load_deadline_s=a.deadline, expr_deadline_s=a.deadline)
    d = sup.done or {"ok": False, "error": {"message": f"child ended: exit {sup.exitcode} {sup.signal} timeout={sup.timeout_scope}"}}
    d["stderr_tail"] = sup.stderr[-400:]
    for k in ("prolog_loop", "metta_loop"):
        if k in d:
            r = d[k]["tick_ratio"]
            if d[k]["seconds"] < 0.3:
                d[k]["reading"] = "call too short to judge: raise --gil-loop / --gil-count"
            else:
                d[k]["reading"] = ("GIL RELEASED during the call" if r is not None and r > 0.5 else
                                   "GIL HELD during the call (the event loop would starve, as with hyperon)" if r is not None and r < 0.1 else "partially")
    print(f"  gil: {d.get('prolog_loop')}  {d.get('metta_loop')}  {d.get('two_threads')}", flush=True)
    return d


# ── report ───────────────────────────────────────────────────────────────────

def render(doc: dict) -> str:
    L = ["# PeTTa measurements next to hyperon", "", f"ran {doc['meta']['ran']}, engine `{doc['meta']['engine']}`, sections {', '.join(doc['sections'])}.", ""]
    if "consult" in doc:
        s = doc["consult"]["summary"]
        L += ["## consult (fresh process, no KB)", "", "| step | min | median | max |", "|---|---|---|---|"]
        for k, label in (("import_janus_s", "import janus_swi (starts SWI-Prolog)"), ("consult_main_pl_s", "consult src/main.pl"), ("consult_helper_pl_s", "consult python/helper.pl"),
                         ("first_calls_s_0", "first process_metta_string('!(+ 1 2)')"), ("first_calls_s_1", "second call"), ("first_calls_s_2", "third call (collapse/superpose)"),
                         ("rss_import_mb", "RSS after import (MB)"), ("rss_main_mb", "RSS after main.pl (MB)"), ("rss_end_mb", "RSS at end (MB)"), ("wall_s", "child wall")):
            v = s.get(k, {})
            if v.get("n"):
                L.append(f"| {label} | {v['min']} | {v['median']} | {v['max']} |")
        L += ["", doc["consult"]["hyperon_reference"], ""]
    if "loads" in doc:
        L += ["## loads", "", f"fixes applied: {', '.join(doc['loads']['fixes']) or 'none'}", "",
              "| stack | KB KB | cold string s (min) | cached miss s | cached hit s (min) | hit? | consult s | RSS MB | hyperon load ms | hyperon RSS MB |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in doc["loads"]["stacks"]:
            cold = _stats([x["load_s"] for x in r["cold_string"] if x["status"] == "ok"])
            hit = _stats([x["load_s"] for x in r["cached_hit"] if x["status"] == "ok"])
            cons = _stats([x["consult_s"] for x in r["cold_string"] if x["status"] == "ok"])
            rss = r["cold_string"][0]["rss_mb"] if r["cold_string"] else None
            h = r.get("hyperon") or {}
            err = next((x["error"] for x in r["cold_string"] if x["status"] != "ok"), None)
            L.append(f"| {r['stack']} | {r['kb_bytes'] // 1000} | {cold.get('min', err and 'FAIL')} | {r['cached_miss']['load_s']} | {hit.get('min')} | "
                     f"{[x['cache_hit_guess'] for x in r['cached_hit']]} | {cons.get('min')} | {rss} | {h.get('load_ms')} | {h.get('rss_mb')} |")
        L.append("")
    if "latency" in doc and "gate" in doc["latency"]:
        g = doc["latency"]["gate"]
        L += ["## latency and the Phase 1 gate", "", f"{g['rule']}: **{'PASS' if g['pass'] else 'FAIL' if g['pass'] is not None else 'not evaluated'}** ({g['over_5x']} of {g['probes']} over 5x). {g['fork_template_note']}.", "",
              "| KB | status | fresh-process cost s (import + consult + load) | expressions |", "|---|---|---|---|"]
        for k in doc["latency"]["kb_groups"]:
            L.append(f"| {k['stack']} | {k['status']} | {k['fresh_process_s']} | {k['n_expr']} |")
        L += ["", "| shape | stack | hyperon query ms | PeTTa first ms | PeTTa warm ms | warm / hyperon | fresh-process total / hyperon total |", "|---|---|---|---|---|---|---|"]
        rows = sorted([r for r in doc["latency"]["rows"] if r.get("ratio_fresh_process")], key=lambda r: -r["ratio_fresh_process"])
        for r in rows[:40]:
            L.append(f"| {r['shape']} | {r['stack']} | {r['hyperon_ms']} | {r['petta_first_ms']} | {r['petta_warm_ms']} | {r.get('ratio_warm_query', '-')} | {r['ratio_fresh_process']} |")
        L.append("")
    if "probes" in doc:
        L += ["## how things print (hyperon | PeTTa raw | PeTTa normalised | class)", "", "| probe | expression | hyperon | PeTTa raw | normalised | class |", "|---|---|---|---|---|---|"]
        for r in doc["probes"]["rows"]:
            if r["source"] == "probe_print":
                L.append(f"| {r['shape'].split(':')[-1]} | `{r['expr']}` | `{r['hyperon']}` | `{r['petta_raw']}` | `{r['petta_norm']}` | {r['class']}{' ' + r['detail'][:90] if r['class'] == 'error' else ''} |")
        L += ["", "## answer order", "", "| probe | expression | hyperon | PeTTa raw | class |", "|---|---|---|---|---|"]
        for r in doc["probes"]["rows"]:
            if r["source"] == "probe_order" and not r["shape"].startswith("probe:site:"):
                L.append(f"| {r['shape'].split(':')[-1]} | `{r['expr'][:70]}` | `{r['hyperon']}` | `{r['petta_raw']}` | {r['class']} |")
        L += ["", "### the seven first-answer sites (verdict section 2 item 7)", "",
              "| site | raw order same | first answer same | hyperon first answer | PeTTa first answer | raw order hyperon | raw order PeTTa |", "|---|---|---|---|---|---|---|"]
        for s in doc["probes"]["sites"]:
            L.append(f"| {s['site']} | {s['same_raw_order']} | {s['same_first_answer']} | `{(s['first_answer']['hyperon'] or [''])[0][:80]}` | `{(s['first_answer']['petta'] or [''])[0][:80]}` | "
                     f"`{(s['raw_order']['hyperon'] or [''])[0][:70]}` | `{(s['raw_order']['petta'] or [''])[0][:70]}` |")
        L.append("")
    if "limits" in doc:
        L += ["## limits and hazards", "", "| probe | expected | status | what happened | answers |", "|---|---|---|---|---|"]
        for k, v in doc["limits"].items():
            what = v["error"] or "; ".join(str(e) for e in v["expr_errors"] if e) or "-"
            if v.get("signal") or v.get("exitcode") not in (None, 0):
                what += f" [exit {v.get('exitcode')} {v.get('signal') or ''}]"
            L.append(f"| {k} | {v['expect']} | {v['status']} | {str(what)[:220]} | `{str(v['answers_raw'])[:160]}` |")
        L.append("")
    if "fork" in doc:
        f = doc["fork"]
        L += ["## fork() after a child-only import", "", f"template stack `{f.get('stack')}`; ok={f.get('ok')} {f.get('error', '')}", "",
              f"* after consult only: {f.get('after_consult_summary')}", f"* after the KB is loaded: {f.get('after_load_summary')}",
              f"* template load {f.get('template_load_s')} s, RSS {f.get('rss_after_load_mb')} MB; template healthy afterwards: `{f.get('template_after')}`", ""]
    if "gil" in doc:
        g = doc["gil"]
        L += ["## GIL", "", f"canary ticks idle: {g.get('canary_ticks_per_s_idle')} per s", "",
              f"* pure Prolog loop: {g.get('prolog_loop')}", f"* PeTTa loop: {g.get('metta_loop')}", f"* two threads: {g.get('two_threads')}", ""]
    return "\n".join(L)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["petta", "hyperon"], default="petta", help="hyperon: only the sections that make sense there (self-test)")
    ap.add_argument("--sections", default=",".join(ALL_SECTIONS))
    ap.add_argument("--stacks", help="loads section: comma list of stacks")
    ap.add_argument("--fixes", default="all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="reps=1 and the small stacks only")
    ap.add_argument("--forks", type=int, default=5)
    ap.add_argument("--fork-stack", default="generic")
    ap.add_argument("--gil-loop", type=int, default=30_000_000)
    ap.add_argument("--gil-count", type=int, default=1_000_000)
    ap.add_argument("--deadline", type=float, default=900.0)
    ap.add_argument("--expr-deadline", type=float, default=180.0)
    ap.add_argument("--stack-limit-mb", type=int, default=A.DEFAULT_STACK_LIMIT_MB)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.quick:
        a.reps = 1
    sections = [s for s in a.sections.split(",") if s]
    if a.engine == "hyperon":
        sections = [s for s in sections if s in ("loads", "latency", "probes")]
    fixes = parse_fix_spec(a.fixes)
    base_path = a.baseline or K.baseline_path()
    baseline = K.json_load(base_path) if base_path.exists() else None
    stack_specs = S.list_stacks()
    doc: dict = {"meta": {"kit_version": K.KIT_VERSION, "ran": K.now_iso(), "engine": a.engine, "petta_home": str(K.petta_home()), "env": K.env_report(),
                          "baseline_git_head": (baseline or {}).get("meta", {}).get("git_head"), "fixes": sorted(fixes)}, "sections": sections}
    t0 = time.perf_counter()
    for sec in ("consult", "loads", "probes", "limits", "fork", "gil", "latency"):   # latency last: it uses the fork result
        if sec not in sections:
            continue
        print(f"== {sec}", flush=True)
        if sec == "consult":
            doc[sec] = section_consult(a)
        elif sec == "loads":
            doc[sec] = section_loads(a, stack_specs, baseline, fixes)
        elif sec == "probes":
            if baseline is None:
                doc[sec] = {"rows": [], "sites": [], "error": "no baseline"}
            else:
                doc[sec] = section_probes(a, fixes)
        elif sec == "limits":
            doc[sec] = section_limits(a)
        elif sec == "fork":
            doc[sec] = section_fork(a, stack_specs, fixes)
        elif sec == "gil":
            doc[sec] = section_gil(a)
        elif sec == "latency":
            doc[sec] = section_latency(a, baseline, fixes, doc.get("fork"))
    doc["meta"]["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out_dir = a.out or K.out_dir()
    K.json_dump(doc, out_dir / "measure.json")
    (out_dir / "measure.md").write_text(render(doc), encoding="utf-8")
    print(f"wrote {out_dir / 'measure.json'} and {out_dir / 'measure.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except A.PettaNotAllowed as exc:
        print(f"not started: {exc}", file=sys.stderr)
        raise SystemExit(3)
