#!/usr/bin/env python
"""Record every query shape's answers and timing on HYPERON 0.2.10  ->  baseline_hyperon.json.

Run with the venv that has hyperon, from anywhere (the repo is read through PLN_REPO):

    /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python record_baseline.py

What it does, per shape part (one KB load and its expressions), in a FRESH child process so that a hyperon
abort (a non-unwinding Rust panic) is recorded as `abort` instead of killing the recorder:
  * the repo's own runner, pln_runner._hyperon_run, on the real files: the answers the app sees;
  * hyperon.MeTTa() directly, `--reps` times: load time and per-expression time, and the answers again;
  * with the three Phase 2 fixes applied to the KB TEXT: are the answers byte-identical to the unfixed ones?
Then the load time and RSS of every stack. The work tree is only read (pln_chat/logs is redirected into out/).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import kit_common as K
import stacks as S
import shapes as SH
from kb_fixes import ALL_FIXES, fix_kb_texts

CHILD = K.KIT_DIR / "hyperon_child.py"
_ERR_ATOM = re.compile(r"\(Error\b")


def run_child(spec: dict, timeout: float) -> dict:
    """Run hyperon_child.py on `spec`; returns {'stages': {...}, 'returncode', 'stderr_tail', 'wall_ms', 'timeout'}."""
    spec = {**spec, "kit_dir": str(K.KIT_DIR), "repo": str(K.repo_root())}
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHON_DOTENV_DISABLED": "1", "HOME": str(Path.home()), "PLN_REPO": str(K.repo_root())}
    t = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-B", str(CHILD)], input=json.dumps(spec), capture_output=True, text=True,
                           timeout=timeout, cwd=str(K.repo_root()), env=env)
        timed_out, rc, out, err = False, p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out, rc = True, None
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    wall = round((time.perf_counter() - t) * 1000, 1)
    stages: dict = {}
    for line in out.splitlines():
        if line.startswith("@@RESULT@@ "):
            try:
                d = json.loads(line[len("@@RESULT@@ "):])
                stages[d.pop("stage")] = d
            except json.JSONDecodeError:
                pass
    return {"stages": stages, "returncode": rc, "stderr_tail": err.strip()[-700:], "wall_ms": wall, "timeout": timed_out}


def _groups_from_atoms(atoms: list[dict], n_expr: int) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(n_expr)]
    for a in atoms:
        i = a.get("expr_index")
        if i is not None and 0 <= i < n_expr:
            groups[i].append(a["atom"])
    return groups


def _flatten_direct(run0: dict, n_expr: int) -> list[list[str]]:
    out = []
    for per_expr in run0.get("groups", []):
        out.append([a for g in per_expr for a in g])
    while len(out) < n_expr:
        out.append([])
    return out


def record_part(part: SH.Part, reps: int, timeout: float, check_fixes: bool, fixes: set[str]) -> dict:
    n_expr = len(part.expressions)
    kb_text = part.kb_text()
    rec: dict = {
        "stack": part.stack, "kb_files": part.kb_files, "extra_atoms": part.extra_atoms, "expressions": part.expressions,
        "kb_sha256": K.sha256_text(kb_text), "kb_bytes": len(kb_text.encode("utf-8")),
    }
    res = run_child({"kb_files": part.kb_files, "extra_atoms": part.extra_atoms, "expressions": part.expressions,
                     "reps": reps, "runner": bool(part.kb_files)}, timeout)
    st = res["stages"]
    hyp: dict = {"returncode": res["returncode"], "child_wall_ms": res["wall_ms"]}
    runner, direct = st.get("runner"), st.get("direct")
    runs = [r for r in (direct or {}).get("runs", []) if "error" not in r]
    hyp["import_ms"] = (st.get("env") or {}).get("import_ms")
    if res["timeout"]:
        hyp.update(status="timeout", error=f"no answer within {timeout:.0f}s", groups=[[] for _ in range(n_expr)])
    elif runner is None and direct is None:
        hyp.update(status="abort", error="the child died before answering (a hyperon abort: trie.rs:179 panics exit via SIGABRT)",
                   stderr_tail=res["stderr_tail"], groups=[[] for _ in range(n_expr)])
    else:
        direct_groups = _flatten_direct(runs[0], n_expr) if runs else None
        runner_groups = _groups_from_atoms(runner["atoms"], n_expr) if runner and runner["status"] != "error" else None
        groups = runner_groups if runner_groups is not None else (direct_groups or [[] for _ in range(n_expr)])
        hyp["groups"] = groups
        hyp["groups_from"] = "runner" if runner_groups is not None else "direct"
        if runner is not None:
            hyp["runner_status"] = runner["status"]
            hyp["runner_error"] = runner.get("error")
        if direct_groups is not None and runner_groups is not None:
            hyp["runner_matches_direct"] = direct_groups == runner_groups
            if direct_groups != runner_groups:
                hyp["direct_groups"] = direct_groups
        if runner is not None and runner["status"] == "error":
            hyp.update(status="error", error=runner.get("error"))
        else:
            hyp["status"] = "ok" if any(groups) else "empty"
        if (direct or {}).get("runs") and "error" in direct["runs"][-1]:
            hyp["direct_error"] = direct["runs"][-1]["error"]
        hyp["has_error_atoms"] = any(_ERR_ATOM.match(a) for g in groups for a in g)
        if runs:
            hyp["timing"] = {
                "runner_ms": runner["query_time_ms"] if runner else None,
                "runner_wall_ms": runner["wall_ms"] if runner else None,
                "init_ms": min(r["init_ms"] for r in runs),
                "load_ms": min(r["load_ms"] for r in runs),
                "per_expr_ms": [min(r["per_expr_ms"][i] for r in runs) for i in range(len(runs[0]["per_expr_ms"]))],
                "load_ms_all": [r["load_ms"] for r in runs],
            }
            hyp["rss_mb"] = runs[-1].get("rss_mb")
            hyp["peak_mb"] = runs[-1].get("peak_mb")
    rec["hyperon"] = hyp

    # the Phase 2 fixes, applied to the KB text, must not change a single answer on hyperon
    if check_fixes and hyp.get("status") in ("ok", "empty"):
        named = []
        for f in part.kb_files:
            p = Path(f)
            p = p if p.is_absolute() else K.repo_root() / f
            named.append((f, p.read_text(encoding="utf-8")))
        fixed, report = fix_kb_texts(named, fixes)
        if not report:
            rec["fixed"] = {"fixes": sorted(fixes), "class": "no-change", "report": {}, "detail": "no fix site in this KB"}
        else:
            fixed_text = "\n".join([t for _, t in fixed] + ([part.extra_atoms] if part.extra_atoms else []))
            r2 = run_child({"kb_files": part.kb_files, "extra_atoms": part.extra_atoms, "expressions": part.expressions,
                            "reps": 1, "runner": False, "kb_text": fixed_text}, timeout)
            d2 = (r2["stages"].get("direct") or {}).get("runs") or []
            if r2["timeout"] or not d2 or "error" in d2[0]:
                rec["fixed"] = {"fixes": sorted(fixes), "class": "error", "report": report,
                                "detail": (d2[0].get("error") if d2 else None) or f"child failed rc={r2['returncode']} {r2['stderr_tail'][-200:]}"}
            else:
                g2 = _flatten_direct(d2[0], n_expr)
                classes = [K.compare_groups(a, b) for a, b in zip(hyp["groups"], g2)]
                worst = K.worst_class(c["class"] for c in classes)
                first_bad = next((c["detail"] for c in classes if c["class"] != "identical"), "")
                rec["fixed"] = {"fixes": sorted(fixes), "class": worst, "report": report, "detail": first_bad}
    return rec


def record_shape(shape: SH.Shape, reps: int, timeout: float, check_fixes: bool, fixes: set[str]) -> dict:
    d = shape.to_json()
    d["parts"] = []
    if shape.skipped:
        return d
    for part in shape.parts:
        d["parts"].append(record_part(part, reps, timeout, check_fixes, fixes))
    return d


def record_stack(spec: S.StackSpec, reps: int, timeout: float) -> dict:
    kb = spec.kb_text()
    res = run_child({"kb_files": spec.files, "extra_atoms": spec.extra_atoms, "expressions": [], "reps": reps, "runner": False}, timeout)
    d = (res["stages"].get("direct") or {}).get("runs") or []
    out = {"files": spec.files, "n_files": len(spec.files), "extra_atoms_bytes": len(spec.extra_atoms),
           "kb_sha256": K.sha256_text(kb), "kb_bytes": len(kb.encode("utf-8")), "description": spec.description}
    if res["timeout"]:
        out["hyperon"] = {"status": "timeout"}
    elif not d:
        out["hyperon"] = {"status": "abort", "returncode": res["returncode"], "stderr_tail": res["stderr_tail"]}
    elif "error" in d[0]:
        out["hyperon"] = {"status": "error", "error": d[0]["error"]}
    else:
        out["hyperon"] = {"status": "ok", "init_ms": min(r["init_ms"] for r in d), "load_ms": min(r["load_ms"] for r in d),
                          "load_ms_all": [r["load_ms"] for r in d], "rss_mb": d[-1].get("rss_mb"), "peak_mb": d[-1].get("peak_mb"),
                          "import_ms": (res["stages"].get("env") or {}).get("import_ms")}
    return out


def git_status() -> str:
    """`git status --porcelain` of the work tree (a read-only call), to show that recording left it as it was."""
    try:
        return subprocess.run(["git", "-C", str(K.repo_root()), "status", "--porcelain"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return "unknown"


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(K.repo_root()), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="default: <kit>/baseline_hyperon.json")
    ap.add_argument("--only", help="regex on shape ids (for a quick test)")
    ap.add_argument("--reps", type=int, default=3, help="timing repetitions inside one child (min is reported)")
    ap.add_argument("--timeout", type=float, default=240.0, help="per child process, seconds")
    ap.add_argument("--jobs", type=int, default=1, help="parallel children. 1 keeps the timings uncontended")
    ap.add_argument("--fixes", default="all", help="the Phase 2 fixes to validate: all | none | comma list")
    ap.add_argument("--no-check-fixes", action="store_true")
    ap.add_argument("--no-stack-timing", action="store_true")
    a = ap.parse_args(argv)

    from kb_fixes import parse_fix_spec
    fixes = parse_fix_spec(a.fixes)
    check_fixes = bool(fixes) and not a.no_check_fixes
    out_path = a.out or K.baseline_path()

    S.repo_import()
    status_before = git_status()
    t_all = time.perf_counter()
    all_shapes = SH.build_shapes()
    todo = [s for s in all_shapes if not a.only or re.search(a.only, s.id)]
    print(f"{len(todo)} of {len(all_shapes)} shapes; hyperon child = {sys.executable}; jobs={a.jobs}; fixes={sorted(fixes)}", flush=True)

    results: list[Optional[dict]] = [None] * len(todo)

    def work(i: int) -> None:
        s = todo[i]
        results[i] = record_shape(s, a.reps, a.timeout, check_fixes, fixes)
        parts = results[i]["parts"]
        stat = "skipped" if s.skipped else "/".join(p["hyperon"]["status"] for p in parts)
        fx = ",".join(p.get("fixed", {}).get("class", "-") for p in parts) if parts else ""
        print(f"[{i + 1:3d}/{len(todo)}] {s.id:34s} {stat:14s} fixed:{fx}", flush=True)
        if (i + 1) % 25 == 0:
            K.json_dump({"partial": True, "shapes": [r for r in results if r]}, out_path.with_suffix(".partial.json"))

    if a.jobs <= 1:
        for i in range(len(todo)):
            work(i)
    else:
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(work, range(len(todo))))

    stack_rec: dict = {}
    if not a.no_stack_timing:
        for name, spec in S.list_stacks().items():
            stack_rec[name] = record_stack(spec, max(a.reps, 3), a.timeout)
            h = stack_rec[name]["hyperon"]
            print(f"stack {name:15s} {h['status']:8s} load {h.get('load_ms', '-')} ms  rss {h.get('rss_mb', '-')} MB", flush=True)

    done = [r for r in results if r]
    counts: dict = {"by_source": {}, "by_status": {}, "parts": 0}
    for r in done:
        counts["by_source"][r["source"]] = counts["by_source"].get(r["source"], 0) + 1
        if r.get("skipped"):
            counts["by_status"]["skipped"] = counts["by_status"].get("skipped", 0) + 1
        for p in r["parts"]:
            counts["parts"] += 1
            k = p["hyperon"]["status"]
            counts["by_status"][k] = counts["by_status"].get(k, 0) + 1
    fx_counts: dict = {}
    for r in done:
        for p in r["parts"]:
            if "fixed" in p:
                fx_counts[p["fixed"]["class"]] = fx_counts.get(p["fixed"]["class"], 0) + 1
    counts["fixes_vs_unfixed"] = fx_counts
    counts["runner_vs_direct_mismatch"] = [r["id"] for r in done for p in r["parts"] if p["hyperon"].get("runner_matches_direct") is False]

    status_after = git_status()
    doc = {
        "meta": {"kit_version": K.KIT_VERSION, "recorded": K.now_iso(), "repo": str(K.repo_root()), "git_head": git_head(),
                 "worktree_status_before": status_before.splitlines(), "worktree_status_after": status_after.splitlines(),
                 "worktree_changed_during_run": status_before != status_after,
                 "python": sys.version.split()[0], "env": K.env_report(), "reps": a.reps, "jobs": a.jobs,
                 "fixes_checked": sorted(fixes) if check_fixes else [], "elapsed_s": round(time.perf_counter() - t_all, 1),
                 "note": "answers are hyperon's raw strings; `groups` has one list per `!` expression"},
        "counts": counts,
        "stacks": stack_rec,
        "shapes": done,
    }
    K.json_dump(doc, out_path)
    partial = out_path.with_suffix(".partial.json")
    if partial.exists():
        partial.unlink()
    print(json.dumps(counts, indent=1))
    print(f"wrote {out_path}  ({len(done)} shapes, {counts['parts']} KB loads, {doc['meta']['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
