"""Run ONE KB load and its expressions on hyperon 0.2.10, in this fresh process. Used by record_baseline.py.

Reads a JSON spec on stdin, writes lines `@@RESULT@@ {json}` on stdout (one per pass), so a hyperon abort
(a non-unwinding Rust panic, trie.rs:179) loses only the passes that had not been printed yet.

spec: {kit_dir, repo, kb_files:[repo-relative], extra_atoms, expressions:["!(…)"], reps, runner:bool, kb_text:null|str}

pass "runner": the repo's own pln_runner._hyperon_run on the real files, exactly as production calls it
               (needs kb_files; ignores kb_text). Gives the answers the app would see and the runner's own timing.
pass "direct": hyperon.MeTTa() directly, `reps` times: constructor, `run(kb_text)`, then every expression on
               its own run(), so load and per-expression time are separate numbers. kb_text may be supplied
               (a fixed KB); otherwise it is rebuilt from the files the way the runner does.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def emit(stage: str, payload: dict) -> None:
    sys.stdout.write("@@RESULT@@ " + json.dumps({"stage": stage, **payload}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def rss_mb() -> float | None:
    try:
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=5).stdout.strip()
        return round(int(out) / 1024.0, 1) if out else None
    except Exception:
        return None


def peak_mb() -> float | None:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round((v / 1e6) if sys.platform == "darwin" else (v / 1024.0), 1)
    except Exception:
        return None


def main() -> int:
    spec = json.loads(sys.stdin.read())
    sys.dont_write_bytecode = True
    sys.path.insert(0, spec["kit_dir"])
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["PLN_REPO"] = spec["repo"]
    import stacks as S
    S.repo_import()                                       # pln_chat on sys.path, .env off, LOGS_DIR in the kit's output
    repo = Path(spec["repo"])
    files = [Path(f) if Path(f).is_absolute() else repo / f for f in spec["kb_files"]]
    exprs = list(spec.get("expressions") or [])
    extra = spec.get("extra_atoms") or ""

    t0 = time.perf_counter()
    import hyperon
    from hyperon import MeTTa
    import_ms = round((time.perf_counter() - t0) * 1000, 1)
    emit("env", {"import_ms": import_ms, "hyperon": getattr(hyperon, "__version__", "?"), "python": sys.version.split()[0],
                 "rss_mb": rss_mb()})

    if spec.get("runner") and exprs:
        from core.pln_runner import _hyperon_run
        t = time.perf_counter()
        res = _hyperon_run("\n".join(exprs), 0.0, files, extra or None)
        wall = round((time.perf_counter() - t) * 1000, 1)
        emit("runner", {
            "status": res.status, "error": res.error, "error_code": res.error_code,
            "query_time_ms": res.query_time_ms, "wall_ms": wall,
            "atoms": [{"atom": a.atom, "expr_index": a.expr_index} for a in res.results],
            "rss_mb": rss_mb(), "peak_mb": peak_mb(),
        })

    if spec.get("kb_text") is not None:
        kb_text = spec["kb_text"]
    else:
        blocks = [p.read_text(encoding="utf-8") for p in files]
        if extra:
            blocks.append(extra)
        kb_text = "\n".join(blocks)

    reps = max(1, int(spec.get("reps") or 1))
    runs = []
    for r in range(reps):
        try:
            t = time.perf_counter()
            m = MeTTa()
            init_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            if kb_text.strip():
                m.run(kb_text)
            load_ms = (time.perf_counter() - t) * 1000
            per, groups = [], []
            for e in exprs:
                t = time.perf_counter()
                raw = m.run(e)
                per.append(round((time.perf_counter() - t) * 1000, 2))
                groups.append([[str(a) for a in g] for g in raw])
            runs.append({"init_ms": round(init_ms, 1), "load_ms": round(load_ms, 1), "per_expr_ms": per, "groups": groups,
                         "rss_mb": rss_mb(), "peak_mb": peak_mb()})
        except Exception as exc:                          # a Python-level failure; an abort never gets here
            runs.append({"error": f"{type(exc).__name__}: {exc}"})
            break
    emit("direct", {"runs": runs, "kb_bytes": len(kb_text.encode("utf-8"))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
