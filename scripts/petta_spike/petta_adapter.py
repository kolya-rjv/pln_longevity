"""A THROWAWAY adapter that runs a MeTTa program on PeTTa:  run_program(kb_text, expressions) -> list[list[str]].

Shaped as the later `pln_chat/core/metta_runtime.py` PeTTa implementation would be, but written for the spike:
everything it learns is returned in a RunReport, nothing is hidden behind a log line.

Design (each point answers a rule of docs/petta_migration/LAUNCH.md, Phase 3):
  * A FRESH CHILD PROCESS PER KB LOAD ('spawn' context). PeTTa's space and its compiled clauses are
    process-global, with no clear; a second load in one process duplicates every clause. The child is the only
    place that imports janus_swi and petta: the parent never touches Prolog, so it can fork/spawn freely.
  * ONE helper call for the KB and ONE PER `!` EXPRESSION. PeTTa hands Python one flat list per call, so a call
    per expression is the only way to keep the per-expression grouping `expr_index` needs.
  * LOAD VERIFICATION. Any exception during the load is a failed load. A load whose goal FAILS without raising
    comes back from PeTTa's own python wrapper as `[]`, the same as a good load of a KB with no `!` forms
    (python/petta/__init__.py _run_helper); this adapter therefore calls the same janus query itself and keeps
    the `None`. After the load it asks Prolog how many clauses each `(= (f …) …)` head of the KB has, so a
    partial load (forms before the failing one stay loaded) is visible.
  * A HARD DEADLINE with SIGKILL, for the whole run, for the load and for every single expression.
  * stdout/stderr are captured at the file-descriptor level (Prolog may write below Python's sys.stdout);
    wall time, CPU time and RSS are taken at every stage.
  * `mode="cached"` writes the KB to a temp file named by its content hash and loads it with
    `load_metta_file_cached`; the snapshot lives in $PETTA_LOAD_CACHE (kept inside the kit's out/).
    The cached load MUST be the first helper call of the process: the snapshot is keyed by the dynamic
    database, and every helper call asserts one `silent(true)` fact.
  * Result strings are NORMALISED to hyperon's rendering (True/False, Rust-style exponents, ints as ints) and the
    raw PeTTa string is kept next to each (Answer.raw / Answer.norm). kit_common.normalise_answer says how.

`engine="hyperon"` runs the same plumbing on hyperon, so the kit can be self-tested without PeTTa
(selftest.py); it is not a production path.

Nothing here runs at import time except standard-library imports.
"""
from __future__ import annotations

import io
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import kit_common as K

DEFAULT_DEADLINE_S = 300.0
DEFAULT_STACK_LIMIT_MB = 2048          # PeTTa's wrapper sets 8 GB; a runaway recursion should die long before the Mac does
_MAX_CAPTURE = 20000


class PettaNotAllowed(RuntimeError):
    """The kit refuses to start the real PeTTa until KIT_ALLOW_PETTA=1 is exported (the owner's explicit go)."""


def petta_allowed(home) -> tuple[bool, str]:
    """May a child import janus_swi and consult PeTTa from `home`?

    Default DENY. The kit was written without running PeTTa; running it is a decision for the owner, taken by
    exporting KIT_ALLOW_PETTA=1 in the shell that starts a script. A directory holding a FAKE_PETTA marker (the
    self-test's own stand-in, never upstream code) is always allowed.
    """
    if (Path(home) / "FAKE_PETTA").exists():
        return True, "fake PeTTa tree (self-test)"
    if os.environ.get("KIT_ALLOW_PETTA") == "1":
        return True, "KIT_ALLOW_PETTA=1"
    return False, (f"refusing to start PeTTa from {home}: export KIT_ALLOW_PETTA=1 to give the go "
                   f"(the kit's author was told not to run it; hyperon-engine self-tests never need it)")


class PettaRunError(RuntimeError):
    """A run that did not complete cleanly. `.report` is the full RunReport."""

    def __init__(self, report: "RunReport", message: str = ""):
        self.report = report
        super().__init__(message or f"{report.status}: {report.error or ''}".strip())


@dataclass
class Answer:
    raw: str          # PeTTa's own string
    norm: str         # in hyperon's rendering, what the repo's parsers expect


@dataclass
class RunReport:
    status: str = "unknown"      # ok | import_error | consult_error | load_error | load_failed | timeout | crash | child_error
    engine: str = "petta"
    mode: str = "string"
    label: str = ""
    error: Optional[str] = None
    kb_bytes: int = 0
    kb_sha256: str = ""
    n_expressions: int = 0
    answers: list = field(default_factory=list)          # list[list[Answer]]
    expr_errors: list = field(default_factory=list)      # list[Optional[dict]]  {type, message}
    expr_seconds: list = field(default_factory=list)
    load: dict = field(default_factory=dict)             # exception, failed_silently, seconds, verify, cache
    timing: dict = field(default_factory=dict)           # import_s consult_s load_s queries_s child_s wall_s cpu_s
    rss_mb: dict = field(default_factory=dict)           # stage -> MB
    peak_rss_mb: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    exitcode: Optional[int] = None
    signal: Optional[str] = None
    timeout_scope: Optional[str] = None                  # total | load | expr:<i>
    stages: list = field(default_factory=list)           # the child's progress messages, in order (not the per-expression ones)
    completed: list = field(default_factory=list)        # indexes of the expressions that finished (all of them on status ok)
    env: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def n_expr_errors(self) -> int:
        return sum(1 for e in self.expr_errors if e)

    def normalised(self) -> list[list[str]]:
        return [[a.norm for a in group] for group in self.answers]

    def raw_answers(self) -> list[list[str]]:
        return [[a.raw for a in group] for group in self.answers]

    def first_error(self) -> Optional[str]:
        """The first thing that went wrong, as one line."""
        if self.status != "ok":
            exc = self.load.get("exception")
            if exc:
                return f"{exc.get('type')}: {exc.get('message')}"
            return self.error
        for i, e in enumerate(self.expr_errors):
            if e:
                return f"expression {i}: {e.get('type')}: {e.get('message')}"
        return None

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# ── the supervisor (parent side) ─────────────────────────────────────────────

@dataclass
class Supervised:
    """What the parent learned from one child: its messages, how it ended."""
    messages: list = field(default_factory=list)         # [(kind, name, payload)]
    done: Optional[dict] = None
    exitcode: Optional[int] = None
    signal: Optional[str] = None
    timeout_scope: Optional[str] = None
    wall_s: float = 0.0
    rss_at_kill_mb: Optional[float] = None
    stdout: str = ""
    stderr: str = ""


def _signal_name(exitcode: Optional[int]) -> Optional[str]:
    if exitcode is not None and exitcode < 0:
        try:
            return signal.Signals(-exitcode).name
        except ValueError:
            return f"SIG{-exitcode}"
    return None


def _read_tail(path: Path, limit: int = _MAX_CAPTURE) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
    return text if len(text) <= limit else text[:limit // 2] + "\n…[truncated]…\n" + text[-limit // 2:]


def spawn_supervised(
    target: Callable,
    spec: dict,
    *,
    total_deadline_s: float = DEFAULT_DEADLINE_S,
    load_deadline_s: Optional[float] = None,
    expr_deadline_s: Optional[float] = None,
    start_method: str = "spawn",
) -> Supervised:
    """Run `target(conn, spec)` in a fresh process and supervise it.

    The child talks over `conn` with tuples ("stage", name, payload) and a final ("done", payload).
    Stage names the supervisor knows: "imported", "consulted" and "loaded" move the clock from the load
    deadline to the expression deadline; "expr_start" (payload {"i": n}) restarts the per-expression clock.
    Past any of the deadlines the child is SIGKILLed (a janus call cannot be interrupted politely).
    `target` must be a module-level function (spawn pickles it by reference).
    """
    if spec.get("engine", "petta") == "petta":
        ok, why = petta_allowed(spec.get("petta_home") or K.petta_home())
        if not ok:
            raise PettaNotAllowed(why)
    load_deadline_s = load_deadline_s or total_deadline_s
    expr_deadline_s = expr_deadline_s or total_deadline_s
    tmp_root = K.out_dir() / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="petta_run_", dir=str(tmp_root)))
    spec = dict(spec, stdout_path=str(tmpdir / "stdout.txt"), stderr_path=str(tmpdir / "stderr.txt"))
    ctx = multiprocessing.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(child_conn, spec), daemon=True)
    sup = Supervised()
    t0 = time.monotonic()
    proc.start()
    child_conn.close()
    stage, stage_t, cur_expr = "load", t0, None
    try:
        while True:
            now = time.monotonic()
            stage_limit = (load_deadline_s if stage == "load" else expr_deadline_s)
            limit = min(t0 + total_deadline_s, stage_t + stage_limit)
            remaining = limit - now
            if remaining <= 0:
                sup.timeout_scope = "total" if (t0 + total_deadline_s) <= (stage_t + stage_limit) + 1e-6 else (
                    "load" if stage == "load" else f"expr:{cur_expr}")
                sup.rss_at_kill_mb = K.rss_mb(proc.pid)
                _kill(proc)
                break
            if parent_conn.poll(min(remaining, 0.25)):
                try:
                    msg = parent_conn.recv()
                except EOFError:
                    break                                                # the child is gone without a "done"
                except Exception as exc:                                 # an unpicklable message: record, keep going
                    sup.messages.append(("error", "recv", {"message": f"{type(exc).__name__}: {exc}"}))
                    continue
                if msg and msg[0] == "done":
                    sup.done = msg[1]
                    break
                sup.messages.append(msg)
                if msg and msg[0] == "stage":
                    name, payload = msg[1], msg[2] if len(msg) > 2 else {}
                    if name in ("imported", "consulted"):
                        stage_t = time.monotonic()
                    elif name == "loaded":
                        stage, stage_t = "expr", time.monotonic()
                    elif name == "expr_start":
                        stage, stage_t, cur_expr = "expr", time.monotonic(), payload.get("i")
            elif not proc.is_alive() and not parent_conn.poll(0):
                break
        proc.join(timeout=10)
        if proc.is_alive():
            _kill(proc)
        sup.exitcode = proc.exitcode
        sup.signal = _signal_name(proc.exitcode)
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        sup.wall_s = round(time.monotonic() - t0, 4)
        sup.stdout = _read_tail(tmpdir / "stdout.txt")
        sup.stderr = _read_tail(tmpdir / "stderr.txt")
        for f in ("stdout.txt", "stderr.txt"):
            try:
                (tmpdir / f).unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass
    return sup


def _kill(proc) -> None:
    try:
        if proc.is_alive():
            proc.kill()                                                   # SIGKILL
            proc.join(timeout=10)
    except Exception:
        pass


# ── the child (the only place that imports janus / petta) ────────────────────

def redirect_output(spec: dict) -> None:
    """Point file descriptors 1 and 2 at the supervisor's capture files, then rebuild sys.stdout/sys.stderr on them."""
    out_p, err_p = spec.get("stdout_path"), spec.get("stderr_path")
    if not out_p:
        return
    fo = open(out_p, "ab", buffering=0)
    fe = open(err_p, "ab", buffering=0)
    os.dup2(fo.fileno(), 1)
    os.dup2(fe.fileno(), 2)
    sys.stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=False), encoding="utf-8", errors="replace", write_through=True)
    sys.stderr = io.TextIOWrapper(io.FileIO(2, "w", closefd=False), encoding="utf-8", errors="replace", write_through=True)


def apply_env(spec: dict) -> None:
    """Environment for the janus / PeTTa import. Done in the child, before any import of them."""
    home = spec.get("petta_home")
    env = {
        "PETTA_PATH": home or "",
        "PETTA_HOME": home or "",
        "PETTA_LOAD_CACHE": spec.get("load_cache") or "",
        "PYTHONDONTWRITEBYTECODE": "1",                                   # never write .pyc into the PeTTa checkout
        "PATH": os.pathsep.join(["/opt/homebrew/bin", "/usr/local/bin", os.environ.get("PATH", "")]),
    }
    if spec.get("pycache_prefix"):
        env["PYTHONPYCACHEPREFIX"] = spec["pycache_prefix"]
    if home:
        env["PYTHONPATH"] = os.pathsep.join([str(Path(home) / "python")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    for k, v in env.items():
        if v:
            os.environ[k] = v
    for k, v in (spec.get("extra_env") or {}).items():
        os.environ[k] = str(v)
    sys.dont_write_bytecode = True
    if home:
        pydir = str(Path(home) / "python")
        if pydir not in sys.path:
            sys.path.insert(1, pydir)                                      # after the script dir, before site-packages


def bootstrap_petta(spec: dict, send: Optional[Callable] = None):
    """Import janus_swi and petta, construct PeTTa (consults main.pl), lower the stack limit.

    Returns (janus, PeTTa instance, info dict). Raises _Stage(stage, exc) naming where it failed.
    """
    info: dict = {}
    t = time.perf_counter()
    try:
        import janus_swi as janus                                          # starts the embedded SWI-Prolog
        info["janus_file"] = getattr(janus, "__file__", None)
        info["janus_version"] = getattr(janus, "__version__", None)
        from petta import PeTTa
        import petta as petta_mod
        info["petta_file"] = getattr(petta_mod, "__file__", None)
        home = spec.get("petta_home")
        if home and info["petta_file"] and not str(Path(info["petta_file"]).resolve()).startswith(str(Path(home).resolve())):
            raise ImportError(f"imported a different `petta` than {home}: {info['petta_file']}")
    except BaseException as exc:
        raise _Stage("import_error", exc) from exc
    info["import_s"] = round(time.perf_counter() - t, 4)
    info["rss_import_mb"] = K.rss_mb()
    if send:
        send("stage", "imported", info)
    t = time.perf_counter()
    try:
        p = PeTTa(verbose=False, petta_path=spec.get("petta_home"))        # janus.consult(main.pl), consult(helper.pl)
        limit_mb = spec.get("stack_limit_mb")
        if limit_mb:
            janus.query_once(f"set_prolog_flag(stack_limit, {int(limit_mb) * 1_000_000})")
    except BaseException as exc:
        raise _Stage("consult_error", exc) from exc
    info["consult_s"] = round(time.perf_counter() - t, 4)
    info["rss_consult_mb"] = K.rss_mb()
    try:
        info["swipl_version"] = str(janus.query_once("current_prolog_flag(version, V)").get("V"))
    except Exception:
        info["swipl_version"] = None
    if send:
        send("stage", "consulted", {k: info[k] for k in ("consult_s", "rss_consult_mb", "swipl_version")})
    return janus, p, info


class _Stage(Exception):
    def __init__(self, stage: str, exc: BaseException):
        super().__init__(f"{stage}: {type(exc).__name__}: {exc}")
        self.stage, self.exc = stage, exc


def finish_child(conn, payload: dict) -> None:
    """Send the final message and leave WITHOUT interpreter finalisation.

    janus / SWI-Prolog keep native threads and atexit hooks; letting Python finalise a process that embeds them can
    stall (and costs the supervisor a join timeout) after the answer is already delivered. Output is flushed first.
    """
    try:
        conn.send(("done", payload))
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass
    os._exit(0)


def exc_info(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)[:4000],
            "traceback_tail": "".join(traceback.format_exception_only(type(exc), exc))[-1000:]}


def helper_call(janus, name: str, argument: str):
    """PeTTa's own `_run_helper`, except that a FAILED goal is returned as None instead of being turned into []."""
    res = janus.query_once(
        "run_metta_helper(Verbose, HelperName, Argument, Results)",
        {"Verbose": "false", "HelperName": name, "Argument": argument},
    )
    if res is None:
        return None
    return [str(x) for x in res.get("Results", [])]


def verify_heads(janus, equations: dict) -> dict:
    """How many clauses does Prolog hold for each (= (f …) …) head of the KB?

    `equations`: {(name, arity): number of equations in the source}. A head with fewer clauses than equations is
    short; a head with none is missing. Never raises: a failure of the check itself is reported, not thrown.
    """
    out: dict = {"checked": len(equations), "defined": 0, "missing": [], "short": [], "error": None}
    if not equations:
        return out
    items = sorted(equations.items())
    heads = ", ".join(f"{K.quote_prolog_atom(n)}-{a}" for (n, a), _ in items)
    goal = (f"findall(S, (member(N-A, [{heads}]), functor(H, N, A), "
            f"( catch(predicate_property(H, number_of_clauses(C)), _, fail) -> true ; C = 0 ), "
            f"format(atom(S), '~w ~w ~w', [N, A, C])), Rows)")
    try:
        res = janus.query_once(goal)
        rows = res.get("Rows", []) if res else []
    except BaseException as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return out
    counts: dict = {}
    for r in rows:
        try:
            name, arity, c = str(r).rsplit(" ", 2)
            counts[(name, int(arity))] = int(c)
        except ValueError:
            continue
    for (name, arity), n_eq in items:
        c = counts.get((name, arity), 0)
        if c == 0:
            out["missing"].append(f"{name}/{arity}")
        else:
            out["defined"] += 1
            if c < n_eq:
                out["short"].append(f"{name}/{arity}: {c} of {n_eq} clauses")
    out["missing"], out["short"] = out["missing"][:40], out["short"][:40]
    out["n_missing"] = sum(1 for k in equations if counts.get(k, 0) == 0)
    out["n_short"] = sum(1 for k, n in equations.items() if 0 < counts.get(k, 0) < n)
    return out


def _cache_listing(cache_dir: str) -> list:
    try:
        return sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in Path(cache_dir).glob("*.snapshot"))
    except OSError:
        return []


def _child_entry(conn, spec: dict) -> None:
    """The spawned child: one KB load, then the expressions, one helper call each."""
    t_child = time.perf_counter()
    cpu0 = time.process_time()

    def send(*msg):
        try:
            conn.send(msg)
        except Exception:
            pass

    redirect_output(spec)
    out: dict = {"status": "ok", "answers_raw": [], "expr_errors": [], "expr_seconds": [], "load": {}, "timing": {}, "rss_mb": {}}
    try:
        if spec["engine"] == "hyperon":
            _run_hyperon(spec, out, send)
        else:
            _run_petta(spec, out, send)
    except _Stage as st:
        out["status"] = st.stage
        out["load"]["exception"] = exc_info(st.exc)
        out["error"] = f"{st.stage}: {type(st.exc).__name__}: {str(st.exc)[:500]}"
    except BaseException as exc:                                            # nothing may leave the child unreported
        out["status"] = "child_error"
        out["error"] = f"{type(exc).__name__}: {str(exc)[:500]}\n{traceback.format_exc()[-1500:]}"
    out["timing"]["child_s"] = round(time.perf_counter() - t_child, 4)
    out["timing"]["cpu_s"] = round(time.process_time() - cpu0, 4)
    out["peak_rss_mb"] = K.peak_rss_mb()
    finish_child(conn, out)


def _run_petta(spec: dict, out: dict, send) -> None:
    apply_env(spec)
    janus, petta, info = bootstrap_petta(spec, send)
    out["env"] = {k: info.get(k) for k in ("janus_file", "janus_version", "petta_file", "swipl_version")}
    out["timing"].update(import_s=info["import_s"], consult_s=info["consult_s"])
    out["rss_mb"].update(import_=info["rss_import_mb"], consult=info["rss_consult_mb"])

    kb_text, mode = spec["kb_text"], spec["mode"]
    load: dict = out["load"]
    load["mode"] = mode
    equations = K.kb_equations(kb_text) if spec.get("verify", True) else {}
    load["heads_expected"] = len(equations)
    load["equations_expected"] = sum(equations.values())

    path = None
    if mode in ("cached", "file"):
        path = Path(spec["kb_dir"]) / f"kb_{K.sha256_text(kb_text)[:16]}.metta"
        if not path.exists() or path.read_text(encoding="utf-8") != kb_text:
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(kb_text, encoding="utf-8")
            tmp.replace(path)
        load["kb_file"] = str(path)
        load["cache_before"] = _cache_listing(spec["load_cache"]) if mode == "cached" else None

    t = time.perf_counter()
    result, exc = None, None
    try:
        if not kb_text.strip():
            result = []
            load["skipped"] = "empty KB"
        elif spec.get("use_wrapper"):
            result = petta.process_metta_string(kb_text) if mode == "string" else (
                petta.load_metta_file_cached(str(path)) if mode == "cached" else petta.load_metta_file(str(path)))
            load["via"] = "PeTTa wrapper (a failed goal is indistinguishable from an empty result)"
        elif mode == "string":
            result = helper_call(janus, "process_metta_string", kb_text)
        elif mode == "cached":
            result = helper_call(janus, "load_metta_file_cached", str(path))
        elif mode == "file":
            result = helper_call(janus, "load_metta_file", str(path))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    except BaseException as e:                                              # includes janus.PrologError
        exc = e
    load["seconds"] = round(time.perf_counter() - t, 4)
    out["timing"]["load_s"] = load["seconds"]
    out["rss_mb"]["load"] = K.rss_mb()
    if mode == "cached":
        load["cache_after"] = _cache_listing(spec["load_cache"])
        # a miss writes (or rewrites) a snapshot; a hit leaves the directory exactly as it was
        load["cache_hit_guess"] = bool(load.get("cache_before")) and load["cache_after"] == load["cache_before"]
    load["n_results"] = len(result) if isinstance(result, list) else None
    if spec.get("verify", True):
        load["verify"] = verify_heads(janus, equations)
    if exc is not None:
        load["exception"] = exc_info(exc)
        out["status"] = "load_error"
        out["error"] = f"load raised {type(exc).__name__}: {str(exc)[:500]}"
        send("stage", "loaded", {"ok": False})
        if not spec.get("run_after_failed_load"):
            return
    elif result is None:
        load["failed_silently"] = True
        out["status"] = "load_failed"
        out["error"] = "the load goal FAILED without raising (PeTTa's wrapper would have returned [])"
        send("stage", "loaded", {"ok": False})
        if not spec.get("run_after_failed_load"):
            return
    else:
        load["failed_silently"] = False
        v = load.get("verify") or {}
        if spec.get("strict_verify") and (v.get("n_missing") or v.get("n_short")):
            out["status"] = "load_partial"
            out["error"] = f"{v.get('n_missing')} heads missing, {v.get('n_short')} short after a load that raised nothing"
            send("stage", "loaded", {"ok": False})
            return
        send("stage", "loaded", {"ok": True, "seconds": load["seconds"]})

    # one helper call per `!` expression
    exprs = spec["expressions"]
    q0 = time.perf_counter()
    for i, expr in enumerate(exprs):
        send("stage", "expr_start", {"i": i})
        t = time.perf_counter()
        err = None
        try:
            res = helper_call(janus, "process_metta_string", expr) if not spec.get("use_wrapper") else petta.process_metta_string(expr)
            answers = res if res is not None else []
            if res is None:
                err = {"type": "GoalFailed", "message": "the helper goal failed without raising (PeTTa would return [])"}
        except BaseException as e:
            answers, err = [], exc_info(e)
        out["expr_seconds"].append(round(time.perf_counter() - t, 5))
        out["answers_raw"].append([str(a) for a in answers])
        out["expr_errors"].append(err)
        send("stage", "expr_done", {"i": i, "answers": out["answers_raw"][-1], "error": err, "seconds": out["expr_seconds"][-1]})
    out["timing"]["queries_s"] = round(time.perf_counter() - q0, 4)
    out["rss_mb"]["end"] = K.rss_mb()


def _run_hyperon(spec: dict, out: dict, send) -> None:
    """The same plumbing on hyperon, for self-tests only."""
    t = time.perf_counter()
    import hyperon
    from hyperon import MeTTa
    out["timing"]["import_s"] = round(time.perf_counter() - t, 4)
    out["env"] = {"hyperon": getattr(hyperon, "__version__", "?")}
    send("stage", "imported", {"import_s": out["timing"]["import_s"]})
    t = time.perf_counter()
    m = MeTTa()
    out["timing"]["consult_s"] = round(time.perf_counter() - t, 4)
    send("stage", "consulted", {"consult_s": out["timing"]["consult_s"]})
    t = time.perf_counter()
    if spec["kb_text"].strip():
        m.run(spec["kb_text"])
    out["load"] = {"mode": spec["mode"], "seconds": round(time.perf_counter() - t, 4), "failed_silently": False,
                   "note": "hyperon engine: file/cached modes are the same as string"}
    out["timing"]["load_s"] = out["load"]["seconds"]
    out["rss_mb"]["load"] = K.rss_mb()
    send("stage", "loaded", {"ok": True})
    q0 = time.perf_counter()
    for i, expr in enumerate(spec["expressions"]):
        send("stage", "expr_start", {"i": i})
        t = time.perf_counter()
        try:
            groups = m.run(expr)
            out["answers_raw"].append([str(a) for g in groups for a in g])
            out["expr_errors"].append(None)
        except BaseException as e:
            out["answers_raw"].append([])
            out["expr_errors"].append(exc_info(e))
        out["expr_seconds"].append(round(time.perf_counter() - t, 5))
        send("stage", "expr_done", {"i": i, "answers": out["answers_raw"][-1], "error": out["expr_errors"][-1], "seconds": out["expr_seconds"][-1]})
    out["timing"]["queries_s"] = round(time.perf_counter() - q0, 4)
    out["rss_mb"]["end"] = K.rss_mb()


# ── the public API (parent side) ─────────────────────────────────────────────

def _child_spec(kb_text: str, expressions: list[str], *, engine: str, mode: str, stack_limit_mb: Optional[int],
                verify: bool, strict_verify: bool, use_wrapper: bool, run_after_failed_load: bool,
                extra_env: Optional[dict], load_cache: Optional[str] = None) -> dict:
    cache = Path(load_cache) if load_cache else K.load_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "engine": engine, "mode": mode, "kb_text": kb_text, "expressions": list(expressions),
        "petta_home": str(K.petta_home()), "load_cache": str(cache), "kb_dir": str(K.kb_cache_dir()),
        "pycache_prefix": str(K.out_dir() / "pycache"), "stack_limit_mb": stack_limit_mb,
        "verify": verify, "strict_verify": strict_verify, "use_wrapper": use_wrapper,
        "run_after_failed_load": run_after_failed_load, "extra_env": extra_env or {},
    }


def run_program_report(
    kb_text: str,
    expressions: list[str],
    *,
    engine: str = "petta",
    mode: str = "string",
    deadline_s: float = DEFAULT_DEADLINE_S,
    load_deadline_s: Optional[float] = None,
    expr_deadline_s: Optional[float] = None,
    stack_limit_mb: Optional[int] = DEFAULT_STACK_LIMIT_MB,
    verify: bool = True,
    strict_verify: bool = False,
    use_wrapper: bool = False,
    run_after_failed_load: bool = False,
    extra_env: Optional[dict] = None,
    load_cache: Optional[str] = None,
    bang: bool = True,
    label: str = "",
) -> RunReport:
    """Load `kb_text` in a fresh process and run each expression; return everything that was observed.

    expressions: top-level MeTTa expressions, each normally starting with '!' (an expression without it is
    sent as '!' + expression, as pln_runner._normalize_query does). mode: 'string' | 'cached' | 'file'.
    load_cache: the directory for load_metta_file_cached snapshots (default $PETTA_LOAD_CACHE or out/petta_load_cache).
    bang=False sends the expressions untouched (a definition or a second KB text for process_metta_string).
    """
    exprs = [e if (not bang or e.lstrip().startswith("!")) else "!" + e.lstrip() for e in expressions]
    t0 = time.perf_counter()
    spec = _child_spec(kb_text, exprs, engine=engine, mode=mode, stack_limit_mb=stack_limit_mb, verify=verify,
                       strict_verify=strict_verify, use_wrapper=use_wrapper, run_after_failed_load=run_after_failed_load,
                       extra_env=extra_env, load_cache=load_cache)
    sup = spawn_supervised(_child_entry, spec, total_deadline_s=deadline_s, load_deadline_s=load_deadline_s,
                           expr_deadline_s=expr_deadline_s)
    rep = RunReport(engine=engine, mode=mode, label=label, kb_bytes=len(kb_text.encode("utf-8")), kb_sha256=K.sha256_text(kb_text),
                    n_expressions=len(exprs), exitcode=sup.exitcode, signal=sup.signal, timeout_scope=sup.timeout_scope,
                    stdout=sup.stdout, stderr=sup.stderr)
    rep.stages = [{"stage": m[1], **(m[2] if len(m) > 2 and isinstance(m[2], dict) else {})}
                  for m in sup.messages if m and m[0] == "stage" and m[1] != "expr_done"]
    done = sup.done
    if done is not None:
        rep.status = done.get("status", "ok")
        rep.error = done.get("error")
        rep.load = done.get("load", {})
        rep.timing = dict(done.get("timing", {}))
        rep.rss_mb = done.get("rss_mb", {})
        rep.peak_rss_mb = done.get("peak_rss_mb")
        rep.env = done.get("env", {})
        raw = done.get("answers_raw", [])
        rep.answers = [[Answer(r, K.normalise_answer(r)) for r in group] for group in raw]
        while len(rep.answers) < len(exprs):
            rep.answers.append([])
        rep.expr_errors = list(done.get("expr_errors", [])) + [None] * (len(exprs) - len(done.get("expr_errors", [])))
        rep.expr_seconds = done.get("expr_seconds", [])
        rep.completed = list(range(len(done.get("answers_raw", []))))
    else:
        rep.answers = [[] for _ in exprs]
        rep.expr_errors = [None] * len(exprs)
        rep.expr_seconds = [0.0] * len(exprs)
        for m in sup.messages:                                   # the expressions that finished before the child died
            if m and m[0] == "stage" and m[1] == "expr_done" and len(m) > 2:
                d, i = m[2], m[2].get("i")
                if isinstance(i, int) and 0 <= i < len(exprs):
                    rep.answers[i] = [Answer(r, K.normalise_answer(r)) for r in d.get("answers", [])]
                    rep.expr_errors[i] = d.get("error")
                    rep.expr_seconds[i] = d.get("seconds", 0.0)
                    rep.completed.append(i)
        if sup.timeout_scope:
            rep.status = "timeout"
            rep.error = (f"killed (SIGKILL) after the {sup.timeout_scope} deadline; RSS at kill {sup.rss_at_kill_mb} MB; "
                         f"last stage: {rep.stages[-1] if rep.stages else 'none'}")
        else:
            rep.status = "crash"
            rep.error = (f"the child exited with code {sup.exitcode}"
                         + (f" ({sup.signal})" if sup.signal else "") + " and sent no result"
                         + (f"; stderr tail: {sup.stderr[-300:]!r}" if sup.stderr else ""))
        rep.load = {"mode": mode}
        rep.rss_mb = {"at_kill": sup.rss_at_kill_mb} if sup.rss_at_kill_mb is not None else {}
        for m in sup.messages:                                   # whatever the child reported before it died
            if m and m[0] == "stage" and len(m) > 2 and isinstance(m[2], dict):
                for k in ("import_s", "consult_s"):
                    if k in m[2]:
                        rep.timing[k] = m[2][k]
    rep.timing["wall_s"] = round(time.perf_counter() - t0, 4)
    return rep


def run_program(kb_text: str, expressions: list[str], **kw) -> list[list[str]]:
    """The seam's shape: load `kb_text`, run each expression, return one list of normalised answer strings per expression.

    Raises PettaRunError (with the RunReport on `.report`) when the load fails, the run times out or crashes,
    or any expression raises. The raw PeTTa strings are on the report: `PettaRunError.report.raw_answers()`,
    or call run_program_report() to get them on success too.
    """
    rep = run_program_report(kb_text, expressions, **kw)
    if rep.status != "ok":
        raise PettaRunError(rep)
    if rep.n_expr_errors:
        raise PettaRunError(rep, f"expression raised: {rep.first_error()}")
    run_program.last_report = rep                                # type: ignore[attr-defined]
    return rep.normalised()


run_program.last_report = None                                  # type: ignore[attr-defined]


def available() -> dict:
    """What can be said about the machine WITHOUT importing PeTTa or janus (nothing is executed)."""
    home = K.petta_home()
    rep = K.env_report()
    rep["main_pl"] = (home / "src" / "main.pl").exists()
    rep["helper_pl"] = (home / "python" / "helper.pl").exists()
    rep["petta_wrapper"] = (home / "python" / "petta" / "__init__.py").exists()
    try:
        import importlib.util
        rep["janus_swi_importable"] = importlib.util.find_spec("janus_swi") is not None
        rep["hyperon_importable"] = importlib.util.find_spec("hyperon") is not None
    except Exception:
        rep["janus_swi_importable"] = None
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(available(), indent=1))
