"""Run MeTTa work OUT OF PROCESS, with a real timeout and a real concurrency bound.

Why a thread pool is not enough
-------------------------------
Every route handler in `api.py` is a plain `def`, so Starlette already runs it in
anyio's worker threadpool (40 slots). That is not the problem. Measured on this
codebase (hyperon 0.2.10, CPython 3.11):

    one MeTTa program                       2.46 s
    the same program twice, serially        4.83 s
    the same program in two threads         4.83 s   <- ratio 0.998

hyperon holds the GIL for the whole of `MeTTa.run()`. A canary thread doing
`time.sleep(0.005)` in a loop got 3 ticks (of ~386 expected) during one 2.3 s
MeTTa call. So while a ranking runs, the event loop — and every other request,
including `GET /health` — is starved. Reproduced live against uvicorn: a
`/health` issued 0.4 s into a 3.87 s MeTTa request took 3.47 s. That is the
"one large request freezes the API for 115 s" finding, and no threadpool size,
`async def` conversion or `anyio.fail_after` can fix it: you cannot preempt a
Rust call that holds the GIL.

Three things follow, and all three need a separate PROCESS:

1. **Concurrency.** With MeTTa in a child process the parent's event loop keeps
   serving. Verified: `/health` during a 3.73 s hyperon call took 0.007 s.
2. **Timeouts.** A runaway query can be abandoned — `future.result(timeout=...)`
   returns to the caller and the worker is discarded. In-process this is
   impossible.
3. **Crash containment.** hyperon 0.2.10 aborts the whole interpreter with a
   NON-UNWINDING Rust panic ("called `Option::unwrap()` on a `None` value" in
   hyperon-space's trie) once a space holds too many DISTINCT HEAD SYMBOLS.
   Measured, and asserted by `tests/test_kb_head_symbol_budget.py`: 400 atoms
   under ONE new head symbol load and query fine, while 4 atoms under 4 NEW head
   symbols abort. Atom count is not the axis; the number of distinct predicates
   is. `except Exception` cannot catch that — in-process it kills the uvicorn
   worker. In a child process it surfaces as `BrokenProcessPool`, which is an
   ordinary exception the API can answer with a 500.

Inline mode
-----------
`PLN_WORKER_POOL_SIZE=0` runs everything inline, exactly as before. That is what
the in-process contract tests use: they monkeypatch `api.run_query`, and a child
process re-imports the module fresh and would never see the patch. Inline mode
calls the caller-supplied thunk, so the monkeypatch is honoured.
"""
from __future__ import annotations

import atexit
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from config import (
    PLN_MAX_INFLIGHT_QUERIES,
    PLN_QUERY_TIMEOUT_SECONDS,
    PLN_WORKER_POOL_SIZE,
)

T = TypeVar("T")


class PLNExecutionTimeout(Exception):
    """A MeTTa task exceeded its per-request budget and was abandoned."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(
            f"PLN execution exceeded the {timeout_s:g}s per-request budget and was "
            f"cancelled. Narrow the query (fewer compounds, a smaller candidate "
            f"pool) or raise PLN_QUERY_TIMEOUT_SECONDS."
        )
        self.timeout_s = timeout_s


class PLNWorkerCrashed(Exception):
    """The MeTTa worker process died — almost always a hyperon abort."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "The PLN worker process terminated while running this query. hyperon "
            "0.2.10 aborts the interpreter (a non-unwinding Rust panic) once the "
            "loaded space holds too many DISTINCT HEAD SYMBOLS -- NOT too many "
            "rows: 400 atoms under one new head symbol are fine, 4 atoms under 4 "
            "new head symbols are not. The usual cause is a generated ETL file "
            "left in the repository root, which the runtime auto-loads; check "
            "GET /ontology/files against what the build should contain. The "
            "worker has been replaced and the service is still up. "
            + detail
        )


class PLNOverloaded(Exception):
    """Too many MeTTa tasks are already queued or running."""

    def __init__(self, limit: int) -> None:
        super().__init__(
            f"Too many PLN queries in flight (limit {limit}). PLN execution is "
            f"CPU-bound and single-threaded per worker; retry shortly."
        )
        self.limit = limit


# ── The task registry ────────────────────────────────────────────────────────
# A child process cannot receive a closure or a Mock, so offloadable work is
# named. Each entry resolves its implementation INSIDE the child, at call time.

def _task_run_query(**kwargs) -> Any:
    from core.pln_runner import run_query
    return run_query(**kwargs)


def _task_rank_drugage(**kwargs) -> Any:
    from core.drugage_router import rank_drugage
    return rank_drugage(**kwargs)


def _task_route_drugage_ranking(**kwargs) -> Any:
    from core.drugage_router import route_drugage_ranking
    return route_drugage_ranking(**kwargs)


def _task_drugage_top(**kwargs) -> Any:
    from core.drugage_router import drugage_top
    return drugage_top(**kwargs)


def _task_cellage_effects(**kwargs) -> Any:
    from core.pln_runner import run_cellage_effects
    return run_cellage_effects(**kwargs)


_TASKS: dict[str, Callable[..., Any]] = {
    "run_query": _task_run_query,
    "rank_drugage": _task_rank_drugage,
    "route_drugage_ranking": _task_route_drugage_ranking,
    "drugage_top": _task_drugage_top,
    "cellage_effects": _task_cellage_effects,
}


def _dispatch(task: str, kwargs: dict) -> Any:
    """The child-side entry point (module level so it is picklable)."""
    return _TASKS[task](**kwargs)


def _child_init(sys_path_entry: str) -> None:
    """Make `core.*` importable in a spawned child (a forked one inherits it)."""
    import sys

    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)


# ── One process per query ────────────────────────────────────────────────────
# NOT a shared ProcessPoolExecutor. A pool looks like the obvious fit and is the
# wrong one here, for a reason that only shows up under concurrency: a pool has
# no way to kill ONE task. `future.cancel()` does nothing to a task that has
# already started, and the only lever that stops a GIL-holding Rust call is a
# signal to the process running it — which a pool does not let you aim.
#
# Measured, with a shared pool: while an eight-compound ranking was running
# normally (3 s, well inside its budget), a DIFFERENT caller's runaway query hit
# its 1 s deadline. Killing the pool to stop the runaway killed the healthy
# request too, and that caller got a 500 blaming a hyperon abort that never
# happened to them.
#
# Forking one process per query makes the deadline exact: the timeout kills the
# process running THAT query and nothing else. On Linux the fork is copy-on-
# write and costs single-digit milliseconds against a 10-600 ms query, which is
# a cheap price for not failing other people's requests. `fork` also avoids the
# __main__ re-import that spawn/forkserver would do in every child — under
# `python app.py` that would rebuild the whole Gradio UI per query.

_inflight = 0
_inflight_lock = threading.Lock()
_slots: Optional[threading.BoundedSemaphore] = None
_slots_size = -1
_slots_lock = threading.Lock()


def _mp_context():
    """fork where available; spawn only as a last resort (see above)."""
    for name in ("fork", "forkserver", "spawn"):
        try:
            return multiprocessing.get_context(name)
        except ValueError:          # pragma: no cover - platform dependent
            continue
    return multiprocessing          # pragma: no cover


def pool_size() -> int:
    """Max queries running at once; 0 means inline execution."""
    return max(0, PLN_WORKER_POOL_SIZE)


def _acquire_slot(timeout: float) -> bool:
    """Bound concurrent MeTTa processes, since each one pins a core."""
    global _slots, _slots_size
    with _slots_lock:
        if _slots is None or _slots_size != pool_size():
            _slots = threading.BoundedSemaphore(max(1, pool_size()))
            _slots_size = pool_size()
        slots = _slots
    return slots.acquire(timeout=max(0.0, timeout))


def _release_slot() -> None:
    with _slots_lock:
        slots = _slots
    if slots is not None:
        try:
            slots.release()
        except ValueError:          # pragma: no cover - resized mid-flight
            pass


def _child_main(conn, task: str, kwargs: dict) -> None:
    """The forked child: run one task, send one message back, exit."""
    try:
        conn.send(("ok", _dispatch(task, kwargs)))
    except BaseException as exc:    # noqa: BLE001 - relayed to the parent verbatim
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            conn.close()
        except Exception:           # pragma: no cover
            pass


def _terminate(proc) -> None:
    """SIGKILL one worker.

    Not SIGTERM: the child is inside a Rust call that does not check signals
    politely, and a runaway MeTTa evaluation grows without bound while it waits
    (measured: 3.7 GB resident within two minutes).
    """
    try:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
    except Exception:               # pragma: no cover - never mask the real error
        pass


def shutdown() -> None:
    """Nothing to release: every query's process exits with the query."""
    return None


def _effective_inflight_limit() -> int:
    if PLN_MAX_INFLIGHT_QUERIES > 0:
        return PLN_MAX_INFLIGHT_QUERIES
    return max(1, pool_size() * 4)


def executor_stats() -> dict:
    """What `GET /health` reports about PLN execution."""
    inline = pool_size() == 0
    return {
        "mode": "inline" if inline else "process_per_query",
        "max_concurrent": pool_size(),
        # A deadline and an admission limit are only enforceable out of process.
        # Saying so is the point: inline mode is the escape hatch, and a caller
        # reading /health should not believe it is protected when it is not.
        "timeout_seconds": None if inline else PLN_QUERY_TIMEOUT_SECONDS,
        "timeout_enforced": not inline,
        "max_inflight": None if inline else _effective_inflight_limit(),
        "admission_control": not inline,
        "inflight": _inflight,
    }


def run_offloaded(
    task: str,
    kwargs: dict,
    inline: Callable[[], T],
    *,
    timeout: Optional[float] = None,
) -> T:
    """Run `task` in a dedicated worker process, or `inline()` when disabled.

    Parameters
    ----------
    task:
        A key of `_TASKS`. Its `kwargs` must be picklable (paths, strings,
        numbers, lists) — everything the PLN entry points take already is.
    inline:
        A zero-argument thunk performing the SAME work in-process. Used when
        `PLN_WORKER_POOL_SIZE` is 0, and it is what keeps the in-process
        contract tests (which monkeypatch module globals) meaningful.
    timeout:
        Per-request budget; defaults to `PLN_QUERY_TIMEOUT_SECONDS`. 0 disables.

    Raises
    ------
    PLNOverloaded, PLNExecutionTimeout, PLNWorkerCrashed
        Mapped to 503 / 504 / 500 by the HTTP layer. Nothing else escapes: an
        exception raised by the task itself comes back as PLNWorkerCrashed with
        its original type name in the message, so a caller never gets a bare
        500 from a path this function was supposed to classify.
    """
    if task not in _TASKS:
        raise KeyError(f"unknown offload task: {task!r}")

    if pool_size() == 0:
        return inline()

    global _inflight
    limit = _effective_inflight_limit()
    with _inflight_lock:
        if _inflight >= limit:
            raise PLNOverloaded(limit)
        _inflight += 1

    budget = PLN_QUERY_TIMEOUT_SECONDS if timeout is None else timeout
    budget = budget if budget and budget > 0 else None

    slot = False
    proc = None
    parent_conn = None
    try:
        # Waiting for a slot is part of the budget, not extra to it.
        slot = _acquire_slot(budget if budget is not None else 30.0)
        if not slot:
            raise PLNOverloaded(pool_size())

        ctx = _mp_context()
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_main, args=(child_conn, task, kwargs), daemon=True
        )
        started = time.monotonic()
        proc.start()
        child_conn.close()          # the parent keeps only the read end

        remaining = None if budget is None else max(
            0.0, budget - (time.monotonic() - started)
        )
        if not parent_conn.poll(remaining):
            # Still inside a GIL-holding Rust call, so it cannot be asked to
            # stop. Kill THIS query's process — and only this one.
            _terminate(proc)
            raise PLNExecutionTimeout(budget or 0.0)

        try:
            status, payload = parent_conn.recv()
        except EOFError:
            # The child died without sending anything: a hyperon abort.
            exit_code = proc.exitcode
            _terminate(proc)
            raise PLNWorkerCrashed(
                f"worker exited with code {exit_code} and no result"
            ) from None

        proc.join(timeout=5)
        if proc.is_alive():         # pragma: no cover - the child already answered
            _terminate(proc)

        if status == "ok":
            return payload
        raise PLNWorkerCrashed(f"the query raised in the worker: {payload}")
    finally:
        if parent_conn is not None:
            try:
                parent_conn.close()
            except Exception:       # pragma: no cover
                pass
        if proc is not None and proc.is_alive():
            _terminate(proc)
        if slot:
            _release_slot()
        with _inflight_lock:
            _inflight -= 1


# Kept so an importer that calls it keeps working; there is nothing to release
# now that a query's process exits with the query.
atexit.register(shutdown)
