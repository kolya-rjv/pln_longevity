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
   hyperon-space's trie) once a space passes a few hundred rows in certain query
   shapes. `except Exception` cannot catch that — in-process it kills the uvicorn
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
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from config import (
    PLN_MAX_INFLIGHT_QUERIES,
    PLN_QUERY_TIMEOUT_SECONDS,
    PLN_WORKER_MAX_TASKS,
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
            "0.2.10 aborts the interpreter (a non-unwinding Rust panic) on some "
            "query shapes once a space grows past a few hundred rows; the worker "
            "has been replaced and the service is still up. "
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


# ── Pool lifecycle ───────────────────────────────────────────────────────────

_pool_lock = threading.Lock()
_pool: Optional[ProcessPoolExecutor] = None
_pool_tasks = 0                      # tasks run by the CURRENT pool
_inflight = 0
_inflight_lock = threading.Lock()


def _mp_context(name: str):
    try:
        return multiprocessing.get_context(name)
    except ValueError:      # pragma: no cover - start method unavailable here
        return None


def _new_pool(size: int) -> ProcessPoolExecutor:
    """Build the worker pool.

    `fork` is deliberately preferred over `spawn`/`forkserver`. Those two
    RE-IMPORT the parent's __main__ module in every child: under `python app.py`
    that rebuilds the whole Gradio UI per worker, and under a bare script it can
    fail outright. Forked children inherit sys.path and the loaded modules and
    start in milliseconds. The cost is that CPython refuses
    `max_tasks_per_child` under `fork`, so worker recycling is done by hand in
    `run_offloaded` (discard the pool every PLN_WORKER_MAX_TASKS tasks) instead.
    """
    base: dict[str, Any] = {
        "max_workers": size,
        "initializer": _child_init,
        "initargs": (str(Path(__file__).resolve().parent.parent),),
    }
    for name in ("fork", "forkserver", "spawn"):
        ctx = _mp_context(name)
        if ctx is not None:
            return ProcessPoolExecutor(mp_context=ctx, **base)
    return ProcessPoolExecutor(**base)   # pragma: no cover - last resort


def pool_size() -> int:
    """Configured worker count; 0 means inline execution."""
    return max(0, PLN_WORKER_POOL_SIZE)


def _get_pool() -> ProcessPoolExecutor:
    global _pool, _pool_tasks
    with _pool_lock:
        if _pool is None:
            _pool = _new_pool(pool_size())
            _pool_tasks = 0
        return _pool


def _note_task_done() -> None:
    """Recycle the pool every PLN_WORKER_MAX_TASKS tasks (see _new_pool)."""
    global _pool_tasks
    if PLN_WORKER_MAX_TASKS <= 0:
        return
    with _pool_lock:
        _pool_tasks += 1
        recycle = _pool_tasks >= PLN_WORKER_MAX_TASKS
    if recycle:
        _discard_pool()


def _kill_workers(pool: ProcessPoolExecutor) -> None:
    """SIGKILL every worker of `pool`.

    Abandoning a timed-out task is NOT enough. `shutdown(wait=False)` leaves the
    running child alive, and a runaway MeTTa evaluation is not merely slow — a
    divergent recursion pins a core at 100% and grows without bound (measured:
    3.7 GB resident and climbing within ~2 minutes). Since the child is inside a
    GIL-holding Rust call it cannot be asked to stop politely, so the deadline
    has to be enforced with a signal.
    """
    processes = getattr(pool, "_processes", None) or {}
    for proc in list(processes.values()):
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:   # noqa: BLE001 - never mask the original failure
            pass


def _discard_pool(*, kill: bool = False) -> None:
    """Throw the pool away after a timeout or a worker abort, and start fresh."""
    global _pool, _pool_tasks
    with _pool_lock:
        dead, _pool = _pool, None
        _pool_tasks = 0
    if dead is None:
        return
    if kill:
        _kill_workers(dead)
    # Do not wait: the point of the deadline is to stop blocking the caller.
    try:
        dead.shutdown(wait=False, cancel_futures=True)
    except Exception:   # noqa: BLE001 - shutdown must never mask the real error
        pass


def shutdown() -> None:
    """Release the pool (tests, and a clean process exit)."""
    _discard_pool(kill=True)


def executor_stats() -> dict:
    """What `GET /health` reports about PLN execution."""
    return {
        "mode": "inline" if pool_size() == 0 else "process_pool",
        "workers": pool_size(),
        "timeout_seconds": PLN_QUERY_TIMEOUT_SECONDS,
        "max_inflight": _effective_inflight_limit(),
        "inflight": _inflight,
        "pool_started": _pool is not None,
    }


def _effective_inflight_limit() -> int:
    if PLN_MAX_INFLIGHT_QUERIES > 0:
        return PLN_MAX_INFLIGHT_QUERIES
    return max(1, pool_size() * 4)


# ── The one function callers use ─────────────────────────────────────────────

def run_offloaded(
    task: str,
    kwargs: dict,
    inline: Callable[[], T],
    *,
    timeout: Optional[float] = None,
) -> T:
    """Run `task` in a worker process, or `inline()` when the pool is disabled.

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
        Mapped to 503 / 504 / 500 by the HTTP layer.
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
    try:
        budget = PLN_QUERY_TIMEOUT_SECONDS if timeout is None else timeout
        try:
            future: Future = _get_pool().submit(_dispatch, task, kwargs)
        except BrokenExecutor:
            _discard_pool()
            future = _get_pool().submit(_dispatch, task, kwargs)
        try:
            out = future.result(timeout=budget if budget and budget > 0 else None)
            _note_task_done()
            return out
        except FutureTimeout:
            # The worker is still inside a GIL-holding Rust call, so it cannot be
            # interrupted — kill it and let the next request build a clean pool.
            _discard_pool(kill=True)
            raise PLNExecutionTimeout(budget) from None
        except BrokenExecutor as exc:
            _discard_pool(kill=True)
            raise PLNWorkerCrashed(str(exc)) from None
    finally:
        with _inflight_lock:
            _inflight -= 1


# Best-effort cleanup so a reloading dev server does not leak workers.
atexit.register(shutdown)
