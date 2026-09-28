"""What actually aborts hyperon 0.2.10, and the guard that keeps it from recurring.

The 2026-09-28 re-test reported every PLN inference query returning HTTP 500
`pln_worker_crashed` — risk, supplements, counterfactuals, ranking and every
caller-supplied patient — against a v2.0.0 deployment. The report attributed it
to the knowledge base growing "past the row count at which hyperon 0.2.10
aborts", which is what the service itself said: `PLNWorkerCrashed` claimed a
space "past a few hundred rows".

That attribution is wrong, and it matters, because it points at the wrong fix.
Measured against the committed KB (and asserted below):

    400 atoms under ONE new head symbol      -> loads and queries fine
      4 atoms under FOUR new head symbols    -> non-unwinding panic, process dies

The axis is the number of DISTINCT HEAD SYMBOLS in the space, not the number of
atoms, rows or facts in it. Trimming rows buys nothing; adding one small file
that introduces a handful of new predicates costs everything. The committed KB
runs every reported query correctly, so the failure was never in the shipped
code: it was a space with extra head symbols in it, and the way that happens is
a generated ETL file left in the repository root, which `_runtime_kb_paths()`
auto-loads. `scripts/run_etl.sh` now refuses to write there.

An abort is NOT a Python exception — it is `abort()` inside Rust, uncatchable by
`except` and fatal to the interpreter. So every probe here runs in a SUBPROCESS,
and the assertion is on the child's exit status. That is also why a plain
pytest run cannot be trusted to notice this on its own: in-process it kills the
run with "Fatal Python error: Aborted" and no failing test to point at.

Run from the repository root:
    pytest tests/test_kb_head_symbol_budget.py -q
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytest.importorskip("hyperon", reason="the PLN runtime is not installed")

#: Mirrors config.PLN_MAX_KB_FILE_BYTES / api._runtime_kb_paths(): the app loads
#: every repo-root .metta under this size into ONE hyperon space.
MAX_KB_FILE_BYTES = 60_000

#: The query the re-test found broken first. It is also the cheapest end-to-end
#: exercise of the inference stack: patient -> markers -> deduction -> risk.
CANARY_QUERY = "!(predict-risk-patient &self Patient001)"

_CHILD = r"""
import sys
from pathlib import Path
from hyperon import MeTTa

repo, query, extra = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
metta = MeTTa()
for path in sorted(repo.glob("*.metta")):
    if path.stat().st_size <= {max_bytes}:
        metta.run(path.read_text(encoding="utf-8"))
if extra:
    metta.run(extra)
result = metta.run(query)
if not result or not result[0]:
    print("EMPTY")
    raise SystemExit(3)
print("OK")
""".replace("{max_bytes}", str(MAX_KB_FILE_BYTES))


def _run(query: str = CANARY_QUERY, extra: str = "") -> subprocess.CompletedProcess:
    """Load the real runtime KB (plus `extra`) and run `query`, out of process."""
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(REPO), query, extra],
        capture_output=True, text=True, timeout=300,
    )


def _new_head_symbols(count: int) -> str:
    """`count` atoms, each under a head symbol the KB has never seen."""
    return "\n".join(
        f"(ProbeHead{i} ProbeRow{i} ProbeValue{i})" for i in range(count)
    )


def _one_head_symbol(rows: int) -> str:
    """`rows` atoms, all under a SINGLE head symbol the KB has never seen."""
    return "\n".join(
        f"(ProbeSingleHead ProbeRow{i} ProbeValue{i})" for i in range(rows)
    )


# ── The regression the re-test reported ──────────────────────────────────────

def test_inference_runs_on_the_committed_kb():
    """The headline finding, asserted: this query must not crash a worker.

    Reported as a 500 for Patient001, Patient002 and Patient003 alike. If this
    fails, the personalized stack is down and no amount of endpoint polish
    substitutes for it.
    """
    done = _run()
    assert done.returncode == 0, (
        f"`{CANARY_QUERY}` did not survive the committed KB "
        f"(exit {done.returncode}).\n"
        f"An exit of -6 is hyperon's non-unwinding abort: something added head "
        f"symbols to the runtime space. Check for generated .metta files in the "
        f"repository root.\nstderr tail:\n{done.stderr[-2000:]}"
    )


@pytest.mark.parametrize(
    "query",
    [
        "!(predict-risk-patient &self Patient002)",
        "!(recommend-supplements-patient &self Patient001)",
        "!(match &self (PartOf $c GrimAge) $c)",
    ],
)
def test_the_other_reported_shapes_run(query):
    """Risk for a second patient, a supplement ranking, and a flat retrieval."""
    done = _run(query=query)
    assert done.returncode == 0, (
        f"`{query}` exited {done.returncode}.\nstderr tail:\n{done.stderr[-2000:]}"
    )


# ── What the limit is, and what it is not ────────────────────────────────────

def test_atom_count_is_not_the_limit():
    """400 atoms under ONE new head symbol are harmless.

    This is the assertion that stops the wrong fix. A future reader who believes
    the "past a few hundred rows" story will try to buy headroom by trimming
    rows, and will spend the effort for nothing.
    """
    done = _run(extra=_one_head_symbol(400))
    assert done.returncode == 0, (
        "400 atoms under a single new head symbol aborted the interpreter. If "
        "this is now genuinely failing, the model of the limit in this file — "
        "and in core/executor.py — needs re-measuring.\n"
        f"stderr tail:\n{done.stderr[-2000:]}"
    )


def test_distinct_head_symbols_are_the_limit():
    """Twelve new head symbols, twelve atoms, and the interpreter dies.

    Twelve rather than the measured threshold of four, so that this stays a
    statement about the mechanism and does not turn red merely because the KB
    shrank. A generated ETL file introduces a head symbol per field predicate,
    so it lands far past this.
    """
    done = _run(extra=_new_head_symbols(12))
    assert done.returncode != 0, (
        "12 new head symbols no longer abort hyperon. That is good news and "
        "makes this file obsolete: re-measure the ceiling, then relax "
        "PLN_MAX_KB_FILE_BYTES and the run_etl.sh guard accordingly."
    )


# ── The way it actually happens ──────────────────────────────────────────────

def test_repo_root_holds_no_generated_metta():
    """Every .metta the runtime auto-loads must be a committed, curated one.

    This is the cheap guard for the real incident: `run_etl.sh` used to document
    `OUT_DIR=.`, and a generated file in the root is auto-loaded into the shared
    space, taking every inference query to a 500. No hyperon needed.
    """
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "*.metta"],
            cwd=REPO, capture_output=True, text=True, timeout=60, check=True,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available to tell committed files from generated ones")

    committed = {Path(name).name for name in tracked if "/" not in name}
    present = {p.name for p in REPO.glob("*.metta")}
    stray = sorted(present - committed)
    assert not stray, (
        f"Untracked .metta in the repository root: {stray}. The runtime "
        f"auto-loads every one of them into the shared hyperon space, and a "
        f"generated file's head symbols abort it on the next inference query. "
        f"Write ETL output to ./build (the run_etl.sh default) instead."
    )
