"""What actually aborts hyperon 0.2.10, and the guards that keep it from recurring.

The 2026-09-28 re-test reported every PLN inference query returning HTTP 500
`pln_worker_crashed` — risk, supplements, counterfactuals, ranking and every
caller-supplied patient — against a v2.0.0 deployment. The report attributed it
to the knowledge base growing "past the row count at which hyperon 0.2.10
aborts", which is what the service itself said: `PLNWorkerCrashed` claimed a
space "past a few hundred rows".

That attribution is wrong, and it matters, because it points at the wrong fix.
The axis is the number of DISTINCT HEAD SYMBOLS in a space, not the number of
atoms, rows or facts in it. Trimming rows buys nothing; adding a handful of new
predicates costs everything. On that deployment the way it happened was a
generated ETL file in the repository root, because execution loaded every root
.metta into one space.

This branch executes differently, so the lesson is asserted against what it
actually loads. `api._runtime_kb_paths()` is the curated stack named in
`api._INFERENCE_STACK`, never "every root file" (tests/test_nhanes_integration.py
and tests/test_linage2.py pin the layers it leaves out), and `api._generic_kb()`
sends a program that names a patient to the smaller patient stack
(`core.pln_runner.patient_stack`). Every probe here loads exactly the files the
app would load for its query. Measured with `predict-risk-patient` for
Patient001:

    patient stack + 400 atoms under ONE new head symbol   -> answers
    patient stack + 86 new head symbols                    -> answers
    patient stack + 87 new head symbols                    -> non-unwinding panic
    full shared stack + 1 new head symbol                  -> non-unwinding panic

The full shared stack has no margin left for the patient forms, which is why they
are routed out of it; tests/test_patient_stack.py guards the patient stack's
margin and that its answers are the full stack's, byte for byte.

An abort is NOT a Python exception — it is `abort()` inside Rust, uncatchable by
`except` and fatal to the interpreter. So every probe here runs in a SUBPROCESS,
and the assertion is on the child's exit status. That is also why a plain
pytest run cannot be trusted to notice this on its own: in-process it kills the
run with "Fatal Python error: Aborted" and no failing test to point at.

Run from the repository root:
    pytest tests/test_kb_head_symbol_budget.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parent.parent

pytest.importorskip("hyperon", reason="the PLN runtime is not installed")

import api as api_module  # noqa: E402  (conftest puts pln_chat on sys.path)

#: The query the re-test found broken first. It is also the cheapest end-to-end
#: exercise of the inference stack: patient -> markers -> deduction -> risk.
CANARY_QUERY = "!(predict-risk-patient &self Patient001)"

_CHILD = r"""
import json
import sys
from pathlib import Path
from hyperon import MeTTa

paths, query, extra = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
metta = MeTTa()
for path in paths:
    metta.run(Path(path).read_text(encoding="utf-8"))
if extra:
    metta.run(extra)
result = metta.run(query)
if not result or not result[0]:
    print("EMPTY")
    raise SystemExit(3)
print("OK")
"""


def _run(query: str = CANARY_QUERY, extra: str = "",
         kb: Optional[list[Path]] = None) -> subprocess.CompletedProcess:
    """Load what the app loads for `query` (plus `extra`) and run it, out of process."""
    paths = api_module._generic_kb(query) if kb is None else kb
    return subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps([str(p) for p in paths]), query, extra],
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
        f"`{CANARY_QUERY}` did not survive the stack the app runs it in "
        f"(exit {done.returncode}).\n"
        f"An exit of -6 is hyperon's non-unwinding abort: something added head "
        f"symbols to that stack, or routed the query out of the patient stack.\n"
        f"stderr tail:\n{done.stderr[-2000:]}"
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
    """Risk for a second patient, a supplement ranking, and a flat retrieval.

    The first two run in the patient stack and the flat match in the full shared
    stack, because that is where the app sends them.
    """
    done = _run(query=query)
    assert done.returncode == 0, (
        f"`{query}` exited {done.returncode}.\nstderr tail:\n{done.stderr[-2000:]}"
    )


# ── What the limit is, and what it is not ────────────────────────────────────

def test_atom_count_is_not_the_limit():
    """400 atoms under ONE new head symbol are harmless where the canary runs.

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
    """256 new head symbols, 256 atoms, and the interpreter dies.

    Fewer atoms than the test above, and fatal, because they bring new predicates.
    256 rather than the measured threshold of 87, so that this stays a statement
    about the mechanism and does not turn red merely because a stack shrank. A
    generated ETL file introduces a head symbol per field predicate.
    """
    done = _run(extra=_new_head_symbols(256))
    assert done.returncode != 0, (
        "256 new head symbols no longer abort hyperon in the patient stack. That "
        "is good news and makes this file obsolete: re-measure the ceiling, then "
        "revisit the scoped stacks in core.pln_runner and the run_etl.sh guard."
    )


# ── Where generated files go ─────────────────────────────────────────────────

def _run_etl_in_scratch_tree(tmp_path: Path, out_dir: Optional[str]) -> subprocess.CompletedProcess:
    """Run a COPY of scripts/run_etl.sh as if `tmp_path / "root"` were the repo.

    The script resolves the repo root from its own location, so in a scratch tree a
    guard that fails to refuse runs the ETLs THERE (and fails at once: there is no
    data/ and PYTHON is `false`), never over the real root's committed layers.
    """
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "scripts" / "run_etl.sh", root / "scripts" / "run_etl.sh")
    env = {k: v for k, v in os.environ.items() if k != "OUT_DIR"}
    env["PYTHON"] = "false"
    if out_dir is not None:
        env["OUT_DIR"] = out_dir
    return subprocess.run(
        ["bash", str(root / "scripts" / "run_etl.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("out_dir", [".", "ROOT", "ROOT/", "scripts/.."])
def test_run_etl_refuses_to_write_into_the_repo_root(tmp_path, out_dir):
    """`OUT_DIR=.` was a documented invocation, and it is refused before anything is written.

    On this branch a stray root file is no longer executed, but the root holds the
    curated layers: the NHANES reference ETL writes nhanes_reference.metta, the name
    of the committed rules file, and the code reads generated records from ./build.
    """
    if shutil.which("bash") is None:
        pytest.skip("bash is not available to run the script")
    root = tmp_path / "root"
    done = _run_etl_in_scratch_tree(tmp_path, out_dir.replace("ROOT", str(root)))
    assert done.returncode == 2, (out_dir, done.returncode, done.stderr[-1000:])
    assert "must not be the repo root" in done.stderr
    assert not list(root.glob("*.metta"))


@pytest.mark.parametrize("out_dir", [None, "OUTSIDE"])
def test_run_etl_lets_any_other_output_directory_through(tmp_path, out_dir):
    """The guard refuses the root and nothing else: the default ./build and a path
    outside the repo get past it (and then stop at the missing data archives)."""
    if shutil.which("bash") is None:
        pytest.skip("bash is not available to run the script")
    target = None if out_dir is None else str(tmp_path / "outside")
    done = _run_etl_in_scratch_tree(tmp_path, target)
    assert "must not be the repo root" not in done.stderr
    assert done.returncode not in (0, 2), (done.returncode, done.stderr[-1000:])
    made = (tmp_path / "root" / "build") if out_dir is None else (tmp_path / "outside")
    assert made.is_dir()
