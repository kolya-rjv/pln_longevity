# Gate G0 harness

Re-runnable evidence that patham9's PLN (vendored in `vendor/pln_fork/`) runs on the real PeTTa and reproduces the legacy chainer's closure under the
`mp_state` encoding of `../decision.md` (with the reviewers' corrections). The findings and numbers are in `notes.md`; the raw output of the last run is in `results/`.

| script | checks |
|---|---|
| `g0_ab.py` | (a) PeTTa's `pln_roman`, `pln_tuffy`, `pln_direct` on copies whose import line is edited to the vendored file (unfiltered) or the restricted artifact, through `run.sh` and through PeTTa's Python interface; (b) the probes (Modus Ponens, the 0.02 leak, Revision at c = 1.0, exactly one `PLN.Derive` result, the Sentence patterns for `(0)` and `(State X)`), compared in Python on both libraries; the per-step trace and the neutrality of the one hunk |
| `g0_c.py` | (c) the closure of the patient (100), LinAge2 (86) and generic (101) stacks and the feedback edge (109) at three budgets against the legacy reference; D+Q levels and lanes; selections by chunked resumption; the `fork_sim.py` oracle; completeness status; determinism; why `PLN.Query kb term` is not used; negative controls (capped mapping, literal mapping, shared ids, empty-stamp premise) |
| `g0_d.py` | (d) wall time per `PLN.Derive` run, the 10-run workload, the 60-run eager precompute, per-request increments against the legacy chainer on PeTTa, a scale probe |
| `g0lib.py`, `common.py`, `petta_py_child.py` | the encoding, the engine wrapper over `core.metta_runtime` sessions, the legacy reference, PeTTa's own Python interface in a fresh process |

```
export PLN_G0_SCRATCH=/some/dir          # optional; a fresh temp dir otherwise
.venv/bin/python -B docs/petta_migration/fork_design/g0/g0_ab.py
.venv/bin/python -B docs/petta_migration/fork_design/g0/g0_c.py
.venv/bin/python -B docs/petta_migration/fork_design/g0/g0_d.py     # about a minute
```

Each script prints `PASS`/`FAIL` per check and writes `results_*.json` into the scratch directory. They need SWI-Prolog, `janus-swi` and the pinned PeTTa under
`.petta/PeTTa`; python-dotenv is switched off before any repo import (`pln_chat/.env` is never read). `tests/test_pln_vendor.py` runs the same gates as pytest tests
(about 10 s); rerun this harness after refreshing the vendored library (`vendor/pln_fork/PROVENANCE.md`, "How to refresh").
