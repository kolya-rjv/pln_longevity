# PeTTa spike kit

A ready-to-run kit for Phase 1 of `docs/petta_migration/LAUNCH.md`: does the app's MeTTa work run on PeTTa, how
fast, and what breaks. It was written **without running PeTTa**. The hyperon half is already recorded:
`baseline_hyperon.json` holds hyperon 0.2.10's answers and timings for 145 query shapes. Everything in
this directory is stdlib plus the repo's own code, read-only; the repo and the `.petta/` checkouts are never written.

Running the kit is **one explicit act**: every script that would start PeTTa refuses until you export

    export KIT_ALLOW_PETTA=1

## Setup (once per shell)

    cd scripts/petta_spike
    PY=/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python   # has hyperon 0.2.10, janus-swi 1.5.3, sexpdata
    export KIT_ALLOW_PETTA=1                                     # the go

## The commands, in order

    $PY selftest.py                      # 1. seconds, no PeTTa: normaliser, comparator, fixes, gate, adapter and measure.py plumbing against a fake janus/petta, parsers
    $PY run_loads.py --stacks smoke10    # 2. does anything load at all? (6 runs, then read out/loads.md)
    $PY run_loads.py                     #    then every stack: as-is and fixed, string / cached miss / cached hit
    $PY run_shapes.py                    # 3. every baseline shape on PeTTa, diffed -> out/breakage_table.md
    $PY measure.py                       # 4. consult / load / latency / printing / order / limits / fork / GIL -> out/measure.md

Cheap first looks: `$PY measure.py --sections consult,probes`, `$PY run_loads.py --stacks smoke10,human_evidence --modes string --variants fixed`.
Useful second runs: `$PY run_shapes.py --fixes none` (which shapes the unfixed KB blocks), `$PY run_shapes.py --mode isolate`
(one process per shape: every cost then includes import + consult + load).
Add `--jobs 2` once `RSS` per stack is known. `selftest.py --hyperon` also drives run_shapes/run_loads on the hyperon engine
(everything must come back `identical`).

## Environment variables

| variable | meaning | default |
|---|---|---|
| `KIT_ALLOW_PETTA` | `1` = PeTTa may be started. Unset: every PeTTa run raises `PettaNotAllowed`. | unset |
| `PETTA_HOME` (or `PETTA_PATH`) | the PeTTa checkout (`src/main.pl`, `python/petta/`, `python/helper.pl`) | `/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa` |
| `PLN_REPO` | the work tree whose `.metta` files and `pln_chat/` are read | `/Users/kolya/PycharmProjects/pln_longevity_petta` |
| `KIT_OUT` | where reports and caches go | `<kit>/out` |
| `KIT_BASELINE` | the hyperon baseline | `<kit>/baseline_hyperon.json` |
| `PETTA_LOAD_CACHE` | snapshot directory of `load_metta_file_cached` | `<out>/petta_load_cache` (per stack/variant in run_loads) |

Set inside the child, not by you: `PYTHONPATH` gets `<PETTA_HOME>/python` (and `sys.path[1]` is set before the import, so
`from petta import PeTTa` finds the pinned wrapper and the adapter checks `petta.__file__` lies under `PETTA_HOME`),
`PYTHONDONTWRITEBYTECODE=1` (no `.pyc` is written into the checkout), `PATH` gets `/opt/homebrew/bin`, the Prolog `stack_limit` is
lowered to `--stack-limit-mb` (default 2048; PeTTa's wrapper sets 8 GB).
If `import janus_swi` fails with a missing `libswipl`: janus is linked against `/opt/homebrew/opt/swi-prolog/lib/swipl/lib/arm64-darwin`;
try `SWI_HOME_DIR=/opt/homebrew/opt/swi-prolog/lib/swipl` or `DYLD_LIBRARY_PATH` (the error text lands in `loads.md` under "First error per stack").

## What is here

| file | what |
|---|---|
| `petta_adapter.py` | the throwaway adapter: `run_program(kb_text, expressions) -> list[list[str]]` and `run_program_report(...) -> RunReport` |
| `stacks.py` | each stack's file order as the repo defines it (imports `pln_chat.core.pln_runner` and `pln_chat.api`; `--source parse` reads the same lists with `ast`; `--check` compares) |
| `shapes.py` | the 145 query shapes as (KB files, extra atoms, `!` expressions) triples, routed the way `api.py` routes them |
| `kb_fixes.py` | the three Phase 2 load fixes as text transformations (`(superpose ())` -> `(empty)`, stored `(evidence-confidence T)` calls -> the tier number, nested `(, A (, B C))` -> `(, A B C)`) |
| `record_baseline.py`, `hyperon_child.py` | records `baseline_hyperon.json` on hyperon (**run already**; rerun it if the work tree changes) |
| `run_loads.py` | task 4: load each stack as string and via the cached file; first error, seconds, RSS; smoke queries vs the answers written in `pln_deduction.metta` |
| `run_shapes.py` | task 5: every shape on PeTTa vs the baseline; classes identical / number_format_only / within_tolerance / ordering_only / different / error / no_baseline |
| `measure.py` | task 6: consult time, cold / cached load, latency, print probes, the seven first-answer sites, limits, fork, GIL |
| `kit_common.py` | paths, s-expression reader, the answer normaliser and comparator |
| `selftest.py` | 75 offline checks (82 with `--hyperon`) |
| `NOTES.md` | what the baseline taught, what to expect, design decisions, open items |

## Outputs (`out/`)

| file | from | content |
|---|---|---|
| `loads.json`, `loads.md` | run_loads | per stack / variant / mode: status, first error, import / consult / load s, RSS, missing or short heads, smoke verdicts, canary classes, hyperon load ms; `gate` block |
| `shapes_report.json`, `breakage_table.md` | run_shapes | per shape and expression: class, hyperon answers, PeTTa raw and normalised answers, both latencies, the first error text; counts by class and source; slowest ratios |
| `measure.json`, `measure.md` | measure | one block per section, rendered tables |
| `measure_probes/` | measure | the probe run behind the print / order tables |
| `kb_files/kb_<sha>.metta`, `petta_load_cache/` | cached mode | the temp KB files and PeTTa's snapshots; delete freely |

## Reading the results against the Phase 1 gate

| gate item | where |
|---|---|
| janus-backed PeTTa runs on this Mac | `loads.md` gate: `janus_backed_petta_ran`; "First error per stack" |
| the smoke answers appear after only the Phase 2 fixes | `loads.md` "Smoke queries"; gate `smoke_answers_after_phase2_fixes_only` (smoke10, fixed) |
| no silent partial loads | gate `silent_partial_loads`; `verify.n_missing / n_short` per run; a load goal that fails without raising is status `load_failed` |
| per-process load is seconds, not tens of seconds | `loads.md` columns import / consult / load; `measure.md` "consult" and "loads" |
| no routed probe more than 5x slower than hyperon, load included | `measure.md` "latency and the Phase 1 gate" (`pass`, worst five); fork-from-warm-template price in "fork" |
| lib_pln's own examples | not in this kit (needs the owner's go on the `repos/PLN` scripts); `stacks.py` has a `fork_pln` stack (Utils Constraints Formulas Rules Config Deriver) that `run_loads.py --stacks fork_pln` loads as a string |

## Adapter contract (what to carry into `pln_chat/core/metta_runtime.py`)

* one fresh `spawn` child per KB load; only the child imports `janus_swi` / `petta`; the parent stays Prolog-free and can fork or spawn freely
* `run_program(kb_text, expressions)`: one `run_metta_helper(process_metta_string)` call for the KB, one per `!` expression -> `list[list[str]]` of normalised strings; raises `PettaRunError(report)` on a failed load, a timeout, a crash or a raising expression
* a load is failed if it raises, **or if the goal fails without raising**: PeTTa's python wrapper returns `[]` for both a failed goal and a good load of a KB without `!` forms
* verification: clause count per `(= (f …) …)` head against the source (`verify.missing`, `verify.short`); `strict_verify=True` makes a short head a failed load
* hard deadlines (total / load / per expression) with SIGKILL; stdout and stderr captured at fd level; RSS at every stage; peak RSS; CPU seconds
* `mode="cached"`: KB written to `kb_<sha256[:16]>.metta`, loaded with `load_metta_file_cached`. It must be the **first** helper call of the process (every helper call asserts a `silent(true)` fact and the snapshot key is the dynamic database)
* `Answer(raw, norm)`: the raw PeTTa string next to the string in hyperon's rendering (`True`, `1e-5`, ints as ints)
* default deny: `KIT_ALLOW_PETTA=1` is the owner's go; drop that gate when the adapter becomes product code
