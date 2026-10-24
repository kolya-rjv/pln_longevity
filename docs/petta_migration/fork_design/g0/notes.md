# Gate G0: patham9's PLN (PeTTa's lib_pln, vendored) on the real PeTTa

Run 2026-10-24 on this Mac (SWI-Prolog 10.0.2, janus-swi 1.5.3, PeTTa `fea83178`, work tree branch `petta-migration`, content = linage2 @ 47130aa
plus the files this task added). Every number below was measured by the scripts in this folder (`g0_ab.py`, `g0_c.py`, `g0_d.py`; raw output
in `results/`). Nothing was assumed from the simulator: `fork_sim.py` was used as an independent oracle and agrees with the library
bit for bit.

## 0. Verdict

**GO, design 1 (`mp_state`).** (a), (b) and (c) pass without exception (56 + 35 checks, 0 failed), and (d) passes with a large margin: the
`pln` lane adds 0.05 to 0.15 s to a request that costs 0.19 to 0.38 s without it (ratio 1.26 to 1.57 against the legacy chainer on PeTTa, limit 5).
Nothing failed. What did not go as the plan assumed is in section 8.

| gate | result |
|---|---|
| (a) upstream examples on the vendored library | `pln_roman`, `pln_tuffy`, `pln_direct` on the UNFILTERED file: all pinned values; `pln_tuffy`, `pln_direct` on the RESTRICTED artifact: all pinned values; `pln_roman` on the restricted artifact fails, as it must (no Deduction) |
| (b) probes, compared in Python | MP, leak, Revision at c = 1.0, exactly one Derive result, `(0)` and `(State X)` through the Sentence patterns: all pass, on the unfiltered file (PeTTa's Python interface) and on the restricted artifact (adapter session) |
| (c) closure on the real library | patient 100 of 100, LinAge2 86 of 86, generic 101 of 101, feedback edge 109 of 109; sign, strength (max deviation 2.2e-16), confidence (2.8e-17) and chain equal to legacy; none extra, none missing; at three budgets each |
| (d) wall time | one `PLN.Derive` run on the patient graph: median 0.9 ms, max 3.6 ms. 10-run workload 24 ms. 60 runs 76 ms. Request increment 0.099 s (10 runs) to 0.142 s (60 runs); ratio 1.26 to 1.37 |

## 1. What was built

| path | what |
|---|---|
| `vendor/pln_fork/lib_pln_fork.metta` | PeTTa `lib/lib_pln.metta` @ fea83178 with a provenance header and ONE hunk (the per-step trace in the 6-argument `PLN.Derive`). sha256 of the upstream file asserted. |
| `vendor/pln_fork/lib_pln_restricted.metta` | generated: Revision, Modus Ponens, Not-elimination; 45 of 82 forms kept, byte-identical to the vendored ones; 37 dropped |
| `vendor/pln_fork/{LICENSE.NARTECH,LICENSE.PeTTa,PROVENANCE.md}` | both MIT licences (sha256 asserted), pins, the hunk as a diff, how to refresh |
| `scripts/make_pln_restricted.py` | the form-level filter; `--check` for staleness; refuses a form or a table entry it does not know |
| `tests/test_pln_vendor.py` | 46 tests (14 without PeTTa, 32 on the real PeTTa): hash and hunk, licences, fresh-build and byte-identity of the artifact, filter refusals, sandbox scan, name collisions, the probes, the examples, the closure of all four graphs at three budgets, D+Q levels, the simulator oracle, completeness status, shared-id and empty-stamp negative controls |
| `docs/petta_migration/fork_design/g0/` | this note, the re-runnable harness (`common.py`, `g0lib.py`, `g0_ab.py`, `g0_c.py`, `g0_d.py`, `petta_py_child.py`) and `results/*.json` |

The default path is untouched: no file outside these was edited; hyperon never reads `vendor/`.

## 2. Method

* **Through the production adapter.** `core.metta_runtime.PettaRuntime.session()` (the forked child, the engine sandbox, the text scan) holds the
  library; each `PLN.Derive` call carries its sentences as data (`!(PLN.Derive (task) (beliefs) 1 maxsteps tq bq)`). One session serves every run; nothing
  accumulates (no per-source equations).
* **PeTTa's own interface** for the unfiltered file, because the adapter's sandbox refuses the token `Predicate` (a PeTTa builtin) in a knowledge
  base and the unfiltered file holds it in three rules (section 8, item 1). `run.sh` on copies of the upstream examples (made by `g0_ab.py` under the scratch directory's `examples/`), the
  import line replaced by the vendored file's absolute path; `python/petta` in a fresh process for the Python comparisons.
* **Encoding** (decision.md section 3 with the reviewers' corrections): node X = `(State X)`; link = `(Sentence ((Implication (State A) (State B)) (stv q c)) (id))`
  for Pos, consequent `(Not (State B))` for Neg, `(End B)` / `(Not (End B))` for a hazard-ratio record; `q = e + (1 - e) p`, `e` read from
  `Truth_ModusPonens (stv 0 1) (stv 0 1)`; the intervention is the task `(Sentence ((State A) (stv L 1.0)) (0))` with L = 1.0 and, for the reference run, 0.0; the sub-KB is
  the links reachable from A (terminal links followed, not expanded); strength `|s1 - s0| / (1 - e)^hops`, sign = parity of Neg links (cross-checked against
  the sign of `s1 - s0` on every route), confidence = native `c` times `0.9^(hops - 1)`; one result per node-simple route; A = C dropped.
* **Oracles.** (1) `closure.json`, the real hyperon output of the legacy chainer; (2) the repo's own `contract/legacy_reference.py` and my node-simple DFS, which
  agree with it bit for bit on the 100 + 86 recorded pairs and are used for the generic stack and the feedback edge, which hyperon never answered;
  (3) `fork_sim.py`, run on identical inputs, compared list for list.
* **Budgets.** `petta_lib_defaults` = 100 steps / task queue 10 / belief queue 100 (the vendored `PLN.Config.*`), `fork_defaults` = 20/20/200
  (patham9's `Config.metta`), `K256` = 1024/256/256 (the schedule's first rung: 4K steps, tq = bq = K).

## 3. (a) Upstream examples (`results/g0_ab.out.txt`)

| example | library | pinned value | result |
|---|---|---|---|
| `pln_roman` | unfiltered vendored | `((stv 0.5 0.9473684210526316) (1 2 3 4))` | pass (run.sh, 1 of 1 `test`; and PeTTa Python interface, exact string) |
| `pln_tuffy` | unfiltered vendored | `((stv 0.5199204545454544 0.8290782204129108) (1 2 4 9 10))` | pass (run.sh; Python interface) |
| `pln_direct` | unfiltered vendored | 3 formula-level values (0.51 / 0.8432620011567381 twice, 0.5198 / 0.7923434575206812) | pass (run.sh 3 of 3; Python interface) |
| `pln_tuffy` | RESTRICTED artifact | same as above | pass (run.sh; and through the adapter session, exact) |
| `pln_direct` | RESTRICTED artifact | same as above | pass (run.sh 3 of 3; Python interface); its fold also reproduced as data on the sandboxed session |
| `pln_roman` | RESTRICTED artifact | - | fails: Deduction is filtered out (negative control: the filter really removed it) |

`plntest` and `plntestdirect` were not run: they load no library (reviewer correction b). The one hunk is neutral: 40 `PLN.Derive` runs return
byte-identical strings on the unpatched upstream file, the vendored file, and the restricted artifact through the adapter; the patched files print no
`SELECTED` line, the upstream file prints one per selection.

## 4. (b) Probes (compared in Python; never MeTTa `test`/`assert`, which `halt(1)`)

Same expectations on both libraries (unfiltered via PeTTa's Python interface, restricted via the adapter):

| probe | expected | measured |
|---|---|---|
| `Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8)` | (0.26, 0.64) | `(stv 0.26 0.6400000000000001)` (0.8 * 0.8 in IEEE doubles; compared to 1e-15) |
| leak: `Truth_ModusPonens (stv 0 1) (stv 0 1)` | strength 0.02 | `(stv 0.02 1.0)` |
| `Truth_Revision` with c = 1.0 in either or both arguments | no result | `[]`, no exception, all three placements |
| `Truth_Revision` with c capped at 0.999999 | a result | `(stv 0.5 0.999999000001)` |
| 4-row MixCompound fold (design 1, section 10) | Neg 0.263050, native c 0.949466 | `(stv 0.263050153531218 0.9494655004859086)`, equal to hand-typed arithmetic to 1e-15 |
| `PLN.Derive` | exactly one result `($Tasks $Beliefs)` | 1 result in all four shapes (normal, empty tasks, steps > maxsteps, K = 256); every `PLN.Derive` call of (c) and (d) asserts it |
| `(PriorityRank / PriorityRankNeg (Sentence ((State X) (stv 1.0 1.0)) (0)))`, `(End X)` with a two-id stamp, `ConfidenceRank`, `BestCandidate` | match the patterns | 1.0 / -1.0 / 0.65 / 1.0, the premise selected first |
| `StampDisjoint (0) (1 2)` / `(0) (0)` / `(0 3) (0 4)` | true / false / false | as expected |
| rules on `(State A)` | MP: one result; Neg link: `(Not (End B))`; NotElimination: `(End B)` with 1 - s | as expected; `(End B)` + `Implication (State B) ..` gives no result (records end a chain, never continue one) |

## 5. (c) Closure on the real library (`results/results_c.json`)

Each cell: ordered pairs found / pairs in the legacy reference, missing, extra, sign-or-value mismatches. Every row: max |ds| 2.2e-16, max |dc| 2.8e-17, one
route per pair, no run left tasks behind, no run reached K-1 beliefs.

| graph | legacy pairs | lib defaults 100/10/100 | fork defaults 20/20/200 | K = 256 |
|---|---|---|---|---|
| patient stack (28 links) | 100 | 100, 0, 0, 0 | 100, 0, 0, 0 | 100, 0, 0, 0 |
| LinAge2 stack (27 links) | 86 | 86, 0, 0, 0 | 86, 0, 0, 0 | 86, 0, 0, 0 |
| generic stack (29 links, `Lifespan -> Mortality` at p = 1.0) | 101 | 101, 0, 0, 0 | 101, 0, 0, 0 | 101, 0, 0, 0 |
| patient + `Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65)` | 109 | 109, 0, 0, 0 | 109, 0, 0, 0 | 109, 0, 0, 0 |

* The feedback edge terminates (a derivation uses each link id once); the 100 old answers are unchanged, 9 pairs are new (`CI -> CS, SASP, DNAmPAI1, CHD`;
  `Omega3 -> CS, SASP, DNAmPAI1, CHD`; `SASP -> CS`). Without the reachable pre-filter (whole KB as beliefs) the patient closure is the same at two budgets.
* **D+Q levels, q = p (the fork read literally, decision section 1 row 2):** CS 0.30000 / 0.98000, SASP 0.26900 / 0.83340, DNAmPAI1 0.20292 / 0.58671, CHD
  0.13769 / 0.36029, difference -0.222599 (applied / absent). With the affine storage the difference divided by 0.98^4 is 0.24989999999999998, the legacy value to the last digit.
* **Lanes for D+Q -> CHD** (measured): legacy Neg 0.24989999999999998 / 0.17017138125000003; R2 (affine + shim) Neg 0.24989999999999998 / 0.17017138125000006; R1 (affine,
  native c) 0.2499 / 0.23343125; R0 (q = p, native c) 0.22259936 / 0.23343125.
* **Selections** (chunked resumption with the 6-argument form, one selection per call; the final state equals the one-shot run's, list for list):
  patient max 12 (Spermidine), D+Q 11, Fisetin 11, Metformin 7, Berberine 7; identical at all three budgets. With the feedback edge: D+Q 12, CellularSenescence 10,
  Omega3 16 (the maximum), as the decision recorded.
* **Oracle:** 236 of 236 runs (patient, LinAge2, generic x two budgets x two levels) have the same final (tasks, beliefs) as `fork_sim.py` in the same order,
  floats bit-identical.
* **Determinism:** 140 runs (as given, reversed, 5 shuffles of each source's links): no difference in routes, sign, strength or confidence.
* **Completeness status:** with tq = bq = K (K = 6, 8, 10, 12, 16) 54 runs lost pairs and all 54 were flagged by "tasks empty and fewer than K-1 beliefs"; with tq 2 to 4 and bq 200
  11 runs lost pairs and none was flagged (decision correction 1, now on the real library).
* **Why `PLN.Query kb term` is not used:** seeding every sentence as a task finds 66 of 100 pairs at the library defaults (the decision's 66) and 47 at 20/20/200 with the
  order-keeping `Unique`; the simulator gives the same counts.
* **Negative controls (the harness sees what it should):** the earlier `q = min(1, p + e)` fails the generic stack on exactly `Lifespan -> Mortality` (0.98 against 1.0, reviewer
  correction a); `q = p` fails all 100 patient pairs (max 0.041); two serial links sharing an evidence id make the library refuse the chain (reviewer correction e), distinct ids derive
  it; a premise with an empty stamp and c = 0.9 is revised with itself (255 copies after 12 steps), the id-0, c = 1.0 premise has 1 copy.

## 6. (d) Wall time (`results/results_d.json`, median of 7 fresh processes unless stated)

Warm, inside one PeTTa session (serialise + evaluate + parse), patient graph, K = 256, reachable sub-KB:

| shape | time |
|---|---|
| one `PLN.Derive` run (300 samples) | median 0.9 ms, mean 1.3 ms, p90 2.8 ms, max 3.6 ms (slowest sources Spermidine 3.2 ms, Resveratrol 2.9 ms) |
| 10-run workload (5 candidates x 2 levels, 96 selections) | 24.2 ms |
| 60 runs (30 sources x 2 levels, eager) | 76 ms (5 repeats: 75 to 77 ms) |
| Python join of one source (parse, path rebuild, shim) | 0.3 ms for all 30 sources together |
| legacy on PeTTa, same session: the 5-candidate `rank-interventions-for-patient` | 0.8 ms; `(collapse (infer &self A $c))` for 30 sources 6.2 ms; one `infer D+Q CHD` 0.2 ms |

Per request, fresh process (`run_program`), patient stack (18 files, 208 KB) and the 5-candidate pool:

| request | wall | engine start | KB load | queries | increment over legacy | ratio |
|---|---|---|---|---|---|---|
| legacy (PeTTa) | 0.380 s | 0.142 | 0.222 | 0.001 | - | 1.00 |
| + library + 10 Derive runs (lazy: the five candidates) | 0.479 s | 0.142 | 0.299 | 0.021 | +0.099 s | 1.26 |
| + library + 60 Derive runs (eager: all 30 sources) | 0.522 s | 0.145 | 0.299 | 0.062 | +0.142 s | 1.37 |
| + 186 answer atoms as KB facts | 0.526 s | | 0.301 | 0.062 | +0.004 s | |

The single-`infer` probe of the cheapest shape (smoke:2, ten-file stack, `infer CellularSenescence CoronaryHeartDisease`): legacy 0.196 s; with the library and the one source
(2 runs) 0.247 s (+0.051 s, ratio 1.26); with the eager precompute of all 30 sources (60 runs) 0.308 s (+0.111 s, ratio 1.57). A trivial program costs 0.148 s (the fixed
start-up); loading the restricted artifact alone adds 0.043 s.

Reading: the increment is the library load (+0.077 s on top of the patient stack in one load, +0.044 s when the library is loaded as its own call after the stack, see section 8
item 3) plus 0.4 to 0.9 ms per Derive run; the Derive work of the eager seam is 0.062 s. In pure compute the pln lane is 25 to 30 times the legacy query (24 ms against 0.8 ms for the
workload), but the request is dominated by process start and KB load, which the lane does not change, so the routed ratio is at most 1.57. Against the hyperon figure (patient stack 1.05 s
on the legacy chainer, LAUNCH.md) the whole `pln` request is faster. Peak RSS 72 MB against 56 MB.

Scale (layered trees, K = 2048 for more than 251 links; one level-1 run): 39 links 25 ms (complete); 340 links 1.5 s (complete); 780 links 7.8 s (complete); 1,364 links 21.7 s and
INCOMPLETE (the belief queue hit K-1, flagged). Cost is quadratic in the reachable links, as the decision said. Irrelevant for the shipped graphs (at most 30 links), relevant for
the 6c control demonstration: a 540-link network is about 3 s per run, 6 s for the two runs, and the first rung of the K schedule (256) will be flagged partial on it.

## 7. Reviewer corrections: status after G0

| | correction | status |
|---|---|---|
| a | `q = e + (1 - e) p`, report `|s1 - s0| / (1 - e)^hops` | holds on the real library for every p in [0, 1] including the generic stack's p = 1.0 link (101 of 101); the capped mapping is caught |
| b | G0(a) scope | done as written: unfiltered on roman/tuffy/direct, restricted on tuffy/direct, import edited in copies, `plntest*` skipped |
| c | Derive returns `(Tasks Beliefs)` only; selections by chunked resumption; K above the sub-KB; exactly one result; probes in Python | all done; resumption equals the one-shot run; K = 256 against 30 reachable links at most |
| d | eager seam, legacy equations not loaded next to the new ones | not G0's to build. G0 measured both the lazy (10 runs) and the eager (60 runs) cost, and showed the library loads next to the legacy stack with no name collision (the legacy answer is byte-identical with the library loaded; `tests/test_pln_vendor.py` checks the 48 function names against every root `.metta` file). Under `pln` the extra `(= (infer ..))` clauses must still be kept out: PeTTa adds clauses, it does not replace them |
| e | shared ids only for parallel evidence of one link | negative control passes on the real library (serial shared id: chain refused); `assert_ids_unique_along_paths` is the adapter guard to keep |
| f | one defined DrugAge net formula and the Resveratrol/Metformin/Rapamycin table | not part of G0 (6b). G0 confirms the `Truth_Revision` fold it builds on, on the real library (MixCompound 0.263050 / 0.949466, equal to hand arithmetic) and that c = 1.0 needs the 0.999999 cap |
| g | generic stack in gate 6a | included: 101 pairs, `Lifespan -> Mortality` at 1.0 |
| h | `drugage_selector.MAX_ROWS = 150` | not part of G0 (6b) |
| i | control demonstration with recall by confidence and by rank-score | not part of G0 (6c); the scale probe above is the only control-related measurement |

## 8. What did not go as planned, and what the next phases need

1. **The unfiltered library cannot be loaded through the production adapter.** `core.metta_runtime` refuses the token `Predicate` in a knowledge base (it is a PeTTa builtin, one of the
   "call any Prolog predicate by name" doors) and the unfiltered file has it in the three Evaluation-with-Inheritance rules. Only the restricted artifact passes the text scan
   (`tests/test_pln_vendor.py` pins both facts). `PLN_RULESET=full` (a research lane, never reachable from a request) therefore cannot run in the sandbox as it is; it has to
   run outside it (PeTTa's own interface, as G0 did) or the owner widens the sandbox for that lane.
2. **Node names are scanned.** The adapter refuses a program whose text contains a refused form anywhere, including inside `(State X)`: a node or compound named `test`, `call`, `Predicate`,
   `halt`, `shell`, `include`, `assert`, ... is refused (measured: `test`, `call`, `Predicate`); `first`, `id`, `exp`, `Sentence` pass. The `pln` seam should send opaque ids (`n0`, `n1`, ..)
   in the sentences and map back, which also shortens the text; DrugAge compound names are free text.
3. **Load order costs time.** The restricted artifact loaded in the same `process_metta_string` as the 18-file patient stack costs +0.079 s; loaded as a second call after the stack it costs
   +0.044 s (its cost alone). Across loads a head that is not yet a function is compiled as data, so the library must be loaded before any file that calls it (the new chainer file), after the
   files it does not touch. Worth doing in 6a: 35 ms of the 99 to 142 ms increment.
4. **`add-atom` is closed after the load.** `_program_child` locks the space once the KB is loaded, so answer atoms for the eager seam must be part of the KB text (or produced before the
   lock). 186 answer atoms as KB facts cost 4 ms.
5. **D+Q takes 11 selections on the plain graph and 12 with the feedback edge**; the decision's "D+Q 12" belongs to the feedback row. The maximum on the plain graph is 12 (Spermidine).
6. **`Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8)` prints `(stv 0.26 0.6400000000000001)`**: probes must compare numerically.
7. The vendored `Unique` keeps first-occurrence order (`list_to_set`); the fork's own `Unique` reverses. The closure does not depend on it (decision row 8, simulator; not re-run here), but the fork's own
   examples at its 20/20/200 defaults are not what this file was checked against (the compatibility study predicts a different `pln_tuffy` answer there; not re-run); the pins checked are
   PeTTa's own `pln_roman/tuffy/direct`.
8. Cost is quadratic in the reachable links (scale probe). K = 256 is enough for the shipped graphs; a network above about 250 links needs the second rung (2048) and a larger deadline.
9. Not measured: the real seam (in-child Python adapter through janus, `infer`/`explain` over answer atoms), the validator/registry/prompt wiring, the DrugAge fold at paper level, hyperon (the
   library is PeTTa-only). The Derive calls here go through the session pipe; inside the child they would not pay the JSON round trip.

## 9. Full-suite check (hard rule 1)

Run last, after every file of this task existed, with `PLN_CHAINER=legacy` (the default; nothing reads it yet):

| runtime | result | failure set |
|---|---|---|
| `PLN_RUNTIME=hyperon` (default) | 32 failed, 2247 passed, 22 skipped, 1 deselected in 432 s | the 32 ids are IDENTICAL to `docs/petta_migration/baseline/hyperon_failures_47130aa.txt` (`diff` empty) |
| `PLN_RUNTIME=petta` | 31 failed, 2255 passed, 16 skipped in 297 s | the same ids minus `test_human_evidence.py::test_the_runtime_kb_still_answers_the_query_that_aborted_the_interpreter`, the hyperon-only abort probe |

The deselected test is `tests/test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move`: it loads the full shared stack and kills the hyperon interpreter
(`Fatal Python error: Aborted`, the known trie.rs:179 bug) with or without these files; run alone it aborts. A first full run without the deselect died at it
(the baseline summary also shows "1 deselected"; I did not find the flag that produced it recorded anywhere). The work tree after the runs holds only the new paths:
`vendor/`, `scripts/make_pln_restricted.py`, `tests/test_pln_vendor.py`, `docs/petta_migration/fork_design/g0/`. The suite counts include tests other agents added in parallel,
so only the failure sets are comparable with the baseline.

## Reproduce

```
PLN_G0_SCRATCH=<dir> .venv/bin/python -B docs/petta_migration/fork_design/g0/g0_ab.py     # (a), (b), the hunk checks
PLN_G0_SCRATCH=<dir> .venv/bin/python -B docs/petta_migration/fork_design/g0/g0_c.py      # (c)
PLN_G0_SCRATCH=<dir> .venv/bin/python -B docs/petta_migration/fork_design/g0/g0_d.py      # (d), about 55 s
.venv/bin/python -m pytest tests/test_pln_vendor.py -q                                     # the same gates as tests, 10 s
```
