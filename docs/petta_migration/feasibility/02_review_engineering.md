# Adversarial review (engineering-reality lens) of verdict/feasibility.md

Reviewer scope: compatibility-table honesty, LOC/effort credibility, missed repo coupling, the adapter and
deadline story with an in-process SWI-Prolog, phase order, gate testability. Read-only on the repo
(linage2 @ 47130aa); PeTTa read at fea83178, trueagi-io/PLN at 4405956 (existing clones under
`scratchpad/feasibility/upstream/`). Nothing was executed against SWI-Prolog. Helper scans I wrote:
`review_engineering/tools/heads.py`, `heads_lib.py` (function-head/fact-head overlap, arity mismatches:
none found; cross-file forward references in `_INFERENCE_STACK` order: none found).

Bottom line: "doable: yes" survives. The phase plan does not survive as written: Phase 3 cannot fan out into
nine independent clusters (O1), and the verdict's claim that the executor already gives one KB load per
process is false for a production task (O2) and for inline mode/tests (O3). Phase 5's performance gate
measures the wrong unit (O5). The NO-GO contingency (warm worker) is unsafe as described (O4). The latency
gates are too lax (O6).

---

## O1 (blocking: phase plan) Phase 3's clusters are not independent

Verdict Phase 3: nine clusters, each in its own worktree, "each verified by its test module on BOTH
runtimes", "Each cluster re-pins only its own strings", PARALLEL (up to 9 agents). The load-blocking edits are
spread across clusters: `(superpose ())` in clusters (b) (c) (d) (g) (h), stored calls in (a) (c) (d) (h).

But each cluster's test modules load other clusters' files, and on PeTTa one non-compiling form fails the
whole load:
- `(superpose ())` has no translation: translator.pl:637-640 commits to the `superpose_literal` branch,
  `build_superpose_branches([], _, [])` (translator.pl:1537) gives `[]`, `disj_list` has clauses only for
  `[G]` and `[G|Gs]` (translator.pl:1533-1534) -> translate_expr fails; `translate_expr([H0|T0],...)` is the
  last clause (translator.pl:598), translate_clause has no catch (translator.pl:30-50), and
  `maplist(process_form(...))` in process_metta_string (filereader.pl:93-98) stops at the first failing form.
- tests/test_pln_deduction.py (cluster a) loads 12 files including pln_abductive_diagnosis.metta, which has
  code `(superpose ())` at :105 (cluster b).
- tests/test_counterfactual.py (cluster d) loads patient_profile.metta (:294, cluster c),
  pln_abductive_diagnosis.metta (b), mechanistic_bridges.metta (a, stored calls).
- tests/test_supplement_recommendation.py (cluster c) loads pln_counterfactual.metta (:98, :109, :202) and
  pln_risk_prediction.metta (:244), both cluster d.
- tests/test_hallmark_targeting.py:256-265 (cluster b) loads the whole `_INFERENCE_STACK`
  (api.py:184-231: every patient layer).
- The stored-call fix is one design decision (reduce in combine-tv/link-effect, pln_deduction.metta:89-91,
  113-117; or rewrite the facts; or change the writer expansion_schema.py:232-240), but the facts sit in
  mechanistic_bridges (a), supplement_evidence (c), lifestyle_evidence (d), linage2_core (h).

No cluster can turn green until the others land, and different agents could pick different stored-call
fixes. Required change: a sequential "load-unblock" step committed before the fan-out (end of Phase 2): all
20 code-level `(superpose ())` -> `(empty)` (8 files), one stored-call strategy for the 25 facts plus the
writer, flatten cellage_calibration.metta:160-163, and a load-verification check. Only then fan out, and
scope each cluster to semantics and pins.

## O2 (major) The executor does not give one KB load per process: run_query_parts

Verdict table row #29: risk "none", "one KB load per process, which the fork-per-query executor already
provides". Phase 4: "If load/latency is bad: keep the stacks (they are harmless)".

`run_query_parts` runs several `run_query` calls, each against its own KB, "one after another in THIS
process" (pln_chat/core/pln_runner.py:709-723). It is an executor task (executor.py:122-124, 149) used for
every mixed LinAge2+generic program by the API (api.py:3279-3286) and the Gradio chat (app.py:391-394). On
PeTTa the space and the compiled clauses are process-global (python/petta/__init__.py:29-60; spaces.pl:2-10;
filereader.pl:160-171 assertz). So the generic half runs on top of the LinAge2 stack and the caller's full
atoms, and every equation of a file that is in both stacks (pln_deduction, epistemic_calibration,
mechanistic_bridges, ...) is asserted twice, which duplicates results at every recursive hop. Real-engine
tests through the worker pool will catch it (tests/test_linage2.py:777-800, 802-822, 951-962, e.g.
`len(atoms) == 1`), but Phase 2 must own it: offload each part to its own child, or make the adapter fork per
load. The Phase 4 "keep the stacks" fallback is not harmless unless this is fixed.

## O3 (major) Inline mode, scripts and the test harness break under one global PeTTa

`PLN_WORKER_POOL_SIZE=0` runs the task in the calling process (executor.py:321-322; config.py:136-140;
API.md:230-237: for the in-process contract tests and for debugging). With an in-process PeTTa adapter the
second call in that process accumulates the previous KB and its caller atoms. Patient atoms from one request
are then visible to the next (debug/test only, since inline mode is not a deployment setting). scripts/*_chat_demo.py
also call `run_query` repeatedly in one process (counterfactual_chat_demo.py:151 in a loop).

The verdict's harness offers "per-module process isolation". That is not enough:
- tests/test_gene_queries.py:77 sets inline. Seven slow tests in one module call `run_cellage_effects`
  in process (:336-372, :406-412), one rewrites `evidence-confidence InVitro` and loads again (:376-402),
  and `/genes/...?infer=true` runs inline (:415-465). Duplicated clauses break `len(effects) == len(rows)`
  (:369) and the retune test.
- tests/test_drugage_ranking_contract.py:229-262 calls `run_query` twice in ONE test.
- tests/test_caller_patients.py:204-249, test_compound_names.py:188/203, test_prompt_size.py:140,
  test_runtime_grounding.py:339 and test_drugage_discovery.py:67 also run real calls in process.

Fix: the PeTTa `run_program` must guarantee a fresh process itself, by forking or spawning internally or by
refusing a second load in the same process. That one change covers inline mode, scripts, tests and O2. Phase
2's 2 agent-phases (2-3 days) should become about 3-4: adapter isolation, run_query_parts, the load-unblock
sweep, per-call test isolation and load verification.

## O4 (major) The warm-worker contingency is unsafe as described

Risk #8 and the Phase 1 NO-GO branch propose "one long-lived PeTTa worker per stack with add-atom/remove-atom
of per-query atoms and restart-on-kill". In PeTTa:
- `remove-atom` is `retractall`, which removes every identical copy (spaces.pl:13-21; open issue #251).
  Removing a per-query atom that equals a KB atom therefore deletes the KB copy too.
- The validator whitelists `add-atom`, `remove-atom` and `=` (metta_validator.py:37, 44).
- A runtime `(add-atom &self (= ...))` compiles a GLOBAL clause (spaces.pl:26-41, 48-71).

So one `/metta/run` caller could permanently change answers for every later caller of a warm worker. Today
those mutations die with the forked child. The contingency needs per-query isolation from a preloaded image,
or a validator ban on space mutation plus a reset strategy. Two candidates: fork per query from a preloaded
"zygote" process (SWI threads make fork safety unknown), or `load_metta_file_cached` on a per-stack
concatenated temp file. The verdict's "applies to files, not strings" is no obstacle, because the stacks are
file lists (load_cache.pl:1-30).

## O5 (major) Phase 5 gate measures one PLN.Query; requests make dozens

PLN.Query reruns PLN.Derive from scratch and then filters the final beliefs by `==` (lib_pln.metta:409-421).
The closure itself does not depend on the query. Requests fan `infer` out:
- `component-cause`: one infer per hallmark (pln_counterfactual.metta:124-126; 12 hallmarks,
  hallmarks_core.metta) for every elevated GrimAge component (`decompose-grimage` :150-155).
- `linage-biomarker-causes`: the same per biomarker (pln_linage2.metta:119-123).
- `diagnose-patient` defaults to 6 hypotheses (patient_profile.metta:334-337), each infer'd per observation
  (pln_abductive_diagnosis.metta:43-45).
- `patient-relevance`: one infer per intervention x elevated marker (patient_profile.metta:354-356, 370-374).
- `linage-scenarios-patient` fans out 4 levers (pln_linage2.metta:410-413).

Request time is therefore N x closure, with N in the tens. Phase 5 needs a derive-once-per-request cache
(one child = one request, so cache once per process). Its gate should be "the largest patient form and the
largest compound complete within the deadline", not "one PLN.Query fits".

## O6 (major) Latency gates accept an order-of-magnitude regression

The Phase 1 GO criterion is "per-process load is seconds, not tens of seconds". The Phase 4 criterion is
"p95 latency within PLN_QUERY_TIMEOUT_SECONDS", which is 60 s (config.py:141-144).

Today's baseline is far lower: fork costs single-digit ms against 10-600 ms queries (executor.py:183-186),
and infer takes 0.02 s (memory note stakeholder-ben-pln-inference). On PeTTa, every forked child first
consults 13,661 lines of PeTTa Prolog (src/*.pl + src/typecheck/*.pl) and then runs the whole-file analysis
and compile of the stack (filereader.pl:75-98) before its first answer.

The gates should be relative to the hyperon baseline per routed probe, with load time included, and Phase 1
should measure the cached-load amortisation.

## O7 (minor) Silent failure is a known, not an unknown

`PeTTa._run_helper` returns `result.get("Results", [])` from `janus.query_once` (python/petta/__init__.py:62-75).
A load that fails, rather than raises, therefore returns `[]` with no exception. Forms processed before the
failing one stay asserted. The adapter has to call janus directly and check `truth`, and it should verify
the load before running queries. Without that, the Phase 2 gate "suite RUNS end to end on PeTTa" passes
trivially while every answer is empty.

## O8 (minor) Deployment is hostage to stakeholder sign-off

The Phase 6 gate needs Phase 5's number table signed off by the owner and Ben. A runtime port can ship with
PLN_CHAINER=legacy first, which decouples "4-6 weeks to deployable" from stakeholder latency.

## O9 (minor) Untestable gates

- Phase 1: "lib_pln's own examples pass or fail for understood reasons" cannot fail.
- Phase 7: "...or an honest report" cannot fail.
- Phase 4 deletes the abort-probe modules (incl. test_generic_stack.py and test_patient_stack.py), but its
  own gate uses the routing-agreement tests that live in those modules (test_generic_stack.py:126-136, 165-172;
  test_patient_stack.py:93-106).

## Corrections (factual)

- Stored-call facts: 25 in 4 runtime root files (mechanistic_bridges 19, supplement_evidence 3,
  lifestyle_evidence 2, linage2_core 1). Another 21 sit in the unloaded probe
  docs/kb_quick_wins/probes/mb_all.metta. Not 46.
- `(superpose ())`: 20 code lines in 8 root files (25 matching lines, 5 of them comments).
- Tests: 38 test_*.py modules, 14,621 lines. The 40 tracked .py files under tests/ include conftest.py and
  nhanes_xport_writer.py.
- Row #29 risk should read "needs rewrite": see O2 and O3.
- Blocking unknown #5 is largely settled from source: see O7.
- Snapshot amortisation is applicable through a concatenated temp file per stack.
- Phase 2 must include: adapter-level fresh-process guarantee, run_query_parts per part, the load-unblock
  sweep, per-call test isolation, and load verification that bypasses `PeTTa.process_metta_string`.
- Phase 5 must include a derive-once-per-request cache and a per-request gate.

Checked and found consistent with the verdict:
- One execution site (pln_runner.py:734-831).
- `(superpose ())` does not compile.
- `(empty)` exists in hyperon 0.2.10 (stdlib `(= (empty) Empty)`).
- The presence-test idiom still works.
- No function/fact head overlap and no arity mismatch in the KB.
- No cross-file forward references in `_INFERENCE_STACK` order.
- `(car-atom ())` throws, `remove-atom` is retractall, `halt(1)` sits only in assert/test (metta.pl:260, 265).
- The `True`/`False` parse to lowercase (parser.pl:82-84).
