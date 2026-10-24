# Fix phase: the two reviews of the pln chainer, finding by finding

Work tree `/Users/kolya/PycharmProjects/pln_longevity_petta` (branch petta-migration), 2026-10-24, PeTTa `fea83178` on SWI-Prolog 10.0.2, `.venv` Python 3.12. Nothing was committed, reset, stashed or pushed;
the orchestrator commits. Every number below was measured by me in this phase: the "before" figures by re-running the reviewers' scenarios (or my own versions of them) on the unfixed tree
(`out/before_*.json`, `out/before_h*.txt`), the "after" figures on the fixed tree (`out/after_*`, `out/timing.json`, `suite/*.out`, `mut/results*.json`). The scripts are in
`docs/petta_migration/fork_design/fix/probes/` (repro.py, calib2.py, mk_etl.py) and the cost model with its calibration is `docs/petta_migration/fork_design/fix/COST_MODEL.md`.

## 0. Result

* 18 findings, 18 reproduced. 16 fixed (13 in code and tests, 3 in the documents; each code fix has a test that fails without it: 40 mutants of the fixes, 40 killed); 1 explained with evidence and documented as a curation
  question (Revision pools weight, not agreement); 1 documented where a code change would be arbitrary (result order). Details in section 1.
* The decision about the null convention (finding 1) is made, as a default the owner and Ben can flip: a reported null counts on BOTH sides. It changes the sign of two compounds, so it is the one
  thing in this phase they should look at (section 2).
* The DrugAge fold is wired into the chainer (hooks 1 to 5 of `b2/notes.md`): ranking by NET, once per compound, no row cap; whole-build ranking and the two semantics blocks.
* The number table and the control demonstration are regenerated from the fixed tree, and their prose corrected.
* Suites, in the order requested (section 7): default runtime 32 failed / 2552 passed, ids IDENTICAL to `hyperon_failures_47130aa.txt`; `PLN_RUNTIME=petta` 31 failed / 2560 passed, ids identical to `petta_failures.txt`
  (the 32 minus the hyperon-only abort probe); the pln-lane tests 375 passed in the default environment and 375 with `PLN_RUNTIME=petta PLN_CHAINER=pln` exported; the 145 shapes through the legacy chainer: classes unchanged,
  0 of 177 raw strings differ from the kit's run.

## 1. The findings

(Numbers: F1..F18 are the 18 findings in the order the task lists them.)

| # | finding | reproduced | verdict | what changed | test that fails without it |
|---|---|---|---|---|---|
| F1 | zero-change DrugAge rows enter only the protective pool: the NET sign of Resveratrol and Metformin is a convention artefact | Resveratrol -0.0460, Metformin -0.0570 with nulls on the protective side only; +0.1363, +0.1182 with nulls on both sides; +0.0321, +0.0629 dropped (fixture); whole build: 70 of 1,035 compounds have a null row, both-vs-protective flips the sign of exactly 2 (these), neither-vs-protective 8, both-vs-neither 6 | real, major | `zero_rows` ('both' default, 'neither', 'protective') in `drugage_fold` (every entry point), `ChainSettings.drugage_zero_rows`, `ZERO_ROW_CONVENTIONS`; a direction whose pool holds only nulls pools to strength 0 and is not an item (it adds nothing to the net and would give a consumer a second, empty derivation), a compound of only nulls keeps its protective item; the three NETs printed in the number table (section 8, four rows per compound) and in `b2/results/measure.json` (`null_conventions`); the table sentence rewritten | `test_resveratrol_and_metformin_leave_exactly_zero_and_the_sign_depends_on_the_null_convention`, `test_a_symmetric_example_keeps_its_sign_whatever_the_nulls_do`, `test_the_net_with_the_null_in_both_pools_typed_by_hand...`, the oracle comparison per convention over the 1,035 compounds, the whole-build ranking pin |
| F2 | no engine ranking returns the NET; the harmful item is dropped; B2's HTTP claim is false | `run_drugage_ranking(best_per_compound=False)` over the 313-row fixture, pln: Resveratrol +0.2145 (Neg item only) against the defined NET; `best_per_compound=True` (the only value a caller passes): identical to legacy, Rapamycin 0.81 | real, major | hook 2 `pln_runner.net_per_compound` (also under `metta_sort`), hook 3 (no cap, every row), hook 4 `drugage_top(fold=None)` following the chainer + entry fields + `fold` block, hook 5 `score_semantics()` and pooled provenance; `b2/notes.md` corrected; legacy response key-for-key unchanged | `test_run_drugage_ranking_lists_a_compound_once_by_its_net_and_equals_the_python_twin` (3 conventions x 2 strategies, 313 rows), `test_drugage_top_follows_the_chainer...`, `test_the_http_top_response_has_exactly_the_legacy_keys...`, `test_net_per_compound_keeps_one_derivation_untouched_and_nets_two` |
| F3 | eager all-sources derivation: a few cycles or ~1,000 routes make a failed request | 75-link layered DAG, 363 routes, one source asked: 15.5 s (legacy 0.2 s); patient stack + 14 random caller edges, 5 sources asked: seeds 24 and 28 killed at the 60 s deadline | real, major | `demand_of` (only the sources a program can ask about; every source when in doubt), `plan_sources` pre-flight, clock check, API.md "A cycle terminates" tempered | `test_demand_of_reads_the_expressions...` (18 cases), `test_a_literal_source_derives_one_source_and_the_answer_equals_the_eager_table`, `test_the_composite_queries_that_the_analysis_narrows_equal_the_legacy_chainer_run_alone`, `test_junk_beside_a_two_link_question...` |
| F4 | `fold_items` picks its within-paper policy for the whole request from one row | one unrelated row without a significance atom: Resveratrol -0.0460 -> -0.0395, Metformin -0.0570 -> -0.0403 | real, minor | the pick is decided per (compound, target) group | `test_one_compounds_missing_policy_keys_never_change_another_compounds_number` |
| F5 | out-of-range or non-numeric truth values skipped with no signal in the answer | `(stv 1.5 0.5)`: legacy 0.75, pln empty answer, no warning | real, minor (same defect as F11) | see F11 | see F11 |
| F6 | a route returning to its source through a hazard-ratio record: answered by legacy, dropped by pln | `explain Metformin Metformin` over a record InsulinResistance -> Metformin (HR 1.3): legacy `(Chain (Metformin InsulinResistance Metformin) Neg 0.45 0.65025)`, pln nothing | real, minor | `_trail(simple)` lets a closing record name a visited node (a record is terminal); `_routes` drops only the premise; an Effect cycle back to the source is still not a route | `test_a_record_that_closes_a_route_on_its_source_is_answered_as_the_legacy_chainer_answers_it`, `test_an_effect_cycle_back_to_the_source_is_still_not_a_route`, `test_the_trail_of_a_record_may_name_a_visited_node...` |
| F7 | docs over-state ("his Modus Ponens arithmetic", "never counted twice", "the fork's Truth_Revision") | the answers equal the legacy product by construction (decision.md section 9 says so; API.md did not); evidence ids are one per link; `revision` is a port | real, minor | API.md and the chainer docstring say it plainly; the fold docstring says "Python PORT of the fork's `Truth_Revision`" | `test_api_md_states_what_the_pln_section_does`, `test_the_module_docstrings_do_not_call_the_fold_the_forks_own_revision...` (and the existing parity test of the port against the library) |
| F8 | Revision pools weight, not agreement; near-1 capped confidences are order dependent | Rapamycin's 18 papers: confidence 0.978 whatever the spread; papers independent; 4.9e-11 over 200 shuffles at c = 0.999999 | NOT a defect | listed as an owner/Ben question next to the null convention (NUMBER_TABLE section 10, `b2/notes.md`, LAUNCH.md); no code change | none (nothing changed) |
| F9 | a bad PLN_QUEUE_SIZE / PLN_MAX_STEPS crashes `import config` under the default chainer | `PLN_QUEUE_SIZE=abc`, `PLN_MAX_STEPS=x`: ValueError at import | real, minor | `config._under_pln_only`: the four pln-only settings are parsed leniently (default + logged warning) unless `PLN_CHAINER=pln`, where they stay a configuration error | `test_a_malformed_pln_only_setting_cannot_stop_the_default_service_from_starting` (5 cases), `test_the_same_malformed_setting_is_a_configuration_error_under_pln` |
| F10 | `pln` cost not proportional to the question; the K estimate counts node-simple routes, the library derives link-simple walks | 20-link complete digraph and a 200-link chain killed at 40 s; 80 dense links beside a plain `match` query killed at 40 s; shipped graph + 8 reversed edges 2.3 s | real, major (same defect as F3) | `walk_stats`/`count_states` (distinct walks), `estimate_beliefs` on them, `plan_sources`, `_rungs` raises, `_check_clock`, adjacency built once, `_LineIndex`; cost model calibrated on 19 graphs | `test_a_complete_digraph_of_five_nodes_is_refused_in_milliseconds...`, `test_a_graph_whose_modelled_time_exceeds...`, `test_rungs_refuse...`, `test_the_clock_is_checked...`, `test_the_model_tracks_the_measured_cost_within_a_factor_of_two` |
| F11 | an unchainable caller link is skipped silently and comes back as a real "no" | HTTP shape `(Effect Foo Bar Pos (stv 1.5 0.5))` + `(Effect Bar Baz Pos (stv 0.5 0.5))`: legacy `(signed Pos (stv 0.75 0.225))`, pln empty, no warning; the same for -0.1, 50, nan | real, major | a skip whose origin is the caller's block is a `PlnInputError`: 422 `refused_form` naming the atoms ("the program's own facts cannot be chained"); a curated file's bad link is still skipped and listed (`PLNSkipped`) | `test_a_callers_own_unchainable_fact_is_refused_with_the_atom_named...`, `test_the_reviewers_http_shape_a_strength_of_one_and_a_half_is_a_422...`, `test_a_truth_value_that_cannot_be_chained_is_skipped_and_listed_when_a_curated_file_holds_it` |
| F12 | DrugAge fold errors fail the whole request as 502 where legacy answers | row with `AvgLifespanChangePercent` stated twice, a whole row pasted twice, a row under two compounds: `FoldItemError` -> 502 (legacy answers, with 4 to 64 duplicate derivations); `RevisionUndefined` for a pooled confidence of 0 | real, minor | identical copies are ONE item; a row whose copies disagree is refused (422, row named); the fold is called per compound so a refused group is a listed skip (`FoldRefused:<type>`) | `test_a_row_pasted_twice_is_one_study...`, `test_a_drugage_row_that_contradicts_itself_is_refused...` (2), `test_the_fold_is_called_compound_by_compound...` |
| F13 | API.md says `PLNAnswer` may not be read by a caller; reading is allowed | `match &self ($h A $b $c $d $e $f) $h` returned `PLNAnswer` atoms | real (wording), minor | "may not be WRITTEN"; and the sentence says what a read returns now (rows only for derived sources) | `test_api_md_states_what_the_pln_section_does` |
| F14 | new settings validated at import under the default chainer | `PLN_RULESET=bogus`, `PLN_CHAIN_CONFIDENCE=zzz`, `PLN_QUEUE_SIZE=`, ...: ValueError (5 cases) | real, minor (same defect as F9) | see F9 | see F9 |
| F15 | three behaviours no test pins | clamp `min(1.0, ..)` (33 of 180 raw strengths exceed 1.0), `steps_clamp`, tasks = intervention only | real, minor | tests added | `test_a_route_strength_never_exceeds_one...`, `test_the_step_bound_is_clamped_to_one_through_a_million`, `test_the_first_derive_of_the_restricted_lane_has_exactly_one_task...` (mutants M18, M19, M20 killed) |
| F16 | documentation drift and an overstated claim | API.md said 120 tests (122 collected); the DrugAge bullet did not say what the ranking endpoints do; "a cycle terminates" / "quadratic" omit the exponential cost on cycles; CONTROL_DEMO "a request must not run on" ran to the kill | real, minor | API.md section rewritten (test count 273, endpoints, null convention, cost, order, refusals); COST_MODEL.md; CONTROL_DEMO regenerated: contract mode on SYNTHETIC-L is refused in 0.2 s | `test_api_md_states_what_the_pln_section_does`; the rendered-block tests of the control demo |
| F17 | script hygiene: `eval` of strings, import-time environment write | `scripts/pln_number_table.py` `eval(ch["pin"][0], ..)`; `os.environ.setdefault(..)` at import of both scripts | real, minor | `arithmetic()` (an AST evaluator of number literals and + - * / ** only; names, calls, attributes refused); the environment is set in `main()` | `test_the_scripts_do_not_eval_a_data_file_and_write_no_environment_at_import` (M37, M38 killed) |
| F18 | result order differs from legacy (hops, stamp against DFS) | with a Pos and a Neg route to one target the two chainers list them in opposite orders; and (found in the 145 shapes) `pos-transmission` takes the FIRST positive route: with a second one injected, legacy picks (0.578, 0.222), pln the shorter (0.64, 0.38) | real, minor | documented in API.md ("do not take the first result", and the one consumer that does); no code change: the legacy order is the engine's match order, which differs between hyperon and PeTTa themselves, a driver has one positive route on every shipped graph, and pln's order is at least defined | `test_api_md_states_what_the_pln_section_does` (the sentence is pinned) |

## 2. The decision this phase makes that the owner and Ben should look at: the null convention

A DrugAge row whose change is exactly 0.0 has strength 0 and keeps its confidence (an ITP null: 0.9). The sign convention has no zero, so the first build filed it on the protective side only. One row of it is neutral (score 0); pooled by
Revision it is not: it pulls the protective mean toward 0 and leaves the harmful mean alone, so every null pushed the NET toward "harmful". The reviewer's table, and mine (`measure_fold.py`, section `null_conventions`):

| compound | rows / papers / null rows | nulls on BOTH sides (default) | nulls dropped | nulls on the protective side only (first build) |
|---|---|---|---|---|
| Resveratrol | 170 / 37 / 23 | **+0.1363** protective, rank 322 of 1,035 | +0.0321 protective, rank 650 | -0.0460 harmful, rank 878 |
| Metformin | 54 / 16 / 5 | **+0.1182** protective, rank 406 | +0.0629 protective, rank 600 | -0.0570 harmful, rank 892 |
| Rapamycin | 37 / 18 / 1 | **+0.3368**, rank 3 | +0.3368, rank 4 | +0.3368, rank 3 |
| Ethanol | 52 / 1 / 0 | **+0.1144**, rank 426 | +0.1144, rank 427 | +0.1144, rank 420 |
| MixCompound (hand-built, 5 rows, one ITP null) | 5 / 5 / 1 | **+0.1757** | +0.1423 | -0.0452 |

Both symmetric conventions say the two headline compounds are PROTECTIVE; only the one-sided one makes them harmful. I chose "both sides" because the repo's own semantics say a well-run null is high-confidence evidence of NO effect
(`SCORE_SEMANTICS.zero_score`, the ITP rule): that is evidence about both directions, and dropping the nulls would discard exactly the evidence the semantics call high-confidence. It is one parameter
(`ChainSettings.drugage_zero_rows`, `drugage_fold.ZERO_ROW_CONVENTIONS`), so the choice can be reversed without touching anything else; the number table prints all three. What stays unresolved, and is not a code question: Revision pools the weight of
evidence, not its agreement (Rapamycin's 18 papers give confidence 0.978 whatever their spread), and treats papers as independent.

## 3. Before and after, the scenarios of the reviews (fresh PeTTa child per request; deadline 40 s unless noted)

| scenario | before | after |
|---|---|---|
| layered DAG, 75 links, 363 routes, `explain` one source | 15.5 s (legacy 0.2 s) | 3.0 s (two runs of that one source; the other sources are no longer derived) |
| layered DAG, 72 links, 120 routes, one source | 2.9 s | 0.6 s |
| complete digraph on 5 nodes, 20 links, `explain A E` | killed at 40 s | refused in 0.2 s: "more than 16,384 distinct walks over 20 reachable links" (422) |
| chain of 200 links, `explain C0 C200` | killed at 40 s | refused in 0.2 s at a 40 s budget ("would take about 32 s ... more than the 40 s"); runs at the default 60 s (modelled 33 s) |
| 80 dense links beside a plain `match` query that never calls `infer` | killed at 40 s | 0.2 s (nothing derived) |
| shipped patient graph + 8 reversed edges, `explain CellularSenescence CHD` | 2.3 s | 0.5 s |
| patient stack + 14 random caller edges, 5 sources, seed 24 (reviewer's `h5_patient_random.py`) | killed at the 60 s deadline | 27.6 s, 403 routes, 0 differences from the reference |
| the same, seed 28 | killed at the 60 s deadline | 36.7 s, 395 routes, 0 differences |
| caller `(Effect Foo Bar Pos (stv 1.5 0.5))` ..., `infer Foo Baz` | legacy `(signed Pos (stv 0.75 0.225))`; pln empty answer, no warning | pln 422 `refused_form`, the atom named |
| `explain Metformin Metformin` through a record back to the source | legacy answers; pln nothing | both answer (equal to 1e-12) |
| DrugAge row pasted twice, `infer X Mortality` | legacy 64 results; pln 502 `FoldItemError` | pln one result |
| `PLN_QUEUE_SIZE=abc` (and 4 more), default chainer | `import config` raises | imports with defaults and a logged warning |

Request cost on the patient stack, legacy chainer 0.395 s on the same runtime (median of 9): no `infer` +0.014 s; a question about one source +0.060 s; every source (unbound source, or a consumer) +0.125 s (ratio 1.04, 1.16 and 1.32).

## 4. DrugAge wiring, status of the B2 hooks

1 (`change`, `significance`, `itp` on every item): already built into `collect_links`; a parity test now holds its links equal to `fold_rows`. 2 (list once, by NET): `net_per_compound`. 3 (no row cap): the slice holds every row
(313 for the four fixture compounds, past the 150 cap). 4 (whole-build ranking): `drugage_top(fold=None)` follows the chainer, entries carry `papers_protective/papers_harmful/rows_protective/rows_harmful`, the response a `fold` block, and the
default JSON is key-for-key unchanged (pinned). 5 (semantics): `score_semantics()`, and the chat route's pooled provenance bullets. Hook 6 (expose `tier*gate` on `RowScore`) and 7 (stamp ids from the fold) were optional and are not done.
Rapamycin over the engine path: NET +0.3368, confidence 0.8801 (it was 0.81 through the endpoint before, because the endpoint handed the engine one row).

## 5. The design decisions I made

* A caller's unchainable fact is a 422, not a skip with a warning: a skip answers "no route", which API.md defines as a real no, and the response model has no field for the engine's notes. Curated files keep the listed skip.
* Cost control is a model plus a demand analysis plus a clock, not a smaller K: `demand_of` is conservative by construction (any consumer, any unbound or computed source, any program that reads `PLNAnswer`/`PLNStatus` means every source), `plan_sources` refuses only
  what is far past the budget (80 percent of the request's), and the research lanes are never refused. The model is fitted to 19 graphs (measured / model 0.45 to 1.12; it over-estimates small graphs, never under-estimates the large ones by more than 12 percent).
* The answer table is written for the sources that were derived; `PLNEvidence` and `PLNSkipped` for every program (they depend on the links only). `PLNStatus` therefore lists derived sources only.
* A closing record may name a visited node; an Effect cycle back to the source is still not a route (legacy does not terminate on it).
* The research mappings (`literal`, `capped`) no longer run the sign cross-check: it compares the library's difference with the product, which is true only for the affine mapping. For the literal reading a link weaker than the fork's leak comes back with a difference of the wrong sign, which is what that lane exists to show.

## 6. Tests

New: `tests/test_pln_fixes.py` (74 tests). Extended: `tests/test_drugage_fold.py` (64 -> 76: the three conventions, the hand-typed MixCompound under both, the symmetric example, the empty-direction rule, per-group exactness, the bit-exactness of a net beside a null, the default ranking pin; the oracle `fold_oracle.py`
takes `zero_rows`, written in its own words, and is compared with the module on all 1,035 compounds under each convention) and `tests/test_pln_chainer.py` (122 -> 124: the skip test split into the curated and the caller case; `sim_table(plans=)`; the summary test asks for every source). Where an old test pinned the first build's null
convention it now says `zero_rows="protective"` explicitly (the same assertion, on the named behaviour) and a new assertion covers the default. All the test files edited here were added by this branch (they are untracked in git); no tracked test file of the baseline suite is modified (`git status`: tracked changes are source and API.md only), and no existing assertion of the baseline suite was weakened.

Mutation check of the new tests (`mut/run_mutants.py`, `mut/results.json`): 40 mutants, each removing ONE fix from a copy of the tree (config laziness, the walk count, the pre-flight and its two refusals, `_rungs`, the clock, the demand analysis and its
three guards, the caller refusal, the unknown-sign origin, the record rule, the premise guard, the DrugAge dedupe and the per-compound fold, the strength and step clamps, the intervention-only task, the null convention and its three parts, the per-group exactness, the
net's bit-exactness, the empty-direction rule, the four DrugAge ranking hooks, the semantics and provenance, the HTTP key sets, the two script rules): 40 killed, 39 by `tests/test_pln_fixes.py` or `tests/test_drugage_fold.py`, 1 by the caller-refusal test of
`tests/test_pln_chainer.py`. One mutant survived the first run (the net rebuilt a strength as score / confidence even when the other part was a null, which differs from the item's own strength in the last bit); the test that kills it was added and is the
reason for `test_a_null_beside_a_real_item_leaves_the_real_items_strength_to_the_bit`.

## 7. Verification

All runs were made on the final tree, one after another, nothing else running (`suite/run2/`):

| run | result | against |
|---|---|---|
| default runtime, `PLN_RUNTIME=hyperon PLN_CHAINER=legacy`, `--deselect tests/test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move` (the known trie.rs:179 abort, the baseline's 1 deselected) | **32 failed, 2552 passed, 22 skipped, 1 deselected in 513 s** | the 32 failure ids are IDENTICAL to `docs/petta_migration/baseline/hyperon_failures_47130aa.txt` (`diff` of the sorted ids is empty) |
| `PLN_RUNTIME=petta PLN_CHAINER=legacy` | **31 failed, 2560 passed, 16 skipped in 377 s** | identical to `baseline/petta_failures.txt`, which is the 32 minus `test_human_evidence.py::test_the_runtime_kb_still_answers_the_query_that_aborted_the_interpreter` (a hyperon-only abort probe that passes on PeTTa by nature); nothing added |
| the pln-lane tests: `test_pln_chainer`, `test_pln_fixes`, `test_drugage_fold`, `test_pln_vendor`, `test_pln_number_table`, `test_prompt_size`, `test_drugage_discovery`, default environment | **375 passed in 80 s** | |
| the same with `PLN_RUNTIME=petta PLN_CHAINER=pln` exported | **375 passed in 83 s** | (a drugage test that compared `drugage_top()` with the legacy ranking now says `fold=False`: the whole-build ranking follows the chainer by design) |
| the whole suite with `PLN_RUNTIME=petta PLN_CHAINER=pln` exported (information) | 43 failed, 2548 passed, 16 skipped in 413 s | the 31 of the petta baseline plus the 12 that fail by design under `pln` (9 exact-digit goldens of the legacy chainer, 3 runtime-switching tests): the same 43 ids as the engineering review's list |
| the 145 query shapes, the kit's own run (`scripts/petta_spike/run_shapes.py`, direct adapter) | {identical 103, number_format_only 8, ordering_only 4, different 25, error 3, no_baseline 1} | UNCHANGED from the recorded {103, 8, 4, 25, 3, 1} |
| the 145 shapes through the production path, legacy chainer (`pln_runner._runtime_run`, 177 expressions) | raw strings identical to the kit's run for all 177 (0 mismatches); against the hyperon baseline {identical 135, number_format_only 8, ordering_only 4, different 25, error 3, no_baseline 2} | classes unchanged |
| the same through the pln chainer, against the legacy chainer on the same runtime (information) | {identical 148, within_tolerance 24, ordering_only 1, different 1, error in both 3} | the one `different` is the `pos-transmission` first-answer probe (F18); the 24 are last-digit differences |

Also: `tests/test_pln_chainer.py` 124 passed (it was 122); `tests/test_drugage_fold.py` 76; `tests/test_pln_fixes.py` 74; the number table (six lanes, 5 min) and the control demonstration (4.5 min) regenerated and re-rendered, `tests/test_pln_number_table.py` 30 passed (it re-renders every
generated block from the stored data); the fold measurements `b2/results/measure.json` regenerated.

## 8. For the owner and Ben

1. The null convention (section 2): both sides (default), dropped, or the protective side only. And is Revision's weight-not-agreement, papers-as-independent pooling what is meant?
2. `PLN_CHAINER=pln` still must not become a default; nothing in this phase moves toward that.
3. Contract mode now answers 422 for a graph it would have to run into the deadline for; the control lane is the place for such graphs. If Ben chooses a large real network, the first number to look at is the modelled time in `COST_MODEL.md`.
4. Not done: `GET /health` does not report the chainer; PeTTa `session()` and hyperon have no step; hook 6 and 7 of the fold notes.

## 9. Files

Source: `pln_chat/core/pln_chainer.py`, `pln_chat/core/pln_runner.py`, `pln_chat/core/drugage_router.py`, `pln_chat/core/metta_runtime.py` (error mapping), `pln_chat/ontology/drugage_fold.py`, `pln_chat/api.py`, `pln_chat/config.py`, `pln_chat/API.md`, `scripts/pln_number_table.py`, `scripts/pln_control_demo.py`.
Tests: `tests/test_pln_fixes.py` (new), `tests/test_drugage_fold.py`, `tests/test_pln_chainer.py`, `tests/fixtures/drugage_fold/fold_oracle.py`.
Docs: `docs/petta_migration/fork_design/{NUMBER_TABLE.md, CONTROL_DEMO.md, b2/notes.md, b2/measure_fold.py, b2/results/measure.json, c/results/*.json, fix/}`, `docs/petta_migration/LAUNCH.md`.
