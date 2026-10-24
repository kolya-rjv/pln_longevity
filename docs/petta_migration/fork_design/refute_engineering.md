# Engineering / PeTTa-reality refutation of the fork-design decision

Reviewer: adversarial, lens = PeTTa runtime, load times, consumer contract, hand-composed confidences, DrugAge/LinAge2 spaces,
150-row cap, validator, effort credibility, gate testability. Read-only. Nothing from `.petta/` was executed (read with cat/grep/sed/git only).
My only script: `refute_eng/affine_check.py` (python3 -I, pure arithmetic).
Tags: [src] upstream/repo line read; [re-run] I recomputed; [designer] taken from a note; unknown.

## 0. Verdict

Decision stands: D1 as the `pln` engine behind G0 with D3 as fallback is defensible. I found NO blocking objection.
I found 6 major defects in the decision text (gates, seam, parity, missing couplings, effort) that must be fixed in the LAUNCH edit
before an unattended run, because two of them make a gate unrunnable or unrepresentative.

## 1. Objections

### O1 (major) G0(a) is mis-specified: two of the four named examples never load the library, one needs a rule the shipped artifact deletes
Decision section 4 G0(a): "upstream examples pln_roman, pln_tuffy, plntest, plntestdirect give their pinned values" on the vendored library.
- [src] `.petta/PeTTa/examples/plntest.metta` and `plntestdirect.metta` have NO import: they define their own `clamp`, `Truth_Deduction`, `|-` (lines 1-60). They test PeTTa, not `vendor/pln_fork`.
- [src] `pln_roman.metta` expects `(stv 0.5 0.947...)` with stamp `(1 2 3 4)`: two Deductions (stamps 1 3 and 2 4) revised. Deduction is deleted by the decision's own restricted filter (lib_pln.metta:280-286). It can only run on the UNFILTERED file.
- [src] `pln_roman` and `pln_tuffy` import `../lib/lib_pln` (PeTTa's bundled copy), so "on the vendored library" needs an edited import line.
- [src] `pln_direct.metta` (imports `(library lib_pln)`, uses `Truth_ModusPonens` and `Truth_Revision` through `foldl-atom`, seed `(stv 0.5 0.0)`) is the one upstream example that exercises exactly D1's formulas, and it is not in the list.
- [derived by hand] `pln_tuffy`'s expected stamp `(1 2 4 9 10)` is reachable with Modus Ponens + Revision only (MP of 4 on 1, MP of 9, revision with 10, MP on 2), so it can run on the restricted artifact; I did not verify it runs. unknown.
- Net: no upstream test exercises NotElimination inside `PLN.Derive`, and none runs on the generated restricted file that will actually ship. G0(b)-(d) are the only coverage of the artifact.
Fix: G0(a1) unfiltered vendored file on pln_roman, pln_tuffy, pln_direct (import edited in a copy); G0(a2) restricted artifact on pln_tuffy and pln_direct; drop plntest/plntestdirect.

### O2 (major) The seam is specified two incompatible ways, and G0(e) measures neither the eager nor the cheapest case
- decision section 3 "Selector": `infer`/`explain` are "over answer atoms produced by a Python adapter" (eager: D2 mechanics). decision section 3 "Adapter, per source and request ... `infer` and `explain` read one table" and D1 design.md:285-302 sketch a lazy MeTTa `chain-table` memo. The cost model ("every infer = 2 runs", "10 runs for rank-interventions-for-patient") is the lazy one.
- [src] consumers bind the source AT EVALUATION TIME: `pln_linage2.metta:123` `(let $h (match $space (Inheritance $hh HallmarkOfAging) $hh) (let (signed Pos $tv) (infer $space $h $b) $h))`, `pln_counterfactual.metta:126`, `patient_profile.metta:355,390`. Answer atoms therefore must be precomputed for EVERY node with an out-link before the query runs: 30 sources x 2 runs = 60 runs on the patient stack (designer: 5,048 rule attempts, design.md section 7) against 2,216 attempts for the 10-run workload G0(e) times, and 6x the number of `process_metta_string` calls. A lazy `py-call` re-entering janus from inside a running query is a third option nobody probes.
- [re-run, spike_kit/baseline_hyperon.json] the cheapest infer probes cost 22-28 ms in-process / 150-190 ms per child on hyperon (smoke:2, smoke:3, R-explain); the LAUNCH 5x limit is then 0.8-0.95 s INCLUDING PeTTa cold start. Those probes pay the full eager precompute for sources they never use. Infer-shaped shapes: median 182 ms in-process, 766 ms per child (36 shapes).
- The decision says the LLM already sets `requires_pln_inference` (system_prompt.txt:13-14,127,244) but raw `/metta/run` has no such flag; the trigger rule for running the adapter is not stated.
Fix: choose one mechanism in the spec; add to G0(e) (i) a single-infer probe (smoke:2 shape) with the real seam, (ii) the all-sources precompute if eager; state the trigger.
Also: define (e) as the INCREMENT of the pln lane over the PeTTa-legacy lane on the same shape. As written ("within the Phase 1 five-times limit", vs hyperon legacy including load) a PeTTa cold-start failure that no design can fix would be read as a PLN.Derive timing failure and send the owner to the D3 fallback, which does not cure it.

### O3 (major) "R2 equals legacy" has a hole exactly where production data lands; a one-line fix exists
- [src] `drugage_calibration.metta:226` materialises `(Effect Lifespan Mortality Neg (stv 1.0 1.0))`; `expander.py` tells the extraction LLM "a life-extending compound is (Effect X Lifespan Pos ..)"; `tests/test_ontology_expansion.py:560-574` pins that path end to end (`infer Taurine Mortality`, `explain`, `rank-interventions`). Every paper that lands through `/ontology/expand` + `/ontology/apply` therefore chains through the 1.0 link.
- [re-run affine_check.py] with q = min(1, p+0.02): any p >= 0.98 gives slope 0.98 (p=0.985, 0.99, 1.0 all -> 0.98). Taurine (0.3548387096774194, then 1.0): R2 gives 0.347741935483871, legacy 0.3548387096774194 (designer's own pair 0.3477 / 0.3548, design.md:185). Caller `Effect` facts with strength in (0.98, 1.0] break the same way; `/metta/run` accepts them (api.py `_guard_extra_atoms` only refuses `(= ..)`).
- The decision handles it only for the DrugAge LIFT (virtual hop). Not for the materialised atom or caller facts. "Pinned numbers equal legacy, caller facts work" is overstated for p >= 0.98.
- [re-run] Fix with no cap: store q = e + (1-e)*p, report strength = |s1 - s0| / (1-e)^hops (hops is already needed for the 0.9^(hops-1) shim). 20,000 random chains with p drawn from {0, 1, 0.98, 0.99, uniform}, both signs, 1-7 hops: worst error 5.6e-16. It also removes the "virtual hop" special case.

### O4 (major) Effort and consumer-change list omit work the chosen design needs; the scoring against D2 is asymmetric
- decision section 2 charges D2 with a "two-file split" and "per-process equations"; D1 needs the same split: section 3 says the legacy `infer`/`explain` equations and DrugAge's third `infer` are "not loaded next to" the new ones. [src] `pln_deduction.metta` (helpers + `infer` eq.1-2 + `explain` eq.1-3) and `drugage_calibration.metta` (lift rules + the `Lifespan->Mortality` atom + `infer` eq.3, line 240) each mix what must stay with what must go. Under PeTTa an extra `(= (infer ..))` clause ADDS answers (filereader.pl asserts every `=` as a clause), so a filter at stack assembly or a split is mandatory.
- [src, contract/dependencies.md section 3,5] the file name is hard-coded in: `app.py:138`, `api.py:205`, `pln_runner.py:36,59,93,291` (5 production stack lists), ~10 test `KB_FILES` lists, `drugage_scoring.py:56,102,106` (regex on the file text), `test_drugage_ranking_contract.py:213` (substring `(= (chain-discount) 0.9)`), the registry/validator inventory (`ontology/loader.py` regexes, MUST 13: names must appear as `(= (name ..))` text in a LOADED file or valid programs get 422), and the translator prompt (files pasted verbatim, 25 KB cap). None is in "changes_to_consumers" (which says only "validator refuses PLNAnswer").
- `drugage_selector.py:41` `MAX_ROWS = 150`; [re-run] CSV: Resveratrol 170 rows, Metformin 54, Rapamycin 37 = 261 rows for the three compounds of gate 6b; `select_rows` truncates `kept[:limit]` by (compound,row id). The cap is a hyperon safety constant shared by both runtimes; running 6b through the engine needs a runtime-dependent cap or per-compound calls.
- Whole-build DrugAge ranking (3,423 rows, 1,046 names) cannot go through the engine (`drugage_scoring.py:15`); it is the Python twin. So the fold that decides "Ethanol not first" is a Python re-implementation of `Truth_Revision`, kept in parity by a test, not patham9's code running. [re-run] only 177 of 1,046 compounds have >= 2 papers; for the other 869 the paper-level fold equals the existing representative-row policy.
- hyperon "stays green" is not a testable gate: baseline `docs/petta_migration/baseline/summary.txt` = 32 failed, 1699 passed. Say "the same 32".
Estimate: 6a = 2.5 agent-phases covers the adapter, seam, validator, goldens and 186-pair gate; with the items above (two file splits and stack variants in 5 lists, registry/prompt wiring, runtime-dependent cap, twin parity, five-lane table over 186 pairs plus DrugAge) I read 4 as credible, 2.5 as optimistic. Phase 6 total 6-7 -> 8-9. unknown without PeTTa timings; flagged as judgement.

### O5 (minor) Measurement gates read quantities the vendored library cannot return
- [src] `PLN.Derive` returns only `($Tasks $Beliefs)` (lib_pln.metta:419); the only step counter is the `trace!` at :430 that hunk 1 deletes. G0(b) ("D+Q 12 selections, max 12") and 6c ("report selections used") cannot be read. The 6-argument form `(PLN.Derive T B steps maxsteps tq bq)` takes `steps` explicitly, so chunked resumption works: call with maxsteps = steps+c-1, feed the returned Tasks/Beliefs back with steps+c. Counting selections = chunk size 1.
- Same mechanism fixes K-schedule waste: the first attempt at K=256 on a 540-link sub-KB is cut at step 1 (initial beliefs 541 >= K, `LimitSize` keeps K-1, lib_pln.metta LimitSize) yet runs up to 4K steps x 255 beliefs ~ 1M attempts before reporting `partial`. Start K above |sub-KB| (D1's own rule of thumb bq = L + 2N + 10) and check the cut between chunks.
- An empty result from `!(PLN.Derive ..)` is a failure, not "no conclusions" (a `BestCandidate`/pattern failure returns nothing; LAUNCH "A query that fails without raising comes back as []"). Assert exactly one result.

### O6 (minor) Gate and rule wording
- Probes "that abort the load" must be Python comparisons: MeTTa `test`/`assert` call `halt(1)` on mismatch (metta.pl:256-266) and kill the janus process without an exception.
- Proposed hard rule "vendored fork library with its own example scripts": the fork's examples/ruletests need `examples/PLN.py` (runs `sh build.sh`, the hyperon `metta` CLI) and cannot run on PeTTa unmodified (compat 4.1: 0 of 11). Name the allowed scripts: PeTTa's `examples/pln_roman|pln_tuffy|pln_direct.metta` and the project's own tests.
- Sequencing: G0 sits in Phase 1 but the vendored + generated restricted artifact is a Phase 6 deliverable. Move the filter script (or a throwaway) into Phase 1.
- decision "library runs in upstream CI on 4-6 sentence KBs": pln_tuffy has 10 sentences; still tiny. Upstream coverage of 50-500 sentences is none.

### O7 (minor) The level-0 run is redundant compute
Both runs share one stamp set and differ by an affine map: after the stamp-rebuilt path is known, s0 and the difference follow in closed form (the adapter already multiplies hop terms for the shim). The engine run is needed for routes, stamps and completeness; keeping the second run only as a G0 self-check would halve the 60-run eager cost. Presentation point, not a defect.

## 2. Things I checked and could not break

| claim | check | result |
|---|---|---|
| bundled lib = file named in decision | `shasum -a 256 lib/lib_pln.metta` | d864c3cf...ed27ece, 465 lines, matches |
| line refs 406/430/370-376/255-257/260-262/328 | grep -n | rule lines 255/260/328, defaults 370/373/376, trace 430, `(- 0.0 $c)` present (E1 already in bundled file) |
| MP is affine in the premise: f1 premise, f2 implication | Rules.metta MP clause + Formulas MP | s = f1*f2 + 0.02*(1-f1); `affine_check.py` reproduces |
| names of lib functions vs KB heads/defs | regex over 49 lib names x all root `*.metta` + `metta_files/` | zero collisions; State/End/Sentence/Implication/Not not PeTTa builtins (grep registry) |
| variable-head result tuples `($B (Truth..))` dispatch | translator.pl:293-309 `reduce/2` | list heads stay data; only atom heads naming a function are called; D1 terms are lists |
| `case` with non-deterministic key | translator.pl:727-740,1482-1495 | key solutions tested one by one, no commit |
| completeness test needs tq >= bq | LimitSize semantics | sound: tasks are a subset of beliefs until the first belief cut; I agree with the judge |
| DrugAge facts | data/drugage/dataset.zip | Ethanol 52 rows / 1 paper; Resveratrol 170/37; Metformin 54/16; Rapamycin 37/18; max |avg change| strength 0.9305 < 0.98; all 3,423 rows have a numeric PMID |
| compat-report claims vs PeTTa src | 12 spot checks: trace! rewrite (translator.pl:412), println! stdout (metta.pl:250), no `(superpose ())` clause (translator.pl:1533), `Predicate` builtin (metta.pl:336,462), import! path rules (metta.pl:343-432), library-import!/library-origin (metta.pl:350, decl_store.pl:43,339), load cache semantics (load_cache.pl:1-80), test/assert halt (metta.pl:256-266), case translation, reduce, process_metta_string pass (filereader.pl:70-100), `-[det]->` check applies to library code too (703b7e2 message) | NO contradiction found |
| user-origin vendored copy | pln_roman/pln_tuffy import by relative path and run in CI | the plain-path (user-origin) load is exercised upstream; my initial concern (residual runtime guards, no trust) is moot |

## 3. Gate testability

| gate | testable as written? |
|---|---|
| G0(a) | no (O1) |
| G0(b) | partly: closure/levels yes; "selections" unreadable (O5) |
| G0(c) | yes if probes are Python-side (O6) |
| G0(d) | yes |
| G0(e) | ambiguous baseline and workload (O2) |
| 6a | yes: goldens come from recorded hyperon (`closure.json`), approx pins exist; exact goldens regenerate |
| 6b | partly: fixture tests yes; Ethanol-rank needs the ETL build (tests skip without it) and the Python twin, not the engine |
| 6c | report only |
| "hyperon stays green" | no: baseline is 32 failed (O4) |

## 4. Unknown (not established, no PeTTa run allowed)
Wall time of one `PLN.Derive`; cost of `process_metta_string` per `!` form with ~60 literal sentences (each call runs the whole-form unified analysis, filereader.pl:70-100); cold start of `consult(main.pl)` + stack + lib; whether nested janus calls from `py-call` inside a running query work; whether `pln_tuffy` runs on the restricted rule set.
