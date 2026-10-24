# Decision: which design puts patham9's PLN behind infer/explain

Judge's record, 2026-10-24. Work tree `/Users/kolya/PycharmProjects/pln_longevity_petta` (branch petta-migration, content = linage2 @ 47130aa), read only.
Upstream checkouts under `.petta/` were read with cat/grep/sed and never executed. My own scripts are in `decision/checks/` (standard library only,
run with `python3 -I`; they load the validated simulator `sim/fork_sim.py` by absolute path and the link/closure data of the contract study).
Nothing was run on PeTTa; every statement about the real PeTTa runtime is therefore `[S]` (source read) or `unknown`.

Tags: `[re-run]` = I reproduced it myself; `[src]` = I read the upstream source line; `[designer]` = taken from a design note, not re-checked; `unknown`.

---------------------------------------------------------------------------------------------------------------------------

## 0. Decision

**Build Design 1 (`mp_state`) as the `pln` chainer, with three corrections taken from the other two designs and two corrections of my own, behind a gate that falls back to Design 3.**

* Engine: Design 1. Every `infer`/`explain` on the shipped networks is computed by patham9's `Truth_ModusPonens`, `Truth_Negation`, `StampDisjoint` and his confidence-ranked `PLN.Derive` loop. The numbers equal the legacy ones by construction of the mapping (two named constants), the library file is untouched except for one trace-removal hunk.
* Taken from Design 2: (a) the evidence unit is the paper (PMID), not the DrugAge row; (b) the completeness test "tasks empty and fewer than K-1 beliefs" with a K schedule, answers handed to MeTTa as validated answer atoms; (c) the noisy-copy `p + (1-p)s` mapping is the only route by which the fork's Deduction reproduces the legacy product, so it is the mapping of the optional Deduction comparison lane, not of the default.
* Taken from Design 3: (a) the hybrid (legacy engine, fork for revision and cycles) is the fallback if gate G0 fails; (b) the honest framing for Ben (section 9); (c) the best-first measurement on synthetic networks as the control demonstration.
* My corrections: (1) Design 2's completeness test is blind unless TaskQueueSize >= BeliefQueueSize, so the seam sets both to one K [re-run: 11 of 11 task-queue cuts undetected when tq < bq, 0 of 86 undetected when tq = bq]; (2) the planned DrugAge prototype demonstrates `Truth_Revision`, not the priority-queue control, so the control needs its own demonstration (section 5).

Why not the others, in one line each:
* Design 3 keeps every pin and has the least PeTTa exposure, but the fork never produces a number or a conclusion on an acyclic network; its own verdict is "mostly a revision step plus a safety net". It fails criterion 2. It is the right fallback, not the target.
* Design 2 is the most principled probability mapping (and its "base rate is a gauge" result is a genuine finding), but it needs a change to patham9's Deduction confidence line (otherwise 12 of 29 replayed invariants break), 60 per-node STV equations that accumulate per process on PeTTa, mirror sentences, cycle-guard ids and a two-file split; and it accepts only the exhaustive closure, so the control never binds. Highest PeTTa exposure, most invasive.

---------------------------------------------------------------------------------------------------------------------------

## 1. What I re-checked (all [re-run] unless marked)

| # | Claim (design) | My result | Script |
|---|---|---|---|
| 1 | D+Q -> CHD legacy Neg 0.2499 / 0.17017138125 | s = 0.24989999999999998, c = 0.17017138125000003 from links.json by right-nested fold | ck1_arith |
| 2 | D1: raw levels 0.13769 applied / 0.36029 absent, risk difference -0.222599; compensated -0.249900000; native c 0.23343125; shim c 0.17017138 | all five equal | ck1_arith |
| 3 | D3: fork-native MP along Up/Down, start fact (1,1): 0.26863439999999994 / 0.23343125 | equal | ck1_arith |
| 4 | D2: noisy-copy deduction, p + (1-p)s, composes to the legacy product; all 5 bracketings; p-independent; literal reading 0.28519 (p=0.1), 0.5112 (p=0.5); min-confidence 0.65 | unshifted 0.2499 for all 5 bracketings at p = 0.01, 0.1, 0.5 (spread <= 4.4e-16); literal 0.28519 and 0.5112; c = 0.65 | ck1_arith |
| 5 | Truth_Revision folds: D3 six-row (0.497903 / 0.911715 / 0.820543), D1 MixCompound (0.263050 / 0.949466 / 0.854519), D2 MultiDrug (0.488278 / 0.908686) | all equal, from my own transcription of Formulas.metta:57-73 | ck1_arith |
| 6 | Truth_Revision gives no result at c = 1.0 (tier RCT_Human) | `/safe` has no result for a zero divisor: Formulas.metta:59-60, Utils.metta:34-37 `[src]`; reproduced | ck1_arith |
| 7 | Ethanol: 52 rows of ONE paper, revised as independent evidence, c 0.21 -> 0.933 (D2) | 52 rows / 1 paper in the repo's `data/drugage/dataset.zip`; c = 0.9325. Resveratrol 170 rows / 37 papers, Metformin 54 / 16, Rapamycin 37 / 18, 3,423 rows in all | ck1_arith + zipfile read |
| 8 | D1 closure: 100 of 100 patient pairs, sign, strength, confidence (shim) and chain equal legacy; max 12 selections; 20/20/200 and 100/10/100 both complete with Tasks = (applied) | my own re-encoding: 100/100 exact at both settings (|d| < 1e-12, chains equal), 0 outside the closure, 0 missing, max 12 selections, 0 unfinished | ck2_closure |
| 9 | D3 closure with Up/Down + End: 100/100 bit-identical under the legacy fold, max 11 selections | 100/100 bit-identical, max 11 | ck2_closure |
| 10 | Seeding every sentence as a task (what `PLN.Query kb term` does) loses most pairs: 52/100 at fork defaults, 66/100 at PeTTa lib defaults | 52/100 (fork: reversing Unique, 20/20/200) and 66/100 (PeTTa lib: order-keeping Unique, 100/10/100) exactly; other combinations 47 and 55: the result depends on the Unique order | ck2b_unique |
| 11 | Feedback edge CI -> CS: terminates; 109 pairs (9 new); D+Q 12 selections (D1) / 11 (D3); max over sources 16 / 15 | exactly 12 / 11 / 16 / 15 / 109 | ck2_closure |
| 12 | Patient001 personalised order: legacy [DQ, Fisetin, Metformin, Berberine, Spermidine]; native c1*c2 flips Spermidine above Berberine (0.1041 vs 0.0948); Deduction minimum puts Spermidine third (0.4799) | legacy reproduces the pinned 0.2306 / 0.1647 / 0.0886 / 0.0768 / 0.0712 to 4 digits; native 0.3036 / 0.2169 / 0.1094 / **0.1041 / 0.0948**; minimum 0.9098 / 0.6499 / **0.4799** / 0.198 / 0.1716. Patient002 order unchanged in all three | ck5_orders |
| 13 | diagnose: legacy c(CS) 0.6253, mass 0.4593; native 0.6961 / 0.5391 (outside the 1e-2 tolerance) | equal | ck8_diagnose |
| 14 | In-loop Revision cannot pool many rows (D3: exact to 12, fails beyond; D1: 1 of 8 at equal confidence; D2: exponential) | Revision-only rule set, rows as tasks and beliefs: all N <= 12 merged exactly at both default budgets, but the loop always spends its whole budget (20 or 100 selections); N = 20: 17 of 20 (fork defaults) and 7 of 20 (PeTTa defaults); N = 52 (ethanol): 5 to 6 rows merged. **D1's "1 of 8" is not reproduced** in this isolated setup (their run had the other rules and the rest of the KB present); the robust finding is D3's | ck3_revision_loop |
| 15 | D3: the fork's queue is a bounded best-first search: first k distinct selected states = top-k by best-path confidence | recall 1.00 at k = 10 and 20 on two synthetic layered DAGs (168 and 540 links); 0.975 at k = 40 on the small one. 540 links, 29,523 legacy paths: 40 selections reach 64 of 137 states. Silent partial coverage (79 of 137 after 160 selections) | ck4_bestfirst |
| 16 | Full rule set creates conclusions outside the closure (D1 64, D2 266, D3 6 states) | in my encoding (uniform STV 0.5/0.9, tasks = KB + fact, 150/100/400): 10 pairs outside the closure from 4 sources (CellularSenescence, SASP, DNAmPAI1 -> AgeAccelGrim / AllCauseMortality / CongestiveHeartFailure; AgeAccelGrim -> DNAmPAI1) and 18 legacy pairs missed in budget. The count is encoding- and budget-dependent (10 / 64 / 266 / 6); the sign of the finding is not | ck6_fullrules |
| 17 | D1 budgets: steps 2N+2, tq max(20,N+2), bq L+2N+10 complete on the shipped graph | complete for 30 of 30 sources (max 12 selections) | ck7_control_seam |
| 18 | D2: completeness test "tasks empty and |beliefs| < K-1" proves nothing was cut | true only if tq >= bq. tq = bq = K: 86 of 86 cut runs flagged. tq < bq (tq = 2, bq = 200): 0 of 11 flagged, pairs silently lost | ck7_control_seam + inline |
| 19 | Exact goldens cannot hold under `pln` (D1: 64 of 186 strength strings, 47 of 186 confidence strings differ in the last digit) | 34 of 100 strength strings and 25 of 100 confidence strings differ; max |d| = 1.7e-16 | inline in ck2 |
| 20 | Fork semantics: Tasks = Beliefs in `PLN.Query kb term` (Deriver.metta:74-75); Unique prepends, so lists reverse each step (Utils.metta:85-92); LimitSize keeps size-1 items (Deriver.metta:27-31); priority is confidence only (Deriver.metta:19, 37); engraved/unary results keep the premise stamp (Deriver.metta:45); Config defaults 20/20/200 (Config.metta:4,7,10) vs the bundled lib 100/10/100 (lib_pln.metta:370-376) | all `[src]` as stated | read |
| 21 | MP is `(f1 f2 + 0.02 (1 - f1), c1 c2)` (Formulas.metta:45-48) with the author's own comment that min overestimates "as we do not have access to node probabilities" | `[src]` | read |

Not re-checked: any PeTTa run; hyperon bit-identity of Design 3's hooked text (+28 to +31 percent cost); the ETL-build DrugAge pins (0.81, 0.315); D2's 29-invariant replay beyond the Patient001, Patient002 and diagnose orders; D1's 300-chain algebra test and the 510-link cost counts.

Findings from the re-check that the designers did not state:

1. **There is no canonical "fork number" for D+Q -> CHD.** Four literal readings of the same fork give four values: 0.2226 (D1 raw difference), 0.2686 (D3 Up/Down start fact), 0.2852 (D2 literal at p = 0.1), 0.5112 (D2 literal at p = 0.5), against legacy 0.2499. Any pin under `pln` is the pin of a mapping. This is why the mapping is chosen to reproduce the legacy product, and why the number table must name the lane.
2. **The planned DrugAge prototype is not a demonstration of the control** (check 14). The loop that would do the merging always burns its entire budget and cannot pool beyond about a dozen rows; all three designs moved the merge out of the loop into a `Truth_Revision` fold.
3. **The fork's priority is confidence only** (Deriver.metta:19). On the application's network what matters is effect times confidence (`rank-score`); a high-confidence, tiny-effect path is expanded first. That is a question for Ben (section 8), not a defect.
4. **LAUNCH.md says `Implication A (Not B)` "never chains"**. That is true of the newer trueagi lib. The fork has a NotElimination rule (Rules.metta:76-77) and Design 1 chains 100 of 100 pairs through it.
5. **The compat study's patch E1 (unary minus) is already in the bundled lib** (lib_pln.metta:406 `(- 0.0 $c)`); only the trace hunk (lib_pln.metta:430) is needed.
6. **Induction call site permutes slots.** Rules.metta:39-44 passes `(STV $A) (STV $B) (STV $C)` with the shared premise node in `$C`; Truth_Induction (Formulas.metta:23-31) expects the shared node in its second slot. Invisible under uniform priors; relevant only to the full-rule lane. A question for Ben or patham9.

---------------------------------------------------------------------------------------------------------------------------

## 2. Scoring against the criteria (in the owner's order)

| | D1 mp_state | D2 up/down deduction | D3 hybrid |
|---|---|---|---|
| 1 Keeps contract behaviour | Yes under the shim. No false chain (100/86 pairs, 0 extra), signs, chains, one derivation per pair, empty for unconnected, terminates on cycles, caller facts without precompile (KB built per call). Orderings hold; with native confidence the Patient001 order flips 4th/5th and diagnose c(CS) leaves its tolerance. Goldens regenerate (last digit). | Yes in lane P, which needs a patch to patham9's Truth_Deduction confidence line. Unpatched: 12 of 29 replayed invariants break (Patient001, NMN > Fisetin, counterfactual c(B) < c(A), every chain value pin). Goldens regenerate. | Yes, trivially: legacy equations produce every number (hybrid bit-identical on 186 pairs, designer-run on hyperon). Only DrugAge multi-row output and cyclic sources change. |
| 2 Genuinely patham9's PLN and his control | Rules MP, NotElimination, Revision-formula; his loop, stamps, queue limits, confidence ranking on every query. The numbers are his MP arithmetic with its 0.02 leak cancelled by a two-run difference and a stored +0.02. Adapter does path rebuild, subtraction, shim. Control never binds on 30 nodes. | His Deduction (the syllogistic rule) with node probabilities, his Revision. But the base rate is a gauge (output independent of p), confidence is patched, and the budget schedule demands the exhaustive closure, so priority never matters. | Fork supplies a revision formula, a cycle fallback and a second-opinion window; legacy supplies every acyclic conclusion. Designer: "does not let him say the app now reasons with patham9's PLN". Fails. |
| 3 PeTTa risk | Moderate. No STV calls, no Translator, no per-request equations (sentences are passed as data), no Predicate collision. The library runs on PeTTa in upstream CI (examples/pln_*, test.sh) on 4-6 sentence KBs; speed on 50-500 sentences unknown. Every `infer` goes through PLN.Derive twice. | Highest. 60 STV equations per process (accumulate), forward-declared STV, division by zero raises for p = 0, patched formula, per-source retries, answer atoms, two-file split. | Lowest. Critical path stays on hyperon/legacy; fork needed only for revision (Python twin as fallback) and cycles. The MeTTa seam is an unrun sketch. |
| 4 Effort | about 5 agent-phases (designer) | about 6 | about 6 (2.5 for the revision part alone) |
| 5 Pinned numbers that move | Under `legacy`: none. Under `pln` (R2): all approx pins hold (|d| <= 1.7e-16); 32 exact goldens regenerate (34 and 25 of 100 strings change in the last digit); DrugAge: Resveratrol and Metformin leave 0, Rapamycin moves. | Same as D1 in lane P; lane N breaks 12. | Under `legacy` and acyclic hybrid: none. DrugAge multi-row only (same changes as the others). |

Verdict: D1 on criteria 2, 3 and 4 against D2; D3 wins 1 and 3 and loses 2 outright; D1 and D3 tie on 5. Criterion 2 is the one the owner and Ben asked for, so D1 is the target and D3 the fallback.

---------------------------------------------------------------------------------------------------------------------------

## 3. The chosen design (spec for the Phase 6 build)

**Vendoring.** PeTTa's bundled `lib/lib_pln.metta` @ fea83178 (sha256 d864c3cf376ad1f9252d153306d92b7aae72a184701c2eb4805c9661eed27ece), which is the fork's code ported to PeTTa by its author, to `vendor/pln_fork/lib_pln_fork.metta` with a provenance header, `LICENSE.NARTECH`, `LICENSE.PeTTa` (both MIT) and `PROVENANCE.md`. One hunk: drop the per-step `trace!` in `PLN.Derive` (lib_pln.metta:430). The restricted rule set is produced by a form-level filter script (keep the Revision, ModusPonens and NotElimination forms, byte-identical to the vendored file, with a test), not by hand edits. No Translator, no `PLN.Init`, no `build.sh`, no metta-morph. Load-time probes that abort the load: `Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8)` equals (0.26, 0.64); `Truth_Revision` with c = 1.0 gives no result (the fold caps c at 0.999999).

**Selector.** `PLN_CHAINER=legacy|pln`; legacy is the default; `pln` needs `PLN_RUNTIME=petta`; hyperon never loads `pln_chainer_state.metta`. Under `pln` a new file defines `infer`/`explain` (same names, same three arguments, same output shapes) over answer atoms produced by a Python adapter in the PeTTa child; the legacy equations and DrugAge's third `infer` equation are not loaded next to it; the validator refuses a caller-supplied `(PLNAnswer ..)` head.

**Encoding (Design 1).**

* Node X is the term `(State X)`. No per-node sentence and no base rate.
* Link `(Effect A B Pos|Neg (stv p c))` with id k: `(Sentence ((Implication (State A) (State B)) (stv q c)) (k))` for Pos, consequent `(Not (State B))` for Neg (MP derives `(Not (State B))`, NotElimination derives `(State B)` with 1 - s and the same stamp). Hazard-ratio record: consequent `(End B)` (or `(Not (End B))` for HR < 1), so a record can end a chain and never continues one. `q = min(1.0, p + e)`, where `e` is read at load from `Truth_ModusPonens (stv 0.0 1.0) (stv 0.0 1.0)` (currently 0.02).
* Intervention A: the single task `(Sentence ((State A) (stv 1.0 1.0)) (0))`; reference run `(stv 0.0 1.0)`. Premise id 0 and c = 1.0 are asserted (an empty stamp with c < 1 floods the queue; designer-run, 5,294 derivations).
* Adapter, per source and request: (1) BFS pre-filter of the links reachable from A; (2) two `PLN.Derive` runs; (3) for every `(State C)`/`(End C)` belief of run 1 with a node-simple path rebuilt from its stamp, join to the run-0 belief with the same stamp; (4) strength `|s1 - s0|`, sign = parity of Neg links on the path (cross-checked against the sign of s1 - s0), confidence `c_native * 0.9^(hops-1)` with the 0.9 read from `chain-discount`; (5) drop A = C, one result per route, sort by (hops, stamp). `explain` and `infer` read one table. Levels are never exposed (they are artefacts of the 0.02 leak).
* Evidence ids (changed from LAUNCH): one per independent source record: curated Effect atom, hazard-ratio record, DrugAge **paper** (PMID; inside a paper the existing representative policy; rows with no PMID get their own range). The same study restated by two layers carries the same id. Side table id -> (kind, file:line or PMID, evaluated tier) travels with every result.
* DrugAge: lift each row as `drugage_calibration.metta` does, emit the pair of lifespan-direction items onto Mortality with the sign flipped and one virtual hop (the materialised `Lifespan -> Mortality (1.0 1.0)` link would cost 2 percent through the leak), group per compound and direction, fold per paper-level item with the fork's `Truth_Revision` (c capped at 0.999999), stamp = union of the paper ids. Output at most one Neg and one Pos per compound; ranking uses the net of the two. `fold = false` reproduces the legacy per-row results (the lift-fidelity test).
* Revision rule in the loop stays registered and is inert by design (every derivation shares premise id 0), which keeps "one result per route" for diamonds.

**Rule set.** Restricted: ModusPonens + NotElimination + Revision. Comparison switch `PLN_RULESET=full` (all 17) and optionally `restricted+deduction` on the Design 2 mapping; both are research lanes only, never reachable from a request.

**Base rates.** None for the restricted set (no rule in it calls `STV`; Formulas.metta confirms MP, Negation and Revision never look up a node). Only the comparison lanes need them: uniform strictly positive `p = 0.1`, node confidence 0.9 (LAUNCH's values), p in (0, 0.5] so the consistency check passes for any strength (Constraints.metta:11-15); every `(State X)`/`(End X)` gets an equation or the rule silently derives nothing; p = 0 raises a division error.

**Control seam.**

| parameter | exposed as | default | reason |
|---|---|---|---|
| MaxSteps | `PLN_MAX_STEPS`, per-call argument of the Python adapter (clamped), never from `/metta/run` | `4K` | loop ends early when the task queue empties (max 12 selections on the shipped graphs) |
| TaskQueueSize and BeliefQueueSize | one value `PLN_QUEUE_SIZE` = K sets both | schedule K = 256, 2048, 16384 | the completeness test is valid only when tq >= bq (check 18); LimitSize keeps K-1 |
| completeness status | returned with every table | `complete` iff tasks empty and `|beliefs| < K-1` | contract mode: retry with the next K, then fail the request (HTTP failure, as other upstream failures). Control mode: return the belief set flagged `partial` |
| tq and bq separately | control lane only | - | they are the beam width and the memory of Ben's heuristic; no completeness claim is made there |
| seeding | not exposed | Tasks = (applied) only | `PLN.Query kb term` seeds every sentence as a task (52/100 pairs) |
| priority function, Unique order, stamp policy, rule set | not exposed | the fork's | the point is to use his control, unmodified |

The worker deadline of Phase 3 stays the backstop. Pre-filtering to the reachable sub-KB is mandatory (5.9 times fewer rule attempts, designer-run); K is checked against the sub-KB, not the whole KB.

---------------------------------------------------------------------------------------------------------------------------

## 4. Staging and gates

* **G0 (add to Phase 1, before any Phase 6 work).** On the vendored library under PeTTa: (a) the upstream examples `pln_roman`, `pln_tuffy`, `plntest`, `plntestdirect` give their pinned values; (b) my section 1 rows 2, 8 and 11 reproduce on the real library to 1e-12 (D+Q levels 0.13769 / 0.36029; closure 100/100 and 86/86 with Tasks = (applied) at 100/10/100 and at 20/20/200; feedback edge 109 pairs, D+Q 12 selections); (c) the MP probe and the `Truth_Revision` c = 1.0 probe; (d) the stamp `(0)` and `(State X)` terms pass the library's `Sentence` patterns; (e) wall time per `PLN.Derive` run on the patient graph and for the 10-run `rank-interventions-for-patient` workload, within the Phase 1 five-times limit. **If (a)-(d) fail**, stop and decide with the owner; **if only (e) fails**, fall back to Design 3 (legacy engine, fork for revision and cycles) and report.
* **6a (parallel, after G0).** The `pln` chainer on the shipped stacks. Gate: 186 pairs (100 patient, 86 LinAge2) equal legacy to 1e-12 in sign, strength, confidence and chain; pair set equals the closure (0 extra, 0 missing); the 12 pinned orderings of the contract; feedback edge terminates with 109 pairs and the 100 old answers unchanged; the number table in five lanes (legacy, pln-R2, pln-R1 native confidence, pln-R0 literal, pln-full), each number with its steps/tq/bq.
* **6b (parallel, after G0).** DrugAge paper-level fold. Gate: fold equals a hand-typed `Truth_Revision` over the per-paper items to 1e-12; the 8-row fixture tests unchanged; with the ETL build, Ethanol does not rank first (designer-run: rank 372 with paper ids, rank 1 with row ids; I confirmed the 52 rows / 1 paper and the 0.93 arithmetic); Resveratrol/Metformin/Rapamycin before/after table; `fold = false` equals legacy per-row output.
* **6c (after 6a and 6b).** Control demonstration (section 5). Gate: the report only; no pin.
* **Sign-off.** Unchanged: the owner and Ben before `pln` becomes the default.

---------------------------------------------------------------------------------------------------------------------------

## 5. Demonstration target for the control

Confirm the DrugAge multi-row revision as **prototype 1, relabelled**: it demonstrates `Truth_Revision` with stamp-disjoint evidence and provenance (170 Resveratrol rows into two truth values with the independence unit explicit), not the priority-queue control. It needs no base rates, so the "uncurated uniform priors" framing drops for it.

Add **prototype 2 for the control**, since the three designs agree and I confirmed that the shipped 30-node network never binds the control (one derivation per pair, 12 selections at most, 28 / 28 / 17 / 22 / 5 pairs by hops):

1. The shipped network plus the feedback edge ChronicInflammation -> CellularSenescence: termination, 109 pairs, 9 new, node-simple, complete status. Real network, real edge, but the edge is a probe, not curated.
2. A synthetic layered network with diamonds and at least 500 links, clearly labelled synthetic: legacy path count against selections used, states reached, recall of the highest-confidence states, `complete|partial` flag. My re-run: 540 links, 29,523 legacy paths, 40 selections reach 64 of 137 states with top-20 recall 1.00; 160 selections reach 79 of 137 (so partial must be reported, not hidden).
3. The real large network waits for Ben's choice (LAUNCH question 4). Candidates from the repo, none chosen by me because each needs curation rulings: CellAge/GenAge gene -> senescence links (927 CellAge rows; cellage_calibration.metta's honesty contract records a direction, not a magnitude, and the contract study found CellAge lifts are never traversed mid-chain), hallmark intervention evidence (hallmark_targeting.metta forbids turning TargetsHallmark into Effect edges).

---------------------------------------------------------------------------------------------------------------------------

## 6. Number-table lanes (Phase 6 deliverable)

| lane | mapping | confidence | purpose |
|---|---|---|---|
| legacy | hyperon `infer` | (c1 c2) 0.9 | baseline, real hyperon output (contract study) |
| pln-R2 (default) | State + MP, q = p + 0.02 | native c x 0.9^(hops-1) | the proposal; all approx pins hold |
| pln-R1 | same | native c1 c2 | what the fork says about confidence; Patient001 flips 4th/5th, diagnose c(CS) 0.6961, NMN/Fisetin margin 1.257 -> 1.071 (designer), hand-composed lever sites inconsistent with `infer` |
| pln-R0 | State + MP, q = p | native | the fork read literally; every strength shrinks 2.4 to 13.7 percent (designer) |
| pln-full | State + 17 rules, uniform STV | native | shows false chains and lost pairs; budget stated |

Optional lane, only if Ben wants syllogistic deduction and base rates: Design 2's up/down mapping with shifted strengths `p + (1-p)s`, restricted+Deduction, with `PLN_DEDUCTION_CONFIDENCE=fork|legacy` as a switch (the legacy value needs a one-hunk patch; owner decision). With the fork's minimum, 12 of 29 invariants break (designer; the Patient001 and Patient002 orders I reproduced).

---------------------------------------------------------------------------------------------------------------------------

## 7. LAUNCH.md edits (line numbers refer to the file at 8a2a6a0)

1. **Lines 10-13 (Goal).** Replace "trueagi-io/PLN (lib_pln)" with "patham9's PLN (the fork, patham9/PLN 55f1751, as ported to PeTTa in PeTTa's bundled lib/lib_pln.metta @ fea83178)"; add "The owner chose the fork on 2026-10-24."
2. **Lines 15-21 (Inputs).** Add `docs/petta_migration/fork_design/decision.md` (this file, to be copied in), the three design notes, `contract.md`, `petta_compat.md`, `fork_facts.md`. Add: "`01_verdict.md` and `13_pln_lib_facts.md` describe trueagi's newer lib_pln; its confidence rule (s1 s2 c1 c2), STV catch-all and 10/100/100 defaults do not carry over to the fork."
3. **Lines 32-34 (Authorized).** The setup script pins patham9/PLN @ `55f1751d993f71b8a24da03e3aec94ab40789a59` at `.petta/PeTTa/repos/PLN` and trueagi-io/PLN @ `4405956947c4b53c7ff01bd565aa3b114bc970a1` at `.petta/PLN-trueagi`, the latter for comparison only (never vendored, imported or executed). This changes the authorized clone list; the owner confirms by launching.
4. **Line 40.** "Making lib_pln the default chainer" -> "Making `pln` the default chainer".
5. **Lines 47-50 (Hard rules).** "lib_pln numbers get their own pins under `PLN_CHAINER=pln`" stays. Replace the last rule with: "Never execute upstream code except the pinned PeTTa runtime and the vendored fork library with its own example scripts. The vendored library differs from the bundled file by one hunk; its sha256 and the generated rule filter are asserted in a test."
6. **Phase 1 (lines 56-73).** Add step 7: "Gate G0: reproduce, on the vendored library, the upstream pln examples and the closure checks of docs/petta_migration/fork_design/decision.md section 1 (100/100 and 86/86 pairs with at most 12 selections per source, D+Q levels 0.13769 / 0.36029, feedback edge 109 pairs), the MP and Truth_Revision probes, and wall time per PLN.Derive run." Extend the Gate with: "G0 holds; if only its timing fails, stop for the owner's decision with Design 3 as the fallback."
7. **Phase 6 (lines 134-157): replace entirely** with the text in section 7a below.
8. **Lines 170-178 (Questions to send Ben now).** Replace by section 8's Ben list (question 1 is answered by the owner's choice of the fork).
9. **Lines 180-190 (Effort).** Phase 1 gains one agent-phase for G0; Phase 6 becomes 6-7 (6a 2.5, 6b 1.5, 6c plus number table and docs 2, review 1), 6a and 6b in parallel.

### 7a. Replacement text for Phase 6

```
### 6. patham9's PLN at the infer/explain seam (design done: docs/petta_migration/fork_design/decision.md; build 6a and 6b in parallel after G0, then 6c)

- Vendor PeTTa's lib/lib_pln.metta @ fea83178 (sha256 d864c3cf376ad1f9252d153306d92b7aae72a184701c2eb4805c9661eed27ece; the fork's code ported by
  its author) under vendor/pln_fork/ with PROVENANCE.md, LICENSE.NARTECH and LICENSE.PeTTa. One hunk: the per-step trace! in PLN.Derive. The restricted
  rule set (ModusPonens, NotElimination, Revision) is generated by a form-level filter and tested byte-identical to the vendored forms. No Translator, no
  PLN.Init, no build.sh. Add PLN_CHAINER=legacy|pln: legacy is the default, pln needs PLN_RUNTIME=petta, hyperon never loads it.
- Mapping ("state sentences with Modus Ponens"): node X is (State X); no node sentence, no base rate. Link = Implication (State A) (State B) for Pos,
  (Not (State B)) for Neg, (End B) for a hazard-ratio record, strength q = p + e (e = the fork's MP leak, read from Truth_ModusPonens at load, 0.02,
  capped at 1.0), confidence c, stamp = the evidence id. Intervention A = the task (State A) (stv 1.0 1.0) id 0; a second run with (stv 0.0 1.0).
  Adapter: s = |s1 - s0| at equal stamp; sign = parity of Neg links on the path rebuilt from the stamp; c = native c x 0.9^(hops-1) (switch
  PLN_CHAIN_CONFIDENCE=legacy|native); one result per route, sorted by (hops, stamp); levels never exposed.
- Evidence ids: one per independent source: curated Effect atom, hazard-ratio record, DrugAge paper (PMID). Not one per DrugAge row: the 52 ethanol
  rows come from one paper and would pool to c = 0.93.
- Revision: Truth_Revision folds the items of one compound and direction before the loop (c capped at 0.999999), stamp = union of ids. The Revision rule
  in the loop stays inert (premise id 0); it pools about 12 rows at best and always spends its whole budget.
- Control: tasks = the intervention only, beliefs = the links reachable from it. PLN.Query kb term is not used by the chainer (it seeds every sentence as a
  task: 52 of 100 pairs). Exposed: PLN_MAX_STEPS and one queue size K for both queues; schedule 256, 2048, 16384; complete iff tasks are empty and fewer
  than K-1 beliefs. Contract mode retries then fails the request; control mode returns the belief set flagged partial. Separate task and belief queue
  sizes only in the control lane.
- Rule sets: restricted is the default. PLN_RULESET=full is a comparison lane only (uniform p = 0.1, c = 0.9 on every node term). Inversion, induction and
  abduction create conclusions outside the closure (e.g. CellularSenescence to AllCauseMortality) and lose true pairs in budget.
- Write a number table in five lanes (legacy, pln default, pln with native confidence, fork read literally, full rules): every pinned (s, c), the
  orderings, the no-false-chain pairs, a feedback edge that must terminate, a mixed-sign multi-row compound, the DrugAge compounds at paper level and row
  level. Record steps, task queue and belief queue next to every number.

6a. The chainer on the shipped stacks. Gate: 186 pairs equal legacy to 1e-12 in sign, strength, confidence and chain, none extra, none missing; the
    pinned orderings hold; the feedback edge terminates (109 pairs, 100 old answers unchanged); exact goldens regenerated under pln only.
6b. DrugAge paper-level fold. Gate: equals a hand-typed Truth_Revision over the per-paper items; the 8-row fixture tests unchanged; Ethanol does not rank
    first; fold = false reproduces the legacy per-row output.
6c. Control demonstration: the shipped network with the feedback edge, and a labelled synthetic layered network of at least 500 links with diamonds;
    report selections, states reached, recall of the highest-confidence states, partial flags. The real large network waits for Ben.

Gate. Stop for the owner's and Ben's sign-off before making pln the default.
```

---------------------------------------------------------------------------------------------------------------------------

## 8. Questions only the owner or Ben can answer

**Owner (each has a default I will assume if unanswered):**

1. Confidence: keep the legacy 0.9 per combination as a shim (default) or adopt the fork's c1 c2? Native costs: Patient001 position 4/5 swap, diagnose c(CS) 0.6253 -> 0.6961 outside its tolerance, NMN/Fisetin margin 1.257 -> 1.071. The test `test_personalization_reorders_across_patients` pins the exact order; its purpose may only be "orders differ across patients".
2. DrugAge evidence unit: paper (default, my recommendation) or row (LAUNCH text). Consequences under paper ids: Resveratrol and Metformin leave exactly 0 for about -0.04 net, Rapamycin leaves rank 2 (designers: rank 3 to 5), HTTP pins 0.81 and 0.315 move when the ETL build is present. Net score or per-direction results in ranking (default: net).
3. Cycles: allow a feedback edge in curated data (default yes; it terminates and creates new pairs: 9 here) or refuse at write time? Separately: add a visited-path guard to the legacy chainer so hyperon stops hanging (changes no pinned number; Design 3's honest note says it gives the same 109 derivations).
4. Authorization change in LAUNCH.md line 33 (clone patham9/PLN @ 55f1751; trueagi PLN comparison-only). Already checked out in `.petta/`.
5. Wall-time budget if the 5x limit fails at G0: accept the hybrid fallback, or wait for a faster path?
6. Edges out of a hazard-ratio record target (e.g. CHD -> Stroke): the chosen encoding keeps records terminal; confirm that legacy's rule stays.
7. Counts differ between notes (1,027 / 1,035 / 1,038 / 1,046 compounds: synonym resolution and scorable filters): which count is the contract for the whole-build ranking test?

**Ben (or patham9):**

1. Priority: the fork ranks tasks by confidence only. For a longevity graph the useful priority is effect times confidence. Is confidence-ranking the control he means, or should the application supply its own priority function?
2. Is the per-combination 0.9 confidence discount a PLN concept, or is MP's c1 c2 (and Deduction's minimum) the intended behaviour? Is MP's fixed 0.02 (Formulas.metta:47) the intended default for P(B | not A)? The comment at line 48 suggests it is a stand-in for missing node probabilities.
3. Which large network should the control be shown on? Is a labelled synthetic layered benchmark (and the cycle probe) acceptable until it exists?
4. Is revision across papers (independent evidence) the intended use of `Truth_Revision` for DrugAge, with rows of one paper collapsed to one item?
5. Induction and Abduction call sites (Rules.metta:39-52) pass the three node probabilities with the shared node in the third slot, while Truth_Induction expects it in the second (Formulas.metta:23-31). Intentional? Only matters if the full lane is ever used with non-uniform priors.
6. Inversion copies strength (Formulas.metta:80-83), correct only when both nodes have equal probability: confirms uniform priors are the only safe start for the full lane.

---------------------------------------------------------------------------------------------------------------------------

## 9. What to tell Ben, plainly

Under `PLN_CHAINER=pln` every `infer` on the app's network runs through your Modus Ponens, Negation-elimination and Revision formulas, your stamp rule and your confidence-ranked PLN.Derive queue, and its answers equal the legacy ones because the mapping is built to make them equal (two named constants: the fork's 0.02 leak, which a two-run difference cancels, and the legacy 0.9 hop discount). On this 30-node network with one derivation per pair the control never binds. What it shows today: termination on a feedback loop, evidence pooling with provenance for DrugAge, and a complete/partial flag. The control itself is shown on a cycle probe and on a labelled synthetic network until you tell us which large network to use.

---------------------------------------------------------------------------------------------------------------------------

## 10. Residual risks

1. Nothing has run on PeTTa. G0 exists for that. The three designs' simulator-based claims were reproduced independently here, which tests the design logic, not the runtime.
2. The adapter carries real logic (pre-filter, two runs, stamp join, path rebuild, shim, fold). If a reviewer argues "that is the legacy chainer with a fork-shaped engine", the answer is the list in section 9; the engine is not decorative, but the numbers are mapping-defined (finding 1).
3. Cost is quadratic in reachable links (no antecedent index) and doubles with the second run. A 510-link tree is 1.55 M attempts per run (designer). Routes are exponential on dense graphs; the K schedule and the deadline are the backstops, and the control lane is where that matters.
4. Revision drives confidence toward 1 and assumes independence between papers; the unit is a curation call.
5. State levels are artefacts. No consumer may read one.
6. Library drift: the trueagi rewrite changes MP confidence to (s1 s2)(c1 c2); the load-time probe catches it.
7. Float noise: strengths differ from legacy in the last digit (max 1.7e-16) so exact-string goldens are regenerated under `pln`.
8. My full-rule reproduction (10 extra pairs) is smaller than the designers' counts; the finding is qualitative, and the lane must state its budget and encoding.

Files: `decision/checks/{common,ck1_arith,ck2_closure,ck2b_unique,ck3_revision_loop,ck4_bestfirst,ck5_orders,ck6_fullrules,ck7_control_seam,ck8_diagnose}.py`; designer notes under `design/{mp_state,d2,design3}/design.md`; facts `facts/fork_facts.md`; contract `contract/contract.md`; compat `compat/petta_compat.md`.
