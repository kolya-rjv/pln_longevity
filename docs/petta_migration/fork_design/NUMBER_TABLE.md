# Number table: the same questions in six lanes

Phase 6, deliverable of gate 6a/6b. Work tree `/Users/kolya/PycharmProjects/pln_longevity_petta` (branch `petta-migration`), measured 2026-10-24 on this Mac
(Python 3.12.10, hyperon 0.2.10, SWI-Prolog 10.0.2, PeTTa `fea83178`, the vendored library of `vendor/pln_fork/`).

**Nothing in the tables below was typed or copied.** Every table is a block that `scripts/pln_number_table.py render` writes from the lane files in
`docs/petta_migration/fork_design/c/results/lane_*.json`; each lane file was written by one command, `python scripts/pln_number_table.py lane --lane <id>`, in a
fresh process, through the production path (`core.pln_runner._runtime_run`, the call the API makes). The block header repeats the command. The only literals are
the questions (which chains, which pairs), the pins as the arithmetic expressions the tests write (shown in their own column and marked as a specification, not
a measurement), and the definitions of the twelve orderings. `tests/test_pln_number_table.py` re-renders every block from the stored data and fails if this
file disagrees with it. The prose between the blocks was written by me from the blocks; if prose and block ever disagree, the block is right.

Regenerate everything: `python scripts/pln_number_table.py all` (about 5 minutes), then `python scripts/pln_number_table.py render`.

## 0. What the table says, in seven lines

1. **pln-R2, the default of `PLN_CHAINER=pln`, answers every pinned question the way the legacy chainer does.** All 100 patient-stack pairs, no pair missing or extra,
   strength within 2.2e-16 and confidence within 2.8e-17 of the legacy value, all twelve orderings hold, every diagnosis pin holds. 39 of 100 strength strings and 25 of
   100 confidence strings differ from legacy in the last printed digit, which is why exact-string goldens are regenerated under `pln` and nowhere else.
2. **The two legacy lanes are the same chainer**: hyperon and PeTTa print identical strings for all 100 pairs (0 of 100 differ from the recording of the contract study, in either),
   give the same values in all 12 orderings, and print 39 of the 50 composite outputs identically; the other 11 differ only in `3.0` against `3` and in the order of a printed set (checked:
   the same tokens once those are normalised). The legacy lane is therefore not a moving reference.
3. **pln-R1 (the library's own confidence, c1 c2, no 0.9 per combination) changes no strength but moves confidences by up to a factor 1.524 (five links).** Two pinned
   things break: Patient001's personalised order (Spermidine 0.1041 passes Berberine 0.0948) and the diagnosis confidence c(CS) (0.6961 against 0.6253 +- 0.01). 11 of 12
   orderings hold.
4. **pln-R0 (the fork read literally, q = p) shrinks every strength**: by 2.4 to 5.1 percent at one link and by 12.4 to 13.7 percent at five (ratios 0.9487 to 0.9765 and 0.8627
   to 0.8762); s(CS) is 0.5644 where the test pins 0.595. It fails the same two things as R1, and s(CS) besides. 100 of 100 strength strings change.
5. **pln-full (all 17 rules) finds 80 of the 100 true pairs and loses 20** (Resveratrol 9 of 9, Spermidine 9 of 10, Fisetin 2 of 9) because it spends its fixed budget of
   150 steps on other conclusions, and **its belief set holds 30 conclusions that no walk of the graph licenses**, among them `MitochondrialDysfunction -> CoronaryHeartDisease`
   (a pair the contract says must stay empty) and `CellularSenescence -> AgeAccelGrim` and `DNAmPAI1 -> AgeAccelGrim` (nothing in the graph reaches AgeAccelGrim). The adapter never
   shows them to a consumer (it answers only routes whose stamp rebuilds a walk), so `infer` returns the empty answer for those pairs and *lost* answers elsewhere. Only 5 of 12 orderings hold.
6. **The feedback edge**: both legacy chainers never end (killed at the deadline); pln R2, R1 and R0 end with 109 pairs, equal to the node-simple reference, the
   100 old answers byte-identical, 9 new pairs, at most 16 selections for any source. pln-full ends too, with 78 of 109 pairs and only 57 of its 80 old answers unchanged.
7. **DrugAge at paper level, one NET formula, and the sign of two compounds is the null convention's** (section 8). A reported null (a change of exactly 0.0) counts on both sides by default:
   Resveratrol +0.1363 and Metformin +0.1182 (both exactly 0 for the representative row), Rapamycin +0.3368 (was +0.3191), Ethanol +0.1144 (was +0.1432). The first build filed a null on the
   protective side only, where the sign convention puts a zero, and got Resveratrol -0.0460 and Metformin -0.0570 (harmful); dropping the nulls gives +0.0321 and +0.0629. Both symmetric conventions
   say protective; the one-sided one is the only route to "harmful", so the earlier sentence "leave 0 for a small harmful net" described the convention, not the evidence. With every row its own unit
   Ethanol becomes +0.5679, the largest of the four, which is the control that shows why the paper is the unit. The mixed-sign compound nets to +0.1757 (nulls on both sides), -0.0452 (nulls on the
   protective side only, the hand-typed figure of the design notes) and +0.1423 (dropped). The engine and the pure-Python twin agree to better than 2e-16 in all 20 cells.

## 1. Lanes and how to read a cell

| lane | what it runs |
|---|---|
| legacy (hyperon) | `PLN_RUNTIME=hyperon PLN_CHAINER=legacy`: the MeTTa chainer of `pln_deduction.metta`, product of strengths, `(c1 c2) 0.9` per combination |
| legacy (PeTTa) | the same equations on PeTTa |
| pln-R2 | `PLN_CHAINER=pln`: links stored as q = 0.02 + 0.98 p, `Truth_ModusPonens` + `Truth_Negation` through `PLN.Derive`, strength = abs(s1 - s0) / 0.98^hops from two runs, confidence = the library's c x 0.9^(hops - 1) |
| pln-R1 | as R2 with the library's c as it stands (`PLN_CHAIN_CONFIDENCE=native`) |
| pln-R0 | the fork read literally: q = p, strength abs(s1 - s0), native confidence (`ChainSettings(mapping="literal")`) |
| pln-full | `PLN_RULESET=full`: all 17 rules of the library, every node probability (0.1, 0.9), every sentence a task, the whole graph as beliefs, a fixed budget of 150 steps / task queue 100 / belief queue 400, native confidence |

A cell reads `Sign strength / confidence` and, under it, the cost of the derivation that produced it: `st / tq / bq` = selections of the level-1 run of the source (the tasks
`PLN.Derive` expanded) / the task-queue size / the belief-queue size. The legacy chainer is a recursion and has no queues: its cells say `no queues`. For pln-full the
step count is the budget (`st 150 (budget)`), because every run of that lane spends all of it (99 tasks are still waiting at the end, the belief queue is full at 399).
Where a table summarises a whole program (the twelve orderings, the closure) the cost is the worst source of the program.

<!-- BEGIN GENERATED environment -->
Measured, one command per lane; script sha256 d64db6cb03be437e.

| lane | command that measured every cell of the lane | wall s | script sha256 (16) | engine |
|---|---|---|---|---|
| legacy (hyperon) | `python scripts/pln_number_table.py lane --lane legacy-hyperon` | 52.1 | d64db6cb03be437e | 0.2.10 |
| legacy (PeTTa) | `python scripts/pln_number_table.py lane --lane legacy-petta` | 13.2 | d64db6cb03be437e | fea83178 |
| pln-R2 | `python scripts/pln_number_table.py lane --lane R2` | 8.8 | d64db6cb03be437e | fea83178 |
| pln-R1 | `python scripts/pln_number_table.py lane --lane R1` | 8.7 | d64db6cb03be437e | fea83178 |
| pln-R0 | `python scripts/pln_number_table.py lane --lane R0` | 8.7 | d64db6cb03be437e | fea83178 |
| pln-full | `python scripts/pln_number_table.py lane --lane full` | 205.0 | d64db6cb03be437e | fea83178 |
<!-- END GENERATED environment -->

## 2. The five smoke chains, and four more

The contract has no list called "smoke chains"; I took the five chain pins of `tests/test_pln_deduction.py` (contract.md 4.1): SASP, CellularSenescence, D+Q, Fisetin and
Spermidine to CoronaryHeartDisease (marked `*`), and added Metformin and Berberine (the metabolic axis), the hazard-ratio record DNAmPAI1 -> CHD, and the empty answer
D+Q -> AllCauseMortality. The pin column is the test's own arithmetic, evaluated; it is a specification.

<!-- BEGIN GENERATED chains -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| chain (* = the five smoke chains) | pin (the test's arithmetic) | legacy (hyperon) | legacy (PeTTa) | pln-R2 | pln-R1 | pln-R0 | pln-full |
|---|---|---|---|---|---|---|---|
| SASP->CHD * | Pos 0.42 / 0.49725<br><sub>test_one_hop_propagation</sub> | Pos 0.42 / 0.49725<br><sub>no queues</sub> | Pos 0.42 / 0.49725<br><sub>no queues</sub> | Pos 0.42 / 0.49725<br><sub>st 8 / tq 256 / bq 256</sub> | Pos 0.42 / 0.5525<br><sub>st 8 / tq 256 / bq 256</sub> | Pos 0.3944 / 0.5525<br><sub>st 8 / tq 256 / bq 256</sub> | Pos 0.432245 / 0.5525<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| CS->CHD * | Pos 0.357 / 0.290891<br><sub>test_novel_cross_source_inference_harmful</sub> | Pos 0.357 / 0.290891<br><sub>no queues</sub> | Pos 0.357 / 0.290891<br><sub>no queues</sub> | Pos 0.357 / 0.290891<br><sub>st 9 / tq 256 / bq 256</sub> | Pos 0.357 / 0.359125<br><sub>st 9 / tq 256 / bq 256</sub> | Pos 0.327352 / 0.359125<br><sub>st 9 / tq 256 / bq 256</sub> | Pos 0.36144 / 0.4225<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| D+Q->CHD * | Neg 0.2499 / 0.170171<br><sub>test_protective_intervention_flips_sign</sub> | Neg 0.2499 / 0.170171<br><sub>no queues</sub> | Neg 0.2499 / 0.170171<br><sub>no queues</sub> | Neg 0.2499 / 0.170171<br><sub>st 11 / tq 256 / bq 256</sub> | Neg 0.2499 / 0.233431<br><sub>st 11 / tq 256 / bq 256</sub> | Neg 0.222599 / 0.233431<br><sub>st 11 / tq 256 / bq 256</sub> | Neg 0.262765 / 0.4225<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Fisetin->CHD * | Neg 0.23205 / 0.130901<br><sub>test_fisetin_protective_weaker_than_dq</sub> | Neg 0.23205 / 0.130901<br><sub>no queues</sub> | Neg 0.23205 / 0.130901<br><sub>no queues</sub> | Neg 0.23205 / 0.130901<br><sub>st 11 / tq 256 / bq 256</sub> | Neg 0.23205 / 0.179563<br><sub>st 11 / tq 256 / bq 256</sub> | Neg 0.206232 / 0.179563<br><sub>st 11 / tq 256 / bq 256</sub> | Neg 0.252892 / 0.325<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Spermidine->CHD * | Neg 0.17136 / 0.0765771<br><sub>test_spermidine_longer_chain_lower_confidence</sub> | Neg 0.17136 / 0.0765771<br><sub>no queues</sub> | Neg 0.17136 / 0.0765771<br><sub>no queues</sub> | Neg 0.17136 / 0.0765771<br><sub>st 12 / tq 256 / bq 256</sub> | Neg 0.17136 / 0.116716<br><sub>st 12 / tq 256 / bq 256</sub> | Neg 0.148094 / 0.116716<br><sub>st 12 / tq 256 / bq 256</sub> | empty<br>st 150 (budget) / tq 100 / bq 400 |
| Metformin->CHD | Neg 0.33 / 0.268515<br><sub>test_patient_grounding::test_metabolic_axis_reaches_chd</sub> | Neg 0.33 / 0.268515<br><sub>no queues</sub> | Neg 0.33 / 0.268515<br><sub>no queues</sub> | Neg 0.33 / 0.268515<br><sub>st 7 / tq 256 / bq 256</sub> | Neg 0.33 / 0.3315<br><sub>st 7 / tq 256 / bq 256</sub> | Neg 0.301782 / 0.3315<br><sub>st 7 / tq 256 / bq 256</sub> | Neg 0.335689 / 0.3315<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Berberine->CHD | Neg 0.286 / 0.268515<br><sub>same mechanism as Metformin (rank_population_5pool)</sub> | Neg 0.286 / 0.268515<br><sub>no queues</sub> | Neg 0.286 / 0.268515<br><sub>no queues</sub> | Neg 0.286 / 0.268515<br><sub>st 7 / tq 256 / bq 256</sub> | Neg 0.286 / 0.3315<br><sub>st 7 / tq 256 / bq 256</sub> | Neg 0.260442 / 0.3315<br><sub>st 7 / tq 256 / bq 256</sub> | Neg 0.291689 / 0.3315<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| DNAmPAI1->CHD (record) | Pos 0.6 / 0.85<br><sub>test_direct_link_is_calibrated_and_signed</sub> | Pos 0.6 / 0.85<br><sub>no queues</sub> | Pos 0.6 / 0.85<br><sub>no queues</sub> | Pos 0.6 / 0.85<br><sub>st 2 / tq 256 / bq 256</sub> | Pos 0.6 / 0.85<br><sub>st 2 / tq 256 / bq 256</sub> | Pos 0.58 / 0.85<br><sub>st 2 / tq 256 / bq 256</sub> | Pos 0.6 / 0.85<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| D+Q->AllCauseMortality | empty | empty | empty | empty<br>st 11 / tq 256 / bq 256 | empty<br>st 11 / tq 256 / bq 256 | empty<br>st 11 / tq 256 / bq 256 | empty<br>st 150 (budget) / tq 100 / bq 400 |
<!-- END GENERATED chains -->

<!-- BEGIN GENERATED chain_deviation -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| lane | max abs strength difference to legacy (hyperon), nine chains | max abs confidence difference to legacy (hyperon) | sign or emptiness differs |
|---|---|---|---|
| legacy (hyperon) | 0.00e+00 | 0.00e+00 | 0 |
| legacy (PeTTa) | 0.00e+00 | 0.00e+00 | 0 |
| pln-R2 | 5.55e-17 | 2.78e-17 | 0 |
| pln-R1 | 5.55e-17 | 6.82e-02 | 0 |
| pln-R0 | 2.96e-02 | 6.82e-02 | 0 |
| pln-full | 2.08e-02 | 2.52e-01 | 1 |
<!-- END GENERATED chain_deviation -->

Reading. R2 is the legacy number to the last digit that floats allow (the 0.9 shim is `c x 0.9^(hops - 1)`; the strength is a difference of two levels divided by 0.98^hops).
R1 prints R2's strengths with the library's confidence (D+Q: 0.2334 = 0.65^3 x 0.85, the product of the link confidences). R0 prints the same confidences as R1 and lower strengths on
every chain that has a link (D+Q 0.2226 against 0.2499). In pln-full the numbers of the chains it does answer differ from every other lane in both columns
(CS -> CHD 0.3614 / 0.4225 against 0.357 / 0.2909) and Spermidine -> CHD is not answered at all: the lane reports the first conclusion that carries a route's stamp, and with
17 rules one stamp can carry several conclusions, so these numbers depend on rules and budget and cannot be pinned.

## 3. The twelve orderings

The twelve are the orderings and relations of contract.md 4.2 and 5.4 that the 50 composite queries of the contract study (`contract/pinned_numbers.json`) contain; the
definitions are in `evaluate_orderings` of the script (the same twelve `tests/test_pln_chainer.py::test_the_twelve_pinned_orderings_hold` asserts, written independently).
Each lane ran all 50 queries once, in one program, on the patient stack.

<!-- BEGIN GENERATED orderings -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| # | ordering (contract.md 4.2 / 5.4) | legacy (hyperon) | legacy (PeTTa) | pln-R2 | pln-R1 | pln-R0 | pln-full |
|---|---|---|---|---|---|---|---|
| 1 | population ranking [D+Q, Fisetin, Spermidine], Elamipretide absent | holds<br><sub>D+Q > Fisetin > Spermidine</sub> | holds<br><sub>D+Q > Fisetin > Spermidine</sub> | holds<br><sub>D+Q > Fisetin > Spermidine</sub> | holds<br><sub>D+Q > Fisetin > Spermidine</sub> | holds<br><sub>D+Q > Fisetin > Spermidine</sub> | **FAILS**<br><sub>D+Q > Fisetin</sub> |
| 2 | 5-pool population ranking [Metformin, Berberine, D+Q, Fisetin, Spermidine] | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | **FAILS**<br><sub>Metformin > D+Q > Berberine > Fisetin</sub> |
| 3 | Patient001 personalised order [D+Q, Fisetin, Metformin, Berberine, Spermidine] | holds<br><sub>D+Q > Fisetin > Metformin > Berberine > Spermidine</sub> | holds<br><sub>D+Q > Fisetin > Metformin > Berberine > Spermidine</sub> | holds<br><sub>D+Q > Fisetin > Metformin > Berberine > Spermidine</sub> | **FAILS**<br><sub>D+Q > Fisetin > Metformin > Spermidine > Berberine</sub> | **FAILS**<br><sub>D+Q > Fisetin > Metformin > Spermidine > Berberine</sub> | **FAILS**<br><sub>D+Q > Fisetin > Metformin > Berberine</sub> |
| 4 | Patient002 personalised order [Metformin, Berberine, D+Q, Fisetin, Spermidine] | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | holds<br><sub>Metformin > Berberine > D+Q > Fisetin > Spermidine</sub> | **FAILS**<br><sub>Metformin > Berberine > D+Q > Fisetin</sub> |
| 5 | relevance to Patient001's markers: D+Q > Fisetin > Spermidine > 0 = Elamipretide | holds<br><sub>D+Q 0.1881, Fisetin 0.1343, Spermidine 0.0580, Elamipretide 0</sub> | holds<br><sub>D+Q 0.1881, Fisetin 0.1343, Spermidine 0.0580, Elamipretide 0</sub> | holds<br><sub>D+Q 0.1881, Fisetin 0.1343, Spermidine 0.0580, Elamipretide 0</sub> | holds<br><sub>D+Q 0.2453, Fisetin 0.1752, Spermidine 0.0841, Elamipretide 0</sub> | holds<br><sub>D+Q 0.2233, Fisetin 0.1591, Spermidine 0.0743, Elamipretide 0</sub> | **FAILS**<br><sub>D+Q 0.5084, Fisetin 0.3632, Spermidine 0.0000, Elamipretide 0</sub> |
| 6 | relevance is patient specific: Patient002 Metformin and Berberine > 0.4, Patient001 Metformin = 0 | holds<br><sub>P002 Metformin 0.5594, P002 Berberine 0.4848, P001 Metformin 0</sub> | holds<br><sub>P002 Metformin 0.5594, P002 Berberine 0.4848, P001 Metformin 0</sub> | holds<br><sub>P002 Metformin 0.5594, P002 Berberine 0.4848, P001 Metformin 0</sub> | holds<br><sub>P002 Metformin 0.6216, P002 Berberine 0.5387, P001 Metformin 0</sub> | holds<br><sub>P002 Metformin 0.5889, P002 Berberine 0.5082, P001 Metformin 0</sub> | holds<br><sub>P002 Metformin 0.6216, P002 Berberine 0.5387, P001 Metformin 0</sub> |
| 7 | diagnosis ranks by coverage: [CS, CI, MD] with coverage [3, 2, 1] (population, Patient001, default causes) | holds<br><sub>Ce/Ch/Mi 321; Ce/Ch/Mi 321; Ce/Ch/Mi 321</sub> | holds<br><sub>Ce/Ch/Mi 321; Ce/Ch/Mi 321; Ce/Ch/Mi 321</sub> | holds<br><sub>Ce/Ch/Mi 321; Ce/Ch/Mi 321; Ce/Ch/Mi 321</sub> | holds<br><sub>Ce/Ch/Mi 321; Ce/Ch/Mi 321; Ce/Ch/Mi 321</sub> | holds<br><sub>Ce/Ch/Mi 321; Ce/Ch/Mi 321; Ce/Ch/Mi 321</sub> | holds<br><sub>Ce/Ch/Mi 321; Ce/Ch/Mi 321; Ce/Ch/Mi 321</sub> |
| 8 | counterfactual: senescence moves the clock more than inflammation and is less certain (c(B) < c(A) <= 0.65) | holds<br><sub>\|dB\| 0.2219 > \|dA\| 0.1219; c(B) 0.3176 < c(A) 0.6500 <= 0.65</sub> | holds<br><sub>\|dB\| 0.2219 > \|dA\| 0.1219; c(B) 0.3176 < c(A) 0.6500 <= 0.65</sub> | holds<br><sub>\|dB\| 0.2219 > \|dA\| 0.1219; c(B) 0.3176 < c(A) 0.6500 <= 0.65</sub> | holds<br><sub>\|dB\| 0.2219 > \|dA\| 0.1219; c(B) 0.3638 < c(A) 0.6500 <= 0.65</sub> | holds<br><sub>\|dB\| 0.2084 > \|dA\| 0.1181; c(B) 0.3647 < c(A) 0.6500 <= 0.65</sub> | holds<br><sub>\|dB\| 0.2273 > \|dA\| 0.1219; c(B) 0.4225 < c(A) 0.6500 <= 0.65</sub> |
| 9 | senolytic lever = 0.70 of senescence's effect, lowest confidence | holds<br><sub>dQ/dB 0.700000; c(D+Q) 0.1858 < c(B) 0.3176</sub> | holds<br><sub>dQ/dB 0.700000; c(D+Q) 0.1858 < c(B) 0.3176</sub> | holds<br><sub>dQ/dB 0.700000; c(D+Q) 0.1858 < c(B) 0.3176</sub> | holds<br><sub>dQ/dB 0.700000; c(D+Q) 0.2128 < c(B) 0.3638</sub> | holds<br><sub>dQ/dB 0.700000; c(D+Q) 0.2134 < c(B) 0.3647</sub> | holds<br><sub>dQ/dB 0.700000; c(D+Q) 0.2472 < c(B) 0.4225</sub> |
| 10 | supplements for Patient001: Omega3 in tier 1, NMN scores above Fisetin | holds<br><sub>Omega3 tier 1: True; NMN 0.1689 vs Fisetin 0.1343</sub> | holds<br><sub>Omega3 tier 1: True; NMN 0.1689 vs Fisetin 0.1343</sub> | holds<br><sub>Omega3 tier 1: True; NMN 0.1689 vs Fisetin 0.1343</sub> | holds<br><sub>Omega3 tier 1: True; NMN 0.1877 vs Fisetin 0.1752</sub> | holds<br><sub>Omega3 tier 1: True; NMN 0.1760 vs Fisetin 0.1591</sub> | **FAILS**<br><sub>Omega3 tier 1: True; NMN 0.1877 vs Fisetin 0.3632</sub> |
| 11 | risk: reduction(senescence) > reduction(inflammation) > reduction(insulin resistance) = 0, confidence(senescence) < confidence(inflammation) | holds<br><sub>reductions 0.00770 > 0.00429 > 0; confidence 0.3176 < 0.6500</sub> | holds<br><sub>reductions 0.00770 > 0.00429 > 0; confidence 0.3176 < 0.6500</sub> | holds<br><sub>reductions 0.00770 > 0.00429 > 0; confidence 0.3176 < 0.6500</sub> | holds<br><sub>reductions 0.00770 > 0.00429 > 0; confidence 0.3638 < 0.6500</sub> | holds<br><sub>reductions 0.00725 > 0.00416 > 0; confidence 0.3647 < 0.6500</sub> | holds<br><sub>reductions 0.00789 > 0.00429 > 0; confidence 0.4225 < 0.6500</sub> |
| 12 | confidence by chain length: D+Q (4 edges) > Fisetin (4) > Spermidine (5), all below a single link's 0.65 | holds<br><sub>c: D+Q 0.1702 > Fisetin 0.1309 > Spermidine 0.0766, all < 0.65</sub> | holds<br><sub>c: D+Q 0.1702 > Fisetin 0.1309 > Spermidine 0.0766, all < 0.65</sub> | holds<br><sub>c: D+Q 0.1702 > Fisetin 0.1309 > Spermidine 0.0766, all < 0.65</sub> | holds<br><sub>c: D+Q 0.2334 > Fisetin 0.1796 > Spermidine 0.1167, all < 0.65</sub> | holds<br><sub>c: D+Q 0.2334 > Fisetin 0.1796 > Spermidine 0.1167, all < 0.65</sub> | **FAILS**<br><sub>only 2 of the 3 chains answered (DasatinibPlusQuercetin, Fisetin)</sub> |
| cost of the program |  | no queues | no queues | max st 12 / tq 256 / bq 256; max beliefs 22; 40 runs, 0.4919 s | max st 12 / tq 256 / bq 256; max beliefs 22; 40 runs, 0.4855 s | max st 12 / tq 256 / bq 256; max beliefs 22; 40 runs, 0.4802 s | st 150 (budget) / tq 100 / bq 400; 40 runs, 75.1516 s; routes complete: False |
| orderings that hold |  | 12 of 12 | 12 of 12 | 12 of 12 | 11 of 12 | 11 of 12 | 5 of 12 |
<!-- END GENERATED orderings -->

The diagnosis pins that have their own tolerances:

<!-- BEGIN GENERATED diagnose -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| lane | s(CS), pin 0.595 (rel 1e-6) | c(CS), population | c(MD), pin 0.65 (rel 1e-6) | mass(CS) | c(CS), Patient001, pin 0.6253 (abs 1e-2) | cost of the program |
|---|---|---|---|---|---|---|
| legacy (hyperon) | 0.595 | 0.625305 | 0.65 | 0.459289 | 0.625305 | no queues |
| legacy (PeTTa) | 0.595 | 0.625305 | 0.65 | 0.459289 | 0.625305 | no queues |
| pln-R2 | 0.595 | 0.625305 | 0.65 | 0.459289 | 0.625305 | max st 12 / tq 256 / bq 256 |
| pln-R1 | 0.595 | 0.696137 | 0.65 | 0.539092 | 0.696137 **FAILS** | max st 12 / tq 256 / bq 256 |
| pln-R0 | 0.5644 **FAILS** | 0.696137 | 0.65 | 0.505216 | 0.696137 **FAILS** | max st 12 / tq 256 / bq 256 |
| pln-full | 0.609286 **FAILS** | 0.8074 | 0.65 | 0.709916 | 0.8074 **FAILS** | st 150 (budget) / tq 100 / bq 400 |
<!-- END GENERATED diagnose -->

Reading. R1 and R0 fail exactly the same two things, both because of confidence: ordering 3 (native confidence rewards the five-link Spermidine chain, which touches three of
Patient001's elevated markers, more than the four-link Berberine chain) and the diagnosis confidence. Orderings 5, 6, 10 (relevance, NMN against Fisetin) still hold in R1 and R0
with different numbers (the relevance of D+Q to Patient001 is 0.2453 in R1 and 0.2233 in R0 against 0.1881). In pln-full seven orderings fail: Spermidine has lost nine of its ten routes,
so it drops out of the rankings (1 to 4, 12) and its relevance is 0 (5); NMN no longer outranks Fisetin (10) because the lane's Fisetin relevance is inflated (0.3632 against 0.1877).

## 4. Patient001 and Patient002 personalised orders

<!-- BEGIN GENERATED personalised -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| patient | lane | personalised order with scores (rank-score + relevance) | signs | cost of the program |
|---|---|---|---|---|
| Patient001 | legacy (hyperon) | D+Q 0.2306 > Fisetin 0.1647 > Metformin 0.0886 > Berberine 0.0768 > Spermidine 0.0712 | all Neg | no queues |
| Patient001 | legacy (PeTTa) | D+Q 0.2306 > Fisetin 0.1647 > Metformin 0.0886 > Berberine 0.0768 > Spermidine 0.0712 | all Neg | no queues |
| Patient001 | pln-R2 | D+Q 0.2306 > Fisetin 0.1647 > Metformin 0.0886 > Berberine 0.0768 > Spermidine 0.0712 | all Neg | max st 12 / tq 256 / bq 256 |
| Patient001 | pln-R1 | D+Q 0.3036 > Fisetin 0.2169 > Metformin 0.1094 > Spermidine 0.1041 > Berberine 0.0948 | all Neg | max st 12 / tq 256 / bq 256 |
| Patient001 | pln-R0 | D+Q 0.2753 > Fisetin 0.1962 > Metformin 0.1000 > Spermidine 0.0916 > Berberine 0.0863 | all Neg | max st 12 / tq 256 / bq 256 |
| Patient001 | pln-full | D+Q 0.6194 > Fisetin 0.4453 > Metformin 0.1113 > Berberine 0.0967 | all Neg | st 150 (budget) / tq 100 / bq 400 |
| Patient002 | legacy (hyperon) | Metformin 0.6480 > Berberine 0.5616 > D+Q 0.0425 > Fisetin 0.0304 > Spermidine 0.0131 | all Neg | no queues |
| Patient002 | legacy (PeTTa) | Metformin 0.6480 > Berberine 0.5616 > D+Q 0.0425 > Fisetin 0.0304 > Spermidine 0.0131 | all Neg | no queues |
| Patient002 | pln-R2 | Metformin 0.6480 > Berberine 0.5616 > D+Q 0.0425 > Fisetin 0.0304 > Spermidine 0.0131 | all Neg | max st 12 / tq 256 / bq 256 |
| Patient002 | pln-R1 | Metformin 0.7310 > Berberine 0.6335 > D+Q 0.0583 > Fisetin 0.0417 > Spermidine 0.0200 | all Neg | max st 12 / tq 256 / bq 256 |
| Patient002 | pln-R0 | Metformin 0.6889 > Berberine 0.5945 > D+Q 0.0520 > Fisetin 0.0370 > Spermidine 0.0173 | all Neg | max st 12 / tq 256 / bq 256 |
| Patient002 | pln-full | Metformin 0.7328 > Berberine 0.6354 > D+Q 0.1110 > Fisetin 0.0822 | all Neg | st 150 (budget) / tq 100 / bq 400 |
<!-- END GENERATED personalised -->

Patient002's order never moves in any lane that answers all five chains. Patient001's moves in R1 and R0 only, and only in the last two places.

## 5. All 100 pairs of the patient stack

The reference is `contract/closure.json`, the recording of the real hyperon output. The legacy (hyperon) row is this run's own measurement and equals it.

<!-- BEGIN GENERATED closure -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| lane | pairs | missing | extra | pairs with 2 results | sign differs | max abs dev strength | max abs dev confidence | strength ratio to recorded (min .. max) | confidence ratio (min .. max) | strings that differ (strength / confidence) | cost (worst source) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy (hyperon) | 100 of 100 | 0 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 0 / 0 | no queues |
| legacy (PeTTa) | 100 of 100 | 0 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 0 / 0 | no queues |
| pln-R2 | 100 of 100 | 0 | 0 | 0 | 0 | 2.22e-16 | 2.78e-17 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 39 / 25 | max st 12 / tq 256 / bq 256 |
| pln-R1 | 100 of 100 | 0 | 0 | 0 | 0 | 2.22e-16 | 6.82e-02 | 1.0000 .. 1.0000 | 1.0000 .. 1.5242 | 39 / 72 | max st 12 / tq 256 / bq 256 |
| pln-R0 | 100 of 100 | 0 | 0 | 0 | 0 | 4.07e-02 | 6.82e-02 | 0.8627 .. 0.9765 | 1.0000 .. 1.5242 | 100 / 72 | max st 12 / tq 256 / bq 256 |
| pln-full | 80 of 100 | 20 | 0 | 0 | 0 | 2.32e-02 | 2.92e-01 | 1.0000 .. 1.0898 | 1.0000 .. 3.2467 | 45 / 53 | st 150 (budget) / tq 100 / bq 400 |
<!-- END GENERATED closure -->

Strength ratio to the recorded value, by number of links:

<!-- BEGIN GENERATED closure_hops -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| lane | 1 hop | 2 hops | 3 hops | 4 hops | 5 hops |
|---|---|---|---|---|---|
| legacy (hyperon) | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 |
| legacy (PeTTa) | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 |
| pln-R2 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 |
| pln-R1 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 | 1.0000 .. 1.0000 |
| pln-R0 | 0.9487 .. 0.9765 | 0.9250 .. 0.9535 | 0.9106 .. 0.9297 | 0.8787 .. 0.9031 | 0.8627 .. 0.8762 |
| pln-full | 1.0000 .. 1.0000 | 1.0000 .. 1.0292 | 1.0124 .. 1.0655 | 1.0451 .. 1.0898 | - |
<!-- END GENERATED closure_hops -->

## 6. No false chain

The pairs the contract says must return nothing. A cell that is not `empty` would be a false chain. For pln-full the second line says what the engine's own belief set holds
(from `full_lane_harness`, which reads the level-1 beliefs of every source): the adapter exposes only conclusions whose stamp is a walk of the graph, so the empty answer
and the conclusion can both be true.

<!-- BEGIN GENERATED nofalse -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| pair (must be empty) | why | legacy (hyperon) | legacy (PeTTa) | pln-R2 | pln-R1 | pln-R0 | pln-full |
|---|---|---|---|---|---|---|---|
| DasatinibPlusQuercetin -> AllCauseMortality | test_no_false_chain | empty | empty | empty<br><sub>st 11 / tq 256 / bq 256</sub> | empty<br><sub>st 11 / tq 256 / bq 256</sub> | empty<br><sub>st 11 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set holds no such conclusion</sub> |
| SmokingCessation -> CoronaryHeartDisease | test_the_smoking_axis_still_does_not_reach_chd_through_infer | empty | empty | empty<br><sub>st 4 / tq 256 / bq 256</sub> | empty<br><sub>st 4 / tq 256 / bq 256</sub> | empty<br><sub>st 4 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set holds no such conclusion</sub> |
| Rapamycin -> CoronaryHeartDisease | test_rapamycin_is_still_unreachable_by_inference | empty | empty | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> |
| Elamipretide -> CoronaryHeartDisease | ranking drops the unconnected candidate | empty | empty | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> |
| CRP -> DNAmGDF15 | no induction: a shared cause does not link its effects | empty | empty | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> |
| Metformin -> CellularSenescence | no inversion | empty | empty | empty<br><sub>st 7 / tq 256 / bq 256</sub> | empty<br><sub>st 7 / tq 256 / bq 256</sub> | empty<br><sub>st 7 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set holds no such conclusion</sub> |
| CoronaryHeartDisease -> CellularSenescence | no inversion | empty | empty | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> |
| ChronicInflammation -> CoronaryHeartDisease | CHD_REACHING_CAUSES excludes it | empty | empty | empty<br><sub>st 5 / tq 256 / bq 256</sub> | empty<br><sub>st 5 / tq 256 / bq 256</sub> | empty<br><sub>st 5 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set holds no such conclusion</sub> |
| MitochondrialDysfunction -> CoronaryHeartDisease | CHD_REACHING_CAUSES excludes it | empty | empty | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set CONTAINS a conclusion (State CoronaryHeartDisease) (FALSE CHAIN, not licensed by any walk)</sub> |
| SmokingPackYears -> CoronaryHeartDisease | CHD_REACHING_CAUSES excludes it | empty | empty | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set holds no such conclusion</sub> |
| CellularSenescence -> AgeAccelGrim | no Effect reaches AgeAccelGrim | empty | empty | empty<br><sub>st 9 / tq 256 / bq 256</sub> | empty<br><sub>st 9 / tq 256 / bq 256</sub> | empty<br><sub>st 9 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set CONTAINS a conclusion (State AgeAccelGrim) (FALSE CHAIN, not licensed by any walk)</sub> |
| DNAmPAI1 -> AgeAccelGrim | no Effect reaches AgeAccelGrim | empty | empty | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 2 / tq 256 / bq 256</sub> | empty<br><sub>st 150 (budget) / tq 100 / bq 400</sub><br><sub>the engine's belief set CONTAINS a conclusion (State AgeAccelGrim) (FALSE CHAIN, not licensed by any walk)</sub> |
| CoronaryHeartDisease -> CoronaryHeartDisease | self pair | empty | empty | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> | empty<br><sub>no run: no link out of the node</sub> |
<!-- END GENERATED nofalse -->

pln-full, source by source (budget 150 steps / task queue 100 / belief queue 400, every sentence a task, whole graph as beliefs, level 1, the links in the order the production step reads them):

<!-- BEGIN GENERATED full_lane -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

Total: 80 of 100 true pairs found, 20 lost; 30 conclusions that no walk of the graph licenses; 30 false pairs (source, target) outside the true closure. Budget (steps / tq / bq) = [150, 100, 400]. The harness's per-source count of unlicensed conclusions equals the production step's (`unattributed`) for every source: True.

| source | true pairs | found as a route | pairs LOST in the budget | conclusions no walk licenses | false-chain targets (not in the true closure) | cost |
|---|---|---|---|---|---|---|
| AgeAccelGrim | 3 | 3 | - | 7 | CRP, CellularSenescence, DNAmGDF15, DNAmPAI1, LowSerumAlbumin, RDW, SASP | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Autophagy | 9 | 9 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Berberine | 5 | 5 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| CellularSenescence | 8 | 8 | - | 4 | AgeAccelGrim, AllCauseMortality, CongestiveHeartFailure, MitochondrialDysfunction | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| ChronicInflammation | 4 | 4 | - | 1 | MitochondrialDysfunction | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| DNAmPAI1 | 1 | 1 | - | 5 | AgeAccelGrim, AllCauseMortality, CongestiveHeartFailure, DNAmGDF15, RDW | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| DasatinibPlusQuercetin | 9 | 9 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| DeregulatedNutrientSensing | 5 | 5 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| FastingGlucose | 1 | 1 | - | 1 | HbA1c | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Fisetin | 9 | 7 | LowSerumAlbumin, RDW | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| InsulinResistance | 4 | 4 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Metformin | 5 | 5 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| MitochondrialDysfunction | 1 | 1 | - | 7 | CRP, CellularSenescence, ChronicInflammation, CoronaryHeartDisease, LowSerumAlbumin, RDW, SASP | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| NMN | 2 | 2 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Omega3 | 5 | 5 | - | 1 | MitochondrialDysfunction | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Resveratrol | 9 | 0 | CRP, CellularSenescence, ChronicInflammation, CoronaryHeartDisease, DNAmGDF15, DNAmPAI1, LowSerumAlbumin, RDW, SASP | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| SASP | 7 | 7 | - | 4 | AgeAccelGrim, AllCauseMortality, CongestiveHeartFailure, MitochondrialDysfunction | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| SmokingCessation | 2 | 2 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| SmokingPackYears | 1 | 1 | - | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |
| Spermidine | 10 | 1 | CRP, CellularSenescence, ChronicInflammation, CoronaryHeartDisease, DNAmGDF15, DNAmPAI1, LowSerumAlbumin, RDW, SASP | 0 | - | st 150 (budget), tasks left 99 / tq 100 / bq 400, beliefs 399 |

| false pair | unlicensed conclusions | best confidence |
|---|---|---|
| AgeAccelGrim -> CRP | 1 | 0.334848 |
| AgeAccelGrim -> CellularSenescence | 1 | 0.393939 |
| AgeAccelGrim -> DNAmGDF15 | 1 | 0.334848 |
| AgeAccelGrim -> DNAmPAI1 | 1 | 0.39015 |
| AgeAccelGrim -> LowSerumAlbumin | 1 | 0.334848 |
| AgeAccelGrim -> RDW | 1 | 0.334848 |
| AgeAccelGrim -> SASP | 1 | 0.393939 |
| CellularSenescence -> AgeAccelGrim | 1 | 0.459459 |
| CellularSenescence -> AllCauseMortality | 1 | 0.459459 |
| CellularSenescence -> CongestiveHeartFailure | 1 | 0.459459 |
| CellularSenescence -> MitochondrialDysfunction | 1 | 0.393939 |
| ChronicInflammation -> MitochondrialDysfunction | 1 | 0.393939 |
| DNAmPAI1 -> AgeAccelGrim | 1 | 0.39015 |
| DNAmPAI1 -> AllCauseMortality | 1 | 0.390541 |
| DNAmPAI1 -> CongestiveHeartFailure | 1 | 0.390541 |
| DNAmPAI1 -> DNAmGDF15 | 1 | 0.393939 |
| DNAmPAI1 -> RDW | 1 | 0.393939 |
| FastingGlucose -> HbA1c | 1 | 0.393939 |
| MitochondrialDysfunction -> CRP | 1 | 0.393939 |
| MitochondrialDysfunction -> CellularSenescence | 1 | 0.393939 |
| MitochondrialDysfunction -> ChronicInflammation | 1 | 0.393939 |
| MitochondrialDysfunction -> CoronaryHeartDisease | 1 | 0.393939 |
| MitochondrialDysfunction -> LowSerumAlbumin | 1 | 0.393939 |
| MitochondrialDysfunction -> RDW | 1 | 0.393939 |
| MitochondrialDysfunction -> SASP | 1 | 0.393939 |
| Omega3 -> MitochondrialDysfunction | 1 | 0.334848 |
| SASP -> AgeAccelGrim | 1 | 0.459459 |
| SASP -> AllCauseMortality | 1 | 0.459459 |
| SASP -> CongestiveHeartFailure | 1 | 0.459459 |
| SASP -> MitochondrialDysfunction | 1 | 0.393939 |
<!-- END GENERATED full_lane -->

Reading. The restricted rules (R2, R1, R0) answer every one of the 13 pairs with the empty answer, and for them the adapter also checks that every conclusion of every run is a route
or a walk of the graph (`unattributed` 0 in all 20 sources). The full rule set concludes things the links do not license: Induction and Abduction run backwards and sideways
(shared cause -> sibling, effect -> cause), so a hallmark "concludes" its effect's effect and an effect "concludes" its cause. The count of those depends on rules, budget and encoding;
this table's 30 conclusions (30 false pairs) are what this budget and this encoding produce. The decision's reproduction found 10 with another encoding; the finding
(false conclusions exist, and true pairs are lost to the budget) is the same.

## 7. The feedback edge

`(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))` added to the patient stack as a caller fact: a probe, not a curated edge. The reference is the node-simple closure
(109 pairs, nine of them new).

<!-- BEGIN GENERATED feedback -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| lane | terminates | pairs (node-simple reference 109) | missing | extra | max dev strength / confidence vs the node-simple reference | wall | cost |
|---|---|---|---|---|---|---|---|
| legacy (hyperon) | NO | - | - | - | - | run in a subprocess, killed after 25.0 s (finished: False) | no queues |
| legacy (PeTTa) | NO | - | - | - | - | PeTTa exceeded the 10 s deadline and was killed | no queues |
| pln-R2 | yes | 109 of 109 | 0 | 0 | 2.2e-16 / 0.0e+00 | 1.199 s | max st 16 / tq 256 / bq 256; max beliefs 26 |
| pln-R1 | yes | 109 of 109 | 0 | 0 | 2.2e-16 / 6.8e-02 | 1.167 s | max st 16 / tq 256 / bq 256; max beliefs 26 |
| pln-R0 | yes | 109 of 109 | 0 | 0 | 4.1e-02 / 6.8e-02 | 1.169 s | max st 16 / tq 256 / bq 256; max beliefs 26 |
| pln-full | yes | 78 of 109 | 31 | 0 | 3.1e-02 / 4.2e-01 | 81.709 s | st 150 (budget) / tq 100 / bq 400 |

| new pair (not in the shipped graph's 100) | legacy (hyperon) | legacy (PeTTa) | pln-R2 | pln-R1 | pln-R0 | pln-full |
|---|---|---|---|---|---|---|
| ChronicInflammation -> CellularSenescence | - | - | Pos 0.5 / 0.65<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.5 / 0.65<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.48 / 0.65<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.5 / 0.65<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| ChronicInflammation -> SASP | - | - | Pos 0.425 / 0.38025<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.425 / 0.4225<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.3984 / 0.4225<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.442347 / 0.4225<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| ChronicInflammation -> DNAmPAI1 | - | - | Pos 0.2975 / 0.222446<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.2975 / 0.274625<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.270912 / 0.274625<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.303509 / 0.4225<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| ChronicInflammation -> CoronaryHeartDisease | - | - | Pos 0.1785 / 0.170171<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.1785 / 0.233431<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.157129 / 0.233431<br><sub>st 14 / tq 256 / bq 256</sub> | Pos 0.189723 / 0.4225<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Omega3 -> CellularSenescence | - | - | Neg 0.3 / 0.49725<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.3 / 0.5525<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.2784 / 0.5525<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.3 / 0.5525<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Omega3 -> SASP | - | - | Neg 0.255 / 0.290891<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.255 / 0.359125<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.231072 / 0.359125<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.264286 / 0.5525<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Omega3 -> DNAmPAI1 | - | - | Neg 0.1785 / 0.170171<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.1785 / 0.233431<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.157129 / 0.233431<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.198244 / 0.5525<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| Omega3 -> CoronaryHeartDisease | - | - | Neg 0.1071 / 0.130181<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.1071 / 0.198417<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.0911348 / 0.198417<br><sub>st 16 / tq 256 / bq 256</sub> | Neg 0.120022 / 0.5525<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |
| SASP -> CellularSenescence | - | - | Pos 0.425 / 0.38025<br><sub>st 12 / tq 256 / bq 256</sub> | Pos 0.425 / 0.4225<br><sub>st 12 / tq 256 / bq 256</sub> | Pos 0.3984 / 0.4225<br><sub>st 12 / tq 256 / bq 256</sub> | Pos 0.435204 / 0.4225<br><sub>st 150 (budget) / tq 100 / bq 400</sub> |

| lane | old answers byte-identical with the edge added (same lane) | new pairs |
|---|---|---|
| pln-R2 | 100 of 100 | 9 |
| pln-R1 | 100 of 100 | 9 |
| pln-R0 | 100 of 100 | 9 |
| pln-full | 57 of 80 | 9 |
<!-- END GENERATED feedback -->

Reading. The legacy chainer has no visited set, so a cycle reachable from the source makes `infer` and `explain` recurse for ever; both legacy lanes are stopped by a deadline (PeTTa
by the request deadline, 10 s here; hyperon cannot stop itself, so it runs in a subprocess that is killed after 25 s). The pln lanes end because a derivation uses each link once
(its stamp); the cost is 16 selections for the worst source (Omega3), against 12 without the edge. The 100 old answers are byte-identical in the same lane. pln-full, with its fixed
budget shared among more tasks, gives up 31 of the 109 pairs and changes 23 of its 80 old answers.

## 8. DrugAge: a mixed-sign compound, paper level and row level, and three ways to count a null

Source: the 313-row fixture `tests/fixtures/drugage_fold/three_compounds.metta` (Ethanol 52 rows / 1 paper, Metformin 54 / 16, Rapamycin 37 / 18, Resveratrol 170 / 37) and the five-row
hand-built `mix_compound.metta` (four extending rows, one of them an ITP null, one shortening). The unit of evidence is the paper (PMID) at paper level and the row at row level
(`evidence_unit="row"`, a control: not reachable from a request). Within a paper the repository's representative-row policy picks the row; each direction is pooled with `Truth_Revision`; the NET is
`net_of_signed` of `ontology/drugage_fold.py`, one formula for every cell: the sum over the compound's pooled items of +strength x confidence (Neg, protective) or -strength x confidence (Pos).
Legacy cells are the application's ranking (`run_drugage_ranking`): one representative row per compound, then the legacy chainer.

**A reported null.** A row with a change of exactly 0.0 has strength 0 and keeps its confidence (an ITP null: 0.9). The sign convention has no zero, so the legacy ranking files it as protective with
strength 0, which is neutral for ONE row (score 0). Pooled, it is not neutral: it drags the mean of the pool it sits in toward 0. Resveratrol has 23 null rows of 170, Metformin 5 of 54, Rapamycin 1 of 37, Ethanol none. The
table prints each compound three times at paper level: nulls counted on BOTH sides (`ChainSettings.drugage_zero_rows="both"`, the default: the repository's reading is that a well-run null is high-confidence evidence of
no effect, which is evidence about both directions), nulls on the protective side only (`"protective"`, what the first build did) and nulls dropped (`"neither"`). The rows labelled "row level" use the default convention.
A direction whose pool would hold nothing but nulls pools to strength 0, contributes nothing to the NET and is not emitted (Rapamycin's one null therefore dilutes its protective item and gives no harmful one); a compound
of only nulls keeps its protective item of strength 0, the legacy shape.

<!-- BEGIN GENERATED drugage -->
*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 d64db6cb03be437e); ids: legacy (hyperon) = `legacy-hyperon`, legacy (PeTTa) = `legacy-petta`, pln-R2 = `R2`, pln-R1 = `R1`, pln-R0 = `R0`, pln-full = `full`. Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*

| compound | rows | papers | unit | legacy (hyperon) | legacy (PeTTa) | pln-R2 | pln-R1 | pln-R0 | pln-full | Python twin (ontology.drugage_fold) |
|---|---|---|---|---|---|---|---|---|---|---|
| Ethanol | 52 | 1 | paper level (nulls on both sides) | score 0.143237 (Neg 0.757869 / 0.189)<br><sub>one representative row; no queues</sub> | score 0.143237 (Neg 0.757869 / 0.189)<br><sub>one representative row; no queues; 52 derivations if every row is given</sub> | NET 0.114445<br><sub>Pos 0.16283 / 0.189; Neg 0.768358 / 0.189<br>st 8 / tq 256 / bq 256</sub> | NET 0.127161<br><sub>Pos 0.16283 / 0.21; Neg 0.768358 / 0.21<br>st 8 / tq 256 / bq 256</sub> | NET 0.124618<br><sub>Pos 0.139973 / 0.21; Neg 0.733391 / 0.21<br>st 8 / tq 256 / bq 256</sub> | NET 0.127161<br><sub>Pos 0.16283 / 0.21; Neg 0.768358 / 0.21<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.114445<br><sub>legacy twin 0.143237</sub> |
| Ethanol | 52 | 1 | row level (nulls on both sides) | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.567866<br><sub>Pos 0.126041 / 0.312397; Neg 0.725476 / 0.837024<br>st 8 / tq 256 / bq 256</sub> | NET 0.630962<br><sub>Pos 0.126041 / 0.347107; Neg 0.725476 / 0.930027<br>st 8 / tq 256 / bq 256</sub> | NET 0.606918<br><sub>Pos 0.103921 / 0.347107; Neg 0.691367 / 0.930027<br>st 8 / tq 256 / bq 256</sub> | NET 0.630962<br><sub>Pos 0.126041 / 0.347107; Neg 0.725476 / 0.930027<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.567866 |
| Ethanol | 52 | 1 | paper level, nulls on the protective side only | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.114445<br><sub>Pos 0.16283 / 0.189; Neg 0.768358 / 0.189<br>st 8 / tq 256 / bq 256</sub> | NET 0.127161<br><sub>Pos 0.16283 / 0.21; Neg 0.768358 / 0.21<br>st 8 / tq 256 / bq 256</sub> | NET 0.124618<br><sub>Pos 0.139973 / 0.21; Neg 0.733391 / 0.21<br>st 8 / tq 256 / bq 256</sub> | NET 0.127161<br><sub>Pos 0.16283 / 0.21; Neg 0.768358 / 0.21<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.114445 |
| Ethanol | 52 | 1 | paper level, nulls dropped | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.114445<br><sub>Pos 0.16283 / 0.189; Neg 0.768358 / 0.189<br>st 8 / tq 256 / bq 256</sub> | NET 0.127161<br><sub>Pos 0.16283 / 0.21; Neg 0.768358 / 0.21<br>st 8 / tq 256 / bq 256</sub> | NET 0.124618<br><sub>Pos 0.139973 / 0.21; Neg 0.733391 / 0.21<br>st 8 / tq 256 / bq 256</sub> | NET 0.127161<br><sub>Pos 0.16283 / 0.21; Neg 0.768358 / 0.21<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.114445 |
| Metformin | 54 | 16 | paper level (nulls on both sides) | score 0 (Neg 0 / 0.81)<br><sub>one representative row; no queues</sub> | score 0 (Neg 0 / 0.81)<br><sub>one representative row; no queues; 54 derivations if every row is given</sub> | NET 0.118159<br><sub>Pos 0.104841 / 0.833282; Neg 0.242015 / 0.849211<br>st 8 / tq 256 / bq 256</sub> | NET 0.131288<br><sub>Pos 0.104841 / 0.925869; Neg 0.242015 / 0.943568<br>st 8 / tq 256 / bq 256</sub> | NET 0.128315<br><sub>Pos 0.0831446 / 0.925869; Neg 0.217574 / 0.943568<br>st 8 / tq 256 / bq 256</sub> | NET 0.131288<br><sub>Pos 0.104841 / 0.925869; Neg 0.242015 / 0.943568<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.118159<br><sub>legacy twin 0</sub> |
| Metformin | 54 | 16 | row level (nulls on both sides) | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.124532<br><sub>Pos 0.166029 / 0.848153; Neg 0.30349 / 0.87433<br>st 8 / tq 256 / bq 256</sub> | NET 0.138369<br><sub>Pos 0.166029 / 0.942392; Neg 0.30349 / 0.971477<br>st 8 / tq 256 / bq 256</sub> | NET 0.135032<br><sub>Pos 0.143109 / 0.942392; Neg 0.27782 / 0.971477<br>st 8 / tq 256 / bq 256</sub> | NET 0.138369<br><sub>Pos 0.166029 / 0.942392; Neg 0.30349 / 0.971477<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.124532 |
| Metformin | 54 | 16 | paper level, nulls on the protective side only | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET -0.0569663<br><sub>Pos 0.375228 / 0.699542; Neg 0.242015 / 0.849211<br>st 8 / tq 256 / bq 256</sub> | NET -0.0632959<br><sub>Pos 0.375228 / 0.777269; Neg 0.242015 / 0.943568<br>st 8 / tq 256 / bq 256</sub> | NET -0.0652894<br><sub>Pos 0.348124 / 0.777269; Neg 0.217574 / 0.943568<br>st 8 / tq 256 / bq 256</sub> | NET -0.0632959<br><sub>Pos 0.375228 / 0.777269; Neg 0.242015 / 0.943568<br>st 150 (budget) / tq 100 / bq 400</sub> | NET -0.0569663 |
| Metformin | 54 | 16 | paper level, nulls dropped | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.0629186<br><sub>Pos 0.375228 / 0.699542; Neg 0.383187 / 0.849211<br>st 8 / tq 256 / bq 256</sub> | NET 0.0699096<br><sub>Pos 0.375228 / 0.777269; Neg 0.383187 / 0.943568<br>st 8 / tq 256 / bq 256</sub> | NET 0.0652519<br><sub>Pos 0.348124 / 0.777269; Neg 0.355923 / 0.943568<br>st 8 / tq 256 / bq 256</sub> | NET 0.0699096<br><sub>Pos 0.375228 / 0.777269; Neg 0.383187 / 0.943568<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.0629186 |
| MixCompound | 5 | 5 | paper level (nulls on both sides) | score 0.405 (Neg 0.5 / 0.81)<br><sub>one representative row; no queues</sub> | score 0.405 (Neg 0.5 / 0.81)<br><sub>one representative row; no queues; 5 derivations if every row is given</sub> | NET 0.17569<br><sub>Pos 0.06 / 0.818182; Neg 0.26305 / 0.854519<br>st 8 / tq 256 / bq 256</sub> | NET 0.195212<br><sub>Pos 0.06 / 0.909091; Neg 0.26305 / 0.949466<br>st 8 / tq 256 / bq 256</sub> | NET 0.190516<br><sub>Pos 0.0392 / 0.909091; Neg 0.238189 / 0.949466<br>st 8 / tq 256 / bq 256</sub> | NET 0.195212<br><sub>Pos 0.06 / 0.909091; Neg 0.26305 / 0.949466<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.17569<br><sub>legacy twin 0.405</sub> |
| MixCompound | 5 | 5 | row level (nulls on both sides) | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.17569<br><sub>Pos 0.06 / 0.818182; Neg 0.26305 / 0.854519<br>st 8 / tq 256 / bq 256</sub> | NET 0.195212<br><sub>Pos 0.06 / 0.909091; Neg 0.26305 / 0.949466<br>st 8 / tq 256 / bq 256</sub> | NET 0.190516<br><sub>Pos 0.0392 / 0.909091; Neg 0.238189 / 0.949466<br>st 8 / tq 256 / bq 256</sub> | NET 0.195212<br><sub>Pos 0.06 / 0.909091; Neg 0.26305 / 0.949466<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.17569 |
| MixCompound | 5 | 5 | paper level, nulls on the protective side only | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET -0.0452187<br><sub>Pos 0.6 / 0.45; Neg 0.26305 / 0.854519<br>st 8 / tq 256 / bq 256</sub> | NET -0.050243<br><sub>Pos 0.6 / 0.5; Neg 0.26305 / 0.949466<br>st 8 / tq 256 / bq 256</sub> | NET -0.0580476<br><sub>Pos 0.5684 / 0.5; Neg 0.238189 / 0.949466<br>st 8 / tq 256 / bq 256</sub> | NET -0.050243<br><sub>Pos 0.6 / 0.5; Neg 0.26305 / 0.949466<br>st 150 (budget) / tq 100 / bq 400</sub> | NET -0.0452187 |
| MixCompound | 5 | 5 | paper level, nulls dropped | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.142299<br><sub>Pos 0.6 / 0.45; Neg 0.504912 / 0.816578<br>st 8 / tq 256 / bq 256</sub> | NET 0.158111<br><sub>Pos 0.6 / 0.5; Neg 0.504912 / 0.907308<br>st 8 / tq 256 / bq 256</sub> | NET 0.146965<br><sub>Pos 0.5684 / 0.5; Neg 0.475213 / 0.907308<br>st 8 / tq 256 / bq 256</sub> | NET 0.158111<br><sub>Pos 0.6 / 0.5; Neg 0.504912 / 0.907308<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.142299 |
| Rapamycin | 37 | 18 | paper level (nulls on both sides) | score 0.319091 (Neg 0.393939 / 0.81)<br><sub>one representative row; no queues</sub> | score 0.319091 (Neg 0.393939 / 0.81)<br><sub>one representative row; no queues; 37 derivations if every row is given</sub> | NET 0.336795<br><sub>Neg 0.382672 / 0.880114<br>st 4 / tq 256 / bq 256</sub> | NET 0.374216<br><sub>Neg 0.382672 / 0.977905<br>st 4 / tq 256 / bq 256</sub> | NET 0.347565<br><sub>Neg 0.355418 / 0.977905<br>st 4 / tq 256 / bq 256</sub> | NET 0.374216<br><sub>Neg 0.382672 / 0.977905<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.336795<br><sub>legacy twin 0.319091</sub> |
| Rapamycin | 37 | 18 | row level (nulls on both sides) | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.337271<br><sub>Neg 0.377142 / 0.894281<br>st 4 / tq 256 / bq 256</sub> | NET 0.374745<br><sub>Neg 0.377142 / 0.993646<br>st 4 / tq 256 / bq 256</sub> | NET 0.347775<br><sub>Neg 0.349999 / 0.993646<br>st 4 / tq 256 / bq 256</sub> | NET 0.374745<br><sub>Neg 0.377142 / 0.993646<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.337271 |
| Rapamycin | 37 | 18 | paper level, nulls on the protective side only | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.336795<br><sub>Neg 0.382672 / 0.880114<br>st 4 / tq 256 / bq 256</sub> | NET 0.374216<br><sub>Neg 0.382672 / 0.977905<br>st 4 / tq 256 / bq 256</sub> | NET 0.347565<br><sub>Neg 0.355418 / 0.977905<br>st 4 / tq 256 / bq 256</sub> | NET 0.374216<br><sub>Neg 0.382672 / 0.977905<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.336795 |
| Rapamycin | 37 | 18 | paper level, nulls dropped | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.336795<br><sub>Neg 0.382672 / 0.880114<br>st 4 / tq 256 / bq 256</sub> | NET 0.374216<br><sub>Neg 0.382672 / 0.977905<br>st 4 / tq 256 / bq 256</sub> | NET 0.347565<br><sub>Neg 0.355418 / 0.977905<br>st 4 / tq 256 / bq 256</sub> | NET 0.374216<br><sub>Neg 0.382672 / 0.977905<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.336795 |
| Resveratrol | 170 | 37 | paper level (nulls on both sides) | score 0 (Neg 0 / 0.81)<br><sub>one representative row; no queues</sub> | score 0 (Neg 0 / 0.81)<br><sub>one representative row; no queues; 170 derivations if every row is given</sub> | NET 0.136348<br><sub>Pos 0.0940302 / 0.830658; Neg 0.245363 / 0.874032<br>st 8 / tq 256 / bq 256</sub> | NET 0.151498<br><sub>Pos 0.0940302 / 0.922953; Neg 0.245363 / 0.971146<br>st 8 / tq 256 / bq 256</sub> | NET 0.147523<br><sub>Pos 0.0725496 / 0.922953; Neg 0.220855 / 0.971146<br>st 8 / tq 256 / bq 256</sub> | NET 0.151498<br><sub>Pos 0.0940302 / 0.922953; Neg 0.245363 / 0.971146<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.136348<br><sub>legacy twin 0</sub> |
| Resveratrol | 170 | 37 | row level (nulls on both sides) | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.1125<br><sub>Pos 0.0581685 / 0.883067; Neg 0.184136 / 0.889918<br>st 8 / tq 256 / bq 256</sub> | NET 0.124999<br><sub>Pos 0.0581685 / 0.981186; Neg 0.184136 / 0.988798<br>st 8 / tq 256 / bq 256</sub> | NET 0.12235<br><sub>Pos 0.0374052 / 0.981186; Neg 0.160854 / 0.988798<br>st 8 / tq 256 / bq 256</sub> | NET 0.124999<br><sub>Pos 0.0581685 / 0.981186; Neg 0.184136 / 0.988798<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.1125 |
| Resveratrol | 170 | 37 | paper level, nulls on the protective side only | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET -0.0460044<br><sub>Pos 0.38946 / 0.668769; Neg 0.245363 / 0.874032<br>st 8 / tq 256 / bq 256</sub> | NET -0.051116<br><sub>Pos 0.38946 / 0.743077; Neg 0.245363 / 0.971146<br>st 8 / tq 256 / bq 256</sub> | NET -0.0545639<br><sub>Pos 0.362071 / 0.743077; Neg 0.220855 / 0.971146<br>st 8 / tq 256 / bq 256</sub> | NET -0.051116<br><sub>Pos 0.38946 / 0.743077; Neg 0.245363 / 0.971146<br>st 150 (budget) / tq 100 / bq 400</sub> | NET -0.0460044 |
| Resveratrol | 170 | 37 | paper level, nulls dropped | n/a (legacy has no pooling) | n/a (legacy has no pooling) | NET 0.0320674<br><sub>Pos 0.38946 / 0.668769; Neg 0.338346 / 0.864578<br>st 8 / tq 256 / bq 256</sub> | NET 0.0356304<br><sub>Pos 0.38946 / 0.743077; Neg 0.338346 / 0.960642<br>st 8 / tq 256 / bq 256</sub> | NET 0.0306535<br><sub>Pos 0.362071 / 0.743077; Neg 0.311979 / 0.960642<br>st 8 / tq 256 / bq 256</sub> | NET 0.0356304<br><sub>Pos 0.38946 / 0.743077; Neg 0.338346 / 0.960642<br>st 150 (budget) / tq 100 / bq 400</sub> | NET 0.0320674 |

NET = sum over the compound's pooled items of +strength x confidence (Neg, protective) or -strength x confidence (Pos), `ontology.drugage_fold.net_of_signed`; confidence is the lane's (R2 includes the 0.9 of the second hop; R1, R0 and full do not). pln-R2 against the pure-Python twin over 20 cells (compound x unit): max abs difference 1.7e-16. Legacy cells are the application's own ranking (one representative row per compound).
<!-- END GENERATED drugage -->

Reading. (1) **The sign of Resveratrol and Metformin is the null convention's.** Nulls on both sides: +0.1363 and +0.1182; dropped: +0.0321 and +0.0629; on the protective side only: -0.0460 and -0.0570. Over the
whole build 70 of 1,035 compounds have a null row; counting the nulls on both sides instead of one flips the sign of exactly these two, dropping them flips 8 (`b2/results/measure.json`, `null_conventions`).
The earlier reading (1) of this table, "the sign does not depend on the unit, the size does", is true of paper against row and false of the null convention, and it is the convention that decided the two compounds
the owner question was built on. (2) Rapamycin is +0.3368 under every convention (one null in 37; row level +0.3373) and stays protective (the legacy representative says +0.3191). (3) Ethanol, whose 52 rows
are one paper, is +0.1144 at paper level (it has no null) and +0.5679 at row level: counting the arms of one experiment as 52 experiments would make it the most protective of the four, which is the effect the
paper unit removes. (4) The mixed compound shows both directions as separate pooled items (Neg 0.2631 / 0.8545, Pos 0.06 / 0.8182 with the ITP null on both sides) and nets to +0.1757; with the null on the
protective side only the Pos item is the single shortening row (0.6 / 0.45) and the net is -0.0452; the legacy ranking picks the ITP row and says +0.405. (5) R1, R0 and full differ from R2 here only through the
confidence convention (no 0.9 for the second hop) and, for R0, through the strength mapping; the NET is not comparable across lanes, only within one.

## 9. What this table does not show, and what I would not conclude from it

* It covers the shipped patient stack (28 links), the patient stack plus one feedback edge, and four DrugAge compounds with one hand-built one. Not the LinAge2 or generic stacks
  (`tests/test_pln_chainer.py` and `tests/test_pln_vendor.py` cover their 86 and 101 pairs), and not the whole-build DrugAge ranking, which does not go through the engine.
* The legacy (hyperon) lane runs the patient stack only. The full shared stack aborts hyperon 0.2.10 on one of the repository's tests (the known trie.rs:179 abort); nothing here goes near it.
* "pln-full" is a research lane with a fixed budget. Its numbers are what that budget and the first-conclusion-with-the-stamp rule give; I make no claim that another budget would not find the 20 lost
  pairs, only that this one does not, and that no budget removes the 30 conclusions the links do not license (they come from rules, not from the budget).
* Timings are not the point of this document; the wall of each lane is in the first block. The cost of the pln lane per request is measured in `g0/notes.md` and `b1/notes.md`.
* Two things are tests of the table, not of the library: the pins are the tests' arithmetic (a pin file could be wrong), and the "recorded closure" is the contract study's recording, which the
  legacy (hyperon) lane of this run reproduces exactly.

## 10. Questions this table puts to the owner and to Ben

1. **Confidence convention.** R1 is "what the fork says"; R2 is "what the application has always printed". The table prices the choice: one ordering (Patient001, places 4 and 5), one diagnosis tolerance
   (c(CS) 0.6961 against 0.6253 +- 0.01), every confidence between 1 and 1.524 times larger. Is the 0.9 per combination a PLN concept or a legacy artefact?
2. **DrugAge NET and the null.** Is the paper the independent unit, and is "NET of the two directions" the quantity to rank by? And how does a reported null count: as evidence of no effect in both
   directions (the default: Resveratrol +0.1363, Metformin +0.1182), dropped (+0.0321, +0.0629), or on the protective side only (-0.0460, -0.0570: the sign convention's accident)? Revision pools the WEIGHT
   of evidence, not its agreement (Rapamycin's 18 papers give confidence 0.978 whatever their spread), and treats papers as independent: is that what Ben means?
3. **pln-full.** If Ben wants Induction, Abduction or Inversion in the service, the table is the list of what that costs: true pairs lost to the budget and conclusions with no license. The restricted rules
   are the ones for which the adapter can prove "no false chain".
