# The chaining contract of the legacy `infer` / `explain` (what a replacement chainer must meet)

Everything here was read from, or measured on, the work tree `/Users/kolya/PycharmProjects/pln_longevity_petta`
(branch `petta-migration`, HEAD 8a2a6a0; `*.metta`, `pln_chat/` and `tests/` are identical to linage2 @ 47130aa).
Numbers were produced by RUNNING the repo's own hyperon 0.2.10 (`/Users/kolya/PycharmProjects/pln_longevity/.venv`)
on the repo's own stacks (`core.pln_runner.patient_stack(api._runtime_kb_paths())` etc.). Nothing from `.petta/` was read or run.
Machine-readable companions in this folder: `links.json` (every link + node roles + lift rules), `closure.json`
(every derivation of every ordered node pair, exact floats), `pinned_numbers.json` (composite outputs + the test that pins each),
`legacy_reference.py` (a 40-line pure-Python executable spec; `verify_reference.py` proves it bit-identical to hyperon on all
100 + 86 closure pairs), plus the probe scripts. "file:line" is relative to the repo root.

Words: "legacy" = the MeTTa chainer in `pln_deduction.metta`. "link" = one traversable edge. "derivation" = one result of `infer`.

---------------------------------------------------------------------------------------------------------------------------

## 1. Interface

### 1.1 Forms (all in `pln_deduction.metta`)

| form | lines | returns |
|---|---|---|
| `(infer <space> <A> <C>)` | 132 (base), 133-135 (step) | 0..n of `(signed <Pos\|Neg> (stv <s> <c>))` |
| `(explain <space> <A> <C>)` | 144-145 (Effect base), 146-148 (record base), 149-151 (step), `prepend-node` 153-155 | 0..n of `(Chain (<n0> ... <nk>) (signed <Pos\|Neg> (stv <s> <c>)))` |
| `(link-effect <space> <A> <C>)` | 113-117 | the single-hop lift, same `(signed ..)` shape; only `infer`/`explain` and tests call it |

* `<space>` is always `&self` and is **threaded explicitly** by every layer (a bare `(match &self ..)` inside an imported
  module reads only that module's space: `pln_deduction.metta:96-110`, `evidence_calibration.metta:43-49`). A replacement keeps the
  3-argument signature. A variable endpoint works in legacy (`(infer &self $x D)` enumerates every source that reaches D; probed) but I found no caller that uses it.
* `A`, `C` are plain atoms. **No type check, no validation**: unknown symbols give an empty result, `(Effect P Q Maybe ..)` passes through,
  a non-number strength yields an unreduced symbolic expression (probed). Validation lives in Python (write gate, section 5 of dependencies.md).
* Return shapes: `s` and `c` are reduced IEEE doubles printed by hyperon (`0.24989999999999998`, `0.17017138125000003`). Python reads them with regexes that require plain
  decimal/exponent literals: `_SCORED_RE` (`pln_chat/core/pln_runner.py:415-420`) and `_STV_RE` (`:588-591`, first `(stv a b)` in the atom wins). A reduced
  number is therefore part of the contract: an unevaluated `(evidence-confidence ..)` in the output would break `confidence_threshold` and `parse_scored`.
* `explain` agrees with `infer`: on all 100 patient-stack and 86 LinAge2-stack pairs `explain`'s `(signed ..)` is identical to `infer`'s, and `(Chain nodes ..)` lists
  both endpoints (`(Chain (A C) ..)` for one hop). `explain` carries **no per-hop truth values**, only the node path.

### 1.2 Multiplicity = one result per derivation, never merged

* The call is non-deterministic: each distinct derivation is a separate result. There is **no revision / no merge** (docs/deduction_layer.md 7.4).
* Duplicate identical `Effect` atoms give duplicate results (bag semantics, probed).
* Derivation count per pair in the SHIPPED graphs: **exactly 1 for every connected pair** (patient stack: 100 pairs / 100 derivations; LinAge2 stack: 86 / 86; `closure.json`).
  The graph is a forest-like DAG (no diamond), max 5 edges (`Spermidine -> Autophagy -> CellularSenescence -> SASP -> DNAmPAI1 -> CoronaryHeartDisease`);
  hop histogram patient: 1 edge 28, 2 edges 28, 3 edges 17, 4 edges 22, 5 edges 5.
* Multiplicity DOES occur in three real situations a replacement must decide about:
  1. **DrugAge compound with several rows**: the third `infer` equation (`drugage_calibration.metta:240-243`) yields one derivation per ROW. Probed:
     `MultiDrug` (ITP mouse +20 sig, worm +40 sig, mouse -30 not sig) gives `(Pos 0.6 0.18000000000000002) (Neg 0.6666666666666666 0.315) (Neg 0.5 0.81)`; `rank-interventions`
     lists the compound three times. Python avoids it by choosing one representative row per compound before MeTTa (`ontology/drugage_selector.py`, `run_drugage_ranking`
     in `pln_runner.py:861-960`) and dedupes by best score (`:950-958`).
  2. **Two provenance paths for one claim** (an extracted block carrying both a raw DrugAge row and a curated `(Effect X Lifespan ..)`): `infer Taurine Mortality` returns the
     SAME `(signed Neg (stv 0.3548387096774194 0.45))` twice, `explain` once (explain has no DrugAge equation), `rank-interventions` lists Taurine twice
     (probed; `tests/test_ontology_expansion.py::test_the_canonical_block_is_consumed_by_the_rules` only asserts the sign set `{Neg}`).
  3. **Caller-injected atoms** through `/metta/run` `extra_atoms` (ground facts are allowed; only `(= ..)` definitions are refused, `pln_chat/api.py:574-590`).
* Observed order inside one call (hyperon 0.2.10, an interpreter artefact, not a documented guarantee): the one-hop lifts first, then step derivations **in reverse of Effect-atom load order**
  (diamond probe: atoms A->B then A->C come back C-path, B-path). Among the one-hop lifts `infer` returns the record lift BEFORE the Effect lift while `explain` returns the Effect first
  (probe with a scratch `(Effect AgeAccelGrim CoronaryHeartDisease Neg ..)` beside the Lu 2019 record: `infer` gives Pos 0.5/0.85, Neg 0.9/0.8, then the 2-edge Pos 0.54/0.612; both lifts are kept, with opposite signs, no preference, no merge).
  No shipped output depends on any of this because no shipped pair has two derivations.

### 1.3 No path

Empty result: hyperon returns one empty group (`[[]]`), the HTTP layer reports status `empty`. Verified empty: `DasatinibPlusQuercetin -> AllCauseMortality`, `SmokingCessation -> CoronaryHeartDisease`,
`Rapamycin -> CoronaryHeartDisease`, `X -> X`, anything `-> AgeAccelGrim` (no Effect reaches it), any DrugAge compound `-> Lifespan` (equation 3 needs a further hop), `explain Compound Mortality` (no equation),
every CellAge gene (`cellage-effect` lifts on demand and is never an `infer` link; probed with the real fixture rows).
The consumers rely on the emptiness: `rank-interventions` drops the candidate, `explains-obs` drops the hypothesis, `relevance-on` contributes 0, `resolve-lever` is separate (section 2).

### 1.4 Several paths

All are returned (no choice, no merge). Consumers' behaviour on >1 derivation is accidental, not designed:
`score-candidate` lists each; `relevance-on` results are SUMMED by `patient-relevance` (double counting); `explains-obs` counts each as a separate explained observation (coverage inflated, mass summed);
`pos-transmission` takes the FIRST Pos derivation (`car-atom` of a collapse, so it depends on result order); `component-cause` lists the hallmark once per derivation (only `pln_linage2.metta:120` dedupes with `unique-tuple`).

### 1.5 Cycle

**Non-terminating.** There is no visited set. A cycle reachable from `A` hangs the call, even when `C` is unreachable and even for a self-loop (`(Effect A A ..)`); probed with 8 s kill:
`cycle_noexit`, `cycle_with_exit`, `cycle_explain`, `selfloop` all TIMEOUT; a cycle NOT reachable from `A` is harmless. The shipped graphs are acyclic. Nothing checks acyclicity at write time
(`ontology/expander.py`, `expansion_schema.py` gate shapes only) and `/metta/run` accepts caller `Effect` facts, so a caller can hang a worker; the process-level deadline
(`core/executor.py`, `PLNExecutionTimeout`) is the only guard. `docs/deduction_layer.md` 7.5 lists "cycle guarding" as undone.

### 1.6 What the legacy rule actually computes

```
derive(A,C) = [ (Effect A C s c) lifts ]                 link-effect eq.1  (pln_deduction.metta:113-114)
            + [ record(A,C) lifts ]                      link-effect eq.2  (:115-117)   TERMINAL ONLY
            + [ combine(hop(A,B), d)  for each (Effect A B ..), d in derive(B,C) ]       infer eq.2  (:133-135)
            + [ DrugAge row lifts ]                      infer eq.3 (drugage_calibration.metta:240-243), only in stacks that load it
combine((g1,s1,c1),(g2,s2,c2)) = ( sign-product(g1,g2),  s1*s2,  (c1*c2)*0.9 )        (:82-95; chain-discount (:87) = 0.9)
```
* sign: Pos*Pos=Pos, Pos*Neg=Neg, Neg*Pos=Neg, Neg*Neg=Pos (`:82-85`). Pos = "A raises C", Neg = "A reduces C"; strength is a magnitude independent of sign.
* strength is the plain product, **no base rates** (docs/deduction_layer.md 7.2). Confidence is multiplied and discounted ONCE PER COMBINATION: n edges give n-1 discounts
  (1 edge: no discount; the pins in test_pln_deduction.py are written as `0.65**3 * 0.85 * 0.9**3` for 4 edges).
* **Floating-point association is part of the golden strings**: right-nested, `s = s_ab * s_rest`, `c = (c_ab * c_rest) * 0.9`. `Metformin->CHD` is `0.33000000000000007`, not `0.33`. `legacy_reference.py` reproduces this exactly.
* Records are TERMINAL only: `infer` steps through `Effect` atoms only, so `DNAmPAI1 -> CoronaryHeartDisease` (HR record) can end a chain but `AgeAccelGrim` (source of 3 records, target of no Effect) cannot be reached mid-chain (docs 7.3).
* Record lift: strength bucket `0.70 if HR>1.5, 0.60 if HR>1.1 else 0.50` (`evidence_calibration.metta:62-64`; HR<1 also gets 0.50), sign `Neg if HR<1.0 else Pos` (`pln_deduction.metta:111`), confidence = `evidence-confidence(HasEvidenceCategory)`
  (`evidence_calibration.metta:35-38`: all four Lu 2019 records `MultipleHumanTrials` 0.85; Fong 2025 `Epidemiological` 0.60 at `linage2_fong2025_evidence.metta:50`). `EffectScale` is ignored.
* Effect atoms store the confidence UNEVALUATED: `(Effect X Y Pos (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))`. `(: Effect (-> Atom Atom Atom Atom Atom))` (`pln_deduction.metta:57`) stops hyperon reducing it;
  `link-effect` / `combine-tv` evaluate it on the way out. 25 atoms are written this way (mechanistic_bridges 19, supplement_evidence 3, lifestyle_evidence 2, linage2_core 1). A literal `(stv 0.9 0.8)` also works (the `confidence_threshold` test injects such atoms).
* Stack dependence: the patient stack's `infer` has 2 equations; the generic/drugage/ontology-expansion stacks have the DrugAge third equation. `infer`/`explain` outputs for non-DrugAge pairs must be identical in both
  (`tests/test_generic_stack.py::ANSWERED` compares atoms across stacks byte-for-byte).

### 1.7 Graph size to hold in mind (`links.json`)

Patient stack (what the API runs for any program that names a patient): 30 nodes in the graph, 28 links = 24 `Effect` atoms (curated) + 4 Lu 2019 records; closure 100 ordered pairs.
LinAge2 stack: 29 nodes, 27 links (drops 3 supplement edges, adds `SmokingCessation->CurrentTobaccoExposure` and the Fong record), closure 86.
Generic stack adds `Lifespan->Mortality` (+DrugAge equation). DrugAge stack: 1 fixed link, rows per query. CellAge stack: 0 links.

---------------------------------------------------------------------------------------------------------------------------

## 2. How each consumer uses the results

(`s*c` below is the product of the returned strength and confidence.)

| consumer | file:line | takes from `infer` | what it does with it |
|---|---|---|---|
| `rank-score` | `pln_intervention_ranking.metta:44-45` | `(signed Neg (stv s c))` / `Pos` | `Neg -> +s*c`, `Pos -> -(s*c)`. Sign first, magnitude and confidence together. Higher = more protective. |
| `score-candidate`, `rank-interventions`, `sort-scored` | `:61-63`, `:108-110`, `:79-91` | every derivation | one `(scored X score (signed ..))` per derivation; `>=` insertion sort (stable: earlier input stays first on ties); unconnected candidates drop out. |
| `relevance-on` / `patient-relevance` | `patient_profile.metta:353-356`, `:370-374` | per (intervention, elevated marker) | `max(0, rank-score)` per marker, **summed** over the patient's elevated markers. Elevated = z > `(elevated-z-threshold)` 1.0 (`:156`). |
| `personalized-score`, `rank-interventions-for-patient` | `:382-383`, `:389-400` | the intervention->outcome derivation | `rank-score(r) + patient-relevance` (additive), then `sort-scored`. |
| `explains-obs` / `hyp-evidence` / `diagnose` | `pln_abductive_diagnosis.metta:43-50`, `:102-110` | only `(signed Pos ..)` derivations (`let (signed Pos $tv)` pattern) | coverage = number of explained obs (`ev-count` :65-67); mass = sum s*c (:70-73); reported strength = max s (:76-79); reported confidence = noisy-OR `1-prod(1-c)` (:84-88); rank coverage first then mass (`hyp-ge` :118-121). |
| `pos-transmission` | `pln_counterfactual.metta:107-109` | Pos derivations, **first** one | `(stv s c)` of the first Pos derivation, or nothing. |
| `component-cause` | `pln_counterfactual.metta:124-126` | Pos derivation hallmark -> GrimAge component | list of hallmarks that raise the component (`DrivenBy`). |
| `cf-component`, `counterfactual` | `:275-289`, `:340-350` | `pos-transmission(driver, component)` | reduction = strength x patient z; delta = `grimage-weight` x reduction; **confidence = contribution-weighted mean** of per-component c (`cf-conf` :320-322); result sign is always `Neg` (a reduction of a harmful clock is protective). |
| `resolve-lever` | `:195-205` | NOT `infer`: direct `match` on `Effect` atoms | intervention -> node it reduces (`Neg` out-edge), leaf marker -> its `Pos` source, driver -> itself, no edge -> nothing. |
| `predict-risk`, `risk-confidence` | `pln_risk_prediction.metta:272-274`, `:298-299` | NOT `infer`: `calibrate-tv` of the Lu 2019 record | confidence = record c x `(risk-conf-discount)` 0.9 = 0.765. Uses infer only indirectly through `project-risk -> counterfactual-patient` (`:382-394`). |
| `supplement-rec`, `supp-targets` | `pln_supplement_recommendation.metta:101-106`, `:116-129` | via `relevance-on` / `patient-relevance` | score = patient-relevance x `safety-factor`; Targets = elevated markers with `relevance-on > 0.0`. Tier comes from `evidence-tier(EvidenceLevel)`, **not** from the chain. |
| LinAge2 `linage-biomarker-causes` | `pln_linage2.metta:119-123` | Pos derivations | hallmark causes of a readout biomarker, deduped with `unique-tuple`. |
| LinAge2 `linage-transmission`, `linage-cf-feature` | `:286-289`, `:298-319` | `pos-transmission`, plus direct `Effect` match for the lever edge | reduction years = composed strength x contribution years; conf-weighted mean (`lin-conf` :344-346). |
| `explain` | none in the repo's MeTTa | - | only the translator/user (`scripts/counterfactual_chat_demo.py:111`, tests); the LLM prompt does not teach it explicitly, but the pasted `pln_deduction.metta` header does. |
| Python twin of the DrugAge chain | `pln_chat/ontology/drugage_scoring.py:score_row` (:170-210, confidence at :197) | not a call: re-implements | strength `|pct|/(|pct|+20)`, confidence `tier x gate x chain_discount`, sign Neg for pct>=0 (through Lifespan->Mortality). Reads `chain-discount` and `lifespan-halfsat` from the `.metta` TEXT with regexes (`drugage_scoring.py:52-56`). `linear` and `metta_sort` strategies must agree (`tests/test_drugage_ranking_contract.py::test_linear_and_metta_sort_agree_on_every_score`). |
| LLM-translated queries | `pln_chat/prompts/few_shot_examples.json` (1 of 46 entries uses `(infer &self CellularSenescence CoronaryHeartDisease)`) | direct | the `(signed Pos\|Neg (stv s c))` is shown to the user; `formatting.py:55-57` prints the first stv. |

Seven `infer` call sites outside `pln_deduction.metta`/`drugage_calibration.metta`: `patient_profile.metta:355`, `:390`; `pln_abductive_diagnosis.metta:44`; `pln_intervention_ranking.metta:62`; `pln_counterfactual.metta:108`, `:126`; `pln_linage2.metta:123`.
No MeTTa file calls `explain`. Direct `(match $space (Effect ..))` reads (a second, graph-structure interface the chainer does not mediate): `pln_counterfactual.metta:196-198`, `:278`; `pln_linage2.metta:276-278`, `:308`.

---------------------------------------------------------------------------------------------------------------------------

## 3. Where confidence is composed by hand OUTSIDE `infer` (verified line numbers)

| # | site | what | note |
|---|---|---|---|
| 1 | `pln_counterfactual.metta:279-284` (the `stv` at **282-284**) | lever edge x driver->component transmission: `(stv (* s_lever s_base) (* (* c_lever c_base) (chain-discount)))` | the same rule as `combine-tv`, re-typed inline; reads the lever's `Effect ... Neg` atom directly (`:278`). Comment `:266-273` says it is inline because the shared space is 4 definitions from the hyperon abort. |
| 2 | `pln_linage2.metta:309-314` (the `stv` at **312-314**) | identical composition for the LinAge2 counterfactual | line numbers confirmed; reads `:308`. |
| 3 | `pln_linage2.metta:286-289` | `linage-transmission`: `(stv 1.0 1.0)` identity when the driver IS the readout biomarker (cessation -> CurrentTobaccoExposure) | a chain of length 0 with full confidence and no discount. |
| 4 | `pln_counterfactual.metta:77` | `grimage-weight` = **0.125 = fixed 1/8** over the eight `(PartOf X GrimAge)` surrogates; contribution = 0.125 x z, delta = 0.125 x strength x z (`:132-135`, `:288`); reused by `pln_risk_prediction.metta:336` | not a chain quantity; part of the counterfactual numbers. |
| 5 | `pln_counterfactual.metta:320-322` (`cf-conf`), `pln_linage2.metta:344-346` (`lin-conf`) | counterfactual confidence = contribution-weighted mean of per-component transmission confidences | not noisy-OR, not a product. |
| 6 | `pln_risk_prediction.metta:164`, `:272-274`, `:298-299` | risk confidence = `calibrate-tv` c of `Lu2019_AgeAccelGrim_CHD` (0.85) x `(risk-conf-discount)` 0.9 = 0.765; the CI half-width is `(1-c) * risk-ci-k` (`:169`, `:304-311`) | `risk-conf-discount` is a SEPARATE literal 0.9 (comment at `:160-163` says "in lockstep" with `chain-discount` but nothing ties them). |
| 7 | `pln_linage2.metta:203-208` | `linage-confidence` = `calibrate-tv` c of the Fong record (0.60) x `(risk-conf-discount)` = 0.54; hazard multiplier `HR^years` (`:213-215`) | the AllCause hazard path never calls `infer`. |
| 8 | `pln_abductive_diagnosis.metta:69-88` | mass = sum s*c, strength = max s, confidence = noisy-OR | consumer-side aggregation over derivations. |
| 9 | `patient_profile.metta:353-374`, `:382-383` | relevance = sum of clamped rank-scores; personalized score = rank-score + relevance | additive. |
| 10 | `pln_supplement_recommendation.metta:116-129` | score = relevance x safety-factor (1.0 / 0.8 / 0.6, `:56-58`) | evidence confidence shown in `(evidence <cat> <conf>)` is `evidence-confidence`, not a chain value. |
| 11 | `drugage_calibration.metta:139-144`, `:173-176`, `:226` | lift strength `|pct|/(|pct|+20)`, confidence `tier x sig-gate`, neutral `(Effect Lifespan Mortality Neg (stv 1.0 1.0))` | composed before `infer`; the python twin repeats it with `* chain_discount`. |
| 12 | `cellage_calibration.metta:90`, `:96`, `:131` | strength 0.70, confidence `(evidence-confidence InVitro)` 0.35 per row | never chained. |

---------------------------------------------------------------------------------------------------------------------------

## 4. What the tests pin (test, value, tolerance)

Default tolerance below is `pytest.approx` (relative 1e-6, absolute 1e-12) unless stated. "EXACT" = string equality (so float association and tuple order matter).

### 4.1 Chain numbers (`tests/test_pln_deduction.py`, patient-stack KB; exact legacy floats in `closure.json`)

| pair | test | value |
|---|---|---|
| `calibrate-tv` Lu2019 AllCauseMortality / CHD / DNAmPAI1_CHD | `test_calibration_values` | (0.5,0.85) / (0.5,0.85) / (0.6,0.85) |
| `sign-product` 4 cases | `test_sign_product` | Pos,Neg,Neg,Pos |
| `link-effect DNAmPAI1 CHD` | `test_direct_link_is_calibrated_and_signed` | Pos (0.6, 0.85) |
| `infer SASP CHD` | `test_one_hop_propagation` | Pos 0.42, c = 0.65*0.85*0.9 = 0.49725 |
| `infer CellularSenescence CHD` | `test_novel_cross_source_inference_harmful` | Pos 0.357, c = 0.65*0.65*0.85*0.9*0.9 (0.29089125); `c < 0.65` |
| `infer DasatinibPlusQuercetin CHD` | `test_protective_intervention_flips_sign` | Neg 0.2499, c = 0.65**3*0.85*0.9**3 (0.17017138125) |
| `explain DasatinibPlusQuercetin CHD` | `test_explain_returns_signed_path` | EXACT substring `(Chain (DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease)` and `(signed Neg` |
| `infer DasatinibPlusQuercetin AllCauseMortality` | `test_no_false_chain` | empty |
| `infer Fisetin CHD` | `test_fisetin_protective_weaker_than_dq` | Neg 0.23205, c = 0.5*0.65*0.65*0.85*0.9**3 (0.1309011) |
| `infer Spermidine CHD` | `test_spermidine_longer_chain_lower_confidence` | Neg 0.17136, c = 0.5*0.65**3*0.85*0.9**4 (0.0765771) |
| `rank-score` | `test_rank_score_signs` | Neg(0.4,0.5) -> +0.2; Pos -> -0.2 |
| ranking `(DQ Fisetin Spermidine Elamipretide)` vs CHD | `test_rank_interventions_orders_by_protection`, `_is_order_independent` | order [DQ, Fisetin, Spermidine]; Elamipretide absent; any input order gives the same list |
| `infer Metformin CHD` | `test_patient_grounding.py::test_metabolic_axis_reaches_chd` | Neg 0.33, c = 0.85*0.65*0.60*0.9**2 = 0.268515 |
| two ad-hoc chains | `test_drugage_ranking_contract.py::test_confidence_threshold_removes_low_confidence_atoms_from_run_query` | `ProbeA->B (0.9,0.8)`, `B->C (0.9,0.9)`: c = 0.648; `D->E (0.9,0.2)`, `E->F (0.9,0.3)`: c = 0.054; loaded with ONLY system_types, logical_predicates, epistemic_calibration, pln_deduction + caller `extra_atoms` |
| no CHD chain for smoking | `test_smoking_lever.py::test_the_smoking_axis_still_does_not_reach_chd_through_infer` | `infer SmokingCessation CHD` == [] |
| Rapamycin unreachable | `test_hallmark_targeting.py::test_rapamycin_is_still_unreachable_by_inference` | `infer Rapamycin CHD` == []; ranking `(DQ Rapamycin)` has no Rapamycin |
| DrugAge | `test_drugage_calibration.py` (HOP = 0.9) | `infer ExtDrug Mortality` Neg 30/50, c = 0.90*HOP; `ToxDrug` Pos 30/50; ranking order [GoodDrug, WeakDrug, NilDrug, BadDrug], scores 0.5*0.90*HOP, (40/60)*0.35*HOP, 0.0 (approx), BadDrug < 0; real rows: Rapamycin first, Resveratrol and Metformin scores 0 (abs 1e-9) |
| DrugAge ranking confidences | `test_drugage_ranking_contract.py` | Rapamycin 0.81, Trimethadione 0.315 (HTTP, skipped without the ETL build) |
| Taurine | `test_ontology_expansion.py::test_the_canonical_block_is_consumed_by_the_rules` | `infer Taurine Mortality` non-empty, sign set == {Neg}; `explain` mentions Lifespan and Mortality; `(scored Taurine` in the ranking |
| Taurine control | `::test_the_evaluations_own_block_stays_inert` | `infer` and `explain Taurine2 Mortality` empty |

### 4.2 Aggregates built on `infer`

| what | test | value / ordering | tolerance |
|---|---|---|---|
| diagnose `(CS MD CI)` over `(DNAmPAI1 DNAmGDF15 CRP)` | `test_pln_deduction.py::test_diagnosis_ranks_by_coverage`, `_provenance_and_tv` | names [CS, CI, MD], coverage [3,2,1]; CS markers {DNAmPAI1,DNAmGDF15,CRP}; MD markers {DNAmGDF15}; s(CS)=0.595; c(MD)=0.65; GenomicInstability dropped | default |
| diagnose-patient Patient001 | `test_patient_grounding.py` | same order/coverage; s(CS) 0.595; c(CS) 0.625 | **abs 1e-2** (actual 0.6253054502923161); AgeAccelGrim in no SupportedBy |
| diagnose-patient Patient002 `(CS DNS MD)` | `test_diagnose_patient002_finds_metabolic_cause` | only DeregulatedNutrientSensing, markers {FastingGlucose, HbA1c} | set |
| default causes | `test_patient_stack.py` default-causes tests | first hypothesis per patient: P001 CS, P002 InsulinResistance, P003 SmokingPackYears; `(Hypothesis InsulinResistance (stv 0.8 0.8775) 2.0 ` and `(SupportedBy (HbA1c FastingGlucose))` for a tab smoker | EXACT substrings |
| patient-relevance P001 | `test_patient_relevance_ranks_by_targeting_the_patients_markers` | DQ > Fisetin > Spermidine > 0 (0.18808, 0.13434, 0.05804); Elamipretide 0.0 | order; approx |
| relevance Metformin | `test_metformin_relevance_is_patient_specific` | P002 > 0.4 (0.55941); P001 0.0 | |
| personalised ranks | `test_personalization_reorders_across_patients` | pool `(DQ Fisetin Spermidine Metformin Berberine)`: P001 = [DQ, Fisetin, Metformin, Berberine, Spermidine]; P002 = [Metformin, Berberine, DQ, Fisetin, Spermidine]; `order[Metformin] > order[Berberine]` | order |
| personalised score = base + relevance | `test_personalized_score_is_population_base_plus_relevance` | DQ 0.2306046747621094 | approx |
| counterfactual P001 | `test_counterfactual.py` | A (ChronicInflammation): delta -0.121875, c 0.65, Via {DNAmGDF15}; B (CellularSenescence): delta -0.2219296875, Via {DNAmPAI1,DNAmGDF15}, c(B) < c(A) (0.3176 < 0.65) and c(B) < 0.65; CRP lever == A; DQ lever: delta(DQ)/delta(B) = 0.70, c(DQ) < c(B); InsulinResistance and Metformin: delta 0, s 0, Via empty; Elamipretide: nothing; `pos-transmission CRP DNAmGDF15` empty; resolve-lever table (6 cases) | default; `test_counterfactual_is_deterministic` equality |
| decomposition P001 | `test_decompose_*` | components {DNAmPAI1 z1.8, DNAmGDF15 z1.5}; contributions 0.225, 0.1875; attributed 0.4125; residual 1.6-0.4125; DrivenBy(PAI1)={CS}, DrivenBy(GDF15)={CS,CI,MD} | sets |
| risk P001/P002 | `test_risk_prediction.py` | baseline 0.08 / 0.15; point = baseline x 1.07**(z x 4.2); confidence 0.765 = 0.85 x 0.9; CI = point x (1 -/+ (1-c)); b > a > c ~ 0 reductions; confidence b < a | default; P001 point 0.12605177716424967 abs 1e-12 (`test_patient_stack.py`, `test_hallmark_targeting.py`) |
| smoking lever P003 | `test_smoking_lever.py` | delta -0.0819, strength 0.0819, c 0.324; ratio to the driver form 0.39 and c ratio 0.60 x 0.9; ProjectedRisk point 0.23091 (abs 1e-4), reduction 0.00544 (abs 1e-4), c 0.324; `(point 0.23634` prefix; decomposition text `(Component DNAmPACKYRS (z 2.1) (weight 0.125) (contribution 0.2625) (DrivenBy ()))` | as stated; P001/P002 answers to 9 queries **byte-identical with and without lifestyle_evidence.metta** |
| LinAge2 | `test_linage2.py` | cessation delta `-0.95 x cotinine years` (rel 1e-6), `(stv .. 0.765)` = 0.85 x 1.0 x 0.9; Metformin `-0.75 x 0.70 x HbA1c years`; `linage-biomarker-causes CRP` == `(ChronicInflammation CellularSenescence)` EXACT, `HbA1c` == `(DeregulatedNutrientSensing)` EXACT; RDW/LowSerumAlbumin DrivenBy `(ChronicInflammation CellularSenescence)` EXACT; hazard confidence 0.60 x 0.9; GrimAge layer unchanged inside this stack (point 0.12605 abs 1e-5, delta -0.0819 abs 1e-4) | |
| causes that reach a reported CHD | `test_patient_stack.py::test_the_causes_that_reach_a_reported_chd_are_the_ones_the_notes_name` (6 params) | `diagnose-patient` with only a reported CHD and ONE cause is non-empty iff the cause is in `CHD_REACHING_CAUSES` = {CellularSenescence, InsulinResistance, DeregulatedNutrientSensing} (`core/patient_builder.py:396`); so **ChronicInflammation, MitochondrialDysfunction and SmokingPackYears -> CHD must stay empty** | empty / non-empty |
| default causes reach every marker with an Effect into it | `test_patient_stack.py::test_the_default_causes_reach_every_marker_the_kb_has_an_edge_into` (9 params) | for each of CRP, DNAmGDF15, DNAmPACKYRS, DNAmPAI1, FastingGlucose, HbA1c, LowSerumAlbumin, RDW, Triglycerides, `(diagnose-patient &self Caller_Me)` over the 6 default causes (`patient_profile.metta:334-337`) is non-empty at z = 2.0 | non-empty |
| markers with an Effect into them | `test_caller_patients.py::test_the_markers_the_kb_has_edges_into_are_read_off_the_kb` | `kb_effect_markers()` == those same 9 symbols; read by regex off the `.metta` TEXT (`core/patient_builder.py:305-306`, `:309-327`) | set equality |
| supplements | `test_supplement_recommendation.py` | Omega3 Tier1 targets {CRP, DNAmGDF15}; NMN score > Fisetin score; Fisetin targets {CRP, DNAmGDF15, DNAmPAI1}; Resveratrol NotRecommended (ITP_Negative) with relevance > 0; Berberine absent for P001, Tier1 for P002 with targets {FastingGlucose, HbA1c}; score = relevance x safety (0.8 for Berberine) | sets/order/approx |

### 4.3 Exact-string goldens (the hard pins)

* `tests/fixtures/patient_stack_builtin_golden.json`, asserted by `tests/test_patient_stack.py::test_a_built_in_patient_is_answered_to_the_digit_as_it_was_recorded`: **32 cases** (22 non-empty), `out["atoms"] == case["atoms"]` -
  Patient001-003 x diagnosis, observations, marker set, personalised ranking, supplement plan, risk, plus supplement-for-patient. These contain `0.33000000000000007`, `0.28600000000000003`, `0.24989999999999998`, `0.17017138125000003` and tuple orders such as `(SupportedBy (CRP DNAmGDF15 DNAmPAI1))`.
  **A replacement chainer can pass them only if it reproduces the legacy float association and consumer-side result order; otherwise the file must be regenerated, which LAUNCH.md forbids under `PLN_CHAINER=legacy`** (new numbers get their own pins under the new chainer).
* `tests/test_patient_stack.py::test_where_the_full_stack_answers_the_patient_stack_answers_identically` and `tests/test_generic_stack.py` (`ANSWERED`, incl. `infer CS CHD` and `explain DQ CHD`): same atoms from two stack variants.
* `tests/test_smoking_lever.py::test_the_existing_patients_answers_are_byte_identical`: 9 queries x 2 patients, with vs without a layer that only adds `Effect` edges.

---------------------------------------------------------------------------------------------------------------------------

## 5. What MUST hold under any replacement

1. **No false chain.** The set of ordered pairs that return a result must equal `closure.json` exactly: 100 pairs on the patient stack, 86 on the LinAge2 stack (and `Lifespan->Mortality` plus the row lifts on the DrugAge side). Every other pair returns nothing: `DQ->AllCauseMortality`, `SmokingCessation->CHD`, `Rapamycin->CHD`, `Elamipretide->*`, `CRP->DNAmGDF15`, `*->AgeAccelGrim`, `Metformin->CellularSenescence`... Consequences: no inversion (A->B gives B->A), no induction (shared cause gives sibling link: `CRP<->DNAmGDF15`), no abduction (effect gives cause); each would create pairs outside the closure and silently re-rank `diagnose`, `patient-relevance` and the counterfactual routing.
2. **Sign.** Net sign = product of hop signs; protective = `Neg`, harmful = `Pos`; record sign from HR; DrugAge: extending lifespan is `Neg` at `Mortality`, shortening is `Pos` (the "sign trap"); `Neg*Neg=Pos`. `explains-obs` and the counterfactual routing keep ONLY `Pos` derivations, so a Neg chain must never come back labelled Pos.
3. **Confidence decays with every extra edge**: a chain's confidence is its first link's c times the suffix chain's c times 0.9, so it is always below the suffix chain's (pinned: `c < 0.65` for 3 edges; D+Q 0.170 > Fisetin 0.131 > Spermidine 0.0766 for 4/4/5 edges; scenario B's c < scenario A's). Strength is the plain product (no base-rate term), so it never exceeds the weakest link.
4. **Orderings** of section 4.2 (rankings, coverage-then-mass, B > A > C, NMN > Fisetin, tiers, `Metformin` 3rd for Patient001) and the stability/tie rule of `insert-scored` (`>=`).
5. **Determinism**: same inputs, byte-identical outputs (`test_counterfactual_is_deterministic`, `test_prediction_is_deterministic`, `test_recommendation_is_deterministic`, the goldens); the result must not depend on candidate listing order (`test_rank_interventions_is_order_independent`) nor on wall-clock/process state.
6. **One derivation per connected pair on the shipped graph**, because coverage, mass, `patient-relevance` and `pos-transmission` count or pick derivations. If the replacement revises/merges multiple paths it must still return exactly one result for these pairs; if it ever returns two, the consumers double-count (section 1.4).
7. **Output shapes** exactly `(signed Pos|Neg (stv s c))` and `(Chain (nodes..) (signed ..))` with reduced decimal numbers (section 1.1); `explain`'s signed tv == `infer`'s on every pair.
8. **Empty, not error**, for unconnected, unknown, self pairs; no exception to the HTTP layer.
9. **Terminate on cycles** (legacy does not): required by LAUNCH.md for the new chainer; with a visited set the shipped acyclic graph's answers are unchanged.
10. **Caller-injected facts work**: `/metta/run` `extra_atoms` may add `(Effect ..)` atoms (literal or `(evidence-confidence T)` confidences) and a per-request patient; the chainer must see them without a precompile step, and must not depend on state surviving between requests (each request loads its own space/process).
11. **Per-stack identity**: the same `infer` answers (non-DrugAge) in patient, generic, LinAge2, NHANES-patient, DrugAge, and the minimal `system_types, logical_predicates, epistemic_calibration, pln_deduction` stack the `confidence_threshold` test uses (it has no `evidence_calibration`: record lifts must fail soft there).
12. **Evidence confidence stays a lookup**: an `Effect`'s confidence is `(evidence-confidence <Tier>)` against `epistemic_calibration.metta` (the single authority; the write gate refuses literals); `link-effect` and the two hand-composed lever sites read the stored `tv` straight off the atom.
13. **Names stay defined** where the validator, inventory and prompt expect them: `infer`, `explain` and every symbol in dependencies.md section 1 appear as `(= (name ..) ..)` text in a loaded `.metta` file.

---------------------------------------------------------------------------------------------------------------------------

## 6. Things I could not resolve or that the design must decide (stated, not guessed)

* Whether any external client relies on `explain` for DrugAge compounds: legacy returns nothing, tests assert nothing about it (`unknown`).
* The order of multiple derivations from one call is an interpreter artefact (section 1.2); PeTTa may order differently. No shipped pin depends on it, but the DrugAge multi-row output order and the `pos-transmission` "first" pick would.
* `test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move` loads the FULL shared stack; `tests/test_patient_stack.py` says the patient-form ranking aborts there on hyperon 0.2.10. I did not run the test suite (hyperon aborts are process-fatal); I only ran isolated probes.
* Whether the replacement must keep the explicit `$space` parameter: all 7 call sites pass it; dropping it changes every consumer signature.
