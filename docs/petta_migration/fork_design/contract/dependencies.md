# Seams: exactly which parts of `pln_deduction.metta` other layers extend, read, or text-parse

Scope: repo `/Users/kolya/PycharmProjects/pln_longevity_petta` (HEAD 8a2a6a0; code identical to linage2 @ 47130aa). All line numbers are relative to the repo root.
Probes mentioned here are in this folder (`probe_*.py`); they were run on real hyperon 0.2.10. Companion: `contract.md` (interface and pins), `links.json` (the links).

`pln_deduction.metta` is 186 lines / 9,808 bytes. It defines, in file order:

| symbol | lines | role |
|---|---|---|
| `(: Effect (-> Atom Atom Atom Atom Atom))`, `EffectSign`, `Pos`, `Neg` | 57, 60-62 | the link type and sign constants |
| `sign-product` (4 equations) | 82-85 | sign algebra |
| `chain-discount` = 0.9 | 87 | per-combination confidence discount |
| `combine-tv` | 89-92 | `(stv (* sab sbc) (* (* cab cbc) (chain-discount)))` |
| `combine-hop` | 94-95 | `(signed (sign-product ..) (combine-tv ..))` |
| `hr->sign` | 111 | HR record sign |
| `link-effect` (2 equations) | 113-117 | the one-hop lift: Effect atom, or HR record through `calibrate-tv` |
| `infer` (2 equations) | 132-135 | base + step |
| `explain` (3 equations), `prepend-node` | 144-155 | same recursion with the node path |
| smoke-test comments | 157-186 | documented expected values (`(signed Pos (stv 0.357 0.29089125))` etc.) |

---------------------------------------------------------------------------------------------------------------------------

## 1. Who reads which symbol (outside `pln_deduction.metta`)

| symbol | outside users |
|---|---|
| `infer` | 7 MeTTa call sites: `patient_profile.metta:355` (`relevance-on`), `:390` (`score-candidate-patient`); `pln_abductive_diagnosis.metta:44` (`explains-obs`); `pln_intervention_ranking.metta:62` (`score-candidate`); `pln_counterfactual.metta:108` (`pos-transmission`), `:126` (`component-cause`); `pln_linage2.metta:123` (`linage-biomarker-causes`). Plus LLM-translated programs (`prompts/few_shot_examples.json`), `/metta/run` callers, 17 test modules that mention `infer`/`explain`/`Chain`, `scripts/test_pln_api.py:318`. |
| `explain` | no MeTTa file. `scripts/counterfactual_chat_demo.py:111`, tests (`test_pln_deduction.py`, `test_generic_stack.py::ANSWERED`, `test_ontology_expansion.py`). |
| `chain-discount` | `pln_counterfactual.metta:284`, `pln_linage2.metta:314` (hand-composed lever edge); **text-parsed** by `pln_chat/ontology/drugage_scoring.py:56,106` (`_RE_CHAIN`, falls back to 0.9 if the line is missing); pinned by `tests/test_drugage_ranking_contract.py:213` (substring `(= (chain-discount) 0.9)` in the file) and `tests/test_drugage_discovery.py:97` (`knobs.chain_discount == 0.9`). |
| `combine-hop` | `drugage_calibration.metta:243`. |
| `combine-tv`, `sign-product`, `link-effect`, `hr->sign`, `prepend-node` | internal only (tests call `sign-product` and `link-effect` directly in `test_pln_deduction.py`). |
| `Effect` (type + atom shape) | data in `mechanistic_bridges.metta` (19), `lifestyle_evidence.metta` (2), `supplement_evidence.metta` (3), `linage2_core.metta` (1), `drugage_calibration.metta:226` (1 fixed); **direct `match` reads that bypass `infer`**: `pln_counterfactual.metta:196-198` (`resolve-lever`), `:278`; `pln_linage2.metta:276-278`, `:308`; Python/text: see section 5. |
| `Pos` / `Neg` / `signed` / `stv` / `Chain` | pattern-matched by `(let (signed Pos $tv) (infer ..) ..)` in abductive/counterfactual/linage2; `rank-score (signed Neg (stv $s $c))` `pln_intervention_ranking.metta:44-45`; produced by `lifespan-sign` (`drugage_calibration.metta:193`) and `senescence-sign` (`cellage_calibration.metta:113-114`); parsed by Python regexes (`pln_runner.py:415-420`, `:588-591`, `core/linage2_router.py:325-328`). |

---------------------------------------------------------------------------------------------------------------------------

## 2. The extension seams (the answer to "what do other layers extend?")

### 2.1 DrugAge extends `infer` itself (the only layer that does)

`drugage_calibration.metta:240-243`:

```metta
(= (infer $s $a $c)
   (match $s (UsesIntervention $r $a)
      (let (Effect $x Lifespan $sgn $tv) (drugage-effect $s $r)
           (combine-hop $sgn $tv (infer $s Lifespan $c)))))
```

* Mechanism: **a second definition of the same function head in another file**. hyperon unions the equations, so `infer` is open-ended. It is not a hook or a parameter; `drugage_calibration.metta` must simply be loaded in the same space.
  A replacement that implements `infer` as one closed function (a derivation closure, a grounded op) has no equivalent seam: it must re-create it, for example by emitting the lifted DrugAge row as a link the chainer reads.
* It reads `combine-hop` (`pln_deduction.metta:94`) and recurses into `infer` for `Lifespan -> <C>`, which bottoms out in the fixed link `(Effect Lifespan Mortality Neg (stv 1.0 1.0))` at `drugage_calibration.metta:226`. So the reachable outcome is `Mortality` only; `infer Compound Lifespan` is empty (probed).
* Link produced on the fly, never stored: `drugage-effect` (`:208-213`) = `(Effect <compound> Lifespan <sign> (stv <strength> <confidence>))`. Strength `|pct|/(|pct|+20)` (`:139-144`, `lifespan-halfsat` `:59`); confidence `tier x sig-gate` (`:173-176`; ITP row 0.90, vertebrate 0.50, invertebrate 0.35 (and untaxonomised), fungi/protozoa 0.20, gates 1.0/0.6/0.4 at `:83-85`); sign `lifespan-sign` (`:193`).
  Multiplicity: **one derivation per DrugAge row** (section 1.2 of `contract.md`).
* **`explain` is NOT extended** (no equation in `drugage_calibration.metta`): `explain Compound Mortality` is empty while `infer` answers (probed). The only DrugAge path `explain` sees is a materialised `(Effect X Lifespan ..)` atom (the Taurine case).
* Stacks that load it: `DRUGAGE_STACK` (`pln_runner.py:29-40`), the API/app shared stack (`api.py:_INFERENCE_STACK` :185-222, `app.py:117-`) and therefore the **generic stack**, and `tests/test_ontology_expansion.py::KB_FILES`. NOT the patient stack: `drugage_calibration.metta` is in `PATIENT_STACK_EXCLUDED` (`pln_runner.py:139-147`).
  So the same `infer` has 3 equations in the generic/DrugAge stacks and 2 in the patient, LinAge2, NHANES-patient and CellAge stacks.
* Python twin: `ontology/drugage_scoring.py:170-210` duplicates the arithmetic (`confidence = tier * gate * chain_discount`), reads the knobs by regex from `drugage_calibration.metta` (`lifespan-halfsat`, `sig-gate`, `clade-category`), `epistemic_calibration.metta`, `pln_deduction.metta` (`chain-discount`) and `species_taxonomy.metta`. `tests/test_drugage_ranking_contract.py::test_halfsat_and_chain_discount_still_match_the_metta_source` pins the match.
* Entry points that call the stack: `run_drugage_ranking` (`pln_runner.py:861-960`) sends `!(score-candidate &self <C> Mortality)` per compound (strategy `linear`) or `!(rank-interventions &self (..) Mortality)` (`metta_sort`); `pln_intervention_ranking.metta` is loaded in `DRUGAGE_STACK` for exactly this.

### 2.2 CellAge adds links but does NOT extend `infer`

`cellage_calibration.metta:126-132` `cellage-effect` produces `(Effect <gene> CellularSenescence <sign> (stv 0.70 0.35))` on demand (`cellage-curated-strength` `:90`, tier `InVitro` `:96`, sign `senescence-sign` `:113-114`).
Nothing materialises it and no `infer` equation reads it: with the real fixture rows (`tests/fixtures/cellage_real_rows.metta`) `(infer &self Gene_AKT1 CellularSenescence)` and `explain` are empty while `(cellage-effect &self CellAgeRow_16)` returns the link (probed). Header `:60-68` says why: slices are capped at 100-125 rows (hyperon abort), and the layer deliberately does not bridge `CellularSenescence -> Mortality` (`:105-112`).
The file's only contact with the chainer is the load-order comment and the reuse of the `Effect` / `Pos` / `Neg` vocabulary. Consumers: `run_cellage_effects` (`pln_runner.py:1044-1106`) and `GET /genes/{key}?infer=true` (`api.py:2760-2870`), which call `cellage-gene-effects`, not `infer`. LAUNCH.md already puts "GenAge/CellAge links as PLN sentences" out of scope.

### 2.3 LinAge2 adds data and calls `infer`; it does not extend it

* Data: `linage2_core.metta:322` `(Effect SmokingCessation CurrentTobaccoExposure Neg (stv 0.95 (evidence-confidence MultipleHumanTrials)))`; `linage2_fong2025_evidence.metta:43-50` the HR record `Fong2025_LinAgeAccel_AllCauseMortality` (HR 1.093, category `Epidemiological`) which `link-effect` eq.2 lifts to `(signed Pos (stv 0.5 0.6))`; `(MeasuresBiomarker ..)` facts (`linage2_core.metta:250-276`) that map a model input to a KB biomarker.
* Calls into the chainer: `infer` at `pln_linage2.metta:123`; `pos-transmission` (itself an `infer` consumer) at `:289`, `:298-319`.
* Reads of the chainer's data bypassing `infer`: `(match $space (Effect $lever $d $sg $tv) $d)` etc. in `linage-lever-drivers` (`:276-278`) and the lever edge at `:308`.
* Hand-composition that duplicates `combine-tv`: `:311-314`; identity transmission `(stv 1.0 1.0)` `:288`; `calibrate-tv` for the hazard confidence `:203-208`; `risk-conf-discount` `:208`.
* The AllCause hazard path (`linage-hazard`, `linage-risk`) never calls `infer`; it reads the record straight (`:195-198`, `:204-205`).

### 2.4 The calibration files feed `link-effect`

* `evidence_calibration.metta`: `calibrate-tv` (`:77-79`) is called by `link-effect` eq.2 (`pln_deduction.metta:117`) and by `explain` eq.2 (`:148`); also directly by `pln_risk_prediction.metta:274` and `pln_linage2.metta:205`. `effect->strength` buckets `:62-64`; `HasEvidenceCategory` for the four Lu 2019 records `:35-38`.
  The tiers it resolves come from `epistemic_calibration.metta:28-42` (`evidence-confidence`, 11 tiers) - the same lookup the unevaluated Effect confidences call at combine time.
* The minimal test stack `system_types, logical_predicates, epistemic_calibration, pln_deduction` (no `evidence_calibration`) works for Effect-only graphs because `link-effect` eq.2 matches nothing; it must keep failing soft.

### 2.5 Layers that must NOT add `Effect` edges (negative seams, tested)

`hallmark_targeting.metta` (`tests/test_hallmark_targeting.py::test_the_layer_adds_no_effect_edges`, `_facts("Effect") == []`) and `human_evidence.metta` (`tests/test_human_evidence.py::test_the_layer_adds_no_effect_edges`): targeting and human-evidence claims are not causal edges. A chainer that reads `TargetsHallmark` / `HumanEvidence` as links would put Rapamycin on a chain to CHD (`test_rapamycin_is_still_unreachable_by_inference`).
`lifestyle_evidence.metta` header (`:44-50`) refuses a `SmokingPackYears -> CoronaryHeartDisease` shortcut: `infer SmokingCessation CoronaryHeartDisease` must stay empty (`test_smoking_lever.py`).

---------------------------------------------------------------------------------------------------------------------------

## 3. Where the chainer is loaded (every space needs it)

| space | defined at | files that matter |
|---|---|---|
| shared API/app stack | `api.py:185-222` and `app.py:117-` (`_INFERENCE_STACK`, mirrored by hand, `test_smoking_lever.py::test_both_inference_stacks_load_the_layer_and_still_agree`) | 26 names, 25 on disk (`cellage_metadata.metta` is a generated file, absent from the repo root) |
| **patient stack** | `pln_runner.py:150-152` `patient_stack()` = shared minus `PATIENT_STACK_EXCLUDED` (`:139-147`) | 18 files; `infer` has 2 equations |
| **generic stack** | `pln_runner.py:193-195` = shared minus `GENERIC_STACK_EXCLUDED` (`:181-186`: `patient_profile`, `pln_counterfactual`, `pln_risk_prediction`, `pln_supplement_recommendation`) | 21 files; 3-equation `infer`; `tests/test_generic_stack.py::test_nothing_outside_the_patient_layers_uses_their_definitions` forbids any generic-stack file using a patient-layer definition (a chainer file loaded here cannot call `patient-z` etc.) |
| NHANES patient | `pln_runner.py:82-101` | 17 files; 19 Effect atoms (no lifestyle/supplement) |
| LinAge2 | `pln_runner.py:281-300` | 17 files; 22 Effect + 5 records |
| DrugAge | `pln_runner.py:29-40` | 8 files; no mechanistic_bridges (to avoid compound/bridge name collisions, `:24-28`) |
| CellAge | `pln_runner.py:53-63` | 6 files |
| human-evidence | `pln_runner.py:231-239` | 4 files; no chainer |
| tests with their own `KB_FILES` | `test_pln_deduction`, `test_patient_grounding`, `test_counterfactual`, `test_risk_prediction`, `test_supplement_recommendation`, `test_smoking_lever`, `test_nhanes_integration`, `test_human_evidence`, `test_ontology_expansion`, `test_drugage_ranking_contract` (inline minimal stack); `test_drugage_calibration` uses `DRUGAGE_STACK`; demos `scripts/{risk_prediction,counterfactual,supplement}_chat_demo.py` | each lists `pln_deduction.metta` by name |

Every stack concatenates the file TEXTS into one `MeTTa()` (`pln_runner.py:_hyperon_run` :734-831): one fresh interpreter per `run_query`; caller `extra_atoms` appended after the files. Files larger than `PLN_MAX_KB_FILE_BYTES` (60,000 bytes, `config.py:78`) are silently dropped from the runtime stack by `_runtime_kb_paths` (`api.py:269-283`); a vendored chainer file above that size would vanish without an error.

---------------------------------------------------------------------------------------------------------------------------

## 4. Hyperon-specific tricks encoded in the chainer files (do not assume they port)

* `(: Effect (-> Atom Atom Atom Atom Atom))` (`pln_deduction.metta:57`) exists to stop hyperon reducing the `(evidence-confidence ..)` call inside the stored `tv` (also `drugage_calibration.metta:202-207`, `cellage_calibration.metta:123-125`: "Atom-typed args SUPPRESS reduction" - which is why the lifts force numbers through `let*` first).
* Explicit `$s` space parameter everywhere (cross-module `&self` trap).
* Non-determinism = several equations / several `match` hits; `collapse` + `car-atom` is the only "pick one" (`pos-transmission`). No cut.
* `(superpose ())` as "no result" (`pln_abductive_diagnosis.metta:105`, `pln_counterfactual.metta:98`, `:109`, `:202`).
* Distinct-key-atom budget of hyperon 0.2.10 (about 1,024 key atoms per space; each new function head or constant is one): this is why `pln_counterfactual.metta:221-240` says rules cannot be added there, why the lever composition is inlined (`:266-273`), and why `tests/test_kb_head_symbol_budget.py` / `test_generic_stack.py` / `test_patient_stack.py` measure margins (patient stack: +86 head symbols answer, +87 abort). On PeTTa this budget disappears; on hyperon (the default and the regression target) a replacement MeTTa chainer pays for every symbol it adds.

---------------------------------------------------------------------------------------------------------------------------

## 5. Python and text coupling to the chainer's files and output

| coupling | where | what breaks if the replacement changes it |
|---|---|---|
| `(= (chain-discount) 0.9)` text | `ontology/drugage_scoring.py:56,106`; `tests/test_drugage_ranking_contract.py:213`; `tests/test_drugage_discovery.py:97` | silent fall-back to a default 0.9 (so a changed discount would not reach the Python DrugAge scorer) |
| `(Effect <from> <to> Pos\|Neg` at the START of a line, in EVERY root `*.metta` | `core/patient_builder.py:305-327` `kb_effect_markers()` (cached) | the set of markers a caller patient can carry that matter shrinks to nothing; pinned to the 9 symbols CRP, DNAmGDF15, DNAmPACKYRS, DNAmPAI1, FastingGlucose, HbA1c, LowSerumAlbumin, RDW, Triglycerides (`test_caller_patients.py::test_the_markers_the_kb_has_edges_into_are_read_off_the_kb`) and used as the witness rule in the patient builder/notes |
| `CHD_REACHING_CAUSES` = {InsulinResistance, DeregulatedNutrientSensing, CellularSenescence} | `core/patient_builder.py:396`, `core/patient_context.py:15,169,211` | a hard-coded copy of "which default causes have a Pos chain to CHD", probed against the engine in `test_patient_stack.py` (non-empty iff in the set) |
| Effect write shape | `ontology/expansion_schema.py` (predicate `Effect` in the closed list `:194-196`; canonical example `:301-310`; gate: literal two-float `stv` on an Effect is refused, confidence must be `(evidence-confidence <Tier>)`, tier must exist `:470-520`), `ontology/expander.py:233-289` (instructions given to the extraction LLM: "Pos = RAISES; Neg = LOWERS; a life-extending compound is (Effect X Lifespan Pos ..)") | new papers stop landing in the schema the rules read |
| Output regexes | `core/pln_runner.py:415-420` (`(scored X score (signed Pos\|Neg (stv a b)))`), `:588-597` (`_stv_from_atom`, first `(stv a b)`), `_filter_tuple_entries`/`_apply_threshold` `:503-540` (per-entry confidence filter), `core/linage2_router.py:325-328` | `confidence_threshold` and the DrugAge/LinAge2 JSON stop working |
| Validator / registry | `ontology/loader.py:24-30` (`_RULE_DEF = ^\(=\s+\((\S+)\s+\$` registers a function only if its FIRST argument is a `$` variable; `_CONST_DEF` for zero/constant-argument heads), `core/metta_validator.py` | `infer`/`explain`/`chain-discount` unknown to the validator, so valid programs get 422 `unknown symbol` |
| Translator prompt | the selected `.metta` files are pasted verbatim into the system prompt, files over `PLN_PROMPT_FILE_MAX_BYTES` = 25,000 bytes are replaced by a schema card (`config.py:51`, `api.py:_build_context`); `pln_deduction.metta` is 9.8 KB | a larger chainer file hides `infer`'s documented contract from the LLM; the budget `PLN_MAX_PROMPT_TOKENS` 200,000 (`tests/test_prompt_size.py`) |
| Injection from the API | `/metta/run` and `/query` `extra_atoms` (`api.py:574-590`, `:1320-1326`): facts allowed (so `Effect` facts are), `(= ..)` definitions refused unless `allow_definitions` | a chainer that precompiles or caches the KB cannot honour per-request edges |
| Stack routing by program text | `core/pln_runner.py:214-224` `needs_patient_layers`, `api._generic_kb` (`api.py:3189-3200`) | a program that names a patient goes to the patient stack (no DrugAge equation), others to the generic stack; the chainer must behave identically in both for non-DrugAge pairs |
| Process model | `_hyperon_run` builds a new `MeTTa()` per call; `core/executor.py` runs it in a worker process with a deadline (`PLNExecutionTimeout`); cycles are bounded only by that deadline | nothing in the chainer may rely on warm state, and a hang costs a worker until the deadline |

---------------------------------------------------------------------------------------------------------------------------

## 6. Minimum a drop-in `infer`/`explain` must provide, in seam order

1. Same names and arity: `(infer $s $a $c)`, `(explain $s $a $c)` visible as `(= (infer $s ..) ..)` text; keep `chain-discount` readable at its current text form until the Python regexes are re-pointed.
2. The lifted-link vocabulary: `Effect` atoms (as written by 4 data files, the write gate and the LLM extractor) plus the record lift (`HasPredictor/HasOutcome/EffectValue/HasEvidenceCategory` through `calibrate-tv`), with sign from HR and the bucket strengths of `evidence_calibration.metta:62-64`.
3. A DrugAge entry: either keep the third `infer` equation working against the new engine, or lift each row into the chainer's input (one evidence item per row; the contract today is one derivation per row and no merge).
4. Preserve `Effect` atoms in the space in their current text form (the two hand-composed lever sites, `resolve-lever`, `linage-lever-drivers` and `kb_effect_markers` read them directly).
5. Keep `explain`'s `(Chain (nodes) (signed ..))` shape and its agreement with `infer`.
6. Keep `pos-transmission`'s "first Pos derivation" meaning deterministic (it is `car-atom` of a collapse) or make the chainer return one derivation per pair.
7. Provide a cycle guard (legacy has none).
