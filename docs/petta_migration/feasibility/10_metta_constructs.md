# MeTTa construct inventory — pln_longevity (branch linage2) for a runtime port

Read-only audit of every language feature the `.metta` knowledge base and the LLM-emitted
queries rely on under hyperon-experimental 0.2.10 (Python bindings). Nothing was executed
against MeTTa. Counts come from `scan_metta.py` (same directory; comments stripped, strings
kept) over all 37 `.metta` files, cross-checked with grep. "all / root" = count over all 37
files / over the 32 repo-root files only. Evidence is `file:line` relative to
`/Users/kolya/PycharmProjects/pln_longevity`.

Companion data: `scan_report.json` (per-file head counts, every example list).

---

## 1. Files

37 `.metta` files, 10,486 lines in total; the 32 in the repo root hold 9,364 lines.

| file | lines | role / which runtime stack loads it |
|---|---|---|
| system_types.metta | 133 | type declarations; every stack |
| logical_predicates.metta | 188 | predicate arrow types; every stack |
| epistemic_calibration.metta | 65 | `evidence-confidence` table; every stack |
| evidence_calibration.metta | 103 | `calibrate-tv`; shared, DrugAge, CellAge, LinAge2 stacks |
| pln_deduction.metta | 186 | `infer` / `explain`; shared, DrugAge, CellAge, LinAge2 |
| pln_intervention_ranking.metta | 130 | ranking + insertion sort; shared, DrugAge |
| pln_abductive_diagnosis.metta | 172 | `diagnose`; shared |
| patient_profile.metta | 421 | patient grounding; patient stack, LinAge2 stack (not generic stack) |
| pln_counterfactual.metta | 415 | do-operator; patient + LinAge2 stacks |
| pln_risk_prediction.metta | 474 | absolute risk; patient + LinAge2 stacks |
| pln_supplement_recommendation.metta | 280 | tiers; patient stack |
| supplement_evidence.metta | 158 | Layer-4 facts; shared |
| mechanistic_bridges.metta | 235 | curated `Effect` facts; shared, LinAge2 |
| hallmark_targeting.metta | 262 | `TargetsHallmark` facts + 2 accessors; shared (not patient stack) |
| human_evidence.metta | 342 | flat `HumanEvidence` records + accessors; shared, human-evidence stack |
| lifestyle_evidence.metta | 300 | smoking lever facts, Patient003; shared, LinAge2 |
| grim_age_core.metta | 167 | GrimAge ontology facts; shared, LinAge2 |
| grim_age_lu2019_evidence.metta | 64 | `EmpiricalAssociation` records; shared, LinAge2 |
| hallmarks_core.metta | 54 | 12 hallmarks; shared, LinAge2 |
| hallmarks_lopezotin2023_anchors.metta | 123 | `HallmarkComponent` facts; shared (not patient stack) |
| hallmarks_lopezotin2023_intervention_evidence.metta | 157 | review records; shared |
| measurement_types.metta | 54 | `Measure` + unit symbols; shared (not patient stack) |
| species_taxonomy.metta | 78 | `Inheritance` taxa; shared, DrugAge |
| drugage_entries.metta | 27 | 20 `Inheritance` facts; shared (not patient stack) |
| drugage_calibration.metta | 261 | DrugAge lift + an extra `infer` equation; shared, DrugAge |
| cellage_calibration.metta | 183 | CellAge lift; CELLAGE_STACK only |
| nhanes_baseline.metta | 408 | baseline record types + lookup rules; NHANES + LinAge2 stacks |
| nhanes_reference.metta | 315 | reference record types + `standardize-z`; NHANES stack |
| linage2_core.metta | 323 | LinAge2 ontology, 59 `ModelInput`; LinAge2 stack only |
| linage2_fong2025_evidence.metta | 71 | one `EmpiricalAssociation`; LinAge2 stack only |
| pln_linage2.metta | 450 | LinAge2 rules; LinAge2 stack only |
| drugage_etl_short.metta | 2,765 | 201 DrugAge rows; NEVER loaded whole (107 KB > `PLN_MAX_KB_FILE_BYTES` 60 KB, pln_chat/config.py:78); it is `SAMPLE_DRUGAGE`, sliced per query by pln_chat/ontology/drugage_selector.py:36 |
| tests/fixtures/cellage_real_rows.metta | 363 | 25 CellAge rows; `SAMPLE_CELLAGE`, sliced per query (pln_chat/ontology/cellage_selector.py:41) |
| tests/fixtures/drugage_real_rows.metta | 137 | 8 real DrugAge rows; test fixture |
| docs/kb_quick_wins/probes/mb_all.metta | 217 | probe copy of mechanistic_bridges + 5 candidate edges; not loaded at runtime (docs/kb_quick_wins/REPORT.md:75) |
| docs/kb_quick_wins/probes/pp_ua.metta | 386 | probe copy of patient_profile (older variant); not loaded |
| pln_chat/logs/last_query.metta | 19 | runtime log of the last program (`;; loaded:` lines + the `!` program); not loaded |

Stacks (all lists in pln_chat/core/pln_runner.py): `DRUGAGE_STACK` :29-40 (8 files + a row
slice as `extra_atoms`), `CELLAGE_STACK` :55-60 (7 files + slice), `NHANES_PATIENT_STACK`
:84-101 (17 files), `LINAGE2_PATIENT_STACK` :287-305 (17 files + optional
build/nhanes_mortality_baseline.metta), the shared `_INFERENCE_STACK` (pln_chat/api.py:185-227,
28 names; `cellage_metadata.metta` :192 is not on disk) from which `generic_stack` drops the 4
patient layers (:177-182), `patient_stack` drops 7 files (:139-147) and `human_evidence_stack`
keeps 4 (:218-221). `build/` (ETL output) is absent in this checkout; its shapes are in §9.

---

## 2. How the files are loaded and queries run (the runner contract)

* pln_chat/core/pln_runner.py:860-895 `_hyperon_run`: every KB file's text is CONCATENATED
  (plus caller `extra_atoms`) and passed to `MeTTa().run(kb_text)` with the result ignored —
  i.e. every top-level expression that is not prefixed by `!` is ADDED AS AN ATOM, including
  `(= …)` equations and `(: …)` declarations. Deliberately not `import!` per file (:866-876,
  the cross-module `&self` trap).
* pln_runner.py:838-843 `_normalize_query` prefixes `!` to each top-level expression of the
  query; `metta.run(normalized)` returns `list[list[Atom]]`, one group per `!` expression in
  program order; each atom is rendered with `str(atom)` and parsed by regex (:914-919).
* One `MeTTa()` instance per query, in a forked worker process with SIGKILL deadline
  (pln_chat/core/executor.py:13-41, :170-188): hyperon holds the GIL and aborts the process on
  the trie panic. The abort-containment design is hyperon-specific.
* KB files contain NO top-level `!` evaluations (only pln_chat/logs/last_query.metta:20).
* Programs are split by Python before execution: `(rank-drugage-lifespan (…))` is not a MeTTa
  function at all — a regex (pln_chat/core/drugage_router.py:70) intercepts it and runs
  `(score-candidate &self C Mortality)` per compound in DRUGAGE_STACK (pln_runner.py:775-777);
  `linage-*` forms are routed by regex (pln_chat/core/linage2_router.py:75-88, 131-157) to the
  LinAge2 stack, a mixed program is run as two parts (api.py:3277-3290); human-evidence-only
  programs go to their 4-file stack (pln_runner.py:234-247); anything naming a patient or a
  patient-layer symbol goes to the patient stack (pln_chat/core/patient_context.py:62-82,
  pln_runner.py:201-212).

---

## 3. Construct inventory

Legend for `kind`: special-form, stdlib, grounded, type-decl, space-op, io, hyperon-workaround.

### 3.1 Pattern matching and non-determinism

| construct | kind | count (all/root) | notes and examples |
|---|---|---|---|
| `match` | special-form | 84 / 78 | Every KB `match` is on a VARIABLE space (`$s`, `$space`) threaded from the caller — 84 of 84; `&self` appears in no KB rule (only last_query.metta:20 and every LLM query). pln_deduction.metta:114 `(match $s (Effect $a $c $sgn $tv) (signed $sgn $tv))`; patient_profile.metta:221; pln_linage2.metta:84 |
| `match` with `,` conjunction | special-form | 13 (+1 nested) | pln_deduction.metta:116 `(match $s (, (HasPredictor $r $a) (HasOutcome $r $c) (EffectValue $r $hr)) …)`; nhanes_baseline.metta:303-307 (3 clauses); drugage_calibration.metta:209; nested `(, A (, B C))` cellage_calibration.metta:160-163 |
| `match` whose template is an EVALUATED expression | special-form | 14 | pln_deduction.metta:117 template `(signed (hr->sign $hr) (calibrate-tv $s $r))` is reduced after the match (smoke result :168 shows numbers); cellage_calibration.metta:142 `(cellage-effect $s $row)`; drugage_calibration.metta:241-243 template is a `let`; pln_counterfactual.metta:125-126 `(let $h (match …) (let (signed Pos $tv) (infer …) $h))`. The system prompt claims the template is "NOT evaluated further" (pln_chat/prompts/system_prompt.txt:25-27) — the KB and few-shots rely on the opposite |
| `match` nested inside another match's template | special-form | 1 | pln_risk_prediction.metta:146-150 `(let $sid (match $space (, …) $s) (match $space (SpreadSDYears $sid $v) $v))` |
| `match` on a type-declaration atom | special-form + hyperon-workaround | 1 | pln_risk_prediction.metta:247 `(match $space (: $bid BaselineRiskRecord) $bid)` — relies on `(: X T)` being an ordinary retrievable atom in the space |
| `match` on an equation atom | special-form (queries) | 2 few-shots | pln_chat/prompts/few_shot_examples.json:44,53 `(match &self (= (evidence-confidence $e) $c) (if (> $c 0.7) (pair $e $c) (empty)))` — relies on `(= …)` being retrievable atoms |
| `match` with a bare-variable pattern | special-form | 1 | drugage_calibration.metta:101 `(match $s $atom found)` (existence test of an arbitrary atom) |
| pattern with a VARIABLE IN HEAD position of a sub-expression | special-form | 3 | cellage_calibration.metta:128 `(HasSenescenceEffect $row ($dir CellularSenescence))`, :151, :160 |
| expression-headed pattern | special-form (queries) | 2 few-shots | few_shot_examples.json:22 `((Extends $drug $target) (stv $s $c))`, :33 `((Causes Rapamycin (Inhibits mTORC1)) (stv $s $c))` |
| constant template (`found`, `True`, `CurrentSmoker`) | special-form | 3 | drugage_calibration.metta:101; pln_linage2.metta:101; few_shot_examples.json:134 `(match &self (Inheritance Saccharomyces_cerevisiae Fungi) True)` |
| template building a tuple `($ff $yy)` / `($a $c)` | special-form | 6 | pln_linage2.metta:165,168,183,329; pln_deduction.metta:145,148 `(Chain ($a $c) …)`; supplement_evidence.metta:158 (comment) |
| `superpose` | stdlib | 41 / 37 | `(superpose ())` as the "decline / yield nothing" idiom: 21 (nhanes_baseline.metta:309,315; nhanes_reference.metta:237,245,295-297; pln_counterfactual.metta:98,109,202; patient_profile.metta:294; pln_supplement_recommendation.metta:106,129,171). Over a variable bound to a collapsed tuple: 17 (patient_profile.metta:291 `(superpose $ms)`, :374; human_evidence.metta:298; pln_supplement_recommendation.metta:104,163,170,199,235). Over a literal tuple: 3 (pln_counterfactual.metta:373; pln_risk_prediction.metta:415; pln_linage2.metta:413). The code assumes superpose does NOT reduce its argument (patient_profile.metta:283-288, pln_supplement_recommendation.metta:240-242) |
| `collapse` | stdlib | 59 / 53 | always `(collapse (match …))` or `(collapse (f …))`; empty stream → `()`; patient_profile.metta:221-222, :259; pln_counterfactual.metta:108,134,152,196-198; human_evidence.metta:275,292,303 |
| `let` binding a variable | special-form | 75 | patient_profile.metta:259,271; pln_intervention_ranking.metta:62,109 |
| `let` destructuring a PATTERN (match failure = filter) | special-form | 9 | pln_abductive_diagnosis.metta:44 `(let (signed Pos $tv) (infer $space $h $o) (explains $o $tv))` keeps only Pos derivations; pln_counterfactual.metta:108,126; pln_linage2.metta:123; drugage_calibration.metta:242 `(let (Effect $x Lifespan $sgn $tv) (drugage-effect $s $r) …)`; pln_linage2.metta:165,168,183,329 `(let ($f $y) (match …) …)` |
| `let` over a NON-DETERMINISTIC value (fan-out) | special-form | 30 | pln_intervention_ranking.metta:62 `(let $r (infer …) (scored …))` — one result per derivation; patient_profile.metta:291; pln_counterfactual.metta:125,153,295; pln_linage2.metta:328-330 |
| `let*` | special-form | 77 / 69 | sequential bindings, max 9 (pln_linage2.metta:253-263, :378-389; pln_risk_prediction.metta:384-394); every binding is a plain variable (no pattern bindings); bindings may be `(collapse …)`, `(match …)`, arithmetic, `(car-atom …)`. Used as the universal "force evaluation" device (pln_intervention_ranking.metta:73-78) |
| `if` | special-form | 109 / 98 | always 3-arg; nested chains 4 deep (pln_risk_prediction.metta:105-109; nhanes_baseline.metta:273-277); condition is `==`, `<`/`>`, `and`/`not`, a user predicate returning True/False (drugage_calibration.metta:160,163,169 `(drugage-itp? …)`; patient_profile.metta:203 `(tuple-has …)`; pln_abductive_diagnosis.metta:127 `(hyp-ge …)`) or a bare Bool variable (pln_linage2.metta:106,142,306). nhanes_reference.metta:253-270: NaN/-inf conditions were observed to produce wrong branches, hence input guards |
| `empty` | stdlib (queries only) | 0 in KB; 2 few-shots | few_shot_examples.json:44,53 `(if … (pair $e $c) (empty))` |
| `case`, `chain`, `function`/`return`, `eval`, `quote`, `unify` | special-form | 0 | not used in KB or prompts; `case`, `quote`, `unify` are whitelisted by the validator (pln_chat/core/metta_validator.py:25,35) and `chain`/`eval`/`quote`/`unify`/`case` are "structural heads" the LinAge2 splitter tolerates (linage2_router.py:122-129) |
| top-level `!` | special-form | 0 in KB | only pln_chat/logs/last_query.metta:20; `_normalize_query` adds `!` to every query expression (pln_runner.py:838-843) |

### 3.2 Definitions and equational semantics

| construct | kind | count | notes and examples |
|---|---|---|---|
| `(= (head args…) body)` | special-form | 282 / 263 | 140 distinct function heads; nothing is defined with `=` outside `(= (f …) …)` shape |
| zero-arity constants `(= (k) v)` | special-form | 11 | pln_deduction.metta:87 `(= (chain-discount) 0.9)`; patient_profile.metta:156; pln_risk_prediction.metta:117,164,169; drugage_calibration.metta:59; pln_linage2.metta:73,78; cellage_calibration.metta:90,96 (`(= (cellage-evidence-tier) InVitro)` — a symbol-valued constant). Python reads these by regex from the file text (§8) |
| dispatch on a LITERAL SYMBOL in the head | special-form | 47 | epistemic_calibration.metta:28-42 `(= (evidence-confidence RCT_Human) 1.0)` (11 equations); pln_deduction.metta:82-85 `sign-product`; drugage_calibration.metta:83-85,117-120; pln_supplement_recommendation.metta:56-77; cellage_calibration.metta:113-114 |
| dispatch on a STRING literal in the head | special-form | 1 | evidence_calibration.metta:62 `(= (effect->strength "hazard_ratio" $hr) …)` — string equality by unification |
| NESTED PATTERN in the head (destructuring accessors) | special-form | 29 | pln_intervention_ranking.metta:44-45 `(= (rank-score (signed Neg (stv $s $c))) (* $s $c))`; pln_deduction.metta:89,94,153; pln_abductive_diagnosis.metta:60-62,118; pln_counterfactual.metta:82-83,139,299-301; pln_risk_prediction.metta:343,376-378; pln_supplement_recommendation.metta:132-134; pln_linage2.metta:334-336,368-370 |
| MULTIPLE equations for one head (non-deterministic union) | special-form | 32 heads | `infer` 3 equations split over TWO files (pln_deduction.metta:132,133 + drugage_calibration.metta:240-243 — a later file extends an earlier function); `explain` 3 (pln_deduction.metta:144-151); `link-effect` 2 (:113-117); `interaction-flag` 2 (pln_supplement_recommendation.metta:186-193, comment :183 "hyperon unions the equations"); `rank-score` 2; `evidence-confidence` 11; `evidence-tier` 11; `selection-basis-confidence` 9; `clade-category` 4; `sign-product` 4; `sig-gate` 3; `safety-factor` 3; `senescence-sign` 2 (the other 18 "multi" heads are only duplicated by the probe copy docs/kb_quick_wins/probes/pp_ua.metta) |
| ARITY OVERLOADING of one head | special-form | 1 | `diagnose-patient` 3-arg patient_profile.metta:313 and 2-arg :334-337 |
| redefinition semantics | hyperon-workaround | — | a second `(= (f …) …)` does NOT shadow; hyperon keeps both (api.py:534-541 comment, measured: 256 atoms from one risk query); the API refuses `(= (` in caller atoms (api.py:541-575) |
| unmatched call stays UNREDUCED (not empty) | hyperon-workaround | documented 3× | pln_risk_prediction.metta:230-239, nhanes_baseline.metta:79-85, pinned by tests/test_nhanes_common.py:732-754 (`!(only-chd AllCauseMortality 61)` → `(only-chd AllCauseMortality 61)`, propagates into `(* 2 …)`); design uses a presence `match` before naming an optional layer's rule (pln_risk_prediction.metta:246-252) |
| recursion | special-form | ~20 heads | insertion sorts pln_intervention_ranking.metta:79-91, pln_abductive_diagnosis.metta:123-136, pln_supplement_recommendation.metta:142-154; tuple folds pln_abductive_diagnosis.metta:65-94, pln_counterfactual.metta:140-143,304-328, pln_risk_prediction.metta:344-347, patient_profile.metta:196-207,359-362; graph recursion through the space `infer`/`explain` pln_deduction.metta:132-151 (no cycle guard) |

### 3.3 Type declarations (`:`) and types used

| construct | kind | count | notes and examples |
|---|---|---|---|
| `(: X Type)` new types | type-decl | 72 | system_types.metta:6-133; patient_profile.metta:52-60; human_evidence.metta:122-129; nhanes_baseline.metta:143-215 |
| `(: x CustomType)` instances | type-decl | 152 | epistemic_calibration.metta:10-20 `(: RCT_Human EvidenceCategory)`; grim_age_lu2019_evidence.metta:18 `(: Lu2019_AgeAccelGrim_AllCauseMortality EmpiricalAssociation)`; measurement_types.metta:24-32 (several per line); mechanistic_bridges.metta:39 `(: SASP Phenotype)`; drugage_calibration.metta:48 `(: Mortality Outcome)` |
| `(: X Atom)` | type-decl | 1 | human_evidence.metta:138 `(: NotStated Atom)` |
| arrow types `(: P (-> … Atom))` for predicates | type-decl | 143 of 153 arrows | logical_predicates.metta:11-187; nhanes_baseline.metta:194-236; nhanes_reference.metta:141-188; patient_profile.metta:70-98; 12-ary human_evidence.metta:147 |
| arrow with `Atom` ARGUMENT types | type-decl + hyperon-workaround | 29 | logical_predicates.metta:11 `(: InstanceOf (-> Atom Type Atom))`, :32 `(: Causes (-> Atom Atom Atom))`; pln_deduction.metta:57 `(: Effect (-> Atom Atom Atom Atom Atom))` — relied on to SUPPRESS reduction of arguments (drugage_calibration.metta:202-207, cellage_calibration.metta:123-125) |
| arrow returning `Expression` | type-decl | 8 | logical_predicates.metta:19-25 `(: Increases (-> Atom Expression))` etc.; measurement_types.metta:9 `(: Measure (-> Number Unit Expression))` |
| arrow returning `Number` (typed FUNCTIONS) | type-decl | 2 | epistemic_calibration.metta:25 `(: evidence-confidence (-> EvidenceCategory Number))`, :56 — the only typed functions; all other functions are deliberately untyped (hallmark_targeting.metta:231-236: a `(-> Atom Atom Atom)` signature made the call return the literal `(match ModuleSpace(...) …)`) |
| argument types used in arrows | type-decl | — | Atom 190, Number 33, String 33, Expression 17, Biomarker 20, ReferenceDistribution 19, BaselineRiskRecord 16, PatientProfile 14, Intervention 14, TreatmentProtocol 11, Experiment 10, Publication 7, ClockAccelSpread 7, EmpiricalAssociation 6, Outcome 6, Gene 6, Sex 5, HallmarkInterventionEvidence 5, EvidenceCategory 4, CellSenescenceRecord 4, SmokingStatus 3, HallmarkOfAging 3, SignificanceLabel 3, ReferenceProvenance 3, `Type` 1 (InstanceOf), plus 20 others used once/twice. `Bool` and `Symbol` are NEVER used as declared types |
| duplicate declarations of one symbol | type-decl | 2 pairs + 3 repeats | logical_predicates.metta:109-110 `AgeAtInitiation` typed twice, :111-112 `TreatmentDuration`; `AgeBand`/`ReferenceProvenance`/`AccelDefinition` declared verbatim in both nhanes_baseline.metta:143-164 and nhanes_reference.metta:76-187 ("repeating an identical atom is harmless", nhanes_baseline.metta:136-139) |
| type checking of facts | type-decl | unknown | facts such as `(HasPredictor Lu2019_… AgeAccelGrim)` are typed `(-> EmpiricalAssociation Biomarker Atom)` while `AgeAccelGrim` has only `(Inheritance AgeAccelGrim EpigeneticAgeAcceleration)` (grim_age_core.metta:48); whether hyperon checks facts at load is unknown; the Python layer never depends on type errors |

### 3.4 Grounded arithmetic, comparison, logic

| construct | kind | count (all/root) | notes and examples |
|---|---|---|---|
| `+` | grounded | 19 / 17 | patient_profile.metta:362,383; pln_linage2.metta:152 (inside foldl-atom) |
| `-` (binary only) | grounded | 18 / 18 | negation written `(- 0.0 $pct)` drugage_calibration.metta:141; nhanes_reference.metta:288 |
| `*` | grounded | 37 / 36 | `(* -1.0 $t)` patient_profile.metta:164; pln_deduction.metta:90-91 operates on `$cab` bound to an UNEVALUATED `(evidence-confidence T)` read from a fact (see §4.1) |
| `/` | grounded | 6 / 6 | drugage_calibration.metta:144; nhanes_reference.metta:289,294; pln_counterfactual.metta:322; pln_linage2.metta:346 |
| `pow-math` | grounded | 3 | pln_risk_prediction.metta:284 `(pow-math $hr $yrs)`; pln_linage2.metta:215,244 |
| `log-math` | grounded | 1 | nhanes_reference.metta:292 `(log-math 10 $raw)` = log10; `-inf`/`NaN` results and the `-inf` → `Low` trap pinned in tests/test_nhanes_common.py:757-773 and nhanes_reference.metta:253-270 (`ln-math`, `log10-math`, `exp-math` noted as absent in hyperon) |
| `%`, `sqrt-math`, `abs-math`, `exp`, `min`, `max`, `round`, `floor` | grounded | 0 | abs via `(if (< $pct 0.0) (- 0.0 $pct) $pct)` drugage_calibration.metta:141; max/clamp via `if` (pln_abductive_diagnosis.metta:79; pln_risk_prediction.metta:310; patient_profile.metta:353) |
| `==` | grounded | 64 / 58 | 51 against the empty tuple `(== $xs ())`; 9 against symbols (`(== $sex Male)` pln_risk_prediction.metta:106, `(== $scale Identity)` nhanes_reference.metta:287, `(== $outcome CoronaryHeartDisease)` pln_risk_prediction.metta:241, `(== $prov Measured)` pln_linage2.metta:138); 1 on a call result `(== (rec-tier $r) $tier)` pln_supplement_recommendation.metta:171; element equality `(== $h $y)` patient_profile.metta:199; documented as NOT reducing its arguments (pln_intervention_ranking.metta:73-78, nhanes_baseline.metta:310-312) |
| `>` / `<` / `>=` | grounded | 25 / 15 / 3 | `(> $hr 1.5)` evidence_calibration.metta:63; `(< $age 50)` on an INTEGER age pln_risk_prediction.metta:106, nhanes_baseline.metta:274; `(>= (score-of $x) (score-of $h))` pln_intervention_ranking.metta:83 (call args inside a comparison) |
| `<=`, `!=` | grounded | 0 | validator whitelists them |
| `and` / `not` / `or` / `xor` | grounded | 3 / 1 / 0 / 0 | pln_linage2.metta:105 `(not $nosm)`, :140-141,305 `(and $meas $pos)` |
| `True` / `False` literals | grounded | ~12 | drugage_calibration.metta:102; patient_profile.metta:197,199; pln_abductive_diagnosis.metta:119-120; pln_linage2.metta:107 |
| numbers | grounded | 779 floats, 78 ints, 95 negative floats, 4 scientific | ints: `(PatientAge Patient001 58)` patient_profile.metta:113, `(n 78241)` human_evidence.metta:199, `(EntrezID Gene_AKT1 207)`, `(+ 1 $r)` pln_abductive_diagnosis.metta:67; floats `0.70`, `-1.5`, `20.0`; scientific `2.0e-75`, `6.2e-24`, `3.6e-12`, `4.9e-9` grim_age_lu2019_evidence.metta:24,37,50,63 (nhanes_common.py:985-991 says some hyperon versions read `1e-05` as a SYMBOL, so the ETL writes fixed notation). Mixed int/float arithmetic and comparison are relied on (age int vs 50 int; Python may render a patient z as `2` via `:.6g`, pln_chat/core/patient_builder.py:1023, then `(> $z 1.0)` and `(* 0.125 $z)`) |

### 3.5 Tuple / list operations

| construct | kind | count | notes and examples |
|---|---|---|---|
| `car-atom` | grounded | 39 / 34 | patient_profile.metta:198,262; pln_counterfactual.metta:109,203,205; nhanes_baseline.metta:314 |
| `cdr-atom` | grounded | 26 / 22 | patient_profile.metta:198,202; nhanes_baseline.metta:312 `(let $rest (cdr-atom $cells) (if (== $rest ()) …))` |
| `cons-atom` | grounded | 17 / 15 | patient_profile.metta:203,207; `(cons-atom $x ())` pln_intervention_ranking.metta:81; inside a constructor without `let`: pln_deduction.metta:154 `(Chain (cons-atom $a $nodes) …)`; `(cons-atom $r $flags)` pln_supplement_recommendation.metta:234 |
| `foldl-atom` (5-arg: list init $acc $x body) | grounded | 3 | pln_linage2.metta:152 `(foldl-atom $ys 0.0 $acc $y (+ $acc $y))`, :338 (body is a `let`), :340 (body is a `let*`); chosen because recursion took 23 s vs 15 ms (:146-149) |
| `unique-atom` | grounded | 3 / 2 | patient_profile.metta:224,321 — must keep the FIRST occurrence (:214-219; order change observed docs/kb_quick_wins/REPORT.md:112) |
| `()` empty tuple literal | grounded | 85 | as value `(if (== $xs ()) () …)` patient_profile.metta:201, `(DrivenBy ())` pln_linage2.metta:135, `(if $ok … ())` :142; as pattern/compare target 51× |
| `decons-atom`, `size-atom`, `index-atom`, `min-atom`, `max-atom`, `map-atom`, `filter-atom`, `sort-atom`, `union-atom` … | grounded | 0 | whitelisted by metta_validator.py:34-35 / linage2_router.py:122-129 only |

### 3.6 Atoms, symbols, variables, strings, layout

| construct | kind | count | notes and examples |
|---|---|---|---|
| variables `$name` | special-form | 2,517 | names with `-`: `$evidence-type` epistemic_calibration.metta:48; no `_` in variable names; a variable in the HEAD of a pattern sub-expression cellage_calibration.metta:128 |
| `&self` | space-op | 1 in KB | pln_chat/logs/last_query.metta:20; 42 of 46 few-shots pass `&self` as the first argument of a KB function (`(infer &self A C)`); no `&other` space names anywhere |
| ground facts (non-`=`, non-`:` top-level) | — | 3,534 | 190 distinct heads; 368 facts carry a nested expression argument: `(Dosage DrugAgeRow_0 (Measure 4.0 percent))` drugage_etl_short.metta:10; `(Causes Rapamycin (Inhibits MTORC1))` hallmark_targeting.metta:213; `(HasSenescenceEffect CellAgeRow_16 (Increases CellularSenescence))` tests/fixtures/cellage_real_rows.metta:24; unary fact `(IsITPStudy DrugAgeRow_1552)` tests/fixtures/drugage_real_rows.metta:11 |
| facts storing an UNEVALUATED function call | hyperon-workaround | 46 (41 root) | `(Effect CellularSenescence SASP Pos (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))` mechanistic_bridges.metta:53-54 and 20 more there; supplement_evidence.metta:91-107; lifestyle_evidence.metta:141-142,172-173; linage2_core.metta:322-323 — the call is reduced later by arithmetic (mechanistic_bridges.metta:50-52) |
| symbols with special characters | — | — | `%` as a symbol drugage_entries.metta:8 `(Inheritance % Unit)`; `/` inside unit symbols `mg/kg/week` drugage_etl_short.metta:1431 (14×), `ppm/day` :1948, `ug/mL` :1768, `mg/kg` :2022, `mg/mL` tests/fixtures/drugage_real_rows.metta:82; `->` inside names `z->status` patient_profile.metta:162, `hr->sign` pln_deduction.metta:111, `effect->strength` evidence_calibration.metta:62, `category->confidence` :72; `?` suffix `drugage-has?` drugage_calibration.metta:100, `drugage-itp?` :104, `drugage-significant?` :105; `*` in `let*`; lowercase symbols `water`, `food` (drugage_etl_short.metta:2718); mixed `A512d`, `UM_HET3`, `Age_lt50`, `CDKN2A_P16`, `N_2_Selenium_bridged_beta_cyclodextrin` drugage_entries.metta:12; no symbol starts with a digit; `Age_lt50` exists because `<`/`>` tokenize (nhanes_reference.metta:66, linage2_core.metta:81) |
| strings | grounded | 163 | contain `;` (9: hallmarks_lopezotin2023_intervention_evidence.metta:84; hallmark_targeting.metta:219), parentheses (12: hallmark_targeting.metta:113 `"Aging (Albany NY)"`), apostrophes (4: lifestyle_evidence.metta:191), non-ASCII em dashes (5: human_evidence.metta:217,231,254,266; supplement_evidence.metta:141); no backslash escapes; used as data (`(pmid "25041462")`), as a pattern literal (evidence_calibration.metta:62) and rendered back to Python with quotes (pln_chat/ontology/human_evidence.py:51 strips `^"(.*)"$`) |
| string functions (`format-args`, `parse`, `repr`, `concat`, `stringToChars`) | stdlib | 0 | not used |
| comments | — | ~4,500 lines | `;` / `;;` to end of line; glued `);` lifestyle_evidence.metta:228; UTF-8 inside comments everywhere (§ × — ≤ ∈ ± → µ); STRUCTURED comments parsed by Python: linage2_core.metta:165-236 `;; LBXCRP — C-reactive protein …` via regex `\(ModelInput (\w+) LinAge2\)[^\n]*?;;\s*(\w+)\s*—\s*([^\n]*)` (pln_chat/core/patient_vocabulary.py:33, linage2_builder.py:54-60) — the em dash is load-bearing |
| layout | — | — | 220 top-level expressions span several lines; 235 code lines hold ≥2 expressions (linage2_core.metta:165-236 two atoms + comment per line; measurement_types.metta:24-32; tests/fixtures/cellage_real_rows.metta 25 lines) |

### 3.7 Space, I/O, interop

| construct | kind | count | notes |
|---|---|---|---|
| `get-atoms`, `add-atom`, `remove-atom`, `new-space`, `bind!`, `import!`, `include`, `load-ascii` | space-op | 0 | composition is done in Python by text concatenation (pln_runner.py:880-895); validator whitelists get-type/get-atoms/new-space/add-atom/remove-atom (metta_validator.py:37) |
| `println!`, `trace!`, `assertEqual`, `assertEqualToResult` | io | 0 | `assertEqual` only in linage2_router.py:128 head list |
| `py-call`, `py-atom`, `py-dot`, `py-list` … | grounded | 0 | none |
| `get-type`, `get-metatype`, `is-expression`, `=alpha`, `noreduce` | stdlib | 0 | `get-type` was used manually to vet the validator whitelist (metta_validator.py:31-33) |

### 3.8 Hyperon-specific workarounds documented in the files (kind: hyperon-workaround)

1. Space threading: every rule takes `$space`/`$s` and callers pass the top-level `&self`,
   because `(match &self …)` inside an imported module read only that module's space —
   evidence_calibration.metta:41-49, pln_deduction.metta:100-104, docs/calibration_layer.md:193-210,
   and the STATUS header of 15 files (e.g. patient_profile.metta:42-47, pln_counterfactual.metta:50-57).
2. `let`/`let*`-forcing of every intermediate because grounded ops (`car-atom`, `cdr-atom`,
   `cons-atom`, `==`, `+`, `*`, `>`, `<`, `pow-math`, `superpose`) "do NOT evaluate their
   arguments" — pln_intervention_ranking.metta:73-78,106-107; pln_abductive_diagnosis.metta:55-59;
   patient_profile.metta:158-161,194-195,283-288; drugage_calibration.metta:134-138,202-207;
   nhanes_reference.metta:279-280; nhanes_baseline.metta:310-312,319-321; pln_risk_prediction.metta:72-79.
3. Collapse-then-test for single-valuedness ("MeTTa has no cut") — patient_profile.metta:250-262,
   pln_risk_prediction.metta:224-228; ambiguity → nothing (nhanes_reference.metta:202-228,
   nhanes_baseline.metta:289-300).
4. `(superpose ())` to yield nothing; "only `match` or an explicit `(superpose ())` yield nothing"
   (tests/test_nhanes_common.py:735).
5. Presence test before naming an optional layer's rule, since an unmatched head stays unreduced —
   pln_risk_prediction.metta:230-252, nhanes_baseline.metta:79-85.
6. Accessors deliberately UNTYPED (a `(-> Atom …)` signature stops argument reduction) —
   hallmark_targeting.metta:231-236; `Effect`'s Atom-typed args used to keep stored calls
   unevaluated — drugage_calibration.metta:202-207, cellage_calibration.metta:123-125.
7. `foldl-atom` instead of recursion (pln_linage2.metta:55,146-149) and `unique-atom` instead of
   `unique-tuple` (patient_profile.metta:214-219) for speed.
8. The 1,024-distinct-key-atom / head-symbol abort (hyperon #1076/#1081/#1095; trie.rs:179):
   scoped stacks (pln_runner.py:22-28,43-50,84-101,149-182,252-305; docs/nhanes_integration.md:385-440;
   linage2_core.metta:44-77), the flat 12-ary `HumanEvidence` record instead of one atom per
   field (human_evidence.metta:94-113), NHANES codes kept as comments (linage2_core.metta:58-67,
   147-155), the smoking precondition NOT enforced in MeTTa (pln_counterfactual.metta:208-240),
   the composition inlined into `cf-component` (:266-270), a separate `baseline-age-band` to avoid
   a duplicated `age-band` answer (nhanes_baseline.metta:269-272), DrugAge/CellAge rows sliced
   per query (drugage_calibration.metta:35-36, cellage_calibration.metta:67-68,146-149).
9. Union of equations (`interaction-flag`, pln_supplement_recommendation.metta:176-193) and the
   `-0.0` avoidance `(if (> $total 0.0) (* -1.0 $total) 0.0)` (pln_counterfactual.metta:346).
10. `log-math` silent `-inf`/`NaN` → guard on the input (nhanes_reference.metta:253-270).
11. Python side: `_normalize_query`, one-space concatenation, per-`!` result groups, the
    process-per-query executor, the 60 KB file cap (pln_chat/config.py:70-78) and the
    `PLN_MAX_METTA_SORT_COMPOUNDS` cap on the O(n²) MeTTa sort (config.py:88-95).

---

## 4. Semantics the KB silently depends on (check each on the target runtime)

4.1 **Arithmetic over an unevaluated call read out of a fact.** `infer` binds `$tv` to
`(stv 0.85 (evidence-confidence AnimalStudies_Replicated))` (mechanistic_bridges.metta:53-54) and
`combine-tv` computes `(* (* $cab $cbc) (chain-discount))` (pln_deduction.metta:89-91); the nested
call must reduce to 0.65 there, and the printed result must show numbers (`(signed Pos (stv 0.357
0.29089125))`, pln_deduction.metta:171-172) because `_STV_RE` only matches numeric literals
(pln_runner.py:588-591). 46 facts (41 in root files) are stored this way.
4.2 **`match` templates are evaluated after binding** (14 KB sites, §3.1; few_shot_examples.json:44,207).
4.3 **A `let` whose pattern fails to unify yields nothing** (filter idiom, 9 sites).
4.4 **A `let` over a multi-valued expression fans out** (30 sites) and `collapse` gathers in
engine order; `(collapse …)` of nothing is `()`.
4.5 **`(superpose ())` yields nothing; `superpose` does not reduce its argument tuple.**
4.6 **An unmatched function call reduces to itself** (tests/test_nhanes_common.py:732-754).
4.7 **A second `=` equation adds an alternative, never replaces** (pln_deduction.metta:132-135 +
drugage_calibration.metta:240-243 extend `infer` across files; api.py:534-541).
4.8 **`(: …)` and `(= …)` are retrievable atoms** (pln_risk_prediction.metta:247; few-shots :44,53).
4.9 **Symbols compare with `==`**, tuples compare with `()`, `True`/`False` are the Bool literals `if` consumes.
4.10 **Mixed int/float arithmetic and comparison** (§3.4 numbers).
4.11 **`unique-atom` keeps the first occurrence; `car-atom` of a `collapse` is the engine's first
answer** — the chosen derivation depends on result order in: pln_counterfactual.metta:107-109
`pos-transmission`, :195-205 `resolve-lever`, :278-281 `cf-component`; drugage_calibration.metta:125-130
`species-tier-conf`; patient_profile.metta:258-262 `patient-z`; pln_risk_prediction.metta:257-261
`patient-baseline`, :153-158; human_evidence.metta:274-281; pln_linage2.metta:132-143,298-319.
4.12 **Result order is otherwise documented as not significant** (pln_abductive_diagnosis.metta:172
`SupportedBy`; human_evidence.metta:312,341-342; pln_linage2.metta:159 "the API sorts by |years|";
patient_profile.metta:296-297), and the sorts are deterministic insertion sorts whose `>=` keeps
collapse order among ties (pln_intervention_ranking.metta:83).
4.13 **Determinism**: "identical inputs -> identical output" is a stated contract
(pln_intervention_ranking.metta:69-71; tests/test_counterfactual.py:304).
4.14 **Reader**: `;` comments (also glued to `)`), UTF-8 in comments and strings, `"…"` strings
containing `;` and parentheses, scientific-notation numbers, negative numbers, `%` and `/` and
`->` and `?` inside symbols, several expressions per line, multi-line expressions.

---

## 5. Result atom shapes and how Python consumes them

| shape | produced by | consumed by |
|---|---|---|
| `(stv s c)` | everywhere | `_STV_RE` pln_runner.py:588-591 (decimal or `1e-05` exponent form); `_apply_threshold` :460-470 and `_split_top_level` :421-458 split a collapsed tuple TEXTUALLY on the printed `((…) (…))` form; API returns `atom`, `strength`, `confidence` (api.py:3577-3583, 3726-3732) |
| `(signed Pos\|Neg (stv s c))` | `infer`/`link-effect` pln_deduction.metta:113-135, `combine-hop` :94-95 | regex in `_SCORED_RE`, `_CELLAGE_EFFECT_RE`; linage2_router.py:270-283 |
| `(Chain (n0 … nk) (signed …))` — list built with `cons-atom` | `explain` pln_deduction.metta:144-155 | shown verbatim to the user (utils/formatting.py:54-60); docs use `(explain &self …)` 5× |
| `(scored C score (signed … (stv s c)))` | pln_intervention_ranking.metta:52-63; patient_profile.metta:389-391 | `_SCORED_RE` pln_runner.py:415-419 → `ScoredCompound`; the ranked tuple `((scored …) (scored …))` is the first atom (pln_runner.py:795-797); tests parse it (tests/test_drugage_calibration.py:342-349) |
| `(Effect G CellularSenescence Pos\|Neg (stv s c))` | cellage_calibration.metta:126-132 | `_CELLAGE_EFFECT_RE` pln_runner.py:1021-1024; result ORDER assumed equal to the injected row order (:1074-1086) |
| `(Hypothesis H (stv s c) cov mass (SupportedBy (…)))`, `(explains O (stv s c))` | pln_abductive_diagnosis.metta:43-45,99-110 | tests/test_hallmark_targeting.py:329 reads `(Hypothesis (\w+)` in order |
| `(Component M (z z) (weight w) (contribution c) (DrivenBy (…)))`, `(Decomposition P (Components (…)) (attributed a) (residual r))`, `(Counterfactual L O (expected-delta d) (signed Neg (stv s c)) (Via (…)))`, `(cf-part …)` | pln_counterfactual.metta:128-165, 275-289, 338-350 | regex in tests/test_counterfactual.py; accessors pln_risk_prediction.metta:376-378 destructure `Counterfactual` |
| `(RiskPrediction P O (point p) (ci lo hi) (confidence c) (baseline p0) (multiplier m) [(clock LinAge2)])`, `(RiskDecomposition …)`, `(RiskFactor M (contribution …) (share …) (attributable-risk …))`, `(ProjectedRisk L O (point …) (reduction …) (delta-clock\|delta-years …) (confidence …) (Via (…)) [(clock LinAge2)])`, `(SDToYears (curated 4.2) (measured none\|v))`, `(RiskInputs O (baseline r) (horizon-months h) (hazard-ratio hr))` | pln_risk_prediction.metta:304-323, 334-366, 376-394, 153-158; nhanes_baseline.metta:380-386; pln_linage2.metta:252-263, 377-389 | linage2_router.py:293-331 `risk_to_dict`, `projected_to_dict`; tests/test_hallmark_targeting.py:322 compares `(point p)` to 1e-12 |
| `(SupplementRecommendation P (Tier1HighConfidence (…)) (Tier2Promising (…)) (NotRecommended (…)) (Interactions (…)))`, `(SuppRec S (tier T) (score sc) (evidence cat conf) (safety lbl f) (Targets (…)))`, `(InteractionFlag S M "note")` | pln_supplement_recommendation.metta:111-128, 186-218, 231-235 | shown verbatim; tests/test_supplement_recommendation.py; `supplement-for-patient` returns the record AND flags as separate results via `(superpose $all)` |
| `(HumanEvidence id intv outcome (design D) (n N\|NotStated) (result R) (tier T\|NotStated) (pmid "…") (measured "…") (found "…") (caveat "…"))`, `(HumanEvidenceCrossReferenced intv (tier T) (see "…"))`, `(NoHumanEvidenceRecord intv "…")` | human_evidence.metta:198-256, 274-304 | pln_chat/ontology/human_evidence.py:51 strips the double quotes hyperon prints around strings; tests/test_human_evidence.py:357,365 `startswith("(NoHumanEvidenceRecord Rapamycin")` |
| `(LinAgeDecomposition P (delta-years d) (Measured (…)) (Imputed (…)) (attributed-measured …) (attributed-imputed …) (age-term-residual …))`, `(Contribution F (years y) Measured\|Imputed (ReadsOut None\|B (witnessed True\|False)) (DrivenBy (…)))`, `(LinAgeHazard P O (delta-years d) (hazard-multiplier m) (confidence c))`, `(LinAgeCounterfactual P L (expected-delta-years d) (signed Neg (stv s c)) (Via (…)))`, `(lin-part …)` | pln_linage2.metta:125-176, 220-224, 357-366 | `parse_sexpr` linage2_router.py:217-245: a real s-expression reader over `str(atom)` — every numeric token → float, `"…"` → str, symbols → str; `True`/`None` compared as strings (:259-266); `_fields` reads `(name v…)` pairs |
| `(Status marker Elevated\|Normal\|Low)`, `(pair a b)`, `True` | translator templates only (few_shot_examples.json:134,207,22) — `pair` is defined nowhere (0 KB uses), so it survives as an unreduced expression | shown verbatim |
| `Elevated` / `Normal` / `Low`, `Age_50_59`, `ITP_Positive`, numbers | `patient-status`, `age-band`, `itp-category`, `evidence-confidence` | tests compare `float(str(atom))` (tests/test_nhanes_common.py:754) |

Float rendering: hyperon prints via Python repr-like shortest round-trip
(`-0.08190000000000001`, lifestyle_evidence.metta:248; `0.17017138125`, pln_deduction.metta:178;
small magnitudes as `1e-05`, pln_runner.py:581-585). Tests mostly use tolerances, but
tests/test_hallmark_targeting.py:322,347 require agreement to 1e-12 with hyperon's own
arithmetic (`pow-math`, `*`), i.e. IEEE doubles and ≥13 printed digits. `-0.0` is avoided by hand
(pln_counterfactual.metta:346). Strings print with double quotes; the empty tuple prints `()`.

Order dependencies: (a) one result group per `!` expression in program order, re-merged by index
(pln_runner.py:784-801 `merge_run_results`); (b) CellAge links aligned to rows by position
(:1074-1086); (c) `car-atom`-of-`collapse` sites (§4.11); (d) `unique-atom` first-occurrence
(§3.5); (e) `(DrivenBy …)`, `(SupportedBy …)`, `(Via …)`, `(Measured (…))` carry collapse order
and are documented as insignificant, the API sorting LinAge2 contributions by |years|
(linage2_router.py:276-290); (f) DrugAge `linear` strategy sorts in Python, `metta_sort` in MeTTa
and the two must agree (pln_runner.py:706-727).

---

## 6. Query shapes the translator is taught (must run on the new runtime)

Sources: pln_chat/prompts/system_prompt.txt (309 lines), few_shot_examples.json (46 examples),
docs/*_chat_test_queries.md and docs/nl_query_examples/ (138 form occurrences tallied).

1. `(match &self <pattern> <template>)` — system_prompt.txt:24-27; `&self` literal (:21-22);
   bare template variable `(match &self (Inheritance $species Vertebrate) $species)`
   few_shot_examples.json:125.
2. Conjunction inside ONE match: `(match &self (, (UsesIntervention $exp Ethanol) (UsesSpecies $exp
   Drosophila_mojavensis) (AvgLifespanChangePercent $exp $change)) $change)` system_prompt.txt:28-39;
   few_shot_examples.json:80,89,98,107,116; up to 4 clauses; rule 15 forms :178-181.
3. `(pair $x $y)` and nested `(pair $x (pair $y (stv $s $c)))` templates — system_prompt.txt:62-63;
   few_shot_examples.json:22,44,53,89,116. `pair` is undefined in the KB.
4. `(if (> $c 0.7) (pair $e $c) (empty))` as a template, i.e. `if`, `>`/`<`, `empty` evaluated
   inside a match template — few_shot_examples.json:44,53.
5. Matching EQUATIONS as atoms: `(match &self (= (evidence-confidence $e) $c) …)` — :44,53.
6. Expression-headed patterns over "TV-annotated" atoms: `((Extends $drug $target) (stv $s $c))`
   :22, `((Causes Rapamycin (Inhibits mTORC1)) (stv $s $c))` :33 (no such atoms exist; the shape
   is still emitted).
7. Literal-constant template `True` (existence test) :134; template calling a KB function with
   `&self`: `(Status $marker (patient-status &self Patient001 $marker))` :207.
8. Nested constant arguments `(Dosage $exp (Measure 4.0 percent))` :107; `percent` not `%`
   (system_prompt.txt:276-278).
9. Direct function calls (no match): `(evidence-confidence RCT_Human)` :4, `(apply-tv 0.8
   AnimalStudies_Replicated)` :13, `(calibrate-tv &self Lu2019_DNAmPAI1_CHD)` :143, `(infer &self A
   C)` :152, `(explain &self A C)` (system_prompt-adjacent docs, 5×), `(rank-interventions &self (C1 C2
   C3) CoronaryHeartDisease)` :163 (a literal tuple argument), `(diagnose &self (H…) (O…))` :174,
   `(diagnose-patient &self P)` / 3-arg form :185,459 (system_prompt.txt:247-257),
   `(rank-interventions-for-patient &self P (…) O)` :196, `(decompose-grimage &self P)` :249,
   `(grimage-share &self P M)` :293, `(counterfactual-patient &self P Lever)` :260,271,282
   (system_prompt.txt:65-89), `(predict-risk-patient &self P)` :304, `(risk-decomposition-patient
   &self P)` :315, `(project-risk-patient &self P Lever)` :326,337 (system_prompt.txt:91-109),
   `(recommend-supplements-patient &self P)` :348,370, `(supplement-for-patient &self P S)` :359,381,
   `(recommend-supplements &self P (S…))` :392 (system_prompt.txt:111-127), `(human-evidence &self
   X)`, `(human-evidence-interventions &self)` (system_prompt.txt:129-154), `(hallmarks-of &self X)`,
   `(interventions-for &self H)`, `(match &self (TargetsHallmark X $h) $h)` (system_prompt.txt:156-181),
   `(linage-decomposition-patient &self P)`, `(linage-drivers-patient …)`, `(linage-hazard-patient
   …)`, `(linage-risk-patient …)`, `(linage-counterfactual-patient &self P Lever)`,
   `(linage-scenarios-patient …)` (system_prompt.txt:191-245; few_shot_examples.json:401-448).
10. `(rank-drugage-lifespan (C1 C2 …))` — system_prompt.txt:280-295, few_shot_examples.json:218,229,238:
    NOT MeTTa-executed; intercepted by regex (drugage_router.py:66-70).
11. Multi-expression programs, one expression per line, each becoming its own `!` (system_prompt.txt:55-57,
    191-196); `!` may also be written by callers (`/metta/run`), and a LinAge2 form nested in `let`
    is recognised (tests/test_linage2.py:332-333).
12. Truth values written `(stv <strength> <confidence>)` with floats in [0,1] (system_prompt.txt:52-53).
13. Symbols: CamelCase entities, `_`-joined species/strains, `PMID_xxxx`, `Patient001`/`Caller_W58`
    ids (patient_context.py:62), lever tokens, numbers (incl. `2.0e-75` — the validator blanks
    scientific literals, metta_validator.py:60-64).

---

## 7. What the validator accepts / rejects (pln_chat/core/metta_validator.py)

* Strips `;` comments (string-aware) :113-136; requires balanced parentheses :139-148, 184-185.
* Tokenises symbols with `[A-Za-z][A-Za-z0-9_\->]*` :57 — so `drugage-has?` is checked as
  `drugage-has` and `mg/kg/week` as `mg`+`kg`+`week`; variables `\$[A-Za-z][A-Za-z0-9_\-]*` :58;
  numbers incl. exponent :64 and strings `"[^"]*"` :71 (no escaped quotes) are blanked first :189.
* Every remaining symbol must be a builtin (:23-55), a variable name, known to the runtime
  inventory (`inventory.knows_symbol`), in the registry, or numeric — otherwise the query is
  INVALID (422 on `/metta/run`, api.py:3663-3672). Whitelisted builtins the KB never uses and the
  runtime must therefore still accept from callers: `empty`, `case`, `decons-atom`, `size-atom`,
  `index-atom`, `min-atom`, `max-atom`, `quote`, `unify`, `get-type`, `get-atoms`, `new-space`,
  `add-atom`, `remove-atom`, `or`, `Similarity`, `Evaluation`, `Member`, `ImplicationLink`,
  `OrLink`, `NotLink`, `<=`, `!=`, `=`, `fst`, `snd`, `sqrt-math`, `abs-math`, `sin-math`, `xor`,
  `Number`, `Bool`, `String`, `Atom`. `flip` is deliberately absent (:33).
* A head that is declared `(: P (-> …))` but holds no facts yields a WARNING, not a rejection
  :212-227. No arity check, no evaluation, no type check.
* The inventory it validates against is built by a hand-rolled text parser of the `.metta`
  files (pln_chat/ontology/inventory.py:131-327): top-level forms `(: …)`, `(= (f …) …)` and
  ground facts; it treats a `(:` with an arrow as a predicate and a `(= (f Const) v)` argument as
  an entity.

---

## 8. Python code that READS the .metta text (format dependencies independent of the engine)

* Knobs by regex: `(= (grimaccel-sd-to-years) 4.2)` and `(= (elevated-z-threshold) 1.0)`
  pln_chat/core/patient_context.py:22-23; `(= (linage-sd-to-years) 8.66)` linage2_builder.py:63;
  `lifespan-halfsat`, `sig-gate`, `clade-category`, `evidence-confidence`, `chain-discount` and
  `(Inheritance X Y)` pln_chat/ontology/drugage_scoring.py:52-57 (a Python re-implementation of the
  DrugAge scoring that must agree with the MeTTa one); expansion_schema.py:120,232-240.
* Facts by regex: `(PatientAge|PatientSex|PatientSmoking …)` api.py:477-479; `(LeverRequiresSmoking
  …)` ontology/lever_gate.py:39-41; `(MeasuresBiomarker …)` linage2_builder.py:61; `(ModelInput X
  LinAge2)` + structured comment patient_vocabulary.py:33; `(: X SmokingStatus)` :32; `(Effect _ T
  Pos|Neg` patient_builder.py:306; `(Interaction …)`, `(Inheritance X Pharmaceutical)` :332-333;
  DrugAge/CellAge row fields drugage_selector.py:99-106, cellage_selector.py:63-72 (rows split on
  `(InstanceOf … Experiment)` boundaries :136).
* Function bodies by regex: `(= (fn …)` and `(match $space (, (Head …` ontology/scoped_forms.py:35-40
  (which predicates each rule matches); `_DEFINED_HEAD_RE` pln_runner.py:188 (what each file
  defines, for stack routing).
* Program splitting: `_iter_top_level_exprs` pln_runner.py:620-690 (character-level, string- and
  comment-aware); `split_linage2_program` linage2_router.py:131-157.
* Patient atoms rendered by Python INTO the space: `(InstanceOf P PatientProfile)`, `(PatientAge P
  58)`, `(PatientSex P Male)`, `(PatientSmoking P CurrentSmoker)`, `(MeasuredZ P M z)` with `:.6g`
  (may print an integer), `(CurrentMedication P Drug)`, `(PatientCondition P CoronaryHeartDisease)`
  patient_builder.py:1014-1035; `(LinAgeDelta P d)` `:.6f`, `(LinAgeContribution P F y
  Measured|Imputed)` linage2_builder.py:353-357.

---

## 9. MeTTa produced by the ETLs / expander (same reader must accept it)

* DrugAge rows (drugage_etl.py:396-480): `(InstanceOf R Experiment)`, `(UsesIntervention R C)`,
  `(UsesSpecies R S)`, `(UsesStrain R St)`, `(HasSex R Male|Female)`, `(ReportedIn R PMID_n)`,
  `(IsITPStudy R)`, `(Dosage|Concentration R (Measure 4.0 percent))`, `(DeliveryMedium R food)`,
  `(AgeAtInitiation R (Measure 9.0 Months)|AfterWeaning)`, `(TreatmentDuration R UntilDeath)`,
  `(AvgLifespanChangePercent R 26.0)`, `(…Significance R Significant|NotSignificant|Unreported)` —
  `NotSignificant` vs the type `NonSignificant` of system_types.metta:20 (drugage_calibration.metta:85-86).
* CellAge rows (cellage_etl.py:171-227): `(InstanceOf R CellSenescenceRecord)`, `(InvolvesGene R
  Gene_X)`, `(GeneSymbol Gene_X "TP53")`, `(GeneName …)`, `(EntrezID Gene_X 7157)`, `(HasSenescenceType
  …)`, `(UsesCellContext …)`, `(HasSenescenceEffectLabel R Induces)`, `(HasSenescenceEffect R
  (Increases CellularSenescence))`, `(Causes Gene_X (Increases CellularSenescence) (stv 0.82 0.70))`,
  `(CorrelatedWith …)`; floats `:.2f`.
* NHANES records (nhanes_reference_etl.py:882-887, nhanes_mortality_etl.py:885-890,
  nhanes_dnam_etl.py:1110-1187): `(: Id ReferenceDistribution|BaselineRiskRecord|ClockAccelSpread)`
  plus one `(RefX Id v)` / `(BaseX Id v)` / `(SpreadX Id v)` atom per field, numbers in fixed
  decimal via `num()` (nhanes_common.py:985-991: "exponent forms like 1e-05 are read as symbols by
  some versions"), strings via `mstr()`.
* GenAge (genage_*_parser.py:44-106): `(: SYMBOL Gene)`, `(HasLifespanEffect R (Increases Lifespan))`,
  `(LifespanChangePercent R 12.0)`.
* LLM-extracted papers (pln_chat/ontology/expander.py, expansion_schema.py:232-240): `(Effect A B
  Pos|Neg (stv s (evidence-confidence T)))` and `(stv s c)` literals, `(InstanceOf …)`, `(Targets…)`
  written into pln_chat/ontology/metta_files/ (empty in this checkout).

---

## 10. Not used anywhere (safe to leave unsupported, but see §7 whitelist)

`import!`, `bind!`, `include`, `new-space`, `add-atom`, `remove-atom`, `get-atoms`, `get-type`,
`get-metatype`, `println!`, `trace!`, `assertEqual*`, `case`, `chain`, `function`/`return`,
`eval`, `quote`, `unify`, `decons-atom`, `size-atom`, `index-atom`, `min-atom`, `max-atom`,
`map-atom`, `filter-atom`, `sort-atom`, `%`, `sqrt-math`, `abs-math`, `exp`, `min`, `max`,
`round`, `<=`, `!=`, `or`, `xor`, string functions, `py-*`, `&`-spaces other than `&self`,
`Bool`/`Symbol` type names, `noreduce`, top-level `!` in KB files, `let*` pattern bindings,
2-branch `if`, unary minus.

---

## 11. Unknowns

* Whether hyperon type-checks ground facts at load time (never relied on by Python).
* Which exact float formatting hyperon's Python binding uses in every case (observed: Python-repr
  style in the smoke comments; exponent form asserted in pln_runner.py:581-585).
* Whether any of the above exists in PeTTa — not checked here (out of scope for this inventory).
