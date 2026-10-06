# LinAge2 Integration — Design (v1)

**Status:** v1, verified against `hyperon 0.2.10`; regression-guarded in
`tests/test_linage2.py`. Implementation: `linage2_core.metta`,
`linage2_fong2025_evidence.metta`, `pln_linage2.metta`, `pln_chat/core/linage2_builder.py`,
`pln_chat/core/linage2_router.py`, and the `linage2` block of a caller-supplied patient in
`pln_chat/api.py`.
**Scope:** let a patient's **LinAge2** result — the clinical (blood-panel) mortality clock
of Fong et al. 2025, as served by `Rejuve/LinAge2-Python` — enter the knowledge base as a
request-scoped patient, and let the inference stack do with it what the service cannot:
credit years to causes under evidence, price the delta as a hazard, and answer "what would
remove those years" through the causal graph. **Patients stay stateless**: nothing is
written to disk; the atoms live in one request's space(s) — the LinAge2 atoms only in
the LinAge2 scoped space (§5, decision 4).

> **Dependency / base.** Built on the `nhanes_integration` branch: it reuses `patient-z`
> (raw-or-standardized lookup), the outcome-keyed `patient-baseline`, and the
> query-scoped-stack pattern that branch measured its way to. It changes **no** existing
> `.metta` file and adds **nothing** to the shared execution space.

---

## 1. What LinAge2 gives, and what it does not

The LinAge2 service returns, for one person:

| field | meaning |
|---|---|
| `biological_age`, `metadata.chronological_age`, `metadata.delta_ba_ca` | BA, CA, and BA − CA in **years** |
| `metadata.feature_contributions[]` | one `{feature, contribution_years, is_imputed}` per model input — 59 of them |
| `metadata.imputed_features` | which inputs the service filled in from its reference cohort |

That is already a decomposition — but a decomposition into **labs**. It says *the cotinine
input adds 2.9 years*; it does not say why, whether the lab is actually high, what
evidence connects it to anything, or what the person could do. Those are the questions
the KB exists to answer, and the honest version of each turned out to need care.

### 1.1 A contribution's sign is not a lab's direction

LinAge2 is linear in its standardized inputs, so `contribution = z × w`. The obvious
inference — *positive years ⇒ the lab is high ⇒ credit the causes of a high lab* — is
**wrong**, and measurably so. `w` is a projection (SVD loadings × Cox coefficients), fitted
per sex, and read off the published artifacts:

| input | male `w` (months/SD) | female `w` |
|---|---|---|
| CRP | **−0.20** | **+4.44** |
| serum glucose | **−3.67** | **+8.65** |
| HbA1c | +7.41 | +4.94 |
| cotinine (smoking) | +35.2 | +37.1 |
| serum albumin | −22.6 | −5.48 |

A positive CRP contribution in a man means his CRP was *low*. So the KB never infers a
lab's direction from its years. Instead the layer requires a **witness** the patient's own
values supply: `patient-z` for the biomarker the input reads out above
`elevated-z-threshold`, or — for the cotinine input — a recorded `CurrentSmoker` status.
No witness, no cause: the years are reported and left unexplained. This is the one
modelling rule in the layer, and `pln_linage2.metta §2–3` is built around it.

### 1.2 Imputed values are the cohort's, not the patient's

The service imputes every missing input (median of a same-sex, same-age-window reference
cohort) and still reports a contribution for it. The layer carries those under an
`Imputed` flag, totals them separately, and **never credits a cause to one**. The API
tells the caller to re-test them.

### 1.3 The delta is on the chronological-age hazard scale by construction

`BA = CA + Σ(PC_i β_i) / β_age`, where `β_age` is the Cox coefficient on age in the
null model. So one year of delta carries exactly the log-hazard of one year of age, and
the paper's reported null-model **mortality rate doubling time of ~7.8 years** fixes it:
`HR per year of delta = 2^(1/7.8) = 1.093`. That is what
`linage2_fong2025_evidence.metta` records — labelled as a statement about the delta's
*scale*, not an independently fitted association — with the sex-specific artifact values
(1.083 male, 1.106 female) noted as a cross-check. Tier `Epidemiological` (0.60): one
cohort, observational, below the multi-cohort meta-analytic GrimAge hazards.

## 2. Where it sits

```
LinAge2 service ──/predict──▶ app ──patient.linage2──▶ pln_chat/api.py
                                                            │ core/linage2_builder.py
                                                            │   validate, map codes→symbols,
                                                            │   render request-scoped atoms:
                                                            │   (LinAgeDelta P d) (MeasuredZ P LinAgeAccel z)
                                                            │   (LinAgeContribution P <Input> y Measured|Imputed)
                                                            ▼
                    LINAGE2_PATIENT_STACK (query-scoped) ── pln_linage2.metta ──▶ atoms ──▶ JSON / LLM
```

The **stack is query-scoped** and **minimal** (`core/pln_runner.py` lists each file's
job). It is *not* the NHANES patient stack plus LinAge2 — see §5.

## 3. The ontology (`linage2_core.metta`)

* `LinAge2` is a `ClinicalClock` and a `MortalityRiskEstimator`; `LinAgeAccel` is its
  acceleration, a `Difference` (BA − CA), not an age-regressed residual — the distinction
  `nhanes_baseline.metta` draws for the DNAm clocks, recorded in `ValueEncoding`.
* **59 inputs**, each `(ModelInput <Input> LinAge2)` with a type
  (`ClinicalLabMeasurement` or `DerivedClinicalScore`) and a **structured comment**
  `;; <NHANES code> — <description>` that `linage2_builder.py` parses. The codes are
  comments, not `(NHANESCode …)` facts, for the budget reason in §5.
* **Six readouts**, `(MeasuresBiomarker <Input> <Biomarker>)`: `CRP → CRP`,
  `HbA1c → HbA1c`, `SerumGlucose → FastingGlucose` (with the caveat that the biochemistry
  glucose is fasting only if the draw was), `SerumCotinine → CurrentTobaccoExposure` (a
  new exposure node: cotinine is current exposure, not the cumulative pack-years
  `DNAmPACKYRS` surrogates), and, since item #7, `RedCellDistributionWidth → RDW` and
  `SerumAlbumin → LowSerumAlbumin` (the albumin DEFICIT; §7 item 3).
* **One Effect edge**: `SmokingCessation ⊣ CurrentTobaccoExposure`, strength 0.95
  (cotinine's ~17 h half-life, Dempsey 2013; SRNT 2002 on cotinine as the verification
  biomarker), tier `MultipleHumanTrials`. It lands on a node no GrimAge component reads,
  so the DNAm counterfactuals are unchanged — `resolve-lever` still finds
  `SmokingPackYears` first (asserted).

## 4. The rules (`pln_linage2.metta`) — what the KB adds

**Decomposition.** `(linage-decomposition-patient &self <P>)` returns every input with its
years, `Measured` and `Imputed` in separate tuples, the two totals, and the **age-term
residual** — LinAge2 adds `(age − mean training age) × w_age` to the feature sum, and the
identity `delta = measured + imputed + residual` holds exactly and is asserted. An input
that reads out a biomarker carries `(ReadsOut <B> (witnessed True|False))` and, only when
measured, positive and witnessed, `(DrivenBy (<Hallmark> …))` — the hallmarks with a
positive `infer` chain to the biomarker, deduped. On the fixture patient (a current
smoker, HbA1c z 1.6, CRP z 0.3):

```
(Contribution SerumCotinine (years 2.93) Measured (ReadsOut CurrentTobaccoExposure (witnessed True)) (DrivenBy ()))
(Contribution HbA1c         (years 0.86) Measured (ReadsOut HbA1c (witnessed True)) (DrivenBy (DeregulatedNutrientSensing)))
(Contribution CRP           (years -0.02) Measured (ReadsOut CRP (witnessed False)) (DrivenBy ()))
(Contribution SerumGlucose  (years 0.11) Measured (ReadsOut FastingGlucose (witnessed False)) (DrivenBy ()))   ; no glucose z sent
(Contribution SerumAlbumin  (years 2.26) Measured (ReadsOut LowSerumAlbumin (witnessed False)) (DrivenBy ()))   ; no albumin z sent
…  (attributed-measured 7.67) (attributed-imputed 0.12) (age-term-residual 0.50)   ; = delta 8.28
```

Albumin's 2.26 years are the second-largest driver and are **left unexplained** here: the KB has a
bridge for it (chronic inflammation lowers albumin, `ChronicInflammation -> LowSerumAlbumin`), but this
patient sent no albumin z, so there is no witness. With one (the tab computes it, see §10) the same record reads
`(ReadsOut LowSerumAlbumin (witnessed True)) (DrivenBy (ChronicInflammation CellularSenescence))`. That is the
honest state of the KB, not a gap the layer papers over.

**Hazard.** `(linage-hazard-patient &self <P>)` = `1.093^delta`, confidence `0.60 × 0.9`
= 0.54. Always available. **Risk.** `(linage-risk-patient &self <P>)` multiplies the
outcome-matched `patient-baseline` for `AllCauseMortality` — which exists **only** as a
generated NHANES record (`build/nhanes_mortality_baseline.metta`), never as a curated
table (`docs/nhanes_integration.md §5`) — through the **survival form**
`1 − (1 − p0)^(HR^delta)`, not `p0 × HR^delta`: an all-cause baseline can be 0.2–0.5 and
a +20-year delta is a ×6 multiplier, so the linear form leaves [0, 1]. Without the
baseline the form yields nothing and the API says why.

**Counterfactual, in years.** `(linage-counterfactual-patient &self <P> <Lever>)`
resolves the lever to **all** the drivers it stands for (not the first, as
`resolve-lever` does — `SmokingCessation` now reduces two exposures), reaches each
measured input's readout biomarker by identity or `pos-transmission`, composes the
lever's own evidence edge (`s_l × s`, `c_l × c × chain-discount`, as
`pln_counterfactual.metta §4`), and removes `s × years` from each **witnessed, positive**
input. `SmokingCessation` on the fixture smoker: **−2.79 years** (0.95 × 2.93), c 0.765,
`Via (SerumCotinine)`; on the same atoms for a never smoker: **0, `Via ()`** — the
`LeverRequiresSmoking` precondition `pln_counterfactual.metta §3b` could not afford to
enforce in the shared space is enforced here by the engine. `Metformin`: −0.45 years via
`HbA1c` only (glucose is measured but no glucose z was sent, so it is unwitnessed). `Elamipretide`: omitted.
`(linage-project-risk-patient …)` restates a counterfactual in absolute risk when a
baseline is loaded; `(linage-scenarios-patient …)` runs the four standing levers.

**Performance note.** Every sum uses the grounded `foldl-atom`: a MeTTa recursion over the
33 measured records took 23 s and over 59 bare numbers 16 s; `foldl-atom` sums 59 in
15 ms. The full `/linage2/analyze` program runs in ~8 s without a baseline.

## 5. The engine budget, again, and what it forced

`docs/nhanes_integration.md §8` established that hyperon 0.2.10 aborts the process on
**distinct head symbols**, not atoms, and that the shared space has none to spare. This
layer re-measured, and the picture is subtler than a head count:

| space | result |
|---|---|
| shared space + 60 `ModelInput`/typing facts | **abort** on an unrelated `decompose-grimage` |
| NHANES patient stack + lifestyle + LinAge2, 59 `(NHANESCode … "…")` string facts | **abort** on every LinAge2 query |
| same, string facts removed | runs; **+5 new heads → abort** |
| minimal stack (§2) + LinAge2, string facts removed | runs at **+32 new heads**; aborts at +64 |
| minimal stack + a full generated all-cause baseline (12 field heads, 8 cells) | runs, and `linage-risk-patient` returns a number |
| a fresh space with 1,000 heads, or 300 children under one node | runs — the trigger is an interaction, not a simple cap |
| **shared** space + one LinAge2 **patient's** atoms (61: `LinAgeDelta`, 59 `LinAgeContribution`, the `LinAgeAccel` z) | **abort** on `diagnose-patient`, `predict-risk-patient`, `recommend-supplements-patient` for that patient; the same three answer without those 61 |

Four decisions follow, each asserted in `tests/test_linage2.py`:

1. **Scoped, not shared.** The LinAge2 files are never in `_INFERENCE_STACK`.
2. **Minimal underneath.** `LINAGE2_PATIENT_STACK` drops `pln_intervention_ranking`,
   `pln_abductive_diagnosis`, the López-Otín intervention records and `nhanes_reference`
   — nothing a LinAge2 form reads — which is what leaves room for the generated baseline.
   The margin test (`+16` heads on top of a full baseline) guards it.
3. **Codes as comments.** The 59 NHANES codes cost more than the space had as string
   atoms; as structured comments they cost nothing and stay the single source of truth
   (`GET /linage2/features` publishes them).
4. **The patient is split too.** v1 kept the layer's *files* out of the shared space but
   still injected every atom of a LinAge2 patient into it, so any non-LinAge2 question
   about that patient — risk, differential, supplements — aborted (last row above;
   measured on the server's process-per-query path, not only under pytest).
   `BuiltPatient.shared_atoms` is the patient minus its LinAge2 atoms, and it is what
   the shared space gets; `atoms` (everything) goes to the scoped space, the preview and
   the translator's prompt. A subprocess test asserts both halves: the three forms answer
   with `shared_atoms`, and still abort with `atoms` — the control that keeps the first
   assertion meaningful.

**Mixed programs.** A question that asks for both — "my LinAge2 drivers and my
supplement plan" — used to route *wholesale* to the scoped space, whose minimal stack has
no supplement layer. `split_linage2_program` now cuts a program per top-level expression
(character-level: two expressions on one line are two; comments are ignored, so a
`linage-*` name inside one routes nothing): the `linage-*` ones run in the scoped space
with every atom, the rest in the shared space with `shared_atoms`, each part validated
against the space it runs in (an issue names its space). Both parts run in **one**
offloaded task (`run_query_parts`: one worker, one deadline, one admission — measured safe
in either order), and each result atom carries the index of the expression that produced
it, so the answers come back in program order even for L, G, L
(`routed: "linage2+generic"`). An expression that *nests* a LinAge2 form inside another
layer's form cannot be split; it runs in the scoped space and the answer carries a warning
naming the form that will not evaluate there. The Gradio chat does the same. Rule 16 of the
translator prompt allows the combination, one form per line.

**The patient stack.** Keeping the LinAge2 atoms out fixed the crash they caused, but
building patients from typed text showed the shared space was still at its edge for *any*
patient: a caller whose CRP or HbA1c z is 1.2, 2.0 or 0.31 aborts diagnose / supplements /
ranking there (15 of 15 runs), while 0.3 or 0.5 happen to run, and dropping any one of a
dozen unrelated files makes it run — the head-symbol budget of §5 again, failing in
hyperon's space index (`trie.rs:179`, `unwrap()` on a hashed atom that is not there). The
built-in patients abort too (`rank-interventions-for-patient`, `recommend-supplements-
patient` for Patient001). So every program that names a patient, or reads patient facts
(`MeasuredZ`, `PatientAge` …; `core.patient_context.reads_patients`), runs in
`core.pln_runner.patient_stack`: the runtime stack minus seven files no patient form reads.
Any other program runs in the full stack *without* a session patient's atoms
(`patient_atoms_for`): holding a caller, the full space aborts as soon as a program
enumerates patient facts.
There all of those answer (+64 head symbols of margin), and wherever the full stack answers
the result is byte-identical — including the three Patient001 outputs
`tests/test_hallmark_targeting.py` captured on an earlier commit.

Two pre-existing defects surfaced on the way and are fixed: `_normalize_query` split only
at line ends, so `!(a) !(b)` on one line evaluated `a` and silently *added* `b` to the
space; and `/query` validated the shared space without the caller's atoms, so every answer
about a caller-supplied patient reported its own id as "not found in loaded ontology".

## 6. The API surface

| call | what |
|---|---|
| `patient.linage2` on `/query`, `/metta/run`, `/patients/preview` | the `/predict` body, forwarded; becomes the `LinAgeAccel` marker + the contribution atoms |
| `GET /linage2/features` | the 59 inputs, codes, descriptions, readouts, how a cause is credited, the stack, whether a baseline is loaded |
| `POST /linage2/analyze` | LLM-free: decomposition, hazard, risk (if baseline), counterfactuals per lever, as JSON |
| `routed: "linage2"` | a `linage-*` form was validated against and run in the scoped space |
| `routed: "linage2+generic"` | a program mixing `linage-*` forms with others: each part ran in its own space, answers joined in order |
| `markers.LinAgeAccel` | the bare delta (years or z) — hazard only; mutually exclusive with the block |

Refusals, each a 422 with a code: `unknown_linage2_feature`, `duplicate_linage2_feature`,
`linage2_inconsistent` (delta ≠ BA − CA), `linage2_age_mismatch` (a result computed for
another age), `implausible_linage2_delta` (> 50 y; > 35 y warns, citing the paper),
`implausible_linage2_contribution`, `duplicate_clock`, `unknown_lever`, `linage2_required`.
A form naming a patient with no LinAge2 result is valid, empty, and **warned** — the
built-in patients have none.

**Send the witnesses.** The block alone yields years without causes and counterfactuals
of 0. Send the patient's own `CRP` / `HbA1c` / `FastingGlucose` (z or value) and
`smoking` alongside; the builder says so when they are missing.

## 7. Honest limits

1. **Two z conventions meet here.** LinAge2 standardizes on Box-Cox-transformed values
   against a ≤50-year-old reference by median/MAD; the KB's `MeasuredZ` is SDs from an
   age/sex-adjusted mean. The join is *qualitative* (Elevated or not), which is why it
   survives the mismatch; a quantitative reconciliation is future work.
2. **`linage-sd-to-years` = 8.66** is the SD of BA − CA across the model's training cohort
   (2,079 NHANES 1999–2000 participants, computed from the shipped training matrices). It
   affects the clock's Elevated/Normal/Low label and nothing numeric.
3. **Six of 59 inputs can be explained.** CRP, HbA1c, glucose, cotinine, and, since item #7 of
   `docs/kb_quick_wins/REPORT.md`, **RDW and albumin**. Blood pressure, NT-proBNP and the rest are carried as
   years. Bridges for them are curation work of the `mechanistic_bridges.metta` kind.
   **RDW and albumin are weak, and their witness is a choice.** Both edges come from chronic inflammation
   (`Epidemiological` tier, strengths 0.50 and 0.55, the weakest priors in the file): hsCRP and ESR predict RDW
   independently of age, sex, MCV, haemoglobin and ferritin, in a cross-sectional study (Lippi 2009, PMID
   19391664), and albumin is a negative acute-phase protein that low protein intake also lowers (Soeters 2018,
   PMID 30288759; Don & Kaysen 2004, PMID 15660573). The albumin node is the DEFICIT, `LowSerumAlbumin`, and the
   patient's z for it is the negated albumin z. The witness is a z above 1 against **LinAge2's reference for people
   up to 50** (sex-specific, not age-adjusted): RDW above 13.09 % (men) / 13.27 % (women) (z = 1 falls at 13.093 and 13.265, so 13.1 % and 13.3 % are the first values that count), albumin at or below 43 g/L /
   41 g/L. That reference is stricter than a laboratory range, so **a value inside the usual range counts** (a man
   with albumin 4.3 g/dL), and it overcalls older people. The tab says so next to the patient (`witness_notes`);
   the alternative was clinical limits (RDW > 15 %, albumin < 3.5 g/dL), which change nothing for the three tab
   examples. A raised RDW is **withheld** as a witness when the person typed a haemoglobin below 13 (men) / 12
   (women) g/dL, a ferritin below 30 µg/L, a B12 below 148 pmol/L or a folate below 10 nmol/L (anaemia and
   deficiency raise RDW by themselves: Bessman 1983, PMID 6881096; Förhécz 2009, PMID 19781428); those limits are
   usual laboratory limits chosen on the sensitive side, not a result of a paper. Albumin has no such gate: nutrition
   and a recent meal are named in the note.
   **Triglycerides** (item #12; urate and white cell count were left out on purpose: the urate edge would imply
   "metformin lowers urate", on which the two studies found disagree, and white cell count has no primary anchor
   and is confounded by smoking) hang on insulin resistance: `(Effect InsulinResistance Triglycerides Pos (stv 0.50
   Epidemiological))` (Ginsberg 2005, PMID 15925013, a mechanism review; McLaughlin 2003, PMID 14623617, uses
   triglycerides only as a marker). Not a LinAge2 input, so no readout and no change to the decomposition. The
   witness is a FASTING value of 150 mg/dL or more, the usual limit: the tab reads it only from "fasting
   triglycerides …" (a plain "triglycerides" gets a note, like glucose), because a non-fasting value runs up to
   0.3 mmol/L (27 mg/dL) higher (Langsted 2008, PMID 18955664). The builder's reference is not a population
   distribution: geometric mean 100 mg/dL and a log-scale sd of 0.4054 put z = 1 at 150 mg/dL.
   Asking what to DO about one of them without a bridge ("what should I do about my kidney function", "what
   if my blood pressure were normal") used to give an empty clarification, an invented lever token
   (`BloodPressure`) or, for a diagnosis, another lever. Rule 16 now maps it to `(linage-decomposition-patient …)` (plus
   `(recommend-supplements-patient …)` when supplements were asked) and tells the translator to say in
   `explanation` that the knowledge base holds no curated cause or lever for it, and that the years shown are
   what the lab adds to the clock, never what "normal" would remove. The rule lists the markers the KB does
   have a curated cause for; `test_rule_16_names_exactly_the_markers_the_knowledge_base_has_a_cause_for`
   reads that list from `mechanistic_bridges.metta`, so a new bridge fails it until the rule is updated.
   RDW and the albumin deficit ARE levers now, by their own names (`RDW`, `LowSerumAlbumin`); both act through
   chronic inflammation, so they return the same years. Checked against the live translator (13 questions before
   and after, plus 3 repeats of the one control that changed): the no-relation questions map as above, and the controls (scenarios, quitting smoking,
   decomposition, drivers + plan, diagnosis, the HbA1c / insulin-resistance / metformin levers) do not change.
   "What if my inflammation were normal?" used to go to the GrimAge `counterfactual-patient` for a patient
   with no GrimAge value; the per-patient hint now sends it to the LinAge2 form.
4. **No cause-specific risk.** The paper reports none per year; `CoronaryHeartDisease`
   through LinAge2 yields nothing rather than borrowing the all-cause hazard.
   "What's my heart risk?" for a patient with a LinAge2 result and **no GrimAge value** (the tab's
   default) maps to the pair `(predict-risk-patient …)` + `(linage-hazard-patient …)`: the CHD model
   has no input and returns nothing, the hazard answers, and a deterministic note
   (`core.patient_context.patient_form_warnings`, on `/query`, `/metta/run` and the chat) says there
   is no heart-specific risk for the patient and that the hazard beside it is the ALL-CAUSE
   multiplier, never a heart risk and never multiplied or added to a GrimAge result. With a GrimAge
   value the CHD model answers and the note does not fire; a hazard-only question is not relabelled.
5. **Two clocks are not combined.** A patient with both GrimAge and LinAge2 gets a CHD
   risk from one and a mortality hazard from the other; multiplying them would double-count
   (`docs/risk_prediction.md §3`).
6. ~~**Not in the chat UI's patient surface.**~~ Closed by §10: the **My Patient** tab.

## 8. Non-goals

- ~~Not a LinAge2 implementation.~~ **Reversed in v2 (§9):** a demo cannot ask a person to
  run a separate service and paste its JSON, so the model is now evaluated in-process from
  parameters extracted out of `Rejuve/LinAge2-Python`. The service's response shape is
  still the interface, and `patient.linage2` still accepts it.
- No per-feature weights in the KB (sex-specific, sign-flipping, retrain-sensitive). This
  still holds: v2 keeps them in a Python-side data file; the KB sees only years per input.
- No numeric AUCs transcribed from figures (§1.3 of the evidence file says what is and is
  not recorded).
- No stored patients.

## 9. v2 — the model, copied in

**Why.** The tab in which a person types their labs (and gets questions answered about
themselves) cannot depend on a second service they would have to run and paste from.
LinAge2 turned out to be small enough to carry: it is *additive* in its inputs.

**What.** For a person of sex *s*, age *a* (months), and input *j* after imputation and
derivation:

```
z_j     = clip( (BoxCox_j(x_j) − median_sj) / mad_sj , −6, 6 )     # ≤50-year-old reference, sex s
years_j = (z_j − μZ_sj) · w_sj / 12                                 # w = SVD loadings × Cox β / β_age(null)
delta   = Σ_j years_j + (a − μAge_s) · wAge_s / 12
```

(five inputs — the three questionnaire scores, cotinine, the basophil count — skip the
median/MAD step). `scripts/extract_linage2_model.py`, run once against a LinAge2-Python
checkout, reads every constant out of the artifacts and writes
`data/linage2/linage2_model.json` (108 KB: 59 features × two sexes, plus a per-sex,
per-year table of the reference medians the service imputes from, plus provenance — the
upstream commit and a SHA-256 of every artifact). `pln_chat/core/linage2_model.py`
evaluates it with no pandas, scipy, scikit-survival or pickles, and returns the service's
`/predict` body, which `build_linage2` consumes unchanged.

**Measured, not assumed.** The same script scores 48 random partial panels (both sexes,
ages 25–84, ~35% of inputs missing, a third with a full questionnaire, cotinine 0–3)
through the service's **own** `process_payload` and stores them as
`tests/fixtures/linage2_golden.json`. The port matches every case to **2 × 10⁻¹⁴ years**,
per input and in total (`tests/test_linage2_model.py`). Linearity is asserted directly:
changing one input moves only that input's years — which is why a partial panel still
gives exact years for everything that *was* measured, while the total assumes the rest
are typical for the person's sex and age.

**Deliberate departures from the service**, each tested:

1. *Provenance.* The service flags only imputed **lab** inputs. Here a derived input is
   flagged when any part of it was imputed (LDL from total cholesterol/HDL/triglycerides;
   the urine albumin/creatinine ratio), and a questionnaire score is flagged as *assumed*
   when its questions were not answered (the service silently assumes no diagnoses,
   "good" health, no visits). Everything not measured carries `is_imputed`, so the KB
   totals it apart and never credits a cause to it.
2. *Cotinine on the training scale.* The model was fitted on `digiCot` bins (0: <10 ng/mL,
   1: 10–100, 2: 100–200, 3: ≥200 — all four occur in the training matrix). The imputation
   medians are taken over the digitized column; the service imputed the **raw** median
   (~0.1 ng/mL) and read it as a level (≈ +0.3 y for a man).
3. *Refusals instead of NaN* — negative values, unknown inputs, cotinine levels other than
   0–3, ages outside 20–90 (warned outside the fitted 40–85).
4. *Imputation per whole year of age* (the service's window at age × 12 months).

**Problems found upstream** while porting (they live in `Rejuve/LinAge2-Python` and are not fixed here):

| where | what | effect |
|---|---|---|
| `db_mapping.UNIT_SCALE` | albumin, total protein, globulin scaled ×0.1 labelled "g/dL → g/L" (that conversion is ×10) | albumin 4.2 g/dL becomes 0.42 g/L: **+10.5 y** from albumin alone vs +1.7 y at the correct 42 g/L |
| `db_mapping` / KB comment | CRP documented as mg/L; NHANES 1999–2002 (and the reference cohort, median 0.14) is **mg/dL** | a mg/L value is read 10× too high |
| `_smoke_db012_to_nhanes_lbx_cot` | a daily smoker maps to level 2; training has 0–3 and most smokers at 3 | ≈ −2.9 y for a male daily smoker |
| `/predict` docstring | HUQ050 example answer `"8"` — HUQ050 is a 0–5 visit *category* | a raw count lands outside the trained range |
| `linage2_service` on numpy ≥ 2.3 / pandas 3 | `float()` of a 1-element array; `foldOutliers` writes a read-only array | every request fails on an unpinned install |
| service imputed flags | LDL / urine ratio built from imputed parts reported as measured; questionnaire defaults silent | overstates what was measured |

`linage2_core.metta`'s CRP and cotinine comments carried the first two of those errors
and are corrected (they are what `GET /linage2/features` publishes).

**Regenerating.** Re-run the script against a new upstream commit; the golden test pins the
model file and the fixture to the same commit, so a half-regenerated pair fails loudly.

## 10. The "My Patient" tab — a patient someone types, for one session

**What it is.** A tab beside *PLN Query* where a person types a few lines —

```
58 year old male, current smoker
albumin 4.1 g/dL
HbA1c 6.4 %
CRP 3.1 mg/L
diagnoses: hypertension
```

— presses **Read** to see what was understood, **Build patient** to become `Caller_Me`, and
then asks about "me" in *PLN Query*. Nothing is stored: the patient is a plain dict in a
`gr.State` (one per browser session, gone on reload), and every chat turn rebuilds its atoms
into that one query's space exactly as `/query` does with a `patient` object — the shared
code is `core/patient_context.py`, and the LinAge2 atoms still go only to the LinAge2 space.
The *Download .metta* copy is written to the system temp directory, never to a folder the
KB loaders scan. `POST /patients/from-text` is the same reader over HTTP.

**Reading the text: the model reads, code checks, the person confirms** (`core/patient_read.py`,
`core/patient_extract.py`, `core/patient_text.py`; design in `docs/patient_extraction/design.md`).
With `OPENAI_API_KEY` set, **Read** (and `POST /patients/from-text`, `reader` "auto" or "model")
sends the text to a model (`PLN_EXTRACT_MODEL`, default `gpt-6-luna`) that returns one item per
fact: its kind, enum keys from `core.patient_vocabulary` (generated from the KB, drift-checked),
the number and unit **as written**, and the quote that holds them. Code checks each item against
the text — the quote is in it, the number is in the quote as a whole number ("5'1" is not in
"5'11"), a lab's, weight's or height's unit is in the quote, nothing inside a statement the model
says is about someone else is the person's — and drops what fails, listing why. A condition or a
smoking status that fails is asked about rather than dropped, since its absence would be read
as an answer. Text no item quotes is listed as not used. Without a key, or when the model cannot
be reached, the text is read as **canonical lines** (`core.patient_canonical`: one fact per
line, like the examples, the exact inverse of the line renderer, round-trip tested); anything
else is listed as not understood. The page's first reading, the examples and Build never call
the model; Build uses the reading stored at Read and refuses if the text changed since.

Both readers end in the same rules (`core.patient_text.assemble`), so **units are never
guessed**. A lab value in the wrong unit is the commonest way to get a confident, wrong
biological age (albumin 4.2 read as g/L is −30 g/L from the median): an unknown unit is
refused with the accepted ones; a value **without** a unit is taken only when exactly one known
unit puts it inside the middle 99% of NHANES adults (then shown as *unit assumed*) — `CRP 3.1`
stops and asks, since both mg/L and mg/dL fit; a value outside everything NHANES observed is
refused, naming the unit that would fit; hemoglobin 9.5 asks rather than being re-read as a
normal value in another unit; urea and urea nitrogen have their own mg/dL factors; a weight,
height or cotinine without its unit, and a GrimAge clock age given where an acceleration is
meant, are asked about. Two different values for one thing are a contradiction to fix. Smoking
follows the conventions the reader always had: cotinine level 0 for never and former, 1 for an
occasional and 2 for a moderate smoker, 3 otherwise; vaping, patches or smokeless tobacco need a
measured cotinine; cannabis is not tobacco; a measured cotinine always wins over words; a
smoking status the model cannot settle blocks the build (0 vs 3 is about 8.8 years). A list of
diagnoses answers every unlisted one No. A medication counts only for a drug the KB holds an
Interaction fact for (today metformin) and only when taken now; it becomes
`(CurrentMedication <id> <drug>)` in the **shared** space, appears in the supplement plan's
`Interactions`, and changes no ranking and no LinAge2 number.

This replaced a regex reader (about 3,200 lines with its model guards, and a 705-line medication
grammar for one drug) that four adversarial review rounds kept finding new failures in. The
model was right where the guards failed it: "A 35 year old female height 5'3'' inch weight 135
lbs" was read correctly by the model and thrown away by the guards. The live evaluation over the
old corpus, the round-4 reproductions and the body phrasings people type is
`docs/patient_extraction/eval_gpt-6-luna.md` (`scripts/eval_patient_extraction.py`).

**One value, two consumers.** CRP typed once is LinAge2's `LBXCRP` (mg/dL) *and* the KB's
`CRP` witness (mg/L); HbA1c likewise; a glucose is the `FastingGlucose` witness only when the
line says *fasting*; the stated smoking status is both the cotinine level (training bins) and
`PatientSmoking`. So causes can be credited without typing anything twice. The KB
standardises its witnesses against coarse pooled priors and refuses |z| > 12 as a unit
mistake, which a real HbA1c of 12 % exceeds; such a witness is passed at z 12 — Elevated is
all it needs to say — and the tab says so, while LinAge2 uses the value as typed.

**What the tab shows.** The read table (value as typed, LinAge2 value, check, and the
same-sex, same-age NHANES median as *Typical*); after Build, the biological age, the measured
labs adding and removing years, the count and total of filled-in inputs, LinAge2's own caveats
(an age outside 40-85 is flagged as an extrapolation under the headline; an unanswered
questionnaire is said to be assumed), the builder's notes reworded for someone who typed text
rather than sent z-scores, the atoms, and suggested questions that jump to *PLN Query* with the question filled in.
A patient none of whose values is both elevated and reached by a curated edge (CRP, HbA1c, a
glucose or triglycerides marked fasting, RDW and a low albumin against LinAge2's reference; typed diagnoses and every other lab count for none) is told that the
diagnosis will come back empty, the supplement plan will have no tiers and the ranking will be
the population's — in the build notes, and again in the chat next to any such answer, since the
chat never shows the builder's notes and an empty `()` reads as "no cause"
(`core.patient_context.patient_form_warnings`).
Verified in a headless browser: build, banner, suggested question, download, and the
unclear-unit refusal (`tests/test_patient_tab.py` covers the handlers and the chat wiring).

