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
written to disk; the atoms live in one request's space.

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
* **Four readouts**, `(MeasuresBiomarker <Input> <Biomarker>)`: `CRP → CRP`,
  `HbA1c → HbA1c`, `SerumGlucose → FastingGlucose` (with the caveat that the biochemistry
  glucose is fasting only if the draw was), `SerumCotinine → CurrentTobaccoExposure` (a
  new exposure node: cotinine is current exposure, not the cumulative pack-years
  `DNAmPACKYRS` surrogates).
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
(Contribution SerumAlbumin  (years 2.26) Measured (ReadsOut None) (DrivenBy ()))
…  (attributed-measured 7.67) (attributed-imputed 0.12) (age-term-residual 0.50)   ; = delta 8.28
```

Albumin's 2.26 years are the second-largest driver and are **left unexplained**: no bridge
reaches it. That is the honest state of the KB, not a gap the layer papers over.

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

Three decisions follow, each asserted in `tests/test_linage2.py`:

1. **Scoped, not shared.** The LinAge2 files are never in `_INFERENCE_STACK`.
2. **Minimal underneath.** `LINAGE2_PATIENT_STACK` drops `pln_intervention_ranking`,
   `pln_abductive_diagnosis`, the López-Otín intervention records and `nhanes_reference`
   — nothing a LinAge2 form reads — which is what leaves room for the generated baseline.
   The margin test (`+16` heads on top of a full baseline) guards it.
3. **Codes as comments.** The 59 NHANES codes cost more than the space had as string
   atoms; as structured comments they cost nothing and stay the single source of truth
   (`GET /linage2/features` publishes them).

## 6. The API surface

| call | what |
|---|---|
| `patient.linage2` on `/query`, `/metta/run`, `/patients/preview` | the `/predict` body, forwarded; becomes the `LinAgeAccel` marker + the contribution atoms |
| `GET /linage2/features` | the 59 inputs, codes, descriptions, readouts, how a cause is credited, the stack, whether a baseline is loaded |
| `POST /linage2/analyze` | LLM-free: decomposition, hazard, risk (if baseline), counterfactuals per lever, as JSON |
| `routed: "linage2"` | a `linage-*` form was validated against and run in the scoped space |
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
3. **Four of 59 inputs can be explained.** Albumin, RDW, blood pressure, NT-proBNP and the
   rest are carried as years. Bridges for them are curation work of the
   `mechanistic_bridges.metta` kind, and each new head symbol spends budget.
4. **No cause-specific risk.** The paper reports none per year; `CoronaryHeartDisease`
   through LinAge2 yields nothing rather than borrowing the all-cause hazard.
5. **Two clocks are not combined.** A patient with both GrimAge and LinAge2 gets a CHD
   risk from one and a mortality hazard from the other; multiplying them would double-count
   (`docs/risk_prediction.md §3`).
6. **Not in the chat UI's patient surface.** The Gradio chat routes and validates the forms
   identically, but has no caller-supplied patient yet, so a LinAge2 form typed there is
   honestly empty.

## 8. Non-goals

- Not a LinAge2 implementation. The model, its artifacts and its numpy pipeline stay in
  `Rejuve/LinAge2-Python`; this repository consumes the service's response.
- No per-feature weights in the KB (sex-specific, sign-flipping, retrain-sensitive).
- No numeric AUCs transcribed from figures (§1.3 of the evidence file says what is and is
  not recorded).
- No stored patients.
