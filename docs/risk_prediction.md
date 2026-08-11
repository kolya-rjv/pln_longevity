# Risk Prediction with Full Uncertainty — Demo 5 (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`pln_risk_prediction.metta`; regression tests in `tests/test_risk_prediction.py`
(14 tests); executed chat queries in `docs/risk_prediction_chat_test_queries.md`.
**Scope:** the capstone demonstration scenario from `pipeline.md` (§4.5, Demo 5) —
turn a grounded patient into a personalized **absolute** cardiovascular risk with a
**propagated confidence interval**, a **decomposition** into attributable factors,
and **projected risk under intervention scenarios**.

> **Dependency / base branch.** Sits on the whole inference + patient + counterfactual
> stack. It reuses, rather than re-derives: the Lu 2019 `AgeAccelGrim → CHD` hazard
> record (`grim_age_lu2019_evidence.metta`), `calibrate-tv` (calibration layer),
> `MeasuredZ` / `PatientAge` / `PatientSex` (`patient_profile.metta`), and — for the
> factor decomposition and the intervention projection — `decompose-grimage` and
> `counterfactual-patient` from Demo 4 (`pln_counterfactual.metta`). Load **last**
> (see the file header for the exact order).

---

## 1. Why this is the demo to build now

Demos 1–4 gave us every ingredient of a real risk number except the number itself:
a signed, uncertainty-quantified transmission to CHD (`infer`), a patient grounded
in standardized measurements (`MeasuredZ`), a GrimAge decomposition, and a
counterfactual do-operator. Demo 5 is the smallest step that composes them into the
**one output a clinician asks for** — `P(CHD in 10y)` for *this* patient — and does
it with the four things `pipeline.md` §4.5 lists and an LLM cannot deliver:

1. a numeric **point estimate**, not hedged language;
2. a **confidence interval** propagated from the source-evidence confidence;
3. a **decomposition** into contributing factors that sum to the excess;
4. **projected risk** under hypothetical interventions.

It is also the demo that shows the *reuse* the architecture was built for: the
decomposition is `decompose-grimage`, and the intervention projection is
`counterfactual-patient`, both consumed unchanged.

## 2. What it computes

```
predict-risk-patient      : space × patient          ⊢ (RiskPrediction … (point p) (ci lo hi) (confidence c) …)
risk-decomposition-patient: space × patient          ⊢ (RiskDecomposition … (Factors (…)) (residual-excess r))
project-risk-patient      : space × patient × lever  ⊢ (ProjectedRisk … (point p') (reduction Δ) …)   (omitted if lever unreachable)
```

The core is a single, standard **proportional-hazards** step:

```
risk = baseline(age, sex) × HR_clock ^ (z_clock × sd→years)
```

- **`baseline(age,sex)`** — a curated 10-year absolute-risk table (§0(a), from
  `pipeline.md` App. A.8).
- **`HR_clock`** — read straight off the empirical record `AgeAccelGrim → CHD`
  (Lu 2019, HR 1.07 per year), never hand-typed. This record is **already in the KB
  and already read by `link-effect`/`infer`**, so Demo 5 adds no new causal edge and
  the graph the rest of the pipeline traverses is untouched.
- **`z_clock`** — the patient's measured GrimAge acceleration in SDs (`MeasuredZ`).
- **`sd→years`** — one documented conversion knob (§0(b)); ~4.2 y/SD reproduces the
  case study's "+6.7 years" for Patient001's 1.6 SD.

## 3. The composite-clock choice (why this is PLN-not-LLM)

The one real modeling decision is *which* predictor to hazard-multiply. GrimAge has
eight DNAm components, several of which (DNAmPAI1, DNAmGDF15) have their own hazard
associations. The naive move — the one an LLM makes — is to multiply **every** hazard
ratio it can name: `HR_GrimAge × HR_PAI1 × HR_GDF15 × …`. That **double-counts**,
because `DNAmPAI1 ⊂ GrimAge`: the component's risk is *already inside* the clock's.

Demo 5 declines. It hazard-multiplies the **composite clock once**, and recovers the
component-level story exactly once more, as the **decomposition** — each component's
*share* of the single clock risk, not an independent extra multiplier. This is the
same restraint Demo 2 shows by omitting Elamipretide and Demo 4 shows by routing CRP
through its shared cause: the model does the causally-correct thing, not the
superficially-more-detailed thing.

## 4. The four deliverables, with real numbers (Patient001)

Patient001: 58yo male, `AgeAccelGrim` z = 1.6 → ~6.72 years, baseline 0.08, HR 1.07.

**(1) Point estimate + (2) confidence interval.**
```
!(predict-risk-patient &self Patient001)
(RiskPrediction Patient001 CoronaryHeartDisease
   (point 0.12605) (ci 0.09643 0.15567) (confidence 0.765)
   (baseline 0.08) (multiplier 1.57565))
```
`P(CHD in 10y) ≈ 0.126, CI [0.096, 0.156], c = 0.765`. Confidence = the evidence
confidence (`MultipleHumanTrials` → 0.85) × one modeling discount (0.9). The interval
is `point × (1 ± (1−c)k)`, so **a lower-confidence estimate carries a wider
interval** — uncertainty is a first-class output, not a caveat.

**(3) Decomposition into contributing factors.**
```
!(risk-decomposition-patient &self Patient001)
(RiskDecomposition Patient001 CoronaryHeartDisease (baseline 0.08) (total 0.12605) (excess 0.04605)
   (Factors ((RiskFactor DNAmPAI1  (contribution 0.225)  (share 0.14063) (attributable-risk 0.00648))
             (RiskFactor DNAmGDF15 (contribution 0.1875) (share 0.11719) (attributable-risk 0.00540))))
   (attributed-excess 0.01187) (residual-excess 0.03418))
```
The excess over baseline (0.046) is apportioned to the elevated clock components by
their elevation share; `baseline + attributed + residual = total` **exactly**. The
coarse v1 weights and the unmeasured surrogates leave an **explicit positive
residual** (0.0342, ~74% of the excess) — reported, never forced to zero.

**(4) Projected risk under interventions** (reuses `counterfactual-patient`):
```
!(risk-scenarios &self Patient001)
   B CellularSenescence   → point 0.11835  reduction 0.00770  c 0.318  Via (DNAmPAI1 DNAmGDF15)
   A ChronicInflammation  → point 0.12176  reduction 0.00429  c 0.65   Via (DNAmGDF15)
   C InsulinResistance    → point 0.12605  reduction 0.0      c 0.0    Via ()
```
The Demo-4 ordering reappears in absolute-risk terms: **clearing senescence (B) cuts
CHD risk more than reducing inflammation alone (A)** — because senescence is upstream
and reaches both clock surrogates — but the bigger effect is held at **lower
confidence** (its surrogates sit further downstream). The metabolic lever (C) reaches
no clock surrogate, so it honestly projects **~0** change; an edge-less lever
(`Elamipretide`) is **omitted** entirely, not given a fabricated number.

## 5. Implementation notes (hyperon 0.2.10)

- **`pow-math` for the hazard exponent.** `risk = p0 × pow-math(HR, z × sd→years)`.
  Grounded ops do not evaluate their arguments, so the exponent is `let*`-forced to a
  concrete float before `pow-math` — the same discipline every layer documents.
- **Space threaded explicitly.** A bare `(match &self …)` inside this imported module
  would read only its own space; every read passes the caller's `$space`
  (`calibration_layer.md` §10.1).
- **Determinism.** `baseline-risk-chd` is a single rule + nested `if`, so identical
  inputs give identical output (`pipeline.md` Table 2). Tests compare floats with
  tolerance (trailing-noise-safe), as the other suites do.
- **No new causal edge.** Demo 5 reads the *existing* `AgeAccelGrim → CHD` record and
  adds only rules + curated constants — the `Effect`/empirical graph `infer`
  traverses is unchanged, so all 93 pre-Demo-5 tests stay green (107 total after this
  layer's 14).
- **Chat is cheap here** (as for Demo 4, unlike the DrugAge slice). `pln_risk_prediction.metta`
  is small and on the focused senescence/CHD stack, well under `PLN_MAX_KB_FILE_BYTES`,
  so it rides the ordinary `run_query`/`_ALL_KB_PATHS` path — added to `_INFERENCE_STACK`
  so the translator sees `predict-risk`/`risk-decomposition`/`project-risk` and the
  symbol validator accepts them; no scoped routing needed (`app.py`, few-shots,
  `system_prompt.txt` rule 12).

## 6. Open questions / next increments

1. **Calibrate the constants.** `sd→years`, the baseline table, the confidence
   discount, and `risk-ci-k` are curated v1 priors; fit them against a real cohort
   (NHANES linked mortality/CHD) and check interval calibration. v1 locks the
   architecture, not the numbers.
2. **Log-hazard decomposition.** The excess is apportioned *linearly* by clock share;
   the hazard is multiplicative, so a log-hazard (or Shapley) attribution is the
   principled refinement. The linear split is a defensible v1 "attributable share".
3. **Multi-predictor risk without double-counting.** To bring in genuinely
   independent axes (e.g. the metabolic FastingGlucose → CHD axis, or an inflammatory
   CRP → CHD hazard), add their *own* calibrated `→ CHD` evidence records and combine
   as independent hazards — but only once collinearity with the clock is handled
   (this is why v1 uses the single composite clock; see §3).
4. **True intervals, not a heuristic band.** The CI is a coarse `point × (1 ± (1−c)k)`;
   a proper propagation would carry the strength/confidence distribution through the
   hazard transform (Monte-Carlo or a delta-method on log-HR).
5. **Multi-outcome risk.** Extend from CHD to a basket (all-cause mortality, CHF —
   both have Lu 2019 hazard records already in the KB) and aggregate.
6. **Multi-lever projection.** Project risk under a *combination* of interventions,
   which needs the counterfactual layer's own multi-lever follow-up
   (`docs/counterfactual_analysis.md`).

## 7. Non-goals

- Not a numerically-validated clinical risk calculator — v1 locks the architecture,
  not the constants; the numbers are coarse and documented as tunable.
- Not a new causal edge or a rewrite of any evidence atom (the honesty invariant):
  the hazard ratio and the patient measurements are read, never materialized.
- Not the supplement-recommendation demo (Demo 6), which needs a supplement knowledge
  layer that does not yet exist.
