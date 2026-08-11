# Risk-Prediction Chat Test Queries (Demo 5) — executed

**Status:** 10 natural-language queries driven through the REAL pipeline
(`translate()` → `validate()` → `run_query()` against live **hyperon 0.2.10**),
loading the exact `_ALL_KB_PATHS` set the Gradio app loads. The `ACTUAL` blocks are
verbatim hyperon output — no hypothetical values. Reproduce with:

```
PLN_RUNTIME_AVAILABLE=true python scripts/risk_prediction_chat_demo.py
```

In this sandbox `OPENAI_API_KEY` is unset, so the live LLM translation step reports
"key not set" and the driver falls back to the **reference MeTTa** each few-shot
targets (`pln_chat/prompts/few_shot_examples.json` + `system_prompt.txt` rule 12).
Every query still runs the real `validate()` and `run_query()`. Each query
`validate()`s clean (`valid=True issues=[]`) against the parsed inference-stack
registry, because `pln_risk_prediction.metta` is in `_INFERENCE_STACK`, so the
`predict-risk` / `risk-decomposition` / `project-risk` symbols are known.

These 10 are deliberately **different** from the counterfactual few-shots: they ask
for an *absolute risk*, a *confidence interval*, a *factor decomposition*, and
*projected risk under an intervention* — Demo 5's four headline deliverables — plus
the honesty cases (metabolic ~0, edge-less lever omitted) and a second patient.

Model note (v1, coarse): risk = `baseline(age,sex) × HR^(z_clock × sd→years)` with
`HR` read from the Lu 2019 `AgeAccelGrim → CHD` record (1.07/yr), `sd→years = 4.2`,
`baseline` a curated age/sex table, and confidence `= 0.85 × 0.9 = 0.765`. The
architecture is locked; the constants are documented tunables (see
`docs/risk_prediction.md`).

---

### Q1. What is Patient001's 10-year risk of coronary heart disease?
- **MeTTa:** `(predict-risk-patient &self Patient001)`
- **Routed behavior:** absolute point estimate + propagated CI + baseline + multiplier.
- **ACTUAL:**
  ```
  (RiskPrediction Patient001 CoronaryHeartDisease (point 0.12605177716424967)
     (ci 0.096429609530651 0.15567394479784832) (confidence 0.765)
     (baseline 0.08) (multiplier 1.5756472145531208))
  ```
- **Reads as:** P(CHD in 10y) ≈ **0.126, 95%-ish CI [0.096, 0.156], c = 0.765** —
  a 58yo male baseline of 0.08 raised ×1.58 by a +6.7-year GrimAge acceleration.

### Q2. Give me Patient001's cardiovascular risk with a confidence interval.
- **MeTTa:** `(predict-risk-patient &self Patient001)`
- **Routed behavior:** same estimate; the CI is `point × (1 ± (1−c)k)`, so a
  lower-confidence estimate would carry a proportionally wider interval.
- **ACTUAL:**
  ```
  (RiskPrediction Patient001 CoronaryHeartDisease (point 0.12605177716424967)
     (ci 0.096429609530651 0.15567394479784832) (confidence 0.765)
     (baseline 0.08) (multiplier 1.5756472145531208))
  ```

### Q3. Which biomarkers are driving Patient001's elevated cardiovascular risk, and by how much?
- **MeTTa:** `(risk-decomposition-patient &self Patient001)`
- **Routed behavior:** excess-over-baseline split by clock-component elevation share,
  with an explicit residual.
- **ACTUAL:**
  ```
  (RiskDecomposition Patient001 CoronaryHeartDisease (baseline 0.08)
     (total 0.12605177716424967) (excess 0.04605177716424967)
     (Factors ((RiskFactor DNAmPAI1  (contribution 0.225)  (share 0.140625)  (attributable-risk 0.00647603116372261))
               (RiskFactor DNAmGDF15 (contribution 0.1875) (share 0.1171875) (attributable-risk 0.005396692636435508))))
     (attributed-excess 0.011872723800158119) (residual-excess 0.03417905336409155))
  ```
- **Reads as:** of the **0.046 excess**, DNAmPAI1 accounts for **0.0065** and
  DNAmGDF15 for **0.0054** (attributed 0.0119); the remaining **0.0342** is an
  honest residual. `baseline + attributed + residual = total` exactly.

### Q4. How much of Patient001's excess heart-disease risk is left unexplained by their measured methylation surrogates?
- **MeTTa:** `(risk-decomposition-patient &self Patient001)`
- **Routed behavior:** the `residual-excess` field — the part the coarse v1 weights +
  unmeasured surrogates do not yet credit, reported, never forced to zero.
- **ACTUAL:** (same record as Q3) → `residual-excess 0.03417905336409155` of
  `excess 0.04605…` (≈ 74%).

### Q5. How much would clearing Patient001's senescent cells be expected to lower their 10-year heart-disease risk?
- **MeTTa:** `(project-risk-patient &self Patient001 CellularSenescence)`
- **Routed behavior:** Scenario B — reuse the counterfactual, lower the clock by the
  expected GrimAge reduction, recompute the risk.
- **ACTUAL:**
  ```
  (ProjectedRisk CellularSenescence CoronaryHeartDisease (point 0.1183478073576214)
     (reduction 0.007703969806628269) (delta-clock -0.22192968749999997)
     (confidence 0.31763845825852793) (Via (DNAmPAI1 DNAmGDF15)))
  ```
- **Reads as:** risk falls 0.126 → **0.118 (−0.0077)**, credited to both clock
  surrogates, at confidence **0.318**.

### Q6. For Patient001, does clearing senescent cells cut cardiovascular risk more than just reducing inflammation?
- **MeTTa:**
  `(project-risk-patient &self Patient001 CellularSenescence)`
  `(project-risk-patient &self Patient001 ChronicInflammation)`
- **Routed behavior:** the B-vs-A comparison in absolute-risk terms.
- **ACTUAL:**
  ```
  (ProjectedRisk CellularSenescence   … (reduction 0.007703969806628269) … (Via (DNAmPAI1 DNAmGDF15)))
  (ProjectedRisk ChronicInflammation  … (reduction 0.004290792412020641) … (Via (DNAmGDF15)))
  ```
- **Reads as:** **reduction_B 0.0077 > reduction_A 0.0043** — senescence is upstream
  and reaches both clock surrogates, so it buys the bigger CHD-risk drop.

### Q7. If Patient001 took metformin for metabolic health, how much would their cardiovascular risk drop?
- **MeTTa:** `(project-risk-patient &self Patient001 Metformin)`
- **Routed behavior:** metformin resolves to InsulinResistance, whose axis reaches no
  clock surrogate → honest ~0 clock-based change (metformin's direct metabolic-axis
  effect on CHD is the *intervention-ranking* demo, not this clock-based risk model).
- **ACTUAL:**
  ```
  (ProjectedRisk Metformin CoronaryHeartDisease (point 0.12605177716424967)
     (reduction 0.0) (delta-clock 0.0) (confidence 0.0) (Via ()))
  ```
- **Reads as:** **reduction 0.0, empty Via** — not a fabricated number.

### Q8. How certain is the estimate that clearing senescence lowers Patient001's CHD risk, compared with reducing inflammation?
- **MeTTa:**
  `(project-risk-patient &self Patient001 CellularSenescence)`
  `(project-risk-patient &self Patient001 ChronicInflammation)`
- **Routed behavior:** uncertainty grows with route length.
- **ACTUAL:** senescence `confidence 0.31763…` (2–3 hops) **<** inflammation
  `confidence 0.65` (single hop). The bigger-effect estimate is the less certain one.

### Q9. What would Patient001's projected heart-disease risk be if we gave them elamipretide?
- **MeTTa:** `(project-risk-patient &self Patient001 Elamipretide)`
- **Routed behavior:** no Effect edge → counterfactual unresolvable → projection
  omitted (`run_query status=empty`).
- **ACTUAL:** `(empty — no result)` — omitted, never a fabricated placement.

### Q10. What is Patient002's 10-year cardiovascular risk?
- **MeTTa:** `(predict-risk-patient &self Patient002)`
- **Routed behavior:** a different patient — 61yo male → the 60–69 baseline band
  (0.15), clock z 1.3 — same model, personalized inputs.
- **ACTUAL:**
  ```
  (RiskPrediction Patient002 CoronaryHeartDisease (point 0.2170334555640622)
     (ci 0.1660305935065076 0.2680363176216168) (confidence 0.765)
     (baseline 0.15) (multiplier 1.4468897037604147))
  ```
- **Reads as:** P(CHD in 10y) ≈ **0.217, CI [0.166, 0.268]** — the identical machine
  yields a different, higher estimate off a higher baseline band and a lower clock.

---

**Summary.** All 10 execute on live hyperon 0.2.10 and validate clean. The four
Demo-5 deliverables — point estimate (Q1), confidence interval (Q1/Q2),
factor decomposition (Q3/Q4), projected risk under interventions (Q5–Q8) — plus the
two honesty cases (metabolic ~0 in Q7, edge-less lever omitted in Q9) and the
second-patient personalization (Q10) all reproduce their intended behavior against
the real pipeline.
