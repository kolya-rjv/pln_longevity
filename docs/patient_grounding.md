# Patient Grounding Layer — Design (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`patient_profile.metta`; regression tests: `tests/test_patient_grounding.py`
(20 tests). Wired into the chat app's inference stack (`pln_chat/app.py`) with
`diagnose-patient` / `rank-interventions-for-patient` few-shots.
**Scope:** turn a patient's **raw, standardized biomarker measurements** into the
**symbolic findings** the inference demos consume, and wire those findings into
abductive diagnosis (Demo 1) and intervention ranking (Demo 2). This is the step
from *synthetic* (hand-typed marker lists) to *personalized*.

> **Dependency / base branch.** Pure consumer of the whole inference stack —
> `infer` / `diagnose` (`pln_deduction.metta`, `pln_abductive_diagnosis.metta`)
> and `rank-score` / `scored` / `sort-scored`
> (`pln_intervention_ranking.metta`). Load **last** (see the file header for the
> exact order). It adds no new inference power; it grounds the inputs.

---

## 1. Why we need it

Every prior demo is **population-level**: `diagnose` and `rank-interventions`
take an explicit, hand-typed list of "observed" markers or candidate
interventions. Both demo docs name the same missing piece as the gap to
*personalized* analysis:

- `abductive_diagnosis.md` §5.3: *"observations are passed explicitly; wiring
  them to a real … profile … turns the demo from synthetic to personalized."*
- `intervention_ranking.md` §5.2: the ranking *"is currently population-level.
  Demo 2 in full multiplies in patient-specific factors (`f(patient factors)`) …
  needs the still-absent NHANES / patient-profile layer."*

This layer is that piece.

## 2. Where it sits in the pipeline

```
raw standardized measurements ─ground─▶ Elevated/Normal/Low findings ─┐
   (MeasuredZ patient marker z)                                        ├─▶ diagnose-patient   (Demo 1)
                                                                       └─▶ rank-…-for-patient  (Demo 2)
```

It **reads** a patient's measurements and **emits** the findings inference needs.
It never mutates a measurement — same immutability contract as the calibration
layer (`calibration_layer.md` §1).

## 3. The data model — one standardized number per marker

A patient is a set of raw, source-faithful measurements:

```metta
(: MeasuredZ (-> PatientProfile Biomarker Number Atom))
(MeasuredZ Patient001 DNAmPAI1  1.8)   ; ~1.8 SD above the age/sex-adjusted norm
(MeasuredZ Patient001 HorvathAgeAccel -0.2)
```

`z` = number of SDs above/below the **age- and sex-adjusted population mean** for
that marker. This one convention is deliberate:

- it is exactly what **age acceleration** already is for an epigenetic clock, and
  what a **standardized lab value** is for a blood marker — so clocks and blood
  panels become commensurable under a single scale;
- it is the number **NHANES / a blood panel actually yields**, so the eventual
  NHANES import is a data-loading step, not a redesign;
- it makes the *magnitude* of dysregulation available (how many SDs), which the
  personalization factor (§5) weighs — and which the **counterfactual layer** now
  scales its expected GrimAge deltas by (`docs/counterfactual_analysis.md`).

Demographics (`PatientAge` / `PatientSex` / `PatientSmoking`) are stored raw but
**not yet reasoned over** — they are the hooks for the sex/age-specific modifiers
(`pipeline.md` App. A.7), present so a profile is complete.

## 4. The grounding rule — the patient-layer analogue of `calibrate-tv`

One documented, tunable threshold turns a number into a qualitative finding:

```metta
(= (elevated-z-threshold) 1.0)          ; the single knob
(= (z->status $z) …)                     ; z ↦ Elevated / Normal / Low
```

This is the patient-layer counterpart of `calibrate-tv`: the **one place** raw
numbers become the symbols inference consumes, so tuning the personalization
loop is a one-function change (the `calibration_layer.md` §8 discipline). A marker
with no measurement yields **no** status — a `Normal` is never invented.

`patient-observations` collapses a patient's **elevated** markers into the
observation tuple `diagnose` expects. Normal/Low markers are filtered out — this
is *grounding*, not a raw dump.

## 5. What it wires up

**Demo 1 — `diagnose-patient <space> <patient> <hypotheses>`.** Pulls the
patient's elevated findings automatically and runs the existing abductive
`diagnose`. On the case-study patient (`Patient001`, `docs/casestudy.md`) it
reproduces the population ranking straight from measured values:

```metta
!(diagnose-patient &self Patient001
    (CellularSenescence MitochondrialDysfunction ChronicInflammation))
;; CellularSenescence (cov 3) ≻ ChronicInflammation (cov 2) ≻ MitochondrialDysfunction (cov 1)
```

`AgeAccelGrim` is elevated too, but no bridge explains it, so it is **carried in
the observation set yet credited to no hypothesis** — grounded, never invented.
(Abductive diagnosis still leaves it un-credited; the composition edge added by the
**counterfactual layer** now *does* credit it — `decompose-grimage` splits the
elevated GrimAge into its DNAm-surrogate components and traces each to its upstream
cause, see `docs/counterfactual_analysis.md` §3.2. The **risk-prediction layer**
(Demo 5) goes one step further and *uses* this same elevated `AgeAccelGrim` as the
hazard predictor for an absolute 10-year CHD risk — `docs/risk_prediction.md`.)

**Demo 2 — `rank-interventions-for-patient <space> <patient> <pool> <outcome>`.**
The `f(patient factors)` the ranking doc asked for. Each candidate's population
protective score on the outcome is added to its **patient-relevance**: how much
it reduces the markers *this* patient presents elevated (reusing `infer` +
`rank-score`, with the same `Neg = protective` convention). On `Patient001`:

```metta
!(rank-interventions-for-patient &self Patient001
    (DasatinibPlusQuercetin Fisetin Spermidine Elamipretide) CoronaryHeartDisease)
;; D+Q (0.231) ≻ Fisetin (0.165) ≻ Spermidine (0.071)   [Elamipretide omitted: no chain]
```

The senolytics target exactly this patient's elevated senescence axis, so their
scores rise sharply over the population baseline (D+Q `0.043 → 0.231`). The
combination is **additive and auditable**: `personalized = population + relevance`
(the full signed TV is kept in each `scored` tuple).

## 6. Open questions / next increments

1. **NHANES import.** `MeasuredZ` is the exact shape a NHANES age-acceleration /
   standardized-blood-panel row provides. The next step is an ETL that emits
   `MeasuredZ` atoms per participant, so patients come from data, not a curated
   example. (Ties into roadmap next-step #2/#3 — ETL→inference wiring.)
2. **Multiplicative `f(patient factors)`.** `pipeline.md` §4.2 writes the patient
   factor as a *product*. v1 is additive (transparent, monotone — it can only
   promote an intervention that also helps this patient, never demote a
   population-protective one below a harmful one). A multiplicative form is the
   documented refinement.
3. ~~**Reordering across patients.**~~ ✅ **Done** — a second, independent
   **metabolic axis** (`mechanistic_bridges.metta`: `DeregulatedNutrientSensing →
   InsulinResistance → FastingGlucose/HbA1c → CHD`; `Metformin`/`Berberine ⊣
   InsulinResistance`) plus a metabolic-dominant **`Patient002`** now make
   personalization *reorder*, not just amplify: from the **same pool against the
   same outcome**, `Patient001` ranks `DasatinibPlusQuercetin` first and
   `Patient002` ranks `Metformin` first — the "personalization changes the
   decision" demo (regression-guarded in `tests/test_patient_grounding.py`).
   Remaining: bridge more axes (mito, inflammation-primary) and let a patient
   present *multiple* co-elevated axes.
4. **Discordance interpretation.** The profile already encodes the normal-Horvath
   / elevated-GrimAge discordance; the `pipeline.md` App. A.4 discordance rules
   (e.g. `(Normal Horvath) ∧ (Elevated GrimAge) ⇒ mortality risk independent of
   first-gen aging`) could be added as their own inference on top of
   `patient-status`.
5. **Contradicting evidence.** A hypothesis whose chain would *lower* a marker the
   patient presents *elevated* is currently just not credited; grounding now makes
   it possible to treat that as evidence *against* (shared with
   `abductive_diagnosis.md` §5.1).
6. **Per-marker reference policy.** v1 uses one global z-threshold; some markers
   may warrant marker-specific cutoffs (e.g. `DunedinPACE > 1.0`), which fits as a
   per-marker override of `elevated-z-threshold`.

## 7. Non-goals

- Not a NHANES importer yet (v1 ships one grounded example patient).
- Not numerically-tuned thresholds (the `1.0` SD cutoff is a documented knob).
- Not new inference machinery — it grounds inputs to the existing engine.
