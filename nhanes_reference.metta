;; nhanes_reference.metta
;; NHANES survey-weighted reference distributions — TYPES AND RULES ONLY.   [v1]
;;
;; This file is the reference half of the raw -> z path. `patient_profile.metta`
;; holds a patient's RAW lab value as (MeasuredRaw <patient> <marker> <raw>); this
;; file holds the age-band × sex population distribution that raw value is measured
;; against, and the arithmetic that turns the pair into the standardized z the whole
;; inference stack already consumes.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; NO NUMBERS LIVE HERE, AND THAT IS THE POINT.
;; ─────────────────────────────────────────────────────────────────────────────
;; The CDC hosts (wwwn.cdc.gov, ftp.cdc.gov) are denied by this environment's egress
;; policy, so no NHANES value could be downloaded, computed *or verified* here. A
;; number written into a .metta file is consumed by inference and becomes a claim, so
;; this file contains types, accessors and rules exclusively — not one mean, not one
;; SD, not an "illustrative" cell behind a TODO. The numeric ReferenceDistribution
;; records are ETL build artifacts: `nhanes_reference_etl.py` writes them under
;; build/ (gitignored) from microdata the user supplies, exactly as the HAGR ETLs
;; already stage their output. Tests prove the machinery against synthetic fixtures
;; whose values are arbitrary by construction. See docs/nhanes_integration.md §2.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; LOAD ORDER — read this before adding the file to a KB list.
;; ─────────────────────────────────────────────────────────────────────────────
;; This file sits ON TOP of the patient layer: it consumes nothing from it, but the
;; patient layer's `derived-z` calls `age-band` and `standardize-z` from here. That
;; is a genuine cycle in the file graph, and it is resolved by SPLIT, not by
;; reordering:
;;
;;   * the ACCESSORS that a patient query touches — MeasuredRaw, patient-markers,
;;     derived-z, patient-z and the tuple helpers — live in patient_profile.metta;
;;   * the REFERENCE RECORD TYPE and the rules over it — ReferenceDistribution, the
;;     Ref* field signatures, the age-band and scale symbols, reference-cell and
;;     standardize-z — live here.
;;
;; The split is safe because `derived-z` matches (MeasuredRaw …) FIRST and yields
;; nothing when the patient has no raw value. With this file absent, that match is
;; always empty, so `age-band` / `standardize-z` are never reached and the patient
;; layer behaves exactly as it did before this change — which is what lets the 194
;; pre-existing tests keep their KB_FILES lists untouched (contract D5).
;;
;; The alternative — loading this file BEFORE patient_profile.metta — was rejected:
;; it would make every existing KB_FILES list wrong, and those test files are
;; additive-only.
;;
;; So: load this file AFTER patient_profile.metta (anywhere after it is fine), e.g.
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile
;;   nhanes_reference                    (this file)
;;   [build/nhanes_reference.metta]      (the generated numeric records, if present)
;;
;; STATUS: v1, verified against hyperon 0.2.10. Uses only core MeTTa builtins
;; (match, superpose, let/let*, +,-,*,/,>,<,==, log-math). As in every layer the
;; space $space is threaded explicitly (never a bare &self — a (match &self …)
;; inside an imported module reads only that module's own space, the cross-module
;; gotcha docs/calibration_layer.md §10.1 documents), and every arithmetic
;; intermediate is let*-forced because grounded ops do NOT evaluate their arguments.

;; ================================================================
;; 1. Age bands
;; ================================================================
;; Symbols are plain (Age_lt50, not Age<50) — MeTTa tokenizes < and > .
;; These bands are the SINGLE shared definition of the age axis. They must agree
;; with two other places, and both agreements were checked by reading the source:
;;   * nhanes_common.AGE_BANDS — the ETL's binning, half-open [lo, hi):
;;       ("Age_lt50", -inf, 50) ("Age_50_59", 50, 60) ("Age_60_69", 60, 70)
;;       ("Age_70p", 70, +inf)
;;   * pln_risk_prediction.metta's `baseline-risk-chd`, whose curated baseline table
;;     switches at (< $age 50), (< $age 60), (< $age 70) — the same three cutpoints.
;; A patient therefore lands in the same band whichever layer asks, and a future
;; data-backed baseline (step 2) keys off the same symbols as the reference cells.
(: AgeBand Type)
(: Age_lt50  AgeBand)
(: Age_50_59 AgeBand)
(: Age_60_69 AgeBand)
(: Age_70p   AgeBand)

;; age -> band. Half-open [lo, hi), identical to nhanes_common.age_band. Single
;; rule + nested if, so it is deterministic (one input -> one output), matching the
;; baseline-risk-chd style it must stay in lockstep with.
(= (age-band $age)
   (if (< $age 50) Age_lt50
       (if (< $age 60) Age_50_59
           (if (< $age 70) Age_60_69
               Age_70p))))

;; ================================================================
;; 2. Measurement scale  (contract D17 / docs/nhanes_integration.md §4)
;; ================================================================
;; A marker's reference moments, and every z taken against them, live on ONE
;; declared scale. This is not a refinement — it changes the arithmetic. z->status
;; in patient_profile.metta uses a symmetric ±1 SD cutoff, and on the RAW scale of a
;; log-normal analyte that cutoff does not blur the `Low` category, it deletes it:
;; on a simulated CRP distribution (geometric mean 0.2 mg/dL, GSD 3) P(z < -1) is
;; 0.00% raw versus 15.86% on log10, and raising the threshold does not help.
;;
;;   Identity — roughly symmetric analytes (HbA1c, total cholesterol): z on the raw
;;              value.
;;   Log10    — right-skewed analytes (CRP, triglycerides, insulin): the reference
;;              mean/SD are computed on log10(value), so the z must be too.
;;
;; The record carries its own scale in (RefScale …), and `standardize-z` reads it
;; from the record rather than from a table here. That is deliberate: it makes the
;; D17 failure mode — a log-scaled reference applied to a raw value — unrepresentable
;; rather than merely discouraged. There is no code path that computes a z from a
;; mean and SD without first reading the scale those two numbers were computed on.
(: MarkerScale Type)
(: Identity MarkerScale)
(: Log10    MarkerScale)

;; ================================================================
;; 3. The ReferenceDistribution record  (Schema A — FIXED)
;; ================================================================
;; Source-faithful style: a typed individual plus one field per atom, exactly as
;; grim_age_lu2019_evidence.metta records an evidence row. `nhanes_reference_etl.py`
;; emits records of precisely this shape:
;;
;;   (: NHANESRef_CRP_Male_Age_50_59_1999_2002 ReferenceDistribution)
;;   (RefMarker         NHANESRef_CRP_Male_Age_50_59_1999_2002 CRP)
;;   (RefSex            … Male)
;;   (RefAgeBand        … Age_50_59)
;;   (RefScale          … Log10)
;;   (RefMean           … <number>)      ; on RefScale
;;   (RefSD             … <number>)      ; on RefScale
;;   …
;;
;; Provenance is mandatory and single-tier (contract D2): every record says which
;; cycles, which weight variable, which source variable and which assay lot produced
;; it, and carries (RefProvenance … NHANES_Microdata). There is deliberately no
;; "published" or "transcribed" tier, because adding one invites someone to fill it.
;; No design-based standard error is emitted (contract D11) — a correct NHANES SE
;; needs Taylor linearization over the strata/PSU design variables, so the record
;; ships the unweighted n instead of a wrong SE.
(: ReferenceDistribution Type)

;; --- identifying fields (the lookup key) -------------------------------------
(: RefMarker  (-> ReferenceDistribution Biomarker Atom))
(: RefSex     (-> ReferenceDistribution Sex Atom))
(: RefAgeBand (-> ReferenceDistribution AgeBand Atom))

;; --- the distribution itself, ON RefScale -------------------------------------
(: RefScale (-> ReferenceDistribution MarkerScale Atom))
(: RefMean  (-> ReferenceDistribution Number Atom))
(: RefSD    (-> ReferenceDistribution Number Atom))

;; --- reliability + provenance --------------------------------------------------
(: ReferenceProvenance Type)
(: NHANES_Microdata ReferenceProvenance)          ; the ONLY tier in v1 (D2)

(: RefUnweightedN    (-> ReferenceDistribution Number Atom))
;; Design-based precision, present only when the source demographics file carried the
;; masked variance stratum and PSU. A naive standard error would ignore clustering (which
;; inflates variance) and stratification (which deflates it), so the emitter computes the
;; Taylor-linearized SE of the weighted mean when it can and emits NOTHING when it cannot
;; — an absent SE is honest, a naive one is not. Nothing here reads these yet; the natural
;; consumer is the calibration layer, widening confidence for an imprecise cell instead of
;; the current hard suppression on an unweighted count.
(: RefDesignSE       (-> ReferenceDistribution Number Atom))
(: RefDesignDF       (-> ReferenceDistribution Number Atom))
(: RefUnit           (-> ReferenceDistribution String Atom))
(: RefWeightVariable (-> ReferenceDistribution String Atom))
(: RefSourceVariable (-> ReferenceDistribution String Atom))
(: RefSourceFiles    (-> ReferenceDistribution String Atom))
(: RefSourceCycles   (-> ReferenceDistribution String Atom))
(: RefAssayLot       (-> ReferenceDistribution String Atom))
(: RefProvenance     (-> ReferenceDistribution ReferenceProvenance Atom))

;; --- conditional fields ---------------------------------------------------------
;; A named published correction applied to the value (contract D24) — e.g. the
;; 1999-2000 serum-creatinine Deming correction. An unlabelled derived number is
;; exactly the failure this repo's calibration discipline exists to prevent, so the
;; transform is named in the record rather than folded silently into the mean.
(: RefTransform (-> ReferenceDistribution String Atom))

;; Epigenetic-clock records only (contract D20). A clock acceleration is either the
;; RESIDUAL of clock age on chronological age (what Lu 2019's HR 1.07/year was fit
;; to, and the default) or the plain DIFFERENCE. Emitting one where the other is
;; meant silently mis-scales pln_risk_prediction.metta's exponent, so the definition
;; is recorded, never assumed.
(: AccelDefinition Type)
(: Residual   AccelDefinition)
(: Difference AccelDefinition)
(: RefAccelDefinition (-> ReferenceDistribution AccelDefinition Atom))
(: RefClockColumn     (-> ReferenceDistribution String Atom))

;; ================================================================
;; 4. Reference lookup
;; ================================================================
;; The record id for one (marker, band, sex) cell, or NOTHING.
;;
;; "Nothing" is the whole design. There is no default mean, no fallback to a
;; neighbouring band, no pooled "all adults" cell, and no `Normal`. A patient whose
;; band/sex has no reference cell — because the cell was suppressed for a small
;; unweighted n (contract D9), or because that marker was never run — simply gets no
;; z, and therefore no status. That mirrors `patient-status`, which already yields
;; nothing for an unmeasured marker: the layer declines rather than invents.
;;
;; SINGLE-VALUED BY CONSTRUCTION. A bare `match` here is non-deterministic: if two
;; ReferenceDistribution records share a marker/band/sex it yields BOTH, and everything
;; downstream then produces two z values for one measurement. That is not hypothetical —
;; loading two generated reference files (say a 1999-2002 run alongside a 2011-2014 one)
;; puts two cells on the same key, and no emitter can prevent it, because each run only
;; sees its own cells.
;;
;; Two values here are worse than one arbitrary value. `patient-relevance` SUMS over the
;; markers a patient presents elevated, and `diagnose` counts coverage over the same
;; observation set, so a duplicated derivation silently inflates an intervention's score
;; and a hypothesis's coverage — exactly the double-counting `patient-z`'s precedence rule
;; exists to prevent for the MeasuredZ-versus-derived case, which covered only that case.
;;
;; So the matches are collapsed — and then, if there is more than one, this REFUSES.
;;
;; An earlier version took the first cell instead. That was wrong, and measurably so.
;; The methylation ETL emits a record per acceleration convention, and the two differ
;; only in `RefAccelDefinition`, which this lookup does not key on. Taking the first
;; therefore let the answer depend on the order records happened to appear in a
;; generated file: the same patient, same query, came out as z 1.12 or z 2.51 — a 48%
;; swing in the reported ten-year risk — purely by text order. Collapsing to one value
;; satisfied the letter of single-valuedness while turning an ambiguity into a confident
;; wrong answer, which is the exact failure this repository exists to prevent.
;;
;; So ambiguity yields NOTHING, like every other thing this layer cannot resolve. A
;; caller that wants a specific cell must disambiguate at the source: run the clock ETL
;; with a single `--accel-definition`, or do not load overlapping reference files.
(= (reference-cell $space $marker $band $sex)
   (let $cells (collapse
                  (match $space
                     (, (RefMarker  $rid $marker)
                        (RefAgeBand $rid $band)
                        (RefSex     $rid $sex))
                     $rid))
      (if (== $cells ())
          (superpose ())                                   ; no cell -> nothing
          ;; `==` does not evaluate its arguments, so the tail must be let*-forced to a
          ;; concrete tuple before it is compared — the grounded-op discipline every
          ;; layer here documents. Comparing the unreduced (cdr-atom …) silently fails
          ;; the test and would make even an unambiguous cell yield nothing.
          (let $rest (cdr-atom $cells)
             (if (== $rest ())
                 (car-atom $cells)                         ; exactly one -> use it
                 (superpose ()))))))                       ; ambiguous -> nothing

;; ================================================================
;; 5. Standardization  —  z = (value - mean) / sd, ON THE RECORD'S SCALE
;; ================================================================
;; The scale, mean and SD all come from the SAME record, so a Log10 reference can
;; only ever be applied to a log10-transformed value (contract D17).
;;
;; THE LOG GUARD IS ON THE INPUT, NOT THE OUTPUT (contract D27 — measured, critical).
;; hyperon 0.2.10 DOES provide a logarithm: (log-math 10 $x) is log10. (ln-math,
;; log10-math and exp-math do not exist — they return unevaluated.) But log-math
;; fails SILENTLY, and the downstream failure is a fabricated clinical finding:
;;
;;     !(log-math 10 0)               =>  -inf
;;     !(log-math 10 -5)              =>  NaN
;;     !(z->status (log-math 10 0))   =>  Low      <- INVENTED FINDING
;;     !(z->status (log-math 10 -5))  =>  (Error …) atom leaked into results
;;
;; -inf is genuinely ordered below -1, so z->status calls it `Low` — "low
;; inflammation" conjured from an impossible measurement. NaN makes every comparison
;; False, which z->status reads as `Normal`. Neither can be caught downstream, so
;; nothing downstream is asked to: a non-positive raw value on a Log10 marker yields
;; NOTHING here, before the logarithm is ever taken. A true 0.0 is not hypothetical —
;; NHANES lab files contain real zeros, and they arrive as zeros only because
;; nhanes_common.fix_ibm_zero repairs pandas' IBM-370 zero bug (contract D6).
;;
;; Three further branches yield nothing rather than a number:
;;   * no reference cell for this marker/band/sex (§4);
;;   * a degenerate SD (<= 0) — division would give ±inf or NaN;
;;   * an unrecognized scale symbol — a record from a future emitter that this file
;;     does not understand must not be silently standardized as Identity. That IS
;;     the D17 bug (a log-scaled reference applied to a raw value), so the unknown
;;     branch declines.
;;
;; Every intermediate is let*-forced: /, - and > do not evaluate their arguments, so
;; feeding (/ (- $raw $mean) $sd) an unreduced (log-math …) would not compute.
(= (standardize-z $space $marker $band $sex $raw)
   (let $rid (reference-cell $space $marker $band $sex)
      (let* (($scale (match $space (RefScale $rid $sc) $sc))
             ($mean  (match $space (RefMean  $rid $mu) $mu))
             ($sd    (match $space (RefSD    $rid $s)  $s)))
         (if (> $sd 0.0)
             (if (== $scale Identity)
                 (let* (($dev (- $raw $mean)))
                    (/ $dev $sd))
                 (if (== $scale Log10)
                     (if (> $raw 0.0)
                         (let* (($lg  (log-math 10 $raw))
                                ($dev (- $lg $mean)))
                            (/ $dev $sd))
                         (superpose ()))      ; non-positive on a log scale -> nothing
                     (superpose ())))         ; unknown scale -> nothing, never Identity
             (superpose ()))))) ; degenerate SD -> nothing

;; ================================================================
;; 6. Smoke-test queries (shapes only — this file ships no numbers, so a live
;;    example needs reference atoms supplied by the caller or by
;;    build/nhanes_reference.metta; tests/ builds them from synthetic fixtures)
;; ================================================================
;; !(age-band 58)                                  ; => Age_50_59
;; !(age-band 70)                                  ; => Age_70p   (half-open [70, inf))
;;
;; With a synthetic Identity cell (RefMean 5.4) (RefSD 0.5) for HbA1c/Male/Age_50_59:
;; !(standardize-z &self HbA1c Age_50_59 Male 6.2) ; => 1.6
;;
;; With a synthetic Log10 cell whose mean/SD are on log10(mg/dL):
;; !(standardize-z &self CRP Age_50_59 Male 0.0)   ; => (empty)  — the D27 guard
;; !(standardize-z &self CRP Age_50_59 Male -1.0)  ; => (empty)  — the D27 guard
;;
;; No cell for this band:
;; !(standardize-z &self HbA1c Age_70p Male 6.2)   ; => (empty)  — never a default
