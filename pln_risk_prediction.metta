;; pln_risk_prediction.metta
;; Cardiovascular risk prediction with FULL uncertainty  (pipeline.md Demo 5). [v1]
;;
;; Demos 1-4 gave us: a signed, uncertainty-quantified transmission to CHD
;; (`infer`), a patient grounded in standardized measurements (`MeasuredZ`), a
;; GrimAge decomposition (`decompose-grimage`), and a counterfactual do-operator
;; (`counterfactual-patient`). Demo 5 is the capstone that turns all of that into
;; the ONE number a clinician asks for — a personalized ABSOLUTE risk — and does
;; it with the four things an LLM cannot (pipeline.md §4.5):
;;
;;   (1) a point estimate  P(CHD in 10y)  for a specific patient;
;;   (2) a CONFIDENCE INTERVAL propagated from the source-evidence confidence
;;       (the interval WIDENS as confidence falls — uncertainty is first-class);
;;   (3) a DECOMPOSITION of the excess risk into attributable contributing factors
;;       that SUM to the excess (residual explicit, never forced to zero);
;;   (4) PROJECTED risk under intervention scenarios.
;;
;; THE MODEL (v1 — coarse and honest; a multiplicative proportional-hazards step):
;;
;;     risk = baseline(outcome,age,sex) × HR_clock ^ (z_clock × sd->years)
;;
;;   * BOTH FACTORS ARE OUTCOME-SPECIFIC, and the OUTCOME IS PART OF THE BASELINE KEY.
;;     The optional nhanes_baseline.metta layer lets a survey-weighted, mortality-
;;     linked NHANES baseline serve the outcomes NHANES can identify (all-cause
;;     mortality, heart-disease MORTALITY); incident CHD keeps the curated prior of
;;     §0(a), because NHANES has no incident-CHD ascertainment to calibrate against.
;;     Threading ONE $outcome through the baseline lookup and the hazard lookup is
;;     what makes a cross-outcome product — a fatal-heart-disease baseline × the
;;     incident-CHD HR — unreachable rather than merely discouraged (§1, §2).
;;     With no generated baseline loaded this file behaves exactly as it did before.
;;
;;   * The predictor is the COMPOSITE clock AgeAccelGrim -> CHD (Lu et al. 2019,
;;     HR 1.07 per year of acceleration — the record already in
;;     grim_age_lu2019_evidence.metta, already read by `link-effect`/`infer`). We
;;     read the HR straight off that evidence record; NO new Effect/empirical atom
;;     is introduced, so the causal graph `infer` traverses is untouched and every
;;     pre-Demo-5 test stays green.
;;   * Using the composite clock (rather than multiplying the individual component
;;     hazard ratios) is the honest choice: DNAmPAI1 ⊂ GrimAge, so stacking
;;     GrimAge×PAI1×… would DOUBLE-COUNT. An LLM will happily multiply every hazard
;;     ratio it can name; PLN declines. The component-level story is recovered, once,
;;     as the DECOMPOSITION below (each clock component's SHARE of the one clock risk).
;;   * The CI is `risk × (1 ± (1−c)k)`: coarser evidence (lower c) -> wider interval.
;;   * The decomposition REUSES decompose-grimage (Demo 4): the clock's elevation is
;;     split into its measured components, and the excess risk is allocated by each
;;     component's elevation share; the unattributed remainder is reported as an
;;     explicit residual.
;;   * The intervention projection REUSES counterfactual-patient (Demo 4): a lever's
;;     expected GrimAge reduction lowers z_clock, and the risk is recomputed — so the
;;     Demo-4 ordering (clear-senescence B > reduce-inflammation A > metabolic C ≈ 0)
;;     reappears here in ABSOLUTE-RISK terms, and a lever with no path to the clock
;;     honestly projects ~0 risk change (never a fabricated number).
;;
;; Load AFTER the whole inference + patient + counterfactual stack (it consumes
;; decompose-grimage / counterfactual-patient / grimage-weight / patient-elevated-z
;; / cf-tv-c / calibrate-tv / MeasuredZ):
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile pln_counterfactual
;;   pln_risk_prediction                 (this file)
;;
;; OPTIONAL, loaded AFTER this file when a data-backed baseline is wanted:
;;   nhanes_baseline                     (Schema B/C types + the outcome-keyed lookup)
;;   build/nhanes_baseline.metta         (the generated numeric records)
;; Both absent is the default, and the default path through this file is unchanged:
;; §1's `patient-baseline` tests for a BaselineRiskRecord atom before it ever names a
;; rule from that layer, so the missing symbol is never evaluated.
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_risk_prediction.py. Uses only core MeTTa builtins plus the grounded
;; math ops (+,-,*,/,>,<,==,car-atom/cdr-atom, pow-math). As in every layer the
;; space $space is threaded explicitly (never a bare &self — a (match &self ...)
;; inside an imported module reads only that module's own space, the cross-module
;; gotcha docs/calibration_layer.md §10.1 documents), and every arithmetic / list
;; intermediate is let*-forced (grounded ops do NOT evaluate their arguments — the
;; discipline pln_intervention_ranking.metta §3 documents; pow-math in particular
;; must be fed a concrete exponent). v1 locks the ARCHITECTURE (the hazard model +
;; CI + decomposition + projection); the constants (§0 knobs) are documented tunables.

;; ================================================================
;; 0. KNOBS  (curated priors — the one place hand-set constants live)
;; ================================================================
;; Every number below is a documented, tunable prior, quarantined here and clearly
;; labelled (the mechanistic_bridges.metta discipline). v1 fixes the architecture;
;; a later increment calibrates these against real cohort data (see doc open-Qs).

;; (a) Baseline 10-year absolute CHD risk by age band × sex. Curated from
;;     pipeline.md App. A.8 (Male 50-59 -> 0.08, Male 60-69 -> 0.15); the <50, 70+
;;     and female bands are coarse expert priors in the same spirit. A dedicated
;;     calibrated baseline table (Framingham / PCE) is the honest upgrade.
;;     Single-rule + nested if -> deterministic ("identical inputs -> identical
;;     output", pipeline.md Table 2).
;;
;;     STILL CURATED, DELIBERATELY — and it is not the table NHANES could replace.
;;     nhanes_baseline.metta adds a survey-weighted, mortality-linked baseline for the
;;     outcomes NHANES can actually identify, and incident CHD is not one of them: the
;;     Linked Mortality File ascertains DEATH, and NHANES's own CHD items are lifetime
;;     "ever told" PREVALENCE at baseline, so there is no incident-CHD ascertainment
;;     to calibrate against. This therefore remains a curated prior, labelled as one,
;;     and it is the fallback for CoronaryHeartDisease ONLY (see §1's resolution
;;     order, and nhanes_baseline.metta §1 for why substituting a heart-disease-
;;     MORTALITY number here would be an estimate of nothing).
(= (baseline-risk-chd $age $sex)
   (if (< $age 50) (if (== $sex Male) 0.04 0.02)
       (if (< $age 60) (if (== $sex Male) 0.08 0.05)
           (if (< $age 70) (if (== $sex Male) 0.15 0.10)
               (if (== $sex Male) 0.25 0.18)))))

;; (b) SD -> years conversion for GrimAge acceleration. Patient z is stored in SDs
;;     (patient_profile.metta) but the Lu 2019 HR is "per one-year increase", so the
;;     exponent must be in years. ~4.2 y/SD reproduces the case study's "+6.7 years"
;;     GrimAge for Patient001's 1.6 SD (docs/casestudy.md §4.4 / pipeline.md §4.4).
;;     This is exactly the EffectScale normalization evidence_calibration.metta §2
;;     flags as the v1 todo, done here as one documented knob.
(= (grimaccel-sd-to-years) 4.2)

;;     A MEASURED spread, placed NEXT TO the knob rather than over it. A
;;     ClockAccelSpread record (Schema C, declared in nhanes_baseline.metta, emitted
;;     by nhanes_dnam_etl.py from the NHANES DNMEPI subsample) carries the weighted SD
;;     of GrimAge acceleration IN YEARS — exactly the quantity this knob guesses at.
;;     `grimaccel-sd-to-years-measured` returns it, and NOTHING when no such record is
;;     loaded; `grimaccel-sd-to-years-check` returns the two side by side so the
;;     comparison is one query.
;;
;;     WHY THIS DOES NOT OVERRIDE THE KNOB. Swapping a curated constant for a sourced
;;     one the moment a record appears would move every number this file produces —
;;     the point estimate, the CI, the decomposition shares, the projections, and the
;;     §7 values the regression tests pin — with nothing in the output saying that it
;;     happened or which value was used. A calibration that changes silently is not a
;;     calibration, it is a drift. So the sourced number is made VISIBLE and the swap
;;     stays a deliberate, reviewable edit to the knob above (whose effect on the §7
;;     values is then a diff, not a surprise).
;;
;;     The lookup requires (SpreadDefinition … Residual). The SD of the age-adjusted
;;     RESIDUAL and the SD of the raw DIFFERENCE (clock age − age) are different
;;     numbers, and only the residual is the scale AgeAccelGrim is defined on
;;     ((ResidualOf AgeAccelGrim GrimAge), grim_age_core.metta) and the scale Lu 2019
;;     fit the hazard ratio on. A Difference record therefore answers NOTHING here
;;     rather than answering approximately.
;;
;;     These are `match`es over record atoms, not calls into nhanes_baseline.metta, so
;;     with that layer absent they yield nothing instead of an unreduced expression.
(= (grimaccel-sd-to-years-measured $space)
   (let $sid (match $space
                (, (SpreadMarker     $s AgeAccelGrim)
                   (SpreadDefinition $s Residual))
                $s)
      (match $space (SpreadSDYears $sid $v) $v)))

;;   (SDToYears (curated 4.2) (measured <v>))  |  (SDToYears (curated 4.2) (measured none))
(= (grimaccel-sd-to-years-check $space)
   (let $m (collapse (grimaccel-sd-to-years-measured $space))
      (if (== $m ())
          (SDToYears (curated (grimaccel-sd-to-years)) (measured none))
          (let $v (car-atom $m)
             (SDToYears (curated (grimaccel-sd-to-years)) (measured $v))))))

;; (c) One modeling-step confidence discount below the source evidence confidence:
;;     the estimate inherits the clock->CHD evidence confidence, then pays a single
;;     discount for the SD->years + baseline modeling layered on top. Reuses the
;;     deduction layer's per-hop chain-discount value so the two stay in lockstep.
(= (risk-conf-discount) 0.9)

;; (d) CI width scaling: the interval's relative half-width is (1 − c) × k, so a
;;     lower-confidence estimate carries a proportionally WIDER interval. k = 1.0 in
;;     v1 (half-width = the confidence deficit); tune against interval calibration.
(= (risk-ci-k) 1.0)

;; ================================================================
;; 1. Patient + predictor accessors (space threaded explicitly)
;; ================================================================
(= (patient-age $space $patient) (match $space (PatientAge $patient $a) $a))
(= (patient-sex $space $patient) (match $space (PatientSex $patient $s) $s))

;; The patient's measured GrimAge acceleration (in SDs). Continuous — risk rises
;; smoothly with the clock, so we read the z directly (a z of 0 -> multiplier 1
;; -> risk == baseline). Nothing if the clock was not measured (no invented value).
;;
;; Reads through `patient-z` (patient_profile.metta) rather than matching `MeasuredZ`
;; directly, so a patient whose clock arrives as a RAW value reaches this layer too.
;; Before this, `clock-z` was the last read site still bypassing the derived path, and
;; the inconsistency showed: a patient carrying only `MeasuredRaw` atoms got a full
;; `decompose-grimage` and a working `counterfactual-patient`, but `predict-risk`
;; silently returned nothing — because those layers had been converted and this one
;; had not.
;;
;; `patient-z` preserves both guarantees this site depends on. It is single-valued, with
;; an explicitly stored `MeasuredZ` winning over a derived one, so the hand-typed example
;; patients are unaffected; and it yields NOTHING when the clock was neither measured nor
;; derivable, so an ungrounded clock still produces no risk estimate rather than a default.
(= (clock-z $space $patient)
   (patient-z $space $patient AgeAccelGrim))

;; ---------------------------------------------------------------------------
;; The patient's baseline absolute risk FOR THE OUTCOME BEING PREDICTED.
;; ---------------------------------------------------------------------------
;; The baseline now carries the outcome, because a baseline without one is the D21
;; trap: the model's product is baseline × HR, both factors are outcome-specific, and
;; pairing them across outcomes yields an estimate of nothing (a fatal-heart-disease
;; baseline is far too small to be an incidence risk, and the HR beside it was fit to
;; a different event process). One $outcome enters here and the SAME one enters
;; `risk-multiplier-at` from `absolute-risk-at` (§2), so the pairing is fixed by the
;; call rather than by a convention a caller could forget. There is deliberately no
;; outcome-free way to obtain a baseline.
;;
;; RESOLUTION ORDER — explicit, and SINGLE-VALUED:
;;   1. a data-backed NHANES BaselineRiskRecord for the REQUESTED OUTCOME, when the
;;      nhanes_baseline.metta layer and a generated record are loaded;
;;   2. otherwise the curated prior of §0(a), WHICH EXISTS FOR CoronaryHeartDisease
;;      ONLY — so an outcome with neither a record nor a curated table yields NOTHING,
;;      never a borrowed number.
;;
;; THE ONE DELIBERATE BEHAVIOR CHANGE, stated plainly. Before the outcome was part of
;; the key, `patient-baseline` returned the CHD-incidence table for WHATEVER outcome
;; was being predicted, so `(predict-risk … CongestiveHeartFailure)` priced CHF off a
;; ten-year CHD-INCIDENCE baseline of 0.08 and multiplied it by the CHF hazard ratio.
;; That product was not a CHF risk; it was the D3 conflation with a different label on
;; it. It now yields NOTHING until a CHF baseline exists. CoronaryHeartDisease — the
;; outcome every driver in §6 defaults to and every §7 value is quoted for — is
;; bit-for-bit unchanged.
;;
;; MeTTa has no cut, so the two candidates are not left to both survive: the
;; data-backed candidates are COLLAPSED and the tuple is tested, exactly the
;; collapse-then-test idiom patient_profile.metta's `patient-z` uses to let an explicit
;; MeasuredZ beat a derived one. Downstream consumers here count on one value — a
;; second baseline would double every risk report, CI and decomposition it appears in.
;;
;; THE PRESENCE TEST IS NOT REDUNDANT. `baseline-risk-data` is defined in the OPTIONAL
;; nhanes_baseline.metta layer, and in MeTTa a call with no matching rule head does not
;; yield nothing — it yields the unreduced expression, which then propagates into the
;; arithmetic (tests/test_nhanes_common.py::
;; test_a_mismatched_rule_head_returns_the_unreduced_expression_not_empty). Testing
;; for a BaselineRiskRecord atom FIRST means that with no generated baseline loaded the
;; symbol is never evaluated, and this file behaves exactly as it did before: () from
;; the match, straight to the curated prior. (The match cannot be satisfied by
;; nhanes_baseline.metta's own (: BaselineRiskRecord Type) declaration — that atom's
;; type is Type, not BaselineRiskRecord.)
(= (curated-baseline $space $patient $outcome)
   (if (== $outcome CoronaryHeartDisease)
       (let* (($a (patient-age $space $patient)) ($s (patient-sex $space $patient)))
          (baseline-risk-chd $a $s))
       (superpose ())))                        ; no curated table for this outcome

(= (data-baseline-candidates $space $patient $outcome)
   (let $recs (collapse (match $space (: $bid BaselineRiskRecord) $bid))
      (if (== $recs ())
          ()                                   ; no data layer -> never names its rules
          (let* (($a (patient-age $space $patient))
                 ($s (patient-sex $space $patient)))
             (collapse (baseline-risk-data $space $outcome $a $s))))))

;; Age and sex are let*-forced above before either table sees them: `<` and `==` do
;; not evaluate their arguments, so an unforced (patient-age …) would be compared as
;; an expression.
(= (patient-baseline $space $patient $outcome)
   (let $data (data-baseline-candidates $space $patient $outcome)
      (if (== $data ())
          (curated-baseline $space $patient $outcome)
          (car-atom $data))))

;; The hazard ratio for the composite clock -> outcome, read straight off the
;; empirical evidence record (never hand-typed): HR 1.07 for AgeAccelGrim -> CHD.
(= (clock-hr $space $outcome)
   (match $space
      (, (HasPredictor $r AgeAccelGrim) (HasOutcome $r $outcome) (EffectValue $r $hr))
      $hr))

;; The calibrated confidence of that clock -> outcome evidence link (reuses the
;; calibration layer's calibrate-tv + the cf-tv-c stv accessor from Demo 4).
(= (clock-conf $space $outcome)
   (match $space (, (HasPredictor $r AgeAccelGrim) (HasOutcome $r $outcome))
      (cf-tv-c (calibrate-tv $space $r))))

;; ================================================================
;; 2. The absolute-risk model:  baseline × HR ^ (z × sd->years)
;; ================================================================
;; The proportional-hazards multiplier for a GIVEN clock z (so §4 can recompute it
;; at a post-intervention z). pow-math is fed a let*-forced concrete exponent.
(= (risk-multiplier-at $space $outcome $z)
   (let* (($hr  (clock-hr $space $outcome))
          ($yrs (* $z (grimaccel-sd-to-years))))
      (pow-math $hr $yrs)))

;; Absolute 10-year risk at a given clock z, and at the patient's measured z.
(= (absolute-risk-at $space $patient $outcome $z)
   (let* (($p0 (patient-baseline $space $patient $outcome))
          ($m  (risk-multiplier-at $space $outcome $z)))
      (* $p0 $m)))
(= (absolute-risk $space $patient $outcome)
   (absolute-risk-at $space $patient $outcome (clock-z $space $patient)))

;; ================================================================
;; 3. Uncertainty: propagated confidence + confidence interval
;; ================================================================
;; Confidence = source evidence confidence × one modeling-step discount.
(= (risk-confidence $space $patient $outcome)
   (* (clock-conf $space $outcome) (risk-conf-discount)))

;; A multiplicative CI around the point estimate whose relative half-width grows as
;; confidence falls: [ p·(1−(1−c)k) , min(1, p·(1+(1−c)k)) ]. Multiplicative keeps
;; the lower bound ≥ 0; the upper bound is clamped to 1.
(= (risk-ci $space $patient $outcome)
   (let* (($p   (absolute-risk $space $patient $outcome))
          ($c   (risk-confidence $space $patient $outcome))
          ($rel (* (- 1.0 $c) (risk-ci-k)))
          ($lo  (* $p (- 1.0 $rel)))
          ($hir (* $p (+ 1.0 $rel)))
          ($hi  (if (> $hir 1.0) 1.0 $hir)))
      (ci $lo $hi)))

;; predict-risk: the full uncertainty-quantified report.
;;   (RiskPrediction <patient> <outcome> (point p) (ci lo hi)
;;       (confidence c) (baseline p0) (multiplier m))
(= (predict-risk $space $patient $outcome)
   (let* (($p0 (patient-baseline $space $patient $outcome))
          ($m  (risk-multiplier-at $space $outcome (clock-z $space $patient)))
          ($p  (absolute-risk $space $patient $outcome))
          ($c  (risk-confidence $space $patient $outcome))
          ($ci (risk-ci $space $patient $outcome)))
      (RiskPrediction $patient $outcome
         (point $p) $ci (confidence $c) (baseline $p0) (multiplier $m))))

;; ================================================================
;; 4. Decomposition of the excess risk into contributing factors
;; ================================================================
;; The excess over baseline is the clock-attributable risk. We split it across the
;; clock's MEASURED-ELEVATED components (reusing patient-elevated-z / grimage-weight
;; from Demo 4) by each component's elevation share of the total clock z, and report
;; the unattributed remainder as an explicit residual (the coarse v1 weights +
;; unmeasured surrogates leave a positive residual — never forced to zero).
;;   (RiskFactor <M> (contribution w·z) (share s) (attributable-risk r))
(= (risk-share-one $space $patient $m $excess $grimz)
   (let* (($z       (patient-elevated-z $space $patient $m))   ; nothing if not elevated
          ($w       (grimage-weight $m))
          ($contrib (* $w $z))
          ($share   (/ $contrib $grimz))
          ($rrisk   (* $excess $share)))
      (RiskFactor $m (contribution $contrib) (share $share) (attributable-risk $rrisk))))

;; Sum the attributable-risk fields (let*-forced; + does not evaluate its args).
(= (rf-attr (RiskFactor $m $contrib $share (attributable-risk $r))) $r)
(= (sum-risk-attr $fs)
   (if (== $fs ()) 0.0
       (let* (($h (car-atom $fs)) ($r (sum-risk-attr (cdr-atom $fs))))
          (+ (rf-attr $h) $r))))

;; risk-decomposition: baseline + attributed component factors + explicit residual,
;; all summing to the total risk.
;;   (RiskDecomposition <patient> <outcome> (baseline p0) (total p) (excess e)
;;       (Factors (...)) (attributed-excess a) (residual-excess r))
(= (risk-decomposition $space $patient $outcome)
   (let* (($p0     (patient-baseline $space $patient $outcome))
          ($p      (absolute-risk $space $patient $outcome))
          ($excess (- $p $p0))
          ($grimz  (clock-z $space $patient)))
      (let* (($factors (collapse
                          (let $m (match $space (PartOf $mm GrimAge) $mm)
                             (risk-share-one $space $patient $m $excess $grimz))))
             ($attr    (sum-risk-attr $factors))
             ($resid   (- $excess $attr)))
         (RiskDecomposition $patient $outcome
            (baseline $p0) (total $p) (excess $excess)
            (Factors $factors)
            (attributed-excess $attr) (residual-excess $resid)))))

;; ================================================================
;; 5. Projected risk under an intervention scenario  (reuses Demo 4)
;; ================================================================
;; A lever's expected GrimAge reduction (counterfactual-patient's signed
;; expected-delta, which is ≤ 0 / protective) lowers the clock z; we recompute the
;; absolute risk at the new z and report the reduction. counterfactual-patient
;; returns NOTHING for an unresolvable lever (e.g. Elamipretide), so project-risk
;; then yields nothing too — the lever is omitted, never given a fabricated number.
(= (cf-delta-of (Counterfactual $l $o (expected-delta $d) $signed $via)) $d)
(= (cf-conf-of  (Counterfactual $l $o $ed (signed $sg (stv $s $c)) $via)) $c)
(= (cf-via-of   (Counterfactual $l $o $ed $signed (Via $v))) $v)

;;   (ProjectedRisk <lever> <outcome> (point p') (reduction Δrisk)
;;       (delta-clock d) (confidence c) (Via (M …)))
(= (project-risk $space $patient $outcome $lever)
   (let $cf (counterfactual-patient $space $patient $lever)
      (let* (($d     (cf-delta-of $cf))
             ($cc    (cf-conf-of $cf))
             ($via   (cf-via-of $cf))
             ($z     (clock-z $space $patient))
             ($znew  (+ $z $d))                                  ; d ≤ 0 lowers the clock
             ($p     (absolute-risk-at $space $patient $outcome $z))
             ($pnew  (absolute-risk-at $space $patient $outcome $znew))
             ($red   (- $p $pnew)))
         (ProjectedRisk $lever $outcome
            (point $pnew) (reduction $red)
            (delta-clock $d) (confidence $cc) (Via $via)))))

;; ================================================================
;; 6. Patient drivers (default the outcome to CoronaryHeartDisease)
;; ================================================================
(= (predict-risk-patient $space $patient)
   (predict-risk $space $patient CoronaryHeartDisease))
(= (risk-decomposition-patient $space $patient)
   (risk-decomposition $space $patient CoronaryHeartDisease))
(= (project-risk-patient $space $patient $lever)
   (project-risk $space $patient CoronaryHeartDisease $lever))

;; The three casestudy §5.2 / §7.3 intervention scenarios as absolute-risk
;; projections, collapsed into one tuple. EXPECTED: Δrisk_B > Δrisk_A > Δrisk_C ≈ 0
;; — clearing senescence (B, upstream, reaches both clock surrogates) buys a bigger
;; absolute-risk reduction than reducing inflammation alone (A), and the metabolic
;; lever (C) reaches no clock surrogate so it moves CHD risk by ~0 (the honest
;; "this lever doesn't touch THIS pathway").
(= (risk-scenarios $space $patient)
   (collapse
      (project-risk-patient $space $patient
         (superpose (ChronicInflammation CellularSenescence InsulinResistance)))))

;; ================================================================
;; 7. Smoke-test queries (intended values; live floats carry trailing noise —
;;    tests/test_risk_prediction.py compares with tolerance)
;; ================================================================
;; Load order: system_types logical_predicates epistemic_calibration
;;             grim_age_core grim_age_lu2019_evidence evidence_calibration
;;             hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;             mechanistic_bridges pln_deduction pln_intervention_ranking
;;             pln_abductive_diagnosis patient_profile pln_counterfactual
;;             pln_risk_prediction
;;
;; Patient001 (58yo male; AgeAccelGrim z 1.6 -> ~6.72 years; baseline 0.08; HR 1.07):
;; !(predict-risk-patient &self Patient001)
;;   => (RiskPrediction Patient001 CoronaryHeartDisease (point 0.12605…)
;;         (ci 0.09644… 0.15566…) (confidence 0.765) (baseline 0.08) (multiplier 1.5756…))
;;
;; Decompose the excess (0.04605) into clock-component shares + residual:
;; !(risk-decomposition-patient &self Patient001)
;;   => (RiskDecomposition Patient001 CoronaryHeartDisease (baseline 0.08) (total 0.12605…)
;;         (excess 0.04605…)
;;         (Factors ( (RiskFactor DNAmGDF15 (contribution 0.1875) (share 0.1172…) (attributable-risk 0.0054…))
;;                    (RiskFactor DNAmPAI1  (contribution 0.225)  (share 0.1406…) (attributable-risk 0.0065…)) ))
;;         (attributed-excess 0.0118…) (residual-excess 0.0342…))
;;
;; Projected risk under the three scenarios (B > A > C ≈ 0):
;; !(risk-scenarios &self Patient001)
;;   => A ChronicInflammation  reduction ~0.00415
;;      B CellularSenescence    reduction ~0.00753   (bigger — reaches both surrogates)
;;      C InsulinResistance     reduction  0.0        (no path to the clock)
;;
;; An edge-less lever is omitted, not placed:
;; !(project-risk-patient &self Patient001 Elamipretide)     ; => (empty)

;; ----------------------------------------------------------------
;; 7b. The outcome-keyed baseline (§1), by query
;; ----------------------------------------------------------------
;; Every value above is UNCHANGED by the baseline work: with no BaselineRiskRecord in
;; the space, `patient-baseline` resolves to the §0(a) curated prior for
;; CoronaryHeartDisease, exactly as it did when it took no outcome.
;;
;; !(patient-baseline &self Patient001 CoronaryHeartDisease)   ; => 0.08  (curated)
;; !(patient-baseline &self Patient001 AllCauseMortality)      ; => (empty)
;;     — no curated all-cause table exists, and inventing one is the D3 conflation.
;;       Load nhanes_baseline.metta + a generated record and this returns the
;;       survey-weighted NHANES value for the patient's band and sex instead.
;;
;; The one changed behavior (see §1): a non-CHD outcome no longer borrows the CHD
;; table. This used to return 0.08 x 1.10^6.72 = 0.1518, a CHD-INCIDENCE baseline
;; priced with the heart-failure hazard ratio:
;; !(predict-risk &self Patient001 CongestiveHeartFailure)     ; => (empty)
;;
;; Exactly one baseline is ever returned, even with several records in the space:
;; !(collapse (patient-baseline &self Patient001 CoronaryHeartDisease))  ; => (0.08)
;;
;; The curated knob and a measured spread, side by side (§0(b)):
;; !(grimaccel-sd-to-years-measured &self)   ; => (empty) with no Schema C record
;; !(grimaccel-sd-to-years-check &self)      ; => (SDToYears (curated 4.2)
;;                                           ;       (measured none))
