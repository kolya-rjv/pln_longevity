;; evidence_calibration.metta
;; Evidence -> PLN truth-value calibration layer.            [STUB / v0]
;;
;; Converts raw, source-faithful evidence records (EmpiricalAssociation,
;; and later DrugAge Experiment rows) into PLN truth values (stv s c):
;;   strength   s <- effect size + scale + direction
;;   confidence c <- evidence weight (study/species, p-value, sample size)
;;
;; Design rationale and open questions: docs/calibration_layer.md
;;
;; Load AFTER:
;;   system_types.metta
;;   logical_predicates.metta
;;   epistemic_calibration.metta        (provides evidence-confidence)
;;   grim_age_core.metta                (typing for predictors/outcomes)
;;   grim_age_lu2019_evidence.metta     (the records calibrated below)
;;
;; STATUS: design stub. Transforms are deliberately coarse and have NOT been
;; run against a live Hyperon runtime (no `hyperon` in the env yet). Do not
;; treat the numbers as validated. Smoke-test queries are at the bottom.

;; ================================================================
;; 0. Input contract
;; ================================================================
;; Every calibratable record must declare an evidence category drawn from
;; epistemic_calibration's EvidenceCategory enumeration.
(: HasEvidenceCategory (-> EmpiricalAssociation EvidenceCategory Atom))

;; The Lu et al. (2019) GrimAge associations are meta-analytic human results.
;; Tagged additively here so the source evidence file stays untouched. A
;; dedicated MetaAnalysis_Human tier (~0.90) could be added to
;; epistemic_calibration later; for the stub we reuse MultipleHumanTrials (0.85).
(HasEvidenceCategory Lu2019_AgeAccelGrim_AllCauseMortality MultipleHumanTrials)
(HasEvidenceCategory Lu2019_AgeAccelGrim_CHD               MultipleHumanTrials)
(HasEvidenceCategory Lu2019_AgeAccelGrim_CHF               MultipleHumanTrials)
(HasEvidenceCategory Lu2019_DNAmPAI1_CHD                   MultipleHumanTrials)

;; ================================================================
;; 1. Field accessors — read a record's measurements out of the space
;; ================================================================
(= (ev-metric   $a) (match &self (EffectMetric        $a $m) $m))
(= (ev-value    $a) (match &self (EffectValue         $a $v) $v))
(= (ev-category $a) (match &self (HasEvidenceCategory $a $c) $c))

;; ================================================================
;; 2. Strength  <-  effect size + metric              [STUB: buckets]
;; ================================================================
;; A hazard ratio is in [0, inf), not a probability, so it must be squashed
;; into [0,1]. v0 buckets a per-unit HR into a coarse strength band. v1 (see
;; doc) should: use a smooth transform such as (- 1 (/ 1.0 $hr)) or a logistic
;; on log-HR; normalize by EffectScale (per-year vs per-SD); handle HR < 1
;; (protective) by direction.
(= (effect->strength "hazard_ratio" $hr)
   (if (> $hr 1.5) 0.70
       (if (> $hr 1.1) 0.60 0.50)))

;; ================================================================
;; 3. Confidence  <-  evidence weight     [reuses epistemic_calibration]
;; ================================================================
;; This is where "shown only in mice => weaker" lives: a non-human / lower-tier
;; category maps to a smaller evidence-confidence, so the link inherits lower c.
;; A p-value gate / sample-size term could be folded in later via (min ...).
(= (category->confidence $cat) (evidence-confidence $cat))

;; ================================================================
;; 4. The bridge: evidence record -> calibrated truth value
;; ================================================================
(= (calibrate-tv $a)
   (stv (effect->strength (ev-metric $a) (ev-value $a))
        (category->confidence (ev-category $a))))

;; ================================================================
;; 5. Storage convention is intentionally NOT committed here
;; ================================================================
;; `calibrate-tv` is a pure function: call it wherever a TV is needed. Once the
;; (not-yet-built) deduction rule fixes HOW it reads truth values, wrap this to
;; persist calibrated links using ONE convention (doc open-question #1), e.g.:
;;
;;   (: (Predicts AgeAccelGrim AllCauseMortality)
;;      (calibrate-tv Lu2019_AgeAccelGrim_AllCauseMortality))          ; TV-as-type
;;
;;   (TruthValue (Predicts AgeAccelGrim AllCauseMortality)
;;      (calibrate-tv Lu2019_AgeAccelGrim_AllCauseMortality))          ; wrapper
;;
;; Picking one now would prejudge the deduction design, so the stub stops here.

;; ================================================================
;; 6. Smoke-test queries (run once hyperon is available)
;; ================================================================
;; !(calibrate-tv Lu2019_AgeAccelGrim_AllCauseMortality)   ; HR 1.10 -> expect (stv 0.50 0.85)
;; !(calibrate-tv Lu2019_DNAmPAI1_CHD)                     ; HR 1.31 -> expect (stv 0.60 0.85)
