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
;; STATUS: design stub, verified against hyperon 0.2.10 (smoke tests at the
;; bottom; regression-guarded in tests/test_pln_deduction.py). Transforms are
;; deliberately coarse — do not treat the numbers as numerically validated; the
;; architecture is what is locked. The downstream deduction layer that consumes
;; these TVs lives in pln_deduction.metta.

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
;; 1. Field accessors — read a record's measurements out of a space
;; ================================================================
;; The space $s is passed in explicitly rather than hard-coding &self.
;; Under `import! &self`, a bare (match &self ...) inside this (imported)
;; module reads only THIS module's space, so cross-module reads of the
;; evidence records — which live in grim_age_lu2019_evidence — silently
;; return nothing. Callers pass the top-level &self, which holds every
;; imported atom. (Verified against hyperon 0.2.10; the deduction rule
;; will need the same treatment when it reads KB atoms.)
(= (ev-metric   $s $a) (match $s (EffectMetric        $a $m) $m))
(= (ev-value    $s $a) (match $s (EffectValue         $a $v) $v))
(= (ev-category $s $a) (match $s (HasEvidenceCategory $a $c) $c))

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
(= (calibrate-tv $s $a)
   (stv (effect->strength (ev-metric $s $a) (ev-value $s $a))
        (category->confidence (ev-category $s $a))))

;; ================================================================
;; 5. Storage convention is intentionally NOT committed here
;; ================================================================
;; `calibrate-tv` is a pure function of (space, record): call it wherever a TV
;; is needed, passing the space that holds the evidence atoms (normally &self).
;; Once the (not-yet-built) deduction rule fixes HOW it reads truth values, wrap
;; this to persist calibrated links using ONE convention (doc open-question #1):
;;
;;   (: (Predicts AgeAccelGrim AllCauseMortality)
;;      (calibrate-tv &self Lu2019_AgeAccelGrim_AllCauseMortality))    ; TV-as-type
;;
;;   (TruthValue (Predicts AgeAccelGrim AllCauseMortality)
;;      (calibrate-tv &self Lu2019_AgeAccelGrim_AllCauseMortality))    ; wrapper
;;
;; Picking one now would prejudge the deduction design, so the stub stops here.

;; ================================================================
;; 6. Smoke-test queries (verified, hyperon 0.2.10)
;; ================================================================
;; Load order: system_types logical_predicates epistemic_calibration
;;             grim_age_core grim_age_lu2019_evidence evidence_calibration
;; !(calibrate-tv &self Lu2019_AgeAccelGrim_AllCauseMortality)  ; HR 1.10 -> (stv 0.5 0.85)
;; !(calibrate-tv &self Lu2019_DNAmPAI1_CHD)                    ; HR 1.31 -> (stv 0.6 0.85)
