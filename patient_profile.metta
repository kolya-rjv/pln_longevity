;; patient_profile.metta
;; Patient grounding layer — raw biomarker profiles -> symbolic findings.   [v1]
;;
;; The inference demos so far are POPULATION-level: `diagnose` and
;; `rank-interventions` take an explicit, hand-typed list of "observed" markers /
;; candidate interventions. Every demo doc names the same missing piece — a real
;; patient profile — as the step from *synthetic* to *personalized*:
;;   * docs/abductive_diagnosis.md §5.3: "observations are passed explicitly;
;;     wiring them to a real NHANES methylation + blood profile ... turns the demo
;;     from synthetic to personalized."
;;   * docs/intervention_ranking.md §5.2: the ranking "is currently population-
;;     level. Demo 2 in full multiplies in patient-specific factors
;;     (f(patient factors)) ... needs the still-absent NHANES / patient-profile
;;     layer."
;;
;; This layer is that piece. It:
;;   1. represents a patient as a set of RAW, standardized biomarker measurements
;;      (immutable, source-faithful — the calibration layer's honesty contract);
;;   2. GROUNDS those numbers into the symbolic findings inference consumes, via a
;;      single documented threshold rule (the patient-layer analogue of
;;      calibrate-tv: the one place raw numbers become Elevated/Normal/Low);
;;   3. wires a patient's elevated findings into `diagnose` (Demo 1) and adds the
;;      patient-relevance factor `rank-interventions` needs for Demo 2.
;;
;; Load AFTER the whole inference stack (it consumes infer / diagnose / rank-score
;; / scored / sort-scored):
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis
;;   patient_profile                     (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_patient_grounding.py. Uses only core MeTTa builtins (match,
;; superpose, collapse, let/let*, car-atom/cdr-atom, +/*/>/<). As in every layer
;; the space $space is threaded explicitly (never hard-coded &self) because a
;; (match &self ...) inside an imported module reads only that module's own space
;; — the cross-module gotcha docs/calibration_layer.md §10.1 documents.

;; ================================================================
;; 0. Types
;; ================================================================
(: PatientProfile Type)

;; Qualitative biomarker status — the symbolic finding inference reasons over.
(: BiomarkerStatus Type)
(: Elevated BiomarkerStatus)
(: Normal   BiomarkerStatus)
(: Low      BiomarkerStatus)

(: SmokingStatus Type)
(: NeverSmoker    SmokingStatus)
(: FormerSmoker   SmokingStatus)
(: CurrentSmoker  SmokingStatus)

;; Standardized measurement: z = number of SDs above/below the age/sex-adjusted
;; population mean for this marker. This is exactly what "age acceleration" is for
;; an epigenetic clock, and what a standardized lab value is for a blood marker —
;; one convention that makes clocks and blood panels commensurable, and the value
;; NHANES / a blood panel actually yields. Positive z = above the norm.
(: MeasuredZ (-> PatientProfile Biomarker Number Atom))

;; Demographics — kept raw. Not yet used in scoring; they are the hooks the
;; sex/age-specific modifiers (pipeline.md App. A.7) will read in the next
;; increment. Present here so a profile is complete, not to be reasoned over yet.
(: PatientAge     (-> PatientProfile Number Atom))
(: PatientSex     (-> PatientProfile Sex Atom))
(: PatientSmoking (-> PatientProfile SmokingStatus Atom))

;; Minimal typing for a marker node introduced here (the first-generation clock's
;; age acceleration). AgeAccelGrim already lives in grim_age_core; the Horvath
;; accel is its first-gen counterpart, added so the discordance is representable.
(Inheritance HorvathAgeAccel EpigeneticAgeAcceleration)

;; ================================================================
;; 1. Example patient — the case study 58yo male (docs/casestudy.md)
;; ================================================================
;; Discordant profile: a NORMAL first-generation clock (Horvath) alongside an
;; ELEVATED mortality-predictive GrimAge and its elevated components + CRP. This
;; is the presentation docs/casestudy.md analyses and the one the abductive demo
;; was hand-fed; here it is grounded in measured values instead.
(InstanceOf Patient001 PatientProfile)
(PatientAge     Patient001 58)
(PatientSex     Patient001 Male)
(PatientSmoking Patient001 NeverSmoker)

(MeasuredZ Patient001 HorvathAgeAccel -0.2)   ; first-gen clock ~ normal
(MeasuredZ Patient001 AgeAccelGrim     1.6)   ; GrimAge elevated (the discordance)
(MeasuredZ Patient001 DNAmPAI1         1.8)   ; elevated GrimAge component
(MeasuredZ Patient001 DNAmGDF15        1.5)   ; elevated GrimAge component
(MeasuredZ Patient001 CRP              1.7)   ; elevated inflammation marker
(MeasuredZ Patient001 DNAmLeptin      -1.5)   ; low — exercises the Low branch

;; ================================================================
;; 1b. Second example patient — metabolic-dominant (contrast to Patient001)
;; ================================================================
;; Same surface discordance (elevated GrimAge) but a DIFFERENT driver: elevated
;; fasting glucose + HbA1c with NORMAL senescence markers. This contrast is what
;; makes personalization REORDER rather than merely amplify — from the same
;; candidate pool against the same outcome, Patient002 ranks the AMPK activators
;; (metformin, berberine) first where Patient001 ranks the senolytics first (see
;; the metabolic axis in mechanistic_bridges.metta and docs/patient_grounding.md
;; §6). GrimAge is elevated for BOTH, but for different upstream reasons — the
;; personalized diagnosis and ranking separate them; an LLM cannot.
(InstanceOf Patient002 PatientProfile)
(PatientAge     Patient002 61)
(PatientSex     Patient002 Male)
(PatientSmoking Patient002 FormerSmoker)

(MeasuredZ Patient002 HorvathAgeAccel  0.3)   ; first-gen clock ~ normal
(MeasuredZ Patient002 AgeAccelGrim     1.3)   ; GrimAge elevated (metabolic aging)
(MeasuredZ Patient002 FastingGlucose   1.9)   ; elevated — the metabolic signal
(MeasuredZ Patient002 HbA1c            1.6)   ; elevated
(MeasuredZ Patient002 DNAmPAI1         0.2)   ; senescence markers NORMAL
(MeasuredZ Patient002 CRP              0.4)
(MeasuredZ Patient002 DNAmGDF15        0.1)

;; ================================================================
;; 2. Grounding: standardized value -> qualitative status
;; ================================================================
;; The single tunable threshold (in SDs) at which a marker is called Elevated /
;; Low. This is the patient-layer analogue of calibrate-tv: the ONE documented
;; place raw numbers become the symbolic findings inference consumes. Tuning the
;; personalization loop means adjusting this one function (calibration_layer.md
;; §8 discipline).
(= (elevated-z-threshold) 1.0)

;; z -> Elevated / Normal / Low. Every grounded intermediate is let*-forced: the
;; comparisons > / < do NOT evaluate their arguments, so the threshold (and its
;; negation) must be reduced to concrete numbers first (the same discipline the
;; deduction / ranking / abduction layers document).
(= (z->status $z)
   (let* (($t  (elevated-z-threshold))
          ($nt (* -1.0 $t)))
      (if (> $z $t) Elevated
          (if (< $z $nt) Low Normal))))

;; Status of ONE marker for a patient. Space threaded explicitly. Yields nothing
;; if the marker was not measured for this patient (no invented Normal).
(= (patient-status $space $patient $marker)
   (let $z (match $space (MeasuredZ $patient $marker $zv) $zv)
        (z->status $z)))

;; ================================================================
;; 3. Observation set: the markers a patient presents ELEVATED
;; ================================================================
;; A non-deterministic stream of the patient's elevated marker NAMES. Each
;; MeasuredZ atom for the patient is destructured; only Elevated ones pass, the
;; rest drop via (superpose ()). This is the set the abductive-diagnosis demo
;; (pln_abductive_diagnosis.metta) consumes.
(= (elevated-marker $space $patient)
   (let ($m $z) (match $space (MeasuredZ $patient $mm $zz) ($mm $zz))
      (let $t (elevated-z-threshold)
         (if (> $z $t) $m (superpose ())))))

;; Collapsed into one tuple, e.g. (DNAmPAI1 DNAmGDF15 CRP AgeAccelGrim). Tuple
;; order is collapse-order (not significant — diagnose ranks by coverage/mass).
(= (patient-observations $space $patient)
   (collapse (elevated-marker $space $patient)))

;; ================================================================
;; 4. Patient-grounded Demo 1 driver (abductive diagnosis)
;; ================================================================
;; Personalized abductive diagnosis: pull the patient's elevated findings
;; automatically, then rank the upstream causes that explain them. Closes
;; docs/abductive_diagnosis.md open-question #3 — the demo goes from a hand-typed
;; marker list to a real (grounded) patient profile. A finding no hypothesis can
;; explain (e.g. AgeAccelGrim, which no bridge reaches) is simply carried in the
;; observation set and left unexplained — never invented into a hypothesis.
;;
;; (diagnose-patient <space> <patient> <candidate-hypotheses>)
;;   -> ((Hypothesis <best> (stv s c) <cov> <mass> (SupportedBy ...)) ...)
(= (diagnose-patient $space $patient $hyps)
   (diagnose $space $hyps (patient-observations $space $patient)))

;; ================================================================
;; 5. Patient factors for Demo 2 (intervention ranking)
;; ================================================================
;; docs/intervention_ranking.md §5.2 wants the population ranking multiplied by
;; f(patient factors). v1 uses the most defensible such factor: an intervention is
;; MORE relevant to a patient the more it pushes DOWN the markers that patient
;; actually presents elevated. That reuses the existing engine wholesale — an
;; intervention "helps" marker M iff forward deduction intv -> M is protective
;; (Neg), scored by the same rank-score the population ranking uses.

;; Protective effect (>= 0) of an intervention on ONE marker. rank-score maps a
;; Neg (protective) chain to +(s*c) and a Pos (harmful) one to -(s*c); we clamp to
;; 0 so relevance measures HELP, not harm, and an unconnected pair contributes
;; nothing (infer yields no result -> this branch is empty -> drops out of the sum).
(= (clamp0 $x) (if (> $x 0.0) $x 0.0))
(= (relevance-on $space $intv $marker)
   (let $r (infer $space $intv $marker)
        (clamp0 (rank-score $r))))

;; Sum a collapsed tuple of numbers (let*-forced; + does not evaluate its args).
(= (sum-tuple $xs)
   (if (== $xs ()) 0.0
       (let* (($h (car-atom $xs)) ($r (sum-tuple (cdr-atom $xs))))
          (+ $h $r))))

;; patient-relevance: total protective effect an intervention has across the
;; markers THIS patient presents elevated — the personalization signal.
;; $obs is let-forced to the CONCRETE tuple before superpose fans it: superpose
;; does not reduce its argument, so superposing the bare (patient-observations …)
;; call would branch over the unevaluated expression, not the markers (the same
;; grounded-op discipline the ranking/abduction layers document).
(= (patient-relevance $space $patient $intv)
   (let $obs (patient-observations $space $patient)
      (sum-tuple
         (collapse
            (relevance-on $space $intv (superpose $obs))))))

;; personalized-score: population protective effect on the outcome PLUS the
;; patient-relevance bonus. v1 combines additively (transparent and monotone: it
;; can only promote an intervention that also helps this patient's presentation,
;; never demote a population-protective one below a harmful one). The pipeline's
;; multiplicative f(patient factors) is the documented refinement
;; (docs/patient_grounding.md open-Q#2).
(= (personalized-score $space $patient $intv $outcome $r)
   (+ (rank-score $r) (patient-relevance $space $patient $intv)))

;; One personalized-scored candidate. $r binds EACH derivation of infer (one
;; scored tuple per derivation, like the population ranking); an unconnected
;; candidate yields no tuple and drops out. The full signed TV is kept alongside
;; the scalar so the ranked output stays auditable.
(= (score-candidate-patient $space $patient $intv $outcome)
   (let $r (infer $space $intv $outcome)
        (scored $intv (personalized-score $space $patient $intv $outcome $r) $r)))

;; (rank-interventions-for-patient <space> <patient> <candidate-tuple> <outcome>)
;;   -> ((scored <best> <score> (signed Neg (stv s c))) ...)  most-relevant first.
;; Reuses sort-scored / score-of from pln_intervention_ranking.metta; only the
;; score is patient-aware. Unconnected candidates are omitted (no false placement),
;; exactly as in the population ranking.
(= (rank-interventions-for-patient $space $patient $cands $outcome)
   (let $pool (collapse (score-candidate-patient $space $patient (superpose $cands) $outcome))
        (sort-scored $pool)))

;; ================================================================
;; 6. Smoke-test queries (intended values; live floats carry trailing noise —
;;    tests/test_patient_grounding.py compares with tolerance)
;; ================================================================
;; !(patient-status &self Patient001 DNAmPAI1)        ; => Elevated
;; !(patient-status &self Patient001 HorvathAgeAccel) ; => Normal
;; !(patient-status &self Patient001 DNAmLeptin)      ; => Low
;;
;; Grounded Demo 1 — same hallmark ranking as the hand-fed demo, now from the
;; patient's measured values (AgeAccelGrim is elevated too but no bridge explains
;; it, so it is carried un-credited):
;; !(diagnose-patient &self Patient001
;;     (CellularSenescence MitochondrialDysfunction ChronicInflammation))
;;  => CellularSenescence (cov 3) > ChronicInflammation (cov 2) > MitochondrialDysfunction (cov 1)
;;
;; Grounded Demo 2 — the senolytics reduce the markers this patient has elevated,
;; so patient-relevance is > 0 and they keep a protective ranking:
;; !(rank-interventions-for-patient &self Patient001
;;     (DasatinibPlusQuercetin Fisetin Spermidine Elamipretide) CoronaryHeartDisease)
;;  => DasatinibPlusQuercetin > Fisetin > Spermidine  (Elamipretide omitted: no chain)
