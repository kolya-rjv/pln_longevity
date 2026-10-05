;; pln_linage2.metta
;; LinAge2 in the inference stack: decomposition, hazard, risk, counterfactual. [v1]
;;
;; A LinAge2 service response gives a patient a biological-age delta in years and
;; ONE years-contribution per model input. That is already a decomposition — but
;; a decomposition into LABS, with no cause, no direction, no evidence weight and
;; no link to anything the patient can do. This layer is what the knowledge base
;; adds on top of it, and nothing an LLM reading the same JSON could reproduce:
;;
;;   (1) an HONEST decomposition: measured vs imputed contributions kept apart,
;;       an explicit age-term residual, and — for the four inputs that read out a
;;       KB biomarker — the upstream hallmark causes, credited ONLY when the
;;       patient's own standardized value witnesses the direction (§3);
;;   (2) the clock's relative HAZARD, read off the Fong 2025 record and
;;       exponentiated by the delta in its native unit (§4);
;;   (3) an ABSOLUTE all-cause-mortality risk with a propagated interval, when
;;       and only when an outcome-matched data-backed baseline is loaded — the
;;       nhanes_baseline.metta path, never a curated all-cause table (§4);
;;   (4) a COUNTERFACTUAL in years: "if <lever> normalized the driver it stands
;;       for, how many of my LinAge2 years would go" — routed through the causal
;;       graph and the lever's own evidence edge, with the smoking lever gated
;;       IN THE ENGINE on the patient being a current smoker (§5).
;;
;; THE PATIENT ATOMS this layer reads (rendered per request by
;; pln_chat/core/linage2_builder.py from the service response; never on disk):
;;   (LinAgeDelta <P> <years>)                          ; BA - CA
;;   (MeasuredZ <P> LinAgeAccel <z>)                    ; years / linage-sd-to-years
;;   (LinAgeContribution <P> <Input> <years> Measured)  ; a lab the caller supplied
;;   (LinAgeContribution <P> <Input> <years> Imputed)   ; a lab the service filled in
;; plus the ordinary PatientAge / PatientSex / PatientSmoking / MeasuredZ atoms of
;; patient_profile.metta for whatever else the caller sent.
;;
;; WHY A CONTRIBUTION NEVER BECOMES A FINDING BY ITSELF (the one modelling rule
;; that matters here). contribution = z x w, and w is a sex-specific projection
;; whose sign is not the lab's clinical direction (linage2_core.metta header:
;; CRP is +4.4 months/SD in women and -0.2 in men). So (> years 0) is evidence
;; that the input AGES this patient under this model, not that the lab is high.
;; Crediting a cause additionally requires a WITNESS: the patient's own
;; (patient-z …) for the readout biomarker above `elevated-z-threshold`, or for
;; the cotinine input (PatientSmoking <P> CurrentSmoker). No witness — no cause,
;; and the years are reported as unexplained. Never invented.
;;
;; Load AFTER the whole NHANES patient stack + lifestyle_evidence + linage2_core +
;; linage2_fong2025_evidence, in the query-scoped LINAGE2_PATIENT_STACK
;; (core/pln_runner.py). It reuses, never re-derives: patient-z / unique-tuple /
;; elevated-z-threshold (patient_profile), infer (pln_deduction), pos-transmission /
;; cf-tv-s / cf-tv-c (pln_counterfactual), chain-discount (pln_deduction),
;; patient-baseline / risk-conf-discount / risk-ci-k (pln_risk_prediction),
;; calibrate-tv (evidence_calibration).
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_linage2.py. Same disciplines as every layer: $space threaded
;; explicitly, every arithmetic / comparison operand let*-forced, collapse-then-
;; test for single-valuedness, (superpose ()) to decline, no MeTTa sort (O(n^2)
;; over 59 inputs; the API sorts), and the grounded foldl-atom for every sum. Rule definitions cost nothing in the scoped
;; space; the four new head symbols do, and are counted in the test.

;; ================================================================
;; 0. KNOBS  (the two hand-set numbers, with their provenance)
;; ================================================================
;; (a) SD -> years for the LinAge2 delta, so (MeasuredZ <P> LinAgeAccel z) can be
;;     stated in the KB's convention and z->status can call the clock Elevated /
;;     Normal / Low. DERIVED, not guessed: the standard deviation of BA-CA across
;;     the model's own training cohort, computed from the training matrices
;;     shipped with the published artifacts (Rejuve/LinAge2-Python @ d0b9973,
;;     artifacts/coxCovsTrainM.csv + coxCovsTrainF.csv, NHANES 1999-2000):
;;         male   n = 1,045  sd = 8.83 y
;;         female n = 1,034  sd = 8.48 y
;;         pooled n = 2,079  sd = 8.66 y   (mean 0.00 by construction)
;;     One pooled knob rather than two, for the same reason grimaccel-sd-to-years
;;     is one: the status threshold is qualitative and a 4% sex difference does not
;;     move it. It enters NOTHING numeric below — the hazard uses years directly.
(= (linage-sd-to-years) 8.66)

;; (b) A contribution is a "driver" worth listing when it adds at least this many
;;     years. Presentation knob for `linage-drivers`; the decomposition itself
;;     lists every input regardless.
(= (linage-driver-threshold-years) 0.5)

;; ================================================================
;; 1. Accessors (space threaded explicitly; nothing when unmeasured)
;; ================================================================
(= (linage-delta $space $patient)
   (match $space (LinAgeDelta $patient $d) $d))

;; The KB biomarker an input reads out, or nothing (53 of 59 inputs).
(= (linage-readout $space $f)
   (match $space (MeasuresBiomarker $f $b) $b))

;; ================================================================
;; 2. The direction witness
;; ================================================================
;; True iff the patient's OWN evidence says the readout biomarker presents in the
;; harmful direction: patient-z above the shared elevated-z-threshold (explicit
;; MeasuredZ or NHANES-derived, single-valued — patient_profile.metta §2a), or, for
;; CurrentTobaccoExposure, a recorded current-smoker status. Both collapses are
;; tested before use: patient-z yields nothing for an unmeasured marker, and a
;; comparison against nothing would not be False, it would be unreduced.
(= (linage-witnessed $space $patient $b)
   (let* (($zs (collapse (patient-z $space $patient $b)))
          ($sm (collapse (match $space (PatientSmoking $patient CurrentSmoker) CurrentSmoker)))
          ($nosm (== $sm ()))
          ($noz  (== $zs ())))
      (if (== $b CurrentTobaccoExposure)
          (not $nosm)
          (if $noz
              False
              (let* (($z (car-atom $zs)) ($t (elevated-z-threshold)))
                 (> $z $t))))))

;; ================================================================
;; 3. Decomposition — every input, measured and imputed apart, causes credited
;;    only under a witness
;; ================================================================
;; The hallmark causes of a readout biomarker: every hallmark with a POSITIVE
;; chain to it through `infer` (deduped — `infer` yields one result per path, and
;; a cause reached by two paths is one cause). Reuses pln_counterfactual's Pos
;; filter idiom: a hallmark that would LOWER the marker does not explain a high one.
(= (linage-biomarker-causes $space $b)
   (unique-tuple
      (collapse
         (let $h (match $space (Inheritance $hh HallmarkOfAging) $hh)
            (let (signed Pos $tv) (infer $space $h $b) $h)))))

;; One input's record.
;;   (Contribution <Input> (years y) Measured|Imputed (ReadsOut None) (DrivenBy ()))
;;   (Contribution <Input> (years y) Measured|Imputed
;;                 (ReadsOut <B> (witnessed True|False)) (DrivenBy (<Hallmark> …)))
;; Causes are credited only when ALL of: the input reads out a KB biomarker, the
;; contribution is measured (an imputed value is the reference cohort's, not the
;; patient's), it is positive (ages the patient), and the witness holds.
(= (linage-contribution-record $space $patient $f $y $prov)
   (let $bs (collapse (linage-readout $space $f))
      (if (== $bs ())
          (Contribution $f (years $y) $prov (ReadsOut None) (DrivenBy ()))
          (let* (($b    (car-atom $bs))
                 ($w    (linage-witnessed $space $patient $b))
                 ($meas (== $prov Measured))
                 ($pos  (> $y 0.0))
                 ($ok1  (and $meas $pos))
                 ($ok   (and $ok1 $w))
                 ($causes (if $ok (linage-biomarker-causes $space $b) ())))
             (Contribution $f (years $y) $prov (ReadsOut $b (witnessed $w)) (DrivenBy $causes))))))

;; The years total for one provenance, straight from the atoms, with the GROUNDED
;; fold. MEASURED, not stylistic: a MeTTa recursion (car-atom / cdr-atom / +) over
;; the 33 measured records took 23 s and over 59 bare numbers 16 s, while
;; foldl-atom sums 59 numbers in 15 ms. Every sum in this file uses it; nothing
;; here recurses over a patient-sized tuple.
(= (linage-sum-years $space $patient $prov)
   (let $ys (collapse (match $space (LinAgeContribution $patient $f $y $prov) $y))
      (foldl-atom $ys 0.0 $acc $y (+ $acc $y))))

;; The decomposition. Measured and imputed inputs are separate tuples with
;; separate totals, and the remainder is the model's AGE TERM — LinAge2 adds
;; (age - training-mean-age) x w_age to the feature sum, so
;;     delta = attributed-measured + attributed-imputed + age-term-residual
;; holds exactly (up to float noise) and the residual is reported, never
;; absorbed. Tuple order is match order; the API sorts by |years|.
;;   (LinAgeDecomposition <P> (delta-years d) (Measured (…)) (Imputed (…))
;;      (attributed-measured am) (attributed-imputed ai) (age-term-residual r))
(= (linage-decomposition $space $patient)
   (let* (($d    (linage-delta $space $patient))
          ($meas (collapse
                    (let ($f $y) (match $space (LinAgeContribution $patient $ff $yy Measured) ($ff $yy))
                       (linage-contribution-record $space $patient $f $y Measured))))
          ($imp  (collapse
                    (let ($f $y) (match $space (LinAgeContribution $patient $ff $yy Imputed) ($ff $yy))
                       (linage-contribution-record $space $patient $f $y Imputed))))
          ($am   (linage-sum-years $space $patient Measured))
          ($ai   (linage-sum-years $space $patient Imputed))
          ($r1   (- $d $am))
          ($resid (- $r1 $ai)))
      (LinAgeDecomposition $patient (delta-years $d)
         (Measured $meas) (Imputed $imp)
         (attributed-measured $am) (attributed-imputed $ai) (age-term-residual $resid))))

;; The measured inputs that add at least linage-driver-threshold-years, as full
;; records (so a driver that reads out a witnessed biomarker arrives with its
;; causes). A de-aging input is never a driver.
(= (linage-drivers $space $patient)
   (collapse
      (let ($f $y) (match $space (LinAgeContribution $patient $ff $yy Measured) ($ff $yy))
         (let $t (linage-driver-threshold-years)
            (if (> $y $t)
                (linage-contribution-record $space $patient $f $y Measured)
                (superpose ()))))))

;; ================================================================
;; 4. Hazard and absolute risk
;; ================================================================
;; The hazard ratio per YEAR of delta, read off the Fong 2025 record for the
;; requested outcome — nothing for an outcome the record does not cover, so
;; (linage-risk … CoronaryHeartDisease) declines rather than borrowing.
(= (linage-hr $space $outcome)
   (match $space
      (, (HasPredictor $r LinAgeAccel) (HasOutcome $r $outcome) (EffectValue $r $hr))
      $hr))

;; The calibrated confidence of that record (calibrate-tv, the calibration layer),
;; discounted once for the modelling on top — the same single discount
;; pln_risk_prediction.metta pays.
(= (linage-conf $space $outcome)
   (match $space (, (HasPredictor $r LinAgeAccel) (HasOutcome $r $outcome))
      (cf-tv-c (calibrate-tv $space $r))))
(= (linage-confidence $space $outcome)
   (let $c (linage-conf $space $outcome)
      (* $c (risk-conf-discount))))

;; HR ^ years: the multiplier on the hazard relative to a same-age, same-sex
;; person with delta 0. The delta is the model's native unit, so no sd->years
;; knob enters (unlike the GrimAge layer, whose z has to be converted first).
(= (linage-multiplier-at $space $outcome $years)
   (let $hr (linage-hr $space $outcome)
      (pow-math $hr $years)))

;; The relative hazard — ALWAYS available for a patient with a delta, because it
;; needs no baseline.
;;   (LinAgeHazard <P> <outcome> (delta-years d) (hazard-multiplier m) (confidence c))
(= (linage-hazard $space $patient $outcome)
   (let* (($d (linage-delta $space $patient))
          ($m (linage-multiplier-at $space $outcome $d))
          ($c (linage-confidence $space $outcome)))
      (LinAgeHazard $patient $outcome (delta-years $d) (hazard-multiplier $m) (confidence $c))))

;; Absolute risk at a given delta, through the proportional-hazards SURVIVAL form
;;     risk = 1 - (1 - p0) ^ m
;; rather than pln_risk_prediction.metta's p0 x m. The two agree when p0 x m is
;; small (the CHD layer: baselines <= 0.25, multipliers ~1.5). They do not here:
;; an all-cause ten-year baseline can be 0.2-0.5 and a delta of +20 years is a
;; multiplier of ~6, so p0 x m would exceed 1. The survival form is the actual
;; Cox relationship and stays in [0, 1].
;;
;; p0 is `patient-baseline` FOR THIS OUTCOME (pln_risk_prediction.metta §1): a
;; data-backed NHANES BaselineRiskRecord when nhanes_baseline.metta and a generated
;; record are loaded, nothing otherwise — there is no curated all-cause table and
;; this layer adds none (docs/nhanes_integration.md §5: inventing one is the
;; incidence/mortality conflation that layer exists to refuse). So without the
;; ETL output this yields nothing, and `linage-hazard` above is the answer.
(= (linage-risk-at $space $patient $outcome $years)
   (let* (($p0 (patient-baseline $space $patient $outcome))
          ($m  (linage-multiplier-at $space $outcome $years))
          ($s0 (- 1.0 $p0))
          ($s  (pow-math $s0 $m)))
      (- 1.0 $s)))

;;   (RiskPrediction <P> <outcome> (point p) (ci lo hi) (confidence c)
;;                   (baseline p0) (multiplier m) (clock LinAge2))
;; Same report shape as predict-risk, plus (clock LinAge2) so the two clocks'
;; numbers can never be mistaken for each other. The interval is the same
;; multiplicative band as pln_risk_prediction.metta §3, clamped to 1.
(= (linage-risk $space $patient $outcome)
   (let* (($d   (linage-delta $space $patient))
          ($p0  (patient-baseline $space $patient $outcome))
          ($m   (linage-multiplier-at $space $outcome $d))
          ($p   (linage-risk-at $space $patient $outcome $d))
          ($c   (linage-confidence $space $outcome))
          ($rel (* (- 1.0 $c) (risk-ci-k)))
          ($lo  (* $p (- 1.0 $rel)))
          ($hir (* $p (+ 1.0 $rel)))
          ($hi  (if (> $hir 1.0) 1.0 $hir)))
      (RiskPrediction $patient $outcome
         (point $p) (ci $lo $hi) (confidence $c) (baseline $p0) (multiplier $m) (clock LinAge2))))

;; ================================================================
;; 5. Counterfactual in years, and its risk projection
;; ================================================================
;; ALL the drivers a lever stands for, as a tuple — the three lever shapes of
;; pln_counterfactual.metta §3 (an intervention's Neg targets; a driver itself; a
;; leaf marker's Pos sources), except that every candidate is kept rather than the
;; first. That matters for SmokingCessation, which now reduces two exposures
;; (SmokingPackYears from lifestyle_evidence.metta, CurrentTobaccoExposure from
;; linage2_core.metta) and must reach the cotinine input through the second. An
;; edge-less lever yields () and the counterfactual below yields nothing.
(= (linage-lever-drivers $space $lever)
   (let* (($neg (collapse (match $space (Effect $lever $d Neg $tv) $d)))
          ($out (collapse (match $space (Effect $lever $d $sg $tv) $d)))
          ($inc (collapse (match $space (Effect $x $lever Pos $tv) $x))))
      (if (== $neg ())
          (if (== $out ()) $inc ($lever))
          $neg)))

;; Driver -> readout biomarker transmission: the identity when the driver IS the
;; biomarker (cessation -> CurrentTobaccoExposure), else the strongest Pos chain
;; (pos-transmission, pln_counterfactual.metta), else nothing.
(= (linage-transmission $space $driver $b)
   (if (== $driver $b)
       (stv 1.0 1.0)
       (pos-transmission $space $driver $b)))

;; One measured input's expected reduction under do(lever): only when it reads
;; out a biomarker, the contribution is positive, the witness holds (§2), and the
;; driver reaches the biomarker. The reduction is (transmission strength) x
;; (the input's years), composed with the lever's own evidence edge exactly as
;; pln_counterfactual.metta §4 does (s_l x s, c_l x c x chain-discount), so an
;; intervention is never scored as if it abolished its driver.
;;   (lin-part <Input> (stv s c) <reduction-years>)
(= (linage-cf-feature $space $patient $lever $driver $f $y)
   (let $bs (collapse (linage-readout $space $f))
      (if (== $bs ())
          (superpose ())
          (let* (($b   (car-atom $bs))
                 ($w   (linage-witnessed $space $patient $b))
                 ($pos (> $y 0.0))
                 ($ok  (and $w $pos)))
             (if $ok
                 (let* (($base (linage-transmission $space $driver $b))
                        ($lvs  (collapse (match $space (Effect $lever $driver Neg $ltv) $ltv)))
                        ($tv   (if (== $lvs ())
                                   $base
                                   (let $l (car-atom $lvs)
                                      (stv (* (cf-tv-s $l) (cf-tv-s $base))
                                           (* (* (cf-tv-c $l) (cf-tv-c $base))
                                              (chain-discount))))))
                        ($s    (cf-tv-s $tv))
                        ($c    (cf-tv-c $tv))
                        ($red  (* $s $y)))
                    (lin-part $f (stv $s $c) $red))
                 (superpose ()))))))

;; Every (driver, measured input) pair, collapsed. Nothing at all for an
;; unresolvable lever.
(= (linage-cf-parts $space $patient $lever)
   (let $drivers (linage-lever-drivers $space $lever)
      (if (== $drivers ())
          (superpose ())
          (collapse
             (let $driver (superpose $drivers)
                (let ($f $y) (match $space (LinAgeContribution $patient $ff $yy Measured) ($ff $yy))
                   (linage-cf-feature $space $patient $lever $driver $f $y)))))))

;; Folds over the collapsed ((lin-part …) …) tuple — grounded (foldl-atom), see
;; linage-sum-years for why; every accessor result is let-forced before `+` / `*`.
(= (lin-red (lin-part $f $tv $r)) $r)
(= (lin-cc  (lin-part $f $tv $r)) (cf-tv-c $tv))
(= (lin-ff  (lin-part $f $tv $r)) $f)
(= (lin-sum-red $ps)
   (foldl-atom $ps 0.0 $acc $p (let $r (lin-red $p) (+ $acc $r))))
(= (lin-sum-redc $ps)
   (foldl-atom $ps 0.0 $acc $p
      (let* (($r (lin-red $p)) ($c (lin-cc $p)) ($rc (* $r $c))) (+ $acc $rc))))
;; Contribution-weighted mean confidence (0 when nothing moves — never a
;; divide-by-zero, never a confidence for a zero).
(= (lin-conf $ps)
   (let* (($den (lin-sum-red $ps)) ($num (lin-sum-redc $ps)))
      (if (> $den 0.0) (/ $num $den) 0.0)))
;; The credited input names, as a tuple (map, not fold: order is collapse order).
(= (lin-vias $ps)
   (collapse (let $p (superpose $ps) (lin-ff $p))))

;; do(lever) -> expected change in the LinAge2 delta, in YEARS. Neg = protective,
;; as everywhere (rank-score's convention); strength = the years removed;
;; confidence = the route-length-sensitive aggregate. A resolvable lever that
;; reaches no witnessed input honestly returns 0 with (Via ()).
;;   (LinAgeCounterfactual <P> <lever> (expected-delta-years d)
;;                         (signed Neg (stv |d| c)) (Via (<Input> …)))
(= (linage-counterfactual $space $patient $lever)
   (let* (($ps    (linage-cf-parts $space $patient $lever))
          ($total (lin-sum-red $ps))
          ($conf  (lin-conf $ps))
          ($vias  (lin-vias $ps))
          ($delta (if (> $total 0.0) (* -1.0 $total) 0.0)))
      (LinAgeCounterfactual $patient $lever
         (expected-delta-years $delta)
         (signed Neg (stv $total $conf))
         (Via $vias))))

(= (lin-cf-delta (LinAgeCounterfactual $p $l (expected-delta-years $d) $sg $v)) $d)
(= (lin-cf-conf  (LinAgeCounterfactual $p $l $ed (signed $s (stv $st $c)) $v)) $c)
(= (lin-cf-via   (LinAgeCounterfactual $p $l $ed $sg (Via $v))) $v)

;; The counterfactual in absolute-risk terms: recompute the risk at the reduced
;; delta. Needs the same outcome-matched baseline as linage-risk, so it yields
;; nothing without one.
;;   (ProjectedRisk <lever> <outcome> (point p') (reduction dr) (delta-years d)
;;                  (confidence c) (Via (…)) (clock LinAge2))
(= (linage-project-risk $space $patient $outcome $lever)
   (let* (($cf   (linage-counterfactual $space $patient $lever))
          ($dd   (lin-cf-delta $cf))
          ($cc   (lin-cf-conf $cf))
          ($via  (lin-cf-via $cf))
          ($d    (linage-delta $space $patient))
          ($dnew (+ $d $dd))
          ($p    (linage-risk-at $space $patient $outcome $d))
          ($pnew (linage-risk-at $space $patient $outcome $dnew))
          ($red  (- $p $pnew)))
      (ProjectedRisk $lever $outcome
         (point $pnew) (reduction $red) (delta-years $dd)
         (confidence $cc) (Via $via) (clock LinAge2))))

;; ================================================================
;; 6. Patient drivers (outcome defaults to AllCauseMortality — the one outcome
;;    the Fong 2025 record covers)
;; ================================================================
(= (linage-decomposition-patient $space $patient)
   (linage-decomposition $space $patient))
(= (linage-drivers-patient $space $patient)
   (linage-drivers $space $patient))
(= (linage-hazard-patient $space $patient)
   (linage-hazard $space $patient AllCauseMortality))
(= (linage-risk-patient $space $patient)
   (linage-risk $space $patient AllCauseMortality))
(= (linage-counterfactual-patient $space $patient $lever)
   (linage-counterfactual $space $patient $lever))
(= (linage-project-risk-patient $space $patient $lever)
   (linage-project-risk $space $patient AllCauseMortality $lever))

;; The four standing scenarios — the three casestudy levers plus smoking — as one
;; tuple, for the "what could I do about it" question.
(= (linage-scenarios-patient $space $patient)
   (collapse
      (linage-counterfactual-patient $space $patient
         (superpose (ChronicInflammation CellularSenescence InsulinResistance SmokingCessation)))))

;; ================================================================
;; 7. Smoke-test queries (shapes; the numbers live in tests/test_linage2.py,
;;    which renders a SYNTHETIC patient from tests/fixtures/linage2_response.json)
;; ================================================================
;; With a caller patient Caller_X carrying (LinAgeDelta Caller_X 8.28…), 33
;; measured and 26 imputed contributions, (MeasuredZ Caller_X HbA1c 1.6),
;; (PatientSmoking Caller_X CurrentSmoker):
;;
;; !(linage-decomposition-patient &self Caller_X)
;;   => (LinAgeDecomposition Caller_X (delta-years 8.28…)
;;         (Measured ((Contribution SerumCotinine (years 2.93…) Measured
;;                       (ReadsOut CurrentTobaccoExposure (witnessed True)) (DrivenBy ()))
;;                    (Contribution HbA1c (years 0.86…) Measured
;;                       (ReadsOut HbA1c (witnessed True)) (DrivenBy (DeregulatedNutrientSensing)))
;;                    (Contribution CRP (years -0.02…) Measured
;;                       (ReadsOut CRP (witnessed False)) (DrivenBy ()))
;;                    …))
;;         (Imputed (…))
;;         (attributed-measured 7.67…) (attributed-imputed 0.11…) (age-term-residual 0.49…))
;;
;; !(linage-hazard-patient &self Caller_X)
;;   => (LinAgeHazard Caller_X AllCauseMortality (delta-years 8.28…)
;;         (hazard-multiplier 2.09…) (confidence 0.54))          ; 1.093^8.28; 0.60 x 0.9
;;
;; !(linage-risk-patient &self Caller_X)         ; => (empty) without a NHANES baseline
;;
;; !(linage-counterfactual-patient &self Caller_X SmokingCessation)
;;   => (LinAgeCounterfactual Caller_X SmokingCessation (expected-delta-years -2.78…)
;;         (signed Neg (stv 2.78… 0.765)) (Via (SerumCotinine)))   ; 0.95 x 2.93; 0.85 x 1.0 x 0.9
;;
;; !(linage-counterfactual-patient &self Caller_X Metformin)
;;   => reaches HbA1c through InsulinResistance (witnessed), not SerumGlucose
;;      (no glucose z sent -> no witness -> not credited)
;;
;; For a NEVER smoker with the same atoms, the cessation counterfactual returns
;; expected-delta-years 0.0 with (Via ()) — the engine reads the precondition.
