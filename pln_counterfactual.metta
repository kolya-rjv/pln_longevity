;; pln_counterfactual.metta
;; Counterfactual GrimAge analysis with uncertainty propagation
;; (pipeline.md Demo 4).                                            [v1]
;;
;; The engine so far reasons FORWARD ("what does A cause?") via
;; (infer $space A C) -> (signed sign (stv s c)), and grounds a real patient via
;; (MeasuredZ Patient001 marker z). This layer adds the two missing pieces Demo 4
;; needs — and NOTHING an LLM can reproduce:
;;
;;   (a) a GrimAge COMPOSITION / decomposition edge. GrimAge is a composite of 8
;;       DNAm surrogates ((PartOf <surrogate> GrimAge) in grim_age_core.metta) but
;;       no truth-valued link maps a change in a component into a change in the
;;       clock. `grimage-weight` (§0) is that edge; `decompose-grimage` (§2) splits
;;       a patient's elevated GrimAge into per-component attributable shares, each
;;       traced to the upstream cause(s) `infer` says raise it. This is Demo §5.1
;;       "Causal Decomposition", and it finally CREDITS the AgeAccelGrim elevation
;;       that patient_profile.metta noted was "carried un-credited".
;;
;;   (b) a COUNTERFACTUAL do-operator (§3-4). `counterfactual $space $patient
;;       $lever $outcome` performs do($lever normalized): it resolves the lever to
;;       the harmful upstream DRIVER it represents, propagates the resulting
;;       reduction through each GrimAge component the driver reaches, aggregates
;;       via the composition edge, and returns the EXPECTED SIGNED change in the
;;       clock with propagated confidence (longer route -> lower c).
;;
;; THE CRP SUBTLETY (why this is PLN-not-LLM). CRP is a LEAF: the only edge is
;; (Effect ChronicInflammation CRP Pos ...) — CRP is a downstream READOUT, not a
;; GrimAge component, and has no outgoing edge. So "if my CRP were normal" has NO
;; direct mechanism to GrimAge. The correct causal reading is that CRP and the
;; inflammation-attributable GrimAge component (DNAmGDF15) are CO-EFFECTS of a
;; shared cause (ChronicInflammation). `resolve-lever` (§3) therefore routes a
;; marker lever THROUGH the cause that drives it — never via a fictitious
;; CRP -> GrimAge edge — and declines (~0, or omits) a lever with no path to any
;; clock surrogate. The restraint an LLM does not show.
;;
;; Load AFTER the whole inference + patient stack (it consumes infer / rank-score
;; conventions / patient MeasuredZ + elevated-z-threshold):
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile
;;   pln_counterfactual                  (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_counterfactual.py. Uses only core MeTTa builtins (match, superpose,
;; collapse, let/let*, car-atom/cdr-atom/cons-atom, +,-,*,/,>,==). As in every
;; layer the space $space is threaded explicitly (never a bare &self — a
;; (match &self ...) inside an imported module reads only that module's own space,
;; the cross-module gotcha docs/calibration_layer.md §10.1 documents), and every
;; arithmetic / list intermediate is let*-forced (grounded ops do NOT evaluate
;; their arguments — the discipline pln_intervention_ranking.metta §3 documents).
;; v1 locks the ARCHITECTURE (composition edge + do-transform); the constants
;; (§0 weights, the reduction model) are documented knobs to tune.

;; ================================================================
;; 0. KNOB: the GrimAge composition edge  (Demo §5.1)
;; ================================================================
;; How a Δ in one DNAm surrogate (in its own SD units) maps into a Δ in GrimAge
;; acceleration (in GrimAge SD units). This is a CURATED strength — hand-set, like
;; the mechanistic bridge priors — so it is quarantined here in a single, clearly
;; labelled knob (the "one place curated strengths live" discipline of
;; mechanistic_bridges.metta), never mixed into a raw evidence file.
;;
;; v1 is deliberately COARSE and honest: equal weights over the 8
;; (PartOf <surrogate> GrimAge) surrogates. The eight sum to 1.0, so if every
;; component were elevated by the same z the clock would move by that z — a
;; transparent normalization. A natural refinement is HasCpGCount-proportional
;; weighting (grim_age_core.metta reports CpG counts for PACKYRS/GDF15/PAI1); that
;; is a one-line change to THIS function and is left as a documented open question
;; (v1 locks the architecture, not the constant).
(= (grimage-weight $m) 0.125)          ; 1/8 over the eight PartOf surrogates

;; ================================================================
;; 1. Small helpers (TV accessors, patient elevation, transmission)
;; ================================================================
(= (cf-tv-s (stv $s $c)) $s)
(= (cf-tv-c (stv $s $c)) $c)

;; The patient's standardized value for a marker IF it is measured AND elevated
;; (> elevated-z-threshold, the patient-layer knob from patient_profile.metta);
;; otherwise nothing. So a Low/Normal or unmeasured component contributes zero and
;; drops out — we never invent an elevation. `elevated-z-threshold` is reused, not
;; forked, so the two layers stay in lockstep.
(= (patient-elevated-z $space $patient $m)
   (let $z (match $space (MeasuredZ $patient $m $zv) $zv)
      (if (> $z (elevated-z-threshold)) $z (superpose ()))))

;; The (stv s c) of the strongest / first POSITIVE transmission $a -> $b, or
;; nothing when $a does not raise $b. We keep only Pos derivations (the
;; explains-obs discipline of pln_abductive_diagnosis.metta): a driver "reaches" a
;; GrimAge component only when raising the driver would raise that component, so
;; normalizing the driver lowers it. `infer` is non-deterministic; collapse +
;; car-atom force ONE result (v1 — one derivation per pair; PLN revision over
;; multiple paths is the documented follow-up, deduction_layer.md §7.4).
(= (pos-transmission $space $a $b)
   (let $ps (collapse (let (signed Pos $tv) (infer $space $a $b) $tv))
      (if (== $ps ()) (superpose ()) (car-atom $ps))))

;; ================================================================
;; 2. Causal decomposition of an elevated GrimAge  (Demo §5.1)
;; ================================================================
;; For each GrimAge component the patient presents elevated, its attributable
;; contribution to the clock (weight × z) plus the upstream hallmark cause(s) that
;; `infer` says raise it. Components with no measured elevation or no causal driver
;; are simply absent from the component list and fall into the reported residual —
;; never invented (the honesty contract).

;; The upstream hallmark cause(s) of a component: every hallmark H with a POSITIVE
;; chain H -> M (reuses `infer` + the Pos filter — abduction, one component at a
;; time). Yields e.g. CellularSenescence for DNAmPAI1; CellularSenescence,
;; ChronicInflammation and MitochondrialDysfunction for DNAmGDF15.
(= (component-cause $space $m)
   (let $h (match $space (Inheritance $hh HallmarkOfAging) $hh)
      (let (signed Pos $tv) (infer $space $h $m) $h)))

;; One component's decomposition record, or nothing if the patient does not present
;; it elevated. (Component <M> (z z) (weight w) (contribution w*z) (DrivenBy (…))).
(= (decompose-one $space $patient $m)
   (let $z (patient-elevated-z $space $patient $m)
      (let* (($w       (grimage-weight $m))
             ($contrib (* $w $z))
             ($causes  (collapse (component-cause $space $m))))
         (Component $m (z $z) (weight $w) (contribution $contrib)
                    (DrivenBy $causes)))))

;; Sum the per-component contributions (let*-forced; + does not evaluate args).
(= (dc-contrib (Component $m (z $z) (weight $w) (contribution $c) (DrivenBy $db))) $c)
(= (sum-contribs $cs)
   (if (== $cs ()) 0.0
       (let* (($h (car-atom $cs)) ($r (sum-contribs (cdr-atom $cs))))
          (+ (dc-contrib $h) $r))))

;; decompose-grimage: the patient's elevated GrimAge split into attributable
;; components + the honest unattributed residual (measured AgeAccelGrim minus the
;; sum of component contributions — the part the coarse v1 weights and the
;; unmeasured components do not yet explain).
;;   (Decomposition <patient> (Components (…)) (attributed a) (residual r))
(= (decompose-grimage $space $patient)
   (let $comps
        (collapse
           (let $m (match $space (PartOf $mm GrimAge) $mm)
              (decompose-one $space $patient $m)))
      (let* (($attributed (sum-contribs $comps))
             ($grimz (match $space (MeasuredZ $patient AgeAccelGrim $z) $z))
             ($residual (- $grimz $attributed)))
         (Decomposition $patient
            (Components $comps)
            (attributed $attributed)
            (residual $residual)))))

;; A single component's share of the patient's GrimAge elevation (the "what is
;; DNAmPAI1's share of my GrimAge?" query). Returns the Component record, or
;; nothing if that component is not elevated for the patient.
(= (grimage-share $space $patient $m)
   (decompose-one $space $patient $m))

;; ================================================================
;; 3. Lever resolution — route to the harmful DRIVER  (the CRP subtlety)
;; ================================================================
;; do($lever normalized) is really "normalize the harmful upstream driver the
;; lever stands for". Three lever shapes all resolve to that driver, so the §4
;; do-operator is uniform (it always normalizes a Pos-transmitting driver and
;; never has to reason about a fictitious marker -> clock edge):
;;
;;   (1) an INTERVENTION — a node with an outgoing (Effect lever D Neg …): it
;;       REDUCES D, so normalizing-via-this-intervention == normalizing D.
;;       DasatinibPlusQuercetin -> CellularSenescence;  Metformin -> InsulinResistance.
;;   (2) a LEAF MARKER — no outgoing Effect, but an incoming (Effect D lever Pos …):
;;       lever is a CO-EFFECT of D, so "if the marker were normal" == normalize D.
;;       CRP -> ChronicInflammation.   (NEVER a CRP -> GrimAge edge.)
;;   (3) a DRIVER — a cause with outgoing Pos edges: itself.
;;       ChronicInflammation, CellularSenescence, InsulinResistance, ...
;;
;;   (4) NO Effect edge at all -> UNRESOLVABLE -> (superpose ()) -> the whole
;;       counterfactual yields nothing (the lever is omitted, not mis-placed —
;;       e.g. Elamipretide, which has only a HallmarkInterventionEvidence record).
;;
;; Every collapse is let*-forced so the (== … ()) tests see reduced tuples.
(= (resolve-lever $space $lever)
   (let* (($neg (collapse (match $space (Effect $lever $d Neg $tv) $d)))
          ($out (collapse (match $space (Effect $lever $d $sg $tv) $d)))
          ($inc (collapse (match $space (Effect $x $lever Pos $tv) $x))))
      (if (== $neg ())
          (if (== $out ())
              (if (== $inc ())
                  (superpose ())          ; (4) no edges -> unresolvable -> omitted
                  (car-atom $inc))        ; (2) leaf marker -> its driver
              $lever)                     ; (3) has outgoing edges -> a driver, itself
          (car-atom $neg))))              ; (1) intervention -> the node it reduces

;; ================================================================
;; 4. The counterfactual do-operator
;; ================================================================
;; Per GrimAge component M that the resolved driver raises (Pos) AND the patient
;; presents elevated: the expected reduction in M ≈ (transmission strength) × (M's
;; current elevated z), and its GrimAge share (grimage-weight × reduction).
;;   (cf-part <M> (stv s c) <reduction> <deltaM>)
(= (cf-component $space $patient $driver $m)
   (let* (($z  (patient-elevated-z $space $patient $m))
          ($tv (pos-transmission $space $driver $m)))
      (let* (($s   (cf-tv-s $tv))
             ($c   (cf-tv-c $tv))
             ($red (* $s $z))
             ($dm  (* (grimage-weight $m) $red)))
         (cf-part $m (stv $s $c) $red $dm))))

;; Collapse the per-component parts for a resolved driver (empty tuple when the
;; driver reaches no elevated GrimAge component — the honest "no path" case).
(= (cf-parts $space $patient $driver)
   (collapse
      (let $m (match $space (PartOf $mm GrimAge) $mm)
         (cf-component $space $patient $driver $m))))

;; Folds over the collapsed ((cf-part …) …) tuple — all let*-forced.
(= (cf-dm (cf-part $m $tv $red $dm)) $dm)
(= (cf-cc (cf-part $m $tv $red $dm)) (cf-tv-c $tv))
(= (cf-mm (cf-part $m $tv $red $dm)) $m)

;; Σ deltaM = the total GrimAge-acceleration reduction (magnitude).
(= (cf-sum-dm $ps)
   (if (== $ps ()) 0.0
       (let* (($h (car-atom $ps)) ($r (cf-sum-dm (cdr-atom $ps))))
          (+ (cf-dm $h) $r))))

;; Σ deltaM · c — numerator of the contribution-weighted mean confidence.
(= (cf-sum-dmc $ps)
   (if (== $ps ()) 0.0
       (let* (($h (car-atom $ps)) ($r (cf-sum-dmc (cdr-atom $ps))))
          (+ (* (cf-dm $h) (cf-cc $h)) $r))))

;; The estimate's confidence: contribution-weighted mean of the per-component
;; transmission confidences. Because a longer causal route carries a lower c
;; (chain-discount, per hop), a lever whose components sit further downstream
;; yields a LOWER counterfactual confidence — "uncertainty grows with route
;; length". Guarded so the no-path case is 0, not a divide-by-zero.
(= (cf-conf $ps)
   (let* (($den (cf-sum-dm $ps)) ($num (cf-sum-dmc $ps)))
      (if (> $den 0.0) (/ $num $den) 0.0)))

;; The list of credited component names — the (Via …) provenance.
(= (cf-vias $ps)
   (if (== $ps ()) ()
       (let* (($h (car-atom $ps)) ($r (cf-vias (cdr-atom $ps))))
          (cons-atom (cf-mm $h) $r))))

;; counterfactual: do($lever normalized) -> expected signed Δ in $outcome (the
;; GrimAge clock). The sign reuses the Neg=beneficial convention (rank-score,
;; pln_intervention_ranking.metta): a REDUCTION in a harmful clock is protective,
;; so the result is (signed Neg …) with a NEGATIVE expected-delta. Strength = the
;; magnitude of that reduction (rank-score-able); confidence = the route-length-
;; sensitive aggregate above. A lever that reaches no clock surrogate returns
;; expected-delta 0 with an empty (Via ()) — honest, not fabricated; an
;; unresolvable lever (no result from resolve-lever) yields nothing at all.
;;   (Counterfactual <lever> <outcome>
;;      (expected-delta d) (signed Neg (stv s c)) (Via (M …)))
(= (counterfactual $space $patient $lever $outcome)
   (let $driver (resolve-lever $space $lever)
      (let* (($ps    (cf-parts $space $patient $driver))
             ($total (cf-sum-dm $ps))
             ($conf  (cf-conf $ps))
             ($vias  (cf-vias $ps))
             ($delta (if (> $total 0.0) (* -1.0 $total) 0.0)))   ; avoid -0.0 for the honest no-path case
         (Counterfactual $lever $outcome
            (expected-delta $delta)
            (signed Neg (stv $total $conf))
            (Via $vias)))))

;; ================================================================
;; 5. Patient drivers (the three casestudy §5.2 scenarios)
;; ================================================================
;; counterfactual-patient defaults the outcome to AgeAccelGrim, the GrimAge
;; acceleration clock the composition edge (§0) models.
(= (counterfactual-patient $space $patient $lever)
   (counterfactual $space $patient $lever AgeAccelGrim))

;; The three casestudy §5.2 scenarios as distinct levers, collapsed into one
;; tuple. `superpose` fans the three lever ATOMS; counterfactual-patient runs for
;; each; `collapse` gathers the three results:
;;   A "reduce inflammation only"    -> ChronicInflammation  (reaches DNAmGDF15 only)
;;   B "clear senescent cells"       -> CellularSenescence    (reaches BOTH components)
;;   C "address metabolic dysfunction" -> InsulinResistance   (reaches NO component -> ~0)
;; EXPECTED: |Δ_B| > |Δ_A| (senescence is upstream of inflammation, so it reaches
;; both DNAmPAI1 and DNAmGDF15, a bigger reduction than inflammation alone), and
;; Δ_C ≈ 0 (the metabolic axis terminates at FastingGlucose -> CHD and reaches no
;; clock surrogate — the honest "this lever does not move THIS clock").
(= (counterfactual-scenarios $space $patient)
   (collapse
      (counterfactual-patient $space $patient
         (superpose (ChronicInflammation CellularSenescence InsulinResistance)))))

;; ================================================================
;; 6. Smoke-test queries (intended values; live floats carry trailing noise —
;;    tests/test_counterfactual.py compares with tolerance)
;; ================================================================
;; Load order: system_types logical_predicates epistemic_calibration
;;             grim_age_core grim_age_lu2019_evidence evidence_calibration
;;             hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;             mechanistic_bridges pln_deduction pln_intervention_ranking
;;             pln_abductive_diagnosis patient_profile pln_counterfactual
;;
;; Decompose Patient001's elevated GrimAge into attributable components:
;; !(decompose-grimage &self Patient001)
;;   => (Decomposition Patient001
;;         (Components ( (Component DNAmGDF15 (z 1.5) (weight 0.125) (contribution 0.1875)
;;                          (DrivenBy (CellularSenescence ChronicInflammation MitochondrialDysfunction)))
;;                       (Component DNAmPAI1  (z 1.8) (weight 0.125) (contribution 0.225)
;;                          (DrivenBy (CellularSenescence))) ))
;;         (attributed 0.4125) (residual 1.1875))
;;
;; Scenario A — normalize inflammation (reaches DNAmGDF15 only, single hop):
;; !(counterfactual-patient &self Patient001 ChronicInflammation)
;;   => (Counterfactual ChronicInflammation AgeAccelGrim (expected-delta -0.121875)
;;         (signed Neg (stv 0.121875 0.65)) (Via (DNAmGDF15)))
;;
;; Scenario B — clear senescent cells (reaches DNAmPAI1 + DNAmGDF15): a BIGGER
;; reduction than A but LOWER confidence (its components sit 2–3 hops downstream):
;; !(counterfactual-patient &self Patient001 CellularSenescence)
;;   => (Counterfactual CellularSenescence AgeAccelGrim (expected-delta -0.22192…)
;;         (signed Neg (stv 0.22192… 0.31764…)) (Via (DNAmPAI1 DNAmGDF15)))
;;
;; Scenario C — metabolic dysfunction reaches NO clock surrogate (honest ~0):
;; !(counterfactual-patient &self Patient001 InsulinResistance)
;;   => (Counterfactual InsulinResistance AgeAccelGrim (expected-delta 0.0)
;;         (signed Neg (stv 0.0 0.0)) (Via ()))
;;
;; "If my CRP were normal?" — routed THROUGH ChronicInflammation (same as A), NOT
;; a fictitious CRP -> GrimAge edge:
;; !(counterfactual-patient &self Patient001 CRP)             ; == scenario A
;;
;; A lever with no Effect edge is omitted, not mis-placed:
;; !(counterfactual-patient &self Patient001 Elamipretide)    ; => (empty)
