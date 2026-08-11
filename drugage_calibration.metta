;; drugage_calibration.metta
;; DrugAge Experiment rows -> calibrated, SIGNED lifespan Effect links.   [v1]
;;
;; This is the ETL->inference bridge for the DrugAge vertical slice. The ETL
;; (drugage_etl.py) emits ~3.4k raw, source-faithful measurement rows:
;;
;;   (InstanceOf            DrugAgeRow_0 Experiment)
;;   (UsesIntervention      DrugAgeRow_0 Ethanol)
;;   (UsesSpecies           DrugAgeRow_0 Drosophila_mojavensis)
;;   (AvgLifespanChangePercent DrugAgeRow_0 81.29)
;;   (AvgLifespanSignificance  DrugAgeRow_0 Unreported)
;;   (IsITPStudy            DrugAgeRow_908)          ; 164 rows: ITP gold-standard
;;   (ReportedIn            DrugAgeRow_0 PMID_13369) ; provenance
;;
;; Nothing lifted these into the (Effect ...) links deduction/ranking consume.
;; This layer is that bridge: it LIFTS one row on the fly into
;;
;;   (Effect <compound> Lifespan <sign> (stv s c))
;;
;; exactly as pln_deduction.metta's `link-effect` lifts a GrimAge empirical
;; record. Raw rows are never rewritten (the calibration-layer immutability
;; invariant, calibration_layer.md §9); the lift is a pure function of the row.
;;
;; Design rationale, calibration map, gotchas: docs/etl_inference_wiring.md
;;
;; Load AFTER (the DrugAge inference stack — see pln_runner.DRUGAGE_STACK):
;;   system_types.metta            (Lifespan, Outcome, EvidenceCategory, ...)
;;   logical_predicates.metta      (Inheritance / InstanceOf predicates)
;;   epistemic_calibration.metta   (evidence-confidence — the confidence authority)
;;   species_taxonomy.metta        (species -> Vertebrate/Invertebrate/Fungi/Protozoa)
;;   evidence_calibration.metta    (calibrate-tv — kept loadable for link-effect eq2)
;;   pln_deduction.metta           (Effect, infer, combine-hop, sign-product, Pos/Neg)
;;   pln_intervention_ranking.metta(rank-interventions, rank-score, sort-scored)
;;   drugage_calibration.metta     (this file — loads LAST)
;;   + a FILTERED slice of DrugAge rows (NEVER the full dump — hyperon panics;
;;     pln_runner.build_drugage_slice / drugage_selector selects the rows)
;;
;; STATUS: v1, verified against hyperon 0.2.10 (tests/test_drugage_calibration.py).
;; Buckets are deliberately coarse: v1 locks the architecture, not the constants
;; (calibration_layer.md §8). Every knob is in §1 so tuning stays local.

;; ================================================================
;; 0. Types introduced here
;; ================================================================
;; Lifespan is already (: Lifespan Phenotype) in system_types.metta — reused as
;; the beneficial target of the lifted link. We add Mortality as the HARMFUL
;; terminal outcome the ranking sorts against (see §6, the sign decision).
(: Mortality Outcome)

;; ================================================================
;; 1. KNOBS — every tunable constant lives here (calibration §8, deduction §8)
;; ================================================================
;; Strength half-saturation (percent). AvgLifespanChangePercent is unbounded in
;; [0, inf), so it is squashed to [0,1] by the saturating transform
;;   s = |pct| / (|pct| + k)   (§4).
;; k is the change (%) at which strength = 0.5. k=20 puts a strong replicated
;; extension (~+20-25%, e.g. ITP rapamycin) near 0.5-0.56 and a huge +80% effect
;; near 0.8. Raise k to be more conservative about large reported effects.
(= (lifespan-halfsat) 20.0)

;; Significance gate (confidence CAP applied to NON-ITP rows; §5). A non-ITP
;; result that did not reach significance is weak evidence, so it caps confidence.
;; Significant imposes no cap (1.0); an unreported test is worse than a reported
;; significant one but better than a reported NULL.
(= (sig-gate Significant)    1.0)
(= (sig-gate Unreported)     0.6)
(= (sig-gate NotSignificant) 0.4)   ; NB: ETL emits NotSignificant (not the
                                    ;     NonSignificant typo in system_types.metta)

;; ================================================================
;; 2. Row accessors (space threaded explicitly — the cross-module &self trap,
;;    calibration_layer.md §10.1; callers pass the top-level &self)
;; ================================================================
(= (drugage-compound     $s $row) (match $s (UsesIntervention $row $c) $c))
(= (drugage-species      $s $row) (match $s (UsesSpecies $row $sp) $sp))
(= (drugage-change       $s $row) (match $s (AvgLifespanChangePercent $row $p) $p))
(= (drugage-significance $s $row) (match $s (AvgLifespanSignificance $row $g) $g))

;; Existence check: does the space hold this exact atom? `collapse` forces the
;; match stream to a concrete tuple so `==` (grounded — does NOT evaluate its
;; args) can compare it to the empty tuple. Returns a real True/False for `if`.
(= (drugage-has? $s $atom)
   (let $hits (collapse (match $s $atom found))
        (if (== $hits ()) False True)))

(= (drugage-itp?         $s $row) (drugage-has? $s (IsITPStudy $row)))
(= (drugage-significant? $s $row) (drugage-has? $s (AvgLifespanSignificance $row Significant)))

;; ================================================================
;; 3. Species -> evidence tier   (reuses species_taxonomy + evidence-confidence)
;; ================================================================
;; "Shown only in a worm is weaker than shown in a mouse" lives here. We map the
;; species' CLADE (from species_taxonomy.metta) to an EvidenceCategory, then read
;; its number from the single confidence authority (epistemic_calibration.metta).
;; The invertebrate/fungi rungs are BORROWED for their numeric value pending a
;; dedicated ModelOrganism tier — the same "no perfect tier yet, map
;; conservatively" move mechanistic_bridges.metta makes for mechanistic
;; consensus (calibration_layer.md open-Q #2/#4). Coarse by design (§8).
(= (clade-category Vertebrate)   AnimalStudies_Single)  ; 0.50  (mouse, rat, fish)
(= (clade-category Invertebrate) InVitro)               ; 0.35  (worm, fly)
(= (clade-category Fungi)        TraditionalUse)        ; 0.20  (yeast)
(= (clade-category Protozoa)     TraditionalUse)        ; 0.20

;; Look up the clade, default to the invertebrate rung for an untaxonomised
;; species (conservative middle: not trusted like a mammal, not sunk like a
;; protozoan). car-atom forced through `let` (grounded op, no arg eval).
(= (species-tier-conf $s $sp)
   (let $clades (collapse (match $s (Inheritance $sp $clade) $clade))
        (if (== $clades ())
            (evidence-confidence InVitro)
            (let $c0 (car-atom $clades)
                 (evidence-confidence (clade-category $c0))))))

;; ================================================================
;; 4. Strength  <-  |AvgLifespanChangePercent|   (saturating, §1 knob)
;; ================================================================
;; Magnitude only (sign is a separate axis, §6). Every intermediate is forced
;; through let* so the grounded arithmetic ops receive REDUCED numbers — skip
;; this and (+ $m (lifespan-halfsat)) stays symbolic and the division misfires
;; (the same discipline pln_intervention_ranking.metta §3 documents).
(= (drugage-row-strength $s $row)
   (let* (($pct (drugage-change $s $row))
          ($m   (if (< $pct 0.0) (- 0.0 $pct) $pct))
          ($k   (lifespan-halfsat))
          ($den (+ $m $k)))
     (/ $m $den)))

;; ================================================================
;; 5. Confidence  <-  MIN(evidence tier, significance gate)   (weakest link)
;; ================================================================
;; Tier: an ITP row is the gold-standard replicated-mouse program, so it earns
;; the ITP confidence (0.90) REGARDLESS of outcome — significance instead selects
;; ITP_Positive vs ITP_Negative (both 0.90: we trust a well-run NULL too, which
;; is precisely what down-weights resveratrol via near-zero STRENGTH, not low
;; confidence). A non-ITP row earns its species tier and is additionally capped
;; by the significance gate (§1). c = min(tier, gate): the weakest link governs
;; (calibration_layer.md §3.2).
(= (itp-category $s $row)
   (if (drugage-significant? $s $row) ITP_Positive ITP_Negative))

(= (drugage-tier-conf $s $row)
   (if (drugage-itp? $s $row)
       (let $cat (itp-category $s $row) (evidence-confidence $cat))
       (species-tier-conf $s (drugage-species $s $row))))

;; ITP folds significance into the tier, so its gate is neutral (1.0).
(= (drugage-sig-gate $s $row)
   (if (drugage-itp? $s $row)
       1.0
       (let $g (drugage-significance $s $row) (sig-gate $g))))

(= (min-num $a $b) (if (< $a $b) $a $b))

(= (drugage-confidence $s $row)
   (let* (($tier (drugage-tier-conf $s $row))
          ($gate (drugage-sig-gate $s $row)))
     (min-num $tier $gate)))

;; ================================================================
;; 6. Sign  <-  direction of the lifespan change   (THE SIGN DECISION)
;; ================================================================
;; Effect sign convention (pln_deduction.metta §0): Pos = <from> RAISES <to>.
;; A compound that EXTENDS lifespan (pct>0) RAISES Lifespan  -> Pos.
;; A compound that SHORTENS lifespan (pct<0) LOWERS Lifespan -> Neg.
;;
;; SIGN TRAP: Lifespan is a *beneficial* outcome, so Pos is GOOD here — the
;; opposite of the harmful CHD/mortality axis rank-score was written for
;; (there Neg=protective=good). Ranking on Lifespan directly would need an
;; inverted scorer, forking the sign convention — the exact footgun that
;; silently inverts a ranking. Instead we keep ONE global convention
;; (Neg = beneficial) by chaining Lifespan -> Mortality (§7): extending lifespan
;; (Pos) reduces mortality (Neg), and Pos*Neg = Neg = protective, which
;; rank-score already scores as good. See docs/etl_inference_wiring.md.
(= (lifespan-sign $pct) (if (< $pct 0.0) Neg Pos))

;; ================================================================
;; 7. THE LIFT — one row -> one signed, calibrated Effect  (the deliverable)
;; ================================================================
;; Pure function of (space, row): reads the row's compound + change (both must
;; be present) and emits the canonical link. Provenance (ReportedIn PMID) stays
;; in the raw row, reachable from <compound>/<row> for the audit trail; nothing
;; is materialised or rewritten.
;; NB: Effect's args are typed Atom (pln_deduction §0), which SUPPRESSES their
;; reduction — a raw (Effect .. (drugage-row-strength ..) ..) would keep the
;; call symbolic (exactly how curated bridges store (evidence-confidence ..)
;; unevaluated). We force sign/strength/confidence through let* FIRST so the
;; emitted link carries concrete numbers, matching the deliverable
;; (Effect <compound> Lifespan <sign> (stv s c)).
(= (drugage-effect $s $row)
   (match $s (, (UsesIntervention $row $comp) (AvgLifespanChangePercent $row $pct))
      (let* (($sgn (lifespan-sign $pct))
             ($str (drugage-row-strength $s $row))
             ($c   (drugage-confidence $s $row)))
         (Effect $comp Lifespan $sgn (stv $str $c)))))

;; ================================================================
;; 8. Lifespan -> Mortality bridge  (the sign adapter)
;; ================================================================
;; Extending lifespan reduces mortality (Neg). This single curated hop is what
;; lets a beneficial lifespan measurement plug into the Neg=beneficial ranking
;; convention unchanged. Its TV is deliberately NEUTRAL (stv 1.0 1.0): it is a
;; definitional sign flip, not new evidence, and being CONSTANT across every
;; compound it cannot change the ranking ORDER — it only reconciles the sign and
;; incurs the standard per-hop confidence discount (combine-tv, 0.9). The real
;; cross-species surrogacy uncertainty is already carried by the species tier in
;; the compound->Lifespan link's confidence (§3).
(Effect Lifespan Mortality Neg (stv 1.0 1.0))

;; ================================================================
;; 9. Step `infer` through the on-the-fly DrugAge lift
;; ================================================================
;; pln_deduction's transitive `infer` steps only through MATERIALISED (Effect ..)
;; bridges; DrugAge links are lifted on the fly, so we add ONE equation that
;; enumerates a compound's rows, lifts each with drugage-effect, and chains it
;; into the recursive result (partially realises deduction_layer.md §7.3,
;; "stepping through empirical links mid-chain", scoped to DrugAge). It only
;; fires for atoms that appear as a DrugAge intervention, so it is inert for
;; every non-DrugAge query (and never loaded by the existing test KB).
;;   infer <compound> Mortality  =>  lift <compound>->Lifespan, then Lifespan->Mortality
;;                               =>  (signed Neg (stv s c)) for a life-EXTENDER.
(= (infer $s $a $c)
   (match $s (UsesIntervention $r $a)
      (let (Effect $x Lifespan $sgn $tv) (drugage-effect $s $r)
           (combine-hop $sgn $tv (infer $s Lifespan $c)))))

;; ================================================================
;; 10. Smoke-test queries (verified, hyperon 0.2.10 — see the test module)
;; ================================================================
;; With the DrugAge stack + a slice loaded into one space:
;;
;; ONE ITP-positive row (rapamycin, mouse, +13%, Significant):
;; !(drugage-effect &self DrugAgeRow_984)  ; (Effect Rapamycin Lifespan Pos (stv 0.393 0.9))
;;
;; ITP-negative null (resveratrol, mouse, 0%, NotSignificant) — high confidence,
;; ~zero strength: the honest down-weight (pipeline.md App. B.3 / Demo 6):
;; !(drugage-effect &self DrugAgeRow_1087) ; (Effect Resveratrol Lifespan Pos (stv 0.0 0.9))
;;
;; Chained to the harmful axis (net protective for a life-extender):
;; !(infer &self Rapamycin Mortality)      ; (signed Neg (stv 0.393 0.81))
;;
;; Ranked over a real slice (most protective first; resveratrol sinks to ~0):
;; !(rank-interventions &self (Rapamycin Resveratrol) Mortality)
