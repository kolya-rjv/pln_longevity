;; supplement_evidence.metta
;; Curated Layer-4 supplement / nutraceutical knowledge  (pipeline.md §2.1.4,
;; Demo 6).                                                            [v1]
;;
;; Demos 1-5 reason over pharmaceutical / hallmark evidence. Demo 6 (personalized
;; supplement recommendations, pipeline.md §4.6) needs a FOURTH knowledge layer the
;; pipeline calls out explicitly: supplements and nutraceuticals, represented with
;; APPROPRIATELY LOWER confidence than pharmaceuticals, plus the modifier atoms
;; (evidence level, safety, interactions) a responsible recommender must carry. The
;; hooks for exactly this were left waiting in logical_predicates.metta — Interaction
;; is declared there but EMPTY, and EvidenceLevel / SafetyProfile sit in its §9
;; "FUTURE PREDICATES (stubs for later phases)". This file is that phase.
;;
;; HONESTY CONTRACT — identical to mechanistic_bridges.metta, and for the same reason:
;;   * confidence  is DERIVED, never invented: a lookup into the single
;;                 evidence-confidence authority (epistemic_calibration.metta) by the
;;                 supplement's evidence tier. Supplement tiers deliberately land LOWER
;;                 than pharmaceutical trials (Animal / Single-trial / InVitro), so a
;;                 supplement recommendation carries a smaller confidence than a drug —
;;                 the Layer-4 intellectual-honesty point (pipeline.md §2.1.4, §4.6).
;;   * strength    is a documented CURATED PRIOR (expert magnitude estimate), quarantined
;;                 here and clearly labelled, exactly like the mechanistic bridge priors —
;;                 never mixed into a raw evidence file. v1 locks the architecture; these
;;                 constants are the tuning surface (calibration_layer.md §8).
;;
;; THE RESVERATROL SUBTLETY (why this is PLN-not-LLM). Resveratrol is given a plausible
;; senescence-clearing Effect link (its SIRT1 story), so by MECHANISM it looks relevant
;; to a senescence-dominant patient — an LLM recommends it on exactly that basis. But it
;; also carries (EvidenceLevel Resveratrol ITP_Negative): the NIA Interventions Testing
;; Program found NO lifespan effect (pipeline.md App. B.3). The recommender's tier map
;; (pln_supplement_recommendation.metta) turns that ITP_Negative into a VETO —
;; NotRecommended, regardless of the mechanism. The restraint an LLM does not show.
;;
;; Load AFTER the whole inference + patient stack (it references Supplement,
;; evidence-confidence, the hallmark/pathway nodes, and PatientProfile):
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile pln_counterfactual pln_risk_prediction
;;   supplement_evidence                 (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_supplement_recommendation.py. Pure DATA (Effect links + modifier atoms) —
;; it defines no recursion, so it needs no runtime beyond the atoms themselves. Every
;; NEW Effect link adds an OUTGOING edge from a NEW supplement node (Omega3 / NMN /
;; Resveratrol); none adds an incoming edge to a node existing forward queries traverse,
;; so infer / counterfactual / diagnose from existing nodes are unchanged.

;; ================================================================
;; 0. Types introduced here (the Layer-4 modifier predicates)
;; ================================================================
;; EvidenceLevel / SafetyProfile were stubbed in logical_predicates.metta §9; we
;; realise them here, tied to the existing EvidenceCategory authority so the tier a
;; supplement claims is the SAME vocabulary the calibration layer already scores.
(: EvidenceLevel  (-> Intervention EvidenceCategory Atom))

(: SafetyLabel Type)
(: GenerallyWellTolerated SafetyLabel)
(: UseWithCaution         SafetyLabel)   ; e.g. GI effects, CYP interactions
(: NotWellStudied         SafetyLabel)   ; sparse human safety data
(: SafetyProfile  (-> Intervention SafetyLabel Atom))

;; The patient's current medications — the context the interaction check reads. Kept
;; HERE (not in patient_profile.metta) so the 107 pre-Demo-6 tests are untouched; it
;; is patient-context data the recommender consumes, declared alongside the modifier
;; atoms that use it.
(: CurrentMedication (-> PatientProfile Intervention Atom))

;; ================================================================
;; 1. New supplement nodes (typing)
;; ================================================================
;; Fisetin, NMN, Spermidine, Berberine, AlphaKetoglutarate, UrolithinA already carry
;; (Inheritance … Supplement). Omega3 and Resveratrol are new to the curated KB.
(Inheritance Omega3      Supplement)
(Inheritance Resveratrol Supplement)

;; ================================================================
;; 2. Curated protective mechanisms  (the personalization substrate)
;; ================================================================
;; A supplement is RELEVANT to a patient (pln_supplement_recommendation.metta) exactly
;; when it protectively reduces a marker the patient presents elevated — computed by the
;; existing `patient-relevance` (patient_profile.metta) over these Effect links. So each
;; link below is the mechanism by which a candidate reaches a patient's biomarker
;; profile. strength = curated magnitude prior; confidence = evidence-confidence lookup.

;; Omega-3: anti-inflammatory (Calder 2017; Tan 2022, Nat Commun). REDUCES chronic
;; inflammation (Neg), so it reaches CRP and the inflammation-attributable DNAmGDF15
;; surrogate that ChronicInflammation drives (mechanistic_bridges.metta). Extensive
;; human evidence -> the strongest supplement tier here.
(Effect Omega3 ChronicInflammation Neg
   (stv 0.60 (evidence-confidence MultipleHumanTrials)))

;; NMN: NAD+ precursor -> improves mitochondrial function, i.e. REDUCES mitochondrial
;; dysfunction (Neg), reaching the mito-attributable DNAmGDF15 surrogate
;; (MitochondrialDysfunction -> DNAmGDF15 in mechanistic_bridges.metta). Preliminary
;; human data (Covarrubias 2021; LopezOtin2023_NMN_Human) -> a single-trial tier.
(Effect NMN MitochondrialDysfunction Neg
   (stv 0.55 (evidence-confidence SingleHumanTrial)))

;; Resveratrol: the CLAIMED SIRT1 / senolytic mechanism (Zhu 2015 lineage) -> reduces
;; cellular senescence (Neg), so by MECHANISM it reaches the senescence markers a
;; senescence-dominant patient shows (DNAmPAI1, DNAmGDF15, CRP). Deliberately given a
;; plausible in-vitro-grade prior so the demo can SHOW the recommender vetoing it on
;; the negative trial below — not on a missing mechanism.
(Effect Resveratrol CellularSenescence Neg
   (stv 0.50 (evidence-confidence InVitro)))

;; NB: Fisetin -> CellularSenescence (Neg) and Berberine -> InsulinResistance (Neg)
;; already exist in mechanistic_bridges.metta and are NOT redeclared here.

;; ================================================================
;; 3. Evidence level per supplement  (the tier the recommender buckets on)
;; ================================================================
;; One EvidenceLevel per supplement (single atom -> deterministic tier lookup). The
;; category is read straight back through evidence-confidence, so a supplement's
;; recommendation confidence is its evidence tier's confidence — and every one of these
;; sits below RCT_Human / ITP, the Layer-4 "supplements carry lower confidence" invariant.
(EvidenceLevel Omega3      MultipleHumanTrials)   ; 0.85  extensive human (Tier1)
(EvidenceLevel Berberine   MultipleHumanTrials)   ; 0.85  multiple trials  (Tier1)
(EvidenceLevel Fisetin     AnimalStudies_Single)  ; 0.50  animal + early human (Tier2)
(EvidenceLevel NMN         SingleHumanTrial)       ; 0.70  preliminary human (Tier2)
(EvidenceLevel Resveratrol ITP_Negative)           ; 0.90  gold-standard NULL (NotRecommended)

;; ================================================================
;; 4. Safety profiles
;; ================================================================
(SafetyProfile Omega3      GenerallyWellTolerated)
(SafetyProfile Fisetin     GenerallyWellTolerated)
(SafetyProfile NMN         GenerallyWellTolerated)
(SafetyProfile Resveratrol GenerallyWellTolerated)   ; safety is not the issue — efficacy is
(SafetyProfile Berberine   UseWithCaution)           ; GI effects; AMPK/CYP interactions

;; ================================================================
;; 5. Interactions  (the first facts for the declared-but-empty predicate)
;; ================================================================
;; Berberine and Metformin share the AMPK-activation mechanism (mechanistic_bridges.metta
;; notes the parallel), so co-administration is additive glucose lowering — worth a
;; flag. This is the pipeline.md §4.6 "Interactions (Berberine Metformin …)" fact.
(Interaction Berberine Metformin
   "Shared AMPK activation — additive glucose lowering; monitor and consult MD")

;; ================================================================
;; 6. Patient medication context (for the interaction demo)
;; ================================================================
;; Patient002 is the metabolic-dominant profile (patient_profile.metta): elevated
;; fasting glucose / HbA1c, and clinically the one who would be on metformin. So the
;; metabolically-relevant Berberine recommendation for Patient002 collides with their
;; metformin -> the interaction flag fires for Patient002 and NOT for the senescence-
;; dominant, med-free Patient001. (Patient001 intentionally has no CurrentMedication.)
(CurrentMedication Patient002 Metformin)

;; ================================================================
;; 7. Smoke-test queries (verified, hyperon 0.2.10 — see the test module)
;; ================================================================
;; !(match &self (EvidenceLevel Resveratrol $c) $c)   ; ITP_Negative
;; !(evidence-confidence InVitro)                     ; 0.35 (Omega3's tier is 0.85)
;; !(match &self (Interaction $a $b $n) ($a $b))      ; (Berberine Metformin)
