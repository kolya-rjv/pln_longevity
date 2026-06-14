;; mechanistic_bridges.metta
;; Curated mechanistic bridge priors for the inference layer.    [v0]
;;
;; These are the connective tissue that lets PLN deduction chain ACROSS sources
;; (pln_deduction.metta). Each is a single truth-valued directed link
;;
;;     (Implication <from> <to> (stv <strength> <confidence>))
;;
;; HONESTY CONTRACT — how these differ from calibrated empirical links:
;;   * confidence  is DERIVED, not invented: it is a lookup into the single
;;                 evidence-confidence authority (epistemic_calibration.metta),
;;                 by the strongest evidence tier behind the mechanism.
;;   * strength    is an explicit CURATED PRIOR (expert estimate). Unlike a
;;                 calibrated empirical link (whose strength is a reproducible
;;                 function of a reported effect size), a mechanistic step has
;;                 no single effect size, so strength is a documented judgment
;;                 to be refined in the tuning loop (calibration_layer.md §8).
;;     This is the one place strengths are hand-set; they are quarantined here,
;;     clearly labelled, and never mixed into the raw evidence files.
;;
;; NOTE on evidence tiers: the EvidenceCategory enumeration is intervention-
;; -evidence oriented (RCT / trials / animal). Well-replicated *mechanistic*
;; consensus has no perfect tier yet, so we map conservatively to
;; AnimalStudies_Replicated (0.65). A dedicated MechanisticConsensus tier is a
;; natural follow-up (calibration_layer.md open-question #4).
;;
;; Load AFTER: system_types, logical_predicates, epistemic_calibration,
;;             grim_age_core (defines DNAmPAI1).

;; ------------------------------------------------------------
;; Minimal typing for the SASP node introduced here.
;; SASP = senescence-associated secretory phenotype.
;; ------------------------------------------------------------
(: SASP Phenotype)

;; ------------------------------------------------------------
;; Senescence -> SASP -> PAI-1 axis
;; ------------------------------------------------------------
;; Cellular senescence drives the SASP; the SASP includes elevated PAI-1
;; (SERPINE1), the plasma protein whose DNAm surrogate (DNAmPAI1) is a GrimAge
;; component (grim_age_core.metta). Together with the calibrated Lu 2019 link
;; DNAmPAI1 -> CoronaryHeartDisease, these bridges let PLN derive the novel,
;; never-directly-stated conclusion CellularSenescence -> CoronaryHeartDisease.
;;
;; strength: curated prior. confidence: = (evidence-confidence ...) lookup,
;; left unevaluated so it stays a single source of truth; the arithmetic in
;; combine-tv reduces it at chain time.
(Implication CellularSenescence SASP
   (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))

(Implication SASP DNAmPAI1
   (stv 0.70 (evidence-confidence AnimalStudies_Replicated)))
