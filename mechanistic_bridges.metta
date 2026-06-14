;; mechanistic_bridges.metta
;; Curated mechanistic bridge priors for the inference layer.    [v1]
;;
;; These are the connective tissue that lets PLN deduction chain ACROSS sources
;; (pln_deduction.metta). Each is a single SIGNED truth-valued directed link
;;
;;     (Effect <from> <to> <sign> (stv <strength> <confidence>))
;;
;;   sign ∈ { Pos, Neg }: Pos = <from> raises <to>, Neg = <from> reduces <to>.
;;
;; HONESTY CONTRACT — how these differ from calibrated empirical links:
;;   * confidence  is DERIVED, not invented: it is a lookup into the single
;;                 evidence-confidence authority (epistemic_calibration.metta),
;;                 by the strongest evidence tier behind the mechanism.
;;   * strength    is an explicit CURATED PRIOR (expert estimate): the effect
;;                 MAGNITUDE. Unlike a calibrated empirical link (whose strength
;;                 is a reproducible function of a reported effect size), a
;;                 mechanistic step has no single effect size, so strength is a
;;                 documented judgment to refine in the tuning loop
;;                 (calibration_layer.md §8).
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
;;             grim_age_core (DNAmPAI1), hallmarks_core (CellularSenescence),
;;             hallmarks_lopezotin2023_intervention_evidence
;;               (DasatinibPlusQuercetin, Fisetin, Spermidine).

;; ------------------------------------------------------------
;; Minimal typing for the SASP node introduced here.
;; SASP = senescence-associated secretory phenotype.
;; ------------------------------------------------------------
(: SASP Phenotype)

;; ------------------------------------------------------------
;; Senescence -> SASP -> PAI-1 axis (harmful direction)
;; ------------------------------------------------------------
;; Cellular senescence drives the SASP; the SASP includes elevated PAI-1
;; (SERPINE1), the plasma protein whose DNAm surrogate (DNAmPAI1) is a GrimAge
;; component (grim_age_core.metta). Together with the calibrated Lu 2019 link
;; DNAmPAI1 -> CoronaryHeartDisease, these bridges let PLN derive the novel,
;; never-directly-stated conclusion CellularSenescence -> CoronaryHeartDisease.
;;
;; strength: curated prior (magnitude). confidence: = (evidence-confidence ...)
;; lookup, left unevaluated so it stays a single source of truth; the arithmetic
;; in combine-tv reduces it at chain time.
(Effect CellularSenescence SASP Pos
   (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))

(Effect SASP DNAmPAI1 Pos
   (stv 0.70 (evidence-confidence AnimalStudies_Replicated)))

;; ------------------------------------------------------------
;; Senolytic intervention (protective direction)
;; ------------------------------------------------------------
;; Dasatinib + Quercetin clears senescent cells, i.e. REDUCES the cellular-
;; senescence burden (Neg). Source-grounded by the López-Otín 2023 review
;; records LopezOtin2023_DQ_Mouse ("increased health- and lifespan") and
;; LopezOtin2023_DQ_Human_DiabeticKidneyDisease ("reduction of senescent cells
;; ... and of pro-inflammatory factors in serum").
;;
;; Chaining this Neg hop into the harmful senescence->...->CHD axis flips the net
;; sign to Neg: PLN derives that D+Q is expected to *reduce* CHD risk — the
;; signed, uncertainty-quantified intervention signal of pipeline.md Demo 2.
(Effect DasatinibPlusQuercetin CellularSenescence Neg
   (stv 0.70 (evidence-confidence AnimalStudies_Replicated)))

;; ------------------------------------------------------------
;; More senolytic / autophagy interventions — the candidate pool the
;; intervention-ranking demo (pln_intervention_ranking.metta, pipeline.md
;; Demo 2) sorts over. Each reaches CHD through the SAME senescence axis above,
;; so PLN ranks them by net protective magnitude x confidence — and confidence
;; falls off with chain length, exactly the LLM-incomparable signal.
;; ------------------------------------------------------------

;; Fisetin is a senolytic in the same class as D+Q — it clears senescent cells
;; (Neg), source-grounded by LopezOtin2023_Fisetin_Mouse ("increased health- and
;; lifespan", hallmark CellularSenescence). Evidence is a SINGLE animal record
;; here (vs. D+Q's mouse + human), so it maps to the lower AnimalStudies_Single
;; tier and a slightly smaller magnitude prior — so PLN ranks D+Q above Fisetin
;; on identical-mechanism grounds. This is the calibrated-evidence distinction an
;; LLM blurs.
(Effect Fisetin CellularSenescence Neg
   (stv 0.65 (evidence-confidence AnimalStudies_Single)))

;; Spermidine acts one step further upstream: it INDUCES autophagy (Pos on
;; Autophagy), and autophagy in turn REDUCES the senescent-cell burden (Neg on
;; CellularSenescence). Representing the mechanism honestly as two hops (rather
;; than collapsing it to a direct senolytic claim) means Spermidine reaches CHD
;; through a LONGER chain than the direct senolytics, so its conclusion carries
;; appropriately lower confidence — it ranks below D+Q and Fisetin. Grounded by
;; LopezOtin2023_Spermidine_Mouse ("extended longevity, reduced cardiac aging and
;; oxidative stress", hallmark DisabledMacroautophagy) and the canonical
;; autophagy-induction / senescence-clearance mechanism (pipeline.md A.6).
(: Autophagy Pathway)

(Effect Spermidine Autophagy Pos
   (stv 0.80 (evidence-confidence AnimalStudies_Single)))

(Effect Autophagy CellularSenescence Neg
   (stv 0.60 (evidence-confidence AnimalStudies_Replicated)))

;; ------------------------------------------------------------
;; Cause -> observable-biomarker links (the abductive-diagnosis graph).
;; ------------------------------------------------------------
;; These connect upstream hallmark CAUSES to the GrimAge components and blood
;; markers a patient actually presents with, so abductive diagnosis
;; (pln_abductive_diagnosis.metta, pipeline.md Demo 1) can run the graph
;; BACKWARD: given observed-elevated markers, rank the upstream causes that
;; would raise them. All Pos (the cause, when elevated, RAISES the marker).
;;
;; Grounded in the GrimAge component interpretations (pipeline.md App. A.2: what
;; each DNAm surrogate reflects) and the senescence / inflammation / mito
;; cascades (App. A.6). strength = curated magnitude prior; confidence = the
;; evidence-confidence lookup, as for every bridge above.

;; CRP (C-reactive protein): the canonical circulating inflammation marker.
(Inheritance CRP Biomarker)

;; Senescence drives chronic inflammation via the SASP (App. A.6:
;; SASP -> ChronicInflammation). This is what lets CellularSenescence explain
;; inflammatory findings, not just DNAmPAI1.
(Effect SASP ChronicInflammation Pos
   (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))

;; Chronic inflammation elevates CRP, and feeds the GDF15 surrogate (App. A.2:
;; "DNAmGDF15 reflects MitochondrialStress, Inflammation"). The GDF15 link is the
;; weaker, secondary contributor (mitochondrial dysfunction below is primary).
(Effect ChronicInflammation CRP Pos
   (stv 0.80 (evidence-confidence AnimalStudies_Replicated)))
(Effect ChronicInflammation DNAmGDF15 Pos
   (stv 0.65 (evidence-confidence AnimalStudies_Replicated)))

;; Mitochondrial dysfunction raises GDF15 (App. A.6: MitochondrialDysfunction ->
;; Increases GDF15), read out by the DNAmGDF15 GrimAge surrogate. This is the
;; primary, direct explanation for an isolated DNAmGDF15 elevation.
(Effect MitochondrialDysfunction DNAmGDF15 Pos
   (stv 0.75 (evidence-confidence AnimalStudies_Replicated)))
