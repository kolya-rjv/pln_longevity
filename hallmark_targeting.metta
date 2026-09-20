;; hallmark_targeting.metta
;; Which intervention TARGETS which hallmark of aging — the relation the
;; ontology has declared since day one and never populated.               [v1]
;;
;; WHY THIS FILE EXISTS
;;   The 2026-09-18 API evaluation asked five variants of one question ("which
;;   hallmarks does rapamycin target?", "which interventions target
;;   mitochondrial dysfunction?", Q3 / Q3b / Q3c / Q13 / Q22). All five came
;;   back EMPTY. Two separate causes:
;;     1. the translator reached for `TargetsHallmark`, which
;;        logical_predicates.metta:58 DECLARES and nothing populated — a raw
;;        `(match &self (TargetsHallmark $i $h) $h)` confirmed zero facts;
;;     2. `Rapamycin`, the single most-asked-about geroprotector, did not exist
;;        as an atom anywhere in the runtime KB. It appeared only inside
;;        comments, so the validator rejected every query that named it.
;;   The relation itself was never missing: the 13 curated interventions reach
;;   their hallmarks through the review-level `HallmarkInterventionEvidence`
;;   records in hallmarks_lopezotin2023_intervention_evidence.metta. What was
;;   missing was ONE form that covers all of them, plus the two headline drugs.
;;   This file supplies both, so a single
;;       (match &self (TargetsHallmark $i $h) …)
;;   answers the question in either direction.
;;
;; HONESTY CONTRACT — read this before adding anything here
;;   * TargetsHallmark carries NO TRUTH VALUE, by design. It is the same kind of
;;     source-backed conceptual claim as `HallmarkComponent` in
;;     hallmarks_lopezotin2023_anchors.metta: "this intervention is studied as
;;     acting on this hallmark". There is no single effect size behind it and no
;;     evidence tier that would make one up honestly, so it gets neither. Every
;;     calibrated number in this repo is either a reproducible function of a
;;     reported effect size (evidence_calibration.metta, drugage_calibration.metta)
;;     or an explicitly labelled curated magnitude prior whose confidence is an
;;     `(evidence-confidence <Tier>)` lookup (mechanistic_bridges.metta). A
;;     targeting claim is neither. Inventing an stv here would be the exact
;;     failure mode the whole calibration layer exists to prevent.
;;   * PROVENANCE INSTEAD. Every back-filled line names the evidence record it
;;     is derived from; every newly added line names a verified publication,
;;     recorded below with its PubMed ID and DOI.
;;
;; *** THE LIMIT, STATED UP FRONT ***
;;   TargetsHallmark is a TARGETING claim, NOT a causal `Effect` edge. It does
;;   NOT make rapamycin reachable by `infer` (pln_deduction.metta), and IT MUST
;;   NOT. This file deliberately adds ZERO `Effect` edges.
;;
;;   The temptation is obvious: rapamycin is the best-evidenced mammalian
;;   geroprotector in the KB's subject matter, the nutrient-sensing axis in
;;   mechanistic_bridges.metta already runs
;;       DeregulatedNutrientSensing -> InsulinResistance -> FastingGlucose -> CHD,
;;   and one `(Effect Rapamycin InsulinResistance Neg …)` would make rapamycin
;;   rank alongside metformin and berberine tomorrow. That edge would be WRONG,
;;   and wrong in its SIGN, which is the one error a signed calculus cannot
;;   absorb: chronic rapamycin CAUSES glucose intolerance in mice — Weiss et al.
;;   2018 (PMID 29579736, doi 10.18632/aging.101401) is titled "Metformin
;;   reduces glucose intolerance caused by rapamycin treatment in genetically
;;   heterogeneous female mice", i.e. the glucose axis is the direction in which
;;   rapamycin does HARM while extending lifespan through another route.
;;   A language model asked "does rapamycin help a metabolic patient?" will
;;   happily chain the plausible-sounding path. PLN's job is to REFUSE to,
;;   until a calibrated edge with the right sign exists. Until then the honest
;;   answer to "rank rapamycin against D+Q for coronary risk" is that rapamycin
;;   is not connected to that outcome in this KB — which is what `infer` says.
;;   This restraint is the point of the whole exercise; do not quietly undo it.
;;
;; WHAT THIS FILE DOES NOT CLAIM
;;   Absence here is not evidence of absence. The hallmark layer is a REVIEW
;;   TABLE (López-Otín 2023, Table 1) plus two hand-added headline drugs, not a
;;   census of the literature. GET /interventions says so in its `note`.
;;
;; Load AFTER: system_types, logical_predicates (the TargetsHallmark / Causes /
;;             Inhibits / Activates declarations), hallmarks_core (the twelve
;;             HallmarkOfAging symbols), hallmarks_lopezotin2023_anchors (MTORC1,
;;             AMPK) and hallmarks_lopezotin2023_intervention_evidence (the 13
;;             interventions and the record ids cited below).
;;             Order relative to mechanistic_bridges.metta does not matter: the
;;             only symbol the two share is Metformin, whose `Inheritance` typing
;;             lives there, and nothing in this file evaluates at load time.
;;
;; STATUS: v1, intended for hyperon 0.2.10. Ground facts plus two accessor
;;         rules — small, additive, and safe to load into the runtime space
;;         (30-odd atoms, nowhere near the ~250-row match ceiling that aborts
;;         hyperon). Regression-guarded in tests/test_hallmark_targeting.py.

;; ================================================================
;; 1. Publication provenance for the facts added in §3
;; ================================================================
;; Verified against PubMed (pmid + doi both checked) before being written here;
;; the repo rule is that no PMID is cited unless it was looked up.
(: HarrisonEtAl2009_RapamycinITP Publication)
(PublicationTitle HarrisonEtAl2009_RapamycinITP "Rapamycin fed late in life extends lifespan in genetically heterogeneous mice")
(PublicationYear HarrisonEtAl2009_RapamycinITP 2009)
(JournalName HarrisonEtAl2009_RapamycinITP "Nature")
(DOI HarrisonEtAl2009_RapamycinITP "10.1038/nature08221")
(PubMedID HarrisonEtAl2009_RapamycinITP "19587680")

(: SaxtonSabatini2017_mTOR Publication)
(PublicationTitle SaxtonSabatini2017_mTOR "mTOR Signaling in Growth, Metabolism, and Disease")
(PublicationYear SaxtonSabatini2017_mTOR 2017)
(JournalName SaxtonSabatini2017_mTOR "Cell")
(DOI SaxtonSabatini2017_mTOR "10.1016/j.cell.2017.02.004")
(PubMedID SaxtonSabatini2017_mTOR "28283069")

(: BannisterEtAl2014_Metformin Publication)
(PublicationTitle BannisterEtAl2014_Metformin "Can people with type 2 diabetes live longer than those without? A comparison of mortality in people initiated with metformin or sulphonylurea monotherapy and matched, non-diabetic controls")
(PublicationYear BannisterEtAl2014_Metformin 2014)
(JournalName BannisterEtAl2014_Metformin "Diabetes, Obesity and Metabolism")
(DOI BannisterEtAl2014_Metformin "10.1111/dom.12354")
(PubMedID BannisterEtAl2014_Metformin "25041462")

;; Cited for the caveat in §4, not for a targeting claim.
(: WeissEtAl2018_RapamycinGlucose Publication)
(PublicationTitle WeissEtAl2018_RapamycinGlucose "Metformin reduces glucose intolerance caused by rapamycin treatment in genetically heterogeneous female mice")
(PublicationYear WeissEtAl2018_RapamycinGlucose 2018)
(JournalName WeissEtAl2018_RapamycinGlucose "Aging (Albany NY)")
(DOI WeissEtAl2018_RapamycinGlucose "10.18632/aging.101401")
(PubMedID WeissEtAl2018_RapamycinGlucose "29579736")

;; ================================================================
;; 2. Back-fill: the curated review table, as TargetsHallmark
;; ================================================================
;; One line per (intervention, hallmark) pair already asserted by a
;; HallmarkInterventionEvidence record in
;; hallmarks_lopezotin2023_intervention_evidence.metta. Each names the record it
;; is derived from, so nothing here is a new claim — this is the SAME evidence,
;; projected onto the declared vocabulary so one query form reaches all of it.
;; The records keep the richer provenance (species model, outcome wording,
;; review reference number); these lines are the index, not a replacement.
;; DasatinibPlusQuercetin has two records for one hallmark, hence 14 records
;; and 13 lines.

;; Epigenetic alterations
(TargetsHallmark AlphaKetoglutarate EpigeneticAlterations)   ;; LopezOtin2023_AlphaKetoglutarate_Human
(TargetsHallmark GH_DHEA_Metformin EpigeneticAlterations)    ;; LopezOtin2023_GH_DHEA_Metformin_Human

;; Disabled macroautophagy
(TargetsHallmark ATG5Overexpression DisabledMacroautophagy)  ;; LopezOtin2023_ATG5Overexpression_Mouse
(TargetsHallmark BECN1_F121A DisabledMacroautophagy)         ;; LopezOtin2023_BECN1_F121A_Mouse
(TargetsHallmark Spermidine DisabledMacroautophagy)          ;; LopezOtin2023_Spermidine_Mouse
(TargetsHallmark NMN DisabledMacroautophagy)                 ;; LopezOtin2023_NMN_Human
(TargetsHallmark UrolithinA DisabledMacroautophagy)          ;; LopezOtin2023_UrolithinA_Human

;; Deregulated nutrient-sensing
(TargetsHallmark CaloricRestriction DeregulatedNutrientSensing) ;; LopezOtin2023_CaloricRestriction_Human

;; Mitochondrial dysfunction  (Q13's hallmark: it had no answer at all before)
(TargetsHallmark Elamipretide MitochondrialDysfunction)      ;; LopezOtin2023_Elamipretide_Human

;; Cellular senescence
(TargetsHallmark DasatinibPlusQuercetin CellularSenescence)  ;; LopezOtin2023_DQ_Mouse + LopezOtin2023_DQ_Human_DiabeticKidneyDisease
(TargetsHallmark Fisetin CellularSenescence)                 ;; LopezOtin2023_Fisetin_Mouse

;; Chronic inflammation
(TargetsHallmark Canakinumab ChronicInflammation)            ;; LopezOtin2023_Canakinumab_Human

;; Dysbiosis
(TargetsHallmark AkkermansiaMuciniphila Dysbiosis)           ;; LopezOtin2023_Akkermansia_Human

;; ================================================================
;; 3. The two headline drugs the KB was missing
;; ================================================================
;; These are NOT derived from the López-Otín table — they are new, individually
;; sourced claims, which is why each carries its own publication link above.

;; --- Rapamycin -------------------------------------------------
;; MANDATORY: without this typing the symbol does not exist for the runtime
;; inventory (ontology/inventory.py) and the validator rejects — with a 422 —
;; every query that so much as names rapamycin. That rejection, not a missing
;; relation, is what made Q3 unanswerable.
(Inheritance Rapamycin Pharmaceutical)

;; Rapamycin is an mTOR inhibitor, and MTORC1 is already an anchor of
;; DeregulatedNutrientSensing (hallmarks_lopezotin2023_anchors.metta:73). The
;; mammalian lifespan result is Harrison 2009 (ITP, three independent sites,
;; +14% female / +9% male age at 90% mortality, dosing begun at 600 days);
;; the hallmark assignment — nutrient sensing AND macroautophagy, the two
;; processes mTORC1 gates — is Saxton & Sabatini 2017's review scope.
(TargetsHallmark Rapamycin DeregulatedNutrientSensing)  ;; HarrisonEtAl2009_RapamycinITP, SaxtonSabatini2017_mTOR
(TargetsHallmark Rapamycin DisabledMacroautophagy)      ;; SaxtonSabatini2017_mTOR
(SupportedByPublication Rapamycin HarrisonEtAl2009_RapamycinITP)
(SupportedByPublication Rapamycin SaxtonSabatini2017_mTOR)

;; --- Metformin -------------------------------------------------
;; Metformin already exists in the KB (mechanistic_bridges.metta types it and
;; gives it a calibrated Neg edge onto InsulinResistance) but had no hallmark
;; link, so "which hallmarks does metformin target?" was empty too.
;; Honest reading of the source: Bannister 2014 is an OBSERVATIONAL survival
;; comparison (78,241 metformin-initiated T2D patients vs. 90,463 matched
;; non-diabetic controls; adjusted median survival 15% LONGER on metformin than
;; in the non-diabetic controls). It is the mortality finding that put metformin
;; on the geroprotector list — it is not itself a mechanism paper. The hallmark
;; assignment rests on the mechanism the KB already holds: AMPK is an anchor of
;; DeregulatedNutrientSensing (anchors.metta:74) and §4 records the activation.
(TargetsHallmark Metformin DeregulatedNutrientSensing)  ;; BannisterEtAl2014_Metformin + AMPK anchor
(SupportedByPublication Metformin BannisterEtAl2014_Metformin)

;; ================================================================
;; 4. Mechanism, in the declared-but-unused wrapper vocabulary
;; ================================================================
;; `Causes` + the unary effect wrappers (`Inhibits`, `Activates`) are declared at
;; logical_predicates.metta:19-32 with the worked example
;; `(Causes Rapamycin (Inhibits mTORC1))` in the comment — and were, until now,
;; the only vocabulary in the file illustrated by an example that did not exist.
;; These two lines make that comment true, using the anchor spellings that are
;; actually in the space (MTORC1, AMPK — not the comment's `mTORC1`).
;;
;; They are a MECHANISM RECORD, not a causal edge in the PLN sense: `Causes`
;; has no lifting rule in pln_deduction.metta and no truth value here, for the
;; same reason §"HONESTY CONTRACT" gives. They say what the drug acts on; they
;; do not say what follows for any outcome.
(Causes Rapamycin (Inhibits MTORC1))    ;; SaxtonSabatini2017_mTOR
(Causes Metformin (Activates AMPK))     ;; canonical AMPK mechanism; BannisterEtAl2014_Metformin is the survival evidence, not the mechanism

;; The caveat that forbids the glucose-axis shortcut, recorded as a fact rather
;; than left in a comment, so a caller reading the KB sees it too. `Limitation`
;; is declared at logical_predicates.metta:75 and this is its first instance.
(Limitation Rapamycin "Chronic rapamycin causes glucose intolerance in mice (Weiss 2018, PMID 29579736) while extending lifespan. Its effect on the glucose / insulin-resistance axis is HARMFUL, so rapamycin must not be given an Effect edge through that axis; it has no chain to CoronaryHeartDisease in this KB and `infer` correctly returns nothing for it.")

;; ================================================================
;; 5. Accessors: one call per natural question
;; ================================================================
;; The space is threaded explicitly (never a bare &self) so these work inside
;; the one-space runtime (pln_chat/core/pln_runner.py) and inside a test's own
;; space, exactly like patient-age / supp-evidence-cat in the other layers.
;; Both are non-deterministic: they yield one result per match and NOTHING for
;; an intervention or hallmark with no targeting link — no placeholder, no
;; fabricated "none found" atom.
;;
;; NOTE (hyperon 0.2.10): these deliberately carry NO `(: f (-> Atom Atom Atom))`
;; signature. Declaring one makes the interpreter treat the arguments and the
;; result as unreduced atoms, and the call comes back as the literal
;; `(match ModuleSpace(...) (TargetsHallmark Rapamycin $h) $h)` instead of the
;; hallmarks — measured, not guessed. The other layers' accessors
;; (patient-age, supp-evidence-cat) are undeclared for the same reason.

;; (hallmarks-of <space> <intervention>) -> each hallmark it targets.
(= (hallmarks-of $space $intv)
   (match $space (TargetsHallmark $intv $h) $h))

;; (interventions-for <space> <hallmark>) -> each intervention targeting it.
(= (interventions-for $space $hallmark)
   (match $space (TargetsHallmark $i $hallmark) $i))

;; ================================================================
;; 6. Smoke-test queries (asserted in tests/test_hallmark_targeting.py)
;; ================================================================
;; !(match &self (TargetsHallmark Rapamycin $h) $h)
;;   => [DeregulatedNutrientSensing, DisabledMacroautophagy]
;;
;; !(match &self (TargetsHallmark $i MitochondrialDysfunction) $i)
;;   => [Elamipretide]
;;
;; !(hallmarks-of &self Rapamycin)
;;   => [DeregulatedNutrientSensing, DisabledMacroautophagy]
;;
;; !(interventions-for &self CellularSenescence)
;;   => [DasatinibPlusQuercetin, Fisetin]
;;
;; And the restraint, still true after this file loads:
;; !(infer &self Rapamycin CoronaryHeartDisease)   => no result.
