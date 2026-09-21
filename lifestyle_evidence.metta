;; lifestyle_evidence.metta
;; The smoking lever: an exposure the KB already measures, finally wired.   [v1]
;;
;; THE FAILURE THIS FIXES. The 2026-09-18 evaluation asked, for a patient whose
;; profile says FormerSmoker, "what would his risk look like if he had never
;; smoked?" and got nothing back. That was the correct answer at the time and a
;; bad one: `PatientSmoking` reached ZERO inference paths. Grep it — before this
;; file it appeared in exactly four places: its type declaration in
;; patient_profile.metta, two patient facts, one documentation line, and a regex
;; in pln_chat/api.py that populates GET /patients. Nothing consumed it. The
;; counterfactual engine had a do-operator, the risk model had a projection, and
;; the one exposure every clinician asks about could not be pulled.
;;
;; WHY THE ROUTE IS THE CLOCK, NOT A SMOKING -> CHD EDGE. The obvious repair is
;; an (Effect Smoking CoronaryHeartDisease Pos …) link. We do not write one, for
;; the same reason pln_counterfactual.metta refuses a CRP -> GrimAge edge: the
;; risk model reads ONE predictor, the composite clock AgeAccelGrim, and a
;; parallel smoking -> CHD term would double-count an exposure the clock already
;; carries. GrimAge carries it explicitly: DNAmPACKYRS is a GrimAge component
;; ((PartOf DNAmPACKYRS GrimAge), grim_age_core.metta:85), it is the DNAm
;; surrogate FOR pack-years ((SurrogateFor DNAmPACKYRS SmokingPackYears), :113),
;; and Lu et al. built it from 172 CpGs. So the honest wiring is the one the
;; clock's own construction implies:
;;
;;     SmokingCessation --Neg--> SmokingPackYears --Pos--> DNAmPACKYRS  (PartOf GrimAge)
;;
;; and the existing machinery does the rest: `resolve-lever` routes the lever to
;; the harmful driver, `cf-parts` iterates (PartOf $m GrimAge) and finds
;; DNAmPACKYRS, and `project-risk` turns the clock reduction into an absolute
;; risk reduction. Not one line of the counterfactual or risk layer changed.
;;
;; HONESTY CONTRACT — identical to mechanistic_bridges.metta, and for the same
;; reason:
;;   * confidence  is DERIVED, never invented: a lookup into the single
;;                 evidence-confidence authority (epistemic_calibration.metta),
;;                 written UNEVALUATED so that file stays the one source of truth.
;;   * strength    is an explicit CURATED PRIOR (§2), quarantined in one numbered
;;                 section, each with the reported quantity it is anchored to and
;;                 the reason it is not larger.
;;   * Patient003  is SYNTHETIC. Every value in §4 is made up by us to exercise a
;;                 code path, exactly as Patient001 and Patient002 are, and is
;;                 labelled as such value by value. It is not a person.
;;
;; WHAT THIS LAYER DELIBERATELY DOES NOT CLAIM. The smoking axis terminates at a
;; clock component. It does NOT reach CoronaryHeartDisease through `infer`, so
;; (rank-interventions-for-patient … CoronaryHeartDisease) still omits
;; SmokingCessation — there is no curated DNAmPACKYRS -> CHD evidence record, and
;; we will not manufacture one to make a ranking look complete. The CHD
;; consequence of quitting is available, and only available, through the risk
;; model's clock: (project-risk-patient &self Patient003 SmokingCessation).
;;
;; Load AFTER the patient + counterfactual + risk stack (it adds Effect edges the
;; deduction layer reads and a patient the counterfactual layer consumes):
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile pln_counterfactual
;;   pln_risk_prediction
;;   lifestyle_evidence                  (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_smoking_lever.py. Pure DATA — it defines no function, so it adds no
;; runtime behaviour beyond the atoms themselves. Every new Effect edge either
;; starts at a NEW node (SmokingCessation) or at a node with no other outgoing
;; edge (SmokingPackYears), and the one new incoming edge lands on DNAmPACKYRS,
;; which no pre-existing patient measures — so Patient001's and Patient002's
;; answers are byte-identical before and after (asserted in the test module).

;; ================================================================
;; 1. Typing and publication provenance
;; ================================================================
;; CaloricRestriction is already (Inheritance CaloricRestriction Lifestyle) in
;; hallmarks_lopezotin2023_intervention_evidence.metta; SmokingCessation follows
;; it, so it inherits Intervention through system_types.metta and every
;; intervention-shaped accessor accepts it without a special case.
(Inheritance SmokingCessation Lifestyle)

;; SmokingPackYears is NOT new — grim_age_core.metta:110 already types it
;; (Inheritance SmokingPackYears ExposureBiomarker) as the biological quantity
;; DNAmPACKYRS is a surrogate for. This file only gives it edges.

(: JoehanesEtAl2016_SmokingEWAS Publication)
(PublicationTitle JoehanesEtAl2016_SmokingEWAS "Epigenetic Signatures of Cigarette Smoking")
(PublicationYear JoehanesEtAl2016_SmokingEWAS 2016)
(JournalName JoehanesEtAl2016_SmokingEWAS "Circulation: Cardiovascular Genetics")
(DOI JoehanesEtAl2016_SmokingEWAS "10.1161/CIRCGENETICS.116.001506")
(PubMedID JoehanesEtAl2016_SmokingEWAS "27651444")

(: DuncanEtAl2019_CessationCVD Publication)
(PublicationTitle DuncanEtAl2019_CessationCVD "Association of Smoking Cessation With Subsequent Risk of Cardiovascular Disease")
(PublicationYear DuncanEtAl2019_CessationCVD 2019)
(JournalName DuncanEtAl2019_CessationCVD "JAMA")
(DOI DuncanEtAl2019_CessationCVD "10.1001/jama.2019.10298")
(PubMedID DuncanEtAl2019_CessationCVD "31429895")

;; ================================================================
;; 2. KNOBS: the two curated strength priors  (the ONLY hand-set numbers here)
;; ================================================================
;; Both are magnitudes, and both are quarantined in this one section so a tuning
;; pass has a single place to look (the mechanistic_bridges.metta discipline).
;; Neither confidence is hand-set: each is an (evidence-confidence <Tier>) lookup
;; left unevaluated, so epistemic_calibration.metta remains the one authority.

;; (a) smoking exposure -> the DNAm pack-years surrogate.
;;
;; CURATED PRIOR 0.80 (magnitude). Anchored on Joehanes 2016 (PMID 27651444), a
;; meta-analysis of genome-wide methylation in 15,907 blood samples across 16
;; cohorts (2,433 current, 6,518 former, 6,956 never smokers): 2,623 CpGs are
;; differentially methylated between current and never smokers at a Bonferroni
;; threshold. Blood methylation is about as strong and as reproducible a readout
;; of tobacco exposure as exists, which is why the prior is high.
;;
;; It is NOT 1.0, and the reason matters: DNAmPACKYRS is an elastic-net PREDICTOR
;; of pack-years, not pack-years itself. Lu et al. fit 172 CpGs to a
;; self-reported exposure, so the surrogate tracks the exposure with real
;; measurement error in both directions.
;;
;; TIER: Epidemiological (0.60).
;;
;; This was MultipleHumanTrials (0.85) on the argument that a 16-cohort
;; meta-analysis is "stronger than a single human study". That argument is about
;; SAMPLE SIZE, and this ladder is not a size ordering — it is a DESIGN ordering.
;; `Epidemiological` (0.60) sits below `SingleHumanTrial` (0.70) precisely
;; because observational evidence is not trial evidence however many cohorts it
;; pools. Joehanes 2016 is typed "Meta-Analysis; Multicenter Study" in PubMed,
;; not a trial: it compares current, former and never smokers in 15,907 blood
;; samples with no intervention and no randomisation.
;;
;; The tell was internal. Thirty lines below, Duncan 2019 — also a prospective
;; observational human cohort — is tiered Epidemiological, with the note "it is
;; not a randomized trial of quitting, and no such trial exists". Two
;; observational studies in one file cannot sit 0.25 apart on a design ladder,
;; and promoting one of them onto a trials rung to reflect its size is the exact
;; move the calibration layer exists to prevent.
;;
;; The EvidenceCategory enumeration still has no meta-analysis tier, and adding
;; one to hold a single edge would be inventing a number. `MetaAnalysis_Human`
;; stays an OPEN QUESTION (docs/calibration_layer.md #4); until it exists, the
;; honest rung is the one that names this design.
(Effect SmokingPackYears DNAmPACKYRS Pos
   (stv 0.80 (evidence-confidence Epidemiological)))
(SupportedByPublication SmokingPackYears JoehanesEtAl2016_SmokingEWAS)

;; (b) cessation -> the exposure burden (protective, Neg).
;;
;; CURATED PRIOR 0.39 (magnitude), anchored on a reported effect size rather than
;; picked: Duncan 2019 (PMID 31429895) followed 8,770 Framingham participants and
;; reports, among heavy ever smokers, a hazard ratio of 0.61 (95% CI 0.49-0.76)
;; for incident cardiovascular disease when quitting within 5 years versus
;; continuing to smoke. 1 - 0.61 = 0.39.
;;
;; The anchor is an ANCHOR, not a derivation, and we say so: that HR is for
;; incident CVD, whereas this edge reduces an exposure burden read out in
;; methylation. We borrow its magnitude because it is the best-quantified
;; statement of "how much of the harm quitting takes back", and we label the
;; result a curated prior accordingly.
;;
;; WHY IT IS THIS FAR BELOW 1.0 — the caveat is the whole point. Quitting does
;; not return a former smoker to a never-smoker. Duncan reports that risk
;; relative to never smokers stays significantly elevated BEYOND 5 years and only
;; ceases to be significant somewhere between 10 and 15 years after cessation.
;; Joehanes shows the same thing in the methylation itself: comparing former with
;; never smokers, 185 of the current-vs-never CpGs remain significant at
;; Bonferroni — "a pattern of persistent altered methylation, with attenuation,
;; after smoking cessation". A residual signal is exactly what a DNAm pack-years
;; surrogate should keep, and a strength of 1.0 here would assert the opposite.
;;
;; TIER: Epidemiological (0.60). Framingham is a prospective observational cohort
;; with time-updated exposure; it is not a randomized trial of quitting, and no
;; such trial exists.
(Effect SmokingCessation SmokingPackYears Neg
   (stv 0.39 (evidence-confidence Epidemiological)))
(SupportedByPublication SmokingCessation DuncanEtAl2019_CessationCVD)

;; THE PRECONDITION. Quitting is worth nothing to someone who does not smoke,
;; and the counterfactual keyed only on the DNAmPACKYRS z — so a never-smoker
;; with an elevated surrogate got a smoker's answer, and `PatientSmoking`
;; reached no inference path despite being what this layer exists to wire up.
;; Declared as a fact so the gate (pln_counterfactual.metta §3b) is generic.
;;
;; CurrentSmoker, not "ever smoked": Duncan's HR 0.61 is for quitting versus
;; CONTINUING. A former smoker has already taken that step; what remains for
;; them is the residual the Limitation below describes, and no curated evidence
;; here quantifies a second cessation for someone who has already quit.
(LeverRequiresSmoking SmokingCessation CurrentSmoker)

;; The caveat above, as a FACT rather than only a comment, so a caller reading
;; the KB over HTTP sees it too (the hallmark_targeting.metta `Limitation`
;; discipline).
(Limitation SmokingCessation "Quitting does not restore a never-smoker's risk. Duncan 2019 (PMID 31429895) reports former smokers' cardiovascular risk remains significantly elevated relative to never smokers beyond 5 years after cessation, ceasing to be significant only between 10 and 15 years; Joehanes 2016 (PMID 27651444) finds 185 smoking-associated CpGs still differentially methylated in former versus never smokers. The curated strength of 0.39 encodes that residual: this lever attenuates the exposure burden, it does not erase it.")

;; ================================================================
;; 3. Type declaration for the third example patient
;; ================================================================
;; Nothing new — Patient003 uses PatientProfile / PatientAge / PatientSex /
;; PatientSmoking / MeasuredZ exactly as declared in patient_profile.metta.

;; ================================================================
;; 4. Third example patient — smoking-dominant  (SYNTHETIC)
;; ================================================================
;; Patient001 is senescence-dominant, Patient002 is metabolic-dominant. Patient003
;; is the third axis: an elevated GrimAge carried almost entirely by its
;; smoking-exposure component, with senescence, inflammation and metabolic markers
;; all unremarkable. That contrast is what makes the lever legible — the same
;; query that returns ~0 for InsulinResistance on Patient001 returns a real number
;; for SmokingCessation here, and vice versa.
;;
;; He is a NEW patient rather than an edit to Patient002, deliberately. Patient002
;; is already a FormerSmoker and it is tempting to just give him a DNAmPACKYRS
;; value — but tests/test_patient_grounding.py:223 pins his observation set
;; exactly, the evaluation report quotes his numbers, and changing a published
;; patient to make a new feature demo well is how a knowledge base stops being
;; trustworthy. Patient003 also lives HERE, not in patient_profile.metta, so every
;; existing test module's KB_FILES list is untouched.
;;
;; EVERY VALUE BELOW IS SYNTHETIC — chosen by us to exercise the code path, in the
;; same spirit as Patient001's case-study profile. None of it is measured data.
(InstanceOf Patient003 PatientProfile)
(PatientAge     Patient003 64)          ; synthetic: puts him in the 60-69 male baseline band
(PatientSex     Patient003 Male)        ; synthetic
;; CurrentSmoker, and the reason is the lever itself: SmokingCessation now
;; requires (LeverRequiresSmoking … CurrentSmoker), so a former smoker correctly
;; gets zero from it. He was written FormerSmoker, which made the file's own
;; flagship demo a case the lever does not apply to — it returned a full benefit
;; only because the gate did not exist yet. A current smoker is who the
;; cessation lever is for.
(PatientSmoking Patient003 CurrentSmoker); synthetic: ~30 pack-years, still smoking

(MeasuredZ Patient003 HorvathAgeAccel  0.1)   ; synthetic: first-gen clock ~ normal
(MeasuredZ Patient003 AgeAccelGrim     1.6)   ; synthetic: GrimAge elevated (the discordance again)
(MeasuredZ Patient003 DNAmPACKYRS      2.1)   ; synthetic: the smoking signature — strongly elevated
(MeasuredZ Patient003 DNAmPAI1         0.2)   ; synthetic: senescence markers NORMAL
(MeasuredZ Patient003 DNAmGDF15        0.3)   ; synthetic
(MeasuredZ Patient003 CRP              0.5)   ; synthetic: inflammation NORMAL
(MeasuredZ Patient003 FastingGlucose   0.4)   ; synthetic: metabolic axis NORMAL

;; ================================================================
;; 5. Smoke-test queries (ACTUAL measured values; live floats carry trailing
;;    noise — tests/test_smoking_lever.py compares with tolerance)
;; ================================================================
;; Load order: the §"Load AFTER" list above, then this file.
;;
;; The lever, named as the INTERVENTION, and the driver it reduces. These are two
;; different questions and they no longer give the same answer:
;;
;; !(counterfactual-patient &self Patient003 SmokingCessation)
;;   => (Counterfactual SmokingCessation AgeAccelGrim (expected-delta -0.08190000000000001)
;;         (signed Neg (stv 0.08190000000000001 0.324)) (Via (DNAmPACKYRS)))
;;   "if he quits": 0.39 (the cessation prior) x 0.125 (grimage-weight) x 0.80
;;   (the §2a prior) x 2.1 (his z) = 0.0819, at 0.60 x 0.60 x 0.9 = 0.324 — the
;;   two Epidemiological hops with the chain discount. It agrees, exactly, with
;;   !(infer &self SmokingCessation DNAmPACKYRS) => (signed Neg (stv 0.312 0.324)).
;;
;; !(counterfactual-patient &self Patient003 SmokingPackYears)
;;   => (Counterfactual SmokingPackYears AgeAccelGrim (expected-delta -0.21000000000000002)
;;         (signed Neg (stv 0.21000000000000002 0.6)) (Via (DNAmPACKYRS)))
;;   "if the exposure burden were gone": the do-operator on the driver itself,
;;   one hop, no cessation prior. 0.125 x 0.80 x 2.1 = 0.21 at 0.60.
;;
;;   The 2.56x gap between them IS the 0.39 prior, and until pln_counterfactual
;;   §3b composed the lever edge it was discarded: both spellings returned -0.21
;;   at 0.85, i.e. the engine asserted that quitting erases the exposure burden
;;   entirely — the exact claim the Limitation fact below says it does not make.
;;
;; The same reduction as an ABSOLUTE 10-year CHD risk, via the clock:
;; !(project-risk-patient &self Patient003 SmokingCessation)
;;   => (ProjectedRisk SmokingCessation CoronaryHeartDisease
;;         (point 0.23091003826731654) (reduction 0.005437043915651585)
;;         (delta-clock -0.08190000000000001) (confidence 0.324) (Via (DNAmPACKYRS)))
;;   i.e. 23.63% -> 23.09% absolute, a 0.54-percentage-point reduction (it read
;;   1.37 points at 0.85 confidence before the two corrections above).
;;
;; And the precondition. Patient001 is a NeverSmoker, so the lever is refused
;; rather than priced:
;; !(lever-applicable &self Patient001 SmokingCessation)   => False
;; !(lever-applicable &self Patient003 SmokingCessation)   => True
;;
;; The decomposition credits the component and names NO hallmark cause — honestly.
;; `component-cause` searches (Inheritance $h HallmarkOfAging), and smoking is an
;; EXPOSURE, not a hallmark. An empty (DrivenBy ()) is the right answer; inventing
;; a hallmark to fill it is the failure this KB exists to avoid:
;; !(decompose-grimage &self Patient003)
;;   => (Decomposition Patient003
;;         (Components ((Component DNAmPACKYRS (z 2.1) (weight 0.125)
;;                         (contribution 0.2625) (DrivenBy ()))))
;;         (attributed 0.2625) (residual 1.3375000000000001))
;;
;; And the contrast that makes the profile legible — the senescence and metabolic
;; levers reach nothing Patient003 presents elevated:
;; !(counterfactual-patient &self Patient003 CellularSenescence)
;;   => (Counterfactual CellularSenescence AgeAccelGrim (expected-delta 0.0)
;;         (signed Neg (stv 0.0 0.0)) (Via ()))
;;
;; Patient001 is a never-smoker AND has no DNAmPACKYRS measurement, so the lever
;; returns the honest zero for two independent reasons — the precondition gate
;; and the absent measurement:
;; !(counterfactual-patient &self Patient001 SmokingCessation)
;;   => (Counterfactual SmokingCessation AgeAccelGrim (expected-delta 0.0)
;;         (signed Neg (stv 0.0 0.0)) (Via ()))
