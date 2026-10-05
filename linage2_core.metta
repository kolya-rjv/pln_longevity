;; linage2_core.metta
;; LinAge2 — a CLINICAL (blood-panel) mortality clock — ontology and composition. [v1]
;;
;; Source: Fong, Denisov, Nefedova, Kennedy & Gruber (2025), "LinAge2: providing
;; actionable insights and benchmarking with epigenetic clocks", npj Aging 11:29.
;; DOI 10.1038/s41514-025-00221-4, PMID 40268972, PMC12019333.
;; Implementation the feature codes below are read from: Rejuve/LinAge2-Python
;; (commit d0b9973), whose Flask service `/predict` returns a biological age, the
;; BA-CA delta, and ONE years-contribution per model feature.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; WHAT THIS LAYER IS, AND WHAT IT IS NOT
;; ─────────────────────────────────────────────────────────────────────────────
;; GrimAge (grim_age_core.metta) is the KB's one clock so far, and it is a DNA-
;; methylation clock almost no app user has. LinAge2 is a mortality clock built
;; from 54 routine lab measurements plus five questionnaire-derived scores (59
;; model inputs), trained on NHANES 1999-2000 and tested on 2001-2002, that the
;; paper reports to predict 10- and 20-year mortality on a par with GrimAge2 and
;; better than PhenoAge DNAm, DunedinPoAm, HorvathAge and HannumAge. It is the
;; clock a user with a blood panel actually has.
;;
;; This file is the STRUCTURAL half: the clock, its residual, its 59 inputs (each
;; with its NHANES variable code, so a LinAge2 service response can be mapped to
;; KB symbols without guessing), and — for the four inputs the KB already reasons
;; about — which KB biomarker each one reads out. It ships NO patient numbers and
;; NO per-feature weights. The per-feature YEARS a patient receives arrive at
;; query time from the LinAge2 service, as caller-supplied atoms
;; (pln_chat/core/linage2_builder.py); the clock's hazard record is in
;; linage2_fong2025_evidence.metta; the rules are in pln_linage2.metta.
;;
;; WHY THE WEIGHTS ARE NOT HERE, AND WHY A CONTRIBUTION'S SIGN IS NOT A LAB'S
;; DIRECTION. LinAge2 is linear in its standardized inputs, so each feature's
;; years-contribution is z x w. But w is a PROJECTION (SVD loadings x Cox
;; coefficients), sex-specific, and NOT the epidemiological direction of the lab:
;; read off the published artifacts, CRP carries +4.4 months/SD in the female
;; model and -0.2 in the male model; serum glucose +8.6 (F) and -3.7 (M). So a
;; positive years-contribution from "CRP" does NOT tell you the CRP was high, and
;; the rules in pln_linage2.metta refuse to infer a cause from a contribution's
;; sign alone: a cause is credited only when the patient's OWN standardized value
;; for that biomarker (patient-z, patient_profile.metta) says it is elevated — or,
;; for the smoking feature, when the patient is recorded as a current smoker. A
;; contribution with no such witness is reported as years, and left unexplained.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; LOAD ORDER — QUERY-SCOPED, NOT IN THE SHARED SPACE
;; ─────────────────────────────────────────────────────────────────────────────
;; This layer does NOT join the chat app's shared execution space
;; (_INFERENCE_STACK). That space is saturated on distinct head symbols — 201,
;; with no margin (docs/nhanes_integration.md §8) — and this file introduces one
;; new one (MeasuresBiomarker) and pln_linage2's patient atoms two more
;; (LinAgeDelta, LinAgeContribution). Measured before deciding: 60 typing facts
;; for these inputs appended to the shared space abort hyperon 0.2.10 on an
;; unrelated decompose-grimage. So LinAge2 questions run in their own
;; query-scoped space, core.pln_runner.LINAGE2_PATIENT_STACK, routed by
;; pln_chat/core/linage2_router.py — the pattern DRUGAGE_STACK, CELLAGE_STACK and
;; NHANES_PATIENT_STACK already use.
;;
;; THE SCOPED SPACE IS TIGHT TOO, and this file was trimmed to fit it. A first
;; version carried each input's NHANES code as a fact, (NHANESCode Hemoglobin
;; "LBXHGB"); with those 59 string atoms loaded, every LinAge2 query aborted, and
;; with them removed all of them ran. The codes are therefore recorded as
;; STRUCTURED COMMENTS on the ModelInput lines (§4) — `;; CODE — description` —
;; which pln_chat/core/linage2_builder.py parses and GET /linage2/features
;; publishes, so the ontology stays the single source of truth for what a caller
;; may send while costing the engine nothing. Likewise the stack underneath is
;; the MINIMAL set a LinAge2 form reads (core/pln_runner.py explains each entry):
;; with the whole NHANES patient stack underneath, the margin was zero.
;;
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core mechanistic_bridges pln_deduction
;;   patient_profile pln_counterfactual pln_risk_prediction nhanes_baseline
;;   lifestyle_evidence                  (SmokingCessation, SmokingPackYears)
;;   linage2_core                        (this file)
;;   linage2_fong2025_evidence
;;   pln_linage2
;;   [build/nhanes_mortality_baseline.metta]   (the generated all-cause baseline, if present)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_linage2.py. Pure DATA — no rule is defined here. Symbols are plain
;; CamelCase because MeTTa tokenizes `<`, `>` and `-`.

;; ================================================================
;; 1. Types: a clinical clock and its acceleration
;; ================================================================
;; EpigeneticClock / EpigeneticAgeAcceleration (system_types.metta) are the DNAm
;; kinds. A clinical clock is a sibling, not a subtype: the two are built from
;; different measurement modalities and the paper's central finding is that they
;; can be BENCHMARKED against each other.
(: ClinicalClock Type)
(Inheritance ClinicalClock Biomarker)

(: ClinicalAgeAcceleration Type)
(Inheritance ClinicalAgeAcceleration Biomarker)

;; A lab measurement that is a model input. CRP, HbA1c and FastingGlucose are
;; already Biomarkers (mechanistic_bridges.metta); the rest are typed here.
(: ClinicalLabMeasurement Type)
(Inheritance ClinicalLabMeasurement Biomarker)

;; The five inputs LinAge2 derives from NHANES questionnaire items or from other
;; labs rather than measuring directly. Kept apart so nothing mistakes a
;; comorbidity count for an analyte.
(: DerivedClinicalScore Type)
(Inheritance DerivedClinicalScore Biomarker)

;; ================================================================
;; 2. Publication provenance
;; ================================================================
(: FongEtAl2025_LinAge2 Publication)
(PublicationTitle FongEtAl2025_LinAge2 "LinAge2: providing actionable insights and benchmarking with epigenetic clocks")
(PublicationYear  FongEtAl2025_LinAge2 2025)
(JournalName      FongEtAl2025_LinAge2 "npj Aging")
(DOI              FongEtAl2025_LinAge2 "10.1038/s41514-025-00221-4")
(PubMedID         FongEtAl2025_LinAge2 "40268972")

;; ================================================================
;; 3. The clock and its acceleration
;; ================================================================
(Inheritance LinAge2 ClinicalClock)
(Inheritance LinAge2 MortalityRiskEstimator)
(SupportedByPublication LinAge2 FongEtAl2025_LinAge2)

;; LinAgeAccel is the BA-CA delta the service returns, in years. It is a
;; DIFFERENCE (clock age minus chronological age), not an age-regressed residual —
;; the AccelDefinition distinction nhanes_baseline.metta draws for the DNAm
;; clocks. By the model's construction (BA = CA + sum(PC x beta) / beta_age, the
;; null model's Cox coefficient on chronological age), one year of delta carries
;; the same log-hazard as one year of chronological age; that is what makes the
;; per-year hazard in linage2_fong2025_evidence.metta a statement about the
;; delta's scale rather than a fitted association.
(Inheritance LinAgeAccel ClinicalAgeAcceleration)
(ResidualOf  LinAgeAccel LinAge2)
(ValueEncoding LinAgeAccel "LinAge2 biological age minus chronological age, in YEARS (AccelDefinition Difference, not Residual). Positive = biologically older than chronological age.")
(SupportedByPublication LinAgeAccel FongEtAl2025_LinAge2)

;; Chronological age and sex enter the model as they do for GrimAge: age as the
;; null-model covariate the delta is scaled by, sex by selecting one of two
;; sex-specific models (separate PC loadings and Cox weights).
(ModelInput ChronologicalAgeAtBloodDraw LinAge2)
(ModelInput FemaleIndicator LinAge2)
(AdjustmentCovariate LinAgeAccel ChronologicalAgeAtBloodDraw)

;; ================================================================
;; 4. The 59 model inputs, with their NHANES variable codes
;; ================================================================
;; One (ModelInput F LinAge2) per input and one type, each line ending in a
;; STRUCTURED COMMENT `;; <NHANES code> — <description>` — the code the LinAge2
;; service uses for that input and the description the implementation publishes
;; for it. pln_chat/core/linage2_builder.py reads symbol, code and description off
;; these lines (regex in that module), so this file is the single source of truth
;; for what a caller may send (GET /linage2/features). They are comments, not
;; (NHANESCode …) facts, because the 59 string atoms cost more of the scoped
;; space's head-symbol/trie budget than it has (file header). Keep the format:
;; `(ModelInput <Sym> LinAge2)  (Inheritance <Sym> <Type>)  ;; <CODE> — <text>`.
;;
;; Fifty-four are measured analytes / vitals in NHANES SI units. Five are derived
;; by the implementation before the model sees them (§4b). The three lipid inputs
;; the implementation collects (LBDTCSI, LBDHDLSI, LBDSTRSI) are NOT model
;; inputs: they are consumed into the calculated LDL and dropped before
;; normalization, so a response never carries them.

;; --- 4a. measured inputs -----------------------------------------------------
;; vitals / anthropometry
(ModelInput PulseRate LinAge2)               (Inheritance PulseRate ClinicalLabMeasurement)   ;; BPXPLS — Pulse rate (beats/min, 60-sec pulse).
(ModelInput SystolicBloodPressure LinAge2)   (Inheritance SystolicBloodPressure ClinicalLabMeasurement)   ;; BPXSAR — Systolic blood pressure, average reported to examinee (mmHg).
(ModelInput DiastolicBloodPressure LinAge2)  (Inheritance DiastolicBloodPressure ClinicalLabMeasurement)   ;; BPXDAR — Diastolic blood pressure, average reported to examinee (mmHg).
(ModelInput BodyMassIndex LinAge2)           (Inheritance BodyMassIndex ClinicalLabMeasurement)   ;; BMXBMI — Body mass index (kg/m2).
;; kidney (urine)
(ModelInput UrineAlbumin LinAge2)            (Inheritance UrineAlbumin ClinicalLabMeasurement)   ;; URXUMASI — Urine albumin (microalbumin), SI units (mg/L).
(ModelInput UrineCreatinine LinAge2)         (Inheritance UrineCreatinine ClinicalLabMeasurement)   ;; URXUCRSI — Urine creatinine, SI units (µmol/L).
;; iron panel
(ModelInput SerumIron LinAge2)               (Inheritance SerumIron ClinicalLabMeasurement)   ;; LBDIRNSI — Serum iron (umol/L).
(ModelInput TotalIronBindingCapacity LinAge2) (Inheritance TotalIronBindingCapacity ClinicalLabMeasurement)   ;; LBDTIBSI — Total iron binding capacity, TIBC (umol/L).
(ModelInput TransferrinSaturation LinAge2)   (Inheritance TransferrinSaturation ClinicalLabMeasurement)   ;; LBXPCT — Transferrin saturation (%) = (serum iron / TIBC) x 100.
(ModelInput Ferritin LinAge2)                (Inheritance Ferritin ClinicalLabMeasurement)   ;; LBDFERSI — Ferritin (ug/L).
;; folate / B12 / smoking exposure
(ModelInput SerumFolate LinAge2)             (Inheritance SerumFolate ClinicalLabMeasurement)   ;; LBDFOLSI — Serum folate (nmol/L).
(ModelInput VitaminB12 LinAge2)              (Inheritance VitaminB12 ClinicalLabMeasurement)   ;; LBDB12SI — Vitamin B12 (pmol/L).
(ModelInput SerumCotinine LinAge2)           (Inheritance SerumCotinine ClinicalLabMeasurement)   ;; LBXCOT — Serum cotinine, digitized as the model was trained: 0 (<10 ng/mL, non-smoker), 1 (10-100), 2 (100-200), 3 (>=200, a typical daily smoker).
;; complete blood count — white cells
(ModelInput WhiteBloodCellCount LinAge2)     (Inheritance WhiteBloodCellCount ClinicalLabMeasurement)   ;; LBXWBCSI — White blood cell count (x10^9/L).
(ModelInput LymphocytePercent LinAge2)       (Inheritance LymphocytePercent ClinicalLabMeasurement)   ;; LBXLYPCT — Lymphocytes (%).
(ModelInput MonocytePercent LinAge2)         (Inheritance MonocytePercent ClinicalLabMeasurement)   ;; LBXMOPCT — Monocytes (%).
(ModelInput NeutrophilPercent LinAge2)       (Inheritance NeutrophilPercent ClinicalLabMeasurement)   ;; LBXNEPCT — Neutrophils (%).
(ModelInput EosinophilPercent LinAge2)       (Inheritance EosinophilPercent ClinicalLabMeasurement)   ;; LBXEOPCT — Eosinophils (%).
(ModelInput BasophilPercent LinAge2)         (Inheritance BasophilPercent ClinicalLabMeasurement)   ;; LBXBAPCT — Basophils (%).
(ModelInput LymphocyteCount LinAge2)         (Inheritance LymphocyteCount ClinicalLabMeasurement)   ;; LBDLYMNO — Lymphocytes (x10^9/L).
(ModelInput MonocyteCount LinAge2)           (Inheritance MonocyteCount ClinicalLabMeasurement)   ;; LBDMONO — Monocytes (x10^9/L).
(ModelInput NeutrophilCount LinAge2)         (Inheritance NeutrophilCount ClinicalLabMeasurement)   ;; LBDNENO — Neutrophils (x10^9/L).
(ModelInput EosinophilCount LinAge2)         (Inheritance EosinophilCount ClinicalLabMeasurement)   ;; LBDEONO — Eosinophils (x10^9/L).
(ModelInput BasophilCount LinAge2)           (Inheritance BasophilCount ClinicalLabMeasurement)   ;; LBDBANO — Basophils (x10^9/L).
;; complete blood count — red cells / platelets
(ModelInput RedBloodCellCount LinAge2)       (Inheritance RedBloodCellCount ClinicalLabMeasurement)   ;; LBXRBCSI — Red blood cell count (x10^12/L).
(ModelInput Hemoglobin LinAge2)              (Inheritance Hemoglobin ClinicalLabMeasurement)   ;; LBXHGB — Hemoglobin (g/dL).
(ModelInput Hematocrit LinAge2)              (Inheritance Hematocrit ClinicalLabMeasurement)   ;; LBXHCT — Hematocrit (%).
(ModelInput MeanCorpuscularVolume LinAge2)   (Inheritance MeanCorpuscularVolume ClinicalLabMeasurement)   ;; LBXMCVSI — Mean corpuscular volume, MCV (fL).
(ModelInput MeanCorpuscularHemoglobin LinAge2) (Inheritance MeanCorpuscularHemoglobin ClinicalLabMeasurement)   ;; LBXMCHSI — Mean corpuscular hemoglobin, MCH (pg).
(ModelInput MeanCorpuscularHemoglobinConcentration LinAge2) (Inheritance MeanCorpuscularHemoglobinConcentration ClinicalLabMeasurement)   ;; LBXMC — Mean corpuscular hemoglobin concentration, MCHC (g/dL).
(ModelInput RedCellDistributionWidth LinAge2) (Inheritance RedCellDistributionWidth ClinicalLabMeasurement)   ;; LBXRDW — Red cell distribution width (%).
(ModelInput PlateletCount LinAge2)           (Inheritance PlateletCount ClinicalLabMeasurement)   ;; LBXPLTSI — Platelet count (x10^9/L).
(ModelInput MeanPlateletVolume LinAge2)      (Inheritance MeanPlateletVolume ClinicalLabMeasurement)   ;; LBXMPSI — Mean platelet volume, MPV (fL).
;; inflammation / glycemia / cardiac
(ModelInput CRP LinAge2)                     (Inheritance CRP ClinicalLabMeasurement)   ;; LBXCRP — C-reactive protein (mg/dL, as NHANES 1999-2002 reports it: 1 mg/L = 0.1 mg/dL).
(ModelInput HbA1c LinAge2)                   (Inheritance HbA1c ClinicalLabMeasurement)   ;; LBXGH — Glycohemoglobin (HbA1c, %).
(ModelInput NTproBNP LinAge2)                (Inheritance NTproBNP ClinicalLabMeasurement)   ;; SSBNP — N-terminal pro-B-type natriuretic peptide (NT-proBNP, pg/mL).
;; standard biochemistry profile (SI)
(ModelInput SerumAlbumin LinAge2)            (Inheritance SerumAlbumin ClinicalLabMeasurement)   ;; LBDSALSI — Albumin (g/L).
(ModelInput ALT LinAge2)                     (Inheritance ALT ClinicalLabMeasurement)   ;; LBXSATSI — Alanine aminotransferase, ALT (U/L).
(ModelInput AST LinAge2)                     (Inheritance AST ClinicalLabMeasurement)   ;; LBXSASSI — Aspartate aminotransferase, AST (U/L).
(ModelInput AlkalinePhosphatase LinAge2)     (Inheritance AlkalinePhosphatase ClinicalLabMeasurement)   ;; LBXSAPSI — Alkaline phosphatase (U/L).
(ModelInput BloodUreaNitrogen LinAge2)       (Inheritance BloodUreaNitrogen ClinicalLabMeasurement)   ;; LBDSBUSI — Urea nitrogen (BUN), SI (mmol/L).
(ModelInput SerumCalcium LinAge2)            (Inheritance SerumCalcium ClinicalLabMeasurement)   ;; LBDSCASI — Calcium, SI (mmol/L).
(ModelInput Bicarbonate LinAge2)             (Inheritance Bicarbonate ClinicalLabMeasurement)   ;; LBXSC3SI — Bicarbonate (total CO2), SI (mmol/L).
(ModelInput SerumGlucose LinAge2)            (Inheritance SerumGlucose ClinicalLabMeasurement)   ;; LBDSGLSI — Glucose from the biochemistry profile, SI (mmol/L); fasting only if the draw was.
(ModelInput LactateDehydrogenase LinAge2)    (Inheritance LactateDehydrogenase ClinicalLabMeasurement)   ;; LBXSLDSI — Lactate dehydrogenase, LDH (U/L).
(ModelInput SerumPhosphorus LinAge2)         (Inheritance SerumPhosphorus ClinicalLabMeasurement)   ;; LBDSPHSI — Phosphorus (mmol/L).
(ModelInput TotalBilirubin LinAge2)          (Inheritance TotalBilirubin ClinicalLabMeasurement)   ;; LBDSTBSI — Total bilirubin (umol/L).
(ModelInput TotalProtein LinAge2)            (Inheritance TotalProtein ClinicalLabMeasurement)   ;; LBDSTPSI — Total protein (g/L).
(ModelInput UricAcid LinAge2)                (Inheritance UricAcid ClinicalLabMeasurement)   ;; LBDSUASI — Uric acid (umol/L).
(ModelInput SerumCreatinine LinAge2)         (Inheritance SerumCreatinine ClinicalLabMeasurement)   ;; LBDSCRSI — Creatinine (umol/L).
(ModelInput SerumSodium LinAge2)             (Inheritance SerumSodium ClinicalLabMeasurement)   ;; LBXSNASI — Sodium (mmol/L).
(ModelInput SerumPotassium LinAge2)          (Inheritance SerumPotassium ClinicalLabMeasurement)   ;; LBXSKSI — Potassium (mmol/L).
(ModelInput SerumChloride LinAge2)           (Inheritance SerumChloride ClinicalLabMeasurement)   ;; LBXSCLSI — Chloride (mmol/L).
(ModelInput Globulin LinAge2)                (Inheritance Globulin ClinicalLabMeasurement)   ;; LBDSGBSI — Globulin (g/L).

;; --- 4b. derived inputs -------------------------------------------------------
;; Computed by the implementation (src.py popPCFIfs1/2/3, populateLDL, and the
;; albumin/creatinine ratio in linage2_service.py) from questionnaire items or
;; other labs; the service reports a years-contribution for each exactly as for
;; a measured analyte.
(ModelInput ComorbidityScore LinAge2)        (Inheritance ComorbidityScore DerivedClinicalScore)   ;; fs1Score — Fraction of 22 self-reported diagnoses/events endorsed (hypertension, diabetes, kidney, asthma, anemia, arthritis, CHD, angina, MI, stroke, emphysema, thyroid, obesity, bronchitis, liver, cancer, hip/wrist/spine fracture, osteoporosis, memory problems, overnight hospital stay).
(ModelInput SelfRatedHealthScore LinAge2)    (Inheritance SelfRatedHealthScore DerivedClinicalScore)   ;; fs2Score — Self-rated general health (HUQ010: fair = 2, poor = 4) scaled by trend vs one year ago (HUQ020: better x0.5, worse x2).
(ModelInput HealthcareUseScore LinAge2)      (Inheritance HealthcareUseScore DerivedClinicalScore)   ;; fs3Score — Number of times healthcare was received in the past year (HUQ050).
(ModelInput CalculatedLDL LinAge2)           (Inheritance CalculatedLDL ClinicalLabMeasurement)   ;; LDLV — LDL cholesterol calculated as total cholesterol - HDL - triglycerides/5 (mmol/L inputs; the implementation's Friedewald variant).
(ModelInput UrineAlbuminCreatinineRatio LinAge2) (Inheritance UrineAlbuminCreatinineRatio ClinicalLabMeasurement)   ;; crAlbRat — Urine albumin / urine creatinine ratio, as computed by the implementation.

;; ================================================================
;; 5. Which inputs read out a biomarker the KB already reasons about
;; ================================================================
;; (MeasuresBiomarker <input> <KB biomarker>): the input IS a measurement of that
;; biomarker, so the patient's own standardized value for the biomarker (patient-z)
;; is the honest witness for the direction of the input, and the biomarker's
;; incoming Effect edges (mechanistic_bridges.metta) are the causes a positive
;; contribution may be credited to. Only SIX of the 59 inputs have one; the
;; other 53 are carried as years and left unexplained until a bridge exists.
;;
;; Two of the six are the same symbol on both sides: LinAge2's CRP and HbA1c ARE
;; the KB's CRP and HbA1c. The identity fact is what lets one rule serve all six.
(: MeasuresBiomarker (-> Atom Biomarker Atom))
(MeasuresBiomarker CRP   CRP)
(MeasuresBiomarker HbA1c HbA1c)

;; Red cell distribution width and serum albumin read out the two inflammation markers
;; mechanistic_bridges.metta added (ChronicInflammation -> RDW, -> LowSerumAlbumin). Albumin is
;; the one input whose KB node is the OPPOSITE direction: LowSerumAlbumin is the DEFICIT, so
;; its patient-z is the negated albumin z. A positive contribution of either input is credited to
;; a cause only when that z is Elevated (a witness), exactly like CRP.
(MeasuresBiomarker RedCellDistributionWidth RDW)
(MeasuresBiomarker SerumAlbumin LowSerumAlbumin)

;; The biochemistry-profile glucose stands in for the KB's FastingGlucose. It is
;; the same analyte, but it is fasting only if the draw was — NHANES has a
;; separate fasting-subsample glucose (LBXGLU) that LinAge2 does not use. A
;; non-fasting reading pushed through the metabolic axis overstates insulin
;; resistance; the caveat travels with the symbol name (SerumGlucose, not
;; FastingGlucose) and is repeated by the API.
(MeasuresBiomarker SerumGlucose FastingGlucose)

;; Serum cotinine is the biochemical readout of CURRENT tobacco exposure, not of
;; cumulative pack-years (the quantity DNAmPACKYRS surrogates, grim_age_core.metta).
;; So it gets its own exposure node rather than being mapped onto
;; SmokingPackYears — mapping it there would let a cessation counterfactual
;; claim that quitting undoes a cumulative exposure, which it does not.
(Inheritance CurrentTobaccoExposure ExposureBiomarker)
(MeasuresBiomarker SerumCotinine CurrentTobaccoExposure)

;; ================================================================
;; 6. The one Effect edge this layer adds: cessation clears current exposure
;; ================================================================
;; HONESTY CONTRACT — identical to mechanistic_bridges.metta: the strength is a
;; CURATED PRIOR, quarantined here with its anchor; the confidence is a lookup
;; into epistemic_calibration.metta, written unevaluated.
;;
;; CURATED PRIOR 0.95 (magnitude). Cotinine's elimination half-life is ~16-18 h
;; (Dempsey et al. 2013 measured 17.9 h, 95% CI 16.5-19.3, "similar to that
;; reported in adults"), so a quitter's serum cotinine reaches non-smoker levels
;; within days, and cotinine is the SRNT's recommended biochemical verification
;; of abstinence. It is not 1.0 because LinAge2's LBXCOT is a four-level
;; digitization that also responds to second-hand exposure and nicotine
;; replacement, both of which survive quitting cigarettes.
;;
;; TIER: MultipleHumanTrials (0.85) — a pharmacokinetic quantity replicated across
;; many human studies, mapped conservatively as lifestyle_evidence.metta maps its
;; meta-analysis (the KB has no pharmacokinetic tier and gains nothing from
;; inventing one for a single edge).
;;
;; WHAT IT DOES NOT DO. This edge lands on CurrentTobaccoExposure, which no
;; GrimAge component reads out, so `counterfactual-patient` / `project-risk-
;; patient` (the DNAm clock's counterfactuals) are unchanged by it: `resolve-lever`
;; still finds SmokingPackYears first, exactly as before (asserted in
;; tests/test_linage2.py). It is consumed only by pln_linage2.metta, where the
;; lever's precondition is finally read by the ENGINE: a cessation counterfactual
;; credits the cotinine contribution only for a patient recorded as
;; (PatientSmoking <P> CurrentSmoker) — the gate lifestyle_evidence.metta declares
;; ((LeverRequiresSmoking SmokingCessation CurrentSmoker)) and
;; pln_counterfactual.metta §3b could not afford to enforce in the shared space.
(: DempseyEtAl2013_CotinineHalfLife Publication)
(PublicationTitle DempseyEtAl2013_CotinineHalfLife "CYP2A6 genotype but not age determines cotinine half-life in infants and children")
(PublicationYear  DempseyEtAl2013_CotinineHalfLife 2013)
(JournalName      DempseyEtAl2013_CotinineHalfLife "Clinical Pharmacology and Therapeutics")
(DOI              DempseyEtAl2013_CotinineHalfLife "10.1038/clpt.2013.114")
(PubMedID         DempseyEtAl2013_CotinineHalfLife "23714690")

(: SRNT2002_BiochemicalVerification Publication)
(PublicationTitle SRNT2002_BiochemicalVerification "Biochemical verification of tobacco use and cessation")
(PublicationYear  SRNT2002_BiochemicalVerification 2002)
(JournalName      SRNT2002_BiochemicalVerification "Nicotine and Tobacco Research")
(DOI              SRNT2002_BiochemicalVerification "10.1080/14622200210123581")
(PubMedID         SRNT2002_BiochemicalVerification "12028847")

(Effect SmokingCessation CurrentTobaccoExposure Neg
   (stv 0.95 (evidence-confidence MultipleHumanTrials)))
