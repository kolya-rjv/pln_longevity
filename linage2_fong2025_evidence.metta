;; linage2_fong2025_evidence.metta
;; The quantitative record behind the LinAge2 clock (Fong et al. 2025).      [v1]
;;
;; Same conventions as grim_age_lu2019_evidence.metta: source-faithful measurement
;; records, NOT PLN truth values. evidence_calibration.metta turns a record into
;; an (stv s c) on demand through `calibrate-tv`; pln_linage2.metta reads the
;; hazard ratio off the record through `linage-hr` and never hand-types it.
;;
;; Load AFTER linage2_core.metta (LinAgeAccel, FongEtAl2025_LinAge2) and
;; grim_age_lu2019_evidence.metta ((Inheritance AllCauseMortality Outcome)).
;; Query-scoped, like the rest of the LinAge2 layer — see linage2_core.metta.

;; ------------------------------------------------------------
;; The per-year hazard of the LinAge2 delta for all-cause mortality
;; ------------------------------------------------------------
;; WHAT THE PAPER REPORTS, and how the number below follows from it. Fong et al.
;; do not print a hazard ratio per year of BA-CA delta. They report that the null
;; Cox model — chronological age as the only covariate, the model the LinAge2
;; delta is SCALED BY — has a mortality rate doubling time (MRDT) of
;; "approximately 7.8 years", in good agreement with the literature value of ~8
;; (Methods, "Motivation for enhancing LinAge2"). Because biological age is
;; defined as BA = CA + sum(PC_i x beta_i) / beta_age, one year of delta carries
;; exactly the log-hazard of one year of chronological age in that null model, so
;;
;;     HR per year of delta = 2 ^ (1 / MRDT) = 2 ^ (1 / 7.8) = 1.093
;;
;; That is a statement about the delta's SCALE BY CONSTRUCTION, not an
;; independently fitted association, and EffectScale says so. Cross-check, not
;; used: read off the published model artifacts (Rejuve/LinAge2-Python, commit
;; d0b9973), the sex-specific null-model coefficients give HR 1.083/yr (male,
;; MRDT 8.7 y) and 1.106/yr (female, MRDT 6.9 y), bracketing the pooled 1.093.
;;
;; WHY NOT a per-SD or per-z scale: the delta is the model's native unit and the
;; unit the service returns; pln_linage2.metta exponentiates the HR by the delta
;; in years directly, so no sd->years knob enters the risk (the knob in that
;; file serves only the qualitative Elevated/Normal/Low status).
;;
;; TIER: Epidemiological (0.60). One cohort (NHANES 1999-2000 training,
;; 2001-2002 test), observational, mortality-linked. It is deliberately below
;; MultipleHumanTrials, which the KB reserves for Lu 2019's multi-cohort
;; meta-analytic GrimAge hazards: a single-cohort clock is weaker evidence than a
;; meta-analysis, whatever its AUC.
(: Fong2025_LinAgeAccel_AllCauseMortality EmpiricalAssociation)
(HasPredictor  Fong2025_LinAgeAccel_AllCauseMortality LinAgeAccel)
(HasOutcome    Fong2025_LinAgeAccel_AllCauseMortality AllCauseMortality)
(EffectMetric  Fong2025_LinAgeAccel_AllCauseMortality "hazard_ratio")
(EffectValue   Fong2025_LinAgeAccel_AllCauseMortality 1.093)
(EffectScale   Fong2025_LinAgeAccel_AllCauseMortality "per one-year increase in the LinAge2 BA-CA delta; by construction equal to the null model's hazard per year of chronological age, derived from the reported mortality rate doubling time of ~7.8 years (2^(1/7.8))")
(SupportedByPublication Fong2025_LinAgeAccel_AllCauseMortality FongEtAl2025_LinAge2)
(HasEvidenceCategory    Fong2025_LinAgeAccel_AllCauseMortality Epidemiological)

;; ------------------------------------------------------------
;; What is deliberately NOT recorded here
;; ------------------------------------------------------------
;; * No cause-specific hazard. The paper shows LinAge2 predicts specific causes
;;   of death within a 5-year window but reports no per-year hazard for any, so
;;   there is no LinAgeAccel -> CoronaryHeartDisease record and
;;   (linage-risk … CoronaryHeartDisease) yields nothing rather than borrowing
;;   the all-cause number.
;; * No numeric AUCs. The 10- and 20-year mortality ROC comparisons against
;;   PhenoAge, GrimAge2, DunedinPoAm, HorvathAge and HannumAge are in the paper's
;;   figures, not its text, and nothing here could be verified against them.
;;   The qualitative findings the text does state — LinAge2 ~ GrimAge2;
;;   LinAge2 > PhenoAge DNAm, DunedinPoAm and chronological age; HorvathAge,
;;   HannumAge and PhenoAge DNAm not significantly better than chronological
;;   age for mortality — are recorded in docs/linage2_integration.md, as prose,
;;   where a comparison without a number belongs.
;; * No per-feature weights. They are sex-specific projections that flip sign
;;   between the male and female models (linage2_core.metta header), the
;;   service applies them, and copying them here would drift the moment the
;;   model is retrained.
