;; grim_age_lu2019_evidence.metta
;; Exact quantitative evidence records from Lu et al. (2019).
;;
;; These are source measurements, not PLN truth values. A separate
;; calibration layer should convert evidence records into TVs using an
;; explicit policy. This avoids presenting arbitrary confidence values
;; as if they were reported by the paper.

;; Outcomes
(Inheritance AllCauseMortality Outcome)
(Inheritance CoronaryHeartDisease Outcome)
(Inheritance CongestiveHeartFailure Outcome)

;; ------------------------------------------------------------
;; AgeAccelGrim predicts all-cause mortality
;; Meta-analysis: HR=1.10 per one-year increase; P=2.0E-75.
;; ------------------------------------------------------------
(: Lu2019_AgeAccelGrim_AllCauseMortality EmpiricalAssociation)
(HasPredictor Lu2019_AgeAccelGrim_AllCauseMortality AgeAccelGrim)
(HasOutcome Lu2019_AgeAccelGrim_AllCauseMortality AllCauseMortality)
(EffectMetric Lu2019_AgeAccelGrim_AllCauseMortality "hazard_ratio")
(EffectValue Lu2019_AgeAccelGrim_AllCauseMortality 1.10)
(EffectScale Lu2019_AgeAccelGrim_AllCauseMortality "per one-year increase in AgeAccelGrim")
(PValue Lu2019_AgeAccelGrim_AllCauseMortality 2.0e-75)
(SupportedByPublication Lu2019_AgeAccelGrim_AllCauseMortality LuEtAl2019_GrimAge)

;; ------------------------------------------------------------
;; AgeAccelGrim predicts incident coronary heart disease
;; Meta-analysis: HR=1.07 per one-year increase; P=6.2E-24.
;; ------------------------------------------------------------
(: Lu2019_AgeAccelGrim_CHD EmpiricalAssociation)
(HasPredictor Lu2019_AgeAccelGrim_CHD AgeAccelGrim)
(HasOutcome Lu2019_AgeAccelGrim_CHD CoronaryHeartDisease)
(EffectMetric Lu2019_AgeAccelGrim_CHD "hazard_ratio")
(EffectValue Lu2019_AgeAccelGrim_CHD 1.07)
(EffectScale Lu2019_AgeAccelGrim_CHD "per one-year increase in AgeAccelGrim")
(PValue Lu2019_AgeAccelGrim_CHD 6.2e-24)
(SupportedByPublication Lu2019_AgeAccelGrim_CHD LuEtAl2019_GrimAge)

;; ------------------------------------------------------------
;; DNAm PAI-1 predicts incident coronary heart disease
;; HR=1.31 per standard-deviation increase; P=3.6E-12.
;; ------------------------------------------------------------
(: Lu2019_DNAmPAI1_CHD EmpiricalAssociation)
(HasPredictor Lu2019_DNAmPAI1_CHD DNAmPAI1)
(HasOutcome Lu2019_DNAmPAI1_CHD CoronaryHeartDisease)
(EffectMetric Lu2019_DNAmPAI1_CHD "hazard_ratio")
(EffectValue Lu2019_DNAmPAI1_CHD 1.31)
(EffectScale Lu2019_DNAmPAI1_CHD "per standard-deviation increase in age-adjusted DNAm PAI-1")
(PValue Lu2019_DNAmPAI1_CHD 3.6e-12)
(SupportedByPublication Lu2019_DNAmPAI1_CHD LuEtAl2019_GrimAge)

;; ------------------------------------------------------------
;; AgeAccelGrim predicts congestive heart failure
;; HR=1.10; P=4.9E-9.
;; ------------------------------------------------------------
(: Lu2019_AgeAccelGrim_CHF EmpiricalAssociation)
(HasPredictor Lu2019_AgeAccelGrim_CHF AgeAccelGrim)
(HasOutcome Lu2019_AgeAccelGrim_CHF CongestiveHeartFailure)
(EffectMetric Lu2019_AgeAccelGrim_CHF "hazard_ratio")
(EffectValue Lu2019_AgeAccelGrim_CHF 1.10)
(EffectScale Lu2019_AgeAccelGrim_CHF "reported AgeAccelGrim scale")
(PValue Lu2019_AgeAccelGrim_CHF 4.9e-9)
(SupportedByPublication Lu2019_AgeAccelGrim_CHF LuEtAl2019_GrimAge)
