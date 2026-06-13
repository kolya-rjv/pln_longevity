;; grim_age_core.cleaned.metta
;; DNAm GrimAge ontology and model composition
;; Source: Lu et al. (2019), "DNA methylation GrimAge strongly predicts
;; lifespan and healthspan", AGING 2019, Vol. 11, No. 2.
;;
;; Load after:
;;   system_types.cleaned.metta
;;   logical_predicates.cleaned.metta
;;
;; This file contains structural, paper-backed atoms. Quantitative
;; association records are kept in grim_age_lu2019_evidence.metta.

;; ------------------------------------------------------------
;; 1. Local type declarations and hierarchy
;; ------------------------------------------------------------
(: DNAmSurrogateBiomarker Type)
(Inheritance DNAmSurrogateBiomarker Biomarker)

(: DNAmProteinSurrogate Type)
(Inheritance DNAmProteinSurrogate DNAmSurrogateBiomarker)

(: DNAmExposureSurrogate Type)
(Inheritance DNAmExposureSurrogate DNAmSurrogateBiomarker)

(: ModelCovariate Type)

(: PlasmaProteinBiomarker Type)
(Inheritance PlasmaProteinBiomarker Biomarker)

(: ExposureBiomarker Type)
(Inheritance ExposureBiomarker Biomarker)

;; ------------------------------------------------------------
;; 2. Publication provenance
;; ------------------------------------------------------------
(: LuEtAl2019_GrimAge Publication)
(PublicationTitle LuEtAl2019_GrimAge "DNA methylation GrimAge strongly predicts lifespan and healthspan")
(PublicationYear LuEtAl2019_GrimAge 2019)
(JournalName LuEtAl2019_GrimAge "AGING")

;; ------------------------------------------------------------
;; 3. GrimAge and its age-adjusted residual
;; ------------------------------------------------------------
(Inheritance GrimAge EpigeneticClock)
(Inheritance GrimAge MortalityRiskEstimator)
(SupportedByPublication GrimAge LuEtAl2019_GrimAge)

(Inheritance AgeAccelGrim EpigeneticAgeAcceleration)
(ResidualOf AgeAccelGrim GrimAge)
(SupportedByPublication AgeAccelGrim LuEtAl2019_GrimAge)

;; ------------------------------------------------------------
;; 4. Non-DNAm model covariates
;; ------------------------------------------------------------
;; The Lu et al. model includes chronological age and a female indicator.
(Inheritance ChronologicalAgeAtBloodDraw ModelCovariate)
(Inheritance FemaleIndicator ModelCovariate)
(ValueEncoding FemaleIndicator "1 for female, 0 otherwise")

(ModelInput ChronologicalAgeAtBloodDraw GrimAge)
(ModelInput FemaleIndicator GrimAge)
(AdjustmentCovariate AgeAccelGrim ChronologicalAgeAtBloodDraw)

;; ------------------------------------------------------------
;; 5. GrimAge DNAm surrogate model inputs
;; ------------------------------------------------------------
(Inheritance DNAmPACKYRS DNAmExposureSurrogate)
(Inheritance DNAmADM DNAmProteinSurrogate)
(Inheritance DNAmB2M DNAmProteinSurrogate)
(Inheritance DNAmCystatinC DNAmProteinSurrogate)
(Inheritance DNAmGDF15 DNAmProteinSurrogate)
(Inheritance DNAmLeptin DNAmProteinSurrogate)
(Inheritance DNAmPAI1 DNAmProteinSurrogate)
(Inheritance DNAmTIMP1 DNAmProteinSurrogate)

(ModelInput DNAmPACKYRS GrimAge)
(ModelInput DNAmADM GrimAge)
(ModelInput DNAmB2M GrimAge)
(ModelInput DNAmCystatinC GrimAge)
(ModelInput DNAmGDF15 GrimAge)
(ModelInput DNAmLeptin GrimAge)
(ModelInput DNAmPAI1 GrimAge)
(ModelInput DNAmTIMP1 GrimAge)

(PartOf DNAmPACKYRS GrimAge)
(PartOf DNAmADM GrimAge)
(PartOf DNAmB2M GrimAge)
(PartOf DNAmCystatinC GrimAge)
(PartOf DNAmGDF15 GrimAge)
(PartOf DNAmLeptin GrimAge)
(PartOf DNAmPAI1 GrimAge)
(PartOf DNAmTIMP1 GrimAge)

;; Main-text CpG counts explicitly reported by Lu et al.
(HasCpGCount DNAmPACKYRS 172)
(HasCpGCount DNAmGDF15 137)
(HasCpGCount DNAmPAI1 211)
(HasTotalUniqueCpGCount GrimAge 1030)

;; ------------------------------------------------------------
;; 6. Observed biological quantities represented by DNAm surrogates
;; ------------------------------------------------------------
(Inheritance SmokingPackYears ExposureBiomarker)

(Inheritance PlasmaADM PlasmaProteinBiomarker)
(Inheritance PlasmaB2M PlasmaProteinBiomarker)
(Inheritance PlasmaCystatinC PlasmaProteinBiomarker)
(Inheritance PlasmaGDF15 PlasmaProteinBiomarker)
(Inheritance PlasmaLeptin PlasmaProteinBiomarker)
(Inheritance PlasmaPAI1 PlasmaProteinBiomarker)
(Inheritance PlasmaTIMP1 PlasmaProteinBiomarker)

(SurrogateFor DNAmPACKYRS SmokingPackYears)
(SurrogateFor DNAmADM PlasmaADM)
(SurrogateFor DNAmB2M PlasmaB2M)
(SurrogateFor DNAmCystatinC PlasmaCystatinC)
(SurrogateFor DNAmGDF15 PlasmaGDF15)
(SurrogateFor DNAmLeptin PlasmaLeptin)
(SurrogateFor DNAmPAI1 PlasmaPAI1)
(SurrogateFor DNAmTIMP1 PlasmaTIMP1)

;; ------------------------------------------------------------
;; 7. Gene and protein entities
;; ------------------------------------------------------------
(: ADM Gene)
(: B2M Gene)
(: CST3 Gene)
(: GDF15 Gene)
(: LEP Gene)
(: SERPINE1 Gene)
(: TIMP1 Gene)

(GeneSymbol ADM "ADM")
(GeneSymbol B2M "B2M")
(GeneSymbol CST3 "CST3")
(GeneSymbol GDF15 "GDF15")
(GeneSymbol LEP "LEP")
(GeneSymbol SERPINE1 "SERPINE1")
(GeneSymbol TIMP1 "TIMP1")

(: Adrenomedullin Protein)
(: Beta2Microglobulin Protein)
(: CystatinC Protein)
(: GrowthDifferentiationFactor15 Protein)
(: Leptin Protein)
(: PlasminogenActivatorInhibitor1 Protein)
(: TissueInhibitorMetalloproteinase1 Protein)

(EncodedBy Adrenomedullin ADM)
(EncodedBy Beta2Microglobulin B2M)
(EncodedBy CystatinC CST3)
(EncodedBy GrowthDifferentiationFactor15 GDF15)
(EncodedBy Leptin LEP)
(EncodedBy PlasminogenActivatorInhibitor1 SERPINE1)
(EncodedBy TissueInhibitorMetalloproteinase1 TIMP1)

;; ------------------------------------------------------------
;; 8. Biological interpretation links
;; ------------------------------------------------------------
;; Reflects is reserved for interpretation rather than model identity.
(Reflects DNAmADM Adrenomedullin)
(Reflects DNAmB2M Beta2Microglobulin)
(Reflects DNAmCystatinC CystatinC)
(Reflects DNAmGDF15 GrowthDifferentiationFactor15)
(Reflects DNAmLeptin Leptin)
(Reflects DNAmPAI1 PlasminogenActivatorInhibitor1)
(Reflects DNAmTIMP1 TissueInhibitorMetalloproteinase1)
