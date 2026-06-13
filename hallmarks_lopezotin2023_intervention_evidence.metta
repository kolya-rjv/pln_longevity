;; hallmarks_lopezotin2023_intervention_evidence.metta
;; Selected review-level intervention examples from Table 1 of
;; Lopez-Otin et al. (2023), "Hallmarks of aging: An expanding universe".
;;
;; Purpose: retain auditable intervention -> hallmark examples for the MVP.
;; These records preserve review wording and reference numbers. They are NOT
;; calibrated PLN truth values and should not be treated as primary-study rows.

;; ------------------------------------------------------------
;; Intervention entity typing
;; ------------------------------------------------------------
(Inheritance AlphaKetoglutarate Supplement)
(Inheritance GH_DHEA_Metformin CombinationIntervention)
(Inheritance ATG5Overexpression GeneticIntervention)
(Inheritance BECN1_F121A GeneticIntervention)
(Inheritance Spermidine Supplement)
(Inheritance NMN Supplement)
(Inheritance UrolithinA Supplement)
(Inheritance CaloricRestriction Lifestyle)
(Inheritance Elamipretide Pharmaceutical)
(Inheritance DasatinibPlusQuercetin CombinationIntervention)
(Inheritance Fisetin Supplement)
(Inheritance Canakinumab Pharmaceutical)
(Inheritance AkkermansiaMuciniphila Supplement)

;; ------------------------------------------------------------
;; Epigenetic alterations
;; ------------------------------------------------------------
(: LopezOtin2023_AlphaKetoglutarate_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_AlphaKetoglutarate_Human EpigeneticAlterations)
(EvidenceIntervention LopezOtin2023_AlphaKetoglutarate_Human AlphaKetoglutarate)
(EvidenceSpeciesModel LopezOtin2023_AlphaKetoglutarate_Human Human)
(EvidenceOutcomeText LopezOtin2023_AlphaKetoglutarate_Human "delayed epigenetic clock")
(EvidenceReferenceNumber LopezOtin2023_AlphaKetoglutarate_Human 26)
(SupportedByPublication LopezOtin2023_AlphaKetoglutarate_Human LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_GH_DHEA_Metformin_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_GH_DHEA_Metformin_Human EpigeneticAlterations)
(EvidenceIntervention LopezOtin2023_GH_DHEA_Metformin_Human GH_DHEA_Metformin)
(EvidenceSpeciesModel LopezOtin2023_GH_DHEA_Metformin_Human Human)
(EvidenceOutcomeText LopezOtin2023_GH_DHEA_Metformin_Human "delayed epigenetic clock")
(EvidenceReferenceNumber LopezOtin2023_GH_DHEA_Metformin_Human 27)
(SupportedByPublication LopezOtin2023_GH_DHEA_Metformin_Human LopezOtinEtAl2023_Hallmarks)

;; ------------------------------------------------------------
;; Disabled macroautophagy
;; ------------------------------------------------------------
(: LopezOtin2023_ATG5Overexpression_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_ATG5Overexpression_Mouse DisabledMacroautophagy)
(EvidenceIntervention LopezOtin2023_ATG5Overexpression_Mouse ATG5Overexpression)
(EvidenceSpeciesModel LopezOtin2023_ATG5Overexpression_Mouse Mouse)
(EvidenceOutcomeText LopezOtin2023_ATG5Overexpression_Mouse "improved longevity, metabolic health and motor function")
(EvidenceReferenceNumber LopezOtin2023_ATG5Overexpression_Mouse 38)
(SupportedByPublication LopezOtin2023_ATG5Overexpression_Mouse LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_BECN1_F121A_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_BECN1_F121A_Mouse DisabledMacroautophagy)
(EvidenceIntervention LopezOtin2023_BECN1_F121A_Mouse BECN1_F121A)
(EvidenceSpeciesModel LopezOtin2023_BECN1_F121A_Mouse Mouse)
(EvidenceOutcomeText LopezOtin2023_BECN1_F121A_Mouse "extended longevity and prolonged neurogenesis")
(EvidenceReferenceNumber LopezOtin2023_BECN1_F121A_Mouse 39)
(SupportedByPublication LopezOtin2023_BECN1_F121A_Mouse LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_Spermidine_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_Spermidine_Mouse DisabledMacroautophagy)
(EvidenceIntervention LopezOtin2023_Spermidine_Mouse Spermidine)
(EvidenceSpeciesModel LopezOtin2023_Spermidine_Mouse Mouse)
(EvidenceOutcomeText LopezOtin2023_Spermidine_Mouse "extended longevity, reduced cardiac aging and oxidative stress")
(EvidenceReferenceNumber LopezOtin2023_Spermidine_Mouse 41)
(SupportedByPublication LopezOtin2023_Spermidine_Mouse LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_NMN_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_NMN_Human DisabledMacroautophagy)
(EvidenceIntervention LopezOtin2023_NMN_Human NMN)
(EvidenceSpeciesModel LopezOtin2023_NMN_Human HumanPrediabeticWomen)
(EvidenceOutcomeText LopezOtin2023_NMN_Human "increased insulin sensitivity of skeletal muscle")
(EvidenceReferenceNumber LopezOtin2023_NMN_Human 44)
(SupportedByPublication LopezOtin2023_NMN_Human LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_UrolithinA_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_UrolithinA_Human DisabledMacroautophagy)
(EvidenceIntervention LopezOtin2023_UrolithinA_Human UrolithinA)
(EvidenceSpeciesModel LopezOtin2023_UrolithinA_Human HumanMiddleAgedAdults)
(EvidenceOutcomeText LopezOtin2023_UrolithinA_Human "improved aerobic endurance and physical performance; reduced plasma CRP")
(EvidenceReferenceNumber LopezOtin2023_UrolithinA_Human 47)
(SupportedByPublication LopezOtin2023_UrolithinA_Human LopezOtinEtAl2023_Hallmarks)

;; ------------------------------------------------------------
;; Deregulated nutrient-sensing
;; ------------------------------------------------------------
(: LopezOtin2023_CaloricRestriction_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_CaloricRestriction_Human DeregulatedNutrientSensing)
(EvidenceIntervention LopezOtin2023_CaloricRestriction_Human CaloricRestriction)
(EvidenceSpeciesModel LopezOtin2023_CaloricRestriction_Human Human)
(EvidenceOutcomeText LopezOtin2023_CaloricRestriction_Human "improved thymopoiesis and anti-inflammatory effects on adipose tissues; reduced PLA2G7")
(EvidenceReferenceNumber LopezOtin2023_CaloricRestriction_Human 50)
(SupportedByPublication LopezOtin2023_CaloricRestriction_Human LopezOtinEtAl2023_Hallmarks)

;; ------------------------------------------------------------
;; Mitochondrial dysfunction
;; ------------------------------------------------------------
(: LopezOtin2023_Elamipretide_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_Elamipretide_Human MitochondrialDysfunction)
(EvidenceIntervention LopezOtin2023_Elamipretide_Human Elamipretide)
(EvidenceSpeciesModel LopezOtin2023_Elamipretide_Human HumanBarthSyndrome)
(EvidenceOutcomeText LopezOtin2023_Elamipretide_Human "improved walking test, muscle and cardiac parameters, overall improvement")
(EvidenceReferenceNumber LopezOtin2023_Elamipretide_Human 56)
(SupportedByPublication LopezOtin2023_Elamipretide_Human LopezOtinEtAl2023_Hallmarks)

;; ------------------------------------------------------------
;; Cellular senescence
;; ------------------------------------------------------------
(: LopezOtin2023_DQ_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_DQ_Mouse CellularSenescence)
(EvidenceIntervention LopezOtin2023_DQ_Mouse DasatinibPlusQuercetin)
(EvidenceSpeciesModel LopezOtin2023_DQ_Mouse Mouse)
(EvidenceOutcomeText LopezOtin2023_DQ_Mouse "increased health- and lifespan")
(EvidenceReferenceNumber LopezOtin2023_DQ_Mouse 59)
(SupportedByPublication LopezOtin2023_DQ_Mouse LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_Fisetin_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_Fisetin_Mouse CellularSenescence)
(EvidenceIntervention LopezOtin2023_Fisetin_Mouse Fisetin)
(EvidenceSpeciesModel LopezOtin2023_Fisetin_Mouse Mouse)
(EvidenceOutcomeText LopezOtin2023_Fisetin_Mouse "increased health- and lifespan")
(EvidenceReferenceNumber LopezOtin2023_Fisetin_Mouse 60)
(SupportedByPublication LopezOtin2023_Fisetin_Mouse LopezOtinEtAl2023_Hallmarks)

(: LopezOtin2023_DQ_Human_DiabeticKidneyDisease HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_DQ_Human_DiabeticKidneyDisease CellularSenescence)
(EvidenceIntervention LopezOtin2023_DQ_Human_DiabeticKidneyDisease DasatinibPlusQuercetin)
(EvidenceSpeciesModel LopezOtin2023_DQ_Human_DiabeticKidneyDisease HumanDiabeticKidneyDisease)
(EvidenceOutcomeText LopezOtin2023_DQ_Human_DiabeticKidneyDisease "reduction of senescent cells and macrophages in adipose tissue, and of pro-inflammatory factors in serum")
(EvidenceReferenceNumber LopezOtin2023_DQ_Human_DiabeticKidneyDisease 62)
(SupportedByPublication LopezOtin2023_DQ_Human_DiabeticKidneyDisease LopezOtinEtAl2023_Hallmarks)

;; ------------------------------------------------------------
;; Chronic inflammation
;; ------------------------------------------------------------
(: LopezOtin2023_Canakinumab_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_Canakinumab_Human ChronicInflammation)
(EvidenceIntervention LopezOtin2023_Canakinumab_Human Canakinumab)
(EvidenceSpeciesModel LopezOtin2023_Canakinumab_Human HumanPriorMyocardialInfarctionHighHsCRP)
(EvidenceOutcomeText LopezOtin2023_Canakinumab_Human "reduced incidence of hypertension and diabetes; reduced frequency of recurrent myocardial infarction and non-small cell lung cancer")
(EvidenceReferenceNumber LopezOtin2023_Canakinumab_Human 87)
(SupportedByPublication LopezOtin2023_Canakinumab_Human LopezOtinEtAl2023_Hallmarks)

;; ------------------------------------------------------------
;; Dysbiosis
;; ------------------------------------------------------------
(: LopezOtin2023_Akkermansia_Human HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_Akkermansia_Human Dysbiosis)
(EvidenceIntervention LopezOtin2023_Akkermansia_Human AkkermansiaMuciniphila)
(EvidenceSpeciesModel LopezOtin2023_Akkermansia_Human HumanObeseOrDiabetic)
(EvidenceOutcomeText LopezOtin2023_Akkermansia_Human "improved metabolic parameters")
(EvidenceReferenceNumber LopezOtin2023_Akkermansia_Human 95)
(SupportedByPublication LopezOtin2023_Akkermansia_Human LopezOtinEtAl2023_Hallmarks)
