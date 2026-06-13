;; hallmarks_core.cleaned.metta
;; Hallmarks of Aging ontology: the 12-hallmark 2023 update
;; Source: Lopez-Otin et al. (2023), "Hallmarks of aging: An expanding universe"
;; Cell 186, 243-278. DOI: 10.1016/j.cell.2022.11.001
;;
;; Load after:
;;   system_types.cleaned.v2.metta
;;   logical_predicates.cleaned.v2.metta
;;
;; This file intentionally contains only structural, paper-backed ontology.
;; Mechanism anchors are kept in hallmarks_lopezotin2023_anchors.metta.
;; Review-level intervention examples are kept in
;; hallmarks_lopezotin2023_intervention_evidence.metta.

;; ------------------------------------------------------------
;; 1. Publication provenance
;; ------------------------------------------------------------
(: LopezOtinEtAl2023_Hallmarks Publication)
(PublicationTitle LopezOtinEtAl2023_Hallmarks "Hallmarks of aging: An expanding universe")
(PublicationYear LopezOtinEtAl2023_Hallmarks 2023)
(JournalName LopezOtinEtAl2023_Hallmarks "Cell")
(DOI LopezOtinEtAl2023_Hallmarks "10.1016/j.cell.2022.11.001")

;; ------------------------------------------------------------
;; 2. Twelve hallmarks proposed in the 2023 update
;; ------------------------------------------------------------
(Inheritance GenomicInstability HallmarkOfAging)
(Inheritance TelomereAttrition HallmarkOfAging)
(Inheritance EpigeneticAlterations HallmarkOfAging)
(Inheritance LossOfProteostasis HallmarkOfAging)
(Inheritance DisabledMacroautophagy HallmarkOfAging)
(Inheritance DeregulatedNutrientSensing HallmarkOfAging)
(Inheritance MitochondrialDysfunction HallmarkOfAging)
(Inheritance CellularSenescence HallmarkOfAging)
(Inheritance StemCellExhaustion HallmarkOfAging)
(Inheritance AlteredIntercellularCommunication HallmarkOfAging)
(Inheritance ChronicInflammation HallmarkOfAging)
(Inheritance Dysbiosis HallmarkOfAging)

;; ------------------------------------------------------------
;; 3. Atom-level provenance
;; ------------------------------------------------------------
(SupportedByPublication GenomicInstability LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication TelomereAttrition LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication EpigeneticAlterations LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication LossOfProteostasis LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication DisabledMacroautophagy LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication DeregulatedNutrientSensing LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication MitochondrialDysfunction LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication CellularSenescence LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication StemCellExhaustion LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication AlteredIntercellularCommunication LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication ChronicInflammation LopezOtinEtAl2023_Hallmarks)
(SupportedByPublication Dysbiosis LopezOtinEtAl2023_Hallmarks)
