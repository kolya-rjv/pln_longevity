;; ==========================================
;; MEASUREMENT & PROTOCOL TYPES
;; ==========================================
;; This file is the single authority for all measurement-related types.
;; Do not redeclare these in system_types.metta.

;; Base metric constructor
;; Usage: (Measure 14.0 ppm) -> Expression
(: Measure (-> Number Unit Expression))

;; ------------------------------------------
;; Type declarations
;; ------------------------------------------
(: Unit Type)
(: Route Type)
(: Frequency Type)
(: LifeStage Type)
(: DurationEvent Type)

;; ------------------------------------------
;; Enumerated Units (Based on DrugAge CSV)
;; ------------------------------------------
;; Concentrations
(: mM Unit) (: uM Unit) (: nM Unit) (: pM Unit)
(: mg_mL Unit) (: ug_mL Unit)
(: ppm Unit) (: percent Unit)

;; Mass/Dosing
(: mg_kg Unit) (: ug_g Unit) (: mg Unit)

;; Time
(: Months Unit) (: Weeks Unit) (: Days Unit)

;; ------------------------------------------
;; Enumerated Routes
;; ------------------------------------------
(: Food Route)
(: Water Route)
(: Intraperitoneal Route)
(: Subcutaneous Route)
(: OralGavage Route)

;; ------------------------------------------
;; Enumerated Frequencies
;; ------------------------------------------
(: Daily Frequency)
(: Weekly Frequency)
(: AlternateDays Frequency)
(: Continuous Frequency)

;; ------------------------------------------
;; Biological Milestones
;; ------------------------------------------
(: UntilDeath DurationEvent)
(: AfterWeaning LifeStage)