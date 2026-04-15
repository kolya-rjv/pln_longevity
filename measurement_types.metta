;; ==========================================
;; MEASUREMENT & PROTOCOL TYPES
;; ==========================================

;; Base metric wrapper
(: Measure (-> Number Unit Expression))

;; Sub-types for the Protocol
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
;; Enumerated Routes & Frequencies
;; ------------------------------------------
(: Food Route)
(: Water Route)
(: Intraperitoneal Route)
(: Subcutaneous Route)
(: OralGavage Route)

(: Daily Frequency)
(: Weekly Frequency)
(: AlternateDays Frequency)
(: Continuous Frequency)

;; ------------------------------------------
;; Biological Milestones
;; ------------------------------------------
(: UntilDeath DurationEvent)
(: AfterWeaning LifeStage)