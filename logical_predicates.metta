;; ==========================================
;; SYSTEM PREDICATES: The Fixed Verbs
;; ==========================================

;; 1. Causal & Mechanistic Links (Used for counterfactuals/intervention ranking)
;; Signature: Origin -> Target -> STV
(: Causes (-> Entity Entity Atom))

;; Functional Wrappers (Used INSIDE Causes to give directionality for the math)
;; e.g., (Causes (Inhibits mTOR) (Decreases Senescence))
(: Increases (-> Entity Entity))
(: Decreases (-> Entity Entity))
(: Inhibits  (-> Entity Entity))
(: Activates (-> Entity Entity))
(: Clears    (-> Entity Entity))

;; 2. Empirical & Predictive Links (Used for diagnostics and risk calculation)
(: Predicts       (-> Biomarker Outcome Atom))
(: CorrelatedWith (-> Entity Entity Atom))
(: Reflects       (-> Biomarker Pathway Atom))  ;; Maps surrogate markers to biology
(: HazardRatio    (-> Biomarker Outcome Atom))

;; 3. Structural & Grouping Logic
(: Inheritance (-> Entity Type Atom))       ;; Connects discovered nouns to system_types
(: PartOf      (-> Entity Entity Atom))     ;; Composition (e.g., DNAmPAI1 PartOf GrimAge)
(: SetLink     (-> Expression Atom))        ;; Grouping multiple targets
(: AndLink     (-> Expression Atom))        ;; Logical AND

;; 4. Modifiers & Caveats
(: Interaction (-> Intervention Intervention String Atom))
(: Limitation  (-> Intervention String Atom))
(: EnhancedBy  (-> Intervention Entity Atom))


;; ==========================================
;; PROTOCOL PREDICATES (Strictly Typed)
;; ==========================================

;; Concentration in medium (e.g., 14 ppm in food)
(: Concentration (-> TreatmentProtocol Expression Atom))
;; Ex: (Concentration Protocol_1 (Measure 14.0 ppm))

;; Normalized bodyweight dose (e.g., 2.24 ppm/day)
(: DosePerMass (-> TreatmentProtocol Expression Atom))
;; Ex: (DosePerMass Protocol_1 (Measure 2.24 ppm))

;; Delivery Mechanics
(: DeliveryRoute (-> TreatmentProtocol Route Atom))
;; Ex: (DeliveryRoute Protocol_1 Food)

(: AdministrationFrequency (-> TreatmentProtocol Frequency Atom))
;; Ex: (AdministrationFrequency Protocol_1 Daily)

;; Temporal Dynamics (Age and Duration)
;; Can take either a precise Measure or a Milestone
(: AgeAtInitiation (-> TreatmentProtocol Expression Atom))
(: AgeAtInitiation (-> TreatmentProtocol LifeStage Atom))
;; Ex: (AgeAtInitiation Protocol_1 (Measure 20.0 Months))
;; Ex: (AgeAtInitiation Protocol_1 AfterWeaning)

(: TreatmentDuration (-> TreatmentProtocol Expression Atom))
(: TreatmentDuration (-> TreatmentProtocol DurationEvent Atom))
;; Ex: (TreatmentDuration Protocol_1 UntilDeath)
;; Ex: (TreatmentDuration Protocol_1 (Measure 3.0 Months))

;; Differentiates fundamental aging rate changes from general healthspan
(: ExtendsMaxLifespan (-> TreatmentProtocol Target Atom))
(: DoesNotExtendMaxLifespan (-> TreatmentProtocol Target Atom))

(: AltersBodyWeight (-> TreatmentProtocol Phenotype Atom))
;; Example usage in ETL if weight_change_significance == 'S':
;; ((AltersBodyWeight Exp_8f7b2c1 BodyWeight) (stv <normalized_percent> <confidence>))