;; ==========================================
;; SYSTEM TYPES: The Base Noun Categories
;; ==========================================

;; Top-level entity type
(: Entity Type)

;; ==========================================
;; 0. Core record / ETL types
;; ==========================================

(: Experiment Type)              ; one imported DrugAge row / study record
(: Target Type)                  ; used by ExtendsMaxLifespan / DoesNotExtendMaxLifespan
(: Expression Type)              ; used in signatures like Measure / SetLink / AndLink

;; Optional but strongly recommended
(: SignificanceLabel Type)       ; e.g., Significant, NonSignificant, Unreported
(: MissingValue Type)            ; for ETL placeholders like Unreported / Unknown

;; ==========================================
;; 1. Biological and Clinical Entities
;; ==========================================
(: Biomarker Type)
(: EpigeneticClock Biomarker)
(: Pathway Type)                  ; e.g., mTOR pathway, Senescence
(: Outcome Type)                  ; e.g., AllCauseMortality, CardiovascularEvent
(: Demographic Type)              ; e.g., Male, OlderAdult
(: Phenotype Type)                ; e.g., Lifespan, Healthspan

;; ==========================================
;; 2. Interventions
;; ==========================================
(: Intervention Type)
(: Pharmaceutical Intervention)
(: Supplement Intervention)
(: Lifestyle Intervention)        ; e.g., Exercise, Fasting

;; ==========================================
;; 3. Taxonomy Hierarchy
;; ==========================================
(: Taxon Type)
(: Vertebrate Taxon)
(: Invertebrate Taxon)
(: Fungi Taxon)
(: Protozoa Taxon)

;; Make Species inherit from Taxon in the type system
(: Species Taxon)
(: Sex Type)

;; ==========================================
;; 4. Experimental & Protocol Context
;; ==========================================
(: TreatmentProtocol Type)
(: Strain Type)

;; Measurement wrappers
(: Measure Type)                  ; Used for (Measure <Number> <Unit>)
(: Unit Type)
(: Route Type)
(: Frequency Type)
(: LifeStage Type)
(: DurationEvent Type)

;; ==========================================
;; 5. Epistemic and Provenance Types
;; ==========================================
(: EvidenceCategory Type)         ; Used for TV calibration mapping
(: SafetyCategory Type)
(: Publication Type)              ; Allows linking protocols to specific papers (e.g., PMID_19587680)

