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

;; Significance labels (enumerated values)
(: SignificanceLabel Type)
(: Significant SignificanceLabel)
(: NonSignificant SignificanceLabel)
(: Unreported SignificanceLabel)

;; Generic missing-value placeholder
(: MissingValue Type)
(: Unknown MissingValue)

;; ==========================================
;; 1. Biological and Clinical Entities
;; ==========================================
(: Biomarker Type)
(: EpigeneticClock Type)
(Inheritance EpigeneticClock Biomarker)

(: Pathway Type)                  ; e.g., mTOR pathway, Senescence
(: Outcome Type)                  ; e.g., AllCauseMortality, CardiovascularEvent
(: Demographic Type)              ; e.g., OlderAdult
(: Phenotype Type)
(: Lifespan Phenotype)
(: Healthspan Phenotype)

;; ==========================================
;; 2. Interventions
;; ==========================================
(: Intervention Type)
(Inheritance Pharmaceutical Intervention)
(Inheritance Supplement Intervention)
(Inheritance Lifestyle Intervention)        ; e.g., Exercise, Fasting

;; ==========================================
;; 3. Taxonomy Hierarchy
;; ==========================================
(: Taxon Type)
(Inheritance Vertebrate Taxon)
(Inheritance Invertebrate Taxon)
(Inheritance Fungi Taxon)
(Inheritance Protozoa Taxon)

;; ==========================================
;; 4. Sex (enumerated values)
;; ==========================================
(: Sex Type)
(: Male Sex)
(: Female Sex)
(: Mixed Sex)
(: Hermaphrodite Sex)

;; ==========================================
;; 5. Experimental & Protocol Context
;; ==========================================
(: TreatmentProtocol Type)
(: Strain Type)

;; NOTE: Measurement types (Measure, Unit, Route, Frequency,
;; LifeStage, DurationEvent) are defined in measurement_types.metta.
;; Do NOT duplicate them here.

;; ==========================================
;; 6. Epistemic and Provenance Types
;; ==========================================
(: EvidenceCategory Type)         ; Used for TV calibration mapping
(: SafetyCategory Type)
(: Publication Type)              ; e.g., PMID_19587680

;; ==========================================
;; 7. GENE TYPES (GenAge)
;; ==========================================

(: Gene Type)

;; Record types — parallel to Experiment, not subtypes of Gene
;; Connection to Gene goes through InvolvesGene predicate
(: GeneticExperiment Type)
(Inheritance GeneticExperiment Experiment)
(: HumanAgingGene Type)

(: SelectionBasis Type)
(: mammal     SelectionBasis)
(: model      SelectionBasis)
(: cell       SelectionBasis)
(: functional SelectionBasis)
(: human      SelectionBasis)
(: downstream SelectionBasis)
(: putative   SelectionBasis)
(: upstream   SelectionBasis)
(: human_link SelectionBasis)