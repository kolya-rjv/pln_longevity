;; ==========================================
;; SYSTEM TYPES: The Base Noun Categories
;; ==========================================

;; Top-level entity type
(: Entity Type)

;; ==========================================
;; 0. Core record / ETL types
;; ==========================================
(: Experiment Type)
(: Target Type)
(: Expression Type)
(: EmpiricalAssociation Type)     ; paper-derived quantitative evidence record
(: HallmarkInterventionEvidence Type) ; review-level intervention evidence record

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

(: Pathway Type)
(: Outcome Type)
(: Demographic Type)
(: Phenotype Type)
(: Lifespan Phenotype)
(: Healthspan Phenotype)

;; ==========================================
;; 1a. GrimAge / Epigenetic Clock Types
;; ==========================================
(: EpigeneticAgeAcceleration Type)
(Inheritance EpigeneticAgeAcceleration Biomarker)

(: MortalityRiskEstimator Type)
(Inheritance MortalityRiskEstimator Biomarker)

(: Protein Type)

;; ==========================================
;; 1b. Hallmarks of Aging Types
;; ==========================================
(: HallmarkOfAging Type)

;; ==========================================
;; 2. Interventions
;; ==========================================
(: Intervention Type)
(Inheritance Pharmaceutical Intervention)
(Inheritance Supplement Intervention)
(Inheritance Lifestyle Intervention)
(Inheritance GeneticIntervention Intervention)
(Inheritance CombinationIntervention Intervention)

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
(: EvidenceCategory Type)
(: SafetyCategory Type)
(: Publication Type)

;; ==========================================
;; 7. GENE TYPES (GenAge)
;; ==========================================
(: Gene Type)
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
;; ==========================================
;; 8. CELLAGE TYPES
;; ==========================================

(: CellSenescenceRecord Type)
(Inheritance CellSenescenceRecord Experiment)

(: SenescenceExpressionRecord Type)
(Inheritance SenescenceExpressionRecord GeneticExperiment)

(: SenescenceType Type)
(: CellContext Type)
(: SenescenceEffectLabel Type)
(: ExpressionDirection Type)
