;; ==========================================
;; SYSTEM PREDICATES: The Fixed Verbs
;; ==========================================

;; ==========================================
;; 0. Instance-of (for ETL row → type membership)
;; ==========================================
;; Use this instead of Inheritance for "X is an instance of Type".
;; This prevents PLN transitive deduction from chaining rows through
;; the type hierarchy.
(: InstanceOf (-> Atom Type Atom))

;; ==========================================
;; 1. Mechanistic Effect Wrappers
;; ==========================================
;; These are UNARY constructors that wrap a target to create a
;; mechanistic-effect expression.  Used inside Causes:
;;   (Causes Rapamycin (Inhibits mTORC1) (stv 0.95 0.98))
;;
;; (Inhibits mTORC1) : Expression

(: Increases (-> Atom Expression))
(: Decreases (-> Atom Expression))
(: Inhibits  (-> Atom Expression))
(: Activates (-> Atom Expression))
(: Clears    (-> Atom Expression))
(: Improves  (-> Atom Expression))
(: Induces   (-> Atom Expression))

;; ==========================================
;; 2. Causal & Mechanistic Links
;; ==========================================
;; Both arguments are Atom so they accept bare entities OR wrapped
;; expressions like (Inhibits mTORC1).
(: Causes (-> Atom Atom Atom))

;; ==========================================
;; 3. Empirical & Predictive Links
;; ==========================================
(: Predicts       (-> Biomarker Outcome Atom))
(: CorrelatedWith (-> Atom Atom Atom))
(: Reflects       (-> Biomarker Atom Atom))      ;; Maps surrogate markers to biology
(: HazardRatio    (-> Biomarker Outcome Atom))

;; ==========================================
;; 4. Structural & Grouping Logic
;; ==========================================
;; Inheritance: taxonomic / subsumption links that PLN reasons over.
;; Do NOT use for row-to-type membership (use InstanceOf instead).
(: Inheritance (-> Atom Atom Atom))

(: PartOf      (-> Atom Atom Atom))        ;; Composition (e.g., DNAmPAI1 PartOf GrimAge)
(: SetLink     (-> Expression Atom))       ;; Grouping multiple targets
(: AndLink     (-> Expression Atom))       ;; Logical AND

;; ==========================================
;; 5. Modifiers & Caveats
;; ==========================================
(: Interaction (-> Intervention Intervention String Atom))
(: Limitation  (-> Intervention String Atom))
(: EnhancedBy  (-> Intervention Atom Atom))

;; ==========================================
;; 6. Lifespan Extension (for DrugAge / ITP data)
;; ==========================================
;; Strength near 0 = negative result; strength near 1 = strong extension.
(: Extends (-> Intervention Atom Atom))

;; ==========================================
;; 7. EXPERIMENT / DATASET RECORD PREDICATES
;; ==========================================

(: UsesIntervention            (-> Experiment Intervention Atom))
(: UsesSpecies                 (-> Experiment Taxon Atom))
(: UsesStrain                  (-> Experiment Strain Atom))
(: HasSex                      (-> Experiment Sex Atom))
(: ReportedIn                  (-> Experiment Publication Atom))

(: Dosage                      (-> Experiment Expression Atom))

(: AvgLifespanChangePercent    (-> Experiment Number Atom))
(: AvgLifespanSignificance     (-> Experiment SignificanceLabel Atom))
(: MaxLifespanSignificance     (-> Experiment SignificanceLabel Atom))
(: WeightChangeSignificance    (-> Experiment SignificanceLabel Atom))

;; ==========================================
;; 8. PROTOCOL PREDICATES (Strictly Typed)
;; ==========================================

;; Concentration in medium (e.g., 14 ppm in food)
(: Concentration (-> TreatmentProtocol Expression Atom))

;; Normalized bodyweight dose (e.g., 2.24 ppm/day)
(: DosePerMass (-> TreatmentProtocol Expression Atom))

;; Delivery Mechanics
(: DeliveryRoute (-> TreatmentProtocol Route Atom))
(: AdministrationFrequency (-> TreatmentProtocol Frequency Atom))

;; Temporal Dynamics — polymorphic via two declarations
(: AgeAtInitiation (-> TreatmentProtocol Expression Atom))
(: AgeAtInitiation (-> TreatmentProtocol LifeStage Atom))

(: TreatmentDuration (-> TreatmentProtocol Expression Atom))
(: TreatmentDuration (-> TreatmentProtocol DurationEvent Atom))

;; Lifespan phenotype flags
(: ExtendsMaxLifespan (-> TreatmentProtocol Target Atom))
(: DoesNotExtendMaxLifespan (-> TreatmentProtocol Target Atom))

(: AltersBodyWeight (-> TreatmentProtocol Phenotype Atom))

;; ==========================================
;; 9. FUTURE PREDICATES (stubs for later phases)
;; ==========================================
;; Uncomment as you implement the corresponding pipeline phases.
;;
;; (: EvidenceLevel    (-> Intervention Atom Atom))
;; (: SafetyProfile    (-> Intervention Atom Atom))
;; (: Implies          (-> Atom Atom Atom))
;; (: SimilarMechanism (-> Intervention Intervention Atom Atom))
;; (: OutperformsFor   (-> Atom Atom Atom Atom))
;; (: SexSpecificEffect (-> Atom Atom Atom Atom))
;; (: Note             (-> Atom String Atom))

;; ==========================================
;; 10. GENE PREDICATES (GenAge)
;; ==========================================

;; Atom as first arg — applies to both GeneticExperiment and HumanAgingGene
(: InvolvesGene          (-> Atom Gene Atom))

;; GeneticExperiment predicates
;; HasLifespanEffect takes (Increases Lifespan) or (Decreases Lifespan)
(: HasLifespanEffect     (-> GeneticExperiment Expression Atom))
(: LifespanChangePercent (-> GeneticExperiment Number Atom))

;; HumanAgingGene predicates
(: HasSelectionBasis     (-> HumanAgingGene SelectionBasis Atom))

;; Gene entity predicates
(: GeneSymbol            (-> Gene String Atom))
(: GeneName              (-> Gene String Atom))
(: EntrezID              (-> Gene Number Atom))
(: UniprotID             (-> Gene String Atom))