;; ==========================================
;; SYSTEM PREDICATES: The Fixed Verbs
;; ==========================================

;; ==========================================
;; 0. Instance-of (for ETL row -> type membership)
;; ==========================================
;; Use this instead of Inheritance for "X is an instance of Type".
;; This prevents PLN transitive deduction from chaining rows through
;; the type hierarchy.
(: InstanceOf (-> Atom Type Atom))

;; ==========================================
;; 1. Mechanistic Effect Wrappers
;; ==========================================
;; These are UNARY constructors that wrap a target to create a
;; mechanistic-effect expression. Used inside Causes:
;;   (Causes Rapamycin (Inhibits mTORC1))
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
(: Reflects       (-> Biomarker Atom Atom))      ;; biological interpretation
(: HazardRatio    (-> Biomarker Outcome Atom))

;; ==========================================
;; 4. Structural & Grouping Logic
;; ==========================================
;; Inheritance: taxonomic / subsumption links that PLN reasons over.
;; Do NOT use for row-to-type membership (use InstanceOf instead).
(: Inheritance (-> Atom Atom Atom))
(: PartOf      (-> Atom Atom Atom))
(: SetLink     (-> Expression Atom))
(: AndLink     (-> Expression Atom))

;; ==========================================
;; 4a. Hallmarks of Aging ontology
;; ==========================================
;; HallmarkComponent is a source-backed conceptual anchor. It is not
;; a statistical correlation and should not receive an invented STV.
(: HallmarkComponent   (-> Atom HallmarkOfAging Atom))
(: TargetsHallmark     (-> Intervention HallmarkOfAging Atom))
(: CandidateBridge     (-> Atom Atom String Atom))

;; Review-level intervention evidence records. These preserve the
;; wording and source reference from a review table without pretending
;; that the review itself reports a calibrated PLN truth value.
(: EvidenceHallmark        (-> HallmarkInterventionEvidence HallmarkOfAging Atom))
(: EvidenceIntervention    (-> HallmarkInterventionEvidence Intervention Atom))
(: EvidenceSpeciesModel    (-> HallmarkInterventionEvidence Atom Atom))
(: EvidenceOutcomeText     (-> HallmarkInterventionEvidence String Atom))
(: EvidenceReferenceNumber (-> HallmarkInterventionEvidence Number Atom))
(: EvidenceNote            (-> Atom String Atom))

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
(: Concentration (-> TreatmentProtocol Expression Atom))
(: DosePerMass (-> TreatmentProtocol Expression Atom))
(: DeliveryRoute (-> TreatmentProtocol Route Atom))
(: AdministrationFrequency (-> TreatmentProtocol Frequency Atom))
(: AgeAtInitiation (-> TreatmentProtocol Expression Atom))
(: AgeAtInitiation (-> TreatmentProtocol LifeStage Atom))
(: TreatmentDuration (-> TreatmentProtocol Expression Atom))
(: TreatmentDuration (-> TreatmentProtocol DurationEvent Atom))
(: ExtendsMaxLifespan (-> TreatmentProtocol Target Atom))
(: DoesNotExtendMaxLifespan (-> TreatmentProtocol Target Atom))
(: AltersBodyWeight (-> TreatmentProtocol Phenotype Atom))

;; ==========================================
;; 9. FUTURE PREDICATES (stubs for later phases)
;; ==========================================
;; (: EvidenceLevel     (-> Intervention Atom Atom))
;; (: SafetyProfile     (-> Intervention Atom Atom))
;; (: Implies           (-> Atom Atom Atom))
;; (: SimilarMechanism  (-> Intervention Intervention Atom Atom))
;; (: OutperformsFor    (-> Atom Atom Atom Atom))
;; (: SexSpecificEffect (-> Atom Atom Atom Atom))
;; (: Note              (-> Atom String Atom))

;; ==========================================
;; 10. GENE PREDICATES (GenAge)
;; ==========================================
(: InvolvesGene          (-> Atom Gene Atom))
(: HasLifespanEffect     (-> GeneticExperiment Expression Atom))
(: LifespanChangePercent (-> GeneticExperiment Number Atom))
(: HasSelectionBasis     (-> HumanAgingGene SelectionBasis Atom))
(: GeneSymbol            (-> Gene String Atom))
(: GeneName              (-> Gene String Atom))
(: EntrezID              (-> Gene Number Atom))
(: UniprotID             (-> Gene String Atom))

;; ==========================================
;; 11. BIOMARKER MODEL + PUBLICATION PREDICATES
;; ==========================================
;; Added for GrimAge curation. These are generic enough to reuse for
;; PhenoAge, DunedinPACE, other methylation clocks and extracted papers.
(: ModelInput             (-> Atom Atom Atom))
(: SurrogateFor           (-> Biomarker Biomarker Atom))
(: ResidualOf             (-> Biomarker Biomarker Atom))
(: AdjustmentCovariate    (-> Biomarker Atom Atom))
(: ValueEncoding          (-> Atom String Atom))
(: EncodedBy              (-> Protein Gene Atom))
(: HasCpGCount            (-> Biomarker Number Atom))
(: HasTotalUniqueCpGCount (-> Biomarker Number Atom))
(: SupportedByPublication (-> Atom Publication Atom))
(: PublicationTitle       (-> Publication String Atom))
(: PublicationYear        (-> Publication Number Atom))
(: JournalName            (-> Publication String Atom))
(: DOI                    (-> Publication String Atom))

;; ==========================================
;; 12. STRUCTURED EMPIRICAL EVIDENCE RECORDS
;; ==========================================
;; Keep exact paper measurements separate from calibrated PLN truth values.
;; A later calibration layer can transform these records into TVs.
(: HasPredictor   (-> EmpiricalAssociation Biomarker Atom))
(: HasOutcome     (-> EmpiricalAssociation Outcome Atom))
(: EffectMetric   (-> EmpiricalAssociation String Atom))
(: EffectValue    (-> EmpiricalAssociation Number Atom))
(: EffectScale    (-> EmpiricalAssociation String Atom))
(: PValue         (-> EmpiricalAssociation Number Atom))
