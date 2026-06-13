# De4F25Ee-Pln Longevity Pipeline


Total Pages: 21


## Page 1

Probabilistic Logic Networks for Personalized
Longevity Biomarker Analysis
A Pipeline for Demonstrating PLN Capabilities Beyond LLMs
Ben Goertzel w/ help from Claude Opus 4.5
January 31, 2026
Abstract
This document describes a pipeline for using Probabilistic Logic Networks (PLN) to
perform personalized longevity biomarker analysis. The system integrates knowledge ex-
tracted from aging research literature—including both pharmaceutical interventions and
supplements/nutraceuticals—with empirical data from methylation databases to provide
uncertainty-quantiﬁed, causally-grounded recommendations. We detail the architecture,
data sources, and demonstrate capabilities that fundamentally exceed what Large Language
Models can provide: genuine uncertainty propagation, auditable inference chains, counter-
factual reasoning, novel cross-source discovery, and calibrated supplement recommendations
that respect the weaker evidence base while still providing useful guidance.
Contents
1 Introduction and Motivation 3
1.1 The Problem with LLM-Based Biomarker Interpretation . . . . . . . . . . . . . . 3
1.2 What PLN Provides . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2 System Architecture 3
2.1 Three-Layer Knowledge Representation . . . . . . . . . . . . . . . . . . . . . . . 3
2.1.1 Layer 1: Ontological Structure . . . . . . . . . . . . . . . . . . . . . . . . 3
2.1.2 Layer 2: Empirical Relations . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.1.3 Layer 3: Mechanistic/Causal Pathways . . . . . . . . . . . . . . . . . . . . 4
2.1.4 Layer 4: Supplement and Nutraceutical Knowledge . . . . . . . . . . . . . 4
2.2 Data Flow Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3 Data Sources for Pipeline Construction 5
3.1 Primary Methylation Database: NHANES . . . . . . . . . . . . . . . . . . . . . . 5
3.2 Secondary Methylation Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2.1 Biolearn Library . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2.2 HALL Database . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.3 Knowledge Extraction Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.3.1 DrugAge Database . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.3.2 GenAge Database . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.3.3 CellAge Database . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
1

## Page 2

Technical Overview PLN Longevity Biomarker Pipeline
4 Demonstration Scenarios 6
4.1 Demo 1: Abductive Diagnosis with Ranked Hypotheses . . . . . . . . . . . . . . 6
4.2 Demo 2: Intervention Ranking with Uncertainty Propagation . . . . . . . . . . . 7
4.3 Demo 3: Novel Cross-Source Inference . . . . . . . . . . . . . . . . . . . . . . . . 7
4.4 Demo 4: Counterfactual Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
4.5 Demo 5: Risk Prediction with Full Uncertainty . . . . . . . . . . . . . . . . . . . 8
4.6 Demo 6: Personalized Supplement Recommendations . . . . . . . . . . . . . . . . 8
5 Implementation Roadmap 9
5.1 Phase 1: Knowledge Base Construction (Weeks 1–4) . . . . . . . . . . . . . . . . 9
5.2 Phase 2: LLM Extraction Pipeline (Weeks 3–6) . . . . . . . . . . . . . . . . . . . 9
5.3 Phase 3: PLN Inference Patterns (Weeks 5–8) . . . . . . . . . . . . . . . . . . . . 9
5.4 Phase 4: Demonstration and Comparison (Weeks 7–10) . . . . . . . . . . . . . . 9
6 Evaluation Metrics 10
A Relationship Types for NHANES Methylation Demo 11
A.1 Clock Prediction Relationships . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
A.2 Clock Component Relationships . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
A.3 Clock–Biomarker Correlations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
A.4 Clock Discordance Interpretations . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
A.5 Intervention Eﬀects on Clocks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
A.6 Mechanistic Pathways Connecting to Clocks . . . . . . . . . . . . . . . . . . . . . 12
A.7 Demographic and Lifestyle Modiﬁers . . . . . . . . . . . . . . . . . . . . . . . . . 13
A.8 Risk Quantiﬁcation Relationships . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
A.9 Summary: Priority Extraction Targets . . . . . . . . . . . . . . . . . . . . . . . . 13
B Open-Access Sources for Supplement Knowledge Extraction 14
B.1 Primary Databases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
B.1.1 DrugAge Database (Supplements Section) . . . . . . . . . . . . . . . . . . 14
B.1.2 Geroprotectors.org . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
B.1.3 LIFESPAN.io Supplement Database . . . . . . . . . . . . . . . . . . . . . 14
B.2 Review Articles and Meta-Analyses (Open Access) . . . . . . . . . . . . . . . . . 14
B.2.1 NAD+ Precursors (NMN, NR) . . . . . . . . . . . . . . . . . . . . . . . . 14
B.2.2 Senolytics and Senomorphics (Fisetin, Quercetin) . . . . . . . . . . . . . . 15
B.2.3 Berberine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
B.2.4 Spermidine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.2.5 Omega-3 Fatty Acids . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.2.6 Curcumin . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
B.2.7 Taurine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
B.2.8 Glycine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
B.2.9 Alpha-Lipoic Acid and CoQ10 . . . . . . . . . . . . . . . . . . . . . . . . . 18
B.3 Negative Evidence Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
B.3.1 NIA Interventions Testing Program (ITP) . . . . . . . . . . . . . . . . . . 18
B.3.2 Cochrane Reviews . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.4 Preprint Servers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.5 Structured Extraction Targets by Compound . . . . . . . . . . . . . . . . . . . . 19
B.6 Extraction Protocol for Supplements . . . . . . . . . . . . . . . . . . . . . . . . . 19
C Key Source Papers for Initial Curation 20
C.1 Pharmaceutical and Mechanistic Sources . . . . . . . . . . . . . . . . . . . . . . . 20
C.2 Supplement and Nutraceutical Sources . . . . . . . . . . . . . . . . . . . . . . . . 20
Page 2

## Page 3

Technical Overview PLN Longevity Biomarker Pipeline
1 Introduction and Motivation
1.1 The Problem with LLM-Based Biomarker Interpretation
Large Language Models can discuss aging biomarkers ﬂuently but suﬀer from fundamental lim-
itations:
(i)No genuine uncertainty quantiﬁcation : LLMs produce hedged language (“may,”
“might,” “somestudiessuggest”)butcannotpropagateuncertaintythroughinferencechains.
(ii)No principled reasoning over structured data : LLMs cannot systematically query
databases or compute over patient-speciﬁc numerical values.
(iii)No novel inference : LLMs retrieve and remix training data; they cannot discover gen-
uinely new implications by chaining facts from multiple sources.
(iv)No auditability : LLM reasoning is opaque; conclusions cannot be traced to speciﬁc
evidence with quantiﬁed conﬁdence.
1.2 What PLN Provides
PLN addresses each limitation:
Truth values (s;c)wheres2[0;1]is strength and c2[0;1]is conﬁdence, propagated
through every inference step.
Atomspace integration enabling queries over structured biomarker databases.
Inference rules (deduction, induction, abduction) that derive new knowledge with ap-
propriately reduced conﬁdence.
Complete provenance : every conclusion traces to source atoms.
2 System Architecture
2.1 Three-Layer Knowledge Representation
The Atomspace contains three complementary knowledge types:
2.1.1 Layer 1: Ontological Structure
Taxonomic relationships providing the conceptual backbone:
1; Hallmarks of Aging hierarchy
2( Inheritance CellularSenescence HallmarkOfAging )
3( Inheritance MitochondrialDysfunction HallmarkOfAging )
4( Inheritance GenomicInstability HallmarkOfAging )
5
6; Biomarker ontology
7( Inheritance GrimAge EpigeneticClock )
8( Inheritance HorvathClock EpigeneticClock )
9( PartOf DNAmPAI1 GrimAge )
10( PartOf DNAmGDF15 GrimAge )
11
12; Intervention classes
13( Inheritance Rapamycin mTORInhibitor )
14( Inheritance DasatinibPlusQuercetin Senolytic )
Page 3

## Page 4

Technical Overview PLN Longevity Biomarker Pipeline
2.1.2 Layer 2: Empirical Relations
Statistical associations derived from population studies (NHANES, Framingham, etc.):
1; Predictive relationships with evidence strength
2( Predicts GrimAge AllCauseMortality (stv 0.85 0.92) )
3( Predicts AgeAccelGrim CardiovascularEvent (stv 0.75 0.85) )
4
5; Correlations from NHANES
6( CorrelatedWith ( Elevated CRP) ( Elevated GrimAge )
7 (stv 0.65 0.92) )
8( CorrelatedWith ( Elevated DNAmPAI1 )
9 ( IncreasedRisk Thrombosis ) (stv 0.60 0.85) )
2.1.3 Layer 3: Mechanistic/Causal Pathways
Causal chains extracted from experimental literature:
1; Intervention mechanisms
2( Causes Rapamycin ( Inhibits mTORC1 ) (stv 0.95 0.98) )
3( Causes ( Inhibits mTORC1 ) ( Increases Autophagy ) (stv 0.88 0.92) )
4( Causes ( Increases Autophagy ) ( Decreases SenescentCellBurden )
5 (stv 0.60 0.65) )
6
7; Pathological cascades
8( Causes CellularSenescence SASP (stv 0.90 0.95) )
9( Causes SASP ChronicInflammation (stv 0.85 0.90) )
10( Causes ChronicInflammation ( Increases GrimAgeAcceleration )
11 (stv 0.70 0.75) )
2.1.4 Layer 4: Supplement and Nutraceutical Knowledge
A fourth layer captures evidence about supplements, nutraceuticals, and natural compounds
with appropriately calibrated conﬁdence reﬂecting the typically weaker evidence base:
1; Supplement mechanisms ( note lower confidence than pharmaceuticals )
2( Causes NMN ( Increases NADplus ) (stv 0.85 0.80) )
3( Causes ( Increases NADplus ) ( Improves MitochondrialFunction )
4 (stv 0.70 0.65) )
5( EvidenceLevel NMN PreliminaryHuman (stv 0.60 0.55) )
6( SafetyProfile NMN GenerallyWellTolerated (stv 0.85 0.80) )
7
8( Causes Fisetin ( Clears SenescentCells ) (stv 0.65 0.55) )
9( EvidenceLevel Fisetin AnimalStudies_EarlyHuman ( stv 0.50 0.60) )
10
11( Causes Berberine ( Activates AMPK ) (stv 0.80 0.75) )
12( EvidenceLevel Berberine MultipleHumanTrials (stv 0.75 0.80) )
13( Interaction Berberine Metformin " Similar mechanisms ")
14
15; Negative evidence is also captured
16( Extends Resveratrol Lifespan (stv 0.15 0.75) )
17( EvidenceLevel Resveratrol ITP_Negative (stv 0.35 0.85) )
Key distinctions from pharmaceutical knowledge:
Lower conﬁdence values reﬂecting weaker evidence
Explicit EvidenceLevel atoms categorizing research quality
SafetyProfile atoms for risk assessment
Interaction atoms for drug-supplement concerns
Negative evidence explicitly represented (e.g., resveratrol ITP failure)
Page 4

## Page 5

Technical Overview PLN Longevity Biomarker Pipeline
2.2 Data Flow Architecture
The pipeline follows this data ﬂow:
1.Research Papers (PubMed, PMC) !Semantic Parsing (LLM-assisted) !Atomspace
2.Databases (NHANES, GEO) !Schema Mapping & Import !Atomspace
3.PatientData (Methylation, BloodPanel) !ProﬁleCreation&Grounding !Atomspace
4.Atomspace!PLN Inference Engine !Results with Uncertainty & Provenance
3 Data Sources for Pipeline Construction
3.1 Primary Methylation Database: NHANES
Attribute Details
URL https://wwwn.cdc.gov/nchs/nhanes/dnam/
Cycles 1999–2002
Platform Illumina EPIC arrays
Sample Size4,000 individuals
Pre-computed Chronological age, phenotypic age,
telomere length, pace of aging, mortality risk
Access Public download, no application required
Table 1: NHANES DNA Methylation Dataset
Key advantage : NHANES links methylation to extensive health outcomes, demographics,
and longitudinal mortality data—enabling validation of PLN predictions against actual out-
comes.
3.2 Secondary Methylation Sources
3.2.1 Biolearn Library
URL: https://bio-learn.github.io
Harmonizes GEO, NHANES, Framingham Heart Study
39 standardized biomarkers, 200,000+ samples
Pre-computed clocks: Horvath, Hannum, GrimAge, GrimAge2, PhenoAge, DunedinPACE
Key GEO accessions: GSE40279, GSE19711, GSE51057, GSE42861, GSE41169
3.2.2 HALL Database
URL: https://academic.oup.com/nar/article/52/D1/D909/7327078
11,256 aging features across 38 tissue types
Cohorts: UK Biobank, CLHLS (China), BLSA (USA)
8 omics data types
Page 5

## Page 6

Technical Overview PLN Longevity Biomarker Pipeline
3.3 Knowledge Extraction Sources
3.3.1 DrugAge Database
URL: https://genomics.senescence.info/drugs/
1,097 compounds tested for lifespan extension
Structured: compound, species, eﬀect size, mechanism, targets
Directly importable to Atomspace
3.3.2 GenAge Database
URL: https://genomics.senescence.info/genes/
307 human aging-related genes
2,205 model organism longevity genes
Includes protein-protein interactions
3.3.3 CellAge Database
URL: https://genomics.senescence.info/cells/
866 genes associated with cellular senescence
Curated senescence markers (p16, p21, SA- -gal, etc.)
4 Demonstration Scenarios
To showcase PLN capabilities that LLMs cannot replicate, we propose the following demonstra-
tion scenarios:
4.1 Demo 1: Abductive Diagnosis with Ranked Hypotheses
Input: Patient methylation proﬁle showing discordant clock readings (e.g., normal Horvath,
elevated GrimAge) plus inﬂammatory markers.
PLN Task : Generate ranked hypotheses for upstream causes with quantiﬁed probabilities.
Why LLMs fail : LLMs can list possible causes but cannot:
Assign calibrated probabilities to each hypothesis
Update probabilities based on speciﬁc marker values
Show how evidence supports/refutes each hypothesis quantitatively
Expected Output :
1( RankedHypotheses Patient001
2 ( Hypothesis CellularSenescence ( stv 0.78 0.72)
3 ( SupportedBy DNAmPAI1_elevated DNAmGDF15_elevated CRP_elevated ))
4 ( Hypothesis MetabolicDysregulation (stv 0.62 0.65)
5 ( SupportedBy FastingGlucose_elevated HbA1c_elevated ))
6 ( Hypothesis MitochondrialDysfunction (stv 0.55 0.50)
7 ( WeaklySupportedBy DNAmGDF15_elevated )))
Page 6

## Page 7

Technical Overview PLN Longevity Biomarker Pipeline
4.2 Demo 2: Intervention Ranking with Uncertainty Propagation
Input: Patient proﬁle + candidate interventions (rapamycin, metformin, senolytics, acarbose).
PLN Task : Rank interventions by expected eﬀect on biological age acceleration, showing
complete inference chains.
Why LLMs fail : LLMs cannot:
Propagate uncertainty through multi-step causal chains
Combine patient-speciﬁc factors with population-level evidence
Provide conﬁdence intervals on predictions
Key demonstration : Show the actual computation:
P(GrimAge reduction jRapamycin ) =nY
i=1sif(patient factors )
wheresiare the strength values along the causal chain.
4.3 Demo 3: Novel Cross-Source Inference
Input: Knowledge base constructed from multiple papers that individually contain:
Paper A: Rapamycin inhibits mTORC1
Paper B: mTORC1 inhibition enhances mitophagy
Paper C: Enhanced mitophagy improves mitochondrial function
Paper D: Mitochondrial dysfunction elevates GDF15
Paper E: DNAmGDF15 is a component of GrimAge
PLNTask : Derivethenovelinference: “RapamycinmayreduceGrimAgeviatheDNAmGDF15
component through mitochondrial improvement”—a conclusion not stated in any single paper.
Why LLMs fail : LLMs may hallucinate such connections without principled conﬁdence
reduction. PLN provides:
1( NovelInference
2 ( Chain Rapamycin -> mTORC1_inhibition -> Mitophagy ->
3 MitoFunction -> GDF15_reduction -> GrimAge_reduction )
4 ( Confidence 0.27) ; appropriately reduced through 5- step chain
5 ( Sources Paper_A Paper_B Paper_C Paper_D Paper_E ))
4.4 Demo 4: Counterfactual Analysis
Input: PatientwithelevatedGrimAge(+6.7years)andelevatedinﬂammation(CRP4.2mg/L).
Query: “If this patient’s CRP were reduced to normal (0.8 mg/L), what would be the
expected change in GrimAge?”
PLN Task : Perform counterfactual reasoning using causal models to decompose GrimAge
into attributable components.
Why LLMs fail : LLMs cannot perform formal counterfactual inference or quantify the
expected eﬀect of hypothetical interventions based on causal structure.
Page 7

## Page 8

Technical Overview PLN Longevity Biomarker Pipeline
4.5 Demo 5: Risk Prediction with Full Uncertainty
Input: Patient proﬁle with multiple biomarkers.
PLN Task : Compute 10-year cardiovascular risk with:
Point estimate
Conﬁdence interval (propagated from all source uncertainties)
Decomposition into contributing factors
Projected risk under various intervention scenarios
Output format :
P(CV event in 10y ) = 0:315 [0:22;0:43]95% (c= 0:70)
4.6 Demo 6: Personalized Supplement Recommendations
Input: Patient proﬁle + user preference statement: “I’m interested in supplements that might
help with healthspan, even if evidence is preliminary. I’ll discuss with my doctor.”
PLN Task :
1. Encode user risk tolerance and preferences
2. Compute pathway activation scores from patient biomarkers
3. Match supplements to activated pathways
4. Rank by: relevance mechanism strength safety
5. Generate tiered recommendations with explicit uncertainty
6. Flag contraindications and interactions
Why LLMs fail : LLMs typically:
Provide generic supplement lists not personalized to biomarker proﬁle
Overstate supplement evidence (treating preliminary studies like RCTs)
Miss drug-supplement interactions
Cannot link recommendations to speciﬁc NHANES variables
May recommend compounds with negative trial results (e.g., resveratrol)
Expected Output :
1( SupplementRecommendation Patient001
2 ( Tier1_HighConfidence
3 ( Omega3 (stv 0.45 0.75) " Targets elevated LBXCRP ")
4 ( Berberine ( stv 0.38 0.70) " Targets elevated LBXGLU "))
5 ( Tier2_Promising
6 ( Fisetin (stv 0.34 0.50) " Targets DNAM_PAI1 elevation ")
7 (NMN (stv 0.39 0.55) " Targets DNAM_GDF15 elevation "))
8 ( NotRecommended
9 ( Resveratrol " ITP negative ; weak human data "))
10 ( Interactions
11 ( Berberine Metformin " Similar mechanisms - consult MD ")))
Key demonstration : The conﬁdence values for supplements (0.35–0.55) are appropriately
lowerthan for pharmaceuticals (0.55–0.70), reﬂecting the weaker evidence base. PLN maintains
intellectual honesty while still providing useful guidance.
Page 8

## Page 9

Technical Overview PLN Longevity Biomarker Pipeline
5 Implementation Roadmap
5.1 Phase 1: Knowledge Base Construction (Weeks 1–4)
1.Manual curation : Extract 50–100 high-conﬁdence relations from landmark papers:
Hallmarks of Aging (2013, 2023)
GrimAge (Lu et al., 2019)
DrugAge database paper (2017)
ITP publications
2.Database import : Convert DrugAge, GenAge, CellAge to MeTTa atoms.
3.NHANES integration : Import methylation clock values with health outcomes.
4.Supplement knowledge : Extract relations from open-access reviews for top 10 com-
pounds (see Appendix B priorities).
5.2 Phase 2: LLM Extraction Pipeline (Weeks 3–6)
1. Develop extraction prompt (see Appendix A)
2. Validate against manual curation
3. Scale to additional papers
4. Implement entity normalization and deduplication
5.Add supplement-speciﬁc extraction : Include evidence level classiﬁcation, safety pro-
ﬁles, and interaction detection
5.3 Phase 3: PLN Inference Patterns (Weeks 5–8)
1. Implement abductive diagnosis pattern
2. Implement intervention ranking with uncertainty propagation
3. Implement counterfactual reasoning
4.Implement supplement matching : Pathway activation scores, relevance computation,
tiered recommendations
5.Implement negative evidence handling : Ensure ITP-negative compounds are appro-
priately down-weighted
6. Test on synthetic patient proﬁles
5.4 Phase 4: Demonstration and Comparison (Weeks 7–10)
1. Create 5–10 demonstration cases
2. Run identical queries through GPT-4/Claude
3. Document diﬀerences in output quality
4.Speciﬁc supplement comparison : Show how PLN avoids recommending resveratrol
(ITP-negative) while LLMs often include it
5. Prepare presentation materials
Page 9

## Page 10

Technical Overview PLN Longevity Biomarker Pipeline
6 Evaluation Metrics
To rigorously demonstrate PLN advantages, we propose the following evaluation criteria:
Criterion PLN LLM
Provides numeric conﬁdence X
Conﬁdence calibrated to evidence quality X
Identical inputs!identical outputs X
Complete inference trace X
Citations veriﬁable XOften hallucinated
Novel inferences ﬂagged X
Uncertainty increases with chain length X
Integrates patient-speciﬁc data formally X Ad hoc
Distinguishes pharma vs. supplement evidence XOften conﬂates
Captures negative trial results X Often omits
Drug-supplement interactions ﬂagged X Inconsistent
Links to speciﬁc NHANES variables X
Table 2: PLN vs. LLM Capability Comparison
Page 10

## Page 11

Technical Overview PLN Longevity Biomarker Pipeline
A Relationship Types for NHANES Methylation Demo
The following relationship types should be prioritized for extraction from research papers to
enable rich inference over NHANES methylation data.
A.1 Clock Prediction Relationships
These enable reasoning about what elevated/reduced clock values imply:
1; Primary predictions
2( Predicts <Clock > <Outcome > ( stv S C))
3; Examples :
4( Predicts GrimAge AllCauseMortality (stv 0.85 0.92) )
5( Predicts GrimAge CardiovascularMortality (stv 0.80 0.90) )
6( Predicts GrimAge CancerIncidence (stv 0.65 0.88) )
7( Predicts PhenoAge AllCauseMortality (stv 0.75 0.88) )
8( Predicts DunedinPACE FutureHealthDecline (stv 0.80 0.85) )
9
10; Comparative predictive power
11( OutperformsFor GrimAge HorvathClock MortalityPrediction (stv 0.80 0.90) )
12( OutperformsFor DunedinPACE GrimAge ShortTermRiskPrediction (stv 0.65 0.70) )
A.2 Clock Component Relationships
Essential for understanding whya clock is elevated:
1; GrimAge components
2( PartOf DNAmPAI1 GrimAge )
3( PartOf DNAmGDF15 GrimAge )
4( PartOf DNAmB2M GrimAge )
5( PartOf DNAmCystatinC GrimAge )
6( PartOf DNAmLeptin GrimAge )
7( PartOf DNAmADM GrimAge )
8( PartOf DNAmTIMP1 GrimAge )
9( PartOf DNAmPackYears GrimAge )
10
11; What components reflect
12( Reflects DNAmPAI1 ( SetLink Thrombosis Fibrosis CellularSenescence )
13 (stv 0.75 0.70) )
14( Reflects DNAmGDF15 ( SetLink MitochondrialStress Inflammation )
15 (stv 0.70 0.65) )
16( Reflects DNAmB2M ImmuneActivation (stv 0.70 0.75) )
17( Reflects DNAmCystatinC KidneyFunction (stv 0.75 0.75) )
18( Reflects DNAmPackYears CumulativeSmokingExposure ( stv 0.85 0.90) )
A.3 Clock–Biomarker Correlations
Link methylation clocks to standard blood markers available in NHANES:
1; Inflammatory markers
2( CorrelatedWith ( Elevated GrimAge ) ( Elevated CRP) (stv 0.65 0.92) )
3( CorrelatedWith ( Elevated GrimAge ) ( Elevated IL6) (stv 0.60 0.88) )
4( CorrelatedWith ( Elevated PhenoAge ) ( Elevated CRP) (stv 0.55 0.85) )
5
6; Metabolic markers
7( CorrelatedWith ( Elevated GrimAge ) ( Elevated FastingGlucose ) (stv 0.50 0.80)
)
8( CorrelatedWith ( Elevated GrimAge ) ( Elevated HbA1c ) (stv 0.55 0.82) )
9( CorrelatedWith ( Elevated PhenoAge ) ( Elevated Triglycerides ) (stv 0.45 0.78)
)
10
Page 11

## Page 12

Technical Overview PLN Longevity Biomarker Pipeline
11; Renal markers
12( CorrelatedWith ( Elevated DNAmCystatinC ) ( Decreased eGFR ) (stv 0.70 0.85) )
A.4 Clock Discordance Interpretations
Critical for personalized analysis—diﬀerent clocks capture diﬀerent biology:
1; Discordance patterns
2( Implies
3 ( AndLink ( Normal HorvathClock ) ( Elevated GrimAge ))
4 ( ElevatedRisk MortalityIndependentOfChronologicalAging )
5 (stv 0.75 0.80) )
6
7( Implies
8 ( AndLink ( Elevated HorvathClock ) ( Normal GrimAge ))
9 ( AcceleratedEpigeneticAging_LowMortalityRisk )
10 (stv 0.60 0.65) )
11
12( Implies
13 ( GreaterThan DunedinPACE 1.0)
14 CurrentlyAcceleratedAging
15 (stv 0.90 0.92) )
16
17; Inter - clock correlations ( weak = measure different things )
18( CorrelatedWith AgeAccelGrim AgeAccelHorvath (stv 0.17 0.90) )
19( CorrelatedWith AgeAccelGrim AgeAccelPheno (stv 0.45 0.90) )
A.5 Intervention Eﬀects on Clocks
Link interventions to expected clock changes:
1; Direct intervention - clock relationships ( from trials )
2( Causes Caloric_Restriction ( Decreases DunedinPACE ) (stv 0.65 0.75) )
3( Causes Exercise ( Decreases GrimAgeAcceleration ) (stv 0.45 0.60) )
4( Causes Smoking_Cessation ( Decreases DNAmPackYears ) (stv 0.70 0.80) )
5
6; Inferred via mechanism ( lower confidence )
7( Causes Senolytics ( Decreases DNAmPAI1 ) (stv 0.55 0.50) )
8( Causes Rapamycin ( Decreases DNAmGDF15 ) (stv 0.40 0.45) )
9( Causes Metformin ( Decreases GrimAgeAcceleration ) (stv 0.50 0.55) )
A.6 Mechanistic Pathways Connecting to Clocks
Enable causal reasoning from interventions to clock changes:
1; Senescence pathway
2( Causes CellularSenescence SASP (stv 0.90 0.95) )
3( Causes SASP ChronicInflammation (stv 0.85 0.90) )
4( Causes SASP ( Elevated PAI1 ) (stv 0.75 0.80) )
5( Causes ChronicInflammation ( Increases GrimAgeAcceleration ) (stv 0.70 0.75) )
6
7; mTOR pathway
8( Causes mTORC1_Hyperactivity ( Decreases Autophagy ) (stv 0.85 0.88) )
9( Causes ( Decreases Autophagy ) ( Increases SenescentCellBurden ) (stv 0.60
0.65) )
10( Causes InsulinResistance mTORC1_Hyperactivity ( stv 0.65 0.70) )
11
12; Mitochondrial pathway
13( Causes MitochondrialDysfunction ( Increases GDF15 ) (stv 0.75 0.70) )
14( Causes MitochondrialDysfunction ( Increases ROS) (stv 0.85 0.88) )
15( Causes ( Increases ROS) OxidativeStress (stv 0.90 0.92) )
Page 12

## Page 13

Technical Overview PLN Longevity Biomarker Pipeline
A.7 Demographic and Lifestyle Modiﬁers
Enable personalization based on NHANES covariates:
1; Sex - specific effects
2( SexSpecificEffect GrimAge MortalityPrediction ( Sex Female )
3 (stv 0.87 0.90) )
4( SexSpecificEffect GrimAge MortalityPrediction ( Sex Male )
5 (stv 0.82 0.90) )
6
7; Lifestyle associations
8( CorrelatedWith HighEducation ( Decreased GrimAgeAcceleration ) (stv 0.50
0.78) )
9( CorrelatedWith MediterraneanDiet ( Decreased GrimAgeAcceleration ) ( stv 0.45
0.65) )
10( CorrelatedWith RegularExercise ( Decreased DunedinPACE ) (stv 0.55 0.70) )
11( CorrelatedWith Obesity ( Increased GrimAgeAcceleration ) (stv 0.60 0.82) )
A.8 Risk Quantiﬁcation Relationships
Enable numeric risk calculations:
1; Hazard ratios (for risk computation )
2( HazardRatio GrimAgeAcceleration AllCauseMortality
3 ( Per5Years 1.35) (stv 0.85 0.92) )
4( HazardRatio ( Elevated CRP ) CardiovascularEvent
5 ( Threshold 3.0) (HR 1.4) ( stv 0.75 0.88) )
6( HazardRatio ( Elevated DNAmPAI1 ) Thrombosis
7 ( Per1SD 1.25) (stv 0.65 0.80) )
8
9; Absolute risk modifiers
10( BaselineRisk CardiovascularEvent (Age 50 -59) ( Sex Male ) 0.08)
11( BaselineRisk CardiovascularEvent (Age 60 -69) ( Sex Male ) 0.15)
A.9 Summary: Priority Extraction Targets
For a compelling NHANES-based demo, prioritize extraction of:
1.GrimAge component interpretations (what each DNAm surrogate reﬂects)
2.Clock–mortality hazard ratios (quantitative risk relationships)
3.Clock–biomarker correlations (link to NHANES blood markers)
4.Intervention–mechanism–clock chains (for recommendation engine)
5.Clock discordance interpretations (for personalized analysis)
These relationships enable the six demonstration scenarios described in Section 4, providing
concrete evidence of PLN’s advantages over LLM-based approaches.
Page 13

## Page 14

Technical Overview PLN Longevity Biomarker Pipeline
B Open-Access Sources for Supplement Knowledge Extraction
The following sources provide non-paywalled content suitable for extracting supplement-related
relationships. SourcesareorganizedbytypeandincludespeciﬁcURLsforsystematicharvesting.
B.1 Primary Databases
B.1.1 DrugAge Database (Supplements Section)
URL:https://genomics.senescence.info/drugs/
Content : Contains many natural compounds alongside pharmaceuticals
Key supplements included : Resveratrol, curcumin, quercetin, spermidine, alpha-lipoic
acid, carnosine, melatonin
Format: Downloadable CSV with compound, species, eﬀect size, mechanism
Extraction value : Pre-structured data; can import directly to Atomspace
B.1.2 Geroprotectors.org
URL:http://geroprotectors.org/
Content : Curated database of geroprotective compounds
Includes : 259 compounds with mechanisms, targets, evidence levels
Key data : Compound!target!pathway mappings
Extraction value : Structured mechanism-of-action data
B.1.3 LIFESPAN.io Supplement Database
URL:https://www.lifespan.io/
Content : Consumer-oriented but with citations
Useful for : Identifying commonly discussed supplements and claims
Extraction value : Claims to verify against primary literature
B.2 Review Articles and Meta-Analyses (Open Access)
These reviews provide consolidated evidence suitable for relationship extraction:
B.2.1 NAD+ Precursors (NMN, NR)
1. Covarrubias et al. (2021). “NAD+ metabolism and its roles in cellular processes during
ageing.” Nature Reviews Molecular Cell Biology .
PMC: Check for green open access version
Preprint: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8527903/
2. Yoshino et al. (2018). “NAD+ Intermediates: The Biology and Therapeutic Potential.”
Cell Metabolism .
PMC7442590
Page 14

## Page 15

Technical Overview PLN Longevity Biomarker Pipeline
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7442590/
3. Shade (2020). “The Science Behind NMN.” Integrative Medicine .
Open access review of human-relevant data
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7238909/
Key extractable relationships :
1( Causes NMN ( Increases NADplus ) (stv 0.85 0.80) )
2( Causes NR ( Increases NADplus ) (stv 0.85 0.82) )
3( Causes ( Increases NADplus ) ( Activates SIRT1 ) ( stv 0.75 0.70) )
4( Causes ( Increases NADplus ) ( Improves MitochondrialFunction )
5 (stv 0.70 0.65) )
6( Causes ( NADplus_Decline ) Aging (stv 0.70 0.75) )
B.2.2 Senolytics and Senomorphics (Fisetin, Quercetin)
1. Kirkland & Tchkonia (2020). “Senolytic drugs: from discovery to translation.” Journal of
Internal Medicine .
PMC7405395
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7405395/
2. Yousefzadeh et al. (2018). “Fisetin is a senotherapeutic that extends health and lifespan.”
EBioMedicine .
PMC6197652
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6197652/
3. Zhu et al. (2015). “The Achilles’ heel of senescent cells.” Aging Cell .
Original D+Q discovery paper
PMC4531078
Key extractable relationships :
1( Causes Fisetin ( Clears SenescentCells ) (stv 0.65 0.55) )
2( Causes Quercetin ( Inhibits BCL2Family ) (stv 0.60 0.60) )
3( Causes Fisetin ( Inhibits mTOR ) (stv 0.50 0.50) )
4( Causes Fisetin ( Inhibits PI3K ) (stv 0.55 0.50) )
5( Extends Fisetin Lifespan_Mouse (stv 0.60 0.65) )
6( EvidenceLevel Fisetin HumanTrials_Ongoing (stv 0.50 0.60) )
B.2.3 Berberine
1. Neag et al. (2018). “Berberine: Botanical Occurrence, Traditional Uses, Extraction Meth-
ods, andRelevanceinCardiovascular, Metabolic, Hepatic, andRenalDisorders.” Frontiers
in Pharmacology .
Fully open access
URL: https://www.frontiersin.org/articles/10.3389/fphar.2018.00557/full
2. Och et al. (2022). “Biological Activity of Berberine—A Summary Update.” Toxins.
Open access MDPI
Page 15

## Page 16

Technical Overview PLN Longevity Biomarker Pipeline
URL: https://www.mdpi.com/2072-6651/14/11/736
3. Yin et al. (2008). “Eﬃcacy of berberine in patients with type 2 diabetes mellitus.”
Metabolism .
Key clinical trial; check for PMC version
Key extractable relationships :
1( Causes Berberine ( Activates AMPK ) (stv 0.80 0.75) )
2( Causes Berberine ( Inhibits mTORC1 ) (stv 0.65 0.60) )
3( Causes Berberine ( Decreases HepaticGlucoseOutput ) (stv 0.75 0.78) )
4( Causes Berberine ( Decreases LDL) (stv 0.65 0.75) )
5( Causes Berberine ( Decreases FastingGlucose ) ( stv 0.70 0.80) )
6( SimilarMechanism Berberine Metformin AMPK_Activation )
7( Interaction Berberine CYP3A4 " Inhibitor - check drug interactions ")
B.2.4 Spermidine
1. Madeo et al. (2018). “Spermidine in health and disease.” Science.
May require author manuscript; check PMC
Key mechanistic review
2. Eisenberget al. (2016). “Cardioprotection and lifespan extensionby the natural polyamine
spermidine.” Nature Medicine .
PMC5806691 (author manuscript)
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5806691/
3. Kiechl et al. (2018). “Higher spermidine intake is linked to lower mortality.” American
Journal of Clinical Nutrition .
Epidemiological study
Open access
Key extractable relationships :
1( Causes Spermidine ( Induces Autophagy ) (stv 0.80 0.75) )
2( Causes Spermidine ( Inhibits EP300 ) (stv 0.70 0.65) )
3( Causes ( Induces Autophagy ) ( Clears DamagedMitochondria ) (stv 0.65 0.60) )
4( Extends Spermidine Lifespan_Mouse (stv 0.55 0.70) )
5( Extends Spermidine Lifespan_Yeast (stv 0.75 0.85) )
6( CorrelatedWith HighDietarySpermidine ReducedMortality_Human
7 (stv 0.55 0.70) )
B.2.5 Omega-3 Fatty Acids
1. Calder (2017). “Omega-3 fatty acids and inﬂammatory processes.” Nutrients .
Open access MDPI
URL: https://www.mdpi.com/2072-6643/9/3/272
2. Tan et al. (2022). “Association of omega-3 fatty acid levels with mortality.” Nature
Communications .
Fully open access
Page 16

## Page 17

Technical Overview PLN Longevity Biomarker Pipeline
URL: https://www.nature.com/articles/s41467-022-31222-0
Key extractable relationships :
1( Causes Omega3 ( Decreases Inflammation ) (stv 0.70 0.85) )
2( Causes Omega3 ( Decreases CRP) (stv 0.60 0.80) )
3( Causes Omega3 ( Decreases IL6) (stv 0.55 0.75) )
4( Causes EPA ( Produces Resolvins ) (stv 0.80 0.85) )
5( Causes DHA ( Improves MembraneFunction ) (stv 0.75 0.80) )
6( CorrelatedWith HighOmega3Index ReducedMortality ( stv 0.60 0.82) )
B.2.6 Curcumin
1. Hewlings & Kalman (2017). “Curcumin: A Review of Its Eﬀects on Human Health.”
Foods.
Open access MDPI
URL: https://www.mdpi.com/2304-8158/6/10/92
2. Dei Cas & Bhosale (2019). “Curcumin: Therapeutic Potential in Human Health.” Antiox-
idants.
Open access
URL: https://www.mdpi.com/2076-3921/8/10/442
Key extractable relationships :
1( Causes Curcumin ( Inhibits NFkB ) (stv 0.75 0.70) )
2( Causes Curcumin ( Activates Nrf2 ) (stv 0.65 0.60) )
3( Causes Curcumin ( Decreases Inflammation ) (stv 0.65 0.65) )
4( Limitation Curcumin PoorBioavailability (stv 0.90 0.95) )
5( EnhancedBy Curcumin Piperine Bioavailability_20x )
6( EnhancedBy Curcumin LiposomalFormulation Bioavailability )
B.2.7 Taurine
1. Singh et al. (2023). “Taurine deﬁciency as a driver of aging.” Science.
Major 2023 paper showing lifespan eﬀects
Check for PMC/preprint version
DOI: 10.1126/science.abn9257
2. Schaﬀer & Kim (2018). “Eﬀects and Mechanisms of Taurine as a Therapeutic Agent.”
Biomolecules & Therapeutics .
PMC5933890
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5933890/
Key extractable relationships :
1( Causes TaurineDeficiency AcceleratedAging (stv 0.70 0.65) )
2( Causes Taurine ( Improves MitochondrialFunction ) (stv 0.60 0.55) )
3( Causes Taurine ( Decreases OxidativeStress ) (stv 0.65 0.60) )
4( Extends Taurine Lifespan_Mouse (stv 0.55 0.60) )
5( Extends Taurine Healthspan_Mouse (stv 0.60 0.60) )
6( CorrelatedWith LowTaurine AcceleratedAging_Human ( stv 0.50 0.55) )
Page 17

## Page 18

Technical Overview PLN Longevity Biomarker Pipeline
B.2.8 Glycine
1. Miller et al. (2019). “Glycine supplementation extends lifespan of male and female mice.”
Aging Cell .
PMC6826123
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6826123/
2. Brind et al. (2011). “Dietary glycine supplementation mimics lifespan extension by dietary
methionine restriction in Fisher 344 rats.” FASEB Journal .
Conference abstract; mechanism paper
Key extractable relationships :
1( Causes Glycine ( Supports GlutathioneSynthesis ) ( stv 0.75 0.80) )
2( Causes Glycine ( MimicsMethioninRestriction ) (stv 0.55 0.55) )
3( Causes ( Supports GlutathioneSynthesis ) ( Decreases OxidativeStress )
4 (stv 0.70 0.75) )
5( Extends Glycine Lifespan_Mouse (stv 0.50 0.65) )
6( Extends Glycine Lifespan_Rat (stv 0.55 0.60) )
B.2.9 Alpha-Lipoic Acid and CoQ10
1. Salehi et al. (2019). “Insights on the Use of -Lipoic Acid for Therapeutic Purposes.”
Biomolecules .
Open access MDPI
URL: https://www.mdpi.com/2218-273X/9/8/356
2. Hernández-Camacho et al. (2018). “Coenzyme Q10 Supplementation in Aging and Dis-
ease.” Frontiers in Physiology .
Fully open access
URL: https://www.frontiersin.org/articles/10.3389/fphys.2018.00044/full
B.3 Negative Evidence Sources
Critically important for avoiding overly optimistic recommendations:
B.3.1 NIA Interventions Testing Program (ITP)
URL:https://www.nia.nih.gov/research/dab/interventions-testing-program-itp
Publications list : See NIA website for current publication list
Key negative results :
–Resveratrol: No lifespan eﬀect
–Green tea extract: No eﬀect
–Curcumin: No eﬀect (bioavailability issues suspected)
–Oxaloacetate: No eﬀect
–Fish oil (standard dose): No eﬀect
Critical for calibration :
Page 18

## Page 19

Technical Overview PLN Longevity Biomarker Pipeline
1( Extends Resveratrol Lifespan_Mouse_ITP (stv 0.10 0.90) )
2( EvidenceLevel Resveratrol ITP_Negative (stv 0.90 0.95) )
3( Note Resveratrol "Do not recommend despite popularity ")
4
5( Extends GreenTeaExtract Lifespan_Mouse_ITP (stv 0.10 0.85) )
6( Extends Curcumin Lifespan_Mouse_ITP (stv 0.10 0.80) )
B.3.2 Cochrane Reviews
URL:https://www.cochranelibrary.com/
Relevant reviews :
–“Antioxidant supplements for prevention of mortality” (negative for most)
–“Omega-3 fatty acids for cardiovascular disease” (modest eﬀects)
–“Vitamin D supplementation” (limited evidence for aging)
Extraction value : Meta-analytic eﬀect sizes with conﬁdence intervals
B.4 Preprint Servers
For emerging research not yet peer-reviewed (ﬂag with lower conﬁdence):
bioRxiv :https://www.biorxiv.org/ (search “aging” + compound name)
medRxiv :https://www.medrxiv.org/ (clinical preprints)
Extraction note : Preprint evidence should receive conﬁdence penalty of 0.1–0.2
B.5 Structured Extraction Targets by Compound
For initial knowledge base construction, prioritize extraction of:
Compound Primary Mechanism Evidence Level Priority
Fisetin Senolytic Animal + early human High
Berberine AMPK activation Multiple human trials High
NMN/NR NAD+ precursor Preliminary human High
Omega-3 Anti-inﬂammatory Extensive human High
Spermidine Autophagy Animal + epidemiology Medium
Glycine Glutathione/Met restriction Animal Medium
Taurine Mitochondrial Recent animal Medium
Curcumin NF- B inhibition Mixed human Medium
CoQ10 Mito ETC support Mechanistic Low
Alpha-lipoic acid Antioxidant/insulin Moderate human Low
Resveratrol SIRT1 activation ITP negative Low (negative)
Table 3: Supplement extraction priorities for initial knowledge base
B.6 Extraction Protocol for Supplements
When extracting from supplement literature, apply these conﬁdence adjustments:
1; Evidence level -> Confidence modifier
2( EvidenceModifier RCT_Human 1.0)
3( EvidenceModifier MultipleHumanTrials 0.85)
4( EvidenceModifier SingleHumanTrial 0.70)
Page 19

## Page 20

Technical Overview PLN Longevity Biomarker Pipeline
5( EvidenceModifier ITP_Positive 0.90)
6( EvidenceModifier ITP_Negative 0.90) ; high confidence in negative
7( EvidenceModifier AnimalStudies_Replicated 0.65)
8( EvidenceModifier AnimalStudies_Single 0.50)
9( EvidenceModifier InVitro 0.35)
10( EvidenceModifier Epidemiological 0.60)
11( EvidenceModifier Preprint 0.40)
12( EvidenceModifier TraditionalUse 0.20)
This calibration ensures that supplement recommendations carry appropriately lower conﬁ-
dence than pharmaceutical interventions, maintaining intellectual honesty while still providing
useful guidance to users who have expressed interest in exploring preliminary evidence.
C Key Source Papers for Initial Curation
C.1 Pharmaceutical and Mechanistic Sources
1. Lu et al. (2019). “DNA methylation GrimAge strongly predicts lifespan and healthspan.”
Aging. PMC6366976
2. Horvath & Raj (2018). “DNA methylation-based biomarkers and the epigenetic clock
theory of ageing.” Nature Reviews Genetics .
3. López-Otín et al. (2023). “Hallmarks of aging: An expanding universe.” Cell.
4. Barardo et al. (2017). “The DrugAge database of aging-related drugs.” Aging Cell .
PMC5418190
5. de Grey (2005). “Curing ageing and the consequences.” EMBO Reports . PMC1299264
6. Harrison et al. (2009). “Rapamycin fed late in life extends lifespan.” Nature.
7. Hickson et al. (2019). “Senolytics decrease senescent cells in humans.” EBioMedicine .
PMC6796530
C.2 Supplement and Nutraceutical Sources
1. Yousefzadeh et al. (2018). “Fisetin is a senotherapeutic that extends health and lifespan.”
EBioMedicine . PMC6197652
2. Yoshino et al. (2018). “NAD+ Intermediates: The Biology and Therapeutic Potential.”
Cell Metabolism . PMC7442590
3. Neag et al. (2018). “Berberine: Botanical Occurrence, Traditional Uses, Extraction Meth-
ods.” Frontiers in Pharmacology . Open access.
4. Eisenberg et al. (2016). “Cardioprotection and lifespan extension by spermidine.” Nature
Medicine . PMC5806691
5. Singhetal. (2023). “Taurinedeﬁciencyasadriverofaging.” Science. DOI:10.1126/science.abn9257
6. Milleretal. (2019). “Glycinesupplementationextendslifespan.” Aging Cell . PMC6826123
7. Calder (2017). “Omega-3 fatty acids and inﬂammatory processes.” Nutrients . Open access
MDPI.
8. Hewlings & Kalman (2017). “Curcumin: A Review of Its Eﬀects on Human Health.”
Foods. Open access MDPI.
Page 20

## Page 21

Technical Overview PLN Longevity Biomarker Pipeline
9. NIAInterventionsTestingProgrampublications. Availableat: nia.nih.gov/research/dab/interventions-
testing-program-itp/publications
Page 21
