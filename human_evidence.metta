;; human_evidence.metta
;; What human evidence exists for an intervention — and what does not.   [v1]
;;
;; THE FAILURE THIS FIXES. Asked "what does the evidence say about metformin in
;; HUMANS?", the system produced a prose apology: the loaded DrugAge stack is
;; overwhelmingly model-organism evidence (worms, flies, mice), so it cannot
;; answer a question about people. The apology was true. It was also un-grounded
;; — there was nothing in the knowledge base to point at, so the honest answer
;; had to be composed in English by a language model, which is precisely the
;; thing this project exists to not do.
;;
;; This layer turns that apology into a queryable record set. It holds a small
;; number of HUMAN studies, each with its design, its n, what was measured, what
;; was found, its PubMed ID and its evidence tier, and it makes three states
;; distinguishable that prose blurs together:
;;
;;   1. a POSITIVE human result        -> a record with (result ReportedBenefit)
;;   2. a NULL human result            -> a record with (result ReportedNull)
;;   3. NO human record at all         -> (NoHumanEvidenceRecord <intv> "…")
;;
;; (2) versus (3) is the whole point. "Dasatinib + quercetin did not change
;; pulmonary function in 14 people with IPF" and "nobody has curated a human
;; study of compound X here" are opposite epistemic states, and an LLM asked to
;; summarise a knowledge base that cannot tell them apart will reliably turn the
;; second into the first. Justice 2019 gives us both states inside ONE trial —
;; physical function improved, pulmonary function did not — so the distinction is
;; exercised by real data rather than asserted in a comment.
;;
;; A FOURTH STATE, AND THE ONE MOST OFTEN FAKED: a trial that is PLANNED and has
;; NOT REPORTED. TAME (Targeting Aging with Metformin) is the headline example.
;; It gets a publication record, a design, an outcome and an explicit
;; (result NotYetReported) — and NO evidence tier and NO truth value. There is no
;; number to have. Assigning one would be the single most tempting fabrication in
;; this entire domain, so the record writes the absence out: `(tier NotStated)`,
;; a positive statement that a planned trial has no tier, sitting where a reader
;; looks for one.
;;
;; HONESTY CONTRACT:
;;   * NOTHING HERE CARRIES A TRUTH VALUE. These are provenance records, not
;;     causal claims — the same stance hallmark_targeting.metta takes for
;;     TargetsHallmark. They do not add an edge to the graph `infer` traverses,
;;     they do not enter any ranking, and they cannot change a single existing
;;     answer. A study record says "this was measured in this many people and
;;     this was found". Turning that into a signed, weighted causal edge is a
;;     separate act of calibration that a curator must perform deliberately.
;;   * Every tier is an EvidenceCategory from epistemic_calibration.metta, the
;;     single authority — never a fresh number, and never an invented tier.
;;   * Every PMID below was verified against PubMed before it was written down.
;;   * An intervention whose human evidence is already recorded elsewhere is
;;     CROSS-REFERENCED, not copied (§4). Duplicating one body of evidence into
;;     two files is how a knowledge base starts double-counting itself.
;;
;; Load AFTER the curated intervention + supplement stack (it names Metformin,
;; DasatinibPlusQuercetin and Omega3, and reuses a Publication that
;; hallmark_targeting.metta already declares):
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   hallmark_targeting mechanistic_bridges pln_deduction
;;   pln_intervention_ranking pln_abductive_diagnosis patient_profile
;;   pln_counterfactual pln_risk_prediction supplement_evidence
;;   pln_supplement_recommendation
;;   human_evidence                      (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_human_evidence.py. Uses only core MeTTa builtins (match, collapse,
;; superpose, let/let*, car-atom, ==). As in every layer the space $space is
;; threaded explicitly (never a bare &self — the cross-module gotcha
;; docs/calibration_layer.md §10.1 documents), and every collapse is let*-forced
;; before an (== … ()) test so the comparison sees a reduced tuple.

;; ================================================================
;; 0. Why there is still no MetaAnalysis_Human tier
;; ================================================================
;; The evaluation reported that the KB has "no human-evidence tier". That is not
;; quite right: epistemic_calibration.metta already has RCT_Human (1.0),
;; MultipleHumanTrials (0.85), SingleHumanTrial (0.70) and Epidemiological
;; (0.60), and every record below lands on one of them. What is genuinely absent
;; is a META-ANALYSIS tier and a MECHANISTIC-CONSENSUS tier, both already written
;; up as open questions (docs/calibration_layer.md #4, docs/deduction_layer.md #6).
;;
;; We did NOT add MetaAnalysis_Human here, and the reason is a rule worth stating:
;; no record in this file needs one. The four studies below are one observational
;; cohort, two open-label pilots and one unreported trial. Adding a tier that
;; nothing uses means picking a confidence number with no record to calibrate it
;; against — inventing precision to close a review comment. The one meta-analysis
;; this repository now cites (Joehanes 2016, in lifestyle_evidence.metta) is
;; mapped conservatively onto MultipleHumanTrials, with that choice argued in
;; place. When a second and third meta-analysis arrive and the conservative
;; mapping starts costing us real discrimination, the tier earns its place and
;; gets added with evidence behind its value.

;; ================================================================
;; 1. The record shape — and the measured reason it is ONE atom
;; ================================================================
;; A record is a SINGLE atom with fixed arity:
;;
;;   (HumanEvidence <record-id> <intervention> <outcome>
;;      (design D) (n N) (result R) (tier T) (pmid "…")
;;      (measured "…") (found "…") (caveat "…"))
;;
;; That is a departure from HallmarkInterventionEvidence, which spends one atom
;; per field, and it is not a stylistic preference. MEASURED on this machine,
;; hyperon 0.2.10: the generic runtime space (every .metta file under
;; PLN_MAX_KB_FILE_BYTES, ~940 top-level expressions before this patch) aborts
;; the process — the non-unwinding trie panic at hyperon-space/src/index/trie.rs
;; — somewhere around 1050-1070 top-level expressions, and TYPE DECLARATIONS are
;; the most expensive kind: adding as few as five `(: X Y)` atoms to an
;; already-loaded space was enough to tip it. A one-predicate-per-field record
;; set cost ~113 expressions here and crashed the whole test suite. The flat
;; record costs ~49 and does not. tests/test_human_evidence.py carries a canary
;; that fails with a message instead of aborting the interpreter, so the next
;; layer gets a red test rather than a dead process.
;;
;; The flat shape also improves the thing this file is FOR. A field a study does
;; not have is written NotStated IN the record, where a reader sees it:
;; TAME's `(tier NotStated)` is a positive statement that a planned trial has no
;; evidence tier, not a line someone forgot to add.

;; Study design. Deliberately coarse and deliberately explicit — "open-label
;; pilot" and "randomized controlled trial" must not both flatten to "a study".
(: StudyDesign Type)
(: RandomizedControlledTrial StudyDesign)
(: ObservationalCohort       StudyDesign)
(: OpenLabelPilot            StudyDesign)
(: PlannedTrial              StudyDesign)

;; What the study found, as a SYMBOL — the state prose collapses.
(: HumanOutcomeResult Type)
(: ReportedBenefit  HumanOutcomeResult)   ; a positive, reported result
(: ReportedNull     HumanOutcomeResult)   ; measured, and no change found
(: ReportedHarm     HumanOutcomeResult)   ; measured, and worse
(: NotYetReported   HumanOutcomeResult)   ; the trial exists; results do not

;; A field this study does not have. A SYMBOL, not an empty string and not a
;; zero: a consumer must not be able to read a missing n as n=0 or a missing
;; tier as a low tier.
(: NotStated Atom)

;; The outcomes these studies measured. AllCauseMortality is already an Outcome
;; (grim_age_lu2019_evidence.metta:10); the rest are introduced here.
(Inheritance SenescentCellBurden        Outcome)
(Inheritance PhysicalFunction           Outcome)
(Inheritance PulmonaryFunction          Outcome)
(Inheritance CompositeAgeRelatedDisease Outcome)

(: HumanEvidence (-> Atom Intervention Outcome Atom Atom Atom Atom Atom Atom Atom Atom Atom))

;; The cross-reference (§4): this intervention's human evidence lives elsewhere
;; in the KB, at this tier, and here is where to find it.
(: HumanEvidenceElsewhere (-> Intervention EvidenceCategory String Atom))

;; ================================================================
;; 2. Publication provenance
;; ================================================================
;; BannisterEtAl2014_Metformin is NOT redeclared — hallmark_targeting.metta §1
;; already carries its title, year, journal, DOI and PubMedID (25041462), and it
;; is cited there as the survival evidence behind Metformin's hallmark link. One
;; publication, one record; this file references the symbol.

(: BarzilaiEtAl2016_TAME Publication)
(PublicationTitle BarzilaiEtAl2016_TAME "Metformin as a Tool to Target Aging")
(PublicationYear BarzilaiEtAl2016_TAME 2016)
(JournalName BarzilaiEtAl2016_TAME "Cell Metabolism")
(DOI BarzilaiEtAl2016_TAME "10.1016/j.cmet.2016.05.011")
(PubMedID BarzilaiEtAl2016_TAME "27304507")

(: HicksonEtAl2019_Senolytics Publication)
(PublicationTitle HicksonEtAl2019_Senolytics "Senolytics decrease senescent cells in humans: Preliminary report from a clinical trial of Dasatinib plus Quercetin in individuals with diabetic kidney disease")
(PublicationYear HicksonEtAl2019_Senolytics 2019)
(JournalName HicksonEtAl2019_Senolytics "EBioMedicine")
(DOI HicksonEtAl2019_Senolytics "10.1016/j.ebiom.2019.08.069")
(PubMedID HicksonEtAl2019_Senolytics "31542391")

(: JusticeEtAl2019_SenolyticsIPF Publication)
(PublicationTitle JusticeEtAl2019_SenolyticsIPF "Senolytics in idiopathic pulmonary fibrosis: Results from a first-in-human, open-label, pilot study")
(PublicationYear JusticeEtAl2019_SenolyticsIPF 2019)
(JournalName JusticeEtAl2019_SenolyticsIPF "EBioMedicine")
(DOI JusticeEtAl2019_SenolyticsIPF "10.1016/j.ebiom.2018.12.052")
(PubMedID JusticeEtAl2019_SenolyticsIPF "30616998")

;; ================================================================
;; 3. The records
;; ================================================================

;; --- Metformin: one observational cohort ------------------------
;; The strongest HUMAN longevity claim metformin actually has, and it is an
;; observational one. Bannister 2014 matched UK primary-care records: type 2
;; diabetics started on metformin monotherapy lived longer than matched
;; non-diabetic controls (adjusted median survival time 15% LOWER in the
;; controls, survival time ratio 0.85, 95% CI 0.81-0.90).
;;
;; The caveat is not decoration. This compares a treated diabetic population with
;; a non-diabetic one; it is confounded by indication, by who gets prescribed
;; metformin first-line and by what the comparator (sulphonylurea) does. It is
;; NOT a trial of metformin in healthy people, and the paper itself only goes as
;; far as "implies that metformin may confer benefit in non-diabetes".
(HumanEvidence Bannister2014_Metformin_Survival Metformin AllCauseMortality
   (design ObservationalCohort) (n 78241) (result ReportedBenefit)
   (tier Epidemiological) (pmid "25041462")
   (measured "all-cause mortality in 78,241 type 2 diabetics initiated on metformin monotherapy versus 90,463 matched non-diabetic controls (and 12,222 on sulphonylurea), UK Clinical Practice Research Datalink, 503,384 person-years of follow-up")
   (found "metformin-initiated diabetics outlived their matched non-diabetic controls: adjusted median survival time was 15% lower in the controls (survival time ratio 0.85, 95% CI 0.81-0.90), and 38% lower on sulphonylurea (0.62, 0.58-0.66)")
   (caveat "Observational and confounded by indication: a treated diabetic population compared with a non-diabetic one, not a trial of metformin in people without diabetes. The authors' own claim is that it 'implies that metformin may confer benefit in non-diabetes'."))
(SupportedByPublication Bannister2014_Metformin_Survival BannisterEtAl2014_Metformin)

;; --- Metformin: TAME, planned, NOT REPORTED ---------------------
;; The record that exists to prevent a fabrication. TAME is the trial everyone
;; cites when asked whether metformin works in humans; it is a DESIGN RATIONALE
;; (Barzilai 2016 describes it as "the initial step in the development of
;; increasingly effective next-generation drugs"), and as of this record it has
;; not reported. `(tier NotStated)` is written out: a trial with no results has
;; no evidence tier, and the absence is a statement rather than an omission.
(HumanEvidence Barzilai2016_TAME_Metformin Metformin CompositeAgeRelatedDisease
   (design PlannedTrial) (n NotStated) (result NotYetReported)
   (tier NotStated) (pmid "27304507")
   (measured "planned: a composite of age-related disease endpoints in older adults, as the TAME (Targeting Aging with Metformin) trial rationale describes")
   (found "nothing — TAME had not reported when this record was curated. Barzilai 2016 is a review stating the rationale and the design, not a result.")
   (caveat "This record carries NO evidence tier and NO truth value, on purpose. A planned trial has no result to calibrate, and attaching a confidence to TAME is the most common fabrication in this field. If you are quoting TAME as evidence that metformin extends human lifespan, you are quoting a study design."))
(SupportedByPublication Barzilai2016_TAME_Metformin BarzilaiEtAl2016_TAME)

;; --- D+Q: the first human senolytic target-engagement result ----
;; Hickson 2019: 9 people with diabetic kidney disease, three days of oral
;; dasatinib 100 mg + quercetin 1000 mg, open label, no control arm. Senescent
;; cell burden fell. This is the study that shows the DRUG DOES WHAT IT SAYS in a
;; human body — target engagement — and it is not an outcome trial.
(HumanEvidence Hickson2019_DQ_SenescentCells DasatinibPlusQuercetin SenescentCellBurden
   (design OpenLabelPilot) (n 9) (result ReportedBenefit)
   (tier SingleHumanTrial) (pmid "31542391")
   (measured "adipose tissue and skin biopsies before and 11 days after a 3-day oral course: p16INK4A- and p21-expressing cells, senescence-associated beta-galactosidase activity, adipose macrophages, and circulating SASP factors")
   (found "adipose senescent-cell burden decreased within 11 days, with fewer p16- and p21-expressing cells and fewer crown-like structures; epidermal p16 and p21 cells were reduced; circulating IL-1alpha, IL-6, MMP-9 and MMP-12 fell")
   (caveat "Open-label phase 1 pilot, n=9, no control arm, in diabetic kidney disease. A target-engagement result — senescent cells went down — not evidence of a clinical outcome, and not a population in which longevity was measured."))
(SupportedByPublication Hickson2019_DQ_SenescentCells HicksonEtAl2019_Senolytics)

;; --- D+Q in IPF: the SAME trial, split by outcome ---------------
;; Justice 2019 measured two things and got two different answers. Splitting it
;; into two records is what makes (2) distinguishable from (3) with real data:
;; ask for D+Q's human evidence and you get a benefit AND a null, from one n=14
;; open-label pilot, rather than a summary sentence that quietly keeps the first.
(HumanEvidence Justice2019_DQ_PhysicalFunction DasatinibPlusQuercetin PhysicalFunction
   (design OpenLabelPilot) (n 14) (result ReportedBenefit)
   (tier SingleHumanTrial) (pmid "30616998")
   (measured "6-minute walk distance, 4-metre gait speed and chair-stands time in 14 people with idiopathic pulmonary fibrosis, after intermittent dasatinib 100 mg/day + quercetin 1250 mg/day, three days a week for three weeks")
   (found "physical function improved significantly and clinically meaningfully (p < 0.05) across all three measures")
   (caveat "First-in-human, open-label, two-centre pilot with no control arm and n=14; the stated primary endpoints were retention and completion rates, i.e. feasibility. Physical function was a secondary measure and an uncontrolled before/after comparison."))
(SupportedByPublication Justice2019_DQ_PhysicalFunction JusticeEtAl2019_SenolyticsIPF)

;; The NULL. Same drug, same 14 people, same three weeks — and the measure that
;; defines the disease did not move. A knowledge base that cannot say this is a
;; knowledge base that will let a summariser say senolytics "improved IPF".
(HumanEvidence Justice2019_DQ_PulmonaryFunction DasatinibPlusQuercetin PulmonaryFunction
   (design OpenLabelPilot) (n 14) (result ReportedNull)
   (tier SingleHumanTrial) (pmid "30616998")
   (measured "pulmonary function tests, clinical chemistries, the FI-LAB frailty index and patient-reported health, in the same 14 idiopathic-pulmonary-fibrosis participants")
   (found "unchanged — pulmonary function, clinical chemistries, frailty index and reported health all showed no change")
   (caveat "A null in an uncontrolled n=14 pilot over three weeks is weak evidence of no effect, not proof of one. It is recorded because 'measured and found nothing' and 'never measured' are different answers, and only the first can be stated here."))
(SupportedByPublication Justice2019_DQ_PulmonaryFunction JusticeEtAl2019_SenolyticsIPF)

;; ================================================================
;; 4. Cross-references — evidence that lives elsewhere in the KB
;; ================================================================
;; Omega-3 already carries (EvidenceLevel Omega3 MultipleHumanTrials) in
;; supplement_evidence.metta, where the supplement recommender reads it and
;; turns it into a Tier1 placement. Re-stating it as a record here would give one
;; body of evidence two homes and let a future aggregation count it twice. The
;; pointer is a fact so the accessor can return it.
(HumanEvidenceElsewhere Omega3 MultipleHumanTrials "supplement_evidence.metta: (EvidenceLevel Omega3 MultipleHumanTrials), the tier the supplement recommender already scores. Not duplicated here — one body of evidence, one home.")

;; ================================================================
;; 5. Accessors
;; ================================================================
;; The absence answer. Checked for a cross-reference FIRST, so "recorded
;; elsewhere" never comes back as "not recorded". Every collapse is let-forced
;; before the (== … ()) test, so the comparison sees a reduced tuple.
(= (human-evidence-absent $space $intv)
   (let $xs (collapse
               (match $space (HumanEvidenceElsewhere $intv $tier $where)
                  (HumanEvidenceCrossReferenced $intv (tier $tier) (see $where))))
      (if (== $xs ())
          (NoHumanEvidenceRecord $intv
             "No human study of this intervention is curated in this knowledge base. That is an absence of a RECORD, not evidence of absence: this layer is a small hand-built table, and the bulk evidence loaded here (DrugAge, GenAge, CellAge) is model-organism and cell data, which cannot answer a question about people.")
          (car-atom $xs))))

;; human-evidence: every human record for an intervention, or the explicit
;; absence answer. Never silence — an empty result and "we have nothing" are the
;; two states this whole file exists to keep apart.
;;
;;   (human-evidence <space> <intervention>)
;;     -> (HumanEvidence …)                       one per record, or
;;     -> (HumanEvidenceCrossReferenced …)        recorded elsewhere in the KB, or
;;     -> (NoHumanEvidenceRecord <intv> "…")      nothing curated
(= (human-evidence $space $intv)
   (let $rs (collapse
               (match $space
                  (HumanEvidence $r $intv $o $d $n $res $t $p $m $f $c)
                  (HumanEvidence $r $intv $o $d $n $res $t $p $m $f $c)))
      (if (== $rs ())
          (human-evidence-absent $space $intv)
          (superpose $rs))))

;; Every intervention this layer holds a record for — the "what CAN you answer
;; in humans?" question, so a caller does not have to guess a name to find out.
(= (human-evidence-interventions $space)
   (collapse
      (match $space (HumanEvidence $r $i $o $d $n $res $t $p $m $f $c) $i)))

;; ================================================================
;; 6. Smoke-test queries (ACTUAL measured output, elided with … where a record
;;    string is long; tests/test_human_evidence.py asserts the full values)
;; ================================================================
;; The question that produced a prose apology. Two records come back — the
;; observational cohort, and TAME with its tier written out as NotStated.
;; (Result order is collapse order — not significant.)
;; !(human-evidence &self Metformin)
;;   => (HumanEvidence Bannister2014_Metformin_Survival Metformin AllCauseMortality
;;         (design ObservationalCohort) (n 78241) (result ReportedBenefit)
;;         (tier Epidemiological) (pmid "25041462")
;;         (measured "all-cause mortality in 78,241 …") (found "metformin-initiated …")
;;         (caveat "Observational and confounded by indication …"))
;;      (HumanEvidence Barzilai2016_TAME_Metformin Metformin CompositeAgeRelatedDisease
;;         (design PlannedTrial) (n NotStated) (result NotYetReported)
;;         (tier NotStated) (pmid "27304507")
;;         (measured "planned: a composite …") (found "nothing — TAME had not reported …")
;;         (caveat "This record carries NO evidence tier and NO truth value, on purpose. …"))
;;
;; A benefit and a null from the same 14-person trial, side by side:
;; !(human-evidence &self DasatinibPlusQuercetin)
;;   => (HumanEvidence Hickson2019_DQ_SenescentCells    … (n 9)  (result ReportedBenefit) (tier SingleHumanTrial) (pmid "31542391") …)
;;      (HumanEvidence Justice2019_DQ_PhysicalFunction  … (n 14) (result ReportedBenefit) (tier SingleHumanTrial) (pmid "30616998") …)
;;      (HumanEvidence Justice2019_DQ_PulmonaryFunction … (n 14) (result ReportedNull)    (tier SingleHumanTrial) (pmid "30616998") …)
;;
;; Recorded elsewhere, not duplicated:
;; !(human-evidence &self Omega3)
;;   => (HumanEvidenceCrossReferenced Omega3 (tier MultipleHumanTrials)
;;         (see "supplement_evidence.metta: (EvidenceLevel Omega3 MultipleHumanTrials), …"))
;;
;; Nothing curated — and it SAYS so, rather than returning empty:
;; !(human-evidence &self Rapamycin)
;;   => (NoHumanEvidenceRecord Rapamycin "No human study of this intervention is curated …")
;;
;; !(human-evidence-interventions &self)
;;   => (Metformin Metformin DasatinibPlusQuercetin DasatinibPlusQuercetin DasatinibPlusQuercetin)
;;      (one entry per RECORD, not per intervention — collapse order, not significant)
