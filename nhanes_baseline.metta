;; nhanes_baseline.metta
;; NHANES data-backed baseline ABSOLUTE RISK — TYPES AND RULES ONLY.          [v1]
;;
;; This file is the baseline half of step 2: `pln_risk_prediction.metta` models
;;
;;     risk = baseline(age, sex) × HR_clock ^ (z_clock × sd->years)
;;
;; and until now the `baseline(age, sex)` factor was a curated prior (§0(a) of that
;; file, "a dedicated calibrated baseline table is the honest upgrade"). This file
;; supplies the record type and the lookup rules that let a SURVEY-WEIGHTED,
;; MORTALITY-LINKED baseline take that factor's place — for the outcomes NHANES can
;; actually identify, and for no others.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; NO NUMBERS LIVE HERE, AND THAT IS THE POINT.
;; ─────────────────────────────────────────────────────────────────────────────
;; The CDC hosts (wwwn.cdc.gov, ftp.cdc.gov) are denied by this environment's egress
;; policy, so no NHANES value could be downloaded, computed *or verified* here. A
;; number written into a .metta file is consumed by inference and becomes a claim, so
;; this file contains types, accessors and rules exclusively — not one risk, not one
;; "illustrative" cell behind a TODO. The numeric BaselineRiskRecord / ClockAccelSpread
;; records are ETL build artifacts: `nhanes_mortality_etl.py` and `nhanes_dnam_etl.py`
;; write them under build/ (gitignored) from microdata the user supplies, exactly as
;; the HAGR ETLs already stage their output. Tests prove the machinery against
;; synthetic fixtures whose values are arbitrary by construction.
;; See docs/nhanes_integration.md §3.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; THE TRAP THIS FILE EXISTS TO CLOSE: INCIDENCE IS NOT MORTALITY.
;; ─────────────────────────────────────────────────────────────────────────────
;; `pln_risk_prediction.metta` predicts P(CHD in 10y) as an INCIDENT
;; coronary-heart-disease event, and reads HR 1.07 per year of AgeAccelGrim off
;; `grim_age_lu2019_evidence.metta`, where Lu et al. (2019) fit it to INCIDENT CHD.
;;
;; NHANES + the public-use Linked Mortality File identifies DEATH, not incidence.
;; Its cause recode gives "Diseases of heart" mortality, which is (a) fatal-only and
;; (b) broader than CHD. NHANES has no incident-CHD ascertainment at all — its
;; questionnaire items (MCQ160C/D/E) are lifetime "ever told" PREVALENCE at baseline,
;; usable only as an exclusion from the at-risk set, never as an outcome.
;;
;; So this file does NOT calibrate the CHD baseline, and deliberately gives no way to.
;; It emits baselines only for outcomes the data identifies, under their own honest
;; names, and the lookup is keyed on the OUTCOME so that a fatal-heart-disease risk
;; can never be served where an incident-CHD risk was asked for. Multiplying a
;; heart-disease-MORTALITY baseline by an incident-CHD hazard ratio is not a
;; conservative estimate or a rough one — it is an estimate of nothing: the baseline
;; is far too small for incidence, and the HR was fit to a different event process.
;; `baseline-risk-chd` therefore stays exactly as it is, curated and labelled.
;;
;; That refusal is the deliverable. It also serves docs/risk_prediction.md open-Q #5
;; (multi-outcome risk): the risk layer becomes outcome-parameterized end to end, so
;; AllCauseMortality — for which BOTH a NHANES baseline and a Lu 2019 hazard ratio
;; exist — becomes the first outcome this repo can price from data on both sides.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; LOAD ORDER
;; ─────────────────────────────────────────────────────────────────────────────
;; This file sits ON TOP of the risk layer. `outcome-matched-risk-inputs` (§6) reads
;; the hazard ratio through `clock-hr`, which pln_risk_prediction.metta defines, and
;; pln_risk_prediction.metta's `patient-baseline` calls `baseline-risk-data` (§5) from
;; here. That mutual reference is safe because every file is loaded into ONE space and
;; `=` rules are just atoms — only `!(…)` queries care about order — but it does mean
;; this file is not useful without the risk layer:
;;
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile pln_counterfactual
;;   pln_risk_prediction
;;   nhanes_baseline                     (this file)
;;   [build/nhanes_baseline.metta]       (the generated numeric records, if present)
;;
;; The chat app's _INFERENCE_STACK lists this file EARLIER, beside nhanes_reference and
;; before pln_risk_prediction. That is not a contradiction: for the reason above, the
;; position of a file carrying only `=` rules and type declarations does not affect the
;; result. The listing here is dependency order for a reader; the app's is readability.
;;
;; With this file ABSENT, pln_risk_prediction.metta is byte-identical to its previous
;; behavior: its `patient-baseline` tests for the presence of a BaselineRiskRecord
;; atom BEFORE it ever names `baseline-risk-data`, precisely so that the missing
;; symbol is never evaluated (a mismatched rule head in MeTTa does not yield nothing,
;; it yields the unreduced expression, which then propagates into arithmetic — the
;; semantics tests/test_nhanes_common.py::
;; test_a_mismatched_rule_head_returns_the_unreduced_expression_not_empty pins down).
;;
;; STATUS: v1, verified against hyperon 0.2.10. Uses only core MeTTa builtins
;; (match, superpose, let/let*, <, ==). As in every layer the space $space is threaded
;; explicitly (never a bare &self — a (match &self …) inside an imported module reads
;; only that module's own space, the cross-module gotcha docs/calibration_layer.md
;; §10.1 documents), and every comparison argument is let*-forced to a concrete value
;; because grounded ops do NOT evaluate their arguments.

;; ================================================================
;; 1. Outcome symbols
;; ================================================================
;; AllCauseMortality and CoronaryHeartDisease are already declared in
;; grim_age_lu2019_evidence.metta and are NOT redeclared here.
;;
;; ─────────────────────────────────────────────────────────────────────────────
;; HeartDiseaseMortality — read this before using it for anything.
;; ─────────────────────────────────────────────────────────────────────────────
;; MEANING, exactly: death with an underlying cause in NCHS's "Diseases of heart"
;; recode — ICD-10 I00-I09, I11, I13, I20-I51 — as identified by UCOD_LEADING == "001"
;; in the NHANES public-use Linked Mortality File.
;;
;; What that is NOT:
;;   * NOT incidence. It is FATAL heart disease only. A participant who has a
;;     myocardial infarction and survives the follow-up window contributes NO event.
;;     The absolute level is therefore far below any incidence risk for the same
;;     age/sex cell, and the gap is not a scale factor anyone here can supply.
;;   * NOT coronary heart disease. The I00-I51 range is BROADER than CHD (I20-I25):
;;     it includes rheumatic, hypertensive and pulmonary-circulation heart disease,
;;     the cardiomyopathies, the dysrhythmias, and HEART FAILURE (I50) — which this
;;     repo already models as its OWN outcome, CongestiveHeartFailure, carrying its
;;     own Lu 2019 hazard ratio (Lu2019_AgeAccelGrim_CHF, HR 1.10). So a
;;     HeartDiseaseMortality baseline overlaps two outcomes the risk layer keeps
;;     separate, and substituting it for either double-counts or mis-scales.
;;   * therefore NOT a substitute for incident CHD, and there is deliberately no rule
;;     in this file that converts between them.
;;
;; The three atoms below record that in machine-readable form, so the refusal is
;; auditable rather than only commented. Nothing in the inference stack traverses
;; them (they are not Effect / Implication / PartOf atoms), so they add no edge to
;; the causal graph `infer` walks.
(Inheritance HeartDiseaseMortality Outcome)

(OutcomeAscertainment  HeartDiseaseMortality FatalOnly)
(OutcomeICD10Range     HeartDiseaseMortality "I00-I09,I11,I13,I20-I51")
(NotSubstitutableFor   HeartDiseaseMortality CoronaryHeartDisease)
(IncludesDeathsFrom    HeartDiseaseMortality CongestiveHeartFailure)

;; ================================================================
;; 2. Shared symbols
;; ================================================================
;; The declarations in this section are REPEATED VERBATIM from nhanes_reference.metta
;; so that this file stands alone when the reference layer is not loaded. Repeating an
;; identical atom is harmless (nothing in the stack matches on a (: …) declaration —
;; verified), but the declarations must not DIVERGE: one symbol, one type.

;; --- age bands -----------------------------------------------------------------
;; Symbols are plain (Age_lt50, not Age<50) — MeTTa tokenizes < and > .
(: AgeBand Type)
(: Age_lt50  AgeBand)
(: Age_50_59 AgeBand)
(: Age_60_69 AgeBand)
(: Age_70p   AgeBand)

;; --- provenance ----------------------------------------------------------------
;; There is exactly ONE provenance tier in v1, because we ship nothing else. A
;; "published"/"transcribed" tier is deliberately absent: adding the tier invites
;; someone to fill it with a number nobody here could verify.
(: ReferenceProvenance Type)
(: NHANES_Microdata ReferenceProvenance)

;; --- clock acceleration definition ---------------------------------------------
;; A clock acceleration is either the RESIDUAL of clock age on chronological age —
;; what grim_age_core.metta means by (ResidualOf AgeAccelGrim GrimAge), and what Lu
;; 2019's HR 1.07/year was fit to — or the plain DIFFERENCE (clock age − age).
;; Emitting one where the other is meant silently mis-scales the risk exponent, so
;; the definition is recorded in the record, never assumed by the reader.
(: AccelDefinition Type)
(: Residual   AccelDefinition)
(: Difference AccelDefinition)

;; ================================================================
;; 3. The BaselineRiskRecord  (Schema B — FIXED)
;; ================================================================
;; Source-faithful style: a typed individual plus one field per atom, exactly as
;; grim_age_lu2019_evidence.metta records an evidence row. `nhanes_mortality_etl.py`
;; emits records of precisely this shape:
;;
;;   (: NHANESBaseline_ACM_Male_50_59_9902 BaselineRiskRecord)
;;   (BaseOutcome       NHANESBaseline_ACM_Male_50_59_9902 AllCauseMortality)
;;   (BaseSex           … Male)
;;   (BaseAgeBand       … Age_50_59)
;;   (BaseHorizonMonths … <number>)
;;   (BaseRisk          … <number>)
;;   …
;;
;; Provenance is mandatory (contract D2): every record says which cycles, which
;; weight variable, which time variable and which linkage vintage produced it.
;; No design-based standard error is emitted — a correct NHANES SE needs Taylor
;; linearization over the strata/PSU design variables, so rather than ship a wrong
;; SE the record ships the unweighted n, the event count and the estimator name and
;; lets a consumer judge reliability. Cells below the minimum n or the minimum event
;; count emit NOTHING at all rather than an atom with a caveat attached — inference
;; consumes atoms, not caveats.
(: BaselineRiskRecord Type)

;; --- identifying fields (the lookup key) ---------------------------------------
;; The OUTCOME is part of the key, not metadata. That is the whole D3/D21 defence:
;; there is no (BaseRisk …) reachable without naming the outcome it belongs to.
(: BaseOutcome (-> BaselineRiskRecord Outcome  Atom))
(: BaseSex     (-> BaselineRiskRecord Sex      Atom))
(: BaseAgeBand (-> BaselineRiskRecord AgeBand  Atom))

;; --- the estimate itself --------------------------------------------------------
;; The horizon is a FIELD, not an assumption. With the linkage ending 2019-12-31 the
;; 2011+ cycles carry under ten years of follow-up, so a record may legitimately be a
;; 96-month risk; a consumer that silently treats it as a ten-year risk is wrong.
;; §6 therefore reports the horizon beside the number it applies to.
(: BaseHorizonMonths (-> BaselineRiskRecord Number Atom))
(: BaseRisk          (-> BaselineRiskRecord Number Atom))

;; The estimator is named because the two are not interchangeable. All-cause
;; mortality uses the weighted Kaplan-Meier product-limit estimator; a CAUSE-SPECIFIC
;; risk uses the weighted Aalen-Johansen cumulative incidence function, because deaths
;; from other causes are COMPETING EVENTS, not censoring. (1 − KM) on a cause-specific
;; outcome over-states the risk, and a naive deaths/total proportion under-states any
;; of them, because follow-up is right-censored by the linkage end date.
(: BaselineEstimator Type)
(: WeightedKaplanMeier   BaselineEstimator)
(: WeightedAalenJohansen BaselineEstimator)
(: BaseEstimator (-> BaselineRiskRecord BaselineEstimator Atom))

;; --- reliability + provenance ---------------------------------------------------
(: BaseUnweightedN    (-> BaselineRiskRecord Number Atom))
(: BaseEvents         (-> BaselineRiskRecord Number Atom))
(: BaseWeightVariable (-> BaselineRiskRecord String Atom))
(: BaseTimeVariable   (-> BaselineRiskRecord String Atom))
(: BaseSourceCycles   (-> BaselineRiskRecord String Atom))
(: BaseLinkageVintage (-> BaselineRiskRecord String Atom))
(: BaseProvenance     (-> BaselineRiskRecord ReferenceProvenance Atom))

;; --- conditional fields ---------------------------------------------------------
;; Aalen-Johansen records only: the naive (1 − KM) figure the same cell would have
;; produced, carried so the competing-risks bias is VISIBLE next to the estimate that
;; corrects it, plus the competing-event count that drives the difference.
(: BaseNaiveKMRisk   (-> BaselineRiskRecord Number Atom))
(: BaseCompetingEvents (-> BaselineRiskRecord Number Atom))

;; Cause-specific records only: the UCOD_LEADING recode value the outcome was read
;; from. CHARACTER and zero-padded in the source file — "001", never 1 — because
;; coercing it to a number is how a cause code silently becomes the wrong cause.
(: BaseCauseCode (-> BaselineRiskRecord String Atom))

;; ================================================================
;; 4. The ClockAccelSpread record  (Schema C — FIXED)
;; ================================================================
;; A measured weighted SD of an epigenetic-clock acceleration, IN YEARS. Its one job
;; is to put a sourced number next to pln_risk_prediction.metta's curated knob
;; (= (grimaccel-sd-to-years) 4.2), which converts a patient's z (in SDs) into the
;; years the Lu 2019 hazard ratio is quoted per. The comparison rule lives in
;; pln_risk_prediction.metta (`grimaccel-sd-to-years-measured`); this file only
;; declares the record.
;;
;; SpreadDefinition is mandatory and is the reason the comparison is meaningful at
;; all: the SD of the RESIDUAL and the SD of the DIFFERENCE are different numbers,
;; and only the residual is on the scale AgeAccelGrim is defined on.
(: ClockAccelSpread Type)
(: SpreadMarker         (-> ClockAccelSpread Biomarker Atom))
(: SpreadDefinition     (-> ClockAccelSpread AccelDefinition Atom))
(: SpreadSDYears        (-> ClockAccelSpread Number Atom))   ; weighted SD, in YEARS
(: SpreadUnweightedN    (-> ClockAccelSpread Number Atom))
(: SpreadWeightVariable (-> ClockAccelSpread String Atom))
(: SpreadSourceCycles   (-> ClockAccelSpread String Atom))
(: SpreadProvenance     (-> ClockAccelSpread ReferenceProvenance Atom))

;; ================================================================
;; 5. The data-backed baseline lookup
;; ================================================================
;; age -> band. Half-open [lo, hi), identical to nhanes_common.AGE_BANDS and to
;; nhanes_reference.metta's `age-band`, and switching at the same three cutpoints as
;; `baseline-risk-chd` (< 50, < 60, < 70) so a patient lands in the same band
;; whichever layer asks. Single rule + nested if, so it is deterministic (one input
;; -> one output).
;;
;; It is a SEPARATE SYMBOL from nhanes_reference.metta's `age-band` on purpose: two
;; files defining the same head with the same body would make (age-band 58) return
;; TWO identical results whenever both layers are loaded, and a duplicated result is
;; a silent double-count downstream. Duplicating four cutpoints is the cheaper bug.
(= (baseline-age-band $age)
   (if (< $age 50) Age_lt50
       (if (< $age 60) Age_50_59
           (if (< $age 70) Age_60_69
               Age_70p))))

;; The record id(s) for one (outcome, band, sex) cell, or NOTHING.
;;
;; "Nothing" is the whole design. There is no default risk, no fallback to a
;; neighbouring band, no pooled "all adults" cell, and no borrowing of another
;; outcome's number. A cell that was suppressed for a small unweighted n or too few
;; events (contract D9) simply has no record, and this yields nothing — which is what
;; makes `patient-baseline` fall through to the curated prior for CHD, and yield
;; nothing at all for an outcome that has no curated prior either. The layer declines
;; rather than invents.
;;
;; AMBIGUITY DECLINES; it is not resolved by picking. This match does not key on
;; `BaseHorizonMonths`, so two runs of the same cycle at different horizons both land on
;; one (outcome, band, sex) — and putting the horizon into the record IDENTIFIER, as an
;; earlier commit did, makes them distinct individuals without making this lookup
;; distinguish them. Measured: with a 60-month and a 120-month record both loaded, the
;; consumer collapsed and took the first, serving a five-year baseline into a rule its own
;; documentation calls a ten-year risk.
;;
;; Picking one would be the same mistake as taking `car-atom` of a collapse in
;; `reference-cell`: single-valued but not single-MEANING, an ambiguity converted into a
;; confident wrong answer. So more than one candidate yields NOTHING, and §6's
;; `baseline-cells` stays available for seeing what the space actually holds.
(= (baseline-cell $space $outcome $band $sex)
   (let $cells (collapse
                  (match $space
                     (, (BaseOutcome $bid $outcome)
                        (BaseAgeBand $bid $band)
                        (BaseSex     $bid $sex))
                     $bid))
      (if (== $cells ())
          (superpose ())                                   ; no record -> nothing
          ;; `==` does not evaluate its arguments, so the tail is let-forced first;
          ;; comparing an unreduced (cdr-atom …) would make even a single record decline.
          (let $rest (cdr-atom $cells)
             (if (== $rest ())
                 (car-atom $cells)                         ; exactly one -> use it
                 (superpose ()))))))                       ; ambiguous -> nothing

;; The data-backed baseline absolute risk for (outcome, age, sex), or NOTHING.
;;
;; $age must already be a concrete number: `<` does not evaluate its arguments, so a
;; caller passing an unforced (patient-age …) would silently compare an expression.
;; pln_risk_prediction.metta's `patient-baseline` let*-forces both age and sex first.
;;
;; MULTIPLICITY: one value per matching record. Loading two cycle vintages for the
;; same cell therefore yields TWO candidates, and the consumer's collapse-then-car
;; would pick one arbitrarily. Do not load two vintages into one space; §6's
;; `baseline-cells` exists so that condition can be checked rather than guessed at.
(= (baseline-risk-data $space $outcome $age $sex)
   (let* (($band (baseline-age-band $age))
          ($bid  (baseline-cell $space $outcome $band $sex)))
      (match $space (BaseRisk $bid $r) $r)))

;; The horizon the number above is a risk OVER, or nothing. Mandatory in Schema B, so
;; a record missing it is malformed and yields nothing rather than a bare risk.
(= (baseline-horizon-months $space $outcome $age $sex)
   (let* (($band (baseline-age-band $age))
          ($bid  (baseline-cell $space $outcome $band $sex)))
      (match $space (BaseHorizonMonths $bid $h) $h)))

;; Audit accessor: every record id that answers this (outcome, age, sex) query, as a
;; tuple. () means no data-backed cell; a tuple of length > 1 means the space holds
;; more than one candidate for one cell (see MULTIPLICITY above).
(= (baseline-cells $space $outcome $age $sex)
   (let $band (baseline-age-band $age)
      (collapse (baseline-cell $space $outcome $band $sex))))

;; ================================================================
;; 6. THE OUTCOME-IDENTITY GUARD  (contract D21)
;; ================================================================
;; The risk model multiplies a BASELINE by a HAZARD RATIO. Both are outcome-specific,
;; and the failure this guard prevents is pairing them across outcomes: a
;; HeartDiseaseMortality baseline × the incident-CHD HR 1.07, or a curated
;; CHD-incidence baseline × the all-cause-mortality HR 1.10.
;;
;; The guard is STRUCTURAL, not documentary. There is exactly ONE $outcome variable
;; in this rule and it is used for BOTH lookups, so the rule cannot be called with a
;; baseline outcome and a hazard outcome that differ — there is no second slot to put
;; the second outcome in. When either side has no record for that one outcome, its
;; `match` yields nothing and the whole let* yields nothing, so nothing is multiplied:
;;
;;   AllCauseMortality      -> data baseline ✓ , Lu 2019 HR 1.10 ✓  -> inputs returned
;;   HeartDiseaseMortality  -> data baseline ✓ , HR for it         ✗ -> NOTHING
;;   CoronaryHeartDisease   -> data baseline ✗ , HR 1.07 ✓           -> NOTHING here
;;                             (the curated prior serves CHD, in pln_risk_prediction)
;;
;; The same single-$outcome discipline is what pln_risk_prediction.metta's
;; `absolute-risk-at` relies on: it threads its one $outcome into `patient-baseline`
;; and into `risk-multiplier-at`, so the pairing is fixed by the call, not by a
;; convention a caller could forget.
;;
;; The hazard ratio is read through the risk layer's `clock-hr`, never hand-typed and
;; never re-matched here, so there is one definition of "the hazard for this outcome".
;;
;; The record id is resolved ONCE and both fields are read from THAT id, rather than
;; calling `baseline-risk-data` and `baseline-horizon-months` separately. Those are two
;; independent matches, so with two candidate records in the space they cross-produce
;; and can report one record's risk beside another record's horizon — a 96-month risk
;; labelled as a 120-month one. Same failure as D21, one level down: a number and the
;; thing that makes it interpretable must come from the same row.
;;   (RiskInputs <outcome> (baseline r) (horizon-months h) (hazard-ratio hr))
(= (outcome-matched-risk-inputs $space $outcome $age $sex)
   (let* (($band (baseline-age-band $age))
          ($bid  (baseline-cell $space $outcome $band $sex))
          ($r0   (match $space (BaseRisk          $bid $r)  $r))
          ($h    (match $space (BaseHorizonMonths $bid $hm) $hm))
          ($hr   (clock-hr $space $outcome)))
      (RiskInputs $outcome (baseline $r0) (horizon-months $h) (hazard-ratio $hr))))

;; ================================================================
;; 7. Smoke-test queries (shapes only — this file ships no numbers, so a live
;;    example needs BaselineRiskRecord atoms supplied by the caller or by
;;    build/nhanes_baseline.metta; tests build them from synthetic fixtures)
;; ================================================================
;; !(baseline-age-band 58)                                    ; => Age_50_59
;; !(baseline-age-band 70)                                     ; => Age_70p
;;
;; With a synthetic AllCauseMortality / Male / Age_50_59 record loaded:
;; !(baseline-risk-data &self AllCauseMortality 58 Male)       ; => its (BaseRisk …)
;; !(baseline-cells     &self AllCauseMortality 58 Male)       ; => (<one record id>)
;; !(outcome-matched-risk-inputs &self AllCauseMortality 58 Male)
;;   => (RiskInputs AllCauseMortality (baseline …) (horizon-months …)
;;         (hazard-ratio 1.10))                                 ; HR from Lu 2019
;;
;; The guard, with the SAME record loaded:
;; !(baseline-risk-data &self CoronaryHeartDisease 58 Male)    ; => (empty)  — no CHD
;;                                                            ;    cell exists, ever
;; !(outcome-matched-risk-inputs &self HeartDiseaseMortality 58 Male)
;;   => (empty)   — a baseline exists, but no hazard ratio was ever fit to this
;;                  outcome, so nothing is multiplied by anything
