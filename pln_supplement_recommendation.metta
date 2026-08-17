;; pln_supplement_recommendation.metta
;; Personalized supplement recommendations  (pipeline.md Demo 6).        [v1]
;;
;; The capstone demo. Demos 1-5 gave us: a signed, uncertainty-quantified transmission
;; (`infer`), a patient grounded in measurements (`MeasuredZ`), and — crucially — the
;; PERSONALIZATION primitive `patient-relevance` (patient_profile.metta): an intervention
;; is more relevant to a patient the more it protectively reduces a marker THAT patient
;; presents elevated. Demo 6 turns that into the pipeline's §4.6 deliverable and does the
;; five things an LLM cannot (pipeline.md §4.6 / Table 2):
;;
;;   (1) TIER by evidence quality, with supplement confidence calibrated BELOW
;;       pharmaceuticals (the Layer-4 honesty of supplement_evidence.metta);
;;   (2) VETO a mechanistically-plausible compound on a negative gold-standard trial
;;       (Resveratrol / ITP_Negative -> NotRecommended, DESPITE a senescence mechanism —
;;       the killer restraint; an LLM recommends it on the mechanism);
;;   (3) PERSONALIZE — the SAME pool recommends differently for a senescence-dominant vs
;;       a metabolic-dominant patient (Berberine appears for the metabolic one, is OMITTED
;;       for the senescence one); relevance reorders, it does not merely amplify;
;;   (4) flag DRUG-SUPPLEMENT INTERACTIONS against the patient's current medications;
;;   (5) determinism + provenance (which elevated markers each recommendation targets).
;;
;; It adds NO new inference power — it is a pure consumer of `patient-relevance` /
;; `relevance-on` / `rank-score` / `infer` and the curated Layer-4 atoms — exactly as
;; pln_intervention_ranking.metta is a pure consumer of `infer`. The ranking machinery
;; (superpose fan-out, collapse, insertion sort) mirrors that file directly.
;;
;; Load AFTER the whole inference + patient + supplement-evidence stack:
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction pln_intervention_ranking
;;   pln_abductive_diagnosis patient_profile pln_counterfactual pln_risk_prediction
;;   supplement_evidence
;;   pln_supplement_recommendation       (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_supplement_recommendation.py. Uses only core MeTTa builtins (match,
;; superpose, collapse, let/let*, car-atom/cdr-atom/cons-atom, +,*,>,==). As in every
;; layer the space $space is threaded explicitly (never a bare &self — a (match &self …)
;; inside an imported module reads only that module's own space, the cross-module gotcha
;; docs/calibration_layer.md §10.1 documents), and every arithmetic / list intermediate
;; is let*-forced (grounded ops do NOT evaluate their arguments — the discipline
;; pln_intervention_ranking.metta §3 documents). Single-rule + `if` keeps the tiering and
;; sort DETERMINISTIC — "identical inputs -> identical output" (pipeline.md Table 2).
;; v1 locks the ARCHITECTURE (relevance × safety score + evidence tier map + veto); the
;; constants (§0) are documented, tunable knobs.

;; ================================================================
;; 0. KNOBS — every tunable constant lives here (calibration §8 discipline)
;; ================================================================

;; (a) Safety -> a multiplicative de-rating of the personalized score. A less-safe
;;     supplement is ranked below an equally-relevant safer one; it does NOT change the
;;     TIER (evidence does that). GenerallyWellTolerated is neutral (1.0). Coarse by
;;     design — the architecture is the point, not these three numbers.
(= (safety-factor GenerallyWellTolerated) 1.0)
(= (safety-factor UseWithCaution)         0.8)
(= (safety-factor NotWellStudied)         0.6)

;; (b) EvidenceCategory -> recommendation tier. This is the tiering authority AND the
;;     negative-evidence veto in one place. ITP_Negative (a well-run NULL — HIGH
;;     confidence in a NEGATIVE result) maps to NotRecommended, overriding any mechanism.
;;     Tier1 requires STRONG human evidence (RCT / multiple trials / ITP-positive);
;;     everything weaker but non-negative is Tier2_Promising — the "preliminary evidence,
;;     discuss with your doctor" band the pipeline's user explicitly asked for (§4.6).
;;     Single-rule per category -> deterministic.
(= (evidence-tier RCT_Human)                Tier1HighConfidence)
(= (evidence-tier MultipleHumanTrials)      Tier1HighConfidence)
(= (evidence-tier ITP_Positive)             Tier1HighConfidence)
(= (evidence-tier SingleHumanTrial)         Tier2Promising)
(= (evidence-tier AnimalStudies_Replicated) Tier2Promising)
(= (evidence-tier Epidemiological)          Tier2Promising)
(= (evidence-tier AnimalStudies_Single)     Tier2Promising)
(= (evidence-tier Preprint)                 Tier2Promising)
(= (evidence-tier InVitro)                  Tier2Promising)
(= (evidence-tier TraditionalUse)           Tier2Promising)
(= (evidence-tier ITP_Negative)             NotRecommended)   ; the veto

;; ================================================================
;; 1. Per-supplement accessors (space threaded explicitly)
;; ================================================================
;; Each is a single curated atom in supplement_evidence.metta, so these are
;; deterministic lookups (one result). A supplement missing an EvidenceLevel /
;; SafetyProfile simply yields nothing here and drops out of the recommendation —
;; we never invent a tier or a safety rating.
(= (supp-evidence-cat $space $supp) (match $space (EvidenceLevel  $supp $cat) $cat))
(= (supp-safety-label $space $supp) (match $space (SafetyProfile $supp $lbl) $lbl))

;; ================================================================
;; 2. Personalized relevance + the targeted markers (provenance)
;; ================================================================
;; RELEVANCE reuses patient-relevance wholesale (patient_profile.metta): the total
;; protective effect a supplement has across the markers THIS patient presents elevated.
;; A supplement that reaches none of them scores 0 and is OMITTED (personalization) —
;; the recommender never lists a supplement irrelevant to the profile in front of it.
;;
;; TARGETS makes that relevance auditable: the patient's elevated markers the supplement
;; protectively reaches (relevance-on > 0 — reuses the patient-layer's per-marker
;; protective score). $obs is let-forced to the CONCRETE tuple before superpose fans it
;; (superpose does not reduce its argument — the grounded-op discipline every layer notes).
(= (supp-targets $space $patient $supp)
   (let $obs (patient-observations $space $patient)
      (collapse
         (let $m (superpose $obs)
            (let $r (relevance-on $space $supp $m)
               (if (> $r 0.0) $m (superpose ())))))))

;; ================================================================
;; 3. One scored, tiered recommendation record  (or nothing if irrelevant)
;; ================================================================
;;   (SuppRec <supp> (tier <T>) (score <sc>) (evidence <cat> <conf>)
;;            (safety <label> <factor>) (Targets (<M> …)))
;; A supplement with relevance 0 for this patient collapses via (superpose ()) — it never
;; appears (no irrelevant recommendation). score = relevance × safety-factor: the safety
;; de-rating orders within a tier but the TIER is set by evidence alone (incl. the veto).
(= (supplement-rec $space $patient $supp)
   (let $rel (patient-relevance $space $patient $supp)
      (if (> $rel 0.0)
          (let* (($cat   (supp-evidence-cat $space $supp))
                 ($conf  (evidence-confidence $cat))
                 ($tier  (evidence-tier $cat))
                 ($lbl   (supp-safety-label $space $supp))
                 ($saf   (safety-factor $lbl))
                 ($score (* $rel $saf))
                 ($tgts  (supp-targets $space $patient $supp)))
             (SuppRec $supp (tier $tier) (score $score)
                      (evidence $cat $conf) (safety $lbl $saf)
                      (Targets $tgts)))
          (superpose ()))))

;; Field accessors (all let*-forced by callers; grounded ops don't evaluate args).
(= (rec-supp  (SuppRec $s $t $sc $e $sf $tg)) $s)
(= (rec-tier  (SuppRec $s (tier $t) $sc $e $sf $tg)) $t)
(= (rec-score (SuppRec $s $t (score $sc) $e $sf $tg)) $sc)

;; ================================================================
;; 4. Descending insertion sort over the SuppRec tuples (score order within a tier)
;; ================================================================
;; Same shape as pln_intervention_ranking.metta's sort, keyed on rec-score. Single-rule
;; + if -> deterministic. Every intermediate forced through let/let*.
(= (rec-ge $x $h) (>= (rec-score $x) (rec-score $h)))
(= (insert-rec $x $sorted)
   (if (== $sorted ())
       (cons-atom $x ())
       (let $h (car-atom $sorted)
          (if (rec-ge $x $h)
              (cons-atom $x $sorted)
              (let* (($t (cdr-atom $sorted)) ($ins (insert-rec $x $t)))
                 (cons-atom $h $ins))))))
(= (sort-recs $xs)
   (if (== $xs ())
       ()
       (let* (($h (car-atom $xs)) ($t (cdr-atom $xs)) ($r (sort-recs $t)))
          (insert-rec $h $r))))

;; ================================================================
;; 5. Tier buckets
;; ================================================================
;; All relevant recommendations for a patient over the candidate pool, collapsed into one
;; tuple. `superpose $cands` fans the (literal) candidate tuple; supplement-rec omits the
;; irrelevant ones; collapse gathers the rest.
(= (all-supp-recs $space $patient $cands)
   (collapse (supplement-rec $space $patient (superpose $cands))))

;; The recs of ONE tier, sorted by score. $recs is the already-collapsed tuple; superpose
;; fans it and the (== (rec-tier …) $tier) test keeps only this tier's entries (== is
;; grounded — rec-tier reduces to a concrete symbol first).
(= (recs-in-tier $recs $tier)
   (collapse
      (let $r (superpose $recs)
         (if (== (rec-tier $r) $tier) $r (superpose ())))))
(= (tier-bucket $recs $tier)
   (sort-recs (recs-in-tier $recs $tier)))

;; ================================================================
;; 6. Interaction flagging against the patient's current medications
;; ================================================================
;; A flag fires when a RECOMMENDED (relevant) supplement has a KB Interaction with a drug
;; the patient is actually taking. Each equation binds a current medication, then matches
;; an Interaction: if the space holds none, the inner `match` yields nothing and that
;; whole branch is empty — so no current meds -> no flags (Patient001), and a med with no
;; interaction -> no flag. Two equations cover both argument orders (Interaction is an
;; unordered clinical fact); hyperon unions the equations, so a fact in either order fires
;; exactly one flag.
;;   (InteractionFlag <supp> <med> <note>)
(= (interaction-flag $space $patient $supp)
   (let* (($med  (match $space (CurrentMedication $patient $m) $m))
          ($note (match $space (Interaction $supp $med $n) $n)))
      (InteractionFlag $supp $med $note)))
(= (interaction-flag $space $patient $supp)
   (let* (($med  (match $space (CurrentMedication $patient $m) $m))
          ($note (match $space (Interaction $med $supp $n) $n)))
      (InteractionFlag $supp $med $note)))

;; All interaction flags over the relevant recommendations (rec-supp of each SuppRec).
(= (supp-interactions $space $patient $recs)
   (collapse
      (let $r (superpose $recs)
         (interaction-flag $space $patient (rec-supp $r)))))

;; ================================================================
;; 7. The driver — the tiered SupplementRecommendation deliverable
;; ================================================================
;; (recommend-supplements <space> <patient> <candidate-tuple>)
;;   -> (SupplementRecommendation <patient>
;;        (Tier1HighConfidence (<sorted SuppRec …>))
;;        (Tier2Promising      (<sorted SuppRec …>))
;;        (NotRecommended      (<SuppRec … — vetoed compounds>))
;;        (Interactions        (<InteractionFlag …>)))
;; $recs is let-forced to the concrete collapsed tuple so every bucket/interaction step
;; sees a reduced tuple (grounded ops don't evaluate their arguments).
(= (recommend-supplements $space $patient $cands)
   (let $recs (all-supp-recs $space $patient $cands)
      (SupplementRecommendation $patient
         (Tier1HighConfidence (tier-bucket $recs Tier1HighConfidence))
         (Tier2Promising      (tier-bucket $recs Tier2Promising))
         (NotRecommended      (tier-bucket $recs NotRecommended))
         (Interactions        (supp-interactions $space $patient $recs)))))

;; ================================================================
;; 8. Patient driver — default the pool to the pipeline §4.6 candidate set
;; ================================================================
;; The five compounds pipeline.md §4.6 recommends over: two Tier-1 (Omega3, Berberine),
;; two Tier-2 (Fisetin, NMN), one vetoed (Resveratrol). A caller can pass any pool.
(= (recommend-supplements-patient $space $patient)
   (recommend-supplements $space $patient
      (Omega3 Berberine Fisetin NMN Resveratrol)))

;; One supplement's standalone recommendation record for a patient (the "should I take
;; X / what tier is X for me?" query). Nothing if X is irrelevant to this patient.
(= (supplement-for-patient $space $patient $supp)
   (supplement-rec $space $patient $supp))

;; ================================================================
;; 9. Smoke-test queries (intended values; live floats carry trailing noise —
;;    tests/test_supplement_recommendation.py compares with tolerance)
;; ================================================================
;; Load order: system_types logical_predicates epistemic_calibration
;;             grim_age_core grim_age_lu2019_evidence evidence_calibration
;;             hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;             mechanistic_bridges pln_deduction pln_intervention_ranking
;;             pln_abductive_diagnosis patient_profile pln_counterfactual
;;             pln_risk_prediction supplement_evidence pln_supplement_recommendation
;;
;; Patient001 (senescence + inflammation): Omega3 -> Tier1; Fisetin, NMN -> Tier2;
;; Resveratrol -> NotRecommended (vetoed despite a senescence mechanism); Berberine
;; OMITTED (metabolic, irrelevant to this patient); no interactions (no meds):
;; !(recommend-supplements-patient &self Patient001)
;;   => (SupplementRecommendation Patient001
;;         (Tier1HighConfidence ((SuppRec Omega3 (tier Tier1HighConfidence) …
;;                                  (evidence MultipleHumanTrials 0.85) …
;;                                  (Targets (CRP DNAmGDF15)))))
;;         (Tier2Promising      ((SuppRec Fisetin …) (SuppRec NMN …)))
;;         (NotRecommended      ((SuppRec Resveratrol (tier NotRecommended) …
;;                                  (evidence ITP_Negative 0.9) …)))
;;         (Interactions        ()))
;;
;; Patient002 (metabolic): Berberine -> Tier1 (relevant via glucose/HbA1c), the others
;; OMITTED (senescence/inflammation markers normal), and the Berberine <-> Metformin
;; interaction FLAGGED against the patient's metformin:
;; !(recommend-supplements-patient &self Patient002)
;;   => (… (Tier1HighConfidence ((SuppRec Berberine …))) …
;;         (Interactions ((InteractionFlag Berberine Metformin "Shared AMPK activation …"))))
;;
;; "Should Patient001 take resveratrol?" — a single-supplement lookup returns the veto:
;; !(supplement-for-patient &self Patient001 Resveratrol)
;;   => (SuppRec Resveratrol (tier NotRecommended) … (evidence ITP_Negative 0.9) …)
;;
;; Berberine is irrelevant to Patient001 -> nothing (never a fabricated placement):
;; !(supplement-for-patient &self Patient001 Berberine)      ; => (empty)
