;; pln_intervention_ranking.metta
;; Intervention ranking with uncertainty propagation  (pipeline.md Demo 2). [v1]
;;
;; This is the first COMPLETE demo built on the deduction keystone
;; (pln_deduction.metta). `infer` already gives, for one intervention -> one
;; outcome, a signed, uncertainty-quantified, provenance-bearing estimate. Demo 2
;; is the application on top: take a POOL of candidate interventions, run that
;; inference for each, and RANK them by expected protective effect on a target
;; outcome — the unit deduction_layer.md §6 says an intervention-ranking demo
;; "sorts over". No LLM can produce this: the ordering is a deterministic
;; function of calibrated evidence + chained uncertainty, not a vibe.
;;
;; Why this is the right next increment (and not, say, deeper engine work):
;;   - it harvests the just-landed primitive into a presentation-ready capability;
;;   - it exercises the whole stack end-to-end (calibration -> bridges ->
;;     deduction -> ranking) against real records;
;;   - candidates reach the outcome through chains of DIFFERENT length, so the
;;     ranking visibly reflects "confidence decreases with chain length"
;;     (pipeline.md Table 2) — the senolytics (4 hops) outrank Spermidine, which
;;     reaches the same outcome through a longer autophagy route (5 hops).
;;
;; Load AFTER everything pln_deduction.metta needs, then pln_deduction itself:
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction
;;   pln_intervention_ranking            (this file)
;;
;; STATUS: v1, intended for hyperon 0.2.10; regression-guarded in
;; tests/test_pln_deduction.py. Uses only core MeTTa builtins (superpose,
;; collapse, let, car-atom/cdr-atom/cons-atom) so it needs no extra runtime.

;; ================================================================
;; 1. Protective score: collapse a signed TV to ONE orderable scalar
;; ================================================================
;; Ranking needs a single number, and "better intervention" must mean "more
;; protective". We fold sign and magnitude/confidence together so that:
;;   - a protective (Neg) effect  ->  POSITIVE score  (s * c), and
;;   - a harmful   (Pos) effect   ->  NEGATIVE score  (-(s * c)).
;; So sorting descending puts the most protective intervention first and any
;; net-harmful candidate last (below zero) — the sign is never silently dropped.
;; Magnitude AND confidence both count: a large effect known with near-zero
;; confidence should not outrank a solid, moderate one.
(= (rank-score (signed Neg (stv $s $c))) (* $s $c))
(= (rank-score (signed Pos (stv $s $c))) (- 0.0 (* $s $c)))

;; ================================================================
;; 2. A scored candidate: (scored <intervention> <score> <signed-tv>)
;; ================================================================
;; We keep the full signed TV alongside the scalar so the ranked output is still
;; auditable (magnitude + confidence + direction), not just an opaque order.
(: scored (-> Atom Number Atom Atom))
(= (score-of (scored $intv $score $tv)) $score)

;; Score ONE candidate against the outcome. `infer` is non-deterministic (one
;; result per derivation, NONE if unconnected), and `let` threads each of those
;; results through — so an unconnected candidate yields no `scored` tuple at all
;; (it simply drops out of the ranking; no false placement), and a candidate with
;; several derivations yields one tuple each (v1; PLN revision over multiple paths
;; is the documented follow-up, deduction_layer.md §7.4).
(= (score-candidate $space $intv $outcome)
   (let $r (infer $space $intv $outcome)
        (scored $intv (rank-score $r) $r)))

;; ================================================================
;; 3. Descending insertion sort over the collapsed candidate tuple
;; ================================================================
;; `collapse` (below) turns the non-deterministic stream of scored candidates
;; into one tuple; here we order it. Single-rule + `if` (no overlapping
;; equations) keeps this deterministic — important, since "identical inputs ->
;; identical output" is a headline PLN-over-LLM property (pipeline.md Table 2).
;;
;; NOTE (hyperon 0.2.10): the grounded list ops (car-atom/cdr-atom/cons-atom)
;; and `==` do NOT evaluate their arguments, so every intermediate is forced
;; through `let`/`let*` before it feeds the next step. Skipping this leaves the
;; recursion symbolic — e.g. `(== (cdr-atom ...) ())` is always False because the
;; cdr-atom call is never reduced. (Same cross-module discipline the calibration
;; and deduction layers document.)
(= (insert-scored $x $sorted)
   (if (== $sorted ())
       (cons-atom $x ())
       (let* (($h (car-atom $sorted)) ($t (cdr-atom $sorted)))
          (if (>= (score-of $x) (score-of $h))
              (cons-atom $x $sorted)
              (let $ins (insert-scored $x $t) (cons-atom $h $ins))))))

(= (sort-scored $xs)
   (if (== $xs ())
       ()
       (let* (($h (car-atom $xs)) ($t (cdr-atom $xs)) ($rest (sort-scored $t)))
          (insert-scored $h $rest))))

;; ================================================================
;; 4. The driver
;; ================================================================
;; (rank-interventions <space> <candidate-tuple> <outcome>)
;;   -> ( (scored <best>  <score> (signed Neg (stv s c)))
;;        (scored <next>  ...)
;;        ... )                     most-protective first, unconnected omitted.
;;
;; `superpose` fans the candidate tuple into the non-deterministic stream
;; `score-candidate` consumes; `collapse` gathers every scored result back into
;; one tuple; `sort-scored` orders it. The candidate set is an explicit argument
;; (not every Intervention in the KB) so a caller can rank exactly the pool a
;; patient/clinician is choosing between.
;; `$pool` is forced with `let` so sort-scored receives a CONCRETE collapsed
;; tuple — see the note in §3 on grounded ops not evaluating their arguments.
(= (rank-interventions $space $cands $outcome)
   (let $pool (collapse (score-candidate $space (superpose $cands) $outcome))
        (sort-scored $pool)))

;; ================================================================
;; 5. Smoke-test query (intended values; live floats carry trailing noise —
;;    tests/test_pln_deduction.py compares with tolerance)
;; ================================================================
;; Rank four senescence/autophagy interventions by protective effect on CHD:
;;
;; !(rank-interventions &self
;;     (DasatinibPlusQuercetin Fisetin Spermidine Elamipretide)
;;     CoronaryHeartDisease)
;;   =>
;;   ( (scored DasatinibPlusQuercetin 0.04253 (signed Neg (stv 0.2499  0.17017)))
;;     (scored Fisetin               0.03038 (signed Neg (stv 0.23205 0.13091)))
;;     (scored Spermidine            0.01312 (signed Neg (stv 0.17136 0.07658))) )
;;
;; Elamipretide is omitted: it acts on MitochondrialDysfunction, which has no
;; chain to CHD in the current KB — so PLN simply does not rank it (no false
;; chain), rather than inventing a plausible-sounding placement the way an LLM
;; would. D+Q outranks Fisetin (same mechanism, stronger evidence tier) which
;; outranks Spermidine (one mechanistic hop further upstream -> lower confidence).
