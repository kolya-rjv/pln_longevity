;; pln_abductive_diagnosis.metta
;; Abductive diagnosis with ranked hypotheses  (pipeline.md Demo 1). [v1]
;;
;; Deduction (pln_deduction.metta) runs the causal graph FORWARD: cause -> effect.
;; Abduction runs the SAME graph BACKWARD: given a set of OBSERVED biomarkers,
;; which upstream causes best explain them? This is the inference pattern behind
;; pipeline.md Demo 1 — a patient presents with elevated markers, and PLN returns
;; ranked cause hypotheses with quantified support and the markers each explains.
;; LLMs can list possible causes but cannot assign calibrated, evidence-weighted
;; support or show which findings each hypothesis accounts for.
;;
;; The whole layer is a CONSUMER of `infer`: a hypothesis H "explains" an observed
;; (elevated) marker O iff forward deduction H -> O is Pos, i.e. H, when elevated,
;; RAISES O. We gather everything a hypothesis explains, then rank hypotheses by
;; explanatory COVERAGE (how many observations) and, as a tiebreak, total evidence
;; mass — the abductive parsimony principle: prefer the single cause that accounts
;; for the most findings.
;;
;; Load AFTER pln_deduction.metta (it calls `infer`) and the cause->biomarker
;; bridges in mechanistic_bridges.metta:
;;   system_types logical_predicates epistemic_calibration
;;   grim_age_core grim_age_lu2019_evidence evidence_calibration
;;   hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;   mechanistic_bridges pln_deduction
;;   pln_abductive_diagnosis            (this file)
;;
;; STATUS: v1, verified against hyperon 0.2.10; regression-guarded in
;; tests/test_pln_deduction.py. The aggregation rules (coverage-then-mass ranking,
;; max-strength, noisy-OR confidence) are deliberately simple and documented —
;; the architecture is locked; the constants are tunable (as in the deduction /
;; calibration layers). Uses only core MeTTa builtins (superpose, collapse,
;; let / let*, car-atom/cdr-atom/cons-atom).

;; ================================================================
;; 1. Does hypothesis H explain observation O?
;; ================================================================
;; H explains O iff forward deduction H -> O is POSITIVE (H raises O), which is
;; consistent with O being observed *elevated*. `let` with the (signed Pos $tv)
;; pattern keeps only Pos derivations — a Neg chain (H would LOWER O) fails the
;; match and yields nothing, so a cause is never credited for a marker it would
;; reduce. `infer` already returns nothing when H and O are unconnected (no false
;; explanation). Result: (explains <O> (stv s c)).
(= (explains-obs $space $h $o)
   (let (signed Pos $tv) (infer $space $h $o)
        (explains $o $tv)))

;; All observations a hypothesis explains, gathered into one tuple via collapse
;; over the observation set. Empty tuple () when it explains nothing.
(= (hyp-evidence $space $h $obs)
   (collapse (explains-obs $space $h (superpose $obs))))

;; ================================================================
;; 2. Aggregate an evidence tuple -> the hypothesis's support
;; ================================================================
;; Folds over the collapsed ((explains O (stv s c)) ...) tuple. Each recursion is
;; let / let*-forced because hyperon grounded ops (car-atom/cdr-atom/cons-atom/==
;; /+/*) do NOT evaluate their arguments — an unforced (cdr-atom ...) is never
;; reduced, so (== ... ()) would never see the empty tuple. (Same discipline the
;; deduction / ranking layers document.)
(= (ev-s (explains $o (stv $s $c))) $s)
(= (ev-c (explains $o (stv $s $c))) $c)
(= (ev-o (explains $o (stv $s $c))) $o)

;; coverage: how many observations the hypothesis explains.
(= (ev-count $ev)
   (if (== $ev ()) 0
       (let $r (ev-count (cdr-atom $ev)) (+ 1 $r))))

;; mass: total supporting evidence, sum of strength*confidence over explanations.
(= (ev-mass $ev)
   (if (== $ev ()) 0.0
       (let* (($h (car-atom $ev)) ($r (ev-mass (cdr-atom $ev))))
          (+ (* (ev-s $h) (ev-c $h)) $r))))

;; reported strength: the strongest single causal pathway (max over explanations).
(= (ev-maxs $ev)
   (if (== $ev ()) 0.0
       (let* (($h (car-atom $ev)) ($s (ev-s $h)) ($r (ev-maxs (cdr-atom $ev))))
          (if (> $s $r) $s $r))))

;; reported confidence: noisy-OR over the per-explanation confidences
;; (1 - prod(1 - c)) — each explained marker is independent corroboration, so
;; confidence in the cause being present RISES as more findings line up.
(= (prob-or $a $b) (- (+ $a $b) (* $a $b)))
(= (ev-orc $ev)
   (if (== $ev ()) 0.0
       (let* (($h (car-atom $ev)) ($r (ev-orc (cdr-atom $ev))))
          (prob-or (ev-c $h) $r))))

;; the list of explained observation names (the provenance / SupportedBy set).
(= (ev-markers $ev)
   (if (== $ev ()) ()
       (let* (($h (car-atom $ev)) ($r (ev-markers (cdr-atom $ev))))
          (cons-atom (ev-o $h) $r))))

;; ================================================================
;; 3. One hypothesis -> its ranked record (or nothing if it explains nothing)
;; ================================================================
;;   (Hypothesis <H> (stv s c) <coverage> <mass> (SupportedBy <markers>))
;; A hypothesis that explains no observation collapses to () and is dropped
;; via (superpose ()) — it never appears in the ranking (no spurious cause).
(= (diagnose-one $space $h $obs)
   (let $ev (hyp-evidence $space $h $obs)
      (if (== $ev ())
          (superpose ())
          (Hypothesis $h
             (stv (ev-maxs $ev) (ev-orc $ev))
             (ev-count $ev)
             (ev-mass $ev)
             (SupportedBy (ev-markers $ev))))))

;; ================================================================
;; 4. Rank hypotheses: coverage first, then evidence mass (abductive parsimony)
;; ================================================================
;; A hypothesis explaining MORE observations outranks one explaining fewer; ties
;; break on total evidence mass. Single-rule + if keeps it deterministic
;; ("identical inputs -> identical output").
(= (hyp-ge (Hypothesis $h1 $tv1 $cov1 $m1 $sb1) (Hypothesis $h2 $tv2 $cov2 $m2 $sb2))
   (if (> $cov1 $cov2) True
       (if (< $cov1 $cov2) False
           (>= $m1 $m2))))

(= (insert-hyp $x $sorted)
   (if (== $sorted ())
       (cons-atom $x ())
       (let $head (car-atom $sorted)
          (if (hyp-ge $x $head)
              (cons-atom $x $sorted)
              (let* (($t (cdr-atom $sorted)) ($ins (insert-hyp $x $t)))
                 (cons-atom $head $ins))))))

(= (sort-hyps $xs)
   (if (== $xs ())
       ()
       (let* (($head (car-atom $xs)) ($t (cdr-atom $xs)) ($r (sort-hyps $t)))
          (insert-hyp $head $r))))

;; ================================================================
;; 5. The driver
;; ================================================================
;; (diagnose <space> <candidate-hypotheses> <observed-markers>)
;;   -> ( (Hypothesis <best>  (stv s c) <cov> <mass> (SupportedBy ...))
;;        (Hypothesis <next>  ...) ... )         best-explaining first.
;; Candidates not connected to any observation are omitted. Both lists are
;; explicit arguments, so a caller passes exactly the patient's findings and the
;; cause space to consider (e.g. the twelve hallmarks of aging).
(= (diagnose $space $hyps $obs)
   (let $cands (collapse (diagnose-one $space (superpose $hyps) $obs))
        (sort-hyps $cands)))

;; ================================================================
;; 6. Smoke-test query (intended values; live floats carry trailing noise —
;;    tests/test_pln_deduction.py compares with tolerance)
;; ================================================================
;; Synthetic patient: elevated DNAmPAI1, DNAmGDF15 and CRP. Rank the hallmark
;; causes that explain this picture:
;;
;; !(diagnose &self
;;     (CellularSenescence MitochondrialDysfunction ChronicInflammation)
;;     (DNAmPAI1 DNAmGDF15 CRP))
;;  =>
;;  ( (Hypothesis CellularSenescence      (stv 0.595 ~0.63) 3 ~0.459
;;        (SupportedBy (DNAmPAI1 CRP DNAmGDF15)))     ; explains all three findings
;;    (Hypothesis ChronicInflammation     (stv 0.80  ~0.88) 2 ~0.943
;;        (SupportedBy (CRP DNAmGDF15)))              ; explains two (and is itself
;;                                                    ;   downstream of senescence)
;;    (Hypothesis MitochondrialDysfunction (stv 0.75 0.65)  1 ~0.488
;;        (SupportedBy (DNAmGDF15))) )                ; explains the lone GDF15 signal
;;
;; CellularSenescence wins on COVERAGE: one root cause accounting for every
;; finding beats causes that each explain a subset — the parsimony an LLM cannot
;; quantify. (Marker order within SupportedBy is collapse-order, not significant.)
