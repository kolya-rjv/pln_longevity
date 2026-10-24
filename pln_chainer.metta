;; pln_chainer.metta
;; `infer` and `explain` over the answers patham9's PLN derived.            [PLN_CHAINER=pln only]
;;
;; This file is NEVER part of a stack list and is never read by hyperon. core/pln_chainer.py appends it to the
;; knowledge base of a request that runs with PLN_CHAINER=pln (which needs PLN_RUNTIME=petta), in place of the
;; legacy chaining equations: the `(= (infer ..) ..)` and `(= (explain ..) ..)` equations of pln_deduction.metta
;; and the third `infer` equation of drugage_calibration.metta are filtered out of the text at assembly time,
;; because under PeTTa an extra `=` clause ADDS answers rather than replacing them. The two equations below keep
;; the names, the three arguments and the output shapes of the legacy ones, so no consumer changes:
;;
;;   (infer   <space> A C)  ->  (signed <Pos|Neg> (stv s c))                     0..n, one per route
;;   (explain <space> A C)  ->  (Chain (A n1 .. C) (signed <Pos|Neg> (stv s c)))  0..n, one per route
;;
;; Both read ONE table, the PLNAnswer atoms. Nothing else writes them: before the first query, a step that
;; core/pln_chainer.py runs inside the PeTTa child (after the knowledge base is loaded, before the space is
;; locked) derives, for every source node of the knowledge base, everything patham9's PLN.Derive concludes
;; from "this node is true" (library: vendor/pln_fork/lib_pln_restricted.metta) and adds one atom per route:
;;
;;   (PLNAnswer <source> <target> <Pos|Neg> (stv s c) (<n0> .. <nk>) (<evidence ids>))
;;
;; The answers are precomputed for all sources, not on demand, because consumers bind the source while they
;; evaluate (`(let $h (match $space (Inheritance $hh HallmarkOfAging) $hh) (infer $space $h $b))`), and the
;; space is read-only once a request starts. A PLNAnswer atom that is not the step's own is refused: the
;; validator refuses the head in a caller's text under PLN_CHAINER=pln, and the step refuses a knowledge base
;; that holds one before it writes its own.
;;
;; The step also writes, for audit: (PLNEvidence <id> <kind> "<source>" <tier|None>) -- one per evidence id, the
;; side table of which curated atom, hazard-ratio record, DrugAge entry or caller fact an id stands for -- and
;; (PLNStatus <source> complete|partial <K>) -- whether the derivation for that source finished (the task queue
;; ran empty and the belief queue never filled), with the queue size K it finished at. Under the contract mode a
;; source that cannot finish fails the request, so `partial` is only ever seen in the research lane.
;;
;; Query the provenance:   !(match &self (PLNEvidence $id $kind $where $tier) ($id $kind $where $tier))
;; Query the status:       !(match &self (PLNStatus $source $status $k) ($source $status $k))

(= (infer $s $a $c)
   (match $s (PLNAnswer $a $c $sgn $tv $nodes $ids) (signed $sgn $tv)))

(= (explain $s $a $c)
   (match $s (PLNAnswer $a $c $sgn $tv $nodes $ids) (Chain $nodes (signed $sgn $tv))))
