; ======================================================================================================
; GENERATED FILE -- do not edit.  scripts/make_pln_restricted.py writes it from lib_pln_fork.metta.
; Source: vendor/pln_fork/lib_pln_fork.metta, sha256 4b01b7c66c5294533e6a2496eb20958d6600d62fab40ce53e361e6190cfaa018
; Rule set: Revision, ModusPonens, NotElimination.  45 of 82 top-level forms kept, 37 dropped
; (consistency checks, STV, Deduction, Induction, Abduction, both Inversions, Equivalence, Transitive
; Similarity, Evaluation-Implication, Member-Deduction, Symmetric Modus Ponens, the three
; Evaluation-with-Inheritance rules).  Every kept form is byte-identical to the vendored one;
; tests/test_pln_vendor.py regenerates this file and compares.
; ======================================================================================================

(= (clamp $v $min $max)
   (min $max (max $v $min)))

;;Concat tuples
(= (TupleConcat $Ev1 $Ev2)
   (append $Ev1 $Ev2))

;;Count size of tuple
(= (TupleCount $tuple)
   (length $tuple))

;Shortcut for and with 5 args
(= (and5 $0 $1 $2 $3 $4)
   (and $0 (and $1 (and $2 (and $3 $4)))))

;Shortcut for min with 5 args
(= (min5 $0 $1 $2 $3 $4)
   (min $0 (min $1 (min $2 (min $3 $4)))))

;Exclude division by 0
(= (/safe $A $B)
   (if (> $B 0.0)
       (/ $A $B)
       (empty)))

;Negate (1 minus)
(= (negate $arg)
   (- 1.0 $arg))

;Invert (1 over)
(= (invert $arg)
   (/safe 1.0 $arg))

;;Test case
(= (Test2 $Is $Should)
   ((Is: $Is) (Should: $Should) (Passed: (== $Is $Should))))

;; InsertionSort helper: insert element x into sorted list L
(= (InsertSorted $x $L)
   (if (== $L ())
       ($x)
       (let* (($head (car-atom $L))
              ($tail (cdr-atom $L)))
             (if (< $x $head)
                 (TupleConcat ($x $head) $tail)
                 (TupleConcat ($head) (InsertSorted $x $tail))))))

;; InsertionSort main
(= (InsertionSort $L $Ret)
   (msort $L))

;;Exclude item from tuple
(= (Without $Tuple $a)
   (exclude-item $a $Tuple))

;;Whether a is an element of the tuple
(= (ElementOf $a $Tuple)
   (is-member $a $Tuple))

;;Fast but TODO use efficient native handling once available: https://github.com/trueagi-io/metta-morph/pull/27
(= (Unique $L $Ret)
   (list_to_set $L))

(: Truth_c2w (-[semidet]-> Number Number))

(= (Truth_c2w $c)
   (/safe $c (- 1 $c)))

(: Truth_w2c (-[semidet]-> Number Number))

(= (Truth_w2c $w)
   (/safe $w (+ $w 1)))

;Modus Ponens: PLN book "5.7.1 Modus Ponens", page 111:
(= (Truth_ModusPonens (stv $f1 $c1) (stv $f2 $c2)) 
   (stv (+ (* $f1 $f2) (* 0.02 (- 1 $f1)))
        (* $c1 $c2)))

;Revision: PLN Book "5.10.2 A Heuristic Revision Rule for Simple Truth-values", page 116:
(= (Truth_Revision (stv $f1 $c1) (stv $f2 $c2))
   (let* (($w1 (Truth_c2w $c1))
          ($w2 (Truth_c2w $c2))
          ($w (+ $w1 $w2))
          ($f (/safe (+ (* $w1 $f1) (* $w2 $f2)) $w))
          ($c (Truth_w2c $w)))
         (stv (min 1.0 $f)
              (min 1.0 (max (max $c $c1) $c2)))))

; negation, see https://github.com/opencog/pln/blob/master/opencog/pln/rules/wip/negation-introduction.scm#L41
; negation elimination, see https://github.com/opencog/pln/blob/master/opencog/pln/rules/wip/not-elimination.scm#L34
(= (Truth_Negation (stv $s $c))
   (stv (- 1.0 $s) $c))

;Revision
(= (|- ($T $T1) 
       ($T $T2))
   ($T (Truth_Revision $T1 $T2)))

;Modus Ponens
(= (|- ($A $T1) 
       ((Implication $A $B) $T2))
   ($B (Truth_ModusPonens $T1 $T2)))

; negation elimination rule (introduction now handled by Translator), see https://github.com/opencog/pln/blob/master/opencog/pln/rules/wip/not-elimination.scm#L1-L8
(= (|- ((Not $A) $T)) ($A (Truth_Negation $T)))

;;How many derivation steps (task selections)
(= (PLN.Config.MaxSteps) 100)

;;Size of active tasks PQ
(= (PLN.Config.TaskQueueSize) 10)

;;Max beliefs buffer size
(= (PLN.Config.BeliefQueueSize) 100)

;;Whether evidence was just counted once
(= (StampDisjoint $Ev1 $Ev2)
   (== () (collapse (let* (($x (superpose $Ev1))
                           ($y (superpose $Ev2)))
                          (case (== $x $y) ((True True)))))))

;;Concat stamp with sorting
(= (StampConcat $stamp $addition)
   (if (== $addition ())
       $stamp
       (InsertionSort (TupleConcat $stamp $addition) ())))

;;retrieve the best candidate (PQ functionality)
(= (BestCandidate $evaluateCandidateFunction $bestCandidate $tuple)
   (if (== $tuple ())
       $bestCandidate
       (let* (($head (car-atom $tuple))
              ($tail (cdr-atom $tuple)))
             (if (> ($evaluateCandidateFunction $head)
                    ($evaluateCandidateFunction $bestCandidate))
                 (BestCandidate $evaluateCandidateFunction $head $tail)
                 (BestCandidate $evaluateCandidateFunction $bestCandidate $tail)))))

;;candidate evaluation based on priority
(= (PriorityRank (Sentence ($x (stv $f $c)) $Ev1)) $c)

(= (PriorityRank ()) -99999.0)

;;candidate elimination based on negated priority
(= (PriorityRankNeg (Sentence ($x (stv $f $c)) $Ev1)) (- 0.0 $c))

(= (PriorityRankNeg ()) -99999.0)

;;Return limited-sized version of $L (bounded PQ functionality)
(= (LimitSize $L $size)
   (if (< (TupleCount $L) $size)
       $L
       (let $lowestPriorityItem (BestCandidate PriorityRankNeg () $L)
            (LimitSize (Without $L $lowestPriorityItem) $size))))

;;Priority-queue based task ranking deriver with belief buffer
(= (PLN.Derive $Tasks $Beliefs $steps $maxsteps $taskqueuesize $beliefqueuesize)
   (if (or (> $steps $maxsteps) (== $Tasks ()))
       ($Tasks $Beliefs)
       (let (Sentence $x $Ev1) (BestCandidate PriorityRank () $Tasks)
            (let $derivations
                 (collapse (superpose ((let* (((Sentence $y $Ev2) (superpose $Beliefs))
                                              ($stamp (InsertionSort (TupleConcat $Ev1 $Ev2) ())))
                                             (if (StampDisjoint $Ev1 $Ev2)
                                                 (case (superpose ((|- $x $y)
                                                                   (|- $y $x)))
                                                       ((($T $TV) (Sentence ($T $TV) $stamp))))
                                                 (empty)))
                                       (case (|- $x) ((($T3 $TV3) (Sentence ($T3 $TV3) $Ev1)))))))
                 (PLN.Derive (LimitSize (Without (Unique (TupleConcat $Tasks $derivations) ()) (Sentence $x $Ev1)) $taskqueuesize)
                             (LimitSize (Unique (TupleConcat $Beliefs $derivations) ()) $beliefqueuesize)
                             (+ $steps 1)
                             $maxsteps
                             $taskqueuesize
                             $beliefqueuesize)))))

(= (PLN.Derive $Tasks $Beliefs $maxsteps $taskqueuesize $beliefqueuesize)
   (PLN.Derive $Tasks $Beliefs 1 $maxsteps $taskqueuesize $beliefqueuesize))

(= (PLN.Derive $Tasks $Beliefs $maxsteps)
   (PLN.Derive $Tasks $Beliefs $maxsteps (PLN.Config.TaskQueueSize) (PLN.Config.BeliefQueueSize)))

(= (PLN.Derive $Tasks $Beliefs)
   (PLN.Derive $Tasks $Beliefs (PLN.Config.MaxSteps)))

;;Candidate evaluation based on confidence
(= (ConfidenceRank ((stv $f $c) $Ev)) $c)

(= (ConfidenceRank ()) 0)

;;Pose a question of a certain term to the system on some knowledge base
(= (PLN.Query $Tasks $Beliefs $term $maxsteps $taskqueuesize $beliefqueuesize)
   (BestCandidate ConfidenceRank () (collapse (let ($TasksRet $BeliefsRet) (PLN.Derive $Tasks $Beliefs $maxsteps $taskqueuesize $beliefqueuesize)
                                                   (case (superpose $BeliefsRet)
                                                         (((Sentence ($Term $TV) $Ev) (case (== $Term $term)
                                                                                            ((True ($TV $Ev)))))))))))

(= (PLN.Query $kb $term $maxsteps $taskqueuesize $beliefqueuesize)
   (PLN.Query $kb $kb $term $maxsteps $taskqueuesize $beliefqueuesize))

(= (PLN.Query $kb $term $maxsteps)
   (PLN.Query $kb $term $maxsteps (PLN.Config.TaskQueueSize) (PLN.Config.BeliefQueueSize)))

(= (PLN.Query $kb $term)
   (PLN.Query $kb $term (PLN.Config.MaxSteps)))
