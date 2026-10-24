"""
fork_sim.py -- a pure-Python (standard library only) port of patham9/PLN @ 55f1751
(Formulas.metta + Constraints.metta + Rules.metta + Deriver.metta), plus the
EFFECT of Translator.metta / PLN.Init (see facts/fork_facts.md for the evidence).

Nothing here imports or executes anything from the upstream checkouts.

Data model
----------
atom        : str (symbol; a leading '$' makes it a variable), int, float
expression  : tuple of atoms/expressions, e.g. ('Inheritance', ('Concept','Sam'), ('Concept','Raven'))
tv          : (strength, confidence) tuple      <->  (stv s c)
stamp       : tuple of numbers                  <->  the evidence-id tuple, e.g. (1, 4, 6)
sentence    : (term, tv, stamp)                 <->  (Sentence (term (stv s c)) (ids...))
              (the lowercase user-facing  (sentence ...)  is the same thing, see S())

API
---
    sim = Sim(rules='full', maxsteps=20, task_queue_size=20, belief_queue_size=200,
              static=[...], variant='fork')
    sim.derive(kb)                       -> (tasks, beliefs)         # PLN.Derive
    sim.query(kb, term)                  -> ((s, c), stamp) | None   # PLN.Query ('()' -> None)
    sim.infer2(x, y) / sim.infer1(x)     -> [(term, tv), ...]        # (|- x y) / (|- x)
    sim.init(static)                     # PLN.Init: install the engraved rules / STV table

`static` is a list of sentences exactly like kbstatic: concept (non-Implication)
sentences become  (= (STV term) tv)  facts, Implication sentences become engraved
rules (the Translator's effect).  Which rules are active is chosen by name, see RULE_NAMES.

Rule names (RULE_NAMES): Revision, ModusPonens, SymmetricModusPonens, Deduction, Induction, Abduction,
EvalInhUnary, EvalInhBinary1, EvalInhBinary2, NotElimination, InheritanceInversion, ImplicationInversion,
EquivalenceToImplicationAB/BA, TransitiveSimilarity, EvaluationImplication, MemberDeduction, and
'Translated' (the engraved static implications; it must be named, the presets 'full', 'restricted' and
'restricted+deduction' include it).  Aliases: UsageOfInheritanceForPredicates, EquivalenceToImplication.
Rules named Hist* are historical variants used only by explore_*.py and are not in any preset.

Evaluation-order choices (all documented, all switchable; see facts/fork_facts.md section G)
-------------------------------------------------------------------------------------------
  unique_mode  'fork'  : Unique prepends each first occurrence  -> the list comes out REVERSED
                         (Deriver/Utils.metta:85-92: (Unique rest (TupleConcat ($x) $Ret)) ).
               'keep'  : first-occurrence order kept (PeTTa's bundled lib: list_to_set).
  eq_order     'file'    : equations for |- tried in file order (library rules first, then the
                           engraved ones in kbstatic order; this is also Prolog/PeTTa order).
               'reverse' : newest equation first (what metta-morph's cons-onto-front table
                           suggests; unverified).
  BestCandidate: strict '>' fold, so the FIRST of equal ranks wins.
  Without      : removes every element equal to the argument (order kept).
  LimitSize    : while count >= size remove the first lowest-confidence element, so a queue of
                 size N keeps at most N-1 items.
  order of the two calls: (|- x y) results first, then (|- y x), per belief, in belief order;
                 the unary (|- x) results come after all pair results.
  numbers      : Python floats, same association order as the MeTTa expressions; int/float
                 distinction is kept (repr-based equality), e.g. the (stv 1 0) of a failed
                 Deduction consistency check is an int pair.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Atoms, unification
# --------------------------------------------------------------------------------------


def is_var(a: Any) -> bool:
    return isinstance(a, str) and a[:1] == "$" and len(a) > 1


def walk(t: Any, s: Dict[str, Any]) -> Any:
    while isinstance(t, str) and t in s:
        t = s[t]
    return t


def const_eq(a: Any, b: Any) -> bool:
    # int 1 and float 1.0 are different atoms (as in MeTTa); bool is not used
    return type(a) is type(b) and a == b


def unify(a: Any, b: Any, s: Dict[str, Any]) -> bool:
    """In-place two-way unification (no occurs check, as in MeTTa's atom matching)."""
    a = walk(a, s)
    b = walk(b, s)
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return True
        if is_var(a):
            s[a] = b
            return True
        if is_var(b):
            s[b] = a
            return True
        return False
    if is_var(a):
        s[a] = b
        return True
    if is_var(b):
        s[b] = a
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not unify(x, y, s):
                return False
        return True
    if isinstance(a, tuple) or isinstance(b, tuple):
        return False
    return const_eq(a, b)


def subst(t: Any, s: Dict[str, Any]) -> Any:
    t = walk(t, s)
    if isinstance(t, tuple):
        return tuple(subst(x, s) for x in t)
    return t


def has_var(t: Any) -> bool:
    if isinstance(t, tuple):
        return any(has_var(x) for x in t)
    return is_var(t)


_fresh = itertools.count(1)


def rename_vars(t: Any, tag: str, mapping: Optional[Dict[str, str]] = None) -> Any:
    if mapping is None:
        mapping = {}
    if isinstance(t, tuple):
        return tuple(rename_vars(x, tag, mapping) for x in t)
    if is_var(t):
        if t not in mapping:
            mapping[t] = "$" + tag + t[1:]
        return mapping[t]
    return t


def key(x: Any) -> str:
    """Equality key that distinguishes int from float (repr does)."""
    return repr(x)


# --------------------------------------------------------------------------------------
# Failure of an expression ((empty), or a call with no matching equation)
# --------------------------------------------------------------------------------------


class Fail(Exception):
    """A MeTTa expression that yields no result: (empty), /safe by a non-positive
    denominator, or a call to a function with no matching equation (e.g. (STV X) for a node
    that has no concept sentence; see facts/fork_facts.md section B)."""


def safe_div(a, b):
    """(/safe $A $B) = (if (> $B 0.0) (/ $A $B) (empty))"""
    if b > 0.0:
        return a / b
    raise Fail("/safe")


def div(a, b):
    """Plain (/ a b): hyperon/PeTTa raise on a zero divisor, treated as no result."""
    try:
        return a / b
    except ZeroDivisionError:
        raise Fail("div by zero")


def negate(x):  # (- 1.0 $arg)
    return 1.0 - x


def invert(x):  # (/safe 1.0 $arg)
    return safe_div(1.0, x)


def clamp(v, lo, hi):  # (min $max (max $v $min))
    return min(hi, max(v, lo))


# --------------------------------------------------------------------------------------
# Constraints.metta
# --------------------------------------------------------------------------------------


def smallest_intersection_probability(As, Bs):
    return clamp(div(As + Bs - 1, As), 0, 1)


def largest_intersection_probability(As, Bs):
    return clamp(div(Bs, As), 0, 1)


def conditional_probability_consistency(As, Bs, ABs):
    if not (0 < As):
        return False
    return (smallest_intersection_probability(As, Bs) <= ABs) and (
        ABs <= largest_intersection_probability(As, Bs)
    )


# --------------------------------------------------------------------------------------
# Formulas.metta   (each function: tv args -> tv, or raises Fail)
# --------------------------------------------------------------------------------------


def Truth_c2w(c):
    return safe_div(c, 1 - c)


def Truth_w2c(w):
    return safe_div(w, w + 1)


def Truth_Deduction(P, Q, R, PQ, QR):
    Ps, Pc = P
    Qs, Qc = Q
    Rs, Rc = R
    PQs, PQc = PQ
    QRs, QRc = QR
    if conditional_probability_consistency(Ps, Qs, PQs) and conditional_probability_consistency(Qs, Rs, QRs):
        if 0.9999 < Qs:
            s = Rs
        else:
            s = (PQs * QRs) + safe_div((1 - PQs) * (Rs - (Qs * QRs)), (1 - Qs))
        c = min(Pc, min(Qc, min(Rc, min(PQc, QRc))))
        return (s, c)
    return (1, 0)  # integers, as in (stv 1 0)


def Truth_Induction(A, B, C, BA, BC):
    sA, cA = A
    sB, cB = B
    sC, cC = C
    sBA, cBA = BA
    sBC, cBC = BC
    s = safe_div((sBA * sBC) * sB, sA) + (
        (1 - safe_div(sBA * sB, sA)) * safe_div(sC - (sB * sBC), 1 - sB)
    )
    return (s, Truth_w2c(min(cBA, cBC)))


def Truth_Abduction(A, B, C, AB, CB):
    sA, cA = A
    sB, cB = B
    sC, cC = C
    sAB, cAB = AB
    sCB, cCB = CB
    s = safe_div((sAB * sCB) * sC, sB) + safe_div(sC * ((1 - sAB) * (1 - sCB)), 1 - sB)
    return (s, Truth_w2c(min(cAB, cCB)))


def Truth_ModusPonens(T1, T2):
    f1, c1 = T1
    f2, c2 = T2
    return ((f1 * f2) + (0.02 * (1 - f1)), c1 * c2)


def Truth_SymmetricModusPonens(TA, TAB):
    sA, cA = TA
    sAB, cAB = TAB
    snotAB = 0.2
    cnotAB = 1.0
    return ((sA * sAB) + ((snotAB * negate(sA)) * (1.0 + sAB)), min(min(cAB, cnotAB), cA))


def Truth_Revision(T1, T2):
    f1, c1 = T1
    f2, c2 = T2
    w1 = Truth_c2w(c1)
    w2 = Truth_c2w(c2)
    w = w1 + w2
    f = safe_div((w1 * f1) + (w2 * f2), w)
    c = Truth_w2c(w)
    return (min(1.0, f), min(1.0, max(max(c, c1), c2)))


def Truth_Negation(T):
    s, c = T
    return (1.0 - s, c)


def Truth_inversion(B, AB):
    Bs, Bc = B
    ABs, ABc = AB
    return (ABs, Bc * (ABc * 0.6))


def Truth_equivalenceToImplication(A, B, AB):
    As, Ac = A
    Bs, Bc = B
    ABs, ABc = AB
    if 0.99 < (ABs * ABc):
        concl_s = ABs
    else:
        concl_s = safe_div((1.0 + safe_div(Bs, As)) * ABs, 1.0 + ABs)
    return (concl_s, ABc)


def TransitiveSimilarityStrength(sA, sB, sC, sAB, sBC):
    T1 = div((1.0 + div(sB, sA)) * sAB, 1.0 + sAB)
    T2 = div((1.0 + div(sC, sB)) * sBC, 1.0 + sBC)
    T3 = div((1.0 + div(sB, sC)) * sBC, 1.0 + sBC)
    T4 = div((1.0 + div(sA, sB)) * sAB, 1.0 + sAB)
    left = invert((T1 * T2) + (negate(T1) * safe_div(sC - (sB * T2), negate(sB))))
    right = invert((T3 * T4) + (negate(T3) * safe_div(sC - (sB * T4), negate(sB))))
    return invert((left + right) - 1.0)


def Truth_transitiveSimilarity(A, B, C, AB, BC):
    s = TransitiveSimilarityStrength(A[0], B[0], C[0], AB[0], BC[0])
    return (s, min(AB[1], BC[1]))


def simpleDeductionStrength(sA, sB, sC, sAB, sBC):
    if conditional_probability_consistency(sA, sB, sAB) and conditional_probability_consistency(sB, sC, sBC):
        if 0.99 < sB:
            return sC
        return (sAB * sBC) + safe_div((1.0 - sAB) * (sC - (sB * sBC)), 1.0 - sB)
    raise Fail("simpleDeductionStrength consistency")  # (empty)


def Truth_evaluationImplication(A, B, C, AB, AC):
    As, Ac = A
    Bs, Bc = B
    Cs, Cc = C
    ABs, ABc = AB
    ACs, ACc = AC
    concl_s = simpleDeductionStrength(Bs, As, Cs, ABs, ACs)
    m = min(Bc, min(Ac, min(Cc, min(ACc, 0.9 * ABc))))  # (min5 $Bc $Ac $Cc $ACc (* 0.9 $ABc))
    return (concl_s, (0.9 * 0.9) * m)


def Truth_Identity(T):
    return T


# --------------------------------------------------------------------------------------
# Rules.metta  (+ the engraved rules produced by the Translator)
# --------------------------------------------------------------------------------------

SYLLOGISTIC_LINKS = ("Inheritance", "Implication")
SYMMETRIC_LINKS = ("Similarity", "IntentionalSimilarity", "ExtensionalSimilarity")


class Bind:
    """Resolved view of a unifier for a rule's build function."""

    def __init__(self, s: Dict[str, Any], sim: "Sim"):
        self.s = s
        self.sim = sim

    def __call__(self, v: str):
        return subst(v, self.s)

    def sub(self, t):
        return subst(t, self.s)

    def stv(self, node):
        """(STV node): all matching (= (STV pattern) tv) equations; no equation -> no result."""
        return self.sim.stv_lookup(self.sub(node))


class Rule:
    def __init__(self, name: str, pats: Sequence[Any], build: Callable[[Bind], List[Tuple[Any, Any]]]):
        self.name = name
        self.pats = tuple(pats)
        self.build = build
        self.arity = len(self.pats)


def _stv_product(b: Bind, nodes: Sequence[Any]):
    lists = [b.stv(n) for n in nodes]
    return itertools.product(*lists)


def _library_rules() -> List[Rule]:
    R: List[Rule] = []

    # Revision
    def revision(b):
        return [(b("$T"), Truth_Revision(b("$T1"), b("$T2")))]

    R.append(Rule("Revision", [("$T", "$T1"), ("$T", "$T2")], revision))

    # Modus Ponens
    def mp(b):
        return [(b("$B"), Truth_ModusPonens(b("$T1"), b("$T2")))]

    R.append(Rule("ModusPonens", [("$A", "$T1"), (("Implication", "$A", "$B"), "$T2")], mp))

    # SymmetricModusPonens
    def smp(b):
        if b("$LinkType") in SYMMETRIC_LINKS:
            return [(b("$B"), Truth_SymmetricModusPonens(b("$TruthA"), b("$TruthAB")))]
        return []  # (empty)

    R.append(Rule("SymmetricModusPonens", [("$A", "$TruthA"), (("$LinkType", "$A", "$B"), "$TruthAB")], smp))

    # Deduction
    def ded(b):
        if b("$LinkType") not in SYLLOGISTIC_LINKS:
            return []
        out = []
        for A, B, C in _stv_product(b, (b("$A"), b("$B"), b("$C"))):
            out.append(((b("$LinkType"), b("$A"), b("$C")), Truth_Deduction(A, B, C, b("$T1"), b("$T2"))))
        return out

    R.append(Rule("Deduction", [(("$LinkType", "$A", "$B"), "$T1"), (("$LinkType", "$B", "$C"), "$T2")], ded))

    # Induction
    def ind(b):
        if b("$LinkType") not in SYLLOGISTIC_LINKS:
            return []
        out = []
        for A, B, C in _stv_product(b, (b("$A"), b("$B"), b("$C"))):
            out.append(((b("$LinkType"), b("$A"), b("$B")), Truth_Induction(A, B, C, b("$T1"), b("$T2"))))
        return out

    R.append(Rule("Induction", [(("$LinkType", "$C", "$A"), "$T1"), (("$LinkType", "$C", "$B"), "$T2")], ind))

    # Abduction
    def abd(b):
        if b("$LinkType") not in SYLLOGISTIC_LINKS:
            return []
        out = []
        for A, B, C in _stv_product(b, (b("$A"), b("$B"), b("$C"))):
            out.append(((b("$LinkType"), b("$A"), b("$B")), Truth_Abduction(A, B, C, b("$T1"), b("$T2"))))
        return out

    R.append(Rule("Abduction", [(("$LinkType", "$A", "$C"), "$T1"), (("$LinkType", "$B", "$C"), "$T2")], abd))

    # Usage of inheritance for predicates: unary arg, binary arg1, binary arg2
    R.append(
        Rule(
            "EvalInhUnary",
            [
                (("Evaluation", ("Predicate", "$x"), ("List", ("Concept", "$C"))), "$T1"),
                (("Inheritance", ("Concept", "$S"), ("Concept", "$C")), "$T2"),
            ],
            lambda b: [
                (
                    b.sub(("Evaluation", ("Predicate", "$x"), ("List", ("Concept", "$S")))),
                    Truth_ModusPonens(b("$T1"), b("$T2")),
                )
            ],
        )
    )
    R.append(
        Rule(
            "EvalInhBinary1",
            [
                (("Evaluation", ("Predicate", "$x"), ("List", ("Concept", "$C1"), ("Concept", "$C2"))), "$T1"),
                (("Inheritance", ("Concept", "$S"), ("Concept", "$C1")), "$T2"),
            ],
            lambda b: [
                (
                    b.sub(("Evaluation", ("Predicate", "$x"), ("List", ("Concept", "$S"), ("Concept", "$C2")))),
                    Truth_ModusPonens(b("$T1"), b("$T2")),
                )
            ],
        )
    )
    R.append(
        Rule(
            "EvalInhBinary2",
            [
                (("Evaluation", ("Predicate", "$x"), ("List", ("Concept", "$C1"), ("Concept", "$C2"))), "$T1"),
                (("Inheritance", ("Concept", "$S"), ("Concept", "$C2")), "$T2"),
            ],
            lambda b: [
                (
                    b.sub(("Evaluation", ("Predicate", "$x"), ("List", ("Concept", "$C1"), ("Concept", "$S")))),
                    Truth_ModusPonens(b("$T1"), b("$T2")),
                )
            ],
        )
    )

    # HISTORICAL (not in the HEAD rule set, never part of any preset): the generic negation INTRODUCTION
    # that existed until 7135ece (see `git show 39b771e:src/Rules.metta`, block "negation introduction").
    # It applied to every task.  Kept only to test whether the 'example-specific minimal resource'
    # comments (written at 39b771e) can be re-derived with it (facts/fork_facts.md section F; they cannot).
    # My reading of its guard `(not (== $x (Not $_)))`: structural comparison with a variable atom, so
    # always true and it also negates Not sentences.
    R.append(
        Rule(
            "HistNegationIntroduction@39b771e",
            [("$x", "$t")],
            lambda b: [(("Not", b("$x")), Truth_Negation(b("$t")))],
        )
    )

    # HISTORICAL variant at 768662a (Aug 5): one rule that eliminates a Not or introduces one with half the confidence
    def hist2(b):
        x = b("$x")
        s_, c_ = b("$t")
        if isinstance(x, tuple) and len(x) == 2 and x[0] == "Not":
            return [(x[1], Truth_Negation((s_, c_)))]
        return [(("Not", x), Truth_Negation((s_, 0.5 * c_)))]

    R.append(Rule("HistNegationBoth@768662a", [("$x", "$t")], hist2))

    # Negation elimination (unary)
    R.append(Rule("NotElimination", [(("Not", "$A"), "$T")], lambda b: [(b("$A"), Truth_Negation(b("$T")))]))

    # Inversion (unary)
    def inh_inv(b):
        return [
            (("Inheritance", b("$B"), b("$A")), Truth_inversion(stv, b("$Truth"))) for (stv,) in _stv_product(b, (b("$B"),))
        ]

    R.append(Rule("InheritanceInversion", [(("Inheritance", "$A", "$B"), "$Truth")], inh_inv))

    def imp_inv(b):
        return [
            (("Implication", b("$B"), b("$A")), Truth_inversion(stv, b("$Truth"))) for (stv,) in _stv_product(b, (b("$B"),))
        ]

    R.append(Rule("ImplicationInversion", [(("Implication", "$A", "$B"), "$Truth")], imp_inv))

    # Equivalence to Implication (two equations, same premise)
    def eq_ab(b):
        return [
            (("Implication", b("$A"), b("$B")), Truth_equivalenceToImplication(sa, sb, b("$Truth")))
            for (sa, sb) in _stv_product(b, (b("$A"), b("$B")))
        ]

    def eq_ba(b):
        return [
            (("Implication", b("$B"), b("$A")), Truth_equivalenceToImplication(sa, sb, b("$Truth")))
            for (sa, sb) in _stv_product(b, (b("$A"), b("$B")))
        ]

    R.append(Rule("EquivalenceToImplicationAB", [(("Equivalence", "$A", "$B"), "$Truth")], eq_ab))
    R.append(Rule("EquivalenceToImplicationBA", [(("Equivalence", "$A", "$B"), "$Truth")], eq_ba))

    # Transitive similarity
    def trans_sim(b):
        return [
            (("Similarity", b("$A"), b("$C")), Truth_transitiveSimilarity(A, B, C, b("$T1"), b("$T2")))
            for (A, B, C) in _stv_product(b, (b("$A"), b("$B"), b("$C")))
        ]

    R.append(
        Rule("TransitiveSimilarity", [(("Similarity", "$A", "$B"), "$T1"), (("Similarity", "$B", "$C"), "$T2")], trans_sim)
    )

    # Evaluation Implication
    def eval_imp(b):
        return [
            (("Evaluation", b("$C"), b("$B")), Truth_evaluationImplication(A, B, C, b("$TruthAB"), b("$TruthAC")))
            for (A, B, C) in _stv_product(b, (b("$A"), b("$B"), b("$C")))
        ]

    R.append(
        Rule(
            "EvaluationImplication",
            [(("Evaluation", "$A", "$B"), "$TruthAB"), (("Implication", "$A", "$C"), "$TruthAC")],
            eval_imp,
        )
    )

    # Member deduction
    def mem_ded(b):
        return [
            (("Member", b("$A"), b("$C")), Truth_Deduction(A, B, C, b("$T1"), b("$T2")))
            for (A, B, C) in _stv_product(b, (b("$A"), b("$B"), b("$C")))
        ]

    R.append(Rule("MemberDeduction", [(("Member", "$A", "$B"), "$T1"), (("Inheritance", "$B", "$C"), "$T2")], mem_ded))
    return R


LIBRARY_RULES: List[Rule] = _library_rules()
HISTORICAL_RULE_NAMES: List[str] = [r.name for r in LIBRARY_RULES if r.name.startswith("Hist")]
RULE_NAMES: List[str] = [r.name for r in LIBRARY_RULES if r.name not in HISTORICAL_RULE_NAMES] + ["Translated"]
ALIASES = {
    "UsageOfInheritanceForPredicates": ["EvalInhUnary", "EvalInhBinary1", "EvalInhBinary2"],
    "EquivalenceToImplication": ["EquivalenceToImplicationAB", "EquivalenceToImplicationBA"],
}
PRESETS = {
    "full": RULE_NAMES,
    # decided direction of the migration: no inversion / induction / abduction
    "restricted": ["Revision", "ModusPonens", "NotElimination", "Translated"],
    "restricted+deduction": ["Revision", "ModusPonens", "Deduction", "NotElimination", "Translated"],
}


# ---- the Translator's effect ---------------------------------------------------------


def _is_impl(t) -> bool:
    return isinstance(t, tuple) and len(t) == 3 and t[0] == "Implication"


def _is_not(t) -> bool:
    return isinstance(t, tuple) and len(t) == 2 and t[0] == "Not"


def translate_implication(term, tv, ordinal: int) -> Rule:
    """The equation PLN.Init engraves for one (Implication L1 L2) sentence.

    Translator.metta:11-36.  A `(Not X)` in L1, L3 or L4 is stripped to X and replaced by a
    Truth_Negation on that slot; the derived term is the STRIPPED L2 / L4 (the sign lives only in the
    truth value).  Only ONE level of Not is stripped; an inner Implication as L4 stays a term.
    """
    tag = "t%d_" % ordinal
    L1, L2 = term[1], term[2]
    if _is_impl(L2):
        L3, L4 = L2[1], L2[2]
        n1, n3, n4 = _is_not(L1), _is_not(L3), _is_not(L4)
        m: Dict[str, str] = {}  # one variable mapping per engraved equation
        L1c = rename_vars(L1[1] if n1 else L1, tag, m)
        L3c = rename_vars(L3[1] if n3 else L3, tag, m)
        L4c = rename_vars(L4[1] if n4 else L4, tag, m)

        def build(b: Bind, n1=n1, n3=n3, n4=n4, L4c=L4c, tv=tv):
            tv1, tv3 = b("$TV1"), b("$TV3")
            p1 = Truth_Negation(tv1) if n1 else Truth_Identity(tv1)
            p3 = Truth_Negation(tv3) if n3 else Truth_Identity(tv3)
            inner = Truth_ModusPonens(p1, p3)
            r = Truth_ModusPonens(inner, tv)
            r = Truth_Negation(r) if n4 else Truth_Identity(r)
            return [(b.sub(L4c), r)]

        name = "Translated#%d nested %s" % (ordinal, _fmt_flags(n1, n3, n4))
        return Rule(name, [(L1c, "$TV1"), (L3c, "$TV3")], build)
    n1, n2 = _is_not(L1), _is_not(L2)
    m = {}
    L1c = rename_vars(L1[1] if n1 else L1, tag, m)
    L2c = rename_vars(L2[1] if n2 else L2, tag, m)

    def build(b: Bind, n1=n1, n2=n2, L2c=L2c, tv=tv):
        tv1 = b("$TV1")
        p1 = Truth_Negation(tv1) if n1 else Truth_Identity(tv1)
        r = Truth_ModusPonens(p1, tv)
        r = Truth_Negation(r) if n2 else Truth_Identity(r)
        return [(b.sub(L2c), r)]

    name = "Translated#%d unary %s" % (ordinal, _fmt_flags(n1, n2))
    return Rule(name, [(L1c, "$TV1")], build)


def _fmt_flags(*fl):
    return "".join("N" if f else "I" for f in fl)


# --------------------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------------------


def S(term, s: float, c: float, ev: Iterable[Any] = (0,)):
    """(sentence (term (stv s c)) (ev...)) as (term, (s, c), (ev...))."""
    if not isinstance(ev, tuple):
        ev = tuple(ev)
    return (term, (s, c), ev)


def stamp_disjoint(e1: Tuple, e2: Tuple) -> bool:
    for x in e1:
        for y in e2:
            if x == y:  # (== $x $y) on the evidence ids (numbers)
                return False
    return True


def insertion_sort(xs: Iterable[Any]) -> Tuple:
    """InsertionSort: (< $x $head) puts x before head, so equal elements go after (stable)."""
    out: List[Any] = []
    for x in xs:
        i = 0
        while i < len(out) and not (x < out[i]):
            i += 1
        out.insert(i, x)
    return tuple(out)


def conf(sent) -> float:
    return sent[1][1]


def best_candidate(rank: Callable[[Any], float], items: Sequence[Any], ties: str = "first"):
    """BestCandidate: fold with strict '>' starting from () whose rank is -99999.0.
    (`ties='last'` is an EXPLORATION switch, not the upstream behaviour.)"""
    best = None
    best_rank = -99999.0
    for it in items:
        r = rank(it)
        if r > best_rank or (ties == "last" and r == best_rank):
            best, best_rank = it, r
    return best


def unique(items: Sequence[Any], mode: str) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    if mode == "fork":
        out.reverse()  # (Unique rest (TupleConcat ($x) $Ret)) prepends
    return out


def without(items: Sequence[Any], a: Any) -> List[Any]:
    ka = key(a)
    return [it for it in items if key(it) != ka]


def limit_size(items: List[Any], size: int, stats: Dict[str, int], ties: str = "first") -> List[Any]:
    items = list(items)
    while not (len(items) < size):
        if not items:
            break  # the MeTTa original would not terminate for size <= 0
        # PriorityRankNeg = -confidence; the max of -c (first on ties) is the lowest confidence
        victim = best_candidate(lambda s: -s[1][1], items, ties)
        items = without(items, victim)
        stats["limit_removals"] += 1
    return items


# --------------------------------------------------------------------------------------
# The reasoner
# --------------------------------------------------------------------------------------


class Sim:
    def __init__(
        self,
        rules: Any = "full",
        maxsteps: Optional[int] = None,
        task_queue_size: Optional[int] = None,
        belief_queue_size: Optional[int] = None,
        static: Optional[Sequence[Any]] = None,
        variant: str = "fork",
        unique_mode: Optional[str] = None,
        eq_order: str = "file",
        trace: bool = False,
        explore: Optional[Dict[str, Any]] = None,
    ):
        if variant == "fork":  # src/Config.metta:4,7,10
            d_steps, d_tq, d_bq, d_unique = 20, 20, 200, "fork"
        elif variant == "petta_lib":  # PeTTa/lib/lib_pln.metta:370-376 and list_to_set
            d_steps, d_tq, d_bq, d_unique = 100, 10, 100, "keep"
        else:
            raise ValueError(variant)
        self.variant = variant
        self.maxsteps = d_steps if maxsteps is None else maxsteps
        self.task_queue_size = d_tq if task_queue_size is None else task_queue_size
        self.belief_queue_size = d_bq if belief_queue_size is None else belief_queue_size
        self.unique_mode = d_unique if unique_mode is None else unique_mode
        self.eq_order = eq_order
        # exploration-only switches (defaults = the upstream reading): tie_task/tie_limit in {'first','last'},
        # pair_order in {'xy_yx','yx_xy'}, unary_first in {False, True}
        self.explore = dict(tie_task="first", tie_limit="first", pair_order="xy_yx", unary_first=False)
        self.explore.update(explore or {})
        self.trace_on = trace
        self.set_rules(rules)
        self.stv_table: List[Tuple[Any, Any]] = []
        self.translated: List[Rule] = []
        self.trace: List[dict] = []
        self.stats = self._new_stats()
        if static:
            self.init(static)

    # -- configuration -----------------------------------------------------------------
    def set_rules(self, rules: Any) -> None:
        if isinstance(rules, str):
            rules = PRESETS[rules] if rules in PRESETS else [rules]
        names: List[str] = []
        for n in rules:
            names.extend(ALIASES.get(n, [n]))
        unknown = [n for n in names if n not in RULE_NAMES and n not in HISTORICAL_RULE_NAMES]
        if unknown:
            raise ValueError("unknown rule names %r; known: %r" % (unknown, RULE_NAMES))
        self.rule_names = set(names)

    @staticmethod
    def _new_stats() -> Dict[str, int]:
        return dict(
            selections=0,
            belief_pairs=0,
            disjoint_pairs=0,
            rule_attempts=0,
            rule_matches=0,
            derivations=0,
            limit_removals=0,
            unique_inputs=0,
        )

    def init(self, static: Sequence[Any]) -> None:
        """PLN.Init: concept sentences -> (= (STV term) tv); Implication sentences -> engraved rules."""
        self.stv_table = []
        self.translated = []
        for i, (term, tv, _stamp) in enumerate(static):
            if _is_impl(term):
                self.translated.append(translate_implication(term, tv, i))
            else:
                self.stv_table.append((rename_vars(term, "s%d_" % i), tv))

    def stv_lookup(self, node) -> List[Tuple[float, float]]:
        out = []
        for pat, tv in self.stv_table:
            s: Dict[str, Any] = {}
            if unify(pat, node, s):
                out.append(tv)
        if not out:
            raise Fail("(STV %r): no equation" % (node,))
        return out

    # -- (|- ...) -----------------------------------------------------------------------
    def _active_rules(self, arity: int) -> List[Rule]:
        rs: List[Rule] = [r for r in LIBRARY_RULES if r.name in self.rule_names and r.arity == arity]
        if "Translated" in self.rule_names:
            rs += [r for r in self.translated if r.arity == arity]
        if self.eq_order == "reverse":
            rs.reverse()
        return rs

    def _apply(self, rule: Rule, sents: Sequence[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        self.stats["rule_attempts"] += 1
        s: Dict[str, Any] = {}
        for pat, sent in zip(rule.pats, sents):
            if not unify(pat, sent, s):
                return []
        self.stats["rule_matches"] += 1
        try:
            res = rule.build(Bind(s, self))
        except Fail:
            return []
        return [(subst(t, s), tv) for (t, tv) in res]

    @staticmethod
    def _standardize(x):
        term, tv = x
        if has_var(term):
            return (rename_vars(term, "b%d_" % next(_fresh)), tv)
        return x

    def infer1(self, x: Tuple[Any, Any]) -> List[Tuple[Any, Any]]:
        """(|- x) for x = (term tv)."""
        x = self._standardize(x)
        out: List[Tuple[Any, Any]] = []
        for r in self._active_rules(1):
            out.extend(self._apply(r, [x]))
        return out

    def infer2(self, x: Tuple[Any, Any], y: Tuple[Any, Any]) -> List[Tuple[Any, Any]]:
        """(|- x y) for x, y = (term tv)."""
        x = self._standardize(x)
        y = self._standardize(y)
        out: List[Tuple[Any, Any]] = []
        for r in self._active_rules(2):
            out.extend(self._apply(r, [x, y]))
        return out

    # -- PLN.Derive ----------------------------------------------------------------------
    def derive(self, tasks, beliefs=None, maxsteps=None, task_queue_size=None, belief_queue_size=None):
        if beliefs is None:
            beliefs = tasks
        maxsteps = self.maxsteps if maxsteps is None else maxsteps
        tq = self.task_queue_size if task_queue_size is None else task_queue_size
        bq = self.belief_queue_size if belief_queue_size is None else belief_queue_size
        tasks = list(tasks)
        beliefs = list(beliefs)
        steps = 1
        mode = self.unique_mode
        while True:
            if steps > maxsteps or not tasks:
                return tasks, beliefs
            x = best_candidate(conf, tasks, self.explore["tie_task"])  # PriorityRank = confidence, first of equals
            xt, xtv, ev1 = x
            self.stats["selections"] += 1
            derivs: List[Any] = []
            unary: List[Any] = []
            for (t, tv) in self.infer1((xt, xtv)):
                unary.append((t, tv, ev1))
            if self.explore["unary_first"]:
                derivs.extend(unary)
            for (yt, ytv, ev2) in beliefs:
                self.stats["belief_pairs"] += 1
                if stamp_disjoint(ev1, ev2):
                    self.stats["disjoint_pairs"] += 1
                    stamp = insertion_sort(ev1 + ev2)
                    calls = [((xt, xtv), (yt, ytv)), ((yt, ytv), (xt, xtv))]
                    if self.explore["pair_order"] == "yx_xy":
                        calls.reverse()
                    for (a, b_) in calls:
                        for (t, tv) in self.infer2(a, b_):
                            derivs.append((t, tv, stamp))
            if not self.explore["unary_first"]:
                derivs.extend(unary)
            self.stats["derivations"] += len(derivs)
            if self.trace_on:
                self.trace.append(dict(step=steps, selected=x, derived=list(derivs)))
            self.stats["unique_inputs"] += len(tasks) + len(beliefs) + 2 * len(derivs)
            new_tasks = limit_size(without(unique(tasks + derivs, mode), x), tq, self.stats, self.explore["tie_limit"])
            new_beliefs = limit_size(unique(beliefs + derivs, mode), bq, self.stats, self.explore["tie_limit"])
            tasks, beliefs = new_tasks, new_beliefs
            steps += 1

    # -- PLN.Query -----------------------------------------------------------------------
    def query(self, kb, term, maxsteps=None, task_queue_size=None, belief_queue_size=None, tasks=None):
        """Returns ((strength, confidence), stamp) of the highest-confidence belief whose term equals
        `term` (first of equals), or None for the empty answer ()."""
        _t, beliefs = self.derive(kb if tasks is None else tasks, kb, maxsteps, task_queue_size, belief_queue_size)
        kt = key(term)
        best = None
        best_rank = 0  # (ConfidenceRank ()) = 0, so a candidate with confidence <= 0 never wins
        for (t, tv, ev) in beliefs:
            if key(t) == kt:
                if tv[1] > best_rank:
                    best, best_rank = (tv, ev), tv[1]
        return best


# --------------------------------------------------------------------------------------
# A small s-expression reader (so test files / KBs can be read as DATA)
# --------------------------------------------------------------------------------------


def _tokenize(text: str):
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif ch in " \t\r\n":
            i += 1
        elif ch in "()!":
            yield ch
            i += 1
        elif ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            yield text[i : j + 1]
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in " \t\r\n()":
                j += 1
            yield text[i:j]
            i = j


def _atom(tok: str):
    if tok.startswith('"'):
        return tok
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def parse_sexprs(text: str) -> List[Any]:
    """Parse all top-level forms; '!' before a form is returned as ('!', form)."""
    toks = list(_tokenize(text))
    pos = 0

    def read():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            lst = []
            while toks[pos] != ")":
                lst.append(read())
            pos += 1
            return tuple(lst)
        if t == "!":
            return ("!", read())
        return _atom(t)

    out = []
    while pos < len(toks):
        out.append(read())
    return out


def sentence_from_metta(form) -> Tuple[Any, Any, Tuple]:
    """(sentence|Sentence (term (stv s c)) (ev...)) -> (term, (s, c), (ev...))"""
    head, body, ev = form
    assert head in ("sentence", "Sentence"), form
    term, stv = body
    assert stv[0] == "stv", stv
    return (term, (stv[1], stv[2]), tuple(ev))


def kb_from_metta(form) -> List[Tuple[Any, Any, Tuple]]:
    return [sentence_from_metta(f) for f in form]


def round15(x: float) -> float:
    """Chicken Scheme prints flonums with 15 significant digits; the upstream pins are in that form."""
    return float("%.15g" % x)


if __name__ == "__main__":
    # tiny smoke test: Sam/Raven/Bird, the first FlyingRaven pin
    static = [S(("Concept", n), 0.2, 0.9, (0,)) for n in ("Sam", "Pingu", "Penguin", "Raven", "Bird")]
    sim = Sim(static=static)
    kb = [
        S(("Inheritance", ("Concept", "Sam"), ("Concept", "Raven")), 0.99, 0.9, (1,)),
        S(("Inheritance", ("Concept", "Pingu"), ("Concept", "Penguin")), 0.99, 0.9, (2,)),
        S(("Evaluation", ("Predicate", "flies"), ("List", ("Concept", "Penguin"))), 0.01, 0.9, (3,)),
        S(("Inheritance", ("Concept", "Raven"), ("Concept", "Bird")), 0.99, 0.9, (4,)),
        S(("Inheritance", ("Concept", "Penguin"), ("Concept", "Bird")), 0.99, 0.9, (5,)),
        S(("Evaluation", ("Predicate", "flies"), ("List", ("Concept", "Bird"))), 0.99, 0.9, (6,)),
    ]
    print(sim.query(kb, ("Evaluation", ("Predicate", "flies"), ("List", ("Concept", "Sam")))))
