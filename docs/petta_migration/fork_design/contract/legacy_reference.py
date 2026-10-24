"""An executable specification of the LEGACY chainer (pln_deduction.metta sections 1-4), in plain Python.

It is NOT part of the repo.  It exists so a replacement chainer can be diffed against the legacy numbers
without hyperon, and so this folder's closure.json can be checked: verify_reference.py runs it against
the real hyperon `infer`/`explain` on every ordered node pair and requires bit-identical floats.

Semantics (pln_deduction.metta):
  derive(A, C)  =  [ base(A,C) ]                       one per (Effect A C ...) atom, in atom order   (infer eq.1 / link-effect eq.1)
                 + [ record(A,C) ]                      one per HasPredictor/HasOutcome record pair     (link-effect eq.2)
                 + [ combine(hop(A,B), d)               for every (Effect A B ...) atom, for every d in derive(B, C) ]   (infer eq.2)
  combine((sgn1,(s1,c1)), (sgn2,(s2,c2))) = (sign_product(sgn1,sgn2), (s1*s2, (c1*c2)*0.9))
  record strength = 0.70 if HR>1.5 else 0.60 if HR>1.1 else 0.50   (evidence_calibration.metta:62-64; HR<1 gets 0.50 too)
  record sign     = Neg if HR<1.0 else Pos                           (pln_deduction.metta:111)
  record conf     = evidence-confidence(HasEvidenceCategory record)
No visited set: a cycle reachable from A makes derive() non-terminating (here: RecursionError); legacy hangs.
"""
import re

CHAIN_DISCOUNT = 0.9
TIER = {"RCT_Human": 1.0, "ITP_Positive": 0.90, "ITP_Negative": 0.90, "MultipleHumanTrials": 0.85,
        "SingleHumanTrial": 0.70, "AnimalStudies_Replicated": 0.65, "Epidemiological": 0.60,
        "AnimalStudies_Single": 0.50, "Preprint": 0.40, "InVitro": 0.35, "TraditionalUse": 0.20}
SIGN = {("Pos", "Pos"): "Pos", ("Pos", "Neg"): "Neg", ("Neg", "Pos"): "Neg", ("Neg", "Neg"): "Pos"}
EFFECT_RE = re.compile(r"^\(Effect (\S+) (\S+) (Pos|Neg) \(stv ([-\d.eE+]+) (?:\(evidence-confidence (\w+)\)|([-\d.eE+]+))\)\)$")


def hr_strength(hr):
    return 0.70 if hr > 1.5 else (0.60 if hr > 1.1 else 0.50)


class Legacy:
    def __init__(self, effect_atoms, records):
        """effect_atoms: list of '(Effect A B Pos (stv s (evidence-confidence T)))' strings, in space order.
        records: list of dicts {id, predictor, outcome, hr, category}."""
        self.effects = []
        for e in effect_atoms:
            m = EFFECT_RE.match(e)
            assert m, e
            a, b, sgn, s, tier, lit_c = m.groups()
            c = TIER[tier] if tier else float(lit_c)
            self.effects.append((a, b, sgn, float(s), c))
        self.records = records

    def derive(self, a, c, depth=0):
        if depth > 60:
            raise RecursionError("cycle: legacy infer does not terminate")
        out = []
        for (x, y, sgn, s, cf) in self.effects:                    # link-effect eq.1
            if x == a and y == c:
                out.append(((a, c), sgn, s, cf))
        for r in self.records:                                      # link-effect eq.2
            if r["predictor"] == a and r["outcome"] == c:
                out.append(((a, c), "Neg" if r["hr"] < 1.0 else "Pos", hr_strength(r["hr"]), TIER[r["category"]]))
        for (x, y, sgn, s, cf) in self.effects:                    # infer eq.2: step through a curated Effect
            if x == a:
                for (nodes, sgn2, s2, c2) in self.derive(y, c, depth + 1):
                    out.append(((a,) + nodes, SIGN[(sgn, sgn2)], s * s2, (cf * c2) * CHAIN_DISCOUNT))
        return out
