"""Every query SHAPE the spike runs, as (KB files, extra atoms, '!' expressions) triples.

A shape is what the app would send to the runtime for one request: the routing that api.py applies
(DrugAge, LinAge2, human-evidence, patient or generic stack) is replicated here by calling the repo's
own routing helpers, so a shape's KB is the KB production would load. Nothing here runs a query.

sources
  smoke         the five section-5 queries of pln_deduction.metta (load order and expected answers parsed from it)
  few_shot      the 46 entries of pln_chat/prompts/few_shot_examples.json (entry 7 is a clarification: no program)
  family        the 13 form families of docs/petta_migration/feasibility/10_metta_constructs.md section 6
                (the "13 form families" the translator is taught), one representative program each, built on
                the patient stack; family 05 and 08 carry two programs
  rule_form     the dedicated forms system_prompt.txt rules 11-17 name that no few-shot entry covers
  route         the scoped engines' own query sets (DrugAge linear / metta_sort, CellAge per-row effects)
  canary        one cheap per-stack check, the same ones run_loads.py runs after a load
  probe_print   how numbers, booleans and error cases print (measure.py)
  probe_order   the seven first-answer sites of 01_verdict.md section 2 item 7, with atoms injected so that
                each has more than one candidate (the curated graph has unique paths, so the real KB cannot
                show an order difference by itself)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from kit_common import repo_root, sha256_text
import stacks as S


@dataclass
class Part:
    """One KB load and the expressions run against it."""
    stack: str
    kb_files: list[str]
    extra_atoms: str
    expressions: list[str]                       # each starts with '!'

    def kb_text(self, fixes=None, repo: Optional[Path] = None) -> str:
        """Exactly the text pln_runner._hyperon_run loads: file contents, then extra atoms, joined by newlines."""
        repo = repo or repo_root()
        named = []
        for f in self.kb_files:
            p = Path(f)
            p = p if p.is_absolute() else repo / p
            named.append((f, p.read_text(encoding="utf-8")))
        if fixes:
            from kb_fixes import fix_kb_texts
            named, _ = fix_kb_texts(named, fixes)
        blocks = [t for _, t in named]
        if self.extra_atoms:
            blocks.append(self.extra_atoms)
        return "\n".join(blocks)

    def kb_key(self) -> str:
        return sha256_text(json.dumps([self.kb_files, self.extra_atoms]))


@dataclass
class Shape:
    id: str
    source: str
    nl: str
    program: str
    route: str
    parts: list[Part] = field(default_factory=list)
    index: Optional[int] = None
    skipped: Optional[str] = None
    expected: Optional[str] = None
    confidence_filter: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Shape":
        d = dict(d)
        parts = [Part(**{k: p[k] for k in ("stack", "kb_files", "extra_atoms", "expressions")}) for p in d.pop("parts", [])]
        keep = {k: d[k] for k in ("id", "source", "nl", "program", "route", "index", "skipped", "expected", "confidence_filter", "notes") if k in d}
        return Shape(parts=parts, **keep)


# ── expressions ──────────────────────────────────────────────────────────────

def expressions_of(program: str) -> list[str]:
    """The '!'-prefixed top-level expressions of a program, exactly as pln_runner._normalize_query builds them."""
    S.repo_import()
    from core import pln_runner as R
    return [e if e.startswith("!") else "!" + e for e in R._iter_top_level_exprs(program or "")]


# ── routing (api.py's, replicated by calling the repo's helpers) ─────────────

def _join(*parts: Optional[str]) -> str:
    return "\n".join(p for p in parts if p)


def _stack_name_of(files: list[str], runtime_stacks: dict[str, list[str]]) -> str:
    for name in ("human_evidence", "patient", "generic"):
        if files == runtime_stacks[name]:
            return name
    return "custom"


def route_program(program: str, caller=None, force_stack: Optional[str] = None) -> tuple[str, list[Part], list[str]]:
    """(route, parts, notes) for a program, the way api.py routes a /query or /metta/run request.

    `caller` is the BuiltPatient whose atoms the request carries (None for the built-in patients).
    `force_stack` pins a program that would run in the generic stack to another stack ('patient'); the note says so.
    """
    S.repo_import()
    import api
    from core import pln_runner as R
    from core.drugage_router import parse_drugage_query
    from core.linage2_router import split_linage2_program
    from core.patient_context import patient_atoms_for

    stacks = S.list_stacks()
    rt = {n: stacks[n].files for n in ("generic", "patient", "human_evidence")}
    notes: list[str] = []

    compounds = parse_drugage_query(program)
    if compounds is not None:
        return "drugage", [drugage_part(compounds, notes)], notes

    split = split_linage2_program(program)
    if split.linage2:
        lin_files = stacks["linage2"].files
        lin_atoms = caller.atoms if caller is not None else ""
        if not split.mixed:
            return "linage2", [Part("linage2", lin_files, lin_atoms, expressions_of(program))], notes
        gen_files = [S._rel(p) for p in api._generic_kb(split.generic)]
        shared = patient_atoms_for(split.generic, caller.shared_atoms if caller is not None else None) or ""
        notes.append("mixed program: two KB loads (LinAge2 space, then the shared stack), run in one worker in production")
        return "linage2+generic", [
            Part("linage2", lin_files, lin_atoms, expressions_of(split.linage2)),
            Part(_stack_name_of(gen_files, rt), gen_files, shared, expressions_of(split.generic)),
        ], notes

    files = [S._rel(p) for p in api._generic_kb(program)]
    name = _stack_name_of(files, rt)
    if force_stack and force_stack != name:
        notes.append(f"pinned to the {force_stack} stack (production would route this program to the {name} stack)")
        files, name = list(stacks[force_stack].files), force_stack
    shared = patient_atoms_for(program, caller.shared_atoms if caller is not None else None) or ""
    return name, [Part(name, files, shared, expressions_of(program))], notes


def drugage_part(compounds: list[str], notes: Optional[list[str]] = None) -> Part:
    """The scoped DrugAge engine's KB and queries for a pool of compounds (run_drugage_ranking, strategy 'linear').

    Rows come from the committed fixture (8 real rows). A compound the fixture lacks is looked up in
    drugage_etl_short.metta (the repo's 201-row sample) when no row id collides.
    """
    S.repo_import()
    from ontology.drugage_selector import SAMPLE_DRUGAGE, load_rows
    notes = notes if notes is not None else []
    spec = S.list_stacks()["drugage_slice"]
    text, rows = S.drugage_slice(compounds)
    have = {r.compound for r in rows}
    missing = [c for c in compounds if c not in have]
    if missing and Path(SAMPLE_DRUGAGE).exists():
        t2, r2 = S.drugage_slice(missing, source=Path(SAMPLE_DRUGAGE))
        ids1 = {str(r.row_id) for r in rows}
        if r2 and not (ids1 & {str(r.row_id) for r in r2}):
            text = _join(text, t2)
            rows = list(rows) + list(r2)
            notes.append(f"{sorted({r.compound for r in r2})} taken from drugage_etl_short.metta (not in the fixture)")
        elif r2:
            notes.append("sample rows dropped: their row ids collide with the fixture's")
    still = sorted(set(compounds) - {r.compound for r in rows})
    if still:
        notes.append(f"no DrugAge row for {still}: they drop out of the ranking, as in production")
    cands = sorted({r.compound for r in rows})
    return Part("drugage_slice", spec.files, text, [f"!(score-candidate &self {c} Mortality)" for c in cands])


# ── the shape catalogue ──────────────────────────────────────────────────────

def _caller_if_named(program: str):
    return S.caller_patient() if "Caller_" in program else None


def _smoke_shapes() -> list[Shape]:
    order, queries = S.parse_smoke()
    out = []
    for i, q in enumerate(queries, 1):
        out.append(Shape(
            id=f"smoke:{i}", source="smoke", index=i, nl=f"pln_deduction.metta section 5, query {i}",
            program=q["expr"], route="smoke10", expected=q["expected"],
            parts=[Part("smoke10", list(order), "", [q["expr"]])],
            notes=["load order and expected answer parsed from the comment block"]))
    return out


def _few_shot_shapes() -> list[Shape]:
    path = repo_root() / "pln_chat" / "prompts" / "few_shot_examples.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, ex in enumerate(data):
        prog = ex.get("metta_query", "")
        sh = Shape(id=f"fs:{i:02d}", source="few_shot", index=i, nl=ex.get("nl", ""), program=prog,
                   route="none", confidence_filter=float(ex.get("confidence_filter") or 0.0))
        if not prog.strip():
            sh.skipped = f"no program (intent={ex.get('intent')}): a clarification, nothing reaches the runtime"
        else:
            sh.route, sh.parts, sh.notes = route_program(prog, _caller_if_named(prog))
            if sh.confidence_filter:
                sh.notes.append(f"confidence_filter {sh.confidence_filter} is applied in Python after the run; the shape records the unfiltered answers")
        out.append(sh)
    return out


#: (id, nl, program). One representative per family of 10_metta_constructs.md section 6, on the patient stack.
FAMILIES: list[tuple[str, str, str]] = [
    ("F01", "match, one pattern, bare template variable",
     "(match &self (MeasuredZ Patient001 $marker $z) $z)"),
    ("F02", "conjunction (, …) of four clauses in ONE match",
     "(match &self (, (InstanceOf $p PatientProfile) (PatientSex $p Male) (PatientAge $p $age) (MeasuredZ $p CRP $z)) $z)"),
    ("F03", "pair and nested pair templates ((pair is defined nowhere)",
     "(match &self (, (PatientAge $p $age) (MeasuredZ $p CRP $z)) (pair $p (pair $age (pair $z (stv 0.5 0.5)))))"),
    ("F04", "if / > / empty evaluated inside a match template",
     "(match &self (MeasuredZ Patient001 $m $z) (if (> $z 1.0) (pair $m $z) (empty)))"),
    ("F05a", "matching an equation as an atom, with an if template",
     "(match &self (= (evidence-confidence $e) $c) (if (> $c 0.7) (pair $e $c) (empty)))"),
    ("F05b", "matching an equation as an atom, a zero-arity knob",
     "(match &self (= (elevated-z-threshold) $v) $v)"),
    ("F06", "expression-headed pattern over a TV-annotated atom (no such atoms exist; still emitted)",
     "(match &self ((Extends $drug $target) (stv $s $c)) (pair $drug (pair $target (stv $s $c))))"),
    ("F07a", "literal constant template True (existence test)",
     "(match &self (PatientSex Patient001 Male) True)"),
    ("F07b", "template that calls a KB function with &self",
     "(match &self (MeasuredZ Patient001 $marker $z) (Status $marker (patient-status &self Patient001 $marker)))"),
    ("F08a", "nested constant arguments with a stored (evidence-confidence …) call in the fact",
     "(match &self (Effect CellularSenescence $b Pos (stv $s $c)) (pair $b (pair $s $c)))"),
    ("F08b", "numeric literal inside a pattern's stv",
     "(match &self (Effect $a $b Pos (stv 0.7 $c)) (pair $a $b))"),
    ("F09", "direct call with &self, a literal tuple argument and an outcome",
     "(rank-interventions-for-patient &self Patient002 (Metformin Berberine CaloricRestriction DasatinibPlusQuercetin) CoronaryHeartDisease)"),
    ("F10", "(rank-drugage-lifespan (…)) intercepted before MeTTa",
     "(rank-drugage-lifespan (Rapamycin Acarbose Resveratrol))"),
    ("F11", "multi-expression program: one per line, a caller-written '!', a comment, an expression with '!' glued",
     "(patient-status &self Patient001 CRP)\n; the second is the Low branch\n!(patient-status &self Patient001 DNAmLeptin)\n(infer &self CellularSenescence CoronaryHeartDisease)"),
    ("F12", "truth values written as (stv s c) literals and combined",
     "(combine-tv (stv 0.9 0.8) (stv 0.5 0.6))"),
    ("F13", "a string answer carrying parentheses, a colon and commas, and PMID-style symbols",
     "(human-evidence-absent &self Berberine)"),
]

#: forms named by system_prompt.txt rules 11-17 that no few-shot entry covers
RULE_FORMS: list[tuple[str, str, str]] = [
    ("R11-smoking", "rule 11: the smoking lever, a patient with DNAmPACKYRS", "(counterfactual-patient &self Patient003 SmokingCessation)"),
    ("R11-marker", "rule 11: a marker as the lever", "(counterfactual-patient &self Patient002 FastingGlucose)"),
    ("R12-smoking", "rule 12: projected risk under the smoking lever", "(project-risk-patient &self Patient003 SmokingCessation)"),
    ("R12-predict3", "rule 12: predict-risk for a second patient", "(predict-risk-patient &self Patient003)"),
    ("R12-decomp2", "rule 12: risk decomposition for a second patient", "(risk-decomposition-patient &self Patient002)"),
    ("R13-one", "rule 13: one supplement for a patient", "(supplement-for-patient &self Patient002 Berberine)"),
    ("R13-pool", "rule 13: an explicit pool", "(recommend-supplements &self Patient002 (Omega3 Berberine Fisetin))"),
    ("R14-metformin", "rule 14: human evidence, records exist", "(human-evidence &self Metformin)"),
    ("R14-list", "rule 14: which interventions have human evidence", "(human-evidence-interventions &self)"),
    ("R14-omega3", "rule 14: cross-referenced elsewhere", "(human-evidence &self Omega3)"),
    ("R14-none", "rule 14: no record", "(human-evidence &self Berberine)"),
    ("R14-mixed", "rule 14: human evidence beside another layer's form (generic stack)",
     "(human-evidence &self Metformin)\n(hallmarks-of &self Metformin)"),
    ("R15-pair", "rule 15: the pair template that aborts the full stack on hyperon",
     "(match &self (TargetsHallmark $i $h) (pair $i $h))"),
    ("R15-one", "rule 15: hallmarks one intervention targets", "(match &self (TargetsHallmark Metformin $h) $h)"),
    ("R15-hallmarks-of", "rule 15: accessor", "(hallmarks-of &self Rapamycin)"),
    ("R15-interventions-for", "rule 15: accessor", "(interventions-for &self CellularSenescence)"),
    ("R15-evidence", "rule 15: López-Otín evidence records, a conjunction",
     "(match &self (, (EvidenceIntervention $r Spermidine) (EvidenceHallmark $r $h)) $h)"),
    ("R16-decomp", "rule 16: LinAge2 decomposition", "(linage-decomposition-patient &self Caller_W58)"),
    ("R16-drivers", "rule 16: LinAge2 drivers", "(linage-drivers-patient &self Caller_W58)"),
    ("R16-risk", "rule 16: absolute risk (needs a baseline: honest empty without one)", "(linage-risk-patient &self Caller_W58)"),
    ("R16-lever", "rule 16: a marker as the LinAge2 lever", "(linage-counterfactual-patient &self Caller_W58 RDW)"),
    ("R16-mixed", "rule 16: a LinAge2 form and a shared-stack form on separate lines (two KB loads)",
     "(linage-decomposition-patient &self Caller_W58)\n(recommend-supplements-patient &self Caller_W58)"),
    ("R17-three", "rule 17: abductive diagnosis with the causes named", "(diagnose-patient &self Patient002 (InsulinResistance ChronicInflammation))"),
    ("R-explain", "explain on a non-patient chain", "(explain &self Metformin CoronaryHeartDisease)"),
    ("R-rank-pool", "rank-interventions on the curated axis, Patient-free", "(rank-interventions &self (Metformin Berberine Spermidine) CoronaryHeartDisease)"),
]


def _family_shapes() -> list[Shape]:
    out = []
    for fid, nl, prog in FAMILIES:
        sh = Shape(id=f"fam:{fid}", source="family", nl=nl, program=prog, route="none")
        # "one representative program per family from the patient stack": the patient stack is the superset of the
        # generic one, so a program production would send to the generic stack is pinned to it. F10 (DrugAge ranking,
        # never executed as MeTTa) and F13 (needs human_evidence.metta, which the patient stack leaves out) keep their routes.
        pin = None if fid in ("F10", "F13") else "patient"
        sh.route, sh.parts, sh.notes = route_program(prog, _caller_if_named(prog), force_stack=pin)
        out.append(sh)
    return out


def _rule_form_shapes() -> list[Shape]:
    out = []
    for rid, nl, prog in RULE_FORMS:
        sh = Shape(id=f"rule:{rid}", source="rule_form", nl=nl, program=prog, route="none")
        sh.route, sh.parts, sh.notes = route_program(prog, _caller_if_named(prog))
        out.append(sh)
    return out


def _route_shapes() -> list[Shape]:
    stacks = S.list_stacks()
    ca_text, ca_rows = S.cellage_slice()
    out = [
        Shape(id="route:drugage-linear", source="route", nl="DrugAge ranking, linear strategy (one score-candidate per compound)",
              program="(rank-drugage-lifespan (Rapamycin Acarbose Resveratrol Metformin))", route="drugage",
              parts=[drugage_part(["Rapamycin", "Acarbose", "Resveratrol", "Metformin"])]),
        Shape(id="route:drugage-metta-sort", source="route", nl="DrugAge ranking, the reference metta_sort strategy (MeTTa insertion sort)",
              program="(rank-interventions &self (Acarbose Metformin Rapamycin) Mortality)", route="drugage",
              parts=[Part("drugage_slice", stacks["drugage_slice"].files, drugage_part(["Acarbose", "Metformin", "Rapamycin"]).extra_atoms,
                          ["!(rank-interventions &self (Acarbose Metformin Rapamycin) Mortality)"])]),
        Shape(id="route:cellage-effects", source="route", nl="CellAge: one cellage-effect per selected row (run_cellage_effects)",
              program="(cellage-effect &self <row>) x rows", route="cellage",
              parts=[Part("cellage_slice", stacks["cellage_slice"].files, ca_text,
                          [f"!(cellage-effect &self {r.row_id})" for r in ca_rows])],
              notes=[f"{len(ca_rows)} rows of tests/fixtures/cellage_real_rows.metta for TP53, CDKN1A, CDKN2A"]),
    ]
    return out


def _canary_shapes() -> list[Shape]:
    out = []
    for name, spec in S.list_stacks().items():
        if not spec.canary or name == "smoke10":
            continue
        out.append(Shape(id=f"canary:{name}", source="canary", nl=f"canary queries of the {name} stack", program="\n".join(spec.canary),
                         route=name, parts=[Part(name, spec.files, spec.extra_atoms, list(spec.canary))]))
    return out


# probes ---------------------------------------------------------------------

#: (id, expression, what it shows). KB: one comment line.
PRINT_PROBES: list[tuple[str, str, str]] = [
    ("int-add", "!(+ 1 2)", "int + int"),
    ("float-mul", "!(* 1.0 3)", "float * int"),
    ("int-div", "!(/ 6 2)", "int / int that divides"),
    ("int-div-frac", "!(/ 1 3)", "int / int that does not"),
    ("float-sum", "!(+ 0.1 0.2)", "shortest round-trip digits"),
    ("float-neg-zero", "!(* -1 0.0)", "negative zero"),
    ("bool-lt", "!(< 1 2)", "comparison result spelling"),
    ("bool-eq-int-float", "!(== 1 1.0)", "structural equality of 1 and 1.0"),
    ("bool-and", "!(and True False)", "boolean literal spelling in, out"),
    ("bool-not", "!(not False)", "not"),
    ("if-true-lit", "!(if True yes no)", "True accepted as a guard"),
    ("log-zero", "!(log-math 10 0)", "log of zero: -inf on hyperon, an evaluation error in SWI"),
    ("log-ten", "!(log-math 10 1000)", "log base 10"),
    ("pow", "!(pow-math 2 0.5)", "pow-math with a float exponent"),
    ("sqrt", "!(sqrt-math 2)", "sqrt-math"),
    ("sci-literal-value", "!(let $x 1e-05 $x)", "1e-05 as a literal: number, or a symbol that prints back as written"),
    ("sci-literal-type", "!(get-metatype 1e-05)", "Grounded or Symbol"),
    ("sci-literal-add", "!(+ 1e-05 1)", "arithmetic on it: works only if it is a number"),
    ("sci-literal-neg", "!(+ 2.0e-75 0)", "mantissa-dot exponent form"),
    ("empty-superpose", "!(superpose ())", "does not compile in PeTTa (translator.pl:1533-1538)"),
    ("empty-atom", "!(empty)", "the replacement the fix uses"),
    ("car-empty", "!(car-atom ())", "hyperon: an Error atom; PeTTa raises car_atom_empty"),
    ("cdr-empty", "!(cdr-atom ())", "cdr-atom of the empty tuple"),
    ("collapse-nothing", "!(collapse (empty))", "collapse of no answers prints ()"),
    ("string-quote", '!(let $s "a (b); c" $s)', "string with a parenthesis and a semicolon"),
    ("unknown-head", "!(no-such-function 1 2)", "an unknown head is data and prints back as written"),
    ("known-no-clause", "!(known-f 3)", "a known function with no matching clause: hyperon returns the call unreduced, PeTTa fails"),
]
_PRINT_KB = "; the print probes need no knowledge base\n(= (known-f 1) one)\n(= (known-f 2) two)"

_ORDER_Q_KB = "(Q a)\n(Q b)\n(Q c)\n(Q d)\n(= (g) 1)\n(= (g) 2)\n(= (g) 3)"
ORDER_PRIMITIVES: list[tuple[str, str]] = [
    ("collapse-match", "!(collapse (match &self (Q $x) $x))"),
    ("car-collapse-match", "!(car-atom (collapse (match &self (Q $x) $x)))"),
    ("collapse-equations", "!(collapse (g))"),
    ("collapse-superpose", "!(collapse (superpose (3 1 2)))"),
    ("collapse-let-fanout", "!(collapse (let $x (superpose (1 2 3)) (* $x 10)))"),
    ("unique-first", "!(unique-atom (b a b c a))"),
    ("collapse-two-vars", "!(collapse (match &self (Q $x) (pair $x $x)))"),
]

#: id, stack, injected atoms, [(label, expression)]; the first expression of each site shows the raw order
ORDER_SITES: list[dict] = [
    {"site": "pos-transmission", "file": "pln_counterfactual.metta:107-109", "stack": "patient",
     "extra": "(Effect CellularSenescence ChronicInflammation Pos (stv 0.8 0.65))",
     "why": "second Pos path CellularSenescence -> ChronicInflammation -> CRP beside the curated one through SASP",
     "exprs": ["!(collapse (let (signed Pos $tv) (infer &self CellularSenescence CRP) $tv))",
               "!(pos-transmission &self CellularSenescence CRP)"]},
    {"site": "resolve-lever/neg", "file": "pln_counterfactual.metta:195-205", "stack": "patient",
     "extra": "(Effect Metformin Autophagy Neg (stv 0.6 0.5))",
     "why": "an intervention that reduces two nodes: car-atom of the Neg targets",
     "exprs": ["!(collapse (match &self (Effect Metformin $d Neg $tv) $d))", "!(resolve-lever &self Metformin)"]},
    {"site": "resolve-lever/inc", "file": "pln_counterfactual.metta:195-205", "stack": "patient",
     "extra": "(Effect InsulinResistance CRP Pos (stv 0.5 0.5))",
     "why": "a leaf marker with two incoming Pos drivers: car-atom of the drivers",
     "exprs": ["!(collapse (match &self (Effect $x CRP Pos $tv) $x))", "!(resolve-lever &self CRP)"]},
    {"site": "cf-component", "file": "pln_counterfactual.metta:278-281", "stack": "patient",
     "extra": "(Effect Metformin InsulinResistance Neg (stv 0.5 0.4))",
     "why": "two lever->driver truth values: car-atom of the lever tvs",
     "exprs": ["!(collapse (match &self (Effect Metformin InsulinResistance Neg $ltv) $ltv))",
               "!(cf-component &self Patient002 Metformin InsulinResistance FastingGlucose)"]},
    {"site": "species-tier-conf", "file": "drugage_calibration.metta:125-130", "stack": "drugage_slice",
     "extra": "(Inheritance Mus_musculus Invertebrate)",
     "why": "a species in two clades: the first decides 0.50 or 0.35",
     "exprs": ["!(collapse (match &self (Inheritance Mus_musculus $clade) $clade))", "!(species-tier-conf &self Mus_musculus)"]},
    {"site": "patient-z", "file": "patient_profile.metta:258-262", "stack": "patient",
     "extra": "(MeasuredZ Patient001 CRP 0.4)",
     "why": "two explicit z values for one marker: car-atom of the explicit ones",
     "exprs": ["!(collapse (match &self (MeasuredZ Patient001 CRP $z) $z))", "!(patient-z &self Patient001 CRP)"]},
    {"site": "patient-baseline", "file": "pln_risk_prediction.metta:257-261", "stack": "linage2",
     "extra": ("(: NB_ORD BaselineRiskRecord)\n(BaseOutcome NB_ORD AllCauseMortality)(BaseSex NB_ORD Male)(BaseAgeBand NB_ORD Age_50_59)\n"
               "(BaseHorizonMonths NB_ORD 120)(BaseRisk NB_ORD 0.2)(BaseRisk NB_ORD 0.3)\n"
               "(BaseEstimator NB_ORD WeightedKaplanMeier)(BaseProvenance NB_ORD NHANES_Microdata)"),
     "why": "one synthetic record carrying two BaseRisk values: car-atom of the data candidates",
     "exprs": ["!(collapse (match &self (BaseRisk NB_ORD $r) $r))",
               "!(patient-baseline &self Patient001 AllCauseMortality)",
               "!(patient-baseline &self Patient001 CoronaryHeartDisease)"]},
    {"site": "human-evidence-absent", "file": "human_evidence.metta:274-281", "stack": "human_evidence",
     "extra": '(HumanEvidenceElsewhere Omega3 SingleHumanTrial "second pointer injected by the order probe")',
     "why": "two cross-references for one intervention: car-atom of the CrossReferenced answers",
     "exprs": ["!(collapse (match &self (HumanEvidenceElsewhere Omega3 $t $w) $t))", "!(human-evidence-absent &self Omega3)"]},
]


def _probe_shapes() -> list[Shape]:
    out = []
    for pid, expr, what in PRINT_PROBES:
        out.append(Shape(id=f"probe:print:{pid}", source="probe_print", nl=what, program=expr, route="probe",
                         parts=[Part("none", [], _PRINT_KB, [expr])]))
    for pid, expr in ORDER_PRIMITIVES:
        out.append(Shape(id=f"probe:order:{pid}", source="probe_order", nl="order of a primitive", program=expr, route="probe",
                         parts=[Part("none", [], _ORDER_Q_KB, [expr])]))
    stacks = S.list_stacks()
    for site in ORDER_SITES:
        spec = stacks[site["stack"]]
        extra = _join(spec.extra_atoms, site["extra"])
        out.append(Shape(id=f"probe:site:{site['site']}", source="probe_order", nl=f"{site['site']} ({site['file']}): {site['why']}",
                         program="\n".join(site["exprs"]), route=site["stack"],
                         parts=[Part(site["stack"], spec.files, extra, list(site["exprs"]))],
                         notes=[f"injected: {site['extra'][:120]}"]))
    return out


def build_shapes(include_probes: bool = True) -> list[Shape]:
    """All shapes, in a stable order."""
    S.repo_import()
    shapes: list[Shape] = []
    shapes += _smoke_shapes()
    shapes += _few_shot_shapes()
    shapes += _family_shapes()
    shapes += _rule_form_shapes()
    shapes += _route_shapes()
    shapes += _canary_shapes()
    if include_probes:
        shapes += _probe_shapes()
    ids = [s.id for s in shapes]
    assert len(ids) == len(set(ids)), "duplicate shape ids"
    return shapes


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="List the shapes (no query is run).")
    ap.add_argument("--source", help="only this source")
    a = ap.parse_args(argv)
    shapes = build_shapes()
    counts: dict[str, int] = {}
    for s in shapes:
        counts[s.source] = counts.get(s.source, 0) + 1
        if a.source and s.source != a.source:
            continue
        n_expr = sum(len(p.expressions) for p in s.parts)
        flag = f"SKIP: {s.skipped}" if s.skipped else f"{len(s.parts)} part(s), {n_expr} expr"
        print(f"{s.id:34s} {s.route:16s} {flag}")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
