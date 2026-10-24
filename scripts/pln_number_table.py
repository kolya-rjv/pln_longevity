#!/usr/bin/env python
"""The number table of the PeTTa migration (gate 6a/6b deliverable): the same questions answered in six lanes.

What it is
----------
Every pinned (strength, confidence) and every pinned ordering of the chaining contract
(docs/petta_migration/fork_design/contract/contract.md, sections 4 and 5), asked of the same stack in six lanes:

  legacy (hyperon)   PLN_RUNTIME=hyperon, PLN_CHAINER=legacy       the recorded engine, run now
  legacy (PeTTa)     PLN_RUNTIME=petta,   PLN_CHAINER=legacy       the same equations on PeTTa
  pln-R2             PLN_CHAINER=pln, the default: q = e + (1 - e) p, strength |s1 - s0| / (1 - e)^hops,
                     confidence = the library's c x 0.9^(hops - 1)
  pln-R1             as R2 with the library's own confidence (c1 c2, no discount)         PLN_CHAIN_CONFIDENCE=native
  pln-R0             the fork read literally: q = p, strength |s1 - s0|, native confidence  ChainSettings(mapping="literal")
  pln-full           all 17 rules of the library, uniform node probabilities (0.1, 0.9), every sentence a task, the whole
                     graph as beliefs, fixed budget 150 steps / task queue 100 / belief queue 400, native confidence
                     PLN_RULESET=full (research lane, never reachable from a request)

Next to every number the table prints the cost of the derivation that produced it: `st` selections of the source's level-1
run (PLN.Derive steps; for the full lane the fixed budget, because every run spends it), `tq` and `bq` the queue sizes. The
legacy chainer is a recursion: it has no queues, and its cells say so. Nothing is copied from an earlier note: each lane is run by
this script, in a fresh process, through the production path (`core.pln_runner._runtime_run`, the same call the API makes).

    python scripts/pln_number_table.py lane --lane R2            # one lane, writes lane_R2.json
    python scripts/pln_number_table.py all                       # the six lanes, one subprocess each (about 5 minutes)
    python scripts/pln_number_table.py render                    # JSON -> the generated blocks of NUMBER_TABLE.md

A cell of the document is generated from the lane JSON, whose `_measured_by` is the command above; the block header repeats it.
The only literals in this file are the QUESTIONS (which chains, which pairs), the contract's pins as arithmetic expressions,
and the definitions of the orderings.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
PLN_CHAT = REPO / "pln_chat"
SCRIPTS = REPO / "scripts"
for p in (str(PLN_CHAT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

FORK_DESIGN = REPO / "docs" / "petta_migration" / "fork_design"
CONTRACT = FORK_DESIGN / "contract"
DEFAULT_OUT = FORK_DESIGN / "c" / "results"
DOC = FORK_DESIGN / "NUMBER_TABLE.md"
FIXTURES = REPO / "tests" / "fixtures" / "drugage_fold"

import pln_control_demo as cd                                  # noqa: E402  (the harness, the encoding, the shipped graph)
from core import pln_chainer as pc                             # noqa: E402

#: lane id -> (document label, runtime, chainer, ChainSettings keywords)
LANES: dict[str, dict] = {
    "legacy-hyperon": {"label": "legacy (hyperon)", "runtime": "hyperon", "chainer": "legacy", "settings": {}},
    "legacy-petta": {"label": "legacy (PeTTa)", "runtime": "petta", "chainer": "legacy", "settings": {}},
    "R2": {"label": "pln-R2", "runtime": "petta", "chainer": "pln", "settings": {}},
    "R1": {"label": "pln-R1", "runtime": "petta", "chainer": "pln", "settings": {"chain_confidence": "native"}},
    "R0": {"label": "pln-R0", "runtime": "petta", "chainer": "pln", "settings": {"chain_confidence": "native", "mapping": "literal"}},
    "full": {"label": "pln-full", "runtime": "petta", "chainer": "pln", "settings": {"chain_confidence": "native", "ruleset": "full"}},
}
LANE_ORDER = list(LANES)

# ══════════════════════════════ the questions ═══════════════════════════════════

DQ, CHD, CS = "DasatinibPlusQuercetin", "CoronaryHeartDisease", "CellularSenescence"

#: The five chains test_pln_deduction.py pins (contract.md 4.1; the contract has no "smoke chains" of its own, these are the five
#: chain pins of that file), then four more the contract names. `pin` = (strength literal, confidence expression) as the tests write
#: them, evaluated here; None where the contract pins only emptiness.
CHAINS: list[dict] = [
    {"id": "SASP->CHD", "a": "SASP", "c": CHD, "smoke": True, "sign": "Pos", "pin": ("0.42", "0.65*0.85*0.9"),
     "test": "test_one_hop_propagation"},
    {"id": "CS->CHD", "a": CS, "c": CHD, "smoke": True, "sign": "Pos", "pin": ("0.357", "0.65*0.65*0.85*0.9*0.9"),
     "test": "test_novel_cross_source_inference_harmful"},
    {"id": "D+Q->CHD", "a": DQ, "c": CHD, "smoke": True, "sign": "Neg", "pin": ("0.2499", "0.65**3*0.85*0.9**3"),
     "test": "test_protective_intervention_flips_sign"},
    {"id": "Fisetin->CHD", "a": "Fisetin", "c": CHD, "smoke": True, "sign": "Neg", "pin": ("0.23205", "0.5*0.65*0.65*0.85*0.9**3"),
     "test": "test_fisetin_protective_weaker_than_dq"},
    {"id": "Spermidine->CHD", "a": "Spermidine", "c": CHD, "smoke": True, "sign": "Neg", "pin": ("0.17136", "0.5*0.65**3*0.85*0.9**4"),
     "test": "test_spermidine_longer_chain_lower_confidence"},
    {"id": "Metformin->CHD", "a": "Metformin", "c": CHD, "smoke": False, "sign": "Neg", "pin": ("0.33", "0.85*0.65*0.60*0.9**2"),
     "test": "test_patient_grounding::test_metabolic_axis_reaches_chd"},
    {"id": "Berberine->CHD", "a": "Berberine", "c": CHD, "smoke": False, "sign": "Neg", "pin": ("0.286", "0.85*0.65*0.60*0.9**2"),
     "test": "same mechanism as Metformin (rank_population_5pool)"},
    {"id": "DNAmPAI1->CHD (record)", "a": "DNAmPAI1", "c": CHD, "smoke": False, "sign": "Pos", "pin": ("0.6", "0.85"),
     "test": "test_direct_link_is_calibrated_and_signed"},
    {"id": "D+Q->AllCauseMortality", "a": DQ, "c": "AllCauseMortality", "smoke": False, "sign": None, "pin": None,
     "test": "test_no_false_chain (empty)"},
]

#: The pairs the contract says must return nothing (contract.md 4.1, 4.2, 5.1). `source` may be any node name; an unknown symbol is a node
#: the graph does not have (Elamipretide has no link at all).
NO_FALSE_CHAIN: list[tuple[str, str, str]] = [
    (DQ, "AllCauseMortality", "test_no_false_chain"),
    ("SmokingCessation", CHD, "test_the_smoking_axis_still_does_not_reach_chd_through_infer"),
    ("Rapamycin", CHD, "test_rapamycin_is_still_unreachable_by_inference"),
    ("Elamipretide", CHD, "ranking drops the unconnected candidate"),
    ("CRP", "DNAmGDF15", "no induction: a shared cause does not link its effects"),
    ("Metformin", CS, "no inversion"),
    (CHD, CS, "no inversion"),
    ("ChronicInflammation", CHD, "CHD_REACHING_CAUSES excludes it"),
    ("MitochondrialDysfunction", CHD, "CHD_REACHING_CAUSES excludes it"),
    ("SmokingPackYears", CHD, "CHD_REACHING_CAUSES excludes it"),
    (CS, "AgeAccelGrim", "no Effect reaches AgeAccelGrim"),
    ("DNAmPAI1", "AgeAccelGrim", "no Effect reaches AgeAccelGrim"),
    (CHD, CHD, "self pair"),
]

EXPECTED_FALSE_PAIRS = {"CellularSenescence", "SASP", "DNAmPAI1"}        # sources the decision's reproduction found false conclusions for


# ══════════════════════════════ running a lane ══════════════════════════════════

class Lane:
    """One configured lane: the runtime, the chainer, and the settings the chainer is given."""

    def __init__(self, lane: str, timeout_s: float = 3000.0):
        spec = LANES[lane]
        self.id = lane
        self.spec = spec
        import config
        config.PLN_RUNTIME = spec["runtime"]
        config.PLN_CHAINER = spec["chainer"]
        config.LOGS_DIR = Path(tempfile.mkdtemp(prefix="pln_number_table_logs_"))
        config.PLN_QUERY_TIMEOUT_SECONDS = timeout_s
        self.config = config
        self.pln = spec["chainer"] == "pln"
        self.settings = dict(spec["settings"])
        if self.pln:
            kw = dict(count_selections=spec["settings"].get("ruleset") != "full", **spec["settings"])
            pc.settings_from_config = lambda: pc.ChainSettings(**kw)
            self._kw = kw

    # the stacks, as the API builds them
    def stack(self, name: str) -> list[Path]:
        from core import pln_runner
        import api
        runtime = api._runtime_kb_paths()
        return {"patient": lambda: pln_runner.patient_stack(runtime), "drugage": lambda: list(pln_runner.DRUGAGE_STACK)}[name]()

    def run(self, files: Sequence[Path], extra: Optional[str], queries: Sequence[str], deadline_s: Optional[float] = None) -> dict:
        """One program through the production path: {'groups': [[atoms] per query], 'status', 'error', 'wall_s', 'chain'}."""
        from core import pln_runner
        from core.metta_runtime import get_runtime
        saved = self.config.PLN_QUERY_TIMEOUT_SECONDS
        if deadline_s is not None:
            self.config.PLN_QUERY_TIMEOUT_SECONDS = deadline_s
        t0 = time.perf_counter()
        try:
            result = pln_runner._runtime_run("\n".join(queries), 0.0, list(files), extra)
        finally:
            self.config.PLN_QUERY_TIMEOUT_SECONDS = saved
        wall = time.perf_counter() - t0
        groups: list[list[str]] = [[] for _ in queries]
        for atom in result.results:
            groups[atom.expr_index].append(atom.atom)
        chain = None
        if self.pln:
            try:
                chain = get_runtime("petta").last_run.get("chain")
            except Exception:                                   # noqa: BLE001
                chain = None
        return {"groups": groups, "status": result.status, "error": result.error, "wall_s": round(wall, 3), "chain": chain}


_SIGNED = re.compile(r"^\(signed (Pos|Neg) \(stv (\S+) (\S+)\)\)$")
_CHAIN = re.compile(r"^\(Chain \(([^()]*)\) \(signed (Pos|Neg) \(stv (\S+) (\S+)\)\)\)$")
_SCORED = re.compile(r"\(scored ([^\s()]+) ([^\s()]+) \(signed (Pos|Neg) \(stv ([^\s()]+) ([^\s()]+)\)\)\)")


def parse_signed(atom: str) -> Optional[tuple[str, float, float]]:
    m = _SIGNED.match(atom)
    return (m.group(1), float(m.group(2)), float(m.group(3))) if m else None


def parse_chain(atom: str) -> Optional[tuple[tuple[str, ...], str, float, float]]:
    m = _CHAIN.match(atom)
    return (tuple(m.group(1).split()), m.group(2), float(m.group(3)), float(m.group(4))) if m else None


def parse_scored(atom: str) -> list[dict]:
    return [{"name": m.group(1), "score": float(m.group(2)), "sign": m.group(3), "s": float(m.group(4)), "c": float(m.group(5))}
            for m in _SCORED.finditer(atom)]


def cost_of(chain: Optional[dict], source: str) -> Optional[dict]:
    """The cost of the derivation of `source` as the chainer reported it: selections of the level-1 run, tq, bq, beliefs."""
    if not chain:
        return None
    st = chain.get("status", {}).get(source)
    if not st:
        return None
    sel = st.get("selections")
    return {"st": sel[0] if sel else None, "st0": sel[1] if sel else None, "cap": st["max_steps"], "tq": st["task_queue"], "bq": st["queue"],
            "beliefs": st["beliefs"][0], "complete": st["complete"], "attempts": st["attempts"], "unattributed": st.get("unattributed", 0)}


def cost_of_program(chain: Optional[dict]) -> Optional[dict]:
    """The worst case over every source of a program: the largest selection count, the queue sizes, the largest belief count."""
    if not chain or not chain.get("status"):
        return None
    sts = list(chain["status"].values())
    sel = [s["selections"][0] for s in sts if s.get("selections")]
    return {"st": max(sel) if sel else None, "cap": max(s["max_steps"] for s in sts), "tq": max(s["task_queue"] for s in sts),
            "bq": max(s["queue"] for s in sts), "beliefs": max(s["beliefs"][0] for s in sts), "sources": len(sts),
            "complete": all(s["complete"] for s in sts), "unattributed": sum(s.get("unattributed", 0) for s in sts),
            "derive_runs": chain.get("derive_runs"), "derive_s": chain.get("derive_s")}


# ══════════════════════════════ the patient-stack program ═══════════════════════

def patient_nodes() -> list[str]:
    return json.loads((CONTRACT / "closure.json").read_text(encoding="utf-8"))["stacks"]["patient"]["nodes"]


def composite_entries() -> list[dict]:
    return json.loads((CONTRACT / "pinned_numbers.json").read_text(encoding="utf-8"))["entries"]


def recorded_closure(stack: str = "patient") -> dict[tuple, tuple]:
    """closure.json: {chain: (sign, s, c)}, the real hyperon output of the legacy chainer (the contract study's recording)."""
    d = json.loads((CONTRACT / "closure.json").read_text(encoding="utf-8"))["stacks"][stack]["derivations"]
    return {tuple(r["chain"]): (r["sign"], r["strength"], r["confidence"]) for r in d}


def section_patient(lane: Lane) -> dict:
    """One program on the patient stack: the chains (infer and explain), the no-false-chain pairs, the 50 composite queries, and an
    `explain` from every node (the closure). One program so that the chainer derives the table once per lane."""
    entries = composite_entries()
    nodes = patient_nodes()
    queries: list[str] = []
    index: dict[str, list[int]] = {"composite": [], "infer": [], "explain": [], "nofalse": [], "closure": []}

    def add(kind: str, q: str) -> None:
        index[kind].append(len(queries))
        queries.append(q)

    for e in entries:
        add("composite", e["query"])
    for ch in CHAINS:
        add("infer", f"!(infer &self {ch['a']} {ch['c']})")
    for ch in CHAINS:
        add("explain", f"!(explain &self {ch['a']} {ch['c']})")
    for a, c, _ in NO_FALSE_CHAIN:
        add("nofalse", f"!(infer &self {a} {c})")
    for n in nodes:
        add("closure", f"!(explain &self {n} $x)")
    res = lane.run(lane.stack("patient"), None, queries)
    if res["status"] == "error":
        return {"error": res["error"], "wall_s": res["wall_s"]}
    g = res["groups"]
    out: dict[str, Any] = {"wall_s": res["wall_s"], "program_cost": cost_of_program(res["chain"]), "queries": len(queries)}
    out["composite"] = [{"tag": e["tag"], "query": e["query"], "atoms": g[i]} for e, i in zip(entries, index["composite"])]
    chains = []
    for ch, ii, ei in zip(CHAINS, index["infer"], index["explain"]):
        chains.append({"id": ch["id"], "a": ch["a"], "c": ch["c"], "smoke": ch["smoke"],
                       "infer": [parse_signed(a) for a in g[ii]], "explain": [parse_chain(a) for a in g[ei]],
                       "cost": cost_of(res["chain"], ch["a"]), "pin": _pin(ch)})
    out["chains"] = chains
    out["nofalse"] = [{"a": a, "c": c, "why": why, "infer": [parse_signed(x) for x in g[i]], "cost": cost_of(res["chain"], a)}
                      for (a, c, why), i in zip(NO_FALSE_CHAIN, index["nofalse"])]
    pairs = {}
    for atoms in (g[i] for i in index["closure"]):
        for atom in atoms:
            ch = parse_chain(atom)
            if ch:
                pairs.setdefault(ch[0], []).append(ch[1:])
    out["closure"] = {"pairs": [{"chain": list(k), "results": [list(r) for r in v]} for k, v in sorted(pairs.items())],
                      "unparsed": sum(1 for i in index["closure"] for a in g[i] if not parse_chain(a))}
    out["costs"] = {n: cost_of(res["chain"], n) for n in nodes if cost_of(res["chain"], n)}
    return out


def _pin(ch: dict) -> Optional[dict]:
    if not ch["pin"]:
        return None
    return {"sign": ch["sign"], "s": arithmetic(ch["pin"][0]), "c": arithmetic(ch["pin"][1]), "expr": ch["pin"], "test": ch["test"]}


def arithmetic(expr: str) -> float:
    """The value of a pin written as arithmetic on number literals (`0.65*0.85*0.9`, `0.65**3*0.85*0.9**3`): numbers, + - * / **
    and parentheses, nothing else. A name, a call or an attribute is an error: the pins are data, never code, so nothing here is
    `eval`."""
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}

    def value(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f"not arithmetic on number literals: {expr!r}")
    return value(ast.parse(expr.strip(), mode="eval"))


# ══════════════════════════════ the orderings ═══════════════════════════════════

def _names(atom: str) -> list[str]:
    return re.findall(r"\(scored (\S+) ", atom)


def evaluate_orderings(t: dict[str, list[str]]) -> list[dict]:
    """The twelve orderings of contract.md 4.2 / 5.4, read off the composite outputs of a lane (`t` maps tag -> atoms).
    Each: {'id', 'name', 'holds', 'values'}. A missing output is a failure with the reason, never an exception."""
    DQ_, F, S, M, B = DQ, "Fisetin", "Spermidine", "Metformin", "Berberine"
    out: list[dict] = []

    def add(i: int, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            holds, values = fn()
        except Exception as exc:                                   # noqa: BLE001
            holds, values = False, f"not evaluable: {type(exc).__name__}: {exc}"
        out.append({"id": i, "name": name, "holds": bool(holds), "values": values})

    def first(tag: str) -> str:
        atoms = t.get(tag) or []
        if not atoms:
            raise KeyError(f"{tag} gave no result")
        return atoms[0]

    def short(names: Sequence[str]) -> str:
        abbr = {DQ_: "D+Q"}
        return " > ".join(abbr.get(n, n) for n in names)

    def o1():
        got = _names(first("rank_population"))
        return got == [DQ_, F, S], short(got)

    def o2():
        got = _names(first("rank_population_5pool"))
        return got == [M, B, DQ_, F, S], short(got)

    def o3():
        sc = parse_scored(first("rank_for_Patient001_5pool"))
        got = [x["name"] for x in sc]
        return got == [DQ_, F, M, B, S], short(got)

    def o4():
        sc = parse_scored(first("rank_for_Patient002_5pool"))
        got = [x["name"] for x in sc]
        return got == [M, B, DQ_, F, S], short(got)

    def o5():
        rel = [float(first(f"relevance_Patient001_{n}")) for n in (DQ_, F, S, "Elamipretide")]
        return rel[0] > rel[1] > rel[2] > 0.0 and abs(rel[3]) < 1e-12, "D+Q %.4f, Fisetin %.4f, Spermidine %.4f, Elamipretide %.1g" % tuple(rel)

    def o6():
        m2, b2, m1 = (float(first(k)) for k in ("relevance_Patient002_Metformin", "relevance_Patient002_Berberine", "relevance_Patient001_Metformin"))
        return m2 > 0.4 and b2 > 0.4 and abs(m1) < 1e-12, "P002 Metformin %.4f, P002 Berberine %.4f, P001 Metformin %.1g" % (m2, b2, m1)

    def o7():
        res = []
        ok = True
        for tag in ("diagnose_population", "diagnose_patient_Patient001", "diagnose_patient_default_P001"):
            a = first(tag)
            names = re.findall(r"\(Hypothesis (\S+) ", a)
            cov = [int(float(x)) for x in re.findall(r"\) (\d+(?:\.\d+)?) [\d.e-]+ \(SupportedBy", a)]
            ok = ok and names == [CS, "ChronicInflammation", "MitochondrialDysfunction"] and cov == [3, 2, 1]
            res.append("/".join(n[:2] for n in names) + " " + "".join(map(str, cov)))
        return ok, "; ".join(res)

    def cf(tag: str) -> tuple[float, float]:
        m = re.search(r"expected-delta (-?[\d.e-]+)\) \(signed Neg \(stv [\d.e-]+ ([\d.e-]+)\)", first(tag))
        return float(m.group(1)), float(m.group(2))

    def o8():
        (dA, cA), (dB, cB) = cf("cf_Patient001_ChronicInflammation"), cf("cf_Patient001_CellularSenescence")
        return abs(dB) > abs(dA) and cB < cA <= 0.65 + 1e-12, "|dB| %.4f > |dA| %.4f; c(B) %.4f < c(A) %.4f <= 0.65" % (abs(dB), abs(dA), cB, cA)

    def o9():
        (dB, cB), (dQ, cQ) = cf("cf_Patient001_CellularSenescence"), cf("cf_Patient001_DasatinibPlusQuercetin")
        return abs(dQ / dB - 0.70) < 1e-9 and cQ < cB, "dQ/dB %.6f; c(D+Q) %.4f < c(B) %.4f" % (dQ / dB, cQ, cB)

    def o10():
        sup = first("supplements_Patient001")
        scores = dict(re.findall(r"\(SuppRec (\S+) \(tier \S+\) \(score ([\d.e-]+)\)", sup))
        tier1 = bool(re.search(r"Tier1HighConfidence \(\(SuppRec Omega3 ", sup))
        return tier1 and float(scores["NMN"]) > float(scores["Fisetin"]), "Omega3 tier 1: %s; NMN %.4f vs Fisetin %.4f" % (tier1, float(scores["NMN"]), float(scores["Fisetin"]))

    def proj(tag: str) -> tuple[float, float]:
        m = re.search(r"\(reduction ([\d.e-]+)\).*\(confidence ([\d.e-]+)\)", first(tag))
        return float(m.group(1)), float(m.group(2))

    def o11():
        (rb, cb), (ra, ca), (rc, _) = proj("project_Patient001_CellularSenescence"), proj("project_Patient001_ChronicInflammation"), proj("project_Patient001_InsulinResistance")
        return rb > ra > rc and abs(rc) < 1e-12 and cb < ca, "reductions %.5f > %.5f > %.1g; confidence %.4f < %.4f" % (rb, ra, rc, cb, ca)

    def o12():
        sc = parse_scored(first("rank_population"))
        conf = [x["c"] for x in sc]
        if len(conf) < 3:
            return False, "only %d of the 3 chains answered (%s)" % (len(conf), ", ".join(x["name"] for x in sc) or "none")
        return conf[0] > conf[1] > conf[2] and conf[0] < 0.65, "c: D+Q %.4f > Fisetin %.4f > Spermidine %.4f, all < 0.65" % tuple(conf[:3])

    add(1, "population ranking [D+Q, Fisetin, Spermidine], Elamipretide absent", o1)
    add(2, "5-pool population ranking [Metformin, Berberine, D+Q, Fisetin, Spermidine]", o2)
    add(3, "Patient001 personalised order [D+Q, Fisetin, Metformin, Berberine, Spermidine]", o3)
    add(4, "Patient002 personalised order [Metformin, Berberine, D+Q, Fisetin, Spermidine]", o4)
    add(5, "relevance to Patient001's markers: D+Q > Fisetin > Spermidine > 0 = Elamipretide", o5)
    add(6, "relevance is patient specific: Patient002 Metformin and Berberine > 0.4, Patient001 Metformin = 0", o6)
    add(7, "diagnosis ranks by coverage: [CS, CI, MD] with coverage [3, 2, 1] (population, Patient001, default causes)", o7)
    add(8, "counterfactual: senescence moves the clock more than inflammation and is less certain (c(B) < c(A) <= 0.65)", o8)
    add(9, "senolytic lever = 0.70 of senescence's effect, lowest confidence", o9)
    add(10, "supplements for Patient001: Omega3 in tier 1, NMN scores above Fisetin", o10)
    add(11, "risk: reduction(senescence) > reduction(inflammation) > reduction(insulin resistance) = 0, confidence(senescence) < confidence(inflammation)", o11)
    add(12, "confidence by chain length: D+Q (4 edges) > Fisetin (4) > Spermidine (5), all below a single link's 0.65", o12)
    return out


def diagnose_values(t: dict[str, list[str]]) -> dict:
    """The tolerances of the diagnosis pins: s(CS) = 0.595, c(CS) = 0.625 +- 1e-2 (test_patient_grounding), c(MD) = 0.65 (test_pln_deduction)."""
    a = (t.get("diagnose_population") or [""])[0]
    p = (t.get("diagnose_patient_Patient001") or [""])[0]
    out: dict[str, Any] = {}
    m = re.search(r"\(Hypothesis CellularSenescence \(stv ([\d.e-]+) ([\d.e-]+)\) ([\d.]+) ([\d.e-]+)", a)
    if m:
        out["population"] = {"s_CS": float(m.group(1)), "c_CS": float(m.group(2)), "coverage": float(m.group(3)), "mass_CS": float(m.group(4))}
    md = re.search(r"\(Hypothesis MitochondrialDysfunction \(stv ([\d.e-]+) ([\d.e-]+)\)", a)
    if md:
        out["population"]["c_MD"] = float(md.group(2))
    m = re.search(r"\(Hypothesis CellularSenescence \(stv ([\d.e-]+) ([\d.e-]+)\)", p)
    if m:
        out["Patient001"] = {"s_CS": float(m.group(1)), "c_CS": float(m.group(2))}
    return out


def personalised(t: dict[str, list[str]]) -> dict:
    out = {}
    for label, tag in (("Patient001", "rank_for_Patient001_5pool"), ("Patient002", "rank_for_Patient002_5pool")):
        atoms = t.get(tag) or []
        out[label] = parse_scored(atoms[0]) if atoms else []
    return out


def section_orderings(patient: dict) -> dict:
    t = {e["tag"]: e["atoms"] for e in patient["composite"]}
    return {"orderings": evaluate_orderings(t), "diagnose": diagnose_values(t), "personalised": personalised(t),
            "relevance": {k: t[k][0] for k in t if k.startswith("relevance_") and t[k]},
            "n_groups_with_results": sum(1 for v in t.values() if v)}


# ══════════════════════════════ closure statistics ═════════════════════════════

def closure_stats(pairs: list[dict], reference: dict[tuple, tuple]) -> dict:
    """The lane's pairs against the recorded hyperon closure: missing, extra, deviations, ratios, last-digit differences."""
    got: dict[tuple, tuple] = {}
    dup = 0
    for p in pairs:
        key = tuple(p["chain"])
        if len(p["results"]) > 1:
            dup += 1
        sign, s, c = p["results"][0]
        got[key] = (sign, s, c)
    missing = [k for k in reference if k not in got]
    extra = [k for k in got if k not in reference]
    both = [k for k in reference if k in got]
    ds = [abs(got[k][1] - reference[k][1]) for k in both]
    dc = [abs(got[k][2] - reference[k][2]) for k in both]
    rs = [got[k][1] / reference[k][1] for k in both if reference[k][1] > 0]
    rc = [got[k][2] / reference[k][2] for k in both if reference[k][2] > 0]
    return {"pairs": len(got), "reference": len(reference), "missing": len(missing), "extra": len(extra), "duplicates": dup,
            "sign_differs": sum(1 for k in both if got[k][0] != reference[k][0]),
            "max_ds": max(ds, default=0.0), "max_dc": max(dc, default=0.0),
            "ratio_s": [min(rs), max(rs)] if rs else None, "ratio_c": [min(rc), max(rc)] if rc else None,
            "strength_digits_differ": sum(1 for k in both if got[k][1] != reference[k][1]),
            "confidence_digits_differ": sum(1 for k in both if got[k][2] != reference[k][2]),
            "missing_examples": [list(k) for k in missing[:6]], "extra_examples": [list(k) for k in extra[:6]]}


def hops_range(pairs: list[dict], ref: dict[tuple, tuple]) -> dict:
    """Strength ratio to the recorded closure by number of hops (R0 shrinks long chains most)."""
    by: dict[int, list[float]] = {}
    for p in pairs:
        k = tuple(p["chain"])
        if k in ref and ref[k][1] > 0:
            by.setdefault(len(k) - 1, []).append(p["results"][0][1] / ref[k][1])
    return {str(h): [min(v), max(v)] for h, v in sorted(by.items())}


# ══════════════════════════════ the feedback edge ══════════════════════════════

def section_feedback(lane: Lane) -> dict:
    """The patient stack plus ChronicInflammation -> CellularSenescence (decision section 5): does it terminate, how many pairs."""
    atom = "(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))\n"
    nodes = patient_nodes()
    queries = [f"!(explain &self {n} $x)" for n in nodes]
    if lane.id == "legacy-hyperon":          # hyperon cannot stop a program itself: never run the hanging one in this process
        hang = hang_check("hyperon", 25.0)
        return {"terminates": bool(hang["finished"]), "status": "hang" if not hang["finished"] else "finished", "error": None, "wall_s": hang["wall_s"],
                "hang": hang}
    deadline = 10.0 if not lane.pln else None
    res = lane.run(lane.stack("patient"), atom, queries, deadline_s=deadline)
    out: dict[str, Any] = {"wall_s": res["wall_s"], "status": res["status"], "error": res["error"]}
    if res["status"] == "error":
        out["terminates"] = False
        return out
    pairs: dict[tuple, list] = {}
    for atoms in res["groups"]:
        for a in atoms:
            ch = parse_chain(a)
            if ch:
                pairs.setdefault(ch[0], []).append(list(ch[1:]))
    out["terminates"] = True
    out["pairs"] = [{"chain": list(k), "results": v} for k, v in sorted(pairs.items())]
    out["program_cost"] = cost_of_program(res["chain"])
    out["costs"] = {n: cost_of(res["chain"], n) for n in ("ChronicInflammation", "Omega3", DQ, CS, "SASP", "Spermidine") if cost_of(res["chain"], n)}
    ref = cd.node_simple_closure(cd.shipped_links(True))
    out["vs_reference"] = closure_stats(out["pairs"], {k: (v[0], v[1], v[2]) for k, v in ref.items()})
    return out


def hang_check(runtime: str, timeout_s: float) -> dict:
    """The feedback program on a runtime that cannot stop itself (hyperon): in a subprocess, killed at the timeout."""
    argv = [sys.executable, str(Path(__file__).resolve()), "_hang", "--runtime", runtime]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"})
        return {"finished": True, "returncode": r.returncode, "wall_s": round(time.perf_counter() - t0, 2), "stdout": r.stdout.strip()[-300:]}
    except subprocess.TimeoutExpired:
        return {"finished": False, "killed_after_s": timeout_s, "wall_s": round(time.perf_counter() - t0, 2)}


def _hang_main(runtime: str) -> None:
    import config
    config.PLN_RUNTIME = runtime
    config.PLN_CHAINER = "legacy"
    config.LOGS_DIR = Path(tempfile.mkdtemp(prefix="pln_number_table_logs_"))
    from core import pln_runner
    import api
    files = pln_runner.patient_stack(api._runtime_kb_paths())
    atom = "(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))\n"
    r = pln_runner._runtime_run("!(explain &self ChronicInflammation $x)", 0.0, files, atom)
    print(r.status, len(r.results))


# ══════════════════════════════ DrugAge ═════════════════════════════════════════

DRUGAGE_COMPOUNDS = ["Resveratrol", "Metformin", "Rapamycin", "Ethanol"]
#: the null conventions run next to the default ('both') at paper level: variant name -> ChainSettings.drugage_zero_rows
NULL_VARIANTS = ("protective", "neither")


def _drugage_rows(source: Path, compounds: Sequence[str]):
    from ontology import drugage_selector as ds
    rows = ds.select_rows(ds.load_rows(source), compounds=compounds, limit=10 ** 6)
    return rows


def python_twin(source: Path, compounds: Sequence[str]) -> dict:
    """The pure-Python fold (ontology.drugage_fold) of the same rows: paper level and row level, and the legacy representative row."""
    from ontology import drugage_fold as df
    from ontology import drugage_selector as ds
    rows = _drugage_rows(source, compounds)
    out = {}
    for c in compounds:
        rs = [r for r in rows if r.compound == c]
        if not rs:
            continue
        paper = df.fold_compound(rs, fold=True, evidence_unit="paper")
        row = df.fold_compound(rs, fold=True, evidence_unit="row")
        legacy = df.fold_compound(rs, fold=False)
        items = lambda f: [(i.sign, i.strength, i.confidence, len(i.units)) for i in f.items]
        out[c] = {"rows": len(rs), "papers": len({r.pmid for r in rs if r.pmid}),
                  "paper": {"net": paper.net.score, "items": items(paper)},
                  "row": {"net": row.net.score, "items": items(row)},
                  # the other two ways a reported null (a change of exactly 0.0) can count: the sign of a compound with nulls depends on it
                  **{f"paper_{z}": {"net": (f := df.fold_compound(rs, fold=True, evidence_unit="paper", zero_rows=z)).net.score, "items": items(f)}
                     for z in NULL_VARIANTS},
                  "legacy": {"net": legacy.net.score, "sign": legacy.net.sign, "strength": legacy.net.strength, "confidence": legacy.net.confidence}}
    return out


def _net(derivations: list[tuple]) -> Optional[dict]:
    from ontology.drugage_fold import net_of_signed
    if not derivations:
        return None
    try:
        score, sign, strength, confidence = net_of_signed([(sg, s, c) for sg, s, c in derivations])
    except ValueError as exc:
        return {"error": str(exc)}
    return {"net": score, "sign": sign, "strength": strength, "confidence": confidence}


def section_drugage(lane: Lane) -> dict:
    """The DrugAge compounds (the 313-row fixture of tests/fixtures/drugage_fold) and the mixed-sign compound, in this lane.

    legacy lanes: the application's own ranking (run_drugage_ranking: ONE representative row per compound, then the legacy chainer).
    pln lanes: every row of the compound (no cap), pooled per paper and direction by ontology.drugage_fold (paper level), or with every row
    its own unit (row level, the evidence_unit='row' control), chained through Lifespan -> Mortality; the NET is net_of_signed, the single
    formula of drugage_fold (sum over the compound's pooled items of +-strength x confidence, Neg +)."""
    from core import pln_runner
    from ontology import drugage_selector as ds
    sources = {"fixture": FIXTURES / "three_compounds.metta", "mix": FIXTURES / "mix_compound.metta"}
    groups = [("fixture", DRUGAGE_COMPOUNDS), ("mix", ["MixCompound"])]
    out: dict[str, Any] = {"compounds": {}}
    for label, comps in groups:
        src = sources[label]
        for c in comps:
            rows = _drugage_rows(src, [c])
            entry: dict[str, Any] = {"source": label, "rows": len(rows), "papers": len({r.pmid for r in rows if r.pmid})}
            if not lane.pln:
                r, picked = pln_runner.run_drugage_ranking([c], source=src)
                sc = parse_scored(r.results[1].atom) if r.status == "ok" and len(r.results) > 1 else []
                entry["legacy_ranking"] = {"status": r.status, "error": r.error, "representative_rows": [x.row_id for x in picked],
                                           "scored": sc[0] if sc else None}
                if lane.id == "legacy-petta":
                    slice_text, allrows = ds.build_drugage_slice([c], best_per_compound=False, limit=10 ** 6, source=src)
                    res = lane.run(lane.stack("drugage"), slice_text, [f"!(infer &self {c} Mortality)"], deadline_s=300)
                    entry["legacy_all_rows"] = {"derivations": len(res["groups"][0]), "status": res["status"]}
            else:
                slice_text, allrows = ds.build_drugage_slice([c], best_per_compound=False, limit=10 ** 6, source=src)
                for variant, fold in (("paper", None), ("row", "row"), *((f"paper_{z}", z) for z in NULL_VARIANTS)):
                    saved, saved_cfg = pc._fold_items, pc.settings_from_config
                    if fold == "row":
                        from ontology.drugage_fold import fold_items
                        pc._fold_items = lambda enabled=True: functools.partial(fold_items, evidence_unit="row")
                    elif fold in NULL_VARIANTS:
                        pc.settings_from_config = lambda z=fold: pc.ChainSettings(**{**lane._kw, "drugage_zero_rows": z})
                    try:
                        res = lane.run(lane.stack("drugage"), slice_text, [f"!(infer &self {c} Mortality)"])
                    finally:
                        pc._fold_items, pc.settings_from_config = saved, saved_cfg
                    if res["status"] == "error":
                        entry[variant] = {"error": res["error"]}
                        continue
                    ders = [parse_signed(a) for a in res["groups"][0]]
                    entry[variant] = {"derivations": [list(d) for d in ders if d], "net": _net([d for d in ders if d]),
                                      "cost": cost_of(res["chain"], c), "links": res["chain"]["links"] if res["chain"] else None,
                                      "complete": res["chain"]["complete"] if res["chain"] else None}
            out["compounds"][c] = entry
    out["python_twin"] = {**python_twin(sources["fixture"], DRUGAGE_COMPOUNDS), **python_twin(sources["mix"], ["MixCompound"])}
    return out


# ══════════════════════════════ the full lane: false chains and lost pairs ══════

def full_lane_harness(sess: "cd.Session", links: list) -> dict:
    """pln-full on the patient graph, source by source, with the beliefs in hand: the true routes found, the true pairs LOST in the budget,
    and the conclusions no walk of the graph licenses (false chains), by target. The budget is the lane's own (150 / 100 / 400).
    `links` are the ones the production step reads off the loaded space, IN THEIR ORDER (a run that spends a fixed budget depends on the
    order of its beliefs)."""
    steps, tq, bq = pc.FULL_LANE_BUDGET
    sess.engine.load(pc.base_rate_text(links))
    enc = cd.Enc(links, sess.leak)
    ref = recorded_closure()
    by_source: dict[str, set] = {}
    for chain in ref:
        by_source.setdefault(chain[0], set()).add(chain[-1])
    per_source: dict[str, Any] = {}
    false_pairs: list[dict] = []
    for source in sorted({l.src for l in links}):
        sub = links                                          # the full lane gives the whole graph as beliefs
        d = cd.Derive(sess.ev)
        beliefs = enc.beliefs_text(sub, source, 1.0)
        T, B = d(beliefs, beliefs, 1, steps, tq, bq)          # every sentence a task, as PLN.Query does
        con = cd.conclusions(enc, source, B)
        found_targets = {r["target"] for r in con.routes.values()}
        want = by_source.get(source, set())
        opaque_source = enc.opaque[source]
        unlicensed: dict[str, list] = {}
        seen_conclusion: set = set()
        for term, s, c, ids in B:
            if not (isinstance(term, tuple) and len(term) == 2 and term[0] in ("State", "End")):
                continue
            if not ids or ids[0] != "0" or term[1] == opaque_source or (term, ids) in seen_conclusion:
                continue
            seen_conclusion.add((term, ids))
            tail = tuple(int(i) for i in ids[1:])
            target = enc.key_of[term[1]]
            if cd.walk_of(enc, source, tail, simple=False, end=target, end_terminal=(term[0] == "End")) is not None:
                continue
            unlicensed.setdefault(target, []).append({"s": float(s), "c": float(c), "stamp_len": len(tail)})
        fp = [t for t in unlicensed if t not in want]
        for t in sorted(fp):
            false_pairs.append({"from": source, "to": t, "conclusions": len(unlicensed[t]), "best_c": max(x["c"] for x in unlicensed[t])})
        per_source[source] = {"true_pairs": len(want), "found": len(want & found_targets), "lost": sorted(want - found_targets),
                              "unlicensed_conclusions": sum(len(v) for v in unlicensed.values()), "false_pair_targets": sorted(fp),
                              "tasks_left": len(T), "beliefs": len(B), "steps": steps, "tq": tq, "bq": bq,
                              "conclusion_targets": sorted({enc.key_of[t[0][1]] for t in B if isinstance(t[0], tuple) and len(t[0]) == 2
                                                           and t[0][0] in ("State", "End") and t[3] and t[3][0] == "0" and t[0][1] != opaque_source})}
    return {"per_source": per_source, "false_pairs": false_pairs, "budget": [steps, tq, bq],
            "true_pairs": sum(v["true_pairs"] for v in per_source.values()), "found": sum(v["found"] for v in per_source.values()),
            "unlicensed": sum(v["unlicensed_conclusions"] for v in per_source.values())}


def section_full_harness(lane: "Lane") -> dict:
    """Run `full_lane_harness` inside the PeTTa child of the production hook, on the links the production step would read (same order)."""
    from core.metta_runtime import PettaRuntime
    files = lane.stack("patient")
    blocks = [(p.name, p.read_text(encoding="utf-8")) for p in files]
    kb_text = "\n".join(t for _, t in blocks)

    def step(engine):
        ev = lambda expr: [str(x) for x in engine.run("!" + expr)]
        links, skipped, nodes = pc.collect_links(ev, blocks)
        sess = cd.Session(engine, "full")
        return full_lane_harness(sess, links)

    rt = PettaRuntime()
    rt.run_program(kb_text, ["!(+ 1 2)"], deadline_s=3000, after_load=step)
    return rt.last_run["chain"]


# ══════════════════════════════ lane main ═══════════════════════════════════════

SECTIONS_ALL = ("patient", "feedback", "drugage", "full_harness")


def environment() -> dict:
    env = cd.environment()
    env["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    env["control_demo_sha256"] = cd.environment()["script_sha256"]
    env["hyperon"] = _version("hyperon")
    return env


def _version(name: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:                                              # noqa: BLE001
        return "unknown"


def run_lane(lane_id: str, out_dir: Path, sections: Sequence[str] = SECTIONS_ALL) -> dict:
    t_all = time.perf_counter()
    lane = Lane(lane_id)
    data: dict[str, Any] = {"lane": lane_id, "label": LANES[lane_id]["label"], "spec": LANES[lane_id],
                            "_measured_by": f"python scripts/pln_number_table.py lane --lane {lane_id}", "_environment": environment()}
    ref = recorded_closure()
    if "patient" in sections:
        t0 = time.perf_counter()
        p = section_patient(lane)
        data["patient"] = p
        if "error" not in p:
            data["orderings"] = section_orderings(p)
            data["closure_stats"] = closure_stats(p["closure"]["pairs"], ref)
            data["closure_hops"] = hops_range(p["closure"]["pairs"], ref)
        data["patient_wall_s"] = round(time.perf_counter() - t0, 1)
    if "feedback" in sections:
        t0 = time.perf_counter()
        data["feedback"] = section_feedback(lane)
        data["feedback_wall_s"] = round(time.perf_counter() - t0, 1)
    if "drugage" in sections:
        t0 = time.perf_counter()
        data["drugage"] = section_drugage(lane)
        data["drugage_wall_s"] = round(time.perf_counter() - t0, 1)
    if "full_harness" in sections and lane_id == "full":
        t0 = time.perf_counter()
        h = section_full_harness(lane)
        costs = data.get("patient", {}).get("costs", {})
        h["cross_check"] = {src: {"production_unattributed": (costs.get(src) or {}).get("unattributed"), "harness_unlicensed": v["unlicensed_conclusions"]}
                            for src, v in h["per_source"].items()}
        h["matches_production"] = all(x["production_unattributed"] == x["harness_unlicensed"] for x in h["cross_check"].values())
        data["full_harness"] = h
        data["full_harness_wall_s"] = round(time.perf_counter() - t0, 1)
    data["_wall_s"] = round(time.perf_counter() - t_all, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lane_{lane_id}.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"lane {lane_id}: {data['_wall_s']} s -> {path}")
    return data


def run_all(out_dir: Path, lanes: Sequence[str] = LANE_ORDER) -> None:
    for lane_id in lanes:
        argv = [sys.executable, str(Path(__file__).resolve()), "lane", "--lane", lane_id, "--out", str(out_dir)]
        r = subprocess.run(argv, env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"})
        if r.returncode != 0:
            print(f"lane {lane_id} FAILED with {r.returncode}", file=sys.stderr)


# ══════════════════════════════ rendering ═══════════════════════════════════════

def g(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}g}"


def cost_text(c: Optional[dict], lane_id: str) -> str:
    if lane_id.startswith("legacy"):
        return "no queues"
    if not c:
        return "-"
    st = c["st"]
    if lane_id == "full":
        return f"st {c['cap']} (budget) / tq {c['tq']} / bq {c['bq']}"
    return f"st {st if st is not None else '?'} / tq {c['tq']} / bq {c['bq']}"


def cell(lane_id: str, results: Sequence, cost: Optional[dict], empty: str = "empty") -> str:
    if not results:
        return f"{empty}<br>{cost_text(cost, lane_id)}" if lane_id != "legacy-hyperon" and lane_id != "legacy-petta" else empty
    lines = [f"{r[0]} {g(r[1])} / {g(r[2])}" for r in results if r]
    return "<br>".join(lines) + f"<br><sub>{cost_text(cost, lane_id)}</sub>"


def _esc(cell: Any) -> str:
    return str(cell).replace("|", "\\|")


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(_esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_esc(c) for c in r) + " |")
    return "\n".join(lines)


def program_cost_text(lane_data: dict, lid: str) -> str:
    """The cost of a whole program (the composite queries): the worst source."""
    if lid.startswith("legacy"):
        return "no queues"
    c = lane_data.get("patient", {}).get("program_cost")
    if not c:
        return "-"
    if lid == "full":
        return f"st {c['cap']} (budget) / tq {c['tq']} / bq {c['bq']}"
    return f"max st {c['st']} / tq {c['tq']} / bq {c['bq']}"


def load_lanes(out_dir: Path) -> dict[str, dict]:
    lanes = {}
    for lid in LANE_ORDER:
        p = out_dir / f"lane_{lid}.json"
        if p.exists():
            lanes[lid] = json.loads(p.read_text(encoding="utf-8"))
    return lanes


def _hdr(lanes: dict) -> list[str]:
    return [LANES[l]["label"] for l in LANE_ORDER if l in lanes]


def block_environment(lanes: dict) -> str:
    rows = []
    for lid in LANE_ORDER:
        if lid not in lanes:
            continue
        d = lanes[lid]
        rows.append([LANES[lid]["label"], f"`{d['_measured_by']}`", d["_wall_s"], d["_environment"].get("script_sha256", ""),
                     d["_environment"].get("hyperon", "") if lid == "legacy-hyperon" else d["_environment"].get("petta_commit", "")])
    return md_table(["lane", "command that measured every cell of the lane", "wall s", "script sha256 (16)", "engine"], rows)


def block_chains(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes]
    first = next(iter(lanes.values()))
    rows = []
    for k, ch in enumerate(first["patient"]["chains"]):
        pin = ch["pin"]
        pin_cell = f"{pin['sign']} {g(pin['s'])} / {g(pin['c'])}<br><sub>{pin['test']}</sub>" if pin else "empty"
        cells = [f"{ch['id']}{' *' if ch['smoke'] else ''}", pin_cell]
        for lid in ids:
            p = lanes[lid].get("patient", {})
            if "chains" not in p:
                cells.append("not measured")
                continue
            c = p["chains"][k]
            cells.append(cell(lid, c["infer"], c["cost"]))
        rows.append(cells)
    return md_table(["chain (* = the five smoke chains)", "pin (the test's arithmetic)"] + _hdr(lanes), rows)


def block_chain_deviation(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "patient" in lanes[l] and "chains" in lanes[l]["patient"]]
    base = lanes["legacy-hyperon"]["patient"]["chains"] if "legacy-hyperon" in lanes else lanes[ids[0]]["patient"]["chains"]
    rows = []
    for lid in ids:
        ch = lanes[lid]["patient"]["chains"]
        ds = dc = 0.0
        flips = 0
        for a, b in zip(ch, base):
            if a["infer"] and b["infer"]:
                ds = max(ds, abs(a["infer"][0][1] - b["infer"][0][1]))
                dc = max(dc, abs(a["infer"][0][2] - b["infer"][0][2]))
                flips += a["infer"][0][0] != b["infer"][0][0]
            elif bool(a["infer"]) != bool(b["infer"]):
                flips += 1
        rows.append([LANES[lid]["label"], f"{ds:.2e}", f"{dc:.2e}", flips])
    return md_table(["lane", "max abs strength difference to legacy (hyperon), nine chains", "max abs confidence difference to legacy (hyperon)", "sign or emptiness differs"], rows)


def block_orderings(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "orderings" in lanes[l]]
    if not ids:
        return "not measured"
    first = lanes[ids[0]]["orderings"]["orderings"]
    rows = []
    for k, o in enumerate(first):
        row = [o["id"], o["name"]]
        for lid in ids:
            oo = lanes[lid]["orderings"]["orderings"][k]
            row.append(("holds" if oo["holds"] else "**FAILS**") + f"<br><sub>{oo['values']}</sub>")
        rows.append(row)
    costs = ["cost of the program"]
    costs.append("")
    for lid in ids:
        c = lanes[lid]["patient"].get("program_cost")
        if lid.startswith("legacy"):
            costs.append("no queues")
        elif c:
            if lid == "full":
                costs.append(f"st {c['cap']} (budget) / tq {c['tq']} / bq {c['bq']}; {c['derive_runs']} runs, {c['derive_s']} s; routes complete: {c['complete']}")
            else:
                costs.append(f"max st {c['st']} / tq {c['tq']} / bq {c['bq']}; max beliefs {c['beliefs']}; {c['derive_runs']} runs, {c['derive_s']} s")
        else:
            costs.append("-")
    rows.append(costs)
    holds = ["orderings that hold", ""] + [f"{sum(1 for o in lanes[l]['orderings']['orderings'] if o['holds'])} of 12" for l in ids]
    rows.append(holds)
    return md_table(["#", "ordering (contract.md 4.2 / 5.4)"] + [LANES[l]["label"] for l in ids], rows)


def block_diagnose(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "orderings" in lanes[l]]
    rows = []
    for lid in ids:
        d = lanes[lid]["orderings"]["diagnose"]
        pop, p1 = d.get("population", {}), d.get("Patient001", {})
        ok_s = abs(pop.get("s_CS", 0) - 0.595) <= 1e-6 * 0.595
        ok_m = abs(pop.get("c_MD", 0) - 0.65) <= 1e-6 * 0.65
        ok_c = abs(p1.get("c_CS", 0) - 0.6253054502923161) <= 1e-2
        rows.append([LANES[lid]["label"], g(pop.get("s_CS")) + ("" if ok_s else " **FAILS**"), g(pop.get("c_CS")), g(pop.get("c_MD")) + ("" if ok_m else " **FAILS**"),
                     g(pop.get("mass_CS")), g(p1.get("c_CS")) + ("" if ok_c else " **FAILS**"), program_cost_text(lanes[lid], lid)])
    return md_table(["lane", "s(CS), pin 0.595 (rel 1e-6)", "c(CS), population", "c(MD), pin 0.65 (rel 1e-6)", "mass(CS)",
                     "c(CS), Patient001, pin 0.6253 (abs 1e-2)", "cost of the program"], rows)


def block_personalised(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "orderings" in lanes[l]]
    rows = []
    for patient in ("Patient001", "Patient002"):
        for lid in ids:
            sc = lanes[lid]["orderings"]["personalised"][patient]
            short = {DQ: "D+Q"}
            order = " > ".join(f"{short.get(x['name'], x['name'])} {x['score']:.4f}" for x in sc)
            signs = {x["sign"] for x in sc}
            rows.append([patient, LANES[lid]["label"], order, ("all " + next(iter(signs))) if len(signs) == 1 else str(sorted(signs)), program_cost_text(lanes[lid], lid)])
    return md_table(["patient", "lane", "personalised order with scores (rank-score + relevance)", "signs", "cost of the program"], rows)


def block_closure(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "closure_stats" in lanes[l]]
    rows = []
    for lid in ids:
        s = lanes[lid]["closure_stats"]
        rs = f"{s['ratio_s'][0]:.4f} .. {s['ratio_s'][1]:.4f}" if s["ratio_s"] else "-"
        rc = f"{s['ratio_c'][0]:.4f} .. {s['ratio_c'][1]:.4f}" if s["ratio_c"] else "-"
        rows.append([LANES[lid]["label"], f"{s['pairs']} of {s['reference']}", s["missing"], s["extra"], s["duplicates"], s["sign_differs"],
                     f"{s['max_ds']:.2e}", f"{s['max_dc']:.2e}", rs, rc, f"{s['strength_digits_differ']} / {s['confidence_digits_differ']}",
                     program_cost_text(lanes[lid], lid)])
    return md_table(["lane", "pairs", "missing", "extra", "pairs with 2 results", "sign differs", "max abs dev strength", "max abs dev confidence",
                     "strength ratio to recorded (min .. max)", "confidence ratio (min .. max)", "strings that differ (strength / confidence)",
                     "cost (worst source)"], rows)


def block_closure_hops(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "closure_hops" in lanes[l]]
    hs = sorted({h for l in ids for h in lanes[l]["closure_hops"]})
    rows = []
    for lid in ids:
        r = lanes[lid]["closure_hops"]
        rows.append([LANES[lid]["label"]] + [f"{r[h][0]:.4f} .. {r[h][1]:.4f}" if h in r else "-" for h in hs])
    return md_table(["lane"] + [f"{h} hop{'s' if h != '1' else ''}" for h in hs], rows)


def block_nofalse(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "patient" in lanes[l] and "nofalse" in lanes[l]["patient"]]
    first = lanes[ids[0]]["patient"]["nofalse"]
    full_h = lanes.get("full", {}).get("full_harness", {}).get("per_source", {})
    rows = []
    for k, nf in enumerate(first):
        row = [f"{nf['a']} -> {nf['c']}", nf["why"]]
        for lid in ids:
            n = lanes[lid]["patient"]["nofalse"][k]
            txt = "empty" if not n["infer"] else "<br>".join(f"**{r[0]} {g(r[1])} / {g(r[2])}**" for r in n["infer"])
            if not lid.startswith("legacy"):
                txt += f"<br><sub>{cost_text(n['cost'], lid) if n['cost'] else 'no run: no link out of the node'}</sub>"
            if lid == "full" and nf["a"] in full_h:
                reached = nf["c"] in full_h[nf["a"]]["conclusion_targets"]
                txt += f"<br><sub>the engine's belief set {'CONTAINS a conclusion (State ' + nf['c'] + ') (FALSE CHAIN, not licensed by any walk)' if reached else 'holds no such conclusion'}</sub>"
            row.append(txt)
        rows.append(row)
    return md_table(["pair (must be empty)", "why"] + [LANES[l]["label"] for l in ids], rows)


def block_full_lane(lanes: dict) -> str:
    h = lanes.get("full", {}).get("full_harness")
    if not h:
        return "pln-full lane not measured"
    rows = []
    for src, v in sorted(h["per_source"].items()):
        rows.append([src, v["true_pairs"], v["found"], ", ".join(v["lost"]) or "-", v["unlicensed_conclusions"],
                     ", ".join(v["false_pair_targets"]) or "-", f"st {v['steps']} (budget), tasks left {v['tasks_left']} / tq {v['tq']} / bq {v['bq']}, beliefs {v['beliefs']}"])
    tot = md_table(["source", "true pairs", "found as a route", "pairs LOST in the budget", "conclusions no walk licenses", "false-chain targets (not in the true closure)", "cost"], rows)
    summary = (f"Total: {h['found']} of {h['true_pairs']} true pairs found, {h['true_pairs'] - h['found']} lost; {h['unlicensed']} conclusions that no walk of the graph licenses; "
               f"{len(h['false_pairs'])} false pairs (source, target) outside the true closure. Budget (steps / tq / bq) = {h['budget']}. "
               f"The harness's per-source count of unlicensed conclusions equals the production step's (`unattributed`) for every source: {h.get('matches_production')}.")
    fp = md_table(["false pair", "unlicensed conclusions", "best confidence"],
                  [[f"{x['from']} -> {x['to']}", x["conclusions"], g(x["best_c"])] for x in h["false_pairs"]])
    return "\n\n".join([summary, tot, fp])


def block_feedback(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "feedback" in lanes[l]]
    rows = []
    ref_new = None
    for lid in ids:
        fb = lanes[lid]["feedback"]
        if not fb.get("terminates"):
            hang = fb.get("hang")
            extra = f"run in a subprocess, killed after {hang['killed_after_s']} s (finished: {hang['finished']})" if hang else ""
            rows.append([LANES[lid]["label"], "NO", "-", "-", "-", "-", f"{fb['error'] or ''}{extra}", "no queues"])
            continue
        vs = fb["vs_reference"]
        pc_ = fb.get("program_cost")
        cost = ("no queues" if lid.startswith("legacy") else
                (f"st {pc_['cap']} (budget) / tq {pc_['tq']} / bq {pc_['bq']}" if lid == "full" else
                 f"max st {pc_['st']} / tq {pc_['tq']} / bq {pc_['bq']}; max beliefs {pc_['beliefs']}"))
        rows.append([LANES[lid]["label"], "yes", f"{vs['pairs']} of {vs['reference']}", vs["missing"], vs["extra"], f"{vs['max_ds']:.1e} / {vs['max_dc']:.1e}",
                     f"{fb['wall_s']} s", cost])
    t1 = md_table(["lane", "terminates", "pairs (node-simple reference 109)", "missing", "extra", "max dev strength / confidence vs the node-simple reference", "wall", "cost"], rows)
    # the nine new pairs and the cost of the sources that matter, lane by lane
    new_pairs = [("ChronicInflammation", "CellularSenescence"), ("ChronicInflammation", "SASP"), ("ChronicInflammation", "DNAmPAI1"),
                 ("ChronicInflammation", "CoronaryHeartDisease"), ("Omega3", "CellularSenescence"), ("Omega3", "SASP"), ("Omega3", "DNAmPAI1"),
                 ("Omega3", "CoronaryHeartDisease"), ("SASP", "CellularSenescence")]
    rows2 = []
    for a, b in new_pairs:
        row = [f"{a} -> {b}"]
        for lid in ids:
            fb = lanes[lid]["feedback"]
            if not fb.get("terminates"):
                row.append("-")
                continue
            hit = [p for p in fb["pairs"] if p["chain"][0] == a and p["chain"][-1] == b]
            c = fb["costs"].get(a)
            row.append(cell(lid, [r for p in hit for r in p["results"][:1]], c) if hit else "absent")
        rows2.append(row)
    t2 = md_table(["new pair (not in the shipped graph's 100)"] + [LANES[l]["label"] for l in ids], rows2)
    # the old 100: unchanged against the same lane's patient closure
    rows3 = []
    for lid in ids:
        fb = lanes[lid]["feedback"]
        if not fb.get("terminates") or "patient" not in lanes[lid] or "closure" not in lanes[lid]["patient"]:
            continue
        old = {tuple(p["chain"]): p["results"][0] for p in lanes[lid]["patient"]["closure"]["pairs"]}
        new = {tuple(p["chain"]): p["results"][0] for p in fb["pairs"]}
        same = sum(1 for k in old if k in new and old[k] == new[k])
        rows3.append([LANES[lid]["label"], f"{same} of {len(old)}", f"{len([k for k in new if k not in old])}"])
    t3 = md_table(["lane", "old answers byte-identical with the edge added (same lane)", "new pairs"], rows3)
    return "\n\n".join([t1, t2, t3])


VARIANTS = ("paper", "row", "paper_protective", "paper_neither")


def block_drugage(lanes: dict) -> str:
    ids = [l for l in LANE_ORDER if l in lanes and "drugage" in lanes[l]]
    twin = lanes[ids[0]]["drugage"]["python_twin"]
    comps = list(lanes[ids[0]]["drugage"]["compounds"])
    rows = []
    labels = {"paper": "paper level (nulls on both sides)", "row": "row level (nulls on both sides)",
              "paper_protective": "paper level, nulls on the protective side only", "paper_neither": "paper level, nulls dropped"}
    for c in comps:
        for variant in VARIANTS:
            row = [c, lanes[ids[0]]["drugage"]["compounds"][c]["rows"], lanes[ids[0]]["drugage"]["compounds"][c]["papers"], labels[variant]]
            for lid in ids:
                e = lanes[lid]["drugage"]["compounds"][c]
                if lid.startswith("legacy"):
                    lr = e.get("legacy_ranking", {}).get("scored")
                    if variant == "paper":
                        txt = (f"score {g(lr['score'])} ({lr['sign']} {g(lr['s'])} / {g(lr['c'])})<br><sub>one representative row; no queues"
                               + (f"; {e['legacy_all_rows']['derivations']} derivations if every row is given" if "legacy_all_rows" in e else "") + "</sub>") if lr else "empty"
                    else:
                        txt = "n/a (legacy has no pooling)"
                elif variant not in e:
                    txt = "-"
                else:
                    v = e.get(variant, {})
                    if "error" in v:
                        txt = f"error: {v['error'][:60]}"
                    else:
                        net = v.get("net") or {}
                        ders = "; ".join(f"{d[0]} {g(d[1])} / {g(d[2])}" for d in v.get("derivations", []))
                        txt = (f"NET {g(net.get('net'))}<br><sub>{ders}<br>{cost_text(v.get('cost'), lid)}</sub>" if net else f"empty<br><sub>{cost_text(v.get('cost'), lid)}</sub>")
                row.append(txt)
            t = twin.get(c)
            if t and variant in t:
                tv = t[variant]
                row.append(f"NET {g(tv['net'])}" + (f"<br><sub>legacy twin {g(t['legacy']['net'])}</sub>" if variant == "paper" else ""))
            else:
                row.append("-")
            rows.append(row)
    table = md_table(["compound", "rows", "papers", "unit"] + _hdr({l: 1 for l in ids}) + ["Python twin (ontology.drugage_fold)"], rows)
    worst = 0.0
    n = 0
    if "R2" in lanes:
        for c, e in lanes["R2"]["drugage"]["compounds"].items():
            for variant in VARIANTS:
                net = (e.get(variant) or {}).get("net")
                if net and c in twin and variant in twin[c]:
                    worst = max(worst, abs(net["net"] - twin[c][variant]["net"]))
                    n += 1
    foot = (f"NET = sum over the compound's pooled items of +strength x confidence (Neg, protective) or -strength x confidence (Pos), "
            f"`ontology.drugage_fold.net_of_signed`; confidence is the lane's (R2 includes the 0.9 of the second hop; R1, R0 and full do not). "
            f"pln-R2 against the pure-Python twin over {n} cells (compound x unit): max abs difference {worst:.1e}. "
            f"Legacy cells are the application's own ranking (one representative row per compound).")
    return table + "\n\n" + foot


BLOCKS: dict[str, Callable[[dict], str]] = {
    "environment": block_environment, "chains": block_chains, "chain_deviation": block_chain_deviation,
    "orderings": block_orderings, "diagnose": block_diagnose, "personalised": block_personalised, "closure": block_closure,
    "closure_hops": block_closure_hops, "nofalse": block_nofalse, "full_lane": block_full_lane, "feedback": block_feedback,
    "drugage": block_drugage,
}


def render(out_dir: Path, doc: Path) -> None:
    lanes = load_lanes(out_dir)
    if not lanes:
        raise SystemExit(f"no lane JSON in {out_dir}")
    text = doc.read_text(encoding="utf-8")
    for name, fn in BLOCKS.items():
        begin, end = f"<!-- BEGIN GENERATED {name} -->", f"<!-- END GENERATED {name} -->"
        if begin not in text:
            continue
        if end not in text:
            raise SystemExit(f"{doc}: block {name} has no end marker")
        pre, rest = text.split(begin, 1)
        _, post = rest.split(end, 1)
        text = pre + begin + "\n" + stamp_line(lanes, name) + "\n\n" + fn(lanes) + "\n" + end + post
    doc.write_text(text, encoding="utf-8")
    print(f"rendered {len(lanes)} lanes into {doc}")


def stamp_line(lanes: dict, name: str) -> str:
    """The provenance line at the head of every generated block: which command measured each column."""
    sha = next(iter(lanes.values()))["_environment"].get("script_sha256", "")
    cols = ", ".join(f"{LANES[l]['label']} = `{l}`" for l in LANE_ORDER if l in lanes)
    if name == "environment":
        return f"Measured, one command per lane; script sha256 {sha}."
    return (f"*Measured: every column by `python scripts/pln_number_table.py lane --lane <id>` (script sha256 {sha}); ids: {cols}. "
            f"Cost notation `st / tq / bq` = selections of the level-1 run / task-queue size / belief-queue size.*")


def main(argv: Optional[Sequence[str]] = None) -> int:
    # never read pln_chat/.env from a script; set here and not at import, so that importing the module (the tests do) leaves the
    # importer's environment alone. Nothing imports `config` before `main` runs.
    os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("lane")
    l.add_argument("--lane", choices=LANE_ORDER, required=True)
    l.add_argument("--out", type=Path, default=DEFAULT_OUT)
    l.add_argument("--sections", default=",".join(SECTIONS_ALL))
    a = sub.add_parser("all")
    a.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a.add_argument("--lanes", default=",".join(LANE_ORDER))
    r = sub.add_parser("render")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    r.add_argument("--doc", type=Path, default=DOC)
    h = sub.add_parser("_hang")
    h.add_argument("--runtime", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "lane":
        run_lane(args.lane, args.out, [s for s in args.sections.split(",") if s])
    elif args.cmd == "all":
        run_all(args.out, [s for s in args.lanes.split(",") if s])
    elif args.cmd == "render":
        render(args.out, args.doc)
    else:
        _hang_main(args.runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
