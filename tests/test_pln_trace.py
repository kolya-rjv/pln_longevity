"""The reasoning trace (core/pln_trace.py): what PLN_CHAINER=pln derived, step by step, for the answer it gave.

Two layers: the building, cutting and rendering on the sim engine of test_pln_chainer (no PeTTa), and the whole path on the real
PeTTa (`-m petta`): the calls recorded in the child, the traced numbers against the answer, the routes a ranking or a diagnosis
read, and that a trace can neither change an answer nor fail one. No test calls a model.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_pln_chainer as base  # noqa: E402
from test_pln_chainer import FEEDBACK_ATOM, SimEngine, _stack, _use, contract_links, real_petta, run, to_links  # noqa: E402

import config  # noqa: E402
from core import pln_chainer as pc  # noqa: E402
from core import pln_runner, pln_trace as pt  # noqa: E402
from core.metta_runtime import PettaRuntime, get_runtime  # noqa: E402

DQ, CHD = "DasatinibPlusQuercetin", "CoronaryHeartDisease"


@pytest.fixture
def pln(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln", PLN_TRACE=True)


def call(source, target, answered=True, expressions=(0,)):
    return {"source": source, "target": target, "answered": answered, "expressions": set(expressions)}


def sim_table(rows=None, **settings):
    links = to_links(rows if rows is not None else contract_links("patient"))
    return pc.compute_table(SimEngine(), links, leak=0.02, discount=0.9, settings=pc.ChainSettings(trace=True, **settings))


def python_truth(exprs):
    """The library's two formulas, as an independent evaluator of the check."""
    out = []
    for e in exprs:
        n = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", e)]
        if e.startswith("(Truth_ModusPonens"):
            f1, c1, f2, c2 = n
            out.append([f"(stv {f1 * f2 + 0.02 * (1 - f1)!r} {c1 * c2!r})"])
        else:
            out.append([f"(stv {1.0 - n[0]!r} {n[1]!r})"])
    return out


def assert_consistent(trace):
    """Every traced conclusion's steps lead to its reported number, and every rule application was reproduced."""
    assert "error" not in trace, trace.get("error")
    assert trace["check"]["mismatched"] == 0
    assert trace["check"]["reproduced"] == trace["check"]["applications"]
    for c in trace["conclusions"]:
        assert len(c["steps"]) == len(c["path"]) - 1 == len(c["evidence_ids"]), c
        assert [s["from"] for s in c["steps"]] == c["path"][:-1] and [s["to"] for s in c["steps"]] == c["path"][1:]
        last = c["steps"][-1]["so_far"]
        assert (last["sign"], last["strength"], last["confidence"]) == (
            c["sign"], pytest.approx(c["strength"], abs=1e-12), pytest.approx(c["confidence"], abs=1e-12))
        assert c["readout"]["strength_equals_product"]
        r = c["readout"]
        assert r["library_confidence"] * r["confidence_factor"] == pytest.approx(c["confidence"], abs=1e-12)
        assert abs(r["level1"] - r["level0"]) / r["divisor"] == pytest.approx(c["strength"], abs=1e-12)
        assert c["strength"] == pytest.approx(math.prod(s["link"]["strength"] for s in c["steps"]), abs=1e-9)


# ══════════════════════════════ building (no PeTTa) ═══════════════════════════════

def test_a_trace_names_every_step_of_the_route_and_its_rules():
    t = pt.build_trace(sim_table(), [call(DQ, CHD)], python_truth)
    assert_consistent(t)
    (c,) = t["conclusions"]
    assert c["path"] == [DQ, "CellularSenescence", "SASP", "DNAmPAI1", CHD]
    assert [[a["rule"] for a in s["rules"]] for s in c["steps"]] == [
        ["ModusPonens", "NegationElimination"], ["ModusPonens"], ["ModusPonens"], ["ModusPonens"]]
    assert c["sign"] == "Neg" and c["strength"] == pytest.approx(0.2499) and c["confidence"] == pytest.approx(0.17017138125)
    mp = c["steps"][0]["rules"][0]
    assert mp["premises"][0] == {"term": f"(State {DQ})", "level1": [1.0, 1.0], "level0": [0.0, 1.0]}
    assert mp["premises"][1]["term"] == f"(Implication (State {DQ}) (Not (State CellularSenescence)))"
    assert mp["premises"][1]["level1"] == [pytest.approx(0.02 + 0.98 * 0.7), 0.65]
    assert c["steps"][-1]["rules"][0]["conclusion"]["term"] == f"(End {CHD})"      # a record ends the chain
    assert not re.search(r"\bn\d+\b", json.dumps(t))                               # no opaque node name leaks out


def test_every_route_of_the_patient_graph_can_be_traced():
    table = sim_table()
    calls = [call(*p) for p in sorted({(r.source, r.target) for r in table.all_routes()})]
    old, pt.MAX_CONCLUSIONS = pt.MAX_CONCLUSIONS, 1000
    try:
        t = pt.build_trace(table, calls, python_truth)
    finally:
        pt.MAX_CONCLUSIONS = old
    assert len(t["conclusions"]) == 100
    assert_consistent(t)


def test_the_native_confidence_is_read_out_without_the_discount():
    t = pt.build_trace(sim_table(chain_confidence="native"), [call(DQ, CHD)], python_truth)
    assert_consistent(t)
    assert t["conclusions"][0]["readout"]["confidence_factor"] == 1.0


def test_a_belief_that_is_not_what_the_rule_gives_is_reported_as_a_mismatch():
    table = sim_table()
    b1, b0 = table.beliefs[DQ]
    table.beliefs[DQ] = ([(t, s + 0.01 if t[0] == "State" and len(ids) == 3 else s, c, ids) for t, s, c, ids in b1], b0)
    t = pt.build_trace(table, [call(DQ, CHD)], python_truth)
    assert t["check"]["mismatched"] >= 1
    assert any(a["checked"] is False for s in t["conclusions"][0]["steps"] for a in s["rules"])
    assert "differ" in pt.render_trace(t, 2)


def test_a_belief_that_is_missing_is_said_and_never_filled_in():
    table = sim_table()
    b1, b0 = table.beliefs[DQ]
    table.beliefs[DQ] = ([x for x in b1 if x[0][0] != "Not"], b0)
    t = pt.build_trace(table, [call(DQ, CHD)], python_truth)
    first = t["conclusions"][0]["steps"][0]
    assert all("missing" in a and a["checked"] is None for a in first["rules"])
    assert first["rules"][0]["conclusion"]["level1"] is None
    assert "not checked" in pt.render_trace(t, 2)


@pytest.mark.parametrize("answers", [[], ["(stv 0.1 0.1)", "(stv 0.2 0.2)"], ["(Error x)"]])
def test_a_formula_that_does_not_evaluate_to_one_value_leaves_the_step_unchecked(answers):
    t = pt.build_trace(sim_table(), [call(DQ, CHD)], lambda exprs: [answers for _ in exprs])
    assert t["check"]["reproduced"] == 0 and t["check"]["mismatched"] == 0
    assert len(t["conclusions"]) == 1


def test_pairs_without_a_route_and_answers_the_chainer_did_not_derive_are_told_apart():
    t = pt.build_trace(sim_table(), [call(DQ, "AllCauseMortality", False), call("Mystery", CHD, True)], python_truth)
    assert t["conclusions"] == []
    assert [x["target"] for x in t["no_route"]] == ["AllCauseMortality"]
    assert [x["source"] for x in t["outside_chainer"]] == ["Mystery"]


def test_a_connected_pair_the_program_asked_for_and_never_got_is_not_called_read():
    t = pt.build_trace(sim_table(), [call(DQ, CHD, answered=False)], python_truth)
    assert t["conclusions"] == t["no_route"] == t["outside_chainer"] == []


def test_what_is_over_the_cap_is_counted(monkeypatch):
    monkeypatch.setattr(pt, "MAX_CONCLUSIONS", 2)
    table = sim_table()
    calls = [call(*p) for p in sorted({(r.source, r.target) for r in table.all_routes()})]
    t = pt.build_trace(table, calls, python_truth)
    assert len(t["conclusions"]) == 2 and t["omitted"] == 98
    assert "100 conclusion(s) read" in pt.render_trace(t, 1)


def test_a_table_made_without_the_switch_keeps_no_beliefs():
    table = pc.compute_table(SimEngine(), to_links(contract_links("patient")), leak=0.02, discount=0.9, settings=pc.ChainSettings())
    assert table.beliefs == {} and table.encoding is None


# ══════════════════════════════ cutting, merging, rendering ═══════════════════════

@pytest.fixture
def built():
    return pt.build_trace(sim_table(), [call(DQ, CHD)], python_truth)


def test_the_levels_cut_the_trace_and_leave_the_original_alone(built):
    before = json.dumps(built)
    assert pt.limit_trace(built, 0) is None and pt.limit_trace(None, 3) is None
    one, two, three = (pt.limit_trace(built, n)["conclusions"][0] for n in (1, 2, 3))
    assert "steps" not in one and "readout" not in one and one["path"][0] == DQ
    assert "evidence" not in two["steps"][0] and two["readout"]["hops"] == 4
    assert three["steps"][0]["evidence"]["kind"] == "curated"
    assert json.dumps(built) == before


def test_rendering_by_level(built):
    assert pt.render_trace(built, 0) == ""
    one, two, three = (pt.render_trace(built, n) for n in (1, 2, 3))
    assert "→" in one and "Modus Ponens" not in one
    assert "Modus Ponens" in two and "Negation elimination" in two and "Read-out" in two and "evidence:" not in two
    assert three.count("evidence:") == 4 and three.count("✓") == 5 and "5 of 5 rule applications" in three


def test_rendering_text_of_the_caller_stays_inline_code(built):
    built["conclusions"][0]["steps"][0]["to"] = "x` <b>\n\n# y"
    text = pt.render_trace(built, 3)
    assert "`x' <b> # y`" in text and not any(line.startswith("# ") for line in text.splitlines())


def test_rendering_without_a_trace_says_why(monkeypatch):
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    assert "legacy chainer" in pt.render_trace(None, 2, pt.unavailable_reason())
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    monkeypatch.setattr(config, "PLN_TRACE", False)
    assert "PLN_TRACE" in pt.render_trace(None, 2, pt.unavailable_reason())
    assert pt.render_trace(None, 0, pt.unavailable_reason()) == ""


def test_a_failed_trace_says_the_answer_stands():
    assert "unaffected" in pt.render_trace({"error": "boom", "conclusions": []}, 2)


def test_merged_traces_keep_program_order_and_say_their_part(built):
    merged = pt.merge_traces([built, None, built], [[2], [0], [1]])
    assert [(c["part"], c["expressions"]) for c in merged["conclusions"]] == [(1, [3]), (3, [2])]
    assert merged["check"]["applications"] == 10
    assert pt.merge_traces([None, None]) is None


def test_the_answer_carries_the_trace_only_when_asked(built):
    from utils.formatting import format_bot_response
    tr = SimpleNamespace(error=None, explanation="", metta_query="(x)", warnings=[], usage=None, intent="", raw_response="",
                         requires_pln_inference=True)
    val = SimpleNamespace(valid=True, issues=[])
    res = pln_runner.PLNRunResult(status="ok", results=[pln_runner.PLNAtomResult("a")], mode="runtime", trace=built)
    args = dict(translation=tr, validation=val, pln_result=res, show_metta=False, show_explanation=False, show_debug=False)
    assert "reasoning trace" not in format_bot_response(**args)
    assert "Modus Ponens" in format_bot_response(**args, trace_level=2)


def test_the_log_holds_the_whole_trace(monkeypatch, tmp_path, built):
    from utils import logging as plog
    monkeypatch.setattr(plog, "_log_file", tmp_path / "s.jsonl")
    plog.log_turn("q", SimpleNamespace(metta_query="(x)"), SimpleNamespace(status="ok", trace=built))
    plog.log_turn("q", SimpleNamespace(metta_query="(x)"), SimpleNamespace(status="ok", trace=None))
    first, second = (json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines())
    assert first["pln_trace"]["conclusions"][0]["steps"] and "pln_trace" not in second


@pytest.mark.parametrize("value, ok", [("on", True), ("off", False), ("", True), ("maybe", None)])
def test_PLN_TRACE_is_on_or_off(monkeypatch, value, ok):
    monkeypatch.setenv("PLN_TRACE", value)
    if ok is None:
        with pytest.raises(ValueError):
            config._trace_choice()
    else:
        assert config._trace_choice() is ok


# ══════════════════════════════ the real PeTTa ═════════════════════════════════════

@real_petta
@pytest.mark.parametrize("form", ["infer", "explain"])
def test_a_direct_question_is_traced_with_its_evidence(pln, form):
    groups, result = run(_stack("patient"), None, [f"!({form} &self {DQ} {CHD})"])
    assert_consistent(result.trace)
    (c,) = result.trace["conclusions"]
    assert f"(stv {c['strength']!r} {c['confidence']!r})" in groups[0][0]          # the traced number is the answered one
    evidence = [s["evidence"] for s in c["steps"]]
    assert evidence[0]["source"].startswith("mechanistic_bridges.metta:") and evidence[0]["tier"] == "AnimalStudies_Replicated"
    assert evidence[-1]["record"] == "Lu2019_DNAmPAI1_CHD"
    line = int(evidence[0]["source"].split(":")[1])
    assert f"(Effect {DQ} CellularSenescence" in (base.REPO / "mechanistic_bridges.metta").read_text().splitlines()[line - 1]


@real_petta
def test_the_traced_conclusions_are_the_ones_the_program_read_and_no_others(pln):
    _, result = run(_stack("patient"), None, [f"!(rank-interventions &self ({DQ} Fisetin) {CHD})", "!(infer &self SASP CRP)"])
    assert_consistent(result.trace)
    assert {(c["source"], c["target"], tuple(c["expressions"])) for c in result.trace["conclusions"]} == {
        (DQ, CHD, (1,)), ("Fisetin", CHD, (1,)), ("SASP", "CRP", (2,))}


@real_petta
def test_a_diagnosis_is_traced_through_the_calls_it_makes(pln):
    groups, result = run(_stack("patient"), None, ["!(diagnose-patient &self Patient001)"])
    assert_consistent(result.trace)
    supported = {(h, o) for h, os_ in re.findall(r"\(Hypothesis (\w+) .*?\(SupportedBy \(([^)]*)\)\)\)", groups[0][0]) for o in os_.split()}
    assert supported and supported <= {(c["source"], c["target"]) for c in result.trace["conclusions"]}


@real_petta
def test_no_route_and_plain_lookups(pln):
    _, result = run(_stack("patient"), None, [f"!(infer &self {DQ} AllCauseMortality)"])
    assert result.trace["conclusions"] == [] and result.trace["no_route"] == [
        {"source": DQ, "target": "AllCauseMortality", "expressions": [1]}]
    _, result = run(_stack("patient"), None, ["!(match &self (Effect SASP $x $s $tv) $x)"])
    assert result.trace["conclusions"] == result.trace["no_route"] == []
    assert "made no PLN inference" in pt.render_trace(result.trace, 1)


@real_petta
def test_an_open_target_traces_every_answer(pln):
    groups, result = run(_stack("patient"), None, [f"!(explain &self {DQ} $x)"])
    assert_consistent(result.trace)
    assert len(result.trace["conclusions"]) == len(groups[0]) > 3


@real_petta
def test_a_feedback_edge_is_traced(pln):
    _, result = run(_stack("patient"), FEEDBACK_ATOM, ["!(infer &self ChronicInflammation SASP)"])
    assert_consistent(result.trace)
    (c,) = result.trace["conclusions"]
    assert c["path"] == ["ChronicInflammation", "CellularSenescence", "SASP"]
    assert c["steps"][0]["evidence"]["kind"] == "caller"


@real_petta
def test_an_answer_of_the_callers_own_equation_is_not_passed_off_as_a_derivation(pln):
    extra = "(= (infer $s MyDrug MyOutcome) (signed Neg (stv 0.9 0.9)))\n"
    groups, result = run(_stack("patient"), extra, ["!(infer &self MyDrug MyOutcome)"])
    assert groups[0] == ["(signed Neg (stv 0.9 0.9))"]
    assert result.trace["conclusions"] == []
    assert [x["source"] for x in result.trace["outside_chainer"]] == ["MyDrug"]


@real_petta
def test_drugage_links_carry_their_rows_and_papers(pln):
    text = base._drugage_rows("TraceRow_", "TraceDrug", [("Mus_musculus", 12.0, "Significant", False, "PMID_1"),
                                                         ("Mus_musculus", 8.0, "Significant", False, "PMID_2")])
    _, result = run(_stack("drugage"), text, ["!(score-candidate &self TraceDrug Mortality)"])
    assert_consistent(result.trace)
    evidence = result.trace["conclusions"][0]["steps"][0]["evidence"]
    assert sorted(evidence["rows"]) == ["TraceRow_0", "TraceRow_1"] and sorted(evidence["papers"]) == ["PMID_1", "PMID_2"]


@real_petta
def test_the_drugage_ranking_hands_the_trace_on(pln):
    from core.drugage_router import rank_drugage
    rows = Path(__file__).resolve().parent / "fixtures" / "drugage_real_rows.metta"
    result = rank_drugage(["Rapamycin"], source=rows).result
    assert result.status == "ok" and result.trace["conclusions"]
    assert_consistent(result.trace)


QUERIES = [f"!(infer &self {DQ} $x)", f"!(explain &self {DQ} {CHD})", "!(diagnose-patient &self Patient001)",
           "!(recommend-supplements-patient &self Patient001)", f"!(rank-interventions &self ({DQ} Fisetin Spermidine) {CHD})",
           "!(decompose-grimage &self Patient001)", f"!(infer &self {DQ} AllCauseMortality)"]


@real_petta
def test_recording_changes_no_answer(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln", PLN_TRACE=False)
    off, result = run(_stack("patient"), None, QUERIES)
    assert result.trace is None
    monkeypatch.setattr(config, "PLN_TRACE", True)
    on, result = run(_stack("patient"), None, QUERIES)
    assert on == off
    assert_consistent(result.trace)


@real_petta
@pytest.mark.parametrize("target", ["build_trace", "watch", "drain"])
def test_a_trace_that_fails_never_fails_the_answer(pln, monkeypatch, target):
    from core import metta_runtime as mr

    def boom(*a, **k):
        raise RuntimeError("trace broke")

    if target == "build_trace":
        monkeypatch.setattr(pt, "build_trace", boom)              # the forked child inherits the patch
    else:
        monkeypatch.setattr(mr._PettaEngine, {"watch": "watch_calls", "drain": "drain_calls"}[target], boom)
    groups, result = run(_stack("patient"), None, [f"!(infer &self {DQ} {CHD})"])
    assert result.status == "ok" and len(groups[0]) == 1
    assert "trace broke" in result.trace["error"]


@real_petta
def test_the_legacy_chainer_records_nothing(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "legacy", PLN_TRACE=True)
    _, result = run(_stack("patient"), None, [f"!(infer &self {DQ} {CHD})"])
    assert result.trace is None and "trace" not in get_runtime("petta").last_run


@real_petta
@pytest.mark.parametrize("program", [
    "!(petta_trace_drain $a $b)", "!(petta_trace_note exit A B)", "!(petta_trace_watch infer x)",
    "!(call (petta_trace_drain $a $b))", "!(let $f (atom_concat petta_trace _drain) ($f $a $b))",
    "!(match petta_trace_note ($a $b $c) $a)",
])
def test_a_program_cannot_reach_the_recorder(pln, program):
    """Whatever the form does (refused, or left as data), the record of the program's own inference is intact."""
    result = pln_runner._runtime_run(f"!(infer &self {DQ} {CHD})\n{program}", 0.0, _stack("patient"), None)
    if result.status == "error":
        assert result.error_code in ("refused_form", "runtime_error")
        return
    assert [(c["source"], c["target"]) for c in result.trace["conclusions"]] == [(DQ, CHD)]
    assert result.trace["no_route"] == [] and result.trace["outside_chainer"] == []


@real_petta
def test_more_calls_than_the_record_holds_are_said(pln):
    facts = "".join(f"(TraceNode T{i})\n" for i in range(2100))
    _, result = run(_stack("patient"), facts, ["!(collapse (match &self (TraceNode $x) (infer &self $x CRP)))"])
    assert result.trace["calls_overflow"] is True
    assert "incomplete" in pt.render_trace(result.trace, 1)


@real_petta
def test_a_program_split_over_two_spaces_has_one_trace(pln):
    parts = [{"metta_query": f"!(infer &self {DQ} {CHD})", "kb_files": _stack("patient"), "extra_atoms": None},
             {"metta_query": "!(infer &self SASP CRP)", "kb_files": _stack("generic"), "extra_atoms": None}]
    merged = pln_runner.merge_run_results(pln_runner.run_query_parts(parts), [[1], [0]])
    assert [(c["source"], c["part"], c["expressions"]) for c in merged.trace["conclusions"]] == [(DQ, 1, [2]), ("SASP", 2, [1])]


@real_petta
def test_the_trace_survives_the_worker_process(pln):
    import pickle
    _, result = run(_stack("patient"), None, [f"!(infer &self {DQ} {CHD})"])
    assert pickle.loads(pickle.dumps(result)).trace == result.trace
    json.dumps(result.trace)


@real_petta
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_the_api_returns_the_level_that_was_asked_for(pln, monkeypatch, level):
    import api
    from test_api import ASGITestClient
    from fastapi.dependencies import utils as dependency_utils
    from fastapi import routing

    async def inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    for mod in (dependency_utils, routing):
        monkeypatch.setattr(mod, "run_in_threadpool", inline)
    monkeypatch.setattr(api, "PLN_RUNTIME_AVAILABLE", True, raising=False)
    monkeypatch.setattr(pln_runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr("core.executor.PLN_WORKER_POOL_SIZE", 0)
    r = ASGITestClient().post("/metta/run", json={"metta_query": f"!(explain &self {DQ} {CHD})", "trace_level": level})
    assert r.status_code == 200, r.text
    trace = r.json()["pln_trace"]
    if level == 0:
        assert trace is None
        return
    c = trace["conclusions"][0]
    assert ("steps" in c) is (level >= 2) and trace["level"] == level
    if level >= 2:
        assert ("evidence" in c["steps"][0]) is (level >= 3)


def test_the_api_refuses_a_level_it_does_not_have():
    import api
    with pytest.raises(ValueError):
        api.MettaRunRequest(metta_query="(x)", trace_level=4)


def test_the_hook_is_optional_for_a_step():
    """A plain callable is still a step: the runtime calls the trace hooks only where they exist."""
    if not base.REAL_PETTA:
        pytest.skip("PeTTa is not installed here")
    rt = PettaRuntime()
    assert rt.run_program("", ["!(+ 1 2)"], after_load=lambda engine: {"x": 1}) == [["3"]]
    assert rt.last_run["chain"] == {"x": 1} and "trace" not in rt.last_run
