"""The findings of the two reviews of the PLN_CHAINER=pln build, each reproduced and each pinned (docs/petta_migration/fork_design/fix/notes.md).

One test group per finding, cheapest first. Every group's tests FAIL without its fix: the notes name the line that was removed to
check it.

  cost            the derivation counts the walks the library really derives (links used once), refuses what a request cannot
                  afford in milliseconds with a named error, and derives only the sources the program asks about
  caller facts    a caller's fact that cannot be chained is refused (422), never answered as "no route"
  DrugAge         the fold is wired into the ranking (the NET, once per compound, no row cap), a reported null counts on both sides,
                  one compound's rows never change another's number, a row stated twice is one row
  routes          a record may close a route on its source
  config          the pln-only settings are parsed only under PLN_CHAINER=pln
  mutation survivors  the strength clamp, the step clamp, the intervention-only task
  scripts         no eval of a data file, no environment write at import

No test calls a model. The ones that need PeTTa are marked `petta` and skip without it.

    pytest tests/test_pln_fixes.py -q
    pytest tests/test_pln_fixes.py -q -m "not petta"
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

import config  # noqa: E402
import test_pln_chainer as T  # noqa: E402
from core import pln_chainer as pc  # noqa: E402
from core import pln_runner  # noqa: E402
from test_pln_chainer import (  # noqa: E402,F401  (the fixtures and helpers of the pln-lane tests)
    FEEDBACK_EDGE, SimEngine, _stack, _use, composite, contract_links, last_chain, legacy_on_petta, legacy_routes, pln, real_petta,
    run, sim_table, to_links)

THREE = REPO / "tests" / "fixtures" / "drugage_fold" / "three_compounds.metta"
PY = sys.executable


def link(a, b, sign="Pos", s=0.5, c=0.8, **kw):
    return pc.Link(a, b, sign, s, c, **kw)


def numbered(links):
    pc.assign_ids(links)
    return links


def digraph(n, prefix="D"):
    nodes = [f"{prefix}{i}" for i in range(n)]
    return numbered([link(a, b) for a in nodes for b in nodes if a != b]), nodes[0]


def chain(n, prefix="C"):
    return numbered([link(f"{prefix}{i}", f"{prefix}{i + 1}", s=0.9, c=0.9) for i in range(n)])


# ══════════════════════════════ cost: the walks the library derives ════════════════════════════════════════════════

def test_the_library_derives_walks_that_use_each_link_once_not_node_simple_routes():
    """A -> B, B -> A, B -> C. The legacy chainer's node-simple routes from A: A-B and A-B-C. The library also derives the
    walk A-B-A (it uses each LINK once), so the queue-size estimate that counted routes was too small on every cyclic graph."""
    links = numbered([link("A", "B"), link("B", "A"), link("B", "C")])
    assert pc.count_routes(links, "A", 100) == 2
    stats = pc.walk_stats(links, "A", 100)
    assert (stats.states, stats.stamp_total, stats.neg_final, stats.capped) == (3, 5, 0, False)
    assert pc.count_states(links, "A", 100) == 3
    assert pc.estimate_beliefs(links, "A", 100) == 3 + 1 + 2 * 3


def test_on_a_graph_without_a_cycle_the_walks_are_the_routes():
    rows = contract_links("patient")
    links = to_links(rows)
    for source in sorted({r["from"] for r in rows}):
        sub = pc.reachable_links(links, source)
        assert pc.count_states(sub, source, 100_000) == pc.count_routes(sub, source, 100_000) == len(legacy_routes(rows, source)), source


def test_a_feedback_edge_adds_walks_and_eight_reversed_edges_make_them_twenty_six_times_the_routes():
    cyc = to_links(contract_links("patient", [FEEDBACK_EDGE]))
    plain = to_links(contract_links("patient"))
    src = "ChronicInflammation"
    assert pc.count_states(pc.reachable_links(cyc, src), src, 10_000) > pc.count_states(pc.reachable_links(plain, src), src, 10_000)
    rows = contract_links("patient")
    reversed_edges = [{**FEEDBACK_EDGE, "from": r["to"], "to": r["from"], "sign": "Pos"} for r in rows if not r["terminal_only"]][:8]
    many = to_links(contract_links("patient", reversed_edges))
    worst = max(pc.count_states(pc.reachable_links(many, s), s, 50_000) for s in {l.src for l in many})
    assert worst == 262, worst
    routes = max(pc.count_routes(pc.reachable_links(many, s), s, 50_000) for s in {l.src for l in many})
    assert routes == 11, routes                         # node-simple routes stay in the tens: that is what the old estimate saw


def test_the_walk_count_stops_at_its_cap():
    links, src = digraph(6)
    stats = pc.walk_stats(links, src, 1000)
    assert stats.capped and stats.states == 1001
    assert pc.estimate_beliefs(links, src, 1000) > 2000


def test_a_closing_record_ends_a_walk_and_continues_none():
    links = numbered([link("A", "B"), link("B", "C", terminal=True), link("C", "D")])
    assert pc.count_states(links, "A", 100) == 2                    # A-B, A-B-C(record); the record starts nothing


# ══════════════════════════════ cost: the pre-flight ═══════════════════════════════════════════════════════════════

def test_a_complete_digraph_of_five_nodes_is_refused_in_milliseconds_with_the_number_of_walks():
    """20 links, 650 bytes of caller atoms: the library would run for minutes (K=256 takes 8 s per source, 512 34 s, 1024 140 s,
    none complete). The request is refused up front, naming the source and the walks."""
    links, src = digraph(5)
    t0 = time.perf_counter()
    with pytest.raises(pc.PlnInputError) as err:
        pc.plan_sources(links, pc.ChainSettings(), budget_s=60)
    assert time.perf_counter() - t0 < 2.0
    message = str(err.value)
    assert "more than 16,384 distinct walks over 20 reachable links" in message and "too dense or too cyclic" in message
    assert isinstance(err.value, pc.PlnChainerError)


def test_a_graph_whose_modelled_time_exceeds_the_requests_budget_is_refused_naming_the_costliest_source():
    links = chain(200)
    plans = pc.plan_sources(links, pc.ChainSettings(), sources=["C0"], budget_s=0)         # no budget: planned, never refused
    assert len(plans) == 1 and plans[0].links == 200 and plans[0].stats.states == 200 and plans[0].seconds > 20
    with pytest.raises(pc.PlnInputError, match=r"would take about \d+ s for 1 source\(s\) \(modelled\), more than the 20 s a request has") as err:
        pc.plan_sources(links, pc.ChainSettings(), sources=["C0"], budget_s=20)
    assert "C0 (200 links, 200 distinct walks" in str(err.value)
    pc.plan_sources(links, pc.ChainSettings(), sources=["C0"], budget_s=120)               # the same source fits a longer budget
    # all 200 sources of that chain are far past any budget a request has: that was the eager table
    with pytest.raises(pc.PlnInputError, match="200 source"):
        pc.plan_sources(links, pc.ChainSettings(), budget_s=60)


def test_the_shipped_graphs_plan_cheap_and_pass_the_default_budget():
    for stack in ("patient", "linage2", "generic"):
        plans = pc.plan_sources(to_links(contract_links(stack)), pc.ChainSettings(), budget_s=60)
        assert plans and sum(p.seconds for p in plans) < 1.0, stack
    cyc = pc.plan_sources(to_links(contract_links("patient", [FEEDBACK_EDGE])), pc.ChainSettings(), budget_s=60)
    assert sum(p.seconds for p in cyc) < 2.0


def test_the_research_lanes_are_never_refused_by_the_pre_flight():
    links, src = digraph(5)
    for st in (pc.ChainSettings(task_queue=64, belief_queue=64), pc.ChainSettings(ruleset="full"),
               pc.ChainSettings(on_incomplete="partial")):
        assert pc.plan_sources(links, st, sources=[src], budget_s=1)[0].stats.capped
    # and compute_table does not plan them at all
    ev = SimEngine()
    leak = pc.probe_library(ev)
    table = pc.compute_table(ev, links[:6], leak=leak, discount=0.9, settings=pc.ChainSettings(task_queue=64, belief_queue=64), sources=[src])
    assert table.status[src].attempts == 1


def test_rungs_refuse_what_no_queue_can_hold_but_the_partial_lane_runs_the_last_one():
    st = pc.ChainSettings(queue_size=16, growth=2, rungs=3)                    # 16, 32, 64
    assert pc._rungs(st, 10) == [16, 32, 64] and pc._rungs(st, 40) == [64]
    with pytest.raises(pc.PlnInputError, match="more than the largest queue \\(K=64\\)"):
        pc._rungs(st, 64)
    partial = pc.ChainSettings(queue_size=16, growth=2, rungs=3, on_incomplete="partial")
    assert pc._rungs(partial, 500) == [64]


def test_compute_table_plans_before_it_derives_and_derives_nothing_for_a_refused_graph():
    links, src = digraph(5)
    ev = SimEngine()
    leak = pc.probe_library(ev)
    with pytest.raises(pc.PlnInputError):
        pc.compute_table(ev, links, leak=leak, discount=0.9, settings=pc.ChainSettings(), sources=[src])
    assert ev.derive_calls == 0


def test_the_clock_is_checked_between_sources_and_ends_the_table_with_a_named_error():
    rows = contract_links("patient")
    links = to_links(rows)
    ev = SimEngine()
    leak = pc.probe_library(ev)
    with pytest.raises(pc.PlnChainerError, match=r"used up the request's time budget after 0 of \d+ source"):
        pc.compute_table(ev, links, leak=leak, discount=0.9, settings=pc.ChainSettings(), deadline_at=time.monotonic() - 1)
    assert ev.derive_calls == 0
    ok = pc.compute_table(SimEngine(), links, leak=leak, discount=0.9, settings=pc.ChainSettings(),
                          sources=["Spermidine"], deadline_at=time.monotonic() + 600)
    assert len(ok.routes["Spermidine"]) > 0


def test_the_model_tracks_the_measured_cost_within_a_factor_of_two():
    """The measured seconds of one PLN.Derive run on PeTTa (docs/petta_migration/fork_design/fix/COST_MODEL.md), against
    `model_seconds / 2` (two levels)."""
    measured = [  # (links, states, mean stamp, beliefs, seconds): chains, layered DAGs, rings with chords, a digraph
        (150, 150, 75.5, 301, 6.531), (120, 120, 60.5, 241, 2.79), (51, 363, 4.52, 586, 1.357), (75, 1092, 5.51, 1819, 15.63),
        (81, 3279, 6.5, 4908, 144.9), (12, 442, 6.1, 455, 1.245), (22, 704, 9.23, 1736, 28.3), (28, 1286, 10.85, 3779, 174.0)]
    for links, states, mean, beliefs, seconds in measured:
        neg = beliefs - links - 1 - states
        stats = pc.WalkStats(states, int(round(mean * states)), neg, False)
        got = pc.model_seconds(links, stats) / 2
        assert seconds / 2 < got < seconds * 2, (links, states, got, seconds)


# ══════════════════════════════ cost: only the sources the program asks about ══════════════════════════════════════

DEMAND_BLOCKS = [("deduction", (REPO / "pln_deduction.metta").read_text(encoding="utf-8")),
                 ("ranking", (REPO / "pln_intervention_ranking.metta").read_text(encoding="utf-8"))]


@pytest.mark.parametrize("expressions, want", [
    (["!(infer &self Spermidine CoronaryHeartDisease)"], {"Spermidine"}),
    (["!(explain &self Spermidine $x)", "!(infer &self CRP DNAmGDF15)"], {"Spermidine", "CRP"}),
    (["!(let $r (explain &self D+Q CoronaryHeartDisease) (R $r))"], {"D+Q"}),
    (["!(match &self (Effect Spermidine $b $s $t) $b)"], set()),                      # nothing reaches infer: nothing to derive
    (["!(+ 1 2)"], set()),
    (["!(infer &self $a CoronaryHeartDisease)"], None),                                # an unbound source: every source
    (["!(explain &self $a $c)"], None),
    (["!(let $h (match &self (Inheritance $x HallmarkOfAging) $x) (infer &self $h CoronaryHeartDisease))"], None),
    (["!(score-candidate &self Rapamycin Mortality)"], None),                          # a consumer binds the source itself
    (["!(rank-interventions &self (Rapamycin Resveratrol) Mortality)"], None),
    (["!(infer &self (Gene TP53) X)"], None),                                          # not a plain symbol
    (["!(infer &self 5 X)"], None),
    (["!(match &self (PLNAnswer $a $c $g $t $n $i) $a)"], None),                       # a program that reads the table wants it whole
    (["!(match &self (PLNStatus $s $st $k) $s)"], None),
    (["!(match &self (PLNEvidence $id $kind $w $t) $id)"], set()),                      # the evidence table needs no derivation
    (["!(let $f infer ($f &self A B))"], None),                                        # `infer` passed as a value
    (["!(infer &self A B C)"], None),                                                  # a malformed call is not guessed at
    (["!(infer &self Spermidine CoronaryHeartDisease) ; (infer &self $z B)"], {"Spermidine"}),     # a comment is not code
], ids=lambda v: str(v)[:60])
def test_demand_of_reads_the_expressions_and_when_in_doubt_asks_for_everything(expressions, want):
    got = pc.demand_of(DEMAND_BLOCKS, expressions)
    assert got == (None if want is None else frozenset(want))


def test_unknown_expressions_mean_every_source():
    assert pc.demand_of(DEMAND_BLOCKS, None) is None


def test_the_functions_that_reach_infer_are_found_through_other_functions():
    reaching = pc.chain_reaching_heads(DEMAND_BLOCKS)
    assert {"score-candidate", "rank-interventions"} <= reaching and "infer" not in reaching and "explain" not in reaching
    extra = [("caller", "(= (my-wrapper $a) (score-candidate &self $a Mortality))\n(= (my-other $a) (my-wrapper $a))\n")]
    assert {"my-wrapper", "my-other"} <= pc.chain_reaching_heads(DEMAND_BLOCKS + extra)
    assert pc.demand_of(DEMAND_BLOCKS + extra, ["!(my-other Rapamycin)"]) is None


def test_every_consumer_of_the_shipped_stacks_is_seen_as_reaching_infer():
    """The call graph is built from the knowledge-base text: every function the consumers (diagnosis, counterfactual, risk
    projection, supplements, rankings) call `infer` through must be in the set, or a narrower table would answer 'no'."""
    import api
    blocks = [(path.name, path.read_text(encoding="utf-8")) for path in api._runtime_kb_paths()]
    reaching = pc.chain_reaching_heads(blocks)
    for name in ("score-candidate", "rank-interventions", "rank-interventions-for-patient", "score-candidate-patient", "patient-relevance",
                 "diagnose", "diagnose-patient", "counterfactual", "counterfactual-patient", "project-risk", "project-risk-patient",
                 "recommend-supplements", "recommend-supplements-patient", "pos-transmission", "personalized-score"):
        assert name in reaching, name
    # the functions that do NOT reach it really do not: they are arithmetic on the patient's own values
    for name in ("predict-risk-patient", "risk-decomposition", "patient-z"):
        assert name not in reaching
        assert not any(sym in reaching for _, uses in [d for _, t in blocks for d in pc._definitions(t) if d[0] == name] for sym in uses), name
    for line in ("!(rank-interventions-for-patient &self Patient001 (Spermidine Metformin) CoronaryHeartDisease)",
                 "!(counterfactual-patient &self Patient001 CellularSenescence)", "!(diagnose-patient &self Patient001)"):
        assert pc.demand_of(blocks, [line]) is None, line


@real_petta
def test_the_composite_queries_that_the_analysis_narrows_equal_the_legacy_chainer_run_alone(composite, monkeypatch, tmp_path):
    """43 of the 50 contract queries call a consumer (every source); 4 name a literal source; 3 reach nothing (the risk
    projections: arithmetic on the patient's values). Run ALONE, the 7 narrowed ones must give what the legacy chainer gives:
    a derivation skipped for a source the program needed would show here as a missing answer."""
    import api
    blocks = [(p.name, p.read_text(encoding="utf-8")) for p in pln_runner.patient_stack(api._runtime_kb_paths())]
    narrowed = []
    for i, entry in enumerate(T.PINNED):
        demand = pc.demand_of(blocks, pln_runner._bang_expressions(entry["query"]))
        if demand is not None:
            narrowed.append((i, entry, demand))
    assert len(narrowed) == 7 and sorted(len(d) for _, _, d in narrowed) == [0, 0, 0, 1, 1, 1, 1]
    _use(monkeypatch, tmp_path, "pln")
    for i, entry, demand in narrowed:
        groups, _ = run(_stack("patient"), None, [entry["query"]])
        assert last_chain()["demand"] == sorted(demand) and last_chain()["sources"] <= len(demand), entry["tag"]
        want = composite["legacy"][i]
        assert len(groups[0]) == len(want), entry["tag"]
        for g, w in zip(groups[0], want):
            (gs, gn), (ws, wn) = T._skeleton(g), T._skeleton(w)
            assert gs == ws and gn == pytest.approx(wn, abs=1e-9), (entry["tag"], g[:160], w[:160])


@real_petta
def test_a_literal_source_derives_one_source_and_the_answer_equals_the_eager_table(monkeypatch, tmp_path):
    """The narrower table is the same answers, never fewer: for EVERY node of the patient graph, the explain answers of a
    program that names the node equal the ones of a program that forces every source."""
    _use(monkeypatch, tmp_path, "pln")
    rows = contract_links("patient")
    nodes = sorted({r["from"] for r in rows} | {r["to"] for r in rows})
    eager, _ = run(_stack("patient"), None, ["!(explain &self $a $c)"])
    assert last_chain()["demand"] is None and last_chain()["sources"] == 20
    by_source: dict[str, list[str]] = {}
    for atom in eager[0]:
        by_source.setdefault(atom.split()[1].strip("("), []).append(atom)
    narrowed = 0
    for node in nodes:
        groups, _ = run(_stack("patient"), None, [f"!(explain &self {node} $x)"])
        s = last_chain()
        assert s["demand"] == [node] and s["sources"] == (1 if node in by_source else 0), node
        assert sorted(groups[0]) == sorted(by_source.get(node, [])), node
        narrowed += s["sources"]
    assert narrowed == 20


@real_petta
def test_a_program_that_only_reads_the_graph_derives_nothing_and_still_writes_the_evidence_table(pln):
    groups, _ = run(_stack("patient"), None, ["!(match &self (Effect Spermidine $b $s $t) $b)",
                                              "!(match &self (PLNEvidence $id $kind $where $tier) $id)"])
    s = last_chain()
    assert s["demand"] == [] and s["sources"] == 0 and s["derive_runs"] == 0 and s["routes"] == 0 and len(s["evidence"]) == 28
    assert len(groups[0]) >= 1 and len(groups[1]) == 28
    assert s["timing_s"]["library"] == 0.0


@real_petta
def test_junk_beside_a_two_link_question_costs_nothing_for_the_junk(pln):
    """The reviewer's shape: an 80-link dense block as UNRELATED junk next to `infer A C`: killed at 30 s before, one source now."""
    import random
    rnd = random.Random(5)
    junk = [(f"J{d}_{i}", f"J{d + 1}_{j}") for d in range(5) for i in range(4) for j in range(4)]
    extra = "(Effect A B Pos (stv 0.5 0.8))\n(Effect B C Pos (stv 0.5 0.8))\n" + "".join(
        f"(Effect {a} {b} {rnd.choice(['Pos', 'Neg'])} (stv 0.6 0.7))\n" for a, b in junk)
    groups, _ = run(_stack("minimal"), extra, ["!(infer &self A C)"])
    s = last_chain()
    assert len(groups[0]) == 1 and T._skeleton(groups[0][0])[0] == T._skeleton("(signed Pos (stv 0.25 0.576))")[0]
    assert T._skeleton(groups[0][0])[1] == pytest.approx([0.25, 0.576], abs=1e-12)
    assert s["sources"] == 1 and s["derive_runs"] == 2 and s["demand"] == ["A"]


@real_petta
def test_a_graph_the_library_cannot_derive_is_refused_as_422_naming_the_walks_before_any_derivation(pln):
    nodes = list("ABCDE")
    extra = "".join(f"(Effect {a} {b} Pos (stv 0.5 0.8))\n" for a in nodes for b in nodes if a != b)
    t0 = time.perf_counter()
    groups, result = run(_stack("minimal"), extra, ["!(explain &self A E)"], allow_error=True)
    assert time.perf_counter() - t0 < 20
    assert result.status == "error" and result.error_code == "refused_form" and result.results == []
    assert "distinct walks over 20 reachable links" in result.error and "refused the program" in result.error
    # ... while a question about a node that reaches only part of it is answered
    ok, res = run(_stack("minimal"), extra + "(Effect Z A Pos (stv 0.5 0.8))\n(Effect Q R Pos (stv 0.5 0.8))\n", ["!(explain &self Q $x)"])
    assert ok[0] == ["(Chain (Q R) (signed Pos (stv 0.5 0.8)))"]


@real_petta
def test_a_step_that_runs_out_of_time_says_so_in_a_named_error(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln", PLN_QUERY_TIMEOUT_SECONDS=60)
    monkeypatch.setattr(pc, "_check_clock", lambda deadline_at, done, total: (_ for _ in ()).throw(
        pc.PlnChainerError(f"the derivation used up the request's time budget after {done} of {total} source(s)")))
    _, result = run(_stack("patient"), None, ["!(explain &self Spermidine $x)"], allow_error=True)
    assert result.status == "error" and "used up the request's time budget" in result.error and result.error_code == "runtime_error"


# ══════════════════════════════ caller facts ═══════════════════════════════════════════════════════════════════════

def test_the_refusal_names_up_to_five_atoms_and_counts_the_rest():
    skipped = [{"from": f"A{i}", "to": "B", "reason": "StrengthOutOfRange", "kind": "effect", "origin": "caller",
                "atom": f"(Effect A{i} B Pos (stv 1.5 0.5))"} for i in range(8)]
    text = pc.caller_skips_message(skipped)
    assert text.count("(Effect A") == 5 and "(and 3 more)" in text and "the strength is not in [0, 1]" in text
    assert pc.caller_skips_message([{**skipped[0], "origin": "curated"}]) is None
    assert pc.caller_skips_message([]) is None


def test_an_input_error_is_a_chainer_error_and_maps_to_a_refusal_not_a_service_error():
    from core.metta_runtime import PettaRuntime
    rt = PettaRuntime()
    refused = rt._error_from({"stage": "chain", "type": "PlnInputError", "message": "bad atom"})
    assert refused.kind == "refused" and "refused the program" in str(refused) and "bad atom" in str(refused)
    failed = rt._error_from({"stage": "chain", "type": "PlnChainerError", "message": "did not finish"})
    assert failed.kind == "chain" and "could not finish" in str(failed)
    other = rt._error_from({"stage": "chain", "type": "KeyError", "message": "x"})
    assert other.kind == "chain" and "failed" in str(other)


@real_petta
def test_the_reviewers_http_shape_a_strength_of_one_and_a_half_is_a_422_not_an_empty_answer(pln):
    extra = "(Effect Foo Bar Pos (stv 1.5 0.5))\n(Effect Bar Baz Pos (stv 0.5 0.5))\n"
    groups, result = run(_stack("minimal"), extra, ["!(infer &self Foo Baz)"], allow_error=True)
    assert result.status == "error" and result.error_code == "refused_form" and "(Effect Foo Bar Pos (stv 1.5 0.5))" in result.error
    for bad in ("(stv -0.1 0.5)", "(stv 50 0.9)", "(stv nan 0.5)"):
        g, r = run(_stack("minimal"), f"(Effect Foo Bar Pos {bad})\n", ["!(infer &self Foo Bar)"], allow_error=True)
        assert r.status == "error" and r.error_code == "refused_form", bad


# ══════════════════════════════ routes: a record that returns to its source ═════════════════════════════════════════

_BACK = """
(: BackRec EmpiricalAssociation)
(HasPredictor BackRec Marker)
(HasOutcome BackRec Source)
(EffectMetric BackRec "hazard_ratio")
(EffectValue BackRec 1.3)
(EffectScale BackRec "x")
(PValue BackRec 0.001)
(SupportedByPublication BackRec LuEtAl2019_GrimAge)
(HasEvidenceCategory BackRec MultipleHumanTrials)
(Effect Source Marker Neg (stv 0.5 0.9))
"""


@real_petta
def test_a_record_that_closes_a_route_on_its_source_is_answered_as_the_legacy_chainer_answers_it(monkeypatch, tmp_path):
    """Source -> Marker (Effect), then a hazard-ratio record Marker -> Source: legacy `explain Source Source` answers the closed
    route (a record is terminal, so it starts no cycle); the first build dropped every conclusion that ended at the source."""
    import api
    stack = pln_runner.patient_stack(api._runtime_kb_paths())
    queries = ["!(explain &self Source Source)", "!(explain &self Source Marker)"]
    _use(monkeypatch, tmp_path, "legacy")
    want, _ = run(stack, _BACK, queries)
    _use(monkeypatch, tmp_path, "pln")
    got, _ = run(stack, _BACK, queries)
    assert len(want[0]) == 1 and want[0][0].startswith("(Chain (Source Marker Source)")
    for a, b in zip(got[0] + got[1], want[0] + want[1]):
        (sa, na), (sb, nb) = T._skeleton(a), T._skeleton(b)
        assert sa == sb and na == pytest.approx(nb, abs=1e-12), (a, b)
    assert len(got[0]) == 1
    assert len(got[1]) == len(want[1]) == 1


@real_petta
def test_an_effect_cycle_back_to_the_source_is_still_not_a_route(pln):
    extra = "(Effect A B Pos (stv 0.5 0.8))\n(Effect B A Pos (stv 0.5 0.8))\n(Effect B C Pos (stv 0.5 0.8))\n"
    groups, _ = run(_stack("minimal"), extra, ["!(explain &self A A)", "!(explain &self A B)", "!(explain &self A C)"])
    assert groups[0] == [] and len(groups[1]) == 1 and len(groups[2]) == 1
    assert last_chain()["status"]["A"]["complete"] and last_chain()["routes"] >= 2


def test_the_trail_of_a_record_may_name_a_visited_node_and_an_effect_may_not():
    links = numbered([link("A", "B"), link("B", "A", terminal=True), link("B", "C")])
    enc = pc._Encoding(links, 0.02)
    ab, ba, bc = links
    assert [l.id for l in pc._trail(enc, "A", [ab.id, ba.id], end="A", end_terminal=True, simple=True)] == [ab.id, ba.id]
    plain = numbered([link("A", "B"), link("B", "A")])
    enc2 = pc._Encoding(plain, 0.02)
    assert pc._trail(enc2, "A", [plain[0].id, plain[1].id], end="A", end_terminal=False, simple=True) is None
    assert pc._trail(enc2, "A", [plain[0].id, plain[1].id], end="A", end_terminal=False, simple=False) is not None


# ══════════════════════════════ DrugAge ════════════════════════════════════════════════════════════════════════════

def _drugage_items_for(compound, rows):
    from ontology import drugage_fold as df
    out = []
    for r in rows:
        if r.compound != compound:
            continue
        lr = df.lift_row(r)
        if lr is None:
            continue
        out.append({"compound": compound, "target": "Lifespan", "direction": "Pos" if lr.sign == "Neg" else "Neg",
                    "strength": lr.strength, "confidence": lr.native_confidence, "rows": (r.row_id,), "papers": (r.pmid or "",),
                    "change": r.avg_change, "significance": r.significance, "itp": r.is_itp, "species": r.species})
    return out


def test_the_chainers_null_conventions_are_the_folds():
    from ontology import drugage_fold as df
    assert pc.ZERO_ROW_CONVENTIONS == df.ZERO_ROW_CONVENTIONS and pc.ChainSettings().drugage_zero_rows == df.DEFAULT_ZERO_ROWS
    with pytest.raises(ValueError, match="drugage_zero_rows"):
        pc.ChainSettings(drugage_zero_rows="nowhere")


def test_a_row_pasted_twice_is_one_row_and_a_row_that_disagrees_with_itself_is_listed():
    base = {"compound": "X", "target": "Lifespan", "direction": "Pos", "strength": 0.3, "confidence": 0.5, "rows": ("R1",), "papers": ("PMID_1",),
            "change": 10.0, "significance": "Significant", "itp": False, "species": "Mus_musculus"}
    skipped: list[dict] = []
    skip = lambda a, b, reason, kind, row="", atom="": skipped.append({"reason": reason, "row": row, "atom": atom})
    assert pc._sanitize_drugage([base, dict(base), dict(base)], skip) == [base] and skipped == []
    other = {**base, "strength": 0.9}
    assert pc._sanitize_drugage([base, other, {**base, "rows": ("R2",), "papers": ("PMID_2",)}], skip) == [
        {**base, "rows": ("R2",), "papers": ("PMID_2",)}]
    assert [s["reason"] for s in skipped] == ["RowStatedTwiceDifferently"] and skipped[0]["row"] == "R1"
    twice_compound = {**base, "compound": "Y"}
    assert pc._sanitize_drugage([base, twice_compound], skip) == [] and len(skipped) == 2


@real_petta
def test_a_drugage_row_pasted_twice_is_one_study_where_the_legacy_chainer_returned_sixty_four_answers(monkeypatch, tmp_path):
    rows = (lambda r: f"(InstanceOf {r} Experiment)\n(UsesIntervention {r} X)\n(UsesSpecies {r} Mus_musculus)\n(HasSex {r} Male)\n"
                      f"(ReportedIn {r} PMID_1)\n(AvgLifespanChangePercent {r} 10.0)\n(AvgLifespanSignificance {r} Significant)\n")
    _use(monkeypatch, tmp_path, "pln")
    twice = rows("DrugAgeRow_1") + rows("DrugAgeRow_1")
    one, _ = run(_stack("drugage"), rows("DrugAgeRow_1"), ["!(infer &self X Mortality)"])
    assert len(one[0]) == 1 and T._skeleton(one[0][0])[1] == pytest.approx([1 / 3, 0.45], abs=1e-12)
    groups, _ = run(_stack("drugage"), twice, ["!(infer &self X Mortality)"])
    assert groups[0] == one[0]
    thrice_atom = rows("DrugAgeRow_1") + "(AvgLifespanChangePercent DrugAgeRow_1 10.0)\n"
    groups, _ = run(_stack("drugage"), thrice_atom, ["!(infer &self X Mortality)"])
    assert groups[0] == one[0]


@real_petta
@pytest.mark.parametrize("extra_atom, what", [("(AvgLifespanChangePercent DrugAgeRow_1 -20.0)\n", "two changes"),
                                              ("(UsesIntervention DrugAgeRow_1 Y)\n", "two compounds")])
def test_a_drugage_row_that_contradicts_itself_is_refused_with_the_row_named_not_a_502(pln, extra_atom, what):
    row = ("(InstanceOf DrugAgeRow_1 Experiment)\n(UsesIntervention DrugAgeRow_1 X)\n(UsesSpecies DrugAgeRow_1 Mus_musculus)\n"
           "(ReportedIn DrugAgeRow_1 PMID_1)\n(AvgLifespanChangePercent DrugAgeRow_1 10.0)\n(AvgLifespanSignificance DrugAgeRow_1 Significant)\n")
    _, result = run(_stack("drugage"), row + extra_atom, ["!(infer &self X Mortality)"], allow_error=True)
    assert result.status == "error" and result.error_code == "refused_form", what
    assert "DrugAgeRow_1" in result.error and "stated twice with different values" in result.error


class _DrugAgeSpace:
    """Enough of a space for `collect_links`: it answers the DrugAge queries with the given rows (row, compound, strength, confidence)."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, expr):
        if expr == pc._DRUGAGE_Q:
            return [f"(D {r} {c} 10.0 (Effect {c} Lifespan Pos (stv {s} {conf})))" for r, c, s, conf in self.rows]
        if expr == pc._PAPER_Q:
            return [f"(P {r} PMID_{i})" for i, (r, *_rest) in enumerate(self.rows)]
        return []


def test_the_fold_is_called_compound_by_compound_so_one_refused_group_is_a_listed_skip():
    """`Truth_Revision` has no result for a pooled confidence of 0; the first build let that ValueError fail the whole request
    (a 502). Now the compound is skipped with the reason and the others are folded."""
    from ontology import drugage_fold as df
    space = _DrugAgeSpace([("R1", "Good", 0.5, 0.5), ("R2", "Bad", 0.5, 0.0), ("R3", "Bad", 0.5, 0.0)])
    blocks = [("kb", "(= (drugage-effect $s $r) (x))\n"), ("extra_atoms", "")]
    with pytest.raises(df.RevisionUndefined):
        df.fold_items([{"compound": "Bad", "target": "Lifespan", "direction": "Pos", "strength": 0.5, "confidence": 0.0, "rows": (r,),
                        "papers": (p,)} for r, p in (("R2", "PMID_1"), ("R3", "PMID_2"))])
    links, skipped, _ = pc.collect_links(space, blocks)
    assert [(l.src, l.sign, l.kind) for l in links] == [("Good", "Pos", "drugage")]
    assert [(s["from"], s["reason"], s["kind"], s["origin"]) for s in skipped] == [("Bad", "FoldRefused:RevisionUndefined", "drugage", "curated")]
    # a skip that comes from the caller's rows is a refusal
    again, skipped2, _ = pc.collect_links(space, [("kb", blocks[0][1]), ("extra_atoms", "(UsesIntervention R2 Bad)\n(UsesIntervention R3 Bad)\n")])
    assert [s["origin"] for s in skipped2] == ["caller"] and "the fold refused it (RevisionUndefined)" in pc.caller_skips_message(skipped2)


# the engine <-> twin parity: the fold in the chainer's links is the fold of the module, row for row (hook 1 of the fold notes)

def test_the_chainers_drugage_links_are_the_folds_items():
    """collect_links keeps the policy keys, so the within-paper pick is the repo's representative-row policy and the links
    equal `fold_rows` (strength, native confidence, rows, papers) to 1e-12."""
    from ontology import drugage_fold as df
    from ontology.drugage_selector import load_rows
    rows = load_rows(THREE)
    for compound in ("Resveratrol", "Rapamycin"):
        out = df.fold_items(_drugage_items_for(compound, rows))
        want = df.fold_compound([r for r in rows if r.compound == compound])
        got = {o["mortality_sign"]: o for o in out}
        for sign, item in (("Neg", want.neg), ("Pos", want.pos)):
            if item is None:
                assert sign not in got
                continue
            assert got[sign]["strength"] == pytest.approx(item.strength, abs=1e-12)
            assert got[sign]["confidence"] == pytest.approx(item.native_confidence, abs=1e-12)
            assert got[sign]["rows"] == tuple(m.row.row_id for m in item.members)
            assert got[sign]["papers"] == tuple(m.row.pmid for m in item.members)
            assert got[sign]["within_paper_pick"] == "representative_row_policy"


@real_petta
def test_the_engine_answers_for_a_compound_are_the_folds_net_parts(pln):
    """`explain <compound> Mortality` over the 313-row fixture: one Pos and one Neg route whose strength is the item's strength
    and whose confidence is native x 0.9, for the chainer's default null convention."""
    from ontology import drugage_fold as df
    from ontology.drugage_selector import load_rows
    rows = load_rows(THREE)
    text = "".join(r.block for r in rows)
    groups, _ = run(_stack("drugage"), text, [f"!(explain &self {c} Mortality)" for c in ("Resveratrol", "Rapamycin", "Ethanol")])
    for compound, atoms in zip(("Resveratrol", "Rapamycin", "Ethanol"), groups):
        want = df.fold_compound([r for r in rows if r.compound == compound])
        got = {}
        for a in atoms:
            m = T._STV_CHAIN.match(a)
            got[m.group(2)] = (float(m.group(3)), float(m.group(4)))
        for sign, item in (("Neg", want.neg), ("Pos", want.pos)):
            if item is None:
                assert sign not in got, (compound, sign)
                continue
            assert got[sign][0] == pytest.approx(item.strength, abs=1e-12), (compound, sign)
            assert got[sign][1] == pytest.approx(item.confidence, abs=1e-12), (compound, sign)


@real_petta
def test_run_drugage_ranking_lists_a_compound_once_by_its_net_and_equals_the_python_twin(monkeypatch, tmp_path):
    """The reviewers' M2: the engine ranking kept the best derivation per compound and so dropped the harmful item (Resveratrol
    +0.2145 against its defined NET). Under pln the compound is listed once, by the net of its two derivations, equal to
    `top_compounds` -- for every null convention the chainer can be asked for."""
    from ontology import drugage_fold as df
    from ontology.drugage_selector import load_rows
    rows = load_rows(THREE)
    names = ["Rapamycin", "Resveratrol", "Metformin", "Ethanol"]
    for zero_rows in df.ZERO_ROW_CONVENTIONS:
        _use(monkeypatch, tmp_path, "pln")
        monkeypatch.setattr(pc, "settings_from_config", lambda z=zero_rows: pc.ChainSettings(drugage_zero_rows=z))
        for strategy in ("linear", "metta_sort"):
            result, picked = pln_runner.run_drugage_ranking(names, source=THREE, limit=None, strategy=strategy)
            assert result.status == "ok" and len(picked) == 313               # every row of the four compounds, past the 150 cap
            scored = pln_runner.parse_scored(result.results[0].atom)
            twin = df.top_compounds(rows, n=10, direction="any", zero_rows=zero_rows)
            assert [s.compound for s in scored] == [e.compound for e in twin.entries], (zero_rows, strategy)
            assert len(scored) == 4 and len({s.compound for s in scored}) == 4            # listed once
            for s, e in zip(scored, twin.entries):
                assert s.score == pytest.approx(e.score, abs=1e-12) and s.confidence == pytest.approx(e.confidence, abs=1e-12), (zero_rows, s.compound)
                assert s.strength == pytest.approx(e.strength, abs=1e-12)
    # and the point of it: the default NET puts Resveratrol where its evidence is, not where its better half is
    _use(monkeypatch, tmp_path, "pln")
    monkeypatch.undo()
    _use(monkeypatch, tmp_path, "pln")
    result, _ = pln_runner.run_drugage_ranking(names, source=THREE, limit=None)
    by = {s.compound: s for s in pln_runner.parse_scored(result.results[0].atom)}
    assert by["Resveratrol"].score == pytest.approx(0.1363, abs=5e-5) and by["Metformin"].score == pytest.approx(0.1182, abs=5e-5)
    assert by["Rapamycin"].score == pytest.approx(0.3368, abs=5e-5) and by["Rapamycin"].confidence == pytest.approx(0.8801, abs=5e-5)


@real_petta
def test_the_legacy_ranking_is_unchanged_and_the_pln_ranking_without_the_fold_is_the_legacy_one(monkeypatch, tmp_path):
    names = ["Rapamycin", "Resveratrol", "Metformin", "Ethanol"]
    _use(monkeypatch, tmp_path, "legacy")
    legacy, rows = pln_runner.run_drugage_ranking(names, source=THREE)
    assert len(rows) == 4 and [s.compound for s in pln_runner.parse_scored(legacy.results[0].atom)] == ["Rapamycin", "Ethanol", "Metformin", "Resveratrol"]
    assert pln_runner.parse_scored(legacy.results[0].atom)[0].confidence == pytest.approx(0.81)
    # the representative-row ranking, row for row, with the fold switched off (the lift-fidelity check)
    _use(monkeypatch, tmp_path, "pln")
    monkeypatch.setattr(pc, "settings_from_config", lambda: pc.ChainSettings(drugage_fold=False))
    unfolded, rows2 = pln_runner.run_drugage_ranking(names, source=THREE, best_per_compound=True, limit=None)
    # ... pln still selects every row (the fold is its business), the unfolded links give one derivation per row, and the
    # compound's net takes the best of each direction: the ranking is defined, finite and lists each compound once
    assert unfolded.status == "ok" and len(pln_runner.parse_scored(unfolded.results[0].atom)) == 4


def test_net_per_compound_keeps_one_derivation_untouched_and_nets_two():
    one = pln_runner.ScoredCompound("A", 0.3, "Neg", 0.5, 0.6, "(scored A 0.3 (signed Neg (stv 0.5 0.6)))")
    pos = pln_runner.ScoredCompound("B", -0.2, "Pos", 0.4, 0.5, "(scored B -0.2 (signed Pos (stv 0.4 0.5)))")
    neg = pln_runner.ScoredCompound("B", 0.35, "Neg", 0.5, 0.7, "(scored B 0.35 (signed Neg (stv 0.5 0.7)))")
    out = {s.compound: s for s in pln_runner.net_per_compound([pos, one, neg])}
    assert out["A"] is one
    assert out["B"].score == pytest.approx(0.35 - 0.2) and out["B"].sign == "Neg" and out["B"].confidence == 0.7
    assert out["B"].strength * out["B"].confidence == pytest.approx(out["B"].score)
    assert pln_runner.parse_scored(out["B"].atom)[0].score == out["B"].score
    # two derivations of one sign (an unfolded research setting): the stronger stands for the direction, nothing raises
    weak = pln_runner.ScoredCompound("C", 0.1, "Neg", 0.2, 0.5, "(scored C 0.1 (signed Neg (stv 0.2 0.5)))")
    strong = pln_runner.ScoredCompound("C", 0.3, "Neg", 0.5, 0.6, "(scored C 0.3 (signed Neg (stv 0.5 0.6)))")
    assert pln_runner.net_per_compound([weak, strong])[0] is strong


@real_petta
def test_the_drugage_ranking_route_reports_pooled_provenance_and_the_fold_semantics(monkeypatch, tmp_path):
    from core import drugage_router as dr
    _use(monkeypatch, tmp_path, "pln")
    ranking = dr.rank_drugage(["Rapamycin", "Resveratrol"], source=THREE, include_all_rows=False)
    atoms = [a.atom for a in ranking.result.results]
    bullets = [a for a in atoms if "pooled by paper" in a]
    assert len(bullets) == 2 and not any("avg lifespan" in a for a in atoms)          # no per-row bullet for a pooled number
    res = next(b for b in bullets if b.startswith("Resveratrol"))
    assert "protective: 35 paper(s)" in res and "harmful: 10 paper(s), 81 row(s)" in res and "net +0.1363" in res
    sem = dr.score_semantics()
    assert sem["fold"]["zero_rows"] == "both" and sem["fold"]["net"].startswith("net = sum") and sem["sign"] == dr.SCORE_SEMANTICS["sign"]
    _use(monkeypatch, tmp_path, "legacy")
    assert dr.score_semantics() is dr.SCORE_SEMANTICS                                  # unchanged object under the default
    legacy = dr.rank_drugage(["Rapamycin", "Resveratrol"], source=THREE, include_all_rows=False)
    assert not any("pooled by paper" in a.atom for a in legacy.result.results) and any("avg lifespan" in a.atom for a in legacy.result.results)


def test_drugage_top_follows_the_chainer_and_the_default_ranking_is_untouched(monkeypatch, tmp_path):
    from core import drugage_router as dr
    from ontology import drugage_fold as df
    from ontology.drugage_selector import load_rows
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    legacy = dr.drugage_top(n=5, source=THREE, direction="any")
    assert legacy.fold is None and type(legacy.entries[0]).__name__ == "RowScore"
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    folded = dr.drugage_top(n=5, source=THREE, direction="any")
    twin = df.top_compounds(load_rows(THREE), n=5, direction="any")
    assert folded.fold is not None and folded.fold["zero_rows"] == "both"
    assert [(e.compound, e.score) for e in folded.entries] == [(e.compound, e.score) for e in twin.entries]
    assert (folded.total_compounds, folded.total_rows) == (twin.total_compounds, twin.total_rows)
    assert dr.drugage_top(n=5, source=THREE, direction="any", fold=False).fold is None            # the switch overrides the chainer
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    assert dr.drugage_top(n=5, source=THREE, direction="any", fold=True).fold is not None


LEGACY_TOP_ENTRY_KEYS = {"rank", "compound", "score", "sign", "direction", "strength", "confidence", "evidence_tier", "species", "sex",
                         "is_itp", "significance", "avg_lifespan_change_percent", "pmid", "row_id"}
LEGACY_TOP_KEYS = {"entries", "total_compounds", "total_compounds_in_source", "unscorable_compounds", "total_rows", "scored_rows",
                   "unscorable_rows", "source", "filters", "semantics"}


def test_the_http_top_response_has_exactly_the_legacy_keys_by_default_and_the_fold_fields_only_under_pln(monkeypatch):
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    client = TestClient(api.app)
    body = client.get("/drugage/top?n=2").json()
    assert set(body) == LEGACY_TOP_KEYS and all(set(e) == LEGACY_TOP_ENTRY_KEYS for e in body["entries"])
    assert body["semantics"] == api.SCORE_SEMANTICS
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    folded = client.get("/drugage/top?n=2").json()
    assert set(folded) == LEGACY_TOP_KEYS | {"fold"} and folded["fold"]["zero_rows"] == "both"
    assert all(set(e) == LEGACY_TOP_ENTRY_KEYS | {"papers_protective", "papers_harmful", "rows_protective", "rows_harmful"} for e in folded["entries"])
    assert "fold" in folded["semantics"]


# ══════════════════════════════ config ═════════════════════════════════════════════════════════════════════════════

def _import_config(**env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("PLN_")}
    clean.update({"PYTHON_DOTENV_DISABLED": "1", "PLN_RUNTIME": "hyperon", **env})
    code = f"import sys; sys.path.insert(0, {str(PLN_CHAT)!r}); import config as c; print(c.PLN_CHAINER, c.PLN_QUEUE_SIZE, c.PLN_MAX_STEPS, c.PLN_RULESET, c.PLN_CHAIN_CONFIDENCE)"
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, env=clean, timeout=120)


@pytest.mark.parametrize("name, value", [("PLN_QUEUE_SIZE", ""), ("PLN_QUEUE_SIZE", "abc"), ("PLN_MAX_STEPS", "x"), ("PLN_RULESET", "bogus"),
                                         ("PLN_CHAIN_CONFIDENCE", "zzz")])
def test_a_malformed_pln_only_setting_cannot_stop_the_default_service_from_starting(name, value):
    done = _import_config(**{name: value})
    assert done.returncode == 0, done.stderr[-300:]
    assert done.stdout.split() == ["legacy", "256", "0", "restricted", "legacy"]
    assert name in done.stderr                                     # ignored with a warning, not silently


def test_the_same_malformed_setting_is_a_configuration_error_under_pln():
    for name, value in (("PLN_QUEUE_SIZE", "abc"), ("PLN_MAX_STEPS", "x"), ("PLN_RULESET", "bogus"), ("PLN_CHAIN_CONFIDENCE", "zzz")):
        done = _import_config(PLN_RUNTIME="petta", PLN_CHAINER="pln", **{name: value})
        assert done.returncode != 0 and "ValueError" in done.stderr, name
    ok = _import_config(PLN_RUNTIME="petta", PLN_CHAINER="pln", PLN_QUEUE_SIZE="512", PLN_MAX_STEPS="40", PLN_RULESET="full",
                        PLN_CHAIN_CONFIDENCE="native")
    assert ok.returncode == 0 and ok.stdout.split() == ["pln", "512", "40", "full", "native"]


def test_an_unknown_chainer_name_is_still_an_error_with_the_default_runtime():
    done = _import_config(PLN_CHAINER="plm")
    assert done.returncode != 0 and "PLN_CHAINER must be legacy or pln" in done.stderr


# ══════════════════════════════ the survivors of the mutation run ═══════════════════════════════════════════════════

def test_a_route_strength_never_exceeds_one_though_the_difference_of_two_levels_can():
    """|s1 - s0| / 0.98^hops lands one ulp above 1.0 for p = 1.0 chains (33 of 180 raw strengths, by up to 2.2e-16); a strength
    above 1 trips every downstream range check. The clamp pins it."""
    over = 0
    for hops in range(1, 10):
        rows = [{"from": f"N{i}", "to": f"N{i + 1}", "sign": "Pos", "strength": 1.0, "confidence": 1.0, "terminal_only": False} for i in range(hops)]
        table = sim_table(rows, sources=["N0"])
        for r in table.routes["N0"]:
            assert r.strength <= 1.0
            over += abs(r.level1 - r.level0) / (1 - 0.02) ** r.hops > 1.0
    assert over > 0, "the premise of the clamp: the raw ratio does exceed 1.0 for some chain"


def _derive_calls(ev):
    return [pc.parse_sexpr(c) for c in ev.calls if c.startswith("(PLN.Derive")]


def test_the_step_bound_is_clamped_to_one_through_a_million():
    rows = contract_links("patient")
    for configured, want in ((-7, 1), (10 ** 9, 1_000_000), (0, 4 * 256)):
        ev = SimEngine()
        leak = pc.probe_library(ev)
        try:
            pc.compute_table(ev, to_links(rows), leak=leak, discount=0.9, settings=pc.ChainSettings(max_steps=configured),
                             sources=["Spermidine"])
        except pc.PlnChainerError:
            pass                                                   # one step cannot finish the table; that is not under test
        calls = _derive_calls(ev)
        assert calls and int(calls[0][4]) == want, (configured, calls[0][4])


def test_the_first_derive_of_the_restricted_lane_has_exactly_one_task_the_intervention():
    """Decision section 3: Tasks = (applied). `PLN.Query kb term` seeds every sentence as a task and loses 48 of 100 pairs."""
    rows = contract_links("patient")
    ev = SimEngine()
    pc.compute_table(ev, to_links(rows), leak=pc.probe_library(ev), discount=0.9, settings=pc.ChainSettings(), sources=["Spermidine"])
    call = _derive_calls(ev)[0]
    tasks, beliefs = call[1], call[2]
    assert len(tasks) == 1 and tasks[0][0] == "Sentence" and tasks[0][1][0][0] == "State" and tasks[0][2] == ("0",)
    assert len(beliefs) > 1 and tasks[0] in beliefs                       # the intervention is also a belief; the links are only beliefs
    assert all(isinstance(b, tuple) and b[0] == "Sentence" for b in beliefs)


# ══════════════════════════════ scripts ════════════════════════════════════════════════════════════════════════════

def test_the_scripts_do_not_eval_a_data_file_and_write_no_environment_at_import():
    sys.path.insert(0, str(REPO / "scripts"))
    import pln_number_table as pt
    table = (REPO / "scripts" / "pln_number_table.py").read_text(encoding="utf-8")
    demo = (REPO / "scripts" / "pln_control_demo.py").read_text(encoding="utf-8")
    assert "eval(" not in table and "def arithmetic" in table
    assert pt.arithmetic("0.65**3*0.85*0.9**3") == pytest.approx(0.65 ** 3 * 0.85 * 0.9 ** 3, abs=0)
    for hostile in ("__import__('os').system('true')", "(1).__class__", "open('x')", "1 if 2 else 3", "[1]", "x"):
        with pytest.raises((ValueError, SyntaxError)):
            pt.arithmetic(hostile)
    # no os.environ write at module level of the demo (its main() sets it)
    top_level = [ln for ln in demo.splitlines() if ln.startswith("os.environ")]
    assert top_level == [], top_level


# ══════════════════════════════ the documents say what the code does ═══════════════════════════════════════════════

def test_api_md_states_what_the_pln_section_does():
    text = (REPO / "pln_chat" / "API.md").read_text(encoding="utf-8")
    section = text[text.index("## The pln chainer"):text.index("## Examples")]
    assert "may not be WRITTEN by a caller" in section and "written or read" not in section
    assert "the answers EQUAL the legacy ones" in section and "not a new estimate" in section        # plain statement, no over-claim
    assert "Do not take the first result" in section
    assert "422" in section and "refused" in section
    assert "n^3.7" in section and "exponential in the number of cycles" in section
    assert "are read only under `PLN_CHAINER=pln`" in section
    assert "may be read" not in section
    # the test count the page quotes is the real one for the files it names
    import re as _re
    m = _re.search(r"pytest tests/test_pln_chainer.py tests/test_pln_fixes.py tests/test_drugage_fold.py -q\s+# (\d+) tests", section)
    assert m, "the page no longer quotes the test count of the pln-lane files"


def test_the_module_docstrings_do_not_call_the_fold_the_forks_own_revision_and_say_plainly_what_the_answers_are():
    fold = (PLN_CHAT / "ontology" / "drugage_fold.py").read_text(encoding="utf-8")
    chainer = (PLN_CHAT / "core" / "pln_chainer.py").read_text(encoding="utf-8")
    assert "Python PORT of the fork's `Truth_Revision`" in fold
    assert "the answers EQUAL the legacy ones" in chainer and "a strength or a confidence is\nthe legacy number" in chainer


def test_the_literal_reading_of_the_fork_shows_a_strength_zero_link_with_the_wrong_difference_instead_of_failing():
    """Rapamycin has one null in 37 rows. Counted on both sides (the default) it is a harmful item of strength 0; the research reading
    `mapping='literal'` (q = p) cannot represent a link weaker than the fork's leak: s1 - s0 = -0.02 for a Pos link of strength 0.
    That is what the lane is for, so it reports the number; the sign cross-check stays on for the service's affine mapping."""
    rows = [{"from": "A", "to": "B", "sign": "Pos", "strength": 0.0, "confidence": 0.9, "terminal_only": False}]
    lit = sim_table(rows, pc.ChainSettings(mapping="literal"))
    (route,) = lit.routes["A"]
    assert route.sign == "Pos" and route.strength == pytest.approx(0.02) and route.level1 < route.level0
    aff = sim_table(rows, pc.ChainSettings())
    assert aff.routes["A"][0].strength == pytest.approx(0.0, abs=1e-15)
    big = [{**rows[0], "strength": 0.8}]
    with pytest.raises(pc.PlnChainerError, match="sign cross-check failed"):                  # ... and stays on for the affine mapping
        sim_table(big, pc.ChainSettings(), engine=SimEngine(flip_levels=True))
    assert sim_table(big, pc.ChainSettings(mapping="literal"), engine=SimEngine(flip_levels=True)).routes["A"]      # not for the research reading
