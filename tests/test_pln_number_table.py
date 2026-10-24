"""Invariants of the number-table script and the control-demonstration script (gate 6a/6c of the PeTTa migration).

`scripts/pln_number_table.py` asks the same questions of six lanes (legacy on hyperon, legacy on PeTTa, pln-R2, pln-R1,
pln-R0, pln-full); `scripts/pln_control_demo.py` shows the control of the library's queues on the shipped graph with a feedback edge
and on seeded SYNTHETIC layered networks. The scripts are measurements, not tests, and take minutes. What is pinned here is what
makes their output trustworthy, cheaply:

* the synthetic networks are what the demonstration says they are (seeded, deterministic, at least 500 links, with diamonds) and
  the ground truth they are scored against (a dynamic programme over the DAG) equals brute-force enumeration;
* the harness that drives `PLN.Derive` agrees with the adapter's (`pln_chainer`) on the same oracle: the same sentences, the same
  routes, the same selection counts, and the flags (complete / partial / unproven) mean what the decision says;
* the twelve orderings hold on the REAL hyperon output the contract study recorded (`pinned_numbers.json`) and stop holding when the
  output is tampered with;
* a few numbers the documents quote, on the real PeTTa (`-m petta`): the five lanes agree where they must and differ where they must.

The generated blocks of NUMBER_TABLE.md and CONTROL_DEMO.md are checked against the stored result files: a number in a table that
the data does not hold fails here. No test calls a model.

    pytest tests/test_pln_number_table.py -q                  # everything that can run here
    pytest tests/test_pln_number_table.py -q -m "not petta"   # without the real PeTTa
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
SCRIPTS = REPO / "scripts"
for p in (str(PLN_CHAT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

FORK_DESIGN = REPO / "docs" / "petta_migration" / "fork_design"
RESULTS = FORK_DESIGN / "c" / "results"

import config  # noqa: E402

# The scripts set PYTHON_DOTENV_DISABLED when imported (they must never read pln_chat/.env when run); importing them from a test must not leave
# that behind in the process, where it would change what a later module's `load_dotenv()` does.
_dotenv_before = os.environ.get("PYTHON_DOTENV_DISABLED")
import pln_control_demo as cd  # noqa: E402
import pln_number_table as nt  # noqa: E402
if _dotenv_before is None:
    os.environ.pop("PYTHON_DOTENV_DISABLED", None)
while str(SCRIPTS) in sys.path:                     # the scripts directory is not an import root for the other test modules
    sys.path.remove(str(SCRIPTS))
from core import pln_chainer as pc  # noqa: E402
from core.metta_runtime import PettaRuntime  # noqa: E402


def _real_petta_available() -> bool:
    try:
        return PettaRuntime().available() and PettaRuntime().run_program("", ["!(+ 1 2)"]) == [["3"]]
    except Exception:
        return False


REAL_PETTA = _real_petta_available()
requires_petta = pytest.mark.skipif(not REAL_PETTA, reason="PeTTa is not installed here (bash scripts/setup_petta.sh)")


def real_petta(fn):
    return pytest.mark.petta(requires_petta(fn))


@pytest.fixture
def isolated_config(monkeypatch):
    """The scripts set the runtime, the chainer and the chainer's settings on the process-wide config; give them back afterwards."""
    for name in ("PLN_RUNTIME", "PLN_CHAINER", "LOGS_DIR", "PLN_QUERY_TIMEOUT_SECONDS"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(pc, "settings_from_config", pc.settings_from_config)
    monkeypatch.setattr(pc, "_fold_items", pc._fold_items)


def _load_sim():
    spec = importlib.util.spec_from_file_location("fork_sim_c", FORK_DESIGN / "sim" / "fork_sim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fork_sim_c"] = mod
    spec.loader.exec_module(mod)
    return mod


SIM = _load_sim()


def _fmt(x) -> str:
    if isinstance(x, tuple):
        return "(" + " ".join(_fmt(i) for i in x) + ")"
    return repr(x) if isinstance(x, float) else str(x)


class SimEvaluator:
    """`Evaluate` backed by the validated simulator of the library: the harness and the adapter run on it unchanged."""

    def __init__(self):
        self.derive_calls = 0

    def __call__(self, expr: str) -> list:
        form = SIM.parse_sexprs("!" + expr)[0][1]
        head = form[0]
        if head == "PLN.Derive":
            self.derive_calls += 1
            tasks_f, beliefs_f, steps, maxsteps, tq, bq = form[1:7]
            sim = SIM.Sim(rules="restricted", maxsteps=maxsteps, task_queue_size=tq, belief_queue_size=bq, variant="petta_lib")
            t, b = sim.derive([SIM.sentence_from_metta(x) for x in tasks_f], [SIM.sentence_from_metta(x) for x in beliefs_f],
                              maxsteps - steps + 1, tq, bq)
            show = lambda ss: "(" + " ".join("(Sentence (%s (stv %r %r)) (%s))" % (_fmt(a), tv[0], tv[1], " ".join(str(i) for i in ev))
                                             for (a, tv, ev) in ss) + ")"
            return ["(%s %s)" % (show(t), show(b))]
        if head == "Truth_ModusPonens":
            s, c = SIM.Truth_ModusPonens((form[1][1], form[1][2]), (form[2][1], form[2][2]))
            return ["(stv %r %r)" % (s, c)]
        if head == "Truth_Revision":
            a, b = (form[1][1], form[1][2]), (form[2][1], form[2][2])
            if a[1] >= 1.0 or b[1] >= 1.0:
                return []
            s, c = SIM.Truth_Revision(a, b)
            return ["(stv %r %r)" % (s, c)]
        if head == "chain-discount":
            return ["0.9"]
        raise AssertionError(f"SimEvaluator cannot evaluate {expr[:80]}")


# ══════════════════════════════ the synthetic networks ══════════════════════════

def test_the_synthetic_generators_are_seeded_and_deterministic():
    for spec in (cd.SYN_S, cd.SYN_M, cd.SYN_L):
        a, b = cd.generate(spec), cd.generate(spec)
        assert [(l.src, l.dst, l.sign, l.strength, l.confidence, l.id) for l in a] == [(l.src, l.dst, l.sign, l.strength, l.confidence, l.id) for l in b]
    other = cd.generate(cd.SynSpec("x", seed=8, layers=cd.SYN_L.layers, width=cd.SYN_L.width, fanout=3, root_fanout=6))
    assert [(l.src, l.dst) for l in other] != [(l.src, l.dst) for l in cd.generate(cd.SYN_L)]


def test_the_large_synthetic_network_is_large_has_diamonds_and_is_labelled_synthetic():
    ctx = cd.synthetic_context(cd.SYN_L)
    net = cd.describe_network(ctx)
    assert net["name"].startswith("SYNTHETIC")
    assert net["links"] >= 500 and net["diamond_nodes"] >= 100 and net["max_indegree"] >= 4
    assert net["links"] == 560 and net["legacy_paths"] == 20719 and net["reachable_states"] == 173      # measured by the script
    assert 0.1 < net["neg_links"] / net["links"] < 0.4                                                  # mixed signs
    # every node is reachable from the root, the graph is acyclic, strengths and confidences are inside their ranges
    assert len(ctx["truth"]) == net["nodes"] - 1
    for l in ctx["links"]:
        assert 0.2 <= l.strength <= 0.9 and 0.5 <= l.confidence <= 0.9 and l.sign in ("Pos", "Neg") and not l.terminal
    assert len({l.id for l in ctx["links"]}) == len(ctx["links"])


def test_the_selections_of_a_complete_run_are_the_premise_the_routes_and_the_neg_routes():
    s = cd.generate(cd.SYN_S)
    paths = cd.enumerate_paths(s, cd.SYN_S.root)
    assert cd.complete_run_selections(s, cd.SYN_S.root) == 1 + len(paths) + sum(1 for p in paths if p[-1].sign == "Neg")
    m = cd.generate(cd.SYN_M)
    assert cd.complete_run_selections(m, cd.SYN_M.root) == 964          # the engine's count on SYNTHETIC-M (control_m.json)


def test_the_synthetic_networks_are_pinned_so_a_change_of_the_generator_is_noticed():
    pins = {cd.SYN_L.name: "9ea6b9ccc03a73ec", cd.SYN_M.name: "8e9f95565e013b6e"}
    for spec in (cd.SYN_L, cd.SYN_M):
        assert cd.describe_network(cd.synthetic_context(spec))["links_sha256"] == pins[spec.name]


def test_the_ground_truth_dynamic_programme_equals_brute_force_enumeration():
    for spec in (cd.SYN_S, cd.SynSpec("tiny", seed=1, layers=4, width=4, fanout=2, root_fanout=2)):
        links = cd.generate(spec)
        paths = cd.enumerate_paths(links, spec.root)
        assert cd.path_count(links, spec.root) == len(paths)
        dp, bf = cd.truth_dag(links, spec.root), cd.truth_enumerated(links, spec.root)
        assert set(dp) == set(bf)
        for v in dp:
            assert dp[v]["paths"] == bf[v]["paths"]
            assert dp[v]["best_c"] == pytest.approx(bf[v]["best_c"], rel=1e-12)
            assert dp[v]["best_rs"] == pytest.approx(bf[v]["best_rs"], rel=1e-12)


def test_the_dynamic_programme_refuses_a_graph_with_a_cycle():
    links = cd.shipped_links(True)
    with pytest.raises(ValueError, match="cycle"):
        cd.topological_order(links, "ChronicInflammation")


def test_recall_counts_reached_and_reached_with_the_true_best_route():
    truth = {"a": {"best_c": 0.9, "best_rs": 0.5}, "b": {"best_c": 0.8, "best_rs": 0.4}, "c": {"best_c": 0.1, "best_rs": 0.1}}
    found = {"a": {"best_c": 0.9, "best_rs": 0.5}, "b": {"best_c": 0.5, "best_rs": 0.4}}
    assert cd.recall(found, truth, "best_c", "best_c", 2) == (1.0, 0.5)          # b reached, but only through a poorer route
    assert cd.recall(found, truth, "best_c", "best_c", 3) == (2 / 3, 1 / 3)
    assert cd.recall({}, truth, "best_c", "best_c", 2) == (0.0, 0.0)
    assert cd.recall(found, {}, "best_c", "best_c", 2) == (1.0, 1.0)
    assert cd.top_k(truth, "best_rs", 2) == ["a", "b"]


def test_the_rank_score_is_strength_times_the_discounted_confidence():
    links = cd.generate(cd.SYN_S)
    path = cd.enumerate_paths(links, cd.SYN_S.root)[7]
    s, c, cdisc, rs = cd.route_values(path)
    assert s == pytest.approx(math.prod(l.strength for l in path)) and c == pytest.approx(math.prod(l.confidence for l in path))
    assert cdisc == pytest.approx(c * 0.9 ** (len(path) - 1)) and rs == pytest.approx(s * cdisc)


# ══════════════════════════════ the harness against the adapter, on the oracle ═══

def test_the_harness_prints_the_same_sentences_as_the_adapter():
    for links in (cd.generate(cd.SYN_S), cd.shipped_links(True)):
        mine, theirs = cd.Enc(links, 0.02), pc._Encoding(links, 0.02)
        assert mine.opaque == theirs.opaque
        assert mine.sentence == theirs.sentence
        src = links[0].src
        assert mine.premise(src, 1.0) == theirs.premise(src, 1.0) and mine.premise(src, 0.0) == theirs.premise(src, 0.0)


def test_the_flags_mean_what_the_decision_says():
    assert cd.flag_of(256, 256, 0, 40)["flag"] == "complete"
    assert cd.flag_of(256, 256, 3, 40)["flag"] == "partial"                          # tasks left
    assert cd.flag_of(256, 256, 0, 255)["flag"] == "partial"                         # the belief queue reached K - 1: something was dropped
    unproven = cd.flag_of(2, 200, 0, 40)
    assert unproven["flag"] == "partial" and unproven["naive_complete"] is True and unproven["proven"] is False
    assert cd.flag_of(300, 200, 0, 40)["flag"] == "complete"                         # tq >= bq is enough


def test_the_harness_equals_the_adapter_on_routes_and_selection_counts_on_the_oracle():
    links = cd.generate(cd.SYN_S)
    root = cd.SYN_S.root
    ev = SimEvaluator()
    leak = pc.probe_library(ev)
    table = pc.compute_table(ev, links, leak=leak, discount=0.9, sources=[root],
                             settings=pc.ChainSettings(count_selections=True, queue_size=256))
    sess = type("S", (), {"ev": ev, "leak": leak})()
    ctx = {"links": links, "truth": cd.truth_dag(links, root), "npaths": cd.path_count(links, root)}
    row = cd.run_config(sess, links, root, ctx["truth"], ctx["npaths"], 256, 256, [1024])[0]
    assert row["routes"] == len(table.routes[root]) == ctx["npaths"]
    assert row["selections"] == table.status[root].selections[0]                     # the adapter counts by chunks of one; the harness bisects
    assert row["selections"] == cd.complete_run_selections(links, root)             # 1 + routes + routes that end in a Neg link
    assert row["complete"] is True and row["truly_complete"] is True and row["false_chains"] == 0
    assert row["c_mismatch"] < 1e-12                                                 # the engine's confidence is the product of the links'
    assert row["recall_c@20"] == row["recall_rs@20"] == 1.0


@pytest.mark.parametrize("spec", [cd.SYN_S, cd.SynSpec("mini", seed=5, layers=4, width=8, fanout=3, root_fanout=3)])
def test_the_plain_python_model_of_the_loop_equals_the_oracle_snapshot_for_snapshot(spec):
    """`model_best_first("confidence")` is the library's loop as a priority queue of routes. If it equals the oracle's run at every
    checkpoint (routes found, states reached, the set of states), its other priorities (rank-score, fifo) mean something."""
    links, root = cd.generate(spec), spec.root
    ev = SimEvaluator()
    leak = pc.probe_library(ev)
    sess = type("S", (), {"ev": ev, "leak": leak})()
    truth, npaths = cd.truth_dag(links, root), cd.path_count(links, root)
    total = cd.complete_run_selections(links, root)
    caps = sorted({1, 2, 3, 5, 8, 13, 21, 34, 55, total - 1, total, total + 3} - {0})
    rows = cd.run_config(sess, links, root, truth, npaths, 4096, 4096, caps)
    enc, sub = cd.Enc(links, leak), pc.reachable_links(links, root)
    snaps = cd.run_snapshots(cd.Derive(ev), enc.tasks_text(root, 1.0), enc.beliefs_text(sub, root, 1.0), caps, 4096, 4096)
    reached = {sn.selections: sorted(cd.conclusions(enc, root, sn.beliefs).best()) for sn in snaps}
    assert cd.model_matches_engine(links, root, rows, reached)["equal"]
    assert rows[-1]["selections"] == total and rows[-1]["routes"] == npaths and not rows[-1]["tasks_left"]
    for pr in cd.PRIORITIES:                                 # every priority ends at the same complete table: only the ORDER differs
        done = cd.model_best_first(links, root, pr, [total + 5])[total + 5]
        assert done["routes"] == npaths and done["states"] == len(truth) and done["tasks_left"] == 0


def test_the_model_priorities_order_the_search_differently_and_the_rank_score_priority_wins_on_its_own_measure():
    links, root = cd.generate(cd.SYN_M), cd.SYN_M.root
    truth = cd.truth_dag(links, root)
    n = 60
    got = {pr: cd.model_row(cd.model_best_first(links, root, pr, [n])[n], truth, (10,)) for pr in cd.PRIORITIES}
    assert got["rank-score"]["exact_rs@10"] >= got["confidence"]["exact_rs@10"] >= got["fifo"]["exact_rs@10"] - 0.2
    assert got["confidence"]["exact_c@10"] >= got["fifo"]["exact_c@10"]


def test_exhaustion_is_located_exactly_whatever_the_chunking():
    links = cd.generate(cd.SYN_S)
    root = cd.SYN_S.root
    ev = SimEvaluator()
    leak = pc.probe_library(ev)
    enc = cd.Enc(links, leak)
    sub = pc.reachable_links(links, root)
    truth_n = pc.count_selections(ev, links, root, leak, queue=256)
    for caps in ([4096], [3, 4096], [1, 2, 5, 9, 4096], [truth_n - 1, truth_n, truth_n + 1]):
        snaps = cd.run_snapshots(cd.Derive(ev), enc.tasks_text(root, 1.0), enc.beliefs_text(sub, root, 1.0), caps, 256, 256)
        last = snaps[-1]
        assert last.selections == truth_n and not last.tasks, caps
        for s in snaps:
            assert s.selections == min(s.cap, truth_n), (caps, s.cap)


def test_a_partial_run_is_flagged_and_loses_routes_a_complete_run_keeps():
    links = cd.generate(cd.SYN_S)
    root = cd.SYN_S.root
    ev = SimEvaluator()
    leak = pc.probe_library(ev)
    sess = type("S", (), {"ev": ev, "leak": leak})()
    truth = cd.truth_dag(links, root)
    npaths = cd.path_count(links, root)
    full = cd.run_config(sess, links, root, truth, npaths, 1024, 1024, [4096])[0]
    cut = cd.run_config(sess, links, root, truth, npaths, 24, 24, [96])[0]
    assert full["complete"] and full["routes"] == npaths
    assert not cut["complete"] and cut["routes"] < npaths
    # tq < bq: the naive test calls it complete although routes were lost, the adapter's test does not
    blind = cd.run_config(sess, links, root, truth, npaths, 2, 1024, [4096])[0]
    assert blind["naive_complete"] is True and blind["complete"] is False and blind["routes"] < npaths and not blind["truly_complete"]


def test_a_conclusion_no_walk_licenses_is_counted_as_a_false_chain():
    links = cd.generate(cd.SYN_S)
    enc = cd.Enc(links, 0.02)
    root = cd.SYN_S.root
    target = next(l for l in links if l.src == root)
    # a belief that claims (State X) from a stamp that is not a walk from the root to X
    other = next(l for l in links if l.src != root and l.dst != target.dst)
    forged = [(("State", enc.opaque[other.dst]), "0.5", "0.5", ("0", str(target.id)))]
    con = cd.conclusions(enc, root, forged)
    assert con.false == 1 and not con.routes
    honest = [(("State", enc.opaque[target.dst]), "0.5", str(target.confidence), ("0", str(target.id)))]
    con = cd.conclusions(enc, root, honest)
    assert con.false == 0 and len(con.routes) == 1


def test_the_feedback_graph_is_the_shipped_graph_plus_one_edge():
    plain, fb = cd.shipped_links(False), cd.shipped_links(True)
    assert len(plain) == 28 and len(fb) == 29
    assert (fb[-1].src, fb[-1].dst, fb[-1].sign, fb[-1].strength, fb[-1].confidence) == ("ChronicInflammation", "CellularSenescence", "Pos", 0.5, 0.65)
    assert len(cd.node_simple_closure(plain)) == 100 and len(cd.node_simple_closure(fb)) == 109


# ══════════════════════════════ the twelve orderings ════════════════════════════

def _recorded_composite() -> dict:
    entries = json.loads((FORK_DESIGN / "contract" / "pinned_numbers.json").read_text(encoding="utf-8"))["entries"]
    return {e["tag"]: [atom for group in e["results"] for atom in group] for e in entries}


def test_the_twelve_orderings_hold_on_the_output_hyperon_recorded():
    got = nt.evaluate_orderings(_recorded_composite())
    assert [o["id"] for o in got] == list(range(1, 13))
    failing = [(o["id"], o["values"]) for o in got if not o["holds"]]
    assert failing == []
    d = nt.diagnose_values(_recorded_composite())
    assert d["population"]["s_CS"] == pytest.approx(0.595) and abs(d["Patient001"]["c_CS"] - 0.6253054502923161) < 1e-12


def test_a_tampered_output_makes_the_matching_ordering_fail_and_no_other():
    t = _recorded_composite()
    base = nt.evaluate_orderings(t)
    # swap Berberine and Spermidine in Patient001's personalised order (what native confidence does)
    atoms = t["rank_for_Patient001_5pool"][0]
    names = re.findall(r"\(scored (\S+) ", atoms)
    assert names[-2:] == ["Berberine", "Spermidine"]
    entries = re.findall(r"\(scored \S+ .*?\(stv [\d.e-]+ [\d.e-]+\)\)\)", atoms)
    assert len(entries) == 5
    t2 = dict(t)
    t2["rank_for_Patient001_5pool"] = ["(" + " ".join(entries[:3] + [entries[4], entries[3]]) + ")"]
    got = nt.evaluate_orderings(t2)
    assert [o["id"] for o in got if not o["holds"]] == [3]
    assert all(o["holds"] for o, b in zip(got, base) if o["id"] != 3 and b["holds"])
    # a missing output is a failure with a reason, never an exception
    t3 = {k: v for k, v in t.items() if k != "supplements_Patient001"}
    bad = [o for o in nt.evaluate_orderings(t3) if not o["holds"]]
    assert [o["id"] for o in bad] == [10] and "not evaluable" in bad[0]["values"]


def test_the_closure_statistics_see_a_missing_pair_a_changed_sign_and_a_shrunk_strength():
    ref = nt.recorded_closure()
    pairs = [{"chain": list(k), "results": [[v[0], v[1], v[2]]]} for k, v in ref.items()]
    s = nt.closure_stats(pairs, ref)
    assert (s["pairs"], s["missing"], s["extra"], s["sign_differs"], s["max_ds"], s["max_dc"]) == (100, 0, 0, 0, 0.0, 0.0)
    pairs2 = [dict(p) for p in pairs][:-1]
    pairs2[0] = {"chain": pairs2[0]["chain"], "results": [["Pos" if pairs2[0]["results"][0][0] == "Neg" else "Neg", pairs2[0]["results"][0][1] * 0.9, pairs2[0]["results"][0][2]]]}
    pairs2.append({"chain": ["X", "Y"], "results": [["Pos", 0.5, 0.5]]})
    s2 = nt.closure_stats(pairs2, ref)
    assert s2["missing"] == 1 and s2["extra"] == 1 and s2["sign_differs"] == 1 and s2["max_ds"] > 0 and s2["ratio_s"][0] == pytest.approx(0.9)


def test_the_pins_are_the_arithmetic_of_the_tests():
    pins = {c["id"]: nt._pin(c) for c in nt.CHAINS}
    assert pins["D+Q->CHD"]["c"] == pytest.approx(0.17017138125) and pins["D+Q->CHD"]["s"] == 0.2499
    assert pins["CS->CHD"]["c"] == pytest.approx(0.29089125) and pins["CS->CHD"]["s"] == pytest.approx(0.357)
    assert pins["Spermidine->CHD"]["c"] == pytest.approx(0.0765771, rel=1e-6)
    assert pins["D+Q->AllCauseMortality"] is None
    assert sum(1 for c in nt.CHAINS if c["smoke"]) == 5
    assert set(nt.LANE_ORDER) == {"legacy-hyperon", "legacy-petta", "R2", "R1", "R0", "full"}


def test_the_private_names_of_the_adapter_that_the_scripts_use_still_exist():
    for name in ("_Encoding", "_fold_items", "settings_from_config", "estimate_beliefs", "FULL_LANE_BUDGET", "ChainSettings", "compute_table",
                 "count_selections", "base_rate_text", "library_text", "reachable_links", "assign_ids", "probe_library", "parse_sexpr", "fnum"):
        assert hasattr(pc, name), name
    assert pc.FULL_LANE_BUDGET == (150, 100, 400)


# ══════════════════════════════ the documents against the data ══════════════════

def _blocks(text: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r"<!-- BEGIN GENERATED (\w+) -->\n(.*?)\n<!-- END GENERATED \1 -->", text, re.S)}


def test_every_generated_block_of_the_number_table_is_what_the_stored_data_renders(tmp_path):
    doc = FORK_DESIGN / "NUMBER_TABLE.md"
    if not doc.exists() or not (RESULTS / "lane_R2.json").exists():
        pytest.skip("the number table has not been generated here")
    copy = tmp_path / "NUMBER_TABLE.md"
    copy.write_text(doc.read_text(encoding="utf-8"), encoding="utf-8")
    nt.render(RESULTS, copy)
    assert copy.read_text(encoding="utf-8") == doc.read_text(encoding="utf-8"), "NUMBER_TABLE.md is out of date with c/results/lane_*.json: run `scripts/pln_number_table.py render`"
    blocks = _blocks(doc.read_text(encoding="utf-8"))
    assert set(nt.BLOCKS) <= set(blocks)


def test_every_number_of_the_chain_table_carries_its_cost_or_says_the_lane_has_none():
    doc = FORK_DESIGN / "NUMBER_TABLE.md"
    if not doc.exists():
        pytest.skip("the number table has not been generated here")
    chains = _blocks(doc.read_text(encoding="utf-8")).get("chains", "")
    assert "python scripts/pln_number_table.py lane --lane" in chains.splitlines()[0]          # every block says which command measured it
    rows = [r for r in chains.splitlines() if r.startswith("|")][2:]
    assert len(rows) == len(nt.CHAINS)
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")][2:]
        assert len(cells) == 6
        for cell, lane in zip(cells, nt.LANE_ORDER):
            if cell in ("empty", "not measured"):
                continue
            if lane.startswith("legacy"):
                assert "Pos" in cell or "Neg" in cell
            else:
                assert re.search(r"st (\d+|\?)( \(budget\))? / tq \d+ / bq \d+", cell), (lane, cell)


def test_every_generated_block_of_the_control_demo_is_what_the_stored_data_renders(tmp_path):
    doc = FORK_DESIGN / "CONTROL_DEMO.md"
    if not doc.exists() or not (RESULTS / "control_l.json").exists():
        pytest.skip("the control demonstration has not been generated here")
    copy = tmp_path / "CONTROL_DEMO.md"
    copy.write_text(doc.read_text(encoding="utf-8"), encoding="utf-8")
    cd.render(RESULTS, copy)
    assert copy.read_text(encoding="utf-8") == doc.read_text(encoding="utf-8"), "CONTROL_DEMO.md is out of date with c/results/control_*.json"


def test_the_control_demo_data_says_what_the_document_says():
    path = RESULTS / "control_l.json"
    if not path.exists():
        pytest.skip("the control demonstration has not been generated here")
    d = json.loads(path.read_text(encoding="utf-8"))
    net = d["network"]
    assert net["name"].startswith("SYNTHETIC") and net["links"] >= 500 and net["diamond_nodes"] > 0
    for key in ("step_sweep", "joint_sweep", "tq_sweep", "bq_sweep"):
        assert d[key], key
        for r in d[key]:
            assert r["false_chains"] == 0 and r["c_mismatch"] < 1e-12            # no false chain at scale, engine confidence = product of links'
            assert 0.0 <= r["recall_c@20"] <= 1.0 and 0.0 <= r["recall_rs@20"] <= 1.0 and r["exact_c@20"] <= r["recall_c@20"] + 1e-12
            assert r["flag"] in ("complete", "partial") and (r["flag"] == "complete") == r["complete"]
            if r["complete"]:
                assert r["truly_complete"]                                       # a run flagged complete never lost a route
    assert not any(r["complete"] for r in d["step_sweep"])                        # 20,719 legacy paths: nothing is complete on SYNTHETIC-L
    assert d["model_check"]["equal"] and d["model_check"]["snapshots"] == len(d["step_sweep"])   # the model of the loop equals the engine here
    m = json.loads((RESULTS / "control_m.json").read_text(encoding="utf-8"))
    done = [r for r in m["contract"] if r["complete"]]
    assert done and all(r["selections"] == m["network"]["complete_run_selections"] and r["routes"] == r["legacy_paths"] for r in done)
    assert all((not r["complete"]) or r["truly_complete"] for r in m["contract"])          # the adapter's flag never over-claims
    assert any(r["naive_complete"] and not r["truly_complete"] for r in m["blind"])        # the naive one does, with tq < bq
    assert all(not r["complete"] for r in m["blind"])
    a = json.loads((RESULTS / "control_a.json").read_text(encoding="utf-8"))
    g = a["experiment"]["graphs"]["shipped+feedback"]
    assert (g["pairs"], g["reference_pairs"], g["missing"], g["extra"], g["false_chains"]) == (109, 109, 0, 0, 0) and g["all_complete"]
    assert a["production"]["legacy"]["status"] == "error" and a["production"]["pln"]["pairs"] == 109 and a["production"]["pln"]["complete"]


# ══════════════════════════════ on the real PeTTa ═══════════════════════════════

@real_petta
def test_the_engine_and_the_oracle_agree_state_for_state_on_the_small_network(isolated_config):
    """The harness on the real library and on the simulator, every source of the small network, three queue settings: identical states."""
    links = cd.generate(cd.SYN_S)

    def fn(sess):
        enc = cd.Enc(links, sess.leak)
        out = []
        for source in sorted({l.src for l in links}):
            sub = pc.reachable_links(links, source)
            for tq, bq, cap in ((256, 256, 1024), (3, 200, 200), (40, 40, 160)):
                d = cd.Derive(sess.ev)
                snaps = cd.run_snapshots(d, enc.tasks_text(source, 1.0), enc.beliefs_text(sub, source, 1.0), [cap], tq, bq)
                s = snaps[0]
                out.append([source, tq, bq, s.selections, [cd.sent_text(x) for x in s.tasks], [cd.sent_text(x) for x in s.beliefs]])
        return {"runs": out}

    real = cd.in_child(fn, deadline_s=300)["runs"]
    ev = SimEvaluator()
    leak = pc.probe_library(ev)
    sim = type("S", (), {"ev": ev, "leak": leak})()
    mine = fn(sim)["runs"]
    assert len(real) == len(mine) == len({l.src for l in links}) * 3
    for r, m in zip(real, mine):
        assert r == m, (r[:4], m[:4])


@real_petta
def test_the_feedback_edge_on_the_engine_terminates_with_109_pairs_and_the_control_never_binds(isolated_config):
    def fn(sess):
        links = cd.shipped_links(True)
        enc = cd.Enc(links, sess.leak)
        ref = cd.node_simple_closure(links)
        pairs, worst = set(), 0
        for source in sorted({l.src for l in links}):
            sub = pc.reachable_links(links, source)
            row = cd.run_snapshots(cd.Derive(sess.ev), enc.tasks_text(source, 1.0), enc.beliefs_text(sub, source, 1.0), [1024], 256, 256)[0]
            con = cd.conclusions(enc, source, row.beliefs)
            pairs |= {tuple(r["nodes"]) for r in con.routes.values()}
            worst = max(worst, row.selections, 0)
            assert cd.flag_of(256, 256, len(row.tasks), len(row.beliefs))["complete"] and con.false == 0
        return {"pairs": len(pairs), "equal": pairs == set(ref), "max_selections": worst}

    out = cd.in_child(fn, deadline_s=300)
    assert out == {"pairs": 109, "equal": True, "max_selections": 16}


@real_petta
def test_the_lanes_agree_where_they_must_and_differ_where_they_must_on_the_pinned_chain(isolated_config):
    """D+Q -> CHD through the production path in four lanes (legacy on PeTTa, R2, R1, R0): the numbers the documents quote."""
    got = {}
    for lane_id in ("legacy-petta", "R2", "R1", "R0"):
        lane = nt.Lane(lane_id)
        res = lane.run(lane.stack("patient"), None, [f"!(infer &self {nt.DQ} {nt.CHD})"])
        assert res["status"] == "ok", res["error"]
        got[lane_id] = (nt.parse_signed(res["groups"][0][0]), nt.cost_of(res["chain"], nt.DQ))
    (sg, s, c), cost = got["legacy-petta"]
    assert (sg, s) == ("Neg", pytest.approx(0.2499, abs=1e-12)) and c == pytest.approx(0.65 ** 3 * 0.85 * 0.9 ** 3, abs=1e-12) and cost is None
    for lane_id in ("R2", "R1"):
        sgn, ss, cc = got[lane_id][0]
        assert sgn == "Neg" and ss == pytest.approx(0.2499, abs=1e-12)                       # affine storage: strength equals legacy
    assert got["R2"][0][2] == pytest.approx(0.17017138125, abs=1e-12)                         # the 0.9 shim: confidence equals legacy
    assert got["R1"][0][2] == pytest.approx(0.65 ** 3 * 0.85, abs=1e-12)                      # native c1 c2: no discount
    assert got["R0"][0][1] == pytest.approx(0.22259936, abs=1e-8) and got["R0"][0][2] == pytest.approx(0.65 ** 3 * 0.85, abs=1e-12)    # q = p
    for lane_id in ("R2", "R1", "R0"):
        cost = got[lane_id][1]
        assert (cost["st"], cost["tq"], cost["bq"], cost["complete"]) == (11, 256, 256, True)    # D+Q: 11 selections, the first rung
