#!/usr/bin/env python
"""The control demonstration (gate 6c of the PeTTa migration): what patham9's priority queues do, and what they do not.

Why this exists
---------------
Under `PLN_CHAINER=pln` every `infer`/`explain` answer is a conclusion of patham9's `PLN.Derive` loop: a task queue
ranked by CONFIDENCE ONLY, a belief queue, both cut to a size, and a step budget. On the shipped 30-node graph that control
never binds (one derivation per pair, at most 12-16 selections, a few dozen beliefs against queues of 256), so the answers
equal the legacy chainer's and nothing of the control is visible. This script shows the control where it can be seen,
with a seeded SYNTHETIC layered network (not a real network: nobody has curated a large one yet) and with the shipped
network plus a feedback edge (a real network and a real edge, but the edge is a probe, not curated).

Parts (each is one command, each writes a JSON file; `render` turns the JSON into the tables of CONTROL_DEMO.md)
---------------------------------------------------------------------------------------------------------------
  a      the shipped patient graph + the feedback edge ChronicInflammation -> CellularSenescence: termination, 109 pairs,
         the complete flag, peak queue occupancy, and the smallest queue size at which the control starts to cut.
  m      SYNTHETIC-M (177 links, 754 legacy paths, diamonds): the completeness test measured against the ground truth:
         tq = bq (valid) and tq < bq (blind: decision.md correction 1).
  l      SYNTHETIC-L (560 links, 20,719 legacy paths, diamonds): selections used against paths, states reached, recall of
         the highest-confidence states AND of the highest rank-score (s x c) states, complete/partial flags, a step sweep,
         a joint queue sweep, a task-queue sweep (the beam) and a belief-queue sweep (the memory); wall time.
  seeds  ten more instances of the SYNTHETIC-L recipe: is the early recall a property of the library's order or of one lucky network?
  legacy the legacy chainer on SYNTHETIC-L (what enumerating every path costs) and contract mode on it (the deadline).

Part l and part seeds also run THE LOOP AS A MODEL (`model_best_first`: PLN.Derive on a DAG with no queue limit, as a priority queue of
routes, in plain Python). It is validated against the engine (same routes and same set of states at every snapshot) and then run with the
priorities the library does not offer: the application's rank-score (strength x confidence x 0.9^(hops-1)) and first-in-first-out. That is
the answer to "what would the recall by rank-score be if the application supplied its own priority?", a model, not a measurement of the library.

    python scripts/pln_control_demo.py run   --part all          # about 5 minutes, writes the JSON files
    python scripts/pln_control_demo.py render                    # JSON -> the generated blocks of CONTROL_DEMO.md
    python scripts/pln_control_demo.py run   --part a            # one part

How a run is made. `PettaRuntime().run_program("", ..., after_load=step)`: the experiment runs inside the forked PeTTa child
(the production sandbox, the production adapter hook), loads the vendored library, and drives `PLN.Derive` with its
6-argument form so the state can be resumed and inspected at chosen step counts. The loop is the library's, unmodified; the
only code of mine is what builds the sentences (the decision's encoding), reads the beliefs, and scores them against an exact
ground truth computed in plain Python (a dynamic programme over the DAG; nothing from the engine).

Definitions used in every table
-------------------------------
  selection    one task taken from the task queue and expanded against every belief (one `PLN.Derive` step).
  legacy paths the number of derivations the legacy chainer returns from the source: every path, any length.
  state        a node reached by some conclusion; a route is one path (one stamp); a state can have many routes (diamonds).
  by confidence    the library's own priority: the best route confidence c = product of the link confidences (no discount).
  by rank-score    the application's: strength x confidence, strength = product of the link strengths, confidence
                   = c x 0.9^(hops-1); the best route per state. Both are magnitudes (sign ignored).
  recall@k     of the k states with the best TRUE value, the share the run has reached. `exact@k`: reached with its true
               best route (a state first reached through a poorer route counts for recall but not for exact).
  flag         the library-side test the adapter uses: complete iff tq >= bq and the task queue ended empty and the belief
               queue never reached bq - 1 (nothing dropped). `naive` drops the tq >= bq condition.

Nothing here is a service path: queue sizes and step caps are arguments of this script, never of a request.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

FORK_DESIGN = REPO / "docs" / "petta_migration" / "fork_design"
CONTRACT = FORK_DESIGN / "contract"
DEFAULT_OUT = FORK_DESIGN / "c" / "results"
DOC = FORK_DESIGN / "CONTROL_DEMO.md"
FEEDBACK = ("ChronicInflammation", "CellularSenescence", "Pos", 0.5, 0.65)      # the probe edge of the decision (section 5)
DISCOUNT = 0.9

from core import pln_chainer as pc                                # noqa: E402  (after the path set-up)

Sent = tuple      # (term tree, strength text, confidence text, stamp tuple of id texts): exact texts, never re-printed floats


# ══════════════════════════════ the SYNTHETIC networks ═════════════════════════

@dataclass(frozen=True)
class SynSpec:
    """A seeded layered network. NOT a real network: it exists to make the control bind."""
    name: str
    seed: int
    layers: int
    width: int
    fanout: int
    root_fanout: int
    neg_fraction: float = 0.25
    root: str = "SynRoot"

    def describe(self) -> str:
        return (f"{self.name}: seed {self.seed}, 1 root -> {self.root_fanout} nodes, then {self.layers} layers of {self.width} nodes, "
                f"each node linked to {self.fanout} random nodes of the next layer (and every node given a parent), "
                f"{int(self.neg_fraction * 100)} percent of links Neg, strengths U(0.2,0.9), confidences U(0.5,0.9)")


SYN_L = SynSpec("SYNTHETIC-L", seed=7, layers=8, width=26, fanout=3, root_fanout=6)
SYN_M = SynSpec("SYNTHETIC-M", seed=7, layers=5, width=14, fanout=3, root_fanout=6)
SYN_S = SynSpec("SYNTHETIC-S", seed=3, layers=3, width=5, fanout=2, root_fanout=3)


def generate(spec: SynSpec) -> list[pc.Link]:
    """The links of `spec`, evidence ids assigned (1..n by content). Deterministic for a seed (a `random.Random` of its own)."""
    rng = random.Random(spec.seed)
    layer = [[f"L{i}_{j}" for j in range(spec.width)] for i in range(spec.layers)]
    edges: list[tuple[str, str]] = [(spec.root, t) for t in rng.sample(layer[0], spec.root_fanout)]
    for i in range(spec.layers - 1):
        indeg = {n: 0 for n in layer[i + 1]}
        for u in layer[i]:
            for v in rng.sample(layer[i + 1], spec.fanout):
                edges.append((u, v))
                indeg[v] += 1
        for v, d in indeg.items():                 # every node of a layer has a parent: all of them are reachable from the root
            if d == 0:
                edges.append((rng.choice(layer[i]), v))
    links = []
    for a, b in edges:
        sign = "Neg" if rng.random() < spec.neg_fraction else "Pos"
        links.append(pc.Link(a, b, sign, round(rng.uniform(0.2, 0.9), 3), round(rng.uniform(0.5, 0.9), 3)))
    pc.assign_ids(links)
    return links


# ══════════════════════════════ the ground truth (plain Python) ════════════════

def _out_by(links: Sequence[pc.Link]) -> dict[str, list[pc.Link]]:
    out: dict[str, list[pc.Link]] = {}
    for l in links:
        out.setdefault(l.src, []).append(l)
    return out


def topological_order(links: Sequence[pc.Link], root: str) -> list[str]:
    """Nodes reachable from `root` in topological order; raises on a cycle (the DP below needs a DAG)."""
    out = _out_by(links)
    state: dict[str, int] = {}
    order: list[str] = []

    def visit(u: str) -> None:
        state[u] = 1
        for l in out.get(u, ()):
            if state.get(l.dst) == 1:
                raise ValueError(f"cycle through {u} -> {l.dst}: the dynamic programme needs a DAG")
            if l.dst not in state:
                visit(l.dst)
        state[u] = 2
        order.append(u)

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    visit(root)
    return order[::-1]


def path_count(links: Sequence[pc.Link], root: str) -> int:
    """The legacy derivations from `root`: every path of length >= 1 (a record ends a path and is a link like the others)."""
    out = _out_by(links)
    memo: dict[str, int] = {}

    def n(u: str) -> int:
        if u not in memo:
            memo[u] = sum(1 + (0 if l.terminal else n(l.dst)) for l in out.get(u, ()))
        return memo[u]

    return n(root)


def complete_run_selections(links: Sequence[pc.Link], root: str) -> int:
    """How many selections a COMPLETE run takes on a DAG: the premise, one for every route, and one more for every route that ends in a Neg
    link (its conclusion is `(Not (State B))`, which the Not-elimination rule must be applied to by a selection of its own). Verified on
    the engine for SYNTHETIC-M (964) and SYNTHETIC-S; for SYNTHETIC-L it is arithmetic, that run was not made."""
    order = topological_order(links, root)
    by_dst: dict[str, list[pc.Link]] = {}
    for l in links:
        by_dst.setdefault(l.dst, []).append(l)
    paths_to = {root: 1}
    for v in order:
        if v != root:
            paths_to[v] = sum(paths_to.get(l.src, 0) for l in by_dst.get(v, ()))
    total = sum(paths_to[v] for v in order if v != root)
    ending_neg = sum(paths_to.get(l.src, 0) for l in links if l.sign == "Neg")
    return 1 + total + ending_neg


def enumerate_paths(links: Sequence[pc.Link], root: str) -> list[tuple[pc.Link, ...]]:
    """Every node-simple path from `root`, brute force (tests and the small graphs; exponential on purpose)."""
    out = _out_by(links)
    found: list[tuple[pc.Link, ...]] = []

    def go(node: str, visited: frozenset, acc: tuple) -> None:
        for l in out.get(node, ()):
            if l.dst in visited:
                continue
            found.append(acc + (l,))
            if not l.terminal:
                go(l.dst, visited | {l.dst}, acc + (l,))

    go(root, frozenset([root]), ())
    return found


def route_values(path: Sequence[pc.Link]) -> tuple[float, float, float, float]:
    """(strength, native confidence, discounted confidence, rank-score) of one path, composed as the legacy chainer does."""
    s = math.prod(l.strength for l in path)
    c = math.prod(l.confidence for l in path)
    cd = c * DISCOUNT ** (len(path) - 1)
    return s, c, cd, s * cd


def truth_dag(links: Sequence[pc.Link], root: str) -> dict[str, dict]:
    """Per node reachable from `root` on a DAG: the number of paths, the best route confidence and the best rank-score.

    best_c[v]  = max over paths of the product of confidences;  best_rs[v] = max over paths of s x c x 0.9^(hops-1)
               = max over paths of prod(p c 0.9) / 0.9, a product of per-link terms, hence a max-product DP in topological order.
    """
    order = topological_order(links, root)
    by_dst: dict[str, list[pc.Link]] = {}
    for l in links:
        by_dst.setdefault(l.dst, []).append(l)
    npaths = {root: 1}
    best_c = {root: 1.0}
    best_t = {root: 1.0}                 # prod(p c 0.9) along the best path
    best_h = {root: 0}
    out: dict[str, dict] = {}
    for v in order:
        if v == root:
            continue
        ps = [l for l in by_dst.get(v, ()) if l.src in npaths]
        npaths[v] = sum(npaths[l.src] for l in ps)
        best_c[v] = max(best_c[l.src] * l.confidence for l in ps)
        best_t[v] = max(best_t[l.src] * l.strength * l.confidence * DISCOUNT for l in ps)
        out[v] = {"paths": npaths[v], "best_c": best_c[v], "best_rs": best_t[v] / DISCOUNT}
    return out


def truth_enumerated(links: Sequence[pc.Link], root: str) -> dict[str, dict]:
    """The same table by brute force over node-simple paths (any graph; small ones only)."""
    out: dict[str, dict] = {}
    for path in enumerate_paths(links, root):
        v = path[-1].dst
        s, c, cd, rs = route_values(path)
        e = out.setdefault(v, {"paths": 0, "best_c": 0.0, "best_rs": 0.0})
        e["paths"] += 1
        e["best_c"] = max(e["best_c"], c)
        e["best_rs"] = max(e["best_rs"], rs)
    return out


def top_k(truth: dict[str, dict], key: str, k: int) -> list[str]:
    return [v for v, _ in sorted(truth.items(), key=lambda kv: (-kv[1][key], kv[0]))[:k]]


# ══════════════════════════════ the encoding and the engine harness ════════════

class Enc:
    """The decision's encoding of a graph, built independently of `pln_chainer._Encoding` (a test asserts they print the same
    sentences): node X is `(State nK)`, a link is an Implication with strength q = e + (1 - e) p, a Neg link concludes `(Not ..)`, a
    record concludes `(End ..)`, the intervention is `(State A)` at level L, confidence 1.0, stamp (0)."""

    def __init__(self, links: Sequence[pc.Link], leak: float):
        names = sorted({l.src for l in links} | {l.dst for l in links})
        self.opaque = {key: f"n{i}" for i, key in enumerate(names)}
        self.key_of = {v: k for k, v in self.opaque.items()}
        self.leak = leak
        self.link_of = {l.id: l for l in links}
        self.sentence: dict[int, str] = {}
        for l in links:
            tgt = f"({'End' if l.terminal else 'State'} {self.opaque[l.dst]})"
            if l.sign == "Neg":
                tgt = f"(Not {tgt})"
            q = leak + (1.0 - leak) * l.strength
            self.sentence[l.id] = (f"(Sentence ((Implication (State {self.opaque[l.src]}) {tgt}) "
                                   f"(stv {pc.fnum(q)} {pc.fnum(l.confidence)})) ({l.id}))")

    def premise(self, source: str, level: float) -> str:
        return f"(Sentence ((State {self.opaque[source]}) (stv {pc.fnum(level)} 1.0)) (0))"

    def beliefs_text(self, sub: Sequence[pc.Link], source: str, level: float) -> str:
        return "(" + " ".join([self.sentence[l.id] for l in sub] + [self.premise(source, level)]) + ")"

    def tasks_text(self, source: str, level: float) -> str:
        return f"({self.premise(source, level)})"


def parse_sentence(tree: Any) -> Sent:
    """('Sentence', (term, ('stv', s, c)), ids) -> (term, s, c, ids), texts kept."""
    try:
        assert tree[0] == "Sentence"
        (term, tv), ids = tree[1], tree[2]
        assert tv[0] == "stv"
        return term, tv[1], tv[2], tuple(ids)
    except (AssertionError, IndexError, TypeError, ValueError):
        raise pc.PlnChainerError(f"a sentence of PLN.Derive's result is malformed: {pc.serialize(tree)[:120]}") from None


def sent_text(x: Sent) -> str:
    term, s, c, ids = x
    return f"(Sentence ({pc.serialize(term)} (stv {s} {c})) ({' '.join(ids)}))"


def list_text(xs: Sequence[Sent]) -> str:
    return "(" + " ".join(sent_text(x) for x in xs) + ")"


class Derive:
    """`PLN.Derive` through an evaluator, with exactly-one-result asserted on every call and the time kept."""

    def __init__(self, ev: Callable[[str], list]):
        self.ev = ev
        self.engine_s = 0.0
        self.parse_s = 0.0
        self.calls = 0

    def __call__(self, tasks: str, beliefs: str, steps: int, maxsteps: int, tq: int, bq: int) -> tuple[list[Sent], list[Sent]]:
        t0 = time.perf_counter()
        out = self.ev(f"(PLN.Derive {tasks} {beliefs} {steps} {maxsteps} {tq} {bq})")
        t1 = time.perf_counter()
        if len(out) != 1:
            raise pc.PlnChainerError(f"PLN.Derive returned {len(out)} results; exactly one is a table")
        tree = pc.parse_sexpr(out[0])
        if not (isinstance(tree, tuple) and len(tree) == 2):
            raise pc.PlnChainerError(f"PLN.Derive did not return (Tasks Beliefs): {out[0][:120]}")
        res = [parse_sentence(x) for x in tree[0]], [parse_sentence(x) for x in tree[1]]
        self.engine_s += t1 - t0
        self.parse_s += time.perf_counter() - t1
        self.calls += 1
        return res


@dataclass
class Snapshot:
    cap: int                    #: the step count this snapshot was asked at
    selections: int             #: tasks actually expanded so far (exact)
    tasks: list
    beliefs: list
    engine_s: float             #: cumulative engine time to reach it (excludes the snapshots' own parsing)
    peak_tasks: int
    peak_beliefs: int


def locate_exhaustion(derive: Derive, tasks_t: str, beliefs_t: str, done: int, cap: int, tq: int, bq: int) -> int:
    """The step at which the task queue emptied inside the chunk (done, cap], found from the chunk's start state: growing sub-chunks
    (1, 2, 4, ...) until one ends with no task, then a bisection inside that sub-chunk only. Costs a small multiple of the chunk, never of
    the whole run, and its time is not part of any reported wall time."""
    step = 1
    while done < cap:
        nxt = min(cap, done + step)
        T, B = derive(tasks_t, beliefs_t, done + 1, nxt, tq, bq)
        if not T:
            lo, hi = done + 1, nxt
            while lo < hi:
                mid = (lo + hi) // 2
                Tm, _ = derive(tasks_t, beliefs_t, done + 1, mid, tq, bq)
                if Tm:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        tasks_t, beliefs_t, done, step = list_text(T), list_text(B), nxt, step * 2
    return cap


def run_snapshots(derive: Derive, tasks0: str, beliefs0: str, caps: Sequence[int], tq: int, bq: int) -> list[Snapshot]:
    """Resume `PLN.Derive` (the 6-argument form) from step count to step count and keep each state. `selections` is exact: when
    the task queue empties inside a chunk, `locate_exhaustion` finds the step. `engine_s` is the engine time of the chunks that make
    the run (resumption included), not of the locating."""
    caps = sorted(set(int(c) for c in caps))
    tasks_t, beliefs_t = tasks0, beliefs0
    done = 0
    snaps: list[Snapshot] = []
    ended_at: Optional[int] = None
    peak_t = peak_b = 0
    t_prev: list[Sent] = []
    b_prev: list[Sent] = []
    main_s = 0.0
    for cap in caps:
        if ended_at is not None:
            snaps.append(Snapshot(cap, ended_at, t_prev, b_prev, main_s, peak_t, peak_b))
            continue
        before = derive.engine_s
        T, B = derive(tasks_t, beliefs_t, done + 1, cap, tq, bq)
        main_s += derive.engine_s - before
        selections = cap
        if not T:
            selections = ended_at = locate_exhaustion(derive, tasks_t, beliefs_t, done, cap, tq, bq)
        peak_t = max(peak_t, len(T))
        peak_b = max(peak_b, len(B))
        snaps.append(Snapshot(cap, selections, T, B, main_s, peak_t, peak_b))
        t_prev, b_prev = T, B
        tasks_t, beliefs_t = list_text(T), list_text(B)
        done = cap
    return snaps


# ── reading conclusions ──────────────────────────────────────────────────────

def walk_of(enc: Enc, source: str, ids: Sequence[int], simple: bool, end: Optional[str] = None,
            end_terminal: Optional[bool] = None) -> Optional[list[pc.Link]]:
    """The links of a stamp in the order a walk from `source` uses them (each link once, a record last), or None when the stamp is
    no walk of the graph. With `end` the walk has to finish at that node (through a record when `end_terminal`); with `simple` no
    node is visited twice. (A node-simple route has exactly one order; a walk that comes back to a node can have several.)"""
    used = [enc.link_of[i] for i in ids if i in enc.link_of]
    if not used or len(used) != len(ids) or len({l.id for l in used}) != len(used):
        return None
    by_src: dict[str, list[pc.Link]] = {}
    for l in used:
        by_src.setdefault(l.src, []).append(l)

    def go(node: str, left: frozenset, acc: list, seen: frozenset) -> Optional[list]:
        if not left:
            if end is not None and (node != end or acc[-1].terminal != bool(end_terminal)):
                return None
            return acc
        for l in by_src.get(node, ()):
            if l.id in left and not (l.terminal and len(left) > 1) and not (simple and l.dst in seen):
                got = go(l.dst, left - {l.id}, acc + [l], seen | {l.dst})
                if got is not None:
                    return got
        return None

    return go(source, frozenset(l.id for l in used), [], frozenset([source]))


@dataclass
class Conclusions:
    routes: dict = field(default_factory=dict)       #: stamp tuple -> dict(target, hops, c, s, rs, end)
    walks: int = 0                                    #: conclusions that are walks revisiting a node (feedback), not routes
    false: int = 0                                    #: conclusions no walk of the graph licenses (always 0 with the restricted rules)
    c_mismatch: float = 0.0                           #: max |engine c - product of the link confidences| over routes

    def best(self) -> dict[str, dict]:
        """Per state: the best route confidence and the best rank-score among the routes found so far."""
        out: dict[str, dict] = {}
        for r in self.routes.values():
            e = out.setdefault(r["target"], {"routes": 0, "best_c": 0.0, "best_rs": 0.0})
            e["routes"] += 1
            e["best_c"] = max(e["best_c"], r["c"])
            e["best_rs"] = max(e["best_rs"], r["rs"])
        return out


def conclusions(enc: Enc, source: str, beliefs: Sequence[Sent]) -> Conclusions:
    """Read the level-1 beliefs: every `(State X)`/`(End X)` whose stamp is a node-simple route of the graph is a route (strength
    from its links, confidence from the engine), a walk that revisits a node is counted, anything else is a false chain."""
    opaque_source = enc.opaque[source]
    out = Conclusions()
    seen: set = set()
    for term, s, c, ids in beliefs:
        if not (isinstance(term, tuple) and len(term) == 2 and term[0] in ("State", "End")):
            continue
        if not ids or ids[0] != "0" or term[1] == opaque_source:
            continue
        if (term, ids) in seen:
            continue
        seen.add((term, ids))
        tail = tuple(int(i) for i in ids[1:])
        target = enc.key_of[term[1]]
        is_end = term[0] == "End"
        path = walk_of(enc, source, tail, simple=True, end=target, end_terminal=is_end)
        if path is None:
            if walk_of(enc, source, tail, simple=False, end=target, end_terminal=is_end) is None:
                out.false += 1
            else:
                out.walks += 1
            continue
        sv, cprod, cd, _ = route_values(path)
        cc = float(c)
        out.c_mismatch = max(out.c_mismatch, abs(cc - cprod))
        out.routes[tail] = {"target": target, "hops": len(path), "c": cc, "s": sv, "rs": sv * cc * DISCOUNT ** (len(path) - 1),
                            "end": term[0], "nodes": [source] + [l.dst for l in path]}
    return out


def recall(found: dict[str, dict], truth: dict[str, dict], key_found: str, key_truth: str, k: int) -> tuple[float, float]:
    """(recall@k, exact@k): the share of the k true-best states that were reached / reached with their true best route."""
    top = top_k(truth, key_truth, k)
    if not top:
        return 1.0, 1.0
    hit = sum(1 for v in top if v in found)
    exact = sum(1 for v in top if v in found and found[v][key_found] >= truth[v][key_truth] - 1e-12)
    return hit / len(top), exact / len(top)


# ══════════════════════════════ one configuration ══════════════════════════════

def flag_of(tq: int, bq: int, tasks_left: int, beliefs: int) -> dict:
    """The adapter's completeness test, and the naive one without the tq >= bq condition."""
    naive = tasks_left == 0 and beliefs < bq - 1
    proven = tq >= bq
    return {"complete": bool(proven and naive), "naive_complete": bool(naive), "proven": bool(proven),
            "flag": "complete" if (proven and naive) else "partial"}


def measure(enc: Enc, source: str, snap: Snapshot, tq: int, bq: int, truth: dict, npaths: int,
            ks: Sequence[int] = (20,)) -> dict:
    """Everything one table row needs from one snapshot."""
    con = conclusions(enc, source, snap.beliefs)
    found = con.best()
    row: dict[str, Any] = {
        "cap": snap.cap, "selections": snap.selections, "tasks_left": len(snap.tasks), "beliefs": len(snap.beliefs),
        "peak_tasks": snap.peak_tasks, "peak_beliefs": snap.peak_beliefs,
        "routes": len(con.routes), "legacy_paths": npaths, "route_share": len(con.routes) / npaths if npaths else 1.0,
        "states": len(found), "reachable_states": len(truth), "false_chains": con.false, "walks": con.walks,
        "c_mismatch": con.c_mismatch, "engine_s": round(snap.engine_s, 4), "tq": tq, "bq": bq,
    }
    row.update(flag_of(tq, bq, len(snap.tasks), len(snap.beliefs)))
    row["truly_complete"] = bool(len(con.routes) >= npaths and len(found) >= len(truth))
    for k in ks:
        rc, xc = recall(found, truth, "best_c", "best_c", k)
        rs, xs = recall(found, truth, "best_rs", "best_rs", k)
        row[f"recall_c@{k}"], row[f"exact_c@{k}"], row[f"recall_rs@{k}"], row[f"exact_rs@{k}"] = rc, xc, rs, xs
    return row


class Session:
    """Everything an experiment needs inside the child: the evaluator, the leak read from the library, the discount."""

    def __init__(self, engine: Any, library: str = "restricted"):
        self.engine = engine
        self.ev: Callable[[str], list] = lambda expr: [str(x) for x in engine.run("!" + expr)]
        engine.load(pc.library_text(library))
        self.leak = pc.probe_library(self.ev)
        self.discount = DISCOUNT                  # (chain-discount) of pln_deduction.metta:87, which these experiments do not load
        self.derive = Derive(self.ev)


def run_config(sess: Session, links: Sequence[pc.Link], source: str, truth: dict, npaths: int, tq: int, bq: int,
               caps: Sequence[int], level: float = 1.0, ks: Sequence[int] = (20,)) -> list[dict]:
    """Run `source` on the links reachable from it with queues (tq, bq) and measure at each step count of `caps`."""
    enc = Enc(links, sess.leak)
    sub = pc.reachable_links(list(links), source)
    d = Derive(sess.ev)
    snaps = run_snapshots(d, enc.tasks_text(source, level), enc.beliefs_text(sub, source, level), caps, tq, bq)
    rows = [measure(enc, source, s, tq, bq, truth, npaths, ks) for s in snaps]
    for r in rows:
        r["parse_s"] = round(d.parse_s, 4)
    return rows


# ══════════════════════════════ part a: the shipped graph + the feedback edge ══

def shipped_links(with_feedback: bool) -> list[pc.Link]:
    rows = [l for l in json.loads((CONTRACT / "links.json").read_text(encoding="utf-8"))["links"] if "patient" in l["stacks"]]
    links = [pc.Link(r["from"], r["to"], r["sign"], r["strength"], r["confidence"], terminal=r["terminal_only"],
                     kind="record" if r["terminal_only"] else "curated") for r in rows]
    if with_feedback:
        a, b, sg, s, c = FEEDBACK
        links.append(pc.Link(a, b, sg, s, c))
    pc.assign_ids(links)
    return links


def node_simple_closure(links: Sequence[pc.Link]) -> dict[tuple, tuple]:
    """The node-simple legacy closure {node path: (sign, s, c)} of a graph (the reference of the 109 pairs)."""
    sign = {("Pos", "Pos"): "Pos", ("Pos", "Neg"): "Neg", ("Neg", "Pos"): "Neg", ("Neg", "Neg"): "Pos"}
    out: dict[tuple, tuple] = {}
    for src in sorted({l.src for l in links}):
        for path in enumerate_paths(links, src):
            sg = "Pos"
            for l in path:
                sg = sign[(sg, l.sign)]
            s, c, cd, _ = route_values(path)
            out[tuple([src] + [l.dst for l in path])] = (sg, s, cd)
    return out


def part_a(sess: Session) -> dict:
    """The shipped patient graph plus the feedback edge, on the engine: every source, per selection."""
    out: dict[str, Any] = {"graphs": {}}
    for label, fb in (("shipped", False), ("shipped+feedback", True)):
        links = shipped_links(fb)
        ref = node_simple_closure(links)
        enc = Enc(links, sess.leak)
        K = 256
        per_source: dict[str, Any] = {}
        pairs: dict[tuple, tuple] = {}
        false = walks = 0
        for source in sorted({l.src for l in links}):
            sub = pc.reachable_links(links, source)
            d = Derive(sess.ev)
            # per-selection resumption: stop as soon as the task queue is empty
            tasks_t, beliefs_t = enc.tasks_text(source, 1.0), enc.beliefs_text(sub, source, 1.0)
            n = 0
            peak_t = peak_b = 0
            T: list = [None]
            B: list = []
            while T and n < 4 * K:
                T, B = d(tasks_t, beliefs_t, n + 1, n + 1, K, K)
                n += 1
                peak_t, peak_b = max(peak_t, len(T)), max(peak_b, len(B))
                tasks_t, beliefs_t = list_text(T), list_text(B)
            con = conclusions(enc, source, B)
            false += con.false
            walks += con.walks
            for tail, r in con.routes.items():
                pairs[tuple(r["nodes"])] = (r["s"], r["c"] * DISCOUNT ** (r["hops"] - 1))
            per_source[source] = {"selections": n, "peak_tasks": peak_t, "peak_beliefs": peak_b, "beliefs": len(B), "tasks_left": len(T),
                                  "links": len(sub), "routes": len(con.routes), "walks": con.walks, "false": con.false,
                                  **flag_of(K, K, len(T), len(B))}
        missing = [p for p in ref if p not in pairs]
        extra = [p for p in pairs if p not in ref]
        worst = max((max(abs(pairs[p][0] - ref[p][1]), abs(pairs[p][1] - ref[p][2])) for p in ref if p in pairs), default=0.0)
        top = max(per_source.items(), key=lambda kv: kv[1]["selections"])
        out["graphs"][label] = {
            "links": len(links), "sources": len(per_source), "pairs": len(pairs), "reference_pairs": len(ref),
            "missing": len(missing), "extra": len(extra), "max_dev": worst, "false_chains": false, "walks_not_answered": walks,
            "max_selections": top[1]["selections"], "max_selections_source": top[0],
            "max_peak_tasks": max(v["peak_tasks"] for v in per_source.values()),
            "max_peak_beliefs": max(v["peak_beliefs"] for v in per_source.values()),
            "all_complete": all(v["complete"] for v in per_source.values()), "K": K, "per_source": per_source,
            "state_levels_checked": "level 1 only; the two-level strengths are checked by the production run (part a, production)",
        }
    # where does the control start to cut? tq = bq = K on the feedback graph, every source, one shot, step cap 4K
    links = shipped_links(True)
    ref = node_simple_closure(links)
    enc = Enc(links, sess.leak)
    sweep = []
    for K in list(range(3, 41)) + [48, 64, 128, 256]:
        found: set = set()
        flagged_partial = lossy = lossy_but_complete = 0
        for source in sorted({l.src for l in links}):
            sub = pc.reachable_links(links, source)
            T, B = Derive(sess.ev)(enc.tasks_text(source, 1.0), enc.beliefs_text(sub, source, 1.0), 1, 4 * K, K, K)
            con = conclusions(enc, source, B)
            mine = {tuple(r["nodes"]) for r in con.routes.values()}
            want = {p for p in ref if p[0] == source}
            found |= mine
            fl = flag_of(K, K, len(T), len(B))
            if not fl["complete"]:
                flagged_partial += 1
            if mine != want:
                lossy += 1
                if fl["complete"]:
                    lossy_but_complete += 1
        sweep.append({"K": K, "pairs": len(found), "of": len(ref), "sources_flagged_partial": flagged_partial,
                      "sources_losing_pairs": lossy, "lossy_but_flagged_complete": lossy_but_complete})
    out["k_sweep"] = sweep
    first_full = next((s["K"] for s in sweep if s["pairs"] == s["of"] and s["sources_flagged_partial"] == 0), None)
    out["smallest_K_complete_and_unflagged"] = first_full
    return out


def part_a_production() -> dict:
    """The same graph through the PRODUCTION path (`pln_runner._runtime_run`, the hook, the sandbox): 109 explain pairs, the complete
    flag and the selections per source from the chainer's own status; and the legacy chainer on the same program (it hangs)."""
    import tempfile
    import config
    from core import pln_runner
    from core.metta_runtime import get_runtime
    import api
    config.PLN_RUNTIME = "petta"
    config.LOGS_DIR = Path(tempfile.mkdtemp(prefix="pln_control_logs_"))
    files = pln_runner.patient_stack(api._runtime_kb_paths())
    atom = "(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))\n"
    nodes = sorted({l.src for l in shipped_links(True)} | {l.dst for l in shipped_links(True)})
    queries = [f"!(explain &self {n} $x)" for n in nodes]
    res: dict[str, Any] = {}
    # legacy first: it never ends, so the request deadline (10 s here) is what stops it
    config.PLN_CHAINER = "legacy"
    config.PLN_QUERY_TIMEOUT_SECONDS = 10.0
    t0 = time.perf_counter()
    r = pln_runner._runtime_run("\n".join(queries), 0.0, files, atom)
    res["legacy"] = {"status": r.status, "error": (r.error or "")[:160], "wall_s": round(time.perf_counter() - t0, 2)}
    # now the pln chainer, with a generous deadline (it needs a fraction of it)
    config.PLN_CHAINER = "pln"
    config.PLN_QUERY_TIMEOUT_SECONDS = 120.0
    orig = pc.settings_from_config
    pc.settings_from_config = lambda: pc.ChainSettings(count_selections=True)
    try:
        t0 = time.perf_counter()
        r = pln_runner._runtime_run("\n".join(queries), 0.0, files, atom)
        wall = time.perf_counter() - t0
        chain = get_runtime("petta").last_run["chain"]
    finally:
        pc.settings_from_config = orig
    pairs = []
    for a in r.results:
        m = re.match(r"^\(Chain \(([^()]*)\) \(signed (Pos|Neg) \(stv (\S+) (\S+)\)\)\)$", a.atom)
        if m:
            pairs.append((tuple(m.group(1).split()), m.group(2), float(m.group(3)), float(m.group(4))))
    ref = node_simple_closure(shipped_links(True))
    got = {p[0]: p[1:] for p in pairs}
    worst = max((max(abs(got[k][1] - ref[k][1]), abs(got[k][2] - ref[k][2])) for k in ref if k in got), default=0.0)
    res["pln"] = {"status": r.status, "pairs": len(got), "duplicates": len(pairs) - len(got), "reference_pairs": len(ref),
                  "missing": len([k for k in ref if k not in got]), "extra": len([k for k in got if k not in ref]),
                  "max_dev": worst, "signs_equal": all(got[k][0] == ref[k][0] for k in ref if k in got),
                  "complete": chain["complete"], "attempts_max": chain["attempts_max"], "queue_max": chain["queue_max"],
                  "unattributed": chain["unattributed"], "wall_s": round(wall, 3),
                  "selections_max": max(max(v["selections"]) for v in chain["status"].values() if v.get("selections")),
                  "beliefs_max": max(max(v["beliefs"]) for v in chain["status"].values()),
                  "derive_runs": chain["derive_runs"], "derive_s": chain["derive_s"]}
    return res


# ══════════════════════════════ a model of the loop, for priorities the library does not offer ═════

PRIORITIES = ("confidence", "rank-score", "fifo")


def model_best_first(links: Sequence[pc.Link], root: str, priority: str, checkpoints: Sequence[int]) -> dict[int, dict]:
    """The loop of PLN.Derive on a DAG with no queue limit, as plain Python, with a choice of priority.

    A task is a route: `(State v)` (expanded against the links out of v) or `(Not (State v))` (a route that ended in a Neg link; one selection
    of its own turns it into the `(State v)`). Selecting a task costs one selection. The library ranks tasks by confidence only
    ("confidence"); "rank-score" is the application's strength x confidence x 0.9^(hops - 1) (monotone along a route, so the order stays
    best-first); "fifo" takes the oldest task (breadth first). Returns, at each checkpoint, the routes found (State conclusions) and the best
    value found per state. `model_matches_engine` checks the "confidence" model against the real engine, snapshot by snapshot."""
    import heapq
    out = _out_by(links)
    heap: list = []
    seq = itertools.count()

    def push(kind: str, node: str, s: float, c: float, hops: int) -> None:
        rs = s * c * DISCOUNT ** max(hops - 1, 0)
        key = {"confidence": -c, "rank-score": -rs, "fifo": float(next(seq))}[priority]
        heapq.heappush(heap, (key, next(seq), kind, node, s, c, hops))

    push("State", root, 1.0, 1.0, 0)
    found: dict[str, dict] = {}
    routes = 0
    snaps: dict[int, dict] = {}
    marks = sorted(set(int(c) for c in checkpoints))
    n = 0
    while heap and marks:
        _, _, kind, node, s, c, hops = heapq.heappop(heap)
        n += 1
        if kind == "Not":
            push("State", node, s, c, hops)               # Not-elimination: the same route, as a (State v) task
            conclusion = True
        else:
            conclusion = False
            for l in out.get(node, ()):
                if l.terminal:
                    continue
                s2, c2 = s * l.strength, c * l.confidence
                push("Not" if l.sign == "Neg" else "State", l.dst, s2, c2, hops + 1)
                if l.sign == "Pos":
                    routes += 1
                    e = found.setdefault(l.dst, {"best_c": 0.0, "best_rs": 0.0})
                    e["best_c"] = max(e["best_c"], c2)
                    e["best_rs"] = max(e["best_rs"], s2 * c2 * DISCOUNT ** hops)
        if conclusion:
            routes += 1
            e = found.setdefault(node, {"best_c": 0.0, "best_rs": 0.0})
            e["best_c"] = max(e["best_c"], c)
            e["best_rs"] = max(e["best_rs"], s * c * DISCOUNT ** max(hops - 1, 0))
        while marks and n == marks[0]:
            snaps[n] = {"routes": routes, "states": len(found), "found": {k: dict(v) for k, v in found.items()}, "tasks_left": len(heap)}
            marks.pop(0)
    for m in marks:                                     # the queue ran dry before this checkpoint
        snaps[m] = {"routes": routes, "states": len(found), "found": {k: dict(v) for k, v in found.items()}, "tasks_left": 0}
    return snaps


def model_row(snap: dict, truth: dict, ks: Sequence[int] = (20,)) -> dict:
    row = {"routes": snap["routes"], "states": snap["states"], "tasks_left": snap["tasks_left"]}
    for k in ks:
        row[f"recall_c@{k}"], row[f"exact_c@{k}"] = recall(snap["found"], truth, "best_c", "best_c", k)
        row[f"recall_rs@{k}"], row[f"exact_rs@{k}"] = recall(snap["found"], truth, "best_rs", "best_rs", k)
    return row


def model_matches_engine(links: Sequence[pc.Link], root: str, engine_rows: Sequence[dict], reached: dict[int, list]) -> dict:
    """The "confidence" model against the engine's snapshots (no queue limit was hit): same routes, same states, same set of states."""
    snaps = model_best_first(links, root, "confidence", [r["selections"] for r in engine_rows])
    ok_routes = all(snaps[r["selections"]]["routes"] == r["routes"] for r in engine_rows)
    ok_states = all(sorted(snaps[r["selections"]]["found"]) == sorted(reached[r["selections"]]) for r in engine_rows)
    return {"snapshots": len(engine_rows), "routes_equal": ok_routes, "state_sets_equal": ok_states, "equal": bool(ok_routes and ok_states)}


# ══════════════════════════════ parts m and l: the SYNTHETIC networks ══════════

def synthetic_context(spec: SynSpec) -> dict:
    links = generate(spec)
    truth = truth_dag(links, spec.root)
    return {"spec": spec, "links": links, "truth": truth, "npaths": path_count(links, spec.root)}


def describe_network(ctx: dict) -> dict:
    spec, links, truth = ctx["spec"], ctx["links"], ctx["truth"]
    indeg: dict[str, int] = {}
    for l in links:
        indeg[l.dst] = indeg.get(l.dst, 0) + 1
    return {"name": spec.name, "describe": spec.describe(), "links": len(links), "nodes": len(truth) + 1,
            "reachable_states": len(truth), "legacy_paths": ctx["npaths"], "complete_run_selections": complete_run_selections(links, spec.root),
            "diamond_nodes": sum(1 for d in indeg.values() if d >= 2), "max_indegree": max(indeg.values()),
            "neg_links": sum(1 for l in links if l.sign == "Neg"), "layers": spec.layers,
            "links_sha256": hashlib.sha256(json.dumps([(l.src, l.dst, l.sign, l.strength, l.confidence) for l in links]).encode()).hexdigest()[:16]}


def part_m(sess: Session) -> dict:
    """SYNTHETIC-M: the completeness test against the ground truth."""
    ctx = synthetic_context(SYN_M)
    links, truth, npaths, root = ctx["links"], ctx["truth"], ctx["npaths"], SYN_M.root
    out: dict[str, Any] = {"network": describe_network(ctx), "contract": [], "blind": []}
    for K in (32, 64, 128, 256, 512, 1024, 1536, 2048):
        row = run_config(sess, links, root, truth, npaths, K, K, [4 * K])[0]
        out["contract"].append(row)
    for tq in (2, 3, 4, 6, 8, 12, 16, 32, 64):
        row = run_config(sess, links, root, truth, npaths, tq, 4096, [4096])[0]
        out["blind"].append(row)
    # both levels (as the chainer runs them) against the product of the link strengths, on the complete run
    out["two_level"] = two_level_check(sess, ctx, 4096, 4096, 16384)
    return out


def two_level_check(sess: Session, ctx: dict, tq: int, bq: int, cap: int) -> dict:
    """Run both levels (as the chainer does) and compare |s1 - s0| / (1 - e)^hops with the product of the link strengths."""
    links, root = ctx["links"], ctx["spec"].root
    enc = Enc(links, sess.leak)
    sub = pc.reachable_links(links, root)
    b = {}
    for level in (1.0, 0.0):
        d = Derive(sess.ev)
        T, B = d(enc.tasks_text(root, level), enc.beliefs_text(sub, root, level), 1, cap, tq, bq)
        b[level] = B
    lvl0 = {(t, ids): s for (t, s, c, ids) in b[0.0]}
    worst = 0.0
    n = 0
    con = conclusions(enc, root, b[1.0])
    for (term, s1, c1, ids) in b[1.0]:
        if not (isinstance(term, tuple) and len(term) == 2 and term[0] in ("State", "End") and ids and ids[0] == "0"):
            continue
        tail = tuple(int(i) for i in ids[1:])
        if tail not in con.routes:
            continue
        s0 = lvl0.get((term, ids))
        if s0 is None:
            continue
        hops = con.routes[tail]["hops"]
        eng = abs(float(s1) - float(s0)) / (1.0 - sess.leak) ** hops
        worst = max(worst, abs(eng - con.routes[tail]["s"]))
        n += 1
    return {"routes_checked": n, "max_abs_diff_strength": worst}


def part_seeds(sess: Session) -> dict:
    """Ten more instances of the SYNTHETIC-L recipe (other seeds): is the recall of the first selections a property of the library's order or
    of one lucky network? Engine at tq = bq = 4096 (the confidence priority) and the model for the three priorities."""
    caps = [20, 40, 80, 160]
    rows = []
    for seed in range(1, 11):
        spec = SynSpec(f"SYNTHETIC-L(seed {seed})", seed=seed, layers=SYN_L.layers, width=SYN_L.width, fanout=SYN_L.fanout, root_fanout=SYN_L.root_fanout)
        ctx = synthetic_context(spec)
        links, truth, npaths = ctx["links"], ctx["truth"], ctx["npaths"]
        eng = run_config(sess, links, spec.root, truth, npaths, 4096, 4096, caps)
        mod = {pr: {n: model_row(sn, truth) for n, sn in model_best_first(links, spec.root, pr, caps).items()} for pr in PRIORITIES}
        agree = all(e["routes"] == mod["confidence"][e["cap"]]["routes"] and e["states"] == mod["confidence"][e["cap"]]["states"]
                    and all(abs(e[k] - mod["confidence"][e["cap"]][k]) < 1e-12 for k in ("recall_c@20", "exact_c@20", "recall_rs@20", "exact_rs@20"))
                    for e in eng)
        rows.append({"seed": seed, "links": len(links), "legacy_paths": npaths, "states": len(truth), "engine": eng, "model_agrees_with_engine": bool(agree),
                     "model": {pr: {str(n): v for n, v in d.items()} for pr, d in mod.items()}})
    return {"caps": caps, "rows": rows}


def part_l(sess: Session) -> dict:
    ctx = synthetic_context(SYN_L)
    links, truth, npaths, root = ctx["links"], ctx["truth"], ctx["npaths"], SYN_L.root
    ks = (5, 10, 20, 50)
    out: dict[str, Any] = {"network": describe_network(ctx), "ks": list(ks)}
    # 1. the step sweep: one long run at K = 4096, snapshots at growing step counts
    caps = [10, 20, 40, 80, 160, 320, 640, 1024]
    enc = Enc(links, sess.leak)
    sub = pc.reachable_links(list(links), root)
    d = Derive(sess.ev)
    snaps = run_snapshots(d, enc.tasks_text(root, 1.0), enc.beliefs_text(sub, root, 1.0), caps, 4096, 4096)
    out["step_sweep"] = [measure(enc, root, sn, 4096, 4096, truth, npaths, ks) for sn in snaps]
    reached = {sn.selections: sorted(conclusions(enc, root, sn.beliefs).best()) for sn in snaps}
    # the library's loop as plain Python, validated against those snapshots, then run with the priorities the library does not offer
    out["model_check"] = model_matches_engine(links, root, out["step_sweep"], reached)
    model_caps = [10, 20, 40, 80, 160, 320, 640, 1024]
    out["model"] = {pr: {str(n): model_row(sn, truth, ks) for n, sn in model_best_first(links, root, pr, model_caps).items()} for pr in PRIORITIES}
    # 2. the baseline both levels (the service runs two) against the stamp products
    out["two_level"] = two_level_check(sess, ctx, 2048, 2048, 256)
    # 3. joint queue sweep tq = bq = K at a fixed step cap
    CAP = 512
    out["cap"] = CAP
    out["joint_sweep"] = [run_config(sess, links, root, truth, npaths, K, K, [CAP])[0] for K in (64, 128, 256, 512, 1024, 2048, 4096)]
    # 4. the beam: bq large, tq small
    out["tq_sweep"] = [run_config(sess, links, root, truth, npaths, tq, 4096, [CAP])[0] for tq in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
    # 5. the memory: tq large (so completeness is provable), bq small
    out["bq_sweep"] = [run_config(sess, links, root, truth, npaths, 4096, bq, [CAP])[0] for bq in (128, 256, 512, 1024, 2048)]
    # 6. the belief count the run starts with: a belief queue below it forgets the lowest-confidence links first
    out["initial_beliefs"] = len(pc.reachable_links(links, root)) + 1
    return out


# ══════════════════════════════ legacy comparison and contract mode ═════════════

def part_legacy() -> dict:
    """The legacy chainer on SYNTHETIC-L (it enumerates every path) and contract mode on it (it must be stopped by the deadline)."""
    import tempfile
    import config
    from core import pln_runner
    config.PLN_RUNTIME = "petta"
    config.LOGS_DIR = Path(tempfile.mkdtemp(prefix="pln_control_logs_"))
    ctx = synthetic_context(SYN_L)
    links = ctx["links"]
    text = "\n".join(f"(Effect {l.src} {l.dst} {l.sign} (stv {l.strength} {l.confidence}))" for l in links) + "\n"
    files = [REPO / f for f in ("system_types.metta", "logical_predicates.metta", "epistemic_calibration.metta", "pln_deduction.metta")]
    res: dict[str, Any] = {"network": describe_network(ctx)}
    config.PLN_CHAINER = "legacy"
    walls = []
    n = 0
    for _ in range(3):
        t0 = time.perf_counter()
        r = pln_runner._runtime_run(f"!(collapse (infer &self {SYN_L.root} $x))", 0.0, files, text)
        walls.append(time.perf_counter() - t0)
        n = len(re.findall(r"\(signed ", r.results[0].atom)) if r.results else 0
    trivial = []
    for _ in range(3):
        t0 = time.perf_counter()
        pln_runner._runtime_run("!(+ 1 2)", 0.0, files, text)
        trivial.append(time.perf_counter() - t0)
    res["legacy"] = {"status": r.status, "derivations": n, "wall_s_median": round(sorted(walls)[1], 3),
                     "wall_s_all": [round(w, 3) for w in walls], "start_and_load_only_s_median": round(sorted(trivial)[1], 3)}
    # contract mode: the first queue size is picked from the route count; the estimate says what the service would attempt
    sub = pc.reachable_links(links, SYN_L.root)
    st = pc.ChainSettings()
    est = pc.estimate_beliefs(sub, SYN_L.root, st.queue_size * st.growth ** st.rungs)
    sizes = [st.queue_size * st.growth ** i for i in range(st.rungs)]
    first = next((k for k in sizes if k > est + 2), sizes[-1])
    res["contract_mode"] = {"estimated_beliefs": est, "rungs": sizes, "first_attempt_K": first, "largest_K": sizes[-1]}
    # and the real thing under a 30 s deadline: the request must fail, not run on
    config.PLN_CHAINER = "pln"
    config.PLN_QUERY_TIMEOUT_SECONDS = 30.0
    t0 = time.perf_counter()
    r = pln_runner._runtime_run(f"!(infer &self {SYN_L.root} L7_0)", 0.0, files, text)
    res["contract_mode"].update({"status": r.status, "error": (r.error or "")[:200], "wall_s": round(time.perf_counter() - t0, 1),
                                 "deadline_s": 30.0})
    return res


# ══════════════════════════════ running ═════════════════════════════════════════

def in_child(fn: Callable[[Session], dict], deadline_s: float = 3000.0) -> dict:
    """Run `fn(Session)` inside the forked PeTTa child (the production hook) and return what it returns."""
    from core.metta_runtime import PettaRuntime
    import config
    config.PLN_RUNTIME = "petta"

    def step(engine: Any) -> dict:
        return fn(Session(engine))

    rt = PettaRuntime()
    rt.run_program("", ["!(+ 1 2)"], deadline_s=deadline_s, after_load=step)
    return rt.last_run["chain"]


def environment() -> dict:
    import platform
    import subprocess
    env = {"python": platform.python_version(), "platform": platform.platform(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    try:
        env["swipl"] = subprocess.run(["swipl", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        env["swipl"] = "unknown"
    for name in ("lib_pln_fork.metta", "lib_pln_restricted.metta"):
        env[f"vendor/{name}"] = hashlib.sha256((REPO / "vendor" / "pln_fork" / name).read_bytes()).hexdigest()[:16]
    try:
        env["petta_commit"] = subprocess.run(["git", "-C", str(REPO / ".petta" / "PeTTa"), "rev-parse", "--short=8", "HEAD"],
                                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        env["petta_commit"] = "unknown"
    return env


PARTS = ("a", "m", "l", "seeds", "legacy")


def run_part(part: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if part == "a":
        data = {"experiment": in_child(part_a), "production": part_a_production()}
    elif part == "m":
        data = in_child(part_m)
    elif part == "l":
        data = in_child(part_l)
    elif part == "seeds":
        data = in_child(part_seeds)
    elif part == "legacy":
        data = part_legacy()
    else:
        raise SystemExit(f"unknown part {part!r}")
    data["_measured_by"] = f"python scripts/pln_control_demo.py run --part {part}"
    data["_wall_s"] = round(time.perf_counter() - t0, 1)
    data["_environment"] = environment()
    (out_dir / f"control_{part}.json").write_text(json.dumps(data, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"part {part}: {data['_wall_s']} s -> {out_dir / f'control_{part}.json'}")


# ══════════════════════════════ rendering ═══════════════════════════════════════

def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def share(x: float) -> str:
    return f"{100 * x:.1f}%" if x < 0.1 else f"{100 * x:.0f}%"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def block_a(d: dict) -> str:
    g = d["experiment"]["graphs"]
    prod = d["production"]
    rows = []
    for label, v in g.items():
        rows.append([label, v["links"], v["sources"], f"{v['pairs']} of {v['reference_pairs']}", v["missing"], v["extra"],
                     f"{v['max_dev']:.1e}", v["false_chains"], f"{v['max_selections']} ({v['max_selections_source']})",
                     v["max_peak_tasks"], v["max_peak_beliefs"], f"{v['K']} / {v['K']}", "yes" if v["all_complete"] else "NO"])
    t1 = md_table(["graph", "links", "sources", "pairs (node-simple)", "missing", "extra", "max dev", "false chains", "max selections (source)",
                   "peak tasks", "peak beliefs", "tq / bq", "all sources complete"], rows)
    legacy, pln = prod["legacy"], prod["pln"]
    t2 = md_table(["chainer", "status", "pairs", "missing / extra / duplicate", "max dev vs node-simple reference", "complete", "attempts", "queue K",
                   "max selections", "max beliefs", "derive runs", "wall"],
                  [["legacy (PeTTa)", legacy["status"], "-", "-", "-", "-", "-", "-", "-", "-", "-", f"{legacy['wall_s']} s ({legacy['error'][:60]})"],
                   ["pln R2", pln["status"], pln["pairs"], f"{pln['missing']} / {pln['extra']} / {pln['duplicates']}", f"{pln['max_dev']:.1e}",
                    pln["complete"], pln["attempts_max"], pln["queue_max"], pln["selections_max"], pln["beliefs_max"], pln["derive_runs"],
                    f"{pln['wall_s']} s"]])
    sweep = d["experiment"]["k_sweep"]
    t3 = md_table(["K (tq = bq)", "pairs found", "sources flagged partial", "sources that lost a pair", "lost a pair but flagged complete"],
                  [[s["K"], f"{s['pairs']} of {s['of']}", s["sources_flagged_partial"], s["sources_losing_pairs"], s["lossy_but_flagged_complete"]]
                   for s in sweep if s["K"] <= 24 or s["K"] in (25, 28, 30, 32, 33, 34, 36, 40, 64, 256)])
    return "\n\n".join([t1, t2, t3])


def _row_cells(r: dict, with_k: bool = True) -> list:
    return [r["cap"], r["selections"], r["tasks_left"], r["beliefs"], f"{r['routes']} ({share(r['route_share'])})",
            f"{r['states']} / {r['reachable_states']} ({pct(r['states'] / r['reachable_states'])})", f"{pct(r['recall_c@20'])} / {pct(r['exact_c@20'])}",
            f"{pct(r['recall_rs@20'])} / {pct(r['exact_rs@20'])}", r["flag"] + ("" if r["proven"] else " (unproven)"), f"{r['engine_s']:.2f}"]


def block_l(d: dict) -> str:
    net = d["network"]
    head = ["step cap", "selections", "tasks left", "beliefs", "routes (share of legacy paths)", "states reached (share; the recall a random set of that size would have)", "recall@20 / exact by confidence",
            "recall@20 / exact by rank-score", "flag", "engine s (one level)"]
    parts = []
    parts.append(md_table(["network", "links", "reachable states", "legacy paths", "selections a complete run needs (1 + routes + routes ending in a Neg link)", "diamond nodes (in-degree >= 2)", "max in-degree", "Neg links", "links sha256 (first 16)"],
                          [[net["name"], net["links"], net["reachable_states"], net["legacy_paths"], net["complete_run_selections"], net["diamond_nodes"], net["max_indegree"], net["neg_links"], net["links_sha256"]]]))
    sw = d["step_sweep"]
    parts.append("STEP SWEEP, tq = bq = 4096, one level, snapshots of one run:\n\n" + md_table(head, [_row_cells(r) for r in sw]))
    ks = d["ks"]
    parts.append("RECALL BY k at those snapshots (recall / exact):\n\n" + md_table(
        ["selections"] + [f"k={k} conf" for k in ks] + [f"k={k} rank-score" for k in ks],
        [[r["selections"]] + [f"{pct(r[f'recall_c@{k}'])} / {pct(r[f'exact_c@{k}'])}" for k in ks]
         + [f"{pct(r[f'recall_rs@{k}'])} / {pct(r[f'exact_rs@{k}'])}" for k in ks] for r in sw]))
    cap = d["cap"]
    qhead = ["tq", "bq"] + head[1:]
    def qrow(r):
        c = _row_cells(r)
        return [r["tq"], r["bq"]] + c[1:]
    parts.append(f"JOINT QUEUE SWEEP, tq = bq = K, step cap {cap}:\n\n" + md_table(qhead, [qrow(r) for r in d["joint_sweep"]]))
    parts.append(f"TASK-QUEUE SWEEP (the beam), bq = 4096, step cap {cap}:\n\n" + md_table(qhead, [qrow(r) for r in d["tq_sweep"]]))
    parts.append(f"BELIEF-QUEUE SWEEP (the memory), tq = 4096, step cap {cap}:\n\n" + md_table(qhead, [qrow(r) for r in d["bq_sweep"]]))
    tl = d["two_level"]
    parts.append(f"Two-level check at 256 steps (as the chainer runs it): {tl['routes_checked']} routes, max |engine strength - product of link strengths| = {tl['max_abs_diff_strength']:.1e}.")
    return "\n\n".join(parts)


def block_m(d: dict) -> str:
    net = d["network"]
    head = ["tq", "bq", "step cap", "selections", "tasks left", "beliefs", "routes / legacy paths", "states", "flag", "naive flag", "truly complete"]
    def row(r):
        return [r["tq"], r["bq"], r["cap"], r["selections"], r["tasks_left"], r["beliefs"], f"{r['routes']} / {r['legacy_paths']}",
                f"{r['states']} / {r['reachable_states']}", r["flag"], "complete" if r["naive_complete"] else "partial", "yes" if r["truly_complete"] else "no"]
    t0 = md_table(["network", "links", "reachable states", "legacy paths", "selections a complete run needs (1 + routes + routes ending in a Neg link)", "diamond nodes", "Neg links"],
                  [[net["name"], net["links"], net["reachable_states"], net["legacy_paths"], net["complete_run_selections"], net["diamond_nodes"], net["neg_links"]]])
    contract = d["contract"]
    wrong = sum(1 for r in contract if r["complete"] and not r["truly_complete"])
    blind = d["blind"]
    naive_wrong = sum(1 for r in blind if r["naive_complete"] and not r["truly_complete"])
    return "\n\n".join([t0, "CONTRACT MODE, tq = bq = K, step cap 4K:\n\n" + md_table(head, [row(r) for r in contract]),
                        f"Runs flagged complete that were not truly complete: {wrong} of {len(contract)}.",
                        "TASK QUEUE SMALLER THAN THE BELIEF QUEUE (tq < bq = 4096, step cap 4096):\n\n" + md_table(head, [row(r) for r in blind]),
                        f"Runs the naive test (tasks empty and beliefs < bq - 1) calls complete although routes were lost: {naive_wrong} of {len(blind)}; the adapter's test calls none of them complete."])


def block_legacy(d: dict) -> str:
    lg, cm, net = d["legacy"], d["contract_mode"], d["network"]
    return md_table(["what", "measured"], [
        ["legacy chainer, every path of SYNTHETIC-L (PeTTa, whole request incl. start and load)", f"{lg['derivations']} derivations in {lg['wall_s_median']} s (median of 3: {lg['wall_s_all']}); start and load alone {lg['start_and_load_only_s_median']} s"],
        ["contract mode on SYNTHETIC-L: the route-count estimate of beliefs", f"{cm['estimated_beliefs']} (largest queue size of the schedule: {cm['largest_K']}; first attempt would be K = {cm['first_attempt_K']})"],
        [f"contract mode on SYNTHETIC-L under a {cm['deadline_s']:.0f} s request deadline", f"{cm['status']} after {cm['wall_s']} s: {cm['error']}"]])


def block_model(d: dict) -> str:
    """The model of the loop (three priorities) on SYNTHETIC-L at the step sweep's selection counts."""
    mc = d["model_check"]
    head = ["selections", "priority", "routes found", "states reached", "recall@20 / exact by confidence", "recall@20 / exact by rank-score"]
    rows = []
    for n in sorted(next(iter(d["model"].values())), key=int):
        for pr in PRIORITIES:
            r = d["model"][pr][n]
            rows.append([n, pr, r["routes"], f"{r['states']} / {d['network']['reachable_states']}", f"{pct(r['recall_c@20'])} / {pct(r['exact_c@20'])}",
                         f"{pct(r['recall_rs@20'])} / {pct(r['exact_rs@20'])}"])
    check = (f"Model check: the `confidence` model equals the engine at all {mc['snapshots']} snapshots of the step sweep in routes found ({mc['routes_equal']}) "
             f"and in the set of states reached ({mc['state_sets_equal']}).")
    return check + "\n\n" + md_table(head, rows)


def block_seeds(d: dict) -> str:
    caps = d["caps"]
    rows = []
    for r in d["rows"]:
        cells = [r["seed"], r["legacy_paths"], r["states"]]
        for e in r["engine"]:
            if e["cap"] in (20, 40, 80):
                cells.append(f"{pct(e['recall_c@20'])} / {pct(e['exact_c@20'])}; {pct(e['recall_rs@20'])} / {pct(e['exact_rs@20'])}")
        cells.append("yes" if r["model_agrees_with_engine"] else "NO")
        rows.append(cells)
    t1 = md_table(["seed", "legacy paths", "reachable states"] + [f"{c} selections: conf recall / exact; rank-score recall / exact" for c in (20, 40, 80)]
                  + ["model equals engine"], rows)
    agg = []
    for pr in PRIORITIES:
        for c in caps:
            vals = [r["model"][pr][str(c)] for r in d["rows"]]
            def mm(k):
                xs = [v[k] for v in vals]
                return f"{pct(sum(xs) / len(xs))} (min {pct(min(xs))})"
            agg.append([pr, c, mm("recall_c@20"), mm("exact_c@20"), mm("recall_rs@20"), mm("exact_rs@20")])
    t2 = md_table(["priority (model)", "selections", "recall@20 by confidence, mean (min)", "exact by confidence", "recall@20 by rank-score", "exact by rank-score"], agg)
    return t1 + "\n\nTen instances, three priorities (the model; `confidence` is the library's):\n\n" + t2


BLOCKS = {"a": block_a, "m": block_m, "l": block_l, "seeds": block_seeds, "legacy": block_legacy}


def render(out_dir: Path, doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    for part, fn in list(BLOCKS.items()) + [("l_model", block_model)]:
        path = out_dir / f"control_{part.split('_')[0]}.json"
        if not path.exists():
            print(f"skip {part}: {path} missing")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        begin, end = f"<!-- BEGIN GENERATED control_{part} -->", f"<!-- END GENERATED control_{part} -->"
        if begin not in text or end not in text:
            raise SystemExit(f"{doc} has no generated block control_{part}")
        pre, rest = text.split(begin, 1)
        _, post = rest.split(end, 1)
        stamp = f"Measured by `{data['_measured_by']}` ({data['_wall_s']} s, script {data['_environment']['script_sha256']}, {data['_environment'].get('swipl', '')}, PeTTa {data['_environment'].get('petta_commit', '')})."
        text = pre + begin + "\n" + stamp + "\n\n" + fn(data) + "\n" + end + post
    doc.write_text(text, encoding="utf-8")
    print(f"rendered into {doc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    # never read pln_chat/.env from a script. Set here, not at import: importing this module (as the tests do) must not
    # change the environment of the importer. Nothing imports `config` before `main` runs.
    os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a part of the demonstration and write its JSON")
    r.add_argument("--part", choices=list(PARTS) + ["all"], default="all")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    g = sub.add_parser("render", help="write the generated blocks of CONTROL_DEMO.md from the JSON files")
    g.add_argument("--out", type=Path, default=DEFAULT_OUT)
    g.add_argument("--doc", type=Path, default=DOC)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        for part in (PARTS if args.part == "all" else (args.part,)):
            run_part(part, args.out)
    else:
        render(args.out, args.doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
