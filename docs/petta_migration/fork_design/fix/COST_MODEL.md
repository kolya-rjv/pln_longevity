# What a PLN.Derive run costs, and how the chainer refuses what a request cannot afford

Work tree `pln_longevity_petta` (branch petta-migration), PeTTa `fea83178` on SWI-Prolog 10.0.2, 2026-10-24. Every number below was measured by the scripts in
`probes/` (`calib2.py` for the calibration, `repro.py cost_*` for the scenarios) on this Mac; none is copied from a note.

## Why the first build could be killed by a few hundred bytes of caller atoms

1. **The library derives walks, not routes.** A derivation uses each LINK once (the stamp rule). On a graph without a cycle that is the number of node-simple routes
   the legacy chainer enumerates. On a cyclic one it is far more: a walk may revisit a node. The first build sized its queues from the node-simple route count, so on
   a cyclic graph the first queue size was hopeless and the retries (8 x, 64 x) multiplied the cost. Eight reversed edges of the shipped patient graph: 10 node-simple routes,
   262 distinct walks from the worst source (`walk_stats`).
2. **The cost is not linear in anything the first build looked at.** One run compares the selected task with every belief, and the stamp checks and sorts walk the stamps, so
   `seconds ~ beliefs^2 x mean stamp size`. A chain of n links is therefore about n^3.7 (50: 0.12 s, 100: 1.4 s, 200: 19.6 s for ONE source), a layered DAG with 363 routes
   1.4 s, one with 1,092 routes 15.6 s, one with 3,279 routes 145 s.
3. **The table was derived for every source before the first query.** A 200-link chain has 200 sources; the sum over all of them is minutes (the first build: killed at the deadline), for a
   question about one source. A program that never calls `infer` paid the same.

## The model

    seconds of one run  =  0.002 + 1.0e-6 x B^2 x max(1, m)         B = links + 1 + S + N
    seconds of a source =  2 x that                                 (the intervention at level 1 and at level 0)

`S` is the number of distinct conclusions the library will hold (walks: distinct (end node, set of links used)), `N` how many of them end in a Neg link (they carry the
`(Not ..)` sentence too), `m` the mean number of links in a conclusion's stamp. All four come from the links alone (`pln_chainer.walk_stats`), in milliseconds, before the
library is loaded. The constant is a planning figure for refusing the hopeless; measured against the runs it is within -50 percent (small graphs, where a fixed cost dominates in the
other direction) and +12 percent (the largest):

| graph | links | S | mean stamp | B | measured | model | measured / model |
|---|---|---|---|---|---|---|---|
| chain 30 | 30 | 30 | 15.5 | 61 | 0.027 s | 0.060 s | 0.45 |
| chain 60 | 60 | 60 | 30.5 | 121 | 0.224 s | 0.449 s | 0.50 |
| chain 90 | 90 | 90 | 45.5 | 181 | 0.963 s | 1.493 s | 0.65 |
| chain 120 | 120 | 120 | 60.5 | 241 | 2.79 s | 3.52 s | 0.79 |
| chain 150 | 150 | 150 | 75.5 | 301 | 6.53 s | 6.84 s | 0.95 |
| chain 100, alternating signs | 100 | 100 | 50.5 | 251 | 3.06 s | 3.18 s | 0.96 |
| layered 4x5, fan 3 | 36 | 120 | 3.55 | 193 | 0.128 s | 0.134 s | 0.95 |
| layered 5x6, fan 3 | 51 | 363 | 4.52 | 586 | 1.36 s | 1.55 s | 0.87 |
| layered 6x6, fan 3 | 57 | 363 | 4.52 | 624 | 1.50 s | 1.76 s | 0.85 |
| layered 7x6, fan 3 | 60 | 363 | 4.52 | 640 | 1.65 s | 1.85 s | 0.89 |
| layered 8x6, fan 3 | 81 | 363 | 4.52 | 628 | 1.44 s | 1.79 s | 0.81 |
| layered 6x7, fan 3 | 75 | 1,092 | 5.51 | 1,819 | 15.6 s | 18.2 s | 0.86 |
| layered 5x8, fan 3 | 81 | 3,279 | 6.50 | 4,908 | 144.9 s | 156.6 s | 0.93 |
| complete digraph, 4 nodes | 12 | 442 | 6.10 | 455 | 1.25 s | 1.27 s | 0.98 |
| ring of 8 + 6 chords | 14 | 99 | 6.41 | 200 | 0.237 s | 0.258 s | 0.92 |
| ring of 10 + 9 chords | 19 | 196 | 8.92 | 430 | 1.55 s | 1.65 s | 0.94 |
| ring of 10 + 12 chords | 22 | 704 | 9.23 | 1,736 | 28.3 s | 27.8 s | 1.02 |
| ring of 12 + 12 chords | 24 | 165 | 8.98 | 288 | 0.620 s | 0.747 s | 0.83 |
| ring of 12 + 16 chords | 28 | 1,286 | 10.85 | 3,779 | 174.0 s | 154.9 s | 1.12 |

(`probes/calib2.py` reprints these 19 rows; `results/calib2.txt` keeps the lines of the run that produced them.)

## What the chainer does with it

* **Plan** (`plan_sources`, before the library is loaded): for every source to derive, the reachable links, the walks, the beliefs and the modelled seconds. Contract mode refuses
  (`PlnInputError`, HTTP 422 `refused_form`, naming the source and the numbers) a source whose walks cannot fit the largest queue (K = 16,384; the count stops at 16,385), and a table
  whose modelled time is more than 80 percent of the request's budget (`PLN_QUERY_TIMEOUT_SECONDS`, 60 s by default). The research lanes choose their own budget and are never refused.
* **Demand** (`demand_of`): only the sources the program asks about are derived. The literal sources of its direct `infer`/`explain` calls; none when nothing in it can reach them; every
  source when it cannot tell (a consumer such as `score-candidate` or `rank-interventions-for-patient`, an unbound or computed source, `infer` passed as a value, a program that reads
  `PLNAnswer` or `PLNStatus` itself). The functions that can reach `infer`/`explain` are found from the knowledge-base text (an over-approximation, so it errs toward "every source").
  Of the 50 composite queries of the contract, 43 call a consumer (every source), 4 name a literal source, 3 reach nothing (the risk projections); the 7 narrowed ones, run alone,
  equal the legacy chainer (`tests/test_pln_fixes.py`).
* **Clock** (`compute_table(deadline_at=)`): checked between sources and between queue sizes; a table that runs out of time ends with a named error before the parent's kill.
* **`_rungs`** raises when no queue size can hold the expected sentences (contract mode), instead of running the last one into the deadline.
* The adjacency is built once per table, and the line number of an `Effect` atom is found by bisection, not by a count per atom. The review's 4,000 independent pairs (`S_i -> T_i`, 153,780 characters of caller atoms): 9.0 s for the whole
  table before; now 0.65 s when the program asks about one source (legacy 0.42 s) and 6.5 s when it needs every source (legacy 0.41 s; 8,000 `PLN.Derive` runs of 0.8 ms are the floor of an eager table, which is why the demand analysis exists).

## Before and after (the reviewers' scenarios, re-run by me; deadline 40 s unless noted)

See section 3 of `NOTES.md` in this directory and `probes/repro.py`: layered DAG 363 routes 15.5 s -> 0.6 s (one source derived); complete
digraph of 5 nodes killed at 40 s -> refused in 0.2 s; chain of 200 links killed at 40 s -> refused in 0.2 s at a 40 s budget (it runs at the default 60 s: 33 s modelled, ~40 s measured);
80 dense links beside a plain `match` query killed at 30 s -> 0.2 s; the shipped graph plus 8 reversed edges 2.3 s -> 0.5 s.

## Reproduce

    FIX_OUT=/some/scratch python docs/petta_migration/fork_design/fix/probes/calib2.py        # the calibration table (needs PeTTa; about 6 minutes)
    FIX_OUT=/some/scratch python docs/petta_migration/fork_design/fix/probes/mk_etl.py        # the DrugAge build the DrugAge scenarios use
    FIX_OUT=/some/scratch python docs/petta_migration/fork_design/fix/probes/repro.py cost_digraph5   # one scenario, one JSON line (zero, rank, exact, skip, back, badcfg, dabad, readans, order, cost_*)
