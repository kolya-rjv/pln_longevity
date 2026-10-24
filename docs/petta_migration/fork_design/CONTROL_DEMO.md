# Control demonstration: what patham9's queues do, and what they do not

Phase 6, gate 6c (report only; no pin). Work tree `/Users/kolya/PycharmProjects/pln_longevity_petta` (branch `petta-migration`), measured 2026-10-24 on this Mac
(SWI-Prolog 10.0.2, PeTTa `fea83178`, the vendored library of `vendor/pln_fork/`).

Every table is a block that `python scripts/pln_control_demo.py render` writes from `docs/petta_migration/fork_design/c/results/control_*.json`; each file was written by one command,
`python scripts/pln_control_demo.py run --part <a|m|l|seeds|legacy>`, and the block header repeats it. `tests/test_pln_number_table.py` re-renders the blocks and fails if this file
disagrees with the data. Regenerate everything with `python scripts/pln_control_demo.py run --part all` (about 5 minutes) and `render`.

## 0. The plain statement

**The shipped graph never binds the control.** The patient graph has 28 links and 30 nodes. For every source the longest run takes 12 selections (Spermidine; 16 with the feedback edge,
Omega3), never more than 6 tasks wait in the task queue, never more than 22 beliefs (26 with the edge) are in the belief queue, against the 256 / 256 the chainer starts with and a
step budget of 1,024. The smallest queue size at which nothing is cut for any source is K = 28 (measured, section 2). One derivation per connected pair means there is nothing to rank:
the answers equal the legacy chainer's whatever the queues do, and the number table (NUMBER_TABLE.md) cannot show the control. **Do not read the pln chainer's agreement with legacy on the
shipped data as evidence that the control works; it is evidence that the control is not needed there.**

What the control *does* show, on a network that makes it bind:

1. **A bounded, ordered search with an honest status.** With a step budget and queue sizes the loop stops, returns what it has, and the adapter's test says whether anything was dropped
   (`complete` or `partial`). On the large synthetic network the library reaches every one of the 20 highest-confidence states after 20 selections out of the 26,429 a complete run needs.
2. **The order is best-first by confidence, and only by confidence.** That is the library's priority. The application's own measure, strength x confidence, is reached a little later and
   a little less exactly (section 3.2): after 20 selections the top-20 states by confidence are 99 percent reached on average over ten networks, the top-20 by rank-score 91 percent.
3. **The two queues are two different things.** The task queue is the beam (width 8 is enough here); the belief queue is the memory, and it holds the knowledge base as well as the conclusions,
   so a belief queue smaller than the graph forgets the low-confidence *links* first and the search dies (section 3.2). That is where the cliff is, and it is not where one would look for it.
4. **A cycle terminates** (section 2): a derivation uses each link once, so the feedback edge costs 4 more selections for the worst source, and the legacy chainer never ends.
5. **The completeness test is sound where the decision says it is and blind where it says it is blind** (section 3.1): with equal queues no run that lost a route was flagged complete;
   with the task queue smaller than the belief queue the naive test calls 9 of 9 lossy runs complete.

What it does **not** show: a speed-up (the legacy chainer enumerates all 20,719 derivations of the large network in 0.5 s including process start; one level of 20 selections takes 0.17 s,
512 selections 3.7 s, section 4); anything about a *real* large network (none exists; nothing here is one); or that confidence-only is the right priority for the application, beyond one
synthetic family (section 3.2, "other priorities").

## 1. What is controlled, and the definitions

`PLN.Derive` takes the best task of the task queue by confidence (the first of equals), pairs it with every belief in both orders under every rule, and adds the conclusions to both
queues; each queue is then cut to its size by dropping the lowest-confidence items (a queue of size N keeps N - 1); it stops when the queue is empty or after a number of steps. Under
`PLN_CHAINER=pln` the task is the intervention alone, the beliefs are the links reachable from it, both queues are one size K, and the answer is accepted only if the run is `complete`
(contract mode: retry at 8K and 64K, then fail the request). The research lane of the adapter (`ChainSettings(task_queue=.., belief_queue=..)`) and this script's harness can set the two
sizes apart and accept a partial table. A request cannot.

* **selection**: one task taken from the task queue and expanded (one step of `PLN.Derive`). A complete run on a DAG takes 1 + (routes) + (routes that end in a Neg link) selections: the premise,
  every route, and one more for each Neg route, whose conclusion `(Not (State B))` needs a selection of its own before Not-elimination yields `(State B)`.
* **legacy paths**: the derivations the legacy chainer returns from the source (every path of any length).
* **state / route**: a state is a node with a conclusion; a route is one path to it (one stamp). A diamond gives a state several routes.
* **by confidence**: the library's priority: the best route's confidence c (the product of the link confidences, no discount). **By rank-score**: the application's: strength x confidence,
  strength the product of the link strengths, confidence c x 0.9^(hops - 1); the best route per state. Both are magnitudes.
* **recall@k**: of the k states with the best true value (computed exactly, by a dynamic programme over the DAG, not by the engine), the share the run has reached. **exact**: reached *with its
  true best route* (a state first reached through a poorer route counts for recall, not for exact).
* **flag**: complete iff tq >= bq, the task queue ended empty, and the belief queue never reached bq - 1. `unproven` marks tq < bq. The naive flag drops the tq >= bq condition.
* The synthetic generator is seeded (`random.Random(seed)`): one root, six first-layer nodes, eight layers of 26 nodes (SYNTHETIC-L) or five of 14 (SYNTHETIC-M), each node linked to three
  random nodes of the next layer, every node given a parent, 25 percent of the links Neg, strengths U(0.2, 0.9), confidences U(0.5, 0.9). **They are synthetic. They are not a model of any real
  network** and nobody should read a biological meaning into them; they exist so that the queues have something to cut.

## 2. (a) The shipped network and the feedback edge

The patient graph (28 links), and the patient graph plus `(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))` (a probe, not a curated edge). First block: the harness, every source
with per-selection resumption, so the peaks are real peaks. Second block: the same graph through the production path (`_runtime_run`, the hook, the sandbox): the legacy chainer is stopped by
the deadline, the pln chainer ends. Third block: the control starts to cut when K is small: tq = bq = K, every source, the step cap 4K.

<!-- BEGIN GENERATED control_a -->
Measured by `python scripts/pln_control_demo.py run --part a` (14.1 s, script 7097e74b988f0cd5, SWI-Prolog version 10.0.2 for arm64-darwin, PeTTa fea83178).

| graph | links | sources | pairs (node-simple) | missing | extra | max dev | false chains | max selections (source) | peak tasks | peak beliefs | tq / bq | all sources complete |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 28 | 20 | 100 of 100 | 0 | 0 | 0.0e+00 | 0 | 12 (Spermidine) | 5 | 22 | 256 / 256 | yes |
| shipped+feedback | 29 | 20 | 109 of 109 | 0 | 0 | 0.0e+00 | 0 | 16 (Omega3) | 6 | 26 | 256 / 256 | yes |

| chainer | status | pairs | missing / extra / duplicate | max dev vs node-simple reference | complete | attempts | queue K | max selections | max beliefs | derive runs | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy (PeTTa) | error | - | - | - | - | - | - | - | - | - | 10.02 s (PeTTa exceeded the 10 s deadline and was killed) |
| pln R2 | ok | 109 | 0 / 0 / 0 | 2.2e-16 | True | 1 | 256 | 16 | 26 | 40 | 1.283 s |

| K (tq = bq) | pairs found | sources flagged partial | sources that lost a pair | lost a pair but flagged complete |
|---|---|---|---|---|
| 3 | 12 of 109 | 20 | 16 | 0 |
| 4 | 19 of 109 | 20 | 16 | 0 |
| 5 | 22 of 109 | 16 | 15 | 0 |
| 6 | 23 of 109 | 16 | 15 | 0 |
| 7 | 26 of 109 | 16 | 13 | 0 |
| 8 | 26 of 109 | 14 | 13 | 0 |
| 9 | 27 of 109 | 13 | 13 | 0 |
| 10 | 31 of 109 | 13 | 12 | 0 |
| 11 | 34 of 109 | 12 | 12 | 0 |
| 12 | 39 of 109 | 12 | 11 | 0 |
| 13 | 44 of 109 | 11 | 9 | 0 |
| 14 | 47 of 109 | 9 | 9 | 0 |
| 15 | 55 of 109 | 9 | 9 | 0 |
| 16 | 61 of 109 | 9 | 9 | 0 |
| 17 | 70 of 109 | 9 | 9 | 0 |
| 18 | 79 of 109 | 9 | 9 | 0 |
| 19 | 83 of 109 | 9 | 8 | 0 |
| 20 | 91 of 109 | 9 | 6 | 0 |
| 21 | 96 of 109 | 8 | 6 | 0 |
| 22 | 101 of 109 | 8 | 6 | 0 |
| 23 | 107 of 109 | 7 | 1 | 0 |
| 24 | 108 of 109 | 3 | 1 | 0 |
| 25 | 109 of 109 | 2 | 0 | 0 |
| 28 | 109 of 109 | 0 | 0 | 0 |
| 30 | 109 of 109 | 0 | 0 | 0 |
| 32 | 109 of 109 | 0 | 0 | 0 |
| 33 | 109 of 109 | 0 | 0 | 0 |
| 34 | 109 of 109 | 0 | 0 | 0 |
| 36 | 109 of 109 | 0 | 0 | 0 |
| 40 | 109 of 109 | 0 | 0 | 0 |
| 64 | 109 of 109 | 0 | 0 | 0 |
| 256 | 109 of 109 | 0 | 0 | 0 |
<!-- END GENERATED control_a -->

Reading. With the edge the pln chainer terminates with 109 pairs, equal to the node-simple reference (0 missing, 0 extra, no pair with two results, deviation 2.2e-16 from the two-level
production run), the complete flag set for all 20 sources on the first rung (K = 256). The library also derives 16 conclusions that are walks revisiting a node (a loop through the edge); the
adapter does not answer them (the contract answers node-simple routes) and counts them separately; there is no conclusion that no walk licenses (`false chains` 0). The legacy chainer hits the
request deadline. In the K table: for K <= 24 some source loses a pair, and in every setting every source that lost a pair was flagged partial ("lost a pair but flagged
complete" is 0 throughout); K = 25 to 27 find all 109 pairs but still flag one or two sources (their belief queues reached K - 1, so something was dropped; the test cannot know that it did not matter);
K = 28 is the first size at which nothing is cut for any source. The shipped K = 256 is nine times that.

## 3. (b) SYNTHETIC networks

### 3.1 SYNTHETIC-M: is the completeness test right?

177 links, 754 legacy paths, 47 diamond nodes, small enough that a complete run exists. Contract mode (tq = bq = K) at a ladder of K, then tq smaller than bq. "Truly complete" is the ground truth:
all 754 routes present and all 55 states.

<!-- BEGIN GENERATED control_m -->
Measured by `python scripts/pln_control_demo.py run --part m` (169.3 s, script 7097e74b988f0cd5, SWI-Prolog version 10.0.2 for arm64-darwin, PeTTa fea83178).

| network | links | reachable states | legacy paths | selections a complete run needs (1 + routes + routes ending in a Neg link) | diamond nodes | Neg links |
|---|---|---|---|---|---|---|
| SYNTHETIC-M | 177 | 55 | 754 | 964 | 47 | 53 |

CONTRACT MODE, tq = bq = K, step cap 4K:

| tq | bq | step cap | selections | tasks left | beliefs | routes / legacy paths | states | flag | naive flag | truly complete |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 32 | 128 | 18 | 0 | 31 | 1 / 754 | 1 / 55 | partial | partial | no |
| 64 | 64 | 256 | 66 | 0 | 63 | 3 / 754 | 3 / 55 | partial | partial | no |
| 128 | 128 | 512 | 486 | 0 | 127 | 13 / 754 | 13 / 55 | partial | partial | no |
| 256 | 256 | 1024 | 964 | 0 | 255 | 99 / 754 | 47 / 55 | partial | partial | no |
| 512 | 512 | 2048 | 964 | 0 | 511 | 298 / 754 | 54 / 55 | partial | partial | no |
| 1024 | 1024 | 4096 | 964 | 0 | 1023 | 695 / 754 | 55 / 55 | partial | partial | no |
| 1536 | 1536 | 6144 | 964 | 0 | 1094 | 754 / 754 | 55 / 55 | complete | complete | yes |
| 2048 | 2048 | 8192 | 964 | 0 | 1094 | 754 / 754 | 55 / 55 | complete | complete | yes |

Runs flagged complete that were not truly complete: 0 of 8.

TASK QUEUE SMALLER THAN THE BELIEF QUEUE (tq < bq = 4096, step cap 4096):

| tq | bq | step cap | selections | tasks left | beliefs | routes / legacy paths | states | flag | naive flag | truly complete |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4096 | 4096 | 8 | 0 | 151 | 13 / 754 | 13 / 55 | partial | complete | no |
| 3 | 4096 | 4096 | 15 | 0 | 166 | 24 / 754 | 21 / 55 | partial | complete | no |
| 4 | 4096 | 4096 | 22 | 0 | 175 | 32 / 754 | 29 / 55 | partial | complete | no |
| 6 | 4096 | 4096 | 42 | 0 | 208 | 55 / 754 | 35 / 55 | partial | complete | no |
| 8 | 4096 | 4096 | 58 | 0 | 240 | 79 / 754 | 48 / 55 | partial | complete | no |
| 12 | 4096 | 4096 | 81 | 0 | 274 | 102 / 754 | 50 / 55 | partial | complete | no |
| 16 | 4096 | 4096 | 98 | 0 | 301 | 119 / 754 | 51 / 55 | partial | complete | no |
| 32 | 4096 | 4096 | 184 | 0 | 420 | 210 / 754 | 54 / 55 | partial | complete | no |
| 64 | 4096 | 4096 | 341 | 0 | 607 | 355 / 754 | 54 / 55 | partial | complete | no |

Runs the naive test (tasks empty and beliefs < bq - 1) calls complete although routes were lost: 9 of 9; the adapter's test calls none of them complete.
<!-- END GENERATED control_m -->

Reading. With equal queues the adapter's flag was never wrong: of 8 runs, 6 lost routes and all 6 were flagged `partial`; the two complete runs (K = 1536 and 2048) found all 754 routes in 964
selections, which is 1 + 754 + 209 (the 209 routes ending in a Neg link). Between K = 1024 (695 of 754 routes, 1023 beliefs) and K = 1536 the queue was the limit: a run needs K above its final belief
count (1,094 here), not above the number of links. With the task queue smaller than the belief queue the same "tasks empty and beliefs below K - 1" test says `complete` in 9 of 9 runs, in which
between 13 and 355 of the 754 routes were found; the adapter's rule (tq >= bq) says `partial` for all of them, which is the decision's correction 1, measured on the engine. Also note the
price of a complete table: 964 selections for 754 legacy derivations, 4.7 s for one level at K >= 1536, against the legacy chainer's milliseconds.

### 3.2 SYNTHETIC-L: 560 links, 20,719 legacy paths

<!-- BEGIN GENERATED control_l -->
Measured by `python scripts/pln_control_demo.py run --part l` (80.7 s, script 7097e74b988f0cd5, SWI-Prolog version 10.0.2 for arm64-darwin, PeTTa fea83178).

| network | links | reachable states | legacy paths | selections a complete run needs (1 + routes + routes ending in a Neg link) | diamond nodes (in-degree >= 2) | max in-degree | Neg links | links sha256 (first 16) |
|---|---|---|---|---|---|---|---|---|
| SYNTHETIC-L | 560 | 173 | 20719 | 26429 | 140 | 9 | 156 | 9ea6b9ccc03a73ec |

STEP SWEEP, tq = bq = 4096, one level, snapshots of one run:

| step cap | selections | tasks left | beliefs | routes (share of legacy paths) | states reached (share; the recall a random set of that size would have) | recall@20 / exact by confidence | recall@20 / exact by rank-score | flag | engine s (one level) |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 10 | 19 | 482 | 22 (0.1%) | 21 / 173 (12%) | 75% / 75% | 65% / 65% | partial | 0.08 |
| 20 | 20 | 39 | 512 | 43 (0.2%) | 37 / 173 (21%) | 100% / 95% | 85% / 80% | partial | 0.17 |
| 40 | 40 | 69 | 562 | 86 (0.4%) | 59 / 173 (34%) | 100% / 100% | 100% / 80% | partial | 0.28 |
| 80 | 80 | 135 | 668 | 163 (0.8%) | 79 / 173 (46%) | 100% / 100% | 100% / 85% | partial | 0.48 |
| 160 | 160 | 261 | 874 | 313 (1.5%) | 110 / 173 (64%) | 100% / 100% | 100% / 90% | partial | 0.89 |
| 320 | 320 | 498 | 1271 | 600 (2.9%) | 142 / 173 (82%) | 100% / 100% | 100% / 100% | partial | 2.10 |
| 640 | 640 | 951 | 2044 | 1160 (5.6%) | 154 / 173 (89%) | 100% / 100% | 100% / 100% | partial | 6.25 |
| 1024 | 1024 | 1485 | 2962 | 1854 (8.9%) | 164 / 173 (95%) | 100% / 100% | 100% / 100% | partial | 14.71 |

RECALL BY k at those snapshots (recall / exact):

| selections | k=5 conf | k=10 conf | k=20 conf | k=50 conf | k=5 rank-score | k=10 rank-score | k=20 rank-score | k=50 rank-score |
|---|---|---|---|---|---|---|---|---|
| 10 | 100% / 100% | 100% / 100% | 75% / 75% | 38% / 38% | 100% / 100% | 80% / 80% | 65% / 65% | 36% / 32% |
| 20 | 100% / 100% | 100% / 100% | 100% / 95% | 66% / 60% | 100% / 100% | 100% / 100% | 85% / 80% | 62% / 52% |
| 40 | 100% / 100% | 100% / 100% | 100% / 100% | 92% / 84% | 100% / 100% | 100% / 100% | 100% / 80% | 86% / 62% |
| 80 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 96% | 100% / 100% | 100% / 100% | 100% / 85% | 94% / 72% |
| 160 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 90% | 100% / 80% |
| 320 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 86% |
| 640 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 94% |
| 1024 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |

JOINT QUEUE SWEEP, tq = bq = K, step cap 512:

| tq | bq | selections | tasks left | beliefs | routes (share of legacy paths) | states reached (share; the recall a random set of that size would have) | recall@20 / exact by confidence | recall@20 / exact by rank-score | flag | engine s (one level) |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 17 | 0 | 63 | 1 (0.0%) | 1 / 173 (1%) | 5% / 5% | 5% / 5% | partial | 0.21 |
| 128 | 128 | 41 | 0 | 127 | 2 (0.0%) | 2 / 173 (1%) | 10% / 10% | 10% / 10% | partial | 0.23 |
| 256 | 256 | 512 | 142 | 255 | 8 (0.0%) | 8 / 173 (5%) | 40% / 40% | 35% / 35% | partial | 1.06 |
| 512 | 512 | 512 | 511 | 511 | 46 (0.2%) | 38 / 173 (22%) | 100% / 100% | 90% / 70% | partial | 2.61 |
| 1024 | 1024 | 512 | 775 | 1023 | 425 (2.1%) | 122 / 173 (71%) | 100% / 100% | 100% / 95% | partial | 3.97 |
| 2048 | 2048 | 512 | 775 | 1740 | 936 (4.5%) | 151 / 173 (87%) | 100% / 100% | 100% / 100% | partial | 3.69 |
| 4096 | 4096 | 512 | 775 | 1740 | 936 (4.5%) | 151 / 173 (87%) | 100% / 100% | 100% / 100% | partial | 3.71 |

TASK-QUEUE SWEEP (the beam), bq = 4096, step cap 512:

| tq | bq | selections | tasks left | beliefs | routes (share of legacy paths) | states reached (share; the recall a random set of that size would have) | recall@20 / exact by confidence | recall@20 / exact by rank-score | flag | engine s (one level) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4096 | 10 | 0 | 483 | 23 (0.1%) | 23 / 173 (13%) | 50% / 45% | 35% / 35% | partial (unproven) | 0.08 |
| 4 | 4096 | 30 | 0 | 530 | 56 (0.3%) | 49 / 173 (28%) | 75% / 75% | 50% / 50% | partial (unproven) | 0.12 |
| 8 | 4096 | 69 | 0 | 606 | 115 (0.6%) | 86 / 173 (50%) | 100% / 95% | 70% / 65% | partial (unproven) | 0.23 |
| 16 | 4096 | 172 | 0 | 794 | 256 (1.2%) | 120 / 173 (69%) | 100% / 100% | 100% / 80% | partial (unproven) | 0.67 |
| 32 | 4096 | 326 | 0 | 1103 | 482 (2.3%) | 133 / 173 (77%) | 100% / 100% | 100% / 85% | partial (unproven) | 1.70 |
| 64 | 4096 | 512 | 49 | 1534 | 799 (3.9%) | 150 / 173 (87%) | 100% / 100% | 100% / 90% | partial (unproven) | 3.57 |
| 128 | 4096 | 512 | 127 | 1716 | 940 (4.5%) | 153 / 173 (88%) | 100% / 100% | 100% / 100% | partial (unproven) | 3.60 |
| 256 | 4096 | 512 | 255 | 1740 | 936 (4.5%) | 151 / 173 (87%) | 100% / 100% | 100% / 100% | partial (unproven) | 3.71 |
| 512 | 4096 | 512 | 511 | 1740 | 936 (4.5%) | 151 / 173 (87%) | 100% / 100% | 100% / 100% | partial (unproven) | 3.88 |
| 1024 | 4096 | 512 | 775 | 1740 | 936 (4.5%) | 151 / 173 (87%) | 100% / 100% | 100% / 100% | partial (unproven) | 3.73 |

BELIEF-QUEUE SWEEP (the memory), tq = 4096, step cap 512:

| tq | bq | selections | tasks left | beliefs | routes (share of legacy paths) | states reached (share; the recall a random set of that size would have) | recall@20 / exact by confidence | recall@20 / exact by rank-score | flag | engine s (one level) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 128 | 41 | 0 | 127 | 2 (0.0%) | 2 / 173 (1%) | 10% / 10% | 10% / 10% | partial | 0.23 |
| 4096 | 256 | 512 | 142 | 255 | 8 (0.0%) | 8 / 173 (5%) | 40% / 40% | 35% / 35% | partial | 1.05 |
| 4096 | 512 | 512 | 775 | 511 | 46 (0.2%) | 38 / 173 (22%) | 100% / 100% | 90% / 70% | partial | 2.45 |
| 4096 | 1024 | 512 | 775 | 1023 | 425 (2.1%) | 122 / 173 (71%) | 100% / 100% | 100% / 95% | partial | 4.00 |
| 4096 | 2048 | 512 | 775 | 1740 | 936 (4.5%) | 151 / 173 (87%) | 100% / 100% | 100% / 100% | partial | 3.71 |

Two-level check at 256 steps (as the chainer runs it): 490 routes, max |engine strength - product of link strengths| = 3.3e-16.
<!-- END GENERATED control_l -->

Reading the step sweep (tq = bq = 4096, so nothing is cut and only the order matters). After 10 selections 21 of 173 states are reached (a random set of that size would hold 12 percent of the top 20);
after 20 selections 37 states (21 percent by chance), and the top 20 by confidence are all reached, 19 of them by their best route; the top 20 by rank-score are 85 percent reached, 80 percent exactly.
After 40 selections both top-20 sets are fully reached, the rank-score one exactly for 80 percent; 160 selections for 90 percent, 320 for 100. The top 50 by confidence are reached after 80 selections and exactly after 160; the top 50 by rank-score are reached after 160 and exactly only after 1,024
(86 percent at 320, 94 at 640). 1,024 selections have found 1,854 of the 20,719 routes (9 percent) and 164 of the 173 states. Run to the end the loop would take 26,429 selections; that run was not made.
The reason the library's confidence order is a good but not exact proxy for the application's rank-score is visible in the "exact" columns: a state's best route by strength x confidence is
sometimes not its best route by confidence alone, and the loop reaches the confident one first.

Reading the queue sweeps (step cap 512).
* **Joint queues.** The run starts with 454 beliefs (the 453 of the 560 links that are reachable from the root, and the premise). With K = 64 and K = 128 the first cut keeps only the most confident beliefs, most of the *links* go, and the search
  ends after 17 and 41 selections with 1 and 2 routes retained (tasks empty, flag partial). K = 256 keeps the 255 most confident beliefs, about half the links: 512 selections, 8 routes retained.
  K = 512 is just above the graph and recalls the top 20 by confidence fully (rank-score 90 percent, 70 percent exactly); from K = 2048 on the result is the same as K = 4096.
* **Task queue (the beam), bq = 4096.** A beam of 8 reaches all of the top 20 by confidence (95 percent exactly) with 69 selections and 115 routes; a beam of 16 reaches the top 20 by rank-score
  completely (80 percent exactly) with 172 selections; a beam of 2 or 4 reaches 50 and 75 percent. Every row is `partial (unproven)` because tq < bq: the adapter would refuse to call any of
  them complete, and the naive test would call every row with tq <= 32 complete (tasks empty) although at most 482 of 20,719 routes exist in it.
* **Belief queue (the memory), tq = 4096.** 128 and 256 forget the graph (2 and 8 routes retained, recall 10 and 40 percent); 512 is enough for the top 20 by confidence; 1024 for the top 20 by
  rank-score (95 percent exact); 2048 changes nothing more at this step cap. A belief queue is the memory of a search that has no other memory: it holds the knowledge base and the conclusions
  in one list, ranked by confidence, so when it is too small the first thing to go is the weakest *evidence*, not the weakest conclusion.
* **Two levels.** The sweeps run one level (the intervention at strength 1). The chainer runs two (1 and 0) and takes the difference; on this network at 256 selections, both levels, 490 routes: the
  difference divided by 0.98^hops equals the product of the link strengths to 3.3e-16.

#### Other priorities (a model, not a measurement of the library)

The library cannot be asked for another priority, so the loop is also written as a priority queue of routes in plain Python (`model_best_first`; the library's rules on a DAG, no queue limit). With the
library's priority (confidence) the model equals the engine at every snapshot above, in routes found and in the set of states reached. Run with the application's rank-score as the priority and with
first-in-first-out (breadth first), at the same selection counts:

<!-- BEGIN GENERATED control_l_model -->
Measured by `python scripts/pln_control_demo.py run --part l` (80.7 s, script 7097e74b988f0cd5, SWI-Prolog version 10.0.2 for arm64-darwin, PeTTa fea83178).

Model check: the `confidence` model equals the engine at all 8 snapshots of the step sweep in routes found (True) and in the set of states reached (True).

| selections | priority | routes found | states reached | recall@20 / exact by confidence | recall@20 / exact by rank-score |
|---|---|---|---|---|---|
| 10 | confidence | 22 | 21 / 173 | 75% / 75% | 65% / 65% |
| 10 | rank-score | 23 | 21 / 173 | 55% / 45% | 80% / 80% |
| 10 | fifo | 20 | 17 / 173 | 55% / 50% | 60% / 60% |
| 20 | confidence | 43 | 37 / 173 | 100% / 95% | 85% / 80% |
| 20 | rank-score | 43 | 37 / 173 | 90% / 90% | 100% / 100% |
| 20 | fifo | 39 | 32 / 173 | 85% / 60% | 95% / 80% |
| 40 | confidence | 86 | 59 / 173 | 100% / 100% | 100% / 80% |
| 40 | rank-score | 81 | 58 / 173 | 100% / 90% | 100% / 100% |
| 40 | fifo | 75 | 53 / 173 | 100% / 85% | 100% / 90% |
| 80 | confidence | 163 | 79 / 173 | 100% / 100% | 100% / 85% |
| 80 | rank-score | 156 | 76 / 173 | 100% / 100% | 100% / 100% |
| 80 | fifo | 151 | 73 / 173 | 100% / 95% | 100% / 100% |
| 160 | confidence | 313 | 110 / 173 | 100% / 100% | 100% / 90% |
| 160 | rank-score | 303 | 95 / 173 | 100% / 100% | 100% / 100% |
| 160 | fifo | 314 | 89 / 173 | 100% / 100% | 100% / 100% |
| 320 | confidence | 600 | 142 / 173 | 100% / 100% | 100% / 100% |
| 320 | rank-score | 593 | 126 / 173 | 100% / 100% | 100% / 100% |
| 320 | fifo | 600 | 114 / 173 | 100% / 100% | 100% / 100% |
| 640 | confidence | 1160 | 154 / 173 | 100% / 100% | 100% / 100% |
| 640 | rank-score | 1219 | 149 / 173 | 100% / 100% | 100% / 100% |
| 640 | fifo | 1180 | 143 / 173 | 100% / 100% | 100% / 100% |
| 1024 | confidence | 1854 | 164 / 173 | 100% / 100% | 100% / 100% |
| 1024 | rank-score | 1972 | 164 / 173 | 100% / 100% | 100% / 100% |
| 1024 | fifo | 1864 | 144 / 173 | 100% / 100% | 100% / 100% |
<!-- END GENERATED control_l_model -->

Reading. Whichever measure the queue is ordered by is the one that is reached first: at 20 selections, on this network, the confidence order has the top 20 by confidence at 100 percent (95 exactly) and the top 20 by
rank-score at 85 (80); the rank-score order has 90 (90) and 100 (100). Over the ten networks of the next table the rank-score order has the top 20 by rank-score at 96 percent (95 exactly) and the top 20 by confidence at 88 (84). Over the ten networks of the next table, breadth first at 20 selections reaches 86 percent of the top 20 by confidence (76 exactly) against 99 (98) for the confidence order, and 92 percent
of the top 20 by rank-score (80 exactly) against 91 (87); it catches up by 80 selections, because on this network the best states are the shallow ones: confidences and strengths are products of numbers below one. The gap between best-first and breadth-first is therefore *modest and early*
here; the queue sizes of the previous tables matter more. How large it is on a real network depends on how long the confident chains are.

#### Ten more instances of the recipe

The step-sweep numbers are for one network. Ten more seeds of the same recipe (the engine at tq = bq = 4096 and the three model priorities; "model equals engine" is checked per seed):

<!-- BEGIN GENERATED control_seeds -->
Measured by `python scripts/pln_control_demo.py run --part seeds` (8.6 s, script 7097e74b988f0cd5, SWI-Prolog version 10.0.2 for arm64-darwin, PeTTa fea83178).

| seed | legacy paths | reachable states | 20 selections: conf recall / exact; rank-score recall / exact | 40 selections: conf recall / exact; rank-score recall / exact | 80 selections: conf recall / exact; rank-score recall / exact | model equals engine |
|---|---|---|---|---|---|---|
| 1 | 20990 | 168 | 95% / 90%; 85% / 85% | 100% / 100%; 90% / 90% | 100% / 100%; 100% / 100% | yes |
| 2 | 20928 | 169 | 100% / 95%; 95% / 90% | 100% / 100%; 95% / 95% | 100% / 100%; 100% / 95% | yes |
| 3 | 20761 | 168 | 100% / 100%; 90% / 80% | 100% / 100%; 100% / 90% | 100% / 100%; 100% / 95% | yes |
| 4 | 20580 | 169 | 100% / 100%; 85% / 80% | 100% / 100%; 100% / 95% | 100% / 100%; 100% / 95% | yes |
| 5 | 22146 | 174 | 100% / 100%; 80% / 80% | 100% / 100%; 85% / 85% | 100% / 100%; 100% / 95% | yes |
| 6 | 20929 | 170 | 100% / 100%; 95% / 90% | 100% / 100%; 100% / 100% | 100% / 100%; 100% / 100% | yes |
| 7 | 20719 | 173 | 100% / 95%; 85% / 80% | 100% / 100%; 100% / 80% | 100% / 100%; 100% / 85% | yes |
| 8 | 22421 | 174 | 100% / 100%; 100% / 100% | 100% / 100%; 100% / 100% | 100% / 100%; 100% / 100% | yes |
| 9 | 19999 | 172 | 100% / 100%; 100% / 100% | 100% / 100%; 100% / 100% | 100% / 100%; 100% / 100% | yes |
| 10 | 21594 | 172 | 100% / 100%; 95% / 85% | 100% / 100%; 100% / 100% | 100% / 100%; 100% / 100% | yes |

Ten instances, three priorities (the model; `confidence` is the library's):

| priority (model) | selections | recall@20 by confidence, mean (min) | exact by confidence | recall@20 by rank-score | exact by rank-score |
|---|---|---|---|---|---|
| confidence | 20 | 99% (min 95%) | 98% (min 90%) | 91% (min 80%) | 87% (min 80%) |
| confidence | 40 | 100% (min 100%) | 100% (min 100%) | 97% (min 85%) | 94% (min 80%) |
| confidence | 80 | 100% (min 100%) | 100% (min 100%) | 100% (min 100%) | 97% (min 85%) |
| confidence | 160 | 100% (min 100%) | 100% (min 100%) | 100% (min 100%) | 99% (min 90%) |
| rank-score | 20 | 88% (min 70%) | 84% (min 65%) | 96% (min 90%) | 95% (min 85%) |
| rank-score | 40 | 96% (min 90%) | 90% (min 80%) | 100% (min 100%) | 100% (min 100%) |
| rank-score | 80 | 99% (min 95%) | 94% (min 85%) | 100% (min 100%) | 100% (min 100%) |
| rank-score | 160 | 100% (min 100%) | 96% (min 85%) | 100% (min 100%) | 100% (min 100%) |
| fifo | 20 | 86% (min 70%) | 76% (min 55%) | 92% (min 85%) | 80% (min 75%) |
| fifo | 40 | 96% (min 85%) | 89% (min 70%) | 98% (min 85%) | 91% (min 80%) |
| fifo | 80 | 100% (min 100%) | 94% (min 80%) | 100% (min 100%) | 98% (min 95%) |
| fifo | 160 | 100% (min 100%) | 98% (min 85%) | 100% (min 100%) | 100% (min 100%) |
<!-- END GENERATED control_seeds -->

Reading. The early recall is not luck: at 20 selections the top 20 by confidence are reached for 99 percent on average over the ten networks (minimum 95 percent), exactly for 98 percent; by rank-score 91
percent (minimum 80) and 87 percent exactly. By 40 selections every network has all of its top 20 by confidence, and by 80 all of its top 20 by rank-score are reached. The model equals the engine in all ten.

## 4. (c) Wall time

All times are the engine's for ONE level (the chainer pays two) on this Mac, SWI-Prolog 10.0.2, measured with nothing else running; the Python parse of the returned state is not included.

| what | measured |
|---|---|
| the shipped patient graph, 20 sources x 2 levels = 40 runs, K = 256 | 0.78 s of `PLN.Derive` in a 1.28 s request (production path, feedback graph); the rest is process start and knowledge-base load |
| SYNTHETIC-L, 20 selections / 160 / 1,024 at K = 4096 | 0.17 s / 0.89 s / 14.7 s (8, 5.5 and 14 ms per selection; the cost per selection grows with the beliefs: 482, 874 and 2,962) |
| SYNTHETIC-L, 512 selections, queues from 512 to 4096 | 2.4 s to 4.0 s |
| SYNTHETIC-L, beam of 8 (the top 20 by confidence) | 0.23 s, 69 selections |
| SYNTHETIC-M, a complete table (K >= 1536) | 4.7 s, 964 selections |

<!-- BEGIN GENERATED control_legacy -->
Measured by `python scripts/pln_control_demo.py run --part legacy` (2.4 s, script 7097e74b988f0cd5, SWI-Prolog version 10.0.2 for arm64-darwin, PeTTa fea83178).

| what | measured |
|---|---|
| legacy chainer, every path of SYNTHETIC-L (PeTTa, whole request incl. start and load) | 20719 derivations in 0.502 s (median of 3: [0.499, 0.521, 0.502]); start and load alone 0.21 s |
| contract mode on SYNTHETIC-L: the route-count estimate of beliefs | 41892 (largest queue size of the schedule: 16384; first attempt would be K = 16384) |
| contract mode on SYNTHETIC-L under a 30 s request deadline | error after 0.2 s: the PLN chainer refused the program: the derivation from SynRoot cannot be done: more than 16,384 distinct walks over 453 reachable links (each link used once) do not fit the largest queue (K=16,384); |
<!-- END GENERATED control_legacy -->

Reading. For the exhaustive question the legacy chainer is far cheaper than the loop: 20,719 derivations in a 0.5 s request, 0.2 s of which is process start and load. The loop's cost is quadratic in the
beliefs (each selection is paired with every belief), and a complete table needs a belief queue above the number of conclusions it keeps. Contract mode, which refuses a partial table, therefore cannot
serve this network: the estimate of the adapter (41,892 beliefs; the library derives every walk that uses each link once, and on this acyclic network that is the 20,719 routes) is above the largest queue size of its
schedule (16,384). The first build ran the request on into the 30 s deadline kill; the request is now refused up front, in 0.2 s, by the pre-flight of `plan_sources`, with the number of walks and the links in the
message (HTTP 422). That is the intended behaviour (a request must not run on), and it is the reason a large real network would use the control lane (partial, flagged, never refused) or a smaller question (a source and
a target, not the whole table).

## 5. What I would take from this, and what I would not

* The shipped graph does not exercise the control; the pln chainer there is a different arithmetic route to the legacy answers. The first things the control shows are on a network that does not exist yet.
* On networks shaped like these, the library's order finds the confident states fast and the loop is cheap for a bounded question and expensive for an exhaustive one.
* The three numbers an operator would have to choose for a large network are the belief queue (above the knowledge base plus the conclusions to be kept: this is the one that fails silently into `partial`),
  the beam (small, 8 to 16), and the step budget. Contract mode chooses none of them: it demands completeness.
* I would not conclude that confidence-only is right or wrong for the application. On the synthetic family it is close to the application's measure early and exact for 87 to 98 percent of the top 20
  after 20 selections; on a network whose strongest effects sit behind low-confidence links it would not be.

## 6. Questions this puts to the owner and to Ben

1. **Priority.** Is confidence-only the control Ben means, or should the application supply strength x confidence (or another function)? The table above prices the choice on a synthetic family:
   a few points of recall at 10 to 40 selections in either direction, nothing after 80.
2. **The belief queue holds the knowledge base.** Is it intended that a belief queue smaller than the graph forgets the weakest links first? For a large curated network that is a hazard the operator must
   know about; the adapter avoids it by starting K above the sub-graph and by refusing a partial table.
3. **Which real network** should the control be shown on? Candidates in the decision: CellAge/GenAge gene -> senescence links, hallmark intervention evidence. Each needs curation rulings; none is chosen here.
4. **Cycles**: allow a feedback edge in curated data (the chainer ends, 4 more selections, 9 new pairs on this probe) or refuse at write time?
