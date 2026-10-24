# Design 1 — "state sentences with Modus Ponens" (`mp_state`)

Seam between the legacy chainer (`infer` / `explain`, `pln_deduction.metta`) and patham9's PLN (the fork, `patham9/PLN` 55f1751, in the form of PeTTa's bundled `lib_pln.metta`).
Written 2026-10-24 by one of three independent designers. Read-only on the repo and on `.petta/`; nothing from `.petta/` was executed.

**Evidence tags.** `[sim]` = `fork_sim.py` (the validated port; my copy is byte-identical to `sim/fork_sim.py`), run by `run_numbers.py`.
`[hand]` = `hand_check.py`, independent arithmetic with no simulator. `[S]` = upstream source read, file:line. `[unknown]` = not established.
The MeTTa port of this design has **not been run** (no PeTTa execution was allowed); every claim about the real runtime is `[S]` or `[unknown]`.
Folder: `/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity/536046cf-1647-43f4-b3c4-093bc03cc2c3/scratchpad/forkdesign/design/mp_state/` (file list in section 18).

---------------------------------------------------------------------------------------------------------------------------

## 0. Verdict in ten lines

1. **It works as a seam.** Each node is a state sentence `(State X)`, each link an `Implication`, an intervention is the task `(State A) (stv 1.0 1.0)`. Only `ModusPonens` and `NotElimination` ever fire in the chain loop.
   With the two named constants of sections 2 and 4 (the 0.02 compensation and the 0.9 discount), all **100 patient and 86 LinAge2 ordered pairs** answer with exactly one route each, with the legacy sign, the legacy node path (`explain`), strength within **1.7e-16** and confidence within **2.8e-17** of the legacy value `[sim]`. No pair outside the legacy closure answers.
2. **Strength must be redefined, and can be recovered exactly.** A state's strength is a probability-like level, not an effect size. The legacy effect magnitude is recovered as a *risk difference*: run the same derivation twice, with the intervention applied (level 1) and absent (level 0), and subtract. A chain of Modus Ponens steps is an affine map; its slope is the signed product of `(stored strength - 0.02)`. Storing `p + 0.02` makes that product the legacy product `p1*p2*...` (section 2).
3. **The level alone cannot carry the sign** (a harmful 3-link chain ends at 0.367, a protective 4-link one at 0.138; both below 0.5). The sign comes from the structure (parity of Neg links, from the stamp) and is cross-checked against the sign of the difference on every derivation (0 mismatches).
4. **The fork's native confidence (`c1*c2`, no discount) breaks one pinned order.** With native confidence the personalised order for Patient001 flips (Spermidine 0.1041 passes Berberine 0.0948); diagnose `c(CS)` moves 0.6253 -> 0.6961, outside its 1e-2 tolerance. The 0.9 chain discount is load-bearing. The design keeps it as an adapter shim `c * 0.9^(hops-1)` (hop count read from the stamp); a switch exposes the native number.
5. **No node base rates are needed** (no `STV` call exists in the restricted rule set). The fork's constant 0.02 inside `Truth_ModusPonens` (`Formulas.metta:46-48`) is the implicit base rate; the second run cancels it.
6. **Revision does not run in the chain loop** (every derived sentence carries the premise id, and the premise has confidence 1.0). It is used as the fork's *formula* in a pre-pass that folds the rows of one link/direction. Folding 5 DrugAge-style rows gives 2 results (Neg 0.2631 / 0.8545, Pos 0.6 / 0.45) instead of 5; 170 rows give 2 (section 10).
7. **Cycles terminate** by construction: link ids are in the stamp, so a derivation uses each link once. The feedback edge `ChronicInflammation -> CellularSenescence` changes no existing answer and needs 12 selections for D+Q, as before (section 9).
8. **The full rule set is a different system.** With inversion/induction/abduction/deduction on (uniform base rate 0.5/0.9) 64 ordered pairs appear outside the legacy closure, from 19 of 30 sources (`CRP -> DNAmGDF15` among them), and D+Q -> CHD moves to 0.2700 / 0.3591 (section 6).
9. **The honest limit.** On a single-parent chain this design is the legacy product re-derived by the fork's own rule; the reasoner adds termination, route/stamp provenance, row revision and a place to put base rates later, not new inferences. Two runs per source, two named constants (the 0.02 compensation and the 0.9 discount shim), and an adapter that reads `PLN.Derive`'s belief set instead of `PLN.Query`'s single answer are the price.
10. **Cost is small on the shipped graphs** (mean 84 rule attempts per run, max 12 selections; section 12) and grows quadratically in the number of reachable links; a 510-link tree costs 1.55 M attempts per run `[sim counts; PeTTa wall time unknown]`.

---------------------------------------------------------------------------------------------------------------------------

## 1. The encoding (generic route: stamped dynamic beliefs, no Translator)

| thing | sentence | note |
|---|---|---|
| node X | term `(State X)` | "X is elevated/active". `State` and `End` are not PeTTa builtins `[S]` (`Predicate` is: `metta.pl:336,462`, so the fork's `Evaluation (Predicate ..)` shapes are avoided). |
| link `A -Pos-> B` | `(Sentence ((Implication (State A) (State B)) (stv q c)) (id))` | `q = p + 0.02` capped at 1.0 (section 2). `p` and `c` are the curated strength and the *evaluated* tier confidence. |
| link `A -Neg-> B` | `(Sentence ((Implication (State A) (Not (State B))) (stv q c)) (id))` | `MP` derives `(Not (State B))`; the fork's rule `(\|- ((Not $A) $T)) ($A (Truth_Negation $T))` (`Rules.metta:77`) then derives `(State B)` with `1 - s`, same stamp. |
| hazard-ratio record | same, sign `Neg` iff HR < 1, `p` = bucket 0.5/0.6/0.7, consequent `(End C)` | `(End C)` is nobody's antecedent, so a chain may end with a record but not continue through one (legacy "records are terminal", doc 7.3). Verified: with an extra `CHD -> Stroke` link only `FastingGlucose` (curated Effect into CHD) reaches Stroke. The adapter reads `(State C)` and `(End C)` as answers for C. |
| intervention A applied | task `(Sentence ((State A) (stv 1.0 1.0)) (0))` | the only task; id 0. The reference run uses `(stv 0.0 1.0)`. |
| evidence id | one integer per source record (curated Effect atom, HR record, DrugAge row); a folded group carries the union of its rows' ids | side table id -> (kind, file:line / record / row). |
| DrugAge row | lifted as in `drugage_calibration.metta:139-213` (strength `|pct|/(|pct|+20)`, confidence tier x gate, sign by `pct`), then **emitted straight onto `Mortality`** with the sign flipped and one *virtual hop* | see section 10 for why the definitional link `Lifespan -> Mortality (1.0 1.0)` is folded into the lift. |

**What a Neg link derives (established from the facts report, then run).** The fork's Translator turns `A -> (Not B)` into an engraved rule whose derived *term* is `B`, never `(Not B)`, with truth `Neg(MP(tvA, T))` (`Translator.metta:35`; `facts` C), and a negated antecedent matches the positive term and flips its strength. On the generic route the same numbers arrive in two steps: `MP` derives `(Not (State B))` with `MP(tvA, T)`, then `NotElimination` derives `(State B)` with `1 - s` and the same stamp. So polarity is carried by the strength alone: B's state falls when A is applied and rises (to ~0.98) when A is absent, and the confidence is `c1*c2` either way. The cost of the generic route is one extra selection per Neg hop (the `(Not ..)` sentence is a dead-end task).

Why the generic route and not the Translator: the Translator needs `PLN.Init`, hyperon + metta-morph, a 30 s compile and hard-coded `~/PLN` paths (`facts` A2), its engraved implications add **no id to the stamp** and it cannot run on PeTTa (`compat` section 1.3). Engraved and generic routes give identical `(s,c)` on all 100 patient pairs (max difference 0.0 `[sim]`), but the engraved stamps are always `(0)`, so `explain` could not be rebuilt and a cycle is traversed again and again until the budget ends: 20 selections, 3 distinct `CellularSenescence` states `[sim]`.

The KB is built **per call from the live space** (`Effect` atoms, record triples, DrugAge rows), so caller `extra_atoms` work without a precompile and nothing survives between requests.

---------------------------------------------------------------------------------------------------------------------------

## 2. What "strength" means now, and the algebra

Let `e = 0.02` (read at load from `Truth_ModusPonens (stv 0 1) (stv 0 1)`, never typed; `Formulas.metta:46-48`: `s = f1*f2 + 0.02*(1 - f1)`, `c = c1*c2`).
For a link with stored strength `q` and a source state `x`:

* Pos: `B = q*x + e*(1-x)`  i.e. `P(B|A) = q`, `P(B|not A) = e`.
* Neg: `B = 1 - (q*x + e*(1-x))`  i.e. `P(B|A) = 1-q`, `P(B|not A) = 1-e`.

Each step is an affine map of `x` with slope `+(q-e)` (Pos) or `-(q-e)` (Neg). A composition of affine maps is affine, so for any chain the final state is `alpha + beta*x_A` with
`beta = prod(sign_i * (q_i - e))`. The risk difference of the whole chain, `state(C | A=1) - state(C | A=0)`, **is** `beta`. `[sim]` 300 random chains (1-7 links, random signs): `|two-run difference - beta|` max 1.9e-16 (raw) and 2.5e-16 (compensated).

* **Raw** (`q = p`): `beta = prod(sign*(p-0.02))`, a shrunk legacy product (every hop loses `0.02/p`).
* **Compensated** (`q = p + 0.02`, capped at 1.0): `beta = prod(sign*p)`, the legacy product. The curated number is then exactly a *risk difference* `P(B|A) - P(B|not A)` under the fork's fixed leak. The legacy contract "strength is a plain product, no base rates" is this chain rule. Largest curated `p` is 0.95, so the cap never binds on shipped data; it binds only on the definitional link `(1.0, 1.0)` (section 10).

Hand check `[hand]` D+Q -> CHD (links Neg 0.70, Pos 0.85, Pos 0.70, Pos 0.60):

| node | level, D+Q applied | level, D+Q absent | difference |
|---|---|---|---|
| CellularSenescence | 0.30000 | 0.98000 | -0.68000 |
| SASP | 0.26900 | 0.83340 | -0.56440 |
| DNAmPAI1 | 0.20292 | 0.58671 | -0.38379 |
| CoronaryHeartDisease | 0.13769 | 0.36029 | **-0.222599** = -(0.68)(0.83)(0.68)(0.58) (legacy 0.2499 = 0.70*0.85*0.70*0.60) |

Compensated: levels 0.14036 / 0.39026, difference **-0.2499000000**.

Consequences:
* The *levels* are not probabilities of anything real: "absent" sources push Pos targets to ~0.02-0.06 and Neg targets to ~0.98 (the fork's leak). Only differences are meaningful; the adapter never reports a level.
* An absent source is not "unknown". `MP((0,c),T)` concludes `B = 0.02` **with confidence c1*c2**: the fork treats "A false" as evidence that B is false. The reference run uses this on purpose and cancels it by subtraction; nothing else in the design may read the level-0 conclusions.
* A null link (`p = 0`) has raw slope **-0.02** (mildly protective, wrong); compensated slope exactly 0 `[sim]`.
* Neg x Neg: `A -Neg(0.6)-> B -Neg(0.5)-> C` gives +0.2784 raw, +0.3000 compensated, sign Pos, levels 0.79 / 0.49 `[sim]`.
* No parallel-path semantics: a node with two reachable parents has two routes, each its own result (as legacy), not a noisy-OR.

---------------------------------------------------------------------------------------------------------------------------

## 3. Posing the intervention and recovering the signed magnitude

For a source A: (1) keep only the links whose antecedent is reachable from A (BFS; record links are followed, not expanded); (2) build the sentences; (3) run `PLN.Derive` twice with the single task `(State A)` at level 1.0 and at level 0.0 (both with confidence 1.0, id 0); (4) for every belief `(State C)` / `(End C)` of the level-1 run whose stamp rebuilds a *node-simple* path A..C, find the level-0 belief with the **same stamp** (identical in both runs: confidence does not depend on level; 5 random link permutations and candidate-order reversal give bit-identical results `[sim]`);
(5) report `s = |s1 - s0|`, `sign = parity of Neg links on the path` (the legacy rule; cross-checked against `sign(s1 - s0)`, which agrees on all 186 closure derivations and is undefined when `|s1 - s0| < 1e-12`, e.g. a null row, which therefore stays `Neg 0.0` as in legacy), `c` per section 4.

The answer to *"can signed effect magnitudes be recovered or must they be redefined?"*: **both.** The reasoner's strength is redefined (a level); the legacy magnitude is recovered exactly as the risk difference, at the price of one extra derivation per source and one named constant (the compensation).
`PLN.Derive` returns all beliefs once per source; the adapter caches the table per source, so `infer A C1`, `infer A C2`, ... cost one pair of runs.

---------------------------------------------------------------------------------------------------------------------------

## 4. Confidence: `c1*c2` and the 0.9 discount

Fork: `c = c1*c2`, no discount, so n edges give `c0*prod(c_i)` with `c0 = 1.0`. Legacy: `(c1*c2)*0.9` per *combination*, so n edges give `prod(c_i)*0.9^(n-1)`.
Native / legacy = `0.9^-(n-1)` = 1.000, 1.111, 1.235, 1.372, 1.524 for n = 1..5 `[hand]`. The legacy **MUST** "confidence decays with every edge and is below the suffix chain's" holds natively too (`c_first*c_suffix < c_suffix`).

| variant | what it is | s vs legacy | c vs legacy |
|---|---|---|---|
| **R0 raw** | the fork read literally: `q = p`, native c | all 100 pairs change, ratio 0.8627 .. 0.9765 (1 hop 0.949-0.977, 5 hops 0.863-0.876) | 72 of 100 change, up to x1.524 |
| **R1 compensated, native c** | `q = p + 0.02` | identical (<= 1.7e-16) | same as R0 |
| **R2 compensated + shim** (recommended) | R1 and `c := c * 0.9^(hops-1)`, `hops` = links in the stamp (+ virtual hops) | identical | identical (<= 2.8e-17) |

Three ways to keep the discount: **(S)** the shim above (chosen: explicit, one named constant read from the same `chain-discount` text the Python twin already parses, switchable `CHAIN_CONFIDENCE=legacy|native`); **(K)** a per-source KB in which only the links leaving A keep their confidence and every other link is stored as `0.9*c` (exact, but the KB then depends on the source and a cycle back through A would mis-discount); **(B)** bake `0.9*c` into every link (uniform x0.9 of legacy for every n: preserves all orders, changes every pinned c including one-hop answers: `c(MD) 0.65 -> 0.585`). The 0.9 is not part of the reasoner; the owner and Ben decide whether the discount is a PLN concept or a legacy artefact. R1 shows what the native choice costs (section 11).

---------------------------------------------------------------------------------------------------------------------------

## 5. What the 0.02 floor does to long chains

`[sim]` p = 0.7, confidence irrelevant, raw encoding:

| links | level, applied | level, absent | difference | legacy 0.7^n | ratio |
|---|---|---|---|---|---|
| 1 | 0.70000 | 0.02000 | 0.68000 | 0.70000 | 0.9714 |
| 2 | 0.49600 | 0.03360 | 0.46240 | 0.49000 | 0.9437 |
| 3 | 0.35728 | 0.04285 | 0.31443 | 0.34300 | 0.9167 |
| 4 | 0.26295 | 0.04914 | 0.21381 | 0.24010 | 0.8905 |
| 5 | 0.19881 | 0.05341 | 0.14539 | 0.16807 | 0.8651 |
| 8 | 0.10536 | 0.05964 | 0.04572 | 0.05765 | 0.7930 |

* The ratio is `((p-e)/p)^n`; the absent-source level converges to `e/(1-(p-e)) = 0.0625` (not 0): long Pos chains never fall to zero, they fall to a floor that grows with chain length.
* Confidence is untouched by the floor (`c1*c2` only).
* On the shipped graph (max 5 links) the raw shrink of the strength is between 2.4% and 13.7% across the 100 pairs (1 hop 2.4-5.1%, 5 hops 12.4-13.7%); compensation removes it entirely.
* A weak antecedent yields a *confident* low conclusion, not an unknown one (previous section). A Neg link makes the absent-source level ~0.98.

---------------------------------------------------------------------------------------------------------------------------

## 6. Rule set, and what actually executes

Restricted (default): `ModusPonens` (`Rules.metta:9-11`), `NotElimination` (`:77`), `Revision` (`:4-6`). Nothing calls `STV`, so **no node base rate is needed** (`facts` B: Revision, MP, Not elimination and every engraved rule never touch `STV`).
Excluded: Deduction, Induction, Abduction, both Inversions, Equivalence->Implication x2, TransitiveSimilarity, EvaluationImplication, MemberDeduction, Symmetric MP, the 3 Evaluation-Inheritance rules. In the vendored copy each carries a guard `(PLN.RuleOn <Name>)` (an `(empty)` fallback); the comparison switch turns them on.

* **`Revision` is inert in the chain loop**: removing it changes no belief set on any of the 30 patient sources `[sim]`. Every derived sentence carries id 0, so `StampDisjoint` refuses to revise two derivations, and the premise (confidence 1.0) cannot be revised at all (`c2w(1.0)` -> `/safe` -> no result). Revision is used as the fork's *formula* `Truth_Revision` (`Formulas.metta:66-73`) in the row fold of section 10.
* **Why the premise has id 0 and confidence 1.0, not an empty stamp.** An empty stamp makes the premise stamp-disjoint from itself; with `c0 < 1` it is revised with itself, again and again: with `c0 = 0.9` and stamp `()` the run produces 5,294 derivations, 199 copies of the premise, and the answer is lost `[sim]`. Id 0 avoids it. (The price: derivations from one source can never be pooled by the loop; see section 10.)
* **Full comparison switch** `[sim]` (inversion, induction, abduction, deduction on; uniform `STV 0.5/0.9` for every `(State X)`; Tasks = Beliefs = KB; 150 steps, tq 100, bq 400):
  * **64 ordered pairs outside the legacy closure, from 19 of 30 sources** (e.g. `CRP -> DNAmGDF15`, `D+Q -> MitochondrialDysfunction`, `CHD -> HbA1c`, `MitochondrialDysfunction -> SASP`), and 27 legacy pairs not found within that budget. The named false chains `D+Q -> AllCauseMortality`, `SmokingCessation -> CHD`, `{ChronicInflammation, MitochondrialDysfunction, SmokingPackYears} -> CHD` stayed empty in this run, but the contract is the closure, not those examples.
  * Numbers move: D+Q -> CHD Neg 0.2700 / 0.3591 (restricted R2: 0.2499 / 0.1702), CS -> CHD Pos 0.3857 / 0.5525 (0.357 / 0.2909), Metformin -> CHD Neg 0.3936 / 0.5100 (0.330 / 0.2685); the best-confidence belief wins and Deduction's `min` confidence outranks the MP product.
  * At the default 20/20/200 the same run finds 0 of the D+Q chain (budget).
* The newer trueagi rewrite changes the MP confidence to `(s1*s2)*(c1*c2)` (`TRUEAGI:124-126`); the shim and section 4 assume the fork's `c1*c2`. A load-time probe asserts `MP((0.5,0.8),(0.5,0.8)).c == 0.64` and `.s - 0.5*0.5 == 0.02*0.5`.

---------------------------------------------------------------------------------------------------------------------------

## 7. Control settings

* Call `PLN.Derive tasks beliefs steps tq bq` with **Tasks = (applied)** and **Beliefs = links + applied**, and read the final belief list. Do **not** use `PLN.Query kb term`: it takes Tasks = Beliefs (the implications become tasks, 41 selections to finish) and returns one answer per term.
  `[sim]` closure completeness (100 patient pairs): Tasks = (applied): 100/100 at the fork defaults 20/20/200, 100/100 at the PeTTa-lib defaults 100/10/100, and identical belief sets for all 30 sources under both. Tasks = KB: **52/100** at 20/20/200 (`Config.metta:4,7,10`), **66/100** at 100/10/100, 100/100 only from 60/40/200.
* Minimal settings with Tasks = (applied) (others generous): patient steps 7, tq 3, bq 41; LinAge2 steps 7, tq 3, bq 40. Longest run: 12 selections. Selection is by confidence (`Deriver.metta:37`), so the premise (1.0) goes first and children follow in confidence order.
* Rule of thumb on the pre-filtered sub-KB (N nodes, L links): `steps = 2N + 2` (each Neg hop also creates a `(Not ..)` task), `tq = max(20, N + 2)`, `bq = L + 2N + 10`. `LimitSize` keeps N-1 items (`Deriver.metta:27-31`), so a belief bound below the KB size silently drops the lowest-confidence links. The loop ends early when the task queue is empty, so the generous values cost nothing on the shipped graphs.
* Pre-filtering the KB to the links reachable from A cuts the rule attempts over all 60 patient runs from 29,568 to 5,048 and changes no answer `[sim]`.

---------------------------------------------------------------------------------------------------------------------------

## 8. Evidence ids, stamps, provenance, `explain`

* Ids: deterministic integers (links sorted by from, to, sign, source key), 0 for the premise. A stamp is the sorted set of ids used, so it names the *set of links* of a derivation; the path is rebuilt by following those links from A (all 100 + 86 paths equal the legacy `Chain` lists `[sim]`). `explain` and `infer` read one table, so `explain`'s signed tv equals `infer`'s by construction.
* Engraved implications would add no id (`facts` A3.4): the reason for the generic route.
* The side table maps id -> (kind, file:line, record id / DrugAge row id, evaluated tier). `(Chain nodes (signed ..))` can append the ids; the legacy shape is unchanged.
* A folded link's stamp is the union of its rows' ids. The same source record must carry the same id wherever it appears (a raw DrugAge row and a curated Effect that restate one study: same id); the fold counts an id once, so a restated claim is not counted twice.
* `explain` now also answers DrugAge compounds (legacy: empty). A harmless extension; flagged because the legacy behaviour is "empty".

---------------------------------------------------------------------------------------------------------------------------

## 9. Feedback edges, self-pairs, route multiplicity

`[sim]` graph + `Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65)` (closes CS -> SASP -> CI -> CS):

| source | selections | pairs before / after | changed answers | new pairs |
|---|---|---|---|---|
| D+Q | 12 | 9 / 9 | none | none |
| CellularSenescence | 10 | 8 / 8 | none | none |
| Omega3 | 16 | 5 / 9 | none | CS, SASP, DNAmPAI1, CHD (genuinely new reachability) |
| SASP | 12 | 7 / 8 | none | CS |

* Termination: a derivation uses each link id at most once (`StampDisjoint`), so the loop is finite even before the step budget. Derivations that revisit a node (`CS'`) exist in the belief set but are dropped by the node-simple filter; they cannot reach any pair the direct route does not.
* Self-loop `A -> A` plus `A -> B`: `infer A B` = one result, `infer A A` = empty. `infer A A` is empty by rule (the premise itself is a `(State A)` belief and must not answer).
* Engraved route + cycle: 20 selections (budget), 3 distinct CS states: it terminates only because the budget ends.
* Routes: a graph with diamonds yields one result per distinct simple route, as legacy; there is **no pooling across routes** (revision of routes would average two mechanisms instead of adding their effects). The shipped graphs have none. A layered DAG (10 x 40, fan-out 3) has exponentially many routes and exhausts any budget (802 selections, 161 reached) `[sim]`: a route cap is a risk for large networks.

---------------------------------------------------------------------------------------------------------------------------

## 10. Multi-row evidence (DrugAge-style)

Rows of one compound and one direction are evidence for **one** link. They are folded before the loop with the fork's `Truth_Revision` (`w = c/(1-c)`, `f = sum(w f)/sum(w)`, `c = w/(w+1)`, left fold; commutative up to rounding). Confidence is capped at 0.999999 before folding (`Truth_Revision` gives *no result* for `c = 1.0`). A folded link is one sentence whose stamp is the union of its row ids.

Direction: rows with `pct >= 0` are `Pos` at Lifespan, hence `Neg` at Mortality (the "sign trap"); `pct < 0` are `Pos` at Mortality. The null row (`pct = 0`) belongs to the extending group with strength 0 (legacy: "a 0.0 change is Pos with strength 0"). The definitional link `Lifespan -> Mortality (1.0, 1.0)` cannot be carried by MP without its leak (slope 0.98), so the **lift emits the flip itself** and adds one virtual hop for the shim; the materialised neutral link, when present as an Effect atom, stays an ordinary link and costs 2% (`infer Lifespan Mortality` -> Neg 0.98, c 1.0; a materialised `Taurine` chain 0.3477 vs 0.3548) `[sim]`.

Example `MixCompound` (rows built with the DrugAge formulas):

| row | legacy `infer MixCompound Mortality` |
|---|---|
| ITP mouse +20% Significant (s 0.5, c 0.9) | Neg 0.5000 / 0.8100 |
| worm +40% Significant (0.6667, 0.35) | Neg 0.6667 / 0.3150 |
| mouse +10% NotSignificant (0.3333, 0.2) | Neg 0.3333 / 0.1800 |
| ITP mouse 0% (null) (0.0, 0.9) | Neg 0.0000 / 0.8100 |
| mouse -30% Significant (0.6, 0.5) | Pos 0.6000 / 0.4500 |

Design (fold), recommended R2: **two results**: `Neg 0.263050 / 0.854519` (rows 1001-1004; weights 9.0, 0.5385, 0.25, 9.0; f = 0.263050, c_native 0.949466 `[hand]`) and `Pos 0.6 / 0.45` (one row, unchanged). With `fold = false` the five legacy results are reproduced exactly (the check that the lift is faithful).
Rank-scores: legacy per row [0.405, 0.210, 0.060, 0.000, -0.270]; design per direction [0.2248, -0.270]. Because a compound listed under both directions is a mixed signal, an **optional net** result is offered: the fork's `Truth_Revision` applied to the two directional *states* in both runs (same weights): `Neg 0.2194 / 0.8567`. Net pooling is the only place that revises across routes, and only for rows of one link. Which one `rank-interventions` should use is the owner's call (the legacy Python path keeps the best row per compound).
170 synthetic rows (resveratrol-like, 6 ITP): 170 legacy derivations -> `Neg 0.1289 / 0.8919` and `Pos 0.1667 / 0.8501`.

**Why a pre-pass fold and not the loop.** Rows as separate belief sentences with Tasks = KB: merged only with generous budgets and is sensitive to confidences `[sim]`: equal-confidence rows, 8 rows, default 20/20/200 -> best belief holds **1 of 8** rows; 20 rows at bq = 80 -> 6 of 20; 20 rows complete at 400/400/4000 (908 derivations, 115,643 belief pairs); 30 rows at the same budget take 1,403 derivations and 212,646 pairs. Same value as the fold when it completes. The fold is O(k).
Risks: rows from one paper are not independent (revision treats them as independent evidence and drives `c` toward 1); the owner's rule "one id per source record" implies that.

---------------------------------------------------------------------------------------------------------------------------

## 11. The number table

All values `[sim]`, cross-checked `[hand]` where marked; "legacy" = `closure.json` / `pinned_numbers.json` (real hyperon 0.2.10). Full per-pair data: `pins_all_variants.json` (186 derivations x legacy/R0/R1/R2).

### 11.1 Pinned chains

| query | legacy | R0 raw | R1 comp + native c | R2 comp + shim |
|---|---|---|---|---|
| D+Q -> CHD | Neg 0.2499 / 0.17017138 | Neg 0.2226 / 0.2334 | Neg 0.2499 / 0.2334 | Neg 0.2499 / 0.17017138 `[hand]` |
| CS -> CHD | Pos 0.357 / 0.29089125 | Pos 0.3274 / 0.3591 | Pos 0.357 / 0.3591 | Pos 0.357 / 0.29089125 |
| Metformin -> CHD | Neg 0.33 / 0.268515 | Neg 0.3018 / 0.3315 | Neg 0.33 / 0.3315 | Neg 0.33 / 0.268515 |
| Berberine -> CHD | Neg 0.286 / 0.268515 | Neg 0.2604 / 0.3315 | Neg 0.286 / 0.3315 | Neg 0.286 / 0.268515 |
| Fisetin -> CHD | Neg 0.23205 / 0.1309011 | Neg 0.2062 / 0.1796 | Neg 0.23205 / 0.1796 | Neg 0.23205 / 0.1309011 |
| Spermidine -> CHD | Neg 0.17136 / 0.07657712 | Neg 0.1481 / 0.1167 | Neg 0.17136 / 0.1167 | Neg 0.17136 / 0.07657712 `[hand]` |
| SASP -> CHD | Pos 0.42 / 0.49725 | Pos 0.3944 / 0.5525 | Pos 0.42 / 0.5525 | Pos 0.42 / 0.49725 |
| DNAmPAI1 -> CHD (record) | Pos 0.6 / 0.85 | Pos 0.58 / 0.85 | Pos 0.6 / 0.85 | Pos 0.6 / 0.85 |
| D+Q -> AllCauseMortality | empty | empty | empty | empty |

R2 differs from legacy only in the last digits: 64 of 186 strength strings and 47 of 186 confidence strings differ in the 16th-17th digit (e.g. D+Q strength prints `0.2499` vs legacy `0.24989999999999998`). Anything pinned as an exact string (the 32-case `patient_stack_builtin_golden.json`) must be regenerated under the new chainer, as LAUNCH.md already says; every `pytest.approx` pin holds.

### 11.2 Orderings and aggregates

| pinned | legacy | R0 | R1 | R2 |
|---|---|---|---|---|
| rank (D+Q, Fisetin, Spermidine; Elamipretide absent) | order holds | holds (0.05196, 0.03703, 0.01728) | holds | holds (0.04253, 0.03038, 0.01312) |
| 5-pool: Metformin > Berberine > D+Q > Fisetin > Spermidine | holds | holds | holds | holds |
| **Patient001 personalised order** DQ, Fisetin, Metformin, Berberine, Spermidine | 0.2306, 0.1647, 0.0886, 0.0768, 0.0712 | **FLIPS**: Spermidine 0.0916 > Berberine 0.0863 | **FLIPS**: Spermidine 0.1041 > Berberine 0.0948 | holds, values equal |
| Patient002 order Metformin, Berberine, DQ, Fisetin, Spermidine | holds | holds | holds | holds |
| relevance P001 DQ / Fisetin / Spermidine | 0.18808 / 0.13434 / 0.05804 | 0.2233 / 0.1591 / 0.0743 | 0.2453 / 0.1752 / 0.0841 | = legacy |
| diagnose order CS, CI, MD; coverage 3,2,1 | holds | holds | holds | holds |
| diagnose s(CS) 0.595, c(MD) 0.65, c(CS) 0.6253 (abs 1e-2) | 0.595 / 0.65 / 0.6253 | s(CS) 0.5644, **c(CS) 0.6961 fails** | **c(CS) 0.6961 fails** | holds |
| diagnose mass CS / CI / MD | 0.4593 / 0.9425 / 0.4875 | 0.5052 / 0.9165 / 0.4745 | 0.5391 / 0.9425 / 0.4875 | = legacy |
| counterfactual A delta, c | -0.121875, 0.65 | -0.118125, 0.65 | -0.121875, 0.65 | = legacy |
| counterfactual B delta, c | -0.2219297, 0.3176 | -0.2084, 0.3647 | -0.2219297, 0.3638 | = legacy |
| delta(DQ)/delta(B) = 0.70, c(DQ) < c(B) | 0.7000, 0.1858 < 0.3176 | holds | holds (hand-composed 0.2128, via `infer` 0.2365: inconsistent) | holds |
| NMN > Fisetin relevance (P001) | 0.1689 > 0.1343 (x1.257) | x1.106 | x1.071 | x1.257 |
| Omega3 targets {CRP, DNAmGDF15} | holds | holds | holds | holds |
| `infer CS CHD` c < 0.65 | holds | holds | holds | holds |

The Patient001 flip `[hand]`: Spermidine total = base 0.02000 + relevance 0.08410 = 0.10410 vs Berberine 0.09481 (native); with the shim 0.07116 vs 0.07680. Native confidence rewards a 5-hop chain that touches three elevated markers more than it should.

### 11.3 The contract's nine MUST items (contract.md section 5)

| # | MUST | Design 1 (R2) |
|---|---|---|
| 1 | no false chain; pair set = closure (100 / 86) | **holds** restricted (0 missing, 0 extra); **fails** with the full rule set (64 extra pairs) |
| 2 | sign = product of hop signs, Neg x Neg = Pos, Pos-only filters never see a Neg chain labelled Pos | holds: sign is the parity of the route's Neg links, cross-checked against the sign of the difference (0 mismatches); Neg x Neg tested |
| 3 | confidence decays with every edge, below the suffix chain's; strength is the plain product | holds (shim) and holds natively (`c_first*c_suffix`); strength exact with the compensation, shrunk 2.4-13.7% without |
| 4 | the orderings of contract 4.2 | holds R2; the Patient001 order and diagnose `c(CS)` fail under R0/R1 |
| 5 | determinism, candidate-order independence, exact goldens | deterministic and order-independent `[sim]`; exact goldens differ in the last digit for 64/186 strengths and 47/186 confidences: regenerate |
| 6 | exactly one derivation per connected pair on the shipped graph | holds; a multi-row compound returns one per direction (<= 2) |
| 7 | output shapes, reduced decimals, `explain` tv == `infer` tv | holds by construction (one table) |
| 8 | empty, not error, for unconnected / unknown / self; terminate on cycles; caller facts without precompile; no cross-request state | holds (KB built per call, node-simple filter, stamps) |
| 9 | same answers in all stacks; `infer`/`explain` stay `(= (infer $s ..) ..)` text; Effect atoms stay at line start | holds for the links of each stack; DrugAge enters through the lift, not a third `infer` equation |

### 11.4 Contract items that do not depend on the variant `[sim]`

No false chain (restricted): `D+Q -> AllCause`, `SmokingCessation -> CHD`, `Rapamycin -> CHD`, `{CI, MD, SmokingPackYears} -> CHD`, `Metformin -> CS`, `CRP -> GDF15`, `anything -> AgeAccelGrim`, `Elamipretide -> *`, self pairs: all empty. Pair sets equal the closure exactly (100, 86). One route per pair. Deterministic and independent of link order and candidate order. Output shape `(signed Pos|Neg (stv s c))`, `(Chain nodes (signed ..))`, numbers are reduced floats.

---------------------------------------------------------------------------------------------------------------------------

## 12. Runtime cost

Exact counts from the simulator (rule attempts = pair-rule tries + unary tries):

| workload | selections | belief pairs | rule attempts |
|---|---|---|---|
| patient graph, one run (mean over 60 runs) | 4.7 (max 12) | 53 | 84 |
| patient graph, one `infer`-table (2 runs) | ~9.3 | ~106 | ~168 |
| LinAge2 graph, mean per run | 4.2 (max 12) | 47 | 74 |
| `rank-interventions-for-patient` on the 5-candidate pool (10 runs) | 96 | 1,408 | 2,216 |
| tree, 30 links | 44 | 2,600 | 4,748 |
| tree, 126 links | 200 | 50,455 | 96,932 |
| tree, 510 links, 510 reached | 770 | 752,127 | 1,549,938 |

Cost ~ selections x beliefs: quadratic in the reachable links, because the fork has no antecedent index (every selected task is paired with every belief). Without the BFS pre-filter the patient workload is 5.9x higher. The wall time on PeTTa is **unknown**; the LAUNCH gate is "no routed probe more than 5x slower than hyperon". The legacy `infer` is one recursive `match`. Per-request memo by source; no state across requests.

---------------------------------------------------------------------------------------------------------------------------

## 13. The seam: what runs where

Under `PLN_CHAINER=legacy` (default) nothing changes. Under `pln` (PeTTa runtime only; hyperon stays green) a new file `pln_chainer_state.metta` defines `infer`/`explain` instead of `pln_deduction.metta`'s equations. Sketch (not run):

```
; links: collapse over (Effect $x $y $sg (stv $s $c)), record triples (terminal), DrugAge rows -> [Link from to sign p c ids hops]
; (evidence-confidence T) is reduced HERE, once; rows of one (from,to,sign) are folded with Truth_Revision (c capped), ids unioned
(= (chain-sentences $links) ...)           ; Link -> (Sentence ((Implication (State f) <(State t) | (Not (State t))>) (stv (+ p 0.02) c)) ids)
(= (chain-run $sentences $a $level)        ; level 1.0 / 0.0
   (PLN.Derive ((Sentence ((State $a) (stv $level 1.0)) (0)))
               (cons-atom <applied> $sentences) $steps $tq $bq))
(= (chain-table $space $a)                 ; memo per source
   (join the two belief lists on the stamp; keep node-simple paths; (hops (stamp))) ... )
(= (infer $space $a $c)   (superpose <routes of table(a) whose target is c>))     ; (signed Pos|Neg (stv s c))
(= (explain $space $a $c) (superpose <same routes>))                                ; (Chain (nodes) (signed ..))
```
Vendored library: PeTTa's bundled `lib_pln.metta` (sha256 d864c3cf..., = the fork's code, 63 of 81 forms identical) with `compat`'s patch set: E1 `(- $c)` -> `(- 0.0 $c)`, E3 drop the per-step `println` trace, plus the `(PLN.RuleOn ..)` guards of section 6. The forward `STV` declaration and the `Predicate` workaround are not needed by this design. The sentence terms `State`/`End` avoid the `Predicate` collision.
Python side: no change to the regex parsers (shape unchanged); a new unit that cross-checks `explain` against `infer` and the ate-vs-structure sign.

---------------------------------------------------------------------------------------------------------------------------

## 14. Changes to the consumers (R2 recommended)

| consumer | change |
|---|---|
| `rank-score`, `rank-interventions`, `patient-relevance`, `diagnose`, `pos-transmission`, LinAge2 | none; numbers equal legacy to rounding, orders hold |
| hand-composed lever sites `pln_counterfactual.metta:279-284`, `pln_linage2.metta:309-314` | none under R2 (`(c_l*c_b)*0.9` still equals `infer`'s shimmed c: 0.18582 both ways). Under native c they become inconsistent with `infer` (0.2128 vs 0.2365): re-route through `infer lever component` or drop the `0.9` |
| `drugage_calibration.metta` third `infer` equation | stays for legacy; under `pln` the DrugAge rows enter the link builder instead (lift, flip, fold) |
| `ontology/drugage_selector.py`, `run_drugage_ranking` (`pln_runner.py:861-960`), `drugage_scoring.py:170-210` (twin) | under `pln` the selector is unnecessary (fold uses all rows); the twin must reproduce fold semantics or be bypassed; `test_linear_and_metta_sort_agree_on_every_score` holds only for single-row compounds |
| `ranking` of a compound with both directions | listed twice (Neg, Pos) unless the net result is used |
| exact goldens (`patient_stack_builtin_golden.json`, 32 cases) | regenerate under `pln` (last-digit differences) |
| Python regexes `pln_runner.py:415-420, 588-591`, `kb_effect_markers` | none (Effect atoms stay in the space at line start) |
| `explain` | also answers DrugAge compounds |
| LLM prompt / few-shot | none |

---------------------------------------------------------------------------------------------------------------------------

## 15. Risks and unknowns

1. **PeTTa never run.** Equivalence of the real library to `fork_sim.py` for this workload is `[unknown]` until the spike reproduces section 2's numbers (D+Q 0.13769 / 0.36029 raw) on the vendored lib. Float printing: PeTTa prints shortest repr.
2. **Two named constants carry the equivalence** (compensation 0.02, discount 0.9). Both are read from text, not typed, and both have switches; R0/R1 are the "no tuning" readings and flip an order (section 11).
3. **State levels are artefacts** (leak 0.02/0.98). Any future consumer that reads a level as a probability is wrong. The adapter must never expose one.
4. **No pooling across routes**; diamonds give several results; a dense network explodes (802 selections, budget exhausted, partial reach). Needs a route cap or a different aggregation.
5. **Quadratic cost** without an antecedent index; pre-filter is mandatory; 510 reachable links = 1.55 M attempts per run `[sim]`.
6. **`Truth_Revision` returns nothing at `c = 1.0`**: a folded group containing a `1.0` confidence row (tier `RCT_Human` is 1.0) would vanish silently without the cap. The capped value changes that row's c by 1e-6.
7. **Pseudo-replication** of rows (same paper, several rows) inflates revised confidence (170 rows -> c 0.89 after the shim).
8. **Premise handling is fragile if changed**: confidence below 1.0 with an empty stamp floods the queue; with a non-empty unique stamp and c < 1 the premise can be revised with a derived loop state. Keep id 0 and c = 1.0 and assert it in the adapter.
9. **Definitional link** gives 0.98 when materialised.
10. **Library drift**: the trueagi rewrite changes MP confidence; assert the MP probe at load.
11. **Order of results** is an interpreter/queue artefact; the adapter sorts by (hops, stamp) so output is stable; legacy order was never pinned.
12. **Hyperon abort class**: `PLN.Derive` and the adapter run only under PeTTa, so the `trie.rs:179` abort does not apply `[unknown for PeTTa memory]`.
13. Unverified: variable endpoints `(infer &self $x D)` (legacy supports, no caller uses): would enumerate sources; not designed.

**Spike checks (all against the vendored lib, nothing else):** (a) section 2's D+Q numbers; (b) closure 100/86 with Tasks = (applied) at 100/10/100; (c) the MP probe; (d) a 2-row fold equals `sim`; (e) stamp `(0)` and node sentences `(State X)` pass the lib's `Sentence` patterns; (f) memory/time of one 510-link tree run; (g) `Truth_Revision` with `c = 1.0` yields no result.

---------------------------------------------------------------------------------------------------------------------------

## 16. Decisions for the owner

1. **Compensation** (`q = p + 0.02`): on (strengths equal legacy, strength = risk difference) or off (the fork read literally: every strength shrinks 2.4-13.7%). Recommended: on.
2. **Chain discount**: shim `c * 0.9^(hops-1)` (recommended; keeps every pin and the Patient001 order) or native `c1*c2` (the PLN number; flips the Patient001 order and moves diagnose `c(CS)` to 0.696). This is a question for Ben: is 0.9 a PLN concept?
3. **Multi-row output**: per direction (what was asked; a compound may be listed twice) or the optional net (one signed result per compound).
4. **Revision in the loop or as a fold**: fold (recommended; O(k), independent of budgets) or rows as belief sentences with budgets that grow with k.
5. **Full rule set as a comparison**: accept that it needs `STV` for every node and a larger budget, and that it leaves the closure (64 pairs).

---------------------------------------------------------------------------------------------------------------------------

## 17. Effort

About **5 agent-phases**: (1) vendor + rule guards + spike (a-g) 1; (2) link builder (Effect, records, DrugAge, evidence-confidence reduction, fold) and the `chain-run/table/infer/explain` MeTTa 2; (3) consumers, DrugAge Python twin, new pins and the number table in both rule-set modes 1-2; (4) review (cycles, caps, cost, harness) 1.

---------------------------------------------------------------------------------------------------------------------------

## 18. Files and reproduction

```
python3 -I run_numbers.py --full     # every number above; ~80 s; writes results.json (stdout: run_numbers.out)
python3 -I hand_check.py             # independent arithmetic (hand_check.out)
python3 -I dump_pins.py              # pins_all_variants.json
```
| file | content |
|---|---|
| `mp_state.py` | the reference adapter: encoding, reachable-links filter, two-run derivation, route join, DrugAge lift and fold, net pooling |
| `consumers.py` | legacy consumers' arithmetic re-typed (ranking, relevance, diagnose, counterfactual) |
| `run_numbers.py`, `run_numbers.out`, `results.json` | S1 algebra, S2 closure, S3 pins, S4 orderings, S5 no-false-chain (+ full), S6 feedback, S7 multi-row, S8 control, S9 engraved/determinism, S10 scale |
| `full_probe.py`, `rows_probe.py` | the full-rule-set and in-loop row probes |
| `hand_check.py`, `hand_check.out` | independent arithmetic |
| `pins_all_variants.json` | 186 derivations x {legacy, R0, R1, R2} |
| `make_structure.py`, `structure.json` | the structured summary returned to the orchestrator (field lengths checked) |
| `fork_sim.py`, `data/` | validated simulator copy; copies of `links.json`, `closure.json`, `pinned_numbers.json`, `legacy_reference.py` |
