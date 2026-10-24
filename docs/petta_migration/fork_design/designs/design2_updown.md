# Design 2: syllogistic deduction on up/down nodes

Work tree `/Users/kolya/PycharmProjects/pln_longevity_petta` (linage2 @ 47130aa), read only. The fork is patham9/PLN `55f1751`; its PeTTa port is `.petta/PeTTa/lib/lib_pln.metta` (fea83178). Nothing from `.petta/` was executed. Every number below comes from `run_all.sh` (output in `numbers.txt`), which uses a copy of the validated simulator of the fork (`sim/fork_sim.py`, 20 of 20 upstream pins reproduced) and my own arithmetic. **Nothing here has run on PeTTa.** Where I could not establish something I say `unknown`.

Files in this folder: `design.md` (this), `d2lib.py` (encoder, runner, adapter; the reference implementation of the seam), `consumers.py` (replays of the consumer arithmetic), `d2drugage.py` (DrugAge rows to links), `exp*.py` + `run_all.sh` + `numbers.txt` (all numbers), `structure/` (generated MeTTa text, the library patch, the id table), `data/` (copies of `links.json`, `closure.json`, `pinned_numbers.json`, the DrugAge CSV from the repo's `data/drugage/dataset.zip`, `species_taxonomy.metta`).

---------------------------------------------------------------------------------------------------------------------------

## 0. Result

1. **It works.** On the shipped patient graph (28 links, 30 nodes) the design answers exactly the 100 ordered pairs of `closure.json`, with the same sign and the same chain for every pair; on the LinAge2 stack it answers its 86. No pair outside the closure answers (`D+Q -> AllCauseMortality`, `SmokingCessation -> CHD`, `Rapamycin -> CHD`, `Elamipretide -> *` stay empty). In 4,899 answers on 300 random graphs with cycles, diamonds and mixed signs the answer key set equals the legacy chainer-with-a-visited-set exactly (`exp15`).
2. **Strength is exactly the legacy product, and the base rate is a gauge, not a knob.** If a link of legacy magnitude `s` is stored as the conditional probability `P(B|A) = p + (1-p)s` ("B copies A with probability s, otherwise B is drawn from its base rate"), the fork's own Deduction formula returns `p + (1-p)*s1*s2`. So `(s_fork - p)/(1 - p)` is the legacy product, for every uniform `p` in (0,1): maximum difference 2.2e-16 on all 100 patient pairs at p = 0.1 (5e-16 at p = 0.5, 1.6e-14 over 22,000 random pairs and 11 priors). Read literally (`P(B|A) = s`) the same formula regresses chains toward the base rate and moves D+Q -> CHD from 0.2499 to 0.2852 at p = 0.1 and 0.5112 at p = 0.5.
3. **The fork's confidence rule is the only real conflict.** Deduction takes the minimum of five confidences; legacy multiplies and discounts 0.9 per combination. With the minimum, **12 of the 29 pinned invariants I replayed break** (listed in section 4), including the personalised order of Patient001 (Spermidine jumps above Metformin and Berberine), NMN > Fisetin, `c(B) < c(A)` of the counterfactual, `c < 0.65` for a 3-edge chain, and every value pin of a chain longer than one edge. With a **one-hunk patch** of `Truth_Deduction`'s confidence line (a named switch, `fork` or `legacy`; `structure/deduction_confidence.diff`) none of the 29 breaks and confidences agree to 2.8e-17. I recommend the patch as the default of `PLN_CHAINER=pln` and the unpatched minimum as a comparison switch. This is a deviation from patham9's rule and needs the owner's decision.
4. **A naive integration silently loses most answers.** `PLN.Query kb term` over the whole up/down KB with the library defaults answers 40 of 100 pairs (fork defaults 20/20/200) or 45 of 100 (PeTTa defaults 100/10/100), and loses the D+Q -> CHD pin outright under the fork defaults. The design therefore derives **per source node** (tasks = the source's first hops, beliefs = the states reachable from it): 100 task selections and about 1,820 rule attempts for all 30 sources, against 687 selections and 290,756 attempts for the whole-graph run.
5. **Without a base rate Deduction derives nothing, silently.** No `(STV X)` equation: 28 of 100 pairs answer (the direct links). One node missing its two equations loses 48 pairs. The encoder generates one equation per state node from the link set and refuses a request if one is missing.
6. **Restricted rules only.** The full rule set on the same KB answers 366 pairs, 266 of them outside the closure, and returns `D+Q -> AllCauseMortality` as Neg (0.0625, 0.132). It also changes pinned confidences (D+Q 0.1702 -> 0.1797).
7. **Feedback edge** `ChronicInflammation -> CellularSenescence` (Pos): terminates, 109 answer keys (9 new, all reachable only through the new edge), the 100 old answers unchanged (`exp4`, `exp11`). Legacy hangs.
8. **Multi-row revision works, but one id per DrugAge row is the wrong unit.** Four synthetic rows (extending, null, shortening) revise into one value per direction: Neg (0.4883, c 0.9087) and Pos (0.0151, c 0.9087), equal to hand arithmetic. But `ethanol` has 52 rows from one paper (each c = 0.21); revised as 52 independent pieces of evidence it reaches c = 0.933 and ranks first of 1,038 compounds. One representative row per PMID, revised across papers, is the unit I recommend (section 7).

Decisions needed from the owner: (a) the confidence lane (item 3); (b) the DrugAge evidence unit and the ranking changes that follow (item 8, section 7); (c) cycles: allow with the guard ids, or refuse at write time; (d) vendoring a second patch hunk.

---------------------------------------------------------------------------------------------------------------------------

## 1. The encoding

**Nodes.** Every node `X` becomes two concepts, `(Concept X-up)` ("X is elevated or applied") and `(Concept X-down)` ("X is reduced"). The sign of a result is path structure, never a truth-value flip: the answer is `Pos` if the target state is `C-up`, `Neg` if it is `C-down`.

**Links.** A legacy link `A -> B` with sign, magnitude `s`, confidence `c` and evidence id `e` becomes two sentences with the same stamp `(e)` and the same truth value:

| legacy | sentences (dynamic beliefs, `Sentence`) | strength stored |
|---|---|---|
| `(Effect A B Pos (stv s c))` | `(Implication A-up B-up)` and its mirror `(Implication A-down B-down)` | `p + (1-p)s` |
| `(Effect A B Neg (stv s c))` | `(Implication A-up B-down)` and its mirror `(Implication A-down B-up)` | `p + (1-p)s` |
| hazard-ratio record (`HasPredictor`/`HasOutcome`/`EffectValue`) | as an Effect: sign Neg if HR < 1; `s` = 0.70 / 0.60 / 0.50 by HR; `c` = tier of `HasEvidenceCategory` | same |
| DrugAge row | two links `X -> Lifespan`, one Pos and one Neg (section 7), plus the fixed `Lifespan -> Mortality` Neg (1.0, 1.0) | same |
| node `X` | `(= (STV (Concept X-up)) (stv p 1.0))` and `-down` (the fork's Translator would engrave these from static concept sentences; with the bundled lib they are hand-written equations) | p |

The mirror is not decoration. A Neg link leaves the chain in the down state of its target, and the next link must be applicable from that state: `Spermidine-up -> Autophagy-up` (Pos), `Autophagy-up -> CellularSenescence-down` (Neg), then the mirror `CellularSenescence-down -> SASP-down`, `SASP-down -> DNAmPAI1-down`, `DNAmPAI1-down -> CHD-down`: the answer is `Neg`, as legacy's `Pos*Neg*Pos*Pos*Pos`. `Neg*Neg = Pos` falls out of the same walk (`X-up -> Y-down -> Z-up`). Mirrors are emitted only for links whose source has an in-link (43 sentences instead of 56 on the patient graph; identical answers, `d2lib` option `prune_mirrors`). Primary and mirror share the link's evidence id, so a chain can never use one record in both orientations.

**Evidence ids.** One id per source record: one per curated `Effect` atom, one per hazard-ratio record, one per PMID for DrugAge (section 7). Ids are per-request ordinals in a canonical order (sorted by from, to, sign, kind, s, c), so load order cannot change any number (`exp13`: identical answers after shuffling). An id that repeats with the same (from, to, sign) is the same evidence and is kept once; this is how a curated `Effect` lifted from a DrugAge row is prevented from counting twice (Taurine: same id gives Neg 0.3548 / 0.45, as one claim; two different ids revise to 0.3548 / 0.60; legacy returns the first twice).

**Cycle-guard ids.** The fork's stamp-disjointness stops one link being used twice in a derivation, not a node being visited twice. In a cycle that leaves edge-simple trails that revisit a node. Every link whose source lies on a directed cycle (Tarjan, `cyclic_nodes`) carries one extra stamp id, a **negative** integer per node (negative so it cannot equal a PMID or an ordinal). A trail would have to leave that node twice and is rejected by the fork's own disjointness test. On the shipped graph there are no cyclic nodes, so no marker is emitted.

**What a sentence looks like** (generated, `structure/derive_DasatinibPlusQuercetin.metta`):

```
(= (STV (Concept CellularSenescence-down)) (stv 0.1 1.0))
(Sentence ((Implication (Concept DasatinibPlusQuercetin-up) (Concept CellularSenescence-down)) (stv 0.73 0.65)) (12))
```
`0.73 = 0.1 + 0.9*0.7`.

---------------------------------------------------------------------------------------------------------------------------

## 2. Strength: the noisy-copy mapping (why the product comes out of PLN)

The fork's Deduction (`Formulas.metta:5-20`) is `s_AC = s_AB*s_BC + (1-s_AB)(s_C - s_B*s_BC)/(1-s_B)`. With the literal reading "legacy strength = P(B|A)" the second term is not small: `(s_C - s_B*s_BC)/(1-s_B)` is the probability of C when B is off, and with a base rate of 0.1 the literal reading moves D+Q -> CHD from 0.2499 to 0.2852, +14%.

The legacy strength is a magnitude with 0 meaning no effect; in PLN "no effect" is `P(B|A) = P(B) = p`. Store `P(B|A) = p + (1-p)s` and `P(B|not A) = p(1-s)`. This is a symmetric channel that copies A to B with probability `s` and otherwise draws B from its base rate; it preserves the marginal `p`, and it composes: two channels in series copy with probability `s1*s2`. Deduction is exactly this composition for a two-state variable, hence

`s_fork(A->C) = p + (1-p) * s1 * s2`, and `(s_fork - p)/(1-p) = s1*s2` for every chain length, for any `p` in (0,1).

**Worked example, D+Q -> CHD (p = 0.1).** Legacy links: D+Q -> CS Neg (0.7, 0.65); CS -> SASP Pos (0.85, 0.65); SASP -> DNAmPAI1 Pos (0.7, 0.65); DNAmPAI1 -> CHD, a hazard-ratio record, Pos (0.6, 0.85). Stored: 0.73, 0.865, 0.73, 0.64. All base rates 0.1, so the consistency bounds are 0 and 1.
1. `D+Q-up -> CS-down` (0.73) with `CS-down -> SASP-down` (0.865): `0.73*0.865 + (1-0.73)*(0.1 - 0.1*0.865)/(1-0.1) = 0.63145 + 0.00405 = 0.6355`; unshifted `(0.6355-0.1)/0.9 = 0.595 = 0.7*0.85`. Confidence: lane P `(0.65*0.65)*0.9 = 0.38025`; lane N `min(1,1,1,0.65,0.65) = 0.65`.
2. with `SASP-down -> DNAmPAI1-down` (0.73): `0.6355*0.73 + (1-0.6355)*(0.1-0.073)/0.9 = 0.463915 + 0.010935 = 0.47485`; unshifted 0.4165; P: `(0.38025*0.65)*0.9 = 0.22244625`; N: 0.65.
3. with `DNAmPAI1-down -> CHD-down` (0.64, 0.85): `0.47485*0.64 + (1-0.47485)*(0.1-0.064)/0.9 = 0.303904 + 0.021006 = 0.32491`; unshifted `(0.32491-0.1)/0.9 = 0.2499`; P: `(0.22244625*0.85)*0.9 = 0.17017138125`; N: `min(.., 0.85) = 0.65`. The stamp is `(6 11 12 25)`, the four link ids; the chain is read from it.
These three intermediate beliefs are the ones listed as the expected result in `structure/derive_DasatinibPlusQuercetin.metta`.

Measured (`exp1`, `exp8`, `exp12`):

| check | result |
|---|---|
| 22,000 random pairs, 11 priors from 1e-6 to 0.99 | max `|unshifted - s1*s2|` = 1.6e-14, no consistency failure |
| patient graph, 100 pairs, p = 0.1 / 0.5 / 0.01 | max `|ds|` = 2.2e-16 / 5.0e-16 / 2.2e-16 |
| LinAge2 graph, 86 pairs | same |
| 5 bracketings of a 4-link chain, shifted or not | spread <= 3.3e-16 (deduction is associative here, so the order the loop picks cannot change a number) |
| every p in (0,1) passes the fork's consistency check with shifted strengths | yes (100 of 100 pairs answer at p = 1e-6 ... 0.999) |

So the fork contributes the engine, the evidence stamps and the revision; for strength the design is the legacy chainer re-expressed in PLN terms. I state that plainly because it is what the numbers say. The **literal lane** (`PLN_STRENGTH_MAP=literal`) keeps the fork's own reading as a comparison switch; its cost is that the answer depends on `p` (section 9).

Assumptions the mapping makes explicit: the channel is symmetric (an increase and a decrease of A act with the same magnitude, which is what the legacy mirror semantics already assumed); `X-up` and `X-down` are separate binary variables, not three-state (`p_up + p_down <= 1` is therefore the semantic bound, so `p <= 0.5`); links are independent channels.

---------------------------------------------------------------------------------------------------------------------------

## 3. Base rates

**Decision: one uniform base rate `p = 0.1` for every state node, node confidence 1.0.** Reasons: under the copy mapping every `p` gives the same legacy-domain output, so the value only has to be strictly positive, at most 0.5 (up and down exclusive), numerically comfortable (float error 2e-16 at 0.1; it grows as 1/(1-p): 1.5e-13 at 0.999) and in the order of the 16% prevalence of `z > 1` (the patient layer's "elevated" threshold). Node confidence 1.0 keeps the nodes out of the confidence minimum.

**What happens with no base rate** (`exp12 (a)`): the fork has no default (`grep STV src/` finds only call sites; the bundled lib adds `(= (STV $stv) (empty))`, `lib_pln.metta:116`). Deduction calls `(STV A)`, `(STV B)`, `(STV C)`; a call with no equation yields no result, so the rule derives nothing, silently. On the patient graph: no equations at all -> 28 of 100 pairs (direct links only); one node missing -> 48 pairs lost. `p = 0` gives 28 of 100 in the simulator, but in the real runtimes `/` raises on a zero divisor (`smallest-intersection-probability` divides by `As`), so there it is an error, not an empty answer (`compat/petta_compat.md` section 2.3); `p = 1` collapses Deduction to `Rs` (the `Qs > 0.9999` branch) and the unshift would divide by zero. The encoder requires 0 < p <= 0.5. The encoder derives the node list from the link list, so a missing equation cannot occur, and `validate()` still checks it.

**The consistency check** (`Constraints.metta:11-15`, `conditional-probability-consistency`) requires `max(0,(As+Bs-1)/As) <= ABs <= min(1,Bs/As)` and otherwise Deduction returns the integer pair `(stv 1 0)`, a zero-confidence belief that PLN.Query never returns, i.e. a lost chain.
* Uniform `p`, **shifted** strengths: always consistent, for every `p` in (0,1) and every strength in [0,1], including 0 and 1 (`exp12 (b)`, and 83,781 chained deductions with strengths drawn from {0, 1e-12, 0.5, 1-1e-12, 1}: zero failures, `exp23`). The derived strengths are conditional probabilities of a real joint, so they stay consistent when reused as premises.
* Uniform `p`, **literal** strengths: consistent for `p <= 0.5`; at `p = 0.75` the lower bound is 0.667 and 47 pairs are lost, at 0.9 it is 0.889 and 72 are lost.
* **Prevalence-style priors** (illustrative numbers, not data: markers and clocks 0.159, mechanisms 0.30, interventions 0.02, CHD 0.06, CHF 0.03, all-cause 0.10): 11 of 28 links fail the check with literal strengths (13 with shifted ones), e.g. `ChronicInflammation -> CRP` needs `P(B|A) <= 0.159/0.30 = 0.53` but carries 0.80; the closure shrinks to 52 of 100 pairs with 48 `(stv 1 0)` beliefs. The mapping also stops being a product when the priors differ (the copy channel only preserves a marginal when `p_A = p_B`). Prevalence priors are therefore a curation project that must come with a per-link consistency validator and re-derived pins; they are not a starting point. The inversion rule (comparison switch only) copies strength, which is right only for equal node probabilities: another reason for uniform.

---------------------------------------------------------------------------------------------------------------------------

## 4. Confidence: the fork's minimum against legacy's product

* Fork (`Formulas.metta:18`): `c = min(Pc, Qc, Rc, PQc, QRc)`; it does not decay with length.
* Legacy (`pln_deduction.metta:82-95`): `c = (c1*c2)*0.9`, so n edges give `n-1` discounts and a longer chain is always below its suffix.
* Patch (`structure/deduction_confidence.diff`, one hunk, 11 added lines): `Truth_Deduction`'s confidence becomes `(Truth_Deduction.Conf ...)` = `min` of five if `(PLN.Config.DeductionConfidence)` is `fork`, else `(* (PLN.Config.HopDiscount) (* PQc QRc))` with `HopDiscount` = the project's `(chain-discount)` 0.9. The product form is associative, so every bracketing of a chain gets `0.9^(n-1) * prod(c)`. The trueagi rewrite also dropped the minimum, for a product that includes the strengths (`TRUEAGI:97`, `(* (* PQs QRs) (* PQc QRc))`); the plain `c1*c2` and the 0.9 are this project's.

Pinned chains (patient graph, p = 0.1; strengths identical in the first two lanes to 2.2e-16):

| chain | legacy (s, c) | **P** parity lane | **N** native lane (min) | **L** literal + min |
|---|---|---|---|---|
| SASP -> CHD | Pos 0.42 / 0.49725 | same | Pos 0.42 / 0.65 | Pos 0.43333 / 0.65 |
| CS -> CHD | Pos 0.357 / 0.29089 | same | Pos 0.357 / 0.65 | Pos 0.37778 / 0.65 |
| D+Q -> CHD | Neg 0.2499 / 0.17017 | same | Neg 0.2499 / 0.65 | Neg 0.28519 / 0.65 |
| Fisetin -> CHD | Neg 0.23205 / 0.13090 | same | Neg 0.23205 / 0.50 | Neg 0.26975 / 0.50 |
| Spermidine -> CHD | Neg 0.17136 / 0.07658 | same | Neg 0.17136 / 0.50 | Neg 0.22003 / 0.50 |
| Metformin -> CHD | Neg 0.33 / 0.268515 | same | Neg 0.33 / 0.60 | Neg 0.35278 / 0.60 |
| Berberine -> CHD | Neg 0.286 / 0.268515 | same | Neg 0.286 / 0.60 | Neg 0.31389 / 0.60 |
| CS -> DNAmPAI1 | Pos 0.595 / 0.38025 | same | Pos 0.595 / 0.65 | Pos 0.6 / 0.65 |
| DNAmPAI1 -> CHD (HR record) | Pos 0.60 / 0.85 | same | same | same |

Pinned invariants replayed (29 checks, `exp10`; the replay code reproduces the pinned composite outputs bit for bit on legacy inference: relevance 0.1880788465877344, personalised D+Q score 0.2306046747621094, counterfactual A/B/DQ, diagnose c(CS) 0.6253054502923161):

* Lane P: **0 of 29 break.**
* Lane N (minimum): **12 break**: the value pins A1-A6 (SASP, CS, D+Q, Fisetin, Spermidine, Metformin chains) by design; `c(CS->CHD) < 0.65` (it equals 0.65); the Patient001 personalised order (`[DQ, Fisetin, Spermidine, Metformin, Berberine]` instead of `[DQ, Fisetin, Metformin, Berberine, Spermidine]`: Spermidine 0.480 against Metformin 0.198, because a 5-edge chain is no longer penalised); NMN > Fisetin for Patient001 (0.268 against 0.534); `c(B) < c(A)` of the counterfactual (both 0.65; legacy 0.3176 against 0.65); `c(B) = 0.3176`; diagnose `c(CS) = 0.625` (0.957). The population ranking `[DQ, Fisetin, Spermidine]` (0.1624, 0.1160, 0.0857), `Metformin > Berberine`, the Patient002 order, relevance order, targets, no-false-chain and all `CHD_REACHING_CAUSES` hold.
* Lane L: the same 12 plus two (`s(CS) = 0.595` and the counterfactual deltas); at p = 0.5 the 5-pool order puts D+Q above Metformin.

Second-order effects of the minimum worth knowing: scores are about 4 times larger (D+Q 0.0425 -> 0.1624), so every threshold or display built on `s*c` changes; the hand-composed sites (`pln_counterfactual.metta:279-284`, `pln_linage2.metta:309-314`) keep the product-with-0.9 rule and become inconsistent with `infer`; and in a cyclic graph a node-revisiting trail ties with the simple path on confidence (both 0.65), so a plain `PLN.Query` returned a trail for 7 of 109 keys without the cycle-guard ids (`exp11`).

**Recommendation.** Ship `PLN_DEDUCTION_CONFIDENCE=legacy` as the default of the `pln` chainer (parity, all invariants hold, and a longer chain is strictly less confident so "highest confidence wins" prefers the simple path), keep `fork` as a documented switch so Ben can see the fork's own numbers, and give the `fork` lane its own pins. That is a change to patham9's formula that I cannot undo for the owner: it is decision (a).

---------------------------------------------------------------------------------------------------------------------------

## 5. Rule set

**Restricted: Revision + Deduction.** Modus Ponens, Not-elimination and the Translator's engraved rules are not needed (no non-implication beliefs, no `Not`). If patient observations later enter as beliefs (`(Concept CRP-up) (stv ..)`), Modus Ponens is the rule to add.

**Full (comparison switch only)**, tested with Revision, Deduction, Induction, Abduction, both Inversions and Modus Ponens (`exp19`, `exp24`): 366 pairs answer, **266 outside the closure** (`AgeAccelGrim -> Autophagy`, ...); `D+Q -> AllCauseMortality` Neg (0.0625, 0.1322) stamp (1,3,6,11,12,25); `MitochondrialDysfunction -> CHD` Pos (0.174, 0.181); the loop does not quiesce: 20 of the 30 sources keep tasks after 200 steps (`exp19`), and none of the 10 pinned sources of `exp24` empties its task queue in 400. For the pinned chains the full set returns the same strength but a higher confidence: with equal node probabilities Induction's strength equals Deduction's, and its confidence is `w2c(min(c1,c2))`, so e.g. D+Q -> CHD at 0.1797 is Induction applied to the inversion of `D+Q -> DNAmPAI1` (as `DNAmPAI1-down -> D+Q-up`, c 0.219) and `DNAmPAI1-down -> CHD-down` (`exp25`), and `PLN.Query` takes the highest confidence (steps 400, queues 800, tasks = first hops, beliefs = whole KB, conf lane P): D+Q 0.1797 (restricted 0.1702), Fisetin 0.1579 (0.1309), Spermidine 0.1364 (0.0766), CS 0.2909 and Metformin 0.2685 unchanged; SmokingCessation -> CHD and ChronicInflammation -> CHD stay empty. These numbers depend on the step and queue settings, so each carries them.

---------------------------------------------------------------------------------------------------------------------------

## 6. Control

**What I rejected, with numbers** (`exp2`, `exp17`). The fork's loop selects the highest-confidence task each step and pairs it with every belief. Seeding it with the whole KB makes every sentence derivable in several bracketings and re-queued: 687 selections, 304 beliefs, 175,074 belief pairs, 290,756 rule attempts to empty the task queue; the closure is complete at 400 steps with queues 400/800 and still 9 pairs short at 200 steps with 200/400. With the library defaults:

| setting (whole KB as tasks and beliefs) | pairs answered of 100 | D+Q -> CHD |
|---|---|---|
| fork 20 steps / 20 / 200, Unique reverses | 40 | lost |
| PeTTa lib 100 / 10 / 100, Unique keeps order | 45 | 0.2499 / 0.65; CS -> CHD and Berberine -> CHD lost |

**What I chose.** One derivation **per source node**, cached for the request:

* `slice(A)` = the sentences reachable from `(Concept A-up)` at state level.
* `PLN.Derive tasks=first-hop sentences of A-up, beliefs=slice, maxsteps=4K, tq=K, bq=K` (the 5-argument form, `Deriver.metta:54`). `PLN.Query` is not used: its single-term answer would rerun the closure per term; the adapter reads every belief with an `A-up` antecedent.
* On a DAG every derived sentence is derived once: selections equal the number of reachable pairs (`exp3`: 100 selections, 1,820 rule attempts, all 30 sources; the widest source, Spermidine, 10 selections, 19 beliefs, 244 attempts). The task queue empties by itself.
* **Budget schedule** `K = 256, 2048, 16384`. A result is accepted only if the returned task list is empty **and** `|beliefs| < K-1`. `LimitSize` keeps at most `K-1` items, and every task is also a belief, so with `tq = bq = K` a belief list shorter than `K-1` proves nothing was ever cut (it is observable in PeTTa, where there are no counters). Otherwise retry with the next K, then fail the request (HTTP failure like the other upstream failures), never return a partial closure. With a 5-step budget the guard raises (`exp13`).
* The library defaults happen to suffice on the shipped graph (slices of at most 10 sentences) and give identical answers; nothing guarantees it for a caller's graph, hence explicit budgets and the guard.
* Robust to the runtime unknowns: answers are identical with `Unique` reversing or keeping order, with the equation order reversed, and after shuffling the load order (`exp13`).
* Feedback edge: 151 selections and 1,784 attempts for the 29-link graph (maximum 2.78 selections per slice sentence). On 447 sources of random cyclic graphs the cycle-guard ids cut the median/p90/max selections from 16/186/1,329 to 12/64/278, and 16 sources that did not quiesce in 1,500 selections to none.

Switches the seam exposes: `PLN_RULESET=restricted|full`, `PLN_DEDUCTION_CONFIDENCE=legacy|fork`, `PLN_STRENGTH_MAP=copy|literal`, `PLN_BASE_RATE=0.1`.

---------------------------------------------------------------------------------------------------------------------------

## 7. Multi-row revision (DrugAge and any parallel evidence)

**Design.** Each DrugAge row `r` of compound `X` contributes evidence about **both directions** at its own confidence `c_r = tier x gate` (read from the existing MeTTa lift `drugage-effect`, so the tier table stays the single authority):

* `X-up -> Lifespan-up` ("extends") with legacy strength `s_r` if the change is positive, else 0;
* `X-up -> Lifespan-down` ("shortens") with `s_r` if negative, else 0.

A strength of 0 is "no effect" (it maps to `p`), which is also how legacy treats a null: `Resveratrol ITP mouse 0%: (stv 0.0 0.9)`. Rows of one direction are folded with the fork's own `Truth_Revision` (weights `c/(1-c)` add; `c = w/(w+1)`), the stamp is the union of the ids, then Deduction continues through `Lifespan -> Mortality`. Result: at most **two** answers per compound, `Neg` (extends) and `Pos` (shortens), not one per row. A direction whose revised strength is 0 is dropped when the other is non-zero (when both are 0, only Neg is kept: legacy's convention for `pct >= 0`).

**The fold happens in the encoder, not in the loop.** Revising `n` rows inside the Deriver builds the lattice of subsets: 68 selections for n = 3, 2,616 for n = 5, and n >= 8 does not finish in 3,000 steps (`exp6`). The best belief is right (it equals the fold to 1e-15) but the cost is exponential, so the fork's formula is applied by the encoder and the Revision rule remains for what it handles well: edge-disjoint parallel paths found during derivation (a diamond is revised into one value; hand-checked: Pos 0.55018 / 0.72737 from paths (0.56, 0.576) and (0.54, 0.567), `exp11`).

**Synthetic example** (ITP mouse +20% significant; worm +40% significant; mouse -30% not significant; fly 0% not significant; `exp6`):

| | legacy (one derivation per row) | this design, P lane | N lane |
|---|---|---|---|
| | Neg 0.5000/0.8100, Neg 0.6667/0.3150, Pos 0.6000/0.1800, Neg 0.0000/0.1260 | **Neg 0.488278 / 0.817818, Pos 0.015073 / 0.817818** | Neg 0.488278 / 0.908686, Pos 0.015073 / 0.908686 |

Hand check: weights 9, 0.5385, 0.25, 0.1628 sum to 9.9513, so `c = 0.908686`; `s_up = (9*0.5 + 0.5385*0.6667)/9.9513 = 0.488278`, `s_down = 0.25*0.6/9.9513 = 0.015073`. The legacy fixtures `GoodDrug`, `WeakDrug`, `NilDrug`, `BadDrug` give exactly their old scores through the new path (0.486, 0.1575, 0.0, -0.486; order Good > Weak > Nil > Bad).

**What counts as one piece of evidence** (`exp21`, `exp22`; the CSV in the repo's `data/drugage` is used as a proxy for the ETL build: 1,038 compounds, 3,372 scorable rows):

| variant | Rapamycin rank | Resveratrol | Metformin | top-20 overlap with legacy | the five of `test_drugage_calibration.py:340-350` |
|---|---|---|---|---|---|
| V0 legacy: one representative row | 2 | 0.000 | 0.000 | 20 | Rapamycin > Trimethadione > Acarbose > Resveratrol = Metformin = 0 |
| V1 revise all rows, id per row | 28 | 0.116 (c 0.891) | 0.164 | 4 | Trimethadione first |
| **V2 one representative per PMID, revise across PMIDs** | 3 | 0.170 (c 0.875) | 0.134 | 9 | Rapamycin > Acarbose > Trimethadione > Resveratrol > Metformin |
| V3 revise all rows, confidence capped at the best row | 1 | 0.105 | 0.151 | 12 | Rapamycin first |
| V5 per PMID: fold its rows with a cap, then revise | 4 | 0.198 | 0.206 | 6 | Rapamycin first |

V1, the literal reading of "one evidence id per row", ranks `ethanol` first of 1,038: its 52 rows come from one paper, one species, each at confidence 0.21, and revision turns them into c = 0.933 (strength 0.698). Resveratrol's 170 rows (37 papers, 6 ITP rows, all null) revise to c = 0.990. Rows of one paper (doses, sexes, strains) are not independent. **Recommendation: V2**, using the existing representative policy (`REPRESENTATIVE_POLICY`) within each paper and the stamp id = PMID (rows without a PMID fall back to a row id from a separate range). It keeps the hierarchy inside a paper, removes the multiplicity artefact, and keeps Rapamycin first among the five of the existing pin. It does change the pinned lines "Resveratrol and Metformin score 0" (`test_drugage_calibration.py:348-349`): that is the intended consequence of no longer keeping one row, and they need new pins. Confidence of a revised value grows without bound toward 1; if the owner wants a ceiling, V5 (cap within a paper) is the knob. The real ranking tests (`test_drugage_ranking_contract.py`: Rapamycin 0.81, Trimethadione 0.315) skip without the ETL build, so I could not replay them.

**Duplicate provenance** (raw row + curated `Effect` for one claim): same id -> one answer (0.3548 / 0.45); different ids -> revised (0.3548 / 0.60). Legacy returned the identical answer twice. **Hazard-ratio records and curated Effects on the same pair** with the same sign revise into one answer; with opposite signs both are kept, as legacy.

---------------------------------------------------------------------------------------------------------------------------

## 8. Cycles, diamonds, and what stays empty

* **Feedback edge** `ChronicInflammation -> CellularSenescence` (Pos, 0.6, c 0.65). The task text calls the node `InflammationChronic`; the node in the KB is `ChronicInflammation`, and an edge from a node that does not exist would not close a cycle. Cycle: CS -> SASP -> CI -> CS. Result: all 30 sources quiesce (151 selections, no queue cut), 109 answer keys against the visited-set reference, 0 mismatches; the 100 old answers keep their values; the 9 new ones are `CI -> CS, SASP, DNAmPAI1, CHD`, `Omega3 -> CS, SASP, DNAmPAI1, CHD` (so `Omega3 -> CHD` becomes Neg 0.1285 / 0.1302: a real consequence of the new edge) and `SASP -> CS`. The fork derives self-implications (`CS-up -> CS-up`, 0.4335 / 0.2224); the adapter drops `A == C` as legacy does. Node-revisiting trails are rejected by the guard ids; a belief is accepted only if the simple `A .. C` paths inside its stamp cover every link of the stamp (`_paths_in`), which also accepts revised diamonds.
* **Diamond** (edge-disjoint parallel paths): one revised answer where legacy returned two. **Paths that share a link** are not revised (their stamps overlap); the more confident one is returned. Mixed-sign diamonds return one Pos and one Neg answer, as legacy. The shipped graphs have no diamond (1 derivation per pair), so none of their answers is affected.
* **Terminal records.** Legacy steps only through curated `Effect` atoms; hazard-ratio records end chains. Encoded as ordinary implications they would chain if their target had an out-link. On every shipped stack the record targets (all-cause mortality, CHD, CHF) are sinks. `validate()` raises an error if a request adds a link out of a record target; the fix, if ever needed, is a terminal twin of the target node (not implemented).
* **Empty, not error** for unconnected, unknown and self pairs: the adapter looks the pair up among the accepted answers.
* **Unverified property**: that a mirror pair or a trail never produces a wrong answer is tested on 300 random graphs, not proven.

---------------------------------------------------------------------------------------------------------------------------

## 9. Number table: before and after, by lane

(strength, confidence; `exp10`, `exp14`; "N" = fork minimum, "L" = literal strengths + minimum at p = 0.1)

| item | legacy | P (recommended) | N | L |
|---|---|---|---|---|
| infer D+Q -> CHD | Neg 0.24990 / 0.17017138125 | Neg 0.2499 / 0.17017138125000003 | Neg 0.2499 / 0.65 | Neg 0.28519 / 0.65 |
| rank D+Q, Fisetin, Spermidine (s*c) | 0.04253, 0.03038, 0.01312 | same | 0.16244, 0.11603, 0.08568 | 0.18537, 0.13488, 0.11001 |
| 5-pool order | Metformin .0886, Berberine .0768, DQ .0425, Fisetin .0304, Spermidine .0131 | same | Metformin .198, Berberine .1716, DQ .1624, Fisetin .1160, Spermidine .0857 | same order at p = 0.1; DQ first at p = 0.5 |
| Patient001 personalised | DQ .2306, Fisetin .1647, Metformin .0886, Berberine .0768, Spermidine .0712 | same | DQ .9098, Fisetin .6499, **Spermidine .4799**, Metformin .198, Berberine .1716 | same as N |
| Patient002 personalised | Metformin .648, Berberine .5616, DQ .0425, Fisetin .0304, Spermidine .0131 | same | same order (.9293, .8054, .1624, .1160, .0857) | same order |
| diagnose (CS, CI, MD) | cov 3,2,1; c(CS) 0.6253; mass(CS) 0.4593 | same | same order; c(CS) 0.9571; mass 1.0677 | same as N |
| counterfactual A / B / DQ confidence | 0.65 / 0.3176 / 0.1858 | same | 0.65 / **0.65** / 0.38 | same as N |
| supplement P001, relevance NMN / Fisetin | 0.1689 / 0.1343 | same | **0.268 / 0.534** | **0.276 / 0.568** |
| D+Q -> AllCauseMortality | empty | empty | empty | empty |
| full rule set, same pair | - | Neg 0.0625 / 0.132 (false chain) | - | - |
| feedback edge | hangs | terminates, 109 keys | same | same |
| MultiDrug (4 rows) | 4 derivations | Neg 0.4883 / 0.8178, Pos 0.0151 / 0.8178 | c 0.9087 | - |
| DrugAge ranking, 1,038 compounds | top: c60 fullerene, Rapamycin, Astaxanthin | V2 top: Spermidine, 17-alpha-estradiol, Rapamycin | - | - |

**The 32-case exact-string golden fixture** (`patient_stack_builtin_golden.json`) cannot pass in any lane: in lane P only 36 of 100 pairs are bit-identical (the other 64 differ in the last digit of the strength, e.g. 0.2860000000000001 against 0.28600000000000003); confidences are equal. That is the LAUNCH.md position already (new pins under `pln`), not a finding.

---------------------------------------------------------------------------------------------------------------------------

## 10. Structure

```
vendor/pln_fork/                     (new; the compat study's recommendation, plus one more hunk)
  lib_pln_fork.metta                 bundled lib_pln.metta @ fea83178 (sha256 d864c3cf..)
                                     - hunk 1: drop the per-step println trace (compat's patch)
                                     - hunk 2: structure/deduction_confidence.diff
  LICENSE.NARTECH, LICENSE.PeTTa, PROVENANCE.md   (patham9/PLN 55f1751, PeTTa fea83178, the two hunks)
pln_chat/core/pln_chainer.py         (new, Python; reference implementation = d2lib.py + d2drugage.py)
  collect_links(space)               Effect atoms (evaluate (evidence-confidence T) from the tier table), HR records, DrugAge lifts
                                     via the existing drugage-effect (+ PMID from ReportedIn), caller extra_atoms included
  encode(links, cfg)                 ids, sentences, STV equations, cycle-guard ids, validate()
  derive_source(a)                   !(PLN.Derive ..) in the PeTTa child, budget schedule, acceptance test; cached per source
  adapt(beliefs)                     drop A==C and failed checks, path-cover filter, best per (C, sign), unshift + clamp,
                                     null-companion pruning; builds (PLNAnswer A C Sign (stv s c) (Chain nodes)) atoms
pln_deduction_pln.metta              (new, MeTTa; loaded INSTEAD of the infer/explain part of pln_deduction.metta)
  (= (infer $s $a $c)   (match $s (PLNAnswer $a $c $sgn $tv $chain) (signed $sgn $tv)))
  (= (explain $s $a $c) (match $s (PLNAnswer $a $c $sgn $tv $chain) (Chain $chain (signed $sgn $tv))))
```

Data flow per request: caller facts are loaded into the space -> `collect_links` -> `encode` -> STV equations loaded **once** per process (a second copy would double every Deduction answer) -> `derive_source` for each source -> `(PLNAnswer ..)` atoms added -> the unchanged MeTTa consumers call `infer`/`explain`. This keeps `infer` and `explain` defined as `(= (infer $s ..) ..)` text, the `$space` argument, the output shapes, the lookup of evidence confidence from the tier table, and the direct `Effect` reads (`resolve-lever`, the two lever-composition sites, `kb_effect_markers`). It also keeps variable endpoints working (`(infer &self $x D)` matches the `PLNAnswer` atoms), which legacy supported.

Generated, checked files in `structure/`:
* `patient_updown_kb.metta`: 60 STV equations, the id table as comments, 43 link sentences.
* `derive_DasatinibPlusQuercetin.metta`, `derive_Spermidine.metta`: a self-contained per-source program for a PeTTa spike, with the simulator's expected beliefs as comments. `exp18` parses these files back as data and re-runs the simulator: identical beliefs.
* `deduction_confidence.diff`: the library hunk.
* `evidence_ids.tsv`: id, link, sign, s, c, kind, source line.

---------------------------------------------------------------------------------------------------------------------------

## 11. Cost

Counts, not times (`exp3`, `exp13`, `exp20`): patient graph, all 30 sources: 100 task selections, 1,083 belief pairs, 1,824 `|-` attempts, 43 sentences + 60 equations; the widest source (Spermidine): 10 selections, 19 beliefs, 244 attempts. LinAge2: 86 selections, 1,564 attempts. The Python simulator does all 30 sources in 0.02 s. **PeTTa time: unknown**; each attempt unifies two sentences against two library rules and, on a match, runs `Truth_Deduction` with three `STV` lookups (each a scan over up to 60 clauses unless Prolog indexes deeper than the functor). The spike must measure it. Because the seam derives every source once per request (or once per KB hash in a child), a request costs at most the 1,824 attempts above plus the load of 103 clauses and sentences. The load itself (one KB per process) is the PeTTa constraint of LAUNCH.md.

---------------------------------------------------------------------------------------------------------------------------

## 12. Changes for the consumers and the surrounding code

Signature, shapes and the seven `infer` call sites are unchanged. What does change:

1. **One load surface.** Equations accumulate in a space. Under `pln` the legacy `infer`/`explain` equations (`pln_deduction.metta:132-135, 144-155`) and DrugAge's third `infer` equation (`drugage_calibration.metta:240-243`) must not be loaded next to `pln_deduction_pln.metta`, so those two files are split into a core (types, `sign-product`, `combine-tv`, `link-effect`, calibration) and a chainer part. Under `legacy` nothing changes (hyperon stays green).
2. **Forgery.** `/metta/run` accepts caller ground facts. A caller `(PLNAnswer ..)` would fake an inference result, so the validator must refuse that head in `extra_atoms` and the seam adds its atoms afterwards.
3. **Multiplicity.** Edge-disjoint parallel evidence (DrugAge rows, duplicate provenance, curated Effect plus record with the same sign, disjoint diamonds) now gives one revised answer instead of several identical or near-identical ones. Consumers that summed or counted derivations (`patient-relevance`, `explains-obs` coverage and mass) stop double counting there; on the shipped graphs there is no such pair, so nothing moves. A DrugAge compound has up to two answers (Neg and Pos) instead of one per row; `run_drugage_ranking`'s dedupe by best score keeps working.
4. **DrugAge ranking** changes by design (section 7): `ontology/drugage_selector.py` picks one representative per compound and PMID instead of one per compound, and the pins `Resveratrol/Metformin = 0` need new values. The Python twin `drugage_scoring.py` must follow, or the `linear` and `metta_sort` strategies stop agreeing (`test_linear_and_metta_sort_agree_on_every_score`).
5. **Confidence lane.** Lane P: no consumer assertion changes. Lane N: the 12 assertions of section 4 need new values, the hand-composed sites need the same rule or a note, and `risk-conf-discount` (a separate literal 0.9) stays unconnected as today.
6. **Cycles.** `infer` terminates. New pairs appear when a cycle is added (section 8); nothing checks acyclicity at write time, so the decision is whether to keep allowing it.
7. **Order of results** inside one call changes (legacy's is an interpreter artefact). The adapter returns Pos before Neg, each at most once per pair. No shipped pair has two. `pos-transmission`'s "first Pos derivation" is therefore unambiguous.
8. **Python.** `_SCORED_RE` / `_STV_RE` keep working: numbers are plain decimals (clamped to [0,1], so `-1e-17` cannot occur). `explain` becomes available for DrugAge compounds at no cost (Compound -> Lifespan -> Mortality from the stamp).
9. **Exact goldens** are regenerated under `PLN_CHAINER=pln` only (section 9).

---------------------------------------------------------------------------------------------------------------------------

## 13. Risks and unknowns

* **Not run on PeTTa.** The simulator is validated against the fork's 14 pins and PeTTa's six; the PeTTa-specific points are unverified: that `(= (STV (Concept X-up)) ..)` composes with the forward declaration `(= (STV $stv) (empty))` (it should: the empty branch yields nothing), `Unique` as `list_to_set`, the `Concept` and `Implication` heads (neither is a PeTTa builtin; grep of `src/*.pl`), and that `PLN.Derive`'s printed `($Tasks $Beliefs)` can be read back reliably. The two generated `derive_*.metta` files are the spike's first test.
* **The confidence patch** is a change to third-party code (decision a). The N lane exists to show the unpatched numbers.
* **Revision overconfidence.** Revised confidence rises toward 1 with every independent row (resveratrol 0.99; MultiDrug 0.909). The independence unit (PMID) is a judgement; ceilings (V3, V5) are available. The dependence of rows within a paper, and of papers on each other, is not modelled.
* **Symmetric channel.** Mirror strengths equal the primary's; separate magnitudes for decreasing and increasing a node have no legacy source.
* **Dense or cyclic caller graphs.** The number of simple paths is exponential, as in legacy. The guard turns a runaway into an HTTP failure after the 16,384 budget; I measured 0 of 447 sources failing at 1,500 selections with the guard ids on random graphs of 6 to 9 nodes, not on larger ones.
* **Shared-link diamonds** return the more confident path, not a merge, and legacy returned both; harmless on the shipped graphs, a visible difference on a caller's.
* **Record targets with out-links** raise an error (section 8).
* **The DrugAge CSV** is a proxy for the ETL build (the ETL drops some rows on dose parsing); rankings are indicative of the effect, not a re-pin.
* **Patient marker sets** used in the consumer replays (P001 = CRP, DNAmGDF15, DNAmPAI1; P002 = FastingGlucose, HbA1c) were recovered by exact match on the pinned relevances (zero error), not read from the patient fixtures.
* **What I replayed and what I did not.** Replayed (patient stack, 29 checks derived from the pins, plus the 100 + 86 pair closures): `infer` values, rankings, personalised scores, relevance, diagnosis, counterfactual, supplement relevance and targets, the DrugAge fixtures. Not replayed: the risk layer (it reuses the counterfactual, so lane P is identical and lane N follows `A13`), the LinAge2 consumers (their `infer` inputs are identical in lane P: 86 of 86 pairs; lane N changes their confidences), the HTTP and Gradio tests, the 32 golden cases, `test_drugage_ranking_contract.py` (skipped without the ETL build).
* **Float noise** in the last digit of strengths (<= 5e-16) is inherent to shift/unshift; exact goldens cannot be met (section 9).
* Unknown: PeTTa speed; whether `chain-discount` and `PLN.Config.HopDiscount` can be bound to one definition across the two files; whether any external client relies on legacy's order of derivations or on `explain` being empty for DrugAge.

---------------------------------------------------------------------------------------------------------------------------

## 14. Effort and order of work

About **6 agent-phases**: (1) PeTTa spike of the vendored lib with hunk 2 and the generated `derive_*.metta`, reading the beliefs back, measuring time; (2) `collect_links` + `encode` + `validate` + ids, including the DrugAge lift and PMID; (3) derive/adapt/`PLNAnswer` atoms, `pln_deduction_pln.metta`, the two-file split, validator refusal of `PLNAnswer`; (4) the number table under `pln` (lane P and lane N) and the new pins for DrugAge; (5) DrugAge selector/ranking/twin change; (6) cycle, diamond and property tests, docs. Phases 2 and 3 can run in parallel once 1 passes its gate (the generated programs produce the simulator's beliefs on PeTTa).

---------------------------------------------------------------------------------------------------------------------------

## 15. Reproduce

`cd` to this folder and run `sh run_all.sh > numbers.txt` (about 4.5 minutes, Python 3 standard library only, no network, nothing from `.petta/` is executed; `gen_structure.py` reads the bundled `lib_pln.metta` as text to compute the diff). Section to script: 2 `exp1`, `exp8`; 3 `exp12`, `exp23`; 4 `exp8`, `exp9`, `exp10`, `exp20`; 5 `exp19`, `exp24`; 6 `exp2`, `exp3`, `exp13`, `exp16`, `exp17`; 7 `exp5`, `exp6`, `exp14`, `exp21`, `exp22`; 8 `exp4`, `exp11`, `exp15`; 10 `gen_structure.py`, `exp18`.
