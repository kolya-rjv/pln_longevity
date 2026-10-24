# Design 3: legacy `infer` stays the engine, the fork enters only where legacy cannot go

Written 2026-10-24 for the PeTTa migration (work tree `pln_longevity_petta`, branch `petta-migration`, content = linage2 @ 47130aa).
Owner's choice: the fork (patham9/PLN @ 55f1751, via PeTTa's bundled `lib_pln.metta` @ fea83178). Restricted rules by default, full rules only as a comparison switch,
`PLN_CHAINER=legacy` stays the default, hyperon stays green.

Everything below was computed, not guessed. `compute_numbers.py` produces every number (log: `numbers_log.txt`, data: `numbers_out.json`); `seam_ref.py` is the executable spec of the seam;
`verify_hooks_worker.py` and `verify_revision_worker.py` ran the legacy side on **real hyperon 0.2.10**. Nothing from `.petta/` was executed, imported or copied; the repo was only read.
What was NOT run: PeTTa, and therefore every line of `pln_fork_seam.sketch.metta` (section 12 lists exactly what is verified and what is not).

---------------------------------------------------------------------------------------------------------------------------

## 0. Verdict

* The seam is three functions that call into the fork: `fork-revise-items` (calls `Truth_Revision`), `fork-closure` (calls `PLN.Derive`), `fork-second-opinion` (reads one `PLN.Derive` run with `PLN.Query`'s selection rule).
  Everything else (every pinned number, every ordering, every one of the 186 closure derivations) is still produced by the unchanged legacy equations.
* **Zero lines of the repo's `.metta` files change.** The legacy stack loads `pln_deduction.metta` byte for byte. The hybrid stack loads a text generated from it by an exact-string rename plus a dispatcher
  (30-line diff, `pln_deduction_hooks.diff`), asserted to match exactly once.
* On the shipped graphs hybrid is **bit-identical to legacy**: `infer` and `explain` strings of all 100 patient pairs and all 86 LinAge2 pairs equal the recorded real-hyperon output (section 12).
* What changes, and only this: (a) a DrugAge compound with several rows gets one revised truth value per direction instead of one derivation per row or one representative row (section 3);
  (b) a source whose forward cone contains a feedback cycle gets an answer instead of a hang (section 4); (c) a new side-channel call shows the fork's own answer for any pair (section 5).
* **Honest answer to Ben's request:** this design mostly adds a revision step plus a safety net. It does run the fork's priority-queue control, but only on cyclic sources and on request; on the shipped network the control is never binding
  (all 100 pairs in at most 11 selections, one derivation each). The fork's own arithmetic cannot be the engine without re-pinning: it changes the pinned Patient001 personalised order (section 7.4). Section 11 gives the full reasoning.

---------------------------------------------------------------------------------------------------------------------------

## 1. Dispatch: which engine answers what

| case | detected by | engine | arithmetic of the returned `(stv s c)` | pinned numbers |
|---|---|---|---|---|
| acyclic cone, at most one derivation per link (every shipped query) | no `(CycleReach A)` mark | legacy `infer-legacy` / `explain-legacy` (the unchanged equations) | legacy | identical, bit for bit |
| DrugAge compound with 2 or more rows | Python slice builder, before the run | revision of evidence items with the fork's `Truth_Revision`, then the legacy chain over ordinary `Effect` atoms | legacy chain arithmetic over the revised link | identical for 1-row compounds; changed for multi-row compounds |
| a directed cycle lies in the forward cone of A | `(CycleReach A)` mark written by a Python pre-flight (Tarjan over the Effect atoms, KB plus caller `extra_atoms`) | fork closure: `PLN.Derive`, tasks = start fact, beliefs = encoded cone | legacy fold along the trail the fork found (default), or the fork's native `(stv s c)` (`PLN_FORK_NUMBERS=native`) | none exist (legacy hangs) |
| any pair, on request | explicit call | `fork-second-opinion`, same closure | the fork's native arithmetic, in a separate result type | none (new) |

Rule for mixing: **numbers from the two arithmetics never appear in one result set.** The fork's own arithmetic differs from legacy (section 7.2), so it only ever travels in the `ForkOpinion` type, never inside `(signed ..)`.

---------------------------------------------------------------------------------------------------------------------------

## 2. The seam

### 2.1 Pieces

| piece | where | runs on | role |
|---|---|---|---|
| `PLN_CHAINER=legacy\|hybrid` (`pln` stays reserved for a full replacement) | env, read by the stack loader | - | `legacy` default: nothing below is loaded |
| hooked `pln_deduction` text | generated at stack assembly by `hooks.hooked_deduction()`; the diff is `pln_deduction_hooks.diff` | hyperon and PeTTa | rename `infer`/`explain` to `infer-legacy`/`explain-legacy` (bodies untouched, recursion renamed) and add a dispatcher |
| `pln_hybrid_guard.metta` (`hooks.GUARD`, 4 lines) | hybrid stack only | both | `(chain-open? $s $a)` = no `(CycleReach $a)` fact in the space |
| `graph_guard.py` (new, about 40 lines; executable twin `seam_ref.reaches_cycle`) | Python, once per request | Python | parse the Effect atoms of the KB files and of `extra_atoms` (the same line-start text form `kb_effect_markers` already relies on, `patient_builder.py:305-327`), Tarjan, write `(CycleReach X)` for every node whose cone has a cycle |
| `vendor/pln_fork/lib_pln_fork.metta` | PeTTa bundled `lib_pln.metta` @ fea83178 (sha256 d864c3cf...) plus the 1-hunk patch that drops the per-step stdout trace (`compat/artifacts/bundled_B0_to_B1.patch`), with `LICENSE.NARTECH`, `LICENSE.PeTTa`, `PROVENANCE.md` | PeTTa | the fork's formulas and `PLN.Derive` |
| `pln_fork_seam.metta` (sketch: `pln_fork_seam.sketch.metta`) | hybrid stack, PeTTa only | PeTTa | `fork-revise-items`, `fork-closure`, `fork-trails`, `infer-cyclic`, `explain-cyclic`, `fork-second-opinion` |
| stub (`hooks.STUB`) | hybrid stack on hyperon | hyperon | `(infer-cyclic ..)` and `(explain-cyclic ..)` return `(CyclicCone A)`: an explicit marker instead of a hang, which the HTTP layer maps to a failure status |
| `drugage_revision.py` (new) | Python | Python, plus a PeTTa child for the fork formula | evidence items from rows, revision, emits revised `Effect` atoms and `(RevisedFrom ..)` provenance atoms |

The cycle probe is deliberately NOT in MeTTa: a first version written in MeTTa (path DFS) took 91 s for the hyperon pair sweep against 9 s, and its `==` raised `BadArgType` on symbols of different declared types.
A lookup of a precomputed mark costs about 2 ms per `infer` call on hyperon (+28 % patient, +31 % LinAge2 on the full 900-query sweep; hybrid only, legacy mode is untouched).

### 2.2 Sentences built from Effect links and rows

The fork reasons over `(Sentence (<term> (stv s c)) (<evidence ids>))`. One query builds:

* **The start fact** `(Sentence ((Up A) (stv 1.0 1.0)) (0))`: "A is raised", certain, stamp 0. Confidence 1.0 is safe here because the restricted rules never revise it (Revision divides by `1-c`).
* **One implication per link and source polarity.** A link `(Effect x y Pos|Neg (stv s c))` gets id `k` (its position in the collapsed link list, 1-based) and two sentences with the same stamp `(k)`:

  | link | sentence 1 | sentence 2 |
  |---|---|---|
  | `Pos`, strength s, confidence c | `(Implication (Up x) (Up y))` `(stv s c)` | `(Implication (Down x) (Down y))` `(stv s c)` |
  | `Neg` | `(Implication (Up x) (Down y))` `(stv s c)` | `(Implication (Down x) (Up y))` `(stv s c)` |
  | hazard-ratio record lift (terminal) | same, but the consequent is wrapped: `(End (Up y))` / `(End (Down y))` | |

  Both polarity copies carry the same id, so one derivation can use one link only once (stamp overlap blocks the second use).
  The `End` wrapper makes a record terminal exactly as in legacy (`link-effect` eq. 2 may end a chain, `infer` eq. 2 never steps through it): no Effect implication has an `End` term as premise.
  Strength and confidence are the legacy link values: `(stv s c)` of the Effect atom with `(evidence-confidence T)` already reduced (whatever Phase 2 of LAUNCH.md decides for the 25 unevaluated confidences), and for a record `effect->strength` and the tier confidence.
* **Only the forward cone of A** is encoded (`seam_ref.cone`; with a target C, intersected with the backward cone of C). This matters: the belief queue must hold the whole KB (section 6), and the cone is 21 sentences at most on the shipped graph against 57 for the whole graph.
* **DrugAge rows are not Sentences.** They become evidence items `(Ev id sign s c)` and are folded with `Truth_Revision` (section 3). Making each row an implication sentence and letting the Deriver's Revision rule merge them works for up to 12 rows at the default budget and then collapses (section 7.8), so it cannot handle 170 rows.

### 2.3 Sign handling

* A belief about `(Up C)` is a `Pos` conclusion ("A raises C"), `(Down C)` is `Neg` ("A lowers C"), `(End (Up C))` and `(End (Down C))` the same for a record-ended chain. The start fact is always `(Up A)`.
* Because a `Neg` link flips the polarity, `Neg * Neg = Pos` comes out of the encoding itself; the decoder asserts that the encoded sign equals the legacy sign product on every trail. It did on all 186 closure derivations and all 109 cyclic ones.
* The fork never builds a `(Not B)` sentence for this: the Translator's way (`A -> (Not B)` yields `B` with `1 - MP`) would turn "protective with strength 0.2" into "B up with strength 0.8" and destroy the magnitude. That is why the up/down split is used (LAUNCH.md already requires it).
* DrugAge keeps the legacy sign convention (`lifespan-sign`: `pct < 0` is `Neg` at Lifespan, `pct >= 0` is `Pos`, so a 0.0 row is `Pos` with strength 0). Revision runs per Lifespan direction; the existing `Lifespan -> Mortality` hop flips the sign afterwards.

### 2.4 Back into the legacy shapes

* `infer-cyclic` returns `(signed Pos|Neg (stv s c))`, `explain-cyclic` returns `(Chain (A n1 .. C) (signed ..))`, one per node-simple trail. The trail is rebuilt from the stamp (the edge ids minus 0): it must be exactly one node-simple path starting at A, otherwise the belief is a loop echo and is dropped.
* The numbers: `legacy` mode folds the legacy arithmetic along the rebuilt path, right-nested, `s = s_ab * s_rest`, `c = (c_ab * c_rest) * 0.9`, sign product; one link takes no discount. `native` mode copies the fork's `(stv s c)` from the belief.
* `explain`'s tv equals `infer`'s tv by construction (same trail, same fold).
* Truncation (the step bound hit with tasks left, or a queue removal) is reported through `(fork-closure-status &self A)` (`complete | truncated`), never as an extra atom in `infer`'s stream: an extra atom would be unreduced junk inside `rank-score`, `score-candidate` and the Python regexes (`pln_runner.py:415-420, 588-591`).
* Empty stays empty: an unconnected or unknown pair has no trail.

### 2.5 The hooks (exact)

`pln_deduction_hooks.diff` (the repo file is not edited; this is the transformation applied at stack assembly):

```
-(= (infer $s $a $c) (link-effect $s $a $c))
+;; --- hybrid chainer hook (Design 3): a rename plus a dispatcher; the two legacy equations are untouched ---
 (= (infer $s $a $c)
+   (if (chain-open? $s $a) (infer-legacy $s $a $c) (infer-cyclic $s $a $c)))
+(= (infer-legacy $s $a $c) (link-effect $s $a $c))
+(= (infer-legacy $s $a $c)
    (match $s (Effect $a $b $sgn $tv)
-          (combine-hop $sgn $tv (infer $s $b $c))))
+          (combine-hop $sgn $tv (infer-legacy $s $b $c))))
 ... explain: the same rename for its three equations (pln_deduction.metta:144-151) ...
```
`infer` and `explain` keep their names and three-argument signatures; the seven call sites (`patient_profile.metta:355,390`, `pln_abductive_diagnosis.metta:44`, `pln_intervention_ranking.metta:62`, `pln_counterfactual.metta:108,126`, `pln_linage2.metta:123`) are untouched.
The DrugAge third equation (`drugage_calibration.metta:240-243`) stays an additional equation of the public `infer`. `pln_deduction.metta` is pasted into the LLM prompt only as a header (`test_prompt_size.py`): the hook sits below the header.

---------------------------------------------------------------------------------------------------------------------------

## 3. Revision: several rows, one truth value per direction

### 3.1 The step

1. Python reads the rows of the requested compounds (no `best_per_compound`, no `MAX_ROWS` cap: the revised slice holds two `Effect` atoms per compound, so the hyperon panic threshold no longer applies to raw rows).
2. Each row is lifted exactly as `drugage_calibration.metta` sections 4-7 do, at the Lifespan level: sign `Neg if pct < 0 else Pos`, strength `|pct|/(|pct|+20)`, confidence `tier * gate`. The Python twin `drugage_scoring.score_row` already equals the engine bit for bit; the six synthetic rows below reproduce on real hyperon.
3. Rows become evidence items `(Ev id sign s c)`. **The evidence id is the paper** (`PMID_n`, parsed to the number n; a row with no PMID gets `1000000 + row index`). Items with the same id and direction are one piece of evidence; only the most confident enters (ties: the lower median strength, the selector's own `REPRESENTATIVE_POLICY` rule 4). This is the fork's stamp discipline: `StampDisjoint` refuses to merge beliefs that share an id.
4. Per direction the fork's `Truth_Revision` is folded left in ascending id (`bundled lib:181-189`: `w = c/(1-c)`, `f = sum(w f)/sum(w)`, `c = W/(W+1)`). A group of one item is returned unchanged, with no `Truth_Revision` call, so every single-row compound keeps its exact legacy numbers.
5. Python emits `(Effect <compound> Lifespan <Pos|Neg> (stv s c))` per direction and `(RevisedFrom <compound> Lifespan <sign> (<ids>) n)` as provenance. The legacy chain does the rest: `infer <compound> Mortality` steps through that Effect and the existing `(Effect Lifespan Mortality Neg (stv 1.0 1.0))`.

The fold runs in a PeTTa child (`fork-revise-items`, the fork's own formula) when the PeTTa runtime is available, otherwise in the bit-identical Python twin; a parity test pins them to each other. The whole-build discovery ranking (1043 compounds, impossible through the engine at 70 ms per call) needs the twin anyway.

### 3.2 Why the paper, not the row, is the evidence id (LAUNCH.md says the row; this design departs from it, with numbers)

| | rows | papers | today (one representative row) | row-level ids | paper-level ids |
|---|---|---|---|---|---|
| Ethanol | 52 | **1** | +0.1432, rank 259 | **+0.5679, rank 1** | +0.1284, rank 372 |
| Resveratrol | 170 | 37 | 0.0000, rank 720 | -0.0272, rank 839 | -0.0395, rank 886 |
| Metformin | 54 | 16 | 0.0000, rank 713 | -0.0518, rank 884 | -0.0403, rank 887 |
| Rapamycin | 37 | 18 | +0.3191, rank 2 | +0.3373, rank 28 | +0.3167, rank 5 |
| Spermidine | 8 | - | +0.2455, rank 15 | +0.4075, rank 8 | +0.3398, rank 1 |

Scores are the net rank-score at Mortality (section 3.4). Row-level ids treat 52 arms of one paper as 52 independent experiments: pooled confidence 0.84 and Ethanol becomes the best life-extender in the build. Paper-level ids keep Ethanol at its one paper (confidence 0.189).
Row-level remains available as `PLN_FORK_EVIDENCE_ID=row`; it is not recommended.

### 3.3 Worked example: a six-row compound (synthetic, in the style of the contract's `MultiDrug` probe)

| row | species | ITP | significance | pct | paper | legacy derivation at Mortality (real hyperon) |
|---|---|---|---|---|---|---|
| SynRow_0 | Mus_musculus | yes | Significant | +20 | 1001 | `(signed Neg (stv 0.5 0.81))` |
| SynRow_1 | C. elegans | no | Significant | +40 | 1002 | `(signed Neg (stv 0.6666666666666666 0.315))` |
| SynRow_2 | D. melanogaster | no | Significant | +25 | 1003 | `(signed Neg (stv 0.5555555555555556 0.315))` |
| SynRow_3 | C. elegans | no | Significant | +35 | 1002 | `(signed Neg (stv 0.6363636363636364 0.315))` |
| SynRow_4 | Mus_musculus | no | NotSignificant | 0 | 1004 | `(signed Neg (stv 0.0 0.18000000000000002))` |
| SynRow_5 | Mus_musculus | no | NotSignificant | -30 | 1005 | `(signed Pos (stv 0.6 0.18000000000000002))` |

Today: six derivations through `infer`, or the single row SynRow_0 through `best_per_compound` (score +0.405).
Paper-level revision, by hand (the Pos-at-Lifespan group = extension plus the null; SynRow_1 and SynRow_3 are one paper and collapse to SynRow_3):

```
paper 1001: s 0.500000, c 0.9   -> w = 0.9/0.1   = 9.000000
paper 1002: s 0.636364, c 0.35  -> w = 0.35/0.65 = 0.538462
paper 1003: s 0.555556, c 0.35  -> w = 0.538462
paper 1004: s 0.000000, c 0.2   -> w = 0.2/0.8   = 0.250000
W = 10.326923     f = (9*0.5 + 0.538462*0.636364 + 0.538462*0.555556 + 0)/W = 5.141802/10.326923 = 0.497903
c = W/(W+1) = 0.911715        at Mortality: (0.497903*1.0, (0.911715*1.0)*0.9) = (0.497903, 0.820543)
```
The shortening group has one item, so it passes through: `(signed Pos (stv 0.6 0.18000000000000002))`. Real hyperon, loading the revised atoms into the DrugAge stack, returns exactly these two results from `infer` (bit-identical to the Python numbers), two `Chain (MultiDrug Lifespan Mortality)` results from `explain` (legacy `explain` has no DrugAge equation and returns nothing for a raw-row compound), and `rank-interventions` lists the compound twice: `+0.40855` and `-0.108`.
Row-level ids give `(Neg 0.506266 0.824149)` and the same Pos group. Net scores: today +0.405, paper-level +0.300551, row-level +0.309239.

### 3.4 Real data (the repo's `data/drugage/dataset.zip`, 3423 rows, parsed with the ETL's mapping)

* Resveratrol, 170 rows, 37 papers. Today: the representative row `DrugAgeRow_1089` is an ITP null, score 0.0. Paper-level: extension group n=35 papers, lifespan (s 0.2454, c 0.9712), at Mortality `(Neg 0.2454 0.8740)`, score +0.2145;
  shortening group n=8, `(Pos 0.3798 0.6688)`, score -0.2540; net -0.0395. Row-level: n=112 `(Neg 0.1841 0.8899)` and n=58 `(Pos 0.2283 0.8370)`, net -0.0272.
* Metformin, 54 rows, 16 papers. Today 0.0 (ITP null). Paper-level: `(Neg 0.2420 0.8492)` n=14 and `(Pos 0.3514 0.6995)` n=6, net -0.0403.
* Rapamycin, 37 rows, 18 papers. Today `(Neg 0.3939 0.81)`, score +0.3191. Paper-level `(Neg 0.3598 0.8801)`, score +0.3167.
* Whole build: 1035 scorable compounds, 592 with more than one scorable row. Spearman rank correlation today against paper-level 0.917 (row-level 0.891). Top-10 overlap 2, top-20 overlap 7, top-50 overlap 18; 167 compounds move more than 100 places (row-level: 469).
  Paper-level top 5: Spermidine 0.3398, Reserpine 0.3281, C60_fullerene 0.3273, Pineal_gland_extract 0.3201, Rapamycin 0.3167. Today: C60_fullerene 0.3682, Rapamycin 0.3191, Astaxanthin 0.3038, N_17alphaestradiol 0.3038, Indolepropionamide 0.2898.

### 3.5 What revision does not fix (stated, not hidden)

* It averages a magnitude (`|pct|/(|pct|+20)`) as if it were a probability and drives confidence toward 1 as evidence accumulates (0.97 for 35 papers). That is PLN's behaviour for independent evidence; whether DrugAge papers are independent is the curator's call. The id granularity is the lever.
* A reported null and a reported effect sit in the same direction group (a 0.0 row is `Pos`), which is the right way for null evidence to pull the pooled strength down, but it means the extension group of a null-heavy compound shows a small strength with a large confidence.
* Two directions are two derivations. `rank-interventions` lists the compound twice, Python's `by_compound` keeps the better one today. Hybrid needs a net score (section 10).
* Duplicate provenance (a raw row plus a curated `(Effect X Lifespan ..)` derived from it, the Taurine case) is still two identical derivations; revising them would double count. They need a shared evidence id, which the curated atom does not carry.

---------------------------------------------------------------------------------------------------------------------------

## 4. The feedback loop

The probe edge is `(Effect ChronicInflammation CellularSenescence Pos (stv 0.6 (evidence-confidence AnimalStudies_Single)))`, closing `CellularSenescence -> SASP -> ChronicInflammation -> CellularSenescence`.

* Legacy: `infer D+Q CHD` does not terminate (the Python reference raises `RecursionError`; hyperon runs past 60 s, the existing process deadline kills it).
* The pre-flight marks 9 of 30 sources whose cone contains the cycle: Autophagy, CellularSenescence, ChronicInflammation, DasatinibPlusQuercetin, Fisetin, Omega3, Resveratrol, SASP, Spermidine. The other 21 are unmarked and take the legacy path.
  On real hyperon with the marks: the 30 pairs from unmarked sources are identical to the acyclic legacy output; a marked source answers `(CyclicCone A)` (no fork on hyperon). With the fork on PeTTa the closure answers instead.
* **The fork's closure terminates on its own, for two reasons:** the step bound, and the stamp rule. A trail that re-enters the cycle must reuse a link already in its stamp, and `StampDisjoint` refuses that pairing. From D+Q the closure stops after 11 selections and ends with a loop echo `(Down CellularSenescence) (0.3209, 0.1373)` stamp (0 1 3 4 25) that cannot be extended because link 1 is already used. Over all 30 sources the maximum is 15 selections (measured with a bound of 60), so the recommended bound of 40 is never reached.
* The answers: 109 connected pairs, equal to plain graph reachability; 9 of them are new pairs created by the edge itself (for example `ChronicInflammation -> CHD`, Pos `(0.2142, 0.130901)`, empty before the edge).
  On every pair the fork's node-simple trails equal what legacy would return with a visited-path guard (109 derivations, 0 mismatches); the decoder dropped 19 loop-echo beliefs.
* `D+Q -> CHD` on the cyclic graph: `(signed Neg (stv 0.24989999999999998 0.17017138125000003))`, the same value as the acyclic pinned number, because the trail is the same four-edge path and the legacy fold is applied to it. Native: `(0.26863439999999994, 0.23343125000000003)`.
  Fisetin, Spermidine, CellularSenescence and SASP to CHD also keep their acyclic values; Metformin (unmarked) is answered by legacy.
* Order of work (confidence descending, ties first-listed): D+Q, CellularSenescence, SASP, DNAmPAI1, ChronicInflammation, End(CHD), CRP, DNAmGDF15, RDW, LowSerumAlbumin, then the loop echo. CHD is derived on selection 4. With a step bound of 3 it is not found and the result is flagged `truncated`; from 4 on it is found.
* Toy cases from the contract: a self-loop `A -> A` terminates in 2 selections; `A<->B, B->C` returns `A -> C` as `(Pos 0.81 0.576)` in 4.

**Honest note:** a visited-path guard in the legacy recursion (a few lines; not written or run here, but its node-simple derivations are exactly what the 109-derivation comparison used) would give the same answers at lower cost. The fork's added value on cycles is the budget (best-first, bounded, anytime), not termination. If only termination is wanted, that guard does it on both runtimes.

---------------------------------------------------------------------------------------------------------------------------

## 5. Query as a second opinion

`(fork-second-opinion &self A C)` runs the same closure and reads the best belief per target state with `PLN.Query`'s own rule (highest confidence, first of equals). It returns `(ForkOpinion <state> (stv s c) (Stamp ids) (Truncated bool))` for `(Up C)`, `(Down C)`, `(End (Up C))`, `(End (Down C))`. It is a separate result type, not part of any consumer's stream, and carries the fork's native arithmetic.
It is what lets the owner show Ben the fork run on the real network: the selection order, the stamps (evidence trail), the native numbers next to the legacy ones. Recommended surface: one optional endpoint; the validator and inventory lists get the new name.

The native numbers (section 7.2) differ systematically from legacy and are never substituted for it.

---------------------------------------------------------------------------------------------------------------------------

## 6. Rule set, base rates, control settings

**Rule set (restricted):** Revision, Modus Ponens (NotElimination is inert: no `Not` term is ever built; no Translator, no engraved rules). No Deduction, Induction, Abduction, inversion, equivalence, similarity or member rules.

**Base rates: none.** Revision, Modus Ponens and the engraved rules never call `(STV X)` (fork facts section B). Deduction would need a probability for every `Up`, `Down` and `End` term; the comparison runs below use a uniform `(stv 0.2 0.9)`.

**Control:** tasks = the start fact only (the 6-argument `PLN.Query`/`PLN.Derive` form), beliefs = the encoded cone.
Seeding every KB sentence as a task (the 2-argument form `PLN.Query kb term`) is wrong for this graph: it spends selections on edges. Measured on the patient graph with the restricted rules (section 7.6): at the fork defaults 20/20/200 it loses 12 of 100 pairs with the cone as KB and 48 with the whole graph; it needs 60/40/400 to be complete.

| setting | recommended | why (measured) |
|---|---|---|
| steps | 40 | shipped closure complete from 6; cyclic probe needs at most 15; 40 leaves a margin of 2.7x |
| task queue | 20 | complete from 3 |
| belief queue | `max(200, 2*|KB| + 100)` | **must exceed the KB**: `LimitSize` keeps bq-1 items and drops the least confident first. Shipped: complete from 32, peak KB 21, peak final beliefs 31. A 400-node synthetic cone of 1273 sentences at bq=200 found 1 of 339 states |
| start fact | `(stv 1.0 1.0)`, stamp `(0)` | first selected, shares id 0 with every derivation, so no path revision |
| numbers | `legacy` | section 1 |
| Unique | either | the closure is identical under PeTTa's order-keeping `list_to_set` and the fork's reversing `Unique` (checked on all 30 sources) |

Fork defaults (20/20/200) and the bundled lib's (100/10/100) are both complete on the shipped graph with this seeding.

---------------------------------------------------------------------------------------------------------------------------

## 7. Numbers

### 7.1 Pinned chains: legacy, hybrid, and the fork's own arithmetic

Hybrid equals legacy exactly (the legacy equations run). "Fork native" is what `fork-second-opinion` shows (restricted rules, start `(1.0 1.0)`).

| query | legacy = hybrid | fork native |
|---|---|---|
| `infer D+Q CHD` (4 edges) | Neg (0.24989999999999998, 0.17017138125000003) | Neg (0.26863439999999994, 0.23343125000000003) |
| `infer CellularSenescence CHD` (3) | Pos (0.357, 0.29089125000000005) | Pos (0.36683999999999994, 0.359125) |
| `infer Metformin CHD` (3) | Neg (0.33000000000000007, 0.268515) | Neg (0.3406500000000001, 0.33149999999999996) |
| `infer Spermidine CHD` (5) | Neg (0.17135999999999998, 0.07657712156250002) | Neg (0.197926368, 0.11671562500000002) |
| `infer Fisetin CHD` (4) | Neg (0.23205, 0.13090106250000003) | Neg (0.2522668, 0.1795625) |
| `infer Berberine CHD` (3) | Neg (0.28600000000000003, 0.268515) | Neg (0.2993100000000001, 0.33149999999999996) |
| `infer SASP CHD` (2) | Pos (0.42, 0.49725) | Pos (0.426, 0.5525) |
| `infer D+Q AllCauseMortality` | empty | empty (the cone slice is one sentence) |

### 7.2 Why the fork's arithmetic differs (and by how much)

Modus Ponens is `(f1 f2 + 0.02 (1 - f1), c1 c2)` (`Formulas.metta:46-48`): a leak of `0.02 (1 - f1)` per hop after the first, and no per-hop discount. Legacy is `(s1 s2, (c1 c2) 0.9)`. Over all pairs on the patient graph (LinAge2 is the same):

| edges | pairs | largest strength excess | confidence ratio fork/legacy |
|---|---|---|---|
| 1 | 28 | 0 (bit-identical) | 1 |
| 2 | 28 | 0.0122 (3.9 %) | 1.1111 = 1/0.9 |
| 3 | 17 | 0.0198 (6.2 %) | 1.2346 = 1/0.9^2 |
| 4 | 22 | 0.0282 (13.8 %) | 1.3717 = 1/0.9^3 |
| 5 | 5 | 0.0312 (15.5 %) | 1.5242 = 1/0.9^4 |

The leak also breaks a legacy invariant ("strength never exceeds the weakest link") for very weak links: s1 = 0.1 then s2 = 0.01 gives 0.019 against a weakest link of 0.01. No shipped chain reaches that.

### 7.3 The closure equals the legacy closure

Restricted rules, tasks = start fact, cone KB: **100 of 100 patient pairs and 86 of 86 LinAge2 pairs** are found, one derivation each, signs equal, and `(sign, s, c)` is **bit-identical to hyperon's** `closure.json` under the legacy fold. Max 11 selections, max KB 21 sentences, 18 ms for all 30 sources in the Python simulator.

### 7.4 Orderings

| ordering (pinned by) | legacy = hybrid | fork native |
|---|---|---|
| D+Q > Fisetin > Spermidine, Elamipretide absent (`test_rank_interventions_orders_by_protection`) | holds: 0.042526, 0.030376, 0.013122 | holds: 0.06271, 0.04530, 0.02310 |
| Metformin 0.0886 > Berberine 0.0768 > D+Q > Fisetin > Spermidine | holds | holds |
| Patient002 [Metformin, Berberine, DQ, Fisetin, Spermidine] | holds | holds |
| **Patient001 [DQ, Fisetin, Metformin, Berberine, Spermidine]** (`test_personalization_reorders_across_patients`) | holds (DQ 0.2306046747621094 reproduced) | **breaks**: [DQ 0.3182, Fisetin 0.2292, **Spermidine 0.1154**, Metformin 0.1129, Berberine 0.0992] |

The fork's undiscounted confidence favours the 5-edge Spermidine chain (x1.52). That is why the fork's arithmetic cannot become the engine without re-pinning, and why hybrid never returns it as `(signed ..)`.
Diagnose coverage (3, 2, 1) is a count over the same pair set, so it cannot move.

### 7.5 No false chain

* Legacy and hybrid: unchanged; `D+Q -> AllCauseMortality`, `SmokingCessation -> CHD`, `Rapamycin -> CHD`, `Elamipretide -> *` stay empty (the closure produces exactly the 100 and 86 pairs).
* Restricted closure: 0 false (node, sign) states on both graphs, with the cone or with the whole graph as KB.
* Comparison switch (patient graph, 30 sources, uniform `(stv 0.2 0.9)` base rates, budgets 20/20/200 and 60/40/400):

| rule set, KB, seeding | false states | legitimate pairs lost | best value differs from restricted |
|---|---|---|---|
| restricted, cone, start fact | 0 | 0 | - |
| restricted, cone, every sentence a task | 0 | 12 at 20/20/200; 0 at 60/40/400 | 0 of 88 |
| restricted+Deduction, cone, every sentence a task | 0 | 19 at 20/20/200; 2 at 60/40/400 | 45 of 81; 68 of 98 |
| full (17 library rules), cone, every sentence a task | 0 | 28; 10 | 41 of 72; 60 of 90 |
| **full, whole graph, every sentence a task** | **6** | 60; 41 | 2 of 40; 22 of 59 |

The 6 false states on the whole graph, for example `DNAmPAI1 -> AllCauseMortality`, `DNAmPAI1 -> CongestiveHeartFailure`, `DNAmPAI1 <-> AgeAccelGrim`, `Berberine <-> Metformin`, are the inversion/abduction artefacts the brief predicts (siblings through a shared record outcome or target). With the cone as KB they cannot occur, because every cone node is already in the closure. Deduction does not add chains but changes the winning value for roughly half the comparable pairs: `PLN.Query` takes the highest confidence and a derived implication can win.

### 7.6 Control sweep (shipped patient graph)

Smallest budgets with the closure complete (tasks = start fact): steps 6, task queue 3, belief queue 32. Settings 20/20/200, 100/10/100 and 40/20/200 are all complete (100/100). Handing over the whole graph (57 sentences) instead of the cone also works at this size.
Anytime behaviour on D+Q (states reached, CHD found): k=1 (2, no), 2 (3, no), 3 (5, no), 4 (6, yes), 5 (10, yes).

### 7.7 Scale and the best-first property (synthetic layered networks with diamonds; the real larger network does not exist yet)

| network | links | cone KB | legacy derivations from one source (mean) | fork, 40 steps | fork, 160 steps |
|---|---|---|---|---|---|
| 64 nodes | 168 | 175 | 566 | 58 of 66 states, 27k rule attempts, 89 ms | 66 of 66, 106k, 382 ms |
| 160 | 420 | 331 | 446 | 80 of 119, 52k, 167 ms | 110 of 119, 205k, 702 ms |
| 400 | 1400 | 1273 | 2239 | 130 of 339, 203k, 631 ms | 241 of 339, 809k, 2.6 s |
| 800 | 2800 | 1505 | 1969 | 134 of 463, 240k, 750 ms | 270 of 463, 957k, 3.1 s |

(timings are the pure-Python simulator; PeTTa is unknown.) The first 20 states the fork selects are exactly the 20 states with the highest best-path confidence on all four networks (recall 1.00), 1.00 up to about 40, falling to 0.74 to 0.97 at 160 steps because a state is re-selected once per trail that reaches it and the 40-task queue drops weak tasks.
Reading: the control gives an anytime, best-first answer under a bound where legacy enumerates every path. It does not index: each selection is paired with the whole belief queue, so cost is steps x |KB| x rules, and the KB has to fit the belief queue.

### 7.8 Can the Deriver itself fold the rows? (it cannot beyond about a dozen)

Rows as N implication sentences of one link with distinct ids, `Revision` only. Exact fold reproduced for N = 2, 3, 5, 8, 12 at 20/20/200 (and 100/10/100). N = 20: 18 of 20 rows merged at 20/20/200, 9 at 100/10/100, exact only at 3000 selections (3.9 s in the simulator). N = 40: 9 of 40 rows. N = 112 (Resveratrol's extension rows): 4 of 112. Hence the formula-level fold in section 3.

### 7.9 Optional: path revision (docs/deduction_layer.md 7.4), off by default

With an empty start stamp and `c0 = 1.0` the Deriver's own Revision rule merges independent paths: diamond A-B-D, A-C-D: legacy two derivations (0.56, 0.441) and (0.54, 0.378); the fork adds the merged belief `(0.55712, 0.62755)` with stamp (1 2 3 4) (native arithmetic: its single paths are (0.564, 0.49) and (0.548, 0.42)). Paths that share their first link (stamps overlap) are not merged. Not enabled: the shipped graph has no diamond, it would break "one derivation per connected pair" for graphs that do, and `c0 < 1` would make the start fact revise with itself.

---------------------------------------------------------------------------------------------------------------------------

## 8. Cost

* Shipped graph: 4336 rule attempts for all 30 sources in total (about 145 per source), 18 ms in the Python simulator; the cyclic probe from D+Q is 616 attempts, 11 selections.
* Revision: n-1 `Truth_Revision` calls per direction (169 for Resveratrol); negligible.
* Pre-flight: Tarjan over at most a few hundred Effect atoms, microseconds.
* Hooked text on hyperon: +28 % (patient sweep 6.91 s to 8.82 s) and +31 % (LinAge2 5.88 s to 7.73 s) over 900+ pair queries; hybrid only.
* Scale law: steps x |KB| x (rules + engraved rules); table in 7.7. `PLN.Query` re-runs the closure per call, so the seam runs it once per source node per request and reads all targets from it.
* **PeTTa timings are unknown**: process start (`consult(main.pl)`), library load, per-attempt cost, the speed of the tuple helpers. The spike must measure them (`compat/spike_runbook.md` G0 and step 6).

---------------------------------------------------------------------------------------------------------------------------

## 9. Provenance

* Of a cyclic result: the stamp lists the edge ids; ids map to the Effect atoms (and records) of the cone; the trail is the `Chain`. Of a revised link: `(RevisedFrom ..)` lists the papers and n; each item's `(s, c)` is recomputable from the raw rows. Second opinion: stamp plus selection order.
* Of the code: `lib_pln_fork.metta` = PeTTa `lib/lib_pln.metta` @ fea83178 (which is the fork's code, patham9/PLN @ 55f1751, ported by the fork's author; 63 of 81 forms identical, all truth formulas identical) + one patch (trace removal); both MIT notices kept; `Translator.metta` and `examples/PLN.py` are not vendored.
  Caveat from the compat study: `(import! &self (library lib_pln))` loads this bundled copy, not trueagi-io's.

---------------------------------------------------------------------------------------------------------------------------

## 10. Changes to consumers

| consumer | change in `legacy` | change in `hybrid` |
|---|---|---|
| all seven `infer` call sites, `explain`, `rank-*`, diagnose, counterfactual, risk, supplement, LinAge2 | none | none on every acyclic cone (bit-identical); a cyclic source returns derivations through `infer-cyclic` instead of hanging |
| `run_drugage_ranking` / `/drugage/rank` | none | `best_per_compound=False`, `revise=True`; `by_compound` takes the SUM of the compound's directional scores (net), not the best |
| `rank-interventions` (MeTTa sort) | none | lists a two-direction compound twice; use a seam-file `net-score-candidate` (sum of `rank-score` over the derivations, dominant direction's tv) in hybrid; the "linear and metta_sort agree" test needs a hybrid variant |
| `drugage_top` (whole-build rows) | none | none: it ranks rows by design (`significant_only`, species filters narrow the evidence); a compound-level revised top is an addition, not a change |
| `drugage_scoring.py` | none | `score_compound_revised` (the twin of the fold) plus a parity test against the fork formula |
| `confidence_threshold`, `parse_scored`, `_STV_RE` | none | unchanged shapes; with two directions the dominant one's confidence is used |
| validator / inventory | none | add `fork-second-opinion`; refuse caller-supplied `(CycleReach ..)` facts (harmless, but they reroute) |
| HTTP layer | none | `(CyclicCone A)` on hyperon maps to a failure status; `(fork-closure-status A)` truncated maps to `partial`; optional second-opinion endpoint |

New pins under hybrid, never edits of legacy pins: the 186 closure derivations equal legacy; feedback edge; the six-row compound; revised Resveratrol/Metformin/Rapamycin (paper-level); native second-opinion values at relative 1e-12.
Pins that change **only in hybrid with the ETL build present**: Rapamycin HTTP confidence 0.81 becomes 0.8801; Trimethadione 0.315 becomes Neg 0.315 plus Pos 0.126 (net score 0.2082 becomes 0.1756, rank 42 to 180). With the 8-row fixture (one row per compound, all pinned DrugAge tests) nothing changes: a one-row group is returned unchanged.

---------------------------------------------------------------------------------------------------------------------------

## 11. Does this answer Ben's request?

Ben's request, as LAUNCH.md records it: use the PLN version from patham9's GitHub on the PeTTa atomspace, with its control mechanism, on a larger network.

**What it does deliver.** (1) The fork's evidence rule (`Truth_Revision` with stamp-disjoint evidence) doing real work that legacy cannot: pooling 170 resveratrol rows into two truth values, with the independence granularity as an explicit, checkable choice. (2) The fork's control loop running where legacy has no answer: termination on feedback, a step and queue budget, best-first order. On the synthetic larger networks the first states it selects are the exact top by best-path confidence, which is a property legacy's path enumeration does not have. (3) A way to run the fork on the real network on demand, with its numbers, stamps and selection order next to legacy's.

**What it does not.** The fork is not the engine: on the shipped 30-node network every conclusion is produced by legacy, the control is never binding (11 selections, one derivation per pair), and the fork's own arithmetic would change the pinned Patient001 order. The one place it supplies numbers is the multi-row revision, and there it uses a single formula, not the inference loop. The Deriver cannot fold rows beyond about a dozen (7.8). On cycles the fork adds nothing a visited-path guard in legacy would not, except the budget. No larger real network exists yet to demonstrate the control on; the synthetic results show the property but not its value for the curators.

**Verdict:** mostly a revision step plus a safety net and a window. It lets the owner say truthfully "the fork runs on this network and agrees on the structure of every conclusion (100 of 100, 86 of 86 pairs, equal signs)", and show Ben where its control matters. It does not let him say "the app now reasons with patham9's PLN".
If the aim is the second statement, this design is the wrong one: the fork must become the engine, with new pins (the Patient001 order is the first casualty) and a decision about its 0.02 leak and the missing per-hop discount.

---------------------------------------------------------------------------------------------------------------------------

## 12. What was verified, and what was not

**Verified by running code** (all re-runnable, `compute_numbers.py`, `verify_*_worker.py`, `summarize_hooks.py`):
* Real hyperon 0.2.10, patient stack (18 files) and LinAge2 stack (17 files): with the hooked text and guard, `infer` and `explain` strings of all 100 and all 86 pairs are identical to the recorded closure; mark lookup absent. Original text on the same sweep: also identical (reference run).
* Real hyperon with the feedback edge and marks: marked sources return `(CyclicCone A)`, the 30 pairs of unmarked sources are identical to legacy, no hang; minimal 4-file stack toys (self loop, 2-cycle) return the marker, `infer X Y` on the side is unchanged.
* Real hyperon, DrugAge stack: six raw rows give six derivations equal to the Python lift (multiset identical); the revised atoms give two derivations equal to the Python numbers, plus `explain` and a two-entry `rank-interventions`.
* The fork port (`fork_sim.py`, 20 of 20 against upstream pins) reproduces the closure; the simulator is exercised under both `Unique` modes.
* Legacy arithmetic along fork trails equals hyperon's closure on all 186 derivations.

**Not verified:**
* Any PeTTa execution. The MeTTa seam is a sketch; whether `PLN.Derive` with five arguments, tuple helpers, `collapse`/`superpose` over runtime lists and the new symbols `Up`, `Down`, `End`, `Link`, `Trail` behave on PeTTa is unknown (a grep of the PeTTa source found no builtin with those names). The Python twin is the specification; the spike has to turn the sketch into code that matches it.
* PeTTa timings and the process-spawn cost of the revision child.
* The DrugAge numbers were computed from the repo's CSV with my re-implementation of the ETL mapping, not from the project's `build/drugage_etl.metta` (not generated here).
* Whether hyperon's panic threshold (150 rows) is irrelevant to the revised path: it should be (two atoms per compound), but the full hybrid stack was not run against 1000 compounds.

---------------------------------------------------------------------------------------------------------------------------

## 13. Risks

1. **Revision semantics.** Magnitude averaged as a probability; confidence tends to 1 with accumulating papers; Ethanol reaches rank 1 under row ids. Paper ids are the default for that reason, and it is still a calibration decision for the curators, not a solved problem.
2. **Net score.** Resveratrol and Metformin move from exactly 0.0 to about -0.04: a visible product change. The supplement tier veto and the pinned fixture tests are not affected, but any narrative that said "scored 0" is.
3. **Pins under hybrid with the build:** Rapamycin 0.81 becomes 0.8801, Trimethadione splits into two directions.
4. **Arithmetic mismatch.** The fork's native values are not legacy's; they change the Patient001 personalised order. Mixing them into `(signed ..)` would silently re-rank. Hence the separate type.
5. **Unrun PeTTa code.** Everything fork-side is specified by the Python twin and unexecuted on PeTTa; a mismatch there would show up as failed parity tests, not as wrong answers, provided the parity tests exist before the mode is switched on.
6. **Belief queue truncation** drops the least confident edges silently when `bq <= |KB|` (400-node cone at bq=200: 1 of 339 states). `bq` is derived from `|KB|`, and truncation is reported through `fork-closure-status`.
7. **Pre-flight blind spot.** The cycle marks come from reading Effect atoms as text; atoms created during evaluation are missed, in which case the hang remains and the deadline is the backstop. A variable source (`(infer &self $x D)`, supported by legacy, no caller uses it) is not supported in hybrid once any mark exists.
8. **The cycle part's value is the budget, not termination** (section 4). A reviewer may prefer a visited-path guard in legacy.
9. **Path revision** needs an empty start stamp and `c0 = 1.0`; with `c0 < 1` the start fact revises with itself. Off by default.
10. **Dispatcher cost** on hyperon (+28 to 31 % per sweep) in hybrid.

---------------------------------------------------------------------------------------------------------------------------

## 14. Effort (agent-phases, in the sense of LAUNCH.md)

| phase | content | agent-phases |
|---|---|---|
| A | vendor the lib with licences and provenance; PeTTa child harness for `fork-revise-items` | 1 |
| B | revision: `drugage_revision.py`, the twin and parity test, selector/ranking route with net score, tests, pins | 1.5 |
| C | hooks, guard, `graph_guard.py`, hybrid stack assembly, stub, hyperon tests (largely done in this study: the worker scripts are the tests) | 1 |
| D | the PeTTa seam: encode, closure, decode, second opinion, parity against `seam_ref.py` | 1.5 |
| E | optional endpoint, number table, docs | 1 |
| total | | about 6 |

A, B, C, E do not need the PeTTa runtime of LAUNCH.md phases 1-5 beyond the fork child; only D needs the whole stack on PeTTa. The revision part (A and B, 2.5) is the useful part and can ship alone.

---------------------------------------------------------------------------------------------------------------------------

## 15. Files (all under `.../forkdesign/design/design3/`)

| file | content |
|---|---|
| `design.md` | this document |
| `compute_numbers.py`, `numbers_log.txt`, `numbers_out.json` | the script behind every number (sections S1-S11) and its output |
| `seam_ref.py` | the executable spec of the seam (links, cone, guard, encode, closure, decode, revise, dispatcher) |
| `drugage_rows.py` | DrugAge rows and the legacy per-row lift (reads the repo's zip and constants) |
| `fork_sim.py`, `legacy_reference.py`, `common.py` | copies of the validated simulator and the legacy spec; loader |
| `hooks.py`, `pln_deduction_hooks.diff` | the exact hook text and its diff against the repo file |
| `verify_hooks_worker.py`, `verify_revision_worker.py`, `summarize_hooks.py`, `hooks_*.txt`, `revision_hyperon.txt`, `hooks_verification.json` | real-hyperon verification runs and their raw output |
| `pln_fork_seam.sketch.metta` | the PeTTa seam, unrun |

Reproduce: `python3 -B compute_numbers.py` (about 2 minutes); the hyperon workers with `/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -B verify_hooks_worker.py <patient|linage2|minimal> <orig|hybrid|cyclic>` and `verify_revision_worker.py`.
