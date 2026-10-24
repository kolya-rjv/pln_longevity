# B2: the DrugAge paper-level evidence fold (gate 6b)

Work tree `/Users/kolya/PycharmProjects/pln_longevity_petta` (branch petta-migration), 2026-10-24. Files this task owns: `pln_chat/ontology/drugage_fold.py`,
`tests/test_drugage_fold.py`, `tests/fixtures/drugage_fold/`, this directory. Nothing else was edited, nothing committed.
Every number below was measured by `docs/petta_migration/fork_design/b2/measure_fold.py` (results in `results/measure.json`) or by `tests/test_drugage_fold.py`;
where a number comes from an earlier design note it says so.

## Corrections after the two reviews (fix phase, 2026-10-24): read these before sections 0, 3 and 5

The reviews of the pln build found three things in this note's subject. Each was reproduced, fixed and tested (`docs/petta_migration/fork_design/fix/`, `tests/test_drugage_fold.py`,
`tests/test_pln_fixes.py`). Where a section below contradicts this block, this block is right; the numbers in the sections are those of the first build, which filed a reported null on the
protective side only (`zero_rows="protective"`, kept as an option so they stay reproducible).

1. **The sign of Resveratrol and Metformin was the null convention, not the evidence.** A row that reports a change of exactly 0.0 (an ITP null: strength 0, confidence 0.9) went into the
   PROTECTIVE pool only, because the sign convention has no zero. Pooled by Revision a null drags the protective mean toward 0 and leaves the harmful mean alone, so every null pushed the NET toward
   "harmful". Resveratrol has 23 null rows of 170, Metformin 5 of 54. The NET under the three conventions (`measure_fold.py`, section `null_conventions` of `results/measure.json`):

   | compound | rows / papers / nulls | nulls on BOTH sides (default now) | nulls dropped | nulls on the protective side only (this note's section 3.1) |
   |---|---|---|---|---|
   | Resveratrol | 170 / 37 / 23 | **+0.1363** protective, rank 322 of 1,035 | +0.0321 protective, rank 650 | -0.0460 harmful, rank 878 |
   | Metformin | 54 / 16 / 5 | **+0.1182** protective, rank 406 | +0.0629 protective, rank 600 | -0.0570 harmful, rank 892 |
   | Rapamycin | 37 / 18 / 1 | **+0.3368** protective, rank 3 | +0.3368, rank 4 | +0.3368, rank 3 |
   | Ethanol | 52 / 1 / 0 | **+0.1144** protective, rank 426 | +0.1144, rank 427 | +0.1144, rank 420 |

   Over the whole build 70 of 1,035 compounds have a null row; counting the nulls on both sides instead of one flips the sign of exactly 2 (Metformin and Resveratrol, the two this note and the number
   table presented as "leave 0 for a small harmful net"), dropping them flips 8, and both-versus-dropped differ in 6. Both symmetric conventions agree that the two compounds are PROTECTIVE; only the
   one-sided one makes them harmful. **Default: both sides**, because the repo's own semantics say a well-run null is high-confidence evidence of NO effect (`SCORE_SEMANTICS.zero_score`, the ITP rule), which is
   evidence about both directions; dropping the nulls would discard exactly the evidence the semantics call high-confidence. The convention is `drugage_fold.ZERO_ROW_CONVENTIONS`, a parameter of every
   entry point and `ChainSettings.drugage_zero_rows`. It is a curation call the owner and Ben should confirm; the three columns above are the whole of the decision. With the default the whole-build
   ranking moves as before in aggregate (9 of the legacy top 20 stay, none keeps its place, Spearman 0.860 over 603 shared protective compounds, 81 move by more than 100 places).
   A direction whose pool would hold nothing but nulls pools to strength 0, adds nothing to the NET and is not emitted as an item (Rapamycin's one null dilutes its protective item and gives no harmful one; a consumer sees one
   derivation, as the legacy chainer gives, not a second empty one). A compound whose ONLY evidence is nulls keeps its protective item of strength 0 under every convention (score exactly 0.0, `no_effect`, the legacy confidence):
   it is never dropped from a ranking.

2. **Hooks 1 to 5 of section 5 are built, and "Rapamycin 0.8801 over HTTP" was false when written.** `run_drugage_ranking` kept the best derivation per compound and a single representative row, so it could
   neither see the harmful item nor the pooled confidence; Resveratrol came back +0.2145 (protective, 2nd of 4) against its defined NET. Now, under `PLN_CHAINER=pln`:

   | hook | status |
   |---|---|
   | 1 `change`, `significance`, `itp` on every item | done in `collect_links` (it was already there when the review ran; a parity test now holds `collect_links`' links equal to `fold_rows`) |
   | 2 list a compound once, by its NET | `core.pln_runner.net_per_compound`, used by `run_drugage_ranking` (also for `strategy="metta_sort"`, whose MeTTa insertion sort cannot combine two derivations); equal to `top_compounds` for all three conventions |
   | 3 no row cap | the slice holds every row of the requested compounds; `best_per_compound` and `MAX_ROWS` are not used (313 rows for the four compounds) |
   | 4 whole-build ranking | `core.drugage_router.drugage_top(fold=None)` follows the chainer; under `pln` GET /drugage/top ranks by NET with `papers_protective/papers_harmful/rows_protective/rows_harmful` per entry and a `fold` block; the default response is key-for-key the old one (a test pins both key sets) |
   | 5 semantics | `core.drugage_router.score_semantics()`: `SCORE_SEMANTICS` itself under the default chainer, plus a `fold` block under `pln`; the chat route's per-row provenance bullets become one pooled bullet per compound |

   Rapamycin over the engine path is now NET +0.3368 with confidence 0.8801 (18 papers pooled), as the Python twin said. HTTP pins that move under `pln` with the build present: Rapamycin 0.81 -> 0.8801.
3. **`fold_items` chose its within-paper pick for the whole request from one row.** One unrelated row with no significance atom moved Resveratrol from -0.0460 to -0.0395. The pick is now per (compound, target) group.

Also: the module docstring said "the fork's Truth_Revision (patham9/PLN, vendored in vendor/pln_fork)"; `revision` is a Python PORT of it (held equal to the library on PeTTa by `test_the_fold_equals_truth_revision_of_the_vendored_library_on_real_papers`; the
reviewers re-ran 382 random pairs, max difference 0.0). Revision pools WEIGHT, not agreement: Rapamycin's 18 papers give confidence 0.978 whatever their spread, and papers are treated as independent;
with confidences capped at 0.999999 the fold differs by 4.9e-11 over 200 shuffles (the canonical order removes it). Those are curation questions for the owner and Ben, not defects.

## 0. Result

| gate | measured | pass |
|---|---|---|
| fold equals a hand-typed `Truth_Revision` over the per-paper items, 1e-12 | MixCompound typed by hand (0.263050 / 0.949466 / 0.854519 as the design notes print); 300 random item sets bit-identical to `fork_sim.Truth_Revision`; the independent oracle (closed form, own grouping) agrees on all 1,035 scorable compounds | yes |
| fold equals `Truth_Revision` of the vendored library on the real PeTTa | 5 pooled items (Resveratrol Neg 35 papers and Pos 8, Metformin Neg 14 and Pos 6, Rapamycin Neg 18): max difference 0.0, bit-identical; `c = 1.0` gives no result, `0.999999` gives one | yes |
| the existing 8-row fixture unchanged under legacy | no existing file was edited; the fold of the 8-row fixture (8 compounds, 8 rows) equals `score_row` bit for bit and the legacy ranking; full suites in section 8 | yes |
| lift fidelity | the lift equals `drugage_scoring.score_row` bit for bit on all 3,423 rows (3,372 scorable) and on the 201-row sample; `fold=False` equals `drugage_top` bit for bit (1,035 entries, labels and counts) under 10 filter combinations | yes |
| Ethanol (52 rows, ONE paper) does not rank first, build absent or present | build present (3,423 rows): legacy rank 259 of 1,035, paper fold 420, every row its own unit 1. Build absent (201-row sample): legacy 7 of 41, paper fold 11, rows as units 1 | yes |
| a mixed-sign compound yields a Pos and a Neg item | MixCompound; 245 of 1,035 compounds of the build have both | yes |
| table Resveratrol / Metformin / Rapamycin before and after under the single NET | section 3.1 | yes |
| whole-build ranking summary | section 3.3: 9 of the top 20 stay in the top 20, none keeps its place | yes |
| the 150-row hyperon-era cap does not apply | Resveratrol, 170 rows, folds with all 170 (rows covered 112 + 58); 400 synthetic papers fold; `MAX_ROWS` is never read | yes |
| one formula for the net | section 2.4 | yes |

`tests/test_drugage_fold.py`: 64 tests (62 plain, 2 `-m petta`), 1.9 s. Mutation check of the tests: 15 mutants of the module, 14 killed, 1 equivalent
(`max(max(c, c1), c2)` against `c`: the library's own redundant clause).

## 1. What was built

| file | what |
|---|---|
| `pln_chat/ontology/drugage_fold.py` | the fold. Pure, deterministic, standard library plus `drugage_scoring` and `drugage_selector`; no runtime import. Entry points: `fold_compound`, `fold_rows`, `fold_lifted` (the core), `fold_items` (the chainer's dict interface), `compute_net` / `net_of_signed`, `top_compounds`, `lift_row`, `revision`, `fold_truth_values`, `assign_stamp_ids`, `semantics` |
| `tests/test_drugage_fold.py` | the gates above |
| `tests/fixtures/drugage_fold/` | `mix_compound.metta` and `paper_structure.metta` (hand-built), `three_compounds.metta` (verbatim ETL blocks of Ethanol, Resveratrol, Metformin, Rapamycin, 313 rows, 168 KB), `expected_three_compounds.json` and `fold_oracle.py` (the independent oracle, standard library only, written from the spec) |
| `docs/petta_migration/fork_design/b2/` | this file, `measure_fold.py`, `make_fixtures.py`, `results/measure.json` |

## 2. Definitions

### 2.1 Evidence unit and lift

The independent evidence unit is the PAPER, the PMID token the ETL writes (`PMID_13369`). A row with no `ReportedIn` atom is its own unit (`row:DrugAgeRow_17`); none of the 3,423
rows lacks one. `evidence_unit="row"` makes every row its own unit; it exists to show why the paper is the unit and is not reachable from a request.

A row is lifted as `drugage_scoring.score_row` does (strength `|pct| / (|pct| + 20)`; direction by the sign of the change, a 0.0 change is protective with strength 0).
The fold revises the NATIVE confidence `tier x significance gate`; `native x chain_discount (0.9)` is the legacy per-row confidence, bit for bit. Rows without a change percent
are unscorable and skipped, as in the legacy path.

### 2.2 Grouping, the pick inside a paper, the pool

1. Group the lifted rows of one compound by (direction, unit). Directions are on the mortality axis, as `RowScore.sign`: Neg protective (lifespan went up), Pos harmful.
2. One row per group, by the repo's own representative-row policy (`drugage_selector.select_rows(best_per_compound=True, limit=len(group))`: ITP first, then significance,
   then the lower median change). A paper that reports male and female results in one direction contributes ONE item to that direction.
   A paper that reports both directions contributes one item to each, and the two carry different stamps.
3. Fold the items of one direction with the fork's `Truth_Revision` (port; `w = c/(1-c)`, `f = sum(w f)/sum(w)`, `c = w/(w+1)`, left fold in a canonical order: numeric PMIDs by number, then
   the non-PMID units by name). Every confidence is capped at 0.999999 first, because the library has NO RESULT at `c = 1.0`. One item is returned unchanged, uncapped.
4. At most one Neg and one Pos item per compound.

The fold runs on `p` (the strength before the chainer's `q = 0.02 + 0.98 p` storage). Revision is a weighted mean, so folding the stored `q` equals storing the folded `p`
(tested to 1e-15); the fold needs no knowledge of the leak.

### 2.3 Stamps

A pooled item carries one token per unit, `"<compound>|<unit>|<direction>"`, unique to that item (reviewer point (e) and (f)): a paper that appears in both directions, or for two compounds,
gets different tokens, so a stamp id maps back to exactly one link and no shared id can sit on two serial links. The paper itself stays in `Item.units` and in the provenance rows
(`Item.provenance()`: token, kind, unit, PMID, representative row, tier, significance, change, species). `assign_stamp_ids(folds, first)` gives deterministic integers and refuses a token that is on two items.
The chainer of this tree assigns one id per pooled link itself; the tokens then travel in the link's `detail`.

### 2.4 THE NET (one formula)

    net = sum over the pooled items of  (+1 if the item is Neg, -1 if Pos) * strength * confidence        confidence includes the 0.9 chain discount

* **Why this one.** `strength * confidence` with a sign is what the ranking already uses for one row (`RowScore.score`); the net is that same quantity summed over the compound's
  (at most) two directional items. A compound whose evidence is one item scores exactly what the legacy ranking gives that item, bit for bit (the sign of a zero included, in the synthetic cases): tested on
  the 705 compounds of the build whose scorable rows sit in one (paper, direction) group (443 of them have a single row); score, strength, confidence, sign, label and lead row all equal the legacy result.
* **Where Revision runs, and where it does not.** `Truth_Revision` is applied only to quantities of ONE sign (magnitudes of effects in one direction, which stay in its [0, 1] domain).
  The two directions are combined by the ranking's own signed arithmetic. Contradicting studies therefore subtract; they do not raise a confidence.
* **Sign and companions.** `net.sign` is Neg when `net >= 0` (the repo's convention has no zero). `net.confidence` is the confidence of the item that carries the sign.
  `net.strength` is that item's own strength when the compound has one item, `|net| / confidence` when it has two, so that `score = +-strength * confidence` holds for every entry.
  `net.direction` is `no_effect` when the net is exactly 0 or when every row behind it is NotSignificant (the legacy label rule), else `protective` / `harmful` by the sign.
  `net.row` is the lead row: the best-evidence representative of the carrying item (so the code that reads `e.row.species`, `e.row.pmid`, `e.tier_category` on a `RowScore` reads a `NetScore`).
* **Properties tested.** Antisymmetric (flipping every direction negates the score); four extending against four shortening papers of equal strength score exactly 0.0 with unchanged confidences;
  the net of a one-item compound is the legacy score.
* **The alternative, rejected.** The earlier design (D1) offered Truth_Revision across the two directions on signed strengths (`+s` for Neg, `-s` for Pos, weights `c/(1-c)`, score `x * c * 0.9`).
  It applies the formula outside its [0, 1] domain and gives the OPPOSITE sign on the compounds that matter (column "pooled-revision" below): Resveratrol +0.171, Metformin +0.116 against -0.046 and
  -0.057, because the confidence mass of 35 weak extending papers outweighs 8 stronger shortening ones. It is not used. It is in `measure_fold.py` as `pooled_revision_net` for the comparison.

## 3. Measurements

### 3.1 Resveratrol, Metformin, Rapamycin (and Ethanol): before and after, one NET formula

"Legacy" = the representative row, as `drugage_top` ranks it. "Fold" = the paper-level fold. Items are (strength, confidence with the 0.9, papers pooled, rows covered).

| compound | rows / papers | legacy score (label) | fold Neg item | fold Pos item | fold NET (label) | rank of 1,035: legacy -> fold | NET if every row were a unit | pooled-revision (rejected) |
|---|---|---|---|---|---|---|---|---|
| Resveratrol | 170 / 37 | 0.0000 (no_effect; ITP null, c 0.81) | 0.2454, 0.8740, 35 papers, 112 rows | 0.3895, 0.6688, 8 papers, 58 rows | **-0.0460** (harmful) | 720 -> 878 | -0.0272 | +0.1709 |
| Metformin | 54 / 16 | 0.0000 (no_effect; ITP null, c 0.81) | 0.2420, 0.8492, 14 papers, 38 rows | 0.3752, 0.6995, 6 papers, 16 rows | **-0.0570** (harmful) | 713 -> 892 | -0.0518 | +0.1161 |
| Rapamycin | 37 / 18 | +0.3191 (protective; 0.3939, 0.81) | 0.3827, 0.8801, 18 papers, 37 rows | none | **+0.3368** (protective) | 2 -> 3 | +0.3373 | n/a |
| Ethanol | 52 / 1 | +0.1432 (protective; 0.7579, 0.189) | 0.7684, 0.1890, 1 paper, 50 rows | 0.1628, 0.1890, 1 paper, 2 rows | **+0.1144** (protective) | 259 -> 420 | +0.5679 (rank 1) | +0.0946 |

Reading: Resveratrol and Metformin leave exactly 0.0 for a small harmful net, as the owner question anticipated (about -0.04); Rapamycin stays protective and moves from rank 2 to 3
(the designers said 3 to 5). The figures of the earlier design (D3: -0.0395, -0.0403, +0.3167) are NOT the same as these, and the difference is fully explained: D3 chose the row of a paper by
"most confident, ties to the lower median strength"; the decision says "the existing representative policy", which also puts a Significant row above a NotSignificant one inside an ITP paper (the
astaxanthin rule). Feeding D3's pick through this module's fold and net gives -0.0395, -0.0403 and +0.3167 to four digits (`measure.json`, `d3_within_paper_pick_through_this_fold`), so the
aggregator is the same code and only the pick differs. The row-level figures of D3 (-0.0272, -0.0518) are reproduced exactly by `evidence_unit="row"`.

HTTP pins that move when the build is present and the fold is on (for the integrator): Rapamycin confidence 0.81 -> 0.8801 (18 papers pooled), Trimethadione (1 paper, 6 rows) Neg 0.661 / 0.315
plus Pos 0.2593 / 0.126, net 0.2082 -> 0.1756, strength 0.661 -> 0.5573 (confidence stays 0.315). `test_confidence_threshold_moves_a_compound_to_filtered_out_over_http` pins 0.81 and 0.315 and keeps them under the
legacy chainer; under `pln` its Rapamycin value is 0.8801.

### 3.2 Ethanol, the gate

| source | rows / compounds | legacy | paper fold | every row a unit |
|---|---|---|---|---|
| `build/drugage_etl.metta` present (regenerated from `data/drugage/dataset.zip`, byte-for-byte what the ETL writes) | 3,423 / 1,043 (1,035 scorable) | rank 259 of 1,035 (258 of 617 in the default protective list) | **420** of 1,035 (417 of 641) | **1** |
| build absent: the committed 201-row sample, which `drugage_top` falls back to | 201 / 41 | 7 of 41 | **11** of 41 | **1** |

The designers' "rank 372 versus 1" used another aggregator; the re-measured value under this net is 420. Ethanol's 52 rows are 50 extending and 2 shortening arms of one paper: the fold has a Neg item
(native confidence 0.21, the tier x gate of one unreported invertebrate row) and a Pos item of the same paper, not a confidence of 0.93 (50 arms as 50 experiments give 0.9300; all 52 give 0.9325).

### 3.3 The whole-build ranking, legacy against fold (default `direction=protective`, as GET /drugage/top)

* Universe: 617 protective compounds under legacy, 641 under the fold (the fold relabels: protective->harmful 14, harmful->protective 27, no_effect->harmful 9, no_effect->protective 11, unchanged 603 + 152 + 219).
  Score unchanged for 715 of 1,035 compounds (those whose evidence is one item).
* Top 20: **9 of the legacy top 20 stay in the top 20; no compound keeps its position.** Top 10: 4 stay. Top 50: 19 stay. Spearman rank correlation 0.854 over the 603 compounds in both lists (0.924 over all 1,035); 83 (160) compounds move by more than 100 places.
* Legacy top 5: C60_fullerene 0.368, Rapamycin 0.319, Astaxanthin 0.304, N_17alphaestradiol 0.304, Indolepropionamide 0.290.
  Fold top 5: Spermidine 0.360 (4 papers, 8 rows, c 0.679), N_17alphaestradiol 0.352 (3 papers), Rapamycin 0.337 (18 papers), Reserpine 0.328 (3 papers), C60_fullerene 0.327 (2 papers).
* Leaving the top 20: Thymus_extract (to 65), N_2_mercaptoethylamine_hydrochloride (368), Canagliflozin (395; a Significant female ITP row shortens lifespan in the same paper), Nordihydroguaiaretic_acid (164),
  Aspirin (451), Hydrogen sulphide (33) and others; entering: Reserpine, Lithocholic_acid, Thioflavin_T, Mianserin, Psora_4, Tanespimycin, Baicalein and others, mostly compounds with several concordant papers; Spermidine moves from 15 to 1.
  The full lists are in `measure.json` (`ranking.protective`).

### 3.4 Size of the problem

3,423 rows, 664 papers, 1,043 compounds (1,035 scorable, 3,372 scorable rows). 172 compounds have two or more papers, 443 have one row, 158 have a pooled item, 245 have both directions, 1,705 stamp tokens in all.
The largest compound has 170 rows; exactly one compound exceeds the 150-row cap. A whole-build fold takes 16 ms.

## 4. The interface for the chainer: `fold_items`

`pln_chat/core/pln_chainer.py` (another task's file, read only here) resolves `ontology.drugage_fold.fold_items` and hands it one dict per lifted row:
`compound, target, direction ('Pos'|'Neg' = the LIFESPAN direction), strength, confidence (tier x gate, no chain discount), rows (1-tuple), papers (1-tuple, '' for none)`. `fold_items` returns pooled items in
the same shape (one per compound, target and direction), `confidence` being the pooled NATIVE confidence, plus extra keys the chainer carries into `detail`: `units, stamp, rows_covered, covered_rows, pooled,
native_confidence, chain_confidence (= native x 0.9), mortality_sign, net, net_sign, net_direction, within_paper_pick`. `fold=False` returns the items unchanged (the legacy per-row results).

I ran the chainer's `collect_links` on the real PeTTa with the DrugAge stack and `three_compounds.metta`: it picks the function up and produces 7 pooled links (Ethanol 2, Metformin 2, Rapamycin 1, Resveratrol 2) in 0.02 s.
**With the 7 keys it sends today, the pick inside a paper is the fallback (most confident row), so the integrated lane gives D3's figures (-0.0395, -0.0403, +0.3167), not the table of 3.1.**
The chainer task's own DrugAge tests (`PLN_RUNTIME=petta pytest tests/test_pln_chainer.py -k "drugage or fold"`, 5 tests, written against the `fold_items` contract) pass against this module.
Every output item says `within_paper_pick: most_confident_row` in that case; with the three optional keys below it says `representative_row_policy` and the figures are those of 3.1 and of the Python twin.

## 5. Hooks for the integrator (I did not edit these files)

1. **`core/pln_chainer.py`, the DrugAge block of `collect_links`: add `change`, `significance`, `itp` to each item.** `_DRUGAGE_Q` already binds `$p` (the AvgLifespanChangePercent) and does not return it.
   One more match gives the other two: `(AvgLifespanSignificance $r $g)` and the presence of `(IsITPStudy $r)`. With all three on every item the fold uses the repo's representative-row policy.
   Parity test to add on that side: for a set of rows, `collect_links`' DrugAge links equal `fold_rows` (strength, native confidence, rows, papers) to 1e-12, and `explain` / `infer` for a compound
   equal `(item.strength, item.confidence)` of the item (the chained answer is `strength` and `native x 0.9`, sign flipped from the lifespan direction to the mortality axis).
2. **The ranking must list a compound once, by its NET.** `run_drugage_ranking` (linear strategy) keeps, per compound, the derivation with the best `score` (`by_compound[sc.compound]`), which would
   show only the better direction. Under `pln` combine a compound's derivations to Mortality with `ontology.drugage_fold.net_of_signed([(sign, strength, confidence), ...])`, which returns
   `(score, sign, strength, confidence)` by the formula of 2.4 (at most one part per sign, as the fold guarantees). The MeTTa-side `rank-interventions` / `score-candidate` that lists one derivation per
   row must do the same or be bypassed.
3. **No row cap on the pln path.** `run_drugage_ranking` calls `build_drugage_slice(compounds, best_per_compound=True, limit=MAX_ROWS)`, which keeps one row per compound and truncates at 150 rows. The
   fold needs every row of the requested compounds: `select_rows(load_rows(src), compounds=canonical, limit=<number of rows>)`, `best_per_compound=False`. The cap stays for hyperon. On PeTTa a space with four compounds' rows (313 rows)
   loaded and was read by `collect_links` in the smoke test of section 4; the whole-build ranking does not go through the engine.
4. **Whole-build ranking (`core.drugage_router.drugage_top`, GET /drugage/top).** Add a `fold` switch (default off = today). With it on, call
   `top_compounds(load_rows(src), n=..., species=..., clade=..., min_confidence=..., itp_only=..., significant_only=..., direction=..., fold=True)` (it applies the filters itself, on rows, before pooling) and build `DrugAgeTop`
   from the returned `FoldTop` (same counters). `NetScore` has `row, score, sign, strength, confidence, direction, tier_category`, so `DrugAgeTopEntry` is built unchanged; add `papers_neg, papers_pos, rows_neg, rows_pos`
   to the entry, because `row` is the lead row of a pooled score and showing its species or percentage next to the pooled number would mislead.
5. **Semantics block.** `SCORE_SEMANTICS` and the representative-row policy text describe the legacy ranking. Under the fold add `drugage_fold.semantics()` (the six-point `FOLD_POLICY`, the net formula, the cap).
6. **Optional clean-up in `drugage_scoring.py`.** The fold repeats the tier / gate arithmetic of `score_row` (it must revise the native confidence, which `RowScore` does not expose). It is guarded by a bit-for-bit
   test over all 3,423 rows. Exposing `tier * gate` on `RowScore` would remove the copy.
7. **Stamp ids.** If the integrator wants the fork's stamps to name papers rather than one id per pooled link, `assign_stamp_ids(folds, first=n+1)` returns the integers; the first id must lie above the curated ids and 0 is the intervention premise.

## 6. Choices against the decision text and the reviewers' list

| point | what was done |
|---|---|
| (a) `q = 0.02 + 0.98 p` | not the fold's business; pooled strengths are in `p`. The commutation of Revision with the affine map is tested (1e-15) |
| (c) probes in Python, never MeTTa `test`/`assert` | the PeTTa tests compare in Python; `c = 1.0` gives no result on the real library, asserted |
| (e) ids shared only for parallel evidence, unique along paths | tokens are unique per link (compound, unit, direction); `assign_stamp_ids` refuses a token on two items |
| (f) one defined net, recomputed table, distinct stamps per direction | 2.4, 3.1, 2.3 |
| (g) generic stack | not this task |
| (h) 150-row cap | not read by the fold; Resveratrol 170 rows and 400 synthetic papers fold |
| (i) recall by rank-score | not this task (6c) |
| decision: "emit the pair onto Mortality with one virtual hop" | superseded by correction (a): with the exact storage the materialised `Lifespan -> Mortality (1.0, 1.0)` link is exact, so the fold emits the lifespan-direction item and the chain adds the flip and the 0.9. `fold_items` returns the lifespan direction accordingly; `fold_rows` / `NetScore` use the mortality axis, as `RowScore` does |
| decision: "inside a paper the existing representative policy" | implemented; the fallback exists only because the chainer's item does not yet carry the three keys it needs (section 4) |

## 7. Open points

* The two within-paper picks give different numbers for the same data (3.1). Until hook 1 is applied the chainer lane and the Python twin disagree on every compound that has a paper with several rows in one
  direction (`test_the_two_picks_agree_wherever_a_paper_has_one_row_per_direction`: they agree on the 400+ compounds where no paper does).
* Resveratrol and Metformin are NET harmful by 0.046 and 0.057. That is what 8 and 6 shortening papers of strength 0.39 and 0.38 do against 35 and 14 extending papers of strength 0.25 and 0.24 when each direction is
  pooled alone. It depends on the net formula (the rejected pooled-revision net says +0.17 and +0.12) and on the within-paper pick (-0.0395 and -0.0403 with D3's); the owner and Ben should look at this
  before any user-facing text says "net".
* Papers are treated as independent. A lab that publishes five papers on one cohort counts five times; the paper is a curation call, not a theorem (decision section 10, risk 4).
* Species are not separated: a paper that tests a compound in two species contributes one item per direction.
* `top_compounds` and `fold_items` assume `drugage_scoring.load_knobs()` constants; `test_the_oracle_constants_are_the_calibration_the_module_reads` and the whole-build pin
  (`test_the_whole_build_ranking_moves_by_the_measured_amount`: 9 of 20, Spermidine first) fail loudly if a constant is tuned.
* Not run: the hyperon runtime does not load this module (it is only used by the whole-build twin and by the `pln` chainer, both PeTTa-side or pure Python).

## 8. Full suite (hard rule 1)

Run last, after every file of this task existed, with `PLN_CHAINER=legacy` (the default; nothing in the legacy lane imports the fold):

| runtime | result | failure set |
|---|---|---|
| `PLN_RUNTIME=hyperon` (default), one test deselected as in the baseline (`test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move`, the known trie.rs:179 abort) | 32 failed, 2312 passed, 22 skipped, 1 deselected in 442 s | the 32 ids are IDENTICAL to `docs/petta_migration/baseline/hyperon_failures_47130aa.txt` (`diff` of the sorted lists is empty) |
| `PLN_RUNTIME=petta` | 31 failed, 2430 passed, 16 skipped in 333 s | the baseline minus `test_human_evidence.py::test_the_runtime_kb_still_answers_the_query_that_aborted_the_interpreter` (the hyperon-only abort probe); nothing added |

No test of `tests/test_drugage_fold.py` fails in either run. The pass counts include tests other agents added in parallel, so only the failure sets are comparable with the baseline.
No existing assertion was edited: the work tree after the runs holds only new paths for this task (`pln_chat/ontology/drugage_fold.py`, `tests/test_drugage_fold.py`, `tests/fixtures/drugage_fold/`,
`docs/petta_migration/fork_design/b2/`).

## Reproduce

```
.venv/bin/python -m pytest tests/test_drugage_fold.py -q                       # 64 tests, 2 s; add -m "not petta" to skip the real library
.venv/bin/python docs/petta_migration/fork_design/b2/measure_fold.py          # every number of section 3 (regenerates the ETL in a temp dir)
.venv/bin/python docs/petta_migration/fork_design/b2/make_fixtures.py          # regenerates three_compounds.metta and expected_three_compounds.json
```
