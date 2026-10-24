# Refutation attempt, lens: semantic faithfulness

Reviewer notes, 2026-10-24. Work tree and upstream checkouts were only read. Nothing from `.petta/` was executed. My code is in
`forkdesign/refute_sem/` (standard library only, run with `python3 -I`): `mydrv.py` is a fresh port of the vendored loop
(`PeTTa/lib/lib_pln.metta:252-445`, restricted rules MP + NotElimination + Revision); it does not import `sim/fork_sim.py` or any designer file.
Data: `contract/links.json`, `contract/closure.json` (real hyperon output). Nothing ran on PeTTa; every statement about the real runtime is source-read.

## 1. What I could NOT refute (re-derived independently)

| claim | my result |
|---|---|
| D1 closure equals legacy | patient 100/100, LinAge2 86/86 pairs: sign, chain equal; max abs strength diff 1.67e-16, confidence 2.8e-17; 0 missing, 0 extra, 0 multi-route; max 12 selections. Holds at 20/20/200, 10/100/100, 256/256/4000, also with the fork's reversing `Unique` and at maxsteps 20 (`refute_sem/run_closure.py`) |
| last-digit differences | 34/100 strength strings and 25/100 confidence strings differ from closure.json (exactly as stated) |
| D+Q levels | raw q=p: 0.13769 / 0.36029, difference -0.222599; compensated -0.2499; native c 0.23343125; x0.9^3 = 0.17017138125 |
| Revision inert | removing Revision changes no belief set on 30 patient sources, with or without the feedback edge (`probes.py`). Also because links are never tasks, a link-link revision can never be attempted |
| feedback edge | 109 pairs, equal to a node-simple legacy reference at route level (0 mismatches), D+Q 12 selections, CS 10, max 16, all sources quiesce (`cyc.py`) |
| random graphs | 30 random graphs (4-5 nodes, 5-8 Effects + up to 2 records, self-loops, parallel edges, cycles): route-level equality with node-simple legacy, 0 mismatches (`rnd.py`). A second batch (seed 11, 45 graphs, 4-6 nodes, 6-9 Effects) also gave 0 mismatches (`rnd2.out`) |
| orderings | Patient001 legacy 0.2306/0.1647/0.0886/0.0768/0.0712; native c flips positions 4/5 (Spermidine 0.1041 over Berberine 0.0948). Patient002 order unchanged. diagnose c(CS) 0.6253 vs 0.6961 native, mass 0.4593 vs 0.5391. NMN/Fisetin 1.257 vs 1.071 (`personal.py`, `diag.py`) |
| four literal readings | 0.2226 (raw MP difference), 0.2686 (Up/Down start fact, hand-chained 0.7, 0.601, 0.42868, 0.2686344), 0.28519 (Deduction p=0.1) and 0.5112 (p=0.5) reproduced by hand |
| completeness test | tq < bq silently loses states (tq 3, bq 4096: flagged complete, 17 of 71 states; tq 8: 41 of 71); tq = bq flagged partial every time. The K design is right (`layered` run) |
| fork source claims | Rules.metta:76-77, Formulas.metta:47-48, Deriver.metta:19/27-31/37/45, Utils.metta:85-92, Config.metta:4-10, Translator.metta:35 all say what the notes say. Induction/Abduction call sites pass (A, target, shared) while Truth_Induction wants (A, shared, target): confirmed, full lane only |
| MixCompound and D3 six-row folds | 0.263050/0.949466 and 0.497903/0.911715 reproduced by hand |

Conclusion on the engine: the mapping, the two-run difference, the shim and the no-false-chain claim follow from the source. The objections below are about the layer around it.

## 2. Objections

### O1 (major). "Net of the two directions" is undefined, and the two candidate definitions have opposite signs on the designers' own examples

Decision section 3 says ranking "uses the net of the two" and Owner Q2 lists "Resveratrol and Metformin leave exactly 0 for about -0.04 net". Two incompatible nets exist in the notes:
* D3 (and the -0.04 figures): sum of signed directional scores, `s_neg*c_neg - s_pos*c_pos`.
* D1 (`mp_state.py:288-300`, design section 10): Truth_Revision of the two directional states, which is the evidence-weighted signed mean of strengths with weights `c/(1-c)` on native c.

Recomputed (`refute_sem` inline, c as post-hop in D3's table, native = c/0.9):
* D1 MixCompound (Neg 0.263050/0.949466 native, Pos 0.6/0.5): summed-score net = -0.0452 (shortens); pooled net = Neg 0.2194, c 0.8567, rank-score +0.1880 (extends). Legacy best row: +0.405.
* Resveratrol, D3 paper-level groups (Neg 0.2454/0.8740, Pos 0.3798/0.6688): summed -0.0395; pooled Neg 0.1959, c 0.876, rank-score +0.1716.
* Metformin, D3 groups (Neg 0.2420/0.8492, Pos 0.3514/0.6995): summed -0.0403; pooled Neg 0.139, rank-score about +0.12.
So the Owner is asked to decide on a table whose sign belongs to Design 3's aggregator while the engine chosen is Design 1's. Also pooling conflicting evidence raises confidence (0.8545 to 0.8567) and 50 extending plus 50 shortening rows give strength near 0 at confidence near 1.
Fix: define the net in the decision (one formula, one sign convention), recompute the Resveratrol/Metformin/Rapamycin rows with it, and make gate 6b print both.

### O2 (major). The 2% compensation cap breaks "strength is the plain product" on a shipped link, and the 6a gate cannot see it
`q = min(1.0, p + 0.02)` gives slope 0.98 for every p above 0.98. The generic stack (the one the API uses) contains `(Effect Lifespan Mortality Neg (stv 1.0 1.0))` (contract links.json, drugage_calibration.metta:226). My engine on the generic stack: 101 pairs, 100 equal to a node-simple legacy reference, 1 mismatch: Lifespan -> Mortality Neg 0.98 (legacy 1.0). nhanes_patient: 81 pairs, 0 mismatches. A chain through it loses 2 percent (Taurine curated chain 0.3548 to 0.347704; p=[1.0,1.0] gives 0.9604; p=0.99 gives 0.98). D1 design section 2 says "largest curated p is 0.95, so the cap never binds on shipped data", which is false for that stack, and section 10 then concedes it. Gate 6a (100 + 86 pairs) and contract MUST 9 ("same answers in all stacks") disagree. A caller's `extra_atoms` with p above 0.98 is shrunk silently.
Fix: at KB build time either assert `p <= 1 - e` and fail loudly, or multiply the adapter strength by `p/(q-e)` for every capped link (exact, slope is known); add the generic stack (101 pairs) to gate 6a.

### O3 (major). The paper-level evidence unit was never run through D1's adapter, and its direction handling is unspecified
Decision section 3: "group per compound and direction, fold per paper-level item ... inside a paper the existing representative policy". Real CSV (data/drugage/dataset.zip, 3,372 scorable rows): 252 (compound, paper) groups contain rows of both directions (220 compounds, e.g. ethanol PMID 13369, sodium borate 2325439). Two readings, both with a defect:
* representative per (compound, paper) across directions (the existing policy: evidence tier, significance, lower median): the opposite-direction rows of that paper are dropped, and in 43 of the 252 the chosen direction disagrees with the majority of that paper's rows (my emulation of the five documented rules; e.g. sodium borate 7 rows, 4 extending and 3 strongly shortening, represented by -21.2).
* one item per (paper, direction) (Design 3): the same paper id then sits in the stamp of both the Neg-group and the Pos-group link of one compound. D1's reference adapter maps `ids[i] = link` one link per id (`mp_state.py:72-92`) and `path_from_stamp` returns None when two links with the same `from` appear (`:131-150`): the Pos-group route is silently dropped (or mislabelled via `route_sign`, `:160-168`). The pooled net would also treat opposite results of one paper as independent evidence.
Fix: state the policy; give every (link group) a unique stamp id and keep paper identity in the side table, or assert ids are unique per link in the adapter.

### O4 (minor). "Same study restated by two layers carries the same id" collides with the stamp rule on serial chains
StampDisjoint refuses to combine two premises that share an id. A -> B (id 7), B -> C (id 7): my engine derives A, B only; (A, C) is lost although legacy answers it. No shipped atom has a study-level id today (ids are per atom), but the expansion pipeline emits `identifier PMID_n` on Effects (tests/test_ontology_expansion.py:505-508) and the decision's rule would invite exactly this. Restrict shared ids to parallel evidence for one link; assert id uniqueness along any derivable path.

### O5 (minor). Control demonstration reports the wrong recall metric
Priority is confidence (lib_pln.metta:402), so "top-k by best-path confidence" recall is 1.00 almost by construction. On three synthetic 6x24 layered DAGs (fan-out 3, p in [0.2, 0.9], c in [0.5, 0.9]) recall of the top-20 states by the application's rank-score (s*c) is 0.65 to 0.85 after 20 selections and 0.85 to 0.95 after 40 to 80, against 0.80 to 1.00 by confidence (`layered2.py`). 6c should print recall for rank-score too, or the demonstration says nothing about the app.

### O6 (minor). Fork source versus vendored file are conflated in the notes
Decision section 1 rows 10 and 20 describe `Unique` as prepending (Utils.metta:85-92). The vendored `lib_pln.metta` uses `list_to_set` (:77-78), `msort` for stamps (:50-52), `exclude-item` (:60-62, `metta.pl:204`, `exclude(==(A),...)`), `append` (:5-6). Order-keeping, not reversing. Rule and Truth forms are byte-identical (my form diff: 63 of 81 identical, the 18 fork-only forms are helpers, type declarations, config defaults, `Truth_Identity` (used only by the Translator route) and the traced Derive; no rule or Truth formula differs). The closure is unaffected (I ran both orders), but tie-breaking among equal-confidence tasks differs and G0 must run on the vendored file, not the fork src.

## 3. Verdict
The decision stands. No objection changes the choice of design 1 or the gate structure; O1 to O3 change what gates 6a and 6b must print and what the Owner is told. I found no case in the shipped stacks where design 1 returns a wrong sign, wrong direction or an outranking belief; the places where it could (O1, O2, O3) are the aggregation and id policy, not the reasoner.

## 4. Unknown
PeTTa wall time and behaviour of the real library (G0). Whether PeTTa's `case`/unmatched-call semantics equal the model (compat study says `case` with no branch fails). The real DrugAge table under any net (needs the ETL build and the species taxonomy). The full-rule lane counts (10, 64, 266, 6 pairs differ by encoding; I did not rebuild it).
