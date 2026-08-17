# Intervention Ranking — Demo 2 (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`pln_intervention_ranking.metta`; candidate-pool bridges added to
`mechanistic_bridges.metta`; regression tests in
`tests/test_pln_deduction.py`.
**Scope:** the first *complete* demonstration scenario from `pipeline.md` (§4.2,
Demo 2) — rank a pool of candidate interventions by their expected, uncertainty-
propagated, **signed** effect on a target outcome.

> **Dependency / base branch.** Sits directly on the deduction layer
> (`pln_deduction.metta` → `infer`) and everything it needs. Load last (see the
> file header for the exact order). It is a pure consumer of `infer`: it adds no
> new inference power, only the application that turns a per-pair estimate into a
> ranked decision.

---

## 1. Why this is the demo to build first

`infer` already produces, for *one* intervention → *one* outcome, the thing an
LLM cannot: a `(signed sign (stv s c))` with a full audit trail. Demo 2 is the
smallest step that turns that primitive into a **decision** — and it is the
headline scenario of the pipeline (`pipeline.md` §4.2). It also exercises the
entire stack end-to-end (calibration → curated bridges → deduction → ranking)
against real records, which is how the architecture gets validated.

## 2. What it computes

```
rank-interventions : space × (candidate …) × outcome
   ⊢  ( (scored <best>  <score> (signed Neg (stv s c)))
        (scored <next>  …) … )          most-protective first
```

For each candidate it runs `infer`, folds the result to one orderable scalar,
and insertion-sorts descending. The scalar (`rank-score`) makes "better = more
protective" explicit and keeps **sign first-class**:

| net effect | score | meaning |
|---|---|---|
| `Neg` (protective) | `+ (s · c)` | reduces the outcome — ranks high |
| `Pos` (harmful) | `− (s · c)` | raises the outcome — ranks below zero |

Both magnitude **and** confidence count, so a big effect known with near-zero
confidence does not outrank a solid moderate one.

## 3. The candidate pool (grounded, not invented)

To make a *ranking* meaningful the pool must reach the outcome through chains
that genuinely differ. Three senescence/autophagy interventions now reach
`CoronaryHeartDisease` through the same calibrated Lu 2019 axis
(`… → DNAmPAI1 → CHD`), each grounded in a López-Otín 2023 record:

| intervention | route to senescence axis | hops | evidence tier |
|---|---|---|---|
| **DasatinibPlusQuercetin** | senolytic → `CellularSenescence` (Neg) | 4 | mouse + human → `AnimalStudies_Replicated` |
| **Fisetin** | senolytic → `CellularSenescence` (Neg) | 4 | single mouse → `AnimalStudies_Single` |
| **Spermidine** | induces `Autophagy` (Pos) → reduces senescence (Neg) | 5 | single mouse → `AnimalStudies_Single` |

Result (`infer` → `rank-interventions … CoronaryHeartDisease`):

```
(scored DasatinibPlusQuercetin 0.04253 (signed Neg (stv 0.2499  0.17017)))
(scored Fisetin               0.03038 (signed Neg (stv 0.23205 0.13090)))
(scored Spermidine            0.01312 (signed Neg (stv 0.17136 0.07658)))
```

Three reads that an LLM cannot reproduce from first principles:

1. **D+Q ≻ Fisetin** on *identical mechanism* — the only difference is evidence
   tier (replicated vs single study), which calibration turns into confidence.
2. **Spermidine ranks last** because representing its mechanism honestly costs an
   extra hop, and confidence decays per hop (`0.170 → 0.131 → 0.077`) — the
   "uncertainty increases with chain length" property made visible.
3. **Elamipretide is omitted, not mis-ranked** — it acts on
   `MitochondrialDysfunction`, which has no chain to CHD in the current KB, so
   PLN simply declines to place it. No false chain.

## 4. Implementation notes (hyperon 0.2.10)

- **`superpose` + `collapse`** fan the candidate tuple into the non-deterministic
  `infer` stream and gather every result back into one tuple to sort. An
  unconnected candidate contributes nothing (it drops out); a candidate with
  several derivations contributes one entry each (v1; PLN revision over multiple
  paths is the documented follow-up, `deduction_layer.md` §7.4).
- **Grounded ops do not evaluate their arguments.** `car-atom`, `cdr-atom`,
  `cons-atom`, and `==` operate on already-reduced atoms, so every intermediate
  in the sort is forced through `let`/`let*`. Skipping this leaves the recursion
  symbolic — e.g. `(== (cdr-atom …) ())` is always False because the `cdr-atom`
  call is never reduced. (Same discipline as the cross-module `&self` threading
  the calibration/deduction layers document.)

## 5. Open questions / next increments

1. **Multiple derivations per candidate** — when `infer` yields several paths to
   the outcome, rank currently lists each; combine them with a PLN **revision**
   rule first (shared with `deduction_layer.md` §7.4).
2. ~~**Patient grounding**~~ ✅ **Done (v1)** — `patient_profile.metta` adds
   `rank-interventions-for-patient`, folding a patient-relevance factor
   (`f(patient factors)`) onto the population score from this layer (see
   `docs/patient_grounding.md`). v1 combines additively; the multiplicative form
   of `pipeline.md` §4.2 and reordering across differently-presenting patients
   are the documented follow-ups.
3. **Multi-outcome ranking** — rank against a basket of outcomes (mortality, CHF,
   CHD) and aggregate, rather than a single target.
4. **Surface in the chat app** — add a few-shot so the LLM translator emits a
   ranking form. ✅ **Done for the DrugAge lifespan/mortality axis (v1):** the
   translator emits a dedicated `(rank-drugage-lifespan (C1 C2 …))` and
   `pln_chat/app.py` routes it to the **scoped** `run_drugage_ranking` — which
   sidesteps the §6 full-KB panic by never touching `_ALL_KB_PATHS` (see
   `docs/etl_inference_wiring.md` §8.5 and `docs/drugage_chat_test_queries.md`).
   The **generic** `(rank-interventions … CoronaryHeartDisease)` over the whole KB
   is still **blocked** by the same pre-existing full-KB runtime issue (§6): the
   scoping trick that unblocks DrugAge is per-vertical, not a general fix.
5. **Reused by the supplement recommender (Demo 6).** ✅ The same
   `superpose/collapse/sort` + `rank-score` machinery, and the patient-relevance
   score it feeds, are consumed wholesale by `pln_supplement_recommendation.metta`
   to rank supplements within evidence tiers — a personalized, tiered recommendation
   with a negative-evidence veto and interaction flags. Because that file is small
   and on the focused stack, it rides the ordinary chat `run_query` path (no scoped
   space). See `docs/supplement_recommendations.md`.

## 6. Known issue: full-KB runtime panic (pre-existing, not introduced here)

The focused KB used by the tests (the calibration→deduction→ranking chain, 11
files) runs cleanly. But loading **every** `.metta` file in the repo — what the
chat app's runtime mode does via an alphabetical glob — aborts `hyperon 0.2.10`
with a Rust panic in its type-index trie on *any* query (even `!(car-atom (a b
c))`). The trigger is the large DrugAge ETL files
(`drugage_entries.metta`, `drugage_etl_short.metta`); removing them makes the
full load query cleanly. This predates the ranking layer and blocks the chat
app's runtime mode generally — worth its own fix (scoped per-query spaces, or
splitting/repairing the DrugAge atoms).
