# PLN Deduction Layer — Design (v0)

**Status:** v0, verified against `hyperon 0.2.10`. Implementation:
`pln_deduction.metta`; curated bridge priors: `mechanistic_bridges.metta`;
regression tests: `tests/test_pln_deduction.py`.
**Scope:** the first inference rule — transitive deduction that chains
truth-valued links and propagates `(stv s c)` so multi-step conclusions carry
appropriately reduced confidence.

> **Dependency / base branch.** Reads the calibration layer
> (`evidence_calibration.metta` → `calibrate-tv`) and the GrimAge evidence /
> ontology it sits on. Must be loaded after them (see file header for the exact
> order). It is the downstream consumer the calibration layer was built for.

---

## 1. Why we need it

Until now the knowledge base was **inert**. The calibration layer turns raw
evidence into `(stv s c)`, but nothing *chained* those truth values. Yet the
entire thesis of the project — the capabilities that exceed an LLM
(`pipeline.md` §1.2, Table 2) — lives in the chaining:

- **uncertainty propagation** through multi-step causal paths;
- **novel cross-source inference** (a conclusion no single source states);
- **complete provenance** (every conclusion traces to its steps);
- confidence that **decreases with chain length**.

This layer is the smallest rule that delivers all four.

## 2. Where it sits in the pipeline

```
raw records ─calibrate─▶ (stv s c) ─┐
                                     ├─lift─▶ (Implication A B (stv s c)) ─infer/explain─▶ (stv s c) + path
curated mechanistic bridges ────────┘
```

## 3. The canonical link (resolves calibration open-Q#1)

The inference layer chains over one dedicated, TV-bearing directed link:

```metta
(: Implication (-> Atom Atom Atom Atom))
;; (Implication <from> <to> (stv s c))
```

Why a dedicated link rather than annotating the raw predicates:

- raw evidence/ETL atoms stay **untouched and TV-free** (the calibration
  layer's immutability invariant);
- deduction stays **generic** — it matches one link shape no matter whether a
  fact came from GrimAge, hallmarks, DrugAge, or a future LLM extraction;
- endpoints are plain `Atom`s, so anything can be chained.

Domain predicates are **lifted** into this form; raw atoms are never rewritten.
For calibrated empirical associations the lift is *on the fly* — `link-tv`
reads `HasPredictor`/`HasOutcome` and calls `calibrate-tv`, so no TV is
materialised and `calibrate-tv` stays the pure function it already was.

## 4. The truth-value combination (v0 — deliberately conservative)

`combine-tv` implements one deduction hop A→B, B→C ⊢ A→C:

| Output | v0 rule | Rationale / v1 |
|---|---|---|
| strength `s` | `sAB · sBC` (independence product) | Full PLN deduction also folds in node base rates `P(B), P(C)`: `sAC = sAB·sBC + (1−sAB)·(sC − sB·sBC)/(1−sB)`. We don't store node strengths yet → v1. |
| confidence `c` | `cAB · cBC · chain-discount` | Guarantees confidence **decays per hop**. `chain-discount` (0.9) is factored out so the §8 tuning loop is a one-line change. |

`s` and `c` stay independent, exactly as in calibration.

## 5. Inference & provenance

`infer` is two equations — a **base case** (any direct link, curated or
calibrated-empirical) and a **transitive case** (step through one curated
bridge, recurse). It is non-deterministic: one result per distinct path, and
**none** when the endpoints are unconnected (no false chains). `explain` runs
the same recursion but also returns the node path:
`(Chain (n0 … nk) (stv s c))` — the auditability LLMs cannot offer.

## 6. The headline demo: a novel cross-source inference

```metta
!(explain &self CellularSenescence CoronaryHeartDisease)
;; (Chain (CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease)
;;        (stv 0.357 0.29089125))
```

This conclusion — *cellular senescence raises coronary-heart-disease risk via
the SASP → PAI-1 axis* — is stated by **no single source**. It is composed from
three:

1. `CellularSenescence` — a hallmark concept (López-Otín 2023, `hallmarks_core`);
2. `CellularSenescence → SASP → DNAmPAI1` — curated mechanistic bridges
   (`mechanistic_bridges.metta`), confidence derived from the evidence tier;
3. `DNAmPAI1 → CoronaryHeartDisease` — Lu 2019 GrimAge evidence, calibrated
   (`HR 1.31 → (stv 0.6 0.85)`).

Strength **and** confidence are attenuated across the 3 hops (confidence 0.85 at
the empirical leaf → ~0.29 at the root), demonstrating the core PLN property.

## 7. Open questions / next increments

1. **Signed / protective effects.** v0 chains monotone "elevation" links. An
   *intervention* that *reduces* an outcome (e.g. D+Q clears senescence →
   *lower* CHD risk) needs sign/polarity handling before intervention-ranking
   chains are correct. (Ties to calibration open-Q#3 and #5.)
2. **Base rates** — add node strengths `P(B), P(C)` to upgrade `combine-tv`
   from the conservative product to the full PLN deduction formula.
3. **Stepping through empirical links mid-chain** — v0 treats empirical
   associations as terminal; generalise the transitive case to step through any
   lifted link, not only curated bridges.
4. **Revision for multiple paths** — when several derivations reach the same
   conclusion, combine them with a PLN revision rule instead of returning each.
5. **Cycle guarding** — the recursion assumes a DAG; add visited-set guarding
   before the bridge graph can contain cycles.
6. **Mechanistic-consensus evidence tier** — curated bridges currently borrow
   `AnimalStudies_Replicated`; a dedicated tier would represent review-level
   mechanistic consensus better (calibration open-Q#4).

## 8. Tuning loop ("poke the pipeline")

The loop the calibration doc anticipated is now runnable: pick a chain, read the
collapsed `(s, c)` from `infer`, and watch for (a) confidence vanishing too fast
with length, (b) implausible strengths. Then adjust `chain-discount` /
`combine-tv` (one file) or a bridge prior and re-run the tests.

## 9. Non-goals

- Not numerically-tuned PLN (v0 locks the architecture, not the constants).
- Not a rewrite of any raw evidence atom.
- Not yet abduction, counterfactuals, or revision (later inference patterns).
