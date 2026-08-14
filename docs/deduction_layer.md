# PLN Deduction Layer — Design (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`pln_deduction.metta`; curated bridge priors: `mechanistic_bridges.metta`;
regression tests: `tests/test_pln_deduction.py` (13 tests).
**Scope:** the inference rule — transitive deduction that chains **signed**
truth-valued links and propagates `(stv s c)` + direction so multi-step
conclusions carry appropriately reduced confidence and the correct sign.

> **Dependency / base branch.** Reads the calibration layer
> (`evidence_calibration.metta` → `calibrate-tv`) and the GrimAge / hallmarks
> evidence and ontology it sits on. Must be loaded after them (see file header
> for the exact order). It is the downstream consumer the calibration layer was
> built for.

---

## 1. Why we need it

Until this layer the knowledge base was **inert**. The calibration layer turns
raw evidence into `(stv s c)`, but nothing *chained* those truth values. The
entire thesis of the project — the capabilities that exceed an LLM
(`pipeline.md` §1.2, Table 2) — lives in the chaining:

- **uncertainty propagation** through multi-step causal paths;
- **directionality** — a protective intervention vs. a harmful exposure;
- **novel cross-source inference** (a conclusion no single source states);
- **complete provenance** (every conclusion traces to its steps);
- confidence that **decreases with chain length**.

## 2. Where it sits in the pipeline

```
raw records ─calibrate─▶ (stv s c) ─┐
                                     ├─lift─▶ (Effect A B sign (stv s c)) ─infer/explain─▶ (signed sign (stv s c)) + path
curated mechanistic bridges ────────┘
```

## 3. The canonical SIGNED link (resolves calibration open-Q#1)

The inference layer chains over one dedicated, TV-bearing directed link:

```metta
(: Effect (-> Atom Atom Atom Atom Atom))
;; (Effect <from> <to> <sign> (stv s c))   sign ∈ { Pos, Neg }
```

`Pos` = *from* raises *to*; `Neg` = *from* reduces *to*. Strength `s` is the
effect **magnitude** (independent of sign); confidence `c` is evidence weight.

Why a dedicated link rather than annotating the raw predicates:

- raw evidence/ETL atoms stay **untouched and TV-free** (the calibration
  layer's immutability invariant);
- deduction stays **generic** — it matches one link shape no matter whether a
  fact came from GrimAge, hallmarks, DrugAge, or a future LLM extraction;
- endpoints are plain `Atom`s, so anything can be chained.

Domain predicates are **lifted** into this form; raw atoms are never rewritten.
For calibrated empirical associations the lift is *on the fly* — `link-effect`
reads `HasPredictor`/`HasOutcome`/`EffectValue`, calls `calibrate-tv` for the
magnitude, and reads the **sign off the hazard-ratio direction** (HR>1 harmful →
`Pos`, HR<1 protective → `Neg`). Nothing is materialised; `calibrate-tv` stays
the pure function it already was.

## 4. The combination rule (three independent axes)

`combine-hop` / `combine-tv` implement one deduction hop A→B, B→C ⊢ A→C:

| Axis | v1 rule | Rationale / next |
|---|---|---|
| **sign** | `sign-product` — product of hop signs (`Neg·Neg = Pos`) | A protective step (Neg) into a harmful axis yields a net protective (Neg) conclusion. This is what intervention ranking needs. |
| strength `s` | `sAB · sBC` (independence product) | Full PLN deduction also folds in node base rates `P(B), P(C)`. We don't store node strengths yet → v1+. |
| confidence `c` | `cAB · cBC · chain-discount` | Guarantees confidence **decays per hop**. `chain-discount` (0.9) is factored out so the §8 tuning loop is a one-line change. |

The three axes are independent; the magnitude/confidence math is unchanged from
v0 — v1 only layers sign on top.

## 5. Inference & provenance

`infer` is two equations — a **base case** (any direct link, curated or
calibrated-empirical) and a **transitive case** (step through one curated
bridge, recurse). It is non-deterministic: one result per distinct path, and
**none** when the endpoints are unconnected (no false chains). It returns
`(signed <sign> (stv s c))`. `explain` runs the same recursion but also returns
the node path: `(Chain (n0 … nk) (signed <sign> (stv s c)))` — the auditability
LLMs cannot offer.

## 6. Two headline demos

**Novel cross-source inference (harmful axis).** Stated by *no single source*;
composed from the López-Otín hallmark concept, the curated SASP/PAI-1 bridges,
and the calibrated Lu 2019 link `DNAmPAI1 → CHD` (`HR 1.31 → (stv 0.6 0.85)`):

```metta
!(infer &self CellularSenescence CoronaryHeartDisease)
;; (signed Pos (stv 0.357 0.29089125))      ; senescence RAISES CHD risk
```

**Intervention effect with direction (Demo 2).** The senolytic D+Q *clears*
senescent cells (`Neg`); chaining that into the harmful axis flips the net sign,
so PLN derives that D+Q is expected to *reduce* CHD risk — with quantified
(modest) confidence and a full audit trail:

```metta
!(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)
;; (Chain (DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease)
;;        (signed Neg (stv 0.2499 0.17017138125)))
```

This signed, uncertainty-quantified, provenance-bearing intervention estimate is
exactly what an LLM cannot produce, and is the unit an intervention-ranking demo
sorts over (rank candidates by net protective magnitude × confidence).

> **Now built (Demo 2).** That ranking demo exists:
> `pln_intervention_ranking.metta` maps `infer` over a candidate pool, scores
> each (protective = higher, harmful = negative), and sorts. On
> `CoronaryHeartDisease` it ranks **D+Q ≻ Fisetin ≻ Spermidine** — D+Q over
> Fisetin on *identical mechanism* but stronger evidence tier, Spermidine last
> because its honest mechanism costs an extra hop (confidence decays per hop),
> and Elamipretide omitted (no chain — no false placement). See
> `docs/intervention_ranking.md`.

## 7. Open questions / next increments

1. ~~**Signed / protective effects.**~~ ✅ **Done in v1** (this layer). Sign is a
   first-class axis; protective interventions chain to a net `Neg`.
2. **Base rates** — add node strengths `P(B), P(C)` to upgrade the magnitude
   rule from the conservative product to the full PLN deduction formula.
3. **Stepping through empirical links mid-chain** — v1 treats empirical
   associations as terminal; generalise the transitive case to step through any
   lifted link, not only curated bridges.
4. **Revision for multiple paths** — when several derivations reach the same
   conclusion, combine them with a PLN revision rule instead of returning each.
   (Now also surfaced by the ranking demo, which lists one entry per derivation
   until this lands — `docs/intervention_ranking.md` §5.1.)
5. **Cycle guarding** — the recursion assumes a DAG; add visited-set guarding
   before the bridge graph can contain cycles.
6. **Mechanistic-consensus evidence tier** — curated bridges currently borrow
   `AnimalStudies_Replicated`; a dedicated tier would represent review-level
   mechanistic consensus better (calibration open-Q#4).
7. **Protective empirical evidence** — `hr->sign` already maps HR<1 to `Neg`,
   but the calibration strength bucket only covers HR>1; extend it for HR<1 so
   protective *empirical* associations (not just curated ones) calibrate.

## 8. Tuning loop ("poke the pipeline")

The loop the calibration doc anticipated is now runnable: pick a chain, read the
collapsed `(signed sign (stv s c))` from `infer`, and watch for (a) confidence
vanishing too fast with length, (b) implausible strengths, (c) wrong signs.
Then adjust `chain-discount` / `combine-tv` / `sign-product` (one file) or a
bridge prior and re-run the tests.

## 9. Non-goals

- Not numerically-tuned PLN (v0/v1 lock the architecture, not the constants).
- Not a rewrite of any raw evidence atom.
- Not yet abduction, counterfactuals, or revision (later inference patterns).
