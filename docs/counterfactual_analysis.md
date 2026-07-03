# Counterfactual GrimAge Analysis — Demo 4 (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`pln_counterfactual.metta`; the GrimAge composition weight lives there as a single
labelled knob; regression tests in `tests/test_counterfactual.py` (21 tests). Wired
into the chat app's inference stack (`pln_chat/app.py`) with `decompose-grimage` /
`counterfactual-patient` / `grimage-share` few-shots; ten executed chat queries in
`docs/counterfactual_chat_test_queries.md`.
**Scope:** `pipeline.md` §4.4 (Demo 4) — for a grounded patient, (a) **decompose**
an elevated GrimAge into per-component attributable shares traced to upstream
causes, and (b) compute, for a normalizing intervention on a chosen **lever**, the
**expected signed change** in GrimAge with propagated uncertainty.

> **Dependency / base branch.** Pure consumer of the deduction keystone
> (`pln_deduction.metta` → `infer`), the curated bridges, the GrimAge ontology
> (`grim_age_core.metta` → `PartOf … GrimAge`), and the patient-grounding layer
> (`patient_profile.metta` → `MeasuredZ` / `elevated-z-threshold`). Load **last**
> (see the file header for the exact order). It reuses the `Neg = beneficial`
> sign convention (`rank-score`) rather than forking it.

---

## 1. The gap this closes

The engine could already reason **forward** — `(infer $space A C)` gives a signed,
uncertainty-quantified transmission `A → C` — and could **ground** a real patient
via `(MeasuredZ Patient001 marker z)`. But it could not answer Demo 4's
**counterfactual**: *"if this patient's inflammation were normalized, what is the
expected change in GrimAge?"* Two pieces were missing:

1. **No GrimAge composition edge.** GrimAge is a composite — `grim_age_core.metta`
   declares `(PartOf <surrogate> GrimAge)` for 8 DNAm surrogates and
   `(ResidualOf AgeAccelGrim GrimAge)` — but nothing aggregated a change in the
   components into a change in the clock. So "expected change in GrimAge" was
   literally uncomputable, and `patient_profile.metta` noted AgeAccelGrim was
   "carried un-credited … no bridge explains it."
2. **No counterfactual operator.** `infer` gives a net transmission; nothing
   performed `do(lever := baseline)` — fix an upstream node to a target, propagate
   the resulting **delta** to a downstream outcome, and report the expected signed
   change with propagated confidence.

This layer adds both.

## 2. Where it sits in the pipeline

```
MeasuredZ (patient) ─┐
                     ├─decompose-grimage─▶ per-component shares + upstream causes + residual
PartOf … GrimAge ────┘

lever ─resolve-lever─▶ harmful driver D ─infer─▶ per-component transmission ─┐
MeasuredZ (elevated z) ─────────────────────────────────────────────────────┼─grimage-weight─▶
                                                                             ▼
              (Counterfactual lever outcome (expected-delta d) (signed Neg (stv s c)) (Via (…)))
```

It **reads** the causal graph and the patient's measured elevations and **emits** a
decomposition or a counterfactual estimate. It mutates nothing.

## 3. Design

### 3.1 The GrimAge composition edge (Demo §5.1) — one curated knob

`grimage-weight` maps a Δ in one DNAm surrogate (in its own SD units) into a Δ in
GrimAge acceleration (in GrimAge SD units). This is a **curated strength** — like
the mechanistic bridge priors — so it is quarantined in a single, clearly-labelled
knob (the "one place curated strengths live" discipline of
`mechanistic_bridges.metta`), never mixed into a raw evidence file.

v1 is deliberately **coarse and honest**: equal weights over the 8 `PartOf`
surrogates (`0.125` each; they sum to 1.0, so if every component were elevated by
the same z the clock would move by that z). A natural refinement is
`HasCpGCount`-proportional weighting — a one-line change to this one function, left
as an open question (§7). **v1 locks the architecture, not the constant.**

### 3.2 Decomposition — `decompose-grimage $space $patient`

For each GrimAge component the patient presents **elevated** (measured `z` above
`elevated-z-threshold`), report its attributable share and the upstream cause(s)
that raise it:

```
(Component <M> (z z) (weight w) (contribution w·z) (DrivenBy (<hallmark causes>)))
```

The causes are found by running `infer` **backward per component** (the abductive
`explains-obs` pattern: keep hallmark `H` iff `H → M` is `Pos`). The driver totals
are aggregated and compared with the measured AgeAccelGrim to report an honest
**unattributed residual**. A component with no measured elevation or no causal
driver is simply absent from the list and falls into the residual — **never
invented**.

### 3.3 Lever resolution — the CRP-as-co-effect subtlety

`do($lever normalized)` is really "normalize the harmful upstream **driver** the
lever stands for". `resolve-lever` routes three lever shapes to that driver, so the
do-operator is uniform (it always normalizes a `Pos`-transmitting driver and never
reasons about a fictitious marker → clock edge):

| lever shape | detected by | resolves to | example |
|---|---|---|---|
| **intervention** | outgoing `(Effect lever D Neg …)` | the node `D` it reduces | `DasatinibPlusQuercetin → CellularSenescence`; `Metformin → InsulinResistance` |
| **leaf marker** | no outgoing Effect, incoming `(Effect D lever Pos …)` | its shared cause `D` | `CRP → ChronicInflammation` |
| **driver** | outgoing `Pos` edges | itself | `ChronicInflammation`, `CellularSenescence`, `InsulinResistance` |
| **edge-less** | no Effect edge at all | *(nothing)* → omitted | `Elamipretide` |

**This is why the demo is PLN-not-LLM.** `CRP` is a **leaf**: the only edge is
`(Effect ChronicInflammation CRP Pos …)`. CRP is a downstream **readout**, not a
GrimAge component, and has no outgoing edge — so *"if my CRP were normal"* has **no
direct mechanism** to GrimAge. The correct causal reading is that CRP and the
inflammation-attributable GrimAge component (DNAmGDF15) are **co-effects of a shared
cause** (ChronicInflammation). `resolve-lever` routes the CRP lever **through** that
cause — never via an invented `CRP → GrimAge` edge — and the estimate is credited to
DNAmGDF15. An LLM asked "lower my CRP → GrimAge?" will happily invent a direct
number; PLN credits the mechanism it can actually trace.

### 3.4 The do-operator — `counterfactual $space $patient $lever $outcome`

After resolving to a driver `D`, for each GrimAge component `M` that `D` **raises**
(`Pos` transmission) **and** the patient presents **elevated**:

- **expected reduction in M** ≈ `(transmission strength) × (M's elevated z)` — the
  part of M's elevation attributable to the driver, removed when the driver is
  normalized;
- aggregate into a GrimAge Δ via `grimage-weight`:
  `expected-delta = −Σ_M grimage-weight(M) · s(D→M) · z_M` (negative = a reduction);
- **confidence** = the contribution-weighted mean of the per-component transmission
  confidences. Because `infer` attenuates confidence **per hop** (`chain-discount`),
  a component reached through a longer route contributes a lower `c` — so a lever
  whose components sit further downstream yields a **lower** counterfactual
  confidence.

The result reuses the `Neg = beneficial` convention (`rank-score`,
`pln_intervention_ranking.metta`): a **reduction** in a harmful clock is
**protective**, so the estimate is `(signed Neg …)` with a **negative**
`expected-delta`. Strength is the magnitude of that reduction (so the estimate is
`rank-score`-able). A lever that reaches no clock surrogate returns
`expected-delta 0` with an empty `(Via ())` — honest, not fabricated; an
unresolvable lever yields nothing at all.

```
(Counterfactual <lever> <outcome> (expected-delta d) (signed Neg (stv s c)) (Via (M …)))
```

`counterfactual-patient` defaults the outcome to `AgeAccelGrim`;
`counterfactual-scenarios` runs the three casestudy §5.2 scenarios in one call.

## 4. The headline demo (real numbers, `Patient001`)

`Patient001`: elevated DNAmPAI1 (z 1.8), DNAmGDF15 (z 1.5), CRP (z 1.7), and
AgeAccelGrim (z 1.6) — the case-study discordance (normal first-gen Horvath clock,
elevated mortality-predictive GrimAge).

**Decomposition** (`decompose-grimage &self Patient001`):

```
(Component DNAmPAI1  (z 1.8) (weight 0.125) (contribution 0.225)
    (DrivenBy (CellularSenescence)))
(Component DNAmGDF15 (z 1.5) (weight 0.125) (contribution 0.1875)
    (DrivenBy (CellularSenescence ChronicInflammation MitochondrialDysfunction)))
attributed 0.4125 ; residual 1.1875
```

(Order within a `Components` / `DrivenBy` list is collapse-order — not significant;
the tests compare sets. See `docs/counterfactual_chat_test_queries.md` for the
verbatim runs.)

**The three scenarios** (`counterfactual-patient &self Patient001 …`):

| Scenario | lever | reaches | expected-delta | confidence | sign |
|---|---|---|---|---|---|
| **A** reduce inflammation | `ChronicInflammation` | DNAmGDF15 (1 hop) | **−0.1219** | 0.65 | Neg |
| **B** clear senescence | `CellularSenescence` | DNAmPAI1 + DNAmGDF15 (2–3 hops) | **−0.2219** | 0.318 | Neg |
| **C** metabolic | `InsulinResistance` | — (no clock surrogate) | **0.0** | 0.0 | Neg |

Three reads no LLM produces from first principles:

1. **B > A** — clearing senescence reduces GrimAge **more** than reducing
   inflammation alone (`0.222 > 0.122`), because senescence is **upstream** of
   inflammation and reaches **both** clock surrogates, not just DNAmGDF15. A
   quantified, mechanism-grounded ordering, not a vibe.
2. **B is the less certain estimate** — its bigger effect is reached through
   2–3-hop routes, so its confidence (`0.318`) is **below** A's single-hop `0.65`.
   Uncertainty grows with route length, and PLN reports it.
3. **C ≈ 0, honestly** — the metabolic axis terminates at `FastingGlucose → CHD`
   and reaches no GrimAge surrogate, so PLN declines to move the clock (empty
   `Via ()`), the exact analogue of Elamipretide being omitted from the CHD
   ranking. It does **not** fabricate a plausible-sounding number.

And the killer, `counterfactual-patient &self Patient001 CRP`:

```
(Counterfactual CRP AgeAccelGrim (expected-delta -0.121875)
   (signed Neg (stv 0.121875 0.65)) (Via (DNAmGDF15)))
```

The lever token is CRP, but the credited mechanism is the **inflammation** CRP is a
co-effect of — never a direct `CRP → GrimAge` edge (§3.3).

## 5. Why LLMs fail here (what the demo shows)

- **Formal counterfactual/do-inference** over a causal model, not a plausible
  narrative — a deterministic function of calibrated evidence and graph structure.
- **Decomposition into attributable components** with quantified shares and an
  explicit residual for what the model cannot yet explain.
- **Propagated confidence** — a longer causal route yields a lower confidence on the
  estimate (B's 0.318 < A's 0.65).
- **The restraint** — it declines to move GrimAge for a lever with no mechanistic
  path to a clock surrogate (metabolic axis → 0), and routes "reduce CRP" through
  the shared cause instead of inventing a direct effect. An LLM shows neither.

## 6. Implementation notes (hyperon 0.2.10)

- **Thread `$space` explicitly.** Every accessor takes the space and callers pass
  the top-level `&self`; a bare `(match &self …)` inside this imported module reads
  only its own space and silently returns empty (`calibration_layer.md` §10.1).
- **`let*`-force every arithmetic / list intermediate.** Grounded ops
  (`+ - * / > == car-atom/cdr-atom/cons-atom`) do **not** evaluate their arguments,
  so each intermediate is forced before it feeds the next step — the same discipline
  the deduction / ranking / abduction layers document. (E.g. a nested
  `(cons-atom a (cons-atom b ()))` leaves the inner call unreduced unless forced.)
- **Determinism.** `infer` is non-deterministic (one result per derivation); the
  per-component transmission is collapsed and `car-atom`-forced to one value, and
  every equation is single-rule + `if`, so identical inputs give identical output —
  a headline PLN property.
- **No panic, ordinary path.** `pln_counterfactual.metta` is small and on the
  focused senescence/CHD stack, so Demo 4 rides the ordinary
  `run_query(_ALL_KB_PATHS)` path — **no scoped space needed** (the key difference
  from the DrugAge slice, which must be query-scoped to stay under the hyperon
  type-index panic; `intervention_ranking.md` §6). Verified: the counterfactual
  queries run cleanly against the full 19-file `_ALL_KB_PATHS`.
- **Float noise.** Live floats carry trailing noise; tests compare with tolerance.

## 7. Open questions / next increments

1. **Base rates.** Like the deduction layer (`deduction_layer.md` §7.2), the
   magnitude uses the conservative independence product without node priors
   `P(B), P(C)`. Adding them upgrades the transmission — and hence the delta —
   toward the full PLN deduction formula.
2. **Multiplicative magnitude model.** v1's `reduction ≈ strength × z` is a
   transparent linear model. A saturating / multiplicative form (a large strength
   should not remove more than the measured elevation) is the documented refinement.
3. **Composition weights.** The equal-weight `grimage-weight` is a coarse v1
   constant; `HasCpGCount`-proportional (or a fitted per-surrogate weight) is a
   one-line change to that one knob.
4. **Metabolic → clock bridge.** Scenario C is honestly 0 because the metabolic
   axis reaches no clock surrogate. If a future curated bridge links the metabolic
   axis to a DNAm surrogate (e.g. glucose → an inflammatory surrogate), C becomes
   non-trivial — but v1 prefers the honest 0 over a speculative edge.
5. **Multi-lever counterfactuals.** v1 normalizes one lever at a time. Combining
   levers (clear senescence **and** reduce inflammation) needs care not to
   double-count the shared DNAmGDF15 path (senescence already routes through
   inflammation) — a revision/inclusion-exclusion rule, shared with the deduction
   layer's multiple-path follow-up (`deduction_layer.md` §7.4).
6. **Multiple derivations per component.** As elsewhere in v1, one derivation per
   (driver, component) pair is used; PLN revision over multiple paths is the
   documented follow-up.

## 8. Non-goals

- Not numerically-tuned constants (v1 locks the composition edge + do-transform;
  the weights and the reduction model are documented knobs).
- Not a modification of any raw evidence record or the causal graph (it reads them).
- Not a new inference engine — it is a consumer of `infer` and the patient grounding.
- Not a general do-calculus (no back-door adjustment / confounding correction yet);
  v1 is a forward propagation of a normalizing intervention over a curated DAG.
