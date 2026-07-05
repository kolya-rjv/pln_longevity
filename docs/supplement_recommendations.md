# Personalized Supplement Recommendations — Demo 6 (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`pln_supplement_recommendation.metta` (the engine) + `supplement_evidence.metta`
(the curated Layer-4 knowledge); regression tests in
`tests/test_supplement_recommendation.py` (14 tests); executed chat queries in
`docs/supplement_chat_test_queries.md`.
**Scope:** the last unimplemented demonstration scenario from `pipeline.md` (§4.6,
Demo 6) — turn a grounded patient + a candidate supplement pool into a **tiered,
personalized** recommendation that respects the weaker supplement evidence base,
vetoes negative-trial compounds, flags drug–supplement interactions, and omits
supplements irrelevant to the patient's profile.

> **Dependency / base branch.** Sits on the whole inference + patient stack. It adds
> **no new inference power** — it is a pure consumer of `patient-relevance` /
> `relevance-on` / `rank-score` / `infer` (patient grounding, Demo 2) and the curated
> Layer-4 atoms. Load last (see the file header for the exact order). This is the same
> "application on top of a primitive" shape as `pln_intervention_ranking.metta`.

---

## 1. Where this sits

Demos 1–5 reason over pharmaceutical / hallmark evidence and answer *diagnostic* and
*causal* questions. Demo 6 is the *recommendation* capstone. The pipeline defines a
**fourth knowledge layer** for it (`pipeline.md` §2.1.4): supplements and
nutraceuticals, represented with **appropriately lower confidence** than
pharmaceuticals, plus the modifier atoms (evidence level, safety, interactions) a
responsible recommender must carry. The hooks were left waiting in the KB:
`logical_predicates.metta` **declares `Interaction` but populates it with zero facts**,
and lists `EvidenceLevel` / `SafetyProfile` in its §9 *"FUTURE PREDICATES (stubs for
later phases)."* This demo is that phase.

The personalization primitive already existed: `patient-relevance`
(`patient_profile.metta`, Demo 2) scores an intervention by how much it *protectively
reduces the markers a patient presents elevated*. Demo 6 rides it directly — a
supplement's relevance to a patient **is** its `patient-relevance`.

## 2. What it computes

```
recommend-supplements : space × patient × (candidate …)
   ⊢  (SupplementRecommendation <patient>
         (Tier1HighConfidence (<SuppRec …>))     ; strong human evidence
         (Tier2Promising      (<SuppRec …>))     ; preliminary / animal
         (NotRecommended      (<SuppRec …>))     ; negative-trial veto
         (Interactions        (<InteractionFlag …>)))
```

Each `SuppRec` is
`(SuppRec <supp> (tier T) (score s) (evidence <cat> <conf>) (safety <label> <f>) (Targets (M …)))`:

- **score** = `patient-relevance` × `safety-factor`. Relevance personalizes (protective
  effect across *this* patient's elevated markers); the safety factor de-rates within a
  tier. A candidate with relevance 0 — reaching none of the patient's elevated markers —
  is **omitted** (no irrelevant recommendation).
- **tier** = a documented map from the candidate's `EvidenceLevel` category to
  `Tier1HighConfidence` / `Tier2Promising` / `NotRecommended`. Strong human evidence
  (RCT / multiple trials / ITP-positive) → Tier1; weaker-but-non-negative → Tier2; and
  **`ITP_Negative` → `NotRecommended`** — the veto.
- **evidence** = the tier's confidence, read from the single `evidence-confidence`
  authority. Every supplement tier lands **below** RCT-grade — the Layer-4 honesty point.
- **Targets** = the patient's elevated markers the supplement protectively reaches — the
  provenance an LLM cannot produce.

## 3. The two design subtleties (why this is PLN-not-LLM)

### 3.1 The negative-evidence veto

Resveratrol is given a **plausible** senescence-clearing `Effect` link (its SIRT1
story), so by *mechanism* it is relevant to a senescence-dominant patient — `Targets`
comes back non-empty and `patient-relevance > 0`. An LLM recommends it on exactly that
mechanism. But it also carries `(EvidenceLevel Resveratrol ITP_Negative)`: the NIA
Interventions Testing Program found **no** lifespan effect (`pipeline.md` App. B.3). The
tier map turns that into `NotRecommended`, overriding the mechanism. The restraint is
the point — PLN declines on the negative trial rather than recommending on the
mechanism. (`test_resveratrol_is_vetoed_despite_a_relevant_mechanism`.)

Note this is **high confidence in a negative result** (`ITP_Negative` = 0.90 in the
calibration authority): we *trust* a well-run null. The same asymmetry down-weights
resveratrol on the DrugAge lifespan axis via near-zero *strength*
(`drugage_calibration.metta` §10); here it acts as a categorical veto.

### 3.2 Personalization reorders across patients

The demo's headline. The **same five-supplement pool** recommends differently for two
patients with the same surface finding (elevated GrimAge) but different drivers:

| candidate | Patient001 (senescence/inflammation) | Patient002 (metabolic) |
|---|---|---|
| **Omega3** | Tier1 — targets `CRP`, `DNAmGDF15` | omitted (no inflammation elevation) |
| **Fisetin** | Tier2 — targets `CRP`, `DNAmGDF15`, `DNAmPAI1` | omitted |
| **NMN** | Tier2 — targets `DNAmGDF15` | omitted |
| **Resveratrol** | NotRecommended (vetoed) | omitted |
| **Berberine** | **omitted** (metabolic — irrelevant here) | **Tier1** — targets `FastingGlucose`, `HbA1c` |

Relevance **reorders**, it does not merely amplify — Berberine moves from omitted to
Tier1 across patients. This is `pipeline.md` §4.6's "not personalized to biomarker
profile" LLM failure mode, inverted.

## 4. The headline demo, with real numbers

`!(recommend-supplements-patient &self Patient001)` (58yo male, senescence + inflammation):

```
Tier1HighConfidence : Omega3      score 0.4326  evidence MultipleHumanTrials 0.85  Targets (CRP DNAmGDF15)
Tier2Promising      : NMN         score 0.1689  evidence SingleHumanTrial     0.70  Targets (DNAmGDF15)
                      Fisetin     score 0.1343  evidence AnimalStudies_Single 0.50  Targets (CRP DNAmGDF15 DNAmPAI1)
NotRecommended      : Resveratrol score 0.0723  evidence ITP_Negative         0.90  Targets (CRP DNAmGDF15 DNAmPAI1)
Interactions        : ()                                                            ; no current medications
```

Berberine is absent — irrelevant to this patient. `!(recommend-supplements-patient
&self Patient002)` (metabolic, on metformin):

```
Tier1HighConfidence : Berberine   score 0.3879  evidence MultipleHumanTrials 0.85  safety UseWithCaution 0.8  Targets (HbA1c FastingGlucose)
Tier2 / NotRecommended : ()
Interactions        : (InteractionFlag Berberine Metformin "Shared AMPK activation — additive glucose lowering; …")
```

Four reads an LLM cannot reproduce from first principles:

1. **Omega3 ≻ Fisetin ≻ NMN by evidence, not vibe** — the tier and confidence are a
   deterministic function of the calibrated evidence category.
2. **Resveratrol is declined despite a relevant mechanism** — the veto (§3.1).
3. **Berberine appears only for the metabolic patient** — personalization (§3.2).
4. **The interaction fires only against the patient's actual medication** — Patient002's
   metformin; Patient001 (no meds) gets an empty `Interactions` block.

## 5. Implementation notes (hyperon 0.2.10)

- **Rides the existing focused stack; chat surfacing is cheap.** Both files are small
  and on the same senescence/CHD stack Demos 4 & 5 use, so Demo 6 rides the ordinary
  `translate()` → `run_query(_ALL_KB_PATHS)` path — **no scoped space, no `chat()`
  routing** (unlike the DrugAge slice, which trips the full-KB panic;
  `docs/intervention_ranking.md` §6). The two files are added to `_INFERENCE_STACK` in
  `pln_chat/app.py` so the translator sees the new functions and the symbol validator
  accepts them; the 10 executed queries in `docs/supplement_chat_test_queries.md` run
  against the full 22-file KB and match the focused-stack results exactly.
- **New edges are inert for existing queries.** Every new `Effect` link
  (`Omega3 → ChronicInflammation`, `NMN → MitochondrialDysfunction`,
  `Resveratrol → CellularSenescence`) adds an **outgoing** edge from a **new** supplement
  node — none adds an incoming edge to a node existing forward queries traverse from — so
  `infer` / `counterfactual` / `diagnose` from existing nodes are unchanged. All 107
  pre-Demo-6 tests stay green.
- **Curated strengths are quarantined.** Supplement magnitude priors live only in
  `supplement_evidence.metta`, clearly labelled, with confidence derived from the
  `evidence-confidence` authority — the `mechanistic_bridges.metta` honesty contract.
- **`$space` threaded explicitly; grounded ops don't evaluate their args.** Every
  arithmetic / list intermediate is `let*`-forced; single-rule + `if` keeps the tiering
  and the score sort deterministic ("identical inputs → identical output").
- **One `EvidenceLevel` / `SafetyProfile` per supplement** keeps the tier and safety
  lookups deterministic (a second atom would make them non-deterministic).

## 6. Open questions / next increments

1. **Calibrated supplement effect sizes.** v1's supplement `Effect` strengths are
   curated magnitude priors. The honest upgrade is to derive them from the DrugAge ETL
   the way `drugage_calibration.metta` already lifts calibrated, signed lifespan effects
   (NMN / Fisetin / Resveratrol all have real DrugAge rows) — but that vertical currently
   lives in its own **scoped** space to dodge the full-KB panic
   (`docs/etl_inference_wiring.md` §8.5), so wiring calibrated supplement evidence into
   this focused recommender is a per-vertical merge, not a free import.
2. **Tier boundary + safety factors are coarse knobs.** The `EvidenceCategory → tier`
   map and the three safety multipliers are documented v1 constants; a richer scheme
   (e.g. `pipeline.md` App. B.6's per-evidence-level modifiers, or a continuous
   confidence threshold) is a one-block change to §0.
3. **Score is `relevance × safety`; mechanism strength is folded into relevance.**
   `pipeline.md` §4.6 phrases the rank as "relevance × mechanism strength × safety". v1
   folds mechanism strength × confidence into `patient-relevance` (via `infer`); breaking
   it out as an explicit third factor is a refinement.
4. **Dosage, formulation, contraindications.** `EnhancedBy` (bioavailability) and
   `Limitation` are declared but unused; per-patient contraindications beyond
   drug–supplement interactions (renal / hepatic status) need patient fields not yet
   modeled.
5. **User risk-tolerance / preference encoding.** `pipeline.md` §4.6 step 1 wants an
   explicit preference statement ("even if evidence is preliminary…") to gate the Tier2
   band. v1 always returns all tiers; a preference filter is a thin wrapper.
6. **Interactions are one-directional facts + a med list.** A supplement–supplement
   interaction matrix and severity levels are natural extensions of the `Interaction`
   predicate.

## 7. Non-goals (v1)

- Not a dosing engine, not medical advice — the output is uncertainty-tagged guidance to
  discuss with a clinician, exactly the framing `pipeline.md` §4.6 specifies.
- Not a re-calibration of the evidence tiers — it consumes the single
  `evidence-confidence` authority unchanged.
- Not a new inference primitive — it is an application over `patient-relevance` / `infer`.
