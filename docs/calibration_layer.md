# Evidence → Truth-Value Calibration Layer — Design Draft

**Status:** DRAFT for team review. Companion stub implementation:
`evidence_calibration.metta`.
**Scope:** the policy that turns raw, source-faithful evidence records into PLN
truth values `(stv s c)` so deduction can consume them.

> **Dependency / base branch.** This layer reads the GrimAge evidence records
> (`EmpiricalAssociation`, `EffectMetric`, `EffectValue`, …) and the types that
> back them, which live on **`grimage_hallmarks`**. It is inert on any branch
> that lacks those atoms. The branch hosting this layer must therefore be based
> on `grimage_hallmarks` (or on `main` *after* `grimage_hallmarks` is merged) —
> **not** plain `main`.

---

## 1. Why we need it

PLN inference requires every link to carry a truth value `(stv s c)`:

- **strength** `s ∈ [0,1]` — how strongly the relation holds;
- **confidence** `c ∈ [0,1]` — how much evidence backs that strength.

But our knowledge base deliberately stores **raw measurements, not truth
values**:

```metta
;; grim_age_lu2019_evidence.metta
(EffectValue  Lu2019_AgeAccelGrim_AllCauseMortality 1.10)
(EffectScale  Lu2019_AgeAccelGrim_AllCauseMortality "per one-year increase in AgeAccelGrim")
(PValue       Lu2019_AgeAccelGrim_AllCauseMortality 2.0e-75)

;; drugage_etl_short.metta
(AvgLifespanChangePercent DrugAgeRow_0 81.29)
(AvgLifespanSignificance  DrugAgeRow_0 Unreported)
```

This is the correct design — note that **no merged ETL file invents an
`(stv …)`**; the only inline truth values in the whole project live in the
design docs (`pipeline.md`, `casestudy.md`). Keeping measurements raw makes
them immutable and auditable. The cost: those records are **inert** — nothing
converts them into the `(stv s c)` deduction needs.

The **calibration layer is that single, explicit, documented bridge.** It is
the one place where all "how do we turn a hazard ratio into a strength"
judgment lives, so the truth values are reproducible functions of the source
data rather than hand-typed magic numbers.

## 2. Where it sits in the pipeline

```
raw evidence records  ──calibrate──▶  truth-valued links (stv s c)  ──▶  PLN deduction / abduction
   (EmpiricalAssociation,                                                  (chains, propagates,
    DrugAge Experiment, …)                                                  ranks with uncertainty)
```

It **reads** evidence records and **emits** calibrated links. It never mutates
the raw records. It applies equally to GrimAge associations *and* DrugAge rows
(and, later, LLM-extracted relations) — it is general, not GrimAge-specific.

## 3. The map (precisely)

Yes — at core it is `calibrate : EvidenceRecord → (stv s c)`. The key subtlety
is that **`s` and `c` are computed from different fields**:

| Output | Driven by | Source atoms |
|---|---|---|
| strength `s` | effect size + scale + direction | `EffectMetric`, `EffectValue`, `EffectScale` |
| confidence `c` | evidence weight | `HasEvidenceCategory` (study/species), `PValue`, sample size |

`s` and `c` are **independent**: `p = 2.0e-75` makes us *certain the effect is
real* (high `c`), not that the effect is *large* (`s` stays modest at HR 1.10).

### 3.1 Strength ← effect size

A hazard ratio lives in `[0, ∞)`, so it must be squashed into `[0,1]`. Two
things the transform must respect:

- **Scale** (`EffectScale`): HR 1.10 *per one year* of AgeAccelGrim is not
  comparable to HR 1.31 *per standard deviation* of DNAmPAI1. The policy has to
  read the scale to make effects commensurable.
- **Direction**: a protective association (HR < 1) and a harmful one (HR > 1)
  of equal magnitude need opposite strength semantics depending on how the
  target link is phrased.

The stub uses coarse buckets; v1 candidates are `s = 1 − 1/HR` or a logistic on
`log(HR)`.

### 3.2 Confidence ← evidence weight

This axis **reuses `epistemic_calibration.metta`**, which already encodes the
table:

```
RCT_Human                1.00
MultipleHumanTrials      0.85
AnimalStudies_Replicated 0.65
AnimalStudies_Single     0.50
InVitro                  0.35
TraditionalUse           0.20
```

**This is where "an effect shown only in mice is weaker" lives:** a mouse /
lower-tier category maps to a smaller `evidence-confidence` than a human RCT, so
the calibrated link inherits lower `c`. (Species arguably also dents `s`, not
just `c` — see open question 2.) A `PValue` gate and/or sample-size term can be
combined in by taking the **min** (weakest link governs).

## 4. It reuses existing machinery

`epistemic_calibration.metta` already supplies `evidence-confidence`,
`selection-basis-confidence`, and `apply-tv`. The calibration layer is **glue**
on top of that table, not new infrastructure.

## 5. Stub policy v0

`evidence_calibration.metta` ships a deliberately dumb v0:

- `s`: coarse HR buckets (`>1.5 → 0.70`, `>1.1 → 0.60`, else `0.50`).
- `c`: straight lookup of the record's evidence category in
  `evidence-confidence`.

The entry point is `(calibrate-tv <space> <record>)` — see §7.1 for why the
space is an explicit parameter. It is marked `STUB`: the point of v0 is **not**
to be numerically right — it is to (a) lock the architecture and (b) make the
clock atoms inference-ready, so the policy can be tuned against real pipeline
behaviour instead of in the abstract.

## 6. Input contract

Every calibratable record must carry an evidence category:

```metta
(HasEvidenceCategory <record> <EvidenceCategory>)
```

The Lu 2019 records don't have this yet. The stub adds the tags **additively**
(in `evidence_calibration.metta`, without editing the source evidence file),
mapping them to `MultipleHumanTrials`. Going forward, ETL and extraction should
emit this tag at write time.

## 7. Open architecture questions (please settle)

1. **TV storage convention** — *how* is `(stv s c)` attached to a link? Options:
   (a) type annotation `(: (Predicts A B) (stv s c))`; (b) a wrapper
   `(TruthValue (Predicts A B) (stv s c))`; (c) `calibrate` stays a **function**
   called at query time and nothing is stored. This **must match how the
   not-yet-built deduction rule reads truth values** — so it should be decided
   *together with* the first inference rule, not now. The stub commits to
   nothing: `calibrate-tv` is a pure function returning the `(stv s c)`.
2. **Species/model discount** — confidence only, or also strength?
3. **Strength transform** — bucket vs `1 − 1/HR` vs logistic; and how to
   normalize across effect scales (per-year vs per-SD).
4. **Meta-analysis tier** — Lu 2019 is meta-analytic across cohorts; add a
   `MetaAnalysis_Human` category (~0.90) above `MultipleHumanTrials`?
5. **Negative evidence** — represent ITP failures as strength ≈ 0, or via a
   distinct predicate? (Affects e.g. `Extends Resveratrol Lifespan`.)
6. **Multiple records, one link** — aggregation / PLN revision rule when several
   evidence records bear on the same relation.

## 8. Tuning loop ("poke the pipeline")

Once deduction exists: run a known multi-step chain (e.g. Rapamycin → … →
GrimAge), inspect the collapsed `(s, c)`, and watch for (a) confidence
vanishing too fast with chain length, (b) implausible strengths. Then adjust the
handful of rules in `evidence_calibration.metta` and re-run. Because all the
judgment lives in one file, tuning is local and auditable.

## 9. Non-goals

- Not validated numbers (v0 is a stub).
- Not a modification of any raw evidence record.
- Not the deduction engine itself.

## 10. Verification & integration notes (hyperon 0.2.10)

The stub has been run against the live interpreter (`hyperon 0.2.10`). Loading
`system_types`, `logical_predicates`, `epistemic_calibration`, `grim_age_core`,
`grim_age_lu2019_evidence`, `evidence_calibration` and querying:

```
!(calibrate-tv &self Lu2019_AgeAccelGrim_AllCauseMortality)  =>  (stv 0.5 0.85)   ; HR 1.10
!(calibrate-tv &self Lu2019_AgeAccelGrim_CHD)                =>  (stv 0.5 0.85)   ; HR 1.07
!(calibrate-tv &self Lu2019_AgeAccelGrim_CHF)                =>  (stv 0.5 0.85)   ; HR 1.10
!(calibrate-tv &self Lu2019_DNAmPAI1_CHD)                    =>  (stv 0.6 0.85)   ; HR 1.31
```

### 10.1 The cross-module `&self` gotcha (applies to the deduction rule too)

A function defined in an imported module that does `(match &self …)` reads only
**that module's own space**, not the merged top-level space. So an accessor like
`(match &self (EffectValue $a $v) $v)` defined in `evidence_calibration` returns
nothing for records that live in `grim_age_lu2019_evidence` — a silent empty
result, not an error.

**Fix:** pass the space in explicitly. The accessors and `calibrate-tv` take a
space parameter, and callers pass the top-level `&self`:

```metta
(= (ev-value $s $a) (match $s (EffectValue $a $v) $v))
!(calibrate-tv &self Lu2019_AgeAccelGrim_AllCauseMortality)
```

**The deduction rule will hit the same trap** when it reads KB atoms to chain
truth values — it must thread the space the same way, not hard-code `&self`.
