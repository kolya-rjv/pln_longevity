# NHANES Integration — Design (v1)

**Status:** v1. Correctness core (`nhanes_common.py`) verified against independent
references; regression-guarded in `tests/test_nhanes_common.py`. **No NHANES-derived
number is committed to this repository** — see §2, which is the single most important
thing to understand about this layer.
**Scope:** two increments the existing docs both name as the next step.

- **Step 1** — NHANES survey-weighted **reference distributions**, and the MeTTa layer
  that turns a patient's **raw lab values** into the standardized `MeasuredZ` the
  inference stack already consumes. This closes `docs/patient_grounding.md` §6 open-Q #1.
- **Step 2** — NHANES + the **Linked Mortality File** → data-backed **absolute baseline
  risk** by age band × sex, wired into `pln_risk_prediction.metta` alongside the curated
  prior. This addresses `docs/risk_prediction.md` §6 open-Q #1, though **not** in the way
  that open question assumes — see §5, which is the other thing to understand.

> **Dependency.** The MeTTa layers are consumers of the existing stack. The ETLs are
> standalone Python (pandas/numpy only) and are never imported by the inference layers.

---

## 1. Why `MeasuredZ` needed this

`patient_profile.metta` defines a patient as a set of standardized measurements:

```metta
(MeasuredZ Patient001 CRP 1.7)   ; ~1.7 SD above the age/sex-adjusted norm
```

and `docs/patient_grounding.md` §3 justifies that convention partly on the grounds that
it "is the number **NHANES / a blood panel actually yields**, so the eventual NHANES
import is a data-loading step, not a redesign".

That is true of the *shape*. But nothing in the repo supplied the **means and SDs** the
`z` is measured against, so the two example patients' values are hand-typed. A real
person with a real lab report could not be entered at all: you cannot convert
`CRP = 0.82 mg/dL` into a `z` without a reference distribution.

Step 1 supplies that distribution, and adds `MeasuredRaw` so a lab report can be entered
in its own units and standardized by rule rather than by hand.

## 2. The constraint that shaped everything: no NHANES data here

The environment this was written in **denies `wwwn.cdc.gov` and `ftp.cdc.gov`** at the
network egress policy (403 on CONNECT). NHANES microdata could not be downloaded, so no
NHANES-derived value could be computed *or verified*.

The response was not to approximate. It was:

- **Repo-root `.metta` files added here contain types, accessors and rules only.** No
  numeric reference value, no baseline risk — not transcribed from a paper, not marked
  "illustrative", not behind a `TODO`. A number in a `.metta` file is consumed by
  inference and becomes a claim, and this repository's whole thesis is that it does not
  make claims it cannot source.
- **All numeric atoms are ETL build artifacts**, written under `build/` (gitignored),
  exactly as `scripts/run_etl.sh` already stages the HAGR ETL outputs.
- **The machinery is proved against synthetic fixtures** whose values are arbitrary by
  construction, so the tests demonstrate correctness without shipping a fake reference.
- **Every NHANES file name and variable name is treated as an unverified claim.** They
  live in one registry per ETL, each entry carrying a confidence; `--show-registry` prints
  them; `--inspect` prints a file's actual columns; a missing variable aborts and lists
  the columns that *do* exist; `--registry` overrides an entry without editing code. A
  wrong guess is therefore loud and user-fixable, never a source of quiet wrong numbers.
  There is deliberately **no fallback to a similar variable name** — CDC explicitly warns
  against substituting `LBXSGL` for `LBXGLU`, `LBXSCH` for `LBXTC`, `LBXSTR` for `LBXTR`,
  and a helpful-looking fallback is exactly how such a substitution would happen.

To produce actual numbers, run the ETLs on a machine that can reach CDC. `data/nhanes/`
carries the manifest and the download instructions; nothing under it is bundled.

## 2b. What ships

| file | what it is |
|---|---|
| `nhanes_common.py` | the correctness core — weighted statistics, the file readers, budgets, scales |
| `nhanes_reference_etl.py` | step 1: blood analytes → `ReferenceDistribution` records |
| `nhanes_mortality_etl.py` | step 2: demographics + linkage → `BaselineRiskRecord` records |
| `nhanes_dnam_etl.py` | methylation clocks → clock references + the `ClockAccelSpread` record |
| `nhanes_reference.metta` | types and rules: `MeasuredRaw`, the reference lookup, `standardize-z` |
| `nhanes_baseline.metta` | types and rules: the baseline lookup, `HeartDiseaseMortality`, the outcome guard |
| `data/nhanes/` | `MANIFEST.tsv` and a README — which files to fetch, and from where |
| `tests/nhanes_xport_writer.py` | a SAS XPORT v5 writer, so the `.XPT` path is genuinely tested |
| `tests/test_nhanes_common.py`, `tests/test_nhanes_integration.py` | the statistics, and the whole chain end to end |

The three ETLs write to `build/` by default, which is gitignored, and
`scripts/run_etl.sh` runs them when data is present and explains what to fetch when it is
not. Both hand-written `.metta` layers are registered in the chat app's inference stack.

## 3. What the statistics had to get right

These are in `nhanes_common.py`, each verified against an independent reference rather
than against its own output (`tests/test_nhanes_common.py`).

**Survey weighting.** NHANES oversamples by design, so an unweighted mean estimates the
sample mixture, not the US population. The weighted mean is a Hájek ratio estimator,
`Σwx / Σw`.

**The SD denominator.** `Σw(x−x̄)²/Σw`, the finite-population form — not `Σw−1`. Survey
weights are probability weights, so `Σw` estimates a population in the hundreds of
millions; a `Σw−1` "bias correction" is both numerically inert and a claim that hundreds
of millions of independent observations were made. The population form is also the one we
*want*: the z asks "how many SDs above the **population** mean does this patient sit", so
the population SD is the correct denominator.

**Two passes, not one.** The one-pass identity `E[wx²] − E[wx]²` cancels catastrophically
at the scale of, say, total cholesterol (mean ~200, SD ~40) and can return a negative
variance. Verified stable at 1e8 scale.

**Design-based standard errors.** This reverses an earlier decision in this same work,
and the reversal is worth stating rather than leaving for a reader to notice. The first
pass deliberately emitted *no* standard error, on the grounds that a correct NHANES SE
needs design-based linearization and that shipping a naive one would be worse than
shipping none. That reasoning was right; the premise was wrong. The naive SE ignores
clustering (which inflates variance) and stratification (which deflates it). The Taylor linearization of the ratio
estimator under the ultimate-cluster approximation needs only the masked variance stratum
and PSU columns, which live in the same demographics file the ETL already reads — so a
correct SE costs nothing and there is no excuse for a wrong one. Verified to reduce
**exactly** to `s/√n` in the degenerate SRS case, and to produce a design effect of ~2.8×
on synthetic clustered data. Single-PSU strata contribute zero variance (the certainty-PSU
convention) and are **counted**, so that treatment is visible rather than hidden inside a
suspiciously small SE.

**Censoring.** A fixed-horizon absolute risk needs a product-limit estimator, not
`deaths/total`. Follow-up is right-censored by the linkage end date, so a naive proportion
counts short-followed participants as non-events for the whole horizon and **understates**
the risk — which `pln_risk_prediction.metta` would then multiply by `HR^(z·4.2)`,
propagating the understatement into every absolute-risk number it reports.

**Competing risks.** For a *cause-specific* risk, deaths from other causes are competing
events, not censoring. Censoring means "still at risk, we stopped watching"; someone who
died of cancer is not still at risk of dying of heart disease. Treating competing deaths
as censoring assumes they would have gone on to the cause of interest at the survivors'
rate, which **overstates** cause-specific incidence — worst in the oldest band, which is
precisely the population NHANES' methylation subsample covers. So all-cause mortality uses
weighted Kaplan-Meier and cause-specific risk uses weighted Aalen-Johansen, whose emitted
record carries the naive figure alongside so the size of the bias is visible rather than
argued about. Verified against the estimator's defining identity,
`CIF_cause + CIF_competing + S = 1` (worst error 6.7e-16 over 50 weighted trials).

**The flat-curve trap.** A product-limit curve is undefined past the last observed time,
so a cell whose follow-up stops short of the horizon returns the risk at the last observed
time instead — a plausible number that is not an estimate of what it claims. On a
three-person example followed to month 30, the "ten-year" risk comes back as 0.33. Both
estimators now report whether follow-up reaches the horizon, and such a cell is suppressed
**regardless of its size**: a large cell with short follow-up is wrong, not noisy. This is
live for NHANES — with linkage ending 2019-12-31, cycles from 2011-2012 on have under ten
years of follow-up.

**Suppression, not caveats.** A cell below the minimum unweighted n (default 30), or below
a minimum event count for a risk cell (default 5), emits **nothing**. Suppressions are
reported on stderr and counted in the run summary, but they do not become atoms with a
caveat attached, because inference consumes atoms and not caveats.

## 4. The measurement-scale problem  ← changes the arithmetic

`patient_profile.metta` grounds a marker symmetrically: `z > 1` is `Elevated`, `z < −1` is
`Low`. That is a Gaussian-flavoured cutoff. Applied to a `z` computed on the **raw** scale
of a log-normal analyte it does not blur the categories — it deletes one.

Measured on a realistic simulated CRP distribution (geometric mean 0.2 mg/dL, geometric
SD 3, n = 100,000):

| scale | P(z > +1) | P(z < −1) | max z |
|---|---|---|---|
| raw | 7.56% | **0.00%** | 80.4 |
| log10 | 15.83% | 15.86% | 5.0 |
| *normal reference* | *15.87%* | *15.87%* | — |

On the raw scale the `Low` branch is **unreachable**, and raising the threshold does not
help — `P(z < −t)` stays at 0.00% for t = 1.0, 1.5 and 2.0. Worse, "Elevated" would then
mean a different population percentile for CRP than for a symmetric marker like HbA1c, so
`diagnose-patient`'s coverage counts would weight the two differently without saying so.

So a marker **declares a scale** (`Identity` or `Log10`), the reference moments are
computed on that scale, the record carries `(RefScale …)`, and a z is only ever taken
against a reference on the same scale.

**The guard goes on the input, not the output.** hyperon 0.2.10 does have a logarithm —
`(log-math 10 $x)` — but it fails silently:

```metta
!(log-math 10 0)                  ; => -inf
!(log-math 10 -5)                 ; => NaN
!(> (log-math 10 -5) 0)           ; => False   (all NaN comparisons are False)
!(z->status (log-math 10 0))      ; => Low      ← a fabricated clinical finding
```

A raw value of exactly `0.0` on a log-scaled marker becomes `−inf`, which genuinely *is*
less than `−1`, so `z->status` returns **`Low`**: "low inflammation" invented from an
impossible measurement. And a true `0.0` is not hypothetical — NHANES lab files contain
real zeros. So the raw value is tested **before** the logarithm, and yields nothing when it is
non-positive. The guard sits in `standardize-z` rather than in `derived-z`, which is the
right place and not merely the convenient one: `derived-z` does not know the marker's
scale without a second record lookup, whereas `standardize-z` already holds the scale,
mean and SD from the same record — so putting it there makes it true that **no path
reaches `log-math` unguarded**, which is the property that matters. Nothing downstream is trusted to catch `−inf` (it is
ordered, and passes comparisons) or `NaN` (it makes every comparison False, which
`z->status` reads as `Normal`).

## 5. What NHANES can and cannot calibrate  ← the honesty finding

`docs/risk_prediction.md` §6 open-Q #1 proposes fitting the risk constants "against a real
cohort (NHANES linked mortality/CHD)". Pursuing that turned up a problem with the premise.

`pln_risk_prediction.metta` models `P(CHD in 10y)` as an **incident coronary-heart-disease
event** risk, and reads `HR 1.07 per year of AgeAccelGrim` from Lu et al. 2019 — which was
fit to **incident CHD**.

NHANES + the public-use Linked Mortality File identifies **death**, and nothing else:

- NHANES 1999+ is a single cross-sectional exam. There is no re-examination, no follow-up
  interview, and no hospital-record surveillance, so there is **no incident-CHD
  ascertainment at all**. The questionnaire items (`MCQ160C/D/E`) are lifetime "ever told
  you had" **prevalence measured at baseline**; `MCQ180C/E` is recalled age at a *past*
  diagnosis. Neither can produce a post-baseline event. Using them as outcomes would make
  exposure and outcome contemporaneous and invite reverse causation — prevalent CHD raises
  GrimAge, not only the converse. They are usable **only as an exclusion**, to remove
  prevalent cases from the at-risk set.
- The linkage's finest cardiac resolution on the public file is `UCOD_LEADING = "001"`,
  **"Diseases of heart" (I00-I09, I11, I13, I20-I51)**. That is fatal-only, and broader
  than CHD: it swallows hypertensive and rheumatic heart disease, cardiomyopathy,
  arrhythmias, and **heart failure** — an outcome this repository already models
  *separately*, as `CongestiveHeartFailure` with its own Lu 2019 hazard ratio of 1.10. No
  public-use value isolates ischemic disease.

So a NHANES fatal-heart-disease baseline multiplied by Lu 2019's incident-CHD hazard ratio
is a category error in **both** factors: the baseline is far too small for incidence, and
the hazard ratio was estimated on a different event process. The product is not an
estimate of anything.

**What this layer does instead:**

- `baseline-risk-chd` **stays exactly as it is**, curated and labelled as such.
- The mortality ETL emits baseline risks only for the outcomes NHANES can identify, under
  their own honest names: `AllCauseMortality`, and the new symbol
  **`HeartDiseaseMortality`** whose declaration records precisely what it means.
- The baseline lookup is **outcome-parameterized**, and the same `$outcome` threads through
  both the baseline and the hazard lookup, so a mismatch **yields nothing** rather than
  multiplying incompatible factors.

That last point is implemented and checked, and it is worth being precise about *how* it
holds, because MeTTa makes the obvious version of it wrong. A rule keyed on the outcome
in its head — `(= (baseline-prior CoronaryHeartDisease $age $sex) …)` — does **not** yield
nothing when asked for a different outcome: hyperon returns the *unreduced expression*
`(baseline-prior AllCauseMortality 61 Male)`, which then propagates into the arithmetic
downstream as a non-numeric atom. Declining has to go through a `match` (which genuinely
yields nothing when no record matches) or an explicit `(superpose ())`. Both are verified
in `tests/test_nhanes_common.py`.

### What is actually usable

Pinning the outcomes down exposes the real state of play, which is narrower than either
open question assumed:

| outcome | data-backed baseline | hazard ratio in the KB |
|---|---|---|
| `AllCauseMortality` | ✅ from NHANES + linkage | ✅ Lu 2019, HR 1.10/yr |
| `HeartDiseaseMortality` | ✅ from NHANES + linkage | ❌ none — no evidence record has this outcome |
| `CoronaryHeartDisease` | ❌ not identifiable in NHANES (§5) | ✅ Lu 2019, HR 1.07/yr |

So **`AllCauseMortality` is the only outcome with both halves**, and it is therefore the
only one on which a fully data-backed absolute risk can currently be computed. That is the
honest headline of step 2. `HeartDiseaseMortality` gets a baseline and waits for a hazard
ratio fit to the same outcome; `CoronaryHeartDisease` keeps its curated baseline and its
incident-CHD hazard ratio, correctly paired with each other. The outcome guard is what
keeps the three rows from being mixed.

Measured, with a fatal-heart-disease baseline record loaded alongside the curated prior:

| query | result |
|---|---|
| `(patient-baseline … CoronaryHeartDisease)` | `0.08` — the curated prior, never the fatal `0.031` |
| `(patient-baseline … HeartDiseaseMortality)` | `0.031` — the data-backed record |
| baselines returned per outcome | exactly one, never two candidates |
| `(predict-risk … HeartDiseaseMortality)` | **nothing** — it has a baseline but no hazard ratio, so the guard declines rather than borrowing CHD's |
| `(predict-risk … CoronaryHeartDisease)` | unchanged at the documented `0.12605…` |

The fourth row is the guard doing its job: an outcome with half the inputs produces no
estimate, instead of an estimate assembled from mismatched halves.

This is a better outcome than the open question asked for. It also serves
`docs/risk_prediction.md` §6 open-Q #5 (multi-outcome risk) directly: the repo now has two
genuinely data-backable outcomes, with the honest observation that the one it currently
models is not among them. Calibrating incident CHD needs a cohort with event
ascertainment — Framingham, ARIC, MESA — or a published risk equation such as the Pooled
Cohort Equations.

## 6. Reading NHANES files correctly

**The pandas IBM-zero bug.** `pandas.read_sas(format="xport")` has no special case for the
all-zero IBM-370 byte pattern SAS writes for a true `0.0`; it runs the general conversion
and returns exactly `2**-260` (verified **bit-identical** against pandas 3.0.6, while every
other value tested round-trips exactly and SAS missing values correctly become NaN).

This corrupts **real NHANES files**, not just fixtures. Left alone, such a value passes
every range check and enters a weighted mean looking harmless — but it does not compare
equal to `0`, so a "drop zero weights" filter silently keeps it. `fix_ibm_zero` runs on
every numeric column of every XPT read; the comparison is exact and cannot collide with a
real analyte (~1e-3 … 1e6).

**The mortality-file layout is quoted, not recalled.** It comes from CDC's own read-in
program `SAS_ReadInProgramAllSurveys.sas` (public-use follow-up through 2019-12-31):

```
SEQN 1-6 · ELIGSTAT 15 · MORTSTAT 16 · UCOD_LEADING 17-19 (CHARACTER) ·
DIABETES 20 · HYPERTEN 21 · PERMTH_INT 43-45 · PERMTH_EXM 46-48 · LRECL 61
```

Four traps are encoded rather than commented:

- **Vintage.** The 2011-vintage file inserts `CAUSEAVL` at column 17 and shifts every
  later field by one. Reading a 2019 file with that layout does **not** fail — it reads two
  cause digits plus the diabetes flag as the cause, and three wrong bytes as follow-up
  time, producing plausible small integers. The vintage is therefore an explicit argument,
  unknown vintages are refused, and a test shows the mismatch being caught.
- **`UCOD_LEADING` is a character field** of zero-padded codes. Coercing it to a number
  turns `"001"` into `1` and breaks every comparison written against the codebook.
- **`MORTSTAT` is present exactly when `ELIGSTAT == 1`.** Filling those blanks with `0`
  would turn every linkage-ineligible participant into a censored survivor, deflating the
  event rate and inflating the denominator. Ineligible records are excluded as a survey
  *domain*, never read as survivors.
- **`DIABETES` / `HYPERTEN` are death-certificate multiple-cause flags**, non-missing
  essentially only among decedents. They are **not** baseline comorbidity; using them as
  covariates would induce outcome-dependent confounding, since every `1` is by
  construction someone who died.

**A SAS XPORT writer, for tests.** pandas reads XPORT but cannot write it, and the PyPI
`xport` package no longer builds on current Python. Without a writer the ETLs' `.XPT` path
— the one every real user takes — could only be tested against CSV extracts.
`tests/nhanes_xport_writer.py` implements the format directly (SAS TS-140 layout, ~100
lines, including the IBM-370 float encoding), so fixtures exercise the real path.

**Pooling cycles.** Refused in code, not warned about in a comment: each `(analyte, cycle)`
carries an assay-lot identifier and aggregating across a disagreement **raises**. Known
breaks include CRP in mg/dL by latex nephelometry (1999-2010) versus hsCRP in mg/L
(2015+) — and note there is **no CRP or hsCRP at all in 2011-2014, the accelerometry
cycles; serum creatinine 1999-2000 versus 2001-2002; the HbA1c instrument changes; and
insulin from 2011-2012 onward. Note also that because the mean is a ratio, the multi-cycle
weight divisor does **not** change it — it affects only `Σw`, unequal-length pooling and
the SE. A test asserting "pooling moved the mean" would fail, and someone might then
"fix" working code.

**Subsample weights, per analyte, from an explicit table.** Using the wrong one is biased,
not merely imprecise: a fasting analyte needs the fasting weight (read from the fasting lab
file, not from demographics), and the methylation subsample ships its own `WTDN4YR` —
which is already a four-year weight, so it must **not** be halved when pooling the two
cycles it spans. Never mix `PERMTH_EXM` with interview weights or `PERMTH_INT` with MEC
weights; their time-zeros and eligible bases differ.

## 7. Honest limits of the z this produces

Two caveats that the `MeasuredZ` docstring's phrase "age- and sex-adjusted" currently
overstates, and which are recorded here rather than glossed:

1. **The age adjustment is piecewise-constant, not smooth.** A within-band weighted
   mean/SD gives "SDs above the mean of my 10-year age band", not SDs above a smooth
   function of age. The residual within-band age gradient can reach roughly ±0.25 SD at a
   band edge for a fast-rising analyte — a quarter of `elevated-z-threshold`, which is
   1.0. Narrower bands reduce it at the cost of cell size.
2. **A clock's age-acceleration z and a band-derived blood z are not the same
   construction.** The first is a regression residual; the second is a band-relative
   standardization. `patient_profile.metta` pools both into one `MeasuredZ` scale. The
   emitted records name their adjustment method so the two conventions are at least
   visible, but they remain only approximately commensurable.

Relatedly, for the methylation clocks: the NHANES release ships **predicted ages, not
accelerations**. `grim_age_core.metta` defines `(ResidualOf AgeAccelGrim GrimAge)`, and Lu
2019's HR was estimated on the **residual**, so emitting `GrimAgeMort − age` where the
residual is meant would silently mis-scale `pln_risk_prediction.metta`'s exponent. Both
conventions are computed and each is labelled with an explicit `(RefAccelDefinition …)`.
The release also ships `DunedinPoAm`, the 2020 pace-of-aging predecessor — **not**
`DunedinPACE` — so `docs/patient_grounding.md` §6 open-Q #6's proposed `PACE > 1.0`
threshold must not be transferred to it.

There is one genuinely high-value by-product here. `pln_risk_prediction.metta` hardcodes
`(= (grimaccel-sd-to-years) 4.2)` as a curated prior. The methylation subsample yields the
**measured** weighted SD of GrimAge acceleration in years, emitted as a `ClockAccelSpread`
record, which either validates that 4.2 or replaces it with a sourced number. The layer
deliberately does **not** override the knob silently: the point is that a sourced number
becomes visible next to the curated one.

## 8. The real KB limit is distinct head symbols, not atoms

`scripts/run_etl.sh` notes that "hyperon 0.2.10 panics when querying a space past a few
thousand atoms". That framing turns out to be wrong, and the correction matters because
the failure mode is an **abort, not an exception** — a non-unwinding Rust panic in
hyperon's space trie during `match`, which no Python guard can catch.

What triggers it is the number of **distinct head symbols** in the space. Measured
against the KB the chat app actually executes (every repo-root `.metta` under
`PLN_MAX_KB_FILE_BYTES` — 24 files, ~137 distinct head symbols):

| added to that KB | result |
|---|---|
| 400 atoms under **one** new head symbol | fine |
| 400 atoms under **three** new head symbols | fine |
| 8 atoms under **8** distinct new head symbols | fine |
| 12 atoms under **12** distinct new head symbols | **abort** |

So the margin is roughly 8–12 new head symbols, and each generated NHANES file
introduces 12–21 — every `Ref*`/`Base*`/`Spread*` field predicate becomes a head symbol
the moment a record atom uses it. (The type declarations alone do not: their head is
`:`.) Loading any one generated file into the app's KB therefore aborts it on the first
inference query, while `!(+ 1 2)` still answers.

Three consequences, all acted on:

- **Generated files stay out of the repo root.** The ETLs default to `build/`, which the
  app does not scan, and `run_etl.sh` no longer advertises `OUT_DIR=.` — that was the
  documented workflow, and it was the crashing one.
- **`MettaWriter`'s atom budget is not the guarantee it looks like.** It was measured
  against the 14-file stack the tests build (fine at +3,456 atoms, abort at +4,608) and is
  worth keeping as a bound on runaway emission, but it does not certify that a file is
  safe to load into the app. `nhanes_common.py` now says so.
- **A per-file byte limit cannot bound a whole-space failure.** `pln_chat/app.py` decides
  what to execute by filtering individual files on `PLN_MAX_KB_FILE_BYTES`, which cannot
  express a constraint on the union. This is **pre-existing and not fixed here**: the app
  works today, but on roughly 8–12 head symbols of margin, so the next `.metta` file added
  to the repo root may break it with no NHANES involvement. Changing what the app executes
  (to `_INFERENCE_STACK` only, say) is a real behaviour change affecting other query
  paths, so it is flagged for the maintainer rather than made silently here.

## 9. Open questions / next increments## 9. Open questions / next increments

1. **Run it.** Nothing here has touched real NHANES data. The first run on real files
   should be treated as part of the work: verify every registry entry against the actual
   columns (`--inspect`), check that `Σw` per cell is plausible against the US population
   for that age range, and record which registry entries were wrong.
2. **Narrower age bands, or a smooth adjustment.** §7's ±0.25 SD band-edge artifact is the
   argument for regressing the analyte on age within sex and standardizing the residual —
   which would also make the blood z and the clock z the same construction. It costs cell
   size and a documented model.
3. **Consume the design-based SE.** Reference records now carry `(RefDesignSE …)` and
   `(RefDesignDF …)` whenever the demographics file supplies `SDMVSTRA`/`SDMVPSU`, and
   honestly omit them when it does not. Nothing reads them yet: the natural use is for the
   calibration layer to widen `stv` confidence for an imprecise cell, rather than the
   current hard suppression on an unweighted count.
4. **NCHS presentation standards properly.** v1 suppresses on a minimum unweighted n. The
   published standard for proportions (Series 2 No. 175) also involves effective sample
   size, Korn-Graubard interval width and degrees of freedom; the SE estimator above is the
   prerequisite for implementing it.
5. **`grimaccel-sd-to-years`.** Once §7's measured spread exists, decide whether it
   replaces the 4.2 prior, and whether the residual or the difference convention is the one
   the Lu 2019 HR should be paired with.
6. **An incidence source.** §5 establishes that NHANES cannot calibrate incident CHD. If
   the CHD baseline is to become data-backed, it needs Framingham/ARIC/MESA or the Pooled
   Cohort Equations, each with its own licensing and provenance story.
7. **Wearables.** Out of scope here, and worth stating why it is a separate increment: the
   methylation cycles (1999-2002) and the accelerometry cycles (2003-2006 waist,
   2011-2014 wrist) **do not overlap**, so no NHANES participant has both a clock and
   wearable data. A behavioural-marker layer would need its clock edges from outside
   NHANES, with NHANES supplying only the mortality leg.

## 10. Non-goals

- **Not a downloader.** The ETLs read local files. `data/nhanes/` gives the URL pattern
  and a manifest; nothing is fetched at build time and nothing is bundled.
- **No shipped NHANES values** (§2).
- **No CHD-incidence calibration** (§5).
- **No new inference machinery.** Like the patient-grounding layer, this grounds inputs and
  parameterizes a baseline; it adds no causal edge and no new reasoning rule.
- **No change to existing behavior.** With no generated file loaded, the stack behaves
  exactly as before — which the existing suite, unchanged, is what proves.
