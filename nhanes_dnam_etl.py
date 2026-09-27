#!/usr/bin/env python3
"""NHANES DNA-methylation release (DNMEPI) -> epigenetic-clock reference distributions.

WHAT THIS PRODUCES, AND WHY IT IS THE HIGHEST-VALUE PIECE OF THE NHANES WORK
---------------------------------------------------------------------------
`pln_risk_prediction.metta` turns a patient's GrimAge acceleration into an absolute
10-year risk. It reads the acceleration as a z-score (in SDs), multiplies by a
hand-set constant to get years, and raises Lu 2019's HR 1.07 to that power:

    (= (grimaccel-sd-to-years) 4.2)          ; pln_risk_prediction.metta:90 — CURATED

That 4.2 was reverse-engineered from the case study ("1.6 SD -> +6.7 years"), not
measured. NHANES pooled 1999-2000 + 2001-2002 shipped a DNA-methylation subsample
(DNMEPI, adults 50+) carrying eleven epigenetic clocks plus a pace-of-aging measure.
The survey-weighted SD of GrimAge acceleration IN YEARS computed from that subsample
is exactly the quantity 4.2 is a guess at. This ETL measures it and emits it as a
Schema C `ClockAccelSpread` record, so the constant can be validated or replaced
against data rather than against a worked example. That is the headline output.

Alongside it, the ETL emits Schema A `ReferenceDistribution` records (mean/SD of the
acceleration by sex x age band) so a patient's raw clock output in years can be
standardized into the `MeasuredZ` the inference stack already consumes.

FIVE THINGS THAT ARE EASY TO GET QUIETLY WRONG HERE
---------------------------------------------------
1. THE CLOCKS ARE PREDICTED AGES, NOT ACCELERATIONS. `HorvathAge`, `GrimAgeMort` and
   the rest are predicted ages in years. `grim_age_core.metta:49` declares
   `(ResidualOf AgeAccelGrim GrimAge)` and `:62` declares
   `(AdjustmentCovariate AgeAccelGrim ChronologicalAgeAtBloodDraw)` — the repo's
   AgeAccelGrim is the RESIDUAL of the predicted age on chronological age, and Lu
   2019's HR 1.07/year was estimated on that residual. The naive
   `GrimAgeMort - RIDAGEYR` difference is a DIFFERENT quantity with a different
   spread, and feeding its SD into `grimaccel-sd-to-years` silently mis-scales the
   exponent of `pow-math`. So both conventions are computed, both are emitted, and
   EVERY clock record carries `(RefAccelDefinition ... Residual|Difference)` and
   spells the definition into its record id. There is no code path that emits an
   unlabelled acceleration: `emit_reference_records` asserts the label is present on
   every record it writes.

   The residual is a survey-WEIGHTED least-squares fit of the predicted age on
   `RIDAGEYR` — two parameters, solved with numpy in `weighted_linear_residuals`, no
   new dependency. It is fit once on the pooled analytic sample (age only, both
   sexes), matching `(AdjustmentCovariate AgeAccelGrim ChronologicalAgeAtBloodDraw)`:
   sex is NOT in the adjustment model, because putting it there would define a
   different estimand from the one the HR was fit to.

2. DNMEPI HAS ITS OWN WEIGHT, AND THE OBVIOUS ONE IS WRONG. The release ships
   `WTDN4YR`, constructed for the methylation subsample from the `WTMEC4YR` base.
   `WTMEC4YR` describes the full MEC-examined sample; applying it to a subsample
   selected on different criteria produces a biased population estimate that looks
   entirely plausible. Passing a base or interview weight therefore ABORTS
   (`BiasedWeight`) rather than warning — see `REFUSED_WEIGHTS`.

3. THE COLUMN IS `DunedinPoAm`, NOT `DunedinPACE`. NHANES ships the 2020
   pace-of-aging predecessor (PoAm), not the 2022 DunedinPACE. They are different
   estimators with different calibrations. `docs/patient_grounding.md` open-Q #6
   proposes a marker-specific `DunedinPACE > 1.0` cutoff; THAT THRESHOLD MUST NOT BE
   TRANSFERRED TO PoAm. The symbol emitted here (were it ever declared) is
   `DunedinPoAm`, and the registry entry says so in its notes.

   Relatedly, `DunedinPoAm` and `HorvathTelo` are not predicted ages at all — one is
   a rate (years of physiological change per chronological year), one is a predicted
   telomere length in kb. Subtracting a chronological age from either is meaningless,
   so the `Difference` convention is REFUSED for them (`AccelDefinitionUnavailable`),
   and neither can produce a Schema C record, whose `SpreadSDYears` field is defined
   to be in years.

4. ONLY DECLARED SYMBOLS MAY BE EMITTED (D22). An atom naming a symbol the KB never
   declares is a dangling reference. Scanning this repo, exactly TWO of the twelve
   DNMEPI columns have a declared acceleration symbol: `AgeAccelGrim`
   (grim_age_core.metta, `Inheritance AgeAccelGrim EpigeneticAgeAcceleration` ->
   `Inheritance EpigeneticAgeAcceleration Biomarker`) and `HorvathAgeAccel`
   (patient_profile.metta). `PhenoAge` and `DunedinPoAm` are declared NOWHERE — they
   appear only as prose in a `logical_predicates.metta` comment. The declaration is
   re-checked against the repo at run time by walking the `Inheritance` chain, so a
   symbol whose declaration is removed stops producing atoms at the next run instead
   of leaving a dangling reference behind. Undeclared clocks are listed as skipped.

5. EVERY NHANES NAME HERE IS AN UNVERIFIED CLAIM (D10). The CDC hosts are unreachable
   from the environment this was written in, so no file name or column name below
   could be checked against the source; all twelve clock columns are recorded at
   'medium' confidence. They live in ONE registry table, `--show-registry` prints it,
   `--inspect` prints a file's real columns, a missing column raises `MissingColumns`
   listing every column present, and `--registry` overrides an entry without editing
   code. A registry entry is NEVER consulted as a fallback guess.

   This matters more than usual for the `.XPT` form: the clock names are up to 12
   characters and SAS XPORT v5 caps variable names at 8, so a `.XPT` export must have
   truncated or renamed them. Column lookup is case-insensitive (the `.sas7bdat` form
   carries mixed case, e.g. `HorvathAge`), but a name that does not match is an ERROR
   naming the columns present plus an explicit hint to use `--registry` — never a
   prefix search for something that looks close.

NO NHANES-DERIVED NUMBER IS COMMITTED TO THIS REPO (D1). The default output is under
``build/`` (gitignored) and the repo ships no microdata.

Usage:
    python3 nhanes_dnam_etl.py --show-registry
    python3 nhanes_dnam_etl.py --manifest
    python3 nhanes_dnam_etl.py --inspect data/nhanes/DNMEPI.XPT
    python3 nhanes_dnam_etl.py \\
        --dnam data/nhanes/DNMEPI.xpt --demo data/nhanes/DEMO.XPT \\
        --output build/nhanes_dnam_clocks.metta
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from nhanes_common import (
    declared_biomarker_symbols,
    short_sex,
    short_cycles,
    short_band,
    AGE_BANDS,
    AtomBudgetExceeded,
    ByteBudgetExceeded,
    MettaWriter,
    MissingColumns,
    SCALE_IDENTITY,
    add_common_args,
    age_band,
    check_symbol,
    load_registry_override,
    mstr,
    num,
    provenance_header,
    read_nhanes,
    run_inspect,
    sex_symbol,
    suppressed_reason,
    weighted_moments,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "nhanes"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "nhanes_dnam_clocks.metta"

#: DNMEPI pools the two cycles into ONE release, so there is no cross-cycle assay
#: discontinuity to refuse (D8/D18): the specimens were assayed together in a single
#: laboratory run. The cycle tag below is therefore fixed rather than derived.
DNMEPI_CYCLES = "1999-2002"
DNMEPI_CYCLE_TAG = "1999_2002"
DNMEPI_ASSAY_LOT = "DNMEPI_1999_2002_single_release"

#: Age topcode for the 1999-2000 and 2001-2002 cycles (D19). Ages at or above this
#: are recorded AS this value, so the top band must not be finer than it (Age_70p is
#: not) and no mean age may be computed inside it (none is). An observed age ABOVE
#: the topcode means the file is not the cycle the registry claims, and aborts.
AGE_TOPCODE = 85.0

#: DNMEPI is adults 50+. The default cut is not cosmetic: the residual fit below is
#: a straight line in age, and extending it down into cycles' full age range would
#: change the slope and therefore every residual.
DEFAULT_MIN_AGE = 50.0

#: Demographic variables, from DEMO (or from the DNAm file when it carries them).
AGE_VAR = "RIDAGEYR"
SEX_VAR = "RIAGENDR"
ID_VAR = "SEQN"

URL_PATTERN = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/DNAm/{basename}.{ext}"
)


# ════════════════════════════════════════════════════════════════════════════
# 1. Survey weight — the wrong one is biased, so the wrong one aborts
# ════════════════════════════════════════════════════════════════════════════
#
# D25: "WTDN4YR for every DNMEPI clock. Abort if the named weight column is absent."
# D20 goes further: "using WTMEC4YR on this subsample is biased and must abort".
#
# The bias is not hypothetical arithmetic pedantry. WTDN4YR is constructed FROM
# WTMEC4YR by re-weighting for selection into the methylation subsample and for that
# subsample's own non-response. Using WTMEC4YR instead silently assumes the subsample
# was a simple random sample of the MEC sample. It produces a number with the right
# units, the right order of magnitude and the wrong value, which is the worst kind.

SUBSAMPLE_WEIGHT = "WTDN4YR"
BASE_WEIGHT = "WTMEC4YR"

#: Weights that are WRONG for this subsample, each with the reason printed on abort.
REFUSED_WEIGHTS: dict[str, str] = {
    "WTMEC4YR": "the 4-year MEC examination weight — DNMEPI's BASE weight, not its "
                "subsample weight. WTDN4YR is derived from it by adjusting for "
                "selection into the methylation subsample and for that subsample's "
                "non-response; substituting the base weight assumes the subsample was "
                "a simple random sample of the MEC sample, which it was not.",
    "WTMEC2YR": "a single-cycle MEC weight — DNMEPI pools two cycles, so a 2-year "
                "weight estimates the wrong population and double-counts one cycle.",
    "WTINT4YR": "an INTERVIEW weight. The methylation specimens come from the MEC "
                "examination; an interview weight covers a different (larger) sample.",
    "WTINT2YR": "a single-cycle INTERVIEW weight (see WTINT4YR, plus the 2-year "
                "pooling problem).",
    "WTSAF4YR": "the fasting-subsample weight, built for a DIFFERENT subsample "
                "(glucose/insulin/triglycerides). It does not describe DNMEPI.",
    "WTSCY4YR": "the surplus-sera cystatin C weight, built for a DIFFERENT and "
                "non-representative subsample. It does not describe DNMEPI.",
}


class BiasedWeight(RuntimeError):
    """A weight variable that does not describe the DNMEPI subsample was requested."""


class AccelDefinitionUnavailable(RuntimeError):
    """The requested acceleration convention is undefined for this clock's quantity."""


class MissingInput(RuntimeError):
    """An input file the run needs is not where the ETL looked."""


class DegenerateFit(RuntimeError):
    """The weighted age regression cannot be solved (no age variation in the sample)."""


def check_weight_variable(name: str) -> str:
    """Refuse a weight that does not describe the DNMEPI methylation subsample."""
    upper = str(name).upper().strip()
    reason = REFUSED_WEIGHTS.get(upper)
    if reason is not None:
        raise BiasedWeight(
            f"refusing to weight the DNMEPI methylation subsample by {upper}: {reason}\n"
            f"  Use {SUBSAMPLE_WEIGHT} (the default). If CDC renamed the subsample "
            f"weight for this release, pass --weight-variable with the NEW SUBSAMPLE "
            f"weight name — this guard only refuses the known base/interview/other-"
            f"subsample weights, it does not second-guess an unfamiliar name.\n"
            f"  Refused: {', '.join(sorted(REFUSED_WEIGHTS))}"
        )
    return upper


# ════════════════════════════════════════════════════════════════════════════
# 2. Acceleration conventions — the decision this ETL exists to make explicit
# ════════════════════════════════════════════════════════════════════════════

ACCEL_RESIDUAL = "Residual"
ACCEL_DIFFERENCE = "Difference"
ACCEL_DEFINITIONS = (ACCEL_RESIDUAL, ACCEL_DIFFERENCE)

#: The named transform recorded in `(RefTransform ...)` (D24: a derived value is
#: labelled as derived). The shipped column is a predicted age; what this ETL emits
#: moments for is a quantity computed FROM it, so the computation is named.
ACCEL_TRANSFORM = {
    ACCEL_RESIDUAL: "WeightedLeastSquaresResidualOn_RIDAGEYR",
    ACCEL_DIFFERENCE: "PredictedAgeMinus_RIDAGEYR",
}

ACCEL_DESCRIPTION = {
    ACCEL_RESIDUAL:
        "residual of the predicted age on chronological age (RIDAGEYR), from a "
        "survey-weighted 2-parameter least-squares fit on the pooled analytic sample. "
        "This is the repo's AgeAccelGrim (grim_age_core.metta: ResidualOf AgeAccelGrim "
        "GrimAge) and the quantity Lu 2019's HR 1.07/year was estimated on.",
    ACCEL_DIFFERENCE:
        "predicted age minus chronological age (RIDAGEYR). A DIFFERENT quantity from "
        "the residual: it retains whatever age trend the clock has, so its spread is "
        "not the spread the HR was fit to. Emitted for comparison and audit, NOT as a "
        "substitute for the residual.",
}


# ════════════════════════════════════════════════════════════════════════════
# 3. Clock registry — ONE table, every entry at 'medium' confidence
# ════════════════════════════════════════════════════════════════════════════
#
# Twelve columns were reported by the research pass. None could be verified against
# CDC, so every entry is 'medium' and every entry is overridable (D10). The per-entry
# `quantity` is what makes the Difference convention refusable for the two entries
# that are not predicted ages.

QUANTITY_PREDICTED_AGE = "PredictedAge"        # years; both conventions defined
QUANTITY_PACE = "PaceOfAging"                  # rate; Difference is meaningless
QUANTITY_TELOMERE = "PredictedTelomereLength"  # kb;   Difference is meaningless

QUANTITIES = (QUANTITY_PREDICTED_AGE, QUANTITY_PACE, QUANTITY_TELOMERE)

CONFIDENCE_LEVELS = ("high", "medium", "low")


@dataclass
class ClockSpec:
    """One DNMEPI clock column: its NHANES name, its repo symbol, and its quantity."""

    column: str                # NHANES column name as the research pass reported it
    symbol: str                # the repo symbol the ACCELERATION would be emitted under
    quantity: str              # QUANTITY_* — decides which conventions are defined
    unit: str                  # unit of the SHIPPED column
    confidence: str            # high | medium | low
    evidence: str
    notes: str = ""
    declared_in: str = ""      # a CLAIM, re-verified by declared_biomarker_symbols()
    emit: bool = True          # False => registry/manifest only, never any atom

    def conventions(self) -> tuple[str, ...]:
        """The acceleration conventions that are DEFINED for this clock's quantity."""
        if self.quantity == QUANTITY_PREDICTED_AGE:
            return ACCEL_DEFINITIONS
        return (ACCEL_RESIDUAL,)

    def accel_unit(self, definition: str) -> str:
        """Unit of the emitted acceleration under one convention."""
        if self.quantity == QUANTITY_PREDICTED_AGE:
            return ("years (age-adjusted residual)" if definition == ACCEL_RESIDUAL
                    else "years (predicted minus chronological)")
        if self.quantity == QUANTITY_PACE:
            return "years per year (age-adjusted residual)"
        return "kilobases (age-adjusted residual)"

    def spreads_in_years(self) -> bool:
        """Whether a Schema C record (SpreadSDYears) is meaningful for this clock."""
        return self.quantity == QUANTITY_PREDICTED_AGE


_COMMON_EVIDENCE = (
    "Column name reported by the research pass for the NHANES DNMEPI release "
    "(1999-2000 + 2001-2002 pooled, adults 50+); NOT verified against CDC, whose "
    "hosts are unreachable from this environment. Override with --registry."
)

REGISTRY: list[ClockSpec] = [
    ClockSpec(
        column="GrimAgeMort",
        symbol="AgeAccelGrim",
        quantity=QUANTITY_PREDICTED_AGE,
        unit="years",
        confidence="medium",
        evidence=_COMMON_EVIDENCE,
        declared_in="grim_age_core.metta (Inheritance AgeAccelGrim "
                    "EpigeneticAgeAcceleration -> Biomarker)",
        notes="THE headline clock. grim_age_core.metta declares (ResidualOf "
              "AgeAccelGrim GrimAge), so the Residual convention is the one that "
              "matches the symbol. Its weighted SD in years is what "
              "pln_risk_prediction.metta's (grimaccel-sd-to-years) 4.2 estimates.",
    ),
    ClockSpec(
        column="HorvathAge",
        symbol="HorvathAgeAccel",
        quantity=QUANTITY_PREDICTED_AGE,
        unit="years",
        confidence="medium",
        evidence=_COMMON_EVIDENCE,
        declared_in="patient_profile.metta (Inheritance HorvathAgeAccel "
                    "EpigeneticAgeAcceleration -> Biomarker)",
        notes="First-generation multi-tissue clock (Horvath 2013). Its acceleration "
              "is the counterpart patient_profile.metta uses to represent the "
              "first-gen / mortality-clock discordance.",
    ),
    ClockSpec(
        column="GrimAge2Mort",
        symbol="AgeAccelGrim2",
        quantity=QUANTITY_PREDICTED_AGE,
        unit="years",
        confidence="medium",
        evidence=_COMMON_EVIDENCE,
        notes="GrimAge version 2. A DISTINCT estimator from GrimAge: it must not be "
              "emitted under AgeAccelGrim, because Lu 2019's HR was fit to v1.",
    ),
    ClockSpec(
        column="PhenoAge",
        symbol="AgeAccelPheno",
        quantity=QUANTITY_PREDICTED_AGE,
        unit="years",
        confidence="medium",
        evidence=_COMMON_EVIDENCE,
        notes="Levine PhenoAge. Appears in this repo only as prose in a "
              "logical_predicates.metta comment, never as a declaration.",
    ),
    ClockSpec(
        column="HannumAge", symbol="AgeAccelHannum", quantity=QUANTITY_PREDICTED_AGE,
        unit="years", confidence="medium", evidence=_COMMON_EVIDENCE,
        notes="First-generation blood clock (Hannum 2013).",
    ),
    ClockSpec(
        column="SkinBloodAge", symbol="AgeAccelSkinBlood", quantity=QUANTITY_PREDICTED_AGE,
        unit="years", confidence="medium", evidence=_COMMON_EVIDENCE,
        notes="Horvath skin-and-blood clock (2018).",
    ),
    ClockSpec(
        column="ZhangAge", symbol="AgeAccelZhang", quantity=QUANTITY_PREDICTED_AGE,
        unit="years", confidence="medium", evidence=_COMMON_EVIDENCE,
        notes="Zhang elastic-net age predictor.",
    ),
    ClockSpec(
        column="LinAge", symbol="AgeAccelLin", quantity=QUANTITY_PREDICTED_AGE,
        unit="years", confidence="medium", evidence=_COMMON_EVIDENCE,
        notes="Lin 99-CpG age predictor.",
    ),
    ClockSpec(
        column="WeidnerAge", symbol="AgeAccelWeidner", quantity=QUANTITY_PREDICTED_AGE,
        unit="years", confidence="medium", evidence=_COMMON_EVIDENCE,
        notes="Weidner 3-CpG age predictor.",
    ),
    ClockSpec(
        column="VidalBraloAge", symbol="AgeAccelVidalBralo", quantity=QUANTITY_PREDICTED_AGE,
        unit="years", confidence="medium", evidence=_COMMON_EVIDENCE,
        notes="Vidal-Bralo 8-CpG age predictor.",
    ),
    ClockSpec(
        column="DunedinPoAm",
        symbol="DunedinPoAm",
        quantity=QUANTITY_PACE,
        unit="years per year",
        confidence="medium",
        evidence=_COMMON_EVIDENCE,
        notes="PACE-OF-AGING, NOT A PREDICTED AGE, and NOT DunedinPACE. NHANES ships "
              "the 2020 PoAm predecessor; DunedinPACE (2022) is a different estimator "
              "with a different calibration. docs/patient_grounding.md open-Q #6's "
              "proposed 'DunedinPACE > 1.0' cutoff MUST NOT be transferred to PoAm. "
              "Subtracting RIDAGEYR from a rate is meaningless, so the Difference "
              "convention is refused for this column.",
    ),
    ClockSpec(
        column="HorvathTelo",
        symbol="DNAmTelomereLength",
        quantity=QUANTITY_TELOMERE,
        unit="kilobases",
        confidence="medium",
        evidence=_COMMON_EVIDENCE,
        notes="DNAm-estimated telomere length (Lu 2019 DNAmTL), in kb — not an age. "
              "The Difference convention is refused for this column.",
    ),
]


# ---------------------------------------------------------------------------
# Declaration check (D22) — walked against the repo at run time, not trusted
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Registry overrides (--registry): the escape hatch for a wrong column name
# ---------------------------------------------------------------------------
_OVERRIDABLE = ("column", "symbol", "quantity", "unit", "confidence", "evidence",
                "notes", "declared_in")


def apply_overrides(registry: Sequence[ClockSpec], override: dict) -> list[ClockSpec]:
    """Apply a ``--registry`` JSON override, keyed by the registry entry's SYMBOL.

    Keyed by symbol rather than by column precisely because the column name is the
    thing most likely to be wrong — a `.XPT` export cannot carry a 12-character name,
    so correcting it is the expected use:

        {"AgeAccelGrim": {"column": "GRIMAGEM"}}

    An unknown symbol or field is an error, not a silent no-op: an override that
    appears to work but matches nothing is exactly the failure this guards against.
    """
    by_symbol = {c.symbol: c for c in registry}
    for symbol, patch in override.items():
        spec = by_symbol.get(symbol)
        if spec is None:
            raise ValueError(
                f"--registry override names unknown clock symbol {symbol!r}; "
                f"known: {sorted(by_symbol)}"
            )
        if not isinstance(patch, dict):
            raise ValueError(f"--registry entry for {symbol!r} must be a JSON object")
        for key, value in patch.items():
            if key == "emit":
                spec.emit = bool(value)
            elif key in _OVERRIDABLE:
                setattr(spec, key, str(value))
            else:
                raise ValueError(
                    f"--registry override for {symbol!r} has unknown field {key!r}; "
                    f"overridable: {sorted(_OVERRIDABLE + ('emit',))}"
                )
        if spec.quantity not in QUANTITIES:
            raise ValueError(f"{symbol}: unknown quantity {spec.quantity!r}; "
                             f"known: {QUANTITIES}")
        if spec.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"{symbol}: confidence must be one of {CONFIDENCE_LEVELS}")
        check_symbol(spec.symbol, what="clock symbol")
    # A rename must not create two entries fighting over one symbol.
    symbols = [c.symbol for c in registry]
    duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
    if duplicates:
        raise ValueError(
            f"--registry override produced duplicate clock symbol(s) {duplicates}; "
            f"each registry entry must emit under its own symbol"
        )
    return list(registry)


# ════════════════════════════════════════════════════════════════════════════
# 4. The survey-weighted residual — two parameters, numpy only
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class WLSFit:
    """A weighted 2-parameter least-squares fit of a clock on chronological age."""

    intercept: float
    slope: float
    n: int
    sum_w: float
    mean_age: float
    mean_clock: float


def weighted_linear_residuals(
    clock: Iterable[float], age: Iterable[float], weights: Iterable[float]
) -> tuple[WLSFit, np.ndarray]:
    """Survey-weighted least-squares residuals of ``clock`` on ``age``.

    Minimizes ``sum(w * (y - a - b*x)^2)``. Two parameters, so the normal equations
    are solved in closed form and no least-squares package is needed.

    Solved in CENTERED form on purpose, for the same reason ``weighted_moments`` is
    two-pass: the uncentered normal equations subtract ``(sum(w x))^2 / sum(w)`` from
    ``sum(w x^2)``, and at NHANES weight magnitudes (individual weights in the tens of
    thousands, ages near 70) those two terms agree to many significant figures before
    cancelling. The centered form sums squared deviations directly and does not
    cancel.

        xbar = sum(w x)/sum(w),  ybar = sum(w y)/sum(w)
        b    = sum(w (x-xbar)(y-ybar)) / sum(w (x-xbar)^2)
        a    = ybar - b*xbar
        e    = y - (a + b x)  ==  (y-ybar) - b*(x-xbar)

    Inputs must already be filtered to finite values and positive weights; the caller
    owns that so the same mask can be applied to every parallel array. Raises
    ``DegenerateFit`` when the ages carry no weighted variance, because a residual on
    a constant covariate is just a centered value and would quietly be a different
    estimand.
    """
    y = np.asarray(list(clock), dtype="float64")
    x = np.asarray(list(age), dtype="float64")
    w = np.asarray(list(weights), dtype="float64")
    if not (y.shape == x.shape == w.shape):
        raise ValueError(f"shape mismatch: clock {y.shape}, age {x.shape}, w {w.shape}")
    n = int(y.size)
    if n < 3:
        raise DegenerateFit(
            f"a 2-parameter weighted fit needs at least 3 observations, got {n}"
        )

    sum_w = float(w.sum())
    if not (sum_w > 0.0):
        raise DegenerateFit("total weight is not positive; cannot fit")

    xbar = float((w * x).sum() / sum_w)
    ybar = float((w * y).sum() / sum_w)
    dx = x - xbar
    dy = y - ybar
    sxx = float((w * dx * dx).sum())
    if not (sxx > 0.0):
        raise DegenerateFit(
            "chronological age has zero weighted variance in this sample, so the "
            "predicted age cannot be regressed on it. A 'residual' here would be a "
            "mean-centred predicted age, which is a different quantity — refusing."
        )
    slope = float((w * dx * dy).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    residuals = dy - slope * dx
    return (
        WLSFit(intercept=intercept, slope=slope, n=n, sum_w=sum_w,
               mean_age=xbar, mean_clock=ybar),
        residuals,
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. Input resolution and column lookup
# ════════════════════════════════════════════════════════════════════════════


#: Suffixes this ETL can read. ``read_nhanes`` covers ``.xpt``/``.csv`` (and applies
#: the mandatory IBM-zero repair to XPORT); ``.sas7bdat`` is handled locally by
#: ``read_dnam_file`` because ``read_nhanes`` raises on it.
READABLE_SUFFIXES = (".XPT", ".xpt", ".sas7bdat", ".csv")


def read_dnam_file(path: Path | str) -> pd.DataFrame:
    """Read an NHANES file, adding the ``.sas7bdat`` form ``read_nhanes`` refuses.

    D20 asks for both forms, and the ``.sas7bdat`` one is the SAFER input here: the
    clock names are up to 12 characters and SAS XPORT v5 caps variable names at 8, so
    only the ``.sas7bdat`` form can carry ``GrimAgeMort`` verbatim. But
    ``nhanes_common.read_nhanes`` accepts only ``.xpt``/``.xport``/``.csv``/``.txt``/
    ``.tsv`` and raises ``ValueError`` on anything else, so the ``.sas7bdat`` branch
    lives here. Everything else is delegated, so the XPORT path keeps the
    IBM-zero repair and the shared ``MissingColumns`` diagnostics unchanged.

    ``fix_ibm_zero`` is deliberately NOT applied to a ``.sas7bdat`` read. That repair
    exists for a bug in pandas' XPORT *IBM-370 hex float* decoder; sas7bdat stores
    IEEE doubles and has no such artifact, so snapping 2**-260 there would be an
    unjustified mutation of a value the file really contains.
    """
    path = Path(path)
    if path.suffix.lower() != ".sas7bdat":
        return read_nhanes(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. NHANES microdata is not bundled with this repo; "
            f"see docs/nhanes_integration.md for how to obtain it."
        )
    frame = pd.read_sas(path, format="sas7bdat")
    frame.columns = [str(c).upper().strip() for c in frame.columns]
    for col in frame.columns:                    # pandas returns bytes for char cols
        if frame[col].dtype == object:
            frame[col] = frame[col].map(
                lambda v: v.decode("latin-1").strip() if isinstance(v, bytes) else v
            )
    return frame


def resolve_inputs(supplied: Sequence[Path], data_dir: Path, basenames: Sequence[str],
                   *, what: str, required: bool = True) -> list[Path]:
    """Return the files the user passed, or look for ``basenames`` under ``data_dir``.

    ``required=False`` returns an empty list instead of raising when nothing is found.
    That is the demographics case: the DNAm file may carry RIDAGEYR/RIAGENDR itself,
    and whether a DEMO file is actually needed is decided later, by looking at the
    columns rather than by guessing here.
    """
    if supplied:
        for path in supplied:
            if not Path(path).exists():
                raise MissingInput(f"{what} file not found: {path}")
        return [Path(p) for p in supplied]
    # Basename matching is case-INSENSITIVE: CDC's own naming is inconsistent
    # (DEMO.XPT but dnmepi.sas7bdat in some mirrors), and scripts/run_etl.sh looks for
    # a lowercase dnmepi.sas7bdat. A case mismatch producing "file not found" for a
    # file sitting right there would be a pointless failure.
    found: list[Path] = []
    if data_dir.is_dir():
        by_key = {}
        for entry in sorted(data_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in (
                s.lower() for s in READABLE_SUFFIXES
            ):
                by_key.setdefault(
                    (entry.stem.upper(), entry.suffix.lower()), entry
                )
        for base in basenames:
            for ext in READABLE_SUFFIXES:
                candidate = by_key.get((base.upper(), ext.lower()))
                if candidate is not None:
                    found.append(candidate)
                    break
    if not found:
        if not required:
            return []
        raise MissingInput(
            f"no {what} file supplied and none of "
            f"{[b + s for b in basenames for s in ('.sas7bdat', '.XPT')]} found "
            f"under {data_dir} (basename matching is case-insensitive).\n"
            f"  NHANES microdata is not bundled with this repo (D1). Expected URL "
            f"pattern: {URL_PATTERN.format(basename=basenames[0], ext='xpt')}\n"
            f"  Pass the file explicitly with --{what}."
        )
    return found


def resolve_column(frame: pd.DataFrame, wanted: str, *, label: str,
                   source: str) -> str:
    """Case-insensitive column lookup. A miss is a loud error, never a guess.

    ``read_nhanes`` upper-cases the frame's columns, so an exact match against
    ``wanted.upper()`` IS the case-insensitive match — this is what lets the
    mixed-case ``.sas7bdat`` names (``HorvathAge``) resolve without an override.

    A miss raises ``MissingColumns`` listing every column present. When every column
    in the file is 8 characters or shorter the message also says so, because that is
    the SAS XPORT v5 limit and the clock names are longer than it — the diagnosis a
    user needs to reach for ``--registry``. That is a DIAGNOSTIC, not a fallback: this
    function never accepts a column it was not asked for (D10/D23).
    """
    key = str(wanted).upper().strip()
    if key in frame.columns:
        return key
    # Leading-underscore names are this ETL's own scratch columns (_AGE/_SEX/_W), not
    # anything the file contains; listing them would send a user hunting for a column
    # CDC never published.
    present = sorted(c for c in frame.columns if not str(c).startswith("_"))
    hint = ""
    if present and max(len(c) for c in present) <= 8:
        hint = (
            f"\n  Every column in this file is <= 8 characters, which is the SAS "
            f"XPORT v5 variable-name limit. {wanted!r} is {len(str(wanted))} "
            f"characters, so an .XPT export MUST have truncated or renamed it. The "
            f".sas7bdat form carries the full name; otherwise map it explicitly:\n"
            f"      --registry over.json   with   {{\"<symbol>\": {{\"column\": "
            f"\"<NAME_IN_FILE>\"}}}}\n"
            f"  This ETL will not guess a truncation (D23: never fall back to a "
            f"similar variable name)."
        )
    raise MissingColumns(
        f"{source}: {label} column {wanted!r} is not present.\n"
        f"  columns present: {present}{hint}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. Collection — join, filter, compute both conventions
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Cell:
    """One (sex, age band) cell's acceleration values and weights."""

    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


@dataclass
class AccelSeries:
    """One clock under one convention: the whole-sample series plus its cells."""

    spec: ClockSpec
    definition: str
    values: np.ndarray
    weights: np.ndarray
    sexes: list[str]
    bands: list[str]
    fit: Optional[WLSFit]           # None for the Difference convention

    def cells(self) -> dict[tuple[str, str], Cell]:
        out: dict[tuple[str, str], Cell] = {}
        for value, weight, sex, band in zip(self.values, self.weights,
                                            self.sexes, self.bands):
            cell = out.setdefault((sex, band), Cell())
            cell.values.append(float(value))
            cell.weights.append(float(weight))
        return out


@dataclass
class Analytic:
    """The joined, filtered analytic sample plus every diagnostic worth reporting."""

    frame: pd.DataFrame
    weight_variable: str
    age_source: str
    sex_source: str
    inputs: list[str]
    rows_read: int
    dropped_zero_weight: int
    dropped_below_min_age: int
    dropped_missing_demographics: int
    topcoded_n: int


def build_analytic_sample(
    dnam_paths: Sequence[Path],
    demo_paths: Sequence[Path],
    *,
    weight_variable: str,
    min_age: float,
    age_topcode: float,
    log,
) -> Analytic:
    """Read DNMEPI (+ DEMO if needed), join on SEQN, and apply the analytic filters."""
    dnam = pd.concat([read_dnam_file(p) for p in dnam_paths], ignore_index=True)
    inputs = [str(p) for p in dnam_paths]
    source = ", ".join(Path(p).name for p in dnam_paths)

    # D25: abort if the named weight column is absent. Never substitute another.
    weight_col = resolve_column(dnam, weight_variable, label="survey weight",
                                source=source)
    id_col = resolve_column(dnam, ID_VAR, label="participant id", source=source)

    have_age = AGE_VAR in dnam.columns
    have_sex = SEX_VAR in dnam.columns
    if have_age and have_sex:
        age_source = sex_source = source
        joined = dnam
        log(f"  demographics taken from the DNAm file itself ({source}): "
            f"{AGE_VAR}, {SEX_VAR}")
    else:
        if not demo_paths:
            absent = [v for v, ok in ((AGE_VAR, have_age), (SEX_VAR, have_sex))
                      if not ok]
            raise MissingInput(
                f"{source} does not carry {' or '.join(absent)}, so a demographics "
                f"file is required. Pass --demo (or put DEMO.XPT under --data-dir).\n"
                f"  columns present: {sorted(dnam.columns)}"
            )
        demo = pd.concat([read_dnam_file(p) for p in demo_paths], ignore_index=True)
        demo_source = ", ".join(Path(p).name for p in demo_paths)
        resolve_column(demo, ID_VAR, label="participant id", source=demo_source)
        resolve_column(demo, AGE_VAR, label="age", source=demo_source)
        resolve_column(demo, SEX_VAR, label="sex", source=demo_source)
        joined = dnam.merge(demo[[ID_VAR, AGE_VAR, SEX_VAR]], on=id_col, how="inner")
        age_source = sex_source = demo_source
        inputs += [str(p) for p in demo_paths]
        log(f"  joined {len(dnam):,} DNAm rows to {len(demo):,} demographics rows "
            f"on {ID_VAR} -> {len(joined):,}")

    rows_read = int(len(joined))
    joined = joined.copy()
    joined["_W"] = pd.to_numeric(joined[weight_col], errors="coerce")
    joined["_AGE"] = pd.to_numeric(joined[AGE_VAR], errors="coerce")
    joined["_SEX"] = [sex_symbol(c) for c in joined[SEX_VAR]]

    # Zero / missing weight = not in the methylation subsample. Dropping these is not
    # a judgement call: a zero-weight row contributes nothing to a weighted estimate
    # but WOULD contribute to the unweighted n that decides suppression, making a
    # thin cell look adequately sized.
    good_w = joined["_W"].notna() & (joined["_W"] > 0.0)
    dropped_zero_weight = int((~good_w).sum())
    joined = joined[good_w]

    good_demo = joined["_AGE"].notna() & joined["_SEX"].notna()
    dropped_missing_demographics = int((~good_demo).sum())
    joined = joined[good_demo]

    # D19: an age ABOVE the topcode means this is not the cycle the registry claims.
    if len(joined):
        observed_max = float(joined["_AGE"].max())
        if observed_max > age_topcode:
            raise ValueError(
                f"observed maximum age {observed_max:g} exceeds the {DNMEPI_CYCLES} "
                f"topcode of {age_topcode:g}. NHANES topcodes age at 85 for "
                f"1999-2006, so this file is not the release this registry describes "
                f"(or --age-topcode is wrong). Refusing to fit an age regression "
                f"against an age variable that is not what it is assumed to be."
            )
    topcoded_n = int((joined["_AGE"] >= age_topcode).sum())

    below = joined["_AGE"] < min_age
    dropped_below_min_age = int(below.sum())
    joined = joined[~below]

    log(f"  analytic sample: {len(joined):,} rows "
        f"(read {rows_read:,}; dropped {dropped_zero_weight:,} zero/missing weight, "
        f"{dropped_missing_demographics:,} missing age/sex, "
        f"{dropped_below_min_age:,} under age {min_age:g})")
    if rows_read and dropped_below_min_age / max(rows_read, 1) > 0.01:
        log(f"  WARNING: {dropped_below_min_age:,} of {rows_read:,} rows are under "
            f"age {min_age:g}. DNMEPI is documented as adults 50+, so a large "
            f"under-50 fraction suggests this is not the DNMEPI release. Check "
            f"--inspect and --min-age before trusting these numbers.")
    if topcoded_n:
        log(f"  note: {topcoded_n:,} row(s) sit AT the age topcode ({age_topcode:g}). "
            f"Their true age is >= the topcode, so the age regression under-predicts "
            f"for them and their residual is biased upward. They are kept (dropping "
            f"the oldest of a 50+ sample would bias the slope) and counted here.")

    return Analytic(
        frame=joined, weight_variable=weight_col, age_source=age_source,
        sex_source=sex_source, inputs=inputs, rows_read=rows_read,
        dropped_zero_weight=dropped_zero_weight,
        dropped_below_min_age=dropped_below_min_age,
        dropped_missing_demographics=dropped_missing_demographics,
        topcoded_n=topcoded_n,
    )


def collect(
    specs: Sequence[ClockSpec],
    analytic: Analytic,
    *,
    definitions: Sequence[str],
    log,
) -> list[AccelSeries]:
    """Compute every requested (clock, convention) acceleration series."""
    frame = analytic.frame
    source = ", ".join(Path(p).name for p in map(Path, analytic.inputs))
    out: list[AccelSeries] = []

    for spec in specs:
        column = resolve_column(frame, spec.column, label=f"{spec.symbol} clock",
                                source=source)
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        age = frame["_AGE"].to_numpy(dtype="float64")
        weight = frame["_W"].to_numpy(dtype="float64")
        sexes = list(frame["_SEX"])
        bands = [age_band(a) for a in age]

        keep = np.isfinite(raw) & np.isfinite(age) & np.isfinite(weight)
        keep &= np.array([b is not None for b in bands])
        dropped = int((~keep).sum())
        if dropped:
            log(f"  {spec.symbol}: dropped {dropped:,} row(s) with a missing "
                f"{spec.column} value or an unbandable age")
        raw_k, age_k, w_k = raw[keep], age[keep], weight[keep]
        sex_k = [s for s, k in zip(sexes, keep) if k]
        band_k = [b for b, k in zip(bands, keep) if k]
        if raw_k.size == 0:
            log(f"  {spec.symbol}: no usable observation — nothing computed")
            continue

        for definition in definitions:
            if definition not in spec.conventions():
                raise AccelDefinitionUnavailable(
                    f"{spec.symbol} ({spec.column}) is a {spec.quantity} in "
                    f"{spec.unit}, not a predicted age, so the {definition!r} "
                    f"convention is undefined for it: subtracting a chronological "
                    f"age from it does not produce a quantity. Defined for this "
                    f"clock: {list(spec.conventions())}. Drop it from --clocks, or "
                    f"run with --accel-definition residual."
                )
            if definition == ACCEL_RESIDUAL:
                fit, values = weighted_linear_residuals(raw_k, age_k, w_k)
                log(f"  {spec.symbol} Residual: weighted fit {spec.column} = "
                    f"{fit.intercept:.4f} + {fit.slope:.4f} * {AGE_VAR} "
                    f"(n={fit.n:,}, sum(w)={fit.sum_w:,.0f})")
            else:
                fit, values = None, raw_k - age_k
                log(f"  {spec.symbol} Difference: {spec.column} - {AGE_VAR} "
                    f"(n={raw_k.size:,})")
            out.append(AccelSeries(
                spec=spec, definition=definition, values=np.asarray(values),
                weights=w_k, sexes=sex_k, bands=band_k, fit=fit,
            ))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 7. Emission — Schema A (with the clock-only fields) and Schema C
# ════════════════════════════════════════════════════════════════════════════


def record_id(spec: ClockSpec, sex: str, band: str, definition: str) -> str:
    """Schema A record id, WITH the convention spelled into the cycle tag.

    Schema A's id pattern is ``NHANESRef_<Marker>_<Sex>_<Band>_<CycleTag>``. Two
    conventions for one (marker, sex, band, cycle) would collide on that id and
    produce two contradictory ``RefMean`` atoms under ONE identifier — the silent
    double-valued failure D4 exists to prevent. The convention therefore goes into
    the cycle-tag component, so the ids stay distinct and the definition is visible
    in the identifier itself rather than only in a field.
    """
    # Terse on purpose: the identifier is repeated on every one of a record's ~17 field
    # atoms, so its length dominates the emitted file size — and the file size is what
    # silently binds (pln_chat drops an oversized .metta from execution with only a
    # print()). A 2-clock x 2-convention run measured 55,047 bytes against the
    # 60,000-byte limit with the long form, so one more declared clock would have
    # overflowed it. Nothing is lost: the record carries its marker, sex, band, cycles
    # and acceleration definition as field atoms, and a readable comment sits above it.
    return check_symbol(
        f"NR_{spec.symbol}_{short_sex(sex)}_{short_band(band)}"
        f"_{short_cycles(DNMEPI_CYCLE_TAG)}_{definition[:3]}",
        what="record id",
    )


def spread_id(spec: ClockSpec, definition: str) -> str:
    return check_symbol(
        f"NCS_{spec.symbol}_{short_cycles(DNMEPI_CYCLE_TAG)}_{definition[:3]}",
        what="record id",
    )


_DEFINITION_WARNING = (
    "TWO ACCELERATION CONVENTIONS MAY BE PRESENT IN THIS FILE.",
    "Every clock record carries (RefAccelDefinition <rid> Residual|Difference), and",
    "the definition is part of the record id. A consumer that matches",
    "(RefMarker $r AgeAccelGrim) WITHOUT also constraining the definition gets BOTH",
    "records and silently doubles its results. Constrain it:",
    "",
    "    (match $space (, (RefMarker          $r AgeAccelGrim)",
    "                     (RefAccelDefinition $r Residual)",
    "                     (RefSex             $r Male)",
    "                     (RefAgeBand         $r Age_50_59)",
    "                     (RefMean $r $mu) (RefSD $r $sd))",
    "       ($mu $sd))",
    "",
    "Residual is the convention that matches grim_age_core.metta's",
    "(ResidualOf AgeAccelGrim GrimAge) and the quantity Lu 2019's HR was fit to.",
    "Run with --accel-definition residual to emit a single-valued file.",
)


_SINGLE_DEFINITION_NOTE = (
    "Every clock record still carries (RefAccelDefinition <rid> ...) and spells the",
    "convention into its record id, so a consumer can — and should — constrain it:",
    "",
    "    (match $space (, (RefMarker          $r AgeAccelGrim)",
    "                     (RefAccelDefinition $r Residual)",
    "                     (RefSex             $r Male)",
    "                     (RefAgeBand         $r Age_50_59)",
    "                     (RefMean $r $mu) (RefSD $r $sd))",
    "       ($mu $sd))",
    "",
    "Doing so keeps the query correct if a later run of this ETL adds the other",
    "convention to the same file (--accel-definition both).",
)


def emit(
    series: Sequence[AccelSeries],
    analytic: Analytic,
    *,
    min_cell_n: int,
    min_age: float,
    age_topcode: float,
    budget: int,
    byte_budget: int,
    generator: str,
    declared: dict[str, str],
    skipped: Sequence[tuple[str, str, str]],
    log,
) -> tuple[MettaWriter, dict]:
    """Write Schema A reference records and the Schema C clock-spread records."""
    writer = MettaWriter(budget=budget, byte_budget=byte_budget)

    notes = [
        "Values are SURVEY-WEIGHTED by the DNMEPI subsample weight "
        f"{analytic.weight_variable}. The base weight {BASE_WEIGHT} is REFUSED by "
        "this ETL: it describes the full MEC sample, not this subsample.",
        "THE SHIPPED CLOCK COLUMNS ARE PREDICTED AGES, NOT ACCELERATIONS. Every "
        "record below is computed FROM a predicted age and names the computation in "
        "(RefTransform ...).",
        f"Cells with fewer than {min_cell_n} unweighted observations emit NOTHING "
        "(suppressed cells are reported on stderr, never emitted with a caveat).",
        f"Restricted to participants aged >= {min_age:g}; DNMEPI is an adults-50+ "
        "subsample. Age is topcoded at "
        f"{age_topcode:g} for {DNMEPI_CYCLES}; {analytic.topcoded_n:,} analytic "
        "row(s) sit at the topcode and no mean age is computed inside the top band.",
        "No design-based standard errors: correct NHANES SEs need Taylor-series "
        "linearization with the strata/PSU variables. RefUnweightedN and sum(w) are "
        "given instead.",
        "DunedinPoAm is the 2020 pace-of-aging measure, NOT DunedinPACE (2022). Any "
        "'PACE > 1.0' threshold belongs to DunedinPACE and must not be applied to it.",
        "SD BOOKKEEPING — the Schema C SpreadSDYears is the WHOLE-SAMPLE weighted SD; "
        "the Schema A RefSD values are band x sex SDs and differ from it. "
        "pln_risk_prediction.metta's (grimaccel-sd-to-years) converts a z BACK into "
        "years, so the constant it uses must be THE SAME SD that produced the z. A "
        "patient standardized against a Schema A record must be converted back with "
        "that record's RefSD; SpreadSDYears is the right replacement for the curated "
        "4.2 only for a z computed against the whole-sample spread.",
    ]
    writer.comment("\n".join(provenance_header(
        title="NHANES DNMEPI epigenetic clocks -> acceleration reference "
              "distributions (Schema A) + clock spread (Schema C)",
        generator=generator,
        inputs=list(analytic.inputs),
        notes=notes,
    )))
    writer.blank()
    writer.rule("READ THIS BEFORE CONSUMING ANY RECORD BELOW")
    present = sorted({item.definition for item in series})
    writer.comment(
        f"This file contains the {' and '.join(present)} convention"
        f"{'s' if len(present) > 1 else ''}."
    )
    for line in (_DEFINITION_WARNING if len(present) > 1
                 else _SINGLE_DEFINITION_NOTE):
        writer.comment(line)
    writer.blank()

    if skipped:
        writer.rule("Clocks present in the registry but NOT emitted (D22)")
        for column, symbol, reason in skipped:
            writer.comment(f"{column:<14} -> {symbol:<20} {reason}")
        writer.blank()

    summary = {"records": 0, "spreads": 0, "suppressed": [], "skipped": list(skipped)}

    for item in series:
        spec, definition = item.spec, item.definition
        unit = spec.accel_unit(definition)
        writer.rule(
            f"{spec.symbol} — from {spec.column} ({spec.unit}), {definition} "
            f"convention, {DNMEPI_CYCLES}, weight {analytic.weight_variable}"
        )
        writer.comment(f"registry confidence: {spec.confidence} — {spec.evidence}")
        writer.comment(f"definition: {ACCEL_DESCRIPTION[definition]}")
        if spec.notes:
            writer.comment(f"note: {spec.notes}")
        writer.comment(f"declared: {declared.get(spec.symbol, '(unchecked)')}")
        if item.fit is not None:
            writer.comment(
                f"weighted fit: {spec.column} = {item.fit.intercept:.6f} + "
                f"{item.fit.slope:.6f} * {AGE_VAR}  "
                f"(n={item.fit.n:,}, sum(w)={item.fit.sum_w:,.0f}); residuals are "
                f"what the records below describe, NOT the fitted values."
            )
        writer.blank()

        # ---- Schema C first: this is the headline deliverable ------------------
        whole = weighted_moments(item.values, item.weights)
        if spec.spreads_in_years():
            if whole is None:
                summary["suppressed"].append(
                    (spec.symbol, definition, "whole-sample", "no usable observation"))
            else:
                reason = suppressed_reason(n=whole.n, min_n=min_cell_n)
                if reason:
                    summary["suppressed"].append(
                        (spec.symbol, definition, "whole-sample", reason))
                else:
                    sid = spread_id(spec, definition)
                    writer.comment(
                        f"Schema C — the measured spread of this acceleration, in "
                        f"years. For AgeAccelGrim/Residual this is the quantity "
                        f"pln_risk_prediction.metta's curated "
                        f"(= (grimaccel-sd-to-years) 4.2) estimates."
                    )
                    writer.comment(
                        f"weighted mean={whole.mean:.6f}  n={whole.n:,}  "
                        f"sum(w)={whole.sum_w:,.0f}  "
                        f"sd_estimator={whole.sd_estimator}"
                    )
                    writer.atom(f"(: {sid} ClockAccelSpread)")
                    writer.atom(f"(SpreadMarker         {sid} {spec.symbol})")
                    writer.atom(f"(SpreadDefinition     {sid} {definition})")
                    writer.atom(f"(SpreadSDYears        {sid} {num(whole.sd)})")
                    writer.atom(f"(SpreadUnweightedN    {sid} {num(whole.n, places=0)})")
                    writer.atom(f"(SpreadWeightVariable {sid} "
                                f"{mstr(analytic.weight_variable)})")
                    writer.atom(f"(SpreadSourceCycles   {sid} {mstr(DNMEPI_CYCLES)})")
                    writer.atom(f"(SpreadProvenance     {sid} NHANES_Microdata)")
                    writer.blank()
                    summary["spreads"] += 1
        else:
            writer.comment(
                f"no Schema C record: SpreadSDYears is defined to be IN YEARS and "
                f"{spec.column} is a {spec.quantity} in {spec.unit}."
            )
            writer.blank()

        # ---- Schema A: one record per (sex, band) ------------------------------
        cells = item.cells()
        emitted_here = 0
        for sex in ("Male", "Female"):
            for band, _lo, _hi in AGE_BANDS:
                cell = cells.get((sex, band))
                if cell is None:
                    summary["suppressed"].append(
                        (spec.symbol, definition, f"{sex} {band}", "no observations"))
                    continue
                moments = weighted_moments(cell.values, cell.weights)
                if moments is None:
                    summary["suppressed"].append(
                        (spec.symbol, definition, f"{sex} {band}",
                         "no observation survived"))
                    continue
                reason = suppressed_reason(n=moments.n, min_n=min_cell_n)
                if reason:
                    summary["suppressed"].append(
                        (spec.symbol, definition, f"{sex} {band}", reason))
                    continue
                if not (moments.sd > 0.0):
                    summary["suppressed"].append(
                        (spec.symbol, definition, f"{sex} {band}",
                         "zero spread — an SD of 0 cannot standardize anything"))
                    continue

                rid = record_id(spec, sex, band, definition)
                writer.comment(
                    f"n={moments.n}  sum(w)={moments.sum_w:,.0f}  "
                    f"sd_estimator={moments.sd_estimator}"
                )
                writer.atom(f"(: {rid} ReferenceDistribution)")
                writer.atom(f"(RefMarker          {rid} {spec.symbol})")
                writer.atom(f"(RefSex             {rid} {sex})")
                writer.atom(f"(RefAgeBand         {rid} {band})")
                # Identity, always: an acceleration is signed and centred near zero,
                # so Log10 is not merely unnecessary, it is undefined for half the
                # sample. D17's log rule is about right-skewed positive analytes.
                writer.atom(f"(RefScale           {rid} {SCALE_IDENTITY})")
                writer.atom(f"(RefMean            {rid} {num(moments.mean)})")
                writer.atom(f"(RefSD              {rid} {num(moments.sd)})")
                writer.atom(f"(RefUnweightedN     {rid} {num(moments.n, places=0)})")
                writer.atom(f"(RefUnit            {rid} {mstr(unit)})")
                writer.atom(f"(RefWeightVariable  {rid} "
                            f"{mstr(analytic.weight_variable)})")
                writer.atom(f"(RefSourceVariable  {rid} {mstr(spec.column)})")
                writer.atom(f"(RefSourceFiles     {rid} "
                            f"{mstr(';'.join(Path(p).name for p in analytic.inputs))})")
                writer.atom(f"(RefSourceCycles    {rid} {mstr(DNMEPI_CYCLES)})")
                writer.atom(f"(RefAssayLot        {rid} {mstr(DNMEPI_ASSAY_LOT)})")
                writer.atom(f"(RefProvenance      {rid} NHANES_Microdata)")
                # D24 — the emitted quantity is DERIVED from the shipped column, so
                # the derivation is named rather than left implicit.
                writer.atom(f"(RefTransform       {rid} "
                            f"{mstr(ACCEL_TRANSFORM[definition])})")
                # D20 — the label that makes a mis-scaled exponent impossible to
                # produce by accident. Never conditional on anything.
                writer.atom(f"(RefAccelDefinition {rid} {definition})")
                writer.atom(f"(RefClockColumn     {rid} {mstr(spec.column)})")
                writer.blank()
                emitted_here += 1
                summary["records"] += 1

        if emitted_here == 0:
            writer.comment("no cell survived suppression — nothing emitted")
            writer.blank()

    # The guard that makes "emit the difference where the residual is meant"
    # impossible to do by accident: every ReferenceDistribution written above must
    # carry exactly one RefAccelDefinition. Checked on the rendered text rather than
    # trusted from the loop, so a future edit that adds an emission path without the
    # label fails here instead of shipping an unlabelled acceleration.
    text = writer.text()
    typed = len(re.findall(r"^\(: \S+ ReferenceDistribution\)", text, re.M))
    labelled = len(re.findall(r"^\(RefAccelDefinition\s+\S+\s+(?:%s)\)"
                              % "|".join(ACCEL_DEFINITIONS), text, re.M))
    if typed != labelled:
        raise AssertionError(
            f"internal error: {typed} ReferenceDistribution record(s) but {labelled} "
            f"(RefAccelDefinition ...) atom(s). An acceleration record without an "
            f"explicit convention would silently mis-scale "
            f"pln_risk_prediction.metta's exponent — refusing to write this file."
        )

    for symbol, definition, cell, reason in summary["suppressed"]:
        log(f"  suppressed {symbol}/{definition} {cell}: {reason}")
    log(f"  {summary['records']} reference record(s), {summary['spreads']} clock-"
        f"spread record(s), {writer.atom_count} atom(s) (budget {budget}), "
        f"{len(summary['suppressed'])} cell(s) suppressed, "
        f"{len(summary['skipped'])} clock(s) skipped as undeclared")
    return writer, summary


# ════════════════════════════════════════════════════════════════════════════
# 8. Registry / manifest reporting
# ════════════════════════════════════════════════════════════════════════════

MANIFEST_COLUMNS = ("release", "cycles", "column", "symbol", "quantity", "unit",
                    "weight", "conventions", "confidence", "declared", "notes")


def manifest_rows(registry: Sequence[ClockSpec],
                  declared: dict[str, str]) -> list[tuple[str, ...]]:
    rows = []
    for spec in registry:
        rows.append((
            "DNMEPI", DNMEPI_CYCLES, spec.column, spec.symbol, spec.quantity,
            spec.unit, SUBSAMPLE_WEIGHT, "+".join(spec.conventions()),
            spec.confidence,
            declared.get(spec.symbol, "UNDECLARED — not emitted"),
            spec.notes.replace("\n", " "),
        ))
    return rows


def render_manifest(registry: Sequence[ClockSpec], declared: dict[str, str]) -> str:
    lines = ["\t".join(MANIFEST_COLUMNS)]
    lines += ["\t".join(r) for r in manifest_rows(registry, declared)]
    return "\n".join(lines) + "\n"


def show_registry(registry: Sequence[ClockSpec], declared: dict[str, str]) -> None:
    print("NHANES DNMEPI clock registry — every entry is an UNVERIFIED claim about")
    print("CDC's published data (the CDC hosts are unreachable from this environment).")
    print(f"Release: DNMEPI, cycles {DNMEPI_CYCLES}, weight {SUBSAMPLE_WEIGHT} "
          f"(base {BASE_WEIGHT}, which is REFUSED).")
    print()
    header = f"{'column':<14} {'symbol':<20} {'quantity':<22} {'conf':<7} declared"
    print(header)
    print("-" * len(header))
    for spec in registry:
        where = declared.get(spec.symbol)
        mark = where if where else "NO  — undeclared, nothing emitted (D22)"
        print(f"{spec.column:<14} {spec.symbol:<20} "
              f"{spec.quantity + ' [' + spec.unit + ']':<22} "
              f"{spec.confidence:<7} {mark}")
    print()
    print("Conventions defined per clock (a Difference is undefined for a rate or a")
    print("length — subtracting a chronological age from one is not a quantity):")
    for spec in registry:
        print(f"  {spec.column:<14} {'+'.join(spec.conventions())}")
    print()
    print("Refused weight variables (using one on this subsample is biased, D20/D25):")
    for name in sorted(REFUSED_WEIGHTS):
        print(f"  {name}")
    print()
    print("Correct a wrong column name without editing code, e.g. for an .XPT export")
    print("whose names were truncated to the 8-character XPORT v5 limit:")
    print('  --registry over.json   with   {"AgeAccelGrim": {"column": "GRIMAGEM"}}')


# ════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="NHANES DNMEPI epigenetic clocks -> survey-weighted acceleration "
                    "reference distributions (Schema A) and clock spread (Schema C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The clock columns are PREDICTED AGES. Both acceleration conventions "
               "(Residual, Difference) are computed and every emitted record is "
               "labelled with the one it used. No NHANES microdata and no NHANES-"
               "derived number is committed to this repo; output defaults to build/.",
    )
    ap.add_argument("--dnam", action="append", type=Path, default=[],
                    help="NHANES DNAm release file (.sas7bdat, .xpt or .csv; "
                         "repeatable)")
    ap.add_argument("--demo", action="append", type=Path, default=[],
                    help="NHANES demographics file, needed only when the DNAm file "
                         f"does not itself carry {AGE_VAR}/{SEX_VAR} (repeatable)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"directory searched for a file not passed explicitly "
                         f"(default {DEFAULT_DATA_DIR})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"MeTTa output path (default {DEFAULT_OUTPUT}; under build/ "
                         f"because NHANES-derived numbers are never committed)")
    ap.add_argument("--clocks", default=None,
                    help="comma-separated clock symbols to emit (default: every "
                         "registry entry whose symbol is declared in the KB)")
    ap.add_argument("--accel-definition", default="residual",
                    choices=("residual", "difference", "both"),
                    help="acceleration convention(s) to emit. 'residual' is the one "
                         "that matches (ResidualOf AgeAccelGrim GrimAge) and the "
                         "quantity Lu 2019's HR was fit to; 'both' (default) emits "
                         "each under its own record id and (RefAccelDefinition ...)")
    ap.add_argument("--weight-variable", default=SUBSAMPLE_WEIGHT,
                    help=f"DNMEPI subsample weight (default {SUBSAMPLE_WEIGHT}). "
                         f"Passing a base/interview/other-subsample weight ABORTS")
    ap.add_argument("--min-age", type=float, default=DEFAULT_MIN_AGE,
                    help=f"exclude participants below this age (default "
                         f"{DEFAULT_MIN_AGE:g}; DNMEPI is an adults-50+ subsample and "
                         f"the age regression's slope depends on the range fit)")
    ap.add_argument("--age-topcode", type=float, default=AGE_TOPCODE,
                    help=f"age topcode for these cycles (default {AGE_TOPCODE:g}); an "
                         f"observed age above it aborts")
    ap.add_argument("--manifest", action="store_true",
                    help="print the clock manifest as TSV and exit")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                    help="repo directory scanned for declared Biomarker symbols")
    add_common_args(ap)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    registry = apply_overrides(REGISTRY, load_registry_override(args.registry))
    declared = declared_biomarker_symbols(args.repo_root)

    if args.manifest:
        sys.stdout.write(render_manifest(registry, declared))
        return 0
    if args.show_registry:
        show_registry(registry, declared)
        return 0
    if args.inspect:
        return run_inspect(args.inspect)

    if not declared:
        raise SystemExit(
            f"no Biomarker declarations found under {args.repo_root} — this ETL "
            f"refuses to emit atoms it cannot check against the KB. Pass --repo-root."
        )

    definitions = {
        "residual": (ACCEL_RESIDUAL,),
        "difference": (ACCEL_DIFFERENCE,),
        "both": ACCEL_DEFINITIONS,
    }[args.accel_definition]

    wanted = None
    if args.clocks:
        asked = [c.strip() for c in args.clocks.split(",") if c.strip()]
        known = {c.symbol.lower(): c.symbol for c in registry}
        known.update({c.column.lower(): c.symbol for c in registry})
        missing = sorted(a for a in asked if a.lower() not in known)
        if missing:
            raise SystemExit(
                f"unknown clock(s) {missing}; registered symbols: "
                f"{sorted(c.symbol for c in registry)} (columns also accepted: "
                f"{sorted(c.column for c in registry)})"
            )
        wanted = {known[a.lower()] for a in asked}

    chosen: list[ClockSpec] = []
    skipped: list[tuple[str, str, str]] = []
    for spec in registry:
        if wanted is not None and spec.symbol not in wanted:
            continue
        if not spec.emit:
            skipped.append((spec.column, spec.symbol,
                            "registry marks it not-emitted"))
            continue
        if spec.symbol not in declared:
            skipped.append((spec.column, spec.symbol,
                            "symbol UNDECLARED in the KB — no atom may name it (D22)"))
            continue
        chosen.append(spec)

    for column, symbol, reason in skipped:
        log(f"  skipped {column} -> {symbol}: {reason}")
    if not chosen:
        raise SystemExit(
            "no selected clock has a declared symbol, so nothing may be emitted. "
            "Declare the acceleration symbol in the hand-written MeTTa layer first "
            "(e.g. (Inheritance AgeAccelPheno EpigeneticAgeAcceleration)), then "
            "re-run. See --show-registry."
        )

    # Check the requested conventions against every chosen clock BEFORE touching a
    # file: an undefined convention is a request that can never succeed, and the user
    # should hear that immediately rather than after a multi-megabyte read.
    for spec in chosen:
        undefined = [d for d in definitions if d not in spec.conventions()]
        if undefined:
            raise AccelDefinitionUnavailable(
                f"{spec.symbol} ({spec.column}) is a {spec.quantity} in {spec.unit}, "
                f"not a predicted age, so {undefined} is undefined for it: "
                f"subtracting a chronological age from it does not produce a "
                f"quantity. Defined for this clock: {list(spec.conventions())}. Drop "
                f"it from --clocks, or run with --accel-definition residual."
            )

    weight_variable = check_weight_variable(args.weight_variable)

    log(f"NHANES DNMEPI clock ETL — cycles {DNMEPI_CYCLES}, weight {weight_variable}, "
        f"conventions {'+'.join(definitions)}, clocks "
        f"{', '.join(c.symbol for c in chosen)}")

    dnam_paths = resolve_inputs(args.dnam, args.data_dir, ("DNMEPI",), what="dnam")
    # Demographics are resolved permissively: the DNAm file may carry RIDAGEYR and
    # RIAGENDR itself, so "no DEMO file anywhere" is not yet an error. The decision is
    # made in build_analytic_sample from the columns actually present.
    demo_paths = resolve_inputs(args.demo, args.data_dir, ("DEMO", "DEMO_B"),
                                what="demo", required=False)
    analytic = build_analytic_sample(
        dnam_paths, demo_paths, weight_variable=weight_variable,
        min_age=args.min_age, age_topcode=args.age_topcode, log=log,
    )
    series = collect(chosen, analytic, definitions=definitions, log=log)
    if not series:
        raise SystemExit("no clock produced a usable series — nothing to emit")

    writer, _summary = emit(
        series, analytic,
        min_cell_n=args.min_cell_n, min_age=args.min_age,
        age_topcode=args.age_topcode, budget=args.atom_budget,
        byte_budget=args.byte_budget, generator=Path(__file__).name,
        declared=declared, skipped=skipped, log=log,
    )
    # The byte budget is the one that binds here, not the atom budget: a clock record
    # id carries the marker, sex, band, cycle tag AND convention, so these records are
    # much wider than a blood-analyte one. Warn before the wall rather than at it —
    # ByteBudgetExceeded is a hard refusal, and a user who has just declared a new
    # clock symbol should learn that the NEXT one will not fit.
    if writer.byte_size > 0.8 * args.byte_budget:
        log(f"  NOTE: output is {writer.byte_size:,} bytes, over 80% of the "
            f"{args.byte_budget:,}-byte budget (pln_chat silently drops a .metta file "
            f"larger than PLN_MAX_KB_FILE_BYTES). Declaring one more clock symbol "
            f"would likely exceed it; --accel-definition residual roughly halves the "
            f"output, or split the run with --clocks.")

    atoms = writer.write(args.output)
    log(f"wrote {args.output}  ({atoms} atoms, {writer.byte_size:,} bytes)")
    return 0


#: The failures a user can actually fix. Each carries a message that names the fix, so
#: they are reported as an error line rather than as a traceback that buries it.
_USER_FIXABLE = (
    MissingInput, MissingColumns, BiasedWeight, AccelDefinitionUnavailable,
    DegenerateFit, AtomBudgetExceeded, ByteBudgetExceeded, ValueError,
)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except _USER_FIXABLE as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
