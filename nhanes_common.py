#!/usr/bin/env python3
"""Shared plumbing for the NHANES ETLs (reference distributions + linked mortality).

This module is the correctness core of the NHANES integration. It owns the four
things that are easy to get quietly wrong and that both ETLs depend on:

  1. READING NHANES FILES  — SAS XPORT v5 (``.XPT``) via ``pandas.read_sas``, plus a
     CSV fallback for pre-converted extracts. Includes a MANDATORY repair for a
     pandas bug that silently corrupts genuine zeros (see ``fix_ibm_zero``).
  2. SURVEY WEIGHTING      — NHANES is a complex multistage probability sample with
     oversampling. Unweighted means/SDs are biased for the US population, so every
     statistic here is weighted, and the weight variable used is recorded in the
     emitted provenance.
  3. CENSORING             — a fixed-horizon absolute risk from follow-up data needs a
     product-limit (Kaplan-Meier) estimator, not a ``deaths / total`` proportion.
  4. HONEST SUPPRESSION    — a cell with too few observations emits NOTHING rather than
     a noisy estimate, matching the repo-wide "never invent a number" discipline.

It deliberately contains NO curated constants and NO NHANES values. It is plumbing:
the numbers come from whatever microdata the caller points it at.

Dependencies: pandas (already a declared repo dependency, see pln_chat/requirements.txt)
and its numpy. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════════
# 1. Reading NHANES files
# ════════════════════════════════════════════════════════════════════════════

# pandas' XPORT reader (pandas/io/sas/sas_xport.py::_parse_float_vec) has no special
# case for the all-zero IBM-370 byte pattern that SAS writes for a true 0.0. It runs
# the general conversion on it, producing an exponent field of 763 and therefore
# exactly 2**-260 instead of 0.0.
#
# Verified empirically against pandas 3.0.6 on a hand-written XPORT v5 file: the
# read-back value is BIT-IDENTICAL to 2**-260 (not merely close). Every other value
# tested — negatives, sub-1, large, irrational — round-trips exactly, and SAS missing
# values correctly become NaN. So the artifact is exactly this one sentinel.
#
# This matters for real NHANES data, not just fixtures: any genuine 0.0 in a lab or
# demographic file (a zero count, a zero weight for a participant outside a subsample)
# comes back as 2**-260. Left alone it would sail through every range check, enter a
# weighted mean as a ~0 value (harmless-looking) but would NOT compare equal to 0, so
# "drop zero weights" style filters would silently keep them.
#
# The repair is exact and collision-free: no NHANES analyte or weight can legitimately
# be 2**-260 (~5.4e-79); real values live between ~1e-3 and ~1e6.
IBM_ZERO_ARTIFACT = 2.0 ** -260


def fix_ibm_zero(frame: pd.DataFrame) -> pd.DataFrame:
    """Snap the pandas XPORT IBM-zero artifact back to a true 0.0, in place.

    Applied to every numeric column of every XPT file this module reads. Exact
    equality is used on purpose — see the IBM_ZERO_ARTIFACT note above.
    """
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame.loc[frame[col] == IBM_ZERO_ARTIFACT, col] = 0.0
    return frame


def _decode_bytes(frame: pd.DataFrame) -> pd.DataFrame:
    """Decode the bytes objects pandas returns for XPORT character columns."""
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = frame[col].map(
                lambda v: v.decode("ascii", "replace").strip() if isinstance(v, bytes) else v
            )
    return frame


class MissingColumns(RuntimeError):
    """A required variable is not present in the file — raised loudly, never guessed."""


def read_nhanes(path: Path | str, *, require: Sequence[str] = ()) -> pd.DataFrame:
    """Read an NHANES ``.XPT`` (SAS XPORT v5) or ``.csv`` extract.

    ``require`` names the variables the caller needs. If any is absent the error
    lists every column the file DOES contain, so a wrong registry entry is
    immediately diagnosable instead of silently producing wrong numbers. NHANES
    variable names are upper-cased by convention; lookups here are case-insensitive
    and the returned frame is upper-cased so downstream code can rely on it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. NHANES microdata is not bundled with this repo; "
            f"see docs/nhanes_integration.md for how to obtain it."
        )
    suffix = path.suffix.lower()
    if suffix in (".xpt", ".xport"):
        frame = pd.read_sas(path, format="xport")
        frame = fix_ibm_zero(frame)
    elif suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"unsupported NHANES input format: {path.suffix} ({path})")

    frame.columns = [str(c).upper().strip() for c in frame.columns]
    frame = _decode_bytes(frame)

    missing = [v for v in (r.upper() for r in require) if v not in frame.columns]
    if missing:
        raise MissingColumns(
            f"{path.name} is missing required variable(s) {missing}.\n"
            f"  columns present: {sorted(frame.columns)}\n"
            f"  If NHANES renamed this variable for this cycle, correct the registry "
            f"or pass --registry with an override (see --help)."
        )
    return frame


def read_fixed_width(path: Path | str, layout: Sequence[tuple]) -> pd.DataFrame:
    """Read a fixed-width ASCII file (the NHANES linked-mortality file format).

    ``layout`` is a sequence of ``(name, start, width)`` or ``(name, start, width, kind)``
    with **1-based inclusive** start columns, matching how record layouts are published.
    ``kind`` is ``"num"`` (default) or ``"str"``. Blank and all-dot values become NA.

    The ``kind`` distinction is not cosmetic. ``UCOD_LEADING`` is a CHARACTER field of
    zero-padded codes (``"001"`` … ``"010"``); coercing it to a number turns ``"001"``
    into ``1`` and silently breaks every comparison written against CDC's codebook.
    """
    names, colspecs, kinds = [], [], {}
    for entry in layout:
        name, start, width = entry[0], entry[1], entry[2]
        kind = entry[3] if len(entry) > 3 else "num"
        name = str(name).upper()
        names.append(name)
        colspecs.append((int(start) - 1, int(start) - 1 + int(width)))
        kinds[name] = kind

    frame = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str, header=None)
    for col in frame.columns:
        stripped = frame[col].astype(str).str.strip()
        stripped = stripped.replace({"": None, "nan": None, ".": None, "..": None, "...": None})
        if kinds.get(col, "num") == "str":
            frame[col] = stripped
        else:
            frame[col] = pd.to_numeric(stripped, errors="coerce")
    return frame


# ---------------------------------------------------------------------------
# The NHANES public-use Linked Mortality File
# ---------------------------------------------------------------------------
#
# The layout below is transcribed from CDC's own read-in program
# (``SAS_ReadInProgramAllSurveys.sas``, header "PUBLIC-USE LINKED MORTALITY FOLLOW-UP
# THROUGH DECEMBER 31, 2019"), so unlike the analyte registries it is not recall.
#
# Two traps it encodes:
#
#   * VINTAGE. The 2011-vintage file inserts ``CAUSEAVL`` at column 17 and shifts every
#     later field by one. Reading a 2019 file with the 2011 layout does not fail — it
#     reads the last two digits of the cause code plus the diabetes flag as the cause,
#     and three wrong bytes as the follow-up time, yielding plausible small integers.
#     So the vintage is a required, explicit argument and unknown vintages are refused.
#
#   * NHANES vs NHIS. Columns 22-42 hold NHIS-only fields and are blank in the NHANES
#     file, which means NHANES carries NO date of death (only person-months) and no
#     linkage-adjusted weight. ``read_linked_mortality`` asserts that blankness as a
#     vintage/­survey sanity check rather than assuming it.

LMF_LAYOUTS: dict[str, tuple[tuple, ...]] = {
    "2019": (
        ("SEQN", 1, 6),
        ("ELIGSTAT", 15, 1),
        ("MORTSTAT", 16, 1),
        ("UCOD_LEADING", 17, 3, "str"),
        ("DIABETES", 20, 1),
        ("HYPERTEN", 21, 1),
        ("PERMTH_INT", 43, 3),
        ("PERMTH_EXM", 46, 3),
    ),
    "2011": (
        ("SEQN", 1, 6),
        ("ELIGSTAT", 15, 1),
        ("MORTSTAT", 16, 1),
        ("CAUSEAVL", 17, 1),
        ("UCOD_LEADING", 18, 3, "str"),
        ("DIABETES", 21, 1),
        ("HYPERTEN", 22, 1),
        ("PERMTH_INT", 44, 3),
        ("PERMTH_EXM", 47, 3),
    ),
}

# UCOD_LEADING — CDC's leading-cause recode, verbatim from the read-in program's
# value labels. Note what "001" is and is NOT: it is all Diseases of heart, which
# includes hypertensive and rheumatic heart disease, cardiomyopathy, arrhythmias and
# heart failure. No public-use value isolates ischemic/coronary disease.
UCOD_LEADING_LABELS: dict[str, str] = {
    "001": "Diseases of heart (I00-I09, I11, I13, I20-I51)",
    "002": "Malignant neoplasms (C00-C97)",
    "003": "Chronic lower respiratory diseases (J40-J47)",
    "004": "Accidents / unintentional injuries (V01-X59, Y85-Y86)",
    "005": "Cerebrovascular diseases (I60-I69)",
    "006": "Alzheimer's disease (G30)",
    "007": "Diabetes mellitus (E10-E14)",
    "008": "Influenza and pneumonia (J09-J18)",
    "009": "Nephritis, nephrotic syndrome and nephrosis (N00-N07, N17-N19, N25-N27)",
    "010": "All other causes (residual)",
}

UCOD_HEART_DISEASE = "001"


class LinkageLayoutError(RuntimeError):
    """The mortality file does not match the declared vintage's record layout."""


def read_linked_mortality(path: Path | str, *, vintage: str = "2019") -> pd.DataFrame:
    """Read an NHANES public-use Linked Mortality File and validate its layout.

    Validations, each of which catches a silent-wrong-numbers failure mode:
      * the declared vintage is one we have an authoritative layout for;
      * the NHIS-only columns are blank, confirming this is the NHANES file at the
        declared vintage rather than a differently shifted one;
      * ``MORTSTAT`` is present exactly when ``ELIGSTAT == 1``. A parser that fills
        blanks with 0 would turn every linkage-ineligible participant into a censored
        survivor, deflating the event rate and inflating the denominator;
      * every observed ``UCOD_LEADING`` code is in CDC's documented value set.
    """
    if vintage not in LMF_LAYOUTS:
        raise LinkageLayoutError(
            f"unknown linked-mortality vintage {vintage!r}; known: "
            f"{sorted(LMF_LAYOUTS)}. Field positions changed between vintages, and "
            f"parsing with the wrong one yields plausible but wrong numbers rather than "
            f"an error — so an unrecognized vintage is refused rather than guessed."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The NHANES linked mortality file is public but is not "
            f"bundled with this repo; see docs/nhanes_integration.md."
        )

    frame = read_fixed_width(path, LMF_LAYOUTS[vintage])

    # NHIS-only block must be blank in the NHANES file.
    nhis_probe = read_fixed_width(path, (("NHIS_BLOCK", 22, 21, "str"),))
    non_blank = int(nhis_probe["NHIS_BLOCK"].notna().sum())
    if non_blank:
        raise LinkageLayoutError(
            f"{path.name}: columns 22-42 should be blank for an NHANES file at vintage "
            f"{vintage}, but {non_blank} of {len(nhis_probe)} records have content there. "
            f"This is either an NHIS file or a different vintage — refusing to parse "
            f"rather than silently misreading the cause and follow-up fields."
        )

    eligible = frame["ELIGSTAT"] == 1
    if not frame.loc[eligible, "MORTSTAT"].notna().all():
        raise LinkageLayoutError(
            f"{path.name}: some ELIGSTAT==1 records have a missing MORTSTAT, which the "
            f"layout does not allow. Suspect a vintage mismatch."
        )
    if frame.loc[~eligible, "MORTSTAT"].notna().any():
        raise LinkageLayoutError(
            f"{path.name}: some ELIGSTAT!=1 records carry a MORTSTAT. Suspect a vintage "
            f"mismatch. Ineligible records must be excluded as a domain, never read as "
            f"censored survivors."
        )

    codes = set(frame["UCOD_LEADING"].dropna().unique())
    unknown = sorted(c for c in codes if c not in UCOD_LEADING_LABELS)
    if unknown:
        raise LinkageLayoutError(
            f"{path.name}: UCOD_LEADING contains undocumented code(s) {unknown}; "
            f"documented values are {sorted(UCOD_LEADING_LABELS)}. Suspect a vintage "
            f"mismatch (the 2011 layout shifts this field by one column)."
        )
    return frame


def max_observed_followup_months(frame: pd.DataFrame, *, time_col: str = "PERMTH_EXM") -> float:
    """Longest follow-up among CENSORED records — the horizon a file can actually support.

    Per-cycle maximum follow-up must be DERIVED, not assumed: with linkage ending
    2019-12-31, the later cycles have less than ten years of follow-up, and estimating a
    120-month risk there produces a number driven by extrapolation. Censored records are
    the right ones to read, since a decedent's follow-up ends at death, not at the
    linkage cut-off.
    """
    censored = frame.loc[frame["MORTSTAT"] == 0, time_col].dropna()
    return float(censored.max()) if len(censored) else float("nan")


# ════════════════════════════════════════════════════════════════════════════
# 2. Age bands and sex — aligned with the existing KB
# ════════════════════════════════════════════════════════════════════════════
#
# The band edges are NOT a free choice: pln_risk_prediction.metta's
# `baseline-risk-chd` already partitions age at 50 / 60 / 70, so a data-backed
# baseline table has to use the same partition to be a drop-in comparison. The
# reference-distribution ETL reuses the same bands for consistency (and because
# `MeasuredZ` is defined against an *age- and sex-adjusted* mean, so the band IS the
# age adjustment).
#
# Symbol names are MeTTa-safe (no '<', no '+', which the reader would choke on).

AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Age_lt50", -math.inf, 50.0),
    ("Age_50_59", 50.0, 60.0),
    ("Age_60_69", 60.0, 70.0),
    ("Age_70p", 70.0, math.inf),
)

SEX_CODES = {1: "Male", 2: "Female"}


def age_band(age: float) -> Optional[str]:
    """Half-open band [lo, hi) containing ``age``; None if age is missing."""
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return None
    for name, lo, hi in AGE_BANDS:
        if lo <= float(age) < hi:
            return name
    return None


def sex_symbol(code) -> Optional[str]:
    """NHANES RIAGENDR code -> the repo's Sex symbol (system_types.metta)."""
    try:
        return SEX_CODES.get(int(code))
    except (TypeError, ValueError):
        return None


# ════════════════════════════════════════════════════════════════════════════
# 3. Survey-weighted statistics
# ════════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Measurement scale — why a marker needs one
# ---------------------------------------------------------------------------
#
# `patient_profile.metta`'s grounding rule is symmetric: z > 1 is Elevated, z < -1 is
# Low. That is a Gaussian-flavoured cutoff, and applying it to a z computed on the RAW
# scale of a log-normal analyte does not merely blur the categories, it removes one.
#
# Measured on a realistic simulated CRP distribution (geometric mean 0.2 mg/dL,
# geometric SD 3, n = 200,000):
#
#     scale     P(z > +1)    P(z < -1)    max z
#     raw          7.56%        0.00%      80.4
#     log10       15.83%       15.86%       5.0
#     normal      15.87%       15.87%        --
#
# On the raw scale the `Low` branch is UNREACHABLE, and raising the threshold does not
# fix it (P(z < -t) stays 0.00% at t = 1.0, 1.5 and 2.0). Worse, "Elevated" would then
# mean a different population percentile for CRP than for a symmetric marker like HbA1c,
# so `diagnose-patient`'s coverage counts would silently weight the two differently.
#
# So a marker's scale is part of its definition, the reference moments are computed on
# that scale, and the z must be computed on the same scale as the reference it is
# measured against. The emitted record carries the scale so the pairing is auditable.

SCALE_IDENTITY = "Identity"
SCALE_LOG10 = "Log10"
MARKER_SCALES = (SCALE_IDENTITY, SCALE_LOG10)


def apply_scale(values, scale: str):
    """Transform values onto a marker's declared scale, dropping what cannot be mapped.

    Returns ``(transformed, kept_mask)``. For ``Log10`` a non-positive value has no
    logarithm, so it is DROPPED rather than clamped or floored: clamping would invent a
    value at the detection limit and quietly pile probability mass onto one point.
    Callers must apply ``kept_mask`` to the weights so the pairing stays aligned.
    """
    array = np.asarray(list(values), dtype="float64")
    if scale == SCALE_IDENTITY:
        return array, np.isfinite(array)
    if scale == SCALE_LOG10:
        keep = np.isfinite(array) & (array > 0.0)
        out = np.full(array.shape, np.nan)
        out[keep] = np.log10(array[keep])
        return out, keep
    raise ValueError(f"unknown marker scale {scale!r}; known: {MARKER_SCALES}")


def standardize(value: float, mean: float, sd: float, scale: str) -> Optional[float]:
    """z for one raw observation against a reference computed on the SAME scale.

    Returns None when the value cannot be placed (non-positive on a log scale, or a
    degenerate reference SD) — never a fabricated z.
    """
    if sd is None or not math.isfinite(sd) or sd <= 0.0:
        return None
    transformed, keep = apply_scale([value], scale)
    if not bool(keep[0]):
        return None
    return float((transformed[0] - mean) / sd)


@dataclass
class WeightedMoments:
    """A weighted mean/SD for one cell, with the honesty fields kept alongside."""

    mean: float
    sd: float
    n: int                      # unweighted observation count — the reliability signal
    sum_w: float                # estimated population size
    sd_estimator: str           # "population" | "n_minus_1_corrected"


def weighted_moments(
    values: Iterable[float],
    weights: Iterable[float],
    *,
    correction: str = "population",
) -> Optional[WeightedMoments]:
    """Survey-weighted mean and SD of one analyte in one cell.

    Two-pass on purpose: the one-pass identity ``E[wx^2] - E[wx]^2`` suffers
    catastrophic cancellation when the mean is large relative to the spread (true of
    e.g. total cholesterol, mean ~200, SD ~40), and can even return a negative
    variance. The two-pass form sums squared deviations about the computed mean and is
    numerically stable.

    ``correction``:
      * ``"population"`` (default) — ``Var = sum(w*(x-mean)^2) / sum(w)``. This is the
        estimator we WANT, and the choice is substantive rather than stylistic: the
        z-score this feeds is "how many SDs above the population mean is this patient",
        so the denominator should be the finite-POPULATION SD, not a sample SD.
      * ``"n_minus_1_corrected"`` — the same, times ``n/(n-1)`` on the UNWEIGHTED count.
        Offered for comparison with survey packages that report a sample-analogue SD.

    A bias "correction" dividing by ``sum(w) - 1`` is deliberately NOT offered: survey
    weights are probability weights, so ``sum(w)`` estimates a population of millions
    and that correction is both numerically inert and conceptually wrong (it is only
    valid for frequency weights).

    Returns None if no observation survives (never a fabricated cell).
    """
    v = np.asarray(list(values), dtype="float64")
    w = np.asarray(list(weights), dtype="float64")
    if v.shape != w.shape:
        raise ValueError(f"values/weights length mismatch: {v.shape} vs {w.shape}")

    keep = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    v, w = v[keep], w[keep]
    n = int(v.size)
    if n == 0:
        return None

    sum_w = float(w.sum())
    if not (sum_w > 0.0):
        return None

    mean = float((w * v).sum() / sum_w)
    var = float((w * (v - mean) ** 2).sum() / sum_w)     # two-pass, stable
    var = max(var, 0.0)                                   # guard float noise at zero spread

    if correction == "n_minus_1_corrected":
        if n < 2:
            return None
        var *= n / (n - 1)
    elif correction != "population":
        raise ValueError(f"unknown SD correction: {correction!r}")

    return WeightedMoments(
        mean=mean, sd=math.sqrt(var), n=n, sum_w=sum_w, sd_estimator=correction
    )


@dataclass
class KMEstimate:
    """A weighted product-limit cumulative-incidence estimate at a fixed horizon."""

    cumulative_incidence: float
    survival: float
    horizon: float
    n: int              # unweighted at-risk count entering the estimate
    events: int         # unweighted event count before the horizon
    sum_w: float
    censored_before_horizon: int


def weighted_kaplan_meier(
    times: Iterable[float],
    events: Iterable[int],
    weights: Iterable[float],
    *,
    horizon: float,
) -> Optional[KMEstimate]:
    """Weighted Kaplan-Meier cumulative incidence at ``horizon``.

    Why not ``deaths / total``: follow-up is right-censored — the linkage ends on a
    fixed calendar date, so a participant examined late has far less than ``horizon``
    of observation. A naive proportion counts every such person as a non-event for the
    full horizon and therefore UNDERESTIMATES the cumulative incidence, the more so the
    shorter the average follow-up. The product-limit estimator removes each censored
    person from the risk set at their own censoring time instead.

    Estimator, over distinct event times t_1 < ... < t_k that are <= horizon:
        W_j = sum of w_i over { i : t_i >= t_j }        (weighted at-risk set)
        D_j = sum of w_i over { i : t_i == t_j, d_i=1 } (weighted events, ties summed)
        S(horizon) = product_j ( 1 - D_j / W_j )
        cumulative incidence = 1 - S(horizon)

    Returns None if nothing is at risk (never a fabricated cell).
    """
    t = np.asarray(list(times), dtype="float64")
    d = np.asarray(list(events), dtype="float64")
    w = np.asarray(list(weights), dtype="float64")
    if not (t.shape == d.shape == w.shape):
        raise ValueError(f"times/events/weights length mismatch: {t.shape} {d.shape} {w.shape}")

    keep = np.isfinite(t) & np.isfinite(d) & np.isfinite(w) & (w > 0.0) & (t >= 0.0)
    t, d, w = t[keep], d[keep], w[keep]
    n = int(t.size)
    if n == 0:
        return None

    survival = 1.0
    event_times = np.unique(t[(d == 1) & (t <= horizon)])
    for tj in event_times:
        at_risk = w[t >= tj].sum()
        if at_risk <= 0.0:
            continue
        dj = w[(t == tj) & (d == 1)].sum()
        survival *= 1.0 - (dj / at_risk)
    survival = min(max(survival, 0.0), 1.0)

    return KMEstimate(
        cumulative_incidence=1.0 - survival,
        survival=survival,
        horizon=float(horizon),
        n=n,
        events=int(((d == 1) & (t <= horizon)).sum()),
        sum_w=float(w.sum()),
        censored_before_horizon=int(((d == 0) & (t < horizon)).sum()),
    )


@dataclass
class CIFEstimate:
    """A weighted Aalen-Johansen cause-specific cumulative-incidence estimate."""

    cumulative_incidence: float       # CIF for the cause of interest at the horizon
    overall_survival: float           # S(horizon), all event types pooled
    competing_incidence: float        # CIF for everything else at the horizon
    naive_km_incidence: float         # 1 - KM treating competing events as censoring
    horizon: float
    n: int
    events: int                       # unweighted cause-of-interest events before horizon
    competing_events: int
    sum_w: float
    censored_before_horizon: int


def weighted_aalen_johansen(
    times: Iterable[float],
    causes: Iterable,
    weights: Iterable[float],
    *,
    horizon: float,
    cause,
) -> Optional[CIFEstimate]:
    """Weighted cause-specific cumulative incidence (Aalen-Johansen) at ``horizon``.

    ``causes`` holds one label per person: the cause of the event, or a falsy value
    (None / 0 / "" / NaN) for someone censored. ``cause`` selects the cause of interest.

    WHY NOT 1 - KAPLAN-MEIER: for a cause-specific risk the other causes of death are
    COMPETING EVENTS, not censoring. Censoring means "this person is still at risk, we
    just stopped watching"; a person who died of cancer is not still at risk of dying of
    heart disease. Treating competing deaths as censoring assumes they would have gone on
    to experience the cause of interest at the same rate as survivors, which OVERSTATES
    the cause-specific incidence — materially so in an older cohort where competing
    mortality is common. ``naive_km_incidence`` is reported alongside precisely so the
    size of that bias is visible rather than assumed away.

    Estimator, over distinct event times t_j <= horizon (any cause):
        W_j   = sum of w over { i : t_i >= t_j }              (weighted at-risk)
        D_j   = sum of w over events of ANY cause at t_j
        Dk_j  = sum of w over events of `cause` at t_j
        S(t_j-) = product over t_l < t_j of ( 1 - D_l / W_l ) (overall survival, lagged)
        CIF_k(horizon) = sum_j  S(t_j-) * ( Dk_j / W_j )

    The lagged survival factor is the whole point: a cause-k event at t_j can only happen
    to someone who survived everything up to t_j.

    Returns None if nothing is at risk (never a fabricated cell).
    """
    t = np.asarray(list(times), dtype="float64")
    raw_causes = list(causes)
    w = np.asarray(list(weights), dtype="float64")
    if not (t.size == len(raw_causes) == w.size):
        raise ValueError(
            f"times/causes/weights length mismatch: {t.size} {len(raw_causes)} {w.size}"
        )

    def _is_event(value) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value) or value == 0     # 0 is a legitimate label, "" / None are not

    def _norm(value):
        return str(value).strip() if value is not None else None

    target = _norm(cause)
    is_event = np.array([_is_event(c) for c in raw_causes], dtype=bool)
    is_target = np.array(
        [e and _norm(c) == target for c, e in zip(raw_causes, is_event)], dtype=bool
    )

    keep = np.isfinite(t) & np.isfinite(w) & (w > 0.0) & (t >= 0.0)
    t, w, is_event, is_target = t[keep], w[keep], is_event[keep], is_target[keep]
    n = int(t.size)
    if n == 0:
        return None

    survival = 1.0                       # S(t_j-) carried forward
    cif_target = 0.0
    cif_competing = 0.0
    naive_survival = 1.0                 # KM treating competing events as censoring

    for tj in np.unique(t[is_event & (t <= horizon)]):
        at_risk = w[t >= tj].sum()
        if at_risk <= 0.0:
            continue
        at_tj = t == tj
        d_all = w[at_tj & is_event].sum()
        d_target = w[at_tj & is_target].sum()
        cif_target += survival * (d_target / at_risk)
        cif_competing += survival * ((d_all - d_target) / at_risk)
        naive_survival *= 1.0 - (d_target / at_risk)
        survival *= 1.0 - (d_all / at_risk)

    return CIFEstimate(
        cumulative_incidence=min(max(cif_target, 0.0), 1.0),
        overall_survival=min(max(survival, 0.0), 1.0),
        competing_incidence=min(max(cif_competing, 0.0), 1.0),
        naive_km_incidence=min(max(1.0 - naive_survival, 0.0), 1.0),
        horizon=float(horizon),
        n=n,
        events=int((is_target & (t <= horizon)).sum()),
        competing_events=int((is_event & ~is_target & (t <= horizon)).sum()),
        sum_w=float(w.sum()),
        censored_before_horizon=int((~is_event & (t < horizon)).sum()),
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. Cell suppression
# ════════════════════════════════════════════════════════════════════════════
#
# NCHS presentation standards suppress estimates from cells that are too small to be
# reliable. We do not attempt to reproduce the full NCHS rule (it needs design-based
# relative standard errors, which need the strata/PSU variables and a linearization
# package). We apply the part that is defensible with the data at hand — a minimum
# unweighted denominator — and we SUPPRESS rather than emit a caveat, because an
# emitted number gets consumed by inference and a caveat does not.

DEFAULT_MIN_CELL_N = 30
DEFAULT_MIN_EVENTS = 5          # for a risk/incidence cell


def suppressed_reason(
    *, n: int, min_n: int = DEFAULT_MIN_CELL_N,
    events: Optional[int] = None, min_events: int = DEFAULT_MIN_EVENTS,
) -> Optional[str]:
    """Return a suppression reason, or None if the cell may be emitted."""
    if n < min_n:
        return f"n={n} below minimum cell size {min_n}"
    if events is not None and events < min_events:
        return f"events={events} below minimum {min_events}"
    return None


# ════════════════════════════════════════════════════════════════════════════
# 5. MeTTa emission
# ════════════════════════════════════════════════════════════════════════════
#
# ATOM BUDGET — measured, not guessed. hyperon 0.2.10 does not merely slow down on a
# large space: it ABORTS the process with a non-unwinding Rust panic in
# hyperon-space/src/index/trie.rs (TrieKeyStorage::get_atom_unchecked unwrapping None)
# during `match`. Measured on this repo's full inference stack + patient layer:
#
#     +3,456 emitted atoms  -> loads and queries fine
#     +4,608 emitted atoms  -> hard abort (SIGABRT, uncatchable from Python)
#
# Because the failure is an abort rather than an exception, no amount of defensive
# Python catches it — the only protection is to not emit that many atoms. The default
# budget below sits under the measured-good point with headroom. This is the same
# constraint scripts/run_etl.sh documents for the HAGR ETLs ("hyperon 0.2.10 panics
# when querying a space past a few thousand atoms"), now quantified.

DEFAULT_ATOM_BUDGET = 3000

_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


class AtomBudgetExceeded(RuntimeError):
    """Emission would produce more atoms than hyperon can query without aborting."""


def check_symbol(sym: str, *, what: str = "symbol") -> str:
    """Reject anything that is not a plain MeTTa symbol before it reaches a KB file."""
    if not _SYMBOL_RE.match(str(sym)):
        raise ValueError(
            f"invalid MeTTa {what}: {sym!r} — must match {_SYMBOL_RE.pattern} "
            f"(no spaces, parens, '<', '+', or leading digit)"
        )
    return str(sym)


def num(value: float, *, places: int = 6) -> str:
    """Format a number for a MeTTa atom: fixed notation, no exponent, no NaN/inf.

    hyperon's reader does not accept ``nan``/``inf``, and exponent forms like ``1e-05``
    are read as symbols rather than numbers by some versions, so everything is written
    in plain decimal and non-finite values are refused rather than silently emitted.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"refusing to emit non-finite number: {value!r}")
    text = f"{v:.{places}f}"
    if text.startswith("-0.") and float(text) == 0.0:
        text = text[1:]                     # avoid '-0.000000'
    return text


def mstr(text: str) -> str:
    """A quoted MeTTa string with quotes/backslashes escaped."""
    escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class MettaWriter:
    """Accumulates atoms with a header, counts them, and enforces the atom budget."""

    def __init__(self, *, budget: int = DEFAULT_ATOM_BUDGET) -> None:
        self.budget = int(budget)
        self._lines: list[str] = []
        self._atoms = 0

    # -- structure -------------------------------------------------------------
    def comment(self, text: str = "") -> "MettaWriter":
        for line in (str(text).splitlines() or [""]):
            self._lines.append(f";; {line}".rstrip())
        return self

    def rule(self, text: str) -> "MettaWriter":
        self._lines.append(";; " + "=" * 68)
        if text:
            self._lines.append(f";; {text}")
            self._lines.append(";; " + "=" * 68)
        return self

    def blank(self) -> "MettaWriter":
        self._lines.append("")
        return self

    # -- atoms -----------------------------------------------------------------
    def atom(self, text: str) -> "MettaWriter":
        self._atoms += 1
        if self._atoms > self.budget:
            raise AtomBudgetExceeded(
                f"emission reached {self._atoms} atoms, over the budget of {self.budget}. "
                f"hyperon 0.2.10 hard-aborts (not raises) past roughly 4,000 extra atoms "
                f"on this KB — see the ATOM BUDGET note in nhanes_common.py. Narrow the "
                f"marker/outcome set, coarsen the cells, or raise --atom-budget knowingly."
            )
        self._lines.append(str(text))
        return self

    @property
    def atom_count(self) -> int:
        return self._atoms

    def text(self) -> str:
        return "\n".join(self._lines).rstrip() + "\n"

    def write(self, path: Path | str) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
        return self._atoms


def provenance_header(
    *, title: str, generator: str, inputs: Sequence[str], notes: Sequence[str] = ()
) -> list[str]:
    """The standard comment header every generated NHANES KB file carries."""
    lines = [
        title,
        "",
        f"GENERATED by {generator} — do not edit by hand; re-run the ETL instead.",
        "",
        "Source microdata (NHANES is public but is NOT redistributed in this repo):",
    ]
    lines += [f"  - {i}" for i in inputs] or ["  - (none recorded)"]
    if notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in notes]
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 6. Registry override plumbing (shared by both ETLs)
# ════════════════════════════════════════════════════════════════════════════
#
# Every NHANES file name and variable name in this integration is a factual claim
# about CDC's published data that the author could not verify against the source (the
# CDC hosts are unreachable from the environment this was written in). Rather than
# bake those claims in as if they were certain, each ETL keeps them in one explicit
# registry table, prints them on request, fails loudly when one does not match the
# file, and accepts a JSON override so a user with the real files can correct an entry
# without editing code.


def load_registry_override(path: Optional[Path | str]) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: registry override must be a JSON object")
    return data


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """CLI flags shared by both NHANES ETLs."""
    parser.add_argument(
        "--registry", type=Path, default=None,
        help="JSON file overriding NHANES file/variable names (see --show-registry)",
    )
    parser.add_argument(
        "--show-registry", action="store_true",
        help="print the built-in NHANES variable registry (with confidence) and exit",
    )
    parser.add_argument(
        "--inspect", type=Path, default=None,
        help="print the column names of an NHANES file and exit (diagnose a name mismatch)",
    )
    parser.add_argument(
        "--atom-budget", type=int, default=DEFAULT_ATOM_BUDGET,
        help=f"maximum atoms to emit (default {DEFAULT_ATOM_BUDGET}; hyperon aborts past ~4000)",
    )
    parser.add_argument(
        "--min-cell-n", type=int, default=DEFAULT_MIN_CELL_N,
        help=f"suppress a cell with fewer unweighted observations (default {DEFAULT_MIN_CELL_N})",
    )
    return parser


def run_inspect(path: Path) -> int:
    """--inspect: show what is actually in a file, so a bad registry entry is obvious."""
    frame = read_nhanes(path)
    print(f"{path}  ({len(frame):,} rows, {len(frame.columns)} columns)")
    for col in sorted(frame.columns):
        series = frame[col]
        non_null = int(series.notna().sum())
        extra = ""
        if pd.api.types.is_numeric_dtype(series) and non_null:
            extra = f"  min={series.min():g} max={series.max():g}"
        print(f"  {col:<12} non-null={non_null:<8}{extra}")
    return 0
