#!/usr/bin/env python3
"""NHANES blood analytes -> survey-weighted reference distributions (Schema A atoms).

WHAT THIS PRODUCES, AND WHY IT IS NEEDED
----------------------------------------
`patient_profile.metta` grounds a patient on ``(MeasuredZ <patient> <marker> <z>)``:
an age- and sex-adjusted standardized value. Today those z-scores are hand-typed for
the two demo patients, so nothing in the repo can turn a real lab report ("CRP
0.42 mg/dL") into the ``MeasuredZ`` the inference stack consumes. A z needs a
reference mean and SD for the patient's age band and sex, and that reference has to
come from a population sample rather than from a remembered textbook range.

This ETL reads NHANES microdata the USER supplies, joins the demographics file to
each lab file on ``SEQN``, bands participants by age and sex, and emits one
``ReferenceDistribution`` record per (marker, sex, age band) as Schema A atoms.

FIVE DECISIONS THAT ARE EASY TO GET QUIETLY WRONG, AND WHAT IS DONE HERE
-----------------------------------------------------------------------
1. SURVEY WEIGHTS ARE PER ANALYTE, NOT GLOBAL. NHANES oversamples, so an unweighted
   mean is biased for the US population — but the *correct* weight depends on which
   subsample the analyte was measured in. ``WTMEC4YR`` for the MEC-examined analytes
   (CRP, HbA1c); ``WTSAF4YR``, read FROM THE FASTING LAB FILE rather than from DEMO,
   for fasting glucose; ``WTSCY4YR`` for the surplus-sera cystatin C. The weight
   variable is looked up in an explicit table keyed by (weight family, pooled cycle
   set) and is recorded in every emitted atom. There is no default and no guess: an
   unlisted cycle set aborts.

2. THE MEASUREMENT SCALE IS PART OF THE MARKER'S DEFINITION. ``z->status`` uses a
   symmetric +/-1 SD cutoff. On the raw scale of a log-normal analyte such as CRP the
   ``Low`` branch is unreachable (measured: P(z < -1) = 0.00%), so CRP's moments are
   computed on ``Log10`` and the record carries ``(RefScale ... Log10)`` so a consumer
   cannot pair a log reference with a raw value. HbA1c is roughly symmetric ->
   ``Identity``. A non-positive value on a log scale is DROPPED, never clamped.

3. CYCLES MAY ONLY BE POOLED ACROSS A COMMON ASSAY. Each (marker, cycle) carries an
   assay-lot id; pooling two cycles whose lots disagree raises ``AssayDiscontinuity``
   rather than warning. The known trap is CRP: mg/dL latex nephelometry through 2010,
   hsCRP in mg/L from 2015, and NO CRP OR hsCRP AT ALL in 2011-2014 — so a "pool
   everything" run would silently average two different quantities.

4. ONLY DECLARED MARKER SYMBOLS MAY BE EMITTED. An atom naming a symbol the KB never
   declares is a dangling reference the inference layer cannot type-check. Exactly
   four blood analytes have a declared ``Biomarker`` symbol in this repo — ``CRP``,
   ``FastingGlucose``, ``HbA1c`` (mechanistic_bridges.metta) and ``PlasmaCystatinC``
   (grim_age_core.metta) — and the declaration is re-verified by scanning the repo at
   run time. Total/HDL cholesterol, creatinine, insulin and eGFR are NOT declared
   anywhere, so they appear in the manifest as "symbol undeclared, not emitted" and
   produce no atoms. Triglycerides IS declared since item #12 (mechanistic_bridges.metta),
   but has no calibrated NHANES reference here, so it stays "declared, not emitted".

5. EVERY NHANES NAME HERE IS AN UNVERIFIED CLAIM. The CDC hosts are unreachable from
   the environment this was written in, so no file name, variable name or assay
   description below could be checked against the source. They therefore live in ONE
   registry table, each entry carrying a confidence and its evidence; ``--show-registry``
   prints it, ``--inspect`` prints a file's real columns, a missing variable raises
   ``MissingColumns`` listing every column present, and ``--registry`` overrides an
   entry without editing code. A registry entry is never consulted as a fallback
   guess: CDC explicitly warns against substituting ``LBXSGL`` for ``LBXGLU``, so a
   missing variable is an error, never a search for something that looks close.

NO NHANES-DERIVED NUMBER IS COMMITTED TO THIS REPO. The default output is under
``build/`` (gitignored), exactly as the HAGR ETL outputs are staged, and the repo
ships no microdata (see ``data/nhanes/README.md``).

Usage:
    python3 nhanes_reference_etl.py --show-registry
    python3 nhanes_reference_etl.py --manifest > data/nhanes/MANIFEST.tsv
    python3 nhanes_reference_etl.py --inspect data/nhanes/LAB11.XPT
    python3 nhanes_reference_etl.py \
        --demo data/nhanes/DEMO.XPT   --demo data/nhanes/DEMO_B.XPT \
        --lab  data/nhanes/LAB11.XPT  --lab  data/nhanes/L11_B.XPT \
        --output build/nhanes_reference.metta
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from nhanes_common import (
    declared_biomarker_symbols,
    short_band,
    short_sex,
    short_cycles,
    AGE_BANDS,
    AtomBudgetExceeded,
    MettaWriter,
    MissingColumns,
    SCALE_IDENTITY,
    SCALE_LOG10,
    add_common_args,
    age_band,
    apply_scale,
    check_symbol,
    load_registry_override,
    mstr,
    num,
    provenance_header,
    read_nhanes,
    run_inspect,
    sex_symbol,
    suppressed_reason,
    design_se_of_weighted_mean,
    weighted_moments,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "nhanes"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "nhanes_reference.metta"

#: Adults only by default. The bands below start at "under 50", which in raw NHANES
#: includes 3-year-olds; a "reference distribution" mixing children into an adult
#: patient's comparison group is simply the wrong reference, so the ETL restricts to
#: adults and records the cut-off in the generated header (Schema A has no field for
#: it, and the schema is fixed).
DEFAULT_MIN_AGE = 20.0

URL_PATTERN = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{first_year}/DataFiles/{basename}.XPT"
)

DEMOGRAPHIC_VARS = ("SEQN", "RIAGENDR", "RIDAGEYR")

# The masked variance stratum and PSU. These are what make a DESIGN-BASED standard error
# possible (nhanes_common.design_se_of_weighted_mean): the naive SE of a weighted mean
# ignores clustering, which inflates variance, and stratification, which deflates it.
# They are OPTIONAL on purpose — a pre-converted extract may not carry them, and a missing
# SE is honest where a naive one would not be. When both are present the SE is computed
# and emitted; when either is absent the record simply carries no SE.
DESIGN_VARS = ("SDMVSTRA", "SDMVPSU")


def file_url(first_year: str, basename: str) -> str:
    return URL_PATTERN.format(first_year=first_year, basename=basename)


# ════════════════════════════════════════════════════════════════════════════
# 1. Cycle registry
# ════════════════════════════════════════════════════════════════════════════
#
# v1 covers 1999-2000 + 2001-2002 only, and that is a deliberate restriction rather
# than laziness: those two cycles are the ones NHANES publishes matching FOUR-YEAR
# subsample weights for (WTMEC4YR / WTSAF4YR / WTSCY4YR), they are the cycles in which
# all four declared analytes were measured on a common assay, and they are the cycles
# the DNMEPI epigenetic-clock subsample was drawn from — so a later step can join the
# clock data to the same reference population.


@dataclass(frozen=True)
class CycleSpec:
    """One NHANES two-year cycle."""

    name: str           # "1999-2000"
    tag: str            # MeTTa-safe atom-id fragment, "1999_2000"
    first_year: str     # the {first_year} path element in the CDC URL
    demo_file: str      # demographics basename for this cycle
    age_topcode: float  # RIDAGEYR is topcoded at this age in this cycle
    confidence: str
    evidence: str


CYCLES: dict[str, CycleSpec] = {
    "1999-2000": CycleSpec(
        name="1999-2000", tag="1999_2000", first_year="1999", demo_file="DEMO",
        age_topcode=85.0, confidence="high",
        evidence="DEMO is the 1999-2000 demographics basename; age topcoded at 85 "
                 "for 1999-2006. Unverified against CDC from this environment.",
    ),
    "2001-2002": CycleSpec(
        name="2001-2002", tag="2001_2002", first_year="2001", demo_file="DEMO_B",
        age_topcode=85.0, confidence="high",
        evidence="'_B' is the 2001-2002 suffix convention; age topcoded at 85 "
                 "for 1999-2006. Unverified against CDC from this environment.",
    ),
}

DEFAULT_CYCLES = ("1999-2000", "2001-2002")


def cycle_set_key(cycles: Sequence[str]) -> str:
    """Key for the weight table: a single cycle name, or the pooled span.

    Pooling 1999-2000 with 2001-2002 gives "1999-2002", which is the span NHANES
    publishes four-year weights for. The key is what selects the weight variable, so
    it must describe the POOLED set and not any one cycle.
    """
    ordered = sorted(cycles, key=lambda c: CYCLES[c].first_year)
    if not ordered:
        raise ValueError("cycle_set_key() needs at least one cycle")
    if len(ordered) == 1:
        return ordered[0]
    return f"{ordered[0].split('-')[0]}-{ordered[-1].split('-')[1]}"


# ════════════════════════════════════════════════════════════════════════════
# 2. Weight registry  (D25: subsample weights, per analyte, from an explicit table)
# ════════════════════════════════════════════════════════════════════════════
#
# The weight is NOT a property of the marker alone: it is a property of
# (which subsample the analyte was measured in, which cycles are being pooled).
# Using WTMEC4YR on the fasting subsample, or a four-year weight on a single cycle,
# both produce numbers that look fine and are wrong. So the mapping is a table with
# no default: a (family, cycle-set) pair that is not listed aborts, and the user
# supplies the real name via --registry rather than the ETL inventing one.


@dataclass(frozen=True)
class WeightSpec:
    variable: str
    source: str          # "DEMO" (read from the demographics file) | "LAB"
    confidence: str
    evidence: str


WEIGHT_TABLE: dict[tuple[str, str], WeightSpec] = {
    # MEC-examined analytes: the weight lives in DEMO.
    ("WTMEC", "1999-2002"): WeightSpec(
        "WTMEC4YR", "DEMO", "medium",
        "Four-year MEC exam weight published in the 1999-2000 and 2001-2002 "
        "demographics files for pooling those two cycles.",
    ),
    ("WTMEC", "1999-2000"): WeightSpec(
        "WTMEC2YR", "DEMO", "low",
        "Two-year MEC exam weight. Only needed for a single-cycle run; the name was "
        "not verifiable here — check with --inspect DEMO.XPT before trusting it.",
    ),
    ("WTMEC", "2001-2002"): WeightSpec(
        "WTMEC2YR", "DEMO", "low",
        "Two-year MEC exam weight. Only needed for a single-cycle run; the name was "
        "not verifiable here — check with --inspect DEMO_B.XPT before trusting it.",
    ),
    # Fasting subsample: the weight lives in the FASTING LAB FILE, not in DEMO.
    ("WTSAF", "1999-2002"): WeightSpec(
        "WTSAF4YR", "LAB", "medium",
        "Four-year fasting subsample weight, carried by LAB10AM / L10AM_B themselves. "
        "Reading a fasting weight from DEMO is a known error mode: DEMO's weights "
        "cover the whole MEC sample, not the morning fasting subsample.",
    ),
    # Surplus-sera cystatin C: a four-year weight only, and see the marker note about
    # the eligible sample not being nationally representative.
    ("WTSCY", "1999-2002"): WeightSpec(
        "WTSCY4YR", "LAB", "low",
        "Surplus-sera cystatin C weight, carried by SSCYST_A / SSCYST_B. Published "
        "only as a four-year weight, so a single-cycle cystatin C run is refused.",
    ),
}


def resolve_weight(family: str, cycles: Sequence[str]) -> WeightSpec:
    key = cycle_set_key(cycles)
    spec = WEIGHT_TABLE.get((family, key))
    if spec is None:
        listed = sorted(k for f, k in WEIGHT_TABLE if f == family)
        raise WeightUnavailable(
            f"no survey weight is registered for weight family {family!r} over cycle "
            f"set {key!r}.\n"
            f"  registered cycle sets for {family}: {listed}\n"
            f"  NHANES publishes a subsample weight for a SPECIFIC pooled span; using "
            f"a four-year weight on one cycle (or the reverse) is biased, and this ETL "
            f"will not guess the name. Either run the registered cycle set, or supply "
            f"the correct variable with --registry (see --help)."
        )
    return spec


class WeightUnavailable(RuntimeError):
    """No registered survey weight for this (analyte subsample, cycle set)."""


class AssayDiscontinuity(RuntimeError):
    """Refusing to pool cycles whose assay lots disagree (D8/D18)."""


class MissingInput(RuntimeError):
    """A required NHANES file was not supplied — names the URL and expected path."""


# ════════════════════════════════════════════════════════════════════════════
# 3. Marker registry — ONE table, every entry carrying a confidence
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class MarkerSpec:
    """One analyte: the repo symbol, the NHANES names, and how to treat it."""

    symbol: str                      # the repo's declared Biomarker symbol
    variable: str                    # NHANES variable holding the value
    files: dict[str, str]            # cycle -> lab-file basename
    unit: str
    scale: str                       # SCALE_IDENTITY | SCALE_LOG10
    weight_family: str               # key into WEIGHT_TABLE
    assay_lots: dict[str, str]       # cycle -> assay-lot id; disagreement blocks pooling
    confidence: str                  # high | medium | low
    evidence: str
    declared_in: str = ""            # where the repo declares the symbol
    emit: bool = True                # False => manifest-only, no atoms
    notes: str = ""

    def cycles(self) -> list[str]:
        return sorted(self.files, key=lambda c: CYCLES[c].first_year)


CONFIDENCE_LEVELS = ("high", "medium", "low")

# ---------------------------------------------------------------------------
# The four analytes with a DECLARED symbol in this repo AND a reference. Nothing else may emit
# (Triglycerides is declared too, but below it has no calibrated reference: emit=False).
# ---------------------------------------------------------------------------
REGISTRY: list[MarkerSpec] = [
    MarkerSpec(
        symbol="CRP",
        variable="LBXCRP",
        files={"1999-2000": "LAB11", "2001-2002": "L11_B"},
        unit="mg/dL",
        scale=SCALE_LOG10,
        weight_family="WTMEC",
        assay_lots={
            "1999-2000": "CRP_mgdL_latex_nephelometry_1999_2010",
            "2001-2002": "CRP_mgdL_latex_nephelometry_1999_2010",
        },
        confidence="medium",
        evidence="LBXCRP in LAB11 (1999-2000) / L11_B (2001-2002), mg/dL, latex-enhanced "
                 "nephelometry. Same assay lot across both cycles, so they pool.",
        declared_in="mechanistic_bridges.metta (Inheritance CRP Biomarker)",
        notes="Right-skewed -> Log10; on the raw scale the Low branch of z->status is "
              "unreachable. Do NOT pool with the 2015+ hsCRP series: that is mg/L on a "
              "different assay, and there is NO CRP or hsCRP variable at all in "
              "2011-2014, so a gap in the series is expected rather than a lost file.",
    ),
    MarkerSpec(
        symbol="HbA1c",
        variable="LBXGH",
        files={"1999-2000": "LAB10", "2001-2002": "L10_B"},
        unit="percent",
        scale=SCALE_IDENTITY,
        weight_family="WTMEC",
        assay_lots={
            "1999-2000": "HbA1c_Primus_CLC330_1999_2004",
            "2001-2002": "HbA1c_Primus_CLC330_1999_2004",
        },
        confidence="medium",
        evidence="LBXGH in LAB10 (1999-2000) / L10_B (2001-2002), percent of total "
                 "haemoglobin, Primus CLC330 boronate-affinity HPLC through 2004.",
        declared_in="mechanistic_bridges.metta (Inheritance HbA1c Biomarker)",
        notes="Roughly symmetric -> Identity. Later cycles switched instrument "
              "(Tosoh 2.2 in 2005-2006, Tosoh G7 in 2007-2010); those carry different "
              "lot ids and will refuse to pool with these.",
    ),
    MarkerSpec(
        symbol="FastingGlucose",
        variable="LBXGLU",
        files={"1999-2000": "LAB10AM", "2001-2002": "L10AM_B"},
        unit="mg/dL",
        scale=SCALE_IDENTITY,
        weight_family="WTSAF",
        assay_lots={
            "1999-2000": "Glucose_hexokinase_1999_2002",
            "2001-2002": "Glucose_hexokinase_1999_2002",
        },
        confidence="low",
        evidence="LBXGLU in LAB10AM (1999-2000) / L10AM_B (2001-2002), mg/dL. The "
                 "morning fasting subsample, so the weight WTSAF4YR is read from THIS "
                 "file and not from DEMO. No assay break is documented within "
                 "1999-2002, but that absence was not verifiable here -> low.",
        declared_in="mechanistic_bridges.metta (Inheritance FastingGlucose Biomarker)",
        notes="NEVER substitute LBXSGL (the biochemistry-profile glucose): CDC warns "
              "these are different assays on a different sample. A missing LBXGLU is "
              "an error, not a cue to look for something similar.",
    ),
    MarkerSpec(
        symbol="PlasmaCystatinC",
        variable="SSCYPC",
        files={"1999-2000": "SSCYST_A", "2001-2002": "SSCYST_B"},
        unit="mg/L",
        scale=SCALE_IDENTITY,
        weight_family="WTSCY",
        assay_lots={
            "1999-2000": "CystatinC_preIFCC_1999_2002",
            "2001-2002": "CystatinC_preIFCC_1999_2002",
        },
        confidence="low",
        evidence="SSCYPC in the surplus-sera files SSCYST_A / SSCYST_B, mg/L, weight "
                 "WTSCY4YR. Pre-IFCC calibration (ERM-DA471 recalibrated values are a "
                 "different lot and must not be pooled with these).",
        declared_in="grim_age_core.metta (Inheritance PlasmaCystatinC PlasmaProteinBiomarker)",
        notes="THE ELIGIBLE SAMPLE IS NOT NATIONALLY REPRESENTATIVE: cystatin C was "
              "assayed on stored surplus sera, so WTSCY4YR reweights an availability-"
              "determined subset. Treat the emitted moments as subsample descriptive "
              "statistics, not US population reference values. Surplus-sera files may "
              "also sit under a different CDC path than the URL pattern used here.",
    ),
    # -----------------------------------------------------------------------
    # Analytes the ETL knows about but MUST NOT emit: no declared symbol exists
    # in any .metta file, so an atom naming them would dangle. They are listed
    # here (and in MANIFEST.tsv) so the omission is visible and auditable rather
    # than looking like an oversight. To enable one, the hand-written MeTTa layer
    # must first declare the symbol with its type; then flip emit=True.
    # -----------------------------------------------------------------------
    MarkerSpec(
        symbol="Triglycerides", variable="LBXTR",
        files={"1999-2000": "LAB13AM", "2001-2002": "L13AM_B"},
        unit="mg/dL", scale=SCALE_LOG10, weight_family="WTSAF",
        assay_lots={"1999-2000": "unverified", "2001-2002": "unverified"},
        confidence="low",
        evidence="Declared a Biomarker in mechanistic_bridges.metta by item #12 (a fasting value, a curated "
                 "threshold in the patient builder), but there is still no calibrated NHANES reference for it: "
                 "the surplus-sera lots are unverified.",
        emit=False, notes="declared, not emitted: no calibrated reference",
    ),
    MarkerSpec(
        symbol="TotalCholesterol", variable="LBXTC",
        files={"1999-2000": "LAB13", "2001-2002": "L13_B"},
        unit="mg/dL", scale=SCALE_IDENTITY, weight_family="WTMEC",
        assay_lots={"1999-2000": "unverified", "2001-2002": "unverified"},
        confidence="low",
        evidence="No Biomarker declaration anywhere in the KB. Never substitute LBXSCH.",
        emit=False, notes="symbol undeclared, not emitted",
    ),
    MarkerSpec(
        symbol="HDLCholesterol", variable="LBDHDL",
        files={"1999-2000": "LAB13", "2001-2002": "L13_B"},
        unit="mg/dL", scale=SCALE_IDENTITY, weight_family="WTMEC",
        assay_lots={"1999-2000": "unverified", "2001-2002": "unverified"},
        confidence="low",
        evidence="No Biomarker declaration anywhere in the KB.",
        emit=False, notes="symbol undeclared, not emitted",
    ),
    MarkerSpec(
        symbol="SerumCreatinine", variable="LBXSCR",
        files={"1999-2000": "LAB18", "2001-2002": "L40_B"},
        unit="mg/dL", scale=SCALE_IDENTITY, weight_family="WTMEC",
        assay_lots={
            "1999-2000": "Creatinine_1999_2000_uncorrected",
            "2001-2002": "Creatinine_2001_2002",
        },
        confidence="low",
        evidence="No Biomarker declaration anywhere in the KB. Also carries a real "
                 "assay discontinuity: 1999-2000 needs the Selvin 2007 Deming "
                 "correction (standard = 1.013 x NHANES + 0.147), which would have to "
                 "be emitted as a labelled RefTransform, never silently applied.",
        emit=False, notes="symbol undeclared, not emitted; 1999-2000 vs 2001-2002 "
                          "assay lots differ, so these two cycles would refuse to pool",
    ),
    MarkerSpec(
        symbol="SerumInsulin", variable="LBXIN",
        files={"1999-2000": "LAB10AM", "2001-2002": "L10AM_B"},
        unit="uU/mL", scale=SCALE_LOG10, weight_family="WTSAF",
        assay_lots={"1999-2000": "unverified", "2001-2002": "unverified"},
        confidence="low",
        evidence="No Biomarker declaration anywhere in the KB.",
        emit=False, notes="symbol undeclared, not emitted",
    ),
]


# ---------------------------------------------------------------------------
# Declaration check (D22) — enforced against the repo at run time, not trusted
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Registry overrides
# ---------------------------------------------------------------------------
_OVERRIDABLE_SCALARS = ("variable", "unit", "scale", "weight_family",
                        "confidence", "evidence", "notes")
_OVERRIDABLE_DICTS = ("files", "assay_lots")


def apply_overrides(registry: Sequence[MarkerSpec], override: dict) -> list[MarkerSpec]:
    """Apply a ``--registry`` JSON override, keyed by marker symbol.

    Dict-valued fields (``files``, ``assay_lots``) are MERGED per cycle so a user can
    correct one cycle's file name without restating the other. Everything else is
    replaced. An unknown marker or field is an error rather than a silent no-op —
    a typo'd override that appears to work is exactly the failure this guards.
    """
    by_symbol = {m.symbol: m for m in registry}
    for symbol, patch in override.items():
        spec = by_symbol.get(symbol)
        if spec is None:
            raise ValueError(
                f"--registry override names unknown marker {symbol!r}; "
                f"known markers: {sorted(by_symbol)}"
            )
        if not isinstance(patch, dict):
            raise ValueError(f"--registry entry for {symbol!r} must be a JSON object")
        for key, value in patch.items():
            if key in _OVERRIDABLE_DICTS:
                merged = dict(getattr(spec, key))
                merged.update({str(k): str(v) for k, v in dict(value).items()})
                setattr(spec, key, merged)
            elif key in _OVERRIDABLE_SCALARS:
                setattr(spec, key, str(value))
            elif key == "emit":
                spec.emit = bool(value)
            else:
                raise ValueError(
                    f"--registry override for {symbol!r} has unknown field {key!r}; "
                    f"overridable: {sorted(_OVERRIDABLE_SCALARS + _OVERRIDABLE_DICTS + ('emit',))}"
                )
        if spec.scale not in (SCALE_IDENTITY, SCALE_LOG10):
            raise ValueError(f"{symbol}: unknown scale {spec.scale!r}")
        if spec.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"{symbol}: confidence must be one of {CONFIDENCE_LEVELS}")
        unknown_cycles = sorted(set(spec.files) - set(CYCLES))
        if unknown_cycles:
            raise ValueError(f"{symbol}: unknown cycle(s) {unknown_cycles}; known: {sorted(CYCLES)}")
    return list(registry)


# ════════════════════════════════════════════════════════════════════════════
# 4. Input resolution — a missing file names its URL and its expected path
# ════════════════════════════════════════════════════════════════════════════


def index_inputs(paths: Iterable[Path]) -> dict[str, Path]:
    """Map uppercased basename (no suffix) -> path, for the files the user supplied."""
    index: dict[str, Path] = {}
    for path in paths:
        index[path.stem.upper()] = path
    return index


def resolve_file(basename: str, supplied: dict[str, Path], data_dir: Path,
                 *, first_year: str, what: str) -> Path:
    """Find one NHANES file, or abort with the download URL and the expected path.

    Order: an explicitly supplied path with this basename, then ``data_dir`` with
    each of the extensions the reader supports. Failure is a hard error carrying the
    exact CDC URL, because the one thing a user needs at that moment is the URL.
    """
    key = basename.upper()
    if key in supplied:
        return supplied[key]
    for suffix in (".XPT", ".xpt", ".csv", ".CSV"):
        candidate = data_dir / f"{basename}{suffix}"
        if candidate.exists():
            return candidate
    raise MissingInput(
        f"{what} file {basename} was not supplied and is not in {data_dir}.\n"
        f"  fetch:          {file_url(first_year, basename)}\n"
        f"  expected local: {data_dir / (basename + '.XPT')}\n"
        f"  or pass it explicitly:  --lab /path/to/{basename}.XPT\n"
        f"  NHANES microdata is NOT bundled with this repo (see data/nhanes/README.md); "
        f"if CDC renamed this file, correct the registry with --registry."
    )


def require_columns(frame, columns: Sequence[str], label: str) -> None:
    """MissingColumns with the full column list — never a guess at a similar name."""
    missing = [c for c in columns if c.upper() not in frame.columns]
    if missing:
        raise MissingColumns(
            f"{label} is missing required variable(s) {missing}.\n"
            f"  columns present: {sorted(frame.columns)}\n"
            f"  This ETL never falls back to a similarly named variable (CDC warns "
            f"that e.g. LBXSGL is not a substitute for LBXGLU). Correct the registry "
            f"entry or pass --registry with an override."
        )


# ════════════════════════════════════════════════════════════════════════════
# 5. Aggregation
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Cell:
    """Accumulated raw values + weights for one (marker, sex, age band) cell."""

    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    strata: list[str] = field(default_factory=list)
    psu: list[str] = field(default_factory=list)
    design_available: bool = True      # cleared if any contributing cycle lacked the columns


@dataclass
class MarkerResult:
    """What one marker produced: emitted cells, suppressed cells, and provenance."""

    spec: MarkerSpec
    cycles: list[str]
    weight: WeightSpec
    cells: dict[tuple[str, str], Cell]
    scale_dropped: int = 0
    rows_joined: int = 0


def assay_lot_for(spec: MarkerSpec, cycles: Sequence[str]) -> str:
    """The shared assay lot, or refuse to pool (D8/D18).

    This is the check that stops the CRP trap: mg/dL latex nephelometry and mg/L hsCRP
    are different quantities, and their weighted average is not an estimate of either.
    Pooling is allowed only when every cycle in the set carries the same lot id.
    """
    lots = {}
    for cycle in cycles:
        lot = spec.assay_lots.get(cycle)
        if lot is None:
            raise AssayDiscontinuity(
                f"{spec.symbol}: no assay lot registered for cycle {cycle}; refusing to "
                f"pool an analyte whose assay comparability is unknown."
            )
        lots.setdefault(lot, []).append(cycle)
    if len(lots) > 1:
        detail = "; ".join(f"{lot}: {sorted(cs)}" for lot, cs in sorted(lots.items()))
        raise AssayDiscontinuity(
            f"{spec.symbol}: refusing to pool cycles measured on different assay lots "
            f"({detail}). Pooling across an assay or unit discontinuity averages two "
            f"different quantities. Run the cycles separately with --cycles, or, if the "
            f"lots really are comparable, say so explicitly with --registry."
        )
    return next(iter(lots))


def check_age_topcode(cycle: CycleSpec, ages) -> list[str]:
    """Respect the age topcode (D19): no band may be finer than it.

    NHANES stores every age above the topcode AS the topcode, so a band boundary above
    it would partition a spike rather than a distribution, and no mean age inside the
    top band would mean anything. This ETL computes no mean age at all; the check here
    is that the band edges are coarser than the topcode, plus a warning when the file's
    observed maximum age disagrees with the registered topcode (a sign the file is from
    a different era than the registry thinks).
    """
    warnings: list[str] = []
    for name, _lo, hi in AGE_BANDS:
        if math.isfinite(hi) and hi > cycle.age_topcode:
            raise ValueError(
                f"age band {name} has an upper edge of {hi} above cycle {cycle.name}'s "
                f"age topcode of {cycle.age_topcode}: NHANES stores every older "
                f"participant AS the topcode, so that boundary splits a spike, not a "
                f"distribution. Coarsen AGE_BANDS or drop the cycle."
            )
    finite = [a for a in ages if a is not None and not (isinstance(a, float) and math.isnan(a))]
    if finite:
        observed = float(max(finite))
        if observed > cycle.age_topcode + 0.5:
            warnings.append(
                f"{cycle.name}: observed max RIDAGEYR {observed:g} exceeds the "
                f"registered topcode {cycle.age_topcode:g} — is this file really from "
                f"this cycle?"
            )
    return warnings


def collect(
    registry: Sequence[MarkerSpec],
    selected_cycles: Sequence[str],
    supplied: dict[str, Path],
    data_dir: Path,
    *,
    min_age: float,
    log,
) -> tuple[list[MarkerResult], list[str]]:
    """Read the files and accumulate per-cell values and weights.

    DEMO is read once per cycle and cached; every lab file is read once and cached.
    The join is an inner join on SEQN — a participant without a lab value, or a lab
    record without demographics, contributes nothing rather than a defaulted value.
    """
    demo_cache: dict[str, tuple[Path, "object"]] = {}
    lab_cache: dict[Path, "object"] = {}
    inputs: list[str] = []
    results: list[MarkerResult] = []
    warnings: list[str] = []

    def demo_for(cycle: str):
        if cycle not in demo_cache:
            spec = CYCLES[cycle]
            path = resolve_file(spec.demo_file, supplied, data_dir,
                               first_year=spec.first_year, what="demographics")
            frame = read_nhanes(path, require=DEMOGRAPHIC_VARS)
            warnings.extend(check_age_topcode(spec, frame["RIDAGEYR"].tolist()))
            inputs.append(f"{cycle}  {spec.demo_file}  {path}  ({len(frame):,} rows)")
            demo_cache[cycle] = (path, frame)
        return demo_cache[cycle]

    def lab_for(path: Path, require):
        if path not in lab_cache:
            lab_cache[path] = read_nhanes(path)
        frame = lab_cache[path]
        require_columns(frame, require, path.name)
        return frame

    for spec in registry:
        cycles = [c for c in spec.cycles() if c in selected_cycles]
        if not cycles:
            log(f"  {spec.symbol}: no selected cycle has a registered file — skipped")
            continue
        weight = resolve_weight(spec.weight_family, cycles)
        assay_lot_for(spec, cycles)          # raises on a discontinuity, before any read
        result = MarkerResult(spec=spec, cycles=cycles, weight=weight, cells={})

        for cycle in cycles:
            cycle_spec = CYCLES[cycle]
            _demo_path, demo = demo_for(cycle)
            basename = spec.files[cycle]
            lab_path = resolve_file(basename, supplied, data_dir,
                                    first_year=cycle_spec.first_year, what="lab")
            lab_require = [spec.variable]
            if weight.source == "LAB":
                lab_require.append(weight.variable)
            lab = lab_for(lab_path, lab_require)
            if weight.source == "DEMO":
                require_columns(demo, [weight.variable], f"{cycle} demographics")
            inputs.append(
                f"{cycle}  {basename}  {lab_path}  ({len(lab):,} rows, {spec.variable})"
            )

            demo_cols = ["SEQN", "RIAGENDR", "RIDAGEYR"]
            has_design = all(v in demo.columns for v in DESIGN_VARS)
            if has_design:
                demo_cols.extend(DESIGN_VARS)
            if weight.source == "DEMO":
                demo_cols.append(weight.variable)
            lab_cols = ["SEQN", spec.variable]
            if weight.source == "LAB":
                lab_cols.append(weight.variable)
            merged = demo[demo_cols].merge(lab[lab_cols], on="SEQN", how="inner")
            result.rows_joined += len(merged)

            for row in merged.itertuples(index=False):
                record = dict(zip(merged.columns, row))
                age = record["RIDAGEYR"]
                band = age_band(age)
                sex = sex_symbol(record["RIAGENDR"])
                if band is None or sex is None:
                    continue
                if age is None or not np.isfinite(float(age)) or float(age) < min_age:
                    continue
                value = record[spec.variable]
                w = record[weight.variable]
                if value is None or w is None:
                    continue
                value, w = float(value), float(w)
                if not (np.isfinite(value) and np.isfinite(w)) or w <= 0.0:
                    continue
                cell = result.cells.setdefault((sex, band), Cell())
                if has_design:
                    cell.strata.append(str(record["SDMVSTRA"]))
                    cell.psu.append(str(record["SDMVPSU"]))
                else:
                    cell.design_available = False
                cell.values.append(value)
                cell.weights.append(w)

        results.append(result)
    return results, sorted(set(inputs))


# ════════════════════════════════════════════════════════════════════════════
# 6. Emission — Schema A, exactly
# ════════════════════════════════════════════════════════════════════════════


def record_id(spec: MarkerSpec, sex: str, band: str, tag: str) -> str:
    """A compact, unique, stable identifier for one emitted reference cell.

    Terse on purpose: the identifier is repeated on every one of the record's ~15 field
    atoms, so its length dominates the emitted file size, and the file size is what
    silently binds (see nhanes_common's BYTE BUDGET note — pln_chat drops an oversized
    .metta file from execution with only a print()). Nothing is lost, because the record
    carries its marker, sex, band and cycles as field atoms and the emitter writes a
    readable comment above it.
    """
    identifier = f"NR_{spec.symbol}_{short_sex(sex)}_{short_band(band)}_{short_cycles(tag)}"
    return check_symbol(identifier, what="record id")


def emit(
    results: Sequence[MarkerResult],
    *,
    inputs: Sequence[str],
    min_cell_n: int,
    min_age: float,
    budget: int,
    generator: str,
    extra_notes: Sequence[str] = (),
    log,
) -> tuple[MettaWriter, dict]:
    """Write Schema A records for every cell that survives suppression."""
    writer = MettaWriter(budget=budget)
    cycle_tags = {cycle_set_key(r.cycles) for r in results}
    notes = [
        "Values are SURVEY-WEIGHTED; the weight variable used is recorded per record.",
        "Moments are computed ON THE MARKER'S DECLARED SCALE (RefScale). A z-score must "
        "be computed on the same scale as the reference it is measured against.",
        f"Cells with fewer than {min_cell_n} unweighted observations emit NOTHING "
        "(a suppressed cell is reported on stderr, never emitted with a caveat).",
        f"Restricted to participants aged >= {min_age:g} (adult reference; Schema A has "
        "no field for this bound, so it is recorded here).",
        "No design-based standard errors: correct NHANES SEs need Taylor-series "
        "linearization with the strata/PSU variables, so RefUnweightedN is provided "
        "instead and sum(w) is noted in the comment above each record.",
        "Cystatin C, if present, comes from stored surplus sera whose eligible sample is "
        "NOT nationally representative — read those records as subsample descriptive "
        "statistics.",
    ]
    # provenance_header returns a LIST of lines; MettaWriter.comment takes text, so the
    # lines are joined rather than passed as a list (a list would be str()'d into a
    # Python repr inside the ';;' comment).
    writer.comment("\n".join(provenance_header(
        title="NHANES survey-weighted reference distributions (Schema A)",
        generator=generator,
        inputs=list(inputs),
        notes=list(notes) + list(extra_notes),
    )))
    writer.blank()

    summary = {"emitted": 0, "suppressed": [], "scale_dropped": 0, "markers": []}

    for result in results:
        spec = result.spec
        tag = CYCLES[result.cycles[0]].tag if len(result.cycles) == 1 else \
            cycle_set_key(result.cycles).replace("-", "_")
        cycles_text = cycle_set_key(result.cycles)
        files_text = ";".join(spec.files[c] for c in result.cycles)
        lot = assay_lot_for(spec, result.cycles)

        writer.rule(f"{spec.symbol} — {spec.variable} ({spec.unit}), {cycles_text}, "
                    f"scale {spec.scale}, weight {result.weight.variable}")
        writer.comment(f"registry confidence: {spec.confidence} — {spec.evidence}")
        if spec.notes:
            writer.comment(f"note: {spec.notes}")
        writer.blank()

        emitted_here = 0
        for sex in ("Male", "Female"):
            for band, _lo, _hi in AGE_BANDS:
                cell = result.cells.get((sex, band))
                if cell is None:
                    summary["suppressed"].append((spec.symbol, sex, band, "no observations"))
                    continue
                transformed, keep = apply_scale(cell.values, spec.scale)
                dropped = int(len(cell.values) - int(keep.sum()))
                result.scale_dropped += dropped
                summary["scale_dropped"] += dropped
                weights = np.asarray(cell.weights, dtype="float64")[keep]
                moments = weighted_moments(transformed[keep], weights)
                if moments is None:
                    summary["suppressed"].append(
                        (spec.symbol, sex, band, "no observation survived the scale transform"))
                    continue
                reason = suppressed_reason(n=moments.n, min_n=min_cell_n)
                if reason:
                    summary["suppressed"].append((spec.symbol, sex, band, reason))
                    continue
                if not (moments.sd > 0.0):
                    summary["suppressed"].append(
                        (spec.symbol, sex, band, "zero spread — SD of 0 cannot standardize"))
                    continue

                design = None
                if cell.design_available and len(cell.strata) == len(cell.values):
                    strata = np.asarray(cell.strata, dtype=object)[keep]
                    psu = np.asarray(cell.psu, dtype=object)[keep]
                    design = design_se_of_weighted_mean(
                        transformed[keep], weights, strata, psu
                    )

                rid = record_id(spec, sex, band, tag)
                writer.comment(
                    f"n={moments.n}  sum(w)={moments.sum_w:,.0f}  "
                    f"sd_estimator={moments.sd_estimator}"
                    + (f"  dropped_nonpositive={dropped}" if dropped else "")
                    + (f"  design_se={design.se:.6g} df={design.degrees_of_freedom}"
                       f" singleton_strata={design.singleton_strata}" if design else
                       "  design_se=unavailable (SDMVSTRA/SDMVPSU absent)")
                )
                writer.atom(f"(: {rid} ReferenceDistribution)")
                writer.atom(f"(RefMarker         {rid} {spec.symbol})")
                writer.atom(f"(RefSex            {rid} {sex})")
                writer.atom(f"(RefAgeBand        {rid} {band})")
                writer.atom(f"(RefScale          {rid} {spec.scale})")
                writer.atom(f"(RefMean           {rid} {num(moments.mean)})")
                writer.atom(f"(RefSD             {rid} {num(moments.sd)})")
                writer.atom(f"(RefUnweightedN    {rid} {num(moments.n, places=0)})")
                if design is not None:

                    writer.atom(f"(RefDesignSE       {rid} {num(design.se)})")

                    writer.atom(f"(RefDesignDF       {rid} {num(design.degrees_of_freedom, places=0)})")

                writer.atom(f"(RefUnit           {rid} {mstr(spec.unit)})")
                writer.atom(f"(RefWeightVariable {rid} {mstr(result.weight.variable)})")
                writer.atom(f"(RefSourceVariable {rid} {mstr(spec.variable)})")
                writer.atom(f"(RefSourceFiles    {rid} {mstr(files_text)})")
                writer.atom(f"(RefSourceCycles   {rid} {mstr(cycles_text)})")
                writer.atom(f"(RefAssayLot       {rid} {mstr(lot)})")
                writer.atom(f"(RefProvenance     {rid} NHANES_Microdata)")
                writer.blank()
                emitted_here += 1
                summary["emitted"] += 1

        if emitted_here == 0:
            writer.comment("no cell survived suppression for this marker — nothing emitted")
            writer.blank()
        summary["markers"].append((spec.symbol, emitted_here, result.rows_joined))

    for symbol, sex, band, reason in summary["suppressed"]:
        log(f"  suppressed {symbol} {sex} {band}: {reason}")
    for symbol, count, joined in summary["markers"]:
        log(f"  {symbol}: {count} cell(s) emitted from {joined:,} joined rows")
    log(f"  {summary['emitted']} record(s), {writer.atom_count} atom(s) "
        f"(budget {budget}), {len(summary['suppressed'])} cell(s) suppressed")
    return writer, summary


# ════════════════════════════════════════════════════════════════════════════
# 7. Registry / manifest reporting
# ════════════════════════════════════════════════════════════════════════════

MANIFEST_COLUMNS = ("cycle", "file", "url", "analyte", "variable", "unit",
                    "weight", "scale", "assay_lot", "confidence", "notes")


def manifest_rows(registry: Sequence[MarkerSpec]) -> list[tuple[str, ...]]:
    """One row per (cycle, file, analyte), plus the demographics files.

    The confidence column is the point of the file: it ships in the repo so that an
    unverified CDC name is visible to a reader who never runs --show-registry.
    """
    rows: list[tuple[str, ...]] = []
    for cycle in DEFAULT_CYCLES:
        spec = CYCLES[cycle]
        mec = WEIGHT_TABLE[("WTMEC", "1999-2002")].variable
        rows.append((
            cycle, spec.demo_file, file_url(spec.first_year, spec.demo_file),
            "(demographics)",
            ";".join(DEMOGRAPHIC_VARS) + ";" + ";".join(DESIGN_VARS) + " (design vars optional)",
            "", mec, "", "",
            spec.confidence,
            f"age topcoded at {spec.age_topcode:g}; {mec} is read from here for the "
            f"MEC-examined analytes. {spec.evidence}",
        ))
    for marker in registry:
        for cycle in marker.cycles():
            weight = WEIGHT_TABLE.get((marker.weight_family, "1999-2002"))
            weight_name = weight.variable if weight else f"{marker.weight_family}? (unregistered)"
            if weight and weight.source == "LAB":
                weight_name += " (read from this file, not DEMO)"
            note = marker.notes or ""
            if marker.emit:
                note = (note + " | " if note else "") + f"declared: {marker.declared_in}"
            rows.append((
                cycle, marker.files[cycle],
                file_url(CYCLES[cycle].first_year, marker.files[cycle]),
                marker.symbol, marker.variable, marker.unit, weight_name,
                marker.scale, marker.assay_lots.get(cycle, ""), marker.confidence,
                note.replace("\n", " "),
            ))
    return rows


def render_manifest(registry: Sequence[MarkerSpec]) -> str:
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for row in manifest_rows(registry):
        lines.append("\t".join(re.sub(r"\s+", " ", str(c)).strip() for c in row))
    return "\n".join(lines) + "\n"


def show_registry(registry: Sequence[MarkerSpec], declared: dict[str, str]) -> None:
    print("NHANES reference-distribution registry")
    print("  EVERY file name, variable name and assay description below is an "
          "UNVERIFIED CLAIM about")
    print("  CDC's published data: the CDC hosts were unreachable from the environment "
          "this was")
    print("  written in. Check a name with --inspect FILE; correct one with "
          "--registry override.json.")
    print()
    for cycle in DEFAULT_CYCLES:
        spec = CYCLES[cycle]
        print(f"cycle {cycle}: demographics {spec.demo_file}, age topcode "
              f"{spec.age_topcode:g} [{spec.confidence}]")
    print()
    print("survey weights (weight family x pooled cycle set -> variable):")
    for (family, key), spec in sorted(WEIGHT_TABLE.items()):
        print(f"  {family:<6} {key:<11} {spec.variable:<10} from {spec.source:<5} "
              f"[{spec.confidence}]  {spec.evidence}")
    print()
    for marker in registry:
        where = declared.get(marker.symbol)
        status = "EMIT" if marker.emit else "not emitted"
        print(f"{marker.symbol}  [{marker.confidence}]  ({status})")
        print(f"  variable   {marker.variable}    unit {marker.unit}    "
              f"scale {marker.scale}    weight family {marker.weight_family}")
        for cycle in marker.cycles():
            print(f"  {cycle}  file {marker.files[cycle]:<9} "
                  f"assay lot {marker.assay_lots.get(cycle, '(none)')}")
            print(f"            url  {file_url(CYCLES[cycle].first_year, marker.files[cycle])}")
        print(f"  symbol     {'DECLARED at ' + where if where else 'NOT DECLARED in any .metta file'}")
        print(f"  evidence   {marker.evidence}")
        if marker.notes:
            print(f"  note       {marker.notes}")
        print()


# ════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="NHANES blood analytes -> survey-weighted reference distributions "
                    "(Schema A MeTTa atoms)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="No NHANES microdata and no NHANES-derived number is committed to this "
               "repo; the default output goes to build/ (gitignored). "
               "See data/nhanes/README.md and data/nhanes/MANIFEST.tsv.",
    )
    ap.add_argument("--demo", action="append", type=Path, default=[],
                    help="NHANES demographics file (repeatable, one per cycle)")
    ap.add_argument("--lab", action="append", type=Path, default=[],
                    help="NHANES lab file (repeatable). Files are matched to the "
                         "registry by basename, so order does not matter")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"directory searched for any file not passed explicitly "
                         f"(default {DEFAULT_DATA_DIR})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"MeTTa output path (default {DEFAULT_OUTPUT}; under build/ "
                         f"because NHANES-derived numbers are never committed)")
    ap.add_argument("--cycles", default=",".join(DEFAULT_CYCLES),
                    help=f"comma-separated cycles to pool (default "
                         f"{','.join(DEFAULT_CYCLES)}; known: {','.join(CYCLES)})")
    ap.add_argument("--markers", default=None,
                    help="comma-separated marker symbols to emit (default: every "
                         "marker with a declared symbol)")
    ap.add_argument("--min-age", type=float, default=DEFAULT_MIN_AGE,
                    help=f"exclude participants below this age (default "
                         f"{DEFAULT_MIN_AGE:g}: an adult reference distribution must "
                         f"not mix children into the under-50 band)")
    ap.add_argument("--manifest", action="store_true",
                    help="print the download manifest as TSV and exit "
                         "(this is what data/nhanes/MANIFEST.tsv contains)")
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
        sys.stdout.write(render_manifest(registry))
        return 0
    if args.show_registry:
        show_registry(registry, declared)
        return 0
    if args.inspect:
        return run_inspect(args.inspect)

    if not declared:
        raise SystemExit(
            f"no Biomarker declarations found under {args.repo_root} — this ETL refuses "
            f"to emit atoms it cannot check against the KB. Pass --repo-root."
        )

    selected = [c.strip() for c in args.cycles.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CYCLES]
    if unknown:
        raise SystemExit(f"unknown cycle(s) {unknown}; known: {sorted(CYCLES)}")

    wanted = None
    if args.markers:
        asked = [m.strip() for m in args.markers.split(",") if m.strip()]
        wanted = {m.lower() for m in asked}
        known = {m.symbol.lower() for m in registry}
        missing = sorted(m for m in asked if m.lower() not in known)
        if missing:
            raise SystemExit(
                f"unknown marker(s) {missing}; registered: "
                f"{sorted(m.symbol for m in registry)}"
            )

    chosen: list[MarkerSpec] = []
    for marker in registry:
        if wanted is not None and marker.symbol.lower() not in wanted:
            continue
        if not marker.emit:
            log(f"  {marker.symbol}: {marker.notes or 'not emitted'} "
                f"(see data/nhanes/MANIFEST.tsv)")
            continue
        if marker.symbol not in declared:
            raise SystemExit(
                f"{marker.symbol} has no (Inheritance {marker.symbol} ...Biomarker) "
                f"declaration in any .metta file under {args.repo_root}. Refusing to "
                f"emit an atom naming an undeclared symbol: the hand-written MeTTa "
                f"layer must declare it first."
            )
        chosen.append(marker)
    if not chosen:
        raise SystemExit("no marker selected has a declared symbol — nothing to emit")

    log(f"NHANES reference ETL — cycles {cycle_set_key(selected)}, markers "
        f"{', '.join(m.symbol for m in chosen)}")
    supplied = index_inputs(list(args.demo) + list(args.lab))
    results, inputs = collect(
        chosen, selected, supplied, args.data_dir, min_age=args.min_age, log=log
    )
    writer, _summary = emit(
        results, inputs=inputs, min_cell_n=args.min_cell_n, min_age=args.min_age,
        budget=args.atom_budget, generator=f"{Path(__file__).name}",
        extra_notes=[
            f"Declared-symbol check passed against {args.repo_root}: "
            + ", ".join(f"{m.symbol} ({declared[m.symbol]})" for m in chosen),
        ],
        log=log,
    )
    atoms = writer.write(args.output)
    log(f"wrote {args.output}  ({atoms} atoms)")
    return 0


#: The failures a user can actually fix: a wrong name, a missing file, an unsupported
#: pooling request. Each carries a message that names the fix, so they are reported as
#: an error line rather than as a Python traceback — the message is the whole point of
#: raising loudly, and a traceback buries it.
_USER_FIXABLE = (
    MissingInput, MissingColumns, AssayDiscontinuity, WeightUnavailable,
    AtomBudgetExceeded, ValueError,
)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except _USER_FIXABLE as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
