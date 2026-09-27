#!/usr/bin/env python3
"""NHANES DEMO + the public-use Linked Mortality File -> data-backed baseline risk.

Emits Schema B ``BaselineRiskRecord`` atoms: survey-weighted absolute risk at a fixed
horizon, by age band x sex, for the outcomes NHANES can actually identify. Output is a
build artifact (default ``build/nhanes_mortality_baseline.metta``), never a committed KB
file: no NHANES-derived number ships in this repo.

WHAT THIS FILE REFUSES TO EMIT, AND WHY IT MATTERS MOST
-------------------------------------------------------
``pln_risk_prediction.metta`` models ``P(CHD in 10y)`` — an *incident coronary heart
disease* event — and reads ``HR 1.07 per year of AgeAccelGrim`` from Lu 2019, which was
fit to **incident CHD** (``grim_age_lu2019_evidence.metta``:
``Lu2019_AgeAccelGrim_CHD``). This ETL emits **no baseline for
``CoronaryHeartDisease``**, and the reason is not caution, it is arithmetic:

  1. NHANES + LMF identifies DEATH, not incidence. The linkage returns
     ``MORTSTAT`` and a leading-cause recode; there is no incident-event
     ascertainment anywhere in NHANES. The questionnaire items that mention heart
     disease (``MCQ160C/D/E``) are lifetime "ever told you had" — *prevalence at
     baseline*, usable as an exclusion and nothing else.
  2. Cause code ``"001"`` is *all* Diseases of heart (I00-I09, I11, I13, I20-I51). It
     includes hypertensive and rheumatic heart disease, cardiomyopathy, arrhythmias
     and **heart failure** — which this repo already models as its own outcome with
     its own hazard ratio (``Lu2019_AgeAccelGrim_CHF``, HR 1.10). Serving "001" as a
     CHD baseline would double-count CHF into CHD. No public-use value isolates
     ischemic/coronary disease.
  3. It is fatal-only. Nonfatal MI and revascularization — most of incident CHD —
     are invisible to it.

So the baseline from "001" is far too small for CHD incidence, *and* the HR it would be
multiplied by was estimated on a different event process. The product is a unit mismatch
in both factors: it is not a biased estimate of CHD risk, it is an estimate of nothing.
``baseline-risk-chd`` therefore stays curated and labelled as curated, and the outcome
symbol is threaded through both the baseline and the hazard lookup so the mismatch can
never be formed (contract D3 / D21).

What is emitted instead, under its own honest name:

  * ``AllCauseMortality``      — weighted Kaplan-Meier. Death from any cause is the
    composite event, so there is no competing event to account for.
  * ``HeartDiseaseMortality``  — weighted Aalen-Johansen cause-specific cumulative
    incidence for ``UCOD_LEADING == "001"``. Deaths from other causes are COMPETING
    EVENTS, not censoring: someone who died of cancer is not still at risk of dying of
    heart disease. Each such record also carries ``BaseNaiveKMRisk`` — the wrong
    ``1 - KM`` figure — so the size of the bias we avoided is visible in the output
    rather than asserted in a comment (contract D15).

OTHER DECISIONS THIS FILE ENCODES
---------------------------------
* TIME BASE AND WEIGHT MUST AGREE (D25). Follow-up is ``PERMTH_EXM``, person-months
  from the MEC *examination*, so the weight must be a MEC weight (``WTMEC2YR``, or
  ``WTMEC4YR`` for the pooled 1999-2002 sample). Pairing exam-based follow-up with an
  interview weight, or interview-based follow-up with a MEC weight, mixes two different
  inclusion probabilities; the pairing is enforced in code, not documented in prose.
* ELIGIBILITY IS A DOMAIN, NOT A FILTER ON THE OUTCOME. ``ELIGSTAT != 1`` records are
  removed from the sample entirely. Reading them as censored survivors (which any
  parser that fills a blank ``MORTSTAT`` with 0 would do) deflates the event rate and
  inflates the denominator at once.
* THE HORIZON IS DERIVED, NOT ASSUMED (D26). Linkage ends 2019-12-31, so a 2015-2016
  cycle cannot support ten years of follow-up. The feasible horizon is read off the
  file as ``max(PERMTH_EXM | MORTSTAT == 0)`` and a horizon the file cannot support is
  REFUSED (``--allow-infeasible-horizon`` downgrades it to a loud flag recorded in the
  emitted header).
* THIN CELLS EMIT NOTHING (D9). Below ``--min-cell-n`` unweighted observations or
  ``--min-events`` events the cell is dropped and reported on stderr. A noisy estimate
  with a caveat attached is worse than no estimate, because inference consumes atoms
  and not caveats.
* NO STANDARD ERRORS (D11). A correct NHANES SE needs Taylor-series linearization with
  the strata/PSU design variables. Rather than ship a wrong SE, each record carries the
  unweighted n, the event count and the estimator name so a consumer can judge it.
* ``DIABETES`` / ``HYPERTEN`` ARE DEATH-CERTIFICATE FLAGS (D16). They are multiple-cause
  entries on the death record, non-missing essentially only among decedents. They are
  NOT baseline comorbidity and conditioning on them would be conditioning on the
  outcome. They are surfaced only in the run summary, named
  ``DiabetesOnDeathCertificate`` / ``HypertensionOnDeathCertificate`` to make that
  impossible to misread, and they become no atoms.
* EVERY NHANES NAME IS AN UNVERIFIED CLAIM (D10). The CDC hosts are unreachable from
  the environment this was written in, so no file or variable name here was checked
  against the source. They live in one registry table with a confidence, printable with
  ``--show-registry``, diagnosable with ``--inspect``, correctable with ``--registry``,
  and a missing variable raises and lists the columns that do exist. A registry entry
  is never consulted as a fallback guess.

Statistics, file reading, suppression and emission all come from ``nhanes_common``;
this module is the mapping layer and the honesty guards, nothing else.

Example:

    python3 nhanes_mortality_etl.py \\
        --demo      data/nhanes/1999-2000/DEMO.XPT \\
        --mortality data/nhanes/1999-2000/NHANES_1999_2000_MORT_2019_PUBLIC.dat \\
        --cycles    1999-2000 \\
        --horizon-months 120 \\
        --output    build/nhanes_mortality_baseline.metta

Dependencies: pandas/numpy (already declared) and ``nhanes_common``. Nothing else.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from nhanes_common import (
    short_band,
    short_sex,
    short_cycles,
    AGE_BANDS,
    DEFAULT_MIN_EVENTS,
    UCOD_HEART_DISEASE,
    UCOD_LEADING_LABELS,
    MettaWriter,
    add_common_args,
    age_band,
    check_symbol,
    load_registry_override,
    max_observed_followup_months,
    mstr,
    num,
    provenance_header,
    read_linked_mortality,
    read_nhanes,
    run_inspect,
    sex_symbol,
    suppressed_reason,
    weighted_aalen_johansen,
    weighted_kaplan_meier,
)

GENERATOR = "nhanes_mortality_etl.py"
DEFAULT_OUTPUT = Path("build/nhanes_mortality_baseline.metta")
DEFAULT_HORIZON_MONTHS = 120

# Schema B fixes the follow-up variable. It is a constant rather than a flag on purpose:
# PERMTH_INT would require an interview weight, and the whole point of D25 is that the
# time base and the weight base cannot be chosen independently.
TIME_VARIABLE = "PERMTH_EXM"


# ════════════════════════════════════════════════════════════════════════════
# 1. The registry — every NHANES name this ETL claims, with its confidence
# ════════════════════════════════════════════════════════════════════════════
#
# Confidence vocabulary:
#   certain — quoted from a document recovered in full (CDC's own read-in program);
#   high    — a long-stable NHANES core name, recalled and consistent across cycles;
#   medium  — recalled name, plausible but unverified against the codebook.
#
# A wrong entry must be loud and user-fixable, never a silently wrong number.


@dataclass(frozen=True)
class RegistryEntry:
    variable: str
    confidence: str
    role: str
    note: str = ""


DEMO_REGISTRY: dict[str, RegistryEntry] = {
    "seqn": RegistryEntry(
        "SEQN", "certain", "respondent sequence number (merge key)",
        "quoted from CDC's linked-mortality read-in program, columns 1-6",
    ),
    "sex": RegistryEntry(
        "RIAGENDR", "high", "sex",
        "1 = Male, 2 = Female; anything else yields no record for that participant",
    ),
    "age": RegistryEntry(
        "RIDAGEYR", "high", "age in years at screening",
        "topcoded at 85 (1999-2006) and at 80 (2007-2008 on); the top band Age_70p is "
        "coarser than either topcode and no mean age is computed inside it (D19)",
    ),
}

# The MEC examination weight, keyed by the number of survey years pooled. CDC publishes
# a 2-year MEC weight per cycle and one special 4-year weight for the pooled 1999-2002
# sample; there is no 6-year or longer MEC weight, so any other span is refused rather
# than guessed at.
MEC_WEIGHT_REGISTRY: dict[int, RegistryEntry] = {
    2: RegistryEntry(
        "WTMEC2YR", "high", "MEC examination weight, single 2-year cycle",
        "pairs with PERMTH_EXM (D25)",
    ),
    4: RegistryEntry(
        "WTMEC4YR", "high", "MEC examination weight, pooled 1999-2002",
        "the only multi-cycle MEC weight CDC publishes; pairs with PERMTH_EXM (D25)",
    ),
}

# Prevalent-CHD exclusion (D26). These are lifetime "ever told you had ..." items:
# PREVALENCE at baseline. They are usable to remove people who already had the disease
# from the at-risk set, and for absolutely nothing else — in particular they are not an
# outcome and not an incident event.
CHD_PREVALENCE_REGISTRY: dict[str, RegistryEntry] = {
    "chd": RegistryEntry(
        "MCQ160C", "medium", "ever told had coronary heart disease",
        "1 = Yes excludes; 2 = No, 7 = Refused, 9 = Don't know, missing -> kept",
    ),
    "angina": RegistryEntry(
        "MCQ160D", "medium", "ever told had angina / angina pectoris",
        "1 = Yes excludes",
    ),
    "mi": RegistryEntry(
        "MCQ160E", "medium", "ever told had a heart attack (myocardial infarction)",
        "1 = Yes excludes",
    ),
}

# Death-certificate multiple-cause flags. Present in the layout, deliberately NOT used
# as covariates and NOT emitted (D16). Kept here so --show-registry can say why.
DEATH_CERTIFICATE_FLAGS: dict[str, RegistryEntry] = {
    "DIABETES": RegistryEntry(
        "DIABETES", "certain", "diabetes listed anywhere on the DEATH CERTIFICATE",
        "not baseline diabetes; non-missing essentially only among decedents, so "
        "conditioning on it would condition on the outcome. Surfaced in the run "
        "summary as DiabetesOnDeathCertificate; emitted as no atom.",
    ),
    "HYPERTEN": RegistryEntry(
        "HYPERTEN", "certain", "hypertension listed anywhere on the DEATH CERTIFICATE",
        "not baseline hypertension; see DIABETES. Surfaced as "
        "HypertensionOnDeathCertificate; emitted as no atom.",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# 2. Outcomes — what this file may and may not estimate
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OutcomeSpec:
    symbol: str
    estimator: str
    cause_code: Optional[str]
    meaning: str
    declared_in: str


OUTCOMES: tuple[OutcomeSpec, ...] = (
    OutcomeSpec(
        symbol="AllCauseMortality",
        estimator="WeightedKaplanMeier",
        cause_code=None,
        meaning="death from any cause (MORTSTAT == 1); the composite event, so no "
                "competing event to account for",
        declared_in="grim_age_lu2019_evidence.metta",
    ),
    OutcomeSpec(
        symbol="HeartDiseaseMortality",
        estimator="WeightedAalenJohansen",
        cause_code=UCOD_HEART_DISEASE,
        meaning="death with UCOD_LEADING == \"001\": ALL Diseases of heart "
                "(I00-I09, I11, I13, I20-I51), fatal only. Broader than coronary heart "
                "disease (it includes heart failure, cardiomyopathy, arrhythmias, "
                "hypertensive and rheumatic heart disease) and narrower than CHD "
                "incidence (nonfatal events are invisible). NOT a CHD baseline.",
        declared_in="the hand-written NHANES baseline layer (new outcome symbol)",
    ),
)

# Outcomes a caller might expect and the reason each is absent. Printed by
# --show-registry and written into the emitted header, so the absence is a stated
# finding rather than an oversight.
REFUSED_OUTCOMES: dict[str, str] = {
    "CoronaryHeartDisease": (
        "NHANES has no incident-CHD ascertainment. UCOD_LEADING \"001\" is all Diseases "
        "of heart, fatal only, and includes heart failure — which this repo models "
        "separately with its own HR. Lu 2019's HR 1.07 was fit to INCIDENT CHD, so "
        "pairing that HR with a fatal-heart-disease baseline is a unit mismatch in both "
        "factors and their product estimates nothing. baseline-risk-chd stays curated "
        "(contract D3)."
    ),
    "CongestiveHeartFailure": (
        "not separable in the public-use file: heart failure is inside cause code "
        "\"001\" and no public-use value isolates it."
    ),
    "Stroke": (
        "cause code \"005\" (Cerebrovascular diseases) identifies stroke DEATH only, and "
        "no outcome symbol in this repo currently consumes a stroke-mortality baseline. "
        "Adding it means declaring StrokeMortality first, not reusing Stroke."
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# 3. Cycle tag -> weight variable and symbol
# ════════════════════════════════════════════════════════════════════════════

_CYCLE_RE = re.compile(r"^(\d{4})-(\d{4})$")


class CycleTagError(RuntimeError):
    """The --cycles tag is not a form whose survey-weight base can be determined."""


class HorizonNotSupported(RuntimeError):
    """The requested horizon exceeds the follow-up the file actually contains."""


class WeightBaseMismatch(RuntimeError):
    """The weight variable does not belong to the same base as the time variable."""


def cycle_span_years(tag: str) -> int:
    """Number of survey years the cycle tag covers, e.g. '1999-2002' -> 4."""
    match = _CYCLE_RE.match(str(tag).strip())
    if not match:
        raise CycleTagError(
            f"--cycles {tag!r} is not of the form YYYY-YYYY (e.g. 1999-2000, 1999-2002). "
            f"The tag determines which MEC weight is correct, so it cannot be guessed."
        )
    start, end = int(match.group(1)), int(match.group(2))
    span = end - start + 1
    if span <= 0:
        raise CycleTagError(f"--cycles {tag!r} ends before it starts")
    return span


def mec_weight_for(tag: str) -> RegistryEntry:
    """The MEC weight registry entry for a cycle tag, refusing spans CDC does not ship."""
    span = cycle_span_years(tag)
    entry = MEC_WEIGHT_REGISTRY.get(span)
    if entry is None:
        raise CycleTagError(
            f"--cycles {tag!r} spans {span} survey years; CDC publishes a MEC weight for "
            f"{sorted(MEC_WEIGHT_REGISTRY)} only (2-year per cycle, plus the special "
            f"4-year 1999-2002 weight). Pooling more cycles needs a weight CDC does not "
            f"provide — construct it deliberately and pass --weight-variable, do not let "
            f"this ETL pick one."
        )
    return entry


def cycle_symbol(tag: str) -> str:
    """'1999-2002' -> '1999_2002', safe inside a MeTTa record identifier."""
    return str(tag).strip().replace("-", "_")


def assert_weight_base(weight_variable: str) -> None:
    """Refuse an interview weight paired with exam-based follow-up (D25)."""
    if not str(weight_variable).upper().startswith("WTMEC"):
        raise WeightBaseMismatch(
            f"weight variable {weight_variable!r} is not a MEC examination weight, but "
            f"follow-up here is {TIME_VARIABLE} — person-months from the MEC EXAM. "
            f"Time-zero and weight base must match: an interview weight carries the "
            f"probability of being interviewed, not of being examined, so the pairing "
            f"weights the wrong denominator (contract D25). Use WTMEC2YR / WTMEC4YR, or "
            f"switch the whole analysis to PERMTH_INT with interview weights — which "
            f"Schema B does not currently cover."
        )


# ════════════════════════════════════════════════════════════════════════════
# 4. Resolving the registry (with --registry overrides)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ResolvedNames:
    seqn: str
    sex: str
    age: str
    weight: str
    weight_confidence: str
    exclusion: dict[str, str] = field(default_factory=dict)


def resolve_names(
    *, cycles: str, override: dict, weight_variable: Optional[str] = None
) -> ResolvedNames:
    """Apply a --registry override on top of the built-in table.

    The override is a flat JSON object keyed by registry key:
    ``{"sex": "RIAGENDR", "age": "RIDAGEYR", "weight": "WTMEC4YR", "chd": "MCQ160C"}``.
    An unknown key is an error rather than a silent no-op — a typo'd override that did
    nothing would be indistinguishable from a correct one that was not needed.
    """
    known = set(DEMO_REGISTRY) | set(CHD_PREVALENCE_REGISTRY) | {"weight"}
    unknown = sorted(set(override) - known)
    if unknown:
        raise ValueError(
            f"--registry contains unknown key(s) {unknown}; known keys are "
            f"{sorted(known)}. Refusing to ignore an override silently."
        )

    weight_entry = mec_weight_for(cycles)
    weight = weight_variable or override.get("weight") or weight_entry.variable
    confidence = weight_entry.confidence
    if weight_variable:
        confidence = "user-supplied (--weight-variable)"
    elif "weight" in override:
        confidence = "user-supplied (--registry)"
    assert_weight_base(weight)

    return ResolvedNames(
        seqn=override.get("seqn", DEMO_REGISTRY["seqn"].variable).upper(),
        sex=override.get("sex", DEMO_REGISTRY["sex"].variable).upper(),
        age=override.get("age", DEMO_REGISTRY["age"].variable).upper(),
        weight=str(weight).upper(),
        weight_confidence=confidence,
        exclusion={
            key: str(override.get(key, entry.variable)).upper()
            for key, entry in CHD_PREVALENCE_REGISTRY.items()
        },
    )


def show_registry(out=sys.stdout) -> int:
    """--show-registry: print every unverified name claim with its confidence."""
    print("NHANES mortality ETL registry — every entry is an UNVERIFIED claim about", file=out)
    print("CDC's published data (the CDC hosts were unreachable when this was written).", file=out)
    print("Correct one with --registry override.json; diagnose with --inspect FILE.", file=out)
    print("", file=out)
    print("DEMO file variables", file=out)
    for key, entry in DEMO_REGISTRY.items():
        print(f"  {key:<10} {entry.variable:<10} [{entry.confidence}]  {entry.role}", file=out)
        if entry.note:
            print(f"             {entry.note}", file=out)
    print("", file=out)
    print(f"Survey weight (paired with {TIME_VARIABLE}; key 'weight')", file=out)
    for span, entry in MEC_WEIGHT_REGISTRY.items():
        print(f"  {span}-year    {entry.variable:<10} [{entry.confidence}]  {entry.role}", file=out)
        print(f"             {entry.note}", file=out)
    print("", file=out)
    print("Prevalent-CHD EXCLUSION variables (questionnaire file, --questionnaire)", file=out)
    print("  lifetime 'ever told' items: PREVALENCE at baseline. Exclusion only —", file=out)
    print("  never an outcome, never an incident event (contract D26).", file=out)
    for key, entry in CHD_PREVALENCE_REGISTRY.items():
        print(f"  {key:<10} {entry.variable:<10} [{entry.confidence}]  {entry.role}", file=out)
        print(f"             {entry.note}", file=out)
    print("", file=out)
    print("Linked-mortality fields deliberately NOT used", file=out)
    for entry in DEATH_CERTIFICATE_FLAGS.values():
        print(f"  {entry.variable:<10} [{entry.confidence}]  {entry.role}", file=out)
        print(f"             {entry.note}", file=out)
    print("", file=out)
    print("Outcomes EMITTED", file=out)
    for spec in OUTCOMES:
        print(f"  {spec.symbol}  [{spec.estimator}]"
              + (f"  cause={spec.cause_code}" if spec.cause_code else ""), file=out)
        print(f"      {spec.meaning}", file=out)
        print(f"      symbol declared in: {spec.declared_in}", file=out)
    print("", file=out)
    print("Outcomes REFUSED", file=out)
    for symbol, reason in REFUSED_OUTCOMES.items():
        print(f"  {symbol}", file=out)
        print(f"      {reason}", file=out)
    return 0


# ════════════════════════════════════════════════════════════════════════════
# 5. Building the analysis sample
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class SampleDiagnostics:
    """Row counts at every step, so a shrinking denominator is never a surprise."""

    demo_rows: int = 0
    mortality_rows: int = 0
    merged_rows: int = 0
    demo_unmatched: int = 0
    ineligible_dropped: int = 0
    missing_time_dropped: int = 0
    nonpositive_weight_dropped: int = 0
    missing_demographics_dropped: int = 0
    prevalent_chd_excluded: int = 0
    exclusion_applied: bool = False
    exclusion_variables: tuple[str, ...] = ()
    deaths_without_cause: int = 0
    cause_counts: dict[str, int] = field(default_factory=dict)
    analysis_rows: int = 0
    diabetes_on_death_certificate: int = 0
    hypertension_on_death_certificate: int = 0
    feasible_horizon_months: float = float("nan")


def concat_cycles(frames: Sequence[pd.DataFrame], *, what: str, seqn: str) -> pd.DataFrame:
    """Stack one file per cycle, refusing a duplicated SEQN.

    Pooling 1999-2002 (the only span with a published 4-year MEC weight) means reading
    two DEMO files and two mortality files. NHANES numbers SEQN sequentially ACROSS
    cycles, so a repeated SEQN never means the same person twice — it means the same
    file was passed twice, or files from the same cycle were mixed up. Either way the
    merge would silently pair the wrong records, so it is refused.
    """
    if len(frames) == 1:
        return frames[0]
    stacked = pd.concat(frames, ignore_index=True)
    duplicated = stacked[seqn][stacked[seqn].duplicated()].tolist()
    if duplicated:
        raise RuntimeError(
            f"{what}: SEQN {sorted(set(duplicated))[:10]} appears in more than one of the "
            f"{len(frames)} files supplied. NHANES numbers SEQN sequentially across "
            f"cycles, so this means the same file was passed twice or two files from the "
            f"same cycle were mixed — refusing rather than merging the wrong records."
        )
    return stacked


def _seqn_key(series: pd.Series) -> pd.Series:
    """SEQN as a nullable integer, so a float XPT column merges with an ASCII int one."""
    values = pd.to_numeric(series, errors="coerce")
    return values.round().astype("Int64")


def build_analysis_frame(
    demo: pd.DataFrame,
    mortality: pd.DataFrame,
    *,
    names: ResolvedNames,
    questionnaire: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, SampleDiagnostics]:
    """Merge DEMO with the LMF, restrict to the linkage-eligible domain, band and sex.

    Every exclusion is counted rather than performed silently: the difference between a
    correct domain restriction and a corrupted denominator is invisible in the output
    numbers but obvious in these counts.
    """
    diag = SampleDiagnostics(demo_rows=len(demo), mortality_rows=len(mortality))

    demo = demo.copy()
    mortality = mortality.copy()
    demo["_SEQN"] = _seqn_key(demo[names.seqn])
    mortality["_SEQN"] = _seqn_key(mortality["SEQN"])

    mortality = mortality.drop(columns=["SEQN"])
    collisions = sorted(set(demo.columns) & set(mortality.columns) - {"_SEQN"})
    if collisions:
        raise RuntimeError(
            f"the DEMO and mortality files share column(s) {collisions}. pandas would "
            f"suffix them on the merge and this ETL would then read the wrong one. "
            f"Supply the plain DEMO file, not an extract that already carries "
            f"linked-mortality fields."
        )
    merged = demo.merge(mortality, on="_SEQN", how="inner", validate="one_to_one")
    diag.merged_rows = len(merged)
    diag.demo_unmatched = len(demo) - len(merged)
    if diag.merged_rows == 0:
        raise RuntimeError(
            "no SEQN is present in both the DEMO file and the mortality file. These are "
            "almost certainly from different cycles — the linked-mortality file is "
            "published per cycle and its SEQN range matches exactly one DEMO file."
        )

    # ELIGSTAT is a DOMAIN restriction. Ineligible participants leave the sample; they
    # are never read as censored survivors (read_linked_mortality already refuses a file
    # where an ineligible record carries a MORTSTAT, which is the parser bug that would
    # cause exactly that).
    eligible = merged["ELIGSTAT"] == 1
    diag.ineligible_dropped = int((~eligible).sum())
    frame = merged.loc[eligible].copy()

    time_present = frame[TIME_VARIABLE].notna()
    diag.missing_time_dropped = int((~time_present).sum())
    frame = frame.loc[time_present].copy()

    weights = pd.to_numeric(frame[names.weight], errors="coerce")
    usable_weight = weights.notna() & (weights > 0)
    diag.nonpositive_weight_dropped = int((~usable_weight).sum())
    frame = frame.loc[usable_weight].copy()
    frame["_WEIGHT"] = pd.to_numeric(frame[names.weight], errors="coerce")

    frame["_SEX"] = frame[names.sex].map(sex_symbol)
    frame["_BAND"] = pd.to_numeric(frame[names.age], errors="coerce").map(age_band)
    classified = frame["_SEX"].notna() & frame["_BAND"].notna()
    diag.missing_demographics_dropped = int((~classified).sum())
    frame = frame.loc[classified].copy()

    if questionnaire is not None:
        diag.exclusion_applied = True
        diag.exclusion_variables = tuple(sorted(names.exclusion.values()))
        quest = questionnaire.copy()
        quest["_SEQN"] = _seqn_key(quest[names.seqn])
        columns = ["_SEQN"] + [v for v in names.exclusion.values()]
        frame = frame.merge(quest[columns].drop_duplicates("_SEQN"), on="_SEQN", how="left")
        prevalent = pd.Series(False, index=frame.index)
        for variable in names.exclusion.values():
            prevalent = prevalent | (pd.to_numeric(frame[variable], errors="coerce") == 1)
        diag.prevalent_chd_excluded = int(prevalent.sum())
        frame = frame.loc[~prevalent].copy()

    # A linked death whose leading cause is missing is still a DEATH. Reading it as
    # censored would silently return it to the risk set; it is carried as an event of
    # unknown cause, which makes it a competing event for the cause-specific estimate
    # and an event for all-cause mortality. Counted so it can never be invisible.
    missing_cause = (frame["MORTSTAT"] == 1) & frame["UCOD_LEADING"].isna()
    diag.deaths_without_cause = int(missing_cause.sum())
    frame["_CAUSE"] = frame["UCOD_LEADING"].where(frame["MORTSTAT"] == 1)
    frame.loc[missing_cause, "_CAUSE"] = "UNKNOWN"
    frame["_EVENT"] = (frame["MORTSTAT"] == 1).astype(int)
    decedents = frame["_EVENT"] == 1
    diag.cause_counts = {
        str(code): int(count)
        for code, count in frame.loc[decedents, "_CAUSE"].value_counts().items()
    }
    # Death-certificate flags, counted on the final analysis set for the summary only.
    # They are diagnostics, never covariates: they are non-missing essentially only among
    # decedents, so conditioning on them would be conditioning on the outcome (D16).
    for column, attribute in (("DIABETES", "diabetes_on_death_certificate"),
                              ("HYPERTEN", "hypertension_on_death_certificate")):
        if column in frame.columns:
            setattr(diag, attribute, int((frame.loc[decedents, column] == 1).sum()))

    diag.analysis_rows = len(frame)
    diag.feasible_horizon_months = max_observed_followup_months(frame, time_col=TIME_VARIABLE)
    return frame, diag


def check_horizon(
    diag: SampleDiagnostics, *, horizon: int, allow_infeasible: bool
) -> Optional[str]:
    """Refuse (or flag) a horizon longer than the file's observed follow-up (D26).

    Returns None when the horizon is supported, or a warning string when it is not and
    ``allow_infeasible`` permits continuing. Raises otherwise.
    """
    feasible = diag.feasible_horizon_months
    if not math.isfinite(feasible):
        raise HorizonNotSupported(
            "no censored record survives the filters, so the file's maximum follow-up "
            f"cannot be derived from max({TIME_VARIABLE} | MORTSTAT == 0). Without it a "
            "fixed-horizon risk cannot be checked for feasibility, and an unfeasible "
            "horizon produces a number driven by extrapolation rather than data."
        )
    if horizon <= feasible:
        return None
    message = (
        f"requested horizon {horizon} months exceeds the longest follow-up this file "
        f"actually contains ({feasible:.0f} months, from max({TIME_VARIABLE}) among "
        f"censored records). Linkage ends 2019-12-31, so cycles from 2011-2012 on carry "
        f"less than ten years of follow-up: at {horizon} months the product-limit "
        f"estimate is extrapolation past the last observation, not an estimate from the "
        f"data (contract D26). Lower --horizon-months to {feasible:.0f} or less, or pass "
        f"--allow-infeasible-horizon to proceed with the limitation recorded in the "
        f"emitted header."
    )
    if allow_infeasible:
        return message
    raise HorizonNotSupported(message)


# ════════════════════════════════════════════════════════════════════════════
# 6. Estimating the cells
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class BaselineCell:
    """One emitted (or suppressed) baseline-risk cell."""

    outcome: OutcomeSpec
    sex: str
    band: str
    horizon: int
    risk: float
    n: int
    events: int
    sum_w: float
    censored_before_horizon: int
    naive_km_risk: Optional[float] = None
    competing_events: Optional[int] = None


@dataclass
class Suppression:
    outcome: str
    sex: str
    band: str
    reason: str


def estimate_cells(
    frame: pd.DataFrame,
    *,
    horizon: int,
    min_cell_n: int,
    min_events: int,
) -> tuple[list[BaselineCell], list[Suppression]]:
    """Estimate every (outcome, sex, band) cell, suppressing the unreliable ones."""
    cells: list[BaselineCell] = []
    suppressed: list[Suppression] = []

    for spec in OUTCOMES:
        for sex in ("Male", "Female"):
            for band, _lo, _hi in AGE_BANDS:
                subset = frame.loc[(frame["_SEX"] == sex) & (frame["_BAND"] == band)]
                if subset.empty:
                    suppressed.append(Suppression(spec.symbol, sex, band, "no observations"))
                    continue

                times = subset[TIME_VARIABLE]
                weights = subset["_WEIGHT"]

                if spec.cause_code is None:
                    estimate = weighted_kaplan_meier(
                        times, subset["_EVENT"], weights, horizon=horizon
                    )
                    if estimate is None:
                        suppressed.append(
                            Suppression(spec.symbol, sex, band, "nothing at risk")
                        )
                        continue
                    reason = suppressed_reason(
                        n=estimate.n, min_n=min_cell_n,
                        events=estimate.events, min_events=min_events,
                    )
                    if reason:
                        suppressed.append(Suppression(spec.symbol, sex, band, reason))
                        continue
                    cells.append(BaselineCell(
                        outcome=spec, sex=sex, band=band, horizon=horizon,
                        risk=estimate.cumulative_incidence, n=estimate.n,
                        events=estimate.events, sum_w=estimate.sum_w,
                        censored_before_horizon=estimate.censored_before_horizon,
                    ))
                    continue

                estimate = weighted_aalen_johansen(
                    times, subset["_CAUSE"], weights,
                    horizon=horizon, cause=spec.cause_code,
                )
                if estimate is None:
                    suppressed.append(Suppression(spec.symbol, sex, band, "nothing at risk"))
                    continue
                reason = suppressed_reason(
                    n=estimate.n, min_n=min_cell_n,
                    events=estimate.events, min_events=min_events,
                )
                if reason:
                    suppressed.append(Suppression(spec.symbol, sex, band, reason))
                    continue
                cells.append(BaselineCell(
                    outcome=spec, sex=sex, band=band, horizon=horizon,
                    risk=estimate.cumulative_incidence, n=estimate.n,
                    events=estimate.events, sum_w=estimate.sum_w,
                    censored_before_horizon=estimate.censored_before_horizon,
                    naive_km_risk=estimate.naive_km_incidence,
                    competing_events=estimate.competing_events,
                ))

    return cells, suppressed


# ════════════════════════════════════════════════════════════════════════════
# 7. Emission — Schema B, exactly
# ════════════════════════════════════════════════════════════════════════════


# Outcome -> short code for the record identifier. Every field atom repeats the
# identifier, so its length is multiplied by ~15 and dominates the file size — and the
# file size is what silently binds (nhanes_common BYTE BUDGET). Nothing is lost by
# shortening it: each record spells out its outcome, sex, band and cycles in its own
# field atoms, and a readable comment sits above every record.
OUTCOME_CODES = {
    "AllCauseMortality": "ACM",
    "HeartDiseaseMortality": "HDM",
}


def record_id(cell: BaselineCell, *, cycles: str, horizon: int) -> str:
    """A compact, unique, stable identifier for one emitted baseline cell.

    The horizon is part of the identifier because it CHANGES BaseRisk: two runs of the
    same cycles at different horizons describe different quantities, and without the
    horizon they would collide on one identifier and silently overwrite each other when
    both files are loaded into the same space.
    """
    try:
        outcome_code = OUTCOME_CODES[cell.outcome.symbol]
    except KeyError:
        raise ValueError(
            f"no short code for outcome {cell.outcome.symbol!r}; add it to OUTCOME_CODES"
        ) from None
    identifier = (
        f"NB_{outcome_code}_{short_sex(cell.sex)}_{short_band(cell.band)}"
        f"_{short_cycles(cycles)}_{int(horizon)}m"
    )
    return check_symbol(identifier, what="baseline record identifier")


def emit(
    cells: Sequence[BaselineCell],
    suppressed: Sequence[Suppression],
    diag: SampleDiagnostics,
    *,
    cycles: str,
    vintage: str,
    horizon: int,
    names: ResolvedNames,
    inputs: Sequence[str],
    budget: int,
    byte_budget: int,
    horizon_warning: Optional[str] = None,
) -> MettaWriter:
    """Build the MeTTa text for the emitted cells (budgets enforced by the writer)."""
    writer = MettaWriter(budget=budget, byte_budget=byte_budget)

    notes = [
        f"Estimand: absolute risk of the named outcome within {horizon} months of the "
        f"MEC examination, by age band x sex.",
        f"Follow-up: {TIME_VARIABLE} (person-months from the MEC exam), paired with the "
        f"MEC weight {names.weight} — time-zero and weight base must agree (D25).",
        f"Linked-mortality vintage: {vintage}. Sample: ELIGSTAT == 1 as a DOMAIN; "
        f"linkage-ineligible participants are removed, never read as censored survivors.",
        "AllCauseMortality uses weighted Kaplan-Meier. HeartDiseaseMortality uses "
        "weighted Aalen-Johansen: deaths from other causes are COMPETING EVENTS, not "
        "censoring. Each cause-specific record also carries BaseNaiveKMRisk, the wrong "
        "1-KM figure, so the size of the avoided bias is visible here (D15).",
        "NO CoronaryHeartDisease BASELINE IS EMITTED. Cause code \"001\" is all Diseases "
        "of heart (includes heart failure, which this repo models separately), it is "
        "fatal-only, and Lu 2019's HR 1.07 was fit to INCIDENT CHD. Pairing that HR "
        "with this baseline is a unit mismatch in both factors, so their product "
        "estimates nothing. baseline-risk-chd stays curated (D3/D21).",
        "DIABETES / HYPERTEN in the mortality file are DEATH-CERTIFICATE multiple-cause "
        "flags, not baseline comorbidity, and are emitted as no atoms (D16).",
        "No standard errors: a correct NHANES SE needs design-based linearization with "
        "the strata/PSU variables. Judge reliability from BaseUnweightedN and "
        "BaseEvents (D11).",
        f"Cells with fewer than the minimum unweighted n or event count emit NOTHING; "
        f"{len(suppressed)} cell(s) were suppressed this run (listed on stderr).",
        f"Sample: {diag.demo_rows} DEMO rows -> {diag.merged_rows} merged -> "
        f"{diag.analysis_rows} in the analysis set. Feasible horizon from this file: "
        f"{diag.feasible_horizon_months:.0f} months.",
    ]
    if diag.exclusion_applied:
        notes.append(
            f"Prevalent-CHD exclusion APPLIED using {', '.join(diag.exclusion_variables)} "
            f"(lifetime 'ever told' items = PREVALENCE at baseline, used as an EXCLUSION "
            f"only): {diag.prevalent_chd_excluded} participant(s) removed from the "
            f"at-risk set (D26)."
        )
    else:
        notes.append(
            "Prevalent-CHD exclusion NOT applied (no --questionnaire supplied): the "
            "at-risk set includes people who already had heart disease at baseline."
        )
    if diag.deaths_without_cause:
        notes.append(
            f"{diag.deaths_without_cause} linked death(s) carry no leading-cause code; "
            f"they are counted as deaths of unknown cause (competing events for the "
            f"cause-specific estimate), never as censored."
        )
    if horizon_warning:
        notes.append("HORIZON LIMITATION (--allow-infeasible-horizon): " + horizon_warning)

    for line in provenance_header(
        title=f"NHANES baseline absolute risk — {cycles}, {horizon}-month horizon",
        generator=GENERATOR,
        inputs=inputs,
        notes=notes,
    ):
        writer.comment(line)
    writer.blank()

    for spec in OUTCOMES:
        writer.comment(f"{spec.symbol} — {spec.meaning}")
        writer.comment(f"  outcome symbol declared in: {spec.declared_in}")
    writer.blank()

    cycles_text = mstr(cycles)
    vintage_text = mstr(vintage)
    weight_text = mstr(names.weight)
    time_text = mstr(TIME_VARIABLE)
    horizon_text = num(horizon, places=0)

    for cell in cells:
        bid = record_id(cell, cycles=cycles, horizon=horizon)
        writer.blank()
        writer.comment(
            f"{cell.outcome.symbol} · {cell.sex} · {cell.band} · n={cell.n} "
            f"events={cell.events} censored-before-horizon={cell.censored_before_horizon}"
        )
        writer.atom(f"(: {bid} BaselineRiskRecord)")
        writer.atom(f"(BaseOutcome {bid} {cell.outcome.symbol})")
        writer.atom(f"(BaseSex {bid} {cell.sex})")
        writer.atom(f"(BaseAgeBand {bid} {cell.band})")
        writer.atom(f"(BaseHorizonMonths {bid} {horizon_text})")
        writer.atom(f"(BaseRisk {bid} {num(cell.risk)})")
        writer.atom(f"(BaseEstimator {bid} {cell.outcome.estimator})")
        writer.atom(f"(BaseUnweightedN {bid} {num(cell.n, places=0)})")
        writer.atom(f"(BaseEvents {bid} {num(cell.events, places=0)})")
        writer.atom(f"(BaseWeightVariable {bid} {weight_text})")
        writer.atom(f"(BaseTimeVariable {bid} {time_text})")
        writer.atom(f"(BaseSourceCycles {bid} {cycles_text})")
        writer.atom(f"(BaseLinkageVintage {bid} {vintage_text})")
        writer.atom(f"(BaseProvenance {bid} NHANES_Microdata)")
        if cell.naive_km_risk is not None:
            writer.atom(f"(BaseNaiveKMRisk {bid} {num(cell.naive_km_risk)})")
        if cell.competing_events is not None:
            writer.atom(f"(BaseCompetingEvents {bid} {num(cell.competing_events, places=0)})")
        if cell.outcome.cause_code is not None:
            writer.atom(f"(BaseCauseCode {bid} {mstr(cell.outcome.cause_code)})")

    return writer


# ════════════════════════════════════════════════════════════════════════════
# 8. Run summary
# ════════════════════════════════════════════════════════════════════════════


def report(
    cells: Sequence[BaselineCell],
    suppressed: Sequence[Suppression],
    diag: SampleDiagnostics,
    *,
    names: ResolvedNames,
    horizon: int,
    out=sys.stderr,
) -> None:
    """Everything a reader needs to judge the run, on stderr so stdout stays the path."""
    print("", file=out)
    print("Sample construction", file=out)
    print(f"  DEMO rows                          {diag.demo_rows}", file=out)
    print(f"  mortality rows                     {diag.mortality_rows}", file=out)
    print(f"  merged on SEQN                     {diag.merged_rows}"
          f"  (DEMO rows with no mortality record: {diag.demo_unmatched})", file=out)
    print(f"  dropped ELIGSTAT != 1 (domain)     {diag.ineligible_dropped}", file=out)
    print(f"  {('dropped missing ' + TIME_VARIABLE):<34} {diag.missing_time_dropped}", file=out)
    print(f"  {('dropped ' + names.weight + ' <= 0 or missing'):<34} "
          f"{diag.nonpositive_weight_dropped}", file=out)
    print(f"  dropped unclassifiable age/sex     {diag.missing_demographics_dropped}", file=out)
    if diag.exclusion_applied:
        print(f"  excluded prevalent CHD             {diag.prevalent_chd_excluded}"
              f"  ({', '.join(diag.exclusion_variables)} == 1)", file=out)
    else:
        print("  prevalent-CHD exclusion            NOT applied (no --questionnaire)", file=out)
    print(f"  analysis set                       {diag.analysis_rows}", file=out)
    if diag.deaths_without_cause:
        print(f"  deaths with no cause code          {diag.deaths_without_cause}"
              f"  (carried as deaths of unknown cause, never censored)", file=out)
    print(f"  feasible horizon (derived)         "
          f"{diag.feasible_horizon_months:.0f} months; requested {horizon}", file=out)

    # The cause distribution is the cheapest check that the record layout was read with
    # the right vintage: the 2011 layout shifts UCOD_LEADING by one column and yields
    # plausible small integers rather than an error, but the resulting mix of codes looks
    # nothing like a population's leading causes.
    if diag.cause_counts:
        print("", file=out)
        print("Underlying cause among decedents in the analysis set "
              "(eyeball this: a vintage mis-read shows up here)", file=out)
        for code, count in sorted(diag.cause_counts.items()):
            label = UCOD_LEADING_LABELS.get(code, "NO CAUSE CODE ON THE LINKED RECORD")
            marker = "  <- cause of interest" if code == UCOD_HEART_DISEASE else ""
            print(f"  {code:<8} {count:>6}  {label}{marker}", file=out)

    print("", file=out)
    print("Death-certificate flags (D16 — NOT baseline comorbidity, emitted as no atoms)",
          file=out)
    print(f"  DiabetesOnDeathCertificate         {diag.diabetes_on_death_certificate}", file=out)
    print(f"  HypertensionOnDeathCertificate     {diag.hypertension_on_death_certificate}",
          file=out)

    print("", file=out)
    print(f"Emitted cells ({len(cells)})", file=out)
    for cell in cells:
        line = (f"  {cell.outcome.symbol:<22} {cell.sex:<7} {cell.band:<10} "
                f"risk={cell.risk:.4f}  n={cell.n:<5} events={cell.events}")
        print(line, file=out)

    competing = [c for c in cells if c.naive_km_risk is not None]
    if competing:
        print("", file=out)
        print("Competing-risks correction (D15) — naive 1-KM treats other deaths as", file=out)
        print("censoring and therefore OVERSTATES the cause-specific risk:", file=out)
        print(f"  {'outcome/cell':<42} {'naive 1-KM':>11} {'Aalen-Joh':>11} "
              f"{'overstated by':>14}", file=out)
        for cell in competing:
            label = f"{cell.outcome.symbol} {cell.sex} {cell.band}"
            delta = cell.naive_km_risk - cell.risk
            relative = (delta / cell.risk * 100.0) if cell.risk > 0 else float("nan")
            print(f"  {label:<42} {cell.naive_km_risk:>11.4f} {cell.risk:>11.4f} "
                  f"{delta:>+9.4f} ({relative:+.1f}%)", file=out)

    if suppressed:
        print("", file=out)
        print(f"Suppressed cells ({len(suppressed)}) — emitted NOTHING (D9)", file=out)
        for item in suppressed:
            print(f"  {item.outcome:<22} {item.sex:<7} {item.band:<10} {item.reason}", file=out)


# ════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NHANES DEMO + Linked Mortality File -> Schema B baseline-risk atoms",
        epilog="No NHANES-derived number is committed to this repo; output goes to build/.",
    )
    parser.add_argument("--demo", type=Path, nargs="+", default=None, metavar="FILE",
                        help="NHANES demographics file(s) (DEMO.XPT, or a CSV extract); "
                             "pass one per cycle when pooling, e.g. 1999-2002 needs two")
    parser.add_argument("--mortality", type=Path, nargs="+", default=None, metavar="FILE",
                        help="public-use linked mortality file(s), fixed-width .dat, one "
                             "per cycle matching --demo")
    parser.add_argument("--questionnaire", type=Path, nargs="+", default=None,
                        metavar="FILE",
                        help="optional medical-conditions file(s) (MCQ.XPT) used ONLY to "
                             "exclude prevalent CHD from the at-risk set (D26)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"MeTTa output path (default {DEFAULT_OUTPUT})")
    parser.add_argument("--cycles", default=None,
                        help="survey cycle tag YYYY-YYYY recorded in the provenance and "
                             "used to pick the MEC weight (e.g. 1999-2000, 1999-2002)")
    parser.add_argument("--vintage", default="2019",
                        help="linked-mortality file vintage / record layout (default 2019)")
    parser.add_argument("--horizon-months", type=int, default=DEFAULT_HORIZON_MONTHS,
                        help=f"risk horizon in months (default {DEFAULT_HORIZON_MONTHS}); "
                             f"refused if longer than the file's observed follow-up")
    parser.add_argument("--allow-infeasible-horizon", action="store_true",
                        help="downgrade the horizon-feasibility refusal to a loud flag "
                             "recorded in the emitted header (D26)")
    parser.add_argument("--weight-variable", default=None,
                        help="override the MEC weight chosen from --cycles; must still "
                             f"be a MEC weight, since follow-up is {TIME_VARIABLE} (D25)")
    parser.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS,
                        help=f"suppress a cell with fewer events (default "
                             f"{DEFAULT_MIN_EVENTS})")
    return add_common_args(parser)


def run(args: argparse.Namespace) -> int:
    override = load_registry_override(args.registry)
    names = resolve_names(
        cycles=args.cycles, override=override, weight_variable=args.weight_variable
    )

    demo = concat_cycles(
        [read_nhanes(p, require=[names.seqn, names.sex, names.age, names.weight])
         for p in args.demo],
        what="DEMO", seqn=names.seqn,
    )
    mortality = concat_cycles(
        [read_linked_mortality(p, vintage=args.vintage) for p in args.mortality],
        what="linked mortality", seqn="SEQN",
    )
    questionnaire = None
    if args.questionnaire:
        questionnaire = concat_cycles(
            [read_nhanes(p, require=[names.seqn] + sorted(names.exclusion.values()))
             for p in args.questionnaire],
            what="questionnaire", seqn=names.seqn,
        )

    frame, diag = build_analysis_frame(
        demo, mortality, names=names, questionnaire=questionnaire
    )
    # A large unmatched fraction is the signature of a DEMO file paired with another
    # cycle's mortality file — which merges to a small but perfectly plausible sample.
    if diag.demo_unmatched > 0.1 * diag.demo_rows:
        print(
            f"WARNING: {diag.demo_unmatched} of {diag.demo_rows} DEMO participants have "
            f"no record in the mortality file(s). The linked-mortality file is published "
            f"per cycle and should cover every participant, so check that --demo and "
            f"--mortality are from the same cycle(s).",
            file=sys.stderr,
        )

    horizon_warning = check_horizon(
        diag, horizon=args.horizon_months, allow_infeasible=args.allow_infeasible_horizon
    )
    if horizon_warning:
        print(f"WARNING: {horizon_warning}", file=sys.stderr)

    cells, suppressed = estimate_cells(
        frame,
        horizon=args.horizon_months,
        min_cell_n=args.min_cell_n,
        min_events=args.min_events,
    )
    report(cells, suppressed, diag, names=names, horizon=args.horizon_months)

    if not cells:
        print(
            "\nNo cell survived suppression, so nothing was written. This is the "
            "intended behaviour, not a failure: every candidate cell was too thin to "
            "support a reliable estimate (D9).",
            file=sys.stderr,
        )
        return 1

    inputs = [f"{path}  (DEMO, weight {names.weight} [{names.weight_confidence}])"
              for path in args.demo]
    inputs += [f"{path}  (public-use Linked Mortality File, vintage {args.vintage})"
               for path in args.mortality]
    inputs += [
        f"{path}  (prevalent-CHD EXCLUSION only: "
        f"{', '.join(sorted(names.exclusion.values()))})"
        for path in (args.questionnaire or ())
    ]

    writer = emit(
        cells, suppressed, diag,
        cycles=args.cycles, vintage=args.vintage, horizon=args.horizon_months,
        names=names, inputs=inputs, budget=args.atom_budget,
        byte_budget=args.byte_budget, horizon_warning=horizon_warning,
    )
    size = writer.byte_size
    atoms = writer.write(args.output)
    print(f"\nWrote {len(cells)} baseline-risk record(s), {atoms} atoms "
          f"(budget {args.atom_budget}), {size:,} bytes "
          f"(budget {args.byte_budget:,}) -> {args.output}", file=sys.stderr)
    print(args.output)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_registry:
        return show_registry()
    if args.inspect is not None:
        return run_inspect(args.inspect)

    missing = [flag for flag, value in
               (("--demo", args.demo), ("--mortality", args.mortality),
                ("--cycles", args.cycles)) if value is None]
    if missing:
        parser.error(
            f"missing required argument(s) {', '.join(missing)}.\n"
            f"  --demo and --mortality have no default because NHANES microdata is not "
            f"bundled with this repo (see data/nhanes/README.md).\n"
            f"  --cycles has no default ON PURPOSE: it selects the survey weight and is "
            f"written into the emitted provenance, so an unstated cycle would become an "
            f"unverifiable claim about which sample the numbers describe. NHANES files "
            f"do not carry their cycle in a readable field, and inferring it from a file "
            f"name would be exactly the kind of guess this integration refuses.\n"
            f"  Example:\n"
            f"    python3 {GENERATOR} --demo data/nhanes/DEMO.XPT \\\n"
            f"      --mortality data/nhanes/NHANES_2001_2002_MORT_2019_PUBLIC.dat \\\n"
            f"      --cycles 2001-2002 --output {DEFAULT_OUTPUT}"
        )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
