#!/usr/bin/env python3
"""DrugAge ETL — convert the DrugAge dataset CSV into MeTTa atoms.

Distilled from DrugAge_EDA.ipynb: only the parsing/emission logic is kept, the
exploratory analysis cells are dropped. Each CSV row becomes a
``DrugAgeRow_<idx>`` Experiment individual carrying its intervention, species,
strain, sex, parsed dosage, age-at-initiation, treatment duration,
lifespan/weight changes (+significance) and provenance (PubMed id, ITP flag).

Two spots where the notebook had drifted from the ontology are corrected here
(authorities: logical_predicates.metta and measurement_types.metta):

  * row-to-type membership is emitted as ``(InstanceOf row Experiment)`` — the
    notebook used ``Inheritance``, which logical_predicates.metta reserves for
    taxonomic links and which the LLM few-shot queries never use for rows.
  * the ``%`` dosage unit is emitted as the declared Unit atom ``percent``.

Every other DrugAge unit string is already a valid symbol and is passed through
unchanged (matching the existing drugage_etl_short.metta sample).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ════════════════════════════ dosage parsing ════════════════════════════
@dataclass
class ParsedDose:
    raw: str
    kept: bool
    keep_reason: str                # simple | medium | frequency | reject_*
    value: Optional[float] = None
    unit: Optional[str] = None
    medium: Optional[str] = None    # food | water | diet | medium
    frequency: Optional[str] = None # e.g. 2_per_week, every_5_days


UNITS = [
    "mg/kg/week",
    "mg/kg/day",
    "mg/kg",
    "ug/mL",
    "ug/L",
    "mg/mL",
    "mg/L",
    "ug/g",
    "g/kg",
    "g/L",
    "ppm/day",
    "ppm",
    "mM",
    "uM",
    "nM",
    "pM",
    "mg",
    "ug",
    "IU",
    "M",
    "%",
]
UNIT_PATTERN = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

STOPWORDS = {"on", "in", "at", "the", "of", "and", "with", "for", "to"}

# Map raw dosage units onto the canonical Unit atoms declared in
# measurement_types.metta. Only "%" needs remapping; the rest already match (or
# are passed through as valid symbols).
UNIT_ATOMS = {"%": "percent"}


def unit_atom(unit: str) -> str:
    return UNIT_ATOMS.get(unit, unit)


def normalize_dose_text(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"\([^)]*\)", "", x).strip()
    x = x.replace("μ", "u").replace("µ", "u")
    x = x.replace(" ", " ").replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x)
    x = x.replace("mg/l", "mg/L").replace("ug/ml", "ug/mL").replace("mg/ml", "mg/mL")
    x = x.replace("µg/ml", "ug/mL").replace("μg/ml", "ug/mL")
    x = x.replace("times per week", "times a week")
    return x


def parse_frequency(text: str) -> Optional[str]:
    t = text.lower()

    m = re.search(r"(\d+(?:\.\d+)?)\s*x\s*/\s*(day|week|month)\b", t)
    if m:
        return f"{m.group(1)}_per_{m.group(2)}"

    m = re.search(r"(\d+(?:\.\d+)?)\s*times a\s*(day|week|month)\b", t)
    if m:
        return f"{m.group(1)}_per_{m.group(2)}"

    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s+(day|days|week|weeks|month|months)\b", t)
    if m:
        unit = m.group(2).rstrip("s")
        return f"every_{m.group(1)}_{unit}s"

    if "alternate days" in t or "every other day" in t:
        return "every_2_days"

    return None


def parse_medium(text: str) -> Optional[str]:
    t = text.lower()
    for medium in ["food", "water", "diet", "medium"]:
        if re.search(rf"\b{medium}\b", t):
            return medium
    return None


def has_reject_pattern(text: str) -> Optional[str]:
    t = text.lower()

    if " plus " in t:
        return "reject_combination"
    if "," in t and re.search(r"\d", t) and re.search(r"\b(month|months|day|days|week|weeks|death)\b", t):
        return "reject_multiphase"
    if re.search(r"\b\d+(?:\.\d+)?x10\^-?\d+\b", t):
        return "reject_ratio_like"
    if re.search(r"/\s*(food|diet|water|medium)\b", t):
        return "reject_ratio_like"
    if "bodyweight" in t:
        return "reject_bodyweight"
    if any(route in t for route in [
        "intraperitoneal",
        "oral gavage",
        "subcutaneous injection",
        "injection",
        "orally",
        " oral ",
    ]):
        return "reject_route"
    return None


def cleanup_remainder(text: str, value: float, unit: str, medium: Optional[str], frequency: Optional[str]) -> str:
    rem = text

    rem = re.sub(rf"\b{re.escape(str(value))}\s*{re.escape(unit)}\b", " ", rem, flags=re.IGNORECASE)

    if medium:
        rem = re.sub(rf"\b{medium}\b", " ", rem, flags=re.IGNORECASE)

    rem = re.sub(r"(\d+(?:\.\d+)?)\s*x\s*/\s*(day|week|month)\b", " ", rem, flags=re.IGNORECASE)
    rem = re.sub(r"(\d+(?:\.\d+)?)\s*times a\s*(day|week|month)\b", " ", rem, flags=re.IGNORECASE)
    rem = re.sub(r"every\s+(\d+(?:\.\d+)?)\s+(day|days|week|weeks|month|months)\b", " ", rem, flags=re.IGNORECASE)
    rem = re.sub(r"\b(alternate days|every other day)\b", " ", rem, flags=re.IGNORECASE)

    rem = re.sub(r"[\s,;()]+", " ", rem).strip().lower()

    toks = [tok for tok in rem.split() if tok not in STOPWORDS]
    return " ".join(toks)


def parse_dose(raw_value) -> ParsedDose:
    raw = normalize_dose_text(raw_value)
    if not raw:
        return ParsedDose(raw=raw, kept=False, keep_reason="reject_empty")

    reject_reason = has_reject_pattern(raw)
    if reject_reason:
        return ParsedDose(raw=raw, kept=False, keep_reason=reject_reason)

    m = re.fullmatch(
        rf"\s*(\d+(?:\.\d+)?)\s*({UNIT_PATTERN})(?:\s+(food|water|diet|medium))?\s*",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        value = float(m.group(1))
        unit = m.group(2)
        medium = m.group(3)
        return ParsedDose(
            raw=raw,
            kept=True,
            keep_reason="medium" if medium else "simple",
            value=value,
            unit=unit,
            medium=medium,
        )

    m = re.search(rf"\b(\d+(?:\.\d+)?)\s*({UNIT_PATTERN})(?![A-Za-z/])", raw, flags=re.IGNORECASE)
    if not m:
        return ParsedDose(raw=raw, kept=False, keep_reason="reject_no_amount_unit")

    value = float(m.group(1))
    unit = m.group(2)
    medium = parse_medium(raw)
    frequency = parse_frequency(raw)

    remainder = cleanup_remainder(raw, value, unit, medium, frequency)
    if remainder:
        return ParsedDose(
            raw=raw,
            kept=False,
            keep_reason="reject_partial",
            value=value,
            unit=unit,
            medium=medium,
            frequency=frequency,
        )

    if frequency:
        return ParsedDose(
            raw=raw,
            kept=True,
            keep_reason="frequency" if not medium else "medium_frequency",
            value=value,
            unit=unit,
            medium=medium,
            frequency=frequency,
        )

    if medium:
        return ParsedDose(
            raw=raw,
            kept=True,
            keep_reason="medium",
            value=value,
            unit=unit,
            medium=medium,
        )

    return ParsedDose(
        raw=raw,
        kept=True,
        keep_reason="simple",
        value=value,
        unit=unit,
    )


# ════════════════════════════ time parsing ══════════════════════════════
@dataclass
class ParsedTime:
    raw: str
    kept: bool
    keep_reason: str          # simple | milestone | event | reject_*
    value: Optional[float] = None
    unit: Optional[str] = None
    milestone: Optional[str] = None


TIME_UNITS = {
    "day": "Days",
    "days": "Days",
    "week": "Weeks",
    "weeks": "Weeks",
    "month": "Months",
    "months": "Months",
}


def normalize_time_text(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = x.replace(" ", " ").replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x)
    x = x.lower()

    # typo normalization
    x = x.replace("unti death", "until death")
    x = x.replace("26 month", "26 months")

    return x


def parse_age_at_initiation(raw_value) -> ParsedTime:
    raw = normalize_time_text(raw_value)
    if not raw or raw == "nan":
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_empty")

    # explicit milestone whitelist
    if raw == "after weaning":
        return ParsedTime(
            raw=raw,
            kept=True,
            keep_reason="milestone",
            milestone="AfterWeaning",
        )

    # reject non-atomic cases
    if " or " in raw:
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_disjunction")
    if re.search(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?", raw):
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_range")
    if "," in raw:
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_multivalue")

    # exact age in months only
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(month|months)", raw)
    if m:
        return ParsedTime(
            raw=raw,
            kept=True,
            keep_reason="simple",
            value=float(m.group(1)),
            unit="Months",
        )

    return ParsedTime(raw=raw, kept=False, keep_reason="reject_unparsed_age")


def parse_treatment_duration(raw_value) -> ParsedTime:
    raw = normalize_time_text(raw_value)
    if not raw or raw == "nan":
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_empty")

    # explicit event whitelist
    if raw == "until death":
        return ParsedTime(
            raw=raw,
            kept=True,
            keep_reason="event",
            milestone="UntilDeath",
        )

    # reject non-atomic cases
    if " or " in raw:
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_disjunction")
    if re.search(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?", raw):
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_range")
    if "," in raw:
        return ParsedTime(raw=raw, kept=False, keep_reason="reject_multivalue")

    # exact duration
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months)", raw)
    if m:
        return ParsedTime(
            raw=raw,
            kept=True,
            keep_reason="simple",
            value=float(m.group(1)),
            unit=TIME_UNITS[m.group(2)],
        )

    return ParsedTime(raw=raw, kept=False, keep_reason="reject_unparsed_duration")


def parse_significance(x) -> Optional[str]:
    if x is None:
        return "Unreported"
    t = str(x).strip().upper()
    if not t or t == "NAN":
        return "Unreported"
    if t == "S":
        return "Significant"
    if t == "NS":
        return "NotSignificant"
    return "Unreported"


# ════════════════════════════ MeTTa emission ════════════════════════════
def is_missing(x) -> bool:
    return pd.isna(x) or str(x).strip() == ""


def sanitize_symbol(x: str) -> str:
    s = str(x).strip()
    s = s.replace("μ", "u").replace("µ", "u")
    s = s.replace("%", "Percent")
    s = s.replace("+", "Plus")
    s = s.replace("/", "_")
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if not s:
        s = "Unknown"
    if re.match(r"^\d", s):
        s = f"N_{s}"
    return s


def metta_time_expr(parsed: ParsedTime) -> Optional[str]:
    if not parsed.kept:
        return None
    if parsed.milestone:
        return parsed.milestone
    return f"(Measure {parsed.value} {parsed.unit})"


def metta_dose_atoms(row_id: str, parsed: ParsedDose) -> list[str]:
    if not parsed.kept:
        return []

    atoms = []
    measure = f"(Measure {parsed.value} {unit_atom(parsed.unit)})"

    # medium-bearing dosages are concentrations/exposure in carrier
    if parsed.medium:
        atoms.append(f"(Concentration {row_id} {measure})")
        atoms.append(f"(DeliveryMedium {row_id} {sanitize_symbol(parsed.medium)})")
    else:
        atoms.append(f"(Dosage {row_id} {measure})")

    if parsed.frequency:
        atoms.append(f"(AdministrationFrequency {row_id} {sanitize_symbol(parsed.frequency)})")

    return atoms


def build_drugage_metta(df: pd.DataFrame, out_path: str) -> None:
    lines = []
    lines.append("; Auto-generated DrugAge ETL")
    lines.append("")

    for idx, row in df.iterrows():
        row_id = f"DrugAgeRow_{idx}"

        compound = sanitize_symbol(row["compound_name"])
        species = sanitize_symbol(row["species"])
        strain_raw = row.get("strain", None)
        gender_raw = row.get("gender", None)
        pubmed_raw = row.get("pubmed_id", None)
        itp_raw = row.get("ITP", None)

        lines.append(f"; row {idx}")
        lines.append(f"(InstanceOf {row_id} Experiment)")
        lines.append(f"(UsesIntervention {row_id} {compound})")
        lines.append(f"(UsesSpecies {row_id} {species})")

        if not is_missing(strain_raw):
            strain = sanitize_symbol(strain_raw)
            lines.append(f"(UsesStrain {row_id} {strain})")

        if not is_missing(gender_raw):
            gender = sanitize_symbol(gender_raw)
            lines.append(f"(HasSex {row_id} {gender})")

        if not is_missing(pubmed_raw):
            try:
                pmid = int(float(pubmed_raw))
                lines.append(f"(ReportedIn {row_id} PMID_{pmid})")
            except Exception:
                lines.append(f"(ReportedIn {row_id} {sanitize_symbol(pubmed_raw)})")

        if not is_missing(itp_raw) and str(itp_raw).strip().lower() == "yes":
            lines.append(f"(IsITPStudy {row_id})")

        # dosage
        dose = parse_dose(row.get("dosage", None))
        lines.extend(metta_dose_atoms(row_id, dose))

        # age at initiation
        age = parse_age_at_initiation(row.get("age_at_initiation", None))
        age_expr = metta_time_expr(age)
        if age_expr is not None:
            lines.append(f"(AgeAtInitiation {row_id} {age_expr})")

        # treatment duration
        dur = parse_treatment_duration(row.get("treatment_duration", None))
        dur_expr = metta_time_expr(dur)
        if dur_expr is not None:
            lines.append(f"(TreatmentDuration {row_id} {dur_expr})")

        # avg lifespan change
        avg_change = row.get("avg_lifespan_change_percent", None)
        if not is_missing(avg_change):
            lines.append(f"(AvgLifespanChangePercent {row_id} {float(avg_change)})")

        avg_sig = parse_significance(row.get("avg_lifespan_significance", None))
        if avg_sig is not None:
            lines.append(f"(AvgLifespanSignificance {row_id} {avg_sig})")

        # max lifespan change
        max_change = row.get("max_lifespan_change_percent", None)
        if not is_missing(max_change):
            lines.append(f"(MaxLifespanChangePercent {row_id} {float(max_change)})")

        max_sig = parse_significance(row.get("max_lifespan_significance", None))
        if max_sig is not None:
            lines.append(f"(MaxLifespanSignificance {row_id} {max_sig})")

        # weight change
        weight_change = row.get("weight_change_percent", None)
        if not is_missing(weight_change):
            lines.append(f"(WeightChangePercent {row_id} {float(weight_change)})")

        weight_sig = parse_significance(row.get("weight_change_significance", None))
        if weight_sig is not None:
            lines.append(f"(WeightChangeSignificance {row_id} {weight_sig})")

        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def generate(csv_path, out_path, limit=None) -> None:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit).copy()
    df["dosage"] = df["dosage"].astype(str)
    build_drugage_metta(df, out_path)
    print(f"Wrote {len(df)} rows → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DrugAge CSV → MeTTa ETL")
    ap.add_argument("--input", default="data/drugage/drugage.csv", type=Path,
                    help="DrugAge dataset CSV")
    ap.add_argument("--output", default="drugage_etl.metta", type=Path,
                    help="MeTTa output path")
    ap.add_argument("--limit", type=int, default=None,
                    help="emit only the first N rows (e.g. --limit 201 reproduces "
                         "the truncated drugage_etl_short.metta sample)")
    args = ap.parse_args()
    generate(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
