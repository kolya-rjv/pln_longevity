"""Turn a LinAge2 service response into KB atoms — honestly, and for this request only.

The LinAge2 service (Rejuve/LinAge2-Python, `POST /predict`) returns a biological
age, the BA-CA delta, and one years-contribution per model input, each flagged
measured or imputed. This module is the typed surface that lets a caller hand that
response to the knowledge base as part of a `patient`, in the same request-scoped,
never-on-disk way `patient_builder.py` handles markers (the user asked for patients
to stay stateless, and they do: nothing here writes anything).

What it produces, per request:

    (LinAgeDelta <P> <years>)                           the BA-CA delta, verbatim
    (MeasuredZ <P> LinAgeAccel <z>)                     years / linage-sd-to-years,
                                                        rendered by patient_builder
    (LinAgeContribution <P> <Input> <years> Measured)   a lab the caller supplied
    (LinAgeContribution <P> <Input> <years> Imputed)    a lab the service filled in

The feature vocabulary is read from `linage2_core.metta` — the `(ModelInput <F>
LinAge2)` facts and the NHANES code recorded beside each — so this module cannot
drift from the ontology: a response naming a code the KB does not declare is a
422, never an atom nothing reads.

Three things this module will NOT do, each for a reason recorded here:

* It does not derive a lab's direction from a contribution's sign. LinAge2's
  per-feature weight is a sex-specific projection whose sign is not the lab's
  clinical direction (CRP: +4.4 months/SD in the female model, -0.2 in the male
  model). Whether a lab is HIGH comes from the patient's own `markers` z; the
  MeTTa layer joins the two and credits a cause only under that witness.
* It does not turn imputed contributions into measurements. They are the
  reference cohort's values, not the patient's; they are rendered with the
  `Imputed` flag so the decomposition can total them separately and the API can
  tell the caller to validate them with real labs.
* It does not guess the sex model that produced the response. The response does
  not say, so `sex` stays the caller's declaration.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import ONTOLOGY_DIR
from core.patient_builder import PatientSpecError, _as_number

LINAGE2_CORE = ONTOLOGY_DIR / "linage2_core.metta"
LINAGE2_RULES = ONTOLOGY_DIR / "pln_linage2.metta"

#: `(ModelInput <Symbol> LinAge2)  (Inheritance <Symbol> <Type>)  ;; CODE — description`
#: The typing fact between is optional to the regex, mandatory to the ontology.
_INPUT_RE = re.compile(
    r'^\(ModelInput\s+([A-Za-z][A-Za-z0-9_]*)\s+LinAge2\)\s*'
    r'(?:\(Inheritance\s+\1\s+\S+\)\s*)?'
    r';;\s*([A-Za-z0-9_]+)\s+[—-]+\s*(.*?)\s*$',
    re.M,
)
#: `(MeasuresBiomarker <Symbol> <Biomarker>)`
_READOUT_RE = re.compile(r"^\(MeasuresBiomarker\s+(\S+)\s+(\S+)\)", re.M)
#: `(= (linage-sd-to-years) 8.66)`
_SD_RE = re.compile(r"\(=\s*\(linage-sd-to-years\)\s*([\d.]+)\s*\)")

#: Fong et al. 2025: even participants with heart failure, a recent cancer diagnosis
#: or on long-term dialysis had BA deltas "of at most 35 years". Beyond that a delta
#: is almost certainly a unit or input mistake; well beyond it we refuse.
PLAUSIBLE_ABS_DELTA_YEARS = 35.0
MAX_ABS_DELTA_YEARS = 50.0
MAX_ABS_CONTRIBUTION_YEARS = 40.0
#: biological_age - chronological_age must be the delta the service reports.
DELTA_TOLERANCE_YEARS = 0.05
#: The response's chronological age must be this patient's age.
AGE_TOLERANCE_YEARS = 0.5


@dataclass(frozen=True)
class LinAge2Feature:
    code: str            #: the NHANES variable code the service uses ("LBXCRP")
    symbol: str          #: the KB symbol ("CRP")
    description: str
    reads_out: Optional[str]   #: the KB biomarker this input measures, if any


@lru_cache(maxsize=1)
def feature_catalog() -> dict[str, LinAge2Feature]:
    """code -> feature, read off linage2_core.metta. The ontology is the authority."""
    text = LINAGE2_CORE.read_text(encoding="utf-8")
    readouts = dict(_READOUT_RE.findall(text))
    catalog: dict[str, LinAge2Feature] = {}
    for symbol, code, description in _INPUT_RE.findall(text):
        catalog[code] = LinAge2Feature(code, symbol, description.strip(), readouts.get(symbol))
    return catalog


def symbol_catalog() -> dict[str, LinAge2Feature]:
    return {f.symbol: f for f in feature_catalog().values()}


def linage_sd_to_years(default: float = 8.66) -> float:
    """The `linage-sd-to-years` knob, read off pln_linage2.metta so the z the API
    renders and the status the engine assigns can never disagree."""
    try:
        match = _SD_RE.search(LINAGE2_RULES.read_text(encoding="utf-8"))
    except OSError:
        return default
    return float(match.group(1)) if match else default


@dataclass
class LinAge2Contribution:
    code: str
    symbol: str
    years: float
    imputed: bool
    reads_out: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "feature": self.code,
            "symbol": self.symbol,
            "years": round(self.years, 6),
            "imputed": self.imputed,
            "reads_out": self.reads_out,
        }


@dataclass
class BuiltLinAge2:
    delta_years: float
    biological_age: float
    chronological_age: float
    z: float
    status: str
    sd_to_years: float
    contributions: list[LinAge2Contribution] = field(default_factory=list)
    atoms: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def measured(self) -> list[LinAge2Contribution]:
        return [c for c in self.contributions if not c.imputed]

    @property
    def imputed(self) -> list[LinAge2Contribution]:
        return [c for c in self.contributions if c.imputed]

    def as_dict(self) -> dict:
        return {
            "delta_years": round(self.delta_years, 6),
            "biological_age": round(self.biological_age, 6),
            "chronological_age": round(self.chronological_age, 6),
            "z": round(self.z, 6),
            "status": self.status,
            "sd_to_years": self.sd_to_years,
            "z_formula": f"z = delta_years / {self.sd_to_years:g}   [linage-sd-to-years]",
            "measured_count": len(self.measured),
            "imputed_count": len(self.imputed),
            "attributed_measured_years": round(sum(c.years for c in self.measured), 6),
            "attributed_imputed_years": round(sum(c.years for c in self.imputed), 6),
            "contributions": [c.as_dict() for c in self.contributions],
            "warnings": list(self.warnings),
        }


def _unwrap(payload: dict) -> dict:
    """Accept the `/predict` response body, or its flattened `data`/`metadata`."""
    if not isinstance(payload, dict):
        raise PatientSpecError(
            "invalid_linage2", "`linage2` must be the LinAge2 /predict response object."
        )
    flat: dict = {}
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        flat.update({k: v for k, v in meta.items() if v is not None})
    flat.update({k: v for k, v in payload.items() if k != "metadata" and v is not None})
    return flat


def _number(flat: dict, key: str, code: str) -> float:
    if key not in flat:
        raise PatientSpecError(code, f"`linage2.{key}` is required.", missing=key)
    return _as_number(flat[key], code=code, what=f"`linage2.{key}`", field=key)


def build_linage2(
    payload: dict,
    patient_id: str,
    *,
    age: Optional[float],
    elevated_threshold: float = 1.0,
    sd_to_years: Optional[float] = None,
) -> BuiltLinAge2:
    """Validate a LinAge2 response for `patient_id` and render it as atoms.

    Raises PatientSpecError (-> 422) for anything that would otherwise become a
    confident wrong atom: an unknown feature code, a duplicated feature, a delta
    that is not biological_age - chronological_age, a response computed for a
    different age than the patient's, an implausible magnitude.
    """
    sd = sd_to_years if sd_to_years is not None else linage_sd_to_years()
    flat = _unwrap(payload)
    catalog = feature_catalog()
    warnings: list[str] = []

    bio = _number(flat, "biological_age", "invalid_linage2")
    chrono = _number(flat, "chronological_age", "invalid_linage2")
    delta = _number(flat, "delta_ba_ca", "invalid_linage2")

    if abs((bio - chrono) - delta) > DELTA_TOLERANCE_YEARS:
        raise PatientSpecError(
            "linage2_inconsistent",
            f"`linage2.delta_ba_ca` ({delta:g}) is not biological_age - chronological_age "
            f"({bio:g} - {chrono:g} = {bio - chrono:g}). The three fields come from one "
            f"LinAge2 run; a mismatch means the payload was edited or mis-assembled.",
            biological_age=bio, chronological_age=chrono, delta_ba_ca=delta,
        )
    if abs(delta) > MAX_ABS_DELTA_YEARS:
        raise PatientSpecError(
            "implausible_linage2_delta",
            f"A LinAge2 delta of {delta:g} years is outside anything the published "
            f"model produces (Fong et al. 2025 report deltas of at most "
            f"{PLAUSIBLE_ABS_DELTA_YEARS:g} years even in seriously ill participants). "
            f"Check the payload's units and inputs.",
            delta_ba_ca=delta,
        )
    if abs(delta) > PLAUSIBLE_ABS_DELTA_YEARS:
        warnings.append(
            f"LinAge2 delta of {delta:+.1f} years exceeds the largest delta Fong et al. "
            f"2025 report on NHANES ({PLAUSIBLE_ABS_DELTA_YEARS:g} years, in participants "
            f"with heart failure, recent cancer or on dialysis). It is carried as sent; "
            f"verify the inputs the service was given."
        )
    if age is not None and abs(age - chrono) > AGE_TOLERANCE_YEARS:
        raise PatientSpecError(
            "linage2_age_mismatch",
            f"The LinAge2 response was computed for chronological age {chrono:g}, but "
            f"this patient's `age` is {age:g}. A clock result for a different age is "
            f"not this patient's; re-run LinAge2 or fix `age`.",
            patient_age=age, linage2_chronological_age=chrono,
        )

    raw_contribs = flat.get("feature_contributions")
    if not isinstance(raw_contribs, list) or not raw_contribs:
        raise PatientSpecError(
            "invalid_linage2",
            "`linage2.feature_contributions` must be a non-empty list of "
            "{feature, contribution_years, is_imputed} objects.",
        )
    imputed_list = flat.get("imputed_features") or []
    if not isinstance(imputed_list, list):
        raise PatientSpecError("invalid_linage2", "`linage2.imputed_features` must be a list.")
    imputed_set = {str(x) for x in imputed_list}

    seen: set[str] = set()
    contributions: list[LinAge2Contribution] = []
    unknown: list[str] = []
    for entry in raw_contribs:
        if not isinstance(entry, dict) or "feature" not in entry:
            raise PatientSpecError(
                "invalid_linage2",
                "Each `feature_contributions` entry needs `feature`, "
                "`contribution_years` and `is_imputed`.",
                received=entry,
            )
        code = str(entry["feature"])
        spec = catalog.get(code)
        if spec is None:
            unknown.append(code)
            continue
        if code in seen:
            raise PatientSpecError(
                "duplicate_linage2_feature",
                f"Feature '{code}' appears twice in `feature_contributions`; two "
                f"contributions for one input would double-count in every total.",
                feature=code,
            )
        seen.add(code)
        years = _as_number(
            entry.get("contribution_years"), code="invalid_linage2",
            what=f"`contribution_years` for {code}", feature=code,
        )
        if abs(years) > MAX_ABS_CONTRIBUTION_YEARS:
            raise PatientSpecError(
                "implausible_linage2_contribution",
                f"Feature '{code}' contributes {years:g} years — larger than the "
                f"whole plausible range of the clock. Check units.",
                feature=code, contribution_years=years,
            )
        flagged = entry.get("is_imputed")
        if isinstance(flagged, bool):
            imputed = flagged
        else:
            imputed = code in imputed_set
        if flagged is not None and not isinstance(flagged, bool):
            raise PatientSpecError(
                "invalid_linage2", f"`is_imputed` for {code} must be true or false.",
                feature=code, received=flagged,
            )
        if imputed != (code in imputed_set) and imputed_set:
            warnings.append(
                f"LinAge2 response disagrees with itself about '{code}': `is_imputed` "
                f"says {imputed}, `imputed_features` says {code in imputed_set}. The "
                f"per-feature flag was used."
            )
        contributions.append(
            LinAge2Contribution(code, spec.symbol, years, imputed, spec.reads_out)
        )

    if unknown:
        raise PatientSpecError(
            "unknown_linage2_feature",
            f"Feature code(s) {', '.join(sorted(unknown))} are not LinAge2 inputs this "
            f"knowledge base declares (linage2_core.metta). Supported: "
            f"{', '.join(sorted(catalog))}.",
            features=sorted(unknown), supported=sorted(catalog),
        )
    missing = sorted(set(catalog) - seen)
    if missing:
        warnings.append(
            f"The LinAge2 response omits {len(missing)} of the {len(catalog)} model "
            f"inputs ({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}). The "
            f"decomposition is over what was sent; the age-term residual will absorb "
            f"the rest."
        )

    # Sort for the reader: measured first, then by |years| descending — the order the
    # LinAge2 UI uses. The atoms themselves are unordered facts.
    contributions.sort(key=lambda c: (c.imputed, -abs(c.years), c.code))

    z = delta / sd
    if not math.isfinite(z):
        raise PatientSpecError("invalid_linage2", "The LinAge2 delta is not finite.")
    status = "Elevated" if z > elevated_threshold else (
        "Low" if z < -elevated_threshold else "Normal"
    )

    n_imputed = sum(1 for c in contributions if c.imputed)
    if n_imputed:
        imputed_years = sum(c.years for c in contributions if c.imputed)
        warnings.append(
            f"{n_imputed} of {len(contributions)} LinAge2 inputs were not measured but "
            f"IMPUTED (the median of a same-sex, same-age reference cohort, a value "
            f"derived from one, or a questionnaire default), contributing "
            f"{imputed_years:+.2f} years in total. Imputed inputs are "
            f"carried under the `Imputed` flag: the engine totals them separately and "
            f"never credits a cause to one. Re-test them with real labs."
        )

    # The clock's own (MeasuredZ <P> LinAgeAccel z) is NOT rendered here: build_patient
    # carries LinAgeAccel as an ordinary resolved marker, so it is rendered once, by
    # the same loop as every other marker, and can never appear twice.
    lines = [f"(LinAgeDelta {patient_id} {delta:.6f})"]
    for c in sorted(contributions, key=lambda c: c.symbol):
        lines.append(
            f"(LinAgeContribution {patient_id} {c.symbol} {c.years:.6f} "
            f"{'Imputed' if c.imputed else 'Measured'})"
        )

    return BuiltLinAge2(
        delta_years=delta, biological_age=bio, chronological_age=chrono,
        z=z, status=status, sd_to_years=sd,
        contributions=contributions, atoms="\n".join(lines), warnings=warnings,
    )


def feature_listing() -> list[dict]:
    """For GET /linage2/features: every input the KB declares, with its NHANES
    code, description, and the KB biomarker it reads out (if any)."""
    out = []
    for f in feature_catalog().values():
        out.append({
            "feature": f.code,
            "symbol": f.symbol,
            "description": f.description,
            "reads_out": f.reads_out,
            "cause_attribution": (
                "credited when the patient's own z for the biomarker is Elevated"
                if f.reads_out and f.reads_out != "CurrentTobaccoExposure" else
                "credited when the patient is recorded as a CurrentSmoker"
                if f.reads_out == "CurrentTobaccoExposure" else
                "none — carried as years, no bridge in the KB yet"
            ),
        })
    return sorted(out, key=lambda e: (e["reads_out"] is None, e["feature"]))
