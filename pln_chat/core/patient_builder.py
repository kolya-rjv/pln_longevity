"""Turn a caller's patient into KB atoms — safely.

The whole personalized stack (risk, decomposition, counterfactuals, ranking,
supplements) already works for a patient it has never seen: it needs only
`PatientAge`, `PatientSex` and some `MeasuredZ` atoms, and `run_query` already
injects caller-supplied atoms into the same hyperon space. The 2026-09-18
evaluation's first recommendation — "caller-supplied patients" — is therefore
not an inference problem. It is a typed surface, a z-scoring policy, and
sanitisation.

The sanitisation is not optional. Three failures were reproduced against the
existing `extra_atoms` path:

* **Atom injection.** A patient id of
  ``Evil) (= (grimage-weight $m) 9.9) (PatientAge Zzz 10`` interpolated into a
  naive f-string redefines a calibration knob. Measured effect:
  `(decompose-grimage &self Patient001)` — a DIFFERENT, pre-existing patient —
  started reporting component weights of 9.9 and a residual of -96.8.
* **Id collision.** Submitting a second `Patient001` does not shadow the first;
  hyperon unions the facts and every accessor goes non-deterministic.
  `(predict-risk-patient &self Patient001)` then took 63 seconds and returned
  512 RiskPrediction atoms, many internally inconsistent — a point estimate from
  one age/sex branch beside a confidence interval from another.
* **Silent nonsense.** `rank-interventions-for-patient` returns a full,
  plausible ranking for a patient that does not exist, because the personal
  term degenerates to zero and the population ranking survives.

So: ids are generated or strictly validated and namespaced, every atom is built
from validated fields rather than interpolated text, markers are checked against
what the KB can actually reason about, and a marker the KB cannot use is
reported rather than silently carried.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ── The vocabulary a patient may use ─────────────────────────────────────────
# Kept as data so the API can publish it (GET /patients/markers) and so a
# rejected marker can say what the alternatives are.

@dataclass(frozen=True)
class MarkerSpec:
    """One biomarker a caller may submit, and what the KB does with it."""
    name: str
    role: str                 # clock | grimage_component | blood_marker
    reaches: str              # plain-English: what inference it feeds
    #: None when only a z-score is accepted (see RAW_VALUE_POLICY).
    reference: Optional["Reference"] = None

    @property
    def accepts_raw(self) -> bool:
        return self.reference is not None


def _normalise_unit(unit: str) -> str:
    """Fold spelling differences that do not change what a unit means.

    Case, internal spaces and the micro sign are noise: `mg/dL`, `mg/dl` and
    `MG / DL` are one unit, and a caller who types `ug/mL` means `µg/mL`.
    """
    folded = re.sub(r"\s+", "", (unit or "").strip().lower())
    return folded.replace("µ", "u").replace("μ", "u")


#: Unit spellings accepted for a marker expressed in YEARS of age acceleration.
YEARS_UNITS = {"years", "year", "yrs", "yr", "y"}


@dataclass(frozen=True)
class Reference:
    """A curated reference distribution for converting a raw value to a z.

    CURATED PRIOR, in the sense mechanistic_bridges.metta uses the term: a
    documented expert estimate, quarantined in one place, to be replaced by a
    calibrated age/sex-stratified table (NHANES is the obvious source) when one
    exists. There is no such table anywhere in this repository today — verified
    by grep — so a caller who needs exact standardisation should send `z`
    directly, which is always accepted. Every z DERIVED here is flagged
    `derived: true` with the formula that produced it, so a consumer can tell a
    measured standardisation from ours.
    """
    unit: str
    mean: float
    sd: float
    #: Markers whose population distribution is strongly right-skewed are
    #: standardised on the log scale, which is how they are analysed clinically.
    log_scale: bool = False
    source: str = ""
    #: (alias, scale, offset) triples mapping a raw value expressed in `alias`
    #: onto `unit`:  value_in_unit = value * scale + offset. A tuple rather than
    #: a dict so the dataclass stays frozen AND hashable. The reference unit
    #: itself is always accepted and needs no entry.
    #:
    #: Every entry is an exact or published clinical conversion, never a guess:
    #: an approximate unit conversion would reintroduce the silent error this
    #: table exists to remove.
    conversions: tuple[tuple[str, float, float], ...] = ()

    def accepted_units(self) -> list[str]:
        return [self.unit] + [alias for alias, _scale, _offset in self.conversions]

    def to_reference_unit(
        self, value: float, unit: Optional[str]
    ) -> tuple[float, Optional[str]]:
        """Express `value` in this reference's own unit.

        Returns `(converted_value, note)`; `note` is None when nothing was
        converted. A caller who sends no unit is read as already using the
        reference unit, which is what GET /patients/markers publishes as
        `raw_unit`.

        Raises ValueError for a unit this reference cannot convert. That is
        deliberate: until this method existed the caller's `unit` was read and
        then ignored, so `CRP {value: 0.4, unit: "mg/dL"}` — 4 mg/L, an ordinary
        result — was standardised as 0.4 mg/L and came back z = -1.61 "Low"
        instead of z = +0.69. A 422 naming the accepted units is recoverable;
        a confidently wrong patient is not.
        """
        if unit is None or not str(unit).strip():
            return value, None
        key = _normalise_unit(unit)
        if key == _normalise_unit(self.unit):
            return value, None
        for alias, scale, offset in self.conversions:
            if key == _normalise_unit(alias):
                converted = value * scale + offset
                operation = f"x {scale:g}"
                if offset:
                    operation += f" + {offset:g}"
                note = (
                    f"{value:g} {unit} = {converted:g} {self.unit}   [{operation}]"
                )
                return converted, note
        raise ValueError(
            f"unit '{unit}' cannot be standardised for this marker. Accepted: "
            f"{', '.join(self.accepted_units())}. Send `z` instead if you "
            f"already have standard deviations."
        )

    def to_z(self, value: float) -> tuple[float, str]:
        if self.log_scale:
            if value <= 0:
                raise ValueError("a log-scaled marker needs a positive value")
            z = (math.log(value) - math.log(self.mean)) / self.sd
            formula = (
                f"z = (ln(value) - ln({self.mean:g})) / {self.sd:g}   [log scale]"
            )
        else:
            z = (value - self.mean) / self.sd
            formula = f"z = (value - {self.mean:g}) / {self.sd:g}"
        return z, formula


#: The epigenetic clock the risk model reads, in SDs of age acceleration.
#: `grimaccel-sd-to-years` in pln_risk_prediction.metta is 4.2, so a caller who
#: has "GrimAge is 6.7 years older than chronological age" can send years and
#: have it divided by that same knob — see `YEARS_PER_SD_MARKERS`.
MARKERS: dict[str, MarkerSpec] = {
    "AgeAccelGrim": MarkerSpec(
        "AgeAccelGrim", "clock",
        "the predictor of the 10-year CHD risk model; REQUIRED for any risk number",
    ),
    "HorvathAgeAccel": MarkerSpec(
        "HorvathAgeAccel", "clock",
        "first-generation clock; carried for the discordance story, no downstream edge",
    ),
    "DNAmPAI1": MarkerSpec(
        "DNAmPAI1", "grimage_component",
        "senescence readout (SASP -> PAI-1); drives the abductive diagnosis and "
        "the senolytic counterfactual",
    ),
    "DNAmGDF15": MarkerSpec(
        "DNAmGDF15", "grimage_component",
        "mitochondrial-stress and inflammation readout; reached by three causes",
    ),
    "DNAmPACKYRS": MarkerSpec(
        "DNAmPACKYRS", "grimage_component",
        "DNAm surrogate of smoking pack-years; a GrimAge component, and since "
        "lifestyle_evidence.metta the one marker the smoking lever acts on. Send "
        "it for a current or former smoker and "
        "(counterfactual-patient &self <Patient> SmokingCessation) and "
        "(project-risk-patient &self <Patient> SmokingCessation) both return a "
        "real number instead of zero",
    ),
    "DNAmADM": MarkerSpec("DNAmADM", "grimage_component", "GrimAge component; no cause edge yet"),
    "DNAmB2M": MarkerSpec("DNAmB2M", "grimage_component", "GrimAge component; no cause edge yet"),
    "DNAmCystatinC": MarkerSpec("DNAmCystatinC", "grimage_component", "GrimAge component; no cause edge yet"),
    "DNAmLeptin": MarkerSpec("DNAmLeptin", "grimage_component", "GrimAge component; no cause edge yet"),
    "DNAmTIMP1": MarkerSpec("DNAmTIMP1", "grimage_component", "GrimAge component; no cause edge yet"),
    "CRP": MarkerSpec(
        "CRP", "blood_marker",
        "chronic inflammation readout; drives the abductive diagnosis and the "
        "omega-3 recommendation",
        Reference(
            unit="mg/L", mean=2.0, sd=1.0, log_scale=True,
            source="coarse prior: adult hs-CRP is strongly right-skewed with a "
                   "geometric mean around 2 mg/L; standardised on the log scale. "
                   "Replace with an age/sex-stratified cohort table.",
            # Exact rescalings: 1 mg/dL = 10 mg/L, and 1 ug/mL = 1 mg/L.
            conversions=(("mg/dL", 10.0, 0.0), ("ug/mL", 1.0, 0.0),
                         ("mcg/mL", 1.0, 0.0)),
        ),
    ),
    "FastingGlucose": MarkerSpec(
        "FastingGlucose", "blood_marker",
        "insulin-resistance readout; drives the metabolic axis and the AMPK "
        "activators (metformin, berberine)",
        Reference(
            unit="mg/dL", mean=95.0, sd=12.0,
            source="coarse prior for non-diabetic adults. Replace with an "
                   "age/sex-stratified cohort table.",
            # 1 mmol/L glucose = 18.0182 mg/dL (molar mass 180.156 g/mol).
            conversions=(("mmol/L", 18.0182, 0.0),),
        ),
    ),
    "HbA1c": MarkerSpec(
        "HbA1c", "blood_marker",
        "insulin-resistance readout; drives the metabolic axis",
        Reference(
            unit="%", mean=5.5, sd=0.5,
            source="coarse prior for non-diabetic adults. Replace with an "
                   "age/sex-stratified cohort table.",
            # NGSP % = 0.09148 x IFCC mmol/mol + 2.152 — the NGSP/IFCC master
            # equation, not an approximation of it.
            conversions=(("mmol/mol", 0.09148, 2.152),),
        ),
    ),
}

#: Markers whose raw value is naturally expressed in YEARS of age acceleration.
#: Converted with the risk layer's own `grimaccel-sd-to-years` knob so the two
#: never drift apart.
YEARS_PER_SD_MARKERS = {"AgeAccelGrim", "HorvathAgeAccel"}

SEXES = {"Male", "Female"}
SMOKING = {"NeverSmoker", "FormerSmoker", "CurrentSmoker"}

#: A caller-supplied id is namespaced so it can never collide with a curated
#: patient, whatever the caller sends.
ID_PREFIX = "Caller_"
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,47}$")

Z_LIMIT = 12.0          # |z| beyond this is a unit mistake, not a patient
AGE_RANGE = (0.0, 130.0)
MAX_MARKERS = 40


class PatientSpecError(ValueError):
    """A caller's patient could not be turned into atoms."""

    def __init__(self, code: str, message: str, **extra) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


@dataclass
class ResolvedMarker:
    name: str
    z: float
    derived: bool = False
    raw_value: Optional[float] = None
    unit: Optional[str] = None
    formula: Optional[str] = None
    status: str = "Normal"        # Elevated | Normal | Low
    note: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "marker": self.name,
            "z": round(self.z, 6),
            "derived": self.derived,
            "raw_value": self.raw_value,
            "unit": self.unit,
            "formula": self.formula,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class BuiltPatient:
    """A validated patient, ready to inject."""
    patient_id: str
    atoms: str
    markers: list[ResolvedMarker] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    age: Optional[float] = None
    sex: Optional[str] = None
    smoking: Optional[str] = None

    @property
    def can_predict_risk(self) -> bool:
        has_clock = any(m.name == "AgeAccelGrim" for m in self.markers)
        return has_clock and self.age is not None and self.sex is not None


def _validate_id(raw_id: Optional[str], existing: Iterable[str]) -> str:
    """Namespace and validate the id, or reject it.

    Never interpolated raw: an id that does not match `_ID_RE` is refused, so
    the parenthesis-and-rule payload that redefined `grimage-weight` cannot be
    constructed. The `Caller_` prefix is added unconditionally, which is what
    makes a collision with a curated patient impossible rather than merely
    unlikely.
    """
    candidate = (raw_id or "Patient").strip()
    if candidate.startswith(ID_PREFIX):
        candidate = candidate[len(ID_PREFIX):]
    if not _ID_RE.match(candidate):
        raise PatientSpecError(
            "invalid_patient_id",
            "A patient id must start with a letter and contain only letters, "
            "digits and underscores (max 48 characters).",
            received=raw_id,
        )
    patient_id = f"{ID_PREFIX}{candidate}"
    if patient_id in set(existing):
        raise PatientSpecError(
            "patient_id_collision",
            f"'{patient_id}' already exists in the knowledge base. Submitting a "
            f"second patient under an existing id does not replace it — the "
            f"engine unions both sets of facts and every answer becomes "
            f"non-deterministic. Choose another id.",
            patient_id=patient_id,
        )
    return patient_id


def _as_number(
    value: object,
    *,
    code: str,
    what: str,
    **context: object,
) -> float:
    """Coerce a caller-supplied scalar to float, or raise a *typed* refusal.

    `float("old")` raises a bare `ValueError`, and the API's patient path maps
    only `PatientSpecError` — so `{"age": "old"}` escaped as an unhandled
    exception and the caller got a 500 with no field name in it. A malformed
    field is the caller's mistake, which is a 422 and has to say which field.
    A bool is refused explicitly: Python makes `float(True)` 1.0, and silently
    reading `{"age": true}` as "one year old" is worse than refusing it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PatientSpecError(code, f"{what} must be a number.", received=value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PatientSpecError(
            code, f"{what} must be a number.", received=value
        ) from None
    if not math.isfinite(number):
        raise PatientSpecError(
            code, f"{what} must be a finite number.", received=value
        )
    return number


def _resolve_marker(
    name: str,
    payload: dict,
    *,
    sd_to_years: float,
    elevated_threshold: float,
) -> ResolvedMarker:
    spec = MARKERS.get(name)
    if spec is None:
        raise PatientSpecError(
            "unknown_marker",
            f"'{name}' is not a biomarker this knowledge base reasons about. "
            f"Supported: {', '.join(sorted(MARKERS))}.",
            marker=name,
            supported=sorted(MARKERS),
        )

    z = payload.get("z")
    value = payload.get("value")
    unit = payload.get("unit")
    derived = False
    formula = None

    if z is None and value is None:
        raise PatientSpecError(
            "marker_needs_a_value",
            f"Marker '{name}' needs either `z` (standard deviations from the "
            f"age/sex-adjusted mean) or `value` (a raw measurement).",
            marker=name,
        )

    if z is None:
        if name in YEARS_PER_SD_MARKERS:
            numeric = _as_number(
                value, code="invalid_marker_value",
                what=f"Marker '{name}' value", marker=name,
            )
            if unit is not None and str(unit).strip() and (
                _normalise_unit(unit) not in YEARS_UNITS
            ):
                raise PatientSpecError(
                    "unsupported_unit",
                    f"Marker '{name}' is an age ACCELERATION: a raw value is "
                    f"read as years, and '{unit}' is not a spelling of years. "
                    f"Send `z` instead if you already have standard deviations.",
                    marker=name,
                    accepted_units=sorted(YEARS_UNITS),
                )
            z = numeric / sd_to_years
            derived = True
            unit = unit or "years"
            formula = f"z = years / {sd_to_years:g}   [grimaccel-sd-to-years]"
        elif spec.reference is not None:
            # Coerced OUTSIDE the try: PatientSpecError is a ValueError, so a
            # coercion refusal caught here would be re-wrapped into itself.
            numeric = _as_number(
                value, code="invalid_marker_value",
                what=f"Marker '{name}' value", marker=name,
            )
            try:
                numeric_in_unit, conversion = spec.reference.to_reference_unit(
                    numeric, unit
                )
            except ValueError as exc:
                raise PatientSpecError(
                    "unsupported_unit", f"Marker '{name}': {exc}", marker=name,
                    accepted_units=spec.reference.accepted_units(),
                ) from None
            try:
                z, formula = spec.reference.to_z(numeric_in_unit)
            except ValueError as exc:
                raise PatientSpecError(
                    "invalid_marker_value", f"Marker '{name}': {exc}", marker=name
                ) from None
            if conversion:
                formula = f"{conversion};   {formula}"
            derived = True
            unit = unit or spec.reference.unit
        else:
            raise PatientSpecError(
                "raw_value_unsupported",
                f"Marker '{name}' has no reference distribution in this KB, so a "
                f"raw value cannot be standardised. Send `z` instead — it is the "
                f"native unit for an epigenetic-clock surrogate.",
                marker=name,
            )

    z = _as_number(
        z, code="invalid_marker_value", what=f"Marker '{name}' z", marker=name,
    )
    if not math.isfinite(z) or abs(z) > Z_LIMIT:
        raise PatientSpecError(
            "implausible_z",
            f"Marker '{name}' has z={z}, beyond +/-{Z_LIMIT:g} standard "
            f"deviations. That is almost always a unit mistake.",
            marker=name,
        )

    status = "Elevated" if z > elevated_threshold else (
        "Low" if z < -elevated_threshold else "Normal"
    )
    note = None
    if spec.role == "grimage_component" and "no cause edge" in spec.reaches:
        note = (
            "Carried and credited in the GrimAge decomposition, but no curated "
            "cause reaches it, so it cannot be explained or intervened on yet."
        )
    elif spec.role == "clock" and name == "HorvathAgeAccel":
        note = "Recorded for the clock-discordance picture; no downstream edge."
    return ResolvedMarker(
        name=name, z=z, derived=derived,
        raw_value=(
            _as_number(
                value, code="invalid_marker_value",
                what=f"Marker '{name}' value", marker=name,
            )
            if value is not None else None
        ),
        unit=unit, formula=formula, status=status, note=note,
    )


def build_patient(
    payload: dict,
    *,
    existing_ids: Iterable[str] = (),
    sd_to_years: float = 4.2,
    elevated_threshold: float = 1.0,
) -> BuiltPatient:
    """Validate a caller's patient and render it as MeTTa atoms.

    Every atom is assembled from a validated field. Nothing the caller sends is
    ever interpolated into the text unchecked.
    """
    patient_id = _validate_id(payload.get("id"), existing_ids)

    age = payload.get("age")
    if age is not None:
        age = _as_number(age, code="invalid_age", what="Age")
        if not (AGE_RANGE[0] <= age <= AGE_RANGE[1]):
            raise PatientSpecError(
                "invalid_age",
                f"Age must be between {AGE_RANGE[0]:g} and {AGE_RANGE[1]:g}.",
            )

    sex = payload.get("sex")
    if sex is not None:
        sex = str(sex).strip().capitalize()
        if sex not in SEXES:
            raise PatientSpecError(
                "invalid_sex",
                f"Sex must be one of {sorted(SEXES)} — the baseline CHD risk "
                f"table is stratified by exactly these two, and inventing a "
                f"third branch would invent a number.",
                received=payload.get("sex"),
            )

    smoking = payload.get("smoking")
    if smoking is not None:
        smoking = str(smoking).strip()
        if smoking not in SMOKING:
            raise PatientSpecError(
                "invalid_smoking_status",
                f"Smoking status must be one of {sorted(SMOKING)}.",
                received=payload.get("smoking"),
            )

    raw_markers = payload.get("markers") or {}
    if not isinstance(raw_markers, dict):
        raise PatientSpecError("invalid_markers", "`markers` must be an object.")
    if len(raw_markers) > MAX_MARKERS:
        raise PatientSpecError(
            "too_many_markers",
            f"At most {MAX_MARKERS} markers per patient.",
        )

    resolved: list[ResolvedMarker] = []
    for name, entry in raw_markers.items():
        if isinstance(entry, (int, float)):
            entry = {"z": float(entry)}   # already a real number
        if not isinstance(entry, dict):
            raise PatientSpecError(
                "invalid_marker",
                f"Marker '{name}' must be a number (a z-score) or an object with "
                f"`z` or `value`.",
                marker=name,
            )
        resolved.append(_resolve_marker(
            name, entry, sd_to_years=sd_to_years,
            elevated_threshold=elevated_threshold,
        ))

    warnings: list[str] = []
    if not any(m.name == "AgeAccelGrim" for m in resolved):
        warnings.append(
            "No AgeAccelGrim measurement: the 10-year CHD risk model reads that "
            "clock and will return nothing for this patient. Diagnosis, "
            "supplement ranking and intervention ranking still work."
        )
    if age is None or sex is None:
        warnings.append(
            "Age and sex are both required for an absolute risk: they select the "
            "baseline the multiplier applies to. Without them the risk queries "
            "return empty rather than guessing a baseline."
        )
    # `smoking` used to be inert metadata — nothing consumed PatientSmoking.
    # lifestyle_evidence.metta changed that, but the lever acts on DNAmPACKYRS,
    # not on the status string, so a smoker with no DNAm pack-years measurement
    # still gets zero from the smoking counterfactual. Say so rather than let
    # the caller read a structural zero as "quitting would not help him".
    has_packyears = any(m.name == "DNAmPACKYRS" for m in resolved)
    if smoking in ("FormerSmoker", "CurrentSmoker") and not has_packyears:
        warnings.append(
            "Smoking status is recorded but no DNAmPACKYRS measurement was sent. "
            "The smoking lever works through that GrimAge component, so "
            "(counterfactual-patient &self <Patient> SmokingCessation) will "
            "return an expected delta of 0 with an empty (Via ()) for this "
            "patient. That zero means 'no measured pack-years signal to act on', "
            "not 'quitting would not help'."
        )
    # The symmetric case, which used to be silent and was the worse one: an
    # elevated DNAm pack-years surrogate in someone who does not smoke. The
    # clock is an elastic-net estimate with real error and it responds to
    # second-hand exposure, so this is an ordinary data state — but the lever
    # requires (LeverRequiresSmoking SmokingCessation CurrentSmoker), so the
    # zero it returns means something different again.
    if smoking in ("NeverSmoker", "FormerSmoker") and has_packyears:
        warnings.append(
            f"DNAmPACKYRS was sent for a {smoking}. The measurement is kept and "
            f"credited in the GrimAge decomposition, but the knowledge base "
            f"declares (LeverRequiresSmoking SmokingCessation CurrentSmoker), so "
            f"a SmokingCessation counterfactual for this patient computes the "
            f"arithmetic of the exposure marker rather than a benefit they can "
            f"obtain. /query and /metta/run repeat that warning next to the "
            f"number; the engine itself does not yet enforce it "
            f"(pln_counterfactual.metta §3b)."
        )

    # A DERIVED z is not an adjusted z, and the atom cannot say so. The space
    # gets `(MeasuredZ <patient> <marker> <z>)` either way, so the inference
    # layer cannot tell a caller's age/sex-adjusted z from one this service
    # standardised against a single pooled mean and sd — and GET
    # /patients/markers publishes the convention as "AGE- AND SEX-ADJUSTED".
    # The provenance is in `derived`/`formula` on the response and in this
    # warning; it deliberately is NOT invented into the KB as an adjustment
    # that was never made.
    derived = [m.name for m in resolved if m.derived and m.name not in YEARS_PER_SD_MARKERS]
    if derived:
        warnings.append(
            "Standardised server-side from a raw value: " + ", ".join(derived) +
            ". These z-scores come from a single POOLED reference mean and sd — "
            "this repository has no age/sex-stratified table — so unlike a z you "
            "send, they are NOT age- and sex-adjusted, and the atoms in the "
            "space cannot be told apart from adjusted ones. Send `z` when you "
            "have a properly standardised measurement."
        )

    inert = [m.name for m in resolved if m.note]
    if inert:
        warnings.append(
            "Carried but not yet reasoned over (no curated cause or effect edge "
            "reaches them): " + ", ".join(inert) + "."
        )

    lines = [f"(InstanceOf {patient_id} PatientProfile)"]
    if age is not None:
        lines.append(f"(PatientAge {patient_id} {age:g})")
    if sex is not None:
        lines.append(f"(PatientSex {patient_id} {sex})")
    if smoking is not None:
        lines.append(f"(PatientSmoking {patient_id} {smoking})")
    for marker in sorted(resolved, key=lambda m: m.name):
        lines.append(f"(MeasuredZ {patient_id} {marker.name} {marker.z:.6g})")

    return BuiltPatient(
        patient_id=patient_id,
        atoms="\n".join(lines),
        markers=resolved,
        warnings=warnings,
        age=age,
        sex=sex,
        smoking=smoking,
    )


def marker_catalog() -> list[dict]:
    """The supported markers, for GET /patients/markers."""
    out: list[dict] = []
    for spec in MARKERS.values():
        entry = {
            "marker": spec.name,
            "role": spec.role,
            "reaches": spec.reaches,
            "accepts_raw_value": spec.accepts_raw or spec.name in YEARS_PER_SD_MARKERS,
        }
        if spec.name in YEARS_PER_SD_MARKERS:
            entry["raw_unit"] = "years of age acceleration"
            entry["accepted_units"] = sorted(YEARS_UNITS)
            entry["conversion"] = "z = years / grimaccel-sd-to-years"
        elif spec.reference is not None:
            entry["raw_unit"] = spec.reference.unit
            entry["accepted_units"] = spec.reference.accepted_units()
            entry["reference_mean"] = spec.reference.mean
            entry["reference_sd"] = spec.reference.sd
            entry["log_scale"] = spec.reference.log_scale
            entry["reference_source"] = spec.reference.source
            entry["provisional"] = True
        out.append(entry)
    return sorted(out, key=lambda e: (e["role"], e["marker"]))
