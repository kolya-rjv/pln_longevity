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
from functools import lru_cache
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
    "LinAgeAccel": MarkerSpec(
        "LinAgeAccel", "clock",
        "the LinAge2 CLINICAL clock's biological-age delta (BA - CA, years) — the "
        "predictor of the all-cause-mortality hazard (linage-hazard-patient). Send "
        "the whole LinAge2 /predict response under `linage2` instead of this bare "
        "marker to also get the per-lab decomposition and the counterfactuals; the "
        "two are mutually exclusive.",
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
        ),
    ),
    "HbA1c": MarkerSpec(
        "HbA1c", "blood_marker",
        "insulin-resistance readout; drives the metabolic axis",
        Reference(
            unit="%", mean=5.5, sd=0.5,
            source="coarse prior for non-diabetic adults. Replace with an "
                   "age/sex-stratified cohort table.",
        ),
    ),
}

#: Markers whose raw value is naturally expressed in YEARS of age acceleration.
#: Converted with the risk layer's own `grimaccel-sd-to-years` knob so the two
#: never drift apart.
YEARS_PER_SD_MARKERS = {"AgeAccelGrim", "HorvathAgeAccel"}
#: The LinAge2 delta is also years, but on the clinical clock's own spread —
#: `linage-sd-to-years` in pln_linage2.metta (8.66 y/SD, derived from the model's
#: training cohort), not GrimAge's 4.2. Same rule, different knob.
LINAGE_YEARS_MARKERS = {"LinAgeAccel"}

#: What a person can say that puts them OUTSIDE the at-risk set of the 10-year CHD model: it
#: estimates a FIRST coronary event (Lu 2019's hazard ratio is for incident CHD; NHANES's own
#: MCQ160C/D/E are lifetime "ever told" prevalence, "usable only as an exclusion from the at-risk
#: set", nhanes_baseline.metta). Heart failure (MCQ160B) is a different event and is not here.
PREVALENT_CHD = ("coronary heart disease", "angina", "heart attack")

SEXES = {"Male", "Female"}
SMOKING = {"NeverSmoker", "FormerSmoker", "CurrentSmoker"}

#: A caller-supplied id is namespaced so it can never collide with a curated
#: patient, whatever the caller sends.
ID_PREFIX = "Caller_"
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,47}$")

Z_LIMIT = 12.0          # |z| beyond this is a unit mistake, not a patient
AGE_RANGE = (0.0, 130.0)
MAX_MARKERS = 40


#: `(Effect <from> <to> Pos|Neg` at the start of a line (a comment line starts with `;`).
_EFFECT_INTO_RE = re.compile(r"^\(Effect\s+\S+\s+(\S+)\s+(?:Pos|Neg)\b", re.MULTILINE)


@lru_cache(maxsize=1)
def kb_effect_markers() -> frozenset[str]:
    """The markers a patient can carry that the knowledge base has a curated Effect edge
    INTO — the only ones a cause can be credited to (diagnose-patient), an intervention
    can be said to act on (the personalised ranking) or a supplement can target. A value
    outside this set (an albumin, a creatinine, a typed diagnosis), or a LOW one, changes
    none of those forms. Read off the KB's own files, as core.patient_context reads the
    thresholds, so a new bridge widens it with no edit here (today: CRP, DNAmGDF15,
    DNAmPACKYRS, DNAmPAI1, FastingGlucose, HbA1c)."""
    from config import ONTOLOGY_DIR, PLN_MAX_KB_FILE_BYTES   # lazily: nothing else here needs config
    found: set[str] = set()
    for path in sorted(ONTOLOGY_DIR.glob("*.metta")):
        try:
            if path.stat().st_size > PLN_MAX_KB_FILE_BYTES:
                continue                                      # not loaded at run time
            found.update(m for m in _EFFECT_INTO_RE.findall(path.read_text(encoding="utf-8"))
                         if m in MARKERS)
        except OSError:
            continue
    return frozenset(found)


#: `(Interaction <A> <B> "<note>"` at the start of a line.
_INTERACTION_RE = re.compile(r"^\(Interaction\s+(\S+)\s+(\S+)", re.MULTILINE)
MAX_MEDICATIONS = 10


@lru_cache(maxsize=1)
def kb_interaction_drugs() -> frozenset[str]:
    """The KB symbols an `(Interaction A B …)` fact names (today Berberine, Metformin): the only
    drugs a `(CurrentMedication <Patient> <Drug>)` can make a supplement flag fire on, and so the
    allow-list for `medications`. Read off the KB's own files, like kb_effect_markers."""
    from config import ONTOLOGY_DIR, PLN_MAX_KB_FILE_BYTES
    found: set[str] = set()
    for path in sorted(ONTOLOGY_DIR.glob("*.metta")):
        try:
            if path.stat().st_size > PLN_MAX_KB_FILE_BYTES:
                continue
            for a, b in _INTERACTION_RE.findall(path.read_text(encoding="utf-8")):
                found.update((a, b))
        except OSError:
            continue
    return frozenset(found)


def medication_note(drugs: Iterable[str]) -> str:
    """What a recorded medication does, and does not do (the KB does not model 'already taking')."""
    return (
        f"Current medication recorded: {', '.join(drugs)}. It is used only to flag a supplement that "
        f"interacts with it (the knowledge base's one interaction fact: Berberine with Metformin), in "
        f"the supplement plan and in the single-supplement answer. It changes no ranking and no "
        f"LinAge2 number: the intervention ranking and the LinAge2 counterfactual still treat it as a "
        f"candidate, not as something already taken."
    )


#: The sentence the no-GrimAge note carries only while it is true.
STILL_WORK = ("Diagnosis, supplement ranking and intervention ranking can still work from the "
              "elevated markers the knowledge base has edges for.")
#: Starts the builder's note that the shared layers have nothing to work from; the tab
#: rewords it by this prefix.
NO_WITNESS_PREFIX = "No elevated marker the knowledge base can use:"


def prevalent_chd_note(conditions: Iterable[str]) -> str:
    """The 10-year CHD risk for someone who reports CHD: the same number with or without that
    history (probed), so it needs saying that it does not describe them."""
    return (
        f"Reported {', '.join(conditions)}: the 10-year heart-disease risk model estimates a FIRST "
        f"coronary event in someone without CHD, so its number does not describe a person who already "
        f"has it — it is the same number with or without that history. Do not read it as the risk "
        f"of another event."
    )


def no_witness_note() -> str:
    """Said when none of a patient's values is elevated AND has a curated edge: the
    diagnosis, the supplement plan and the intervention ranking have nothing to read."""
    return (
        f"{NO_WITNESS_PREFIX} none of the values given is elevated and has a curated cause or "
        f"effect (the knowledge base has them for {', '.join(sorted(kb_effect_markers()))}). For "
        f"this patient the diagnosis returns (), every supplement tier is empty and the "
        f"intervention ranking is the population ranking, the same as for an unknown patient. "
        f"That is 'nothing to work from', not 'no cause'. A typed diagnosis, a lab with no edge "
        f"(albumin, creatinine, blood pressure, RDW), a low value, and a glucose not marked "
        f"fasting count for none of them."
    )


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

    #: Set when the caller sent a LinAge2 response under `linage2`.
    linage2: Optional["BuiltLinAge2"] = None

    #: The markers this patient has ELEVATED that the knowledge base has a curated edge
    #: into (kb_effect_markers) — what the diagnosis, the supplement plan and the
    #: intervention ranking can read. Empty: they have nothing to work from.
    witnesses: list[str] = field(default_factory=list)

    #: Coronary heart disease / angina / heart attack the person reports (PREVALENT_CHD). A
    #: flag, not an atom: nothing in the KB reads it; it qualifies the 10-year CHD risk.
    prevalent_chd: list[str] = field(default_factory=list)

    #: Drugs the person takes now that the KB has an Interaction fact for (kb_interaction_drugs).
    #: `(CurrentMedication <id> <drug>)` goes to `shared_atoms` ONLY: the LinAge2 space has no
    #: head-symbol room for a new head, and nothing there reads it.
    medications: list[str] = field(default_factory=list)

    #: The same patient WITHOUT its LinAge2 atoms — what the SHARED execution space
    #: gets. `atoms` (everything) is for the LinAge2 scoped space, the preview and
    #: the translator's prompt. The split is measured, not tidiness: with the
    #: LinAgeDelta / LinAgeContribution / LinAgeAccel atoms of one /predict response
    #: in the shared space, hyperon 0.2.10 aborts on diagnose-patient,
    #: predict-risk-patient and recommend-supplements-patient for that patient, and
    #: the same three forms answer without them (tests/test_linage2.py). Nothing in
    #: the shared space reads those atoms — its stack has no LinAge2 layer.
    shared_atoms: str = ""

    @property
    def has_grimage(self) -> bool:
        """A GrimAge acceleration value was given — the one input the 10-year CHD risk model
        reads. Without it there is no heart-specific risk for this patient."""
        return any(m.name == "AgeAccelGrim" for m in self.markers)

    @property
    def can_predict_risk(self) -> bool:
        return self.has_grimage and self.age is not None and self.sex is not None

    @property
    def has_linage2(self) -> bool:
        """A LinAge2 delta is present (as a block or a bare LinAgeAccel marker), so
        the LinAge2 forms — hazard, decomposition, counterfactuals — have input."""
        return self.linage2 is not None or any(m.name == "LinAgeAccel" for m in self.markers)


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
    linage_sd_to_years: float = 8.66,
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
            z = _as_number(
                value, code="invalid_marker_value",
                what=f"Marker '{name}' value", marker=name,
            ) / sd_to_years
            derived = True
            unit = unit or "years"
            formula = f"z = years / {sd_to_years:g}   [grimaccel-sd-to-years]"
        elif name in LINAGE_YEARS_MARKERS:
            z = _as_number(
                value, code="invalid_marker_value",
                what=f"Marker '{name}' value", marker=name,
            ) / linage_sd_to_years
            derived = True
            unit = unit or "years"
            formula = f"z = years / {linage_sd_to_years:g}   [linage-sd-to-years]"
        elif spec.reference is not None:
            # Coerced OUTSIDE the try: PatientSpecError is a ValueError, so a
            # coercion refusal caught here would be re-wrapped into itself.
            numeric = _as_number(
                value, code="invalid_marker_value",
                what=f"Marker '{name}' value", marker=name,
            )
            try:
                z, formula = spec.reference.to_z(numeric)
            except ValueError as exc:
                raise PatientSpecError(
                    "invalid_marker_value", f"Marker '{name}': {exc}", marker=name
                ) from None
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
    elif name == "LinAgeAccel":
        note = (
            "Read by the LinAge2 forms only (linage-hazard-patient and friends), "
            "which run in their own query-scoped space. The CHD risk model reads "
            "AgeAccelGrim, not this."
        )
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
    linage_sd_to_years: Optional[float] = None,
    effect_markers: Optional[Iterable[str]] = None,
) -> BuiltPatient:
    """Validate a caller's patient and render it as MeTTa atoms.

    Every atom is assembled from a validated field. Nothing the caller sends is
    ever interpolated into the text unchecked.

    `linage2`, when present, is a LinAge2 /predict response for THIS patient; it is
    validated and rendered by core.linage2_builder and its atoms are appended. It
    is request-scoped like everything else here.
    """
    # Imported here rather than at module top: linage2_builder imports this module
    # for PatientSpecError / _as_number, and a top-level import in both directions
    # would be a cycle.
    from core.linage2_builder import build_linage2, linage_sd_to_years as _linage_knob

    if linage_sd_to_years is None:
        linage_sd_to_years = _linage_knob()

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

    prevalent_chd = payload.get("prevalent_chd") or []
    if (not isinstance(prevalent_chd, (list, tuple)) or len(prevalent_chd) > len(PREVALENT_CHD)
            or any(c not in PREVALENT_CHD for c in prevalent_chd)):
        raise PatientSpecError(
            "invalid_prevalent_chd",
            f"`prevalent_chd` must be a list drawn from {list(PREVALENT_CHD)}.",
            received=payload.get("prevalent_chd"),
        )
    prevalent_chd = [c for c in PREVALENT_CHD if c in prevalent_chd]     # canonical order, once each

    medications = payload.get("medications") or []
    allowed_drugs = kb_interaction_drugs()
    if (not isinstance(medications, (list, tuple)) or len(medications) > MAX_MEDICATIONS
            or any(not isinstance(m, str) or m not in allowed_drugs for m in medications)):
        raise PatientSpecError(
            "invalid_medication",
            f"`medications` must be a list (at most {MAX_MEDICATIONS}) drawn from the drugs the knowledge "
            f"base holds an interaction fact for: {sorted(allowed_drugs)}.",
            received=payload.get("medications"), supported=sorted(allowed_drugs),
        )
    medications = sorted(set(medications))

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
            linage_sd_to_years=linage_sd_to_years,
        ))

    warnings: list[str] = []

    # ── the LinAge2 block ──────────────────────────────────────────────────
    # A whole /predict response, validated and rendered by core.linage2_builder.
    # It becomes the LinAgeAccel clock marker (so it appears in `markers` and is
    # rendered by the same loop as every other z, exactly once) plus the
    # LinAgeDelta / LinAgeContribution atoms the LinAge2 forms read.
    linage2_payload = payload.get("linage2")
    built_linage2 = None
    if linage2_payload is not None:
        if any(m.name == "LinAgeAccel" for m in resolved):
            raise PatientSpecError(
                "duplicate_clock",
                "Send EITHER `markers.LinAgeAccel` OR a `linage2` block, not both: "
                "two z values for one clock double-count in every sum that reads it.",
            )
        built_linage2 = build_linage2(
            linage2_payload, patient_id, age=age,
            elevated_threshold=elevated_threshold, sd_to_years=linage_sd_to_years,
        )
        if age is None:
            age = float(built_linage2.chronological_age)
            warnings.append(
                f"`age` was not sent; taken from the LinAge2 response's "
                f"chronological_age ({age:g})."
            )
        resolved.append(ResolvedMarker(
            name="LinAgeAccel", z=built_linage2.z, derived=True,
            raw_value=built_linage2.delta_years, unit="years",
            formula=f"z = years / {linage_sd_to_years:g}   [linage-sd-to-years]",
            status=built_linage2.status,
            note=(
                "From the `linage2` block. Read by the LinAge2 forms "
                "(linage-hazard-patient, linage-decomposition-patient, "
                "linage-counterfactual-patient), which run in their own query-scoped "
                "space; the CHD risk model reads AgeAccelGrim, not this."
            ),
        ))
        warnings.extend(built_linage2.warnings)
        # The join that makes a cause creditable needs a witness the CALLER sends:
        # the patient's own z for CRP / HbA1c / FastingGlucose, or a current-smoker
        # status. Say so up front, or the decomposition comes back with every
        # DrivenBy empty and reads as "the KB knows no causes".
        witnesses = {"CRP", "HbA1c", "FastingGlucose"}
        if not (witnesses & {m.name for m in resolved}) and smoking != "CurrentSmoker":
            warnings.append(
                "The LinAge2 block was sent without any of the markers the knowledge "
                "base can join it to (CRP, HbA1c, FastingGlucose as z or value) and "
                "without smoking = CurrentSmoker — the witnesses a cause needs. The "
                "per-lab years will be reported, "
                "but no contribution can be credited to a cause and every "
                "counterfactual will return 0: a contribution's sign is not a lab's "
                "direction (LinAge2's weights are sex-specific projections), so the "
                "engine needs the patient's own value to say a lab is high."
            )
    elif any(m.name == "LinAgeAccel" for m in resolved):
        warnings.append(
            "LinAgeAccel was sent as a bare marker: the LinAge2 hazard is computable, "
            "but the per-lab decomposition and the counterfactuals need the whole "
            "/predict response under `linage2`."
        )
    edge_markers = kb_effect_markers() if effect_markers is None else frozenset(effect_markers)
    witnesses = sorted(m.name for m in resolved if m.status == "Elevated" and m.name in edge_markers)
    if not any(m.name == "AgeAccelGrim" for m in resolved):
        warnings.append(
            "No AgeAccelGrim measurement: the 10-year CHD risk model reads that "
            "clock and will return nothing for this patient."
            + (f" {STILL_WORK}" if witnesses else "")
        )
    if not witnesses:
        warnings.append(no_witness_note())
    if prevalent_chd and any(m.name == "AgeAccelGrim" for m in resolved) and age is not None and sex is not None:
        warnings.append(prevalent_chd_note(prevalent_chd))
    if medications:
        warnings.append(medication_note(medications))
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
            + (" The LinAge2 smoking counterfactual (linage-counterfactual-patient "
               "&self <Patient> SmokingCessation) reads the cotinine input instead and "
               "does not need it." if built_linage2 is not None and smoking == "CurrentSmoker"
               else " The LinAge2 smoking counterfactual returns 0 for a former smoker "
               "as well: the knowledge base credits cotinine years to smoking only for "
               "a stated current smoker."
               if built_linage2 is not None else "")
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
    derived = [
        m.name for m in resolved
        if m.derived and m.name not in YEARS_PER_SD_MARKERS and m.name not in LINAGE_YEARS_MARKERS
    ]
    if derived:
        warnings.append(
            "Standardised server-side from a raw value: " + ", ".join(derived) +
            ". These z-scores come from a single POOLED reference mean and sd — "
            "this repository has no age/sex-stratified table — so unlike a z you "
            "send, they are NOT age- and sex-adjusted, and the atoms in the "
            "space cannot be told apart from adjusted ones. Send `z` when you "
            "have a properly standardised measurement."
        )

    # LinAgeAccel carries a note (it is read only in the LinAge2 scoped space) but it
    # IS reasoned over there — the hazard is computed from it — so it is not inert.
    inert = [m.name for m in resolved if m.note and m.name not in LINAGE_YEARS_MARKERS]
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
    shared_lines = list(lines)
    for marker in sorted(resolved, key=lambda m: m.name):
        line = f"(MeasuredZ {patient_id} {marker.name} {marker.z:.6g})"
        lines.append(line)
        if marker.name not in LINAGE_YEARS_MARKERS:
            shared_lines.append(line)
    # a medication goes to the SHARED space only (BuiltPatient.medications): not into `atoms`,
    # which also feeds the LinAge2 space, where a new head symbol has no room
    shared_lines.extend(f"(CurrentMedication {patient_id} {drug})" for drug in medications)
    if built_linage2 is not None:
        lines.append(built_linage2.atoms)

    return BuiltPatient(
        patient_id=patient_id,
        atoms="\n".join(lines),
        shared_atoms="\n".join(shared_lines),
        markers=resolved,
        warnings=warnings,
        age=age,
        sex=sex,
        smoking=smoking,
        linage2=built_linage2,
        witnesses=witnesses,
        prevalent_chd=prevalent_chd,
        medications=medications,
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
            entry["conversion"] = "z = years / grimaccel-sd-to-years"
        elif spec.name in LINAGE_YEARS_MARKERS:
            entry["raw_unit"] = "years of LinAge2 BA - CA delta"
            entry["conversion"] = "z = years / linage-sd-to-years"
            entry["scoped"] = (
                "read in the LinAge2 query-scoped space only; prefer the `linage2` "
                "block (the whole /predict response) — see GET /linage2/features"
            )
        elif spec.reference is not None:
            entry["raw_unit"] = spec.reference.unit
            entry["reference_mean"] = spec.reference.mean
            entry["reference_sd"] = spec.reference.sd
            entry["log_scale"] = spec.reference.log_scale
            entry["reference_source"] = spec.reference.source
            entry["provisional"] = True
        out.append(entry)
    return sorted(out, key=lambda e: (e["role"], e["marker"]))
