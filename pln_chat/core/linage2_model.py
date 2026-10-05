"""LinAge2, evaluated in-process from extracted parameters.

The clinical clock of Fong et al. 2025 is additive in its inputs (see the header of
`scripts/extract_linage2_model.py` for the formula and how every constant was read
out of Rejuve/LinAge2-Python's artifacts). So instead of calling that service, the
knowledge base evaluates the same formula from `data/linage2/linage2_model.json` —
~100 KB of numbers — and produces the same `/predict` response body the service
returns. `core.linage2_builder.build_linage2` consumes it unchanged; nothing
downstream knows where the numbers came from.

What this module adds over the service, on purpose:

* **Honest provenance per model input.** The service flags only lab inputs it
  imputed. Here a derived input is also flagged when any part of it was imputed
  (LDL from total cholesterol / HDL / triglycerides; the urine albumin/creatinine
  ratio), and a questionnaire score is flagged when its questions were not
  answered (the service silently assumes "no diagnoses, good health, no visits").
  Every non-measured input carries `is_imputed: true`, so the knowledge base totals
  it apart and never credits a cause to it.
* **Cotinine on the training scale** (0-3), and imputed from digitized values —
  see the extraction script.
* **Refusals instead of NaN.** An unknown input code, a non-positive value where the
  model takes a logarithm, an age outside 20-90: each is a PatientSpecError (422),
  not a NaN that the service would silently drop from its contributions.

The weights stay in Python. The knowledge base still sees only years per input
(`LinAgeContribution`), so the modelling rule of `pln_linage2.metta` — a cause is
credited only under the patient's own witness — is unchanged.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from core.patient_builder import PatientSpecError

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "linage2" / "linage2_model.json"

#: Ages the imputation table covers. The model itself was fitted on 40-85; a
#: result outside that is computed and warned, not refused.
MIN_AGE, MAX_AGE = 20, 90

#: The questionnaire items each derived score reads. fs1 reads 23 items but counts
#: 22 — upstream reads MCQ160B (heart failure) and never adds it; reproduced.
FS1_ITEMS = (
    "BPQ020", "DIQ010", "KIQ020", "MCQ010", "MCQ053",
    "MCQ160A", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
    "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K", "MCQ160L",
    "MCQ220", "OSQ010A", "OSQ010B", "OSQ010C", "OSQ060",
    "PFQ056", "HUQ070",
)
FS2_ITEMS = ("HUQ010", "HUQ020")
FS3_ITEMS = ("HUQ050",)

#: Allowed NHANES answer codes per questionnaire item.
_ANSWER_CODES: dict[str, frozenset[int]] = {
    **{q: frozenset({1, 2}) for q in FS1_ITEMS if q != "DIQ010"},
    "MCQ160B": frozenset({1, 2}),
    "DIQ010": frozenset({1, 2, 3}),                      # 3 = borderline, counted as yes
    "HUQ010": frozenset({1, 2, 3, 4, 5}),                # excellent .. poor
    "HUQ020": frozenset({1, 2, 3}),                      # better, worse, about the same
    "HUQ050": frozenset({0, 1, 2, 3, 4, 5}),             # visit-count category
}

#: Provenance labels, in the order a reader should worry about them.
MEASURED = "measured"
IMPUTED = "imputed"                        # a cohort median stood in for the value
DERIVED_FROM_IMPUTED = "derived_from_imputed"
ASSUMED = "assumed"                        # a questionnaire default stood in


@dataclass(frozen=True)
class LinAge2Model:
    raw: dict

    @property
    def lab_inputs(self) -> list[str]:
        return list(self.raw["lab_inputs"])

    @property
    def features(self) -> list[str]:
        return [f["code"] for f in self.raw["features"]]

    def nhanes_range(self, code: str) -> tuple[float, float]:
        lo, hi = self.raw["nhanes_range"][code]
        return float(lo), float(hi)

    def description(self, code: str) -> str:
        return self.raw["descriptions"].get(code, "")

    def training_ages(self, sex: str) -> tuple[float, float]:
        lo, hi = self.raw["sex"][sex.lower()]["training_age_years"]
        return float(lo), float(hi)

    def imputed_value(self, sex: str, age: float, code: str) -> float:
        ages = self.raw["imputation"]["ages"]
        idx = min(max(int(round(age)), ages[0]), ages[-1]) - ages[0]
        value = self.raw["imputation"][sex.lower()][code][idx]
        if value is None:                                   # pragma: no cover - data guard
            raise PatientSpecError("linage2_model_incomplete",
                                   f"No reference median for {code} at age {age:g}.")
        return float(value)


@lru_cache(maxsize=1)
def load_model() -> LinAge2Model:
    return LinAge2Model(json.loads(MODEL_PATH.read_text(encoding="utf-8")))


def young_reference_z(code: str, value: float, sex: str) -> float:
    """`value` of the lab `code` (in the model's own unit) as a z against LinAge2's reference for `sex`:
    (Box-Cox(value) - the sex's median) / its MAD, the reference being the NHANES 1999-2000 participants aged up
    to 50 in the model's own reference matrix (scripts/extract_linage2_model.py), not its 40-85 training cohort. NOT age-adjusted: a lab that drifts with age (RDW up, albumin down) reads higher
    in an older person for that reason alone. The one place the KB's z for a LinAge2 input is made."""
    model = load_model()
    feats = model.raw["features"]
    idx = next((i for i, f in enumerate(feats) if f["code"] == code), None)
    if idx is None:
        raise KeyError(f"{code} is not a LinAge2 input")
    side = model.raw["sex"][_check_sex(sex).lower()]
    return (_box_cox(float(value), feats[idx]["boxcox_lambda"]) - side["median"][idx]) / side["mad"][idx]


@dataclass
class LinAge2Result:
    """The service-shaped response plus what the service does not say."""
    #: The `/predict` body: what `build_linage2` (and `patient.linage2`) consume.
    response: dict
    #: feature code -> measured | imputed | derived_from_imputed | assumed
    provenance: dict[str, str]
    #: every lab input's value as the model used it (measured or imputed), by code
    inputs_used: dict[str, float]
    imputed_inputs: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def delta_years(self) -> float:
        return float(self.response["metadata"]["delta_ba_ca"])

    @property
    def biological_age(self) -> float:
        return float(self.response["biological_age"])


def _box_cox(x: float, lam: Optional[float]) -> float:
    if lam is None:
        return x
    if lam == 0:
        return math.log(x) if x > 0 else -math.inf
    return (x ** lam - 1.0) / lam


def _fs1(answers: Mapping[str, int]) -> float:
    yes = 0
    for q in FS1_ITEMS:
        a = answers[q]
        yes += (a in (1, 3)) if q == "DIQ010" else (a == 1)
    return yes / 22


def _fs2(answers: Mapping[str, int]) -> float:
    h10, h20 = answers["HUQ010"], answers["HUQ020"]
    a = (2 if h10 == 4 else 0) + (4 if h10 == 5 else 0)
    d = 1 - (0.5 if h20 == 1 else 0) + (1 if h20 == 2 else 0)
    return float(a * d)


def _fs3(answers: Mapping[str, int]) -> float:
    v = answers["HUQ050"]
    return 0.0 if v in (77, 99) else float(v)


def _check_sex(sex: Optional[str]) -> str:
    if sex not in ("Male", "Female"):
        raise PatientSpecError(
            "linage2_sex_required",
            "LinAge2 has separate male and female models; `sex` must be Male or Female.",
            received=sex,
        )
    return sex


def _check_age(age: Optional[float]) -> float:
    if age is None or not isinstance(age, (int, float)) or not math.isfinite(age):
        raise PatientSpecError("linage2_age_required",
                               "LinAge2 needs the person's age in years.", received=age)
    if not MIN_AGE <= float(age) <= MAX_AGE:
        raise PatientSpecError(
            "linage2_age_out_of_range",
            f"Age {age:g} is outside {MIN_AGE}-{MAX_AGE}, the ages the reference cohort "
            f"covers (the model itself was fitted on 40-85).",
            age=age, supported=[MIN_AGE, MAX_AGE],
        )
    return float(age)


def compute_linage2(
    *,
    sex: Optional[str],
    age: Optional[float],
    labs: Mapping[str, float],
    questionnaire: Optional[Mapping[str, int]] = None,
) -> LinAge2Result:
    """Score one person. `labs` maps NHANES codes to values in the model's units
    (`load_model().raw["descriptions"]`); `LBXCOT` is the cotinine level 0-3.
    `LDLV` may be given directly (an LDL in mmol/L) instead of the three lipids.
    Anything missing is imputed from the same-sex, same-age reference cohort and
    flagged."""
    model = load_model()
    sex = _check_sex(sex)
    age = _check_age(age)
    sexb = model.raw["sex"][sex.lower()]
    lab_inputs = set(model.lab_inputs)
    warnings: list[str] = []

    unknown = sorted(k for k in labs if k not in lab_inputs and k != "LDLV")
    if unknown:
        raise PatientSpecError(
            "linage2_unknown_input",
            f"Not a LinAge2 input: {', '.join(unknown)}. Inputs are NHANES codes "
            f"(GET /linage2/features).",
            inputs=unknown,
        )
    values: dict[str, float] = {}
    for code, raw in labs.items():
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise PatientSpecError("linage2_invalid_value",
                                   f"{code} must be a number, got {raw!r}.", input=code) from None
        if not math.isfinite(v):
            raise PatientSpecError("linage2_invalid_value", f"{code} is not finite.", input=code)
        if v < 0:
            # the service would take log(negative) = NaN and silently drop the input
            raise PatientSpecError("linage2_invalid_value",
                                   f"{code} = {v:g}: a lab value cannot be negative.", input=code)
        values[code] = v
    if "LBXCOT" in values and values["LBXCOT"] not in (0.0, 1.0, 2.0, 3.0):
        raise PatientSpecError(
            "linage2_invalid_cotinine_level",
            "LBXCOT is the cotinine LEVEL the model was trained on: 0 (<10 ng/mL, "
            "non-smoker), 1 (10-100), 2 (100-200), 3 (>=200).",
            received=values["LBXCOT"],
        )

    answers_given = dict(questionnaire or {})
    bad_q = sorted(q for q in answers_given if q not in _ANSWER_CODES)
    if bad_q:
        raise PatientSpecError("linage2_invalid_questionnaire",
                               f"Not a LinAge2 questionnaire item: {', '.join(bad_q)}.",
                               items=bad_q)
    for q, a in answers_given.items():
        if a not in _ANSWER_CODES[q]:
            raise PatientSpecError(
                "linage2_invalid_questionnaire",
                f"{q} = {a!r} is not an NHANES answer code for that item "
                f"({sorted(_ANSWER_CODES[q])}).", item=q, received=a,
            )
    answers = {**{k: int(v) for k, v in model.raw["questionnaire_defaults"].items()},
               **answers_given}

    # ── fill, derive, and record where each number came from ──────────────────
    x: dict[str, float] = {}
    imputed_inputs: list[str] = []
    for code in model.lab_inputs:
        if code in values:
            x[code] = values[code]
        else:
            x[code] = model.imputed_value(sex, age, code)
            imputed_inputs.append(code)
    provenance: dict[str, str] = {
        code: (IMPUTED if code in imputed_inputs else MEASURED) for code in model.lab_inputs
    }

    lipids = ("LBDTCSI", "LBDSTRSI", "LBDHDLSI")
    if "LDLV" in values:
        x["LDLV"] = values["LDLV"]
        provenance["LDLV"] = MEASURED
    else:
        x["LDLV"] = x["LBDTCSI"] - x["LBDSTRSI"] / 5 - x["LBDHDLSI"]
        provenance["LDLV"] = (MEASURED if all(c in values for c in lipids) else
                              IMPUTED if not any(c in values for c in lipids) else
                              DERIVED_FROM_IMPUTED)
    x["crAlbRat"] = x["URXUMASI"] / (x["URXUCRSI"] * 1.1312e-4)
    urine = ("URXUMASI", "URXUCRSI")
    provenance["crAlbRat"] = (MEASURED if all(c in values for c in urine) else
                              IMPUTED if not any(c in values for c in urine) else
                              DERIVED_FROM_IMPUTED)
    x["fs1Score"], x["fs2Score"], x["fs3Score"] = _fs1(answers), _fs2(answers), _fs3(answers)
    for score, items in (("fs1Score", FS1_ITEMS), ("fs2Score", FS2_ITEMS), ("fs3Score", FS3_ITEMS)):
        provenance[score] = MEASURED if any(q in answers_given for q in items) else ASSUMED

    # ── the clock ─────────────────────────────────────────────────────────────
    z_max = float(model.raw["z_max"])
    contributions: list[dict] = []
    total = 0.0
    for i, feat in enumerate(model.raw["features"]):
        code = feat["code"]
        lam = feat["boxcox_lambda"]
        value = x[code]
        if lam is not None and value <= 0 and lam != 0:
            raise PatientSpecError(
                "linage2_invalid_value",
                f"{code} = {value:g}: the model takes a power transform of this input "
                f"and needs it positive.", input=code,
            )
        t = _box_cox(value, lam)
        med, mad = sexb["median"][i], sexb["mad"][i]
        z = t if med is None else (t - med) / mad
        z = max(-z_max, min(z_max, z))
        years = (z - sexb["mu_z"][i]) * sexb["w_months_per_sd"][i] / 12.0
        total += years
        contributions.append({
            "feature": code,
            "contribution_years": years,
            "is_imputed": provenance[code] != MEASURED,
        })
    age_months = age * 12.0
    delta = total + (age_months - sexb["mu_age_months"]) * sexb["w_age"] / 12.0
    contributions.sort(key=lambda d: d["contribution_years"], reverse=True)

    lo, hi = model.training_ages(sex)
    if not lo <= age <= hi + 1:
        warnings.append(
            f"Age {age:g} is outside the ages LinAge2 was fitted on ({lo:g}-{hi:.0f}); "
            f"the result is an extrapolation."
        )
    if imputed_inputs:
        warnings.append(
            f"{len(imputed_inputs)} of {len(model.lab_inputs)} lab inputs were not given "
            f"and were filled with the median of same-sex NHANES 1999-2000 participants "
            f"aged about {age:.0f}."
        )
    assumed = [s for s in ("fs1Score", "fs2Score", "fs3Score") if provenance[s] == ASSUMED]
    if assumed:
        default = {"fs1Score": "no diagnoses (fs1Score)",
                   "fs2Score": "'good' self-rated health, unchanged from a year ago (fs2Score)",
                   "fs3Score": "no healthcare visits in the past year (fs3Score)"}
        warnings.append(
            "Questionnaire not answered for " + ", ".join(assumed) + ": assumed "
            + "; ".join(default[s] for s in assumed) + ", as the LinAge2 service does — "
            "flagged as not measured."
        )

    not_measured = sorted({c for c, p in provenance.items() if p != MEASURED})
    response = {
        "code": 200,
        "biological_age": age + delta,
        "message": "Prediction successful",
        "metadata": {
            "chronological_age": age,
            "delta_ba_ca": delta,
            "features_used": len(contributions),
            "total_features": len(contributions),
            "warnings": list(warnings) or None,
            "feature_contributions": contributions,
            "imputed_features": not_measured or None,
        },
    }
    return LinAge2Result(
        response=response,
        provenance=provenance,
        inputs_used={c: x[c] for c in model.lab_inputs},
        imputed_inputs=imputed_inputs,
        warnings=warnings,
    )
