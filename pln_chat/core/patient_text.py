"""A few lines of plain text about a person -> a patient. Fixed rules, no LLM.

    58 year old male, current smoker
    albumin 4.2 g/dL
    HbA1c 6.3 %
    CRP 1.2 mg/L
    blood pressure 138/86
    diagnoses: hypertension

`read_patient_text` returns what it READ — every value with the unit it was typed
in, the value in the unit LinAge2 takes, and a status — before anything is built.
That table is the point: a lab value in the wrong unit is the commonest way to get
a confident, wrong biological age (albumin 4.2 read as g/L instead of g/dL is
-30 g/L from the median, and LinAge2 turns it into ~10 years), so the reader is
built to refuse rather than guess:

* a name it does not know is listed as not understood, never matched loosely;
* a unit it does not know is refused, with the units it accepts;
* a value with NO unit is accepted only when exactly one known unit puts it inside
  the range NHANES 1999-2002 actually observed — and then the assumed unit is
  shown — otherwise it asks for the unit;
* a value outside that range is refused, naming the unit that would fit if one does.

`ParsedPatient.to_patient(...)` then scores LinAge2 (core.linage2_model) and returns
the ordinary caller-patient payload `/query`, `/metta/run` and `build_patient`
already accept — with the same values doubling as the knowledge base's own
witnesses (CRP, HbA1c, fasting glucose, smoking status), so a cause can be credited
without typing anything twice.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from core.linage2_model import compute_linage2, load_model
from core.patient_builder import MARKERS, Z_LIMIT, PatientSpecError

# ═══════════════════════════ units ═════════════════════════════════════════════
# Each LinAge2 lab input: its canonical (model) unit, and every unit accepted for
# it as (scale, offset) with model_value = typed * scale + offset. Factors are the
# standard clinical conversions; the canonical units are NHANES 1999-2002's, which
# tests/test_patient_text.py checks against the reference cohort's medians.

_COUNT = {"e9/l": (1.0, 0.0), "e3/ul": (1.0, 0.0), "k/ul": (1.0, 0.0), "thou/ul": (1.0, 0.0),
          "/nl": (1.0, 0.0), "/ul": (0.001, 0.0), "cells/ul": (0.001, 0.0)}
_PCT = {"%": (1.0, 0.0), "percent": (1.0, 0.0)}
_MMOL = {"mmol/l": (1.0, 0.0), "meq/l": (1.0, 0.0)}
_UL = {"u/l": (1.0, 0.0), "iu/l": (1.0, 0.0)}
_CHOL = {"mmol/l": (1.0, 0.0), "mg/dl": (0.02586, 0.0)}
_GPERL = {"g/l": (1.0, 0.0), "g/dl": (10.0, 0.0)}


@dataclass(frozen=True)
class LabSpec:
    code: str
    label: str
    unit: str                                  # the model's unit, for display
    units: dict                                # normalised unit -> (scale, offset)
    aliases: tuple[str, ...]


LABS: tuple[LabSpec, ...] = (
    LabSpec("BPXPLS", "Pulse", "beats/min", {"bpm": (1, 0), "beats/min": (1, 0), "/min": (1, 0)},
            ("pulse", "heart rate", "resting heart rate", "hr")),
    LabSpec("BPXSAR", "Systolic blood pressure", "mmHg", {"mmhg": (1, 0)}, ("systolic", "systolic bp", "sbp")),
    LabSpec("BPXDAR", "Diastolic blood pressure", "mmHg", {"mmhg": (1, 0)}, ("diastolic", "diastolic bp", "dbp")),
    LabSpec("BMXBMI", "Body mass index", "kg/m²", {"kg/m2": (1, 0)}, ("bmi", "body mass index")),
    LabSpec("URXUMASI", "Urine albumin", "mg/L", {"mg/l": (1, 0), "ug/ml": (1, 0), "mg/dl": (10, 0)},
            ("urine albumin", "urinary albumin", "microalbumin", "urine microalbumin")),
    LabSpec("URXUCRSI", "Urine creatinine", "µmol/L", {"umol/l": (1, 0), "mmol/l": (1000, 0), "mg/dl": (88.4, 0)},
            ("urine creatinine", "urinary creatinine")),
    LabSpec("LBDIRNSI", "Serum iron", "µmol/L", {"umol/l": (1, 0), "ug/dl": (0.179, 0)}, ("iron", "serum iron")),
    LabSpec("LBDTIBSI", "TIBC", "µmol/L", {"umol/l": (1, 0), "ug/dl": (0.179, 0)},
            ("tibc", "total iron binding capacity", "iron binding capacity")),
    LabSpec("LBXPCT", "Transferrin saturation", "%", _PCT,
            ("transferrin saturation", "tsat", "iron saturation")),
    LabSpec("LBDFERSI", "Ferritin", "µg/L", {"ug/l": (1, 0), "ng/ml": (1, 0)}, ("ferritin",)),
    LabSpec("LBDFOLSI", "Folate", "nmol/L", {"nmol/l": (1, 0), "ng/ml": (2.266, 0), "ug/l": (2.266, 0)},
            ("folate", "serum folate", "folic acid")),
    LabSpec("LBDB12SI", "Vitamin B12", "pmol/L", {"pmol/l": (1, 0), "pg/ml": (0.738, 0), "ng/l": (0.738, 0)},
            ("b12", "vitamin b12", "cobalamin")),
    LabSpec("LBDTCSI", "Total cholesterol", "mmol/L", _CHOL, ("total cholesterol", "cholesterol", "tc")),
    LabSpec("LBDHDLSI", "HDL cholesterol", "mmol/L", _CHOL, ("hdl", "hdl cholesterol", "hdl-c")),
    LabSpec("LBDSTRSI", "Triglycerides", "mmol/L", {"mmol/l": (1, 0), "mg/dl": (0.01129, 0)},
            ("triglycerides", "triglyceride", "tg", "trigs")),
    LabSpec("LDLV", "LDL cholesterol", "mmol/L", _CHOL, ("ldl", "ldl cholesterol", "ldl-c")),
    LabSpec("LBXWBCSI", "White blood cells", "10⁹/L", _COUNT,
            ("wbc", "white blood cells", "white blood cell count", "white cells", "leukocytes")),
    LabSpec("LBXLYPCT", "Lymphocytes %", "%", _PCT, ("lymphocytes", "lymphs", "lymphocyte")),
    LabSpec("LBXMOPCT", "Monocytes %", "%", _PCT, ("monocytes", "monos", "monocyte")),
    LabSpec("LBXNEPCT", "Neutrophils %", "%", _PCT, ("neutrophils", "neuts", "neutrophil")),
    LabSpec("LBXEOPCT", "Eosinophils %", "%", _PCT, ("eosinophils", "eos", "eosinophil")),
    LabSpec("LBXBAPCT", "Basophils %", "%", _PCT, ("basophils", "basos", "basophil")),
    LabSpec("LBDLYMNO", "Lymphocytes (count)", "10⁹/L", _COUNT, ("lymphocytes", "lymphs", "lymphocyte")),
    LabSpec("LBDMONO", "Monocytes (count)", "10⁹/L", _COUNT, ("monocytes", "monos", "monocyte")),
    LabSpec("LBDNENO", "Neutrophils (count)", "10⁹/L", _COUNT, ("neutrophils", "neuts", "neutrophil")),
    LabSpec("LBDEONO", "Eosinophils (count)", "10⁹/L", _COUNT, ("eosinophils", "eos", "eosinophil")),
    LabSpec("LBDBANO", "Basophils (count)", "10⁹/L", _COUNT, ("basophils", "basos", "basophil")),
    LabSpec("LBXRBCSI", "Red blood cells", "10¹²/L",
            {"e12/l": (1, 0), "e6/ul": (1, 0), "m/ul": (1, 0), "mil/ul": (1, 0), "million/ul": (1, 0)},
            ("rbc", "red blood cells", "red blood cell count", "red cells", "erythrocytes")),
    LabSpec("LBXHGB", "Hemoglobin", "g/dL", {"g/dl": (1, 0), "g/l": (0.1, 0), "mmol/l": (1.611, 0)},
            ("hemoglobin", "haemoglobin", "hgb", "hb")),
    LabSpec("LBXHCT", "Hematocrit", "%", {"%": (1, 0), "percent": (1, 0), "l/l": (100, 0)},
            ("hematocrit", "haematocrit", "hct")),
    LabSpec("LBXMCVSI", "MCV", "fL", {"fl": (1, 0)}, ("mcv", "mean corpuscular volume")),
    LabSpec("LBXMCHSI", "MCH", "pg", {"pg": (1, 0)}, ("mch", "mean corpuscular hemoglobin")),
    LabSpec("LBXMC", "MCHC", "g/dL", {"g/dl": (1, 0), "g/l": (0.1, 0)},
            ("mchc", "mean corpuscular hemoglobin concentration")),
    LabSpec("LBXRDW", "RDW", "%", _PCT, ("rdw", "red cell distribution width", "rdw-cv")),
    LabSpec("LBXPLTSI", "Platelets", "10⁹/L", _COUNT, ("platelets", "platelet count", "plt")),
    LabSpec("LBXMPSI", "MPV", "fL", {"fl": (1, 0)}, ("mpv", "mean platelet volume")),
    LabSpec("LBXCRP", "C-reactive protein", "mg/dL", {"mg/dl": (1, 0), "mg/l": (0.1, 0)},
            ("crp", "c-reactive protein", "c reactive protein", "hs-crp", "hscrp", "hs crp")),
    LabSpec("LBXGH", "HbA1c", "%", {"%": (1, 0), "percent": (1, 0), "mmol/mol": (0.09148, 2.152)},
            ("hba1c", "a1c", "hemoglobin a1c", "haemoglobin a1c", "glycated hemoglobin",
             "glycohemoglobin")),
    LabSpec("SSBNP", "NT-proBNP", "pg/mL", {"pg/ml": (1, 0), "ng/l": (1, 0), "pmol/l": (8.457, 0)},
            ("nt-probnp", "ntprobnp", "nt probnp", "nt-pro-bnp", "probnp")),
    LabSpec("LBDSALSI", "Albumin", "g/L", _GPERL, ("albumin", "serum albumin")),
    LabSpec("LBXSATSI", "ALT", "U/L", _UL, ("alt", "sgpt", "alanine aminotransferase")),
    LabSpec("LBXSASSI", "AST", "U/L", _UL, ("ast", "sgot", "aspartate aminotransferase")),
    LabSpec("LBXSAPSI", "Alkaline phosphatase", "U/L", _UL, ("alp", "alkaline phosphatase", "alk phos")),
    LabSpec("LBDSBUSI", "Urea nitrogen (BUN)", "mmol/L", {"mmol/l": (1, 0), "mg/dl": (0.357, 0)},
            ("bun", "urea nitrogen", "blood urea nitrogen")),
    # Urea itself: the same molar amount as urea nitrogen in mmol/L, but in mg/dL it
    # weighs 60.06/28.02 times more (BUN mg/dL x 0.357 vs urea mg/dL x 0.1665).
    LabSpec("LBDSBUSI", "Urea", "mmol/L", {"mmol/l": (1, 0), "mg/dl": (0.1665, 0)},
            ("urea", "serum urea", "blood urea")),
    LabSpec("LBDSCASI", "Calcium", "mmol/L", {"mmol/l": (1, 0), "mg/dl": (0.2495, 0)}, ("calcium", "ca")),
    LabSpec("LBXSC3SI", "Bicarbonate", "mmol/L", _MMOL, ("bicarbonate", "co2", "total co2", "hco3")),
    LabSpec("LBDSGLSI", "Glucose", "mmol/L", {"mmol/l": (1, 0), "mg/dl": (0.0555, 0)},
            ("glucose", "blood glucose", "serum glucose", "random glucose", "plasma glucose",
             "fasting glucose", "fasting blood glucose", "fasting plasma glucose", "fbg", "fpg")),
    LabSpec("LBXSLDSI", "LDH", "U/L", _UL, ("ldh", "lactate dehydrogenase")),
    LabSpec("LBDSPHSI", "Phosphorus", "mmol/L", {"mmol/l": (1, 0), "mg/dl": (0.3229, 0)},
            ("phosphorus", "phosphate")),
    LabSpec("LBDSTBSI", "Total bilirubin", "µmol/L", {"umol/l": (1, 0), "mg/dl": (17.1, 0)},
            ("bilirubin", "total bilirubin")),
    LabSpec("LBDSTPSI", "Total protein", "g/L", _GPERL, ("total protein", "protein")),
    LabSpec("LBDSUASI", "Uric acid", "µmol/L", {"umol/l": (1, 0), "mg/dl": (59.48, 0)}, ("uric acid", "urate")),
    LabSpec("LBDSCRSI", "Creatinine", "µmol/L", {"umol/l": (1, 0), "mg/dl": (88.4, 0)},
            ("creatinine", "serum creatinine")),
    LabSpec("LBXSNASI", "Sodium", "mmol/L", _MMOL, ("sodium", "na")),
    LabSpec("LBXSKSI", "Potassium", "mmol/L", _MMOL, ("potassium", "k")),
    LabSpec("LBXSCLSI", "Chloride", "mmol/L", _MMOL, ("chloride", "cl")),
    LabSpec("LBDSGBSI", "Globulin", "g/L", _GPERL, ("globulin",)),
)
SPECS: dict[str, LabSpec] = {}
for _spec in LABS:
    SPECS.setdefault(_spec.code, _spec)          # urea and BUN share an input; BUN names it
_FASTING_ALIASES = {"fasting glucose", "fasting blood glucose", "fasting plasma glucose", "fbg", "fpg"}

#: LDL is not an NHANES variable (LinAge2 derives it); a plausible clinical range.
_LDL_RANGE = (0.3, 10.0)

_ALIAS_INDEX: list[tuple[str, list[LabSpec]]] = []
for _spec in LABS:
    for _alias in _spec.aliases:
        for _entry in _ALIAS_INDEX:
            if _entry[0] == _alias:
                _entry[1].append(_spec)
                break
        else:
            _ALIAS_INDEX.append((_alias, [_spec]))
_ALIAS_INDEX.sort(key=lambda e: -len(e[0]))           # longest alias wins


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def normalise_unit(text: str) -> str:
    u = text.strip().lower().translate(_SUPERSCRIPT)
    for micro in ("µ", "μ", "mc"):
        u = u.replace(micro, "u")
    u = re.sub(r"\bper\b", "/", u)                 # "beats per min"; never inside "percent"
    u = u.replace("×", "x").replace(" ", "").rstrip(".")
    u = re.sub(r"^[x*]", "", u)
    u = re.sub(r"10\^?(?:e)?(\d+)", r"e\1", u)
    u = u.replace("liter", "l").replace("litre", "l")
    if u in ("beats/minute", "beats/min", "/minute"):
        return "bpm" if u != "/minute" else "/min"
    return u


_DISPLAY_UNIT = {
    "bpm": "bpm", "beats/min": "beats/min", "/min": "/min", "mmhg": "mmHg", "kg/m2": "kg/m²",
    "mg/l": "mg/L", "ug/ml": "µg/mL", "mg/dl": "mg/dL", "umol/l": "µmol/L", "mmol/l": "mmol/L",
    "ug/dl": "µg/dL", "%": "%", "percent": "%", "ug/l": "µg/L", "ng/ml": "ng/mL",
    "nmol/l": "nmol/L", "pmol/l": "pmol/L", "pg/ml": "pg/mL", "ng/l": "ng/L", "e9/l": "10⁹/L",
    "e3/ul": "10³/µL", "k/ul": "K/µL", "thou/ul": "thou/µL", "/nl": "/nL", "/ul": "/µL",
    "cells/ul": "cells/µL", "e12/l": "10¹²/L", "e6/ul": "10⁶/µL", "m/ul": "M/µL",
    "mil/ul": "mil/µL", "million/ul": "million/µL", "g/dl": "g/dL", "g/l": "g/L", "l/l": "L/L",
    "fl": "fL", "pg": "pg", "mmol/mol": "mmol/mol", "u/l": "U/L", "iu/l": "IU/L", "meq/l": "mEq/L",
}


def display_unit(normalised: str) -> str:
    """A normalised unit key as a person writes it (mg/l -> mg/L)."""
    return _DISPLAY_UNIT.get(normalised, normalised)


# ═══════════════════════════ what was read ═════════════════════════════════════

OK, ASSUMED_UNIT, NEEDS_UNIT, UNKNOWN_UNIT, OUT_OF_RANGE, DUPLICATE = (
    "ok", "unit assumed", "needs a unit", "unknown unit", "out of range", "duplicate")
_BLOCKING = {NEEDS_UNIT, UNKNOWN_UNIT, OUT_OF_RANGE, DUPLICATE}


@dataclass
class Reading:
    code: str
    label: str
    typed: str                                 # the statement as written
    value_typed: float
    unit_typed: Optional[str]
    value: Optional[float] = None              # in the model's unit
    unit: str = ""
    status: str = OK
    note: str = ""
    fasting: bool = False

    @property
    def blocking(self) -> bool:
        return self.status in _BLOCKING

    def as_dict(self) -> dict:
        return {
            "code": self.code, "label": self.label, "typed": self.typed,
            "value": None if self.value is None else round(self.value, 6),
            "unit": self.unit, "status": self.status, "note": self.note,
        }


@dataclass
class ParsedPatient:
    age: Optional[float] = None
    sex: Optional[str] = None
    smoking: Optional[str] = None              # NeverSmoker | FormerSmoker | CurrentSmoker
    cotinine_level: Optional[int] = None
    cotinine_note: str = ""
    readings: list[Reading] = field(default_factory=list)
    questionnaire: dict[str, int] = field(default_factory=dict)
    questionnaire_notes: list[str] = field(default_factory=list)
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    #: knowledge-base markers that are not LinAge2 inputs (a GrimAge acceleration)
    extra_markers: dict[str, dict] = field(default_factory=dict)
    #: statements about someone else (or second-hand smoke), not used
    set_aside: list[str] = field(default_factory=list)
    #: set by kb_markers(): a witness value the KB's coarse reference cannot hold
    witness_notes: list[str] = field(default_factory=list)
    not_understood: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not any(r.blocking for r in self.readings)

    def all_problems(self) -> list[str]:
        out = list(self.problems)
        out += [f"{r.label}: {r.note}" for r in self.readings if r.blocking]
        return out

    def labs(self) -> dict[str, float]:
        """NHANES code -> value in the model's unit, for compute_linage2."""
        labs = {r.code: r.value for r in self.readings if not r.blocking and r.value is not None}
        if self.cotinine_level is not None:
            labs["LBXCOT"] = float(self.cotinine_level)
        if "BMXBMI" not in labs and self.weight_kg and self.height_cm:
            labs["BMXBMI"] = self.weight_kg / (self.height_cm / 100.0) ** 2
        return labs

    def kb_markers(self) -> dict[str, dict]:
        """The same values as the knowledge base's own witnesses, in ITS units:
        CRP mg/L, HbA1c %, fasting glucose mg/dL (only when the text said fasting).

        The KB standardises these against coarse pooled priors, and refuses |z| > 12 as
        a unit mistake — which a real HbA1c of 12 % or a fasting glucose of 250 mg/dL
        exceeds. A witness only has to say Elevated, so such a value is passed as the
        largest z the KB accepts, and `witness_notes` says so; LinAge2 still gets the
        value as typed."""
        by_code = {r.code: r for r in self.readings if not r.blocking and r.value is not None}
        raw: dict[str, tuple[float, str]] = {}
        if "LBXCRP" in by_code:
            raw["CRP"] = (round(by_code["LBXCRP"].value * 10.0, 6), "mg/L")
        if "LBXGH" in by_code:
            raw["HbA1c"] = (round(by_code["LBXGH"].value, 6), "%")
        glucose = by_code.get("LBDSGLSI")
        if glucose is not None and glucose.fasting:
            raw["FastingGlucose"] = (round(glucose.value / 0.0555, 4), "mg/dL")
        markers: dict[str, dict] = {}
        self.witness_notes = []
        for name, (value, unit) in raw.items():
            spec = MARKERS.get(name)
            z = spec.reference.to_z(value)[0] if spec is not None and spec.reference else 0.0
            if abs(z) > Z_LIMIT:
                capped = math.copysign(Z_LIMIT, z)
                markers[name] = {"z": capped}
                self.witness_notes.append(
                    f"{name} {value:g} {unit} lies beyond the knowledge base's coarse reference "
                    f"(z {z:.1f}); passed to it as z {capped:g}, which is all a witness needs "
                    f"(Elevated). LinAge2 uses the value as typed.")
            else:
                markers[name] = {"value": value, "unit": unit}
        markers.update(self.extra_markers)
        return markers

    def to_patient(self, patient_id: str = "Me", extra_markers: Optional[dict] = None) -> tuple[dict, object]:
        """The caller-patient payload (with its LinAge2 block) and the LinAge2 result.

        Raises PatientSpecError when anything read is unusable — the read table says
        what — or when LinAge2 cannot be scored (it needs age and sex)."""
        problems = self.all_problems()
        if problems:
            raise PatientSpecError("patient_text_unusable",
                                   "Fix these before building: " + " | ".join(problems),
                                   problems=problems)
        result = compute_linage2(sex=self.sex, age=self.age, labs=self.labs(),
                                 questionnaire=self.questionnaire)
        payload: dict = {"id": patient_id, "age": self.age, "sex": self.sex,
                         "markers": {**self.kb_markers(), **(extra_markers or {})},
                         "linage2": result.response}
        if self.smoking:
            payload["smoking"] = self.smoking
        return payload, result

    def as_dict(self) -> dict:
        return {
            "age": self.age, "sex": self.sex, "smoking": self.smoking,
            "cotinine_level": self.cotinine_level,
            "readings": [r.as_dict() for r in self.readings],
            "questionnaire": dict(self.questionnaire),
            "questionnaire_notes": list(self.questionnaire_notes),
            "weight_kg": self.weight_kg, "height_cm": self.height_cm,
            "not_understood": list(self.not_understood),
            "set_aside": list(self.set_aside),
            "witness_notes": list(self.witness_notes),
            "problems": self.all_problems(),
            "notes": list(self.notes),
            "ok": self.ok,
        }


# ═══════════════════════════ the reader ════════════════════════════════════════

_NUM = r"(\d+(?:\.\d+)?|\.\d+)"
#: Only an explicit age phrase is an age: "58 year old", "58-year-old", "58 yo",
#: "age 58" / "aged 58" at the start, or "58 years" as a whole statement. A bare
#: "N years" elsewhere ("quit smoking 20 years ago", "diabetic for 5 years") is not.
_AGE_RES = (
    re.compile(r"\b(\d{1,3})\s*(?:-|\s)?(?:years?|yrs?)(?:\s*|-)old\b"),
    re.compile(r"\b(\d{1,3})\s*(?:-|\s)?(?:y/?o|year-old|years-old)\b"),
    re.compile(r"^(?:i'?m\s+|i am\s+|my\s+)?(?:age|aged)\s*(?:is|:|=)?\s*(\d{1,3})\b"),
    re.compile(r"^(\d{1,3})\s*(?:years?|yrs?)$"),
)
_SEX_RES = (
    (re.compile(r"\b(?:female|woman|lady)\b|\bsex\s*[:=]\s*f\b"), "Female"),
    (re.compile(r"\b(?:male|man|gentleman)\b|\bsex\s*[:=]\s*m\b"), "Male"),
)
#: A statement about somebody else — or about smoke the person did not smoke — is set
#: aside whole: "my husband smokes", "male partner", "family history of diabetes".
_SOMEONE_ELSE = re.compile(
    r"\b(?:husband|wife|partner|spouse|boyfriend|girlfriend|father|mother|dad|mum|mom|parents?|"
    r"son|daughter|child|children|kids?|brother|sister|siblings?|friends?|room-?mate|colleague|"
    r"co-?worker|family|grand(?:mother|father|parent)s?|uncle|aunt|cousin)\b"
    r"|second[- ]?hand|passive(?:ly)? smok")
#: (pattern, status, cotinine level). Order matters: negated and past before current.
_SMOKING = (
    (r"\bnever[- ]?(?:a\s+)?smok\w*|\bnon[- ]?smok\w*|\b(?:do|does)(?:n'?t| not) smoke\b"
     r"|\bnot an? smoker\b|\bno smoking\b|\bsmoke[- ]free\b"
     r"|\bsmok(?:ing|er|es)\s*[:=]\s*(?:no|never|none|n)\b",
     "NeverSmoker", 0),
    # still smoking, whatever the sentence says about quitting
    (r"\b(?:trying|want(?:s|ing)?|plan(?:s|ning)?|hoping|going) to (?:quit|stop)\b",
     "CurrentSmoker", 3),
    (r"\b(?:former|ex|previous|past|reformed)[- ]?(?:\w+[- ])?smoker\b"
     r"|\b(?:quit|stopped|gave up) smoking\b|\bused to smoke\b|\bsmok(?:ed|er)(?= until\b)"
     r"|\bsmok(?:ing|er)\s*[:=]\s*(?:former|ex|past|quit)\b",
     "FormerSmoker", 0),
    (r"\b(?:light|occasional|social)[- ]?smoker\b|\bsmokes? (?:some|a few) days\b|\bsmokes? occasionally\b",
     "CurrentSmoker", 1),
    (r"\bmoderate[- ]?smoker\b", "CurrentSmoker", 2),
    (r"\b(?:heavy|daily|current)[- ]?smoker\b|\bsmokes? (?:daily|every day)\b|\bsmoker\b|\bi smoke\b"
     r"|\bsmokes\b|\bsmok(?:ing|er|es)\s*[:=]\s*(?:yes|current|daily|y)\b", "CurrentSmoker", 3),
)
_FILLER = re.compile(r"\b(?:i'm|im|i am|a|an|the|i|am|is|who|and|old|patient|person|me|my|year|years|yo|"
                     r"aged?|sex|smoking|currently|current|status)\b|[,.:;=/\-()']")
_DURATION = re.compile(r"\b\d+\s*(?:years?|yrs?|months?)\s+ago\b|\b(?:since|until|in)\s+\d{4}\b"
                       r"|\bfor\s+(?:the\s+(?:last|past)\s+)?\d+\s*(?:years?|yrs?|months?)\b")
_NEGATION = re.compile(r"^(?:no|not|never|without|denies|denied|negative for|free of|nor)\b\s*")

#: diagnosis words -> NHANES item (1 = yes). DIQ010 3 = borderline.
_DIAGNOSES: tuple[tuple[str, str, int], ...] = (
    (r"pre-?diabet\w*|borderline diabet\w*", "DIQ010", 3),
    (r"diabet\w*", "DIQ010", 1),
    (r"hypertension|high blood pressure", "BPQ020", 1),
    (r"chronic kidney disease|kidney disease|ckd|weak kidneys|failing kidneys", "KIQ020", 1),
    (r"asthma", "MCQ010", 1),
    (r"an(?:a)?emia", "MCQ053", 1),
    (r"arthritis", "MCQ160A", 1),
    (r"heart failure|chf", "MCQ160B", 1),
    (r"coronary (?:heart|artery) disease|chd|cad", "MCQ160C", 1),
    (r"angina", "MCQ160D", 1),
    (r"heart attack|myocardial infarction", "MCQ160E", 1),
    (r"stroke", "MCQ160F", 1),
    (r"emphysema|copd", "MCQ160G", 1),
    (r"thyroid\w*(?: disease| problem| condition)?|hypothyroid\w*|hyperthyroid\w*", "MCQ160I", 1),
    (r"obes\w*|overweight", "MCQ160J", 1),
    (r"chronic bronchitis", "MCQ160K", 1),
    (r"fatty liver|liver disease|liver condition|cirrhosis|hepatitis", "MCQ160L", 1),
    (r"cancer|malignan\w*", "MCQ220", 1),
    (r"hip fracture|broken hip|fractured hip", "OSQ010A", 1),
    (r"wrist fracture|broken wrist|fractured wrist", "OSQ010B", 1),
    (r"spine fracture|spinal fracture|vertebral fracture|broken spine|fractured spine", "OSQ010C", 1),
    (r"osteoporosis", "OSQ060", 1),
    (r"memory problems?|memory loss|confusion", "PFQ056", 1),
    (r"hospitali[sz]ed|overnight (?:in )?hospital|hospital stay", "HUQ070", 1),
)
_ITEM_LABEL = {
    "BPQ020": "hypertension", "DIQ010": "diabetes", "KIQ020": "kidney disease", "MCQ010": "asthma",
    "MCQ053": "anemia", "MCQ160A": "arthritis", "MCQ160B": "heart failure",
    "MCQ160C": "coronary heart disease", "MCQ160D": "angina", "MCQ160E": "heart attack",
    "MCQ160F": "stroke", "MCQ160G": "emphysema", "MCQ160I": "thyroid disease", "MCQ160J": "obesity",
    "MCQ160K": "chronic bronchitis", "MCQ160L": "liver condition", "MCQ220": "cancer",
    "OSQ010A": "hip fracture", "OSQ010B": "wrist fracture", "OSQ010C": "spine fracture",
    "OSQ060": "osteoporosis", "PFQ056": "memory problems", "HUQ070": "overnight hospital stay",
}
_FS1_ITEMS = ("BPQ020", "DIQ010", "KIQ020", "MCQ010", "MCQ053", "MCQ160A", "MCQ160B", "MCQ160C",
              "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K",
              "MCQ160L", "MCQ220", "OSQ010A", "OSQ010B", "OSQ010C", "OSQ060", "PFQ056", "HUQ070")
_NO_CONDITIONS = re.compile(r"^(?:i have\s+)?no (?:known |chronic |medical |other )?(?:conditions?|"
                            r"diagnos[ie]s|diseases?|medical history|health problems)\b"
                            r"(?:\s*,?\s*(?:except|apart from|other than|besides|but)\s+(.+))?$")
_DIAG_HEADER = re.compile(r"^(?:diagnos[ie]s|known conditions|conditions?|medical history|history of|"
                          r"history|has|with|i have)\b\s*[:=]?\s*", re.I)
_HEALTH = {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}
_TREND = {"better": 1, "worse": 2, "same": 3, "about the same": 3, "unchanged": 3}


def _visits_category(n: int) -> int:
    """HUQ050's answer categories: 0, 1, 2-3, 4-9, 10-12, 13+."""
    return 0 if n <= 0 else 1 if n == 1 else 2 if n <= 3 else 3 if n <= 9 else 4 if n <= 12 else 5


def _cotinine_level(ng_ml: float) -> int:
    """`digiCot`, the binning the model was trained on."""
    return 0 if ng_ml < 10 else 1 if ng_ml < 100 else 2 if ng_ml < 200 else 3


def _statements(text: str) -> list[str]:
    out: list[str] = []
    for line in (text or "").splitlines():
        for seg in line.split(";"):
            seg = seg.strip().strip("-•*").strip()
            if not seg:
                continue
            if _DIAG_HEADER.match(seg) and not re.match(r"^(?:has|history of|with)\b", seg, re.I):
                out.append(seg)                     # "diagnoses: a, b, c" stays whole
            else:
                out.extend(p.strip() for p in re.split(r",\s+(?=[A-Za-z])", seg) if p.strip())
    return out


def _range(code: str) -> tuple[float, float]:
    """Every value NHANES 1999-2002 observed: outside it, a typed unit is wrong."""
    return _LDL_RANGE if code == "LDLV" else load_model().nhanes_range(code)


def _central(code: str) -> tuple[float, float]:
    """The middle 99% of adults: what a value with no unit is checked against."""
    if code == "LDLV":
        return (1.0, 6.5)
    lo, _, hi = load_model().raw["nhanes_central"][code]
    return float(lo), float(hi)


def _resolve(specs: list[LabSpec], typed: str, value: float, unit_text: str,
             fasting: bool) -> Reading:
    """Pick the input and unit a typed value means, or say why not."""
    unit_text = unit_text.strip()
    if unit_text:
        u = normalise_unit(unit_text)
        matches = [s for s in specs if u in s.units]
        if not matches:
            spec = specs[0]
            accepted = sorted({display_unit(k) for s in specs for k in s.units})
            return Reading(spec.code, spec.label, typed, value, unit_text, status=UNKNOWN_UNIT,
                           note=f"'{unit_text}' is not a unit I know for {spec.label.lower()}; "
                                f"use one of: {', '.join(accepted)}")
        spec = matches[0]
        scale, offset = spec.units[u]
        v = value * scale + offset
        lo, hi = _range(spec.code)
        reading = Reading(spec.code, spec.label, typed, value, unit_text, v, spec.unit, fasting=fasting)
        if not lo <= v <= hi:
            fits = [k for k, (s, o) in spec.units.items() if k != u and lo <= value * s + o <= hi]
            reading.status = OUT_OF_RANGE
            reading.note = (f"{value:g} {unit_text} = {v:g} {spec.unit}, outside the {lo:g}-{hi:g} "
                            f"{spec.unit} seen in NHANES 1999-2002"
                            + (f"; did you mean {' or '.join(display_unit(f) for f in fits)}?"
                               if fits else ""))
        return reading

    # No unit. A reading counts as plausible in the unit the model takes if it is inside
    # everything NHANES observed, and in any OTHER unit only if it lands in the typical
    # adult range (0.5th-99.5th percentile). So an abnormal value typed in the usual
    # unit (hemoglobin 9.5, anaemic) is never quietly re-read as a normal value in
    # another unit: that leaves two plausible readings, and the reader asks. Exactly
    # one plausible reading -> take it (flagged if it is not the model's unit).
    options: dict[tuple[str, float], tuple[LabSpec, str]] = {}
    for spec in specs:
        lo, hi = _range(spec.code)
        clo, chi = _central(spec.code)
        for k, (s_, o) in spec.units.items():
            v = value * s_ + o
            canonical = (s_, o) == (1, 0)
            if (lo <= v <= hi) if canonical else (clo <= v <= chi):
                options.setdefault((spec.code, round(v, 9)), (spec, k))
    if len(options) == 1:
        (code, v), (spec, k) = next(iter(options.items()))
        canonical = spec.units[k] == (1, 0)
        clo, chi = _central(spec.code)
        notes = []
        if not canonical:
            notes.append(f"no unit given; read as {display_unit(k)}, the only unit that gives "
                         f"a plausible value")
        elif not clo <= v <= chi:
            notes.append("unusual: outside the middle 99% of NHANES adults")
        return Reading(spec.code, spec.label, typed, value, None, v, spec.unit,
                       status=OK if canonical else ASSUMED_UNIT, note="; ".join(notes),
                       fasting=fasting)
    spec = specs[0]
    if not options:
        lo, hi = _range(spec.code)
        return Reading(spec.code, spec.label, typed, value, None, status=OUT_OF_RANGE,
                       note=f"{value:g} is implausible in every unit I know for {spec.label.lower()} "
                            f"(NHANES range {lo:g}-{hi:g} {spec.unit})")
    readings = sorted({f"{v:g} {s.unit} ({s.label.lower()}, if typed in {display_unit(k)})"
                       for (c, v), (s, k) in options.items()})
    return Reading(spec.code, spec.label, typed, value, None, status=NEEDS_UNIT,
                   note="no unit given and it could be " + "; or ".join(readings))


def _read_lab(stmt: str, low: str) -> Optional[Reading]:
    for alias, specs in _ALIAS_INDEX:
        m = re.match(rf"{re.escape(alias)}(?![a-z0-9])\s*(?:level|value)?\s*(?:[:=]|is|of|was)?\s*"
                     rf"{_NUM}\s*(.*)$", low)
        if m:
            value = float(m.group(1))
            unit_text = stmt[len(stmt) - len(m.group(2)):] if m.group(2) else ""
            return _resolve(specs, stmt, value, unit_text, fasting=alias in _FASTING_ALIASES)
    return None


def _read_diagnoses(text: str) -> Optional[list[tuple[str, int]]]:
    """'hypertension, no diabetes and prediabetes' -> [(item, answer)…], or None when
    any piece is not a diagnosis this reader knows (the statement is then not used)."""
    found: list[tuple[str, int]] = []
    for piece in re.split(r",|;|&|\band\b|\bor\b", text):
        piece = piece.strip(" .")
        if not piece:
            continue
        negated = bool(_NEGATION.match(piece))
        piece = _NEGATION.sub("", piece)
        # "hypertension for 25 years", "diabetes since 2015": the duration is not a diagnosis
        piece = re.sub(r"\b(?:for|since)\s+(?:\d+\s*(?:years?|yrs?|months?)|\d{4})\b", " ", piece)
        hits = []
        rest = piece
        for pattern, item, answer in _DIAGNOSES:
            if re.search(rf"\b(?:{pattern})\b", rest):
                hits.append((item, 2 if negated else answer))
                rest = re.sub(rf"\b(?:{pattern})\b", " ", rest)
        if not hits or re.sub(r"\s|\b(?:with|has|of|history|type [12]|diagnosed|known)\b", "", rest):
            return None
        found.extend(hits)
    return found or None


def read_patient_text(text: str) -> ParsedPatient:
    """Read `text` into a ParsedPatient. Never raises on content: everything that
    could not be used is reported on the result."""
    p = ParsedPatient()
    seen: dict[str, Reading] = {}

    def add(reading: Reading) -> None:
        prev = seen.get(reading.code)
        if prev is not None and not prev.blocking and not reading.blocking \
                and abs((prev.value or 0) - (reading.value or 0)) > 1e-9:
            reading.status = DUPLICATE
            reading.note = f"{reading.label} was already given as {prev.typed!r}"
        seen.setdefault(reading.code, reading)
        p.readings.append(reading)

    def set_once(attr: str, value, what: str, stmt: str) -> None:
        current = getattr(p, attr)
        if current is not None and current != value:
            p.problems.append(f"two different {what}s: {current} and {value} (from '{stmt}')")
        else:
            setattr(p, attr, value)

    for stmt in _statements(text):
        low = stmt.lower().strip()

        # ── about somebody else: set aside, and say so ────────────────────────
        if _SOMEONE_ELSE.search(low):
            p.set_aside.append(stmt)
            continue

        # ── a GrimAge result (before demographics: "4.5 years" is not an age) ─
        m = re.match(r"(?:grim\s*age|ageaccelgrim)(\s+age)?(\s+accel(?:eration)?)?\s*[:=]?\s*"
                     r"([+-]?)(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)?\s*$", low)
        if m:
            years = float(m.group(3) + m.group(4))
            if not (m.group(2) or m.group(3) or low.startswith("ageaccelgrim")):
                p.problems.append(f"'{stmt}' looks like a GrimAge clock AGE; give the acceleration "
                                  f"(clock age minus your age), e.g. 'GrimAge acceleration +4 years'")
            elif abs(years) > 30:
                p.problems.append(f"GrimAge acceleration {years:+g} years is outside anything the "
                                  f"clock produces (±30); check the value")
            else:
                p.extra_markers["AgeAccelGrim"] = {"value": years, "unit": "years"}
                p.notes.append(f"GrimAge acceleration {years:+g} years: the knowledge base's 10-year "
                               f"heart-disease model reads it; it is never combined with LinAge2")
            continue

        # ── demographics and smoking (may share one statement) ─────────────────
        rest = low
        for rx in _AGE_RES:
            m = rx.search(rest)
            if m:
                set_once("age", float(m.group(1)), "age", stmt)
                rest = rest[:m.start()] + " " + rest[m.end():]
                break
        for rx, sex in _SEX_RES:
            if rx.search(rest):
                set_once("sex", sex, "sex", stmt)
                rest = rx.sub(" ", rest)
                break
        for pattern, status, level in _SMOKING:
            if re.search(pattern, rest):
                set_once("smoking", status, "smoking status", stmt)
                if p.smoking == status:
                    p.cotinine_level = level
                    p.cotinine_note = (f"cotinine level {level} from '{stmt}' (training bins: 0 <10, "
                                       f"1 10-100, 2 100-200, 3 >=200 ng/mL)")
                rest = re.sub(pattern, " ", rest)
                break
        if rest != low:
            rest = _DURATION.sub(" ", rest)         # "quit smoking 20 years ago", "until 2015"
            leftover = re.sub(r"\s+", " ", _FILLER.sub(" ", rest)).strip()
            if not leftover:
                continue
            low = stmt = leftover               # e.g. "58 year old man with diabetes"

        # ── questionnaire ────────────────────────────────────────────────────
        m = _NO_CONDITIONS.match(low)
        if m:
            exceptions = _read_diagnoses(m.group(1)) if m.group(1) else []
            if exceptions is None:
                p.not_understood.append(stmt)
                continue
            for q in _FS1_ITEMS:
                p.questionnaire[q] = 2
            for item, answer in exceptions:
                p.questionnaire[item] = answer
            p.questionnaire_notes.append(
                "no known conditions" + (" except " + ", ".join(_ITEM_LABEL[i] for i, a in exceptions
                                                                 if a != 2) if exceptions else "")
                + ": every other diagnosis answered No")
            continue
        m = re.match(r"(?:self[- ]rated |general |overall )?health\s*(?:is|:|=)?\s*"
                     r"(excellent|very good|good|fair|poor)\b", low)
        if m:
            p.questionnaire["HUQ010"] = _HEALTH[m.group(1)]
            p.questionnaire_notes.append(f"self-rated health: {m.group(1)}")
            continue
        m = re.match(r"health (?:compared (?:to|with)|vs\.?|versus) (?:a|one|1) year ago\s*[:=]?\s*"
                     r"(better|worse|about the same|same|unchanged)\b", low) or \
            re.match(r"health (?:is )?(?:getting |got )?(better|worse)\b", low)
        if m:
            p.questionnaire["HUQ020"] = _TREND[m.group(1)]
            p.questionnaire_notes.append(f"health vs a year ago: {m.group(1)}")
            continue
        m = re.match(r"(?:healthcare|health care|doctor|medical|clinic) visits?"
                     r"(?: (?:last|in the (?:last|past)) year| per year| a year)?\s*[:=]?\s*(\d+)\b", low)
        if m:
            n = int(m.group(1))
            p.questionnaire["HUQ050"] = _visits_category(n)
            p.questionnaire_notes.append(f"healthcare visits {n} -> NHANES category "
                                         f"{_visits_category(n)}")
            continue
        diag_text = _DIAG_HEADER.sub("", low) if _DIAG_HEADER.match(low) else low
        found = _read_diagnoses(diag_text)
        if found:
            for item, answer in found:
                p.questionnaire[item] = answer
            for q in _FS1_ITEMS:
                p.questionnaire.setdefault(q, 2)
            yes = [_ITEM_LABEL[i] + (" (borderline)" if a == 3 else "") for i, a in found if a != 2]
            no = [_ITEM_LABEL[i] for i, a in found if a == 2]
            p.questionnaire_notes.append(
                "diagnoses: " + (", ".join(yes) if yes else "none")
                + (f"; not: {', '.join(no)}" if no else "")
                + " (anything not listed counts as No)")
            continue

        # ── blood pressure, weight, height ─────────────────────────────────────
        m = re.match(r"(?:blood pressure|bp)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(mm\s*hg)?\s*$", low)
        if m:
            add(_resolve([SPECS["BPXSAR"]], stmt, float(m.group(1)), "mmHg", False))
            add(_resolve([SPECS["BPXDAR"]], stmt, float(m.group(2)), "mmHg", False))
            continue
        m = re.match(rf"weight\s*[:=]?\s*{_NUM}\s*(kg|kgs|lb|lbs|pounds)?\s*$", low)
        if m:
            if not m.group(2):
                p.problems.append(f"'{stmt}': give the weight's unit (kg or lb)")
                continue
            w = float(m.group(1))
            w = w * 0.45359237 if m.group(2).startswith(("lb", "pound")) else w
            if not 25 <= w <= 350:
                p.problems.append(f"'{stmt}' = {w:.0f} kg, outside 25-350 kg; check the value and unit")
            else:
                set_once("weight_kg", w, "weight", stmt)
            continue
        m = re.match(rf"height\s*[:=]?\s*{_NUM}\s*(cm|m|in|inches)?\s*$", low)
        m2 = re.match(r"height\s*[:=]?\s*(\d)\s*'\s*(\d{1,2})\s*(?:\"|'')?\s*$", low)
        if m or m2:
            if m and not m.group(2):
                p.problems.append(f"'{stmt}': give the height's unit (cm, m or in, or 5'10\")")
                continue
            if m:
                h, unit = float(m.group(1)), m.group(2)
                h = h * 100 if unit == "m" else h * 2.54 if unit.startswith("in") else h
            else:
                h = (int(m2.group(1)) * 12 + int(m2.group(2))) * 2.54
            if not 100 <= h <= 230:
                p.problems.append(f"'{stmt}' = {h:.0f} cm, outside 100-230 cm; check the value and unit")
            else:
                set_once("height_cm", h, "height", stmt)
            continue

        # ── cotinine as a lab: ng/mL, or an explicit level ────────────────────
        m = re.match(rf"(?:serum )?cotinine\s*(level)?\s*[:=]?\s*{_NUM}\s*(ng/ml|ug/l|µg/l)?\s*$", low)
        if m:
            v = float(m.group(2))
            if m.group(3):
                level = _cotinine_level(v)
                note = f"cotinine {v:g} ng/mL -> level {level}"
            elif m.group(1) and v in (0, 1, 2, 3):
                level, note = int(v), f"cotinine level {int(v)} as given"
            else:
                p.problems.append(f"'{stmt}': give cotinine in ng/mL (e.g. 'cotinine 250 ng/mL') "
                                  f"or as 'cotinine level 0-3'")
                continue
            if p.cotinine_level is not None and p.cotinine_level != level and p.smoking is None:
                p.problems.append(f"two different cotinine levels: {p.cotinine_level} and {level}")
                continue
            p.cotinine_level, p.cotinine_note = level, note
            continue

        # ── a lab value ────────────────────────────────────────────────────────
        reading = _read_lab(stmt, low)
        if reading is not None:
            add(reading)
        else:
            p.not_understood.append(stmt)

    if p.age is None:
        p.problems.append("no age found (e.g. '58 year old' or 'age 58')")
    elif not 20 <= p.age <= 90:
        p.problems.append(f"age {p.age:g} is outside 20-90, the ages LinAge2's reference covers")
    if p.sex is None:
        p.problems.append("no sex found ('male' or 'female'): LinAge2 has a separate model for each")
    if p.cotinine_level is not None and p.smoking is None:
        p.notes.append("cotinine was given without a smoking status: the knowledge base credits "
                       "cotinine years to smoking only for a stated current smoker")
    if p.set_aside:
        p.notes.append("set aside (about someone else, or smoke you did not smoke): "
                       + "; ".join(f"'{s_}'" for s_ in p.set_aside))
    if p.weight_kg and p.height_cm and "BMXBMI" not in seen:
        bmi = p.weight_kg / (p.height_cm / 100) ** 2
        lo, hi = _range("BMXBMI")
        if not lo <= bmi <= hi:
            p.problems.append(f"BMI {bmi:.1f} from weight and height is outside the {lo:g}-{hi:g} "
                              f"NHANES observed; check them")
        else:
            p.notes.append(f"BMI {bmi:.1f} from weight and height")
    return p


EXAMPLES: dict[str, str] = {
    "58-year-old smoker": (
        "58 year old male, current smoker\n"
        "albumin 4.1 g/dL\n"
        "HbA1c 6.4 %\n"
        "CRP 3.1 mg/L\n"
        "fasting glucose 112 mg/dL\n"
        "blood pressure 142/88\n"
        "pulse 78\n"
        "total cholesterol 228 mg/dL\n"
        "HDL 41 mg/dL\n"
        "triglycerides 190 mg/dL\n"
        "creatinine 1.1 mg/dL\n"
        "RDW 14.1 %\n"
        "WBC 8.2\n"
        "BMI 29.4\n"
        "diagnoses: hypertension\n"
        "self-rated health: fair"
    ),
    "healthy 45-year-old woman": (
        "45 year old female, never smoked\n"
        "albumin 4.5 g/dL\n"
        "HbA1c 5.2 %\n"
        "CRP 0.6 mg/L\n"
        "fasting glucose 88 mg/dL\n"
        "blood pressure 112/72\n"
        "pulse 62\n"
        "total cholesterol 182 mg/dL\n"
        "HDL 68 mg/dL\n"
        "triglycerides 75 mg/dL\n"
        "creatinine 0.8 mg/dL\n"
        "RDW 12.6 %\n"
        "hemoglobin 13.4 g/dL\n"
        "platelets 245\n"
        "weight 61 kg\n"
        "height 168 cm\n"
        "no known conditions\n"
        "self-rated health: very good"
    ),
    "six labs only": (
        "age 66, male, former smoker\n"
        "albumin 3.8 g/dL\n"
        "HbA1c 7.1 %\n"
        "CRP 6.5 mg/L\n"
        "creatinine 1.3 mg/dL\n"
        "systolic 151\n"
        "NT-proBNP 410 pg/mL"
    ),
}
