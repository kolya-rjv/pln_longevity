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
from functools import lru_cache
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
#: Typography as keyboards and word processors make it, one character for one (so
#: offsets into a statement still hold): dashes, curly quotes, odd spaces.
_UNIFY = str.maketrans({**{c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"},
                        **{c: "'" for c in "\u2018\u2019\u201a\u201b\u2032\u02bc"},
                        **{c: '"' for c in "\u201c\u201d\u201e\u201f\u2033"},
                        **{c: " " for c in "\u00a0\u2002\u2003\u2007\u2009\u200a\u202f\u3000\t"}})


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


#: What a problem is, so a caller can tell a judgement ("cannot tell", "contradicts",
#: "about someone else", "vaping", a unit, a condition that would be lost) from a
#: statement that was simply not understood, or something missing altogether.
PROBLEM_KINDS = ("not_understood", "ambiguous", "contradiction", "someone_else", "vaping", "unit",
                 "lost_condition", "missing")


class Problem(str):
    """A problem, as the text the person sees — still a plain `str` everywhere it was one —
    carrying its kind, its topic and the statements (indexes into
    ParsedPatient.statements) it is about."""
    kind: str
    topic: str
    statements: tuple

    def __new__(cls, text: str, kind: str = "ambiguous", topic: str = "other", statements=()):
        obj = super().__new__(cls, text)
        obj.kind, obj.topic, obj.statements = kind, topic, tuple(statements)
        return obj


class Remark(str):
    """A statement that was not understood, or set aside: the text, and its statement."""
    statement: int

    def __new__(cls, text: str, statement: int = -1):
        obj = super().__new__(cls, text)
        obj.statement = statement
        return obj


@dataclass
class Statement:
    """One statement as the reader split the text: where it is, and what was read from it.

    `start`/`end` are offsets into line `line` of the text as typed. `facts` holds what the
    rules read from this statement alone (age, sex, smoking, labs, conditions, ...);
    `outcome` sums it up: read | partly_read | not_understood | refused | set_aside."""
    index: int
    line: int
    start: int
    end: int
    text: str
    facts: dict = field(default_factory=dict)
    outcome: str = "read"
    leftover: str = ""                         # the part that was not understood
    joined_to: int = -1                        # a smoking 'when' clause: the clause it modifies

    def as_dict(self) -> dict:
        return {"index": self.index, "line": self.line, "start": self.start, "end": self.end,
                "text": self.text, "outcome": self.outcome}


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
    statement: int = -1                        # index into ParsedPatient.statements

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
    #: the level came from a cotinine value, not from words — words never replace it
    cotinine_measured: bool = False
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
    #: every statement as the reader split the text, with what was read from each
    statements: list[Statement] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not any(r.blocking for r in self.readings)

    def all_problems(self) -> list[str]:
        """Every problem, as strings (each a Problem: .kind, .topic, .statements)."""
        out = list(self.problems)
        out += [Problem(f"{r.label}: {r.note}", "contradiction" if r.status == DUPLICATE else "unit",
                        "lab", (r.statement,) if r.statement >= 0 else ())
                for r in self.readings if r.blocking]
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
#: words before an age phrase that make it the age something happened at, or someone
#: else's, or a clock's — not the person's age now
_AGE_CONTEXT = re.compile(r"\b(?:since|at|when|until|till|from|by|after|before|diagnosed|dx|onset|was|were|became|"
                          r"retired|menopause|biological|bio|metabolic|epigenetic|grim\s*age|linage|pheno\s*age|"
                          r"clock|heart|lung|brain|child|son|daughter|husband|wife|partner|father|mother|born|"
                          r"quit|started|stopped|age of)\b")
_SEX_RES = (
    (re.compile(r"\b(?:female|woman|lady)\b|\bsex\s*[:=]\s*f\b"), "Female"),
    (re.compile(r"\b(?:male|man|gentleman)\b|\bsex\s*[:=]\s*m\b"), "Male"),
)
#: A statement about somebody else — or about smoke the person did not smoke — is set
#: aside whole: "my husband smokes", "male partner", "family history of diabetes".
_OTHERS = (r"husband|wife|partner|spouse|boyfriend|girlfriend|father|mother|dad|mum|mom|parents?|"
           r"son|daughter|child|children|kids?|brother|sister|siblings?|friends?|room-?mate|colleague|"
           r"co-?worker|family|grand(?:mother|father|parent)s?|uncle|aunt|cousin")
#: smoke the person did not smoke: set aside whatever else the statement says
_EXPOSURE = re.compile(rf"second[- ]?hand|passive(?:ly)? smok|\b(?:lives?|living|works?|working|stay(?:s|ing)?|"
                       rf"grew up|married to|surrounded by|around)\s+(?:with\s+)?(?:a\s+|other\s+|heavy\s+)?"
                       rf"smokers?\b|\b(?:lives?|living|works?|working|stay(?:s|ing)?|grew up)\s+with\s+"
                       rf"(?:(?:a|my|his|her|their)\s+)?(?:\w+\s+)?(?:{_OTHERS})\b|\b(?:{_OTHERS})\s+(?:who|that)\s+smok")
_SOMEONE_ELSE = re.compile(rf"\b(?:{_OTHERS})\b|{_EXPOSURE.pattern}")
#: ... and a statement whose subject is the other person: "my husband smokes", "but my
#: wife doesn't", "family history of diabetes". One that only mentions someone ("smokes
#: with friends", "diabetes like my mother") may be the person's own, and is asked about.
_OTHER_SUBJECT = re.compile(rf"^(?:(?:but|and|also|though|although|while|whereas)\s+)?"
                            rf"(?:(?:my|his|her|our|their|the|a|an)\s+)?(?:(?:step|grand|older|younger|late|ex)"
                            rf"[- ]?)?(?:{_OTHERS})\b")
#: Only a clause that is about smoking is read for a smoking status ("can't stop
#: snacking" is not a smoker). Cotinine level 0 vs 3 is about 8.8 years on the clock.
_SMOKING_TOPIC = re.compile(r"\bsmok\w*|\b(?:non|ex|chain)-?smok\w*|\bcigs?\b|\bcigar\w*|\btobacco\b|\bnicotin\w*"
                            r"|\bvap(?:e|es|ed|ing|er|ers)\b|\be-?cig\w*|\bpack[- ]?years?\b|\bpacks?\b"
                            r"|\bsnus\b|\bzyn\b|\bchewing tobacco\b|\bnicotine (?:patch\w*|gum|pouch\w*)")
_VAPING = re.compile(r"\bvap(?:e|es|ed|ing|er|ers)\b|\be-?cig\w*")
#: nicotine that is not smoked: raises cotinine, is not smoking to the knowledge base
_OTHER_NICOTINE = re.compile(r"\bvap\w*|\be-?cig\w*|\bjuul\w*|\bsnus\b|\bchew\w*|\bdip\b|\bzyn\b|\bpouch(?:es)?\b"
                             r"|\bpatch(?:es)?\b|\bnicotine (?:gum|lozenges?|spray|inhaler|replacement)\b|\bnrt\b")
#: smoked, but not tobacco: LinAge2 reads tobacco exposure
_CANNABIS = re.compile(r"\b(?:weed|cannabis|marijuana|marihuana|pot|joints?|cbd|thc|hash(?:ish)?|ganja|"
                       r"blunts?|spliffs?)\b")
#: how often, when it is not every day
_OCCASIONAL_WORDS = re.compile(r"\b(?:occasional(?:ly)?|rarely|seldom|sometimes|social(?:ly)?|weekends?|"
                               r"part(?:y|ies)|now and then|once in a while|a little|some ?days?|lightly|"
                               r"few|(?:a|per|each|every|/)\s*(?:week|month))\b")
#: (pattern, status, cotinine level, kind). Order matters: negated and past before
#: current; within an alternation, longer answers first. "generic" is a bare "smoker",
#: which a following "quit 2015" turns into a former smoker.
_SMOKING = (
    (r"\b(?:smok(?:ing|er|es)|tobacco(?: use)?|cigarettes?)(?: status| history| use)?\s*(?:[:=?]|-)\s*"
     r"(?:never smoker|never smoked|non-?smoker|none|never|no|n)\b(?!/|\s+longer)"
     r"|\bnever (?:been |was |were |once |ever )?(?:a\s+)?(?:smok\w*|used tobacco|used cigarettes)"
     r"|\bnever-?smok\w*|\bnon[- ]?smok\w*|\b(?:do|does|did)(?:n'?t| not) (?:ever )?smoke\b(?! any ?more)"
     r"|\bnot an? smoker\b|\bno smoking\b|\bsmoke[- ]free\b"
     r"|\bdenie[sd] (?:any )?(?:smoking|tobacco(?: use)?|cigarettes?)\b"
     r"|\bno (?:history of |prior )?(?:smoking|tobacco(?: use)?|cigarettes?)\b",
     "NeverSmoker", 0, "never"),
    # still smoking, whatever the clause says about quitting
    (r"\b(?:trying|want(?:s|ing)?|plan(?:s|ning)?|hoping|going) to (?:quit|stop)\b"
     r"|\b(?:can'?t|cannot|can not|unable to|fail(?:ed|s)? to|struggl\w* to) (?:quit|stop)\b",
     "CurrentSmoker", 3, "still"),
    (r"\b(?:smok(?:ing|er)|tobacco(?: use)?)(?: status| history)?\s*(?:[:=?]|-)\s*"
     r"(?:former smoker|ex-?smoker|former|ex|past|quit|no longer|stopped)\b"
     r"|\b(?:former|ex|previous|past|reformed|prior)[- ]?(?:\w+[- ])?smoker\b"
     r"|\b(?:quit|stopped|gave up) (?:smoking|cigarettes|tobacco)\b|\bused to (?:smoke|be an? (?:\w+ )?smoker)\b"
     r"|\bwas an? (?:\w+ )?smoker\b|\bno longer smok\w*|\b(?:don'?t|do not|doesn'?t|does not) smoke any ?more\b"
     r"|\bsmok(?:ed|er)(?= until\b)|\bsmok\w*\s*[(\-]?\s*(?:quit|stopped|gave up)\s+(?:in\s+)?"
     r"(?:(?:19|20)\d{2}|\d+\s*(?:years?|yrs?|months?)\s+ago)\b\s*\)?",
     "FormerSmoker", 0, "former"),
    (r"\b(?:smok(?:ing|er))(?: status)?\s*[:=]\s*(?:some days|occasional(?:ly)?|light)\b"
     r"|\b(?:light|occasional|social)[- ]?smoker\b|\bsmokes? (?:some|a few) days\b"
     r"|\bsmokes? (?:occasionally|socially|with friends|at parties|on weekends)\b",
     "CurrentSmoker", 1, "light"),
    (r"\bmoderate[- ]?smoker\b", "CurrentSmoker", 2, "moderate"),
    (r"\b(?:smok(?:ing|er|es)|tobacco(?: use)?)(?: status)?\s*[:=]\s*(?:current smoker|current|yes|daily|"
     r"every day|y)\b|\b(?:heavy|daily|current|chain)[- ]?smoker\b|\bsmokes? (?:daily|every day)\b"
     r"|\bi(?:'m| am) an? (?:\w+ )?smoker\b|\bi smoke\b|\bsmokes\b|\bsmoking(?= (?:since|for)\b)"
     r"|\b(?!0+\b)\d+\s*(?:cigarettes?|cigs?|packs?)\s*(?:a|per|each|/)\s*day\b|\b(?:half|one|two|a) packs? (?:a|per) day\b",
     "CurrentSmoker", 3, "current"),
    (r"\bsmoker\b", "CurrentSmoker", 3, "generic"),
)
#: What the rest of a smoking clause, or the clause after it, says about time.
_STRONG_PAST = re.compile(r"\b(?:quit|quitted|stopped|gave up|given up|until|till|no longer|any ?more|used to|"
                          r"former|ex|previous(?:ly)?|formerly|prior)\b"
                          r"|\b(?:19|20)\d{2}\s*(?:-|–|to)\s*(?:19|20)\d{2}\b")
_PAST_CUE = re.compile(_STRONG_PAST.pattern + r"|\b(?:was|were|ago|past|before|smoked|once|youth|college|"
                       r"university|school|ever|teens?|army|military|navy)\b")
_PRESENT_CUE = re.compile(r"\b(?:still|again|restarted|relaps\w*|back on|back to|now|current(?:ly)?|daily|"
                          r"every day|a day|per day|each day|a week|per week|as much|as often|as many|heavily|"
                          r"down to|cut(?:ting)? down|less|trying|want\w*|plans?|planning|hoping|going to|"
                          r"can'?t|cannot|unable|fail\w*|struggl\w*)\b|/\s*day\b")
_NEGATION_CUE = re.compile(r"\b(?:no|not|never|nope|nor|none|unknown|unsure|n/?a)\b|\bnot sure\b|n't\b"
                           r"|\b(?:dont|doesnt|didnt|isnt|wasnt|havent|hasnt|cant|wont)\b|\?")
_SINCE = re.compile(r"\bsince\s+(?:age\s+|i was\s+)?\d+\b|\bsince\s+(?:my\s+)?(?:teens?|childhood|school|"
                    r"college|university)\b|\bfor\s+(?:about\s+|over\s+)?\d+\s*(?:years?|yrs?|months?)\b"
                    r"|\b(?:started|began)\s+(?:at|aged?|when)\b[^,]*")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
#: A clause right after a smoking clause that only says when: "quit 2010", "still am",
#: "started again", "until my first child was born". Read before the someone-else
#: check, which would otherwise set "quit when my son was born" aside.
_MODIFIER = re.compile(
    r"^(?:(?:but|and|though|although|then)\s+)?(?:i\s+)?(?:have\s+|had\s+)?"
    r"(?P<cue>started again|restarted|relapsed|back on it|back on|back|still|trying to quit|"
    r"want(?:s|ing)? to quit|plan(?:s|ning)? to quit|hoping to quit|no plans to quit|"
    r"not (?:trying|planning) to quit|can'?t quit|cannot quit|unable to quit|"
    r"quit|stopped|gave up|given up|until|till|no longer|not any ?more|since|started)\b(?P<rest>.*)$")
_MODIFIER_REST = re.compile(r"^\s*(?:$|(?:smok\w*|cig\w*|tobacco|it|in|at|on|when|after|before|for|since|around|"
                            r"about|age|aged|the|a|an|my|last|this|that|again|am|do|is|ago|years?|months?|"
                            r"completely|cold|entirely|recently|back|times?|but|with)\b|\d|[()\-])")
_TEMPORARY_QUIT = re.compile(r"\b(?:quit\w*|stopped|gave up)\s+(?:smoking\s+)?for\s+(?:a|an|\d+|a few|several|"
                             r"some)\s+(?:days?|weeks?|months?|years?|while)\b|\bfor a while\b"
                             r"|\b(?:quit|stopped)\s+(?:\d+|a few|several|many)\s+times\b")
_STILL_KINDS = ("still", "light", "moderate", "current")
#: "stopped drinking", "gave up alcohol": a quit that is about something else
_OTHER_HABIT = re.compile(
    r"\b(?:quit\w*|stopped|gave up|given up|used to|relaps\w*)\s+(?!(?:smok|cig|tobacco|it\b|in\b|at\b|on\b|"
    r"when|after|before|for\b|since|cold|complet|entire|recent|again|last|this|years?\b|months?\b|ago\b|the\b|"
    r"a\b|an\b|about|around|over|almost|nearly|\d|one|two|three|four|five|six|seven|eight|nine|ten|twenty|"
    r"thirty|forty|fifty|several|many|few|some|yet|now|then|but|and|too|already|recently|last|just|"
    r"\W|$))[a-z]+")
#: a clause that only says the one before is uncertain
_UNSURE_ALONE = re.compile(r"^(?:but\s+|though\s+)?(?:(?:i'?m|i am)\s+)?(?:not sure|unsure|uncertain|maybe|"
                           r"possibly|probably|i think|can'?t remember|don'?t remember)\b")
_WHO_SMOKING = {("NeverSmoker", 0): "never smoked", ("FormerSmoker", 0): "former smoker",
                ("CurrentSmoker", 3): "current smoker", ("CurrentSmoker", 1): "occasional smoker",
                ("CurrentSmoker", 2): "moderate smoker"}
#: words in the clause (or line) after a smoking clause that change when or how much
_AFTER_SMOKING = re.compile(r"\b(?:quit|quitted|stopped|gave up|given up|relaps\w*|started again|restarted|back on|"
                            r"used to|no longer|any ?more|unknown|unsure|not sure|occasional(?:ly)?|rarely|"
                            r"socially|weekends?|now and then)\b|\bn/?a\b|\?")
#: Everything a read smoking clause may say besides its status; what is left after
#: removing it is read like any other statement ("58 year old male smoker no diabetes").
_SMOKING_DETAIL = re.compile(
    r"\ba day in (?:my|his|her|their) life\b|\b(?:smok\w*|cigs?|cigar\w*|tobacco|nicotin\w*|vap\w*|e-?cig\w*|packs?|"
    r"pack[- ]?years?|use[sd]?|user|status|history|quit|quitted|stopped|gave|given|up|until|till|since|ago|for|at|in|"
    r"on|age|aged|about|around|approx\w*|years?|yrs?|months?|weeks?|days?|daily|a|an|per|each|every|or|and|"
    r"but|with|friends|weekends?|parties|socially|occasionally|still|now|currently|current|heavy|light|social|"
    r"occasional|moderate|chain|former|ex|previous(?:ly)?|formerly|prior|past|reformed|was|were|used|to|be|been|i|"
    r"i'm|im|i'?ve|ve|am|is|have|"
    r"has|had|my|me|half|one|two|three|cold|turkey|when|then|last|this|year|started|began|teens?|"
    r"jan(?:uary)?|feb(?:ruary)?|march|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|times?|too|also|only|approximately)\b|\d+(?:\.\d+)?|[,.:;=/\-()'?~+–]")
#: "never vaped", "does not vape": says nothing about smoking, and agrees with no cotinine
_VAPE_NEGATION = re.compile(r"^(?:(?:but|and|also)\s+)?(?:i\s+)?(?:never|don'?t|do not|doesn'?t|does not|no|not|"
                            r"denies)\b.*(?:\bvap\w*|\be-?cig\w*)")
_PACK_YEARS_ONLY = re.compile(r"^(?:about\s+|~)?\d+(?:\.\d+)?\s*pack[- ]?years?(?:\s+history)?$")
#: "my wife and I smoke": someone else AND the person — not set aside, asked about.
_FIRST_PERSON_SMOKING = re.compile(r"^(?:(?:but|and|also|though|although)\s+)?(?:i(?:'m| am|'ve| have)?\s+)?"
                                   r"(?:smoke|smoked|smoking|vape\w*|quit|stopped|started)\b")
_FILLER = re.compile(r"\b(?:i'm|im|i am|a|an|the|i|am|is|have|has|who|and|old|patient|person|me|my|year|years|yo|"
                     r"aged?|sex|smoking|currently|current|status)\b|[,.:;=/\-()']")
_DURATION = re.compile(r"\b\d+\s*(?:years?|yrs?|months?)\s+ago\b|\b(?:since|until|in)\s+\d{4}\b"
                       r"|\bfor\s+(?:the\s+(?:last|past)\s+)?\d+\s*(?:years?|yrs?|months?)\b")
_NEGATION = re.compile(r"^(?:no|not|never|without|denies|denied|negative for|free of|nor)\b\s*")

#: diagnosis words -> NHANES item (1 = yes). DIQ010 3 = borderline.
_DIAGNOSES: tuple[tuple[str, str, int], ...] = (
    (r"pre-?diabet\w*|borderline diabet\w*", "DIQ010", 3),
    (r"diabet\w*|t[12]\s*dm|dm\s*(?:type\s*)?[12]|type [12] dm|iddm|niddm|sugar diabetes", "DIQ010", 1),
    (r"hypertension|high blood pressure|htn|high bp|(?:elevated|raised) (?:blood pressure|bp)", "BPQ020", 1),
    (r"chronic kidney disease|kidney disease|ckd|weak kidneys|failing kidneys|kidney failure"
     r"|renal (?:failure|disease|insufficiency)", "KIQ020", 1),
    (r"asthma", "MCQ010", 1),
    (r"an(?:a)?emia", "MCQ053", 1),
    (r"arthritis", "MCQ160A", 1),
    (r"heart failure|chf", "MCQ160B", 1),
    (r"coronary (?:heart|artery) disease|chd|cad", "MCQ160C", 1),
    (r"angina", "MCQ160D", 1),
    (r"heart attack|myocardial infarction", "MCQ160E", 1),
    (r"stroke|cva|cerebrovascular accident", "MCQ160F", 1),
    (r"emphysema|copd", "MCQ160G", 1),
    (r"thyroid\w*(?: disease| problem| condition)?|hypothyroid\w*|hyperthyroid\w*", "MCQ160I", 1),
    (r"obes\w*|overweight", "MCQ160J", 1),
    (r"chronic bronchitis", "MCQ160K", 1),
    (r"fatty liver|liver disease|liver condition|cirrhosis|hepatitis", "MCQ160L", 1),
    (r"cancer|malignan\w*|carcinoma|lymphoma|leuk(?:a)?emia|melanoma", "MCQ220", 1),
    (r"hip fracture|broken hip|fractured hip", "OSQ010A", 1),
    (r"wrist fracture|broken wrist|fractured wrist", "OSQ010B", 1),
    (r"spine fracture|spinal fracture|vertebral fracture|broken spine|fractured spine", "OSQ010C", 1),
    (r"osteoporosis", "OSQ060", 1),
    (r"memory problems?|memory loss|confusion", "PFQ056", 1),
    (r"overnight (?:in )?hospital(?: stay)?|hospitali[sz]ed|hospital stay", "HUQ070", 1),
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
_NO_CONDITIONS = re.compile(
    r"^(?:i have\s+|there are\s+)?no\s+(?:(?:other|known|chronic|medical|significant|major|serious|current|"
    r"past|prior|relevant|health)\s+)*(?:conditions?|diagnos[ie]s|diseases?|illness(?:es)?|"
    r"medical (?:history|problems?|conditions?|issues?)|health (?:problems?|issues?|conditions?)|"
    r"problems?|issues?|history|comorbidit(?:y|ies)|pmh)\b"
    r"(?:\s*,?\s*(?:except(?: for)?|apart from|other than|besides|but|aside from)\s+(.+))?$")
#: "no OTHER conditions" leaves the diagnoses already listed alone; "other than" does not
_NO_OTHER = re.compile(r"^(?:i have\s+|there are\s+)?no\s+(?:\w+\s+)*?other\b(?!\s+than)")
#: "asthma, otherwise healthy": every diagnosis not listed is a No
_OTHERWISE = re.compile(r"^(?:(?:but|and)\s+)?(?:otherwise|else)\s+(?:healthy|well|fit(?: and well)?|"
                        r"in good health|no (?:other )?(?:problems|issues|conditions))\.?$"
                        r"|^(?:healthy|well) otherwise$|^nothing else$")
#: What may surround a known diagnosis in a piece: dates, stages, severity, history words.
_DIAG_QUALIFIERS = re.compile(
    r"\b(?:with|has|have|had|a|an|the|of|history|hx|type [12]|t[12]|diagnosed|dx|known|mild|moderate|severe|"
    r"controlled|uncontrolled|well|poorly|treated|untreated|chronic|stage|grade|class|early|"
    r"in|since|for|from|on|at|ago|years?|yrs?|months?|now|currently|current|previous|past|prior|recent|"
    r"recently|also|too|both|some|(?:19|20)\d{2}|\d+[ab]?|i{1,3}|iv)\b|[()\-–.:/]")
#: Words that suggest one of the 23 conditions, phrased in a way the reader cannot place.
#: Such a piece is never dropped as "not one of the conditions LinAge2 counts".
_MEDICAL_WORDS = re.compile(
    r"\b(?:heart|cardiac|coronary|kidney|renal|liver|hepat\w*|lung|pulmonary|bronch\w*|thyroid|bone|"
    r"fractur\w*|broken|osteo\w*|cancers?|tumou?rs?|malignan\w*|strokes?|tia|blood pressure|bp|hip|wrist|spine|"
    r"spinal|vertebra\w*|"
    r"diabet\w*|dm|anemi\w*|anaemi\w*|arthrit\w*|joints?|memory|dementia|confus\w*|hospital\w*|asthma|"
    r"emphysema|copd|obes\w*|overweight|angina|infarct\w*|attack|failure)\b")
_DIAG_HEADER = re.compile(r"^(?:diagnos[ie]s|known conditions|conditions?|medical history|history of|"
                          r"history|has|with|i have)\b\s*[:=]?\s*", re.I)
@lru_cache(maxsize=1)
def _condition_terms() -> "re.Pattern":
    """Every reviewed word for one of the 23 conditions: the rules' own patterns and the
    synonyms core.patient_vocabulary adds ('T2D', 'MI', 'hypertensive', 'strokes')."""
    from core.patient_vocabulary import vocabulary
    return re.compile(r"\b(?:" + "|".join(c.terms for c in vocabulary().condition_info.values()) + r")\b")


def _names_condition(text: str) -> bool:
    low = text.lower()
    return bool(_condition_terms().search(low) or _MEDICAL_WORDS.search(low))


_HEALTH = {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}
_TREND = {"better": 1, "worse": 2, "same": 3, "about the same": 3, "unchanged": 3}


def _visits_category(n: int) -> int:
    """HUQ050's answer categories: 0, 1, 2-3, 4-9, 10-12, 13+."""
    return 0 if n <= 0 else 1 if n == 1 else 2 if n <= 3 else 3 if n <= 9 else 4 if n <= 12 else 5


def _cotinine_level(ng_ml: float) -> int:
    """`digiCot`, the binning the model was trained on."""
    return 0 if ng_ml < 10 else 1 if ng_ml < 100 else 2 if ng_ml < 200 else 3


def _strip_offset(seg: str) -> tuple[int, str]:
    """`seg.strip().strip("-•*").strip()`, and where the result starts in `seg`."""
    t = seg.strip()
    a = len(seg) - len(seg.lstrip())
    u = t.strip("-•*")
    b = len(t) - len(t.lstrip("-•*"))
    w = u.strip()
    c = len(u) - len(u.lstrip())
    return a + b + c, w


#: a list headed by a negation: "denies HTN, DM2, CAD", "no diabetes, prediabetes"
_NEGATION_HEAD = re.compile(r"^(?:i\s+)?(?:have\s+|had\s+)?(?:no|not|never|denies|denied|negative for|free of|"
                            r"without|no (?:history|hx) of|never had)\b")
#: a lab value: one of the reader's names, then a number
_LAB_START = re.compile(r"^(?:" + "|".join(re.escape(a) for a, _ in _ALIAS_INDEX) + r")(?![a-z0-9])\s*"
                        r"(?:level|value)?\s*(?:[:=]|is|of|was)?\s*(?:\d|\.\d)"
                        r"|^(?:blood pressure|bp)\s*[:=]?\s*\d{2,3}\s*/")
_ABBREVIATIONS = {"vs", "dr", "mr", "mrs", "ms", "st", "approx", "eg", "ie", "etc", "incl", "wt", "ht", "yr",
                  "yrs", "hx", "dx", "pt", "no", "y", "o", "a"}


def _segments(line: str):
    """(offset, text) of each part of a line between ';' and sentence ends ('. ' before a
    word: "No diabetes. Hypertension." is two statements), never after an abbreviation."""
    cuts = [m.start() for m in re.finditer(";", line)]
    for m in re.finditer(r"\.\s+(?=[A-Za-z])", line):
        word = re.search(r"([A-Za-z.]+)$", line[:m.start()])
        if word and (len(word.group(1).replace(".", "")) <= 1
                     or word.group(1).replace(".", "").lower() in _ABBREVIATIONS):
            continue
        cuts.append(m.start())
    start = 0
    for cut in sorted(cuts) + [len(line)]:
        yield start, line[start:cut]
        start = cut + 1


def _statements(text: str) -> list[tuple[int, int, int, str]]:
    """(line number, start, end, statement): a line splits at ';' and at ', ' before a
    word. A diagnosis list ("diagnoses: a, b", "no known conditions except a, b") stays
    whole, except for a piece about smoking, which is always its own statement.
    `start`/`end` are offsets into the line as typed (a whole list spans its pieces)."""
    out: list[tuple[int, int, int, str]] = []
    for line_no, line in enumerate((text or "").splitlines()):
        for raw_start, raw in _segments(line):
            off, seg = _strip_offset(raw)
            if not seg:
                continue
            seg_start = raw_start + off
            pieces: list[tuple[int, int, str]] = []
            cut = 0
            for m in list(re.finditer(r",\s+(?=[A-Za-z])", seg)) + [None]:
                end = m.start() if m else len(seg)
                part = seg[cut:end]
                if part.strip():
                    lead = len(part) - len(part.lstrip())
                    text_ = part.strip()
                    a = seg_start + cut + lead
                    pieces.append((a, a + len(text_), text_))
                cut = m.end() if m else cut
            whole = (_DIAG_HEADER.match(seg) and not re.match(r"^(?:has|history of|with)\b", seg, re.I)) \
                or re.match(r"^(?:i have\s+)?no\b.*\b(?:except|apart from|other than|besides|aside from)\b",
                            seg, re.I) \
                or (len(pieces) > 1 and _NEGATION_HEAD.match(seg.translate(_UNIFY).lower())
                    and any(_names_condition(part) for _, _, part in pieces[1:]))
            if not whole:
                out.extend((line_no, a, b, part) for a, b, part in pieces)
                continue
            group: list[tuple[int, int, str]] = []

            def flush() -> None:
                out.append((line_no, group[0][0], group[-1][1], ", ".join(g[2] for g in group)))
                group.clear()

            for a, b, part in pieces:
                if _SMOKING_TOPIC.search(part.lower()) or _LAB_START.match(part.lower()) \
                        or _SOMEONE_ELSE.search(part.lower()):
                    if group:           # smoking, a lab value, someone else, in a list: its own
                        flush()
                    out.append((line_no, a, b, part))
                else:
                    group.append((a, b, part))
            if group:
                flush()
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


#: "asthma or COPD, not sure which": two candidates are not two diagnoses
_UNCERTAIN = re.compile(r"\bnot sure\b|\bunsure\b|\buncertain\b|\bmaybe\b|\bpossibl[ey]\b|\bprobabl[ey]\b|"
                        r"\bwhich one\b|\beither\b|\bsuspected\b|\?")
#: negations that cover every item after them, whatever joins the items
_LIST_NEGATION = re.compile(r"^(?:denies|denied|negative for|free of|without|no (?:history|hx) of|never had)\b")
#: a form's answer slot left empty, dashed or 0 after a condition: not a Yes
_FORM_ANSWER = re.compile(r"[:=]\s*(?:0|-+|–+|\.+)?\s*$|\s-\s*0\s*$")
#: a negation that qualifies treatment, control or time, not the diagnosis itself
_NEGATED_QUALIFIED = re.compile(r"\b(?:treated|untreated|controlled|uncontrolled|well|poorly|since|for|recent(?:ly)?|"
                                r"past|severe|mild|ago|years?|yrs?|months?|any ?more|longer|currently|now)\b")
_WINDOWED = ("HUQ070", "MCQ053")


def _old(body: str, years_back: int) -> bool:
    """A time reference that puts the event before the item's window."""
    from datetime import date
    this_year = date.today().year
    if any(int(y) < this_year - years_back for y in re.findall(r"\b((?:19|20)\d{2})\b", body)):
        return True
    return bool(re.search(r"\b(?:\d+|two|three|four|five|several|many)\s+years?\s+ago\b|\bhistory of\b|\bhx of\b"
                          r"|\bas a (?:child|kid|teen\w*)\b|\bprevious(?:ly)?\b|\bprior\b|\bonce\b|\blong ago\b"
                          r"|\bin the past\b(?!\s+(?:\d+|twelve|three|year|few))", body))


#: the item's own window, as its canonical phrase says it: not a qualifier of a 'no'
_WINDOW_PHRASE = re.compile(r"\b(?:in|during|over) the (?:past|last) (?:(?:12|twelve|3|three) months|year)\b")


#: HUQ070 is an overnight stay in the past 12 months; MCQ053 treatment for anemia in
#: the past 3 months — a dated or 'history of' event is outside them
_OUTSIDE_WINDOW = {"HUQ070": lambda body: _old(body, 1), "MCQ053": lambda body: _old(body, 0)}


def _read_diagnoses(text: str, context: str = "") -> tuple[list[tuple[str, int]], list[str], bool]:
    """'hypertension, no diabetes and prediabetes' -> (found, ignored, unclear).

    found: [(item, answer)…]. ignored: pieces that name no condition LinAge2 counts and
    nothing like one ("high cholesterol"). unclear: a piece names or suggests one of
    the 23 conditions but cannot be placed ("heart disease", "hypertension on
    lisinopril") — the statement is then not used, and the caller refuses it."""
    found: list[tuple[str, int]] = []
    ignored: list[str] = []
    if _UNCERTAIN.search(text):
        return found, ignored, True             # "asthma or COPD, not sure which"
    parts = re.split(r"(,|;|&|\band\b|\bor\b|\bnor\b)", text)
    head = _NEGATION.match(parts[0].strip(" ."))
    covering = bool(head and _LIST_NEGATION.match(parts[0].strip(" .")))
    head_items: set = set()
    for n in range(0, len(parts), 2):
        piece = parts[n].strip(" .")
        sep = parts[n - 1].strip() if n else ""
        if not piece:
            continue
        if _FORM_ANSWER.search(piece):
            return found, ignored, True         # "diabetes: 0", "stroke: -", "cancer:"
        negated = bool(_NEGATION.match(piece))
        if not negated and n and head:
            # a negation heading the list: "denies X, Y", "no X or Y" cover the rest;
            # a bare "no X, Y" or "no X and Y" might not — unless Y is X's borderline
            if covering or sep in ("or", "nor"):
                negated = True
            elif not (head_items == {"DIQ010"} and re.fullmatch(r"(?:has\s+|have\s+)?(?:pre-?diabet\w*|"
                                                               r"borderline diabet\w*)", piece)):
                return found, ignored, True
        body = _NEGATION.sub("", piece)
        hits = []
        rest = body
        for pattern, item, answer in _DIAGNOSES:
            if re.search(rf"\b(?:{pattern})\b", rest):
                hits.append((item, 2 if negated else answer))
                rest = re.sub(rf"\b(?:{pattern})\b", " ", rest)
        if not hits:
            if _MEDICAL_WORDS.search(body) or _condition_terms().search(body):
                return found, ignored, True
            ignored.append(piece)
            continue
        if re.sub(r"\s", "", _DIAG_QUALIFIERS.sub(" ", rest)):
            return found, ignored, True
        qualifiers = _WINDOW_PHRASE.sub(" ", body)
        if {item for item, _ in hits} == {"MCQ053"}:
            qualifiers = re.sub(r"\b(?:treated|treatment)(?:\s+for)?\b", " ", qualifiers)   # the item is treatment
        if negated and (len(hits) > 1 or _NEGATED_QUALIFIED.search(qualifiers)):
            return found, ignored, True         # "never treated hypertension", "no asthma since 2010"
        if not negated and any(item in _WINDOWED and _OUTSIDE_WINDOW[item](f"{context} {body}")
                               for item, _ in hits):
            return found, ignored, True         # "hospitalized in 2010": not the past 12 months
        if n == 0:
            head_items = {item for item, _ in hits}
        found.extend(hits)
    return found, ignored, False


def read_patient_text(text: str) -> ParsedPatient:
    """Read `text` into a ParsedPatient. Never raises on content: everything that
    could not be used is reported on the result."""
    p = ParsedPatient()
    seen: dict[str, Reading] = {}

    _TOPIC = {"age": "age", "sex": "sex", "smoking": "smoking", "weight_kg": "weight",
              "height_cm": "height"}

    def problem(text_: str, kind: str, topic: str, statements=None) -> None:
        p.problems.append(Problem(text_, kind, topic,
                                  (cur.index,) if statements is None else statements))

    def add(reading: Reading) -> None:
        reading.statement = cur.index
        cur.facts.setdefault("labs", {})[reading.code] = reading.value
        prev = seen.get(reading.code)
        if prev is not None and not prev.blocking and not reading.blocking \
                and abs((prev.value or 0) - (reading.value or 0)) > 1e-9:
            reading.status = DUPLICATE
            reading.note = f"{reading.label} was already given as {prev.typed!r}"
        seen.setdefault(reading.code, reading)
        p.readings.append(reading)

    diagnosed: set[str] = set()         # items a statement answered, Yes or No
    no_conditions = False               # "no known conditions" (not "no OTHER conditions")
    allowed: set[str] = set()           # ... and the Yes it allowed: "except hypertension"
    said_none = False                   # either: every diagnosis not listed is a No
    #: the last smoking clause: its line, statement index, kind, and whether it set the status
    last_smoking: dict = {"line": -1, "idx": -1, "kind": None, "set_here": False, "stmt": "",
                          "origin": -1}
    cur = Statement(-1, -1, 0, 0, "")           # the statement being read

    def set_once(attr: str, value, what: str, stmt: str) -> None:
        cur.facts[attr] = value
        current = getattr(p, attr)
        if current is not None and current != value:
            problem(f"two different {what}s: {current} and {value} (from '{stmt}')", "contradiction",
                    _TOPIC.get(attr, "other"))
        else:
            setattr(p, attr, value)

    def unsure(stmt: str, statements=None) -> None:
        problem(f"'{stmt}': cannot tell whether you smoke now, used to, or never did; "
                f"write 'current smoker', 'former smoker, quit 2010' or 'never smoked'",
                "ambiguous", "smoking", statements)

    def set_smoking(status: str, level: int, stmt: str) -> None:
        before = p.smoking
        set_once("smoking", status, "smoking status", stmt)
        cur.facts["smoking"] = (status, level)          # what the words say, level included
        if p.smoking != status:
            return
        if before == status and not p.cotinine_measured and p.cotinine_level not in (None, level):
            problem(f"two different smoking intensities: cotinine level {p.cotinine_level} and {level} "
                    f"(from '{stmt}'); keep one ('current smoker', 'occasional smoker')",
                    "contradiction", "smoking")
            return
        if p.cotinine_measured:
            if level != p.cotinine_level:
                p.notes.append(f"'{stmt}' would put cotinine at level {level}; the measured "
                               f"value is used ({p.cotinine_note})")
        else:
            p.cotinine_level = level
            p.cotinine_note = (f"cotinine level {level} from '{stmt}' (training bins: 0 <10, "
                               f"1 10-100, 2 100-200, 3 >=200 ng/mL)")

    def read_smoking(rest: str, stmt: str, line_no: int, idx: int) -> Optional[str]:
        """Read the smoking status of a clause about smoking. Returns what is left of
        the clause to read as anything else, or None when the clause is used up."""
        if re.match(r"^\s*smokeless\b", rest):
            if re.match(r"^\s*smokeless(?: tobacco)?(?: use)?\s*[:=\-?]?\s*(?:never(?: used)?|no|none|n|denies)\s*$", rest):
                p.notes.append(f"'{stmt}': smokeless tobacco is not smoking; it does not change the smoking status")
                cur.facts["note"] = "smokeless"
                return None
            problem(f"'{stmt}': smokeless tobacco raises cotinine, which LinAge2 reads, but is not smoking to "
                    f"the knowledge base; give 'cotinine N ng/mL' if you have it, and your smoking status on "
                    f"its own line", "vaping", "smoking")
            return None
        if _CANNABIS.search(rest):
            problem(f"'{stmt}': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); "
                    f"say your tobacco smoking on its own ('never smoked', 'former smoker', "
                    f"'current smoker')", "ambiguous", "smoking")
            return None
        for pattern, status, level, kind in _SMOKING:
            if not re.search(pattern, rest):
                continue
            reduced = re.sub(pattern, " ", rest)
            if any(st != status and re.search(pat, reduced) for pat, st, _, _ in _SMOKING):
                unsure(stmt)            # "quit smoking after I failed to quit"
                return None
            if kind == "never":         # "no history of smoking or diabetes": the 'no' carries
                carried = re.match(r"\s*(or|nor|and)\b(.*)$", reduced)
                if carried and _names_condition(carried.group(2)):
                    if carried.group(1) == "and":
                        unsure(stmt)
                        return None
                    reduced = " no " + carried.group(2)
            timed = _SINCE.sub(" ", re.sub(r"\ba day in (?:my|his|her|their) life\b", " ", reduced))
            if kind == "never":
                doubt = (_PRESENT_CUE.search(timed) or _PAST_CUE.search(timed) or _YEAR.search(timed)
                         or re.search(r"\b(?:since|until|ago)\b", reduced)
                         or re.search(r"^\s*[:=?-]\s*(?:no|n|false|0)\b", reduced)       # "never smoker: no"
                         or re.search(r"\bfor\s+(?:the\s+)?(?:past\s+|last\s+)?(?:about\s+|over\s+)?\d+\s*"
                                      r"(?:years?|yrs?|months?|weeks?|days?)\b", reduced))  # "smoke-free for 10 years"
            elif kind == "former":
                doubt = (_PRESENT_CUE.search(timed) or re.search(r"\b(?:smoke|smokes|smoking)\b", timed)
                         or _TEMPORARY_QUIT.search(rest))
            elif kind == "generic":
                doubt = _NEGATION_CUE.search(timed) or (
                    _YEAR.search(timed) and not _PAST_CUE.search(timed)) or (
                    _PAST_CUE.search(timed) and _PRESENT_CUE.search(timed)) or (
                    re.search(r"[:=?-]\s*(?:0|nil|none|false|neg(?:ative)?|denie[sd]|absent|-+)?\s*$", rest))
                if not doubt and _PAST_CUE.search(timed):
                    if not _STRONG_PAST.search(timed):
                        doubt = True    # "smoker in my youth", "was hospitalized": said, not when
                    else:
                        status, level, kind = "FormerSmoker", 0, "former"   # "smoker (1990-2015)"
            else:                       # explicitly smoking now
                doubt = _NEGATION_CUE.search(timed) or _PAST_CUE.search(timed) or _YEAR.search(timed)
            if not doubt and status == "CurrentSmoker":
                count = re.search(r"\b(\d+)\s*(?:-\s*\d+\s*)?(?:cigarettes?|cigs?)\s*(?:a|per|each|/)\s*day\b", rest)
                if level == 3 and (_OCCASIONAL_WORDS.search(rest) or (count and int(count.group(1)) < 10)):
                    doubt = True        # "occasionally smokes", "smokes 2 cigarettes a day": not level 3
                elif level == 1 and (re.search(r"\bpacks?\b", rest) or (count and int(count.group(1)) >= 10)):
                    doubt = True        # "light smoker (20 cigarettes a day)"
            if doubt:
                unsure(stmt)
                return None
            nicotine = _OTHER_NICOTINE.search(reduced)
            if nicotine and not re.search(r"\b(?:never|no|not|nor|without|don'?t|doesn'?t)\b|n't\b|\bor\s*$",
                                          re.split(r"[,;.]|\bbut\b", reduced[:nicotine.start()])[-1]):
                problem(f"'{stmt}': vaping or other nicotine (snus, chew, patches, pouches) raises cotinine, "
                        f"which LinAge2 reads, but is not smoking to the knowledge base; give 'cotinine N "
                        f"ng/mL' if you have it, and your smoking status on its own line", "vaping", "smoking")
                return None
            before = p.smoking
            set_smoking(status, level, stmt)
            last_smoking.update(line=line_no, idx=idx, kind=kind, stmt=stmt, origin=idx,
                                set_here=before is None and p.smoking == status)
            return _SMOKING_DETAIL.sub(" ", reduced)
        if _VAPE_NEGATION.match(rest.strip()) and not re.search(r"\bsmok|\bcig|\btobacco", rest):
            p.notes.append(f"'{stmt}': vaping is not read; it does not change the smoking status")
            cur.facts["note"] = "vaping"
            return None
        if _PACK_YEARS_ONLY.match(rest.strip()):
            p.notes.append(f"'{stmt}': pack-years are not a LinAge2 input (it reads cotinine); "
                           f"they do not set a smoking status")
            cur.facts["note"] = "pack-years"
            return None
        if _VAPING.search(rest) and not re.search(r"\bsmok|\bcig|\btobacco", rest):
            problem(f"'{stmt}': vaping raises cotinine but is not smoking to the knowledge "
                    f"base; give 'cotinine N ng/mL' if you have it, and your smoking status "
                    f"('never smoked', 'former smoker', 'current smoker')", "vaping", "smoking")
            return None
        if _OTHER_NICOTINE.search(rest) and not re.search(r"\bsmok|\bcig|\btobacco(?! use)", rest):
            problem(f"'{stmt}': nicotine that is not smoked (snus, chew, patches, pouches) raises cotinine, "
                    f"which LinAge2 reads, but is not smoking to the knowledge base; give 'cotinine N "
                    f"ng/mL' if you have it, and your smoking status on its own line", "vaping", "smoking")
            return None
        problem(f"'{stmt}' is about smoking and was not understood; say it in one "
                f"phrase: 'current smoker', 'former smoker, quit 2010' or 'never smoked'",
                "not_understood", "smoking")
        return None

    def modify_smoking(low: str, stmt: str, cue: str) -> None:
        """A clause right after a smoking clause that says when: check it against it."""
        kind = last_smoking["kind"]
        if kind is None:                # that clause was refused: already a problem
            return
        said = f"{last_smoking['stmt']}, {stmt}"
        origin = last_smoking["origin"]
        cur.joined_to = origin
        both = (origin, cur.index)
        if _names_condition(low):       # "quit after my heart attack": both, so neither is lost
            problem(f"'{said}' says when you quit and names a condition; put the condition on its own "
                    f"line ('diagnoses: …') and your smoking in one phrase ('former smoker, quit 2010')",
                    "ambiguous", "smoking", both)
            last_smoking["kind"] = None
            return
        if _TEMPORARY_QUIT.search(low) and not _PRESENT_CUE.search(low):
            when = "temporary"
        elif _PRESENT_CUE.search(low) or cue in ("started again", "restarted", "relapsed", "back on it",
                                                  "back on", "back", "still"):
            when = "present"
        elif cue in ("started", "since") and not _STRONG_PAST.search(low):
            when = "neutral"            # "since 1990", "started 30 years ago"
        elif cue in ("quit", "stopped", "gave up", "given up", "until", "till", "no longer") \
                or cue.startswith("not any") or _PAST_CUE.search(low):
            when = "past"
        else:
            when = "neutral"            # "started at 16"
        if when == "temporary" or kind == "never":
            unsure(said, both)
        elif when == "present" and kind == "former":
            unsure(said, both)
        elif when == "past" and kind in _STILL_KINDS:
            unsure(said, both)
        elif when == "past" and kind == "generic":
            if not last_smoking["set_here"] or p.smoking != "CurrentSmoker":
                unsure(said, both)
                return
            p.smoking = None            # "smoker, quit 2015": a former smoker
            if not p.cotinine_measured:
                p.cotinine_level = None
            set_smoking("FormerSmoker", 0, said)
            p.statements[origin].facts["smoking"] = ("FormerSmoker", 0)
            last_smoking["kind"] = "former"
            p.notes.append(f"'{said}' read as a former smoker")
            return
        else:
            return
        last_smoking["kind"] = None

    def answer(found: list[tuple[str, int]], stmt: str) -> None:
        """Record one statement's diagnoses; a different answer on another line is a
        problem — except "no diabetes" with "prediabetes", which is borderline (3)."""
        merged: dict[str, int] = {}
        for item, value in found:
            prev = merged.get(item)
            if prev is not None and prev != value and {prev, value} != {2, 3}:
                problem(f"'{stmt}' answers {_ITEM_LABEL[item]} both ways; keep one",
                        "contradiction", "condition")
                return
            merged[item] = 3 if {prev, value} == {2, 3} else value
        cur.facts.setdefault("conditions", {}).update(merged)
        for item, value in list(merged.items()):
            prev = p.questionnaire.get(item) if item in diagnosed else None
            if prev is not None and prev != value:
                if {prev, value} == {2, 3}:
                    merged[item] = 3
                else:
                    problem(f"'{stmt}' answers {_ITEM_LABEL[item]} differently from an "
                            f"earlier line; keep one", "contradiction", "condition")
                    return
        for item, value in merged.items():
            p.questionnaire[item] = value
            diagnosed.add(item)

    def answer_once(item: str, value: int, what: str, stmt: str) -> bool:
        """A second, different answer to a health question is a contradiction."""
        cur.facts.setdefault("questionnaire", {})[item] = value
        if item in p.questionnaire and p.questionnaire[item] != value:
            problem(f"two different {what} answers (from '{stmt}'); keep one", "contradiction", "other")
            return False
        return True

    def note_ignored(ignored: list[str]) -> None:
        if ignored:
            cur.facts["ignored"] = list(ignored)
            p.questionnaire_notes.append(
                "not among the conditions LinAge2's comorbidity score counts (not used): "
                + ", ".join(f"'{x}'" for x in ignored))

    for idx, (line_no, s_start, s_end, stmt) in enumerate(_statements(text)):
        cur = Statement(idx, line_no, s_start, s_end, stmt)
        p.statements.append(cur)
        low = stmt.translate(_UNIFY).lower().strip()
        orig_stmt = stmt

        # ── the clause right after a smoking clause, saying when ───────────────
        right_after = last_smoking["idx"] == idx - 1 and last_smoking["kind"] is not None
        if last_smoking["line"] == line_no and last_smoking["idx"] == idx - 1:
            m = _MODIFIER.match(low)
            if m and _MODIFIER_REST.match(m.group("rest")):
                last_smoking["idx"] = idx
                modify_smoking(low, stmt, m.group("cue"))
                continue
        if right_after and (last_smoking["line"] == line_no or last_smoking["line"] == line_no - 1) \
                and not _SMOKING_TOPIC.search(low) and _AFTER_SMOKING.search(low) \
                and not _OTHER_HABIT.search(low) \
                and (last_smoking["line"] == line_no or _MODIFIER.match(low)):
            # "smoker, now quit", "smoker\nquit 2015", "former smoker\nrelapsed": when, unread
            unsure(f"{last_smoking['stmt']}, {stmt}", (last_smoking["origin"], idx))
            cur.joined_to = last_smoking["origin"]
            last_smoking["kind"] = None
            continue

        # ── "asthma or COPD, not sure which": the clause before is not a diagnosis ──
        if idx and p.statements[idx - 1].line == line_no and _UNSURE_ALONE.match(low):
            problem(f"'{p.statements[idx - 1].text}, {stmt}': not sure is not a diagnosis; write only what a "
                    f"doctor told you ('diagnoses: …')", "ambiguous", "condition", (idx - 1, idx))
            continue

        # ── about somebody else: set aside, and say so ────────────────────────
        if _SOMEONE_ELSE.search(low):
            if _SMOKING_TOPIC.search(low) and re.search(r"\b(?:i|we|both)\b", low) \
                    and not _FIRST_PERSON_SMOKING.match(low):
                problem(f"'{stmt}' is about you and someone else; say your own smoking "
                        f"on its own line ('never smoked', 'former smoker', 'current smoker')",
                        "someone_else", "smoking")
                continue
            own = _SMOKING_TOPIC.search(low) and _FIRST_PERSON_SMOKING.match(low)
            if not own and not _EXPOSURE.search(low) and not _OTHER_SUBJECT.match(low) and (
                    _SMOKING_TOPIC.search(low) or _names_condition(low) or _AGE_RES[0].search(low)
                    or _AGE_RES[1].search(low)):
                # "smokes with friends", "diabetes like my mother", "58 year old male with family
                # history of diabetes": someone else is mentioned, but the statement is the person's
                problem(f"'{stmt}' mentions someone else and something about you; put what is yours on "
                        f"its own line, without the other person", "someone_else",
                        "smoking" if _SMOKING_TOPIC.search(low) else "condition")
                continue
            if not own:
                p.set_aside.append(Remark(stmt, idx))
                cur.facts["set_aside"] = True
                continue

        # ── a GrimAge result (before demographics: "4.5 years" is not an age) ─
        m = re.match(r"(?:grim\s*age|ageaccelgrim)(\s+age)?(\s+accel(?:eration)?)?\s*[:=]?\s*"
                     r"([+-]?)(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)?\s*$", low)
        if m:
            years = float(m.group(3) + m.group(4))
            if not (m.group(2) or m.group(3) or low.startswith("ageaccelgrim")):
                problem(f"'{stmt}' looks like a GrimAge clock AGE; give the acceleration "
                        f"(clock age minus your age), e.g. 'GrimAge acceleration +4 years'",
                        "ambiguous", "grimage")
            elif abs(years) > 30:
                problem(f"GrimAge acceleration {years:+g} years is outside anything the "
                        f"clock produces (±30); check the value", "unit", "grimage")
            else:
                cur.facts["grimage"] = years
                prev = p.extra_markers.get("AgeAccelGrim")
                if prev is not None and prev["value"] != years:
                    problem(f"two different GrimAge accelerations: {prev['value']:+g} and {years:+g}; keep one",
                            "contradiction", "grimage")
                    continue
                p.extra_markers["AgeAccelGrim"] = {"value": years, "unit": "years"}
                p.notes.append(f"GrimAge acceleration {years:+g} years: the knowledge base's 10-year "
                               f"heart-disease model reads it; it is never combined with LinAge2")
            continue

        # ── demographics and smoking (may share one statement) ─────────────────
        rest = low
        for rx in _AGE_RES:
            m = rx.search(rest)
            if m:
                if _AGE_CONTEXT.search(rest[:m.start()]):
                    break               # "diagnosed at 45 years old", "biological age 62 years old"
                set_once("age", float(m.group(1)), "age", stmt)
                rest = rest[:m.start()] + " " + rest[m.end():]
                break
        for rx, sex in _SEX_RES:
            if rx.search(rest):
                set_once("sex", sex, "sex", stmt)
                rest = rx.sub(" ", rest)
                break
        if re.match(r"^\s*(?:serum\s+)?cotinine\b", rest):
            pass                                # a lab, read below
        elif _SMOKING_TOPIC.search(rest):
            left = read_smoking(rest, stmt, line_no, idx)
            rest = left if left is not None else ""
        if rest != low:
            rest = _DURATION.sub(" ", rest)         # "quit smoking 20 years ago", "until 2015"
            leftover = re.sub(r"\s+", " ", _FILLER.sub(" ", rest)).strip()
            if not leftover:
                continue
            low = stmt = leftover               # e.g. "58 year old man with diabetes"

        # ── questionnaire ────────────────────────────────────────────────────
        if _OTHERWISE.match(low):
            said_none = True                    # "asthma, but otherwise healthy"
            cur.facts["no_conditions"] = "otherwise"
            continue
        m = _NO_CONDITIONS.match(low)
        if m:
            exceptions, ignored, unclear = _read_diagnoses(m.group(1)) if m.group(1) else ([], [], False)
            if unclear:
                problem(f"'{stmt}': the exception was not understood, and every "
                        f"other diagnosis would be answered No; name it as a diagnosis "
                        f"(e.g. 'no other conditions except hypertension')", "ambiguous", "condition")
                continue
            other = bool(_NO_OTHER.match(low))
            excepted = {i for i, a in exceptions if a != 2}
            listed_yes = [i for i in sorted(diagnosed) if p.questionnaire.get(i) in (1, 3)
                          and i not in excepted]
            if listed_yes and not other:
                problem(f"'{stmt}' contradicts the diagnoses already given "
                        f"({', '.join(_ITEM_LABEL[i] for i in listed_yes)}); "
                        f"write 'no other conditions' if those are all", "contradiction", "condition")
                continue
            if no_conditions and excepted - allowed:
                problem(f"'{stmt}' contradicts an earlier 'no known conditions'; "
                        f"keep one of the two", "contradiction", "condition")
                continue
            if not other:
                no_conditions, allowed = True, allowed | excepted
            said_none = True
            cur.facts["no_conditions"] = "other" if other else "all"
            answer(exceptions, stmt)
            note_ignored(ignored)
            continue
        m = re.match(r"(?:(?:self[- ](?:rated|reported|assessed)|general|overall|my)\s+)?health"
                     r"(?:\s+status)?\s*(?:is|:|=|-)?\s*(excellent|very good|good|fair|poor)\s*\.?\s*$", low)
        if m:
            if not answer_once("HUQ010", _HEALTH[m.group(1)], "self-rated health", stmt):
                continue
            p.questionnaire["HUQ010"] = _HEALTH[m.group(1)]
            cur.facts.setdefault("questionnaire", {})["HUQ010"] = _HEALTH[m.group(1)]
            p.questionnaire_notes.append(f"self-rated health: {m.group(1)}")
            continue
        m = re.match(r"health (?:compared (?:to|with)|vs\.?|versus) (?:(?:a|one|1) year ago|last year)"
                     r"\s*[:=]?\s*(better|worse|about the same|same|unchanged)\s*\.?\s*$", low) or \
            re.match(r"health (?:is )?(?:getting |got )?(better|worse)(?:\s+than\s+(?:(?:a|one|1) year ago|"
                     r"last year))?\s*\.?\s*$", low)
        if m:
            if not answer_once("HUQ020", _TREND[m.group(1)], "health-trend", stmt):
                continue
            p.questionnaire["HUQ020"] = _TREND[m.group(1)]
            cur.facts.setdefault("questionnaire", {})["HUQ020"] = _TREND[m.group(1)]
            p.questionnaire_notes.append(f"health vs a year ago: {m.group(1)}")
            continue
        m = re.match(r"(?:healthcare|health care|doctor'?s?|gp|physician|medical|clinic) visits?"
                     r"(?: (?:last|in the (?:last|past)) year| per year| a year)?\s*[:=]?\s*(\d+)"
                     r"(?:\s*(?:per year|a year|/\s*year|last year|in the (?:last|past) (?:year|12 months)))?"
                     r"\s*\.?\s*$", low)
        if m:
            n = int(m.group(1))
            if not answer_once("HUQ050", _visits_category(n), "healthcare-visits", stmt):
                continue
            p.questionnaire["HUQ050"] = _visits_category(n)
            cur.facts.setdefault("questionnaire", {})["HUQ050"] = _visits_category(n)
            p.questionnaire_notes.append(f"healthcare visits {n} -> NHANES category "
                                         f"{_visits_category(n)}")
            continue
        diag_text = _DIAG_HEADER.sub("", low) if _DIAG_HEADER.match(low) else low
        if _DIAG_HEADER.match(low) and re.fullmatch(r"\s*(?:none|nil|no|n/?a)\.?\s*", diag_text):
            no_conditions = said_none = True    # "diagnoses: none"
            cur.facts["no_conditions"] = "all"
            continue
        found, ignored, unclear = _read_diagnoses(diag_text, context=low)
        if found and not unclear:
            if no_conditions and any(a != 2 and i not in allowed for i, a in found):
                problem(f"'{stmt}' contradicts 'no known conditions'; write 'no other "
                        f"conditions' with the diagnoses, or drop one of the two",
                        "contradiction", "condition")
                continue
            answer(found, stmt)
            note_ignored(ignored)
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
                problem(f"'{stmt}': give the weight's unit (kg or lb)", "unit", "weight")
                continue
            w = float(m.group(1))
            w = w * 0.45359237 if m.group(2).startswith(("lb", "pound")) else w
            if not 25 <= w <= 350:
                problem(f"'{stmt}' = {w:.0f} kg, outside 25-350 kg; check the value and unit",
                        "unit", "weight")
            else:
                set_once("weight_kg", w, "weight", stmt)
            continue
        m = re.match(rf"height\s*[:=]?\s*{_NUM}\s*(cm|m|in|inches)?\s*$", low)
        m2 = re.match(r"height\s*[:=]?\s*(\d)\s*'\s*(\d{1,2})\s*(?:\"|'')?\s*$", low)
        if m or m2:
            if m and not m.group(2):
                problem(f"'{stmt}': give the height's unit (cm, m or in, or 5'10\")", "unit", "height")
                continue
            if m:
                h, unit = float(m.group(1)), m.group(2)
                h = h * 100 if unit == "m" else h * 2.54 if unit.startswith("in") else h
            else:
                h = (int(m2.group(1)) * 12 + int(m2.group(2))) * 2.54
            if not 100 <= h <= 230:
                problem(f"'{stmt}' = {h:.0f} cm, outside 100-230 cm; check the value and unit",
                        "unit", "height")
            else:
                set_once("height_cm", h, "height", stmt)
            continue

        # ── cotinine as a lab: ng/mL, or an explicit level ────────────────────
        m = re.match(rf"(?:serum )?cotinine\s*(level)?\s*(?:[:=]|\bis\b|\bwas\b)?\s*{_NUM}\s*"
                     rf"(ng\s*/\s*ml|[uµμ]g\s*/\s*l)?\s*\.?\s*$", low)
        if m:
            v = float(m.group(2))
            if m.group(3):
                level = _cotinine_level(v)
                note = f"cotinine {v:g} ng/mL -> level {level}"
            elif m.group(1) and v in (0, 1, 2, 3):
                level, note = int(v), f"cotinine level {int(v)} as given"
            else:
                problem(f"'{stmt}': give cotinine in ng/mL (e.g. 'cotinine 250 ng/mL') "
                        f"or as 'cotinine level 0-3'", "unit", "cotinine")
                continue
            cur.facts["cotinine"] = level
            if p.cotinine_measured and p.cotinine_level != level:
                problem(f"two different cotinine levels: {p.cotinine_level} and {level}",
                        "contradiction", "cotinine")
                continue
            if p.cotinine_level is not None and p.cotinine_level != level:
                said = p.cotinine_note.split(" (")[0]
                p.notes.append(f"{said[:1].upper()}{said[1:]} replaced by the measured value ({note})")
            p.cotinine_level, p.cotinine_note, p.cotinine_measured = level, note, True
            continue

        # ── a lab value ────────────────────────────────────────────────────────
        reading = _read_lab(stmt, low)
        if reading is not None:
            add(reading)
        elif "smoking" in cur.facts:
            # what the smoking clause did not use may change what it means: "I never quit
            # smoking" (never), "tried to quit smoking" (tried), "non- smoker" (non)
            said = _WHO_SMOKING.get(cur.facts["smoking"], cur.facts["smoking"][0])
            problem(f"'{orig_stmt}' reads as '{said}', but '{stmt}' in it was not understood; say your "
                    f"smoking in one phrase ('current smoker', 'former smoker, quit 2010', 'never smoked') "
                    f"and put anything else on its own line", "ambiguous", "smoking")
            cur.leftover = stmt
            if last_smoking["origin"] == idx and last_smoking["set_here"]:
                p.smoking = None                # not a status anyone should see as read
                if not p.cotinine_measured:
                    p.cotinine_level, p.cotinine_note = None, ""
                last_smoking["kind"] = None
        elif "cotinine" in low:
            problem(f"'{stmt}' mentions cotinine but was not read; write it as 'cotinine 250 ng/mL' or "
                    f"'cotinine level 0-3', on its own line", "unit", "cotinine")
            cur.leftover = stmt
        else:
            p.not_understood.append(Remark(stmt, idx))
            cur.leftover = stmt

    # An unread line that names one of the 23 conditions would be lost (assumed No, or
    # answered No next to a list); one that only suggests one is lost only next to a list.
    lost = [u for u in p.not_understood
            if any(re.search(rf"\b(?:{pat})\b", u.lower()) for pat, _, _ in _DIAGNOSES)
            or _condition_terms().search(u.lower())
            or ((diagnosed or said_none) and _MEDICAL_WORDS.search(u.lower()))]
    for u in lost:
        p.problems.append(Problem(
            f"'{u}' mentions a medical condition but was not understood, and it "
            f"would be answered No; write it as 'diagnoses: …' with the condition "
            f"(e.g. 'diagnoses: stroke') or remove it", "lost_condition", "condition",
            (getattr(u, "statement", -1),)))
    p.not_understood = [u for u in p.not_understood if u not in lost]
    if diagnosed or said_none:
        yes = [_ITEM_LABEL[i] + (" (borderline)" if p.questionnaire[i] == 3 else "")
               for i in _FS1_ITEMS if i in diagnosed and p.questionnaire[i] != 2]
        no = [_ITEM_LABEL[i] for i in _FS1_ITEMS if i in diagnosed and p.questionnaire[i] == 2]
        for q in _FS1_ITEMS:
            p.questionnaire.setdefault(q, 2)
        p.questionnaire_notes.insert(0, (
            "diagnoses: " + (", ".join(yes) if yes else "none")
            + (f"; not: {', '.join(no)}" if no else "")
            + " — every diagnosis not listed is answered No"))
    def said(attr: str) -> tuple:
        return tuple(st.index for st in p.statements if attr in st.facts)

    if p.age is None:
        p.problems.append(Problem("no age found (e.g. '58 year old' or 'age 58')", "missing", "age"))
    elif not 20 <= p.age <= 90:
        p.problems.append(Problem(f"age {p.age:g} is outside 20-90, the ages LinAge2's reference covers",
                                  "unit", "age", said("age")))
    if p.sex is None:
        p.problems.append(Problem("no sex found ('male' or 'female'): LinAge2 has a separate model for each",
                                  "missing", "sex"))
    if p.cotinine_level is not None and p.smoking is None:
        p.notes.append("cotinine was given without a smoking status: the knowledge base credits "
                       "cotinine years to smoking only for a stated current smoker")
    glucose = next((r for r in p.readings if r.code == "LBDSGLSI" and not r.blocking), None)
    if glucose is not None and not glucose.fasting:
        p.notes.append("glucose was not marked fasting: LinAge2 uses it as typed, but the "
                       "knowledge base's FastingGlucose witness needs a fasting value — write "
                       "'fasting glucose …' if it was")
    if p.set_aside:
        p.notes.append("set aside (about someone else, or smoke you did not smoke): "
                       + "; ".join(f"'{s_}'" for s_ in p.set_aside))
    if p.weight_kg and p.height_cm and "BMXBMI" not in seen:
        bmi = p.weight_kg / (p.height_cm / 100) ** 2
        lo, hi = _range("BMXBMI")
        if not lo <= bmi <= hi:
            p.problems.append(Problem(f"BMI {bmi:.1f} from weight and height is outside the {lo:g}-{hi:g} "
                                      f"NHANES observed; check them", "unit", "weight",
                                      said("weight_kg") + said("height_cm")))
        else:
            p.notes.append(f"BMI {bmi:.1f} from weight and height")
    elif p.weight_kg and p.height_cm and not seen["BMXBMI"].blocking:
        bmi = p.weight_kg / (p.height_cm / 100) ** 2
        if abs(bmi - seen["BMXBMI"].value) > 1.5:
            p.problems.append(Problem(f"BMI {seen['BMXBMI'].value:g} was typed, but weight and height give "
                                      f"{bmi:.1f}; keep the one that is right", "contradiction", "weight",
                                      said("weight_kg") + said("height_cm")))
    _outcomes(p)
    return p


def _outcomes(p: ParsedPatient) -> None:
    """Sum up each statement: refused, set aside, (partly) not understood, or read."""
    refused = {i for x in p.all_problems() for i in getattr(x, "statements", ())}
    for st in p.statements:
        if st.index in refused:
            st.outcome = "refused"
        elif st.facts.get("set_aside"):
            st.outcome = "set_aside"
        elif st.leftover:
            st.outcome = "partly_read" if any(k != "ignored" for k in st.facts) else "not_understood"
        else:
            st.outcome = "read"


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
