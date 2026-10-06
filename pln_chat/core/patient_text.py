"""A few lines about a person -> a patient: the units, the domain rules, and the canonical lines.

    58 year old male, current smoker
    albumin 4.2 g/dL
    HbA1c 6.3 %
    CRP 1.2 mg/L
    blood pressure 138/86
    diagnoses: hypertension

Free text is read by a model (core.patient_read); these canonical lines are read by
`read_lines` with no model. Either way the Facts end in `assemble`, which returns what was
READ — every value with the unit it was typed in, the value in the unit LinAge2 takes, and a
status — before anything is built. That table is the point: a lab value in the wrong unit is
the commonest way to get a confident, wrong biological age (albumin 4.2 read as g/L instead
of g/dL is -30 g/L from the median, and LinAge2 turns it into ~10 years), so units are
never guessed:

* a unit it does not know is refused, with the units it accepts;
* a value with NO unit is accepted only when exactly one known unit puts it inside
  the range NHANES 1999-2002 actually observed — and then the assumed unit is
  shown — otherwise it asks for the unit;
* a value outside that range is refused, naming the unit that would fit if one does.

`ParsedPatient.to_patient(...)` then scores LinAge2 (core.linage2_model) and returns
the ordinary caller-patient payload `/query`, `/metta/run` and `build_patient`
already accept — with the same values doubling as the knowledge base's own
witnesses (CRP, HbA1c, fasting glucose, RDW and a low albumin, a fasting triglyceride, smoking status), so a cause can be credited
without typing anything twice.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from core.linage2_model import compute_linage2, load_model, young_reference_z
from core.patient_builder import MARKERS, Z_LIMIT, PatientSpecError
from core.patient_canonical import LIST_HEAD, Fact, parse
from core.patient_vocabulary import vocabulary

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
            ("triglycerides", "triglyceride", "tg", "trigs", "fasting triglycerides", "fasting triglyceride",
             "fasting tg")),
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
_FASTING_ALIASES = {"fasting glucose", "fasting blood glucose", "fasting plasma glucose", "fbg", "fpg",
                    "fasting triglycerides", "fasting triglyceride", "fasting tg"}

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


#: RDW and albumin reach the KB as a z against LinAge2's reference for people up to 50 (sex-specific, NOT
#: age-adjusted): name, LinAge2 code, and the sign that makes a positive z the harmful direction (a LOW albumin).
_YOUNG_REFERENCE_MARKERS = (("RDW", "LBXRDW", 1.0, "RDW"), ("LowSerumAlbumin", "LBDSALSI", -1.0, "albumin"))
#: Below one of these a raised RDW is what anaemia or a deficiency looks like (Bessman 1983, PMID 6881096; Förhécz
#: 2009, PMID 19781428), so it is not passed as a witness for inflammation. The usual laboratory lower limits: a
#: CONVENTION, not a result of a paper, chosen on the sensitive side (more values withheld, never fewer).
_RDW_WITHHELD_BELOW = (
    ("LBXHGB", "hemoglobin", "g/dL", {"Male": 13.0, "Female": 12.0}),
    ("LBDFERSI", "ferritin", "µg/L", {"Male": 30.0, "Female": 30.0}),
    ("LBDB12SI", "vitamin B12", "pmol/L", {"Male": 148.0, "Female": 148.0}),
    ("LBDFOLSI", "folate", "nmol/L", {"Male": 10.0, "Female": 10.0}),
)


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
    #: drugs the person takes now that the KB has an interaction fact for (core.patient_medications),
    #: as KB symbols; and those said NOT to be taken
    medications: list[str] = field(default_factory=list)
    medications_stopped: list[str] = field(default_factory=list)
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
        CRP mg/L, HbA1c %, fasting glucose and fasting triglycerides mg/dL (only when the text said fasting).

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
        # typed more than once (the same value twice: a different one is a duplicate and blocks): a fasting
        # reading is the witness, whichever line came first
        triglycerides = next((r for r in self.readings if r.code == "LBDSTRSI" and r.fasting and not r.blocking
                              and r.value is not None), None)
        if triglycerides is not None:
            raw["Triglycerides"] = (round(triglycerides.value / 0.01129, 1), "mg/dL")
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
        if self.sex in ("Male", "Female"):
            self._young_reference_markers(by_code, markers)
        markers.update(self.extra_markers)
        return markers

    def _young_reference_markers(self, by_code: dict, markers: dict) -> None:
        """RDW and albumin as z against LinAge2's reference (see _YOUNG_REFERENCE_MARKERS), with the notes
        that say what that means: it is stricter than a laboratory range, not age-adjusted, and a raised RDW
        is withheld when an anaemia or a deficiency explains it."""
        for name, code, sign, label in _YOUNG_REFERENCE_MARKERS:
            reading = by_code.get(code)
            if reading is None:
                continue
            z = sign * young_reference_z(code, reading.value, self.sex)
            if z <= 1.0:
                continue                       # not a witness either way; LinAge2 still has the value
            if name == "RDW":
                low = [f"{lab} {by_code[c].value:g} {unit}" for c, lab, unit, limits in _RDW_WITHHELD_BELOW
                       if c in by_code and by_code[c].value < limits[self.sex]]
                if low:
                    self.witness_notes.append(
                        f"RDW {reading.value:g} % was not passed on as a sign of inflammation: {', '.join(low)} "
                        f"{'is' if len(low) == 1 else 'are'} below the usual lower limit, and anaemia or an iron, B12 or folate deficiency "
                        f"raises RDW by itself. LinAge2 still uses the value as typed.")
                    continue
            capped = abs(z) > Z_LIMIT
            markers[name] = {"z": math.copysign(Z_LIMIT, z) if capped else z}
            sex = "men" if self.sex == "Male" else "women"
            self.witness_notes.append(
                f"'{reading.typed}' counts as {'low' if name == 'LowSerumAlbumin' else 'high'} here: it is z {z:.1f} "
                f"against LinAge2's reference for {sex} up to 50. That is stricter than a laboratory range, so a "
                f"value inside the usual range can still count, and it is not age-adjusted ({label} drifts with "
                f"age, so this overcalls older people). It is a hint of inflammation, not a finding."
                + (f" Beyond the reference's useful range; passed on as z {Z_LIMIT:g}." if capped else "")
                + (" Low protein intake and a recent meal also lower albumin." if name == "LowSerumAlbumin" else ""))

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
        chd = [_ITEM_LABEL[i] for i in _CHD_ITEMS if self.questionnaire.get(i) == 1]
        if chd:
            payload["prevalent_chd"] = chd     # the 10-year CHD risk does not read it; the builder emits ONE shared PatientCondition atom, for diagnose-patient only
        if self.medications:
            payload["medications"] = list(self.medications)    # KB symbols; shared atoms only
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
            "medications": list(self.medications),
            "medications_stopped": list(self.medications_stopped),
            "set_aside": list(self.set_aside),
            "witness_notes": list(self.witness_notes),
            "problems": self.all_problems(),
            "notes": list(self.notes),
            "ok": self.ok,
        }


# ═══════════════════════════ the questionnaire ═════════════════════════════════

_ITEM_LABEL = {
    "BPQ020": "hypertension", "DIQ010": "diabetes", "KIQ020": "kidney disease", "MCQ010": "asthma",
    "MCQ053": "anemia", "MCQ160A": "arthritis", "MCQ160B": "heart failure",
    "MCQ160C": "coronary heart disease", "MCQ160D": "angina", "MCQ160E": "heart attack",
    "MCQ160F": "stroke", "MCQ160G": "emphysema", "MCQ160I": "thyroid disease", "MCQ160J": "obesity",
    "MCQ160K": "chronic bronchitis", "MCQ160L": "liver condition", "MCQ220": "cancer",
    "OSQ010A": "hip fracture", "OSQ010B": "wrist fracture", "OSQ010C": "spine fracture",
    "OSQ060": "osteoporosis", "PFQ056": "memory problems", "HUQ070": "overnight hospital stay",
}

#: the items that put a person outside the 10-year CHD model's at-risk set (heart failure, MCQ160B, is
#: a different event); the labels are core.patient_builder.PREVALENT_CHD
_CHD_ITEMS = ("MCQ160C", "MCQ160D", "MCQ160E")
_FS1_ITEMS = ("BPQ020", "DIQ010", "KIQ020", "MCQ010", "MCQ053", "MCQ160A", "MCQ160B", "MCQ160C",
              "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K",
              "MCQ160L", "MCQ220", "OSQ010A", "OSQ010B", "OSQ010C", "OSQ060", "PFQ056", "HUQ070")
_HEALTH = {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}
_TREND = {"better": 1, "worse": 2, "same": 3, "about the same": 3, "unchanged": 3}


def _visits_category(n: int) -> int:
    """HUQ050's answer categories: 0, 1, 2-3, 4-9, 10-12, 13+."""
    return 0 if n <= 0 else 1 if n == 1 else 2 if n <= 3 else 3 if n <= 9 else 4 if n <= 12 else 5


def _cotinine_level(ng_ml: float) -> int:
    """`digiCot`, the binning the model was trained on."""
    return 0 if ng_ml < 10 else 1 if ng_ml < 100 else 2 if ng_ml < 200 else 3

# ═══════════════════════════ lab values ════════════════════════════════════════

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

# ═══════════════════════════ Facts -> what was read ════════════════════════════
# Both readers end here: the model (core.patient_read: free text -> verified Facts) and
# read_lines (canonical lines -> Facts). `assemble` applies every domain rule a reading
# answers to — units and ranges, contradictions, the smoking conventions and measured
# cotinine, the questionnaire, the checks on the whole person — in one place.

@dataclass
class Span:
    """A piece of the text and the Facts read from it. `start`/`end` are offsets into line
    `line` of the text as typed; `unread` marks a piece nothing was read from (it is listed
    as not understood)."""
    line: int
    start: int
    end: int
    text: str
    facts: list = field(default_factory=list)
    unread: bool = False


_NICOTINE_WHAT = {"vaping": "vaping", "nicotine_replacement": "nicotine replacement",
                  "smokeless": "smokeless tobacco"}
_ANSWER = {"yes": 1, "no": 2, "borderline": 3}
_VISIT_FACTOR = {"year": 1, "unstated": 1, "month": 12, "week": 52}
_SPECS_OF = dict(_ALIAS_INDEX)


def unify(text: str) -> str:
    """Typography as keyboards make it, one character for one (offsets still hold)."""
    return (text or "").translate(_UNIFY)


def number(text: str) -> Optional[float]:
    """A number as typed — '4.1', '4,1' (a decimal comma), '+4.5' — or None. '1,900' is None:
    a thousands separator and a decimal comma read 1000 times apart, so it is asked about."""
    t = unify(text).strip().replace(" ", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", t):
        return None
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None


def feet_inches(text: str) -> Optional[tuple[int, int]]:
    """5'3, 5'3", 5'3'', 5 ft 3 in -> (5, 3); None unless feet then inches 0-11."""
    m = re.fullmatch(r"\s*(\d)\s*(?:'|ft\.?|feet|foot)\s*(\d{1,2})\s*(?:\"|''|in\.?|inch(?:es)?)?\s*",
                     unify(text).lower())
    if not m or int(m.group(2)) > 11:
        return None
    return int(m.group(1)), int(m.group(2))


def assemble(spans: list[Span], lost=()) -> ParsedPatient:
    """What was read: the ParsedPatient for these Spans, in text order. `lost` holds what a
    reader found but could not check, as (quote, kind, why), for the kinds whose absence would
    be read as an answer (a condition, a smoking status): each is a question to the person.
    Never raises on content: everything that could not be used is reported on the result."""
    p = ParsedPatient()
    seen: dict[str, Reading] = {}
    diagnosed: set[str] = set()          # items a statement answered, Yes or No
    said_none = False                    # "no (other) conditions": every item not listed is a No
    smoking: list[tuple[Fact, Statement]] = []
    cur = Statement(-1, -1, 0, 0, "")

    def problem(text_: str, kind: str, topic: str, statements=None) -> None:
        p.problems.append(Problem(text_, kind, topic, (cur.index,) if statements is None else statements))

    def set_once(attr: str, value, what: str) -> None:
        cur.facts[attr] = value
        current = getattr(p, attr)
        if current is not None and current != value:
            problem(f"two different {what}s: {current} and {value} (from '{cur.text}')", "contradiction",
                    {"weight_kg": "weight", "height_cm": "height"}.get(attr, attr))
        else:
            setattr(p, attr, value)

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

    def answer(item: str, value: int, what: str, note: str) -> None:
        cur.facts.setdefault("questionnaire", {})[item] = value
        if item in p.questionnaire and p.questionnaire[item] != value:
            problem(f"two different {what} answers (from '{cur.text}'); keep one", "contradiction", "other")
        elif item not in p.questionnaire:
            p.questionnaire[item] = value
            p.questionnaire_notes.append(note)

    def unclear(f: Fact) -> None:
        why = f" ({f.detail})" if f.detail else ""
        if f.key == "smoking":
            problem(f"'{cur.text}': cannot tell what this says about your smoking{why}; write 'current smoker', "
                    f"'former smoker' or 'never smoked'", "ambiguous", "smoking")
        elif f.key == "condition":
            problem(f"'{cur.text}': cannot tell whether this is one of the conditions LinAge2 counts{why}; "
                    f"write it as 'diagnoses: …' or 'no …' with the condition's name", "ambiguous", "condition")
        elif f.key == "grimage":
            problem(f"'{cur.text}' does not say it is a GrimAge ACCELERATION with its sign{why}; give the "
                    f"acceleration (clock age minus your age), e.g. 'GrimAge acceleration +4 years'",
                    "ambiguous", "grimage")
        elif f.key == "weight":
            problem(f"'{cur.text}': give the weight's unit (kg or lb)", "unit", "weight")
        elif f.key == "height":
            problem(f"'{cur.text}': give the height's unit (cm, m or in, or 5'10\")", "unit", "height")
        elif f.key == "cotinine":
            problem(f"'{cur.text}': give cotinine in ng/mL (e.g. 'cotinine 250 ng/mL') or as "
                    f"'cotinine level 0-3'", "unit", "cotinine")
        elif f.key == "medication":
            p.notes.append(f"'{cur.text}': cannot tell whether you take it now{why}; it is not counted — "
                           f"say 'medications: metformin' or 'not taking metformin'")
        else:
            p.notes.append(f"could not read '{cur.text}'{why}")

    for idx, span in enumerate(sorted(spans, key=lambda s_: (s_.line, s_.start, s_.end))):
        cur = Statement(idx, span.line, span.start, span.end, span.text)
        p.statements.append(cur)
        if span.unread:
            p.not_understood.append(Remark(span.text, idx))
            cur.leftover = span.text
            continue
        for f in span.facts:
            if f.kind == "someone_else":
                p.set_aside.append(Remark(span.text, idx))
                cur.facts["set_aside"] = True
            elif f.kind == "unclear":
                unclear(f)
            elif f.kind == "smoking":
                smoking.append((f, cur))
            elif f.kind == "age":
                a = number(f.number)
                if a is None:
                    problem(f"'{span.text}': the age is not a number", "unit", "age")
                else:
                    set_once("age", a, "age")
            elif f.kind == "sex":
                set_once("sex", f.key, "sex")
            elif f.kind == "cotinine":
                v = number(f.number)
                if f.unit == "level":
                    if v not in (0, 1, 2, 3):
                        problem(f"'{span.text}': a cotinine level is 0, 1, 2 or 3", "unit", "cotinine")
                        continue
                    level, note = int(v), f"cotinine level {int(v)} as given"
                elif v is None or v < 0:
                    problem(f"'{span.text}': give cotinine in ng/mL (e.g. 'cotinine 250 ng/mL') or as "
                            f"'cotinine level 0-3'", "unit", "cotinine")
                    continue
                else:
                    level, note = _cotinine_level(v), f"cotinine {v:g} ng/mL -> level {_cotinine_level(v)}"
                cur.facts["cotinine"] = level
                if p.cotinine_measured and p.cotinine_level != level:
                    problem(f"two different cotinine levels: {p.cotinine_level} and {level}", "contradiction",
                            "cotinine")
                    continue
                p.cotinine_level, p.cotinine_note, p.cotinine_measured = level, note, True
            elif f.kind == "lab":
                group = vocabulary().labs[f.key]
                v = number(f.number)
                if v is None:
                    problem(f"'{span.text}': write the value as a plain number (4.1, or 1900 without a "
                            f"thousands comma)", "unit", "lab")
                    continue
                add(_resolve(_SPECS_OF[group.aliases[0]], span.text, v, unify(f.unit), fasting=group.fasting))
            elif f.kind == "weight":
                w = number(f.number)
                kg = None if w is None else w * 0.45359237 if f.unit == "lb" else w
                if kg is None or not 25 <= kg <= 350:
                    problem(f"'{span.text}' = {kg:.0f} kg, outside 25-350 kg; check the value and unit"
                            if kg is not None else f"'{span.text}': the weight is not a number", "unit", "weight")
                else:
                    set_once("weight_kg", kg, "weight")
            elif f.kind == "height":
                if f.unit == "ft-in":
                    fi = feet_inches(f.number)
                    cm = None if fi is None else (fi[0] * 12 + fi[1]) * 2.54
                else:
                    h = number(f.number)
                    cm = None if h is None else h * 100 if f.unit == "m" else h * 2.54 if f.unit == "in" else h
                if cm is None or not 100 <= cm <= 230:
                    problem(f"'{span.text}' = {cm:.0f} cm, outside 100-230 cm; check the value and unit"
                            if cm is not None else f"'{span.text}': the height is not a height (5'10\", 178 cm)",
                            "unit", "height")
                else:
                    set_once("height_cm", cm, "height")
            elif f.kind == "condition":
                value = _ANSWER[f.answer]
                label = _ITEM_LABEL[f.key]
                if value == 3 and f.key != "DIQ010":
                    problem(f"'{span.text}': LinAge2's question about {label} has no borderline answer; write "
                            f"'diagnoses: {label}' or 'no {label}'", "ambiguous", "condition")
                    continue
                prev = p.questionnaire.get(f.key) if f.key in diagnosed else None
                if prev is not None and prev != value:
                    if {prev, value} == {2, 3}:
                        value = 3                   # "no diabetes" and "prediabetes": borderline
                    else:
                        problem(f"'{span.text}' answers {label} differently from an earlier line; keep one",
                                "contradiction", "condition")
                        continue
                cur.facts.setdefault("conditions", {})[f.key] = value
                p.questionnaire[f.key] = value
                diagnosed.add(f.key)
            elif f.kind == "no_other_conditions":
                said_none = True
                cur.facts["no_other_conditions"] = True
            elif f.kind == "medication":
                now = f.answer == "now"
                mine, other = (p.medications, p.medications_stopped) if now else (p.medications_stopped,
                                                                                    p.medications)
                cur.facts.setdefault("medications", {})[f.key] = "current" if now else "not_current"
                if f.key in other:
                    problem(f"'{span.text}' says you {'take' if now else 'do not take'} {f.key} but another "
                            f"line says the opposite; keep one", "contradiction", "medication")
                elif f.key not in mine:
                    mine.append(f.key)
            elif f.kind == "self_rated_health":
                answer("HUQ010", _HEALTH[f.key], "self-rated health", f"self-rated health: {f.key}")
            elif f.kind == "health_vs_year_ago":
                answer("HUQ020", _TREND[f.key], "health compared to a year ago",
                       f"health vs a year ago: {f.key}")
            elif f.kind == "healthcare_visits":
                n = number(f.number)
                if n is None or n < 0 or n != int(n):
                    problem(f"'{span.text}': the number of visits is not a count", "unit", "other")
                    continue
                visits = int(n) * _VISIT_FACTOR[f.key or "year"]
                answer("HUQ050", _visits_category(visits), "healthcare visits",
                       f"healthcare visits {visits} -> NHANES category {_visits_category(visits)}")
            elif f.kind == "grimage":
                years = number(f.number)
                if years is None or abs(years) > 30:
                    problem(f"GrimAge acceleration {f.number} years is outside anything the clock produces "
                            f"(±30); check the value", "unit", "grimage")
                    continue
                cur.facts["grimage"] = years
                prev = p.extra_markers.get("AgeAccelGrim")
                if prev is not None and prev["value"] != years:
                    problem(f"two different GrimAge accelerations: {prev['value']:+g} and {years:+g}; keep one",
                            "contradiction", "grimage")
                    continue
                p.extra_markers["AgeAccelGrim"] = {"value": years, "unit": "years"}
                p.notes.append(f"GrimAge acceleration {years:+g} years: the knowledge base's 10-year "
                               f"heart-disease model reads it; it is never combined with LinAge2")
            else:
                raise ValueError(f"no rule for a {f.kind!r} fact")

    for quote, kind, why in lost:
        if kind == "smoking":
            p.problems.append(Problem(
                f"'{quote}': this was read as your smoking but could not be checked against your text ({why}); "
                f"say your own smoking on its own line ('current smoker', 'former smoker' or 'never smoked')",
                "ambiguous", "smoking", ()))
        else:
            p.problems.append(Problem(
                f"'{quote}': a condition was read here but could not be checked against your text ({why}), so "
                f"it would be answered wrongly; write it as 'diagnoses: …' or 'no …' with the condition's name",
                "lost_condition", "condition", ()))

    # ── smoking, after the measurements: a measured cotinine always wins over words ──
    for f, cur in smoking:
        what = _NICOTINE_WHAT.get(f.detail)
        if what:
            if p.cotinine_measured:
                p.notes.append(f"'{cur.text}': {what} — the measured cotinine is used for it")
            else:
                problem(f"'{cur.text}': {what} raises cotinine, which LinAge2 reads, but is not smoking to the "
                        f"knowledge base; give 'cotinine N ng/mL' from a test, or remove the {what} to be "
                        f"read without it", "vaping", "smoking")
        elif f.detail == "cannabis":
            problem(f"'{cur.text}': cannabis is not tobacco, and LinAge2 reads tobacco exposure (cotinine); "
                    f"say your tobacco smoking on its own ('never smoked', 'former smoker', 'current smoker')",
                    "ambiguous", "smoking")
        elif f.detail == "secondhand":
            p.notes.append(f"'{cur.text}': second-hand smoke is not smoking; it is not counted")
        if f.key == "unclear":
            if not what and f.detail not in ("cannabis", "secondhand"):
                problem(f"'{cur.text}': cannot tell whether you smoke now, used to, or never did; write "
                        f"'current smoker', 'former smoker, quit 2010' or 'never smoked'", "ambiguous", "smoking")
            continue
        level = {"occasional": 1, "moderate": 2}.get(f.amount, 3) if f.key == "CurrentSmoker" else 0
        before = p.smoking
        set_once("smoking", f.key, "smoking status")
        cur.facts["smoking"] = (f.key, level)
        if p.smoking != f.key:
            continue
        if p.cotinine_measured:
            if level != p.cotinine_level:
                p.notes.append(f"'{cur.text}' would put cotinine at level {level}; the measured value is used "
                               f"({p.cotinine_note})")
        elif before == f.key and p.cotinine_level not in (None, level):
            problem(f"two different smoking intensities: cotinine level {p.cotinine_level} and {level} (from "
                    f"'{cur.text}'); keep one ('current smoker', 'moderate smoker', 'occasional smoker')",
                    "contradiction", "smoking")
        else:
            p.cotinine_level = level
            p.cotinine_note = (f"cotinine level {level} from '{cur.text}' (training bins: 0 <10, 1 10-100, "
                               f"2 100-200, 3 >=200 ng/mL)")

    # ── the questionnaire: a list of diagnoses answers the rest No ───────────────
    if diagnosed or said_none:
        typed = {}                       # what was typed for an item, when it is not the item's name
        for st in p.statements:
            for item in st.facts.get("conditions", {}):
                if _ITEM_LABEL[item].lower() not in st.text.lower():
                    typed.setdefault(item, st.text)

        def named(i: str) -> str:
            return _ITEM_LABEL[i] + (f" ('{typed[i]}')" if i in typed else "")

        yes = [named(i) + (" (borderline)" if p.questionnaire[i] == 3 else "")
               for i in _FS1_ITEMS if i in diagnosed and p.questionnaire[i] != 2]
        no = [named(i) for i in _FS1_ITEMS if i in diagnosed and p.questionnaire[i] == 2]
        for q in _FS1_ITEMS:
            p.questionnaire.setdefault(q, 2)
        p.questionnaire_notes.insert(0, (
            "diagnoses: " + (", ".join(yes) if yes else "none")
            + (f"; not: {', '.join(no)}" if no else "")
            + " — every diagnosis not listed is answered No"))
    if p.medications:
        p.notes.append(f"medication: {', '.join(p.medications)} read as a current medication — used only "
                       f"to flag supplement interactions; no ranking and no LinAge2 number changes")
    not_taken = [m for m in p.medications_stopped if m not in p.medications]
    if not_taken:
        p.notes.append(f"not counted as a current medication (you said you do not take it): {', '.join(not_taken)}")

    # ── the whole person ─────────────────────────────────────────────────────────
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
    tg_readings = [r for r in p.readings if r.code == "LBDSTRSI" and not r.blocking]
    if tg_readings and not any(r.fasting for r in tg_readings):
        has_ldl = any(r.code == "LDLV" and not r.blocking for r in p.readings)
        p.notes.append("triglycerides were not marked fasting: "
                       + ("LinAge2 does not use them (you gave an LDL), and " if has_ldl
                          else "LinAge2 uses them as typed (in the calculated LDL), but ")
                       + "the knowledge base's Triglycerides witness needs a fasting value, which can run about 27 mg/dL "
                         "lower — write 'fasting triglycerides …' if it was")
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


# ═══════════════════════════ canonical lines, no model ═════════════════════════

def _pieces(line: str):
    """(start, end, piece) of each piece of a line: a 'diagnoses:' or 'medications:' list is
    one piece; any other line splits at ';' and at ', '."""
    lead = len(line) - len(line.lstrip(" \t-•*·"))
    body = line[lead:].rstrip()
    if not body:
        return
    if LIST_HEAD.match(body):
        yield lead, lead + len(body), body
        return
    start = 0
    for m in list(re.finditer(r";\s*|,\s+", body)) + [None]:
        end = m.start() if m else len(body)
        piece = body[start:end]
        if piece.strip():
            a = lead + start + len(piece) - len(piece.lstrip())
            yield a, a + len(piece.strip()), piece.strip()
        if m:
            start = m.end()


def read_lines(text: str) -> ParsedPatient:
    """Read canonical lines (core.patient_canonical) with no model: a piece that is not one
    of the canonical forms is listed as not understood, never guessed."""
    spans = []
    for line_no, line in enumerate(unify(text).splitlines()):
        for a, b, piece in _pieces(line):
            facts = parse(piece)
            spans.append(Span(line_no, a, b, piece, facts or [], unread=facts is None))
    return assemble(spans)


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
