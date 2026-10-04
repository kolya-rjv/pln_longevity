"""The words a model may use when it reads a patient: the knowledge base's own.

The My Patient tab can have a language model read free text (core.patient_extract). The
model never writes MeTTa and never chooses a symbol of its own: every identifier it may
return is an enum generated here from the files the rest of the system already trusts —

    smoking status   the `(: X SmokingStatus)` declarations in patient_profile.metta
    LinAge2 inputs   the `(ModelInput <Symbol> LinAge2) ;; <NHANES code> — <description>`
                     lines of linage2_core.metta (59 inputs)
    lab codes        the inputs a person can type a value for: the LinAge2 model file's
                     lab inputs and the sources of its derived inputs, as the reader's unit
                     table (core.patient_text.SPECS) knows them, plus serum cotinine
    conditions       the 23 NHANES questionnaire items behind fs1Score

`drift()` lists every way these sources disagree with each other or with the reader and
the patient builder; tests/test_patient_extract.py requires it to be empty, so editing a
.metta file without the code (or the reverse) fails a test instead of a patient.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import ONTOLOGY_DIR

LINAGE2_MODEL_FILE = ONTOLOGY_DIR / "data" / "linage2" / "linage2_model.json"
_STATUS_RE = re.compile(r"^\(:\s+(\w+)\s+SmokingStatus\)", re.M)
_INPUT_RE = re.compile(r"\(ModelInput (\w+) LinAge2\)[^\n]*?;;\s*(\w+)\s*—\s*([^\n]*)")
#: the questionnaire items that are not diagnoses (self-rated health, its trend, visits)
HEALTH_ITEMS = ("HUQ010", "HUQ020", "HUQ050")
COTININE = "LBXCOT"


@dataclass(frozen=True)
class Vocabulary:
    smoking_statuses: tuple[str, ...]
    #: NHANES code -> (KB symbol, description), as linage2_core.metta declares them
    model_inputs: dict
    #: code -> (label, unit the model takes, accepted units, typical names)
    labs: dict
    #: NHANES item -> plain label, the 23 conditions LinAge2's comorbidity score reads
    conditions: dict


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def vocabulary() -> Vocabulary:
    from core.patient_text import SPECS, _ITEM_LABEL, display_unit

    statuses = tuple(_STATUS_RE.findall(_read(ONTOLOGY_DIR / "patient_profile.metta")))
    inputs = {code: (symbol, desc.strip())
              for symbol, code, desc in _INPUT_RE.findall(_read(ONTOLOGY_DIR / "linage2_core.metta"))}
    model = json.loads(_read(LINAGE2_MODEL_FILE))
    labs = {}
    for code, spec in SPECS.items():
        labs[code] = (spec.label, spec.unit, sorted({display_unit(u) for u in spec.units}),
                      list(spec.aliases))
    labs[COTININE] = ("Serum cotinine (smoking exposure)", "ng/mL", ["ng/mL"],
                      ["cotinine", "serum cotinine"])
    items = [q for q in model["questionnaire_items"] if q not in HEALTH_ITEMS]
    conditions = {q: _ITEM_LABEL.get(q, q) for q in items}
    return Vocabulary(statuses, inputs, labs, conditions)


def drift() -> list[str]:
    """Every disagreement between the KB, the LinAge2 model file, the reader and the
    builder that would let the model's vocabulary and the knowledge base part ways."""
    from core.patient_builder import SMOKING
    from core.patient_text import SPECS, _FS1_ITEMS, _ITEM_LABEL

    v = vocabulary()
    model = json.loads(_read(LINAGE2_MODEL_FILE))
    out: list[str] = []
    if set(v.smoking_statuses) != set(SMOKING):
        out.append(f"patient_profile.metta declares smoking statuses {sorted(v.smoking_statuses)}, "
                   f"the patient builder accepts {sorted(SMOKING)}")
    features = {f["code"] for f in model["features"]}
    if set(v.model_inputs) != features:
        out.append(f"linage2_core.metta ModelInput codes and the model file's features differ: "
                   f"{sorted(set(v.model_inputs) ^ features)}")
    derived = model["derived"]
    sources = {s for d in derived.values() for s in d["from"]}
    typed = set(SPECS) | {COTININE}
    for code in v.model_inputs:
        if code not in typed and code not in derived:
            out.append(f"KB input {code} can be neither typed nor derived")
    for code in model["lab_inputs"]:
        if code not in typed:
            out.append(f"model lab input {code} has no unit table in the reader")
    for code in sources - set(model["questionnaire_items"]):
        if code not in typed:
            out.append(f"derived-input source {code} has no unit table in the reader")
    for code in set(SPECS) - set(model["lab_inputs"]) - sources:
        if code not in v.model_inputs:
            out.append(f"the reader reads {code}, which no model input uses")
    if set(_FS1_ITEMS) != set(v.conditions):
        out.append(f"the reader's 23 conditions and the model file's questionnaire differ: "
                   f"{sorted(set(_FS1_ITEMS) ^ set(v.conditions))}")
    if set(_ITEM_LABEL) != set(v.conditions):
        out.append("every condition needs a plain label in the reader")
    return out
