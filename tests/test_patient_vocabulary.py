"""The model that reads a patient may only use the knowledge base's own words.

core.patient_vocabulary builds every identifier the extraction schema allows from the
.metta files and the LinAge2 model file; these tests pin that those sources, the reader's
unit table and the patient builder agree, so neither side can drift alone.

    pytest tests/test_patient_vocabulary.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from core.patient_vocabulary import COTININE, drift, vocabulary  # noqa: E402


def test_the_kb_the_model_file_the_reader_and_the_builder_agree():
    assert drift() == []


def test_the_vocabulary_is_the_kbs():
    v = vocabulary()
    assert v.smoking_statuses == ("NeverSmoker", "FormerSmoker", "CurrentSmoker")
    assert len(v.model_inputs) == 59 and v.model_inputs["LBDSALSI"][0] == "SerumAlbumin"
    assert COTININE in v.labs and "LBDSALSI" in v.labs and len(v.conditions) == 23
    assert v.conditions["BPQ020"] == "hypertension"
