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
    assert len(v.conditions) == 23 and v.conditions["BPQ020"] == "hypertension"
    # one lab entry per alias group: the reader, not the model, picks the input from the unit
    assert COTININE not in v.labs and v.labs["albumin"].codes == ("LBDSALSI",)
    assert v.labs["urea"].aliases != v.labs["urea nitrogen (bun)"].aliases
    assert v.labs["urea"].codes == v.labs["urea nitrogen (bun)"].codes == ("LBDSBUSI",)
    assert v.labs["lymphocytes"].codes == ("LBXLYPCT", "LBDLYMNO")
    assert "%" in v.labs["lymphocytes"].units and "10⁹/L" in v.labs["lymphocytes"].units
    assert v.labs["fasting glucose"].fasting and not v.labs["glucose"].fasting


def test_drift_catches_a_description_in_the_wrong_unit(tmp_path, monkeypatch):
    """The model file as extracted from upstream gave urine creatinine in mmol/L, CRP in
    mg/L and cotinine as 0-2; the model would have been told the wrong unit."""
    import json

    import core.patient_vocabulary as pv

    raw = json.loads(pv.LINAGE2_MODEL_FILE.read_text(encoding="utf-8"))
    raw["descriptions"]["URXUCRSI"] = "Urine creatinine, SI units (mmol/L)."
    raw["descriptions"]["LBXCRP"] = "C-reactive protein (mg/L)."
    raw["descriptions"]["LBXCOT"] = "Smoking status: 0 - Non-smoker, 1 - Light/Recent, 2 - Heavy/Current"
    bad = tmp_path / "model.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pv, "LINAGE2_MODEL_FILE", bad)
    pv.vocabulary.cache_clear()
    try:
        found = drift()
    finally:
        pv.vocabulary.cache_clear()
    assert any("URXUCRSI in mmol/L" in d for d in found), found
    assert any("LBXCRP in mg/L" in d for d in found), found
    assert any("cotinine as the levels 0-3" in d for d in found), found
