"""The "My Patient" tab (pln_chat/patient_tab.py) and its hand-off to the chat.

The tab's promise: what you typed is read back before anything is built; the
patient lives in one browser session (a gr.State), never in the knowledge base;
and a question about "me" in PLN Query reaches that patient exactly as /query
would — LinAge2 atoms to the LinAge2 space only.

    pytest tests/test_patient_tab.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("gradio")

import patient_tab  # noqa: E402
from core.patient_text import EXAMPLES  # noqa: E402

SMOKER = EXAMPLES["58-year-old smoker"]


def _build(text: str, state=None):
    reading, summary, atoms, download, banner, new_state, suggestions = patient_tab.on_build(text, state)
    return dict(reading=reading, summary=summary, atoms=atoms, download=download,
                banner=banner, state=new_state, suggestions=suggestions)


def test_read_shows_every_value_with_its_conversion_and_the_typical_value():
    md = patient_tab.on_read(SMOKER)
    assert "| Albumin | `albumin 4.1 g/dL` | 41 g/L | ✓" in md
    assert "| C-reactive protein | `CRP 3.1 mg/L` | 0.31 mg/dL | ✓" in md
    assert "Ready to build" in md
    assert "Smoking → cotinine level 3" in md


def test_read_stops_on_an_ambiguous_unit_and_says_why():
    md = patient_tab.on_read("58 year old male\nCRP 3.1")
    assert "needs a unit" in md and "Fix before building" in md and "Ready to build" not in md


def test_build_makes_a_session_patient_and_never_touches_the_kb(tmp_path):
    out = _build(SMOKER)
    state = out["state"]
    assert state["id"] == "Me" and state["sex"] == "Male" and state["smoking"] == "CurrentSmoker"
    assert "linage2" in state and state["markers"]["CRP"] == {"value": 3.1, "unit": "mg/L"}
    assert "LinAge2 biological age" in out["summary"] and "Caller_Me" in out["summary"]
    assert "Active patient: Caller_Me" in out["banner"]
    atoms = out["atoms"]["value"]
    assert "(LinAgeDelta Caller_Me " in atoms and "(PatientSmoking Caller_Me CurrentSmoker)" in atoms
    # the download is a temp file, nowhere near a folder the KB loaders scan
    path = Path(out["download"]["value"])
    assert path.exists() and path.suffix == ".metta"
    assert REPO not in path.parents


def test_a_failed_build_keeps_the_previous_patient():
    first = _build(SMOKER)["state"]
    out = _build("albumin 4.1 g/dL", first)                     # no age, no sex
    assert out["state"] is first and "Not built" in out["summary"]


def test_clear_forgets_the_patient():
    summary, atoms, download, banner, state, suggestions = patient_tab.on_clear()
    assert state is None and "No patient loaded" in banner


def test_every_example_builds():
    for name, text in EXAMPLES.items():
        assert _build(text)["state"] is not None, name


# ═══════════════════════════ the chat sees the session patient ═══════════════

def _chat_with(monkeypatch, form: str, state):
    import app as app_module
    from core.llm_translator import TranslationResult
    from core.pln_runner import PLNRunResult
    seen = {}

    def translate(**kw):
        seen["prompt"] = kw["system_prompt"]
        return TranslationResult(metta_query=form, explanation="", intent="inference",
                                 requires_pln_inference=True, confidence_filter=0.0)

    def offload(task, kwargs, inline, **kw):
        seen.setdefault("calls", []).append((task, kwargs))
        empty = PLNRunResult(status="empty", mode="runtime")
        return [empty, empty] if task == "run_query_parts" else empty

    monkeypatch.setattr(app_module, "translate", translate)
    monkeypatch.setattr(app_module, "run_offloaded", offload)
    monkeypatch.setattr(app_module, "log_query", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "log_turn", lambda *a, **k: None)
    history, _ = app_module.chat(
        user_message="q", history=[], selected_files=[], model="m", temperature=0.0,
        confidence_threshold=0.0, show_metta=False, show_explanation=False, show_debug=False,
        patient_state=state)
    return app_module, seen, history


def test_a_question_about_me_reaches_the_session_patient_like_query_does(monkeypatch):
    from core.pln_runner import patient_stack
    state = _build(SMOKER)["state"]
    app_module, seen, history = _chat_with(
        monkeypatch, "(diagnose-patient &self Caller_Me (InsulinResistance))", state)
    assert "--- THIS REQUEST'S PATIENT ---" in seen["prompt"] and "it means Caller_Me" in seen["prompt"]
    assert "linage-decomposition-patient &self Caller_Me" in seen["prompt"]
    (task, kwargs), = seen["calls"]
    assert kwargs["kb_files"] == patient_stack(app_module._ALL_KB_PATHS)
    assert "(PatientSmoking Caller_Me CurrentSmoker)" in kwargs["extra_atoms"]
    assert "LinAge" not in kwargs["extra_atoms"]                # never in the shared space
    assert "Validation Issues" not in history[-1]["content"]     # Caller_Me is known


def test_a_linage2_question_gets_every_atom_and_a_mixed_one_is_split(monkeypatch):
    state = _build(SMOKER)["state"]
    _, seen, _ = _chat_with(monkeypatch, "(linage-hazard-patient &self Caller_Me)", state)
    (task, kwargs), = seen["calls"]
    assert task == "run_query" and "(LinAgeDelta Caller_Me " in kwargs["extra_atoms"]
    _, seen, _ = _chat_with(monkeypatch, "(linage-drivers-patient &self Caller_Me)\n"
                                         "(recommend-supplements-patient &self Caller_Me)", state)
    (task, kwargs), = seen["calls"]
    lin, gen = kwargs["parts"]
    assert task == "run_query_parts"
    assert "LinAgeContribution" in lin["extra_atoms"] and "LinAge" not in gen["extra_atoms"]


def test_without_a_patient_the_chat_is_as_before(monkeypatch):
    _, seen, _ = _chat_with(monkeypatch, "(predict-risk-patient &self Patient001)", None)
    assert "--- THIS REQUEST'S PATIENT ---" not in seen["prompt"]
    assert seen["calls"][0][1]["extra_atoms"] is None


# ═══════════════════════════ the same, over HTTP ══════════════════════════════

def test_the_api_reads_text_into_the_same_patient(monkeypatch):
    import asyncio
    import httpx
    import api as api_module
    import core.executor as executor
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)

    async def post(path, body):
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=300) as c:
            return await c.post(path, json=body)

    r = asyncio.run(post("/patients/from-text", {"text": SMOKER}))
    body = r.json()
    assert r.status_code == 200 and body["ok"] and body["problems"] == []
    assert body["preview"]["patient_id"] == "Caller_Me" and body["preview"]["has_linage2"]
    assert body["patient"] == patient_tab.on_build(SMOKER, None)[5]       # the tab's patient
    unclear = asyncio.run(post("/patients/from-text", {"text": "58 year old male\nCRP 3.1"})).json()
    assert unclear["ok"] is False and unclear["patient"] is None
    assert any("needs a unit" in p or "could be" in p for p in unclear["problems"])
    # and the returned patient is accepted as-is by the other endpoints
    again = asyncio.run(post("/patients/preview", body["patient"]))
    assert again.status_code == 200 and again.json()["atoms"] == body["preview"]["atoms"]


def test_each_build_gets_its_own_download_file():
    a = Path(_build(SMOKER)["download"]["value"])
    b = Path(_build(SMOKER)["download"]["value"])
    assert a != b and a.exists() and b.exists()


def test_the_tab_the_chat_and_the_api_build_the_same_patient(monkeypatch):
    """One builder (core.patient_context.build_caller_patient), the KB's own knobs."""
    import api as api_module
    state = _build(SMOKER)["state"]
    tab_atoms = _build(SMOKER)["atoms"]["value"]
    api_atoms = api_module._build_caller_patient(state).atoms
    assert tab_atoms == api_atoms


def test_a_session_patient_that_cannot_be_rebuilt_is_said_not_dropped(monkeypatch):
    _, seen, history = _chat_with(monkeypatch, "(predict-risk-patient &self Patient001)",
                                  {"id": "bad id", "age": 58, "sex": "Male"})
    assert "--- THIS REQUEST'S PATIENT ---" not in seen["prompt"]
    assert "could not be rebuilt" in history[-1]["content"]


def test_from_text_refuses_a_bad_id_with_a_422_not_a_500():
    import asyncio
    import httpx
    import api as api_module

    async def post(body):
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60) as c:
            return await c.post("/patients/from-text", json=body)

    assert asyncio.run(post({"text": SMOKER, "id": "X" * 60})).status_code == 422
    assert asyncio.run(post({"text": SMOKER, "id": "bad id"})).status_code == 422


def test_the_models_caveats_reach_the_tab():
    summary = _build("30 year old female\nalbumin 4.2 g/dL")["summary"]
    assert "⚠ Age 30 is outside the ages LinAge2 was fitted on" in summary
    assert "Questionnaire not answered" in summary


def test_the_notes_speak_to_someone_who_typed_text():
    summary = _build("60 year old female\ncreatinine 0.9 mg/dL")["summary"]
    assert "None of your values can be a knowledge-base witness" in summary
    assert "block was sent" not in summary and "Send `z`" not in summary
    assert "GrimAge acceleration +3 years" in summary
    former = _build(EXAMPLES["six labs only"])["summary"]
    assert "returns 0 for a former smoker" in former and "does not need it" not in former
    assert "turned into z-scores against one pooled reference" in former


def test_old_downloads_are_pruned(monkeypatch):
    import os
    import time
    old = Path(_build(SMOKER)["download"]["value"])
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    new = Path(_build(SMOKER)["download"]["value"])
    assert new.exists() and not old.exists() and new.parent == old.parent



def test_the_questionnaire_caveat_names_only_what_was_assumed():
    summary = _build("62 year old female, never smoked\ndiagnoses: hypertension\nhealth: poor\n"
                     "albumin 4.0 g/dL")["summary"]
    assert "no healthcare visits in the past year (fs3Score)" in summary
    assert "assumed no diagnoses" not in summary and "'good' self-rated health" not in summary


# ═══════════════════════════ the model reader in the tab ═══════════════════════
# Read may use a model; Build never does. Recorded extractions only.

class _Recorded:
    model = "recorded"

    def __init__(self, *items):
        from core.patient_extract import Extraction
        self.items, self.Extraction, self.calls = list(items), Extraction, 0

    def __call__(self, text):
        self.calls += 1
        return self.Extraction(list(self.items), "recorded")


_TEXT = "58 yo M\nalbumin 4.1 g/dL\nmy CRP was 3.1 mg/L"
_ITEMS = ({"quote": "58 yo M", "kind": "age"}, {"quote": "58 yo M", "kind": "sex", "sex": "male"},
          {"quote": "CRP was 3.1 mg/L", "kind": "lab", "lab": "c-reactive protein"})


def test_read_marks_model_rows_and_leaves_rules_rows_byte_identical():
    md, read, *buttons = patient_tab.on_read_model(_TEXT, _Recorded(*_ITEMS))
    assert md.startswith("### What was read\n<sub>Read by rules + recorded</sub>")
    assert "**Age** 58 (model) · **Sex** Male (model)" in md
    assert "| C-reactive protein | `my CRP was 3.1 mg/L` | 0.31 mg/dL | ✓ · model |" in md
    rules_row = "| Albumin | `albumin 4.1 g/dL` | 41 g/L | ✓ | 45 g/L |"
    assert rules_row in md and rules_row in patient_tab.on_read("58 year old male\nalbumin 4.1 g/dL")
    assert "**Read as**" in md and "58 year old; male\nalbumin 4.1 g/dL\nC-reactive protein 3.1 mg/L" in md
    assert all(b["visible"] is False for b in buttons)


def test_the_first_reading_and_the_examples_are_the_rules_alone():
    md = patient_tab.on_read(SMOKER)
    assert "<sub>Read by rules only</sub>" in md
    text, md2, read, *_ = patient_tab.on_example("58-year-old smoker")
    assert text == SMOKER and read.reader == "rules" and "Read by rules only" in md2


def test_a_model_error_says_why_and_reads_by_rules():
    from core.patient_extract import ExtractError

    def failing(text):
        raise ExtractError("timeout", "slow")
    md, read, *_ = patient_tab.on_read_model(_TEXT, failing)
    assert "<sub>Read by rules only — the model did not answer in time</sub>" in md
    assert read.reader == "rules" and read.parsed.sex is None


def test_build_uses_the_stored_reading_and_never_the_model():
    extractor = _Recorded(*_ITEMS)
    _, read, *_ = patient_tab.on_read_model(_TEXT, extractor)
    out = _build(_TEXT, None) if False else dict(zip(
        ("reading", "summary", "atoms", "download", "banner", "state", "suggestions"),
        patient_tab.on_build(_TEXT, None, read)))
    assert extractor.calls == 1                                   # Read only
    assert out["state"]["sex"] == "Male" and out["state"]["markers"]["CRP"] == {"value": 3.1, "unit": "mg/L"}
    header = Path(out["download"]["value"]).read_text(encoding="utf-8")
    assert ";; Read as (the fixed rules alone rebuild this patient from these lines):\n;;   58 year old; male" in header
    # the rules alone, on the 'read as' lines, build the same patient
    again = dict(zip(("reading", "summary", "atoms", "download", "banner", "state", "suggestions"),
                     patient_tab.on_build(read.read_as, None)))
    assert again["atoms"]["value"] == out["atoms"]["value"]


def test_build_refuses_text_that_changed_since_read():
    _, read, *_ = patient_tab.on_read_model(_TEXT, _Recorded(*_ITEMS))
    first = _build(SMOKER)["state"]
    out = patient_tab.on_build(_TEXT + "\nHbA1c 6.1 %", first, read)
    assert "The text changed since Read" in out[1] and out[5] is first


def test_a_suggestion_button_writes_the_wording_and_the_next_read_uses_it():
    text = "58 year old male\nI never quit smoking"
    smoking = {"quote": "I never quit smoking", "kind": "smoking", "status": "current", "occasional": False,
               "other_nicotine": "none"}
    md, read, *buttons = patient_tab.on_read_model(text, _Recorded(smoking))
    assert "Fix before building" in md and buttons[0]["visible"] is True
    assert buttons[0]["value"] == "Use “current smoker” for “I never quit smoking”"
    new_text = patient_tab.on_use_suggestion(text, read, 0)
    assert new_text == "58 year old male\ncurrent smoker"
    md2, read2, *_ = patient_tab.on_read_model(new_text, _Recorded())
    assert "Ready to build" in md2 and read2.parsed.smoking == "CurrentSmoker"
    assert patient_tab.on_use_suggestion("edited", read, 0) == "edited"   # stale: nothing changes


def test_the_reading_survives_the_session_state_copy():
    import copy
    _, read, *_ = patient_tab.on_read_model(_TEXT, _Recorded(*_ITEMS))
    clone = copy.deepcopy(read)
    assert clone.read_as == read.read_as and clone.parsed.as_dict() == read.parsed.as_dict()
    from core.patient_extract import ExtractError
    err = copy.deepcopy(ExtractError("timeout", "slow"))
    assert (err.code, err.message) == ("timeout", "slow")


def test_the_read_note_says_where_the_text_goes(monkeypatch):
    import config
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    assert "fixed rules only (no OPENAI_API_KEY)" in patient_tab.read_note()
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "PLN_EXTRACT_MODEL", "gpt-6-luna")
    assert "sends your text to OpenAI (gpt-6-luna)" in patient_tab.read_note()
    monkeypatch.setattr(config, "PLN_EXTRACT_MODEL", "gpt-4-turbo")
    assert "misconfigured" in patient_tab.read_note()


def _post(body):
    import asyncio

    import httpx

    import api as api_module

    async def go():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=120) as c:
            return await c.post("/patients/from-text", json=body)
    return asyncio.run(go())


def test_the_api_reads_by_rules_unless_asked_and_says_so(monkeypatch):
    import core.executor as executor
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)
    body = _post({"text": SMOKER}).json()
    assert body["reader_used"] == "rules" and body["read_as_text"] == SMOKER and body["model_error"] is None
    assert body["suggestions"] == [] and {s["source"] for s in body["statements"]} == {"rules"}


def test_the_api_model_reader_needs_a_key_and_a_short_text(monkeypatch):
    import config
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    r = _post({"text": SMOKER, "reader": "model"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_reader_not_configured"
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "PLN_EXTRACT_MODEL", "gpt-6-luna")
    r = _post({"text": "x" * (config.PLN_EXTRACT_MAX_CHARS + 1), "reader": "model"})
    assert r.status_code == 413
    assert _post({"text": SMOKER, "reader": "llm"}).status_code == 422


def test_the_api_model_reader_rewrites_and_suggests(monkeypatch):
    import config
    import core.executor as executor
    from core import patient_extract
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "PLN_EXTRACT_MODEL", "gpt-6-luna")
    recorded = _Recorded(*_ITEMS)
    monkeypatch.setattr(patient_extract.OpenAIExtractor, "__call__", lambda self, text: recorded(text))
    body = _post({"text": _TEXT, "reader": "model"}).json()
    assert body["ok"] and body["reader_used"] == "rules+model" and body["model_error"] is None
    assert body["read_as_text"] == "58 year old; male\nalbumin 4.1 g/dL\nC-reactive protein 3.1 mg/L"
    by_text = {s["text"]: s for s in body["statements"]}
    assert by_text["male"]["source"] == "model" and by_text["male"]["typed"] == "58 yo M"
    assert by_text["albumin 4.1 g/dL"]["source"] == "rules"
    again = _post({"text": body["read_as_text"]}).json()        # the rules alone, same patient
    assert again["patient"] == body["patient"]

    def timing_out(self, text):
        raise patient_extract.ExtractError("timeout", "slow")
    monkeypatch.setattr(patient_extract.OpenAIExtractor, "__call__", timing_out)
    body = _post({"text": _TEXT, "reader": "model"}).json()
    assert body["reader_used"] == "rules" and body["model_error"]["code"] == "timeout"
    assert body["ok"] is False                                   # '58 yo M' alone: no sex


def test_a_generic_question_with_a_patient_loaded_runs_without_the_patient(monkeypatch):
    """The full shared space aborts when it holds a caller and a program enumerates
    patient facts; a program that reads none never gets the patient's atoms."""
    from core.pln_runner import patient_stack
    state = _build(SMOKER)["state"]
    app_module, seen, _ = _chat_with(monkeypatch, "(infer &self Metformin CoronaryHeartDisease)", state)
    (task, kwargs), = seen["calls"]
    assert kwargs["extra_atoms"] is None and kwargs["kb_files"] == app_module._ALL_KB_PATHS
    _, seen, _ = _chat_with(monkeypatch, "(match &self (MeasuredZ $p CRP $z) ($p $z))", state)
    (task, kwargs), = seen["calls"]
    assert kwargs["kb_files"] == patient_stack(app_module._ALL_KB_PATHS)
    assert "(MeasuredZ Caller_Me CRP " in kwargs["extra_atoms"]



# ═══════════════════════════ what drives my abnormal labs is the diagnosis ════

def test_the_translator_is_told_that_the_causes_of_abnormal_labs_are_the_diagnosis(monkeypatch):
    """The tab's own suggested question: without this the translator copies the one
    diagnose few-shot's three hallmarks, which do not reach HbA1c or glucose (an empty
    answer for the patient whose witnesses those are)."""
    import json
    import re
    state = _build(SMOKER)["state"]
    _, seen, _ = _chat_with(monkeypatch, "(diagnose-patient &self Caller_Me)", state)
    assert "(diagnose-patient &self Caller_Me)" in seen["prompt"]          # the patient's own hint
    assert "17. CAUSES OF A PATIENT'S ABNORMAL LABS" in seen["prompt"]     # the static rule
    assert "drivers of BIOLOGICAL AGE, not of abnormal labs" in seen["prompt"]
    few = json.loads((PLN_CHAT / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8"))
    asks = [e for e in few if e["nl"] == "What is the likely driver of my abnormal labs?"]
    assert [e["metta_query"] for e in asks] == ["(diagnose-patient &self Caller_W58)"]
    # no patient example hands a typed hallmark list to the diagnosis any more
    assert not [e["metta_query"] for e in few
                if re.search(r"diagnose-patient &self \S+ \(", e["metta_query"])]
    assert patient_tab.SUGGESTED_QUESTIONS[-1] == "What is the likely driver of my abnormal labs?"


# ═══════════════ "nothing to work from": the tab and the chat say it ═════════════

#: diagnoses and labs with no curated edge: LinAge2 uses all of it, the shared layers none
NO_WITNESS = ("58 year old male\ncreatinine 1.8 mg/dL\nblood pressure 150/90\n"
              "total cholesterol 240 mg/dL\ndiagnoses: diabetes, hypertension, kidney disease")


def test_the_tab_says_when_nothing_typed_gives_the_shared_layers_a_witness():
    notes = _build(NO_WITNESS)["summary"]
    assert "Nothing you typed gives the knowledge base a witness" in notes
    assert "will come back empty" in notes and "still work" not in notes
    # LinAge2 itself is unaffected: the clock and its years are built as usual
    assert "LinAge2 biological age" in notes
    smoker = _build(SMOKER)["summary"]
    assert "Nothing you typed gives" not in smoker                    # HbA1c and CRP witness
    assert "The diagnosis can still work from your elevated labs" in smoker   # the no-GrimAge note keeps its promise


def test_the_chat_explains_an_empty_diagnosis_for_a_patient_with_nothing_to_work_from(monkeypatch):
    """The chat never shows the builder's notes, and () reads as 'no cause'."""
    state = _build(NO_WITNESS)["state"]
    _, _, history = _chat_with(monkeypatch, "(diagnose-patient &self Caller_Me)", state)
    answer = history[-1]["content"]
    assert "> **Note.** Caller_Me has no elevated value the knowledge base can use" in answer
    assert "the diagnosis returns ()" in answer
    _, _, history = _chat_with(monkeypatch, "(diagnose-patient &self Caller_Me)", _build(SMOKER)["state"])
    assert "has no elevated value" not in history[-1]["content"]


# ═══════════════ "what's my heart risk?" without a GrimAge value ═════════════════

PAIR = "(predict-risk-patient &self Caller_Me)\n(linage-hazard-patient &self Caller_Me)"


def test_the_translator_is_told_the_pair_to_emit_for_a_patient_with_no_grimage(monkeypatch):
    """The tab's default patient has no GrimAge line: "my heart risk" maps to the empty CHD
    model plus the LinAge2 hazard, to be called all-cause. With a GrimAge line it is the plain
    CHD risk and the hint is absent."""
    state = _build(SMOKER)["state"]
    _, seen, _ = _chat_with(monkeypatch, PAIR, state)
    assert "This patient has NO AgeAccelGrim value" in seen["prompt"]
    assert "  (predict-risk-patient &self Caller_Me)\n  (linage-hazard-patient &self Caller_Me)" in seen["prompt"]
    assert "ALL-CAUSE LinAge2 mortality hazard, not a heart risk" in seen["prompt"]
    assert "This model reads AgeAccelGrim ONLY" in seen["prompt"]                      # rule 12
    assert "A patient with no AgeAccelGrim value gets nothing from predict-risk-patient" in seen["prompt"]   # rule 16
    clock = _build(SMOKER + "\nGrimAge acceleration +3 years")["state"]
    assert clock["markers"]["AgeAccelGrim"]
    _, seen, _ = _chat_with(monkeypatch, "(predict-risk-patient &self Caller_Me)", clock)
    assert "NO AgeAccelGrim value" not in seen["prompt"]


def test_the_chat_labels_the_hazard_beside_an_empty_heart_risk_and_stays_silent_otherwise(monkeypatch):
    state = _build(SMOKER)["state"]
    _, _, history = _chat_with(monkeypatch, PAIR, state)
    answer = history[-1]["content"]
    assert "> **Note.** Caller_Me has no AgeAccelGrim value" in answer
    assert "ALL-CAUSE mortality multiplier" in answer and "never multiplied or added" in answer
    # a hazard-only question (battery C1) and a patient with a GrimAge value (battery C2) get no such note
    _, _, history = _chat_with(monkeypatch, "(linage-hazard-patient &self Caller_Me)", state)
    assert "no AgeAccelGrim value" not in history[-1]["content"]
    clock = _build(SMOKER + "\nGrimAge acceleration +3 years")["state"]
    _, _, history = _chat_with(monkeypatch, "(predict-risk-patient &self Caller_Me)", clock)
    assert "no AgeAccelGrim value" not in history[-1]["content"]


def test_the_tab_says_what_a_heart_risk_question_will_return_without_a_grimage_line():
    summary = _build(SMOKER)["summary"]
    assert "labelled all-cause mortality — it is not a heart risk" in summary


# ═══════════════ reported CHD: a first-event model ═══════════════════════════════

CHD_TEXT = (SMOKER + "\nGrimAge acceleration +3 years\n"
            "diagnoses: hypertension, coronary heart disease, heart attack, angina")


def test_the_tab_the_api_and_the_chat_carry_reported_chd_to_the_risk_answer(monkeypatch):
    import asyncio
    import httpx
    import api as api_module
    import core.executor as executor
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)

    out = _build(CHD_TEXT)
    state = out["state"]
    assert state["prevalent_chd"] == ["coronary heart disease", "angina", "heart attack"]
    assert "FIRST coronary event" in out["summary"]                      # the tab's build notes

    async def post(path, body):
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=300) as c:
            return await c.post(path, json=body)

    r = asyncio.run(post("/patients/from-text", {"text": CHD_TEXT}))      # no 500: PatientIn has the field
    assert r.status_code == 200 and r.json()["preview"]["prevalent_chd"] == state["prevalent_chd"]
    assert r.json()["patient"] == state

    _, seen, history = _chat_with(monkeypatch, "(predict-risk-patient &self Caller_Me)", state)
    assert "This patient reports coronary heart disease, angina, heart attack" in seen["prompt"]
    assert "> **Note.** Reported coronary heart disease, angina, heart attack" in history[-1]["content"]
    # without the history, or for a form that is not a heart-risk form, neither the hint nor the note
    plain = _build(SMOKER + "\nGrimAge acceleration +3 years\ndiagnoses: hypertension")["state"]
    _, seen, history = _chat_with(monkeypatch, "(predict-risk-patient &self Caller_Me)", plain)
    assert "This patient reports" not in seen["prompt"] and "Reported" not in history[-1]["content"]
    _, _, history = _chat_with(monkeypatch, "(diagnose-patient &self Caller_Me)", state)
    assert "FIRST coronary event" not in history[-1]["content"]
    # no GrimAge line: the CHD model has no input, so nothing to qualify
    no_clock = _build(SMOKER + "\ndiagnoses: hypertension, heart attack")["state"]
    _, seen, _ = _chat_with(monkeypatch, "(predict-risk-patient &self Caller_Me)", no_clock)
    assert "estimates a FIRST coronary event" not in seen["prompt"]       # no first-event hint (no risk model input)
    assert "reads that as one more observation to explain" in seen["prompt"]      # the diagnosis hint is another matter


# ═══════════════ "takes metformin": read, shown, carried to the supplement forms ══════

MET_TEXT = SMOKER + "\nmedications: metformin"


def test_the_tab_the_api_and_the_chat_carry_a_medication_to_the_supplement_forms_only(monkeypatch):
    import asyncio
    import httpx
    import api as api_module
    import core.executor as executor
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)

    reading = patient_tab.on_read(MET_TEXT)
    assert "medication: Metformin read as a current medication" in reading and "Not understood" not in reading
    out = _build(MET_TEXT)
    state = out["state"]
    assert state["medications"] == ["Metformin"]
    assert "Current medication recorded: Metformin" in out["summary"]                      # the build notes
    assert "(CurrentMedication Caller_Me Metformin)" in out["atoms"]["value"]             # the box shows it too ...
    assert "(CurrentMedication Caller_Me Metformin)" in Path(out["download"]["value"]).read_text()   # ... and the file
    assert "CurrentMedication" not in _build(SMOKER)["atoms"]["value"]
    assert _build(SMOKER + "\nstopped metformin")["state"].get("medications") is None

    async def post(path, body):
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=300) as c:
            return await c.post(path, json=body)

    r = asyncio.run(post("/patients/from-text", {"text": MET_TEXT}))                      # no 500: PatientIn has it
    assert r.status_code == 200 and r.json()["preview"]["medications"] == ["Metformin"] and r.json()["patient"] == state

    _, seen, _ = _chat_with(monkeypatch, "(recommend-supplements-patient &self Caller_Me)", state)
    assert "This patient currently takes Metformin" in seen["prompt"]
    assert "(CurrentMedication Caller_Me Metformin)" in seen["calls"][0][1]["extra_atoms"]
    _, seen, _ = _chat_with(monkeypatch, "(linage-hazard-patient &self Caller_Me)\n(recommend-supplements-patient &self Caller_Me)", state)
    (task, kwargs), = seen["calls"]
    lin, gen = kwargs["parts"]
    assert "CurrentMedication" not in lin["extra_atoms"] and "CurrentMedication" in gen["extra_atoms"]
    plain = _build(SMOKER)["state"]
    _, seen, _ = _chat_with(monkeypatch, "(recommend-supplements-patient &self Caller_Me)", plain)
    assert "currently takes" not in seen["prompt"] and "CurrentMedication" not in seen["calls"][0][1]["extra_atoms"]


def test_the_download_of_a_built_patient_passes_the_validator_it_is_meant_to_be_loaded_through():
    """The file starts with `;;` header lines; validate() used to tokenize them, so loading the tab's own
    download as extra_atoms of /metta/run answered 422."""
    import api as api_module
    from core.metta_validator import validate
    from core.patient_context import validation_text, with_injected
    out = _build(MET_TEXT)
    text = Path(out["download"]["value"]).read_text(encoding="utf-8")
    assert text.startswith(";; Caller_Me") and "(CurrentMedication Caller_Me Metformin)" in text
    registry, inventory = with_injected(api_module._runtime_registry(), api_module._runtime_inventory(), text)
    result = validate(validation_text(text, "!(supplement-for-patient &self Caller_Me Berberine)"), registry, inventory)
    assert result.valid, result.issues


# ═══════════════ the suggested questions depend on the patient (#8) ═══════════════

#: measured: for each patient, which button forms answered (scratch probe, 8 patients x 7 forms)
SMOKER_NORMAL_LABS = ("52 year old female, current smoker\nalbumin 4.5 g/dL\nHbA1c 5.2 %\n"
                      "CRP 0.6 mg/L\nfasting glucose 88 mg/dL\ncreatinine 0.8 mg/dL")
FORMER_CRP_ONLY = "60 year old female, former smoker\nalbumin 4.2 g/dL\nCRP 8 mg/L\ncreatinine 0.9 mg/dL"
HEALTHY = EXAMPLES["healthy 45-year-old woman"]
Q1, Q2, Q3, Q4, Q5, Q6 = patient_tab.SUGGESTED_QUESTIONS


def _offered(text: str) -> list[str]:
    state = _build(text)["state"]
    buttons = patient_tab.on_show_questions(state)[:-1]
    return [b["value"] for b in buttons if b["visible"]]


def test_a_smoker_with_an_elevated_lab_is_offered_every_question():
    assert _offered(SMOKER) == list(patient_tab.SUGGESTED_QUESTIONS)
    assert not patient_tab.on_show_questions(_build(SMOKER)["state"])[-1]["visible"]     # nothing to say


def test_a_patient_with_no_witness_and_no_smoking_is_offered_only_what_answers():
    for text in (HEALTHY, NO_WITNESS):
        offered = _offered(text)
        assert offered == [Q1, Q2, patient_tab.DRIVERS_ONLY_QUESTION], text
        note = patient_tab.on_show_questions(_build(text)["state"])[-1]
        assert note["visible"] and "quitting smoking" in note["value"]
        assert "the diagnosis and the supplement plan" in note["value"] and "what could I do" in note["value"]


def test_a_smoker_with_normal_labs_keeps_quitting_and_scenarios_but_loses_the_diagnosis():
    assert _offered(SMOKER_NORMAL_LABS) == [Q1, Q2, Q3, Q4, patient_tab.DRIVERS_ONLY_QUESTION]
    note = patient_tab.on_show_questions(_build(SMOKER_NORMAL_LABS)["state"])[-1]["value"]
    assert "the diagnosis and the supplement plan" in note and "quitting smoking" not in note
    assert "what could I do" not in note                      # a smoker's lever moves: the -8 y


def test_a_non_smoker_with_a_witness_loses_only_the_quitting_question():
    assert _offered(FORMER_CRP_ONLY) == [Q1, Q2, Q4, Q5, Q6]
    assert _offered(EXAMPLES["six labs only"]) == [Q1, Q2, Q4, Q5, Q6]    # a former smoker


def test_with_no_patient_only_the_always_answering_questions_are_offered():
    out = patient_tab.on_show_questions(None)
    assert [b["visible"] for b in out[:-1]] == [True, True, False, False, False, False]
    assert not out[-1]["visible"]


def test_each_gate_matches_what_the_forms_return():
    """The gate is a claim about the knowledge base, so check it against the knowledge base: the
    diagnosis and the plan are offered exactly when they answer."""
    from core.patient_context import build_caller_patient
    from core.patient_text import read_patient_text
    import test_patient_stack as ps                                   # the subprocess runner
    for text in (HEALTHY, SMOKER_NORMAL_LABS, FORMER_CRP_ONLY, NO_WITNESS, SMOKER, NO_WITNESS + "\ndiagnoses: heart attack"):
        payload, _ = read_patient_text(text).to_patient("Me")
        built = build_caller_patient(payload, ())
        shown = set(_offered(text))
        diag = ps._run("patient", "!(diagnose-patient &self Caller_Me)", built.shared_atoms)
        plan = ps._run("patient", "!(recommend-supplements-patient &self Caller_Me)", built.shared_atoms)
        assert diag["rc"] == 0 and plan["rc"] == 0, text
        diagnoses = any(a.strip() not in ("()", "") for a in diag["atoms"])
        recommends = any("(SuppRec" in a for a in plan["atoms"])
        assert (Q6 in shown) == diagnoses, text
        assert (Q5 in shown) == recommends, text


def test_the_tab_lays_out_with_the_gated_buttons_wired():
    import gradio as gr
    with gr.Blocks():
        state = gr.State(None)
        banner = gr.Markdown()
        box = gr.Textbox()
        with gr.Tabs() as tabs:
            with gr.Tab("My Patient"):
                patient_tab.build_tab(state, banner, box, tabs, "query")
            with gr.Tab("Query", id="query"):
                pass


# ═══════════ a lab or condition with no curated relation (#10) ═══════════════

def test_rule_16_names_exactly_the_markers_the_knowledge_base_has_a_cause_for():
    """The rule tells the translator which markers HAVE a curated cause; every other lab or condition gets the
    decomposition plus a plain 'no lever'. The list is read from the KB, so a bridge added to
    mechanistic_bridges.metta fails here until the rule says so."""
    import re
    from core.patient_builder import kb_effect_markers
    rule = (PLN_CHAT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    m = re.search(r"nor a marker the knowledge base has a curated cause for\s*\(([^)]*)\)", rule)
    assert m, "rule 16 lost its list of markers with a curated cause"
    assert set(re.findall(r"[A-Za-z0-9]+", m.group(1))) == set(kb_effect_markers())


def test_the_translator_is_told_what_to_emit_for_a_lab_with_no_lever(monkeypatch):
    state = _build(SMOKER)["state"]
    _, seen, _ = _chat_with(monkeypatch, "(linage-decomposition-patient &self Caller_Me)", state)
    prompt = " ".join(seen["prompt"].split())            # the rule wraps lines; the sentences are what is pinned
    assert "has NO curated cause or lever here" in prompt
    assert "Never make a lever token out of a lab, organ or diagnosis name" in prompt
    assert 'NOT what "normal" would remove' in prompt                            # never a counterfactual to normal
    assert "never a lever token made from its name (rule 16)" in prompt          # the per-patient hint
    assert "(linage-counterfactual-patient &self <Patient> LowSerumAlbumin)" in prompt   # the albumin deficit IS a lever
    assert "A what-if about a lever" in prompt                                    # no GrimAge: the LinAge2 form


def test_the_kidney_few_shot_is_the_decomposition_and_invents_no_lever():
    import json
    few = json.loads((PLN_CHAT / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8"))
    (ex,) = [e for e in few if e["nl"] == "What should I do about my kidney function?"]
    assert ex["metta_query"] == "(linage-decomposition-patient &self Caller_W58)"
    assert "no curated cause or lever" in ex["explanation"] and ex["warnings"]
    assert "counterfactual" not in ex["metta_query"]


# ═══════════ a reported CHD is an observation for the diagnosis only (#11) ═══════════

def test_the_atoms_box_and_the_download_carry_the_reported_condition_and_the_tab_words_the_notes():
    out = _build(SMOKER + "\ndiagnoses: heart attack")
    assert "(PatientCondition Caller_Me CoronaryHeartDisease)" in out["atoms"]["value"]
    assert "(PatientCondition Caller_Me CoronaryHeartDisease)" in Path(out["download"]["value"]).read_text(encoding="utf-8")
    assert "Reported heart attack is read by the diagnosis as one observation" in out["summary"]
    assert "(PatientCondition" not in _build(SMOKER)["atoms"]["value"]
    # a patient whose only finding is the report: the tab words the no-witness note for the plan, not the diagnosis
    only = _build("58 year old male\ncreatinine 1.8 mg/dL\nblood pressure 150/90\ndiagnoses: heart attack")["summary"]
    assert "The diagnosis answers from your reported heart disease alone" in only and "will come back empty" not in only
    assert "Nothing you typed gives the supplement plan or the ranking a witness" in only


def test_the_translator_is_told_a_reported_chd_is_one_more_observation_not_a_measurement(monkeypatch):
    state = _build(SMOKER + "\ndiagnoses: heart attack")["state"]
    _, seen, history = _chat_with(monkeypatch, "(diagnose-patient &self Caller_Me)", state)
    prompt = " ".join(seen["prompt"].split())
    assert "reads that as one more observation to explain" in prompt and "not a measured value" in prompt
    assert "do not read it" in prompt                                    # the plan and the ranking
    assert "> **Note.** Reported heart attack is read by the diagnosis as one observation" in history[-1]["content"]
    plain = _build(SMOKER)["state"]
    _, seen, history = _chat_with(monkeypatch, "(diagnose-patient &self Caller_Me)", plain)
    assert "one more observation to explain" not in seen["prompt"] and "Reported" not in history[-1]["content"]


def test_a_reported_heart_disease_keeps_the_diagnosis_button_because_the_diagnosis_answers_from_it():
    """Item #11: with no elevated witness the diagnosis still explains the reported heart disease; the plan does not."""
    text = NO_WITNESS + "\ndiagnoses: heart attack"
    assert _offered(text) == [Q1, Q2, patient_tab.DRIVERS_ONLY_QUESTION, Q6]
    note = patient_tab.on_show_questions(_build(text)["state"])[-1]["value"]
    assert "the supplement plan" in note and "the diagnosis and the supplement plan" not in note


# ═══════ the gate follows the engine's credit rule (review round 3) ═══════

#: patients the review found the first gate wrong for, plus controls: (text, why it matters)
GATE_PATIENTS = {
    "tg_only": "58 year old male, never smoked\nalbumin 4.8 g/dL\nRDW 12.3 %\nCRP 0.4 mg/L\nHbA1c 5.0 %\nfasting triglycerides 220 mg/dL",
    "male_crp_only": "58 year old male, never smoked\nalbumin 4.8 g/dL\nRDW 12.3 %\nHbA1c 5.0 %\nCRP 8 mg/L",
    "male_glucose_only": "58 year old male, never smoked\nalbumin 4.8 g/dL\nRDW 12.3 %\nCRP 0.4 mg/L\nHbA1c 5.0 %\nfasting glucose 126 mg/dL",
    "smoker_low_cotinine": "52 year old male, current smoker\nalbumin 4.8 g/dL\nRDW 12.3 %\nCRP 0.4 mg/L\nHbA1c 5.0 %\ncotinine 5 ng/mL",
    "witnessless_no_drivers": "45 year old female, never smoked\nalbumin 4.8 g/dL\nRDW 12.3 %\nCRP 0.4 mg/L\nHbA1c 5.0 %",
    "smoker_example": SMOKER,
    "six_labs": EXAMPLES["six labs only"],
    "healthy": HEALTHY,
    "female_crp": "58 year old female, never smoked\nalbumin 4.5 g/dL\nRDW 12.3 %\nCRP 12 mg/L",
}


def _linage2_answers(text: str):
    """What the three LinAge2 button forms return for this patient, from the real rules (in-process)."""
    import re
    import test_linage2 as tl
    from core.patient_context import build_caller_patient
    from core.patient_text import read_patient_text
    built = build_caller_patient(read_patient_text(text).to_patient("Me")[0], ())
    m = tl._space(built.atoms)
    scenarios = tl._one(m, "!(linage-scenarios-patient &self Caller_Me)")
    levers = dict(re.findall(r"\(LinAgeCounterfactual Caller_Me (\w+) \(expected-delta-years ([-\d.eE]+)\)", scenarios))
    drivers = tl._one(m, "!(linage-drivers-patient &self Caller_Me)")
    return built, {k: float(v) for k, v in levers.items()}, drivers


@pytest.mark.parametrize("name", sorted(GATE_PATIENTS))
def test_a_linage2_button_is_offered_exactly_when_its_form_moves_or_answers(name):
    text = GATE_PATIENTS[name]
    built, levers, drivers = _linage2_answers(text)
    buttons = dict(patient_tab.question_buttons(built))
    quit_moves = abs(levers["SmokingCessation"]) > 1e-9
    anything_moves = any(abs(v) > 1e-9 for v in levers.values())
    assert buttons[Q3] is quit_moves, (name, levers)
    assert buttons[Q4] is anything_moves, (name, levers)
    drivers_answer = drivers.strip() not in ("()", "")
    if built.witnesses:
        assert buttons[Q5] is True                                   # the plan half answers; the drivers half may not
    else:
        assert buttons[patient_tab.DRIVERS_ONLY_QUESTION] is drivers_answer, (name, drivers[:80])


def test_the_drivers_threshold_the_gate_uses_is_the_one_in_the_rules():
    import re
    text = (REPO / "pln_linage2.metta").read_text(encoding="utf-8")
    assert float(re.search(r"\(linage-driver-threshold-years\) ([\d.]+)\)", text).group(1)) == patient_tab.DRIVER_THRESHOLD_YEARS
