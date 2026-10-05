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
    summary = _build("60 year old female\nalbumin 4.0 g/dL")["summary"]
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
