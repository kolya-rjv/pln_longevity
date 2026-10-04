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
