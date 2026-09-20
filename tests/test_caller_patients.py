"""A caller's own patient — the evaluation's first recommendation.

"A 45-year-old woman with LDL 130 and CRP 4 got 'I can't compute a personalized
risk from the current KB'. The whole personalized stack works only for
Patient001 and Patient002, so an app user's biomarkers cannot be scored today."

The inference never needed changing; the sanitisation did. Three failures were
reproduced against the pre-existing `extra_atoms` path and each has a test here:
an id that closes a parenthesis redefines a calibration knob and corrupts OTHER
patients' answers; a colliding id unions two patients into 512 inconsistent
results; and a rule definition in `extra_atoms` poisons the whole request.

Run from the repository root:
    pytest tests/test_caller_patients.py -q
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import api as api_module  # noqa: E402
import core.executor as executor_module  # noqa: E402
from core.patient_builder import PatientSpecError, build_patient  # noqa: E402


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)


def _request(method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(send())


THE_REPORTED_PATIENT = {
    "id": "W45",
    "age": 45,
    "sex": "Female",
    "smoking": "NeverSmoker",
    "markers": {
        "AgeAccelGrim": {"value": 2.1, "unit": "years"},
        "CRP": {"value": 4.0, "unit": "mg/L"},
        "DNAmGDF15": 1.3,
    },
}


# ── sanitisation: the three reproduced failures ──────────────────────────────

def test_a_patient_id_cannot_inject_metta():
    """`Evil) (= (grimage-weight $m) 9.9) (PatientAge Zzz 10` redefined a knob."""
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({"id": "Evil) (= (grimage-weight $m) 9.9) (PatientAge Zzz 10"})
    assert excinfo.value.code == "invalid_patient_id"

    for hostile in ("a b", "x)(y", "(", "$var", "a;b", "a\nb", ""):
        built_or_error = None
        try:
            built_or_error = build_patient({"id": hostile})
        except PatientSpecError:
            continue
        assert "(" not in built_or_error.patient_id
        assert ")" not in built_or_error.patient_id


def test_every_generated_atom_is_well_formed():
    built = build_patient(THE_REPORTED_PATIENT)
    for line in built.atoms.splitlines():
        assert line.startswith("(") and line.endswith(")")
        assert line.count("(") == line.count(")")
    assert "(= " not in built.atoms          # never a definition


def test_a_colliding_id_is_refused_rather_than_unioned():
    """A second Patient001 does not replace the first; it makes both wrong."""
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({"id": "Patient001"}, existing_ids={"Caller_Patient001"})
    assert excinfo.value.code == "patient_id_collision"


def test_ids_are_namespaced_so_a_collision_is_impossible_by_construction():
    assert build_patient({"id": "Patient001"}).patient_id == "Caller_Patient001"
    assert build_patient({}).patient_id == "Caller_Patient"


def test_a_rule_definition_in_extra_atoms_is_rejected():
    """It does not shadow the KB's rule — the engine keeps both."""
    response = _request(
        "POST", "/metta/run",
        json={
            "metta_query": "!(predict-risk-patient &self Patient001)",
            "extra_atoms": "(= (baseline-risk-chd $a $s) 0.999)",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "definition_in_extra_atoms"


def test_a_deliberate_redefinition_is_still_possible_when_asked_for(monkeypatch):
    from core.pln_runner import PLNAtomResult, PLNRunResult

    monkeypatch.setattr(api_module, "run_query", Mock(return_value=PLNRunResult(
        status="ok", results=[PLNAtomResult("(ok)")], mode="runtime")))
    response = _request(
        "POST", "/metta/run",
        json={
            "metta_query": "!(baseline-risk-chd 58 Male)",
            "extra_atoms": "(= (baseline-risk-chd $a $s) 0.999)",
            "allow_definitions": True,
        },
    )
    assert response.status_code == 200


# ── validation says what is wrong, and what would be right ───────────────────

def test_an_unsupported_marker_names_the_supported_ones():
    """LDL 130 is the other half of the reported question."""
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({"markers": {"LDL": {"value": 130}}})
    assert excinfo.value.code == "unknown_marker"
    assert "CRP" in excinfo.value.message


def test_an_invented_sex_branch_is_refused():
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({"sex": "Other"})
    assert excinfo.value.code == "invalid_sex"
    assert "inventing a third branch" in excinfo.value.message


def test_an_implausible_z_is_caught_as_a_unit_mistake():
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({"markers": {"CRP": 99.0}})
    assert excinfo.value.code == "implausible_z"


def test_a_marker_with_no_reference_refuses_a_raw_value():
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({"markers": {"DNAmPAI1": {"value": 1.2}}})
    assert excinfo.value.code == "raw_value_unsupported"


def test_a_patient_without_a_clock_is_warned_not_silently_empty():
    built = build_patient({"age": 50, "sex": "Male", "markers": {"CRP": 1.4}})
    assert not built.can_predict_risk
    assert any("AgeAccelGrim" in w for w in built.warnings)


# ── z-scoring is explicit about what it derived ──────────────────────────────

def test_a_raw_value_reports_the_formula_that_standardised_it():
    built = build_patient(THE_REPORTED_PATIENT)
    by_name = {m.name: m for m in built.markers}

    clock = by_name["AgeAccelGrim"]
    assert clock.derived and clock.z == pytest.approx(2.1 / 4.2)
    assert "grimaccel-sd-to-years" in clock.formula

    crp = by_name["CRP"]
    assert crp.derived and "ln(" in crp.formula
    assert crp.raw_value == 4.0 and crp.unit == "mg/L"

    supplied = by_name["DNAmGDF15"]
    assert not supplied.derived and supplied.z == 1.3


def test_the_years_conversion_uses_the_kbs_own_knob():
    risk = (REPO / "pln_risk_prediction.metta").read_text(encoding="utf-8")
    sd_to_years, elevated = api_module._patient_knobs()
    assert f"(= (grimaccel-sd-to-years) {sd_to_years})" in risk
    profile = (REPO / "patient_profile.metta").read_text(encoding="utf-8")
    assert f"(= (elevated-z-threshold) {elevated})" in profile


def test_marker_status_uses_the_kbs_threshold():
    built = build_patient({"markers": {"CRP": 1.4, "DNAmLeptin": -1.5, "DNAmPAI1": 0.2}})
    status = {m.name: m.status for m in built.markers}
    assert status == {"CRP": "Elevated", "DNAmLeptin": "Low", "DNAmPAI1": "Normal"}


# ── it actually answers the question that was refused ────────────────────────

@pytest.mark.slow
def test_the_reported_question_now_gets_a_real_answer():
    pytest.importorskip("hyperon")
    from core.pln_runner import run_query

    built = build_patient(THE_REPORTED_PATIENT)
    kb = api_module._runtime_kb_paths()

    risk = run_query(
        f"!(predict-risk-patient &self {built.patient_id})",
        kb_files=kb, extra_atoms=built.atoms,
    )
    assert risk.status == "ok"
    atom = risk.results[0].atom
    assert atom.startswith(f"(RiskPrediction {built.patient_id}")
    assert "(baseline 0.02)" in atom          # the 45yo-female band

    supplements = run_query(
        f"!(recommend-supplements-patient &self {built.patient_id})",
        kb_files=kb, extra_atoms=built.atoms,
    )
    assert supplements.status == "ok"
    assert "Omega3" in supplements.results[0].atom


@pytest.mark.slow
def test_a_caller_patient_leaves_the_builtin_patients_untouched():
    pytest.importorskip("hyperon")
    from core.pln_runner import run_query

    built = build_patient(THE_REPORTED_PATIENT)
    kb = api_module._runtime_kb_paths()
    baseline = run_query("!(predict-risk-patient &self Patient001)", kb_files=kb)
    with_caller = run_query(
        "!(predict-risk-patient &self Patient001)", kb_files=kb, extra_atoms=built.atoms
    )
    assert [r.atom for r in baseline.results] == [r.atom for r in with_caller.results]


# ── over HTTP ────────────────────────────────────────────────────────────────

def test_preview_shows_the_atoms_without_running_anything(monkeypatch):
    run_query = Mock()
    monkeypatch.setattr(api_module, "run_query", run_query)

    body = _request("POST", "/patients/preview", json=THE_REPORTED_PATIENT).json()

    assert body["patient_id"] == "Caller_W45"
    assert "(PatientAge Caller_W45 45)" in body["atoms"]
    assert body["can_predict_risk"] is True
    assert {m["marker"] for m in body["markers"]} == {"AgeAccelGrim", "CRP", "DNAmGDF15"}
    run_query.assert_not_called()


def test_preview_rejects_a_bad_patient_with_a_code():
    response = _request("POST", "/patients/preview", json={"markers": {"LDL": 130}})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_marker"


def test_the_marker_catalog_is_honest_about_its_reference_values():
    body = _request("GET", "/patients/markers").json()
    by_name = {m["marker"]: m for m in body["markers"]}
    assert by_name["CRP"]["provisional"] is True
    assert "coarse prior" in by_name["CRP"]["reference_source"]
    assert by_name["DNAmPAI1"]["accepts_raw_value"] is False
    assert "CURATED PRIORS" in body["raw_value_note"]
    assert body["elevated_threshold"] == 1.0


def test_query_injects_the_patient_and_tells_the_translator_about_it(monkeypatch):
    from core.llm_translator import TranslationResult
    from core.pln_runner import PLNAtomResult, PLNRunResult

    captured = {}

    def fake_translate(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return TranslationResult(
            metta_query="!(predict-risk-patient &self Caller_W45)",
            explanation="", intent="inference", requires_pln_inference=True,
            confidence_filter=0.0,
        )

    run_query = Mock(return_value=PLNRunResult(
        status="ok", results=[PLNAtomResult("(RiskPrediction Caller_W45)")], mode="runtime"))
    monkeypatch.setattr(api_module, "translate", fake_translate)
    monkeypatch.setattr(api_module, "run_query", run_query)
    monkeypatch.setattr(api_module, "log_turn", Mock())

    body = _request(
        "POST", "/query",
        json={"message": "what is my 10-year CHD risk?", "patient": THE_REPORTED_PATIENT},
    ).json()

    assert "Caller_W45" in captured["system_prompt"]
    assert "(MeasuredZ Caller_W45 CRP" in captured["system_prompt"]
    assert run_query.call_args.kwargs["extra_atoms"].startswith("(InstanceOf Caller_W45")
    assert body["patient"]["patient_id"] == "Caller_W45"


def test_a_query_about_an_unknown_patient_is_flagged_as_unpersonalized(monkeypatch):
    from core.llm_translator import TranslationResult
    from core.pln_runner import PLNAtomResult, PLNRunResult

    monkeypatch.setattr(api_module, "translate", lambda **kw: TranslationResult(
        metta_query="!(rank-interventions-for-patient &self Patient404 (Fisetin) CoronaryHeartDisease)",
        explanation="", intent="inference", requires_pln_inference=True, confidence_filter=0.0,
    ))
    monkeypatch.setattr(api_module, "run_query", Mock(return_value=PLNRunResult(
        status="ok", results=[PLNAtomResult("(scored Fisetin 0.03)")], mode="runtime")))
    monkeypatch.setattr(api_module, "log_turn", Mock())

    body = _request("POST", "/query", json={"message": "rank fisetin for Patient404"}).json()
    assert any("Patient404" in w and "unpersonalized" in w for w in body["warnings"])


def test_metta_run_accepts_a_patient_too(monkeypatch):
    from core.pln_runner import PLNAtomResult, PLNRunResult

    run_query = Mock(return_value=PLNRunResult(
        status="ok", results=[PLNAtomResult("(RiskPrediction Caller_W45)")], mode="runtime"))
    monkeypatch.setattr(api_module, "run_query", run_query)

    body = _request(
        "POST", "/metta/run",
        json={
            "metta_query": "!(predict-risk-patient &self Caller_W45)",
            "patient": THE_REPORTED_PATIENT,
        },
    ).json()

    assert body["patient_id"] == "Caller_W45"
    assert "(MeasuredZ Caller_W45" in run_query.call_args.kwargs["extra_atoms"]
