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
    # Routed as production routes a question that names a patient: the patient stack
    # (core.pln_runner.patient_stack). In the full shared space both forms abort the
    # process for this patient — tests/test_patient_stack.py measures why.
    kb = api_module._generic_kb(f"(predict-risk-patient &self {built.patient_id})")

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


# ── a derived z is not an adjusted z, and the caller is told ─────────────────

def test_a_server_derived_z_is_declared_unadjusted():
    """GET /patients/markers publishes z as AGE- AND SEX-ADJUSTED. Derived ones aren't.

    `Reference.to_z` standardises against a single POOLED mean and sd — this
    repository has no stratified reference table — and the atom it produces,
    `(MeasuredZ <patient> <marker> <z>)`, is indistinguishable from one built
    from a properly adjusted z the caller sent. The provenance cannot go into
    the space without inventing an adjustment that was never made, so it goes
    into the response: `derived: true`, the formula, and this warning.
    """
    from core.patient_builder import build_patient

    derived = build_patient({
        "age": 45, "sex": "Female",
        "markers": {"AgeAccelGrim": 1.2, "CRP": {"value": 4.0, "unit": "mg/L"}},
    })
    warning = next(
        (w for w in derived.warnings if "NOT age- and sex-adjusted" in w), None
    )
    assert warning, derived.warnings
    assert "CRP" in warning
    # The marker that WAS sent as a z must not be named.
    assert "AgeAccelGrim" not in warning
    assert [m.derived for m in derived.markers if m.name == "CRP"] == [True]

    # A patient whose every marker arrived as a z gets no such warning.
    sent_as_z = build_patient({
        "age": 45, "sex": "Female", "markers": {"AgeAccelGrim": 1.2, "CRP": 0.9},
    })
    assert not any("NOT age- and sex-adjusted" in w for w in sent_as_z.warnings)


def test_the_published_z_convention_says_which_z_it_describes():
    """The convention text is what an integrator reads before sending values."""
    response = _request("GET", "/patients/markers")
    convention = response.json()["z_convention"]
    assert "AGE- AND SEX-ADJUSTED" in convention
    # …and, since the server also PRODUCES z-scores, which of the two it means.
    assert "DERIVES" in convention and "NOT" in convention


# ═══════════════ "nothing to work from" is said, and only when it is true ════════
#
# The shared layers (diagnosis, supplement plan, intervention ranking) read ONE thing of a
# patient: a marker that is ELEVATED and has a curated Effect edge into it. A patient with
# diabetes, hypertension, CKD, an albumin, a creatinine and a blood pressure — and none of
# those — was told "Diagnosis, supplement ranking and intervention ranking still work";
# the diagnosis returned (), every supplement tier was empty and the ranking was the
# population's, byte for byte (docs/kb_quick_wins/REPORT.md #4).

NOTHING_USABLE = {"age": 58, "sex": "Male", "markers": {"AgeAccelGrim": 2.0, "DNAmADM": 1.5, "CRP": 0.2}}


def _notes(built, prefix):
    return [w for w in built.warnings if w.startswith(prefix)]


def test_the_markers_the_kb_has_edges_into_are_read_off_the_kb():
    """A new bridge widens this with no edit — and this test then asks whether the default
    cause list of diagnose-patient (patient_profile.metta) still reaches every one of them
    (tests/test_patient_stack.py checks that)."""
    from core.patient_builder import kb_effect_markers
    assert kb_effect_markers() == frozenset(
        {"CRP", "DNAmGDF15", "DNAmPACKYRS", "DNAmPAI1", "FastingGlucose", "HbA1c", "LowSerumAlbumin", "RDW",
         "Triglycerides"})


def test_a_patient_with_nothing_the_kb_can_use_is_told_the_shared_layers_have_nothing_to_work_from():
    from core.patient_builder import NO_WITNESS_PREFIX
    built = build_patient(NOTHING_USABLE)
    assert built.witnesses == []
    (note,) = _notes(built, NO_WITNESS_PREFIX)
    assert "diagnosis returns ()" in note and "population ranking" in note and "not 'no cause'" in note
    assert not any("still work" in w for w in built.warnings)       # the old promise is gone


def test_an_elevated_marker_with_an_edge_keeps_the_promise_and_gets_no_such_note():
    from core.patient_builder import NO_WITNESS_PREFIX, STILL_WORK
    built = build_patient({"age": 58, "sex": "Male", "markers": {"HbA1c": 1.8, "AgeAccelGrim": 0.1}})
    assert built.witnesses == ["HbA1c"] and not _notes(built, NO_WITNESS_PREFIX)
    no_clock = build_patient({"age": 58, "sex": "Male", "markers": {"HbA1c": 1.8}})
    (grim,) = _notes(no_clock, "No AgeAccelGrim measurement")
    assert grim.endswith(STILL_WORK)
    # with nothing to work from the clock note stops promising what is not so
    (grim,) = _notes(build_patient({"age": 58, "sex": "Male", "markers": {"CRP": 0.2}}),
                     "No AgeAccelGrim measurement")
    assert "can still work" not in grim and "still work" not in grim


@pytest.mark.parametrize("z, witnessed", [(1.0, False), (1.01, True), (-2.0, False), (0.0, False)])
def test_only_an_elevated_value_is_a_witness_and_the_boundary_is_strict(z, witnessed):
    built = build_patient({"age": 58, "sex": "Male", "markers": {"CRP": z}})
    assert bool(built.witnesses) is witnessed
    assert bool(_notes(built, "No elevated marker the knowledge base can use")) is not witnessed


def test_the_form_warning_names_only_forms_that_name_this_patient():
    from core.patient_context import patient_form_warnings
    built = build_patient(NOTHING_USABLE)
    pid = built.patient_id
    both = patient_form_warnings(
        f"(diagnose-patient &self {pid})\n!(rank-interventions-for-patient &self {pid} (Metformin) "
        f"CoronaryHeartDisease)", built)
    assert len(both) == 1
    assert "the diagnosis returns ()" in both[0] and "the ranking is the population ranking" in both[0]
    assert "recommend-supplements" not in both[0] and "every supplement tier" not in both[0]
    assert "every supplement tier is empty" in patient_form_warnings(
        f"(recommend-supplements-patient &self {pid})", built)[0]
    assert "returns nothing" in patient_form_warnings(f"(supplement-for-patient &self {pid} Berberine)", built)[0]
    # not for another patient, not for a form that reads the clock or LinAge2, not without a patient
    assert patient_form_warnings("(diagnose-patient &self Patient001)", built) == []
    assert patient_form_warnings(f"(predict-risk-patient &self {pid})", built) == []
    assert patient_form_warnings(f"(linage-drivers-patient &self {pid})", built) == []
    assert patient_form_warnings(f"(diagnose-patient &self {pid})", None) == []
    # and not when the patient has something to work from
    witnessed = build_patient({"age": 58, "sex": "Male", "markers": {"HbA1c": 1.8}})
    assert patient_form_warnings(f"(diagnose-patient &self {witnessed.patient_id})", witnessed) == []


def _translation(query):
    from core.llm_translator import TranslationResult
    return TranslationResult(metta_query=query, explanation="", intent="inference",
                             requires_pln_inference=True, confidence_filter=0.0)


def test_query_and_metta_run_attach_the_same_form_warning(monkeypatch):
    from core.pln_runner import PLNRunResult
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(diagnose-patient &self Caller_N)"))
    monkeypatch.setattr(api_module, "run_query", Mock(return_value=PLNRunResult(status="empty", mode="runtime")))
    monkeypatch.setattr(api_module, "log_turn", Mock())
    nothing = {**NOTHING_USABLE, "id": "N"}
    body = _request("POST", "/query", json={"message": "what drives my labs?", "patient": nothing}).json()
    ours = [w for w in body["warnings"] if w.startswith("Caller_N has no elevated value")]
    assert len(ours) == 1 and "the diagnosis returns ()" in ours[0]
    run = _request("POST", "/metta/run", json={"metta_query": "(diagnose-patient &self Caller_N)",
                                               "patient": nothing}).json()
    assert [w for w in run["warnings"] if w.startswith("Caller_N has no elevated value")] == ours
    # a patient with a witness gets neither
    witnessed = {"id": "N", "age": 58, "sex": "Male", "markers": {"HbA1c": 1.8}}
    body = _request("POST", "/query", json={"message": "q", "patient": witnessed}).json()
    assert not [w for w in body["warnings"] if "has no elevated value" in w]


# ═══════════════ "what's my heart risk?" without a GrimAge value ═════════════════
#
# The 10-year CHD model reads AgeAccelGrim and nothing else, so a patient with a LinAge2 result
# and no GrimAge value gets (predict-risk-patient …) = nothing — and in a mixed program the CHD
# part silently dropped out, leaving only the all-cause LinAge2 hazard under a heart question.
# The note says there is no heart-specific risk, labels the hazard as all-cause mortality, and
# the two clocks are never combined (docs/risk_prediction.md §3).

NO_GRIM = {"age": 58, "sex": "Male", "markers": {"HbA1c": 1.8}}
WITH_GRIM = {"age": 58, "sex": "Male", "markers": {"HbA1c": 1.8, "AgeAccelGrim": {"value": 4.5, "unit": "years"}}}


def _with_linage2(payload):
    """The payload with the LinAge2 fixture's block (a real LinAgeDelta), as the tab's patient has."""
    import json
    fixture = json.loads((REPO / "tests" / "fixtures" / "linage2_response.json").read_text(encoding="utf-8"))
    return {**payload, "age": 58, "sex": "Male", "linage2": fixture}


def test_a_heart_risk_form_for_a_patient_with_no_grimage_says_there_is_no_heart_risk():
    from core.patient_context import patient_form_warnings
    built = build_patient(_with_linage2(NO_GRIM))
    pid = built.patient_id
    assert not built.has_grimage and not built.can_predict_risk and built.linage2 is not None
    (alone,) = patient_form_warnings(f"(predict-risk-patient &self {pid})", built)
    assert "no AgeAccelGrim value" in alone and "predict-risk-patient returns nothing" in alone
    assert "ALL-CAUSE" not in alone                       # no hazard in the program: nothing to label
    (pair,) = patient_form_warnings(f"(predict-risk-patient &self {pid})\n(linage-hazard-patient &self {pid})", built)
    assert "ALL-CAUSE mortality multiplier" in pair and "not a heart risk" in pair
    assert "never multiplied or added to a GrimAge result" in pair
    for form in ("risk-decomposition-patient", "project-risk-patient"):
        assert patient_form_warnings(f"({form} &self {pid} Metformin)", built)
    # the battery's C2 shapes must stay silent: a patient WITH a GrimAge value asking for the CHD
    # risk, a hazard-only question, another patient, a form that is not a heart-risk form
    with_clock = build_patient(WITH_GRIM)
    assert with_clock.has_grimage and with_clock.can_predict_risk
    assert patient_form_warnings(f"(predict-risk-patient &self {with_clock.patient_id})", with_clock) == []
    assert patient_form_warnings(f"(linage-hazard-patient &self {pid})", built) == []
    assert patient_form_warnings("(predict-risk-patient &self Patient001)", built) == []
    assert patient_form_warnings(f"(decompose-grimage &self {pid})", built) == []


# ═══════════════ the 10-year CHD risk is a first-event model ═════════════════════
#
# Lu 2019's hazard ratio is for INCIDENT CHD and NHANES's own CHD items are prevalence, "usable only
# as an exclusion from the at-risk set" (nhanes_baseline.metta). A person who reports CHD, a heart
# attack or angina is outside that set, yet gets the same number: the history is not an input.

def test_reported_chd_qualifies_the_risk_only_where_the_risk_model_answers():
    from core.patient_context import patient_form_warnings
    built = build_patient({**WITH_GRIM, "prevalent_chd": ["heart attack", "coronary heart disease", "heart attack"]})
    assert built.prevalent_chd == ["coronary heart disease", "heart attack"]       # canonical order, once each
    (note,) = [w for w in built.warnings if "FIRST coronary event" in w]
    assert "FIRST coronary event" in note and "same number with or without that history" in note
    # not an atom for the RISK model, the LinAge2 space or any marker: only one PatientCondition line, in the SHARED
    # space, which `diagnose-patient` alone reads (item #11); everything else is the patient without it
    base = build_patient(WITH_GRIM)
    assert built.atoms == base.atoms
    assert built.shared_atoms == base.shared_atoms + f"\n(PatientCondition {built.patient_id} CoronaryHeartDisease)"
    pid = built.patient_id
    assert patient_form_warnings(f"(predict-risk-patient &self {pid})", built) == [note]
    assert patient_form_warnings(f"(project-risk-patient &self {pid} Metformin)", built) == [note]
    # a diagnosis carries the OTHER note (item #11): the report is one observation it explains, not a first event
    (observed,) = [w for w in patient_form_warnings(f"(diagnose-patient &self {pid})", built)
                   if "read by the diagnosis as one observation" in w]
    assert "FIRST coronary event" not in observed and "prevalence item" in observed
    assert patient_form_warnings("(predict-risk-patient &self Patient001)", built) == []
    # where the model returns nothing anyway there is nothing to qualify
    no_clock = build_patient({**NO_GRIM, "prevalent_chd": ["angina"]})
    assert not [w for w in no_clock.warnings if "FIRST coronary event" in w]
    assert not any("first" in w.lower() and "coronary" in w for w in
                   patient_form_warnings(f"(predict-risk-patient &self {no_clock.patient_id})", no_clock))
    assert build_patient({**WITH_GRIM, "prevalent_chd": ["angina", "angina"]}).prevalent_chd == ["angina"]


@pytest.mark.parametrize("bad", [["heart failure"], ["x) (= (grimage-weight $m) 9.9) (y"], "angina", ["angina"] * 4, [5]])
def test_prevalent_chd_is_checked_against_an_allow_list_never_interpolated(bad):
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({**WITH_GRIM, "prevalent_chd": bad})
    assert excinfo.value.code == "invalid_prevalent_chd"


def test_the_api_takes_prevalent_chd_and_the_response_says_so(monkeypatch):
    from core.pln_runner import PLNRunResult
    patient = {**WITH_GRIM, "id": "H", "prevalent_chd": ["heart attack"]}
    body = _request("POST", "/patients/preview", json=patient).json()
    assert body["prevalent_chd"] == ["heart attack"]
    assert any(w.startswith("Reported heart attack") for w in body["warnings"])
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(predict-risk-patient &self Caller_H)"))
    monkeypatch.setattr(api_module, "run_query", Mock(return_value=PLNRunResult(status="empty", mode="runtime")))
    monkeypatch.setattr(api_module, "log_turn", Mock())
    answer = _request("POST", "/query", json={"message": "my heart risk?", "patient": patient}).json()
    assert answer["patient"]["prevalent_chd"] == ["heart attack"]
    assert [w for w in answer["warnings"] if w.startswith("Reported heart attack")]
    run = _request("POST", "/metta/run", json={"metta_query": "(predict-risk-patient &self Caller_H)",
                                               "patient": patient}).json()
    assert [w for w in run["warnings"] if w.startswith("Reported heart attack")]
    # an unknown key is still refused: the field is declared, not `extra="allow"`
    assert _request("POST", "/patients/preview", json={**WITH_GRIM, "prevalent_cvd": ["x"]}).status_code == 422


# ═══════════════ a current medication: shared atoms only, from an allow-list ═════════
#
# `(CurrentMedication <id> <drug>)` is what the supplement forms read to flag an interaction. It
# goes ONLY to the shared (patient-stack) space: the LinAge2 space has no head-symbol room for a new
# head and nothing there reads it. The drug is checked against the drugs the KB holds an Interaction
# fact for — never interpolated — and it changes no ranking and no LinAge2 number.

WITH_METFORMIN = {**NO_GRIM, "markers": {"HbA1c": 1.8}, "medications": ["Metformin"]}


def test_a_medication_is_a_shared_atom_never_a_linage2_or_preview_atom():
    built = build_patient(WITH_METFORMIN)
    line = f"(CurrentMedication {built.patient_id} Metformin)"
    assert built.medications == ["Metformin"] and line in built.shared_atoms.splitlines()
    assert line not in built.atoms                    # `atoms` feeds the LinAge2 space and the preview
    assert built.atoms == build_patient({**WITH_METFORMIN, "medications": []}).atoms
    (note,) = [w for w in built.warnings if w.startswith("Current medication recorded")]
    assert "changes no ranking and no LinAge2 number" in note and "not as something already taken" in note
    assert not [w for w in build_patient(NO_GRIM).warnings if w.startswith("Current medication")]


@pytest.mark.parametrize("bad", [["Lisinopril"], ["metformin"], ["Metformin) (= (grimage-weight $m) 9.9) (X"],
                                 "Metformin", [3], ["Metformin"] * 11, ["Rapamycin"], ["Berberine"],
                                 ["Metformin\n(= (x) 1)"], [["Metformin"]], {"Metformin": 1}])
def test_a_medication_is_checked_against_the_pharmaceuticals_the_kb_has_an_interaction_for(bad):
    """Berberine is the SUPPLEMENT side of the one Interaction fact: no flag can fire for it as a drug."""
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient({**NO_GRIM, "medications": bad})
    assert excinfo.value.code == "invalid_medication" and excinfo.value.extra["supported"] == ["Metformin"]


def test_a_medication_listed_twice_is_one_atom():
    built = build_patient({**NO_GRIM, "medications": ["Metformin", "Metformin"]})
    assert built.medications == ["Metformin"]
    assert sum("CurrentMedication" in ln for ln in built.shared_atoms.splitlines()) == 1


def test_the_api_takes_medications_keeps_them_out_of_the_preview_atoms_and_into_the_shared_space(monkeypatch):
    from core.pln_runner import PLNRunResult
    body = _request("POST", "/patients/preview", json={**WITH_METFORMIN, "id": "M"}).json()
    assert body["medications"] == ["Metformin"] and "CurrentMedication" not in body["atoms"]
    assert any(w.startswith("Current medication recorded: Metformin") for w in body["warnings"])
    assert _request("POST", "/patients/preview", json={**NO_GRIM, "medications": ["Lisinopril"]}).status_code == 422
    seen = {}

    def run(**kw):
        seen["extra_atoms"] = kw["extra_atoms"]
        return PLNRunResult(status="empty", mode="runtime")

    monkeypatch.setattr(api_module, "run_query", run)
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(recommend-supplements-patient &self Caller_M)"))
    monkeypatch.setattr(api_module, "log_turn", Mock())
    answer = _request("POST", "/query", json={"message": "what supplements?", "patient": {**WITH_METFORMIN, "id": "M"}}).json()
    assert "(CurrentMedication Caller_M Metformin)" in seen["extra_atoms"]
    assert answer["patient"]["medications"] == ["Metformin"]
    # a question that reads no patient fact never gets the patient (and so never the medication)
    seen.clear()
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(infer &self Metformin CoronaryHeartDisease)"))
    _request("POST", "/query", json={"message": "q", "patient": {**WITH_METFORMIN, "id": "M"}})
    assert not seen.get("extra_atoms")


# ═══════════════ what the second review round-1 pass confirmed about the notes ═══════

def test_the_builder_decides_elevated_on_the_value_the_engine_will_read():
    """The atom carries 6 significant digits: z 1.0000004 is written `1`, which the KB (> z 1.0) does not call
    Elevated — so the witness promise must not either."""
    near = build_patient({"age": 58, "sex": "Male", "markers": {"CRP": 1.0000004}})
    assert near.witnesses == [] and near.markers[0].status == "Normal"
    assert "(MeasuredZ Caller_Patient CRP 1)" in near.atoms
    assert build_patient({"age": 58, "sex": "Male", "markers": {"CRP": 1.00001}}).witnesses == ["CRP"]
    # 6 significant digits, not 7: z 1.0000006 is written 1.0000006 -> `1.00000` (6 digits) and still not above 1.0
    for z in (1.0000006, 1.0000014):
        built = build_patient({"age": 58, "sex": "Male", "markers": {"CRP": z}})
        assert built.witnesses == [] and built.markers[0].status == "Normal", z
        assert "(MeasuredZ Caller_Patient CRP 1)" in built.atoms, z


def test_the_no_grimage_promise_is_hedged_to_what_the_plan_and_the_ranking_can_do():
    from core.patient_builder import STILL_WORK
    assert STILL_WORK.startswith("The diagnosis can still work")
    assert "personalise only where a supplement or an intervention reaches them" in STILL_WORK
    # a patient whose only witness is the smoking surrogate: the diagnosis names a cause, the plan is empty
    built = build_patient({"age": 58, "sex": "Male", "markers": {"DNAmPACKYRS": 2.0}})
    assert built.witnesses == ["DNAmPACKYRS"]


def test_a_bare_linageaccel_marker_has_no_hazard_to_pair_and_the_note_does_not_claim_one():
    from core.patient_context import no_grimage_prompt_hint, patient_form_warnings
    bare = build_patient({"age": 58, "sex": "Male", "markers": {"LinAgeAccel": {"value": 4.0, "unit": "years"}}})
    assert bare.has_linage2 and bare.linage2 is None                 # a marker, not a LinAgeDelta block
    assert any("hazard, the risk, the decomposition and the projections return nothing" in w
               and "counterfactuals and scenarios return 0 years" in w for w in bare.warnings)
    assert "hazard is computable" not in " ".join(bare.warnings)
    from core.patient_context import patient_prompt_section
    assert "linage-hazard-patient" not in patient_prompt_section(bare)      # no LinAge2 hint for a bare marker
    hint = no_grimage_prompt_hint(bare)
    assert "(predict-risk-patient &self" in hint and "linage-hazard-patient" not in hint
    pid = bare.patient_id
    (note,) = patient_form_warnings(f"(predict-risk-patient &self {pid})\n(linage-hazard-patient &self {pid})", bare)
    assert "ALL-CAUSE" not in note
    # neither a clock nor a LinAge2 result: only the form that returns nothing
    plain = build_patient(NO_GRIM)
    assert "linage-hazard-patient" not in no_grimage_prompt_hint(plain)
    with_block = build_patient(_with_linage2(NO_GRIM))
    assert "(linage-hazard-patient &self" in no_grimage_prompt_hint(with_block)


@pytest.mark.parametrize("form", [
    "(predict-risk-patient &self {p})", "(risk-decomposition-patient &self {p})", "(project-risk-patient &self {p} Metformin)",
    "(risk-scenarios &self {p})", "(predict-risk &self {p} CoronaryHeartDisease)",
    "(risk-decomposition &self {p} CoronaryHeartDisease)", "(project-risk &self {p} CoronaryHeartDisease CellularSenescence)",
    "(absolute-risk &self {p} CoronaryHeartDisease)", "(absolute-risk-at &self {p} CoronaryHeartDisease 1.0)",
    "(risk-ci &self {p} CoronaryHeartDisease)", "(risk-confidence &self {p} CoronaryHeartDisease)",
])
def test_every_form_that_prints_the_chd_number_gets_the_heart_risk_notes(form):
    from core.patient_context import patient_form_warnings
    no_clock = build_patient(NO_GRIM)
    assert patient_form_warnings(form.format(p=no_clock.patient_id), no_clock)
    reported = build_patient({**WITH_GRIM, "prevalent_chd": ["angina"]})
    (note,) = patient_form_warnings(form.format(p=reported.patient_id), reported)
    assert "FIRST coronary event" in note
    # not for another outcome, not for another patient
    assert patient_form_warnings(form.format(p="Patient001"), reported) == []
    other = form.format(p=reported.patient_id).replace("CoronaryHeartDisease", "AllCauseMortality")
    if "CoronaryHeartDisease" in form:
        assert patient_form_warnings(other, reported) == []


def test_a_note_that_two_checks_both_make_is_in_the_response_once(monkeypatch):
    from core.pln_runner import PLNRunResult
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(predict-risk-patient &self Caller_C2)"))
    monkeypatch.setattr(api_module, "run_query", Mock(return_value=PLNRunResult(status="empty", mode="runtime")))
    monkeypatch.setattr(api_module, "log_turn", Mock())
    patient = {"id": "C2", "age": 58, "sex": "Male", "markers": {"AgeAccelGrim": {"z": 0.7}}, "prevalent_chd": ["angina"]}
    for path, body in (("/query", {"message": "my heart risk?", "patient": patient}),
                       ("/metta/run", {"metta_query": "(predict-risk-patient &self Caller_C2)", "patient": patient})):
        warnings = _request("POST", path, json=body).json()["warnings"]
        assert len([w for w in warnings if "FIRST coronary event" in w]) == 1, (path, warnings)
        assert len(warnings) == len(set(warnings))


def test_the_form_notes_say_what_they_claim_and_only_for_the_forms_they_name():
    from core.patient_context import patient_form_warnings
    built = build_patient(NOTHING_USABLE)
    pid = built.patient_id
    (note,) = patient_form_warnings(f"(recommend-supplements &self {pid} (Omega3 Berberine))", built)   # the pool form
    assert "every supplement tier is empty" in note
    assert "Read that as 'nothing to work from', not 'no cause' or 'no benefit'" in note
    hazard_of_another = build_patient(_with_linage2(NO_GRIM))
    (heart,) = patient_form_warnings(
        f"(predict-risk-patient &self {hazard_of_another.patient_id})\n(linage-hazard-patient &self Patient001)", hazard_of_another)
    assert "ALL-CAUSE" not in heart                      # the hazard is another patient's: nothing of this one's to label


def test_a_grimage_value_with_no_age_or_sex_means_no_risk_model_input_and_no_first_event_note():
    built = build_patient({"markers": {"AgeAccelGrim": 1.2}, "prevalent_chd": ["angina"]})
    assert built.has_grimage and not built.can_predict_risk
    assert not [w for w in built.warnings if "FIRST coronary event" in w]
    from core.patient_context import medication_prompt_hint, prevalent_chd_prompt_hint
    assert prevalent_chd_prompt_hint(built) == "" and medication_prompt_hint(built) == ""


def test_the_linage2_block_marker_is_elevated_only_if_the_atom_the_engine_reads_is():
    """z 1.0000004 is written `1`, which the KB (> 1.0) does not call Elevated — for the clock too."""
    import copy
    import json
    fixture = json.loads((REPO / "tests" / "fixtures" / "linage2_response.json").read_text(encoding="utf-8"))
    d = copy.deepcopy(fixture)
    delta = 8.66 * 1.0000004
    d["metadata"]["delta_ba_ca"] = delta
    d["biological_age"] = d["metadata"]["chronological_age"] + delta
    built = build_patient({"id": "L", "age": 58, "sex": "Male", "markers": {"CRP": 2.0}, "linage2": d}, linage_sd_to_years=8.66)
    clock = next(m for m in built.markers if m.name == "LinAgeAccel")
    assert "(MeasuredZ Caller_L LinAgeAccel 1)" in built.atoms and clock.status == "Normal"


# ═══════════ RDW and the albumin deficit are z-only markers (#7) ═══════════

@pytest.mark.parametrize("name", ["RDW", "LowSerumAlbumin"])
def test_rdw_and_the_albumin_deficit_take_a_z_and_refuse_a_raw_value(name):
    built = build_patient({"age": 58, "sex": "Male", "markers": {name: {"z": 1.5}}})
    assert built.witnesses == [name] and f"(MeasuredZ Caller_Patient {name} 1.5)" in built.atoms
    with pytest.raises(PatientSpecError) as exc:
        build_patient({"age": 58, "sex": "Male", "markers": {name: {"value": 14.1, "unit": "%"}}})
    assert exc.value.code == "raw_value_unsupported"
    assert build_patient({"age": 58, "sex": "Male", "markers": {name: 0.4}}).witnesses == []      # not above the threshold


def test_a_linage2_block_is_joinable_through_rdw_or_the_albumin_deficit():
    """The 'no witness a cause can be credited to' warning is about the markers a LinAge2 input reads out."""
    import json
    fx = json.loads((REPO / "tests" / "fixtures" / "linage2_response.json").read_text(encoding="utf-8"))
    base = {"id": "J", "age": 58, "sex": "Male", "linage2": fx}
    nothing = build_patient({**base, "markers": {"DNAmADM": 1.0}})
    assert any("RDW, LowSerumAlbumin as z only" in w for w in nothing.warnings)
    for name in ("RDW", "LowSerumAlbumin"):
        joined = build_patient({**base, "markers": {name: 0.5}})
        assert not any("without any of the markers the knowledge base can join it to" in w for w in joined.warnings), name


# ═══════════ triglycerides: a curated threshold on the z scale (#12) ═══════════

def test_the_triglyceride_reference_puts_z_one_at_150_mg_dl_and_says_it_is_not_a_cohort():
    from core.patient_builder import MARKERS
    ref = MARKERS["Triglycerides"].reference
    assert ref.to_z(150.0)[0] > 1.0 > ref.to_z(149.0)[0]
    assert "not a cohort" in ref.source and "fasting" in ref.source
    built = build_patient({"age": 58, "sex": "Male", "markers": {"Triglycerides": {"value": 190, "unit": "mg/dL"}}})
    assert built.witnesses == ["Triglycerides"]
    assert any("FASTING value" in w and "cannot tell" in w for w in built.warnings)
    assert any("Standardised server-side" in w and "Triglycerides" in w for w in built.warnings)


# ═══════════ a reported CHD as an observation for the diagnosis (#11) ═══════════

def test_a_reported_chd_is_one_shared_atom_that_never_reaches_the_linage2_space_or_a_generic_query(monkeypatch):
    from core.pln_runner import PLNRunResult
    built = build_patient({**WITH_GRIM, "id": "H", "prevalent_chd": ["angina", "heart attack"]})
    assert built.shared_atoms.count("(PatientCondition Caller_H CoronaryHeartDisease)") == 1     # one, whatever the items
    assert "PatientCondition" not in built.atoms                  # LinAge2 and the preview: no room for a new head
    assert "PatientCondition" not in build_patient({**WITH_GRIM, "id": "H"}).shared_atoms
    seen = {}

    def run(**kw):
        seen["extra_atoms"] = kw["extra_atoms"]
        return PLNRunResult(status="empty", mode="runtime")

    monkeypatch.setattr(api_module, "run_query", run)
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(diagnose-patient &self Caller_H)"))
    monkeypatch.setattr(api_module, "log_turn", Mock())
    patient = {**WITH_GRIM, "id": "H", "prevalent_chd": ["angina"]}
    answer = _request("POST", "/query", json={"message": "why are my labs off?", "patient": patient}).json()
    assert "(PatientCondition Caller_H CoronaryHeartDisease)" in seen["extra_atoms"]
    assert any("read by the diagnosis as one observation" in w for w in answer["warnings"])
    seen.clear()
    monkeypatch.setattr(api_module, "translate", lambda **kw: _translation("(infer &self Metformin CoronaryHeartDisease)"))
    _request("POST", "/query", json={"message": "q", "patient": patient})
    assert not seen.get("extra_atoms")           # a question that reads no patient fact never gets the patient


def test_the_no_witness_note_for_a_patient_who_reports_heart_disease_does_not_say_the_diagnosis_is_empty():
    from core.patient_builder import NO_WITNESS_CHD_PREFIX, NO_WITNESS_PREFIX
    from core.patient_context import patient_form_warnings
    built = build_patient({**NOTHING_USABLE, "prevalent_chd": ["angina"]})
    assert built.witnesses == []
    (note,) = _notes(built, NO_WITNESS_CHD_PREFIX)
    assert not _notes(built, NO_WITNESS_PREFIX)
    assert "every supplement tier is empty" in note.lower() and "population ranking" in note
    assert "diagnosis answers from the reported heart disease alone" in note and "returns ()" not in note
    (diag,) = [w for w in patient_form_warnings(f"(diagnose-patient &self {built.patient_id})", built) if "no elevated value" in w]
    assert "answers from the reported heart disease alone" in diag and "returns ()" not in diag
    (plan,) = [w for w in patient_form_warnings(f"(recommend-supplements-patient &self {built.patient_id})", built)
               if "no elevated value" in w]
    assert "every supplement tier is empty" in plan
    # without the report the old note stands
    assert _notes(build_patient(NOTHING_USABLE), NO_WITNESS_PREFIX)


def test_a_helper_over_a_per_request_patient_fact_is_not_called_data_less():
    """patient-conditions reads PatientCondition, which only a caller's patient has: the runtime KB holds no row for
    it, and the 'this space holds NO facts' warning would be false."""
    from ontology.scoped_forms import dataless_forms, scoped_form_warnings
    kb, inv = api_module._runtime_kb_paths(), api_module._runtime_inventory()
    assert "patient-conditions" not in dataless_forms(kb, inv)
    assert scoped_form_warnings("!(patient-conditions &self Caller_Me)", kb, inv) == []


def test_the_reported_heart_disease_notes_follow_the_cause_list_the_diagnosis_was_given():
    from core.patient_context import CHD_REACHING_CAUSES, patient_form_warnings
    built = build_patient({**NOTHING_USABLE, "prevalent_chd": ["angina"]})
    pid = built.patient_id
    default = patient_form_warnings(f"(diagnose-patient &self {pid})", built)
    assert any("answers from the reported heart disease alone" in w for w in default)
    reaching = patient_form_warnings(f"(diagnose-patient &self {pid} (InsulinResistance ChronicInflammation))", built)
    assert any("answers from the reported heart disease alone" in w for w in reaching)
    listed = patient_form_warnings(f"(diagnose-patient &self {pid} (ChronicInflammation MitochondrialDysfunction))", built)
    assert any("adds nothing to this diagnosis" in w for w in listed)
    assert any("returns ()" in w for w in listed) and not any("answers from the reported heart disease alone" in w for w in listed)
    assert CHD_REACHING_CAUSES == {"InsulinResistance", "DeregulatedNutrientSensing", "CellularSenescence"}


def test_the_reported_heart_disease_note_is_generated_from_the_causes_the_engine_reaches_and_says_what_it_claims():
    from core.patient_builder import CHD_REACHING_CAUSES, _words, chd_observation_note
    note = chd_observation_note(["angina"])
    for cause in CHD_REACHING_CAUSES:
        assert _words(cause) in note
    assert "chronic inflammation" not in note and "mitochondrial" not in note
    assert "smoking has no curated edge to heart disease" in note and "population-level associations" in note
    assert "supplement plan and the intervention ranking do not see it" in note
    assert "prevalence item" in note and "not a measured value" in note


def test_two_diagnose_forms_for_one_patient_get_the_observation_note_if_either_reaches_the_report():
    from core.patient_context import patient_form_warnings
    built = build_patient({**NOTHING_USABLE, "prevalent_chd": ["angina"]})
    pid = built.patient_id
    both = (f"(diagnose-patient &self {pid} (ChronicInflammation))\n"
            f"(diagnose-patient &self {pid} (InsulinResistance))")
    assert any("read by the diagnosis as one observation" in w for w in patient_form_warnings(both, built))
    neither = (f"(diagnose-patient &self {pid} (ChronicInflammation))\n"
               f"(diagnose-patient &self {pid} (MitochondrialDysfunction))")
    assert any("adds nothing to this diagnosis" in w for w in patient_form_warnings(neither, built))
