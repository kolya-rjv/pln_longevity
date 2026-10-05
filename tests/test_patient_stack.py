"""Questions that name a patient run in the patient stack (core.pln_runner.patient_stack).

The full shared space sits at hyperon 0.2.10's head-symbol edge: patient forms abort
it (a panic in its space index, uncatchable from Python) depending on details as
small as one float — rank-interventions-for-patient for Patient001, the supplement
forms for Patient001/002, and diagnose / supplements / ranking for a caller whose
CRP z is 1.2. The patient stack drops seven files no patient form reads. These tests
pin the three claims that justify it: the answers do not change where the full
stack answers at all; the forms that aborted now answer; and there is margin.

Every MeTTa run here is a SUBPROCESS: an abort must fail one test, not kill pytest.

    pytest tests/test_patient_stack.py -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")

import api as api_module  # noqa: E402
from core.patient_context import names_a_patient, patient_atoms_for, reads_patients  # noqa: E402
from core.pln_runner import PATIENT_STACK_EXCLUDED, patient_stack  # noqa: E402

CALLER = ("(InstanceOf Caller_Me PatientProfile)\n(PatientAge Caller_Me 58)\n(PatientSex Caller_Me Male)\n"
          "(PatientSmoking Caller_Me CurrentSmoker)\n(MeasuredZ Caller_Me AgeAccelGrim 0.9375)\n"
          "(MeasuredZ Caller_Me CRP {z})\n(MeasuredZ Caller_Me HbA1c {z})")
DIAGNOSE = "!(diagnose-patient &self {P} (CellularSenescence ChronicInflammation InsulinResistance))"
SUPPLEMENTS = "!(recommend-supplements-patient &self {P})"
RANK = "!(rank-interventions-for-patient &self {P} (Metformin Berberine CaloricRestriction DasatinibPlusQuercetin) CoronaryHeartDisease)"
RISK = "!(predict-risk-patient &self {P})"


def _run(stack: str, query: str, extra: str = "") -> dict:
    """Run one query in a fresh process; {'rc', 'status', 'atoms'}."""
    probe = "\n".join([
        "import sys, json",
        f"sys.path.insert(0, {str(PLN_CHAT)!r})",
        "import api",
        "from core.pln_runner import patient_stack, run_query",
        "kb = api._runtime_kb_paths()",
        "kb = patient_stack(kb) if sys.argv[1] == 'patient' else kb",
        "r = run_query(sys.argv[2], kb_files=kb, extra_atoms=sys.argv[3] or None)",
        "print('RESULT ' + json.dumps({'status': r.status, 'atoms': [x.atom for x in r.results]}))",
    ])
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", probe, stack, query, extra],
                          capture_output=True, text=True, timeout=600, cwd=str(REPO))
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    out = json.loads(line[0][7:]) if line else {"status": "abort", "atoms": []}
    out["rc"] = done.returncode
    out["secs"] = time.monotonic() - started      # wall clock of the whole process
    return out


# ═══════════════════════════ what it is ═══════════════════════════════════════

def test_the_patient_stack_is_the_runtime_stack_minus_seven_named_files():
    runtime = api_module._runtime_kb_paths()
    names = [p.name for p in runtime]
    assert set(PATIENT_STACK_EXCLUDED) <= set(names), "an excluded file left the runtime stack"
    stack = patient_stack(runtime)
    assert [p.name for p in stack] == [n for n in names if n not in PATIENT_STACK_EXCLUDED]
    for needed in ("patient_profile.metta", "pln_abductive_diagnosis.metta", "pln_risk_prediction.metta",
                   "pln_counterfactual.metta", "pln_intervention_ranking.metta",
                   "pln_supplement_recommendation.metta", "lifestyle_evidence.metta"):
        assert needed in {p.name for p in stack}


@pytest.mark.parametrize("program, expected", [
    ("(diagnose-patient &self Caller_Me (A))", True),
    ("(predict-risk-patient &self Patient001)", True),
    ("(match &self (Inheritance $x PatientProfile) $x)", False),
    ("(match &self (UsesSpecies $e Mus_musculus) $e)", False),
    ("(rank-drugage-lifespan (Rapamycin Metformin))", False),
])
def test_a_patient_question_is_recognised_by_its_patient(program, expected):
    assert names_a_patient(program) is expected


# ═══════════════════════════ it changes no answer ═════════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("patient, form", [
    ("Patient001", DIAGNOSE), ("Patient001", RISK), ("Patient002", DIAGNOSE),
    ("Patient002", RISK), ("Patient003", RANK), ("Patient003", SUPPLEMENTS),
    ("Patient001", "!(counterfactual-patient &self {P} CellularSenescence)"),
    ("Patient001", "!(decompose-grimage &self {P})"),
])
def test_where_the_full_stack_answers_the_patient_stack_answers_identically(patient, form):
    q = form.format(P=patient)
    full, scoped = _run("full", q), _run("patient", q)
    assert full["rc"] == 0 and full["status"] == "ok", f"control: the full stack no longer answers {q}"
    assert scoped["rc"] == 0 and scoped["atoms"] == full["atoms"]


@pytest.mark.slow
def test_patient001s_captured_outputs_are_what_a_patient_now_sees():
    """tests/test_hallmark_targeting.py captured these three outputs on an earlier
    commit and asserts them in the FULL stack, where the ranking now aborts the
    process. Routed as production routes them, they are reproduced to the digit."""
    risk = _run("patient", RISK.format(P="Patient001"))["atoms"][0]
    assert "(point 0.12605177716424967)" in risk
    dx = _run("patient", "!(diagnose-patient &self Patient001 "
                         "(CellularSenescence MitochondrialDysfunction ChronicInflammation))")["atoms"][0]
    assert re.findall(r"\(Hypothesis (\w+)", dx) == ["CellularSenescence", "ChronicInflammation",
                                                     "MitochondrialDysfunction"]
    rank = _run("patient", "!(rank-interventions-for-patient &self Patient001 (DasatinibPlusQuercetin "
                           "Fisetin Spermidine Elamipretide) CoronaryHeartDisease)")["atoms"][0]
    scored = re.findall(r"\(scored (\w+) ([-\d.eE]+)", rank)
    assert [n for n, _ in scored] == ["DasatinibPlusQuercetin", "Fisetin", "Spermidine"]
    assert float(scored[0][1]) == pytest.approx(0.2306046747621094, abs=1e-12)


# ═══════════════════════════ it answers what aborted ══════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("patient, form", [
    ("Patient001", RANK), ("Patient001", SUPPLEMENTS), ("Patient002", SUPPLEMENTS),
])
def test_the_built_in_patient_forms_that_aborted_now_answer(patient, form):
    q = form.format(P=patient)
    assert _run("patient", q)["status"] == "ok"
    assert _run("full", q)["rc"] != 0, (
        f"the full stack no longer aborts on {q}: the control is void (the engine or "
        f"the KB changed) — the patient stack is still correct, re-measure the margin")


@pytest.mark.slow
@pytest.mark.parametrize("z", ["1.2", "2.0", "0.31"])
@pytest.mark.parametrize("form", [DIAGNOSE, SUPPLEMENTS, RANK])
def test_a_caller_patient_is_answered_whatever_its_values(z, form):
    out = _run("patient", form.format(P="Caller_Me"), CALLER.format(z=z))
    assert out["rc"] == 0 and out["status"] == "ok"


@pytest.mark.slow
def test_the_patient_stack_keeps_a_head_symbol_margin():
    pad = "\n".join(f'(ProbeHead{i} ProbeSym{i} "probe {i}")' for i in range(32))
    for form in (DIAGNOSE, SUPPLEMENTS, RANK):
        out = _run("patient", form.format(P="Caller_Me"), CALLER.format(z="1.2") + "\n" + pad)
        assert out["rc"] == 0 and out["status"] == "ok", (
            "the patient stack no longer tolerates 32 extra head symbols on top of a caller "
            "patient; something spent the margin")


# ═══════════════════════════ it is where they run ═════════════════════════════

def test_api_routes_a_patient_program_to_the_patient_stack(monkeypatch):
    runtime = api_module._runtime_kb_paths()
    assert api_module._generic_kb("(predict-risk-patient &self Patient001)") == patient_stack(runtime)
    assert api_module._generic_kb("(match &self (UsesSpecies $e Mus_musculus) $e)") == runtime


def test_the_chat_routes_the_same_way():
    import app as app_module
    assert app_module._generic_kb("(diagnose-patient &self Caller_Me (A))") == patient_stack(app_module._ALL_KB_PATHS)
    assert app_module._generic_kb("(match &self (HasSex $e Hermaphrodite) $e)") == app_module._ALL_KB_PATHS


# ═══════════════════════════ a loaded patient never reaches the full space ════

@pytest.mark.parametrize("program, reads", [
    ("(match &self (MeasuredZ $p CRP $z) ($p $z))", True),
    ("(match &self (PatientAge $p $a) ($p $a))", True),
    ("(match &self (InstanceOf $p PatientProfile) $p)", True),
    ("(diagnose-patient &self Caller_Me (A))", True),
    ("(infer &self Metformin CoronaryHeartDisease)", False),
    ("(match &self (InstanceOf $x $t) ($x $t))", False),
])
def test_a_program_that_reads_patient_facts_is_recognised(program, reads):
    assert reads_patients(program) is reads
    assert patient_atoms_for(program, "(PatientAge Caller_Me 58)") == ("(PatientAge Caller_Me 58)" if reads else None)


@pytest.mark.slow
def test_listing_patient_facts_with_a_session_patient_loaded_does_not_abort():
    """It used to: a generic program ran in the FULL space with the session patient's
    atoms, and enumerating MeasuredZ there panics hyperon (trie.rs:179). Routed as
    production routes it now, it answers and lists the caller too."""
    q = "!(match &self (MeasuredZ $p CRP $z) ($p $z))"
    caller = CALLER.format(z="1.2")
    full = _run("full", q, caller)
    assert full["rc"] != 0, "control: the full space no longer aborts here — the guard is still right"
    assert api_module._generic_kb(q) == patient_stack(api_module._runtime_kb_paths())
    routed = _run("patient", q, patient_atoms_for(q, caller) or "")
    assert routed["rc"] == 0 and any("Caller_Me" in a for a in routed["atoms"])
    generic = "!(infer &self Metformin CoronaryHeartDisease)"
    assert patient_atoms_for(generic, caller) is None and api_module._generic_kb(generic) == api_module._runtime_kb_paths()



# ═══════════════════════════ the marker set: one dedupe, no latency wall ══════
#
# patient_profile.metta's `patient-markers` runs once per candidate in the ranking and
# supplement forms. Its dedupe used to be the interpreted, quadratic `unique-tuple`:
# a supplement plan cost 1.4 s with no extra lab, 13 s with 8 and 50 s with 16 (104 s on
# the cloud box, past the 60 s query timeout). The grounded `unique-atom` costs 5 s at
# 16 — if it is let-forced (the bare call dedupes the unevaluated expression and every
# observation silently disappears). docs/kb_quick_wins/REPORT.md S2.

GOLDEN = json.loads(
    (REPO / "tests" / "fixtures" / "patient_stack_builtin_golden.json").read_text(encoding="utf-8"))["cases"]

TAB_SMOKER = ("(InstanceOf Caller_Me PatientProfile)\n(PatientAge Caller_Me 58)\n(PatientSex Caller_Me Male)\n"
              "(PatientSmoking Caller_Me CurrentSmoker)\n(MeasuredZ Caller_Me CRP 0.438255)\n"
              "(MeasuredZ Caller_Me FastingGlucose 1.41667)\n(MeasuredZ Caller_Me HbA1c 1.8)")
#: labs no curated edge reaches, under names no future bridge can claim
INERT_LABS = "\n".join(f"(MeasuredZ Caller_Me InertLab{i:02d} 0.3)" for i in range(1, 17))


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_a_built_in_patient_is_answered_to_the_digit_as_it_was_recorded(name):
    """Patient001-003 through diagnosis, observations, marker set, ranking, supplements and
    risk, against outputs recorded before the dedupe changed (and to be moved only with an
    intended change to a built-in patient or to what a form says)."""
    case = GOLDEN[name]
    out = _run("patient", case["query"])
    assert out["rc"] == 0 and out["status"] == "ok"
    assert out["atoms"] == case["atoms"]


@pytest.mark.slow
def test_a_marker_recorded_twice_is_one_marker_and_its_observations_survive():
    """A MeasuredZ and a MeasuredRaw for one marker are ONE marker (counting it twice
    would double its weight in diagnose and in patient-relevance), the explicit z wins,
    and the observations are the elevated ones — an un-forced unique-atom returns ()."""
    atoms = ("(InstanceOf Caller_X PatientProfile)\n(PatientAge Caller_X 58)\n(PatientSex Caller_X Male)\n"
             "(MeasuredZ Caller_X HbA1c 1.0)\n(MeasuredZ Caller_X CRP 0.5)\n(MeasuredRaw Caller_X HbA1c 6.0)\n"
             "(MeasuredZ Caller_X FastingGlucose 1.5)\n(MeasuredRaw Caller_X CRP 3.0)")
    markers = _run("patient", "!(patient-markers &self Caller_X)", atoms)
    assert markers["rc"] == 0 and sorted(re.findall(r"\w+", markers["atoms"][0])) == [
        "CRP", "FastingGlucose", "HbA1c"]
    obs = _run("patient", "!(patient-observations &self Caller_X)", atoms)
    assert obs["atoms"] == ["(FastingGlucose)"]


@pytest.mark.slow
def test_sixteen_more_markers_make_the_supplement_plan_neither_slow_nor_different():
    base = _run("patient", SUPPLEMENTS.format(P="Caller_Me"), TAB_SMOKER)
    wide = _run("patient", SUPPLEMENTS.format(P="Caller_Me"), TAB_SMOKER + "\n" + INERT_LABS)
    assert base["status"] == wide["status"] == "ok"
    assert wide["atoms"] == base["atoms"], "labs no edge reaches must not change the plan"
    # before the fix: 50 s here, 104 s on the cloud box, 35x the no-extra-lab run
    assert wide["secs"] < 30, f"{wide['secs']:.1f} s for 16 extra markers (limit 30; the query timeout is 60)"
    assert wide["secs"] < 15 * base["secs"], (
        f"16 extra markers cost {wide['secs'] / base['secs']:.0f}x the plan without them")


# ═══════════════════════════ "what drives my abnormal labs?": the default causes ══
#
# The 3-argument diagnose-patient needs a typed list of candidate causes. The tab's own
# question names none, and a translator that copies the only example it has seen hands in
# three hallmarks that do not reach HbA1c or glucose — () for the patient whose witnesses
# those are. The 2-argument form carries the knowledge base's default list.

DEFAULT_CAUSES = ("(CellularSenescence ChronicInflammation MitochondrialDysfunction "
                  "InsulinResistance DeregulatedNutrientSensing SmokingPackYears)")
THREE_HALLMARKS = "(CellularSenescence MitochondrialDysfunction ChronicInflammation)"
HEALTHY_WOMAN = ("(InstanceOf Caller_Me PatientProfile)\n(PatientAge Caller_Me 45)\n(PatientSex Caller_Me Female)\n"
                 "(PatientSmoking Caller_Me NeverSmoker)\n(MeasuredZ Caller_Me CRP -1.20397)\n"
                 "(MeasuredZ Caller_Me FastingGlucose -0.583333)\n(MeasuredZ Caller_Me HbA1c -0.6)")


@pytest.mark.slow
def test_diagnosing_with_no_cause_list_searches_the_default_causes_and_finds_the_metabolic_one():
    two = _run("patient", "!(diagnose-patient &self Caller_Me)", TAB_SMOKER)
    assert two["rc"] == 0 and two["status"] == "ok"
    assert two["atoms"] == _run("patient", f"!(diagnose-patient &self Caller_Me {DEFAULT_CAUSES})",
                                TAB_SMOKER)["atoms"]
    assert re.findall(r"\(Hypothesis (\w+) ", two["atoms"][0])[0] == "InsulinResistance"
    assert "(Hypothesis InsulinResistance (stv 0.8 0.8775) 2.0 " in two["atoms"][0]
    assert "(SupportedBy (HbA1c FastingGlucose))" in two["atoms"][0]
    # the control: the list the translator used to copy does not reach these two labs
    assert _run("patient", f"!(diagnose-patient &self Caller_Me {THREE_HALLMARKS})",
                TAB_SMOKER)["atoms"] == ["()"]


@pytest.mark.slow
@pytest.mark.parametrize("patient, first", [("Patient001", "CellularSenescence"),
                                            ("Patient002", "InsulinResistance"),
                                            ("Patient003", "SmokingPackYears")])
def test_the_default_causes_answer_each_built_in_patient_and_leave_the_three_argument_form_alone(patient, first):
    two = _run("patient", f"!(diagnose-patient &self {patient})")
    assert re.findall(r"\(Hypothesis (\w+) ", two["atoms"][0])[0] == first
    assert two["atoms"] == _run("patient", f"!(diagnose-patient &self {patient} {DEFAULT_CAUSES})")["atoms"]
    three = GOLDEN[f"{patient}_dx"]
    assert _run("patient", three["query"])["atoms"] == three["atoms"]


@pytest.mark.slow
def test_a_patient_with_nothing_elevated_gets_an_honest_empty_not_a_cause():
    out = _run("patient", "!(diagnose-patient &self Caller_Me)", HEALTHY_WOMAN)
    assert out["rc"] == 0 and out["atoms"] == ["()"]
