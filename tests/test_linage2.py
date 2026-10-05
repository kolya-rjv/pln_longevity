"""The LinAge2 clinical clock, end to end: response -> atoms -> scoped inference -> JSON.

What is asserted, in the order the chain runs:

  * the feature vocabulary is read off linage2_core.metta (59 inputs, four of which
    read out a KB biomarker), so the Python surface cannot drift from the ontology;
  * a LinAge2 /predict response becomes request-scoped atoms — one LinAgeDelta, one
    MeasuredZ for the clock, one LinAgeContribution per input flagged Measured or
    Imputed — and every way a payload can lie is a 422 with a code;
  * in the LinAge2 scoped space the decomposition's totals reproduce the delta EXACTLY
    (measured + imputed + age-term residual), a cause is credited only under a witness
    the patient's own values supply, the counterfactual gates the smoking lever on the
    patient being a current smoker, the hazard is HR^delta off the Fong 2025 record, and
    the absolute risk yields nothing until a data-backed baseline is loaded;
  * the GrimAge layer's published numbers are byte-identical inside the LinAge2 stack,
    and `resolve-lever` still finds SmokingPackYears first;
  * the LinAge2 files are NOT in the shared execution space (putting them there aborts
    the process), and the scoped stack keeps a measured head-symbol margin — the budget
    that decided this layer's shape (linage2_core.metta header);
  * the HTTP surfaces route a `linage-*` form to the scoped space, report `routed:
    "linage2"`, and warn when a form names a patient with no LinAge2 result.

The fixture (tests/fixtures/linage2_response.json) is SYNTHETIC: a 58-year-old male
with 33 measured and 26 imputed inputs, whose contributions were computed from the
published model weights for an arbitrary z vector. Its numbers are properties of the
fixture, never claims about a person or a population.

Run from the repository root:
    pytest tests/test_linage2.py -q
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")

from core.linage2_builder import (  # noqa: E402
    build_linage2,
    feature_catalog,
    feature_listing,
    linage_sd_to_years,
)
from core.linage2_router import (  # noqa: E402
    DEFAULT_LEVERS,
    LINAGE2_FORMS,
    analysis_program,
    collect_analysis,
    linage2_form_warnings,
    parse_linage2_query,
    parse_sexpr,
    split_linage2_program,
)
from core.metta_validator import ValidationResult, merge_validation_results  # noqa: E402
from core.patient_builder import PatientSpecError, build_patient  # noqa: E402
from core.pln_runner import (  # noqa: E402
    LINAGE2_LAYER_FILES,
    LINAGE2_PATIENT_STACK,
    PLNAtomResult,
    PLNRunResult,
    linage2_patient_kb,
    merge_run_results,
)

FIXTURE = REPO / "tests" / "fixtures" / "linage2_response.json"
HAZARD_PER_YEAR = 1.093          # linage2_fong2025_evidence.metta
EVIDENCE_CONFIDENCE = 0.60        # Epidemiological tier
RISK_DISCOUNT = 0.9               # risk-conf-discount, pln_risk_prediction.metta


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _patient(**overrides) -> dict:
    base = {
        "id": "W58", "age": 58, "sex": "Male", "smoking": "CurrentSmoker",
        "markers": {"HbA1c": 1.6, "CRP": 0.3},
        "linage2": _fixture(),
    }
    base.update(overrides)
    return base


# ═══════════════════════════ the vocabulary ═══════════════════════════════════

def test_the_catalog_is_read_off_the_ontology_and_has_every_input():
    catalog = feature_catalog()
    assert len(catalog) == 59
    assert catalog["LBXCRP"].symbol == "CRP" and catalog["LBXCRP"].reads_out == "CRP"
    assert catalog["LBXGH"].symbol == "HbA1c" and catalog["LBXGH"].reads_out == "HbA1c"
    assert catalog["LBDSGLSI"].reads_out == "FastingGlucose"
    assert catalog["LBXCOT"].reads_out == "CurrentTobaccoExposure"
    assert catalog["LBXRDW"].reads_out == "RDW" and catalog["LBDSALSI"].reads_out == "LowSerumAlbumin"
    assert sum(1 for f in catalog.values() if f.reads_out) == 6
    assert catalog["fs1Score"].symbol == "ComorbidityScore"
    assert all(f.description for f in catalog.values())
    # every symbol is a (ModelInput <F> LinAge2) fact in the file, and no code is a fact
    core = (REPO / "linage2_core.metta").read_text(encoding="utf-8")
    for f in catalog.values():
        assert f"(ModelInput {f.symbol} LinAge2)" in core
    assert not re.search(r"^\(NHANESCode", core, re.M), "the codes are structured comments, not atoms (budget)"


def test_the_fixture_matches_the_catalog_exactly():
    fx = _fixture()
    codes = {c["feature"] for c in fx["metadata"]["feature_contributions"]}
    assert codes == set(feature_catalog())


def test_the_sd_knob_is_read_from_the_rules_file():
    text = (REPO / "pln_linage2.metta").read_text(encoding="utf-8")
    m = re.search(r"\(=\s*\(linage-sd-to-years\)\s*([\d.]+)\s*\)", text)
    assert m and float(m.group(1)) == linage_sd_to_years()


def test_feature_listing_says_how_a_cause_is_credited():
    listing = feature_listing()
    assert len(listing) == 59
    by_code = {e["feature"]: e for e in listing}
    assert "Elevated" in by_code["LBXCRP"]["cause_attribution"]
    assert "CurrentSmoker" in by_code["LBXCOT"]["cause_attribution"]
    assert "none" in by_code["LBXHGB"]["cause_attribution"]


# ═══════════════════════════ response -> atoms ═══════════════════════════════

def test_a_response_becomes_one_delta_and_one_contribution_per_input():
    built = build_linage2(_fixture(), "Caller_W58", age=58)
    fx = _fixture()["metadata"]
    assert built.delta_years == pytest.approx(fx["delta_ba_ca"])
    assert built.z == pytest.approx(fx["delta_ba_ca"] / linage_sd_to_years())
    assert built.status == "Normal"            # 8.28 y / 8.66 y per SD = 0.96 SD
    assert len(built.measured) == 33 and len(built.imputed) == 26
    atoms = built.atoms.splitlines()
    assert sum(a.startswith("(LinAgeDelta Caller_W58 ") for a in atoms) == 1
    assert sum(a.startswith("(LinAgeContribution Caller_W58 ") for a in atoms) == 59
    assert sum(a.endswith(" Measured)") for a in atoms) == 33
    assert sum(a.endswith(" Imputed)") for a in atoms) == 26
    # the clock's z is rendered by build_patient, not twice
    assert "MeasuredZ" not in built.atoms
    # measured first, then by |years|
    assert not built.contributions[0].imputed
    assert built.contributions[0].symbol == "SerumCotinine"
    assert any("IMPUTED" in w for w in built.warnings)


def test_the_flattened_data_shape_is_accepted_too():
    fx = _fixture()
    flat = {"biological_age": fx["biological_age"], **fx["metadata"]}
    assert build_linage2(flat, "P", age=None).delta_years == pytest.approx(fx["metadata"]["delta_ba_ca"])


@pytest.mark.parametrize("mutate, code", [
    (lambda fx: fx["metadata"]["feature_contributions"][0].__setitem__("feature", "LBXFOO"),
     "unknown_linage2_feature"),
    (lambda fx: fx["metadata"]["feature_contributions"].append(
        dict(fx["metadata"]["feature_contributions"][0])), "duplicate_linage2_feature"),
    (lambda fx: fx["metadata"].__setitem__("delta_ba_ca", fx["metadata"]["delta_ba_ca"] + 1.0),
     "linage2_inconsistent"),
    (lambda fx: (fx["metadata"].__setitem__("delta_ba_ca", 60.0),
                 fx.__setitem__("biological_age", 118.0)), "implausible_linage2_delta"),
    (lambda fx: fx["metadata"]["feature_contributions"][0].__setitem__("contribution_years", 99.0),
     "implausible_linage2_contribution"),
    (lambda fx: fx["metadata"].__setitem__("feature_contributions", []), "invalid_linage2"),
    (lambda fx: fx.__delitem__("biological_age"), "invalid_linage2"),
])
def test_every_way_a_payload_can_lie_is_a_422_with_a_code(mutate, code):
    fx = _fixture()
    mutate(fx)
    with pytest.raises(PatientSpecError) as excinfo:
        build_linage2(fx, "P", age=58)
    assert excinfo.value.code == code


def test_a_response_for_a_different_age_is_refused():
    with pytest.raises(PatientSpecError) as excinfo:
        build_linage2(_fixture(), "P", age=40)
    assert excinfo.value.code == "linage2_age_mismatch"


def test_a_large_but_published_delta_warns_rather_than_refuses():
    fx = _fixture()
    fx["metadata"]["delta_ba_ca"] = 38.0
    fx["biological_age"] = fx["metadata"]["chronological_age"] + 38.0
    built = build_linage2(fx, "P", age=58)
    assert built.status == "Elevated"
    assert any("35" in w for w in built.warnings)


def test_build_patient_carries_the_block_as_the_clock_marker_once():
    built = build_patient(_patient())
    assert built.has_linage2 and built.linage2 is not None
    clock = [m for m in built.markers if m.name == "LinAgeAccel"]
    assert len(clock) == 1 and clock[0].derived and clock[0].unit == "years"
    assert built.atoms.count("(MeasuredZ Caller_W58 LinAgeAccel ") == 1
    assert built.atoms.count("(LinAgeDelta Caller_W58 ") == 1
    assert built.age == 58 and built.can_predict_risk is False   # no AgeAccelGrim -> CHD model silent


def test_age_is_taken_from_the_response_when_not_sent():
    built = build_patient(_patient(age=None))
    assert built.age == 58 and any("taken from the LinAge2" in w for w in built.warnings)


def test_a_block_and_a_bare_clock_marker_together_are_refused():
    with pytest.raises(PatientSpecError) as excinfo:
        build_patient(_patient(markers={"LinAgeAccel": 1.0}))
    assert excinfo.value.code == "duplicate_clock"


def test_a_bare_clock_marker_alone_works_and_says_what_it_cannot_do():
    built = build_patient({"id": "B", "age": 60, "sex": "Female",
                           "markers": {"LinAgeAccel": {"value": 8.66, "unit": "years"}}})
    z = [m for m in built.markers if m.name == "LinAgeAccel"][0]
    assert z.z == pytest.approx(1.0) and "linage-sd-to-years" in z.formula
    assert built.has_linage2 and built.linage2 is None
    assert any("bare marker" in w for w in built.warnings)


def test_the_builder_says_when_no_witness_was_sent():
    built = build_patient(_patient(markers={}, smoking="NeverSmoker"))
    assert any("witness" in w for w in built.warnings)
    with_witness = build_patient(_patient())
    assert not any("without any of the markers" in w for w in with_witness.warnings)


# ═══════════════════════════ the router's reader ══════════════════════════════

def test_parse_recognises_only_linage_forms():
    assert parse_linage2_query("!(linage-hazard-patient &self Caller_X)") == ["linage-hazard-patient"]
    assert parse_linage2_query("!(predict-risk-patient &self Patient001)") is None
    assert parse_linage2_query("(linage-nonsense &self X)") is None
    assert set(LINAGE2_FORMS[:7]) == {
        "linage-decomposition-patient", "linage-drivers-patient", "linage-hazard-patient",
        "linage-risk-patient", "linage-counterfactual-patient", "linage-project-risk-patient",
        "linage-scenarios-patient",
    }


def test_the_sexpr_reader_handles_the_shapes_the_layer_emits():
    sx = parse_sexpr('(Contribution HbA1c (years 0.86) Measured (ReadsOut HbA1c (witnessed True)) '
                     '(DrivenBy (DeregulatedNutrientSensing)))')
    assert sx[0] == "Contribution" and sx[2] == ["years", 0.86]
    assert parse_sexpr('(Via ())') == ["Via", []]
    assert parse_sexpr('(x "a string" -1.5e-3)') == ["x", "a string", -0.0015]
    with pytest.raises(ValueError):
        parse_sexpr("(unbalanced")


def test_collect_analysis_dispatches_by_head():
    program = analysis_program("Caller_X", ("SmokingCessation",), with_projections=False)
    assert "linage-risk-patient" not in program
    atoms = [
        "(LinAgeHazard Caller_X AllCauseMortality (delta-years 8.28) (hazard-multiplier 2.09) (confidence 0.54))",
        "(LinAgeCounterfactual Caller_X SmokingCessation (expected-delta-years -2.79) (signed Neg (stv 2.79 0.765)) (Via (SerumCotinine)))",
        "(LinAgeDecomposition Caller_X (delta-years 8.28) (Measured ((Contribution SerumCotinine (years 2.93) Measured "
        "(ReadsOut CurrentTobaccoExposure (witnessed True)) (DrivenBy ())))) (Imputed ()) (attributed-measured 2.93) "
        "(attributed-imputed 0.0) (age-term-residual 5.35))",
    ]
    result = PLNRunResult(status="ok", results=[PLNAtomResult(a, None) for a in atoms], mode="runtime")
    a = collect_analysis(program, result)
    assert a.hazard["hazard_multiplier"] == 2.09
    assert a.counterfactuals[0]["via"] == ["SerumCotinine"] and a.counterfactuals[0]["confidence"] == 0.765
    assert a.decomposition["measured"][0]["witnessed"] is True
    assert a.decomposition["age_term_residual_years"] == 5.35
    assert a.unparsed == []


def test_analysis_program_refuses_a_lever_that_is_not_a_symbol():
    with pytest.raises(ValueError):
        analysis_program("Caller_X", ("Evil) (= (x) 1",))


def test_a_form_for_a_patient_with_no_delta_is_warned():
    assert linage2_form_warnings("!(linage-hazard-patient &self Patient001)")
    assert not linage2_form_warnings("!(linage-hazard-patient &self Caller_X)",
                                     extra_atoms="(LinAgeDelta Caller_X 8.28)")
    assert not linage2_form_warnings("!(predict-risk-patient &self Patient001)")


def test_the_shared_space_gets_the_patient_without_its_linage2_atoms():
    """BuiltPatient.shared_atoms: the patient the SHARED space may hold. Everything
    but the LinAge2 atoms, which abort that space on the patient forms."""
    built = build_patient(_patient())
    shared = built.shared_atoms.splitlines()
    assert "(PatientSmoking Caller_W58 CurrentSmoker)" in shared
    assert "(MeasuredZ Caller_W58 HbA1c 1.6)" in shared
    for head in ("LinAgeDelta", "LinAgeContribution", "LinAgeAccel"):
        assert head not in built.shared_atoms
        assert head in built.atoms
    assert set(shared) <= set(built.atoms.splitlines())
    # nothing else is withheld: the difference is exactly the LinAge2 atoms
    withheld = set(built.atoms.splitlines()) - set(shared)
    assert all(("LinAge" in line) for line in withheld) and len(withheld) == 61
    # a patient with no LinAge2 result loses nothing
    plain = build_patient({"id": "P", "age": 50, "sex": "Male", "markers": {"CRP": 1.2}})
    assert plain.shared_atoms == plain.atoms
    # the bare clock marker is LinAge2-only too
    bare = build_patient({"id": "B", "age": 50, "sex": "Male", "markers": {"LinAgeAccel": 0.5}})
    assert "LinAgeAccel" in bare.atoms and "LinAgeAccel" not in bare.shared_atoms


def test_a_mixed_program_is_split_by_space_in_order():
    split = split_linage2_program(
        "!(diagnose-patient &self Caller_W58 (InsulinResistance))\n"
        ";; a comment line\n"
        "!(linage-hazard-patient &self Caller_W58)\n"
        "!(recommend-supplements-patient\n   &self Caller_W58)"
    )
    assert split.mixed and not split.linage2_first
    assert split.linage2 == "!(linage-hazard-patient &self Caller_W58)"
    assert split.generic.splitlines() == [
        "!(diagnose-patient &self Caller_W58 (InsulinResistance))",
        "!(recommend-supplements-patient &self Caller_W58)",
    ]
    pure = split_linage2_program("(linage-hazard-patient &self X)\n(linage-drivers-patient &self X)")
    assert not pure.mixed and pure.linage2_first and pure.generic == ""
    # a LinAge2 form nested inside another expression still runs where its layer is
    nested = split_linage2_program("!(let $h (linage-hazard-patient &self X) $h)\n!(match &self (A $x) $x)")
    assert nested.mixed and nested.linage2.startswith("!(let $h (linage-hazard")


def test_the_split_is_per_expression_not_per_line():
    """Two forms on ONE line are two expressions (the old line-based splitter glued
    them, ran both in the scoped space, and evaluated only the first)."""
    one_line = split_linage2_program(
        "!(linage-hazard-patient &self X) !(recommend-supplements-patient &self X)")
    assert one_line.mixed
    assert one_line.linage2 == "!(linage-hazard-patient &self X)"
    assert one_line.generic == "!(recommend-supplements-patient &self X)"
    # a linage-* name inside a comment is not a LinAge2 form
    commented = split_linage2_program(
        ";; unlike (linage-hazard-patient &self X)\n"
        "!(recommend-supplements-patient &self X) ; vs (linage-drivers-patient\n"
        "!(diagnose-patient &self X (A))")
    assert commented.linage2 == "" and len(commented.generic.splitlines()) == 2
    # interleaved: every expression remembers where it sat
    inter = split_linage2_program("(linage-hazard-patient &self X)\n(predict-risk-patient &self X)\n"
                                  "(linage-drivers-patient &self X)")
    assert inter.positions() == [[0, 2], [1]] and inter.linage2_first


def test_a_nested_combination_is_named_not_silently_run():
    from core.linage2_router import nesting_warnings
    split = split_linage2_program("!(let $h (linage-hazard-patient &self X) (recommend-supplements-patient &self X))")
    assert not split.mixed and split.nested
    warnings = nesting_warnings(split)
    assert warnings and "recommend-supplements-patient" in warnings[0]
    assert not split_linage2_program("!(let $h (linage-hazard-patient &self X) $h)").nested


def test_pieces_come_back_in_program_order():
    lin = PLNRunResult(status="ok", mode="runtime", results=[
        PLNAtomResult(atom="(L0)", expr_index=0), PLNAtomResult(atom="(L2)", expr_index=1)])
    gen = PLNRunResult(status="ok", mode="runtime", results=[PLNAtomResult(atom="(G1)", expr_index=0)])
    merged = merge_run_results([lin, gen], [[0, 2], [1]])
    assert [r.atom for r in merged.results] == ["(L0)", "(G1)", "(L2)"]


def test_issues_from_two_spaces_say_which_space():
    v = merge_validation_results(
        ValidationResult(valid=False, issues=["Unbalanced parentheses in MeTTa query."]),
        ValidationResult(valid=False, issues=["Unbalanced parentheses in MeTTa query.", "x"]),
        labels=("LinAge2 space", "shared space"),
    )
    assert v.issues == ["[LinAge2 space] Unbalanced parentheses in MeTTa query.",
                        "[shared space] Unbalanced parentheses in MeTTa query.", "[shared space] x"]


def test_merged_answers_keep_order_and_an_error_is_never_half_an_answer():
    a = PLNRunResult(status="ok", results=[PLNAtomResult(atom="(A)")], query_time_ms=5, mode="runtime")
    b = PLNRunResult(status="empty", results=[], query_time_ms=7, mode="runtime")
    c = PLNRunResult(status="ok", results=[PLNAtomResult(atom="(C)")], query_time_ms=1, mode="runtime")
    merged = merge_run_results([c, b, a])
    assert [r.atom for r in merged.results] == ["(C)", "(A)"] and merged.status == "ok"
    assert merged.query_time_ms == 13
    assert merge_run_results([b, b]).status == "empty"
    boom = PLNRunResult(status="error", error="aborted", mode="runtime")
    assert merge_run_results([a, boom]) is boom
    v = merge_validation_results(
        ValidationResult(valid=True, warnings=["w1"], ungrounded_predicates=["P"]),
        ValidationResult(valid=False, issues=["i1"], ungrounded_predicates=["P", "Q"]),
    )
    assert not v.valid and v.issues == ["i1"] and v.warnings == ["w1"]
    assert v.ungrounded_predicates == ["P", "Q"]


# ═══════════════════════════ the scoped space ════════════════════════════════

def test_the_layer_is_absent_from_the_shared_execution_space():
    """Regression guard, asserted on the lists: measured, 60 typing facts for these
    inputs appended to the shared space abort hyperon on decompose-grimage."""
    import api as api_module
    runtime = {p.name for p in api_module._runtime_kb_paths()}
    assert not (set(LINAGE2_LAYER_FILES) & runtime)
    assert set(LINAGE2_LAYER_FILES) & {p.name for p in LINAGE2_PATIENT_STACK} == set(LINAGE2_LAYER_FILES)
    assert all(p.exists() for p in LINAGE2_PATIENT_STACK)
    # not the NHANES patient stack plus LinAge2: those four cost the whole margin
    names = {p.name for p in LINAGE2_PATIENT_STACK}
    for absent in ("pln_intervention_ranking.metta", "pln_abductive_diagnosis.metta",
                   "hallmarks_lopezotin2023_intervention_evidence.metta", "nhanes_reference.metta"):
        assert absent not in names
    assert linage2_patient_kb(REPO / "build" / "definitely-not-here.metta") == list(LINAGE2_PATIENT_STACK) \
        or (REPO / "build" / "nhanes_mortality_baseline.metta").exists()


hyperon = pytest.importorskip("hyperon", reason="the MeTTa tests need the engine")
from hyperon import MeTTa  # noqa: E402


def _space(extra: str, *more_files: Path) -> MeTTa:
    text = "\n".join(p.read_text(encoding="utf-8") for p in [*LINAGE2_PATIENT_STACK, *more_files])
    m = MeTTa()
    m.run(text + "\n" + extra)
    return m


def _one(m: MeTTa, query: str) -> str:
    res = m.run(query)
    assert res and res[0], f"no result for {query}: {res}"
    assert len(res[0]) == 1, f"expected one result for {query}: {res}"
    return str(res[0][0])


def _empty(m: MeTTa, query: str) -> bool:
    res = m.run(query)
    return res == [[]] or all(len(g) == 0 for g in res)


def _num(text: str, field: str) -> float:
    m = re.search(r"\(" + re.escape(field) + r"\s+([-\d.eE]+)\)", text)
    assert m, f"{field} not in {text[:200]}"
    return float(m.group(1))


@pytest.fixture(scope="module")
def smoker() -> tuple[MeTTa, dict]:
    built = build_patient(_patient())
    return _space(built.atoms), built.linage2.as_dict()


@pytest.fixture(scope="module")
def never_smoker() -> MeTTa:
    built = build_patient(_patient(id="N58", smoking="NeverSmoker"))
    return _space(built.atoms)


def test_the_decomposition_totals_reproduce_the_delta_exactly(smoker):
    m, lin = smoker
    out = _one(m, "!(linage-decomposition-patient &self Caller_W58)")
    delta = _num(out, "delta-years")
    am, ai, resid = _num(out, "attributed-measured"), _num(out, "attributed-imputed"), _num(out, "age-term-residual")
    assert delta == pytest.approx(lin["delta_years"], abs=1e-5)
    assert am == pytest.approx(lin["attributed_measured_years"], abs=1e-5)
    assert ai == pytest.approx(lin["attributed_imputed_years"], abs=1e-5)
    assert am + ai + resid == pytest.approx(delta, abs=1e-6)
    assert resid > 0.4, "the fixture's age term is ~0.5 y and must be reported, not absorbed"
    assert out.count("(Contribution ") == 59
    assert out.count(" Measured ") == 33 and out.count(" Imputed ") == 26


def test_a_cause_is_credited_only_under_a_witness(smoker):
    m, _ = smoker
    out = _one(m, "!(linage-decomposition-patient &self Caller_W58)")
    # HbA1c: measured, positive, patient z 1.6 (Elevated) -> credited to the metabolic hallmark
    assert re.search(r"\(Contribution HbA1c \(years 0\.86\d*\) Measured \(ReadsOut HbA1c \(witnessed True\)\) "
                     r"\(DrivenBy \(DeregulatedNutrientSensing\)\)\)", out), out
    # CRP: measured but the patient's own CRP z is 0.3 -> witnessed False -> no cause, whatever the years say
    assert re.search(r"\(Contribution CRP \(years [-\d.]+\) Measured \(ReadsOut CRP \(witnessed False\)\) \(DrivenBy \(\)\)\)", out)
    # cotinine: the witness is the smoking status
    assert "(ReadsOut CurrentTobaccoExposure (witnessed True))" in out
    # glucose is measured and FastingGlucose has a bridge, but no glucose z was sent -> no witness -> no cause
    assert re.search(r"\(Contribution SerumGlucose \(years [-\d.]+\) Measured \(ReadsOut FastingGlucose \(witnessed False\)\) \(DrivenBy \(\)\)\)", out)
    # an imputed input is never credited, whatever its readout: all 26 imputed records have empty causes
    imputed_block = out.split("(Imputed (")[1]
    assert "(DrivenBy (" in imputed_block and not re.search(r"\(DrivenBy \([A-Za-z]", imputed_block)
    # albumin reads out the LowSerumAlbumin deficit, but this patient sent no z for it: no witness, no cause
    assert re.search(r"\(Contribution SerumAlbumin \(years 2\.26\d*\) Measured \(ReadsOut LowSerumAlbumin \(witnessed False\)\) "
                     r"\(DrivenBy \(\)\)\)", out)
    # an input with no readout is carried, unexplained
    assert re.search(r"\(Contribution SerumCreatinine \(years [-\d.]+\) Measured \(ReadsOut None\) \(DrivenBy \(\)\)\)", out)


def test_the_witness_reads_the_patients_own_values(smoker):
    m, _ = smoker
    assert _one(m, "!(linage-witnessed &self Caller_W58 HbA1c)") == "True"
    assert _one(m, "!(linage-witnessed &self Caller_W58 CRP)") == "False"
    assert _one(m, "!(linage-witnessed &self Caller_W58 FastingGlucose)") == "False"   # no z sent
    assert _one(m, "!(linage-witnessed &self Caller_W58 CurrentTobaccoExposure)") == "True"
    assert _one(m, "!(linage-biomarker-causes &self CRP)") == "(ChronicInflammation CellularSenescence)"
    assert _one(m, "!(linage-biomarker-causes &self HbA1c)") == "(DeregulatedNutrientSensing)"


def test_the_hazard_is_the_fong_2025_hr_to_the_power_of_the_delta(smoker):
    m, lin = smoker
    out = _one(m, "!(linage-hazard-patient &self Caller_W58)")
    assert out.startswith("(LinAgeHazard Caller_W58 AllCauseMortality ")
    assert _num(out, "hazard-multiplier") == pytest.approx(HAZARD_PER_YEAR ** lin["delta_years"], rel=1e-6)
    assert _num(out, "confidence") == pytest.approx(EVIDENCE_CONFIDENCE * RISK_DISCOUNT)


def test_the_absolute_risk_yields_nothing_without_a_baseline(smoker):
    m, _ = smoker
    assert _empty(m, "!(linage-risk-patient &self Caller_W58)")
    assert _empty(m, "!(linage-project-risk-patient &self Caller_W58 SmokingCessation)")
    # and nothing for an outcome the Fong 2025 record does not cover
    assert _empty(m, "!(linage-hazard &self Caller_W58 CoronaryHeartDisease)")


def test_the_smoking_counterfactual_is_gated_on_being_a_current_smoker(smoker, never_smoker):
    m, lin = smoker
    out = _one(m, "!(linage-counterfactual-patient &self Caller_W58 SmokingCessation)")
    cotinine = next(c["years"] for c in lin["contributions"] if c["symbol"] == "SerumCotinine")
    assert _num(out, "expected-delta-years") == pytest.approx(-0.95 * cotinine, rel=1e-6)
    assert "(Via (SerumCotinine))" in out
    # 0.85 (MultipleHumanTrials) x 1.0 (identity transmission) x 0.9 (chain-discount)
    assert re.search(r"\(stv [\d.]+ 0\.765\)", out), out
    never = _one(never_smoker, "!(linage-counterfactual-patient &self Caller_N58 SmokingCessation)")
    assert _num(never, "expected-delta-years") == 0.0 and "(Via ())" in never


def test_metformin_reaches_the_witnessed_input_only(smoker):
    m, lin = smoker
    out = _one(m, "!(linage-counterfactual-patient &self Caller_W58 Metformin)")
    hba1c = next(c["years"] for c in lin["contributions"] if c["symbol"] == "HbA1c")
    # Metformin -| InsulinResistance (0.75) -> HbA1c (0.70): reduction = 0.75 x 0.70 x years
    assert _num(out, "expected-delta-years") == pytest.approx(-0.75 * 0.70 * hba1c, rel=1e-6)
    assert "(Via (HbA1c))" in out                      # SerumGlucose: measured, but no glucose z -> unwitnessed


def test_an_edgeless_lever_is_omitted_and_a_pathless_one_is_zero(smoker):
    m, _ = smoker
    assert _empty(m, "!(linage-counterfactual-patient &self Caller_W58 Elamipretide)")
    out = _one(m, "!(linage-counterfactual-patient &self Caller_W58 CellularSenescence)")
    assert _num(out, "expected-delta-years") == 0.0 and "(Via ())" in out


def test_the_scenarios_return_one_counterfactual_per_standing_lever(smoker):
    m, _ = smoker
    out = _one(m, "!(linage-scenarios-patient &self Caller_W58)")
    for lever in DEFAULT_LEVERS:
        assert f"(LinAgeCounterfactual Caller_W58 {lever} " in out


def test_the_grimage_layer_is_unchanged_inside_the_linage2_stack(smoker):
    m, _ = smoker
    risk = _one(m, "!(predict-risk-patient &self Patient001)")
    assert _num(risk, "point") == pytest.approx(0.12605, abs=1e-5)          # docs/risk_prediction.md §4
    cf = _one(m, "!(counterfactual-patient &self Patient003 SmokingCessation)")
    assert _num(cf, "expected-delta") == pytest.approx(-0.0819, abs=1e-4)    # lifestyle_evidence.metta
    assert _one(m, "!(resolve-lever &self SmokingCessation)") == "SmokingPackYears"


def _baseline_records() -> str:
    """A full Schema-B all-cause baseline set, the shape nhanes_mortality_etl.py emits.
    SYNTHETIC values, arbitrary by construction."""
    out = []
    for sex in ("Male", "Female"):
        for band, risk in (("Age_lt50", 0.03), ("Age_50_59", 0.0812), ("Age_60_69", 0.17), ("Age_70p", 0.40)):
            rid = f"NB_ACM_{sex}_{band}"
            out.append(
                f"(: {rid} BaselineRiskRecord)(BaseOutcome {rid} AllCauseMortality)(BaseSex {rid} {sex})"
                f"(BaseAgeBand {rid} {band})(BaseHorizonMonths {rid} 120)(BaseRisk {rid} {risk})"
                f"(BaseEstimator {rid} WeightedKaplanMeier)(BaseUnweightedN {rid} 500)(BaseEvents {rid} 60)"
                f'(BaseWeightVariable {rid} "WTMEC4YR")(BaseTimeVariable {rid} "PERMTH_EXM")'
                f'(BaseSourceCycles {rid} "1999-2002")(BaseLinkageVintage {rid} "2019")'
                f"(BaseProvenance {rid} NHANES_Microdata)"
            )
    return "\n".join(out)


def test_with_a_data_backed_baseline_the_risk_is_the_survival_form():
    built = build_patient(_patient())
    m = _space(built.atoms + "\n" + _baseline_records())
    out = _one(m, "!(linage-risk-patient &self Caller_W58)")
    mult = HAZARD_PER_YEAR ** built.linage2.delta_years
    expected = 1.0 - (1.0 - 0.0812) ** mult
    assert _num(out, "baseline") == pytest.approx(0.0812)
    assert _num(out, "multiplier") == pytest.approx(mult, rel=1e-6)
    assert _num(out, "point") == pytest.approx(expected, rel=1e-6)
    assert "(clock LinAge2)" in out
    lo, hi = map(float, re.search(r"\(ci ([-\d.eE]+) ([-\d.eE]+)\)", out).groups())
    assert lo < expected < hi <= 1.0
    proj = _one(m, "!(linage-project-risk-patient &self Caller_W58 SmokingCessation)")
    assert _num(proj, "reduction") > 0 and _num(proj, "point") < expected
    assert "(Via (SerumCotinine))" in proj


@pytest.mark.slow
def test_the_scoped_stack_keeps_a_head_symbol_margin():
    """The budget that shaped this layer. hyperon 0.2.10 aborts the PROCESS (a
    non-unwinding Rust panic) when a space carries too many distinct head symbols,
    so the probe runs in a subprocess and reads the return code. With the 59
    NHANES-code string facts loaded the stack had NO margin; without them, and on
    the minimal stack, it takes 16 new heads plus a full generated baseline. If this
    fails because the margin SHRANK, something was added to LINAGE2_PATIENT_STACK or
    to a file in it; if it fails at n=0, the layer does not load at all."""
    probe = "\n".join([
        "import sys, json",
        f"sys.path.insert(0, {str(PLN_CHAT)!r}); sys.path.insert(0, {str(REPO / 'tests')!r})",
        "from core.pln_runner import LINAGE2_PATIENT_STACK, run_query",
        "from core.patient_builder import build_patient",
        "import test_linage2 as t",
        "n = int(sys.argv[1])",
        "built = build_patient(t._patient())",
        r'pad = "\n".join(f"(ProbeHead{i} ProbeSym{i} \"probe string {i}\")" for i in range(n))',
        "extra = built.atoms + '\\n' + pad + '\\n' + t._baseline_records()",
        "for q in ('!(linage-decomposition-patient &self Caller_W58)',",
        "          '!(linage-counterfactual-patient &self Caller_W58 SmokingCessation)',",
        "          '!(linage-risk-patient &self Caller_W58)'):",
        "    r = run_query(q, kb_files=list(LINAGE2_PATIENT_STACK), extra_atoms=extra)",
        "    assert r.status == 'ok', (q, r.status, r.error)",
        "print('ok')",
    ])

    def survives(n: int) -> tuple[bool, str]:
        done = subprocess.run(
            [sys.executable, "-c", probe, str(n)],
            capture_output=True, text=True, timeout=600, cwd=str(REPO),
        )
        return done.returncode == 0, done.stderr[-600:]

    ok, err = survives(0)
    assert ok, f"the LinAge2 stack does not load at all: {err}"
    ok, err = survives(16)
    assert ok, (
        "the LinAge2 scoped stack no longer tolerates 16 extra head symbols on top of a "
        f"generated baseline; something spent the margin: {err}"
    )


@pytest.mark.slow
def test_the_shared_space_survives_a_patient_with_a_linage2_result():
    """The crash this split exists for, measured in a subprocess because hyperon
    0.2.10 aborts the PROCESS: with every atom of a LinAge2 patient in the shared
    space, diagnose-patient, predict-risk-patient and recommend-supplements-patient
    abort; with `shared_atoms` all three answer. The control run is what makes the
    first assertion mean something — if it stops aborting, the engine changed and
    the split may no longer be needed (it is still correct)."""
    probe = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(PLN_CHAT)!r}); sys.path.insert(0, {str(REPO / 'tests')!r})",
        "import api",
        "from core.patient_builder import build_patient",
        "from core.pln_runner import run_query",
        "import test_linage2 as t",
        "built = build_patient(t._patient(markers={'HbA1c': 1.6, 'CRP': 0.3, 'AgeAccelGrim': 1.6}))",
        "atoms = built.shared_atoms if sys.argv[1] == 'shared' else built.atoms",
        "kb = api._runtime_kb_paths()",
        "for q in ('!(diagnose-patient &self Caller_W58 (CellularSenescence ChronicInflammation InsulinResistance))',",
        "          '!(predict-risk-patient &self Caller_W58)',",
        "          '!(recommend-supplements-patient &self Caller_W58)'):",
        "    if sys.argv[2] and sys.argv[2] not in q: continue",
        "    r = run_query(q, kb_files=kb, extra_atoms=atoms)",
        "    assert r.status == 'ok', (q, r.status, r.error)",
        "print('ok')",
    ])

    def run(which: str, only: str = "") -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", probe, which, only],
            capture_output=True, text=True, timeout=900, cwd=str(REPO),
        )

    shared = run("shared")
    assert shared.returncode == 0 and "ok" in shared.stdout, shared.stderr[-800:]
    # The control, per form: each one must die of hyperon's ABORT (SIGABRT, a
    # non-unwinding panic) with the full atoms — not of any other error, which
    # would make this pass for the wrong reason.
    for form in ("diagnose-patient", "predict-risk-patient", "recommend-supplements-patient"):
        full = run("full", form)
        assert full.returncode == -6 and "panic" in full.stderr, (
            f"{form} no longer aborts the shared space with a LinAge2 patient's full "
            f"atoms (rc={full.returncode}); the control is void, so this test no longer "
            f"proves the split is what saves it. stderr: {full.stderr[-400:]}"
        )


# ═══════════════════════════ HTTP ════════════════════════════════════════════

httpx = pytest.importorskip("httpx")
import api as api_module  # noqa: E402
import core.executor as executor_module  # noqa: E402
from core.llm_translator import TranslationResult  # noqa: E402


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(api_module, "log_turn", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)


def _request(method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver",
                                     timeout=300) as c:
            return await c.request(method, path, **kwargs)
    return asyncio.run(send())


def test_features_endpoint_publishes_the_vocabulary():
    body = _request("GET", "/linage2/features").json()
    assert len(body["features"]) == 59
    assert {f["symbol"] for f in body["features"] if f["reads_out"]} == {
        "CRP", "HbA1c", "SerumGlucose", "SerumCotinine", "RedCellDistributionWidth", "SerumAlbumin"}
    assert all(f["reads_out"] for f in body["features"][:6])       # explainable inputs listed first
    assert body["clock"]["outcome"] == "AllCauseMortality"
    assert "pln_linage2.metta" in body["stack"] and body["forms"][0] == "linage-decomposition-patient"
    markers = _request("GET", "/patients/markers").json()
    assert any(m["marker"] == "LinAgeAccel" for m in markers["markers"])
    assert markers["linage2"]["field"] == "patient.linage2"


def test_preview_reports_the_block_and_the_clock_marker():
    body = _request("POST", "/patients/preview", json=_patient()).json()
    assert body["has_linage2"] and body["linage2"]["measured_count"] == 33
    clock = [m for m in body["markers"] if m["marker"] == "LinAgeAccel"][0]
    assert clock["derived"] and "linage-sd-to-years" in clock["formula"]
    assert body["can_predict_risk"] is False
    bad = _patient(); bad["linage2"]["metadata"]["feature_contributions"][0]["feature"] = "LBXFOO"
    r = _request("POST", "/patients/preview", json=bad)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_linage2_feature"
    r = _request("POST", "/patients/preview", json={**_patient(), "linage2": {"biological_age": 1, "stray": 2}})
    assert r.status_code == 422                         # extra="forbid" on the block too


def test_metta_run_routes_a_linage_form_to_the_scoped_space_and_warns_without_a_delta():
    r = _request("POST", "/metta/run", json={
        "metta_query": "!(linage-hazard-patient &self Caller_W58)", "patient": _patient()})
    body = r.json()
    assert r.status_code == 200 and body["routed"] == "linage2" and body["pln_status"] == "ok"
    assert body["pln_results"][0]["atom"].startswith("(LinAgeHazard Caller_W58 AllCauseMortality")
    # the generic path is untouched
    body = _request("POST", "/metta/run", json={"metta_query": "!(predict-risk-patient &self Patient001)"}).json()
    assert body["routed"] is None and body["pln_status"] == "ok"
    # a curated patient has no LinAge2 result: empty, and SAID to be empty for that reason
    body = _request("POST", "/metta/run", json={"metta_query": "!(linage-hazard-patient &self Patient001)"}).json()
    assert body["routed"] == "linage2" and body["pln_status"] == "empty"
    assert any("no LinAge2 result" in w for w in body["warnings"])


@pytest.mark.slow
@pytest.mark.parametrize("program", [
    "!(diagnose-patient &self Caller_W58 (InsulinResistance ChronicInflammation))",
    "!(predict-risk-patient &self Caller_W58)",
    "!(recommend-supplements-patient &self Caller_W58)",
])
def test_a_plain_question_about_a_linage2_patient_is_answered(monkeypatch, program):
    """THE crash, end to end: a non-LinAge2 question about a patient who carries a
    LinAge2 result used to abort the worker (500 pln_worker_crashed). Through the
    worker pool, so a regression is a 500, not a dead pytest."""
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 2)
    patient = _patient(markers={"HbA1c": 1.6, "CRP": 0.3, "AgeAccelGrim": 1.6})
    r = _request("POST", "/metta/run", json={"metta_query": program, "patient": patient})
    body = r.json()
    assert r.status_code == 200, body
    assert body["routed"] is None and body["pln_status"] == "ok"
    assert body["validation_valid"] is True


@pytest.mark.slow
def test_a_mixed_program_runs_each_part_in_its_own_space(monkeypatch):
    """"My LinAge2 hazard AND my differential": the LinAge2 form in the scoped space
    with every atom, the diagnosis in the shared space without the LinAge2 atoms.
    Run through the worker pool so a regression is a 500, not a dead pytest."""
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 2)
    program = ("!(linage-hazard-patient &self Caller_W58)\n"
               "!(diagnose-patient &self Caller_W58 (InsulinResistance ChronicInflammation))")
    r = _request("POST", "/metta/run", json={"metta_query": program, "patient": _patient()})
    body = r.json()
    assert r.status_code == 200, body
    assert body["routed"] == "linage2+generic" and body["pln_status"] == "ok"
    atoms = [x["atom"] for x in body["pln_results"]]
    assert atoms[0].startswith("(LinAgeHazard Caller_W58")          # program order kept
    assert any("Hypothesis InsulinResistance" in a for a in atoms[1:])
    # the shared part is validated against the shared space: a LinAge2-only symbol
    # used outside a linage-* form is refused there, before anything runs
    bad = _request("POST", "/metta/run", json={
        "metta_query": "!(linage-hazard-patient &self Caller_W58)\n!(match &self (LinAgeDelta Caller_W58 $d) $d)",
        "patient": _patient()})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_metta_query"


@pytest.mark.slow
def test_an_interleaved_program_answers_in_order_with_one_offload(monkeypatch):
    """L, G, L: the risk answer stays between the two LinAge2 answers, and the whole
    program is ONE offloaded task — one worker, one deadline, one admission."""
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 2)
    calls = []
    real = api_module.run_offloaded

    def spy(task, kwargs, inline, **kw):
        calls.append(task)
        return real(task, kwargs, inline, **kw)

    monkeypatch.setattr(api_module, "run_offloaded", spy)
    patient = _patient(markers={"HbA1c": 1.6, "CRP": 0.3, "AgeAccelGrim": 1.6})
    program = ("!(linage-hazard-patient &self Caller_W58) !(predict-risk-patient &self Caller_W58)\n"
               "!(linage-delta &self Caller_W58)")
    r = _request("POST", "/metta/run", json={"metta_query": program, "patient": patient})
    body = r.json()
    assert r.status_code == 200, body
    assert calls == ["run_query_parts"]
    heads = [x["atom"].split()[0] for x in body["pln_results"]]
    assert heads[0] == "(LinAgeHazard" and heads[1] == "(RiskPrediction" and len(heads) == 3


@pytest.mark.slow
def test_query_runs_a_mixed_translation_in_both_spaces(monkeypatch):
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 2)
    translation = TranslationResult(
        metta_query="(recommend-supplements-patient &self Caller_W58)\n(linage-drivers-patient &self Caller_W58)",
        explanation="supplements, then LinAge2 drivers", intent="inference",
        requires_pln_inference=True, confidence_filter=0.0,
    )
    monkeypatch.setattr(api_module, "translate", Mock(return_value=translation))
    monkeypatch.setattr(api_module, "build_system_prompt", lambda registry, raw, inventory=None: "prompt")
    r = _request("POST", "/query", json={"message": "supplements and LinAge2 drivers", "patient": _patient()})
    body = r.json()
    assert r.status_code == 200, body
    assert body["routed"] == "linage2+generic" and body["pln_status"] == "ok"
    assert "SupplementRecommendation Caller_W58" in body["answer"]
    assert "(Contribution SerumCotinine" in body["answer"]          # the LinAge2 drivers
    # both halves validated against the space they run in, patient atoms included:
    # the caller's own id is not "not found in loaded ontology"
    assert body["validation_valid"] is True, body["validation_issues"]


def test_the_chat_splits_a_mixed_program_like_the_api(monkeypatch):
    """Parity, behaviourally: the Gradio chat sends a mixed program as ONE
    run_query_parts task, LinAge2 part to the scoped stack, and answers in order."""
    import app as app_module
    translation = TranslationResult(
        metta_query="(predict-risk-patient &self Patient001)\n(linage-hazard-patient &self Patient001)",
        explanation="", intent="inference", requires_pln_inference=True, confidence_filter=0.0)
    monkeypatch.setattr(app_module, "translate", lambda **kw: translation)
    monkeypatch.setattr(app_module, "log_query", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "log_turn", lambda *a, **k: None)
    calls = []

    def fake(task, kwargs, inline, **kw):
        calls.append((task, kwargs))
        return [PLNRunResult(status="ok", mode="runtime", results=[PLNAtomResult("(LIN)", expr_index=0)]),
                PLNRunResult(status="ok", mode="runtime", results=[PLNAtomResult("(GEN)", expr_index=0)])]

    monkeypatch.setattr(app_module, "run_offloaded", fake)
    history, _ = app_module.chat(
        user_message="q", history=[], selected_files=[], model="m", temperature=0.0,
        confidence_threshold=0.0, show_metta=False, show_explanation=False, show_debug=False)
    assert [task for task, _ in calls] == ["run_query_parts"]
    parts = calls[0][1]["parts"]
    assert parts[0]["metta_query"].startswith("(linage-hazard-patient")
    assert parts[0]["kb_files"] == linage2_patient_kb()
    assert parts[1]["metta_query"].startswith("(predict-risk-patient")
    answer = history[-1]["content"]
    assert answer.index("(GEN)") < answer.index("(LIN)")             # program order


def test_a_generic_question_about_a_caller_patient_validates_clean(monkeypatch):
    """Pre-existing: /query validated the shared space WITHOUT the patient's atoms,
    so every answer about Caller_<id> carried 'Symbol(s) not found: Caller_<id>'."""
    translation = TranslationResult(
        metta_query="(diagnose-patient &self Caller_W58 (InsulinResistance))",
        explanation="", intent="inference", requires_pln_inference=True, confidence_filter=0.0)
    monkeypatch.setattr(api_module, "translate", Mock(return_value=translation))
    monkeypatch.setattr(api_module, "build_system_prompt", lambda registry, raw, inventory=None: "prompt")
    monkeypatch.setattr(api_module, "_offloaded_run",
                        lambda *a, **k: PLNRunResult(status="empty", mode="runtime"))
    r = _request("POST", "/query", json={"message": "q", "patient": _patient()})
    body = r.json()
    assert r.status_code == 200 and body["validation_valid"] is True, body["validation_issues"]


def test_query_routes_a_translated_linage_form(monkeypatch):
    translation = TranslationResult(
        metta_query="(linage-hazard-patient &self Caller_W58)",
        explanation="the LinAge2 hazard", intent="inference",
        requires_pln_inference=True, confidence_filter=0.0,
    )
    monkeypatch.setattr(api_module, "translate", Mock(return_value=translation))
    monkeypatch.setattr(api_module, "build_system_prompt", lambda registry, raw, inventory=None: "prompt")
    r = _request("POST", "/query", json={"message": "how much does LinAge2 raise my risk?", "patient": _patient()})
    body = r.json()
    assert r.status_code == 200, body
    assert body["routed"] == "linage2" and body["pln_status"] == "ok"
    assert body["validation_valid"] is True
    assert "(LinAgeHazard Caller_W58" in body["answer"]


def test_the_prompt_tells_the_translator_about_the_forms_only_when_there_is_a_delta(monkeypatch):
    seen: dict = {}

    def fake_translate(**kwargs):
        seen["prompt"] = kwargs["system_prompt"]
        return TranslationResult(metta_query="(linage-hazard-patient &self Caller_W58)",
                                 explanation="", intent="inference",
                                 requires_pln_inference=True, confidence_filter=0.0)

    monkeypatch.setattr(api_module, "translate", fake_translate)
    monkeypatch.setattr(api_module, "build_system_prompt", lambda registry, raw, inventory=None: "STATIC")
    _request("POST", "/query", json={"message": "q", "patient": _patient()})
    assert "linage-decomposition-patient &self Caller_W58" in seen["prompt"]
    assert seen["prompt"].startswith("STATIC")          # the static prefix stays cacheable
    _request("POST", "/query", json={"message": "q", "patient": {"id": "P", "age": 50, "sex": "Male",
                                                                  "markers": {"CRP": 1.2}}})
    assert "linage-decomposition-patient" not in seen["prompt"]


def test_analyze_returns_the_whole_picture_as_json():
    r = _request("POST", "/linage2/analyze", json={**_patient(), "levers": ["SmokingCessation", "Metformin"]})
    body = r.json()
    assert r.status_code == 200, body
    d = body["decomposition"]
    assert d["attributed_measured_years"] + d["attributed_imputed_years"] + d["age_term_residual_years"] == pytest.approx(d["delta_years"], abs=1e-6)
    assert d["measured"][0]["symbol"] == "SerumCotinine" and d["measured"][0]["witnessed"] is True
    hba1c = next(c for c in d["measured"] if c["symbol"] == "HbA1c")
    assert hba1c["driven_by"] == ["DeregulatedNutrientSensing"]
    assert body["hazard"]["hazard_multiplier"] == pytest.approx(HAZARD_PER_YEAR ** d["delta_years"], rel=1e-6)
    assert body["risk"] is None and "No absolute risk" in body["risk_note"]
    cfs = {c["lever"]: c for c in body["counterfactuals"]}
    assert cfs["SmokingCessation"]["via"] == ["SerumCotinine"] and cfs["Metformin"]["via"] == ["HbA1c"]
    assert body["projected_risks"] == [] and body["unparsed"] == []
    assert "linage-risk-patient" not in body["metta_query"]     # skipped: no baseline to multiply
    r = _request("POST", "/linage2/analyze", json={**_patient(), "levers": ["NotAThing"]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_lever"
    r = _request("POST", "/linage2/analyze", json={"id": "Q", "age": 50, "sex": "Female"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "linage2_required"


def test_the_never_smoker_is_told_the_cessation_number_is_not_theirs():
    body = _request("POST", "/linage2/analyze", json={**_patient(smoking="NeverSmoker"),
                                                      "levers": ["SmokingCessation"]}).json()
    cf = body["counterfactuals"][0]
    assert cf["expected_delta_years"] == 0.0 and cf["via"] == []
    assert any("LeverRequiresSmoking" in w for w in body["warnings"])


@pytest.mark.slow
def test_a_heart_risk_pair_for_a_patient_with_no_grimage_is_labelled_all_cause(monkeypatch):
    """"What's my heart risk?" for a patient with a LinAge2 result and no GrimAge value: the CHD
    model has no input (its part of the program returns nothing), the hazard answers, and the
    response says the hazard is the ALL-CAUSE multiplier — never a heart risk, never combined."""
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 2)
    program = "!(predict-risk-patient &self Caller_W58)\n!(linage-hazard-patient &self Caller_W58)"
    body = _request("POST", "/metta/run", json={"metta_query": program, "patient": _patient()}).json()
    assert body["routed"] == "linage2+generic" and body["pln_status"] == "ok"
    atoms = [x["atom"] for x in body["pln_results"]]
    assert len(atoms) == 1 and atoms[0].startswith("(LinAgeHazard Caller_W58 AllCauseMortality")
    (note,) = [w for w in body["warnings"] if "no AgeAccelGrim value" in w]
    assert "ALL-CAUSE mortality multiplier" in note and "never multiplied or added" in note
    # a hazard-only question is the all-cause question it says it is: no heart note
    body = _request("POST", "/metta/run", json={"metta_query": "!(linage-hazard-patient &self Caller_W58)",
                                                "patient": _patient()}).json()
    assert not [w for w in body["warnings"] if "no AgeAccelGrim value" in w]
    # with a GrimAge value the CHD model answers, in program order, and nothing is labelled
    clock = _patient(markers={"HbA1c": 1.6, "CRP": 0.3, "AgeAccelGrim": 1.07143})
    body = _request("POST", "/metta/run", json={"metta_query": program, "patient": clock}).json()
    atoms = [x["atom"] for x in body["pln_results"]]
    assert atoms[0].startswith("(RiskPrediction Caller_W58 CoronaryHeartDisease") and atoms[1].startswith("(LinAgeHazard")
    assert not [w for w in body["warnings"] if "no AgeAccelGrim value" in w]


# ═══════════ RDW and low albumin: two more readouts of chronic inflammation (#7) ═══════════

@pytest.fixture(scope="module")
def inflamed() -> MeTTa:
    """The fixture patient with an RDW z of 2.7 and an albumin DEFICIT z of 1.7 (the tab smoker's), CRP normal."""
    built = build_patient(_patient(id="R58", markers={"HbA1c": 1.6, "CRP": 0.3, "RDW": 2.7, "LowSerumAlbumin": 1.7}))
    assert built.witnesses == ["HbA1c", "LowSerumAlbumin", "RDW"]
    return _space(built.atoms)


def test_an_elevated_rdw_and_a_low_albumin_are_credited_to_inflammation_and_senescence(inflamed):
    out = _one(inflamed, "!(linage-decomposition-patient &self Caller_R58)")
    for symbol, readout in (("RedCellDistributionWidth", "RDW"), ("SerumAlbumin", "LowSerumAlbumin")):
        assert re.search(rf"\(Contribution {symbol} \(years [-\d.]+\) Measured \(ReadsOut {readout} \(witnessed True\)\) "
                         rf"\(DrivenBy \(ChronicInflammation CellularSenescence\)\)\)", out), (symbol, out)
    assert _one(inflamed, "!(linage-biomarker-causes &self RDW)") == "(ChronicInflammation CellularSenescence)"
    assert _one(inflamed, "!(linage-biomarker-causes &self LowSerumAlbumin)") == "(ChronicInflammation CellularSenescence)"


def test_a_positive_contribution_is_not_credited_without_the_patients_own_z(smoker):
    """The fixture patient sent no RDW or albumin z: the years are reported, no cause is credited, however large."""
    m, _ = smoker
    out = _one(m, "!(linage-decomposition-patient &self Caller_W58)")
    assert re.search(r"\(Contribution RedCellDistributionWidth \(years [-\d.]+\) \w+ \(ReadsOut RDW \(witnessed False\)\) "
                     r"\(DrivenBy \(\)\)\)", out)


def test_the_rdw_and_albumin_levers_are_chronic_inflammations_and_share_its_years(inflamed):
    inflammation = _one(inflamed, "!(linage-counterfactual-patient &self Caller_R58 ChronicInflammation)")
    years = _num(inflammation, "expected-delta-years")
    assert years < -1.0 and "(Via (RedCellDistributionWidth SerumAlbumin))" in inflammation
    for lever in ("RDW", "LowSerumAlbumin"):
        got = _one(inflamed, f"!(linage-counterfactual-patient &self Caller_R58 {lever})")
        assert _num(got, "expected-delta-years") == pytest.approx(years)             # one cause, one set of years
        assert "(Via (RedCellDistributionWidth SerumAlbumin))" in got
    # the LinAge2 INPUT's name is not a KB marker: no lever, nothing invented
    assert _empty(inflamed, "!(linage-counterfactual-patient &self Caller_R58 SerumAlbumin)")


def test_the_weaker_evidence_tier_sets_the_confidence_of_the_new_edges():
    """Human observational evidence: Epidemiological (0.60), not the animal-replicated tier of the CRP edge."""
    text = (REPO / "mechanistic_bridges.metta").read_text(encoding="utf-8")
    for node in ("RDW", "LowSerumAlbumin"):
        assert re.search(rf"\(Effect ChronicInflammation {node} Pos\s+\(stv 0\.\d+ \(evidence-confidence Epidemiological\)\)\)", text), node
    assert "(Inheritance RDW Biomarker)" in text and "(Inheritance LowSerumAlbumin Biomarker)" in text
    assert re.search(r"\(Effect InsulinResistance Triglycerides Pos\s+\(stv 0\.50 \(evidence-confidence Epidemiological\)\)\)", text)
    assert re.search(r"\(Effect ChronicInflammation RDW Pos\s+\(stv 0\.50 ", text)
    assert "(Inheritance Triglycerides Biomarker)" in text
    assert "Hypoalbuminemia" not in text.replace(";; The albumin node is the DEFICIT, named LowSerumAlbumin and never \"Hypoalbuminemia\"", "")
