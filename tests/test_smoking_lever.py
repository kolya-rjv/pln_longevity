"""The smoking counterfactual, and the patient it acts on (lifestyle_evidence.metta).

The 2026-09-18 evaluation, §9:

    "the smoking counterfactual for Patient002 returned nothing because no
    smoking -> CHD path is loaded, even though smoking status is in his profile."

That was an honest Gap and a real hole. `PatientSmoking` reached ZERO inference
paths: before this layer it appeared in its type declaration, two patient facts,
one documentation line and a regex in `api.py`, and nothing consumed it.

The repair is deliberately NOT a smoking -> CHD edge. The risk model reads one
predictor, the composite clock, and a parallel smoking term would double-count
an exposure the clock already carries — `DNAmPACKYRS` is a GrimAge component and
is the DNAm surrogate FOR pack-years. So the wiring follows the clock's own
construction:

    SmokingCessation --Neg--> SmokingPackYears --Pos--> DNAmPACKYRS (PartOf GrimAge)

and the counterfactual and risk layers are untouched. These tests assert that the
lever produces a real number, that it produces it through DNAmPACKYRS, that a
patient with no pack-years measurement honestly gets zero, and — the regression
that matters most — that Patient001's and Patient002's answers are byte-identical
with and without this file.

Skipped automatically if `hyperon` is not installed.

Run from the repository root:
    pytest tests/test_smoking_lever.py -q
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

LIFESTYLE = REPO / "lifestyle_evidence.metta"
PROFILE = REPO / "patient_profile.metta"

# Dependency-ordered load, matching the header in each .metta file. The lifestyle
# layer loads last, on top of the whole patient + counterfactual + risk stack.
BASE_FILES = [
    "system_types.metta",
    "logical_predicates.metta",
    "epistemic_calibration.metta",
    "grim_age_core.metta",
    "grim_age_lu2019_evidence.metta",
    "evidence_calibration.metta",
    "hallmarks_core.metta",
    "hallmarks_lopezotin2023_intervention_evidence.metta",
    "mechanistic_bridges.metta",
    "pln_deduction.metta",
    "pln_intervention_ranking.metta",
    "pln_abductive_diagnosis.metta",
    "patient_profile.metta",
    "pln_counterfactual.metta",
    "pln_risk_prediction.metta",
    "supplement_evidence.metta",
    "pln_supplement_recommendation.metta",
]

_NUM = r"([-\d.eE]+)"
_CF_RE = re.compile(
    r"\(Counterfactual\s+(\S+)\s+(\S+)\s+"
    r"\(expected-delta\s+" + _NUM + r"\)\s+"
    r"\(signed\s+(Pos|Neg)\s+\(stv\s+" + _NUM + r"\s+" + _NUM + r"\)\)\s+"
    r"\(Via\s+\(([^)]*)\)\)"
)
_PROJ_RE = re.compile(
    r"\(ProjectedRisk\s+(\S+)\s+(\S+)\s+"
    r"\(point\s+" + _NUM + r"\)\s+\(reduction\s+" + _NUM + r"\)\s+"
    r"\(delta-clock\s+" + _NUM + r"\)\s+\(confidence\s+" + _NUM + r"\)\s+"
    r"\(Via\s+\(([^)]*)\)\)"
)


# ── the file's own contract, checkable without an engine ─────────────────────

def _body(path: Path) -> str:
    """The file with its `;;` commentary removed — only the atoms."""
    return "\n".join(line.split(";;")[0] for line in path.read_text(encoding="utf-8").splitlines())


def _facts(path: Path, head: str) -> list[list[str]]:
    from ontology.inventory import iter_top_level, split_args

    out = []
    for expr in iter_top_level(path.read_text(encoding="utf-8")):
        if not (expr.startswith("(") and expr.endswith(")")):
            continue
        parts = split_args(expr[1:-1].strip())
        if parts and parts[0] == head:
            out.append(parts[1:])
    return out


def test_no_confidence_in_this_layer_is_a_bare_number():
    """Confidence is an (evidence-confidence <Tier>) lookup, never a literal."""
    for args in _facts(LIFESTYLE, "Effect"):
        stv = args[-1]
        assert "(evidence-confidence " in stv, stv
        # ...and the strength beside it IS a literal, because a curated prior is
        # exactly that. This asserts the pairing, not just the presence.
        assert re.match(r"^\(stv\s+[\d.]+\s+\(evidence-confidence\s+\w+\)\)$", stv), stv


def test_the_tiers_used_exist_in_the_single_calibration_authority():
    declared = {
        args[0] for args in _facts(REPO / "epistemic_calibration.metta", ":")
        if len(args) >= 2 and args[1] == "EvidenceCategory"
    }
    used = set(re.findall(r"\(evidence-confidence\s+(\w+)\)", _body(LIFESTYLE)))
    assert used == {"MultipleHumanTrials", "Epidemiological"}
    assert used <= declared


def test_the_two_edges_are_the_route_through_the_clock():
    edges = {(a[0], a[1], a[2]) for a in _facts(LIFESTYLE, "Effect")}
    assert edges == {
        ("SmokingPackYears", "DNAmPACKYRS", "Pos"),
        ("SmokingCessation", "SmokingPackYears", "Neg"),
    }
    # The shortcut this layer refuses: no direct edge into an outcome.
    assert "CoronaryHeartDisease" not in _body(LIFESTYLE)


def test_every_pmid_cited_is_attached_to_a_full_publication_record():
    pmids = {args[1].strip('"') for args in _facts(LIFESTYLE, "PubMedID")}
    assert pmids == {"27651444", "31429895"}          # Joehanes 2016, Duncan 2019
    subjects = {args[0] for args in _facts(LIFESTYLE, "PubMedID")}
    assert subjects == {args[0] for args in _facts(LIFESTYLE, "PublicationTitle")}
    assert subjects == {args[0] for args in _facts(LIFESTYLE, "DOI")}


def test_the_cessation_caveat_is_a_fact_not_only_a_comment():
    limits = {args[0]: args[1] for args in _facts(LIFESTYLE, "Limitation")}
    note = limits["SmokingCessation"]
    assert "remains significantly elevated" in note
    assert "31429895" in note and "27651444" in note


def test_patient002_was_not_edited_to_make_this_demo_work():
    """His numbers are quoted in the evaluation and pinned by another test."""
    profile = _body(PROFILE)
    assert "DNAmPACKYRS" not in profile
    assert "Patient003" not in profile          # the new patient lives elsewhere


# ── both API stacks load it ──────────────────────────────────────────────────

def _stack_literal(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_INFERENCE_STACK":
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_INFERENCE_STACK":
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no _INFERENCE_STACK in {module_path}")


def test_both_inference_stacks_load_the_layer_and_still_agree():
    api_stack = _stack_literal(PLN_CHAT / "api.py")
    app_stack = _stack_literal(PLN_CHAT / "app.py")
    assert "lifestyle_evidence.metta" in api_stack
    assert api_stack == app_stack


def test_the_layer_is_small_enough_to_load_into_a_hyperon_space():
    """hyperon 0.2.10 aborts on a variable-slot match past a few hundred rows."""
    assert LIFESTYLE.stat().st_size < 60_000


# ── a caller's own smoker is told what the lever needs ───────────────────────

def test_a_caller_smoker_without_packyears_is_warned_not_silently_zeroed():
    from core.patient_builder import build_patient

    built = build_patient({
        "id": "Smoker", "age": 64, "sex": "Male", "smoking": "FormerSmoker",
        "markers": {"AgeAccelGrim": 1.6},
    })
    assert any("DNAmPACKYRS" in w for w in built.warnings)
    assert any("not 'quitting would not help'" in w for w in built.warnings)


def test_a_caller_smoker_with_packyears_gets_no_such_warning():
    from core.patient_builder import build_patient

    built = build_patient({
        "id": "Smoker2", "age": 64, "sex": "Male", "smoking": "FormerSmoker",
        "markers": {"AgeAccelGrim": 1.6, "DNAmPACKYRS": 2.1},
    })
    assert not any("DNAmPACKYRS" in w for w in built.warnings)
    assert "(MeasuredZ Caller_Smoker2 DNAmPACKYRS 2.1)" in built.atoms


def test_the_marker_catalog_says_what_packyears_now_reaches():
    from core.patient_builder import marker_catalog

    entry = next(e for e in marker_catalog() if e["marker"] == "DNAmPACKYRS")
    assert "SmokingCessation" in entry["reaches"]


# ── and the engine agrees (real MeTTa) ───────────────────────────────────────

hyperon = pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402


def _space(files: list[str]) -> MeTTa:
    m = MeTTa()
    m.run("\n".join((REPO / f).read_text(encoding="utf-8") for f in files))
    return m


@pytest.fixture(scope="module")
def before() -> MeTTa:
    """The stack WITHOUT the new layer — the regression baseline."""
    return _space(BASE_FILES)


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    """The stack WITH lifestyle_evidence.metta."""
    return _space(BASE_FILES + ["lifestyle_evidence.metta"])


def _run(m: MeTTa, query: str) -> list[str]:
    return sorted(str(atom) for result in m.run(query) for atom in result)


def _one(m: MeTTa, query: str) -> str:
    out = _run(m, query)
    assert len(out) == 1, f"{query} -> {out}"
    return out[0]


@pytest.mark.slow
def test_the_lever_returns_a_real_delta_through_the_packyears_component(kb):
    """The Gap closed: a smoking counterfactual with a number in it."""
    m = _CF_RE.search(_one(kb, "!(counterfactual-patient &self Patient003 SmokingCessation)"))
    assert m, "no Counterfactual atom"
    lever, outcome, delta, sign, strength, conf, via = m.groups()
    assert (lever, outcome, sign) == ("SmokingCessation", "AgeAccelGrim", "Neg")
    # 0.125 (grimage-weight) x 0.80 (the curated prior) x 2.1 (his z) = 0.21
    assert float(delta) == pytest.approx(-0.21)
    assert float(strength) == pytest.approx(0.21)
    # one hop at the MultipleHumanTrials tier
    assert float(conf) == pytest.approx(0.85)
    assert via.split() == ["DNAmPACKYRS"]


@pytest.mark.slow
def test_naming_the_driver_and_naming_the_intervention_agree(kb):
    """resolve-lever routes SmokingCessation to the burden it reduces."""
    as_intervention = _one(kb, "!(counterfactual-patient &self Patient003 SmokingCessation)")
    as_driver = _one(kb, "!(counterfactual-patient &self Patient003 SmokingPackYears)")
    assert as_intervention.replace("SmokingCessation", "X", 1) == \
        as_driver.replace("SmokingPackYears", "X", 1)


@pytest.mark.slow
def test_the_clock_reduction_becomes_an_absolute_risk_reduction(kb):
    m = _PROJ_RE.search(_one(kb, "!(project-risk-patient &self Patient003 SmokingCessation)"))
    assert m, "no ProjectedRisk atom"
    lever, outcome, point, reduction, delta, conf, via = m.groups()
    assert (lever, outcome) == ("SmokingCessation", "CoronaryHeartDisease")
    assert float(delta) == pytest.approx(-0.21)
    assert float(point) == pytest.approx(0.22266, abs=1e-4)
    assert float(reduction) == pytest.approx(0.01369, abs=1e-4)
    assert float(conf) == pytest.approx(0.85)
    assert via.split() == ["DNAmPACKYRS"]
    # The untreated risk, for the comparison the number is only meaningful against.
    assert "(point 0.23634" in _one(kb, "!(predict-risk-patient &self Patient003)")


@pytest.mark.slow
def test_the_decomposition_credits_the_component_and_names_no_hallmark(kb):
    """`component-cause` searches hallmarks; smoking is an EXPOSURE, not one.

    An empty (DrivenBy ()) is the honest answer. Inventing a hallmark to fill it
    is exactly the failure this KB exists to avoid.
    """
    out = _one(kb, "!(decompose-grimage &self Patient003)")
    assert "(Component DNAmPACKYRS (z 2.1) (weight 0.125) (contribution 0.2625) (DrivenBy ()))" in out
    assert "(attributed 0.2625)" in out
    # He presents nothing else elevated, so the rest of the clock is residual.
    assert out.count("(Component ") == 1


@pytest.mark.slow
def test_a_patient_with_no_packyears_measurement_honestly_gets_zero(kb):
    """Patient001 never smoked and has no DNAmPACKYRS — the lever finds nothing."""
    m = _CF_RE.search(_one(kb, "!(counterfactual-patient &self Patient001 SmokingCessation)"))
    _, _, delta, _, strength, conf, via = m.groups()
    assert float(delta) == 0.0 and float(strength) == 0.0 and float(conf) == 0.0
    assert via.strip() == ""


@pytest.mark.slow
def test_the_smoking_patient_is_not_a_senescence_or_metabolic_patient(kb):
    """The contrast that makes the profile legible: the other levers reach nothing."""
    for lever in ("CellularSenescence", "ChronicInflammation", "InsulinResistance"):
        m = _CF_RE.search(_one(kb, f"!(counterfactual-patient &self Patient003 {lever})"))
        assert float(m.group(3)) == 0.0, lever
        assert m.group(7).strip() == "", lever
    assert _one(kb, "!(patient-observations &self Patient003)") == \
        "(AgeAccelGrim DNAmPACKYRS)"


@pytest.mark.slow
def test_the_smoking_axis_still_does_not_reach_chd_through_infer(kb):
    """No curated DNAmPACKYRS -> CHD record exists, so no chain is manufactured."""
    assert _run(kb, "!(infer &self SmokingCessation CoronaryHeartDisease)") == []


@pytest.mark.slow
@pytest.mark.parametrize("patient", ["Patient001", "Patient002"])
@pytest.mark.parametrize("query", [
    "(decompose-grimage &self {p})",
    "(predict-risk-patient &self {p})",
    "(risk-decomposition-patient &self {p})",
    "(counterfactual-scenarios &self {p})",
    "(risk-scenarios &self {p})",
    "(patient-observations &self {p})",
    "(diagnose-patient &self {p} (CellularSenescence MitochondrialDysfunction ChronicInflammation))",
    "(rank-interventions-for-patient &self {p} "
    "(DasatinibPlusQuercetin Fisetin Spermidine Elamipretide Metformin Berberine) "
    "CoronaryHeartDisease)",
    "(recommend-supplements-patient &self {p})",
])
def test_the_existing_patients_answers_are_byte_identical(before, kb, patient, query):
    """The regression that matters: a new Effect edge must not move old answers.

    It cannot, and here is why: the only new incoming edge lands on DNAmPACKYRS,
    which neither curated patient measures, so `patient-elevated-z` yields
    nothing for it and the component drops out of every fold.
    """
    q = "!" + query.format(p=patient)
    assert _run(before, q) == _run(kb, q)
