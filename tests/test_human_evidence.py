"""Human evidence, and the three absences it keeps apart (human_evidence.metta).

The 2026-09-18 evaluation, §9:

    "No human-evidence tier for metformin or senolytics ... Q11 ('metformin in
    humans') was recorded as an honest Gap."

The prose answer the system gave was true — the bulk data loaded here (DrugAge,
GenAge, CellAge) is model-organism and cell evidence and cannot speak to people
— and it was un-grounded: there was nothing in the knowledge base to point at,
so the honest answer had to be composed in English by a language model.

`human_evidence.metta` is the record set that replaces that paragraph, and these
tests are about the distinctions it exists to preserve:

  * a NULL result (D+Q did not change pulmonary function in 14 people with IPF)
    is not the same as NO RECORD, and both are answerable;
  * a PLANNED trial that has not reported (TAME) carries no evidence tier and no
    truth value, and the record shape makes that structural rather than an
    oversight;
  * evidence already recorded elsewhere in the KB (omega-3) is cross-referenced,
    not duplicated, so one body of evidence cannot be counted twice.

Skipped automatically if `hyperon` / `fastapi` are not installed.

Run from the repository root:
    pytest tests/test_human_evidence.py -q
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

HUMAN = REPO / "human_evidence.metta"

from ontology.inventory import iter_top_level, split_args  # noqa: E402

# Dependency-ordered load, matching the header in each .metta file.
KB_FILES = [
    "system_types.metta",
    "logical_predicates.metta",
    "epistemic_calibration.metta",
    "grim_age_core.metta",
    "grim_age_lu2019_evidence.metta",
    "evidence_calibration.metta",
    "hallmarks_core.metta",
    "hallmarks_lopezotin2023_intervention_evidence.metta",
    "hallmark_targeting.metta",
    "mechanistic_bridges.metta",
    "pln_deduction.metta",
    "pln_intervention_ranking.metta",
    "pln_abductive_diagnosis.metta",
    "patient_profile.metta",
    "pln_counterfactual.metta",
    "pln_risk_prediction.metta",
    "lifestyle_evidence.metta",
    "supplement_evidence.metta",
    "pln_supplement_recommendation.metta",
    "human_evidence.metta",
]


def _facts(path: Path, head: str) -> list[list[str]]:
    out = []
    for expr in iter_top_level(path.read_text(encoding="utf-8")):
        if not (expr.startswith("(") and expr.endswith(")")):
            continue
        parts = split_args(expr[1:-1].strip())
        if parts and parts[0] == head:
            out.append(parts[1:])
    return out


def _body(path: Path) -> str:
    return "\n".join(line.split(";;")[0] for line in path.read_text(encoding="utf-8").splitlines())


# ── the layer's honesty contract, checkable without an engine ────────────────

def test_the_layer_carries_no_truth_values_at_all():
    """A study record is provenance, not a calibrated causal edge."""
    assert "(stv" not in _body(HUMAN)


def test_the_layer_adds_no_effect_edges():
    """It must not change a single answer `infer` already gives."""
    assert _facts(HUMAN, "Effect") == []


def _records() -> dict[str, dict[str, str]]:
    """The `(HumanEvidence …)` atoms, as {record_id: {keyword: value}}."""
    out: dict[str, dict[str, str]] = {}
    for args in _facts(HUMAN, "HumanEvidence"):
        fields = {"intervention": args[1], "outcome": args[2]}
        for chunk in args[3:]:
            kv = split_args(chunk[1:-1].strip())
            fields[kv[0]] = " ".join(kv[1:]).strip('"')
        out[args[0]] = fields
    return out


def test_every_tier_used_exists_in_the_single_calibration_authority():
    declared = {
        args[0] for args in _facts(REPO / "epistemic_calibration.metta", ":")
        if len(args) >= 2 and args[1] == "EvidenceCategory"
    }
    used = {f["tier"] for f in _records().values()} - {"NotStated"}
    used |= {args[1] for args in _facts(HUMAN, "HumanEvidenceElsewhere")}
    assert used == {"Epidemiological", "SingleHumanTrial", "MultipleHumanTrials"}
    assert used <= declared


def test_tame_states_that_it_has_no_tier_rather_than_omitting_one():
    """Not a low tier, not a default, not a forgotten line — written out."""
    tame = _records()["Barzilai2016_TAME_Metformin"]
    assert tame["tier"] == "NotStated"
    assert tame["n"] == "NotStated"
    assert tame["result"] == "NotYetReported"
    assert tame["design"] == "PlannedTrial"
    # ...and no OTHER record leans on that escape hatch for its tier.
    untiered = [r for r, f in _records().items() if f["tier"] == "NotStated"]
    assert untiered == ["Barzilai2016_TAME_Metformin"]


def test_every_pmid_cited_is_attached_to_a_full_publication_record():
    pmids = {args[1].strip('"') for args in _facts(HUMAN, "PubMedID")}
    # Barzilai 2016 (TAME), Hickson 2019 (D+Q, DKD), Justice 2019 (D+Q, IPF).
    assert pmids == {"27304507", "31542391", "30616998"}
    subjects = {args[0] for args in _facts(HUMAN, "PubMedID")}
    assert subjects == {args[0] for args in _facts(HUMAN, "PublicationTitle")}
    assert subjects == {args[0] for args in _facts(HUMAN, "DOI")}


def test_bannisters_publication_is_reused_rather_than_redeclared():
    """One paper, one record. It already lives in hallmark_targeting.metta."""
    cited = {args[1] for args in _facts(HUMAN, "SupportedByPublication")}
    assert "BannisterEtAl2014_Metformin" in cited
    declared_here = {
        args[0] for args in _facts(HUMAN, ":")
        if len(args) >= 2 and args[1] == "Publication"
    }
    assert "BannisterEtAl2014_Metformin" not in declared_here
    targeting = _facts(REPO / "hallmark_targeting.metta", "PubMedID")
    assert ("BannisterEtAl2014_Metformin", '"25041462"') in {tuple(a) for a in targeting}


def test_every_record_carries_every_field_and_a_publication():
    records = _records()
    assert len(records) == 5
    for record_id, fields in records.items():
        for key in ("intervention", "outcome", "design", "n", "result",
                    "tier", "pmid", "measured", "found", "caveat"):
            assert fields.get(key), f"{record_id} is missing {key}"
    cited = {args[0] for args in _facts(HUMAN, "SupportedByPublication")}
    assert cited == set(records)


def test_the_pmid_in_a_record_agrees_with_the_publication_it_cites():
    """The record's own pmid is a convenience, not a second source of truth."""
    pubs = {}
    for path in (HUMAN, REPO / "hallmark_targeting.metta"):
        pubs.update({args[0]: args[1].strip('"') for args in _facts(path, "PubMedID")})
    cited = {args[0]: args[1] for args in _facts(HUMAN, "SupportedByPublication")}
    for record_id, fields in _records().items():
        assert fields["pmid"] == pubs[cited[record_id]], record_id


def test_both_inference_stacks_load_the_layer_and_still_agree():
    def stack(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = node.targets
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "_INFERENCE_STACK":
                    return ast.literal_eval(node.value)
        raise AssertionError(f"no _INFERENCE_STACK in {path}")

    api_stack = stack(PLN_CHAT / "api.py")
    assert "human_evidence.metta" in api_stack
    assert api_stack == stack(PLN_CHAT / "app.py")


# ── the Python index reads the records the same way ──────────────────────────

@pytest.fixture(scope="module")
def index():
    from ontology.human_evidence import build_human_evidence_index

    return build_human_evidence_index(
        [REPO / "human_evidence.metta", REPO / "hallmark_targeting.metta"]
    )


def test_the_index_turns_notstated_into_none_not_into_a_zero(index):
    tame = index.studies["Barzilai2016_TAME_Metformin"]
    assert tame.design == "PlannedTrial"
    assert tame.result == "NotYetReported"
    assert tame.tier is None                 # not "" and not a low tier
    assert tame.n is None                    # not 0
    assert tame.pmid == "27304507"
    assert tame.publication.pmid == "27304507"


def test_the_index_distinguishes_a_null_result_from_a_missing_record(index):
    dq = {s.outcome: s for s in index.for_intervention("DasatinibPlusQuercetin")}
    assert dq["PulmonaryFunction"].result == "ReportedNull"
    assert dq["PhysicalFunction"].result == "ReportedBenefit"
    # Same trial, same people, two different answers.
    assert dq["PulmonaryFunction"].n == dq["PhysicalFunction"].n == 14
    assert dq["PulmonaryFunction"].publication.pmid == \
        dq["PhysicalFunction"].publication.pmid == "30616998"
    # ...and nothing at all for a compound with no curated human study.
    assert index.for_intervention("Rapamycin") == []


def test_the_index_joins_a_publication_another_file_owns(index):
    bannister = index.studies["Bannister2014_Metformin_Survival"]
    assert bannister.publication.pmid == "25041462"
    assert bannister.publication.doi == "10.1111/dom.12354"
    assert bannister.tier == "Epidemiological"
    assert bannister.n == 78241


def test_omega3_is_cross_referenced_rather_than_duplicated(index):
    xrefs = index.cross_references_for("Omega3")
    assert len(xrefs) == 1
    assert xrefs[0].tier == "MultipleHumanTrials"
    assert "supplement_evidence.metta" in xrefs[0].where
    assert index.for_intervention("Omega3") == []


# ── the HTTP surface says the same thing ─────────────────────────────────────

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import asyncio  # noqa: E402
from unittest.mock import Mock  # noqa: E402

import api as api_module  # noqa: E402
import core.executor as executor_module  # noqa: E402


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)


def _get(path: str):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(path)

    return asyncio.run(send())


def test_the_endpoint_answers_the_question_q11_asked():
    body = _get("/evidence/human?intervention=Metformin").json()
    by_id = {s["record_id"]: s for s in body["studies"]}
    assert set(by_id) == {"Bannister2014_Metformin_Survival", "Barzilai2016_TAME_Metformin"}

    observational = by_id["Bannister2014_Metformin_Survival"]
    assert observational["design"] == "ObservationalCohort"
    assert observational["evidence_tier"] == "Epidemiological"
    assert observational["publication"]["pmid"] == "25041462"
    assert "confounded by indication" in observational["caveat"]

    tame = by_id["Barzilai2016_TAME_Metformin"]
    assert tame["evidence_tier"] is None
    assert tame["result"] == "NotYetReported"
    assert "quoting a study design" in tame["caveat"]


def test_an_intervention_with_no_record_gets_a_note_not_silence():
    body = _get("/evidence/human?intervention=Rapamycin").json()
    assert body["studies"] == []
    assert "absence of a RECORD" in body["note"]
    assert "Metformin" in body["covered_interventions"]


def test_a_cross_reference_is_reported_as_one():
    body = _get("/evidence/human?intervention=Omega3").json()
    assert body["studies"] == []
    assert body["cross_references"][0]["provenance"] == "cross_reference"
    assert body["cross_references"][0]["evidence_tier"] == "MultipleHumanTrials"
    assert "cross-referenced rather than duplicated" in body["note"]


def test_the_whole_table_lists_every_record_and_the_cross_reference():
    body = _get("/evidence/human").json()
    assert len(body["studies"]) == 5
    assert len(body["cross_references"]) == 1
    assert body["note"] is None
    assert "PulmonaryFunction" in body["covered_outcomes"]


# ── and the engine agrees (real MeTTa) ───────────────────────────────────────

hyperon = pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402


@pytest.fixture(scope="module")
def kb() -> MeTTa:
    m = MeTTa()
    m.run("\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES))
    return m


def _run(m: MeTTa, query: str) -> list[str]:
    return [str(atom) for result in m.run(query) for atom in result]


@pytest.mark.slow
def test_the_accessor_returns_both_metformin_records(kb):
    out = _run(kb, "!(human-evidence &self Metformin)")
    assert len(out) == 2
    joined = "\n".join(out)
    assert "(design ObservationalCohort) (n 78241) (result ReportedBenefit)" in joined
    assert "(tier Epidemiological) (pmid \"25041462\")" in joined
    # TAME: no n, no tier, and the reason is written into the record.
    assert "(design PlannedTrial) (n NotStated) (result NotYetReported)" in joined
    assert "(tier NotStated) (pmid \"27304507\")" in joined
    assert "you are quoting a study design" in joined


@pytest.mark.slow
def test_a_null_and_a_benefit_come_back_from_the_same_trial(kb):
    out = _run(kb, "!(human-evidence &self DasatinibPlusQuercetin)")
    assert len(out) == 3
    results = sorted(re.search(r"\(result (\w+)\)", o).group(1) for o in out)
    assert results == ["ReportedBenefit", "ReportedBenefit", "ReportedNull"]
    null_record = next(o for o in out if "(result ReportedNull)" in o)
    assert "DasatinibPlusQuercetin PulmonaryFunction" in null_record
    assert "unchanged" in null_record


@pytest.mark.slow
def test_an_absent_record_is_a_statement_not_an_empty_result(kb):
    out = _run(kb, "!(human-evidence &self Rapamycin)")
    assert len(out) == 1
    assert out[0].startswith("(NoHumanEvidenceRecord Rapamycin")
    assert "absence of a RECORD, not evidence of absence" in out[0]


@pytest.mark.slow
def test_a_cross_reference_points_instead_of_copying(kb):
    out = _run(kb, "!(human-evidence &self Omega3)")
    assert len(out) == 1
    assert out[0].startswith("(HumanEvidenceCrossReferenced Omega3 (tier MultipleHumanTrials)")
    assert "supplement_evidence.metta" in out[0]


@pytest.mark.slow
def test_the_layer_changes_no_existing_answer(kb):
    """It adds no edge, so the recommender and the deduction layer are untouched."""
    assert _run(kb, "!(infer &self Metformin PhysicalFunction)") == []
    assert "(point 0.2363" in _run(kb, "!(predict-risk-patient &self Patient003)")[0]
    rec = _run(kb, "!(supplement-for-patient &self Patient001 Resveratrol)")
    assert rec and "NotRecommended" in rec[0]


# ── the canary for the failure this patch actually hit ───────────────────────
# Adding these two layers to the shared runtime space CRASHED the suite before
# the record shape was flattened: hyperon 0.2.10 aborts the interpreter with a
# non-unwinding Rust panic (hyperon-space/src/index/trie.rs) once the generic
# KB space grows past roughly a thousand top-level expressions, and an abort
# cannot be caught — `pytest` died mid-run with "Fatal Python error: Aborted"
# and no failing test to point at.
#
# So the guard runs the query that died, in a SUBPROCESS. A regression then
# shows up as one red test with a readable message instead of a dead suite.
# Measured: one-predicate-per-field records cost ~113 top-level expressions and
# aborted; the flat records cost ~49 and do not.

_BUDGET = 1_100        # top-level expressions in the runtime KB, with margin


def test_the_runtime_kb_stays_inside_its_measured_expression_budget():
    total = 0
    for path in api_module._runtime_kb_paths():
        total += sum(1 for _ in iter_top_level(path.read_text(encoding="utf-8")))
    assert total <= _BUDGET, (
        f"The generic runtime KB is now {total} top-level expressions. hyperon "
        f"0.2.10 ABORTS the process (uncatchable) somewhere past ~1050 here, "
        f"which kills the whole test run rather than failing a test. Shrink the "
        f"new layer (fewer atoms per record — see human_evidence.metta §1) or "
        f"exclude it from _runtime_kb_paths()."
    )


@pytest.mark.slow
def test_the_runtime_kb_still_answers_the_query_that_aborted_the_interpreter():
    """Out of process, because an abort here would take the suite with it."""
    import subprocess

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import api\n"
        "from core.pln_runner import run_query\n"
        "from core.patient_builder import build_patient\n"
        "b = build_patient({'id': 'Canary', 'age': 45, 'sex': 'Female',\n"
        "                   'markers': {'AgeAccelGrim': 0.5, 'CRP': 1.2}})\n"
        "r = run_query('!(predict-risk-patient &self ' + b.patient_id + ')',\n"
        "              kb_files=api._runtime_kb_paths(), extra_atoms=b.atoms)\n"
        "print(r.status)\n"
    ) % str(PLN_CHAT)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, (
        "hyperon aborted the interpreter on the full runtime KB "
        f"(exit {proc.returncode}). Tail:\n{proc.stderr[-2000:]}"
    )
    assert proc.stdout.strip().endswith("ok")
