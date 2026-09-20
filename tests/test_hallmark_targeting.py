"""The hallmark questions must be answerable — and must stop there.

The 2026-09-18 API evaluation, §3:

    "The translator used TargetsHallmark three times; the runtime holds zero
     such facts (raw match confirmed)... Symbols missing from the loaded
     layers: Rapamycin and MTORC1 in mechanistic bridges... Rapamycin and
     metformin have no hallmark links."

Q3, Q3b, Q3c, Q13 and Q22 all came back empty. `hallmark_targeting.metta`
closes that, and this file guards both halves of the fix:

* the RELATION now answers — `TargetsHallmark` is populated for the 13 curated
  interventions and for the two headline drugs, in both directions, through one
  query form and through the two accessors;
* the RESTRAINT holds — the new layer adds no `Effect` edge, so rapamycin is
  still unreachable by `infer`, and the patient-facing outputs are unchanged.

That second half is the load-bearing one. Chaining rapamycin into the existing
nutrient-sensing axis would put it in the intervention ranking tomorrow, with
the WRONG SIGN: chronic rapamycin causes glucose intolerance in mice (Weiss
2018, PMID 29579736). An LLM asked the same question will make that chain
happily. Refusing to is the point.

Run from the repository root:
    pytest tests/test_hallmark_targeting.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

TARGETING = REPO / "hallmark_targeting.metta"

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import api as api_module  # noqa: E402
from core.metta_validator import validate  # noqa: E402
from ontology.hallmarks import hallmark_index  # noqa: E402
from ontology.inventory import inventory_for, iter_top_level, split_args  # noqa: E402


@pytest.fixture(scope="module")
def index():
    return hallmark_index(api_module._runtime_kb_paths())


@pytest.fixture(scope="module")
def inventory():
    return inventory_for(api_module._runtime_kb_paths())


def _facts(head: str) -> list[list[str]]:
    """Every top-level `(head …)` expression in the new layer, as arg lists."""
    out = []
    for expr in iter_top_level(TARGETING.read_text(encoding="utf-8")):
        if not (expr.startswith("(") and expr.endswith(")")):
            continue
        parts = split_args(expr[1:-1].strip())
        if parts and parts[0] == head:
            out.append(parts[1:])
    return out


# ── the layer is registered, in both copies of the stack ─────────────────────

def _stack_literal(module_path: Path) -> list[str]:
    """Read `_INFERENCE_STACK = [...]` out of a source file without importing it.

    app.py imports gradio, which the test environment does not install, and the
    list is a plain literal — so parse it rather than drag the UI in.
    """
    import ast

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "_INFERENCE_STACK":
                return ast.literal_eval(node.value)
    raise AssertionError(f"no _INFERENCE_STACK in {module_path}")


def test_both_inference_stacks_load_the_new_layer():
    """api.py and app.py duplicate _INFERENCE_STACK verbatim and must agree."""
    assert "hallmark_targeting.metta" in api_module._INFERENCE_STACK
    app_stack = _stack_literal(PLN_CHAT / "app.py")
    assert "hallmark_targeting.metta" in app_stack
    assert api_module._INFERENCE_STACK == app_stack


def test_the_layer_is_small_enough_to_load_into_a_hyperon_space():
    """hyperon 0.2.10 aborts on a variable-slot match past a few hundred rows."""
    assert TARGETING.stat().st_size < 60_000


# ── the relation answers, in both directions ─────────────────────────────────

def test_every_curated_evidence_record_has_a_targeting_line(index):
    """The back-fill is the review table projected, not a new set of claims."""
    from_records = {
        (e.intervention, e.hallmark) for e in index.records()
    }
    from_targeting = {(i, h) for i, h in _facts("TargetsHallmark")}
    assert from_records <= from_targeting
    assert len(from_records) == 13          # 14 records, D+Q contributes two


def test_rapamycin_exists_at_all_and_targets_two_hallmarks(inventory):
    """The evaluation's "Rapamycin ... missing from the loaded layers"."""
    assert inventory.knows_symbol("Rapamycin")
    targets = {h for i, h in _facts("TargetsHallmark") if i == "Rapamycin"}
    assert targets == {"DeregulatedNutrientSensing", "DisabledMacroautophagy"}


def test_metformin_finally_has_a_hallmark_link():
    targets = {h for i, h in _facts("TargetsHallmark") if i == "Metformin"}
    assert targets == {"DeregulatedNutrientSensing"}


def test_mitochondrial_dysfunction_has_an_intervention(index):
    """Q13 asked which interventions target it and got nothing back."""
    assert index.interventions_for("MitochondrialDysfunction") == ["Elamipretide"]


def test_the_declared_mechanism_vocabulary_finally_has_instances():
    """`(Causes Rapamycin (Inhibits mTORC1))` was a comment in the declarations."""
    causes = {(subject, target) for subject, target in _facts("Causes")}
    assert ("Rapamycin", "(Inhibits MTORC1)") in causes
    assert ("Metformin", "(Activates AMPK)") in causes


def test_the_mechanism_uses_symbols_that_actually_exist(inventory):
    for symbol in ("MTORC1", "AMPK"):
        assert inventory.knows_symbol(symbol), symbol


# ── nothing here invents a number ────────────────────────────────────────────

def test_the_layer_adds_no_truth_values_at_all():
    """TargetsHallmark is a targeting claim; an stv here would be invented."""
    text = TARGETING.read_text(encoding="utf-8")
    body = "\n".join(
        line.split(";;")[0] for line in text.splitlines()
    )
    assert "(stv" not in body


def test_the_layer_adds_no_effect_edges():
    """THE restraint: an Effect edge would chain rapamycin with the wrong sign."""
    assert _facts("Effect") == []


def test_every_pmid_cited_is_attached_to_a_publication_record():
    pmids = {args[1].strip('"') for args in _facts("PubMedID")}
    assert pmids == {"19587680", "28283069", "25041462", "29579736"}
    subjects = {args[0] for args in _facts("PubMedID")}
    titled = {args[0] for args in _facts("PublicationTitle")}
    dois = {args[0] for args in _facts("DOI")}
    assert subjects == titled == dois


def test_the_rapamycin_caveat_is_a_fact_not_just_a_comment():
    limits = {subject: note for subject, note in _facts("Limitation")}
    assert "glucose intolerance" in limits["Rapamycin"]
    assert "29579736" in limits["Rapamycin"]


# ── the runtime inventory has moved the predicate to the other list ──────────

def test_targets_hallmark_is_no_longer_declared_but_empty(inventory):
    assert inventory.is_grounded("TargetsHallmark")
    assert "TargetsHallmark" not in inventory.declared_only
    assert inventory.predicates["TargetsHallmark"].fact_count == 16


def test_a_query_naming_rapamycin_is_no_longer_rejected(inventory):
    """It used to 422: the symbol did not exist, so the validator refused it."""
    registry = api_module._runtime_registry()
    result = validate(
        "!(match &self (TargetsHallmark Rapamycin $h) $h)", registry, inventory
    )
    assert result.valid, result.issues
    assert result.ungrounded_predicates == []


# ── the HTTP surface keeps the two shapes apart ──────────────────────────────

def _get(path: str):
    import asyncio

    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(path)

    return asyncio.run(send())


def test_rapamycin_comes_back_as_a_targeting_fact_not_a_dressed_up_record():
    body = _get("/interventions?intervention=Rapamycin").json()
    assert body["evidence"] == []                      # no review record exists
    hallmarks = {t["hallmark"] for t in body["targeting"]}
    assert hallmarks == {"DeregulatedNutrientSensing", "DisabledMacroautophagy"}
    for link in body["targeting"]:
        assert link["provenance"] == "targeting_fact"
        assert "HarrisonEtAl2009_RapamycinITP" in link["publications"]
        # The fields a review record would carry are ABSENT, not fabricated.
        assert "species_model" not in link and "outcome_text" not in link
    assert "TargetsHallmark facts only" in body["note"]


def test_a_record_backed_intervention_keeps_its_richer_provenance():
    body = _get("/interventions?intervention=Fisetin").json()
    assert [e["hallmark"] for e in body["evidence"]] == ["CellularSenescence"]
    assert body["evidence"][0]["species_model"] == "Mouse"
    assert body["evidence"][0]["reference_number"] == 60
    # ...and the back-filled line for the same claim is not repeated as a
    # second, thinner copy of it.
    assert body["targeting"] == []


def test_the_hallmark_listing_covers_the_two_new_drugs():
    body = _get("/hallmarks").json()
    by_name = {h["name"]: h for h in body["hallmarks"]}
    nutrient = by_name["DeregulatedNutrientSensing"]
    assert {"Rapamycin", "Metformin", "CaloricRestriction"} <= set(nutrient["interventions"])
    assert nutrient["intervention_count"] == len(nutrient["interventions"])
    assert "Rapamycin" in by_name["DisabledMacroautophagy"]["interventions"]


def test_the_prompt_no_longer_tells_the_translator_to_avoid_the_predicate():
    rules = (PLN_CHAT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    assert "NOT TargetsHallmark" not in rules
    assert "TargetsHallmark IS now populated" in rules
    assert "(hallmarks-of &self <Intervention>)" in rules


# ── and the engine agrees (real MeTTa) ───────────────────────────────────────

hyperon = pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402

_NUM = r"([-\d.eE]+)"


@pytest.fixture(scope="module")
def metta():
    """The full runtime stack, loaded into one space as pln_runner does."""
    files = {p.name: p for p in api_module._runtime_kb_paths()}
    m = MeTTa()
    for name in api_module._INFERENCE_STACK:
        path = files.get(name)
        if path is not None:
            m.run(path.read_text(encoding="utf-8"))
    return m


def _run(m, query: str) -> list[str]:
    return [str(atom) for result in m.run(query) for atom in result]


@pytest.mark.slow
def test_the_raw_match_returns_rapamycins_two_hallmarks(metta):
    out = _run(metta, "!(match &self (TargetsHallmark Rapamycin $h) $h)")
    assert sorted(out) == ["DeregulatedNutrientSensing", "DisabledMacroautophagy"]


@pytest.mark.slow
def test_the_reverse_match_answers_the_question_q13_asked(metta):
    out = _run(metta, "!(match &self (TargetsHallmark $i MitochondrialDysfunction) $i)")
    assert out == ["Elamipretide"]


@pytest.mark.slow
def test_the_accessors_say_the_same_thing_in_one_call(metta):
    assert sorted(_run(metta, "!(hallmarks-of &self Rapamycin)")) == [
        "DeregulatedNutrientSensing",
        "DisabledMacroautophagy",
    ]
    assert sorted(_run(metta, "!(interventions-for &self CellularSenescence)")) == [
        "DasatinibPlusQuercetin",
        "Fisetin",
    ]


@pytest.mark.slow
def test_an_intervention_with_no_targeting_link_yields_nothing_not_a_placeholder(metta):
    assert _run(metta, "!(hallmarks-of &self Berberine)") == []


@pytest.mark.slow
def test_rapamycin_is_still_unreachable_by_inference(metta):
    """The restraint, measured: targeting is not a causal chain."""
    assert _run(metta, "!(infer &self Rapamycin CoronaryHeartDisease)") == []
    ranked = _run(
        metta,
        "!(rank-interventions &self (DasatinibPlusQuercetin Rapamycin) CoronaryHeartDisease)",
    )
    assert ranked and "Rapamycin" not in ranked[0]


@pytest.mark.slow
def test_the_patient_facing_outputs_did_not_move(metta):
    """The new layer adds no Effect edge, so these must be bit-for-bit as before.

    The three values below were captured on the parent commit (patch 07) and
    re-captured with this layer loaded; they matched byte for byte.
    """
    risk = _run(metta, "!(predict-risk-patient &self Patient001)")
    assert len(risk) == 1
    point = re.search(rf"\(point {_NUM}\)", risk[0])
    assert point and abs(float(point.group(1)) - 0.12605177716424967) < 1e-12

    diagnosis = _run(
        metta,
        "!(diagnose-patient &self Patient001 "
        "(CellularSenescence MitochondrialDysfunction ChronicInflammation))",
    )
    order = re.findall(r"\(Hypothesis (\w+)", diagnosis[0])
    assert order == [
        "CellularSenescence",
        "ChronicInflammation",
        "MitochondrialDysfunction",
    ]

    ranking = _run(
        metta,
        "!(rank-interventions-for-patient &self Patient001 "
        "(DasatinibPlusQuercetin Fisetin Spermidine Elamipretide) CoronaryHeartDisease)",
    )
    scored = re.findall(rf"\(scored (\w+) {_NUM}", ranking[0])
    assert [name for name, _ in scored] == [
        "DasatinibPlusQuercetin",
        "Fisetin",
        "Spermidine",
    ]
    assert abs(float(scored[0][1]) - 0.2306046747621094) < 1e-12
