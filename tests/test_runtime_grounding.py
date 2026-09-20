"""The translator and the validator must speak about atoms that exist.

Six of the 36 questions in the 2026-09-18 API evaluation produced a query that
validated cleanly and returned nothing, because `logical_predicates.metta`
declares a vocabulary far larger than the runtime populates. The symmetric bug
is worse and went unnoticed: the registry never harvested the ARGUMENTS of
ordinary ground facts, so 71 symbols that really are in the KB — `MTORC1`,
`AMPK`, `Mouse`, `Human` — were reported unknown, and `/metta/run` answered 422
to queries the engine would have served.

Run from the repository root:
    pytest tests/test_runtime_grounding.py -q
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
from core.metta_validator import validate  # noqa: E402
from ontology.inventory import (  # noqa: E402
    build_inventory,
    inventory_for,
    schema_card,
    split_args,
    summarise_oversized,
)
from ontology.registry import BUILTIN_REGISTRY  # noqa: E402


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)


@pytest.fixture(scope="module")
def inventory():
    return inventory_for(api_module._runtime_kb_paths())


def _get(path: str):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(path)

    return asyncio.run(send())


# ── the inventory sees what the registry cannot ──────────────────────────────

def test_arguments_of_ordinary_ground_facts_are_known(inventory):
    """The 71 false negatives: real atoms the registry never harvested."""
    registry = api_module._runtime_registry()
    for symbol in ("MTORC1", "AMPK", "SIRT1", "Mouse", "Human", "CDKN2A_P16"):
        assert inventory.knows_symbol(symbol), symbol
        # ...and the registry alone still does not, which is why this exists.
        assert registry.get(symbol) is None, symbol


def test_a_declared_predicate_with_no_facts_is_told_apart_from_a_populated_one(inventory):
    assert inventory.is_grounded("EvidenceIntervention")
    assert not inventory.is_grounded("TargetsHallmark")
    assert "TargetsHallmark" in inventory.declared_only
    # A DrugAge row predicate is declared but deliberately absent from the
    # generic runtime KB — the translator emitted these three times.
    assert "UsesIntervention" in inventory.declared_only
    assert "AvgLifespanChangePercent" in inventory.declared_only


def test_a_function_the_rules_define_is_not_an_empty_predicate(inventory):
    """`evidence-confidence` is a lookup table, not a fact-free predicate."""
    assert "evidence-confidence" not in inventory.declared_only
    assert "evidence-confidence" in inventory.functions
    # A constructor used in a rule's pattern is structural, not empty either.
    assert "scored" not in inventory.declared_only


def test_nested_expressions_contribute_their_symbols():
    inv = build_inventory_from_text("(Causes Rapamycin (Inhibits MTORC1))")
    assert inv.knows_symbol("Rapamycin")
    assert inv.knows_symbol("MTORC1")
    assert inv.predicates["Causes"].fact_count == 1
    assert inv.predicates["Causes"].arity == 2


def build_inventory_from_text(text: str, tmp: Path | None = None):
    import tempfile

    directory = tmp or Path(tempfile.mkdtemp())
    path = directory / "probe.metta"
    path.write_text(text, encoding="utf-8")
    return build_inventory([path])


def test_comments_and_strings_do_not_create_facts():
    inv = build_inventory_from_text(
        ';; (TargetsHallmark Rapamycin NutrientSensing) — only a comment\n'
        '(PublicationTitle P "A study; with a semicolon (and parens)")\n'
    )
    assert "TargetsHallmark" not in inv.predicates
    assert inv.predicates["PublicationTitle"].fact_count == 1


def test_split_args_respects_nesting_and_strings():
    assert split_args('a (b c) "d e" f') == ["a", "(b c)", '"d e"', "f"]


# ── the validator stops being wrong in both directions ───────────────────────

def test_a_real_symbol_is_no_longer_rejected(inventory):
    registry = api_module._runtime_registry()
    result = validate("!(match &self (HallmarkComponent MTORC1 $h) $h)", registry, inventory)
    assert result.valid, result.issues


def test_an_empty_predicate_is_flagged_rather_than_silently_returning_nothing(inventory):
    registry = api_module._runtime_registry()
    result = validate("!(match &self (TargetsHallmark $i $h) $h)", registry, inventory)
    assert result.valid                                   # it IS well-formed MeTTa
    assert result.ungrounded_predicates == ["TargetsHallmark"]
    assert any("return nothing" in w for w in result.warnings)


def test_the_correct_form_for_the_same_question_is_clean(inventory):
    registry = api_module._runtime_registry()
    result = validate(
        "!(match &self (, (EvidenceIntervention $r $i) (EvidenceHallmark $r $h)) (pair $i $h))",
        registry,
        inventory,
    )
    assert result.valid and not result.ungrounded_predicates


def test_scientific_notation_is_not_mistaken_for_a_symbol(inventory):
    """`2.0e-75` used to contribute two 'unknown symbols', e and e-75."""
    registry = api_module._runtime_registry()
    result = validate("!(match &self (PValue $e 2.0e-75) $e)", registry, inventory)
    assert result.valid, result.issues


def test_a_genuinely_unknown_symbol_is_still_rejected(inventory):
    registry = api_module._runtime_registry()
    result = validate("!(match &self (Effect Flubberol $x Neg $tv) $x)", registry, inventory)
    assert not result.valid
    assert "Flubberol" in result.issues[0]


def test_validate_still_works_without_an_inventory():
    """The two-argument signature is used elsewhere and must keep working."""
    result = validate("!(match &self $x $x)", BUILTIN_REGISTRY)
    assert result.valid
    assert result.warnings == [] and result.ungrounded_predicates == []


# ── the prompt stops carrying bulk data ──────────────────────────────────────

def test_an_oversized_file_becomes_a_schema_card(tmp_path):
    big = tmp_path / "huge_etl.metta"
    big.write_text(
        "\n".join(f"(InvolvesGene Row_{i} Gene_{i})" for i in range(5_000)),
        encoding="utf-8",
    )
    raw = {big.name: big.read_text(encoding="utf-8")}
    out, summarised = summarise_oversized(raw, [big], max_bytes=25_000)

    assert summarised == [big.name]
    assert len(out[big.name]) < len(raw[big.name]) / 50
    assert "InvolvesGene" in out[big.name] and "5,000 ground facts" in out[big.name]


def test_a_curated_layer_is_left_verbatim(tmp_path):
    small = tmp_path / "curated.metta"
    small.write_text("(Effect A B Neg (stv 0.5 0.5))\n", encoding="utf-8")
    raw = {small.name: small.read_text(encoding="utf-8")}
    out, summarised = summarise_oversized(raw, [small], max_bytes=25_000)
    assert summarised == []
    assert out[small.name] == raw[small.name]


def test_the_schema_card_names_the_empty_predicates(inventory):
    card = schema_card(inventory, max_predicates=200)
    assert "DECLARED BUT EMPTY" in card
    assert "TargetsHallmark" in card
    assert "(EvidenceIntervention/2) x14" in card


def test_the_prompt_leads_with_the_grounded_card():
    from core.context_builder import build_system_prompt

    selected = api_module._default_selection(list(api_module._discover_metta_files()))
    registry, raw = api_module._build_context(selected)
    inv = api_module._runtime_inventory()

    without = build_system_prompt(registry, raw)
    with_inv = build_system_prompt(registry, raw, inv)

    assert "What the runtime KB actually holds" in with_inv
    assert "DECLARED BUT EMPTY" in with_inv
    # The flat symbol index it replaces was ~7,000 tokens of names with no counts.
    assert len(with_inv) < len(without)


# ── and a caller can read the same inventory ─────────────────────────────────

def test_kb_schema_answers_list_your_data_sources_and_counts():
    body = _get("/kb/schema").json()
    assert body["ground_facts"] > 0
    assert body["files"]
    names = {p["name"] for p in body["grounded_predicates"]}
    assert "EvidenceIntervention" in names
    assert "TargetsHallmark" in body["declared_but_empty_predicates"]
    assert "TargetsHallmark" not in names
    assert sum(body["facts_by_file"].values()) == body["ground_facts"]
    top = body["grounded_predicates"][0]
    assert top["fact_count"] >= body["grounded_predicates"][-1]["fact_count"]
    assert "schema_card" in body and "DECLARED BUT EMPTY" in body["schema_card"]


def test_metta_run_reports_an_ungrounded_predicate_instead_of_an_empty_answer(monkeypatch):
    from core.pln_runner import PLNRunResult

    monkeypatch.setattr(
        api_module, "run_query", Mock(return_value=PLNRunResult(status="empty", mode="runtime"))
    )

    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post(
                "/metta/run",
                json={"metta_query": "!(match &self (TargetsHallmark $i $h) $h)"},
            )

    body = asyncio.run(send()).json()
    assert body["pln_status"] == "empty"
    assert body["ungrounded_predicates"] == ["TargetsHallmark"]
    assert body["validation_warnings"]
