"""The translator and the validator must speak about atoms that exist.

Six of the 36 questions in the 2026-09-18 API evaluation produced a query that
validated cleanly and returned nothing, because `logical_predicates.metta`
declares a vocabulary far larger than the runtime populates. The symmetric bug
is worse and went unnoticed: the registry never harvested the ARGUMENTS of
ordinary ground facts, so 71 symbols that really are in the KB — `MTORC1`,
`AMPK`, `Mouse`, `Human` — were reported unknown, and `/metta/run` answered 422
to queries the engine would have served.

NOTE on the example predicate: this file used `TargetsHallmark` throughout as
its specimen "declared but empty". `hallmark_targeting.metta` populated it — the
evaluation asked for exactly that — so the specimen is now `Predicts`, which is
still declared (logical_predicates.metta:36) and still holds zero facts. The
assertions are unchanged in intent; only the predicate they point at moved.

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
    # `Predicts` stands in for what `TargetsHallmark` used to demonstrate here:
    # declared at logical_predicates.metta:36, zero facts in the runtime. (The
    # original example stopped being empty when hallmark_targeting.metta
    # back-filled it — which is the point of that patch, not a regression.)
    assert not inventory.is_grounded("Predicts")
    assert "Predicts" in inventory.declared_only
    # And the predicate that WAS the example is now grounded, so the same
    # machinery has to put it on the other side of the line.
    assert inventory.is_grounded("TargetsHallmark")
    assert "TargetsHallmark" not in inventory.declared_only
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


def test_a_comment_is_not_validated_as_if_it_were_a_symbol_or_a_parenthesis(inventory):
    """A patient file starts with `;;` header lines (the My Patient tab's download does). Validated word
    by word they were "Symbol(s) not found", so /metta/run answered 422 to the tab's own file."""
    registry = api_module._runtime_registry()
    program = (";; Caller_Me — built in the PLN 'My Patient' tab for one browser (session). Not part of the KB;\n"
               ";; load it as extra_atoms or send the same text again to rebuild it. (unbalanced\n"
               "!(match &self (HallmarkComponent MTORC1 $h) $h) ;; trailing words qwertyuiop zxcvbnm\n")
    result = validate(program, registry, inventory)
    assert result.valid, result.issues
    # what is NOT a comment is still checked: a quoted ';' is data, an unquoted one starts a comment
    assert validate('!(match &self (PublicationTitle $p "a; b") $p)', registry, inventory).valid
    assert not validate("!(match &self (Nonexistent_Thing_Zzz $x) $x)", registry, inventory).valid
    assert not validate("!(match &self (HallmarkComponent MTORC1 $h) $h", registry, inventory).valid   # unbalanced for real


def test_an_empty_predicate_is_flagged_rather_than_silently_returning_nothing(inventory):
    registry = api_module._runtime_registry()
    result = validate("!(match &self (Predicts $b $o) $o)", registry, inventory)
    assert result.valid                                   # it IS well-formed MeTTa
    assert result.ungrounded_predicates == ["Predicts"]
    assert any("return nothing" in w for w in result.warnings)


def test_the_predicate_the_evaluation_asked_us_to_populate_is_no_longer_flagged(inventory):
    """TargetsHallmark used to be this file's headline empty predicate."""
    registry = api_module._runtime_registry()
    result = validate("!(match &self (TargetsHallmark $i $h) $h)", registry, inventory)
    assert result.valid
    assert result.ungrounded_predicates == []


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
    assert "Predicts" in card
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
    assert "Predicts" in body["declared_but_empty_predicates"]
    assert "Predicts" not in names
    # TargetsHallmark moved from the second list to the first in patch 08.
    assert "TargetsHallmark" in names
    assert "TargetsHallmark" not in body["declared_but_empty_predicates"]
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
                json={"metta_query": "!(match &self (Predicts $b $o) $o)"},
            )

    body = asyncio.run(send()).json()
    assert body["pln_status"] == "empty"
    assert body["ungrounded_predicates"] == ["Predicts"]
    assert body["validation_warnings"]


# ── a rule whose data lives elsewhere must not answer "no" ───────────────────

def test_a_form_with_no_rows_in_the_generic_space_says_so():
    """The silent-empty that API.md's own reading rule turns into a wrong answer.

    `!(genes-affecting-senescence &self Increases)` validates, executes, and
    returns `pln_status: "empty"` with `ungrounded_predicates: []` — and
    API.md tells an agent that an empty result WITHOUT ungrounded predicates is
    a real "no". The data exists: `GET /genes/TP53?infer=true` returns three
    `(Effect Gene_TP53 CellularSenescence Pos …)` links from the same rules.
    """
    from ontology.scoped_forms import dataless_forms, scoped_form_warnings

    inventory = api_module._runtime_inventory()
    kb = api_module._runtime_kb_paths()

    # Derived, not listed: every CellAge and DrugAge accessor whose predicates
    # hold no facts in the generic space.
    dataless = dataless_forms(kb, inventory)
    assert "genes-affecting-senescence" in dataless
    assert "cellage-effect" in dataless
    assert "drugage-effect" in dataless
    # …and nothing that IS backed by facts.
    assert "predict-risk-patient" not in dataless
    assert "counterfactual-patient" not in dataless

    warned = scoped_form_warnings(
        "!(genes-affecting-senescence &self Increases)", kb, inventory
    )
    assert warned and "GET /genes" in warned[0]
    assert not scoped_form_warnings(
        "!(predict-risk-patient &self Patient001)", kb, inventory
    )


def test_the_warning_reaches_the_http_response():
    """The warning has to survive the trip through the HTTP layer.

    Needs a working engine, unlike its companion above, which drives the pure
    Python helpers. `/metta/run` has to EXECUTE for a MettaRunResponse (and so
    for `warnings`) to come back at all; without hyperon the query fails and the
    caller gets 502 `runtime_error` instead, with no response body to inspect.
    Reported on macOS as `assert 502 == 200` before this guard existed.
    """
    pytest.importorskip("hyperon")

    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/metta/run",
                json={"metta_query": "!(genes-affecting-senescence &self Increases)"},
            )

    response = asyncio.run(send())
    assert response.status_code == 200
    body = response.json()
    assert body["pln_status"] == "empty"
    assert any("no rows" in w.lower() or "NO facts" in w for w in body["warnings"]), (
        "an empty result from a data-less form must not look like a real 'no'"
    )
