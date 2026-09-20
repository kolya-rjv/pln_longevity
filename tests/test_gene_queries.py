"""CellAge and GenAge, made answerable.

The 2026-09-18 evaluation, §4:

    "Gene data is present on disk but not queryable. 'Which genes drive
    cellular senescence?' returned four hallmark components, not the CellAge
    gene set. GenAge human genes: empty. CellAge n GenAge: impossible.
    Selecting the CellAge or GenAge ETL files as ontology_files pushed the
    prompt to 417k and 386k tokens, over OpenAI's 272k limit."

The prompt half was fixed earlier (a file over PLN_PROMPT_FILE_MAX_BYTES becomes
a schema card). This module covers the other half — the data itself:

* `ontology.gene_index` reads the four SOURCE TABLES under `data/` (falling back
  to the committed zips) rather than the generated MeTTa, because `build/` is
  gitignored, is 8-14x over PLN_MAX_KB_FILE_BYTES, and because
  `genage_models_parser.py:57` destroys the real symbols on the way out;
* `GET /genes`, `/genes/{key}`, `/genes/sources` and `/genes/intersection`
  answer the four questions above without an LLM;
* `cellage_calibration.metta` + `ontology.cellage_selector` lift a SELECTED,
  CAPPED slice of curated rows into `(Effect <gene> CellularSenescence <sign>
  (stv s c))` — the only inference in this patch, and the one place a number is
  produced.

The load-bearing tests here are the honesty ones: that the lift's confidence is
the `evidence-confidence` lookup and not a literal, that the ETL's own
`(Causes … (stv 0.82 0.70))` numbers never reach a response, that a symbol join
against GenAge models is warned about, and that the row cap is enforced BEFORE
hyperon is called — that last one because the failure it prevents aborts the
process rather than raising.

Run from the repository root:
    pytest tests/test_gene_queries.py -q
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
pytest.importorskip("pandas")
httpx = pytest.importorskip("httpx")

import api as api_module  # noqa: E402
import core.executor as executor_module  # noqa: E402
import ontology.cellage_selector as selector  # noqa: E402
from core.pln_runner import (  # noqa: E402
    CELLAGE_STACK,
    parse_cellage_effects,
    run_cellage_effects,
)
from ontology.gene_index import (  # noqa: E402
    SOURCE_KEYS,
    GeneRecord,
    build_gene_index,
    gene_index,
    normalise_pmid,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CELLAGE_SAMPLE = FIXTURES / "cellage_real_rows.metta"
CELLAGE_BUILD = REPO / "build" / "cellage_genes.metta"


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


def _index():
    index = gene_index()
    if not index.status["cellage_curated"].available:
        pytest.skip("no CellAge source table and no committed archive to read")
    return index


# ── the four questions the evaluation could not ask ──────────────────────────

def test_which_genes_drive_cellular_senescence_returns_genes_not_hallmarks():
    """The headline failure: four hallmark components instead of the gene set."""
    _index()
    body = _get("/genes?source=cellage_curated&effect=Induces&limit=200").json()

    assert body["total"] > 400, body["total"]
    symbols = {r["symbol"] for r in body["records"]}
    # Not a hallmark component in sight; these are real gene symbols.
    assert not symbols & {"TelomereAttrition", "DNADamage", "CellularSenescence"}
    for record in body["records"]:
        assert record["senescence_effect"] == "Induces"
        assert record["senescence_direction"] == "Increases"
        assert record["pmids"], "every curated row keeps the PMID it came from"

    # The genes the question is really about are in there, past the first page.
    for symbol in ("TP53", "CDKN1A", "CDKN2A"):
        rows = [r for r in _get(f"/genes/{symbol}").json()["records"]
                if r["source"] == "cellage_curated"]
        assert rows, symbol
        assert any(r["senescence_effect"] == "Induces" for r in rows), symbol


def test_the_other_direction_is_a_separate_answer_not_the_same_one():
    """`Inhibits` is 510 rows and must never be folded into "drives"."""
    _index()
    induces = _get("/genes?source=cellage_curated&effect=Induces&limit=1").json()
    inhibits = _get("/genes?source=cellage_curated&effect=Inhibits&limit=1").json()
    assert induces["total"] != inhibits["total"]
    assert induces["total"] + inhibits["total"] < 949   # `Unclear` is neither


def test_genage_human_genes_is_no_longer_empty():
    """"GenAge human genes: empty." — 307 rows, with uniprot and selection basis."""
    _index()
    body = _get("/genes?source=genage_human&limit=5").json()
    assert body["total"] == 307
    first = body["records"][0]
    assert first["symbol"] and first["entrez"]
    assert first["uniprot"]
    # The selection basis vocabulary is the one epistemic_calibration.metta
    # already maps to a confidence — it is preserved, not re-invented.
    assert set(first["selection_basis"]) <= {
        "mammal", "model", "cell", "functional", "human",
        "downstream", "putative", "upstream", "human_link",
    }


def test_cellage_intersect_genage_is_113_genes_by_entrez():
    """"CellAge n GenAge: impossible." — it is 113 genes, joined on entrez."""
    _index()
    body = _get("/genes/intersection?limit=200").json()
    assert body["key"] == "entrez"
    assert body["total"] == 113, body["total"]
    symbols = {s for g in body["genes"] for s in g["symbols"]}
    assert {"TP53", "SIRT1", "AKT1", "PARP1"} <= symbols
    assert body["warnings"] == []


def test_one_gene_answers_across_every_source_with_provenance():
    _index()
    body = _get("/genes/TP53").json()
    assert body["resolved_as"] == "symbol"
    assert body["entrez"] == 7157
    assert "cellage_curated" in body["sources"]
    # TP53 is annotated once per senescence type, and stays three rows.
    curated = [r for r in body["records"] if r["source"] == "cellage_curated"]
    assert len(curated) == 3
    assert {r["senescence_type"] for r in curated} == {
        "ReplicativeSenescence", "StressInducedSenescence",
        "OncogeneInducedSenescence",
    }
    assert all(r["pmids"] for r in curated)


def test_an_entrez_id_and_a_symbol_reach_the_same_gene():
    _index()
    by_symbol = _get("/genes/TP53").json()
    by_entrez = _get("/genes/7157").json()
    assert by_entrez["resolved_as"] == "entrez"
    assert by_symbol["records"] == by_entrez["records"]


# ── honesty: the symbol join against GenAge models ───────────────────────────

def test_a_symbol_join_against_genage_models_is_warned_about():
    """67 by symbol, 1 by entrez — and the 67 are orthologues, not identities."""
    _index()
    by_symbol = _get(
        "/genes/intersection?a=cellage_curated&b=genage_models&key=symbol").json()
    by_entrez = _get(
        "/genes/intersection?a=cellage_curated&b=genage_models&key=entrez").json()

    assert by_symbol["total"] > by_entrez["total"]
    assert by_symbol["warnings"], "a cross-species symbol join must not be silent"
    assert "ORTHOLOGY" in by_symbol["warnings"][0]
    # Even the entrez join, which is nearly empty, says why rather than
    # letting the caller read emptiness as "unrelated".
    assert by_entrez["warnings"]


def test_the_index_keeps_the_real_symbol_the_models_etl_destroys():
    """`aak-2` survives here; genage_models_etl.metta has `aak_2`."""
    index = _index()
    if not index.status["genage_models"].available:
        pytest.skip("no GenAge models table")
    symbols = {r.symbol for r in index.records if r.source == "genage_models"}
    assert "aak-2" in symbols
    assert "aak_2" not in symbols


# ── honesty: absences are absences ───────────────────────────────────────────

def test_an_unknown_gene_is_an_explicit_note_not_silence():
    _index()
    body = _get("/genes/NOTAREALGENE").json()
    assert body["records"] == []
    assert body["note"] and "not a statement about the gene" in body["note"]


def test_every_listing_is_capped_and_says_it_was_truncated():
    _index()
    body = _get("/genes?source=genage_models&limit=10").json()
    assert body["returned"] == 10
    assert body["total"] > 10
    assert body["truncated"] is True
    over = _get("/genes?limit=100000")
    assert over.status_code == 422
    assert over.json()["detail"]["code"] == "invalid_limit"


def test_an_unknown_source_names_the_known_ones():
    _index()
    r = _get("/genes?source=cellage")
    assert r.status_code == 422
    assert r.json()["detail"]["known_sources"] == list(SOURCE_KEYS)


def test_a_missing_source_degrades_to_unavailable_rather_than_raising():
    """`data/**/*.csv` is gitignored; a checkout with nothing must still answer."""
    from ontology.gene_index import SOURCES, SourceSpec

    missing = {
        k: SourceSpec(
            key=v.key, label=v.label,
            csv=Path("/nonexistent/nope.csv"),
            archive=Path("/nonexistent/nope.zip"),
            member=v.member, sep=v.sep, row_atom_prefix=v.row_atom_prefix,
        )
        for k, v in SOURCES.items()
    }
    index = build_gene_index(missing)
    assert index.records == []
    assert all(not s.available for s in index.status.values())
    assert all(s.note for s in index.status.values())
    assert index.lookup("TP53") == ([], "symbol")
    assert index.intersect("cellage_curated", "genage_human") == []


# ── the PMID_PMID_ ETL defect ────────────────────────────────────────────────

def test_the_doubled_pmid_prefix_normalises_from_either_spelling():
    """A build/ generated before the ETL fix is still on disk and still readable."""
    assert normalise_pmid("PMID_PMID_26583757") == "PMID_26583757"
    assert normalise_pmid("PMID_26583757") == "PMID_26583757"
    assert normalise_pmid("26583757") == "PMID_26583757"
    assert normalise_pmid("") is None
    assert normalise_pmid("not-a-pmid") is None


def test_the_etl_now_emits_exactly_one_pmid_prefix():
    cellage_etl = pytest.importorskip("cellage_etl")
    assert cellage_etl.pmid_atom("26583757") == "PMID_26583757"
    assert cellage_etl.pmid_atom(26583757) == "PMID_26583757"


def test_no_response_carries_a_doubled_prefix():
    _index()
    body = _get("/genes?source=cellage_curated&limit=50").json()
    pmids = [p for r in body["records"] for p in r.get("pmids", [])]
    assert pmids
    assert not [p for p in pmids if p.startswith("PMID_PMID_")]


# ── the row atom ids point at real atoms ─────────────────────────────────────

@pytest.mark.skipif(not CELLAGE_BUILD.exists(), reason="build/ has not been generated")
def test_metta_row_ids_match_the_generated_etl_exactly():
    """The index duplicates each ETL's filter; this is what pins it."""
    import re

    index = _index()
    for source, filename in (
        ("cellage_curated", "cellage_genes.metta"),
        ("cellage_expression", "cellage_expression.metta"),
        ("genage_human", "genage_human_etl.metta"),
        ("genage_models", "genage_models_etl.metta"),
    ):
        path = REPO / "build" / filename
        if not path.exists():
            continue
        emitted = set(re.findall(r"\(InstanceOf\s+(\S+)\s", path.read_text(encoding="utf-8")))
        mine = {r.metta_row_id for r in index.records
                if r.source == source and r.metta_row_id}
        assert mine == emitted, f"{source}: index and ETL disagree on row atoms"


# ── the scoped hyperon slice ─────────────────────────────────────────────────

def test_the_row_cap_is_enforced_before_hyperon_is_ever_called():
    """The abort is uncatchable, so a cap checked afterwards is not a cap."""
    rows = selector.load_rows(CELLAGE_SAMPLE)
    assert rows, "the committed fixture must parse"
    assert len(selector.select_rows(rows, limit=10_000)) <= selector.MAX_ROWS
    # And the slice builder honours it end to end, on the real build too.
    _text, selected = selector.build_cellage_slice(limit=10_000)
    assert len(selected) <= selector.MAX_ROWS


def test_an_unclear_row_is_never_selected_for_lifting():
    """The source declined to state a direction, so nothing may be lifted."""
    rows = selector.load_rows(CELLAGE_SAMPLE)
    assert all(r.liftable for r in selector.select_rows(rows))
    unclear = GeneRecord(source="cellage_curated", row_index=0, symbol="X",
                         senescence_effect="Unclear")
    assert unclear.senescence_direction is None


@pytest.mark.slow
def test_the_lift_turns_a_curated_row_into_a_calibrated_effect_link():
    pytest.importorskip("hyperon")
    result, rows, effects = run_cellage_effects(["TP53"], source=CELLAGE_SAMPLE)

    assert result.status == "ok"
    assert rows and effects
    for effect in effects:
        assert effect.gene_atom == "Gene_TP53"
        assert effect.sign == "Pos"          # TP53 INDUCES senescence
        assert effect.direction == "induces_senescence"
        # Confidence is the evidence-confidence lookup for InVitro, not a
        # literal written into this layer.
        assert effect.confidence == pytest.approx(0.35)
        assert effect.strength == pytest.approx(0.70)
        # Every link says where its numbers came from.
        assert "evidence-confidence" in effect.as_dict()["confidence_source"]
        assert "curated prior" in effect.as_dict()["strength_source"]


@pytest.mark.slow
def test_an_inhibiting_gene_lifts_to_the_opposite_sign():
    pytest.importorskip("hyperon")
    _result, _rows, effects = run_cellage_effects(["SIRT1"], source=CELLAGE_SAMPLE)
    assert effects
    assert {e.sign for e in effects} == {"Neg"}
    assert {e.direction for e in effects} == {"inhibits_senescence"}


@pytest.mark.slow
def test_each_lifted_link_carries_the_row_it_came_from():
    """Three identical links for TP53 are three ROWS, and must be legible as such."""
    pytest.importorskip("hyperon")
    _result, rows, effects = run_cellage_effects(["TP53"], source=CELLAGE_SAMPLE)
    assert len(effects) == len(rows)
    assert len({e.row_id for e in effects}) == len(effects)
    assert {e.senescence_type for e in effects} == {r.senescence_type for r in rows}
    assert all(e.pmid and not e.pmid.startswith("PMID_PMID_") for e in effects)


@pytest.mark.slow
def test_the_confidence_really_comes_from_epistemic_calibration():
    """Retune the tier, retune the layer. A literal would not move."""
    pytest.importorskip("hyperon")
    from hyperon import MeTTa

    rows = selector.select_rows(selector.load_rows(CELLAGE_SAMPLE), ["TP53"])
    assert rows

    # Retune the tier AT ITS AUTHORITY — rewrite the one line in
    # epistemic_calibration.metta rather than appending a second equation, which
    # MeTTa would keep alongside the first.
    blocks = []
    for path in CELLAGE_STACK:
        text = path.read_text(encoding="utf-8")
        if path.name == "epistemic_calibration.metta":
            assert "(= (evidence-confidence InVitro) 0.35)" in text
            text = text.replace("(= (evidence-confidence InVitro) 0.35)",
                                "(= (evidence-confidence InVitro) 0.99)")
        blocks.append(text)
    retuned = "\n".join(blocks)

    metta = MeTTa()
    metta.run(retuned + "\n" + selector.slice_metta(rows))
    out = str(metta.run(f"!(cellage-effect &self {rows[0].row_id})"))
    lifted = parse_cellage_effects(out)
    assert lifted, out
    assert lifted[0].confidence == pytest.approx(0.99)


@pytest.mark.slow
def test_a_gene_with_no_curated_row_lifts_nothing_and_says_why():
    pytest.importorskip("hyperon")
    result, rows, effects = run_cellage_effects(["GHR"], source=CELLAGE_SAMPLE)
    assert rows == []
    assert effects == []
    assert result.status == "empty"


@pytest.mark.slow
def test_the_inference_endpoint_never_presents_the_etl_truth_values():
    """build/cellage_genes.metta carries (Causes … (stv 0.82 0.70)). Not ours."""
    pytest.importorskip("hyperon")
    _index()
    body = _get("/genes/TP53?infer=true").json()
    inference = body["inference"]
    assert inference["available"] is True
    assert inference["effects"]
    for effect in inference["effects"]:
        assert effect["strength"] == pytest.approx(0.70)
        assert effect["confidence"] == pytest.approx(0.35)
        assert (effect["strength"], effect["confidence"]) != (0.82, 0.70)
    assert "not_the_etl_numbers" in inference["semantics"]
    assert "cellage_calibration.metta" in inference["stack"]
    assert inference["rows_injected"] <= inference["rows_cap"]


def test_inference_is_opt_in_and_absent_by_default():
    _index()
    assert _get("/genes/TP53").json()["inference"] is None


@pytest.mark.slow
def test_a_gene_outside_cellage_reports_why_it_cannot_be_lifted():
    pytest.importorskip("hyperon")
    _index()
    inference = _get("/genes/GHR?infer=true").json()["inference"]
    assert inference["available"] is False
    assert inference["effects"] == []
    assert "No CellAge curated record" in inference["note"]


def test_the_slice_falls_back_to_the_committed_fixture_without_a_build(monkeypatch):
    """build/ is gitignored, so the inference path must survive its absence."""
    monkeypatch.setattr(selector, "BUILD_CELLAGE", Path("/nonexistent/cellage.metta"))
    assert selector.build_available() is False
    assert selector.resolved_source().startswith("tests/fixtures/")
    rows = selector.load_rows()
    assert rows, "the committed fixture is the fallback and must parse"
    assert all(r.liftable for r in selector.select_rows(rows))


def test_a_fixture_backed_answer_says_it_is_a_sample(monkeypatch):
    """25 rows of 927 must never be presented as the corpus."""
    pytest.importorskip("hyperon")
    _index()
    monkeypatch.setattr(selector, "BUILD_CELLAGE", Path("/nonexistent/cellage.metta"))
    inference = _get("/genes/TP53?infer=true").json()["inference"]
    assert inference["source"].startswith("tests/fixtures/")
    assert inference["note"] and "sample" in inference["note"]


def test_the_calibration_layer_is_discoverable_and_small_enough_to_load():
    """A hand-written layer belongs in the runtime space; an ETL dump does not."""
    from config import PLN_MAX_KB_FILE_BYTES

    path = REPO / "cellage_calibration.metta"
    assert path.exists()
    assert path.stat().st_size < PLN_MAX_KB_FILE_BYTES
    files = _get("/ontology/files").json()
    assert "cellage_calibration.metta" in files["files"]


# ── discovery ────────────────────────────────────────────────────────────────

def test_sources_reports_what_can_and_cannot_be_answered():
    index = _index()
    body = _get("/genes/sources").json()
    keys = [s["source"] for s in body["sources"]]
    assert keys == list(SOURCE_KEYS)
    assert body["total_records"] == len(index.records)
    # The filter vocabularies are read off the data, so a caller can never be
    # told to pass a value the data does not contain.
    assert "Induces" in body["vocabularies"]["effect"]
    assert "Caenorhabditis elegans" in body["vocabularies"]["organism"]
    assert "curated" in body["note"].lower()


def test_the_cross_source_filter_is_the_intersection_in_row_form():
    _index()
    listing = _get(
        "/genes?source=cellage_curated"
        "&in_sources=cellage_curated,genage_human&limit=200").json()
    shared = _get("/genes/intersection?limit=200").json()
    genes = {r["entrez"] for r in listing["records"]}
    assert genes <= {g["entrez"] for g in shared["genes"]}
    # Rows, not genes: CellAge annotates some of them more than once.
    assert listing["total"] >= shared["total"]
