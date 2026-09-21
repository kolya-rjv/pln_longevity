"""A paper expansion must land in the schema the rules already read.

The 2026-09-18 API evaluation, §10:

    "From the taurine abstract it minted new predicates (increases-life-span,
     declines-with-aging, reduces) instead of the KB's EvidenceIntervention /
     EvidenceHallmark, assigned LLM-made truth values (stv 0.93 0.9, a 'study
     confidence' constant of 0.92) that bypass the calibration tables, and
     recorded no PMID or DOI in the facts. Applied as-is, the new knowledge
     would not feed the existing PLN rules."

Every claim in that paragraph has a test here. The evaluation's own output is a
fixture (`THE_EVALUATIONS_BLOCK`) and must come back refused, with reasons. The
CANONICAL block — the shape this patch steers the extractor towards — is built
by hand from the same paper (Singh et al. 2023, Science, PMID 37289866,
doi 10.1126/science.abn9257: taurine deficiency as a driver of aging; taurine
supplementation raised median mouse lifespan ~10-12%) and loaded into hyperon
alongside the runtime knowledge base, where `infer`, `explain`,
`rank-interventions` and `drugage-effect` are asserted to consume it. That last
test is the one that proves the schema is the right schema rather than merely a
tidier one.

No OpenAI call is made anywhere in this module: `call_extraction_llm` is
monkeypatched with a canned response.

Run from the repository root:
    pytest tests/test_ontology_expansion.py -q
"""
from __future__ import annotations

import asyncio
import re
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
import ontology.expander as expander  # noqa: E402
from ontology.expansion_schema import (  # noqa: E402
    CALIBRATION_FUNCTIONS,
    CANONICAL_PREDICATES,
    SCHEMA_EXEMPLAR,
    check_entry,
    evidence_categories,
    lifespan_halfsat,
    lifespan_strength,
    normalise_truth_values,
    unconsumed_predicates,
)
from ontology.inventory import inventory_for, iter_top_level, split_args  # noqa: E402


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(api_module, "log_http_request", Mock())
    monkeypatch.setattr(executor_module, "PLN_WORKER_POOL_SIZE", 0)


@pytest.fixture(scope="module")
def inventory():
    return inventory_for(api_module._runtime_kb_paths())


def _post(path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post(path, **kwargs)

    return asyncio.run(send())


# ── the fixtures: what the evaluation got, and what it should have got ───────

#: Verbatim in spirit from the evaluation's §10 output: three minted predicates,
#: two-float truth values, a hand-rolled confidence constant, no identifier.
THE_EVALUATIONS_BLOCK = (
    "(: increases-life-span (-> Intervention Lifespan Atom))\n"
    "(Evaluation (increases-life-span Taurine Lifespan) (stv 0.93 0.9))\n"
    "(Evaluation (declines-with-aging TaurineLevel) (stv 0.95 0.9))\n"
    "(Evaluation (reduces Taurine CellularSenescence) (stv 0.88 0.92))\n"
    "(= (study-confidence SingleLargeCohortStudy) 0.92)"
)

TAURINE_PMID = "37289866"
TAURINE_DOI = "10.1126/science.abn9257"

#: The canonical block, by hand. Six shapes: a Publication record with a DOI,
#: the typing of the new node, a raw source-faithful measurement row carrying
#: its PMID, a review-level audit record, the signed Effect link the inference
#: layer consumes (confidence = an UNEVALUATED calibration lookup), and the
#: Layer-4 modifier atoms. The strength is the DrugAge transform applied to the
#: paper's own +11%, so the curated link and the lifted row agree by
#: construction — which `test_the_raw_row_lifts_to_the_same_link_the_curated_atom_states`
#: then checks rather than assumes.
CANONICAL_TAURINE_BLOCK = f"""
(: SinghEtAl2023_Taurine Publication)
(PublicationTitle SinghEtAl2023_Taurine "Taurine deficiency as a driver of aging")
(PublicationYear SinghEtAl2023_Taurine 2023)
(JournalName SinghEtAl2023_Taurine "Science")
(DOI SinghEtAl2023_Taurine "{TAURINE_DOI}")

(Inheritance Taurine Supplement)

(InstanceOf Singh2023_TaurineRow_1 Experiment)
(UsesIntervention Singh2023_TaurineRow_1 Taurine)
(UsesSpecies Singh2023_TaurineRow_1 Mus_musculus)
(AvgLifespanChangePercent Singh2023_TaurineRow_1 11.0)   ; midpoint of the paper's 10-12%
(AvgLifespanSignificance Singh2023_TaurineRow_1 Significant)
(ReportedIn Singh2023_TaurineRow_1 PMID_{TAURINE_PMID})

(: Singh2023_Taurine_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark Singh2023_Taurine_Mouse CellularSenescence)
(EvidenceIntervention Singh2023_Taurine_Mouse Taurine)
(EvidenceSpeciesModel Singh2023_Taurine_Mouse Mouse)
(EvidenceOutcomeText Singh2023_Taurine_Mouse "taurine supplementation increased median lifespan")
(SupportedByPublication Singh2023_Taurine_Mouse SinghEtAl2023_Taurine)

(TargetsHallmark Taurine CellularSenescence)

(Effect Taurine Lifespan Pos
   (stv {lifespan_strength(11.0, 20.0)!r} (evidence-confidence AnimalStudies_Single)))

(EvidenceLevel Taurine AnimalStudies_Single)
(SafetyProfile Taurine GenerallyWellTolerated)
"""


def _fake_llm(entries: list[dict], **extra):
    payload = {
        "paper_title": "Taurine deficiency as a driver of aging",
        "paper_summary": "Taurine declines with age; supplementation extends mouse lifespan.",
        "identifier": f"PMID:{TAURINE_PMID}",
        "entries": entries,
    }
    payload.update(extra)
    return lambda **kwargs: payload


def _run(monkeypatch, entries: list[dict], **extra):
    monkeypatch.setattr(expander, "call_extraction_llm", _fake_llm(entries, **extra))
    return expander.run_expansion_pipeline(
        paper_data=b"Taurine deficiency as a driver of aging.",
        filename="taurine.txt",
        target_file_path=REPO / "pln_chat" / "ontology" / "metta_files" / "unused.metta",
        model="gpt-4o",
        temperature=0.1,
        apply=False,
    )


# ── the gate refuses exactly what the evaluation produced ────────────────────

def test_the_evaluations_own_output_is_rejected_with_reasons(inventory):
    findings = check_entry(THE_EVALUATIONS_BLOCK, inventory=inventory)
    codes = [f.code for f in findings]
    details = {f.detail for f in findings}

    # All three minted predicates are NAMED — not just the outer `Evaluation`
    # wrapper they were nested inside.
    for minted in ("increases-life-span", "declines-with-aging", "reduces"):
        assert minted in details, f"{minted} was not reported: {details}"

    assert codes.count("unknown_predicate") >= 3
    assert "invented_truth_value" in codes
    assert "invented_confidence_constant" in codes
    assert "missing_identifier" in codes

    # And every refusal is readable by a human, not just a code.
    for finding in findings:
        assert str(finding).startswith(finding.code + ": ")
        assert len(finding.message) > 30


def test_a_rejected_entry_is_reported_not_dropped(monkeypatch):
    result = _run(monkeypatch, [
        {"kind": "predicate", "name": "increases-life-span",
         "metta": THE_EVALUATIONS_BLOCK, "description": "The evaluation's block."},
    ])
    assert result.ok
    assert result.new_entries == []
    assert result.metta_block == ""
    assert len(result.rejected_entries) == 1

    rejected = result.rejected_entries[0]
    assert rejected.name == "increases-life-span"
    assert "unknown_predicate" in rejected.codes
    # The offending MeTTa comes back too, so a reviewer can see what was tried.
    assert "study-confidence" in rejected.metta


def test_an_entry_with_no_pmid_and_no_doi_is_refused(inventory):
    valid_shape = "(Inheritance Taurine Supplement)"
    assert [f.code for f in check_entry(valid_shape, inventory=inventory)] == [
        "missing_identifier"
    ]
    assert check_entry(valid_shape, identifier_text=f"PMID: {TAURINE_PMID}",
                       inventory=inventory) == []
    assert check_entry(valid_shape, identifier_text=f"doi {TAURINE_DOI}",
                       inventory=inventory) == []


def test_a_tier_outside_the_calibration_table_is_refused(inventory):
    made_up = ("(Effect Taurine Lifespan Pos "
               "(stv 0.35 (evidence-confidence SingleLargeCohortStudy)))")
    codes = [f.code for f in check_entry(
        made_up, identifier_text=f"PMID:{TAURINE_PMID}", inventory=inventory)]
    assert "unknown_evidence_category" in codes


def test_the_calibration_layer_cannot_be_redefined(inventory):
    for function in sorted(CALIBRATION_FUNCTIONS)[:4]:
        block = f"(= ({function} NovelTier) 0.92)"
        codes = [f.code for f in check_entry(
            block, identifier_text=f"PMID:{TAURINE_PMID}", inventory=inventory)]
        assert "redefines_calibration" in codes, function


# ── confidence is a lookup, strength is derived or labelled ──────────────────

def test_confidence_is_never_a_number_the_model_proposed():
    """The model may name a tier; the atom carries the lookup, unevaluated."""
    proposed = "(Effect Taurine Lifespan Pos (stv 0.93 0.9))"
    normalised = normalise_truth_values(
        proposed, evidence_tier="AnimalStudies_Single", effect_size_pct=11.0)

    assert "(evidence-confidence AnimalStudies_Single)" in normalised.metta
    # No two-float truth value survives anywhere in the emitted atom.
    assert not re.search(r"\(stv\s+[\d.]+\s+[\d.]+\)", normalised.metta)
    # The tier itself is still the extractor's reading, so it is provisional.
    assert "evidence_tier" in normalised.provisional_fields


def test_a_reported_effect_size_produces_a_derived_strength():
    k = lifespan_halfsat()
    expected = 11.0 / (11.0 + k)
    normalised = normalise_truth_values(
        "(Effect Taurine Lifespan Pos (stv 0.93 0.9))",
        evidence_tier="AnimalStudies_Single",
        effect_size_pct=11.0,
    )
    assert repr(expected) in normalised.metta
    # The model's own 0.93 is gone — it was a guess, and there was a transform.
    assert "0.93" not in normalised.metta
    assert "strength" not in normalised.provisional_fields
    assert any("DERIVED" in note for note in normalised.notes)


def test_a_strength_with_no_effect_size_is_labelled_a_curated_prior():
    normalised = normalise_truth_values(
        "(Effect Taurine CellularSenescence Neg (stv 0.6 (evidence-confidence InVitro)))",
        evidence_tier="InVitro",
        effect_size_pct=None,
    )
    assert "strength" in normalised.provisional_fields
    assert any("CURATED PRIOR" in note for note in normalised.notes)
    assert "(evidence-confidence InVitro)" in normalised.metta


def test_a_truth_value_with_no_tier_at_all_is_left_for_the_gate_to_refuse():
    """Never emit `(evidence-confidence None)` to paper over a missing tier."""
    proposed = "(Effect Taurine Lifespan Pos (stv 0.93 0.9))"
    normalised = normalise_truth_values(proposed, evidence_tier=None, effect_size_pct=11.0)
    assert normalised.metta == proposed
    assert [f.code for f in check_entry(
        normalised.metta, identifier_text=f"PMID:{TAURINE_PMID}")] == [
        "invented_truth_value"
    ]


def test_the_halfsat_knob_is_read_from_drugage_calibration():
    """Not hard-coded here: a retune of the knob retunes generated blocks."""
    text = (REPO / "drugage_calibration.metta").read_text(encoding="utf-8")
    declared = float(re.search(r"\(=\s*\(lifespan-halfsat\)\s*([\d.]+)\)", text).group(1))
    assert lifespan_halfsat() == declared
    assert lifespan_strength(11.0) == 11.0 / (11.0 + declared)


def test_the_evidence_category_enum_is_read_from_the_calibration_authority():
    categories = evidence_categories()
    text = (REPO / "epistemic_calibration.metta").read_text(encoding="utf-8")
    declared = re.findall(r"\(:\s+(\w+)\s+EvidenceCategory\)", text)
    assert list(categories) == declared
    assert len(categories) == 11
    assert "RCT_Human" in categories and "TraditionalUse" in categories


# ── the prompt no longer asks for the failure ────────────────────────────────

def test_the_prompt_no_longer_tells_the_model_to_invent_confidence():
    prompt = expander.build_extraction_prompt()
    for banished in (
        "Extract constant definitions for confidence levels of novel study types",
        "set (stv s c) accordingly",
    ):
        assert banished not in prompt, banished
    assert "MUST NOT propose a confidence" in prompt
    assert "MUST NOT define a new confidence constant" in prompt


def test_the_prompt_carries_the_schema_and_the_closed_lists():
    prompt = expander.build_extraction_prompt()
    # One verbatim example of each canonical form, not an alphabetical slice.
    for form in ("(: LopezOtinEtAl2023_Hallmarks Publication)",
                 "(InstanceOf DrugAgeRow_0 Experiment)",
                 "(: LopezOtin2023_Spermidine_Mouse HallmarkInterventionEvidence)",
                 "(Effect CellularSenescence SASP Pos",
                 "(TargetsHallmark Fisetin CellularSenescence)",
                 "(EvidenceLevel Omega3 MultipleHumanTrials)"):
        assert form in prompt, form
    for tier in evidence_categories():
        assert tier in prompt, tier
    for predicate in ("EvidenceIntervention", "EvidenceHallmark", "Effect", "ReportedIn"):
        assert predicate in prompt, predicate
    # ~4 KB of exemplar, not the 6 KB alphabetical slice of 290 KB it replaced.
    assert 2_000 < len(SCHEMA_EXEMPLAR) < 6_000


def test_every_exemplar_fact_is_verbatim_from_the_kb():
    """The exemplar claims to be verbatim; re-derive that from the repository."""
    corpus = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(REPO.glob("*.metta"))
    )
    normalised_corpus = {
        re.sub(r"\s+", " ", expr).strip() for expr in iter_top_level(corpus)
    }
    checked = 0
    for expr in iter_top_level(SCHEMA_EXEMPLAR):
        norm = re.sub(r"\s+", " ", expr).strip()
        assert norm in normalised_corpus, f"not verbatim in the KB: {norm}"
        checked += 1
    assert checked >= 20


# ── the two duplicate-detection hazards ──────────────────────────────────────

def test_the_two_line_effect_form_is_recognised_as_a_duplicate():
    """`_strip_stv` was anchored to the end of a LINE and matched two floats.

    The canonical form is neither: it wraps, and its confidence slot is a
    nested `(evidence-confidence …)` lookup. Both halves missed, so
    re-extracting a bridge the KB already holds looked net-new.
    """
    bridge = ("(Effect CellularSenescence SASP Pos\n"
              "   (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))")
    existing = _build_kb_form_set()
    forms = expander._expression_forms(bridge)
    assert any(form in existing for form in forms), forms

    # A different strength for the same link is still the same link.
    retuned = ("(Effect CellularSenescence SASP Pos\n"
               "   (stv 0.42 (evidence-confidence InVitro)))")
    assert any(form in existing for form in expander._expression_forms(retuned))

    # And the stripper itself is indifferent to how the STV is written.
    assert expander._strip_stv("(Effect A B Pos (stv 0.5 0.5))") == "(Effect A B Pos"
    assert expander._strip_stv(
        "(Effect A B Pos (stv 0.5 (evidence-confidence InVitro)))"
    ) == "(Effect A B Pos"


def _build_kb_form_set() -> set[str]:
    return expander._build_normalised_set(expander._load_all_raw_content())


@pytest.mark.parametrize("name, where", [
    ("MechanisticConsensus",
     "mentioned only in mechanistic_bridges.metta's header comment, as a tier "
     "that does NOT exist yet"),
    ("Ascorbic_acid",
     "present only in the 107 KB drugage_etl_short.metta dump the runtime "
     "excludes"),
])
def test_a_name_seen_only_in_prose_or_in_the_excluded_dump_is_not_a_duplicate(
    name, where, inventory
):
    """The old fallback regex-searched 290 KB of RAW text, comments included."""
    all_raw = expander._load_all_raw_content()
    # The hazard is real: the bare word IS in the raw text.
    assert re.search(rf"\b{re.escape(name)}\b", all_raw), where
    # ...and the runtime knows no such atom, so it must not count as existing.
    assert not inventory.knows_symbol(name)

    entry = expander.ExtractedEntry(
        kind="type", name=name, metta=f"(Inheritance {name} Supplement)",
        description="",
    )
    assert not expander._is_duplicate(
        entry, api_module.BUILTIN_REGISTRY, all_raw,
        _build_kb_form_set(), inventory,
    )


def test_a_symbol_the_runtime_really_holds_is_still_a_duplicate(inventory):
    entry = expander.ExtractedEntry(
        kind="type", name="Spermidine", metta="(Inheritance Spermidine Supplement)",
        description="",
    )
    assert expander._is_duplicate(
        entry, api_module.BUILTIN_REGISTRY, expander._load_all_raw_content(),
        _build_kb_form_set(), inventory,
    )


# ── the pipeline end to end, and the HTTP surface ────────────────────────────

GOOD_ENTRIES = [
    {"kind": "publication", "name": "SinghEtAl2023_Taurine",
     "metta": '(: SinghEtAl2023_Taurine Publication)\n'
              f'(DOI SinghEtAl2023_Taurine "{TAURINE_DOI}")',
     "description": "Source publication.",
     "identifier": f"PMID:{TAURINE_PMID}"},
    {"kind": "fact", "name": "Singh2023_TaurineRow_1",
     "metta": "(InstanceOf Singh2023_TaurineRow_1 Experiment)\n"
              "(UsesIntervention Singh2023_TaurineRow_1 Taurine)\n"
              "(UsesSpecies Singh2023_TaurineRow_1 Mus_musculus)\n"
              "(AvgLifespanChangePercent Singh2023_TaurineRow_1 11.0)\n"
              "(AvgLifespanSignificance Singh2023_TaurineRow_1 Significant)",
     "description": "Median lifespan +11% in mice.",
     "identifier": f"PMID:{TAURINE_PMID}",
     "evidence_tier": "AnimalStudies_Single", "effect_size_pct": 11.0},
    # The prompt forbids a two-float truth value, so a well-behaved extractor
    # writes the lookup — but it still GUESSES the strength (0.9), and the
    # pipeline replaces that guess with the derived transform of the +11%.
    {"kind": "effect", "name": "TaurineLifespan",
     "metta": "(Effect Taurine Lifespan Pos\n"
              "   (stv 0.9 (evidence-confidence AnimalStudies_Single)))",
     "description": "Taurine raises lifespan.",
     "identifier": f"PMID:{TAURINE_PMID}",
     "evidence_tier": "AnimalStudies_Single", "effect_size_pct": 11.0},
    {"kind": "predicate", "name": "increases-life-span",
     "metta": THE_EVALUATIONS_BLOCK, "description": "The evaluation's block."},
]


def test_the_pipeline_accepts_the_canonical_shapes_and_refuses_the_rest(monkeypatch):
    result = _run(monkeypatch, GOOD_ENTRIES)
    assert result.ok
    assert [e.name for e in result.new_entries] == [
        "SinghEtAl2023_Taurine", "Singh2023_TaurineRow_1", "TaurineLifespan",
    ]
    assert [r.name for r in result.rejected_entries] == ["increases-life-span"]


def test_the_identifier_lands_in_the_atoms_not_only_in_the_comment(monkeypatch):
    result = _run(monkeypatch, GOOD_ENTRIES)
    row = next(e for e in result.new_entries if e.name == "Singh2023_TaurineRow_1")
    # The extractor never wrote a ReportedIn edge; the pipeline added one.
    assert f"(ReportedIn Singh2023_TaurineRow_1 PMID_{TAURINE_PMID})" in row.metta
    assert row.identifier == f"PMID_{TAURINE_PMID}"
    pub = next(e for e in result.new_entries if e.name == "SinghEtAl2023_Taurine")
    assert pub.doi == TAURINE_DOI


def test_the_generated_block_labels_every_provisional_value(monkeypatch):
    result = _run(monkeypatch, GOOD_ENTRIES)
    block = result.metta_block
    assert "HONESTY CONTRACT" in block
    assert "PROVISIONAL" in block
    assert "(evidence-confidence AnimalStudies_Single)" in block
    # The model guessed a strength of 0.9; the derived transform replaces it,
    # and no two-float truth value reaches the file at all.
    assert repr(lifespan_strength(11.0)) in block
    assert "(stv 0.9 " not in block
    assert not re.search(r"\(stv\s+[\d.]+\s+[\d.]+\)", block)
    effect = next(e for e in result.new_entries if e.name == "TaurineLifespan")
    assert effect.provisional and "evidence_tier" in effect.provisional_fields


def test_the_block_reports_predicates_that_would_land_with_zero_facts(monkeypatch, inventory):
    result = _run(monkeypatch, GOOD_ENTRIES)
    # The DrugAge row predicates are declared but grounded only in the ETL
    # build, which is gitignored — worth telling a reviewer, not an error.
    assert "UsesIntervention" in result.unconsumed_predicates
    assert "Effect" not in result.unconsumed_predicates
    assert unconsumed_predicates(result.metta_block, None) == []


def test_the_http_response_carries_the_refusals(monkeypatch):
    monkeypatch.setattr(expander, "call_extraction_llm", _fake_llm(GOOD_ENTRIES))
    response = _post("/ontology/expand", json={
        "paper_text": "Taurine deficiency as a driver of aging.",
        "new_filename": "taurine_expansion_test",
        "apply": False,
    })
    assert response.status_code == 200, response.text
    body = response.json()

    assert [r["name"] for r in body["rejected_entries"]] == ["increases-life-span"]
    assert "unknown_predicate" in body["rejected_entries"][0]["codes"]
    assert body["rejected_entries"][0]["reasons"]

    effect = next(e for e in body["new_entries"] if e["name"] == "TaurineLifespan")
    assert effect["identifier"] == f"PMID_{TAURINE_PMID}"
    assert effect["evidence_tier"] == "AnimalStudies_Single"
    assert effect["provisional"] is True
    assert effect["notes"]
    assert "UsesIntervention" in body["unconsumed_predicates"]


def test_nothing_is_written_to_disk_by_a_preview(monkeypatch):
    before = sorted(p.name for p in REPO.glob("*.metta"))
    _run(monkeypatch, GOOD_ENTRIES)
    assert sorted(p.name for p in REPO.glob("*.metta")) == before


# ── the proof: the rules consume the canonical block ─────────────────────────

hyperon = pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402

#: Dependency-ordered, matching each file's own header. drugage_calibration is
#: what supplies the Lifespan -> Mortality sign adapter the ranking needs.
KB_FILES = [
    "system_types.metta",
    "logical_predicates.metta",
    "epistemic_calibration.metta",
    "species_taxonomy.metta",
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
    "drugage_calibration.metta",
]

_SIGNED_RE = re.compile(r"\(signed\s+(Pos|Neg)\s+\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)\)")


@pytest.fixture(scope="module")
def kb_with_taurine() -> MeTTa:
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES)
    engine = MeTTa()
    engine.run(text + "\n" + CANONICAL_TAURINE_BLOCK)
    return engine


def _empty(res) -> bool:
    return res == [[]] or all(len(g) == 0 for g in res)


@pytest.mark.slow
def test_the_canonical_block_is_consumed_by_the_rules(kb_with_taurine):
    """The whole point: a block in this schema reaches the inference layer.

    The evaluation's schema returns [[]] on every one of these.
    """
    inferred = kb_with_taurine.run("!(infer &self Taurine Mortality)")
    assert not _empty(inferred), inferred
    signs = {m.group(1) for m in _SIGNED_RE.finditer(str(inferred))}
    # Extending lifespan LOWERS mortality: Neg, which the ranking scores as
    # protective (drugage_calibration.metta §6, the sign trap).
    assert signs == {"Neg"}, inferred

    explained = kb_with_taurine.run("!(explain &self Taurine Mortality)")
    assert "Lifespan" in str(explained) and "Mortality" in str(explained)

    ranked = kb_with_taurine.run("!(rank-interventions &self (Taurine) Mortality)")
    assert "(scored Taurine" in str(ranked), ranked

    hallmarks = kb_with_taurine.run("!(hallmarks-of &self Taurine)")
    assert "CellularSenescence" in str(hallmarks)


@pytest.mark.slow
def test_the_raw_row_lifts_to_the_same_link_the_curated_atom_states(kb_with_taurine):
    """Both provenance paths agree — that is what makes the redundancy honest.

    The block carries BOTH a source-faithful measurement row and a curated
    Effect link. `drugage-effect` lifts the row; the link is stated. They must
    produce the same strength, or the curated number is drifting from the
    paper's.
    """
    lifted = kb_with_taurine.run("!(drugage-effect &self Singh2023_TaurineRow_1)")
    text = str(lifted)
    assert "(Effect Taurine Lifespan Pos" in text, lifted

    match = re.search(r"\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)", text)
    assert match, lifted
    assert float(match.group(1)) == pytest.approx(lifespan_strength(11.0, 20.0), abs=1e-6)
    # Confidence: Mus_musculus -> Vertebrate -> AnimalStudies_Single = 0.50,
    # the same tier the curated link looks up.
    assert float(match.group(2)) == pytest.approx(0.50, abs=1e-6)


@pytest.mark.slow
def test_the_provenance_edge_is_queryable(kb_with_taurine):
    rows = kb_with_taurine.run(f"!(match &self (ReportedIn $r PMID_{TAURINE_PMID}) $r)")
    assert "Singh2023_TaurineRow_1" in str(rows), rows
    doi = kb_with_taurine.run("!(match &self (DOI SinghEtAl2023_Taurine $d) $d)")
    assert TAURINE_DOI in str(doi), doi


@pytest.mark.slow
def test_the_evaluations_own_block_stays_inert(kb_with_taurine):
    """The control. Same engine, the schema the evaluation actually got."""
    engine = MeTTa()
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES)
    engine.run(text + "\n" + THE_EVALUATIONS_BLOCK.replace("Taurine", "Taurine2"))
    assert _empty(engine.run("!(infer &self Taurine2 Mortality)"))
    assert _empty(engine.run("!(explain &self Taurine2 Mortality)"))
    assert _empty(engine.run("!(hallmarks-of &self Taurine2)"))


# ── the closed list is a claim about the repository; check it ────────────────

def test_every_canonical_predicate_is_one_the_repository_actually_reads():
    """A closed list is only useful if each entry earns its place."""
    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(REPO.glob("*.metta"))
        if p.name != "drugage_etl_short.metta"
    )
    for predicate in sorted(CANONICAL_PREDICATES - {":"}):
        assert re.search(rf"\b{re.escape(predicate)}\b", sources), predicate


def test_split_args_backs_the_gate_rather_than_a_second_parser():
    """The gate reuses ontology/inventory.py's parser, not a rival one."""
    expr = "(Effect Taurine Lifespan Pos (stv 0.35 (evidence-confidence InVitro)))"
    parts = split_args(expr[1:-1])
    assert parts[0] == "Effect"
    assert parts[-1] == "(stv 0.35 (evidence-confidence InVitro))"


# ── provenance must be in an atom, not in a comment ──────────────────────────

def test_a_pmid_in_a_comment_does_not_satisfy_the_provenance_rule():
    """The gate's own docstring names this failure, and the gate allowed it.

    `check_entry` searched the entry's RAW text, comments included, so an entry
    whose only identifier was a `;;` header line passed `missing_identifier` —
    the exact case the module lists as one of the four it exists to catch
    ("it recorded no PMID and no DOI in any generated fact, only in a header
    comment"). `;;` lines never reach the space, so nothing in the knowledge
    base would carry the citation.
    """
    from ontology.expansion_schema import check_entry

    body = (
        "(Inheritance Taurine Supplement)\n"
        "(Effect Taurine Lifespan Pos "
        "(stv 0.375 (evidence-confidence AnimalStudies_Single)))"
    )
    commented = ";; Singh 2023 (PMID 37289866) - taurine, mouse lifespan +12%\n" + body
    in_an_atom = commented + "\n(ReportedIn Taurine PMID_37289866)"

    codes = {f.code for f in check_entry(commented)}
    assert "missing_identifier" in codes
    # …and the refusal says where to put it, because "no PMID" is confusing
    # when the author can see a PMID two lines up.
    message = next(
        f.message for f in check_entry(commented) if f.code == "missing_identifier"
    )
    assert "comment" in message.lower()

    assert not check_entry(in_an_atom), "an atom-level PMID must be accepted"


def test_the_write_gate_does_not_accept_a_block_as_its_own_provenance():
    """`identifier_text` is the caller's SOURCE, not the block being written.

    Passing the block as its own identifier_text re-opened the comment hole on
    the /ontology/apply path, where the block is all there is.
    """
    from ontology.write_gate import OntologyWriteRefused, guard_ontology_write
    from config import CUSTOM_ONTOLOGY_DIR

    commented = (
        ";; Singh 2023 (PMID 37289866)\n"
        "(Inheritance Taurine Supplement)\n"
        "(Effect Taurine Lifespan Pos "
        "(stv 0.375 (evidence-confidence AnimalStudies_Single)))"
    )
    target = CUSTOM_ONTOLOGY_DIR / "never_written.metta"
    with pytest.raises(OntologyWriteRefused) as excinfo:
        guard_ontology_write(target, commented, allow_curated=False)
    assert excinfo.value.code == "block_failed_schema_gate"
    assert {f.code for f in excinfo.value.findings} >= {"missing_identifier"}
    assert not target.exists()
