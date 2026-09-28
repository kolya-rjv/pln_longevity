"""Regression tests for shared DrugAge compound-name resolution.

Every case in `test_names_the_api_evaluation_lost` is a name the 2026-09-18 HTTP
API evaluation sent to `POST /drugage/rank` and got back nothing for, although
the compound is in the DrugAge build under another spelling.

Run from the repository root:
    pytest tests/test_compound_names.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from ontology.compound_names import (  # noqa: E402
    CompoundResolver,
    canonical_key,
    loose_key,
    transliterate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_ROWS = FIXTURES / "drugage_real_rows.metta"

# A slice of the REAL DrugAge vocabulary (symbols exactly as drugage_etl.py
# emits them, verified against the regenerated build). Hard-coded rather than
# read from build/ so the suite does not depend on the ETL having been run.
DRUGAGE_VOCAB = [
    "Acarbose",
    "Alpha_ketoglutarate",
    "Alpha_lipoic_acid",
    "Ascorbic_acid",
    "Aspirin",
    "Astaxanthin",
    "Beta_estradiol",
    "Coenzyme_Q10",
    "Cysteine",
    "Cysteine_hydrochloride",
    "D_glucosamine",
    "Epigallocatechin_3_gallate",
    "Ethanol",
    "Fisetin",
    "Green_tea_extract",
    "Lithium_Chloride",
    "Metformin",
    "MitoQ",
    "N_16_alpha_hydroxyestradiol",
    "N_17alphaestradiol",
    "N_acetyl_L_cysteine",
    "N_acetylglucosamine",
    "Nicotinamide",
    "Nicotinamide_adenine_dinucleotide",
    "Nicotinamide_mononucleotide",
    "Nicotinamide_riboside",
    "Quercetin",
    "Rapamycin",
    "Resveratrol",
    "Spermidine",
    "Trolox",
    "Urolithin_A",
    "Urolithin_B",
    "Vitamin_E",
]


@pytest.fixture(scope="module")
def resolver() -> CompoundResolver:
    return CompoundResolver(DRUGAGE_VOCAB)


# ── the reported failures ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query, expected",
    [
        ("sirolimus", "Rapamycin"),                              # INN synonym
        ("Sirolimus", "Rapamycin"),
        ("NMN", "Nicotinamide_mononucleotide"),                  # abbreviation
        ("nmn", "Nicotinamide_mononucleotide"),
        ("NAD+", "Nicotinamide_adenine_dinucleotide"),           # abbreviation + charge
        ("EGCG", "Epigallocatechin_3_gallate"),                  # abbreviation
        ("epigallocatechin gallate", "Epigallocatechin_3_gallate"),
        ("NAC", "N_acetyl_L_cysteine"),                          # abbreviation
        ("N-acetylcysteine", "N_acetyl_L_cysteine"),             # stereo descriptor
        ("17-alpha-estradiol", "N_17alphaestradiol"),            # ETL symbol artefact
        ("17alpha-estradiol", "N_17alphaestradiol"),
        ("17a-estradiol", "N_17alphaestradiol"),
        ("17-α-estradiol", "N_17alphaestradiol"),           # Greek letter
    ],
)
def test_names_the_api_evaluation_lost(resolver, query, expected):
    result = resolver.resolve(query)
    assert result.matched == expected, result
    assert result.resolved


def test_every_non_literal_match_is_reported(resolver):
    """A caller must be able to tell an exact hit from a judgement call."""
    assert resolver.resolve("Rapamycin").method == "exact"
    assert resolver.resolve("RAPAMYCIN").method == "normalized"
    assert resolver.resolve("sirolimus").method == "synonym"
    assert resolver.resolve("rapamicin").method == "fuzzy"

    for query in ("sirolimus", "rapamicin", "NMN"):
        assert resolver.resolve(query).warning, query
    # An exact/normalized hit is silent — no noise for the common case.
    assert resolver.resolve("Rapamycin").warning is None
    assert resolver.resolve("RAPAMYCIN").warning is None


# ── the other half of the contract: never invent a match ──────────────────────

def test_unknown_name_is_not_guessed_but_gets_directions(resolver):
    result = resolver.resolve("definitelynotacompound")
    assert result.matched is None
    assert result.method == "unmatched"

    hinted = resolver.resolve("estradiol")
    assert hinted.matched is None
    assert "Beta_estradiol" in hinted.suggestions


def test_ambiguous_name_resolves_to_nothing_and_lists_candidates():
    """Two DrugAge symbols sharing a normalised key must not be decided by order."""
    resolver = CompoundResolver(["Vitamin_E", "VitaminE", "Rapamycin"])
    result = resolver.resolve("vitamin e")
    assert result.matched is None
    assert result.method == "ambiguous"
    assert set(result.suggestions) == {"Vitamin_E", "VitaminE"}


def test_typo_correction_requires_a_clear_winner():
    """A near-miss with a close runner-up is suggested, never applied."""
    resolver = CompoundResolver(["ABC23", "ABC26", "Rapamycin"])
    result = resolver.resolve("ABC24")
    assert result.matched is None, result
    assert result.method == "unmatched"
    assert set(result.suggestions) >= {"ABC23", "ABC26"}

    # ...whereas an unambiguous typo is corrected, with a warning.
    corrected = resolver.resolve("rapamicin")
    assert corrected.matched == "Rapamycin"
    assert corrected.method == "fuzzy"
    assert "similarity" in (corrected.warning or "")


def test_a_known_synonym_for_a_compound_drugage_lacks_says_so():
    resolver = CompoundResolver(["Rapamycin", "Quercetin"])
    result = resolver.resolve("Sprycel")      # brand name for dasatinib
    assert result.matched is None
    assert "Dasatinib" in (result.note or "")


# ── key construction ──────────────────────────────────────────────────────────

def test_canonical_key_is_a_superset_of_the_old_normaliser():
    from ontology.drugage_selector import _norm

    for name in DRUGAGE_VOCAB:
        assert canonical_key(name) == _norm(name)


def test_greek_and_unicode_are_transliterated_not_stripped():
    assert canonical_key("17-α-estradiol") == "17alphaestradiol"
    assert canonical_key("β-estradiol") == "betaestradiol"
    assert canonical_key("17‐alpha‐estradiol") == "17alphaestradiol"
    assert "alpha" in transliterate("α-ketoglutarate")


def test_loose_key_drops_stereo_and_salt_but_the_primary_key_does_not(resolver):
    assert loose_key("N-acetyl-L-cysteine") == loose_key("N-acetylcysteine")
    # Cysteine and its hydrochloride share a loose key, so neither resolves
    # through that rung — each still resolves by its own exact name.
    assert resolver.resolve("Cysteine").matched == "Cysteine"
    assert resolver.resolve("Cysteine_hydrochloride").matched == "Cysteine_hydrochloride"


# ── wiring: the ranking endpoint and the translator prompt share the table ────

def test_ranking_resolves_a_synonym_end_to_end():
    pytest.importorskip("hyperon")
    from core.drugage_router import resolve_compounds, route_drugage_ranking

    result = route_drugage_ranking(["sirolimus"], source=REAL_ROWS)
    assert result.status == "ok", result.error
    joined = " ".join(r.atom for r in result.results)
    assert "(scored Rapamycin" in joined
    assert "sirolimus" in joined and "Rapamycin" in joined   # the note explains it

    [resolution] = resolve_compounds(["sirolimus"], source=REAL_ROWS)
    assert resolution.matched == "Rapamycin"
    assert resolution.method == "synonym"


def test_unresolvable_request_reports_instead_of_ranking_nothing():
    pytest.importorskip("hyperon")
    from core.drugage_router import route_drugage_ranking

    result = route_drugage_ranking(["definitelynotacompound"], source=REAL_ROWS)
    assert result.status == "empty"
    joined = " ".join(r.atom for r in result.results)
    assert "definitelynotacompound" in joined


def test_translator_prompt_carries_the_same_alias_table():
    from core.context_builder import build_system_prompt
    from ontology.registry import BUILTIN_REGISTRY

    prompt = build_system_prompt(BUILTIN_REGISTRY, {})
    assert "COMPOUND NAME ALIASES" in prompt
    assert "sirolimus -> Rapamycin" in prompt
    assert "nmn -> Nicotinamide_mononucleotide" in prompt


# ── A family stem is a question, not a compound ──────────────────────────────
# The 2026-09-28 re-test: bare "urolithin" resolved to `Urolithin_D` and bare
# "vitamin" to `Vitamin_E`, both reported as `synonym` at score 1.0 with the
# note "same active moiety". They are different substances, and the resolver
# already had an `ambiguous` rung that should have caught it.
#
# The cause was in `loose_key`, not in the ladder: `d`, `e`, `l` and `s` are
# stereo descriptors, so `Urolithin_D` was stripped to `urolithin` while
# `Urolithin_A` (no stereo letter) stayed `urolithina`. The two never shared a
# loose bucket, so the collision check saw one candidate and matched it.

def test_a_trailing_series_letter_is_not_a_stereo_descriptor():
    """`Urolithin D` and `Vitamin E` keep the letter that names them."""
    assert loose_key("Urolithin_D") != loose_key("urolithin")
    assert loose_key("Vitamin_E") != loose_key("vitamin")
    # Both members of a family stay distinguishable from each other.
    assert loose_key("Urolithin_A") != loose_key("Urolithin_D")


def test_a_leading_or_medial_stereo_descriptor_still_strips():
    """The cases the loose key exists for must keep working."""
    assert loose_key("n-acetylcysteine") == loose_key("N_acetyl_L_cysteine")
    assert loose_key("trans-resveratrol") == loose_key("Resveratrol")
    assert loose_key("glucosamine") == loose_key("D_glucosamine")


def test_a_family_stem_resolves_to_ambiguous_with_its_members():
    resolver = CompoundResolver(
        ["Urolithin_A", "Urolithin_D", "Vitamin_C", "Vitamin_E", "Rapamycin"]
    )

    urolithin = resolver.resolve("urolithin")
    assert urolithin.method == "ambiguous"
    assert urolithin.matched is None
    assert urolithin.suggestions == ["Urolithin_A", "Urolithin_D"]

    vitamin = resolver.resolve("vitamin")
    assert vitamin.method == "ambiguous"
    assert vitamin.suggestions == ["Vitamin_C", "Vitamin_E"]


def test_naming_a_family_member_still_resolves_exactly():
    """Declining the stem must not cost the names that are not ambiguous."""
    resolver = CompoundResolver(
        ["Urolithin_A", "Urolithin_D", "Vitamin_C", "Vitamin_E", "Rapamycin"]
    )
    for query, expected in [
        ("urolithin a", "Urolithin_A"),
        ("Urolithin_D", "Urolithin_D"),
        ("vitamin e", "Vitamin_E"),
        ("rapamycin", "Rapamycin"),
    ]:
        resolution = resolver.resolve(query)
        assert resolution.matched == expected, query
        assert resolution.method in {"exact", "normalized"}, query

    # And a typo is still corrected, at a score that says it was a correction.
    typo = resolver.resolve("rapamicin")
    assert typo.matched == "Rapamycin"
    assert typo.method == "fuzzy"
    assert typo.score < 1.0
