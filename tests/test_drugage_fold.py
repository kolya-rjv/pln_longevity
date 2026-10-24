"""Tests for the paper-level DrugAge fold (gate 6b of the PeTTa migration).

The fold (`pln_chat/ontology/drugage_fold.py`) groups the lifted rows of one compound by direction and by
independent evidence unit (the paper), pools each direction with patham9's `Truth_Revision`, and returns at most
one Neg and one Pos item plus ONE net. The gates, one test group each:

* the arithmetic: the fold equals a hand-typed Truth_Revision over the per-paper items to 1e-12, equals the
  fork simulator, and (when PeTTa runs here, `-m petta`) equals `Truth_Revision` of the vendored library;
* lift fidelity: the lift equals `drugage_scoring.score_row` bit for bit on every row of the build, and
  `fold=False` reproduces the legacy ranking (`drugage_top`) exactly;
* structure: papers, directions, representative rows, stamps, the missing-PMID unit, no row cap;
* the net: one formula, equal to the legacy score for a compound whose evidence is one item;
* the ranking: Ethanol (52 rows, ONE paper) does not rank first with the build absent or present, and does rank
  first when every row is its own evidence, which is why the paper is the unit;
* Resveratrol (170 rows / 37 papers), Metformin (54 / 16), Rapamycin (37 / 18) against an independent oracle.

The independent oracle is `tests/fixtures/drugage_fold/fold_oracle.py` (standard library only, written from the
spec). The fork simulator is `docs/petta_migration/fork_design/sim/fork_sim.py`, used as an oracle and never as
production code. No test calls a model.

    pytest tests/test_drugage_fold.py -q
    pytest tests/test_drugage_fold.py -q -m "not petta"
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

from ontology import drugage_fold as fold_mod  # noqa: E402
from ontology.drugage_fold import (  # noqa: E402
    CONFIDENCE_CAP,
    DEFAULT_ZERO_ROWS,
    ZERO_ROW_CONVENTIONS,
    RevisionUndefined,
    assign_stamp_ids,
    compute_net,
    evidence_unit_of,
    FoldItemError,
    fold_compound,
    fold_items,
    fold_rows,
    fold_truth_values,
    lift_row,
    net_of_signed,
    revision,
    stamp_token,
    top_compounds,
)
from ontology.drugage_scoring import load_knobs, score_row  # noqa: E402
from ontology.drugage_selector import (  # noqa: E402
    MAX_ROWS,
    SAMPLE_DRUGAGE,
    DrugAgeRow,
    _norm,
    load_rows,
    select_rows,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "drugage_fold"
MIX = FIXTURES / "mix_compound.metta"
STRUCT = FIXTURES / "paper_structure.metta"
THREE = FIXTURES / "three_compounds.metta"
EXPECTED = FIXTURES / "expected_three_compounds.json"
FORK_SIM = REPO / "docs" / "petta_migration" / "fork_design" / "sim" / "fork_sim.py"
RESTRICTED = REPO / "vendor" / "pln_fork" / "lib_pln_restricted.metta"
TOL = 1e-12


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ORACLE = _load_module(FIXTURES / "fold_oracle.py", "drugage_fold_oracle")


@pytest.fixture(scope="module")
def sim():
    if not FORK_SIM.exists():
        pytest.skip("the fork simulator is not here")
    return _load_module(FORK_SIM, "fork_sim_for_fold")


@pytest.fixture(scope="session")
def build(tmp_path_factory) -> Path:
    """The full DrugAge ETL output (3,423 rows), regenerated from the committed zip.

    The same bytes `scripts/run_etl.sh` writes to build/drugage_etl.metta, produced here so the test does not
    depend on whether a developer has run the ETL.
    """
    pytest.importorskip("pandas")
    zip_path = REPO / "data" / "drugage" / "dataset.zip"
    if not zip_path.exists():
        pytest.skip("data/drugage/dataset.zip is not here")
    tmp = tmp_path_factory.mktemp("drugage_build")
    csv_path = tmp / "drugage.csv"
    with zipfile.ZipFile(zip_path) as z:
        csv_path.write_bytes(z.read("drugage.csv"))
    etl = _load_module(REPO / "drugage_etl.py", "drugage_etl_for_fold")
    out = tmp / "drugage_etl.metta"
    etl.generate(csv_path, out)
    return out


@pytest.fixture(scope="session")
def build_rows(build):
    return load_rows(build)


@pytest.fixture(scope="session")
def three():
    return load_rows(THREE)


def _by_compound(rows):
    out: dict[str, list[DrugAgeRow]] = {}
    for r in rows:
        out.setdefault(r.compound, []).append(r)
    return out


def _row(rid, compound, pct, sig="Significant", *, pmid=None, species="Mus_musculus", itp=False):
    """A DrugAgeRow built the way the ETL text would parse."""
    block = [f"(InstanceOf {rid} Experiment)", f"(UsesIntervention {rid} {compound})",
             f"(UsesSpecies {rid} {species})"]
    if pmid is not None:
        block.append(f"(ReportedIn {rid} PMID_{pmid})")
    if itp:
        block.append(f"(IsITPStudy {rid})")
    if pct is not None:
        block.append(f"(AvgLifespanChangePercent {rid} {pct})")
    block.append(f"(AvgLifespanSignificance {rid} {sig})")
    return DrugAgeRow(row_id=rid, compound=compound, species=species, is_itp=itp, significance=sig,
                      avg_change=pct, block="\n".join(block))


def _legacy_tuple(e):
    return (e.row.compound, e.score, e.strength, e.confidence, e.sign, e.direction, e.row.row_id, e.tier_category)


# ══════════════════════════════ the arithmetic ═════════════════════════════════

def hand_revision(tvs):
    """Truth_Revision folded over (s, c) pairs, typed out from the formula (PLN book 5.10.2, Formulas.metta):
    weights w = c / (1 - c), strength = sum(w s) / sum(w), confidence = sum(w) / (sum(w) + 1)."""
    ws = [c / (1.0 - c) for _, c in tvs]
    total = sum(ws)
    return (sum(w * s for w, (s, _) in zip(ws, tvs)) / total, total / (total + 1.0))


def test_the_mix_compound_fold_equals_a_hand_typed_truth_revision():
    """The five-row example of the design notes. Four rows extend lifespan, one shortens it; every row is its own
    paper, so the per-paper items are the lifted rows. Typed by hand: the four Neg items are
    (0.5, 0.9) ITP, (2/3, 0.35) worm, (1/3, 0.2) not significant, (0.0, 0.9) ITP null. The design notes file the null
    on the protective side only, which is `zero_rows="protective"`; the default counts it in both directions (next test)."""
    f = fold_compound(load_rows(MIX), zero_rows="protective")
    neg, pos = f.neg, f.pos
    assert neg is not None and pos is not None
    items = [(0.5, 0.9), (40 / 60, 0.35), (10 / 30, 0.2), (0.0, 0.9)]
    s, c = hand_revision(items)
    assert neg.strength == pytest.approx(s, abs=TOL)
    assert neg.native_confidence == pytest.approx(c, abs=TOL)
    assert neg.confidence == pytest.approx(c * 0.9, abs=TOL)
    # the numbers the design notes print for this example
    assert round(neg.strength, 6) == 0.263050
    assert round(neg.native_confidence, 6) == 0.949466
    assert round(neg.confidence, 6) == 0.854519
    # the single shortening row is unchanged by the fold
    assert (pos.strength, pos.native_confidence, pos.confidence) == (0.6, 0.5, 0.45)
    assert neg.units == ("PMID_1001", "PMID_1002", "PMID_1003", "PMID_1004")


def test_the_fold_equals_the_fork_simulator_and_a_closed_form_on_random_item_sets(sim):
    rng = random.Random(20261024)
    for _ in range(300):
        n = rng.randint(2, 40)
        tvs = [(rng.random(), rng.choice([0.08, 0.126, 0.189, 0.21, 0.315, 0.35, 0.5, 0.9, rng.random() * 0.99]))
               for _ in range(n)]
        got = fold_truth_values(tvs)
        acc = tvs[0]
        for tv in tvs[1:]:
            acc = sim.Truth_Revision(acc, tv)
        assert got == acc                      # the same expression order: identical, not just close
        want = hand_revision(tvs)
        assert got[0] == pytest.approx(want[0], abs=TOL)
        assert got[1] == pytest.approx(want[1], abs=TOL)
        shuffled = tvs[:]
        rng.shuffle(shuffled)
        again = fold_truth_values(shuffled)
        assert again[0] == pytest.approx(got[0], abs=TOL) and again[1] == pytest.approx(got[1], abs=TOL)


def test_revision_has_no_result_at_confidence_one_and_the_fold_caps_it(sim):
    """The library gives no result for c = 1.0 (/safe refuses the zero divisor). The fold caps at 0.999999."""
    with pytest.raises(RevisionUndefined):
        revision((0.5, 1.0), (0.4, 0.5))
    with pytest.raises(sim.Fail):
        sim.Truth_Revision((0.5, 1.0), (0.4, 0.5))
    s, c = fold_truth_values([(0.5, 1.0), (0.4, 0.5)])
    assert 0.0 < s < 1.0 and CONFIDENCE_CAP <= c < 1.0
    assert CONFIDENCE_CAP == 0.999999
    # revision of two zero-confidence items has no weight to pool
    with pytest.raises(RevisionUndefined):
        revision((0.5, 0.0), (0.4, 0.0))


def test_one_item_is_returned_unchanged_even_at_confidence_one():
    assert fold_truth_values([(0.37, 1.0)]) == (0.37, 1.0)
    assert fold_truth_values([(0.37, 0.21)]) == (0.37, 0.21)
    with pytest.raises(ValueError):
        fold_truth_values([])


def test_pooling_strengths_commutes_with_the_stored_link_strength():
    """The chainer stores a link strength as q = e + (1 - e) p (e = the fork's 0.02 leak). Revision is a weighted
    mean, so folding the stored q equals storing the folded p: the fold can run on p and the engine can store q."""
    e = 0.02
    rng = random.Random(7)
    for _ in range(200):
        tvs = [(rng.random(), rng.random() * 0.95) for _ in range(rng.randint(2, 30))]
        folded_then_stored = e + (1 - e) * fold_truth_values(tvs)[0]
        stored_then_folded = fold_truth_values([(e + (1 - e) * s, c) for s, c in tvs])[0]
        assert folded_then_stored == pytest.approx(stored_then_folded, abs=1e-15)


# ══════════════════════════════ the lift ═══════════════════════════════════════

def test_the_lift_equals_score_row_on_every_row_of_the_build(build_rows):
    """The fold lifts rows itself (it needs the NATIVE confidence). Bit for bit with the legacy scorer."""
    knobs = load_knobs()
    assert len(build_rows) == 3423
    lifted = 0
    for r in build_rows:
        want = score_row(r, knobs)
        got = lift_row(r, knobs)
        if want is None:
            assert got is None
            continue
        lifted += 1
        assert (got.strength, got.confidence, got.sign, got.score, got.tier_category) == \
               (want.strength, want.confidence, want.sign, want.score, want.tier_category), r.row_id
        assert got.native_confidence * knobs.chain_discount == got.confidence
    assert lifted == 3372


def test_the_lift_equals_score_row_on_the_committed_sample():
    knobs = load_knobs()
    rows = load_rows(SAMPLE_DRUGAGE)
    assert len(rows) == 201
    for r in rows:
        want, got = score_row(r, knobs), lift_row(r, knobs)
        assert (want is None) == (got is None)
        if want is not None:
            assert (got.strength, got.confidence, got.sign, got.score) == (want.strength, want.confidence, want.sign, want.score)


def test_the_oracle_constants_are_the_calibration_the_module_reads():
    k = load_knobs()
    assert ORACLE.HALFSAT == k.halfsat
    assert ORACLE.GATE == k.sig_gates
    assert ORACLE.CHAIN == k.chain_discount
    for category, value in ORACLE.TIER.items():
        assert k.evidence_confidence[category] == value
    assert ORACLE.CLADE_TIER == {c: k.clade_categories[c] for c in ORACLE.CLADE_TIER}


# ══════════════════════════════ fold=False is the legacy ranking ══════════════

LEGACY_CASES = [
    {},
    {"direction": "any"},
    {"direction": "harmful"},
    {"direction": "none"},
    {"species": "Mus_musculus"},
    {"clade": "Invertebrate", "direction": "any"},
    {"itp_only": True, "direction": "any"},
    {"significant_only": True},
    {"min_confidence": 0.4},
    {"species": "Mus_musculus", "itp_only": True, "min_confidence": 0.5},
]


@pytest.mark.parametrize("case", LEGACY_CASES, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "defaults")
def test_fold_false_reproduces_drugage_top_exactly(build, build_rows, case):
    from core.drugage_router import drugage_top
    kwargs = {"n": 5000, **case}
    want = drugage_top(source=build, fold=False, **kwargs)        # the representative-row ranking, whatever the chainer
    got = top_compounds(build_rows, fold=False, **kwargs)
    assert [_legacy_tuple(e) for e in got.entries] == [_legacy_tuple(e) for e in want.entries]
    for field in ("total_compounds", "total_compounds_in_source", "unscorable_compounds", "total_rows",
                  "scored_rows", "unscorable_rows", "filters"):
        assert getattr(got, field) == getattr(want, field), field
    assert want.entries or case.get("direction") == "harmful", "an empty comparison proves nothing"


def test_fold_false_keeps_every_row_as_its_own_item(three):
    f = fold_compound([r for r in three if r.compound == "Rapamycin"], fold=False)
    assert not f.folded and len(f.items) == 37
    for it in f.items:
        row = it.members[0].row
        want = score_row(row)
        assert (it.strength, it.confidence, it.sign, it.score) == (want.strength, want.confidence, want.sign, want.score)
        assert it.units == (f"row:{row.row_id}",) and it.rows_covered == 1
    assert f.net.folded is False


@pytest.mark.parametrize("zero_rows", ["protective", "both", "neither"])
def test_the_eight_row_fixture_is_unchanged_by_the_fold(zero_rows):
    """The fixture every existing DrugAge test is built on: eight rows, eight compounds. A group of one is returned
    unchanged, so the fold must give each compound exactly its legacy score (and the legacy confidence tiers) under every
    null convention. The one ITP null (Resveratrol) is a pool of nulls in either direction: under "both" the harmful pool is
    nothing but a null, pools to strength 0 and is not an item, so the compound keeps the legacy shape, one protective item."""
    rows = load_rows(REPO / "tests" / "fixtures" / "drugage_real_rows.metta")
    assert len(rows) == 8 and len({r.compound for r in rows}) == 8
    folded = fold_rows(rows, zero_rows=zero_rows)
    for r in rows:
        want = score_row(r)
        f = folded[r.compound]
        assert [i.sign for i in f.items] == [want.sign] and f.items[0].pooled is False
        n = f.net
        assert (n.score, n.strength, n.confidence, n.sign, n.direction, n.row.row_id) == \
               (want.score, want.strength, want.confidence, want.sign, want.direction, r.row_id)
    assert folded["Resveratrol"].net.score == 0.0 and folded["Resveratrol"].net.confidence == 0.81      # the ITP null
    top_fold = top_compounds(rows, n=100, direction="any", zero_rows=zero_rows)
    top_legacy = top_compounds(rows, n=100, direction="any", fold=False)
    assert [_legacy_tuple(e) for e in top_fold.entries] == [_legacy_tuple(e) for e in top_legacy.entries]


# ══════════════════════════════ grouping, units, stamps ════════════════════════

def test_the_paper_is_the_unit_and_a_row_without_a_pmid_is_its_own():
    with_pmid = _row("DrugAgeRow_1", "X", 10.0, pmid=4242)
    without = _row("DrugAgeRow_2", "X", 10.0)
    assert evidence_unit_of(with_pmid) == "PMID_4242"
    assert evidence_unit_of(without) == "row:DrugAgeRow_2"
    assert evidence_unit_of(with_pmid, "row") == "row:DrugAgeRow_1"
    with pytest.raises(ValueError):
        evidence_unit_of(with_pmid, "study")


def test_rows_of_one_paper_and_direction_are_one_item_chosen_by_the_representative_policy():
    f = fold_compound(load_rows(STRUCT))
    neg, pos = f.neg, f.pos
    assert f.rows_total == 10 and f.rows_scorable == 9 and f.units == 5     # the unreported-change row is unscorable
    # Neg: PMID_2001 contributes ONE item (the Significant +12 row, not the +5 or the +8), then the ITP male row
    # of PMID_2002, the worm paper, and the yeast row that has no PMID and is its own unit
    assert neg.units == ("PMID_2001", "PMID_2002", "PMID_2003", "row:DrugAgeRow_9109")
    assert [m.row.row_id for m in neg.members] == ["DrugAgeRow_9102", "DrugAgeRow_9104", "DrugAgeRow_9107", "DrugAgeRow_9109"]
    assert neg.rows_covered == 6                       # 3 + 1 + 1 + 1 scorable rows stand behind those four items
    # Pos: the ITP paper's female rows (-6 Significant beats -1 NotSignificant), and the fly paper
    assert pos.units == ("PMID_2002", "PMID_2004")
    assert [m.row.row_id for m in pos.members] == ["DrugAgeRow_9105", "DrugAgeRow_9110"]
    assert pos.rows_covered == 3
    assert len(f.items) == 2 and [i.sign for i in f.items] == ["Neg", "Pos"]


def test_a_paper_that_reports_both_directions_gets_a_distinct_stamp_in_each():
    f = fold_compound(load_rows(STRUCT))
    both = "PMID_2002"
    assert both in f.neg.units and both in f.pos.units
    neg_tok = stamp_token("StructCompound", both, "Neg")
    pos_tok = stamp_token("StructCompound", both, "Pos")
    assert neg_tok != pos_tok and neg_tok in f.neg.stamp and pos_tok in f.pos.stamp
    assert set(f.neg.stamp).isdisjoint(f.pos.stamp)
    ids = assign_stamp_ids([f], first=100)
    assert sorted(ids.values()) == list(range(100, 100 + len(f.neg.stamp) + len(f.pos.stamp)))
    assert ids[neg_tok] != ids[pos_tok]
    with pytest.raises(ValueError):
        assign_stamp_ids([f], first=0)                  # 0 is the intervention premise
    with pytest.raises(ValueError):
        assign_stamp_ids([f, f])                        # a token on two items is refused


def test_provenance_rows_name_the_paper_the_row_and_the_tier():
    f = fold_compound(load_rows(STRUCT))
    prov = f.neg.provenance()
    assert [p["token"] for p in prov] == list(f.neg.stamp)
    assert prov[0]["pmid"] == "PMID_2001" and prov[0]["representative_row"] == "DrugAgeRow_9102"
    assert prov[0]["kind"] == "drugage_paper" and prov[-1]["kind"] == "drugage_row" and prov[-1]["pmid"] is None
    assert prov[1]["tier"] == "ITP_Positive" and prov[1]["is_itp"] is True


def test_every_compound_of_the_build_has_at_most_one_item_per_direction_and_unique_stamps(build_rows):
    folds = fold_rows(build_rows)
    assert len(folds) == 1043
    both = 0
    for f in folds.values():
        signs = [i.sign for i in f.items]
        assert len(signs) == len(set(signs)) <= 2
        assert all(len(i.stamp) == len(i.units) == len(i.members) for i in f.items)
        both += len(signs) == 2
        for it in f.items:
            assert len(set(it.units)) == len(it.units)
            assert it.rows_covered >= len(it.units)
    ids = assign_stamp_ids(folds.values())          # raises when a token is on two items
    assert len(ids) == len(set(ids)) == sum(len(i.stamp) for f in folds.values() for i in f.items)
    assert both > 200, "the build has many mixed-direction compounds; the fold must give each a Pos and a Neg item"
    scorable = [f for f in folds.values() if f.net is not None]
    assert len(scorable) == 1035 and len(folds) - len(scorable) == 8


def test_a_mixed_sign_compound_yields_both_a_pos_and_a_neg_item():
    f = fold_compound(load_rows(MIX), zero_rows="protective")
    assert [i.sign for i in f.items] == ["Neg", "Pos"]
    assert f.neg.pooled and not f.pos.pooled
    g = fold_compound(load_rows(MIX))                     # the default: the ITP null also stands in the harmful pool
    assert [i.sign for i in g.items] == ["Neg", "Pos"] and g.neg.pooled and g.pos.pooled and len(g.pos.units) == 2


def test_input_order_does_not_change_the_result(three):
    rows = [r for r in three if r.compound == "Metformin"]
    base = fold_compound(rows)
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)
    again = fold_compound(shuffled)
    assert [(i.sign, i.strength, i.native_confidence, i.units) for i in again.items] == \
           [(i.sign, i.strength, i.native_confidence, i.units) for i in base.items]
    assert again.net == base.net


def test_bad_input_is_refused():
    a, b = _row("DrugAgeRow_1", "A", 5.0, pmid=1), _row("DrugAgeRow_2", "B", 5.0, pmid=2)
    with pytest.raises(ValueError, match="compounds"):
        fold_compound([a, b])
    with pytest.raises(ValueError, match="repeats"):
        fold_compound([a, a])
    with pytest.raises(ValueError):
        fold_compound([])
    with pytest.raises(ValueError):
        fold_compound([a], evidence_unit="study")
    empty = fold_compound([_row("DrugAgeRow_3", "C", None, pmid=3)])
    assert empty.net is None and empty.items == () and empty.rows_scorable == 0


def test_the_evidence_units_of_an_item_come_in_one_canonical_order():
    """Numeric PMIDs by number, then the units that are not PMIDs by name. The order fixes the left fold, the stamp
    tuple and the provenance rows, whatever order the rows arrive in."""
    rows = [_row("DrugAgeRow_1", "Order", 10.0, pmid=100), _row("DrugAgeRow_2", "Order", 10.0, pmid=9),
            _row("DrugAgeRow_3", "Order", 10.0), _row("DrugAgeRow_4", "Order", 10.0, pmid=20),
            _row("DrugAgeRow_5", "Order", 10.0)]
    f = fold_compound(rows)
    assert f.neg.units == ("PMID_9", "PMID_20", "PMID_100", "row:DrugAgeRow_3", "row:DrugAgeRow_5")
    assert [t.split("|")[1] for t in f.neg.stamp] == list(f.neg.units)
    assert [m.row.row_id for m in f.neg.members] == ["DrugAgeRow_2", "DrugAgeRow_4", "DrugAgeRow_1", "DrugAgeRow_3", "DrugAgeRow_5"]


def test_a_zero_percent_row_is_a_protective_item_with_strength_zero():
    f = fold_compound([_row("DrugAgeRow_1", "Null", 0.0, "NotSignificant", pmid=9, itp=True)], zero_rows="protective")
    assert [i.sign for i in f.items] == ["Neg"] and f.items[0].strength == 0.0
    assert f.net.score == 0.0 and f.net.direction == "no_effect" and f.net.sign == "Neg"


@pytest.mark.parametrize("zero_rows", ["both", "neither"])
def test_a_compound_whose_only_evidence_is_a_null_is_kept_as_one_protective_item_of_strength_zero(zero_rows):
    """Never dropped from a ranking, whatever the convention: a null is a result (score exactly 0.0, no_effect, the legacy
    confidence). Its harmful direction would hold nothing but the null, pool to strength 0 and add nothing: it is not an item."""
    row = _row("DrugAgeRow_1", "Null", 0.0, "NotSignificant", pmid=9, itp=True)
    f = fold_compound([row], zero_rows=zero_rows)
    assert [i.sign for i in f.items] == ["Neg"] and [i.strength for i in f.items] == [0.0]
    assert f.net.score == 0.0 and f.net.direction == "no_effect" and f.net.sign == "Neg"
    assert (f.net.strength, f.net.confidence) == (score_row(row).strength, score_row(row).confidence)


def test_a_direction_that_holds_only_nulls_is_not_an_item_but_the_nulls_still_dilute_the_other_direction():
    """Rapamycin-shaped: real extending papers and one null. Under "both" the null sits in the harmful pool too, which pools to
    strength 0: no harmful item (the NET is the same either way, its term is 0, and a consumer sees one derivation, not two).
    The null has diluted the protective pool, as it does under "protective"."""
    rows = [_row("DrugAgeRow_1", "R", 20.0, "Significant", pmid=1), _row("DrugAgeRow_2", "R", 30.0, "Significant", pmid=2),
            _row("DrugAgeRow_3", "R", 0.0, "NotSignificant", pmid=3)]
    both = fold_compound(rows, zero_rows="both")
    prot = fold_compound(rows, zero_rows="protective")
    drop = fold_compound(rows, zero_rows="neither")
    assert [i.sign for i in both.items] == ["Neg"] and [i.sign for i in prot.items] == ["Neg"] and [i.sign for i in drop.items] == ["Neg"]
    assert both.neg.units == prot.neg.units == ("PMID_1", "PMID_2", "PMID_3") and drop.neg.units == ("PMID_1", "PMID_2")
    assert both.net.score == prot.net.score < drop.net.score and (both.net.papers_pos, both.net.rows_pos) == (0, 0)
    # with a real harmful paper the null dilutes that pool as well
    harm = rows + [_row("DrugAgeRow_4", "R", -25.0, "Significant", pmid=4)]
    g_both, g_prot = fold_compound(harm, zero_rows="both"), fold_compound(harm, zero_rows="protective")
    assert [i.sign for i in g_both.items] == ["Neg", "Pos"] and g_both.pos.units == ("PMID_3", "PMID_4") and g_prot.pos.units == ("PMID_4",)
    assert g_both.pos.strength < g_prot.pos.strength


# ══════════════════════════════ no row cap ═════════════════════════════════════

def test_every_row_as_its_own_unit_gives_the_row_level_figures_of_the_earlier_design(three):
    """An independent cross-check: the earlier design (Design 3, other code) printed these row-level nets."""
    nets = {c: fold_compound([r for r in three if r.compound == c], evidence_unit="row", zero_rows="protective").net.score
            for c in ("Resveratrol", "Metformin")}
    assert nets["Resveratrol"] == pytest.approx(-0.0272, abs=5e-5)
    assert nets["Metformin"] == pytest.approx(-0.0518, abs=5e-5)
    # and the 52 Ethanol arms revised as 52 experiments reach the confidence the paper unit exists to prevent
    eth = fold_compound([r for r in three if r.compound == "Ethanol"], evidence_unit="row")
    w = 0.21 / 0.79                                   # one unreported invertebrate row: tier 0.35 x gate 0.6
    assert eth.neg.native_confidence == pytest.approx(50 * w / (50 * w + 1), abs=TOL)      # 0.9300 for the 50 extending arms
    assert 52 * w / (52 * w + 1) == pytest.approx(0.9325, abs=1e-4)                       # the figure quoted for all 52
    assert len(eth.neg.units) == 50 and len(eth.pos.units) == 2


def test_the_hyperon_era_row_cap_does_not_apply_to_the_fold(three):
    """drugage_selector.MAX_ROWS = 150 is a hyperon safety constant; Resveratrol alone has 170 rows."""
    assert MAX_ROWS == 150
    rows = [r for r in three if r.compound == "Resveratrol"]
    assert len(rows) == 170 > MAX_ROWS
    f = fold_compound(rows, zero_rows="protective")
    assert f.rows_scorable == 170
    assert sum(i.rows_covered for i in f.items) == 170
    assert sum(len(i.units) for i in f.items) == 35 + 8
    nulls = sum(1 for r in rows if r.avg_change == 0)
    g = fold_compound(rows)                                  # the default: each of the 23 nulls also counts on the harmful side
    assert nulls == 23 and g.rows_scorable == 170 and sum(i.rows_covered for i in g.items) == 170 + nulls
    # a pool far past the cap, one paper per row
    many = [_row(f"DrugAgeRow_{i}", "Big", 10.0 + (i % 7), "Significant", pmid=50000 + i) for i in range(400)]
    big = fold_compound(many)
    assert len(big.neg.units) == 400 and big.neg.rows_covered == 400
    assert big.neg.native_confidence < 1.0
    # the same rows in one paper are ONE item however many there are
    one_paper = fold_compound([_row(f"DrugAgeRow_{i}", "Big", 10.0 + (i % 7), "Significant", pmid=7) for i in range(400)])
    assert len(one_paper.neg.units) == 1 and one_paper.neg.rows_covered == 400
    assert one_paper.neg.native_confidence == 0.5


# ══════════════════════════════ the net ════════════════════════════════════════

def test_the_net_is_the_sum_of_the_signed_item_scores_typed_by_hand():
    f = fold_compound(load_rows(MIX), zero_rows="protective")
    s_n, c_n = hand_revision([(0.5, 0.9), (40 / 60, 0.35), (10 / 30, 0.2), (0.0, 0.9)])
    c_n *= 0.9
    want = s_n * c_n - 0.6 * 0.45
    assert f.net.score == pytest.approx(want, abs=TOL)
    assert round(f.net.score, 4) == -0.0452
    assert f.net.sign == "Pos" and f.net.direction == "harmful"
    # the confidence is the one of the item that carries the sign, the strength is what makes score = s * c
    assert f.net.confidence == f.pos.confidence
    assert f.net.strength * f.net.confidence == pytest.approx(abs(f.net.score), abs=TOL)
    assert f.net.row.row_id == "DrugAgeRow_9005"
    assert (f.net.papers_neg, f.net.papers_pos, f.net.rows_neg, f.net.rows_pos) == (4, 1, 4, 1)


def test_the_net_with_the_null_in_both_pools_typed_by_hand_and_the_convention_decides_the_sign():
    """MixCompound: the ITP null (0.0, 0.9) is one of the four protective items and, by default, also one of the two
    harmful ones next to the single shortening row (0.6, 0.5). The null halves the harmful strength exactly as it halves
    the protective one, so the NET turns from -0.0452 (the null on one side) to a protective one: the sign of this
    compound depends on where a reported null is allowed to count, not on the evidence."""
    f = fold_compound(load_rows(MIX))
    s_n, c_n = hand_revision([(0.5, 0.9), (40 / 60, 0.35), (10 / 30, 0.2), (0.0, 0.9)])
    s_p, c_p = hand_revision([(0.6, 0.5), (0.0, 0.9)])
    want = s_n * (c_n * 0.9) - s_p * (c_p * 0.9)
    assert f.net.score == pytest.approx(want, abs=TOL) and want > 0
    assert f.net.sign == "Neg" and f.net.direction == "protective"
    assert (f.neg.rows_covered, f.pos.rows_covered) == (4, 2) and f.pos.units == ("PMID_1004", "PMID_1005")
    # dropped from the pools it is neither side's evidence: Neg = the three real extending rows, Pos = the one shortening row
    n = fold_compound(load_rows(MIX), zero_rows="neither")
    s3, c3 = hand_revision([(0.5, 0.9), (40 / 60, 0.35), (10 / 30, 0.2)])
    assert n.net.score == pytest.approx(s3 * (c3 * 0.9) - 0.6 * 0.45, abs=TOL)
    signs = {z: fold_compound(load_rows(MIX), zero_rows=z).net.sign for z in ZERO_ROW_CONVENTIONS}
    assert signs == {"both": "Neg", "neither": "Neg", "protective": "Pos"}


def test_a_symmetric_example_keeps_its_sign_whatever_the_nulls_do():
    """Two papers of equal strength and confidence, one extending and one shortening, and a reported null: with the null on
    one side only the net tips toward harm; counted on both sides (or on neither) it stays exactly 0."""
    rows = [_row("DrugAgeRow_1", "Sym", 20.0, "Significant", pmid=1), _row("DrugAgeRow_2", "Sym", -20.0, "Significant", pmid=2),
            _row("DrugAgeRow_3", "Sym", 0.0, "NotSignificant", pmid=3)]
    nets = {z: fold_compound(rows, zero_rows=z).net.score for z in ZERO_ROW_CONVENTIONS}
    assert nets["both"] == 0.0 and nets["neither"] == 0.0
    assert nets["protective"] < 0.0
    sym = fold_compound(rows[:2], zero_rows="neither").net
    assert sym.score == 0.0 and sym.direction == "no_effect"
    # counted on both sides, the null enters each pool with the same weight, so the two pooled items stay symmetric
    both = fold_compound(rows, zero_rows="both")
    assert both.neg.strength == both.pos.strength and both.neg.confidence == both.pos.confidence and both.net.score == 0.0


def test_the_net_of_one_item_is_exactly_the_legacy_score():
    """The justification of the formula: it is the legacy `RowScore.score` summed over the items."""
    for pct, sig, itp, species in [(26.0, "Significant", True, "Mus_musculus"), (-8.0, "Significant", False, "Mus_musculus"),
                                   (3.0, "NotSignificant", False, "Caenorhabditis_elegans"), (0.0, "NotSignificant", True, "Mus_musculus"),
                                   (81.29, "Unreported", False, "Drosophila_melanogaster")]:
        row = _row("DrugAgeRow_1", "One", pct, sig, pmid=1, itp=itp, species=species)
        legacy = score_row(row)
        net = fold_compound([row]).net
        assert (net.score, net.strength, net.confidence, net.sign) == (legacy.score, legacy.strength, legacy.confidence, legacy.sign)
        assert net.direction == legacy.direction
        assert repr(net.score) == repr(legacy.score)           # not even a -0.0 versus 0.0 difference


def test_the_net_equals_the_legacy_score_wherever_the_evidence_is_one_item(build_rows):
    """Every compound whose scorable rows sit in ONE (paper, direction) group: the pooled item IS the legacy
    representative, so the net must be the legacy score bit for bit, label included."""
    legacy = {c: f.net for c, f in fold_rows(build_rows, fold=False).items() if f.net is not None}
    checked = single_row = 0
    for compound, f in fold_rows(build_rows).items():
        if f.net is None or len(f.items) != 1 or f.items[0].pooled:
            continue
        item = f.items[0]
        # one item from ONE paper; the legacy representative is picked from all scorable rows, so compare only when
        # the item covers every scorable row of the compound
        if len(item.members) != 1 or item.rows_covered != f.rows_scorable:
            continue
        want = legacy[compound]
        got = f.net
        assert (got.score, got.strength, got.confidence, got.sign, got.direction, got.row.row_id) == \
               (want.score, want.strength, want.confidence, want.sign, want.direction, want.row.row_id), compound
        checked += 1
        single_row += f.rows_scorable == 1
    assert checked > 600 and single_row > 400, (checked, single_row)


def test_the_net_is_antisymmetric_and_contradicting_papers_cancel():
    up = [_row(f"DrugAgeRow_{i}", "Flip", 20.0, "Significant", pmid=100 + i) for i in range(4)]
    down = [_row(f"DrugAgeRow_{10 + i}", "Flip", -20.0, "Significant", pmid=200 + i) for i in range(4)]
    balanced = fold_compound(up + down)
    assert balanced.net.score == 0.0 and balanced.net.direction == "no_effect"
    assert balanced.neg.confidence == balanced.pos.confidence       # not raised by the contradiction
    only_up = fold_compound(up).net
    only_down = fold_compound(down).net
    assert only_up.score == -only_down.score > 0
    # more papers on one side wins, and the score does not become the revision of the two directions
    lopsided = fold_compound(up + down[:1])
    assert lopsided.net.score > 0 and lopsided.net.direction == "protective"
    assert lopsided.net.score == pytest.approx(lopsided.neg.score + lopsided.pos.score, abs=TOL)


def test_the_net_label_follows_the_legacy_rule():
    not_sig = fold_compound([_row("DrugAgeRow_1", "N", 5.0, "NotSignificant", pmid=1),
                             _row("DrugAgeRow_2", "N", 6.0, "NotSignificant", pmid=2)]).net
    assert not_sig.score > 0 and not_sig.direction == "no_effect"        # nothing in it claims significance
    sig = fold_compound([_row("DrugAgeRow_1", "N", 5.0, "NotSignificant", pmid=1),
                         _row("DrugAgeRow_2", "N", 6.0, "Significant", pmid=2)]).net
    assert sig.direction == "protective"


def test_net_of_signed_is_the_same_net_for_callers_that_hold_derivations(build_rows):
    """A ranking in the engine has (sign, strength, confidence) derivations, not rows. Same formula, same bits."""
    assert net_of_signed([("Neg", 0.5, 0.8)]) == (0.5 * 0.8, "Neg", 0.5, 0.8)
    assert net_of_signed([("Pos", 0.5, 0.8)]) == (-(0.5 * 0.8), "Pos", 0.5, 0.8)
    score, sign, strength, confidence = net_of_signed([("Pos", 0.6, 0.45), ("Neg", 0.3, 0.85)])      # the order is free
    assert score == pytest.approx(0.3 * 0.85 - 0.6 * 0.45, abs=TOL) and sign == "Pos" and confidence == 0.45
    assert strength * confidence == pytest.approx(abs(score), abs=TOL)
    for bad in ([], [("Up", 0.1, 0.1)], [("Neg", 0.1, 0.1), ("Neg", 0.2, 0.2)]):
        with pytest.raises(ValueError):
            net_of_signed(bad)
    for f in fold_rows(build_rows).values():
        if f.net is None:
            continue
        assert net_of_signed([(i.sign, i.strength, i.confidence) for i in f.items]) == \
               (f.net.score, f.net.sign, f.net.strength, f.net.confidence), f.compound


def test_compute_net_takes_one_item_per_direction():
    f = fold_compound(load_rows(MIX))
    with pytest.raises(ValueError):
        compute_net([f.neg, f.neg])
    with pytest.raises(ValueError):
        compute_net([])


# ══════════════════════════════ the ranking, and Ethanol ═══════════════════════

def _rank(top, compound):
    return next(i for i, e in enumerate(top.entries, 1) if e.compound == compound)


def test_ethanol_is_52_rows_of_one_paper(three):
    rows = [r for r in three if r.compound == "Ethanol"]
    assert len(rows) == 52 and len({r.pmid for r in rows}) == 1
    f = fold_compound(rows)
    assert f.units == 1 and f.rows_scorable == 52
    assert f.neg.units == f.pos.units == (rows[0].pmid,)
    # ONE paper: the confidence is the tier x gate of a single unreported invertebrate row, not 0.93
    assert f.neg.native_confidence == pytest.approx(0.35 * 0.6, abs=TOL)
    assert f.neg.confidence == pytest.approx(0.35 * 0.6 * 0.9, abs=TOL)
    revised_as_52_papers = fold_compound(rows, evidence_unit="row")
    assert revised_as_52_papers.neg.native_confidence > 0.93


@pytest.mark.parametrize("source", ["sample", "build"])
def test_ethanol_does_not_rank_first_under_the_fold_with_the_etl_build_absent_or_present(source, request):
    """`drugage_top` reads build/drugage_etl.metta when it exists and the committed 201-row sample otherwise; Ethanol's
    52 rows are in both. Counting every row as independent evidence puts it first in both (the failure this fold
    exists to prevent); counting the paper does not."""
    rows = load_rows(SAMPLE_DRUGAGE) if source == "sample" else request.getfixturevalue("build_rows")
    per_row = top_compounds(rows, n=10_000, direction="any", evidence_unit="row")
    per_paper = top_compounds(rows, n=10_000, direction="any", evidence_unit="paper")
    legacy = top_compounds(rows, n=10_000, direction="any", fold=False)
    assert per_row.entries[0].compound == "Ethanol", "the control: row-level revision must reproduce the failure"
    assert per_paper.entries[0].compound != "Ethanol"
    assert _rank(per_paper, "Ethanol") > 1 and _rank(legacy, "Ethanol") > 1
    protective = top_compounds(rows, n=10_000, fold=True)
    assert "Ethanol" not in [e.compound for e in protective.entries[:3]]
    if source == "build":
        assert _rank(per_row, "Ethanol") == 1
        assert _rank(per_paper, "Ethanol") > 300           # measured 420 of 1,035 (legacy 259)


def test_the_ranking_is_sorted_and_the_direction_filters_partition_it(build_rows):
    everything = top_compounds(build_rows, n=10_000, direction="any")
    scores = [e.score for e in everything.entries]
    assert scores == sorted(scores, reverse=True)
    parts = {d: top_compounds(build_rows, n=10_000, direction=d) for d in ("protective", "harmful", "none")}
    assert sum(len(p.entries) for p in parts.values()) == len(everything.entries) == 1035
    assert all(e.score > 0 for e in parts["protective"].entries)
    assert all(e.score < 0 for e in parts["harmful"].entries)
    assert parts["harmful"].entries[0].score == min(scores)
    assert everything.total_compounds == 1035 and everything.unscorable_compounds == 8


def test_the_fold_ranking_filters_pool_only_the_rows_that_pass_them(build_rows):
    mice = top_compounds(build_rows, n=10_000, direction="any", species="Mus_musculus")
    assert mice.total_compounds < 1035
    assert mice.entries and all(_norm(e.row.species) == _norm("Mus_musculus") for e in mice.entries)
    itp = top_compounds(build_rows, n=10_000, direction="any", itp_only=True)
    assert itp.entries and all(e.row.is_itp for e in itp.entries)
    strong = top_compounds(build_rows, n=10_000, direction="any", min_confidence=0.6)
    assert strong.entries and all(e.confidence >= 0.6 for e in strong.entries)


# ══════════════════════════════ Resveratrol, Metformin, Rapamycin ══════════════

def test_the_three_compound_fixture_is_what_the_etl_writes(build_rows, three):
    want = {r.row_id: r.block for r in build_rows if r.compound in {"Ethanol", "Resveratrol", "Metformin", "Rapamycin"}}
    assert {r.row_id: r.block for r in three} == want
    assert len(three) == 52 + 170 + 54 + 37


@pytest.mark.parametrize("compound,rows,papers", [("Ethanol", 52, 1), ("Resveratrol", 170, 37),
                                                  ("Metformin", 54, 16), ("Rapamycin", 37, 18)])
def test_the_fold_equals_the_independent_oracle_on_the_three_compounds(three, compound, rows, papers):
    """The recorded snapshot of the oracle (expected_three_compounds.json) is the earlier convention, the null on the
    protective side only; the live oracle below covers the other two."""
    expected = json.loads(EXPECTED.read_text())[compound]
    assert (expected["rows"], expected["papers"]) == (rows, papers)
    f = fold_compound([r for r in three if r.compound == compound], zero_rows="protective")
    assert f.rows_total == rows and f.units == papers
    for sign, item in (("Neg", f.neg), ("Pos", f.pos)):
        want = expected[sign]
        if want is None:
            assert item is None
            continue
        assert item.strength == pytest.approx(want["strength"], abs=TOL)
        assert item.native_confidence == pytest.approx(want["native"], abs=TOL)
        assert item.confidence == pytest.approx(want["confidence"], abs=TOL)
        assert sorted(item.units) == sorted(want["papers"])
        assert sorted(m.row.row_id for m in item.members) == sorted(want["members"])
        assert item.rows_covered == want["rows"]
    assert f.net.score == pytest.approx(expected["net"], abs=TOL)


@pytest.mark.parametrize("zero_rows", ZERO_ROW_CONVENTIONS)
def test_the_fold_equals_the_independent_oracle_on_every_compound_of_the_build(build_rows, zero_rows):
    knobs = load_knobs()
    by_compound = _by_compound(build_rows)
    folds = fold_rows(build_rows, zero_rows=zero_rows)
    compared = 0
    for compound, rows in by_compound.items():
        plain = [{"row_id": r.row_id, "compound": r.compound, "clade": knobs.species_clade.get(r.species or ""),
                  "itp": r.is_itp, "sig": r.significance or "Unreported", "pct": r.avg_change, "pmid": r.pmid} for r in rows]
        want = ORACLE.fold(plain, zero_rows=zero_rows)
        f = folds[compound]
        if f.net is None:
            assert set(want) == {"net"}
            continue
        for sign, item in (("Neg", f.neg), ("Pos", f.pos)):
            if sign not in want:
                assert item is None, (compound, sign)
                continue
            assert item is not None, (compound, sign)
            assert item.strength == pytest.approx(want[sign]["strength"], abs=TOL), (compound, sign)
            assert item.confidence == pytest.approx(want[sign]["confidence"], abs=TOL), (compound, sign)
            assert sorted(item.units) == sorted(want[sign]["papers"]), (compound, sign)
            assert sorted(m.row.row_id for m in item.members) == sorted(want[sign]["members"]), (compound, sign)
            assert item.rows_covered == want[sign]["rows"], (compound, sign)
        assert f.net.score == pytest.approx(want["net"], abs=TOL), compound
        compared += 1
    assert compared == 1035


def test_resveratrol_and_metformin_leave_exactly_zero_and_the_sign_depends_on_the_null_convention(three):
    """The before/after: the legacy representative of Resveratrol and Metformin is an ITP null (score exactly 0.0). Under the
    fold the NET of the two is small, and its SIGN is the convention's: with the nulls on the protective side only (the
    earlier behaviour) both are harmful (-0.0460, -0.0570); with the nulls counted on both sides, or on neither, both are
    protective (+0.1363/+0.1182, +0.0321/+0.0629). Rapamycin and Ethanol, whose nulls are one row of 37 and none, do not move.
    That the sign of the two headline compounds is decided by the convention and not by the evidence is the finding the
    default (both sides) answers."""
    legacy = {c: fold_compound([r for r in three if r.compound == c], fold=False).net for c in ("Resveratrol", "Metformin", "Rapamycin")}
    assert legacy["Resveratrol"].score == 0.0 and legacy["Metformin"].score == 0.0
    assert legacy["Rapamycin"].score == pytest.approx(0.3190909, abs=1e-6)
    table = {z: {c: fold_compound([r for r in three if r.compound == c], zero_rows=z).net for c in ("Resveratrol", "Metformin", "Rapamycin", "Ethanol")}
             for z in ZERO_ROW_CONVENTIONS}
    want = {"protective": {"Resveratrol": -0.0460, "Metformin": -0.0570, "Rapamycin": 0.3368, "Ethanol": 0.1144},
            "both": {"Resveratrol": 0.1363, "Metformin": 0.1182, "Rapamycin": 0.3368, "Ethanol": 0.1144},
            "neither": {"Resveratrol": 0.0321, "Metformin": 0.0629, "Rapamycin": 0.3368, "Ethanol": 0.1144}}
    for z, row in want.items():
        for c, score in row.items():
            assert table[z][c].score == pytest.approx(score, abs=5e-5), (z, c)
    assert table["protective"]["Resveratrol"].direction == "harmful" and table["protective"]["Metformin"].direction == "harmful"
    for z in ("both", "neither"):
        assert table[z]["Resveratrol"].direction == "protective" and table[z]["Metformin"].direction == "protective"
    assert DEFAULT_ZERO_ROWS == "both"
    folded = table["both"]
    assert folded["Rapamycin"].direction == "protective"
    assert folded["Rapamycin"].papers_neg == 18 and folded["Rapamycin"].papers_pos == 0     # its one null pools in a harmful direction that holds nothing else: no item


def test_the_whole_build_ranking_moves_by_the_measured_amount(build_rows):
    """A pin of the CURRENT calibration (halfsat 20, the tier and gate constants), so a tuned constant fails here with
    the numbers the notes quote. Not a property of the fold."""
    legacy = top_compounds(build_rows, n=20, fold=False)
    folded = top_compounds(build_rows, n=20, fold=True, zero_rows="protective")
    legacy_names = [e.compound for e in legacy.entries]
    fold_names = [e.compound for e in folded.entries]
    assert len(set(legacy_names) & set(fold_names)) == 9              # 11 of the legacy top 20 leave the top 20
    assert sum(a == b for a, b in zip(legacy_names, fold_names)) == 0   # and no compound keeps its place
    assert legacy_names[:3] == ["C60_fullerene", "Rapamycin", "Astaxanthin"]
    assert fold_names[:3] == ["Spermidine", "N_17alphaestradiol", "Rapamycin"]
    assert folded.total_compounds == 1035


# ══════════════════════════════ fold_items: the chainer's interface ═══════════

def _items_from_rows(rows, *, policy_keys=True):
    """The lifted items core.pln_chainer builds: one per scorable row, the LIFESPAN direction, the native confidence."""
    out = []
    for r in rows:
        lr = lift_row(r)
        if lr is None:
            continue
        item = {"compound": r.compound, "target": "Lifespan", "direction": "Pos" if lr.sign == "Neg" else "Neg",
                "strength": lr.strength, "confidence": lr.native_confidence, "rows": (r.row_id,), "papers": (r.pmid or "",)}
        if policy_keys:
            item.update(change=r.avg_change, significance=r.significance, itp=r.is_itp)
        out.append(item)
    return out


def test_fold_items_equals_fold_compound_on_every_compound_of_the_build(build_rows):
    folds = fold_rows(build_rows)
    out = fold_items(_items_from_rows(build_rows))
    assert {o["within_paper_pick"] for o in out} == {"representative_row_policy"}
    by_key = {(o["compound"], o["mortality_sign"]): o for o in out}
    want_items = [(f.compound, i) for f in folds.values() for i in f.items]
    assert len(out) == len(want_items) == len(by_key)
    for compound, it in want_items:
        o = by_key[(compound, it.sign)]
        assert o["direction"] == ("Pos" if it.sign == "Neg" else "Neg")          # the lifespan direction is the flip
        assert o["target"] == "Lifespan"
        assert o["strength"] == it.strength and o["confidence"] == it.native_confidence == o["native_confidence"]
        assert o["chain_confidence"] == it.confidence
        assert o["rows"] == tuple(m.row.row_id for m in it.members)
        assert o["papers"] == tuple(m.row.pmid for m in it.members)
        assert o["units"] == it.units and o["stamp"] == it.stamp and o["pooled"] == it.pooled
        assert o["rows_covered"] == it.rows_covered == len(o["covered_rows"])
        assert o["net"] == folds[compound].net.score and o["net_sign"] == folds[compound].net.sign
        assert o["net_direction"] == folds[compound].net.direction
    assert out == sorted(out, key=lambda d: (d["compound"], d["target"], d["direction"]))


def test_fold_items_without_the_policy_keys_falls_back_and_says_so(three):
    """The chainer hands over only the seven keys. The most confident row of a paper then stands for it, which is the
    pick of the earlier design, so its printed figures (-0.0395, -0.0403, +0.3167) are an independent check."""
    out = fold_items(_items_from_rows(three, policy_keys=False), zero_rows="protective")
    assert {o["within_paper_pick"] for o in out} == {"most_confident_row"}
    net = {}
    for o in out:
        net[o["compound"]] = o["net"]
    assert net["Resveratrol"] == pytest.approx(-0.0395, abs=5e-5)
    assert net["Metformin"] == pytest.approx(-0.0403, abs=5e-5)
    assert net["Rapamycin"] == pytest.approx(0.3167, abs=5e-5)
    # the confidences and weights do not depend on the pick, only which row of a paper is used
    exact = {(o["compound"], o["mortality_sign"]): o for o in fold_items(_items_from_rows(three), zero_rows="protective")}
    for o in out:
        e = exact[(o["compound"], o["mortality_sign"])]
        assert o["units"] == e["units"] and o["rows_covered"] == e["rows_covered"]


def test_the_two_picks_agree_wherever_a_paper_has_one_row_per_direction(build_rows):
    """The fallback and the repo's policy differ only inside a paper; where no paper has two rows of one direction they
    give the same bits."""
    by = _by_compound(build_rows)
    agree = 0
    for compound, rows in by.items():
        counts: dict[tuple, int] = {}
        for r in rows:
            if r.scorable:
                key = (r.pmid, r.avg_change < 0)
                counts[key] = counts.get(key, 0) + 1
        if not counts or max(counts.values()) > 1:
            continue
        a = fold_items(_items_from_rows(rows), zero_rows="protective")
        b = fold_items(_items_from_rows(rows, policy_keys=False), zero_rows="protective")
        strip = lambda xs: [{k: v for k, v in x.items() if k not in ("within_paper_pick", "net_direction")} for x in xs]
        assert strip(a) == strip(b), compound
        assert all(x["net_direction"] is None for x in b)       # the fallback does not know the significance
        agree += 1
    assert agree > 400


def test_fold_items_false_is_the_identity_the_legacy_lane_gets(three):
    items = _items_from_rows([r for r in three if r.compound == "Rapamycin"])
    assert fold_items(items, fold=False) == items
    assert fold_items([], fold=True) == []


def test_fold_items_keeps_two_compounds_and_two_targets_apart():
    a = {"compound": "A", "target": "Lifespan", "direction": "Pos", "strength": 0.5, "confidence": 0.5, "rows": ("DrugAgeRow_1",), "papers": ("PMID_1",)}
    b = {**a, "compound": "B", "rows": ("DrugAgeRow_2",)}
    c = {**a, "target": "Other", "rows": ("DrugAgeRow_3",)}
    out = fold_items([b, a, c])
    assert [(o["compound"], o["target"]) for o in out] == [("A", "Lifespan"), ("A", "Other"), ("B", "Lifespan")]
    assert all(not o["pooled"] and o["strength"] == 0.5 and o["confidence"] == 0.5 for o in out)
    # one row of one paper is returned as it came in, and the rows without a PMID carry '' as the chainer expects
    anon = fold_items([{**a, "papers": ("",)}])[0]
    assert anon["papers"] == ("",) and anon["units"] == ("row:DrugAgeRow_1",)


def test_fold_items_refuses_what_it_cannot_read():
    ok = {"compound": "A", "target": "Lifespan", "direction": "Pos", "strength": 0.5, "confidence": 0.5, "rows": ("DrugAgeRow_1",), "papers": ("PMID_1",)}
    for bad, why in [({k: v for k, v in ok.items() if k != "strength"}, "missing"),
                     ({**ok, "direction": "Up"}, "direction"),
                     ({**ok, "rows": ("a", "b")}, "ONE row"),
                     ({**ok, "confidence": 1.5}, "in \\[0, 1\\]"),
                     ({**ok, "strength": float("nan")}, "in \\[0, 1\\]"),
                     ({**ok, "strength": "high"}, "numbers"),
                     ({**ok, "change": -5.0, "significance": "Significant", "itp": False}, "disagrees")]:
        with pytest.raises(FoldItemError, match=why):
            fold_items([bad])
    with pytest.raises(FoldItemError, match="twice"):
        fold_items([ok, ok])
    with pytest.raises(ValueError):
        fold_items([ok], evidence_unit="study")


def test_fold_items_input_order_does_not_matter(three):
    items = _items_from_rows([r for r in three if r.compound == "Metformin"])
    shuffled = items[:]
    random.Random(11).shuffle(shuffled)
    assert fold_items(shuffled) == fold_items(items)


def test_fold_items_speaks_the_chainers_contract_and_is_json_ready(three):
    import json as _json
    try:
        from core.pln_chainer import DRUGAGE_ITEM_KEYS
    except Exception:           # the chainer is another task's file; the shape below is the documented contract
        DRUGAGE_ITEM_KEYS = fold_mod.ITEM_KEYS
    assert tuple(DRUGAGE_ITEM_KEYS) == fold_mod.ITEM_KEYS
    out = fold_items(_items_from_rows(three))
    assert out
    for o in out:
        assert all(k in o for k in DRUGAGE_ITEM_KEYS)
        assert isinstance(o["rows"], tuple) and isinstance(o["papers"], tuple)
        assert len(o["rows"]) == len(o["papers"]) == len(o["units"]) == len(o["stamp"])
    _json.dumps(out)


# ══════════════════════════════ the real PeTTa library ═════════════════════════

def _real_petta_available() -> bool:
    try:
        from core.metta_runtime import PettaRuntime
        rt = PettaRuntime()
        return rt.available() and rt.run_program("", ["!(+ 1 2)"]) == [["3"]]
    except Exception:
        return False


requires_petta = pytest.mark.skipif(not _real_petta_available(), reason="PeTTa is not installed here")


def _stv(text: str) -> tuple[float, float]:
    inner = text.strip()
    assert inner.startswith("(stv ") and inner.endswith(")"), text
    s, c = inner[5:-1].split()
    return float(s), float(c)


def _nested_revision(tvs) -> str:
    expr = f"(stv {tvs[0][0]!r} {min(tvs[0][1], CONFIDENCE_CAP)!r})"
    for s, c in tvs[1:]:
        expr = f"(Truth_Revision {expr} (stv {s!r} {min(c, CONFIDENCE_CAP)!r}))"
    return expr


@pytest.mark.petta
@requires_petta
def test_the_fold_equals_truth_revision_of_the_vendored_library_on_real_papers(three):
    """The fold is a Python port of patham9's Truth_Revision. This runs the REAL library (the restricted artifact,
    through the production adapter's session) on the per-paper items of Resveratrol, Metformin and Rapamycin."""
    from core.metta_runtime import PettaRuntime
    session = PettaRuntime().session(RESTRICTED.read_text(encoding="utf-8"), 300)
    try:
        checked = 0
        for compound in ("Resveratrol", "Metformin", "Rapamycin"):
            f = fold_compound([r for r in three if r.compound == compound])
            for item in f.items:
                if not item.pooled:
                    continue
                tvs = [(m.strength, m.native_confidence) for m in item.members]
                out = session.run("!" + _nested_revision(tvs), 300)
                assert len(out) == 1 and len(out[0]) == 1, (compound, item.sign, out)
                s, c = _stv(out[0][0])
                assert s == pytest.approx(item.strength, abs=TOL), (compound, item.sign)
                assert c == pytest.approx(item.native_confidence, abs=TOL), (compound, item.sign)
                checked += 1
        assert checked == 5                    # Resveratrol Neg+Pos, Metformin Neg+Pos, Rapamycin Neg
        # the cap exists because the library has no result at c = 1.0, and has one at 0.999999
        assert session.run("!(Truth_Revision (stv 0.5 1.0) (stv 0.4 0.5))", 300) == [[]]
        out = session.run(f"!(Truth_Revision (stv 0.5 {CONFIDENCE_CAP!r}) (stv 0.4 0.5))", 300)
        assert len(out[0]) == 1
    finally:
        session.close()


@pytest.mark.petta
@requires_petta
def test_the_leak_the_chainer_must_cancel_is_the_one_the_fold_leaves_alone():
    """The fold never touches the Modus Ponens leak: the chainer stores q = e + (1 - e) p with e read from the
    library. This pins the value the notes assume, from the real library."""
    from core.metta_runtime import PettaRuntime
    session = PettaRuntime().session(RESTRICTED.read_text(encoding="utf-8"), 300)
    try:
        out = session.run("!(Truth_ModusPonens (stv 0.0 1.0) (stv 0.0 1.0))", 300)
        assert _stv(out[0][0]) == (0.02, 1.0)
    finally:
        session.close()


# ══════════════════════════════ housekeeping ═══════════════════════════════════

def test_the_module_is_pure_and_names_its_semantics():
    sem = fold_mod.semantics()
    assert "paper" in sem["evidence_unit"] and sem["confidence_cap"] == CONFIDENCE_CAP
    assert len(sem["policy"]) == 6 and "Truth_Revision" in sem["policy"][3]
    import ast
    src = Path(fold_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"__future__", "dataclasses", "typing", "ontology.drugage_scoring", "ontology.drugage_selector"}, imported
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"open", "print", "exec", "eval"}, called
    # select_rows is called with the group's own size as the limit, never the default cap
    assert "limit=len(rows)" in src


def test_select_rows_keeps_its_signature_the_fold_relies_on():
    rows = load_rows(MIX)
    picked = select_rows(rows, best_per_compound=True, limit=len(rows))
    assert len(picked) == 1


# ══════════════════════════════ review fixes: exactness per compound, the default ranking ═══════════════

def test_one_compounds_missing_policy_keys_never_change_another_compounds_number(three):
    """fold_items used to choose the within-paper pick for the WHOLE request from the least-complete row: one unrelated row
    with no significance atom switched Resveratrol from -0.0460 to -0.0395. The pick is now per (compound, target) group."""
    keyed = _items_from_rows([r for r in three if r.compound in ("Resveratrol", "Metformin")])
    alone = {o["compound"]: o for o in fold_items(keyed, zero_rows="protective")}
    stray = {"compound": "OtherDrug", "target": "Lifespan", "direction": "Pos", "strength": 0.3, "confidence": 0.5,
             "rows": ("DrugAgeRow_X",), "papers": ("PMID_9",), "change": 10.0, "significance": None, "itp": False}
    mixed = fold_items(keyed + [stray], zero_rows="protective")
    for compound in ("Resveratrol", "Metformin"):
        a = [o for o in mixed if o["compound"] == compound]
        b = [o for o in fold_items(keyed, zero_rows="protective") if o["compound"] == compound]
        assert a == b, compound
        assert {o["within_paper_pick"] for o in a} == {"representative_row_policy"}
    other = [o for o in mixed if o["compound"] == "OtherDrug"]
    assert {o["within_paper_pick"] for o in other} == {"most_confident_row"} and other[0]["net_direction"] is None
    assert round(alone["Resveratrol"]["net"], 4) == -0.0460 and round(alone["Metformin"]["net"], 4) == -0.0570


def test_the_whole_build_ranking_under_the_default_convention(build_rows):
    """A pin of the default (nulls on both sides): the numbers the notes quote, so a tuned constant fails here."""
    legacy = top_compounds(build_rows, n=20, fold=False)
    folded = top_compounds(build_rows, n=20, fold=True)
    assert folded.zero_rows == "both" and folded.total_compounds == 1035
    names = [e.compound for e in folded.entries]
    assert len(set(e.compound for e in legacy.entries) & set(names)) == 9
    assert sum(a == b for a, b in zip([e.compound for e in legacy.entries], names)) == 0
    assert names[:5] == ["Spermidine", "N_17alphaestradiol", "Rapamycin", "Reserpine", "C60_fullerene"]
    ranks = {e.compound: i for i, e in enumerate(top_compounds(build_rows, n=2000, direction="any").entries, 1)}
    assert {c: ranks[c] for c in ("Ethanol", "Resveratrol", "Metformin", "Rapamycin", "Spermidine")} == \
        {"Ethanol": 426, "Resveratrol": 322, "Metformin": 406, "Rapamycin": 3, "Spermidine": 1}
    # the convention moves the sign of exactly the compounds the review counted: 2 of 1,035 when the nulls count on both sides
    def signs(top_kwargs):
        t = top_compounds(build_rows, n=2000, direction="any", **top_kwargs)
        return {e.compound: ("harmful" if e.score < 0 else "zero" if e.score == 0 else "protective") for e in t.entries}
    base, both = signs({"zero_rows": "protective"}), signs({"zero_rows": "both"})
    flipped = sorted(c for c in base if base[c] != both[c])
    assert len(base) == 1035 and flipped[:0] == [] and {"Metformin", "Resveratrol"} <= set(flipped) and len(flipped) == 2


def test_a_null_beside_a_real_item_leaves_the_real_items_strength_to_the_bit():
    """`net_of_signed` rebuilds a strength as |score| / confidence only when BOTH parts contribute; a part that contributes exactly
    0 (a reported null, from a caller that holds derivations) must leave the other's own strength, bit for bit, or a compound
    whose only evidence is one item would no longer score exactly what the legacy ranking scores it ((s * c) / c is not s in
    floating point: 0.2 and 0.81 give 0.20000000000000004)."""
    assert (0.2 * 0.81) / 0.81 != 0.2
    inexact = 0
    for s_, c_ in ((0.2, 0.81), (0.375, 0.189), (0.2, 0.18), (0.375, 0.18), (0.2, 0.405)):
        for nul_sign, real_sign in (("Pos", "Neg"), ("Neg", "Pos")):
            for null_conf in (0.18, 0.5, 0.81):
                score, sign, strength, confidence = net_of_signed([(real_sign, s_, c_), (nul_sign, 0.0, null_conf)])
                assert (sign, strength, confidence) == (real_sign, s_, c_) and abs(score) == s_ * c_
                inexact += (abs(score) / c_) != s_
    assert inexact > 0, "the premise: for some inputs the rebuild would NOT be exact"
    # two real parts still rebuild (the strength that makes score = s * c), and a lone part is its own
    score, sign, strength, confidence = net_of_signed([("Neg", 0.5, 0.8), ("Pos", 0.2, 0.5)])
    assert sign == "Neg" and confidence == 0.8 and strength * confidence == pytest.approx(score, abs=1e-15) and strength != 0.5
