"""Tests for the NHANES ETL correctness core (nhanes_common.py) and the XPORT writer.

These are the tests that guard the parts of the NHANES integration where a quiet error
would poison every downstream number, so each one checks against an INDEPENDENT
reference rather than against the implementation's own output:

  * survey-weighted mean / SD — checked against an explicit hand computation, against
    ``numpy.std`` in the equal-weight case (both ddof=0 and ddof=1 variants), and for
    numerical stability at a scale where a one-pass variance would cancel;
  * weighted Kaplan-Meier — checked against a hand-worked product-limit example, for
    horizon truncation, for tied event times, for invariance between frequency weights
    and row-expanded data, and for agreement with a naive proportion in exactly the case
    where the two must agree (nothing censored before the horizon);
  * the pandas IBM-zero bug — that a genuine 0.0 written in SAS XPORT format is read back
    as the 2**-260 artifact by raw pandas, and is repaired to 0.0 by ``read_nhanes``;
  * XPORT v5 round-tripping — values of mixed sign and magnitude survive exactly, SAS
    missing values become NaN, character columns decode;
  * the honesty guards — degenerate input yields None rather than a number, unreliable
    cells are suppressed, a missing variable raises and names the columns that do exist,
    non-finite values are refused before they can reach a KB file, and the atom budget
    trips before hyperon's uncatchable abort threshold.

Unlike the other suites here these need no ``hyperon``: they are pure Python. They do need
``pandas``, which is already a declared repo dependency (pln_chat/requirements.txt).

Run:  pytest tests/ -v       (from the repo root)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

pd = pytest.importorskip("pandas")

from nhanes_common import (  # noqa: E402
    AtomBudgetExceeded,
    IBM_ZERO_ARTIFACT,
    MettaWriter,
    MissingColumns,
    age_band,
    check_symbol,
    fix_ibm_zero,
    mstr,
    num,
    read_nhanes,
    sex_symbol,
    suppressed_reason,
    weighted_kaplan_meier,
    weighted_moments,
)
from nhanes_xport_writer import CHAR, NUM, ieee_to_ibm, write_xport  # noqa: E402

TOL = 1e-12


# ════════════════════════════ weighted moments ════════════════════════════
def test_weighted_mean_and_sd_match_hand_computation():
    values, weights = [1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 4.0]
    total = sum(weights)
    mean = sum(w * v for v, w in zip(values, weights)) / total
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total

    got = weighted_moments(values, weights)
    assert got is not None
    assert got.mean == pytest.approx(mean, rel=TOL)
    assert got.sd == pytest.approx(math.sqrt(var), rel=TOL)
    assert got.n == 4
    assert got.sum_w == pytest.approx(total, rel=TOL)
    assert got.sd_estimator == "population"


def test_equal_weights_reproduce_numpy_population_sd():
    values = [200.0, 180.0, 240.0, 150.0, 310.0]
    got = weighted_moments(values, [1.0] * len(values))
    assert got.sd == pytest.approx(float(np.std(values)), rel=TOL)


def test_n_minus_one_correction_reproduces_numpy_sample_sd():
    values = [200.0, 180.0, 240.0, 150.0, 310.0]
    got = weighted_moments(values, [1.0] * len(values), correction="n_minus_1_corrected")
    assert got.sd == pytest.approx(float(np.std(values, ddof=1)), rel=TOL)
    assert got.sd_estimator == "n_minus_1_corrected"


def test_weighted_sd_is_numerically_stable_at_large_mean():
    """A one-pass E[wx^2]-E[wx]^2 variance cancels catastrophically here."""
    values = [1e8 - 1.0, 1e8, 1e8 + 1.0]
    got = weighted_moments(values, [1.0, 1.0, 1.0])
    assert got.sd == pytest.approx(float(np.std(values)), rel=1e-9)
    assert got.sd > 0.0


def test_zero_spread_gives_zero_sd_not_a_negative_root():
    got = weighted_moments([5.0, 5.0, 5.0], [2.0, 3.0, 4.0])
    assert got.mean == pytest.approx(5.0, rel=TOL)
    assert got.sd == 0.0


def test_weights_are_scale_invariant_for_the_mean():
    values = [1.0, 7.0, 13.0]
    a = weighted_moments(values, [2.0, 4.0, 6.0])
    b = weighted_moments(values, [1.0, 2.0, 3.0])
    assert a.mean == pytest.approx(b.mean, rel=TOL)
    assert a.sd == pytest.approx(b.sd, rel=TOL)


def test_non_positive_and_non_finite_rows_are_dropped():
    got = weighted_moments([1.0, float("nan"), 3.0, 99.0], [1.0, 1.0, 1.0, 0.0])
    assert got.n == 2                      # the NaN value and the zero-weight row are gone


@pytest.mark.parametrize("values,weights", [([], []), ([1.0], [0.0]), ([float("nan")], [1.0])])
def test_degenerate_moments_return_none_never_a_number(values, weights):
    assert weighted_moments(values, weights) is None


def test_moments_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_moments([1.0, 2.0], [1.0])


def test_unknown_sd_correction_is_rejected():
    with pytest.raises(ValueError):
        weighted_moments([1.0, 2.0], [1.0, 1.0], correction="bessel")


# ════════════════════════════ weighted Kaplan-Meier ════════════════════════
def test_km_matches_hand_worked_product_limit():
    # t=2: at risk 5, 1 event -> 4/5 | t=3: at risk 4, 1 event -> 3/4 | t=5: 2, 1 -> 1/2
    times, events = [2, 3, 3, 5, 7], [1, 1, 0, 1, 0]
    got = weighted_kaplan_meier(times, events, [1.0] * 5, horizon=10.0)
    expected_survival = 0.8 * 0.75 * 0.5
    assert got.survival == pytest.approx(expected_survival, rel=TOL)
    assert got.cumulative_incidence == pytest.approx(1.0 - expected_survival, rel=TOL)
    assert got.n == 5 and got.events == 3


def test_km_truncates_at_the_horizon():
    times, events = [2, 3, 3, 5, 7], [1, 1, 0, 1, 0]
    got = weighted_kaplan_meier(times, events, [1.0] * 5, horizon=4.0)
    assert got.survival == pytest.approx(0.8 * 0.75, rel=TOL)
    assert got.events == 2


def test_km_sums_tied_event_times():
    got = weighted_kaplan_meier([3, 3, 5], [1, 1, 0], [1.0] * 3, horizon=10.0)
    assert got.survival == pytest.approx(1.0 - 2.0 / 3.0, rel=TOL)


def test_km_is_invariant_between_frequency_weights_and_expanded_rows():
    rng = np.random.default_rng(7)
    for _trial in range(5):
        n = 200
        times = rng.integers(1, 200, n).astype(float)
        events = (rng.random(n) < 0.35).astype(int)
        weights = rng.integers(1, 5, n).astype(float)

        weighted = weighted_kaplan_meier(times, events, weights, horizon=120.0)
        counts = weights.astype(int)
        expanded = weighted_kaplan_meier(
            np.repeat(times, counts), np.repeat(events, counts),
            np.ones(int(counts.sum())), horizon=120.0,
        )
        assert weighted.cumulative_incidence == pytest.approx(
            expanded.cumulative_incidence, rel=1e-12
        )


def test_km_equals_naive_proportion_when_nothing_is_censored_early():
    """The one case the two estimators must agree: full follow-up for every non-event."""
    times = np.full(50, 150.0)
    events = np.zeros(50, dtype=int)
    events[:10] = 1
    times[:10] = np.linspace(10.0, 110.0, 10)
    got = weighted_kaplan_meier(times, events, np.ones(50), horizon=120.0)
    assert got.cumulative_incidence == pytest.approx(10 / 50, rel=1e-9)


def test_naive_proportion_underestimates_when_censoring_precedes_the_horizon():
    rng = np.random.default_rng(0)
    n = 400
    times = rng.integers(1, 200, n).astype(float)
    events = (rng.random(n) < 0.3).astype(int)
    got = weighted_kaplan_meier(times, events, np.ones(n), horizon=120.0)
    naive = float(((events == 1) & (times <= 120)).sum()) / n
    assert naive < got.cumulative_incidence
    assert got.censored_before_horizon > 0


def test_km_is_bounded_and_counts_censoring():
    got = weighted_kaplan_meier([10, 20, 30], [1, 1, 1], [1.0] * 3, horizon=100.0)
    assert 0.0 <= got.cumulative_incidence <= 1.0
    assert got.survival == pytest.approx(0.0, abs=1e-12)   # everyone has the event


@pytest.mark.parametrize(
    "times,events,weights",
    [([], [], []), ([5.0], [1], [0.0]), ([float("nan")], [1], [1.0])],
)
def test_degenerate_km_returns_none_never_a_number(times, events, weights):
    assert weighted_kaplan_meier(times, events, weights, horizon=120.0) is None


def test_km_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_kaplan_meier([1.0, 2.0], [1], [1.0, 1.0], horizon=10.0)


# ════════════════════════════ bands and codes ══════════════════════════════
@pytest.mark.parametrize(
    "age,band",
    [(0, "Age_lt50"), (49.9, "Age_lt50"), (50, "Age_50_59"), (59.9, "Age_50_59"),
     (60, "Age_60_69"), (69.9, "Age_60_69"), (70, "Age_70p"), (85, "Age_70p")],
)
def test_age_bands_are_half_open_and_match_the_risk_layer_cutpoints(age, band):
    assert age_band(age) == band


def test_age_band_of_missing_is_none():
    assert age_band(None) is None
    assert age_band(float("nan")) is None


def test_sex_symbols_match_system_types():
    assert sex_symbol(1) == "Male"
    assert sex_symbol(2) == "Female"
    assert sex_symbol(9) is None           # NHANES 'refused/missing' style code
    assert sex_symbol(None) is None


# ════════════════════════════ the pandas IBM-zero bug ══════════════════════
def test_raw_pandas_corrupts_ibm_zero_and_read_nhanes_repairs_it(tmp_path):
    """Documents the bug this integration has to work around, and proves the fix."""
    path = write_xport(
        tmp_path / "L11_B.XPT", "L11_B",
        [("SEQN", NUM), ("LBXCRP", NUM)],
        [{"SEQN": 1.0, "LBXCRP": 0.0}, {"SEQN": 2.0, "LBXCRP": 0.3}],
    )
    raw = pd.read_sas(path, format="xport")
    assert float(raw["LBXCRP"][0]) == IBM_ZERO_ARTIFACT      # exact, not approximate
    assert float(raw["LBXCRP"][0]) != 0.0

    repaired = read_nhanes(path, require=["SEQN", "LBXCRP"])
    assert float(repaired["LBXCRP"][0]) == 0.0
    assert float(repaired["LBXCRP"][1]) == pytest.approx(0.3, rel=1e-15)


def test_fix_ibm_zero_leaves_other_values_and_strings_alone():
    frame = pd.DataFrame({"X": [IBM_ZERO_ARTIFACT, 1.5, -2.0], "S": ["a", "b", "c"]})
    fixed = fix_ibm_zero(frame)
    assert list(fixed["X"]) == [0.0, 1.5, -2.0]
    assert list(fixed["S"]) == ["a", "b", "c"]


# ════════════════════════════ XPORT round-trip ═════════════════════════════
XPORT_VALUES = [
    0.0, 1.0, -1.0, 55.0, 0.5, 123.456, -0.00789, 1e6,
    3.141592653589793, 85.0, 1e-5, 99999.9, 0.0001, -250.75,
]


def test_xport_round_trip_is_exact_for_mixed_magnitudes(tmp_path):
    rows = [{"SEQN": float(i + 1), "LBXVAL": v, "NAME": f"r{i}"}
            for i, v in enumerate(XPORT_VALUES)]
    rows.append({"SEQN": 999.0, "LBXVAL": None, "NAME": "miss"})
    path = write_xport(
        tmp_path / "T.XPT", "T",
        [("SEQN", NUM), ("LBXVAL", NUM), ("NAME", CHAR, 8)], rows,
    )
    assert path.stat().st_size % 80 == 0            # XPORT is a multiple of 80 bytes

    frame = read_nhanes(path, require=["SEQN", "LBXVAL", "NAME"])
    for i, expected in enumerate(XPORT_VALUES):
        assert float(frame["LBXVAL"][i]) == pytest.approx(expected, rel=1e-15, abs=1e-300)
    assert pd.isna(frame["LBXVAL"].iloc[-1])        # SAS missing -> NaN
    assert frame["NAME"][0] == "r0"                 # bytes decoded and stripped
    assert frame["NAME"].iloc[-1] == "miss"


def test_ibm_encoding_of_one_is_the_documented_byte_pattern():
    # 1.0 = 0.1(hex) * 16**1 -> exponent 65 (0x41), fraction 0x10 followed by zeros
    assert ieee_to_ibm(1.0) == bytes([0x41, 0x10, 0, 0, 0, 0, 0, 0])
    assert ieee_to_ibm(0.0) == b"\x00" * 8
    assert ieee_to_ibm(-1.0)[0] == 0xC1              # sign bit set on the same exponent


def test_ibm_encoding_of_missing_and_nan_is_the_sas_missing_code():
    assert ieee_to_ibm(None)[0] == 0x2E
    assert ieee_to_ibm(float("nan"))[0] == 0x2E


def test_xport_writer_rejects_over_long_variable_names(tmp_path):
    with pytest.raises(ValueError):
        write_xport(tmp_path / "T.XPT", "T", [("TOOLONGNAME", NUM)], [{"TOOLONGNAME": 1.0}])


def test_csv_extracts_are_accepted_as_an_alternative_input(tmp_path):
    csv = tmp_path / "demo.csv"
    csv.write_text("seqn,ridageyr,riagendr\n1,55,1\n2,62,2\n", encoding="utf-8")
    frame = read_nhanes(csv, require=["SEQN", "RIDAGEYR"])     # case-insensitive
    assert list(frame["RIDAGEYR"]) == [55, 62]


def test_missing_variable_raises_and_names_the_columns_present(tmp_path):
    path = write_xport(tmp_path / "T.XPT", "T", [("SEQN", NUM)], [{"SEQN": 1.0}])
    with pytest.raises(MissingColumns) as excinfo:
        read_nhanes(path, require=["LBXGH"])
    message = str(excinfo.value)
    assert "LBXGH" in message and "SEQN" in message


def test_absent_file_explains_that_nhanes_is_not_bundled(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        read_nhanes(tmp_path / "nope.XPT")
    assert "not bundled" in str(excinfo.value)


def test_unsupported_extension_is_refused(tmp_path):
    bad = tmp_path / "x.sav"
    bad.write_bytes(b"\x00")
    with pytest.raises(ValueError):
        read_nhanes(bad)


# ════════════════════════════ suppression ══════════════════════════════════
def test_small_cells_are_suppressed_with_a_reason():
    assert suppressed_reason(n=29, min_n=30) is not None
    assert suppressed_reason(n=30, min_n=30) is None
    assert suppressed_reason(n=100, events=4, min_events=5) is not None
    assert suppressed_reason(n=100, events=5, min_events=5) is None


# ════════════════════════════ MeTTa emission guards ════════════════════════
def test_number_formatting_is_plain_decimal_and_refuses_non_finite():
    assert num(0.0) == "0.000000"
    assert num(3.14159265) == "3.141593"
    assert "e" not in num(1e-7) and "E" not in num(1e-7)
    assert not num(-1e-9).startswith("-")            # no '-0.000000'
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            num(bad)


def test_symbol_validation_rejects_names_metta_cannot_read():
    assert check_symbol("Age_50_59") == "Age_50_59"
    for bad in ("Age<50", "70+", "1CRP", "has space", "(paren)", ""):
        with pytest.raises(ValueError):
            check_symbol(bad)


def test_strings_are_escaped():
    assert mstr('a "b" \\c') == '"a \\"b\\" \\\\c"'


def test_atom_budget_trips_before_hyperons_uncatchable_abort():
    writer = MettaWriter(budget=3)
    for i in range(3):
        writer.atom(f"(Foo bar{i})")
    assert writer.atom_count == 3
    with pytest.raises(AtomBudgetExceeded):
        writer.atom("(Foo overflow)")


def test_comments_and_rules_do_not_consume_the_atom_budget():
    writer = MettaWriter(budget=1)
    writer.comment("a note").rule("a section").blank()
    assert writer.atom_count == 0
    writer.atom("(Foo bar)")
    assert writer.atom_count == 1


def test_writer_emits_a_trailing_newline_and_counts_atoms(tmp_path):
    writer = MettaWriter()
    writer.comment("header").atom("(A b)").atom("(C d)")
    count = writer.write(tmp_path / "out.metta")
    assert count == 2
    text = (tmp_path / "out.metta").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert ";; header" in text and "(A b)" in text
