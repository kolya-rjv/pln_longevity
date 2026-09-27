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
    SCALE_IDENTITY,
    SCALE_LOG10,
    UCOD_HEART_DISEASE,
    UCOD_LEADING_LABELS,
    AtomBudgetExceeded,
    ByteBudgetExceeded,
    LinkageLayoutError,
    IBM_ZERO_ARTIFACT,
    MettaWriter,
    MissingColumns,
    age_band,
    apply_scale,
    check_symbol,
    design_se_of_weighted_mean,
    fix_ibm_zero,
    mstr,
    num,
    max_observed_followup_months,
    read_linked_mortality,
    read_nhanes,
    sex_symbol,
    standardize,
    suppressed_reason,
    weighted_aalen_johansen,
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


# ══════════════════════════ design-based standard error ════════════════════
def test_design_se_reduces_exactly_to_the_srs_formula():
    """The analytic identity the linearization must satisfy: equal weights, one stratum,
    one PSU per observation => s/sqrt(n) with s the ddof=1 sample SD."""
    values = [200.0, 180.0, 240.0, 150.0, 310.0, 275.0, 190.0]
    n = len(values)
    got = design_se_of_weighted_mean(
        values, [1.0] * n, ["1"] * n, [str(i) for i in range(n)]
    )
    expected = float(np.std(values, ddof=1)) / math.sqrt(n)
    assert got.se == pytest.approx(expected, rel=1e-13)
    assert got.n_strata == 1 and got.n_psu == n and got.singleton_strata == 0
    assert got.degrees_of_freedom == n - 1


def test_clustering_inflates_the_se_above_the_naive_srs_value():
    """A design effect above 1 is the whole reason this estimator exists."""
    rng = np.random.default_rng(1)
    n = 600
    strata = np.repeat([f"s{i}" for i in range(15)], n // 15)
    psu = np.array([f"{s}_p{i % 4}" for i, s in enumerate(strata)])
    cluster_effect = {p: rng.normal(0, 8) for p in np.unique(psu)}
    values = np.array([100 + cluster_effect[p] + rng.normal(0, 3) for p in psu])
    weights = rng.uniform(1000, 9000, n)

    design = design_se_of_weighted_mean(values, weights, strata, psu)
    naive = float(np.std(values, ddof=1)) / math.sqrt(n)
    assert design.se > naive
    assert design.n_strata == 15


def test_singleton_strata_are_counted_not_silently_dropped():
    got = design_se_of_weighted_mean(
        [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], ["a", "b", "b"], ["p1", "p2", "p3"]
    )
    assert got.singleton_strata == 1
    assert got.n_strata == 2


def test_design_se_guards():
    assert design_se_of_weighted_mean([], [], [], []) is None
    assert design_se_of_weighted_mean([1.0], [0.0], ["a"], ["p"]) is None
    with pytest.raises(ValueError):
        design_se_of_weighted_mean([1.0], [1.0], ["a", "b"], ["p"])


# ═════════════════════ horizon feasibility (flat-curve trap) ═══════════════
def test_short_followup_is_flagged_because_the_curve_is_flat_past_it():
    """A product-limit curve is undefined past the last observed time, so it returns
    S(last observed) — a plausible number that is not the risk at the horizon."""
    short = weighted_kaplan_meier([10, 20, 30], [1, 0, 0], [1.0] * 3, horizon=120.0)
    assert short.followup_reaches_horizon is False
    assert short.cumulative_incidence > 0.0          # it DID return a plausible number
    assert suppressed_reason(n=100, followup_reaches_horizon=False) is not None

    adequate = weighted_kaplan_meier([10, 200, 300], [1, 0, 0], [1.0] * 3, horizon=120.0)
    assert adequate.followup_reaches_horizon is True
    assert suppressed_reason(n=100, followup_reaches_horizon=True) is None


def test_short_followup_suppresses_regardless_of_cell_size():
    """Wrong is not the same as noisy: a huge cell with short follow-up is still wrong."""
    assert suppressed_reason(n=100_000, events=9_999, followup_reaches_horizon=False)


def test_aalen_johansen_also_reports_horizon_feasibility():
    short = weighted_aalen_johansen(
        [10, 20], ["001", "002"], [1.0, 1.0], horizon=120.0, cause="001"
    )
    assert short.followup_reaches_horizon is False
    adequate = weighted_aalen_johansen(
        [10, 200], ["001", "002"], [1.0, 1.0], horizon=120.0, cause="001"
    )
    assert adequate.followup_reaches_horizon is True


# ════════════════════════════ measurement scale ════════════════════════════
def test_identity_scale_passes_values_through():
    values, keep = apply_scale([1.0, 2.0, float("nan")], SCALE_IDENTITY)
    assert values[0] == 1.0 and values[1] == 2.0
    assert list(keep) == [True, True, False]


def test_log10_scale_drops_non_positive_rather_than_clamping():
    values, keep = apply_scale([100.0, 1.0, 0.0, -3.0], SCALE_LOG10)
    assert values[0] == pytest.approx(2.0)
    assert values[1] == pytest.approx(0.0)
    assert list(keep) == [True, True, False, False]      # not floored at a detection limit


def test_unknown_scale_is_rejected():
    with pytest.raises(ValueError):
        apply_scale([1.0], "Ln")


def test_raw_scale_makes_the_low_branch_unreachable_for_a_lognormal_marker():
    """The measurement this repo's symmetric z threshold depends on. See D17 / the
    MARKER_SCALES note in nhanes_common.py."""
    rng = np.random.default_rng(42)
    values = rng.lognormal(mean=np.log(0.2), sigma=np.log(3.0), size=100_000)
    weights = np.ones_like(values)

    raw = weighted_moments(values, weights)
    raw_z = (values - raw.mean) / raw.sd
    assert (raw_z < -1.0).mean() == 0.0                  # Low is literally unreachable
    assert (raw_z > 1.0).mean() < 0.10                   # and Elevated is far off 15.87%

    transformed, keep = apply_scale(values, SCALE_LOG10)
    logged = weighted_moments(transformed[keep], weights[keep])
    log_z = (transformed[keep] - logged.mean) / logged.sd
    assert (log_z < -1.0).mean() == pytest.approx(0.1587, abs=0.01)
    assert (log_z > 1.0).mean() == pytest.approx(0.1587, abs=0.01)


def test_standardize_refuses_rather_than_inventing_a_z():
    assert standardize(0.0, -0.7, 0.48, SCALE_LOG10) is None      # no log of zero
    assert standardize(1.0, 0.0, 0.0, SCALE_IDENTITY) is None     # degenerate reference
    assert standardize(1.0, 0.0, float("nan"), SCALE_IDENTITY) is None
    assert standardize(7.0, 5.0, 2.0, SCALE_IDENTITY) == pytest.approx(1.0)


# ═══════════════════════ linked mortality file reader ══════════════════════
def _lmf_record(seqn, elig, mort, ucod, diabetes, hyperten, pm_int, pm_exm,
                nhis=" " * 21):
    """One 2019-vintage NHANES LMF record, per CDC's published field positions."""
    text = (f"{seqn:<6}" + " " * 8 + f"{elig}" + f"{mort}" + f"{ucod:<3}"
            + f"{diabetes}" + f"{hyperten}" + nhis + f"{pm_int:>3}" + f"{pm_exm:>3}")
    return text.ljust(61)


GOOD_LMF = "\n".join([
    _lmf_record(1, 1, 1, "001", 0, 1, 120, 118),        # heart-disease death
    _lmf_record(2, 1, 0, "   ", " ", " ", 240, 238),    # censored survivor
    _lmf_record(3, 1, 1, "002", 0, 0, 60, 58),          # cancer death (competing)
    _lmf_record(4, 2, " ", "   ", " ", " ", "  ", "  "),  # linkage-ineligible
]) + "\n"


def _write_lmf(tmp_path, text, name="NHANES_2001_2002_MORT_2019_PUBLIC.dat"):
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return path


def test_linked_mortality_parses_the_documented_2019_layout(tmp_path):
    frame = read_linked_mortality(_write_lmf(tmp_path, GOOD_LMF))
    assert list(frame["SEQN"]) == [1, 2, 3, 4]
    assert list(frame["ELIGSTAT"]) == [1, 1, 1, 2]
    assert frame["PERMTH_EXM"][0] == 118
    assert pd.isna(frame["MORTSTAT"][3])                 # ineligible stays missing


def test_ucod_leading_stays_a_zero_padded_string(tmp_path):
    frame = read_linked_mortality(_write_lmf(tmp_path, GOOD_LMF))
    assert frame["UCOD_LEADING"][0] == "001"             # not the integer 1
    assert frame["UCOD_LEADING"][0] in UCOD_LEADING_LABELS
    assert pd.isna(frame["UCOD_LEADING"][1])


def test_heart_disease_code_is_documented_as_broader_than_chd():
    label = UCOD_LEADING_LABELS[UCOD_HEART_DISEASE]
    assert "Diseases of heart" in label
    assert "I50" in label or "I51" in label              # the range that swallows failure


def test_unknown_vintage_is_refused_not_guessed(tmp_path):
    path = _write_lmf(tmp_path, GOOD_LMF)
    with pytest.raises(LinkageLayoutError) as excinfo:
        read_linked_mortality(path, vintage="2022")
    assert "2019" in str(excinfo.value)                  # names what it does know


def test_reading_a_2019_file_with_the_2011_layout_is_caught(tmp_path):
    """The off-by-one vintage trap: it yields plausible integers, so it must be caught."""
    path = _write_lmf(tmp_path, GOOD_LMF)
    with pytest.raises(LinkageLayoutError):
        read_linked_mortality(path, vintage="2011")


def test_non_blank_nhis_block_is_refused(tmp_path):
    text = _lmf_record(1, 1, 1, "001", 0, 1, 120, 118, nhis="X" * 21) + "\n"
    with pytest.raises(LinkageLayoutError) as excinfo:
        read_linked_mortality(_write_lmf(tmp_path, text))
    assert "22-42" in str(excinfo.value)


def test_mortstat_on_an_ineligible_record_is_refused(tmp_path):
    text = _lmf_record(1, 2, 1, "001", 0, 1, 120, 118) + "\n"
    with pytest.raises(LinkageLayoutError):
        read_linked_mortality(_write_lmf(tmp_path, text))


def test_undocumented_cause_code_is_refused(tmp_path):
    text = _lmf_record(1, 1, 1, "099", 0, 1, 120, 118) + "\n"
    with pytest.raises(LinkageLayoutError) as excinfo:
        read_linked_mortality(_write_lmf(tmp_path, text))
    assert "099" in str(excinfo.value)


def test_feasible_horizon_is_derived_from_censored_records(tmp_path):
    frame = read_linked_mortality(_write_lmf(tmp_path, GOOD_LMF))
    # Only SEQN 2 is censored, with 238 months of follow-up.
    assert max_observed_followup_months(frame) == pytest.approx(238.0)


def test_absent_mortality_file_explains_it_is_not_bundled(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        read_linked_mortality(tmp_path / "nope.dat")
    assert "not bundled" in str(excinfo.value)


# ════════════════════════ Aalen-Johansen competing risks ═══════════════════
def test_aalen_johansen_reduces_to_one_minus_km_with_a_single_cause():
    times, causes = [2, 3, 3, 5, 7], ["001", "001", None, "001", None]
    weights = [1.0] * 5
    cif = weighted_aalen_johansen(times, causes, weights, horizon=10.0, cause="001")
    km = weighted_kaplan_meier(
        times, [1 if c else 0 for c in causes], weights, horizon=10.0
    )
    assert cif.cumulative_incidence == pytest.approx(km.cumulative_incidence, rel=TOL)
    assert cif.competing_incidence == pytest.approx(0.0, abs=1e-15)


def test_aalen_johansen_matches_hand_worked_competing_risks_example():
    # t=1 cause A, at risk 4 -> CIF_A += 1.00 * 1/4 = 0.25 ; S = 0.75
    # t=2 cause B, at risk 3 -> CIF_B += 0.75 * 1/3 = 0.25 ; S = 0.50
    # t=3 cause A, at risk 2 -> CIF_A += 0.50 * 1/2 = 0.25 ; S = 0.25
    got = weighted_aalen_johansen(
        [1, 2, 3, 4], ["A", "B", "A", None], [1.0] * 4, horizon=10.0, cause="A"
    )
    assert got.cumulative_incidence == pytest.approx(0.5, rel=TOL)
    assert got.competing_incidence == pytest.approx(0.25, rel=TOL)
    assert got.overall_survival == pytest.approx(0.25, rel=TOL)
    assert got.events == 2 and got.competing_events == 1


def test_aalen_johansen_identity_holds_under_random_survey_weights():
    """CIF_cause + CIF_competing + S(horizon) == 1 is the estimator's defining identity."""
    rng = np.random.default_rng(3)
    for _trial in range(50):
        n = 150
        times = rng.integers(1, 200, n).astype(float)
        draw = rng.random(n)
        causes = ["001" if u < 0.12 else "002" if u < 0.30 else None for u in draw]
        weights = rng.uniform(0.5, 5000.0, n)          # NHANES-like weight spread
        got = weighted_aalen_johansen(times, causes, weights, horizon=120.0, cause="001")
        total = got.cumulative_incidence + got.competing_incidence + got.overall_survival
        assert total == pytest.approx(1.0, abs=1e-12)


def test_treating_competing_deaths_as_censoring_overstates_the_risk():
    """The bias this estimator exists to avoid, asserted in direction and reported."""
    rng = np.random.default_rng(11)
    n = 300
    times = rng.integers(1, 200, n).astype(float)
    draw = rng.random(n)
    causes = ["001" if u < 0.15 else "002" if u < 0.45 else None for u in draw]
    got = weighted_aalen_johansen(times, causes, np.ones(n), horizon=120.0, cause="001")
    assert got.naive_km_incidence > got.cumulative_incidence
    assert got.competing_events > 0


def test_aalen_johansen_is_invariant_between_frequency_weights_and_expanded_rows():
    rng = np.random.default_rng(5)
    n = 120
    times = rng.integers(1, 150, n).astype(float)
    draw = rng.random(n)
    causes = ["001" if u < 0.2 else "002" if u < 0.35 else None for u in draw]
    weights = rng.integers(1, 4, n).astype(float)

    weighted = weighted_aalen_johansen(times, causes, weights, horizon=100.0, cause="001")
    index = np.repeat(np.arange(n), weights.astype(int))
    expanded = weighted_aalen_johansen(
        times[index], [causes[i] for i in index], np.ones(index.size),
        horizon=100.0, cause="001",
    )
    assert weighted.cumulative_incidence == pytest.approx(
        expanded.cumulative_incidence, rel=1e-12
    )


def test_absent_cause_gives_zero_incidence_not_none():
    got = weighted_aalen_johansen([5.0], ["002"], [1.0], horizon=120.0, cause="001")
    assert got.cumulative_incidence == 0.0
    assert got.events == 0 and got.competing_events == 1


def test_degenerate_aalen_johansen_returns_none():
    assert weighted_aalen_johansen([], [], [], horizon=120.0, cause="001") is None
    assert weighted_aalen_johansen([5.0], ["001"], [0.0], horizon=120.0, cause="001") is None


def test_aalen_johansen_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_aalen_johansen([1.0, 2.0], ["001"], [1.0, 1.0], horizon=10.0, cause="001")


def test_blank_and_nan_cause_codes_count_as_censored_not_as_events():
    got = weighted_aalen_johansen(
        [1, 2, 3, 4], ["001", "", None, float("nan")], [1.0] * 4,
        horizon=10.0, cause="001",
    )
    assert got.events == 1
    assert got.competing_events == 0
    assert got.censored_before_horizon == 3      # t=2, 3 and 4 all precede the horizon


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


def test_byte_budget_matches_the_chat_apps_silent_skip_threshold():
    """pln_chat drops an oversized .metta file from execution with only a print(), so the
    app answers 'empty' and nothing raises. The writer must refuse first."""
    from nhanes_common import DEFAULT_BYTE_BUDGET

    config = (REPO / "pln_chat" / "config.py").read_text(encoding="utf-8")
    assert f'"{DEFAULT_BYTE_BUDGET}"' in config, (
        "DEFAULT_BYTE_BUDGET must track PLN_MAX_KB_FILE_BYTES in pln_chat/config.py"
    )


def test_oversized_emission_is_refused_before_it_is_written(tmp_path):
    writer = MettaWriter(byte_budget=200)
    for i in range(30):
        writer.atom(f"(RefMarker NHANESRef_CRP_Male_50_59_9902_{i} CRP)")
    assert writer.byte_size > 200
    target = tmp_path / "out.metta"
    with pytest.raises(ByteBudgetExceeded):
        writer.write(target)
    assert not target.exists()          # nothing half-written


def test_a_mismatched_rule_head_returns_the_unreduced_expression_not_empty():
    """MeTTa semantics that the baseline lookup design depends on: relying on a rule head
    failing to match as a way of 'yielding nothing' is wrong -- the unreduced expression
    propagates into arithmetic. Only `match` or an explicit (superpose ()) yield nothing."""
    hyperon = pytest.importorskip("hyperon")
    from hyperon import MeTTa

    metta = MeTTa()
    metta.run("""
        (= (only-chd CoronaryHeartDisease $age) 0.08)
        (BaseRec R1 AllCauseMortality 0.11)
        (= (via-match $o) (match &self (BaseRec $r $o $v) $v))
        (= (guarded $o) (if (== $o AllCauseMortality) 0.11 (superpose ())))
    """)

    mismatched = str(metta.run("!(only-chd AllCauseMortality 61)")[0][0])
    assert mismatched == "(only-chd AllCauseMortality 61)"        # not empty
    propagated = str(metta.run("!(* 2 (only-chd AllCauseMortality 61))")[0][0])
    assert "only-chd" in propagated                              # and it spreads

    assert metta.run("!(via-match CoronaryHeartDisease)")[0] == []
    assert metta.run("!(guarded CoronaryHeartDisease)")[0] == []
    assert float(str(metta.run("!(guarded AllCauseMortality)")[0][0])) == pytest.approx(0.11)


def test_log_math_fails_silently_so_the_guard_must_be_on_the_input():
    """A raw 0.0 on a log-scaled marker becomes -inf, which z->status reads as Low -- a
    fabricated finding. See docs/nhanes_integration.md section 4."""
    hyperon = pytest.importorskip("hyperon")
    from hyperon import MeTTa

    metta = MeTTa()
    metta.run((REPO / "patient_profile.metta").read_text(encoding="utf-8"))

    assert str(metta.run("!(log-math 10 0)")[0][0]) == "-inf"
    assert str(metta.run("!(log-math 10 -5)")[0][0]) == "NaN"
    # -inf is ordered, so it passes the Low comparison and invents a finding
    assert str(metta.run("!(z->status (log-math 10 0))")[0][0]) == "Low"
    # a correct guard on the raw value yields nothing instead
    metta.run("(= (guarded-z $raw) (if (> $raw 0.0) (log-math 10 $raw) (superpose ())))")
    assert metta.run("!(guarded-z 0.0)")[0] == []
    assert metta.run("!(guarded-z -5.0)")[0] == []
    assert float(str(metta.run("!(guarded-z 100.0)")[0][0])) == pytest.approx(2.0)
