"""End-to-end tests for the NHANES integration: fixtures -> ETL -> MeTTa -> inference.

`test_nhanes_common.py` covers the statistics and the file readers in isolation. These
tests close the loop that actually matters, and that nothing else covers: that a synthetic
NHANES file run through the real ETL produces a KB file the real inference stack can
consume, and that a patient holding only RAW lab values reaches the same conclusions a
hand-typed standardized patient does.

What is asserted, in the order the chain runs:

  * the reference ETL reads genuine SAS XPORT input (written by `nhanes_xport_writer`),
    joins demographics to a lab file, and emits Schema A records whose scale, unit and
    provenance fields match the registry;
  * thin cells are SUPPRESSED rather than emitted, and the run says which and why;
  * the emitted file loads into hyperon alongside the full inference stack, and a patient
    carrying only `MeasuredRaw` atoms gets a derived z and a qualitative status from it;
  * a marker whose cell was suppressed yields NOTHING -- no z, no status, no invented
    `Normal`;
  * the log-scaled marker is standardized on its own scale, so the z differs from the
    identity-scale arithmetic on the same numbers (the §4 bug, absent);
  * a raw-fed patient flows all the way into `diagnose-patient`;
  * the emitted file stays inside both the atom and the byte budget.

Everything here is SYNTHETIC. The ETL is run against values that are arbitrary by
construction; no NHANES data is present in this repository and none is downloaded. The
numbers asserted below are properties of the fixture, never claims about any population.

Needs `hyperon` and `pandas`; skipped automatically without them.

Run:  pytest tests/ -v       (from the repo root)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

pytest.importorskip("pandas")
pytest.importorskip("hyperon")
from hyperon import MeTTa  # noqa: E402

from nhanes_common import DEFAULT_BYTE_BUDGET  # noqa: E402
from nhanes_xport_writer import NUM, write_xport  # noqa: E402

# Dependency-ordered load. nhanes_reference.metta sits after patient_profile.metta (it
# holds the record types and standardize-z, which the patient layer's derived-z calls),
# and the GENERATED numeric file must come after the layer that declares its types.
KB_FILES = [
    "system_types.metta",
    "logical_predicates.metta",
    "epistemic_calibration.metta",
    "grim_age_core.metta",
    "grim_age_lu2019_evidence.metta",
    "evidence_calibration.metta",
    "hallmarks_core.metta",
    "hallmarks_lopezotin2023_intervention_evidence.metta",
    "mechanistic_bridges.metta",
    "pln_deduction.metta",
    "pln_intervention_ranking.metta",
    "pln_abductive_diagnosis.metta",
    "patient_profile.metta",
    "nhanes_reference.metta",
]

# One 50-59 male cell gets plenty of observations; the 70+ cell gets too few, so it must
# be suppressed. Values are arbitrary.
_N_MAIN = 60
_N_THIN = 4


def _write_fixture(directory: Path) -> None:
    """A synthetic NHANES 2001-2002 demographics file and CRP/HbA1c lab files."""
    demo_rows, crp_rows, hba1c_rows = [], [], []
    seqn = 1

    def add(age: float, sex: float, crp: float | None, hba1c: float | None) -> None:
        nonlocal seqn
        demo_rows.append({
            "SEQN": float(seqn), "RIDAGEYR": age, "RIAGENDR": sex,
            "WTMEC2YR": 40000.0 + (seqn % 7) * 1500.0,
            "SDMVSTRA": float(1 + seqn % 12), "SDMVPSU": float(1 + seqn % 2),
        })
        crp_rows.append({"SEQN": float(seqn), "LBXCRP": crp})
        hba1c_rows.append({"SEQN": float(seqn), "LBXGH": hba1c})
        seqn += 1

    for i in range(_N_MAIN):                       # Male, 50-59 — the populated cell
        add(50.0 + (i % 10), 1.0, 0.10 + 0.02 * (i % 25), 5.0 + 0.05 * (i % 20))
    for i in range(_N_MAIN):                       # Female, 50-59 — also populated
        add(50.0 + (i % 10), 2.0, 0.12 + 0.02 * (i % 25), 5.1 + 0.05 * (i % 20))
    for i in range(_N_THIN):                       # Male, 70+ — must be suppressed
        add(72.0 + i, 1.0, 0.30, 6.0)
    add(55.0, 1.0, 0.0, 5.5)                       # a true zero: log scale must drop it
    add(56.0, 1.0, None, None)                     # missing values

    write_xport(directory / "DEMO_B.XPT", "DEMO_B",
                [("SEQN", NUM), ("RIDAGEYR", NUM), ("RIAGENDR", NUM),
                 ("WTMEC2YR", NUM), ("SDMVSTRA", NUM), ("SDMVPSU", NUM)], demo_rows)
    write_xport(directory / "L11_B.XPT", "L11_B",
                [("SEQN", NUM), ("LBXCRP", NUM)], crp_rows)
    write_xport(directory / "L10_B.XPT", "L10_B",
                [("SEQN", NUM), ("LBXGH", NUM)], hba1c_rows)


@pytest.fixture(scope="module")
def etl_output(tmp_path_factory) -> dict:
    """Run the real reference ETL over a synthetic fixture; return its output + log."""
    work = tmp_path_factory.mktemp("nhanes")
    data = work / "data"
    data.mkdir()
    _write_fixture(data)
    out = work / "nhanes_reference.metta"

    proc = subprocess.run(
        [sys.executable, str(REPO / "nhanes_reference_etl.py"),
         "--data-dir", str(data), "--cycles", "2001-2002",
         "--markers", "CRP,HbA1c", "--output", str(out)],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
    )
    assert proc.returncode == 0, f"ETL failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert out.exists(), proc.stdout + proc.stderr
    return {"path": out, "text": out.read_text(encoding="utf-8"),
            "log": proc.stdout + proc.stderr}


@pytest.fixture(scope="module")
def kb(etl_output) -> MeTTa:
    """The full inference stack plus the ETL-generated reference records, one space.

    Loaded by reading the file text and running it, NOT via `import!`: in hyperon 0.2.10
    `(import! &self <path>)` silently yields nothing here, so every match would come back
    empty and the tests would pass vacuously.
    """
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in KB_FILES)
    text += "\n" + etl_output["text"]
    # A patient known only by RAW lab values -- the case this whole layer exists for.
    text += """
(InstanceOf PatientRaw PatientProfile)
(PatientAge     PatientRaw 58)
(PatientSex     PatientRaw Male)
(PatientSmoking PatientRaw NeverSmoker)
(MeasuredRaw PatientRaw CRP   0.820000)
(MeasuredRaw PatientRaw HbA1c 6.400000)

(InstanceOf PatientOld PatientProfile)
(PatientAge     PatientOld 75)
(PatientSex     PatientOld Male)
(MeasuredRaw PatientOld HbA1c 6.400000)
"""
    metta = MeTTa()
    metta.run(text)
    return metta


def _results(kb: MeTTa, query: str) -> list:
    out = kb.run(query)
    return list(out[0]) if out and out[0] else []


def _one_number(kb: MeTTa, query: str) -> float:
    res = _results(kb, query)
    assert len(res) == 1, f"expected exactly one result for {query}, got {res}"
    return float(str(res[0]))


# ════════════════════════════ the ETL's output ═════════════════════════════
def test_etl_emits_schema_a_records_for_both_markers(etl_output):
    text = etl_output["text"]
    assert "(: NR_CRP_M_5059_0102 ReferenceDistribution)" in text
    assert "(: NR_HbA1c_M_5059_0102 ReferenceDistribution)" in text
    for field in ("RefMarker", "RefSex", "RefAgeBand", "RefScale", "RefMean", "RefSD",
                  "RefUnweightedN", "RefUnit", "RefWeightVariable", "RefSourceVariable",
                  "RefSourceCycles", "RefAssayLot", "RefProvenance"):
        assert f"({field}" in text, f"Schema A field {field} missing from the output"
    assert "NHANES_Microdata" in text


def test_etl_records_the_declared_scale_per_marker(etl_output):
    text = etl_output["text"]
    assert "(RefScale          NR_CRP_M_5059_0102 Log10)" in text
    assert "(RefScale          NR_HbA1c_M_5059_0102 Identity)" in text


def test_etl_suppresses_the_thin_cell_and_says_why(etl_output):
    assert "NR_HbA1c_M_70p_0102" not in etl_output["text"]
    assert "suppressed" in etl_output["log"]
    assert "below minimum cell size" in etl_output["log"]


def test_etl_emits_a_design_based_se_when_the_design_columns_are_present(etl_output):
    assert "(RefDesignSE" in etl_output["text"]
    assert "(RefDesignDF" in etl_output["text"]


def test_etl_output_is_inside_both_budgets(etl_output):
    text = etl_output["text"]
    atoms = sum(1 for line in text.splitlines() if line.startswith("("))
    size = len(text.encode("utf-8"))
    assert 0 < atoms < 3000, atoms
    assert size < DEFAULT_BYTE_BUDGET, f"{size} bytes would be silently skipped by pln_chat"


def test_the_hand_written_layers_ship_no_nhanes_records():
    """The rule the whole integration is built around: no NHANES number is committed.

    Comments are stripped first, because the layers legitimately DOCUMENT the record shape
    in comments; what must not appear is an actual record atom.
    """
    for name in ("nhanes_reference.metta", "patient_profile.metta", "nhanes_baseline.metta"):
        path = REPO / name
        if not path.exists():                 # nhanes_baseline.metta may not exist yet
            continue
        code = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith(";;")
        )
        for record_type in ("ReferenceDistribution)", "BaselineRiskRecord)", "ClockAccelSpread)"):
            offenders = [
                line for line in code.splitlines()
                if line.strip().startswith("(: ") and line.strip().endswith(record_type)
            ]
            assert not offenders, (
                f"{name} declares a {record_type[:-1]} RECORD: {offenders}. The "
                f"hand-written layers carry types and rules only; records are ETL output."
            )


# ═══════════════════ the generated records reach inference ══════════════════
def test_a_raw_only_patient_gets_a_derived_z(kb):
    z = _one_number(kb, "!(patient-z &self PatientRaw HbA1c)")
    assert z > 1.0                          # the fixture puts 6.4 well above its cell mean
    assert _results(kb, "!(patient-status &self PatientRaw HbA1c)") == ["Elevated"] or \
        str(_results(kb, "!(patient-status &self PatientRaw HbA1c)")[0]) == "Elevated"


def test_the_log_scaled_marker_is_standardized_on_its_own_scale(kb):
    """If CRP were standardized on the raw scale the z would differ -- that is the §4 bug."""
    z_log = _one_number(kb, "!(patient-z &self PatientRaw CRP)")
    mean = _one_number(kb, "!(match &self (RefMean NR_CRP_M_5059_0102 $m) $m)")
    sd = _one_number(kb, "!(match &self (RefSD NR_CRP_M_5059_0102 $s) $s)")
    import math
    expected_log = (math.log10(0.82) - mean) / sd
    identity_z = (0.82 - mean) / sd
    assert z_log == pytest.approx(expected_log, rel=1e-6)
    assert abs(z_log - identity_z) > 1e-6, "the two scales must not coincide here"


def test_a_suppressed_cell_yields_nothing_rather_than_a_default(kb):
    """PatientOld is 70+, whose HbA1c cell was suppressed for a small n."""
    assert _results(kb, "!(patient-z &self PatientOld HbA1c)") == []
    assert _results(kb, "!(patient-status &self PatientOld HbA1c)") == []


def test_an_unmeasured_marker_still_invents_nothing(kb):
    assert _results(kb, "!(patient-z &self PatientRaw FastingGlucose)") == []
    assert _results(kb, "!(patient-status &self PatientRaw DNAmPAI1)") == []


def test_the_raw_patients_markers_are_enumerated_exactly_once(kb):
    markers = str(_results(kb, "!(patient-markers &self PatientRaw)")[0])
    assert markers.count("CRP") == 1
    assert markers.count("HbA1c") == 1


def test_a_raw_only_patient_reaches_the_observation_set(kb):
    observations = str(_results(kb, "!(patient-observations &self PatientRaw)")[0])
    assert "HbA1c" in observations


def test_a_raw_only_patient_reaches_abductive_diagnosis(kb):
    """The point of the layer: data-fed patients drive the same demos as hand-typed ones."""
    results = _results(
        kb,
        "!(diagnose-patient &self PatientRaw "
        "(CellularSenescence MitochondrialDysfunction ChronicInflammation))",
    )
    assert results, "diagnose-patient returned nothing for a raw-only patient"
    assert "Hypothesis" in str(results[0])


# ══════════════════════ the hand-typed patients are intact ═════════════════
def test_the_existing_example_patients_are_unchanged_by_the_new_layer(kb):
    assert str(_results(kb, "!(patient-status &self Patient001 DNAmPAI1)")[0]) == "Elevated"
    assert str(_results(kb, "!(patient-status &self Patient001 HorvathAgeAccel)")[0]) == "Normal"
    assert str(_results(kb, "!(patient-status &self Patient001 DNAmLeptin)")[0]) == "Low"
    observations = str(_results(kb, "!(patient-observations &self Patient001)")[0])
    for marker in ("AgeAccelGrim", "DNAmPAI1", "DNAmGDF15", "CRP"):
        assert marker in observations


def test_an_explicitly_standardized_marker_still_wins_over_a_derived_one(kb):
    """Patient001 has an explicit CRP z of 1.7; reference cells exist for CRP too."""
    z = _one_number(kb, "!(patient-z &self Patient001 CRP)")
    assert z == pytest.approx(1.7, rel=1e-9)


# ═════════════════ ambiguity declines rather than being picked ═════════════
_STACK_WITH_BASELINE = KB_FILES + ["pln_counterfactual.metta", "pln_risk_prediction.metta",
                                   "nhanes_baseline.metta"]


def _space(extra: str, files=None) -> MeTTa:
    text = "\n".join((REPO / f).read_text(encoding="utf-8") for f in (files or KB_FILES))
    metta = MeTTa()
    metta.run(text + "\n" + extra)
    return metta


_PATIENT = ("(InstanceOf PX PatientProfile)(PatientAge PX 58)(PatientSex PX Male)"
            "(MeasuredRaw PX HbA1c 6.200000)")


def _ref(rid: str, mean: str, sd: str) -> str:
    return (f"(: {rid} ReferenceDistribution)\n"
            f"(RefMarker {rid} HbA1c)(RefSex {rid} Male)(RefAgeBand {rid} Age_50_59)\n"
            f"(RefScale {rid} Identity)(RefMean {rid} {mean})(RefSD {rid} {sd})\n")


def test_one_reference_cell_resolves():
    kb = _space(_ref("RA", "5.400000", "0.500000") + _PATIENT)
    assert _results(kb, "!(patient-z &self PX HbA1c)")


def test_two_reference_cells_decline_rather_than_pick_one():
    """Single-valued is not enough: it must be single-MEANING. Taking the first cell made
    the answer depend on the order records happened to appear in a generated file."""
    both = _ref("RA", "5.400000", "0.500000") + _ref("RB", "5.900000", "1.000000")
    assert _results(_space(both + _PATIENT), "!(patient-z &self PX HbA1c)") == []
    reversed_order = _ref("RB", "5.900000", "1.000000") + _ref("RA", "5.400000", "0.500000")
    assert _results(_space(reversed_order + _PATIENT), "!(patient-z &self PX HbA1c)") == []


def _baseline(rid: str, horizon: str, risk: str) -> str:
    return (f"(: {rid} BaselineRiskRecord)\n"
            f"(BaseOutcome {rid} AllCauseMortality)(BaseSex {rid} Male)"
            f"(BaseAgeBand {rid} Age_50_59)\n"
            f"(BaseHorizonMonths {rid} {horizon})(BaseRisk {rid} {risk})\n"
            f"(BaseEstimator {rid} WeightedKaplanMeier)"
            f"(BaseProvenance {rid} NHANES_Microdata)\n")


def test_one_baseline_record_resolves():
    kb = _space(_baseline("NB_A", "120", "0.214324"), _STACK_WITH_BASELINE)
    assert _results(kb, "!(patient-baseline &self Patient001 AllCauseMortality)")


def test_two_horizons_decline_rather_than_serve_the_shorter_one():
    """A 60-month baseline must never be served into a rule documented as ten-year."""
    both = _baseline("NB_A", "120", "0.214324") + _baseline("NB_B", "60", "0.094011")
    assert _results(_space(both, _STACK_WITH_BASELINE),
                    "!(patient-baseline &self Patient001 AllCauseMortality)") == []
    flipped = _baseline("NB_B", "60", "0.094011") + _baseline("NB_A", "120", "0.214324")
    assert _results(_space(flipped, _STACK_WITH_BASELINE),
                    "!(patient-baseline &self Patient001 AllCauseMortality)") == []


def test_curated_chd_baseline_is_unaffected_by_ambiguous_mortality_records():
    both = _baseline("NB_A", "120", "0.214324") + _baseline("NB_B", "60", "0.094011")
    kb = _space(both, _STACK_WITH_BASELINE)
    chd = _results(kb, "!(patient-baseline &self Patient001 CoronaryHeartDisease)")
    assert len(chd) == 1 and float(str(chd[0])) == pytest.approx(0.08)


def test_the_clock_etl_defaults_to_the_residual_convention():
    """Emitting both by default put two cells on one key, which now declines entirely."""
    source = (REPO / "nhanes_dnam_etl.py").read_text(encoding="utf-8")
    assert '"--accel-definition", default="residual"' in source
