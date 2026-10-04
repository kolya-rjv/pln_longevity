"""Extract the LinAge2 clock into a small parameter file, plus golden test cases.

    python scripts/extract_linage2_model.py --linage2-repo /path/to/LinAge2-Python

Run against a checkout of Rejuve/LinAge2-Python (it needs that repository's own
dependencies: pandas, scipy, joblib, scikit-survival). It writes

    data/linage2/linage2_model.json      what core/linage2_model.py evaluates
    tests/fixtures/linage2_golden.json   cases scored by the SERVICE's own code

and nothing else. Re-run it when the upstream model is retrained; the JSON records
the upstream commit and a hash of every artifact it was read from.

WHY THIS IS POSSIBLE. LinAge2 is additive in its inputs. For one person of sex s
and age a (months), with x_j the value of input j after imputation and the two
derived inputs (LDL, the urine albumin/creatinine ratio):

    z_j      = clip( (BoxCox_j(x_j) - median_sj) / mad_sj , -6, 6 )   # median/MAD of the
                                                                      # <=50-year-old 1999-2000
                                                                      # reference, sex s
    years_j  = (z_j - muZ_sj) * w_sj / 12                             # w: SVD loadings x Cox
                                                                      # betas / beta_age(null)
    delta    = sum_j years_j + (a - muAge_s) * wAge_s / 12

so every number the service computes is a fixed per-input function of that one
input. Five inputs skip the median/MAD step (the three questionnaire scores,
cotinine and the basophil count); everything else is as written. The script reads
those constants out of the artifacts once, and `core/linage2_model.py` evaluates
the formula with no pandas, scipy, scikit-survival or pickles at runtime.

TWO DELIBERATE DEPARTURES from the service, both recorded in the JSON:

1. Cotinine is on the TRAINING scale. The model was fitted on `digiCot` bins of
   serum cotinine (0: <10 ng/mL, 1: 10-100, 2: 100-200, 3: >=200; the training
   matrix holds all four levels). The service's imputation pool holds RAW cotinine,
   so a missing cotinine was imputed as a raw ng/mL median (~0.1) read as a level.
   Here the imputation medians are taken over the digitized column. Golden cases
   therefore always supply cotinine, so they test everything else exactly.
2. Imputation medians are tabulated per sex and WHOLE year of age (20-90), the
   service's window evaluated at age*12 months. A fractional age is rounded.

COMPATIBILITY SHIMS. The service does not run on numpy >= 2.3 / pandas >= 3 as
published: `foldOutliers` writes into a read-only `.values`, and two `float()`
calls receive 1-element arrays. The golden run patches exactly those three spots
(the arithmetic is untouched) — see `_load_service`.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_MODEL = REPO / "data" / "linage2" / "linage2_model.json"
OUT_GOLDEN = REPO / "tests" / "fixtures" / "linage2_golden.json"

DERIVED_DROP = ("LBDTCSI", "LBDHDLSI", "LBDSTRSI")      # consumed by LDLV, not model inputs
SKIP_NORMALISATION = ("fs1Score", "fs2Score", "fs3Score", "LBXCOT", "LBDBANO")
QUESTIONNAIRE_ITEMS = (
    "BPQ020", "DIQ010", "KIQ020", "MCQ010", "MCQ053",
    "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
    "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K", "MCQ160L", "MCQ220",
    "OSQ010A", "OSQ010B", "OSQ010C", "OSQ060", "PFQ056", "HUQ070",
    "HUQ010", "HUQ020", "HUQ050",
)
#: process_payload's questionnaire defaults: every diagnosis "No", general health
#: "Good", health "about the same" as a year ago, no healthcare visits.
QUESTIONNAIRE_DEFAULTS = {**{q: 2 for q in QUESTIONNAIRE_ITEMS}, "HUQ010": 3, "HUQ020": 3, "HUQ050": 0}
AGE_TABLE = range(20, 91)
#: Upstream's ui_sliders.nhanes_desc gets three units wrong against the values the
#: model is fitted on (NHANES 1999-2002 and digiCot); the reader and
#: linage2_core.metta use these, and core.patient_vocabulary.drift() checks them.
DESCRIPTION_FIXES = {
    "URXUCRSI": "Urine creatinine, SI units (µmol/L).",
    "LBXCRP": "C-reactive protein (mg/dL).",
    "LBXCOT": "Serum cotinine, digitized as trained: 0 (<10 ng/mL, non-smoker), 1 (10-100), "
              "2 (100-200), 3 (>=200 ng/mL)",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_service(lin: Path):
    """Import the service from `lin` with the three numpy-2 / pandas-3 shims."""
    os.chdir(lin)
    sys.path.insert(0, str(lin))
    with contextlib.redirect_stdout(io.StringIO()):
        import numpy as np
        src = (lin / "linage2_service.py").read_text(encoding="utf-8")
        for old in ("float(Z_centered @ w_feature_months_per_sd)",
                    "float((initAge_user - mu_age) * w_age)"):
            assert old in src, f"upstream changed; re-check the shim for {old!r}"
            src = src.replace(old, f"float(np.asarray({old[6:-1]}).reshape(-1)[0])")
        svc = types.ModuleType("linage2_service")
        svc.__file__ = str(lin / "linage2_service.py")
        sys.modules["linage2_service"] = svc
        exec(compile(src, svc.__file__, "exec"), svc.__dict__)

        def fold(mat, zmax):
            out = mat.copy()
            for col in mat.columns[1:]:
                out[col] = np.clip(mat[col].to_numpy(dtype=float, copy=True), -zmax, zmax)
            return out

        svc.foldOutliers = fold
    return svc


def _f(x) -> float | None:
    import math
    x = float(x)
    return None if math.isnan(x) else x


def extract(lin: Path) -> tuple[dict, object]:
    import numpy as np
    import pandas as pd
    from scipy import stats

    svc = _load_service(lin)
    with contextlib.redirect_stdout(io.StringIO()):
        from imputation import imputation_pool
        from src import digiCot
        from ui_sliders import LAB_RANGES, LAB_VARIABLES, nhanes_desc
        bundle = svc.load_linage2_bundle("artifacts")

    lab_inputs = list(LAB_VARIABLES)
    features = [c for c in lab_inputs if c not in DERIVED_DROP] + [
        "fs1Score", "fs2Score", "fs3Score", "LDLV", "crAlbRat"]
    assert len(features) == 59, len(features)

    lam_row = bundle.boxCox_lam
    lam = {c: (_f(lam_row[c].iloc[0]) if c in lam_row.columns else None) for c in features}

    ref = bundle.qDataMat_R
    sel = ((ref["yearsNHANES"] == 9900) | (ref["yearsNHANES"].astype(str) == "9900")) & (ref["RIDAGEYR"] <= 50)
    ref_data = bundle.dataMat_trans_ref.loc[sel]
    ref_male = ref.loc[sel, "RIAGENDR"] == 1

    def sex_block(code: int) -> dict:
        rows = ref_data.loc[ref_male] if code == 1 else ref_data.loc[~ref_male]
        median, mad = [], []
        for c in features:
            if c in SKIP_NORMALISATION:
                median.append(None)
                mad.append(None)
                continue
            v = pd.to_numeric(rows[c], errors="coerce").to_numpy()
            m = float(np.nanmedian(v))
            d = float(stats.median_abs_deviation(v, scale="normal", nan_policy="omit"))
            if not d > 0:
                raise SystemExit(f"MAD of {c} is {d} for sex {code}: the service would return "
                                 f"NaN for every non-median value; refusing to extract")
            median.append(m)
            mad.append(d)
        v = bundle.vMatDat99_M if code == 1 else bundle.vMatDat99_F
        cox = bundle.cox_full_M if code == 1 else bundle.cox_full_F
        null = bundle.cox_null_M if code == 1 else bundle.cox_null_F
        train = bundle.coxCovsTrainM if code == 1 else bundle.coxCovsTrainF
        pcs = [int(x[2:]) - 1 for x in cox.feature_names_in_ if x.startswith("PC")]
        beta = np.zeros(v.shape[1])
        beta[pcs] = np.asarray(cox.coef_[1:], dtype=float)
        beta_age_null = float(null.coef_[0])
        mu_pc = np.zeros(v.shape[1])
        mu_pc[pcs] = train.mean().loc[cox.feature_names_in_].iloc[1:].values
        return {
            "median": median,
            "mad": mad,
            "mu_z": [float(x) for x in (mu_pc @ v.T)],
            "w_months_per_sd": [float(x) for x in ((v @ beta) / beta_age_null)],
            "w_age": float(cox.coef_[0]) / beta_age_null - 1.0,
            "mu_age_months": float(train["chronAge"].mean()),
            "training_age_years": [round(float(train["chronAge"].min()) / 12, 2),
                                   round(float(train["chronAge"].max()) / 12, 2)],
            "training_n": int(train.shape[0]),
        }

    # imputation medians: the service's window, per whole year of age, cotinine digitized
    with contextlib.redirect_stdout(io.StringIO()):
        pool = digiCot(imputation_pool.copy())
    def window(code: int, age: int):
        lo, hi = (max(age - 5, 40), age + 5) if age >= 40 else (age - 5, min(age + 5, 40))
        rows = pool.query(f"RIAGENDR == {code} & RIDAGEEX >= {lo * 12} & RIDAGEEX <= {hi * 12}")
        return rows if rows.shape[0] else pool.query(f"RIAGENDR == {code}")
    imputation = {}
    for code, name in ((1, "male"), (2, "female")):
        table = {c: [] for c in lab_inputs}
        for age in AGE_TABLE:
            med = window(code, age).median(numeric_only=True, skipna=True)
            for c in lab_inputs:
                table[c].append(_f(med[c]))
        imputation[name] = table

    adults = imputation_pool.query("RIDAGEYR >= 20")
    upstream_commit = subprocess.run(["git", "-C", str(lin), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip() or None
    artifacts = sorted((lin / "artifacts").glob("*")) + [lin / "mergedDataNHANES9902.csv"]
    model = {
        "model": "LinAge2",
        "publication": "Fong et al. 2025, npj Aging 11:29 (doi:10.1038/s41514-025-00221-4)",
        "source": {
            "repository": "Rejuve/LinAge2-Python",
            "commit": upstream_commit,
            "extracted_by": "scripts/extract_linage2_model.py",
            "artifact_sha256": {p.name: _sha256(p) for p in artifacts},
        },
        "formula": "years_j = (clip((BoxCox_j(x_j) - median_sj) / mad_sj, -z_max, z_max) - mu_z_sj) "
                   "* w_sj / 12; delta = sum_j years_j + (age_months - mu_age_s) * w_age_s / 12. "
                   "median/mad null: no normalisation (clip only). boxcox_lambda null: no "
                   "transform; 0: natural log.",
        "z_max": float(bundle.zScoreMax),
        "lab_inputs": lab_inputs,
        "features": [{"code": c, "boxcox_lambda": lam[c]} for c in features],
        "derived": {
            "LDLV": {"from": ["LBDTCSI", "LBDSTRSI", "LBDHDLSI"],
                     "formula": "LBDTCSI - LBDSTRSI / 5 - LBDHDLSI (mmol/L)"},
            "crAlbRat": {"from": ["URXUMASI", "URXUCRSI"],
                         "formula": "URXUMASI / (URXUCRSI * 1.1312e-4) (mg albumin per g creatinine)"},
            "fs1Score": {"from": [q for q in QUESTIONNAIRE_ITEMS if q not in ("MCQ160B", "HUQ010", "HUQ020", "HUQ050")],
                         "formula": "fraction of 22 items answered 1 (yes); DIQ010 also counts 3 (borderline). "
                                    "MCQ160B (heart failure) is read by the service but not counted, as upstream."},
            "fs2Score": {"from": ["HUQ010", "HUQ020"],
                         "formula": "(2 if HUQ010 == 4 else 4 if HUQ010 == 5 else 0) * "
                                    "(1 - 0.5*(HUQ020 == 1) + (HUQ020 == 2))"},
            "fs3Score": {"from": ["HUQ050"], "formula": "HUQ050 category code (77, 99 -> 0)"},
        },
        "questionnaire_items": list(QUESTIONNAIRE_ITEMS),
        "questionnaire_defaults": QUESTIONNAIRE_DEFAULTS,
        "cotinine_levels": {
            "0": "< 10 ng/mL (non-smoker)", "1": "10-100 ng/mL", "2": "100-200 ng/mL", "3": ">= 200 ng/mL",
            "note": "the training scale (digiCot). The service maps a daily smoker to 2 and never "
                    "uses 3; this port keeps the training scale.",
        },
        "sex": {"male": sex_block(1), "female": sex_block(2)},
        "imputation": {
            "ages": list(AGE_TABLE),
            "method": "median of the same-sex NHANES 1999-2000 non-accidental-death pool in "
                      "[age-5, age+5] years, truncated at 40 (the service's window); cotinine "
                      "digitized before the median",
            **imputation,
        },
        "nhanes_range": {c: [float(LAB_RANGES[c][0]), float(LAB_RANGES[c][1])] for c in lab_inputs},
        # The adult reference's 0.5th / 50th / 99.5th percentiles, both sexes. What
        # "typical" means when a value arrives without a unit (core/patient_text.py):
        # the extremes above are too wide to tell mmol/mol from % for HbA1c.
        "nhanes_central": {
            c: [_f(adults[c].quantile(q)) for q in (0.005, 0.5, 0.995)]
            for c in lab_inputs if c != "LBXCOT"
        },
        "descriptions": {c: DESCRIPTION_FIXES.get(c, nhanes_desc.get(c, "")) for c in lab_inputs},
    }
    return model, svc


def golden_cases(lin: Path, svc, n: int, seed: int) -> list[dict]:
    """Score `n` random partial panels with the SERVICE's process_payload."""
    import numpy as np

    with contextlib.redirect_stdout(io.StringIO()):
        from db_mapping import UNIT_SCALE
        from imputation import count_reference_values
        from ui_sliders import LAB_VARIABLES
        bundle = svc.load_linage2_bundle("artifacts")
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        sex = 1 if i % 2 == 0 else 2
        age = int(rng.integers(40, 85)) if i % 7 else int(rng.integers(25, 40))
        med = count_reference_values(sex=sex, age=age * 12)
        labs: dict[str, float] = {"LBXCOT": float(rng.integers(0, 4))}
        for c in LAB_VARIABLES:
            if c == "LBXCOT" or rng.random() < 0.35:
                continue
            labs[c] = round(float(med[c]) * float(rng.lognormal(0.0, 0.3)), 4)
        questionnaire: dict[str, int] = {}
        if i % 3 == 0:
            for q in QUESTIONNAIRE_ITEMS:
                if q == "HUQ010":
                    questionnaire[q] = int(rng.integers(1, 6))
                elif q == "HUQ020":
                    questionnaire[q] = int(rng.integers(1, 4))
                elif q == "HUQ050":
                    questionnaire[q] = int(rng.integers(0, 6))
                elif q == "DIQ010":
                    questionnaire[q] = int(rng.choice([1, 2, 3]))
                else:
                    questionnaire[q] = 1 if rng.random() < 0.2 else 2
        # Hand the service its own lab and questionnaire dicts, in NHANES units and
        # codes, exactly as `_split_and_remap_survey_items` would return them. The DB
        # adapter in front of it (ques_id mapping, UNIT_SCALE) is bypassed on purpose:
        # it has no ids for the three lipids, cannot express cotinine level 3, and is
        # not the model. What is tested is everything from imputation onwards.
        # process_payload multiplies by UNIT_SCALE AFTER this step (its DB units ->
        # NHANES units), so the dict is handed over pre-divided, as the DB would send it.
        payload = {"biometrics": {"age": age, "gender": sex}, "surveys": []}
        original = svc._split_and_remap_survey_items
        db_labs = {c: v / UNIT_SCALE[c] if c in UNIT_SCALE else v for c, v in labs.items()}

        def direct(_payload, _labs=db_labs, _ques=dict(questionnaire)):
            return dict(_labs), dict(_ques), []

        svc._split_and_remap_survey_items = direct
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = svc.process_payload(payload, bundle)
        finally:
            svc._split_and_remap_survey_items = original
        assert out["success"], out
        data = out["data"]
        cases.append({
            "sex": "Male" if sex == 1 else "Female",
            "age": age,
            "labs": labs,
            "questionnaire": questionnaire,
            "service": {
                "delta_ba_ca": data["delta_ba_ca"],
                "biological_age": data["biological_age"],
                "feature_contributions": {fc["feature"]: fc["contribution_years"]
                                          for fc in data["feature_contributions"]},
                "imputed_features": data["imputed_features"],
            },
        })
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--linage2-repo", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20251004)
    args = ap.parse_args()
    lin = args.linage2_repo.resolve()
    model, svc = extract(lin)
    cases = golden_cases(lin, svc, args.cases, args.seed)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    head = json.dumps({
        "source": model["source"],
        "note": "Scored by Rejuve/LinAge2-Python's own process_payload (numpy-2 shims "
                "only). Cotinine is always supplied: its imputation deliberately differs.",
    }, indent=1)
    body = ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in cases)
    OUT_GOLDEN.write_text(head[:-2] + ',\n "cases": [\n' + body + "\n ]\n}\n", encoding="utf-8")
    print(f"wrote {OUT_MODEL.relative_to(REPO)} ({OUT_MODEL.stat().st_size / 1024:.0f} KB) "
          f"and {OUT_GOLDEN.relative_to(REPO)} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
