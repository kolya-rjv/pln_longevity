"""Regenerate tests/fixtures/drugage_fold/three_compounds.metta and expected_three_compounds.json.

    python docs/petta_migration/fork_design/b2/make_fixtures.py [--csv path/to/drugage.csv]

The .metta is the verbatim row blocks of Ethanol, Resveratrol, Metformin and Rapamycin as `drugage_etl.py` writes
them from data/drugage/dataset.zip (regenerated here, not read from build/). The JSON is computed by the
INDEPENDENT oracle tests/fixtures/drugage_fold/fold_oracle.py, never by pln_chat/ontology/drugage_fold.py, so the
test that compares the two is not circular.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "pln_chat"))
FIX = REPO / "tests" / "fixtures" / "drugage_fold"
COMPOUNDS = ["Ethanol", "Resveratrol", "Metformin", "Rapamycin"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    from ontology.drugage_scoring import load_knobs
    from ontology.drugage_selector import load_rows

    etl = _load(REPO / "drugage_etl.py", "drugage_etl_b2")
    oracle = _load(FIX / "fold_oracle.py", "fold_oracle_b2")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = args.csv
        if csv_path is None:
            csv_path = Path(tmp) / "drugage.csv"
            with zipfile.ZipFile(REPO / "data" / "drugage" / "dataset.zip") as z:
                csv_path.write_bytes(z.read("drugage.csv"))
        out = Path(tmp) / "drugage_etl.metta"
        etl.generate(csv_path, out)
        rows = load_rows(out)

    knobs = load_knobs()
    keep = [r for r in rows if r.compound in COMPOUNDS]
    keep.sort(key=lambda r: (COMPOUNDS.index(r.compound), int(r.row_id.split("_")[1])))
    header = ("; Real DrugAge rows (verbatim row blocks from drugage_etl.py applied to data/drugage/dataset.zip):\n"
              f"; {', '.join(COMPOUNDS)}. Committed so the paper-level fold tests are deterministic without the ETL build.\n"
              "; Regenerate with docs/petta_migration/fork_design/b2/make_fixtures.py; tests/test_drugage_fold.py checks\n"
              "; that this file is still what the ETL writes.\n\n")
    (FIX / "three_compounds.metta").write_text(header + "\n\n".join(r.block for r in keep) + "\n", encoding="utf-8")

    expected = {}
    for c in COMPOUNDS:
        rs = [r for r in keep if r.compound == c]
        plain = [{"row_id": r.row_id, "compound": r.compound, "clade": knobs.species_clade.get(r.species or ""),
                  "itp": r.is_itp, "sig": r.significance or "Unreported", "pct": r.avg_change,
                  "pmid": r.pmid} for r in rs]
        res = oracle.fold(plain)
        expected[c] = {
            "rows": len(rs),
            "papers": len({r.pmid for r in rs}),
            "Neg": res.get("Neg"), "Pos": res.get("Pos"), "net": res["net"],
        }
    (FIX / "expected_three_compounds.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
    print({c: (e["rows"], e["papers"], round(e["net"], 6)) for c, e in expected.items()})


if __name__ == "__main__":
    main()
