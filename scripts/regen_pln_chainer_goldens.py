#!/usr/bin/env python3
"""Regenerate the pln goldens of tests/test_pln_chainer.py (tests/fixtures/pln_chainer/).

The goldens are the EXACT strings the pln chainer (PLN_CHAINER=pln on PLN_RUNTIME=petta) prints:

  closure_pln.json     every derivation of the four graphs gate 6a runs (patient 100, LinAge2 86, generic 101, patient +
                       the feedback edge 109): chain, sign, strength, confidence, as PeTTa prints them
  composite_pln.json   the 50 composite queries of docs/petta_migration/fork_design/contract/pinned_numbers.json (rankings,
                       diagnosis, counterfactuals, risk, supplements) with the atoms the pln chainer returns
  patient_stack_builtin_golden_pln.json
                       the 32 cases of tests/fixtures/patient_stack_builtin_golden.json (Patient001-003 through diagnosis,
                       observations, marker set, personalised ranking, supplement plan, risk) as the pln chainer prints them

They are separate from the legacy goldens (tests/fixtures/patient_stack_builtin_golden.json, linage2_golden.json), which are
never written here: the pln chainer agrees with the legacy one to rounding (max 2.2e-16) but not digit for digit, so
the legacy files cannot hold under it and must not be changed to fit it. Regenerate only after a deliberate change of the
chainer, and read the diff: a changed digit is a changed number.

    PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/regen_pln_chainer_goldens.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pln_chat"))
sys.path.insert(0, str(REPO / "tests"))
sys.dont_write_bytecode = True

import pytest  # noqa: E402

import test_pln_chainer as T  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "pln_chainer"


def main() -> int:
    if not T.REAL_PETTA:
        print("PeTTa is not installed here (scripts/setup_petta.sh); nothing written", file=sys.stderr)
        return 1
    mp = pytest.MonkeyPatch()
    mp.setattr(T.config, "PLN_RUNTIME", "petta")
    mp.setattr(T.config, "PLN_CHAINER", "pln")
    mp.setattr(T.config, "LOGS_DIR", Path(tempfile.mkdtemp(prefix="pln_goldens_")))
    OUT.mkdir(parents=True, exist_ok=True)
    closure = {}
    for name, g in T.GRAPHS.items():
        rows = g["rows"]()
        got = T.explain_closure(T._stack(g["stack"]), g["extra"], rows)
        closure[name] = [{"chain": list(chain), "sign": v[0], "strength": v[1], "confidence": v[2]}
                         for chain, v in sorted(got.items(), key=lambda kv: (kv[0][0], len(kv[0]), kv[0]))]
        print(f"{name}: {len(got)} pairs")
    (OUT / "closure_pln.json").write_text(json.dumps(closure, indent=1) + "\n", encoding="utf-8")
    queries = [e["query"] for e in T.PINNED]
    groups = T._batch(queries)
    composite = {"about": "What the pln chainer returns for the 50 composite queries of contract/pinned_numbers.json (patient "
                          "stack, one program). Exact strings; the legacy hyperon strings are in that file.",
                 "entries": [{"tag": e["tag"], "query": e["query"], "results": g} for e, g in zip(T.PINNED, groups)]}
    (OUT / "composite_pln.json").write_text(json.dumps(composite, indent=1) + "\n", encoding="utf-8")
    print(f"composite: {len(groups)} queries")
    legacy = json.loads((REPO / "tests" / "fixtures" / "patient_stack_builtin_golden.json").read_text(encoding="utf-8"))
    names = sorted(legacy["cases"])
    cases = T._batch([legacy["cases"][n]["query"] for n in names])
    golden = {"_about": "The 32 cases of patient_stack_builtin_golden.json, answered by the pln chainer (PLN_CHAINER=pln on PeTTa). "
                        "Exact strings. They agree with the legacy recording to 1e-12 and differ from it in the last digit of "
                        "some numbers, so the legacy file cannot be asserted under pln and is not touched; "
                        "tests/test_pln_chainer.py asserts both this file (exactly) and the agreement (to 1e-12).",
              "cases": {n: {"query": legacy["cases"][n]["query"], "atoms": a} for n, a in zip(names, cases)}}
    (OUT / "patient_stack_builtin_golden_pln.json").write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"builtin patients: {len(names)} cases ({sum(1 for a in cases if a)} non-empty)")
    mp.undo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
