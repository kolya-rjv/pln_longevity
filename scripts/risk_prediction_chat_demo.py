#!/usr/bin/env python3
"""Drive the 10 risk-prediction chat test queries through the REAL pipeline.

For each natural-language query this exercises the SAME path the Gradio app uses:

    translate()  — NL -> MeTTa via the OpenAI translator (needs OPENAI_API_KEY;
                   when absent it returns its "key not set" error, and we fall back
                   to the reference MeTTa the few-shots target — clearly labelled);
    validate()   — symbol/paren check against the parsed inference-stack registry;
    run_query()  — execution against the live hyperon 0.2.10 interpreter, loading
                   the exact _ALL_KB_PATHS set the app loads (repo-root *.metta minus
                   any file over PLN_MAX_KB_FILE_BYTES).

The run_query outputs are REAL hyperon results — no hypothetical values. Run with
PLN_RUNTIME_AVAILABLE=true (and an OPENAI_API_KEY if you also want the live
translation step). Prints one block per query; the results are transcribed into
docs/risk_prediction_chat_test_queries.md.

    PLN_RUNTIME_AVAILABLE=true python scripts/risk_prediction_chat_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PLN_RUNTIME_AVAILABLE", "true")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "pln_chat"))

from config import ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR, PLN_MAX_KB_FILE_BYTES  # noqa: E402
from ontology.loader import load_specific_files  # noqa: E402
from core.context_builder import build_system_prompt  # noqa: E402
from core.llm_translator import translate  # noqa: E402
from core.metta_validator import validate  # noqa: E402
from core.pln_runner import run_query  # noqa: E402

# The inference-stack files the app injects into the translator context (app.py
# _INFERENCE_STACK), used here to build the registry + system prompt.
_INFERENCE_STACK = [
    "system_types.metta", "logical_predicates.metta", "epistemic_calibration.metta",
    "grim_age_core.metta", "grim_age_lu2019_evidence.metta", "evidence_calibration.metta",
    "hallmarks_core.metta", "hallmarks_lopezotin2023_intervention_evidence.metta",
    "mechanistic_bridges.metta", "pln_deduction.metta", "pln_intervention_ranking.metta",
    "pln_abductive_diagnosis.metta", "patient_profile.metta", "pln_counterfactual.metta",
    "pln_risk_prediction.metta",
]


def _all_kb_paths() -> list[Path]:
    """Replicate app.py::_discover_metta_files + _runtime_kb_paths exactly."""
    files: dict[str, Path] = {}
    for base in (ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR):
        if base.exists():
            for p in sorted(base.glob("*.metta")):
                files[p.name] = p
    return [p for p in files.values() if p.stat().st_size <= PLN_MAX_KB_FILE_BYTES]


# (nl, reference MeTTa the few-shots target, expected routed behavior)
QUERIES = [
    ("What is Patient001's 10-year risk of coronary heart disease?",
     "(predict-risk-patient &self Patient001)",
     "Absolute point estimate 0.126 with a propagated CI [0.096, 0.156] at c=0.765, "
     "baseline 0.08, clock multiplier 1.576 — a number an LLM can only hedge about."),

    ("Give me Patient001's cardiovascular risk with a confidence interval.",
     "(predict-risk-patient &self Patient001)",
     "Same estimate: the CI is (1±(1−c)k) around the point, so a lower-confidence "
     "estimate would carry a wider interval — uncertainty is first-class."),

    ("Which biomarkers are driving Patient001's elevated cardiovascular risk, and by "
     "how much?",
     "(risk-decomposition-patient &self Patient001)",
     "Excess 0.046 split by clock-component share: DNAmPAI1 0.00648, DNAmGDF15 0.00540 "
     "(attributed 0.0119) + explicit residual 0.0342; baseline+factors+residual = total."),

    ("How much of Patient001's excess heart-disease risk is left unexplained by their "
     "measured methylation surrogates?",
     "(risk-decomposition-patient &self Patient001)",
     "The residual-excess field: 0.0342 of the 0.046 excess is unattributed (coarse v1 "
     "weights + unmeasured surrogates) — reported honestly, never forced to zero."),

    ("How much would clearing Patient001's senescent cells be expected to lower their "
     "10-year heart-disease risk?",
     "(project-risk-patient &self Patient001 CellularSenescence)",
     "Scenario B. Reuses the counterfactual: clock drops 0.222, risk falls to 0.118 "
     "(reduction 0.0077), Via (DNAmPAI1 DNAmGDF15), confidence 0.318."),

    ("For Patient001, does clearing senescent cells cut cardiovascular risk more than "
     "just reducing inflammation?",
     "(project-risk-patient &self Patient001 CellularSenescence)\n"
     "(project-risk-patient &self Patient001 ChronicInflammation)",
     "Yes: reduction_B 0.0077 > reduction_A 0.0043 — senescence is upstream and reaches "
     "both clock surrogates, so it buys a bigger absolute-risk drop."),

    ("If Patient001 took metformin for metabolic health, how much would their "
     "cardiovascular risk drop?",
     "(project-risk-patient &self Patient001 Metformin)",
     "Metformin resolves to InsulinResistance, whose axis reaches NO clock surrogate, so "
     "the clock-based risk change is honestly ~0 (reduction 0.0, empty Via) — not fabricated."),

    ("How certain is the estimate that clearing senescence lowers Patient001's CHD "
     "risk, compared with reducing inflammation?",
     "(project-risk-patient &self Patient001 CellularSenescence)\n"
     "(project-risk-patient &self Patient001 ChronicInflammation)",
     "Uncertainty grows with route length: the senescence projection carries c=0.318 "
     "(2–3 hops) vs inflammation c=0.65 (single hop). The bigger effect is less certain."),

    ("What would Patient001's projected heart-disease risk be if we gave them "
     "elamipretide?",
     "(project-risk-patient &self Patient001 Elamipretide)",
     "Elamipretide has no Effect edge -> the counterfactual is unresolvable -> the risk "
     "projection is OMITTED (no result), never a fabricated placement."),

    ("What is Patient002's 10-year cardiovascular risk?",
     "(predict-risk-patient &self Patient002)",
     "A different patient: 61yo male -> the 60–69 baseline band (0.15), clock z 1.3 -> "
     "point 0.217, CI [0.166, 0.268] — same model, personalized inputs."),
]


def main() -> None:
    paths = [ONTOLOGY_DIR / f for f in _INFERENCE_STACK]
    registry, raw = load_specific_files(paths)
    system_prompt = build_system_prompt(registry, raw)
    kb = _all_kb_paths()

    print(f"# Risk-prediction chat demo — {len(QUERIES)} queries")
    print(f"# hyperon runtime KB: {len(kb)} files "
          f"(pln_risk_prediction.metta included: "
          f"{'pln_risk_prediction.metta' in [p.name for p in kb]})\n")

    for i, (nl, ref_metta, expected) in enumerate(QUERIES, 1):
        # 1) real translate() — reports missing key in this sandbox; falls back.
        tr = translate(nl, system_prompt, history=[], model="gpt-5.4-mini",
                       temperature=0.0)
        used_metta = tr.metta_query if (tr.ok and tr.metta_query.strip()) else ref_metta
        translate_note = ("live translator" if (tr.ok and tr.metta_query.strip())
                          else f"translator unavailable ({tr.error}); using reference MeTTa")

        # 2) real validate() against the parsed registry.
        val = validate(used_metta, registry)

        # 3) real run_query() against live hyperon 0.2.10.
        res = run_query(used_metta, kb_files=kb)

        print(f"## Q{i}. {nl}")
        print(f"MeTTa      : {used_metta.replace(chr(10), '  |  ')}")
        print(f"translate(): {translate_note}")
        print(f"validate() : valid={val.valid} issues={val.issues}")
        print(f"run_query(): status={res.status} mode={res.mode} err={res.error}")
        print(f"expected   : {expected}")
        if res.results:
            for atom in res.results:
                print(f"ACTUAL     : {atom.atom}")
        else:
            print("ACTUAL     : (empty — no result)")
        print()


if __name__ == "__main__":
    main()
