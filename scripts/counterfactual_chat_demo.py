#!/usr/bin/env python3
"""Drive the 10 counterfactual chat test queries through the REAL pipeline.

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
docs/counterfactual_chat_test_queries.md.

    PLN_RUNTIME_AVAILABLE=true python scripts/counterfactual_chat_demo.py
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
    ("Which methylation surrogates account for Patient001's high GrimAge, and how "
     "much does each contribute?",
     "(decompose-grimage &self Patient001)",
     "Decompose the elevated GrimAge: DNAmPAI1 (contribution 0.225) + DNAmGDF15 "
     "(0.1875) credited, each traced to its upstream cause(s); attributed 0.4125; "
     "honest residual 1.1875 (unmeasured surrogates + coarse v1 weights)."),

    ("Patient001 has high inflammation — if it were brought back to normal, how much "
     "GrimAge acceleration would that be expected to remove?",
     "(counterfactual-patient &self Patient001 ChronicInflammation)",
     "Scenario A. Inflammation reaches only DNAmGDF15 (single hop), so expected "
     "delta -0.121875, signed Neg (protective), confidence 0.65, Via (DNAmGDF15)."),

    ("What GrimAge reduction could Patient001 expect from a senolytic that clears "
     "their senescent cells?",
     "(counterfactual-patient &self Patient001 CellularSenescence)",
     "Scenario B. Senescence is upstream, reaching BOTH DNAmPAI1 and DNAmGDF15 -> "
     "bigger delta -0.22193, Via (DNAmPAI1 DNAmGDF15); confidence 0.318 (lower, "
     "longer routes)."),

    ("For Patient001, does clearing senescent cells help GrimAge more than just "
     "reducing inflammation?",
     "(counterfactual-patient &self Patient001 CellularSenescence)\n"
     "(counterfactual-patient &self Patient001 ChronicInflammation)",
     "B vs A: |delta_B|=0.222 > |delta_A|=0.122 — clearing senescence helps MORE "
     "because it is upstream and reaches both clock surrogates."),

    ("If Patient001 lowers their CRP, what's the expected effect on their GrimAge?",
     "(counterfactual-patient &self Patient001 CRP)",
     "CRP is a leaf readout — routed THROUGH its shared cause ChronicInflammation "
     "(no fictitious CRP->GrimAge edge). Identical to scenario A: delta -0.121875, "
     "Via (DNAmGDF15)."),

    ("Patient001 is considering metformin for metabolic health — what GrimAge change "
     "should they expect?",
     "(counterfactual-patient &self Patient001 Metformin)",
     "Metabolic axis reaches NO GrimAge surrogate (terminates at FastingGlucose->"
     "CHD). Honest expected-delta 0.0, empty Via () — not a fabricated number."),

    ("How confident is the estimate that clearing senescence lowers Patient001's "
     "GrimAge, versus reducing inflammation — which is more certain?",
     "(counterfactual-patient &self Patient001 CellularSenescence)\n"
     "(counterfactual-patient &self Patient001 ChronicInflammation)",
     "Uncertainty grows with route length: senescence estimate c=0.318 (2-3 hops) < "
     "inflammation c=0.65 (single hop). The bigger effect is the less certain one."),

    ("Which GrimAge components would a senescence-clearing intervention act on for "
     "Patient001, and through what causal route?",
     "(counterfactual-patient &self Patient001 CellularSenescence)\n"
     "(explain &self CellularSenescence DNAmPAI1)",
     "Provenance: the counterfactual credits Via (DNAmPAI1 DNAmGDF15); explain gives "
     "the auditable chain CellularSenescence -> SASP -> DNAmPAI1 (signed Pos)."),

    ("If Patient001 took elamipretide, what GrimAge change would you predict?",
     "(counterfactual-patient &self Patient001 Elamipretide)",
     "Elamipretide has no Effect edge in the KB -> unresolvable -> OMITTED (no "
     "result), never a fabricated placement."),

    ("How much of Patient001's GrimAge acceleration is explained by the GDF15 "
     "methylation surrogate alone?",
     "(grimage-share &self Patient001 DNAmGDF15)",
     "Single-component decomposition: DNAmGDF15 z=1.5, weight 0.125, contribution "
     "0.1875, DrivenBy the three hallmark causes that raise it."),
]


def main() -> None:
    paths = [ONTOLOGY_DIR / f for f in _INFERENCE_STACK]
    registry, raw = load_specific_files(paths)
    system_prompt = build_system_prompt(registry, raw)
    kb = _all_kb_paths()

    print(f"# Counterfactual chat demo — {len(QUERIES)} queries")
    print(f"# hyperon runtime KB: {len(kb)} files "
          f"(pln_counterfactual.metta included: "
          f"{'pln_counterfactual.metta' in [p.name for p in kb]})\n")

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
