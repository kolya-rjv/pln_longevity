#!/usr/bin/env python3
"""Drive the 10 supplement-recommendation chat test queries through the REAL pipeline.

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
docs/supplement_chat_test_queries.md.

    PLN_RUNTIME_AVAILABLE=true python scripts/supplement_chat_demo.py
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
    "pln_risk_prediction.metta", "supplement_evidence.metta",
    "pln_supplement_recommendation.metta",
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
    ("What supplements should Patient001 consider, and how strong is the evidence "
     "for each?",
     "(recommend-supplements-patient &self Patient001)",
     "Tiered rec: Tier1 Omega3 (MultipleHumanTrials); Tier2 NMN then Fisetin; "
     "NotRecommended Resveratrol; Berberine OMITTED (metabolic, irrelevant); no "
     "interactions (no meds)."),

    ("Should Patient001 take resveratrol for their elevated GrimAge?",
     "(supplement-for-patient &self Patient001 Resveratrol)",
     "The VETO: Resveratrol is relevant by mechanism (Targets the senescence markers) "
     "but ITP_Negative -> tier NotRecommended. Declined on the negative trial, not the "
     "mechanism — the restraint an LLM lacks."),

    ("Recommend supplements for Patient002 and flag any interactions with their "
     "medications.",
     "(recommend-supplements-patient &self Patient002)",
     "Personalization reorder: Berberine -> Tier1 (metabolic markers), the senescence "
     "supplements omitted; the Berberine<->Metformin interaction FLAGGED against "
     "Patient002's metformin."),

    ("Does the same supplement pool give Patient002 different advice than Patient001?",
     "(recommend-supplements-patient &self Patient002)\n"
     "(recommend-supplements-patient &self Patient001)",
     "Same five candidates, DIFFERENT output: Patient002 gets Berberine (metabolic), "
     "Patient001 gets Omega3/Fisetin/NMN (senescence/inflammation) — relevance reorders, "
     "it does not merely amplify."),

    ("Would berberine help Patient001?",
     "(supplement-for-patient &self Patient001 Berberine)",
     "Berberine acts on the metabolic axis; Patient001 has no metabolic marker elevated "
     "-> irrelevant -> OMITTED (empty), never a fabricated placement."),

    ("Which of Patient001's elevated markers would omega-3 actually address?",
     "(supplement-for-patient &self Patient001 Omega3)",
     "Provenance: Omega3 is Tier1 and its (Targets ...) names CRP and DNAmGDF15 — the "
     "inflammation markers it protectively reaches; the recommendation is auditable."),

    ("Rank omega-3, fisetin and NMN for Patient001 by how well they fit their "
     "biomarker profile.",
     "(recommend-supplements &self Patient001 (Omega3 Fisetin NMN))",
     "Explicit pool: Omega3 Tier1; NMN, Fisetin Tier2 (sorted by score). Each carries "
     "its evidence tier + confidence + targeted markers."),

    ("Is the evidence for Fisetin as strong as for a pharmaceutical like metformin?",
     "(supplement-for-patient &self Patient001 Fisetin)",
     "Layer-4 honesty: Fisetin's evidence is AnimalStudies_Single (conf 0.5), well below "
     "a pharmaceutical's trial-grade confidence — supplements are tiered lower, not "
     "conflated with RCT evidence."),

    ("Give Patient002 a supplement plan and show which markers each recommendation "
     "targets.",
     "(recommend-supplements-patient &self Patient002)",
     "Berberine Tier1 with (Targets (FastingGlucose HbA1c)); Interactions flags "
     "Berberine<->Metformin. The metabolic profile drives a wholly different plan than "
     "Patient001's."),

    ("Between fisetin and NMN, which fits Patient001's profile better?",
     "(recommend-supplements &self Patient001 (Fisetin NMN))",
     "Both Tier2, sorted by personalized score: NMN (single mito hop, higher-confidence "
     "transmission to DNAmGDF15) ranks above Fisetin here; each lists its targeted "
     "markers for the comparison."),
]


def main() -> None:
    paths = [ONTOLOGY_DIR / f for f in _INFERENCE_STACK]
    registry, raw = load_specific_files(paths)
    system_prompt = build_system_prompt(registry, raw)
    kb = _all_kb_paths()

    print(f"# Supplement recommendation chat demo — {len(QUERIES)} queries")
    print(f"# hyperon runtime KB: {len(kb)} files "
          f"(pln_supplement_recommendation.metta included: "
          f"{'pln_supplement_recommendation.metta' in [p.name for p in kb]})\n")

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
