#!/usr/bin/env python3
"""Drive the 10 DrugAge chat test queries through the REAL routing + engine.

For each natural-language query this runs the exact pipeline the chat app uses
once the few-shot-primed translator has emitted its MeTTa form:

    NL --(translate, few-shot deterministic)--> metta_query
       --parse_drugage_query--> [compounds]
       --route_drugage_ranking--> PLNRunResult   (scoped hyperon 0.2.10 space)

`translate()` itself is an OpenAI call (needs a key), but it is a thin,
low-temperature mapping pinned by the few-shots in
prompts/few_shot_examples.json — so the `metta_query` below is what it emits.
Everything downstream (the routing decision AND the scoped DrugAge inference) is
executed here for real against `build/drugage_etl.metta`; nothing is hypothetical.

Usage:
    bash scripts/run_etl.sh                 # generate build/drugage_etl.metta
    PLN_RUNTIME_AVAILABLE=true python scripts/drive_drugage_chat_queries.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PLN_RUNTIME_AVAILABLE", "true")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "pln_chat"))

from core.drugage_router import parse_drugage_query, route_drugage_ranking  # noqa: E402
from ontology.drugage_selector import BUILD_DRUGAGE  # noqa: E402

# (label, NL query, metta_query the few-shot-primed translator emits)
QUERIES: list[tuple[str, str, str]] = [
    ("Q1 varied real set",
     "Between rapamycin, curcumin and metformin, which one most lowers mortality "
     "by extending lifespan?",
     "(rank-drugage-lifespan (Rapamycin Curcumin Metformin))"),
    ("Q2 ITP-negative down-weight",
     "Everyone hypes resveratrol — rank it honestly against rapamycin and "
     "metformin for real lifespan and mortality benefit.",
     "(rank-drugage-lifespan (Resveratrol Rapamycin Metformin))"),
    ("Q3 magnitude vs confidence (worm vs ITP mouse)",
     "A huge lifespan gain was reported for trimethadione in worms — does that "
     "outweigh acarbose's smaller but ITP-grade mouse result?",
     "(rank-drugage-lifespan (Trimethadione Acarbose))"),
    ("Q4 single-compound effect lookup",
     "On its own, how much does rapamycin lower mortality through its lifespan "
     "effect?",
     "(rank-drugage-lifespan (Rapamycin))"),
    ("Q5 absent/unconnected compound",
     "Rank dasatinib, rapamycin and acarbose by lifespan-driven mortality benefit.",
     "(rank-drugage-lifespan (Dasatinib Rapamycin Acarbose))"),
    ("Q6 provenance (which PMID)",
     "Rank metformin and acarbose for lifespan benefit and show which study each "
     "ranking rests on.",
     "(rank-drugage-lifespan (Metformin Acarbose))"),
    ("Q7 full canonical spread",
     "Give me a lifespan-benefit leaderboard across rapamycin, resveratrol, "
     "acarbose, metformin and trimethadione.",
     "(rank-drugage-lifespan (Rapamycin Resveratrol Acarbose Metformin Trimethadione))"),
    ("Q8 casing robustness + varied set",
     "compare CAFFEINE, aspirin and Taurine on lifespan and mortality",
     "(rank-drugage-lifespan (CAFFEINE aspirin Taurine))"),
    ("Q9 hyped senolytics, mixed evidence tiers",
     "Rank the senolytics fisetin, quercetin and spermidine by lifespan-extension "
     "benefit.",
     "(rank-drugage-lifespan (Fisetin Quercetin Spermidine))"),
    ("Q10 several absent compounds mixed with present",
     "Which of sildenafil, lithium, aspirin and rapamycin actually has lifespan "
     "evidence, ranked?",
     "(rank-drugage-lifespan (Sildenafil Lithium Aspirin Rapamycin))"),
]

_SCORED = re.compile(r"\(scored\s+(\S+)\s+([-\d.eE]+)\s+\(signed\s+(Pos|Neg)\s+"
                     r"\(stv\s+([-\d.eE]+)\s+([-\d.eE]+)\)")


def _fmt_ranking(atom: str) -> list[str]:
    out = []
    for m in _SCORED.finditer(atom):
        name, score, sign, s, c = m.groups()
        out.append(f"{name} score={float(score):+.4f} ({sign} stv {float(s):.3f} "
                   f"{float(c):.3f})")
    return out


def main() -> None:
    if not BUILD_DRUGAGE.exists():
        sys.exit("build/drugage_etl.metta missing — run `bash scripts/run_etl.sh` first.")
    for label, nl, mq in QUERIES:
        print("=" * 78)
        print(label)
        print(f"  NL:    {nl}")
        print(f"  METTA: {mq}")
        compounds = parse_drugage_query(mq)
        print(f"  parse_drugage_query -> {compounds}  (routed: {compounds is not None})")
        result = route_drugage_ranking(compounds)
        print(f"  status: {result.status}   ({result.query_time_ms} ms)")
        if result.error:
            print(f"  error: {result.error}")
        ranking = _fmt_ranking(result.results[0].atom) if result.results else []
        if ranking:
            print("  RANKING (most protective first):")
            for line in ranking:
                print(f"    • {line}")
        for r in result.results[1:]:
            print(f"    · {r.atom}")
    print("=" * 78)


if __name__ == "__main__":
    main()
