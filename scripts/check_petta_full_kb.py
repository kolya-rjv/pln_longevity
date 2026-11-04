#!/usr/bin/env python3
"""Measure full-KB load/query behavior without an LLM or network calls.

Optionally add N synthetic ground facts to check capacity under an explicit
per-child memory limit. Reports failures as well as successes; no partial answer
is considered a pass. Run from any directory with the project's Python env.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-atoms", type=int, default=0)
    parser.add_argument("--memory-limit-mb", type=int, default=2048)
    parser.add_argument("--deadline", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.synthetic_atoms < 0:
        parser.error("--synthetic-atoms must be nonnegative")
    os.environ["PLN_RUNTIME"] = "petta"
    os.environ["PLN_CHAINER"] = "legacy"
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "pln_chat"))
    from core.knowledge_base import full_kb_paths, unique_fact_blocks
    from core.metta_runtime import PettaRuntime, split_top_level_forms

    started = time.monotonic()
    paths = full_kb_paths()
    blocks = [p.read_text(encoding="utf-8") for p in paths]
    source_forms = sum(len(split_top_level_forms(block)) for block in blocks)
    blocks = unique_fact_blocks(blocks)
    unique_forms = sum(len(split_top_level_forms(block)) for block in blocks)
    text = "\n".join(blocks)
    report = {"files": [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in paths],
              "source_forms": source_forms, "unique_forms": unique_forms,
              "synthetic_atoms": args.synthetic_atoms, "memory_limit_mb": args.memory_limit_mb,
              "deadline_s": args.deadline}
    queries = [
        "!(infer &self CellularSenescence CoronaryHeartDisease)",
        "!(let $rows (collapse (match &self (InstanceOf $r Experiment) $r)) (size-atom $rows))",
        "!(let $rows (collapse (match &self (InstanceOf $r CellSenescenceRecord) $r)) (size-atom $rows))",
        '!(match &self (GeneSymbol GHR $name) $name)',
    ]
    if args.synthetic_atoms:
        text += "\n" + "\n".join(f"(FullKBCapacity {i})" for i in range(args.synthetic_atoms))
        queries += [f"!(match &self (FullKBCapacity {i}) Present)"
                    for i in (0, args.synthetic_atoms // 2, args.synthetic_atoms - 1)]
    report["kb_bytes"] = len(text.encode())
    report["preparation_s"] = round(time.monotonic() - started, 3)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}), flush=True)
    runtime = PettaRuntime(memory_limit_mb=args.memory_limit_mb)
    try:
        answers = runtime.run_program(text, queries, deadline_s=args.deadline)
        assert answers[0] and answers[1] and answers[2], "a full-KB probe returned no answer"
        if args.synthetic_atoms:
            assert answers[-3:] == [["Present"]] * 3
        report.update(status="passed", answers=answers, runtime=runtime.last_run)
    except Exception as exc:
        report.update(status="failed", error=str(exc), error_kind=getattr(exc, "kind", type(exc).__name__))
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
