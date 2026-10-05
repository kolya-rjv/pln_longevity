"""A program made only of human-evidence forms runs in a stack that answers it.

`(human-evidence &self Metformin)` aborts hyperon 0.2.10 in the full shared space (the head-symbol edge:
rc -6, trie.rs:179), so the chat's rule 14 — "what does the evidence say about metformin in humans?" — crashed
the worker. The layer needs only four files (core.pln_runner.HUMAN_EVIDENCE_FILES). Every MeTTa run here is a
SUBPROCESS: an abort must fail one test, not kill pytest.

    pytest tests/test_human_evidence_route.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")

import api as api_module  # noqa: E402
from core.pln_runner import (HUMAN_EVIDENCE_FILES, human_evidence_stack, is_human_evidence_program,  # noqa: E402
                             patient_stack)


@pytest.mark.parametrize("program, expected", [
    ("(human-evidence &self Metformin)", True), ("!(human-evidence &self Metformin)", True),
    ("(human-evidence-interventions &self)", True), ("(human-evidence &self Metformin)\n(human-evidence &self Omega3)", True),
    ("!(human-evidence-absent &self X)", True),
    ("(human-evidence &self Metformin)\n(infer &self Metformin CoronaryHeartDisease)", False),   # mixed: not routed
    ("(infer &self Metformin CoronaryHeartDisease)", False), ("", False),
    (";; (human-evidence &self Metformin)", False), ("(match &self (HumanEvidence $r $i $o $d $n $x) $r)", False),
    ("(diagnose-patient &self Caller_Me)", False),
])
def test_only_a_program_made_of_human_evidence_forms_is_routed(program, expected):
    assert is_human_evidence_program(program) is expected


def test_both_surfaces_route_it_to_the_same_files():
    import app as app_module
    runtime = api_module._runtime_kb_paths()
    q = "(human-evidence &self Metformin)"
    assert api_module._generic_kb(q) == human_evidence_stack(runtime)
    assert {p.name for p in api_module._generic_kb(q)} == set(HUMAN_EVIDENCE_FILES)
    assert app_module._generic_kb(q) == human_evidence_stack(app_module._ALL_KB_PATHS)
    # everything else routes as before
    assert api_module._generic_kb("(infer &self Metformin CoronaryHeartDisease)") == runtime
    assert api_module._generic_kb("(predict-risk-patient &self Patient001)") == patient_stack(runtime)


def _run(kb_files: list[Path], query: str) -> dict:
    probe = "\n".join([
        "import sys, json",
        f"sys.path.insert(0, {str(PLN_CHAT)!r})",
        "import api",
        "from core.pln_runner import run_query",
        "from pathlib import Path",
        "kb = [Path(p) for p in json.loads(sys.argv[1])]",
        "r = run_query(sys.argv[2], kb_files=kb)",
        "print('RESULT ' + json.dumps({'status': r.status, 'atoms': [x.atom for x in r.results]}))",
    ])
    done = subprocess.run([sys.executable, "-c", probe, json.dumps([str(p) for p in kb_files]), query],
                          capture_output=True, text=True, timeout=600, cwd=str(REPO))
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    out = json.loads(line[0][7:]) if line else {"status": "abort", "atoms": []}
    out["rc"] = done.returncode
    return out


@pytest.mark.slow
@pytest.mark.parametrize("query, needle", [
    ("!(human-evidence &self Metformin)", "(HumanEvidence Bannister2014_Metformin_Survival Metformin"),
    ("!(human-evidence &self Rapamycin)", "(NoHumanEvidenceRecord Rapamycin"),
    ("!(human-evidence &self Berberine)", "(NoHumanEvidenceRecord Berberine"),
    ("!(human-evidence &self Omega3)", "(HumanEvidenceCrossReferenced Omega3"),
    ("!(human-evidence-interventions &self)", "Metformin"),
])
def test_the_routed_program_answers_what_the_full_space_aborts_on(query, needle):
    runtime = api_module._runtime_kb_paths()
    routed = _run(api_module._generic_kb(query), query)
    assert routed["rc"] == 0 and routed["status"] == "ok" and needle in " ".join(routed["atoms"])
    control = _run(runtime, query)
    if "interventions" not in query:                     # the control: the full space does abort on the per-drug form
        assert control["rc"] != 0, (
            "the full stack no longer aborts on this form: the route is still correct, re-measure whether it is needed")
