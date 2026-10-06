"""Programs that read no patient run in the generic stack (core.pln_runner.generic_stack).

hyperon 0.2.10 (the latest release) decodes a space's trie keys wrongly once the space
stores more than about 1,024 distinct key atoms — every distinct symbol, variable, number
and string (upstream issues #1076 and #1095, fix in PR #1081, unreleased). Such a space
loads and answers exact lookups, and aborts the process when a query reads back an atom
stored late. The full shared stack is past that edge: the pair template the translator is
taught, (match &self (TargetsHallmark $i $h) (pair $i $h)), aborted it. The generic stack
leaves out the four patient layers, which no program that reads no patient needs. These
tests pin that the programs that aborted now answer, that wherever the full stack answers
the generic stack answers identically, that nothing outside the patient layers uses their
definitions, and that there is margin.

Every MeTTa run here is a SUBPROCESS: an abort must fail one test, not kill pytest.

    pytest tests/test_generic_stack.py -q
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
from core.pln_runner import (  # noqa: E402
    GENERIC_STACK_EXCLUDED,
    defined_heads,
    generic_stack,
    human_evidence_stack,
    needs_patient_layers,
    patient_layer_symbols,
    patient_stack,
)

#: programs that aborted the full shared stack (each seen live, or in the chat's replay of
#: a stakeholder's session), with the number of answers the generic stack gives
ABORTED = [
    ("(match &self (TargetsHallmark $i $h) (pair $i $h))", 16),
    ("(match &self (SupportedByPublication $i $p) $i)", 42),
    ("(match &self (PartOf $component GrimAge) (pair $component (match &self (Reflects $component $what) $what)))", 7),
    ("(match &self (, (EvidenceIntervention $r $i) (EvidenceSpeciesModel $r $s) (EvidenceHallmark $r $h)) "
     "(pair $i (pair $s $h)))", 14),
    ("(match &self (, (TargetsHallmark Metformin $h) (SupportedByPublication Metformin $p)) (pair $h $p))", 1),
    ("(match &self (HallmarkComponent $c $h) (foo $c $h))", 71),
    ("(match &self (Interaction $a $b $n) (pair $a $b))", 1),
    ("(human-evidence &self Metformin)\n(hallmarks-of &self Metformin)", 3),
]
#: generic programs the full stack does answer: the generic stack must give the same atoms
ANSWERED = [
    "(infer &self CellularSenescence CoronaryHeartDisease)",
    "(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)",
    "(rank-interventions &self (DasatinibPlusQuercetin Fisetin Spermidine) CoronaryHeartDisease)",
    "(diagnose &self (CellularSenescence ChronicInflammation MitochondrialDysfunction) (CRP))",
    "(hallmarks-of &self Rapamycin)",
    "(interventions-for &self CellularSenescence)",
    "(match &self (TargetsHallmark $i $h) ($i $h))",
    "(match &self (PartOf $c GrimAge) $c)",
    "(match &self (, (EvidenceIntervention $r Spermidine) (EvidenceHallmark $r $h)) $h)",
]


def _run(stack: str, query: str, extra: str = "") -> dict:
    """Run one query in a fresh process against the 'full', 'generic' or 'routed' stack."""
    probe = "\n".join([
        "import sys, json",
        f"sys.path.insert(0, {str(PLN_CHAT)!r})",
        "import api",
        "from core.pln_runner import generic_stack, run_query",
        "rt = api._runtime_kb_paths()",
        "kb = {'full': rt, 'generic': generic_stack(rt), 'routed': api._generic_kb(sys.argv[2])}[sys.argv[1]]",
        "r = run_query(sys.argv[2], kb_files=kb, extra_atoms=sys.argv[3] or None)",
        "print('RESULT ' + json.dumps({'status': r.status, 'atoms': [x.atom for x in r.results]}))",
    ])
    done = subprocess.run([sys.executable, "-c", probe, stack, query, extra],
                          capture_output=True, text=True, timeout=600, cwd=str(REPO))
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    out = json.loads(line[0][7:]) if line else {"status": "abort", "atoms": []}
    out["rc"] = done.returncode
    return out


# ═══════════════════════════ what it is ═══════════════════════════════════════

def test_the_generic_stack_is_the_runtime_stack_minus_the_four_patient_layers():
    runtime = api_module._runtime_kb_paths()
    names = [p.name for p in runtime]
    assert set(GENERIC_STACK_EXCLUDED) <= set(names), "a patient layer left the runtime stack"
    assert [p.name for p in generic_stack(runtime)] == [n for n in names if n not in GENERIC_STACK_EXCLUDED]
    # the patient stack keeps every patient layer: what the generic stack leaves out, it has
    assert set(GENERIC_STACK_EXCLUDED) <= {p.name for p in patient_stack(runtime)}


def test_nothing_outside_the_patient_layers_uses_their_definitions():
    """The claim that makes leaving them out safe: no other file of the stack calls or reads
    anything only they define. A file that starts to has to move with them, or they back."""
    runtime = api_module._runtime_kb_paths()
    only = patient_layer_symbols(tuple(runtime))
    assert {"predict-risk-patient", "decompose-grimage", "recommend-supplements", "diagnose-patient"} <= only
    for path in generic_stack(runtime):
        text = "\n".join(line.split(";")[0] for line in path.read_text(encoding="utf-8").splitlines())
        used = {t for t in only if f"({t} " in text or f"({t})" in text}
        assert not used, f"{path.name} uses patient-layer definitions {sorted(used)}"
    assert all(defined_heads(p) for p in runtime if p.name in GENERIC_STACK_EXCLUDED)


@pytest.mark.parametrize("program, stack", [
    ("(match &self (TargetsHallmark $i $h) (pair $i $h))", "generic"),
    ("(infer &self CellularSenescence CoronaryHeartDisease)", "generic"),
    ("(predict-risk-patient &self Patient001)", "patient"),
    ("(match &self (MeasuredZ $p CRP $z) ($p $z))", "patient"),
    ("(curated-baseline Male 60)", "patient"),                 # a patient layer's helper, no patient named
    ("(decompose-grimage &self $p)", "patient"),
    ("(human-evidence &self Metformin)", "human-evidence"),
    ("(human-evidence &self Metformin)\n(hallmarks-of &self Metformin)", "generic"),
])
def test_a_program_runs_where_its_forms_are_and_both_surfaces_agree(program, stack):
    import app as app_module
    runtime = api_module._runtime_kb_paths()
    expected = {"generic": generic_stack, "patient": patient_stack,
                "human-evidence": human_evidence_stack}[stack](runtime)
    assert api_module._generic_kb(program) == expected
    assert [p.name for p in app_module._generic_kb(program)] == [p.name for p in expected]


@pytest.mark.parametrize("program, needs", [
    ("(curated-baseline Male 60)", True), ("(predict-risk-patient &self Patient001)", True),
    ("(recommend-supplements &self $p (Omega3))", True), ("(match &self (MeasuredZ $p CRP $z) $z)", False),
    ("(match &self (TargetsHallmark $i $h) (pair $i $h))", False), ("(infer &self A B)", False),
    ("(match &self (Note \"curated-baseline\") $x)", False),        # a word in a string is not a form
])
def test_a_patient_layer_form_is_recognised_without_a_list(program, needs):
    assert needs_patient_layers(program, api_module._runtime_kb_paths()) is needs


# ═══════════════════════════ it answers what aborted ══════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("program, n", ABORTED, ids=[f"aborted-{i}" for i in range(len(ABORTED))])
def test_the_programs_that_aborted_the_full_stack_answer(program, n):
    out = _run("routed", program)
    assert out["rc"] == 0 and out["status"] == "ok" and len(out["atoms"]) == n, out


@pytest.mark.slow
def test_the_full_stack_still_aborts_the_control():
    """The control. If the full stack stops aborting, hyperon changed (#1081 released?): the
    generic stack is still correct, but re-measure, and the patient-stack note may be moot."""
    assert _run("full", ABORTED[0][0])["rc"] != 0


# ═══════════════════════════ it changes no answer ═════════════════════════════

@pytest.mark.slow
@pytest.mark.parametrize("program", ANSWERED)
def test_where_the_full_stack_answers_the_generic_stack_answers_identically(program):
    full, generic = _run("full", program), _run("generic", program)
    assert full["rc"] == 0 and full["status"] in ("ok", "empty"), f"control: the full stack no longer answers {program}"
    assert generic["rc"] == 0 and generic["atoms"] == full["atoms"]


# ═══════════════════════════ margin ═══════════════════════════════════════════

@pytest.mark.slow
def test_the_generic_stack_keeps_a_margin():
    """48 rows of three new key atoms each on top of the generic stack (it survives 77), read
    back: the atoms stored last are the ones a wrong decode would lose first."""
    pad = "\n".join(f'(ProbeHead{i} ProbeSym{i} "probe {i}")' for i in range(48))
    for program in (ABORTED[0][0], "(match &self (ProbeHead47 $s $t) (pair $s $t))"):
        out = _run("generic", program, pad)
        assert out["rc"] == 0 and out["status"] == "ok", (
            "the generic stack no longer tolerates 144 more key atoms; something spent the margin "
            "(a new layer belongs in a scoped stack, or another layer must leave this one)")
