"""PeTTa must see all application data, independent of the query's route."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from core import knowledge_base as kb
from core import pln_runner as runner
from core.metta_runtime import PettaRuntime, split_top_level_forms


@pytest.fixture
def small_kb(tmp_path, monkeypatch):
    root = tmp_path / "ontology"
    custom = tmp_path / "custom"
    (root / "build").mkdir(parents=True)
    custom.mkdir()
    (root / "rules.metta").write_text("(RootFact Root)\n")
    (root / "large.metta").write_text("; larger than the Hyperon byte cap\n" * 3000 + "(LargeFact Large)\n")
    (root / "build" / "data.metta").write_text("(GeneratedFact Generated)\n(RootFact Root)\n")
    (custom / "custom.metta").write_text("(CustomFact Custom)\n")
    (root / "pln_chainer.metta").write_text("(= (infer $s $a $b) (alternative))")
    (root / "tests").mkdir()
    (root / "tests" / "fixture.metta").write_text("(TestOnly Fixture)")
    monkeypatch.setattr(config, "ONTOLOGY_DIR", root)
    monkeypatch.setattr(config, "CUSTOM_ONTOLOGY_DIR", custom)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    return root


def test_full_discovery_includes_generated_custom_and_oversized_sources(small_kb):
    assert {p.name for p in kb.full_kb_paths()} == {
        "rules.metta", "large.metta", "data.metta", "custom.metta",
    }


@pytest.mark.parametrize("runtime_name", ["petta", "hyperon"])
def test_runner_expands_only_petta_and_deduplicates_copied_facts(small_kb, monkeypatch, runtime_name):
    captured = {}

    def run_program(text, expressions, **options):
        captured.update(text=text, expressions=expressions)
        return [["ok"]]

    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    monkeypatch.setattr(runner, "get_runtime", lambda: SimpleNamespace(
        name=runtime_name, run_program=run_program, enforces_deadline=False))
    result = runner._runtime_run("(RootFact $x)", 0, [small_kb / "rules.metta"], "(RootFact Root)")
    assert result.ok, result.error
    text = captured["text"]
    if runtime_name == "petta":
        assert all(head in text for head in ("GeneratedFact", "CustomFact", "LargeFact"))
        assert text.count("(RootFact Root)") == 1
    else:
        assert "GeneratedFact" not in text and "CustomFact" not in text
        assert text.count("(RootFact Root)") == 2


def test_deduplication_preserves_distinct_facts_strings_rules_and_bad_tails():
    rule = "(= (f $x) $x)"
    text = '\n'.join(kb.unique_fact_blocks([
        '(Fact A "a b")\n' + rule,
        '(Fact ; comment\n A "a b")\n(Fact A "a  b")\n' + rule + '\n(broken',
    ]))
    assert text.count('"a b"') == 1
    assert '(Fact A "a  b")' in text
    assert text.count(rule) == 2
    assert text.endswith("(broken")


def test_all_stack_helpers_return_the_same_full_kb_on_petta(monkeypatch):
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    expected = kb.full_kb_paths()
    for choose in (runner.generic_stack, runner.patient_stack, runner.human_evidence_stack):
        assert choose([]) == expected
    assert runner.linage2_patient_kb() == expected
    assert runner.nhanes_patient_kb() == expected


def test_ui_and_api_use_the_full_kb_regardless_of_query_kind(monkeypatch):
    import api
    pytest.importorskip("gradio")
    import app
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    expected = kb.full_kb_paths()
    assert api._runtime_kb_paths() == app._runtime_kb_paths() == expected
    for query in ("(match &self $atom $atom)", "(human-evidence &self Metformin)",
                  "(predict-risk-patient &self Patient001)"):
        assert api._generic_kb(query) == app._generic_kb(query) == expected


@pytest.mark.petta
def test_real_petta_queries_the_complete_kb_in_one_space(monkeypatch):
    rt = PettaRuntime()
    if not rt.available():
        pytest.skip("PeTTa is not installed")
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    monkeypatch.setattr(runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(runner, "get_runtime", lambda: rt)

    # Independent oracle: exact unique Experiment row ids in EVERY application file.
    import re
    paths = kb.full_kb_paths()
    source = "\n".join(p.read_text() for p in paths)
    rows = set(re.findall(r"\(InstanceOf\s+(\S+)\s+Experiment\)", source))
    expressions = [
        "(let $rows (collapse (match &self (InstanceOf $r Experiment) $r)) (size-atom $rows))",
        "(infer &self CellularSenescence CoronaryHeartDisease)",
        '(match &self (, (Inheritance CellularSenescence HallmarkOfAging) '
        '(FullKBProbe $value)) $value)',
    ]
    if any(p.name == "genage_human_etl.metta" for p in paths):
        expressions.append('(match &self (GeneSymbol GHR $name) $name)')
    expressions.append("(patient-status &self Patient001 DNAmPAI1)")
    result = runner.run_query("\n".join(expressions), kb_files=[], extra_atoms="(FullKBProbe Joined)")
    assert result.ok, result.error
    by_expr = {i: [r.atom for r in result.results if r.expr_index == i] for i in range(len(expressions))}
    assert by_expr[0] == [str(len(rows))]
    assert any("(signed Pos" in a for a in by_expr[1])
    assert by_expr[2] == ["Joined"]
    if len(expressions) == 5:
        assert by_expr[3] == ['"GHR"']
    assert by_expr[len(expressions) - 1] == ["Elevated"]
    print({"files": len(paths), "source_forms": len(split_top_level_forms(source)), **rt.last_run})


@pytest.mark.petta
def test_raw_api_can_join_root_ontology_and_generated_data(monkeypatch):
    import asyncio
    import httpx
    import api
    import fastapi.routing
    import fastapi.dependencies.utils
    from core import executor

    if not PettaRuntime().available():
        pytest.skip("PeTTa is not installed")
    if not (config.ONTOLOGY_DIR / "build" / "genage_human_etl.metta").exists():
        pytest.skip("generated GenAge dataset is not present")
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    monkeypatch.setattr(runner, "PLN_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 0)
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(fastapi.routing, "run_in_threadpool", run_inline)
    monkeypatch.setattr(fastapi.dependencies.utils, "run_in_threadpool", run_inline)
    program = ('(linage-delta &self Patient001)\n'
               '(match &self (, (Inheritance CellularSenescence HallmarkOfAging) (GeneSymbol GHR $name)) $name)')

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            return await client.post("/metta/run", json={
                "metta_query": program, "extra_atoms": "(LinAgeDelta Patient001 4.0)",
            })

    response = asyncio.run(request())
    assert response.status_code == 200, response.text
    results = response.json()["pln_results"]
    assert len(results) == 2
    assert float(results[0]["atom"]) == 4.0
    assert results[1]["atom"] == '"GHR"'
