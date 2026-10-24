"""The runtime-neutral MeTTa harness (tests/runtime_harness.py) behaves the same on both runtimes.

The fifteen modules that load `.metta` files and call `.run("!(...)")` all go through `new_kb`; if it
disagreed with `hyperon.MeTTa` about the shape of an answer, every one of them would be wrong in the same
way. These tests pin the shape, on whichever runtime PLN_RUNTIME selects (hyperon by default).
"""
from __future__ import annotations

import pytest

from runtime_harness import (
    AtomText, Kb, hyperon_only, new_kb, on_hyperon, on_petta, require_runtime, runtime_name,
)

require_runtime()

FACTS = "(Pair a b)\n(Pair c d)\n(= (twice $x) (* 2 $x))"


@pytest.fixture(scope="module")
def kb() -> "Kb":
    with new_kb(extra_text=FACTS) as k:
        yield k


def test_the_harness_runs_on_the_selected_runtime(kb):
    assert kb.runtime == runtime_name() in ("hyperon", "petta")
    assert on_petta() != on_hyperon()


def test_one_group_per_bang_form_and_str_gives_the_atom(kb):
    groups = kb.run("!(twice 21) !(match &self (Pair a $y) $y) !(match &self (Pair z $y) $y)")
    assert [[str(x) for x in g] for g in groups] == [["42"], ["b"], []]


def test_a_whole_result_prints_as_hyperon_prints_it(kb):
    """`str(result)` and an f-string over a result read `[[(Pair a b)]]`, not `[['(Pair a b)']]`."""
    out = kb.run("!(match &self (Pair a $y) (Pair a $y))")
    assert str(out) == "[[(Pair a b)]]"
    assert f"{out}" == "[[(Pair a b)]]"


def test_text_without_a_bang_adds_facts_and_answers_nothing():
    with new_kb(extra_text=FACTS) as k:
        assert k.run("(Pair e f)") == []
        assert [str(x) for g in k.run("!(match &self (Pair e $y) $y)") for x in g] == ["f"]


def test_files_then_extra_text_are_one_space():
    with new_kb(["system_types.metta"], "(Probe one)") as k:
        assert [str(x) for g in k.run("!(match &self (Probe $x) $x)") for x in g] == ["one"]
        # a declaration the file makes is there too
        assert [str(x) for g in k.run("!(match &self (: Significant $t) $t)") for x in g] == ["SignificanceLabel"]


def test_separate_runs_load_the_same_space_as_one_joined_run():
    one = new_kb(["system_types.metta"], "(Probe one)")
    sep = new_kb(["system_types.metta"], "(Probe one)", separate=True)
    try:
        query = "!(match &self (: $x $t) $x)"
        assert sorted(str(x) for g in one.run(query) for x in g) == \
            sorted(str(x) for g in sep.run(query) for x in g)
    finally:
        one.close()
        sep.close()


def test_an_empty_knowledge_base_answers_pure_expressions():
    with new_kb() as k:
        assert [[str(x) for x in g] for g in k.run("!(+ 1 2)")] == [["3"]]


@hyperon_only
def test_a_hyperon_knowledge_base_is_a_real_metta_and_answers_with_real_atoms(kb):
    from hyperon import MeTTa
    assert isinstance(kb.metta, MeTTa)
    atom = kb.run("!(match &self (Pair a $y) $y)")[0][0]
    assert not isinstance(atom, str)
    assert str(atom) == "b"


def test_a_petta_answer_is_a_str_that_prints_as_itself():
    if not on_petta():
        pytest.skip("PeTTa answers only")
    with new_kb(extra_text=FACTS) as k:
        atom = k.run("!(match &self (Pair a $y) $y)")[0][0]
        assert isinstance(atom, AtomText) and str(atom) == "b" and repr(atom) == "b"
        with pytest.raises(AttributeError):
            k.metta
