"""The answer comparator (tests/runtime_compare.py) is exactly as loose as it says and no looser.

Its whole value is what it refuses: under hyperon it is string equality; under PeTTa it forgives a whole number
printed as `3` for `3.0` and the order of a tuple that `ORDER_FREE` names as a set, and nothing else. These tests
pin both halves, on either runtime (the runtime is switched by patching `on_petta`; no engine runs here).

    pytest tests/test_runtime_compare.py -q
"""
from __future__ import annotations

import pytest

import runtime_compare as rc

HYP = ("((Hypothesis CellularSenescence (stv 0.595 0.6253054502923161) 3.0 0.45928900265625006 "
       "(SupportedBy (CRP DNAmGDF15 DNAmPAI1))) (Hypothesis ChronicInflammation (stv 0.8 0.8775) 2.0 "
       "0.9425000000000001 (SupportedBy (CRP DNAmGDF15))))")
HYP_PETTA = ("((Hypothesis CellularSenescence (stv 0.595 0.6253054502923161) 3 0.45928900265625006 "
             "(SupportedBy (DNAmPAI1 DNAmGDF15 CRP))) (Hypothesis ChronicInflammation (stv 0.8 0.8775) 2 "
             "0.9425000000000001 (SupportedBy (DNAmGDF15 CRP))))")


@pytest.fixture
def petta(monkeypatch):
    monkeypatch.setattr(rc, "on_petta", lambda: True)


@pytest.fixture
def hyperon(monkeypatch):
    monkeypatch.setattr(rc, "on_petta", lambda: False)


# ═══════════════════════════ hyperon: string equality ═════════════════════════

def test_under_hyperon_the_comparison_is_string_equality(hyperon):
    assert rc.same_answer(HYP, HYP)
    assert not rc.same_answer(HYP_PETTA, HYP), "the PeTTa rendering is not the recorded hyperon one"
    assert not rc.same_answer("(a 3)", "(a 3.0)")
    assert not rc.same_answer("(SupportedBy (a b))", "(SupportedBy (b a))")
    assert rc.same_answers(["(a)", "(b)"], ["(a)", "(b)"]) and not rc.same_answers(["(a)", "(b)"], ["(b)", "(a)"])


def test_under_hyperon_the_fragment_helpers_are_the_plain_string_operations(hyperon):
    assert rc.contains("(x 2.0 y)", " 2.0 ") and not rc.contains("(x 2 y)", " 2.0 ")
    assert not rc.contains("(SupportedBy (a b))", "(SupportedBy (b a))")
    assert rc.startswith("((H (stv 0.5 0.6) 1.0", "((H (stv 0.5 0.6) 1.0")
    assert not rc.startswith("((H (stv 0.5 0.6) 1 0.3", "((H (stv 0.5 0.6) 1.0")


def test_assert_same_answer_raises_under_hyperon_where_petta_would_forgive(hyperon):
    with pytest.raises(AssertionError):
        rc.assert_same_answer("(a 3)", "(a 3.0)")
    with pytest.raises(AssertionError):
        rc.assert_same_answers(["(a)"], ["(a)", "(b)"])


# ═══════════════════════════ petta: the two allowances ════════════════════════

def test_the_recorded_hyperon_answer_and_the_petta_rendering_are_the_same_answer(petta):
    assert rc.same_answer(HYP_PETTA, HYP) and rc.same_answer(HYP, HYP_PETTA)
    assert rc.first_difference(HYP_PETTA, HYP) is None
    rc.assert_same_answer(HYP_PETTA, HYP)


@pytest.mark.parametrize("a, b", [("(n 3)", "(n 3.0)"), ("(n 1.0e-05)", "(n 1e-05)"), ("(n -2)", "(n -2.0)"),
                                  ("(n 0.1)", "(n 0.10000000000000001)"), ("(n 58)", "(n 58.0)")])
def test_a_number_is_a_number_whatever_its_rendering(petta, a, b):
    assert rc.same_answer(a, b)


@pytest.mark.parametrize("a, b", [
    ("(n 0.3)", "(n 0.30000000000000004)"),            # one ulp is a difference, not a rounding to forgive
    ("(n 3)", "(n 3.0000001)"),
    ('(n "3")', "(n 3)"),                              # a string is not a number
    ("(n abc3)", "(n 3)"),                             # nor is a symbol
    ("(n 3)", "(n 4)"),
    ("(n true)", "(n True)"),                          # the adapter's job (normalise_result), not this module's
])
def test_a_different_value_stays_different(petta, a, b):
    assert not rc.same_answer(a, b)


def test_the_members_of_a_named_set_may_come_in_any_order(petta):
    assert rc.same_answer("(SupportedBy (a b c))", "(SupportedBy (c b a))")
    assert rc.same_answer("(SuppRec Fisetin (Targets (DNAmPAI1 DNAmGDF15 CRP)))", "(SuppRec Fisetin (Targets (CRP DNAmGDF15 DNAmPAI1)))")


def test_a_set_is_a_multiset_not_a_set(petta):
    assert not rc.same_answer("(SupportedBy (a a b))", "(SupportedBy (a b b))")
    assert not rc.same_answer("(SupportedBy (a b))", "(SupportedBy (a b b))")
    assert not rc.same_answer("(SupportedBy (a b))", "(SupportedBy (a c))")


def test_a_tuple_that_is_not_named_keeps_its_order(petta):
    assert not rc.same_answer("(Tier1HighConfidence (a b))", "(Tier1HighConfidence (b a))")
    assert not rc.same_answer("(Foo (a b))", "(Foo (b a))")
    assert not rc.same_answer("(SupportedBy x (a b))", "(SupportedBy x (b a))"), "only position 1 is the set"


def test_a_ranked_list_keeps_its_order_even_when_every_member_is_the_same(petta):
    swapped = HYP_PETTA.replace("CellularSenescence", "@@").replace("ChronicInflammation", "CellularSenescence") \
                       .replace("@@", "ChronicInflammation")
    assert swapped != HYP_PETTA
    assert not rc.same_answer(swapped, HYP)
    first_wins = "((Hypothesis A (stv 0.5 0.6) 1 0.3 (SupportedBy (x))) (Hypothesis B (stv 0.5 0.6) 1 0.3 (SupportedBy (y))))"
    second_wins = "((Hypothesis B (stv 0.5 0.6) 1 0.3 (SupportedBy (y))) (Hypothesis A (stv 0.5 0.6) 1 0.3 (SupportedBy (x))))"
    assert not rc.same_answer(first_wins, second_wins)


def test_the_answers_of_a_query_keep_their_number_and_their_order(petta):
    assert rc.same_answers(["(a 1)", "(b 2)"], ["(a 1.0)", "(b 2.0)"])
    assert not rc.same_answers(["(a)", "(b)"], ["(b)", "(a)"])
    assert not rc.same_answers(["(a)"], ["(a)", "(b)"])
    with pytest.raises(AssertionError, match="1 answers where 2 were recorded"):
        rc.assert_same_answers(["(a)"], ["(a)", "(b)"])


def test_a_difference_is_reported_where_it_is(petta):
    where = rc.first_difference("((Hypothesis A (stv 0.5 0.6) 1 0.3 (SupportedBy (x))))",
                                "((Hypothesis A (stv 0.5 0.61) 1 0.3 (SupportedBy (x))))")
    assert where is not None and "0.6" in where and "0.61" in where and "stv" in where
    with pytest.raises(AssertionError, match="first difference"):
        rc.assert_same_answer("(a 1)", "(a 2)")


def test_fragments_are_compared_as_numbers_and_as_subtrees(petta):
    answer = HYP_PETTA
    assert rc.contains(answer, "(Hypothesis ChronicInflammation (stv 0.8 0.8775) 2.0 ")
    assert rc.contains(answer, " 3.0 ")
    assert rc.contains(answer, "(SupportedBy (CRP DNAmGDF15))") and rc.contains(answer, "(SupportedBy (DNAmGDF15 CRP))")
    assert not rc.contains(answer, "(SupportedBy (CRP))")
    assert not rc.contains(answer, " 4.0 ")
    assert rc.startswith("((Hypothesis InsulinResistance (stv 0.5 0.6) 1 0.3", "((Hypothesis InsulinResistance (stv 0.5 0.6) 1.0 0.3")
    assert not rc.startswith("((Hypothesis InsulinResistance (stv 0.5 0.6) 2 0.3", "((Hypothesis InsulinResistance (stv 0.5 0.6) 1.0 0.3")


def test_numbers_as_floats_touches_numbers_outside_strings_only():
    assert rc.numbers_as_floats('(a 2 "2" b2 2.50 1.0e-05 -3)') == '(a 2.0 "2" b2 2.5 1e-05 -3.0)'


# ═══════════════════════════ the list is documented ═══════════════════════════

def test_every_order_free_entry_is_justified_in_the_module_documentation():
    for head, positions in rc.ORDER_FREE.items():
        assert positions and all(isinstance(p, int) and p >= 1 for p in positions), (head, positions)
        assert f"{head}, position" in rc.__doc__, f"{head} is allowed to reorder but is not justified in runtime_compare.py"


def test_parse_keeps_a_symbol_a_string_and_a_number_apart():
    tree = rc.parse('(f 3 "3" three 3.5)')
    assert tree == ("f", 3.0, rc.Quoted('"3"'), "three", 3.5)
    assert isinstance(tree[2], rc.Quoted) and not isinstance(tree[3], rc.Quoted)
    with pytest.raises(ValueError):
        rc.parse("(a (b)")
    with pytest.raises(ValueError):
        rc.parse("a) b")
