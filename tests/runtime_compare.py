"""Compare one MeTTa answer with a recorded one, across the two runtimes.

Why this module exists
----------------------
Several tests hold answers recorded on hyperon, to the digit (tests/fixtures/*.json, string literals in the
tests themselves). PeTTa (PLN_RUNTIME=petta) computes the same answer with the same knowledge base, but
renders it differently in two ways that are not a change of meaning:

  * integer-valued numbers: PeTTa prints `3` where hyperon prints `3.0` (an arithmetic result that is whole);
  * the order of the members of a SET-LIKE collapse: hyperon often reverses the order a space was asserted in,
    PeTTa keeps assertion order, so `(SupportedBy (CRP DNAmGDF15 DNAmPAI1))` reads
    `(SupportedBy (DNAmPAI1 DNAmGDF15 CRP))`.

The rule of this module
-----------------------
  * Under hyperon (the default) `same_answer(actual, expected)` is `actual == expected`, string equality. Not
    one assertion that goes through it is looser than it was.
  * Under petta both strings are parsed into s-expressions and compared with exactly two allowances:
      (a) a number is compared as a number: `3` equals `3.0` (and `1.0e-05` equals `1e-05`); a string
          literal `"3"` stays a string and a symbol stays a symbol; two different numbers stay different
          (no tolerance: a one-ulp difference in a float is reported, never absorbed);
      (b) the members of a tuple are compared as a MULTISET only where `ORDER_FREE` says that argument
          of that head is a set (see below). Everything else, including the order of the answers a query
          returns and every ranked list, keeps its position.

`ORDER_FREE` is the allow-list of (head symbol, argument position) pairs whose argument is a set. An entry
is added only after reading the knowledge-base function that produces it and finding that nothing downstream
reads the position of an element. Each entry carries its justification here, and in the notes file of the
stage that introduced it (docs/petta_migration/). A tuple that is not listed is compared in order.

    SupportedBy, position 1   `(SupportedBy (m1 m2 ...))`: the observed markers a hypothesis explains,
        built by `ev-markers` (pln_abductive_diagnosis.metta) by consing over the `collapse` of
        `(explains-obs $space $h (superpose $obs))`, i.e. in the order the engine enumerates the observation
        set. Its own header says "Marker order within SupportedBy is collapse-order, not significant". The
        other aggregates over the same tuple are order-free in value: `ev-count` and `ev-maxs` exactly,
        `ev-orc` (noisy-OR) and `ev-mass` (a sum) up to the last bit, see "Sums" below.

    Targets, position 1       `(Targets (m1 m2 ...))` inside a `SuppRec`: the patient's elevated markers a
        supplement protectively reaches, built by `supp-targets` (pln_supplement_recommendation.metta) as
        `(collapse (let $m (superpose $obs) ...))`, again in the engine's enumeration order of the
        patient's observation set. It is provenance: no equation of the knowledge base and no line of Python
        reads it (grep: `Targets` occurs only where it is written and in comments), and the score is computed
        from `patient-relevance`, not from this tuple.

Sums
----
A sum over a set is as order-free as the set, but floating-point addition is not associative: the same
terms added in another order can differ in the last digit. This module does NOT absorb that: if a sum
over a reordered collapse differs from the recording in the last digits, the comparison fails and the
difference is a finding to report (it is not hidden behind a tolerance). `first_difference` names it.
It is not hypothetical: for a caller with ten markers PeTTa's `(stv 0.8 0.951)` is hyperon's
`(stv 0.8 0.9510000000000001)`. The recorded answers of the built-in patients are bit-exact on both runtimes
(32 of 32), so nothing in the suite hits it today.

What this module does not do
----------------------------
It does not turn a different RANKED order, a different first answer or a different number of answers into a
pass. `[a, b]` against `[b, a]` fails on both runtimes; so does a rank flip inside a `Hypothesis` list.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

import pytest

__all__ = [
    "ORDER_FREE", "Quoted", "runtime_name", "on_petta", "petta_only", "parse", "canonical", "numbers_as_floats",
    "same_answer", "same_answers", "contains", "startswith", "first_difference", "assert_same_answer",
    "assert_same_answers",
]

#: head symbol -> positions (0 is the head itself) of arguments that are SETS. See the module docstring.
ORDER_FREE: Mapping[str, tuple[int, ...]] = {
    "SupportedBy": (1,),
    "Targets": (1,),
}

_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[()]|[^\s()"]+')
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class Quoted(str):
    """A string literal, kept apart from a symbol (`"3"` is not `3` and not the symbol `3`)."""
    __slots__ = ()

    def __repr__(self) -> str:                  # pragma: no cover - cosmetic
        return "Quoted(" + str.__repr__(self) + ")"


# ── which runtime ─────────────────────────────────────────────────────────────

def runtime_name() -> str:
    """'hyperon' or 'petta': what PLN_RUNTIME selects here (`auto` resolved the way the app resolves it).
    The same rule as tests/runtime_harness.py, so the two helpers never disagree."""
    from core.metta_runtime import selected_runtime_name
    return selected_runtime_name()


def on_petta() -> bool:
    return runtime_name() == "petta"


#: `@petta_only` marks a test that asserts what the PeTTa runtime does and hyperon does not (for instance
#: "the full stack answers where hyperon aborts"). It is skipped under hyperon. `@pytest.mark.hyperon_only`
#: (registered in conftest.py) is its mirror.
petta_only = pytest.mark.skipif(not on_petta(), reason="asserts a PeTTa behaviour (PLN_RUNTIME=petta)")


# ── parsing ───────────────────────────────────────────────────────────────────

def parse(text: str) -> Any:
    """One answer string as a tree: a tuple for an expression, a float for a number, a `Quoted` for a string
    literal, a plain str for a symbol. Raises ValueError for an unbalanced string.

    Several top-level atoms (`a b`) come back as a tuple of them, marked with a leading None, so that
    `parse("a b") != parse("(a b)")`.
    """
    tokens = _TOKEN_RE.findall(text)
    out: list[Any] = []
    stack: list[list[Any]] = []
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                raise ValueError(f"unbalanced ')' in {text[:120]!r}")
            done = tuple(stack.pop())
            (stack[-1] if stack else out).append(done)
        else:
            (stack[-1] if stack else out).append(_leaf(tok))
    if stack:
        raise ValueError(f"unbalanced '(' in {text[:120]!r}")
    return out[0] if len(out) == 1 else (None,) + tuple(out)


def _leaf(tok: str) -> Any:
    if tok[0] == '"':
        return Quoted(tok)
    if _NUMBER_RE.fullmatch(tok):
        return float(tok)
    return tok


# ── canonical form ────────────────────────────────────────────────────────────

def canonical(node: Any, order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> Any:
    """`node` with the members of every order-free tuple put in a fixed order, so two answers that differ
    only by the permitted allowances are equal."""
    if not isinstance(node, tuple):
        return node
    children = [canonical(c, order_free) for c in node]
    head = node[0] if node and isinstance(node[0], str) and not isinstance(node[0], Quoted) else None
    if head in order_free:
        for position in order_free[head]:
            if position < len(children) and isinstance(children[position], tuple):
                children[position] = tuple(sorted(children[position], key=_sort_key))
    return tuple(children)


def _sort_key(node: Any) -> str:
    return repr(node)


# ── comparison ────────────────────────────────────────────────────────────────

def same_answer(actual: str, expected: str, *, order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> bool:
    """Is `actual` the recorded `expected`? Strict string equality on hyperon; see the module docstring."""
    if not on_petta():
        return actual == expected
    return actual == expected or canonical(parse(actual), order_free) == canonical(parse(expected), order_free)


def same_answers(actual: Sequence[str], expected: Sequence[str], *,
                 order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> bool:
    """The answer lists of a query: the same number of answers, in the same order, each `same_answer`."""
    if not on_petta():
        return list(actual) == list(expected)
    return len(actual) == len(expected) and all(
        same_answer(a, e, order_free=order_free) for a, e in zip(actual, expected))


def numbers_as_floats(text: str) -> str:
    """`text` with every number outside a string literal written the way Python writes the float: `2` and
    `2.0` both become `2.0`, `1.0e-05` becomes `1e-05`. Nothing else changes (not a space, not the order).
    The tool for a fragment that is not a whole expression (a prefix, a piece with a trailing space)."""
    def _one(m: "re.Match[str]") -> str:
        tok = m.group()
        return repr(float(tok)) if tok[0] != '"' and _NUMBER_RE.fullmatch(tok) else tok
    return _TOKEN_RE.sub(_one, text)


def contains(answer: str, fragment: str, *, order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> bool:
    """`fragment in answer`: plain substring on hyperon. Under petta the same, with numbers compared as
    numbers (`numbers_as_floats` on both), or, when `fragment` is a whole expression, found as a subtree of
    the answer with the order-free tuples compared as multisets."""
    if not on_petta():
        return fragment in answer
    if numbers_as_floats(fragment) in numbers_as_floats(answer):
        return True
    wanted = fragment.strip()
    if not wanted.startswith("("):
        return False
    try:
        target = canonical(parse(wanted), order_free)
        tree = canonical(parse(answer), order_free)
    except ValueError:
        return False
    return isinstance(target, tuple) and _has_subtree(tree, target)


def _has_subtree(tree: Any, target: Any) -> bool:
    if tree == target:
        return True
    return isinstance(tree, tuple) and any(_has_subtree(child, target) for child in tree)


def startswith(answer: str, prefix: str) -> bool:
    """`answer.startswith(prefix)`: plain on hyperon; under petta with numbers compared as numbers."""
    if not on_petta():
        return answer.startswith(prefix)
    return numbers_as_floats(answer).startswith(numbers_as_floats(prefix))


def first_difference(actual: str, expected: str, *,
                     order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> Optional[str]:
    """None when `same_answer`, else the path and the two subtrees where the canonical forms first differ."""
    a, e = canonical(parse(actual), order_free), canonical(parse(expected), order_free)
    return None if a == e else _diff(a, e, "answer")


def _diff(a: Any, e: Any, path: str) -> str:
    if isinstance(a, tuple) and isinstance(e, tuple):
        if len(a) != len(e):
            return f"{path}: {len(a)} members vs {len(e)} members: {_show(a)} vs {_show(e)}"
        for i, (x, y) in enumerate(zip(a, e)):
            if x != y:
                head = a[0] if a and isinstance(a[0], str) else "?"
                return _diff(x, y, f"{path}/{head}[{i}]")
    return f"{path}: {_show(a)} vs {_show(e)}"


def _show(node: Any) -> str:
    if isinstance(node, tuple):
        return "(" + " ".join(_show(c) for c in node if c is not None) + ")"
    if isinstance(node, float):
        return repr(node)
    return str(node)


def assert_same_answer(actual: str, expected: str, *, what: str = "",
                       order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> None:
    """`assert same_answer(...)`, with the first difference in the message under petta."""
    if same_answer(actual, expected, order_free=order_free):
        return
    detail = ""
    if on_petta():
        try:
            detail = "\nfirst difference: " + str(first_difference(actual, expected, order_free=order_free))
        except ValueError as exc:
            detail = f"\n(not parseable: {exc})"
    raise AssertionError(f"{what + ': ' if what else ''}the answer is not the recorded one{detail}\n"
                         f"actual:   {actual[:1500]}\nrecorded: {expected[:1500]}")


def assert_same_answers(actual: Sequence[str], expected: Sequence[str], *, what: str = "",
                        order_free: Mapping[str, tuple[int, ...]] = ORDER_FREE) -> None:
    """The list form of `assert_same_answer`."""
    if same_answers(actual, expected, order_free=order_free):
        return
    if len(actual) != len(expected):
        raise AssertionError(f"{what + ': ' if what else ''}{len(actual)} answers where {len(expected)} were recorded"
                             f"\nactual:   {list(actual)!r:.1500}\nrecorded: {list(expected)!r:.1500}")
    for a, e in zip(actual, expected):
        assert_same_answer(a, e, what=what, order_free=order_free)
