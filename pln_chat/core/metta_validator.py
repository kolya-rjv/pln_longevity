"""Syntax, symbol and grounding validation for generated MeTTa queries.

Three checks, no MeTTa evaluation (which would need the Hyperon runtime):

1. balanced parentheses;
2. every symbol is known — measured against the ACTUAL ground atoms when a
   `RuntimeInventory` is supplied, not just the type declarations the registry
   harvests;
3. every predicate the query calls holds facts. A query over a declared-but-
   empty predicate is well-formed and returns nothing, which is the single most
   confusing failure mode the 2026-09-18 API evaluation reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ontology.inventory import RuntimeInventory
from ontology.registry import OntologyRegistry

# ── Always-valid built-in MeTTa / PLN symbols ─────────────────────────────────
_BUILTINS: set[str] = {
    # Control flow
    "match", "let", "if", "empty", "case",
    # Logic
    "not", "and", "or",
    # PLN links
    "stv", "Inheritance", "Similarity", "Evaluation", "Member",
    "ImplicationLink", "AndLink", "OrLink", "NotLink",
    # Arithmetic / comparison
    ">", "<", ">=", "<=", "=", "!=", "+", "-", "*", "/",
    "pair", "fst", "snd",
    # Atoms / types
    "&self",
    # "&self" is written as the token `&self` in MeTTa but the symbol regex
    # [A-Za-z][A-Za-z0-9_-]* strips the leading `&` and captures only `self`.
    # Add both spellings so the validator never flags &self as unknown.
    "self",
    "Number", "Bool", "String", "Atom",
    "True", "False",
}

_SYMBOL_RE  = re.compile(r"[A-Za-z][A-Za-z0-9_\->]*")
_VARIABLE_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9_\-]*")

# Numbers, including scientific notation. The symbol regex above happily matches
# the `e-75` inside `2.0e-75`, so a query carrying a real p-value used to be
# rejected for two "unknown symbols" that are not symbols at all. Blank numeric
# literals out before tokenising.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# The head of each top-level application in the query — i.e. the predicates and
# functions the query actually calls.
_HEAD_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9_\-]*)")


@dataclass
class ValidationResult:
    valid: bool
    issues: List[str] = field(default_factory=list)
    #: Non-fatal observations. A query can be perfectly well-formed and still be
    #: guaranteed to return nothing — that is the "valid but empty" failure the
    #: 2026-09-18 evaluation hit 6 times in 36 questions.
    warnings: List[str] = field(default_factory=list)
    #: Predicates the query uses that are DECLARED in the ontology but hold no
    #: ground facts in the runtime KB.
    ungrounded_predicates: List[str] = field(default_factory=list)


def _balanced_parens(expr: str) -> bool:
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate(
    metta_query: str,
    registry: OntologyRegistry,
    inventory: Optional[RuntimeInventory] = None,
) -> ValidationResult:
    """Check a MeTTa query for syntax, symbol and GROUNDING issues.

    `registry` is the historical symbol source: it harvests type declarations,
    rule heads, `Inheritance` and `InstanceOf`, and nothing else. That leaves it
    wrong in both directions, which is why `inventory` exists:

    * it does not know the arguments of ordinary ground facts, so real symbols
      (`MTORC1`, `AMPK`, `Mouse`, `Human`) were reported as unknown and
      `/metta/run` answered 422 to queries the runtime would have served;
    * it cannot tell a declared predicate from a populated one, so a query over
      `TargetsHallmark` — declared in logical_predicates.metta, zero facts —
      validated cleanly and returned nothing.

    When an inventory is supplied, an unknown symbol means unknown to the ACTUAL
    atoms, and using a declared-but-empty predicate becomes an explicit warning
    instead of a silent empty result.
    """
    if not metta_query.strip():
        return ValidationResult(valid=True)

    issues: list[str] = []
    warnings: list[str] = []
    ungrounded: list[str] = []

    if not _balanced_parens(metta_query):
        issues.append("Unbalanced parentheses in MeTTa query.")

    # Blank out numeric literals first so `2.0e-75` cannot contribute `e-75`.
    scrubbed = _NUMBER_RE.sub(" ", metta_query)

    if inventory is not None or not registry.is_empty():
        variable_names = {v.lstrip("$") for v in _VARIABLE_RE.findall(metta_query)}
        unknown: list[str] = []
        for tok in sorted(set(_SYMBOL_RE.findall(scrubbed))):
            if tok in _BUILTINS or tok in variable_names:
                continue
            if inventory is not None and inventory.knows_symbol(tok):
                continue
            if registry.get(tok) is not None:
                continue
            try:
                float(tok)
                continue
            except ValueError:
                pass
            unknown.append(tok)
        if unknown:
            issues.append(
                f"Symbol(s) not found in loaded ontology: {', '.join(sorted(unknown))}"
            )

    if inventory is not None:
        for head in sorted(set(_HEAD_RE.findall(metta_query))):
            if head in _BUILTINS:
                continue
            # `declared_only` is checked FIRST: a declared predicate is also in
            # `functions` (it carries a `(-> …)` signature), and skipping on
            # that would hide exactly the case this check exists for.
            if head in inventory.declared_only:
                ungrounded.append(head)
        if ungrounded:
            warnings.append(
                "Declared but EMPTY in the runtime knowledge base — this query "
                "is well-formed and will return nothing: "
                + ", ".join(ungrounded)
                + ". See GET /kb/schema for the predicates that hold facts."
            )

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        ungrounded_predicates=ungrounded,
    )
