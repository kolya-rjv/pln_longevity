"""Lightweight syntax and symbol validation for generated MeTTa queries.

This is intentionally simple: it checks balanced parentheses and, when the
registry is populated, flags symbols not present in the ontology.
No full MeTTa parser is required — that would need the Hyperon runtime.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

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

_SYMBOL_RE  = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
_VARIABLE_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9_\-]*")


@dataclass
class ValidationResult:
    valid: bool
    issues: List[str] = field(default_factory=list)


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


def validate(metta_query: str, registry: OntologyRegistry) -> ValidationResult:
    """Check a MeTTa query for syntax and (when possible) symbol issues."""
    if not metta_query.strip():
        return ValidationResult(valid=True)

    issues: list[str] = []

    if not _balanced_parens(metta_query):
        issues.append("Unbalanced parentheses in MeTTa query.")

    # Symbol resolution only when registry has content
    if not registry.is_empty():
        variable_names = {v.lstrip("$") for v in _VARIABLE_RE.findall(metta_query)}
        unknown: list[str] = []
        for tok in set(_SYMBOL_RE.findall(metta_query)):
            if tok in _BUILTINS:
                continue
            if tok in variable_names:
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

    return ValidationResult(valid=len(issues) == 0, issues=issues)
