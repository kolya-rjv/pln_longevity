"""Parse .metta files into an OntologyRegistry and provide their raw text.

Parsing is best-effort: the goal is to extract a symbol inventory for
validation, not to implement a full MeTTa evaluator.

Supported constructs
--------------------
  (: Symbol TypeExpr)             -- type / function declaration
  (= (fn Constant) value)         -- constant definition (arg has no $)
  (= (fn $var ...) body)          -- rewrite rule

Multi-line expressions are joined before matching.
Inline comments (everything after an unquoted ';') are stripped.
"""
from __future__ import annotations

import re
from pathlib import Path

from ontology.registry import OntologyEntry, OntologyRegistry

# -- Regexes applied to single fully-joined S-expressions -------------------
# (: Symbol TypeExpression)
_TYPE_DECL = re.compile(r'^\(:\s+(\S+)\s+(.+?)\s*\)$', re.DOTALL)

# (= (fn-name <arg>) <value>) where <arg> has no $
_CONST_DEF = re.compile(r'^\(=\s+\(\S+\s+([^$\s)]+)\)\s+.+\)$', re.DOTALL)

# (= (fn-name $var ...) body) -- rewrite rule
_RULE_DEF = re.compile(r'^\(=\s+\((\S+)\s+\$', re.DOTALL)

# (Inheritance <symbol> <parent>) -- concrete instance declaration
_INHERITANCE = re.compile(r'^\(Inheritance\s+(\S+)\s+(\S+)\s*\)$')


def _strip_inline_comment(line: str) -> str:
    """Remove a trailing ; comment outside parentheses (simple heuristic)."""
    depth = 0
    for i, ch in enumerate(line):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ';' and depth == 0:
            return line[:i]
    return line


def _categorise(type_sig: str) -> str:
    return "function" if "->" in type_sig else "type"


def _iter_top_level_expressions(text: str):
    """Yield each top-level S-expression as a single stripped string."""
    buf: list[str] = []
    depth = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(';;'):
            continue
        line = _strip_inline_comment(line).strip()
        if not line:
            continue
        for ch in line:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
        buf.append(line)
        if depth <= 0 and buf:
            expr = ' '.join(buf).strip()
            if expr:
                yield expr
            buf = []
            depth = 0
    if buf:
        expr = ' '.join(buf).strip()
        if expr:
            yield expr


def parse_metta_file(path: Path) -> OntologyRegistry:
    """Parse a single .metta file and return a populated OntologyRegistry."""
    registry = OntologyRegistry()
    filename = path.name
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return registry

    for expr in _iter_top_level_expressions(text):
        # Type / function declarations
        m = _TYPE_DECL.match(expr)
        if m:
            symbol = m.group(1)
            type_sig = m.group(2).strip()
            registry.register(OntologyEntry(
                name=symbol,
                category=_categorise(type_sig),
                type_signature=type_sig,
                source_file=filename,
            ))
            continue

        # Constant definitions: (= (fn Constant) value)
        m2 = _CONST_DEF.match(expr)
        if m2:
            constant = m2.group(1)
            if not registry.get(constant):
                registry.register(OntologyEntry(
                    name=constant,
                    category="constant",
                    source_file=filename,
                ))
            continue

        # Rewrite rules: (= (fn $var ...) body)
        m3 = _RULE_DEF.match(expr)
        if m3:
            fn_name = m3.group(1)
            if not registry.get(fn_name):
                registry.register(OntologyEntry(
                    name=fn_name,
                    category="function",
                    source_file=filename,
                ))
            continue

        # Concrete instance declarations: (Inheritance Symbol Parent)
        # e.g. (Inheritance Ethanol Intervention)
        #      (Inheritance Drosophila_mojavensis Invertebrate)
        m4 = _INHERITANCE.match(expr)
        if m4:
            symbol, parent = m4.group(1), m4.group(2)
            # Register the symbol as a typed constant if not already known
            if not registry.get(symbol):
                registry.register(OntologyEntry(
                    name=symbol,
                    category="constant",
                    type_signature=parent,
                    source_file=filename,
                ))

    return registry


def read_raw(path: Path) -> str:
    """Return the raw text of a .metta file, or an empty string on error."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def load_ontology_dir(directory: Path) -> OntologyRegistry:
    """Load all .metta files found in *directory* into one merged registry."""
    merged = OntologyRegistry()
    if not directory.exists():
        return merged
    for metta_file in sorted(directory.glob("*.metta")):
        merged.merge(parse_metta_file(metta_file))
    return merged


def load_specific_files(file_paths: list[Path]) -> tuple[OntologyRegistry, dict[str, str]]:
    """Load specific .metta files.

    Returns
    -------
    registry : OntologyRegistry
        Merged parsed symbol registry (plus built-ins).
    raw_contents : dict[str, str]
        Filename -> raw file text, only for non-empty files.
    """
    from ontology.registry import BUILTIN_REGISTRY
    merged = OntologyRegistry()
    merged.merge(BUILTIN_REGISTRY)
    raw_contents: dict[str, str] = {}
    for path in file_paths:
        if path.exists():
            merged.merge(parse_metta_file(path))
            raw = read_raw(path)
            if raw.strip():
                raw_contents[path.name] = raw
    return merged, raw_contents
