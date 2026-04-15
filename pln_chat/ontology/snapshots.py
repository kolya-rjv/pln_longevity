"""Serialise loaded .metta files into a text block for LLM injection.

Primary context: raw .metta file content (verbatim), grouped by file.
The LLM sees real MeTTa syntax, which is the most accurate source of truth.

Secondary context: a compact parsed symbol list for quick reference.
"""
from __future__ import annotations

from ontology.registry import OntologyRegistry

_CATEGORY_ORDER = ["type", "function", "predicate", "constant", "rule"]
_CATEGORY_LABELS = {
    "type":      "Types",
    "function":  "Functions / Rules",
    "predicate": "Predicates",
    "constant":  "Constants",
    "rule":      "Rules",
}


def registry_to_symbol_list(registry: OntologyRegistry) -> str:
    """Compact parsed symbol list (secondary reference for the LLM)."""
    if registry.is_empty():
        return "_No symbols parsed._\n"
    lines: list[str] = []
    for cat in _CATEGORY_ORDER:
        symbols = registry.symbols_by_category(cat)
        if not symbols:
            continue
        lines.append(f"\n### {_CATEGORY_LABELS[cat]}")
        for sym in sorted(symbols):
            entry = registry.get(sym)
            row = f"  {sym}"
            if entry and entry.type_signature:
                row += f" : {entry.type_signature}"
            if entry and entry.source_file and entry.source_file != "builtin":
                row += f"  [{entry.source_file}]"
            lines.append(row)
    return "\n".join(lines)


def build_ontology_context(
    raw_contents: dict[str, str],
    registry: OntologyRegistry,
) -> str:
    """Build the full ontology context block to inject into the system prompt.

    Parameters
    ----------
    raw_contents:
        Mapping of filename -> raw .metta file text (non-empty files only).
    registry:
        Parsed symbol registry (used for the secondary symbol list).
    """
    sections: list[str] = ["# Loaded MeTTa Knowledge Base\n"]

    if raw_contents:
        sections.append("## Raw MeTTa Definitions\n")
        sections.append(
            "_The following is the verbatim content of the loaded .metta files. "
            "Use these definitions exactly when constructing queries._\n"
        )
        for filename, content in raw_contents.items():
            sections.append(f"### {filename}\n```metta\n{content.strip()}\n```\n")
    else:
        sections.append(
            "## Raw MeTTa Definitions\n"
            "_No .metta files with content are currently loaded. "
            "New ontology and rules files will appear here automatically once added._\n"
        )

    # Always show the parsed symbol list so the LLM has a quick index
    sections.append("## Parsed Symbol Index\n" + registry_to_symbol_list(registry))

    return "\n".join(sections)
