from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OntologyEntry:
    name: str
    category: str          # "type" | "function" | "predicate" | "constant" | "rule"
    type_signature: Optional[str] = None
    description: Optional[str] = None
    source_file: Optional[str] = None


@dataclass
class OntologyRegistry:
    entries: Dict[str, OntologyEntry] = field(default_factory=dict)

    def register(self, entry: OntologyEntry) -> None:
        self.entries[entry.name] = entry

    def get(self, name: str) -> Optional[OntologyEntry]:
        return self.entries.get(name)

    def symbols_by_category(self, category: str) -> List[str]:
        return [n for n, e in self.entries.items() if e.category == category]

    def all_symbols(self) -> List[str]:
        return list(self.entries.keys())

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def merge(self, other: "OntologyRegistry") -> None:
        self.entries.update(other.entries)


# ---------------------------------------------------------------------------
# PLN / MeTTa built-ins -- always valid regardless of which .metta files are
# loaded. The real ontology lives entirely in .metta files.
# ---------------------------------------------------------------------------
_BUILTIN_ENTRIES: List[OntologyEntry] = [
    OntologyEntry("stv",         "function", "(-> Number Number STV)",          "Simple Truth Value constructor", "builtin"),
    OntologyEntry("Inheritance", "function", "(-> Atom Atom Bool)",             "PLN Inheritance link",           "builtin"),
    OntologyEntry("Similarity",  "function", "(-> Atom Atom Bool)",             "PLN Similarity link",            "builtin"),
    OntologyEntry("Evaluation",  "function", "(-> Predicate ListLink Bool)",    "PLN Evaluation link",            "builtin"),
    OntologyEntry("Member",      "function", "(-> Atom Atom Bool)",             "PLN Member link",                "builtin"),
    OntologyEntry("match",       "function", "(-> Grnd Pattern Template Atom)", "MeTTa match builtin",            "builtin"),
]

BUILTIN_REGISTRY = OntologyRegistry()
for _e in _BUILTIN_ENTRIES:
    BUILTIN_REGISTRY.register(_e)
