"""The complete application KB for PeTTa; Hyperon keeps its scoped stacks.

Application sources live in the ontology root, build/ (generated data), and the
custom ontology directory. Tests, examples and vendored interpreter libraries are
not application data. pln_chainer.metta is an alternative implementation installed
by core.pln_chainer only when PLN_CHAINER=pln, not an additional set of facts.
"""
from pathlib import Path
from typing import Iterable

import config
from core.metta_runtime import petta_tokens, selected_runtime_name, split_top_level_forms


def uses_full_kb() -> bool:
    return selected_runtime_name() == "petta"


def full_kb_paths(additional: Iterable[Path] = ()) -> list[Path]:
    """Discover every application .metta file, without query or byte-size filters."""
    roots = (config.ONTOLOGY_DIR, config.ONTOLOGY_DIR / "build", config.CUSTOM_ONTOLOGY_DIR)
    paths = [p for root in roots for p in sorted(root.glob("*.metta"))
             if p.name != "pln_chainer.metta"]
    # Explicit extra sources are supported by the ETL tools and callers too.
    return list(dict.fromkeys(p.resolve() for p in [*paths, *additional]))


def unique_fact_blocks(blocks: list[str]) -> list[str]:
    """Avoid counting sample/generated/injected copies of the same fact twice.

    Preserve rule clauses, executable forms, quoted string values and unparsed tails.
    Only identical facts/declarations are deduplicated, by their token sequence.
    Different values for the same relation remain distinct facts.
    """
    seen: set[str] = set()
    out = []
    for text in blocks:
        pieces = []
        last = 0
        for start, end in split_top_level_forms(text):
            form = text[start:end]
            tokens = petta_tokens(form)
            if form.startswith("!") or len(tokens) < 2 or tokens[1][1] == "=":
                continue
            key = " ".join(token for _, token in tokens)
            if key in seen:
                pieces.append(text[last:start])
                last = end
            else:
                seen.add(key)
        pieces.append(text[last:])
        out.append("".join(pieces))
    return out
