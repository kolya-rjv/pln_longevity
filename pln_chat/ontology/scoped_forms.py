"""Tell a caller when a query is well-formed, loaded, and structurally empty.

The knowledge base holds RULES whose DATA is deliberately not in the generic
runtime space. `cellage_calibration.metta` defines `cellage-effect`,
`cellage-gene-effects` and `genes-affecting-senescence`; the CellAge rows they
match live in `build/cellage_genes.metta`, which is excluded because a
variable-slot match over ~900 rows aborts hyperon 0.2.10. `drugage_entries.metta`
is the same story for DrugAge.

So `POST /metta/run {"metta_query": "!(genes-affecting-senescence &self
Increases)"}` validated, executed, and returned `pln_status: "empty"`,
`pln_results: []`, `ungrounded_predicates: []`, `validation_warnings: []`.
API.md's own rule for reading a response — "an empty `pln_results` next to a
NON-empty `ungrounded_predicates` means the KB cannot express this relation, not
no" — tells an agent to read that as **no genes increase senescence**. The data
is right there: `GET /genes/TP53?infer=true` returns three
`(Effect Gene_TP53 CellularSenescence Pos …)` links.

The check is derived, not listed. For every `(= (fn …) …)` in the runtime KB we
read the predicate heads its body matches, and a function whose every matched
predicate has ZERO facts in the runtime inventory is data-less. That way a new
scoped layer is covered the day it is added, and a layer that gains data stops
warning without anyone remembering to edit a list.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from ontology.inventory import RuntimeInventory, iter_top_level

#: `(= (fn-name $a $b) …)` — a function definition and everything after it.
_DEF_RE = re.compile(r"^\(=\s*\((?P<name>[A-Za-z][A-Za-z0-9_*!?\-]*)\s")
#: `(match $space (Predicate …` and `(match $space (, (Predicate …` — the
#: predicate heads a body reads. `,` is MeTTa's conjunction, not a predicate.
_MATCH_RE = re.compile(r"\(match\s+\$\w+\s+\((?:,\s*\()?(?P<head>[A-Za-z][A-Za-z0-9_\-]*)")
#: Nested conjuncts, `(, (A …) (B …))`, after the first.
_CONJUNCT_RE = re.compile(r"\(,\s*\((?P<head>[A-Za-z][A-Za-z0-9_\-]*)")

#: Where the data for a data-less form IS reachable. Only used for the hint at
#: the end of the warning — the DETECTION above is derived, so a form missing
#: from this map still warns, just without a suggested route.
_ROUTES: dict[str, str] = {
    "cellage-effect": "GET /genes/{symbol}?infer=true",
    "cellage-gene-effects": "GET /genes/{symbol}?infer=true",
    "genes-affecting-senescence": "GET /genes?senescence_effect=Increases",
    "genes-affecting-senescence-in": "GET /genes with the context filters",
    "drugage-effect": "POST /drugage/rank or GET /drugage/top",
    "rank-drugage-lifespan": "POST /drugage/rank",
}


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";;")[0] for line in text.splitlines())


def _matched_predicates(body: str) -> set[str]:
    heads = {m.group("head") for m in _MATCH_RE.finditer(body)}
    heads |= {m.group("head") for m in _CONJUNCT_RE.finditer(body)}
    return {h for h in heads if h not in {"let", "let*", "if", "case"}}


@lru_cache(maxsize=8)
def _definitions(paths: tuple[Path, ...]) -> dict[str, set[str]]:
    """function name -> the predicate heads its body matches."""
    out: dict[str, set[str]] = {}
    for path in paths:
        try:
            text = _strip_comments(Path(path).read_text(encoding="utf-8"))
        except OSError:
            continue
        for expr in iter_top_level(text):
            match = _DEF_RE.match(expr)
            if not match:
                continue
            reads = _matched_predicates(expr)
            if reads:
                out.setdefault(match.group("name"), set()).update(reads)
    return out


#: Predicates whose facts exist only per request (a caller's patient, the shared space of one query): the runtime
#: KB never holds a row for them, which says nothing about a rule that reads them.
_PER_REQUEST_FACTS = frozenset({"PatientCondition", "CurrentMedication"})


def dataless_forms(
    kb_paths: Iterable[Path], inventory: RuntimeInventory
) -> dict[str, tuple[str, ...]]:
    """Functions in the runtime KB whose every matched predicate holds no facts."""
    populated = {name for name, count in inventory.predicates.items() if count}
    # `declared_only` is the inventory's own term for "the ontology declares
    # this predicate and the runtime space holds not one fact for it" — exactly
    # the state that makes a rule over it structurally silent.
    empty = set(inventory.declared_only)
    out: dict[str, tuple[str, ...]] = {}
    for name, reads in _definitions(tuple(kb_paths)).items():
        # Only predicates the INVENTORY classifies count as evidence either
        # way; a body that matches a helper the scan did not classify is left
        # alone rather than guessed at.
        known = {r for r in reads if (r in populated or r in empty) and r not in _PER_REQUEST_FACTS}
        if known and not (known & populated):
            out[name] = tuple(sorted(known))
    return out


def scoped_form_warnings(
    metta_query: str,
    kb_paths: Iterable[Path],
    inventory: RuntimeInventory,
) -> list[str]:
    """One warning per data-less form the query names.

    Deliberately about the FORM, not the result: the warning is attached
    whether or not the query happened to return something, because the thing
    worth saying is "this space holds no rows for that rule", and a caller who
    gets it alongside a non-empty answer learns something true either way.
    """
    dataless = dataless_forms(kb_paths, inventory)
    if not dataless:
        return []
    warnings: list[str] = []
    for name, predicates in sorted(dataless.items()):
        if not re.search(rf"\(\s*{re.escape(name)}\b", metta_query):
            continue
        route: Optional[str] = _ROUTES.get(name)
        warnings.append(
            f"`{name}` is defined and loaded, but the generic runtime space "
            f"holds NO facts for the predicate(s) it reads "
            f"({', '.join(predicates)}) — those rows are excluded because a "
            f"variable-slot match over them aborts hyperon 0.2.10. An empty "
            f"result here means 'this space has no rows', not 'no'."
            + (f" Use {route} instead." if route else "")
        )
    return warnings
