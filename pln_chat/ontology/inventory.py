"""What the runtime KB ACTUALLY holds, as opposed to what it declares.

The problem this solves
-----------------------
`ontology/registry.py` harvests five shapes out of a .metta file — `(: X T)`,
`(= (f …) …)`, a 1-ary constant definition, `(Inheritance a b)` and
`(InstanceOf a T)` — and drops every other ground fact on the floor. That single
design choice produces both halves of the "valid but empty" failure the
2026-09-18 API evaluation hit 6 times in 36 questions:

* **False positives.** `logical_predicates.metta` DECLARES a vocabulary —
  `(: TargetsHallmark (-> Intervention HallmarkOfAging Atom))` and about thirty
  more. The registry records the declaration, the symbol index shows it to the
  LLM, and the validator waves it through. The runtime holds ZERO
  `TargetsHallmark` facts, so the query validates and returns nothing. The
  translator used `TargetsHallmark` three times.

* **False negatives, which are worse.** An argument of an ordinary ground fact
  is never registered, so 71 symbols that genuinely exist in the runtime KB —
  `MTORC1`, `AMPK`, `SIRT1`, `IL6`, `TNFAlpha`, `Mouse`, `Human`, `TERT`,
  `CDKN2A_P16` — are unknown to the registry. `/metta/run` rejects those queries
  with 422 although the runtime would have answered them.

This module reads the runtime files once and records what is really there:
every ground predicate with its arities and fact count, every entity symbol
that appears as an argument, and every function the rules define. That inventory
is the single source of truth for three things:

1. the validator, which can now tell an unknown symbol from an un-harvested one;
2. a "schema card" for the LLM that says which predicates have data and how
   much, instead of a flat list where a declaration and 400 facts look alike;
3. a compact substitute for pasting a multi-megabyte ETL dump into the prompt.

Nothing here evaluates MeTTa — it is a text scan, so it costs milliseconds and
cannot panic the way loading the same data into hyperon would.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Expressions that are DECLARATIONS or RULES rather than ground facts.
_DECL_HEADS = {":", "="}

# A bare symbol argument: letters/digits/_-. and nothing exotic. Deliberately
# excludes variables ($x), numbers, strings and nested expressions.
_SYMBOL_ARG = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


@dataclass
class PredicateInfo:
    """One ground predicate as it actually occurs in the KB."""
    name: str
    fact_count: int = 0
    arities: Counter = field(default_factory=Counter)
    sources: set[str] = field(default_factory=set)
    #: A few representative first arguments, for the schema card.
    sample_args: list[str] = field(default_factory=list)

    @property
    def arity(self) -> Optional[int]:
        """The dominant arity, or None when the predicate is never grounded."""
        if not self.arities:
            return None
        return self.arities.most_common(1)[0][0]


@dataclass
class RuntimeInventory:
    """Ground truth about one set of .metta files."""
    predicates: dict[str, PredicateInfo] = field(default_factory=dict)
    #: symbol -> the predicates it appears in as an argument
    entities: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    #: function name -> declared/defined arity (rules and (: f (-> …)) decls)
    functions: dict[str, Optional[int]] = field(default_factory=dict)
    #: declared type names, from (: T Type) and the right-hand side of (: x T)
    types: set[str] = field(default_factory=set)
    #: every symbol DECLARED as a predicate but never grounded and never
    #: defined by a rule — i.e. a query using it validates and returns nothing
    declared_only: set[str] = field(default_factory=set)
    #: names the rule layer DEFINES `(= (name …) …)` or uses as a constructor
    #: inside a rule's left-hand side; those are callable/structural, not empty
    #: predicates, so they are excluded from `declared_only`
    defined: set[str] = field(default_factory=set)
    files: list[str] = field(default_factory=list)

    # -- queries the rest of the code asks ------------------------------------

    def knows_symbol(self, token: str) -> bool:
        return (
            token in self.predicates
            or token in self.entities
            or token in self.functions
            or token in self.types
        )

    def is_grounded(self, predicate: str) -> bool:
        info = self.predicates.get(predicate)
        return bool(info and info.fact_count)

    def grounded_predicates(self) -> list[PredicateInfo]:
        return sorted(
            (p for p in self.predicates.values() if p.fact_count),
            key=lambda p: (-p.fact_count, p.name),
        )

    def fact_total(self) -> int:
        return sum(p.fact_count for p in self.predicates.values())

    def stats(self) -> dict:
        return {
            "files": len(self.files),
            "ground_facts": self.fact_total(),
            "grounded_predicates": len(self.grounded_predicates()),
            "declared_but_empty_predicates": sorted(self.declared_only),
            "entities": len(self.entities),
            "functions": len(self.functions),
            "types": len(self.types),
        }


# ── Parsing ──────────────────────────────────────────────────────────────────

def _strip_comment(line: str) -> str:
    depth = 0
    in_string = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ";":
            return line[:i]
    return line


def iter_top_level(text: str) -> Iterable[str]:
    """Yield each top-level S-expression, comments removed."""
    buf: list[str] = []
    depth = 0
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        for ch in line:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        buf.append(line)
        if depth <= 0:
            expr = " ".join(buf).strip()
            buf = []
            depth = 0
            if expr:
                yield expr
    if buf:
        expr = " ".join(buf).strip()
        if expr:
            yield expr


def split_args(body: str) -> list[str]:
    """Split an expression body into its top-level arguments."""
    args: list[str] = []
    depth = 0
    token: list[str] = []
    in_string = False
    for ch in body:
        if ch == '"':
            in_string = not in_string
            token.append(ch)
            continue
        if in_string:
            token.append(ch)
            continue
        if ch == "(":
            depth += 1
            token.append(ch)
        elif ch == ")":
            depth -= 1
            token.append(ch)
        elif ch.isspace() and depth == 0:
            if token:
                args.append("".join(token))
                token = []
        else:
            token.append(ch)
    if token:
        args.append("".join(token))
    return args


def _head_and_args(expr: str) -> Optional[tuple[str, list[str]]]:
    text = expr.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return None
    parts = split_args(inner)
    if not parts:
        return None
    return parts[0], parts[1:]


def _declared_name(args: list[str]) -> Optional[tuple[str, Optional[int], str]]:
    """For `(: X <sig>)` return (X, arity or None, sig)."""
    if len(args) < 2:
        return None
    name = args[0]
    sig = " ".join(args[1:])
    if not _SYMBOL_ARG.match(name):
        return None
    if sig.startswith("(->"):
        # (-> A B Atom) — the last element is the return type.
        inner = split_args(sig[1:-1].strip())
        arity = max(0, len(inner) - 2)   # drop the "->" and the return type
        return name, arity, sig
    return name, None, sig


def build_inventory(paths: Iterable[Path]) -> RuntimeInventory:
    """Scan `paths` and record what is actually grounded in them."""
    inv = RuntimeInventory()
    declared_predicates: set[str] = set()

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        inv.files.append(path.name)
        for expr in iter_top_level(text):
            parsed = _head_and_args(expr)
            if parsed is None:
                continue
            head, args = parsed

            if head == ":":
                decl = _declared_name(args)
                if decl is None:
                    continue
                name, arity, sig = decl
                if sig == "Type":
                    inv.types.add(name)
                elif arity is not None:
                    inv.functions[name] = arity
                    declared_predicates.add(name)
                else:
                    # (: Elamipretide Pharmaceutical) — an entity with a type.
                    inv.entities[name].add(":")
                    inv.types.add(sig)
                continue

            if head == "=":
                # (= (fn $a $b) body) or (= (fn Const) value)
                if args and args[0].startswith("("):
                    fn = _head_and_args(args[0])
                    if fn is not None:
                        inv.functions.setdefault(fn[0], len(fn[1]))
                        inv.defined.add(fn[0])
                        # A 1-ary definition over a CONSTANT is a lookup table
                        # entry — its argument is a real entity.
                        for a in fn[1]:
                            if _SYMBOL_ARG.match(a):
                                inv.entities[a].add(fn[0])
                            elif a.startswith("("):
                                # A destructuring pattern like
                                # `(= (score-of (scored $i $s $tv)) $s)` — its
                                # head is a CONSTRUCTOR the rules build and
                                # match on, not an empty predicate.
                                nested = _head_and_args(a)
                                if nested is not None and _SYMBOL_ARG.match(nested[0]):
                                    inv.defined.add(nested[0])
                continue

            # Anything else is a ground fact.
            info = inv.predicates.get(head)
            if info is None:
                info = PredicateInfo(name=head)
                inv.predicates[head] = info
            info.fact_count += 1
            info.arities[len(args)] += 1
            info.sources.add(path.name)
            for a in args:
                if _SYMBOL_ARG.match(a) and not _NUMBER.match(a):
                    inv.entities[a].add(head)
                elif a.startswith("("):
                    # A nested expression: harvest its own head and symbols so
                    # (Causes Rapamycin (Inhibits MTORC1)) registers MTORC1.
                    nested = _head_and_args(a)
                    if nested is not None:
                        n_head, n_args = nested
                        if _SYMBOL_ARG.match(n_head):
                            inv.entities[n_head].add(head)
                        for na in n_args:
                            if _SYMBOL_ARG.match(na) and not _NUMBER.match(na):
                                inv.entities[na].add(head)
            if len(info.sample_args) < 3 and args:
                first = args[0]
                if _SYMBOL_ARG.match(first) and first not in info.sample_args:
                    info.sample_args.append(first)

    inv.declared_only = {
        name for name in declared_predicates
        if not inv.predicates.get(name, PredicateInfo(name)).fact_count
        and name not in inv.defined
    }
    return inv


# ── Caching ──────────────────────────────────────────────────────────────────

_CACHE: dict[tuple, RuntimeInventory] = {}


def _stamp(paths: list[Path]) -> tuple:
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime, st.st_size))
        except OSError:
            out.append((str(p), 0.0, 0))
    return tuple(out)


def inventory_for(paths: Iterable[Path]) -> RuntimeInventory:
    """`build_inventory`, cached per (path, mtime, size) set."""
    path_list = list(paths)
    key = _stamp(path_list)
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_inventory(path_list)
        _CACHE[key] = cached
    return cached


# ── The schema card ──────────────────────────────────────────────────────────

def schema_card(
    inv: RuntimeInventory,
    *,
    max_predicates: int = 60,
    max_entities: int = 0,
) -> str:
    """A compact, honest description of what can be queried.

    Used in two places: as the header of the ontology snapshot (so the LLM sees
    fact counts next to predicate names), and INSTEAD of dumping a bulk ETL file
    into the prompt — a 525 KB `cellage_genes.metta` becomes a dozen lines.
    """
    lines: list[str] = []
    grounded = inv.grounded_predicates()[:max_predicates]
    if grounded:
        lines.append("Grounded predicates (name/arity — fact count):")
        for info in grounded:
            arity = info.arity if info.arity is not None else "?"
            sample = ""
            if info.sample_args:
                sample = "  e.g. " + ", ".join(info.sample_args[:2])
            lines.append(f"  ({info.name}/{arity}) x{info.fact_count}{sample}")
    if inv.declared_only:
        lines.append("")
        lines.append(
            "DECLARED BUT EMPTY — these predicates exist in the type declarations "
            "and hold NO facts. A query using one validates and returns nothing, "
            "so do not use them:"
        )
        lines.append("  " + ", ".join(sorted(inv.declared_only)))
    if max_entities and inv.entities:
        names = sorted(inv.entities)[:max_entities]
        lines.append("")
        lines.append(f"Entities ({len(inv.entities)} total, {len(names)} shown):")
        lines.append("  " + ", ".join(names))
    return "\n".join(lines)


def file_card(path: Path, *, max_predicates: int = 20) -> str:
    """A schema card for ONE file, to paste instead of its contents."""
    inv = inventory_for([path])
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    header = (
        f";; {path.name} — {size:,} bytes, {inv.fact_total():,} ground facts. "
        f"Summarised rather than pasted verbatim: the full text would not fit "
        f"the model's context window. Query it through the structured endpoints."
    )
    return header + "\n" + schema_card(inv, max_predicates=max_predicates)


def summarise_oversized(
    raw_contents: dict[str, str],
    paths: Iterable[Path],
    *,
    max_bytes: int,
) -> tuple[dict[str, str], list[str]]:
    """Replace any file over `max_bytes` with its schema card.

    The LLM context is the only place bulk data is ever pasted verbatim, and it
    is where the 2026-09-18 evaluation's two hard failures came from: selecting
    a CellAge or GenAge ETL file pushed the prompt to 417,000 tokens, past the
    model's window, and the request came back as a billed upstream 400. The
    curated inference layers — the demo-function contracts, the calibration
    tables, the honesty notes — are all well under the limit and stay verbatim,
    because their prose is what the translator reasons from. A 525 KB row dump
    is not; a dozen lines saying which predicates it holds and how many facts
    each has is strictly more useful per token.

    Returns (contents, summarised_filenames).
    """
    by_name = {p.name: p for p in paths}
    out: dict[str, str] = {}
    summarised: list[str] = []
    for name, text in raw_contents.items():
        path = by_name.get(name)
        size = len(text.encode("utf-8"))
        if max_bytes and size > max_bytes and path is not None:
            out[name] = file_card(path)
            summarised.append(name)
        else:
            out[name] = text
    return out, summarised
