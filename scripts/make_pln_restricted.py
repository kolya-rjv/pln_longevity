#!/usr/bin/env python3
"""Generate vendor/pln_fork/lib_pln_restricted.metta from vendor/pln_fork/lib_pln_fork.metta.

The restricted rule set is Modus Ponens + Not-elimination + Revision (docs/petta_migration/fork_design/decision.md,
section 3).  It is produced by a FORM-LEVEL FILTER, never by hand edits: the vendored file is split into its
top-level forms, every form is classified as KEEP or DROP by a name this script spells out, and the kept forms are
copied byte for byte.  A form the script has not classified makes it fail, so refreshing the vendored file from a
newer upstream cannot silently add or lose a rule.

    python scripts/make_pln_restricted.py            # (re)write the artifact
    python scripts/make_pln_restricted.py --check    # exit 1 if the artifact on disk differs from a fresh build

Standard library only.  `build()` and `split_forms()` are imported by tests/test_pln_vendor.py.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT / "vendor" / "pln_fork"
SOURCE = VENDOR_DIR / "lib_pln_fork.metta"
TARGET = VENDOR_DIR / "lib_pln_restricted.metta"
#: the marker that ends the provenance header of the vendored file
HEADER_END = "; ---- END OF PROVENANCE HEADER ----"

#: Functions whose every clause is kept (a clause is `(= (NAME ...) ...)`).
KEEP_FUNCTIONS = (
    # helpers
    "clamp", "TupleConcat", "TupleCount", "and5", "min5", "/safe", "negate", "invert", "Test2",
    "InsertSorted", "InsertionSort", "Without", "ElementOf", "Unique",
    # truth functions of the three kept rules (Truth_c2w and Truth_w2c are Revision's confidence <-> weight maps)
    "Truth_c2w", "Truth_w2c", "Truth_ModusPonens", "Truth_Revision", "Truth_Negation",
    # the control loop: stamps, priority queue, Derive, Query, Config
    "StampDisjoint", "StampConcat", "BestCandidate", "PriorityRank", "PriorityRankNeg", "LimitSize",
    "PLN.Derive", "ConfidenceRank", "PLN.Query",
    "PLN.Config.MaxSteps", "PLN.Config.TaskQueueSize", "PLN.Config.BeliefQueueSize",
)
#: `(: NAME type)` declarations that are kept
KEEP_DECLARATIONS = ("Truth_c2w", "Truth_w2c")

#: Functions whose every clause is dropped: node-probability consistency checks, STV, and the truth functions
#: and guards of the dropped rules.
DROP_FUNCTIONS = (
    "smallest-intersection-probability", "largest-intersection-probability", "conditional-probability-consistency",
    "Consistency_ImplicationImplicantConjunction", "STV",
    "Truth_Deduction", "Truth_Induction", "Truth_Abduction", "Truth_SymmetricModusPonens", "Truth_inversion",
    "Truth_equivalenceToImplication", "TransitiveSimilarityStrength", "Truth_transitiveSimilarity",
    "simpleDeductionStrength", "Truth_evaluationImplication",
    "SymmetricModusPonensRuleGuard", "SyllogisticRuleGuard",
)
DROP_DECLARATIONS = (
    "smallest-intersection-probability", "largest-intersection-probability", "conditional-probability-consistency",
)

#: Inference rules `(= (|- PREMISES) ...)`, identified by their premise patterns (whitespace collapsed).
KEEP_RULES = {
    "(|- ($T $T1) ($T $T2))": "Revision",
    "(|- ($A $T1) ((Implication $A $B) $T2))": "ModusPonens",
    "(|- ((Not $A) $T))": "NotElimination",
}
DROP_RULES = {
    "(|- ($A $TruthA) (($LinkType $A $B) $TruthAB))": "SymmetricModusPonens",
    "(|- (($LinkType $A $B) $T1) (($LinkType $B $C) $T2))": "Deduction",
    "(|- (($LinkType $C $A) $T1) (($LinkType $C $B) $T2))": "Induction",
    "(|- (($LinkType $A $C) $T1) (($LinkType $B $C) $T2))": "Abduction",
    "(|- ((Evaluation (Predicate $x) (List (Concept $C))) $T1) ((Inheritance (Concept $S) (Concept $C)) $T2))":
        "EvaluationWithInheritance, unary",
    "(|- ((Evaluation (Predicate $x) (List (Concept $C1) (Concept $C2))) $T1) "
    "((Inheritance (Concept $S) (Concept $C1)) $T2))": "EvaluationWithInheritance, binary arg 1",
    "(|- ((Evaluation (Predicate $x) (List (Concept $C1) (Concept $C2))) $T1) "
    "((Inheritance (Concept $S) (Concept $C2)) $T2))": "EvaluationWithInheritance, binary arg 2",
    "(|- ((Inheritance $A $B) $Truth))": "InheritanceInversion",
    "(|- ((Implication $A $B) $Truth))": "ImplicationInversion",
    "(|- ((Equivalence $A $B) $Truth))": "EquivalenceToImplication (both clauses)",
    "(|- ((Similarity $A $B) $T1) ((Similarity $B $C) $T2))": "TransitiveSimilarity",
    "(|- ((Evaluation $A $B) $TruthAB) ((Implication $A $C) $TruthAC))": "EvaluationImplication",
    "(|- ((Member $A $B) $T1) ((Inheritance $B $C) $T2))": "MemberDeduction",
}


# ---------------------------------------------------------------------------------------------- reading forms

@dataclass(frozen=True)
class Form:
    start: int            # index of the opening paren in the source
    end: int              # index one past the matching closing paren
    text: str             # source[start:end], byte for byte
    lead: str             # the comment-only lines directly above the form (no blank line between), or ""
    kind: str             # "def" | "decl" | "other"
    name: str             # function name (def, decl) or ""
    rule: str             # collapsed premise patterns for a `|-` rule, else ""


def _scan(text: str):
    """Yield (kind, start, end) tokens: 'open', 'close', 'atom', 'string', 'comment'."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ";":
            j = text.find("\n", i)
            j = n if j < 0 else j
            yield ("comment", i, j)
            i = j
        elif c == "(":
            yield ("open", i, i + 1)
            i += 1
        elif c == ")":
            yield ("close", i, i + 1)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                raise ValueError(f"unterminated string at offset {i}")
            yield ("string", i, j + 1)
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '();"':
                j += 1
            yield ("atom", i, j)
            i = j


def _collapse(s: str) -> str:
    s = re.sub(r";[^\n]*", "", s)
    return re.sub(r"\s+", " ", s).strip().replace("( ", "(").replace(" )", ")")


def split_forms(text: str) -> list[Form]:
    """The top-level forms of `text`, each with its source span and its directly attached leading comment."""
    forms: list[Form] = []
    depth = 0
    start = -1
    # (child spans of the form being read, depth 1 only) -> the children give name and premises
    children: list[tuple[int, int]] = []
    child_start = -1
    last_comment_block: list[tuple[int, int]] = []   # consecutive comment-only lines seen since the last form
    prev_end_line = -1
    line_of = lambda off: text.count("\n", 0, off)  # noqa: E731
    for kind, a, b in _scan(text):
        if depth == 0:
            if kind == "comment":
                ln = line_of(a)
                if last_comment_block and ln != line_of(last_comment_block[-1][0]) + 1:
                    last_comment_block = []
                last_comment_block.append((a, b))
                continue
            if kind != "open":
                raise ValueError(f"stray token {text[a:b]!r} outside a form at offset {a}")
            # a blank line between the comment block and the form detaches it
            if last_comment_block and line_of(a) != line_of(last_comment_block[-1][0]) + 1:
                last_comment_block = []
            start = a
            depth = 1
            children = []
            child_start = -1
            continue
        if kind == "comment":
            continue
        if kind == "open":
            if depth == 1:
                child_start = a
            depth += 1
        elif kind == "close":
            depth -= 1
            if depth == 1:
                children.append((child_start, b))
            elif depth == 0:
                lead = ""
                if last_comment_block:
                    lead = text[last_comment_block[0][0]:last_comment_block[-1][1]]
                forms.append(_classify_shape(text, start, b, lead, children))
                last_comment_block = []
        else:  # atom / string at depth 1 is a child; deeper is inside a child
            if depth == 1:
                children.append((a, b))
    if depth != 0:
        raise ValueError("unbalanced parentheses")
    return forms


def _classify_shape(text: str, start: int, end: int, lead: str, children: list[tuple[int, int]]) -> Form:
    form = text[start:end]
    kids = [text[a:b] for a, b in children]
    head = kids[0] if kids else ""
    if head == "=" and len(kids) >= 3 and kids[1].startswith("("):
        lhs = kids[1]
        inner = _collapse(lhs)
        name = inner[1:].split(" ", 1)[0].rstrip(")")
        if name == "|-":
            return Form(start, end, form, lead, "def", "|-", inner)
        return Form(start, end, form, lead, "def", name, "")
    if head == ":" and len(kids) >= 3:
        return Form(start, end, form, lead, "decl", kids[1], "")
    return Form(start, end, form, lead, "other", "", "")


def classify(form: Form) -> str | None:
    """'keep', 'drop', or None for a form this script has not been told about."""
    if form.kind == "def" and form.name == "|-":
        if form.rule in KEEP_RULES:
            return "keep"
        if form.rule in DROP_RULES:
            return "drop"
        return None
    if form.kind == "def":
        if form.name in KEEP_FUNCTIONS:
            return "keep"
        if form.name in DROP_FUNCTIONS:
            return "drop"
        return None
    if form.kind == "decl":
        if form.name in KEEP_DECLARATIONS:
            return "keep"
        if form.name in DROP_DECLARATIONS:
            return "drop"
        return None
    return None


# ---------------------------------------------------------------------------------------------- building

def vendored_body(source_text: str) -> str:
    """The vendored file without its provenance header."""
    if HEADER_END not in source_text:
        raise ValueError("the vendored file has no provenance header marker")
    return source_text.split(HEADER_END + "\n", 1)[1]


def build(source_path: Path = SOURCE) -> str:
    """The text of the restricted artifact for the vendored file at `source_path`."""
    raw = source_path.read_bytes()
    body = vendored_body(raw.decode("utf-8"))
    forms = split_forms(body)
    unknown = [f for f in forms if classify(f) is None]
    if unknown:
        lines = [f"  offset {f.start}: {_collapse(f.text)[:100]}" for f in unknown]
        raise SystemExit("make_pln_restricted: forms the filter has no rule for (refresh of the vendored file?):\n"
                         + "\n".join(lines))
    # every name the tables mention must still exist, or the tables are stale
    present_fn = {f.name for f in forms if f.kind == "def"}
    present_decl = {f.name for f in forms if f.kind == "decl"}
    present_rules = {f.rule for f in forms if f.name == "|-"}
    stale = ([f"function {n}" for n in KEEP_FUNCTIONS + DROP_FUNCTIONS if n not in present_fn]
             + [f"declaration {n}" for n in KEEP_DECLARATIONS + DROP_DECLARATIONS if n not in present_decl]
             + [f"rule {r}" for r in list(KEEP_RULES) + list(DROP_RULES) if r not in present_rules])
    if stale:
        raise SystemExit("make_pln_restricted: the filter tables name forms the vendored file no longer has:\n  "
                         + "\n  ".join(stale))
    kept = [f for f in forms if classify(f) == "keep"]
    dropped = [f for f in forms if classify(f) == "drop"]
    sha = hashlib.sha256(raw).hexdigest()
    rules_kept = ", ".join(KEEP_RULES.values())
    header = (
        "; ======================================================================================================\n"
        "; GENERATED FILE -- do not edit.  scripts/make_pln_restricted.py writes it from lib_pln_fork.metta.\n"
        f"; Source: vendor/pln_fork/lib_pln_fork.metta, sha256 {sha}\n"
        f"; Rule set: {rules_kept}.  {len(kept)} of {len(forms)} top-level forms kept, {len(dropped)} dropped\n"
        "; (consistency checks, STV, Deduction, Induction, Abduction, both Inversions, Equivalence, Transitive\n"
        "; Similarity, Evaluation-Implication, Member-Deduction, Symmetric Modus Ponens, the three\n"
        "; Evaluation-with-Inheritance rules).  Every kept form is byte-identical to the vendored one;\n"
        "; tests/test_pln_vendor.py regenerates this file and compares.\n"
        "; ======================================================================================================\n\n"
    )
    pieces = []
    for f in kept:
        pieces.append((f.lead + "\n" if f.lead else "") + f.text)
    return header + "\n\n".join(pieces) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the artifact on disk is not a fresh build")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--target", type=Path, default=TARGET)
    args = ap.parse_args(argv)
    text = build(args.source)
    if args.check:
        current = args.target.read_text(encoding="utf-8") if args.target.exists() else None
        if current != text:
            print(f"{args.target} is stale or missing; run scripts/make_pln_restricted.py", file=sys.stderr)
            return 1
        print(f"{args.target} is current ({len(text)} bytes)")
        return 0
    args.target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.target} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
