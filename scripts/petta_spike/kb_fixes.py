"""The three Phase 2 load fixes of docs/petta_migration/LAUNCH.md, applied to KB TEXT in memory.

The repo is read-only for the spike, so the fixes are text transformations of what the kit
reads from the work tree, never edits of it. They are meant to be validated on HYPERON first
(record_baseline.py --check-fixes runs every shape with and without them and records whether
the answers are byte-identical), and only then fed to PeTTa (run_loads.py / run_shapes.py
--fixes all).

  superpose_empty   (superpose ())                      -> (empty)
                    PeTTa: build_superpose_branches([], _, []) then disj_list([], _) has no clause
                    (translator.pl:1533-1538), so a form containing (superpose ()) does not compile.
  stored_calls      (Effect A B Pos (stv 0.85 (evidence-confidence T)))  as a FACT
                    -> (Effect A B Pos (stv 0.85 0.65))
                    PeTTa does not evaluate a value bound out of a space (examples/eval.metta), so
                    combine-tv would multiply a list. The number is read from the tier table in
                    epistemic_calibration.metta, so the table stays the single source of truth.
                    Rule bodies, `(= ...)` and `(: ...)` forms are left alone: they evaluate normally.
  nested_conj       (, A (, B C))                       -> (, A B C)
                    PeTTa turns the inner (, B C) into a lookup of a ','-headed fact (spaces.pl:220).
"""
from __future__ import annotations

import bisect
import re
from typing import Iterable, Optional

from kit_common import child_spans, top_level_spans

ALL_FIXES = ("superpose_empty", "stored_calls", "nested_conj")


def code_mask(text: str) -> bytearray:
    """1 where a character is code, 0 inside a ';' comment or a "string"."""
    mask = bytearray(b"\x01") * len(text)
    in_str = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            mask[i] = 0
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == ";":
            while i < n and text[i] != "\n":
                mask[i] = 0
                i += 1
            continue
        if ch == '"':
            in_str = True
            mask[i] = 0
        i += 1
    return mask


def _code_matches(text: str, pattern: re.Pattern, mask: Optional[bytearray] = None):
    mask = mask if mask is not None else code_mask(text)
    for m in pattern.finditer(text):
        if mask[m.start()] and mask[m.end() - 1]:
            yield m


_SUPERPOSE_RE = re.compile(r"\(superpose\s*\(\s*\)\s*\)")
_EVCONF_RE = re.compile(r"\(evidence-confidence\s+([^\s()\";]+)\s*\)")
_TIER_RE = re.compile(r"^\(=\s*\(evidence-confidence\s+([^\s()]+)\)\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*\)", re.M)


def fix_superpose_empty(text: str) -> tuple[str, int]:
    out, last, n = [], 0, 0
    for m in _code_matches(text, _SUPERPOSE_RE):
        out.append(text[last:m.start()])
        out.append("(empty)")
        last = m.end()
        n += 1
    out.append(text[last:])
    return "".join(out), n


def read_tier_table(epistemic_text: str) -> dict[str, str]:
    """{'RCT_Human': '1.0', ...}: the literal text of each tier number, so the substituted fact parses to the same float."""
    mask = code_mask(epistemic_text)
    return {m.group(1): m.group(2) for m in _code_matches(epistemic_text, _TIER_RE, mask)}


def fix_stored_calls(text: str, tiers: dict[str, str]) -> tuple[str, int, list[str]]:
    """Replace (evidence-confidence T) inside FACT forms by the tier's number. Returns (text, replaced, unresolved tiers)."""
    mask = code_mask(text)
    pieces, last, replaced, unresolved = [], 0, 0, []
    for s, e in top_level_spans(text):
        if text[s] == "!":
            continue
        kids = child_spans(text, s, e)
        head = text[kids[0][0]:kids[0][1]] if kids else ""
        if head in ("=", ":"):
            continue
        seg = text[s:e]
        if "evidence-confidence" not in seg:
            continue
        seg_mask = mask[s:e]
        new, seg_last = [], 0
        for m in _EVCONF_RE.finditer(seg):
            if not (seg_mask[m.start()] and seg_mask[m.end() - 1]):
                continue
            tier = m.group(1)
            if tier not in tiers:
                unresolved.append(tier)
                continue
            new.append(seg[seg_last:m.start()])
            new.append(tiers[tier])
            seg_last = m.end()
            replaced += 1
        if seg_last:
            new.append(seg[seg_last:])
            pieces.append(text[last:s])
            pieces.append("".join(new))
            last = e
    pieces.append(text[last:])
    return "".join(pieces), replaced, unresolved


_CONJ_RE = re.compile(r"\(,(?=\s)")


def _form_end(text: str, start: int) -> int:
    """Index just past the form that opens at text[start] == '('."""
    depth, i, n, in_str, esc = 0, start, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def fix_nested_conjunction(text: str) -> tuple[str, int]:
    """Flatten (, A (, B C)) into (, A B C), to any depth. Returns (text, number of splices)."""
    total = 0
    while True:
        mask = code_mask(text)
        starts = [m.start() for m in _code_matches(text, _CONJ_RE, mask)]
        changed = False
        for st in reversed(starts):                    # last first: earlier offsets stay valid
            en = _form_end(text, st)
            kids = child_spans(text, st, en)
            for cs, ce in reversed(kids[1:]):
                if text[cs:ce].startswith("(,") and re.match(r"\(,\s", text[cs:ce]):
                    inner = text[cs + 2:ce - 1].strip()
                    text = text[:cs] + inner + text[ce:]
                    total += 1
                    changed = True
            if changed:
                break                                  # offsets moved: rescan from the top
        if not changed:
            return text, total


def apply_fixes(text: str, which: Iterable[str], tiers: Optional[dict[str, str]] = None) -> tuple[str, dict]:
    """Apply the named fixes in a fixed order to one file's text. The report counts what changed."""
    which = set(which)
    report: dict = {}
    if "superpose_empty" in which:
        text, n = fix_superpose_empty(text)
        report["superpose_empty"] = n
    if "stored_calls" in which:
        text, n, unresolved = fix_stored_calls(text, tiers or {})
        report["stored_calls"] = n
        if unresolved:
            report["stored_calls_unresolved"] = sorted(set(unresolved))
    if "nested_conj" in which:
        text, n = fix_nested_conjunction(text)
        report["nested_conj"] = n
    return text, report


def parse_fix_spec(spec: Optional[str]) -> set[str]:
    """'none' | 'all' | 'superpose_empty,stored_calls' -> the set of fix names."""
    if not spec or spec == "none":
        return set()
    if spec == "all":
        return set(ALL_FIXES)
    names = {s.strip() for s in spec.split(",") if s.strip()}
    bad = names - set(ALL_FIXES)
    if bad:
        raise ValueError(f"unknown fix(es) {sorted(bad)}; choose from {ALL_FIXES}, 'all' or 'none'")
    return names


def fix_kb_texts(named_texts: list[tuple[str, str]], which: Iterable[str],
                 epistemic_text: Optional[str] = None) -> tuple[list[tuple[str, str]], dict]:
    """Apply the fixes to several (name, text) files. The tier table comes from `epistemic_text`, or from the
    entry named epistemic_calibration.metta when it is among the files."""
    which = set(which)
    if not which:
        return named_texts, {}
    if epistemic_text is None:
        for name, t in named_texts:
            if name.endswith("epistemic_calibration.metta"):
                epistemic_text = t
                break
    tiers = read_tier_table(epistemic_text or "")
    out, per_file = [], {}
    for name, t in named_texts:
        new, rep = apply_fixes(t, which, tiers)
        if any(v for k, v in rep.items() if not k.endswith("unresolved")) or rep.get("stored_calls_unresolved"):
            per_file[name] = rep
        out.append((name, new))
    return out, per_file
