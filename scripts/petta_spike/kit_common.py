"""Shared helpers for the PeTTa spike kit. Standard library only.

Everything here is runtime-neutral: paths and environment, s-expression
reading and printing, the answer NORMALISER (PeTTa rendering -> the rendering the
repo's parsers were written against) and the answer COMPARATOR that run_shapes.py
and run_loads.py use. It imports nothing from the repo and nothing from PeTTa.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

KIT_VERSION = 1
KIT_DIR = Path(__file__).resolve().parent

DEFAULT_REPO = "/Users/kolya/PycharmProjects/pln_longevity_petta"
DEFAULT_PETTA_HOME = DEFAULT_REPO + "/.petta/PeTTa"
DEFAULT_PYTHON = "/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python"


# ── paths and environment ────────────────────────────────────────────────────

def repo_root() -> Path:
    """The work tree (READ-ONLY for the kit). Override with PLN_REPO."""
    return Path(os.environ.get("PLN_REPO") or DEFAULT_REPO).resolve()


def petta_home() -> Path:
    """The PeTTa checkout. PETTA_HOME, else PETTA_PATH (the name PeTTa's own python wrapper reads), else the pinned one."""
    return Path(os.environ.get("PETTA_HOME") or os.environ.get("PETTA_PATH") or DEFAULT_PETTA_HOME)


def out_dir() -> Path:
    """Where every run writes. KIT_OUT, else <kit>/out."""
    p = Path(os.environ.get("KIT_OUT") or (KIT_DIR / "out"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def baseline_path() -> Path:
    return Path(os.environ.get("KIT_BASELINE") or (KIT_DIR / "baseline_hyperon.json"))


def load_cache_dir() -> Path:
    """PeTTa's load_metta_file_cached keeps snapshots in $PETTA_LOAD_CACHE; keep them inside the kit's output."""
    p = Path(os.environ.get("PETTA_LOAD_CACHE") or (out_dir() / "petta_load_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def kb_cache_dir() -> Path:
    """Stable directory for the temp KB files of the cached-load mode (the snapshot key contains the file path)."""
    p = out_dir() / "kb_files"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def env_report() -> dict:
    """Facts about the machine, gathered WITHOUT running swipl or importing PeTTa/janus."""
    import shutil
    swipl = shutil.which("swipl") or ("/opt/homebrew/bin/swipl" if Path("/opt/homebrew/bin/swipl").exists() else None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "swipl_path": swipl,
        "petta_home": str(petta_home()),
        "petta_home_exists": (petta_home() / "src" / "main.pl").exists(),
        "repo": str(repo_root()),
        "cpu_count": os.cpu_count(),
    }


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def json_load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rss_mb(pid: Optional[int] = None) -> Optional[float]:
    """Resident set size in MB. /proc on Linux, `ps` on macOS. None when unknown."""
    pid = pid or os.getpid()
    try:
        statm = Path(f"/proc/{pid}/statm")
        if statm.exists():
            pages = int(statm.read_text().split()[1])
            return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return round(int(out) / 1024.0, 1) if out else None
    except Exception:
        return None


def peak_rss_mb() -> Optional[float]:
    """Peak RSS of THIS process (ru_maxrss: bytes on macOS, KiB on Linux)."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round((v / 1e6) if sys.platform == "darwin" else (v / 1024.0), 1)
    except Exception:
        return None


# ── s-expression reading and printing ────────────────────────────────────────

#: strings first (they may contain ; and parentheses), then parens, then symbols.
_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[()]|[^\s()"]+')
_NUM_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
_INT_RE = re.compile(r"^[-+]?\d+$")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def parse_sexpr(text: str) -> Any:
    """A MeTTa answer string -> nested lists of token strings (a string keeps its quotes).

    Several top-level items (never produced by an answer) come back as a list.
    """
    stack: list[list] = [[]]
    for tok in tokens(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:               # unbalanced: be forgiving, keep what we have
                continue
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    while len(stack) > 1:                    # unbalanced the other way
        done = stack.pop()
        stack[-1].append(done)
    top = stack[0]
    return top[0] if len(top) == 1 else top


def fmt_sexpr(node: Any) -> str:
    if isinstance(node, list):
        return "(" + " ".join(fmt_sexpr(c) for c in node) + ")"
    return str(node)


def is_number_token(tok: str) -> bool:
    return isinstance(tok, str) and bool(_NUM_RE.match(tok))


def _token_float(tok: str) -> Optional[float]:
    if isinstance(tok, str) and _NUM_RE.match(tok):
        try:
            return float(tok)
        except ValueError:
            return None
    return None


# ── the NORMALISER: PeTTa rendering -> the rendering the repo's parsers expect ─
#
# What the repo's parsers were written against is hyperon's rendering:
#   * shortest-round-trip floats (0.17017138125000003, 1e-5, 3.0)  -> _STV_RE in pln_runner.py (accepts any exponent form)
#   * booleans True/False                                      -> linage2_router.py:259-266 string compares
#   * strings with double quotes, the empty tuple as ()
# PeTTa prints with SWI-Prolog's write/1: shortest round-trip floats (1.0e-5, 1.0e22),
# ints as ints, booleans in lower case, and ±1.0Inf / 1.5NaN for the specials.

def normalise_atom(tok: str) -> str:
    """One token. Strings, parentheses and symbols other than the cases below pass through unchanged."""
    if tok in ("true", "false"):
        return "True" if tok == "true" else "False"
    if tok.startswith('"') or tok in ("(", ")"):
        return tok
    if tok in ("1.0Inf", "inf", "+inf"):
        return "inf"
    if tok in ("-1.0Inf", "-inf"):
        return "-inf"
    if re.fullmatch(r"[-+]?\d+\.\d+NaN", tok) or tok == "nan":
        return "NaN"
    if _NUM_RE.match(tok):
        if _INT_RE.match(tok):
            try:
                return str(int(tok))
            except ValueError:
                return tok
        try:
            return hyperon_float(float(tok))
        except (ValueError, OverflowError):
            return tok
    return tok


def hyperon_float(v: float) -> str:
    """A float the way hyperon 0.2.10 prints it: shortest round-trip digits, Rust-style exponent.

    Measured (baseline_hyperon.json, probe:print:sci-literal-*): 1e-05 -> `1e-5`, 2.0e-75 -> `2e-75`, 0.1+0.2 ->
    `0.30000000000000004`, -0.0 -> `-0.0`, 3.0 stays `3.0`. Python's repr agrees except for the exponent
    (`1e-05`, `1e+22`), which is rewritten here.
    """
    s = repr(v)
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant}e{int(exp)}"
    return s


def normalise_answer(raw: str) -> str:
    """A whole answer string -> hyperon-style rendering. Only the tokens change; the text between them is kept."""
    return _TOKEN_RE.sub(lambda m: normalise_atom(m.group(0)), raw or "")


# ── the COMPARATOR ───────────────────────────────────────────────────────────

CLASS_ORDER = ["identical", "number_format_only", "within_tolerance", "ordering_only", "different", "error"]
_SEVERITY = {"identical": 0, "number_format_only": 1, "within_tolerance": 2, "ordering_only": 3, "different": 4, "error": 5}


def _close(a: float, b: float, rel: float, abs_: float) -> bool:
    if a == b:
        return True
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _tree_class(a: Any, b: Any, rel: float, abs_: float, tags: set, maxrel: list) -> str:
    """identical | number_format_only | within_tolerance | different for two parsed answers."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return "different"
        worst = "identical"
        for x, y in zip(a, b):
            c = _tree_class(x, y, rel, abs_, tags, maxrel)
            if _SEVERITY[c] > _SEVERITY[worst]:
                worst = c
            if worst == "different":
                return worst
        return worst
    if isinstance(a, list) or isinstance(b, list):
        return "different"
    if a == b:
        return "identical"
    if a.lower() in ("true", "false") and a.lower() == b.lower():
        tags.add("bool_spelling")
        return "number_format_only"
    if a.startswith("$") and b.startswith("$"):
        tags.add("variable_names")
        return "number_format_only"
    fa, fb = _token_float(a), _token_float(b)
    if fa is not None and fb is not None:
        if fa == fb:
            tags.add("int_vs_float" if (_INT_RE.match(a) is None) != (_INT_RE.match(b) is None) else "float_format")
            return "number_format_only"
        if _close(fa, fb, rel, abs_):
            denom = max(abs(fa), abs(fb)) or 1.0
            maxrel.append(abs(fa - fb) / denom)
            return "within_tolerance"
        return "different"
    # inf / nan spellings
    if normalise_atom(a) == normalise_atom(b):
        tags.add("special_float_spelling")
        return "number_format_only"
    return "different"


def compare_answers(a: str, b: str, rel: float = 1e-9, abs_: float = 1e-12) -> dict:
    """Compare two answer strings (baseline a, candidate b). Returns {'class', 'tags', 'max_rel'}."""
    if a == b:
        return {"class": "identical", "tags": [], "max_rel": 0.0}
    tags: set = set()
    maxrel: list = []
    cls = _tree_class(parse_sexpr(a), parse_sexpr(b), rel, abs_, tags, maxrel)
    return {"class": cls, "tags": sorted(tags), "max_rel": max(maxrel) if maxrel else 0.0}


def _children_match(a: Any, b: Any, rel: float, abs_: float) -> Optional[dict]:
    """Two tuple answers that differ only in the order of their elements -> the worst element class, else None."""
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and len(a) > 1):
        return None
    if not all(isinstance(x, list) for x in a + b):
        return None
    used = [False] * len(b)
    worst, tags, maxrel = "identical", set(), []
    for x in a:
        best = None
        for j, y in enumerate(b):
            if used[j]:
                continue
            t, m = set(), []
            c = _tree_class(x, y, rel, abs_, t, m)
            if c != "different" and (best is None or _SEVERITY[c] < _SEVERITY[best[0]]):
                best = (c, j, t, m)
        if best is None:
            return None
        used[best[1]] = True
        if _SEVERITY[best[0]] > _SEVERITY[worst]:
            worst = best[0]
        tags |= best[2]
        maxrel += best[3]
    return {"class": worst, "tags": sorted(tags), "max_rel": max(maxrel) if maxrel else 0.0}


def compare_groups(base: list[str], other: list[str], rel: float = 1e-9, abs_: float = 1e-12) -> dict:
    """Compare the answer lists of ONE `!` expression (hyperon baseline vs candidate).

    class: identical | number_format_only | within_tolerance | ordering_only | different
    'ordering_only' = the same answers in another order, either as the list or inside one tuple answer;
    its 'inner' key carries the worst per-element class (number_format_only etc.) under the reordering.
    """
    if base == other:
        return {"class": "identical", "tags": [], "max_rel": 0.0, "detail": ""}
    if len(base) == len(other):
        worst, tags, maxrel, diffs = "identical", set(), [], []
        for i, (x, y) in enumerate(zip(base, other)):
            r = compare_answers(x, y, rel, abs_)
            tags |= set(r["tags"])
            if r["max_rel"]:
                maxrel.append(r["max_rel"])
            if r["class"] != "identical":
                diffs.append((i, x, y, r["class"]))
            if _SEVERITY[r["class"]] > _SEVERITY[worst]:
                worst = r["class"]
        if worst != "different":
            return {"class": worst, "tags": sorted(tags), "max_rel": max(maxrel) if maxrel else 0.0,
                    "detail": f"{len(diffs)} of {len(base)} answers differ in rendering only" if diffs else ""}
        # position-wise they differ: try a reordering of the list
        used = [False] * len(other)
        ok, wtags, wmax, inner = True, set(), [], "identical"
        for x in base:
            best = None
            for j, y in enumerate(other):
                if used[j]:
                    continue
                r = compare_answers(x, y, rel, abs_)
                if r["class"] != "different" and (best is None or _SEVERITY[r["class"]] < _SEVERITY[best[1]["class"]]):
                    best = (j, r)
            if best is None:
                ok = False
                break
            used[best[0]] = True
            wtags |= set(best[1]["tags"])
            if best[1]["max_rel"]:
                wmax.append(best[1]["max_rel"])
            if _SEVERITY[best[1]["class"]] > _SEVERITY[inner]:
                inner = best[1]["class"]
        if ok:
            return {"class": "ordering_only", "tags": sorted(wtags), "max_rel": max(wmax) if wmax else 0.0,
                    "inner": inner, "detail": "same answers, different order"}
        # a single tuple answer whose elements were reordered
        if len(base) == 1:
            m = _children_match(parse_sexpr(base[0]), parse_sexpr(other[0]), rel, abs_)
            if m:
                return {"class": "ordering_only", "tags": m["tags"], "max_rel": m["max_rel"], "inner": m["class"],
                        "detail": "same elements inside one tuple answer, different order"}
        i, x, y, _ = diffs[0]
        return {"class": "different", "tags": sorted(tags), "max_rel": 0.0,
                "detail": f"answer {i}: hyperon {x[:300]!r} vs candidate {y[:300]!r}"}
    return {"class": "different", "tags": [], "max_rel": 0.0,
            "detail": f"{len(base)} answers on hyperon, {len(other)} on candidate; "
                      f"hyperon first {base[0][:200]!r}" if base else
                      f"0 answers on hyperon, {len(other)} on candidate; first {other[0][:200]!r}"}


def worst_class(classes: Iterable[str]) -> str:
    worst = "identical"
    for c in classes:
        if _SEVERITY.get(c, 5) > _SEVERITY[worst]:
            worst = c
    return worst


def matches_expected(expected: str, answers: list[str], rel: float = 1e-6, abs_: float = 1e-9) -> dict:
    """Does one of `answers` equal the `expected` text of a smoke comment, numbers within a tolerance?

    `expected` of '(empty)' (or empty) means: no answer at all.
    """
    exp = (expected or "").strip()
    if exp in ("", "(empty)", "empty"):
        return {"ok": not answers, "why": "expected no answer" + ("" if not answers else f", got {len(answers)}")}
    ta = parse_sexpr(exp)
    for ans in answers:
        tags: set = set()
        m: list = []
        c = _tree_class(ta, parse_sexpr(ans), rel, abs_, tags, m)
        if c != "different":
            return {"ok": True, "why": f"{c}" + (f" (max rel diff {max(m):.2e})" if m else ""), "answer": ans}
    return {"ok": False, "why": "no answer matches" if answers else "no answer", "answers": answers[:3]}


# ── a position-preserving scanner for .metta SOURCE text ─────────────────────

def top_level_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level parenthesised form (a leading '!' is part of the span).

    String-aware (a ';' or a parenthesis inside "..." is text) and comment-aware (';' to end of line).
    Text between forms (comments, blank lines) belongs to no span.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    in_str = escaped = False
    start = -1
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
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
                i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "!" and depth == 0 and i + 1 < n and text[i + 1] == "(":
            start = i
        elif ch == "(":
            if depth == 0 and start < 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
            elif depth < 0:
                depth = 0
        i += 1
    return spans


def split_code_comment(line: str) -> tuple[str, str]:
    """Split a source line at the first ';' that is not inside a string."""
    in_str = escaped = False
    for i, ch in enumerate(line):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == ";":
            return line[:i], line[i:]
    return line, ""


def child_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Spans of the direct children of the form text[start:end] (atoms and sub-forms), string- and comment-aware."""
    spans: list[tuple[int, int]] = []
    i = start + 1
    stop = end - 1
    while i < stop:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
        elif ch == ";":
            while i < stop and text[i] != "\n":
                i += 1
        elif ch == "(":
            depth, j, in_str, esc = 0, i, False, False
            while j < stop:
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == ";":
                    while j < stop and text[j] != "\n":
                        j += 1
                    continue
                elif c == '"':
                    in_str = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            spans.append((i, j))
            i = j
        elif ch == '"':
            j = i + 1
            esc = False
            while j < stop:
                c = text[j]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    j += 1
                    break
                j += 1
            spans.append((i, j))
            i = j
        else:
            j = i
            while j < stop and text[j] not in " \t\r\n()\";":
                j += 1
            spans.append((i, j))
            i = j
    return spans


def kb_equations(kb_text: str) -> dict[tuple[str, int], int]:
    """{(function name, PeTTa arity): number of (= (name args…) body) equations} of a KB text.

    PeTTa registers a function with arity = number of arguments + 1 (the output slot):
    filereader.pl parse_form. Heads that are not a plain symbol (variables, nested expressions) are skipped.
    """
    out: dict[tuple[str, int], int] = {}
    for s, e in top_level_spans(kb_text):
        if kb_text[s] == "!":
            continue
        kids = child_spans(kb_text, s, e)
        if len(kids) < 3 or kb_text[kids[0][0]:kids[0][1]] != "=":
            continue
        hs, he = kids[1]
        if kb_text[hs] != "(":
            continue                                  # (= name body): not a function head in PeTTa's sense
        hk = child_spans(kb_text, hs, he)
        if not hk:
            continue
        name = kb_text[hk[0][0]:hk[0][1]]
        if name.startswith("(") or name.startswith("$") or name.startswith('"'):
            continue
        key = (name, len(hk))                         # len(hk) = 1 name + n args  ->  arity n + 1
        out[key] = out.get(key, 0) + 1
    return out


def quote_prolog_atom(name: str) -> str:
    """'name' quoted for embedding in Prolog source text."""
    return "'" + name.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def short(text: Any, n: int = 300) -> str:
    s = str(text)
    return s if len(s) <= n else s[: n - 3] + "..."
