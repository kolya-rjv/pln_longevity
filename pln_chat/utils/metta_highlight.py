"""Lightweight syntax highlighter for MeTTa source, rendered as HTML.

Gradio's gr.Code component has no MeTTa (or Lisp/Scheme) language mode, so
raw .metta text can't get accurate highlighting there. This module tokenizes
MeTTa's S-expression syntax directly and renders it to HTML `<span>`s with
`mh-*` classes, meant for display in a `gr.HTML` component.
"""
from __future__ import annotations

import html
import re

# One alternative per token kind; matched in order, so `number` is tried
# before the catch-all `symbol`. The whole string is covered (including
# whitespace), so there are no un-classified gaps to stitch back together.
_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<comment>;[^\n]*)
    | (?P<string>"(?:[^"\\]|\\.)*"?)
    | (?P<lparen>\()
    | (?P<rparen>\))
    | (?P<variable>\$[^\s()]*)
    | (?P<number>-?\d+\.?\d*(?=[\s()]|$))
    | (?P<symbol>[^\s()]+)
    """,
    re.VERBOSE,
)


def highlight_metta(text: str) -> str:
    """Render MeTTa source as HTML with span-based syntax highlighting.

    The symbol immediately after an opening paren (the "head" of an
    S-expression, e.g. `Inheritance` in `(Inheritance Foo Bar)`) is styled
    distinctly, mirroring how Lisp-aware editors highlight operator position
    without needing a maintained keyword list.
    """
    out: list[str] = []
    prev_kind: str | None = None
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        chunk = html.escape(m.group())
        if kind == "ws":
            out.append(chunk)
        elif kind in ("lparen", "rparen"):
            out.append(f'<span class="mh-paren">{chunk}</span>')
        elif kind == "symbol":
            cls = "mh-head" if prev_kind == "lparen" else "mh-symbol"
            out.append(f'<span class="{cls}">{chunk}</span>')
        else:
            out.append(f'<span class="mh-{kind}">{chunk}</span>')
        prev_kind = kind
    return "".join(out)
