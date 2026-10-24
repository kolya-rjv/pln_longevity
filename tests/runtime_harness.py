"""A runtime-neutral MeTTa interpreter for the tests that drive the engine directly.

Why this module exists
----------------------
Fifteen test modules used to build a `hyperon.MeTTa()` themselves, load a list of `.metta` files into it and
call `.run("!(...)")`. They therefore tested hyperon whatever `PLN_RUNTIME` said. This module is the one place
that decides which engine such a test talks to:

    kb = new_kb(KB_FILES)                       # the .metta files, joined with a newline, loaded once
    kb = new_kb(KB_FILES, extra_text=SYNTH)     # ... plus facts appended after them
    kb.run("!(f 1)") -> [[x, ...], ...]         # one group per `!` form; str(x) is what hyperon prints
    kb.close()                                  # releases the interpreter (a PeTTa child process)

Under `PLN_RUNTIME=hyperon` (the default) `new_kb` returns a thin wrapper around a REAL `hyperon.MeTTa()`:
the same object a test built before, the same loading calls, and `run` returns the real atoms (so
`repr`, `==` and the atom API of the hyperon objects work unchanged; `kb.metta` is the underlying
interpreter). Nothing is stringified, nothing is reordered, no call is added.

Under `PLN_RUNTIME=petta` it opens `core.metta_runtime.PettaRuntime.session(...)`: one forked PeTTa child that
holds the KB (PeTTa's space is process-global, so one KB load per process), and `run` returns the session's
result strings, after the adapter's own normalisation (`true` -> `True`). Each string is wrapped in `AtomText`,
a `str` whose `repr` is the text itself, so a whole result printed with `str(kb.run(...))` or an f-string
reads as hyperon's does (`[[(stv 0.5 0.9)]]`, not `[['(stv 0.5 0.9)']]`).

What the two runtimes still do differently is NOT hidden here (that would be an assertion loosened for
both): integers print as `3` on PeTTa where hyperon prints `3.0`, the members of a `collapse`/`match`
answer come back in assertion order where hyperon often reverses them, and a call to a known function with
no matching clause returns nothing on PeTTa where hyperon returns the call unreduced. A test that asserts
one of those is either gated (`on_petta()`), compared canonically (tests/runtime_compare.py) or marked
`hyperon_only`. The notes of the harness stage list which is which.

Availability
------------
`require_runtime()` replaces `pytest.importorskip("hyperon")` at the top of a converted module. Under hyperon
it IS that call, so the module is skipped exactly when it used to be; under PeTTa it skips the module when
PeTTa cannot run here (no janus_swi, no checkout), not when hyperon is missing.

This module imports neither hyperon nor janus at import time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:               # conftest does this too; a child process started by a test needs it
    sys.path.insert(0, str(PLN_CHAT))

__all__ = [
    "AtomText", "Kb", "new_kb", "load_text", "runtime_name", "on_petta", "on_hyperon", "require_runtime",
    "runtime_available", "hyperon_only", "HARNESS_DEADLINE_S",
]

#: A hang must fail a test, not the suite, and a loaded machine (other agents run suites at the same time)
#: must not turn a slow query into a failure: far above the adapter's 60 s default, far below "forever".
HARNESS_DEADLINE_S = float(os.getenv("PLN_TEST_DEADLINE_S", "300"))

#: `@hyperon_only` marks a test that asserts something only hyperon does; conftest.py skips it under PeTTa.
hyperon_only = pytest.mark.hyperon_only

Source = Union[str, "os.PathLike[str]"]


# ── which runtime ─────────────────────────────────────────────────────────────

def runtime_name() -> str:
    """'hyperon' or 'petta': what PLN_RUNTIME selects (`auto` resolved by the adapter)."""
    from core.metta_runtime import selected_runtime_name
    return selected_runtime_name()


def on_petta() -> bool:
    return runtime_name() == "petta"


def on_hyperon() -> bool:
    return runtime_name() == "hyperon"


def runtime_available() -> bool:
    """Can the selected runtime run here? Imports neither engine."""
    from core.metta_runtime import runtime_available_for
    return runtime_available_for(runtime_name())


def require_runtime(reason: Optional[str] = None) -> str:
    """Skip (the module, when called at module level; the test, inside one) unless the selected runtime
    can run. Returns the runtime's name.

    Under hyperon this is `pytest.importorskip("hyperon")`, verbatim, so a converted module is skipped under
    exactly the conditions it was before. Under PeTTa it checks what the adapter checks (janus_swi and the
    pinned checkout), without importing either.
    """
    name = runtime_name()
    if name == "hyperon":
        if reason is None:
            pytest.importorskip("hyperon")
        else:
            pytest.importorskip("hyperon", reason=reason)
    elif not runtime_available():
        pytest.skip(reason or "the selected PeTTa runtime is not installed here", allow_module_level=True)
    return name


# ── results ───────────────────────────────────────────────────────────────────

class AtomText(str):
    """A result atom as PeTTa returns it: its text, printing as itself.

    `str(atom)` is the text (what every test reads); `repr(atom)` is the text too, as hyperon's
    `Atom.__repr__` is, so `str(result)` of a list of groups prints as hyperon's does.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return str.__str__(self)


# ── the interpreter ───────────────────────────────────────────────────────────

def _read(source: Source) -> str:
    path = Path(source)
    if not path.is_absolute():
        path = REPO / path
    return path.read_text(encoding="utf-8")


def load_text(files: Iterable[Source]) -> str:
    """The files' text joined with a newline, as the tests always joined it. Relative names are the repo's."""
    return "\n".join(_read(f) for f in files)


class Kb:
    """A loaded knowledge base. `run(text)` is `MeTTa.run`; `close()` releases it. Use `new_kb`."""

    #: 'hyperon' or 'petta'
    runtime: str

    def run(self, text: str, deadline_s: Optional[float] = None) -> list[list[Any]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def metta(self) -> Any:
        """The underlying `hyperon.MeTTa` (hyperon only)."""
        raise AttributeError(f"a {self.runtime} knowledge base has no hyperon MeTTa object")

    def __enter__(self) -> "Kb":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class _HyperonKb(Kb):
    """A real `hyperon.MeTTa()`: loaded with the same `run` calls, answering with the real atoms."""

    runtime = "hyperon"

    def __init__(self, parts: Sequence[str]) -> None:
        from hyperon import MeTTa
        self._metta = MeTTa()
        for part in parts:
            self._metta.run(part)

    @property
    def metta(self) -> Any:
        return self._metta

    def run(self, text: str, deadline_s: Optional[float] = None) -> list[list[Any]]:
        # `deadline_s` is accepted for symmetry and ignored: a Rust call holding the GIL cannot be preempted.
        return self._metta.run(text)

    def close(self) -> None:
        self._metta = None


class _PettaKb(Kb):
    """One PeTTa child process holding the KB (the adapter's `PettaRuntime.session`)."""

    runtime = "petta"

    def __init__(self, kb_text: str, deadline_s: Optional[float]) -> None:
        from core.metta_runtime import get_runtime
        self._deadline_s = deadline_s
        self._session = get_runtime("petta").session(kb_text, deadline_s)

    def run(self, text: str, deadline_s: Optional[float] = None) -> list[list[Any]]:
        groups = self._session.run(text, self._deadline_s if deadline_s is None else deadline_s)
        return [[AtomText(s) for s in group] for group in groups]

    @property
    def last_raw(self) -> list[list[str]]:
        """The strings of the last `run` before the adapter normalised them."""
        return self._session.last_raw

    def close(self) -> None:
        self._session.close()


def new_kb(files: Iterable[Source] = (), extra_text: str = "", *, separate: bool = False,
           deadline_s: Optional[float] = None) -> Kb:
    """An interpreter with `files` loaded, then `extra_text`, on the selected runtime.

    `files` are paths, or names relative to the repo root; their text is joined with a newline and loaded
    in one `run` (what `m.run("\\n".join(texts))` was), followed by `extra_text` appended after a newline.
    `separate=True` is for a test that loaded each file with its own `run` call and then ran `extra_text` in
    one more: hyperon does exactly that; PeTTa, which loads one KB per process, gets the same text joined.
    `deadline_s` bounds a PeTTa start and each of its `run`s (default `HARNESS_DEADLINE_S`; 0 for none).
    """
    texts = [_read(f) for f in files]
    if extra_text:
        texts.append(extra_text)
    name = runtime_name()
    if name == "hyperon":
        parts = texts if separate else (["\n".join(texts)] if texts else [])
        return _HyperonKb(parts)
    return _PettaKb("\n".join(texts), HARNESS_DEADLINE_S if deadline_s is None else deadline_s)
