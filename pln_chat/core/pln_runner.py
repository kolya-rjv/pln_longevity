"""Execute MeTTa queries against the PLN knowledge base.

Two modes:
  stub    — returns pattern-matched mock results; no runtime dependency.
  runtime — delegates to the Hyperon MeTTa interpreter (requires `hyperon`
            package and knowledge base files to be loaded).

To enable runtime mode, set PLN_RUNTIME_AVAILABLE=true in your .env and
ensure `hyperon` is installed.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import PLN_RUNTIME_AVAILABLE


@dataclass
class PLNAtomResult:
    atom: str
    stv: Optional[dict] = None   # {"strength": float, "confidence": float}


@dataclass
class PLNRunResult:
    status: str                              # "ok" | "empty" | "error"
    results: list[PLNAtomResult] = field(default_factory=list)
    query_time_ms: int = 0
    mode: str = "stub"                       # "stub" | "runtime"
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


# ── Stub mode ──────────────────────────────────────────────────────────────────

_STUB_DATA: list[PLNAtomResult] = [
    PLNAtomResult("RCT_Human",                {"strength": 1.00, "confidence": 1.00}),
    PLNAtomResult("ITP_Positive",             {"strength": 0.90, "confidence": 0.90}),
    PLNAtomResult("ITP_Negative",             {"strength": 0.90, "confidence": 0.90}),
    PLNAtomResult("MultipleHumanTrials",      {"strength": 0.85, "confidence": 0.85}),
    PLNAtomResult("SingleHumanTrial",         {"strength": 0.70, "confidence": 0.70}),
    PLNAtomResult("AnimalStudies_Replicated", {"strength": 0.65, "confidence": 0.65}),
    PLNAtomResult("Epidemiological",          {"strength": 0.60, "confidence": 0.60}),
    PLNAtomResult("AnimalStudies_Single",     {"strength": 0.50, "confidence": 0.50}),
    PLNAtomResult("Preprint",                 {"strength": 0.40, "confidence": 0.40}),
    PLNAtomResult("InVitro",                  {"strength": 0.35, "confidence": 0.35}),
    PLNAtomResult("TraditionalUse",           {"strength": 0.20, "confidence": 0.20}),
]

_STUB_DRUGS: list[PLNAtomResult] = [
    PLNAtomResult("Rapamycin",   {"strength": 0.90, "confidence": 0.90}),
    PLNAtomResult("Metformin",   {"strength": 0.75, "confidence": 0.70}),
    PLNAtomResult("Resveratrol", {"strength": 0.55, "confidence": 0.50}),
    PLNAtomResult("Acarbose",    {"strength": 0.65, "confidence": 0.65}),
]


def _apply_threshold(results: list[PLNAtomResult], threshold: float) -> list[PLNAtomResult]:
    if threshold <= 0:
        return results
    return [r for r in results if r.stv is None or r.stv.get("confidence", 1.0) >= threshold]


def _stub_run(metta_query: str, confidence_threshold: float) -> PLNRunResult:
    """Return plausible mock results based on simple keyword matching."""
    start = time.monotonic()
    time.sleep(0.05)   # simulate slight latency
    q = metta_query.lower()

    if "evidence-confidence" in q:
        # Try to match a specific constant first
        for atom in _STUB_DATA:
            if atom.atom.lower() in q:
                results = [PLNAtomResult(str(atom.stv["confidence"] if atom.stv else "?"))]
                break
        else:
            results = list(_STUB_DATA)
    elif "apply-tv" in q:
        # Extract strength from query text
        m = re.search(r"apply-tv\s+([\d.]+)", metta_query)
        strength = float(m.group(1)) if m else 0.8
        # Find evidence type
        evidence = next(
            (a.atom for a in _STUB_DATA if a.atom in metta_query),
            "AnimalStudies_Replicated",
        )
        conf = next((a.stv["confidence"] for a in _STUB_DATA if a.atom == evidence), 0.65)
        results = [PLNAtomResult(f"(stv {strength:.2f} {conf:.2f})")]
    elif any(kw in q for kw in ("lifespan", "lifespanextender", "extends-lifespan")):
        results = list(_STUB_DRUGS)
    else:
        results = [PLNAtomResult("(stub-result)", {"strength": 0.50, "confidence": 0.30})]

    results = _apply_threshold(results, confidence_threshold)
    elapsed = int((time.monotonic() - start) * 1000)
    return PLNRunResult(
        status="ok" if results else "empty",
        results=results,
        query_time_ms=elapsed,
        mode="stub",
    )


# ── Runtime mode (Hyperon) ─────────────────────────────────────────────────────

def _stv_from_atom(atom_str: str) -> Optional[dict]:
    m = re.search(r"\(stv\s+([\d.]+)\s+([\d.]+)\)", atom_str)
    if m:
        return {"strength": float(m.group(1)), "confidence": float(m.group(2))}
    return None


def _iter_top_level_exprs(text: str):
    """Yield each top-level S-expression from a MeTTa text block."""
    buf: list[str] = []
    depth = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Strip comment-only lines
        if line.startswith(";;") or line.startswith(";"):
            continue
        for ch in line:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        buf.append(line)
        if depth <= 0 and buf:
            expr = " ".join(buf).strip()
            if expr:
                yield expr
            buf = []
            depth = 0
    if buf:
        expr = " ".join(buf).strip()
        if expr:
            yield expr


def _normalize_query(metta_query: str) -> str:
    """Prefix each top-level expression with ! so MeTTa evaluates it."""
    parts: list[str] = []
    for expr in _iter_top_level_exprs(metta_query):
        parts.append(expr if expr.startswith("!") else "!" + expr)
    return "\n".join(parts)


def _hyperon_run(
    metta_query: str,
    confidence_threshold: float,
    kb_files: Optional[list[Path]],
) -> PLNRunResult:
    """Execute query using the real Hyperon MeTTa interpreter.

    Mirrors the standalone script pattern exactly:

        !(import! &self <stem>)   ; one per KB file
        !(match &self (, ...) $template)

    Parameters
    ----------
    metta_query:
        One or more top-level MeTTa expressions (``!`` prefix optional —
        added automatically if absent).
    confidence_threshold:
        Filter out results whose STV confidence is below this value.
    kb_files:
        Ordered list of .metta knowledge-base files to load before the query.
        The parent directory of the first file is used as the working directory
        so that `import!` can resolve module names by stem.
    """
    import os

    try:
        from hyperon import MeTTa  # type: ignore

        normalized = _normalize_query(metta_query)
        if not normalized:
            return PLNRunResult(status="empty", mode="runtime")

        # Build a single script: one !(import! &self <stem>) per KB file,
        # followed by the query — exactly like a standalone .metta script.
        import_lines: list[str] = []
        kb_dir: Optional[Path] = None
        for path in (kb_files or []):
            import_lines.append(f"!(import! &self {path.stem})")
            if kb_dir is None:
                kb_dir = path.parent

        full_script = "\n".join(import_lines + [normalized])

        # Log the full script for debugging (overwrite on each query)
        try:
            from config import LOGS_DIR
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            (LOGS_DIR / "last_query.metta").write_text(full_script, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass  # logging must never break query execution

        # Run from the KB directory so import! resolves module names correctly.
        original_dir = os.getcwd()
        if kb_dir is not None:
            os.chdir(kb_dir)
        try:
            metta = MeTTa()
            start = time.monotonic()
            # ─── DEBUG ───
            print("=" * 60)
            print("FULL SCRIPT SENT TO METTA:")
            print(full_script)
            print("=" * 60)
            # ─── END DEBUG ───
            raw: list[list] = metta.run(full_script)
            elapsed = int((time.monotonic() - start) * 1000)
        except Exception as run_exc:
            return PLNRunResult(status="error", mode="runtime", error=str(run_exc))
        finally:
            os.chdir(original_dir)

        # raw is a list of result-lists, one per ! expression — skip import
        # results (first len(import_lines) groups) and only process query results.
        query_raw = raw[len(import_lines):] if len(raw) > len(import_lines) else raw
        results: list[PLNAtomResult] = []
        for result_group in query_raw:
            for atom in result_group:
                atom_str = str(atom)
                results.append(PLNAtomResult(atom=atom_str, stv=_stv_from_atom(atom_str)))

        results = _apply_threshold(results, confidence_threshold)
        return PLNRunResult(
            status="ok" if results else "empty",
            results=results,
            query_time_ms=elapsed,
            mode="runtime",
        )
    except Exception as exc:   # noqa: BLE001
        return PLNRunResult(status="error", mode="runtime", error=str(exc))


# ── Public API ─────────────────────────────────────────────────────────────────

def run_query(
    metta_query: str,
    confidence_threshold: float = 0.0,
    kb_files: Optional[list[Path]] = None,
) -> PLNRunResult:
    """Execute a MeTTa query, using stub or runtime mode as configured.

    Parameters
    ----------
    metta_query:
        MeTTa expression(s) to execute.
    confidence_threshold:
        Minimum STV confidence for results to be included.
    kb_files:
        .metta KB files to load into the Hyperon space (runtime mode only).
    """
    if not metta_query.strip():
        return PLNRunResult(status="empty", mode="stub" if not PLN_RUNTIME_AVAILABLE else "runtime")
    if PLN_RUNTIME_AVAILABLE:
        return _hyperon_run(metta_query, confidence_threshold, kb_files)
    return _stub_run(metta_query, confidence_threshold)
