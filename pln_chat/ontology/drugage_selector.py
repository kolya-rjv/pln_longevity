"""Select a SMALL, query-relevant slice of DrugAge rows for scoped inference.

Why this exists
---------------
The DrugAge ETL emits ~3,423 rows (~46k atoms). hyperon 0.2.10 aborts with a
non-unwinding Rust panic once ONE space exceeds ~a few thousand atoms (measured
boundary: ~420 DrugAge rows loaded alone already panics on the next query). So
the inference engine can never see the whole dump. This module is the Python
half of the fix: given the compounds/species a query cares about, it pulls only
the matching row blocks out of the generated `build/drugage_etl.metta`, capped
well under the panic threshold, so `pln_runner` can inject them into a
query-scoped space alongside the inference stack.

It never rewrites a row — it copies verbatim row blocks, so the MeTTa
calibration layer (`drugage_calibration.metta`) does all the lifting on the fly.

See docs/etl_inference_wiring.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Repo root is two levels up from this file (pln_chat/ontology/ -> repo).
_REPO = Path(__file__).resolve().parent.parent.parent

# Preferred source is the full regenerated ETL (build/), falling back to the
# committed 201-row sample if the ETL has not been run. NB: the sample has NO
# ITP rows (they start ~row 908), so ITP demos need the build/ file.
BUILD_DRUGAGE = _REPO / "build" / "drugage_etl.metta"
SAMPLE_DRUGAGE = _REPO / "drugage_etl_short.metta"

# Hard cap on injected rows. The panic boundary is ~420 rows loaded alone; the
# inference stack adds ~1.5k atoms, so we stay well below with 150 rows (~1.8k
# atoms). Real queries inject far fewer (one compound = a handful of rows).
MAX_ROWS = 150

# Evidence rank for picking one representative row per compound (higher = better
# evidence). ITP (gold-standard replicated mouse program) beats any single-lab
# result; among non-ITP, a reported-significant result beats an unreported one,
# which beats a reported NULL. Mirrors the MeTTa confidence tiers.
_EVIDENCE_RANK = {
    ("itp",): 3,
    ("Significant",): 2,
    ("Unreported",): 1,
    ("NotSignificant",): 0,
}


@dataclass
class DrugAgeRow:
    """One parsed DrugAge row block + the metadata we filter/rank on."""
    row_id: str
    compound: str
    species: Optional[str]
    is_itp: bool
    significance: Optional[str]   # Significant | NotSignificant | Unreported | None
    avg_change: Optional[float]
    block: str                    # the verbatim MeTTa text for this row

    @property
    def evidence_rank(self) -> int:
        if self.is_itp:
            return _EVIDENCE_RANK[("itp",)]
        return _EVIDENCE_RANK.get((self.significance or "Unreported",), 1)

    @property
    def pmid(self) -> Optional[str]:
        """Provenance token (e.g. 'PMID_24341993') parsed from the row block, if any.

        The raw row is never rewritten, so provenance is read back from the
        verbatim `(ReportedIn <row> PMID_…)` atom — the audit trail the chat app
        surfaces alongside a ranked compound (docs/etl_inference_wiring.md §5).
        """
        m = _RE_PMID.search(self.block)
        return m.group(1) if m else None


_RE_ROWID = re.compile(r"\(InstanceOf\s+(\S+)\s+Experiment\)")
_RE_COMPOUND = re.compile(r"\(UsesIntervention\s+\S+\s+(\S+?)\)")
_RE_SPECIES = re.compile(r"\(UsesSpecies\s+\S+\s+(\S+?)\)")
_RE_ITP = re.compile(r"\(IsITPStudy\s+\S+\)")
_RE_SIG = re.compile(r"\(AvgLifespanSignificance\s+\S+\s+(\S+?)\)")
_RE_CHANGE = re.compile(r"\(AvgLifespanChangePercent\s+\S+\s+([-\d.eE]+)\)")
_RE_PMID = re.compile(r"\(ReportedIn\s+\S+\s+(\S+?)\)")


def _parse_block(block: str) -> Optional[DrugAgeRow]:
    m_id = _RE_ROWID.search(block)
    m_c = _RE_COMPOUND.search(block)
    if not (m_id and m_c):
        return None
    m_change = _RE_CHANGE.search(block)
    m_sig = _RE_SIG.search(block)
    m_sp = _RE_SPECIES.search(block)
    return DrugAgeRow(
        row_id=m_id.group(1),
        compound=m_c.group(1),
        species=m_sp.group(1) if m_sp else None,
        is_itp=bool(_RE_ITP.search(block)),
        significance=m_sig.group(1) if m_sig else None,
        avg_change=float(m_change.group(1)) if m_change else None,
        block=block.strip(),
    )


def load_rows(source: Optional[Path] = None) -> list[DrugAgeRow]:
    """Parse every row block out of the DrugAge ETL file into DrugAgeRow objects."""
    path = source or (BUILD_DRUGAGE if BUILD_DRUGAGE.exists() else SAMPLE_DRUGAGE)
    text = path.read_text(encoding="utf-8")
    # Split at each row's opening (InstanceOf <id> Experiment) atom. Robust to
    # both the ETL's "; row N"-delimited output and a bare committed fixture.
    parts = re.split(r"(?=^\(InstanceOf\s+\S+\s+Experiment\))", text, flags=re.MULTILINE)
    rows: list[DrugAgeRow] = []
    for part in parts:
        row = _parse_block(part)
        if row is not None:
            rows.append(row)
    return rows


def _norm(name: str) -> str:
    """Loose compound/species matching: case-insensitive, drop separators."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def select_rows(
    rows: Iterable[DrugAgeRow],
    compounds: Optional[Iterable[str]] = None,
    species: Optional[Iterable[str]] = None,
    itp_only: bool = False,
    significant_only: bool = False,
    best_per_compound: bool = False,
    limit: int = MAX_ROWS,
) -> list[DrugAgeRow]:
    """Filter (and optionally collapse-to-best) DrugAge rows, capped at `limit`.

    Parameters
    ----------
    compounds / species:
        Keep only rows whose compound / species matches (loose, normalised).
    itp_only:
        Keep only ITP rows.
    significant_only:
        Keep only rows with a Significant average-lifespan result.
    best_per_compound:
        Collapse to ONE representative row per compound — the highest evidence
        tier, tie-broken by the MEDIAN average-lifespan change (a deliberately
        non-cherry-picked representative; full multi-row PLN revision is the
        documented follow-up). Yields a clean one-entry-per-compound ranking.
    limit:
        Hard cap on returned rows (panic safety). Truncates deterministically.
    """
    comp_set = {_norm(c) for c in compounds} if compounds else None
    sp_set = {_norm(s) for s in species} if species else None

    kept: list[DrugAgeRow] = []
    for r in rows:
        if comp_set is not None and _norm(r.compound) not in comp_set:
            continue
        if sp_set is not None and (r.species is None or _norm(r.species) not in sp_set):
            continue
        if itp_only and not r.is_itp:
            continue
        if significant_only and r.significance != "Significant":
            continue
        kept.append(r)

    if best_per_compound:
        kept = _collapse_best(kept)

    # Deterministic order (by compound then row id) before the cap.
    kept.sort(key=lambda r: (r.compound, r.row_id))
    return kept[:limit]


def _collapse_best(rows: list[DrugAgeRow]) -> list[DrugAgeRow]:
    """One representative row per compound: best evidence tier, median change."""
    by_compound: dict[str, list[DrugAgeRow]] = {}
    for r in rows:
        by_compound.setdefault(r.compound, []).append(r)

    picked: list[DrugAgeRow] = []
    for group in by_compound.values():
        top = max(r.evidence_rank for r in group)
        tier = [r for r in group if r.evidence_rank == top]
        # Median-by-change representative (lower median on even counts); rows
        # with no reported change sort as 0 so they don't dominate.
        tier.sort(key=lambda r: (r.avg_change if r.avg_change is not None else 0.0, r.row_id))
        picked.append(tier[(len(tier) - 1) // 2])
    return picked


def slice_metta(rows: Iterable[DrugAgeRow]) -> str:
    """Concatenate selected row blocks into one MeTTa text slice."""
    return "\n\n".join(r.block for r in rows)


def build_drugage_slice(
    compounds: Optional[Iterable[str]] = None,
    species: Optional[Iterable[str]] = None,
    *,
    itp_only: bool = False,
    significant_only: bool = False,
    best_per_compound: bool = False,
    limit: int = MAX_ROWS,
    source: Optional[Path] = None,
) -> tuple[str, list[DrugAgeRow]]:
    """One-shot: load, select, and render a scoped DrugAge slice.

    Returns (metta_text, selected_rows). The text is safe to concatenate into a
    query-scoped hyperon space; `selected_rows` lets the caller list/format the
    provenance of what was injected.
    """
    rows = load_rows(source)
    selected = select_rows(
        rows,
        compounds=compounds,
        species=species,
        itp_only=itp_only,
        significant_only=significant_only,
        best_per_compound=best_per_compound,
        limit=limit,
    )
    return slice_metta(selected), selected
