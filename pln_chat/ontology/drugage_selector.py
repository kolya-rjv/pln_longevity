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
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from ontology.compound_names import CompoundResolver

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
    # Appended LAST, with a default, because tests/test_drugage_calibration.py
    # constructs DrugAgeRow positionally.
    sex: Optional[str] = None     # Male | Female | Mixed | Hermaphrodite | Unknown | Pooled

    @property
    def scorable(self) -> bool:
        """True when the calibration layer can actually lift this row.

        `drugage-effect` (drugage_calibration.metta §7) matches
        `(, (UsesIntervention …) (AvgLifespanChangePercent …))`, so a row with no
        reported average-lifespan change produces NO Effect link and therefore no
        score — silently. A row reporting exactly 0.0 IS scorable (that is the
        ITP-negative story: full confidence, ~zero strength).
        """
        return self.avg_change is not None

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
_RE_SEX = re.compile(r"\(HasSex\s+\S+\s+(\S+?)\)")


def _parse_block(block: str) -> Optional[DrugAgeRow]:
    m_id = _RE_ROWID.search(block)
    m_c = _RE_COMPOUND.search(block)
    if not (m_id and m_c):
        return None
    m_change = _RE_CHANGE.search(block)
    m_sig = _RE_SIG.search(block)
    m_sp = _RE_SPECIES.search(block)
    m_sex = _RE_SEX.search(block)
    return DrugAgeRow(
        row_id=m_id.group(1),
        compound=m_c.group(1),
        species=m_sp.group(1) if m_sp else None,
        is_itp=bool(_RE_ITP.search(block)),
        significance=m_sig.group(1) if m_sig else None,
        avg_change=float(m_change.group(1)) if m_change else None,
        block=block.strip(),
        sex=m_sex.group(1) if m_sex else None,
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
    """Loose compound/species matching: case-insensitive, drop separators.

    Kept as the ROW-side key. A caller-typed name goes through
    `ontology.compound_names.CompoundResolver` first (synonyms, abbreviations,
    Greek letters, ETL symbol artefacts); this function only has to match the
    canonical DrugAge string that comes back. `canonical_key` is a strict
    superset of this normalisation, so the two agree on every name `_norm`
    already matched.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ── Compound-name vocabulary (for the shared resolver) ───────────────────────
# Parsing the 1,043-symbol vocabulary out of the 1.8 MB build takes ~30 ms, so
# it is cached per (path, mtime): regenerating the ETL invalidates the cache
# without a restart, and a stable file costs one parse per process.
_VOCAB_CACHE: dict[tuple[str, float, int], list[str]] = {}
_RESOLVER_CACHE: dict[tuple[str, float, int], "CompoundResolver"] = {}


def _source_stamp(path: Path) -> tuple[str, float, int]:
    try:
        st = path.stat()
        return (str(path), st.st_mtime, st.st_size)
    except OSError:
        return (str(path), 0.0, 0)


def _resolve_source(source: Optional[Path]) -> Path:
    """The file to read rows from: the caller's, else build/, else the sample.

    A caller-supplied path that does not exist falls back the same way rather
    than raising, so name RESOLUTION still works on a checkout where the ETL has
    not been run — the ranking itself still reports the missing build.
    """
    if source is not None and source.exists():
        return source
    return BUILD_DRUGAGE if BUILD_DRUGAGE.exists() else SAMPLE_DRUGAGE


def load_vocabulary(source: Optional[Path] = None) -> list[str]:
    """Every distinct DrugAge intervention symbol in `source`, sorted."""
    path = _resolve_source(source)
    stamp = _source_stamp(path)
    cached = _VOCAB_CACHE.get(stamp)
    if cached is None:
        cached = sorted({r.compound for r in load_rows(path)})
        _VOCAB_CACHE[stamp] = cached
    return cached


def build_resolver(source: Optional[Path] = None) -> "CompoundResolver":
    """A `CompoundResolver` over `source`'s vocabulary (cached per file stamp)."""
    from ontology.compound_names import CompoundResolver

    path = _resolve_source(source)
    stamp = _source_stamp(path)
    cached = _RESOLVER_CACHE.get(stamp)
    if cached is None:
        cached = CompoundResolver(load_vocabulary(path))
        _RESOLVER_CACHE[stamp] = cached
    return cached


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


#: The documented representative-row policy, in the order the rules apply. Kept
#: as data so `/drugage/rank` can return it verbatim — the 2026-09-18 evaluation
#: could not tell WHY astaxanthin came back as "+3%, not significant" when the
#: cited paper headlines +12% (p = 0.003), because the policy was undocumented.
REPRESENTATIVE_POLICY: tuple[str, ...] = (
    "1. Only rows with a reported AvgLifespanChangePercent are eligible — a row "
    "without one cannot be lifted into an Effect link and would score empty.",
    "2. Highest evidence tier wins: an ITP row (replicated NIA mouse program) "
    "beats any single-lab row.",
    "3. Within a tier, a reported-significant result beats an unreported one, "
    "which beats a reported null.",
    "4. Remaining ties are broken by the MEDIAN average-lifespan change (the "
    "lower median on even counts), never by the maximum — no cherry-picking.",
    "5. If every row for a compound is ineligible under rule 1, the compound is "
    "reported as unscorable rather than silently omitted.",
)

#: Preference among significance labels WITHIN one evidence tier (higher wins).
#: For a non-ITP row the significance also caps confidence (`sig-gate`), but for
#: an ITP row the gate is neutral — ITP folds significance into the TIER
#: (ITP_Positive / ITP_Negative, both confidence 0.90). So without this rule the
#: two ITP astaxanthin rows tie and the median tie-break picks the
#: NotSignificant +3% female row over the Significant +12% male one.
_SIGNIFICANCE_RANK = {"Significant": 2, "Unreported": 1, "NotSignificant": 0}


def _collapse_best(rows: list[DrugAgeRow]) -> list[DrugAgeRow]:
    """One representative row per compound, per REPRESENTATIVE_POLICY."""
    by_compound: dict[str, list[DrugAgeRow]] = {}
    for r in rows:
        by_compound.setdefault(r.compound, []).append(r)

    picked: list[DrugAgeRow] = []
    for group in by_compound.values():
        # Rule 1: a scoreless row can never produce a ranking entry, so it must
        # never be the representative when a scorable row exists.
        eligible = [r for r in group if r.scorable] or group
        # Rules 2-3.
        top = max(
            (r.evidence_rank, _SIGNIFICANCE_RANK.get(r.significance or "Unreported", 1))
            for r in eligible
        )
        tier = [
            r for r in eligible
            if (r.evidence_rank,
                _SIGNIFICANCE_RANK.get(r.significance or "Unreported", 1)) == top
        ]
        # Rule 4: median-by-change representative (lower median on even counts).
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
