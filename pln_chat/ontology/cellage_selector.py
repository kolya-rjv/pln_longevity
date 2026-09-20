"""Select a SMALL, query-relevant slice of CellAge rows for scoped inference.

Why this exists
---------------
The exact shape of `drugage_selector`, for the same reason. The CellAge curated
ETL emits 927 row blocks (~10k atoms) into `build/cellage_genes.metta`, and
hyperon 0.2.10 ABORTS the interpreter — a non-unwinding Rust panic in
hyperon-space's trie, which `except Exception` cannot catch — on a `match` with
a variable in the row slot once one space passes a few hundred data rows. So the
engine can never see the dump, and a query that wants real inference over CellAge
has to be handed only the rows it asked about.

Rows are copied VERBATIM. `cellage_calibration.metta` does all the lifting on the
fly, so the raw measurement atoms are never rewritten (the calibration-layer
immutability invariant).

Two readers, one source
-----------------------
`ontology.gene_index` reads the CSVs and answers lookups in Python; this module
reads the generated MeTTa and feeds the engine. They agree because they come
from the same table, and `tests/test_gene_index.py` pins that agreement.

`build/` is gitignored, so `load_rows` falls back to a committed fixture of real
row blocks (`tests/fixtures/cellage_real_rows.metta`) and `build_available()`
lets a caller say plainly which one it got — a slice off the fixture is 40 genes,
not the corpus, and must never be presented as the corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Repo root is two levels up from this file (pln_chat/ontology/ -> repo).
_REPO = Path(__file__).resolve().parent.parent.parent

BUILD_CELLAGE = _REPO / "build" / "cellage_genes.metta"
#: A committed slice of real row blocks, so the inference path is exercisable on
#: a checkout where `scripts/run_etl.sh` has never been run.
SAMPLE_CELLAGE = _REPO / "tests" / "fixtures" / "cellage_real_rows.metta"

#: Hard cap on injected rows, enforced BEFORE the hyperon call because the abort
#: is uncatchable — a cap checked afterwards is not a cap.
#:
#: MEASURED on this machine (hyperon 0.2.10), running
#: `!(genes-affecting-senescence &self Increases)` — a variable-in-the-row-slot
#: match — against the CELLAGE_STACK plus a slice of that many real row blocks:
#:
#:     100 rows   ok (52 results,  2 ms)
#:     110 rows   ok (57 results)
#:     125 rows   ok (64 results)
#:     150 rows   ABORT  (trie.rs:179 `Option::unwrap()` on None, non-unwinding)
#:     200 rows   ABORT
#:
#: A CellAge block is 11-13 atoms, so the boundary sits near ~1,500 space atoms
#: — LOWER than DrugAge's (drugage_selector.MAX_ROWS is 150 of ~12 atoms) because
#: this layer's query shape leaves two variables in the pattern instead of one.
#: 100 is the cap with ~25% measured headroom. Do not raise it without re-running
#: that bisection; the failure mode is the worker process dying, not an exception.
MAX_ROWS = 100

_RE_ROWID = re.compile(r"\(InstanceOf\s+(\S+)\s+CellSenescenceRecord\)")
_RE_GENE = re.compile(r"\(InvolvesGene\s+\S+\s+(\S+?)\)")
_RE_SYMBOL = re.compile(r'\(GeneSymbol\s+\S+\s+"([^"]*)"\)')
_RE_NAME = re.compile(r'\(GeneName\s+\S+\s+"([^"]*)"\)')
_RE_ENTREZ = re.compile(r"\(EntrezID\s+\S+\s+(\d+)\)")
_RE_PMID = re.compile(r"\(ReportedIn\s+\S+\s+(\S+?)\)")
_RE_TYPE = re.compile(r"\(HasSenescenceType\s+\S+\s+(\S+?)\)")
_RE_CONTEXT = re.compile(r"\(UsesCellContext\s+\S+\s+(\S+?)\)")
_RE_LABEL = re.compile(r"\(HasSenescenceEffectLabel\s+\S+\s+(\S+?)\)")
_RE_DIRECTION = re.compile(
    r"\(HasSenescenceEffect\s+\S+\s+\((Increases|Decreases)\s+CellularSenescence\)\)")


@dataclass
class CellAgeRow:
    """One parsed CellAge row block + the metadata we filter on."""
    row_id: str
    gene_atom: str                 # e.g. Gene_TP53 — the atom the lift binds
    symbol: Optional[str]          # e.g. "TP53"
    entrez: Optional[int]
    gene_name: Optional[str]
    effect_label: Optional[str]    # Induces | Inhibits | Unclear
    direction: Optional[str]       # Increases | Decreases | None
    senescence_type: Optional[str]
    cell_context: Optional[str]
    pmid: Optional[str]            # normalised (see ontology.gene_index)
    block: str                     # the verbatim MeTTa text for this row

    @property
    def liftable(self) -> bool:
        """True when `cellage-effect` can actually produce an Effect link.

        The lift matches `(, (InvolvesGene …) (HasSenescenceEffect … ($dir
        CellularSenescence)))`, and the ETL emits `HasSenescenceEffect` only for
        an Induces/Inhibits row. An `Unclear` row therefore yields NO link — the
        source asserts no direction, so neither does the engine.
        """
        return self.direction is not None


def _parse_block(block: str) -> Optional[CellAgeRow]:
    from ontology.gene_index import normalise_pmid

    m_id = _RE_ROWID.search(block)
    m_gene = _RE_GENE.search(block)
    if not (m_id and m_gene):
        return None
    m_symbol = _RE_SYMBOL.search(block)
    m_name = _RE_NAME.search(block)
    m_entrez = _RE_ENTREZ.search(block)
    m_pmid = _RE_PMID.search(block)
    m_type = _RE_TYPE.search(block)
    m_ctx = _RE_CONTEXT.search(block)
    m_label = _RE_LABEL.search(block)
    m_dir = _RE_DIRECTION.search(block)
    return CellAgeRow(
        row_id=m_id.group(1),
        gene_atom=m_gene.group(1),
        symbol=m_symbol.group(1) if m_symbol else None,
        entrez=int(m_entrez.group(1)) if m_entrez else None,
        gene_name=m_name.group(1) if m_name else None,
        effect_label=m_label.group(1) if m_label else None,
        direction=m_dir.group(1) if m_dir else None,
        senescence_type=m_type.group(1) if m_type else None,
        cell_context=m_ctx.group(1) if m_ctx else None,
        # `PMID_PMID_26583757` in any build/ generated before the ETL fix.
        pmid=normalise_pmid(m_pmid.group(1)) if m_pmid else None,
        block=block.strip(),
    )


def build_available() -> bool:
    """True when the full CellAge build is on disk (not the committed sample)."""
    return BUILD_CELLAGE.exists()


def _resolve_source(source: Optional[Path]) -> Optional[Path]:
    if source is not None and source.exists():
        return source
    if BUILD_CELLAGE.exists():
        return BUILD_CELLAGE
    if SAMPLE_CELLAGE.exists():
        return SAMPLE_CELLAGE
    return None


def resolved_source(source: Optional[Path] = None) -> Optional[str]:
    """The repo-relative path rows would actually be read from, or None.

    The API reports this verbatim: a slice off `tests/fixtures/…` is 25 rows of a
    927-row table, and a caller has to be able to see which one they got.
    """
    path = _resolve_source(source)
    if path is None:
        return None
    try:
        return str(path.relative_to(_REPO))
    except ValueError:      # a caller-supplied path outside the repo
        return str(path)


def load_rows(source: Optional[Path] = None) -> list[CellAgeRow]:
    """Parse every row block out of the CellAge ETL file. [] when absent."""
    path = _resolve_source(source)
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    parts = re.split(
        r"(?=^\(InstanceOf\s+\S+\s+CellSenescenceRecord\))", text, flags=re.MULTILINE)
    rows: list[CellAgeRow] = []
    for part in parts:
        row = _parse_block(part)
        if row is not None:
            rows.append(row)
    return rows


def select_rows(
    rows: Iterable[CellAgeRow],
    genes: Optional[Iterable[str]] = None,
    *,
    direction: Optional[str] = None,
    senescence_type: Optional[str] = None,
    cell_context: Optional[str] = None,
    liftable_only: bool = True,
    limit: int = MAX_ROWS,
) -> list[CellAgeRow]:
    """Filter CellAge rows, capped at `limit` (never above `MAX_ROWS`).

    `genes` matches a symbol, an entrez id or the `Gene_…` atom, case-insensitively
    — whichever the caller happens to have.

    The cap is applied with `min(limit, MAX_ROWS)` rather than trusted from the
    caller: every path into the engine goes through here, and a caller that
    passes `limit=5000` must not be able to abort the worker.
    """
    wanted = {str(g).strip().upper() for g in genes if str(g).strip()} if genes else None
    kept: list[CellAgeRow] = []
    for r in rows:
        if liftable_only and not r.liftable:
            continue
        if direction and (r.direction or "") != direction:
            continue
        if senescence_type and (r.senescence_type or "") != senescence_type:
            continue
        if cell_context and (r.cell_context or "") != cell_context:
            continue
        if wanted is not None:
            keys = {r.gene_atom.upper()}
            if r.symbol:
                keys.add(r.symbol.upper())
            if r.entrez is not None:
                keys.add(str(r.entrez))
            if not (keys & wanted):
                continue
        kept.append(r)

    kept.sort(key=lambda r: (r.symbol or r.gene_atom, r.row_id))
    return kept[:max(0, min(limit, MAX_ROWS))]


def slice_metta(rows: Iterable[CellAgeRow]) -> str:
    """Concatenate selected row blocks into one MeTTa text slice."""
    return "\n\n".join(r.block for r in rows)


def build_cellage_slice(
    genes: Optional[Iterable[str]] = None,
    *,
    direction: Optional[str] = None,
    senescence_type: Optional[str] = None,
    cell_context: Optional[str] = None,
    limit: int = MAX_ROWS,
    source: Optional[Path] = None,
) -> tuple[str, list[CellAgeRow]]:
    """One-shot: load, select and render a scoped CellAge slice."""
    rows = load_rows(source)
    selected = select_rows(
        rows, genes,
        direction=direction,
        senescence_type=senescence_type,
        cell_context=cell_context,
        limit=limit,
    )
    return slice_metta(selected), selected
