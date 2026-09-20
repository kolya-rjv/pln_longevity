"""Make CellAge and GenAge ANSWERABLE, in Python, with no hyperon and no LLM.

The finding
-----------
The 2026-09-18 evaluation (§4) found the gene data present on disk and
unreachable from every direction at once:

* "Which genes drive cellular senescence?" returned four hallmark *components*
  (`TelomereAttrition`, `DNADamage`, …) — not one of CellAge's 927 curated
  genes.
* "GenAge human genes" came back empty.
* "CellAge ∩ GenAge" was not expressible at all.
* Selecting `cellage_genes.metta` or `genage_models_etl.metta` as
  `ontology_files` pushed the prompt to 417k / 386k tokens, over OpenAI's 272k
  limit, so the one workaround failed too.

The prompt half of that is already fixed (a file over
`PLN_PROMPT_FILE_MAX_BYTES` is summarised into a schema card:
`cellage_genes.metta` went from ~131,000 tokens to ~215). This module fixes the
other half — the data itself.

Why Python, and why the CSVs
----------------------------
Three constraints decide the design, and none of them leaves a choice:

1. The generated `.metta` files live in `build/`, which is **gitignored** and is
   not on `_discover_metta_files`'s search path, so they are not selectable as
   `ontology_files` even before size enters the picture.
2. They are 8-14x over `PLN_MAX_KB_FILE_BYTES`, and loading a bulk ETL file into
   a hyperon 0.2.10 space ABORTS the interpreter (a non-unwinding Rust panic in
   hyperon-space's trie, uncatchable from Python) on the next `match` with a
   variable in the row slot. So the engine can never hold this data.
3. `genage_models_parser.py:57` writes the SANITISED symbol back out as the gene
   name — `aak-2` is emitted as `(GeneSymbol aak_2 "aak_2")`, destroying the
   real symbol. Anything built from that file inherits the damage.

So the index is built from the **source tables under `data/`**, not from the
generated MeTTa, and it answers in Python. Reading the four tables costs
~185 ms warm (~400 ms cold), and the result is cached on a (path, mtime, size)
stamp exactly as `drugage_selector._source_stamp` does it — regenerating a table
invalidates the cache without a restart.

`data/**/*.csv` and `*.tsv` are gitignored; the `.zip` archives they were
unpacked from ARE committed. So every source is read from the unpacked file when
it is there and from the zip member in memory when it is not, and a source that
is missing entirely is reported as `available: false` rather than raising. On a
fresh checkout with no ETL run, every endpoint still answers — it just says the
source is unavailable.

What is and is not claimed
--------------------------
CellAge's curated effect labels are **experimental annotations**, not inference:
a curator read a paper that reported gene X inducing or inhibiting senescence in
a cell line and recorded the direction. There is no effect size anywhere in the
table, and this module does not manufacture one. The `(Causes … (stv 0.82 0.70))`
atoms in `build/cellage_genes.metta` are the ETL's own invention, computed by
`cellage_etl.calibrated_stv` from the senescence type and the cancer-cell flag —
they do NOT come from `epistemic_calibration.metta` and are not surfaced here as
calibrated confidence. `cellage_calibration.metta` is the layer that assigns a
calibrated truth value, and it ignores those numbers.

Entrez is the join key
----------------------
Symbols are not a join key across these four tables:

* CellAge ∩ GenAge human is 113 genes by entrez and 112 by symbol — close, but
  the symbol join silently loses one and would silently gain aliases.
* CellAge ∩ GenAge models is **1** gene by entrez and 67 by symbol, and the 67
  are an artefact: GenAge models is worm/yeast/fly/mouse, whose entrez ids live
  in different species namespaces, and whose symbols collide with human ones
  (`ATM`, `AKT1`, `BRCA1`) as ORTHOLOGUES, not as the same gene. A symbol join
  against GenAge models is therefore reported with a warning, never silently.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Repo root is two levels up from this file (pln_chat/ontology/ -> repo).
_REPO = Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "data"

#: Hard cap on any listing this module returns. The four tables are 4,720 rows
#: together; handing a caller all of them in one JSON body is not an answer.
MAX_LIMIT = 200

#: Canonical source keys, in the order they are reported.
SOURCE_KEYS: tuple[str, ...] = (
    "cellage_curated",
    "cellage_expression",
    "genage_human",
    "genage_models",
)


@dataclass(frozen=True)
class SourceSpec:
    """Where one source table lives and how to read it.

    `csv` is the unpacked file (gitignored, present after `scripts/run_etl.sh`);
    `archive`/`member` is the committed zip fallback, read in memory.
    """
    key: str
    label: str
    csv: Path
    archive: Path
    member: str
    sep: str
    #: The atom-id prefix the corresponding ETL gives rows of this table, so a
    #: record can point at its MeTTa counterpart. None = no ETL emits rows.
    row_atom_prefix: Optional[str]


SOURCES: dict[str, SourceSpec] = {
    "cellage_curated": SourceSpec(
        key="cellage_curated",
        label="CellAge curated senescence genes",
        csv=_DATA / "cellage" / "cellage3.tsv",
        archive=_DATA / "cellage" / "cellAge.zip",
        member="cellage3.tsv",
        sep="\t",
        row_atom_prefix="CellAgeRow_",
    ),
    "cellage_expression": SourceSpec(
        key="cellage_expression",
        label="CellAge senescence expression signatures",
        csv=_DATA / "cellage" / "signatures1.csv",
        archive=_DATA / "cellage" / "cellSignatures.zip",
        member="signatures1.csv",
        sep=";",
        row_atom_prefix="CellAgeExpressionRow_",
    ),
    "genage_human": SourceSpec(
        key="genage_human",
        label="GenAge human ageing-associated genes",
        csv=_DATA / "genage" / "genage_human.csv",
        archive=_DATA / "genage" / "human_genes.zip",
        member="genage_human.csv",
        sep=",",
        row_atom_prefix="GenAgeHumanRow_",
    ),
    "genage_models": SourceSpec(
        key="genage_models",
        label="GenAge model-organism longevity genes",
        csv=_DATA / "genage" / "genage_models.csv",
        archive=_DATA / "genage" / "models_genes.zip",
        member="genage_models.csv",
        sep=",",
        row_atom_prefix="GenAgeModelRow_",
    ),
}

#: Sources whose entrez ids and symbols are NOT in the human namespace. A join
#: that crosses this boundary is an orthology claim, and this module does not
#: make orthology claims.
NON_HUMAN_SOURCES: frozenset[str] = frozenset({"genage_models"})

#: Organisms `genage_models_parser.ORGANISM_MAP` knows how to emit. A row with
#: any other organism is skipped by that ETL, so it has no MeTTa row atom — the
#: set is duplicated here only to keep `metta_row_id` honest about that.
_ETL_KNOWN_ORGANISMS: frozenset[str] = frozenset({
    "Caenorhabditis elegans", "Mus musculus", "Saccharomyces cerevisiae",
    "Drosophila melanogaster", "Mesocricetus auratus", "Podospora anserina",
    "Schizosaccharomyces pombe", "Danio rerio", "Caenorhabditis briggsae",
})

# CellAge's "Senescence Effect" column, normalised to the ETL's vocabulary.
_SENESCENCE_TYPES = {
    "oncogene-induced": "OncogeneInducedSenescence",
    "stress-induced": "StressInducedSenescence",
    "replicative": "ReplicativeSenescence",
    "unclear": "UnspecifiedSenescenceType",
}
_EFFECT_DIRECTION = {"induces": "Increases", "inhibits": "Decreases"}

#: GenAge human's `why` column — the curators' selection basis. The vocabulary
#: is the one `epistemic_calibration.metta` already maps to a confidence
#: (`selection-basis-confidence`), which is why it is preserved verbatim.
VALID_SELECTION_BASES: frozenset[str] = frozenset({
    "mammal", "model", "cell", "functional", "human",
    "downstream", "putative", "upstream", "human_link",
})


def normalise_pmid(raw: str) -> Optional[str]:
    """`PMID_PMID_26583757` / `26583757` / `PMID_26583757` -> `PMID_26583757`.

    THE DOUBLED PREFIX IS A REAL ETL DEFECT, not a quirk of this reader.
    `cellage_etl.write_curated` built the token as ``f"PMID_{atom(ref, 'PMID_')}"``
    and `atom()`'s `prefix_if_numeric` fires on every all-digit reference, so
    every CellAge provenance atom in an existing `build/` reads `PMID_PMID_…`.
    The ETL is fixed in this commit, but a `build/` generated before the fix is
    still on disk and still has to be read, so both spellings normalise here.
    """
    token = str(raw).strip()
    if not token:
        return None
    while token.upper().startswith("PMID_"):
        token = token[5:]
    token = token.strip()
    if not token or not token.isdigit():
        return None
    return f"PMID_{token}"


@dataclass(frozen=True)
class GeneRecord:
    """One (source, row) fact about one gene, with everything its table said.

    Deliberately one record per ROW rather than per gene: CellAge curated has 83
    duplicate symbols (the same gene annotated in two senescence types or two
    cell contexts), and collapsing them would pick a winner no source picked.
    """
    source: str
    row_index: int
    symbol: str
    entrez: Optional[int] = None
    gene_name: Optional[str] = None

    # ── CellAge curated ──
    #: "Induces" | "Inhibits" | "Unclear", verbatim from the table.
    senescence_effect: Optional[str] = None
    #: "Increases" | "Decreases" | None — the effect label as a direction on
    #: CellularSenescence. None for "Unclear", which asserts no direction.
    senescence_direction: Optional[str] = None
    senescence_type: Optional[str] = None
    cell_context: Optional[str] = None
    pmids: tuple[str, ...] = ()

    # ── CellAge expression ──
    expression_direction: Optional[str] = None
    expression_samples: Optional[int] = None
    p_value: Optional[float] = None

    # ── GenAge human ──
    uniprot: Optional[str] = None
    selection_basis: tuple[str, ...] = ()

    # ── GenAge models ──
    organism: Optional[str] = None
    lifespan_effect: Optional[str] = None
    longevity_influence: Optional[str] = None
    avg_lifespan_change_percent: Optional[float] = None

    #: The atom id the corresponding ETL gives this row in `build/`, or None
    #: when that ETL filters the row out (see `_row_atom_id`).
    metta_row_id: Optional[str] = None

    def as_dict(self) -> dict:
        """Only the fields this record's source actually has.

        A GenAge models row has no cell context and a CellAge row has no
        organism; emitting them as explicit nulls would read as "measured and
        absent" rather than "this table does not have that column".
        """
        out: dict = {
            "source": self.source,
            "row_index": self.row_index,
            "symbol": self.symbol,
            "entrez": self.entrez,
            "gene_name": self.gene_name,
            "metta_row_id": self.metta_row_id,
        }
        optional = {
            "senescence_effect": self.senescence_effect,
            "senescence_direction": self.senescence_direction,
            "senescence_type": self.senescence_type,
            "cell_context": self.cell_context,
            "pmids": list(self.pmids) or None,
            "expression_direction": self.expression_direction,
            "expression_samples": self.expression_samples,
            "p_value": self.p_value,
            "uniprot": self.uniprot,
            "selection_basis": list(self.selection_basis) or None,
            "organism": self.organism,
            "lifespan_effect": self.lifespan_effect,
            "longevity_influence": self.longevity_influence,
            "avg_lifespan_change_percent": self.avg_lifespan_change_percent,
        }
        out.update({k: v for k, v in optional.items() if v is not None})
        return out


@dataclass
class SourceStatus:
    """Whether one source table could be read, and what came back."""
    key: str
    label: str
    available: bool
    rows: int = 0
    file: Optional[str] = None
    #: "csv" (unpacked file), "zip" (committed archive) or None.
    origin: Optional[str] = None
    note: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "source": self.key,
            "label": self.label,
            "available": self.available,
            "rows": self.rows,
            "file": self.file,
            "origin": self.origin,
            "note": self.note,
        }


# ── Reading a source table ───────────────────────────────────────────────────

def _read_table(spec: SourceSpec):
    """(DataFrame, origin) for one source, or (None, None) when unreadable.

    Never raises: a missing, truncated or unparseable table has to degrade into
    `available: false`, because `data/**/*.csv` is gitignored and a fresh
    checkout legitimately has nothing but the zips.
    """
    import pandas as pd

    if spec.csv.exists():
        try:
            return pd.read_csv(spec.csv, sep=spec.sep), "csv"
        except Exception:   # noqa: BLE001 - an unreadable table is "unavailable"
            pass
    if spec.archive.exists():
        try:
            with zipfile.ZipFile(spec.archive) as zf:
                raw = zf.read(spec.member)
            return pd.read_csv(io.BytesIO(raw), sep=spec.sep), "zip"
        except Exception:   # noqa: BLE001
            pass
    return None, None


def _text(value) -> Optional[str]:
    import pandas as pd

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    out = str(value).strip()
    return out or None


def _int(value) -> Optional[int]:
    import pandas as pd

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _float(value) -> Optional[float]:
    import pandas as pd

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _row_atom_id(spec: SourceSpec, row_index: int, emitted: bool) -> Optional[str]:
    """`CellAgeRow_3` etc., or None when the ETL drops the row.

    The ETLs index their output by the SOURCE row number (`df.iterrows()`), and
    they filter rather than renumber, so a surviving row's atom id is derivable
    without reading `build/` at all. `emitted` is the caller's per-source
    restatement of the ETL's filter; it is a duplicated policy, and
    `tests/test_gene_index.py` pins it against the real generated files.
    """
    if not emitted or spec.row_atom_prefix is None:
        return None
    return f"{spec.row_atom_prefix}{row_index}"


def _load_cellage_curated(spec: SourceSpec) -> tuple[list[GeneRecord], SourceStatus]:
    df, origin = _read_table(spec)
    if df is None:
        return [], _unavailable(spec)
    records: list[GeneRecord] = []
    for i, row in df.iterrows():
        symbol = _text(row.get("Gene symbol"))
        if not symbol:
            continue
        effect = _text(row.get("Senescence Effect"))
        direction = _EFFECT_DIRECTION.get((effect or "").lower())
        sen_type_raw = _text(row.get("Type of senescence")) or "Unclear"
        sen_type = _SENESCENCE_TYPES.get(
            sen_type_raw.lower(), "UnspecifiedSenescenceType")
        cancer = (_text(row.get("Cancer Cell")) or "").lower()
        context = ("CancerCellContext" if cancer == "yes"
                   else "NonCancerCellContext" if cancer == "no"
                   else "UnknownCellContext")
        pmid = normalise_pmid(str(row.get("Reference", "")))
        records.append(GeneRecord(
            source=spec.key,
            row_index=int(i),
            symbol=symbol,
            entrez=_int(row.get("Entrez ID")),
            gene_name=_text(row.get("Gene name")),
            senescence_effect=effect,
            senescence_direction=direction,
            senescence_type=sen_type,
            cell_context=context,
            pmids=(pmid,) if pmid else (),
            # cellage_etl.write_curated drops a row whose effect is neither
            # Induces nor Inhibits (22 of 949), so those have no row atom.
            metta_row_id=_row_atom_id(spec, int(i), direction is not None),
        ))
    return records, _available(spec, origin, len(records))


def _load_cellage_expression(spec: SourceSpec) -> tuple[list[GeneRecord], SourceStatus]:
    df, origin = _read_table(spec)
    if df is None:
        return [], _unavailable(spec)
    records: list[GeneRecord] = []
    for i, row in df.iterrows():
        symbol = _text(row.get("gene_symbol"))
        if not symbol:
            continue
        over = _int(row.get("ovevrexp"))          # the source's own typo
        if over is None:
            over = _int(row.get("overexp"))
        p_value = _float(row.get("p_value"))
        records.append(GeneRecord(
            source=spec.key,
            row_index=int(i),
            symbol=symbol,
            entrez=_int(row.get("entrez_id")),
            gene_name=_text(row.get("gene_name")),
            expression_direction=("OverexpressedInSenescentCells" if over == 1
                                  else "UnderexpressedInSenescentCells"),
            expression_samples=_int(row.get("total")),
            p_value=p_value,
            # cellage_etl.write_expression keeps p <= 0.05 (the default).
            metta_row_id=_row_atom_id(
                spec, int(i), p_value is not None and p_value <= 0.05),
        ))
    return records, _available(spec, origin, len(records))


def _load_genage_human(spec: SourceSpec) -> tuple[list[GeneRecord], SourceStatus]:
    df, origin = _read_table(spec)
    if df is None:
        return [], _unavailable(spec)
    records: list[GeneRecord] = []
    for i, row in df.iterrows():
        symbol = _text(row.get("symbol"))
        if not symbol:
            continue
        why = _text(row.get("why")) or ""
        bases = tuple(
            t for t in (p.strip() for p in why.split(","))
            if t in VALID_SELECTION_BASES
        )
        records.append(GeneRecord(
            source=spec.key,
            row_index=int(i),
            symbol=symbol,
            entrez=_int(row.get("entrez gene id")),
            gene_name=_text(row.get("name")),
            uniprot=_text(row.get("uniprot")),
            selection_basis=bases,
            metta_row_id=_row_atom_id(spec, int(i), True),
        ))
    return records, _available(spec, origin, len(records))


def _load_genage_models(spec: SourceSpec) -> tuple[list[GeneRecord], SourceStatus]:
    df, origin = _read_table(spec)
    if df is None:
        return [], _unavailable(spec)
    records: list[GeneRecord] = []
    for i, row in df.iterrows():
        symbol = _text(row.get("symbol"))
        if not symbol:
            continue
        organism = _text(row.get("organism"))
        records.append(GeneRecord(
            source=spec.key,
            row_index=int(i),
            # The REAL symbol, straight from the CSV. `genage_models_etl.metta`
            # has `aak_2` here because the parser writes the sanitised atom back
            # out as the symbol string; that is the bug this index routes around.
            symbol=symbol,
            entrez=_int(row.get("entrez gene id")),
            gene_name=_text(row.get("name")),
            organism=organism,
            lifespan_effect=_text(row.get("lifespan effect")),
            longevity_influence=_text(row.get("longevity influence")),
            avg_lifespan_change_percent=_float(
                row.get("avg lifespan change (max obsv)")),
            metta_row_id=_row_atom_id(
                spec, int(i), organism in _ETL_KNOWN_ORGANISMS),
        ))
    return records, _available(spec, origin, len(records))


_LOADERS = {
    "cellage_curated": _load_cellage_curated,
    "cellage_expression": _load_cellage_expression,
    "genage_human": _load_genage_human,
    "genage_models": _load_genage_models,
}


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (a test may point elsewhere)."""
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def _available(spec: SourceSpec, origin: Optional[str], rows: int) -> SourceStatus:
    path = spec.csv if origin == "csv" else spec.archive
    note = None
    if origin == "zip":
        note = (f"Read from the committed archive {spec.archive.name}; the "
                f"unpacked {spec.csv.name} is gitignored and absent. Run "
                f"scripts/run_etl.sh to unpack it.")
    return SourceStatus(
        key=spec.key, label=spec.label, available=True, rows=rows,
        file=_rel(path), origin=origin, note=note,
    )


def _unavailable(spec: SourceSpec) -> SourceStatus:
    return SourceStatus(
        key=spec.key, label=spec.label, available=False, rows=0,
        file=_rel(spec.csv), origin=None,
        note=(f"Neither {spec.csv.name} nor the archive {spec.archive.name} could "
              f"be read, so this source contributes no records. This is an "
              f"absence of DATA, not an assertion about any gene."),
    )


# ── The index ────────────────────────────────────────────────────────────────

@dataclass
class GeneIndex:
    """Every gene record, indexed by entrez and by upper-cased symbol."""
    records: list[GeneRecord] = field(default_factory=list)
    by_entrez: dict[int, list[GeneRecord]] = field(default_factory=dict)
    by_symbol: dict[str, list[GeneRecord]] = field(default_factory=dict)
    #: entrez -> the set of sources holding it. Precomputed because it IS the
    #: intersection answer: `CellAge ∩ GenAge human` is every key whose value
    #: contains both.
    sources_by_entrez: dict[int, frozenset[str]] = field(default_factory=dict)
    sources_by_symbol: dict[str, frozenset[str]] = field(default_factory=dict)
    status: dict[str, SourceStatus] = field(default_factory=dict)

    # ── lookup ──

    def lookup(self, key: str) -> tuple[list[GeneRecord], str]:
        """Records for a symbol or an entrez id, and which of the two it was.

        An all-digit key is an entrez id; anything else is a symbol, matched
        case-insensitively. Returns ([], "symbol") for an unknown name rather
        than raising — an unknown gene is an answer.
        """
        token = key.strip()
        if token.isdigit():
            return list(self.by_entrez.get(int(token), [])), "entrez"
        return list(self.by_symbol.get(token.upper(), [])), "symbol"

    def symbols_for_entrez(self, entrez: int) -> list[str]:
        """Every distinct symbol the tables give one entrez id."""
        seen: dict[str, None] = {}
        for rec in self.by_entrez.get(entrez, []):
            seen.setdefault(rec.symbol, None)
        return list(seen)

    # ── filtering ──

    def select(
        self,
        *,
        source: Optional[str] = None,
        effect: Optional[str] = None,
        senescence_type: Optional[str] = None,
        cell_context: Optional[str] = None,
        organism: Optional[str] = None,
        in_sources: Optional[Iterable[str]] = None,
    ) -> list[GeneRecord]:
        """Every record matching the filters, in load order (source, row).

        `in_sources` is the cross-source filter: a record is kept when its GENE
        (by entrez, the only reliable key) appears in all of the named sources.
        """
        want = {s.strip() for s in in_sources if s.strip()} if in_sources else None
        out: list[GeneRecord] = []
        for rec in self.records:
            if source and rec.source != source:
                continue
            if effect and (rec.senescence_effect or "").lower() != effect.lower():
                continue
            if senescence_type and (rec.senescence_type or "").lower() != senescence_type.lower():
                continue
            if cell_context and (rec.cell_context or "").lower() != cell_context.lower():
                continue
            if organism and (rec.organism or "").lower() != organism.lower():
                continue
            if want is not None:
                if rec.entrez is None:
                    continue
                if not want.issubset(self.sources_by_entrez.get(rec.entrez, frozenset())):
                    continue
            out.append(rec)
        return out

    # ── intersection ──

    def intersect(self, a: str, b: str, key: str = "entrez") -> list[dict]:
        """Genes held by BOTH sources, joined on entrez (default) or symbol.

        One entry per joined gene, carrying the symbols each side uses, so a
        symbol-keyed join cannot hide that the two sides disagree.
        """
        table = self.sources_by_entrez if key == "entrez" else self.sources_by_symbol
        index = self.by_entrez if key == "entrez" else self.by_symbol
        out: list[dict] = []
        for gene_key in sorted(k for k, srcs in table.items() if {a, b} <= srcs):
            recs = index[gene_key]  # type: ignore[index]
            a_recs = [r for r in recs if r.source == a]
            b_recs = [r for r in recs if r.source == b]
            out.append({
                "key": gene_key if key == "symbol" else int(gene_key),  # type: ignore[arg-type]
                "entrez": a_recs[0].entrez if key == "symbol" else int(gene_key),  # type: ignore[arg-type]
                "symbols": sorted({r.symbol for r in recs}),
                f"{a}_rows": len(a_recs),
                f"{b}_rows": len(b_recs),
                "gene_name": next(
                    (r.gene_name for r in recs if r.gene_name), None),
            })
        return out

    def vocabulary(self, attr: str) -> list[str]:
        """Sorted distinct values of one record field — the filter vocabularies."""
        return sorted({
            v for v in (getattr(r, attr, None) for r in self.records)
            if isinstance(v, str) and v
        })


def build_gene_index(specs: Optional[dict[str, SourceSpec]] = None) -> GeneIndex:
    """Read every source table and build the indexes. Never raises."""
    specs = specs or SOURCES
    index = GeneIndex()
    for key in SOURCE_KEYS:
        spec = specs.get(key)
        if spec is None:
            continue
        records, status = _LOADERS[key](spec)
        index.status[key] = status
        index.records.extend(records)

    for rec in index.records:
        if rec.entrez is not None:
            index.by_entrez.setdefault(rec.entrez, []).append(rec)
        index.by_symbol.setdefault(rec.symbol.upper(), []).append(rec)

    for entrez, recs in index.by_entrez.items():
        index.sources_by_entrez[entrez] = frozenset(r.source for r in recs)
    for symbol, recs in index.by_symbol.items():
        index.sources_by_symbol[symbol] = frozenset(r.source for r in recs)
    return index


# Cached on the (path, mtime, size) stamp of every source, exactly as
# drugage_selector caches its vocabulary: re-running the ETL (or unpacking the
# zips) invalidates the cache without a restart, and a stable tree costs one
# parse per process (~185 ms for 4,720 rows).
_CACHE: dict[tuple, GeneIndex] = {}


def _source_stamp(path: Path) -> tuple[str, float, int]:
    try:
        st = path.stat()
        return (str(path), st.st_mtime, st.st_size)
    except OSError:
        return (str(path), 0.0, 0)


def _cache_key(specs: dict[str, SourceSpec]) -> tuple:
    return tuple(
        _source_stamp(specs[k].csv) + _source_stamp(specs[k].archive)
        for k in SOURCE_KEYS if k in specs
    )


def gene_index(specs: Optional[dict[str, SourceSpec]] = None) -> GeneIndex:
    """The shared, cached `GeneIndex` over `data/`."""
    specs = specs or SOURCES
    key = _cache_key(specs)
    cached = _CACHE.get(key)
    if cached is None:
        cached = build_gene_index(specs)
        _CACHE[key] = cached
    return cached
