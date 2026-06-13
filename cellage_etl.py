#!/usr/bin/env python3
"""
CellAge -> MeTTa ETL for pln_longevity.

Inputs:
  1) Curated CellAge gene table with columns like:
     Entrez ID, Gene symbol, Gene name, Cancer Cell, Type of senescence,
     Senescence Effect, Reference
  2) CellAge expression/signature table with columns like:
     gene_symbol, gene_name, entrez_id, total, overexp/ovevrexp, underexp, p_value

Outputs:
  - cellage_genes.metta
  - cellage_expression.metta
  - cellage_metadata.metta

Default policy:
  - keep curated rows with Senescence Effect in {Induces, Inhibits}
  - drop Unclear Senescence Effect from causal atoms, but report counts
  - keep both cancer and non-cancer cell evidence, annotated as context
  - keep rows with Type of senescence == Unclear as UnspecifiedSenescenceType
  - keep expression rows passing p-value threshold; expression evidence is associative, not causal
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


def atom(value: Any, prefix_if_numeric: str = "") -> str:
    """Convert arbitrary table value into a conservative MeTTa atom."""
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip()
    if not s:
        return "Unknown"
    s = s.replace("β", "beta")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "Unknown"
    if s[0].isdigit():
        s = f"{prefix_if_numeric}{s}"
    return s


def qstr(value: Any) -> str:
    if pd.isna(value):
        return '""'
    s = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{s}"'


def num_or_unknown(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "0"
    try:
        x = float(value)
        if x.is_integer():
            return str(int(x))
        return repr(x)
    except Exception:
        return "0"


def norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in out.columns]
    return out


def pick_col(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing required column. Tried: {names}; found: {list(df.columns)}")


def senescence_type_atom(value: Any) -> str:
    mapping = {
        "oncogene_induced": "OncogeneInducedSenescence",
        "stress_induced": "StressInducedSenescence",
        "replicative": "ReplicativeSenescence",
        "unclear": "UnspecifiedSenescenceType",
    }
    key = atom(value).lower()
    return mapping.get(key, atom(value, "SenescenceType_"))


def cancer_context_atom(value: Any) -> str:
    key = str(value).strip().lower()
    if key == "yes":
        return "CancerCellContext"
    if key == "no":
        return "NonCancerCellContext"
    return "UnknownCellContext"


def effect_expr(value: Any) -> str | None:
    key = str(value).strip().lower()
    if key == "induces":
        return "(Increases CellularSenescence)"
    if key == "inhibits":
        return "(Decreases CellularSenescence)"
    return None


def calibrated_stv(effect: str, sen_type: str, cancer_context: str) -> str:
    # Curated CellAge records are useful but single-record database evidence.
    # Penalize unspecified senescence type and cancer-only context slightly.
    strength = 0.82 if effect == "Induces" else 0.78
    conf = 0.78
    if sen_type == "UnspecifiedSenescenceType":
        conf -= 0.08
    if cancer_context == "CancerCellContext":
        conf -= 0.05
    return f"(stv {strength:.2f} {conf:.2f})"


def write_curated(df: pd.DataFrame, out_path: Path, keep_unclear_effect: bool) -> dict[str, int]:
    df = norm_columns(df)
    entrez = pick_col(df, "entrez_id")
    symbol = pick_col(df, "gene_symbol")
    gene_name = pick_col(df, "gene_name")
    cancer = pick_col(df, "cancer_cell")
    sen_type = pick_col(df, "type_of_senescence")
    effect = pick_col(df, "senescence_effect")
    ref = pick_col(df, "reference")

    kept = dropped_unclear = 0
    lines: list[str] = ["; Auto-generated CellAge curated gene ETL", ""]

    for i, row in df.iterrows():
        eff_expr = effect_expr(row[effect])
        if eff_expr is None and not keep_unclear_effect:
            dropped_unclear += 1
            continue

        row_id = f"CellAgeRow_{i}"
        gene = f"Gene_{atom(row[symbol])}"
        pub = f"PMID_{atom(row[ref], 'PMID_')}"
        stype = senescence_type_atom(row[sen_type])
        ctx = cancer_context_atom(row[cancer])
        effect_label = atom(row[effect], "SenescenceEffect_")

        lines.extend([
            f"; source row {i}",
            f"(InstanceOf {row_id} CellSenescenceRecord)",
            f"(InvolvesGene {row_id} {gene})",
            f"(GeneSymbol {gene} {qstr(row[symbol])})",
            f"(GeneName {gene} {qstr(row[gene_name])})",
            f"(EntrezID {gene} {num_or_unknown(row[entrez])})",
            f"(ReportedIn {row_id} {pub})",
            f"(HasSenescenceType {row_id} {stype})",
            f"(UsesCellContext {row_id} {ctx})",
            f"(HasSenescenceEffectLabel {row_id} {effect_label})",
        ])
        if eff_expr is not None:
            stv = calibrated_stv(str(row[effect]).strip(), stype, ctx)
            lines.extend([
                f"(HasSenescenceEffect {row_id} {eff_expr})",
                f"(Causes {gene} {eff_expr} {stv})",
            ])
        lines.append("")
        kept += 1

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"kept_curated": kept, "dropped_unclear_effect": dropped_unclear}


def write_expression(df: pd.DataFrame, out_path: Path, p_value_max: float) -> dict[str, int]:
    df = norm_columns(df)
    symbol = pick_col(df, "gene_symbol")
    gene_name = pick_col(df, "gene_name")
    entrez = pick_col(df, "entrez_id")
    total = pick_col(df, "total")
    over = pick_col(df, "overexp", "ovevrexp")
    under = pick_col(df, "underexp")
    pval = pick_col(df, "p_value")

    before = len(df)
    df = df[pd.to_numeric(df[pval], errors="coerce") <= p_value_max].copy()
    kept = len(df)

    lines: list[str] = ["; Auto-generated CellAge expression/signature ETL", ""]
    for i, row in df.iterrows():
        row_id = f"CellAgeExpressionRow_{i}"
        gene = f"Gene_{atom(row[symbol])}"
        direction = "OverexpressedInSenescentCells" if int(row[over]) == 1 else "UnderexpressedInSenescentCells"
        assoc = "(Increases CellularSenescence)" if direction == "OverexpressedInSenescentCells" else "(Decreases CellularSenescence)"
        # Associative evidence, not causal. Strength is bounded by p-value but kept moderate.
        p = float(row[pval])
        conf = max(0.55, min(0.95, 1.0 - p))
        lines.extend([
            f"; source row {i}",
            f"(InstanceOf {row_id} SenescenceExpressionRecord)",
            f"(InvolvesGene {row_id} {gene})",
            f"(GeneSymbol {gene} {qstr(row[symbol])})",
            f"(GeneName {gene} {qstr(row[gene_name])})",
            f"(EntrezID {gene} {num_or_unknown(row[entrez])})",
            f"(ExpressionSampleCount {row_id} {num_or_unknown(row[total])})",
            f"(DifferentialExpressionDirection {row_id} {direction})",
            f"(PValue {row_id} {num_or_unknown(row[pval])})",
            f"(CorrelatedWith {gene} {assoc} (stv 0.65 {conf:.2f}))",
            "",
        ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"kept_expression": kept, "dropped_expression_by_pvalue": before - kept}


def write_metadata(out_path: Path) -> None:
    out_path.write_text("""; CellAge ETL metadata / value declarations

(: OncogeneInducedSenescence SenescenceType)
(: StressInducedSenescence SenescenceType)
(: ReplicativeSenescence SenescenceType)
(: UnspecifiedSenescenceType SenescenceType)

(: CancerCellContext CellContext)
(: NonCancerCellContext CellContext)
(: UnknownCellContext CellContext)

(: Induces SenescenceEffectLabel)
(: Inhibits SenescenceEffectLabel)
(: Unclear SenescenceEffectLabel)

(: OverexpressedInSenescentCells ExpressionDirection)
(: UnderexpressedInSenescentCells ExpressionDirection)
""", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True, type=Path, help="CellAge curated gene CSV/TSV/XLSX")
    ap.add_argument("--expression", required=True, type=Path, help="CellAge expression/signature CSV/TSV/XLSX")
    ap.add_argument("--outdir", default=Path("."), type=Path)
    ap.add_argument("--p-value-max", default=0.05, type=float)
    ap.add_argument("--keep-unclear-effect", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    curated_df = pd.read_csv(args.curated, sep="\t")
    expression_df = pd.read_csv(args.expression, sep=";")

    stats = {}
    stats.update(write_curated(curated_df, args.outdir / "cellage_genes.metta", args.keep_unclear_effect))
    stats.update(write_expression(expression_df, args.outdir / "cellage_expression.metta", args.p_value_max))
    write_metadata(args.outdir / "cellage_metadata.metta")

    print("CellAge ETL complete")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"wrote: {args.outdir / 'cellage_genes.metta'}")
    print(f"wrote: {args.outdir / 'cellage_expression.metta'}")
    print(f"wrote: {args.outdir / 'cellage_metadata.metta'}")


if __name__ == "__main__":
    main()
