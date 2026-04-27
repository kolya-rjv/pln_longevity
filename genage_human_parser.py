# genage_human_parser.py
import pandas as pd
import re
import sys
from pathlib import Path


VALID_SELECTION_BASES = {
    "mammal", "model", "cell", "functional", "human",
    "downstream", "putative", "upstream", "human_link",
}


def sanitize_atom(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def parse_why(raw) -> list[str]:
    if not isinstance(raw, str):
        return []
    tokens = [t.strip() for t in raw.split(",")]
    unknown = [t for t in tokens if t and t not in VALID_SELECTION_BASES]
    if unknown:
        print(f"  WARNING: unknown selection basis tokens: {unknown}", file=sys.stderr)
    return [t for t in tokens if t in VALID_SELECTION_BASES]


def emit_gene_entities(df: pd.DataFrame) -> list[str]:
    lines = [
        ";; -----------------------------------------------",
        ";; Gene entity declarations (one per unique symbol)",
        ";; -----------------------------------------------",
        "",
    ]
    for _, row in df.drop_duplicates(subset="symbol").iterrows():
        symbol  = sanitize_atom(str(row["symbol"]))
        name    = str(row["name"]).replace('"', '\\"')
        entrez  = row.get("entrez gene id")
        uniprot = str(row.get("uniprot", "")).strip()

        lines.append(f"(: {symbol} Gene)")
        lines.append(f'(GeneSymbol {symbol} "{symbol}")')
        lines.append(f'(GeneName   {symbol} "{name}")')

        if pd.notna(entrez):
            lines.append(f"(EntrezID  {symbol} {int(entrez)})")

        if uniprot and uniprot.lower() not in ("nan", ""):
            lines.append(f'(UniprotID {symbol} "{uniprot}")')

        lines.append("")

    return lines


def emit_human_rows(df: pd.DataFrame) -> list[str]:
    lines = [
        ";; -----------------------------------------------",
        ";; HumanAgingGene association records (one per row)",
        ";; -----------------------------------------------",
        "",
    ]
    for idx, row in df.iterrows():
        row_id = f"GenAgeHumanRow_{idx}"
        symbol = sanitize_atom(str(row["symbol"]))
        bases  = parse_why(row.get("why", ""))

        lines.append(f"; GenAge ID {row['GenAge ID']} — {row['symbol']}")
        lines.append(f"(InstanceOf   {row_id} HumanAgingGene)")
        lines.append(f"(InvolvesGene {row_id} {symbol})")

        for basis in bases:
            lines.append(f"(HasSelectionBasis {row_id} {basis})")

        lines.append("")

    return lines


def generate(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)

    lines = [
        ";; Auto-generated GenAge Human ETL",
        ";; Source: genage_human.csv",
        ";; Do not hand-edit — regenerate from genage_human_parser.py",
        "",
    ]
    lines += emit_gene_entities(df)
    lines += emit_human_rows(df)

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    csv_in = 'data/GenAge/human_genes/genage_human.csv'
    metta_out = "genage_human_etl.metta"
    generate(csv_in, metta_out)