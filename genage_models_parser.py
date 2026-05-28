# genage_models_parser.py
import pandas as pd
import re
import sys
from pathlib import Path


# longevity influence is the curated gene-level annotation.
# Pro-Longevity: gene promotes longevity → (Increases Lifespan)
# Anti-Longevity: gene works against longevity → (Decreases Lifespan)
# Unclear / Unannotated / Necessary for fitness → discarded, no atom emitted
LONGEVITY_INFLUENCE_MAP = {
    "Pro-Longevity":  "(Increases Lifespan)",
    "Anti-Longevity": "(Decreases Lifespan)",
}

ORGANISM_MAP = {
    "Caenorhabditis elegans":    "Caenorhabditis_elegans",
    "Mus musculus":              "Mus_musculus",
    "Saccharomyces cerevisiae":  "Saccharomyces_cerevisiae",
    "Drosophila melanogaster":   "Drosophila_melanogaster",
    "Mesocricetus auratus":      "Mesocricetus_auratus",
    "Podospora anserina":        "Podospora_anserina",
    "Schizosaccharomyces pombe": "Schizosaccharomyces_pombe",
    "Danio rerio":               "Danio_rerio",
    "Caenorhabditis briggsae":   "Caenorhabditis_briggsae",
}


def sanitize_atom(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def emit_gene_entities(df: pd.DataFrame) -> list[str]:
    # Model organism genes: no UniprotID since these are non-human
    lines = [
        ";; -----------------------------------------------",
        ";; Gene entity declarations (one per unique symbol)",
        ";; -----------------------------------------------",
        "",
    ]
    seen: set[str] = set()

    for _, row in df.iterrows():
        symbol = sanitize_atom(str(row["symbol"]))
        if symbol in seen:
            continue
        seen.add(symbol)

        name   = str(row["name"]).replace('"', '\\"')
        entrez = row.get("entrez gene id")

        lines.append(f"(: {symbol} Gene)")
        lines.append(f'(GeneSymbol {symbol} "{symbol}")')
        lines.append(f'(GeneName   {symbol} "{name}")')

        if pd.notna(entrez):
            lines.append(f"(EntrezID  {symbol} {int(entrez)})")

        lines.append("")

    return lines


def emit_model_rows(df: pd.DataFrame) -> list[str]:
    lines = [
        ";; -----------------------------------------------",
        ";; GeneticExperiment records (one per GenAge row)",
        ";; -----------------------------------------------",
        "",
    ]
    skipped = 0

    for idx, row in df.iterrows():
        row_id       = f"GenAgeModelRow_{idx}"
        genage_id    = row["GenAge ID"]
        symbol       = sanitize_atom(str(row["symbol"]))
        organism_raw = str(row["organism"]).strip()

        organism = ORGANISM_MAP.get(organism_raw)
        if organism is None:
            print(
                f"  WARNING: unknown organism '{organism_raw}' "
                f"at GenAge ID {genage_id} — row skipped",
                file=sys.stderr,
            )
            skipped += 1
            continue

        longevity_raw   = str(row.get("longevity influence", "")).strip()
        lifespan_effect = LONGEVITY_INFLUENCE_MAP.get(longevity_raw)
        lifespan_change = row.get("avg lifespan change (max obsv)")

        lines.append(f"; GenAge ID {genage_id} — {row['symbol']} in {organism_raw}")
        lines.append(f"(InstanceOf   {row_id} GeneticExperiment)")
        lines.append(f"(InvolvesGene {row_id} {symbol})")
        lines.append(f"(UsesSpecies  {row_id} {organism})")

        if lifespan_effect is not None:
            lines.append(f"(HasLifespanEffect {row_id} {lifespan_effect})")

        if pd.notna(lifespan_change):
            lines.append(f"(LifespanChangePercent {row_id} {float(lifespan_change)})")

        lines.append("")

    if skipped:
        print(f"  INFO: {skipped} rows skipped (unknown organism)", file=sys.stderr)

    return lines


def generate(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)

    lines = [
        ";; Auto-generated GenAge Models ETL",
        ";; Source: genage_models.csv",
        ";; Do not hand-edit — regenerate from genage_models_parser.py",
        "",
    ]
    lines += emit_gene_entities(df)
    lines += emit_model_rows(df)

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    csv_in = "data/GenAge/models_genes/genage_models.csv"
    metta_out = "genage_models_etl.metta"
    generate(csv_in, metta_out)