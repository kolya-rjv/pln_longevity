#!/usr/bin/env bash
#
# Meta-ETL: unpack the raw HAGR data archives and regenerate every MeTTa KB
# file from them. Runs the per-dataset ETL scripts (DrugAge, GenAge human,
# GenAge models, CellAge) against the zips committed under data/.
#
# Outputs are written to OUT_DIR (default: ./build). They are NOT dropped into
# the repo root by default on purpose: the chat app auto-discovers every *.metta
# at the root, and hyperon 0.2.10 panics when querying a space past a few
# thousand atoms (see drugage_etl.py --limit). Staging in ./build keeps the
# full-size KB out of the app's auto-load path until it has been truncated or
# the runtime can handle it.
#
#   scripts/run_etl.sh                 # regenerate into ./build
#   PYTHON=python3.11 scripts/run_etl.sh
#
# DO NOT set OUT_DIR=. to write into the repo root. The chat app executes against EVERY
# repo-root *.metta under PLN_MAX_KB_FILE_BYTES, and hyperon 0.2.10 aborts the process
# (SIGABRT, uncatchable) on the first match query once the space carries too many
# DISTINCT head symbols. Measured: the root KB already holds ~137, the margin is between
# 8 and 12 more, and each generated NHANES file introduces 12-21. The per-file byte limit
# the app applies cannot bound this, because the failure is a property of the whole space
# rather than of any one file.
#
set -euo pipefail

# Resolve the repo root (this script lives in <root>/scripts) and run from it
# so every relative data path below is stable regardless of the caller's CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-$ROOT/build}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT_DIR"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# extract <zip> <member> <dest> — unpack a single named member to an exact path.
extract() {
  local zip="$1" member="$2" dest="$3"
  if [[ ! -f "$zip" ]]; then
    echo "ERROR: missing archive $zip" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dest")"
  unzip -p "$zip" "$member" > "$dest"
}

log "Unpacking raw archives → data/<dataset>/"
extract data/drugage/dataset.zip        drugage.csv       data/drugage/drugage.csv
extract data/genage/human_genes.zip     genage_human.csv  data/genage/genage_human.csv
extract data/genage/models_genes.zip    genage_models.csv data/genage/genage_models.csv
extract data/cellage/cellAge.zip        cellage3.tsv      data/cellage/cellage3.tsv
extract data/cellage/cellSignatures.zip signatures1.csv   data/cellage/signatures1.csv

log "DrugAge → drugage_etl.metta"
"$PYTHON" drugage_etl.py \
  --input  data/drugage/drugage.csv \
  --output "$OUT_DIR/drugage_etl.metta"

log "GenAge human → genage_human_etl.metta"
"$PYTHON" genage_human_parser.py \
  --input  data/genage/genage_human.csv \
  --output "$OUT_DIR/genage_human_etl.metta"

log "GenAge models → genage_models_etl.metta"
"$PYTHON" genage_models_parser.py \
  --input  data/genage/genage_models.csv \
  --output "$OUT_DIR/genage_models_etl.metta"

log "CellAge → cellage_genes.metta / cellage_expression.metta / cellage_metadata.metta"
"$PYTHON" cellage_etl.py \
  --curated    data/cellage/cellage3.tsv \
  --expression data/cellage/signatures1.csv \
  --outdir     "$OUT_DIR"

# ── NHANES (optional) ─────────────────────────────────────────────────────────
# NHANES microdata is public but is NOT redistributed in this repo, so these two
# ETLs are SKIPPED unless the files are present under data/nhanes/. See
# docs/nhanes_integration.md and data/nhanes/README.md for how to obtain them —
# the download needs network access to wwwn.cdc.gov and ftp.cdc.gov.
#
# Unlike the HAGR outputs above, staging these under $OUT_DIR is not merely
# tidiness: pln_chat drops a root-level .metta file over PLN_MAX_KB_FILE_BYTES
# from execution with only a print(), so an oversized KB file at the root fails
# silently rather than loudly. The ETLs refuse to write one, but build/ keeps
# them out of the app's auto-load path regardless.
NHANES_DIR="${NHANES_DIR:-$ROOT/data/nhanes}"
# NHANES_CYCLES has NO default, deliberately. It selects the survey weight and is written
# into the emitted provenance, so an unstated cycle would become an unverifiable claim
# about which sample the numbers describe — and NHANES files do not carry their cycle in
# any readable field, so inferring it from a filename would be exactly the sort of guess
# this integration refuses. Set it explicitly, e.g. NHANES_CYCLES=1999-2000,2001-2002.
NHANES_CYCLES="${NHANES_CYCLES:-}"

nhanes_files() { find "$NHANES_DIR" -maxdepth 1 -type f \( -iname '*.XPT' -o -iname '*.dat' -o -iname '*.sas7bdat' \) 2>/dev/null; }

if [[ -d "$NHANES_DIR" ]] && [[ -n "$(nhanes_files)" ]]; then
  log "NHANES reference distributions → nhanes_reference.metta"
  "$PYTHON" nhanes_reference_etl.py \
    --data-dir "$NHANES_DIR" \
    --output   "$OUT_DIR/nhanes_reference.metta" \
    || echo "WARNING: NHANES reference ETL failed — see the message above" >&2

  # The mortality ETL takes explicit file paths (it must pair each DEMO with its own
  # linkage file), unlike the reference ETL's --data-dir. Discover them here rather
  # than invoking it bare, which would exit on an argparse error.
  mapfile -t NH_DEMO < <(find "$NHANES_DIR" -maxdepth 1 -type f -iname 'DEMO*.XPT' | sort)
  mapfile -t NH_MORT < <(find "$NHANES_DIR" -maxdepth 1 -type f -iname '*MORT*.dat' | sort)
  if [[ -z "$NHANES_CYCLES" ]]; then
    log "NHANES linked mortality: skipped (set NHANES_CYCLES, e.g. NHANES_CYCLES=2001-2002)"
  elif [[ ${#NH_DEMO[@]} -gt 0 && ${#NH_MORT[@]} -gt 0 ]]; then
    log "NHANES linked mortality → nhanes_mortality_baseline.metta"
    "$PYTHON" nhanes_mortality_etl.py \
      --demo      "${NH_DEMO[@]}" \
      --mortality "${NH_MORT[@]}" \
      --cycles    "$NHANES_CYCLES" \
      --output    "$OUT_DIR/nhanes_mortality_baseline.metta" \
      || echo "WARNING: NHANES mortality ETL failed — see the message above" >&2
  else
    log "NHANES linked mortality: skipped (need DEMO*.XPT and *MORT*.dat in $NHANES_DIR)"
  fi

  if [[ -f "$NHANES_DIR/DNMEPI.xpt" || -f "$NHANES_DIR/dnmepi.sas7bdat" ]]; then
    log "NHANES DNA methylation clocks → nhanes_dnam_clocks.metta"
    "$PYTHON" nhanes_dnam_etl.py \
      --data-dir "$NHANES_DIR" \
      --output   "$OUT_DIR/nhanes_dnam_clocks.metta" \
      || echo "WARNING: NHANES DNAm ETL failed — see the message above" >&2
  fi
else
  log "NHANES: skipped (no data files under $NHANES_DIR)"
  echo "    NHANES microdata is not bundled with this repo. To generate the" >&2
  echo "    reference-distribution and baseline-risk KBs, fetch the files listed in" >&2
  echo "    data/nhanes/MANIFEST.tsv into $NHANES_DIR and re-run. Details:" >&2
  echo "    docs/nhanes_integration.md" >&2
fi

log "ETL complete — outputs in $OUT_DIR"
ls -1 "$OUT_DIR"/*.metta
