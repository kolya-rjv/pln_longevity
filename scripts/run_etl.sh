#!/usr/bin/env bash
#
# Meta-ETL: unpack the raw HAGR data archives and regenerate every MeTTa KB
# file from them. Runs the per-dataset ETL scripts (DrugAge, GenAge human,
# GenAge models, CellAge) against the zips committed under data/, and the NHANES
# ETLs when their microdata is present (it is not redistributed with the repo).
#
# Outputs are written to OUT_DIR (default: ./build, gitignored), and MUST NOT be
# written to the repo root, which this script refuses:
#
#   - the root holds the committed, curated layers, and the NHANES reference
#     ETL's output has the same name as one of them: OUT_DIR=. would overwrite
#     the rules in nhanes_reference.metta with generated records;
#   - the code reads generated records from ./build (the DrugAge and CellAge
#     selectors, and the LinAge2 baseline in core.pln_runner), so in the root
#     they would sit where nothing that needs them looks.
#
# A stray file in the root no longer reaches inference: execution loads only the
# curated stack named in api._INFERENCE_STACK, plus the query-scoped stacks in
# core.pln_runner. When execution loaded every root .metta into one space, a
# generated file there was what the 2026-09-28 re-test crashed on: hyperon 0.2.10
# aborts (a non-unwinding panic, uncatchable) once a space holds too many DISTINCT
# HEAD SYMBOLS, and a generated file adds one per field predicate -- the NHANES
# files 12-21 each. See core/executor.py and tests/test_kb_head_symbol_budget.py.
#
#   scripts/run_etl.sh                 # regenerate into ./build
#   OUT_DIR=/tmp/kb scripts/run_etl.sh # anywhere OUTSIDE the repo root
#   PYTHON=python3.11 scripts/run_etl.sh
#
set -euo pipefail

# Resolve the repo root (this script lives in <root>/scripts) and run from it
# so every relative data path below is stable regardless of the caller's CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-$ROOT/build}"
PYTHON="${PYTHON:-python3}"

# Refuse the repo root outright, before anything is written (see the header).
if [ "$(cd "$OUT_DIR" 2>/dev/null && pwd || echo "$OUT_DIR")" = "$ROOT" ]; then
  echo "run_etl.sh: OUT_DIR must not be the repo root -- it holds the curated" >&2
  echo "  layers (nhanes_reference.metta would be overwritten with generated" >&2
  echo "  records), and the code reads generated output from ./build." >&2
  echo "  Use the default ./build, or any path outside the repo root." >&2
  exit 2
fi

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
# Where these land matters more than for the HAGR outputs above: the reference
# ETL's output has the name of the committed rules file in the root (see the
# header), and the LinAge2 projections read the mortality baseline from build/
# only (core.pln_runner.LINAGE2_GENERATED_BASELINE). The ETLs also refuse to
# write a file over PLN_MAX_KB_FILE_BYTES, which pln_chat would drop silently.
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

  # Case-insensitive, and covering every form the ETL can actually read. A test for two
  # exact spellings missed DNMEPI.XPT — the spelling MANIFEST.tsv itself lists — and
  # skipped the step silently.
  if [[ -n "$(find "$NHANES_DIR" -maxdepth 1 -type f \( -iname 'dnmepi.xpt' -o -iname 'dnmepi.sas7bdat' -o -iname 'dnmepi.csv' \) 2>/dev/null)" ]]; then
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
