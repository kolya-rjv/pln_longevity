#!/usr/bin/env bash
#
# Meta-ETL: unpack the raw HAGR data archives and regenerate every MeTTa KB
# file from them. Runs the per-dataset ETL scripts (DrugAge, GenAge human,
# GenAge models, CellAge) against the zips committed under data/.
#
# Outputs are written to OUT_DIR (default: ./build), and MUST NOT be written to
# the repo root. The app auto-discovers every *.metta at the root and loads it
# into one hyperon space, and hyperon 0.2.10 aborts the interpreter (a
# non-unwinding Rust panic, uncatchable) once that space holds too many DISTINCT
# HEAD SYMBOLS. Measured on the shipped KB: 400 atoms under ONE new head symbol
# are fine, while 4 atoms under 4 NEW head symbols abort it. Every generated
# file here introduces a head symbol per field predicate, so dropping one into
# the root takes every inference query to HTTP 500 — a whole-space failure that
# the per-file byte cap (PLN_MAX_KB_FILE_BYTES) does not bound.
#
# Staging in ./build (gitignored) keeps generated data off the auto-load path.
# It stays queryable: ./build is what the DrugAge selector reads.
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

# Refuse the repo root outright. This was a documented invocation until it was
# found to be the one that takes the running app down (see the header).
if [ "$(cd "$OUT_DIR" 2>/dev/null && pwd || echo "$OUT_DIR")" = "$ROOT" ]; then
  echo "run_etl.sh: OUT_DIR must not be the repo root -- the app auto-loads" >&2
  echo "  every *.metta there into one hyperon space, and a generated file's" >&2
  echo "  new head symbols abort the interpreter on the next inference query." >&2
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

log "ETL complete — outputs in $OUT_DIR"
ls -1 "$OUT_DIR"/*.metta
