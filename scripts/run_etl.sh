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
# DO NOT set OUT_DIR=. to write into the repo root. hyperon 0.2.10 aborts the process
# (SIGABRT, uncatchable) on a match query once one space carries too many DISTINCT HEAD
# SYMBOLS, and the generated NHANES files introduce 12-21 each. The NHANES layers are run
# in their own query-scoped space for exactly this reason (core.pln_runner
# .NHANES_PATIENT_STACK, and docs/nhanes_integration.md section 8); dropping generated
# records into the repo root puts them somewhere that scoping cannot protect.
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
