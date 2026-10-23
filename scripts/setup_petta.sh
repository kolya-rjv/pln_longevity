#!/bin/sh
# Pinned checkouts for the PeTTa migration. Idempotent; runs nothing from the clones.
#   .petta/PeTTa               trueagi-io/PeTTa            (MeTTa compiled to SWI-Prolog)
#   .petta/PeTTa/repos/PLN     patham9/PLN main            (the chainer Ben asked for; 24 commits behind trueagi-io)
#   .petta/PLN-trueagi         trueagi-io/PLN main         (March 2026 single-file rewrite; comparison only)
# Requires: SWI-Prolog >= 9.3 (brew install swi-prolog) and, in the venv, janus-swi + sexpdata.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PETTA_SHA=fea83178af3722666d2948d074591d10a3134f56     # 2026-10-15
FORK_SHA=55f1751d993f71b8a24da03e3aec94ab40789a59      # 2025-08-15, patham9/PLN main
TRUEAGI_SHA=4405956947c4b53c7ff01bd565aa3b114bc970a1   # 2026-03-15, trueagi-io/PLN main

pin() { # url dir sha
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin
  git -C "$2" checkout --quiet --detach "$3"
  printf '%s @ %s\n' "$2" "$(git -C "$2" rev-parse --short HEAD)"
}
mkdir -p "$ROOT/.petta"
pin https://github.com/trueagi-io/PeTTa.git "$ROOT/.petta/PeTTa" "$PETTA_SHA"
mkdir -p "$ROOT/.petta/PeTTa/repos"
pin https://github.com/patham9/PLN.git "$ROOT/.petta/PeTTa/repos/PLN" "$FORK_SHA"
pin https://github.com/trueagi-io/PLN.git "$ROOT/.petta/PLN-trueagi" "$TRUEAGI_SHA"
command -v swipl >/dev/null || { echo "swipl not found: brew install swi-prolog" >&2; exit 1; }
swipl --version
