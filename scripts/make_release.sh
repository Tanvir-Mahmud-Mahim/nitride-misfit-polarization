#!/bin/bash
# Build the public release tree (code, data, results, DFT inputs and outputs),
# without the paper and without generated figures.
#   bash scripts/make_release.sh OUTDIR
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: make_release.sh OUTDIR}"
REL="$OUT/nitride-misfit-polarization"
rm -rf "$REL"
mkdir -p "$REL"
cd "$ROOT"
for f in README.md LICENSE CITATION.cff CHANGELOG.md requirements.txt .gitignore; do cp "$f" "$REL/"; done
mkdir -p "$REL/src/nmp" "$REL/scripts" "$REL/tests" "$REL/data" "$REL/results" "$REL/dft/pseudo"
cp src/nmp/*.py "$REL/src/nmp/"
cp scripts/*.py scripts/*.sh "$REL/scripts/"
cp tests/*.py "$REL/tests/"
cp data/*.json "$REL/data/"
for f in results/*.json; do
  case "$f" in *_partial.json) ;; *) cp "$f" "$REL/results/";; esac
done
cp dft/snprintf_shim.c "$REL/dft/"
cp dft/pseudo/*.UPF dft/pseudo/*.in "$REL/dft/pseudo/"
# DFT runs: inputs, outputs and the averaged potentials; no wave functions or densities
tar czf "$REL/dft/dft_runs.tar.gz" --exclude='*/tmp' --exclude='pot.dat' --exclude='avg.dat' \
    --exclude='*.wfc*' --exclude='*.save' -C dft runs
echo "release tree in $REL"
find "$REL" -type f | wc -l
du -sh "$REL"
