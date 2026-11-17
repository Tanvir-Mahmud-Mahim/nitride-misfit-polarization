#!/bin/bash
# Full DFT chain (about 12 h on two cores): bulk structures, polarization constants, slab test.
set -e
cd "$(dirname "$0")/.."
python3 scripts/dft_relax_bulk.py AlN GaN
python3 scripts/dft_polarization.py AlN GaN
python3 scripts/dft_slab.py AlN GaN
