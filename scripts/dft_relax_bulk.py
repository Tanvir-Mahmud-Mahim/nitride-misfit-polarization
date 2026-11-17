"""Equilibrium structure of wurtzite AlN and GaN with PBEsol (pw.x vc-relax).

Plane-wave cutoffs 60, 70 and 80 Ry for the wave functions (8x for the
density), Monkhorst-Pack 8x8x5 k-points, insulating occupations. The 70 Ry
structure is used everywhere else; the 80 Ry run checks convergence.

Outputs results/dft_bulk.json with a (angstrom), c (angstrom) and u.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import qe  # noqa: E402

START = {"AlN": (3.11, 4.98, 0.382), "GaN": (3.19, 5.19, 0.377)}
CAT = {"AlN": "Al", "GaN": "Ga"}


def relax(name, ecut):
    a, c, u = START[name]
    d = os.path.join(qe.RUNS, "bulk_%s_e%d" % (name, ecut))
    os.makedirs(d, exist_ok=True)
    frac = [(CAT[name] if s == "M" else "N", x) for s, x in qe.wurtzite_positions(u)]
    inp, out = os.path.join(d, "vc.in"), os.path.join(d, "vc.out")
    qe.write_input(inp, "vc-relax", None, qe.hex_cell(a, c), frac, (8, 8, 5),
                   extra_system="  occupations='fixed'\n", extra_ions="  ion_dynamics='bfgs'\n",
                   extra_cell="  cell_dynamics='bfgs'\n  press_conv_thr=0.05\n",
                   params={"ecutwfc": float(ecut), "ecutrho": 8.0 * ecut}, prefix="b")
    if not qe.done(out):
        qe.run(inp, out)
    cell, fr = qe.final_structure(out)
    a = float(np.linalg.norm(cell[0]))
    c = float(np.linalg.norm(cell[2]))
    zc = sorted(f[1][2] for f in fr if f[0] != "N")[0]
    zn = sorted((f[1][2] - zc) % 1.0 for f in fr if f[0] == "N")[0]
    return {"a": a, "c": c, "u": float(zn), "ecutwfc": ecut, "gap_eV": qe.gap(out)}


def main():
    path = os.path.join(qe.ROOT, "results", "dft_bulk.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in sys.argv[1:] or ["AlN", "GaN"]:
        for ecut in (70, 80, 60):
            r = relax(name, ecut)
            out["%s_ecut%d" % (name, ecut)] = r
            print(name, ecut, r, flush=True)
            json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
