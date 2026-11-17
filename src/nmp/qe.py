"""Small driver for Quantum ESPRESSO pw.x (version 6.7, Ubuntu package).

Pseudopotentials (dft/pseudo): PBEsol ultrasoft potentials generated with
ld1.x from the input files of pslibrary (A. Dal Corso, Comput. Mater. Sci. 95,
337 (2014); github.com/dalcorso/pslibrary): Al.pbesol-nl-rrkjus_psl.1.0.0,
Ga.pbesol-dnl-rrkjus_psl.1.0.0 (3d in valence), N.pbesol-n-rrkjus_psl.0.1
(pslibrary notes that N 1.0.0 does not work with PBEsol). Fractional-charge
pseudo-hydrogens (Z = 0.75, 1.25) are norm-conserving, generated with ld1.x
(dft/pseudo/H*.in).
"""

import os
import re
import subprocess

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PSEUDO = os.path.join(ROOT, "dft", "pseudo")
RUNS = os.path.join(ROOT, "dft", "runs")
SHIM = "/opt/shim/libsnp.so"
BOHR = 0.529177210903          # angstrom
E_CHARGE = 1.602176634e-19

PP = {"Al": "Al.pbesol-nl-rrkjus_psl.1.0.0.UPF",
      "Ga": "Ga.pbesol-dnl-rrkjus_psl.1.0.0.UPF",
      "N": "N.pbesol-n-rrkjus_psl.0.1.UPF"}
MASS = {"Al": 26.9815, "Ga": 69.723, "N": 14.007}
DEFAULT = {"ecutwfc": 50.0, "ecutrho": 400.0, "conv_thr": 1e-10}


def wurtzite_positions(u):
    """Metal-polar wurtzite (fractional): cation 'M' at z = 0, anion directly above at z = u."""
    return [("M", (1 / 3, 2 / 3, 0.0)), ("N", (1 / 3, 2 / 3, u)),
            ("M", (2 / 3, 1 / 3, 0.5)), ("N", (2 / 3, 1 / 3, 0.5 + u))]


def hex_cell(a, c):
    return np.array([[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]])


def write_input(path, calc, species, cell, frac, kpts, extra_control="",
                extra_system="", extra_ions="", extra_cell="", params=None,
                prefix="x", outdir="./tmp", extra_electrons=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    els = sorted(set(s for s, _ in frac))
    lines = []
    lines.append("&control\n  calculation='%s'\n  prefix='%s'\n  outdir='%s'\n  pseudo_dir='%s'\n"
                 "  tprnfor=.true.\n  tstress=.true.\n  forc_conv_thr=1.0d-5\n  etot_conv_thr=1.0d-8\n%s/\n"
                 % (calc, prefix, outdir, PSEUDO, extra_control))
    lines.append("&system\n  ibrav=0\n  nat=%d\n  ntyp=%d\n  ecutwfc=%.1f\n  ecutrho=%.1f\n%s/\n"
                 % (len(frac), len(els), p["ecutwfc"], p["ecutrho"], extra_system))
    lines.append("&electrons\n  conv_thr=%.1e\n%s/\n" % (p["conv_thr"], extra_electrons or "  mixing_beta=0.4\n"))
    if calc in ("relax", "vc-relax"):
        lines.append("&ions\n%s/\n" % extra_ions)
    if calc == "vc-relax":
        lines.append("&cell\n%s/\n" % extra_cell)
    lines.append("ATOMIC_SPECIES\n" + "".join("  %s %.4f %s\n" % (e, MASS[e], PP[e]) for e in els))
    lines.append("CELL_PARAMETERS angstrom\n" + "".join("  %.10f %.10f %.10f\n" % tuple(v) for v in cell))
    lines.append("ATOMIC_POSITIONS crystal\n" + "".join("  %s %.10f %.10f %.10f\n" % (s, *x) for s, x in frac))
    if isinstance(kpts, str):
        lines.append(kpts)
    else:
        lines.append("K_POINTS automatic\n  %d %d %d 0 0 0\n" % tuple(kpts))
    with open(path, "w") as f:
        f.write("".join(lines))


def run(inp, out, np_=2, cwd=None, pools=None):
    cwd = cwd or os.path.dirname(inp)
    # LD_PRELOAD: see dft/snprintf_shim.c (false-positive fortify abort of the
    # Ubuntu 24.04 build of pw.x 6.7; the shim keeps the safe bound).
    pre = "LD_PRELOAD=%s " % SHIM if os.path.exists(SHIM) else ""
    cmd = ("%smpirun --allow-run-as-root -x LD_PRELOAD -np %d pw.x -nk %d -in %s > %s 2>&1"
           % (pre, np_, np_ if pools is None else pools, os.path.basename(inp), os.path.basename(out)))
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("pw.x failed: see " + out)
    return parse(out)


def done(out):
    return os.path.exists(out) and "JOB DONE" in open(out).read()


def parse(out):
    s = open(out).read()
    res = {}
    m = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+) Ry", s)
    res["etot_Ry"] = float(m[-1]) if m else None
    m = re.findall(r"highest occupied, lowest unoccupied level \(ev\):\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", s)
    if m:
        res["homo_lumo_eV"] = [float(m[-1][0]), float(m[-1][1])]
    res["done"] = "JOB DONE" in s
    return res


def final_structure(out):
    """Last CELL_PARAMETERS (angstrom) and ATOMIC_POSITIONS (crystal) of a relax/vc-relax run.

    For a fixed-cell relax the cell is read from the input echo (lattice vectors)."""
    s = open(out).read()
    cells = re.findall(r"CELL_PARAMETERS \((alat=\s*([\d.]+)|angstrom)\)\n(.*?\n.*?\n.*?\n)", s)
    if cells:
        head, alat, body = cells[-1]
        cell = np.array([[float(x) for x in l.split()] for l in body.strip().split("\n")])
        if alat:
            cell = cell * float(alat) * BOHR
    else:
        alat = float(re.search(r"lattice parameter \(alat\)\s+=\s+([\d.]+)", s).group(1)) * BOHR
        v = re.findall(r"a\(\d\) = \(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)", s)[:3]
        cell = np.array([[float(x) for x in r] for r in v]) * alat
    blocks = re.findall(r"ATOMIC_POSITIONS \(crystal\)\n(.*?)(?:\n\n|End final)", s, re.S)
    frac = []
    for l in blocks[-1].strip().split("\n"):
        t = l.split()
        if len(t) >= 4:
            frac.append((t[0], tuple(float(x) for x in t[1:4])))
    return cell, frac


def berry_phase_c(out):
    """Electronic + ionic polarization along the Berry direction: (P, quantum) in C/m^2."""
    s = open(out).read()
    m = re.findall(r"P\s+=\s+(-?\d+\.\d+)\s+\(mod\s+(-?\d+\.\d+)\)\s+C/m\^2", s)
    if not m:
        raise RuntimeError("no Berry-phase polarization in " + out)
    return float(m[-1][0]), float(m[-1][1])


def gap(out):
    h = parse(out).get("homo_lumo_eV")
    return None if h is None else h[1] - h[0]
