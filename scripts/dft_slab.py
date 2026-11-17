"""Electric field inside electron-counting-passivated c-plane slabs (pw.x).

A metal-polar slab of N formula units of AlN or GaN at the bulk PBEsol
structure (results/dft_bulk.json), cut through the bonds along c, is
terminated so that every dangling bond obeys the electron-counting rule: the
top cation (dangling bond along +c, 3/4 electron) is saturated by a
pseudo-hydrogen of nuclear charge 5/4, the bottom nitrogen (dangling bond
along -c, 5/4 electron) by a pseudo-hydrogen of charge 3/4 (pseudopotentials
generated with ld1.x, dft/pseudo/H*.in). Only the two pseudo-hydrogens are
relaxed; the nitride atoms stay at their bulk positions, so the dielectric
response inside the slab is the clamped-ion one, eps_inf.

The planar average of the electrostatic potential (pp.x, plot_num=11) is
macroscopically averaged over one c period (average.x). The field inside the
slab, E_s (central 40 percent), and in the middle of the vacuum, E_v, give the
bare charge on the top face from Gauss's law with electronic screening,
sigma(top) = eps0 (E_v - eps_inf E_s): the screening charge
eps0 (eps_inf - 1) E_s sits on the face together with the bare charge. No
dipole correction is applied (it slowed the self-consistency badly), so the
vacuum field is not zero and is measured instead.

The electron-counting rule removes 3/4 of an electron per surface site from
the bulk-terminated value (Vanderbilt and King-Smith, Phys. Rev. B 48, 4442
(1993), Table I), i.e. e sqrt(3)/(2 a^2) = P_f,ZB(a). The prediction tested
here is sigma = P_H - P_f,ZB(a) (the ZB-referenced spontaneous polarization),
with P_H from results/dft_polarization.json. A bulk-terminated value would be
sigma = P_H, fifteen to forty times larger.

A small Gaussian smearing (0.005 Ry) with empty bands is used for
robust convergence; the band gap of the slab is reported to show that it is
insulating, so the smearing has no effect. eps_inf along c comes from
density-functional perturbation theory (ph.x, epsil=.true.) on the bulk.

Outputs results/dft_slab.json.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import qe  # noqa: E402

CAT = {"AlN": "Al", "GaN": "Ga"}
VAL = {"Al": 3, "Ga": 13, "N": 5, "Hc": 1.25, "Ha": 0.75}
ECUT = dict(ecutwfc=70.0, ecutrho=560.0)
VAC = 16.0                       # angstrom of vacuum
# Slab sizes (formula units). The face charge P_ZB sets up a field inside the
# slab, and the potential drop across it grows with thickness; it must stay
# below the band gap for the slab to remain insulating (checked in the output).
SIZES = {"AlN": (4, 6), "GaN": (4, 6)}
D_H = {"Al": 1.60, "Ga": 1.55}   # starting cation-H distance (angstrom); relaxed
D_HN = 1.03                      # starting N-H distance (angstrom); relaxed

qe.PP["Hc"] = "H1.25.pbesol-nc.UPF"
qe.MASS["Hc"] = 1.008
qe.PP["Ha"] = "H0.75.pbesol-nc.UPF"
qe.MASS["Ha"] = 1.008
PRE = "LD_PRELOAD=%s " % qe.SHIM


def relaxed_h_distances(name):
    """Cation-H and N-H distances (angstrom) from a finished slab of the same material, if any."""
    for n in SIZES[name]:
        out = os.path.join(qe.RUNS, "slab_%s_%d" % (name, n), "relax.out")
        if qe.done(out):
            cell, fr = qe.final_structure(out)
            L = np.linalg.norm(cell[2])
            z = {s: [] for s in ("Hc", "Ha", CAT[name], "N")}
            for sp, x in fr:
                z[sp].append(x[2] * L)
            return z["Hc"][0] - max(z[CAT[name]]), min(z["N"]) - z["Ha"][0]
    return None


def slab(name, nbl, bulk):
    """Metal-polar slab cut through the vertical (c-axis) bonds, so that each face
    atom has a single dangling bond along c: a nitrogen at the bottom (bond
    pointing along -c, 5/4 electron) and a cation at the top (bond along +c,
    3/4 electron). In the bulk the cation of layer k sits at z = k c/2 and its
    nitrogen directly above at k c/2 + u c; the nitrogen below a cation along
    c belongs to layer k-1 and sits at (k-1) c/2 + u c with the other lateral
    site. The slab holds nbl cations and nbl nitrogens."""
    cat = CAT[name]
    dh = relaxed_h_distances(name)
    d_h, d_hn = dh if dh else (D_H[cat], D_HN)
    a, c, u = bulk["a"], bulk["c"], bulk["u"]
    site = lambda k: (1 / 3, 2 / 3) if k % 2 == 0 else (2 / 3, 1 / 3)
    zs = [("N", site(-1), -c / 2 + u * c)]
    for k in range(nbl):
        zs.append((cat, site(k), k * c / 2))
        if k < nbl - 1:
            zs.append(("N", site(k), k * c / 2 + u * c))
    zbot = zs[0][2] - d_hn              # pseudo-hydrogen 3/4 below the bottom nitrogen
    ztop = zs[-1][2] + d_h              # pseudo-hydrogen 5/4 above the top cation
    L = (ztop - zbot) + VAC
    shift = VAC / 2 - zbot
    atoms = [("Ha", zs[0][1], zbot)] + zs + [("Hc", zs[-1][1], ztop)]
    frac = [(s, (x[0], x[1], (z + shift) / L)) for s, x, z in atoms]
    return qe.hex_cell(a, L), frac, L


def bands_gap(out, nocc):
    s = open(out).read()
    i = s.rfind("End of self-consistent calculation")
    blocks = re.findall(r"bands \(ev\):\s*\n\n(.*?)\n\n", s[i:], re.S)
    e = [np.array(b.replace("-", " -").split(), float) for b in blocks]
    vbm = max(x[nocc - 1] for x in e)
    cbm = min(x[nocc] for x in e)
    return float(vbm), float(cbm), float(cbm - vbm)


def run_slab(name, nbl, bulk):
    d = os.path.join(qe.RUNS, "slab_%s_%d" % (name, nbl))
    os.makedirs(d, exist_ok=True)
    cell, frac, L = slab(name, nbl, bulk)
    nel = sum(VAL[s] for s, _ in frac)
    nocc = int(round(nel / 2))
    sysx = "  occupations='smearing'\n  smearing='gaussian'\n  degauss=0.005\n  nbnd=%d\n" % (nocc + 8)
    ctrl = ""
    inp = os.path.join(d, "relax.in")
    qe.write_input(inp, "relax", None, cell, frac, (6, 6, 1), extra_control=ctrl, extra_system=sysx,
                   extra_ions="  ion_dynamics='bfgs'\n", prefix="s", params=dict(ECUT, conv_thr=1e-7),
                   extra_electrons="  mixing_mode='local-TF'\n  mixing_beta=0.2\n  electron_maxstep=400\n")
    # fix all atoms but the two pseudo-hydrogens
    out_lines, in_pos = [], False
    for l in open(inp).read().split("\n"):
        if l.startswith("ATOMIC_POSITIONS"):
            in_pos = True
        elif l.startswith("K_POINTS"):
            in_pos = False
        elif in_pos and l.strip() and not l.strip().startswith("H"):
            l = l + " 0 0 0"
        out_lines.append(l)
    txt = "\n".join(out_lines)
    # slab: looser force and energy thresholds for the two relaxed pseudo-hydrogens
    txt = txt.replace("forc_conv_thr=1.0d-5", "forc_conv_thr=1.0d-3").replace("etot_conv_thr=1.0d-8", "etot_conv_thr=1.0d-5")
    open(inp, "w").write(txt)
    out = os.path.join(d, "relax.out")
    if not qe.done(out):
        qe.run(inp, out)
    vbm, cbm, gap = bands_gap(out, nocc)
    open(os.path.join(d, "pp.in"), "w").write(
        "&inputpp\n  prefix='s'\n  outdir='./tmp'\n  filplot='pot.dat'\n  plot_num=11\n/\n")
    subprocess.run(PRE + "pp.x -in pp.in > pp.out 2>&1", shell=True, cwd=d, check=True)
    c = bulk["c"]
    open(os.path.join(d, "avg.in"), "w").write("1\npot.dat\n1.0\n3000\n3\n%.6f\n" % (c / qe.BOHR))
    subprocess.run(PRE + "average.x < avg.in > avg.out 2>&1", shell=True, cwd=d, check=True)
    data = np.loadtxt(os.path.join(d, "avg.dat"))
    z = data[:, 0] * qe.BOHR                    # angstrom
    vplan = data[:, 1] * 13.605693122994        # Ry -> eV (potential energy of an electron)
    vmac = data[:, 2] * 13.605693122994
    cell_out, fr = qe.final_structure(out)
    L = float(np.linalg.norm(cell_out[2]))      # the cell actually computed
    zat = sorted(f[1][2] * L for f in fr if f[0] in (CAT[name], "N"))
    z0, z1 = zat[0], zat[-1]
    m0, m1 = z0 + 0.3 * (z1 - z0), z0 + 0.7 * (z1 - z0)
    sel = (z > m0) & (z < m1)
    slope = np.polyfit(z[sel], vmac[sel], 1)[0]      # eV / angstrom = e E
    # vacuum: the vacuum wraps around the cell boundary; use 6 angstrom on each side of it
    zz = np.where(z > L / 2, z - L, z)
    vsel = np.abs(zz) < 6.0
    slope_v = np.polyfit(zz[vsel], vplan[vsel], 1)[0]
    np.savetxt(os.path.join(d, "potential_profile.txt"), np.c_[z, vplan, vmac],
               header="z (A), planar average (eV), macroscopic average over c (eV)")
    return {"nbilayers": nbl, "L_A": L, "slope_eV_per_A": float(slope),
            "field_V_per_m": float(slope * 1e10), "central_region_A": [float(m0), float(m1)],
            "vacuum_field_V_per_m": float(slope_v * 1e10),
            "periodicity_check_V": float(slope * (z1 - z0) + slope_v * (L - (z1 - z0))),
            "nitride_extent_A": [float(z0), float(z1)],
            "vbm_eV": vbm, "cbm_eV": cbm, "gap_eV": gap, "n_electrons": nel}


def eps_inf(name, bulk):
    d = os.path.join(qe.RUNS, "eps_" + name)
    os.makedirs(d, exist_ok=True)
    frac = [(CAT[name] if s == "M" else "N", x) for s, x in qe.wurtzite_positions(bulk["u"])]
    inp = os.path.join(d, "scf.in")
    qe.write_input(inp, "scf", None, qe.hex_cell(bulk["a"], bulk["c"]), frac, (8, 8, 5),
                   extra_system="  occupations='fixed'\n", prefix="e", params=ECUT)
    out = os.path.join(d, "scf.out")
    if not qe.done(out):
        qe.run(inp, out)
    open(os.path.join(d, "ph.in"), "w").write(
        "eps\n&inputph\n  prefix='e'\n  outdir='./tmp'\n  epsil=.true.\n  trans=.false.\n"
        "  tr2_ph=1.0d-16\n  fildyn='dyn'\n/\n0.0 0.0 0.0\n")
    pho = os.path.join(d, "ph.out")
    if not qe.done(pho):
        subprocess.run(PRE + "mpirun --allow-run-as-root -x LD_PRELOAD -np 2 ph.x -in ph.in > ph.out 2>&1",
                       shell=True, cwd=d, check=True)
    s = open(pho).read()
    m = re.search(r"Dielectric constant in cartesian axis\s*\n\s*\n((?:\s*\(.*\)\s*\n){3})", s)
    rows = [[float(x) for x in l.replace("(", "").replace(")", "").split()]
            for l in m.group(1).strip().split("\n")]
    return {"eps_xx": rows[0][0], "eps_zz": rows[2][2]}


def main():
    bulk = json.load(open(os.path.join(qe.ROOT, "results", "dft_bulk.json")))
    pol = json.load(open(os.path.join(qe.ROOT, "results", "dft_polarization.json")))
    path = os.path.join(qe.ROOT, "results", "dft_slab.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    eps0 = 8.8541878128e-12
    for name in sys.argv[1:] or ["AlN", "GaN"]:
        b = bulk[name + "_ecut70"]
        e = eps_inf(name, b)
        runs = [run_slab(name, n, b) for n in SIZES[name]]
        a = b["a"] * 1e-10
        pzb = qe.E_CHARGE * np.sqrt(3) / (2 * a * a)
        sig = [eps0 * (r["vacuum_field_V_per_m"] - e["eps_zz"] * r["field_V_per_m"]) for r in runs]
        out[name] = {"eps_inf": e, "slabs": runs, "sigma_from_field_Cm2": sig,
                     "P_H": pol[name]["P_H"], "P_f_ZB": float(pzb),
                     "prediction_P_H_minus_P_f_ZB": float(pol[name]["P_H"] - pzb)}
        print(name, json.dumps(out[name]), flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
