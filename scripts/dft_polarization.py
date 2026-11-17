"""Polarization constants of wurtzite AlN and GaN from Berry-phase DFT (pw.x, PBEsol).

1. Spontaneous polarization, layered-hexagonal (H) reference. At the relaxed a
   and c, the internal parameter u is moved linearly from its equilibrium value
   to 1/2, where the crystal becomes the centrosymmetric layered hexagonal
   structure with zero formal polarization. The Berry-phase polarization
   (modulo the quantum Q = e c / Omega) is followed along the 13-point path and
   unwrapped; P_H = P(u_eq) - P(1/2). The band gap along the path is recorded
   to show that the path stays insulating.
2. Zinc-blende (ZB) reference: an ideal ZB crystal in a six-atom hexagonal
   cell with the wurtzite in-plane constant a (c = sqrt(6) a). Its formal
   polarization modulo Q is compared with e sqrt(3)/(2 a^2) = (3/4) Q
   (Dreyer et al., Phys. Rev. X 6, 021038 (2016)).
3. Piezoelectric constants with relaxed ions. Biaxial in-plane strain
   +-0.5 % and +-1 % at fixed c gives dP/d eps1 = 2 e31 on the branch of the
   computed P; the proper constant is e31_prop = e31_branch + P_branch
   (Vanderbilt, J. Phys. Chem. Solids 61, 147 (2000), Eq. (15)), and the
   improper constant on the H branch is e31_prop - P_H. Strain of c by
   +-0.5 % and +-1 % at fixed a gives e33 (proper and improper coincide).
4. C13/C33 from plane stress: biaxial strain +-1 % with c and u relaxed
   (cell_dofree='z'); C13/C33 = -eps3/(2 eps1).

Settings: 70 Ry (560 Ry for the density), scf k-mesh 6x6x4, Berry phase on a
6x6x10 mesh with 10 points per string along c (nscf, one pool).
Outputs results/dft_polarization.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import qe  # noqa: E402

ECUT = {"ecutwfc": 70.0, "ecutrho": 560.0}
KSCF = (6, 6, 4)
CAT = {"AlN": "Al", "GaN": "Ga"}
STRAINS = (-0.01, -0.005, 0.005, 0.01)
NPATH = 13


def berry(d, cell, frac, pre="p"):
    """scf + Berry-phase nscf in directory d; returns (P raw, quantum, gap)."""
    os.makedirs(d, exist_ok=True)
    scf, nscf = os.path.join(d, "scf.in"), os.path.join(d, "berry.in")
    so, bo = os.path.join(d, "scf.out"), os.path.join(d, "berry.out")
    nb = int(round(sum(5 if s == "N" else (3 if s == "Al" else 13) for s, _ in frac) / 2))
    qe.write_input(scf, "scf", None, cell, frac, KSCF, prefix=pre, params=ECUT,
                   extra_system="  occupations='fixed'\n  nbnd=%d\n" % (nb + 4))
    if not qe.done(so):
        qe.run(scf, so)
    qe.write_input(nscf, "nscf", None, cell, frac, (6, 6, 10), prefix=pre, params=ECUT,
                   extra_control="  lberry=.true.\n  gdir=3\n  nppstr=10\n",
                   extra_system="  occupations='fixed'\n")
    if not qe.done(bo):
        qe.run(nscf, bo, pools=1)
    P, q = qe.berry_phase_c(bo)
    return P, q, qe.gap(so)


def relax_fixed_cell(d, cell, frac, pre="r"):
    os.makedirs(d, exist_ok=True)
    inp, out = os.path.join(d, "relax.in"), os.path.join(d, "relax.out")
    qe.write_input(inp, "relax", None, cell, frac, KSCF, prefix=pre, params=ECUT,
                   extra_system="  occupations='fixed'\n", extra_ions="  ion_dynamics='bfgs'\n")
    if not qe.done(out):
        qe.run(inp, out)
    return qe.final_structure(out)


def unwrap(values, q):
    out = [values[0]]
    for v in values[1:]:
        k = np.round((v - out[-1]) / q)
        out.append(v - k * q)
    return out


def wz(name, u):
    cat = CAT[name]
    return [(cat if s == "M" else "N", x) for s, x in qe.wurtzite_positions(u)]


def u_of(frac):
    zc = sorted(f[1][2] for f in frac if f[0] != "N")[0]
    return sorted((f[1][2] - zc) % 1.0 for f in frac if f[0] == "N")[0]


def compute(name, bulk):
    a, c, u0 = bulk["a"], bulk["c"], bulk["u"]
    base = os.path.join(qe.RUNS, "pol_" + name)
    res = {"a": a, "c": c, "u": u0}
    cell = qe.hex_cell(a, c)
    # 1. path to the layered hexagonal structure
    us = np.linspace(u0, 0.5, NPATH)
    path = []
    for i, u in enumerate(us):
        P, q, g = berry(os.path.join(base, "path_%02d" % i), cell, wz(name, u))
        path.append({"u": float(u), "P_raw": P, "gap_eV": g})
        print(name, "path", i, u, P, g, flush=True)
    pw = unwrap([p["P_raw"] for p in path], q)
    for p, v in zip(path, pw):
        p["P_unwrapped"] = v
    PH = pw[0] - pw[-1]
    res["quantum_Cm2"] = q
    res["path"] = path
    res["P_H"] = PH
    res["P_LH_mod_Q"] = float(pw[-1] % q)
    res["min_gap_on_path_eV"] = min(p["gap_eV"] for p in path if p["gap_eV"] is not None)
    # 2. zinc-blende reference (six-atom hexagonal cell)
    czb = np.sqrt(6) * a
    sites = [(0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]
    frac = []
    for i, (x, y) in enumerate(sites):
        frac.append((CAT[name], (x, y, i / 3)))
        frac.append(("N", (x, y, i / 3 + 0.25)))
    Pz, qz, gz = berry(os.path.join(base, "zb"), qe.hex_cell(a, czb), frac, pre="z")
    res["zb"] = {"P_raw": Pz, "quantum": qz, "P_mod_Q": float(Pz % qz),
                 "e_sqrt3_over_2a2": float(qe.E_CHARGE * np.sqrt(3) / (2 * (a * 1e-10) ** 2)),
                 "gap_eV": gz}
    # 3. C13/C33 (plane stress, c and u relaxed)
    ratios = []
    for e in (-0.01, 0.01):
        d = os.path.join(base, "biax_crelax_%+.3f" % e)
        os.makedirs(d, exist_ok=True)
        inp, out = os.path.join(d, "vc.in"), os.path.join(d, "vc.out")
        qe.write_input(inp, "vc-relax", None, qe.hex_cell(a * (1 + e), c), wz(name, u0), KSCF, prefix="c",
                       params=ECUT, extra_system="  occupations='fixed'\n",
                       extra_ions="  ion_dynamics='bfgs'\n",
                       extra_cell="  cell_dynamics='bfgs'\n  cell_dofree='z'\n  press_conv_thr=0.05\n")
        if not qe.done(out):
            qe.run(inp, out)
        cl, _ = qe.final_structure(out)
        e3 = np.linalg.norm(cl[2]) / c - 1
        ratios.append({"eps1": e, "eps3": float(e3), "C13_over_C33": float(-e3 / (2 * e))})
    res["C13_ratio_runs"] = ratios
    res["C13_over_C33"] = float(np.mean([r["C13_over_C33"] for r in ratios]))

    # 4. piezoelectric constants with relaxed ions
    def series(kind):
        pts = [(0.0, path[0]["P_raw"], u0)]
        for e in STRAINS:
            cl = qe.hex_cell(a * (1 + e), c) if kind == "e31" else qe.hex_cell(a, c * (1 + e))
            d = os.path.join(base, "%s_%+.3f" % (kind, e))
            cr, fr = relax_fixed_cell(os.path.join(d, "relax"), cl, wz(name, u0))
            P, qq, _ = berry(d, cl, fr)
            pts.append((e, P, float(u_of(fr))))
            print(name, kind, e, P, flush=True)
        pts.sort()
        eps = np.array([p[0] for p in pts])
        P0 = path[0]["P_raw"]
        # quantum changes with a for e31: unwrap each point to the branch nearest P0
        Pb = []
        for e, P, _ in pts:
            qq = q / (1 + e) ** 2 if kind == "e31" else q
            Pb.append(P - np.round((P - P0) / qq) * qq)
        Pb = np.array(Pb)
        fit = np.polyfit(eps, Pb, 2)
        return {"eps": eps.tolist(), "P_branch": Pb.tolist(), "u": [p[2] for p in pts],
                "slope": float(fit[1]), "curvature": float(fit[0]),
                "slope_linear_05": float((Pb[3] - Pb[1]) / 0.01)}
    s31 = series("e31")
    s33 = series("e33")
    res["e31_series"] = s31
    res["e33_series"] = s33
    res["P_branch_used"] = path[0]["P_raw"]
    res["e31_branch"] = s31["slope"] / 2.0
    res["e31_proper"] = s31["slope"] / 2.0 + path[0]["P_raw"]
    res["e31_improper_H"] = res["e31_proper"] - PH
    res["e33"] = s33["slope"]
    return res


def main():
    bulk = json.load(open(os.path.join(qe.ROOT, "results", "dft_bulk.json")))
    path = os.path.join(qe.ROOT, "results", "dft_polarization.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in sys.argv[1:] or ["AlN", "GaN"]:
        r = compute(name, bulk[name + "_ecut70"])
        out = json.load(open(path)) if os.path.exists(path) else {}
        out[name] = r
        json.dump(out, open(path, "w"), indent=1)
        print(name, "P_H %.4f  Q %.4f  min gap %.2f  ZB P mod Q %.4f (e sqrt3/2a^2 %.4f)  e31 prop %.4f  "
              "imp(H) %.4f  e33 %.4f  C13/C33 %.4f"
              % (r["P_H"], r["quantum_Cm2"], r["min_gap_on_path_eV"], r["zb"]["P_mod_Q"],
                 r["zb"]["e_sqrt3_over_2a2"], r["e31_proper"], r["e31_improper_H"], r["e33"],
                 r["C13_over_C33"]), flush=True)


if __name__ == "__main__":
    main()
