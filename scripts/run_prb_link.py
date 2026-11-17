"""What the misfit-line charge means for the GaN/AlN hole gas.

In T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, Phys. Rev. B 114, L291304
(2026), the four measured mobilities of the GaN/AlN two-dimensional hole gas
were fitted with interface roughness, threading line charges and a flatter
component, one candidate being charged lines lying in the interface (misfit
dislocations). The fit fixes only the product L_A f^2, where L_A is the line
length per area and f the line charge in units of e per in-plane lattice
constant of GaN (a = 0.3189 nm). The archived fit values are copied in
data/prb2026_misfit_fit.json; that Letter converted them into a relaxation of
0.35 to 0.5 percent of the mismatch by taking f = 1.

Here f is replaced by the line charges of this work. For a line charge Lambda
(e/nm), f = |Lambda| a and the relaxation needed is r = r(f = 1) / f^2 (same
conversion, L_A b / 2 = relieved strain, b = a). The change of the hole-gas
bound charge caused by that relaxation, with lines of charge Lambda, follows
from the decomposition of src/nmp/polarization.py:
Delta sigma = 2 r eps_c (e31_prop - e33 C13/C33) + Lambda L_A
(GaN film on AlN, eps_c = (a_AlN - a_GaN)/a_GaN < 0; for this compressive film
the misfit lines are missing film columns and the bulk-terminated value is
Lambda = +P_H b; for the bound from the Shen data the sign is not known and
the sign that removes holes is taken, the worse case).

Outputs results/prb_link.json.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import materials as M, polarization as PZ  # noqa: E402

ROOT = os.path.join(HERE, "..")
QE = 1.602176634e-19
PS_MEASURED = 4.6e13     # cm^-2, hole density of the GaN/AlN sample analysed in the Letter (stated there)


def main():
    d = M.dft()
    fit = json.load(open(os.path.join(ROOT, "data", "prb2026_misfit_fit.json")))
    a_gan = fit["a_lattice_m"]
    mismatch = fit["mismatch"]
    gan = M.algan(0.0, 300.0, d)
    b = a_gan
    pzb = PZ.p_zb_formal(a_gan * 1e10)
    shen = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]
    bounds = {k: v["mobility"]["Lambda_strict_upper_bound_e_per_nm"] for k, v in shen.items()}
    ests = {k: v["mobility"]["Lambda_estimate_range"][1] for k, v in shen.items()}
    cases = {"f = 1 (assumed in the Letter)": 1.0 / (a_gan * 1e9),
             "bulk-terminated (H, improper)": gan["P_H"] * b / QE * 1e-9,
             "Shen et al., largest strict mobility bound": max(bounds.values()),
             "Shen et al., largest estimate from the mobility drop": max(ests.values()),
             "electron counting": abs(gan["P_H"] - pzb) * b / QE * 1e-9}
    k = gan["e31_prop"] - gan["e33"] * gan["C13_over_C33"]
    eps_c = -mismatch
    out = {"input": fit, "P_ZB_GaN_Cm2": gan["P_H"] - pzb, "k_proper_Cm2": k, "cases": {}}
    for name, lam in cases.items():
        f = abs(lam) * a_gan * 1e9
        rows = {}
        for key, r1 in fit["relaxed_fraction_f1"].items():
            r = r1 / f ** 2
            LA = 2 * r * mismatch / b                                   # m^-1
            sgn = -1.0 if name == "electron counting" else 1.0          # P_ZB < 0
            dsig = 2 * r * eps_c * k + sgn * abs(lam) * QE * 1e9 * LA     # C/m^2
            rows[key] = {"relaxed_fraction": r, "relieved_strain_percent": 100 * r * mismatch,
                         "L_A_per_cm": LA / 100, "line_spacing_nm": 3.0 / LA * 1e9,
                         "delta_bound_charge_cm2": dsig / QE * 1e-4,
                         "delta_relative_to_measured_density": dsig / QE * 1e-4 / PS_MEASURED,
                         "possible": bool(r <= 1.0)}
        out["cases"][name] = {"Lambda_e_per_nm": lam, "f": f, "fits": rows}
    json.dump(out, open(os.path.join(ROOT, "results", "prb_link.json"), "w"), indent=1)
    print(json.dumps(out["cases"], indent=1))


if __name__ == "__main__":
    main()
