"""A third series with the Shen design: Jiang et al., Appl. Phys. Lett. 79, 374
(2001). Shubnikov-de Haas densities at 1.4 K of Al0.22Ga0.78N/GaN samples grown
with the recipe of Shen et al. (data/jiang2001_table1.json).

These densities count only electrons in the 2DEG subbands; a parallel channel
of low mobility does not oscillate. The samples are probably not those of
Shen et al. (see the notes in the data file), so the series is used as a
control and not combined with Shen's sample by sample.

The script compares the Shubnikov-de Haas densities with Shen's Hall densities
at the same thickness, and compares the 75 and 100 nm densities with the 77 K model of
scripts/run_shen.py (no barrier conduction) calibrated on Jiang's 25 and 50 nm
samples at each phi_B of the grid, and gives the relaxation that would lower
the 100 nm density by 10 percent for each line charge. The relaxation of
these samples was not measured. Outputs results/jiang_analysis.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_shen as S  # noqa: E402
from nmp import materials as M  # noqa: E402

ROOT = os.path.join(HERE, "..")
T = 77.0


def main():
    d = M.dft()
    J = json.load(open(os.path.join(ROOT, "data", "jiang2001_table1.json")))
    sh = json.load(open(os.path.join(ROOT, "data", "shen2000_fig2_digitized.json")))
    ts = J["t_nm"]
    nJ = np.array([a + (b or 0.0) for a, b in zip(J["n0_1e12"], J["n1_1e12"])]) * 1e12
    cmp_rows = []
    for t, n in zip(ts, nJ):
        i = sh["thickness_nm"].index(t)
        cmp_rows.append({"t_nm": t, "jiang_SdH_1.4K": n, "shen_Hall_77K": sh["ns_77K_1e13cm2"][i] * 1e13,
                         "shen_Hall_300K": sh["ns_300K_1e13cm2"][i] * 1e13})
    out = {"description": __doc__.split("\n\n")[0], "comparison": cmp_rows, "phi": {}}
    strained = [nJ[0], nJ[1]]
    out["strained_mean_ratio_to_shen"] = {
        "77K": float(np.mean([r["jiang_SdH_1.4K"] / r["shen_Hall_77K"] for r in cmp_rows[:2]])),
        "300K": float(np.mean([r["jiang_SdH_1.4K"] / r["shen_Hall_300K"] for r in cmp_rows[:2]]))}
    gan, bar, eps_c = S.setup(T, d)
    lims = S.limits(bar)
    shen = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]["77K_mu2_0"]
    lims["Shen fraction"] = shen["Lambda_fit_e_per_nm"] / shen["predictions_t100"][
        "bulk-terminated (H, improper)"]["Lambda_e_per_nm"] * lims[S.LIMITS[0]]
    for phi in (0.5, 1.0, 1.5, 2.0):
        def cost(ds):
            return sum(np.log(S.model(t, T, phi, gan, bar, eps_c, dsig=ds)[0] / n)
                       for t, n in zip(ts[:2], strained))
        ds = brentq(cost, -0.05, 0.05, xtol=1e-7)
        row = {"dsig_cm2": ds / S.QE * 1e-4,
               "fit": [S.model(t, T, phi, gan, bar, eps_c, dsig=ds)[0] for t in ts]}
        # Jiang's 75 and 100 nm densities are at or above the strained model, so no
        # loss of electrons has to be explained. As a bound, the relaxation that
        # would lower the 100 nm density by 10 percent (about the scatter of the
        # strained samples about the model) for each line charge:
        need = {}
        n_coh = S.model(100.0, T, phi, gan, bar, eps_c, dsig=ds)[0]
        for name, lam in lims.items():
            g = lambda R: S.model(100.0, T, phi, gan, bar, eps_c, R=R, Lam=lam, dsig=ds)[0] - 0.9 * n_coh
            need[name] = float(brentq(g, 1e-4, 1.0, xtol=1e-4)) if g(1.0) < 0 else None
        row["R_for_10pct_drop_100nm"] = need
        row["measured_over_model"] = [float(n / m) for n, m in zip(nJ, row["fit"])]
        out["phi"][str(phi)] = row
        print(phi, json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "jiang_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
