"""Where the extra 300 K electrons of the Shen series could come from.

At 300 K every sample of Shen et al. (Appl. Phys. Lett. 76, 2746 (2000)) shows
a higher Hall density than at 77 K, by 0.4-0.5e13 cm^-2 in the strained
samples and 0.1-0.2e13 cm^-2 in the relaxed ones. Polarization charge does not
change with temperature, so this script tests one explanation: a single
parallel channel, the same in all six samples (for example in the GaN buffer,
which was grown the same way for the whole series), that conducts at 300 K
and is frozen out at 77 K.

Step 1 uses only the data. A Hall measurement on two channels returns
n_H = (n1 mu1 + n2 mu2)^2 / (n1 mu1^2 + n2 mu2^2) and
mu_H = (n1 mu1^2 + n2 mu2^2) / (n1 mu1 + n2 mu2). For each trial channel
(n_p, mu_p) on a grid the 300 K pairs (n_H, mu_H) are inverted for the 2DEG
(n1, mu1), and n1 is compared with the 77 K density of the same sample. The
grid point with the smallest rms log difference over all six samples is kept,
and the region where the rms is below 10 percent is reported.

Step 2 repeats step 1 with, in addition, the conduction of the undepleted
doped barrier that the model of scripts/run_shen.py predicts at 300 K (its
sheet density from results/shen_analysis.json, mobility 83 cm^2/Vs). The
inversion stays exact, because the sums n mu and n mu^2 are additive.

Step 3 removes the best channel of step 1 from the 300 K Hall data and
repeats the 300 K analysis of scripts/run_shen.py with no barrier channel
(mu2 = 0), as at 77 K.

Outputs results/shen_300K.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import run_shen as S  # noqa: E402
from nmp import materials as M  # noqa: E402

ROOT = os.path.join(HERE, "..")


def scan(nH, mH, n77):
    rows = []
    for n_p in np.logspace(11, 14, 241):
        for mu_p in np.logspace(0, 4, 321):
            n1 = np.array([S.split(nH[i], mH[i], n_p, mu_p)[0] for i in range(len(nH))])
            if not np.all(np.isfinite(n1)):
                continue
            r = np.log(n1 / n77)
            rows.append((float(np.sqrt(np.mean(r ** 2))), float(n_p), float(mu_p), r.tolist()))
    rows.sort(key=lambda t: t[0])
    return rows


def main():
    data = json.load(open(os.path.join(ROOT, "data", "shen2000_fig2_digitized.json")))
    nH = np.array(data["ns_300K_1e13cm2"]) * 1e13
    mH = np.array(data["mu_300K_cm2Vs"], dtype=float)
    n77 = np.array(data["ns_77K_1e13cm2"]) * 1e13
    rows = scan(nH, mH, n77)
    best = rows[0]
    good = [r for r in rows if r[0] < 0.10]
    out = {"description": __doc__.split("\n\n")[0],
           "ratio_300K_over_77K": (nH / n77).tolist(),
           "no_channel_rms_log": float(np.sqrt(np.mean(np.log(nH / n77) ** 2))),
           "best": {"rms_log": best[0], "n_p_cm2": best[1], "mu_p_cm2Vs": best[2],
                    "log_n1_over_n77": best[3]},
           "rms_below_0.10": {"n_p_range_cm2": [min(r[1] for r in good), max(r[1] for r in good)],
                              "mu_p_range_cm2Vs": [min(r[2] for r in good), max(r[2] for r in good)],
                              "n_points": len(good)}}
    b300 = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]["300K_mu2_83"]
    n2 = np.array([smp["n_barrier_model"] for smp in b300["samples"]])
    A0 = nH * mH - n2 * S.MU_BARRIER_300
    B0 = nH * mH ** 2 - n2 * S.MU_BARRIER_300 ** 2
    rows2 = []
    for n_p in np.logspace(10, 14, 241):
        for mu_p in np.logspace(0, 4, 321):
            A, B = A0 - n_p * mu_p, B0 - n_p * mu_p ** 2
            if np.any(A <= 0) or np.any(B <= 0):
                continue
            r = np.log(A * A / B / n77)
            rows2.append((float(np.sqrt(np.mean(r ** 2))), float(n_p), float(mu_p), r.tolist()))
    rows2.sort(key=lambda t: t[0])
    out["with_model_barrier_channel"] = {
        "barrier_sheet_density_cm2": n2.tolist(),
        "barrier_only_n1_over_n77": ((A0 * A0 / B0) / n77).tolist(),
        "best": {"rms_log": rows2[0][0], "n_p_cm2": rows2[0][1], "mu_p_cm2Vs": rows2[0][2],
                 "log_n1_over_n77": rows2[0][3]}}
    print(json.dumps(out, indent=1), flush=True)
    corr = dict(data)
    n1, mu1 = zip(*[S.split(nH[i], mH[i], best[1], best[2]) for i in range(len(nH))])
    corr["ns_300K_1e13cm2"] = [v / 1e13 for v in n1]
    corr["mu_300K_cm2Vs"] = list(mu1)
    out["corrected_300K"] = {"ns_1e13": corr["ns_300K_1e13cm2"], "mu": corr["mu_300K_cm2Vs"]}
    d = M.dft()
    res = S.analyse(300.0, d, corr, 0.0)
    out["analysis_300K_channel_removed"] = {k: v for k, v in res.items() if k != "samples"}
    out["analysis_300K_channel_removed"]["samples"] = res["samples"]
    print(json.dumps({k: v for k, v in out["analysis_300K_channel_removed"].items()
                      if k in ("phi_B_eV", "dsig_cm2", "rms_log_misfit_pseudomorphic", "Lambda_fit_e_per_nm",
                               "Lambda_fit_range_over_phi_grid", "mobility", "R_needed_t75")}, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "shen_300K.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
