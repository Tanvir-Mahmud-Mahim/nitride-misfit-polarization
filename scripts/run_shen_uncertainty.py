"""Statistical error bar of the line charge from the Shen series, with labplan.

The baseline analysis (scripts/run_shen.py) fixes the interface offset dsig on
the four coherent samples and then reads the line charge Lambda off the relaxed
sample. Here both are fitted together to the five 77 K densities with
labplan (T. M. Mahim, doi:10.5281/zenodo.22826765), a least-squares package
that also gives error bars, a goodness-of-fit test, and a fit with errors in
the settings:

  fit A  labplan.fit, no error bars supplied: the error scale is taken from the
         scatter of the readings about the model (3 degrees of freedom), so
         the error bar of Lambda reflects the sample-to-sample scatter.
  fit B  labplan.fit_xy: the same readings with error bars equal to that
         scatter, and the relaxation R = 0.78 of the relaxed sample given an
         error bar of 0.05 (the range 0.70-0.85 used in the sensitivity tests
         of run_shen.py), so the uncertainty of R enters the error bar of
         Lambda.

The parameters are dsig (1e12 cm^-2) and Lambda (e/nm); the settings are the
barrier thickness t (nm) and the relaxation R; the readings are the 77 K Hall
densities (1e12 cm^-2), taken as 2DEG densities (no parallel channel at 77 K,
the baseline case mu2 = 0 of run_shen.py). The surface barrier is fixed at
each phi_B of the grid in turn. The ratio Lambda / (-P_H b), the fraction of
the bulk-terminated charge, is propagated with labplan.propagate.

Outputs results/shen_uncertainty.json. With the argument "hse" the same fits
are made with the hybrid-functional constants of Dreyer et al., Phys. Rev. X 6,
021038 (2016) (data/dreyer2016_constants.json) and written to
results/shen_uncertainty_hse.json; the starting values are then the baseline
fit at phi_B = 0.5 eV.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import labplan  # noqa: E402
import run_shen as S  # noqa: E402
from nmp import materials as M  # noqa: E402

ROOT = os.path.join(HERE, "..")
QE = 1.602176634e-19
T = 77.0
R_SIGMA = 0.05


def main(hse=False):
    d = M.dft()
    if hse:
        dre = json.load(open(os.path.join(ROOT, "data", "dreyer2016_constants.json")))
        d = {m: {"P_H": dre[m]["P_H"], "e31_proper": dre[m]["e31_proper"], "e33": dre[m]["e33"],
                 "C13_over_C33": d[m]["C13_over_C33"]} for m in ("AlN", "GaN")}
    data = json.load(open(os.path.join(ROOT, "data", "shen2000_fig2_digitized.json")))
    gan, bar, eps_c = S.setup(T, d)
    ts = data["thickness_nm"]
    ns = np.array(data["ns_77K_1e13cm2"]) * 10.0           # 1e12 cm^-2
    use = [i for i, t in enumerate(ts) if t != 75]           # R of the 75 nm sample unknown
    x = np.array([[ts[i], S.R_RELAXED if ts[i] == S.T_RELAXED else 0.0] for i in use])
    y = ns[use]
    lam_bulk = -bar["P_H"] * bar["a"] * 1e-10 / QE * 1e-9   # e/nm
    out = {"description": __doc__.split("\n\n")[0], "labplan_version": labplan.__version__,
           "readings_1e12": list(y), "settings_t_R": x.tolist(), "Lambda_bulk_e_per_nm": lam_bulk,
           "R_sigma": R_SIGMA, "phi": {}}
    outname = "shen_uncertainty_hse.json" if hse else "shen_uncertainty.json"
    for phi in ((0.5,) if hse else (0.5, 1.0, 1.5, 2.0)):
        def f(theta, xx):
            ds = theta[0] * 1e12 * 1e4 * QE                  # C/m^2
            lam = theta[1] * QE * 1e9                        # C/m
            return np.array([S.model(t, T, phi, gan, bar, eps_c, R=R, Lam=lam, dsig=ds)[0] / 1e12
                             for t, R in xx])
        model = labplan.Model("Shen 77 K, Schroedinger-Poisson, phi_B = %g eV" % phi, f,
                              ("dsig_1e12cm2", "Lambda_e_per_nm"),
                              "scripts/run_shen.py of this work (model of the AlGaN/GaN series of "
                              "Shen et al., Appl. Phys. Lett. 76, 2746 (2000))")
        base = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]["77K_mu2_0"]
        scan = {s["phi_B_eV"]: s for s in base["phi_scan"]}[phi]
        theta0 = np.array([scan["dsig_cm2"] / 1e12, scan["Lambda_fit_e_per_nm"]])
        A = labplan.fit(model, x, y, theta0)
        resid = y - f(A.theta, x)
        s = float(np.sqrt((resid ** 2).sum() / (len(y) - 2)))
        B = labplan.fit_xy(model, x, y, A.theta, sigmas=s,
                           x_sigmas=np.array([[0.0, R_SIGMA if R > 0 else 0.0] for _, R in x]))
        gB = labplan.goodness_of_fit(B)
        frac = lambda th: np.array([th[1] / lam_bulk])
        pA = labplan.propagate(frac, A.theta, A.cov)
        pB = labplan.propagate(frac, B.theta, B.cov)
        out["phi"][str(phi)] = {
            "fit_A": {"values": A.values, "sigma": A.sigma, "residual_sd_1e12": s,
                      "fraction_of_bulk": float(np.ravel(pA["value"])[0]),
                      "fraction_sigma": float(np.ravel(pA["sigma"])[0])},
            "fit_B": {"values": B.values, "sigma": B.sigma, "chi2": B.chi2, "dof": B.chi2_dof,
                      "goodness": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                                   for k, v in gB.items()},
                      "R_adjusted": float(B.x_adjusted[-1][1]),
                      "fraction_of_bulk": float(np.ravel(pB["value"])[0]),
                      "fraction_sigma": float(np.ravel(pB["sigma"])[0])}}
        print(phi, json.dumps(out["phi"][str(phi)]), flush=True)
        json.dump(out, open(os.path.join(ROOT, "results", outname), "w"), indent=1)


if __name__ == "__main__":
    main(hse=len(sys.argv) > 1 and sys.argv[1] == "hse")
