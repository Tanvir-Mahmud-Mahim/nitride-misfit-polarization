"""The spacer series of Shen, Someya and Arakawa, Appl. Phys. Lett. 76, 2746
(2000), Fig. 3: a second, independent test from the same paper.

Data: data/shen2000_fig3_digitized.json. Structure (from the text): undoped
GaN, undoped Al0.22Ga0.78N spacer of thickness s (0 to 50 nm), Si-doped
Al0.22Ga0.78N of 25 nm (the doping and barrier mobility of the thickness
series), free surface. The total AlGaN thickness is 25 + s nm. In the thickness
series (Fig. 2, run_shen.py) the barrier is pseudomorphic at 53 nm and relaxed
by R = 0.78 at 103 nm, with the drop of density and mobility between 68 and 78
nm. The spacer samples up to s = 30 nm (55 nm in total) are therefore
pseudomorphic; the s = 50 nm sample (75 nm in total) lies in the range where
relaxation starts, and its relaxation was not measured.

What the script does.
1. Prediction of the pseudomorphic spacer samples with the calibration of the
   thickness series (surface barrier phi_B and fixed interface sheet dsig from
   results/shen_analysis.json), with nothing refitted: a test of the model on
   samples it was not fitted to.
2. Refit of dsig on the pseudomorphic spacer samples (phi_B kept), as a
   sensitivity test.
3. For the s = 50 nm sample: the relaxation R each line charge needs to give
   its density (bare ends, electron counting, neutral, and the line charge
   fitted to the thickness series). A physical answer lies below 0.78, the
   relaxation of the thicker (103 nm) sample.
4. Mobility check for the same sample: with R fixed by the density for each
   line charge, the largest line charge its mobility allows (strict bound,
   every scattering event from the lines), as in run_shen.py.

Parallel conduction in the n-AlGaN is removed as in run_shen.py (300 K:
mu2 = 83 cm^2/Vs; 77 K: mu2 = 0 and 83 bracket it).

Outputs results/shen_spacer_analysis.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from nmp import lines, materials as M, polarization as PZ, sp1d  # noqa: E402
import run_shen as RS  # noqa: E402

ROOT = os.path.join(HERE, "..")
T_NALGAN = 25.0
S_RELAXED = 50.0
R_MAX = 0.78
QE = RS.QE


def structure(s, T, phi_B, gan, bar, eps_c, R=0.0, Lam=0.0, dz=0.1, dsig=0.0):
    eps = (1 - R) * eps_c
    P_bar = M.strained_P(bar, eps)
    sheet = dsig
    if R > 0:
        LA = PZ.misfit_line_density(R, eps_c, bar["a"])
        sheet += (Lam + bar["P_H"] * bar["a"] * 1e-10) * LA
    Ev_gan = -gan["Eg"]
    L = [sp1d.Layer("GaN", RS.GAN_T, gan["eps"], 0.0, Ev_gan, gan["me"], 1.0, gan["P_H"])]
    if s > 0:
        L.append(sp1d.Layer("spacer", s, bar["eps"], bar["dEc"], bar["dEc"] - bar["Eg"], bar["me"], 1.0, P_bar))
    L.append(sp1d.Layer("nAlGaN", T_NALGAN, bar["eps"], bar["dEc"], bar["dEc"] - bar["Eg"], bar["me"], 1.0,
                        P_bar, Nd=RS.ND))
    return sp1d.Structure(L, phi_B=phi_B, dz=dz, T=T, extra_sheets={0: sheet} if (R > 0 or dsig) else None,
                          quantum_window=(RS.GAN_T - RS.QW_BELOW, RS.GAN_T + RS.QW_ABOVE), n_states=40,
                          classical_outside=True, holes=False)


def model(s, T, phi_B, gan, bar, eps_c, **kw):
    st = structure(s, T, phi_B, gan, bar, eps_c, **kw)
    r = st.solve()
    if not r["converged"]:
        raise RuntimeError("not converged: s=%g T=%g %s" % (s, T, kw))
    return RS.two_deg(st, r)


def analyse(T, d, data, mu2, phi_B, dsig_series, Lfit_series):
    gan, bar, eps_c = RS.setup(T, d)
    tag = "%dK" % T
    ss = data["spacer_nm"]
    nH = np.array(data["ns_%s_1e13cm2" % tag]) * 1e13
    muH = np.array(data["mu_%s_cm2Vs" % tag])
    coh = [i for i, s in enumerate(ss) if s < S_RELAXED]

    def run(s, dsig, **kw):
        return model(s, T, phi_B, gan, bar, eps_c, dsig=dsig, **kw)

    # 1. prediction with the thickness-series calibration
    rows = []
    for i, s in enumerate(ss):
        n1, n2 = run(s, dsig_series)
        n1x, mu1x = RS.split(nH[i], muH[i], n2, mu2)
        rows.append({"spacer_nm": s, "n2DEG_model": n1, "n_barrier_model": n2,
                     "n2DEG_from_data": n1x, "mu2DEG_from_data": mu1x})
    logm = np.array([np.log(rows[i]["n2DEG_model"] / rows[i]["n2DEG_from_data"]) for i in coh])
    out = {"T": T, "mu_barrier": mu2, "phi_B_eV": phi_B, "dsig_series_cm2": dsig_series / QE * 1e-4,
           "samples": rows,
           "prediction_rms_log_misfit": float(np.sqrt((logm ** 2).mean())),
           "prediction_mean_log_misfit": float(logm.mean())}

    # 2. refit dsig on the pseudomorphic spacer samples
    def logs(ds):
        v = []
        for i in coh:
            n1, n2 = run(ss[i], ds)
            n1x, _ = RS.split(nH[i], muH[i], n2, mu2)
            v.append(np.log(n1 / n1x))
        return np.array(v)
    ds_fit = brentq(lambda x: logs(x).sum(), -0.03, 0.03, xtol=1e-6)
    out["dsig_refit_cm2"] = ds_fit / QE * 1e-4
    out["refit_rms_log_misfit"] = float(np.sqrt((logs(ds_fit) ** 2).mean()))

    # 3. relaxation needed by the s = 50 nm sample, for each line charge
    i50 = ss.index(S_RELAXED)
    cases = dict(RS.limits(bar))
    cases["line charge of the thickness series"] = Lfit_series * QE * 1e9
    m, eps = gan["me"], gan["eps"]
    res = {}
    for calib, ds in (("series calibration", dsig_series), ("refit on spacer samples", ds_fit)):
        res[calib] = {}
        for name, Lam in cases.items():
            def h(Rx):
                n1, n2 = run(S_RELAXED, ds, R=Rx, Lam=Lam)
                n1x, _ = RS.split(nH[i50], muH[i50], n2, mu2)
                return n1 - n1x
            try:
                Rn = float(brentq(h, 1e-3, 1.0, xtol=1e-3))
            except ValueError:
                Rn = None
            entry = {"Lambda_e_per_nm": Lam / QE * 1e-9, "R_needed": Rn}
            if Rn is not None:
                # 4. mobility: strict bound at that relaxation
                _, n2 = run(S_RELAXED, ds, R=Rn, Lam=Lam)
                n1x, mu1x = RS.split(nH[i50], muH[i50], n2, mu2)
                LA = PZ.misfit_line_density(Rn, eps_c, bar["a"])
                unit = lines.rate_lines(n1x * 1e4, 1.0, LA, m, eps)
                total = QE / (m * sp1d.M0) / (mu1x * 1e-4)
                entry["mobility_strict_bound_e_per_nm"] = float(np.sqrt(total / unit) / QE * 1e-9)
                entry["mu2DEG_from_data"] = mu1x
                entry["n2DEG_from_data"] = n1x
            res[calib][name] = entry
    out["s50"] = res
    out["eps_c"] = eps_c
    return out


def main():
    d = M.dft()
    data = json.load(open(os.path.join(ROOT, "data", "shen2000_fig3_digitized.json")))
    series = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]
    out = {"description": __doc__.split("\n\n")[0], "n_AlGaN_nm": T_NALGAN, "R_max": R_MAX, "cases": {}}
    for T, mu2 in ((77.0, 0.0), (77.0, RS.MU_BARRIER_300), (300.0, RS.MU_BARRIER_300)):
        key = "%dK_mu2_%g" % (T, mu2)
        b = series[key]
        r = analyse(T, d, data, mu2, b["phi_B_eV"], b["dsig_Cm2"], b["Lambda_fit_e_per_nm"])
        out["cases"][key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k != "samples"}), flush=True)
        json.dump(out, open(os.path.join(ROOT, "results", "shen_spacer_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
