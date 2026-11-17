"""A second, independent AlGaN/GaN series: Hiroki, Maeda and Kobayashi,
J. Cryst. Growth 237-239, 956 (2002), Fig. 3 (room temperature) and Fig. 2(a).

Data: data/hiroki2002_digitized.json. Undoped Al(x)Ga(1-x)N barriers of thickness
d on 2-3 um of GaN, x = 0.25, 0.35, 0.45. The thicker samples of each
composition show a lower density; reciprocal-space maps show that the 70 nm,
x = 0.35 barrier is coherent and that the 85 nm, x = 0.35 and 95 nm, x = 0.25
barriers are relaxed. The degree of relaxation R is not reported.

Because R is unknown, the line charge cannot be fitted. The script asks the
reverse question: how much relaxation R would each assumed line charge need to
produce the measured drop? An assumption that needs R > 1 cannot explain the
data. Line charges tested (per unit length, b = a of the barrier):
  bulk-terminated   Lambda = -P_H b                 (layered-hexagonal recipe)
  electron counting Lambda = -(P_H - P_fZB) b       (close to zero)
  neutral           Lambda = 0                      (zinc-blende recipe)
  Shen fraction     Lambda = f * (-P_H b), with f = Lambda_fit / (-P_H b) from
                    the 77 K analysis of the Shen series (results/shen_analysis.json),
                    and its sensitivity range.

Model: as scripts/run_shen.py (Schroedinger-Poisson, surface pinned at phi_B,
fixed interface sheet dsig fitted to the coherent samples of each composition),
with an undoped barrier and no parallel channel in the model. Room temperature.
For x = 0.45 only one coherent sample exists, so dsig is fitted to it at each
phi_B. All results are given for every phi_B of the grid.

Room-temperature excess. At 40 nm Fig. 2(a) shows room-temperature densities
1.34-1.39 times the 77 K ones, close to the ratio seen in the Shen series.
Two treatments bracket its effect: (a) the room-temperature values as measured,
with the excess absorbed by dsig; (b) every value scaled by the 77 K / room-
temperature ratio measured at 40 nm for the same composition (x = 0.35 ratio
used for x = 0.45), i.e. an excess proportional to the density.

Outputs results/hiroki_analysis.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import materials as M, polarization as PZ, sp1d  # noqa: E402

ROOT = os.path.join(HERE, "..")
QE = 1.602176634e-19
GAN_T = 80.0
QW_BELOW, QW_ABOVE = 40.0, 4.0
T = 300.0
PHI_GRID = (0.5, 1.0, 1.5, 2.0)
RELAXED = {"0.25": [90], "0.35": [85], "0.45": [30]}


def setup(x, d):
    gan = M.algan(0.0, T, d)
    bar = M.algan(x, T, d)
    eps_c = (gan["a"] - bar["a"]) / bar["a"]
    return gan, bar, eps_c


def n2deg(t, phi_B, gan, bar, eps_c, R=0.0, Lam=0.0, dsig=0.0):
    P_bar = M.strained_P(bar, (1 - R) * eps_c)
    sheet = dsig
    if R > 0:
        LA = PZ.misfit_line_density(R, eps_c, bar["a"])
        sheet += (Lam + bar["P_H"] * bar["a"] * 1e-10) * LA
    L = [sp1d.Layer("GaN", GAN_T, gan["eps"], 0.0, -gan["Eg"], gan["me"], 1.0, gan["P_H"]),
         sp1d.Layer("AlGaN", t, bar["eps"], bar["dEc"], bar["dEc"] - bar["Eg"], bar["me"], 1.0, P_bar)]
    st = sp1d.Structure(L, phi_B=phi_B, dz=0.1, T=T, extra_sheets={0: sheet},
                        quantum_window=(GAN_T - QW_BELOW, GAN_T + QW_ABOVE), n_states=40,
                        classical_outside=True, holes=False)
    r = st.solve()
    if not r["converged"]:
        raise RuntimeError("not converged t=%g phi=%g R=%g" % (t, phi_B, R))
    return r["n_quantum"].sum() * st.dz * 1e-9 * 1e-4


def line_cases(bar, shen_f):
    b = bar["a"] * 1e-10
    bulk = -bar["P_H"] * b
    cases = {"bulk-terminated": bulk,
             "electron counting": -(bar["P_H"] - PZ.p_zb_formal(bar["a"])) * b,
             "neutral": 0.0}
    for k, f in shen_f.items():
        cases["Shen fraction %s (f=%.2f)" % (k, f)] = f * bulk
    return cases


def analyse_x(xs, data, scale, d, shen_f):
    x = float(xs)
    gan, bar, eps_c = setup(x, d)
    ts = data["fig3_RT"][xs]["t_nm"]
    ns = np.array(data["fig3_RT"][xs]["ns_1e12"]) * 1e12 * scale
    coh = [i for i, t in enumerate(ts) if t not in RELAXED[xs]]
    cases = line_cases(bar, shen_f)
    out = {"x": x, "eps_c": eps_c, "scale": scale, "t_nm": ts, "ns_used": list(ns),
           "Lambda_e_per_nm": {k: v / QE * 1e-9 for k, v in cases.items()}, "phi": {}}
    for phi in PHI_GRID:
        def cost(ds):
            return sum(np.log(n2deg(ts[i], phi, gan, bar, eps_c, dsig=ds) / ns[i]) for i in coh)
        ds = brentq(cost, -0.05, 0.05, xtol=1e-6)
        fit = {ts[i]: n2deg(ts[i], phi, gan, bar, eps_c, dsig=ds) for i in coh}
        rms = float(np.sqrt(np.mean([np.log(fit[ts[i]] / ns[i]) ** 2 for i in coh])))
        rel = {}
        for tr in RELAXED[xs]:
            i = ts.index(tr)
            n_coh = n2deg(tr, phi, gan, bar, eps_c, dsig=ds)
            need = {}
            for name, Lam in cases.items():
                g = lambda R: n2deg(tr, phi, gan, bar, eps_c, R=R, Lam=Lam, dsig=ds) - ns[i]
                if g(1.0) > 0:
                    need[name] = None            # even full relaxation leaves too many electrons
                else:
                    need[name] = float(brentq(g, 0.0, 1.0, xtol=1e-3))
            n_full = {name: n2deg(tr, phi, gan, bar, eps_c, R=1.0, Lam=Lam, dsig=ds)
                      for name, Lam in cases.items()}
            rel[str(tr)] = {"n_measured": float(ns[i]), "n_coherent_model": n_coh,
                            "R_needed": need, "n_model_at_R1": n_full}
        out["phi"][str(phi)] = {"dsig_cm2": ds / QE * 1e-4, "rms_log_misfit_coherent": rms,
                                "fit_coherent": {str(k): v for k, v in fit.items()}, "relaxed": rel}
        print(xs, scale, phi, json.dumps(out["phi"][str(phi)]["relaxed"]), flush=True)
    return out


def main():
    d = M.dft()
    data = json.load(open(os.path.join(ROOT, "data", "hiroki2002_digitized.json")))
    shen = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))
    b77 = shen["baseline"]["77K_mu2_0"]
    bulk22 = b77["predictions_t100"]["bulk-terminated (H, improper)"]["Lambda_e_per_nm"]
    lam = [v["Lambda_fit_e_per_nm"] for k, v in shen["sensitivity"].items()
           if k.startswith("77K") and v["Lambda_fit_e_per_nm"] is not None]
    lam += [b77["Lambda_fit_e_per_nm"]] + list(b77["Lambda_fit_range_over_phi_grid"])
    shen_f = {"best": b77["Lambda_fit_e_per_nm"] / bulk22,
              "low": max(lam) / bulk22, "high": min(lam) / bulk22}
    f2 = data["fig2a_40nm"]
    ratio = {"0.25": f2["ns_77K_1e12"][0] / f2["ns_RT_1e12"][0],
             "0.35": f2["ns_77K_1e12"][1] / f2["ns_RT_1e12"][1]}
    ratio["0.45"] = ratio["0.35"]
    out = {"description": __doc__.split("\n\n")[0], "shen_fraction": shen_f,
           "ratio_77K_over_RT_40nm": ratio, "as_measured": {}, "scaled_to_77K": {}}
    for xs in ("0.35", "0.25", "0.45"):
        out["as_measured"][xs] = analyse_x(xs, data, 1.0, d, shen_f)
        out["scaled_to_77K"][xs] = analyse_x(xs, data, ratio[xs], d, shen_f)
        json.dump(out, open(os.path.join(ROOT, "results", "hiroki_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
