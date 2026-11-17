"""The Al0.22Ga0.78N/GaN thickness series of Shen, Someya and Arakawa,
Appl. Phys. Lett. 76, 2746 (2000): what charge do the misfit dislocations of
the relaxed barrier carry?

Data: data/shen2000_fig2_digitized.json (Fig. 2 read at native resolution).
Structure (from the text): undoped GaN, 3 nm undoped Al0.22Ga0.78N spacer, n-AlGaN
of thickness t (25-100 nm) with 1.26e18 cm^-3 electrons and a mobility of
83 cm^2/Vs at 300 K, free surface. The barrier is pseudomorphic at
t + 3 = 53 nm and relaxed by R = 0.78 at t + 3 = 103 nm (X-ray reciprocal-space
maps); the critical thickness lies between t = 65 and 75 nm. The 2DEG values were
obtained with a multilayer model that removes the undoped GaN layer; the
n-AlGaN layer was not removed.

Model: Schroedinger-Poisson (src/nmp/sp1d.py) with the surface Fermi level
pinned at e*phi_B below the AlGaN conduction band. Electrons within 40 nm below
and 4 nm above the interface are treated quantum mechanically (the 2DEG);
elsewhere they are a classical Fermi gas (the undepleted part of the n-AlGaN).
Donors are fully ionized at the measured 300 K electron density.

Parallel conduction. Where the n-AlGaN is not fully depleted it conducts in
parallel with the 2DEG. A single-field Hall measurement then returns
n_H = (n1 mu1 + n2 mu2)^2 / (n1 mu1^2 + n2 mu2^2) and
mu_H = (n1 mu1^2 + n2 mu2^2) / (n1 mu1 + n2 mu2), with 1 = 2DEG and 2 = barrier.
Given the calculated n2 and the barrier mobility mu2, these two equations give
the 2DEG density and mobility (n1, mu1) behind each measured pair. At 300 K,
mu2 = 83 cm^2/Vs (measured). At 77 K mu2 is not reported; mu2 = 0 (no parallel
conduction, e.g. frozen donors) and mu2 = 83 cm^2/Vs bracket it.

Calibration. In the pseudomorphic samples the n-AlGaN is not fully depleted,
so the 2DEG density hardly depends on phi_B; it depends on the interface
charge, which carries the uncertainties of the polarization constants, of the
Al content and of any fixed interface charge. A fixed interface sheet charge
dsig (the same in all samples, unchanged by relaxation) is fitted to the four
pseudomorphic samples (t = 25, 35, 50, 65 nm) for each phi_B on a grid, and the
phi_B with the smallest misfit is kept. The line charge is then fixed by the
change of the density on relaxation, which does not depend on dsig to first
order. An uncalibrated fit (dsig = 0) is reported as a sensitivity test.

For the relaxed sample the interface charge is

   sigma = sigma_coherent + 2 R eps_c (e31_prop - e33 C13/C33) + Lambda * L_A,

with L_A = 2 R eps_c / b the misfit-line length per area (b = a of the barrier)
and Lambda the actual charge per unit length of a misfit line (src/nmp/
polarization.py). Limits: Lambda = -P_H b ('H, improper': bulk-terminated half-
planes), Lambda = -(P_H - P_f,ZB) b (electron-counting cores), Lambda = 0.
The script reports the Lambda that reproduces the measured density, and,
independently, the |Lambda| allowed by the mobility (src/nmp/lines.py):
(i) a strict upper bound, if all scattering of the relaxed sample came from the
lines, and (ii) an estimate, if only the extra scattering relative to the
pseudomorphic samples does (Matthiessen's rule).

Sensitivity tests: R (0.70-0.85), conduction-band offset fraction (0.65, 0.75),
dielectric constant (Eq. (17) of Ambacher et al.), spontaneous-polarization
bowing (Vurgaftman and Meyer), fit without the t = 65 nm sample, donor
freeze-out bracket at 77 K (mu2), grid step, no calibration, and the HSE
polarization constants of Dreyer et al., Phys. Rev. X 6, 021038 (2016)
(data/dreyer2016_constants.json) in place of the PBEsol ones.

Outputs results/shen_analysis.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import lines, materials as M, polarization as PZ, sp1d  # noqa: E402

ROOT = os.path.join(HERE, "..")
X = 0.22
SPACER = 3.0
ND = 1.26e18
MU_BARRIER_300 = 83.0
GAN_T = 80.0
QW_BELOW, QW_ABOVE = 40.0, 4.0      # quantum window around the interface (nm)
QE = 1.602176634e-19
T_RELAXED, R_RELAXED = 100.0, 0.78
LIMITS = ("bulk-terminated (H, improper)", "electron counting", "neutral lines")


def setup(T, d, **opt):
    gan = M.algan(0.0, T, d, **opt)
    bar = M.algan(X, T, d, **opt)
    eps_c = (gan["a"] - bar["a"]) / bar["a"]
    return gan, bar, eps_c


def limits(bar):
    b = bar["a"] * 1e-10
    pzb = PZ.p_zb_formal(bar["a"])
    return {LIMITS[0]: -bar["P_H"] * b, LIMITS[1]: -(bar["P_H"] - pzb) * b, LIMITS[2]: 0.0}


def structure(t, T, phi_B, gan, bar, eps_c, R=0.0, Lam=0.0, dz=0.1, dsig=0.0):
    eps = (1 - R) * eps_c
    # P_bar is the improper (H-branch) polarization of the partly relaxed barrier.
    # Used alone, it already contains the charge of bulk-terminated misfit lines,
    # Lambda = -P_H b per unit length (src/nmp/polarization.py). For lines of
    # actual charge Lambda the difference (Lambda + P_H b) L_A is added.
    P_bar = M.strained_P(bar, eps)
    sheet = dsig
    if R > 0:
        LA = PZ.misfit_line_density(R, eps_c, bar["a"])          # m^-1
        sheet += (Lam + bar["P_H"] * bar["a"] * 1e-10) * LA       # C/m^2
    Ev_gan = -gan["Eg"]
    L = [sp1d.Layer("GaN", GAN_T, gan["eps"], 0.0, Ev_gan, gan["me"], 1.0, gan["P_H"]),
         sp1d.Layer("spacer", SPACER, bar["eps"], bar["dEc"], bar["dEc"] - bar["Eg"], bar["me"], 1.0, P_bar),
         sp1d.Layer("nAlGaN", t, bar["eps"], bar["dEc"], bar["dEc"] - bar["Eg"], bar["me"], 1.0, P_bar, Nd=ND)]
    return sp1d.Structure(L, phi_B=phi_B, dz=dz, T=T, extra_sheets={0: sheet} if (R > 0 or dsig) else None,
                          quantum_window=(GAN_T - QW_BELOW, GAN_T + QW_ABOVE), n_states=40,
                          classical_outside=True, holes=False)


def two_deg(st, r):
    """Sheet densities (cm^-2): quantum electrons near the interface (the 2DEG) and
    classical electrons elsewhere (essentially the undepleted part of the n-AlGaN)."""
    dz = st.dz * 1e-9
    return r["n_quantum"].sum() * dz * 1e-4, r["n_classical"].sum() * dz * 1e-4


def model(t, T, phi_B, gan, bar, eps_c, **kw):
    st = structure(t, T, phi_B, gan, bar, eps_c, **kw)
    r = st.solve()
    if not r["converged"]:
        raise RuntimeError("not converged: t=%g T=%g phi_B=%g %s" % (t, T, phi_B, kw))
    return two_deg(st, r)


def hall(n1, mu1, n2, mu2):
    s = n1 * mu1 + n2 * mu2
    s2 = n1 * mu1 ** 2 + n2 * mu2 ** 2
    return s * s / s2, s2 / s


def split(nH, muH, n2, mu2):
    """2DEG (n1, mu1) behind a measured Hall pair, given the parallel channel."""
    A = nH * muH - n2 * mu2
    B = nH * muH ** 2 - n2 * mu2 ** 2
    if A <= 0 or B <= 0:
        return float("nan"), float("nan")
    return A * A / B, B / A


def analyse(T, d, data, mu2, opt=None, R=R_RELAXED, pseudo_max=65.0, dz=0.1,
            phi_grid=(0.5, 1.0, 1.5, 2.0), offset=True):
    opt = opt or {}
    gan, bar, eps_c = setup(T, d, **opt)
    tag = "%dK" % T
    ts = data["thickness_nm"]
    nH = np.array(data["ns_%s_1e13cm2" % tag]) * 1e13
    muH = np.array(data["mu_%s_cm2Vs" % tag])
    pseudo = [i for i, t in enumerate(ts) if t <= pseudo_max]

    def run(t, phi, dsig, **kw):
        return model(t, T, phi, gan, bar, eps_c, dz=dz, dsig=dsig, **kw)

    def logs(phi, dsig):
        out = []
        for i in pseudo:
            n1, n2 = run(ts[i], phi, dsig)
            n1x, _ = split(nH[i], muH[i], n2, mu2)
            out.append(np.log(n1 / n1x) if np.isfinite(n1x) else 10.0)
        return np.array(out)

    scan = []
    if offset:
        for phi in phi_grid:
            ds = brentq(lambda x: logs(phi, x).sum(), -0.03, 0.03, xtol=1e-6)
            c = float((logs(phi, ds) ** 2).sum())
            scan.append({"phi_B_eV": phi, "dsig_Cm2": ds, "dsig_cm2": ds / QE * 1e-4, "cost": c})
        best = min(scan, key=lambda r: r["cost"])
        phi_B, dsig, cost = best["phi_B_eV"], best["dsig_Cm2"], best["cost"]
    else:
        res = minimize_scalar(lambda p: float((logs(p, 0.0) ** 2).sum()), bounds=(0.3, 2.5),
                              method="bounded", options={"xatol": 1e-4})
        phi_B, dsig, cost = float(res.x), 0.0, float(res.fun)
    out = _relaxed(T, mu2, R, ts, nH, muH, pseudo, run, phi_B, dsig, gan, bar, eps_c)
    out.update({"phi_B_eV": phi_B, "dsig_Cm2": dsig, "dsig_cm2": dsig / QE * 1e-4,
                "rms_log_misfit_pseudomorphic": float(np.sqrt(cost / len(pseudo))), "phi_scan": scan,
                "pseudomorphic_mean_n2DEG_from_data": float(np.mean([out["samples"][i]["n2DEG_from_data"]
                                                                    for i in pseudo]))})
    if offset and len(scan) > 1:
        spread = []
        for sc in scan:
            r2 = _relaxed(T, mu2, R, ts, nH, muH, pseudo, run, sc["phi_B_eV"], sc["dsig_Cm2"],
                          gan, bar, eps_c, light=True)
            sc["Lambda_fit_e_per_nm"] = r2["Lambda_fit_e_per_nm"]
            if r2["Lambda_fit_e_per_nm"] is not None:
                spread.append(r2["Lambda_fit_e_per_nm"])
        out["Lambda_fit_range_over_phi_grid"] = [min(spread), max(spread)] if spread else None
    return out


def _relaxed(T, mu2, R, ts, nH, muH, pseudo, run, phi_B, dsig, gan, bar, eps_c, light=False):
    rows = []
    for i, t in enumerate(ts):
        n1, n2 = run(t, phi_B, dsig)
        n1x, mu1x = split(nH[i], muH[i], n2, mu2)
        rows.append({"t_nm": t, "n2DEG_model": n1, "n_barrier_model": n2,
                     "n2DEG_from_data": n1x, "mu2DEG_from_data": mu1x,
                     "nH_measured": float(nH[i]), "muH_measured": float(muH[i])})
    out = {"T": T, "mu_barrier": mu2, "eps_c": eps_c, "samples": rows, "R": R}
    i100 = ts.index(T_RELAXED)
    LA = PZ.misfit_line_density(R, eps_c, bar["a"])
    out["L_A_per_cm"] = LA / 100.0
    out["line_spacing_nm_hexagonal"] = 3.0 / LA * 1e9      # three sets of lines: spacing = 3/L_A
    lims = limits(bar)
    out["predictions_t100"] = {}
    for name, Lam in lims.items():
        n1, n2 = run(T_RELAXED, phi_B, dsig, R=R, Lam=Lam)
        n1x, mu1x = split(nH[i100], muH[i100], n2, mu2)
        out["predictions_t100"][name] = {"Lambda_e_per_nm": Lam / QE * 1e-9, "n2DEG_model": n1,
                                         "n_barrier_model": n2, "n2DEG_from_data": n1x}
    n1c, n2c = run(T_RELAXED, phi_B, dsig)
    out["coherent_t100"] = {"n2DEG_model": n1c, "n_barrier_model": n2c}

    def g(L):
        n1, n2 = run(T_RELAXED, phi_B, dsig, R=R, Lam=L * QE * 1e9)
        n1x, _ = split(nH[i100], muH[i100], n2, mu2)
        return n1 - n1x
    try:
        Lfit = float(brentq(g, -8.0, 4.0, xtol=1e-3))
    except ValueError:
        Lfit = None
    out["Lambda_fit_e_per_nm"] = Lfit
    if light:
        return out
    # the t = 75 nm sample: relaxation not measured. Relaxation needed under each line charge
    i75 = ts.index(75)
    out["R_needed_t75"] = {}
    cases75 = dict(lims)
    if Lfit is not None:
        cases75["fitted line charge"] = Lfit * QE * 1e9
    for name in cases75:
        Lam = cases75[name]

        def h(Rx):
            n1, n2 = run(75.0, phi_B, dsig, R=Rx, Lam=Lam)
            n1x, _ = split(nH[i75], muH[i75], n2, mu2)
            return n1 - n1x
        try:
            out["R_needed_t75"][name] = float(brentq(h, 1e-3, 1.0, xtol=1e-3))
        except ValueError:
            out["R_needed_t75"][name] = None
    # mobility: 2DEG mobility of the relaxed sample vs the pseudomorphic ones
    n1x, mu1x = rows[i100]["n2DEG_from_data"], rows[i100]["mu2DEG_from_data"]
    if Lfit is not None:
        _, n2_ = run(T_RELAXED, phi_B, dsig, R=R, Lam=Lfit * QE * 1e9)
        n1x, mu1x = split(nH[i100], muH[i100], n2_, mu2)
    mu_ref = [rows[i]["mu2DEG_from_data"] for i in pseudo]
    m, eps = gan["me"], gan["eps"]
    unit = lines.rate_lines(n1x * 1e4, 1.0, LA, m, eps)                  # rate for 1 C/m
    total = QE / (m * sp1d.M0) / (mu1x * 1e-4)
    strict = np.sqrt(total / unit) / QE * 1e-9
    est = [float(lines.lambda_from_mobility_drop(mr, mu1x, n1x * 1e4, LA, m, eps) / QE * 1e-9)
           if mr > mu1x else 0.0 for mr in mu_ref]
    out["mobility"] = {"mu2DEG_relaxed": mu1x, "n2DEG_relaxed": n1x, "mu2DEG_pseudomorphic": mu_ref,
                       "Lambda_strict_upper_bound_e_per_nm": float(strict),
                       "Lambda_estimate_from_drop_e_per_nm": est,
                       "Lambda_estimate_range": [min(est), max(est)]}
    return out


def main():
    d = M.dft()
    data = json.load(open(os.path.join(ROOT, "data", "shen2000_fig2_digitized.json")))
    out = {"x": X, "spacer_nm": SPACER, "Nd_cm3": ND, "description": __doc__.split("\n\n")[0]}
    part = os.path.join(ROOT, "results", "shen_analysis_partial.json")
    base = {}
    for T in (300.0, 77.0):
        for mu2 in ((MU_BARRIER_300,) if T == 300.0 else (0.0, MU_BARRIER_300)):
            key = "%dK_mu2_%g" % (T, mu2)
            base[key] = analyse(T, d, data, mu2)
            print(key, json.dumps({k: v for k, v in base[key].items() if k != "samples"}), flush=True)
            json.dump({"baseline": base}, open(part, "w"), indent=1)
    out["baseline"] = base
    sens = {}
    variants = {"cbo_0.65": dict(opt={"cbo": 0.65}), "cbo_0.75": dict(opt={"cbo": 0.75}),
                "eps_eq17": dict(opt={"eps_rule": "eq17"}),
                "psp_bowing_0.021": dict(opt={"psp_bowing": 0.021}),
                "R_0.70": dict(R=0.70), "R_0.85": dict(R=0.85),
                "fit_without_65nm": dict(pseudo_max=60.0),
                "dz_0.05": dict(dz=0.05),
                "no_offset_calibration": dict(offset=False)}
    dre = json.load(open(os.path.join(ROOT, "data", "dreyer2016_constants.json")))
    d_dreyer = {m: {"P_H": dre[m]["P_H"], "e31_proper": dre[m]["e31_proper"], "e33": dre[m]["e33"],
                    "C13_over_C33": d[m]["C13_over_C33"]} for m in ("AlN", "GaN")}
    variants["constants_Dreyer2016_HSE"] = dict(d=d_dreyer)
    for T, mu2 in ((300.0, MU_BARRIER_300), (77.0, 0.0), (77.0, MU_BARRIER_300)):
        for name, kw in variants.items():
            kw = dict(kw)
            dd = kw.pop("d", d)
            kw.setdefault("phi_grid", (base["%dK_mu2_%g" % (T, mu2)]["phi_B_eV"],))
            r = analyse(T, dd, data, mu2, **kw)
            k = "%dK_mu2_%g/%s" % (T, mu2, name)
            sens[k] = {"phi_B_eV": r["phi_B_eV"], "dsig_cm2": r["dsig_cm2"],
                       "rms_log_misfit_pseudomorphic": r["rms_log_misfit_pseudomorphic"],
                       "Lambda_fit_e_per_nm": r["Lambda_fit_e_per_nm"],
                       "predictions_t100": {kk: vv["n2DEG_model"] for kk, vv in r["predictions_t100"].items()},
                       "coherent_t100": r["coherent_t100"]["n2DEG_model"],
                       "n2DEG_from_data_t100": r["samples"][-1]["n2DEG_from_data"],
                       "R_needed_t75": r["R_needed_t75"],
                       "Lambda_strict_upper_bound_e_per_nm": r["mobility"]["Lambda_strict_upper_bound_e_per_nm"],
                       "Lambda_estimate_range": r["mobility"]["Lambda_estimate_range"]}
            print(k, json.dumps(sens[k]), flush=True)
            out["sensitivity"] = sens
            json.dump(out, open(os.path.join(ROOT, "results", "shen_analysis.json"), "w"), indent=1)
    out["sensitivity"] = sens
    json.dump(out, open(os.path.join(ROOT, "results", "shen_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
