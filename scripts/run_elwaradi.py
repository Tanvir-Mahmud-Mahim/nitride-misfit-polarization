"""A second series with measured relaxation: Elwaradi et al., J. Appl. Phys.
133, 145705 (2023) (data/elwaradi2023_tables.json).

AlGaN/GaN/AlN structures on bulk AlN: an AlN buffer, a GaN channel of thickness
t whose relaxation on AlN was measured by X-ray diffraction (Table I), a 1 nm
AlN spacer, a 19 nm Al0.3Ga0.7N barrier and a thin GaN cap. On the 20 nm
channel the GaN is only 8 percent relaxed, so its in-plane lattice is close to
that of AlN and the barrier is under compression; it relaxed by 25 percent
(Table II). This is a misfit network of the opposite sign to that of the Shen
series, at an AlGaN/GaN interface.

Model (src/nmp/sp1d.py, electrons and holes): bottom-up AlN (zero field at the
bottom), GaN channel, AlN spacer, AlGaN barrier, GaN cap, surface Fermi level
pinned phi_B below the conduction band. The bound charge of each interface is
the difference of the H-branch (improper) polarizations of the two layers at
their actual strains; this already contains bare-end misfit lines, so lines of
charge Lambda add (Lambda - Lambda_bare) L_A at the interface where they lie
(the GaN/AlN interface for the channel, the GaN/spacer interface for the
barrier). One fixed sheet dsig at the 2DEG interface is fitted, for each
phi_B, to the four samples whose barrier is coherent (t = 500, 200, 100 and
50 nm), with the channel lines set to each hypothesis in turn. The 20 nm
sample is then predicted for each hypothesis of the barrier line charge.

Hypotheses (Lambda of the AlGaN data of Shen et al. entered as a fraction f
of the bare-end charge, or as electrons trapped with the same magnitude):
bare ends, neutral, electron counting, bound (f Lambda_bare), trapped
(-|f Lambda_bare|, negative under any strain).

usage: python3 scripts/run_elwaradi.py [quick]
Outputs results/elwaradi_analysis.json.
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
T = 300.0
X = 0.30
T_BAR, T_SPACER = 19.0, 1.0
T_ALN = 60.0


def setup(d, x=X):
    return M.algan(0.0, T, d), M.algan(1.0, T, d), M.algan(x, T, d)


def stack(t_ch, R_ch, R_bar, lam_ch, lam_bar, phi_B, dsig, mats, t_cap=1.0, dz=0.1, mh=1.0, nd_gan=0.0):
    """Returns (structure, info). lam_ch, lam_bar: either a number (C/m) or a key
    of the limits dictionary of that interface."""
    gan, aln, bar = mats
    a_g, a_a, a_b = gan["a"], aln["a"], bar["a"]
    epsc_ch = (a_a - a_g) / a_g                      # GaN on AlN, negative
    eps_ch = (1 - R_ch) * epsc_ch
    a_ch = a_g * (1 + eps_ch)                         # in-plane lattice of the channel
    epsc_b = (a_ch - a_b) / a_b                       # barrier on the channel
    eps_b = (1 - R_bar) * epsc_b
    a_ip = a_b * (1 + eps_b)                          # in-plane lattice of barrier, spacer, cap
    eps_sp = (a_ip - a_a) / a_a
    eps_cap = (a_ip - a_g) / a_g
    P = lambda m, e: PZ.p_h_improper(m, e)
    lim_ch = PZ.line_charge_limits(gan, epsc_ch)
    lim_b = PZ.line_charge_limits(bar, epsc_b)
    L_ch = PZ.misfit_line_density(R_ch, epsc_ch, a_g) if R_ch > 0 else 0.0
    L_b = PZ.misfit_line_density(R_bar, epsc_b, a_b) if R_bar > 0 else 0.0
    lc = lim_ch[lam_ch] if isinstance(lam_ch, str) else lam_ch
    lb = lim_b[lam_bar] if isinstance(lam_bar, str) else lam_bar
    extra = {0: (lc - lim_ch["bulk-terminated"]) * L_ch,
             1: (lb - lim_b["bulk-terminated"]) * L_b + dsig}
    Ev = lambda m: m["dEc"] - m["Eg"]
    L = [sp1d.Layer("AlN", T_ALN, aln["eps"], aln["dEc"], Ev(aln), aln["me"], mh, P(aln, 0.0)),
         sp1d.Layer("GaN", t_ch, gan["eps"], 0.0, -gan["Eg"], gan["me"], mh, P(gan, eps_ch), Nd=nd_gan),
         sp1d.Layer("spacer", T_SPACER, aln["eps"], aln["dEc"], Ev(aln), aln["me"], mh, P(aln, eps_sp)),
         sp1d.Layer("AlGaN", T_BAR, bar["eps"], bar["dEc"], Ev(bar), bar["me"], mh, P(bar, eps_b))]
    if t_cap > 0:
        L.append(sp1d.Layer("cap", t_cap, gan["eps"], 0.0, -gan["Eg"], gan["me"], mh, P(gan, eps_cap)))
    st = sp1d.Structure(L, phi_B=phi_B, dz=dz, T=T, extra_sheets=extra, n_states=40,
                        classical_outside=False, holes=True)
    info = {"eps_ch": eps_ch, "epsc_bar": epsc_b, "eps_bar": eps_b, "L_ch_per_cm": L_ch / 100,
            "L_bar_per_cm": L_b / 100, "Lambda_bare_ch_e_per_nm": lim_ch["bulk-terminated"] / QE * 1e-9,
            "Lambda_bare_bar_e_per_nm": lim_b["bulk-terminated"] / QE * 1e-9}
    return st, info


def densities(st, r):
    dz = st.dz * 1e-9
    zi = st.edges[2]                                  # GaN/spacer interface (2DEG)
    up = st.z > zi - 30.0
    n = (r["n"] * up).sum() * dz * 1e-4
    p = r["p"].sum() * dz * 1e-4
    return n, p


def solve(*a, **k):
    st, info = stack(*a, **k)
    r = st.solve()
    if not r["converged"]:
        raise RuntimeError("not converged %s %s" % (a, k))
    return densities(st, r), info


SAMPLES = {500: 1.0, 200: 0.88, 100: 0.72, 50: 0.5}      # coherent barrier: t_GaN (nm): GaN relaxation
N_MEAS = {500: 1.0e13, 200: 9.0e12, 100: 8.0e12, 50: 7.8e12}
T20, R20_CH, N20, MU20, MU50 = 20.0, 0.08, 2.2e12, 310.0, 460.0
R20_BAR = (0.20, 0.25)
DS_PROBE = -3e12                                          # cm^-2, for the linear calibration


def constants(name, d):
    if name == "PBEsol":
        return d
    dre = json.load(open(os.path.join(ROOT, "data", "dreyer2016_constants.json")))
    return {m: {"P_H": dre[m]["P_H"], "e31_proper": dre[m]["e31_proper"], "e33": dre[m]["e33"],
                "C13_over_C33": d[m]["C13_over_C33"]} for m in ("AlN", "GaN")}


def cm2_to_C(x):
    return x * 1e4 * QE


def main(quick=False):
    d = M.dft()
    pr = json.load(open(os.path.join(ROOT, "results", "prediction.json")))["f"]
    out = {"description": __doc__.split("\n\n")[0], "data": "data/elwaradi2023_tables.json",
           "f_Shen": pr, "cases": {}}
    phis = (1.0,) if quick else (0.5, 1.0, 1.5, 2.0)
    variants = [(c, phi, 1.0) for c in ("PBEsol", "HSE") for phi in phis]
    if not quick:
        variants += [(c, 1.0, cap) for c in ("PBEsol", "HSE") for cap in (0.0, 2.0)]
    for cname, phi, cap in variants:
        mats = setup(constants(cname, d))
        key = "%s/phi_%g/cap_%g" % (cname, phi, cap)
        row = {"coherent": {}}
        # linear response of each coherent sample to a fixed sheet at the 2DEG interface
        for t, R in SAMPLES.items():
            dz = 0.25 if t >= 100 else 0.1
            (n0, _), _ = solve(t, R, 0.0, "neutral", "neutral", phi, 0.0, mats, t_cap=cap, dz=dz)
            (n1, _), _ = solve(t, R, 0.0, "neutral", "neutral", phi, cm2_to_C(DS_PROBE), mats, t_cap=cap, dz=dz)
            k = (n1 - n0) / DS_PROBE
            row["coherent"][t] = {"n_model_dsig0": n0, "dn_ddsig": k, "dsig_to_match": (N_MEAS[t] - n0) / k}
        ds_all = [v["dsig_to_match"] for v in row["coherent"].values()]
        ds_thick = [row["coherent"][t]["dsig_to_match"] for t in (500, 200, 100)]
        cal = {"none": 0.0, "thick (500-100 nm)": float(np.mean(ds_thick)),
               "50 nm": row["coherent"][50]["dsig_to_match"],
               "all four": float(np.mean(ds_all))}
        row["calibration_dsig_cm2"] = cal
        row["rms_all_four_after_mean_cal"] = float(np.sqrt(np.mean([
            ((row["coherent"][t]["n_model_dsig0"] + row["coherent"][t]["dn_ddsig"] * cal["all four"]) / N_MEAS[t] - 1) ** 2
            for t in SAMPLES])))
        # 20 nm sample: compressed barrier relaxed by R_bar
        row["t20"] = {}
        for Rb in R20_BAR:
            st, info = stack(T20, R20_CH, Rb, "neutral", "neutral", phi, 0.0, mats, t_cap=cap)
            bare = info["Lambda_bare_bar_e_per_nm"] * QE * 1e9
            lim = PZ.line_charge_limits(mats[2], info["epsc_bar"])
            f = pr[cname]
            hyp = {"bare ends": bare, "electron counting": lim["electron counting"], "neutral": 0.0,
                   "bound (Shen fraction)": f * bare, "trapped (Shen magnitude)": -abs(f * bare)}
            rr = {"info": info, "pred": {}}
            for cn, ds in cal.items():
                rr["pred"][cn] = {}
                for hn, lam in hyp.items():
                    (n, p), _ = solve(T20, R20_CH, Rb, "neutral", lam, phi, cm2_to_C(ds), mats, t_cap=cap)
                    rr["pred"][cn][hn] = n
            # line charge that reproduces the measured density, for each calibration
            rr["Lambda_fit_e_per_nm"] = {}
            for cn, ds in cal.items():
                g = lambda L: solve(T20, R20_CH, Rb, "neutral", L * QE * 1e9, phi, cm2_to_C(ds), mats,
                                    t_cap=cap)[0][0] - N20
                try:
                    rr["Lambda_fit_e_per_nm"][cn] = float(brentq(g, -4.0, 4.0, xtol=0.02))
                except ValueError:
                    rr["Lambda_fit_e_per_nm"][cn] = None
            row["t20"]["R_bar_%g" % Rb] = rr
        out["cases"][key] = row
        print(key, json.dumps({"cal": {k: round(v / 1e12, 2) for k, v in cal.items()},
                               "rms4": round(row["rms_all_four_after_mean_cal"], 3),
                               "pred_R0.25": {c: {h: round(v / 1e12, 2) for h, v in hh.items()}
                                              for c, hh in row["t20"]["R_bar_0.25"]["pred"].items()},
                               "Lfit": row["t20"]["R_bar_0.25"]["Lambda_fit_e_per_nm"]}), flush=True)
        json.dump(out, open(os.path.join(ROOT, "results", "elwaradi_analysis.json"), "w"), indent=1)
    # mobility of the 20 nm sample: strict bound on |Lambda| of the barrier lines
    from nmp import lines
    mats = setup(d)
    gan = mats[0]
    mob = {}
    for Rb in R20_BAR:
        _, info = stack(T20, R20_CH, Rb, "neutral", "neutral", 1.0, 0.0, mats)
        LA = info["L_bar_per_cm"] * 100
        unit = lines.rate_lines(N20 * 1e4, 1.0, LA, gan["me"], gan["eps"])
        total = QE / (gan["me"] * sp1d.M0) / (MU20 * 1e-4)
        strict = float(np.sqrt(total / unit) / QE * 1e-9)
        est = float(lines.lambda_from_mobility_drop(MU50, MU20, N20 * 1e4, LA, gan["me"], gan["eps"]) / QE * 1e-9)
        mob["R_bar_%g" % Rb] = {"L_A_per_cm": LA / 100, "Lambda_strict_upper_bound_e_per_nm": strict,
                                "Lambda_estimate_vs_50nm_e_per_nm": est,
                                "Lambda_bare_e_per_nm": info["Lambda_bare_bar_e_per_nm"]}
    out["mobility_t20"] = mob
    print("mobility", json.dumps(mob), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "elwaradi_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main(quick=len(sys.argv) > 1 and sys.argv[1] == "quick")
