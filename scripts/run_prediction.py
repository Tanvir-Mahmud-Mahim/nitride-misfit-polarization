"""A test that does not depend on the polarization constants: how fast the
interface charge changes as a layer relaxes.

The absolute charge of a relaxed interface depends on the spontaneous
polarizations, which differ between calculations (PBEsol here, HSE in Dreyer
et al. 2016). Its rate of change with the relaxed fraction R does not depend on
them to first order: from Eq. (2) of the Letter,

    d sigma / d R = 2 eps_c (e31 - e33 C13/C33) + Lambda * 2 |eps_c| / b,

the proper piezoelectric response plus the line charge. For each interface the
script gives this slope for bare ends (Lambda = -/+ P_H b), electron counting,
neutral lines, and the fraction f of the bare-end charge found in the Shen
series (0.31 with the PBEsol constants, 0.21 with the HSE constants, from
results/shen_analysis.json), with both sets of constants, and the charge of the
fully strained and fully relaxed interface.

The line charge measured in the Shen series may be bound charge of partly
compensated cores, which follows the sign of the bare-end charge, or electrons
trapped on neutral cores, which is negative whatever the sign of the mismatch.
For a film under tension (AlGaN on GaN) the two give the same slope; for a film
under compression (GaN on AlN) they give opposite signs of Lambda. The case
"trapped (Shen)" uses Lambda = -|Lambda_Shen|. For each AlGaN case the script
also gives the fraction of the bare-end charge at which the fully relaxed
interface charge changes sign (where a hole gas would become possible), and for
GaN on AlN the charge differences at R = 0.2 in C/m^2.
Outputs results/prediction.json.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import materials as M, polarization as PZ  # noqa: E402

ROOT = os.path.join(HERE, "..")
CASES = {"GaN on AlN": (0.0, 1.0), "AlN on GaN": (1.0, 0.0),
         "Al0.30Ga0.70N on GaN": (0.30, 0.0), "Al0.22Ga0.78N on GaN": (0.22, 0.0),
         "Al0.30Ga0.70N on GaN strained to AlN": (0.30, "GaN strained to AlN"),
         "In0.07Ga0.93N on AlN": None}


def main():
    d = M.dft()
    dre = json.load(open(os.path.join(ROOT, "data", "dreyer2016_constants.json")))
    d_hse = {m: {"P_H": dre[m]["P_H"], "e31_proper": dre[m]["e31_proper"], "e33": dre[m]["e33"],
                 "C13_over_C33": d[m]["C13_over_C33"]} for m in ("AlN", "GaN")}
    sh = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))
    b0 = sh["baseline"]["77K_mu2_0"]
    bulk = b0["predictions_t100"]["bulk-terminated (H, improper)"]["Lambda_e_per_nm"]
    f = {"PBEsol": b0["Lambda_fit_e_per_nm"] / bulk,
         "HSE": sh["sensitivity"]["77K_mu2_0/constants_Dreyer2016_HSE"]["Lambda_fit_e_per_nm"] / bulk}
    out = {"description": __doc__.split("\n\n")[0], "f": f, "cases": {}}
    for name, xs in CASES.items():
        if xs is None:
            continue
        out["cases"][name] = {}
        for lab, D in (("PBEsol", d), ("HSE", d_hse)):
            F = M.algan(xs[0], 300.0, D)
            if xs[1] == "GaN strained to AlN":
                # the substrate is a GaN channel held at the in-plane lattice of AlN (as on
                # the thinnest channels of Elwaradi et al. 2023): its formal polarization is
                # the strained (improper) value and its in-plane lattice that of AlN
                G, A = M.algan(0.0, 300.0, D), M.algan(1.0, 300.0, D)
                S = dict(G, a=A["a"], P_H=PZ.p_h_improper(G, (A["a"] - G["a"]) / G["a"]))
            else:
                S = M.algan(xs[1], 300.0, D)
            eps = (S["a"] - F["a"]) / F["a"]
            lim = PZ.line_charge_limits(F, eps)
            lams = {"bare ends": lim["bulk-terminated"], "electron counting": lim["electron counting"],
                    "neutral": 0.0, "Shen fraction": f[lab] * lim["bulk-terminated"],
                    "trapped (Shen)": -abs(f[lab] * lim["bulk-terminated"])}
            row = {"eps_c": eps, "coherent_cm2": PZ.to_cm2(PZ.sigma_decomposed(F, S, 0.0, 0.0))}
            for k, lam in lams.items():
                s1 = PZ.to_cm2(PZ.sigma_decomposed(F, S, 1.0, lam))
                row[k] = {"relaxed_cm2": s1, "slope_cm2_per_unit_R": s1 - row["coherent_cm2"],
                          "Lambda_e_per_nm": lam / PZ.QE * 1e-9 if hasattr(PZ, "QE") else lam / 1.602176634e-19 * 1e-9}
            # fraction of the bare-end charge at which the relaxed charge is zero
            s_n, s_b = row["neutral"]["relaxed_cm2"], row["bare ends"]["relaxed_cm2"]
            row["f_zero_relaxed"] = s_n / (s_n - s_b) if s_n != s_b else None
            row["relaxed_C_m2"] = {k: row[k]["relaxed_cm2"] * 1e4 * 1.602176634e-19 for k in lams}
            row["diff_R0.2_C_m2"] = {
                "bound minus trapped": 0.2 * (row["Shen fraction"]["slope_cm2_per_unit_R"]
                                             - row["trapped (Shen)"]["slope_cm2_per_unit_R"]) * 1.602176634e-15,
                "bound minus neutral": 0.2 * (row["Shen fraction"]["slope_cm2_per_unit_R"]
                                             - row["neutral"]["slope_cm2_per_unit_R"]) * 1.602176634e-15}
            out["cases"][name][lab] = row
            print(name, lab, json.dumps({k: (round(v["slope_cm2_per_unit_R"] / 1e13, 2), round(v["relaxed_cm2"] / 1e13, 2))
                                         for k, v in row.items() if isinstance(v, dict) and "slope_cm2_per_unit_R" in v}),
                  "f_zero %.2f" % row["f_zero_relaxed"],
                  "coh %.2f" % (row["coherent_cm2"] / 1e13))
    # Splitting the relaxation charge by the sign of the mismatch. To first order,
    # sigma(r) - sigma_coh = 2 r eps (k - f P_H) - 2 r |eps| q / b, with k the proper
    # piezoelectric combination, f the bound fraction and q >= 0 the trapped charge per
    # length. Per unit 2 r |eps|, y = s (k - f P_H) - q / b with s the sign of eps. The
    # same alloy stretched (s = +1) and compressed (s = -1) gives an even part
    # (y+ + y-)/2 = -q/b, free of all polarization constants, and an odd part
    # (y+ - y-)/2 = k - f P_H. Al0.3Ga0.7N on relaxed GaN is stretched, Al0.3Ga0.7N on a
    # GaN channel held at the lattice of AlN is compressed.
    QE_ = 1.602176634e-19
    eo = {}
    for lab in ("PBEsol", "HSE"):
        t, c = out["cases"]["Al0.30Ga0.70N on GaN"][lab], out["cases"]["Al0.30Ga0.70N on GaN strained to AlN"][lab]
        r = {}
        for k in ("bare ends", "neutral", "Shen fraction", "trapped (Shen)", "electron counting"):
            yp = t[k]["slope_cm2_per_unit_R"] * 1e4 * QE_ / (2 * abs(t["eps_c"]))      # C/m^2
            ym = c[k]["slope_cm2_per_unit_R"] * 1e4 * QE_ / (2 * abs(c["eps_c"]))
            r[k] = {"y_stretched_C_m2": yp, "y_compressed_C_m2": ym, "even_C_m2": (yp + ym) / 2, "odd_C_m2": (yp - ym) / 2}
        a = M.algan(0.30, 300.0, d if lab == "PBEsol" else d_hse)["a"] * 1e-10
        lam_t = abs(t["trapped (Shen)"]["Lambda_e_per_nm"]) * QE_ * 1e9
        r["minus_q_over_b_trapped_C_m2"] = -lam_t / a
        # holography: error of Lambda from an error dsig of the interface charge at
        # relaxation r of the compressed barrier, dLambda = dsig / L_A
        L = lambda rr: 2 * rr * abs(c["eps_c"]) / a
        r["dLambda_e_per_nm_for_dsig_1e-3"] = {str(rr): 1e-3 / L(rr) / QE_ * 1e-9 for rr in (0.25, 0.5)}
        # error of the trapped charge q from the sum y+ + y- = -2 q / b, with an error
        # dsig = 1e-3 C/m^2 on each measured charge change and both layers relaxed by rr
        def dq(rr):
            e1 = 1e-3 / (2 * rr * abs(t["eps_c"]))
            e2 = 1e-3 / (2 * rr * abs(c["eps_c"]))
            return a / 2 * (e1 ** 2 + e2 ** 2) ** 0.5 / QE_ * 1e-9          # e/nm
        r["dq_e_per_nm_for_dsig_1e-3"] = {str(rr): dq(rr) for rr in (0.25, 0.5)}
        r["sum_trapped_C_m2"] = 2 * r["minus_q_over_b_trapped_C_m2"]
        r["q_trapped_e_per_nm"] = lam_t / QE_ * 1e-9
        eo[lab] = r
    out["odd_even_Al0.30"] = eo
    print("odd/even", json.dumps({lab: {k: (round(v["even_C_m2"], 4), round(v["odd_C_m2"], 4)) for k, v in r.items()
                                        if isinstance(v, dict) and "even_C_m2" in v} for lab, r in eo.items()}))
    print("q/b", {lab: round(r["minus_q_over_b_trapped_C_m2"], 4) for lab, r in eo.items()},
          "dLambda", {lab: r["dLambda_e_per_nm_for_dsig_1e-3"] for lab, r in eo.items()},
          "dq", {lab: r["dq_e_per_nm_for_dsig_1e-3"] for lab, r in eo.items()})
    json.dump(out, open(os.path.join(ROOT, "results", "prediction.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
