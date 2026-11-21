"""Write paper/numbers.tex: every number quoted in the Letter and the
Supplemental Material that comes from a calculation, read from results/*.json,
so that the text cannot drift from the results.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import materials as M, polarization as PZ  # noqa: E402

ROOT = os.path.join(HERE, "..")
R = os.path.join(ROOT, "results")
QE = 1.602176634e-19


def J(n):
    return json.load(open(os.path.join(R, n)))


def f(x, n=2):
    s = ("%." + str(n) + "f") % round(float(x), 6)
    return s.replace("-", "-") if not s.startswith("-") else "-" + s[1:]


def main():
    out = {}
    pol, bulk = J("dft_polarization.json"), J("dft_bulk.json")
    for m in ("AlN", "GaN"):
        p = pol[m]
        a = bulk[m + "_ecut70"]["a"]
        pzb_f = p["zb"]["P_mod_Q"] / p["zb"]["quantum"] * p["quantum_Cm2"]
        out["PH" + m] = f(p["P_H"], 3)
        out["PZB" + m] = f(p["P_H"] - 0.75 * p["quantum_Cm2"], 3)
        out["e" + m] = f(p["e31_proper"], 3)
        out["eI" + m] = f(p["e31_improper_H"], 3)
        out["eT" + m] = f(p["e33"], 3)
        out["C" + m] = f(p["C13_over_C33"], 3)
        out["Q" + m] = f(p["quantum_Cm2"], 3)
        out["ZB" + m] = f(p["zb"]["P_mod_Q"], 4)
        out["ZBratio" + m] = f(p["zb"]["P_mod_Q"] / p["zb"]["quantum"], 4)
        out["PfZB" + m] = f(0.75 * p["quantum_Cm2"], 4)
        out["GapMin" + m] = f(p["min_gap_on_path_eV"], 1)
        out["a" + m] = f(a, 4)
        out["c" + m] = f(bulk[m + "_ecut70"]["c"], 4)
        out["u" + m] = f(bulk[m + "_ecut70"]["u"], 4)
        out["aConv" + m] = f(100 * abs(bulk[m + "_ecut80"]["a"] / a - 1), 2)
    if os.path.exists(os.path.join(R, "dft_slab.json")):
        s = J("dft_slab.json")
        for m in ("AlN", "GaN"):
            sig = s[m]["sigma_from_field_Cm2"]
            out["SlabSig" + m] = f(sig[-1], 3)
            out["SlabSigSix" + m] = f(sig[0], 3)   # thinner slab (first size)
            out["EpsInf" + m] = f(s[m]["eps_inf"]["eps_zz"], 2)
            out["SlabDiff" + m] = f(abs(sig[-1] - sig[0]), 3)
        out["SlabGapMin"] = f(min(r["gap_eV"] for m in ("AlN", "GaN") for r in s[m]["slabs"]), 1)
    # AlGaN line-charge limits (300 K lattice constants)
    d = M.dft()
    bar = M.algan(0.22, 300.0, d)
    b = bar["a"] * 1e-10
    out["LamBulkAlGaN"] = f(bar["P_H"] * b / QE * 1e-9, 2)
    out["LamECRAlGaN"] = f(abs(bar["P_H"] - PZ.p_zb_formal(bar["a"])) * b / QE * 1e-9, 2)
    # Shen analysis
    sh = J("shen_analysis.json")
    b0 = sh["baseline"]["77K_mu2_0"]
    pr = b0["predictions_t100"]
    out["MeasSeventySeven"] = f(b0["samples"][-1]["n2DEG_from_data"] / 1e13, 2)
    out["PredBulkSeventySeven"] = f(pr["bulk-terminated (H, improper)"]["n2DEG_model"] / 1e13, 2)
    out["PredNeutralSeventySeven"] = f(pr["neutral lines"]["n2DEG_model"] / 1e13, 2)
    out["PredECRSeventySeven"] = f(pr["electron counting"]["n2DEG_model"] / 1e13, 2)
    out["CohSeventySeven"] = f(b0["coherent_t100"]["n2DEG_model"] / 1e13, 2)
    out["OverNeutral"] = "%d" % round(100 * (pr["neutral lines"]["n2DEG_model"] /
                                             b0["samples"][-1]["n2DEG_from_data"] - 1))
    out["DsigSeventySeven"] = f(b0["dsig_cm2"] / 1e12, 1)
    out["RmsSeventySeven"] = "%d" % round(100 * b0["rms_log_misfit_pseudomorphic"])
    out["LamFitSeventySeven"] = f(b0["Lambda_fit_e_per_nm"], 2)
    out["MuRelaxSeventySeven"] = "%d" % round(b0["mobility"]["mu2DEG_relaxed"])
    out["LamMobStrictBase"] = f(b0["mobility"]["Lambda_strict_upper_bound_e_per_nm"], 2)
    out["LamMobEst"] = f(np.mean(b0["mobility"]["Lambda_estimate_range"]), 1)
    lb = abs(pr["bulk-terminated (H, improper)"]["Lambda_e_per_nm"])
    out["ScatFactor"] = "%.0f" % np.floor((lb / b0["mobility"]["Lambda_strict_upper_bound_e_per_nm"]) ** 2)
    fits, nocal, stricts, ests = [], [], [], []
    for key in ("77K_mu2_0", "77K_mu2_83"):
        bb = sh["baseline"][key]
        fits += [abs(x) for x in bb["Lambda_fit_range_over_phi_grid"]]
        stricts.append(bb["mobility"]["Lambda_strict_upper_bound_e_per_nm"])
        ests += bb["mobility"]["Lambda_estimate_range"]
        for k, v in sh["sensitivity"].items():
            if k.startswith(key + "/"):
                if k.endswith("no_offset_calibration"):
                    nocal.append(abs(v["Lambda_fit_e_per_nm"]))
                elif v["Lambda_fit_e_per_nm"] is not None:
                    fits.append(abs(v["Lambda_fit_e_per_nm"]))
                stricts.append(v["Lambda_strict_upper_bound_e_per_nm"])
                ests += v["Lambda_estimate_range"]
    out["LamFitLo"] = f(min(fits), 1)
    out["LamFitHi"] = f(max(fits), 1)
    out["LamFitNoCalLo"] = f(min(nocal), 1)
    out["LamFitNoCalHi"] = f(max(nocal), 1)
    out["LamMobStrictLo"] = f(min(stricts), 2)
    out["LamMobStrict"] = f(max(stricts), 2)
    out["LamMobEstLo"] = f(min(ests), 1)
    out["LamMobEstHi"] = f(max(ests), 1)
    b8 = sh["baseline"]["77K_mu2_83"]
    out["LamFitParallel"] = f(b8["Lambda_fit_e_per_nm"], 2)
    out["LamMobStrictParallel"] = f(b8["mobility"]["Lambda_strict_upper_bound_e_per_nm"], 2)
    out["MeasParallel"] = f(b8["samples"][-1]["n2DEG_from_data"] / 1e13, 2)
    out["RSevFiveFit"] = f(b0["R_needed_t75"]["fitted line charge"], 2)
    out["RSevFiveNeutral"] = f(b0["R_needed_t75"]["neutral lines"], 2)
    out["RSevFiveBulk"] = f(b0["R_needed_t75"]["bulk-terminated (H, improper)"], 2)
    out["LamFitPhiLo"] = f(abs(max(b0["Lambda_fit_range_over_phi_grid"])), 2)
    out["LamFitPhiHi"] = f(abs(min(b0["Lambda_fit_range_over_phi_grid"])), 2)
    b3 = sh["baseline"]["300K_mu2_83"]
    out["LamFitThreeHundred"] = f(b3["Lambda_fit_e_per_nm"], 1)
    out["LamMobThreeHundred"] = f(b3["mobility"]["Lambda_strict_upper_bound_e_per_nm"], 1)
    t300 = [abs(v["Lambda_fit_e_per_nm"]) for k, v in sh["sensitivity"].items()
            if k.startswith("300K") and v["Lambda_fit_e_per_nm"] is not None]
    out["LamFitThreeHundredLo"] = f(min(t300), 1)
    out["LamFitThreeHundredHi"] = f(max(t300), 1)
    lec = pr["electron counting"]["Lambda_e_per_nm"]
    lfit = b0["Lambda_fit_e_per_nm"]
    lbt = pr["bulk-terminated (H, improper)"]["Lambda_e_per_nm"]
    out["CompFrac"] = "%d" % round(100 * (lfit - lbt) / (lec - lbt))
    out["CompFracLo"] = "%d" % round(100 * (-float(out["LamFitHi"]) - lbt) / (lec - lbt))
    out["CompFracHi"] = "%d" % round(100 * (-float(out["LamFitLo"]) - lbt) / (lec - lbt))
    # Wang et al. example: AlN half relaxed on GaN
    aln = M.algan(1.0, 300.0, d)
    gan = M.algan(0.0, 300.0, d)
    eps_c = (gan["a"] - aln["a"]) / aln["a"]
    lim = PZ.line_charge_limits(aln, eps_c)
    s0 = PZ.sigma_decomposed(aln, gan, 0.5, 0.0)
    out["WangBulk"] = f(PZ.to_cm2(PZ.sigma_decomposed(aln, gan, 0.5, lim["bulk-terminated"]) - s0) / 1e13, 2)
    out["WangECR"] = f(PZ.to_cm2(PZ.sigma_decomposed(aln, gan, 0.5, lim["electron counting"]) - s0) / 1e13, 2)
    frac = (lfit - lbt) / (lec - lbt)
    lam_w = lim["bulk-terminated"] + frac * (lim["electron counting"] - lim["bulk-terminated"])
    out["WangFit"] = f(PZ.to_cm2(PZ.sigma_decomposed(aln, gan, 0.5, lam_w) - s0) / 1e13, 2)
    # GaN/AlN hole gas
    if os.path.exists(os.path.join(R, "prb_link.json")):
        L = J("prb_link.json")
        key = "misfit + threading lines, mu_q_H = 95.0"
        out["RelaxBulk"] = f(100 * L["cases"]["bulk-terminated (H, improper)"]["fits"][key]["relaxed_fraction"], 1)
        out["RelaxShen"] = f(100 * L["cases"]["Shen et al., largest strict mobility bound"]["fits"][key]
                             ["relaxed_fraction"], 1)
        dps = [abs(v["fits"][k]["delta_relative_to_measured_density"]) for v in L["cases"].values()
               for k in v["fits"] if v["fits"][k]["possible"]]
        out["DpsMax"] = "%d" % np.ceil(100 * max(dps))
    extra(out)
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/make_numbers.py from results/*.json; do not edit\n")
        for k, v in sorted(out.items()):
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(json.dumps(out, indent=1))


def extra(out):
    """Numbers from the analyses added in the second revision."""
    sh = J("shen_analysis.json")
    data = json.load(open(os.path.join(ROOT, "data", "shen2000_fig2_digitized.json")))
    r = np.array(data["ns_300K_1e13cm2"]) / np.array(data["ns_77K_1e13cm2"])
    out["RatioStrainedLo"], out["RatioStrainedHi"] = f(min(r[:4]), 2), f(max(r[:4]), 2)
    out["RatioRelaxLo"], out["RatioRelaxHi"] = f(min(r[4:]), 2), f(max(r[4:]), 2)
    if os.path.exists(os.path.join(R, "shen_uncertainty.json")):
        U = J("shen_uncertainty.json")
        b = U["phi"]["0.5"]["fit_B"]
        out["LamLP"] = f(b["values"]["Lambda_e_per_nm"], 2)
        out["LamLPsig"] = f(b["sigma"]["Lambda_e_per_nm"], 2)
        out["FracLP"] = f(b["fraction_of_bulk"], 2)
        out["FracLPsig"] = f(b["fraction_sigma"], 2)
        out["NeutralSig"] = f(b["fraction_of_bulk"] / b["fraction_sigma"], 1)
        out["BulkSig"] = f((1 - b["fraction_of_bulk"]) / b["fraction_sigma"], 1)
        out["ScatterLP"] = f(U["phi"]["0.5"]["fit_A"]["residual_sd_1e12"], 1)
        out["PvalLP"] = f(b["goodness"]["p_value"], 2)
        fr = [v["fit_B"]["fraction_of_bulk"] for v in U["phi"].values()]
        out["FracLPLo"], out["FracLPHi"] = f(min(fr), 2), f(max(fr), 2)
        bs = [(1 - v["fit_B"]["fraction_of_bulk"]) / v["fit_B"]["fraction_sigma"] for v in U["phi"].values()]
        ns = [v["fit_B"]["fraction_of_bulk"] / v["fit_B"]["fraction_sigma"] for v in U["phi"].values()]
        out["BulkSigLo"], out["NeutralSigHi"] = f(min(bs), 1), f(max(ns), 1)
    if os.path.exists(os.path.join(R, "shen_300K.json")):
        C = J("shen_300K.json")
        out["NoChanRms"] = "%d" % round(100 * C["no_channel_rms_log"])
        out["ChanRms"] = "%d" % round(100 * C["best"]["rms_log"])
        out["ChanN"] = f(C["best"]["n_p_cm2"] / 1e12, 1)
        out["ChanMu"] = "%d" % (10 * round(C["best"]["mu_p_cm2Vs"] / 10))
        out["ChanBarrierRms"] = "%d" % round(100 * C["with_model_barrier_channel"]["best"]["rms_log"])
        if "analysis_300K_channel_removed" in C:
            a = C["analysis_300K_channel_removed"]
            out["LamChanRemoved"] = f(a["Lambda_fit_e_per_nm"], 2)
    if os.path.exists(os.path.join(R, "hiroki_analysis.json")):
        Hh = J("hiroki_analysis.json")
        hd = json.load(open(os.path.join(ROOT, "data", "hiroki2002_digitized.json")))
        rt = hd["fig2a_40nm"]
        rr = [a / b for a, b in zip(rt["ns_RT_1e12"][:2], rt["ns_77K_1e12"][:2])]
        out["HRatioLo"], out["HRatioHi"] = f(min(rr), 2), f(max(rr), 2)

        def rng(xs, t, name):
            v = [Hh[tr][xs]["phi"][p]["relaxed"][t]["R_needed"][name] for tr in ("as_measured", "scaled_to_77K")
                 for p in Hh[tr][xs]["phi"]]
            return v
        shen_best = [k for k in Hh["as_measured"]["0.35"]["phi"]["0.5"]["relaxed"]["85"]["R_needed"]
                     if k.startswith("Shen fraction best")][0]
        for xs, t, tag in (("0.35", "85", "Hc"), ("0.25", "90", "Hq"), ("0.45", "30", "Hf")):
            for name, nm in (("neutral", "Neutral"), ("bulk-terminated", "Bulk"), (shen_best, "Shen")):
                v = rng(xs, t, name)
                ok = [x for x in v if x is not None]
                out[tag + nm + "Lo"] = f(min(ok), 2) if ok else "none"
                out[tag + nm + "Hi"] = f(max(ok), 2) if ok else "none"
                out[tag + nm + "None"] = "%d" % (len(v) - len(ok))
            out[tag + "Cases"] = "%d" % len(rng(xs, t, "neutral"))
        m = hd["fig4b_map_85nm_x0.35"]["R_of_contour_pixels"]
        out["HmapRmax"], out["HmapRmed"], out["HmapRmin"] = f(m["max"], 2), f(m["median"], 2), f(-m["min"], 2)
        out["HShenF"] = f(Hh["shen_fraction"]["best"], 2)
    if os.path.exists(os.path.join(R, "hiroki_map.json")):
        Mp = J("hiroki_map.json")
        v = [c["0.2"] for c in Mp["cases"].values() if c["0.2"] is not None]
        out["HfracLo"], out["HfracHi"] = f(min(v), 2), f(max(v), 2)
        v1 = [c["0.1"] for c in Mp["cases"].values() if c["0.1"] is not None]
        out["HfracOneLo"] = f(min(v1), 2)
    if os.path.exists(os.path.join(R, "kp_check.json")):
        K = J("kp_check.json")
        ref = [s[0] for s in K["subbands_ref"]]
        dev = [abs(a / b - 1) for a, b in zip(K["occupations_cm2"][:4], ref)]
        out["KpOccDev"] = f(100 * max(dev), 1)
        out["KpCentroid"] = f(K["centroid_nm"]["kpenvelope"], 2)
        out["KpCentroidRef"] = f(K["centroid_nm"]["gan2dhg (PRB archive)"], 2)
        out["KpRateDev"] = f(100 * abs(K["line_rate_ratio_kp_over_ref_same_subbands"] - 1), 1)
    if os.path.exists(os.path.join(R, "prediction.json")):
        P = J("prediction.json")
        for case, pre in (("Al0.30Ga0.70N on GaN strained to AlN", "Slope"), ("GaN on AlN", "SlopeGA")):
            g = P["cases"][case]
            for k, nm in (("bare ends", "Bare"), ("Shen fraction", "Bound"), ("neutral", "Neutral"),
                          ("trapped (Shen)", "Trap")):
                v = sorted(g[c][k]["slope_cm2_per_unit_R"] / 1e13 for c in ("PBEsol", "HSE"))
                out[pre + nm + "Lo"], out[pre + nm + "Hi"] = f(v[0], 1), f(v[1], 1)
            v = sorted(g[c]["diff_R0.2_C_m2"]["bound minus trapped"] for c in ("PBEsol", "HSE"))
            out[pre.replace("Slope", "DiffBT") + "Lo"], out[pre.replace("Slope", "DiffBT") + "Hi"] = f(v[0], 3), f(v[1], 3)
            v = sorted(abs(g[c]["eps_c"]) * 100 for c in ("PBEsol", "HSE"))
            out[pre.replace("Slope", "Mis")] = f(v[0], 1)
            if pre == "Slope":
                c = g["PBEsol"]
                out["SlopePerLam"] = f((c["Shen fraction"]["slope_cm2_per_unit_R"] - c["neutral"]["slope_cm2_per_unit_R"])
                                       / c["Shen fraction"]["Lambda_e_per_nm"] / 1e13, 2)
        a3 = P["cases"]["Al0.30Ga0.70N on GaN"]
        out["YeBareHSE"] = f(-a3["HSE"]["relaxed_C_m2"]["bare ends"], 3)
        v = sorted(a3[c]["Shen fraction"]["relaxed_cm2"] / 1e13 for c in ("PBEsol", "HSE"))
        out["AlThirtyShenLo"], out["AlThirtyShenHi"] = f(v[0], 1), f(v[1], 1)
        v = sorted(a3[c]["neutral"]["relaxed_cm2"] / 1e13 for c in ("PBEsol", "HSE"))
        out["AlThirtyNeutralLo"], out["AlThirtyNeutralHi"] = f(v[0], 1), f(v[1], 1)
        out["FzeroHSE"] = f(a3["HSE"]["f_zero_relaxed"], 2)
        out["FzeroPBE"] = f(a3["PBEsol"]["f_zero_relaxed"], 2)
        eo = P["odd_even_Al0.30"]
        v = sorted(abs(eo[c]["sum_trapped_C_m2"]) for c in ("PBEsol", "HSE"))
        out["SumTrapLo"], out["SumTrapHi"] = f(v[0], 2), f(v[1], 2)
        out["EvenBoundMax"] = "%.0e" % max(abs(eo[c][k]["even_C_m2"]) for c in ("PBEsol", "HSE")
                                          for k in ("bare ends", "neutral", "Shen fraction", "electron counting"))
        out["DqHalf"] = f(eo["PBEsol"]["dq_e_per_nm_for_dsig_1e-3"]["0.5"], 2)
        out["DqQuarter"] = f(eo["PBEsol"]["dq_e_per_nm_for_dsig_1e-3"]["0.25"], 2)
        out["DlamHalf"] = f(eo["PBEsol"]["dLambda_e_per_nm_for_dsig_1e-3"]["0.5"], 2)
        out["DlamQuarter"] = f(eo["PBEsol"]["dLambda_e_per_nm_for_dsig_1e-3"]["0.25"], 2)
        out["MisStretch"] = f(abs(P["cases"]["Al0.30Ga0.70N on GaN"]["PBEsol"]["eps_c"]) * 100, 2)
        out["FracHSE"] = f(P["f"]["HSE"], 2)
        out["FracPBE"] = f(P["f"]["PBEsol"], 2)
    if os.path.exists(os.path.join(R, "shen_uncertainty_hse.json")):
        bh = J("shen_uncertainty_hse.json")["phi"]["0.5"]["fit_B"]
        out["FracLPHSE"] = f(bh["fraction_of_bulk"], 2)
        out["FracLPHSEsig"] = f(bh["fraction_sigma"], 2)
    if os.path.exists(os.path.join(R, "shen_uncertainty.json")):
        b = J("shen_uncertainty.json")["phi"]["0.5"]["fit_B"]
        out["LamOneSigLo"] = f(abs(b["values"]["Lambda_e_per_nm"]) - b["sigma"]["Lambda_e_per_nm"], 1)
        out["LamOneSigHi"] = f(abs(b["values"]["Lambda_e_per_nm"]) + b["sigma"]["Lambda_e_per_nm"], 1)
        b0 = J("shen_analysis.json")["baseline"]["77K_mu2_0"]
        out["LineSpacing"] = "%d" % (10 * round(b0["line_spacing_nm_hexagonal"] / 10))
        out["StripWidth"] = "%d" % round(abs(b["values"]["Lambda_e_per_nm"]) / (b0["mobility"]["n2DEG_relaxed"] * 1e-14))
        out["StripWidthBare"] = "%d" % round(abs(b0["predictions_t100"]["bulk-terminated (H, improper)"]["Lambda_e_per_nm"])
                                            / (b0["mobility"]["n2DEG_relaxed"] * 1e-14))
    if os.path.exists(os.path.join(R, "elwaradi_analysis.json")):
        E = J("elwaradi_analysis.json")
        bare, others, lfit, rms = [], {}, [], []
        for key, row in E["cases"].items():
            rms.append(row["rms_all_four_after_mean_cal"])
            for rk, rr in row["t20"].items():
                for cal, pred in rr["pred"].items():
                    if cal == "none":
                        continue
                    bare.append(pred["bare ends"])
                    for h, v in pred.items():
                        others.setdefault(h, []).append(v)
                for cal, L in rr["Lambda_fit_e_per_nm"].items():
                    if cal != "none" and L is not None:
                        lfit.append(L)
        out["ElwBareLo"], out["ElwBareHi"] = f(min(bare) / 1e12, 1), f(max(bare) / 1e12, 1)
        out["ElwBareBelow"] = "%d" % sum(1 for b in bare if b < 2.2e12)
        for h, nm in (("neutral", "Neutral"), ("bound (Shen fraction)", "Bound"), ("trapped (Shen magnitude)", "Trap")):
            out["Elw" + nm + "Lo"], out["Elw" + nm + "Hi"] = f(min(others[h]) / 1e12, 1), f(max(others[h]) / 1e12, 1)
        out["ElwLamLo"], out["ElwLamHi"] = f(min(lfit), 1), f(max(lfit), 1)
        out["ElwRmsLo"], out["ElwRmsHi"] = "%d" % round(100 * min(rms)), "%d" % round(100 * max(rms))
        out["ElwNcases"] = "%d" % len(bare)
        if "mobility_t20" in E:
            mb = E["mobility_t20"]
            out["ElwMobStrictLo"] = f(min(v["Lambda_strict_upper_bound_e_per_nm"] for v in mb.values()), 1)
            out["ElwMobStrictHi"] = f(max(v["Lambda_strict_upper_bound_e_per_nm"] for v in mb.values()), 1)
            out["ElwLamBare"] = f(abs(list(mb.values())[0]["Lambda_bare_e_per_nm"]), 2)
            out["ElwMobEstLo"] = f(min(v["Lambda_estimate_vs_50nm_e_per_nm"] for v in mb.values()), 1)
            out["ElwMobEstHi"] = f(max(v["Lambda_estimate_vs_50nm_e_per_nm"] for v in mb.values()), 1)
            out["ElwScat"] = "%.1f" % (abs(list(mb.values())[0]["Lambda_bare_e_per_nm"])
                                       / max(v["Lambda_strict_upper_bound_e_per_nm"] for v in mb.values())) ** 2
    if os.path.exists(os.path.join(R, "jiang_analysis.json")):
        Jg = J("jiang_analysis.json")
        rat = [x for p in Jg["phi"].values() for x in p["measured_over_model"][2:]]
        out["JiangRatioLo"], out["JiangRatioHi"] = f(min(rat), 2), f(max(rat), 2)
        for k, nm in (("bulk-terminated (H, improper)", "Bare"), ("Shen fraction", "Shen"),
                      ("neutral lines", "Neutral")):
            v = [p["R_for_10pct_drop_100nm"][k] for p in Jg["phi"].values()]
            out["JiangR" + nm + "Lo"], out["JiangR" + nm + "Hi"] = f(min(v), 2), f(max(v), 2)
        c = {r["t_nm"]: r for r in Jg["comparison"]}
        out["JiangHundred"] = f(c[100]["jiang_SdH_1.4K"] / 1e13, 2)
        out["ShenHundred"] = f(c[100]["shen_Hall_77K"] / 1e13, 2)
    # straight three-direction misfit network (scripts/run_line_network_check.py)
    if os.path.exists(os.path.join(R, "line_network_check.json")):
        Nw = J("line_network_check.json")
        rat = [c["tau_network_over_kernel_N2400"] for c in Nw["cases"]]
        out["NetTauLo"], out["NetTauHi"] = f(min(rat), 1), f(max(rat), 1)
        out["NetDirCheck"] = "%d" % round(100 * max(abs(c["tau_24_directions_over_kernel"] - 1) for c in Nw["cases"]))
        sh = Nw["shen"]
        out["LamMobNet"] = f(sh["77K_mu2_0"]["strict_network"], 1)
        sn = [v["strict_network"] for v in sh.values()]
        out["LamMobNetLo"], out["LamMobNetHi"] = f(min(sn), 1), f(max(sn), 1)
        e77 = [x for k, v in sh.items() if k.startswith("77K") for x in v["estimate_network_with_background"]]
        out["LamMobNetEstLo"], out["LamMobNetEstHi"] = f(min(e77), 1), f(max(e77), 1)
        fac = [v["strict_network"] / v["strict_kernel"] for v in sh.values()]
        el = Nw["elwaradi"]
        fac += [v["strict_network"] / v["strict_kernel"] for v in el.values()]
        out["NetFacLo"], out["NetFacHi"] = f(min(fac), 1), f(max(fac), 1)
        out["ElwMobNetLo"] = f(min(v["strict_network"] for v in el.values()), 1)
        out["ElwMobNetHi"] = f(max(v["strict_network"] for v in el.values()), 1)
        out["ElwMobNetEstLo"] = f(min(v["estimate_network_with_background"] for v in el.values()), 1)
        out["ElwMobNetEstHi"] = f(max(v["estimate_network_with_background"] for v in el.values()), 1)
    # second order in the cell area: exact dilution (1 + e)^-2 against its linear form
    def geo(eps_c, r):
        e1, e2 = eps_c, (1 - r) * eps_c
        return abs(((1 + e2) ** -2 - (1 + e1) ** -2) / (-2 * (e2 - e1)) - 1)
    dd = M.dft()
    g0, f22, f1 = M.algan(0.0, 300.0, dd), M.algan(0.22, 300.0, dd), M.algan(1.0, 300.0, dd)
    out["GeoShen"] = f(100 * geo((g0["a"] - f22["a"]) / f22["a"], 0.78), 1)
    out["GeoAlN"] = f(100 * geo((g0["a"] - f1["a"]) / f1["a"], 0.5), 1)
    if os.path.exists(os.path.join(R, "second_order_check.json")):
        so = J("second_order_check.json")
        out["SecondShen"] = f(100 * max(v["ratio_to_bare_end_term"] for v in so[0]["results"].values()), 1)
        out["SecondAlN"] = f(100 * max(v["ratio_to_bare_end_term"] for v in so[1]["results"].values()), 1)
    if os.path.exists(os.path.join(R, "shen_spacer_analysis.json")):
        sp = J("shen_spacer_analysis.json")["cases"]
        c77 = [v for k, v in sp.items() if k.startswith("77K")]
        out["SpacerRms"] = "%d" % round(100 * max(v["prediction_rms_log_misfit"] for v in c77))
        out["SpacerDrop"] = "%d" % round(100 * (1 - c77[0]["samples"][6]["n2DEG_from_data"]
                                               / c77[0]["samples"][0]["n2DEG_from_data"]))
        rb = [v["s50"][cal]["bulk-terminated (H, improper)"]["R_needed"] for v in c77 for cal in v["s50"]]
        rf = [v["s50"][cal]["line charge of the thickness series"]["R_needed"] for v in c77 for cal in v["s50"]]
        mb = [v["s50"][cal]["bulk-terminated (H, improper)"]["mobility_strict_bound_e_per_nm"]
              for v in c77 for cal in v["s50"]]
        loc = J("shen_spacer_local.json")
        rb += [v["bulk-terminated (H, improper)"] for v in loc.values()]
        rn = [v["neutral lines"] for v in loc.values()]
        mb += [sc[2] for v in loc.values() for sc in v["scan"] if abs(sc[0] + 2.645) < 1e-6]
        out["SpacerRBareLo"], out["SpacerRBareHi"] = f(min(rb), 2), f(max(rb), 2)
        out["SpacerRFitLo"], out["SpacerRFitHi"] = f(min(rf), 2), f(max(rf), 2)
        out["SpacerRNeutLo"], out["SpacerRNeutHi"] = f(min(rn), 2), f(max(rn), 2)
        out["SpacerMobBareLo"], out["SpacerMobBareHi"] = f(min(mb), 1), f(max(mb), 1)


if __name__ == "__main__":
    main()


def tables():
    """paper/supp_tables.tex: the tables of the Supplemental Material."""
    sh = J("shen_analysis.json")
    L = []
    b0 = sh["baseline"]["77K_mu2_0"]
    b3 = sh["baseline"]["300K_mu2_83"]
    L.append(r"\begin{table}[h]\caption{\label{tab:shen}Series of Shen \textit{et al.}, measured Hall density"
             r" $n_\mathrm{H}$ and mobility $\mu_\mathrm{H}$ (digitized), and the calibrated model"
             r" for strained barriers, 2DEG density $n_1$ and electrons in the undepleted barrier"
             r" $n_2$. Densities in $10^{13}$~cm$^{-2}$, mobilities in cm$^2$V$^{-1}$s$^{-1}$.}")
    L.append(r"\begin{ruledtabular}\begin{tabular}{rcccccccc}")
    L.append(r"$t$ (nm) & \multicolumn{4}{c}{77 K} & \multicolumn{4}{c}{300 K} \\")
    L.append(r" & $n_\mathrm{H}$ & $\mu_\mathrm{H}$ & $n_1$ & $n_2$ & $n_\mathrm{H}$ & $\mu_\mathrm{H}$ & $n_1$ & $n_2$ \\ \hline")
    for s7, s3 in zip(b0["samples"], b3["samples"]):
        L.append("%d & %.3f & %d & %.3f & %.3f & %.3f & %d & %.3f & %.3f \\\\" % (
            s7["t_nm"], s7["nH_measured"] / 1e13, round(s7["muH_measured"]), s7["n2DEG_model"] / 1e13,
            s7["n_barrier_model"] / 1e13, s3["nH_measured"] / 1e13, round(s3["muH_measured"]),
            s3["n2DEG_model"] / 1e13, s3["n_barrier_model"] / 1e13))
    L.append(r"\end{tabular}\end{ruledtabular}\end{table}")
    # baseline
    L.append(r"\begin{table*}[t]\caption{\label{tab:base}Relaxed sample ($t=100$~nm, $R=0.78$),"
             r" calibration, predicted 2DEG density for the three line charges, the measured"
             r" 2DEG density (after removing the parallel channel where $\mu_2>0$), the line charge"
             r" that reproduces it (baseline, and range over $\phi_B=0.5$ to $2$~eV), the mobility"
             r" bounds, and the relaxation of the $t=75$~nm sample that each line charge requires."
             r" Densities in $10^{13}$~cm$^{-2}$, $\Lambda$ in e\,nm$^{-1}$, $\delta\sigma$ in $10^{12}$~cm$^{-2}$.}")
    L.append(r"\begin{ruledtabular}\begin{tabular}{lccc}")
    L.append(r" & 77 K, $\mu_2=0$ & 77 K, $\mu_2=83$ & 300 K, $\mu_2=83$ \\ \hline")
    keys = ("77K_mu2_0", "77K_mu2_83", "300K_mu2_83")
    B = [sh["baseline"][k] for k in keys]

    def row(lab, fn):
        L.append(lab + " & " + " & ".join(fn(b) for b in B) + r" \\")
    row(r"$\phi_B$ (eV)", lambda b: "%.1f" % b["phi_B_eV"])
    row(r"$\delta\sigma$", lambda b: "%.2f" % (b["dsig_cm2"] / 1e12))
    row(r"rms misfit, strained samples (\%)", lambda b: "%.1f" % (100 * b["rms_log_misfit_pseudomorphic"]))
    row(r"coherent barrier", lambda b: "%.3f" % (b["coherent_t100"]["n2DEG_model"] / 1e13))
    for nm, lab in (("bulk-terminated (H, improper)", "bare-end lines"),
                    ("electron counting", "electron-counting lines"), ("neutral lines", "neutral lines")):
        row(lab, lambda b, nm=nm: "%.3f" % (b["predictions_t100"][nm]["n2DEG_model"] / 1e13))
    row("measured 2DEG", lambda b: "%.3f" % (b["samples"][-1]["n2DEG_from_data"] / 1e13))
    row(r"$\Lambda$ from density", lambda b: "%.2f" % b["Lambda_fit_e_per_nm"])
    row(r"$\Lambda$ range over $\phi_B$", lambda b: "%.2f to %.2f" % tuple(sorted(b["Lambda_fit_range_over_phi_grid"])))
    row(r"$|\Lambda|$, strict mobility bound", lambda b: "%.2f" % b["mobility"]["Lambda_strict_upper_bound_e_per_nm"])
    row(r"$|\Lambda|$, from mobility drop", lambda b: "%.2f to %.2f" % tuple(b["mobility"]["Lambda_estimate_range"]))
    for nm, lab in (("bulk-terminated (H, improper)", "bare ends"), ("electron counting", "electron counting"),
                    ("neutral lines", "neutral"), ("fitted line charge", r"fitted $\Lambda$")):
        row(r"$R$ needed at $t=75$~nm, " + lab,
            lambda b, nm=nm: ("%.2f" % b["R_needed_t75"][nm]) if b["R_needed_t75"].get(nm) is not None else "$>1$")
    L.append(r"\end{tabular}\end{ruledtabular}\end{table*}")
    # sensitivity
    L.append(r"\begin{table*}[t]\caption{\label{tab:sens}Sensitivity tests (one change at a time from the"
             r" baseline). $\Lambda$ from the density and the strict mobility bound on $|\Lambda|$, in e\,nm$^{-1}$.}")
    L.append(r"\begin{ruledtabular}\begin{tabular}{lcccccc}")
    L.append(r" & \multicolumn{2}{c}{77 K, $\mu_2=0$} & \multicolumn{2}{c}{77 K, $\mu_2=83$} & \multicolumn{2}{c}{300 K, $\mu_2=83$} \\")
    L.append(r"test & $\Lambda$ & bound & $\Lambda$ & bound & $\Lambda$ & bound \\ \hline")
    names = {"cbo_0.65": "band-offset fraction 0.65", "cbo_0.75": "band-offset fraction 0.75",
             "eps_eq17": r"$\epsilon(x)$ from Eq.~(17) of Ref.~\onlinecite{Ambacher2000}",
             "psp_bowing_0.021": "spontaneous-polarization bowing", "R_0.70": "$R=0.70$", "R_0.85": "$R=0.85$",
             "fit_without_65nm": "calibration without $t=65$~nm", "dz_0.05": "grid 0.05~nm",
             "no_offset_calibration": r"no calibration ($\delta\sigma=0$)",
             "constants_Dreyer2016_HSE": r"constants of Ref.~\onlinecite{Dreyer2016}"}
    for v, lab in names.items():
        cells = []
        for k in keys:
            s = sh["sensitivity"].get(k + "/" + v)
            if s is None:
                cells += ["", ""]
                continue
            lf = s["Lambda_fit_e_per_nm"]
            cells += ["%.2f" % lf if lf is not None else "none", "%.2f" % s["Lambda_strict_upper_bound_e_per_nm"]]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\end{tabular}\end{ruledtabular}\end{table*}")
    # PRB link: kept in results/prb_link.json, no longer printed in the Supplemental
    # Material (its conversion assumes randomly oriented lines; see CHANGELOG 1.4.0)
    if False and os.path.exists(os.path.join(R, "prb_link.json")):
        P = J("prb_link.json")
        key = "misfit + threading lines, mu_q_H = 95.0"
        L.append(r"\begin{table}[h]\caption{\label{tab:prb}GaN/AlN hole gas, relaxation $r$ (percent of the"
                 r" mismatch) that misfit lines of charge $\Lambda$ need to supply the in-plane line scattering"
                 r" of Ref.~\onlinecite{Mahim2026} (fit with threading lines at $\mu_{q,H}=95$~cm$^2$V$^{-1}$s$^{-1}$),"
                 r" line spacing of a hexagonal network, and the resulting change of the bound charge relative"
                 r" to the measured $4.6\times10^{13}$~cm$^{-2}$.}")
        L.append(r"\begin{ruledtabular}\begin{tabular}{lcccc}")
        L.append(r"lines & $|\Lambda|$ (e/nm) & $r$ (\%) & spacing (nm) & $\Delta\sigma/ep_s$ (\%) \\ \hline")
        for nm, c in P["cases"].items():
            ff = c["fits"][key]
            L.append("%s & %.2f & %s & %s & %s \\\\" % (
                nm.replace("Shen et al.", r"Shen \textit{et al.}").replace("bulk-terminated (H, improper)", "bare ends (H recipe)"), abs(c["Lambda_e_per_nm"]),
                "%.2f" % (100 * ff["relaxed_fraction"]) if ff["possible"] else "$>100$",
                "%.0f" % ff["line_spacing_nm"] if ff["possible"] else "",
                "%.1f" % (100 * ff["delta_relative_to_measured_density"]) if ff["possible"] else ""))
        L.append(r"\end{tabular}\end{ruledtabular}\end{table}")
    # Hiroki et al.
    if os.path.exists(os.path.join(R, "hiroki_analysis.json")):
        Hh = J("hiroki_analysis.json")
        best = [k for k in Hh["as_measured"]["0.35"]["phi"]["0.5"]["relaxed"]["85"]["R_needed"]
                if k.startswith("Shen fraction best")][0]
        L.append(r"\begin{table*}[t]\caption{\label{tab:hiroki}Series of Hiroki \textit{et al.}, calibration"
                 r" of each composition on its strained samples (fixed sheet $\delta\sigma$ in $10^{12}$~cm$^{-2}$,"
                 r" rms misfit), model density of a strained barrier of the thickness of the relaxed sample, the"
                 r" measured density (both in $10^{12}$~cm$^{-2}$), and the relaxation $R$ that each line charge needs"
                 r" to reproduce the measured density (``none'' means not reached even at $R=1$). Upper half, room-temperature"
                 r" densities as measured; lower half, all densities multiplied by the 77~K/room-temperature ratio"
                 r" measured at $40$~nm.}")
        L.append(r"\begin{ruledtabular}\begin{tabular}{llcccccccc}")
        L.append(r"$x$, $t$ & $\phi_B$ (eV) & $\delta\sigma$ & rms (\%) & strained & measured & bare ends &"
                 r" Shen fraction & neutral & electron counting \\ \hline")
        for tr in ("as_measured", "scaled_to_77K"):
            for xs, t in (("0.25", "90"), ("0.35", "85"), ("0.45", "30")):
                for p, v in Hh[tr][xs]["phi"].items():
                    z = v["relaxed"][t]
                    rn = z["R_needed"]
                    cell = lambda k: ("%.2f" % rn[k]) if rn[k] is not None else "none"
                    L.append("%s, %s~nm & %s & %.2f & %.1f & %.2f & %.2f & %s & %s & %s & %s \\\\" % (
                        xs, t, p, v["dsig_cm2"] / 1e12, 100 * v["rms_log_misfit_coherent"],
                        z["n_coherent_model"] / 1e12, z["n_measured"] / 1e12, cell("bulk-terminated"),
                        cell(best), cell("neutral"), cell("electron counting")))
            if tr == "as_measured":
                L.append(r"\hline")
        L.append(r"\end{tabular}\end{ruledtabular}\end{table*}")
    if os.path.exists(os.path.join(R, "prediction.json")):
        P = J("prediction.json")
        L.append(r"\begin{table*}[t]\caption{\label{tab:pred}Interface charge of a fully strained"
                 r" layer, its change per unit relaxation $R$ for each line charge, and the charge of the"
                 r" fully relaxed interface with the line charge of the Shen series taken as bound, from"
                 r" Eq.~(2) of the Letter (\texttt{scripts/run\_prediction.py}). For a film under tension"
                 r" the bound and trapped readings coincide. Charges in $10^{13}$~cm$^{-2}$.}")
        L.append(r"\begin{ruledtabular}\begin{tabular}{llccccccc}")
        L.append(r"Interface & Constants & $\varepsilon_\mathrm{c}$ & strained & \multicolumn{4}{c}{change per unit $R$} & relaxed, \\")
        L.append(r" & & & & bare ends & bound & neutral & trapped & bound \\ \hline")
        for name, rows in P["cases"].items():
            for lab, r in rows.items():
                g = lambda k: "$%.2f$" % (r[k]["slope_cm2_per_unit_R"] / 1e13)
                tex = {"Al0.30Ga0.70N on GaN": "Al$_{0.30}$Ga$_{0.70}$N on GaN",
                       "Al0.22Ga0.78N on GaN": "Al$_{0.22}$Ga$_{0.78}$N on GaN",
                       "Al0.30Ga0.70N on GaN strained to AlN": "Al$_{0.30}$Ga$_{0.70}$N on GaN held at AlN"}.get(name, name)
                L.append("%s & %s & $%.4f$ & $%.2f$ & %s & %s & %s & %s & $%.2f$ \\\\" % (
                    tex, "PBEsol" if lab == "PBEsol" else "HSE", r["eps_c"], r["coherent_cm2"] / 1e13,
                    g("bare ends"), g("Shen fraction"), g("neutral"), g("trapped (Shen)"),
                    r["Shen fraction"]["relaxed_cm2"] / 1e13))
        L.append(r"\end{tabular}\end{ruledtabular}\end{table*}")
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    open(os.path.join(ROOT, "paper", "supp_tables.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    tables()
