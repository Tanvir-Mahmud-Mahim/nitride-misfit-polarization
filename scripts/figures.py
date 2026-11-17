"""Figures of the Letter (figures/fig1-3.pdf/png), from the results files only."""

import json
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import figstyle as FS  # noqa: E402
from figstyle import C, plt  # noqa: E402
from nmp import materials as M, polarization as PZ  # noqa: E402

R = os.path.join(FS.ROOT, "results")
QE = 1.602176634e-19


def load(n):
    return json.load(open(os.path.join(R, n)))


# --------------------------------------------------------------------- Fig. 1
def fig1():
    d = M.dft()
    fig = plt.figure(figsize=(7.0, 2.55))
    ax = fig.add_axes([0.0, 0.02, 0.50, 0.93])
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-1.4, 5.9)
    ax.axis("off")
    FS.label(ax, "(a)", x=0.0, y=0.97)
    # substrate (GaN) columns and film (AlGaN) columns, one extra film column
    ys, yi, yt = 0.4, 3.0, 5.6
    xs_sub = np.arange(0.9, 9.6, 1.0)
    xs_film = np.linspace(0.9, 9.5, 10)
    ax.add_patch(Rectangle((0.4, ys), 9.4, yi - ys, fc="#e8e8e8", ec="none"))
    ax.add_patch(Rectangle((0.4, yi), 9.4, yt - yi, fc="#d7e6f3", ec="none"))
    for x in xs_sub:
        ax.plot([x, x], [ys, yi], color="0.35", lw=0.8)
    for x in xs_film:
        ax.plot([x, x], [yi, yt], color=C["ecr"], lw=0.8)
    # mark the extra half-plane (the film column that has no partner below)
    xe = xs_film[5]
    ax.plot([xe, xe], [yi, yt], color=C["bulk"], lw=1.6)
    ax.plot(xe, yi, "o", ms=5, mfc="white", mec=C["bulk"], mew=1.2)
    box = dict(fc="white", ec="none", pad=1.2, alpha=0.95)
    ax.text(0.55, ys + 0.15, "relaxed substrate (GaN)", fontsize=6.5, color="0.25", bbox=box)
    ax.text(0.55, yt - 0.45, "partly relaxed film (AlGaN)", fontsize=6.5, color=C["ecr"], bbox=box)
    ax.annotate("end of an extra\nhalf-plane (misfit line,\nrunning into the page)",
                xy=(xe + 0.05, yi - 0.05), xytext=(6.3, 1.1), fontsize=6.3, ha="left", bbox=box,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2"))
    ax.add_patch(FancyArrowPatch((9.95, yi + 0.3), (9.95, yt - 0.3), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color="black"))
    ax.text(10.05, (yi + yt) / 2, "[0001]", rotation=90, va="center", fontsize=6.3)
    # line-charge options
    tx = 0.55
    ax.text(tx, -0.25, r"Charge per length of the line, $|\Lambda| = P_\mathrm{H}\,b$ for a bare end,"
            "\n" r"$|\Lambda| = |P_\mathrm{ZB}|\,b$ for electron counting, $\Lambda = 0$ for a neutral line",
            fontsize=6.3, va="top")
    # (b) bound charge vs relaxation for AlN on GaN (Wang et al.'s example)
    ax2 = fig.add_axes([0.60, 0.17, 0.38, 0.75])
    FS.label(ax2, "(b)", x=-0.2, y=1.0)
    aln = M.algan(1.0, 300.0, d)
    gan = M.algan(0.0, 300.0, d)
    eps_c = (gan["a"] - aln["a"]) / aln["a"]
    lim = PZ.line_charge_limits(aln, eps_c)
    r = np.linspace(0, 1, 101)
    s0 = PZ.sigma_decomposed(aln, gan, 0.0, 0.0)
    for name, col, lab in (("neutral", C["neutral"], r"neutral lines ($\Lambda=0$; ZB recipe)"),
                           ("electron counting", C["ecr"], "electron-counting lines"),
                           ("bulk-terminated", C["bulk"], "bare ends (H recipe)")):
        s = [PZ.to_cm2(PZ.sigma_decomposed(aln, gan, x, lim[name])) / 1e13 for x in r]
        ax2.plot(r, s, color=col, label=lab, ls="--" if name == "electron counting" else "-")
    ax2.set_xlabel(r"relaxed fraction $r$ of the AlN film")
    ax2.set_ylabel(r"interface charge ($10^{13}\,e\,\mathrm{cm}^{-2}$)")
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=False, loc="lower left", handlelength=1.8)
    FS.save(fig, "fig1")


# --------------------------------------------------------------------- Fig. 2
def fig2():
    pol = load("dft_polarization.json")
    slab = load("dft_slab.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    FS.label(ax, "(a)", x=-0.2)
    for name, col in (("AlN", C["ecr"]), ("GaN", C["bulk"])):
        p = pol[name]["path"]
        u = [x["u"] for x in p]
        P = np.array([x["P_unwrapped"] for x in p]) - p[-1]["P_unwrapped"]
        ax.plot(u, P, "o-", color=col, ms=2.5, label=name)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"internal parameter $u$")
    ax.set_ylabel(r"$P - P(u=1/2)$ (C m$^{-2}$)")
    ax.legend(frameon=False, loc="upper right")
    ax = axs[1]
    FS.label(ax, "(b)", x=-0.2)
    txt = []
    for name, col in (("AlN", C["ecr"]), ("GaN", C["bulk"])):
        s = slab[name]["slabs"][-1]
        dd = np.loadtxt(os.path.join(FS.ROOT, "dft", "runs", "slab_%s_%d" % (name, s["nbilayers"]),
                                     "potential_profile.txt"))
        z, vm = dd[:, 0], dd[:, 2]
        z0, z1 = s["nitride_extent_A"]
        zm = (z0 + z1) / 2
        sel = (z > z0 + 1.0) & (z < z1 - 1.0)
        v0 = np.interp(zm, z, vm)
        ax.plot(z[sel] - zm, vm[sel] - v0, color=col, label=name)
        m0, m1 = s["central_region_A"]
        zz = np.linspace(m0, m1, 2)
        ax.plot(zz - zm, s["slope_eV_per_A"] * (zz - zm), color="black", lw=0.6, ls="--")
        txt.append(r"%s   $\sigma=%.3f$, $P_\mathrm{ZB}=%.3f$" % (
            name, slab[name]["sigma_from_field_Cm2"][-1],
            pol[name]["P_H"] - 0.75 * pol[name]["quantum_Cm2"]))
    ax.text(0.04, 0.96, "\n".join(txt) + "\n" + r"(C m$^{-2}$)", transform=ax.transAxes, va="top", fontsize=6.3)
    ax.set_xlabel(r"position along [0001] ($\mathrm{\AA}$)")
    ax.set_ylabel("averaged potential energy (eV)")
    ax.legend(frameon=False, loc="lower right")
    FS.save(fig, "fig2")


# --------------------------------------------------------------------- Fig. 3
def fig3():
    sh = load("shen_analysis.json")
    hk = load("hiroki_analysis.json")
    unc = load("shen_uncertainty.json")
    c300 = load("shen_300K.json")
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
    axs = axs.ravel()
    b = sh["baseline"]["77K_mu2_0"]
    ax = axs[0]
    FS.label(ax, "(a)", x=-0.2)
    t = [s["t_nm"] for s in b["samples"]]
    ax.plot(t, [s["nH_measured"] / 1e13 for s in b["samples"]], "o", mfc="white", mec="black", ms=3.5,
            label="measured, 77 K")
    ax.plot(t[:4], [s["n2DEG_model"] / 1e13 for s in b["samples"][:4]], "-", color="0.4", label="model, strained")
    for name, col, mk in (("bulk-terminated (H, improper)", C["bulk"], "v"),
                          ("electron counting", C["ecr"], "^"), ("neutral lines", C["neutral"], "s")):
        ax.plot(100, b["predictions_t100"][name]["n2DEG_model"] / 1e13, mk, color=col, ms=4)
    ax.annotate("bare ends", (100, b["predictions_t100"]["bulk-terminated (H, improper)"]["n2DEG_model"] / 1e13),
                xytext=(70, 0.15), fontsize=6, color=C["bulk"], arrowprops=dict(arrowstyle="-", lw=0.4, color=C["bulk"]))
    ax.annotate("neutral or\nelectron counting",
                (100, b["predictions_t100"]["neutral lines"]["n2DEG_model"] / 1e13),
                xytext=(40, 0.55), fontsize=6, color=C["neutral"],
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C["neutral"]))
    ax.axvspan(65, 75, color="0.92", lw=0)
    ax.set_xlabel("doped AlGaN thickness (nm)")
    ax.set_ylabel(r"2DEG density ($10^{13}$ cm$^{-2}$)")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(15, 108)
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    # (b) line charge: density versus mobility
    ax = axs[1]
    FS.label(ax, "(b)", x=-0.2)
    rows = []
    for key, lab in (("77K_mu2_0", "77 K"), ("77K_mu2_83", "77 K, barrier conducts"),
                     ("300K_mu2_83", "300 K, as measured")):
        bb = sh["baseline"][key]
        sens = [v["Lambda_fit_e_per_nm"] for k, v in sh["sensitivity"].items()
                if k.startswith(key + "/") and v["Lambda_fit_e_per_nm"] is not None]
        lo = min([abs(x) for x in sens] + [abs(x) for x in bb["Lambda_fit_range_over_phi_grid"]])
        hi = max([abs(x) for x in sens] + [abs(x) for x in bb["Lambda_fit_range_over_phi_grid"]])
        strict = [v["Lambda_strict_upper_bound_e_per_nm"] for k, v in sh["sensitivity"].items()
                  if k.startswith(key + "/")] + [bb["mobility"]["Lambda_strict_upper_bound_e_per_nm"]]
        rows.append((lab, abs(bb["Lambda_fit_e_per_nm"]), lo, hi, max(strict)))
    a3 = c300["analysis_300K_channel_removed"]
    rows.insert(2, ("300 K, parallel channel removed", abs(a3["Lambda_fit_e_per_nm"]),
                    min(abs(x) for x in a3["Lambda_fit_range_over_phi_grid"]),
                    max(abs(x) for x in a3["Lambda_fit_range_over_phi_grid"]),
                    a3["mobility"]["Lambda_strict_upper_bound_e_per_nm"]))
    for i, (lab, v, lo, hi, smax) in enumerate(rows):
        y = len(rows) - i
        ax.plot([lo, hi], [y, y], color="black", lw=2.2, solid_capstyle="butt")
        ax.plot(v, y, "o", color="black", ms=3)
        ax.plot([0, smax], [y - 0.25, y - 0.25], color=C["fit"], lw=1.2)
        ax.text(0.17, y + 0.2, lab, fontsize=6, ha="left")
    u = unc["phi"]["0.5"]["fit_B"]
    lv, ls = abs(u["values"]["Lambda_e_per_nm"]), u["sigma"]["Lambda_e_per_nm"]
    y0 = len(rows) + 0.0
    ax.errorbar([lv], [y0 - 0.12], xerr=[[ls], [ls]], fmt="none", ecolor="0.45", elinewidth=0.7, capsize=1.5)
    bt = abs(b["predictions_t100"]["bulk-terminated (H, improper)"]["Lambda_e_per_nm"])
    ec = abs(b["predictions_t100"]["electron counting"]["Lambda_e_per_nm"])
    ax.axvline(bt, color=C["bulk"], lw=0.8, ls="--")
    ax.axvline(ec, color=C["ecr"], lw=0.8, ls="--")
    ax.text(bt - 0.07, 0.05, "bare\nends", color=C["bulk"], fontsize=6, ha="right")
    ax.text(ec + 0.07, 0.05, "electron\ncounting", color=C["ecr"], fontsize=6)
    ax.set_yticks([])
    ax.set_ylim(-0.25, len(rows) + 0.7)
    ax.set_xlim(-0.1, 3.6)
    ax.set_xlabel(r"$|\Lambda|$ (e nm$^{-1}$)")
    ax.plot([], [], color="black", lw=2.2, label="from the density")
    ax.plot([], [], color=C["fit"], lw=1.2, label="allowed by the mobility")
    ax.errorbar([], [], xerr=[], fmt="none", ecolor="0.45", elinewidth=0.7, capsize=1.5,
                label=r"joint fit, $\pm1\sigma$")
    ax.legend(frameon=False, loc="lower center", fontsize=5.8, bbox_to_anchor=(0.56, 0.99), ncol=3, columnspacing=0.8, handlelength=1.6)
    # (c) Hiroki et al.: relaxation each line charge needs
    ax = axs[2]
    FS.label(ax, "(c)", x=-0.2)
    sets = (("0.25", "90", "$x=0.25$\n90 nm"), ("0.35", "85", "$x=0.35$\n85 nm"), ("0.45", "30", "$x=0.45$\n30 nm"))
    best = [k for k in hk["as_measured"]["0.35"]["phi"]["0.5"]["relaxed"]["85"]["R_needed"]
            if k.startswith("Shen fraction best")][0]
    kinds = (("bulk-terminated", C["bulk"], "bare ends", -0.2),
             (best, C["fit"], "Shen fraction", 0.0),
             ("neutral", C["neutral"], "neutral", 0.2))
    ax.axhspan(0, 1, color="white")
    ax.fill_between([0.6, 1.4], 0, 0.20, color="0.88", lw=0)
    ax.text(0.63, 0.205, "map", fontsize=5.5, ha="left", va="bottom", color="0.35")
    for i, (xs, t, lab) in enumerate(sets):
        for name, col, kl, dx in kinds:
            v = [hk[tr][xs]["phi"][p]["relaxed"][t]["R_needed"][name] for tr in ("as_measured", "scaled_to_77K")
                 for p in hk[tr][xs]["phi"]]
            ok = [x for x in v if x is not None]
            xx = i + dx
            if ok:
                ax.plot([xx, xx], [min(ok), max(ok)], color=col, lw=3, solid_capstyle="butt")
            if len(ok) < len(v):
                ax.annotate("", (xx, 1.08), (xx, 0.95 if not ok else max(ok)),
                            arrowprops=dict(arrowstyle="->", lw=0.8, color=col))
        ax.text(i, -0.2, lab, fontsize=6, ha="center", va="top")
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
    ax.text(-0.55, 1.015, "full relaxation", fontsize=5.5, color="0.4", va="bottom", ha="left")
    for name, col, kl, dx in kinds:
        ax.plot([], [], color=col, lw=3, label=kl)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 0.93))
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([])
    ax.set_ylabel("relaxation needed, $r$")
    # (d) prediction: change of the GaN/AlN interface charge as the GaN relaxes
    ax = axs[3]
    FS.label(ax, "(d)", x=-0.27)
    pr = load("prediction.json")["cases"]["Al0.30Ga0.70N on GaN strained to AlN"]
    rr = np.linspace(0, 0.5, 2)
    kinds = (("bare ends", C["bulk"], "bare ends (H recipe)"),
             ("Shen fraction", C["fit"], "bound, as in Shen"),
             ("neutral", C["neutral"], "neutral (ZB recipe)"),
             ("trapped (Shen)", C["ecr"], "trapped, as in Shen"))
    for k, col, lab in kinds:
        sl = sorted(pr[c][k]["slope_cm2_per_unit_R"] / 1e13 for c in ("PBEsol", "HSE"))
        ax.fill_between(rr, sl[0] * rr, sl[1] * rr, color=col, alpha=0.35, lw=0)
        ax.plot(rr, sl[0] * rr, color=col, lw=0.7)
        ax.plot(rr, sl[1] * rr, color=col, lw=0.7)
        ax.text(0.505, 0.5 * (sl[0] + sl[1]) / 2, lab, color=col, fontsize=5.8, va="center", ha="left")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 2.6)
    ax.set_xlabel(r"relaxed fraction $r$ of the compressed AlGaN")
    ax.set_ylabel(r"change of charge ($10^{13}\,e\,\mathrm{cm}^{-2}$)")
    FS.save(fig, "fig3")


# --------------------------------------------------------------------- Fig. S1
def figS1():
    link = load("prb_link.json")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    fit = link["input"]
    key = "misfit + threading lines, mu_q_H = 95.0"
    r1 = fit["relaxed_fraction_f1"][key]
    a = fit["a_lattice_m"] * 1e9
    lam = np.linspace(0.05, 3.3, 300)
    ax.semilogy(lam, 100 * r1 / (lam * a) ** 2, color="black")
    ax.axhline(100, color="0.6", lw=0.6, ls=":")
    ax.text(1.4, 125, "full relaxation", fontsize=6, color="0.4")
    for name, col, lab in (("bulk-terminated (H, improper)", C["bulk"], "bare ends"),
                           ("Shen et al., largest strict mobility bound", C["fit"],
                            "largest charge allowed\nby the AlGaN mobility")):
        c = link["cases"][name]
        x, y = abs(c["Lambda_e_per_nm"]), 100 * c["fits"][key]["relaxed_fraction"]
        ax.plot(x, y, "o", color=col, ms=3.5)
        ax.annotate(lab, (x, y), xytext=(x + 0.25, y * 1.8), fontsize=6, color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=col))
    ax.set_xlabel(r"$|\Lambda|$ (e nm$^{-1}$)")
    ax.set_ylabel("relaxation needed (% of mismatch)")
    ax.set_xlim(0, 3.4)
    FS.save(fig, "figS1")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "S1"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "S1": figS1}[w]()
