"""Figures of the Letter (figures/fig1-3) and of the Supplemental Material
(figures/figS1-S2). Fig. 1 also reads paper/numbers.tex, so run
scripts/make_numbers.py first."""

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
def numbers():
    """The macros of paper/numbers.tex (written by make_numbers.py), so that the
    overview figure quotes exactly the numbers of the text."""
    import re
    txt = open(os.path.join(FS.ROOT, "paper", "numbers.tex")).read()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt))


def _block(ax, x0, y0, w, h, dx, dy, fc, ec="0.25", lw=0.5, z=1, top=True):
    """An oblique-projection box: front face, right side face and top face."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(fc)
    side = (0.80 * r, 0.80 * g, 0.80 * b)
    lid = (min(1, 0.55 + 0.45 * r), min(1, 0.55 + 0.45 * g), min(1, 0.55 + 0.45 * b))
    ax.add_patch(plt.Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)],
                             fc=fc, ec=ec, lw=lw, zorder=z))
    ax.add_patch(plt.Polygon([(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                             fc=side, ec=ec, lw=lw, zorder=z))
    if top:
        ax.add_patch(plt.Polygon([(x0, y0 + h), (x0 + w, y0 + h), (x0 + w + dx, y0 + h + dy), (x0 + dx, y0 + h + dy)],
                                 fc=lid, ec=ec, lw=lw, zorder=z))


def _clip(p, d, W, D):
    """Segment of the line p + t d inside the rectangle [0, W] x [0, D] (Liang-Barsky)."""
    t0, t1 = -1e9, 1e9
    for pi, di, lo, hi in ((p[0], d[0], 0.0, W), (p[1], d[1], 0.0, D)):
        if abs(di) < 1e-12:
            if pi < lo or pi > hi:
                return None
            continue
        a, b = (lo - pi) / di, (hi - pi) / di
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    if t0 >= t1:
        return None
    return (p[0] + t0 * d[0], p[1] + t0 * d[1]), (p[0] + t1 * d[0], p[1] + t1 * d[1])


def _box(ax, x, y, w, h, col, title, body, note=None, reg=None):
    """A rounded box with a coloured title, body text and a grey note."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc="white", ec=col, lw=1.0, zorder=2))
    cx = x + w / 2
    ts = [ax.text(cx, y + h - 0.45, title, ha="center", va="top", fontsize=7.3, color=col, zorder=3),
          ax.text(cx, y + h - 1.85, body, ha="center", va="top", fontsize=6.8, linespacing=1.3, zorder=3)]
    if note:
        ts.append(ax.text(cx, y + 0.45, note, ha="center", va="bottom", fontsize=6.5,
                          color="0.35", linespacing=1.25, zorder=3))
    if reg is not None:
        reg.append(((x, y, w, h), ax, ts))


def _check_layout(fig, boxes, groups):
    """Fail if any text leaves its box or if two texts overlap."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    bad = []
    for (x, y, w, h), ax, ts in boxes:
        (bx0, by0), (bx1, by1) = ax.transData.transform([(x, y), (x + w, y + h)])
        ext = [t.get_window_extent(rnd) for t in ts]
        for t, e in zip(ts, ext):
            if e.x0 < bx0 + 2 or e.x1 > bx1 - 2 or e.y0 < by0 + 1 or e.y1 > by1 - 1:
                bad.append("outside box: " + t.get_text()[:40])
        for a in range(len(ext)):
            for b in range(a + 1, len(ext)):
                if ext[a].overlaps(ext[b]):
                    bad.append("overlap in box: " + ts[a].get_text()[:30] + " / " + ts[b].get_text()[:30])
    for ts in groups:
        ext = [(t, t.get_window_extent(rnd)) for t in ts if t.get_text().strip()]
        for a in range(len(ext)):
            for b in range(a + 1, len(ext)):
                if ext[a][1].overlaps(ext[b][1]):
                    bad.append("overlap: " + ext[a][0].get_text()[:30] + " / " + ext[b][0].get_text()[:30])
    W = fig.bbox.width
    for ax in fig.axes:
        for t in ax.texts:
            e = t.get_window_extent(rnd)
            if e.x0 < 0 or e.x1 > W:
                bad.append("off the figure: " + t.get_text()[:40])
    if bad:
        raise SystemExit("Fig. 1 layout problems:\n  " + "\n  ".join(bad))


def fig1():
    """Overview figure.
    (a) The relaxed sample of Shen et al. Layer sequence, thicknesses, growth
    temperature of the nucleation layer, substrate and the electron density of
    the doped layer are those stated in the text of Ref. Shen2000; R and the
    77 K electron density are from data/shen2000_fig2_digitized.json; the strain
    values and the dislocation spacing come from the model (results files).
    The box shows one misfit dislocation and its charge per length.
    (b) Evidence, findings and consequences, with the numbers of paper/numbers.tex."""
    N = numbers()
    sh = load("shen_analysis.json")["baseline"]["77K_mu2_0"]
    shen = json.load(open(os.path.join(FS.ROOT, "data", "shen2000_fig2_digitized.json")))
    R = sh["R"]
    eps_c = sh["eps_c"]
    lam = {k: v["Lambda_e_per_nm"] for k, v in sh["predictions_t100"].items()}
    ns100 = shen["ns_77K_1e13cm2"][shen["thickness_nm"].index(100)]
    # cross-checks between the figure inputs and the text
    assert abs(R - shen["text_values"]["relaxation_R_total_103nm"]) < 1e-12
    assert shen["text_values"]["n_AlGaN_carrier_concentration_300K_cm3"] == 1.26e18
    assert "%.2f" % abs(lam["bulk-terminated (H, improper)"]) == N["LamBulkAlGaN"]
    assert "%.2f" % abs(lam["electron counting"]) == N["LamECRAlGaN"]
    assert "%.2f" % ns100 == N["MeasSeventySeven"]
    assert "%d" % (10 * round(sh["line_spacing_nm_hexagonal"] / 10)) == N["LineSpacing"]

    W_IN, H_IN = 7.0, 3.8
    fig = plt.figure(figsize=(W_IN, H_IN))
    fa = 0.48
    ax = fig.add_axes([0.0, 0.0, fa, 1.0])     # coordinates in units of 0.1 inch
    WA, HA = 10 * W_IN * fa, 10 * H_IN
    ax.set_xlim(0, WA)
    ax.set_ylim(0, HA)
    ax.axis("off")
    ta = [ax.text(0.0, HA, "(a)", fontsize=9, fontweight="bold", va="top")]
    fs = 6.8

    # ------------------------------------------------------------ the stack
    x0, w, dx, dy = 12.3, 4.4, 2.0, 1.6
    gan_c, algan_c, sap_c, nuc_c = "#E8C27A", "#AFC3D6", "#C9CCD3", "#C9A15A"
    _block(ax, x0, 1.0, w, 3.4, dx, dy, sap_c, top=False)
    zx = np.linspace(x0, x0 + w + dx, 15)
    zy = 2.2 + np.where(np.arange(15) % 2 == 0, 0.0, 0.4) + np.clip(zx - (x0 + w), 0, None) * dy / dx
    ax.fill_between(zx, zy - 0.3, zy + 0.1, color="white", zorder=2, lw=0)
    ax.plot(zx, zy + 0.1, color="0.25", lw=0.5, zorder=3)
    ax.plot(zx, zy - 0.3, color="0.25", lw=0.5, zorder=3)
    _block(ax, x0, 4.4, w, 0.7, dx, dy, nuc_c, top=False, z=4)
    yg0, yg1 = 5.1, 13.0
    _block(ax, x0, yg0, w, yg1 - yg0, dx, dy, gan_c, z=4)
    t2 = 0.35   # drawn thickness of the electron sheet
    ax.add_patch(plt.Polygon([(x0, yg1 - t2), (x0 + w, yg1 - t2), (x0 + w, yg1), (x0, yg1)],
                             fc=C["bulk"], ec="none", zorder=5))
    ax.add_patch(plt.Polygon([(x0 + w, yg1 - t2), (x0 + w + dx, yg1 - t2 + dy), (x0 + w + dx, yg1 + dy),
                              (x0 + w, yg1)], fc="#B54B00", ec="none", zorder=5))
    # misfit network on the interface: three families of lines at 60 degrees
    to_xy = lambda X, Z: (x0 + X * w + Z * dx, yg1 + Z * dy)  # noqa: E731
    for ang, off in ((0.0, 0.17), (60.0, 0.05), (120.0, 0.11)):
        dvec = (np.cos(np.radians(ang)), np.sin(np.radians(ang)))
        nvec = (-dvec[1], dvec[0])
        for k in range(-6, 7):
            p = (0.5 + (k * 0.30 + off) * nvec[0], 0.5 + (k * 0.30 + off) * nvec[1])
            seg = _clip(p, dvec, 1.0, 1.0)
            if seg is None or np.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) < 0.08:
                continue
            (xa, ya), (xb, yb) = to_xy(*seg[0]), to_xy(*seg[1])
            ax.plot([xa, xb], [ya, yb], color="0.15", lw=0.7, zorder=6, solid_capstyle="butt")
    ya0 = yg1 + dy + 3.2          # the barrier is drawn lifted above the interface
    for cx_, cy_ in ((x0, yg1), (x0 + w, yg1), (x0 + w + dx, yg1 + dy)):
        ax.plot([cx_, cx_], [cy_, ya0 + (cy_ - yg1)], color="0.45", lw=0.45, ls=(0, (2, 2)), zorder=3)
    hs, hd = 0.5, 6.3
    _block(ax, x0, ya0, w, hs, dx, dy, "#C9D6E3", top=False, z=7)
    _block(ax, x0, ya0 + hs, w, hd, dx, dy, algan_c, z=7)
    ytop = ya0 + hs + hd

    lx = x0 + w + dx + 0.7
    xs_ = x0 + w + dx / 2
    lab_t = []

    def lab(xa, ya, yt, text, col="black"):
        ax.plot([xa, lx - 0.2], [ya, yt], color="0.4", lw=0.4, zorder=8)
        lab_t.append(ax.text(lx, yt, text, fontsize=fs, va="center", ha="left", color=col, linespacing=1.25))

    lab(x0 + w / 2 + dx / 2, ytop + dy / 2, ytop + 2.6,
        "whole barrier 103 nm, relaxed by\n"
        f"{100 * R:.0f} % (X-ray); model strain {100 * (1 - R) * eps_c:.2f} %\n"
        f"({100 * eps_c:.2f} % before relaxing)")
    lab(xs_, ya0 + hs + hd / 2 + dy / 2, ya0 + hs + hd / 2 - 0.4,
        "Si-doped Al$_{0.22}$Ga$_{0.78}$N, 100 nm,\n"
        "$1.26\\times10^{18}$ cm$^{-3}$ electrons (300 K)")
    lab(xs_, ya0 + hs / 2 + dy / 2, ya0 - 0.05, "unintentionally doped\nAl$_{0.22}$Ga$_{0.78}$N spacer, 3 nm")
    xm, ym = to_xy(0.85, 0.6)
    lab(xm, ym, yg1 + dy + 0.15, f"misfit dislocations, about\n{N['LineSpacing']} nm apart (estimate)")
    lab(xs_, yg1 - t2 / 2 + dy / 2, yg1 - 1.3,
        f"electron sheet (2DEG), measured\n${ns100:.2f}\\times10^{{13}}$ cm$^{{-2}}$ at 77 K", col=C["bulk"])
    lab(xs_, (yg0 + yg1) / 2 - 0.8 + dy / 2, (yg0 + yg1) / 2 - 1.6, "unintentionally doped\nGaN, 2.0 µm")
    lab(xs_, 4.75 + dy / 2, 4.5, "GaN nucleation layer,\ngrown at 488 °C")
    lab(xs_, 1.7 + dy / 2, 1.6, "sapphire (0001)")
    xc = (x0 + WA) / 2
    ta.append(ax.text(xc, HA, "relaxed sample of Shen $et\\ al.$", ha="center", va="top", fontsize=7.3))
    ta.append(ax.text(xc, HA - 1.4, "vertical scale broken; barrier lifted\nto show the interface",
                      ha="center", va="top", fontsize=6.5, color="0.4", linespacing=1.2))

    # ------------------------------------------------------------ the box: one dislocation
    bx0, bx1, by0, by1 = 0.3, 11.2, 0.4, HA - 1.4
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fc="white", ec="0.3", lw=0.6))
    xm_ = (bx0 + bx1) / 2
    tb = [ax.text(xm_, by1 - 0.5, "one misfit dislocation", ha="center", va="top", fontsize=7.3),
          ax.text(xm_, by1 - 2.2, "Al$_{0.22}$Ga$_{0.78}$N film, stretched", ha="center", va="center", fontsize=6.5,
                  color=C["ecr"])]
    ys_, yi_, yt_ = 24.6, 28.4, 32.4
    ax.add_patch(Rectangle((1.2, ys_), 9.1, yi_ - ys_, fc="#F3E3C2", ec="none"))
    ax.add_patch(Rectangle((1.2, yi_), 9.1, yt_ - yi_, fc="#DCE6EF", ec="none"))
    for xx in np.linspace(1.6, 9.9, 8):
        ax.plot([xx, xx], [ys_, yi_], color="0.4", lw=0.6)
    xf = np.linspace(1.6, 9.9, 9)
    for k, xx in enumerate(xf):
        ax.plot([xx, xx], [yi_, yt_], color=C["bulk"] if k == 4 else C["ecr"], lw=1.4 if k == 4 else 0.6)
    ax.plot(xf[4], yi_, "o", ms=4.2, mfc="white", mec=C["bulk"], mew=1.0, zorder=5)
    tb.append(ax.text(xf[4] + 0.3, yi_ - 0.35, r"$\Lambda$", fontsize=8, color=C["bulk"], va="top",
                      bbox=dict(fc="#F3E3C2", ec="none", pad=0.3)))
    tb.append(ax.text(xm_, ys_ - 0.8, "GaN", ha="center", va="center", fontsize=6.5, color="0.3"))
    tb.append(ax.text(xm_, ys_ - 1.9, "one extra plane of atoms\n(red) ends at the interface;\n"
                      "its end carries a charge\n$\\Lambda$ per unit length",
                      ha="center", va="top", fontsize=6.5, linespacing=1.25))
    fx0, fx1 = (bx0 + 0.9) / (10 * W_IN), (bx1 - 0.5) / (10 * W_IN)
    axf = fig.add_axes([fx0, 3.3 / HA, fx1 - fx0, 13.4 / HA])
    rows = [("bare end (H recipe)", lam["bulk-terminated (H, improper)"], None, C["bulk"], "v"),
            ("measured (Shen density)", float(N["LamLP"]), float(N["LamLPsig"]), "black", "o"),
            ("allowed by the mobility", None, float(N["LamMobStrictBase"]), C["fit"], None),
            ("electron counting", lam["electron counting"], None, C["ecr"], "^"),
            ("neutral (ZB recipe)", lam["neutral lines"], None, C["neutral"], "s")]
    xl = (-3.2, 1.6)
    tf = []
    for k, (name, v, e, col, mk) in enumerate(rows):
        yy = -k
        axf.axhline(yy, color="0.85", lw=0.5, zorder=0)     # row guide; rows are cases, not values
        if mk is None:
            axf.plot([-e, min(e, xl[1])], [yy, yy], color=col, lw=3.0, alpha=0.45, solid_capstyle="butt")
        elif e is not None:
            axf.errorbar(v, yy, xerr=e, fmt=mk, color=col, ms=3.4, mfc="white", capsize=1.5, elinewidth=0.7)
        else:
            axf.plot(v, yy, mk, color=col, ms=3.6)
        tf.append(axf.text(xl[0] + 0.12, yy + 0.28, name, fontsize=6.5, color=col, va="bottom", ha="left"))
    axf.set_xlim(*xl)
    axf.set_ylim(-len(rows) + 0.5, 1.0)
    axf.set_yticks([])
    for side in ("left", "right", "top"):
        axf.spines[side].set_visible(False)
    axf.set_xticks([-3, -2, -1, 0, 1])
    axf.tick_params(labelsize=6.5, top=False, right=False, length=2)
    axf.set_xlabel(r"$\Lambda$ (e nm$^{-1}$)", fontsize=6.8, labelpad=1)

    # ------------------------------------------------------------ panel (b)
    axb = fig.add_axes([fa, 0.0, 1.0 - fa, 1.0])
    Wb = 10 * W_IN * (1.0 - fa)
    axb.set_xlim(0, Wb)
    axb.set_ylim(0, HA)
    axb.axis("off")
    tbb = [axb.text(0.2, HA, "(b)", fontsize=9, fontweight="bold", va="top")]
    gap, m0, mr = 0.7, 0.9, 0.3
    bw = (Wb - m0 - mr - 2 * gap) / 3
    xs = [m0 + k * (bw + gap) for k in range(3)]
    h1, h2, h3 = 8.9, 9.7, 10.9
    y3 = 2.3
    y2 = y3 + h3 + 2.5
    y1 = y2 + h2 + 2.5
    hdr = dict(fontsize=7.0, style="italic", color="0.3", va="center", ha="left")
    tbb.append(axb.text(2.0, y1 + h1 + 0.75, "evidence", **hdr))
    tbb.append(axb.text(2.0, y2 + h2 + 1.25, "findings", **hdr))
    tbb.append(axb.text(2.0, y3 + h3 + 1.25, "what follows", **hdr))
    dark, cons = "0.15", "0.35"
    boxes = []
    _box(axb, xs[0], y1, bw, h1, C["ecr"], "computed faces (DFT)",
         "broken bonds filled by\n"
         "electron counting leave\n"
         f"${N['SlabSigAlN']}$ and ${N['SlabSigGaN']}$ C m$^{{-2}}$",
         "(AlN, GaN); a bare face\ncarries more than 1.3 C m$^{-2}$", reg=boxes)
    _box(axb, xs[1], y1, bw, h1, dark, "Shen samples, 77 K",
         f"103 nm barrier relaxed\nby {100 * R:.0f} %; electron density\n"
         f"${N['MeasSeventySeven']}\\times10^{{13}}$ cm$^{{-2}}$ (measured)",
         f"predicted {N['PredBulkSeventySeven']} for bare ends\n"
         f"and {N['PredNeutralSeventySeven']} for neutral lines", reg=boxes)
    _box(axb, xs[2], y1, bw, h1, C["fit"], "mobility, two series",
         f"$|\\Lambda|\\leq{N['LamMobStrictBase']}$ e nm$^{{-1}}$ (Shen),\n"
         f"$|\\Lambda|\\leq{N['ElwMobStrictLo']}$ to ${N['ElwMobStrictHi']}$ e nm$^{{-1}}$\n"
         "(Elwaradi, compressed)",
         f"Shen at 77 K; bare ends\nwould carry {N['LamBulkAlGaN']} and {N['ElwLamBare']}", reg=boxes)
    _box(axb, xs[0], y2, bw, h2, C["ecr"], "nearly neutral ends",
         "electron counting leaves\n"
         f"$|\\Lambda|={N['LamECRAlGaN']}$ e nm$^{{-1}}$, against\n"
         f"{N['LamBulkAlGaN']} e nm$^{{-1}}$ for a bare end",
         "ideal faces, not real\ndislocations", reg=boxes)
    _box(axb, xs[1], y2, bw, h2, dark, "partly filled",
         f"$\\Lambda={N['LamLP']}\\pm{N['LamLPsig']}$ e nm$^{{-1}}$,\n"
         f"about a third (${N['FracLP']}\\pm{N['FracLPsig']}$)\n"
         "of the bare-end charge",
         f"neutral lines {N['NeutralSig']} standard\n"
         "deviations away; Hiroki loss\nexceeds neutral lines alone", reg=boxes)
    _box(axb, xs[2], y2, bw, h2, C["bulk"], "bare ends ruled out",
         "the H recipe fails in\nstretched (Shen) and\ncompressed (Elwaradi)",
         f"they would scatter at least\n{N['ElwScat']} times too strongly (both);\n"
         "the Shen density agrees", reg=boxes)
    arrows = ((0, 0, 0.62, 0.62), (1, 1, 0.62, 0.62), (2, 2, 0.62, 0.62))
    for a_, b_, fa_, fb_ in arrows:
        axb.add_patch(FancyArrowPatch((xs[a_] + fa_ * bw, y1 - 0.1), (xs[b_] + fb_ * bw, y2 + h2 + 0.1),
                                      arrowstyle="-|>", mutation_scale=6, lw=0.6, color="0.45", zorder=1))
    bw3 = (Wb - m0 - mr - gap) / 2
    _box(axb, m0, y3, bw3, h3, cons, "holes need charged dislocations",
         "Al$_{0.3}$Ga$_{0.7}$N fully relaxed on GaN\n"
         "keeps a positive charge with the\n"
         f"measured $\\Lambda$, ${N['AlThirtyShenLo']}$ to ${N['AlThirtyShenHi']}\\times10^{{13}}$ cm$^{{-2}}$",
         f"a negative charge needs more than {N['FzeroHSE']}\n"
         "of the bare-end charge (HSE constants),\n"
         f"against the measured ${N['FracLPHSE']}\\pm{N['FracLPHSEsig']}$", reg=boxes)
    _box(axb, m0 + bw3 + gap, y3, bw3, h3, cons, "new test, bound or trapped?",
         "stretch one layer and compress one of\n"
         "the same alloy; $y_+ + y_- = -2q/b$ is 0 for\n"
         f"bound charge, $-{N['SumTrapLo']}$ to $-{N['SumTrapHi']}$ C m$^{{-2}}$ if the\n"
         "Shen charge is trapped (Eq. 3)",
         "assumes both kinds of end are filled alike;\n"
         "the holography error alone ($\\pm10^{-3}$ C m$^{-2}$)\n"
         f"gives $q$ to $\\pm{N['DqHalf']}$ e nm$^{{-1}}$ at $r=0.5$", reg=boxes)
    tbb.append(axb.text(Wb / 2 + 0.3, 0.55,
                        "to first order in the strain, the H and ZB recipes differ only in the dislocation charge $\\Lambda$",
                        ha="center", va="bottom", fontsize=6.8))
    _check_layout(fig, boxes, [ta + lab_t, tb, tf, tbb])
    FS.save(fig, "fig1")


def figS1():
    """Former Fig. 1(b): interface charge of AlN on relaxed GaN versus relaxation."""
    d = M.dft()
    fig, ax2 = plt.subplots(figsize=(3.4, 2.5))
    aln = M.algan(1.0, 300.0, d)
    gan = M.algan(0.0, 300.0, d)
    eps_c = (gan["a"] - aln["a"]) / aln["a"]
    lim = PZ.line_charge_limits(aln, eps_c)
    r = np.linspace(0, 1, 101)
    for name, col, lab in (("neutral", C["neutral"], r"neutral lines ($\Lambda=0$; ZB recipe)"),
                           ("electron counting", C["ecr"], "electron-counting lines"),
                           ("bulk-terminated", C["bulk"], "bare ends (H recipe)")):
        s = [PZ.to_cm2(PZ.sigma_decomposed(aln, gan, x, lim[name])) / 1e13 for x in r]
        ax2.plot(r, s, color=col, label=lab, ls="--" if name == "electron counting" else "-")
    ax2.set_xlabel(r"relaxed fraction $r$ of the AlN film")
    ax2.set_ylabel(r"interface charge ($10^{13}\,e\,\mathrm{cm}^{-2}$)")
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=False, loc="lower left", handlelength=1.8)
    FS.save(fig, "figS1")


# --------------------------------------------------------------------- Fig. 2
def fig2():
    pol = load("dft_polarization.json")
    slab = load("dft_slab.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"wspace": 0.42})
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
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.25), gridspec_kw={"wspace": 0.42, "hspace": 0.58})
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
    ax.set_ylabel(r"electron density ($10^{13}$ cm$^{-2}$)")
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
def figS2():
    """Relaxation needed by the GaN/AlN hole gas versus the dislocation charge."""
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
    FS.save(fig, "figS2")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "S1", "S2"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "S1": figS1, "S2": figS2}[w]()
