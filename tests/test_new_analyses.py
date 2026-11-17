"""Checks of the analyses added for the second series, the 300 K test and the
map reading."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_shen as RS  # noqa: E402


def test_two_channel_removal_is_exact_and_order_free():
    # three channels: 2DEG, buffer, barrier
    n = np.array([8e12, 3e12, 5e12])
    mu = np.array([900.0, 800.0, 83.0])
    s1, s2 = (n * mu).sum(), (n * mu ** 2).sum()
    nH, muH = s1 * s1 / s2, s2 / s1
    a = RS.split(*RS.split(nH, muH, n[1], mu[1]), n[2], mu[2])
    b = RS.split(*RS.split(nH, muH, n[2], mu[2]), n[1], mu[1])
    for x in (a, b):
        assert abs(x[0] / n[0] - 1) < 1e-12 and abs(x[1] / mu[0] - 1) < 1e-12


def test_hiroki_interface_matches_decomposition():
    import run_hiroki as RH
    from nmp import polarization as PZ
    fake = {"AlN": {"P_H": 1.33, "e31_proper": -0.66, "e33": 1.64, "C13_over_C33": 0.30},
            "GaN": {"P_H": 1.31, "e31_proper": -0.50, "e33": 0.80, "C13_over_C33": 0.25}}
    gan, bar, eps_c = RH.setup(0.35, fake)
    from nmp import materials as M

    def sig(R, Lam):
        P_bar = M.strained_P(bar, (1 - R) * eps_c)
        extra = 0.0
        if R > 0:
            LA = PZ.misfit_line_density(R, eps_c, bar["a"])
            extra = (Lam + bar["P_H"] * bar["a"] * 1e-10) * LA
        return gan["P_H"] - P_bar + extra
    coh = sig(0.0, 0.0)
    k = bar["e31_prop"] - bar["e33"] * bar["C13_over_C33"]
    for R in (0.1, 0.5):
        for Lam in (-3e-10, 0.0):
            LA = PZ.misfit_line_density(R, eps_c, bar["a"])
            assert abs(sig(R, Lam) - (coh + 2 * R * eps_c * k + Lam * LA)) < 1e-12


def test_map_relaxation_scale():
    d = json.load(open(os.path.join(ROOT, "data", "hiroki2002_digitized.json")))["fig4b_map_85nm_x0.35"]
    q0 = d["Qx_GaN_centroid"]
    full = q0 * (3.189 / (0.35 * 3.112 + 0.65 * 3.189) - 1)
    assert abs(full - d["R_scale_full_relaxation_dQx"]) < 2e-3
    lo, hi = d["Qx_AlGaN_contour_range"]
    assert abs((hi - q0) / full - d["R_of_contour_pixels"]["max"]) < 0.01
    assert abs((lo - q0) / full - (d["R_of_contour_pixels"]["min"])) < 0.01


def test_prediction_slopes_follow_eq2():
    """Slopes of results/prediction.json equal 2 eps (e31 - e33 C13/C33) + 2|eps| Lambda / b,
    and the bound and trapped readings coincide for a film under tension."""
    from nmp import materials as M
    P = json.load(open(os.path.join(ROOT, "results", "prediction.json")))
    d = M.dft()
    F, S = M.algan(0.0, 300.0, d), M.algan(1.0, 300.0, d)        # GaN film on AlN
    eps = (S["a"] - F["a"]) / F["a"]
    k = F["e31_prop"] - F["e33"] * F["C13_over_C33"]
    g = P["cases"]["GaN on AlN"]["PBEsol"]
    to_cm2 = 1.0 / 1.602176634e-19 / 1e4
    assert abs(g["neutral"]["slope_cm2_per_unit_R"] - 2 * eps * k * to_cm2) < 1e-3 * abs(g["neutral"]["slope_cm2_per_unit_R"])
    lam = g["Shen fraction"]["Lambda_e_per_nm"] * 1.602176634e-19 * 1e9       # C/m
    extra = 2 * abs(eps) * lam / (F["a"] * 1e-10) * to_cm2
    assert abs(g["Shen fraction"]["slope_cm2_per_unit_R"] - g["neutral"]["slope_cm2_per_unit_R"] - extra) < 1e-3 * abs(extra)
    assert g["trapped (Shen)"]["slope_cm2_per_unit_R"] < g["neutral"]["slope_cm2_per_unit_R"] < g["Shen fraction"]["slope_cm2_per_unit_R"]
    for name in ("AlN on GaN", "Al0.30Ga0.70N on GaN", "Al0.22Ga0.78N on GaN"):
        for c in ("PBEsol", "HSE"):
            r = P["cases"][name][c]
            assert r["eps_c"] > 0
            assert r["Shen fraction"]["slope_cm2_per_unit_R"] == r["trapped (Shen)"]["slope_cm2_per_unit_R"]


def test_bare_end_relaxed_algan_matches_ye2025():
    """Ye et al., Nat. Commun. 16, 3863 (2025): fully relaxed Al0.3Ga0.7N/GaN with the
    H reference and HSE constants gives Delta P = +0.012 C/m^2 (interface charge -0.012)."""
    P = json.load(open(os.path.join(ROOT, "results", "prediction.json")))
    s = P["cases"]["Al0.30Ga0.70N on GaN"]["HSE"]["relaxed_C_m2"]["bare ends"]
    assert abs(s - (-0.012)) < 0.0006


def test_elwaradi_stack_charge_matches_eq2_on_relaxed_channel():
    """With a fully relaxed GaN channel the charge of the GaN/spacer and
    spacer/barrier interfaces together equals Eq. (2) for the barrier on GaN."""
    import run_elwaradi as RE
    from nmp import materials as M, polarization as PZ
    d = M.dft()
    mats = RE.setup(d)
    gan, aln, bar = mats
    for Rb, lam in ((0.0, 0.0), (0.3, 0.0), (0.3, -2e-10), (0.6, 1e-10)):
        st, info = RE.stack(50.0, 1.0, Rb, "neutral", lam, 1.0, 0.0, mats)
        k1 = int(round(st.edges[2] / st.dz))
        k2 = int(round(st.edges[3] / st.dz))
        total = st.sheet[k1] + st.sheet[k2]
        eps_c = (gan["a"] - bar["a"]) / bar["a"]
        ref = PZ.sigma_decomposed(bar, gan, Rb, lam, eps_c)
        assert abs(total - ref) < 1e-4 * abs(ref) + 1e-6
