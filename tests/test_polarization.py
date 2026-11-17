"""Checks of src/nmp/polarization.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nmp import polarization as PZ  # noqa: E402

ALN = {"a": 3.112, "P_H": 1.351, "e31_prop": -0.676, "e33": 1.569, "C13_over_C33": 0.29}
GAN = {"a": 3.189, "P_H": 1.312, "e31_prop": -0.551, "e33": 1.020, "C13_over_C33": 0.27}


def test_zb_formal_is_three_quarters_of_quantum():
    for a in (3.0, 3.112, 3.189, 3.54):
        assert abs(PZ.p_zb_formal(a) / PZ.quantum_c(a) - 0.75) < 1e-12


def test_quantum_value():
    # e c / Omega with Omega = sqrt(3)/2 a^2 c
    a = 3.112e-10
    assert abs(PZ.quantum_c(3.112) - PZ.E / (np.sqrt(3) / 2 * a * a)) < 1e-9


def test_prescriptions_agree_when_coherent_up_to_known_term():
    """Coherent AlN on GaN: H-improper minus ZB-proper equals the Dreyer et al.
    correction Delta P_corr = e sqrt3/2 (1/a_sub^2 - 1/a_film^2) plus the
    improper term -2 eps_c P_H,F (exact in this linear model)."""
    eps_c = (GAN["a"] - ALN["a"]) / ALN["a"]
    d = PZ.sigma_interface(ALN, GAN, 0.0, "H_improper") - PZ.sigma_interface(ALN, GAN, 0.0, "ZB_proper")
    corr = PZ.p_zb_formal(GAN["a"]) - PZ.p_zb_formal(ALN["a"])
    assert abs(d - (corr + 2 * eps_c * ALN["P_H"])) < 1e-12


def test_wang_estimate():
    """Wang et al., APL 124, 263502 (2024): 2 x 0.5 x 0.022 x 1.35 = 0.0297 C/m^2."""
    assert abs(2 * 0.5 * 0.022 * 1.35 - 0.0297) < 1e-12


def test_decomposition_reproduces_prescriptions():
    eps_c = (GAN["a"] - ALN["a"]) / ALN["a"]
    lim = PZ.line_charge_limits(ALN, eps_c)
    for r in (0.0, 0.2, 0.5, 1.0):
        h = PZ.sigma_interface(ALN, GAN, r, "H_improper")
        assert abs(PZ.sigma_decomposed(ALN, GAN, r, lim["bulk-terminated"]) - h) < 1e-12
    # neutral lines reproduce the r-dependence of ZB-proper
    zb = [PZ.sigma_interface(ALN, GAN, r, "ZB_proper") for r in (0.0, 0.5, 1.0)]
    de = [PZ.sigma_decomposed(ALN, GAN, r, 0.0) for r in (0.0, 0.5, 1.0)]
    assert abs((zb[2] - zb[0]) - (de[2] - de[0])) < 1e-12
    assert abs((zb[1] - zb[0]) - (de[1] - de[0])) < 1e-12


def test_difference_of_prescriptions_is_the_line_charge():
    """H-improper minus ZB-proper changes with r by -2 r eps_c P_H,F (Wang et al.)."""
    eps_c = (GAN["a"] - ALN["a"]) / ALN["a"]
    f = lambda r: PZ.sigma_interface(ALN, GAN, r, "H_improper") - PZ.sigma_interface(ALN, GAN, r, "ZB_proper")
    assert abs((f(0.5) - f(0.0)) - (-2 * 0.5 * eps_c * ALN["P_H"])) < 1e-12


def test_line_density_and_units():
    LA = PZ.misfit_line_density(0.5, 0.02, 3.112)
    assert abs(LA * 3.112e-10 - 0.02) < 1e-15
    assert abs(PZ.to_cm2(PZ.E * 1e4) - 1.0) < 1e-12


def test_line_charge_signs():
    lim_t = PZ.line_charge_limits(ALN, +0.02)   # tensile film: extra columns
    lim_c = PZ.line_charge_limits(GAN, -0.02)   # compressive film: missing columns
    assert lim_t["bulk-terminated"] < 0 < lim_c["bulk-terminated"]
    assert abs(lim_t["electron counting"]) < 0.1 * abs(lim_t["bulk-terminated"])
