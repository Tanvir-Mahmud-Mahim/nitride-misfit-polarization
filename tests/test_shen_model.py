"""Bookkeeping of the interface charge in scripts/run_shen.py.

The structure is built with the improper (H-branch) polarization of the partly
relaxed barrier plus an extra sheet (Lambda + P_H b) L_A. The total interface
charge must equal the decomposition of src/nmp/polarization.py,
sigma_coh + 2 R eps_c (e31_prop - e33 C13/C33) + Lambda L_A, for any Lambda.
Also checks the two-carrier Hall inversion and the calibration sheet.
"""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import run_shen as RS  # noqa: E402
from nmp import materials as M, polarization as PZ  # noqa: E402

FAKE = {"AlN": {"P_H": 1.33, "e31_proper": -0.66, "e33": 1.64, "C13_over_C33": 0.30},
        "GaN": {"P_H": 1.31, "e31_proper": -0.50, "e33": 0.80, "C13_over_C33": 0.25}}


def interface_sigma(R, Lam, dsig=0.0):
    gan, bar, eps_c = RS.setup(300.0, FAKE)
    st = RS.structure(25.0, 300.0, 1.0, gan, bar, eps_c, R=R, Lam=Lam, dz=0.1, dsig=dsig)
    k = int(round(RS.GAN_T / st.dz))
    return st.sheet[k], gan, bar, eps_c


def test_interface_charge_matches_decomposition():
    coh, *_ = interface_sigma(0.0, 0.0)
    for R in (0.3, 0.78):
        for Lam in (-4e-10, -1e-10, 0.0, 2e-11):
            s, gan, bar, eps_c = interface_sigma(R, Lam)
            k = bar["e31_prop"] - bar["e33"] * bar["C13_over_C33"]
            LA = PZ.misfit_line_density(R, eps_c, bar["a"])
            assert abs(s - (coh + 2 * R * eps_c * k + Lam * LA)) < 1e-12


def test_bulk_terminated_lines_equal_improper_prescription():
    R = 0.78
    gan, bar, eps_c = RS.setup(300.0, FAKE)
    s, *_ = interface_sigma(R, RS.limits(bar)[RS.LIMITS[0]])
    improper = gan["P_H"] - M.strained_P(bar, (1 - R) * eps_c)
    assert abs(s - improper) < 1e-12


def test_offset_adds_to_line_sheet():
    a, *_ = interface_sigma(0.5, -1e-10)
    b, *_ = interface_sigma(0.5, -1e-10, dsig=0.003)
    assert abs(b - a - 0.003) < 1e-14


def test_hall_split_roundtrip():
    n1, mu1, n2, mu2 = 1.1e13, 2900.0, 4e12, 83.0
    nH, muH = RS.hall(n1, mu1, n2, mu2)
    a, b = RS.split(nH, muH, n2, mu2)
    assert abs(a / n1 - 1) < 1e-12 and abs(b / mu1 - 1) < 1e-12
    assert RS.split(1e13, 1000.0, 0.0, 0.0) == (1e13, 1000.0)
