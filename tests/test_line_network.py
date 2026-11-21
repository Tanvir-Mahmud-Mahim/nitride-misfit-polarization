"""Checks of scripts/run_line_network_check.py (straight misfit networks)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import run_line_network_check as NC  # noqa: E402
from nmp import lines, materials as M  # noqa: E402


def _setup():
    gan = M.algan(0.0, 77.0, M.dft())
    return gan["me"], gan["eps"], NC.QE * 1e9, 1e7, 0.645e17


def test_many_directions_reproduce_random_kernel():
    """With many line directions the exact Boltzmann solution must approach the
    relaxation-time kernel for randomly oriented lines."""
    m, eps, lam, LA, ns = _setup()
    t = NC.tau_network(ns, lam, LA, m, eps, normals_deg=tuple(np.arange(0, 180, 7.5)), N=1200)
    t_ker = 1.0 / lines.rate_lines(ns, lam, LA, m, eps)
    assert abs(t / t_ker - 1) < 0.06


def test_three_directions_scatter_less():
    m, eps, lam, LA, ns = _setup()
    t = NC.tau_network(ns, lam, LA, m, eps, N=1200)
    t_ker = 1.0 / lines.rate_lines(ns, lam, LA, m, eps)
    assert 4.0 < t / t_ker < 12.0


def test_rate_scales_with_line_charge_squared():
    m, eps, lam, LA, ns = _setup()
    t1 = NC.tau_network(ns, lam, LA, m, eps, N=600)
    t2 = NC.tau_network(ns, 2 * lam, LA, m, eps, N=600)
    assert abs(t1 / t2 - 4.0) < 1e-6
