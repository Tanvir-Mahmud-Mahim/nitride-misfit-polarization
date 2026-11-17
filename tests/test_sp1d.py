"""Checks of the Schroedinger-Poisson solver src/nmp/sp1d.py."""

import os
import sys

import numpy as np
from scipy.special import ai_zeros

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nmp import sp1d  # noqa: E402


def flat(t=20.0, m=0.2):
    return sp1d.Structure([sp1d.Layer("X", t, 10.0, 0.0, -3.0, m, 1.0, 0.0)], phi_B=0.0, dz=0.02)


def test_infinite_well_levels():
    """Hard-wall well: the grid ends act as walls one step outside the end points."""
    s = flat()
    m = 0.2
    U = np.zeros(len(s.z))
    _, w, _, _ = s._subbands(U, np.full(len(s.z), m))
    L = (s.z[-1] - s.z[0] + 2 * s.dz) * 1e-9
    for n in (1, 2, 3):
        exact = (n * np.pi * sp1d.HBAR / L) ** 2 / (2 * m * sp1d.M0) / sp1d.QE
        assert abs(w[n - 1] - exact) / exact < 2e-3


def test_triangular_well_matches_airy():
    """Uniform field F with a hard wall at the bottom: E_n = (hbar^2 e^2 F^2/2m)^(1/3) |a_n|."""
    s = flat(t=40.0)
    m, F = 0.2, 2e7            # V/m
    z = (s.z - s.z[0]) * 1e-9
    U = F * z                  # eV, since energy = e F z
    _, w, _, _ = s._subbands(U, np.full(len(s.z), m))
    a = -ai_zeros(3)[0]
    scale = (sp1d.HBAR ** 2 * (sp1d.QE * F) ** 2 / (2 * m * sp1d.M0)) ** (1 / 3) / sp1d.QE
    for n in range(2):
        # the wall sits one grid step below z = 0: shift by F dz
        exact = scale * a[n] - F * s.dz * 1e-9
        assert abs(w[n] - exact) / exact < 5e-3


def two_layer(d=10.0, sigma=0.016, phi_B=1.0):
    gan = sp1d.Layer("GaN", 60.0, 10.0, 0.0, -3.4, 0.2, 1.5, 0.0)
    bar = sp1d.Layer("B", d, 9.6, 0.42, -3.55, 0.25, 1.6, -sigma)
    return sp1d.Structure([gan, bar], phi_B=phi_B, dz=0.1, T=300.0)


def test_gauss_law_and_surface_pinning():
    s = two_layer()
    r = s.solve()
    assert r["converged"]
    assert abs(r["Ec"][-1] - 1.0) < 1e-6            # pinned surface
    dz = s.dz * 1e-9
    total = s.sheet.sum() + sp1d.QE * (r["p"] - r["n"]).sum() * dz
    Es = (r["Ec"][-1] - r["Ec"][-2]) / dz            # field from the band slope (V/m)
    assert abs(s.eps[-1] * Es - total) / abs(total) < 0.03


def test_density_bounded_by_bound_charge_and_grows_with_barrier():
    ns = []
    for d in (5.0, 10.0, 20.0, 40.0):
        r = two_layer(d=d).solve()
        ns.append(r["ns_cm2"])
    sig = 0.016 / sp1d.QE * 1e-4
    assert all(n < sig for n in ns)
    assert all(b > a for a, b in zip(ns, ns[1:]))


def test_grid_convergence():
    a = two_layer().solve()["ns_cm2"]
    s = two_layer()
    s2 = sp1d.Structure(s.layers, phi_B=1.0, dz=0.05, T=300.0)
    b = s2.solve()["ns_cm2"]
    assert abs(a - b) / b < 0.01


def test_fermi_half_limits():
    """F_1/2 tends to exp(eta) for eta << 0 and to (4/3 sqrt(pi)) eta^1.5 for eta >> 0;
    its derivative is checked by finite differences."""
    F, _ = sp1d.fermi_half(np.array([-10.0]))
    assert abs(F[0] / np.exp(-10.0) - 1) < 1e-3
    F, _ = sp1d.fermi_half(np.array([35.0]))
    assert abs(F[0] / (4 / (3 * np.sqrt(np.pi)) * 35.0 ** 1.5) - 1) < 2e-3
    e = np.linspace(-5, 10, 31)
    F1, dF = sp1d.fermi_half(e)
    F2, _ = sp1d.fermi_half(e + 1e-4)
    assert np.allclose((F2 - F1) / 1e-4, dF, rtol=2e-3)


def test_neutral_doped_layer():
    """A thick n-doped layer with classical electrons is neutral in its interior: n = N_d+."""
    kT = sp1d.KB * 300.0
    lay = sp1d.Layer("n", 200.0, 10.0, 0.0, -3.4, 0.2, 1.0, 0.0, Nd=1e18)
    s0 = sp1d.Structure([lay], phi_B=0.1, dz=0.1, T=300.0, quantum_window=(0.0, 0.0),
                        classical_outside=True, holes=False)
    r = s0.solve()
    mid = len(s0.z) // 3
    assert abs(r["n"][mid] / r["Nd_ionized"][mid] - 1) < 1e-3
    F, _ = sp1d.fermi_half(np.array([-r["Ec"][mid] / kT]))
    assert abs(s0.Nc[mid] * F[0] / 1e24 - 1) < 1e-3
