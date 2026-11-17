"""Scattering of a two-dimensional carrier gas by charged lines lying in the
interface plane (misfit dislocations), and the misfit-line charge it implies.

The kernel is the one used for the GaN/AlN hole gas in T. M. Mahim, A. S. M.
Mohsin and M. M. Rahman, Phys. Rev. B 114, L291304 (2026) (code archived at
doi:10.5281/zenodo.21791358, src/gan2dhg/scatter2d.py, w_misfit_lines): a
straight line of charge lambda per unit length in the interface plane has the
in-plane transform lambda e / (2 eps q); for randomly oriented lines of total
length L_A per unit area the orientation average contributes 2/q, so

    W(q) = L_A (lambda e / (2 eps q))^2 (2/q) F(q)^2,

with F(q) = (b/(b+q))^3 the Fang-Howard form factor for a charge at the
interface. The transport rate is

    1/tau_tr = m / (2 pi hbar^3) * int_0^{2 pi} d theta W(q)/eps(q)^2 (1 - cos theta),
    q = 2 k_F sin(theta/2),

with Thomas-Fermi screening eps(q) = 1 + (q_TF/q) G(q), q_TF = m e^2/(2 pi eps
hbar^2) and the Fang-Howard screening form factor
G(q) = (8b^3 + 9b^2 q + 3 b q^2)/(8 (b+q)^3) (Ando, Fowler and Stern, Rev.
Mod. Phys. 54, 437 (1982)). The variational parameter is
b = (33 m e^2 N*/(8 eps hbar^2))^(1/3), N* = N_depl + 11 n_s/32.
"""

import numpy as np

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12


def fang_howard_b(n_s_m2, m_rel, eps_r, n_depl_m2=0.0):
    m = m_rel * M0
    eps = eps_r * EPS0
    return (33 * m * QE ** 2 * (n_depl_m2 + 11 * n_s_m2 / 32) / (8 * eps * HBAR ** 2)) ** (1 / 3)


def rate_lines(n_s_m2, lam_Cm, L_A_m, m_rel, eps_r, n_depl_m2=0.0, nth=2000):
    """Transport scattering rate (1/s) from randomly oriented interface lines."""
    m = m_rel * M0
    eps = eps_r * EPS0
    kF = np.sqrt(2 * np.pi * n_s_m2)
    b = fang_howard_b(n_s_m2, m_rel, eps_r, n_depl_m2)
    th = (np.arange(nth) + 0.5) * (2 * np.pi / nth)
    q = 2 * kF * np.sin(th / 2)
    F = (b / (b + q)) ** 3
    G = (8 * b ** 3 + 9 * b ** 2 * q + 3 * b * q ** 2) / (8 * (b + q) ** 3)
    qtf = m * QE ** 2 / (2 * np.pi * eps * HBAR ** 2)
    epsq = 1 + qtf / q * G
    W = L_A_m * (lam_Cm * QE / (2 * eps * q)) ** 2 * (2 / q) * F ** 2
    integrand = W / epsq ** 2 * (1 - np.cos(th))
    return m / (2 * np.pi * HBAR ** 3) * integrand.sum() * (2 * np.pi / nth)


def mobility_from_rate(rate, m_rel):
    """cm^2/Vs."""
    return QE / (m_rel * M0 * rate) * 1e4


def lambda_from_mobility_drop(mu_ref, mu_relaxed, n_s_m2, L_A_m, m_rel, eps_r):
    """Line charge (C/m) whose scattering, added to the reference sample's,
    lowers the mobility from mu_ref to mu_relaxed (Matthiessen's rule)."""
    extra = QE / (m_rel * M0) * (1 / (mu_relaxed * 1e-4) - 1 / (mu_ref * 1e-4))   # 1/s
    unit = rate_lines(n_s_m2, 1.0, L_A_m, m_rel, eps_r)                             # rate for lambda = 1 C/m
    return np.sqrt(extra / unit)
