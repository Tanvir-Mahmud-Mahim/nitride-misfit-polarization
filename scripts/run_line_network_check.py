"""Check of the line-scattering kernel used for the mobility bound
(src/nmp/lines.py) against a direct solution of the Boltzmann equation for a
hexagonal misfit network.

The kernel treats the misfit lines as randomly oriented. For randomly oriented
lines the scattering is isotropic on average and the relaxation-time formula
is exact. A real (0001) misfit network has three line directions at 60
degrees. Each line family scatters an electron only by reversing the momentum
component normal to the lines (a mirror reflection on the Fermi circle), so
the scattering is anisotropic and the relaxation-time formula is no longer
exact. Here the linearized Boltzmann equation is solved on the Fermi circle
for the three-family network, with the same screened line potential, form
factor and total line length per area, and the resulting transport time is
compared with the kernel.

Single family of parallel lines (normal at angle theta_n, length per area n_L):
an electron at angle phi with normal momentum k_n = k |cos(phi - theta_n)|
scatters only to its mirror state, at the total rate (Fermi golden rule)

    Gamma(phi) = n_L m |U(2 k_n)|^2 / (hbar^3 k_n),

with U(q) = lambda e F(q) / (2 eps q eps_s(q)) the screened in-plane
transform of one line (same F, eps_s as src/nmp/lines.py). For three
families of n_L = L_A/3 each, the linearized Boltzmann equation reads

    cos(phi) = sum_j Gamma_j(phi) [chi(phi) - chi(phi_j*)],

and the transport time is tau = 2 <chi cos(phi)>. The kernel gives
tau_kernel = 1 / rate_lines(...). Their ratio does not depend on lambda or
L_A; it is reported for the 2DEG densities of the relaxed samples.

Outputs results/line_network_check.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import lines, materials as M  # noqa: E402

HBAR, M0, QE, EPS0 = lines.HBAR, lines.M0, lines.QE, lines.EPS0
ROOT = os.path.join(HERE, "..")


def screened_U(q, lam, n_s, m_rel, eps_r):
    m = m_rel * M0
    eps = eps_r * EPS0
    b = lines.fang_howard_b(n_s, m_rel, eps_r)
    F = (b / (b + q)) ** 3
    G = (8 * b ** 3 + 9 * b ** 2 * q + 3 * b * q ** 2) / (8 * (b + q) ** 3)
    qtf = m * QE ** 2 / (2 * np.pi * eps * HBAR ** 2)
    return lam * QE * F / (2 * eps * q * (1 + qtf / q * G))


def tau_network(n_s, lam, L_A, m_rel, eps_r, normals_deg=(0.0, 60.0, 120.0), N=1200):
    m = m_rel * M0
    k = np.sqrt(2 * np.pi * n_s)
    phi = (np.arange(N) + 0.5) * 2 * np.pi / N
    A = np.zeros((N, N))
    nL = L_A / len(normals_deg)
    for th in np.radians(normals_deg):
        kn = k * np.abs(np.cos(phi - th))
        U = screened_U(2 * kn, lam, n_s, m_rel, eps_r)
        Gam = nL * m * U ** 2 / (HBAR ** 3 * kn)
        # mirror state: phi* = 2 th + pi - phi
        mirror = np.mod(2 * th + np.pi - phi, 2 * np.pi)
        j = np.rint(mirror / (2 * np.pi / N) - 0.5).astype(int) % N
        assert np.allclose(phi[j], mirror, atol=1e-9)
        A[np.arange(N), np.arange(N)] += Gam
        A[np.arange(N), j] -= Gam
    chi = np.linalg.lstsq(A, np.cos(phi), rcond=None)[0]
    chi -= chi.mean()
    return 2 * np.mean(chi * np.cos(phi))


def main():
    gan = M.algan(0.0, 77.0, M.dft())
    m, eps = gan["me"], gan["eps"]
    lam, L_A = QE * 1e9, 1e7          # 1 e/nm, 10 lines per um (ratio is independent of both)
    out = {"m_rel": m, "eps_r": eps, "cases": []}
    for n13 in (0.50, 0.645, 0.75, 1.0, 1.2):
        n_s = n13 * 1e17
        t_net = tau_network(n_s, lam, L_A, m, eps)
        t_net_fine = tau_network(n_s, lam, L_A, m, eps, N=2400)
        t_rand_bte = tau_network(n_s, lam, L_A, m, eps, normals_deg=tuple(np.arange(0, 180, 7.5)), N=2400)
        t_ker = 1.0 / lines.rate_lines(n_s, lam, L_A, m, eps)
        row = {"n_s_1e13cm2": n13, "tau_network_over_kernel": t_net / t_ker,
               "tau_network_over_kernel_N2400": t_net_fine / t_ker,
               "tau_24_directions_over_kernel": t_rand_bte / t_ker,
               "bound_factor_sqrt": float(np.sqrt(t_net_fine / t_ker))}
        out["cases"].append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "line_network_check.json"), "w"), indent=1)


if __name__ == "__main__" and "--shen" not in sys.argv:
    main()


def tau_network_with_background(n_s, lam, L_A, m_rel, eps_r, tau_b, normals_deg=(0.0, 60.0, 120.0), N=1200):
    """Transport time with the line network plus an isotropic background of transport time tau_b."""
    m = m_rel * M0
    k = np.sqrt(2 * np.pi * n_s)
    phi = (np.arange(N) + 0.5) * 2 * np.pi / N
    A = np.eye(N) / tau_b
    nL = L_A / len(normals_deg)
    for th in np.radians(normals_deg):
        kn = k * np.abs(np.cos(phi - th))
        U = screened_U(2 * kn, lam, n_s, m_rel, eps_r)
        Gam = nL * m * U ** 2 / (HBAR ** 3 * kn)
        mirror = np.mod(2 * th + np.pi - phi, 2 * np.pi)
        j = np.rint(mirror / (2 * np.pi / N) - 0.5).astype(int) % N
        A[np.arange(N), np.arange(N)] += Gam
        A[np.arange(N), j] -= Gam
    chi = np.linalg.solve(A, np.cos(phi))
    return 2 * np.mean(chi * np.cos(phi))


def shen_bounds():
    """Mobility limits of the relaxed Shen sample with the hexagonal network."""
    from scipy.optimize import brentq
    gan = M.algan(0.0, 77.0, M.dft())
    m, eps = gan["me"], gan["eps"]
    base = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]
    res = {}
    for key, b in base.items():
        mo = b["mobility"]
        n_s = mo["n2DEG_relaxed"] * 1e4
        L_A = b["L_A_per_cm"] * 100.0
        mstar = m * M0
        tau_meas = mo["mu2DEG_relaxed"] * 1e-4 * mstar / QE
        # strict: lines only, network
        f = lambda L: tau_network(n_s, L * QE * 1e9, L_A, m, eps) - tau_meas
        strict_net = brentq(f, 0.05, 20.0, xtol=1e-3)
        est = []
        for mu_ref in mo["mu2DEG_pseudomorphic"]:
            tau_b = mu_ref * 1e-4 * mstar / QE
            g = lambda L: tau_network_with_background(n_s, L * QE * 1e9, L_A, m, eps, tau_b) - tau_meas
            est.append(float(brentq(g, 0.01, 20.0, xtol=1e-3)))
        res[key] = {"strict_kernel": mo["Lambda_strict_upper_bound_e_per_nm"],
                    "strict_network": float(strict_net),
                    "estimate_kernel": mo["Lambda_estimate_range"],
                    "estimate_network_with_background": [min(est), max(est)]}
        print(key, json.dumps(res[key]), flush=True)
    return res


def elwaradi_bounds():
    """Same for the compressed barrier of Elwaradi et al. (20 nm channel), with the
    inputs of run_elwaradi.py: n_s = 2.2e12 cm^-2, mu = 310 cm^2/Vs, reference
    (50 nm channel) 460 cm^2/Vs, room temperature."""
    from scipy.optimize import brentq
    gan = M.algan(0.0, 300.0, M.dft())
    m, eps = gan["me"], gan["eps"]
    mb = json.load(open(os.path.join(ROOT, "results", "elwaradi_analysis.json")))["mobility_t20"]
    n_s, mstar = 2.2e12 * 1e4, gan["me"] * M0
    tau_meas, tau_b = 310e-4 * mstar / QE, 460e-4 * mstar / QE
    res = {}
    for key, v in mb.items():
        L_A = v["L_A_per_cm"] * 100.0
        s = brentq(lambda L: tau_network(n_s, L * QE * 1e9, L_A, m, eps) - tau_meas, 0.05, 30.0, xtol=1e-3)
        e = brentq(lambda L: tau_network_with_background(n_s, L * QE * 1e9, L_A, m, eps, tau_b) - tau_meas,
                   0.01, 30.0, xtol=1e-3)
        res[key] = {"strict_kernel": v["Lambda_strict_upper_bound_e_per_nm"], "strict_network": float(s),
                    "estimate_kernel": v["Lambda_estimate_vs_50nm_e_per_nm"],
                    "estimate_network_with_background": float(e),
                    "bare_e_per_nm": abs(v["Lambda_bare_e_per_nm"])}
        print("elwaradi", key, json.dumps(res[key]), flush=True)
    return res


if __name__ == "__main__" and "--shen" in sys.argv:
    out = json.load(open(os.path.join(ROOT, "results", "line_network_check.json")))
    out["shen"] = shen_bounds()
    out["elwaradi"] = elwaradi_bounds()
    json.dump(out, open(os.path.join(ROOT, "results", "line_network_check.json"), "w"), indent=1)
