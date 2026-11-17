"""Independent check of the hole-gas input with kpenvelope.

The hole-gas link (scripts/run_prb_link.py) uses the misfit-line fit archived
with Phys. Rev. B 114, L291304 (2026). There, the scattering by lines lying in
the interface was computed with a Fang-Howard envelope, and the self-consistent
six-band envelope came from that paper's own solver (gan2dhg, archive
doi:10.5281/zenodo.21791358, results/well.json). This script recomputes the
self-consistent hole gas with a separate code, kpenvelope (T. M. Mahim,
doi:10.5281/zenodo.22015269; six-band Chuang-Chang model, Rinke et al. 2008
parameters and the same deformation potentials), for the same problem: GaN strained to AlN, hard walls 15 nm apart,
the fixed negative sheet at the interface, p = 4.6e13 cm^-2, eps_r = 10.4.

It compares the two hole distributions through the quantity the line
scattering depends on, the remote form factor F(q) = Int rho(z) exp(-q z) dz,
and turns the difference into a factor on the transport scattering rate of
randomly oriented interface lines (kernel L_A (Lambda e / 2 eps q)^2 (2/q)
F(q)^2, the kernel of that paper, weighted by (1 - cos theta) over the Fermi
circle of each subband and by its density; screening is the same in both
cases and cancels in the ratio to first order, so it is left out). Because the
fit fixes the product L_A f^2 from the measured mobility, the relaxation the
lines need scales with the inverse of this factor.

Inputs: data/prb2026_well_profile.json (z and p(z) copied from results/
well.json of the archive). Output: results/kp_check.json.
"""

import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

import kpenvelope as kp  # noqa: E402

PS_CM2 = 4.6e13
L_NM = 15.0
N = 161
EPS_R = 10.4
STRAIN = np.diag([-0.024145500156788947, -0.024145500156788947, 0.012861422194068484])


def remote_ff(z, rho, q):
    rho = rho / np.trapezoid(rho, z)
    return np.trapezoid(rho * np.exp(-q * (z - z.min())), z)


def line_rate(z, rho, subbands):
    """Density-weighted transport rate of interface lines, arbitrary units."""
    tot = 0.0
    for n, kF in subbands:
        th = np.linspace(1e-4, 2 * np.pi - 1e-4, 721)
        q = 2 * kF * np.sin(th / 2)
        F = np.array([remote_ff(z, rho, qq) for qq in q])
        w = (1.0 / q) ** 2 * (2.0 / q) * F ** 2
        tot += n * np.trapezoid(w * (1 - np.cos(th)), th)
    return tot / sum(n for n, _ in subbands)


def main():
    ref = json.load(open(os.path.join(ROOT, "data", "prb2026_well_profile.json")))
    z_ref, p_ref = np.array(ref["z_nm"]), np.array(ref["p_of_z"])
    sub_ref = [(m["n_cm2"], m["kF_per_nm"]) for m in ref["subbands"]]

    # Deformation potentials (eV) as used in the archived calculation: Extended
    # Data Table 1 of Chang et al., Nature Electronics 9, 346 (2026), which
    # attributes them to Yan et al., Phys. Rev. B 90, 125118 (2014):
    # D1 = acz - (acz - D1) = -11.3 + 6.07, D2 = act - (act - D2) = -4.9 + 8.88.
    gan = dataclasses.replace(kp.gan_rinke2008(), D1=-11.3 + 6.07, D2=-4.9 + 8.88,
                              D3=5.38, D4=-2.69, D5=-2.56, D6=-3.88)
    z = np.linspace(0.0, L_NM, N)
    params, edge = kp.layered_profile(z, [(L_NM, gan, 0.0)])
    res = kp.solve_self_consistent_hetero(
        z, params, edge, ps=PS_CM2 * 1e-14, eps_r=EPS_R, n_states=8, mixing=0.3,
        tol=2e-5, max_iter=150, strain_list=[STRAIN] * z.size, filling="kgrid",
        kmax=2.4, nk=40, ntheta=1, sheet_z=0.0)
    rho = np.asarray(res.density)
    occ = np.asarray(res.occupations)          # nm^-2 per state

    def stats(zz, rr):
        rr = rr / np.trapezoid(rr, zz)
        c = np.trapezoid(zz * rr, zz)
        return c, np.sqrt(np.trapezoid((zz - c) ** 2 * rr, zz))

    c_ref, w_ref = stats(z_ref, p_ref)
    c_kp, w_kp = stats(z, rho)
    qs = [0.5, 1.0, 2.0, 3.2]
    rate_ref = line_rate(z_ref, p_ref, sub_ref)
    rate_kp_refsub = line_rate(z, rho, sub_ref)
    out = {"description": __doc__.split("\n\n")[0],
           "kpenvelope_version": kp.__version__,
           "converged": bool(res.converged), "iterations": int(res.iterations),
           "residual_eV": float(res.residual),
           "n_states_sufficient": bool(getattr(res, "n_states_sufficient", True)),
           "occupations_cm2": [float(o * 1e14) for o in occ],
           "centroid_nm": {"gan2dhg (PRB archive)": c_ref, "kpenvelope": c_kp},
           "rms_width_nm": {"gan2dhg (PRB archive)": w_ref, "kpenvelope": w_kp},
           "F_remote": {str(q): {"gan2dhg": float(remote_ff(z_ref, p_ref, q)),
                                 "kpenvelope": float(remote_ff(z, rho, q))} for q in qs},
           "line_rate_ratio_kp_over_ref_same_subbands": rate_kp_refsub / rate_ref,
           "subbands_ref": sub_ref}
    json.dump(out, open(os.path.join(ROOT, "results", "kp_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
