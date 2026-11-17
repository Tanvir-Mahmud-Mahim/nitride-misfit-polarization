"""Bound charge at c-plane nitride interfaces, coherent or partly relaxed, and
the charge carried by misfit dislocations.

Conventions. z runs along [0001] of a metal-polar crystal. P is the formal
polarization along z (C/m^2). The bound charge at an interface with material
B below and film F above is sigma = P_B - P_F. "H" denotes the layered-
hexagonal reference (formal polarization zero), "ZB" the zinc-blende reference.

1. Formal polarization of the zinc-blende reference. In the hexagonal setting
   its formal polarization along [111] is e sqrt(3)/(2 a^2) = (3/4) Q, with
   Q = e c / Omega = 2 e/(sqrt(3) a^2) the polarization quantum (Dreyer et al.,
   Phys. Rev. X 6, 021038 (2016), Sec. S1 of their Supplemental Material).
   Hence P_ZB = P_H - e sqrt(3)/(2 a^2).

2. Strain. Under biaxial in-plane strain eps (and the c-axis response
   eps_3 = -2 (C13/C33) eps) the formal polarization on the H branch changes as
   dP = 2 eps (e31_prop - P_H) + eps_3 e33: the improper coefficient
   e31_prop - P_H appears because the polarization is a dipole per volume and
   the area changes (Vanderbilt, J. Phys. Chem. Solids 61, 147 (2000), Eq. (15)).

3. Relaxed interfaces. A film of mismatch eps_c = (a_B - a_F)/a_F relaxed by a
   fraction r keeps strain (1 - r) eps_c. The relaxation is made by misfit
   dislocations whose in-plane edge components b relieve 2 r |eps_c| of areal
   strain, so their length per area is L_A = 2 r |eps_c| / b. Writing the
   bound charge of the relaxed interface with the H-branch polarizations
   alone,

       sigma = sigma_coh + 2 r eps_c (e31_prop - e33 C13/C33) - 2 r eps_c P_H,F,

   the last term is the charge of L_A lines, each carrying lambda = P_H,F b per
   unit length: the bulk-terminated ends of the extra (or missing) half-planes.
   For lines of any actual charge Lambda the exact first-order result is

       sigma = sigma_coh + 2 r eps_c (e31_prop - e33 C13/C33) + Lambda L_A,

   with Lambda = -lambda for a tensile film (extra film columns, eps_c > 0)
   and Lambda = +lambda for a compressive one. The usual prescriptions are
   special cases: "H, improper" is Lambda = -/+ P_H b (bulk-terminated half-
   planes); "ZB, proper" is Lambda = 0 (neutral lines) up to an r-independent
   offset; electron-counting cores give Lambda = -/+ (P_H - P_f,ZB) b = -/+ P_ZB b.

4. Electron counting. A cation dangling bond of a bulk-terminated metal-polar
   column carries 3/4 e, an anion one 5/4 e. Emptying the cation bond (or
   filling the anion bond) removes 3/4 e per site, i.e. e sqrt(3)/(2 a^2) per
   area (Vanderbilt and King-Smith, Phys. Rev. B 48, 4442 (1993), Table I).
"""

import numpy as np

E = 1.602176634e-19


def quantum_c(a_ang):
    """Polarization quantum along c, e c / Omega = 2 e / (sqrt(3) a^2) (C/m^2)."""
    a = a_ang * 1e-10
    return 2 * E / (np.sqrt(3) * a * a)


def p_zb_formal(a_ang):
    """Formal polarization of the zinc-blende reference along [111], e sqrt(3)/(2 a^2) (C/m^2)."""
    a = a_ang * 1e-10
    return E * np.sqrt(3) / (2 * a * a)


def eps3(m, eps):
    """c-axis strain of a biaxially strained layer (plane stress)."""
    return -2 * m["C13_over_C33"] * eps


def p_h_improper(m, eps):
    """H-branch formal polarization of a layer with biaxial strain eps (improper response)."""
    return m["P_H"] + 2 * eps * (m["e31_prop"] - m["P_H"]) + eps3(m, eps) * m["e33"]


def p_zb_proper(m, eps):
    """ZB-referenced polarization with the proper response (conventional prescription)."""
    return (m["P_H"] - p_zb_formal(m["a"])) + 2 * eps * m["e31_prop"] + eps3(m, eps) * m["e33"]


def sigma_interface(film, sub, r, prescription, eps_c=None):
    """Bound charge (C/m^2) below a film relaxed by fraction r on a relaxed substrate."""
    if eps_c is None:
        eps_c = (sub["a"] - film["a"]) / film["a"]
    e = (1 - r) * eps_c
    if prescription == "H_improper":
        return sub["P_H"] - p_h_improper(film, e)
    if prescription == "ZB_proper":
        return p_zb_proper(sub, 0.0) - p_zb_proper(film, e)
    raise ValueError(prescription)


def misfit_line_density(r, eps_c, b_ang):
    """Total length of misfit-dislocation edge component per unit area (m^-1) for relaxation r.

    Relieved areal strain 2 r |eps_c| equals sum(b L)/A, so L/A = 2 r |eps_c| / b.
    """
    return 2.0 * r * abs(eps_c) / (b_ang * 1e-10)


def sigma_decomposed(film, sub, r, Lam, eps_c=None):
    """Coherent charge + proper relaxation + misfit-line charge Lam (C/m) times L_A."""
    if eps_c is None:
        eps_c = (sub["a"] - film["a"]) / film["a"]
    s_coh = sigma_interface(film, sub, 0.0, "H_improper", eps_c)
    k = film["e31_prop"] - film["e33"] * film["C13_over_C33"]
    return s_coh + 2 * r * eps_c * k + Lam * misfit_line_density(r, eps_c, film["a"])


def line_charge_limits(film, eps_c):
    """Lambda (C/m) for bulk-terminated, electron-counting and neutral misfit lines."""
    b = film["a"] * 1e-10
    s = -1.0 if eps_c > 0 else 1.0
    return {"bulk-terminated": s * film["P_H"] * b,
            "electron counting": s * (film["P_H"] - p_zb_formal(film["a"])) * b,
            "neutral": 0.0}


def to_cm2(sigma_Cm2):
    """Sheet charge in C/m^2 to elementary charges per cm^2."""
    return sigma_Cm2 / E * 1e-4
