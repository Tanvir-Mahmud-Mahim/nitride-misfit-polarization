"""Material parameters of wurtzite AlN, GaN and Al(x)Ga(1-x)N used in the
electrostatic models, each with its source.

Polarization (H reference), proper e31, e33 and C13/C33 come from the
Berry-phase calculations of this work (results/dft_polarization.json,
scripts/dft_polarization.py). Everything else is taken from:

* Vurgaftman and Meyer, J. Appl. Phys. 94, 3675 (2003), Table I: in-plane
  lattice constants at 300 K (GaN 3.189, AlN 3.112 angstrom), band gaps at 0 K
  (GaN 3.510, AlN 6.25 eV) with Varshni alpha (0.909, 1.799 meV/K) and beta
  (830, 1462 K), electron masses parallel to c (GaN 0.20, AlN 0.32 m0); AlGaN
  gap bowing C = 0.7 eV (Table III).
* Ambacher et al., J. Appl. Phys. 87, 334 (2000): static dielectric constants
  along c, eps33 = 10.4 (GaN) and 10.7 (AlN) (Table I); conduction-band offset
  rule Delta E_c = 0.7 [E_g(x) - E_g(0)] (Eq. (19)).

Alloys: linear interpolation (Vegard) except the gap (bowing). The
polarization constants are interpolated linearly; the bowing of the
spontaneous polarization recommended by Vurgaftman and Meyer, and Eq. (17) of
Ambacher et al. for the dielectric constant, are used as sensitivity tests
(scripts/run_shen.py).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

A300 = {"GaN": 3.189, "AlN": 3.112}
EG0 = {"GaN": 3.510, "AlN": 6.25}
VARSHNI = {"GaN": (0.909e-3, 830.0), "AlN": (1.799e-3, 1462.0)}
ME_PAR = {"GaN": 0.20, "AlN": 0.32}
EPS33 = {"GaN": 10.4, "AlN": 10.7}
BOW_EG = 0.7
CBO_FRACTION = 0.7


def dft():
    return json.load(open(os.path.join(ROOT, "results", "dft_polarization.json")))


def gap(name, T):
    a, b = VARSHNI[name]
    return EG0[name] - a * T * T / (T + b)


def algan(x, T=300.0, d=None, cbo=CBO_FRACTION, eps_rule="table", psp_bowing=0.0):
    """Parameters of relaxed Al(x)Ga(1-x)N. d: output of dft() (loaded if None).

    Sensitivity options: cbo (fraction of the gap difference in the conduction band),
    eps_rule ("table": linear between the Table I values of Ambacher et al.;
    "eq17": their Eq. (17), eps = -0.3 x + 10.4), psp_bowing (C/m^2; the
    spontaneous polarization of the alloy is the linear value plus
    psp_bowing * x (1 - x); Vurgaftman and Meyer recommend a bowing parameter
    C = -0.021 C/m^2 with P(x) = linear - C x (1 - x), i.e. psp_bowing = 0.021)."""
    d = d or dft()
    lin = lambda f: x * f("AlN") + (1 - x) * f("GaN")
    eg = lin(lambda n: gap(n, T)) - BOW_EG * x * (1 - x)
    eps = lin(lambda n: EPS33[n]) if eps_rule == "table" else -0.3 * x + 10.4
    return {
        "x": x,
        "a": lin(lambda n: A300[n]),
        "P_H": lin(lambda n: d[n]["P_H"]) + psp_bowing * x * (1 - x),
        "e31_prop": lin(lambda n: d[n]["e31_proper"]),
        "e33": lin(lambda n: d[n]["e33"]),
        "C13_over_C33": lin(lambda n: d[n]["C13_over_C33"]),
        "Eg": eg,
        "dEc": cbo * (eg - gap("GaN", T)),
        "me": lin(lambda n: ME_PAR[n]),
        "eps": eps,
    }


def strained_P(m, eps):
    """Formal polarization (H branch) of a layer under biaxial strain eps
    (improper coefficient e31_prop - P_H; Dreyer et al., Phys. Rev. X 6, 021038 (2016))."""
    return (m["P_H"] + 2 * eps * (m["e31_prop"] - m["P_H"])
            - 2 * m["C13_over_C33"] * eps * m["e33"])
