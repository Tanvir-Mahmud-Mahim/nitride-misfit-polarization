"""Size of the second-order (nonlinear) piezoelectric term next to the terms of
Eq. (1), using the second-order coefficients of Pal, Tse, Haxha, Migliorato and
Tomic, Phys. Rev. B 84, 085211 (2011), Table I and Eq. (3):

    P = Psp + e33 e_perp + 2 e31 e + e311 e^2 + e333 e_perp^2 + e133 e e_perp,

with e the in-plane and e_perp = -2 (C13/C33) e the c-axis strain. Alloy
coefficients are interpolated linearly. Reported: the change of the
second-order part between the coherent strain eps_c and the relaxed strain
(1 - r) eps_c, compared with the bare-end term 2 r |eps_c| P_H and with the
first-order (proper) piezoelectric change of Eq. (1).

Outputs results/second_order_check.json.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from nmp import materials as M  # noqa: E402

PAL = {"GaN": {"e311": 6.185, "e333": -8.090, "e133": 1.543},
       "AlN": {"e311": 5.850, "e333": -10.750, "e133": 4.533}}


def coeff(x, k):
    return (1 - x) * PAL["GaN"][k] + x * PAL["AlN"][k]


def case(name, x, sub_x, r):
    d = M.dft()
    f = M.algan(x, 300.0, d)
    s = M.algan(sub_x, 300.0, d)
    eps_c = (s["a"] - f["a"]) / f["a"]
    nu = 2 * f["C13_over_C33"] if "C13_over_C33" in f else None
    out = {}
    for nu_ in ([nu] if nu else []) + [0.50, 0.60]:
        Q = lambda e: (coeff(x, "e311") * e ** 2 + coeff(x, "e333") * (nu_ * e) ** 2
                       + coeff(x, "e133") * e * (-nu_ * e))
        dQ = Q((1 - r) * eps_c) - Q(eps_c)
        bare = 2 * r * abs(eps_c) * f["P_H"]
        first = 2 * r * eps_c * (f["e31_prop"] - f["e33"] * f["C13_over_C33"])
        out["nu=%.3f" % nu_] = {"dQ_Cm2": dQ, "ratio_to_bare_end_term": abs(dQ) / bare,
                                "ratio_to_first_order_piezo": abs(dQ / first)}
    return {"name": name, "x": x, "eps_c": eps_c, "r": r, "results": out}


if __name__ == "__main__":
    f = M.algan(0.22, 300.0, M.dft())

    res = [case("Shen Al0.22Ga0.78N on GaN, r = 0.78", 0.22, 0.0, 0.78),
           case("AlN on GaN, r = 0.5", 1.0, 0.0, 0.5)]
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "..", "results", "second_order_check.json"), "w"), indent=1)
