"""Spacer series of Shen et al. (Fig. 3 of Appl. Phys. Lett. 76, 2746 (2000)),
local calibration.

The model of scripts/run_shen_spacer.py does not reproduce the strained spacer
samples with the calibration of the thickness series. Here the fixed interface
sheet is instead fitted only to the spacer samples closest to the s = 50 nm
sample (s = 30 nm alone, and s = 20 and 30 nm), at 77 K, with the barrier
mobility bracketed by 0 and 83 cm^2/Vs as in run_shen.py. For the s = 50 nm
sample (75 nm of AlGaN in total, relaxation not measured) the script then
finds the relaxation R that each line charge needs to give the measured
density, and, for each R, the strict mobility bound for randomly oriented
lines. The relaxation of a physical answer lies below 0.78, the value measured
on the thicker (103 nm) relaxed sample.

Outputs results/shen_spacer_local.json (read by scripts/make_numbers.py).
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from nmp import lines, materials as M, polarization as PZ, sp1d  # noqa: E402
import run_shen as RS  # noqa: E402
import run_shen_spacer as SP  # noqa: E402

ROOT = os.path.join(HERE, "..")
QE = RS.QE
SCAN = [-0.3, -0.5, -0.7, -0.9, -1.1, -1.3, -1.6, -2.0, -2.645]


def main():
    d = M.dft()
    data = json.load(open(os.path.join(ROOT, "data", "shen2000_fig3_digitized.json")))
    series = json.load(open(os.path.join(ROOT, "results", "shen_analysis.json")))["baseline"]
    out = {}
    for key, (T, mu2) in {"77K_mu2_0": (77.0, 0.0), "77K_mu2_83": (77.0, 83.0)}.items():
        phi = series[key]["phi_B_eV"]
        gan, bar, eps_c = RS.setup(T, d)
        ss = data["spacer_nm"]
        nH = np.array(data["ns_77K_1e13cm2"]) * 1e13
        muH = np.array(data["mu_77K_cm2Vs"])
        m, eps = gan["me"], gan["eps"]
        i50 = ss.index(50)

        def run(s, ds, **kw):
            return SP.model(s, T, phi, gan, bar, eps_c, dsig=ds, **kw)

        def r_needed(ds, Lam):
            def h(Rx):
                n1, n2 = run(50.0, ds, R=Rx, Lam=Lam)
                n1x, _ = RS.split(nH[i50], muH[i50], n2, mu2)
                return n1 - n1x
            try:
                return float(brentq(h, 1e-3, 1.0, xtol=1e-3))
            except ValueError:
                return None

        for cal in ([30], [20, 30]):
            idx = [ss.index(c) for c in cal]

            def logs(ds):
                v = []
                for i in idx:
                    n1, n2 = run(ss[i], ds)
                    n1x, _ = RS.split(nH[i], muH[i], n2, mu2)
                    v.append(np.log(n1 / n1x))
                return np.array(v)

            ds = brentq(lambda x: logs(x).sum(), -0.05, 0.05, xtol=1e-6)
            res = {"dsig_cm2": ds / QE * 1e-4, "scan": []}
            for L in SCAN:
                Rn = r_needed(ds, L * QE * 1e9)
                bnd = None
                if Rn:
                    _, n2 = run(50.0, ds, R=Rn, Lam=L * QE * 1e9)
                    n1x, mu1x = RS.split(nH[i50], muH[i50], n2, mu2)
                    LA = PZ.misfit_line_density(Rn, eps_c, bar["a"])
                    unit = lines.rate_lines(n1x * 1e4, 1.0, LA, m, eps)
                    bnd = float(np.sqrt(QE / (m * sp1d.M0) / (mu1x * 1e-4) / unit) / QE * 1e-9)
                res["scan"].append((L, Rn, bnd))
                print(key, cal, L, Rn, bnd, flush=True)
            for nm, Lam in RS.limits(bar).items():
                res[nm] = r_needed(ds, Lam)
            out["%s_calib_%s" % (key, "_".join(map(str, cal)))] = res
    json.dump(out, open(os.path.join(ROOT, "results", "shen_spacer_local.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
