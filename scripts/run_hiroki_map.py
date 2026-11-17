"""Line charge needed by the x = 0.35, 85 nm sample of Hiroki et al. at the
relaxation read from their reciprocal-space map.

The AlGaN contours of their Fig. 4(b) lie between R = -0.17 and +0.20 of the
full-relaxation distance, centred near R = 0.03
(data/hiroki2002_digitized.json, "fig4b_map_85nm_x0.35"), so the average
relaxation is at most about 0.2. For R = 0.1 and 0.2 this script finds the
line charge that reproduces the measured density, with the calibration of
scripts/run_hiroki.py (results/hiroki_analysis.json) for each phi_B and for
both treatments of the room-temperature excess. The result is given as the
fraction f of the bulk-terminated charge -P_H b; f > 1 means that even
bulk-terminated lines cannot produce the drop at that relaxation.

Outputs results/hiroki_map.json.
"""

import json
import os
import sys

from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_hiroki as H  # noqa: E402
from nmp import materials as M  # noqa: E402

ROOT = os.path.join(HERE, "..")
QE = H.QE


def main():
    d = M.dft()
    res = json.load(open(os.path.join(ROOT, "results", "hiroki_analysis.json")))
    gan, bar, eps_c = H.setup(0.35, d)
    bulk = -bar["P_H"] * bar["a"] * 1e-10
    out = {"description": __doc__.split("\n\n")[0], "Lambda_bulk_e_per_nm": bulk / QE * 1e-9, "cases": {}}
    for treat in ("as_measured", "scaled_to_77K"):
        r = res[treat]["0.35"]
        for phi, v in r["phi"].items():
            ds = v["dsig_cm2"] * 1e4 * QE
            n_meas = v["relaxed"]["85"]["n_measured"]
            row = {}
            for R in (0.1, 0.2):
                g = lambda f: H.n2deg(85.0, float(phi), gan, bar, eps_c, R=R, Lam=f * bulk, dsig=ds) - n_meas
                try:
                    row[str(R)] = float(brentq(g, -1.0, 3.0, xtol=1e-3))
                except ValueError:
                    row[str(R)] = None
            out["cases"]["%s/phi_%s" % (treat, phi)] = row
            print(treat, phi, row, flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "hiroki_map.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
