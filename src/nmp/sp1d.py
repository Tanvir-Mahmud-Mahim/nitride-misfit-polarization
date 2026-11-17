"""One-dimensional Schroedinger-Poisson solver for c-plane nitride
heterostructures (electrons and holes, single-band effective mass).

Geometry: z runs along [0001] from the bottom of the structure (z = 0, deep in
the substrate or buffer) to the free surface (z = L). Layers are listed from
the bottom up. Each layer carries its own formal polarization P (C/m^2, H
branch, strained value); the bound sheet charge at an interface is
P_below - P_above (src/nmp/polarization.py), and extra sheet charges (for
example the misfit-line charge) can be added at any interface.

Boundary conditions
* top surface: the conduction-band edge sits phi_B above the Fermi level
  (surface Fermi-level pinning; the surface-state charge adjusts freely);
* bottom: zero electric field (a thick, neutral buffer).
The Fermi level is E_F = 0. Energies in eV, lengths in nm, densities in
m^-3 internally; sheet densities are returned in cm^-2.

Carriers in the quantum region are obtained from the eigenstates of the
effective-mass Hamiltonian (BenDaniel-Duke boundary conditions, finite
differences) with two-dimensional Fermi-Dirac occupation. Outside the quantum
window electrons are treated as a classical three-dimensional Fermi gas,
n = N_c F_1/2((E_F - E_c)/kT), with N_c = 2 (m kT / 2 pi hbar^2)^(3/2) (only
when classical_outside=True). Donors are fully ionized unless a donor depth
E_d is given; then N_d+ = N_d / (1 + 2 exp((E_F - E_c + E_d)/kT)) (spin
degeneracy 2).

Self-consistency uses the predictor-corrector scheme of Trellakis et al.,
J. Appl. Phys. 81, 7880 (1997): with the eigenstates of the current potential
held fixed, the Poisson equation is solved by Newton iteration with each
subband energy shifted by the local change of the potential, then the
eigenstates are recomputed; the loop ends when the potential changes by less
than tol.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal, solve_banded

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 8.617333262e-5     # eV/K


def _fd_table():
    """Normalized Fermi-Dirac integral F_1/2(eta) = (2/sqrt(pi)) int sqrt(x)/(1+exp(x-eta)) dx,
    tabulated for -40 <= eta <= 40; returns a cubic spline of its natural log."""
    eta = np.linspace(-40.0, 40.0, 4001)
    f = []
    for e in eta:
        up = max(e, 0.0) + 60.0
        v = quad(lambda x: np.sqrt(x) / (1.0 + np.exp(min(x - e, 700.0))), 0.0, up,
                 points=[max(e, 0.0)] if e > 0 else None, limit=200, epsabs=0, epsrel=1e-10)[0]
        f.append(2.0 / np.sqrt(np.pi) * v)
    return eta, CubicSpline(eta, np.log(np.array(f)))


_ETA, _LOGF = None, None


def fermi_half(eta):
    """F_1/2(eta) and its derivative F_-1/2(eta) (= dF_1/2/deta)."""
    global _ETA, _LOGF
    if _ETA is None:
        _ETA, _LOGF = _fd_table()
    eta = np.asarray(eta, float)
    lo = eta < _ETA[0]
    hi = eta > _ETA[-1]
    e = np.clip(eta, _ETA[0], _ETA[-1])
    lf = _LOGF(e)
    dlf = _LOGF(e, 1)
    F = np.exp(lf)
    dF = F * dlf
    ex = np.exp(np.minimum(eta, _ETA[0]))
    F = np.where(lo, ex, F)
    dF = np.where(lo, ex, dF)
    # degenerate limit beyond the table: (4/3 sqrt(pi)) eta^1.5
    F = np.where(hi, 4.0 / (3.0 * np.sqrt(np.pi)) * np.abs(eta) ** 1.5, F)
    dF = np.where(hi, 2.0 / np.sqrt(np.pi) * np.abs(eta) ** 0.5, dF)
    return F, dF


class Layer:
    def __init__(self, name, t_nm, eps_r, Ec, Ev, me, mh, P, Nd=0.0, Na=0.0, Ed=None,
                 me_dos=None):
        """Ec, Ev: band edges (eV) relative to a common reference, before band bending.
        Nd, Na: donor / acceptor densities (cm^-3); Ed: donor depth below E_c (eV),
        None for complete ionization. me_dos: density-of-states mass for the classical
        electron gas (default me)."""
        self.name, self.t, self.eps_r = name, t_nm, eps_r
        self.Ec, self.Ev, self.me, self.mh, self.P = Ec, Ev, me, mh, P
        self.Nd, self.Na = Nd * 1e6, Na * 1e6
        self.Ed = Ed
        self.me_dos = me if me_dos is None else me_dos


class Structure:
    def __init__(self, layers, phi_B, dz=0.05, T=300.0, extra_sheets=None,
                 quantum_window=None, n_states=40, classical_outside=False, holes=True):
        """extra_sheets: dict {interface index i (between layer i and i+1): sigma C/m^2}.
        quantum_window: (zmin, zmax) in nm where carriers are treated quantum mechanically;
        default: the whole structure. classical_outside: classical electrons outside the
        window. holes: include holes (quantum, in the window)."""
        self.layers, self.phi_B, self.T = layers, phi_B, T
        self.n_states = n_states
        self.classical_outside, self.holes = classical_outside, holes
        edges = np.cumsum([0.0] + [l.t for l in layers])
        self.edges = edges
        L = edges[-1]
        n = int(round(L / dz)) + 1
        self.z = np.linspace(0.0, L, n)
        self.dz = self.z[1] - self.z[0]
        # node properties: assign each node to the layer containing it (interfaces -> layer above)
        idx = np.clip(np.searchsorted(edges, self.z, side="right") - 1, 0, len(layers) - 1)
        self.idx = idx
        g = lambda f: np.array([f(layers[i]) for i in idx])
        self.eps = g(lambda l: l.eps_r) * EPS0
        self.Ec0, self.Ev0 = g(lambda l: l.Ec), g(lambda l: l.Ev)
        self.me, self.mh = g(lambda l: l.me), g(lambda l: l.mh)
        self.Nd = g(lambda l: l.Nd)
        self.Na = g(lambda l: l.Na)
        self.Ed = g(lambda l: np.nan if l.Ed is None else l.Ed)
        self.incomplete = ~np.isnan(self.Ed)
        mdos = g(lambda l: l.me_dos) * M0
        self.Nc = 2.0 * (mdos * KB * T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5      # m^-3
        # bound charge: sheets at interfaces, placed on the nearest node
        self.sheet = np.zeros(n)
        for i in range(len(layers) - 1):
            s = layers[i].P - layers[i + 1].P
            if extra_sheets and i in extra_sheets:
                s += extra_sheets[i]
            k = int(round(edges[i + 1] / self.dz))
            self.sheet[k] += s
        self.qw = quantum_window or (0.0, L)
        self.inq = (self.z >= self.qw[0] - 1e-9) & (self.z <= self.qw[1] + 1e-9)

    # ---------------------------------------------------------------- Poisson
    def _A(self):
        """Discrete -d/dz(eps d/dz) with zero field at the bottom (Neumann) and
        a Dirichlet row at the top (z in metres)."""
        dz = self.dz * 1e-9
        em = 0.5 * (self.eps[1:] + self.eps[:-1]) / dz ** 2
        n = len(self.z)
        A = np.zeros((n, n))
        k = np.arange(n - 1)
        A[k, k] += em
        A[k + 1, k + 1] += em
        A[k, k + 1] -= em
        A[k + 1, k] -= em
        # first row is a half cell: A[0,0] = em[0] already, rho is halved there
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        return A

    @staticmethod
    def _banded(J):
        """Tridiagonal bands of J for solve_banded (J is tridiagonal by construction)."""
        n = J.shape[0]
        ab = np.zeros((3, n))
        ab[0, 1:] = np.diag(J, 1)
        ab[1] = np.diag(J)
        ab[2, :-1] = np.diag(J, -1)
        return ab

    # ------------------------------------------------------------ carriers
    def _classical(self, phi):
        """Classical electrons outside the window and ionized donors (m^-3), with
        derivatives with respect to phi. Ec = Ec0 - phi, E_F = 0."""
        kT = KB * self.T
        n = np.zeros(len(self.z))
        dn = np.zeros(len(self.z))
        if self.classical_outside:
            out = ~self.inq
            eta = -(self.Ec0[out] - phi[out]) / kT
            F, dF = fermi_half(eta)
            n[out] = self.Nc[out] * F
            dn[out] = self.Nc[out] * dF / kT
        ndp = self.Nd.copy()
        dndp = np.zeros(len(self.z))
        m = self.incomplete & (self.Nd > 0)
        if m.any():
            x = (-(self.Ec0[m] - phi[m]) + self.Ed[m]) / kT       # (E_F - E_donor)/kT
            ex = np.exp(np.clip(x, -700, 700))
            ndp[m] = self.Nd[m] / (1 + 2 * ex)
            dndp[m] = -self.Nd[m] * 2 * ex / (1 + 2 * ex) ** 2 / kT
        return n, dn, ndp, dndp

    def _subbands(self, U, m_eff):
        """Lowest eigenstates of -hbar^2/2 d/dz (1/m) d/dz + U in the quantum window.

        U in eV (energy of the carrier, increasing upward for that carrier)."""
        sel = self.inq
        if sel.sum() < 3:
            return sel, np.zeros(0), np.zeros((sel.sum(), 0)), np.zeros(0)
        Us = U[sel]
        m = m_eff[sel] * M0
        dz = self.dz * 1e-9
        mh = 0.5 * (m[1:] + m[:-1])
        t = HBAR ** 2 / (2 * dz ** 2) / QE
        diag = Us + t * (np.r_[1 / mh, 1 / mh[-1]] + np.r_[1 / mh[0], 1 / mh])
        off = -t / mh
        k = min(self.n_states, len(diag) - 1)
        w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
        v = v / np.sqrt(dz)                                # normalized in m^-1/2
        md = (m[:, None] * v ** 2).sum(0) / (v ** 2).sum(0)  # state-averaged in-plane mass
        g = md / (np.pi * HBAR ** 2) * QE                  # 2D DOS incl. spin, per eV per m^2
        return sel, w, v ** 2, g

    def _carriers(self, states, dphi, sign):
        """Density (m^-3) and its derivative with respect to phi (m^-3 / V).

        Electron energies move by -dphi, hole energies by +dphi."""
        kT = KB * self.T
        sel, w, v2, g = states
        d = dphi[sel]
        n = np.zeros(len(self.z))
        dn = np.zeros(len(self.z))
        for j in range(len(w)):
            x = (w[j] - sign * d) / kT
            n[sel] += g[j] * kT * np.logaddexp(0.0, -x) * v2[:, j]
            dn[sel] += g[j] * (0.5 * (1 - np.tanh(0.5 * x))) * v2[:, j] * sign
        return n, dn

    def solve(self, tol=1e-6, max_outer=200, verbose=False):
        """Self-consistent solution. Returns band edges, densities and sheet densities."""
        dzm = self.dz * 1e-9
        sheet_rho = self.sheet / dzm
        phi_top = self.Ec0[-1] - self.phi_B          # Ec(L) = phi_B with Ec = Ec0 - phi
        A = self._A()
        half = np.ones(len(self.z))
        half[0] = 0.5
        N = len(self.z)
        idx = np.arange(N - 1)

        def fixed(phi):
            _, _, ndp, dndp = self._classical(phi)
            return sheet_rho + QE * (ndp - self.Na), QE * dndp

        rho0, _ = fixed(np.zeros(N))
        b = rho0 * half
        b[-1] = phi_top
        phi = solve_banded((1, 1), self._banded(A), b)
        hist = []
        zero_states = (self.inq, np.zeros(0), np.zeros((self.inq.sum(), 0)), np.zeros(0))
        for it in range(max_outer):
            se = self._subbands(self.Ec0 - phi, self.me)
            sh = self._subbands(-(self.Ev0 - phi), self.mh) if self.holes else zero_states
            phi_k = phi.copy()
            for inner in range(200):
                dphi = phi - phi_k
                n, dn = self._carriers(se, dphi, +1)
                p, dp = self._carriers(sh, dphi, -1)
                nc, dnc, _, _ = self._classical(phi)
                rf, drf = fixed(phi)
                rho = (rf + QE * (p - n - nc)) * half
                R = A @ phi - rho
                R[-1] = phi[-1] - phi_top
                J = A.copy()
                jd = (QE * (dn + dnc - dp) - drf) * half
                J[idx, idx] += jd[:-1]
                step = solve_banded((1, 1), self._banded(J), -R)
                step = np.clip(step, -0.2, 0.2)
                phi = phi + step
                if np.max(np.abs(step)) < 1e-9:
                    break
            d = np.max(np.abs(phi - phi_k))
            hist.append(d)
            if verbose:
                print(it, d)
            if d < tol:
                break
        se = self._subbands(self.Ec0 - phi, self.me)
        sh = self._subbands(-(self.Ev0 - phi), self.mh) if self.holes else zero_states
        zero = np.zeros(len(self.z))
        nq, _ = self._carriers(se, zero, +1)
        p, _ = self._carriers(sh, zero, -1)
        nc, _, ndp, _ = self._classical(phi)
        n = nq + nc
        return {"z": self.z, "Ec": self.Ec0 - phi, "Ev": self.Ev0 - phi, "n": n, "p": p,
                "n_quantum": nq, "n_classical": nc, "Nd_ionized": ndp,
                "ns_cm2": n.sum() * dzm * 1e-4, "ps_cm2": p.sum() * dzm * 1e-4,
                "Ee": se[1], "Eh": sh[1],
                "converged": bool(hist and hist[-1] < tol), "iterations": len(hist),
                "last_change": hist[-1] if hist else 0.0}
