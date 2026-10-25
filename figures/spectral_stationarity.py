"""
Spectral (Chebyshev) stationarity experiment -- Panel (b) of Fig. 8
===================================================================
Panel (a): finite-difference radial operator (reproduces the published
           Fig. 8 of bernoulli_eigenstate.py, N_r=128).
Panel (b): spectral (Chebyshev) radial operator + FFT angular on the same
           mapped polar grid (N_r=256), with the boundary rows held at the
           stationary state at every sub-step.

Purpose: demonstrate that with a fully spectral radial operator the chain
rule holds to machine precision in the interior, so the ANALYTIC Bernoulli
potential V=-|v|^2/2 becomes an exact zero-energy eigenstate and the
velocity error pins at the t=0 decoding floor, coinciding with the exact
eigenstate V_ex -- the claim of Sec. 6.2 that "the QFT spectral Laplacian
removes it entirely".
"""
import os, sys, time
import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import eig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.colors import LogNorm

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

from unbounded_mapping import (
    MappedPolarGrid, cylinder_potential, cylinder_velocity_analytic,
    R_CYL, U_INF, HBAR, MASS,
)
from mapped_bernoulli import (
    bernoulli_potential as bernoulli_potential_fd,
    eigenstate_residual, RadialSpectralPropagator,
    evolve_bernoulli, evolve_free,
)

DT_A = 1e-4          # Panel (a) time step (matches published figure)
DT_B = 2.5e-5         # Panel (b) time step (splitting-error floor kept flat)
T_FIG = np.linspace(0.001, 0.04, 13)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '_spectral_run.log')

def log(msg):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

# ----------------------------------------------------------------------
# Chebyshev spectral radial operator on the mapped polar grid
# ----------------------------------------------------------------------
def chebyshev_d1(xi):
    """Chebyshev differentiation matrix D1 on nodes xi (ascending)."""
    N = len(xi)
    xi = np.asarray(xi, dtype=float)
    w = (-1.0) ** np.arange(N)
    w[0] *= 0.5
    w[-1] *= 0.5
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = w[j] / w[i] / (xi[i] - xi[j])
        D[i, i] = -np.sum(D[i, :])
    return D

def cheb_to_uniform_matrix(n_c, xi_c, xi_u):
    """Barycentric interpolation Chebyshev nodes -> arbitrary nodes."""
    N = n_c
    w = (-1.0) ** np.arange(N)
    w[0] *= 0.5
    w[-1] *= 0.5
    Q = np.zeros((len(xi_u), N))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i, x in enumerate(xi_u):
            d = x - xi_c
            denom = np.sum(w / d)
            if np.isinf(denom):
                j = int(np.argmin(np.abs(d)))
                Q[i, j] = 1.0
            else:
                Q[i, :] = (w / d) / denom
    return Q

class SpectralPolarOperator:
    """Chebyshev radial + FFT angular on the mapped polar grid.

    Boundary rows are frozen (zero rows in the radial operator) and the
    boundary is pinned to the stationary state at every sub-step.
    """

    def __init__(self, ugrid, N):
        self.ug = ugrid
        self.Nr, self.Nt = N, 2 * N
        self.L_map = ugrid.L_map
        rt_min, rt_max = ugrid.r_tilde_min, ugrid.r_tilde_max
        k = np.arange(N)
        self.xi = -np.cos(np.pi * k / (N - 1))          # ascending in [-1,1]
        self.rt = 0.5 * (rt_min + rt_max) + 0.5 * (rt_max - rt_min) * self.xi
        self.r = self.L_map * self.rt / (1 - self.rt)
        om = 1.0 - self.rt
        self.metric_d1 = om ** 2 / self.L_map
        self.metric_d2 = om ** 4 / self.L_map ** 2
        self.metric_extra = -2.0 * om ** 3 / self.L_map ** 2
        self.theta = ugrid.theta
        self.RR, self.TT = np.meshgrid(self.r, self.theta, indexing='ij')

        D1 = chebyshev_d1(self.xi)
        D2 = D1 @ D1
        s = 2.0 / (rt_max - rt_min)
        c = self.metric_extra + self.metric_d1 / self.r
        Lr = np.diag(self.metric_d2) @ (s * s * D2) + np.diag(c) @ (s * D1)
        Lr[0, :] = 0.0
        Lr[-1, :] = 0.0
        self.Lr = Lr

        xi_unif = (2.0 * ugrid.r_tilde - rt_min - rt_max) / (rt_max - rt_min)
        self.Q = cheb_to_uniform_matrix(N, self.xi, xi_unif)

        evals, evecs = eig(Lr)
        self.evals = evals
        self.evecs = evecs
        self.inv_evecs = np.linalg.inv(evecs)
        self.ksq = (np.fft.fftfreq(self.Nt, d=1.0) * self.Nt) ** 2
        self._prop_cache = {}

    # -- operators -----------------------------------------------------
    def encode(self):
        phi = cylinder_potential(self.RR, self.TT, 0.0)
        return np.exp(1j * phi / HBAR)

    def bernoulli_potential(self):
        vr, vt = cylinder_velocity_analytic(self.RR, self.TT, 0.0)
        return -(vr ** 2 + vt ** 2) / 2.0

    def spectral_laplacian(self, psi):
        """laplacian = Lr psi + (1/r^2) d^2psi/dtheta^2."""
        lap = self.Lr @ psi
        lap = lap - ifft(self.ksq[None, :] * fft(psi, axis=1),
                         axis=1) / (self.RR ** 2 + 1e-12)
        return lap

    def apply_radial(self, psi, dt):
        """psi_new = exp(i dt Lr / 2) psi (frozen boundary rows).
        The full propagator matrix is cached per dt for speed."""
        E = self._prop_cache.get(float(dt))
        if E is None:
            E = (self.evecs
                 @ (np.exp(1j * dt / 2.0 * self.evals)[:, None] * self.inv_evecs))
            self._prop_cache[float(dt)] = E
        return E @ psi

    def to_uniform(self, psi_c):
        return self.Q @ psi_c

    # -- evolution (boundary pinned to the stationary state) -----------
    def evolve(self, psi, V, T, dt):
        n = max(1, int(round(T / dt)))
        dta = T / n
        expV = np.exp(-1j * V * dta / 2.0)
        angf = np.exp(-1j * dta / 2.0 * self.ksq[None, :] / (self.RR ** 2 + 1e-12))
        ps = psi.copy()
        for _ in range(n):
            ps = ps * expV
            ps[0, :], ps[-1, :] = psi[0, :], psi[-1, :]
            ps = self.apply_radial(ps, dta / 2.0)
            ps = ifft(angf * fft(ps, axis=1), axis=1)
            ps[0, :], ps[-1, :] = psi[0, :], psi[-1, :]
            ps = self.apply_radial(ps, dta / 2.0)
            ps = ps * expV
            ps[0, :], ps[-1, :] = psi[0, :], psi[-1, :]
        return ps

    def evolve_free(self, psi, T, dt):
        n = max(1, int(round(T / dt)))
        dta = T / n
        angf = np.exp(-1j * dta / 2.0 * self.ksq[None, :] / (self.RR ** 2 + 1e-12))
        ps = psi.copy()
        for _ in range(n):
            ps[0, :], ps[-1, :] = psi[0, :], psi[-1, :]
            ps = self.apply_radial(ps, dta / 2.0)
            ps = ifft(angf * fft(ps, axis=1), axis=1)
            ps[0, :], ps[-1, :] = psi[0, :], psi[-1, :]
            ps = self.apply_radial(ps, dta / 2.0)
        return ps

def annulus_error(ugrid, psi_u):
    vr, vt, _ = ugrid.decode(psi_u)
    return ugrid.to_cartesian_bulk(vr, vt, 0.0)['vel_err']

# ----------------------------------------------------------------------
def panel_a_fd():
    """Reproduce the published single-panel curves (finite-difference radial)."""
    grid = MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
    rsp = RadialSpectralPropagator(grid)
    psi0 = grid.encode()
    V = bernoulli_potential_fd(grid)
    Hf, _ = eigenstate_residual(grid, psi0, V)
    Vex = -(Hf / psi0)
    ef, eb, ex = [], [], []
    for T in T_FIG:
        ef.append(annulus_error(grid, evolve_free(grid, rsp, psi0.copy(), T, DT_A)))
        eb.append(annulus_error(grid, evolve_bernoulli(grid, rsp, psi0.copy(), V, T, DT_A)))
        ex.append(annulus_error(grid, evolve_bernoulli(grid, rsp, psi0.copy(), Vex, T, DT_A)))
        log(f"  (a) T={T:5.3f}: free={ef[-1]:.4f} analytic={eb[-1]:.4f} exact={ex[-1]:.4f}")
    return np.array(ef), np.array(eb), np.array(ex)

def panel_b_spectral():
    """Spectral (Chebyshev) radial operator, boundary pinned at psi0."""
    ugrid = MappedPolarGrid(Nr=256, Nt=512, L_map=2.5, R_max_phys=50.0)
    op = SpectralPolarOperator(ugrid, 256)
    psi0 = op.encode()
    V = op.bernoulli_potential()
    lap = op.spectral_laplacian(psi0)
    Vex = -(-0.5 * lap / psi0)
    ef, eb, ex = [], [], []
    for T in T_FIG:
        ef.append(annulus_error(ugrid, op.to_uniform(op.evolve_free(psi0.copy(), T, DT_B))))
        eb.append(annulus_error(ugrid, op.to_uniform(op.evolve(psi0.copy(), V, T, DT_B))))
        ex.append(annulus_error(ugrid, op.to_uniform(op.evolve(psi0.copy(), Vex, T, DT_B))))
        log(f"  (b) T={T:5.3f}: free={ef[-1]:.4f} analytic={eb[-1]:.4f} exact={ex[-1]:.4f}")
    return np.array(ef), np.array(eb), np.array(ex)

def panel_c_maps():
    """Pointwise velocity-deviation fields at T_FIG[-1] (FD radial, N_r=128).

    Returns a dict with the three error fields on the polar grid plus the
    physical-plane node coordinates for rendering.
    """
    grid = MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
    rsp = RadialSpectralPropagator(grid)
    psi0 = grid.encode()
    V = bernoulli_potential_fd(grid)
    Hf, _ = eigenstate_residual(grid, psi0, V)
    Vex = -(Hf / psi0)
    T_END = T_FIG[-1]
    res_a = grid.to_cartesian_bulk(*grid.decode(psi0.copy())[:2], 0.0)
    fields = {'XX': res_a['XX'], 'YY': res_a['YY']}
    for name, psi in (
            ('free', evolve_free(grid, rsp, psi0.copy(), T_END, DT_A)),
            ('analytic', evolve_bernoulli(grid, rsp, psi0.copy(), V, T_END, DT_A)),
            ('exact', evolve_bernoulli(grid, rsp, psi0.copy(), Vex, T_END, DT_A))):
        vr, vt, _ = grid.decode(psi)
        res = grid.to_cartesian_bulk(vr, vt, 0.0)
        err = np.abs(res['vmag'] - res['vmag_ref']) / U_INF
        err = np.where((grid.RR <= R_CYL * 1.02) | (grid.RR > 4.5), np.nan, err)
        fields[name] = err
        log(f"  (c) {name}: annulus RMS = {res['vel_err']:.4f} at T={T_END}")
    return fields

def make_map_figure(fields):
    """One-row strip of the three pointwise error fields, shared log scale."""
    C_GREY = '#636e72'
    plt.rcParams.update({'font.size': 12})
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')
    order = [('free', 'Free'), ('analytic', 'Analytic $V$'),
             ('exact', 'Exact $V_\\mathrm{ex}$')]
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.75), constrained_layout=True)
    vmin, vmax = 2.5e-4, 1.5e-1
    for ax, (key, title) in zip(axes, order):
        XX, YY, err = fields['XX'], fields['YY'], fields[key]
        # Close the angular mesh: theta spans [0, 2*pi) without a wrap
        # column, so gouraud quads would leave a seam along the theta=0 ray.
        XX = np.hstack([XX, XX[:, :1]])
        YY = np.hstack([YY, YY[:, :1]])
        err = np.hstack([err, err[:, :1]])
        im = ax.pcolormesh(XX, YY, err, cmap='viridis', shading='gouraud',
                           norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
        ax.add_patch(plt.Circle((0, 0), R_CYL, fc='0.93', ec='0.35', lw=0.7,
                                zorder=3))
        ax.add_patch(plt.Circle((0, 0), 1.5, fill=False, ec='white', ls='--',
                                lw=0.8, zorder=3))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, pad=2.5)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02, aspect=16)
    cb.ax.tick_params(labelsize=8, pad=1.5)
    cb.set_ticks([1e-3, 1e-2, 1e-1])
    cb.outline.set_linewidth(0.5)
    out_c = os.path.join(outdir, 'bernoulli_stationarity_c.pdf')
    # 650 dpi, deliberately higher than the 400 dpi of panels (a)/(b): this
    # strip has a 2.6:1 aspect ratio and is placed at half text width, so its
    # effective resolution comes out ~1.5x lower than the two squarer panels
    # for the same source dpi.  650 puts it at ~850 dpi on the page, in line
    # with the other figures.
    fig.savefig(out_c, dpi=650)
    plt.close(fig)
    print(f"  -> saved {out_c}")
    return out_c

def make_figure(pa, pb):
    C_FREE = '#d63031'
    C_AN = '#0984e3'
    C_EX = '#00b894'
    C_GREY = '#636e72'
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.fontsize': 11})
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')

    def draw_panel(p, ylabel, offset):
        fig, ax = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
        ax.set_ylim(1e-4, 2e-1)
        ax.axhspan(2.5e-4, 3.5e-4, color=C_GREY, alpha=0.13, lw=0, zorder=0)
        ax.axhline(3e-4, color=C_GREY, ls=':', lw=1.4, zorder=1)
        ax.grid(True, which='both', alpha=0.25, lw=0.6)
        ax.set_xlabel('Evolution time $T$ [s]')
        if ylabel:
            ax.set_ylabel(r'Velocity error $\|\mathbf{v}-\mathbf{v}_\mathrm{ref}\|_\mathrm{rel}$ (annulus)')
        kw = dict(markeredgecolor='white', markeredgewidth=0.6)
        ax.semilogy(T_FIG, p[0], 'o-', color=C_FREE, lw=1.6, ms=4,
                    label='Free-particle evolution $e^{-iH_{\\rm free}t}$', **kw)
        # Panel (b): analytic and exact coincide to roundoff, so the
        # analytic curve is offset by a small dT to make it visible.
        dT = 0.0012 if offset else 0.0
        ax.semilogy(T_FIG + dT, p[1], 's-' if offset else 's--',
                    color=C_AN, lw=1.6, ms=4.5,
                    label='$H_B$, analytic $V=-|\\nabla\\Phi|^2/2$', **kw)
        ax.semilogy(T_FIG, p[2], '^-', color=C_EX, lw=2.0, ms=5.5,
                    label='$H_B$, exact eigenstate', **kw)
        return fig, ax

    fig_a, ax_a = draw_panel(pa, ylabel=True, offset=False)
    ax_a.legend(loc='center right', framealpha=0.95, edgecolor='0.6')
    out_a = os.path.join(outdir, 'bernoulli_stationarity_a.pdf')
    fig_a.savefig(out_a, dpi=400)
    plt.close(fig_a)
    print(f"  -> saved {out_a}")

    fig_b, ax_b = draw_panel(pb, ylabel=True, offset=True)
    ax_b.legend(loc='upper left', framealpha=0.95, edgecolor='0.6')
    out_b = os.path.join(outdir, 'bernoulli_stationarity_b.pdf')
    fig_b.savefig(out_b, dpi=400)
    plt.close(fig_b)
    print(f"  -> saved {out_b}")
    return out_a

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'data', 'bernoulli_stationarity_spectral.npz')

def save_data(**kwargs):
    kwargs.setdefault('T', np.array(T_FIG))
    try:
        old = np.load(DATA_PATH)
        for k in old.files:
            if k not in kwargs:
                kwargs[k] = old[k]
    except Exception:
        pass
    np.savez(DATA_PATH, **kwargs)
    log(f"  -> saved data {DATA_PATH}")

def load_data():
    d = np.load(DATA_PATH)
    return (d.get('panel_a'), d.get('panel_b'))

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    log("=" * 60)
    log(f" Fig. 8 two-panel experiment [mode={mode}]")
    log("=" * 60)
    pa, pb = None, None

    if mode in ('all', 'panel_a'):
        log("[Panel (a)] finite-difference radial operator (N_r=128)")
        pa = panel_a_fd()
        log(f"[Panel (a)] done in {time.time()-t0:.0f}s")
        save_data(pa=pa)

    if mode in ('all', 'panel_b'):
        log("[Panel (b)] spectral (Chebyshev) radial operator (N_r=256, pinned)")
        pb = panel_b_spectral()
        log(f"[Panel (b)] done in {time.time()-t0:.0f}s")
        save_data(pb=pb)

    if mode in ('all', 'maps'):
        log("[Panel (c)] pointwise error fields at T=0.04 (FD radial, N_r=128)")
        fields = panel_c_maps()
        save_data(map_free=fields['free'], map_analytic=fields['analytic'],
                  map_exact=fields['exact'], map_XX=fields['XX'],
                  map_YY=fields['YY'])
        make_map_figure(fields)
        log(f"[Panel (c)] done in {time.time()-t0:.0f}s")

    if mode in ('figure', 'all'):
        if pa is None or pb is None:
            saved = load_data()
            pa = pa if pa is not None else saved[0]
            pb = pb if pb is not None else saved[1]
        if pa is None or pb is None:
            raise RuntimeError("need panel data; run panel_a / panel_b first")
        out = make_figure(pa, pb)
        if mode == 'figure':
            try:
                d = np.load(DATA_PATH)
                if all(k in d.files for k in
                       ('map_free', 'map_analytic', 'map_exact',
                        'map_XX', 'map_YY')):
                    make_map_figure({k[4:]: d[k] for k in
                                     ('map_free', 'map_analytic', 'map_exact',
                                      'map_XX', 'map_YY')})
            except FileNotFoundError:
                pass
        log(f"[done in {time.time()-t0:.0f}s] -> {out}")

if __name__ == "__main__":
    main()
