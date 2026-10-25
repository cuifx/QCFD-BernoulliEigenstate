# -*- coding: utf-8 -*-
"""Figures for the non-axisymmetric three-dimensional verification.

Two panels, mirroring the structure of the axisymmetric figure
(fig/3d_stationarity_*.pdf): one versus evolution time, one versus resolution.

  fig/3d_nonaxisym_a.pdf   Triaxial ellipsoid (1.0, 0.8, 0.6): panel solve
                           against the closed-form ellipsoidal-harmonic
                           potential.  This is the independent baseline -- the
                           wing case has neither a conformal map nor a closed
                           form, so its potential is necessarily the panel solve
                           itself and its verification is a self-consistency
                           check of the encode-evolve-decode chain.
  fig/3d_nonaxisym_b.pdf   AGARD-B calibration model, 96^3 box, L = 6, same
                           protocol as Sec. 6.2: free propagator, H_B with the
                           analytic Bernoulli potential, H_B with the potential
                           exact at each grid point.

The (a)/(b) order above matches \\caption in Section 6.3, which the main()
call sequence below also follows; an earlier revision of this docstring had
the two panels the other way round.

Geometry conventions: the AGARD-B panel solve is read from
code/data/agardb_panels.npz (11904 panels, the set the geometry audit and the
24-qubit Phi build both used); AXS is the axial half-length of the agardb
parametrisation, NOT the winged-body W.AX -- using the latter mis-places the
fluid mask on an 8.5 D long body.
"""
import os
import sys

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import winged_body_3d_probe as W                                    # noqa: E402
import ellipsoid_exact_vs_panel as E                                # noqa: E402

W.BODY_STYLE = 'agardb'
W.NW = 2
W.FIN_AZ = 2.0 * np.pi / W.NW
AXS = W.AB_AX

import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
# match the paper's other panels: Times body text, Computer Modern math,
# vector PDF with TrueType-embedded fonts.  The sizes are the ones wanted ON THE
# PAGE, because the panels are authored at their placed size -- see PANEL_FS.
#
# Calibrated against the neighbouring panel rather than chosen by eye.  The
# three-dimensional panels of Fig. 9 are authored at 5.4 in and placed at
# \panelw = 0.35 of the 481.5 pt column, a scale of 0.433, so their 11 pt ticks
# land at 4.8 pt and their 12 pt axis labels at 5.2 pt on the page.  These are
# the numbers to match: an earlier revision of this file used 7 and 8.5 pt,
# which on the page were 46% and 63% larger than Fig. 9.
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 5.2,
    'axes.labelsize': 5.2,
    'axes.titlesize': 5.2,
    'legend.fontsize': 4.8,
    'xtick.labelsize': 4.8,
    'ytick.labelsize': 4.8,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIGDIR = 'fig'
# Authored at the size it is placed.  The figure carries \panelw=0.35 of a
# 481.5 pt column, i.e. 168.5 pt = 2.34 in per panel, so the scale is 1 and the
# point sizes above are the sizes on the page.  Authoring at 5.4 in and letting
# LaTeX shrink it by 0.43 is what put the tick labels near 3 pt and the
# fitted-order annotation at 3.5 pt on the page, on top of the frame.
PANEL_FS = (2.34, 1.77)
TICK_FS = 4.8
U = 1.0
NPZ = 'code/data/nonaxisym_3d.npz'


# ============================================================ ellipsoid (A)
def ellipsoid_study(res_list=((12, 24), (20, 40), (32, 64), (48, 96))):
    """|Phi_panel - Phi_exact| at a near and a far probe, versus resolution."""
    near = np.array([[1.25, 0.0, 0.0]])
    far = np.array([[4.0, 0.0, 0.0]])
    exact_n = float(E.phi_exact(*near[0]))
    exact_f = float(E.phi_exact(*far[0]))
    rows = []
    for nu, nv in res_list:
        cen, nrm, area = E.make_panels(nu, nv)
        sig, _, _ = W.solve_sources(cen, nrm, area)
        got = W.potential(np.vstack([near, far]), cen, area, sig)
        dn, df = abs(got[0] - exact_n), abs(got[1] - exact_f)
        rows.append((nu, nv, len(cen), dn, df,
                     dn / abs(exact_n), df / abs(exact_f)))
        print('  %3dx%-3d panels %5d   |dPhi| near %.3e (rel %.3e)   far %.3e (rel %.3e)'
              % (nu, nv, len(cen), dn, dn / abs(exact_n), df, df / abs(exact_f)))
    return np.array(rows), exact_n, exact_f


# ============================================================ AGARD-B (B)
def agardb_stationarity(N=96, L=6.0, T_list=(0.01, 0.02, 0.03, 0.04), dt=1e-3,
                        panel_path='code/data/agardb_panels.npz'):
    """Encode-evolve-decode protocol on the AGARD-B panel potential.

    ``panel_path`` is a parameter so that an alternative discretisation of the
    same body -- for instance the re-meshed tail of the route-A study -- can be
    run through the identical protocol and compared number for number.
    """
    h = 2.0 * L / (N - 1)
    gg = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(gg, gg, gg, indexing='ij')

    def fd_gradient(f):
        gx = np.zeros_like(f); gy = np.zeros_like(f); gz = np.zeros_like(f)
        gx[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)
        gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * h)
        gz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * h)
        return gx, gy, gz

    def fd_laplacian(f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1, 1:-1] = (
            f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
            - 6.0 * f[1:-1, 1:-1, 1:-1]) / (h * h)
        return lap

    _k = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
    _KX, _KY, _KZ = np.meshgrid(_k, _k, _k, indexing='ij')
    _K2 = _KX ** 2 + _KY ** 2 + _KZ ** 2

    def kinetic_spectral(f):
        return ifftn(0.5 * _K2 * fftn(f))          # -lap_spec(f)/2

    def spectral_gradient(f):
        return (ifftn(1j * _KX * fftn(f)), ifftn(1j * _KY * fftn(f)),
                ifftn(1j * _KZ * fftn(f)))

    def window3d(LL, tr=0.12):
        w = LL * tr
        win = np.ones_like(X)
        for C in (X, Y, Z):
            wc = np.ones_like(X)
            wc[C < -LL + w] = 0.5 * (1 - np.cos(np.pi * (C[C < -LL + w] + LL) / w))
            wc[C > LL - w] = 0.5 * (1 - np.cos(np.pi * (LL - C[C > LL - w]) / w))
            win *= wc
        return win

    d = np.load(panel_path)
    cen, nrm, area, sig = d['cen'], d['nrm'], d['area'], d['sig']
    print('  panels %d   Q = %+.3e   (%s)'
          % (len(cen), (sig * area).sum(), panel_path))
    PHI = W.potential(np.stack([X, Y, Z], -1), cen, area, sig)

    PHIp = np.arccos(np.clip(X / AXS, -1.0, 1.0))
    THp = np.arctan2(Z, Y)
    RPp = np.hypot(Y, Z)
    fluid = RPp > W.radius(PHIp, THp)
    probe = binary_erosion(fluid, iterations=2) & (np.sqrt(X ** 2 + Y ** 2 + Z ** 2) < 3.6)
    print('  probe points %d of %d' % (probe.sum(), PHI.size))

    _pp = PHI - U * X
    _gx, _gy, _gz = spectral_gradient(_pp)
    VRX, VRY, VRZ = (U + _gx).real, _gy.real, _gz.real
    V_ANA = -0.5 * (VRX ** 2 + VRY ** 2 + VRZ ** 2)
    WIND = window3d(L)
    psi0 = WIND * np.exp(1j * PHI)

    def decode_velocity(psi):
        gx, gy, gz = fd_gradient(psi)
        rho = np.abs(psi) ** 2 + 1e-12
        return (np.imag(np.conj(psi) * gx) / rho,
                np.imag(np.conj(psi) * gy) / rho,
                np.imag(np.conj(psi) * gz) / rho)

    def vel_error(psi):
        vx, vy, vz = decode_velocity(psi)
        dv2 = (vx - VRX) ** 2 + (vy - VRY) ** 2 + (vz - VRZ) ** 2
        return np.sqrt(np.mean(dv2[probe])) / U

    def rms_over(f, mask):
        return np.sqrt(np.mean(np.abs(f)[mask] ** 2))

    Hf = kinetic_spectral(psi0)
    norm0 = rms_over(psi0, probe)
    V_ex = np.zeros_like(psi0)
    den = psi0.copy()
    small = np.abs(den) < 1e-3 * np.max(np.abs(den))
    den[small] = 1.0
    V_ex[~small] = -(Hf[~small] / den[~small])
    lap_fd = fd_laplacian(PHI)
    res = dict(rms_lap=rms_over(lap_fd, probe),
               r_ana=rms_over(Hf + V_ANA * psi0, probe) / norm0,
               r_ex=rms_over(Hf + V_ex * psi0, probe) / norm0,
               r_im=rms_over(np.imag(V_ex) + 0.5 * lap_fd, probe) / rms_over(PHI, probe),
               err_t0=vel_error(psi0))
    print('  RMS(lap_h Phi)            = %.4e' % res['rms_lap'])
    print('  analytic V residual       = %.4e' % res['r_ana'])
    print('  exact V residual          = %.4e' % res['r_ex'])
    print('  Im(V_ex) vs -lap_h/2      = %.4e' % res['r_im'])
    print('  t=0 decoding floor        = %.4e' % res['err_t0'])

    def kinetic_step(psi, dtt):
        return ifftn(fftn(psi) * np.exp(-1j * 0.5 * _K2 * dtt))

    def run(V):
        psi = psi0.copy()
        t_cur, out = 0.0, []
        for T in T_list:
            n = max(1, int(round((T - t_cur) / dt)))
            dtt = (T - t_cur) / n
            eV = np.exp(-1j * V * dtt / 2)
            for _ in range(n):
                psi = psi * eV
                psi = kinetic_step(psi, dtt)
                psi = psi * eV
            t_cur = T
            out.append(vel_error(psi))
        return np.array(out)

    prot = {}
    for tag, V in (('free', np.zeros_like(PHI)), ('analytic V', V_ANA),
                   ('exact V', V_ex)):
        prot[tag] = run(V)
        print('  %-11s ' % tag + '  '.join('T=%.2f: %.3e' % (t, e)
                                           for t, e in zip(T_list, prot[tag])))
    return np.array(T_list), prot, res, (gg, PHI)


def check_text_inside(fig, ax, path, tol=2.0):
    """Fail loudly if a text sits on the frame or is clipped by the canvas.

    The fitted-order annotation used to be placed in data coordinates near the
    top of the y range, where its box crossed the frame and the frame clipped
    the exponent.  That is invisible in a thumbnail and obvious to a reader, so
    it is asserted rather than eyeballed.

    Two filters keep the assertion honest rather than noisy.  Matplotlib creates
    tick-label Text objects for ticks *outside* the axis limits and never draws
    them, so those phantoms are skipped; and only texts whose box intersects the
    axes are checked against the canvas, since anything far outside it is by
    definition not drawn.  Texts anchored with ax.text() are checked against the
    frame itself, unconditionally.
    """
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    ab = ax.get_window_extent(renderer=r)
    bad = []
    for t in ax.texts:
        bb = t.get_window_extent(renderer=r)
        if bb.x0 < ab.x0 or bb.x1 > ab.x1 or bb.y0 < ab.y0 or bb.y1 > ab.y1:
            bad.append(('crosses the frame', t.get_text()))
    checked = 0
    for t in fig.findobj(mtext.Text):
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=r)
        if bb.x1 < ab.x0 or bb.x0 > ab.x1 or bb.y1 < ab.y0 or bb.y0 > ab.y1:
            continue                       # phantom tick outside the limits
        checked += 1
        if (bb.x0 < -tol or bb.x1 > W + tol or bb.y0 < -tol or bb.y1 > H + tol):
            bad.append(('clipped by the canvas', t.get_text()))
    print('  [%s] %d anchored texts, %d in-plot texts, %d problems'
          % (os.path.basename(path), len(ax.texts), checked, len(bad)))
    for b in bad[:6]:
        print('    %s: %r' % b)
    return not bad


# ------------------------------------------------ ellipsoid (A), surface metric
# Two probes on the axis carry no information about the tangential structure of
# the solve, and an axial stream leaves that structure close to a body of
# revolution.  The sample is therefore the whole wetted surface, and a second
# stream oblique to all three axes removes the remaining degeneracy.  The
# oblique potential is the superposition of the three principal-direction
# harmonics, so it costs one elliptic integral per axis and reduces to the
# published axial form when two components vanish.
_UG, _WG = np.polynomial.legendre.leggauss(400)
_UG, _WG = 0.5 * (_UG + 1.0), 0.5 * _WG                  # nodes / weights on [0,1]
_AX = (E.A_, E.B_, E.C_)


def _lam_of(P):
    """Largest root of sum_i p_i^2/(a_i^2+L) = 1, for points outside."""
    P = np.asarray(P, dtype=float)
    lo = np.zeros(len(P))
    hi = (P ** 2).sum(1) + 4.0 * max(_AX) ** 2
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        f = (P[:, 0] ** 2 / (_AX[0] ** 2 + mid)
             + P[:, 1] ** 2 / (_AX[1] ** 2 + mid)
             + P[:, 2] ** 2 / (_AX[2] ** 2 + mid) - 1.0)
        neg = f < 0.0
        hi = np.where(neg, mid, hi)
        lo = np.where(neg, lo, mid)
    return 0.5 * (lo + hi)


def _I_axis(lam, i):
    """int_lam^inf ds/((a_i^2+s) D(s)) with D = sqrt(prod_j (a_j^2+s))."""
    s = np.asarray(lam, dtype=float)[:, None] + (_UG / (1.0 - _UG))[None, :]
    D = np.sqrt((_AX[0] ** 2 + s) * (_AX[1] ** 2 + s) * (_AX[2] ** 2 + s))
    return (1.0 / ((_AX[i] ** 2 + s) * D) / (1.0 - _UG)[None, :] ** 2) @ _WG


def oblique_coeffs():
    """Depolarization factors n_i (sum = 1) and c_i = abc/[2(1-n_i)]."""
    vol = E.A_ * E.B_ * E.C_
    n = np.array([0.5 * vol * _I_axis(np.zeros(1), i)[0] for i in range(3)])
    return n, vol / (2.0 * (1.0 - n))


def phi_oblique(P, Uvec):
    """Closed-form exterior potential of the triaxial ellipsoid, any direction."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    _n, c = oblique_coeffs()
    lam = _lam_of(P)
    out = P @ np.asarray(Uvec, dtype=float)
    for i in range(3):
        out = out + c[i] * Uvec[i] * P[:, i] * _I_axis(lam, i)
    return out


def oblique_stream(pitch=20.0, yaw=30.0):
    """Unit free stream oblique to all three axes."""
    a, b = np.deg2rad(pitch), np.deg2rad(yaw)
    U = np.array([np.cos(a) * np.cos(b), np.sin(b), np.cos(a) * np.sin(b)])
    return U / np.linalg.norm(U)


def ellipsoid_surface_study(nu_list=(16, 24, 32, 40, 48), NS=30, NT=60,
                            fac=1.05):
    """Relative L2 error of the panel potential over the wetted surface.

    The sample is a fixed angular grid at fac times the local radius, so the
    same physical points are used at every resolution.  The axial near-body
    probe of the earlier form is carried along as a continuity check on the
    published number.
    """
    n, c = oblique_coeffs()
    print('  depolarisation n = (%.6f, %.6f, %.6f), sum %.9f'
          % (n[0], n[1], n[2], n.sum()))
    print('  c_i = (%.6f, %.6f, %.6f)   axial c_x = Uabc/[2(1-n_x)] = %.5f'
          % (c[0], c[1], c[2], c[0]))
    s = np.linspace(0.12, np.pi - 0.12, NS)
    t = np.linspace(0.0, 2.0 * np.pi, NT, endpoint=False)
    S, T = np.meshgrid(s, t, indexing='ij')
    dv = np.stack([np.sin(S) * np.cos(T), np.sin(S) * np.sin(T), np.cos(S)], -1)
    rad = 1.0 / np.sqrt((dv[..., 0] / E.A_) ** 2 + (dv[..., 1] / E.B_) ** 2
                        + (dv[..., 2] / E.C_) ** 2)
    Pf = ((fac * rad)[..., None] * dv).reshape(-1, 3)
    Uob = oblique_stream()
    print('  oblique stream direction (%.4f, %.4f, %.4f)'
          % (Uob[0], Uob[1], Uob[2]))
    ex_ax = np.array([E.phi_exact(*q) for q in Pf])
    ex_ob = phi_oblique(Pf, Uob)
    q0 = np.array([[1.25, 0.0, 0.0]])
    e0 = float(E.phi_exact(*q0[0]))
    out = {'nu': np.array(nu_list, float), 'ax': [], 'ob': [], 'probe': []}
    for nu in nu_list:
        cen, nrm, area = E.make_panels(nu, 2 * nu)
        sig, _, _ = W.solve_sources(cen, nrm, area)
        got = W.potential(Pf, cen, area, sig)
        out['ax'].append(np.linalg.norm(got - ex_ax) / np.linalg.norm(ex_ax))
        out['probe'].append(abs(W.potential(q0, cen, area, sig)[0] - e0)
                            / abs(e0))
        sigb, _, _ = W.solve_sources(cen, nrm, area, Uvec=Uob)
        gotb = W.potential(Pf, cen, area, sigb, Uvec=Uob)
        out['ob'].append(np.linalg.norm(gotb - ex_ob) / np.linalg.norm(ex_ob))
    for k in ('ax', 'ob', 'probe'):
        out[k] = np.array(out[k])
        out[k + '_slope'] = float(np.polyfit(np.log(out['nu']),
                                             np.log(out[k]), 1)[0])
    print('  surface L2 relative potential error, %.2fx shell, %dx%d points'
          % (fac, NS, NT))
    print('    %-10s ' % 'n' + '  '.join('%9d' % v for v in nu_list))
    for k, lab in (('ax', 'axial'), ('ob', 'oblique'), ('probe', 'near probe')):
        print('    %-10s ' % lab + '  '.join('%.3e' % v for v in out[k])
              + '   fitted %+.2f' % out[k + '_slope'])
    return out


def plot_ellipsoid(res, path):
    """Panel (a): panel solve against the closed-form ellipsoidal harmonic.

    The metric is the relative L2 error of the potential over the wetted
    surface, for an axial and for an oblique stream, so the sample constrains
    the tangential structure of the solve and not only two points on the axis.
    The convergence order is fitted rather than assumed; it is first order in
    the panels per meridional direction, not second, so no h^2 reference line is
    drawn.
    """
    fig, ax = plt.subplots(figsize=PANEL_FS, constrained_layout=True)
    nu = res['nu']
    slope_n = res['ax_slope']
    print('  fitted log-log slopes vs n_phi:  axial %+.3f   oblique %+.3f'
          '   near probe %+.3f'
          % (slope_n, res['ob_slope'], res['probe_slope']))
    ax.loglog(nu, res['probe'], 'o', color='0.42', ms=3.4, mfc='white', mew=0.8,
              label='probe $x=1.25$, axial')
    ax.loglog(nu, res['ax'], 's-', color='#d63031', ms=4.5, lw=1.2,
              mfc='white', mew=0.9, label='wetted surface, axial')
    ax.loglog(nu, res['ob'], '^-', color='#0984e3', ms=4.5, lw=1.2,
              mfc='white', mew=0.9, label='wetted surface, oblique')
    ref = res['ax'][-1] * (nu / nu[-1]) ** slope_n
    ax.loglog(nu, ref, '--', color='0.55', lw=1.0)
    # Anchored in axes fraction, inside the plot area: the caption already
    # identifies the dashed line as the fitted order, so no arrow is needed and
    # nothing can collide with the frame.
    ax.text(0.96, 0.95, r'$\propto n_\varphi^{%+.2f}$' % slope_n,
            transform=ax.transAxes, ha='right', va='top', fontsize=4.8,
            color='0.30')
    ax.set_xlabel(r'panels per meridional direction $n_\varphi$')
    ax.set_ylabel(r'$\|\Phi_{\rm panel}-\Phi_{\rm exact}\|_2'
                  r'/\|\Phi_{\rm exact}\|_2$')
    ax.set_xticks(list(nu))
    ax.set_xticklabels(['%d' % v for v in nu])
    ax.minorticks_off()
    # 'lower left' now sits in the empty band under the oblique curve, which is
    # the lowest of the three over the whole range.
    ax.legend(fontsize=4.8, loc='lower left')
    ax.tick_params(labelsize=TICK_FS)
    assert check_text_inside(fig, ax, path), 'text on the frame or off canvas'
    fig.savefig(path)
    print('figure -> %s' % path)


def plot_agardb(Tv, prot, err_t0, path):
    """Panel (b): AGARD-B stationarity over the evolution."""
    fig, ax = plt.subplots(figsize=PANEL_FS, constrained_layout=True)
    for tag, c, lab in (('free', '0.35', 'free propagator'),
                        ('analytic V', '#0984e3', r'$H_B$, analytic $V$'),
                        ('exact V', '#00b894', r'$H_B$, exact eigenstate')):
        ax.semilogy(Tv, prot[tag], 'o-', color=c, ms=4.5, lw=1.2, label=lab,
                    mfc='white', mew=0.9)
    ax.axhline(err_t0, color='0.45', ls=':', lw=1.1)
    # Anchored to the right of the third sample, in the empty band above the
    # floor line: at the first sample the label sat exactly on the rising
    # free-propagator and analytic-V curves and both strokes crossed it.  The
    # abscissa is in axes fraction and the ordinate in data units, so the box
    # cannot run past the frame however wide the text is.
    ax.text(0.97, err_t0 * 1.05, r'$t{=}0$ decoding floor',
            transform=ax.get_yaxis_transform(),
            fontsize=4.8, color='0.35', ha='right', va='bottom')
    ax.set_xlabel(r'evolution time $T$')
    ax.set_ylabel(r'$\|v-v_{\rm ref}\|_{\rm rms}/U_\infty$')
    ax.legend(fontsize=4.8, loc='upper left')
    ax.tick_params(labelsize=TICK_FS)
    assert check_text_inside(fig, ax, path), 'text on the frame or off canvas'
    fig.savefig(path)
    print('figure -> %s' % path)


def _npz_ellipsoid(z):
    """Panel-(a) result dict out of an npz written by this file."""
    return {k[4:]: z[k] for k in z.files if k.startswith('ell_')}


if __name__ == '__main__':
    if '--a-only' in sys.argv:
        # Panel (a) depends only on the fast ellipsoid solve, so it can be
        # redrawn and re-cached without re-running the 96^3 hull study.
        sres = ellipsoid_surface_study()
        plot_ellipsoid(sres, os.path.join(FIGDIR, '3d_nonaxisym_a.pdf'))
        z = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
        z.update({('ell_%s' % k): v for k, v in sres.items()})
        np.savez(NPZ, **z)
        print('numbers -> %s' % NPZ)
        raise SystemExit(0)

    if '--replot' in sys.argv:
        z = np.load(NPZ)
        plot_ellipsoid(_npz_ellipsoid(z), os.path.join(FIGDIR, '3d_nonaxisym_a.pdf'))
        plot_agardb(z['T'], {'free': z['free'], 'analytic V': z['ana'],
                             'exact V': z['exact']}, float(z['err_t0']),
                    os.path.join(FIGDIR, '3d_nonaxisym_b.pdf'))
        raise SystemExit(0)

    print('=== A: triaxial ellipsoid, panel solve vs closed-form potential ===')
    sres = ellipsoid_surface_study()
    print()
    print('=== B: AGARD-B calibration model, 96^3 box, L = 6 ===')
    Tv, prot, res, (gg, PHI) = agardb_stationarity()

    plot_ellipsoid(sres, os.path.join(FIGDIR, '3d_nonaxisym_a.pdf'))
    plot_agardb(Tv, prot, res['err_t0'], os.path.join(FIGDIR, '3d_nonaxisym_b.pdf'))

    np.savez(NPZ, T=Tv,
             free=prot['free'], ana=prot['analytic V'], exact=prot['exact V'],
             err_t0=res['err_t0'], rms_lap=res['rms_lap'], r_ana=res['r_ana'],
             r_ex=res['r_ex'], r_im=res['r_im'],
             **{('ell_%s' % k): v for k, v in sres.items()})
    print('numbers -> %s' % NPZ)
