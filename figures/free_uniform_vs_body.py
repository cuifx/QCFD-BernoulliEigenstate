"""
Free-particle evolution: uniform flow (exact eigenstate) vs cylinder flow (dispersion).
=========================================================================================
Control experiment for the error-wall attribution: the free-particle scheme is an
exact propagator for the uniform stream -- Phi = U x has |grad Phi|^2 = U^2 constant,
so e^{i Phi} is an exact eigenstate of the free Hamiltonian and the decoded velocity
holds the uniform field to the O(h^2) decoding floor with no growth in T. The same
scheme fails as soon as a body makes |grad Phi|^2 spatially varying (cylinder case),
which is precisely the error wall established in Sec. 3.

Single control experiment, companion to Sec. 3.6 (does not carry the main argument
alone).  Two panels: (a) the time traces on a logarithmic ordinate, (b) the
initial error against radius on log-log, which is what the R^2 statement in the
text refers to.

Output: fig/free_uniform_vs_body_a.pdf, fig/free_uniform_vs_body_b.pdf
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

import airfoil_validation_composite as avc
from numpy.fft import fft2, ifft2, fftfreq

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')

def free_evolve(phi, N, n_steps, R, gamma, alpha, use_window=True):
    """Free-particle spectral evolution of encoded phase phi (cylinder masked)."""
    dx = 2 * avc.ZETA_RANGE / N
    zx = (np.arange(N) - N // 2) * dx
    XX, YY = np.meshgrid(zx, zx)
    r = np.abs(XX + 1j * YY)
    th = np.angle(XX + 1j * YY)
    psi = avc.madelung_encode(phi)
    if use_window:
        window = avc.smooth_boundary_window(XX, YY, avc.ZETA_RANGE)
        psi = psi * window
    if R > 0:
        inside = r < R
        psi[inside] = 0.0
    kx = 2 * np.pi * fftfreq(N)
    KX, KY = np.meshgrid(kx, kx)
    k_sq = KX ** 2 + KY ** 2
    k_cut = np.pi * 32 / (2 * avc.ZETA_RANGE)
    filt = np.exp(-(k_sq / k_cut ** 2) ** 2)
    psi_k = fft2(psi) * np.exp(-1j * k_sq * n_steps * avc.DT / (2 * avc.MASS)) * filt
    psi_ev = ifft2(psi_k)
    u, v = avc.madelung_decode(psi_ev, dx=dx, dy=dx)
    r2 = np.maximum(r ** 2, 1e-12)
    u = u - gamma * YY / (2 * np.pi * r2)
    v = v + gamma * XX / (2 * np.pi * r2)
    return XX, YY, u, v

def main():
    N = 256
    alpha = np.radians(5.0)

    dx = 2 * avc.ZETA_RANGE / N
    zx = (np.arange(N) - N // 2) * dx
    XX, YY = np.meshgrid(zx, zx)

    # uniform flow: Phi = U x  (no body)
    phi_uniform = avc.U_INF * XX

    # cylinder radii: 5 curves evenly spaced from 0.2 to 2.0
    radii = [0.2, 0.6, 1.0, 1.4, 2.0]
    # dense time sampling (step 0.25) for smooth curves
    Ts = np.arange(0.0, 5.0 + 1e-9, 0.25)

    def ring_err(u, v, XXc, YYc, vr, vt, rmin, rmax):
        """Relative L2 error on the fixed physical probe band r in [rmin, rmax]."""
        r_c = np.sqrt(XXc ** 2 + YYc ** 2)
        mask = (r_c > rmin) & (r_c < rmax) \
            & (np.abs(XXc) < 0.6 * avc.ZETA_RANGE) & (np.abs(YYc) < 0.6 * avc.ZETA_RANGE)
        vm = np.sqrt(u ** 2 + v ** 2)
        vref = np.sqrt(vr ** 2 + vt ** 2)
        d2 = np.where(np.isfinite(vm - vref), (vm - vref) ** 2, 0.0)
        return np.sqrt(np.sum(d2[mask])) / np.sqrt(np.sum(vref[mask] ** 2))

    # --- uniform-flow baseline (R=0) ---
    err_uni = []
    for T in Ts:
        n_steps = int(round(T / avc.DT))
        XXu, YYu, uu, vu = free_evolve(phi_uniform, N, n_steps, 0.0, 0.0, alpha,
                                       use_window=False)
        err_uni.append(ring_err(uu, vu, XXu, YYu,
                                avc.U_INF * np.ones_like(XXu), 0 * YYu,
                                1.5, 3.0))

    # --- cylinder flows at radii 0.2..2.0, probe band starting outside the body ---
    err_cyl = {R: [] for R in radii}
    for T in Ts:
        n_steps = int(round(T / avc.DT))
        for R in radii:
            gamma = -4 * np.pi * R * avc.U_INF * np.sin(alpha)
            r = np.abs(XX + 1j * YY)
            th = np.angle(XX + 1j * YY)
            phi_cyl = avc.U_INF * (r + R ** 2 / np.maximum(r, 1e-9) ** 2) * np.cos(th - alpha)
            XXc, YYc, uc, vc = free_evolve(phi_cyl, N, n_steps, R, gamma, alpha,
                                           use_window=True)
            r_c = np.abs(XXc + 1j * YYc)
            th_c = np.angle(XXc + 1j * YYc)
            vr = avc.U_INF * (1 - R ** 2 / np.maximum(r_c, 1e-9) ** 2) * np.cos(th_c - alpha)
            vt = -avc.U_INF * (1 + R ** 2 / np.maximum(r_c, 1e-9) ** 2) * np.sin(th_c - alpha) \
                + gamma / (2 * np.pi * np.maximum(r_c, 1e-9))
            err_cyl[R].append(ring_err(uc, vc, XXc, YYc, vr, vt,
                                       max(1.5, 1.02 * R), 3.0))
        print(f"  T={T:.3f}: uni={err_uni[0]:.3e}..{err_uni[-1]:.3e} " +
              "  ".join(f"R{R:.1f}={err_cyl[R][-1]:.3f}" for R in radii))

    # the panels are placed at 0.35\columnwidth, i.e. scaled by 0.64, so the
    # source sizes are set to land near the 4.8-5.2 pt panel baseline on paper
    plt.rcParams.update({'font.size': 7.2, 'axes.labelsize': 7.6,
                         'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0})

    # --- panel (a): the time traces, logarithmic so that the radii the paper
    # actually uses (R ~ 0.27) are not crushed onto the axis of a linear plot.
    # The curves are labelled at their right ends: six series spanning three
    # decades leave no empty corner big enough for a legend box.
    fig, ax = plt.subplots(figsize=(3.62, 2.92), constrained_layout=True)
    # Qualitative palette rather than a continuous map: the bright end of a
    # sequential map is a pale yellow that vanishes on white at this line
    # width, and its light end also carries the label text.  These five keep a
    # minimum pairwise CIELAB distance of 40 and all sit below L* = 62, so both
    # the strokes and the labels stay legible in print.
    cols = {0.2: '#3B6FB6', 0.6: '#2E8B72', 1.0: '#C8873A',
            1.4: '#A0548C', 2.0: '#B23A32'}
    # label positions, nudged where two curves end within 0.15 decade
    lab_y = {0.2: 0.010, 0.6: 0.033, 1.0: 0.184, 1.4: 0.62, 2.0: 1.05}
    for i, R in enumerate(radii):
        c = cols[R]
        ax.semilogy(Ts, err_cyl[R], '-', color=c, lw=1.5, ms=0)
        ax.text(5.12, lab_y[R], f'$R={R:.1f}$', color=c, fontsize=7.0,
                va='center', ha='left')
    ax.semilogy(Ts, err_uni, 'o-', color='k', lw=1.5, ms=3.4, zorder=5)
    ax.text(5.12, 1.05e-3, 'uniform', color='k', fontsize=7.0,
            va='center', ha='left')
    ax.set_xlabel('evolution time $T$')
    ax.set_ylabel('interior velocity error')
    ax.set_ylim(1.8e-4, 1.6)
    ax.set_xlim(0, 6.6)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3, which='both')
    out_a = os.path.join(FIG_DIR, 'free_uniform_vs_body_a.pdf')
    fig.savefig(out_a)
    plt.close(fig)
    print(f"  -> {out_a}")

    # --- panel (b): the initial error against radius, which is what the R^2
    # statement in the text refers to; the slope-2 guide is anchored at R = 1
    e0 = np.array([err_cyl[R][0] for R in radii])
    fig, ax = plt.subplots(figsize=(3.62, 2.92), constrained_layout=True)
    ax.loglog(radii, e0, 'o', color='#3B6FB6', ms=5.5, label='decoded error')
    Rg = np.array([0.18, 2.2])
    p = np.polyfit(np.log(radii), np.log(e0), 1)[0]
    ax.loglog(Rg, e0[radii.index(1.0)] * Rg ** 2, '--', color='0.45', lw=1.0,
              label=r'$R^{2}$')
    ax.set_xlabel('cylinder radius $R$')
    ax.set_ylabel('error at $t{=}0$')
    ax.set_xlim(0.15, 2.6)
    ax.set_xticks([0.2, 1.0, 2.0])
    ax.set_xticklabels(['0.2', '1', '2'])
    ax.minorticks_off()
    ax.grid(True, alpha=0.3, which='major')
    ax.legend(fontsize=6.4, loc='upper left', framealpha=0.95,
              handlelength=1.8, labelspacing=0.3, borderpad=0.35)
    ax.annotate(r'$p=%.1f$' % p, (0.62, 0.055), xycoords='axes fraction',
                fontsize=6.6)
    out_b = os.path.join(FIG_DIR, 'free_uniform_vs_body_b.pdf')
    fig.savefig(out_b)
    plt.close(fig)
    print(f"  -> {out_b}   (fitted exponent p = %.2f)" % p)

if __name__ == '__main__':
    main()
