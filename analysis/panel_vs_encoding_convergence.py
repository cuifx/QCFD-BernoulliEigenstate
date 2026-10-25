# -*- coding: utf-8 -*-
"""Refinement convergence of the boundary-integral solve and of the stationary
encoding on the same circular cylinder, against the same analytic reference.

Geometry : circle of radius R_CYL = 0.276, the wall radius of the mapped polar
           grid, with Gamma = 0.
Reference: cylinder_velocity_analytic (one function for both curves).
Metric   : the surface-tangential metric of Section III, max |Delta v_t| / U at
           the body.  The body is sampled by N_p panels for the boundary
           integral and by N_theta = 2 N_r grid points for the encoding, so the
           shared x axis is the number of points resolving the body.

Both curves fall as the square of the body sampling interval, so they share one
reference slope.  The annulus metric that underlies the paper's t=0 decode floor
(1.05 R_CYL < r < 1.5, rms relative speed error, the to_cartesian_bulk call of
the paper) is printed alongside; the two metrics are not conflated.

Output: fig/panel_encoding_convergence.pdf and a console table.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401
import airfoil_panel_method as pm  # noqa: E402
from unbounded_mapping import (  # noqa: E402
    MappedPolarGrid, R_CYL, cylinder_velocity_analytic,
)

U, GAMMA, ALPHA = 1.0, 0.0, 0.0
R_PROBE = 1.5
R_IN = 1.05 * R_CYL

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'fig', 'panel_encoding_convergence.pdf')

N_PANEL = (16, 32, 64, 128, 256, 512, 1024)
N_RADIAL = (32, 64, 128, 256, 512)

# paper figure conventions: figsize is the placed physical size, so the numbers
# below are the point sizes that reach the page
PLACED_W_IN = 0.5 * 481.50787 / 72.27
FS_TICK, FS_LABEL, FS_LEG = 4.8, 5.2, 4.8
C_PANEL, C_ENC = '#2f6690', '#c0504d'


def annulus_err(spd, spd_ref):
    return np.sqrt(np.mean((spd - spd_ref) ** 2)) / np.sqrt(np.mean(spd_ref ** 2))


def fixed_sample():
    g = MappedPolarGrid(Nr=256, Nt=512, L_map=2.5, R_max_phys=50.0)
    m = (g.RR > R_IN) & (g.RR < R_PROBE)
    return g.RR[m], g.TT[m]


def circle_boundary(n):
    th = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return pm.order_boundary(R_CYL * np.exp(1j * th))


def encoding_curve():
    nb, surf, ann, nr = [], [], [], []
    for N in N_RADIAL:
        g = MappedPolarGrid(Nr=N, Nt=2 * N, L_map=2.5, R_max_phys=50.0)
        psi = g.encode(GAMMA)
        vr, vt, _ = g.decode(psi)

        _, vt_ref = cylinder_velocity_analytic(R_CYL, g.theta, GAMMA)
        surf.append(np.abs(vt[0, :] - vt_ref).max() / U)

        res = g.to_cartesian_bulk(vr, vt, GAMMA, R_probe=R_PROBE)
        ann.append(res['vel_err'])
        nb.append(2 * N)
        nr.append(N)
    return np.array(nb), np.array(surf), np.array(ann), np.array(nr)


def panel_curve(rs, ths):
    np_, surf, ann = [], [], []
    for N in N_PANEL:
        b = circle_boundary(N)
        sol = pm.solve(b, ALPHA, U=U, Gamma=GAMMA)

        c = sol['centers']
        vx, vy = pm.field_velocity(c.real, c.imag, b, sol, ALPHA, U=U)
        tvec = sol['t']
        vt_num = vx * tvec[:, 0] + vy * tvec[:, 1]
        ex = (U * (np.exp(-1j * ALPHA) - R_CYL ** 2 * np.exp(1j * ALPHA) / c ** 2)
              + GAMMA / (2 * np.pi * 1j * c))
        vt_ex = ex.real * tvec[:, 0] + (-ex.imag) * tvec[:, 1]
        surf.append(np.abs(vt_num - vt_ex).max() / U)

        x, y = rs * np.cos(ths), rs * np.sin(ths)
        vx, vy = pm.field_velocity(x, y, b, sol, ALPHA, U=U)
        vrr, vtt = cylinder_velocity_analytic(rs, ths, GAMMA)
        ann.append(annulus_err(np.sqrt(vx ** 2 + vy ** 2),
                               np.sqrt(vrr ** 2 + vtt ** 2)))
        np_.append(N)
    return np.array(np_), np.array(surf), np.array(ann)


def order(x, y):
    return np.diff(np.log(y)) / np.diff(np.log(x))


def main():
    t0 = time.time()
    rs, ths = fixed_sample()
    print("annulus %.4f < r < %.2f ; panel field sampled on %d mapped nodes"
          % (R_IN, R_PROBE, rs.size))

    enb, esurf, eann, enr = encoding_curve()
    pnb, psurf, pann = panel_curve(rs, ths)

    print("\n%-34s %-14s %-14s" % ("", "surface max|dvt|/U", "annulus rms"))
    for i in range(len(enb)):
        print("  encoding  N_theta=%-6d N_r=%-5d %-14.4e %-14.4e"
              % (enb[i], enr[i], esurf[i], eann[i]))
    for i in range(len(pnb)):
        print("  panel     N_p=%-10d %-14s %-14.4e"
              % (pnb[i], "", pann[i]))

    print("\nfitted orders, surface metric (per doubling of body points)")
    print("  encoding : %s" % np.round(order(enb, esurf), 2))
    print("  panel    : %s" % np.round(order(pnb, psurf), 2))
    print("fitted orders, annulus metric, encoding vs N_r")
    print("  encoding : %s" % np.round(order(enr, eann), 2))
    print("fitted orders, annulus metric, panel vs N_p")
    print("  panel    : %s" % np.round(order(pnb, pann), 2))

    h_in = PLACED_W_IN / 1.75
    fig, ax = plt.subplots(figsize=(PLACED_W_IN, h_in), constrained_layout=True)

    ax.loglog(pnb, psurf, marker='o', ms=2.4, lw=0.9, color=C_PANEL, mfc='white',
              mew=0.7, label='boundary-integral solve')
    ax.loglog(enb, esurf, marker='s', ms=2.4, lw=0.9, color=C_ENC,
              label='stationary encoding')

    x0, x1 = 2.2e1, 1.1e3
    ax.plot([x0, x1], [5.5e-2, 5.5e-2 * (x1 / x0) ** -2],
            ls='--', lw=0.7, color='0.45', zorder=0)
    ax.annotate(r'$N_b^{-2}$', xy=(2.6e2, 5.0e-4), fontsize=FS_LEG,
                color='0.35', ha='left', va='center')

    ax.set_xlabel(r'points on the body, $N_b$', fontsize=FS_LABEL)
    ax.set_ylabel(r'$\max|\Delta v_t|/U$', fontsize=FS_LABEL)
    ax.set_xlim(1.35e1, 1.5e3)
    ax.set_ylim(4.0e-6, 1.2e-1)
    ax.tick_params(labelsize=FS_TICK, length=2.0, width=0.5, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.grid(True, which='major', lw=0.35, color='0.85', zorder=0)
    ax.legend(fontsize=FS_LEG, loc='upper right', framealpha=0.95,
              edgecolor='0.7', handlelength=1.6, borderpad=0.4, labelspacing=0.3)
    fig.savefig(FIG, dpi=400)
    print("\nwrote %s   (%.1f s)" % (FIG, time.time() - t0))


if __name__ == "__main__":
    main()
