# -*- coding: utf-8 -*-
"""Geometry of the three-dimensional cases, as one vector schematic figure.

Nothing is solved here.  Every outline is evaluated from the same
parametrisation the solvers use, so the drawn shapes are the verified ones:

  fig/3d_geom_a.pdf   von Karman double ogive, (L, R_max) = (4.0, 0.5), side
                      view.  R(xi) is Equation (25) of the manuscript.
  fig/3d_geom_b.pdf   triaxial ellipsoid (a, b, c) = (1.0, 0.8, 0.6): the x-y
                      and x-z principal sections, which is what makes the
                      triaxiality visible, with the oblique free stream.
  fig/3d_geom_c.pdf   AGARD-B wing-body, planform.  Body radius, nose length,
                      root chord and span are AB_R0, AB_LN, AB_CR and AB_B of
                      winged_body_3d_probe.py, i.e. the geometry the panel solve
                      and the 24-qubit build both used.

Every panel is authored at the size it is placed.  The figure carries
\\panelw = 0.29 of the 481.5 pt column, i.e. 139.6 pt = 1.9394 in per panel, so
the scale is 1 and the point sizes below are the sizes on the page.  Authored
at 1.9394 in rather than at 5.4 in and shrunk by LaTeX, for the reason recorded
in nonaxisym_3d_figures.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the canvas/frame sentinel, shared with the other three-dimensional figure
from nonaxisym_3d_figures import check_text_inside               # noqa: E402

import matplotlib                                                # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 5.2,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIGDIR = 'fig'
# 0.29 of the 481.5 pt column = 139.6 pt, the width the three-panel figures of
# this manuscript already use (sm_real_position_*), authored at the size it is
# placed so the point sizes below are the sizes on the page.
PANEL_FS = (1.9394, 1.47)
INK = '#1f3b73'
ALT = '#b03a2e'
FAINT = '#8fa8c8'


def _frame(ax, xlo, xhi, ylo, yhi):
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('equal')
    ax.axis('off')


def panel_a(path):
    """von Karman double ogive, side view with two station rings."""
    D = 1.0                       # lengths in body diameters
    L, rm = 4.0 * D, 0.5 * D      # (L, R_max) = (4.0, 0.5), Eq. (25)
    xi = np.linspace(0.0, 1.0, 801)
    th = np.where(xi <= 0.5,
                  np.arccos(np.clip(1.0 - 4.0 * xi, -1.0, 1.0)),
                  np.arccos(np.clip(1.0 - 4.0 * (1.0 - xi), -1.0, 1.0)))
    r = rm * np.sqrt(np.maximum(th - 0.5 * np.sin(2.0 * th), 0.0) / np.pi)
    x = (xi - 0.5) * L

    fig, ax = plt.subplots(figsize=PANEL_FS, constrained_layout=True)
    ax.plot(x, r, '-', color=INK, lw=0.9)
    ax.plot(x, -r, '-', color=INK, lw=0.9)
    # a station ring projects to an ellipse of vertical semi-axis r and
    # horizontal semi-axis r*sin(view angle)
    t = np.linspace(0.0, 2.0 * np.pi, 240)
    for xj in (-0.9, 0.0, 1.1):
        rj = float(r[int(np.argmin(np.abs(x - xj)))])
        ax.plot(xj + 0.30 * rj * np.cos(t), rj * np.sin(t), '-',
                color=FAINT, lw=0.5, zorder=1)
    # dimension line under the hull
    yb = -1.05
    ax.annotate('', xy=(-L / 2, yb), xytext=(L / 2, yb),
                arrowprops=dict(arrowstyle='<|-|>', color='0.25', lw=0.5,
                                mutation_scale=3.5))
    ax.plot([-L / 2, -L / 2], [yb - 0.16, -rm * 0.1], '-', color='0.55', lw=0.4)
    ax.plot([L / 2, L / 2], [yb - 0.16, -rm * 0.1], '-', color='0.55', lw=0.4)
    ax.text(0.0, yb - 0.10, r'$L/D=4$', ha='center', va='top', fontsize=5.2,
            color='0.15')
    # x/y matched to the panel aspect, so the drawing box fills the canvas
    _frame(ax, -2.65, 2.65, -2.01, 2.01)
    assert check_text_inside(fig, ax, path), 'text on the frame or off canvas'
    fig.savefig(path)
    plt.close(fig)
    print('  panel a -> %s' % path)


def panel_b(path):
    """Triaxial ellipsoid: the two principal sections and the oblique stream."""
    a, b, c = 1.0, 0.8, 0.6
    t = np.linspace(0.0, 2.0 * np.pi, 480)

    fig, ax = plt.subplots(figsize=PANEL_FS, constrained_layout=True)
    ax.plot(a * np.cos(t), b * np.sin(t), '-', color=INK, lw=0.9)
    ax.plot(a * np.cos(t), c * np.sin(t), '--', color=ALT, lw=0.7)
    # the oblique stream, drawn outside the body
    ang = np.deg2rad(26.0)
    u = np.array([np.cos(ang), np.sin(ang)])
    ax.annotate('', xy=1.34 * u, xytext=2.20 * u,
                arrowprops=dict(arrowstyle='-|>', color='0.15', lw=0.6,
                                mutation_scale=4.0))
    ax.text(2.26 * u[0], 2.26 * u[1], r'$U$', ha='left', va='bottom',
            fontsize=5.2, color='0.15')
    ax.plot([-a, a], [0.0, 0.0], '-', color='0.75', lw=0.4, zorder=0)
    ax.text(-1.62, -1.62, r'$(a,b,c)=(1,\,0.8,\,0.6)$', ha='left', va='bottom',
            fontsize=5.2, color='0.15')
    # the two sections are told apart by a legend rather than by a lone axis
    # letter, which read as an orphan in the first draft of this panel
    ax.legend(handles=[Line2D([], [], color=INK, lw=0.9, label=r'$x$-$y$'),
                       Line2D([], [], color=ALT, lw=0.7, ls='--',
                              label=r'$x$-$z$')],
              fontsize=4.8, loc='lower right', frameon=False, handlelength=1.5,
              borderaxespad=0.15, labelspacing=0.25, handletextpad=0.5)
    _frame(ax, -2.55, 2.55, -1.93, 1.93)
    assert check_text_inside(fig, ax, path), 'text on the frame or off canvas'
    fig.savefig(path)
    plt.close(fig)
    print('  panel b -> %s' % path)


def panel_c(path):
    """AGARD-B wing-body, planform.  Bodies are slender, so the wing defines
    the vertical extent and the equal aspect is kept."""
    D = 1.0
    r0, ln, xte_w, cr, bspan = 0.5 * D, 3.0 * D, 7.098 * D, 2.0 * np.sqrt(3.0) * D, 2.0 * D
    ltot, lb = 8.5 * D, 0.25 * D
    xle_w = xte_w - cr
    assert abs(cr / bspan - np.tan(np.deg2rad(60.0))) < 0.02, 'sweep is not 60'

    xb = np.linspace(0.0, ltot, 900)
    # cubic-ogive nose on [0, 3D], cylinder, then the base fairing
    nose = np.where(xb <= ln,
                    (xb / 3.0) * (1.0 - (xb / D) ** 2 / 9.0 + (xb / D) ** 3 / 54.0),
                    r0)
    x0 = ltot - lb
    cap = r0 * np.sqrt(np.clip(1.0 - ((xb - x0) / lb) ** 2, 0.0, None))
    rb = np.where(xb > x0, cap, nose)

    fig, ax = plt.subplots(figsize=PANEL_FS, constrained_layout=True)
    ax.plot(xb, rb, '-', color=INK, lw=0.9)
    ax.plot(xb, -rb, '-', color=INK, lw=0.9)
    for s in (1.0, -1.0):                      # delta planform, 60 deg LE sweep
        ax.plot([xle_w, xte_w, xte_w, xle_w],
                [0.0, 0.0, s * bspan, 0.0], '-', color=ALT, lw=0.8)
    # the wing-body crease, where the two surfaces are combined with max()
    ax.plot([xle_w, xte_w], [r0, r0], '--', color=FAINT, lw=0.5)
    ax.plot([xle_w, xte_w], [-r0, -r0], '--', color=FAINT, lw=0.5)
    # dimension lines
    yb = -bspan - 1.05
    ax.annotate('', xy=(0.0, yb), xytext=(ltot, yb),
                arrowprops=dict(arrowstyle='<|-|>', color='0.25', lw=0.5,
                                mutation_scale=3.5))
    ax.text(ltot / 2, yb + 0.22, r'$8.5D$', ha='center', va='bottom',
            fontsize=5.2, color='0.15')
    ax.annotate('', xy=(ltot + 0.55, -bspan), xytext=(ltot + 0.55, bspan),
                arrowprops=dict(arrowstyle='<|-|>', color='0.25', lw=0.5,
                                mutation_scale=3.5))
    ax.text(ltot + 0.72, 0.0, r'span $4D$', ha='left', va='center',
            rotation=90.0, fontsize=5.2, color='0.15')
    # leading-edge sweep, drawn as an arc between the LE ray and the spanwise
    # direction at the root; a bare label there rode on top of the LE line
    th_arc = np.linspace(np.deg2rad(30.0), np.deg2rad(90.0), 80)
    ra = 0.85
    ax.plot(xle_w + ra * np.cos(th_arc), ra * np.sin(th_arc), '-', color=ALT,
            lw=0.5)
    ax.text(xle_w + 1.18 * np.cos(np.deg2rad(60.0)),
            1.18 * np.sin(np.deg2rad(60.0)),
            r'$60^\circ$', ha='center', va='center', fontsize=5.2, color=ALT)
    # the limits are chosen so that x/y matches the canvas aspect: with
    # set_aspect('equal') a mismatched range shrinks the drawing box and leaves
    # the panel looking narrower than its two neighbours
    _frame(ax, -1.0, 10.0, -4.17, 4.17)
    assert check_text_inside(fig, ax, path), 'text on the frame or off canvas'
    fig.savefig(path)
    plt.close(fig)
    print('  panel c -> %s' % path)


if __name__ == '__main__':
    print('three-dimensional geometry schematic -> %s' % FIGDIR)
    panel_a(os.path.join(FIGDIR, '3d_geom_a.pdf'))
    panel_b(os.path.join(FIGDIR, '3d_geom_b.pdf'))
    panel_c(os.path.join(FIGDIR, '3d_geom_c.pdf'))
