# -*- coding: utf-8 -*-
"""Independent panel-method lift baseline for the mapped airfoil sections.

Why this exists
---------------
The manuscript reports the surface C_L of the three sections at alpha = 5 deg
from the Kutta-Joukowski relation C_L = 2|Gamma|/(U c), with the circulation
taken from the section's own Kutta condition
Gamma = -4 pi R U sin(alpha - beta), where beta is the ideal angle of the
section (0 for the symmetric NACA 0012, -2.07 deg for the cambered NACA 2412,
and the cusp pre-image angle for Joukowski).  No classical CFD solver was ever
run on that
lift: the panel method in this repository is used only for the cylinder's
surface velocity.  This script closes the gap with a boundary-integral solver
that works on the exact mapped section, determines Gamma from its own
numerical Kutta condition instead of from the circle, and gets C_L from a
surface-pressure integral instead of from the Kutta-Joukowski relation.

Method
------
Hess-Smith type: N constant-strength SOURCE panels plus a constant-strength
VORTEX sheet over the whole perimeter, so the circulation is one extra
unknown.  N no-penetration conditions + 1 Kutta condition close the system.

  local unit-source velocity on panel [0, L] at local point (x0, y0):
      u_s = (1/4pi) ln((x0^2+y0^2) / ((x0-L)^2+y0^2))
      v_s = (1/2pi) [atan2(y0, x0-L) - atan2(y0, x0)]
  a unit vortex sheet is the source sheet rotated: (u_v, v_v) = (-v_s, u_s);
  just above a panel centre that gives u = -1/2, i.e. the standard jump
  u_above - u_below = -gamma.

  A panel's own influence is NEVER evaluated from the kernel.  At the exact
  mid-point the kernel is a 0/0 limit whose sign is decided by a round-off
  level sign of the local normal offset: a -1e-17 offset makes atan2 return
  -pi and flips the self term, which silently corrupts the surface velocity
  (and hence the circulation and the pressure integral).  Analytic self
  terms are used instead:  source -> (+1/2) n,  vortex -> (-1/2) t.

Kutta conditions
----------------
  'morino' : V_t(first cp) + V_t(last cp) = 0 -- equal speeds either side of
             the TE, the usual closure for a closed body.
  'fixed'  : circulation prescribed at the manuscript's value, no Kutta row,
             so the pressure integral can be checked against Kutta-Joukowski
             on identical data.

Sign conventions
----------------
Gamma < 0 is clockwise and gives positive lift (matches the manuscript and
airfoil_hb_quantum.py).  C_L comes from
    C_L = -(1/c) sum_j Cp_j n_y_j L_j,   n outward,
so it carries no circulation convention.

Validation on a circular cylinder of radius R, where the exact surface speed
is -2U sin(theta - alpha) + Gamma/(2 pi R) and C_L must equal 2|Gamma|/(U 2R).
The solver reproduces the cylinder to 4e-4 in Gamma and 3.8e-3 in C_L.

STATUS -- read before trusting a number
---------------------------------------
  * kutta='fixed'  (circulation taken from the manuscript): VALIDATED.  On the
    mapped sections the pressure integral returns C_L = 0.6024 (NACA 0012) and
    0.5988 (NACA 2412) against 0.6048 / 0.6011 from C_L = 2|Gamma|/(U c), i.e.
    the manuscript's shortcut is consistent with the true surface pressure to
    0.4-0.5%.  sum(v_t L) reproduces -Gamma to 1e-4.
  * kutta='morino' (circulation solved from a numerical Kutta condition): NOT
    TRUSTWORTHY on these sections and reported only to document the obstacle.
    The mapped trailing edge is BLUNT (thickness 0.0022 c at x/c = 0.999), so
    the inviscid circulation is not fixed by the geometry: moving the assumed
    TE point across the small TE face changes Gamma by ~50%, and the numerical
    closure does not converge with N (NACA 0012: |Gamma| = 0.299, 0.337, 0.341
    at N = 400, 800, 1600).  On the cusped Joukowski, where the Kutta value is
    unambiguous, it undershoots the exact value by 5%.  Use the analytic
    reference in lift_slope_reference.py instead.
"""
import os
import sys

import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401

import airfoil_panel_method as pm  # noqa: E402

U_INF = 1.0
ALPHA_DEG = 5.0

# Ideal (zero-lift) angle of each mapped section, in degrees.  The Kutta
# condition puts the rear stagnation point at the pre-image of the trailing
# edge, whose circle-plane angle equals this value, so the circulation of the
# section is Gamma = -4 pi R U sin(alpha - beta).  For the cambered NACA 2412 it
# is the thin-airfoil value of the section's own camber line (m = 0.02,
# p = 0.4); for the cusped Joukowski it is the exact cusp pre-image angle,
# angle(1 - zc), computed below.
IDEAL_ANGLE_DEG = {'NACA 0012': 0.0, 'NACA 2412': -2.0685}


# ------------------------------------------------------------------ geometry
def fwd_map(zeta, kind, coeffs, b2=1.0):
    if kind == 'joukowski':
        return zeta + b2 / zeta
    z = zeta + coeffs[0]
    for k in range(1, min(len(coeffs), 64)):
        z = z + coeffs[k] / (zeta ** k)
    return z


def load_sections(chord_j=1.0):
    d = np.load(os.path.join(_d, 'data', 'laurent_coeffs.npz'), allow_pickle=True)
    eps, delta = 0.1, 0.05
    zc_raw = complex(-eps, delta)
    a_raw = np.sqrt((1 + eps) ** 2 + delta ** 2)
    # The Joukowski family z = zeta + b^2/zeta has no free scale once the cusp
    # is pinned at zeta = b, so b = 1 gives a chord of 4.03.  Scale the circle
    # plane by s so that this section, like the two NACA ones, has chord
    # chord_j.  Every dimensionless quantity is unchanged by s; what changes is
    # that the per-chord grid resolution and the error windows become the same
    # for the three sections.
    th = np.linspace(0.0, 2.0 * np.pi, 4001)
    z_raw = zc_raw + a_raw * np.exp(1j * th)
    w_raw = z_raw + 1.0 / z_raw
    s = chord_j / float(w_raw.real.max() - w_raw.real.min())
    zc_j, a_j = zc_raw * s, a_raw * s
    b = s
    return [
        dict(name='NACA 0012', kind='naca', R=float(d['R_naca12']), zc=0j,
             coeffs=np.asarray(d['c_naca12']), alpha=np.radians(ALPHA_DEG),
             beta=float(np.radians(IDEAL_ANGLE_DEG['NACA 0012']))),
        dict(name='NACA 2412', kind='naca', R=float(d['R_naca24']), zc=0j,
             coeffs=np.asarray(d['c_naca24']), alpha=np.radians(ALPHA_DEG),
             beta=float(np.radians(IDEAL_ANGLE_DEG['NACA 2412']))),
        dict(name='Joukowski', kind='joukowski', R=a_j, zc=zc_j, coeffs=None,
             alpha=0.0, b2=s * s, beta=float(np.angle(b - zc_j))),
    ]


def _orient_and_close(z):
    """Rotate an OPEN node ring to start at the TE and close it with N+1 nodes.

    pm.order_boundary does this on an already-closed array, which combined
    with its reversal step leaves one duplicated node (a zero-length panel)
    and makes the system singular.  Rotating before closing avoids that.
    """
    z = np.asarray(z)
    area = 0.5 * np.sum(z.real * np.roll(z.imag, -1) - np.roll(z.real, -1) * z.imag)
    if area > 0:                     # want clockwise (interior to the right)
        z = z[::-1]
    k = int(np.argmax(z.real))
    z = np.roll(z, -k)
    return np.append(z, z[:1])


def body_nodes(sec, N):
    """Closed node list (N+1), clockwise, starting at the mapped TE.

    Nodes are uniform in the circle angle.  Through the conformal map this
    already clusters nodes at the trailing edge and the leading edge (the
    panel length there is ~30x shorter than at mid-chord), which is where the
    resolution is needed.  Arc-length reclustering was tried and rejected: it
    produced a half-chord-long panel and broke the circulation identity
    sum(v_t L) = -gamma S at the 6% level.
    """
    th = 2.0 * np.pi * np.arange(N) / N
    z = fwd_map(sec['zc'] + sec['R'] * np.exp(1j * th), sec['kind'],
                sec['coeffs'], sec.get('b2', 1.0))
    return _orient_and_close(z)


def circle_nodes(R, N):
    th = 2.0 * np.pi * np.arange(N) / N
    return _orient_and_close(R * np.exp(1j * th))


# ------------------------------------------------------------------ influence
def panel_global(nodes, t, n, j, XP, YP):
    """Global velocity at points (XP, YP) from unit panels at panel j.

    Returns (su, sv, vu, vv): source and vortex velocities.  The entry for the
    point that coincides with panel j's own control point is replaced by the
    analytic self term.
    """
    x1, y1 = nodes[j].real, nodes[j].imag
    x2, y2 = nodes[j + 1].real, nodes[j + 1].imag
    tx, ty = float(t[j, 0]), float(t[j, 1])
    nx, ny = float(n[j, 0]), float(n[j, 1])
    lx = (XP - x1) * tx + (YP - y1) * ty
    ly = (XP - x1) * nx + (YP - y1) * ny
    L = float(np.hypot(x2 - x1, y2 - y1))
    r1sq = np.maximum(lx * lx + ly * ly, 1e-24)
    r2sq = np.maximum((lx - L) ** 2 + ly * ly, 1e-24)
    us = np.log(r1sq / r2sq) / (4.0 * np.pi)
    vs = (np.arctan2(ly, lx - L) - np.arctan2(ly, lx)) / (2.0 * np.pi)
    us = us.copy(); vs = vs.copy()
    us[j] = 0.0          # source self: no tangential component
    vs[j] = 0.5          # source self: +sigma/2 along the outward normal
    return (us * tx + vs * nx, us * ty + vs * ny,
            -vs * tx + us * nx, -vs * ty + us * ny)


def te_face_length(n):
    """Number of panels in the blunt TE face.

    The traversal starts at the upper TE corner and immediately runs down the
    near-vertical TE face, so the first K panels are that face, not the lower
    surface.  Applying Morino across panels (0, N-1) would therefore compare
    two points on the face itself.  The face is identified by its outward
    normal pointing downstream (n_x > 0.5); the airfoil-side neighbours are
    then panel K (lower surface) and panel N-1 (upper surface).
    """
    k = 0
    while k < len(n) and n[k, 0] > 0.5:
        k += 1
    return max(k, 1)


def assemble(nodes):
    L, t, n, centers = pm._panel_geom(nodes)
    N = len(L)
    cx, cy = centers.real, centers.imag
    Sn = np.zeros((N, N)); St = np.zeros((N, N))
    Vn = np.zeros((N, N)); Vt = np.zeros((N, N))
    for j in range(N):
        su, sv, vu, vv = panel_global(nodes, t, n, j, cx, cy)
        Sn[:, j] = su * n[:, 0] + sv * n[:, 1]
        St[:, j] = su * t[:, 0] + sv * t[:, 1]
        Vn[:, j] = vu * n[:, 0] + vv * n[:, 1]
        Vt[:, j] = vu * t[:, 0] + vv * t[:, 1]
    return L, t, n, centers, Sn, St, Vn, Vt


def solve_on(nodes, alpha, kutta='morino', Gamma=None, U=U_INF):
    L, t, n, centers, Sn, St, Vn, Vt = assemble(nodes)
    N = len(L)
    ca, sa = np.cos(alpha), np.sin(alpha)
    un = U * (ca * n[:, 0] + sa * n[:, 1])
    ut = U * (ca * t[:, 0] + sa * t[:, 1])

    if kutta == 'morino':
        K = te_face_length(n)
        i1, i2 = K, N - 1          # lower-surface and upper-surface panels at the TE
        A = np.zeros((N + 1, N + 1))
        A[:N, :N] = Sn
        A[:N, N] = Vn.sum(axis=1)
        b = np.zeros(N + 1)
        b[:N] = -un
        A[N, :N] = St[i1, :] + St[i2, :]
        A[N, N] = Vt[i1, :].sum() + Vt[i2, :].sum()
        b[N] = -(ut[i1] + ut[i2])
        x = np.linalg.solve(A, b)
        sigma, gamma = x[:N], float(x[N])
    elif kutta == 'fixed':
        gamma = Gamma / float(L.sum())
        sigma = np.linalg.solve(Sn, -un - gamma * Vn.sum(axis=1))
    else:
        raise ValueError(kutta)

    cx, cy = centers.real, centers.imag
    vx = np.full(N, U * ca); vy = np.full(N, U * sa)
    for j in range(N):
        su, sv, vu, vv = panel_global(nodes, t, n, j, cx, cy)
        vx += sigma[j] * su + gamma * vu
        vy += sigma[j] * sv + gamma * vv

    vt = vx * t[:, 0] + vy * t[:, 1]
    vn = vx * n[:, 0] + vy * n[:, 1]
    speed = np.hypot(vx, vy)
    chord = float(nodes.real.max() - nodes.real.min())
    Gamma_trav = float(np.sum(vt * L))
    Cp = 1.0 - (speed / U) ** 2
    return dict(nodes=nodes, L=L, t=t, n=n, centers=centers, sigma=sigma,
                gamma=gamma, vt=vt, vn=vn, speed=speed, chord=chord,
                Gamma_trav=Gamma_trav, Cp=Cp,
                CL_pressure=-float(np.sum(Cp * n[:, 1] * L)) / chord,
                CL_kj=2.0 * abs(Gamma_trav) / (U * chord))


# ------------------------------------------------------------------ validation
def validate(N=800, R=1.0, alpha=np.radians(5.0), U=1.0, quiet=False):
    nodes = circle_nodes(R, N)
    Gamma = -4.0 * np.pi * R * U * np.sin(alpha)
    s = solve_on(nodes, alpha, kutta='fixed', Gamma=Gamma, U=U)
    th_c = np.angle(s['centers'])
    vt_exact = -2 * U * np.sin(th_c - alpha) + Gamma / (2 * np.pi * R)
    rel = abs(s['CL_pressure'] - s['CL_kj']) / s['CL_kj']
    if not quiet:
        print('  cylinder N=%-5d  max|V.n|=%.2e  max|vt-vt_exact|=%.2e'
              '  sum(vt*L)/Gamma=%+.4f  C_L(p)=%.5f  C_L(KJ)=%.5f  reldiff=%.1e'
              % (N, np.abs(s['vn']).max(), np.abs(s['vt'] - vt_exact).max(),
                 s['Gamma_trav'] / Gamma, s['CL_pressure'], s['CL_kj'], rel))
    return s, rel


# ------------------------------------------------------------------ reporting
def main():
    print('=' * 78)
    print('Panel-method lift baseline (Hess-Smith, numerical Kutta)')
    print('=' * 78)
    print()
    print('--- validation: circular cylinder, exact solution known ---')
    worst = 0.0
    for N in (200, 400, 800, 1600):
        _, rel = validate(N)
        worst = max(worst, rel)
    print('  worst relative gap between C_L(pressure) and C_L(KJ) = %.2e' % worst)
    print()
    for sec in load_sections():
        print('#### %s   alpha = %.3f deg ####'
              % (sec['name'], np.degrees(sec['alpha'])))
        Gam = -4.0 * np.pi * sec['R'] * U_INF * np.sin(sec['alpha'] - sec['beta'])
        print('  manuscript Gamma = %+.6f   (C_L = 2|Gamma|/(U c))' % Gam)
        for kutta, tag in (('fixed', 'Gamma fixed at the manuscript value'),
                           ('morino', 'Gamma from numerical Kutta  [NOT CONVERGED]')):
            print('  --- %s ---' % tag)
            last = None
            for N in (400, 800, 1600):
                last = solve_on(body_nodes(sec, N), sec['alpha'],
                                kutta=kutta, Gamma=Gam)
                print('    N=%-5d c=%.6f |Gamma|=%.6f  C_L(press)=%.5f'
                      '  C_L(KJ)=%.5f  max|V.n|=%.2e'
                      % (N, last['chord'], abs(last['Gamma_trav']),
                         last['CL_pressure'], last['CL_kj'],
                         np.abs(last['vn']).max()))
            print('    TE control-point v_t = (%+.4f, %+.4f)   [N=1600]'
                  % (last['vt'][0], last['vt'][-1]))
        print()


if __name__ == '__main__':
    main()
