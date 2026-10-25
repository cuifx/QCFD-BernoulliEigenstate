# -*- coding: utf-8 -*-
"""Geometry A: triaxial ellipsoid, exact ellipsoidal-harmonic potential vs the
panel solve -- a genuinely independent baseline for the non-axisymmetric case.

Uniform stream U along x, semi-axes a > b > c.  The exact exterior potential is

    Phi = U x + C x I(lam),
    I(lam) = int_lam^inf ds / [(a^2+s)^{3/2} (b^2+s)^{1/2} (c^2+s)^{1/2}],

with lam the ellipsoidal coordinate (largest root of
x^2/(a^2+lam) + y^2/(b^2+lam) + z^2/(c^2+lam) = 1).

Two independent routes fix C and they agree:
  * no penetration on lam = 0   ->  C = U a b c / [2 (1 - n_x)]
  * far-field dipole matching   ->  C = 3 p_x / 2
using n_x and p_x from ellipsoid_dipole_check.py, which was itself calibrated on
the sphere.  For (1.0, 0.8, 0.6) both give C = 0.31579.

The panel solve is the same centroid-source solver used for the winged body, so
agreement here validates the solver itself rather than the winged geometry.
"""
import sys
import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, 'code/analysis')
import winged_body_3d_probe as W

U = 1.0
A_, B_, C_ = 1.0, 0.8, 0.6


# ---------------------------------------------------------------- exact solve
def n_axis(a, b, c):
    f = lambda s: 1.0 / ((a * a + s) ** 1.5 * (b * b + s) ** 0.5 * (c * c + s) ** 0.5)
    val, _ = integrate.quad(f, 0.0, np.inf)
    return a * b * c / 2.0 * val


def dipole(a, b, c, U=U):
    return (a * b * c / 3.0) * U / (1.0 - n_axis(a, b, c))


def I_of(lam, a, b, c):
    f = lambda s: 1.0 / ((a * a + s) ** 1.5 * (b * b + s) ** 0.5 * (c * c + s) ** 0.5)
    val, _ = integrate.quad(f, lam, np.inf)
    return val


def lam_of(x, y, z, a, b, c):
    """Largest root of the ellipsoidal-coordinate equation (exterior points)."""
    f = lambda L: x * x / (a * a + L) + y * y / (b * b + L) + z * z / (c * c + L) - 1.0
    if f(0.0) <= 0.0:
        return np.nan                                   # inside the body
    hi = max(abs(x), abs(y), abs(z)) ** 2 + a * a
    while f(hi) > 0.0:
        hi *= 2.0
    return optimize.brentq(f, 0.0, hi, xtol=1e-13, rtol=1e-15)


def phi_exact(x, y, z, a=A_, b=B_, c=C_, U=U):
    Cc = 1.5 * dipole(a, b, c, U)
    L = lam_of(x, y, z, a, b, c)
    if np.isnan(L):
        return np.nan
    return U * x + Cc * x * I_of(L, a, b, c)


# ---------------------------------------------------------------- panels
def make_panels(nu, nv, a=A_, b=B_, c=C_):
    """Midpoint grid on the triaxial ellipsoid, analytic area and normal."""
    u = (np.arange(nu) + 0.5) * np.pi / nu
    v = (np.arange(nv) + 0.5) * 2.0 * np.pi / nv
    Uu, Vv = np.meshgrid(u, v, indexing='ij')
    du, dv = np.pi / nu, 2.0 * np.pi / nv

    def r(u, v):
        return np.stack([a * np.sin(u) * np.cos(v),
                         b * np.sin(u) * np.sin(v),
                         c * np.cos(u)], -1)

    def ru(u, v):
        return np.stack([a * np.cos(u) * np.cos(v),
                         b * np.cos(u) * np.sin(v),
                         -c * np.sin(u)], -1)

    def rv(u, v):
        return np.stack([-a * np.sin(u) * np.sin(v),
                         b * np.sin(u) * np.cos(v),
                         np.zeros_like(u)], -1)

    cen = r(Uu, Vv).reshape(-1, 3)
    cr = np.cross(ru(Uu, Vv).reshape(-1, 3), rv(Uu, Vv).reshape(-1, 3))
    area = np.linalg.norm(cr, axis=1) * du * dv
    nrm = cr / np.linalg.norm(cr, axis=1)[:, None]
    if float((nrm * cen).sum()) < 0.0:
        nrm = -nrm
    return cen, nrm, area


if __name__ == '__main__':
    print('=== C from the two independent routes ===')
    nx = n_axis(A_, B_, C_)
    px = dipole(A_, B_, C_)
    print('  n_x = %.6f   p_x = %.6f' % (nx, px))
    print('  C (no penetration) = Uabc/[2(1-n_x)] = %.6f' % (U * A_ * B_ * C_ / (2 * (1 - nx))))
    print('  C (far-field dipole) = 3 p_x / 2     = %.6f' % (1.5 * px))
    print()

    print('=== sphere limit check (a=b=c=1): Phi(2,0,0) must be 2.125 ===')
    print('  exact   = %.6f' % phi_exact(2.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    cen, n, area = make_panels(40, 80, 1.0, 1.0, 1.0)
    sig, _, _ = W.solve_sources(cen, n, area)
    print('  panel   = %.6f' % W.potential(np.array([[2.0, 0.0, 0.0]]), cen, area, sig)[0])
    print()

    # ---------------------------------------------------------- field compare
    nu, nv = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (32, 64)
    cen, n, area = make_panels(nu, nv)
    sig, _, _ = W.solve_sources(cen, n, area)
    print('=== triaxial ellipsoid (%.1f, %.1f, %.1f), %d panels ===' % (A_, B_, C_, len(cen)))

    print('  %-22s %12s %12s %10s' % ('sample point', 'exact', 'panel', 'rel.err'))
    pts = []
    for t in (1.25, 1.6, 2.2, 3.0, 4.0):
        pts.append((t * A_, 0.0, 0.0))
        pts.append((0.0, t * B_, 0.0))
        pts.append((0.0, 0.0, t * C_))
        pts.append((t * 0.6, t * 0.6, t * 0.6))
    P = np.array(pts)
    pe = np.array([phi_exact(*q) for q in pts])
    pp = W.potential(P, cen, area, sig)
    for q, e, p in zip(pts, pe, pp):
        print('  %-22s %12.6f %12.6f %9.2e'
              % ('(%.2f,%.2f,%.2f)' % q, e, p, abs(p - e) / max(abs(e), 1e-12)))

    # also compare the disturbance potential, where the panel error lives
    print()
    print('  disturbance potential phi_p = Phi - U x:')
    print('  %-22s %12s %12s %10s' % ('sample point', 'exact', 'panel', 'abs.err'))
    for q, e, p in zip(pts, pe, pp):
        print('  %-22s %12.6f %12.6f %9.2e'
              % ('(%.2f,%.2f,%.2f)' % q, e - q[0], p - q[0], abs((p - q[0]) - (e - q[0]))))

    # resolution study on a fixed probe
    print()
    print('  resolution study at (2.2, 0, 0) and (1.6,1.6,1.6):')
    q1 = np.array([[2.2, 0.0, 0.0], [1.6, 1.6, 1.6]])
    e1 = np.array([phi_exact(*q) for q in q1])
    for nu2, nv2 in ((12, 24), (20, 40), (32, 64), (48, 96)):
        cn, nn, aa = make_panels(nu2, nv2)
        sg, _, _ = W.solve_sources(cn, nn, aa)
        pp2 = W.potential(q1, cn, aa, sg)
        print('    %3dx%-3d  panels %5d   err  %8.2e  %8.2e'
              % (nu2, nv2, len(cn), abs(pp2[0] - e1[0]), abs(pp2[1] - e1[1])))
