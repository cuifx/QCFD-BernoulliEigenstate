"""
High-precision 2D panel method (source panels + prescribed Kutta circulation)
==============================================================================
Independent classical baseline for the H_B airfoil validation figure.

The circulation is NOT solved by a numerical Kutta condition (which is
delicate for cusped/rounded trailing edges) but prescribed at the exact
potential-flow value Gamma = 4*pi*R*U*sin(alpha) (or the camber formula for
the Joukowski), matching the analytic circulation superposed classically in
airfoil_hb_quantum.py / airfoil_validation_composite.py.  Given the body, the
freestream and the circulation Gamma, the potential flow around the airfoil is
UNIQUE, so the panel method reproduces the analytic conformal-map solution to
the panel discretization error regardless of the interior vortex position.

Formulation:
  * N constant-strength SOURCE panels (unknown sigma_1..sigma_N) enforce
    no-penetration at the control points;
  * a point VORTEX of strength Gamma sits inside the airfoil (the centroid)
    and carries the circulation;
  * BC:  U.n_i + sum_j sigma_j A_n[i,j] + Gamma * B_n[i] = 0,
    where B_n[i] is the normal velocity at control point i of a unit vortex;
  * field velocity = U + sources + Gamma * (point vortex).

Airfoil boundary convention: N+1 complex nodes, CLOCKWISE, starting at the
trailing edge (upper surface first, then lower surface back to the TE).
"""
import numpy as np


def order_boundary(z, tol=1e-12):
    """Normalize a closed boundary curve to clockwise order starting at the
    TE (rightmost node).  z: (N,) complex array, either orientation."""
    z = np.asarray(z)
    x, y = z.real, z.imag
    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if A > tol:                     # positive area -> counterclockwise
        z = z[::-1]
    ite = int(np.argmax(z.real))
    return np.roll(z, -ite)


def _panel_geom(nodes):
    """Return (lengths, tangents, normals, centers) for the closed polygon."""
    seg = nodes[1:] - nodes[:-1]
    L = np.abs(seg)
    tc = seg / np.maximum(L, 1e-300)
    t = np.column_stack([tc.real, tc.imag])      # (N, 2)
    n = np.column_stack([-tc.imag, tc.real])     # (t_y, -t_x) = outward (CCW rot)
    centers = 0.5 * (nodes[:-1] + nodes[1:])
    return L, t, n, centers


def source_vel(X, Y, x1, y1, x2, y2, t, n):
    """Velocity at (X, Y) due to a unit constant SOURCE panel [1->2]."""
    tx, ty = t
    nx, ny = n
    lx = (X - x1) * tx + (Y - y1) * ty
    ly = (X - x1) * nx + (Y - y1) * ny
    L = np.hypot(x2 - x1, y2 - y1)
    r1sq = np.maximum(lx * lx + ly * ly, 1e-24)
    r2sq = np.maximum((lx - L) ** 2 + ly * ly, 1e-24)
    th1 = np.arctan2(ly, lx)
    th2 = np.arctan2(ly, lx - L)
    u = np.log(r1sq / r2sq) / (4 * np.pi)
    v = (th2 - th1) / (2 * np.pi)
    return u * tx + v * nx, u * ty + v * ny


def point_vortex_vel(X, Y, xv, yv):
    """Velocity at (X, Y) due to a unit (Gamma=1, CCW-positive) point vortex."""
    dx = X - xv
    dy = Y - yv
    r2 = np.maximum(dx * dx + dy * dy, 1e-24)
    return -dy / (2 * np.pi * r2), dx / (2 * np.pi * r2)


def solve(airfoil, alpha, U=1.0, Gamma=None, vortex_pos=None):
    """Solve source panels with a prescribed point-vortex circulation.

    airfoil : (N+1,) complex nodes, clockwise, TE first.
    alpha   : angle of attack (rad).
    Gamma   : signed circulation (defaults to the analytic Kutta value used
              by airfoil_hb_quantum.py if None -- i.e. caller must pass it,
              since the airfoil->Gamma mapping differs per geometry).
    vortex_pos : complex interior point (default: boundary centroid).
    Returns dict(sigma, Gamma, vortex_pos, L, t, n, centers).
    """
    if Gamma is None:
        raise ValueError("Gamma must be prescribed (analytic Kutta value); "
                         "pass it explicitly.")
    nodes = np.asarray(airfoil)
    N = len(nodes) - 1
    L, t, n, centers = _panel_geom(nodes)
    cx, cy = centers.real, centers.imag
    if vortex_pos is None:
        vortex_pos = np.mean(centers)

    # source normal influence matrix (self-influence +1/2 on the diagonal)
    An = np.zeros((N, N))
    for j in range(N):
        x1, y1 = nodes[j].real, nodes[j].imag
        x2, y2 = nodes[j + 1].real, nodes[j + 1].imag
        u, v = source_vel(cx, cy, x1, y1, x2, y2, t[j], n[j])
        An[:, j] = u * n[:, 0] + v * n[:, 1]
    for i in range(N):
        An[i, i] = 0.5

    # point-vortex normal influence at the control points
    vxv, vyv = point_vortex_vel(cx, cy, vortex_pos.real, vortex_pos.imag)
    Bn = vxv * n[:, 0] + vyv * n[:, 1]

    ca, sa = np.cos(alpha), np.sin(alpha)
    un = U * (ca * n[:, 0] + sa * n[:, 1])
    sigma = np.linalg.solve(An, -un - Gamma * Bn)
    return dict(sigma=sigma, Gamma=Gamma, vortex_pos=vortex_pos,
                L=L, t=t, n=n, centers=centers)


def field_velocity(X, Y, airfoil, sol, alpha, U=1.0):
    """Velocity (vx, vy) at field points (X, Y): U + sources + Gamma vortex."""
    nodes = np.asarray(airfoil)
    N = len(nodes) - 1
    L, t, n, centers = sol['L'], sol['t'], sol['n'], sol['centers']
    sigma, Gamma = sol['sigma'], sol['Gamma']
    vp = sol['vortex_pos']
    X, Y = np.asarray(X), np.asarray(Y)
    ca, sa = np.cos(alpha), np.sin(alpha)
    vx = np.full_like(X, U * ca, dtype=float)
    vy = np.full_like(Y, U * sa, dtype=float)
    for j in range(N):
        x1, y1 = nodes[j].real, nodes[j].imag
        x2, y2 = nodes[j + 1].real, nodes[j + 1].imag
        u, v = source_vel(X, Y, x1, y1, x2, y2, t[j], n[j])
        vx = vx + sigma[j] * u
        vy = vy + sigma[j] * v
    u, v = point_vortex_vel(X, Y, vp.real, vp.imag)
    vx = vx + Gamma * u
    vy = vy + Gamma * v
    return vx, vy
