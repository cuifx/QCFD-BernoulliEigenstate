# -*- coding: utf-8 -*-
"""Orthographic surface renderer with an isotropy self-check.

Why not mplot3d: it maps every data range onto its own box dimension and its
default view stretches the model badly (the 8.5 D x 1 D AGARD-B body came out
looking like a fat peanut), and with box_aspect it is easy to stretch one axis
by tens of percent without noticing.  Here the projection is written out,

    screen right  r = (-sin a, cos a, 0)
    screen up     u = (-sin e cos a, -sin e sin a, cos e)
    depth         v = ( cos e cos a,  cos e sin a, sin e)

the quads are drawn back to front (painter's algorithm) with Lambert shading,
and the two screen axes are plotted on equal scales.  ``isotropy_check``
measures the pixels-per-unit along both axes afterwards, so the claim "the
picture has the proportions of the model" is a number rather than an opinion.

Two practical notes:
  * azimuth -90 deg is the only value for which the body axis (along x here)
    projects exactly onto the screen horizontal -- at other azimuths the body
    tilts as the elevation rises, which reads as an angle of attack the model
    does not have;
  * a thin wing is a sheet whose two sides are almost coincident in depth, so
    the depth sort flickers and the plate comes out saw-toothed unless the
    shading contrast is reduced (``contrast``).
"""
import numpy as np
from matplotlib.collections import PolyCollection

LIGHT = np.array([-0.35, -0.45, 0.82])


def ortho(S, elev, azim):
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    r = np.array([-np.sin(a), np.cos(a), 0.0])
    u = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    P = np.stack(S, -1)
    return P @ r, P @ u, P @ v, r, u, v


def normals(S):
    X, Y, Z = S
    t0 = np.stack([np.gradient(A, axis=0) for A in (X, Y, Z)], -1)
    t1 = np.stack([np.gradient(A, axis=1) for A in (X, Y, Z)], -1)
    n = np.cross(t0, t1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)


def draw_surface(ax, S, elev, azim, base=(0.62, 0.68, 0.76), val=None,
                 cmap='turbo', vmin=None, vmax=None, contrast=0.35,
                 n_stride=1, gamma=1.0, norm=None):
    """Draw the surface.  If ``val`` is given it colours the faces instead of
    ``base``; ``val`` has the shape of the surface lattice, one scalar per
    vertex, and is averaged over each quad.

    Pass ``norm`` (a matplotlib Normalize) whenever the field needs a
    non-linear scale.  Without it the faces are mapped LINEARLY between
    ``vmin`` and ``vmax``, so a LogNorm given only to the colorbar leaves every
    face in the first colour of the map: a field whose median is a hundredth of
    its 99th percentile comes out uniformly black while its colorbar claims
    three decades of range.
    """
    sx, sy, dep, r, u, v = ortho(S, elev, azim)
    n = normals(S)
    nl = np.stack([(n @ r), (n @ u), (n @ v)], -1)
    lo = 1.0 - (1.0 - 0.30) * contrast
    lam = np.clip(np.abs(nl @ LIGHT), lo, 1.0)
    if val is not None:
        import matplotlib.pyplot as plt
        cm = plt.get_cmap(cmap)
        if norm is not None:
            t = np.ma.filled(np.clip(norm(val), 0.0, 1.0), 0.0) ** gamma
        else:
            vmin = np.nanmin(val) if vmin is None else vmin
            vmax = np.nanmax(val) if vmax is None else vmax
            t = np.clip((val - vmin) / max(vmax - vmin, 1e-30), 0, 1) ** gamma

    ni, nj = sx.shape[0] - 1, sx.shape[1] - 1
    quads, cols, depth = [], [], []
    for i in range(0, ni, n_stride):
        for j in range(0, nj, n_stride):
            quads.append([(sx[i, j], sy[i, j]), (sx[i + 1, j], sy[i + 1, j]),
                          (sx[i + 1, j + 1], sy[i + 1, j + 1]),
                          (sx[i, j + 1], sy[i, j + 1])])
            g = 0.5 * (lam[i, j] + lam[min(i + 1, ni), min(j + 1, nj)])
            if val is None:
                rgb = np.array(base)
            else:
                rgb = np.array(cm(0.25 * (t[i, j] + t[i + 1, j] + t[i + 1, j + 1]
                                          + t[i, j + 1]))[:3])
            cols.append(np.clip(rgb * g, 0, 1))
            depth.append(0.25 * (dep[i, j] + dep[i + 1, j] + dep[i + 1, j + 1]
                                 + dep[i, j + 1]))
    order = np.argsort(depth)
    ax.add_collection(PolyCollection(
        [quads[k] for k in order], facecolors=[cols[k] for k in order],
        edgecolors='none', antialiased=False))
    pad = 0.05 * max(sx.max() - sx.min(), sy.max() - sy.min())
    ax.set_xlim(sx.min() - pad, sx.max() + pad)
    ax.set_ylim(sy.min() - pad, sy.max() + pad)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return isotropy_check(ax)


def draw_mesh(ax, tri, elev, azim, val=None, cmap='turbo', vmin=None,
              vmax=None, contrast=0.45, base=(0.70, 0.72, 0.78)):
    """Draw a triangle mesh (m, 3, 3), one flat colour per face.

    Preferable to a shell lattice whenever the body has a thin component: the
    mesh *is* the geometry, so a 4% plate is drawn as the plate it is and no
    offset normal can flip from one side of it to the other.
    """
    P = np.asarray(tri, dtype=float)
    V = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    r = np.array([-np.sin(a), np.cos(a), 0.0])
    u = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    nl = np.stack([V @ r, V @ u, V @ v], -1)
    lo = 1.0 - (1.0 - 0.30) * contrast
    lam = np.clip(np.abs(nl @ LIGHT), lo, 1.0)
    sx, sy, dep = (P @ r), (P @ u), (P @ v)
    if val is not None:
        import matplotlib.pyplot as plt
        cm = plt.get_cmap(cmap)
        vmin = np.nanmin(val) if vmin is None else vmin
        vmax = np.nanmax(val) if vmax is None else vmax
        t = np.clip((np.asarray(val) - vmin) / max(vmax - vmin, 1e-30), 0, 1)
        cols = cm(t)[:, :3] * lam[:, None]
    else:
        cols = np.array(base)[None, :] * lam[:, None]
    order = np.argsort(dep.mean(axis=1))
    ax.add_collection(PolyCollection(
        [np.stack([sx[k], sy[k]], -1) for k in order],
        facecolors=np.clip(cols[order], 0, 1), edgecolors='none',
        antialiased=False))
    pad = 0.05 * max(sx.max() - sx.min(), sy.max() - sy.min())
    ax.set_xlim(sx.min() - pad, sx.max() + pad)
    ax.set_ylim(sy.min() - pad, sy.max() + pad)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return isotropy_check(ax)


def isotropy_check(ax):
    """Pixels per data unit along x and along y; they must be equal."""
    ax.figure.canvas.draw()
    tr = ax.transData
    x0, y0 = tr.transform((0.0, 0.0))
    x1, _ = tr.transform((1.0, 0.0))
    _, y1 = tr.transform((0.0, 1.0))
    return (x1 - x0) / (y1 - y0)
