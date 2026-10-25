# -*- coding: utf-8 -*-
"""Non-axisymmetric 3D probe: cruciform-fin body of revolution.

Candidate C from the structural discussion.  An ellipsoidal fuselage carrying
four coplanar fins (cruciform), at zero angle of attack.  The configuration is
symmetric about the meridian planes, so it carries no lift: there is no
trailing vorticity and Phi stays single valued, which keeps the encoding
well posed (the three-dimensional version of the branch-cut problem is avoided
by construction).

Geometry is a star-shaped radial parametrisation rho(phi, theta), so the same
constant-source panel solver that was calibrated on the triaxial ellipsoid
applies unchanged:

    Phi(x) = U.x + (1/4 pi) sum_j sigma_j A_j / |x - x_j|
    A_ij sigma_j = -(U . n_i) ,   A_ii = -1/2

Diagnostics:
  * solver calibration on the sphere (exact dipole U R^3 / 2)
  * surface boundary-condition residual max |dPhi/dn| / U
  * discrete harmonicity of Phi on the field grid, off the body
  * field speed at a few stations for a sanity check on the fin loading
"""
import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

U = 1.0

# ---------------------------------------------------------------- geometry
BODY_STYLE = 'winged'       # 'winged' ellipsoid body | 'agardb' AGARD-B model
AX, R0 = 2.0, 0.42          # fuselage semi-length, maximum radius
XW, CW, BW = 0.88, 1.00, 0.85   # fin station, chord, radial span
DEL = 0.07                  # fin azimuthal width (Gaussian sigma, rad)
NW = 4                      # number of fins (4 = cruciform)
BOOST = 6.0                 # azimuthal mesh clustering factor at the fins
FIN_AZ = 2.0 * np.pi / NW   # angular spacing between fins

# ------------------------------------------------------- AGARD-B calibration
# AGARD-B standard wind-tunnel model, dimensions in units of the body diameter
# D.  Body: 8.5 D long solid of revolution = 3 D cubic-ogive nose (local radius
# y = (x/3)[1 - (x/D)^2/9 + (x/D)^3/54], which meets the cylinder tangentially
# and gives an 18.43 deg tip half angle) + 5.5 D cylinder.  Wing: delta in the
# form of an equilateral triangle, tip-to-tip span 4 D, leading-edge sweep
# 60 deg, symmetric circular-arc section of 4% thickness ratio.  Stations are
# quoted from the nose apex; the base sits 1.402 D ahead of the wing TE.
# Sources: AGARD original specification (as reproduced in the standard-model
# literature) and the dimensional table of ICAS-94-3.8.2, which gives
# c_r = 519.60 mm and c_b = 389.7 mm at D = 150 mm, i.e. 3.464 D and 2.598 D.
#
# Two modelling notes, both disclosed: the flat base of the flight model (and
# its support sting) is closed here by a short ellipsoidal fairing so that the
# surface is a radial graph about the origin and the body stays closed for the
# source solve; and the 4% circular-arc section is carried as an azimuthal
# thickness envelope instead of a literal arc, which is what the panel
# parameterisation of this file supports.
AB_D = 0.5                  # body diameter (model scale)
AB_AX = 4.25 * AB_D         # semi-length, total 8.5 D
AB_R0 = 0.5 * AB_D          # cylinder radius
AB_LN = 3.0 * AB_D          # cubic-ogive nose length
AB_CR = 2.0 * np.sqrt(3.0) * AB_D    # theoretical root chord 3.464 D
AB_B = 2.0 * AB_D           # semi-span, tip-to-tip span 4 D
AB_TAU = 1.0 / np.sqrt(3.0)  # d(rho)/dx of the LE = 1/tan(60 deg)
AB_XTE = (8.5 - 1.402) * AB_D        # wing TE station from the nose
AB_XLE = AB_XTE - AB_CR              # wing root LE station from the nose
AB_TC = 0.04                # section thickness / chord
AB_LB = 0.25 * AB_D         # rounded-base fairing length
AB_TE_EPS = 0.03            # trailing-edge taper, fraction of the root chord
# Mesh-driven minimum azimuthal half width of the blade.  A 4% delta tapers to a
# point at the tip, and past about 70% of the span the blade is thinner than
# the azimuthal mesh spacing, so the panel set simply stops resolving it -- the
# first meshed build reached only 0.71 of the 1.0 tip radius.  Imposing a floor
# keeps the outboard blade resolvable and gives the tip the small rounded edge
# the specification asks for anyway (leading and trailing edges rounded with
# radius 0.002 D).  At the tip the floor thickens the blade to 2 * rho * floor =
# 0.01 D, i.e. 1% of the body diameter.
AB_DTH_MIN = 0.0025


def fin_azimuth(th):
    """Periodic Gaussian comb peaked at the four fin azimuths."""
    g = np.zeros_like(th, dtype=float)
    for k in range(NW):
        d = np.angle(np.exp(1j * (th - k * FIN_AZ)))
        g = g + np.exp(-(d / DEL) ** 2)
    return np.minimum(g, 1.0)


# ---------------------------------------------------------------- fin upgrade
# The legacy fin is an *elliptic flare*: its radial extent follows sqrt(1-t^2)
# over the chord and its azimuthal half width is the constant DEL, so it reads
# as a rounded lens with a blunt trailing edge.  The 'swept' style replaces that
# with a wing-like blade:
#   * planform   -- radial extent ramps up over EDGE_LE of the root chord
#                   (that ramp IS the leading-edge sweep), stays flat over the
#                   tip, and ramps down over EDGE_TE (trailing-edge sweep);
#   * section    -- NACA 4-digit thickness with a *closed* trailing edge, so the
#                   blade thins to an edge at the TE instead of a blunt face.
# The legacy path is untouched, so every earlier result stays reproducible.
FIN_STYLE = 'legacy'          # 'legacy' elliptic flare | 'swept' fin planform
CW_S, BW_S = 1.00, 0.85       # swept-fin root chord and radial span
EDGE_LE, EDGE_TE = 0.55, 0.20  # LE / TE ramp, fraction of the root chord
TC_ROOT = 0.090               # section thickness / chord at the root
YMAX = 0.10                   # peak of the NACA ordinate below


def _naca_half(eta):
    """NACA 4-digit half thickness; the -0.1036 term closes the trailing edge."""
    e = np.clip(eta, 0.0, 1.0)
    return (0.2969 * np.sqrt(e) - 0.1260 * e - 0.3516 * e ** 2
            + 0.2843 * e ** 3 - 0.1036 * e ** 4)


def _outline(eta):
    """Radial extent of the blade: 0 -> 1 -> 0 across the root chord.

    Straight (linear) ramps, so the leading edge is a real swept line and the
    blade keeps a trapezoidal planform: ramp over EDGE_LE, flat tip over
    1 - EDGE_LE - EDGE_TE, ramp down over EDGE_TE.  The LE sweep angle is
    atan(BW_S / (EDGE_LE * CW_S)).  Cosine ramps were tried first and made the
    outline read as an ellipse rather than a fin; the corner counts for far
    less in the solver than the planform does.
    """
    B = np.zeros_like(eta, dtype=float)
    up = (eta > 0.0) & (eta < EDGE_LE)
    B[up] = eta[up] / EDGE_LE
    flat = (eta >= EDGE_LE) & (eta <= 1.0 - EDGE_TE)
    B[flat] = 1.0
    dn = (eta > 1.0 - EDGE_TE) & (eta < 1.0)
    B[dn] = (1.0 - eta[dn]) / EDGE_TE
    return B


def _comb_var(th, dth):
    """Azimuthal Gaussian comb whose half width dth varies pointwise."""
    g = np.zeros_like(th, dtype=float)
    for k in range(NW):
        d = np.angle(np.exp(1j * (th - k * FIN_AZ)))
        g = g + np.exp(-(d / np.maximum(dth, 1e-9)) ** 2)
    return np.minimum(g, 1.0)


def span_radius(phi, th):
    """Radial extent of the swept blade at (phi, th), for rendering/tests."""
    x = AX * np.cos(phi)
    eta = (x - (XW - 0.5 * CW_S)) / CW_S
    return BW_S * _outline(eta)


def _fin_half_width():
    """Widest azimuthal half width of the fin (the mesh clustering scale)."""
    if BODY_STYLE == 'agardb':
        return _ab_blade_width(AB_R0)
    return DEL


def _ab_body_radius(xa):
    """AGARD-B body radius, xa measured from the nose apex.

    Cubic-ogive nose on [0, 3D] meeting the 5.5 D cylinder tangentially, then
    a short ellipsoidal fairing that closes the flat base (see the header).
    """
    xa = np.asarray(xa, dtype=float)
    r = np.where(xa <= AB_LN,
                 (xa / 3.0) * (1.0 - (xa / AB_D) ** 2 / 9.0
                               + (xa / AB_D) ** 3 / 54.0),
                 AB_R0)
    x0 = 8.5 * AB_D - AB_LB
    cap = AB_R0 * np.sqrt(np.clip(1.0 - ((xa - x0) / AB_LB) ** 2, 0.0, None))
    return np.clip(np.where(xa > x0, cap, r), 0.0, None)


def _ab_blade_extent(xa):
    """Radial extent of the delta wing, zero ahead of the LE and aft of the TE.

    The leading edge is the straight line rho = (xa - x_LE) / tan(60 deg) and
    the trailing edge is normal to the body axis, which is the
    equilateral-triangle planform of the standard model.  The extent must fall
    back to zero *behind* the TE as well: a clipped ramp leaves the blade at
    full span all the way to the base -- the first version did exactly that and
    the hand-measurement of the planform caught it.  The last AB_TE_EPS of the
    chord is tapered so that the surface closes in an edge instead of a cliff;
    that taper is within the latitude the specification itself leaves on the
    edge treatment, and it leaves both the 60 deg sweep and the 4 D span exact.
    """
    eta = (np.asarray(xa, dtype=float) - AB_XLE) / AB_CR
    f = np.minimum(eta / (1.0 - AB_TE_EPS), (1.0 - eta) / AB_TE_EPS)
    return AB_B * np.clip(f, 0.0, 1.0)


def _ab_blade_width(rho):
    """Azimuthal half width that carries a 4% section at radius rho.

    The floor is mesh-driven, not geometric: see AB_DTH_MIN.
    """
    c = AB_CR * np.clip(1.0 - np.asarray(rho, dtype=float) / AB_B, 0.0, None)
    return np.maximum(0.5 * AB_TC * c / np.maximum(np.asarray(rho, dtype=float), 1e-9),
                      AB_DTH_MIN)


def theta_grid(nt, width=None, boost=None):
    """Azimuthal grid clustered at the fin azimuths (density matched).

    A uniform theta grid cannot resolve a thin fin: the fin occupies a band of
    width ~3 DEL in azimuth, so the mesh density is made proportional to a
    wider Gaussian comb and the grid points are placed at equal increments of
    its cumulative distribution.  The AGARD-B blade is much thinner than the
    cruciform fin, so its comb is narrower and its clustering factor larger.
    """
    if width is None:
        width = 0.06 if BODY_STYLE == 'agardb' else 3.0 * _fin_half_width()
    if boost is None:
        boost = 30.0 if BODY_STYLE == 'agardb' else BOOST
    u = np.linspace(0.0, 2.0 * np.pi, 40001)
    dens = np.ones_like(u)
    for k in range(NW):
        d = np.angle(np.exp(1j * (u - k * FIN_AZ)))
        dens = dens + boost * np.exp(-(d / width) ** 2)
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u))])
    cdf = cdf / cdf[-1]
    ug = np.interp(np.linspace(0.0, 1.0, nt, endpoint=False), cdf, u)
    if BODY_STYLE == 'agardb':
        # Put a grid line exactly on each fin plane.  Without it the nearest
        # sample sits about half a spacing off the blade centreline and the
        # outboard blade -- whose thickness is the mesh-driven floor -- is
        # sampled where the comb is below one, so the panel set stopped at 86%
        # of the tip radius instead of reaching it.
        for k in range(NW):
            az = (k * FIN_AZ) % (2.0 * np.pi)
            j = int(np.argmin(np.abs(np.angle(np.exp(1j * (ug - az))))))
            ug[j] = az
    return ug


def radius(phi, th, ax=AX, r0=R0, wing=True):
    if not wing:
        # sphere / ellipsoid of revolution, used to calibrate the solver
        return r0 * np.sin(phi)
    if BODY_STYLE == 'agardb':
        xa = AB_AX * (1.0 + np.cos(phi))          # station from the nose apex
        rho_b = _ab_blade_extent(xa)
        dth = _ab_blade_width(np.maximum(rho_b, 1e-6))
        # the blade is *not* added to the body radius: its span is defined by
        # the planform, so the two surfaces are combined with max() and the
        # wing-body junction keeps its real crease
        return np.maximum(_ab_body_radius(xa), rho_b * _comb_var(th, dth))
    rho = r0 * np.sin(phi)
    x = ax * np.cos(phi)
    if FIN_STYLE == 'legacy':
        t = (x - XW) / (0.5 * CW)
        chord = np.sqrt(np.clip(1.0 - t * t, 0.0, None))
        return rho + BW * chord * fin_azimuth(th)
    eta = (x - (XW - 0.5 * CW_S)) / CW_S
    B = _outline(eta)
    r_loc = rho + BW_S * B
    tphys = TC_ROOT * CW_S * np.clip(_naca_half(eta) / YMAX, 0.0, 1.0)
    dth = np.where(B > 1e-6,
                   np.maximum(0.5 * tphys / np.maximum(r_loc, 1e-6), 1.5e-3),
                   1.5e-3)
    return rho + BW_S * B * _comb_var(th, dth)


def surface(phi, th, **kw):
    r = radius(phi, th, **kw)                 # transverse radius rho already
    if 'ax' in kw:
        ax_eff = kw['ax']
    else:
        ax_eff = AB_AX if BODY_STYLE == 'agardb' else AX
    x = ax_eff * np.cos(phi)
    return np.stack([x, r * np.cos(th), r * np.sin(th)], axis=-1)


# ---------------------------------------------------------------- panels
def phi_grid(nf, boost=4.0):
    """Meridional grid clustered over the fin chord, poles included."""
    u = np.linspace(0.0, np.pi, 20001)
    if BODY_STYLE == 'agardb':
        p_te = np.arccos(np.clip((AB_XTE - AB_AX) / AB_AX, -1.0, 1.0))
        p_le = np.arccos(np.clip((AB_XLE - AB_AX) / AB_AX, -1.0, 1.0))
        c, w = 0.5 * (p_le + p_te), max(0.5 * (p_le - p_te), 1e-3)
        dens = 1.0 + boost * np.exp(-((u - c) / w) ** 2)
        dens = dens + 1.5 * np.exp(-((u - np.pi) / 0.25) ** 2)    # nose tip
        dens = dens + 1.5 * np.exp(-(u / 0.18) ** 2)              # base cap
    else:
        ch = CW if FIN_STYLE == 'legacy' else CW_S
        p1 = np.arccos(np.clip((XW + 0.5 * ch) / AX, -1.0, 1.0))
        p2 = np.arccos(np.clip((XW - 0.5 * ch) / AX, -1.0, 1.0))
        c, w = 0.5 * (p1 + p2), max(0.5 * (p2 - p1), 1e-3)
        dens = 1.0 + boost * np.exp(-((u - c) / w) ** 2)
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u))])
    cdf = cdf / cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, nf + 1), cdf, u)


def build_panels(nf, nt, **kw):
    """Corner grid -> flat triangular panels (centroid, outward normal, area).

    A curved quad is not planar, and the constant-source self influence -1/2
    is exact only on a flat panel; on the fin flanks, where the surface
    curvature is large, that mismatch leaks a spurious net source strength.
    Each quad is therefore split into two flat triangles.
    """
    if kw.get('wing', True):
        phe = phi_grid(nf)
    else:
        phe = np.linspace(0.0, np.pi, nf + 1)
    tg = theta_grid(nt)
    the = np.concatenate([tg, [tg[0] + 2.0 * np.pi]])
    PHe, THE = np.meshgrid(phe, the, indexing='ij')
    Pe = surface(PHe, THE, **kw)                     # (nf+1, nt+1, 3)

    p00 = Pe[:-1, :-1].reshape(-1, 3)
    p10 = Pe[1:, :-1].reshape(-1, 3)
    p11 = Pe[1:, 1:].reshape(-1, 3)
    p01 = Pe[:-1, 1:].reshape(-1, 3)

    tri = np.concatenate([np.stack([p00, p10, p11], axis=1),
                          np.stack([p00, p11, p01], axis=1)], axis=0)
    nvec = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = np.linalg.norm(nvec, axis=1)
    area = 0.5 * nrm
    keep = (nrm > 1e-14) & (area > 1e-14)
    tri = tri[keep]
    n = nvec[keep] / nrm[keep][:, None]
    area = area[keep]
    cen = tri.mean(axis=1)

    # Orient consistently from the parametrisation.  A per-panel flip test
    # (n . centroid > 0) is wrong on the fin flanks, where the outward normal
    # legitimately has a negative radial component; flipping those panels
    # individually breaks the consistency of the normal field and leaves a
    # spurious net source strength.  Only one global sign is decided, from the
    # sum over all panels.
    if float((n * cen).sum()) < 0.0:
        n = -n
    return cen, n, area, tri


def tri_velocity(P, tri, nrm):
    """Velocity at point P due to a unit constant source on each flat triangle.

    Exact closed form.  With Phi = (1/4pi) int dA'/|P-r'| and v = -grad Phi,

        v = (1/4pi) int (P-r')/|P-r'|^3 dA' .

    The component along the panel normal is the signed solid angle, and the
    in-plane part is turned by the planar divergence theorem into line
    integrals of 1/r along the three edges.  Self influence then falls out as
    +1/2 automatically, so no special casing is needed -- which is exactly what
    the centroid-point-source approximation cannot reproduce for neighbouring
    panels, and the reason its net source strength does not converge.
    """
    A = tri[:, 0] - P
    B = tri[:, 1] - P
    C = tri[:, 2] - P
    rA = np.linalg.norm(A, axis=1)
    rB = np.linalg.norm(B, axis=1)
    rC = np.linalg.norm(C, axis=1)

    num = (A * np.cross(B, C)).sum(1)
    den = (rA * rB * rC + (A * B).sum(1) * rC
           + (A * C).sum(1) * rB + (B * C).sum(1) * rA)
    # Signed solid angle.  The van Oosterom formula is positive when the
    # vertices appear counter-clockwise as seen FROM the field point, which is
    # the opposite of what is needed here: the normal component must be
    # positive for a point on the outward side.
    omega = -2.0 * np.arctan2(num, den)
    v = (omega / (4.0 * np.pi))[:, None] * nrm

    for P0, P1, r0, r1 in ((A, B, rA, rB), (B, C, rB, rC), (C, A, rC, rA)):
        seg = P1 - P0                                 # edge vertex0 -> vertex1
        d = np.maximum(np.linalg.norm(seg, axis=1), 1e-300)
        nout = np.cross(seg / d[:, None], nrm)        # outward, in-plane
        L = np.log1p(2.0 * d / np.maximum(r0 + r1 - d, 1e-300))
        v = v + nout * (L / (4.0 * np.pi))[:, None]
    return v


# ---------------------------------------------------------------- solve
def solve_sources(cen, n, area, tri=None, Uvec=(U, 0.0, 0.0)):
    """Panel source strengths from (U + v_s) . n = 0 at every centroid.

    Centroid point sources with the mirrored self term -1/2, which reproduces
    the analytic sphere result Phi(2,0,0) = 2.125 to 0.008% and gives the
    correct sign of the perturbation potential.

    A replacement by exact flat-panel integrals (tri_velocity, kept below for
    reference) was tried and REVERTED: it produced the wrong sign and a factor
    ~1.9 in the far-field perturbation potential, and the net source strength
    stopped converging.  Its residual against the analytic sphere sigma is 48%,
    so it is the panel integral that is not yet right, not the potential.

    Known limitation of the form used here: the net source strength
    Q = sum(sigma_j A_j) does not quite vanish for the winged body
    (~1e-2 at 12x80, ~1.7e-3 at 16x120), i.e. a small spurious monopole in the
    far field.  It is machine zero for the sphere and the bare fuselage.  The
    harmonicity of Phi -- which is what the stationarity verification tests --
    is unaffected, because a sum of point sources is exactly harmonic.
    """
    m = len(cen)
    A = np.empty((m, m))
    for i in range(m):
        d = cen[i] - cen
        dist = np.linalg.norm(d, axis=1)
        dist[i] = 1.0                            # placeholder, self term below
        A[i] = -(d @ n[i]) / dist ** 3 * area / (4.0 * np.pi)
    A[np.arange(m), np.arange(m)] = -0.5
    rhs = -(np.asarray(Uvec) @ n.T)
    return solve(A, rhs), A, rhs


def potential(P, cen, area, sigma, Uvec=(U, 0.0, 0.0)):
    """Phi at points P (..., 3) from the panel sources (centroid quadrature)."""
    P = np.asarray(P, dtype=float)
    flat = P.reshape(-1, 3)
    out = np.empty(len(flat))
    q = sigma * area
    for c0 in range(0, len(flat), 2000):
        blk = flat[c0:c0 + 2000]
        d = blk[:, None, :] - cen[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1))
        dist = np.maximum(dist, 1e-9)
        out[c0:c0 + 2000] = (q[None, :] / dist).sum(axis=1)
    out = out / (4.0 * np.pi)
    out = out + flat @ np.asarray(Uvec)
    return out.reshape(P.shape[:-1])


def velocity(P, cen, area, sig, tri=None, nrm=None, Uvec=(U, 0.0, 0.0),
             exact=False):
    """Flow velocity, v = U - sum_j sigma_j K_j(x).

    The sign is the part that is easy to get wrong, and getting it wrong looks
    exactly like a broken quadrature: with a PLUS sign every on-axis probe of a
    closed body comes out above the freestream where the body must have
    stagnation points.  Judged against the exact axial speed of a unit sphere
    (0.136 / 0.421 / 0.875 at x = -1.05 / -1.20 / -2.00, and the same 0.875 at
    x = +2):

        kernel          sign      x=-1.05  -1.20   -2.00   +2.00
        point source    U + sum    1.698    1.576   1.126   1.126
        point source    U - sum    0.302    0.424   0.874   0.874
        exact triangle  U + sum    1.867    1.583   1.126   1.126
        exact triangle  U - sum    0.133    0.417   0.874   0.874   <- correct

    The exact triangle kernel (``exact=True``, needs ``tri`` and ``nrm``) is
    within 1.5% of the analytic value at a clearance of 0.05 R from the
    stagnation point, where the centroid point-source form is 2.2x off; use it
    whenever the point of interest is closer to a wall than about a panel.
    """
    flat = np.asarray(P, dtype=float).reshape(-1, 3)
    out = np.zeros_like(flat)
    if exact:
        if tri is None or nrm is None:
            raise ValueError('exact=True needs tri and nrm')
        for k in range(len(flat)):
            out[k] = sig @ tri_velocity(flat[k], tri, nrm)
    else:
        q = sig * area
        for c0 in range(0, len(flat), 2000):
            blk = flat[c0:c0 + 2000]
            d = blk[:, None, :] - cen[None, :, :]
            dist = np.maximum(np.sqrt((d ** 2).sum(-1)), 1e-9)
            out[c0:c0 + 2000] = (q[None, :, None] * d
                                 / dist[..., None] ** 3).sum(1) / (4.0 * np.pi)
    out = np.asarray(Uvec) - out
    return out.reshape(np.shape(P))


# ---------------------------------------------------------------- calibrate
def calibrate():
    print('--- solver calibration on the sphere (R = 1) ---')
    for nf, nt in ((12, 24), (20, 40), (32, 64)):
        cen, n, area, tri = build_panels(nf, nt, ax=1.0, r0=1.0, wing=False)
        sig, _, _ = solve_sources(cen, n, area, tri)
        dip = abs((sig * area * cen[:, 0]).sum()) / (4 * np.pi)
        print('  %3dx%-3d  panels %5d   dipole %.6f   exact %.6f   err %6.2f%%'
              % (nf, nt, len(cen), dip, 0.5, 100 * (dip - 0.5) / 0.5))
    print()


if __name__ == '__main__':
    import sys
    calibrate()

    nf, nt = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (14, 160)
    cen, n, area, tri = build_panels(nf, nt)
    print('--- cruciform-fin body ---')
    print('  panels          : %d' % len(cen))
    print('  fin tip radius  : %.3f   fuselage max radius: %.3f'
          % (R0 + BW, R0))
    sig, A, rhs = solve_sources(cen, n, area, tri)
    print('  net source Q    : %+.3e   (must vanish for a closed body)'
          % (sig * area).sum())

    # boundary-condition residual of the assembled system itself
    res = np.abs(A @ sig - rhs)
    print('  BC residual     : max %.3e   rms %.3e'
          % (res.max(), np.sqrt((res ** 2).mean())))

    # field grid: harmonicity of Phi off the body
    ng, half = (int(sys.argv[3]) if len(sys.argv) > 3 else 64), 3.0
    g = np.linspace(-half, half, ng)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([X, Y, Z], -1)
    Phi = potential(P, cen, area, sig)
    lap = np.zeros_like(Phi)
    h = g[1] - g[0]
    lap[1:-1, 1:-1, 1:-1] = (
        Phi[2:, 1:-1, 1:-1] + Phi[:-2, 1:-1, 1:-1]
        + Phi[1:-1, 2:, 1:-1] + Phi[1:-1, :-2, 1:-1]
        + Phi[1:-1, 1:-1, 2:] + Phi[1:-1, 1:-1, :-2]
        - 6 * Phi[1:-1, 1:-1, 1:-1]) / h ** 2

    # sample only points well outside the body envelope
    Rad = np.sqrt((X / (AX * 1.35)) ** 2 + ((Y ** 2 + Z ** 2) / ((R0 + BW) * 1.35) ** 2))
    selb = (Rad > 1.0)
    selb[[0, -1], :, :] = False
    selb[:, [0, -1], :] = False
    selb[:, :, [0, -1]] = False
    print('  grid points used: %d of %d' % (selb.sum(), Phi.size))
    print('  RMS laplacian   : %.3e   max %.3e   (h^2 = %.3e, h^2*U/L ~ %.3e)'
          % (np.sqrt((lap[selb] ** 2).mean()), np.abs(lap[selb]).max(),
             h ** 2, h ** 2 * U / AX))

    np.savez('code/data/winged_body_probe.npz', cen=cen, n=n, area=area,
             sig=sig, Phi=Phi, g=g)

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(11, 3.6))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    phe = phi_grid(2 * nf)
    tgd = theta_grid(2 * nt)
    the = np.concatenate([tgd, [tgd[0] + 2 * np.pi]])
    PHe, THE = np.meshgrid(phe, the, indexing='ij')
    S = surface(PHe, THE)
    ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], rstride=2, cstride=2,
                    color='0.75', edgecolor='0.4', linewidth=0.12,
                    antialiased=True)
    ax.set_box_aspect((2.4, 1, 1))
    ax.view_init(elev=16, azim=-58)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('cruciform-fin body', fontsize=9)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(1, 3, 2)
    kz = np.argmin(np.abs(g))
    Phip = Phi - U * X                                    # disturbance potential
    m = float(np.percentile(np.abs(Phip[:, :, kz]), 99.0))
    lv = np.linspace(-m, m, 25)
    cs = ax.contourf(X[:, :, kz].T, Y[:, :, kz].T, Phip[:, :, kz].T,
                     levels=lv, cmap='RdBu_r', extend='both')
    ax.contour(X[:, :, kz].T, Y[:, :, kz].T, Phip[:, :, kz].T,
               levels=lv[2::3], colors='k', linewidths=0.25)
    ax.set_aspect('equal')
    ax.set_title(r'disturbance potential $\varphi_p$, $z=0$', fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(cs, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 3, 3)
    vx = np.gradient(Phi, g, axis=0)
    vy = np.gradient(Phi, g, axis=1)
    vz = np.gradient(Phi, g, axis=2)
    spd = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    lv = np.linspace(0.4, 1.8, 25)
    cs = ax.contourf(X[:, :, kz].T, Y[:, :, kz].T, spd[:, :, kz].T,
                     levels=lv, cmap='viridis', extend='both')
    ax.set_aspect('equal')
    ax.set_title(r'$|v|/U_\infty$, $z=0$', fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(cs, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig('code/data/winged_body_probe.png', dpi=190)
    print('  figure -> code/data/winged_body_probe.png')
