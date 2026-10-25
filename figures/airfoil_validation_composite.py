"""
Three-airfoil application snapshot (main-text figure)
=====================================================
The conventional free-particle scheme's |V| at T = 0.01 in the physical
(z) plane for Joukowski, NACA 0012, and NACA 2412.

Per-panel pipeline (identical to the manuscript's conventional pipeline):
  1. canonical cylinder flow in the zeta plane (uniform flow + doublet
     + Kutta circulation);
  2. free-particle Schrödinger evolution to T = 0.01 (FFT spectral, with
     the same spectral filter as the manuscript);
  3. Madelung decoding of the velocity, circulation added back;
  4. conformal map to the physical plane (Joukowski: closed form
     z = zeta + 1/zeta; NACA: Theodorsen Laurent series), the velocity
     transformed by dz/dzeta;
  5. relative error over the window-effective exterior (same metric as
     the manuscript tables).

Output: fig/airfoil_validation.pdf  (1x3 panels, shared color scale).
"""
import os, sys, time
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

from flow_common import smooth_boundary_window, madelung_encode, madelung_decode

def _prepare_velocity_grid(a, dilate=2):
    """NaN-safe velocity grid for RegularGridInterpolator. The cylinder
    interior carries undefined velocities (NaN); scipy's interpolator would
    spread those NaNs into neighbouring interpolation points. Dilation of the
    NaN region by `dilate` grid points plus nearest-finite fill keeps
    interpolation points near the body away from the undefined interior
    values, which would otherwise paint spurious low-speed (dark blue)
    pixels around the airfoil."""
    from scipy.ndimage import binary_dilation, distance_transform_edt
    b = np.array(a, dtype=float)
    mask = ~np.isfinite(b)
    if not mask.any():
        return b
    mask = binary_dilation(mask, iterations=dilate)
    b[mask] = np.nan
    inds = distance_transform_edt(mask, return_distances=False,
                                  return_indices=True)
    return b[tuple(inds)]

U_INF, HBAR, MASS = 1.0, 1.0, 1.0
ZETA_RANGE = 5.0
PLOT_LIM = 4.4            # window-effective domain: |zeta| < 4.4 => window == 1
N_EVOLVE = 128
DT = 0.001
N_STEPS = 10              # T = 0.01
ALPHA = np.radians(5.0)   # angle of attack for the NACA panels

def inverse_joukowski_map(z, zc, a, lim, af_x=None, af_y=None):
    """Inverse Joukowski map zeta = (z +/- sqrt(z^2-4))/2, keeping the
    branch |zeta| >= 1 (exterior). Returns (zeta, valid-mask)."""
    s = np.sqrt(z ** 2 - 4)
    z_p = 0.5 * (z + s)
    zeta = np.where(np.abs(z_p) >= 1.0, z_p, 0.5 * (z - s))
    valid = (np.abs(zeta.real) <= lim) & (np.abs(zeta.imag) <= lim)
    if af_x is not None and af_y is not None:
        inside = Path(np.column_stack([af_x, af_y])).contains_points(
            np.column_stack([z.real.ravel(), z.imag.ravel()])).reshape(z.shape)
        valid = valid & ~inside
    else:
        valid = valid & (np.abs(zeta - zc) > a + 1e-8)
    return zeta, valid

def zeta_cylinder_scheme(R, zc, gamma, alpha):
    """Free-particle scheme on the zeta-plane cylinder at T = 0.01.

    Uniform convention (matches the manuscript's NACA pipeline):
    circulation is added back as u -= gamma*y/(2pi r^2), v += gamma*x/(2pi r^2)
    and the analytic reference uses utheta = -U(1+R^2/r^2) sin(th-alpha)
    + gamma/(2 pi r). For the Joukowski panel pass gamma < 0, which reproduces
    the (already validated) Joukowski pipeline exactly.
    """
    ne = N_EVOLVE
    dx = 2 * ZETA_RANGE / ne
    zx = (np.arange(ne) - ne // 2) * dx
    XX, YY = np.meshgrid(zx, zx)
    ZS = (XX - zc.real) + 1j * (YY - zc.imag)
    r = np.abs(ZS)
    th = np.angle(ZS)
    inside = r < R

    # encode (potential carries no circulation)
    phi = U_INF * (r + R ** 2 / np.maximum(r, 1e-9)) * np.cos(th - alpha)
    window = smooth_boundary_window(XX, YY, ZETA_RANGE)
    psi = madelung_encode(phi) * window
    psi[inside] = 0.0

    kx = 2 * np.pi * fftfreq(ne)
    KX, KY = np.meshgrid(kx, kx)
    k_sq = KX ** 2 + KY ** 2
    k_cut = np.pi * 32 / (2 * ZETA_RANGE)
    filt = np.exp(-(k_sq / k_cut ** 2) ** 2)
    psi_k = fft2(psi) * np.exp(-1j * k_sq * N_STEPS * DT / (2 * MASS)) * filt
    psi_ev = ifft2(psi_k)
    psi_ev[inside] = 0.0
    psi_ev *= window

    u, v = madelung_decode(psi_ev, dx=dx, dy=dx)
    r2 = np.maximum(r ** 2, 1e-12)
    u = u - gamma * (YY - zc.imag) / (2 * np.pi * r2)
    v = v + gamma * (XX - zc.real) / (2 * np.pi * r2)
    u[inside] = np.nan
    v[inside] = np.nan

    # analytic reference (same circulation convention)
    ur = U_INF * (1 - R ** 2 / np.maximum(r, 1e-9) ** 2) * np.cos(th - alpha)
    ut = -U_INF * (1 + R ** 2 / np.maximum(r, 1e-9) ** 2) * np.sin(th - alpha) \
        + gamma / (2 * np.pi * np.maximum(r, 1e-9))
    ua = ur * np.cos(th) - ut * np.sin(th)
    va = ur * np.sin(th) + ut * np.cos(th)
    ua[inside] = np.nan
    va[inside] = np.nan
    return XX, YY, u, v, ua, va, inside, window

def rel_error_zeta(u, v, ua, va, inside, window, vel_ana_thresh=0.05):
    vel_q = np.sqrt(u ** 2 + v ** 2)
    vel_a = np.sqrt(ua ** 2 + va ** 2)
    valid = (~inside) & (window > 0.99) & (vel_a > vel_ana_thresh)
    return np.nanmean(np.abs(vel_q[valid] - vel_a[valid]) / (vel_a[valid] + 1e-12))

# ---------------------------------------------------------------- Joukowski
def joukowski_panel_data():
    from quantum_potential_flow import compute_kutta_gamma, generate_joukowski_airfoil
    eps, delta = 0.1, 0.05
    zc = complex(-eps, delta)
    a = np.sqrt((1 + eps) ** 2 + delta ** 2)
    gamma = -compute_kutta_gamma(eps, delta, U_INF)   # negative: Joukowski pipeline convention
    XX, YY, u, v, ua, va, inside, window = zeta_cylinder_scheme(a, zc, gamma, 0.0)
    err = rel_error_zeta(u, v, ua, va, inside, window)
    af_x, af_y, _, _ = generate_joukowski_airfoil(eps, delta)
    return dict(u=u, v=v, err=err, af_x=af_x, af_y=af_y, zc=zc, a=a,
                gamma=abs(gamma), title='Joukowski')

def render_joukowski(ax, d, vmax=None):
    from scipy.ndimage import binary_erosion
    LZ = 4.15
    nd = 400
    zd = np.linspace(-LZ, LZ, nd)
    XD, YD = np.meshgrid(zd, zd)
    zc, a = d['zc'], d['a']
    ZI, valid = inverse_joukowski_map(XD + 1j * YD, zc, a, PLOT_LIM,
                                      af_x=d['af_x'], af_y=d['af_y'])
    ne = d['u'].shape[0]
    dx = 2 * ZETA_RANGE / ne
    zx = (np.arange(ne) - ne // 2) * dx
    iu = RegularGridInterpolator((zx, zx), _prepare_velocity_grid(d['u']),
                                 bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((zx, zx), _prepare_velocity_grid(d['v']),
                                 bounds_error=False, fill_value=0.0)
    pts = np.stack([ZI.imag.ravel(), ZI.real.ravel()], axis=-1)
    dz = (1.0 - 1.0 / ZI ** 2).ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(np.abs(dz) > 1e-12, (iu(pts) - 1j * iv(pts)) / dz, 0.0)
    # erode the valid mask by 1 px: just enough to cut the bilinear NaN-edge
    # blending (jagged white fringe) without widening the white rim
    valid_e = binary_erosion(valid.reshape(nd, nd), iterations=1)
    vel = np.where(valid_e.ravel(), np.abs(w), np.nan).reshape(nd, nd)
    # per-panel color scale based on the displayed (physical-plane) field.
    # Round the 98th-percentile speed UP to the next 0.1 (instead of 0.5).
    # With the previous 0.5 rounding the scale jumped to 1.5 for all panels,
    # saturating the leading-edge suction peak and pinning the bright region
    # at the LE; the finer step (~1.2 here) reveals the velocity gradient
    # along the suction side while keeping <2% of pixels saturated.
    if vmax is None:
        vv = vel[np.isfinite(vel)]
        vmax = np.ceil(float(np.percentile(vv, 98)) * 10) / 10 if vv.size else 1.3
    im = ax.imshow(vel, extent=[-LZ, LZ, -LZ, LZ], cmap='viridis', vmin=0,
                   vmax=vmax, origin='lower', aspect='equal', interpolation='bilinear')
    ax.set_xlim(-LZ, LZ)
    ax.set_ylim(-LZ, LZ)
    ax.fill(d['af_x'], d['af_y'], color='#333', alpha=1.0, zorder=5)
    ax.plot(d['af_x'], d['af_y'], 'w-', lw=1.0, zorder=5)
    # --- freestream / angle-of-attack annotation (alpha = 0) ---------------
    # horizontal freestream arrow in the lower-left (upstream) region
    xa0, ya0 = -3.55, -2.5
    La = 0.9
    ax.annotate('', xy=(xa0 + La, ya0), xytext=(xa0, ya0),
                arrowprops=dict(arrowstyle='-|>', color='#2d3436',
                                lw=1.6, mutation_scale=15))
    ax.text(xa0 + La + 0.06, ya0 - 0.28, r'$U_\infty$', fontsize=11,
            color='#2d3436')
    # chord line at the leading edge; alpha = 0 so it coincides with the
    # freestream direction
    x_le = np.min(d['af_x'])
    ax.text(x_le - 0.65, 0.08, r'$\alpha=0^\circ$', fontsize=10,
            color='0.20', zorder=6)
    return im

# ---------------------------------------------------------------- NACA panels
def naca_panel_data(name):
    if name == 'NACA 0012':
        from theodorsen_mapping import compute_conformal_mapping
        R, coeffs, theta, z_af, *_ = compute_conformal_mapping()
    else:
        from naca2412_simulation import theodorsen_mapping_naca2412
        R, coeffs, theta, z_af, best_dist = theodorsen_mapping_naca2412()
    # Kutta circulation at alpha = 5 deg. NEGATIVE: clockwise-positive
    # convention, identical to the Joukowski panel, so that the suction side
    # (upper surface, where the circulation adds to the freestream) carries
    # the low pressure for alpha > 0. (A positive gamma would reverse the
    # lift direction, rendering an equivalent -5 deg flow.)
    gamma = -4 * np.pi * R * U_INF * np.sin(ALPHA)
    XX, YY, u, v, ua, va, inside, window = zeta_cylinder_scheme(R, 0j, gamma, ALPHA)
    err = rel_error_zeta(u, v, ua, va, inside, window)
    return dict(R=R, coeffs=coeffs, z_af=z_af, u=u, v=v, err=err,
                gamma=abs(gamma), title=name)

def render_naca(ax, d, vmax=None):
    from scipy.ndimage import binary_dilation
    R, coeffs = d['R'], d['coeffs']
    # display grid in the zeta plane
    n_disp = 400
    dx_d = 2 * ZETA_RANGE / n_disp
    zx_d = (np.arange(n_disp) - n_disp // 2) * dx_d
    XXd, YYd = np.meshgrid(zx_d, zx_d)
    ZETA_D = XXd + 1j * YYd
    # interpolate velocity to the display grid
    ne = d['u'].shape[0]
    dx = 2 * ZETA_RANGE / ne
    zx = (np.arange(ne) - ne // 2) * dx
    iu = RegularGridInterpolator((zx, zx), _prepare_velocity_grid(d['u']),
                                 bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((zx, zx), _prepare_velocity_grid(d['v']),
                                 bounds_error=False, fill_value=0.0)
    pts = np.stack([YYd.ravel(), XXd.ravel()], axis=-1)
    u_d = iu(pts).reshape(n_disp, n_disp)
    v_d = iv(pts).reshape(n_disp, n_disp)
    # mask: cylinder interior, window-taper band, AND interpolation out-of-range
    # (fill_value=0.0 would otherwise paint spurious dark-blue zero-velocity)
    in_grid = (np.abs(XXd) <= zx[-1]) & (np.abs(YYd) <= zx[-1])
    inside_d = (np.abs(ZETA_D) < R) | (np.abs(XXd) > PLOT_LIM) | (np.abs(YYd) > PLOT_LIM) | ~in_grid
    u_d[inside_d] = np.nan
    v_d[inside_d] = np.nan
    # conformal map to the physical plane.  NOTE: the coefficient vector has
    # FFT length (hundreds of entries) but only the first few are nonzero;
    # summing zero coefficients over the full length would raise zeta**k for
    # k ~ 1000, which UNDERFLOWS to 0 for |zeta| < ~0.48 and makes 0/0 = NaN,
    # silently destroying every point that maps into the central z-window
    # (the airfoil and its surroundings). Skip vanishing coefficients.
    z = ZETA_D + coeffs[0]
    for k in range(1, len(coeffs)):
        if abs(coeffs[k]) < 1e-14:
            continue
        z = z + coeffs[k] / (ZETA_D ** k)
    dz = np.ones_like(ZETA_D)
    for k in range(1, len(coeffs)):
        if abs(coeffs[k]) < 1e-14:
            continue
        dz = dz - k * coeffs[k] / (ZETA_D ** (k + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(np.abs(dz) > 1e-12, (u_d - 1j * v_d) / dz, 0.0)
    vel = np.abs(w)
    vel[inside_d] = np.nan

    # square z-window around the airfoil (chord ~ 1). The half-width is
    # matched to the Joukowski panel so the airfoil occupies the same
    # fraction of every panel: Joukowski chord 4.014 over a [-4.15, 4.15]
    # window (48.4%), hence 1.0/0.484/2 = 1.033 for a unit-chord airfoil.
    xz = z.real
    yz = z.imag
    zaf = d['z_af']
    xc = 0.5 * (np.nanmax(zaf.real) + np.nanmin(zaf.real))
    half = 1.033
    # Present the flow in the wind frame: rotate the entire physical field
    # (sample points) and the airfoil contour by -alpha about the mid-chord so
    # the airfoil pitches UP (leading edge rises) by the angle of attack while
    # the freestream stays horizontal. The scalar |v| is invariant under this
    # coordinate rotation, so only the sample positions and the contour are
    # rotated. (A +alpha rotation would depress the leading edge, rendering a
    # visually negative angle of attack.)
    ca, sa = np.cos(ALPHA), np.sin(ALPHA)
    yc_af = 0.5 * (np.nanmax(zaf.imag) + np.nanmin(zaf.imag))
    xr = xc + (xz - xc) * ca + (yz - yc_af) * sa
    yr = yc_af - (xz - xc) * sa + (yz - yc_af) * ca
    zaf_r = xc + (zaf.real - xc) * ca + (zaf.imag - yc_af) * sa \
        + 1j * (yc_af - (zaf.real - xc) * sa + (zaf.imag - yc_af) * ca)
    ok = (~inside_d) & np.isfinite(vel) & (np.abs(xr - xc) < half) & (np.abs(yr) < half)
    # regrid the mapped scattered points onto a regular z-grid, then imshow
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    nz = 500
    zgx = np.linspace(xc - half, xc + half, nz)
    zgy = np.linspace(-half, half, nz)
    ZGX, ZGY = np.meshgrid(zgx, zgy)
    vel_reg = griddata((xr[ok], yr[ok]), vel[ok], (ZGX, ZGY), method='linear')
    # Delaunay interpolation is unreliable where the mapped sample points are
    # sparse (the corners of the z-window): mask by the distance to the
    # nearest sample point (~3x the mean sample spacing ~0.04)
    tree = cKDTree(np.column_stack([xr[ok], yr[ok]]))
    dmin, _ = tree.query(np.column_stack([ZGX.ravel(), ZGY.ravel()]))
    vel_reg[dmin.reshape(ZGX.shape) > 0.12] = np.nan
    # mask the airfoil interior, dilated by 2 px: enough to cover the tiny
    # polygon-vs-contour gaps (<1 px, the contour has 2048 points) while
    # keeping the white rim around the airfoil thin
    inside_af = Path(np.column_stack([zaf_r.real, zaf_r.imag])).contains_points(
        np.column_stack([ZGX.ravel(), ZGY.ravel()])).reshape(ZGX.shape)
    inside_af = binary_dilation(inside_af, iterations=2)
    vel_reg[inside_af] = np.nan
    # per-panel color scale based on the displayed (physical-plane) field.
    # Same 0.1 rounding as the Joukowski panel (~1.2, see render_joukowski).
    if vmax is None:
        vv = vel_reg[np.isfinite(vel_reg)]
        vmax = np.ceil(float(np.percentile(vv, 98)) * 10) / 10 if vv.size else 1.3
    im = ax.imshow(vel_reg, extent=[zgx[0], zgx[-1], zgy[0], zgy[-1]],
                   cmap='viridis', vmin=0, vmax=vmax, origin='lower',
                   aspect='equal', interpolation='bilinear')
    ax.fill(zaf_r.real, zaf_r.imag, color='#333', alpha=1.0, zorder=5)
    ax.plot(zaf_r.real, zaf_r.imag, 'w-', lw=1.0, zorder=5)
    # --- angle-of-attack annotation -------------------------------------
    # The flow is solved in the body frame (airfoil horizontal, freestream
    # inclined by alpha; the physical field is rotated back to the wind frame
    # so the airfoil pitches up while the freestream stays horizontal).
    # (1) horizontal freestream arrow in the lower-left (upstream) region
    xa0, ya0 = zgx[0] + 0.07, -0.72
    La = 0.34
    ax.annotate('', xy=(xa0 + La, ya0), xytext=(xa0, ya0),
                arrowprops=dict(arrowstyle='-|>', color='#2d3436',
                                lw=1.6, mutation_scale=15))
    ax.text(xa0 + La + 0.03, ya0 - 0.07,
            r'$U_\infty$', fontsize=11, color='#2d3436')
    # (2) angle-of-attack label, placed upstream of the leading edge
    x_le = np.min(zaf_r.real)
    y_le = zaf_r.imag[np.argmin(zaf_r.real)]
    ax.text(x_le - 0.18, y_le + 0.045, r'$\alpha=5^\circ$', fontsize=10,
            color='0.20', zorder=6)
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(-half, half)
    ax.set_box_aspect(1)
    return im

def main():
    t0 = time.time()
    print("=" * 70)
    print(" Three-airfoil application snapshot (T = 0.01, free-particle scheme)")
    print("=" * 70)

    panels = []
    panels.append(joukowski_panel_data())
    panels.append(naca_panel_data('NACA 0012'))
    panels.append(naca_panel_data('NACA 2412'))
    for p in panels:
        print(f"  {p['title']:<10s}  Gamma={p['gamma']:8.4f}  rel. err = {p['err']:.4e}")

    # Per-panel color scale: each panel computes its own vmax from its
    # displayed field's 98th percentile (rounded up to 0.1). The three
    # percentiles are nearly identical (Joukowski ~1.12, NACA 0012 ~1.15,
    # NACA 2412 ~1.14), so all panels land on vmax = 1.2 and the shared
    # colorbar remains directly comparable, but the finer step stops the
    # leading-edge suction peak from saturating and lets the upper-surface
    # velocity gradient show instead of a spot pinned at the LE.
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.3), constrained_layout=True)
    ims = []
    for ax, p in zip(axes, panels):
        if p['title'] == 'Joukowski':
            im = render_joukowski(ax, p, vmax=None)
        else:
            im = render_naca(ax, p, vmax=None)
        ax.set_title(f"{p['title']}\nrel. err $= {p['err']:.2e}$", fontsize=12)
        ax.set_xlabel('$x/c$')
        ax.set_ylabel('$y/c$')
        ax.set_aspect('equal')
        ax.set_box_aspect(1)
        ims.append(im)
        print(f"  {p['title']:<10s}  displayed vmax = {im.get_clim()[1]:.1f}")
    cb = fig.colorbar(ims[-1], ax=axes[-1], shrink=0.85, pad=0.02)
    cb.set_label(r'$|\mathbf{V}|/U_\infty$')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig',
                       'airfoil_validation.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  -> saved {out}  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
