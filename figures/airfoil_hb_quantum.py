"""
Airfoil end-to-end H_B (Bernoulli-eigenstate) quantum simulation
================================================================
The airfoil in the zeta plane IS the canonical cylinder of radius R
(center zc), so the airfoil H_B computation is the mapped-grid cylinder
flow at angle of attack, evolved under H_B = -laplacian/2 - |v|^2/2,
decoded, and mapped back to the physical plane by z = f(zeta).

Per-airfoil pipeline (identical conventions to airfoil_validation_composite):
  1. mapped polar grid (no window) with cylinder radius R, center zc;
  2. encode Phi = U(r + R^2/r) cos(theta - alpha), psi = exp(i Phi);
  3. Bernoulli potential V = -|v_ref|^2/2 (analytic cylinder velocity);
  4. H_B evolution (exact spectral radial propagator + angular FFT,
     Strang V/2 - radial/2 - angular - radial/2 - V/2);
  5. Madelung decode (vr, vt); circulation superposed classically
     (the quantized-encoding alternative is analyzed in
     circulation_aero_error.py);
  6. conformal map z = f(zeta) to the physical plane, velocity
     transformed by w_z = w_zeta / (dz/dzeta);
  7. velocity error vs the analytic conformal-map reference.

Outputs (fig/):
  fig/airfoil_hb_validation.pdf  : three-airfoil physical-plane |v|
      (decoded vs analytic, shared scale)
  fig/airfoil_hb_err_vs_T.pdf    : velocity error vs T, H_B vs free
  fig/airfoil_hb_surface_cp.pdf  : surface C_p, decoded vs analytic
"""
import os, sys, time
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

import unbounded_mapping as um
from mapped_bernoulli import RadialSpectralPropagator, eigenstate_residual
import airfoil_panel_method as pm
import panel_cl_baseline as pb

U_INF, HBAR, MASS = 1.0, 1.0, 1.0
ALPHA_DEG = 5.0
RING_DX = 2.0            # near-surface ring r = R + RING_DX*dr_phys
N_SURF = 600
TS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08]
DT = 1e-4

# ---------------------------------------------------------------- geometry
# Kutta circulation convention: NEGATIVE gamma = clockwise = positive lift
# (matches airfoil_validation_composite.py).  Adding gamma/(2pi r) to v_theta
# with gamma<0 decelerates the bottom surface and accelerates the top, which
# is the lifting flow at alpha > 0 (and for the cambered Joukowski at alpha=0).
def joukowski_geometry():
    """Joukowski section, scaled to unit chord, camber lift.

    The family z = zeta + b^2/zeta has no free scale once the cusp is pinned at
    zeta = b, so b = 1 gives a chord of 4.03.  The circle plane is therefore
    scaled by s = 1/chord (b -> s) so that all three sections of the paper have
    unit chord: every dimensionless quantity is unchanged, but the per-chord
    grid resolution and the error windows become comparable across columns.

    The circulation is 4 pi a U sin(alpha - beta) with alpha = 0 and beta the
    cusp pre-image angle, so the section carries its camber through the ideal
    angle.
    """
    eps, delta = 0.1, 0.05
    zc_raw = complex(-eps, delta)
    a_raw = np.sqrt((1 + eps) ** 2 + delta ** 2)
    th = np.linspace(0.0, 2.0 * np.pi, 4001)
    z_raw = zc_raw + a_raw * np.exp(1j * th)
    w_raw = z_raw + 1.0 / z_raw
    s = 1.0 / float(w_raw.real.max() - w_raw.real.min())
    zc, a = zc_raw * s, a_raw * s
    gamma = -4 * np.pi * a * U_INF * np.sin(abs(np.angle(s - zc)))
    return dict(name='Joukowski', R=a, zc=zc, alpha=0.0, gamma=gamma,
                kind='joukowski', b2=s * s)

def naca_geometry(name):
    """Mapped NACA section: radius, Laurent coefficients and ideal angle.

    The circulation is the section's own Kutta value.  The rear stagnation
    point sits at the pre-image of the trailing edge, i.e. at the circle-plane
    angle beta, so Gamma = -4 pi R U sin(alpha - beta).  beta vanishes for the
    symmetric NACA 0012 and is -2.07 deg for the cambered NACA 2412, whose lift
    therefore differs from the symmetric section's by the camber alone.
    """
    sec = {s['name']: s for s in pb.load_sections()}[name]
    alpha = np.radians(ALPHA_DEG)
    gamma = -4 * np.pi * sec['R'] * U_INF * np.sin(alpha - sec['beta'])
    return dict(name=name, R=float(sec['R']), zc=sec['zc'], alpha=alpha,
                gamma=gamma, kind='naca', coeffs=sec['coeffs'])

# ---------------------------------------------------------------- physics
def encode_potential(grid, R, alpha):
    """Phi = U(r + R^2/r) cos(theta - alpha) on the mapped grid."""
    r = grid.RR
    return U_INF * (r + R ** 2 / r) * np.cos(grid.TT - alpha)

def analytic_velocity(grid, R, alpha, gamma):
    """Analytic cylinder velocity (vr, vt) with Kutta circulation."""
    r = np.maximum(grid.RR, 1e-9)
    r2 = r ** 2
    vr = U_INF * (1 - R ** 2 / r2) * np.cos(grid.TT - alpha)
    vt = -U_INF * (1 + R ** 2 / r2) * np.sin(grid.TT - alpha) \
        + gamma / (2 * np.pi * r)
    return vr, vt

def bernoulli_potential(grid, R, alpha, gamma=0.0):
    """V = -|v|^2/2 with the ANALYTIC velocity (no circulation in V for the
    superposition pipeline; the exact V_ex alternative is tested below)."""
    vr, vt = analytic_velocity(grid, R, alpha, 0.0)
    return -(vr ** 2 + vt ** 2) / 2.0

def evolve_HB(grid, rsp, psi, V, T, dt=DT):
    """Strang split-step H_B evolution (V/2 - radial/2 - angular - radial/2 - V/2)."""
    n_steps = max(1, int(T / dt))
    dt_a = T / n_steps
    k_theta = np.fft.fftfreq(grid.Nt, d=1.0) * grid.Nt
    k_theta_sq = k_theta ** 2
    expV = np.exp(-1j * V * dt_a / 2)
    psi_t = psi.copy()
    for _ in range(n_steps):
        psi_t = psi_t * expV
        psi_t = rsp.apply(psi_t, dt_a / 2)
        psi_kt = np.fft.fft(psi_t, axis=1)
        psi_kt *= np.exp(-1j * HBAR * dt_a / (2 * MASS)
                         * k_theta_sq[np.newaxis, :] / (grid.RR ** 2 + 1e-12))
        psi_t = np.fft.ifft(psi_kt, axis=1)
        psi_t = rsp.apply(psi_t, dt_a / 2)
        psi_t = psi_t * expV
    return psi_t

def evolve_free(grid, rsp, psi, T, dt=DT):
    """Free evolution (same splitting, no V)."""
    n_steps = max(1, int(T / dt))
    dt_a = T / n_steps
    k_theta = np.fft.fftfreq(grid.Nt, d=1.0) * grid.Nt
    k_theta_sq = k_theta ** 2
    psi_t = psi.copy()
    for _ in range(n_steps):
        psi_t = rsp.apply(psi_t, dt_a / 2)
        psi_kt = np.fft.fft(psi_t, axis=1)
        psi_kt *= np.exp(-1j * HBAR * dt_a / (2 * MASS)
                         * k_theta_sq[np.newaxis, :] / (grid.RR ** 2 + 1e-12))
        psi_t = np.fft.ifft(psi_kt, axis=1)
        psi_t = rsp.apply(psi_t, dt_a / 2)
    return psi_t

def decode_velocity(grid, psi, gamma):
    """Madelung decode (vr, vt) + classical circulation superposition."""
    vr, vt, rho = grid.decode(psi)
    vt = vt + gamma / (2 * np.pi * np.maximum(grid.RR, 1e-9))
    return vr, vt, rho

def to_cartesian(grid, zc, vr, vt):
    """(vr, vt) -> Cartesian (vx, vy) in the zeta plane about center zc."""
    ct, st = np.cos(grid.TT), np.sin(grid.TT)
    vx = vr * ct - vt * st
    vy = vr * st + vt * ct
    zeta = zc + grid.RR * np.exp(1j * grid.TT)
    return vx, vy, zeta

# ---------------------------------------------------------------- mapping
def fwd_map(zeta, geom):
    if geom['kind'] == 'joukowski':
        return zeta + geom.get('b2', 1.0) / zeta
    coeffs = np.asarray(geom['coeffs'])
    z = zeta + coeffs[0]
    for k in range(1, min(len(coeffs), 64)):
        z = z + coeffs[k] / (zeta ** k)
    return z

def dz_map(zeta, geom):
    if geom['kind'] == 'joukowski':
        return 1.0 - geom.get('b2', 1.0) / zeta ** 2
    coeffs = np.asarray(geom['coeffs'])
    dz = np.ones_like(zeta)
    for k in range(1, min(len(coeffs), 64)):
        dz = dz - k * coeffs[k] / (zeta ** (k + 1))
    return dz

def physical_velocity(vx, vy, zeta, geom):
    """w_z = w_zeta / (dz/dzeta); |v| in the physical plane."""
    dz = dz_map(zeta, geom)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(np.abs(dz) > 1e-12, (vx - 1j * vy) / dz, 0.0)
    return np.abs(w)

# ---------------------------------------------------------------- metrics
def interior_mask(grid, R):
    return (grid.RR > R * 1.05) & (grid.RR < 1.5)

def rel_err_zeta(grid, R, vr, vt, gamma, alpha):
    vmag = np.sqrt(vr ** 2 + vt ** 2)
    vra, vta = analytic_velocity(grid, R, alpha, gamma)
    vref = np.sqrt(vra ** 2 + vta ** 2)
    m = interior_mask(grid, R)
    return (np.sqrt(np.sum((vmag[m] - vref[m]) ** 2))
            / np.sqrt(np.sum(vref[m] ** 2)))

def surface_cp(grid, geom, vr, vt):
    """C_p on the near-surface ring r = R + RING_DX*dr, mapped to x/c."""
    R = geom['R']
    dr = grid.dr_phys
    i_ring = np.argmin(np.abs(grid.r - (R + RING_DX * dr[0])))
    ths = grid.TT[i_ring, :]
    vr_s, vt_s = vr[i_ring, :], vt[i_ring, :]
    r_ring = grid.r[i_ring]
    zeta_s = geom['zc'] + r_ring * np.exp(1j * ths)
    ct, st = np.cos(ths), np.sin(ths)
    vx = vr_s * ct - vt_s * st
    vy = vr_s * st + vt_s * ct
    w = physical_velocity(vx, vy, zeta_s, geom)
    # analytic reference on the same ring
    vra, vta = analytic_velocity(grid, R, geom['alpha'], geom['gamma'])
    vx_a = vra[i_ring, :] * ct - vta[i_ring, :] * st
    vy_a = vra[i_ring, :] * st + vta[i_ring, :] * ct
    wa = physical_velocity(vx_a, vy_a, zeta_s, geom)
    Cp_q = 1.0 - (w / U_INF) ** 2
    Cp_a = 1.0 - (wa / U_INF) ** 2
    x_surf = np.real(fwd_map(zeta_s, geom))
    # chord-normalize
    x_c = (x_surf - x_surf.min()) / (x_surf.max() - x_surf.min())
    return x_c, Cp_q, Cp_a

# ---------------------------------------------------------------- run
def run_airfoil(geom, T_list=TS):
    """Run the full H_B and free pipelines for one airfoil geometry."""
    R, zc, alpha, gamma = geom['R'], geom['zc'], geom['alpha'], geom['gamma']
    um.R_CYL = R
    grid = um.MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
    rsp = RadialSpectralPropagator(grid)

    phi = encode_potential(grid, R, alpha)
    psi0 = np.exp(1j * phi / HBAR)
    V = bernoulli_potential(grid, R, alpha)
    Hf0, _ = eigenstate_residual(grid, psi0, V)
    V_ex = -Hf0 / psi0

    out = dict(grid=grid, geom=geom, err_hb_ana=[], err_hb_exact=[],
               err_free=[], err_zeta_ana=[])
    for T in T_list:
        ps_hb = evolve_HB(grid, rsp, psi0.copy(), V, T)
        ps_ex = evolve_HB(grid, rsp, psi0.copy(), V_ex, T)
        ps_fr = evolve_free(grid, rsp, psi0.copy(), T)
        vr_h, vt_h, _ = decode_velocity(grid, ps_hb, gamma)
        vr_x, vt_x, _ = decode_velocity(grid, ps_ex, gamma)
        vr_f, vt_f, _ = decode_velocity(grid, ps_fr, gamma)
        e_h = rel_err_zeta(grid, R, vr_h, vt_h, gamma, alpha)
        e_x = rel_err_zeta(grid, R, vr_x, vt_x, gamma, alpha)
        e_f = rel_err_zeta(grid, R, vr_f, vt_f, gamma, alpha)
        out['err_hb_ana'].append(e_h)
        out['err_hb_exact'].append(e_x)
        out['err_free'].append(e_f)
    return out

def _sci(x):
    """Format a number as $a.b\\times10^{n}$ for the figure annotations."""
    e = int(np.floor(np.log10(abs(x))))
    return f'{x / 10 ** e:.1f}' + r'\times10^{' + str(e) + '}'

def plot_surface_cp(geoms, fig_dir, T=0.01):
    """Surface C_p decoded vs analytic, with the pointwise decode error.

    Top row: C_p along the chord, decoded (solid) against the analytic
    conformal-map reference (black dashed/dotted).  The two agree on this
    scale, so the quantitative content sits in the bottom row, |Delta C_p|
    between decoded and analytic on a log axis with the mean marked.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 5.6), sharex='col')
    means = []
    for j, g in enumerate(geoms):
        um.R_CYL = g['R']
        grid = um.MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
        rsp = RadialSpectralPropagator(grid)
        phi = encode_potential(grid, g['R'], g['alpha'])
        psi0 = np.exp(1j * phi / HBAR)
        V = bernoulli_potential(grid, g['R'], g['alpha'])
        Hf0, _ = eigenstate_residual(grid, psi0, V)
        V_ex = -Hf0 / psi0
        ps_ex = evolve_HB(grid, rsp, psi0.copy(), V_ex, T)
        vr, vt, _ = decode_velocity(grid, ps_ex, g['gamma'])
        x_c, Cp_q, Cp_a = surface_cp(grid, g, vr, vt)
        n = len(x_c)
        up = np.zeros(n, dtype=bool)
        up[:n // 2] = True

        ax = axes[0, j]
        ax.plot(x_c[up], Cp_q[up], 'r-', lw=1.4, label='decoded (upper)')
        ax.plot(x_c[~up], Cp_q[~up], 'b-', lw=1.4, label='decoded (lower)')
        ax.plot(x_c[up], Cp_a[up], 'k--', lw=1.0, alpha=0.7,
                label='analytic (upper)')
        ax.plot(x_c[~up], Cp_a[~up], 'k:', lw=1.0, alpha=0.7,
                label='analytic (lower)')
        ax.invert_yaxis()
        ax.set_ylabel('$C_p$')
        ax.set_title(f"{g['name']} surface $C_p$ (T=0.01)", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right', ncol=2,
                  framealpha=0.85, columnspacing=0.8, handlelength=1.5,
                  borderpad=0.3, handletextpad=0.4, labelspacing=0.35)

        ax2 = axes[1, j]
        dCp = np.abs(Cp_q - Cp_a)
        ax2.semilogy(x_c[up], dCp[up], 'r-', lw=1.2, label='upper')
        ax2.semilogy(x_c[~up], dCp[~up], 'b-', lw=1.2, label='lower')
        m = float(np.nanmean(dCp))
        means.append(m)
        ax2.axhline(m, color='grey', ls=':', lw=1.2)
        ax2.text(0.04, m * 2.4, f'mean $= {_sci(m)}$', fontsize=8, color='grey')
        ax2.set_xlabel('$x/c$')
        ax2.set_ylabel(r'$|\Delta C_p|$')
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(1e-4, 4e-2)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=7, loc='lower right', framealpha=0.8)
        print(f"   {g['name']:<10s} mean |dC_p| = {m:.3e}")

    fig.tight_layout(pad=0.6, h_pad=1.0)
    out = os.path.join(fig_dir, 'airfoil_hb_surface_cp.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out)}")
    return means

def main():
    t0 = time.time()
    geoms = [joukowski_geometry(), naca_geometry('NACA 0012'),
             naca_geometry('NACA 2412')]
    results = []
    for g in geoms:
        r = run_airfoil(g)
        results.append(r)
        print(f"  {g['name']:<10s} R={g['R']:.4f} Gamma={g['gamma']:.4f} "
              f"alpha={np.degrees(g['alpha']):.0f}deg")
        for T, ea, ex, ef in zip(TS, r['err_hb_ana'], r['err_hb_exact'],
                                 r['err_free']):
            print(f"    T={T:<6.3f}  H_B(ana)={ea:.4e}  "
                  f"H_B(exact)={ex:.4e}  free={ef:.4e}")
    print(f"  [{(time.time()-t0):.0f}s]")

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                           'fig')
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Fig 1: error vs T, three schemes, three airfoils ----
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colors = ['#185FA5', '#0F6E56', '#993C1D']
    for r, c in zip(results, colors):
        g = r['geom']
        ax.semilogy(TS, r['err_hb_exact'], '-o', color=c, lw=1.8, ms=5,
                    label=f"{g['name']} (H$_B$ exact)")
        ax.semilogy(TS, r['err_hb_ana'], '-s', color=c, lw=1.3, ms=4,
                    alpha=0.55, label=f"{g['name']} (H$_B$ analytic $V$)")
        ax.semilogy(TS, r['err_free'], '--x', color=c, lw=1.2, ms=5,
                    alpha=0.5, label=f"{g['name']} (free)")
    ax.set_xlabel('evolution time $T$')
    ax.set_ylabel('velocity error (bulk)')
    ax.set_title('Airfoil end-to-end: $H_B$ stationary encoding vs free '
                 'evolution')
    ax.legend(fontsize=7, ncol=3, framealpha=0.8)
    ax.grid(True, alpha=0.3, which='both')
    out = os.path.join(fig_dir, 'airfoil_hb_err_vs_T.pdf')
    fig.tight_layout()
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out)}")

    # ---- Fig 2: physical-plane |v| at T = 0.01 ----
    # Three rows: H_B(exact) decoded, analytic conformal reference, and a
    # high-precision panel method (source panels + prescribed Kutta
    # circulation, independent of the conformal map).  All rows are rotated
    # to the WIND frame (-alpha about the airfoil centroid) so the airfoil
    # pitches up while the freestream stays horizontal.
    #
    # Rendering: a regular Cartesian display grid is built per column and the
    # velocity magnitude is interpolated onto it (LinearNDInterpolator).  This
    # replaces the previous per-point scatter plot, whose sparse point density
    # near small-radius NACA sections amplified visually-perceived differences
    # that are not present in the underlying numbers.  All three rows of a
    # column share the same vmax (column-wise 98th percentile of the analytic
    # field), so H_B / analytic / panel are rendered on an identical colour
    # scale, and any visible difference reflects a real field mismatch.
    from scipy.interpolate import LinearNDInterpolator
    from matplotlib.path import Path as _MplPath

    fig, axes = plt.subplots(3, 3, figsize=(15.6, 14.0))
    ims = {}
    for j, r in enumerate(results):
        g = r['geom']
        grid = r['grid']
        T = 0.01
        phi = encode_potential(grid, g['R'], g['alpha'])
        psi0 = np.exp(1j * phi / HBAR)
        rsp = RadialSpectralPropagator(grid)
        V = bernoulli_potential(grid, g['R'], g['alpha'])
        Hf0, _ = eigenstate_residual(grid, psi0, V)
        V_ex = -Hf0 / psi0
        ps_ex = evolve_HB(grid, rsp, psi0.copy(), V_ex, T)
        vr, vt, _ = decode_velocity(grid, ps_ex, g['gamma'])
        vx, vy, zeta = to_cartesian(grid, g['zc'], vr, vt)
        vel_q = physical_velocity(vx, vy, zeta, g)
        vra, vta = analytic_velocity(grid, g['R'], g['alpha'], g['gamma'])
        vx_a, vy_a, _ = to_cartesian(grid, g['zc'], vra, vta)
        vel_a = physical_velocity(vx_a, vy_a, zeta, g)
        z = fwd_map(zeta, g)
        # airfoil outline in the physical plane: r = R, theta in [0, 2pi)
        th_outline = np.linspace(0, 2 * np.pi, 400)
        zeta_outline = g['zc'] + g['R'] * np.exp(1j * th_outline)
        z_outline = fwd_map(zeta_outline, g)
        # window centered on the airfoil geometric centroid (Theodorsen
        # mapping adds an offset c0, so the airfoil does not sit at the
        # origin; using a fixed [-LZ, LZ] window would push NACA off-center)
        cx = float(np.mean(z_outline.real))
        cy = float(np.mean(z_outline.imag))
        zc = cx + 1j * cy
        # half-window: 1.2 chords about the centroid.  All three sections now
        # have unit chord, so one window serves the three columns.
        LZ = 1.2 * float(z_outline.real.max() - z_outline.real.min())
        # wind-frame rotation first: airfoil pitches up by alpha, freestream
        # horizontal.  |v| is rotation-invariant, so only the sample positions
        # and the outline are rotated.
        zr = (z - zc) * np.exp(-1j * g['alpha']) + zc
        zo_r = (z_outline - zc) * np.exp(-1j * g['alpha']) + zc

        # window mask evaluated on the ROTATED positions, so the axis-aligned
        # display window is exactly covered (masking on the body frame would
        # leave the corners of the tilted data square blank).
        m = (grid.RR > g['R'] * 1.05) \
            & (np.abs(zr.real - cx) < LZ) \
            & (np.abs(zr.imag - cy) < LZ)
        # interpolation-input mask: same annulus on a window widened by
        # PAD.  Rings clipped to the display window alone cannot fill the
        # square corners (the outermost in-window ring bounds a chord that
        # cuts them off), so the Delaunay hull would leave white corner
        # triangles.  Sampling a wider window lets rings beyond the display
        # corners cover them; display still clips to the LZ window and the
        # error metrics below keep the original mask m.
        PAD = 1.5
        m_int = (grid.RR > g['R'] * 1.05) \
            & (np.abs(zr.real - cx) < PAD * LZ) \
            & (np.abs(zr.imag - cy) < PAD * LZ)
        zr_m = zr[m_int]

        # high-precision panel method: source panels on the airfoil boundary +
        # the analytic Kutta circulation (independent of the conformal map)
        nodes = pm.order_boundary(z_outline)
        sol_p = pm.solve(nodes, g['alpha'], Gamma=g['gamma'])
        vx_p, vy_p = pm.field_velocity(z.real, z.imag, nodes, sol_p,
                                       g['alpha'])
        vel_p = np.hypot(vx_p, vy_p)

        # regular Cartesian display grid (wind frame), per column.
        n_disp = 320
        xs = np.linspace(cx - LZ, cx + LZ, n_disp)
        ys = np.linspace(cy - LZ, cy + LZ, n_disp)
        GX, GY = np.meshgrid(xs, ys)

        # mask out the airfoil interior so it renders as background (white).
        # Using the rotated outline polygon in the wind frame.
        poly_pts = np.column_stack([zo_r.real, zo_r.imag])
        inside_mask = _MplPath(poly_pts).contains_points(
            np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

        # field interpolators on the wind-frame sample points (widened
        # window, see m_int above).
        pts = np.column_stack([zr_m.real, zr_m.imag])
        f_q = LinearNDInterpolator(pts, vel_q[m_int], fill_value=np.nan)
        f_a = LinearNDInterpolator(pts, vel_a[m_int], fill_value=np.nan)
        f_p = LinearNDInterpolator(pts, vel_p[m_int], fill_value=np.nan)
        FQ = f_q(GX, GY); FA = f_a(GX, GY); FP = f_p(GX, GY)
        FQ[inside_mask] = np.nan; FA[inside_mask] = np.nan
        FP[inside_mask] = np.nan

        # column-wise vmax: 98th percentile of the analytic field, so all
        # three rows share an identical colour scale and visible differences
        # are physical, not dynamic-range artefacts.
        vmax_col = float(np.percentile(FA[~np.isnan(FA)], 98))
        vmax_col = np.ceil(vmax_col * 10) / 10
        vmin = 0.0

        # per-column contour levels: the three rows of the column share the
        # same levels (and the same colour scale), so solver differences show
        # up as shifted / deformed isolines instead of identical colormaps.
        levels = np.linspace(vmin, vmax_col, 64)

        rows = [(0, FQ, f"{g['name']} $H_B$ (exact) $|v|$ (T=0.01)"),
                (1, FA, f"{g['name']} analytic $|v|$"),
                (2, FP, f"{g['name']} panel method $|v|$")]
        for row, F, ttl in rows:
            ax = axes[row, j]
            sc = ax.pcolormesh(GX, GY, F, cmap='viridis',
                              vmin=vmin, vmax=vmax_col, shading='auto',
                              rasterized=True)
            ax.contour(GX, GY, F, levels=levels, colors='k',
                       linewidths=0.35, alpha=0.5)
            ax.plot(zo_r.real, zo_r.imag, 'k-', lw=0.8)
            ax.set_xlim(cx - LZ, cx + LZ)
            ax.set_ylim(cy - LZ, cy + LZ)
            ax.set_aspect('equal')
            ax.set_title(ttl, fontsize=11)
            ims[row] = sc
        err = np.abs(vel_q[m] - vel_a[m]) / (vel_a[m] + 1e-12)
        err_p = np.abs(vel_p[m] - vel_a[m]) / (vel_a[m] + 1e-12)
        print(f"  {g['name']}: physical-plane mean rel err (T=0.01) = "
              f"{np.nanmean(err):.4e}; panel-vs-analytic = {np.nanmean(err_p):.4e}")
    for j in range(3):
        axes[0, j].set_xlabel('Re $z$')
        axes[1, j].set_xlabel('Re $z$')
        axes[2, j].set_xlabel('Re $z$')
    # Im z once per row, on the left column only.  Each panel title already
    # names the solver, so the former bold row labels just collided with the
    # y-label and the y-tick numbers.
    for row in range(3):
        axes[row, 0].set_ylabel('Im $z$')
    # one colorbar per solver row, vertically aligned with the row on the
    # right edge.  The three columns keep their own column-wise colour
    # scales (each airfoil has a different vmax), so a single shared bar
    # over the whole figure would misrepresent the non-uniform vmax.
    fig.subplots_adjust(left=0.06, right=0.90, top=0.96, bottom=0.05,
                        wspace=0.32, hspace=0.30)
    for row in range(3):
        pos = axes[row, -1].get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.016, pos.height])
        fig.colorbar(ims[row], cax=cax, label=r'$|v|/U_\infty$')
    out = os.path.join(fig_dir, 'airfoil_hb_validation.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out)}")

    # ---- Fig 3: surface C_p, decoded vs analytic ----
    plot_surface_cp(geoms, fig_dir)

if __name__ == '__main__':
    main()
