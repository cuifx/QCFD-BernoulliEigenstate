"""
3D Bodies of Revolution: Exact Bernoulli Encoding for the von Karman Double-Ogive
==================================================================================
Extends the stationary-encoding framework of the paper to three-dimensional
axisymmetric potential flow. The ingredients of the theorem are harmonicity
(lap Phi = 0 in 3D) and the Bernoulli potential, both dimension-independent;
the 3D obstruction is geometric (no conformal machinery generalizes, by
Liouville), so we use the canonical hypersonic/space body of revolution.

von Karman double-ogive (a pair of LD-Haack minimum-drag ogives joined
base-to-base at mid-span; the same Haack family furnishes the classic
hypersonic re-entry forebody), generated as a uniform stream + an axial
line-source distribution. The source strength is not the slender-body
approximation sigma = dS/dx (which leaves psi=0 visibly thinner than the
design meridian, max|dR| ~ 7e-2 on this body) but the fixed-point match of
the psi=0 stream surface to the design meridian:
  design meridian (Haack series),  xi = (x+L/2)/L in [0,1]:
    theta_f = arccos(1 - 4 xi) on the forebody  (0 <= xi <= 1/2),
    theta_a = arccos(1 - 4(1-xi)) on the afterbody (1/2 < xi <= 1),
    R(xi) = R_max sqrt((theta - sin(2 theta)/2)/pi)
  line-source potential (fixed-length quadrature sum, M panels):
    Phi(x,r) = U_inf x - (ds/4pi) sum_j sigma_j / d_j,
    d_j = sqrt((x-x_j)^2 + r^2),  r^2 = y^2+z^2,
  axisymmetric stream function (body = closed streamline psi = 0,
  interior psi < 0 under the standard Stokes convention u_x=(1/r) dpsi/dr):
    psi(x,r) = U_inf r^2/2 - (ds/4pi) sum_j sigma_j (x-x_j)/d_j.
Each panel term is a square root of a polynomial, so the potential is a
fixed-length composition of elementary functions that the arithmetic-compiled
phase oracle of the paper evaluates by a chain of multiply--add and
inverse-square-root operations (Strategy B). Numerical verification on a
uniform Cartesian box (fluid region):
  A. lap Phi = 0            (3D harmonicity, finite-difference check)
  B. H_B e^{iPhi} = 0       (eigenstate residual; operator-matching floor)
  C. stationarity: free 3D-FFT evolution grows, H_B with the analytic V
     grows at the operator-matching rate, H_B with the pointwise-exact
     potential V_ex stays pinned at the t=0 decoding floor.
"""
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- journal-style defaults (CMAME/Elsevier) ---
# Times-like serif text, Computer-Modern math (matches the LaTeX body),
# 9-10 pt lettering so that labels/ticks stay legible (>= 8 pt) at the
# published size (the 7.4-in flowfield figure prints at 0.96\textwidth,
# i.e. ~0.87 downscale), 300 dpi output, hairline lines in 0.25-1.5 pt.
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'savefig.dpi': 300,
    'mathtext.fontset': 'cm',
})
import os, time

HBAR, MASS = 1.0, 1.0

# ------------------------------------------------------------
# Geometry / flow parameters (von Karman double-ogive, hypersonic re-entry body)
# ------------------------------------------------------------
L_BODY = 4.0                 # body length (centered: x in [-L_BODY/2, L_BODY/2])
R_MAX = 0.5                  # design maximum radius at mid-span (L/D = 4)
U_INF = 1.0
M = 256                      # line-source panels (fixed-length quadrature sum)
DS = L_BODY / M              # panel spacing ds
N = 96                       # grid per dimension
L = 6.0                      # half box size


def vk_ogive_radius(xi):
    """von Karman double-ogive meridian R(xi), xi in [0,1]: two LD-Haack
    minimum-drag ogives joined base-to-base at mid-span, a symmetric closed
    body pointed at both ends with R_max at xi=1/2.  Haack series:
    theta = arccos(1-4 xi) on the forebody, arccos(1-4(1-xi)) on the
    afterbody;  R = R_max sqrt((theta - sin(2 theta)/2)/pi)."""
    xi = np.asarray(xi)
    u = np.where(xi <= 0.5, 1.0 - 4.0 * xi, 1.0 - 4.0 * (1.0 - xi))
    th = np.arccos(np.clip(u, -1.0, 1.0))
    return R_MAX * np.sqrt((th - 0.5 * np.sin(2.0 * th)) / np.pi)


# panel midpoints and the matched line-source strength: the slender-body seed
# sigma = dS/dx = 2 pi R R' leaves psi=0 visibly thinner than the design
# meridian on the von Karman double-ogive (max|dR| ~ 7e-2), so sigma is matched
# by a fixed-point iteration on the cross-sectional area until the actual
# psi=0 surface passes through the design stations (max|dR| < 3e-5).
XI_J = (np.arange(M) + 0.5) / M
XJ = XI_J * L_BODY - 0.5 * L_BODY


def _design_radius():
    """Design meridian R(xi) at the panel stations (von Karman double-ogive)."""
    _FORE = XI_J <= 0.5
    _XI1 = np.where(_FORE, XI_J, 1.0 - XI_J)
    _U = np.clip(1.0 - 4.0 * _XI1, -1.0, 1.0)
    _TH = np.arccos(_U)
    return R_MAX * np.sqrt((_TH - 0.5 * np.sin(2.0 * _TH)) / np.pi)


def _psi_zero_radius(sig, nq=1201):
    """Radius of the psi=0 stream surface at every station of the line-source
    flow.  psi(r) = U r^2/2 - (ds/4pi) sum_j sig_j (x_i-x_j)/d_j is monotone
    in r (uniform stream + axial sources), so the body root is the single
    negative->positive crossing, found by linear interpolation."""
    r = np.linspace(1e-5, 0.75, nq)
    R = np.empty(M)
    for i in range(M):
        dx = XJ[i] - XJ
        d = np.sqrt(dx[:, None] ** 2 + r[None, :] ** 2 + 1e-30)
        psi = 0.5 * U_INF * r ** 2 \
              - (DS / (4.0 * np.pi)) * (sig[:, None] * dx[:, None] / d).sum(axis=0)
        z = np.where((psi[:-1] <= 0.0) & (psi[1:] > 0.0))[0]
        if z.size == 0:
            R[i] = r[np.argmin(np.abs(psi))]
        else:
            k = z[0]
            R[i] = r[k] - psi[k] * (r[k + 1] - r[k]) / (psi[k + 1] - psi[k])
    return R


def _match_sigma(rtol=3e-5, maxit=60):
    """Fixed-point matching on the cross-sectional area: iterate
    sig <- sig + 0.9 * U * d/dx (S_design - pi R_act^2) until the psi=0
    surface tracks the design meridian within rtol.  Returns the matched
    strength (per unit length, U_inf=1 convention) and the final residual."""
    Rdes = _design_radius()
    Sdes = np.pi * Rdes ** 2
    sig = np.gradient(Sdes, XJ)                    # slender-body seed dS/dx
    err = np.inf
    for k in range(maxit):
        Ract = _psi_zero_radius(sig)
        err = np.max(np.abs(Ract - Rdes))
        if err < rtol:
            break
        sig = sig + 0.9 * U_INF * np.gradient(Sdes - np.pi * Ract ** 2, XJ)
    return sig, err, Ract


SIGMA, MATCH_ERR, R_MATCH = _match_sigma()


def _line_source_sum(kind, x, r):
    """Fixed-length sum over the M line-source panels (chunked for memory).
    kind: 'pot' -> sum_j sigma_j / d_j
          'vx'  -> sum_j sigma_j (x-x_j) / d_j^3
          'vr'  -> sum_j sigma_j r / d_j^3
          'psi' -> sum_j sigma_j (x-x_j) / d_j
    x, r: broadcastable arrays (arbitrary shape)."""
    shape = np.broadcast_shapes(np.shape(x), np.shape(r))
    xf = np.asarray(x).reshape(-1, 1)
    rf = np.asarray(r).reshape(-1, 1)
    out = np.zeros(xf.shape[0])
    chunk = 32
    for b in range(0, M, chunk):
        xj = XJ[b:b + chunk]
        sj = SIGMA[b:b + chunk]
        dx = xf - xj
        d = np.sqrt(dx * dx + rf * rf + 1e-30)
        if kind == 'pot':
            out += (sj / d).sum(axis=1)
        elif kind == 'vx':
            out += (sj * dx / d ** 3).sum(axis=1)
        elif kind == 'vr':
            out += (sj * rf / d ** 3).sum(axis=1)
        else:  # 'psi'
            out += (sj * dx / d).sum(axis=1)
    return out.reshape(shape)


def stream_psi(x, r):
    """Axisymmetric Stokes stream function of the line-source hull:
    psi = U r^2/2 - (ds/4pi) sum_j sigma_j (x-x_j)/d_j.  Body interior < 0."""
    return 0.5 * U_INF * r * r - (DS / (4.0 * np.pi)) * _line_source_sum('psi', x, r)


def potential_3d(x, y, z):
    """Phi(x,y,z) = U_inf x - (ds/4pi) sum_j sigma_j / d_j  (fluid region)."""
    r = np.sqrt(y * y + z * z)
    return U_INF * x - (DS / (4.0 * np.pi)) * _line_source_sum('pot', x, r)


def velocity_3d(x, y, z):
    """grad Phi from the closed-form derivatives of the fixed-length sum
    (no numerical differencing).  v_r is the radial component in the (y,z)
    plane; vy, vz are its Cartesian split."""
    r = np.sqrt(y * y + z * z)
    vx = U_INF + (DS / (4.0 * np.pi)) * _line_source_sum('vx', x, r)
    vr = (DS / (4.0 * np.pi)) * _line_source_sum('vr', x, r)
    vy = np.zeros_like(vr)
    vz = np.zeros_like(vr)
    nz = r > 1e-12
    vy[nz] = vr[nz] * y[nz] / r[nz]
    vz[nz] = vr[nz] * z[nz] / r[nz]
    return vx, vy, vz


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(Y * Y + Z * Z)
h = 2.0 * L / (N - 1)

# body and fluid masks (body interior = psi < 0 of the line-source flow)
inside_body = stream_psi(X, R) < 0.0
fluid = ~inside_body
# erode by 2 cells so FD stencils stay inside the fluid (away from the body)
probe0 = binary_erosion(fluid, iterations=2)
probe = probe0 & (R < 4.2) & (np.abs(X) < 4.2)


def fd_gradient(f):
    """Central-difference gradient (clamped at boundaries)."""
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gz = np.zeros_like(f)
    gx[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * h)
    gy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * h)
    gz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * h)
    return gx, gy, gz


def fd_laplacian(f):
    """7-point Laplacian (interior only)."""
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1, 1:-1] = (
        f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
        + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
        + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
        - 6.0 * f[1:-1, 1:-1, 1:-1]) / (h * h)
    return lap


def spectral_laplacian(f):
    """FFT (spectral) Laplacian on the uniform periodic box (complex output)."""
    k = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx * kx + ky * ky + kz * kz
    return ifftn(-k2 * fftn(f))


def kinetic_spectral(f):
    """H_free f = -lap_spec(f)/2 (the operator implemented by QFT^3)."""
    return -0.5 * spectral_laplacian(f)


# ------------------------------------------------------------
# FFT-periodicity window (2D convention: cosine taper, W=1 interior)
# ------------------------------------------------------------
TAPER_RATIO = 0.12


def smooth_boundary_window_3d(XX, YY, ZZ, L, taper_ratio=TAPER_RATIO):
    """Cosine-taper window product win_x * win_y * win_z; W=1 for |c|<L-w."""
    w = L * taper_ratio
    win = np.ones_like(XX)
    for C in (XX, YY, ZZ):
        wc = np.ones_like(XX)
        m_l = C < -L + w
        m_r = C > L - w
        wc[m_l] = 0.5 * (1 - np.cos(np.pi * (C[m_l] + L) / w))
        wc[m_r] = 0.5 * (1 - np.cos(np.pi * (L - C[m_r]) / w))
        win *= wc
    return win


WIND = smooth_boundary_window_3d(X, Y, Z, L)

# ------------------------------------------------------------
# Analytic fields on the grid and the encoded state
# ------------------------------------------------------------
PHI = potential_3d(X, Y, Z)
VRX, VRY, VRZ = velocity_3d(X, Y, Z)
V_ANA = -0.5 * (VRX ** 2 + VRY ** 2 + VRZ ** 2)   # analytic Bernoulli potential
psi0 = WIND * np.exp(1j * PHI)

# probe region: fluid, eroded 2 cells off the body, inside a radius-3.6 sphere
R3 = np.sqrt(X * X + Y * Y + Z * Z)
probe = probe0 & (R3 < 3.6)


def decode_velocity(psi):
    """Madelung decode v = Im(psi* grad psi)/|psi|^2 (central diff, clamped)."""
    gx, gy, gz = fd_gradient(psi)
    rho = np.abs(psi) ** 2 + 1e-12
    vx = np.imag(np.conj(psi) * gx) / rho
    vy = np.imag(np.conj(psi) * gy) / rho
    vz = np.imag(np.conj(psi) * gz) / rho
    return vx, vy, vz


def vel_error(psi):
    """||v - v_ref||_RMS(probe) / U_inf (interior-box metric, per review)."""
    vx, vy, vz = decode_velocity(psi)
    dv2 = (vx - VRX) ** 2 + (vy - VRY) ** 2 + (vz - VRZ) ** 2
    return np.sqrt(np.mean(dv2[probe])) / U_INF


def rms_over(f, mask):
    return np.sqrt(np.mean(np.abs(f)[mask] ** 2))


def eigenstate_checks():
    """A. harmonicity (FD lap of Phi) and B. H_B e^{iPhi} residuals."""
    lap_fd = fd_laplacian(PHI)
    g2 = VRX ** 2 + VRY ** 2 + VRZ ** 2
    rms_lap = rms_over(lap_fd, probe)
    rms_g2 = rms_over(g2, probe)
    rms_phi = rms_over(PHI, probe)

    Hf = kinetic_spectral(psi0)                    # H_free psi0 (spectral)
    res_ana = Hf + V_ANA * psi0                    # (H_free + V_ana) psi0
    norm0 = rms_over(psi0, probe)
    r_ana = rms_over(res_ana, probe) / norm0
    V_ex = _build_vex()                            # pointwise exact potential
    r_ex = rms_over(Hf + V_ex * psi0, probe) / norm0
    # Sec. 4.5 identity: Im(V_ex) vs -lap_h(Phi)/2 (finite-difference analog)
    r_im = rms_over(np.imag(V_ex) + 0.5 * lap_fd, probe) / rms_phi
    return dict(rms_lap=rms_lap, rms_g2=rms_g2, r_ana=r_ana, r_ex=r_ex,
                r_im=r_im, err_t0=vel_error(psi0))


# ------------------------------------------------------------
# Split-step evolution (Strang): [V/2] [kinetic] [V/2]
# ------------------------------------------------------------
def kinetic_step(psi, dt):
    """exp(-i H_free dt) via QFT^3 on the uniform box."""
    k = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx * kx + ky * ky + kz * kz
    return ifftn(fftn(psi) * np.exp(-1j * 0.5 * k2 * dt))


def split_step(psi, V, T, dt):
    """Strang split-step with H = H_free + V (V=0 -> free evolution)."""
    n = max(1, int(round(T / dt)))
    dt_a = T / n
    expV2 = np.exp(-1j * V * dt_a / 2)
    for _ in range(n):
        psi = psi * expV2
        psi = kinetic_step(psi, dt_a)
        psi = psi * expV2
    return psi


def run_protocol(psi0, V, T_list, dt):
    """Incrementally evolve through sorted T_list; return (T, err) list."""
    out = []
    psi = psi0.copy()
    t_cur = 0.0
    for T in T_list:
        psi = split_step(psi, V, T - t_cur, dt)
        t_cur = T
        out.append((T, vel_error(psi)))
    return out


def body_surface_figure():
    """Visualize the von Karman double-ogive hull alone: the design meridian
    revolved about the axis (three-dimensional body surface)."""
    plt.rcParams.update({'axes.labelsize': 12, 'axes.titlesize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})
    fig = plt.figure(figsize=(8.0, 5.4))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92], projection='3d')
    xi_s = np.linspace(0.0, 1.0, 120)
    ph_s = np.linspace(0.0, 2.0 * np.pi, 64)
    XI_S, PH_S = np.meshgrid(xi_s, ph_s)
    XS = XI_S * L_BODY - 0.5 * L_BODY
    RS = vk_ogive_radius(XI_S)
    ax.plot_surface(XS, RS * np.cos(PH_S), RS * np.sin(PH_S),
                    color='0.62', rstride=3, cstride=4,
                    linewidth=0.2, edgecolor='0.3', rasterized=True)
    ax.set_box_aspect((2.6, 1.0, 1.0), zoom=1.3)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.8, 0.8)
    ax.view_init(elev=22, azim=78)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_zlabel('$z$')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig',
                       'body_revolution_surface.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print("    -> saved %s" % out)


def flowfield_figure():
    """Visualize the potential-flow field of the von Karman double-ogive hull
    in the symmetry plane z=0: velocity magnitude |v|/U_inf with streamlines
    (the pressure coefficient is omitted here because Cp = 1-(|v|/U_inf)^2
    carries no information beyond the magnitude panel). The solid body
    (psi < 0) is shaded and the design von Karman double-ogive meridian is
    drawn dashed for comparison.

    The field is reconstructed analytically on a fine 401x401 symmetric-plane
    grid (supersampled relative to the N=96 evolution grid), so the
    colour-mapped field, the body mask and the psi=0 contour share the same
    high-resolution source and render without pixelation or jagged edges."""
    NF = 501
    xf = np.linspace(-7.0, 7.0, NF)
    yf = np.linspace(-3.8, 3.8, NF)
    XF, YF = np.meshgrid(xf, yf, indexing='ij')
    ZF = np.zeros_like(XF)
    vx2, vy2, vz2 = velocity_3d(XF, YF, ZF)   # analytic gradients, z=0 plane
    spd = np.sqrt(vx2 ** 2 + vy2 ** 2 + vz2 ** 2) / U_INF
    psi2 = stream_psi(XF, np.abs(YF))
    body2 = psi2 < 0.0
    spd_m = np.ma.masked_where(body2, spd)
    # design von Karman double-ogive meridian (comparison with the psi=0 surface)
    xi_p = np.linspace(0.0, 1.0, 200)
    xp = xi_p * L_BODY - 0.5 * L_BODY
    rp = vk_ogive_radius(xi_p)

    # the figure prints at 0.52\textwidth; enlarge lettering so labels/ticks
    # stay >= 8 pt at the published size.
    plt.rcParams.update({'axes.labelsize': 12, 'axes.titlesize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})
    fig, ax = plt.subplots(figsize=(9.6, 4.3), constrained_layout=True)
    rng_x, rng_y = (-7.0, 7.0), (-3.0, 3.0)

    pcm = ax.pcolormesh(XF, YF, spd_m, cmap='viridis', vmin=0.5, vmax=1.25,
                        shading='auto', rasterized=True)
    # Vectorized RK4 streamlines from the left boundary: seed points on the
    # far-upstream line, integrate the analytic velocity field, and stop at
    # the body surface (psi = 0) or the domain edge.  This yields the clean,
    # textbook streamlines of axisymmetric potential flow (no NaN masking, no
    # spurious short lines near the body), with every line entering from the
    # left and leaving downstream.
    def _vfield(xs, ys):
        vx, vy, _ = velocity_3d(xs, ys, np.zeros_like(xs))
        return vx, vy

    n_seed = 19
    y_seed = np.linspace(-3.5, 3.5, n_seed)
    xs = np.full(n_seed, -6.7)
    ys = y_seed.copy()
    dt = 0.02
    lines = [[(x0, y0)] for x0, y0 in zip(xs, ys)]
    active = np.ones(n_seed, dtype=bool)
    for _ in range(3200):
        if not active.any():
            break
        xa, ya = xs[active], ys[active]
        k1x, k1y = _vfield(xa, ya)
        k2x, k2y = _vfield(xa + 0.5 * dt * k1x, ya + 0.5 * dt * k1y)
        k3x, k3y = _vfield(xa + 0.5 * dt * k2x, ya + 0.5 * dt * k2y)
        k4x, k4y = _vfield(xa + dt * k3x, ya + dt * k3y)
        xs[active] += dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        ys[active] += dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        for k in np.where(active)[0]:
            lines[k].append((xs[k], ys[k]))
        hit_body = stream_psi(xs[active], np.abs(ys[active])) < 0.0
        out = (np.abs(xs[active]) > 6.8) | (np.abs(ys[active]) > 3.7)
        active[active] = ~(hit_body | out)

    for k in range(n_seed):
        arr = np.array(lines[k])
        if len(arr) > 5:
            ax.plot(arr[:, 0], arr[:, 1], color='0.35', lw=0.9, zorder=3)
    # body surface = closed streamline psi = 0 of the line-source flow
    ax.contourf(XF, YF, psi2, levels=[psi2.min() - 1.0, 0.0], colors='0.35',
                zorder=4)
    ax.contour(XF, YF, psi2, levels=[0.0], colors='k', linewidths=0.8,
               zorder=5)
    ax.plot(xp, rp, '--', color='0.85', lw=1.0, zorder=5)
    ax.plot(xp, -rp, '--', color='0.85', lw=1.0, zorder=5)
    ax.set_xlim(rng_x); ax.set_ylim(rng_y)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$|\mathbf{v}|/U_\infty$')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig',
                       'body_revolution_flowfield.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print("    -> saved %s" % out)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print(" 3D bodies of revolution: exact Bernoulli encoding, von Karman double-ogive")
    print(" (L=4.0, R_max=0.5, M=%d line-source panels), U_inf=1 along x, "
          "N=%d, box L=%.1f" % (M, N, L))
    print(" psi=0 matched to design meridian: max|dR| = %.1e" % MATCH_ERR)
    print("=" * 78)

    # ---- A/B. harmonicity + eigenstate residuals (no evolution) ----
    ck = eigenstate_checks()
    print("\n[A] Harmonicity of the analytic potential (probe, FD 7-point):")
    print("    RMS(lap_h Phi)          = %.4e" % ck['rms_lap'])
    print("    RMS(|grad Phi|)         = %.4e" % ck['rms_g2'])
    print("    RMS(lap_h Phi)/U_inf/L  = %.4e" % (ck['rms_lap'] / L))
    print("\n[B] Eigenstate residuals (spectral H_free, probe):")
    print("    ||(H_free+V_ana) psi0||/||psi0|| = %.4e" % ck['r_ana'])
    print("    ||(H_free+V_ex)   psi0||/||psi0|| = %.4e" % ck['r_ex'])
    print("    Im(V_ex) vs -lap_h(Phi)/2 (Sec. 4.5 analog) = %.4e" % ck['r_im'])
    print("    t=0 decoding floor err_t0 = %.4e" % ck['err_t0'])
    print("   [%.0fs]" % (time.time() - t0))

    # ---- geometry + flow field figures ----
    print("\n[Geometry + flow field]")
    body_surface_figure()
    flowfield_figure()
    print("   [%.0fs]" % (time.time() - t0))

    VEX = _build_vex()   # pointwise exact potential (cached)

    # ---- C. stationarity table ----
    DT = 1e-3
    T_TABLE = [0.001, 0.005, 0.01, 0.02, 0.04]
    print("\n[C] Stationarity (split-step, dt=%.1e, probe sphere r<3.6):" % DT)
    print("    %-8s | %-11s %-15s %-13s" %
          ("T", "err[free]", "err[H_B analytic]", "err[H_B exact]"))
    print("    " + "-" * 58)
    res_f = run_protocol(psi0, np.zeros_like(psi0), T_TABLE, DT)
    res_b = run_protocol(psi0, V_ANA, T_TABLE, DT)
    res_x = run_protocol(psi0, VEX, T_TABLE, DT)
    for (T, ef), (_, eb), (_, ex) in zip(res_f, res_b, res_x):
        print("    %8.3f | %11.4f %15.4f %13.4f" % (T, ef, eb, ex))
    # dt-convergence spot check of the pinned protocol at T=0.04
    ex_dt2 = run_protocol(psi0, VEX, [0.04], 2.5e-4)[0][1]
    print("    [dt check] V_ex at T=0.04: dt=1e-3 -> %.4e, dt=2.5e-4 -> %.4e"
          % (res_x[-1][1], ex_dt2))
    print("   [%.0fs]" % (time.time() - t0))

    # ---- Figure: dense T scan ----
    T_FIG = np.linspace(0.002, 0.04, 10)
    print("\n[Figure] dense T scan over %d points..." % len(T_FIG))
    f_f = run_protocol(psi0, np.zeros_like(psi0), list(T_FIG), DT)
    b_f = run_protocol(psi0, V_ANA, list(T_FIG), DT)
    x_f = run_protocol(psi0, VEX, list(T_FIG), DT)
    err_f = np.array([e for _, e in f_f])
    err_b = np.array([e for _, e in b_f])
    err_x = np.array([e for _, e in x_f])

    # This figure prints at 0.74\textwidth (~0.73 downscale); enlarge the
    # lettering so labels/ticks/legend stay >= 8 pt at the published size.
    plt.rcParams.update({'axes.labelsize': 12, 'axes.titlesize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.fontsize': 11})

    fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
    # t=0 decoding floor as a shaded band (the value is read directly from the
    # y-axis tick at 3e-3, so no arrow annotation is needed)
    floor = ck['err_t0']
    ax.axhspan(0.8 * floor, 1.2 * floor, color='#636e72', alpha=0.13, lw=0, zorder=0)
    ax.axhline(floor, color='#636e72', ls=':', lw=1.4, zorder=1)
    ax.semilogy(T_FIG, err_f, 'o-', color='#d63031', lw=1.4, ms=4,
                label='Free-particle evolution $e^{-iH_{\\rm free}t}$',
                markeredgecolor='white', markeredgewidth=0.6)
    ax.semilogy(T_FIG, err_b, 's--', color='#0984e3', lw=1.4, ms=4,
                label='$H_B$, analytic $V=-|\\nabla\\Phi|^2/2$',
                markeredgecolor='white', markeredgewidth=0.6)
    ax.semilogy(T_FIG, err_x, '^-', color='#00b894', lw=1.5, ms=5,
                label='$H_B$, exact eigenstate',
                markeredgecolor='white', markeredgewidth=0.6)
    ax.set_xlabel('Evolution time $T$ [s]')
    ax.set_ylabel(r'Velocity error $\|\mathbf{v}-\mathbf{v}_\mathrm{ref}\|_\mathrm{RMS}/U_\infty$ (interior)')
    # y-range tight to the data (floor 2.7e-3 .. free 8.5e-3) to avoid dead space
    ax.set_ylim(1.5e-3, 2e-2)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='0.6')
    ax.grid(True, which='both', alpha=0.25, lw=0.6)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig',
                       'body_revolution_stationarity.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print("    -> saved %s" % out)
    print("   [%.0fs]" % (time.time() - t0))

    # ---- separate stacked panels (a) stationarity + (b) staircase ----
    # Two standalone single-column figures (no embedded panel titles;
    # (a)/(b) tags are typeset by LaTeX above each panel).
    N_arr = np.array([32, 64, 128, 256, 512])
    cp_shell = np.array([2.2647e-02, 5.5076e-03, 1.3576e-03,
                         3.3686e-04, 8.3903e-05])
    vel_probe = np.array([6.2790e-02, 1.0506e-02, 5.3513e-03,
                          1.5730e-03, 3.8640e-04])
    vel_far = np.array([1.1074e-02, 2.6882e-03, 6.6190e-04,
                        1.6421e-04, 4.0892e-05])
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                          'fig')

    fig_a, ax1 = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
    ax1.axhspan(0.8 * floor, 1.2 * floor, color='#636e72', alpha=0.13, lw=0)
    ax1.axhline(floor, color='#636e72', ls=':', lw=1.4)
    ax1.semilogy(T_FIG, err_f, 'o-', color='#d63031', lw=1.4, ms=4,
                 label='Free-particle',
                 markeredgecolor='white', markeredgewidth=0.6)
    # The analytic-Bernoulli curve (err_b) is deliberately NOT drawn here.
    # On the uniform grid its O(h^2) chain-rule deficit dominates the residual
    # and leaves it on top of the free curve (measured: 8.35e-3 vs 8.49e-3 at
    # T = 0.04, i.e. 1.6% apart), so plotting it as a third, separate branch
    # would misrepresent the result.  It is reported in the caption instead.
    ax1.semilogy(T_FIG, err_x, '^-', color='#00b894', lw=1.5, ms=5,
                 label=r'$H_B$, exact eigenstate',
                 markeredgecolor='white', markeredgewidth=0.6)
    ax1.set_xlabel('Evolution time $T$ [s]')
    ax1.set_ylabel(r'Velocity error (interior)')
    ax1.set_ylim(1.5e-3, 2e-2)
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='0.6')
    ax1.grid(True, which='both', alpha=0.25, lw=0.6)
    out_a = os.path.join(figdir, '3d_stationarity_a.pdf')
    fig_a.savefig(out_a, dpi=400)
    plt.close(fig_a)
    print("    -> saved %s" % out_a)

    fig_b, ax2 = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
    ax2.loglog(N_arr, cp_shell, 's-', color='#1a365d', lw=1.6, ms=6,
               label=r'near-body $C_p$')
    ax2.loglog(N_arr, vel_probe, 'o-', color='#c0392b', lw=1.6, ms=6,
               label=r'interior probe')
    ax2.loglog(N_arr, vel_far, '^-', color='#2e8b57', lw=1.6, ms=6,
               label=r'far field')
    ax2.loglog([N_arr[0], N_arr[-1]],
               [cp_shell[0], cp_shell[0] * (N_arr[0] / N_arr[-1]) ** 2],
               'k--', lw=1.1, label=r'$p=2$')
    ax2.annotate('transient $p\\approx1$\n(64$\\to$128)',
                 (N_arr[2], vel_probe[2]),
                 xytext=(N_arr[2] * 0.60, vel_probe[2] * 3.5),
                 color='#c0392b', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.9))
    ax2.set_xscale('log')
    ax2.set_xticks(N_arr)
    ax2.set_xticklabels(['32', '64', '128', '256', '512'])
    ax2.minorticks_off()
    ax2.set_xlabel(r'Grid resolution $N$ (doubled per step)')
    ax2.set_ylabel(r'RMS error')
    ax2.legend(frameon=False, loc='upper right')
    ax2.grid(True, which='major', alpha=0.25)

    out_b = os.path.join(figdir, '3d_stationarity_b.pdf')
    fig_b.savefig(out_b, dpi=400)
    plt.close(fig_b)
    print("    -> saved %s" % out_b)

    # ---- summary numbers for the manuscript ----
    print("\n[Summary for paper]")
    print("    harmonicity   : RMS(lap_h Phi) = %.2e" % ck['rms_lap'])
    print("    eigen residual: analytic V = %.2e, exact V = %.2e" % (ck['r_ana'], ck['r_ex']))
    print("    floor         : err_t0 = %.2e" % ck['err_t0'])
    print("    T=0.04        : free %.2e, H_B analytic %.2e, H_B exact %.2e"
          % (res_f[-1][1], res_b[-1][1], res_x[-1][1]))


def _build_vex():
    """Pointwise exact potential from the spectral operator.
    Guarded: where |psi0|~0 (window taper) the potential is irrelevant to the
    (vanishing) state, so set it to V_ANA to keep the field finite for the FFT."""
    Hf = kinetic_spectral(psi0)
    with np.errstate(divide='ignore', invalid='ignore'):
        V = -Hf / psi0
    small = np.abs(psi0) < 1e-8
    V[small] = V_ANA[small]
    return V


if __name__ == "__main__":
    main()
