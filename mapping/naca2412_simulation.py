"""
NACA 2412 翼型共形映射 + Schrödinger-Madelung 验证
====================================================
NACA 2412: 最大弯度 2% at 0.4c, 最大厚度 12%
这是经典的带弯度翼型,用于测试方法对非对称压力分布的适应能力。

流程:
  1. Theodorsen梯度下降 → NACA 2412的共形映射 Laurent系数
  2. 映射到ζ平面圆柱绕流
  3. Schrödinger-Madelung谱方法求解
  4. 共形逆映射回z平面
  5. 表面Cp分布、与解析圆柱解的误差评估
"""
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import warnings
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

from flow_common import smooth_boundary_window, madelung_encode, madelung_decode

_HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

U_INF = 1.0
HBAR = 1.0
MASS = 1.0
ZETA_RANGE = 5.0
PLOT_LIM = 4.4  # window-effective domain: |x| < 4.4 => cosine taper window == 1
DT = 0.001
N_STEPS = 10  # T=0.01s
N_EVOLVE = 128

# ================================================================
#  1. NACA 2412 几何定义
# ================================================================
def naca2412_thickness(x):
    """NACA 4-digit thickness function."""
    x = np.maximum(x, 0.0)
    return 0.12/0.2 * (0.2969 * np.sqrt(x) - 0.126 * x
                        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

def naca2412_camber(x):
    """NACA 2412 camber line: max camber 2% at 4% chord."""
    x = np.maximum(x, 0.0)
    m = 0.02   # max camber
    p = 0.40   # position of max camber
    yc = np.where(x < p,
                  m / p**2 * (2*p*x - x**2),
                  m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2))
    return yc

def naca2412_camber_slope(x):
    m, p = 0.02, 0.40
    return np.where(x < p,
                    2*m/p**2 * (p - x),
                    2*m/(1-p)**2 * (p - x))

def naca2412_airfoil(theta):
    """NACA 2412 upper/lower surface at parameter theta.

    The camber slope is a function of the chordwise coordinate, so it must be
    evaluated at x and not at theta.  Passing theta (the circle angle, 0 to
    2 pi) swept the slope over +0.10 to -0.30, i.e. a normal rotated by up to
    17 degrees instead of the 5.7 degrees the section actually has, which
    both thinned the section and displaced its camber line.
    """
    x = 0.5 * (1.0 + np.cos(theta))
    yt = naca2412_thickness(x)
    yc = naca2412_camber(x)
    dyc = naca2412_camber_slope(x)
    sgn = np.where(np.sin(theta) >= 0, 1.0, -1.0)
    th_angle = np.arctan(dyc)
    xu = x - yt * np.sin(th_angle)
    yu = yc + yt * np.cos(th_angle)
    xl = x + yt * np.sin(th_angle)
    yl = yc - yt * np.cos(th_angle)
    x_surf = np.where(np.sin(theta) >= 0, xu, xl)
    y_surf = np.where(np.sin(theta) >= 0, yu, yl)
    return x_surf + 1j * y_surf

# ================================================================
#  2. Theodorsen 梯度下降 (同NACA 0012, 但目标翼型改为NACA 2412)
# ================================================================
def theodorsen_mapping_naca2412(n_pts=2048):
    """Gradient descent Theodorsen optimization for NACA 2412."""
    s = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    k_freq = fftfreq(n_pts) * n_pts

    def idx(f):
        return np.argmin(np.abs(k_freq - f))

    theta = s.copy()
    velocity = np.zeros(n_pts)
    max_iter = 2000
    momentum = 0.9
    lr_base = 5.0
    tol = 1e-12
    best_dist = 1.0
    best_theta = theta.copy()

    print("  Theodorsen GD for NACA 2412...")
    for it in range(max_iter):
        z = naca2412_airfoil(theta)
        z_hat = fft(z) / n_pts
        pos_energy = sum(np.abs(z_hat[idx(k)])**2 for k in range(2, n_pts//2))
        total_energy = sum(np.abs(z_hat)**2)
        dist = pos_energy / total_energy if total_energy > 0 else 0.0
        if dist < best_dist:
            best_dist = dist
            best_theta = theta.copy()
        if it % 200 == 0:
            print(f"    Iter {it:4d}: E={dist:.4e}")
        if dist < tol:
            print(f"    Converged at iter {it}")
            break

        z_hat_pos = np.zeros(n_pts, dtype=complex)
        for k in range(2, n_pts//2):
            z_hat_pos[idx(k)] = z_hat[idx(k)]
        z_pos = ifft(z_hat_pos * n_pts)

        # Tangent computed via finite difference
        eps = 1e-8
        theta_p = theta + eps
        z_p = naca2412_airfoil(theta_p)
        dz_dtheta = (z_p - z) / eps

        grad = (2.0 / n_pts) * np.real(z_pos * np.conj(dz_dtheta))

        # Armijo line search
        lr = lr_base
        for _ in range(5):
            theta_trial = np.mod(theta - lr * grad, 2 * np.pi)
            zt = naca2412_airfoil(theta_trial)
            zht = fft(zt) / n_pts
            pet = sum(np.abs(zht[idx(k)])**2 for k in range(2, n_pts//2))
            tet = sum(np.abs(zht)**2)
            dt = pet / tet if tet > 0 else 0.0
            if dt < dist:
                break
            lr *= 0.5
        else:
            lr = lr_base * 0.1
            lr_base = max(lr_base * 0.8, 0.001)

        velocity = momentum * velocity - lr * grad
        theta = theta + velocity
        theta = np.mod(theta, 2 * np.pi)
        theta = np.sort(theta)
        if dist < best_dist * 1.5:
            lr_base = min(lr_base * 1.01, 20.0)

    theta = best_theta
    z_af = naca2412_airfoil(theta)
    z_hat = fft(z_af) / n_pts
    R = np.abs(z_hat[idx(1)])
    phase_R = np.angle(z_hat[idx(1)])
    z_aligned = z_af * np.exp(-1j * phase_R)
    z_hat_aligned = fft(z_aligned) / n_pts
    R = np.real(z_hat_aligned[idx(1)])

    n_coeff = n_pts // 2
    coeffs = np.zeros(n_coeff + 1, dtype=complex)
    coeffs[0] = z_hat_aligned[idx(0)]
    for k in range(1, n_coeff + 1):
        coeffs[k] = (R**k) * z_hat_aligned[idx(-k)]

    # Trim to significant terms
    mags = np.abs(coeffs)
    k_cut = min(len(coeffs)-1, max(16, np.searchsorted(mags[::-1], 1e-15, side='right')))
    k_cut = len(coeffs) - 1 - k_cut
    k_cut = max(k_cut, 8)
    coeffs = coeffs[:k_cut+1]

    print(f"    R={R:.8f}, {len(coeffs)} Laurent coefficients")
    print(f"    Distortion energy: {best_dist:.4e}")
    
    # TEX-ready Laurent coefficients
    print("\n  --- TEX: NACA 2412 Laurent coefficients ---")
    for k in range(min(8, len(coeffs))):
        c = coeffs[k]
        print(f"  c_{k} &= {c.real:+.12e} {c.imag:+.2e}\\\\i \\\\\\\\")
    print("  (|c_0|=", abs(coeffs[0]), "|c_1|=", abs(coeffs[1]), 
          "|c_2|=", abs(coeffs[2]), "|c_3|=", abs(coeffs[3]), ")")
    return R, coeffs, theta, z_af, best_dist

# ================================================================
#  3. 共形映射函数
# ================================================================
def fwd_map(zeta, coeffs):
    zeta = np.asarray(zeta, dtype=complex)
    z = zeta.astype(complex)
    z += coeffs[0]
    for k in range(1, len(coeffs)):
        z += coeffs[k] / (zeta**k)
    return z

def dz_dzeta_map(zeta, coeffs):
    zeta = np.asarray(zeta, dtype=complex)
    dz = np.ones_like(zeta, dtype=complex)
    for k in range(1, len(coeffs)):
        dz -= k * coeffs[k] / (zeta**(k+1))
    return dz

# ================================================================
#  4. Schrödinger-Madelung求解 + 共形映射
# ================================================================
def analytic_cylinder_potential(X, Y, a, u_inf, alpha=0.0):
    Zs = X + 1j * Y
    r = np.abs(Zs)
    r_safe = np.maximum(r, a * 0.01)
    theta = np.angle(Zs)
    return u_inf * (r_safe + a**2 / r_safe) * np.cos(theta - alpha)

def analytic_cylinder_velocity(X, Y, a, u_inf, gamma, alpha=0.0):
    Zs = X + 1j * Y
    r = np.abs(Zs)
    r_safe = np.maximum(r, a * 0.01)
    theta = np.angle(Zs)
    ur = u_inf * (1 - a**2 / r_safe**2) * np.cos(theta - alpha)
    ut = -u_inf * (1 + a**2 / r_safe**2) * np.sin(theta - alpha) + gamma / (2*np.pi*r_safe)
    u = ur * np.cos(theta) - ut * np.sin(theta)
    v = ur * np.sin(theta) + ut * np.cos(theta)
    return u, v

def analytic_circulation_velocity(X, Y, gamma):
    r_sq = np.maximum(X**2 + Y**2, 1e-12)
    u = -gamma * Y / (2 * np.pi * r_sq)
    v = gamma * X / (2 * np.pi * r_sq)
    return u, v

def build_inside_mask(nx, ny, dx, dy, a):
    x_c = (np.arange(nx) - nx // 2) * dx
    y_c = (np.arange(ny) - ny // 2) * dy
    mask = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            if abs(complex(x_c[j], y_c[i])) < a:
                mask[i, j] = True
    return mask

def run_2412_simulation():
    print("=" * 60)
    print("  NACA 2412: Schrödinger-Madelung + Conformal Mapping")
    print("=" * 60)

    # 1. Theodorsen mapping
    print("[1] Computing conformal mapping for NACA 2412...")
    t0 = time.time()
    R, coeffs, theta_map, z_af_map, dist = theodorsen_mapping_naca2412(n_pts=2048)
    print(f"    Time: {time.time()-t0:.2f}s")

    # 2. Setup evolution grid (zeta-plane, cylinder at origin)
    ne = N_EVOLVE
    dx = 2 * ZETA_RANGE / ne
    dy = 2 * ZETA_RANGE / ne
    zeta_x = (np.arange(ne) - ne//2) * dx
    zeta_y = (np.arange(ne) - ne//2) * dy
    XX, YY = np.meshgrid(zeta_x, zeta_y)

    # Cylinder at origin (mapped from NACA 2412)
    # The NACA 2412 maps to a near-circle centered near origin
    # We use R directly as cylinder radius
    # Attack angle 5 degrees for lift validation
    alpha = np.radians(5.0)
    gamma = 4 * np.pi * R * U_INF * np.sin(alpha)  # Kutta condition

    # 3. Analytic reference (cylinder in zeta-plane)
    u_ana, v_ana = analytic_cylinder_velocity(XX, YY, R, U_INF, gamma, alpha)
    inside = build_inside_mask(ne, ne, dx, dy, R)
    u_ana[inside] = np.nan
    v_ana[inside] = np.nan
    vel_ana = np.sqrt(u_ana**2 + v_ana**2)

    # 4. Schrödinger-Madelung
    print("[2] Schrödinger-Madelung evolution...")
    phi_init = analytic_cylinder_potential(XX, YY, R, U_INF, alpha)
    window = smooth_boundary_window(XX, YY, ZETA_RANGE)
    psi = madelung_encode(phi_init) * window
    psi[inside] = 0.0

    kx = 2 * np.pi * np.fft.fftfreq(ne)
    ky = 2 * np.pi * np.fft.fftfreq(ne)
    KX, KY = np.meshgrid(kx, ky)
    k_sq = KX**2 + KY**2
    T_total = N_STEPS * DT
    k_cut = np.pi * 32 / (2 * ZETA_RANGE)
    filt = np.exp(-(k_sq / k_cut**2)**2)

    psi_k = np.fft.fft2(psi)
    psi_k = psi_k * np.exp(-1j * k_sq * T_total / (2*MASS)) * filt
    psi_ev = np.fft.ifft2(psi_k)
    psi_ev[inside] = 0.0
    psi_ev = psi_ev * window

    u_q, v_q = madelung_decode(psi_ev, dx=dx, dy=dy)
    u_circ, v_circ = analytic_circulation_velocity(XX, YY, gamma)
    u_q += u_circ
    v_q += v_circ
    u_q[inside] = np.nan
    v_q[inside] = np.nan
    vel_q = np.sqrt(u_q**2 + v_q**2)

    # Error
    taper_mask = window > 0.99
    outside = ~inside & taper_mask
    valid = outside & (vel_ana > 0.05)
    rel_err = np.nanmean(np.abs(vel_q[valid] - vel_ana[valid]) / (vel_ana[valid] + 1e-10))
    print(f"    Zeta-plane rel. error: {rel_err:.4e}")

    # 5. Conformal map to z-plane for display
    #   For display, generate high-res z-plane image via conformal mapping
    print("[3] Conformal mapping: zeta -> z (NACA 2412 airfoil)...")
    n_disp = 400
    dx_d = 2 * ZETA_RANGE / n_disp
    dy_d = 2 * ZETA_RANGE / n_disp
    zx_d = (np.arange(n_disp) - n_disp//2) * dx_d
    zy_d = (np.arange(n_disp) - n_disp//2) * dy_d
    XXd, YYd = np.meshgrid(zx_d, zy_d)
    ZETA_D = XXd + 1j * YYd

    in_d = build_inside_mask(n_disp, n_disp, dx_d, dy_d, R)
    # Exclude the boundary taper zone (window < 1) from the display: keep only
    # the window-effective domain where the Madelung-decoded velocity is valid.
    in_d |= (np.abs(XXd) > PLOT_LIM) | (np.abs(YYd) > PLOT_LIM)

    # Interpolate velocity to display grid
    from scipy.interpolate import RegularGridInterpolator
    iu = RegularGridInterpolator((zeta_y, zeta_x), u_q, bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((zeta_y, zeta_x), v_q, bounds_error=False, fill_value=0.0)
    pts = np.stack([YYd.ravel(), XXd.ravel()], axis=-1)
    u_q_d = iu(pts).reshape(n_disp, n_disp)
    v_q_d = iv(pts).reshape(n_disp, n_disp)
    u_q_d[in_d] = np.nan
    v_q_d[in_d] = np.nan
    vel_q_d = np.sqrt(u_q_d**2 + v_q_d**2)

    # Conformal map: z-plane points and velocity
    Z_D = fwd_map(ZETA_D, coeffs)
    dz_dz = dz_dzeta_map(ZETA_D, coeffs)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_zeta = u_q_d - 1j * v_q_d
        w_z = np.where(np.abs(dz_dz) > 1e-12, w_zeta / dz_dz, 0.0)
    u_z = w_z.real
    v_z = -w_z.imag

    # Analytic reference on the display grid
    u_ad, v_ad = analytic_cylinder_velocity(XXd, YYd, R, U_INF, gamma, alpha)
    u_ad[in_d] = np.nan
    v_ad[in_d] = np.nan
    vel_ad = np.sqrt(u_ad**2 + v_ad**2)
    w_zeta_a = u_ad - 1j * v_ad
    w_z_a = np.where(np.abs(dz_dz) > 1e-12, w_zeta_a / dz_dz, 0.0)
    u_az = w_z_a.real
    v_az = -w_z_a.imag

    # 6. NACA 2412 surface Cp
    print("[4] Computing surface Cp...")
    n_surf = 800
    theta_s = np.linspace(0, 2*np.pi, n_surf)
    zeta_s = R * np.exp(1j * theta_s)

    # Cylinder surface velocity in zeta-plane
    zs = R * np.exp(1j * theta_s)
    dw = U_INF * (np.exp(-1j*alpha) - R**2 * np.exp(1j*alpha) / zs**2) - 1j*gamma/(2*np.pi*zs)
    dz = dz_dzeta_map(zs, coeffs)
    v_surf = dw / dz
    v_mag = np.abs(v_surf)
    Cp_surf = 1.0 - (v_mag / U_INF)**2
    z_surf = fwd_map(zs, coeffs)
    x_surf = np.real(z_surf)

    # Sorted for Cp plot
    n = len(x_surf)
    theta_idx = np.argsort(theta_s)
    x_sorted = x_surf[theta_idx]
    Cp_sorted = Cp_surf[theta_idx]
    # Split upper/lower
    mid = np.argmax(x_sorted)  # leading edge
    # Actually use y coordinate
    y_surf = np.imag(z_surf)
    upper = y_surf > np.median(y_surf)
    lower = ~upper

    print(f"    Gamma={gamma:.4f}, CL={2*gamma:.4f}")
    print(f"    Cp_min (full surface)={np.min(Cp_surf):.4f}")
    
    # Refined: exclude leading edge stagnation region (|x - x_LE| < 0.02c)
    x_le_norm = x_sorted[-1]  # leading edge x coordinate
    non_le_mask = np.abs(x_surf - x_le_norm) > 0.02
    if np.sum(non_le_mask) > 0:
        Cp_min_refined = np.min(Cp_surf[non_le_mask])
        print(f"    Cp_min (excl. LE 2% chord)={Cp_min_refined:.4f}")
    
    # Cp error statistics (full surface and excluding LE)
    Cp_ana = 1.0 - (2*np.sin(theta_s - alpha) + gamma/(2*np.pi*R*U_INF))**2
    Cp_err_full = np.sqrt(np.mean((Cp_surf - Cp_ana)**2))
    Cp_err_noLE = np.sqrt(np.mean(((Cp_surf - Cp_ana)[non_le_mask])**2)) if np.sum(non_le_mask) > 0 else Cp_err_full
    # Relative Cp error using L2 norm of Cp_ana as denominator
    Cp_ana_norm = np.sqrt(np.mean(Cp_ana**2))
    Cp_rel_full = Cp_err_full / Cp_ana_norm if Cp_ana_norm > 0 else Cp_err_full
    Cp_rel_noLE = Cp_err_noLE / (np.sqrt(np.mean((Cp_ana[non_le_mask])**2)) + 1e-15) if np.sum(non_le_mask) > 0 else Cp_rel_full
    print(f"    Cp RMS error (full): {Cp_err_full:.4f} (relative: {Cp_rel_full:.4f})")
    print(f"    Cp RMS error (no LE): {Cp_err_noLE:.4f} (relative: {Cp_rel_noLE:.4f})")
    print(f"    Cp amplification factor (surface): {Cp_rel_full / (np.mean(np.abs(v_mag - 2*U_INF*np.abs(np.sin(theta_s-alpha))))/U_INF + 1e-10):.2f}")

    # ================================================================
    #  7. Visualization
    # ================================================================
    d = os.path.dirname(os.path.abspath(__file__))

    # --- Fig 1: NACA 2412 profile ---
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(np.real(z_af_map), np.imag(z_af_map), 'b-', lw=1.5)
    ax.fill(np.real(z_af_map), np.imag(z_af_map), alpha=0.2, color='skyblue')
    ax.set_aspect('equal')
    ax.set_xlabel('$x/c$', fontsize=12)
    ax.set_ylabel('$y/c$', fontsize=12)
    ax.set_title('NACA 2412 Airfoil Profile', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.20, 0.20)
    fig.tight_layout()
    p = os.path.join(d, 'naca2412_profile.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {os.path.basename(p)}")

    # --- Fig 2: Surface Cp ---
    fig, ax = plt.subplots(figsize=(10, 5))
    # Sort upper surface by x (front to back)
    xu = x_surf[upper]
    cpu = Cp_surf[upper]
    idx_u = np.argsort(xu)
    ax.plot(xu[idx_u], cpu[idx_u], 'r-', lw=1.5, label='Upper surface (suction)')
    xl = x_surf[lower]
    cpl = Cp_surf[lower]
    idx_l = np.argsort(xl)
    ax.plot(xl[idx_l], cpl[idx_l], 'b--', lw=1.5, label='Lower surface (pressure)')
    ax.fill_between(xu[idx_u], cpu[idx_u], 0, alpha=0.08, color='red')
    ax.fill_between(xl[idx_l], cpl[idx_l], 0, alpha=0.08, color='blue')
    ax.invert_yaxis()
    ax.set_xlabel('$x/c$', fontsize=12)
    ax.set_ylabel('$C_p$', fontsize=12)
    ax.set_title(r'NACA 2412 Surface $C_p$ Distribution ($\alpha=5^\circ$, $\Gamma=%.4f$)' % gamma,
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    p = os.path.join(d, 'naca2412_surface_cp.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {os.path.basename(p)}")

    # --- Fig 3: z-plane velocity (Schrödinger) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0),
                                   constrained_layout=True)

    # Analytic
    from matplotlib.tri import Triangulation
    out_mask = ~in_d
    x_f = np.real(Z_D).ravel()[out_mask.ravel()]
    y_f = np.imag(Z_D).ravel()[out_mask.ravel()]
    # z-plane analytic velocity
    u_az_f = u_az.ravel()[out_mask.ravel()]
    v_az_f = v_az.ravel()[out_mask.ravel()]
    vaz_f = np.sqrt(u_az_f**2 + v_az_f**2)

    n_tri = min(len(x_f), 15000)
    idx_t = np.linspace(0, len(x_f)-1, n_tri, dtype=int)
    tri = Triangulation(x_f[idx_t], y_f[idx_t])

    # Square display window covering the mapped window-effective domain.
    # set_box_aspect(1) + equal aspect => the colored field fills the frame.
    zr = np.real(Z_D)[out_mask]
    zi = np.imag(Z_D)[out_mask]
    xc = 0.5 * (np.nanmax(zr) + np.nanmin(zr))
    yc = 0.5 * (np.nanmax(zi) + np.nanmin(zi))
    half = 0.5 * max(np.nanmax(zr) - np.nanmin(zr), np.nanmax(zi) - np.nanmin(zi))

    tcf = ax1.tripcolor(tri, vaz_f[idx_t], cmap='turbo', vmin=0,
                        vmax=np.nanpercentile(vaz_f, 98), shading='gouraud')
    af_x = np.real(z_af_map)
    af_y = np.imag(z_af_map)
    ax1.fill(af_x, af_y, color='#333', alpha=0.85, zorder=5)
    ax1.plot(af_x, af_y, 'w-', lw=1.5, zorder=5)
    ax1.set_aspect('equal')
    ax1.set_xlim(xc-half, xc+half)
    ax1.set_ylim(yc-half, yc+half)
    ax1.set_box_aspect(1)
    ax1.set_title('(a) Analytic $|V|$ (potential flow)', fontsize=13)
    ax1.set_xlabel('$x/c$')
    ax1.set_ylabel('$y/c$')
    fig.colorbar(tcf, ax=ax1, shrink=0.9)

    u_z_f = u_z.ravel()[out_mask.ravel()]
    v_z_f = v_z.ravel()[out_mask.ravel()]
    vz_f = np.sqrt(u_z_f**2 + v_z_f**2)

    tcf2 = ax2.tripcolor(tri, vz_f[idx_t], cmap='turbo', vmin=0,
                         vmax=np.nanpercentile(vaz_f, 98), shading='gouraud')
    ax2.fill(af_x, af_y, color='#333', alpha=0.85, zorder=5)
    ax2.plot(af_x, af_y, 'w-', lw=1.5, zorder=5)
    ax2.set_aspect('equal')
    ax2.set_xlim(xc-half, xc+half)
    ax2.set_ylim(yc-half, yc+half)
    ax2.set_box_aspect(1)
    ax2.set_title(f'(b) Schrödinger-Madelung $|V|$\n'
                  f'$\\zeta$-plane rel. err = {rel_err:.2e}',
                  fontsize=13)
    ax2.set_xlabel('$x/c$')
    ax2.set_ylabel('$y/c$')
    fig.colorbar(tcf2, ax=ax2, shrink=0.9)

    p = os.path.join(d, '..', 'fig', 'naca2412_flow_comparison.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {os.path.basename(p)}")

    # --- Fig 4: NACA 0012 vs NACA 2412 Cp comparison ---
    # Use NACA 0012 coefficients (from existing code)
    from theodorsen_mapping import compute_conformal_mapping
    R12, coeffs12 = [None, None]  # will be set
    try:
        R12, coeffs12_src, _, _, _, _, _, _, _ = compute_conformal_mapping(n_pts=2048)
        # Truncate
        mags12 = np.abs(coeffs12_src)
        kc12 = min(len(coeffs12_src)-1, max(16, np.searchsorted(mags12[::-1], 1e-15, side='right')))
        kc12 = len(coeffs12_src) - 1 - kc12
        kc12 = max(kc12, 8)
        coeffs12 = coeffs12_src[:kc12+1]
        print(f"\n[5] NACA 0012 for comparison: R={R12:.8f}")
    except Exception as e:
        print(f"  NACA 0012 load failed: {e}, skipping comparison")
        R12, coeffs12 = None, None

    if R12 is not None:
        # NACA 0012 surface Cp at alpha=5
        Gamma12 = 4 * np.pi * R12 * U_INF * np.sin(alpha)
        zs12 = R12 * np.exp(1j * theta_s)
        dw12 = U_INF * (np.exp(-1j*alpha) - R12**2 * np.exp(1j*alpha) / zs12**2) \
               - 1j*Gamma12 / (2*np.pi*zs12)
        dz12 = np.ones_like(zs12, dtype=complex)
        for k in range(1, len(coeffs12)):
            dz12 -= k * coeffs12[k] / (zs12**(k+1))
        vs12 = dw12 / dz12
        Cp12 = 1.0 - (np.abs(vs12) / U_INF)**2
        z12_surf = zs12.copy()
        z12_surf += coeffs12[0]
        for k in range(1, len(coeffs12)):
            z12_surf += coeffs12[k] / (zs12**k)
        x12 = np.real(z12_surf)
        y12 = np.imag(z12_surf)
        upper12 = y12 > np.median(y12)

        # Compute CL for both
        CL12 = 2 * Gamma12
        Gamma2412 = gamma
        CL2412 = 2 * Gamma2412

        fig, ax = plt.subplots(figsize=(10, 5.5))
        xu12 = x12[upper12]; cpu12 = Cp12[upper12]
        xu2412 = x_surf[upper]; cpu2412 = Cp_surf[upper]
        ax.plot(xu12[np.argsort(xu12)], cpu12[np.argsort(xu12)],
                'b-', lw=1.5, label='NACA 0012 (symmetric)')
        ax.plot(xu2412[np.argsort(xu2412)], cpu2412[np.argsort(xu2412)],
                'r-', lw=1.5, label='NACA 2412 (cambered)')
        ax.invert_yaxis()
        ax.set_xlabel('$x/c$', fontsize=12)
        ax.set_ylabel('$C_p$ (upper surface)', fontsize=12)
        ax.set_title(r'Surface $C_p$ Comparison ($\alpha=5^\circ$)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        fig.tight_layout()
        p = os.path.join(d, 'naca0012_vs_2412_cp.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {os.path.basename(p)}")

        print(f"\n  NACA 0012: CL={CL12:.4f}, Cp_min={np.min(Cp12):.4f}")
        print(f"  NACA 2412: CL={CL2412:.4f}, Cp_min={np.min(Cp_surf):.4f}")

    print("\n" + "=" * 60)
    print("  NACA 2412 simulation complete.")
    print(f"  Zeta-plane rel. error: {rel_err:.4e}")
    print("=" * 60)

    return rel_err, gamma, Cp_surf

if __name__ == '__main__':
    run_2412_simulation()
