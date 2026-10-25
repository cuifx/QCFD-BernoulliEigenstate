"""
Unbounded Domain Mapping Experiment
====================================
Tests the algebraic coordinate mapping r_tilde = r / (r + L_map) as an
alternative to the spatial window function.

Key comparison:
  A. Cartesian FFT + window   -> ~26% encoding error at t=0
  B. Polar mapped grid, no window (r_max=50) -> much lower encoding error

Evolution runs on the mapped grid only (angular FFT + radial FD) to
demonstrate stability without a window.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
from scipy.fft import fft, ifft
from scipy.linalg import solve_banded

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
U_INF, HBAR, MASS = 1.0, 1.0, 1.0
R_CYL = 0.276

def cylinder_potential(r, theta, Gamma=0.0):
    return U_INF * (r + R_CYL**2 / r) * np.cos(theta) + Gamma/(2*np.pi) * theta

def cylinder_velocity_analytic(r, theta, Gamma=0.0):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    vr = U_INF * (1 - R_CYL**2 / r**2) * cos_t
    vt = -U_INF * (1 + R_CYL**2 / r**2) * sin_t + Gamma/(2*np.pi*r)
    return vr, vt

# ============================================================
# Method A: Cartesian FFT + window (replicates paper's pipeline)
# ============================================================
def method_cartesian_window(N=128, L=2.5, window_margin=0.2, Gamma=0.0):
    """Cartesian FFT-based encoding with spatial window, matching paper."""
    dx = 2*L/N
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    XX, YY = np.meshgrid(x, y)

    # Velocity potential (cylinder in ζ-plane)
    R_grid = np.sqrt(XX**2 + YY**2) + 1e-15
    theta_grid = np.arctan2(YY, XX)
    phi = cylinder_potential(R_grid, theta_grid, Gamma)

    # Encode wavefunction
    psi = np.exp(1j * phi / HBAR)

    # Spatial window
    wx = np.ones(N)
    wy = np.ones(N)
    for i, xi in enumerate(x):
        xn = abs(xi)/L
        if xn > 1 - window_margin:
            wx[i] = 0.5 * (1 + np.cos(np.pi*(xn - 1 + window_margin)/window_margin))
    for j, yj in enumerate(y):
        yn = abs(yj)/L
        if yn > 1 - window_margin:
            wy[j] = 0.5 * (1 + np.cos(np.pi*(yn - 1 + window_margin)/window_margin))
    psi_w = psi * wx[:, np.newaxis] * wy[np.newaxis, :]

    # Madelung decode (central difference, same as paper)
    rho = np.abs(psi_w)**2
    vx = np.zeros((N, N)); vy = np.zeros((N, N))
    for i in range(N):
        im1, ip1 = (i-1)%N, (i+1)%N
        for j in range(N):
            jm1, jp1 = (j-1)%N, (j+1)%N
            dpsi_x = (psi_w[ip1, j] - psi_w[im1, j])/(2*dx)
            dpsi_y = (psi_w[i, jp1] - psi_w[i, jm1])/(2*dx)
            if rho[i, j] > 1e-15:
                vx[i, j] = HBAR/MASS * np.imag(np.conj(psi_w[i, j])*dpsi_x)/rho[i, j]
                vy[i, j] = HBAR/MASS * np.imag(np.conj(psi_w[i, j])*dpsi_y)/rho[i, j]

    # Reference
    vx_ref, vy_ref = np.zeros((N,N)), np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            vr, vt = cylinder_velocity_analytic(R_grid[i,j], theta_grid[i,j], Gamma)
            vx_ref[i,j] = vr*np.cos(theta_grid[i,j]) - vt*np.sin(theta_grid[i,j])
            vy_ref[i,j] = vr*np.sin(theta_grid[i,j]) + vt*np.cos(theta_grid[i,j])

    # Error in near-cylinder region
    interior = (R_grid > R_CYL*1.05) & (R_grid < 1.5) & (np.abs(XX) < 1.0) & (np.abs(YY) < 1.0)
    vmag = np.sqrt(vx**2 + vy**2)
    vmag_ref = np.sqrt(vx_ref**2 + vy_ref**2)
    vel_err = np.sqrt(np.mean((vmag[interior] - vmag_ref[interior])**2)) / np.sqrt(np.mean(vmag_ref[interior]**2))
    rho_dev = np.std(rho[interior]) / np.mean(rho[interior])

    return {'vel_err': vel_err, 'rho_dev': rho_dev, 'vx': vx, 'vy': vy,
            'vx_ref': vx_ref, 'vy_ref': vy_ref, 'XX': XX, 'YY': YY,
            'R_grid': R_grid, 'interior': interior,
            'vmag': vmag, 'vmag_ref': vmag_ref}

# ============================================================
# Method B: Polar mapped grid (no window, r_max → 50)
# ============================================================
class MappedPolarGrid:
    """Polar grid with algebraic radial mapping r_tilde = r/(r+L_map)."""
    def __init__(self, Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0):
        self.Nr, self.Nt = Nr, Nt
        self.L_map = L_map
        self.R_max_phys = R_max_phys

        # Mapping endpoints in r_tilde
        self.r_tilde_min = R_CYL / (R_CYL + L_map)
        self.r_tilde_max = R_max_phys / (R_max_phys + L_map)

        # Uniform grid in computational coordinate
        self.r_tilde = np.linspace(self.r_tilde_min, self.r_tilde_max, Nr)
        self.dr_tilde = self.r_tilde[1] - self.r_tilde[0]

        # Physical radial grid
        self.r = L_map * self.r_tilde / (1 - self.r_tilde)

        # Angular grid
        self.theta = np.linspace(0, 2*np.pi, Nt, endpoint=False)
        self.dtheta = self.theta[1] - self.theta[0]

        # Jacobian and metric
        # dr/d(r_tilde) = L_map/(1-r_tilde)²
        self.J = L_map / (1 - self.r_tilde)**2   # dr/d(r_tilde)
        self.dr_phys = self.J * self.dr_tilde     # physical spacing at each point

        # d²r_tilde/dr² = -2*L_map/(r+L_map)³ = -2*(1-r_tilde)³/L_map²
        # Actually we need: ∂²/∂r² in physical coordinates, expressed in r_tilde
        # ∂/∂r = (dr_tilde/dr) ∂/∂r_tilde = (1/J) ∂/∂r_tilde
        self.metric_d1 = 1.0 / self.J  # (dr_tilde/dr)
        self.metric_d2 = (1.0 / self.J)**2  # (dr_tilde/dr)²
        self.metric_extra = -2 * (1 - self.r_tilde)**3 / (L_map**2)  # d²r_tilde/dr²

        # Precompute full grid for vectorized operations
        self.RR, self.TT = np.meshgrid(self.r, self.theta, indexing='ij')

    def encode(self, Gamma=0.0):
        """Encode wavefunction ψ = exp(iΦ/ħ), NO window."""
        phi = cylinder_potential(self.RR, self.TT, Gamma)
        return np.exp(1j * phi / HBAR)

    def decode(self, psi):
        """Madelung decode in physical polar coordinates."""
        Nr, Nt = psi.shape
        rho = np.abs(psi)**2
        vr = np.zeros_like(psi, dtype=float)
        vt = np.zeros_like(psi, dtype=float)

        for i in range(Nr):
            im1 = max(i-1, 0)
            ip1 = min(i+1, Nr-1)
            dr_i = self.dr_phys[i]
            for j in range(Nt):
                jm1 = (j-1) % Nt
                jp1 = (j+1) % Nt
                dpsi_r = (psi[ip1, j] - psi[im1, j]) / (2*dr_i) if i>0 and i<Nr-1 else 0
                dpsi_t = (psi[i, jp1] - psi[i, jm1]) / (2*self.dtheta)
                if rho[i, j] > 1e-15:
                    vr[i, j] = HBAR/MASS * np.imag(np.conj(psi[i, j])*dpsi_r) / rho[i, j]
                    vt[i, j] = HBAR/MASS * np.imag(np.conj(psi[i, j])*dpsi_t) / (rho[i, j] * self.RR[i, j])
        return vr, vt, rho

    def divergence(self, vr, vt):
        """∇·v in polar coordinates (vectorized)."""
        Nr, Nt = vr.shape
        rv_r = self.RR * vr
        # radial derivative: central difference in physical dr; boundaries 0
        d_rv_r = np.zeros_like(vr)
        dr_i = self.dr_phys[1:-1]
        d_rv_r[1:-1, :] = (rv_r[2:, :] - rv_r[:-2, :]) / (2 * dr_i[:, None])
        # angular derivative (periodic)
        d_vt = (np.roll(vt, -1, axis=1) - np.roll(vt, 1, axis=1)) / (2 * self.dtheta)
        return d_rv_r / self.RR + d_vt / self.RR

    def evolve(self, psi, T, dt=0.0001):
        """Split-step: angular via FFT, radial via finite-difference (Crank-Nicolson-like).
        Uses smaller dt for stability with non-uniform grid."""
        n_steps = max(1, int(T/dt))
        dt_actual = T / n_steps

        k_theta = np.fft.fftfreq(self.Nt, d=1.0) * self.Nt
        k_theta_sq = k_theta**2   # e^{i n theta}: d^2/dtheta^2 -> -n^2 (n = DFT bin index)

        psi_t = psi.copy()

        for step in range(n_steps):
            # Half-step radial
            psi_half = self._radial_step(psi_t, dt_actual/2)

            # Full-step angular via FFT
            psi_kt = fft(psi_half, axis=1)
            psi_kt *= np.exp(-1j * HBAR * dt_actual / (2*MASS) * k_theta_sq[np.newaxis,:] / (self.RR**2 + 1e-12))
            psi_full = np.real(ifft(psi_kt, axis=1)) + 1j*np.imag(ifft(psi_kt, axis=1))

            # Half-step radial
            psi_t = self._radial_step(psi_full, dt_actual/2)

        return psi_t

    def _radial_step(self, psi, dt):
        """Implicit Crank-Nicolson radial kinetic step.

        The former explicit forward-Euler step (psi += i*dt/2*L*psi) is
        unconditionally unstable for the oscillatory Schr\"odinger operator
        at long T (the update multiplies each Laplacian mode by a factor of
        modulus > 1). CN replaces it by

            (I - i*theta*L) psi_new = (I + i*theta*L) psi,  theta = dt/(4m/hbar)

        which is unitarily stable for any dt. Boundary rows (i=0, Nr-1) are
        frozen, as in the explicit version.
        """
        Nr, Nt = psi.shape
        alpha = HBAR / (2 * MASS)
        theta_cn = alpha * dt / 2.0

        # Tridiagonal coefficients of the radial Laplacian on interior rows:
        # L psi = a[i] psi[i-1] + b[i] psi[i] + c[i] psi[i+1]
        a = np.zeros(Nr)
        b = np.zeros(Nr)
        c = np.zeros(Nr)
        for i in range(1, Nr - 1):
            coef = self.metric_d2[i] / self.dr_tilde**2
            lin = (self.metric_extra[i] + self.metric_d1[i] / self.r[i]) / (2 * self.dr_tilde)
            a[i] = coef - lin
            b[i] = -2 * coef
            c[i] = coef + lin

        n = Nr - 2
        idx = np.arange(1, Nr - 1)
        # Banded storage for the (n x n) matrix M = I - i*theta*L
        # (lower bandwidth 1, upper bandwidth 1): ab[u + i - j, j] = M[i, j],
        # i.e. ab[0,j]=M[j-1,j] (upper diag), ab[1,j]=M[j,j], ab[2,j]=M[j+1,j].
        ab = np.zeros((3, n), dtype=complex)
        rhs = np.zeros((n, Nt), dtype=complex)
        for k, i in enumerate(idx):
            ab[1, k] = 1.0 - 1j * theta_cn * b[i]
            if k > 0:
                ab[0, k] = -1j * theta_cn * c[i - 1]
            if k < n - 1:
                ab[2, k] = -1j * theta_cn * a[i + 1]
            rhs[k, :] = psi[i, :] + 1j * theta_cn * (
                a[i] * psi[i - 1, :] + b[i] * psi[i, :] + c[i] * psi[i + 1, :]
            )
            # Frozen boundary rows contribute twice to the interior rhs:
            # once through (I + i*theta*L) psi and once from moving the
            # boundary unknowns of (I - i*theta*L) y to the right-hand side.
            if i == 1:
                rhs[k, :] += 1j * theta_cn * a[i] * psi[i - 1, :]
            if i == Nr - 2:
                rhs[k, :] += 1j * theta_cn * c[i] * psi[i + 1, :]

        sol = solve_banded((1, 1), ab, rhs)
        psi_new = psi.copy()
        psi_new[idx, :] = sol
        return psi_new

    def to_cartesian_bulk(self, vr, vt, Gamma, R_probe=1.5):
        """Compute bulk errors for r < R_probe, convert to Cartesian for viz."""
        vr_ref, vt_ref = cylinder_velocity_analytic(self.RR, self.TT, Gamma)

        interior = (self.RR > R_CYL*1.05) & (self.RR < R_probe)

        vmag = np.sqrt(vr**2 + vt**2)
        vmag_ref = np.sqrt(vr_ref**2 + vt_ref**2)

        vel_err = np.sqrt(np.mean((vmag[interior] - vmag_ref[interior])**2)) / np.sqrt(np.mean(vmag_ref[interior]**2))
        div = self.divergence(vr, vt)
        div_rms = np.sqrt(np.mean(div[interior]**2))

        # To Cartesian
        cos_t, sin_t = np.cos(self.TT), np.sin(self.TT)
        vx = vr*cos_t - vt*sin_t
        vy = vr*sin_t + vt*cos_t
        XX = self.RR * cos_t
        YY = self.RR * sin_t

        return {'vel_err': vel_err, 'div_rms': div_rms,
                'vx': vx, 'vy': vy, 'XX': XX, 'YY': YY,
                'vmag': vmag, 'vmag_ref': vmag_ref,
                'div': div, 'interior': interior}


def plot_profiles(res_cart, grid, res_map_t0, fig_dir):
    """Figure 3: radial velocity profile along the downstream axis (theta=0),
    with a spatial-anchor inset (cylinder + downstream ray) and filled error
    bands so that the near-wall error concentration is visible at a glance."""
    C_RED = '#d63031'
    C_BLUE = '#0984e3'
    C_INK = '#2d3436'
    C_GREY = '#636e72'

    # Cartesian: row nearest y=0 (the x-axis)
    iy0 = int(np.argmin(np.abs(res_cart['YY'][:, 0])))
    x_cart = res_cart['XX'][iy0, :]
    vmag_cart = res_cart['vmag'][iy0, :]
    vmag_ref_cart = res_cart['vmag_ref'][iy0, :]
    mask_c = (np.abs(x_cart) > R_CYL * 1.05) & (np.abs(x_cart) < 2.0)

    # Mapped: ray at theta=0 (downstream)
    idx_t0 = 0
    r_map = grid.r
    vmag_map = res_map_t0['vmag'][:, idx_t0]
    vmag_ref_map = res_map_t0['vmag_ref'][:, idx_t0]
    mask_m = (r_map > R_CYL * 1.05) & (r_map < 2.0)

    rn_c = x_cart[mask_c] / R_CYL
    rn_m = r_map[mask_m] / R_CYL

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # (a) velocity magnitude along the downstream axis
    ax1.plot(rn_c, vmag_cart[mask_c] / U_INF, '-', color=C_RED, lw=2,
             label='Windowed Cartesian')
    ax1.plot(rn_c, vmag_ref_cart[mask_c] / U_INF, '--', color=C_INK, lw=1.5,
             label='Analytic')
    ax1.plot(rn_m, vmag_map[mask_m] / U_INF, '-', color=C_BLUE, lw=2,
             label='Mapped (window-free)')
    ax1.axhline(1.0, color=C_GREY, ls=':', lw=1)
    ax1.axvline(1.0, color=C_GREY, lw=1.0, alpha=0.6)
    ax1.text(1.03, 0.06, 'cylinder\nsurface', fontsize=8, color=C_GREY, va='bottom')
    ax1.set_xlabel(r'$r/R$')
    ax1.set_ylabel(r'$|\mathbf{v}|/U_\infty$')
    ax1.set_xlim(0.85, 2.0)
    ax1.legend(fontsize=8, framealpha=0.95, edgecolor='0.6')
    ax1.grid(True, alpha=0.25, lw=0.6)

    # spatial anchor: cylinder + downstream ray
    iax = ax1.inset_axes([0.60, 0.52, 0.36, 0.42])
    th = np.linspace(0, 2 * np.pi, 120)
    iax.plot(np.cos(th), np.sin(th), color=C_INK, lw=1.2)
    iax.annotate('', xy=(1.7, 0), xytext=(0.05, 0),
                 arrowprops=dict(arrowstyle='-|>', color=C_RED, lw=1.4))
    iax.text(1.12, -0.30, r'$\theta=0$ (downstream)', fontsize=7.5,
             color=C_RED, ha='left')
    iax.plot(0, 0, 'o', ms=4, color=C_BLUE)
    iax.text(0.08, 0.08, 'cylinder', fontsize=7, color=C_GREY)
    iax.set_xlim(-1.7, 1.9)
    iax.set_ylim(-1.35, 1.35)
    iax.set_aspect('equal')
    iax.axis('off')

    # (b) relative error with filled bands (log axis)
    verr_c = np.abs(vmag_cart[mask_c] - vmag_ref_cart[mask_c]) / (np.abs(vmag_ref_cart[mask_c]) + 1e-12) * 100.0
    verr_m = np.abs(vmag_map[mask_m] - vmag_ref_map[mask_m]) / (np.abs(vmag_ref_map[mask_m]) + 1e-12) * 100.0
    ax2.semilogy(rn_c, verr_c, '-', color=C_RED, lw=2, label='Windowed Cartesian')
    ax2.semilogy(rn_m, verr_m, '-', color=C_BLUE, lw=2, label='Mapped (window-free)')
    ax2.fill_between(rn_c, 1e-3, verr_c, color=C_RED, alpha=0.10, lw=0)
    ax2.fill_between(rn_m, 1e-3, verr_m, color=C_BLUE, alpha=0.15, lw=0)
    ax2.axhline(1.0, color=C_GREY, ls=':', lw=1.2)
    ax2.text(1.03, 1.25, '1%', fontsize=8, color=C_GREY)
    ax2.set_xlabel(r'$r/R$')
    ax2.set_ylabel('Relative error [%]')
    ax2.set_xlim(0.85, 2.0)
    ax2.legend(fontsize=8, framealpha=0.95, edgecolor='0.6')
    ax2.grid(True, which='both', alpha=0.25, lw=0.6)

    out = os.path.join(fig_dir, 'unbounded_mapping_profiles.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  -> {out}")


def run_experiment():
    d = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(d, '..', 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    print("="*70)
    print(" Unbounded Mapping vs Window Function")
    print(" ="*70)

    # ================================================================
    # 1. Encoding error comparison at t=0
    # ================================================================
    print("\n--- t=0: Encoding error comparison ---")

    # Method A: Cartesian FFT + window (paper's method)
    t0 = time.time()
    res_cart = method_cartesian_window(N=128, L=2.5, window_margin=0.2)
    print(f"  Cartesian+window:  vel_err={res_cart['vel_err']:.4f}, rho_dev={res_cart['rho_dev']:.4f}  [{time.time()-t0:.1f}s]")

    # Method B: Polar mapped (no window)
    t0 = time.time()
    grid = MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
    psi0 = grid.encode()
    vr_m, vt_m, rho_m = grid.decode(psi0)
    res_map_t0 = grid.to_cartesian_bulk(vr_m, vt_m, 0.0)
    print(f"  Mapped (no window): vel_err={res_map_t0['vel_err']:.4f}, div_rms={res_map_t0['div_rms']:.4e}  [{time.time()-t0:.1f}s]")

    impr_t0 = (1 - res_map_t0['vel_err'] / res_cart['vel_err']) * 100
    print(f"  Improvement: {impr_t0:.1f}%")

    # ================================================================
    # 2. Evolution on mapped grid (no window needed)
    # ================================================================
    print("\n--- Evolution on mapped grid ---")
    T_evolve = 0.01

    # Test stability with various dt
    for dt_test in [0.00005, 0.0001, 0.0005]:
        t0 = time.time()
        try:
            psi_t = grid.evolve(psi0, T=T_evolve, dt=dt_test)
            vr_mt, vt_mt, rho_mt = grid.decode(psi_t)
            res_map_t = grid.to_cartesian_bulk(vr_mt, vt_mt, 0.0)
            stable = res_map_t['vel_err'] < 1.0
            print(f"  dt={dt_test:.5f}: vel_err={res_map_t['vel_err']:.4f}, div_rms={res_map_t['div_rms']:.4e}  [{time.time()-t0:.1f}s] {'STABLE' if stable else 'UNSTABLE'}")
            if stable: break
        except Exception as e:
            print(f"  dt={dt_test:.5f}: FAILED ({e})")

    # ================================================================
    # 3. Comparison with analytic vector field
    # ================================================================
    print("\n--- Vector field diagnostics ---")

    # Bulk velocity error vectors
    vx_m, vy_m = res_map_t0['vx'], res_map_t0['vy']
    XX_m, YY_m = res_map_t0['XX'], res_map_t0['YY']
    vmag_m = res_map_t0['vmag']
    vmag_m_ref = res_map_t0['vmag_ref']
    div_m = res_map_t0['div']

    # ================================================================
    # 4. Scan L_map parameter
    # ================================================================
    print("\n--- L_map parameter scan ---")
    L_maps = [1.0, 2.5, 5.0, 10.0, 25.0]
    vel_errs_lmap = []
    div_errs_lmap = []
    for Lm in L_maps:
        g = MappedPolarGrid(Nr=128, Nt=256, L_map=Lm, R_max_phys=50.0)
        p = g.encode()
        vr_, vt_, _ = g.decode(p)
        res_ = g.to_cartesian_bulk(vr_, vt_, 0.0)
        vel_errs_lmap.append(res_['vel_err'])
        div_errs_lmap.append(res_['div_rms'])
        print(f"  L_map={Lm:4.1f}: vel_err={res_['vel_err']:.4f}, div_rms={res_['div_rms']:.4e}")

    # ================================================================
    # 5. Scan r_max (domain extent) with fixed L_map
    # ================================================================
    print("\n--- R_max scan (mapped, L_map=2.5) ---")
    R_maxs = [10.0, 25.0, 50.0, 100.0]
    vel_errs_rmax = []
    div_errs_rmax = []
    for Rm in R_maxs:
        g = MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=Rm)
        p = g.encode()
        vr_, vt_, _ = g.decode(p)
        res_ = g.to_cartesian_bulk(vr_, vt_, 0.0)
        vel_errs_rmax.append(res_['vel_err'])
        div_errs_rmax.append(res_['div_rms'])
        print(f"  R_max={Rm:5.0f}: vel_err={res_['vel_err']:.4f}, div_rms={res_['div_rms']:.4e}")

    # ================================================================
    # Visualization
    # ================================================================
    print("\n--- Generating figures ---")

    # Figure 1: Domain comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    r_plot = np.linspace(R_CYL, 2.5, 500)
    margin = 0.2
    w_plot = np.ones_like(r_plot)
    mask = r_plot > 2.5*(1-margin)
    w_plot[mask] = 0.5*(1+np.cos(np.pi*(r_plot[mask]-2.5*(1-margin))/(2.5*margin)))
    ax1.plot(r_plot, w_plot, 'b-', lw=2)
    ax1.axvline(x=R_CYL, color='gray', ls='--', alpha=0.5)
    ax1.axvline(x=2.5, color='red', ls=':', alpha=0.7)
    ax1.fill_between([2.0, 2.5], 0, 1, alpha=0.15, color='red')
    ax1.set_xlabel('r'); ax1.set_ylabel('W(r)')
    ax1.set_title('(a) Windowed: finite domain + cosine taper', fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.set_xlim(0, 3.5)

    r_t_plot = np.linspace(R_CYL/(R_CYL+2.5), 50/(50+2.5), 500)
    r_p_plot = 2.5 * r_t_plot/(1-r_t_plot)
    ax2.plot((r_t_plot-r_t_plot[0])/(r_t_plot[-1]-r_t_plot[0]), r_p_plot, 'b-', lw=2)
    ax2.set_yscale('log')
    ax2.axhline(y=R_CYL, color='gray', ls='--', alpha=0.5)
    ax2.axhline(y=50, color='green', ls=':', alpha=0.7)
    ax2.axhline(y=2.5, color='red', ls=':', alpha=0.5)
    ax2.set_xlabel('Normalized computational coordinate')
    ax2.set_ylabel('Physical r')
    ax2.set_title('(b) Mapped: $\\tilde{r}=r/(r+L_m)$, domain 0.28--50', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'unbounded_mapping_domain.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> unbounded_mapping_domain.png")

    # Figure 2: t=0 encoding error, 3-panel comparison:
    # (a) Cartesian+window |V|, (b) Mapped no-window |V|,
    # (c) signed error Delta|V| of both schemes in a single split-field
    # panel on a shared color scale (the mapped error is ~20x smaller, so
    # the windowed half saturates while its direction stays visible; the
    # RMS magnitudes are annotated inside the panel).
    fig = plt.figure(figsize=(15.0, 5.4))
    ax1 = fig.add_axes([0.095, 0.10, 0.24, 0.78])
    ax2 = fig.add_axes([0.355, 0.10, 0.24, 0.78])
    ax3 = fig.add_axes([0.615, 0.10, 0.24, 0.78])
    cax = fig.add_axes([0.885, 0.10, 0.02, 0.78])

    def plot_field(ax, XX, YY, Z, title, cmap='RdBu_r', vmin=None, vmax=None):
        im = ax.pcolormesh(XX, YY, Z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        circle = plt.Circle((0, 0), R_CYL, color='gray', fill=True, alpha=0.8)
        ax.add_patch(circle)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        return im

    # Cartesian: |V|
    verr_cart = np.abs(res_cart['vmag'] - res_cart['vmag_ref'])
    im1 = plot_field(ax1, res_cart['XX'], res_cart['YY'], res_cart['vmag'],
                     '(a) Cartesian+window: |V|', vmin=0.4, vmax=1.6)

    # Mapped: |V|
    verr_map = np.abs(vmag_m - vmag_m_ref)
    im2 = plot_field(ax2, XX_m, YY_m, vmag_m,
                     '(b) Mapped no-window: |V|', vmin=0.4, vmax=1.6)

    # (c) signed error of both schemes merged into one split-field panel:
    # windowed Cartesian in the left half-plane, mapped in the right half,
    # sharing a single color scale.
    dv_cart = res_cart['vmag'] - res_cart['vmag_ref']
    dv_map = vmag_m - vmag_m_ref
    dv_lim = 0.10
    dv_cart_half = np.where(res_cart['XX'] >= 0.0, np.nan, dv_cart)
    dv_map_half = np.where(XX_m < 0.0, np.nan, dv_map)
    im3 = ax3.pcolormesh(res_cart['XX'], res_cart['YY'], dv_cart_half, shading='auto',
                         cmap='RdBu_r', vmin=-dv_lim, vmax=dv_lim)
    ax3.pcolormesh(XX_m, YY_m, dv_map_half, shading='auto',
                   cmap='RdBu_r', vmin=-dv_lim, vmax=dv_lim)
    ax3.set_aspect('equal')
    ax3.set_xlim(-2.5, 2.5); ax3.set_ylim(-2.5, 2.5)
    circle = plt.Circle((0, 0), R_CYL, color='gray', fill=True, alpha=0.8)
    ax3.add_patch(circle)
    ax3.axvline(0.0, color='k', lw=0.8, ls=':')
    ax3.set_title('(c) Signed error $\\Delta|V|$ (shared scale)', fontsize=10)
    ax3.text(-1.6, -2.12, 'windowed', ha='center', fontsize=9, color='red',
             fontweight='bold')
    ax3.text(-1.6, -2.42, f'RMS {res_cart["vel_err"]:.1e}', ha='center',
             fontsize=8, color='red')
    ax3.text(1.6, -2.12, 'mapped', ha='center', fontsize=9, color='green',
             fontweight='bold')
    ax3.text(1.6, -2.42, f'RMS {res_map_t0["vel_err"]:.1e}', ha='center',
             fontsize=8, color='green')

    cb = fig.colorbar(im3, cax=cax)
    cb.set_label(r'$\Delta|V| = |V|-|V_\mathrm{ref}|$')


    fig.savefig(os.path.join(fig_dir, 'unbounded_mapping_t0.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> unbounded_mapping_t0.png")

    # Figure 3: Radial velocity profile along the downstream x-axis (theta=0)
    plot_profiles(res_cart, grid, res_map_t0, fig_dir)

    # Figure 4: Parameter scans
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(L_maps, vel_errs_lmap, 'bo-', lw=2)
    ax1.set_xlabel('L_map'); ax1.set_ylabel('Relative velocity error')
    ax1.set_title('Sensitivity to L_map (R_max=50)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=res_cart['vel_err'], color='red', ls='--', alpha=0.7, label=f'Cartesian+window ({res_cart["vel_err"]:.3f})')
    ax1.legend(fontsize=8)

    ax2.plot(R_maxs, vel_errs_rmax, 'go-', lw=2)
    ax2.set_xlabel('R_max (physical)'); ax2.set_ylabel('Relative velocity error')
    ax2.set_title('Sensitivity to R_max (L_map=2.5)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=res_cart['vel_err'], color='red', ls='--', alpha=0.7, label=f'Cartesian+window ({res_cart["vel_err"]:.3f})')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'unbounded_mapping_params.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> unbounded_mapping_params.png")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"  t=0 encoding error:")
    print(f"    Cartesian+window:  {res_cart['vel_err']:.4f}")
    print(f"    Mapped no-window:  {res_map_t0['vel_err']:.4f}")
    print(f"    Improvement:       {impr_t0:.1f}%")
    if T_evolve > 0:
        print(f"  T={T_evolve} mapped evolution:")
        print(f"    vel_err:           {res_map_t['vel_err']:.4f}")
        print(f"    div_rms:           {res_map_t['div_rms']:.4e}")

    return res_cart, res_map_t0


if __name__ == '__main__':
    run_experiment()
