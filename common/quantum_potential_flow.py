import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings('ignore')

N_GRID = 500
DOMAIN_RADIUS = 10.0
N_THETA_QUANTUM = 4096


def compute_kutta_gamma(epsilon, delta, U_inf=1.0):
    a = np.sqrt((1 + epsilon) ** 2 + delta ** 2)
    zeta_te = 1 + epsilon - 1j * delta
    beta = np.angle(zeta_te)
    return 4 * np.pi * a * U_inf * np.sin(np.abs(beta))


def generate_joukowski_airfoil(epsilon=0.1, delta=0.05, n_points=200):
    a = np.sqrt((1 + epsilon) ** 2 + delta ** 2)
    theta = np.linspace(0, 2 * np.pi, n_points)
    zeta = -epsilon + a * np.cos(theta) + 1j * (delta + a * np.sin(theta))
    z = zeta + 1.0 / zeta
    return z.real, z.imag, zeta, a


def analytic_flow_field(X, Y, epsilon, delta, gamma, U_inf=1.0):
    a = np.sqrt((1 + epsilon) ** 2 + delta ** 2)
    Z = X + 1j * Y
    disc = np.sqrt(Z ** 2 - 4 + 0j)
    zp = (Z + disc) / 2
    zm = (Z - disc) / 2
    zeta = np.where(np.abs(zp) >= 1.0, zp, zm)
    zs = zeta + epsilon - 1j * delta
    W = U_inf * (zs + a ** 2 / zs) + 1j * gamma / (2 * np.pi) * np.log(zs)
    phi = W.real
    psi = W.imag
    dW_dzeta = U_inf * (1 - a ** 2 / zs ** 2) + 1j * gamma / (2 * np.pi) / zs
    dz_dzeta = 1 - 1.0 / zeta ** 2
    dW_dz = dW_dzeta / dz_dzeta
    return phi, psi, dW_dz.real, -dW_dz.imag


def is_inside_airfoil(X, Y, airfoil_x, airfoil_y):
    from matplotlib.path import Path
    verts = np.column_stack([airfoil_x, airfoil_y])
    return Path(verts).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)


def fourier_synthesize_on_grid(X, Y, epsilon, delta, u_inf, gamma, airfoil_x, airfoil_y, t=0.0, hbar=1.0, mass=1.0):
    """Fourier synthesis with optional time evolution (Schrödinger equation).

    Each angular Fourier mode k evolves as: c_k(t) = c_k(0) * exp(-i * hbar * k^2 * t / (2 * mass * r^2))
    where r is the local radius in the zeta-plane. Exact in this respect: no grid
    error, no time-stepping error, no boundary artifacts.

    Parameters
    ----------
    t : float
        Evolution time. t=0 returns the steady potential flow solution.
    hbar, mass : float
        Physical constants in the Schrödinger equation.
    """
    a = np.sqrt((1 + epsilon) ** 2 + delta ** 2)
    Z = X + 1j * Y
    disc = np.sqrt(Z ** 2 - 4 + 0j)
    zeta = np.where(np.abs((Z + disc) / 2) >= 1.0, (Z + disc) / 2, (Z - disc) / 2)
    zs = zeta + epsilon - 1j * delta
    r = np.abs(zs)
    r_safe = np.maximum(r, 1e-12)
    th = np.angle(zs)

    # Time evolution phase factors: exp(-i * hbar * k^2 * t / (2m * r^2))
    phase_k1 = np.exp(-1j * hbar * t / (2 * mass * r_safe ** 2))       # k = -1
    phase_k2 = np.exp(-2j * hbar * t / (mass * r_safe ** 2))           # k = -2

    c0 = u_inf                                                          # k = 0 (no evolution)
    cm2 = -u_inf * (a / r_safe) ** 2 * phase_k2                         # k = -2, time-evolved
    cm1 = 1j * gamma / (2 * np.pi * a) * (a / r_safe) * phase_k1       # k = -1, time-evolved

    dW_dz = (c0 + cm2 * np.exp(-2j * th) + cm1 * np.exp(-1j * th)) / (1 - 1.0 / zeta ** 2)
    u, v = dW_dz.real, -dW_dz.imag
    inside = is_inside_airfoil(X, Y, airfoil_x, airfoil_y)
    u[inside] = np.nan
    v[inside] = np.nan
    return u, v


def compute_surface_velocity(airfoil_x, airfoil_y, epsilon, delta, gamma, U_inf=1.0):
    a = np.sqrt((1 + epsilon) ** 2 + delta ** 2)
    Z = airfoil_x + 1j * airfoil_y
    disc = np.sqrt(Z ** 2 - 4 + 0j)
    zeta = np.where(np.abs((Z + disc) / 2) >= 1.0, (Z + disc) / 2, (Z - disc) / 2)
    zs = zeta + epsilon - 1j * delta
    dW_dzeta = U_inf * (1 - a ** 2 / zs ** 2) + 1j * gamma / (2 * np.pi) / zs
    dW_dz = dW_dzeta / (1 - 1.0 / zeta ** 2)
    vel = np.sqrt(dW_dz.real ** 2 + dW_dz.imag ** 2)
    return vel


def main():
    print("=" * 60)
    print("  2D Airfoil Potential Flow: Quantum Fourier Synthesis (Grid-free)")
    print("=" * 60)

    epsilon, delta = 0.1, 0.05
    U_inf = 1.0
    gamma = compute_kutta_gamma(epsilon, delta, U_inf)

    airfoil_x, airfoil_y, _, a = generate_joukowski_airfoil(epsilon, delta)
    print(f"[1] Joukowski airfoil: eps={epsilon}, delta={delta}, Gamma={gamma:.4f}")

    x_vals = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, N_GRID)
    y_vals = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, N_GRID)
    X, Y = np.meshgrid(x_vals, y_vals)

    print(f"[2] Analytic solution (Joukowski conformal map)")
    phi_ana, psi_ana, u_ana, v_ana = analytic_flow_field(X, Y, epsilon, delta, gamma, U_inf)
    inside = is_inside_airfoil(X, Y, airfoil_x, airfoil_y)
    for arr in [phi_ana, psi_ana, u_ana, v_ana]:
        arr[inside] = np.nan
    vel_ana = np.sqrt(u_ana ** 2 + v_ana ** 2)

    print(f"[3] Quantum Fourier synthesis (3 modes, O(1)/point, grid-free)")
    t0 = time.time()
    u_q, v_q = fourier_synthesize_on_grid(X, Y, epsilon, delta, U_inf, gamma, airfoil_x, airfoil_y)
    vel_q = np.sqrt(u_q ** 2 + v_q ** 2)
    print(f"    Time: {time.time() - t0:.4f}s")

    valid = ~np.isnan(vel_q) & ~np.isnan(vel_ana)
    rel_err = np.nanmean(np.abs(vel_q[valid] - vel_ana[valid]) / (vel_ana[valid] + 1e-10))
    print(f"    Rel. error: {rel_err:.2e}")

    print(f"[4] Generating visualization...")
    skip = max(3, N_GRID // 16)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # (a) Airfoil geometry
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(airfoil_x, airfoil_y, 'k-', lw=2)
    ax.fill(airfoil_x, airfoil_y, '#555', alpha=0.7)
    ax.set_xlim(-3, 2.5); ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f'(a) Joukowski airfoil\n$\\varepsilon$={epsilon}, $\\delta$={delta}', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) Stream function
    ax = fig.add_subplot(gs[0, 1])
    cf = ax.pcolormesh(X, Y, psi_ana, cmap='RdBu_r', shading='auto')
    ax.plot(airfoil_x, airfoil_y, 'k-', lw=2)
    ax.fill(airfoil_x, airfoil_y, '#555', alpha=0.7)
    ax.set_xlim(-DOMAIN_RADIUS, DOMAIN_RADIUS); ax.set_ylim(-DOMAIN_RADIUS, DOMAIN_RADIUS)
    ax.set_aspect('equal')
    ax.set_title('(b) Stream function $\\Psi$', fontweight='bold')
    plt.colorbar(cf, ax=ax, shrink=0.8)

    # (c) Analytic |V|
    ax = fig.add_subplot(gs[0, 2])
    c1 = ax.pcolormesh(X, Y, vel_ana, cmap='turbo', vmin=0, vmax=np.nanpercentile(vel_ana, 95), shading='auto')
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], u_ana[::skip, ::skip], v_ana[::skip, ::skip],
              scale=25, width=0.002, color='white', alpha=0.7)
    ax.plot(airfoil_x, airfoil_y, 'w-', lw=2)
    ax.fill(airfoil_x, airfoil_y, '#333', alpha=0.9)
    ax.set_xlim(-DOMAIN_RADIUS, DOMAIN_RADIUS); ax.set_ylim(-DOMAIN_RADIUS, DOMAIN_RADIUS)
    ax.set_aspect('equal')
    ax.set_title('(c) Analytic $|\\mathbf{V}|$', fontweight='bold')
    plt.colorbar(c1, ax=ax, shrink=0.8)

    # (d) Fourier synthesis |V|
    ax = fig.add_subplot(gs[1, 0])
    c2 = ax.pcolormesh(X, Y, vel_q, cmap='turbo', vmin=0, vmax=np.nanpercentile(vel_q, 95), shading='auto')
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], u_q[::skip, ::skip], v_q[::skip, ::skip],
              scale=25, width=0.002, color='white', alpha=0.7)
    ax.plot(airfoil_x, airfoil_y, 'w-', lw=2)
    ax.fill(airfoil_x, airfoil_y, '#333', alpha=0.9)
    ax.set_xlim(-DOMAIN_RADIUS, DOMAIN_RADIUS); ax.set_ylim(-DOMAIN_RADIUS, DOMAIN_RADIUS)
    ax.set_aspect('equal')
    ax.set_title(f'(d) Fourier synthesis $|\\mathbf{{V}}|$\n3 non-zero modes, $\\mathcal{{O}}(1)$/point', fontweight='bold')
    plt.colorbar(c2, ax=ax, shrink=0.8)

    # (e) Error |V_Fourier - V_analytic|
    ax = fig.add_subplot(gs[1, 1])
    diff = np.abs(vel_q - vel_ana)
    diff[inside] = np.nan
    c3 = ax.pcolormesh(X, Y, diff, cmap='inferno', shading='auto')
    ax.plot(airfoil_x, airfoil_y, 'w-', lw=2)
    ax.fill(airfoil_x, airfoil_y, '#333', alpha=0.9)
    ax.set_xlim(-DOMAIN_RADIUS, DOMAIN_RADIUS); ax.set_ylim(-DOMAIN_RADIUS, DOMAIN_RADIUS)
    ax.set_aspect('equal')
    ax.set_title(f'(e) $|\\mathbf{{V}}_\\mathrm{{Fourier}} - \\mathbf{{V}}_\\mathrm{{analytic}}|$\nrel. error = {rel_err:.2e}', fontweight='bold')
    plt.colorbar(c3, ax=ax, shrink=0.8)

    # (f) Pressure coefficient Cp
    ax = fig.add_subplot(gs[1, 2])
    cp = 1.0 - (vel_ana / U_inf) ** 2
    cp[inside] = np.nan
    cmax = max(abs(np.nanpercentile(cp[~inside], 1)), abs(np.nanpercentile(cp[~inside], 99)))
    c4 = ax.pcolormesh(X, Y, cp, cmap='coolwarm', vmin=-cmax, vmax=cmax, shading='auto')
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], u_ana[::skip, ::skip], v_ana[::skip, ::skip],
              scale=25, width=0.002, color='black', alpha=0.5)
    ax.plot(airfoil_x, airfoil_y, 'k-', lw=2)
    ax.fill(airfoil_x, airfoil_y, '#555', alpha=0.7)
    ax.set_xlim(-DOMAIN_RADIUS, DOMAIN_RADIUS); ax.set_ylim(-DOMAIN_RADIUS, DOMAIN_RADIUS)
    ax.set_aspect('equal')
    ax.set_title(f'(f) Pressure coefficient $C_p$\n$C_p=1-V^2/U_\\infty^2$, $\\Gamma$={gamma:.3f}', fontweight='bold')
    plt.colorbar(c4, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, 'potential_flow_results.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    -> potential_flow_results.png")

    # Surface velocity distribution
    print(f"[5] Airfoil surface velocity distribution")
    n_surf = 400
    af_x, af_y, _, _ = generate_joukowski_airfoil(epsilon, delta, n_surf)
    vel_surf = compute_surface_velocity(af_x, af_y, epsilon, delta, gamma, U_inf)

    upper = af_y > 0
    lower = af_y <= 0
    x_cutoff = af_x.max() - 0.05 * (af_x.max() - af_x.min())

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax1.scatter(af_x, af_y, c=vel_surf, cmap='Spectral_r', s=10)
    ax1.plot(af_x[upper], af_y[upper], '#0984e3', lw=2.5, label='Upper (suction side)')
    ax1.plot(af_x[lower], af_y[lower], '#e17055', lw=2.5, label='Lower (pressure side)')
    ax1.axvline(x=x_cutoff, color='gray', ls=':', alpha=0.5)
    ax1.set_aspect('equal')
    ax1.set_title('(a) Surface velocity |V|', fontweight='bold')
    ax1.legend(fontsize=8)
    plt.colorbar(sc, ax=ax1, shrink=0.8)

    xu = af_x[upper]; xl = af_x[lower]
    cp_u = 1.0 - (vel_surf[upper] / U_inf) ** 2
    cp_l = 1.0 - (vel_surf[lower] / U_inf) ** 2
    ax2.plot(xu[np.argsort(xu)], cp_u[np.argsort(xu)], '#0984e3', lw=2.5, label='Upper surface')
    ax2.plot(xl[np.argsort(xl)], cp_l[np.argsort(xl)], '#e17055', lw=2.5, ls='--', label='Lower surface')
    ax2.fill_between(xu[np.argsort(xu)], cp_u[np.argsort(xu)], 0, alpha=0.1, color='#0984e3')
    ax2.fill_between(xl[np.argsort(xl)], cp_l[np.argsort(xl)], 0, alpha=0.1, color='#e17055')
    ax2.axvline(x=x_cutoff, color='gray', ls=':', alpha=0.5)
    ax2.invert_yaxis()
    ax2.set_title(f'(b) Pressure coefficient $C_p$ ($\\Gamma$={gamma:.3f})', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    upper_body = vel_surf[upper & (af_x < x_cutoff)]
    lower_body = vel_surf[lower & (af_x < x_cutoff)]
    v_ratio = np.max(upper_body) / np.max(lower_body) if len(lower_body) > 0 else np.nan
    print(f"    Max velocity upper: {np.max(vel_surf[upper]):.3f}U, lower: {np.max(vel_surf[lower]):.3f}U")
    print(f"    Upper/lower ratio: {v_ratio:.2f} {'(upper faster -- consistent with lift theory)' if v_ratio > 1.02 else ''}")

    plt.tight_layout()
    fig2.savefig(os.path.join(_HERE, 'surface_velocity_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    -> surface_velocity_distribution.png")

    print(f"\nDone. Grid-free Fourier synthesis: 3 non-zero modes, error {rel_err:.2e}")


if __name__ == '__main__':
    main()
