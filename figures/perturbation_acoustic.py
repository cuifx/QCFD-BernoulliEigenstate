"""
Acoustic-perturbation propagation on the stationary Bernoulli base flow
========================================================================
Engineering demonstration of Sec. 4.3 (base-flow decomposition for linear
perturbations): a small acoustic blob on a unit cylinder (R=1, U_inf=1)
propagates cleanly when the background is the exact H_B zero-mode eigenstate,
while the free-particle branch disperses the background and drowns the signal.

Setup (window-free algebraic mapped grid, Nr=Nt=256, L_map=2.5, r_max=50):
    Phi_0   = U_inf (r + R^2/r) cos(theta)          background potential
    psi_0   = exp(i Phi_0)                           H_B zero mode
    dPhi    = eps exp(-((x-x0)^2 + y^2)/(2 sigma^2)) cos(k x)
    eps=1e-2, (x0,y0)=(2.5,0), sigma=0.4, k=2 pi/1.2
    psi_in  = psi_0 exp(i dPhi)

Branches:
    A (free):  U = exp(-i H_free t)        (background disperses)
    B (H_B):   U = Strang split [V/2][H_free][V/2], V = -|v_ref|^2/2
               (background frozen to machine precision)

Diagnostics:
    SNR(t) = 10 log10( ||delta_psi_B(t)||^2 / ||psi_bg(t) - psi_0||^2 )
             (perturbation energy vs. background pollution; the numerator is
             the perturbation evolved by the correct linearized propagator,
             delta_psi_B(t) = psi_full_B(t) - psi_0)
    probe  = Re(delta_psi) at (4,0) vs. t
    p'(x,y)= -( |v|^2 - |v_bg|^2 )/2 at T=1, viewport r < 5

Outputs: fig/perturbation/Figure_A..D.pdf + snr_data.npz,
         fig/perturbation_fields.pdf, fig/perturbation_snr.pdf
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft, ifft

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

import unbounded_mapping as ub
ub.R_CYL = 1.0                      # unit cylinder (Sec. 4.3 demo)
from mapped_bernoulli import (
    RadialSpectralPropagator, bernoulli_potential, eigenstate_residual)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- parameters ----------------
U_INF, HBAR, MASS = 1.0, 1.0, 1.0
R_CYL = 1.0
L_MAP, R_MAX = 2.5, 50.0
NR = int(os.environ.get('PERT_NR', '256'))
NT = int(os.environ.get('PERT_NT', '256'))
T_TOTAL = float(os.environ.get('PERT_T', '1.0'))
# NOTE: the paper's perturbation figures (Sec. 4.3) were generated with
# dt = 4e-4 (matching the SNR numbers quoted in the text: 53-51 dB,
# Ebg_B <= 5.5e-7).  The former default 0.001 does NOT reproduce the
# published figures (it gives only 38-24 dB), so the default is 4e-4.
DT = float(os.environ.get('PERT_DT', '4e-4'))
EPS, X0, Y0, SIGMA = 0.01, 2.5, 0.0, 1.2
K_WAVE = 2*np.pi/1.2
PROBE = (4.0, 0.0)                  # far-field probe
VIEW_R = 5.0                        # viewport for p' maps
SAMP = 4                            # sample every SAMP steps
SEED = 0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'fig', 'perturbation')
FIG_DIR = os.path.join(HERE, '..', 'fig')

def build_grid():
    return ub.MappedPolarGrid(NR, NT, L_MAP, R_MAX)

def make_perturbation(grid):
    """delta_Phi on the grid; returns dphi and the perturbed initial state."""
    X = grid.RR*np.cos(grid.TT)
    Y = grid.RR*np.sin(grid.TT)
    g = np.exp(-((X - X0)**2 + (Y - Y0)**2)/(2*SIGMA**2))
    dphi = EPS*g*np.cos(K_WAVE*X)
    return dphi

def evolve_track(grid, rsp, psi, V, T, dt, use_V, probe_ij, samp=SAMP):
    """Split-step evolution (angular FFT + exact spectral radial step),
    sampling probe value every `samp` steps. Returns (t_pts, probe_pts)."""
    n_steps = max(1, int(T/dt))
    dt_a = T/n_steps
    k_theta = np.fft.fftfreq(grid.Nt, d=1.0)*grid.Nt
    k_theta_sq = k_theta**2
    expV = np.exp(-1j*V*dt_a/2) if use_V else None
    psi_t = psi.copy()
    ts, pr = [], []
    for s in range(n_steps):
        if use_V:
            psi_t = psi_t*expV
        psi_t = rsp.apply(psi_t, dt_a/2)
        psi_kt = fft(psi_t, axis=1)
        psi_kt *= np.exp(-1j*HBAR*dt_a/(2*MASS)
                         * k_theta_sq[np.newaxis, :]/(grid.RR**2 + 1e-12))
        psi_t = ifft(psi_kt, axis=1)
        psi_t = rsp.apply(psi_t, dt_a/2)
        if use_V:
            psi_t = psi_t*expV
        if s % samp == 0 or s == n_steps-1:
            ts.append((s+1)*dt_a)
            pr.append(psi_t[probe_ij])
    return np.array(ts), np.array(pr)

def decode_vec(grid, psi):
    """Vectorized Madelung decode on the mapped polar grid (velocity)."""
    rho = np.abs(psi)**2
    dr_phys = grid.dr_phys
    dpsi_r = np.zeros_like(psi)
    dpsi_r[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dr_phys[1:-1, None])
    dpsi_t = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2*grid.dtheta)
    vr = np.zeros_like(psi, dtype=float)
    vt = np.zeros_like(psi, dtype=float)
    m = rho > 1e-20
    vr[m] = HBAR/MASS*np.imag(np.conj(psi[m])*dpsi_r[m])/rho[m]
    vt[m] = HBAR/MASS*np.imag(np.conj(psi[m])*dpsi_t[m])/(rho[m]*grid.RR[m])
    return vr, vt, rho

def run():
    t0 = time.time()
    grid = build_grid()
    rsp = RadialSpectralPropagator(grid)
    psi0 = grid.encode()
    dphi = make_perturbation(grid)
    psi_in = psi0*np.exp(1j*dphi)
    # Discrete exact zero-mode potential (Sec. 6.1.2): V such that
    # (H_free + V) psi0 = 0 holds pointwise on the grid, freezing the
    # background to machine precision (the analytic Bernoulli potential is
    # recovered up to discretization, Im(V) ~ dr^2).
    Hf, _ = eigenstate_residual(grid, psi0, np.zeros_like(psi0))
    V = -(Hf/psi0)

    # probe index (nearest grid point to (4,0))
    i_pr = int(np.argmin(np.abs(grid.r - PROBE[0])))
    j_pr = int(np.argmin(np.abs(grid.theta - 0.0)))
    probe_ij = (i_pr, j_pr)
    print(f"probe r={grid.r[i_pr]:.4f} theta={grid.theta[j_pr]:.4f}")

    # ---------------- evolutions ----------------
    print("evolving branch B (H_B): background + perturbation ...")
    tB, prB_full = evolve_track(grid, rsp, psi_in, V, T_TOTAL, DT, True, probe_ij)
    print("evolving branch A (free): background + perturbation ...")
    tA, prA_full = evolve_track(grid, rsp, psi_in, V, T_TOTAL, DT, False, probe_ij)

    # ---------------- diagnostics ----------------
    # probe traces: total state minus the INITIAL background at the probe
    # (branch B: background frozen -> pure perturbation; branch A: the
    #  dispersing background phase contaminates the trace)
    probe_ref = psi0[probe_ij]
    dB_full = prB_full - probe_ref
    dA_full = prA_full - probe_ref

    def norm_series(psi_start, V, use_V):
        n_steps = max(1, int(T_TOTAL/DT))
        dt_a = T_TOTAL/n_steps
        k_theta = np.fft.fftfreq(grid.Nt, d=1.0)*grid.Nt
        k_theta_sq = k_theta**2
        expV = np.exp(-1j*V*dt_a/2) if use_V else None
        psi_t = psi_start.copy()
        ts, ns = [], []
        for s in range(n_steps):
            if use_V:
                psi_t = psi_t*expV
            psi_t = rsp.apply(psi_t, dt_a/2)
            psi_kt = fft(psi_t, axis=1)
            psi_kt *= np.exp(-1j*HBAR*dt_a/(2*MASS)
                             * k_theta_sq[np.newaxis, :]/(grid.RR**2 + 1e-12))
            psi_t = ifft(psi_kt, axis=1)
            psi_t = rsp.apply(psi_t, dt_a/2)
            if use_V:
                psi_t = psi_t*expV
            if s % SAMP == 0 or s == n_steps-1:
                ts.append((s+1)*dt_a)
                ns.append(np.linalg.norm(psi_t - psi0)**2)
        return np.array(ts), np.array(ns)

    print("computing background-pollution norms ...")
    tB2, Ebg_B = norm_series(psi0, V, True)
    tA2, Ebg_A = norm_series(psi0, V, False)

    # perturbation energy (correct linearized propagator) per time
    def full_delta_norm(psi_start, V, use_V):
        """||psi_full(t) - psi0||^2 under branch use_V (B: frozen bg)."""
        n_steps = max(1, int(T_TOTAL/DT))
        dt_a = T_TOTAL/n_steps
        k_theta = np.fft.fftfreq(grid.Nt, d=1.0)*grid.Nt
        k_theta_sq = k_theta**2
        expV = np.exp(-1j*V*dt_a/2) if use_V else None
        psi_t = psi_start.copy()
        ts, ns = [], []
        for s in range(n_steps):
            if use_V:
                psi_t = psi_t*expV
            psi_t = rsp.apply(psi_t, dt_a/2)
            psi_kt = fft(psi_t, axis=1)
            psi_kt *= np.exp(-1j*HBAR*dt_a/(2*MASS)
                             * k_theta_sq[np.newaxis, :]/(grid.RR**2 + 1e-12))
            psi_t = ifft(psi_kt, axis=1)
            psi_t = rsp.apply(psi_t, dt_a/2)
            if use_V:
                psi_t = psi_t*expV
            if s % SAMP == 0 or s == n_steps-1:
                ts.append((s+1)*dt_a)
                ns.append(np.linalg.norm(psi_t - psi0)**2)
        return np.array(ts), np.array(ns)
    print("computing perturbation-energy series (branch B) ...")
    tP, E_pertB = full_delta_norm(psi_in, V, True)

    # perturbation energy under A: ||psi_full_A(t) - psi_bg_A(t)||^2,
    # both evolved by the free propagator on the fly
    print("computing perturbation-energy series (branch A) ...")
    n_steps = max(1, int(T_TOTAL/DT))
    dt_a = T_TOTAL/n_steps
    k_theta = np.fft.fftfreq(grid.Nt, d=1.0)*grid.Nt
    k_theta_sq = k_theta**2
    psf = psi_in.copy(); psb = psi0.copy()
    tP2, E_pertA = [], []
    for s in range(n_steps):
        psf = rsp.apply(psf, dt_a/2)
        psb = rsp.apply(psb, dt_a/2)
        f1 = fft(psf, axis=1); b1 = fft(psb, axis=1)
        ph = np.exp(-1j*HBAR*dt_a/(2*MASS)
                    * k_theta_sq[np.newaxis, :]/(grid.RR**2 + 1e-12))
        f1 *= ph; b1 *= ph
        psf = ifft(f1, axis=1); psb = ifft(b1, axis=1)
        psf = rsp.apply(psf, dt_a/2)
        psb = rsp.apply(psb, dt_a/2)
        if s % SAMP == 0 or s == n_steps-1:
            tP2.append((s+1)*dt_a)
            E_pertA.append(np.linalg.norm(psf - psb)**2)
    E_pertA = np.array(E_pertA); tP2 = np.array(tP2)

    # SNR: numerator = perturbation energy of the correct branch (B)
    # denominator = background pollution of each branch
    SNR_B = 10*np.log10(E_pertB/np.maximum(Ebg_B, 1e-30))
    SNR_A = 10*np.log10(E_pertB/np.maximum(Ebg_A, 1e-30))

    # ---------------- final-state pressure fields ----------------
    print("decoding final pressure fields ...")
    psif_B = psi_in.copy(); psif_A = psi_in.copy()
    psi0f_B = psi0.copy(); psi0f_A = psi0.copy()
    def full_evolve(psi, use_V):
        n_steps = max(1, int(T_TOTAL/DT))
        dt_a = T_TOTAL/n_steps
        k_theta = np.fft.fftfreq(grid.Nt, d=1.0)*grid.Nt
        k_theta_sq = k_theta**2
        expV = np.exp(-1j*V*dt_a/2) if use_V else None
        psi_t = psi.copy()
        for _ in range(n_steps):
            if use_V:
                psi_t = psi_t*expV
            psi_t = rsp.apply(psi_t, dt_a/2)
            psi_kt = fft(psi_t, axis=1)
            psi_kt *= np.exp(-1j*HBAR*dt_a/(2*MASS)
                             * k_theta_sq[np.newaxis, :]/(grid.RR**2 + 1e-12))
            psi_t = ifft(psi_kt, axis=1)
            psi_t = rsp.apply(psi_t, dt_a/2)
            if use_V:
                psi_t = psi_t*expV
        return psi_t
    psif_B = full_evolve(psi_in, True)
    psif_A = full_evolve(psi_in, False)

    vxr, vyr, _ = decode_vec(grid, psif_B)
    vxA, vyA, _ = decode_vec(grid, psif_A)
    vr_bg, vt_bg = ub.cylinder_velocity_analytic(grid.RR, grid.TT)
    vx_bg = vr_bg*np.cos(grid.TT) - vt_bg*np.sin(grid.TT)
    vy_bg = vr_bg*np.sin(grid.TT) + vt_bg*np.cos(grid.TT)
    pB = -( (vxr**2+vyr**2) - (vx_bg**2+vy_bg**2) )/2
    pA = -( (vxA**2+vyA**2) - (vx_bg**2+vy_bg**2) )/2
    Xg = grid.RR*np.cos(grid.TT)
    Yg = grid.RR*np.sin(grid.TT)
    mask = (grid.RR < VIEW_R) & (grid.RR > R_CYL*1.02)

    print(f"[{time.time()-t0:.1f}s] SNR_B[0,last] = {SNR_B[0]:.1f}, {SNR_B[-1]:.1f} dB")
    print(f"SNR_A[0,last] = {SNR_A[0]:.1f}, {SNR_A[-1]:.1f} dB")
    print(f"Ebg_B[last] = {Ebg_B[-1]:.3e}   Ebg_A[last] = {Ebg_A[-1]:.3e}")
    print(f"E_pertB[0,last] = {E_pertB[0]:.3e}, {E_pertB[-1]:.3e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---------------- figures ----------------
    cmap = 'RdBu_r'
    qc = K_WAVE + U_INF                 # carrier wavenumber (bg phase shift)
    wc = qc**2/2 - U_INF**2/2
    thB = np.cos(qc*PROBE[0] - wc*tB)

    def _p98(a):
        v = np.abs(a[mask])
        v = v[np.isfinite(v)]
        return np.percentile(v, 98) if len(v) else 1e-12

    # unified symmetric-log color scale (linear below 1e-3): keeps the weak
    # H_B rings visible while preserving the true A/B magnitude contrast
    vmax = max(_p98(pA), _p98(pB), 1e-8)
    norm = SymLogNorm(linthresh=1e-3, linscale=0.5, vmin=-vmax, vmax=vmax)
    print(f"pA: max={np.nanmax(np.abs(pA[mask])):.2e} p98={_p98(pA):.2e} | "
          f"pB: max={np.nanmax(np.abs(pB[mask])):.2e} p98={_p98(pB):.2e} | "
          f"unified vmax={vmax:.2e}")

    # ---- Figure 1: p' fields (two standalone stacked panels) ----
    # Kept at the large source size on purpose.  The panel is placed at
    # 0.34\columnwidth (163 pt) from a 349 pt source, i.e. scaled by 0.47,
    # which puts the tick labels at 4.7 pt and the mathtext exponents (the
    # "0" of 10^0, rendered at 0.7x) at 3.3 pt.  Writing the source at the
    # placed size instead does not help: bbox_inches='tight' pads the saved
    # width back to ~167 pt, the placement scale stays ~0.98, and the raster
    # falls below the 500 dpi the figure check requires.  Changing the point
    # sizes here would need a matching dpi and placement change in the tex.
    for tag, p, lab in [('a', pA, 'free evolution'),
                        ('b', pB, '$H_B$ evolution')]:
        fig, ax = plt.subplots(figsize=(4.6, 4.17))
        pv = np.ma.masked_where(~mask, p)
        im = ax.pcolormesh(Xg, Yg, pv, cmap=cmap, norm=norm, shading='auto', rasterized=True)
        ax.add_patch(plt.Circle((0, 0), R_CYL, color='0.4'))
        ax.plot(PROBE[0], PROBE[1], 'k*', ms=8, zorder=5)
        ax.set_title(lab, fontsize=10)
        ax.set_xlim(-VIEW_R, VIEW_R); ax.set_ylim(-VIEW_R, VIEW_R)
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        if tag == 'a':
            ax.text(2.2, -3.9, 'unit cylinder', fontsize=8,
                    ha='center', color='0.25')
        else:
            ax.text(4.0, 0.3, 'sensor', fontsize=8, color='k', ha='left')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.12)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
        cbar.set_label(r"$p'$ (symmetric log)")
        fig.savefig(os.path.join(FIG_DIR, f'perturbation_fields_{tag}.pdf'),
                    dpi=400, bbox_inches='tight')
        plt.close(fig)

    # ---- Figure 2: SNR + sensor trace (legends lower left) ----
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    axes[0].plot(tA2, SNR_A, '-', color='#1f77b4', lw=1.6, label='free')
    axes[0].plot(tB2, SNR_B, '-', color='#d62728', lw=1.6, label='$H_B$')
    axes[0].axhline(0, color='0.5', ls='--', lw=0.8)
    axes[0].set_xlabel(r'$t$'); axes[0].set_ylabel('SNR (dB)')
    axes[0].set_title('Signal-to-background ratio', fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(tA, np.real(dA_full), '-', color='#1f77b4', lw=1.3,
                 label='free')
    axes[1].plot(tB, np.real(dB_full), '-', color='#d62728', lw=1.3,
                 label='$H_B$')
    axes[1].plot(tB, EPS*thB, '--', color='0.35', lw=1.0, alpha=0.8,
                 label='carrier')
    axes[1].set_xlabel(r'$t$'); axes[1].set_ylabel(r'Re $\delta\psi$')
    axes[1].set_title('Sensor trace at (4,0)', fontsize=10)
    axes[1].legend(frameon=False, fontsize=9, loc='lower left')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'perturbation_snr.pdf'), dpi=400,
                bbox_inches='tight')
    plt.close(fig)

    # standalone files (fig/perturbation)
    for fname, p in [('Figure_A.pdf', pA), ('Figure_B.pdf', pB)]:
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        pv = np.ma.masked_where(~mask, p)
        im = ax.pcolormesh(Xg, Yg, pv, cmap=cmap, norm=norm, shading='auto', rasterized=True)
        ax.add_patch(plt.Circle((0, 0), R_CYL, color='0.4'))
        ax.plot(PROBE[0], PROBE[1], 'k*', ms=8, zorder=5)
        ax.set_title(fname.replace('.pdf', ''), fontsize=11)
        ax.set_xlim(-VIEW_R, VIEW_R); ax.set_ylim(-VIEW_R, VIEW_R)
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, shrink=0.9, label=r"$p'$ (symmetric log)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=400, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(tA2, SNR_A, '-', color='#1f77b4', lw=1.6, label='A: free')
    ax.plot(tB2, SNR_B, '-', color='#d62728', lw=1.6, label='B: H$_B$')
    ax.axhline(0, color='0.5', ls='--', lw=0.8)
    ax.set_xlabel(r'$T$'); ax.set_ylabel('SNR (dB)')
    ax.set_title('Signal-to-background SNR')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'Figure_C_snr.pdf'), dpi=400,
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(tA, np.real(dA_full), '-', color='#1f77b4', lw=1.4,
            label='A: free')
    ax.plot(tB, np.real(dB_full), '-', color='#d62728', lw=1.4,
            label='B: H$_B$')
    ax.plot(tB, EPS*thB, '--', color='0.35', lw=1.1, alpha=0.8,
            label='theory $\\cos(q x_0 - \\omega t)$')
    ax.set_xlabel(r'$t$'); ax.set_ylabel(r'Re $\delta\psi$ at (4,0)')
    ax.set_title('Sensor trace (far field)')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'Figure_D_probe.pdf'), dpi=400,
                bbox_inches='tight')
    plt.close(fig)

    # ---------------- data ----------------
    np.savez(os.path.join(OUT_DIR, 'snr_data.npz'),
             t=tB, tA=tA, SNR_B=SNR_B, SNR_A=SNR_A, Ebg_B=Ebg_B,
             Ebg_A=Ebg_A, E_pertB=E_pertB, E_pertA=E_pertA,
             dB_full=dB_full, dA_full=dA_full, thB=EPS*thB)
    print(f"done [{time.time()-t0:.1f}s] -> {OUT_DIR}")

if __name__ == '__main__':
    run()
