"""
映射网格上的 Bernoulli 势精确本征态编码验证
================================================
在 MappedPolarGrid 上（无窗口、r 从圆柱面 R 开始、径向算子为局部差分）：

  H_map = (径向度规三对角) + (1/r^2) d^2/dtheta^2   物理 Laplacian
  V     = -|v_ref|^2 / 2                            Bernoulli 势 (-动压)
  H_B   = H_map + V       ==>      H_B e^{iPhi} = 0  精确零能本征态

验证结论（RadialSpectralPropagator 谱径向传播子）:
  A. 自由演化 e^{-iH_map T}  : 色散误差随时间增长 (T=0.04 时 ~0.05-0.10)
  B. H_B (解析 V)            : 误差随网格加密按 dr^2 收敛
  C. H_B exact (逐点 V_exact): 误差 ~3e-4 恒定不随 T 增长
"""
import numpy as np
import os, sys

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

from unbounded_mapping import (
    MappedPolarGrid, cylinder_velocity_analytic, R_CYL, U_INF, HBAR, MASS
)
from scipy.fft import fft, ifft

def bernoulli_potential(grid):
    """V = -|v_ref|^2/2 on the mapped grid (analytic cylinder flow)."""
    vr, vt = cylinder_velocity_analytic(grid.RR, grid.TT)
    return -(vr**2 + vt**2) / 2.0

class RadialSpectralPropagator:
    """Exact unitary radial propagator via eigendecomposition of the FULL
    mapped radial Laplacian (boundary rows are zero rows -> frozen).
    Includes boundary coupling of interior rows (a[0]*psi[0] etc.), which a
    pure interior-M x M matrix omits and which proved essential for accuracy.

    Propagator: exp(i dt L_full / 2) = V diag(e^{i dt lambda/2}) V^-1.
    """

    def __init__(self, grid):
        Nr = grid.Nr
        i_in = np.arange(1, Nr - 1)
        dr = grid.dr_tilde
        dr2 = dr * dr
        c = grid.metric_extra[i_in] + grid.metric_d1[i_in] / grid.r[i_in]
        m2 = grid.metric_d2[i_in]
        a = m2 / dr2 - c / (2 * dr)
        b = -2 * m2 / dr2
        d = m2 / dr2 + c / (2 * dr)

        Lf = np.zeros((Nr, Nr))
        for i in range(1, Nr - 1):
            Lf[i, i - 1] = a[i - 1]
            Lf[i, i] = b[i - 1]
            Lf[i, i + 1] = d[i - 1]
        # boundary rows stay zero -> psi[0], psi[-1] frozen

        self.Nr = Nr
        self.Lf = Lf
        evals, evecs = np.linalg.eig(Lf)
        self.evals = evals
        self.evecs = evecs
        self.inv_evecs = np.linalg.inv(evecs)

    def apply(self, psi, dt):
        """psi_new = exp(i dt L/2) psi  (radial kinetic step, dt = total step)."""
        V = self.evecs
        e = np.exp(1j * dt / 2 * self.evals)
        spec = self.inv_evecs @ psi         # (Nr, Nt)
        spec *= e[:, None]
        return V @ spec

def evolve_bernoulli(grid, rsp, psi, V, T, dt=0.0001):
    """Split-step evolution with H_B = H_map + V.
    Structure: [V/2] [radial/2] [angular] [radial/2] [V/2] per step.
    Radial step: exact unitary spectral propagator (RadialSpectralPropagator)."""
    n_steps = max(1, int(T / dt))
    dt_a = T / n_steps
    k_theta = np.fft.fftfreq(grid.Nt, d=1.0) * grid.Nt
    k_theta_sq = k_theta**2   # e^{in theta}: d^2/dtheta^2 -> -n^2

    expV = np.exp(-1j * V * dt_a / 2)
    psi_t = psi.copy()
    for _ in range(n_steps):
        psi_t = psi_t * expV                                    # V/2
        psi_t = rsp.apply(psi_t, dt_a / 2)                      # radial/2
        psi_kt = fft(psi_t, axis=1)                             # angular (full)
        psi_kt *= np.exp(-1j * HBAR * dt_a / (2*MASS) * k_theta_sq[np.newaxis, :] / (grid.RR**2 + 1e-12))
        psi_t = ifft(psi_kt, axis=1)
        psi_t = rsp.apply(psi_t, dt_a / 2)                      # radial/2
        psi_t = psi_t * expV                                    # V/2
    return psi_t

def eigenstate_residual(grid, psi, V):
    """Residuals of H_free psi - (|v|^2/2)psi 与 H_B psi, over the bulk."""
    Nr, Nt = psi.shape
    lap = np.zeros_like(psi)
    for i in range(1, Nr-1):
        dpsi = (psi[i+1, :] - psi[i-1, :]) / (2*grid.dr_tilde)
        d2psi = (psi[i+1, :] - 2*psi[i, :] + psi[i-1, :]) / grid.dr_tilde**2
        dpr = grid.metric_d1[i] * dpsi
        d2pr = grid.metric_d2[i] * d2psi + grid.metric_extra[i] * dpsi
        lap[i, :] = d2pr + dpr / grid.r[i]
    k_theta = np.fft.fftfreq(Nt, d=1.0) * Nt
    k_theta_sq = k_theta**2   # e^{in theta}: d^2/dtheta^2 -> -n^2
    ang = ifft(k_theta_sq[np.newaxis, :] * fft(psi, axis=1), axis=1)  # = -d^2psi/dtheta^2
    lap += -ang / (grid.RR**2 + 1e-12)                                 # +(1/r^2)d^2psi/dtheta^2
    kin = -0.5 * lap
    Hfree_psi = kin
    HB_psi = kin + V * psi
    return Hfree_psi, HB_psi

def evolve_free(grid, rsp, psi, T, dt=0.0001):
    """Free evolution with CORRECT angular wavenumber n^2 (fixed bug in grid.evolve)
    and exact unitary spectral radial step."""
    n_steps = max(1, int(T / dt))
    dt_a = T / n_steps
    k_theta = np.fft.fftfreq(grid.Nt, d=1.0) * grid.Nt
    k_theta_sq = k_theta**2
    psi_t = psi.copy()
    for _ in range(n_steps):
        psi_t = rsp.apply(psi_t, dt_a / 2)
        psi_kt = fft(psi_t, axis=1)
        psi_kt *= np.exp(-1j * HBAR * dt_a / (2*MASS) * k_theta_sq[np.newaxis, :] / (grid.RR**2 + 1e-12))
        psi_t = ifft(psi_kt, axis=1)
        psi_t = rsp.apply(psi_t, dt_a / 2)
    return psi_t

def run_case(grid, rsp, T, dt):
    psi0 = grid.encode()
    V = bernoulli_potential(grid)

    # eigenstate residuals in the bulk (R_CYL*1.05 < r < 1.5)
    bulk = (grid.RR > R_CYL*1.05) & (grid.RR < 1.5)
    Hf, HB = eigenstate_residual(grid, psi0, V)
    rB = np.sqrt(np.mean(np.abs(HB)[bulk]**2))
    rnorm = np.sqrt(np.mean(np.abs(psi0)[bulk]**2))

    # V_exact: pointwise division so that (H_map + V_exact) psi0 = 0 exactly
    Vex = -(Hf / psi0)

    # free vs Bernoulli evolution (all with exact spectral radial step)
    ps_f = evolve_free(grid, rsp, psi0.copy(), T, dt)
    ps_b = evolve_bernoulli(grid, rsp, psi0.copy(), V, T, dt)
    ps_x = evolve_bernoulli(grid, rsp, psi0.copy(), Vex, T, dt)
    vr_f, vt_f, _ = grid.decode(ps_f)
    vr_b, vt_b, _ = grid.decode(ps_b)
    vr_x, vt_x, _ = grid.decode(ps_x)
    res_f = grid.to_cartesian_bulk(vr_f, vt_f, 0.0, R_probe=1.5)
    res_b = grid.to_cartesian_bulk(vr_b, vt_b, 0.0, R_probe=1.5)
    res_x = grid.to_cartesian_bulk(vr_x, vt_x, 0.0, R_probe=1.5)
    return rB/rnorm, res_f['vel_err'], res_b['vel_err'], res_x['vel_err']

if __name__ == "__main__":
    grid = MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
    rsp = RadialSpectralPropagator(grid)
    dt = 0.0001
    print(f"Mapped grid: Nr=128, Nt=256, L_map=2.5, R_max=50, dt={dt}, bulk r in (1.05R, 1.5)")
    print(f"  {'T':>8s} | {'eigen res.':>11s} | "
          f"{'err[free]':>10s} {'err[H_B]':>10s} {'err[H_B exact]':>14s}")
    print("-"*70)
    for T in [0.001, 0.005, 0.01, 0.02, 0.04]:
        rb, ef, eb, ex = run_case(grid, rsp, T, dt)
        print(f"  {T:8.3f} | {rb:11.4f} | {ef:10.4f} {eb:10.4f} {ex:14.4f}")
