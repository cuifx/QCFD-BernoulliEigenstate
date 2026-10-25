"""
Qiskit demo on the mapped polar grid (mapped_bernoulli route)
================================================================
Replaces the 128x128 Cartesian uniform-QFT pipeline of airfoil_qiskit_demo.py
with the window-free algebraic mapping of unbounded_mapping.MappedPolarGrid.

Strategy (avoids materializing DxD dense matrices), D = Nr*Nt = 2^15:
  - T_r = exp(i dt Lf/2) (x) I_theta : 128x128 dense unitary (x) identity.
  - T_theta = QFT_theta -> Diag(exp(-i k_n^2 dt/(2 r^2))) -> QFT_theta^-1,
    the diagonal is r-dependent, so T_theta is kept as three sub-gates.
  - The V/2 phase is diagonal.

Per step: Diag(V/2) -> T_r -> QFT_theta -> Diag(kin) -> QFT_theta^-1 -> T_r -> Diag(V/2).

Output: fig/airfoil_qiskit_validation.pdf (overwrites the old file).
  Row 1: Qiskit-circuit |v|, colour scale and isolines follow the same
  convention as fig/airfoil_hb_validation.pdf (analytic-anchored vmax,
  64 black isolines), where the analytic reference itself is displayed;
  row 2: pointwise |Qiskit - analytic|/U_inf on a log colour scale
  (LogNorm 1e-6..1e-2) resolving the ~3e-4 mapped-grid floor that is
  invisible to the linear rendering above.
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator
from matplotlib.path import Path as _MplPath

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import DiagonalGate, QFTGate, UnitaryGate
    from qiskit.quantum_info import Statevector, Operator
    from qiskit_aer import AerSimulator
    _HAS_QISKIT = True
except Exception as _e:
    _HAS_QISKIT = False
    _QISKIT_ERR = _e

# state-vector simulator that executes the Strang circuit step by step
_SIM = AerSimulator(method='statevector') if _HAS_QISKIT else None

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

import airfoil_hb_quantum as hb
import unbounded_mapping as um
from mapped_bernoulli import (
    RadialSpectralPropagator, eigenstate_residual,
    evolve_bernoulli as _mapped_evolve_classic,
)

U_INF, HBAR, MASS = 1.0, 1.0, 1.0
T_EVOL = 0.01
DT = 1e-4
N_STEPS = int(round(T_EVOL / DT))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')

def _unit_mod(v):
    """Force a complex vector to strictly unit modulus (DiagonalGate
    requires |d_k|=1 to within Qiskit's tolerance)."""
    a = np.asarray(v, dtype=complex)
    return a / np.abs(a)

def _radial_unitary(grid, rsp, dt):
    """exp(i dt Lf / 4) on the radial index only; shape (Nr, Nr).

    The factor 1/4 (not 1/2) matches the classical RadialSpectralPropagator
    convention: rsp.apply(psi, dt) computes exp(i dt/2 * Lf), so a half-step
    rsp.apply(psi, dt/2) = exp(i dt/4 * Lf).  Since the Strang split uses
    rsp.apply(psi, dt/2) for each radial half-step, the Qiskit UnitaryGate
    must encode exp(i dt/4 * Lf) to match.

    Lf is the full mapped radial Laplacian (real but NON-symmetric due
    to r-dependent metric terms), so exp(i dt Lf / 4) is not strictly
    unitary.  We pass it to Qiskit's UnitaryGate with check_input=False.
    The non-unitarity is an O(dr^2) discretization artifact that vanishes
    under mesh refinement, the same artifact tolerated by the classical
    eig-based propagator that achieves ~5e-4 bulk velocity error.
    """
    from scipy.linalg import expm
    return expm(1j * dt / 4.0 * rsp.Lf)

def _angular_phase_diag(grid, dt):
    """diag(exp(-i dt k_n^2 / (2 r_i^2))) for every (i, n), as a length-D
    vector in row-major order k = i * Nt + n."""
    Nr, Nt = grid.Nr, grid.Nt
    k_theta = np.fft.fftfreq(Nt, d=1.0) * Nt
    k_theta_sq = k_theta ** 2
    phase = np.zeros((Nr, Nt), dtype=complex)
    for i in range(Nr):
        phase[i, :] = np.exp(-1j * HBAR * dt / (2 * MASS)
                             * k_theta_sq / (grid.RR[i, 0] ** 2 + 1e-12))
    return _unit_mod(phase.ravel())

def _v_phase_diag(V, dt):
    """Phase vector for the V/2 half-step, strictly unit-modulus."""
    Vr = np.real(V).ravel()
    p = np.exp(-1j * Vr * dt / 2.0)
    return _unit_mod(p)

def build_strang_circuit(grid, rsp, V, dt, n_steps):
    """Build one Qiskit QuantumCircuit that applies n_steps Strang steps
    on the (Nr*Nt)-qubit-amplitude register.

    Hilbert space layout (Qiskit little-endian):
      n_qubits = log2(Nr*Nt) = 7 (radial) + 8 (angular) = 15.
      Amplitude index k = i_r * Nt + i_th  (row-major).
      In binary (little-endian): low n_q_r=8 bits = i_th, high 7 bits = i_r.
      So angular qubits = q[0..7], radial qubits = q[8..14].

      T_r = E_r (x) I_{Nt}  acts on the 7 RADIAL qubits only.
      T_th = (I_r (x) F) D (I_r (x) F^-1):
        - QFT on the 8 ANGULAR qubits,
        - diagonal D (length D, r-dependent),
        - inverse QFT on the 8 angular qubits.

    Layer order per step (Strang):
      [Diag(V/2)] [T_r] [QFT_th] [Diag(ang)] [QFT_th^-1] [T_r] [Diag(V/2)]
    """
    Nr, Nt = grid.Nr, grid.Nt
    D = Nr * Nt
    n_qubits = int(round(np.log2(D)))
    if 2 ** n_qubits != D:
        raise ValueError("D = Nr*Nt must be a power of two; got %d" % D)
    n_q_r = int(round(np.log2(Nt)))      # angular qubits = 8 (Nt=256)
    n_q_R = n_qubits - n_q_r             # radial qubits = 7 (Nr=128)

    # Qiskit little-endian: low-index qubits hold the low bits of k.
    # k = i_r * Nt + i_th  =>  i_th = k mod Nt (low n_q_r bits),
    #                          i_r = k // Nt  (high n_q_R bits).
    ang_q = list(range(n_q_r))           # q[0..7]  -> angular
    rad_q = list(range(n_q_r, n_qubits)) # q[8..14] -> radial
    all_q = list(range(n_qubits))

    qc = QuantumCircuit(n_qubits)

    # --- Radial unitary: E_r on the 7 radial qubits (= E_r (x) I_{Nt}) ---
    E_r = _radial_unitary(grid, rsp, dt)        # (Nr, Nr) complex near-unitary
    # UnitaryGate on rad_q applies E_r (x) I on the rest (identity on ang_q).
    # check_input=False because the discretized Laplacian propagator is not
    # strictly unitary (see _radial_unitary docstring); the non-unitarity is
    # an O(dr^2) discretization artifact, same as the classical reference.
    op_T_r = UnitaryGate(E_r, label='T_r', check_input=False)

    diag_V = _v_phase_diag(V, dt)               # length D
    diag_ang = _angular_phase_diag(grid, dt)    # length D

    for _ in range(n_steps):
        qc.append(DiagonalGate(diag_V), all_q)
        qc.append(op_T_r, rad_q)
        qc.append(QFTGate(num_qubits=n_q_r), ang_q)
        qc.append(DiagonalGate(diag_ang), all_q)
        qc.append(QFTGate(num_qubits=n_q_r).inverse(), ang_q)
        qc.append(op_T_r, rad_q)
        qc.append(DiagonalGate(diag_V), all_q)
    return qc, ang_q, rad_q, all_q

def evolve_qiskit_mapped(grid, rsp, psi0, V, T, dt=DT):
    """Execute the Strang circuit on the Qiskit state-vector simulator.

    The gate sequence of build_strang_circuit is run by Qiskit itself, one
    Strang step at a time, and the result is read back with
    get_statevector.  No part of the gate sequence is reproduced by hand.

    One ingredient cannot be a gate.  The exact potential is complex, and
    its modulus exp(Im V dt/2) is non-unitary, while DiagonalGate carries
    only the unit-modulus phase exp(-i Re V dt/2).  The modulus is applied
    as a per-amplitude rescaling between the two potential half-steps of
    each Strang step, which is the split the paper describes.

    Aer normalizes a saved statevector, so the modulus accumulated by the
    rescaling is carried in lognorm and restored at the end, which keeps
    the returned amplitude on the same scale as the classical reference.
    """
    if not _HAS_QISKIT:
        raise RuntimeError("Qiskit is required for this route: %s"
                           % _QISKIT_ERR)
    n_steps = max(1, int(round(T / dt)))
    dt_a = T / n_steps

    # one Strang step as a Qiskit circuit, transpiled once for Aer
    qc_step, _, _, _ = build_strang_circuit(grid, rsp, V, dt_a, 1)
    n_qubits = qc_step.num_qubits
    tqc = transpile(qc_step, _SIM, optimization_level=0)

    # modulus of the complex potential, applied classically at both ends
    rescale = np.exp(np.asarray(V, dtype=complex).imag * dt_a / 2.0).ravel()

    Nr, Nt = grid.Nr, grid.Nt
    data = np.asarray(psi0, dtype=complex).ravel()
    nrm = np.linalg.norm(data)
    lognorm = float(np.log(nrm))
    data = data / nrm

    t0 = time.time()
    for s in range(n_steps):
        data = data * rescale
        nrm = np.linalg.norm(data)
        lognorm += float(np.log(nrm))
        data = data / nrm
        qc_sv = QuantumCircuit(n_qubits)
        qc_sv.set_statevector(Statevector(data))
        qc_sv.compose(tqc, inplace=True)
        qc_sv.save_statevector()
        data = np.asarray(
            _SIM.run(qc_sv, shots=1).result().get_statevector(0)).ravel()
        nrm = np.linalg.norm(data)
        lognorm += float(np.log(nrm))
        data = data / nrm
        data = data * rescale
        if (s + 1) % 25 == 0:
            print("    qiskit step %3d/%d  [%.1fs]"
                  % (s + 1, n_steps, time.time() - t0))

    data = data * np.exp(lognorm)
    return data.reshape(Nr, Nt)

def evolve_classic_mapped(grid, rsp, psi0, V, T, dt=DT):
    """Classical (numpy) reference evolution using the SAME expm-based
    radial propagator as the Qiskit circuit (via _radial_unitary), so
    the fidelity comparison is apples-to-apples.  The original
    mapped_bernoulli.evolve_bernoulli uses the eig-based propagator
    (rsp.apply), which is mathematically identical to expm for the
    diagonalizable Lf but slightly less accurate in practice."""
    n_steps = max(1, int(round(T / dt)))
    dt_a = T / n_steps
    Nr, Nt = grid.Nr, grid.Nt
    E_r = _radial_unitary(grid, rsp, dt_a)
    expV = np.exp(-1j * V * dt_a / 2.0)            # full complex V (not just real)
    k_theta = np.fft.fftfreq(Nt, d=1.0) * Nt
    k_theta_sq = k_theta ** 2
    psi = psi0.copy().astype(complex)
    for _ in range(n_steps):
        psi = psi * expV
        psi = E_r @ psi
        psi_k = np.fft.fft(psi, axis=1)
        psi_k *= np.exp(-1j * HBAR * dt_a / (2 * MASS)
                        * k_theta_sq[np.newaxis, :] / (grid.RR ** 2 + 1e-12))
        psi = np.fft.ifft(psi_k, axis=1)
        psi = E_r @ psi
        psi = psi * expV
    return psi

def run_airfoil(geom, T=T_EVOL):
    R, zc, alpha, gamma = geom['R'], geom['zc'], geom['alpha'], geom['gamma']
    um.R_CYL = R
    grid = um.MappedPolarGrid(Nr=128, Nt=256, L_map=2.5, R_max_phys=50.0)
    rsp = RadialSpectralPropagator(grid)

    phi = hb.encode_potential(grid, R, alpha)
    psi0 = np.exp(1j * phi / HBAR)
    V = hb.bernoulli_potential(grid, R, alpha)
    Hf0, _ = eigenstate_residual(grid, psi0, V)
    V_ex = -Hf0 / psi0

    psi_ref = evolve_classic_mapped(grid, rsp, psi0.copy(), V_ex, T, dt=DT)
    psi_q = evolve_qiskit_mapped(grid, rsp, psi0, V_ex, T, dt=DT)
    f = abs(np.sum(np.conj(psi_q.ravel()) * psi_ref.ravel())) ** 2 \
        / (np.linalg.norm(psi_q.ravel()) * np.linalg.norm(psi_ref.ravel())) ** 2
    print(f"  {geom['name']:<10s} mapped grid state fidelity "
          f"Qiskit vs numpy = {f:.6f}")

    vr, vt, _ = hb.decode_velocity(grid, psi_q, gamma)
    vx, vy, zeta = hb.to_cartesian(grid, zc, vr, vt)
    vel_q = hb.physical_velocity(vx, vy, zeta, geom)

    vra, vta = hb.analytic_velocity(grid, R, alpha, gamma)
    vx_a, vy_a, _ = hb.to_cartesian(grid, zc, vra, vta)
    vel_a = hb.physical_velocity(vx_a, vy_a, zeta, geom)

    e_x = hb.rel_err_zeta(grid, R, vr, vt, gamma, alpha)
    print(f"  {'':10s} |v| rel err (Qiskit H_B(exact V_ex) vs analytic, "
          f"bulk annulus 1.05R<r<1.5) = {e_x:.4e}")

    th = np.linspace(0, 2 * np.pi, 400)
    zeta_outline = zc + R * np.exp(1j * th)
    z_outline = hb.fwd_map(zeta_outline, geom)
    z = hb.fwd_map(zeta, geom)
    cx = float(np.mean(z_outline.real))
    cy = float(np.mean(z_outline.imag))
    zc_w = cx + 1j * cy
    # window half-width, derived from the mapped outline instead of assumed:
    # every section of the paper is normalised to unit chord (see
    # airfoil_hb_quantum.joukowski_geometry, which rescales by 1/chord), so a
    # fixed value on the non-NACA branch draws the Joukowski panel at the
    # pre-normalisation zoom and the airfoil fills a fraction of the frame.
    chord = float(z_outline.real.max() - z_outline.real.min())
    LZ = 1.2 * chord
    zr = (z - zc_w) * np.exp(-1j * alpha) + zc_w
    zo_r = (z_outline - zc_w) * np.exp(-1j * alpha) + zc_w
    m = (grid.RR > R * 1.05) & (np.abs(zr.real - cx) < LZ) \
        & (np.abs(zr.imag - cy) < LZ)
    # interpolation-input mask: display window widened by PAD so rings
    # beyond the display corners cover them (see airfoil_hb_quantum.py);
    # the window mask m above stays for metrics/prints.
    PAD = 1.5
    m_int = (grid.RR > R * 1.05) & (np.abs(zr.real - cx) < PAD * LZ) \
        & (np.abs(zr.imag - cy) < PAD * LZ)
    return dict(vel_q=vel_q, vel_a=vel_a, zr=zr, zo_r=zo_r, cx=cx, cy=cy,
                LZ=LZ, m=m, m_int=m_int)

def plot_three(geoms, out):
    # 10.6 in wide: the panel is placed at 0.8\textwidth (383 pt), so this
    # gives a placement scale of 0.50 and puts the 10 pt source labels at the
    # 5 pt panel baseline; at the previous 15.6 in they came out at 3.6 pt.
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.66))
    ims = {}
    # log colour scale for the difference row: matched to the signal
    # (bulk ~3e-4, stagnation hot spots ~1e-2), which the linear |v|
    # rendering above cannot resolve.
    d_norm = LogNorm(vmin=1e-6, vmax=1e-2)
    for j, g in enumerate(geoms):
        r = run_airfoil(g)
        cx, cy, LZ = r['cx'], r['cy'], r['LZ']
        m = r['m']
        m_int = r['m_int']
        zr = r['zr']
        zo_r = r['zo_r']

        n_disp = 320
        xs = np.linspace(cx - LZ, cx + LZ, n_disp)
        ys = np.linspace(cy - LZ, cy + LZ, n_disp)
        GX, GY = np.meshgrid(xs, ys)

        poly_pts = np.column_stack([zo_r.real, zo_r.imag])
        inside_mask = _MplPath(poly_pts).contains_points(
            np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

        # column-wise vmax: 98th percentile of the interpolated analytic
        # field, ceil to 0.1 -- identical to the pipeline of
        # fig/airfoil_hb_validation.pdf (same grid, window, mask and
        # interpolator), so the two figures share the same colour scale
        # and the same 64 isoline levels per column.
        pts = np.column_stack([zr.real[m_int], zr.imag[m_int]])
        f_a = LinearNDInterpolator(pts, r['vel_a'][m_int], fill_value=np.nan)
        FA_v = f_a(GX, GY)
        FA_v[inside_mask] = np.nan
        vmax_col = float(np.percentile(FA_v[~np.isnan(FA_v)], 98))
        vmax_col = np.ceil(vmax_col * 10) / 10

        # interpolate the RAW decoded field (hb-script convention, no
        # pre-interpolation clipping or thresholding): clipping the
        # above-vmax suction-peak values to exactly vmax creates a flat
        # plateau among unclipped neighbours, and the two top isoline
        # levels then trace ragged plateau-fragment loops that render as
        # dark speckles.  pcolormesh(vmin, vmax) already saturates the
        # display colours, so the raw field needs no manual clipping.
        f_q = LinearNDInterpolator(pts, r['vel_q'][m_int], fill_value=np.nan)
        FQ = f_q(GX, GY)
        FQ[inside_mask] = np.nan

        # per-column contour levels: same analytic-anchored convention as
        # fig/airfoil_hb_validation.pdf, so the decoded isolines can be
        # compared against that figure directly.
        levels = np.linspace(0.0, vmax_col, 64)

        sc = axes[0, j].pcolormesh(GX, GY, FQ, cmap='viridis', vmin=0,
                                   vmax=vmax_col, shading='auto',
                                   rasterized=True)
        ims[0] = sc
        axes[0, j].contour(GX, GY, FQ, levels=levels, colors='k',
                           linewidths=0.35, alpha=0.5)
        axes[0, j].plot(zo_r.real, zo_r.imag, 'k-', lw=0.8)
        axes[0, j].set_xlim(cx - LZ, cx + LZ)
        axes[0, j].set_ylim(cy - LZ, cy + LZ)
        axes[0, j].set_aspect('equal')
        axes[0, j].set_title(
            f"{g['name']} Qiskit H$_B$ circuit $|v|$ (T=0.01)", fontsize=11)
        axes[0, j].set_xlabel('Re $z$')
        axes[0, j].set_ylabel('Im $z$')

        # row 2: pointwise Qiskit-analytic difference on a log colour
        # scale (raw fields, no clipping/thresholding, so the stagnation
        # hot spots stay physical).  The differences are ~3e-4 in the
        # bulk -- invisible against the |v| colour/isoline rendering
        # above, but resolved here.  The printed window statistics keep
        # the display mask m; the interpolation uses the widened m_int.
        d = np.abs(r['vel_q'] - r['vel_a']) / U_INF
        print(f"  {'':10s} |dv|/U: mean={d[m].mean():.2e} "
              f"max={d[m].max():.2e}")
        f_d = LinearNDInterpolator(pts, d[m_int], fill_value=np.nan)
        FD = f_d(GX, GY)
        FD[inside_mask] = np.nan
        ax_d = axes[1, j]
        ims[1] = ax_d.pcolormesh(GX, GY, FD, cmap='magma', norm=d_norm,
                                 shading='auto', rasterized=True)
        ax_d.plot(zo_r.real, zo_r.imag, 'k-', lw=0.8)
        ax_d.set_xlim(cx - LZ, cx + LZ)
        ax_d.set_ylim(cy - LZ, cy + LZ)
        ax_d.set_aspect('equal')
        ax_d.set_title(f"{g['name']}  $|$Qiskit $-$ analytic$|/U_\\infty$",
                       fontsize=11)
        ax_d.set_xlabel('Re $z$')
        ax_d.set_ylabel('Im $z$')
    # one colorbar per row, vertically aligned with the row on the right
    # edge (each airfoil column has its own vmax, so a single shared bar
    # over the whole figure would be ambiguous; the difference row carries
    # its own logarithmic scale).
    fig.subplots_adjust(left=0.06, right=0.90, top=0.95, bottom=0.06,
                        wspace=0.35, hspace=0.32)
    cb_labels = {0: r'$|v|/U_\infty$', 1: r'$|\Delta v|/U_\infty$'}
    for row in range(2):
        pos = axes[row, -1].get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.016, pos.height])
        fig.colorbar(ims[row], cax=cax, label=cb_labels[row])
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out)}")

def main():
    t0 = time.time()
    print("=" * 78)
    print(f" Qiskit demo on the mapped polar grid: Strang-split H_B, "
          f"T={T_EVOL}, {N_STEPS} steps, dt={DT}")
    print(f"  Qiskit available: {_HAS_QISKIT}")
    if not _HAS_QISKIT:
        print(f"  import error: {_QISKIT_ERR}")
    print("=" * 78)
    geoms = [hb.joukowski_geometry(), hb.naca_geometry('NACA 0012'),
             hb.naca_geometry('NACA 2412')]
    out = os.path.join(FIG_DIR, 'airfoil_qiskit_validation.pdf')
    plot_three(geoms, out)
    print(f"  done [{time.time()-t0:.0f}s]")

if __name__ == '__main__':
    main()
