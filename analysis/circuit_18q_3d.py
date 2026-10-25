# -*- coding: utf-8 -*-
"""18-qubit experiment: the three-dimensional stationary encoding on a 64^3 grid.

Register map.  A 64^3 field needs 3*log2(64) = 18 qubits.  The flattened field
index is n = i*64^2 + j*64 + k (x, y, z), and qubit p carries bit p of n, so
q0-q5 = z index, q6-q11 = y index, q12-q17 = x index (Qiskit little-endian).

What is gate-level here
-----------------------
  * the three QFTs and their inverses (H, controlled-phase, swaps);
  * the diagonal kinetic operator D(k^2), synthesized as an explicit
    phase-polynomial circuit (CNOT ladder + RZ).  That is possible because the
    fftfreq index is a *quadratic* function of the register bits, so its Walsh
    spectrum is sparse -- the same reason code/real_device/sm/verify_phase_poly.py
    works.  Cost is O(n_axis^2), not 2^18.
  * for the statevector branch, nothing else.

What is a disclosed simulator shortcut
--------------------------------------
  * the encoded state exp(i Phi) is installed with set_statevector.  The
    phase-encoding diagonal is 2^18 wide and is exactly the part the paper's
    Sec. VII arithmetic oracle is for; it is NOT synthesized here.
  * the Bernoulli phase exp(-i V dt/2) of the H_B branch is applied through a
    native Diagonal instruction, same caveat.

Checks
------
  1. the 6-qubit kinetic phase polynomial equals diag(exp(i phi)) on random
     states, per axis;
  2. the full 18-qubit circuit equals the classical spectral step
     ifftn(exp(-iTk^2/2) fftn(.)) on random states;
  3. free-propagator velocity error vs T, circuit vs classical, decoded on the
     probe.
"""
import os
import sys
import time

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import winged_body_3d_probe as W                                    # noqa: E402
# Geometry: the AGARD-B calibration model.  FIN_AZ is derived from NW at import
# time, so it has to be reset alongside NW (leaving it at the cruciform value
# silently builds four fins instead of the single wing pair of the standard
# model); AXS is the axial half-length of the parametrisation.
W.BODY_STYLE = 'agardb'
W.NW = 2
W.FIN_AZ = 2.0 * np.pi / W.NW
AXS = W.AB_AX if W.BODY_STYLE == 'agardb' else W.AX

from qiskit import QuantumCircuit, transpile                        # noqa: E402
from qiskit_aer import AerSimulator                                 # noqa: E402
from qiskit.circuit.library import QFTGate                          # noqa: E402

U = 1.0
N = 64                      # 2^6 per axis -> 18 qubits
L = 4.0                     # box half-size; h = 0.127, same resolution as the 96^3/L=6 study
NF, NT = 32, 192            # the panel set that the geometry audit verified
TAPER = 0.12
CACHE = 'code/data/circuit18q_agardb_cache.npz'

h = 2.0 * L / (N - 1)
gg = np.linspace(-L, L, N)
X, Y, Z = np.meshgrid(gg, gg, gg, indexing='ij')
_k1 = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
KX, KY, KZ = np.meshgrid(_k1, _k1, _k1, indexing='ij')
K2 = KX ** 2 + KY ** 2 + KZ ** 2

sim = AerSimulator(method='statevector')


# ---------------------------------------------------------------- set-up
def build_phi():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if d['PHI'].shape == (N, N, N):
            print('loaded Phi from %s' % CACHE)
            return d['PHI']
    print('solving panels and evaluating Phi on %d^3 ...' % N)
    t0 = time.time()
    cen, nrm, area, tri = W.build_panels(NF, NT)
    sig, Aop, rhs = W.solve_sources(cen, nrm, area, tri)
    print('  panels %d  Q = %+.3e  BC residual %.1e'
          % (len(cen), (sig * area).sum(), np.abs(Aop @ sig - rhs).max()))
    P = np.stack([X, Y, Z], -1)
    out = np.empty((N, N, N))
    step = 20000
    flat = P.reshape(-1, 3)
    for c0 in range(0, len(flat), step):
        out.reshape(-1)[c0:c0 + step] = W.potential(flat[c0:c0 + step], cen, area, sig)
    np.savez(CACHE, PHI=out)
    print('  cached (%.0f s)' % (time.time() - t0))
    return out


PHI = build_phi()
PHIp = np.arccos(np.clip(X / AXS, -1.0, 1.0))
THp = np.arctan2(Z, Y)
RPp = np.hypot(Y, Z)
fluid = RPp > W.radius(PHIp, THp)
probe = binary_erosion(fluid, iterations=2) & (np.sqrt(X ** 2 + Y ** 2 + Z ** 2) < 3.0)


def window3d(LL, tr=TAPER):
    w = LL * tr
    win = np.ones_like(X)
    for C in (X, Y, Z):
        wc = np.ones_like(X)
        ml = C < -LL + w
        mr = C > LL - w
        wc[ml] = 0.5 * (1 - np.cos(np.pi * (C[ml] + LL) / w))
        wc[mr] = 0.5 * (1 - np.cos(np.pi * (LL - C[mr]) / w))
        win *= wc
    return win


def grad_spec(f):
    return (ifftn(1j * KX * fftn(f)), ifftn(1j * KY * fftn(f)), ifftn(1j * KZ * fftn(f)))


def grad_fd(f):
    gx = np.zeros_like(f); gy = np.zeros_like(f); gz = np.zeros_like(f)
    gx[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    gz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * h)
    return gx, gy, gz


_g = grad_spec(PHI - U * X)
# grad_spec returns a complex array; Phi - U x is real, so keep the real part
# (otherwise every downstream array inherits the complex dtype, which silently
# broke the Normalize() call that colours the surface panels).
VRX, VRY, VRZ = (U + _g[0]).real, _g[1].real, _g[2].real
V_ANA = -0.5 * (VRX ** 2 + VRY ** 2 + VRZ ** 2)
WIND = window3d(L)
psi0 = WIND * np.exp(1j * PHI)
psi0 = psi0 / np.linalg.norm(psi0)          # set_statevector requires a unit vector
print('grid %d^3  h = %.5f  probe %d points  qubits = 18' % (N, h, probe.sum()))


# ---------------------------------------------------------------- circuit parts
def kinetic_phases(T):
    """phase vector -T k^2/2 indexed by the flattened register value n."""
    return -0.5 * T * K2.ravel()


def poly_terms_axis():
    """Coefficients of g(y) = c * (y - 64 b5)^2 expanded in the six bits.

    g = sum_p 4^p b_p + 2 sum_{p<q<5} 2^{p+q} b_p b_q - sum_{p<5} 2^{p+6} b_p b_5
    (the b5 linear terms cancel identically).
    """
    lin = np.zeros(6)
    quads = {}
    for p in range(6):
        lin[p] += 4.0 ** p
    for p in range(5):
        for q in range(p + 1, 5):
            quads[(p, q)] = quads.get((p, q), 0.0) + 2.0 * 2.0 ** (p + q)
        quads[(p, 5)] = quads.get((p, 5), 0.0) - 2.0 ** (p + 6)
    return lin, quads


def apply_kinetic_poly(qc, reg, T):
    """Gate-level diagonal exp(-i T k^2 / 2) on one 6-qubit Fourier register."""
    c = -0.5 * T * (2.0 * np.pi / (N * h)) ** 2
    lin, quads = poly_terms_axis()
    lin = lin * c
    for (i, j), a in quads.items():
        a = a * c
        theta = -a / 2.0
        lin[i] += a / 2.0
        lin[j] += a / 2.0
        qc.cx(reg[i], reg[j])
        qc.rz(theta, reg[j])
        qc.cx(reg[i], reg[j])
    for p in range(6):
        qc.rz(lin[p], reg[p])


def qft_block(qc, reg, inverse=False):
    """Textbook QFT on reg: |x> -> N^-1/2 sum_y exp(2 pi i x y / N) |y>, with
    reg[0] the least significant bit of the register value.  Written out in
    elementary gates because Aer does not accept the ``qft`` custom
    instruction.  The Hadamard/cphase cascade runs MSB-first (r[0] = reg[n-1]),
    which is the ordering that reproduces the natural-index DFT; verified
    against exp(2 pi i x y / N)/sqrt(N) in check [0] below.
    """
    n = len(reg)
    r = list(reversed(reg))
    if not inverse:
        for i in range(n):
            qc.h(r[i])
            for j in range(i + 1, n):
                qc.cp(np.pi / 2.0 ** (j - i), r[j], r[i])
        for i in range(n // 2):
            qc.swap(reg[i], reg[n - 1 - i])
    else:
        for i in range(n // 2):
            qc.swap(reg[i], reg[n - 1 - i])
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                qc.cp(-np.pi / 2.0 ** (j - i), r[j], r[i])
            qc.h(r[i])


def build_free_circuit(T):
    qc = QuantumCircuit(18)
    qc.set_statevector(psi0.ravel())
    regs = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for r in regs:
        qft_block(qc, r, inverse=False)
    for r in regs:
        apply_kinetic_poly(qc, r, T)
    for r in regs:
        qft_block(qc, r, inverse=True)
    qc.save_statevector()
    return qc


# ---------------------------------------------------------------- check 0
print()
print('[0] QFT convention: 3-qubit circuit vs the exact DFT matrix')
from qiskit.quantum_info import Operator                        # noqa: E402
_qc = QuantumCircuit(3)
qft_block(_qc, [0, 1, 2], inverse=False)
_x = np.arange(8)
_ref = np.exp(2j * np.pi * np.outer(_x, _x) / 8) / np.sqrt(8)
_u = Operator(_qc).data
print('    max|U - DFT| = %.3e    (and the conjugate transpose for the inverse)'
      % np.max(np.abs(_u - _ref)))
_qc = QuantumCircuit(3)
qft_block(_qc, [0, 1, 2], inverse=True)
print('    max|U_inv - DFT^dag| = %.3e'
      % np.max(np.abs(Operator(_qc).data - _ref.conj().T)))

# ---------------------------------------------------------------- check 1
print()
print('[1] 6-qubit kinetic phase polynomial vs the exact diagonal')
rng = np.random.default_rng(7)
reg = list(range(6))
for T in (0.04, 0.2):
    qc = QuantumCircuit(6)
    inp = rng.normal(size=64) + 1j * rng.normal(size=64)
    inp /= np.linalg.norm(inp)
    qc.set_statevector(inp)
    apply_kinetic_poly(qc, reg, T)
    qc.save_statevector()
    out = np.asarray(sim.run(qc, shots=1).result().get_statevector())
    yv = np.arange(64)
    b5 = (yv >> 5) & 1
    g = (yv - 64 * b5) ** 2
    c = -0.5 * T * (2.0 * np.pi / (N * h)) ** 2
    ref = np.exp(1j * c * g) * inp
    ov = abs(np.vdot(ref, out)) / (np.linalg.norm(ref) * np.linalg.norm(out))
    print('    T = %.2f   |<ref|circ>| = %.15f   1-fidelity = %.2e'
          % (T, ov, 1 - ov))

# ---------------------------------------------------------------- check 2
print()
print('[2] full 18-qubit circuit vs the classical spectral step')
for T in (0.04, 0.2):
    qc = build_free_circuit(T)
    out = np.asarray(sim.run(qc, shots=1).result().get_statevector())
    ref = ifftn(np.exp(-0.5j * T * K2) * fftn(psi0))
    ov = abs(np.vdot(ref.ravel(), out)) / (np.linalg.norm(ref) * np.linalg.norm(out))
    print('    T = %.2f   |<ref|circ>| = %.15f   1-fidelity = %.2e'
          % (T, ov, 1 - ov))
qc = build_free_circuit(0.04)
try:
    tqc = transpile(qc, sim, optimization_level=1)
    print('    circuit size (T = 0.04):  %s' % dict(tqc.count_ops()))
    print('    total ops = %d   depth = %d' % (sum(tqc.count_ops().values()), tqc.depth()))
except Exception as e:
    print('    (transpile/decompose not available: %s)' % e)
    print('    circuit as built: %s' % dict(qc.count_ops()))


# ---------------------------------------------------------------- check 3
print()
print('[3] free-propagator velocity error vs T, circuit vs classical')
def vel_err(psi):
    a, b, c_ = grad_fd(psi)
    rr = np.abs(psi) ** 2 + 1e-12
    dd = ((np.imag(np.conj(psi) * a) / rr - VRX) ** 2
          + (np.imag(np.conj(psi) * b) / rr - VRY) ** 2
          + (np.imag(np.conj(psi) * c_) / rr - VRZ) ** 2)
    return np.sqrt(np.mean(dd[probe])) / U


floor = vel_err(psi0)
print('    t=0 decoding floor = %.4e' % floor)
print('    %-6s %-13s %-13s %s' % ('T', 'circuit', 'classical', 'infidelity'))
rows = []
for T in (0.005, 0.01, 0.02, 0.04, 0.10, 0.20, 0.40):
    qc = build_free_circuit(T)
    out = np.asarray(sim.run(qc, shots=1).result().get_statevector()).reshape(N, N, N)
    ref = ifftn(np.exp(-0.5j * T * K2) * fftn(psi0))
    # phase-insensitive: the phase polynomial and the QFT both leave a global phase
    dif = 1.0 - abs(np.vdot(ref.ravel(), out.ravel()))
    rows.append((T, vel_err(out), vel_err(ref), dif))
    print('    %-6.3f %-13.4e %-13.4e %.2e' % rows[-1])


# ---------------------------------------------------------------- check 4
print()
print('[4] exact end-to-end at 18 qubits, NO simulator shortcut: uniform flow')
print('    encoding is gate-level (18 H + 6 RZ, the circuit of Sec. VII),')
print('    and exp(i U x) is an exact eigenstate of the free propagator')
m_wave = 2
UU = 2.0 * np.pi * m_wave / (N * h)        # on-grid wavenumber -> exact Fourier mode
dx = gg[1] - gg[0]
print('    U = %.6f  (grid wavenumber %d, spacing %.6f)' % (UU, m_wave, 2 * np.pi / (N * h)))
qc = QuantumCircuit(18)
for q_ in range(18):
    qc.h(q_)
for p in range(6):
    qc.rz(UU * dx * 2 ** p, 12 + p)        # only the x register carries Phi = U x
for r in [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]:
    qft_block(qc, r, inverse=False)
for r in [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]:
    apply_kinetic_poly(qc, r, 0.04)
for r in [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]:
    qft_block(qc, r, inverse=True)
qc.save_statevector()
psiA = np.exp(1j * UU * gg[:, None, None] * np.ones((1, N, N)))
psiA = psiA / np.linalg.norm(psiA)
outA = np.asarray(sim.run(qc, shots=1).result().get_statevector())
ov = np.vdot(psiA.ravel(), outA)
print('    |<psi(0)|psi(T)>|   = %.15f      (1 - |.| = %.2e)' % (abs(ov), 1 - abs(ov)))
print('    arg<psi(0)|psi(T)>  = %+.9f rad' % np.angle(ov))
print('      the absolute phase is convention dependent (the six encoding RZ and')
print('      the 63 phase-polynomial RZ each leave a global phase); what is')
print('      observable is that the state comes back unchanged to %.1e, so the' % (1 - abs(ov)))
print('      velocity field is preserved exactly.')


def vel_err_uniform(psi, Uref):
    a, b, c_ = grad_fd(psi)
    rr = np.abs(psi) ** 2 + 1e-12
    dd = ((np.imag(np.conj(psi) * a) / rr - Uref) ** 2
          + (np.imag(np.conj(psi) * b) / rr) ** 2
          + (np.imag(np.conj(psi) * c_) / rr) ** 2)
    return np.sqrt(np.mean(dd[probe]))


print('    decoded velocity vs (U,0,0):')
print('      circuit state   rms error = %.4e' % vel_err_uniform(outA.reshape(N, N, N), UU))
print('      analytic state  rms error = %.4e' % vel_err_uniform(psiA, UU))
print('      difference                   %.2e' % abs(
    vel_err_uniform(outA.reshape(N, N, N), UU) - vel_err_uniform(psiA, UU)))
print('      the residual is not a circuit error: the centred-difference decode')
print('      of a plane wave returns sin(U h)/h = %.6f instead of U = %.6f'
      % (np.sin(UU * h) / h, UU))
print('      (the same O(h^2) decode artifact that sets the t=0 floor).')

np.savez('code/data/circuit18q_result.npz',
         T=np.array([r[0] for r in rows]),
         circ=np.array([r[1] for r in rows]),
         clas=np.array([r[2] for r in rows]),
         infid=np.array([r[3] for r in rows]),
         floor=floor)

# ---------------------------------------------------------------- figure
import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
matplotlib.rcParams['pdf.fonttype'] = 42

Tv = np.array([r[0] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
ax[0].loglog(Tv, [r[1] for r in rows], 'o', ms=5, color='#d63031',
             label='18-qubit circuit', mfc='white', mew=1.2)
ax[0].loglog(Tv, [r[2] for r in rows], '-', lw=1.2, color='#0984e3',
             label='classical spectral step')
ax[0].axhline(floor, color='#636e72', ls=':', lw=1.2)
ax[0].text(Tv[0], floor * 1.15, r'$t{=}0$ decoding floor', fontsize=7,
           color='#636e72')
ax[0].set_xlabel(r'evolution time $T$')
ax[0].set_ylabel(r'$\|v-v_{\rm ref}\|_{\rm rms}/U_\infty$')
ax[0].set_title(r'free propagator, $64^3$ field on 18 qubits', fontsize=8)
ax[0].legend(fontsize=7, loc='upper left')

ax[1].semilogy(Tv, np.maximum([abs(r[3]) for r in rows], 1e-17), 's-', ms=4,
               lw=1.2, color='#00b894')
ax[1].axhline(2.2e-16, color='#636e72', ls=':', lw=1.2)
ax[1].set_xlabel(r'evolution time $T$')
ax[1].set_ylabel(r'$1-|\langle\psi_{\rm class}|\psi_{\rm circ}\rangle|$')
ax[1].set_title('circuit vs classical, 299 gates / depth 52', fontsize=8)
ax[1].set_ylim(1e-17, 1e-13)
for a in ax:
    a.grid(True, which='both', alpha=0.25, lw=0.6)
    a.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig('code/data/circuit18q_3d.png', dpi=190)
print('figure -> code/data/circuit18q_3d.png')


# ---------------------------------------------------------------- check 5 / 3-D view
print()
print('[5] body-surface comparison: theoretical flow vs the 18-qubit state')
from scipy.interpolate import RegularGridInterpolator              # noqa: E402


def decoded_v(psi):
    a, b, c_ = grad_fd(psi)
    rr = np.abs(psi) ** 2 + 1e-12
    return (np.imag(np.conj(psi) * a) / rr,
            np.imag(np.conj(psi) * b) / rr,
            np.imag(np.conj(psi) * c_) / rr)


RS = 1.12                        # shell, inflated from the wetted surface
n_phi, n_th = 72, 264
PHg, THg = np.meshgrid(np.linspace(0.02, np.pi - 0.02, n_phi),
                       np.linspace(0.0, 2.0 * np.pi, n_th), indexing='ij')
rho_s = RS * W.radius(PHg, THg)
Xs, Ys, Zs = AXS * np.cos(PHg), rho_s * np.cos(THg), rho_s * np.sin(THg)
pts = np.stack([Xs, Ys, Zs], -1)


def sample(f):
    it = RegularGridInterpolator((gg, gg, gg), f, bounds_error=False, fill_value=np.nan)
    return it(pts.reshape(-1, 3)).reshape(pts.shape[:-1])


s0 = np.asarray(sim.run(build_free_circuit(0.0), shots=1).result()
                .get_statevector()).reshape(N, N, N)
s4 = np.asarray(sim.run(build_free_circuit(0.4), shots=1).result()
                .get_statevector()).reshape(N, N, N)
v0, v4 = decoded_v(s0), decoded_v(s4)
SPEED = lambda v: np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
GAP = lambda v: np.sqrt((v[0] - VRX) ** 2 + (v[1] - VRY) ** 2 + (v[2] - VRZ) ** 2)
ST, S0, S4 = sample(SPEED((VRX, VRY, VRZ))), sample(SPEED(v0)), sample(SPEED(v4))
G0, G4 = sample(GAP(v0)), sample(GAP(v4))
inb = np.isfinite(ST)
print('    nan fraction  theory %.4f   circ0 %.4f   gap0 %.4f   gap4 %.4f'
      % (np.mean(~np.isfinite(ST)), np.mean(~np.isfinite(S0)),
         np.mean(~np.isfinite(G0)), np.mean(~np.isfinite(G4))))
for nm_, F_ in (('theory', ST), ('circ0', S0), ('circ4', S4)):
    print('    shell |v|/U %-7s p5/p50/p95 = %.3f %.3f %.3f'
          % (nm_, *np.nanpercentile(F_, [5, 50, 95])))
for nm_, F_ in (('gap0', G0), ('gap4', G4)):
    print('    shell gap   %-7s p5/p50/p95 = %.2e %.2e %.2e'
          % (nm_, *np.nanpercentile(F_, [5, 50, 95])))
print('    shell rms |v|/U     theory %.4f   circuit(T=0) %.4f   circuit(T=0.4) %.4f'
      % (np.sqrt(np.nanmean(ST ** 2)), np.sqrt(np.nanmean(S0 ** 2)),
         np.sqrt(np.nanmean(S4 ** 2))))
print('    shell rms gap/U     T=0      %.3e        T=0.4  %.3e'
      % (np.sqrt(np.nanmean(G0 ** 2)), np.sqrt(np.nanmean(G4 ** 2))))

vC0 = decoded_v(psi0)
vC4 = decoded_v(ifftn(np.exp(-0.5j * 0.4 * K2) * fftn(psi0)))
GC0, GC4 = sample(GAP(vC0)), sample(GAP(vC4))
print('    shell rms gap/U  T=0  : circuit %.4e  classical %.4e  difference %.2e'
      % (np.sqrt(np.nanmean(G0 ** 2)), np.sqrt(np.nanmean(GC0 ** 2)),
         abs(np.sqrt(np.nanmean(G0 ** 2)) - np.sqrt(np.nanmean(GC0 ** 2)))))
print('    shell rms gap/U  T=0.4: circuit %.4e  classical %.4e  difference %.2e'
      % (np.sqrt(np.nanmean(G4 ** 2)), np.sqrt(np.nanmean(GC4 ** 2)),
         abs(np.sqrt(np.nanmean(G4 ** 2)) - np.sqrt(np.nanmean(GC4 ** 2)))))

import matplotlib.colors as mcolors                                # noqa: E402


def shade(rgba, Xq, Yq, Zq, light=(-0.35, -0.55, 0.76)):
    """Cheap Lambertian shading, so the shell reads as a 3-D surface."""
    P = np.stack([Xq, Yq, Zq], -1)
    nvec = np.cross(np.roll(P, -1, 0) - P, np.roll(P, -1, 1) - P)
    nvec /= np.maximum(np.linalg.norm(nvec, axis=-1, keepdims=True), 1e-12)
    Lv = np.asarray(light, float)
    Lv /= np.linalg.norm(Lv)
    f = 0.45 + 0.55 * np.abs((nvec * Lv).sum(-1))
    out = np.array(rgba, dtype=float, copy=True)
    out[..., :3] *= f[..., None]
    return np.clip(out, 0.0, 1.0)


def draw_panel(fig, where, spec):
    ttl, F, cmap, vmn, vmx, logt = spec
    ax = fig.add_subplot(*where, projection='3d')
    nm = mcolors.LogNorm(vmn, vmx) if logt else mcolors.Normalize(vmn, vmx)
    cm = plt.get_cmap(cmap)
    fc = shade(cm(nm(np.nan_to_num(F, nan=vmn))), Xs, Ys, Zs)
    ax.plot_surface(Xs, Ys, Zs, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1.85, 1.0, 1.0))
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    ax.set_title(ttl, fontsize=9, pad=0)
    smp = plt.cm.ScalarMappable(norm=nm, cmap=cm)
    fig.colorbar(smp, ax=ax, shrink=0.52, pad=-0.03, aspect=17)
    return ax


plt.rcParams['font.size'] = 9
panels = [(r'(a) theory  $|\nabla\Phi|/U_\infty$', ST, 'turbo', 0.2, 1.75, False),
          (r'(b) 18-qubit state, $T=0$:  $|\mathbf{v}|/U_\infty$', S0, 'turbo',
           0.2, 1.75, False),
          (r'(c) gap $T=0$:  $|\mathbf{v}-\mathbf{v}_{\rm th}|/U_\infty$  (rms %.1e)'
           % np.sqrt(np.nanmean(G0 ** 2)), G0, 'magma', 3e-3, 3e-1, True),
          (r'(d) gap $T=0.4$, free propagator  (rms %.1e)'
           % np.sqrt(np.nanmean(G4 ** 2)), G4, 'magma', 3e-3, 3e-1, True)]

fig = plt.figure(figsize=(11.0, 8.6))
for k, spec in enumerate(panels):
    draw_panel(fig, (2, 2, k + 1), spec)
fig.suptitle('flow over the cruciform body on an 18-qubit register  '
             r'($64^3$ field, 2$^{18}$ amplitudes)', fontsize=10.5, y=0.975)
plt.tight_layout(rect=(0, 0, 1, 0.955))
plt.savefig('code/data/circuit18q_surface.png', dpi=185)
print('figure -> code/data/circuit18q_surface.png')

# three-panel version: the question that was asked is theory vs quantum state,
# so (d) -- which answers "what the free propagator does to it", a different
# question already carried by circuit18q_3d.png and by Sec. III/V -- is dropped.
fig3 = plt.figure(figsize=(12.6, 3.9))
for k in (0, 1, 2):
    draw_panel(fig3, (1, 3, k + 1), panels[k])
fig3.suptitle('the 18-qubit register carries the theoretical flow field:  '
              r'$64^3$ field, 2$^{18}$ amplitudes', fontsize=10.5, y=0.99)
plt.tight_layout(rect=(0, 0.02, 1, 0.93))
plt.savefig('code/data/circuit18q_surface3.png', dpi=185)
print('figure -> code/data/circuit18q_surface3.png')
print()
