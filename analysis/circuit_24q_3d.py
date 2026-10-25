# -*- coding: utf-8 -*-
"""24-qubit experiment: the three-dimensional stationary encoding on a 256^3 grid.

The 18-qubit run (circuit_18q_3d.py) established the circuit against the
classical spectral step at 64^3.  This file repeats the *same* five checks one
size up: 256^3 needs 3*log2(256) = 24 qubits and 2^24 = 16 777 216 amplitudes.

Register map.  The flattened field index is n = i*256^2 + j*256 + k (x, y, z),
and qubit p carries bit p of n, so q0-q7 = z, q8-q15 = y, q16-q23 = x
(Qiskit little-endian).

What changed relative to the 18-qubit file
------------------------------------------
Only the kinetic phase polynomial.  At 18 qubits it was written from a
hand-expanded (y - 64 b5)^2; at 24 qubits the same expansion in 8 bits is
neither short nor safe to hand-expand, so the coefficients are obtained
numerically as the Mobius (multilinear) transform of phi(b) = -T k(b)^2 / 2.
That is legitimate for exactly the reason the hand expansion worked: k(b) is
*linear* in the register bits once the top bit is used to fold the negative
frequencies, so phi is a quadratic form and its multilinear expansion has
terms of order <= 2 only -- the Walsh spectrum stays O(n^2) sparse, never 2^24.
Check [1] verifies the synthesized diagonal against the exact one.

Everything else (QFT blocks, checks, encodings) is the 18-qubit code path with
the qubit indices widened.

Disclosed simulator shortcuts (unchanged from 18 qubits)
--------------------------------------------------------
  * the encoded state exp(i Phi) is installed with set_statevector; the
    2^24-wide phase-encoding diagonal is the thing Sec. VII's arithmetic
    oracle exists to synthesize, and is not synthesized here;
  * likewise exp(-i V dt/2) for the H_B branch.

Phi is read from the cache produced by build_phi_256.py (AGARD-B geometry,
11904 panels) -- 91.6 min of classical pre-processing, versus minutes of
circuit time.  That asymmetry is itself the Sec. VII point.
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
# time, so it has to be reset alongside NW.
W.BODY_STYLE = 'agardb'
W.NW = 2
W.FIN_AZ = 2.0 * np.pi / W.NW
AXS = W.AB_AX if W.BODY_STYLE == 'agardb' else W.AX

from qiskit import QuantumCircuit, transpile                        # noqa: E402
from qiskit_aer import AerSimulator                                 # noqa: E402

U = 1.0
NA = 8                      # bits per axis
NQ = 3 * NA                 # 24 qubits
N = 1 << NA                 # 256 cells per axis
L = 4.0
PHI_CACHE = 'code/data/wb_phi_256.npz'
CACHE_V = 'code/data/circuit24q_cache.npz'

h = 2.0 * L / (N - 1)
gg = np.linspace(-L, L, N)
X, Y, Z = np.meshgrid(gg, gg, gg, indexing='ij')
_k1 = np.fft.fftfreq(N, d=h) * 2.0 * np.pi          # k(b) is this, indexed by y
KX, KY, KZ = np.meshgrid(_k1, _k1, _k1, indexing='ij')
K2 = KX ** 2 + KY ** 2 + KZ ** 2

sim = AerSimulator(method='statevector')
print('register: %d^3 field, %d qubits, %d amplitudes, h = %.5f'
      % (N, NQ, 2 ** NQ, h))


# ---------------------------------------------------------------- set-up
def load_phi():
    if not os.path.exists(PHI_CACHE):
        raise SystemExit('missing %s -- run build_phi_256.py first' % PHI_CACHE)
    d = np.load(PHI_CACHE)
    PHI = d['PHI']
    assert PHI.shape == (N, N, N), PHI.shape
    print('loaded Phi from %s  (finite %s, range %.4f .. %.4f)'
          % (PHI_CACHE, np.isfinite(PHI).all(), PHI.min(), PHI.max()))
    return PHI


PHI = load_phi()
PHIp = np.arccos(np.clip(X / AXS, -1.0, 1.0))
THp = np.arctan2(Z, Y)
RPp = np.hypot(Y, Z)
fluid = RPp > W.radius(PHIp, THp)
probe = binary_erosion(fluid, iterations=3) & (np.sqrt(X ** 2 + Y ** 2 + Z ** 2) < 3.0)
del fluid, PHIp, THp, RPp


def window3d(LL, tr=0.12):
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
    return (ifftn(1j * KX * fftn(f)), ifftn(1j * KY * fftn(f)),
            ifftn(1j * KZ * fftn(f)))


def grad_fd(f):
    gx = np.zeros_like(f); gy = np.zeros_like(f); gz = np.zeros_like(f)
    gx[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    gz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * h)
    return gx, gy, gz


_g = grad_spec(PHI - U * X)
# grad_spec returns a complex array; keep only the real part, or every
# downstream array inherits the complex dtype.
VRX, VRY, VRZ = (U + _g[0]).real, _g[1].real, _g[2].real
del _g
WIND = window3d(L)
psi0 = WIND * np.exp(1j * PHI)
psi0 = psi0 / np.linalg.norm(psi0)
print('probe %d points   freestream %.1f   psi0 norm %.15f'
      % (probe.sum(), U, np.linalg.norm(psi0)))


# ---------------------------------------------------------------- circuit parts
def kinetic_coeffs(T):
    """Mobius coefficients of phi(b) = -T k(b)^2 / 2 on one axis register.

    k(b) = (2 pi / (N h)) * (y - N b_{n-1}) with y = sum_p 2^p b_p, so phi is a
    quadratic form in the bits and the multilinear transform is exact and
    sparse.  In-place subset transform: for each bit i, subtract the partner
    with that bit cleared.
    """
    a = (-0.5 * T * _k1 ** 2).copy()
    idx = np.arange(N)
    for i in range(NA):
        bit = 1 << i
        sel = (idx & bit) != 0
        a[sel] -= a[idx[sel] ^ bit]
    return a


def apply_kinetic_poly(qc, reg, T, tol=1e-12):
    """Gate-level diagonal exp(-i T k^2 / 2) on one 8-qubit Fourier register.

    RZ(theta) contributes exp(i theta b), and CNOT-RZ-CNOT on (i, j) gives
    exp(i theta (b_i + b_j - 2 b_i b_j)), so a quadratic coefficient c is
    emitted with theta = -c/2 and the two spurious linear halves are credited
    back to the linear accumulator.
    """
    a = kinetic_coeffs(T)
    scale = max(float(np.abs(a).max()), 1e-30)
    lin = np.zeros(NA)
    for b in range(N):
        c = float(a[b])
        if abs(c) < tol * scale:
            continue
        bits = [i for i in range(NA) if (b >> i) & 1]
        if len(bits) == 1:
            lin[bits[0]] += c
        elif len(bits) == 2:
            i, j = bits
            lin[i] += 0.5 * c
            lin[j] += 0.5 * c
            qc.cx(reg[i], reg[j])
            qc.rz(-0.5 * c, reg[j])
            qc.cx(reg[i], reg[j])
        else:
            raise RuntimeError('multilinear order %d: phi is not quadratic'
                               % len(bits))
    for p in range(NA):
        if abs(lin[p]) > 1e-15:
            qc.rz(float(lin[p]), reg[p])


def qft_block(qc, reg, inverse=False):
    """Textbook QFT: |x> -> N^-1/2 sum_y exp(2 pi i x y / N) |y>, reg[0] being
    the least significant bit.  Aer does not accept the ``qft`` custom
    instruction, so it is written out.  The cascade runs MSB-first, which is the
    ordering that reproduces the natural-index DFT; check [0] verifies it.
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


REGS = [list(range(0, NA)), list(range(NA, 2 * NA)), list(range(2 * NA, 3 * NA))]


def build_free_circuit(T):
    qc = QuantumCircuit(NQ)
    qc.set_statevector(psi0.ravel())
    for r in REGS:
        qft_block(qc, r, inverse=False)
    for r in REGS:
        apply_kinetic_poly(qc, r, T)
    for r in REGS:
        qft_block(qc, r, inverse=True)
    qc.save_statevector()
    return qc


def run(qc):
    return np.asarray(sim.run(qc, shots=1).result().get_statevector())


# ---------------------------------------------------------------- check 0
print()
print('[0] QFT convention: the full 8-qubit circuit vs the exact 256 x 256 DFT')
from qiskit.quantum_info import Operator                            # noqa: E402
_x = np.arange(N)
_ref = np.exp(2j * np.pi * np.outer(_x, _x) / N) / np.sqrt(N)
_qc = QuantumCircuit(NA)
qft_block(_qc, list(range(NA)), inverse=False)
print('    max|U - DFT|         = %.3e' % np.max(np.abs(Operator(_qc).data - _ref)))
_qc = QuantumCircuit(NA)
qft_block(_qc, list(range(NA)), inverse=True)
print('    max|U_inv - DFT^dag|  = %.3e' % np.max(np.abs(Operator(_qc).data - _ref.conj().T)))
del _ref, _x, _qc

# ---------------------------------------------------------------- check 1
print()
print('[1] 8-qubit kinetic phase polynomial vs the exact diagonal')
rng = np.random.default_rng(11)
reg = list(range(NA))
for T in (0.04, 0.2):
    qc = QuantumCircuit(NA)
    inp = rng.normal(size=N) + 1j * rng.normal(size=N)
    inp /= np.linalg.norm(inp)
    qc.set_statevector(inp)
    apply_kinetic_poly(qc, reg, T)
    qc.save_statevector()
    out = run(qc)
    ref = np.exp(-0.5j * T * _k1 ** 2) * inp
    ov = abs(np.vdot(ref, out)) / (np.linalg.norm(ref) * np.linalg.norm(out))
    print('    T = %.2f   |<ref|circ>| = %.15f   1-fidelity = %.2e'
          % (T, ov, 1 - ov))

# ---------------------------------------------------------------- check 2
print()
print('[2] full 24-qubit circuit vs the classical spectral step')
for T in (0.04, 0.2):
    out = run(build_free_circuit(T))
    ref = ifftn(np.exp(-0.5j * T * K2) * fftn(psi0))
    ov = abs(np.vdot(ref.ravel(), out)) / (np.linalg.norm(ref) * np.linalg.norm(out))
    print('    T = %.2f   |<ref|circ>| = %.15f   1-fidelity = %.2e'
          % (T, ov, 1 - ov))
qc = build_free_circuit(0.04)
ops = dict(qc.count_ops())
NGATE, NDEPTH = sum(ops.values()), qc.depth()
print('    circuit size (T = 0.04): %s' % ops)
print('    total ops = %d   depth = %d' % (NGATE, NDEPTH))
# transpile() is only a sanity check here; at 24 qubits it is expensive and the
# circuit is already in elementary gates (h, cp, swap, cx, rz), so a failure is
# not fatal.
try:
    tqc = transpile(qc, sim, optimization_level=1)
    print('    transpiled: %s' % dict(tqc.count_ops()))
    del tqc
except Exception as e:                                              # noqa: BLE001
    print('    (transpile skipped/failed: %s)' % e)
del qc

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
t3 = time.time()
for T in (0.005, 0.01, 0.02, 0.04, 0.10, 0.20, 0.40):
    out = run(build_free_circuit(T)).reshape(N, N, N)
    ref = ifftn(np.exp(-0.5j * T * K2) * fftn(psi0))
    dif = 1.0 - abs(np.vdot(ref.ravel(), out.ravel()))
    rows.append((T, vel_err(out), vel_err(ref), dif))
    print('    %-6.3f %-13.4e %-13.4e %.2e' % rows[-1])
print('    (%.0f s for the seven T points)' % (time.time() - t3))

# ---------------------------------------------------------------- check 4
print()
print('[4] exact end-to-end at 24 qubits, NO simulator shortcut: uniform flow')
m_wave = 2
UU = 2.0 * np.pi * m_wave / (N * h)
dx = gg[1] - gg[0]
print('    encoding is gate-level (24 H + 8 RZ, the circuit of Sec. VII), and')
print('    exp(i U x) is an exact eigenstate of the free propagator')
qc = QuantumCircuit(NQ)
for q_ in range(NQ):
    qc.h(q_)
for p in range(NA):
    qc.rz(UU * dx * 2 ** p, 2 * NA + p)          # only the x register carries U x
for r in REGS:
    qft_block(qc, r, inverse=False)
for r in REGS:
    apply_kinetic_poly(qc, r, 0.04)
for r in REGS:
    qft_block(qc, r, inverse=True)
qc.save_statevector()
psiA = np.exp(1j * UU * gg[:, None, None] * np.ones((1, N, N)))
psiA = psiA / np.linalg.norm(psiA)
outA = run(qc)
ov = np.vdot(psiA.ravel(), outA)
print('    |<psi(0)|psi(T)>|  = %.15f      (1 - |.| = %.2e)' % (abs(ov), 1 - abs(ov)))


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
print('      the residual is the centred-difference decode sin(U h)/h = %.6f'
      % (np.sin(UU * h) / h))
print('      against U = %.6f -- the same O(h^2) decode artifact that sets the' % UU)
print('      t=0 floor, not a circuit error.')

np.savez('code/data/circuit24q_result.npz',
         T=np.array([r[0] for r in rows]),
         circ=np.array([r[1] for r in rows]),
         clas=np.array([r[2] for r in rows]),
         infid=np.array([r[3] for r in rows]),
         floor=floor, gates=NGATE, depth=NDEPTH)

# ---------------------------------------------------------------- figure
import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
matplotlib.rcParams['pdf.fonttype'] = 42

Tv = np.array([r[0] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
ax[0].loglog(Tv, [r[1] for r in rows], 'o', ms=5, color='#d63031',
             label='24-qubit circuit', mfc='white', mew=1.2)
ax[0].loglog(Tv, [r[2] for r in rows], '-', lw=1.2, color='#0984e3',
             label='classical spectral step')
ax[0].axhline(floor, color='#636e72', ls=':', lw=1.2)
ax[0].text(Tv[0], floor * 1.15, r'$t{=}0$ decoding floor', fontsize=7,
           color='#636e72')
ax[0].set_xlabel(r'evolution time $T$')
ax[0].set_ylabel(r'$\|v-v_{\rm ref}\|_{\rm rms}/U_\infty$')
ax[0].set_title(r'free propagator, $256^3$ field on 24 qubits', fontsize=8)
ax[0].legend(fontsize=7, loc='upper left')

ax[1].semilogy(Tv, np.maximum([abs(r[3]) for r in rows], 1e-17), 's-', ms=4,
               lw=1.2, color='#00b894')
ax[1].axhline(2.2e-16, color='#636e72', ls=':', lw=1.2)
ax[1].set_xlabel(r'evolution time $T$')
ax[1].set_ylabel(r'$1-|\langle\psi_{\rm class}|\psi_{\rm circ}\rangle|$')
ax[1].set_title('circuit vs classical, %d gates / depth %d' % (NGATE, NDEPTH),
                fontsize=8)
ax[1].set_ylim(1e-17, 1e-13)
for a in ax:
    a.grid(True, which='both', alpha=0.25, lw=0.6)
    a.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig('code/data/circuit24q_3d.png', dpi=190)
print('figure -> code/data/circuit24q_3d.png')

# ---------------------------------------------------------------- check 5
print()
print('[5] body-surface comparison: theoretical flow vs the 24-qubit state')
from scipy.interpolate import RegularGridInterpolator                # noqa: E402


def decoded_v(psi):
    a, b, c_ = grad_fd(psi)
    rr = np.abs(psi) ** 2 + 1e-12
    return (np.imag(np.conj(psi) * a) / rr,
            np.imag(np.conj(psi) * b) / rr,
            np.imag(np.conj(psi) * c_) / rr)


RS = 1.12
n_phi, n_th = 72, 264
PHg, THg = np.meshgrid(np.linspace(0.02, np.pi - 0.02, n_phi),
                       np.linspace(0.0, 2.0 * np.pi, n_th), indexing='ij')
rho_s = RS * W.radius(PHg, THg)
Xs, Ys, Zs = AXS * np.cos(PHg), rho_s * np.cos(THg), rho_s * np.sin(THg)
pts = np.stack([Xs, Ys, Zs], -1)


def sample(f):
    it = RegularGridInterpolator((gg, gg, gg), f, bounds_error=False,
                                 fill_value=np.nan)
    return it(pts.reshape(-1, 3)).reshape(pts.shape[:-1])


s0 = run(build_free_circuit(0.0)).reshape(N, N, N)
s4 = run(build_free_circuit(0.4)).reshape(N, N, N)
v0, v4 = decoded_v(s0), decoded_v(s4)
SPEED = lambda v: np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)        # noqa: E731
GAP = lambda v: np.sqrt((v[0] - VRX) ** 2 + (v[1] - VRY) ** 2 + (v[2] - VRZ) ** 2)  # noqa: E731
ST, S0, S4 = sample(SPEED((VRX, VRY, VRZ))), sample(SPEED(v0)), sample(SPEED(v4))
G0, G4 = sample(GAP(v0)), sample(GAP(v4))
print('    shell rms |v|/U   theory %.4f   circuit(T=0) %.4f   circuit(T=0.4) %.4f'
      % (np.sqrt(np.nanmean(ST ** 2)), np.sqrt(np.nanmean(S0 ** 2)),
         np.sqrt(np.nanmean(S4 ** 2))))
print('    shell rms gap/U   T=0  %.3e        T=0.4  %.3e'
      % (np.sqrt(np.nanmean(G0 ** 2)), np.sqrt(np.nanmean(G4 ** 2))))
vC0 = decoded_v(psi0)
vC4 = decoded_v(ifftn(np.exp(-0.5j * 0.4 * K2) * fftn(psi0)))
GC0, GC4 = sample(GAP(vC0)), sample(GAP(vC4))
print('    T=0   : circuit %.4e  classical %.4e  difference %.2e'
      % (np.sqrt(np.nanmean(G0 ** 2)), np.sqrt(np.nanmean(GC0 ** 2)),
         abs(np.sqrt(np.nanmean(G0 ** 2)) - np.sqrt(np.nanmean(GC0 ** 2)))))
print('    T=0.4 : circuit %.4e  classical %.4e  difference %.2e'
      % (np.sqrt(np.nanmean(G4 ** 2)), np.sqrt(np.nanmean(GC4 ** 2)),
         abs(np.sqrt(np.nanmean(G4 ** 2)) - np.sqrt(np.nanmean(GC4 ** 2)))))

import matplotlib.colors as mcolors                                # noqa: E402


def shade(rgba, Xq, Yq, Zq, light=(-0.35, -0.55, 0.76)):
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
          (r'(b) 24-qubit state, $T=0$:  $|\mathbf{v}|/U_\infty$', S0, 'turbo',
           0.2, 1.75, False),
          (r'(c) gap $T=0$:  $|\mathbf{v}-\mathbf{v}_{\rm th}|/U_\infty$  (rms %.1e)'
           % np.sqrt(np.nanmean(G0 ** 2)), G0, 'magma', 3e-3, 3e-1, True)]

fig3 = plt.figure(figsize=(12.6, 3.9))
for k, spec in enumerate(panels):
    draw_panel(fig3, (1, 3, k + 1), spec)
fig3.suptitle('the 24-qubit register carries the theoretical flow field:  '
              r'AGARD-B, $256^3$ field, 2$^{24}$ amplitudes', fontsize=10.5, y=0.99)
plt.tight_layout(rect=(0, 0.02, 1, 0.93))
plt.savefig('code/data/circuit24q_surface3.png', dpi=185)
print('figure -> code/data/circuit24q_surface3.png')
print()
