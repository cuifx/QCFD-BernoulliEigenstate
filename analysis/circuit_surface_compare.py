# -*- coding: utf-8 -*-
"""Body-surface field comparison at a stated register size (figure only).

The full experiments (circuit_18q_3d.py, circuit_24q_3d.py) carry their own
version of this panel, but they sample the shell on a *uniform* azimuthal grid.
That grid misses AGARD-B's wing: the 4%-thick delta plate is narrower in
azimuth than the lattice spacing except within a few degrees of its own plane,
so the wing comes out as a stray dark sliver, and a reader reasonably asks where
the wing went.  Here the shell lattice uses the same density-matched grids the
panel builder uses -- clustered in azimuth at the two wing planes and in the
meridional angle over the chord -- so the plate is resolved.

Splitting this out also means the figure can be redrawn in seconds: it needs two
circuit runs (T = 0 and T = 0.4) and no re-solve.

    python circuit_surface_compare.py 64     # 18 qubits
    python circuit_surface_compare.py 256    # 24 qubits
"""
import os
import sys
import time

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import winged_body_3d_probe as W                                    # noqa: E402
import ortho_render as OR                                           # noqa: E402

W.BODY_STYLE = 'agardb'
W.NW = 2
W.FIN_AZ = 2.0 * np.pi / W.NW
AXS = W.AB_AX

from qiskit import QuantumCircuit                                   # noqa: E402
from qiskit_aer import AerSimulator                                 # noqa: E402

U = 1.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NA = int(round(np.log2(N)))
NQ = 3 * NA
L = 4.0
RS = 1.12
h = 2.0 * L / (N - 1)
gg = np.linspace(-L, L, N)
X, Y, Z = np.meshgrid(gg, gg, gg, indexing='ij')
_k1 = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
KX, KY, KZ = np.meshgrid(_k1, _k1, _k1, indexing='ij')

CACHE = ('code/data/circuit18q_agardb_cache.npz' if N == 64
         else 'code/data/wb_phi_256.npz')
OUT = 'code/data/circuit%dq_surface3.png' % (NQ)
sim = AerSimulator(method='statevector')
print('%d qubits, %d^3, h = %.5f, Phi from %s' % (NQ, N, h, CACHE))

PHI = np.load(CACHE)['PHI']
assert PHI.shape == (N, N, N)


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
VRX, VRY, VRZ = (U + _g[0]).real, _g[1].real, _g[2].real
del _g


def window3d(LL, tr=0.12):
    w = LL * tr
    win = np.ones_like(X)
    for C in (X, Y, Z):
        wc = np.ones_like(X)
        wc[C < -LL + w] = 0.5 * (1 - np.cos(np.pi * (C[C < -LL + w] + LL) / w))
        wc[C > LL - w] = 0.5 * (1 - np.cos(np.pi * (LL - C[C > LL - w]) / w))
        win *= wc
    return win


psi0 = window3d(L) * np.exp(1j * PHI)
psi0 /= np.linalg.norm(psi0)
del PHI


# ---- circuit, identical construction to the experiment files ----------------
def kinetic_coeffs(T):
    """Mobius coefficients of phi(b) = -T k(b)^2/2 (phi is quadratic in the
    bits, so the multilinear expansion is O(n^2) sparse)."""
    a = (-0.5 * T * _k1 ** 2).copy()
    idx = np.arange(N)
    for i in range(NA):
        bit = 1 << i
        sel = (idx & bit) != 0
        a[sel] -= a[idx[sel] ^ bit]
    return a


def apply_kinetic_poly(qc, reg, T, tol=1e-12):
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
            qc.cx(reg[i], reg[j]); qc.rz(-0.5 * c, reg[j]); qc.cx(reg[i], reg[j])
        else:
            raise RuntimeError('order %d' % len(bits))
    for p in range(NA):
        if abs(lin[p]) > 1e-15:
            qc.rz(float(lin[p]), reg[p])


def qft_block(qc, reg, inverse=False):
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


def run_T(T):
    qc = QuantumCircuit(NQ)
    qc.set_statevector(psi0.ravel())
    for r in REGS:
        qft_block(qc, r, inverse=False)
    for r in REGS:
        apply_kinetic_poly(qc, r, T)
    for r in REGS:
        qft_block(qc, r, inverse=True)
    qc.save_statevector()
    return np.asarray(sim.run(qc, shots=1).result().get_statevector()).reshape(N, N, N)


def decoded_v(psi):
    a, b, c_ = grad_fd(psi)
    rr = np.abs(psi) ** 2 + 1e-12
    return (np.imag(np.conj(psi) * a) / rr,
            np.imag(np.conj(psi) * b) / rr,
            np.imag(np.conj(psi) * c_) / rr)


# ---- shell lattice, clustered at the wing ----------------------------------
NPH, NTH = 240, 864             # twice the earlier 120 x 432: the shell sample
                                # is interpolated from the 256^3 volume, so
                                # refining it sharpens the flat-shaded quads
                                # without touching the circuit or the cached Phi
phe = W.phi_grid(NPH - 1)                      # meridional, clustered on the chord
tg = W.theta_grid(NTH)                         # azimuthal, clustered on the wing
th = np.concatenate([tg, [tg[0] + 2.0 * np.pi]])
PHg, THg = np.meshgrid(phe, th, indexing='ij')
xa_s = W.AB_AX * (1.0 + np.cos(PHg))            # station from the nose apex
rb_s = W._ab_body_radius(xa_s)
rw_s = W._ab_blade_extent(xa_s)
dt_s = W._ab_blade_width(np.maximum(rw_s, 1e-6))
# Stand-off of the display surface.  The body gets an *absolute* floor, not a
# fraction of the radius: a fixed fraction collapses to sub-cell stand-off at
# the nose, where the body radius vanishes, and the gridded source potential is
# then unresolved, so the speed panel picks up a grid-scale ripple there
# (measured: 0.077 rms, 0.28 peak to peak at the nose against 0.0013 mid-body).
# Two cells is the band the error metrics are measured on, so this keeps every
# displayed body point inside the validated region.  The wing keeps the
# fractional rule, because its thickness direction is azimuthal and an absolute
# radial offset would inflate its planform instead of its thickness.
DELTA = 2.0 * h
rho_s = np.maximum(rb_s + DELTA, RS * rw_s * W._comb_var(THg, dt_s))
Xs, Ys, Zs = AXS * np.cos(PHg), rho_s * np.cos(THg), rho_s * np.sin(THg)
pts = np.stack([Xs, Ys, Zs], -1)
S = (Xs, Ys, Zs)


def sample(f):
    it = RegularGridInterpolator((gg, gg, gg), f, bounds_error=False,
                                 fill_value=np.nan)
    return it(pts.reshape(-1, 3)).reshape(pts.shape[:-1])


print('shell %d x %d   wing azimuthal half width %.4f rad at the junction, '
      '%.4f mid span, %.4f at the tip   (uniform spacing would be %.4f)'
      % (NPH, NTH, W._ab_blade_width(W.AB_R0), W._ab_blade_width(0.5 * W.AB_B),
         W._ab_blade_width(W.AB_B), 2 * np.pi / NTH))

t0 = time.time()
s0 = run_T(0.0)
s4 = run_T(0.4)
print('two circuit runs: %.0f s' % (time.time() - t0))
v0, v4 = decoded_v(s0), decoded_v(s4)
del s0, s4

SPEED = lambda v: np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)        # noqa: E731
GAP = lambda v: np.sqrt((v[0] - VRX) ** 2 + (v[1] - VRY) ** 2 + (v[2] - VRZ) ** 2)  # noqa: E731
ST, S0, S4 = sample(SPEED((VRX, VRY, VRZ))), sample(SPEED(v0)), sample(SPEED(v4))
G0, G4 = sample(GAP(v0)), sample(GAP(v4))

# ---- the same field on the symmetry plane perpendicular to the wing plane ----
# Every error metric in this work, and in Section 6.2, is measured on the fluid
# region eroded by two cells, because closer to the wall than that the gridded
# representation of the source potential is not resolved.  Since the shell now
# carries an absolute floor of two cells, those are the same points, and this
# plane cut is the independent check that the shell view is honest.
IY = int(np.argmin(np.abs(gg)))


def plane_of(F):
    return F[:, IY, :]


PL_ST = plane_of(SPEED((VRX, VRY, VRZ)))
PL_S0 = plane_of(SPEED(v0))
PL_G0 = plane_of(GAP(v0))
# in-plane coordinates are (x, z); the radial coordinate is hypot(y, z) with
# y = gg[IY] held fixed, and the azimuth is arctan2(z, y), which is near
# +/- pi/2 on this plane and therefore away from the wing azimuths.
_nx, _nz = np.meshgrid(gg, gg, indexing='ij')
_phi = np.arccos(np.clip(_nx / AXS, -1.0, 1.0))
_th = np.arctan2(_nz, gg[IY] + 0.0 * _nz)
_rho = np.hypot(gg[IY], _nz)
PL_fluid = _rho > W.radius(_phi, _th)
PL_mask = binary_erosion(PL_fluid, iterations=2)
PL_x, PL_z = gg.copy(), gg.copy()
print('plane y = %+.4f : %d of %d points inside the two-cell probe band'
      % (gg[IY], PL_mask.sum(), PL_mask.size))
print('shell rms |v|/U   theory %.4f   circuit(T=0) %.4f   circuit(T=0.4) %.4f'
      % (np.sqrt(np.nanmean(ST ** 2)), np.sqrt(np.nanmean(S0 ** 2)),
         np.sqrt(np.nanmean(S4 ** 2))))
print('shell rms gap/U   T=0  %.4e        T=0.4  %.4e'
      % (np.sqrt(np.nanmean(G0 ** 2)), np.sqrt(np.nanmean(G4 ** 2))))

vC0 = decoded_v(psi0)
vC4 = decoded_v(ifftn(np.exp(-0.5j * 0.4 * (KX ** 2 + KY ** 2 + KZ ** 2)) * fftn(psi0)))
GC0, GC4 = sample(GAP(vC0)), sample(GAP(vC4))
R_ST = float(np.sqrt(np.nanmean(ST ** 2)))
R_S0 = float(np.sqrt(np.nanmean(S0 ** 2)))
R_S4 = float(np.sqrt(np.nanmean(S4 ** 2)))
R_G0 = float(np.sqrt(np.nanmean(G0 ** 2)))
R_G4 = float(np.sqrt(np.nanmean(G4 ** 2)))
R_GC0 = float(np.sqrt(np.nanmean(GC0 ** 2)))
R_GC4 = float(np.sqrt(np.nanmean(GC4 ** 2)))
print('T=0   : circuit %.4e  classical %.4e  difference %.2e'
      % (R_G0, R_GC0, abs(R_G0 - R_GC0)))
print('T=0.4 : circuit %.4e  classical %.4e  difference %.2e'
      % (R_G4, R_GC4, abs(R_G4 - R_GC4)))

# The shell fields are cached so that a figure can be redrawn in seconds without
# re-running the circuits: this file is the single place the circuit is built,
# and duplicating that construction elsewhere is how the two drift apart.
SHELL = 'code/data/circuit%dq_surface.npz' % NQ
np.savez(SHELL, Xs=Xs, Ys=Ys, Zs=Zs, ST=ST, S0=S0, S4=S4, G0=G0, G4=G4,
         GC0=GC0, GC4=GC4, rms_ST=R_ST, rms_S0=R_S0, rms_S4=R_S4,
         rms_G0=R_G0, rms_G4=R_G4, rms_GC0=R_GC0, rms_GC4=R_GC4, NQ=NQ, N=N,
         PL_ST=PL_ST, PL_S0=PL_S0, PL_G0=PL_G0, PL_mask=PL_mask,
         PL_x=PL_x, PL_z=PL_z, PL_IY=IY, PL_h=h)
print('shell fields -> %s' % SHELL)

# ---- draw ------------------------------------------------------------------
import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
import matplotlib.colors as mcolors                                  # noqa: E402
matplotlib.rcParams['pdf.fonttype'] = 42

plt.rcParams['font.size'] = 9
panels = [(r'(a) theory  $|\nabla\Phi|/U_\infty$', ST, 'turbo', 0.2, 1.75, False),
          (r'(b) %d-qubit state, $T=0$:  $|\mathbf{v}|/U_\infty$' % NQ, S0, 'turbo',
           0.2, 1.75, False),
          (r'(c) gap $T=0$:  $|\mathbf{v}-\mathbf{v}_{\rm th}|/U_\infty$  '
           r'(rms %.1e)' % np.sqrt(np.nanmean(G0 ** 2)), G0, 'magma',
           3e-3, 3e-1, True)]

fig = plt.figure(figsize=(12.6, 3.9))
ratio = None
for k, (ttl, F, cmap, vmn, vmx, logt) in enumerate(panels):
    ax = fig.add_subplot(1, 3, k + 1)
    nm = mcolors.LogNorm(vmn, vmx) if logt else mcolors.Normalize(vmn, vmx)
    # the norm goes to draw_surface as well as to the colorbar: on its own it
    # would scale only the bar and leave every face in the first colour of the
    # map, which is how the gap panel came out uniformly black
    ratio = OR.draw_surface(ax, S, 40.0, -90.0, val=F, cmap=cmap,
                            vmin=vmn, vmax=vmx, contrast=0.30, norm=nm)
    ax.set_title(ttl, fontsize=8.5)
    smp = plt.cm.ScalarMappable(norm=nm, cmap=plt.get_cmap(cmap))
    fig.colorbar(smp, ax=ax, shrink=0.72, pad=0.01, aspect=16)
fig.suptitle('the %d-qubit register carries the theoretical flow field:  '
             r'AGARD-B, $%d^3$ field, 2$^{%d}$ amplitudes'
             % (NQ, N, NQ), fontsize=10.5, y=0.99)
plt.tight_layout(rect=(0, 0.02, 1, 0.93))
plt.savefig(OUT, dpi=185)
print('figure -> %s   (screen isotropy %.6f)' % (OUT, ratio))
