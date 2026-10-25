# -*- coding: utf-8 -*-
"""Build Phi on the 256^3 grid (the 24-qubit register) and cache it.

Why this file exists in this shape
----------------------------------
1.  Cost.  The straightforward centroid-point-sum evaluator runs at ~1600
    field points/s on the swept-fin geometry (9936 panels), i.e. ~2.9 h for
    2^24 points.  The evaluator here accumulates the same sum but forms the
    distances through a matrix product, |p-c|^2 = |p|^2 + |c|^2 - 2 p.c,
    which removes the (chunk, M, 3) temporary and is several times faster.
    It is checked against the original evaluator before any long run.

2.  Resumability.  A detached ``nohup ... &`` job is killed when the agent
    turn that spawned it ends -- that already cost one full run.  So the
    output lives in a memmap that is written as the sweep advances, together
    with a completed-chunk counter; re-running the script picks up where it
    stopped instead of starting over.

Usage
-----
    python build_phi_256.py verify      # accuracy check vs the old evaluator
    python build_phi_256.py             # (re)start / resume the sweep
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import winged_body_3d_probe as W                                    # noqa: E402

# Geometry: the AGARD-B calibration model, at the panel set the geometry audit
# verified (32 x 192).  FIN_AZ is derived from NW at import time, so it has to
# be reset with NW; leaving it at the cruciform value builds four fins.
W.BODY_STYLE = 'agardb'
W.NW = 2
W.FIN_AZ = 2.0 * np.pi / W.NW

N, L = 256, 4.0
NPTS = N ** 3
CHUNK = 2048                       # (CHUNK x 9936) doubles ~= 163 MB per matmul
DATA = 'code/data'
PHI_PATH = os.path.join(DATA, 'wb_phi_256.dat')      # memmap, resumable
CNT_PATH = os.path.join(DATA, 'wb_phi_256.count')    # completed chunks


def build_solver(nf=32, nt=192):
    cen, nrm, area, tri = W.build_panels(nf, nt)
    sig, Aop, rhs = W.solve_sources(cen, nrm, area, tri)
    q = sig * area
    return cen, q, dict(panels=len(cen), Q=float(q.sum()),
                        bc=float(np.abs(Aop @ sig - rhs).max()))


def phi_fast(blk, cen, q, c2):
    """sum_j q_j / |p_i - c_j| , the 1/(4 pi) factor left to the caller."""
    a = blk @ cen.T                                   # |p||c|cos(angle)
    r2 = (blk * blk).sum(1)[:, None] + c2[None, :] - 2.0 * a
    np.maximum(r2, 1e-24, out=r2)
    return (1.0 / np.sqrt(r2)) @ q


def verify(nsub=20000, seed=3):
    """phi_fast against W.potential on a subsample, near field included."""
    cen, q, info = build_solver()
    rng = np.random.default_rng(seed)
    # half the sample in the far field, half within 1.5 of the body
    far = rng.uniform(-L, L, size=(nsub // 2, 3))
    near = rng.normal(0.0, 1.0, size=(nsub - nsub // 2, 3))
    near *= (rng.uniform(0.3, 1.5, size=(len(near), 1))
             / np.linalg.norm(near, axis=1, keepdims=True))
    P = np.vstack([far, near])
    P = np.vstack([P, np.array([[2.5, 0.0, 0.0], [0.0, 1.4, 0.0],
                                [0.0, 0.0, 1.4], [3.9, 0.0, 0.0]])])

    # reference: the original per-chunk diffform evaluator
    ref = np.empty(len(P))
    for c0 in range(0, len(P), 2000):
        blk = P[c0:c0 + 2000]
        d = blk[:, None, :] - cen[None, :, :]
        dist = np.maximum(np.sqrt((d ** 2).sum(-1)), 1e-9)
        ref[c0:c0 + 2000] = (q[None, :] / dist).sum(axis=1) / (4.0 * np.pi)
    ref += P[:, 0] * W.U

    c2 = (cen * cen).sum(1)
    got = phi_fast(P, cen, q, c2) / (4.0 * np.pi) + P[:, 0] * W.U
    dphi = np.abs(got - ref)
    print('panels %d   Q = %+.3e   BC residual %.1e' % (info['panels'], info['Q'], info['bc']))
    print('verify: %d points (incl. the four named probes)' % len(P))
    print('        max |dPhi| = %.3e     rms |dPhi| = %.3e'
          % (dphi.max(), np.sqrt((dphi ** 2).mean())))
    print('        scale |Phi| ~ %.3f  -> relative %.2e'
          % (np.abs(ref).mean(), dphi.max() / np.abs(ref).mean()))
    print('        probes: ' + '  '.join('%+.4f/%+.4f' % (g, r)
          for g, r in zip(got[-4:], ref[-4:])))
    ok = dphi.max() < 1e-6
    print('        VERDICT:', 'PASS (<< 1e-6, far below the encoding budget)'
          if ok else 'FAIL -- do not use phi_fast')
    return ok


def sweep():
    cen, q, info = build_solver()
    print('panels %d   Q = %+.3e   BC residual %.1e'
          % (info['panels'], info['Q'], info['bc']), flush=True)
    c2 = (cen * cen).sum(1)

    nchunk = (NPTS + CHUNK - 1) // CHUNK
    if os.path.exists(PHI_PATH):
        out = np.memmap(PHI_PATH, dtype=np.float64, mode='r+', shape=(NPTS,))
        done = int(open(CNT_PATH).read()) if os.path.exists(CNT_PATH) else 0
        print('resuming at chunk %d / %d  (%.1f%% already on disk)'
              % (done, nchunk, 100.0 * done / nchunk), flush=True)
    else:
        out = np.memmap(PHI_PATH, dtype=np.float64, mode='w+', shape=(NPTS,))
        done = 0

    gg = np.linspace(-L, L, N)
    # generator over grid points, so the 256^3 coordinate arrays are never
    # all materialised at once
    t0 = time.time()
    for c in range(done, nchunk):
        i0, i1 = c * CHUNK, min((c + 1) * CHUNK, NPTS)
        idx = np.arange(i0, i1)
        ix, rem = np.divmod(idx, N * N)
        iy, iz = np.divmod(rem, N)
        blk = np.stack([gg[ix], gg[iy], gg[iz]], axis=1)
        out[i0:i1] = phi_fast(blk, cen, q, c2)
        if c % 20 == 19 or c == nchunk - 1:
            np.memmap.flush(out)
            with open(CNT_PATH, 'w') as f:
                f.write(str(c + 1))
            el = time.time() - t0
            rate = (i1 - done * CHUNK) / max(el, 1e-9)
            print('  %5.1f%%   %6.0f s   (%.0f pts/s)   eta %.0f min'
                  % (100.0 * i1 / NPTS, el, rate,
                     (NPTS - i1) / max(rate, 1e-9) / 60), flush=True)

    # finaliser: add the free-stream term and fold into the canonical npz
    PHI = np.asarray(out).reshape(N, N, N)
    PHI = PHI / (4.0 * np.pi)
    PHI = PHI + gg[:, None, None] * W.U
    np.savez('code/data/wb_phi_256.npz', PHI=PHI.astype(np.float64), N=N, L=L,
             U=W.U)
    print('cached -> code/data/wb_phi_256.npz   total %.1f min'
          % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        sys.exit(0 if verify() else 1)
    sweep()
