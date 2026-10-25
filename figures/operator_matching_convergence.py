"""
Operator-matching residual convergence on the cylinder test case (main-text figure)
===================================================================================
The discrete Laplacian fails the chain rule on the encoded phase. For the exact
eigenstate psi = e^{i Phi} the free Hamiltonian acts as

    H_free psi = -(1/2) lap psi = -(1/2) e^{i Phi} [-(grad Phi)^2 + i lap Phi],

so its residual is purely imaginary, Im[e^{-i Phi} lap_h e^{i Phi}] = lap_h Phi,
which vanishes in the continuum because lap Phi = 0. The discrete residual
therefore measures the truncation error of the stencil applied to the harmonic
phase, and it separates by the moments M4 and M22 of the stencil.

All four curves are members of one cross family, obtained as the tensor product of
a one-dimensional second-derivative stencil along each axis.

  (i) Nine-point cross, offsets 0, +-1, +-2.  With the 1D weights (a, b) at +-1,
      +-2 and centre c = -(2a + 2b), the normalisation is a + 4b = 1 and the
      moment M4 = (a + 16b)/12.  One parameter, three members:

      b =  0       five-point stencil: (sum_+-1 - 4 psi) / h^2, M4 = M22 != 0
      b = -1/18    widened but NOT moment matched: M4 = M22 != 0, still O(h^2),
                   so widening buys only a constant factor
      b = -1/12    moment matched: M4 = M22 = 0, Eq. (moment_matched_stencil) of
                   the manuscript, so the residual drops to O(h^4)

 (ii) Thirteen-point cross, offsets 0, +-1, +-2, +-3.  Two free weights, so both
      the h^4 and the h^6 terms can be cancelled at once:

          a +  4b +  9c = 1        ->  c = 1/90
          a + 16b + 81c = 0            b = -3/20
          a + 64b + 729c = 0           a = 3/2,   centre = -49/9

      i.e. 3/2 at +-1, -3/20 at +-2, 1/90 at +-3 and -49/9 at the centre.  M2 = 1
      and M4 = M6 = 0, and because a cross stencil has no point with both indices
      nonzero, M22 = M42 = M24 = 0 identically; all moment conditions through
      order 6 hold, so the residual falls as O(h^6).

The annulus 1.05 R_cyl < r < 1.5 is the same one used for the velocity-error
floor elsewhere in the paper.

Output: fig/operator_matching_convergence.pdf (overwrites the existing file).
Console: the RMS table, the per-doubling orders and the constant factors.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

U_INF, HBAR, MASS = 1.0, 1.0, 1.0
R_CYL = 0.276
L_DOM = 2.5                        # half-width of the periodic square domain
N_LIST = [64, 128, 256, 512, 1024, 2048]
R_IN, R_OUT = 1.05 * R_CYL, 1.5    # annulus used for the RMS

# N = 2048 is the top of the ladder the moment-matched curve can be trusted at.
# The stencil numerator is O(1) and is divided by h^2, so double precision puts a
# round-off floor of about 1e-16/h^2 under the residual (1.7e-11 at N = 2048,
# 6.7e-11 at N = 4096).  Measured: the moment-matched residual still falls as
# h^4 (order 4.00) up to 2048, and collapses to order 1.2 at 4096, where its
# value 4.4e-10 is within an order of magnitude of that floor.  The sixth-order
# curve is an order of magnitude lower still, so it is only drawn up to N = 1024.

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig')


# ---------------------------------------------------------------- stencils
def _off(p, k):
    """Sum of the four neighbours at distance k along the axes."""
    return (np.roll(p, -k, axis=1) + np.roll(p, k, axis=1) +
            np.roll(p, -k, axis=0) + np.roll(p, k, axis=0))


def lap_cross(b, psi, h):
    """Nine-point cross Laplacian with second-neighbour weight b.

    Edge weight a = 1 - 4b, centre -4(a + b).  b = 0 reproduces the plain
    five-point stencil; b = -1/12 gives M4 = M22 = 0.
    """
    a = 1.0 - 4.0 * b
    return (a * _off(psi, 1) + b * _off(psi, 2) - 4.0 * (a + b) * psi) / h ** 2


def lap_cross13(psi, h):
    """Sixth-order thirteen-point cross Laplacian (offsets 0, +-1, +-2, +-3).

    Edge weights 3/2, -3/20, 1/90 at +-1, +-2, +-3 and centre -49/9, so
    M2 = 1 and M4 = M6 = 0; the mixed moments vanish by construction.
    """
    return (1.5 * _off(psi, 1) - 0.15 * _off(psi, 2) + (1.0 / 90.0) * _off(psi, 3)
            - (49.0 / 9.0) * psi) / h ** 2


# label, line style, colour, operator, largest N this curve is trusted to.
# The legend leads with the stencil width and states the moment content, because
# in this family those two facts determine everything else; the resulting
# convergence order is the consequence, not the label.
STENCILS = [
    dict(short='5-pt  b=0', label='five-point', fmt='o-', color='#185FA5',
         op=lambda p, h: lap_cross(0.0, p, h), nmax=None),
    dict(short='9-pt  b=-1/18', label='nine-point',
         fmt='d--', color='#6A3D9A',
         op=lambda p, h: lap_cross(-1.0 / 18.0, p, h), nmax=None),
    dict(short='9-pt  moment-matched',
         label='nine-point, matched',
         fmt='s-', color='#C0392B',
         op=lambda p, h: lap_cross(-1.0 / 12.0, p, h), nmax=None),
    dict(short='13-pt moment-matched',
         label='thirteen-point, matched',
         fmt='^-', color='#1B7F4B', op=lap_cross13, nmax=1024),
]
GUIDE_COLOR = '#8A8A8A'          # reference slopes: neutral, so no curve colour is doubled


def annulus_window(N, h):
    """Index range covering the annulus plus a margin of the stencil radius.

    Working on this crop instead of the whole square keeps the memory at N = 2048
    and above in proportion to the annulus.  It is not an approximation: the crop
    edge is more than three cells outside R_OUT, so the periodic wrap of np.roll
    never reaches a point inside the mask, and the values are bit-identical to
    the full-square evaluation.  The margin is sized for the widest stencil (13
    points, radius 3).
    """
    lim = R_OUT + max(4.0 * h, 0.08)
    k = int(np.ceil((lim + h) / h))
    c = N // 2
    return max(0, c - k), min(N, c + k)


def residual_rms(N):
    """RMS Im[e^{-i Phi} lap_h e^{i Phi}] in R_IN < r < R_OUT for one grid."""
    h = 2.0 * L_DOM / N
    lo, hi = annulus_window(N, h)
    x = -L_DOM + np.arange(lo, hi) * h
    XX, YY = np.meshgrid(x, x)
    r = np.sqrt(XX ** 2 + YY ** 2)
    th = np.arctan2(YY, XX)
    rs = np.where(r > 1e-9, r, 1.0)
    Phi = U_INF * (rs + R_CYL ** 2 / rs) * np.cos(th)
    psi = np.exp(1j * Phi / HBAR)
    mask = (r > R_IN) & (r < R_OUT)
    return [float(np.sqrt(np.mean(np.imag(s['op'](psi, h) / psi)[mask] ** 2)))
            for s in STENCILS]


def main():
    t0 = time.time()
    print("=" * 72)
    print(" Operator-matching residual: RMS Im[e^{-i Phi} lap_h e^{i Phi}]")
    print(" cylinder test case, R_cyl = %.3f, %.2f < r < %.1f" % (R_CYL, R_IN, R_OUT))
    print("=" * 72)
    hdr = "   %6s" % 'N' + "".join("   %11s" % s['short'] for s in STENCILS)
    print(hdr)
    curves = [[] for _ in STENCILS]
    for N in N_LIST:
        vals = residual_rms(N)
        for c, v in zip(curves, vals):
            c.append(v)
        print("   %6d" % N + "".join("   %11.3e" % v for v in vals))

    print("\n   order per doubling (log2 of the successive ratios):")
    print("     %-34s %s" % ('stencil', " ".join("%7s" % ("%d->%d" % (N_LIST[i], N_LIST[i + 1]))
                                                 for i in range(len(N_LIST) - 1))))
    for s, c in zip(STENCILS, curves):
        row = []
        for i in range(len(c) - 1):
            bad = s['nmax'] is not None and N_LIST[i + 1] > s['nmax']
            row.append("%7s" % ("--" if bad else "%.2f" % np.log2(c[i] / c[i + 1])))
        print("     %-34s %s" % (s['short'], " ".join(row)))
    ratio = np.log(float(N_LIST[-1]) / N_LIST[0])
    print("\n   order over the whole range N = %d -> %d:" % (N_LIST[0], N_LIST[-1]))
    for s, c in zip(STENCILS, curves):
        last = N_LIST.index(s['nmax']) if s['nmax'] in N_LIST else len(N_LIST) - 1
        p = np.log(c[0] / c[last]) / np.log(float(N_LIST[last]) / N_LIST[0])
        print("     %-34s p = %.2f  (N = %d -> %d)" % (s['short'], p, N_LIST[0], N_LIST[last]))
    print("\n   factor vs five-point at the finest trusted N:")
    for s, c in zip(STENCILS, curves):
        last = N_LIST.index(s['nmax']) if s['nmax'] in N_LIST else len(N_LIST) - 1
        print("     %-34s %.3e   (N = %d)" % (s['short'], c[last] / curves[0][last], N_LIST[last]))

    fig, ax = plt.subplots(figsize=(6.9, 3.55), constrained_layout=True)
    Ne = np.array(N_LIST, dtype=float)
    for s, c in zip(STENCILS, curves):
        last = N_LIST.index(s['nmax']) + 1 if s['nmax'] in N_LIST else len(N_LIST)
        ax.loglog(Ne[:last], c[:last], s['fmt'], color=s['color'], lw=1.7, ms=6.5,
                  markeredgecolor='white', markeredgewidth=0.6, label=s['label'])
    # Reference slopes, drawn thin and neutral so that no data colour is doubled.
    # Each is anchored on the first point of the curve it belongs to.  The labels
    # are placed in axes fractions just below their own guide and then pushed down
    # until they are at least 0.095 apart, which keeps all three readable even
    # though the h^4 and h^6 guides nearly coincide on the left.
    guides = [(curves[0][0], 2.0, r'$p \approx 2$'),
              (curves[2][0], 4.0, r'$p \approx 4$'),
              (curves[3][0], 6.0, r'$p \approx 6$')]
    for anchor, p, _ in guides:
        ax.loglog(Ne, anchor * (Ne / Ne[0]) ** (-p), '--', color=GUIDE_COLOR, lw=1.0,
                  alpha=0.85, zorder=0)
    ylo, yhi = ax.get_ylim()
    lylo, lyhi = np.log10(ylo), np.log10(yhi)
    xlab = Ne[0] * 1.25
    trans = ax.get_xaxis_transform()
    fr = []
    for anchor, p, _ in guides:
        gy = anchor * (xlab / Ne[0]) ** (-p)
        fr.append((np.log10(gy) - lylo) / (lyhi - lylo) - 0.055)
    for i in range(1, len(fr)):
        fr[i] = min(fr[i], fr[i - 1] - 0.095)
    for (anchor, p, txt), f in zip(guides, fr):
        ax.annotate(txt, (xlab, max(min(f, 0.92), 0.02)), xycoords=trans,
                    color=GUIDE_COLOR, fontsize=12, ha='left', va='center')
    ax.set_xlabel('Grid size $N$')
    ax.set_ylabel(r'RMS Im$[e^{-i\Phi}\nabla_h^2e^{i\Phi}]$')
    ax.set_xticks([100, 1000])
    ax.set_xticklabels(['$10^2$', '$10^3$'])
    ax.minorticks_on()
    ax.grid(True, which='both', alpha=0.28, lw=0.6)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='0.6', fontsize=10,
              handlelength=2.0, borderpad=0.45, labelspacing=0.3)

    out = os.path.join(FIG_DIR, 'operator_matching_convergence.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    try:
        if out.lower().endswith('.pdf'):
            import pymupdf
            _p = pymupdf.open(out)[0].rect
            size = '%.0fx%.0f pt' % (_p.width, _p.height)
        else:
            from PIL import Image
            size = '%dx%d' % Image.open(out).size
    except Exception:
        size = 'unknown size'
    print("\n  -> saved %s  (%s)  [%.1fs]" % (out, size, time.time() - t0))


if __name__ == '__main__':
    main()
