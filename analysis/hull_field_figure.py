# -*- coding: utf-8 -*-
"""Hull flow field at 24 qubits, as the Section 7.6 figure.

The section claims the 24-qubit register carries the encoded field, and until
now it showed two line plots to support that.  These panels show the field
itself, together with the only thing that distinguishes the register from the
classical step: their difference.

Two views of the same cached fields, selected by flag:

  * default, the wetted shell.  What the paper uses.  The body carries an
    absolute stand-off of two grid cells -- the band the error metrics of
    Section 6.2 are measured on -- and the wing keeps the 1.12 fractional rule,
    because its thickness direction is azimuthal and an absolute offset would
    inflate the planform instead of the thickness.  Until 2026-10-08 the
    stand-off was the fraction alone, which at the nose (body radius -> 0)
    collapsed below one grid cell and put a grid-scale ripple of 0.26 peak to
    peak in |v|/U on the speed panel; the absolute floor cuts that to 0.046.
  * `--plane`, the symmetry plane y = 0 with the two-cell band masked out, so
    every displayed point is one the metrics cover.  It removes the nose ripple
    from the speed panel but the residual panel then shows the grid-scale
    structure of the finite-difference decode against the spectral reference,
    and the winged hull is no longer legible.  Kept as the alternative, not
    used in the paper.

Shell fields come from `circuit_surface_compare.py`, which owns the circuit
construction and caches to code/data/circuit<NQ>q_surface.npz, so a redraw costs
seconds and the circuit is built in exactly one place.

Three choices are made from measurements and printed on every run:

  * the view.  The projected field is 2.98:1 at elev 40 / azim -90 (extents
    4.250 x 1.424), so the panels are sized to that; a square panel would spend
    two thirds of its height on white space.
  * the speed window.  AGARD-B at zero incidence is nearly axisymmetric -- the
    wing carries no load -- so the wing and mid-body span only 1.00-1.05 U and
    the nose spans 0.91-1.03, while the extremes sit at the aft end: the base
    cap dips to 0.64 and the shoulder ahead of it reaches 1.16.  A full range
    leaves the wing one flat colour.  The window is set to 0.83-1.10 and a
    piecewise norm stretches the neighbourhood of 1.0 to bring out the wing and
    body detail, so only the aft end saturates.  A log norm does nothing here:
    even at this width the mapping is nearly linear (the wing band lands at
    0.108 of the colorbar against 0.107 for linear).
  * the gap window, which needs three decades of log: the median is 2.3e-3 and
    the 99th percentile 0.24 on the shell, and the plane sets it from its own
    measured percentiles.

    python hull_field_figure.py --preview    # side-by-side PNG for review
    python hull_field_figure.py              # fig/hull_field_a/b.pdf (paper)
    python hull_field_figure.py --three      # one 3-panel strip, the variant
                                             # not used in the paper
    python hull_field_figure.py --plane      # the validated-band view, also
                                             # not used in the paper
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ortho_render as OR                                             # noqa: E402
import winged_body_3d_probe as W                                      # noqa: E402

W.BODY_STYLE = 'agardb'
W.NW = 2
W.FIN_AZ = 2.0 * np.pi / W.NW
AXS = W.AB_AX

import matplotlib                                                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                       # noqa: E402
import matplotlib.colors as mcolors                                   # noqa: E402

# match the other three-dimensional panels (fig/3d_nonaxisym_*.pdf): Times body
# text, Computer Modern math, TrueType-embedded vector PDF
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

SHELL = 'code/data/circuit24q_surface.npz'
DPR = 600.0                     # dpi of the embedded raster field layers
ELEV, AZIM = 40.0, -90.0
VLO, VHI = 0.83, 1.10           # speed window: the shell spans 0.64 (base cap)
                                # to 1.16 (shoulder ahead of it) with the nose
                                # at 0.91-1.03, so only the aft end saturates
GLO, GHI = 3e-4, 5e-1           # shell gap window, log, three decades
PANEL_FS = (5.4, 2.25)          # house panel width; 5.4/2.98 = 1.81 in of field,
                                # plus the colorbar and its labels.  Placed at
                                # 0.42\columnwidth it scales by 0.52, the same
                                # as fig/3d_nonaxisym_*.pdf, so the type matches
XLO, XHI = -2.75, 2.75          # plane view crop, in body diameters
ZLO, ZHI = -1.35, 1.35
PLANE_FS = (5.4, 2.70)


def speed_norm():
    """Piecewise speed norm: the two end bands are compressed and 0.99-1.03 is
    stretched to half the colorbar, which is where the wing and mid-body speed
    actually lives.  See the module docstring for why a log norm is not used.
    """
    fwd = lambda v: np.interp(v, [VLO, 0.99, 1.03, VHI], [0.0, 0.25, 0.75, 1.0])
    inv = lambda t: np.interp(t, [0.0, 0.25, 0.75, 1.0], [VLO, 0.99, 1.03, VHI])
    return mcolors.FuncNorm((fwd, inv), VLO, VHI)


def spec_shell(key, clab):
    return dict(val_key=key, cmap='turbo', vmin=VLO, vmax=VHI, log=False,
                clab=clab, norm=speed_norm())


SHELL_SPECS = {
    'theory': spec_shell('ST', r'$|\nabla\Phi|/U_\infty$'),
    '24q': spec_shell('S0', r'$|\mathbf{v}|/U_\infty$'),
    'gap': dict(val_key='G0', cmap='magma', vmin=GLO, vmax=GHI, log=True,
                clab=r'$|\mathbf{v}-\mathbf{v}_{\rm th}|/U_\infty$'),
}


def check_pdf_text_inside(path, tol=0.6):
    """Assert, on the saved PDF, that no text span touches the page edge.

    The check is made on the artifact rather than on the canvas because the
    panel is drawn with `bbox_inches='tight'`, which grows the page to fit; the
    canvas coordinates at draw time therefore say nothing about the result.  A
    colorbar's tick labels are drawn outside the colorbar axes, so they are
    exactly what goes missing when a page is too small -- and a clipped axis
    label is invisible in a thumbnail and obvious to a referee.
    """
    import pymupdf
    d = pymupdf.open(path)
    pg = d[0]
    R = pg.rect
    bad, n, skipped = [], 0, 0
    for blk in pg.get_text('dict')['blocks']:
        for ln in blk.get('lines', []):
            # rotated text (a colorbar's vertical unit label) gets an inflated
            # horizontal-advance bbox from the extractor and would report a
            # false positive; its direction vector gives it away
            if tuple(round(v) for v in ln.get('dir', (1, 0))) != (1, 0):
                skipped += len(ln['spans'])
                continue
            for sp in ln['spans']:
                n += 1
                x0, y0, x1, y1 = sp['bbox']
                if (x0 < R.x0 - tol or x1 > R.x1 + tol
                        or y0 < R.y0 - tol or y1 > R.y1 + tol):
                    bad.append((sp['text'], [round(v, 1) for v in sp['bbox']]))
    print('  [%s] %.1f x %.1f pt page, %d horizontal spans (%d rotated skipped),'
          ' %d at/over the edge'
          % (os.path.basename(path), R.width, R.height, n, skipped, len(bad)))
    for b in bad[:8]:
        print('    EDGE %r %s' % b)
    d.close()
    return not bad


def colorbar(ax, norm, spec):
    smp = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(spec['cmap']))
    # `shrink` matters: the bar's length is tied to the parent's height, which
    # is itself tied to the equal-aspect box, so with a full-length bar the
    # topmost tick label is drawn centred on the top edge and half of it lands
    # outside the canvas -- a circular constraint constrained layout cannot
    # resolve on its own.  `aspect` is deliberately not passed: it would set the
    # length from the width instead and make the bar taller than the figure.
    cb = ax.figure.colorbar(smp, ax=ax, fraction=0.045, pad=0.02, shrink=0.90)
    if not spec['log']:
        # the piecewise norm would otherwise place ticks by its own spacing;
        # hand the reader four round values instead, unevenly spaced on the bar
        cb.set_ticks([VLO, 1.00, 1.05, VHI])
    cb.ax.tick_params(labelsize=9, length=2.2, width=0.5)
    cb.set_label(spec['clab'], fontsize=12, labelpad=2.0)
    cb.outline.set_linewidth(0.4)


# ------------------------------------------------------------------ shell view
def draw_shell(ax, S, spec, d):
    norm = (mcolors.LogNorm(spec['vmin'], spec['vmax']) if spec['log']
            else spec.get('norm') or mcolors.Normalize(spec['vmin'],
                                                       spec['vmax']))
    ratio = OR.draw_surface(ax, S, ELEV, AZIM, val=d[spec['val_key']],
                            cmap=spec['cmap'], vmin=spec['vmin'],
                            vmax=spec['vmax'], contrast=0.30, norm=norm)
    # ~52k quads per panel: keep them out of the vector stream.  At DPR the
    # embedded raster is far above the 500 dpi the figure check asks for.
    for c in ax.collections:
        c.set_rasterized(True)
    colorbar(ax, norm, spec)
    return ratio


# ------------------------------------------------------------------ plane view
def meridian():
    """Hull meridian in the plane, (x, rho), for the reference outline."""
    ph = np.linspace(0.0, np.pi, 1201)
    return AXS * np.cos(ph), W.radius(ph, np.full_like(ph, 0.5 * np.pi))


def draw_plane(ax, X, Z, mask, spec, d, plane_key):
    F = np.where(mask, d[plane_key], np.nan)
    norm = (mcolors.LogNorm(spec['vmin'], spec['vmax']) if spec['log']
            else spec.get('norm') or mcolors.Normalize(spec['vmin'],
                                                       spec['vmax']))
    cmap = plt.get_cmap(spec['cmap']).copy()
    cmap.set_bad('0.90')            # the masked band, marked as deliberate
    # pcolormesh(X, Y, C) puts C[i, j] at (X[j], Y[i]), while the plane arrays
    # are indexed (x, z).  The two coordinate vectors hold the same values, so
    # getting this wrong is invisible in the ticks and shows up only as the
    # mask and the overlaid meridian disagreeing.
    ax.pcolormesh(X, Z, F.T, cmap=cmap, norm=norm, shading='gouraud',
                  rasterized=True)
    mx, mr = meridian()
    for s in (1.0, -1.0):
        ax.plot(mx, s * mr, '-', color='0.15', lw=0.7)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(ZLO, ZHI)
    ax.set_aspect('equal')
    colorbar(ax, norm, spec)


def save_row(figs, box_pad):
    boxes = []
    for _nm, fig in figs:
        fig.canvas.draw()
        boxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    from matplotlib.transforms import Bbox
    # 0.10 in of padding for the plane row: the measured tight bbox and the
    # rendering at DPR disagree by a couple of points, which at 0.02 shaves the
    # bottom of the x tick labels.
    box = Bbox.union(boxes).padded(box_pad)
    for nm, fig in figs:
        fig.savefig(nm, dpi=DPR, bbox_inches=box)
        plt.close(fig)
        assert check_pdf_text_inside(nm), 'text at or over the page edge'
        print('figure -> %s' % nm)


def main():
    d = np.load(SHELL)
    S = (d['Xs'], d['Ys'], d['Zs'])
    sx, sy, _dep, _r, _u, _v = OR.ortho(S, ELEV, AZIM)
    print('shell %s   theory rms %.4f   circuit rms %.4f   gap rms %.3e'
          % (S[0].shape, float(d['rms_ST']), float(d['rms_S0']),
             float(d['rms_G0'])))
    print('projected extents %.3f x %.3f (aspect %.3f) at elev %.0f azim %.0f'
          % (sx.max() - sx.min(), sy.max() - sy.min(),
             (sx.max() - sx.min()) / (sy.max() - sy.min()), ELEV, AZIM))
    print('speed window %.2f-%.2f   gap window %.1e-%.1e (log)'
          % (VLO, VHI, GLO, GHI))

    if '--plane' in sys.argv:
        X, Z, mask = d['PL_x'], d['PL_z'], d['PL_mask'].astype(bool)
        iy = int(d['PL_IY'])
        gap = np.where(mask, d['PL_G0'], np.nan)
        ghi = 10.0 ** np.ceil(np.log10(float(np.nanpercentile(gap, 90))))
        print('plane y = %+.4f   h = %.5f   two-cell band holds %d of %d points'
              % (X[iy], float(d['PL_h']), mask.sum(), mask.size))
        print('plane gap/U median %.3e   90th %.2e   99th %.2e   window 3e-4..%.2g'
              % (np.nanmedian(gap), np.nanpercentile(gap, 90),
                 np.nanpercentile(gap, 99), ghi))
        specs = {
            '24q': dict(cmap='turbo', vmin=VLO, vmax=VHI, log=False,
                        clab=r'$|\mathbf{v}|/U_\infty$', norm=speed_norm()),
            'gap': dict(cmap='magma', vmin=GLO, vmax=ghi, log=True,
                        clab=r'$|\mathbf{v}-\mathbf{v}_{\rm th}|/U_\infty$'),
        }
        if '--preview' in sys.argv:
            fig, axs = plt.subplots(1, 2, figsize=(6.2, 1.55))
            for ax, k, pk in zip(axs, ('24q', 'gap'), ('PL_S0', 'PL_G0')):
                draw_plane(ax, X, Z, mask, specs[k], d, pk)
            fig.subplots_adjust(left=0.004, right=0.988, top=0.98,
                                bottom=0.02, wspace=0.30)
            out = 'code/data/hull_field_plane_preview.png'
            fig.savefig(out, dpi=200)
            print('preview -> %s' % out)
            return
        figs = []
        for k, pk, nm in (('24q', 'PL_S0', 'code/data/hull_field_plane_a.pdf'),
                          ('gap', 'PL_G0', 'code/data/hull_field_plane_b.pdf')):
            fig, ax = plt.subplots(figsize=PLANE_FS, layout='constrained')
            draw_plane(ax, X, Z, mask, specs[k], d, pk)
            figs.append((nm, fig))
        save_row(figs, 0.10)
        return

    if '--preview' in sys.argv:
        fig, axs = plt.subplots(1, 2, figsize=(6.2, 1.30))
        for ax, k in zip(axs, ('24q', 'gap')):
            r = draw_shell(ax, S, SHELL_SPECS[k], d)
            assert abs(r - 1.0) < 1e-5, r
        fig.subplots_adjust(left=0.004, right=0.988, top=0.98, bottom=0.02,
                            wspace=0.30)
        out = 'code/data/hull_field_preview.png'
        fig.savefig(out, dpi=200)
        print('preview -> %s   (screen isotropy 1.000000)' % out)
        return

    if '--three' in sys.argv:
        fig, axs = plt.subplots(1, 3, figsize=(6.7, 1.35))
        for ax, k in zip(axs, ('theory', '24q', 'gap')):
            r = draw_shell(ax, S, SHELL_SPECS[k], d)
            assert abs(r - 1.0) < 1e-5, r
        fig.subplots_adjust(left=0.004, right=0.978, top=0.80, bottom=0.02,
                            wspace=0.34)
        out = 'fig/hull_field_3panel.pdf'
        fig.savefig(out, dpi=DPR)
        print('figure -> %s   (variant not used in the paper)' % out)
        return

    # Draw both panels first and save them through ONE bounding box: cropping
    # each to its own tight bbox gives them different page heights (the log
    # panel's tick labels are shorter), and two panels placed at the same width
    # would then differ in height by a few per cent in the row.
    figs = []
    for k, nm in (('24q', 'fig/hull_field_a.pdf'),
                  ('gap', 'fig/hull_field_b.pdf')):
        fig, ax = plt.subplots(figsize=PANEL_FS, layout='constrained')
        r = draw_shell(ax, S, SHELL_SPECS[k], d)
        assert abs(r - 1.0) < 1e-5, r
        figs.append((nm, fig))
    save_row(figs, 0.02)


if __name__ == '__main__':
    main()
