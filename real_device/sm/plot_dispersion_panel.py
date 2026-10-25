# -*- coding: utf-8 -*-
"""Figure for the position-space panel at T = 2.

Three panels:
  (a) ideal |psi_T|^2 (T=2, U=4) on the 4x4 grid
  (b) the same quantity on the device, readout-corrected
  (c) structure TV(P, 1/16) against U: ideal curve, device points, and the
      noise floor from the T=0 echo

The panel is drawn at the parameters where the structure is visible: at T = 0
the ideal distribution is exactly flat, and at U = 1 the structure is 0.10,
the same size as the readout noise, whereas at U = 4 it is 0.37.

Output fig/sm_real_dispersion.pdf plus a .png preview.
"""
import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_d = os.path.dirname(os.path.abspath(__file__))
_DEV = os.path.dirname(_d)
_ROOT = os.path.dirname(_DEV)
sys.path.insert(0, _d)
import sm_qft_circuit as sm  # noqa: E402

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 7,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
})

DATA = os.path.join(_DEV, 'data', 'results_dispersion_panel.json')
OUT = os.path.join(_ROOT, 'fig', 'sm_real_dispersion.pdf')
UN = 1.0 / 16.0


def load():
    d = json.load(open(DATA, encoding='utf-8'))
    pr = {k: {int(kk, 2): v for kk, v in d[k]['probs'].items()}
          for k in ['cal0', 'cal1', 'echo', 'U1', 'U2', 'U4']}
    return d, pr


def marg(p):
    return [sum(v for k, v in p.items() if (k >> q) & 1) for q in range(4)]


def correction(pr):
    e0 = marg(pr['cal0'])
    e1 = [1.0 - m for m in marg(pr['cal1'])]
    M = np.array([[1.0]])
    for q in range(4):
        A = np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]])
        M = np.kron(M, A)
    return M, e0, e1


def vec(p):
    v = np.array([p.get(k, 0.0) for k in range(16)], dtype=float)
    return v / v.sum()


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def field(v):
    """idx = ix*4+iy -> 4x4 数组 (ix 为行/横轴, iy 为纵轴)"""
    return np.array([[v[ix * 4 + iy] for iy in range(4)] for ix in range(4)])


def main():
    d, pr = load()
    M, e0, e1 = correction(pr)
    cases = {}
    for k in ['echo', 'U1', 'U2', 'U4']:
        p = vec(pr[k])
        pc = np.clip(np.linalg.solve(M, p), 0, None)
        cases[k] = pc / pc.sum()
    t, U, R = d['U4']['case']
    ref4 = vec(sm.classic_qft_pipeline(*sm.grid_4x4(), t, U, R))
    floor = tv(cases['echo'], np.full(16, UN))

    print(f"readout: P(1|0)={['%.3f' % x for x in e0]}  P(0|1)={['%.3f' % x for x in e1]}")
    print(f"U=4: TV(dev,ideal)={tv(cases['U4'], ref4):.3f}  "
          f"structure dev={tv(cases['U4'], np.full(16, UN)):.3f} ideal={tv(ref4, np.full(16, UN)):.3f}  "
          f"r={np.corrcoef(cases['U4'], ref4)[0, 1]:+.3f}")
    print(f"echo floor (structure)={floor:.3f}")

    xs = np.linspace(-2, 2, 4)
    dx = xs[1] - xs[0]
    ext = [xs[0] - dx / 2, xs[-1] + dx / 2, xs[0] - dx / 2, xs[-1] + dx / 2]

    # 手工定位: 三个子图等高 (方形场图与折线图共用同一条高度), 色标高度
    # 直接由 cax 的 rect 决定, 不再依赖 shrink/aspect
    W, H = 5.20, 1.90
    Y0, PH = 0.37, 1.26          # 底边 / 面板高度 (in)
    fig = plt.figure(figsize=(W, H))

    def rect(x, w):
        return [x / W, Y0 / H, w / W, PH / H]

    ax0 = fig.add_axes(rect(0.50, PH))
    ax1 = fig.add_axes(rect(1.94, PH))

    vmin = 0.0
    vmax = max(field(ref4).max(), field(cases['U4']).max()) * 1.05

    for ax, v, ttl in ((ax0, ref4, r'ideal'), (ax1, cases['U4'], r'device')):
        # pcolormesh 而非 imshow: 4x4 单元直接画成矢量方格, 不嵌位图
        # (imshow 在 PDF 里会被压成 ~100 dpi 的栅格层, 投稿检查会判不合格)
        edges = np.append(xs - dx / 2, xs[-1] + dx / 2)
        im = ax.pcolormesh(edges, edges, field(v).T, cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='flat')
        ax.set_aspect('equal')
        ax.set_title(ttl, pad=2)
        ax.set_xlabel(r'$x$')
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), '--', color='white',
                lw=0.7, alpha=0.85, zorder=3)
        if ax is ax0:
            ax.set_ylabel(r'$y$')
        else:
            ax.set_yticklabels([])
    # 色标也画成矢量: fig.colorbar 会把渐变渲染成 ~100 dpi 的位图条
    for xl, letter in ((0.50, '(a)'), (3.98, '(b)')):
        fig.text((xl - 0.17) / W, (Y0 + PH + 0.10) / H, letter,
                 fontsize=8, va='bottom', ha='right')

    cax = fig.add_axes(rect(3.25, 0.075))
    ylev = np.linspace(vmin, vmax, 65)
    cax.pcolormesh([0.0, 1.0], ylev, ylev[:-1, None], cmap='viridis',
                   vmin=vmin, vmax=vmax, shading='flat')
    cax.set_xlim(0, 1)
    cax.set_ylim(vmin, vmax)
    cax.set_xticks([])
    cax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    cax.tick_params(length=1.8, width=0.5, labelsize=6.5,
                    axis='y', labelright=True, labelleft=False,
                    direction='out')
    cax.yaxis.set_ticks_position('right')

    # (c) 结构量 vs U
    ax2 = fig.add_axes(rect(3.98, 1.17))
    Us = [1.0, 2.0, 4.0]
    ideal, dev, rr = [], [], []
    for u in Us:
        ref = vec(sm.classic_qft_pipeline(*sm.grid_4x4(), 2.0, u, 0.5))
        key = f'U{int(u)}'
        ideal.append(tv(ref, np.full(16, UN)))
        dev.append(tv(cases[key], np.full(16, UN)))
        rr.append(np.corrcoef(cases[key], ref)[0, 1])
    ax2.axhspan(0, floor, color='0.82', lw=0, zorder=0)
    ax2.text(3.05, 0.028, 'device floor (T=0 echo)', fontsize=5.6,
             color='0.25', va='bottom', ha='center')
    ax2.plot(Us, ideal, '-', color='#0984e3', lw=1.2, label='ideal')
    ax2.plot(Us, dev, 'o', color='#d63031', ms=4.2, mfc='white', mew=1.2,
             label='device')
    # 直接用数据坐标放文字 (annotate 的 offset points 让 get_window_extent
    # 量不准, 自检会漏判), 位置按曲线走向逐点定
    place = {1.0: (1.00, 0.058, 'left'), 2.0: (2.00, 0.270, 'center'),
             4.0: (3.95, 0.398, 'center')}
    for u, r in zip(Us, rr):
        xt, yt, ha = place[u]
        ax2.text(xt, yt, f'$r={r:+.2f}$', ha=ha, va='bottom', fontsize=6,
                 color='0.25')
    ax2.set_xscale('log')
    ax2.set_xticks(Us)
    ax2.set_xticklabels(['1', '2', '4'])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.tick_params(axis='x', which='minor', length=0)
    ax2.set_xlim(0.82, 5.4)
    ax2.set_ylim(0, 0.45)
    ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax2.set_xlabel(r'free-stream speed $U_\infty$')
    ax2.set_ylabel(r'$\mathrm{TV}(|\psi|^2,\ 1/16)$')
    ax2.legend(frameon=False, loc='upper left', handletextpad=0.4)
    ax2.tick_params(length=2, width=0.5)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)

    # 自检: 每个 r 标注的文字框不得与折线相交
    # 自检: 文字框 (换算回数据坐标) 与该段曲线的最高点比较
    fig.canvas.draw()
    inv = ax2.transData.inverted()
    for t in ax2.texts:
        bb = t.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        xs_ = np.linspace(max(x0, 0.82), min(x1, 5.4), 60)
        yy = np.interp(np.log(xs_), np.log(Us), ideal)
        ymax, ymin = yy.max(), yy.min()
        flag = 'OVERLAP' if (ymax > y0 and ymin < y1) else 'clear'
        print(f"  标注 {t.get_text()!r}: 曲线在该段最高 {ymax:.3f}, "
              f"文字框底 {y0:.3f} -> {flag}")

    fig.savefig(OUT, bbox_inches='tight', pad_inches=0.07)
    fig.savefig(OUT.replace('.pdf', '.png'), dpi=300)
    print('saved', OUT)


if __name__ == '__main__':
    main()
