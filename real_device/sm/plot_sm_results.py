# -*- coding: utf-8 -*-
"""Figures for the hardware runs.

Momentum panel: full classical reference distribution (blue) against the four
most probable device outcomes (red), annotated with the dominant-mode match
and the Hellinger fidelity.
Position panel: full classical FFT reference (blue) against the measured
distribution read from JSON (red).

Output  ../fig/sm_real_momentum.pdf, ../fig/sm_real_position.pdf
Input   results_sm_*.txt and results_sm_*_probs.json next to this script.
"""
import json, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os, sys

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

import sm_qft_circuit as sm

_DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # real_device/
_HERE = os.path.join(_DEV, 'data')  # results / calibration files

OUT = os.path.join(_DEV, '..', 'fig')
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

CP_REF = '#0984e3'   # 蓝: 经典
CP_QM = '#d63031'    # 红: 真机
LBL_REF = 'Classical'
LBL_QM = 'Device'

def parse_momentum_txt(path):
    """解析 results_sm_momentum.txt → [{idx, U, R, hell, top, exp, ok, top4:[(k,p)]}]"""
    rows = []
    cur = None
    for ln in open(path, encoding='utf-8'):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(OK|FAIL)", ln)
        if m:
            if cur:
                rows.append(cur)
            cur = {"idx": int(m.group(1)), "U": float(m.group(2)), "R": float(m.group(3)),
                   "hell": float(m.group(4)), "top": int(m.group(5)), "exp": int(m.group(6)),
                   "ok": m.group(7) == 'OK', "top4": []}
            continue
        m4 = re.match(r"\s*Top4: (.+)", ln)
        if m4 and cur:
            for tok in m4.group(1).split(','):
                k, v = tok.strip().split(':')
                cur["top4"].append((int(k), float(v.rstrip('%')) / 100.0))
    if cur:
        rows.append(cur)
    return rows

def hell_bootstrap(qm, ref, n_shots=4000, n_boot=2000, seed=42):
    """Parametric bootstrap s.d. of the Hellinger fidelity f=(sum sqrt(p*q))^2.

    Resamples the measured distribution as multinomial counts at the known
    shot number and propagates the sampling noise into f.
    """
    keys = sorted(set(ref) | set(qm))
    q = np.array([qm.get(k, 0.0) for k in keys], dtype=float)
    r = np.array([ref.get(k, 0.0) for k in keys], dtype=float)
    if q.sum() <= 0:
        return 0.0
    q = q / q.sum()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_shots, q, size=n_boot)
    fs = (np.sqrt(counts / n_shots * r[None, :]).sum(axis=1)) ** 2
    return float(np.std(fs))

def bar_pair(ax, ref, qm, title, anno, show_legend=True, n_shots=None,
             ylim=(0, 1.03)):
    """参考全分布 + 测量分布 (可缺省的部分用 0)"""
    keys = sorted(set(ref) | set(qm))
    x = np.arange(len(keys))
    r = np.array([ref.get(k, 0.0) for k in keys])
    q = np.array([qm.get(k, 0.0) for k in keys])
    w = 0.36
    ax.bar(x - w / 2, r, w, label=LBL_REF, color=CP_REF, alpha=0.9,
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, q, w, label=LBL_QM, color=CP_QM, alpha=0.85,
           edgecolor='white', linewidth=0.5)
    if n_shots:
        se = np.sqrt(np.clip(q * (1 - q), 0, None) / n_shots)
        ax.errorbar(x + w / 2, q, yerr=se, fmt='none', ecolor='#2d3436',
                    elinewidth=0.8, capsize=1.5, zorder=5)
    # dotted guideline on the classical dominant mode
    if len(r) and np.nanmax(r) > 0:
        ax.axvline(x[int(np.nanargmax(r))], color=CP_REF, ls=':', lw=1.0,
                   alpha=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys], fontsize=6.5)
    ax.set_xlabel('Basis state index')
    ax.set_ylabel('Probability')
    ax.set_title(title, fontsize=9.5)
    ax.text(0.02, 0.97, anno, transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
    if show_legend:
        ax.legend(fontsize=7.5, framealpha=0.9)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:g}')

def main():
    # ================= 动量谱 =================
    mrows = parse_momentum_txt(os.path.join(_HERE, 'results_sm_momentum.txt'))
    full = {}
    if os.path.exists(os.path.join(_HERE, 'results_sm_momentum_probs.json')):
        for v in json.load(open(os.path.join(_HERE, 'results_sm_momentum_probs.json'),
                                encoding='utf-8')).values():
            full[tuple(v["params"])] = {int(k): float(p)
                                        for k, p in v["probs"].items()}
    sel = [(1, 0.5), (4.0, 0.5)]  # 匹配与失败的典型代表
    xs, ys = sm.grid_4x4()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (U, R) in zip(axes, sel):
        row = next(r for r in mrows if r["U"] == U and r["R"] == R)
        ref = sm.classic_momentum_pipeline(xs, ys, U, R)
        qm = dict(row["top4"])  # 只有 Top4 实测
        ok = "match" if row["ok"] else "mismatch"
        sig = (hell_bootstrap(full[(U, R)], ref)
               if (U, R) in full else 0.0)
        anno = (f"$U_\\infty$={U},  $R$={R}\n"
                f"dominant {row['top']} vs expected {row['exp']} [{ok}]\n"
                f"Hellinger $f$={row['hell']:.3f}±{sig:.3f}, 14 CNOT")
        bar_pair(ax, ref, qm, f"Momentum spectrum, $U_\\infty$={U}",
                 anno, show_legend=(ax is axes[0]), n_shots=4000)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sm_real_momentum.pdf", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {OUT}/sm_real_momentum.pdf  (动量 {len(mrows)} 组)")

    # ================= 位置传播子 =================
    if os.path.exists(os.path.join(_HERE, 'results_sm_position_probs.json')):
        data = json.load(open(os.path.join(_HERE, 'results_sm_position_probs.json'), encoding='utf-8'))
        n = len(data)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6)) if n > 1 else \
            (plt.subplots(1, 1, figsize=(4.6, 3.6)) and (fig, [ax]))
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, sorted(data, key=int)):
            d = data[key]
            t, U, R = d["params"]
            ref = sm.classic_qft_pipeline(xs, ys, t, U, R)
            qm = {int(k): v for k, v in d["probs"].items()}
            # no Hellinger annotation here: at this grid the position-space
            # reference is itself close to uniform, so the metric cannot
            # discriminate and the manuscript reports this panel qualitatively
            anno = (f"$t$={t},  $U_\\infty$={U},  $R$={R}")
            bar_pair(ax, ref, qm, f"Propagator, $t$={t}", anno,
                     show_legend=(ax is axes[0]), n_shots=4000, ylim=(0, 0.3))
        fig.tight_layout()
        fig.savefig(f"{OUT}/sm_real_position.pdf", dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"[ok] {OUT}/sm_real_position.pdf  (位置 {n} 组)")
    else:
        print("[skip] 位置实验尚未产生 JSON")

if __name__ == "__main__":
    main()
