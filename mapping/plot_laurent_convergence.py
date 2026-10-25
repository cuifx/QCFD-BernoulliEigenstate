# -*- coding: utf-8 -*-
"""三翼型洛朗级数重建收敛图 (RMS 数值对应正文 tab:laurent_rms)

3x4 矩阵:
  行: Joukowski / NACA 0012 / NACA 2412
  列: 截断阶数 K = 2 / 4 / 6 / 8
每子图: 标准边界(黑色实线, 物理显著洛朗级数重建) 与 截断重建边界(彩色虚线)
叠加, 右上角标注归一化 RMS 重建误差 (单位: 弦长).

洛朗系数来源:
  - Joukowski: 解析映射 z = zeta + 1/zeta, 偏移圆 zc = -eps + i*delta,
    展开 z = zeta' + c0 + sum_k c_k/zeta'^k, c0 = zc, c1 = 1,
    c_k = (-zc)^(k-1) (k>=2), |zeta'| = a.
  - NACA 0012 / 2412: theodorsen_mapping / naca2412_simulation 的
    梯度下降 Theodorsen 优化 (结果与论文表一致).

噪声平台处理: 级数项在边界上的振幅 a_k = |c_k|/R^k. 对 Theodorsen 映射,
a_k 在 k>=9 后进入 FFT 数值噪声平台 (a_k 不再单调衰减, ~1e-4),
这些项非物理, 从"标准边界"中剔除 (k < k_noise, k_noise 为首个
a_k > a_{k-1} 的 k). 物理显著的级数项到 k=8, K>=8 截断与标准边界重合.
"""
import os
import sys
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

_THIS = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(_THIS, "..", "fig")
N_GRID = 2048
CACHE = os.path.join(_THIS, "..", "data", "laurent_coeffs.npz")

def joukowski_laurent(eps=0.1, delta=0.05, kmax=40):
    """Joukowski 映射 z = zeta + 1/zeta 的洛朗系数 (见模块 docstring)."""
    zc = complex(-eps, delta)
    a = np.sqrt((1 + eps) ** 2 + delta ** 2)
    coeffs = np.zeros(kmax + 1, dtype=complex)
    coeffs[0] = zc
    coeffs[1] = 1.0
    for k in range(2, kmax + 1):
        coeffs[k] = (-zc) ** (k - 1)
    return a, coeffs

def fwd(zeta, coeffs):
    z = zeta.astype(complex).copy()
    z += coeffs[0]
    for k in range(1, len(coeffs)):
        if abs(coeffs[k]) < 1e-15:
            continue  # 跳过数值噪声/下溢项, 避免 zeta**k 下溢导致 0/0
        z += coeffs[k] / (zeta ** k)
    return z

def chord_of(z):
    return np.max(z.real) - np.min(z.real)

def noise_cutoff(R, coeffs):
    """首个 a_k > a_{k-1} (k>=5) 的位置; 无则 len(coeffs). 之前为物理显著项."""
    a = np.abs(coeffs) / (R ** np.arange(len(coeffs)))
    for k in range(5, len(coeffs)):
        if a[k] > a[k - 1]:
            return k
    return len(coeffs)

def build_dataset():
    """返回 [(title, R, coeffs, color, k_noise), ...] 三翼型数据集 (带 npz 缓存)."""
    titles = {0: "Joukowski", 1: "NACA 0012", 2: "NACA 2412"}
    colors = ("tab:red", "tab:green", "tab:blue")
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        out = []
        for i, (Rn, cn) in enumerate((("R_joukowski", "c_joukowski"),
                                      ("R_naca12", "c_naca12"),
                                      ("R_naca24", "c_naca24"))):
            R, c = float(d[Rn]), d[cn]
            out.append((titles[i], R, c, colors[i], noise_cutoff(R, c)))
        return out

    # ---- Joukowski (解析) ----
    a, cj = joukowski_laurent()
    # ---- NACA 0012 (Theodorsen GD) ----
    from theodorsen_mapping import compute_conformal_mapping
    R, c12, _, _, _, _, _, _, _ = compute_conformal_mapping(n_pts=N_GRID)
    # ---- NACA 2412 (cambered Theodorsen GD) ----
    from naca2412_simulation import theodorsen_mapping_naca2412
    R24, c24, _, _, _ = theodorsen_mapping_naca2412(n_pts=N_GRID)

    # joukowski_chord records that the Joukowski entries are the raw
    # shifted-form fit (chord 4.03, the family's natural scale); the production
    # geometry of the manuscript is the same section scaled to unit chord and
    # lives in the modules, not here.
    np.savez(CACHE, R_joukowski=a, c_joukowski=cj,
             R_naca12=R, c_naca12=c12, R_naca24=R24, c_naca24=c24,
             joukowski_chord=4.03340)
    return [(titles[i], Rv, cv, colors[i], noise_cutoff(Rv, cv))
            for i, (Rv, cv) in enumerate(((a, cj), (R, c12), (R24, c24)))]

def main():
    Ks = (2, 4, 6, 8)
    data = build_dataset()

    # 自检: Joukowski 级数重建 vs 直接解析映射 z = zeta + 1/zeta
    Rj, cj = data[0][1], data[0][2]
    thj = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
    zj_full = fwd(Rj * np.exp(1j * thj), cj)
    zj_direct = ((Rj * np.exp(1j * thj)) + (-0.1 + 0.05j) +
                 1.0 / (Rj * np.exp(1j * thj) + (-0.1 + 0.05j)))
    print("[check] Joukowski 级数 vs 直接映射: max|diff| = %.3e" %
          np.max(np.abs(zj_full - zj_direct)))

    # 每行: 数据 + 坐标范围 (行高 ∝ y 范围, 三行同尺度显示)
    row_meta = []
    for title, R, coeffs, color, k_noise in data:
        th = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
        zeta = R * np.exp(1j * th)
        z_exact = fwd(zeta, coeffs[:k_noise])      # 物理显著级数
        c = chord_of(z_exact)
        xlim = (z_exact.real.min() / c - 0.02, z_exact.real.max() / c + 0.02)
        ylim = (z_exact.imag.min() / c - 0.02, z_exact.imag.max() / c + 0.02)
        row_meta.append((title, R, coeffs, color, k_noise, z_exact, c,
                         xlim, ylim))

    yspans = [m[8][1] - m[8][0] for m in row_meta]

    # 行高 ∝ 归一化 y 范围 (三翼型厚度比接近, 行高近似相等, 同尺度显示)
    # 图高取适中值: 标题不被裁剪, 行内留白适度
    fig_w = 12.5
    left, right, wspace = 0.10, 0.97, 0.22
    top, bottom, hspace = 0.93, 0.095, 0.45
    fig_h = 4.0

    fig, axes = plt.subplots(3, len(Ks), figsize=(fig_w, fig_h),
                             gridspec_kw={"height_ratios": yspans})
    linestyles = {2: "--", 4: "-.", 6: ":", 8: (0, (4, 2))}
    summary = []
    for i, (title, R, coeffs, color, k_noise, z_exact, c, xlim, ylim) in enumerate(row_meta):
        th = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
        zeta = R * np.exp(1j * th)
        errs = {}
        for K in Ks:
            z_K = fwd(zeta, coeffs[:min(K, k_noise - 1) + 1])
            d = np.abs(z_exact - z_K) / c
            errs[K] = (np.sqrt(np.mean(d ** 2)), np.max(d))
        summary.append((title, errs))
        for j, K in enumerate(Ks):
            ax = axes[i, j]
            z_K = fwd(zeta, coeffs[:min(K, k_noise - 1) + 1])
            ax.plot(z_exact.real / c, z_exact.imag / c, "k-", lw=1.6)
            ax.plot(z_K.real / c, z_K.imag / c, ls=linestyles[K], color=color,
                    lw=1.1)
            rms, mx = errs[K]
            if rms > 0:
                ax.set_title(r"$K=%d$,  RMS$=% .1e$" % (K, rms), fontsize=10)
            else:
                ax.set_title(r"$K=%d$,  RMS$=0$" % K, fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")  # 不重排网格, 只扩坐标范围
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
            if i == 2:
                ax.set_xlabel(r"$x/c$", fontsize=9)
            if j == 0:
                ax.set_ylabel(title + "\n" + r"$y/c$", fontsize=9)
    from matplotlib.ticker import FormatStrFormatter
    n_rows, n_cols = axes.shape
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].yaxis.set_major_formatter(FormatStrFormatter("%+.2f"))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=wspace, hspace=hspace)
    fig.savefig(os.path.join(FIG_DIR, "laurent_convergence.png"), dpi=200)
    plt.close(fig)
    print("[OK] fig/laurent_convergence.png")

    print("\n归一化重建误差 (RMS / max, 单位: 弦长):")
    for title, R, _, _, k_noise in data:
        print("  噪声平台剔除起点 k_noise = %d (%s)" % (k_noise, title))
    for title, errs in summary:
        print("  %-10s: " % title +
              ", ".join("K=%d: %.3e / %.3e" % (K, errs[K][0], errs[K][1])
                        for K in Ks))
    return summary

if __name__ == "__main__":
    main()
