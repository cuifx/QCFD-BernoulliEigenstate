"""
从圆到 NACA 0012 翼型的共形映射 -- DFT 频谱投影 + 梯度下降 Theodorsen 优化
==============================================================================
映射: z = f(zeta) = zeta + c_0 + c_1/zeta + c_2/zeta^2 + ...
边界: |zeta| = R

方法:
  1. 余弦参数化 NACA 0012, DFT 频谱投影得初始共形映射 (所有 k<=1 分量)
  2. 梯度下降优化 theta(s), 最小化 k>=2 频率能量
  3. 收敛后提取 Laurent 系数

参考文献:
  Theodorsen, T., "Theory of Wing Sections", McGraw-Hill, 1948.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def naca0012_thickness(x):
    x = np.maximum(x, 0.0)
    return 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x
                  - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)


def naca0012_thickness_derivative(x):
    x = np.maximum(x, 1e-16)
    sqrt_x = np.sqrt(x)
    return 0.6 * (0.2969 * 0.5 / sqrt_x - 0.126
                  - 0.3516 * 2 * x + 0.2843 * 3 * x**2
                  - 0.1015 * 4 * x**3)


def generate_airfoil(n=512):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 0.5 * (1.0 + np.cos(theta))
    yt = naca0012_thickness(x)
    y = np.where(np.sin(theta) >= 0, yt, -yt)
    return theta, x, y, x + 1j * y


def airfoil_at_theta(theta):
    x = 0.5 * (1.0 + np.cos(theta))
    yt = naca0012_thickness(x)
    y = np.where(np.sin(theta) >= 0, yt, -yt)
    return x + 1j * y


def airfoil_tangent(theta):
    x = 0.5 * (1.0 + np.cos(theta))
    dx = -0.5 * np.sin(theta)
    yt = naca0012_thickness(x)
    dyt_dx = naca0012_thickness_derivative(x)
    dy = np.where(np.sin(theta) >= 0, dyt_dx * dx, -dyt_dx * dx)
    return dx + 1j * dy


def theodorsen_correction(n_pts=2048):
    """
    梯度下降优化 Theodorsen 参数化 theta(s).

    目标: 最小化 E(theta) = sum_{k>=2} |zhat_k|^2
    梯度: dE/dtheta_j = (2/N) Re[z_pos(s_j) * conj(dz/dtheta(s_j))]

    返回: s, theta, z_new, final_dist
    """
    s = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    k_freq = fftfreq(n_pts) * n_pts

    def idx(f):
        return np.argmin(np.abs(k_freq - f))

    theta = s.copy()
    velocity = np.zeros(n_pts)

    max_iter = 2000
    momentum = 0.9
    lr_base = 5.0
    tol = 1e-12
    best_dist = 1.0
    best_theta = theta.copy()

    print("梯度下降 Theodorsen 优化 (n_pts=%d)" % n_pts)
    print("-" * 55)

    for iteration in range(max_iter):
        z = airfoil_at_theta(theta)
        z_hat = fft(z) / n_pts

        pos_energy = sum(np.abs(z_hat[idx(k)])**2 for k in range(2, n_pts // 2))
        total_energy = sum(np.abs(z_hat)**2)
        dist = pos_energy / total_energy if total_energy > 0 else 0.0

        if dist < best_dist:
            best_dist = dist
            best_theta = theta.copy()

        if iteration % 100 == 0:
            max_k2 = max(np.abs(z_hat[idx(k)]) for k in range(2, min(n_pts // 2, 10)))
            print("  Iter %4d: E=%.4e  max|zhat_{k>=2}|=%.4e  lr=%.4f" %
                  (iteration, dist, max_k2, lr_base))

        if dist < tol:
            print("  V 收敛于迭代 %d (E=%.1e)" % (iteration, dist))
            break

        z_hat_pos = np.zeros(n_pts, dtype=complex)
        for k in range(2, n_pts // 2):
            z_hat_pos[idx(k)] = z_hat[idx(k)]
        z_pos = ifft(z_hat_pos * n_pts)

        dz_dtheta = airfoil_tangent(theta)
        grad = (2.0 / n_pts) * np.real(z_pos * np.conj(dz_dtheta))

        def compute_dist(theta_test):
            zt = airfoil_at_theta(theta_test)
            zht = fft(zt) / n_pts
            pe = sum(np.abs(zht[idx(k)])**2 for k in range(2, n_pts // 2))
            te = sum(np.abs(zht)**2)
            return pe / te if te > 0 else 0.0

        lr = lr_base
        dist_curr = dist
        for _ in range(5):
            theta_trial = np.mod(theta - lr * grad, 2 * np.pi)
            dist_trial = compute_dist(theta_trial)
            if dist_trial < dist_curr:
                break
            lr *= 0.5
        else:
            lr = lr_base * 0.1
            lr_base = max(lr_base * 0.8, 0.001)

        velocity = momentum * velocity - lr * grad
        theta = theta + velocity
        theta = np.mod(theta, 2 * np.pi)
        theta = np.sort(theta)

        if dist < best_dist * 1.5:
            lr_base = min(lr_base * 1.01, 20.0)

    print("  最佳畸变: %.4e (恢复最佳参数)" % best_dist)
    theta = best_theta
    z_new = airfoil_at_theta(theta)

    _, _, _, z0 = generate_airfoil(n_pts)
    z_hat0 = fft(z0) / n_pts
    dist_old = (sum(np.abs(z_hat0[idx(k)])**2 for k in range(2, n_pts // 2)) /
                sum(np.abs(z_hat0)**2))

    print("  初始畸变: %.4e  ->  最终畸变: %.4e" % (dist_old, best_dist))
    if best_dist < dist_old:
        print("  改善倍数: %.2fx" % (dist_old / best_dist))
    print("-" * 55)

    return s, theta, z_new, best_dist


def compute_conformal_mapping(n_pts=2048):
    s, theta, z_af, _ = theodorsen_correction(n_pts=n_pts)

    z_hat = fft(z_af) / n_pts
    freqs = fftfreq(n_pts) * n_pts

    def idx_of(f):
        return np.where(np.abs(freqs - f) < 0.5)[0][0]

    R = np.abs(z_hat[idx_of(1)])
    phase_R = np.angle(z_hat[idx_of(1)])
    z_af_aligned = z_af * np.exp(-1j * phase_R)
    z_hat_aligned = fft(z_af_aligned) / n_pts
    R = np.real(z_hat_aligned[idx_of(1)])

    print("\n共形映射参数 (梯度下降优化后)")
    print("  离散点: %d" % n_pts)
    print("  圆半径 R = %.8f (弦长 c = 1)" % R)
    print("  圆直径 2R = %.8f (vs 翼型弦长 1.0)" % (2 * R))

    n_coeff = n_pts // 2
    coeffs = np.zeros(n_coeff + 1, dtype=complex)
    coeffs[0] = z_hat_aligned[idx_of(0)]
    for k in range(1, n_coeff + 1):
        coeffs[k] = (R ** k) * z_hat_aligned[idx_of(-k)]

    # 共形映射允许的频谱 (所有 k<=1)
    z_hat_conformal = np.zeros(n_pts, dtype=complex)
    z_hat_conformal[idx_of(1)] = R
    z_hat_conformal[idx_of(0)] = z_hat_aligned[idx_of(0)]
    for k in range(1, n_coeff + 1):
        z_hat_conformal[idx_of(-k)] = z_hat_aligned[idx_of(-k)]

    z_hat_positive = np.zeros(n_pts, dtype=complex)
    for k in range(2, n_pts // 2):
        z_hat_positive[idx_of(k)] = z_hat_aligned[idx_of(k)]

    z_proj_aligned = ifft(z_hat_conformal * n_pts)
    z_pos_aligned = ifft(z_hat_positive * n_pts)
    z_proj = z_proj_aligned * np.exp(1j * phase_R)
    z_pos = z_pos_aligned * np.exp(1j * phase_R)

    proj_err = np.abs(z_proj - z_af)
    max_err = np.max(proj_err)
    rms_err = np.sqrt(np.mean(proj_err**2))

    pos_energy = np.sum(np.abs(z_hat_positive)**2)
    total_energy = np.sum(np.abs(z_hat_aligned)**2)
    distortion_ratio = pos_energy / total_energy if total_energy > 0 else 0.0

    print("  正频率能量占比: %.6e" % distortion_ratio)
    print("  投影最大偏差: %.4e" % max_err)
    print("  投影 RMS 偏差: %.4e" % rms_err)

    print("\n  === 优化后 Laurent 系数 (高精度) ===")
    for k in range(min(10, n_coeff + 1)):
        c = coeffs[k]
        print("    c_%d = %+.15e %+.15e i" % (k, c.real, c.imag))
    print("  系数衰减:")
    for k in [0, 1, 2, 4, 8, 16, 32, 64]:
        if k < len(coeffs):
            print("    |c_%3d| = %.6e" % (k, abs(coeffs[k])))

    # ===== TEX-ready 输出 =====
    print("\n" + "=" * 60)
    print("  TEX 可用的共形映射表达式 (复制到 tex 文件)")
    print("=" * 60)
    print("  z = f(zeta) = zeta + c_0 + c_1/zeta + c_2/zeta^2 + c_3/zeta^3 + ...")
    print("  |zeta| = R = %.8f (弦长 c = 1)" % R)
    print()
    for k in range(6):
        c = coeffs[k]
        print("  c_%d &= %+.8f %+.2e\\,i \\qquad (|c_%d| = %.4e) \\\\" %
              (k, c.real, c.imag, k, abs(c)))
    print("\n  级数截断到前 5 项:")
    terms = []
    for k in range(6):
        c = coeffs[k]
        if k == 0:
            terms.append("(%+.8f%+.2e\\,i)" % (c.real, c.imag))
        else:
            terms.append("\\frac{%+.8f%+.2e\\,i}{\\zeta^{%d}}" % (c.real, c.imag, k))
    print("  z = \\zeta + " + " + ".join(terms[:3]) + " + \\cdots")
    print("=" * 60)

    return R, coeffs, theta, z_af, z_proj, z_pos, distortion_ratio, phase_R, s


# ============================================================
# 可视化
# ============================================================

def plot_airfoil(x, y, path):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(x, y, 'b-', lw=1.5)
    ax.fill(x, y, alpha=0.2, color='skyblue')
    ax.set_aspect('equal')
    ax.set_xlabel('$x/c$', fontsize=12)
    ax.set_ylabel('$y/c$', fontsize=12)
    ax.set_title('NACA 0012 Airfoil Profile', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def plot_projection(theta, z_af, z_proj, path, subtitle=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.0),
                                   gridspec_kw={'width_ratios': [2.5, 1]})

    ax1.plot(np.real(z_af), np.imag(z_af), 'b-', lw=1.5, label='NACA 0012')
    ax1.plot(np.real(z_proj), np.imag(z_proj), 'r--', lw=2.0,
             label='Conformal Mapping Projection')
    ax1.set_xlabel('$x/c$', fontsize=12)
    ax1.set_ylabel('$y/c$', fontsize=12)
    title_full = 'NACA 0012 vs Conformal Mapping Boundary'
    if subtitle:
        title_full += ' (%s)' % subtitle
    ax1.set_title(title_full, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.15, 0.15)

    idx_le = np.argmin(np.real(z_af))  # 前缘: x坐标最小处 (≈0)
    margin = 80
    i_start = max(0, idx_le - margin)
    i_end = min(len(z_af), idx_le + margin)
    ax2.plot(np.real(z_af[i_start:i_end]), np.imag(z_af[i_start:i_end]),
             'b-', lw=2.0, label='NACA 0012')
    ax2.plot(np.real(z_proj[i_start:i_end]), np.imag(z_proj[i_start:i_end]),
             'r--', lw=2.5, label='Conformal Mapping Projection')
    ax2.set_xlabel('$x/c$', fontsize=12)
    ax2.set_ylabel('$y/c$', fontsize=12)
    ax2.set_title('Leading Edge Zoom' + (' (%s)' % subtitle if subtitle else ''),
                  fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def plot_theta_correspondence(s, theta, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(s, theta, 'b-', lw=0.8)
    ax1.plot(s, s, 'r--', lw=1.2, label='$\\theta(s)=s$ (余弦)')
    ax1.set_xlabel('$s$ (rad)', fontsize=12)
    ax1.set_ylabel('$\\theta(s)$ (rad)', fontsize=12)
    ax1.set_title('Theodorsen Correspondence $\\theta(s)$', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_ylim(0, 2 * np.pi)

    dtheta = theta - s
    ax2.plot(s, dtheta, 'g-', lw=0.8)
    ax2.set_xlabel('$s$ (rad)', fontsize=12)
    ax2.set_ylabel('$\\theta(s) - s$ (rad)', fontsize=12)
    ax2.set_title('Parameterization Correction $\\theta(s)-s$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2 * np.pi)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def plot_distortion(theta, z_pos, path):
    eps_x = np.real(z_pos)
    eps_y = np.imag(z_pos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.plot(theta, eps_x, 'b-', lw=0.8)
    ax1.set_xlabel('$\\theta$ (rad)', fontsize=12)
    ax1.set_ylabel('$\\Delta x$ (chord units)', fontsize=12)
    ax1.set_title('x Distortion (Positive Frequency)', fontsize=13)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.grid(True, alpha=0.3)
    ax2.plot(theta, eps_y, 'r-', lw=0.8)
    ax2.set_xlabel('$\\theta$ (rad)', fontsize=12)
    ax2.set_ylabel('$\\Delta y$ (chord units)', fontsize=12)
    ax2.set_title('y Distortion (Positive Frequency)', fontsize=13)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))
    return max(np.max(np.abs(eps_x)), np.max(np.abs(eps_y))), \
           np.sqrt(np.mean(np.abs(z_pos)**2))


def plot_conformal_grid(R, coeffs, R_ext=3.0, path="conformal_grid.png"):
    mags = np.abs(coeffs)
    k_cutoff = min(len(coeffs) - 1,
                   max(8, np.searchsorted(mags[::-1], 1e-15, side='right')))
    k_cutoff = len(coeffs) - 1 - k_cutoff
    k_cutoff = max(k_cutoff, 8)
    coeffs_trunc = coeffs[:k_cutoff + 1]
    nc = len(coeffs_trunc)

    def fwd(zeta):
        zeta = np.asarray(zeta, dtype=complex)
        z = zeta.astype(complex).copy()
        z += coeffs_trunc[0]
        for k in range(1, nc):
            if np.abs(coeffs_trunc[k]) < 1e-16:
                break
            z += coeffs_trunc[k] / (zeta ** k)
        return z

    th = np.linspace(0, 2 * np.pi, 200)
    radii_zeta = np.linspace(R, R_ext, 28)
    n_rays = 33

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for r in radii_zeta:
        c = r * np.exp(1j * th)
        ax1.plot(np.real(c), np.imag(c), 'b-', lw=0.3, alpha=0.6)
    for a in np.linspace(0, 2 * np.pi, n_rays):
        rr = np.linspace(R, R_ext, 60)
        ax1.plot(np.real(rr * np.exp(1j * a)), np.imag(rr * np.exp(1j * a)),
                 'r-', lw=0.3, alpha=0.6)
    circle_R = R * np.exp(1j * th)
    ax1.plot(np.real(circle_R), np.imag(circle_R), 'k-', lw=2,
             label='$|\\zeta|=R=%.4f$' % R)
    ax1.set_aspect('equal')
    ax1.set_title('$\\zeta$-plane', fontsize=14)
    ax1.set_xlabel('$\\xi$', fontsize=12)
    ax1.set_ylabel('$\\eta$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-R_ext, R_ext)
    ax1.set_ylim(-R_ext, R_ext)

    for r in radii_zeta:
        cz = fwd(r * np.exp(1j * th))
        ax2.plot(np.real(cz), np.imag(cz), 'b-', lw=0.3, alpha=0.6)
    for a in np.linspace(0, 2 * np.pi, n_rays):
        rr = np.linspace(R, R_ext, 120)
        ax2.plot(np.real(fwd(rr * np.exp(1j * a))),
                 np.imag(fwd(rr * np.exp(1j * a))), 'r-', lw=0.3, alpha=0.6)
    af = fwd(R * np.exp(1j * np.linspace(0, 2 * np.pi, 800)))
    ax2.plot(np.real(af), np.imag(af), 'k-', lw=2, label='NACA 0012')
    ax2.fill(np.real(af), np.imag(af), alpha=0.3, color='skyblue')
    ax2.set_aspect('equal')
    ax2.set_title('$z$-plane (Airfoil Domain)', fontsize=14)
    ax2.set_xlabel('$x/c$', fontsize=12)
    ax2.set_ylabel('$y/c$', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    # 根据最外层映射圆的实际范围自动设置边界
    outer_circle_mapped = fwd(R_ext * np.exp(1j * th))
    x_min, x_max = np.min(np.real(outer_circle_mapped)), np.max(np.real(outer_circle_mapped))
    y_min, y_max = np.min(np.imag(outer_circle_mapped)), np.max(np.imag(outer_circle_mapped))
    margin = 0.3
    ax2.set_xlim(x_min - margin, x_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def plot_spectrum(z_af, n_pts, path):
    z_hat = fft(z_af) / n_pts
    freqs = fftfreq(n_pts) * n_pts
    mag = np.abs(z_hat)
    idx_sort = np.argsort(freqs)
    f_sorted, m_sorted = freqs[idx_sort], mag[idx_sort]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['red' if f >= 2 else 'blue' for f in f_sorted]
    ax.stem(f_sorted, m_sorted, linefmt='grey', markerfmt=' ', basefmt=' ')
    ax.scatter(f_sorted, m_sorted, c=colors, s=15, zorder=5)
    ax.axvspan(2, n_pts // 2, alpha=0.1, color='red',
               label='Angular Distortion ($k\\geq 2$)')
    ax.axvspan(-n_pts // 2, 1.5, alpha=0.1, color='blue',
               label='Conformal Mapping Allowed ($k\\leq 1$)')
    ax.set_xlabel('Frequency $k$', fontsize=12)
    ax.set_ylabel('$|\\hat{z}_k|$', fontsize=12)
    ax.set_title('DFT Spectral Decomposition (After Gradient Descent)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-n_pts // 2, n_pts // 2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def plot_coefficient_decay(coeffs, path):
    mags = np.abs(coeffs)
    kmax = 32  # k≳32受有限DFT分辨率影响，不绘制
    kmax = min(kmax, len(mags)-1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(np.arange(kmax+1), mags[:kmax+1], 'bo-', markersize=4, lw=1.0)
    ax.set_xlabel('$k$', fontsize=12)
    ax.set_ylabel('$|c_k|$', fontsize=12)
    ax.set_title('Laurent Coefficient Decay (After Gradient Descent, $k\\leq%d$)' % kmax, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, kmax+0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def plot_theta_sampling(theta, n_pts, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    dtheta = np.diff(np.sort(theta))
    dtheta = np.append(dtheta, dtheta[0])
    dtheta_uniform = 2 * np.pi / n_pts
    ax.plot(np.sort(theta), dtheta, 'b-', lw=0.8)
    ax.axhline(y=dtheta_uniform, color='r', linestyle='--', lw=1.2,
               label='Uniform Spacing ($2\\pi/N=%.4f$)' % dtheta_uniform)
    ax.set_xlabel('$\\theta$ (rad)', fontsize=12)
    ax.set_ylabel('$\\Delta\\theta$ Local Spacing (rad)', fontsize=12)
    ax.set_title('Non-uniform Sampling Density of $\\theta(s)$ after GD', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2 * np.pi)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(path))


def main():
    print("=" * 60)
    print("  共形映射: 圆 -> NACA 0012 (梯度下降优化 + DFT 频谱投影)")
    print("=" * 60)

    n_pts = 2048
    (R, coeffs, theta, z_af, z_proj, z_pos,
     dist_ratio, phase_R, s) = compute_conformal_mapping(n_pts=n_pts)

    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'fig')

    # ---- 优化前的映射对比图 (余弦/初步参数化, 使用所有 k<=1 频率) ----
    # 先做少量梯度下降迭代 (~50步), 作为"初步优化"的中间状态,
    # 既比纯余弦参数化有明显改善, 又与完全优化 (2000步) 有可见差距
    from numpy.fft import fftfreq as _fftfreq
    k_freq0 = _fftfreq(n_pts) * n_pts

    def _idx(f):
        return np.argmin(np.abs(k_freq0 - f))

    theta0_init = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)  # 余弦参数化
    theta0 = theta0_init.copy()
    vel0 = np.zeros(n_pts)
    lr0 = 3.0
    dist_best0 = 1.0
    theta_best0 = theta0.copy()

    for it in range(200):
        zt = airfoil_at_theta(theta0)
        zht = fft(zt) / n_pts
        pe = sum(np.abs(zht[_idx(kk)])**2 for kk in range(2, n_pts // 2))
        te = sum(np.abs(zht)**2)
        dist_cur = pe / te if te > 0 else 0.0
        if dist_cur < dist_best0:
            dist_best0 = dist_cur
            theta_best0 = theta0.copy()

        zht_pos = np.zeros(n_pts, dtype=complex)
        for kk in range(2, n_pts // 2):
            zht_pos[_idx(kk)] = zht[_idx(kk)]
        zt_pos = ifft(zht_pos * n_pts)
        dzd = airfoil_tangent(theta0)
        grad0 = (2.0 / n_pts) * np.real(zt_pos * np.conj(dzd))

        # 简易线搜索
        lr_try = lr0
        for _ in range(4):
            tt = np.mod(theta0 - lr_try * grad0, 2 * np.pi)
            ztt = airfoil_at_theta(tt)
            zhtt = fft(ztt) / n_pts
            pet = sum(np.abs(zhtt[_idx(kk)])**2 for kk in range(2, n_pts // 2))
            tet = sum(np.abs(zhtt)**2)
            dt = pet / tet if tet > 0 else 0.0
            if dt < dist_cur:
                break
            lr_try *= 0.5
        else:
            lr_try = lr0 * 0.1

        vel0 = 0.9 * vel0 - lr_try * grad0
        theta0 = theta0 + vel0
        theta0 = np.mod(theta0, 2 * np.pi)
        theta0 = np.sort(theta0)
        if dist_cur < dist_best0 * 1.5:
            lr0 = min(lr0 * 1.01, 10.0)

    theta0 = theta_best0

    z0 = airfoil_at_theta(theta0)
    z_hat0 = fft(z0) / n_pts
    freqs = fftfreq(n_pts) * n_pts

    def idx0(f):
        return np.argmin(np.abs(freqs - f))

    phase_R0 = np.angle(z_hat0[idx0(1)])
    z0_aligned = z0 * np.exp(-1j * phase_R0)
    z_hat0_aligned = fft(z0_aligned) / n_pts
    R0 = np.real(z_hat0_aligned[idx0(1)])

    # 所有 k<=1: k=1,0,-1,-2,...,-N/2
    z_hat_proj0 = np.zeros(n_pts, dtype=complex)
    z_hat_proj0[idx0(1)] = z_hat0_aligned[idx0(1)]
    z_hat_proj0[idx0(0)] = z_hat0_aligned[idx0(0)]
    for k in range(1, n_pts // 2 + 1):
        z_hat_proj0[idx0(-k)] = z_hat0_aligned[idx0(-k)]
    z_proj0 = ifft(z_hat_proj0 * n_pts) * np.exp(1j * phase_R0)

    proj_err0 = np.abs(z_proj0 - z0)
    max_err0 = np.max(proj_err0)
    rms_err0 = np.sqrt(np.mean(proj_err0**2))
    pos_energy0 = sum(np.abs(z_hat0_aligned[idx0(k)])**2
                      for k in range(2, n_pts // 2))
    total_energy0 = sum(np.abs(z_hat0_aligned)**2)
    dist0 = pos_energy0 / total_energy0 if total_energy0 > 0 else 0.0

    coeffs0 = np.zeros(n_pts // 2 + 1, dtype=complex)
    coeffs0[0] = z_hat0_aligned[idx0(0)]
    for k in range(1, n_pts // 2 + 1):
        coeffs0[k] = (R0 ** k) * z_hat0_aligned[idx0(-k)]

    print("\n优化前 (余弦参数化 + 200步初步梯度下降, 全部 k<=1 分量) 精度:")
    print("  圆半径 R0 = %.8f" % R0)
    print("  正频率能量占比: %.6e" % dist0)
    print("  投影最大偏差: %.6e" % max_err0)
    print("  投影 RMS 偏差: %.6e" % rms_err0)
    print("  优化前 Laurent 系数 (前 6 项):")
    for k in range(min(6, len(coeffs0))):
        c = coeffs0[k]
        print("    c_%d = %+.15e %+.15e i  (|c_%d| = %.6e)" %
              (k, c.real, c.imag, k, abs(c)))
    # TEX-ready 优化前系数
    print("\n  --- TEX: 优化前 Laurent 系数 ---")
    for k in range(6):
        c = coeffs0[k]
        print("  c_%d &= %+.12e %+.2e\\\\,i \\\\\\\\" % (k, c.real, c.imag))

    plot_projection(theta0, z0, z_proj0,
                    os.path.join(d, "mapping_comparison_before.png"),
                    subtitle="Preliminary Optimization (200 GD steps)")

    # REMOVED: low-value figure
    # 1. 翼型轮廓
    # plot_airfoil(np.real(z_af), np.imag(z_af),
    #              os.path.join(d, "naca0012_profile.png"))

    # 2. 映射对比 (优化后)
    plot_projection(theta, z_af, z_proj,
                    os.path.join(d, "mapping_comparison.png"),
                    subtitle="After Gradient Descent")

    # 3. Theodorsen theta(s) 对应
    plot_theta_correspondence(s, theta,
                              os.path.join(d, "theodorsen_correspondence.png"))

    # 4. 角度畸变
    mx_eps, rm_eps = plot_distortion(
        theta, z_pos,
        os.path.join(d, "angle_distortion.png"))

    # 5. 共形网格
    plot_conformal_grid(R, coeffs, R_ext=3.0,
                        path=os.path.join(d, "conformal_grid.png"))

    # 6. 频谱
    plot_spectrum(z_af, n_pts,
                  os.path.join(d, "mapping_error.png"))

    # 7. 系数衰减
    plot_coefficient_decay(coeffs,
                           os.path.join(d, "coefficient_decay.png"))

    # 8. theta(s) 非均匀采样
    plot_theta_sampling(theta, n_pts,
                        os.path.join(d, "theta_sampling.png"))

    # ===== 最终汇总 =====
    print("\n" + "=" * 60)
    print("  数值结果汇总")
    print("=" * 60)
    print("  离散点:              %d" % n_pts)
    print("  圆半径 R (优化后):   %.8f" % R)
    print("  圆半径 R0 (优化前):  %.8f" % R0)
    print("  Laurent 项数:        %d" % len(coeffs))
    print("  优化前畸变能量:      %.4e" % dist0)
    print("  优化后畸变能量:      %.4e" % dist_ratio)
    print("  改善倍数:            %.1f" % (dist0 / dist_ratio))
    print("  优化后畸变 max:      %.4e" % mx_eps)
    print("  优化后畸变 RMS:      %.4e" % rm_eps)
    print("=" * 60)

    return R, coeffs, theta, z_af, z_proj


if __name__ == "__main__":
    main()