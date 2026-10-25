# -*- coding: utf-8 -*-
"""Hardware-efficient encoder trained on the simulator.

Prepares the SM phase-encoded state |psi> = (1/4) sum_j e^{i Phi_j} |j>.

Ansatz, matched to the coupling path of WK_C180_2 ([39,49,58,67]): from
|0...0>, L layers of alternating structure, each layer being

    single-qubit layer   RZ(theta) . RY(theta) on every qubit
    entangling layer     CNOT chain 0-1, 1-2, 2-3, nearest neighbours only,
                         so transpilation inserts no swaps

Adam with the parameter-shift rule maximises |<psi_target|psi(theta)>|^2. The
result is self-checked on the pyqpanda CPUQVM and written to JSON, from which
the hardware circuit reads it.
"""
import json, builtins, time
import numpy as np
from scipy.optimize import minimize

try:
    from pyqpanda import *
    HAS_PYQPANDA = True
except ImportError:
    HAS_PYQPANDA = False

import os, sys

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

import sm_qft_circuit as sm
import os

_DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # real_device/
_HERE = os.path.join(_DEV, 'data')  # results / calibration files

# ---------------- 目标态 ----------------

def target_state(U, R):
    """|ψ_target⟩ = (1/4)Σ_j e^{iΦ_j}|j⟩, j = ix*4+iy"""
    xs, ys = sm.grid_4x4()
    Phi = np.array([[sm.potential_cylinder(x, y, U, R) for y in ys] for x in xs]).ravel()
    psi = np.exp(1j * Phi) / 4.0
    return psi

# ---------------- 门矩阵 (pyqpanda 约定) ----------------
# RY(θ) = exp(-iθY/2), RZ(θ) = exp(-iθZ/2); 比特序 q[0] 为最低位 (little-endian)

def ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]])

def rz(a):
    return np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]])

def cnot():
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

def embed_single(g2, q, n):
    """单比特门嵌入 n 比特空间 (q[0] 最低位). 返回 2^n x 2^n"""
    N = 1 << n
    U = np.zeros((N, N), dtype=complex)
    for j in range(N):
        b = (j >> q) & 1
        jz = j & ~(1 << q)
        for nb in (0, 1):
            U[jz | (nb << q), j] = g2[nb, b]
    return U

def embed_two(g4, c, t, n):
    """双比特门嵌入 (c 控制, t 目标). 返回 2^n x 2^n"""
    N = 1 << n
    U = np.zeros((N, N), dtype=complex)
    for j in range(N):
        bc, bt = (j >> c) & 1, (j >> t) & 1
        jz = j & ~((1 << c) | (1 << t))
        for nc in (0, 1):
            for nt_ in (0, 1):
                U[jz | (nc << c) | (nt_ << t), j] = g4[nc * 2 + nt_, bc * 2 + bt]
    return U

# ---------------- ansatz ----------------

def build_ansatz(theta, L, n=4):
    """硬件高效 ansatz 酉矩阵. 每层: RZ·RY 每比特 + CNOT 链 0-1,1-2,2-3.
    参数索引: 每层 2n 个单比特角, 共 L*2n 个. 层间共享纠缠结构."""
    U = np.eye(1 << n, dtype=complex)
    p = 0
    for l in range(L):
        # 单比特层
        for q in range(n):
            U = embed_single(rz(theta[p]), q, n) @ U
            U = embed_single(ry(theta[p + 1]), q, n) @ U
            p += 2
        # 纠缠层: 链状近邻 CNOT
        for q in range(n - 1):
            U = embed_two(cnot(), q, q + 1, n) @ U
    assert p == len(theta)
    return U

def fidelity(theta, psi_t, L, n=4):
    psi = build_ansatz(theta, L, n) @ np.eye(1 << n)[:, 0]
    return abs(np.vdot(psi_t, psi)) ** 2

# ---------------- 训练 ----------------

def train(L, psi_t, steps=800, lr=0.03, n=4, seed=20260804):
    """Adam 预热 + L-BFGS-B 精修, 多个种子取最优. 返回 (theta, fid)"""
    def loss_grad(theta):
        """1 - fidelity 及其参数位移梯度"""
        f = fidelity(theta, psi_t, L, n)
        g = np.zeros_like(theta)
        for i in range(len(theta)):
            dp = np.zeros_like(theta)
            dp[i] = np.pi / 2
            fp = fidelity(theta + dp, psi_t, L, n)
            fm = fidelity(theta - dp, psi_t, L, n)
            g[i] = (fm - fp) / 2.0   # d(1-f)/dθ = -df/dθ
        return 1.0 - f, g

    best_t, best_f = None, 0.0
    rng = np.random.default_rng(seed)
    for trial in range(3):
        theta = rng.uniform(-np.pi, np.pi, L * 2 * n)
        # Adam 预热
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for it in range(steps):
            f = fidelity(theta, psi_t, L, n)
            g = np.zeros_like(theta)
            for i in range(len(theta)):
                dp = np.zeros_like(theta)
                dp[i] = np.pi / 2
                fp = fidelity(theta + dp, psi_t, L, n)
                fm = fidelity(theta - dp, psi_t, L, n)
                g[i] = (fp - fm) / 2.0
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            mh = m / (1 - beta1 ** (it + 1))
            vh = v / (1 - beta2 ** (it + 1))
            theta += lr * mh / (np.sqrt(vh) + eps)
        # L-BFGS-B 精修
        res = minimize(loss_grad, theta, method='L-BFGS-B', jac=True,
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
        f = 1.0 - res.fun
        if f > best_f:
            best_t, best_f = res.x.copy(), f
        print(f"    L={L} trial {trial+1}  fid={f:.7f}  best={best_f:.7f}")
    return best_t, best_f

def verify_pyqpanda(theta, U, R, L):
    """用 pyqpanda 构建变分编码电路, 模拟得到分布, 与目标态分布对比"""
    xs, ys = sm.grid_4x4()
    psi_t = target_state(U, R)
    ref = np.abs(psi_t) ** 2
    qvm = init_quantum_machine(QMachineType.CPU)
    pos = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)
    prog = QProg()
    p = 0
    for l in range(L):
        for q in range(4):
            prog << RZ(pos[q], theta[p])
            prog << RY(pos[q], theta[p + 1])
            p += 2
        for q in range(3):
            prog << CNOT(pos[q], pos[q + 1])
    for i in range(4):
        prog << Measure(pos[i], c[i])
    res = qvm.run_with_configuration(prog, c, 50000)
    qm = sm.decode_probs(res)
    f = sm.hellinger(qm, {i: float(v) for i, v in enumerate(ref)})
    finalize()
    return f

def main():
    CASES = [(1.0, 0.5), (2.0, 0.5), (4.0, 0.5), (6.0, 0.5),
             (2.0, 1.0), (4.0, 1.0), (1.0, 1.0)]
    out = {}
    for U, R in CASES:
        psi_t = target_state(U, R)
        print(f"训练 (U={U}, R={R}):")
        # 先试 L=3 (9 CNOT, 实测可达 fidelity=1.0), 不足则 L=4
        best = None
        for L in [3, 4]:
            theta, f = train(L, psi_t, steps=300, seed=20260804)
            print(f"  -> L={L} 最终 fidelity={f:.7f}")
            if best is None or f > best[2]:
                best = (L, theta, f)
            if f > 0.999999:
                break
        L, theta, f = best
        f_pq = verify_pyqpanda(theta, U, R, L) if HAS_PYQPANDA else -1.0
        print(f"  [选定] L={L}  numpy fid={f:.6f}  pyqpanda Hellinger={f_pq:.6f}")
        key = f"{U:.2f}_{R:.2f}"
        out[key] = {"L": L, "theta": [float(t) for t in theta],
                    "numpy_fidelity": float(f), "pyqpanda_hellinger": float(f_pq)}
        with open(os.path.join(_HERE, 'vqe_encoder_params.json'), "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=1)
    print("全部完成, 参数已保存 vqe_encoder_params.json")
    return out

if __name__ == "__main__":
    main()
