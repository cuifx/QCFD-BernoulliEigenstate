# -*- coding: utf-8 -*-
"""QFT propagator circuit for the four-qubit hardware run (4x4 grid).

Circuit: uniform superposition, phase encoding as the diagonal unitary
diag(e^{i Phi_j}), QFT, diagonal kinetic step D(k^2), inverse QFT, measure.
The measured distribution is compared with a classical FFT reference.

The diagonal unitary is synthesized in Gray-code order (Shende et al., the
qiskit ucrz construction): 2^n RZ rotations and 2^n - 1 CNOTs, all between
neighbouring qubits.
"""
import math
import builtins
import os
import json
import numpy as np
import re

try:
    from pyqpanda import *
    HAS_PYQPANDA = True
except ImportError:
    HAS_PYQPANDA = False

_THIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_VQE_PATH = os.path.join(_THIS_DIR, "vqe_encoder_params.json")
_VQE_CACHE = None


def _load_vqe_params():
    """加载模拟机训练的硬件高效编码器参数 {key(U_R): {L, theta, ...}}"""
    global _VQE_CACHE
    if _VQE_CACHE is None:
        with open(_VQE_PATH, encoding="utf-8") as f:
            _VQE_CACHE = json.load(f)
    return _VQE_CACHE


def _vqe_key(U, R):
    """查找与 (U,R) 最接近的已训练参数键"""
    pars = _load_vqe_params()
    keys = [(k, float(k.split('_')[0]), float(k.split('_')[1])) for k in pars]
    best = min(keys, key=lambda t: abs(t[1] - U) + abs(t[2] - R))
    return best[0]


def apply_vqe_encoder(prog, qubits, U, R):
    """硬件高效变分编码: V_enc(θ*)|0...0⟩ ≈ (1/4)Σ_j e^{iΦ_j}|j⟩

    模拟机训练结果 (train_vqe_encoder.py), L 层:
      单比特层 RZ·RY 每比特 + 链状近邻 CNOT 0-1,1-2,2-3.
    替代 H⊗4 + 对角酉 (仅 9 CNOT, 且全部近邻, 编译零桥接).
    """
    pars = _load_vqe_params()
    key = _vqe_key(U, R)
    rec = pars[key]
    L = int(rec["L"])
    theta = rec["theta"]
    p = 0
    for l in range(L):
        for q in range(len(qubits)):
            prog << RZ(qubits[q], theta[p])
            prog << RY(qubits[q], theta[p + 1])
            p += 2
        for q in range(len(qubits) - 1):
            prog << CNOT(qubits[q], qubits[q + 1])


# ============ 经典参考 ============

def potential_cylinder(x, y, U_inf=1.0, R=0.5):
    """圆柱绕流速度势 Φ = U∞(r + R²/r)cosθ = U∞ x(1 + R²/(x²+y²))"""
    r2 = x * x + y * y
    return U_inf * x * (1.0 + R * R / r2)


def grid_4x4(L=2.0):
    """4x4 均匀网格, 坐标范围 [-L, L]"""
    xs = np.linspace(-L, L, 4)
    ys = np.linspace(-L, L, 4)
    return xs, ys


def classic_qft_pipeline(xs, ys, t=0.01, U_inf=1.0, R=0.5):
    """经典 FFT 传播子参考: 编码 e^{iΦ} → FFT → 对角动能 → 逆 FFT
    返回: 输出概率分布 dict{idx: prob}, idx 按 (x_idx, y_idx) 编码
    """
    n = len(xs)
    Phi = np.array([[potential_cylinder(x, y, U_inf, R) for y in ys] for x in xs])
    # 编码态: psi_0[j] = e^{iΦ_j} (对应 2D FFT 的行列索引)
    psi = np.exp(1j * Phi)
    # 2D FFT (按量子位约定: 索引顺序)
    psi_k = np.fft.fft2(psi)
    # 频率网格
    kx = np.fft.fftfreq(n, d=2 * 2.0 / n) * n  # 整数波数 -2..1 (未移位的 FFT 序)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    k2 = KX ** 2 + KY ** 2
    # 对角动能 e^{-i t k²/2}
    psi_k *= np.exp(-1j * t * k2 / 2.0)
    psi_t = np.fft.ifft2(psi_k)
    probs = np.abs(psi_t) ** 2
    probs /= probs.sum()
    # 编码为 dict{idx: prob}
    d = {}
    for ix in range(n):
        for iy in range(n):
            idx = ix * n + iy
            d[idx] = probs[ix, iy]
    return d


def classic_momentum_pipeline(xs, ys, U_inf=1.0, R=0.5):
    """Classical momentum spectrum |ifft2(e^{i Phi})|^2, in the same
    convention as the quantum QFT. Independent of time and of the kinetic
    step: it is fixed by the phase encoding alone."""
    n = len(xs)
    Phi = np.array([[potential_cylinder(x, y, U_inf, R) for y in ys] for x in xs])
    psi = np.exp(1j * Phi)
    psi_k = np.fft.ifft2(psi)
    probs = np.abs(psi_k) ** 2
    probs /= probs.sum()
    d = {}
    for ix in range(n):
        for iy in range(n):
            d[ix * n + iy] = probs[ix, iy]
    return d


# ============ 对角酉合成 (Gray-code / ucrz) ============

def _extract_rz(phi1, phi2):
    """返回 (phase, z_angle) 使 e^{i*phase}*Rz(z_angle) = diag(e^{iφ1}, e^{iφ2})"""
    phase = (phi1 + phi2) / 2.0
    z_angle = phi2 - phi1
    return phase, z_angle


def _dec_uc_rotations(angles, start, end, rev):
    """qiskit 算法: 计算均匀受控旋转的分解角度"""
    half = (end - start) // 2
    for i in range(start, start + half):
        a, b = angles[i], angles[i + half]
        if not rev:
            angles[i], angles[i + half] = (a + b) / 2.0, (a - b) / 2.0
        else:
            angles[i + half], angles[i] = (a + b) / 2.0, (a - b) / 2.0
    if half > 1:
        _dec_uc_rotations(angles, start, start + half, False)
        _dec_uc_rotations(angles, start + half, end, True)


def _ucrz(prog, angles, ctrl, target, eps=1e-10):
    """均匀受控 Rz: angles 长度 2^k, 对应 ctrl(长度 k) 的 2^k 个状态"""
    k = len(ctrl)
    if k == 0:
        if abs(angles[0]) > eps:
            prog << RZ(target, angles[0])
        return
    ang = list(angles)
    _dec_uc_rotations(ang, 0, len(ang), False)
    for i, a in enumerate(ang):
        if abs(a) > eps:
            prog << RZ(target, a)
        if i != len(ang) - 1:
            # i+1 的二进制 trailing zeros 数 = 控制比特索引
            s = bin(i + 1)[2:]
            q_idx = len(s) - len(s.rstrip('0'))
            prog << CNOT(ctrl[q_idx], target)
        else:
            prog << CNOT(ctrl[k - 1], target)


def apply_diagonal(prog, qubits, phases):
    """施加对角酉 diag(e^{i*phases}) 到 qubits (长度 2^n)"""
    n = len(qubits)
    N = 1 << n
    dp = list(phases)
    while N >= 2:
        angles_rz = []
        for i in range(0, N, 2):
            dp[i // 2], rz = _extract_rz(dp[i], dp[i + 1])
            angles_rz.append(rz)
        num_act = int(math.log2(N))
        if num_act == 0:
            break
        ctrl = qubits[n - num_act + 1: n]
        target = qubits[n - num_act]
        _ucrz(prog, angles_rz, ctrl, target)
        N //= 2


def _bits_of(x):
    b = 0
    while x:
        if x & 1:
            yield b
        x >>= 1
        b += 1


def apply_phase_poly(prog, qubits, phases, eps=1e-9):
    """相位多项式实现对角酉 (利用 Walsh 稀疏性, 建立-RZ-uncompute)

    对连续势场 (圆柱绕流 Φ 在 4x4 网格上关于 y=0 对称), Walsh 谱高度
    稀疏 (本例精确 4/15 非零): 两个单比特 mask 直接 RZ, 两个 3 比特
    parity 各 4 CNOT (建立 2 + uncompute 2), 总 8 CNOT < 通用 15 CNOT.
    每个 mask 独立处理 (CNOT 自逆, uncompute 后态还原), 无冲突.
    """
    n = len(qubits)
    N = 1 << n
    W = np.array([[1 if builtins.bin(i & j).count('1') % 2 == 0 else -1
                   for j in range(N)] for i in range(N)], dtype=float)
    theta = W @ np.array(phases, dtype=float) / N
    masks = [m for m in range(1, N) if abs(theta[m]) > eps]
    if not masks:
        return
    masks.sort(key=lambda x: builtins.bin(x).count('1'))
    for m in masks:
        bits = list(_bits_of(m))
        if len(bits) == 1:
            # 单比特奇偶性: 直接 RZ (该 qubit 尚未被修改)
            prog << RZ(qubits[bits[0]], theta[m])
        else:
            # 多比特: 把 parity(m) 汇聚到最高位 qubit → RZ → uncompute
            t = max(bits)
            for b in bits:
                if b != t:
                    prog << CNOT(qubits[b], qubits[t])
            prog << RZ(qubits[t], theta[m])
            for b in reversed(bits):
                if b != t:
                    prog << CNOT(qubits[b], qubits[t])


# ============ 量子电路 ============

def build_sm_circuit(qvm, xs, ys, t=0.01, U_inf=1.0, R=0.5):
    """SM QFT 传播子电路. 返回 (prog, c_reg, n_qubits)
    qubit 布局: q[0..3] = 位置比特 (4x4 网格)
    """
    n = len(xs)
    pos = qvm.qAlloc_many(4)          # 4 位置比特
    c = qvm.cAlloc_many(4)
    prog = QProg()
    # 1. 相位编码: 硬件高效变分线路 V_enc(θ*) (模拟机训练, 替代 H⊗4 + 对角酉)
    apply_vqe_encoder(prog, pos, U_inf, R)
    # 2. 2D QFT: 索引 j = ix*4+iy → q[2],q[3] 编码 ix, q[0],q[1] 编码 iy
    prog << QFT(pos[2:4])   # x 方向
    prog << QFT(pos[0:2])   # y 方向
    # 3. 对角动能 D(k²): 可分离 D(kx²+ky²)=D(kx²)⊗D(ky²)
    #    → 拆成两个 2-qubit 对角酉 (各 2²-1=3 CNOT, 共 6), 替代整体 4-qubit 合成 (15 CNOT)
    kvals = np.fft.fftfreq(4, d=1.0) * 4  # 0,1,2,-1 (FFT 序)
    kin_x = [-t * k * k / 2.0 for k in kvals]   # x 维 (pos[2:4])
    kin_y = [-t * k * k / 2.0 for k in kvals]   # y 维 (pos[0:2])
    apply_diagonal(prog, pos[2:4], kin_x)
    apply_diagonal(prog, pos[0:2], kin_y)
    # 4. 逆 2D QFT
    prog << QFT(pos[0:2]).dagger()
    prog << QFT(pos[2:4]).dagger()
    # 5. 测量
    for i in range(4):
        prog << Measure(pos[i], c[i])
    return prog, c


def build_sm_circuit_momentum(qvm, xs, ys, U_inf=1.0, R=0.5):
    """动量谱版本电路: 变分编码 → QFT → 测量 (无动能/逆QFT)
    测量得到的正是编码态 e^{iΦ} 的傅里叶谱, 分布尖锐、判别力强.
    """
    n = len(xs)
    pos = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)
    prog = QProg()
    apply_vqe_encoder(prog, pos, U_inf, R)
    # 2D QFT (索引 j = ix*4+iy)
    prog << QFT(pos[2:4])   # x 方向
    prog << QFT(pos[0:2])   # y 方向
    for i in range(4):
        prog << Measure(pos[i], c[i])
    return prog, c


def decode_probs(result, n=4):
    """结果 dict{bitstr: count} → 概率 dict{idx: prob}"""
    total = builtins.sum(result.values())
    d = {}
    for k, v in result.items():
        d[int(k, 2)] = v / total
    return d


def hellinger(p1, p2):
    """Hellinger fidelity f = [Σ_j sqrt(p1_j p2_j)]²"""
    keys = set(p1) | set(p2)
    s = builtins.sum(math.sqrt(p1.get(k, 0.0) * p2.get(k, 0.0)) for k in keys)
    return s * s


def local_verify(t=0.01, U_inf=1.0, R=0.5):
    """本地 CPUQVM 模拟, 与经典 FFT 对比"""
    xs, ys = grid_4x4()
    ref = classic_qft_pipeline(xs, ys, t, U_inf, R)
    qvm = init_quantum_machine(QMachineType.CPU)
    prog, c = build_sm_circuit(qvm, xs, ys, t, U_inf, R)
    res = run_with_configuration(prog, c, 20000)
    qm = decode_probs(res)
    f = hellinger(qm, ref)
    # 资源统计
    n_cnot = 0
    n_ry = 0
    n_rz = 0
    n_h = 0
    oir = convert_qprog_to_originir(prog, qvm)
    for ln in oir.split('\n'):
        s = ln.strip()
        if s.startswith('CNOT'):
            n_cnot += 1
        elif s.startswith('RY'):
            n_ry += 1
        elif s.startswith('RZ'):
            n_rz += 1
        elif s.startswith('H '):
            n_h += 1
    finalize()
    print("=" * 60)
    print("SM QFT 传播子本地验证 (4x4 网格, t=%.3f, U=%.1f, R=%.2f)" % (t, U_inf, R))
    print("=" * 60)
    print(f"Hellinger fidelity vs 经典 FFT: {f:.4f}")
    print(f"资源: CNOT={n_cnot}  RZ={n_rz}  H={n_h}  qubits=4")
    print(f"经典参考 Top5: {sorted(ref.items(), key=lambda kv:-kv[1])[:5]}")
    print(f"量子模拟 Top5: {sorted(qm.items(), key=lambda kv:-kv[1])[:5]}")
    return f, ref, qm


if __name__ == "__main__":
    if HAS_PYQPANDA:
        f, ref, qm = local_verify()
    else:
        print("pyqpanda 不可用, 仅运行经典参考")
        xs, ys = grid_4x4()
        ref = classic_qft_pipeline(xs, ys)
        print(sorted(ref.items(), key=lambda kv: -kv[1])[:5])
