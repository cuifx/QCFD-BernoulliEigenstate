# -*- coding: utf-8 -*-
"""Hardware runs on the WK_C180_2 superconducting processor: QFT propagator.

Two experiments:

  A) momentum. H, phase encoding e^{i Phi}, QFT, measure. The spectrum is
     sharply peaked (dominant mode 50-96% of the probability), so the
     identification is unambiguous; this tests the phase encoding and the QFT.

  B) position. The full propagator: H, phase encoding, QFT, kinetic step
     D(k^2), inverse QFT, measure, compared with the classical FFT reference.
     At small t the distribution is close to flat, so this run mainly tests
     that the density is preserved.

Usage:
  python run_sm_real.py momentum               all parameter sets (saved incrementally)
  python run_sm_real.py position <t> <U> <R>   one set of the full propagator
  python run_sm_real.py position all           every parameter set
"""
import os, sys

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path

from qcloud_key import get_api_key
import builtins, collections, re, time, json, os, sys, math, itertools, random
import numpy as np
from pyqpanda import *
from pyqpanda import convert_qprog_to_originir as old_conv
from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog, convert_qprog_to_originir as new_conv
from pyqpanda3.transpilation import Transpiler
import sm_qft_circuit as sm

_DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # real_device/
_HERE = os.path.join(_DEV, 'data')  # results / calibration files

CHIP_NAME = "WK_C180_2"
SHOTS = 4000
N_LOGICAL = 4

d = json.load(open(os.path.join(_HERE, 'cnot_fidelity_measured_wk2_cloud17.json')))
FID = {tuple(map(int, k.split('_'))): v for k, v in d['edges'].items()}
READ = {int(k): v for k, v in d['single'].items()}
info = json.load(open(os.path.join(_HERE, 'wk2_chip_info_now.json')))
AVAIL = info['available']
ALL_EDGES = [tuple(sorted(e)) for e in info['edges']]
print(f"云端: {len(AVAIL)} 比特, {len(ALL_EDGES)} 边")

FALLBACK = 0.70

def build_adj(vertices, edges):
    adj = {v: [] for v in vertices}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj

def connected(cl, adj):
    s = set(cl)
    stack, vis = [cl[0]], {cl[0]}
    while stack:
        q = stack.pop()
        for nb in adj.get(q, []):
            if nb in s and nb not in vis:
                vis.add(nb)
                stack.append(nb)
    return len(vis) == len(s)

def choose_cluster(k=4):
    """在 17 比特中选连通 k 比特簇, 评分 = (边数, 边保真乘积, 读出和)"""
    adj = build_adj(AVAIL, ALL_EDGES)
    best = None
    for comb in itertools.combinations(AVAIL, k):
        cl = list(comb)
        if not connected(cl, adj):
            continue
        ce = [e for e in ALL_EDGES if e[0] in cl and e[1] in cl]
        prod = 1.0
        for e in ce:
            prod *= FID.get(e, FALLBACK)
        read_sum = builtins.sum(READ.get(q, 0.85) for q in cl)
        score = (len(ce), prod, read_sum)
        if best is None or score > best[0]:
            best = (score, cl, ce)
    _, cl, ce = best
    print(f"{k} 比特簇: {sorted(cl)}  边数: {len(ce)}")
    return cl, ce

def score_oir(oir_t, mapping):
    cost = 0.0
    cnt = 0
    last = {}
    depth = 0
    for l in oir_t.split('\n'):
        s = l.strip()
        if not s or s.startswith('//') or s.startswith('QINIT') or s.startswith('CREG'):
            continue
        m = re.match(r'CNOT q\[(\d+)\],q\[(\d+)\]', s)
        if m:
            edge = tuple(sorted((int(m.group(1)), int(m.group(2)))))
            p = FID.get(edge, FALLBACK)
            cost -= math.log(max(p, 1e-6))
            cnt += 1
        qs = [int(mm) for mm in re.findall(r'q\[(\d+)\]', s)]
        if not qs:
            continue
        e = max((last.get(q, 0) for q in qs), default=0)
        layer = e + 1
        for q in qs:
            last[q] = layer
        depth = max(depth, layer)
    for lq in range(N_LOGICAL):
        pr = READ.get(mapping[lq], 0.85)
        cost -= math.log(max(pr, 1e-6))
    return cost, cnt, depth

def sim_check(oir_t, cl):
    """物理比特重编号为逻辑比特后本地模拟, 返回测量概率 dict{idx:prob}"""
    phys_to_sim = {p: i for i, p in enumerate(sorted(cl))}
    oir_sim = re.sub(r'q\[(\d+)\]', lambda m: f'q[{phys_to_sim[int(m.group(1))]}]', oir_t)
    oir_sim = re.sub(r'QINIT \d+', f'QINIT {N_LOGICAL}', oir_sim)
    qvm = init_quantum_machine(QMachineType.CPU)
    ret = convert_originir_str_to_qprog(oir_sim, qvm)
    prog2, _, cvec = ret
    res = qvm.run_with_configuration(prog2, cvec, 20000)
    finalize()
    return sm.decode_probs(res)

def search_mapping(cl, ce, prog_sm, iters=100, hill=20):
    """在簇上搜索最优逻辑->物理映射 (逻辑 0..3)"""
    tr = Transpiler()
    random.seed(20260804)

    def eval_map(mapping, attempts=1):
        best = None
        for _ in range(attempts):
            tp = tr.transpile(prog_sm, ce, mapping, 2, ['CNOT', 'U3', 'X'])
            cost, cnt, depth = score_oir(new_conv(tp), mapping)
            if best is None or (cost, cnt, depth) < best:
                best = (cost, cnt, depth)
        return best

    best = None
    for _ in range(iters):
        perm = cl[:]
        random.shuffle(perm)
        mapping = {l: p for l, p in enumerate(perm)}
        r = eval_map(mapping)
        if best is None or r < (best[0], best[1], best[2]):
            best = r + (mapping,)

    cur = best
    rounds = 0
    while rounds < hill:
        improved = False
        for x in range(N_LOGICAL):
            for y in range(x + 1, N_LOGICAL):
                nm = dict(cur[3])
                nm[x], nm[y] = nm[y], nm[x]
                r = eval_map(nm)
                if r < (cur[0], cur[1], cur[2]):
                    cur = r + (nm,)
                    improved = True
        if not improved:
            break
        rounds += 1
    cost, cnt, depth, mapping = cur
    print(f"最优映射: cost={cost:.3f} 编译CNOT={cnt} 深度={depth}")
    print(f"  物理比特: {[mapping[l] for l in range(N_LOGICAL)]}")
    return mapping

def build_transpiled_oir(builder, args, cl, ce, mapping, ref):
    """构建 SM 电路并编译. builder: sm.build_sm_circuit 或 build_sm_circuit_momentum
    返回 (编译后 OriginIR, 编译前CNOT, 编译后CNOT)"""
    tr = Transpiler()
    xs, ys = sm.grid_4x4()
    qvm = init_quantum_machine(QMachineType.CPU)
    prog, c = builder(qvm, xs, ys, *args)
    oir = old_conv(prog, qvm)
    n_cnot_pre = oir.count('CNOT ')
    finalize()
    prog_p3 = convert_originir_string_to_qprog(oir)
    best = None
    for _ in range(4):
        tp = tr.transpile(prog_p3, ce, mapping, 2, ['CNOT', 'U3', 'X'])
        oir_t = new_conv(tp)
        qm = sim_check(oir_t, cl)
        f = sm.hellinger(qm, ref)
        if f > 0.999:
            cost, cnt, depth = score_oir(oir_t, mapping)
            if best is None or (cnt, -cost) < best[0]:
                best = ((cnt, -cost), oir_t)
    if best is None:
        tp = tr.transpile(prog_p3, ce, mapping, 2, ['CNOT', 'U3', 'X'])
        best = ((10 ** 9, 0.0), new_conv(tp))
    oir_t = best[1]
    _, cnt, _ = score_oir(oir_t, mapping)
    return oir_t, n_cnot_pre, cnt

def make_service():
    from pyqpanda3.qcloud import QCloudOptions, QCloudService
    service = QCloudService(get_api_key())
    backend = service.backend(CHIP_NAME)
    options = QCloudOptions()
    options.set_mapping(False)
    options.set_optimization(False)
    options.set_amend(globals().get("AMEND", False))
    print(f"后端: {backend.name()}  Shots: {SHOTS}")
    return service, backend, options

def find_mapping(builder, args):
    """针对电路结构做一次映射搜索 (结构相同, 与参数无关)"""
    cl, ce = choose_cluster(4)
    xs, ys = sm.grid_4x4()
    qvm = init_quantum_machine(QMachineType.CPU)
    prog, _ = builder(qvm, xs, ys, *args)
    oir = old_conv(prog, qvm)
    finalize()
    prog_p3 = convert_originir_string_to_qprog(oir)
    mapping = search_mapping(cl, ce, prog_p3)
    return cl, ce, mapping

def save_probs(path, idx, params, probs, hell, top, exp, n_cnot):
    """增量保存完整测量分布到 JSON (供绘图用)"""
    data = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path, encoding='utf-8'))
        except Exception:
            data = {}
    data[str(idx)] = {"params": list(params), "probs": {str(k): float(v) for k, v in probs.items()},
                      "hell": float(hell), "top": int(top), "exp": int(exp), "cnot": int(n_cnot)}
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

def run_momentum():
    """动量谱实验: 7 组 (U, R), 分布尖锐, 主态匹配有意义"""
    cl, ce, mapping = find_mapping(sm.build_sm_circuit_momentum, (1.0, 0.5))
    service, backend, options = make_service()
    cases = [(1.0, 0.5), (2.0, 0.5), (4.0, 0.5), (6.0, 0.5),
             (2.0, 1.0), (4.0, 1.0), (1.0, 1.0)]
    out_path = os.path.join(_HERE, 'results_sm_momentum.txt')
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for ln in f:
                m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)", ln)
                if m:
                    done.add(int(m.group(1)))
    with open(out_path, "a", encoding="utf-8") as f:
        if not done:
            f.write("SM 动量谱真机实验 (WK_C180_2, 4x4网格, 4 qubit)\n")
            f.write(f"配置: 芯片 {CHIP_NAME}, Shots {SHOTS}, 簇 {sorted(cl)}\n")
            f.write(f"映射(逻辑->物理): {mapping}\n")
            f.write("-" * 72 + "\n")
            f.write(f"{'序号':>4}  {'U':>4}  {'R':>4}  {'Hellinger':>10}  {'主态':>6}  {'期望':>6}  {'匹配':>4}  {'CNOT':>4}\n")
            f.write("-" * 72 + "\n")
        for i, (U, R) in enumerate(cases):
            if i + 1 in done:
                continue
            ref = sm.classic_momentum_pipeline(*sm.grid_4x4(), U, R)
            ref_top = max(ref, key=ref.get)
            oir_t, n_pre, n_post = build_transpiled_oir(sm.build_sm_circuit_momentum, (U, R), cl, ce, mapping, ref)
            prog_p3 = convert_originir_string_to_qprog(oir_t)
            print(f"[{i+1}/{len(cases)}] 提交 U={U:.1f}, R={R:.2f} ...", end=" ", flush=True)
            job = backend.run(prog_p3, SHOTS, options)
            result = job.result()
            probs = result.get_probs()
            qm = {int(k, 2): v for k, v in probs.items()}
            f_hell = sm.hellinger(qm, ref)
            qm_top = max(qm, key=qm.get)
            is_match = (qm_top == ref_top)
            match_str = "OK" if is_match else "FAIL"
            print(f"Hellinger={f_hell:.4f} 主态={qm_top} 期望={ref_top} {match_str} job={job.job_id()}")
            tops = sorted(qm.items(), key=lambda kv: -kv[1])[:4]
            f.write(f"{i+1:>4}  {U:4.1f}  {R:4.2f}  {f_hell:10.4f}  {qm_top:>6}  {ref_top:>6}  {match_str:>4}  CNOT={n_post}\n")
            f.write("      Top4: " + ", ".join(f"{k}:{p*100:.2f}%" for k, p in tops) + "\n")
            f.flush()
            save_probs(os.path.join(_HERE, 'results_sm_momentum_probs.json'), i + 1, (U, R), qm, f_hell, qm_top, ref_top, n_post)
            time.sleep(0.5)
    print(f"结果已保存至: {out_path}")

def run_position():
    """完整传播子 (fig:quantum_circuit) 端到端实验"""
    cl, ce, mapping = find_mapping(sm.build_sm_circuit, (0.01, 1.0, 0.5))
    service, backend, options = make_service()
    if len(sys.argv) > 4:
        cases = [(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))]
    else:
        cases = [(0.01, 1.0, 0.5), (0.10, 1.0, 0.5), (0.50, 1.0, 0.5)]
    out_path = os.path.join(_HERE, 'results_sm_position.txt')
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for ln in f:
                m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", ln)
                if m:
                    done.add(int(m.group(1)))
    with open(out_path, "a", encoding="utf-8") as f:
        if not done:
            f.write("SM QFT传播子真机实验 (WK_C180_2, 4x4网格, 4 qubit)\n")
            f.write(f"配置: 芯片 {CHIP_NAME}, Shots {SHOTS}, 簇 {sorted(cl)}\n")
            f.write(f"映射(逻辑->物理): {mapping}\n")
            f.write("-" * 72 + "\n")
            f.write(f"{'序号':>4}  {'t':>6}  {'U':>4}  {'R':>4}  {'Hellinger':>10}  {'主态':>6}  {'期望':>6}  {'CNOT':>4}\n")
            f.write("-" * 72 + "\n")
        for i, (t, U, R) in enumerate(cases):
            if i + 1 in done:
                continue
            ref = sm.classic_qft_pipeline(*sm.grid_4x4(), t, U, R)
            ref_top = max(ref, key=ref.get)
            oir_t, n_pre, n_post = build_transpiled_oir(sm.build_sm_circuit, (t, U, R), cl, ce, mapping, ref)
            prog_p3 = convert_originir_string_to_qprog(oir_t)
            print(f"[{i+1}/{len(cases)}] 提交 t={t:.2f}, U={U:.1f}, R={R:.2f} ...", end=" ", flush=True)
            job = backend.run(prog_p3, SHOTS, options)
            result = job.result()
            probs = result.get_probs()
            qm = {int(k, 2): v for k, v in probs.items()}
            f_hell = sm.hellinger(qm, ref)
            qm_top = max(qm, key=qm.get)
            print(f"Hellinger={f_hell:.4f} 主态={qm_top} 期望={ref_top} job={job.job_id()}")
            tops = sorted(qm.items(), key=lambda kv: -kv[1])[:4]
            f.write(f"{i+1:>4}  {t:6.2f}  {U:4.1f}  {R:4.2f}  {f_hell:10.4f}  {qm_top:>6}  {ref_top:>6}  CNOT={n_post}\n")
            f.write("      Top4: " + ", ".join(f"{k}:{p*100:.2f}%" for k, p in tops) + "\n")
            f.flush()
            save_probs(os.path.join(_HERE, 'results_sm_position_probs.json'), i + 1, (t, U, R), qm, f_hell, qm_top, ref_top, n_post)
            time.sleep(0.5)
    print(f"结果已保存至: {out_path}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "momentum"
    if mode == "position":
        run_position()
    else:
        run_momentum()
