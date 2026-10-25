# -*- coding: utf-8 -*-
"""Position-space panel at T=2: parameter choice and hardware execution.

Thin driver, reusing the mapping and compilation of run_sm_real.py.

Why the earlier panel could not work (classical argument, quantified in
probe_coherence_design.py, not shipped here): |psi0| = 1 puts the reference
position distribution exactly at flat, the momentum spectrum on 4x4 with
L = 2 sits in a single mode, and the free propagator is diagonal in k, so a
single-qubit observable is unchanged by the evolution. The only quantity that
sees the dispersion is the structure of the position distribution,
TV(q_T, uniform):

    (U=1, R=.5, T=2) 0.10   old parameters, same size as the readout noise
    (U=2, R=.5, T=2) 0.18
    (U=4, R=.5, T=2) 0.37

The panel therefore fixes T = 2, sweeps U = 1, 2, 4, and adds a T = 0 echo
point (structure zero by construction) plus a readout calibration.

Usage: python run_dispersion_panel.py          all
       python run_dispersion_panel.py U4       one case only
"""
import os
import sys
import json

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401

from qcloud_key import get_api_key  # noqa: E402
from pyqpanda3.qcloud import QCloudService  # noqa: E402

_DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.join(_DEV, 'data')

CHIP = "WK_C180_2"
_SHOTS = int(os.environ.get("DP_SHOTS", "20000"))
AMEND = os.environ.get("DP_AMEND", "0") == "1"


def refresh_chip():
    """把当天可用的比特与拓扑写回 wk2_chip_info_now.json (run_sm_real 在
    import 时读它, 所以必须先刷新再 import)."""
    service = QCloudService(get_api_key())
    info = service.backend(CHIP).chip_info()
    avail = sorted(info.available_qubits())
    edges = sorted(tuple(sorted(e)) for e in info.get_chip_topology())
    json.dump({"chip": CHIP, "available": avail, "edges": [list(e) for e in edges]},
              open(os.path.join(_HERE, 'wk2_chip_info_now.json'), 'w'), indent=1)
    print(f"当日可用比特 {len(avail)}: {avail}")
    return avail


CASES = {
    "echo": (0.0, 4.0, 0.5),   # 结构量恒为 0: 噪声底/读出锚点
    "U1": (2.0, 1.0, 0.5),
    "U2": (2.0, 2.0, 0.5),
    "U4": (2.0, 4.0, 0.5),
}
OUT = os.path.join(_HERE, 'results_dispersion_panel'
                   + ('_amend' if AMEND else '') + '.json')


def main():
    refresh_chip()
    import run_sm_real as R
    import sm_qft_circuit as sm
    from pyqpanda import (init_quantum_machine, QMachineType, QProg, X, Measure,
                          convert_qprog_to_originir, finalize)

    R.SHOTS = _SHOTS
    R.AMEND = AMEND
    res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    res.setdefault("setup", {})

    cl, ce = R.choose_cluster(4)
    service, backend, options = R.make_service()
    res["setup"].update({"chip": CHIP, "shots": _SHOTS, "cluster": sorted(cl),
                         "cases": {k: list(v) for k, v in CASES.items()}})
    json.dump(res, open(OUT, 'w'), indent=1)

    want = sys.argv[1:] or list(CASES)

    # 读出标定: |0000> 与 |1111>, 每个比特给 P(1|0), P(1|1)
    for tag in ["cal0", "cal1"]:
        if tag in res:
            continue
        def builder(qvm, xs, ys, tag=tag):
            pos = qvm.qAlloc_many(4)
            c = qvm.cAlloc_many(4)
            prog = QProg()
            if tag == "cal1":
                for q in pos:
                    prog << X(q)
            for i in range(4):
                prog << Measure(pos[i], c[i])
            return prog, c
        ref = {0: 1.0} if tag == "cal0" else {15: 1.0}
        _, _, mp = R.find_mapping(builder, ())
        oir_t, _, n_post = R.build_transpiled_oir(builder, (), cl, ce, mp, ref)
        from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog
        job = backend.run(convert_originir_string_to_qprog(oir_t), _SHOTS, options)
        pr = job.result().get_probs()
        res[tag] = {"probs": {str(k): float(v) for k, v in pr.items()},
                    "job": job.job_id(), "cnot": n_post}
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f"{tag}: job={job.job_id()} cnot={n_post}")

    from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog
    mapping = None
    for name in want:
        if name in res:
            print(f"{name} 已有结果, 跳过")
            continue
        t, U, Rv = CASES[name]
        ref = sm.classic_qft_pipeline(*sm.grid_4x4(), t, U, Rv)
        if mapping is None:
            _, _, mapping = R.find_mapping(sm.build_sm_circuit, (t, U, Rv))
        oir_t, _, n_post = R.build_transpiled_oir(sm.build_sm_circuit, (t, U, Rv),
                                                  cl, ce, mapping, ref)
        job = backend.run(convert_originir_string_to_qprog(oir_t), _SHOTS, options)
        pr = job.result().get_probs()
        res[name] = {"case": [t, U, Rv], "probs": {str(k): float(v) for k, v in pr.items()},
                     "job": job.job_id(), "cnot": n_post}
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f"{name} (t={t},U={U},R={Rv}): job={job.job_id()} cnot={n_post} "
              f"态数={len(pr)}")
    print("saved", OUT)


if __name__ == '__main__':
    main()
