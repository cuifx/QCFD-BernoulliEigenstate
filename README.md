# Quantum stationary encoding of potential flow past streamlined bodies

Code and data supporting

> F. Cui, P. Bai, J. Yang, K. Cao, J. Kang, L. Xu,
> *Quantum stationary encoding of potential flow past streamlined bodies via
> the Bernoulli potential* (submitted).

The repository contains the scripts that produce the numerical results and the
figures of the paper, grouped by function:

| directory | contents |
| --- | --- |
| `common/` | shared libraries: Madelung encode/decode, the algebraic mapped polar grid, the Bernoulli operators, the constant-strength source panel method |
| `mapping/` | Theodorsen conformal mapping of the NACA sections and the Laurent coefficients |
| `figures/` | one script per manuscript figure |
| `analysis/` | scripts that back numbers quoted in the text and in the tables |
| `real_device/` | four-qubit superconducting runs: circuit construction, submission, figure |
| `tools/` | panel splitting and small utilities |
| `data/` | cached coefficients needed by the figure scripts |

`_paths.py` puts every subdirectory on `sys.path`; each script imports it
first, so a script runs from any working directory.

## Requirements

```
python >= 3.9
numpy >= 1.24
scipy >= 1.9
matplotlib >= 3.6
qiskit >= 1.0        # figures/airfoil_qiskit_mapped.py, analysis/circuit_{18,24}q_3d.py
qiskit-aer >= 0.13
pymupdf              # tools/split_panels_fig78.py, tools/split_sm_panels.py
```

The hardware module additionally needs the Origin Quantum stack
(`pyqpanda`, `pyqpanda3`) and access to their cloud service. Only
`real_device/` imports it, so every figure and analysis script runs without it.

```bash
pip install -r requirements.txt
```

## Reproducing the figures

Each script writes into `fig/` at the repository root, which is where the
manuscript expects the figure files to sit. The directory ships with a
`.gitkeep` placeholder; scripts that do not create it themselves rely on it
being present. Run a script directly from the repository root:

```bash
python figures/free_uniform_vs_body.py
python figures/spectral_stationarity.py
```

| output file (as `\includegraphics`) | script |
| --- | --- |
| `fig/free_uniform_vs_body_{a,b}.pdf` | `figures/free_uniform_vs_body.py` |
| `fig/operator_matching_convergence.pdf` | `figures/operator_matching_convergence.py` |
| `fig/circulation_quantized_{a,b}.pdf` | `figures/circulation_quantization.py` |
| `fig/circulation_remedy_{a,b}.pdf` | `figures/circulation_gauge_remedy_map.py`, then `tools/split_panels_fig78.py` |
| `fig/perturbation_fields_{a,b}.pdf` | `figures/perturbation_acoustic.py` |
| `fig/bernoulli_stationarity_{a,b,c}.pdf` | `figures/spectral_stationarity.py` |
| `fig/panel_encoding_convergence.pdf` | `analysis/panel_vs_encoding_convergence.py` |
| `fig/airfoil_hb_validation.pdf` | `figures/airfoil_validation_composite.py` |
| `fig/airfoil_hb_surface_cp.pdf` | `figures/airfoil_hb_quantum.py` |
| `fig/3d_geom_{a,b,c}.pdf` | `analysis/3d_geometry_figure.py` |
| `fig/3d_stationarity_{a,b}.pdf` | `figures/body_of_revolution_3d.py` |
| `fig/airfoil_qiskit_validation.pdf` | `figures/airfoil_qiskit_mapped.py` |
| `fig/sm_real_momentum_{a,b}.pdf` | `real_device/sm/plot_sm_results.py`, then `tools/split_sm_panels.py` |
| `fig/sm_real_dispersion.pdf` | `real_device/sm/plot_dispersion_panel.py` |
| `fig/hull_field_{a,b}.pdf` | `analysis/hull_field_figure.py` |

The two gate-level circuit figures of the manuscript (`hb_circuit`,
`sm_circuits`) are typeset directly in the LaTeX source, not generated here.

## Numbers quoted in the text and tables

| quantity | script |
| --- | --- |
| airfoil error table (free-particle and $H_B$ columns) | `analysis/airfoil_error_table.py` |
| moment-matched versus three-point radial stencil on the wing | `analysis/airfoil_hb_stencil.py` |
| panel-method cylinder baseline, surface-tangential metric | `analysis/panel_cl_baseline.py` |
| Kutta circulation of the three mapped sections | `analysis/panel_cl_baseline.py` (`IDEAL_ANGLE_DEG`) |
| ellipsoid panel solve against the ellipsoidal harmonic | `analysis/ellipsoid_exact_vs_panel.py` |
| 18- and 24-qubit register verification, gate counts and depths | `analysis/circuit_18q_3d.py`, `analysis/circuit_24q_3d.py` |

## Data and caches

`data/laurent_coeffs.npz` holds the Laurent coefficients and mapping radii of
the three sections. It is a cache: `mapping/plot_laurent_convergence.py`
rebuilds it from the Theodorsen fit in about two minutes, and the rebuilt
values reproduce the stored ones (R = 0.276222 and 0.276442 for the two NACA
sections).

The three-dimensional figures read two caches that are too large to ship. They
are rebuilt by `analysis/build_phi_256.py` (the 256^3 line-source potential,
about 90 min) and `analysis/circuit_surface_compare.py` (the surface shell and
the 24-qubit decode).

`real_device/data/` holds the measured four-qubit distributions in JSON and
text form, so the hardware figures can be replotted without resubmitting jobs.

## Hardware runs

`real_device/sm/run_sm_real.py` and `run_dispersion_panel.py` submit jobs to
the Origin Quantum cloud. The API key is read from the environment:

```bash
export QCIRCUIT_API_KEY=...        # Windows: set QCIRCUIT_API_KEY=...
```

or from a local `real_device/qcloud_key.txt` containing the key on its first
non-comment line. That file is listed in `.gitignore` and is not part of the
repository. `real_device/sm/sm_qft_circuit.py` builds the circuit locally and
can be run without a key.

## Scope

Included are the classical preprocessing, the circuit construction, the
simulator verification and the hardware runs behind every figure and table of
the paper.

Two lines of work are not part of this repository. The incompressible Euler
solution used as the third solver in Section 6.2 was run with an external
flow solver and its inputs are not produced here. The exploratory scripts
used during development, one-off diagnostics and the quantum-Minmod hardware
line are also left out; they do not reproduce anything quoted in the paper.
