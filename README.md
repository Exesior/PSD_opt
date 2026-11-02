# PSD_opt — WMCPBE wet granulation (branch `dev_eric`)

This branch holds the code of a master's thesis on the Monte Carlo simulation of
wet granulation in a high-shear mixer (Eirich EL1) and the fit of its kernel
parameters against measured sieve data. It builds on the `PSD_opt` framework of
the group (`pdhs-group/PSD_opt`); the `main` branch is the group's earlier state
and does not contain this work.

## Contents

| Folder | What it is |
|---|---|
| `mcpbe/` | Monte Carlo solvers. `src/wmcpbe/` is the **weighted solver for wet granulation** used in the thesis (nucleation, agglomeration, breakage, liquid and porosity processes, particle merger, exchangeable kernels in `kernels/`). Documentation in `mcpbe/docs/`, regression and conservation tests in `mcpbe/tests/`. |
| `mcpbe/studies/operator_splitting/` | Diagnosis of the operator-splitting error on the production case. |
| `pbe-optimizer/` | Ray-Tune-based parameter optimization; `adapter/wet_granulation_adapter.py` connects it to `wmcpbe`. |
| `pbe-core/`, `dpbe/` | Shared base classes and functions, discrete PBE solver (required by `pbe-optimizer`). |
| `studies/optimization/` | Setup of the kernel-parameter fit: configs for the cost functions KF1–KF4, runner, result export, data preparation (`dataprep/`) and the processed measurement data (`data/`). |
| `studies/sensitivity/` | Global sensitivity study (Sobol indices) of the fitted model, including calibration. |
| `studies/merger/` | Study of the particle-merger tolerance. |
| `lmc/`, `breakage-rate-model/`, `agggenerator/`, `qmom/`, `scripts/`, `docs/` | Further packages and scripts of the group framework, not specific to this thesis. |

## Installation

Python 3.11 (`>=3.9,<3.12`). From the repository root:

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pip install -e ./pbe-core -e ./mcpbe -e ./dpbe -e ./pbe-optimizer
```

## Running

Tests (from `mcpbe/`, details in `mcpbe/tests/README.md`):

```bash
python -m pytest tests -v
```

Fit for one cost function (from the repository root; without `--config` the
runner uses `opt_config_wet_granulation_full.py`):

```bash
python studies/optimization/run_wet_granulation_fit.py --config opt_config_wet_granulation_kf3_nosampling_logw.py
```

Sensitivity study: the start files `studies/sensitivity/start_*.py` are run one
after another (Spyder, F5), see `studies/sensitivity/README.md`.

Merger tolerance study: steps and arguments are described at the top of
`studies/merger/merger_tol_test.py` (it takes its candidate parameter set from
a fit history `.sqlite`).

Operator-splitting diagnosis (from `mcpbe/`):

```bash
python -m studies.operator_splitting.diagnose_splitting --check-identical
```

## Data

`studies/optimization/data/` contains the processed sieve data (Q3 at the sieve
meshes for KF1–KF4, x(Q) for AJ30) that the fit, the sensitivity study and the
merger study read. The raw measurement files are not part of the repository; they
are only needed to regenerate `data/` with the scripts in
`studies/optimization/dataprep/`.

Results of the runs (SQLite histories, Ray Tune output, study results) are not
included either.
