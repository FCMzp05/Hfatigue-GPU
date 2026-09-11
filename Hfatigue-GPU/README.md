# Hfatigue-GPU

GPU-resident phase-field solvers for hydrogen-assisted fatigue. The package
contains the numerical source code and required meshes only. Figure notebooks,
post-processing utilities, manuscript files, and computed results are excluded.

## Contents

- `code/reproductions/cui2024/`: Q8 compact-tension solvers.
- `code/paper/`: SENT geometry-transfer and error-controlled block drivers.
- `code/reproductions/yang2026/`: P1 SENT operators.
- `code/reproductions/yang2026_jmrt_pipeline/`: welded-pipeline GPU solver.
- `outputs/`: CT and SENT meshes required by the default command-line options.
- `open_source_code/`: released welded-pipeline mesh required by the pipeline solver.

## Environment

Linux or WSL2, Python 3.12, an NVIDIA CUDA 12 driver, and an FP64-capable
NVIDIA GPU are recommended.

```bash
conda env create -f environment.yml
conda activate hfatigue-gpu
```

## Run

CT example:

```bash
python code/reproductions/cui2024/cui2024_fig6_paris_curves_ct_q8.py \
  --case p55_r01_f1 --platform gpu
```

SENT geometry-transfer options:

```bash
python code/paper/08_case_SENT_Cui_Q8_geometry_transfer.py --help
python code/paper/08_case_SENT_Yang_error_controlled_cycle_blocks.py --help
```

Prepare the welded-pipeline HAZ field and run one GPU block:

```bash
python code/reproductions/yang2026_jmrt_pipeline/solve_haz_map.py
python code/reproductions/yang2026_jmrt_pipeline/yang2026_pipeline_p1_fullgpu.py \
  --pressure-mpa 10 --blocks 1 --outdir outputs/pipeline_smoke
```

Each driver records command-line settings, numerical results, and runtime
metadata in its output directory. Run `python <driver> --help` for all options.

## Reference

If this code is used in published work, cite the associated article:
*Time-consistent cycle-accelerated phase-field modelling of
hydrogen-assisted fatigue*.
