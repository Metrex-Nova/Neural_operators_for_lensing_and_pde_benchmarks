# BALANI-NO / Neural Operators for Lensing and PDE Benchmarks

Code for a GSoC project spanning one ICLR submission and (currently) one
ML4PS workshop paper, built around a common family of neural-operator
architectures (BALANI-NO, FNO, ALNO, LNO, PDNO, UNO, U-FNO, HC-UNO, CNN,
DeepONet, POD-DeepONet) evaluated across Darcy flow, Navier-Stokes
vorticity, and gravitational lensing tasks.

- **ICLR submission**: the 11-architecture comparison across Darcy,
  Navier-Stokes, and two lensing variants (full-field and subhalo-only).
- **ML4PS workshop paper** ("Pooling Discards Resolution-Invariant
  Structure"): a separate, narrower study — CNN/FNO/U-FNO/UNO/HC-UNO plus a
  no-pooling ablation of UNO/HC-UNO, on Darcy only.
- A DeepONet branch/trunk ablation study (7 variants + an FNO baseline) is
  also included; see the file table below for which paper each script maps to.

Two more ML4PS workshop papers (deflection-to-convergence reconstruction,
and the DeepONet-variant lensing comparison this repo's
`train_lensing_deeponet_variants.py` partially covers) exist but their code
isn't fully reflected here yet — see "Not included" below.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for a full 7-seed x 11-architecture run in reasonable
time; each script also runs on CPU (slower) with no code changes.

## Files

| File | What it reproduces |
|---|---|
| `train_darcy.py` | Downloads Darcy, trains/evals all 11 architectures, 7 seeds, @16/@32 |
| `train_ns.py` | Downloads Navier-Stokes, trains/evals all 11 architectures, 7 seeds, @32/@64 |
| `train_lensing.py` | Trains/evals all 11 architectures on lensing, 7 seeds, WDM/CDM/Axion. Set `LENSING_VARIANT = "sub"` or `"full"` at the top to switch tasks |
| `train_lensing_deeponet_variants.py` | Separate ablation: 7 DeepONet branch/trunk variants (Vanilla, Stacked, Conv, Fourier, Attention, POD, BelNet) + an FNO baseline row, 3 seeds, rel_L2 + SSIM |
| `train_darcy_pooling_ablation.py` | Workshop paper: CNN, FNO, U-FNO, UNO, HC-UNO + no-pool ablation of UNO/HC-UNO, 3 seeds, Darcy @16/@32. Separate architecture set from the 11-arch comparison |

Each script is fully self-contained (download → load → train → eval) and
runs standalone, matching how it was originally validated as a single
Kaggle-notebook cell.

## Reproducing each table

```bash
python train_darcy.py      # writes darcy_all11_7seed_summary.csv
python train_ns.py         # writes ns_all11_7seed_summary.csv

LENSING_DATA_ROOT=/path/to/modela python train_lensing.py
# edit LENSING_VARIANT ("sub" / "full") at the top and re-run for the other lensing table

python train_lensing_deeponet_variants.py   # writes all7_summary_table.csv
```

`LENSING_DATA_ROOT` must point at a directory containing `val/` and `test/`
subfolders, each holding per-class subfolders (`wdm/`, `cdm/`, `axion/`) of
`.npz` files with `kappa`/`psi` (full) or `kappa_sub`/`psi_sub` (subhalo)
arrays. `train_lensing_deeponet_variants.py` instead expects a
`manifest.csv`-indexed dataset (see its data-loading section) and is not
compatible with the same `LENSING_DATA_ROOT` layout.

## Seeds

The 11-architecture tables (Darcy, NS, both lensing variants) use the same 7
seeds: `13, 42, 97, 7, 64, 128, 2024`. The DeepONet-variant ablation uses 3
seeds: `0, 1, 2`.

## What's in `configs/hyperparameters.md`

A table of everything that differs *across* datasets (batch size, optimizer,
`FMAX` formula, the two different `evaluate()` conventions used by Darcy/NS
vs. lensing, and how the DeepONet-variant study's data pipeline differs from
the main lensing pipeline) — read this before comparing numbers across
tasks, since some of these differences change what a given metric actually
measures.

## Reproducibility notes

- Model selection (best-checkpoint) uses only the in-distribution validation
  set for every architecture and every dataset; zero-shot splits (Darcy @32,
  NS @64, lensing CDM/Axion) are never used for early stopping.
- Every architecture within the 11-architecture comparison is
  parameter-matched to BALANI-NO's parameter count (~112K) per dataset; see
  the `MODEL_ZOO` list at the top of each script for exact configs.
- Full per-seed results (not just mean ± std) are saved in the `.pt`
  checkpoint files each script writes alongside its summary CSV.

## Not included in this repo

- Trained model checkpoints (`*.pt`) — too large; excluded via `.gitignore`.
- Raw datasets — downloaded automatically by `train_darcy.py`/`train_ns.py`/
  `train_darcy_pooling_ablation.py` (all use `neuraloperator`'s built-in
  Darcy loaders); lensing data is not publicly redistributed here.
- A standalone exploratory FNO-only lensing script (superseded by
  `train_lensing.py`'s FNO entry) is intentionally left out as redundant.
- An earlier draft of the Darcy pooling-ablation code (relative-frequency
  filter bank, no no-pool ablation, only 5 architectures) is superseded by
  `train_darcy_pooling_ablation.py` and left out.
- Code for the deflection-to-convergence workshop paper (α→κ, the reverse
  of `train_lensing.py`'s task) and the classification-probe / hybrid
  POD+FNO experiments from the DeepONet-variant paper are not yet in this
  repo — ask if these are needed.

## Known issue to resolve before camera-ready

`train_darcy_pooling_ablation.py`'s paper states results are "mean ± std
over 5 seeds," but the code's `SEEDS` list has 3 entries (`[7, 21, 42]`).
Either the paper text needs correcting to 3, or 2 more seeds need to be run
and the table regenerated.
