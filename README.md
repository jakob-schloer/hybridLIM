# HybridLIM

This repository includes the code of the paper
[A Hybrid Deep-Learning Model for El Niño Southern Oscillation in the Low-Data Regime](http://arxiv.org/abs/2412.03743).

Our hybrid deep-learning model combines the Linear Inverse Model (LIM) with a Long-Short Term Memory (LSTM) network.
The LSTM captures the residuals between linear LIM forecasts and target data.


## Install packages

1. Create your virtual environment, e.g. venv or conda
2. In root directory of repo, run `pip install -e .` to install all required packages, including `hyblim` as an editable package

## Structure of repository

```
├── data
├── hyblim
│   ├── data
│   ├── model
│   └── utils
├── models
├── plots
└── scripts
```


## Download data and create dataset

Download sea surface temperature and sea surface height from
- CESM2 at [NCAR](https://www.cesm.ucar.edu/community-projects/lens2)
- ORAS5 at [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5?tab=overview)

Run the respective scripts in `scripts/create_datasets` to crop the tropical Pacific, interpolate the data to a lat-lon grid, and compute the anomalies. These are stored as netcdf-files in `/data`.

1. `create_cesm2-picontrol.py` / `create_oras5_dataset.py` — build the SSTA/SSHA netcdf inputs.
2. `pca_to_file.py` — fit the EOFs (20 PCs SSTA, 10 PCs SSHA) on the training period and save them to `data/cesm2-picontrol/pca/`. Every downstream model loads these precomputed EOFs.


## Workflow: train and evaluate the models

The stages are **sequential and file-coupled** — each writes netcdf/checkpoints that the next reads. Run them in the order below. The CS-LIM hindcast must exist before the hybrid LIM-LSTM can be trained.

Data splits are fixed: `train = years 1–1500`, `val = 1500–1800`, `test = 1800–2000`. All models are evaluated on the **test set** at lead times `1 3 6 9 12 15 18 21 24` months. Each `eval_*` script writes per-gridpoint, per-time and Niño-index scores to the model's `metrics/` folder, which the plotting scripts consume.

| Step | Model | Train | Evaluate |
|---|---|---|---|
| 1 | **CS-LIM** (base linear model) | `scripts/lim/ensemble_hindcast.py -v ssta ssha -lim cslim` | `scripts/lim/eval_lim_hindcast.py -path <hindcast.nc> -datasplit test`; `scripts/lim/compute_optimals.py -v ssta ssha -lim cslim` (optimal-growth patterns for Figs 6 & 7) |
| 2 | **LIM-LSTM** (hybrid) — needs step 1 | `scripts/limlstm/train_limlstm.py -horiz 20 -hidden 32 -l 2 -film -loss crps -gamma 0.65` | `scripts/limlstm/eval_limlstm.py -path <model_dir> -datasplit test` |
| 3 | **LSTM** (PC-space baseline) | `scripts/lstm/train_lstm.py -hist 12 -horiz 20 -hidden 32 -l 2 -film -members 16 -loss crps -gamma 0.65` | `scripts/lstm/eval_lstm.py -path <model_dir> -data test` |
| 4 | **ConvLSTM** (grid-space baseline) | `scripts/convlstm/train_convlstm.py -hist 12 -horiz 20 -channels 256 -layers 2 -film -members 16 -loss crps -gamma 0.65` | `scripts/convlstm/eval_convlstm.py -path <model_dir> -datasplit test` |

Notes:
- The `<model_dir>` for the deep-learning models is auto-named from the hyperparameters (and prefixed with the SLURM job id); `train_*` prints the path it writes to.
- The **"skill vs. training-data length"** results (Fig. 1, Fig. 5c-d) come from re-running steps 1–4 with `-ntrain <n_months>` over `[600, 1200, 2400, 3600, 6000, 9000, 12000, 18000]`. Outputs go to `.../num_traindata/n_<n>/`.
- **Optimal-growth patterns** (needed for Figs. 6 and 7) are produced by `scripts/lim/compute_optimals.py`, which fits the CS-LIM and writes the optimal initial / evolved conditions (`optimal_{init,evolved}_{pc,map}.nc`) into the CS-LIM `metrics/` folder.

### Run everything on SLURM

`scripts/slurm/submit_workflow.py` submits all four model families with the paper hyperparameters and the correct job dependencies (LIM → LIM-LSTM, train → eval). Individual `scripts/slurm/submit_*_{train,eval}.py` launchers handle the `num_traindata` sweeps.

```bash
# Print the sbatch commands without submitting:
python scripts/slurm/submit_workflow.py --tag repro -d
# Submit all models:
python scripts/slurm/submit_workflow.py --tag repro
# Subset of models:
python scripts/slurm/submit_workflow.py --tag repro --models lim limlstm
```


## Plotting the manuscript figures

The figure scripts live in `scripts/plotting/`. **Run them from that directory** — they read the experiment registry `experiments.yaml` (which maps experiment names → model paths/colors) and use relative paths to the saved `metrics/`. Helpers are shared via `scripts/plotting/utils.py`.

| Figure | Script | Shows |
|---|---|---|
| Fig. 1 | `fig_skill_ntrain.py` | Niño4 ACC at a fixed lead vs. number of training years, one curve per model (CS-LIM, LIM-LSTM, LSTM). |
| Fig. 2 | `fig_nino_skill_lim.py` | Niño4 RMSESS and CRPSS over lead time for the LIM-version progression (ST-LIM → CS-LIM → CS-LIM ssta,ssha) and the LIM-LSTM, with persistence reference. |
| Fig. 3 | `fig_skill_map.py` | Spatial RMSE skill-score maps: absolute CS-LIM skill (SSTA/SSHA) and the LIM-LSTM − CS-LIM difference. |
| Fig. 4 | `fig_example_nino_frcst.py` | Example El Niño forecast: Niño4 trajectory (mean + spread) and τ=12 SSTA/SSHA maps for CS-LIM, LIM-LSTM and target. |
| Fig. 5 | `fig_nino_skill_baselines.py` | Niño4 RMSESS/CRPSS over lead time (full training set) and vs. training-data length for CS-LIM, LIM-LSTM, LSTM, ConvLSTM. |
| Fig. 6 | `fig_project_optimals.py` | Predictability via linear optimals: CS-LIM optimal initial/evolved patterns and ACC stratified by optimal-growth percentile for CS-LIM, LIM-LSTM, LSTM. |
| Fig. 7 | `fig_enso_asymmetry.py` | ENSO asymmetry composites (warm−cold, warm+cold) of initial and evolved states for CS-LIM, LIM-LSTM and LSTM. |
| Fig. A1 | `fig_nino_season_skill.py` | Appendix: seasonal (month × lead) RMSESS for several Niño indices and the LIM-LSTM − CS-LIM difference. |
