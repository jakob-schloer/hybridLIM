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
