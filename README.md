# HybridLIM

A hybrid model for ENSO forecasting.

The paper is still in preparation, and the code will be published with the paper. For a first draft of the paper, see: 
https://jakob-schloer.github.io/publications/24-01-25_hybrid_lim.pdf


## Install packages with poetry

1. If not already, install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. In root directory of repo, run `poetry install` to create your environment and install all required packages, including `hyblim` as an editable package

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
    └── create_datasets
```


## Download data and create dataset

Download sea surface temperature and sea surface height from
- CESM2 at [link]
- ORAS5 at [link]

Run the respective scripts in `scripts/create_datasets` to crop the tropical Pacific, interpolate the data to a lat-lon grid, and compute the anomalies. These are stored as netcdf-files in `/data`.

