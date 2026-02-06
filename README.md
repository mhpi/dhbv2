<h1 align="center">δHBV2.0: Differentiable Rainfall-Runoff Module</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9--3.13-blue?labelColor=333333" alt="Python"></a>
  <a href="https://pypi.org/project/dhbv2/"><img src="https://img.shields.io/pypi/v/dhbv2?logo=pypi&logoColor=white&labelColor=333333&cacheSeconds=60" alt="PyPI version"></a>
  <a href="https://pypi.org/project/torch/"><img src="https://img.shields.io/badge/dynamic/json?label=PyTorch&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorch%2Fjson&logo=pytorch&color=EE4C2C&logoColor=F900FF&labelColor=333333" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&labelColor=333333" alt="Ruff"></a>
  <a href="https://github.com/mhpi/dhbv2/actions/workflows/lint.yaml"><img src="https://img.shields.io/github/actions/workflow/status/mhpi/dhbv2/lint.yaml?branch=master&logo=github&label=lint&labelColor=333333" alt="Build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Non--Commercial_(PSU)-yellow?labelColor=333333" alt="License"></a>
</p>

---

**δHBV 2.0** is a state-of-the-art, distributed differentiable HBV model leveraging intelligent parameterization, big data, and highly-parallelized GPU compute with PyTorch to deliver CONUS-scale, high-resolution inference of parameters and fluxes.

This repository serves as an **operations-level module** for NOAA-OWP’s [Next Generation National Water Modeling Framework (NextGen)](https://github.com/NOAA-OWP/ngen). It provides **Basic Model Interface (BMI)** adapters for two modeling modalities:

1. **δHBV 2.0**: Daily timestep simulation.
2. **δHBV 2.0 MTS**: Hourly timestep simulation using a Multi-TimeScale (MTS) architecture.

</br>

## Installation

```bash
uv pip install dhbv2
```

For operational deployment and NextGen setup, see [docs](https://github.com/mhpi/dhbv2/tree/master/docs).

</br>

## Model Descriptions

> Models are built on the generic differentiable modeling framework [δMG](https://github.com/mhpi/generic_deltamodel).

### 1. δHBV 2.0 (Daily)

*First introduced by Song et al. (2024) [[1]](#publications).*

The daily model uses an LSTM and MLP to learn parameters for the differentiable physical model HBV 2.0. Weather forcings (precipitation, temperature, PET) and static catchment attributes are used as inputs to simulate hydrological states and fluxes:

$$
    \begin{align}
    \theta_{d, m}^{1:t} &= \text{LSTM}( x_m^{1:t}, A_m ) \\
    \theta_{s, m} &= \text{MLP}( A_m ) \\
    Q_k^{1:t}, S_k^{1:t} &= \text{HBV}(x_m^{1:t}, \theta_{d, m}^{1:t}, \theta_{s, m})
    \end{align}
$$

where:

* $\theta$: Learned dynamic ($d$) and static ($s$) parameters.
* $x_m, A_m$: Forcings and attributes for unit basin $m$.
* $Q, S$: Model fluxes (e.g., streamflow) and states (e.g., snowpack).

### 2. δHBV 2.0 MTS (Hourly)

*Introduced by Yang et al. (2025) [[2]](#publications).*

The **Multi-TimeScale (MTS)** variant adapts the architecture for hourly simulation. It incorporates a rolling window input caching mechanism to bridge the gap between long-term hydrologic memory and high-frequency forcing:

* **Caching:** Caches ~351 days of aggregated daily inputs and ~7 days of hourly inputs.
* **Warmup:** Performs warmup steps using the cache to prime low-frequency (daily) and high-frequency (hourly) model states before generating hourly predictions.
* **Rolling Window**: After 7 days of hourly simulation, the cache window shifts forward 7 days and the warmup is repeated.

> Note: To run a simulation in NextGen for a given time period, the **prior 358 days** of forcing data must be included in the input to satisfy warmup described above.
>
> E.g., simulations starting 01/01/2009 01:00 require an input dataset timeseries starting at 01/08/2008 01:00.

</br>

## NextGen Configuration

To use these models in NextGen, reference the specific class in your realization configuration.

### Daily Simulation

Use `dhbv2.bmi.DeltaModelBmi`.

```json
{
    "time_step": 86400,
    "tag": "ngen_dhbv",
    "formulation": {
        "params": {
            "python_type": "dhbv2.bmi.DeltaModelBmi",
            "model_type_name": "dhbv2.0",
            "init_config": "/path/to/bmi_config.yaml"
        }
    }
}
```

### Hourly (MTS) Simulation

Use `dhbv2.mts_bmi.MtsDeltaModelBmi`.

```json
{
    "time_step": 3600,
    "tag": "ngen_dhbv_mts",
    "formulation": {
        "params": {
            "python_type": "dhbv2.mts_bmi.MtsDeltaModelBmi",
            "model_type_name": "dhbv2.0 mts",
            "init_config": "/path/to/mts_bmi_config.yaml"
        }
    }
}
```

</br>

## Operational Deployment

See [docs](https://github.com/mhpi/dhbv2/tree/master/docs) for installation and runtime instructions.

</br>

## Repository Organization

This package is designed to be installed as a Python dependency or placed in NextGen's `extern/` directory.

```text
src/dhbv2/
├── bmi.py          # Daily BMI adapter
├── mts_bmi.py      # Hourly (MTS) BMI adapter
├── pet.py          # Utility for PET calculations
└── utils.py        # Shared utilities
```

## Publications

1. Song, Y., Bindas, T., Shen, C., Ji, H., Knoben, W. J. M., Lonzarich, L., Clark, M. P., et al. "High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning." Water Resources Research (2025). <https://doi.org/10.1029/2024WR038928>

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @article{shen2023differentiable,
    title = {High-Resolution National-Scale Water Modeling Is Enhanced by Multiscale Differentiable Physics-Informed Machine Learning},
    author = {Song, Yalan and Bindas, Tadd and Shen, Chaopeng and Ji, Haoyu and Knoben, Wouter J. M. and Lonzarich, Leo and Clark, Martyn P. and Liu, Jiangtao and van Werkhoven, Katie and Lamont, Sam and Denno, Matthew and Pan, Ming and Yang, Yuan and Rapp, Jeremy and Kumar, Mukesh and Rahmani, Farshid and Thébault, Cyril and Adkins, Richard and Halgren, James and Patel, Trupesh and Patel, Arpita and Sawadekar, Kamlesh Arun and Lawson, Kathryn},
    journal = {Water Resources Research},
    volume = {61},
    number = {4},
    pages = {e2024WR038928},
    year={2025},
    doi = {https://doi.org/10.1029/2024WR038928},
    url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024WR038928}
    }
    ```
    </details>

    </br>

2. Yang, W., Ji, H., Lonzarich, L., Song, Y., Lawson, K., Shen, C. (2025). **[In Review]**

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for details.

---

*Please submit an [issue](https://github.com/mhpi/dhbv2/issues) to report any questions, concerns, or bugs.*
