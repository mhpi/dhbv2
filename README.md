<h1 align="center">δHBV2.0: Differentiable Rainfall-Runoff Module</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9--3.13-blue?labelColor=333333" alt="Python"></a>
  <a href="https://pypi.org/project/dhbv2/"><img src="https://img.shields.io/pypi/v/dhbv2?logo=pypi&logoColor=white&labelColor=333333" alt="PyPI"></a>
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

As this model is designed to operate at the daily timescale, forcings are aggregated at the end of every day to make a single daily prediction. This prediction is then distributed accross the proceeding 24 hours, at which point a new daily prediction is made.

Since HBV is a recurrent bucket-based model, it must be "warmed up" making simulations on a period of data just prior to the target simulation window so that it's states can be allowed to saturate. Incidentally, this process also initializes internal states of the parameterization LSTM.

> Note: To run a simulation in NextGen for a given time period, we require the **prior 365 days** of forcing data be included in the input to satisfy warmup described above.
>
> E.g., simulations starting 01/01/2009 00:00 require an input dataset timeseries starting at 01/01/2008 00:00.

### 2. δHBV 2.0 MTS (Hourly)

*Introduced by Yang et al. (2025) [[2]](#publications).*

The **Multi-TimeScale (MTS)** variant adapts the architecture for hourly simulation. It incorporates a rolling window input caching mechanism to bridge the gap between long-term hydrologic memory and high-frequency forcing:

* **Caching:** Caches ~351 days of aggregated daily inputs and ~7 days of hourly inputs.
* **Warmup:** Performs warmup steps using the cache to prime low-frequency (daily) and high-frequency (hourly) model states before generating hourly predictions.
* **Rolling Window**: After 7 days of hourly simulation, the cache window shifts forward 7 days and the warmup is repeated.

> Note: To run a simulation in NextGen for a given time period, the **prior 358 days** of forcing data must be included in the input to satisfy warmup described above.
>
> E.g., simulations starting 01/01/2009 00:00 require an input dataset timeseries starting at 01/08/2008 00:00.

</br>

## NextGen Configuration

To use these models in NextGen, reference the specific class in your realization configuration.

### Daily Simulation

Use `dhbv2.bmi.DeltaModelBmi`.

```json
{
    "time_step": 3600,
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

## Citation

1. Song, Y., Bindas, T., Shen, C., Ji, H., Knoben, W. J. M., Lonzarich, L., et al. (2025). High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning. Water Resources Research, 61, e2024WR038928. <https://doi.org/10.1029/2024WR038928>

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @article{https://doi.org/10.1029/2024WR038928,
        author = {Song, Yalan and Bindas, Tadd and Shen, Chaopeng and Ji, Haoyu and Knoben, Wouter J. M. and Lonzarich, Leo and Clark, Martyn P. and Liu, Jiangtao and van Werkhoven, Katie and Lamont, Sam and Denno, Matthew and Pan, Ming and Yang, Yuan and Rapp, Jeremy and Kumar, Mukesh and Rahmani, Farshid and Thébault, Cyril and Adkins, Richard and Halgren, James and Patel, Trupesh and Patel, Arpita and Sawadekar, Kamlesh Arun and Lawson, Kathryn},
        title = {High-Resolution National-Scale Water Modeling Is Enhanced by Multiscale Differentiable Physics-Informed Machine Learning},
        journal = {Water Resources Research},
        volume = {61},
        number = {4},
        pages = {e2024WR038928},
        keywords = {differentiable modeling, physics-informed machine learning, National Water Model, routing, Muskingum Cunge, multiscale training},
        doi = {https://doi.org/10.1029/2024WR038928},
        year = {2025},
    }
    ```

    </details>

</br>

2. Yang, W., Ji, H., Lonzarich, L., Song, Y., Shen, C. (2025). Diffusion-Based Probabilistic Modeling for Hourly Streamflow Prediction and Assimilation. arXiv. <https://arxiv.org/abs/2510.08488> **[Under Review]**

    <details>
    <summary>BibTeX</summary>

    ```bibtex
    @misc{yang2025diffusionbasedprobabilisticmodelinghourly,
          title={Diffusion-Based Probabilistic Modeling for Hourly Streamflow Prediction and Assimilation},
          author={Wencong Yang and Haoyu Ji and Leo Lonzarich and Yalan Song and Chaopeng Shen},
          year={2025},
          eprint={2510.08488},
          archivePrefix={arXiv},
          primaryClass={physics.geo-ph},
          url={https://arxiv.org/abs/2510.08488},
    }
    ```

    </details>

</br>

<table><tr>
<td><img src="./docs/images/ciroh.png" alt="CIROH Logo"></td>
<td>Funding for this project was provided by the National Oceanic & Atmospheric Administration (NOAA), awarded to the Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement with The University of Alabama (NA22NWS4320003).</td>
</tr></table>

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for details.

---

*Please submit an [issue](https://github.com/mhpi/dhbv2/issues) to report any questions, concerns, or bugs.*
