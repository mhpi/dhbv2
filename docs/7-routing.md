# Routing Runoff Simulations

After obtaining runoff simulations from δHBV 2.0, there are a few options for routing flow through the river network.

The network can be defined in several ways, with [MERIT-hydro](https://www.reachhydro.org/home/params/merit-basins) and the [NextGen HydroFabric](https://github.com/NOAA-OWP/hydrofabric) being notable examples. In the context of NextGen (ngen), we demonstrate routing on the HydroFabric v2.2 (download [v2.2 source](https://www.lynker-spatial.com) or [AWI-patched v2.2](https://github.com/CIROH-UA/community_hf_patcher/tree/main)) river network with ~4km resolution.

<br/>

## T-Route

[T-Route](https://github.com/NOAA-OWP/t-route) is the standard routing package shipped with [ngen](https://github.com/NOAA-OWP/ngen) and is installed by default when [building ngen with Docker](./5-run_ngen.md/#). This includes support for e.g., Muskingum-Cunge (MC) and diffusive wave routing methods.

For the purposes of this module, we only demonstrate usage of troute within ngen as a post-processor. If you wish to do routing standalone, please see the repo's [official documentation](https://github.com/NOAA-OWP/t-route/blob/master/readme.md).

### T-Route Setup

Setup of T-Route and its dependencies is included in the Docker image build process described for ngen in [5-run_ngen](./5-run_ngen.md), so no further effort is required for this step. It should be noted that the [CIROH-UA/t-route](https://github.com/CIROH-UA/t-route) fork is used here for compatibility, but will not amount to a functional difference from source.

If you wish to build T-Route manually, see [NOAA-OWP/t-route/readme.md](https://github.com/NOAA-OWP/t-route/blob/master/readme.md) for instructions.

### T-Route Example

See [5-run_ngen](./5-run_ngen.md) for instructions on setting up ngen.

To run e.g. MC routing inside ngen with T-Route, an additional routing config is necessary and specified inside the realization:

```json
{
  "routing":{
    "t_route_connection_path": "./extern/t-route/src/ngen_routing/src",
    "t_route_config_file_with_path": "./data/dhbv_2_mts/config/routing_config.yaml"
  }
}
```

With an updated realization and routing config, runoff simulation and routing can be run with

```bash
cd ./ngen

./cmake_build/ngen \
    data/geo/camels_subset_hf2.gpkg '' \
    data/geo/camels_subset_hf2.gpkg '' \
    data/dhbv_2_mts/realizations/realization_routing_cat-2453.json

# Or with Docker

docker run --rm \
    -v $(pwd)/data:/ngen/data \
    -v $(pwd)/output:/ngen/output \
    localbuild/ngen:latest \
    ngen \
    data/geo/camels_subset_hf2.gpkg '' \
    data/geo/camels_subset_hf2.gpkg '' \
    data/dhbv_2_mts/realizations/realization_troute_cat-2453.json
```

Outputs will save by default to `./ngen/output/stream_output`.

In addition to running in NextGen, T-Route can be deployed standalone and comes with a few examples to demonstrate:

```bash
docker run --rm \
    -v $(pwd)/output:/ngen/output
    -w /ngen/t-route/test/LowerColorado_TX \
    localbuild/ngen:latest \
    python -m nwm_routing -f -V4 test_AnA_V4_NHD.yaml


docker run --rm \
    -v $(pwd)/output:/ngen/output
    -w /ngen/t-route/test/LowerColorado_TX_v4 \
    localbuild/ngen:latest \
    python -m nwm_routing -f -V4 test_AnA_V4_HYFeature.yaml
```

<br/>

## Distributed Differentiable Routing (DDR)

Another option for routing is to take the ML approach applied to HBV: make the model differentiable and parameterize intelligently with a neural network.

T-route uses parameterizations which are static in time and are not guaranteed to avoid issues of equifinality: the set of optimal parameters is not guaranteed to be unique and can result in spatial incoherence.

Tadd Bindas et. al (2025, in prep) have shown there is value in generalization, physical consistency, and streamflow forecast performance by leveraging big data to parameterize the Muskingum-Cunge (MC) model. We call this differentiable MC, denoted by δMC, named after the requirement that the MC model itself be differentiable to support the neural network's gradient-based training.

This routing method has been demonstrated alongside naive hydrograph routing methods to improve skill for δHBV 2.0.

[Distributed Differentiable Routing (DDR)](https://github.com/DeepGroundwater/ddr) is an operational formalization of this philosophy currently supporting δMC. While there are plans to support DDR within ngen in some capacity, routing must be done separately on ngen's runoff outputs.

### DDR Setup

#### (1) Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/leoglonz/ddr.git
    cd ddr
    ```

2. Optional - Create a conda/venv environment to isolate dependencies:

    DDR requires `>=py3.11`, so a new env may be required if dhbv2 was previously installed for ngen with `py3.9`.

    ```bash
    conda create -n ddr python=3.11
    conda activate ddr

    # or

    uv venv --python=3.11 .venv_ddr
    source .venv_ddr/bin/activate
    ```

3. Install dependencies:

    We recommend [Astral UV](https://docs.astral.sh/uv/) to install packages (available via `pip install uv`), however standard `pip install` will also work.

    ```bash
    cd ./ddr
    uv pip install . ./engine

    # or in editable mode
    uv pip install -e . -e ./engine
    ```

#### (2) Data

> *Coming soon.*

### DDR Example

To run δMC routing for our example catchments `cat-2453`, `cat-2454`, `cat-2455`

```bash
Coming soon.
```
