# Running with NextGen

To run δHBV 2.0 within the NextGen framework, you must configure a **realization** file (JSON) that points to the correct Python class and BMI configuration.

</br>

## Installation

Clone NOAA-OWP's NextGen distribution:

```bash
git clone git@github.com:NOAA-OWP/ngen.git
cd ngen
```

> For AWI's NextGen IN A Box (NGIAB), see the fork [CIROH-UA/ngen](https://github.com/CIROH-UA/ngen).

### (1) Submodule

Install dhbv2 as a Git submodule in `./extern` as follows:

```bash
git submodule add https://github.com/mhpi/dhbv2.git extern/dhbv2/dhbv2

git submodule update --init --recursive
```

> Alternatively, clone MHPI's fork of ngen with dhbv2 preinstalled;
>
> ```bash
> git clone https://github.com/mhpi/dhbv2.git
> git submodule update --init --recursive
> ```

### (2) NextGen

NextGen is a C++ compiled framework. There are two options for installing its dependencies and building: manually or with Docker. The latter builds an isolated container for NextGen and and simplifies the build process.

We recommend using Docker as the [developers suggest](https://github.com/NOAA-OWP/ngen/blob/master/INSTALL.md), and a [Dockerfile](../ngen_resources/docker/) supporting dhbv2 is included with this repo. We will not cover manual installation here; for instructions, see [NOAA-OWP/ngen/INSTALL.md](https://github.com/NOAA-OWP/ngen/blob/master/INSTALL.md).

For Docker installation, copy the Dockerfile to ngen and build:

```bash
cp ./dhbv2/ngen_resources/docker/ngen_dhbv2.dockerfile ./ngen/docker/

docker build . \
    --build-arg NPROC=8 \
    --file ./docker/ngen_dhbv2.dockerfile \
    --tag localbuild/ngen:latest \
    --network=host
```

> For e.g., HPCs, the additional argument `--network=host` should avoid any failure due to network connection.
>
> For faster builds, set `NPROC` to the number of CPU cores you have available.

To inspect the container after building an image:

```bash
docker run -it --rm localbuild/ngen:latest /bin/bash
```

To cleanup old Docker images/containers:

```bash
docker system prune -f
```

</br>

## Python Types

* **Daily Model:** `dhbv2.bmi.DeltaModelBmi`
* **Hourly (MTS) Model:** `dhbv2.mts_bmi.MtsDeltaModelBmi`

</br>

## Configuration Examples

### (1) Daily Simulation

```json
# ./ngen_resources/data/dhbv_2/config/bmi_cat-2453.yaml

{
  "global": { "time": { "output_interval": 86400 } },
  "catchments": {
    "cat-88306": {
      "formulations": [
        {
          "name": "bmi_python",
          "params": {
            "python_type": "dhbv2.bmi.DeltaModelBmi",
            "model_type_name": "DeltaModelBmi",
            "init_config": "./data/dhbv_2/config/bmi_cat-2453.yaml",
            "uses_forcing_file": false,
            "main_output_variable": "land_surface_water__runoff_volume_flux",
            ...
          }
        }
      ]
    }
  },
  "output_root": "..."
}
```

### (2) Hourly (MTS) Simulation

```json
# ./ngen_resources/data/dhbv_2_mts/config/bmi_cat-2453.yaml

{
  "global": { "time": { "output_interval": 3600 } },
  "catchments": {
    "cat-2453': {
      "formulations": [
        {
          "name": "bmi_python",
          "params": {
            "python_type": "dhbv2.mts_bmi.MtsDeltaModelBmi",
            "model_type_name": "DeltaModelBmi",
            "init_config": "./data/dhbv_2_mts/config/bmi_cat-2453.yaml",
            "uses_forcing_file": false,
            "main_output_variable": "land_surface_water__runoff_volume_flux",
            ...
          }
        }
      ]
    }
  },
  "output_root": "..."
}
```

</br>

## Execution

We give a few examples to illustrate NextGen execution. In all cases, we will require

1. **NextGen HydroFabric** (subset) for desired catchments stored as either a geojson or geopackage;
2. **realization** Json that configures the ngen runtime
3. the **name of each catchment and nexus** to simulate. If doing simulation for all catchments in your HydroFabric, these need not be specified.

To run, we point our ngen executable to the above. For a single catchment,

```bash
cd ./ngen

# Geojson
./cmake_build/ngen \
    /path/to/catchment_data.geojson 'cat-2453' \
    /path/to/nexus_data.geojson 'nex-2454' \
    data/dhbv_2_mts/realizations/realization_cat-2453.json

# Geopackage
./cmake_build/ngen \
    data/geo/camels_subset_hf2.gpkg 'cat-2453' \
    data/geo/camels_subset_hf2.gpkg 'nex-2454' \
    data/dhbv_2_mts/realizations/realization_cat-2453.json

# Or with Docker

docker run --rm \
    -v $(pwd)/data:/ngen/data \
    -v $(pwd)/output:/ngen/output \
    localbuild/ngen:latest \
    ngen \
    data/geo/camels_subset_hf2.gpkg 'cat-2453' \
    data/geo/camels_subset_hf2.gpkg 'nex-2454' \
    data/dhbv_2_mts/realizations/realization_cat-2453.json
```

With default settings, ngen outputs will save to `./ngen/output/`.

> Notes on Docker:
>
> We use `-v $(pwd)/data:/ngen/data` to replace the container's internal data directory with that of your local directory. This enables usate and modification of realizations, configs, etc. outside of the container. `-v $(pwd)/output:/ngen/output` similarly ensures outputs are accessible outside of the container. `localbuild/ngen:latest` is the name of the Docker image.
>
> `output_root` in your realization should begin with `./output/` or otherwise matches your flag `-v $(pwd)/output:/ngen/output`. This ensures ngen outputs are saved outside of the Docker container.

To run all catchments defined in the geopackage/geojson (3 in our example), leave catchment and nexus arguments (e.g., `'cat-2453'` and `'nex-2454'`) undefined like so:

```bash
cd ./ngen

./cmake_build/ngen \
    data/geo/camels_subset_hf2.gpkg '' \
    data/geo/camels_subset_hf2.gpkg '' \
    data/dhbv_2_mts/realizations/realization_cat-2453.json

# Or with Docker

docker run --rm \
    -v $(pwd)/data:/ngen/data \
    -v $(pwd)/output:/ngen/output \
    localbuild/ngen:latest \
    ngen \
    data/geo/camels_subset_hf2.gpkg '' \
    data/geo/camels_subset_hf2.gpkg '' \
    data/dhbv_2_mts/realizations/realization_cat-2453.json
```

Realizations can accomodate catchment-specific formulations in addition to the "global" formulation. This is useful, for example, if each catchment needs a specific BMI config containing spatially distinct physical attributes (as is the case for dhbv2). See `data/dhbv_2_mts/realizations/realization_multi_cat.json`, which can also be used like

```bash
cd ./ngen

./cmake_build/ngen \
    data/geo/camels_subset_hf2.gpkg '' \
    data/geo/camels_subset_hf2.gpkg '' \
    data/dhbv_2_mts/realizations/realization_multi_cat-2453.json

# Or with Docker

docker run --rm \
    -v $(pwd)/data:/ngen/data \
    -v $(pwd)/output:/ngen/output \
    localbuild/ngen:latest \
    ngen \
    data/geo/camels_subset_hf2.gpkg '' \
    data/geo/camels_subset_hf2.gpkg '' \
    data/dhbv_2_mts/realizations/realization_multi_cat-2453.json
```

For instructions on routing NextGen runoff simulations, see [7-routing](./7-routing.md).

<br/>

## Validation

Tests supplied by ngen and troute repositories can be used to verify your Docker installation is behaving as expected.

### (1) Build Info

To view configuration info of the NextGen binary, use
the --info flag:

```bash
docker run --rm \
    -v $(pwd)/data:/ngen/data \
    localbuild/ngen:latest \
    ngen --info
```

### (2) NextGen Tests

To run stock ngen tests within a Docker container, use e.g.,

```bash
# Unit tests
docker run --rm localbuild/ngen:latest ./test/test_unit

# or all tests
docker run --rm localbuild/ngen:latest ./test/test_all
```

A list of all available tests can be found with `docker run --rm localbuild/ngen:latest ls /ngen/test`.

To break output at a failure, append `--gtest_break_on_failure` to the above.

### (3) NextGen Example

As demonstrated previously, ngen's example realizations can be run like

```bash
docker run --rm \
    -v $(pwd)/data:/ngen/data \
    -v $(pwd)/output:/ngen/output \
    localbuild/ngen:latest \
    ngen \
    data/catchment_data.geojson '' \
    data/nexus_data.geojson '' \
    data/example_bmi_multi_realization_config.json
```

> To make the outputs of this test run available, you may need to add `output_root` to the end of the realization like so:
>
> ```bash
> {
>    ...,
>    "time": {
>        ...
>    },
>    "output_root": "./output/"
>}
>```

### (4) T-Route Tests

NextGen-integrated T-route can be validated using

```bash
docker run --rm localbuild/ngen:latest ./test/test_routing_pybind
```

### (5) T-Route Example

T-Route examples can be run in Docker containers like

```bash
docker run --rm \
    -v $(pwd)/output:/ngen/output \
    -w /ngen/t-route/test/LowerColorado_TX \
    localbuild/ngen:latest \
    python -m nwm_routing -f -V4 test_AnA_V4_NHD.yaml
```

See [NOAA-OWP/t-route](https://github.com/NOAA-OWP/t-route) for more details.
