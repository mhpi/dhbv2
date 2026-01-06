# Running with NextGen

> Valid as of 28 Dec. 2025

To run δHBV 2.0 within the NextGen framework, you must configure a **Realization** file (JSON) that points to the correct Python class and BMI configuration.

</br>

## Installation

Clone NOAA-OWP's NextGen distribution:

```bash
git clone git@github.com:NOAA-OWP/ngen.git
cd ngen
```

### 1. Submodule

Install dhbv2 as a Git submodule in `./extern` as follows:

```bash
git submodule add git@github.com:mhpi/dhbv2.git extern/dhbv2/dhbv2

git submodule update --init --recursive
```

> Alternatively, clone MHPI's fork of ngen with dhbv2 preinstalled;
>
> ```bash
> git clone git@github.com:mhpi/ngen.git
> git submodule update --init --recursive
> ```

### 2. NextGen

To build, we recommend using Docker as the [developers suggest](https://github.com/NOAA-OWP/ngen/blob/master/INSTALL.md). A [Dockerfile](../ngen_resources/docker/) supporting dhbv2 is included with this repo. Copy this Dockerfile to ngen and build:

```bash
cp ./dhbv2/ngen_resources/docker/CENTOS_MHPI_NGEN_RUN.dockerfile ./ngen/docker/

docker build . --build-arg NPROC=8 --file ./docker/CENTOS_MHPI_NGEN_RUN.dockerfile --tag localbuild/ngen:latest --network=host
```

> For e.g., HPCs, include the additional argument `--network=host` with docker build if you encounter failure due to network connection.

#### Docker

To inspect the container after building an image:

```bash
docker run -it --rm localbuild/ngen:latest /bin/bash
```

To cleanup old Docker images/containers:

```bash
docker rm `docker ps --no-trunc -aq`
docker images -q --filter "dangling=true" | xargs docker rmi
```

or

```bash
docker system prune -f
```

## Python Types

* **Daily Model:** `dhbv2.bmi.DeltaModelBmi`
* **Hourly (MTS) Model:** `dhbv2.mts_bmi.MtsDeltaModelBmi`

</br>

## Configuration Examples

### 1. Daily Simulation (`realization_cat-88306.json`)

```json
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
            "init_config": "./data/dhbv2/config/bmi_cat-88306.yaml",
            "uses_forcing_file": true,
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

### 2. Hourly (MTS) Simulation (`realization_cat-2453.json`)

```json
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

Run the NextGen engine pointing to the realization file:

```bash
./cmake_build/ngen \
    /path/to/catchment_data.geojson "cat-2453" \
    /path/to/nexus_data.geojson "nex-2454" \
    /path/to/realization_cat-2453.json
```

or, with a docker image,

```bash
# With geojson for 1 catchment
docker run --rm \
    -v $(pwd)/data:/app/data \
    localbuild/ngen:latest \
    ./ngen \
    /path/to/catchment_data.geojson 'cat-2453' \
    /path/to/nexus_data.geojson 'nex-2454' \
    ./data/dhbv2_mts/realizations/realization_cat-2453.json

# With geopackage
docker run --rm \
    -v $(pwd)/data:/app/data \
    localbuild/ngen:latest \
    ./ngen \
    ./data/camels_hf2.gpkg 'cat-2453' \
    ./data/camels_hf2.gpkg 'nex-2454' \
    ./data/dhbv2_mts/realizations/realization_cat-2453.json
```

To run all catchments defined in the geopackage/geojson, leave catchment and nexus (e.g., `'cat-2453'` and `'nex-2454'`) defined as `''`.

> Note: If using Docker, make sure `output_root` in your realization begins with `/app/data/`. This will ensure ngen outputs are accessible outside of your Docker container in `./ngen/data/`.
> Note: with default settings, ngen output will save to `./ngen/data/output/`

## Validation

> Tests supplied by ngen and troute repositories.

### 1. Compile Time

To view the compile-time configuration of an pre-compiled NextGen binary use
the --info flag:

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    localbuild/ngen:latest \
    ./ngen --info
```

### 2. NextGen Tests

To run stock ngen tests (visible with `docker run --rm localbuild/ngen:latest ls /app/test`) within a Docker container, use e.g.,

```bash
docker run --rm localbuild/ngen:latest ./test/test_unit
```

Example realizations can also be run with

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    localbuild/ngen:latest \
    ./ngen \
    data/catchment_data.geojson '' \
    data/nexus_data.geojson '' \
    data/example_bmi_multi_realization_config.json
```

### 3. T-Route Tests

NextGen-integrated troute can be validated as follows:

```bash
docker run --rm localbuild/ngen:latest ./test/test_routing_pybind
```

Troute can be run standalone with the examples it ships with:

```bash
docker run --rm \
  -w /app/troute/test/LowerColorado_TX \
  localbuild/ngen:latest \
  python -m nwm_routing -f -V4 test_AnA_V4_NHD.yaml


docker run --rm \
  -w /app/troute/test/LowerColorado_TX_v4 \
  localbuild/ngen:latest \
  python -m nwm_routing -f -V4 test_AnA_V4_HYFeature.yaml
```
