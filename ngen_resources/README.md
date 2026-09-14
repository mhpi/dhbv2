# Data

This directory contains files necessary for operating NextGen and for minimal
operation of a BMI standalone.

1. Realizations: Configure a NextGen runtime;
2. BMI configs: Configure a BMI instance;
3. Model: Downloaded separately (see `docs/1-module_setup.md`), contains

    - Trained model weights
    - Input normalization statistics
    - Model config

4. Forcings: AORC forcings for a catchment or catchments. A demo forcing file is
    included which matches the structure of NextGen NetCDF forcing files.

5. Hydrofabric: Static catchment attributes stored as a geojson or geopkg for the
    catchments included in the forcing data.

# Docker

For running ngen with Docker, copy the Dockerfile into NextGen at `./ngen/docker/`. See instructions in [docs](../docs/5-run_ngen.md) for building and running the ngen image.
