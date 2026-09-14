# Example Data

To run dhbv2 models, you will need two types of data:

1. **Forcings:** Precipitation, Temperature, and PET (Potential EvapoTranspiration).
2. **Catchment Attributes:** Static spatial properties (soil, slope, elevation, etc.).

For NextGen runtime, a **hydrofabric geopackage** containing desired catchments is also required.

</br>

## Quick Start

This module provides an example dataset containing AORC forcings and catchment attributes for a subset of catchments in the defined by CAMELs.

See [NCAR](https://ral.ucar.edu/solutions/products/camels) for more information on the CAMELS.

Example data is as follows:

- Catchments: [`2453`, `2454`, `2455`]

- Time: `2008-01-09 00:00:00` to `2015-12-30 23:00:00`.

- Forcings:

  - NetCDF: `./ngen_resources/data/forcing/camels_subset_2008-01-09 00_00_00_2015-12-30 23_00_00.nc`

  - CSV: `./ngen_resources/data/forcing/cat-xxxx_2004-10-01 00_00_00_2018-09-30 23_00_00.csv`

- Attributes:

  - Stored in BMI configs `./ngen_resources/data/dhbv_2_mts/config/bmi_cat-2453.yaml`. *BMI will later support direct reading from a remotely hosted HydroFabric geopackage with attributes for all 800k catchments.*

- Geopackage:
  - `./ngen_resources/data/geo/camels_subset_hf2.gpkg`

<br/>

> (i) To create NextGen HydroFabric geopackages for other catchments, see `./scripts/utils/make_gpkg.py`.
>
> (ii) A script for getting static attributes for other catchments will be added at a later time.

<!-- **Download Link:**
[AWS S3 - NextGen Demo Data](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/aorc_hydrofabric/ngen_demo.zip) -->

</br>

## Forcing Format

### CSV/NetCDF Format (NextGen Standard)

The dhbv2 BMIs expects a CSV/NetCDF file with minimum attributes:

- `time`: Timestamp (ns)

- `precip_rate[mm h-1]`: Precipitation in mm/h (note that NextGen will assume `precip_rate` is in `mm s-1` unless a unit header as included as is done here.)

- `TMP_2maboveground`: Air Temperature in K.

- `PET_hargreaves`: Potential evapotranspiration in mm/h. (This can be calculated and added to an existing dataset with `./scripts/utils/add_pet.py`.)

<br/>

> The [MTS model](../src/dhbv2/mts_bmi.py) requires hourly data, while the [standard model](../src/dhbv2/bmi.py) operates on daily aggregates.

## Data Placement

Example data in `./ngen_resources/` is arranged to mirror organization within NextGen. Therefore, usage with NextGen simply requires moving its contents to `ngen/data/`. See [5-run_ngen](./5-run_ngen.md) for more detail.

```text
dhbv2/
└── ngen_resources/
    └── data/
        ├── dhbv2/            # Daily model resources
        │   ├── config/       # BMI/Routing YAML configs
        │   ├── models/       # PyTorch weights & normalization stats
        │   └── realizations/ # NextGen JSON realizations
        │
        ├── dhbv2_mts/        # MTS (hourly) model resources
        │   ├── config/
        │   ├── models/
        │   └── realizations/
        │
        ├── forcing/          # CSV/NetCDF forcings
        │   ├── camels_subset_2008...nc
        │   └── cat-2453_2008...csv
        │
        └── geo/          # GeoJSON/Geopackage HydroFabric data
            └── camels_subset_hf2.gpkg
```
