# Example Data

To run the models, you will need two types of data:

1. **Forcing Data:** Precipitation, Temperature, and PET (Potential EvapoTranspiration).
2. **Catchment Attributes:** Static properties (soil, slope, elevation, etc.).

</br>

## Quick Start

We provide prepared datasets containing a subset of AORC forcings and NextGen HydroFabric 2.2 attributes for CAMELS catchments.

See [NCAR](https://ral.ucar.edu/solutions/products/camels) for more information on the CAMELS dataset.

### Daily

>[Coming Soon]

### Hourly

- Catchments: [`2453`, `2454`, `2455`]

- Time: `2008-01-09 00:00:00` to `2010-12-30 23:00:00`.

Forcing location:

- NetCDF: `./ngen_resources/data/forcing/camels_2008-01-09 00_00_00_2010-12-30 23_00_00.nc`
- CSV: `./ngen_resources/data/forcing/cat-2453_2004-10-01 00_00_00_2018-09-30 23_00_00.csv`

Attribute location: Currently stored in BMI config. BMI will later support direct reading from a remotely hosted HydroFabric geopackage with attributes for all 800k catchments.

<!-- **Download Link:**
[AWS S3 - NextGen Demo Data](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/aorc_hydrofabric/ngen_demo.zip) -->

</br>

## Forcing Format

### CSV/NetCDF Format (NextGen Standard)

dhbv2 BMI expects a CSV/NetCDF file with minimum attributes:

- `time`: Timestamp (ns)

- `precip_rate[mm h-1]`: Precipitation mm/h (note ngen will assume `precip_rate` is in `mm s-1` unless we include the unit header as is done here.)

- `TMP_2maboveground`: Air Temperature (K)

- `PET_hargreaves`: Potential Evapotranspiration (This can be calculated and added to an existing dataset with `dhbv2/scripts/add_pet.py`)

Note: The MTS model requires hourly data, while the standard model operates on daily aggregates.

## Data Placement

<!-- Unzip the contents and place them so that the configuration files can find them. We recommend the following structure inside the `dhbv2` package or your NextGen data directory: -->

```text
dhbv2/
└── ngen_resources/
    └── data/
        ├── dhbv2/            # Daily model resources
        │   ├── config/       # BMI YAML configs
        │   ├── models/       # PyTorch weights & stats
        │   └── realizations/ # NextGen JSON realizations
        │
        ├── dhbv2_mts/        # MTS (hourly) model resources
        │   ├── config/
        │   ├── models/
        │   └── realizations/
        │
        ├── forcing/          # CSV or NetCDF forcings
        │   └── cat-2453_2008...csv
        │
        └── spatial/          # GeoJSON/HydroFabric data
            └── catchment_data_cat-2453.geojson
```
