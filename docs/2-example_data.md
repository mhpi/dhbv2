# Example Data

To run the models, you will need two types of data:

1. **Forcing Data:** Precipitation, Temperature, and PET (Potential EvapoTranspiration).
2. **Basin Attributes:** Static properties (soil, slope, elevation, etc.).

</br>

## Quick Start

We provide prepared datasets containing a subset of AORC forcings and NextGen Hydrofabric 2.2 attributes for CAMELS catchments.

### Daily

>[Coming Soon]

### Hourly

- Catchments: [`2453`, `2454`, `2455`]

- Time: `2008-01-09 00:00:00` to `2010-12-30 23:00:00`.

Forcing location: `/ngen_resources/data/forcing/camels_2008-01-09_00_00_00_2010-12-30_23_00_00.nc`

Attribute location: Currently stored in BMI config. BMI will later support direct reading from a remotely hosted HydroFabric geopackage containing attributes for all 800k catchments.

<!-- **Download Link:**
[AWS S3 - NextGen Demo Data](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/aorc_hydrofabric/ngen_demo.zip) -->

</br>

## Forcing Format

### CSV/NetCDF Format (NextGen Standard)

dhbv2 BMI expects a CSV/NetCDF file with minimum attributes:

- `time`: Timestamp (ns)

- `precip_rate`: Precipitation (mm/s or mm/h depending on config)

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
        ├── dhbv2_mts/        # MTS (Hourly) model resources
        │   ├── config/
        │   ├── models/
        │   └── realizations/
        │
        ├── forcing/          # CSV or NetCDF forcings
        │   └── cat-2453_2008...csv
        │
        └── spatial/          # GeoJSON/Hydrofabric files
            └── catchment_data_cat-2453.geojson
```
