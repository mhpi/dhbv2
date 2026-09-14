# Running Standalone

dhbv2 models can be run "standalone" with provided Python scripts. This may be useful for debugging, developing intutions for the structure, or running inference without compiling the full NextGen engine.

</br>

## Scripts

The `./scripts/` directory contains BMI forward examples for both daily and MTS (hourly) models.

### (1) Running the Daily Model

> *Coming soon.*

### (2) Running the MTS (Hourly) Model

The `forward_mts_cat-2453.py` script runs the MTS hourly model for a specific test catchment (cat-2453; 2454, 2455 are also available).

- **Config**: Uses `./ngen_resources/data/dhbv_2_mts/config/bmi_cat-2453.yaml`.

- **Input**: Uses NetCDF forcing file `./ngen_resources/data/forcing/camels_subset_2008-01-09 00_00_00_2015-12-30 23_00_00.nc`.

- **Output**: Streamflow (m3/s) for each hour.

</br>

## Configuration Files

Standalone runs rely on yaml **BMI config files**. These define the physics options and provide static catchment attributes.

Example `bmi_cat-2453.yaml`:

```yaml
# ... list of static attributes (aridity, meanP, etc.) ...

catchment_id: 'cat-2453'
model_dir: ./data/dhbv_2_mts/model/dhbv_2_mts/
dtype: float32
verbose: false
time_step: 1 hour

warmup:
  cycle_days: 7
  daily_mode: periodic
  daily_warmup_days: 351
  hourly_mode: periodic
  hourly_warmup_hours: 168
```
