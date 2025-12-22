# Running Standalone

You can run the models in "standalone" mode using Python scripts. This is useful for debugging, quick testing, or running inference without compiling the full NextGen engine.

</br>

## Scripts

The `scripts/` directory contains BMI forward scripts for both daily and MTS hourly models.

### (1) Running the Daily Model

The `forward_daily_cat-88306.py` script mimics the BMI lifecycle for the daily model and uses example forcing data contained within this module.

```bash
python scripts/forward_daily_cat-88306.py
```

### (2) Running the MTS (Hourly) Model

The `forward_mts_cat-2453.py` script runs the MTS hourly model for a specific test catchment (cat-2453; 2453, 2454, 2455 are available).

- Config: Uses `ngen_resources/data/dhbv2_mts/config/bmi_cat-2453.yaml`.

- Input: Uses NetCDF forcing file `ngen_resources/data/forcing/camels_2008-01-09_00_00_00_2010-12-30_23_00_00.nc`.

- Output: Streamflow (m3/s) for each hour.

</br>

## Configuration Files

Standalone runs rely on **BMI config files** (YAML). These define the physics options and point to the static attributes.

Example `bmi_cat-2453.yaml`:

```yaml
catchment_id: 'cat-2453'
time_step: 1 hour
model_path: models/
use_daily_states: false
# ... list of static attributes (aridity, meanP, etc.) ...
```
