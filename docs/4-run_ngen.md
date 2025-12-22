# Running with NextGen

To run δHBV 2.0 within the NextGen framework, you must configure a **Realization** file (JSON) that points to the correct Python class and BMI configuration.

</br>

## Python Types

* **Daily Model:** `dhbv2.bmi.DeltaModelBmi`
* **Hourly (MTS) Model:** `dhbv2.mts_bmi.MtsDeltaModelBmi`

</br>

## Configuration Examples

### (1) Daily Simulation (`realization_cat-88306.json`)

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
  }
}
```

### (2) Hourly MTS Simulation (`realization_cat-2453.json`)

```json
{
  "global": { "time": { "output_interval": 3600 } },
  "catchments": {
    "cat-2453": {
      "formulations": [
        {
          "name": "bmi_python",
          "params": {
            "python_type": "dhbv2.mts_bmi.MtsDeltaModelBmi",
            "model_type_name": "HBV2.0 MTS",
            "init_config": "./data/dhbv2_mts/config/bmi_cat-2453.yaml",
            "uses_forcing_file": true,
            "main_output_variable": "land_surface_water__runoff_volume_flux",
            ...
          }
        }
      ]
    }
  }
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
