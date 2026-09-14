# Configurations

Running δHBV 2.0 requires two configuration files, which serve distinct roles:

1. **BMI config** (YAML): One per catchment. Tells a *single* model instance where its weights live, what static attributes describe its catchment, and how to spin up its states.
2. **Realization** (JSON): One per NextGen run. Tells *ngen* which BMI class to instantiate, where to find each catchment's BMI config, how to map forcing columns onto BMI variables, and what window to simulate.

A third file, the **model config** (`config.yaml`), ships with the downloaded weights in `model_dir` and describes the trained network architecture. It is read automatically and **should not be edited** — changing it will desynchronize the code from the weights.

```text
realization.json          # ngen runtime      -> points at:
└── bmi_cat-XXXX.yaml     # BMI instance      -> points at:
    └── model_dir/config.yaml   # trained model (do not edit)
```

</br>

## BMI Config

BMI configs live in `./ngen_resources/data/dhbv_2{_mts}/config/` and are passed to `initialize()` via the realization's `init_config` key. One file is needed per catchment, because each carries that catchment's static attributes.

### (1) Runtime Keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `model_dir` | path | *required* | Directory holding the trained weights, `config.yaml`, and `normalization_statistics.json`. Resolved relative to the ngen working directory; if that path does not exist, it is retried relative to `./ngen_resources/` so the same config works for standalone runs. |
| `dtype` | str | `float64` | Precision of BMI arrays and model input. `float32` matches the trained weights and is what the shipped configs use. |
| `time_step_size` | int (s) | `3600` | BMI timestep reported to ngen through `get_time_step()`. Both models are driven with **hourly** input, so this should stay `3600` even for the daily model, which aggregates 24 hourly steps internally. |

> **Careful:** the shipped configs also contain `time_step: 1 hour` and `catchment_id: 'cat-2453'`. These are **bookkeeping only** — neither is read by the BMI. The key that actually sets the timestep is `time_step_size` (in seconds), and catchment identity comes from ngen, not the config. Setting `time_step: 1 day` does not make the model run daily.

### (2) Catchment Attributes

The remaining top-level keys are the 28 (daily) or 30 (MTS) static attributes the parameterization network expects. These must be computed the same way as the attributes the model was trained on; substituting a differently-derived value will silently degrade predictions rather than raise an error.

Each attribute is exposed through BMI under a CSDMS standard name. The internal (config) name is what you write in the YAML:

| Config key | CSDMS standard name | Units |
| --- | --- | --- |
| `aridity` | `ratio__mean_potential_evapotranspiration__mean_precipitation` | - |
| `meanP` | `atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate` | mm d-1 |
| `ETPOT_Hargr` | `land_surface_water__Hargreaves_potential_evaporation_volume_flux` | mm d-1 |
| `NDVI` | `land_vegetation__normalized_diff_vegetation_index` | - |
| `FW` | `free_land_surface_water` | mm d-1 |
| `meanslope` | `basin__mean_of_slope` | m km-1 |
| `SoilGrids1km_sand` | `soil_sand__grid` | km2 |
| `SoilGrids1km_clay` | `soil_clay__grid` | km2 |
| `SoilGrids1km_silt` | `soil_silt__grid` | km2 |
| `glaciers` | `land_surface_water__glacier_fraction` | percent |
| `HWSD_clay` | `soil_clay__attr` | percent |
| `HWSD_gravel` | `soil_gravel__attr` | percent |
| `HWSD_sand` | `soil_sand__attr` | percent |
| `HWSD_silt` | `soil_silt__attr` | percent |
| `meanelevation` | `basin__mean_of_elevation` | m |
| `meanTa` | `atmosphere_water__daily_mean_of_temperature` | degC |
| `permafrost` | `land_surface_water__permafrost_fraction` | - |
| `permeability` | `bedrock__permeability` | m2 |
| `seasonality_P` | `p_seasonality` | - |
| `seasonality_PET` | `land_surface_water__potential_evaporation_volume_flux_seasonality` | - |
| `snow_fraction` | `land_surface_water__snow_fraction` | percent |
| `snowfall_fraction` | `atmosphere_water__precipitation_falling_as_snow_fraction` | percent |
| `T_clay` | `soil_clay__volume_fraction` | percent |
| `T_gravel` | `soil_gravel__volume_fraction` | percent |
| `T_sand` | `soil_sand__volume_fraction` | percent |
| `T_silt` | `soil_silt__volume_fraction` | percent |
| `Porosity` | `soil_active-layer__porosity` | - |
| `uparea` | `basin__area` | km2 |
| `catchsize` † | `catchment__area` | km2 |
| `lengthkm` † | `basin__length` | km |

† MTS only.

A missing attribute is **not fatal**: the BMI logs `Static variable '<name>' not in BMI config. Skipping.` and leaves the value at `0.0`, which will propagate into the parameterization. Watch the log on first run rather than assuming silence means success.

### (3) Daily-Only Keys

The daily BMI (`dhbv2.bmi.DeltaModelBmi`) computes PET internally and needs to know the calendar date to do so:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `start_time` | str | *required* | Simulation start as `YYYY/MM/DD HH`, `YYYY/MM/DD HH:MM`, or `YYYY/MM/DD HH:MM:SS`. Used to derive day-of-year for Hargreaves and to stamp daily aggregates. Must match the realization's `time.start_time`. |
| `pet_method` | str | `penman_monteith` | `penman_monteith` or `hargreaves`. |
| `latitude` | float (deg) | *required if Hargreaves* | Catchment centroid latitude. Converted to radians internally. |

The two PET methods have different **forcing** requirements, so this choice constrains the realization's `variables_names_map`:

* `penman_monteith` — needs the full AORC set: temperature, specific humidity, longwave and shortwave radiation, surface pressure, and both wind components.
* `hargreaves` — derives PET from the daily min/max/mean of temperature alone, plus `latitude` and day-of-year. Only precipitation and temperature are required.

### (4) MTS-Only Keys: `warmup`

HBV is a recurrent bucket model, so its states (and the parameterization LSTM's hidden states) must saturate before predictions are meaningful. The MTS BMI exposes this as a `warmup` block, set independently per timescale. The daily model always runs and always hands its states to the hourly model every `cycle_days`; the modes only change *how each model gets there*.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `cycle_days` | int ≥ 1 | `7` | How often the daily model hands its states to the hourly model — and therefore how long the hourly model simulates per cycle. Always active. |
| `daily_mode` | str | `periodic` | `periodic`, `once`, or `cold`. See below. |
| `daily_warmup_days` | int ≥ 1 | `351` | Length of the daily warmup. Inactive under `daily_mode: cold` (no daily warmup is run); the daily model then ingests `cycle_days` of new forcing per cycle. |
| `hourly_mode` | str | `periodic` | `periodic` or `cold`. |
| `hourly_warmup_hours` | multiple of 24 | `168` | Length of the hourly warmup. Inactive under `hourly_mode: cold` (no hourly warmup is run); the hourly model then simulates `cycle_days` per cycle. |

`cycle_days` is the only key that is always live. The two `*_warmup_*` keys are warmup lengths, so each is inactive — neither validated nor used, and warned about if set — when its model's mode is `cold`. To change how long the hourly model runs before being re-anchored, set `cycle_days`, not `hourly_warmup_hours`.

**`daily_mode`:**

* `periodic` *(default, legacy behavior)* — re-spin from default states over the full `daily_warmup_days` at every warmup. Most expensive, most reproducible.
* `once` — spin up over `daily_warmup_days` at the start of simulation, then advance `cycle_days` at a time from cached physics and LSTM states.
* `cold` — never spin up; start from default states and advance `cycle_days` at a time, warming up as the simulation proceeds.

**`hourly_mode`:**

* `periodic` *(default)* — take the daily states, spin up over `hourly_warmup_hours`, then simulate.
* `cold` — take the daily states and simulate immediately, with no hourly spin-up.

The choice directly sets **how much forcing data must precede your first prediction**. This is the most common cause of a failed or empty MTS run:

| `daily_mode` | `hourly_mode` | Required lead-in |
| --- | --- | --- |
| `periodic` / `once` | `periodic` | `daily_warmup_days` + `hourly_warmup_hours` (~358 days) |
| `periodic` / `once` | `cold` | `daily_warmup_days` (~351 days) |
| `cold` | `periodic` | `cycle_days` + `hourly_warmup_hours` (~14 days) |
| `cold` | `cold` | `cycle_days` (~7 days) |

So with the shipped defaults, a simulation starting `2009-01-01 00:00` needs a forcing timeseries beginning `2008-01-08 00:00`. Shortening the lead-in with `cold` modes trades spin-up cost for less-saturated initial states — useful for smoke tests, not for benchmarking.

An invalid `daily_mode`, `hourly_mode`, or `cycle_days < 1` raises at `initialize()` rather than failing quietly. The `*_warmup_*` keys are validated only when their mode is not `cold`, since a `cold` mode neither reads nor needs them.

### (5) Example

```yaml
# ./ngen_resources/data/dhbv_2_mts/config/bmi_cat-2453.yaml

catchment_id: 'cat-2453'      # bookkeeping only

# Catchment attributes
aridity: 0.6210572673615424
meanP: 1199.3666323047523
ETPOT_Hargr: 743.6801723462127
# ... remaining attributes ...
lengthkm: 2.147179500288239

# BMI
model_dir: ./data/dhbv_2_mts/model/dhbv_2_mts/
dtype: float32
verbose: false                # bookkeeping only
time_step: 1 hour             # bookkeeping only; see `time_step_size`

warmup:
  cycle_days: 7
  daily_mode: periodic
  daily_warmup_days: 351
  hourly_mode: periodic
  hourly_warmup_hours: 168
```

</br>

## Realization

Realizations live in `./ngen_resources/data/dhbv_2{_mts}/realizations/` and are the third positional argument to the `ngen` executable. See [5-run_ngen](./5-run_ngen.md) for invocation.

### (1) Top-Level Keys

| Key | Description |
| --- | --- |
| `global` | Formulation and forcing defaults applied to every catchment in the hydrofabric. |
| `catchments` | Optional per-catchment overrides, keyed by catchment ID. Required for dhbv2 whenever catchments differ in static attributes — which is nearly always, since each needs its own `init_config`. |
| `time` | Simulation window and output cadence. |
| `routing` | Optional T-Route block. See [7-routing](./7-routing.md). |
| `output_root` | Where ngen writes per-catchment CSVs. Under Docker this must match your `-v $(pwd)/output:/ngen/output` mount, e.g. `./output/`. |

### (2) `formulations[].params`

Every dhbv2 formulation uses ngen's `bmi_python` adapter.

| Key | Value for dhbv2 | Description |
| --- | --- | --- |
| `python_type` | `dhbv2.mts_bmi.MtsDeltaModelBmi` (hourly) or `dhbv2.bmi.DeltaModelBmi` (daily) | Importable path to the BMI class. This is the one key that selects which model runs. |
| `model_type_name` | e.g. `dhbv2.0 mts` | Free-form label used in ngen logs and output naming. Has no effect on behavior. |
| `init_config` | e.g. `./data/dhbv_2_mts/config/bmi_cat-2453.yaml` | Path to the BMI config above, relative to the ngen working directory. Under NGIAB this must be a container-absolute path — see [6-run_ngiab](./6-run_ngiab.md). |
| `main_output_variable` | `land_surface_water__runoff_volume_flux` | The variable ngen reads each step and forwards to routing. |
| `uses_forcing_file` | `false` | The BMI does not read forcings itself; ngen pushes them in via `set_value()`. Keep this `false`. |
| `fixed_time_step` | `true` | Timestep never varies during a run. |
| `allow_exceed_end_time` | `true` | Permits the model to be stepped past `time.end_time`, which ngen does while flushing routing. |
| `variables_names_map` | see below | Maps CSDMS standard names to the column names in your forcing file. |

### (3) `variables_names_map`

Keys are CSDMS standard names the BMI exposes; values are the column headers in your forcing CSV/NetCDF. If a name on the left is misspelled or a column on the right is absent, ngen raises at initialization.

**Inputs** (identical for both models):

| CSDMS standard name | AORC column | Units | Needed when |
| --- | --- | --- | --- |
| `atmosphere_water__liquid_equivalent_precipitation_rate` | `precip_rate[mm h-1]` | mm h-1 | always |
| `land_surface_air__temperature` | `TMP_2maboveground` | degC | always |
| `atmosphere_air_water~vapor__relative_saturation` | `SPFH_2maboveground` | g g-1 | Penman-Monteith |
| `land_surface_radiation~incoming~longwave__energy_flux` | `DLWRF_surface` | W m-2 | Penman-Monteith |
| `land_surface_radiation~incoming~shortwave__energy_flux` | `DSWRF_surface` | W m-2 | Penman-Monteith |
| `land_surface_air__pressure` | `PRES_surface` | Pa | Penman-Monteith |
| `land_surface_wind__x_component_of_velocity` | `UGRD_10maboveground` | m s-1 | Penman-Monteith |
| `land_surface_wind__y_component_of_velocity` | `VGRD_10maboveground` | m s-1 | Penman-Monteith |

**Output:**

| CSDMS standard name | Mapped to | Units |
| --- | --- | --- |
| `land_surface_water__runoff_volume_flux` | `streamflow_cms` | m h-1 |

> Two unit notes worth internalizing:
>
> (i) ngen assumes `precip_rate` is **mm s-1** unless the header carries an explicit unit, which is why the shipped CSVs use `precip_rate[mm h-1]`. The NetCDF realization maps the bare `precip_rate` instead, because the NetCDF variable carries its own units attribute. Mixing these up produces runoff wrong by a factor of 3600.
>
> (ii) The output name says `cms`, but the BMI emits **m h-1** (runoff depth), converted from the model's internal mm h-1. Multiply by catchment area to get a volumetric rate. The name is retained for downstream compatibility.

> **Careful:** `water_potential_evaporation_flux` appears in `realization_multi_cat.json`'s per-catchment maps. It is **not** a recognized BMI variable — neither model accepts external PET, both compute it internally from the keys above — so that entry is inert and any `PET_hargreaves` column it names is ignored. To use Hargreaves, set `pet_method` in the *BMI config*, not here.

### (4) `forcing`

Three providers are demonstrated in the shipped realizations:

```json
// Per-catchment CSV; {{id}} is substituted with the catchment ID
"forcing": {
  "file_pattern": ".*{{id}}.*.csv",
  "path": "./data/forcing/",
  "provider": "CsvPerFeature"
}
```

```json
// Single NetCDF holding all catchments
"forcing": {
  "path": "./data/forcing/camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc",
  "provider": "NetCDF"
}
```

With `CsvPerFeature`, `path` is a directory and `file_pattern` selects within it. With `NetCDF`, `path` is the file itself and `file_pattern` is omitted.

### (5) `time`

| Key | Description |
| --- | --- |
| `start_time` | First simulated timestep, `YYYY-MM-DD HH:MM:SS`. For the daily model this must agree with the BMI config's `start_time` (note the differing format: `/` there, `-` here). |
| `end_time` | Last simulated timestep. |
| `output_interval` | Seconds between outputs: `3600` for MTS, `86400` for daily. |

Remember that `start_time` is the first *predicted* step, not the first step of *data*. Your forcing file must begin one full warmup lead-in earlier — 358 days for MTS defaults, 365 days for the daily model.

### (6) Global vs. Per-Catchment

`realization_cat-2453.json` defines only a `global` formulation, which ngen applies to every catchment it is given. That is correct for a single catchment, but running multiple catchments this way would point them all at `cat-2453`'s attributes.

For multi-catchment runs, give each catchment its own block under `catchments`, each with its own `init_config`:

```json
{
  "global": { "formulations": [ ... ], "forcing": { ... } },
  "catchments": {
    "cat-2453": {
      "formulations": [{
        "name": "bmi_python",
        "params": {
          "python_type": "dhbv2.mts_bmi.MtsDeltaModelBmi",
          "init_config": "./data/dhbv_2_mts/config/bmi_cat-2453.yaml",
          "...": "..."
        }
      }],
      "forcing": {
        "file_pattern": ".*cat-2453.*.csv",
        "path": "./data/forcing/",
        "provider": "CsvPerFeature"
      }
    },
    "cat-2454": { "...": "..." },
    "cat-2455": { "...": "..." }
  },
  "time": { ... },
  "output_root": "./output/"
}
```

A catchment present in the hydrofabric but absent from `catchments` falls back to the `global` formulation. See `realization_multi_cat.json` for the complete file.

</br>

## Shipped Examples

| File | Purpose |
| --- | --- |
| `realization_cat-2453.json` | Single catchment, per-feature CSV forcing. The baseline. |
| `realization_nc_cat-2453.json` | Same, reading a NetCDF forcing file. |
| `realization_multi_cat.json` | Three catchments with per-catchment `init_config`. |
| `realization_troute_cat-2453.json` | Adds the `routing` block for T-Route. |

Each exists under both `dhbv_2/` and `dhbv_2_mts/`, differing in `python_type`, `init_config`, and `output_interval`.

</br>

## Checklist

Before a run, verify:

1. `model_dir` resolves and contains `config.yaml`, `normalization_statistics.json`, and a `.pt` weights file — these are downloaded separately (see [1-module_setup](./1-module_setup.md)).
2. `python_type` matches the model whose `init_config` you pointed at — an MTS config under the daily class will not error cleanly.
3. `output_interval` matches the model: `3600` (MTS) or `86400` (daily).
4. Your forcing timeseries starts a full warmup lead-in before `time.start_time`.
5. Column names in `variables_names_map` match your forcing headers exactly, units included.
6. `output_root` is inside your Docker output mount, if using Docker.

With configs in hand, see [4-run_standalone](./4-run_standalone.md) for a direct Python forward or [5-run_ngen](./5-run_ngen.md) to run under NextGen.
