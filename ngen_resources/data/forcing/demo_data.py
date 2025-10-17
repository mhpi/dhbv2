import xarray as xr
import numpy as np


FORC_PATH = "./awi_forcings_dhbv.nc"
ATTR_PATH = "./awi_attributes_dhbv.nc"


# 1. Add random PET data to forings file for demo
ds = xr.open_dataset(FORC_PATH)
rand_pet = np.random.uniform(
    low=0.0,
    high=10.0,
    size=ds["APCP_surface"].shape,
).astype(np.float32)

ds["PET"] = xr.DataArray(
    data=rand_pet,
    dims=["catchment-id", "time"],
    attrs={
        "units": "mm/day",  # or whatever unit you prefer
        "long_name": "Potential Evapotranspiration",
        "description": "Randomly generated PET for demonstration.",
    },
)


# 2. Create dummy attribute nc file
static_vars = [
    ("aridity", "-"),
    ("meanP", "mm d-1"),
    ("ETPOT_Hargr", "mm d-1"),
    ("NDVI", "-"),
    ("FW", "mm d-1"),
    ("meanslope", "m km-1"),
    ("SoilGrids1km_sand", "km2"),
    ("SoilGrids1km_clay", "km2"),
    ("SoilGrids1km_silt", "km2"),
    ("glaciers", "percent"),
    ("HWSD_clay", "percent"),
    ("HWSD_gravel", "percent"),
    ("HWSD_sand", "percent"),
    ("HWSD_silt", "percent"),
    ("meanelevation", "m"),
    ("meanTa", "degC"),
    ("permafrost", "-"),
    ("permeability", "m2"),
    ("seasonality_P", "-"),
    ("seasonality_PET", "-"),
    ("snow_fraction", "percent"),
    ("snowfall_fraction", "percent"),
    ("T_clay", "percent"),
    ("T_gravel", "percent"),
    ("T_sand", "percent"),
    ("T_silt", "percent"),
    ("Porosity", "-"),
    ("uparea", "km2"),
]

catchment_ids = ds["catchment-id"].values  # or ds.coords["catchment-id"].values


static_ds = xr.Dataset(
    coords={"catchment-id": catchment_ids},
)

for var_name, unit in static_vars:
    # Generate random data: shape = (n_catchments,)
    # Adjust distribution as needed (e.g., uniform, normal, bounded)
    if unit == "percent":
        data = np.random.uniform(0, 100, size=len(catchment_ids)).astype(np.float32)
    elif unit == "m2":
        # Permeability is often very small (e.g., 1e-15 to 1e-10)
        data = np.random.uniform(1e-16, 1e-10, size=len(catchment_ids)).astype(
            np.float32,
        )
    elif unit == "km2":
        # Basin or grid areas: assume 10 to 10,000 km2
        data = np.random.uniform(10, 10_000, size=len(catchment_ids)).astype(np.float32)
    elif unit == "m":
        # Elevation: 0 to 5000 m
        data = np.random.uniform(0, 5000, size=len(catchment_ids)).astype(np.float32)
    elif unit == "m km-1":
        # Slope: 0 to 200 m/km (i.e., 0–20% slope)
        data = np.random.uniform(0, 200, size=len(catchment_ids)).astype(np.float32)
    elif unit == "degC":
        # Mean temperature: -20 to 30°C
        data = np.random.uniform(-20, 30, size=len(catchment_ids)).astype(np.float32)
    elif unit == "mm d-1":
        # Fluxes: 0 to 10 mm/day
        data = np.random.uniform(0, 10, size=len(catchment_ids)).astype(np.float32)
    elif unit == "-":
        # Dimensionless: 0 to 1 (or 0 to 2 for ratios)
        if "ratio" in var_name or "seasonality" in var_name:
            data = np.random.uniform(0, 5, size=len(catchment_ids)).astype(np.float32)
        else:
            data = np.random.uniform(0, 1, size=len(catchment_ids)).astype(np.float32)
    else:
        # Fallback
        data = np.random.random(size=len(catchment_ids)).astype(np.float32)

    # Add to dataset
    static_ds[var_name] = xr.DataArray(
        data=data,
        dims=["catchment-id"],
        attrs={"units": unit},
    )

static_ds.attrs["title"] = "Static catchment attributes"
static_ds.attrs["created_by"] = "demo"
static_ds.attrs["date_created"] = str(np.datetime64("now"))

# --- 6. Save to NetCDF ---
static_ds.to_netcdf(ATTR_PATH)

print(f"✅ Static attributes saved to: {ATTR_PATH}")
