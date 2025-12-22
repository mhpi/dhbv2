"""
Add Hargreaves PET calculation to NextGen forcing datasets.

Usage:
    - e.g., on nextgen forcing NetCDF/CSV files (./ngen/data/forcing/)
@leoglonz
"""

from pathlib import Path

import pandas as pd

from dhbv2.pet import calc_hourly_hargreaves_pet

# Setup pathing
pkg_root = Path(__file__).parent.parent.parent.parent

# -------- Settings --------
# Input CSV file with NextGen forcing data
# Make sure it has at least 'time', 'TMP_2maboveground', 'DSWRF_surface' columns
input_csv = (
    f"{pkg_root}/ngen/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv"
)
lat = 46.5393476  # Latitude of the catchment (in degrees)

# Output CSV file with added PET column
output_csv = f"{pkg_root}/ngen_resources/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv"
# --------------------------


df = pd.read_csv(input_csv)
df["time"] = pd.to_datetime(df["time"])
temp_c = df["TMP_2maboveground"].values - 273.15  # convert K to C

# shortwave radiation directly
srad = df["DSWRF_surface"].values

# compute hourly PET
pet_hourly = calc_hourly_hargreaves_pet(
    temp=temp_c,
    srad=srad,
    lat=lat,
    time_idx=df["time"].values,
)
df["PET_hargreaves"] = pet_hourly

df.to_csv(output_csv, index=False)
