"""
Fetch catchment attributes from CAMELS netCDF and update existing BMI config
YAML files accordingly.

TODO: This script should support adding catchments attributes for any catchment
in the CONUS HF.
"""

import xarray as xr
from ruamel.yaml import YAML

CAT = 2453
attrs_path = '/projects/mhpi/yxs275/hourly_model/mtsHBV/data/CAMELS_HFs_attr_new.nc'
target_path = f'/projects/mhpi/leoglonz/ciroh-ua/dhbv2_mts/ngen_resources/data/dhbv2_mts/config/bmi_cat-{CAT}.yaml'


attrs_ds = xr.open_dataset(attrs_path).sel(gauge=CAT)

# Initialize the preserved-format loader
yml = YAML()
yml.preserve_quotes = True

# 1. Load data
with open(target_path) as f:
    config_data = yml.load(f)

# 2. Update existing keys
for var_name in attrs_ds.data_vars:
    key = str(var_name)
    if key in config_data:
        # Rounding is often nice for config files to avoid 10.55000019
        val = attrs_ds[var_name].item()
        config_data[key] = val

# 3. Dump back (Comments and structure are preserved automatically)
with open(target_path, 'w') as f:
    yml.dump(config_data, f)
