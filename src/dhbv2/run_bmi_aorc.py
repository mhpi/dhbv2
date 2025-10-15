"""
Forward BMI on USGS AORC data for the Juniata River Basin using the NextGen
hydrofabric.
"""

import os
from pathlib import Path

import numpy as np
from bmi import DeltaModelBmi as Bmi

### Configuration Settings (Single-catchment Run) ###
BASIN_ID = "cat-88306"
BMI_CFG_PATH = f"bmi_config/bmi_config_{BASIN_ID}_5yr.yaml"

FORC_PATH = f"data/aorc/juniata_river_basin/forcings_5yr_{BASIN_ID}.npy"
ATTR_PATH = f"data/aorc/juniata_river_basin/attributes_5yr_{BASIN_ID}.npy"
# OBS_PATH = f'../../data/aorc/juniata_river_basin/obs_5yr_{basin_id}.npy'
### ------------------------------------ ###


pkg_root = Path(__file__).parent.parent.parent
forc = np.load(os.path.join(pkg_root, Path(FORC_PATH)))
attr = np.load(os.path.join(pkg_root, Path(ATTR_PATH)))
bmi_cfg_path_full = os.path.join(pkg_root, Path(BMI_CFG_PATH))
# obs = np.load(os.path.join(pkg_root, Path(OBS_PATH)))

# Create dHBV 2.0 BMI instance
model = Bmi(config_path=bmi_cfg_path_full)

streamflow_pred = np.zeros(forc.shape[0])
nan_idx = []

# # 1) Compile forcing data within BMI to do batch run.
# for i in range(0, forc.shape[0]):
#     # Extract forcing/attribute data for the current time step
#     prcp = forc[i, :, 0]
#     temp = forc[i, :, 1]
#     pet = forc[i, :, 2]

#     ## Check if any of the inputs are NaN
#     if np.isnan([prcp, temp, pet]).any():
#         # if model.verbose > 0:
#         print(f"Skipping timestep {i} due to NaN values in inputs.")
#         nan_idx.append(i)
#         continue

#     model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate', prcp)
#     model.set_value('land_surface_air__temperature', temp)
#     model.set_value('land_surface_water__potential_evaporation_volume_flux', pet)


# ### BMI initialization ###
# model.initialize()

# # 2) DO pseudo model forward and return pre-predicted values at each timestep
# for i in range(0, forc.shape[0]):
#     if i in nan_idx:
#         # Skip the update for this timestep
#         continue

#     ### BMI update ###
#     model.update()

#     # Retrieve and scale the runoff output
#     dest_array = np.zeros(1)
#     model.get_value('land_surface_water__runoff_volume_flux', dest_array)

#     streamflow_pred[i] = dest_array[0]  # Convert to mm/day -> mm/hr

#  ### BMI finalization ###
# model.finalize()

# print("\n=/= -- Streamflow prediction completed -- =/=")
# print(f"    Basin ID:              {BASIN_ID}")
# print(f"    Total Process Time:    {model.bmi_process_time:.4f} seconds")
# print(f"    Mean streamflow:       {streamflow_pred.mean():.4f} mm/day")
# print(f"    Max streamflow:        {streamflow_pred.max():.4f} mm/day")
# print(f"    Min streamflow:        {streamflow_pred.min():.4f} mm/day")
# print("=/= ------------------------------------- =/=")


# # Calculate NSE for the model predictions
# obs = obs.dropna()
# sim = streamflow_pred.dropna()

# denom = ((obs - obs.mean()) ** 2).sum()
# num = ((sim - obs) ** 2).sum()
# nse = 1 - num / denom
# print(f"NSE: {nse:.2f}")
