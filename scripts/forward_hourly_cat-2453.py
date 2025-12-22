"""
Forward dHBV2 MTS (hourly) BMI on single catchment with pseudo-NextGen
operating behavior.

We use catchment `cat-2453` (2454 and 2455 also available) on the CAMELS
dataset as an example, with forcing timeseries available from 2008 to 2011.

NOTE: As of writing, the MTS model requires 1yr (558 days) of spinup data prior
to simulation start. Therefore, this script will provide simulations starting
from 2009-01-01 using the provided data.

@leoglonz
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dhbv2.mts_bmi import MtsDeltaModelBmi as Bmi

log = logging.getLogger('BMI_Demo')
logging.basicConfig(level=logging.INFO)

### Configuration Settings (single-catchment) ###
CAT_ID = 'cat-2453'  # Options: cat-2453, cat-2454, cat-2455
BMI_CONFIG_PATH = './ngen_resources/data/dhbv2_mts/config/bmi_cat-2453.yaml'
FORCING_PATH = (
    './ngen_resources/data/forcing/camels_2008-01-09_00_00_00_2010-12-30_23_00_00.nc'
)
SAVE_OUTPUT = True
SAVE_PATH = f'./output/dhbv2_mts_hourly_{CAT_ID}_runoff.npy'
### ----------------------------------------- ###


# Setup pathing
pkg_root = Path(__file__).parent.parent
bmi_config_path = os.path.join(pkg_root, Path(BMI_CONFIG_PATH))
forcing_path = os.path.join(pkg_root, Path(FORCING_PATH))


# Create dHBV 2.0 BMI instance
log.info("Creating BMI instance")
model = Bmi(verbose=False)


### BMI initialization ###
log.info("Initializing BMI")
model.initialize(config_path=bmi_config_path)


log.info(f"Preparing data for catchment ID: {CAT_ID}")
ds = xr.open_dataset(forcing_path).set_coords('ids').swap_dims({'catchment-id': 'ids'})
forcings = ds.sel(ids=CAT_ID)
t_steps = len(forcings['time'])

# Maintain strict typing of forcing arrays
precip = forcings['precip_rate'].values.astype(np.float64)
temp = forcings['TMP_2maboveground'].values.astype(np.float64)
pet = forcings['PET_hargreaves'].values.astype(np.float64)


runoff_sim = []

log.info(f"Begin BMI update loop for {t_steps} steps")
for t in range(t_steps):
    time = pd.to_datetime(forcings['Time'].isel({'time': t}), unit='ns')
    # print(f"Current time: {time}, step {t}")

    # Set forcing values
    model.set_value(
        'atmosphere_water__liquid_equivalent_precipitation_rate',
        precip[t],
    )
    model.set_value(
        'land_surface_air__temperature',
        temp[t],
    )
    model.set_value(
        'land_surface_water__potential_evaporation_volume_flux',
        pet[t],
    )

    ### BMI update ###
    if t == 0:
        log.info("First timestep | Initial data loaded")
    model.update()

    dest_array = np.zeros(1)
    model.get_value('land_surface_water__runoff_volume_flux', dest_array)
    runoff_sim.append(dest_array[-1])

    if (t > 24 * 365) and (t % 1000 == 0):
        log.info(
            f" Time {model.get_current_time()} {model.get_time_units()} ({time}, step {t}) | Runoff {runoff_sim[-1]:.4f} m3/s",
        )


### BMI finalization ###
log.info("Finalizing BMI")
model.finalize()

if SAVE_OUTPUT:
    log.info(f"Saving output to {SAVE_PATH}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Remove spinup period (first 8592 hours)
    np.save(SAVE_PATH, np.array(runoff_sim)[8592:])
