"""
Forward BMI on single catchment with pseudo-NextGen operating behavior.

We use catchment 67 which is included with NOAA-OWP/ngen.

@leoglonz
"""

import logging
import os
from pathlib import Path
import pandas as pd

import numpy as np
import xarray as xr

from dhbv2.mts_bmi import MtsDeltaModelBmi as Bmi

log = logging.getLogger('BMI_Demo')
logging.basicConfig(level=logging.INFO)

### Configuration Settings (single-catchment) ###
CAT_ID = 'cat-2453'
BMI_CONFIG_PATH = 'ngen_resources/data/dhbv2_mts/config/bmi_cat-2453.yaml'
FORCING_PATH = (
    # 'ngen_resources/data/forcing/camels_2010-01-01_00_00_00_2011-12-30_23_00_00.nc'
    '/projects/mhpi/leoglonz/ciroh-ua/dhbv2_mts/ngen_resources/data/forcing/camels_2008-01-09_00_00_00_2015-12-30_23_00_00.nc'
)
### ----------------------------------------- ###


# Setup pathing
pkg_root = Path(__file__).parent.parent
bmi_config_path = os.path.join(pkg_root, Path(BMI_CONFIG_PATH))
forcing_path = os.path.join(pkg_root, Path(FORCING_PATH))


# Create dHBV 2.0 BMI instance
log.info("Creating BMI instance")
model = Bmi(verbose=True)


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
    print(f"Current time: {time}, step {t}")

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

    if t > 24 * 365:
        log.info(
            f"Result: Streamflow at time {model.get_current_time()} ({model.get_time_units()}) is {runoff_sim[-1]:.4f} m3/s",
        )
        # runoff_sim_array = torch.tensor(runoff_sim)
        # torch.save(runoff_sim_array, f'/projects/mhpi/leoglonz/ciroh-ua/dmg/hf_outputs/cat-2543/ngen_qs.pt')


### BMI finalization ###
log.info("Finalizing BMI")
model.finalize()
