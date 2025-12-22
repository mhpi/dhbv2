"""
Forward BMI on dummy data with pseudo-NextGen operating behavior to test
implementation.

TODO: WIP post-dhbv2 MTS update.

@leoglonz
"""

import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from dhbv2.bmi import DeltaModelBmi as Bmi

### Configuration Settings (single-catchment) ###
BASIN_ID = 'cat-88306'
CAT_IDX = 0
BMI_CONFIG_PATH = f'ngen_resources/data/dhbv2/config/bmi_{BASIN_ID}.yaml'
FORC_PATH = 'ngen_resources/data/forcing/dhbv_forcings.nc'
### ------------------------------------ ###


pkg_root = Path(__file__).parent.parent.parent
forc_path_full = os.path.join(pkg_root, Path(FORC_PATH))
bmi_config_path_full = os.path.join(pkg_root, Path(BMI_CONFIG_PATH))


def execute():
    """Execute the BMI forward model on dummy data with pseudo-NextGen operating
    behavior to test implementation.
    """
    # Create dHBV 2.0 BMI instance
    print(">>> Creating DeltaModelBmi instance")
    model = Bmi()

    ### BMI initialization ###
    print(">>> Initializing the BMI")
    model.initialize(config_path=bmi_config_path_full)

    print("[Preparing data]")
    forcing_data = Dataset(forc_path_full, mode="r")

    print(
        "[Looping through timesteps | Setting forcing/attribute values & forwarding model]",
    )
    f_dict = {}
    t_steps = forcing_data['Time'][:].shape[-1]

    for key in forcing_data.variables.keys():
        if key in ['P', 'Temp', 'PET']:
            f_dict[key] = forcing_data[key][CAT_IDX, :]

    f_dict['P'] = f_dict['P'] * 1000

    for t in range(t_steps):
        print(f"\n--- Timestep {t + 1}/{t_steps} ---")

        # Forcings
        model.set_value(
            'atmosphere_water__liquid_equivalent_precipitation_rate',
            f_dict['P'][t],
        )
        model.set_value('land_surface_air__temperature', f_dict['Temp'][t])
        model.set_value(
            'land_surface_water__potential_evaporation_volume_flux',
            f_dict['PET'][t],
        )

        ### BMI update ###
        print(">>> Doing BMI model update")
        model.update()

        dest_array = np.zeros(1)
        model.get_value('land_surface_water__runoff_volume_flux', dest_array)
        runoff = dest_array[0]

        print(
            f"Result: Streamflow at time {model.get_current_time()} ({model.get_time_units()}) is {runoff:.4f} m3/s",
        )

        # if t > 100:
        #     print(">>> Stopping the loop")
        #     break

    ### BMI finalization ###
    print(">>> Finalizing the BMI")
    model.finalize()


if __name__ == '__main__':
    execute()
