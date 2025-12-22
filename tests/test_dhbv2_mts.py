"""
Check that runoff produced by dhbv2_mts matches validation benchmark.

This ensures your dhbv2 installation is behaving correctly.
"""

from pathlib import Path

import numpy as np

# Setup pathing
pkg_root = Path(__file__).parent


### -------- Settings -------- ###
# Numpy file with dhbv2 runoff data
SIM_PATH = f"{pkg_root}/output/dhbv2_mts_hourly_cat-2453_runoff.npy"
VAL_PATH = f"{pkg_root}/dhbv2_mts_hourly_cat-2453_runoff_benchmark.npy"
### -------------------------- ###


if __name__ == "__main__":
    # Load simulation and validation data
    runoff_sim = np.load(SIM_PATH)
    runoff_val = np.load(VAL_PATH)

    if runoff_sim.shape != runoff_val.shape:
        raise ValueError(
            f"Shape mismatch between simulation {runoff_sim.shape} "
            f"and validation {runoff_val.shape} data.",
        )

    # Compute difference
    difference = runoff_sim - runoff_val
    max_diff = np.max(np.abs(difference))
    mean_diff = np.mean(np.abs(difference))

    # Define acceptable tolerance
    TOLERANCE = 1e-7
    if max_diff > TOLERANCE:
        raise AssertionError(
            f"Runoff simulation does not match validation benchmark within "
            f"tolerance of {TOLERANCE}. Max error: {max_diff}",
        )
    else:
        print(
            f"Runoff simulation matches validation benchmark within "
            f"tolerance of {TOLERANCE}.\n"
            f"Max error: {max_diff}\n"
            f"Mean error: {mean_diff}",
        )
