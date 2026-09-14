"""Generate a CSV of HydroFabric catchments used in dhbv2 runoff simulations.

This CSV is used to build a catchment adjacency matrix necessary for routing
simulations with the Distributed Differentiable Routing (DDR) package.

See ./docs/7-routing.md for more information on routing with DDR.
"""

import os
from pathlib import Path

import pandas as pd

# Setup pathing
pkg_root = Path(__file__).parent

### -------- Settings -------- ###
# Define all simulated catchments as full names ('wb-123456') or ids ('123456').
TARGET_CATCHMENTS = [
    'wb-2453',
    'wb-2454',
    'wb-2455',
]
OUT_PATH = os.path.join(pkg_root, 'data/catchments.csv')
### -------------------------- ###


def main():
    """Main function to create catchment CSV."""
    if not TARGET_CATCHMENTS:
        print("Error: Target catchment list is empty!")
        return

    print(f"Preparing to write {len(TARGET_CATCHMENTS)} catchments to CSV...")

    # Create the DataFrame
    # 'STAID' maps to the ID
    # 'DRAIN_SQKM' is required by the validator, so we fill it with 1.0
    df = pd.DataFrame(
        {
            "STAID": TARGET_CATCHMENTS,
            "DRAIN_SQKM": 1.0,
        },
    )

    # Ensure STAID is treated as a string to match the CSV format expected
    df["STAID"] = df["STAID"].astype(str)
    df["STANAME"] = df["STAID"]  # Name can just be the ID
    df["LAT_GAGE"] = 0.0  # Dummy Latitude
    df["LNG_GAGE"] = 0.0
    # Save to CSV
    # index=False ensures we don't write the row numbers to the file
    df.to_csv(OUT_PATH, index=False)

    print(f"Success! CSV saved to: {os.path.abspath(OUT_PATH)}")
    print("\nPreview of file content:")
    print(df.head())


if __name__ == "__main__":
    main()
