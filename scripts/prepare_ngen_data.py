"""Building hourly CAMELS forcing dataset.
3 catchments, 2010-2012.

Mirrors structure of ./ngen/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv

@leoglonz
"""

import numpy as np
import pandas as pd
import xarray as xr

n_cat = 3
t_start = '2008-01-09 00:00:00'
t_end = '2015-12-30 23:00:00'

example_path = '/projects/mhpi/leoglonz/ciroh-ua/ciroh-ua-ngen/data/forcing/cat-67_2015-12-01 00_00_00_2015-12-30 23_00_00.csv'
camels_path = '/gpfs/yxs275/data/hourly/CAMELS_HF/forcing/forcing_1990_2018_gauges_hourly_00000_00499.nc'
out_path = f'/projects/mhpi/leoglonz/ciroh-ua/dhbv2_mts/ngen_resources/data/forcing/camels_{t_start.replace(":", "_").replace(" ", "_")}_{t_end.replace(":", "_").replace(" ", "_")}.nc'

df = pd.read_csv(example_path)
camels_xr = xr.open_dataset(camels_path)

out_df = pd.DataFrame()


def transform_dataset(ds_in):
    """Transform CAMELS dataset to match target structure."""
    # 1. Rename Dimensions first
    ds = ds_in.rename({'gauge': 'catchment-id'})

    # --- INSERT SUBSETTING HERE (Before creating new vars or dropping coords) ---
    # Use .isel (index select) for the first 100 catchments
    # Use .sel (label select) for the specific date range
    ds = ds.isel({'catchment-id': slice(0, n_cat)})
    ds = ds.sel(time=slice(t_start, t_end))
    # --------------------------------------------------------------------------

    # 2. Rename Existing Variables
    ds = ds.rename(
        {
            'P': 'precip_rate',
            'T': 'TMP_2maboveground',
            'PET': 'PET_hargreaves',
        },
    )

    for var in ds.variables:
        ds[var].encoding = {}

    # 3. Create 'ids' Variable
    raw_ids = ds['catchment-id'].values.astype(str)

    # Prepend "cat-" using numpy's string operations
    # cat_ids will be like ["cat-2453", "cat-2454", ...]
    cat_ids = np.char.add('cat-', raw_ids)

    ds['ids'] = (('catchment-id',), cat_ids)

    # 4. Create 'Time' Variable with Nanosecond Units
    # .view('int64') accesses the raw nanoseconds from the datetime64[ns] object
    # .astype('float64') converts that integer count to a float
    time_values = ds['time'].values.view('int64').astype('float64')

    time_broadcasted = np.tile(
        time_values,
        (ds.sizes['catchment-id'], 1),
    )

    ds['Time'] = (('catchment-id', 'time'), time_broadcasted)
    ds['Time'].attrs = {'units': 'ns'}

    # 5. Create Zero-Filled Variables
    zero_vars = [
        'APCP_surface',
        'DLWRF_surface',
        'DSWRF_surface',
        'PRES_surface',
        'SPFH_2maboveground',
        'UGRD_10maboveground',
        'VGRD_10maboveground',
    ]

    # Create zeros based on the new subset shape
    shape = (ds.sizes['catchment-id'], ds.sizes['time'])
    zeros = np.zeros(shape, dtype='float64')

    for var in zero_vars:
        ds[var] = (('catchment-id', 'time'), zeros)
        ds[var].encoding = {'_FillValue': None}  # Ensure no default fill values exist

    # Convert to temp to kelvin
    ds['TMP_2maboveground'] = ds['TMP_2maboveground'].astype('float64') + np.float64(
        273.15,
    )
    ds['TMP_2maboveground'].attrs['units'] = 'K'
    ds['precip_rate'].attrs['units'] = 'mm hr-1'
    ds['PET_hargreaves'].attrs['units'] = 'mm hr-1'
    ds['APCP_surface'].attrs['units'] = 'kg m-2'
    ds['DLWRF_surface'].attrs['units'] = 'W m-2'
    ds['DSWRF_surface'].attrs['units'] = 'W m-2'
    ds['PRES_surface'].attrs['units'] = 'Pa'
    ds['SPFH_2maboveground'].attrs['units'] = 'g g-1'
    ds['UGRD_10maboveground'].attrs['units'] = 'm s-1'
    ds['VGRD_10maboveground'].attrs['units'] = 'm s-1'

    ds['TMP_2maboveground'].attrs['units'] = 'K'
    ds['precip_rate'].attrs['units'] = 'mm hr-1'
    ds['PET_hargreaves'].attrs['units'] = 'mm hr-1'

    ds['TMP_2maboveground'].encoding = {'dtype': 'float64'}

    # 6. NOW it is safe to drop the coordinates
    ds = ds.drop_vars(['time', 'catchment-id'])

    return ds


formatted_ds = transform_dataset(camels_xr)

# save to nc
formatted_ds.to_netcdf(
    out_path,
    format='NETCDF4',
    engine='netcdf4',
)


def verify_integrity(original_path, new_path, t_start, t_end, n_cat):
    """Verify that the data transformation preserved integrity."""
    print("Verifying data integrity...")
    ds_old = xr.open_dataset(original_path)
    ds_new = xr.open_dataset(new_path)

    # Slice old dataset exactly like the new one for comparison
    ds_old_sub = ds_old.isel(gauge=slice(0, n_cat)).sel(time=slice(t_start, t_end))

    # 1. Verify Precipitation (Direct Copy Check)
    # Using strict equality because no math was done on P
    np.testing.assert_array_equal(
        ds_old_sub['P'].values,
        ds_new['precip_rate'].values,
        err_msg="Precipitation values drifted!",
    )
    print("✓ Precipitation matches exactly.")

    # 2. Verify Temperature (Math Check)
    # Allow float32 epsilon tolerance (approx 1e-6)
    expected_temp = ds_old_sub['T'].values
    print(ds_old_sub['T'].sel(time='2008-01-09T00:00:00')[0].item())
    print((ds_new['TMP_2maboveground'].values - 273.15)[0][0].item())
    np.testing.assert_allclose(
        expected_temp,
        ds_new['TMP_2maboveground'].values - 273.15,
        rtol=1e-16,
        err_msg="Temperature math introduced errors!",
    )
    print("✓ Temperature conversion matches within float64 tolerance.")

    print("SUCCESS: No numerical discrepancies detected.")


# # Run verification
# verify_integrity(camels_path, out_path, t_start, t_end, n_cat)
