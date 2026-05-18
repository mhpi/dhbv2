from typing import Optional

import numpy as np
import pandas as pd


def hargreaves_pet(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tmean: np.ndarray,
    lat: np.ndarray,
    day_of_year: np.ndarray,
) -> np.ndarray:
    """
    Calculate potential evapotranspiration (PET) using Hargreaves method and
    daily forcings.

    Parameters
    ----------
    tmin
        Minimum daily temperature [degC]
    tmax
        Maximum daily temperature [degC]
    tmean
        Mean daily temperature [degC]
    lat
        Latitude [deg]
    day_of_year
        Day of the year (1-365/366)

    Returns
    -------
    ndarray
        Potential evapotranspiration [mm day-1]
    """
    SOLAR_CONSTANT = 0.0820

    trange = np.maximum(tmax - tmin, 0)

    sol_dec = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)
    sha = np.arccos(np.clip(-np.tan(lat) * np.tan(sol_dec), -1, 1))
    ird = 1 + 0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year)

    et_rad = (
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT
        * ird
        * (
            sha * np.sin(lat) * np.sin(sol_dec)
            + np.cos(lat) * np.cos(sol_dec) * np.sin(sha)
        )
    )

    pet = 0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * et_rad
    pet = np.maximum(pet, 0)
    return pet


def calc_hourly_hargreaves_pet(
    temp,
    srad,
    lat,
    time_idx,
):
    """
    Compute hourly PET using daily Hargreaves but with hourly inputs.
    PET is allocated to hours according to fractional shortwave radiation.

    Returns
    -------
    ndarray
        PET in mm/hour
    """
    # 1. Convert inputs
    lat_rad = np.deg2rad(lat)
    Nh = len(temp)

    # 2. Extract daily groupings
    time_index = pd.to_datetime(time_idx)
    days = time_index.normalize().unique()
    Ndays = len(days)

    # Prepare arrays
    tmin_d = np.zeros(Ndays)
    tmax_d = np.zeros(Ndays)
    tmean_d = np.zeros(Ndays)
    doy_d = np.zeros(Ndays, dtype=int)
    sw_sum_d = np.zeros(Ndays)

    # Map each hour to its day index
    day_index = time_index.normalize()
    day_to_idx = {d: i for i, d in enumerate(days)}
    idx_daily = np.array([day_to_idx[d] for d in day_index])

    # 3. Aggregate hourly → daily
    for d_idx, day in enumerate(days):
        mask = idx_daily == d_idx
        th = temp[mask]
        sw = srad[mask]

        tmin_d[d_idx] = th.min()
        tmax_d[d_idx] = th.max()
        tmean_d[d_idx] = th.mean()
        sw_sum_d[d_idx] = sw.sum()
        doy_d[d_idx] = day.timetuple().tm_yday

    # 4. Compute DAILY PET
    pet_daily = hargreaves_pet(tmin_d, tmax_d, tmean_d, lat_rad, doy_d)

    # 5. Distribute daily PET to hourly PET
    pet_hourly = np.zeros(Nh)

    for h in range(Nh):
        d = idx_daily[h]
        daily_sw = sw_sum_d[d]
        # If no radiation (polar night or bad data), spread uniformly
        if daily_sw > 0:
            frac = srad[h] / daily_sw
        else:
            frac = 1.0 / np.sum(idx_daily == d)
        pet_hourly[h] = pet_daily[d] * frac

    return pet_hourly


def penman_monteith_pet(
    temp: np.ndarray,
    spfh: np.ndarray,
    dlwrf: np.ndarray,
    dswrf: np.ndarray,
    pres: np.ndarray,
    ugrd_10m: np.ndarray,
    vgrd_10m: np.ndarray,
    albedo: Optional[float] = 0.23,
    eps_s: Optional[float] = 0.98,
) -> np.ndarray:
    """
    Calculate hourly potential evapotranspiration (PET; mm hr-1) using the FAO-56
    Penman-Monteith method.

    Reference: https://www.fao.org/4/x0490e/x0490e06.htm

    NOTE: This formulation assumes (and converts) AORC inputs
        https://registry.opendata.aws/noaa-nws-aorc/

    NOTE: True FAO-56 PM will use atmospheric emissivity to estimate net
        longwave from air temperature. Since we have longwave directly from AORC,
        we compute net longwave as Rl_down - Rl_up, where upwelling longwave is
        estimated via the Stefan-Boltzmann law using air temperature as a proxy
        for surface temperature (true LST is not available in AORC).

    Parameters
    ----------
    temp
        Air temperature [degC].
    spfh
        Specific humidity [g g-1].
    dlwrf
        Downward longwave radiation [W m-2].
    dswrf
        Downward shortwave radiation [W m-2].
    pres
        Atmospheric pressure [Pa].
    ugrd_10m
        U-component of wind at 10 meters [m s-1].
    vgrd_10m
        V-component of wind at 10 meters [m s-1].
    albedo
        Surface albedo (default 0.23 is for grass reference crop).
    eps_s
        Surface emissivity (default 0.98 is for grass reference crop).

    Returns
    -------
    ndarray
        Potential evapotranspiration [mm hr-1].
    """
    # Auto-detect AORC spfh (g/g * 1000) and convert to fraction
    spfh = np.where(spfh > 0.02, spfh / 1000.0, spfh)

    # Convert pressure to kPa
    pressure = pres / 1000.0

    # Convert wind speed at 10m elevation (AORC) to 2m
    u10 = np.sqrt(ugrd_10m**2 + vgrd_10m**2)
    u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)

    # Psychrometric constant [kPa/degC]
    gamma = 0.000665 * pressure

    # Saturation vapor pressure [kPa]
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))

    # Actual vapor pressure [kPa]
    ea = (spfh * pressure) / (0.622 + 0.378 * spfh)

    # Slope of saturation vapor pressure curve [kPa/degC]
    delta = (4098 * es) / (temp + 237.3) ** 2

    # Convert srad to MJ/m2/hr
    Rs = dswrf * 0.0036  # shortwave (incoming)
    Rl_down = dlwrf * 0.0036  # longwave (downwelling)

    # Estimate upwelling longwave w/ Stefan–Boltzmann [MJ/K4/m2/hr]
    sigma = 4.903e-9 / 24.0

    Rl_up = eps_s * sigma * ((temp + 273.15) ** 4)

    Rn = (1 - albedo) * Rs + (Rl_down - Rl_up)

    # Soil heat flux
    G = np.where(Rn > 0, 0.1 * Rn, 0.5 * Rn)

    # drag coefficient
    cd = np.where(Rn > 0, 0.24, 0.96)

    # Penman–Monteith FAO-56 hourly PET [mm/hr]
    num = 0.408 * delta * (Rn - G) + gamma * (37.0 / (temp + 273.15)) * u2 * (es - ea)
    denom = delta + gamma * (1 + cd * u2)
    pet = num / denom

    return np.maximum(pet, 0.0)
