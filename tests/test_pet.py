"""
Tests for PET (potential evapotranspiration) calculation functions.

Coverage:
- hargreaves_pet: non-negativity, shape, monotonicity, edge cases
- penman_monteith_pet: non-negativity, shape, humidity conversion, edge cases
- calc_hourly_hargreaves_pet: distribution correctness, zero-radiation fallback

NOTE: These are pure numerical functions with no external dependencies.
Reference values are verified against known physical constraints rather
than hard-coded outputs, since the formulas are well-established
(FAO-56, Hargreaves-Samani).
"""

import numpy as np
import pandas as pd
import pytest

from dhbv2.pet import (
    calc_hourly_hargreaves_pet,
    hargreaves_pet,
    penman_monteith_pet,
)


# ---------------------------------------------------------------------------- #
#  Hargreaves PET
# ---------------------------------------------------------------------------- #


class TestHargreavesPet:
    """Verify Hargreaves-Samani PET calculation."""

    def test_output_non_negative(self):
        """PET should never be negative."""
        pet = hargreaves_pet(
            tmin=np.array([5.0]),
            tmax=np.array([25.0]),
            tmean=np.array([15.0]),
            lat=np.deg2rad(40.0),
            day_of_year=np.array([172]),
        )
        assert np.all(pet >= 0), f"Negative PET: {pet}"

    def test_output_shape_matches_input(self):
        """Output shape should match input array shapes."""
        n = 10
        pet = hargreaves_pet(
            tmin=np.zeros(n),
            tmax=np.ones(n) * 30,
            tmean=np.ones(n) * 15,
            lat=np.deg2rad(40.0),
            day_of_year=np.arange(1, n + 1),
        )
        assert pet.shape == (n,)

    def test_higher_temp_range_increases_pet(self):
        """Larger temperature range should increase PET (more solar energy)."""
        kwargs = {
            'tmean': np.array([20.0]),
            'lat': np.deg2rad(40.0),
            'day_of_year': np.array([172]),
        }
        pet_narrow = hargreaves_pet(
            tmin=np.array([18.0]),
            tmax=np.array([22.0]),
            **kwargs,
        )
        pet_wide = hargreaves_pet(
            tmin=np.array([10.0]),
            tmax=np.array([30.0]),
            **kwargs,
        )
        assert pet_wide > pet_narrow, (
            f"Wider temp range should give higher PET: "
            f"wide={pet_wide[0]:.4f}, narrow={pet_narrow[0]:.4f}"
        )

    def test_zero_temp_range_gives_zero_pet(self):
        """Zero temperature range should give zero PET (sqrt(0) = 0)."""
        pet = hargreaves_pet(
            tmin=np.array([20.0]),
            tmax=np.array([20.0]),
            tmean=np.array([20.0]),
            lat=np.deg2rad(40.0),
            day_of_year=np.array([172]),
        )
        np.testing.assert_allclose(pet, 0.0, atol=1e-10)

    def test_reversed_tmin_tmax_clipped(self):
        """When tmin > tmax, temp range should be clipped to 0."""
        pet = hargreaves_pet(
            tmin=np.array([30.0]),
            tmax=np.array([20.0]),
            tmean=np.array([25.0]),
            lat=np.deg2rad(40.0),
            day_of_year=np.array([172]),
        )
        np.testing.assert_allclose(pet, 0.0, atol=1e-10)

    def test_equator_summer_reasonable_value(self):
        """Equatorial summer PET should be in a reasonable range (2-8 mm/d)."""
        pet = hargreaves_pet(
            tmin=np.array([20.0]),
            tmax=np.array([35.0]),
            tmean=np.array([27.5]),
            lat=0.0,
            day_of_year=np.array([172]),
        )
        assert 2.0 < pet[0] < 8.0, (
            f"Equatorial summer PET outside reasonable range: {pet[0]:.2f} mm/d"
        )

    def test_scalar_inputs(self):
        """Function should handle scalar (non-array) inputs gracefully."""
        pet = hargreaves_pet(
            tmin=5.0,
            tmax=25.0,
            tmean=15.0,
            lat=np.deg2rad(45.0),
            day_of_year=180,
        )
        assert np.isfinite(pet)
        assert pet >= 0

    def test_output_finite(self):
        """Output should contain no NaN or Inf values."""
        pet = hargreaves_pet(
            tmin=np.array([-10.0, 0.0, 10.0]),
            tmax=np.array([0.0, 15.0, 30.0]),
            tmean=np.array([-5.0, 7.5, 20.0]),
            lat=np.deg2rad(45.0),
            day_of_year=np.array([1, 91, 182]),
        )
        assert np.all(np.isfinite(pet)), f"Non-finite PET values: {pet}"


# ---------------------------------------------------------------------------- #
#  Penman-Monteith PET
# ---------------------------------------------------------------------------- #


class TestPenmanMonteithPet:
    """Verify FAO-56 Penman-Monteith PET calculation."""

    @pytest.fixture
    def typical_inputs(self):
        """Typical midday summer conditions."""
        return {
            'temp': np.array([25.0]),
            'spfh': np.array([0.010]),
            'dlwrf': np.array([300.0]),
            'dswrf': np.array([500.0]),
            'pres': np.array([101.3]),
            'ugrd_10m': np.array([2.0]),
            'vgrd_10m': np.array([2.0]),
        }

    def test_output_non_negative(self, typical_inputs):
        """PET should never be negative."""
        pet = penman_monteith_pet(**typical_inputs)
        assert np.all(pet >= 0), f"Negative PET: {pet}"

    def test_output_shape_matches_input(self):
        """Output shape should match input array shapes."""
        n = 5
        pet = penman_monteith_pet(
            temp=np.ones(n) * 25,
            spfh=np.ones(n) * 0.01,
            dlwrf=np.ones(n) * 300,
            dswrf=np.ones(n) * 500,
            pres=np.ones(n) * 101.3,
            ugrd_10m=np.ones(n) * 2,
            vgrd_10m=np.ones(n) * 2,
        )
        assert pet.shape == (n,)

    def test_daytime_reasonable_value(self, typical_inputs):
        """Daytime PET should be in a reasonable range (0.05-1.0 mm/hr)."""
        pet = penman_monteith_pet(**typical_inputs)
        assert 0.05 < pet[0] < 1.0, (
            f"Daytime PET outside reasonable range: {pet[0]:.4f} mm/hr"
        )

    def test_nighttime_low_pet(self):
        """Nighttime (no shortwave) PET should be near zero or zero."""
        pet = penman_monteith_pet(
            temp=np.array([15.0]),
            spfh=np.array([0.008]),
            dlwrf=np.array([250.0]),
            dswrf=np.array([0.0]),
            pres=np.array([101.3]),
            ugrd_10m=np.array([1.0]),
            vgrd_10m=np.array([1.0]),
        )
        assert pet[0] < 0.15, f"Nighttime PET unexpectedly high: {pet[0]:.4f} mm/hr"

    def test_higher_radiation_increases_pet(self):
        """Higher shortwave radiation should increase PET."""
        base = {
            'temp': np.array([25.0]),
            'spfh': np.array([0.01]),
            'dlwrf': np.array([300.0]),
            'pres': np.array([101.3]),
            'ugrd_10m': np.array([2.0]),
            'vgrd_10m': np.array([2.0]),
        }
        pet_low = penman_monteith_pet(dswrf=np.array([200.0]), **base)
        pet_high = penman_monteith_pet(dswrf=np.array([800.0]), **base)
        assert pet_high > pet_low, (
            f"Higher radiation should give higher PET: "
            f"high={pet_high[0]:.4f}, low={pet_low[0]:.4f}"
        )

    def test_zero_wind_still_computes(self, typical_inputs):
        """Zero wind speed should produce valid (non-NaN) PET."""
        typical_inputs['ugrd_10m'] = np.array([0.0])
        typical_inputs['vgrd_10m'] = np.array([0.0])
        pet = penman_monteith_pet(**typical_inputs)
        assert np.isfinite(pet[0]), "PET is not finite with zero wind"
        assert pet[0] >= 0

    def test_humidity_conversion_high_spfh(self):
        """AORC convention: spfh > 0.02 should be divided by 1000."""
        # With spfh=0.01 (normal): no conversion
        pet_normal = penman_monteith_pet(
            temp=np.array([25.0]),
            spfh=np.array([0.01]),
            dlwrf=np.array([300.0]),
            dswrf=np.array([500.0]),
            pres=np.array([101.3]),
            ugrd_10m=np.array([2.0]),
            vgrd_10m=np.array([2.0]),
        )
        # With spfh=10.0 (AORC g/g * 1000): converted to 0.01
        pet_aorc = penman_monteith_pet(
            temp=np.array([25.0]),
            spfh=np.array([10.0]),
            dlwrf=np.array([300.0]),
            dswrf=np.array([500.0]),
            pres=np.array([101.3]),
            ugrd_10m=np.array([2.0]),
            vgrd_10m=np.array([2.0]),
        )
        np.testing.assert_allclose(
            pet_normal,
            pet_aorc,
            rtol=1e-6,
            err_msg=(
                "PET should be equivalent whether spfh is in fraction "
                "or AORC (g/g * 1000) units"
            ),
        )

    def test_output_finite(self, typical_inputs):
        """Output should contain no NaN or Inf values."""
        pet = penman_monteith_pet(**typical_inputs)
        assert np.all(np.isfinite(pet)), f"Non-finite PET values: {pet}"

    def test_custom_albedo(self, typical_inputs):
        """Higher albedo should reduce PET (less net radiation absorbed)."""
        pet_default = penman_monteith_pet(**typical_inputs)
        pet_high_albedo = penman_monteith_pet(**typical_inputs, albedo=0.50)
        assert pet_default > pet_high_albedo, (
            f"Higher albedo should reduce PET: "
            f"default={pet_default[0]:.4f}, high={pet_high_albedo[0]:.4f}"
        )


# ---------------------------------------------------------------------------- #
#  Hourly Hargreaves PET
# ---------------------------------------------------------------------------- #


class TestCalcHourlyHargreavesPet:
    """Verify hourly PET distribution via Hargreaves method."""

    @pytest.fixture
    def hourly_inputs(self):
        """48 hours of synthetic hourly data (2 days)."""
        n_hours = 48
        time_idx = pd.date_range('2020-06-21', periods=n_hours, freq='h')

        # Diurnal temperature cycle
        hours = np.arange(n_hours) % 24
        temp = 20.0 + 8.0 * np.sin(2 * np.pi * (hours - 6) / 24)

        # Shortwave radiation: positive during daytime (6-18h)
        srad = np.where(
            (hours >= 6) & (hours <= 18),
            500 * np.sin(np.pi * (hours - 6) / 12),
            0.0,
        )

        return {
            'temp': temp,
            'srad': srad,
            'lat': 40.0,
            'time_idx': time_idx,
        }

    def test_output_non_negative(self, hourly_inputs):
        """Hourly PET should never be negative."""
        pet = calc_hourly_hargreaves_pet(**hourly_inputs)
        assert np.all(pet >= 0), "Negative hourly PET values found"

    def test_output_shape_matches_input(self, hourly_inputs):
        """Output length should match number of input hours."""
        pet = calc_hourly_hargreaves_pet(**hourly_inputs)
        assert pet.shape == (48,)

    def test_daily_sum_matches_daily_pet(self, hourly_inputs):
        """Sum of hourly PET for each day should equal daily PET."""
        pet_hourly = calc_hourly_hargreaves_pet(**hourly_inputs)

        # Compute daily PET directly for day 1 (first 24h)
        day1_temp = hourly_inputs['temp'][:24]
        tmin, tmax, tmean = day1_temp.min(), day1_temp.max(), day1_temp.mean()
        lat_rad = np.deg2rad(hourly_inputs['lat'])
        doy = pd.Timestamp('2020-06-21').timetuple().tm_yday

        pet_daily = hargreaves_pet(
            np.array([tmin]),
            np.array([tmax]),
            np.array([tmean]),
            lat_rad,
            np.array([doy]),
        )

        hourly_sum = pet_hourly[:24].sum()
        np.testing.assert_allclose(
            hourly_sum,
            pet_daily[0],
            rtol=1e-6,
            err_msg=(
                f"Sum of hourly PET ({hourly_sum:.4f}) != daily PET ({pet_daily[0]:.4f})"
            ),
        )

    def test_zero_radiation_uniform_distribution(self):
        """If daily shortwave is zero, PET should be distributed uniformly."""
        n_hours = 24
        time_idx = pd.date_range('2020-12-21', periods=n_hours, freq='h')
        temp = np.ones(n_hours) * 5.0 + np.arange(n_hours) * 0.5  # some range
        srad = np.zeros(n_hours)  # No radiation

        pet = calc_hourly_hargreaves_pet(
            temp=temp,
            srad=srad,
            lat=70.0,  # High latitude polar night
            time_idx=time_idx,
        )

        # All hourly values should be equal (uniform distribution)
        if pet.sum() > 0:
            nonzero = pet[pet > 0]
            np.testing.assert_allclose(
                nonzero,
                nonzero[0],
                rtol=1e-6,
                err_msg="Hourly PET should be uniformly distributed with zero radiation",
            )

    def test_radiation_weighted_distribution(self, hourly_inputs):
        """Hours with more radiation should receive more PET."""
        pet = calc_hourly_hargreaves_pet(**hourly_inputs)
        srad = hourly_inputs['srad']

        # Noon (hour 12) should have more PET than 6AM (hour 6)
        noon_idx = 12
        dawn_idx = 6
        if srad[noon_idx] > srad[dawn_idx] and pet[noon_idx] > 0:
            assert pet[noon_idx] > pet[dawn_idx], (
                f"Noon PET ({pet[noon_idx]:.4f}) should exceed dawn PET "
                f"({pet[dawn_idx]:.4f}) due to higher radiation"
            )
