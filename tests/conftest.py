"""Shared fixtures for dhbv2 BMI tests."""

import numpy as np
import pytest

from dhbv2.bmi import DeltaModelBmi
from dhbv2.mts_bmi import MtsDeltaModelBmi


@pytest.fixture
def daily_bmi():
    """Fresh DeltaModelBmi instance (not initialized).

    Provides access to constructor defaults, BMI info methods,
    variable dicts, and helper methods without requiring config
    files or model weights.
    """
    return DeltaModelBmi(verbose=False)


@pytest.fixture
def mts_bmi():
    """Fresh MtsDeltaModelBmi instance (not initialized).

    Provides access to constructor defaults, BMI info methods,
    variable dicts, and helper methods without requiring config
    files or model weights.
    """
    return MtsDeltaModelBmi(verbose=False)


@pytest.fixture
def daily_bmi_with_accumulator(daily_bmi):
    """DeltaModelBmi with a pre-filled day accumulator for aggregation tests.

    Sets up a 3-variable (P, T, PET) accumulator with 24 hours of
    synthetic data and Penman-Monteith PET method.
    """
    n_vars = 3
    daily_bmi._day_accumulator = np.zeros((24, 1, n_vars), dtype=np.float64)

    # Fill with known hourly values
    for h in range(24):
        daily_bmi._day_accumulator[h, 0, 0] = 0.5  # P: 0.5 mm/hr each hour
        daily_bmi._day_accumulator[h, 0, 1] = 15.0 + 5.0 * np.sin(
            2 * np.pi * h / 24,
        )  # T: diurnal cycle around 15C
        daily_bmi._day_accumulator[h, 0, 2] = max(
            0.0,
            0.3 * np.sin(2 * np.pi * (h - 6) / 24),
        )  # PET: daytime only

    daily_bmi._pet_method = 'penman_monteith'
    return daily_bmi
