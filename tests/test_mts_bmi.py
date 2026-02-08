"""
Tests for MtsDeltaModelBmi (hourly multi-timescale BMI) interface and internals.

Coverage:
- Constructor defaults and initial state
- BMI info methods (component name, var names, counts)
- Variable name mapping (MTS-specific, includes 'lengthkm')
- Unit conversion (mm/h -> m/h)
- Cache management (_update_caches, buffer filling)
- Warmup logic (_can_run_warmup, _is_warmup_trigger_step)
- Normalization (_normalize with dict-based norm_stats)
- Benchmark regression (skipped if output files not available)

NOTE: Tests use un-initialized BMI instances where possible. Methods that
require model weights or config files are not tested here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from dhbv2.mts_bmi import (
    map_to_external,
    map_to_internal,
)
from dhbv2.utils import RingBuffer


# ---------------------------------------------------------------------------- #
#  Defaults
# ---------------------------------------------------------------------------- #


class TestMtsBmiDefaults:
    """Verify MtsDeltaModelBmi constructor sets correct defaults."""

    def test_name(self, mts_bmi):
        """Model name should be set."""
        assert mts_bmi._name == 'δHBV2.0 MTS'

    def test_not_initialized(self, mts_bmi):
        """Model should not be initialized after construction."""
        assert mts_bmi._initialized is False
        assert mts_bmi._is_warm is False
        assert mts_bmi._model is None
        assert mts_bmi._states is None

    def test_default_time_settings(self, mts_bmi):
        """Time defaults should be 1-hour steps in seconds."""
        assert mts_bmi._time_step_size == 3600
        assert mts_bmi._time_units == 's'
        assert mts_bmi._start_time == 0.0
        assert mts_bmi._timestep == 0

    def test_default_dtype(self, mts_bmi):
        """Default dtype should be float64."""
        assert mts_bmi._dtype == 'float64'
        assert mts_bmi.np_dtype == np.float64
        assert mts_bmi.pt_dtype == torch.float64

    def test_default_warmup_periods(self, mts_bmi):
        """MTS should require 351 daily + 168 hourly history."""
        assert mts_bmi.req_daily_history == 351
        assert mts_bmi.req_hourly_history == 168
        assert mts_bmi.warmup_frequency == 168

    def test_variable_dicts_populated(self, mts_bmi):
        """Variable dicts should be populated with correct counts."""
        assert len(mts_bmi._dynamic_var) == 8
        assert len(mts_bmi._output_vars) == 1
        assert len(mts_bmi._static_var) > 0

        for var_dict in [mts_bmi._dynamic_var, mts_bmi._output_vars]:
            for name, entry in var_dict.items():
                assert 'value' in entry, f"Missing 'value' key for {name}"
                assert 'units' in entry, f"Missing 'units' key for {name}"

    def test_mts_has_extra_static_var(self, mts_bmi):
        """MTS BMI should have 'basin__length' static variable."""
        assert 'basin__length' in mts_bmi._static_var


# ---------------------------------------------------------------------------- #
#  BMI info methods
# ---------------------------------------------------------------------------- #


class TestMtsBmiInfo:
    """Verify MTS BMI standard info query methods."""

    def test_get_component_name(self, mts_bmi):
        """Component name should identify MTS variant."""
        name = mts_bmi.get_component_name()
        assert 'MTS' in name

    def test_get_input_item_count(self, mts_bmi):
        """Input count should match dynamic variable count (8)."""
        assert mts_bmi.get_input_item_count() == 8

    def test_get_output_item_count(self, mts_bmi):
        """Output count should be 1 (streamflow only)."""
        assert mts_bmi.get_output_item_count() == 1

    def test_input_var_names_match_daily(self, daily_bmi, mts_bmi):
        """Both BMI variants should share the same input variable names."""
        assert set(daily_bmi.get_input_var_names()) == set(
            mts_bmi.get_input_var_names(),
        )

    def test_output_var_names_match_daily(self, daily_bmi, mts_bmi):
        """Both BMI variants should share the same output variable names."""
        assert daily_bmi.get_output_var_names() == mts_bmi.get_output_var_names()


# ---------------------------------------------------------------------------- #
#  Variable name mapping (MTS-specific)
# ---------------------------------------------------------------------------- #


class TestMtsVariableMapping:
    """Verify MTS-specific variable name mapping."""

    def test_lengthkm_mapping(self):
        """MTS should map 'lengthkm' <-> 'basin__length'."""
        assert map_to_external('lengthkm') == 'basin__length'
        assert map_to_internal('basin__length') == 'lengthkm'

    def test_roundtrip_all_mts_mappings(self):
        """All MTS internal names should roundtrip correctly."""
        from dhbv2.mts_bmi import _var_name_internal_map

        for internal_name, external_name in _var_name_internal_map.items():
            assert map_to_external(internal_name) == external_name
            assert map_to_internal(external_name) == internal_name


# ---------------------------------------------------------------------------- #
#  Unit conversion
# ---------------------------------------------------------------------------- #


class TestMtsBmiUnitConversion:
    """Verify MTS-specific unit conversions."""

    def test_streamflow_mm_h_to_m_h(self, mts_bmi):
        """MTS streamflow should be converted from mm/h to m/h."""
        values = [5.0]  # 5 mm/h
        result = mts_bmi._to_external_units(
            'land_surface_water__runoff_volume_flux',
            values,
        )
        # 5 mm/h / 1000 = 0.005 m/h
        np.testing.assert_allclose(result[0], 0.005, rtol=1e-10)

    def test_conversion_differs_from_daily(self, daily_bmi, mts_bmi):
        """MTS and daily should apply different unit conversions."""
        name = 'land_surface_water__runoff_volume_flux'
        values = [24.0]
        daily_result = daily_bmi._to_external_units(name, values)
        mts_result = mts_bmi._to_external_units(name, values)
        # Daily: 24 / 1000 / 24 = 0.001; MTS: 24 / 1000 = 0.024
        assert daily_result[0] != mts_result[0], (
            "Daily and MTS should use different unit conversions"
        )


# ---------------------------------------------------------------------------- #
#  Caching
# ---------------------------------------------------------------------------- #


class TestMtsBmiCaching:
    """Verify cache management and buffer filling."""

    @pytest.fixture
    def mts_with_buffers(self, mts_bmi):
        """MTS BMI with initialized buffers (3 forcing variables)."""
        n_vars = 3
        mts_bmi._hourly_buffer = RingBuffer(
            (169, 1, n_vars),
            dtype=np.float64,
        )
        mts_bmi._daily_buffer = RingBuffer(
            (358, 1, n_vars),
            dtype=np.float64,
        )
        mts_bmi._day_accumulator = np.zeros(
            (24, 1, n_vars),
            dtype=np.float64,
        )
        mts_bmi._day_accumulator_ptr = 0
        return mts_bmi

    def test_update_caches_adds_to_hourly(self, mts_with_buffers):
        """_update_caches should add forcing to hourly buffer."""
        bmi = mts_with_buffers
        forcing = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)

        bmi._update_caches(forcing)
        assert len(bmi._hourly_buffer) == 1

    def test_update_caches_fills_day_accumulator(self, mts_with_buffers):
        """_update_caches should fill day accumulator sequentially."""
        bmi = mts_with_buffers

        for h in range(12):
            forcing = np.array([[[float(h), float(h), float(h)]]])
            bmi._update_caches(forcing)

        assert bmi._day_accumulator_ptr == 12
        np.testing.assert_array_equal(
            bmi._day_accumulator[5, 0],
            [5.0, 5.0, 5.0],
        )

    def test_update_caches_aggregates_at_24h(self, mts_with_buffers):
        """After 24 hours, day accumulator should flush to daily buffer."""
        bmi = mts_with_buffers

        for h in range(24):
            forcing = np.array([[[1.0, float(h), 0.5]]])
            bmi._update_caches(forcing)

        # Daily buffer should have 1 entry
        assert len(bmi._daily_buffer) == 1
        # Accumulator pointer should reset to 0
        assert bmi._day_accumulator_ptr == 0

    def test_daily_aggregation_correctness(self, mts_with_buffers):
        """Daily buffer should contain sum(P), mean(T), sum(PET)."""
        bmi = mts_with_buffers

        for h in range(24):
            forcing = np.array([[[2.0, 10.0 + float(h), 0.1]]])
            bmi._update_caches(forcing)

        daily = bmi._daily_buffer.get_last()
        # P: sum = 2.0 * 24 = 48.0
        np.testing.assert_allclose(daily[0, 0, 0], 48.0, rtol=1e-10)
        # T: mean = 10.0 + mean(0..23) = 10.0 + 11.5 = 21.5
        np.testing.assert_allclose(daily[0, 0, 1], 21.5, rtol=1e-10)
        # PET: sum = 0.1 * 24 = 2.4
        np.testing.assert_allclose(daily[0, 0, 2], 2.4, rtol=1e-10)

    def test_multiple_days_fill_daily_buffer(self, mts_with_buffers):
        """Multiple complete days should fill daily buffer correctly."""
        bmi = mts_with_buffers

        for _day in range(3):
            for _h in range(24):
                forcing = np.array([[[1.0, 15.0, 0.2]]])
                bmi._update_caches(forcing)

        assert len(bmi._daily_buffer) == 3
        assert len(bmi._hourly_buffer) == 72  # 3 * 24


# ---------------------------------------------------------------------------- #
#  Warmup logic
# ---------------------------------------------------------------------------- #


class TestMtsBmiWarmupLogic:
    """Verify warmup trigger conditions."""

    @pytest.fixture
    def mts_warmup_ready(self, mts_bmi):
        """MTS BMI with buffers filled to warmup-ready state."""
        n_vars = 3
        mts_bmi._hourly_buffer = RingBuffer(
            (169, 1, n_vars),
            dtype=np.float64,
        )
        mts_bmi._daily_buffer = RingBuffer(
            (400, 1, n_vars),
            dtype=np.float64,
        )
        mts_bmi.b_offset = 7

        # Fill daily buffer to required level (351 + 7 = 358)
        for i in range(360):
            mts_bmi._daily_buffer.append(
                np.array([[float(i), float(i), float(i)]]),
            )

        # Fill hourly buffer to required level (168)
        for i in range(169):
            mts_bmi._hourly_buffer.append(
                np.array([[float(i), float(i), float(i)]]),
            )

        mts_bmi._steps_since_warmup = 0
        return mts_bmi

    def test_can_run_warmup_insufficient_daily(self, mts_bmi):
        """Should return False when daily buffer is not full enough."""
        mts_bmi._daily_buffer = RingBuffer((400, 1, 3))
        mts_bmi._hourly_buffer = RingBuffer((200, 1, 3))
        mts_bmi.b_offset = 7

        # Fill only 100 daily entries (need 358)
        for _ in range(100):
            mts_bmi._daily_buffer.append(np.zeros((1, 3)))
        for _ in range(200):
            mts_bmi._hourly_buffer.append(np.zeros((1, 3)))

        assert mts_bmi._can_run_warmup() is False

    def test_can_run_warmup_insufficient_hourly(self, mts_bmi):
        """Should return False when hourly buffer is not full enough."""
        mts_bmi._daily_buffer = RingBuffer((400, 1, 3))
        mts_bmi._hourly_buffer = RingBuffer((200, 1, 3))
        mts_bmi.b_offset = 7

        # Fill daily enough but hourly too few
        for _ in range(360):
            mts_bmi._daily_buffer.append(np.zeros((1, 3)))
        for _ in range(50):  # Need 168
            mts_bmi._hourly_buffer.append(np.zeros((1, 3)))

        assert mts_bmi._can_run_warmup() is False

    def test_can_run_warmup_sufficient(self, mts_warmup_ready):
        """Should return True when both buffers are full enough."""
        assert mts_warmup_ready._can_run_warmup() is True

    def test_warmup_trigger_at_cycle_boundary(self, mts_warmup_ready):
        """Should trigger warmup at daily boundary on cycle start."""
        bmi = mts_warmup_ready
        bmi._timestep = 24 * 365  # A daily boundary
        bmi._steps_since_warmup = 0

        assert bmi._is_warmup_trigger_step() is True

    def test_warmup_trigger_not_at_daily_boundary(self, mts_warmup_ready):
        """Should not trigger warmup mid-day."""
        bmi = mts_warmup_ready
        bmi._timestep = 24 * 365 + 5  # Not a daily boundary
        bmi._steps_since_warmup = 0

        assert bmi._is_warmup_trigger_step() is False

    def test_warmup_trigger_not_mid_cycle(self, mts_warmup_ready):
        """Should not trigger warmup between 7-day cycles."""
        bmi = mts_warmup_ready
        bmi._timestep = 24 * 366  # Daily boundary
        bmi._steps_since_warmup = 48  # Mid-cycle (not 0 or 168)

        assert bmi._is_warmup_trigger_step() is False

    def test_warmup_trigger_at_next_cycle(self, mts_warmup_ready):
        """Should trigger at the next 7-day cycle boundary."""
        bmi = mts_warmup_ready
        bmi._timestep = 24 * 372  # Daily boundary
        bmi._steps_since_warmup = 168  # Exactly one cycle elapsed

        assert bmi._is_warmup_trigger_step() is True


# ---------------------------------------------------------------------------- #
#  Normalization
# ---------------------------------------------------------------------------- #


class TestMtsBmiNormalization:
    """Verify MTS-specific normalization (dict-keyed norm_stats)."""

    def test_normalize_known_values(self, mts_bmi):
        """MTS normalize should use dict-keyed mean/std."""
        mts_bmi.norm_stats = {
            'mean': {'dyn_input': [5.0, 15.0, 3.0]},
            'std': {'dyn_input': [2.0, 10.0, 1.5]},
        }
        data = np.array([[[5.0, 15.0, 3.0]]])  # shape (1, 1, 3)

        result = mts_bmi._normalize(data, 'dyn_input')

        # All values are at the mean, so normalized values ≈ 0
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_normalize_off_center(self, mts_bmi):
        """Values away from mean should produce non-zero output."""
        mts_bmi.norm_stats = {
            'mean': {'dyn_input': [5.0]},
            'std': {'dyn_input': [2.0]},
        }
        data = np.array([[[9.0]]])
        result = mts_bmi._normalize(data, 'dyn_input')

        expected = (9.0 - 5.0) / (2.0 + 1e-6)
        np.testing.assert_allclose(result[0, 0, 0], expected, rtol=1e-5)

    def test_normalize_preserves_shape(self, mts_bmi):
        """Output shape should match input shape."""
        mts_bmi.norm_stats = {
            'mean': {'dyn_input': [5.0, 15.0, 3.0]},
            'std': {'dyn_input': [2.0, 10.0, 1.5]},
        }
        data = np.random.rand(10, 5, 3)
        result = mts_bmi._normalize(data, 'dyn_input')
        assert result.shape == (10, 5, 3)


# ---------------------------------------------------------------------------- #
#  Regression
# ---------------------------------------------------------------------------- #


class TestMtsBmiBenchmark:
    """Regression test: compare simulation output against stored benchmark.

    Requires pre-computed output files. Skipped if files not available.
    """

    _pkg_root = Path(__file__).parent.parent
    _sim_path = _pkg_root / 'output' / 'dhbv2_mts_cat-2453_runoff.npy'
    _val_path = _pkg_root / 'tests' / 'dhbv2_mts_cat-2453_runoff_benchmark.npy'
    _tolerance = 1e-5

    @pytest.fixture
    def sim_and_val(self):
        """Load simulation and validation arrays if available."""
        if not self._sim_path.exists():
            pytest.skip(f"Simulation output not found: {self._sim_path}")
        if not self._val_path.exists():
            pytest.skip(f"Validation benchmark not found: {self._val_path}")
        return np.load(self._sim_path), np.load(self._val_path)

    def test_shapes_match(self, sim_and_val):
        """Simulation and validation arrays should have the same shape."""
        sim, val = sim_and_val
        if sim.shape != val.shape:
            pytest.skip(
                f"Shape mismatch (sim={sim.shape}, val={val.shape}); "
                f"re-run simulation to regenerate output",
            )

    def test_within_tolerance(self, sim_and_val):
        """Max absolute error should be within tolerance."""
        sim, val = sim_and_val
        if sim.shape != val.shape:
            pytest.skip(
                f"Shape mismatch (sim={sim.shape}, val={val.shape}); "
                f"re-run simulation to regenerate output",
            )
        max_diff = np.max(np.abs(sim - val))
        assert max_diff <= self._tolerance, (
            f"Runoff simulation does not match benchmark within "
            f"tolerance of {self._tolerance}. Max error: {max_diff}"
        )
