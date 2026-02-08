"""
Tests for DeltaModelBmi (daily BMI) interface and internals.

Coverage:
- Constructor defaults and initial state
- BMI info methods (component name, var names, counts)
- Time and grid query methods
- Variable name mapping (internal <-> CSDMS standard names)
- Unit conversion (mm/d -> m/h)
- Get/set value operations on variable dicts
- Helper methods (_bmi_array, _bmi_tensor, _set_dtype)
- Normalization (_normalize)
- Daily aggregation (_aggregate_to_daily)

NOTE: Tests use un-initialized BMI instances where possible. Methods that
require model weights or config files are not tested here.
"""

import datetime

import numpy as np
import pytest
import torch

from dhbv2.bmi import DeltaModelBmi, map_to_external, map_to_internal
from dhbv2.pet import hargreaves_pet


# ---------------------------------------------------------------------------- #
#  Constructor
# ---------------------------------------------------------------------------- #


class TestDailyBmiDefaults:
    """Verify DeltaModelBmi constructor sets correct defaults."""

    def test_name(self, daily_bmi):
        """Model name should be set."""
        assert daily_bmi._name == 'δHBV2.0 Daily'

    def test_not_initialized(self, daily_bmi):
        """Model should not be initialized after construction."""
        assert daily_bmi._initialized is False
        assert daily_bmi._is_warm is False
        assert daily_bmi._model is None

    def test_default_time_settings(self, daily_bmi):
        """Time defaults should be 1-hour steps in seconds."""
        assert daily_bmi._time_step_size == 3600
        assert daily_bmi._time_units == 's'
        assert daily_bmi._start_time == 0.0
        assert daily_bmi._timestep == 0

    def test_default_dtype(self, daily_bmi):
        """Default dtype should be float64."""
        assert daily_bmi._dtype == 'float64'
        assert daily_bmi.np_dtype == np.float64
        assert daily_bmi.pt_dtype == torch.float64

    def test_default_device(self, daily_bmi):
        """Default device should be CPU."""
        assert daily_bmi.device == torch.device('cpu')

    def test_default_warmup_period(self, daily_bmi):
        """Default warmup should require 365 days of history."""
        assert daily_bmi.req_daily_history == 365

    def test_variable_dicts_populated(self, daily_bmi):
        """Variable dicts should be populated with correct structure."""
        assert len(daily_bmi._dynamic_var) == 8
        assert len(daily_bmi._output_vars) == 1
        assert len(daily_bmi._static_var) > 0

        # Each variable should have 'value' and 'units' keys
        for var_dict in [daily_bmi._dynamic_var, daily_bmi._output_vars]:
            for name, entry in var_dict.items():
                assert 'value' in entry, f"Missing 'value' key for {name}"
                assert 'units' in entry, f"Missing 'units' key for {name}"

    def test_pet_method_default(self, daily_bmi):
        """Default PET method should be penman_monteith."""
        assert daily_bmi._pet_method == 'penman_monteith'


# ---------------------------------------------------------------------------- #
#  BMI info methods
# ---------------------------------------------------------------------------- #


class TestDailyBmiInfo:
    """Verify BMI standard info query methods."""

    def test_get_component_name(self, daily_bmi):
        """Component name should be a non-empty string."""
        name = daily_bmi.get_component_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_input_item_count(self, daily_bmi):
        """Input item count should match dynamic variable count (8)."""
        count = daily_bmi.get_input_item_count()
        assert count == 8

    def test_get_output_item_count(self, daily_bmi):
        """Output item count should be 1 (streamflow only)."""
        count = daily_bmi.get_output_item_count()
        assert count == 1

    def test_get_input_var_names(self, daily_bmi):
        """Input var names should be CSDMS standard names."""
        names = daily_bmi.get_input_var_names()
        assert isinstance(names, tuple)
        assert len(names) == 8
        # Spot check: precipitation should be in inputs
        assert any('precipitation' in n for n in names), (
            f"No precipitation variable found in {names}"
        )

    def test_get_output_var_names(self, daily_bmi):
        """Output var names should include runoff."""
        names = daily_bmi.get_output_var_names()
        assert isinstance(names, tuple)
        assert len(names) == 1
        assert 'runoff' in names[0]

    def test_get_var_units(self, daily_bmi):
        """Variable units should be retrievable strings."""
        input_names = daily_bmi.get_input_var_names()
        units = daily_bmi.get_var_units(input_names[0])
        assert isinstance(units, str)
        assert len(units) > 0

    def test_get_var_type(self, daily_bmi):
        """Variable type should be a valid numpy dtype name."""
        name = daily_bmi.get_output_var_names()[0]
        vtype = daily_bmi.get_var_type(name)
        assert vtype == 'float64'

    def test_get_var_itemsize(self, daily_bmi):
        """Item size should be positive (8 bytes for float64)."""
        name = daily_bmi.get_output_var_names()[0]
        itemsize = daily_bmi.get_var_itemsize(name)
        assert itemsize == 8

    def test_get_var_nbytes(self, daily_bmi):
        """Total bytes should be itemsize * array length."""
        name = daily_bmi.get_output_var_names()[0]
        nbytes = daily_bmi.get_var_nbytes(name)
        assert nbytes > 0

    def test_get_var_grid(self, daily_bmi):
        """Grid ID should be 0 for all variables."""
        for name in daily_bmi.get_input_var_names():
            assert daily_bmi.get_var_grid(name) == 0

    def test_get_var_grid_invalid_name(self, daily_bmi):
        """Invalid variable name should raise KeyError."""
        with pytest.raises(KeyError):
            daily_bmi.get_var_grid('nonexistent_variable')

    def test_get_var_location(self, daily_bmi):
        """Variable location should be 'node' for all variables."""
        for name in daily_bmi.get_output_var_names():
            assert daily_bmi.get_var_location(name) == 'node'

    def test_get_var_location_invalid_name(self, daily_bmi):
        """Invalid variable name should raise KeyError."""
        with pytest.raises(KeyError):
            daily_bmi.get_var_location('nonexistent_variable')


# ---------------------------------------------------------------------------- #
#  Time and grid
# ---------------------------------------------------------------------------- #


class TestDailyBmiTimeGrid:
    """Verify BMI time and grid query methods."""

    def test_get_start_time(self, daily_bmi):
        """Start time should be 0.0."""
        assert daily_bmi.get_start_time() == 0.0

    def test_get_end_time(self, daily_bmi):
        """End time should be float max."""
        assert daily_bmi.get_end_time() == np.finfo('d').max

    def test_get_current_time_at_start(self, daily_bmi):
        """Current time should equal start time at step 0."""
        assert daily_bmi.get_current_time() == daily_bmi.get_start_time()

    def test_get_current_time_advances(self, daily_bmi):
        """Current time should advance by time_step_size per timestep."""
        daily_bmi._timestep = 10
        expected = 10 * 3600.0
        assert daily_bmi.get_current_time() == expected

    def test_get_time_step(self, daily_bmi):
        """Time step should be 3600 seconds (1 hour)."""
        assert daily_bmi.get_time_step() == 3600.0

    def test_get_time_units(self, daily_bmi):
        """Time units should be seconds."""
        assert daily_bmi.get_time_units() == 's'

    def test_get_grid_rank(self, daily_bmi):
        """Grid rank should be 1 (node-based)."""
        assert daily_bmi.get_grid_rank(0) == 1

    def test_get_grid_rank_invalid(self, daily_bmi):
        """Invalid grid ID should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            daily_bmi.get_grid_rank(1)

    def test_get_grid_size(self, daily_bmi):
        """Grid size should be 1 (single catchment)."""
        assert daily_bmi.get_grid_size(0) == 1

    def test_get_grid_type(self, daily_bmi):
        """Grid type should be 'scalar'."""
        assert daily_bmi.get_grid_type(0) == 'scalar'

    def test_unimplemented_grid_methods(self, daily_bmi):
        """Unimplemented grid methods should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            daily_bmi.get_grid_shape(0, None)
        with pytest.raises(NotImplementedError):
            daily_bmi.get_grid_spacing(0, None)
        with pytest.raises(NotImplementedError):
            daily_bmi.get_grid_origin(0, None)


# ---------------------------------------------------------------------------- #
#  Variable name mapping
# ---------------------------------------------------------------------------- #


class TestVariableMapping:
    """Verify internal <-> external variable name mapping."""

    def test_map_to_external_known_key(self):
        """Known internal name should map to CSDMS standard name."""
        ext = map_to_external('P')
        assert ext == 'atmosphere_water__liquid_equivalent_precipitation_rate'

    def test_map_to_internal_known_key(self):
        """Known external name should map back to internal name."""
        internal = map_to_internal(
            'atmosphere_water__liquid_equivalent_precipitation_rate',
        )
        assert internal == 'P'

    def test_roundtrip_all_mappings(self):
        """Every internal name should roundtrip through external and back."""
        from dhbv2.bmi import _var_name_internal_map

        for internal_name, external_name in _var_name_internal_map.items():
            assert map_to_external(internal_name) == external_name
            assert map_to_internal(external_name) == internal_name

    def test_unknown_internal_name_raises(self):
        """Unknown internal name should raise KeyError."""
        with pytest.raises(KeyError):
            map_to_external('NONEXISTENT')

    def test_unknown_external_name_raises(self):
        """Unknown external name should raise KeyError."""
        with pytest.raises(KeyError):
            map_to_internal('nonexistent__variable')


# ---------------------------------------------------------------------------- #
#  Unit conversion
# ---------------------------------------------------------------------------- #


class TestDailyBmiUnitConversion:
    """Verify internal -> external unit conversions."""

    def test_streamflow_mm_d_to_m_h(self, daily_bmi):
        """Streamflow should be converted from mm/d to m/h."""
        values = [24.0]  # 24 mm/d
        result = daily_bmi._to_external_units(
            'land_surface_water__runoff_volume_flux',
            values,
        )
        # 24 mm/d / 1000 / 24 = 0.001 m/h
        np.testing.assert_allclose(result[0], 0.001, rtol=1e-10)

    def test_non_streamflow_passthrough(self, daily_bmi):
        """Non-streamflow variables should pass through unchanged."""
        values = [5.0, 10.0]
        result = daily_bmi._to_external_units('some_other_variable', values)
        assert result == values


# ---------------------------------------------------------------------------- #
#  Get/Set values
# ---------------------------------------------------------------------------- #


class TestDailyBmiValues:
    """Verify set_value, get_value, and get_value_ptr operations."""

    def test_set_value_and_get_value_ptr(self, daily_bmi):
        """set_value should store data retrievable via get_value_ptr."""
        name = daily_bmi.get_input_var_names()[0]
        daily_bmi.set_value(name, np.array([42.0]))

        ptr = daily_bmi.get_value_ptr(name)
        assert 42.0 in ptr

    def test_set_value_scalar(self, daily_bmi):
        """set_value should accept scalar (non-array) input."""
        name = daily_bmi.get_input_var_names()[0]
        daily_bmi.set_value(name, 99.0)
        ptr = daily_bmi.get_value_ptr(name)
        assert 99.0 in ptr

    def test_get_value_copies_to_dest(self, daily_bmi):
        """get_value should copy values into the dest array."""
        name = daily_bmi.get_output_var_names()[0]
        dest = np.zeros(1)
        result = daily_bmi.get_value(name, dest)
        assert result is dest
        assert isinstance(result[0], (float, np.floating))

    def test_get_value_at_indices(self, daily_bmi):
        """get_value_at_indices should return values at specified indices."""
        name = daily_bmi.get_output_var_names()[0]
        dest = np.zeros(1)
        result = daily_bmi.get_value_at_indices(name, dest, [0])
        assert result is dest

    def test_set_value_at_indices(self, daily_bmi):
        """set_value_at_indices should update specific indices."""
        name = daily_bmi.get_input_var_names()[0]
        daily_bmi.set_value(name, np.array([0.0]))
        daily_bmi.set_value_at_indices(name, [0], [77.0])
        ptr = daily_bmi.get_value_ptr(name)
        assert ptr.flat[0] == 77.0

    def test_get_value_ptr_returns_reference(self, daily_bmi):
        """get_value_ptr should return a reference, not a copy."""
        name = daily_bmi.get_input_var_names()[0]
        daily_bmi.set_value(name, np.array([1.0]))

        ptr1 = daily_bmi.get_value_ptr(name)
        ptr2 = daily_bmi.get_value_ptr(name)
        assert ptr1 is ptr2


# ---------------------------------------------------------------------------- #
#  Helper methods
# ---------------------------------------------------------------------------- #


class TestDailyBmiHelpers:
    """Verify internal helper methods."""

    def test_bmi_array_dtype(self, daily_bmi):
        """_bmi_array should create array with configured dtype."""
        arr = daily_bmi._bmi_array([1.0, 2.0, 3.0])
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_bmi_tensor_dtype_and_device(self, daily_bmi):
        """_bmi_tensor should create tensor with configured dtype and device."""
        tensor = daily_bmi._bmi_tensor([1.0, 2.0])
        assert tensor.dtype == torch.float64
        assert tensor.device == torch.device('cpu')

    def test_bmi_tensor_from_ndarray(self, daily_bmi):
        """_bmi_tensor should accept numpy arrays."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        tensor = daily_bmi._bmi_tensor(arr)
        assert tensor.dtype == torch.float64  # should cast to configured dtype
        assert tensor.shape == (2,)

    def test_set_dtype_valid(self, daily_bmi):
        """_set_dtype should update both numpy and torch dtypes."""
        daily_bmi._dtype = 'float32'
        daily_bmi._set_dtype()
        assert daily_bmi.np_dtype == np.float32
        assert daily_bmi.pt_dtype == torch.float32

    def test_set_dtype_invalid(self, daily_bmi):
        """Invalid dtype should raise ValueError."""
        daily_bmi._dtype = 'invalid_type'
        with pytest.raises(ValueError, match="Could not parse dtype"):
            daily_bmi._set_dtype()

    def test_set_value_internal(self, daily_bmi):
        """_set_value_internal should create correct dict structure."""
        vars_list = [('var_a', 'mm'), ('var_b', 'degC')]
        value = np.array([0.0])
        result = DeltaModelBmi._set_value_internal(vars_list, value)

        assert 'var_a' in result
        assert 'var_b' in result
        assert result['var_a']['units'] == 'mm'
        assert result['var_b']['units'] == 'degC'
        np.testing.assert_array_equal(result['var_a']['value'], [0.0])

    def test_set_value_internal_values_are_copies(self, daily_bmi):
        """Each variable should get its own copy of the value array."""
        vars_list = [('a', '-'), ('b', '-')]
        value = np.array([0.0])
        result = DeltaModelBmi._set_value_internal(vars_list, value)

        result['a']['value'][0] = 999.0
        assert result['b']['value'][0] != 999.0, (
            "Variables should have independent value copies"
        )

    def test_set_empty_outputs(self, daily_bmi):
        """_set_empty_outputs should zero out all output variables."""
        # First set a non-zero value
        name = daily_bmi.get_output_var_names()[0]
        daily_bmi._output_vars[name]['value'] = np.array([42.0])

        daily_bmi._set_empty_outputs()
        ptr = daily_bmi.get_value_ptr(name)
        np.testing.assert_array_equal(ptr, 0.0)


# ---------------------------------------------------------------------------- #
#  Normalization
# ---------------------------------------------------------------------------- #


class TestDailyBmiNormalization:
    """Verify Gaussian normalization of model inputs."""

    def test_normalize_known_values(self, daily_bmi):
        """Normalization should apply (X - mean) / (std + eps)."""
        daily_bmi.norm_stats = {
            'P': [0, 10, 5.0, 2.0],  # [min, max, mean, std]
            'T': [-5, 35, 15.0, 10.0],
        }
        data = np.array([[[5.0, 15.0]]])  # shape (1, 1, 2)

        result = daily_bmi._normalize(data, ['P', 'T'])

        # (5 - 5) / (2 + 1e-6) ≈ 0.0
        # (15 - 15) / (10 + 1e-6) ≈ 0.0
        np.testing.assert_allclose(result[0, 0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 1], 0.0, atol=1e-5)

    def test_normalize_off_center(self, daily_bmi):
        """Values away from mean should produce non-zero normalized output."""
        daily_bmi.norm_stats = {
            'P': [0, 10, 5.0, 2.0],
        }
        data = np.array([[[9.0]]])  # shape (1, 1, 1)
        result = daily_bmi._normalize(data, ['P'])

        expected = (9.0 - 5.0) / (2.0 + 1e-6)
        np.testing.assert_allclose(result[0, 0, 0], expected, rtol=1e-5)

    def test_normalize_preserves_shape(self, daily_bmi):
        """Output shape should match input shape."""
        daily_bmi.norm_stats = {
            'P': [0, 10, 5.0, 2.0],
            'T': [-5, 35, 15.0, 10.0],
            'PET': [0, 8, 3.0, 1.5],
        }
        data = np.random.rand(10, 5, 3)
        result = daily_bmi._normalize(data, ['P', 'T', 'PET'])
        assert result.shape == (10, 5, 3)


# ---------------------------------------------------------------------------- #
#  Daily aggregation
# ---------------------------------------------------------------------------- #


class TestDailyBmiAggregation:
    """Verify hourly-to-daily aggregation logic."""

    def test_aggregate_penman_monteith(self, daily_bmi_with_accumulator):
        """Aggregation should sum P, mean T, sum PET for Penman-Monteith."""
        bmi = daily_bmi_with_accumulator
        result = bmi._aggregate_to_daily()

        assert result.shape == (1, 1, 3), (
            f"Expected shape (1, 1, 3), got {result.shape}"
        )

        # P should be sum of hourly values
        expected_p = bmi._day_accumulator[:, :, 0].sum(axis=0)
        np.testing.assert_allclose(result[0, 0, 0], expected_p, rtol=1e-10)

        # T should be mean of hourly values
        expected_t = bmi._day_accumulator[:, :, 1].mean(axis=0)
        np.testing.assert_allclose(result[0, 0, 1], expected_t, rtol=1e-10)

        # PET should be sum of hourly values (Penman-Monteith)
        expected_pet = bmi._day_accumulator[:, :, 2].sum(axis=0)
        np.testing.assert_allclose(result[0, 0, 2], expected_pet, rtol=1e-10)

    def test_aggregate_hargreaves(self, daily_bmi):
        """Aggregation should compute Hargreaves PET from daily T range."""
        bmi = daily_bmi
        n_vars = 3
        bmi._day_accumulator = np.zeros((24, 1, n_vars), dtype=np.float64)
        bmi._pet_method = 'hargreaves'
        bmi._latitude_rad = np.deg2rad(40.0)
        bmi._start_date = datetime.datetime(2020, 6, 21)
        bmi._day_count = 0
        bmi.np_dtype = np.float64

        # Fill temperature with diurnal cycle: min=10, max=30
        for h in range(24):
            bmi._day_accumulator[h, 0, 0] = 1.0  # P: 1 mm/hr
            bmi._day_accumulator[h, 0, 1] = 20.0 + 10.0 * np.sin(
                2 * np.pi * h / 24,
            )
            bmi._day_accumulator[h, 0, 2] = 0.0  # PET: unused for hargreaves

        result = bmi._aggregate_to_daily()

        # P should be sum = 24.0
        np.testing.assert_allclose(result[0, 0, 0], 24.0, rtol=1e-10)

        # T should be mean ≈ 20.0
        expected_t = bmi._day_accumulator[:, 0, 1].mean()
        np.testing.assert_allclose(result[0, 0, 1], expected_t, rtol=1e-10)

        # PET should match hargreaves_pet computation
        temps = bmi._day_accumulator[:, 0, 1]
        expected_pet = hargreaves_pet(
            tmin=np.array([temps.min()]),
            tmax=np.array([temps.max()]),
            tmean=np.array([temps.mean()]),
            lat=bmi._latitude_rad,
            day_of_year=np.array(
                [bmi._start_date.timetuple().tm_yday],
                dtype=np.float64,
            ),
        )
        np.testing.assert_allclose(
            result[0, 0, 2],
            expected_pet[0],
            rtol=1e-6,
            err_msg="Hargreaves PET in aggregation does not match direct computation",
        )
