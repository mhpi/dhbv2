"""Pytest unit tests for the BMI interface.

Authors: Jessica Garrett, Jonathan Frame | Leo Lonzarich (Adapted)
"""

from pathlib import Path

import pytest
import numpy as np

from dhbv2.bmi import DeltaModelBmi as bmi


# --- Fixtures ---


@pytest.fixture(scope="module")
def cfg_file():
    """Path to the BMI config file."""
    config_path = (
        Path(__file__).parent.parent.parent
        / "bmi_config_files"
        / "bmi_config_cat-88306_5yr.yaml"
    )
    if not config_path.exists():
        pytest.skip(f"Configuration file not found: {config_path}")
    return str(config_path)


@pytest.fixture(scope="module")
def bmi_model(cfg_file):
    """Initialized BMI model instance."""
    model = bmi.DeltaModelBmi(cfg_file, verbose=True)
    model.stepwise = True
    model.initialize()
    yield model
    # Finalize after all tests
    try:
        model.finalize()
    except RuntimeError:
        pass  # Ignore errors during finalize in teardown


# --- Helper Fixtures ---


@pytest.fixture
def all_var_names(bmi_model):
    return bmi_model.get_output_var_names() + bmi_model.get_input_var_names()


# --- Tests: Model Information ---


def test_initialize(bmi_model):
    # initialize() is called in fixture; just verify it worked
    assert bmi_model is not None


def test_get_component_name(bmi_model):
    name = bmi_model.get_component_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_input_item_count(bmi_model):
    count = bmi_model.get_input_item_count()
    assert isinstance(count, int)
    assert count >= 0


def test_get_output_item_count(bmi_model):
    count = bmi_model.get_output_item_count()
    assert isinstance(count, int)
    assert count >= 0


def test_get_input_var_names(bmi_model):
    names = bmi_model.get_input_var_names()
    assert isinstance(names, list)
    for name in names:
        assert isinstance(name, str)


def test_get_output_var_names(bmi_model):
    names = bmi_model.get_output_var_names()
    assert isinstance(names, list)
    for name in names:
        assert isinstance(name, str)


# --- Tests: Variable Information (parametrized over all variables) ---


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_var_units(bmi_model, var_name):
    units = bmi_model.get_var_units(var_name)
    assert isinstance(units, str)


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_var_itemsize(bmi_model, var_name):
    itemsize = bmi_model.get_var_itemsize(var_name)
    assert isinstance(itemsize, int)
    assert itemsize > 0


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_var_type(bmi_model, var_name):
    vtype = bmi_model.get_var_type(var_name)
    assert isinstance(vtype, str)


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_var_nbytes(bmi_model, var_name):
    nbytes = bmi_model.get_var_nbytes(var_name)
    assert isinstance(nbytes, int)
    assert nbytes >= 0


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_var_grid(bmi_model, var_name):
    grid_id = bmi_model.get_var_grid(var_name)
    assert isinstance(grid_id, int)
    assert grid_id >= 0


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_var_location(bmi_model, var_name):
    location = bmi_model.get_var_location(var_name)
    assert isinstance(location, str)


# --- Tests: Time Functions ---


def test_get_start_time(bmi_model):
    start = bmi_model.get_start_time()
    assert isinstance(start, (int, float))


def test_get_end_time(bmi_model):
    end = bmi_model.get_end_time()
    assert isinstance(end, (int, float))
    assert end >= bmi_model.get_start_time()


def test_get_current_time(bmi_model):
    current = bmi_model.get_current_time()
    assert isinstance(current, (int, float))
    assert current >= bmi_model.get_start_time()


def test_get_time_step(bmi_model):
    dt = bmi_model.get_time_step()
    assert isinstance(dt, (int, float))
    assert dt > 0


def test_get_time_units(bmi_model):
    units = bmi_model.get_time_units()
    assert isinstance(units, str)


# --- Tests: Grid Functions ---


def test_grid_functions(bmi_model):
    grid_id = 0  # assumed single grid
    rank = bmi_model.get_grid_rank(grid_id)
    size = bmi_model.get_grid_size(grid_id)
    grid_type = bmi_model.get_grid_type(grid_id)

    assert isinstance(rank, int) and rank >= 0
    assert isinstance(size, int) and size >= 0
    assert isinstance(grid_type, str)


# --- Tests: Get/Set Values (parametrized) ---


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_set_get_value(bmi_model, var_name):
    original = np.zeros(1)
    bmi_model.get_value(var_name, original)

    test_val = -99.0
    bmi_model.set_value(var_name, np.array([test_val]))

    retrieved = np.zeros(1)
    bmi_model.get_value(var_name, retrieved)

    # Note: exact match may fail for floats; consider tolerance if needed
    assert retrieved[0] == test_val


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_get_value_ptr(bmi_model, var_name):
    ptr = bmi_model.get_value_ptr(var_name)
    assert isinstance(ptr, np.ndarray)
    assert ptr.size > 0


@pytest.mark.parametrize("var_name", lambda: pytest.lazy_fixture("all_var_names"))
def test_set_get_value_at_indices(bmi_model, var_name):
    test_val = -11.0
    indices = np.array([0])
    values = np.array([test_val])

    bmi_model.set_value_at_indices(var_name, indices, values)

    dest = np.zeros(1)
    bmi_model.get_value_at_indices(var_name, dest, indices)

    assert dest[0] == test_val


# --- Tests: Control Functions ---


def test_update(bmi_model):
    initial_time = bmi_model.get_current_time()
    bmi_model.update()
    new_time = bmi_model.get_current_time()
    assert new_time > initial_time


def test_update_until(bmi_model):
    current = bmi_model.get_current_time()
    target = current + 10 * bmi_model.get_time_step()
    bmi_model.update_until(target)
    final = bmi_model.get_current_time()
    assert final >= target - 1e-6  # allow small floating error
