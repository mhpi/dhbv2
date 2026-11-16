"""Tests for MtsDeltaModelBmi checkpoint/restore (ngen serialization protocol).

Coverage:
- Reserved variable introspection (units, types, item sizes)
- Reserved names hidden from the input/output lists
- Save -> restore -> continue reproduces an uninterrupted run, per warmup mode
- Restore without forcing history keeps stepping, defers warmups until the
  buffers refill, and rejoins the uninterrupted run after a full re-spin
- Restore rejects a foreign payload, and rolls back a failed write
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import xarray as xr
import yaml

from dhbv2.mts_bmi import MtsDeltaModelBmi

PKG_ROOT = Path(__file__).parent.parent
DATA_DIR = PKG_ROOT / 'ngen_resources' / 'data'
BMI_CONFIG = DATA_DIR / 'dhbv_2_mts' / 'config' / 'bmi_cat-2453.yaml'
MODEL_DIR = DATA_DIR / 'dhbv_2_mts' / 'model' / 'dhbv_2_mts'
FORCING = (
    DATA_DIR / 'forcing' / 'camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc'
)

CREATE = 'ngen::serialization_create'
FREE = 'ngen::serialization_free'
SIZE = 'ngen::serialization_size'
STATE = 'ngen::serialization_state'
RUNOFF = 'land_surface_water__runoff_volume_flux'

# Short windows: first warmup on day 7 (5-day daily + 2-day hourly window),
# then one every 2 days.
N_STEPS = 24 * 12
SAVE_STEP = 24 * 7 + 5

# Without forcing history, the warmup cycle restarts at the day boundary after
# LEAN_SAVE_STEP (step 216), which lines up with the uninterrupted cycle.
LEAN_SAVE_STEP = 24 * 8 + 5
LEAN_N_STEPS = 24 * 17

pytestmark = [
    pytest.mark.skipif(
        not (FORCING.exists() and MODEL_DIR.exists()),
        reason="Shipped cat-2453 forcing or model weights not found.",
    ),
    pytest.mark.filterwarnings("ignore:dropout option adds dropout:UserWarning"),
]


# ---------------------------------------------------------------------------- #
#  Fixtures
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='module')
def forcing():
    """Hourly cat-2453 forcing, keyed by BMI input name."""
    ds = xr.open_dataset(FORCING).set_coords('ids').swap_dims({'catchment-id': 'ids'})
    ds = ds.sel(ids='cat-2453').isel(time=slice(0, max(N_STEPS, LEAN_N_STEPS)))
    return {
        'atmosphere_water__liquid_equivalent_precipitation_rate': (
            ds['precip_rate'].values * 3600.0
        ),
        'land_surface_air__temperature': ds['TMP_2maboveground'].values - 273.15,
        'atmosphere_air_water~vapor__relative_saturation': (
            ds['SPFH_2maboveground'].values
        ),
        'land_surface_radiation~incoming~longwave__energy_flux': (
            ds['DLWRF_surface'].values
        ),
        'land_surface_radiation~incoming~shortwave__energy_flux': (
            ds['DSWRF_surface'].values
        ),
        'land_surface_air__pressure': ds['PRES_surface'].values,
        'land_surface_wind__x_component_of_velocity': (
            ds['UGRD_10maboveground'].values
        ),
        'land_surface_wind__y_component_of_velocity': (
            ds['VGRD_10maboveground'].values
        ),
    }


def write_config(
    tmp_path: Path,
    serialization: Optional[dict] = None,
    **warmup,
) -> Path:
    """Write a cat-2453 BMI config with short warmup windows."""
    with open(BMI_CONFIG) as f:
        config = yaml.safe_load(f)
    config['model_dir'] = str(MODEL_DIR)
    if serialization is not None:
        config['serialization'] = serialization
    config['warmup'] = {
        'cycle_days': 2,
        'daily_mode': 'periodic',
        'daily_warmup_days': 5,
        'hourly_mode': 'periodic',
        'hourly_warmup_hours': 48,
        **warmup,
    }
    if config['warmup']['daily_mode'] == 'cold':
        del config['warmup']['daily_warmup_days']
    if config['warmup']['hourly_mode'] == 'cold':
        del config['warmup']['hourly_warmup_hours']

    path = tmp_path / f"bmi_{len(list(tmp_path.iterdir()))}.yaml"
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)
    return path


def make_bmi(config_path: Path) -> MtsDeltaModelBmi:
    """Initialize a BMI from a config path."""
    bmi = MtsDeltaModelBmi(verbose=False)
    bmi.initialize(str(config_path))
    return bmi


def step(bmi: MtsDeltaModelBmi, forcing: dict, t: int) -> float:
    """Set step t's forcing, update, and return the runoff."""
    for name, values in forcing.items():
        bmi.set_value(name, np.float64(values[t]))
    bmi.update()
    dest = np.zeros(1)
    bmi.get_value(RUNOFF, dest)
    return float(dest[0])


def save(bmi: MtsDeltaModelBmi) -> np.ndarray:
    """Run the protocol's save sequence and return the payload bytes."""
    bmi.set_value(CREATE, np.zeros(1, dtype=np.int32))
    size = np.zeros(1, dtype=np.int64)
    bmi.get_value(SIZE, size)
    blob = np.zeros(int(size[0]), dtype=np.uint8)
    bmi.get_value(STATE, blob)
    bmi.set_value(FREE, np.zeros(1, dtype=np.int32))
    return blob


def restore(bmi: MtsDeltaModelBmi, blob: np.ndarray) -> None:
    """Run the protocol's restore sequence."""
    bmi.set_value(SIZE, np.array([blob.size], dtype=np.int64))
    bmi.set_value(STATE, blob)


# ---------------------------------------------------------------------------- #
#  Protocol surface
# ---------------------------------------------------------------------------- #


class TestReservedVariables:
    """Reserved variables resolve as the protocol requires."""

    @pytest.mark.parametrize(
        'name, units, var_type, itemsize',
        [
            (CREATE, 'ngen::trigger', 'int32', 4),
            (FREE, 'ngen::trigger', 'int32', 4),
            (SIZE, 'bytes', 'int64', 8),
            (STATE, 'ngen::opaque', 'uint8', 1),
        ],
    )
    def test_introspection(self, mts_bmi, name, units, var_type, itemsize):
        """Units, type, and item size match the protocol table."""
        assert mts_bmi.get_var_units(name) == units
        assert mts_bmi.get_var_type(name) == var_type
        assert mts_bmi.get_var_itemsize(name) == itemsize

    def test_not_listed(self, mts_bmi):
        """Reserved names are not in the input or output variable lists."""
        listed = set(mts_bmi.get_input_var_names()) | set(
            mts_bmi.get_output_var_names(),
        )
        assert not listed & {CREATE, FREE, SIZE, STATE}


# ---------------------------------------------------------------------------- #
#  Save and restore
# ---------------------------------------------------------------------------- #


class TestRoundTrip:
    """Save -> restore -> continue matches an uninterrupted run."""

    @pytest.mark.parametrize(
        'daily_mode, hourly_mode',
        [
            ('periodic', 'periodic'),
            ('once', 'periodic'),
        ],
    )
    def test_resume_matches_uninterrupted(
        self,
        tmp_path,
        forcing,
        daily_mode,
        hourly_mode,
    ):
        """A restored BMI continues bit-for-bit, across later warmups."""
        config = write_config(
            tmp_path,
            daily_mode=daily_mode,
            hourly_mode=hourly_mode,
        )
        source = make_bmi(config)
        for t in range(SAVE_STEP):
            step(source, forcing, t)
        blob = save(source)
        expected = [step(source, forcing, t) for t in range(SAVE_STEP, N_STEPS)]

        target = make_bmi(config)
        restore(target, blob)
        assert target.get_current_time() == SAVE_STEP * target.get_time_step()
        result = [step(target, forcing, t) for t in range(SAVE_STEP, N_STEPS)]

        assert np.count_nonzero(expected) > 0
        np.testing.assert_array_equal(result, expected)

    def test_restore_without_history(self, tmp_path, forcing, monkeypatch):
        """Without forcing history, a restored BMI refills and rejoins the run.

        Output matches up to the uninterrupted run's next warmup (step 216).
        Warmups wait for the buffers: none at 216 or 264, a daily rollout at
        312, then a full re-spin at 360, after which output matches again.
        """
        config = write_config(tmp_path, serialization={'forcing_history': False})
        source = make_bmi(config)
        for t in range(LEAN_SAVE_STEP):
            step(source, forcing, t)
        blob = save(source)
        assert blob.size < len(source._capture_state(forcing_history=True))
        expected = [
            step(source, forcing, t) for t in range(LEAN_SAVE_STEP, LEAN_N_STEPS)
        ]

        target = make_bmi(config)
        restore(target, blob)
        warmups = []
        run_warmup = target._run_warmup

        def record_warmup():
            run_warmup()
            phy_model = target._model.dpl_model.phy_model
            warmups.append((target._timestep, phy_model.lof_rollout))

        monkeypatch.setattr(target, '_run_warmup', record_warmup)
        result = [step(target, forcing, t) for t in range(LEAN_SAVE_STEP, LEAN_N_STEPS)]

        assert warmups[:2] == [(312, True), (360, False)]
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(
            result[: 216 - LEAN_SAVE_STEP],
            expected[: 216 - LEAN_SAVE_STEP],
        )
        np.testing.assert_array_equal(
            result[360 - LEAN_SAVE_STEP :],
            expected[360 - LEAN_SAVE_STEP :],
        )


# ---------------------------------------------------------------------------- #
#  Rejected restores
# ---------------------------------------------------------------------------- #


class TestRejectedRestore:
    """Restore fails loudly and leaves the model unchanged."""

    @pytest.fixture
    def warm(self, tmp_path, forcing):
        """A config path, a BMI stepped past its first warmup, and its bytes."""
        config = write_config(tmp_path)
        bmi = make_bmi(config)
        for t in range(SAVE_STEP):
            step(bmi, forcing, t)
        return config, bmi, save(bmi)

    @staticmethod
    def assert_unchanged(bmi: MtsDeltaModelBmi, blob: np.ndarray) -> None:
        np.testing.assert_array_equal(save(bmi), blob)

    def test_bad_magic(self, warm):
        """Bytes from another model are rejected."""
        _, bmi, blob = warm
        bad = blob.copy()
        bad[:8] = np.frombuffer(b'NOTDHBV2', dtype=np.uint8)
        with pytest.raises(ValueError, match='magic'):
            restore(bmi, bad)
        self.assert_unchanged(bmi, blob)

    def test_failed_apply_rolls_back(self, warm, monkeypatch):
        """If writing the model state fails midway, the old state returns."""
        config, bmi, blob = warm
        fresh = make_bmi(config)
        before = save(fresh)

        dpl_model = fresh._model.dpl_model
        original = dpl_model.load_runtime_state
        calls = []

        def fail_once(state):
            calls.append(state)
            if len(calls) == 1:
                raise RuntimeError("Injected failure")
            original(state)

        monkeypatch.setattr(dpl_model, 'load_runtime_state', fail_once)
        with pytest.raises(RuntimeError, match='Injected'):
            restore(fresh, blob)
        self.assert_unchanged(fresh, before)
