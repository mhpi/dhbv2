"""Supporting fixtures for δHBV2.0 MTS validation suite.

The suite compares three runoff series that should all describe the same
hydrograph for catchment `cat-2453`:

1. standalone -- the BMI driven directly by `scripts/mts_forward_example.py`
2. ngen -- the same BMI driven by NextGen through its Python BMI adapter
3. t-route -- NextGen's runoff routed through Muskingum-Cunge

(1) ensures model reproduces benchmark, (2) ensures ngen wiring feeds the model
what it expects, and (3) ensures the routing is consistent.

@leoglonz
"""

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_ROOT = Path(__file__).parent.parent

# Settings -------- #
ATOL = 1e-6
ROUTED_VOLUME_TOL = 0.01
ROUTED_MIN_CORR = 0.95


CATCHMENT = 'cat-2453'
CATCHMENT_FEATURE_ID = 2453

CATCHMENT_AREA_KM2 = 7.7805003825041545  # km2

# Full forcing window covered by the NetCDF file.
SIM_START = '2008-01-09 00:00:00'
SIM_END = '2010-12-30 23:00:00'
N_STEPS_FULL = 26088

# First step at which the model has cleared warmup, with `bmi_cat-2453.yaml`
# (351 daily + 336 hourly warmup, 14-day cycle).
FIRST_FLOW_STEP = 8760

# t-route routes only the post-spin-up remainder -- `start_datetime` in
# `routing_config.yaml` is set to where warmup ends, `nts` to the hours after.
N_STEPS_ROUTED = N_STEPS_FULL - FIRST_FLOW_STEP

RUNOFF_VAR = 'land_surface_water__runoff_volume_flux'
RUNOFF_UNITS = 'm h-1'

DATA_ROOT = PKG_ROOT / 'ngen_resources' / 'data'
FORCING_PATH = (
    DATA_ROOT / 'forcing' / 'camels_subset_2008-01-09 00_00_00_2010-12-30 23_00_00.nc'
)
MODEL_CONFIG_PATH = DATA_ROOT / 'dhbv_2_mts' / 'model' / 'dhbv_2_mts' / 'config.yaml'
BENCHMARK_PATH = PKG_ROOT / 'tests' / 'benchmarks' / 'mts_cat-2453_runoff.npz'

STANDALONE_RUN = PKG_ROOT / 'output' / f'dhbv_2_mts_{CATCHMENT}_runoff.npy'
# ----------------- #


# ---------------------------------------------------------------------------- #
#  Benchmark provenance
# ---------------------------------------------------------------------------- #


def sha256(path) -> str:
    """Return a file's SHA-256 hex digest, or '<missing>' if it is absent."""
    path = Path(path)
    if not path.exists():
        return '<missing>'
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bmi_config_path(catchment: str = CATCHMENT) -> Path:
    """Return the path to a catchment's BMI config."""
    return DATA_ROOT / 'dhbv_2_mts' / 'config' / f'bmi_{catchment}.yaml'


def package_versions() -> dict[str, str]:
    """Return installed versions of the packages that determine the numbers."""
    versions = {}
    for name in ('dhbv2', 'hydrodl2', 'dmg', 'torch', 'numpy'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '<unknown>')
        except Exception:  # noqa: BLE001 - a missing package is a valid answer
            versions[name] = '<not installed>'
    return versions


# ---------------------------------------------------------------------------- #
#  Loaders
# ---------------------------------------------------------------------------- #


def load_benchmark() -> tuple[np.ndarray, dict]:
    """Load the committed benchmark runoff [m h-1] and its dict."""
    with np.load(BENCHMARK_PATH, allow_pickle=False) as data:
        runoff = data['runoff'].astype(np.float64)
        metadata = json.loads(str(data['metadata']))
    return runoff, metadata


def load_ngen_csv(path) -> pd.Series:
    """Load a NextGen `cat-*.csv` as runoff [m h-1] indexed by timestamp."""
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    frame['Time'] = pd.to_datetime(frame['Time'].astype(str).str.strip())
    return frame.set_index('Time')[RUNOFF_VAR].astype(np.float64)


def load_troute_flow(stream_dir, feature_id: int = CATCHMENT_FEATURE_ID) -> pd.Series:
    """Load one feature's routed flow [m3 s-1] from t-route's NetCDF output.

    Parameters
    ----------
    stream_dir
        Directory holding `troute_output_*.nc`.
    feature_id
        Numeric hydrofabric feature id, e.g. `2453`.

    Returns
    -------
    pd.Series
        Routed flow indexed by timestamp.
    """
    import xarray as xr

    files = sorted(glob.glob(str(Path(stream_dir) / 'troute_output_*.nc')))
    if not files:
        raise FileNotFoundError(f"No troute_output_*.nc under {stream_dir}")

    times, flows = [], []
    for file in files:
        with xr.open_dataset(file) as dataset:
            flow = dataset['flow'].sel(feature_id=feature_id).values.ravel()
            times.extend(pd.to_datetime(dataset['time'].values))
            flows.extend(np.asarray(flow, dtype=np.float64))

    return pd.Series(flows, index=pd.DatetimeIndex(times)).sort_index()


def find_ngen_csv(run_dir, catchment: str = CATCHMENT) -> Path:
    """Locate a catchment's ngen output CSV anywhere under a run directory.

    Parameters
    ----------
    run_dir
        Directory holding the validator's own ngen run of the shipped example.
    catchment
        Catchment id, e.g. `cat-2453`.

    Returns
    -------
    Path
        The matching `cat-*.csv`.
    """
    matches = sorted(Path(run_dir).rglob(f'{catchment}.csv'))
    if not matches:
        pytest.skip(
            f"No {catchment}.csv under {run_dir}. Run the shipped example "
            f"through your own ngen first -- see docs/8-validation.md.",
        )
    return matches[0]


def find_troute_dir(run_dir) -> Path:
    """Locate the directory of t-route NetCDF output under a run directory.

    Parameters
    ----------
    run_dir
        Directory holding the validator's own ngen run of the shipped example.

    Returns
    -------
    Path
        Directory containing `troute_output_*.nc`.
    """
    matches = sorted(Path(run_dir).rglob('troute_output_*.nc'))
    if not matches:
        pytest.skip(
            f"No troute_output_*.nc under {run_dir}. Re-run the shipped example "
            f"with realization_troute_cat-2453.json to validate routing.",
        )

    # t-route may leave stray files beside its real output directory, so take
    # the directory holding the most rather than the first one found.
    by_dir: dict[Path, int] = {}
    for match in matches:
        by_dir[match.parent] = by_dir.get(match.parent, 0) + 1
    return max(by_dir, key=lambda d: (by_dir[d], str(d)))


def runoff_to_cms(runoff: pd.Series, area_km2: float = CATCHMENT_AREA_KM2) -> pd.Series:
    """Convert runoff [m h-1] to volumetric flow [m3 s-1]."""
    return runoff * area_km2 * 1e6 / 3600.0


def forcing_precip_mm_h(catchment: str = CATCHMENT) -> np.ndarray:
    """Return a catchment's hourly precipitation forcing [mm h-1]."""
    import xarray as xr

    with xr.open_dataset(FORCING_PATH) as dataset:
        series = (
            dataset.set_coords('ids')
            .swap_dims({'catchment-id': 'ids'})
            .sel(ids=catchment)['precip_rate']
            .values
        )
    return np.asarray(series, dtype=np.float64) * 3600.0


# ---------------------------------------------------------------------------- #
#  Metrics
# ---------------------------------------------------------------------------- #


def compare(sim: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compute agreement metrics between two aligned runoff series.

    Parameters
    ----------
    sim
        Series under test.
    ref
        Reference series.

    Returns
    -------
    dict
        `max_abs_diff`, `mean_abs_diff`, `nse`, `volume_ratio`, and
        `pearson_r`.
    """
    sim = np.asarray(sim, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = np.abs(sim - ref)

    denom = np.sum((ref - ref.mean()) ** 2)
    nse = 1.0 - np.sum((sim - ref) ** 2) / denom if denom > 0 else np.nan

    ref_total = ref.sum()
    volume_ratio = sim.sum() / ref_total if ref_total != 0 else np.nan

    varies = sim.std() > 0 and ref.std() > 0
    pearson_r = np.corrcoef(sim, ref)[0, 1] if varies else np.nan

    return {
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
        'nse': float(nse),
        'volume_ratio': float(volume_ratio),
        'pearson_r': float(pearson_r),
    }


def format_metrics(label: str, metrics: dict[str, float]) -> str:
    """Metric formatting."""
    return (
        f"{label}: max|diff|={metrics['max_abs_diff']:.3e} "
        f"mean|diff|={metrics['mean_abs_diff']:.3e} "
        f"NSE={metrics['nse']:.9f} "
        f"vol_ratio={metrics['volume_ratio']:.9f} "
        f"r={metrics['pearson_r']:.9f}"
    )


def align(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Restrict two time-indexed series to their common timestamps.

    Parameters
    ----------
    left, right
        Time-indexed series to align.

    Returns
    -------
    tuple
        The aligned series and the number of overlapping points.
    """
    index = left.index.intersection(right.index)
    return left.reindex(index).to_numpy(), right.reindex(index).to_numpy(), len(index)


def require_artifact(path, produced_by: str) -> None:
    """Skip the calling test if a validation artifact has not been produced.

    A *missing* artifact is a legitimate skip -- the user has not run that leg.
    An artifact that exists but disagrees is never skipped; that is the failure
    the suite is here to report.

    Parameters
    ----------
    path
        Expected artifact location.
    produced_by
        Command that produces it, quoted back to the user.
    """
    if not Path(path).exists():
        pytest.skip(
            f"{Path(path).name} not found at {path}. Produce it with: {produced_by}",
        )
