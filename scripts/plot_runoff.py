"""
Plot hydrograph of dhbv2 runoff generated from BMI/NextGen.

Be aware that if you use outputs generated from ./scripts/forward_mts_xx.py, the
first 358 days of runtime are already discarded. NextGen output will include
spinup, and therefore plots will show 0 flow for that period unless you remove it.
@leoglonz
"""

from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Assuming these are available in your environment
from dmg.core import format_resample_interval, timestep_resample
from numpy.typing import NDArray

# Setup pathing
pkg_root = Path(__file__).parent.parent


### -------- Settings -------- ###
# Numpy file with dhbv2 runoff data
SIM_PATH = f'{pkg_root}/output/dhbv_2_mts_cat-2453_runoff.npy'
SIM_PATH = f'{pkg_root}/output/dhbv_2_cat-2453_runoff.npy'
SAVE_PATH = f'{pkg_root}/output/runoff_simulation.png'
TIME_START = '2008-01-09 00:00:00'
TIME_END = '2010-12-30 23:00:00'

# Optional: subset the plot to a narrower time window (set to None to plot all)
PLOT_TIME_START = '2009-01-01'
PLOT_TIME_END = '2009-12-31'
PLOT_TIME_START = '2009-03-01'
PLOT_TIME_END = '2009-06-30'
### -------------------------- ###


def plot_hydrograph(
    timesteps: pd.DatetimeIndex,
    predictions: Union[NDArray[np.float32], torch.Tensor],
    resample: Literal['d', 'w', 'm', 'y'] = 'd',
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    title: str = 'Hydrograph',
    ylabel: str = 'Runoff',
    line_label: str = 'Prediction',
    color: str = 'blue',
    minor_ticks: bool = False,
    figsize: tuple = (12, 8),
    fontsize: int = 12,
    dpi: int = 100,
    save_path: Optional[str] = None,
) -> None:
    """Plot hydrograph for a single catchment (1D time series).

    Parameters
    ----------
    time_start : str, optional
        Start of the time window to plot (e.g. '2009-01-01'). If None, uses
        the beginning of the data.
    time_end : str, optional
        End of the time window to plot (e.g. '2009-12-31'). If None, uses
        the end of the data.
    """
    # --- 1. Data preparation ---
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Flatten data to ensure it is 1D (Time,) and not (Time, 1) or (1, Time)
    predictions = np.ravel(predictions)
    timesteps = timesteps[: -len(timesteps) + len(predictions)]

    if len(predictions) != len(timesteps):
        raise ValueError(
            f"Shape mismatch: Predictions length ({len(predictions)}) "
            f"does not match Timesteps length ({len(timesteps)})",
        )

    # Create df for resampling
    data = pd.DataFrame({'time': timesteps, 'pred': predictions})

    # Subset to requested time window
    if time_start is not None:
        data = data[data['time'] >= pd.Timestamp(time_start)]
    if time_end is not None:
        data = data[data['time'] <= pd.Timestamp(time_end)]

    # Resample
    data = timestep_resample(data, resolution=resample, method='mean')

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(
        data['time'],
        data['pred'],
        label=line_label,
        color=color,
        zorder=3,
    )

    # --- 3. Styling ---
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(f"Time [{format_resample_interval(resample)}]", fontsize=fontsize)

    ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    if minor_ticks:
        ax.minorticks_on()

    ax.tick_params(axis='x', rotation=45, labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=fontsize)

    plt.tight_layout()

    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    # Load simulation data
    runoff_sim = np.load(SIM_PATH)
    print(f"Loaded {runoff_sim.shape[0]} hours")

    # print("Removing first 358 days of warmup")
    # runoff_sim = runoff_sim[8592:]  # Remove first 8592 hours (358 days)

    # Generate time index
    time_index = pd.date_range(start=TIME_START, end=TIME_END, freq='h')

    # Plot hydrograph
    plot_hydrograph(
        timesteps=time_index,
        predictions=runoff_sim,
        resample='h',
        time_start=PLOT_TIME_START,
        time_end=PLOT_TIME_END,
        title='δHBV2.0 Runoff Simulation',
        ylabel='Runoff (m h-1)',
        line_label='Simulated Runoff',
        color='blue',
        minor_ticks=True,
        save_path=SAVE_PATH,
    )
