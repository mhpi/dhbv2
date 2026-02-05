"""
Compare daily and MTS (hourly) δHBV2.0 runoff simulations.

Aggregates MTS hourly output to daily means, extracts daily values from the
daily model (which repeats prediction/24 across 24 hours), then
computes agreement metrics and generates comparison plots.

@leoglonz
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup pathing
pkg_root = Path(__file__).parent.parent


### -------- Settings -------- ###
DAILY_PATH = f'{pkg_root}/output/dhbv_2_cat-2453_runoff.npy'
MTS_PATH = f'{pkg_root}/output/dhbv_2_mts_cat-2453_runoff.npy'
SAVE_DIR = f'{pkg_root}/output'

# Time range for the full simulation (including warmup)
TIME_START = '2008-01-09 00:00:00'
TIME_END = '2010-12-30 23:00:00'

# Warmup: daily uses 365 days, MTS uses 358 days. Use the longer one so both
# models have cleared their spinup.
WARMUP_HOURS = 365 * 24  # 8760

# Optional: restrict comparison to a narrower window (None = use all post-warmup)
PLOT_TIME_START: Optional[str] = None
PLOT_TIME_END: Optional[str] = None
### -------------------------- ###


def compute_metrics(daily: np.ndarray, mts: np.ndarray) -> dict:
    """Compute agreement metrics between daily and MTS daily-aggregated series."""
    mask = (daily != 0) | (mts != 0)  # skip joint-zero timesteps
    d, m = daily[mask], mts[mask]

    if len(d) == 0:
        return dict.fromkeys(['pearson_r', 'rmse', 'rel_bias', 'nse', 'kge'], np.nan)

    # Pearson correlation
    r = np.corrcoef(d, m)[0, 1] if len(d) > 1 else np.nan

    # RMSE
    rmse = np.sqrt(np.mean((d - m) ** 2))

    # Relative bias (MTS vs Daily)
    rel_bias = (m.sum() - d.sum()) / d.sum() if d.sum() != 0 else np.nan

    # NSE (treating daily as "observed")
    ss_res = np.sum((m - d) ** 2)
    ss_tot = np.sum((d - d.mean()) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # KGE
    r_kge = r
    alpha = m.std() / d.std() if d.std() != 0 else np.nan
    beta = m.mean() / d.mean() if d.mean() != 0 else np.nan
    kge = 1 - np.sqrt((r_kge - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {
        'pearson_r': r,
        'rmse': rmse,
        'rel_bias': rel_bias,
        'nse': nse,
        'kge': kge,
    }


def main():
    """Main comparison routine."""
    # Load data
    daily_raw = np.load(DAILY_PATH)
    mts_raw = np.load(MTS_PATH)
    print(f"Loaded daily: {daily_raw.shape}, MTS: {mts_raw.shape}")

    # Build hourly time index
    time_index = pd.date_range(start=TIME_START, end=TIME_END, freq='h')
    n = min(len(daily_raw), len(mts_raw), len(time_index))
    daily_raw = daily_raw[:n]
    mts_raw = mts_raw[:n]
    time_index = time_index[:n]

    # Strip warmup
    daily_raw = daily_raw[WARMUP_HOURS:]
    mts_raw = mts_raw[WARMUP_HOURS:]
    time_index = time_index[WARMUP_HOURS:]
    print(
        f"Post-warmup: {len(daily_raw)} hourly steps "
        f"({time_index[0]} to {time_index[-1]})",
    )

    # Aggregate to daily
    df = pd.DataFrame(
        {
            'time': time_index,
            'daily': daily_raw,
            'mts': mts_raw,
        },
    )
    df['date'] = df['time'].dt.date
    daily_agg = (
        df.groupby('date')
        .agg(
            daily_mean=('daily', 'mean'),
            mts_mean=('mts', 'mean'),
        )
        .reset_index()
    )
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])

    # Optional time subsetting
    if PLOT_TIME_START is not None:
        daily_agg = daily_agg[daily_agg['date'] >= pd.Timestamp(PLOT_TIME_START)]
    if PLOT_TIME_END is not None:
        daily_agg = daily_agg[daily_agg['date'] <= pd.Timestamp(PLOT_TIME_END)]

    d_vals = daily_agg['daily_mean'].values
    m_vals = daily_agg['mts_mean'].values

    # Metrics
    metrics = compute_metrics(d_vals, m_vals)
    print("\n===== Daily vs MTS Agreement =====")
    print(f"  Pearson r  : {metrics['pearson_r']:.4f}")
    print(f"  RMSE       : {metrics['rmse']:.6e}")
    print(f"  Rel. Bias  : {metrics['rel_bias']:+.2%}")
    print(f"  NSE        : {metrics['nse']:.4f}")
    print(f"  KGE        : {metrics['kge']:.4f}")
    print("==================================\n")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    # (1) Time series overlay
    ax = axes[0, 0]
    ax.plot(
        daily_agg['date'],
        d_vals,
        label='Daily model',
        color='tab:blue',
        linewidth=1,
    )
    ax.plot(
        daily_agg['date'],
        m_vals,
        label='MTS (daily avg)',
        color='tab:orange',
        linewidth=1,
        alpha=0.85,
    )
    ax.set_title('Daily-Aggregated Runoff Comparison')
    ax.set_ylabel('Runoff [m h-1]')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.4)

    # (2) Scatter plot
    ax = axes[0, 1]
    ax.scatter(d_vals, m_vals, s=8, alpha=0.5, edgecolors='none')
    lims = [0, max(d_vals.max(), m_vals.max()) * 1.05]
    ax.plot(lims, lims, 'k--', linewidth=0.8, label='1:1')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Daily model [m h-1]')
    ax.set_ylabel('MTS (daily avg) [m h-1]')
    ax.set_title(f'Scatter (r={metrics["pearson_r"]:.3f})')
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.4)

    # (3) Residuals over time
    ax = axes[1, 0]
    residuals = m_vals - d_vals
    ax.bar(daily_agg['date'], residuals, width=1, color='tab:green', alpha=0.6)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title('Residuals (MTS - Daily)')
    ax.set_ylabel('Difference [m h-1]')
    ax.set_xlabel('Time [hr]')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.4)

    # (4) Cumulative runoff
    ax = axes[1, 1]
    ax.plot(daily_agg['date'], np.cumsum(d_vals), label='Daily model', color='tab:blue')
    ax.plot(
        daily_agg['date'],
        np.cumsum(m_vals),
        label='MTS (daily avg)',
        color='tab:orange',
    )
    ax.set_title('Cumulative Runoff')
    ax.set_ylabel('Cumulative Runoff [m h-1]')
    ax.set_xlabel('Time [hr]')

    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.4)

    # Suptitle with metrics
    fig.suptitle(
        f'Daily vs MTS Comparison  |  r={metrics["pearson_r"]:.3f}  '
        f'NSE={metrics["nse"]:.3f}  KGE={metrics["kge"]:.3f}  '
        f'Bias={metrics["rel_bias"]:+.1%}',
        fontsize=12,
        y=1.01,
    )

    plt.tight_layout()
    save_path = f'{SAVE_DIR}/daily_vs_mts_comparison.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
