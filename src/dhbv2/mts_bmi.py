"""
BMI wrapper for interfacing δHBV2.0 MTS (hourly) with the NOAA-OWP NextGen
framework.

@Leo Lonzarich
"""

import json
import logging
import os
import time
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml
from bmipy import Bmi
from dmg import MtsModelHandler
from dmg.core.utils.dates import Dates
from numpy.typing import NDArray

from dhbv2.utils import bmi_array

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('MTS_BMI')

root_path = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------- #
# Dynamic input variables (CSDMS standard names)
# -------------------------------------------- #
_dynamic_input_vars = [
    ('atmosphere_water__liquid_equivalent_precipitation_rate', 'mm h-1'),
    ('land_surface_air__temperature', 'degK'),
    ('land_surface_water__potential_evaporation_volume_flux', 'mm h-1'),
]

# ------------------------------------------- #
# Static input variables (CSDMS standard names)
# ------------------------------------------- #
_static_input_vars = [
    ('ratio__mean_potential_evapotranspiration__mean_precipitation', '-'),
    ('atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate', 'mm d-1'),
    ('land_surface_water__Hargreaves_potential_evaporation_volume_flux', 'mm d-1'),
    ('land_vegetation__normalized_diff_vegetation_index', '-'),
    ('free_land_surface_water', 'mm d-1'),
    ('basin__mean_of_slope', 'm km-1'),
    ('soil_sand__grid', 'km2'),
    ('soil_clay__grid', 'km2'),
    ('soil_silt__grid', 'km2'),
    ('land_surface_water__glacier_fraction', 'percent'),
    ('soil_clay__attr', 'percent'),
    ('soil_gravel__attr', 'percent'),
    ('soil_sand__attr', 'percent'),
    ('soil_silt__attr', 'percent'),
    ('basin__mean_of_elevation', 'm'),
    ('atmosphere_water__daily_mean_of_temperature', 'degC'),
    ('land_surface_water__permafrost_fraction', '-'),
    ('bedrock__permeability', 'm2'),
    ('p_seasonality', '-'),
    ('land_surface_water__potential_evaporation_volume_flux_seasonality', '-'),
    ('land_surface_water__snow_fraction', 'percent'),
    ('atmosphere_water__precipitation_falling_as_snow_fraction', 'percent'),
    ('soil_clay__volume_fraction', 'percent'),
    ('soil_gravel__volume_fraction', 'percent'),
    ('soil_sand__volume_fraction', 'percent'),
    ('soil_silt__volume_fraction', 'percent'),
    ('soil_active-layer__porosity', '-'),
    ('basin__area', 'km2'),
    ('catchment__area', 'km2'),
    ('basin__length', 'km'),
]

# ------------------------------------- #
# Output variables (CSDMS standard names)
# ------------------------------------- #
_output_vars = [
    ('land_surface_water__runoff_volume_flux', 'm3 s-1'),
]

# ---------------------------------------------- #
# Internal variable names <-> CSDMS standard names
# ---------------------------------------------- #
_var_name_internal_map = {
    # ----------- Dynamic inputs -----------
    'P': 'atmosphere_water__liquid_equivalent_precipitation_rate',
    'T': 'land_surface_air__temperature',
    'PET': 'land_surface_water__potential_evaporation_volume_flux',
    # ----------- Static inputs -----------
    'aridity': 'ratio__mean_potential_evapotranspiration__mean_precipitation',
    'meanP': 'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate',
    'ETPOT_Hargr': 'land_surface_water__Hargreaves_potential_evaporation_volume_flux',
    'NDVI': 'land_vegetation__normalized_diff_vegetation_index',
    'FW': 'free_land_surface_water',
    'meanslope': 'basin__mean_of_slope',
    'SoilGrids1km_sand': 'soil_sand__grid',
    'SoilGrids1km_clay': 'soil_clay__grid',
    'SoilGrids1km_silt': 'soil_silt__grid',
    'glaciers': 'land_surface_water__glacier_fraction',
    'HWSD_clay': 'soil_clay__attr',
    'HWSD_gravel': 'soil_gravel__attr',
    'HWSD_sand': 'soil_sand__attr',
    'HWSD_silt': 'soil_silt__attr',
    'meanelevation': 'basin__mean_of_elevation',
    'meanTa': 'atmosphere_water__daily_mean_of_temperature',
    'permafrost': 'land_surface_water__permafrost_fraction',
    'permeability': 'bedrock__permeability',
    'seasonality_P': 'p_seasonality',
    'seasonality_PET': 'land_surface_water__potential_evaporation_volume_flux_seasonality',
    'snow_fraction': 'land_surface_water__snow_fraction',
    'snowfall_fraction': 'atmosphere_water__precipitation_falling_as_snow_fraction',
    'T_clay': 'soil_clay__volume_fraction',
    'T_gravel': 'soil_gravel__volume_fraction',
    'T_sand': 'soil_sand__volume_fraction',
    'T_silt': 'soil_silt__volume_fraction',
    'Porosity': 'soil_active-layer__porosity',
    'uparea': 'basin__area',
    'catchsize': 'catchment__area',
    'lengthkm': 'basin__length',
    # ----------- Outputs -----------
    'streamflow': 'land_surface_water__runoff_volume_flux',
}

_var_name_external_map = {v: k for k, v in _var_name_internal_map.items()}


def map_to_external(name: str):
    """Return the external name (exposed via BMI) for a given internal name."""
    return _var_name_internal_map[name]


def map_to_internal(name: str):
    """Return the internal name for a given external name (exposed via BMI)."""
    return _var_name_external_map[name]


# =============================================================================#
# MAIN BMI >>>>
# =============================================================================#


class MtsDeltaModelBmi(Bmi):
    """
    δHBV2.0 MTS BMI: NextGen-compatible, differentiable, physics-informed ML
    model for hydrologic forecasting (Yang et al., 2025; Song et al., 2024).

    Incorporates rolling window input caching for 358-day* lagged hourly
    runoff simulation.

    *We cache 351 days of aggregated daily inputs + 7 days of hourly inputs to
    warmup low and high-frequency model states for the following 7 days of
    hourly simulation. This window then rolls 7-days forward, repeating the
    warmup steps in preparation for the next 7 days of simulation.
    This may be removed in the future to support direct streaming, but for now
    we maintain a lag for representative model performance.

    Note: BMI can only run forward inference. Training code will be released in
        the δMG package (https://github.com/mhpi/generic_deltamodel) at a later
        date.
    """

    _att_map = {
        'model_name': 'δHBV2.0 MTS',
        'version': '0.1',
        'author_name': 'Leo Lonzarich',
        'time_step_size': 3600,
        'time_units': 's',
    }

    def __init__(self, verbose: bool = False) -> None:
        """Create a δHBV2.0 MTS BMI ready for initialization.

        This is a multitimescale (hourly) version of the δHBV2.0 BMI at
        (dhbv2/bmi.py).

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        verbose
            Enables debug print statements if True.
        """
        super().__init__()
        if verbose:
            t_start = time.time()
        self.proc_time = 0.0
        self._name = self._att_map['model_name']
        self._time_units = self._att_map['time_units']
        self._time_step_size = self._att_map['time_step_size']
        self.verbose = verbose

        # BMI state variables
        self._model = None
        self._states = None
        self._initialized = False
        self._is_warm = False
        self._timestep = 0
        self._start_time = 0.0
        self._end_time = np.finfo('d').max
        self._var_loc = 'node'
        self._var_grid_id = 0
        self.eps = 1e-6

        # Caching and warmup
        self.req_daily_history = 351  # 351d of daily data
        self.req_hourly_history = 168  # 7d/168hr of hourly data
        self.warmup_frequency = 168  # How often to run warmup (every 7d/168hr)
        self._steps_since_warmup = 0

        # Cache buffers
        self._hourly_buffer = []  # Sliding window of 168hr
        self._daily_buffer = []  # Sliding window of 351d
        self._current_day_accumulator = []  # Buffer for a single day of 24hr

        # Input/output vars
        self._dynamic_var = self._set_value_internal(_dynamic_input_vars, bmi_array([]))
        self._static_var = self._set_value_internal(_static_input_vars, bmi_array([]))
        self._output_vars = self._set_value_internal(_output_vars, bmi_array([]))

        # Other
        self.norm_stats = None
        self.bmi_config = None
        self.model_config = None

        if self.verbose:
            self.proc_time = time.time() - t_start
            log.debug(f"BMI init took {time.time() - t_start} s")

    def initialize(self, config_path: Optional[str] = None) -> None:
        """(Control function) Initialize the BMI model.

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        """
        t_start = time.time()

        # Read BMI configuration file
        try:
            with open(config_path) as f:
                self.bmi_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load BMI configuration: {e}") from e

        # Read model configuration file
        try:
            model_dir = os.path.join(
                root_path,
                '..',
                '..',
                'ngen_resources/data/dhbv2_mts/',
                self.bmi_config.get('model_path'),
            )
            model_config_path = os.path.join(model_dir, 'config.yaml')
            with open(model_config_path) as f:
                self.model_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model configuration: {e}") from e

        self.model_config = self.initialize_config(self.model_config)
        self.model_config['model_dir'] = model_dir

        # Load normalization statistics
        self._load_norm_stats()

        # Load static input vars from BMI config
        for name in self._static_var.keys():
            ext_name = map_to_internal(name)
            if ext_name in self.bmi_config.keys():
                self._static_var[name]['value'] = bmi_array(self.bmi_config[ext_name])
            else:
                log.warning(f"Static variable '{name}' not in BMI config. Skipping.")

        # Update internal parameters
        self._time_step_size = self.bmi_config.get(
            'time_step_size',
            self._time_step_size,
        )
        self.current_time = self.bmi_config.get('start_time', self._start_time)
        self._end_time = self.bmi_config.get('end_time', self._end_time)

        # Daily buffer extra size so that daily and hourly buffers don't overlap.
        self.dbuff = self.req_hourly_history // 24

        # Load model
        self.device = self.model_config['device']
        self.external_dtype = eval(self.bmi_config['dtype'])
        self.internal_dtype = eval(self.bmi_config['dtype'])
        self._model = self._load_model().to(self.device)
        # self._load_states()

        self._initialized = True
        if self.verbose:
            self.proc_time += time.time() - t_start
            log.info(
                f"BMI Initialize took {time.time() - t_start:.4f} s | Total runtime: {self.proc_time:.4f} s",
            )

    def update(self) -> None:
        """(Control function) Advance model state by one time step."""
        t_start = time.time()

        # 1. Cache raw data (no normalization) to allow daily aggregation.
        raw_forcing_t = self._get_current_forcing_raw()
        self._update_caches(raw_forcing_t)

        # 2. Check if we have enough history to run a prediction.
        #    (We need at least 351 days + 168 hours of data to do first warmup).
        if self._can_run_warmup():
            # --- WARMUP ---
            if self._is_warmup_trigger_step():
                self._model.dpl_model.phy_model.use_from_cache = False

                if self.verbose:
                    log.info(f"Step {self._timestep}: Running Warmup")

                # Prepare batch data (excludes current timestep)
                warmup_dict = self._prepare_input_dict(mode='warmup')

                # Run batch forward purely for side-effect: priming self.states
                self._do_forward(warmup_dict, batched=True)

                self._is_warm = True
                self._steps_since_warmup = 0

            # --- STEP ---
            if self._is_warm:
                self._model.dpl_model.phy_model.use_from_cache = True

                # Standard forward pass (single current timestep)
                # Run prediction for current hour using either fresh primed states
                # or states carried over from t-1.
                step_dict = self._prepare_input_dict(mode='step')
                predictions = self._do_forward(step_dict, batched=False)

                self._format_outputs(predictions)
                self._steps_since_warmup += 1
            else:
                self._set_empty_outputs()

        else:
            # Buffers are not full yet. Return zeros.
            if self.verbose and (self._timestep % 24 == 0):
                log.info(f"Step {self._timestep}: Filling buffers...")
            self._set_empty_outputs()

        self._timestep += 1

        # Track BMI runtime
        if self.verbose:
            self.proc_time += time.time() - t_start

    def update_until(self, end_time: float) -> None:
        """(Control function) Update model until a particular time.

        Note: Models should be trained standalone with dPLHydro_PMI first before
        forward predictions with this BMI.

        Parameters
        ----------
        end_time
            Time to run model until.
        """
        t_start = time.time()

        if end_time < self.get_current_time():
            log.warning(
                f"No update performed: end_time ({end_time}) <= current time ({self.get_current_time()}).",
            )
            return None

        n_steps, remainder = divmod(
            end_time - self.get_current_time(),
            self.get_time_step(),
        )

        if remainder != 0:
            log.warning(
                f"End time is not multiple of time step size. Updating until: {end_time - remainder}",
            )

        for _ in range(int(n_steps)):
            self.update()
        # self.update_frac(n_steps - int(n_steps))  # Fractional step updates.

        # Track BMI runtime
        self.proc_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Update Until took {time.time() - t_start:.4f} s | Total runtime: {self.proc_time:.4f} s",
            )

    def finalize(self) -> None:
        """(Control function) Finalize model."""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
        self._initialized = False

        if self.verbose:
            log.info("BMI model finalized.")

    # =========================================================================#
    # Caching Logic
    # =========================================================================#

    def _update_caches(self, raw_forcing: np.ndarray) -> None:
        """Manages the rolling windows.

        raw_forcing shape: (time=1, space nvars)
        """
        # 1. Add to hourly buffer
        self._hourly_buffer.append(raw_forcing)

        # Maintain hourly buffer size
        # keep exactly what's needed for the 7d/168hr warmup + current
        if len(self._hourly_buffer) - 1 > self.req_hourly_history:
            self._hourly_buffer.pop(0)

        # 2. Add to day accumulator
        self._current_day_accumulator.append(raw_forcing)

        # 3. Check if 24 hours have passed to create a daily entry
        if len(self._current_day_accumulator) == 24:
            # Aggregate: take mean across time dimension
            day_stack = np.concatenate(self._current_day_accumulator, axis=0)

            prcp = day_stack[:, :, 0].sum(axis=0)
            temp = day_stack[:, :, 1].mean(axis=0)
            pet = day_stack[:, :, 2].sum(axis=0)

            daily_agg = np.expand_dims(np.stack([prcp, temp, pet], axis=-1), axis=1)

            self._daily_buffer.append(daily_agg)
            self._current_day_accumulator = []  # Reset accumulator

            if not self._is_warm:
                # Maintain daily buffer size (keep 351 warmup + 1 current day)
                if len(self._daily_buffer) - 1 > self.req_daily_history + self.dbuff:
                    self._daily_buffer.pop(0)
            else:
                # Maintain daily buffer size during warm operation
                if len(self._daily_buffer) > self.req_daily_history + self.dbuff:
                    self._daily_buffer.pop(0)

    def _prepare_input_dict(self, mode='step') -> dict:
        """
        Constructs inputs for either history/cache warmup or single-step
        inference.

        Normalizes data on the fly.
        """
        # 1. Retrieve data from buffers
        raw_hourly_list = self._hourly_buffer  # Last 7 days of hourly
        raw_daily_list = self._daily_buffer  # Last 351 days of daily

        if mode == 'warmup':
            # CASE 1: BATCH WARMUP
            # We want history UP TO the current step, but NOT including it.
            # Slice: [-169 : -1] -> The 168 hours prior to current
            raw_hourly = np.concatenate(raw_hourly_list[0:-1], axis=0)

            # Daily: Just take the full available daily history (up to 351)
            # **Since daily buffer only updates every 24h, it naturally lags
            # correctly behind the current hourly-only window.
            raw_daily = np.concatenate(raw_daily_list[0 : -self.dbuff], axis=0)

        else:
            # CASE 2: SINGLE STEP INFERENCE
            # We want ONLY the current timestep.
            # Slice: [-1] -> The very last entry we just added.
            raw_hourly = np.concatenate(raw_hourly_list[-1:], axis=0)

            # For daily input during a single hourly step, we usually repeat
            # the last known daily value or use zeros if architecture implies.
            # Assuming the model handles the daily/hourly mismatch via the daily input tensor:
            if len(raw_daily_list) > 0:
                raw_daily = np.concatenate(raw_daily_list[-1:], axis=0)
            else:
                raw_daily = np.zeros_like(raw_hourly)

        # 2. Normalize
        x_norm_hourly = self._normalize(raw_hourly, 'dyn_input')
        x_norm_daily = self._normalize(raw_daily, 'dyn_input_daily')

        # 3. Format static attributes as tensors
        c_nn_norm, rc_nn_norm, outlet_topo, areas, elev_all, ac_all = (
            self._get_static_tensors()
        )

        # 4. Construct input tensors
        x_nn_norm_high_freq = torch.from_numpy(x_norm_hourly).to(self.internal_dtype)
        x_nn_norm_low_freq = torch.from_numpy(x_norm_daily).to(self.internal_dtype)

        x_phy_high_freq = torch.from_numpy(raw_hourly).to(self.internal_dtype)
        x_phy_low_freq = torch.from_numpy(raw_daily).to(self.internal_dtype)

        # Append static attributes to dynamic inputs
        c_nn_expanded1 = c_nn_norm.unsqueeze(0).repeat(
            x_nn_norm_high_freq.shape[0],
            1,
            1,
        )
        xc_nn_norm_high_freq = torch.cat((x_nn_norm_high_freq, c_nn_expanded1), dim=-1)

        c_nn_expanded2 = c_nn_norm.unsqueeze(0).repeat(
            x_nn_norm_low_freq.shape[0],
            1,
            1,
        )
        xc_nn_norm_low_freq = torch.cat((x_nn_norm_low_freq, c_nn_expanded2), dim=-1)

        data_dict = {
            'xc_nn_norm_high_freq': xc_nn_norm_high_freq.to(self.device),
            'x_phy_high_freq': x_phy_high_freq.to(self.device),
            'c_nn_norm': c_nn_norm.to(self.device),
            'rc_nn_norm': rc_nn_norm.to(self.device),
            'ac_all': ac_all.to(self.device),
            'elev_all': elev_all.to(self.device),
            'areas': areas.to(self.device),
            'outlet_topo': outlet_topo.to(self.device),
            # Add low freq items for warmup only
            'xc_nn_norm_low_freq': xc_nn_norm_low_freq.to(self.device)
            if mode == 'warmup'
            else None,
            'x_phy_low_freq': x_phy_low_freq.to(self.device)
            if mode == 'warmup'
            else None,
        }

        return data_dict

    def _normalize(self, data: np.ndarray, stat_key: str) -> np.ndarray:
        """Apply (X - Mean) / Std."""
        # Data is (Vars, Time, Space)
        mean = np.asarray(self.norm_stats['mean'][stat_key], dtype=np.float32)
        std = np.asarray(self.norm_stats['std'][stat_key], dtype=np.float32)

        while mean.ndim < data.ndim:
            mean = mean[np.newaxis, ...]
            std = std[np.newaxis, ...]

        return (data - mean) / (std + self.eps)

    def _get_current_forcing_raw(self) -> np.ndarray:
        """
        Extracts current BMI forcing variables into a (Vars, 1, Catchments)
        array.
        """
        var_x_list = self.model_config['model']['nn']['hif_model']['forcings']
        hourly_forcing = []
        for var in var_x_list:
            # Map name, get value, expand dims to (1, Catchments)
            val = self._dynamic_var[map_to_external(var)]['value']  # [time, space]
            hourly_forcing.append(val)

        return np.stack(hourly_forcing, axis=-1)  # [time, space, vars]

    def _get_static_tensors(self):
        """Helper to get static attributes."""
        mean_attr = np.asarray(
            self.norm_stats['mean']['static_input'],
            dtype=np.float32,
        )
        std_attr = np.asarray(self.norm_stats['std']['static_input'], dtype=np.float32)

        mean_attr_rout = np.asarray(
            self.norm_stats['mean']['rout_static_input'],
            dtype=np.float32,
        )
        std_attr_rout = np.asarray(
            self.norm_stats['std']['rout_static_input'],
            dtype=np.float32,
        )

        while mean_attr.ndim < 2:
            mean_attr = mean_attr[np.newaxis, ...]
            std_attr = std_attr[np.newaxis, ...]

        while mean_attr_rout.ndim < 2:
            mean_attr_rout = mean_attr_rout[np.newaxis, ...]
            std_attr_rout = std_attr_rout[np.newaxis, ...]

        var_c_list = self.model_config['model']['nn']['hif_model']['attributes']
        var_c_list2 = self.model_config['model']['nn']['hif_model']['attributes2']

        n_units = self._dynamic_var['land_surface_air__temperature']['value'].shape[0]
        outlet_topo = torch.eye(n_units)

        attr = []
        for var in var_c_list:
            attr.append(
                np.expand_dims(
                    self._static_var[map_to_external(var)]['value'],
                    axis=-1,
                ),
            )
        attr = np.stack(attr, axis=-1)

        attr_rout = []
        for var in var_c_list2:
            attr_rout.append(
                np.expand_dims(
                    self._static_var[map_to_external(var)]['value'],
                    axis=-1,
                ),
            )
        attr_rout = np.stack(attr_rout, axis=-1)

        attr_norm = (attr - mean_attr) / (std_attr + self.eps)
        attr_norm_rout = (attr_rout - mean_attr_rout) / (std_attr_rout + self.eps)

        c_nn_norm = torch.from_numpy(attr_norm)
        rc_nn_norm = torch.from_numpy(attr_norm_rout)

        elev_all = torch.from_numpy(
            self._static_var[map_to_external('meanelevation')]['value'],
        )
        ac_all = torch.from_numpy(self._static_var[map_to_external('uparea')]['value'])
        areas = torch.from_numpy(
            self._static_var[map_to_external('catchsize')]['value'],
        )

        if elev_all.ndim < 2:
            elev_all = elev_all.unsqueeze(0)
        if ac_all.ndim < 2:
            ac_all = ac_all.unsqueeze(0)
        if areas.ndim < 2:
            areas = areas.unsqueeze(0)

        return c_nn_norm, rc_nn_norm, outlet_topo, areas, elev_all, ac_all

    # =========================================================================#
    # Logic Helpers
    # =========================================================================#

    def _is_warmup_trigger_step(self) -> bool:
        """Trigger if we are at the start of a 7-day (freq=168 hour) cycle.

        We also need to ensure we actually have enough history (freq+1 hours)
        to slice [-freq:-1].
        """
        freq = self.warmup_frequency
        steps_active = self._steps_since_warmup

        # Check if buffer has history + current
        if len(self._hourly_buffer) <= freq:
            return False

        # Check if at a daily boundary
        if self._timestep % 24 != 0:
            return False

        # Every freq steps after:
        return (steps_active % freq) == 0

    def _can_run_warmup(self) -> bool:
        """
        Check if buffers have enough history to support a warmup run.

        Requires:
        - 351 days of daily history
        - 168 hours of hourly history
        """
        daily_ready = len(self._daily_buffer) >= self.req_daily_history + self.dbuff
        hourly_ready = len(self._hourly_buffer) >= self.req_hourly_history

        return daily_ready and hourly_ready

    # =========================================================================#
    # Helper functions
    # =========================================================================#

    def _set_empty_outputs(self):
        """Set output vars to 0 during warmup phase."""
        n_units = self._dynamic_var[
            'atmosphere_water__liquid_equivalent_precipitation_rate'
        ]['value'].shape[0]

        for name in self._output_vars:
            # Assuming output is 1D array of size [Catchments]
            # Get size from a known variable
            self._output_vars[name]['value'] = np.zeros(n_units)

    def _do_forward(
        self,
        data_dict: dict[str, Any],
        batched: bool = True,
    ) -> dict[str, NDArray]:
        """Forward model on the pre-formatted dictionary."""
        with torch.no_grad():
            prediction = self._model.dpl_model(data_dict, batched=batched)
            output = {
                'streamflow': prediction['Qs'].detach().cpu().numpy(),
            }
        return output

    def _load_model(self) -> MtsModelHandler:
        """Load a pre-trained model based on the configuration."""
        try:
            model = MtsModelHandler(self.model_config, verbose=self.verbose)
            model.load_model(epoch=self.model_config['test']['test_epoch'])
            model.dpl_model.eval()

            model.dpl_model.nn_model.lstm_mlp2.cache_states = True
            model.dpl_model.phy_model.low_freq_model.cache_states = True
            model.dpl_model.phy_model.high_freq_model.cache_states = True

            model.dpl_model.phy_model.lof_from_cache = True
            model.dpl_model.phy_model.load_from_cache = True

            model.dpl_model.phy_model.high_freq_model.use_distr_routing = False
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}") from e

    # def _load_states(self) -> None:
    #     """Load saved model states if specified in BMI config."""
    #     if self._states is None:
    #         path = os.path.join(
    #             self.model_config['model_dir'],
    #             '..',
    #             self.bmi_config['states_name'],
    #         )
    #         self._states = torch.load(os.path.abspath(path))
    #         try:
    #             self._model.dpl_model.phy_model.load_states(self._states)
    #             self._model.dpl_model.phy_model.low_freq_model.load_states(
    #                 self._states[0],
    #             )
    #         except RuntimeError as e:
    #             raise RuntimeError(f"Failed to load model states: {e}") from e

    def _format_outputs(self, outputs):
        """Format model outputs as BMI outputs."""
        for name in self._output_vars.keys():
            internal_name = map_to_internal(name)
            if outputs is None:
                log.error("No outputs to format. Check model predictions.")
                output_val = np.zeros(1)
            elif not isinstance(outputs[internal_name], np.ndarray):
                output_val = outputs[internal_name].detach().cpu().numpy()
            else:
                output_val = outputs[internal_name]

            if output_val.ndim != 1:
                output_val = output_val.squeeze()
            self._output_vars[name]['value'] = output_val

            # self._output_vars[name]['value'] = np.append(
            #     self._output_vars[name]['value'],
            #     output_val,
            # )

    def _load_norm_stats(self) -> None:
        """Load normalization statistics."""
        path = os.path.join(
            self.model_config["model_dir"],
            "normalization_statistics.json",
        )
        try:
            with open(os.path.abspath(path)) as f:
                self.norm_stats = json.load(f)
        except ValueError as e:
            raise ValueError("Normalization statistics not found.") from e

    def _to_internal_units(self, var_name: str, values: list) -> list:
        """Convert external units to internal model units."""
        if var_name == 'land_surface_air__temperature':
            # degK to degC
            return [v - 273.15 for v in values]
        return values

    def _to_external_units(self, var_name: str, values: list) -> list:
        """Convert internal model units to external units."""
        if var_name == 'atmosphere_water__liquid_equivalent_precipitation_rate':
            # mm h-1 to m3 s-1  (use catchment area)
            area = self._static_var[map_to_external('catchment__area')]['value']
            return [v * 1000 / 3600 * area for v in values]
        return values

    def array_to_tensor(self) -> None:
        """Converts input values into Torch tensor object to be read by model."""
        raise NotImplementedError("array_to_tensor")

    def tensor_to_array(self) -> None:
        """
        Converts model output Torch tensor into date + gradient arrays to be
        passed out of BMI for backpropagation, loss, optimizer tuning.
        """
        raise NotImplementedError("tensor_to_array")

    def get_tensor_slice(self):
        """Get tensor of input data for a single timestep."""
        raise NotImplementedError("get_tensor_slice")

    def get_var_type(self, var_name):
        """Data type of variable."""
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_standard_name):
        """Get units of variable.

        Parameters
        ----------
        var_standard_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        return {**self._dynamic_var, **self._output_vars}[var_standard_name]['units']

    def get_var_nbytes(self, var_name):
        """Get units of variable."""
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        """Get item size of variable."""
        return self.get_value_ptr(name).itemsize

    def get_var_location(self, name):
        """Location of variable."""
        if name in {**self._dynamic_var, **self._output_vars}.keys():
            return self._var_loc
        else:
            raise KeyError(f"Variable '{name}' not supported.")

    def get_var_grid(self, var_name):
        """Grid id for a variable."""
        if var_name in {**self._dynamic_var, **self._output_vars}.keys():
            return self._var_grid_id
        else:
            raise KeyError(f"Variable '{var_name}' not supported.")

    def get_grid_rank(self, grid_id: int):
        """Rank of grid."""
        if grid_id == 0:
            return 1
        raise RuntimeError(f"Unsupported grid rank: {grid_id!s}. only support 0")

    def get_grid_size(self, grid_id):
        """Size of grid."""
        if grid_id == 0:
            return 1
        raise RuntimeError(f"unsupported grid size: {grid_id!s}. only support 0")

    def get_value_ptr(self, var_standard_name: str) -> np.ndarray:
        """Reference to values."""
        return {**self._dynamic_var, **self._static_var, **self._output_vars}[
            var_standard_name
        ]['value']

    def get_value(self, var_name: str, dest: NDArray):
        """Return copy of variable values."""
        try:
            # tmp = self.get_value_ptr(var_name)[self._timestep - 1,].flatten()
            tmp = self.get_value_ptr(var_name).flatten()
            dest[:] = self._to_external_units(var_name, tmp.tolist())
        except RuntimeError as e:
            raise e
        return dest

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at indices."""
        tmp = self.get_value_ptr(var_name).take(indices)
        dest[:] = self._to_external_units(var_name, tmp.tolist())
        return dest

    @staticmethod
    def _set_value_internal(
        vars: list[tuple[str, str]],
        var_value: NDArray,
    ) -> dict[str, dict[str, Union[NDArray, str]]]:
        """Set the values of given variables.

        Returns
        -------
        dict
            Dictionary of variable names mapping to their values and units.
            e.g.,
            {
                'var_name_1': {'value': array([...]), 'units': 'unit_1'},
                'var_name_2': {'value': array([...]), 'units': 'unit_2'},
                ...
            }
        """
        var_dict = {}
        for item in vars:
            var_dict[item[0]] = {'value': var_value.copy(), 'units': item[1]}
        return var_dict

    def set_value(self, var_name, values: list):
        """Set variable value."""
        if not isinstance(values, list):
            values = [values]
        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if var_name in dict.keys():
                values = self._to_internal_units(var_name, values)
                dict[var_name]['value'] = np.expand_dims(
                    np.array(values),
                    axis=1,
                )  # [time, space]
                break

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices."""
        if not isinstance(src, list):
            src = [src]

        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if name in dict.keys():
                src = self._to_internal_units(name, src)
                for i in inds:
                    dict[name]['value'][i] = src[i]
                break

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._dynamic_var)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_vars)

    def get_input_var_names(self):
        """Get names of input variables."""
        return list(self._dynamic_var.keys())

    def get_output_var_names(self):
        """Get names of output variables."""
        return list(self._output_vars.keys())

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        raise NotImplementedError("get_grid_shape")

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        raise NotImplementedError("get_grid_spacing")

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        raise NotImplementedError("get_grid_origin")

    def get_grid_type(self, grid_id):
        """Type of grid."""
        if grid_id == 0:
            return "scalar"
        raise RuntimeError(f"unsupported grid type: {grid_id!s}. only support 0")

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        """Current time of model."""
        return self._timestep * self._att_map["time_step_size"] + self._start_time

    def get_time_step(self):
        """Time step size of model."""
        return self._att_map["time_step_size"]

    def get_time_units(self):
        """Time units of model."""
        return self._att_map["time_units"]

    def get_grid_edge_count(self, grid):
        """Get grid edge count."""
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        """Get grid edge nodes."""
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        """Get grid face count."""
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        """Get grid face nodes."""
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        """Get grid node count."""
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        """Get grid nodes per face."""
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        """Get grid face edges."""
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        """Get grid x-coordinates."""
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        """Get grid y-coordinates."""
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        """Get grid z-coordinates."""
        raise NotImplementedError("get_grid_z")

    def initialize_config(
        self,
        config: Union[dict, dict],
    ) -> dict[str, Any]:
        """Parse and initialize configuration settings.

        Parameters
        ----------
        config
            Configuration settings from Hydra.

        Returns
        -------
        dict
            Formatted configuration settings.
        """
        config['device'], config['dtype'] = self.set_system_spec(config)

        # Convert date ranges to integer values.
        rho = config['model']['rho']
        # train_time = Dates(config['train'], rho)
        # test_time = Dates(config['test'], rho)
        sim_time = Dates(config['sim'], rho)
        # all_time = Dates(config['observations'], rho)

        # exp_time_start = min(
        #     train_time.start_time,
        #     train_time.end_time,
        #     test_time.start_time,
        #     test_time.end_time,
        # )
        # exp_time_end = max(
        #     train_time.start_time,
        #     train_time.end_time,
        #     test_time.start_time,
        #     test_time.end_time,
        # )

        # config['train_time'] = [train_time.start_time, train_time.end_time]
        # config['test_time'] = [test_time.start_time, test_time.end_time]
        config['sim_time'] = [sim_time.start_time, sim_time.end_time]
        # config['experiment_time'] = [exp_time_start, exp_time_end]
        # config['all_time'] = [all_time.start_time, all_time.end_time]

        if config.get('model_dir') is None:
            config['model_dir'] = ''
        config['plot_dir'] = ''
        config['sim_dir'] = ''
        config['log_dir'] = ''

        # Convert string back to data type.
        config['dtype'] = eval(config['dtype'])

        for name in ['hif_model', 'lof_model']:
            config['model']['phy'][name]['nearzero'] = float(
                config['model']['phy'][name]['nearzero'],
            )

        return config

    def set_system_spec(self, config: dict) -> tuple[str, str]:
        """Set the device and data type for the model on user's system.

        Parameters
        ----------
        cuda_devices
            List of CUDA devices to use. If None, the first available device is used.

        Returns
        -------
        tuple[str, str]
            The device type and data type for the model.
        """
        if config["device"] == "cpu":
            device = torch.device("cpu")
        elif config["device"] == "mps":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                raise ValueError("MPS is not available on this system.")
        elif config["device"] == "cuda":
            # Set the first device as the active device.
            if (
                torch.cuda.is_available()
                and config["gpu_id"] < torch.cuda.device_count()
            ):
                device = torch.device(f"cuda:{config['gpu_id']}")
                torch.cuda.set_device(device)
            else:
                raise ValueError(
                    f"Selected CUDA device {config['gpu_id']} is not available.",
                )
        else:
            raise ValueError(f"Invalid device: {config['device']}")

        dtype = torch.float32
        return str(device), str(dtype)
