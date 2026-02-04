"""
BMI wrapper for interfacing δHBV2.0 (daily) with the NOAA-OWP/ngen framework.

This BMI receives hourly inputs from ngen, aggregates them to daily values,
and makes a daily prediction (then repeated for each of the following 24 hours until a new prediction is made).

@Leo Lonzarich
"""

import json
import os
import time
from typing import Union

import numpy as np
import torch
import yaml
from bmipy import Bmi
from dmg import ModelHandler
from dmg.core import Dates
from numpy.typing import NDArray

from dhbv2.log import configure_logging, log
from dhbv2.pet import penman_monteith_pet
from dhbv2.utils import RingBuffer

root_path = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------ #
# (1) Dynamic input variables (CSDMS standard names)
# ------------------------------------------------ #
_dynamic_input_vars = [
    ('atmosphere_water__liquid_equivalent_precipitation_rate', 'mm h-1'),
    ('land_surface_air__temperature', 'degC'),
    ('atmosphere_air_water~vapor__relative_saturation', 'g g-1'),
    ('land_surface_radiation~incoming~longwave__energy_flux', 'W m-2'),
    ('land_surface_radiation~incoming~shortwave__energy_flux', 'W m-2'),
    ('land_surface_air__pressure', 'kPa'),
    ('land_surface_wind__x_component_of_velocity', 'm s-1'),
    ('land_surface_wind__y_component_of_velocity', 'm s-1'),
]

# ----------------------------------------------- #
# (2) Static input variables (CSDMS standard names)
# ----------------------------------------------- #
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
]

# ----------------------------------------- #
# (3) Output variables (CSDMS standard names)
# ----------------------------------------- #
_output_vars = [
    ('land_surface_water__runoff_volume_flux', 'm d-1'),
]

# -------------------------------------------------- #
# (4) Internal variable names <-> CSDMS standard names
# -------------------------------------------------- #
_var_name_internal_map = {
    # ----------- Dynamic inputs -----------
    'P': 'atmosphere_water__liquid_equivalent_precipitation_rate',
    'T': 'land_surface_air__temperature',
    'SPFH': 'atmosphere_air_water~vapor__relative_saturation',
    'DLWRF': 'land_surface_radiation~incoming~longwave__energy_flux',
    'DSWRF': 'land_surface_radiation~incoming~shortwave__energy_flux',
    'PRES': 'land_surface_air__pressure',
    'U': 'land_surface_wind__x_component_of_velocity',
    'V': 'land_surface_wind__y_component_of_velocity',
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


# -------------------------------------------------- #
# (5) BMI
# -------------------------------------------------- #
class DeltaModelBmi(Bmi):
    """
    (δHBV2.0 Daily BMI) NextGen-compatible, differentiable, physics-informed ML
    rainfall-runoff model for hydrologic forecasting (Song et al., 2025).

    A daily version of the δHBV2.0 model that receives hourly inputs from ngen,
    aggregates them to daily values, and makes predictions at the end of each
    day. Predictions are repeated for all 24 hourly outputs until a new
    prediction is available.

    Parameters
    ----------
    verbose
        Enables debug print statements if True.

    Source Code
    -----------
    -> https://github.com/mhpi/hydrodl2 for core HBV models.
    -> https://github.com/mhpi/dmg for differentiable model framework.
    -> https://github.com/csdms/bmi-python for BMI interface.

    ---

    NOTE: This BMI uses both numpy arrays and pytorch tensors for internal
        computations (dtype is preserved).
    NOTE: At least `rho` days (default 365) of data are required before the
        first model prediction is returned.
    NOTE: BMI can only run forward inference. Training code will be released in
        the δMG package (https://github.com/mhpi/generic_deltamodel) at a later
        date.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self._name = 'δHBV2.0 Daily'
        self._version = '1.0'
        self._author_name = 'Leo Lonzarich'

        self.verbose = verbose

        # --- BMI state variables ---
        self._model = None
        self._initialized = False
        self._is_warm = False

        self._dtype = 'float64'
        self._set_dtype()
        self.device = torch.device('cpu')

        self.n_units = 1

        self._time_units = 's'
        self._time_step_size = 3600
        self._timestep = 0
        self._start_time = 0.0
        self._end_time = np.finfo('d').max
        self._proc_time = 0.0
        self._var_loc = 'node'
        self._var_grid_id = 0
        self.eps = 1e-6

        # --- Caching and warmup ---
        self.req_daily_history = 365  # 365d of previous data for warmup
        self._hour_in_day = 0  # Track position within current day (0-23)

        # --- Cache buffers ---
        self._daily_buffer = None  # RingBuffer for rolling window of daily data
        self._day_accumulator = None  # Buffer for a single day of 24hr
        self._day_accumulator_ptr = 0

        # --- Current prediction (repeated for 24 hours) ---
        self._current_prediction = None

        # --- Model states (LSTM hidden/cell, physical states) ---
        # self._lstm_states = None
        # self._phy_states = None

        # --- Model variables ---
        self._dynamic_var = self._set_value_internal(
            _dynamic_input_vars,
            self._bmi_array([0.0]),
        )
        self._static_var = self._set_value_internal(
            _static_input_vars,
            self._bmi_array([0.0]),
        )
        self._output_vars = self._set_value_internal(
            _output_vars,
            self._bmi_array([0.0]),
        )

        # --- Other ---
        self.norm_stats = None
        self.bmi_config = None
        self.model_config = None

        if self.verbose:
            # Logging appears in CLI during ngen runtimes.
            configure_logging('debug')

    def initialize(self, config_file: str) -> None:
        """(Control function) Perform startup tasks for the BMI.

        Parameters
        ----------
        config_file
            The path to the BMI configuration file.
        """
        t_start = time.time()

        # --- Read BMI configuration file ---
        try:
            with open(config_file) as f:
                self.bmi_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed loading BMI configuration: {e}") from e

        # --- Read model configuration file ---
        try:
            core_path = self.bmi_config.get('model_dir')
            if os.path.exists(core_path):
                # Path inside ngen
                model_dir = core_path
            else:
                # Path for local testing
                model_dir = os.path.join(
                    root_path,
                    '..',
                    '..',
                    'ngen_resources/',
                    core_path,
                )
            model_config_path = os.path.join(model_dir, 'config.yaml')
            with open(model_config_path) as f:
                self.model_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed loading model configuration: {e}") from e

        self.model_config = self.initialize_config(self.model_config)
        self.model_config['model_dir'] = model_dir

        # --- Load model input statistics for normalization ---
        self._load_norm_stats()

        # --- Load static input variables ---
        for name in self._static_var.keys():
            ext_name = map_to_internal(name)
            if ext_name in self.bmi_config.keys():
                self._static_var[name]['value'] = self._bmi_array(
                    self.bmi_config[ext_name],
                )
            else:
                log.warning(f"Static variable '{name}' not in BMI config. Skipping.")

        # --- Update internal parameters ---
        self._time_step_size = self.bmi_config.get(
            'time_step_size',
            self._time_step_size,
        )
        self._dtype = self.bmi_config.get('dtype', self._dtype)
        self._set_dtype()
        self.device = self.model_config.get('device', self.device)

        # --- Set warmup period from config ---
        self.req_daily_history = self.model_config['model'].get('rho', 365)

        # --- Load model ---
        self._model = self._load_model()

        # --- Buffer initialization ---
        n_vars = len(self.model_config['model']['phy']['forcings'])

        # RingBuffer for daily aggregated data (fixed-size circular buffer)
        self._daily_buffer = RingBuffer(
            shape=(self.req_daily_history + 1, 1, n_vars),
            dtype=self.np_dtype,
        )
        self._day_accumulator = np.zeros(
            (24, 1, n_vars),
            dtype=self.np_dtype,
        )
        self._day_accumulator_ptr = 0

        self._initialized = True

        if self.verbose:
            self._proc_time += time.time() - t_start
            log.info(
                f"BMI Initialize took {time.time() - t_start:.4f} s | ",
                f"Total runtime: {self._proc_time:.4f} s",
            )

    def update(self) -> None:
        """(Control function) Advance BMI state by one time step.

        NOTE: ngen uses this method for model forward.

        This method:
        1. Accumulates hourly forcing data
        2. Every 24 hours, aggregates to daily values and runs prediction
        3. Returns the current prediction (repeated until new one available)
        """
        t_start = time.time()

        # 1. Cache raw data (no normalization) to allow daily aggregation.
        forcing = self._get_current_forcing()
        self._day_accumulator[self._day_accumulator_ptr] = forcing[0]
        self._day_accumulator_ptr += 1
        self._hour_in_day += 1

        # 2. Check if we've completed a day (24 hours)
        if self._day_accumulator_ptr == 24:
            # Aggregate hourly to daily
            daily_data = self._aggregate_to_daily()

            # Add to daily buffer (RingBuffer handles trimming automatically)
            self._daily_buffer.append(daily_data[0])  # Shape (1, n_vars) for RingBuffer

            # Reset accumulator
            self._day_accumulator_ptr = 0
            self._hour_in_day = 0

            # 3. Check if we have enough history to make a prediction
            if len(self._daily_buffer) > self.req_daily_history:
                if self.verbose:
                    log.info(f"Step {self._timestep}: Running daily prediction")

                # Prepare input data
                data_dict = self._prepare_input_data()

                # Run prediction
                predictions = self._do_forward(data_dict)

                # Store prediction to be repeated for next 24 hours
                self._current_prediction = predictions
                self._is_warm = True

        # 4. Format outputs
        if self._is_warm and self._current_prediction is not None:
            self._format_outputs(self._current_prediction)
        else:
            if self.verbose and (self._timestep % 24 == 0):
                log.info(
                    f"Step {self._timestep}: Filling buffers... "
                    f"({len(self._daily_buffer)}/{self.req_daily_history} days)",
                )
            self._set_empty_outputs()

        self._timestep += 1

        if self.verbose:
            self._proc_time += time.time() - t_start

    def update_until(self, time: float) -> None:
        """(Control function) Advance BMI state until the given time.

        Parameters
        ----------
        time
            A model time later than the current model time.
        """
        t_start = time.time()  # Renamed to avoid shadowing

        if t_start < self.get_current_time():
            log.warning(
                f"No update performed: end_time ({t_start}) <= current time "
                f"({self.get_current_time()}).",
            )
            return None

        n_steps, remainder = divmod(
            t_start - self.get_current_time(),
            self.get_time_step(),
        )

        if remainder != 0:
            log.warning(
                f"End time is not multiple of time step size. "
                f"Updating until: {t_start - remainder}",
            )

        for _ in range(int(n_steps)):
            self.update()

        if self.verbose:
            import time as time_module

            self._proc_time += time_module.time() - t_start
            log.info(
                f"BMI Update Until took {time_module.time() - t_start:.4f} s | "
                f"Total runtime: {self._proc_time:.4f} s",
            )

    def finalize(self) -> None:
        """(Control function) Perform tear-down tasks for the model."""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
        self._initialized = False

        if self.verbose:
            log.info("BMI model finalized.")

    # =========================================================================#

    # Caching Logic

    # =========================================================================#

    def _aggregate_to_daily(self) -> NDArray:
        """Aggregate 24 hours of data to daily values.

        Returns
        -------
        NDArray
            Daily aggregated forcing data. Shape (1, 1, n_vars).
        """
        # P: sum over day
        prcp = self._day_accumulator[:, :, 0].sum(axis=0, keepdims=True)

        # T: mean over day
        temp = self._day_accumulator[:, :, 1].mean(axis=0, keepdims=True)

        # PET: sum over day
        pet = self._day_accumulator[:, :, 2].sum(axis=0, keepdims=True)

        daily_agg = np.concatenate([prcp, temp, pet], axis=-1)
        return daily_agg[np.newaxis, ...]  # Shape (1, 1, n_vars)

    def _prepare_input_data(
        self,
        batched: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Constructs inputs for daily model inference and normalizes data.

        All calculations in numpy, converted to torch tensors at the end.

        Parameters
        ----------
        batched
            If True, prepares data for warmup (batch mode).
            If False, prepares data for single-step inference (sequential).

        Returns
        -------
        dict
            Dictionary of input tensors for the model.
        """
        # Get daily buffer: shape (n_days, 1, n_vars)
        raw_daily = self._daily_buffer.get_ordered()

        # Normalize
        x_norm = self._normalize(raw_daily, 'dyn_input')

        # Get static variables
        c_nn_norm, areas, elev_all, ac_all = self._get_static_var_tensors()

        # Convert to tensors
        x_nn_norm = self._bmi_tensor(x_norm)
        x_phy = self._bmi_tensor(raw_daily)

        # Append static variables to dynamic inputs
        c_nn_expanded = c_nn_norm.unsqueeze(0).repeat(x_nn_norm.shape[0], 1, 1)
        xc_nn_norm = torch.cat((x_nn_norm, c_nn_expanded), dim=-1)

        return {
            'xc_nn_norm': xc_nn_norm,
            'x_phy': x_phy,
            'c_nn_norm': c_nn_norm,
            'ac_all': ac_all,
            'elev_all': elev_all,
            'areas': areas,
        }

    def _normalize(self, data: NDArray, name: str) -> NDArray:
        """Normalize model inputs with saved training data statistics.

        Gaussian norm: (X - Mean) / Std.

        Parameters
        ----------
        data
            Raw input data to normalize. Shape (time, space, vars).
        name
            Name of the variable to normalize.

        Returns
        -------
        NDArray
            Normalized data. Shape (time, space, vars).
        """
        mean = np.asarray(self.norm_stats['mean'][name])  # , dtype=self.np_dtype)
        std = np.asarray(self.norm_stats['std'][name])  # , dtype=self.np_dtype)

        while mean.ndim < data.ndim:
            mean = mean[np.newaxis, ...]
            std = std[np.newaxis, ...]

        return (data - mean) / (std + self.eps)

    def _get_current_forcing(self) -> NDArray:
        """
        Extracts current step forcing variables into an array.

        Returns
        -------
        NDArray
            Current forcing data. Shape (time, space, vars).
        """
        var_x_list = self.model_config['model']['phy']['forcings']
        hourly_forcing = []
        for var in var_x_list:
            if var == 'PET':
                # Calculate PET on-the-fly (would be nice to do in parallel)
                val = penman_monteith_pet(
                    temp=self._dynamic_var[map_to_external('T')]['value'],
                    spfh=self._dynamic_var[map_to_external('SPFH')]['value'],
                    dlwrf=self._dynamic_var[map_to_external('DLWRF')]['value'],
                    dswrf=self._dynamic_var[map_to_external('DSWRF')]['value'],
                    pres=self._dynamic_var[map_to_external('PRES')]['value'],
                    ugrd_10m=self._dynamic_var[map_to_external('U')]['value'],
                    vgrd_10m=self._dynamic_var[map_to_external('V')]['value'],
                )
            else:
                val = self._dynamic_var[map_to_external(var)]['value']  # [time, space]
            hourly_forcing.append(val)

        return np.stack(hourly_forcing, axis=-1)  # [time, space, vars]

    def _get_static_var_tensors(self) -> tuple[torch.Tensor, ...]:
        """Helper to prepare static variables.

        Returns
        -------
        tuple
            Tensors for:
            - c_nn_norm: Normalized catchment attributes for NN.
            - areas: Catchment areas.
            - elev_all: Catchment elevations.
            - ac_all: Catchment upstream areas.
        """
        mean_attr = np.asarray(
            self.norm_stats['mean']['static_input'],
            dtype=self.np_dtype,
        )
        std_attr = np.asarray(
            self.norm_stats['std']['static_input'],
            dtype=self.np_dtype,
        )

        while mean_attr.ndim < 2:
            mean_attr = mean_attr[np.newaxis, ...]
            std_attr = std_attr[np.newaxis, ...]

        var_c_list = self.model_config['model']['nn']['attributes']

        attr = []
        for var in var_c_list:
            attr.append(
                np.expand_dims(
                    self._static_var[map_to_external(var)]['value'],
                    axis=-1,
                ),
            )
        attr = np.stack(attr, axis=-1)

        attr_norm = (attr - mean_attr) / (std_attr + self.eps)

        c_nn_norm = self._bmi_tensor(attr_norm)
        elev_all = self._bmi_tensor(
            self._static_var[map_to_external('meanelevation')]['value'],
        )
        ac_all = self._bmi_tensor(
            self._static_var[map_to_external('uparea')]['value'],
        )
        areas = self._bmi_tensor(
            self._static_var[map_to_external('catchsize')]['value'],
        )

        if elev_all.ndim < 2:
            elev_all = elev_all.unsqueeze(0)
        if ac_all.ndim < 2:
            ac_all = ac_all.unsqueeze(0)
        if areas.ndim < 2:
            areas = areas.unsqueeze(0)

        return (c_nn_norm, areas, elev_all, ac_all)

    # =========================================================================#

    # Helper Functions

    # =========================================================================#

    def _set_dtype(self) -> None:
        """Set the numpy and pytorch dtype for all model variables."""
        try:
            self.pt_dtype = eval(f"torch.{self._dtype}")
            self.np_dtype = eval(f"np.{self._dtype}")
        except Exception as e:
            raise ValueError(f"Could not parse dtype: {self._dtype}") from e

    def _bmi_array(self, arr: list[float]) -> NDArray:
        """Wrapper for standard array creation."""
        return np.array(arr, dtype=self.np_dtype)

    def _bmi_tensor(
        self,
        arr: Union[list[float], NDArray],
    ) -> torch.Tensor:
        """Wrapper for standard tensor creation."""
        return torch.as_tensor(arr, dtype=self.pt_dtype, device=self.device)

    def _set_empty_outputs(self) -> None:
        """Set output vars to 0 during warmup phase."""
        for name in self._output_vars:
            # Assuming output is 1D array of size [Catchments]
            self._output_vars[name]['value'] = np.zeros(
                self.n_units,
                dtype=self.np_dtype,
            )

    def _do_forward(
        self,
        data_dict: dict[str, torch.Tensor],
        batched: bool = True,
    ) -> dict[str, NDArray]:
        """Forward model on the pre-formatted dictionary.

        Parameters
        ----------
        data_dict
            Dictionary of input tensors for the model.

        Returns
        -------
        dict
            Dictionary of model outputs.
        """
        with torch.no_grad():
            prediction = self._model.dpl_model(data_dict, batched=batched)
            output = {
                'streamflow': prediction['streamflow'][-1].detach().cpu().numpy(),
            }

            # # # Load cached states if available
            # # self._model.load_states(self._lstm_states, self._phy_states)

            # # # Forward pass
            # # prediction = self._model.forward(data_dict, eval=True)

            # # # Cache states for next run
            # # self._lstm_states, self._phy_states = self._model.get_states()

            # # # Extract streamflow (last timestep after warmup)
            # # model_name = self.model_config['model']['phy']['name'][0]
            # output = {
            #     'streamflow': prediction[model_name]['streamflow'][-1]
            #     .detach()
            #     .cpu()
            #     .numpy(),
            # }
        return output

    def _load_model(self) -> ModelHandler:
        """Load a pre-trained model based on the configuration.

        Returns
        -------
        ModelHandler
            The loaded δMG model handler.
        """
        try:
            model = ModelHandler(
                self.model_config,
                device=self.device,
                verbose=self.verbose,
            )
            model.load_model(epoch=self.model_config['test']['test_epoch'])
            model.dpl_model.eval()

            # Enable state caching for stepwise inference (temporary)
            model.dpl_model.phy_model.cache_states = True
            model.dpl_model.nn_model.cache_states = True

            return model.to(dtype=self.pt_dtype, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}") from e

    def _format_outputs(self, outputs: dict[str, NDArray]) -> None:
        """Format model outputs as BMI outputs.

        Parameters
        ----------
        outputs
            Dictionary of model outputs.
        """
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

    def _load_norm_stats(self) -> None:
        """Load normalization statistics."""
        path = os.path.join(
            self.model_config['model_dir'],
            'normalization_statistics.json',
        )
        try:
            with open(os.path.abspath(path)) as f:
                self.norm_stats = json.load(f)
        except ValueError as e:
            raise ValueError("Normalization statistics not found.") from e

    def _to_external_units(self, name: str, values: list[float]) -> list[float]:
        """Convert internal model units to external units."""
        if name == 'atmosphere_water__liquid_equivalent_precipitation_rate':
            # # mm h-1 --> m3 s-1 (depth to volumetric rate)
            # area = self._static_var[map_to_external('catchment__area')]['value']
            # return [v * 1000 / 3600 * area for v in values]

            # mm h-1 --> m h-1
            return [v / 1000 for v in values]
        return values

    def initialize_config(
        self,
        config: Union[dict, dict],
    ) -> dict[str, NDArray]:
        """Parse and initialize configuration settings.

        Parameters
        ----------
        config
            Model configuration settings from Hydra.

        Returns
        -------
        dict
            Formatted configuration settings.
        """
        config['device'] = self.set_system_spec(config)

        # Convert date ranges to integer values.
        rho = config['model']['rho']

        sim_time = Dates(config['sim'], rho)
        config['sim_time'] = [sim_time.start_time, sim_time.end_time]

        if config.get('model_dir') is None:
            config['model_dir'] = ''
        config['plot_dir'] = ''
        config['sim_dir'] = ''
        config['log_dir'] = ''

        config['model']['phy']['nearzero'] = float(
            config['model']['phy']['nearzero'],
        )
        return config

    def set_system_spec(self, config: dict) -> torch.device:
        """Set the device and data type for the model on user's system.

        Parameters
        ----------
        config
            Model configuration settings from Hydra.

        Returns
        -------
        torch.device
            The device type for the model.
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

        return device

    @staticmethod
    def _set_value_internal(
        vars: list[tuple[str, str]],
        value: NDArray,
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
            var_dict[item[0]] = {'value': value.copy(), 'units': item[1]}
        return var_dict

    # =========================================================================#

    # BMI Helper Functions
    # See: https://github.com/csdms/bmi-python/blob/master/src/bmipy/bmi.py

    # =========================================================================#

    def get_component_name(self) -> str:
        """Name of the component."""
        return self._name

    def get_input_item_count(self) -> int:
        """Count of a model's input variables."""
        return len(self._dynamic_var)

    def get_output_item_count(self) -> int:
        """Count of a model's output variables."""
        return len(self._output_vars)

    def get_input_var_names(self) -> tuple[str, ...]:
        """Get the model's input variables."""
        return tuple(self._dynamic_var.keys())

    def get_output_var_names(self) -> tuple[str, ...]:
        """Get the model's output variables."""
        return tuple(self._output_vars.keys())

    def get_var_grid(self, name: str) -> int:
        """Get grid identifier for the given variable."""
        if name in {**self._dynamic_var, **self._output_vars}.keys():
            return self._var_grid_id
        else:
            raise KeyError(f"Variable '{name}' not supported.")

    def get_var_type(self, name: str) -> str:
        """Get data type of the given variable."""
        return self.get_value_ptr(name).dtype.name

    def get_var_units(self, name: str) -> str:
        """Get units of the given variable."""
        return {**self._dynamic_var, **self._output_vars}[name]['units']

    def get_var_itemsize(self, name: str) -> int:
        """Get memory use for each array element in bytes."""
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        """Get size, in bytes, of the given variable."""
        return self.get_var_itemsize(name) * len(self.get_value_ptr(name))

    def get_var_location(self, name: str) -> str:
        """Get the grid element type that the given variable is defined on."""
        if name in {**self._dynamic_var, **self._output_vars}.keys():
            return self._var_loc
        else:
            raise KeyError(f"Variable '{name}' not supported.")

    def get_current_time(self) -> float:
        """Return the current time of the model."""
        return self._timestep * self._time_step_size + self._start_time

    def get_start_time(self) -> float:
        """Start time of the model."""
        return self._start_time

    def get_end_time(self) -> float:
        """End time of the model."""
        return self._end_time

    def get_time_units(self) -> str:
        """Time units of the model."""
        return self._time_units

    def get_time_step(self) -> float:
        """Return the current time step of the model."""
        return self._time_step_size

    def get_value(self, name: str, dest: NDArray) -> NDArray:
        """Get a copy of values of the given variable."""
        tmp = self.get_value_ptr(name).flatten()
        dest[:] = self._to_external_units(name, tmp.tolist())
        return dest

    def get_value_ptr(self, name: str) -> NDArray:
        """Get a reference to values of the given variable."""
        return {**self._dynamic_var, **self._static_var, **self._output_vars}[name][
            'value'
        ]

    def get_value_at_indices(
        self,
        name: str,
        dest: NDArray,
        inds: list[int],
    ) -> NDArray:
        """Get values at indices.

        NOTE: ngen retrieves values via this method (twice):
        1. to the nexus for routing
        2. to write catchment-level output
        """
        tmp = self.get_value_ptr(name).take(inds)
        dest[:] = self._to_external_units(name, tmp.tolist())

        if (self._timestep > 24 * 365) and (self._timestep % 1000 == 0):
            log.debug(
                f"Time {self.get_current_time()} {self.get_time_units()} "
                f"(step {self._timestep}) | Runoff {tmp[-1]:.4f} mm d-1",
            )

        return dest

    def set_value(self, name: str, src: NDArray) -> None:
        """Specify a new value for a model variable.

        NOTE: ngen uses this for setting dynamic inputs.
        """
        if not isinstance(src, np.ndarray):
            src = np.array([src])
        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if name in dict.keys():
                dict[name]['value'] = np.expand_dims(
                    np.array(src),
                    axis=1,
                )  # [time, space]
                break

    def set_value_at_indices(
        self,
        name: str,
        inds: list[int],
        src: list[float],
    ) -> None:
        """Specify a new value for a model variable at particular indices."""
        if not isinstance(src, list):
            src = [src]

        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if name in dict.keys():
                for i in inds:
                    dict[name]['value'][i] = src[i]
                break

    def get_grid_rank(self, grid):
        """Get number of dimensions of the computational grid.

        NOTE: grid is always 0 for catchment/node-based models.
        """
        if grid == 0:
            return 1
        raise RuntimeError(f"Unsupported grid rank: {grid!s} | Only grid 0 is allowed.")

    def get_grid_size(self, grid):
        """Get the total number of elements in the computational grid."""
        if grid == 0:
            return 1
        raise RuntimeError(f"Unsupported grid size: {grid!s} | Only grid 0 is allowed.")

    def get_grid_type(self, grid):
        """Get the grid type as a string."""
        if grid == 0:
            return 'scalar'
        raise RuntimeError(f"Unsupported grid type: {grid!s} | Only grid 0 is allowed.")

    def get_grid_shape(self, grid, shape):
        """Get dimensions of the computational grid."""
        raise NotImplementedError('get_grid_shape')

    def get_grid_spacing(self, grid, spacing):
        """Get distance between nodes of the computational grid."""
        raise NotImplementedError('get_grid_spacing')

    def get_grid_origin(self, grid, origin):
        """Get coordinates for the lower-left corner of the computational grid."""
        raise NotImplementedError('get_grid_origin')

    def get_grid_x(self, grid, x):
        """Get coordinates of grid nodes in the x direction."""
        raise NotImplementedError('get_grid_x')

    def get_grid_y(self, grid, y):
        """Get coordinates of grid nodes in the y direction."""
        raise NotImplementedError('get_grid_y')

    def get_grid_z(self, grid, z):
        """Get coordinates of grid nodes in the z direction."""
        raise NotImplementedError('get_grid_z')

    def get_grid_node_count(self, grid):
        """Get the number of nodes in the grid."""
        raise NotImplementedError('get_grid_node_count')

    def get_grid_edge_count(self, grid):
        """Get the number of edges in the grid."""
        raise NotImplementedError('get_grid_edge_count')

    def get_grid_face_count(self, grid):
        """Get the number of faces in the grid."""
        raise NotImplementedError('get_grid_face_count')

    def get_grid_edge_nodes(self, grid, edge_nodes):
        """Get the edge-node connectivity."""
        raise NotImplementedError('get_grid_edge_nodes')

    def get_grid_face_edges(self, grid, face_edges):
        """Get the face-edge connectivity."""
        raise NotImplementedError('get_grid_face_edges')

    def get_grid_face_nodes(self, grid, face_nodes):
        """Get the face-node connectivity."""
        raise NotImplementedError('get_grid_face_nodes')

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        """Get the number of nodes for each face."""
        raise NotImplementedError('get_grid_nodes_per_face')
