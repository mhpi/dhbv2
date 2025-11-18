"""BMI wrapper for interfacing dHBV 2.0 with NOAA-OWP NextGen framework.

Author: Leo Lonzarich
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import torch
import yaml
from bmipy import Bmi
from dmg.core.utils.factory import import_data_sampler
from dmg.core.utils.dates import Dates

from dmg import ModelHandler
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------- #
# Dynamic input variables (CSDMS standard names)
# -------------------------------------------- #
_dynamic_input_vars = [
    ("atmosphere_water__liquid_equivalent_precipitation_rate", "mm d-1"),
    ("land_surface_air__temperature", "degC"),
    ("land_surface_water__potential_evaporation_volume_flux", "mm d-1"),
]

# ------------------------------------------- #
# Static input variables (CSDMS standard names)
# ------------------------------------------- #
_static_input_vars = [
    ("ratio__mean_potential_evapotranspiration__mean_precipitation", "-"),
    ("atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate", "mm d-1"),
    ("land_surface_water__Hargreaves_potential_evaporation_volume_flux", "mm d-1"),
    ("land_vegetation__normalized_diff_vegetation_index", "-"),
    ("free_land_surface_water", "mm d-1"),
    ("basin__mean_of_slope", "m km-1"),
    ("soil_sand__grid", "km2"),
    ("soil_clay__grid", "km2"),
    ("soil_silt__grid", "km2"),
    ("land_surface_water__glacier_fraction", "percent"),
    ("soil_clay__attr", "percent"),
    ("soil_gravel__attr", "percent"),
    ("soil_sand__attr", "percent"),
    ("soil_silt__attr", "percent"),
    ("basin__mean_of_elevation", "m"),
    ("atmosphere_water__daily_mean_of_temperature", "degC"),
    ("land_surface_water__permafrost_fraction", "-"),
    ("bedrock__permeability", "m2"),
    ("p_seasonality", "-"),
    ("land_surface_water__potential_evaporation_volume_flux_seasonality", "-"),
    ("land_surface_water__snow_fraction", "percent"),
    ("atmosphere_water__precipitation_falling_as_snow_fraction", "percent"),
    ("soil_clay__volume_fraction", "percent"),
    ("soil_gravel__volume_fraction", "percent"),
    ("soil_sand__volume_fraction", "percent"),
    ("soil_silt__volume_fraction", "percent"),
    ("soil_active-layer__porosity", "-"),
    ("basin__area", "km2"),
]

# ------------------------------------- #
# Output variables (CSDMS standard names)
# ------------------------------------- #
_output_vars = [
    ("land_surface_water__runoff_volume_flux", "m3 s-1"),
]

# ---------------------------------------------- #
# Internal variable names <-> CSDMS standard names
# ---------------------------------------------- #
_var_name_internal_map = {
    # ----------- Dynamic inputs -----------
    "P": "atmosphere_water__liquid_equivalent_precipitation_rate",
    "Temp": "land_surface_air__temperature",
    "PET": "land_surface_water__potential_evaporation_volume_flux",
    # ----------- Static inputs -----------
    "aridity": "ratio__mean_potential_evapotranspiration__mean_precipitation",
    "meanP": "atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate",
    "ETPOT_Hargr": "land_surface_water__Hargreaves_potential_evaporation_volume_flux",
    "NDVI": "land_vegetation__normalized_diff_vegetation_index",
    "FW": "free_land_surface_water",
    "meanslope": "basin__mean_of_slope",
    "SoilGrids1km_sand": "soil_sand__grid",
    "SoilGrids1km_clay": "soil_clay__grid",
    "SoilGrids1km_silt": "soil_silt__grid",
    "glaciers": "land_surface_water__glacier_fraction",
    "HWSD_clay": "soil_clay__attr",
    "HWSD_gravel": "soil_gravel__attr",
    "HWSD_sand": "soil_sand__attr",
    "HWSD_silt": "soil_silt__attr",
    "meanelevation": "basin__mean_of_elevation",
    "meanTa": "atmosphere_water__daily_mean_of_temperature",
    "permafrost": "land_surface_water__permafrost_fraction",
    "permeability": "bedrock__permeability",
    "seasonality_P": "p_seasonality",
    "seasonality_PET": "land_surface_water__potential_evaporation_volume_flux_seasonality",
    "snow_fraction": "land_surface_water__snow_fraction",
    "snowfall_fraction": "atmosphere_water__precipitation_falling_as_snow_fraction",
    "T_clay": "soil_clay__volume_fraction",
    "T_gravel": "soil_gravel__volume_fraction",
    "T_sand": "soil_sand__volume_fraction",
    "T_silt": "soil_silt__volume_fraction",
    "Porosity": "soil_active-layer__porosity",
    "uparea": "basin__area",
    # ----------- Outputs -----------
    "streamflow": "land_surface_water__runoff_volume_flux",
}

_var_name_external_map = {v: k for k, v in _var_name_internal_map.items()}


def map_to_external(name: str):
    """Return the external name (exposed via BMI) for a given internal name."""
    return _var_name_internal_map[name]


def map_to_internal(name: str):
    """Return the internal name for a given external name (exposed via BMI)."""
    return _var_name_external_map[name]


def bmi_array(arr: list[float]) -> NDArray:
    """Trivial wrapper function to ensure the expected numpy array datatype is used."""
    return np.array(arr, dtype='float64')


# =============================================================================#
# =============================================================================#
# =============================================================================#


# MAIN BMI >>>>


# =============================================================================#
# =============================================================================#
# =============================================================================#


class DeltaModelBmi(Bmi):
    """
    dHBV 2.0 BMI: NextGen-compatible, differentiable, physics-informed ML
    model for hydrologic forecasting (Song et al., 2024).

    Note: This dHBV 2.0 BMI can only run forward inference. See the dMG package
        (https://github.com/mhpi/generic_deltamodel) for training.
    """

    _att_map = {
        'model_name': 'Differentiable Model',
        'version': '1.0',
        'author_name': 'Leo Lonzarich',
        'time_step_size': 86400,
        'time_units': 'seconds',
        # 'time_step_type':     '',
        # 'grid_type':          'scalar',
        # 'step_method':        '',
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        verbose=False,
    ) -> None:
        """Create a BMI model ready for initialization.

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        verbose
            Enables debug print statements if True.
        """
        super().__init__()
        self._name = self._att_map['model_name']
        self._model = None
        self._initialized = False
        self.verbose = verbose

        self._var_loc = 'node'
        self._var_grid_id = 0

        self._start_time = 0.0
        self._end_time = np.finfo('d').max
        self._time_units = 's'
        self._timestep = 0

        self.config_bmi = None
        self.config_model = None

        # Model states:
        self.hn = None  # LSTM hidden state
        self.cn = None  # LSTM cell state
        self.pstate = None  # Physical model states (e.g., buckets)

        # Timing BMI computations
        t_start = time.time()
        self.bmi_process_time = 0

        # Read BMI and model configuration files.
        if config_path is not None:
            if not Path(config_path).is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path) as f:
                self.config_bmi = yaml.safe_load(f)
            self.stepwise = self.config_bmi.get('stepwise', True)

            try:
                # model_config_path = os.path.join(
                #     script_dir,
                #     "..",
                #     "..",
                #     self.config_bmi.get('config_model'),
                # )
                model_config_path = os.path.join(
                    script_dir,
                    '..',
                    '..',
                    'ngen_resources/data/dhbv2/',
                    self.config_bmi.get('config_model'),
                )
                with open(model_config_path) as f:
                    self.config_model = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model configuration: {e}") from e

        # Initialize variables.
        self._dynamic_var = self._set_vars(_dynamic_input_vars, bmi_array([]))
        self._static_var = self._set_vars(_static_input_vars, bmi_array([]))
        self._output_vars = self._set_vars(_output_vars, bmi_array([]))

        # Track total BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(f"BMI init took {time.time() - t_start} s")

        self.x_list = []
        self.xc_nn_list = []
        self.c_nn_list = []

    @staticmethod
    def _set_vars(
        vars: list[tuple[str, str]],
        var_value: NDArray,
    ) -> dict[str, dict[str, Union[NDArray, str]]]:
        """Set the values of the given variables."""
        var_dict = {}
        for item in vars:
            var_dict[item[0]] = {'value': var_value.copy(), 'units': item[1]}
        return var_dict

    def initialize(self, config_path: Optional[str] = None) -> None:
        """(Control function) Initialize the BMI model.

        This BMI operates in two modes:
            (Necessesitated by the fact that dhBV 2.0's internal NN must forward
            on all data at once. <-- Forwarding on each timestep one-by-one with
            saving/loading hidden states would slash LSTM performance. However,
            feeding in hidden states day-by-day leeds to great efficiency losses
            vs simply feeding all data at once due to carrying gradients at each
            step.)

            1) Feed all input data to BMI before
                'bmi.initialize()'. Then internal model is forwarded on all data
                and generates predictions during '.initialize()'.

            2) Run '.initialize()', then pass data day by day as normal during
                'bmi.update()'. If forwarding period is sufficiently small (say,
                <100 days), then forwarding LSTM on individual days with saved
                states is reasonable.

        To this end, a configuration file can be specified either during
        `bmi.__init__()`, or during `.initialize()`. If running BMI as type (1),
        config must be passed in the former, otherwise passed in the latter for (2).

        Parameters
        ----------
        config_path
            Path to the BMI configuration file.
        """
        t_start = time.time()

        # Read BMI configuration file if provided.
        if config_path is not None:
            if not Path(config_path).is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path) as f:
                self.config_bmi = yaml.safe_load(f)
            self.stepwise = self.config_bmi.get('stepwise', True)

        if self.config_bmi is None:
            raise ValueError(
                "No configuration file given. A config path"
                "must be passed at time of bmi init() or"
                "initialize() call.",
            )

        # Load model configuration.
        if self.config_model is None:
            try:
                model_config_path = os.path.join(
                    script_dir,
                    '..',
                    '..',
                    'ngen_resources/data/dhbv2/',
                    self.config_bmi.get('config_model'),
                )
                with open(model_config_path) as f:
                    self.config_model = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model configuration: {e}") from e

        self.config_model = self.initialize_config(self.config_model)
        self.config_model['model_path'] = os.path.join(
            script_dir,
            '..',
            '..',
            'ngen_resources/data/dhbv2/',
            self.config_model.get('model_dir'),
        )
        self._name = self.config_model['model']['phy']['name'][0]  # Overwrite
        self.device = self.config_model['device']
        self.internal_dtype = self.config_model['dtype']
        self.external_dtype = eval(self.config_bmi['dtype'])
        self.sampler = import_data_sampler(self.config_model['data_sampler'])(
            self.config_model,
        )

        # Load static variables from BMI conf
        for name in self._static_var.keys():
            ext_name = map_to_internal(name)
            if ext_name in self.config_bmi.keys():
                self._static_var[name]['value'] = bmi_array(self.config_bmi[ext_name])
            else:
                log.warning(f"Static variable '{name}' not in BMI config. Skipping.")

        # Set simulation parameters.
        self.current_time = self.config_bmi.get('start_time', 0.0)
        # self._time_step_size = self.config_bmi.get('time_step_size', 86400)  # Default to 1 day in seconds.
        # self._end_time = self.config_bmi.get('end_time', np.finfo('d').max)\

        # Load a trained model.
        try:
            self._model = self._load_trained_model(self.config_model).to(self.device)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}") from e

        # Forward simulation on all data in one go.
        if not self.stepwise:
            predictions = self._do_forward()
            self._format_outputs(predictions)  # Process and store predictions.

        # Track BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Initialize took {time.time() - t_start:.4f} s | Total runtime: {self.bmi_process_time:.4f} s",
            )

    def update(self) -> None:
        """(Control function) Advance model state by one time step."""
        t_start = time.time()

        # Forward model on individual timesteps if not initialized with forward_init.
        if self.stepwise:
            data_dict = self._format_inputs()
            predictions = self._do_forward(data_dict)
            self._format_outputs(predictions)

        # Increment model time.
        self._timestep += 1

        # Track BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Update took {time.time() - t_start:.4f} s | Total runtime: {self.bmi_process_time:.4f} s",
            )

    def update_until(self, end_time: float) -> None:
        """(Control function) Update model until a particular time.

        Note: Models should be trained standalone with dmg first before
        forward predictions with this BMI.

        Parameters
        ----------
        end_time : float
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

        # Track BMI runtime.
        self.bmi_process_time += time.time() - t_start
        if self.verbose:
            log.info(
                f"BMI Update Until took {time.time() - t_start:.4f} s | Total runtime: {self.bmi_process_time:.4f} s",
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
    # =========================================================================#

    # Helper functions for BMI

    # =========================================================================#
    # =========================================================================#

    def _do_forward(self, data_dict: dict[str, Any]):
        """Forward model and save outputs to return on update call."""
        if data_dict == {}:
            log.error("No data to forward. Check input variables.")
            return

        n_samples = data_dict['xc_nn_norm'].shape[1]
        batch_start = np.arange(
            0,
            n_samples,
            self.config_model['sim']['batch_size'],
        )
        batch_end = np.append(batch_start[1:], n_samples)

        batch_predictions = []
        # Forward through basins in batches.
        with torch.no_grad():
            for i in range(len(batch_start)):
                dataset_sample = self.sampler.get_validation_sample(
                    data_dict,
                    batch_start[i],
                    batch_end[i],
                )

                # 1. Load model states
                self._model.load_states((self.hn, self.cn), self.pstate)

                # 2. Forward model
                self.prediction = self._model.forward(dataset_sample, eval=True)
                print(dataset_sample['x_phy'])
                print(self.prediction)
                # 3. Cache model states
                (self.hn, self.cn), self.pstate = self._model.get_states()
                print(self.pstate)

                prediction = {
                    key: tensor.cpu().detach()
                    for key, tensor in self.prediction[self._name].items()
                }
                batch_predictions.append(prediction)

                self.x_list.append(dataset_sample['x_phy'])
                self.xc_nn_list.append(dataset_sample['xc_nn_norm'])
                self.c_nn_list.append(dataset_sample['c_nn_norm'])

        # Combine batches
        # x_combined = torch.cat(self.x_list, dim=0)
        # xc_nn_combined = torch.cat(self.xc_nn_list, dim=0)
        # c_nn_combined = torch.cat(self.c_nn_list, dim=0)

        # data_dict['x_phy'] = x_combined.numpy()
        # data_dict['xc_nn_norm'] = xc_nn_combined.numpy()
        # data_dict['c_nn_norm'] = c_nn_combined.numpy()

        # self.prediction = self._model.forward(dataset_sample, eval=True)
        # print(self.prediction)

        return self._batch_data(batch_predictions)

    @staticmethod
    def _load_trained_model(config: dict):
        """Load a pre-trained model based on the configuration."""
        model_path = config.get('model_path')
        if not model_path:
            raise ValueError("No model path specified in configuration.")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return ModelHandler(config, verbose=True)

    # def update_frac(self, time_frac: float) -> None:
    #     """
    #     Update model by a fraction of a time step.

    #     Parameters
    #     ----------
    #     time_frac : float
    #         Fraction fo a time step.
    #     """
    #     if self.verbose:
    #         print("Warning: This model is trained to make predictions on one day timesteps.")
    #     time_step = self.get_time_step()
    #     self._time_step_size = self._time_step_size * time_frac
    #     self.update()
    #     self._time_step_size = time_step

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

            if self.stepwise:
                self._output_vars[name]['value'] = np.append(
                    self._output_vars[name]['value'],
                    output_val,
                )
            else:
                self._output_vars[name]['value'] = output_val

    def _format_inputs(self):
        """Format dynamic and static inputs for the model."""
        # =====================================================================#
        x_list = []
        c_list = []

        for name, data in self._dynamic_var.items():
            if data['value'].ndim == 0:
                data['value'] = np.expand_dims(
                    data['value'],
                    axis=(0, 1),
                )  # shape (1,1)
            if data['value'].ndim == 1:
                data['value'] = np.expand_dims(
                    data['value'],
                    axis=(1, 2),
                )  # Shape: (n, 1, 1) # TODO: Fix dims
            elif data['value'].ndim == 2:
                data['value'] = np.expand_dims(
                    data['value'],
                    axis=2,
                )  # Shape: (n, m, 1) # TODO: Fix dims
            elif data['value'].ndim != 3:
                raise ValueError(
                    f"Dynamic variable '{name}' has unsupported "
                    f"dimensions ({data['value'].ndim}).",
                )
            x_list.append(data['value'])

        for name, data in self._static_var.items():
            if data['value'].size == 0:
                raise ValueError(f"Static variable '{name}' has no value.")
            if data['value'].ndim != 2:
                data['value'] = np.expand_dims(data['value'], axis=(0, 1))
            c_list.append(data['value'])

        x = np.concatenate(x_list, axis=2)  # Shape [nt, nb, nx]
        x = self._fill_nan(x)
        c = np.concatenate(c_list, axis=1)  # Shape [nb, nx_static]

        xc_nn_norm, c_nn_norm = self.normalize(x.copy(), c)

        # Get upstream area and elevation
        try:
            ac_name = self.config_model['observations']['upstream_area_name']
            ac_array = self._static_var[map_to_external(ac_name)]['value']
        except ValueError as e:
            raise ValueError(
                "Upstream area is not provided. This is needed for high-resolution streamflow model.",
            ) from e
        try:
            elevation_name = self.config_model['observations']['elevation_name']
            elev_array = self._static_var[map_to_external(elevation_name)]['value']
        except ValueError as e:
            raise ValueError(
                "Elevation is not provided. This is needed for high-resolution streamflow model.",
            ) from e

        dataset = {
            'ac_all': ac_array.squeeze(-1),
            'elev_all': elev_array.squeeze(-1),
            'c_nn': c,
            'xc_nn_norm': xc_nn_norm,
            'c_nn_norm': c_nn_norm,
            'x_phy': x,
        }
        return dataset
        # =====================================================================#

    def _time_conversion(self):
        """Convert time units if necessary."""
        pass

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize data for neural network."""
        self.load_norm_stats()
        x_nn_norm = self._to_norm(x_nn, _dynamic_input_vars)
        c_nn_norm = self._to_norm(c_nn, _static_input_vars)

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        c_nn_norm_repeat = np.repeat(
            np.expand_dims(c_nn_norm, 0),
            x_nn_norm.shape[0],
            axis=0,
        )

        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm_repeat), axis=2)
        del x_nn_norm, x_nn

        return xc_nn_norm, c_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """Standard data normalization."""
        log_norm_vars = self.config_model['model']['phy']['use_log_norm']

        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[map_to_internal(var[0])]

            if len(data.shape) == 3:
                if map_to_internal(var[0]) in log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var[0] in log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")
        return data_norm

    def load_norm_stats(self) -> None:
        """Load normalization statistics."""
        path = os.path.join(
            self.config_model['model_path'],
            '..',
            'normalization_statistics.json',
        )
        try:
            with open(os.path.abspath(path)) as f:
                self.norm_stats = json.load(f)
        except ValueError as e:
            raise ValueError("Normalization statistics not found.") from e

    def _process_predictions(self, predictions):
        """Process model predictions and store them in output variables."""
        for var_name, prediction in predictions.items():
            if var_name in self._output_vars:
                self._output_vars[var_name]['value'] = prediction.cpu().numpy()
            else:
                log.warning(f"Output variable '{var_name}' not recognized. Skipping.")

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> list[dict[str, np.ndarray]]:
        """Merge list of batch data dictionaries into a single dictionary."""
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    dim = 1
                else:
                    dim = 0
                data[key] = (
                    torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
                )
            return data
        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    @staticmethod
    def _fill_nan(array_3d):
        # Define the x-axis for interpolation
        x = np.arange(array_3d.shape[1])

        # Iterate over the first and third dimensions to interpolate the second dimension
        for i in range(array_3d.shape[0]):
            for j in range(array_3d.shape[2]):
                # Select the 1D slice for interpolation
                slice_1d = array_3d[i, :, j]

                # Find indices of NaNs and non-NaNs
                nans = np.isnan(slice_1d)
                non_nans = ~nans

                # Only interpolate if there are NaNs and at least two non-NaN values for reference
                if np.any(nans) and np.sum(non_nans) > 1:
                    # Perform linear interpolation using numpy.interp
                    array_3d[i, :, j] = np.interp(
                        x,
                        x[non_nans],
                        slice_1d[non_nans],
                        left=None,
                        right=None,
                    )
        return array_3d

    def array_to_tensor(self) -> None:
        """Converts input values into Torch tensor object to be read by model."""
        raise NotImplementedError('array_to_tensor')

    def tensor_to_array(self) -> None:
        """
        Converts model output Torch tensor into date + gradient arrays to be
        passed out of BMI for backpropagation, loss, optimizer tuning.
        """
        raise NotImplementedError('tensor_to_array')

    def get_tensor_slice(self):
        """Get tensor of input data for a single timestep."""
        # sample_dict = take_sample_test(self.bmi_config, self.dataset_dict)
        # self.input_tensor = torch.Tensor()

        raise NotImplementedError('get_tensor_slice')

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
        # Combine input/output variable dicts: NOTE: should add to init.
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
        ]["value"]

    def get_value(self, var_name: str, dest: NDArray):
        """Return copy of variable values."""
        # TODO: will need to properly account for multiple basins.
        try:
            dest[:] = self.get_value_ptr(var_name)[self._timestep - 1,].flatten()
        except RuntimeError as e:
            raise e
        return dest

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at indices."""
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    def set_value(self, var_name, values: np.ndarray):
        """Set variable value."""
        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if var_name in dict.keys():
                if self.stepwise:
                    dict[var_name]['value'] = values
                else:
                    dict[var_name]['value'] = np.append(
                        dict[var_name]['value'],
                        values,
                    )
                break

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices."""
        if not isinstance(src, list):
            src = [src]

        for dict in [self._dynamic_var, self._static_var, self._output_vars]:
            if name in dict.keys():
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
        # var_name = self._grids[grid_id][0]
        # shape[:] = self.get_value_ptr(var_name).shape
        # return shape
        raise NotImplementedError('get_grid_shape')

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        # spacing[:] = self._model.spacing
        # return spacing
        raise NotImplementedError('get_grid_spacing')

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        # origin[:] = self._model.origin
        # return origin
        raise NotImplementedError('get_grid_origin')

    def get_grid_type(self, grid_id):
        """Type of grid."""
        if grid_id == 0:
            return 'scalar'
        raise RuntimeError(f"unsupported grid type: {grid_id!s}. only support 0")

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        """Current time of model."""
        return self._timestep * self._att_map['time_step_size'] + self._start_time

    def get_time_step(self):
        """Time step size of model."""
        return self._att_map['time_step_size']

    def get_time_units(self):
        """Time units of model."""
        return self._att_map['time_units']

    def get_grid_edge_count(self, grid):
        """Get grid edge count."""
        raise NotImplementedError('get_grid_edge_count')

    def get_grid_edge_nodes(self, grid, edge_nodes):
        """Get grid edge nodes."""
        raise NotImplementedError('get_grid_edge_nodes')

    def get_grid_face_count(self, grid):
        """Get grid face count."""
        raise NotImplementedError('get_grid_face_count')

    def get_grid_face_nodes(self, grid, face_nodes):
        """Get grid face nodes."""
        raise NotImplementedError('get_grid_face_nodes')

    def get_grid_node_count(self, grid):
        """Get grid node count."""
        raise NotImplementedError('get_grid_node_count')

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        """Get grid nodes per face."""
        raise NotImplementedError('get_grid_nodes_per_face')

    def get_grid_face_edges(self, grid, face_edges):
        """Get grid face edges."""
        raise NotImplementedError('get_grid_face_edges')

    def get_grid_x(self, grid, x):
        """Get grid x-coordinates."""
        raise NotImplementedError('get_grid_x')

    def get_grid_y(self, grid, y):
        """Get grid y-coordinates."""
        raise NotImplementedError('get_grid_y')

    def get_grid_z(self, grid, z):
        """Get grid z-coordinates."""
        raise NotImplementedError('get_grid_z')

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
        train_time = Dates(config['train'], config['model']['rho'])
        test_time = Dates(config['test'], config['model']['rho'])
        sim_time = Dates(config['sim'], config['model']['rho'])
        all_time = Dates(config['observations'], config['model']['rho'])

        exp_time_start = min(
            train_time.start_time,
            train_time.end_time,
            test_time.start_time,
            test_time.end_time,
        )
        exp_time_end = max(
            train_time.start_time,
            train_time.end_time,
            test_time.start_time,
            test_time.end_time,
        )

        config['train_time'] = [train_time.start_time, train_time.end_time]
        config['test_time'] = [test_time.start_time, test_time.end_time]
        config['sim_time'] = [sim_time.start_time, sim_time.end_time]
        config['experiment_time'] = [exp_time_start, exp_time_end]
        config['all_time'] = [all_time.start_time, all_time.end_time]

        if config.get('model_dir') is None:
            config['model_dir'] = ''
        config['plot_dir'] = ''
        config['sim_dir'] = ''
        config['log_dir'] = ''

        # Convert string back to data type.
        config['dtype'] = eval(config['dtype'])
        config['model']['phy']['nearzero'] = float(config['model']['phy']['nearzero'])

        # Raytune
        config['do_tune'] = config.get('do_tune', False)

        # Set batch size
        if self.stepwise:
            config['sim']['batch_size'] = 1

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
        if config["device"] == 'cpu':
            device = torch.device('cpu')
        elif config['device'] == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                raise ValueError("MPS is not available on this system.")
        elif config["device"] == 'cuda':
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

    # def scale_output(self) -> None:
    #     """
    #     Scale and return more meaningful output from wrapped model.
    #     """
    #     models = self.config['hydro_models'][0]

    #     # TODO: still have to finish finding and undoing scaling applied before
    #     # model run. (See some checks used in bmi_lstm.py.)

    #     # Strip unnecessary time and variable dims. This gives 1D array of flow
    #     # at each basin.
    #     # TODO: setup properly for multiple models later.
    #     self.streamflow_cms = self.preds[models]['flow_sim'].squeeze()
