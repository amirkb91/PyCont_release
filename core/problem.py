import yaml
from typing import Union, Dict, Any


class Problem:
    def __init__(self):
        self.parameters = None
        self.zero_function = None

        # Default parameters as instance variables
        self.defaults = {
            "continuation": {
                "method": "pseudo_arclength",
                "tangent_predictor": "nullspace_previous",
                "parameter": "period",
                "min_parameter_value": 0.1,
                "max_parameter_value": 30.0,
                "direction": 1,
                "num_points": 1500,
                "corrections_tolerance": 0.0001,
                "min_iterations": 1,
                "optimal_iterations": 2,
                "max_iterations": 4,
                "adaptive_step_start": 2,
                "initial_step_size": 0.005,
                "min_step_size": 0.001,
                "max_step_size": 1.0,
                "phase_condition_index": "",
            },
            "shooting": {
                "integration_tolerance": 1e-08,
                "steps_per_period": 300,
            },
            "forcing": {
                "amplitude": 0.1,
                "frequency": 0.5,
            },
            "starting_point": {
                "source": "function",
                "file_info": {
                    "file_name": "",
                    "restart_index": 0,
                    "recompute_tangent": False,
                },
            },
            "logger": {
                "enable_live_plot": True,
                "output_file_name": "Sol",
            },
        }

        # Validation schema
        self.schema = {
            "continuation": {
                "method": {"type": str, "choices": ["pseudo_arclength", "sequential"]},
                "tangent_predictor": {
                    "type": str,
                    "choices": ["nullspace_previous", "nullspace_pinned", "secant"],
                },
                "parameter": {"type": str, "choices": ["force_amp", "force_freq", "period"]},
                "min_parameter_value": {"type": (int, float)},
                "max_parameter_value": {"type": (int, float)},
                "direction": {"type": int, "choices": [1, -1]},
                "num_points": {"type": int, "min": 1},
                "corrections_tolerance": {"type": (int, float), "min": 0},
                "min_iterations": {"type": int, "min": 1},
                "optimal_iterations": {"type": int, "min": 1},
                "max_iterations": {"type": int, "min": 1},
                "adaptive_step_start": {"type": int, "min": 0},
                "initial_step_size": {"type": (int, float), "min": 0},
                "min_step_size": {"type": (int, float), "min": 0},
                "max_step_size": {"type": (int, float), "min": 0},
                "phase_condition_index": str,
            },
            "shooting": {
                "integration_tolerance": {"type": (int, float), "min": 0},
                "steps_per_period": {"type": int, "min": 1},
            },
            "forcing": {
                "amplitude": {"type": (int, float), "min": 0},
                "frequency": {"type": (int, float), "min": 0},
            },
            "starting_point": {
                "source": {"type": str, "choices": ["function", "file"]},
                "file_info": {
                    "file_name": str,
                    "restart_index": {"type": int, "min": 0},
                    "recompute_tangent": bool,
                },
            },
            "logger": {
                "enable_live_plot": bool,
                "output_file_name": str,
            },
        }

    def configure_parameters(self, cont_paramfile):
        with open(cont_paramfile) as f:
            data = yaml.safe_load(f)

        self._convert_strings_to_floats(data)

        self.parameters = self.fill_defaults(data, self.defaults)
        self.validate_parameters()

    def _convert_strings_to_floats(self, data: Union[Dict, list]):
        """
        Recursively traverse a dictionary or list and convert string values to floats.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        pass  # Ignore conversion errors
                elif isinstance(value, (dict, list)):
                    self._convert_strings_to_floats(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    try:
                        data[i] = float(item)
                    except (ValueError, TypeError):
                        pass  # Ignore conversion errors
                elif isinstance(item, (dict, list)):
                    self._convert_strings_to_floats(item)

    def validate_parameters(self):
        """
        Validate continuation parameters using schema-based validation
        """
        self._validate_dict(self.parameters, self.schema, "")
        self._validate_logical_rules()

    def _validate_dict(self, data: Dict, schema: Dict, path: str):
        """
        Recursively validate a dictionary against a schema
        """
        for key, expected in schema.items():
            current_path = f"{path}.{key}" if path else key

            if key not in data:
                continue  # Missing keys are handled by fill_defaults

            value = data[key]

            if isinstance(expected, dict) and not self._is_validation_spec(expected):
                # Nested dictionary
                self._validate_dict(value, expected, current_path)
            else:
                # Validate single parameter
                self._validate_parameter(value, expected, current_path)

    def _is_validation_spec(self, spec: Dict) -> bool:
        """
        Check if a dictionary is a validation specification or nested structure
        """
        return "type" in spec or isinstance(spec, type)

    def _validate_parameter(self, value: Any, spec: Union[type, Dict], path: str):
        """
        Validate a single parameter against its specification
        """
        if isinstance(spec, type):
            # Simple type validation
            if not isinstance(value, spec):
                raise TypeError(
                    f"Parameter '{path}' must be of type {spec.__name__}, got {type(value).__name__}: {value}"
                )
        elif isinstance(spec, dict):
            # Complex validation specification
            expected_type = spec.get("type")

            # Type validation
            if expected_type:
                if not isinstance(value, expected_type):
                    type_names = (
                        expected_type.__name__
                        if hasattr(expected_type, "__name__")
                        else str(expected_type)
                    )
                    raise TypeError(
                        f"Parameter '{path}' must be of type {type_names}, got {type(value).__name__}: {value}"
                    )

            # Choice validation
            if "choices" in spec and value not in spec["choices"]:
                raise ValueError(
                    f"Parameter '{path}' must be one of {spec['choices']}, got: {value}"
                )

            # Range validation
            if "min" in spec and value < spec["min"]:
                raise ValueError(f"Parameter '{path}' must be >= {spec['min']}, got: {value}")

            if "max" in spec and value > spec["max"]:
                raise ValueError(f"Parameter '{path}' must be <= {spec['max']}, got: {value}")

    def _validate_logical_rules(self):
        """
        Validate logical consistency rules that depend on multiple parameters
        """
        cont = self.parameters["continuation"]

        # Rule 1: Iteration consistency
        if cont["min_iterations"] >= cont["max_iterations"]:
            raise ValueError(
                f"continuation.min_iterations ({cont['min_iterations']}) must be less than max_iterations ({cont['max_iterations']})"
            )

        if cont["optimal_iterations"] > cont["max_iterations"]:
            raise ValueError(
                f"continuation.optimal_iterations ({cont['optimal_iterations']}) must be <= max_iterations ({cont['max_iterations']})"
            )

        # Rule 2: Step size consistency
        if cont["min_step_size"] >= cont["max_step_size"]:
            raise ValueError(
                f"continuation.min_step_size ({cont['min_step_size']}) must be less than max_step_size ({cont['max_step_size']})"
            )

        if not (cont["min_step_size"] <= cont["initial_step_size"] <= cont["max_step_size"]):
            raise ValueError(
                f"continuation.initial_step_size ({cont['initial_step_size']}) must be between min_step_size ({cont['min_step_size']}) and max_step_size ({cont['max_step_size']})"
            )

        # Rule 3: ContPar bounds consistency
        if cont["min_parameter_value"] >= cont["max_parameter_value"]:
            raise ValueError(
                f"continuation.min_parameter_value ({cont['min_parameter_value']}) must be less than max_parameter_value ({cont['max_parameter_value']})"
            )

        # Rule 4: Phase condition consistency for forced systems
        if cont["parameter"] in ["force_freq", "force_amp"] and cont["phase_condition_index"] != "":
            raise ValueError(
                f"continuation.phase_condition_index must be empty string for forced continuation (parameter='{cont['parameter']}'), got: '{cont['phase_condition_index']}'"
            )

    def fill_defaults(self, data: Dict, defaults: Dict) -> Dict:
        """
        Fill missing parameters with default values
        """
        result = data.copy()
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self.fill_defaults(result[key], value)
        return result

    def set_zero_function(self, fxn):
        self.zero_function = fxn
