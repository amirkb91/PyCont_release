import os
import json
from typing import Union, List, Dict, Any


class Prob:
    def __init__(self):
        self.cont_params = None
        self.doffunction = None
        self.icfunction = None
        self.zerofunction = None
        self.zerofunction_firstpoint = None
        self.partitionfunction = None

        # Default parameters as instance variables
        self.defaults = {
            "continuation": {
                "forced": False,
                "method": "psa",
                "tangent": "peeters",
                "continuation_parameter": "frequency",
                "dir": 1,
                "npts": 500,
                "tol": 0.001,
                "itermin": 1,
                "iteropt": 3,
                "itermax": 5,
                "iterjac": 1,
                "nadapt": 2,
                "s0": 1e-3,
                "smin": 1e-6,
                "smax": 1e-1,
                "betacontrol": False,
                "betamax": 20,
                "ContParMin": 20,
                "ContParMax": 100,
                "phase_index_unforced": "allvel",
            },
            "shooting": {
                "method": "single",
                "scaling": False,
                "rel_tol": 1e-08,
                "single": {"nperiod": 1, "nsteps_per_period": 200},
                "multiple": {"npartition": 3, "nsteps_per_partition": 100},
            },
            "forcing": {
                "amplitude": 1,
                "frequency": 1,
                "phase_ratio": 0.5,
                "tau0": 1e-4,
                "tau1": 1e-4,
                "rho_GA": 0.95,
            },
            "first_point": {
                "from_eig": True,
                "itermax": 30,
                "eig_start": {"NNM": 1, "scale": 0.01},
                "restart": {
                    "file_name": "",
                    "index": 50,
                    "recompute_tangent": False,
                    "fixF": False,
                    "F": 60,
                },
            },
            "Logger": {"plot": True, "file_name": "temp"},
        }

        # Validation schema
        self.schema = {
            "continuation": {
                "forced": bool,
                "method": {"type": str, "choices": ["psa", "seq"]},
                "tangent": {"type": str, "choices": ["peeters", "keller", "secant"]},
                "continuation_parameter": {"type": str, "choices": ["frequency", "amplitude"]},
                "dir": {"type": int, "choices": [1, -1]},
                "npts": {"type": int, "min": 1},
                "tol": {"type": (int, float), "min": 0},
                "itermin": {"type": int, "min": 1},
                "iteropt": {"type": int, "min": 1},
                "itermax": {"type": int, "min": 1},
                "iterjac": {"type": int, "min": 1},
                "nadapt": {"type": int, "min": 1},
                "s0": {"type": (int, float), "min": 0},
                "smin": {"type": (int, float), "min": 0},
                "smax": {"type": (int, float), "min": 0},
                "betacontrol": bool,
                "betamax": {"type": (int, float), "min": 0},
                "ContParMin": {"type": (int, float)},
                "ContParMax": {"type": (int, float)},
                "phase_index_unforced": str,
            },
            "shooting": {
                "method": {"type": str, "choices": ["single", "multiple"]},
                "scaling": bool,
                "rel_tol": {"type": (int, float), "min": 0},
                "single": {
                    "nperiod": {"type": int, "min": 1},
                    "nsteps_per_period": {"type": int, "min": 1},
                },
                "multiple": {
                    "npartition": {"type": int, "min": 1},
                    "nsteps_per_partition": {"type": int, "min": 1},
                },
            },
            "forcing": {
                "amplitude": {"type": (int, float), "min": 0},
                "frequency": {"type": (int, float), "min": 0},
                "phase_ratio": {"type": (int, float)},
                "tau0": {"type": (int, float), "min": 0},
                "tau1": {"type": (int, float), "min": 0},
                "rho_GA": {"type": (int, float), "min": 0, "max": 1},
            },
            "first_point": {
                "from_eig": bool,
                "itermax": {"type": int, "min": 1},
                "eig_start": {
                    "NNM": {"type": int, "min": 1},
                    "scale": {"type": (int, float), "min": 0},
                },
                "restart": {
                    "file_name": str,
                    "index": {"type": int, "min": 0},
                    "recompute_tangent": bool,
                    "fixF": bool,
                    "F": {"type": (int, float)},
                },
            },
            "Logger": {
                "plot": bool,
                "file_name": str,
            },
        }

    def read_contparams(self, cont_paramfile):
        with open(cont_paramfile) as f:
            data = json.load(f)

        self.cont_params = self.fill_defaults(data, self.defaults)
        self.validate_parameters()

    def validate_parameters(self):
        """
        Validate continuation parameters using schema-based validation
        """
        self._validate_dict(self.cont_params, self.schema, "")
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
        cont = self.cont_params["continuation"]

        # Rule 1: If forced is False, continuation_parameter must be "frequency"
        if not cont["forced"] and cont["continuation_parameter"] != "frequency":
            raise ValueError(
                "When continuation is not forced (forced=False), "
                f"continuation_parameter must be 'frequency', not '{cont['continuation_parameter']}'"
            )

        # Rule 2: Iteration consistency
        if cont["itermin"] >= cont["itermax"]:
            raise ValueError(
                f"continuation.itermin ({cont['itermin']}) must be less than itermax ({cont['itermax']})"
            )

        if cont["iteropt"] > cont["itermax"]:
            raise ValueError(
                f"continuation.iteropt ({cont['iteropt']}) must be <= itermax ({cont['itermax']})"
            )

        # Rule 3: Step size consistency
        if cont["smin"] >= cont["smax"]:
            raise ValueError(
                f"continuation.smin ({cont['smin']}) must be less than smax ({cont['smax']})"
            )

        if not (cont["smin"] <= cont["s0"] <= cont["smax"]):
            raise ValueError(
                f"continuation.s0 ({cont['s0']}) must be between smin ({cont['smin']}) and smax ({cont['smax']})"
            )

        # Rule 4: ContPar bounds consistency
        if cont["ContParMin"] >= cont["ContParMax"]:
            raise ValueError(
                f"continuation.ContParMin ({cont['ContParMin']}) must be less than ContParMax ({cont['ContParMax']})"
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

    def add_doffunction(self, fxn):
        self.doffunction = fxn

    def add_icfunction(self, fxn):
        self.icfunction = fxn

    def add_zerofunction(self, fxn, fxn2=None):
        self.zerofunction = fxn
        if not fxn2:
            self.zerofunction_firstpoint = fxn
        else:
            self.zerofunction_firstpoint = fxn2

    def add_zerofunction_firstpoint(self, fxn):
        self.zerofunction_firstpoint = fxn

    def add_partitionfunction(self, fxn):
        self.partitionfunction = fxn
