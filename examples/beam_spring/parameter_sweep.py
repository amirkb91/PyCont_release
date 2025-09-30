#!/usr/bin/env python3
"""
Parameter Step Script for PyCont Beam Spring Model

This script runs the continuation solver for different types of parameter steps
using the parameters file (parameters.yaml). It supports:
- S-curve (frequency step): fix forcing frequency and continue in forcing amplitude
- Forced Response Curve (amplitude step): fix forcing amplitude and continue in frequency

Usage:
    python parameter_sweep.py

Configuration:
    Set `step_mode = 'S'` for S-curves (frequency stepping) or `step_mode = 'F'`
    for Forced Response Curves (amplitude stepping) in the configuration section below.

    Once completed, the full response surface can be visualised by running:
    python ../../postprocess/FRC_Scurve_3D.py *.h5
"""

# =============================================================================
num_processes = 6  # Number of parallel processes

# Step parameters (Universal for F or S)
param_start = 10
param_end = 20
param_step = 0.5

# Choose step mode: 'S' for multiple S-curves (stepping frequencies),
# 'F' for multiple Forced Responses (stepping amplitude)
step_mode = "S"
# =============================================================================

import yaml
import subprocess
import sys
import os
import multiprocessing as mp
from functools import partial
import shutil


def load_config(config_file):
    """Load YAML configuration file"""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in '{config_file}': {e}")
        sys.exit(1)


def save_config(config, config_file):
    """Save YAML configuration file"""
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def cleanup_temp_configs(step_type):
    """Remove any leftover temporary files (YAML configs and temp runscripts)"""
    import glob

    yaml_files = glob.glob(f"parameters_{step_type}_*.yaml")
    run_files = glob.glob(f"runscript_temp_{step_type}_*.py")
    for temp_file in yaml_files + run_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass  # Don't fail if cleanup doesn't work


def _create_temp_runscript(temp_yaml_filename, step_type, param_value):
    """Create a temporary runscript that loads the provided YAML file."""
    try:
        with open("runscript.py", "r") as rf:
            content = rf.read()
        # Replace only the configure_parameters line safely
        new_content = content.replace(
            'prob.configure_parameters("parameters.yaml")',
            f'prob.configure_parameters("{temp_yaml_filename}")',
        )
        temp_script = f"runscript_temp_{step_type}_{param_value:.3f}.py"
        with open(temp_script, "w") as wf:
            wf.write(new_content)
        return temp_script
    except Exception:
        return None


def run_solver(temp_runscript_file):
    """Run the solver with the given temporary runscript file (output suppressed)."""
    try:
        subprocess.run(
            [sys.executable, temp_runscript_file],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def run_time_histories(h5_filename):
    """Run generate_time_histories.py on the generated h5 file (in-place)."""
    try:
        # Provide inputs: run_bif = 'n' (no bifurcation), store_physical = 'p' (physical tip disp)
        input_data = "n\np\n"
        subprocess.run(
            [sys.executable, "generate_time_histories.py", f"{h5_filename}.h5", "-i"],
            input=input_data,
            text=True,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def run_single_parameter(param_value, base_config, base_filename, step_type):
    """Run solver for a single parameter value (YAML-based) and generate time histories."""
    # Unique names for this parameter
    yaml_filename = f"parameters_{step_type}_{param_value:.3f}.yaml"
    unique_out = f"{base_filename}_{step_type}_{param_value:.3f}"

    # Create a deep copy of the base configuration to avoid shared mutation
    import copy

    config = copy.deepcopy(base_config)

    # Update the appropriate parameter based on step type and set continuation parameter
    if step_type == "frequency":
        config["forcing"]["frequency"] = float(param_value)
        config["continuation"]["parameter"] = "force_amp"
    elif step_type == "amplitude":
        config["forcing"]["amplitude"] = float(param_value)
        config["continuation"]["parameter"] = "force_freq"
    else:
        raise ValueError(f"Unknown step type: {step_type}")

    # Update logger settings for this run
    config["logger"]["output_file_name"] = unique_out
    config["logger"]["enable_live_plot"] = False

    # Save YAML configuration and create temporary runscript pointing to it
    try:
        save_config(config, yaml_filename)

        # Create a temp runscript that loads this YAML
        temp_runscript = _create_temp_runscript(yaml_filename, step_type, param_value)
        if temp_runscript is None:
            # Fallback: overwrite parameters.yaml temporarily and restore later
            backup_file = "parameters.yaml.bak"
            if os.path.exists("parameters.yaml"):
                shutil.copy("parameters.yaml", backup_file)
            shutil.copy(yaml_filename, "parameters.yaml")
            temp_runscript = "runscript.py"
            use_backup = True
        else:
            backup_file = None
            use_backup = False

        # Print parameter being processed
        print(f"[{step_type.title()} {param_value:.3f}]")

        # Run solver silently
        solver_success = run_solver(temp_runscript)

        # Restore original parameters.yaml if needed
        if use_backup:
            try:
                shutil.copy(backup_file, "parameters.yaml")
                os.remove(backup_file)
            except Exception:
                pass

        # Clean up temp files
        try:
            if temp_runscript != "runscript.py" and os.path.exists(temp_runscript):
                os.remove(temp_runscript)
            if os.path.exists(yaml_filename):
                os.remove(yaml_filename)
        except Exception:
            pass

        if solver_success:
            # Generate time histories in-place on the produced H5 file
            timesim_success = run_time_histories(unique_out)
            return (param_value, unique_out, True, timesim_success)
        else:
            return (param_value, unique_out, False, False)

    except Exception:
        # Attempt to clean temp files
        try:
            if os.path.exists(yaml_filename):
                os.remove(yaml_filename)
        except Exception:
            pass
        return (param_value, unique_out, False, False)


def parameter_step():
    """Main function to perform parameter step with parallel processing"""

    # Determine step type from internal configuration
    if step_mode.upper() == "S":
        step_type = "frequency"
        step_name = "S-curve"
    elif step_mode.upper() == "F":
        step_type = "amplitude"
        step_name = "Forced Response Curve"
    else:
        print("Error: Invalid step_mode in script. Use 'S' or 'F'.")
        sys.exit(1)

    # Configuration
    config_file = "parameters.yaml"
    base_filename = f"{step_type}_step"

    parameter_list = [
        param_start + i * param_step for i in range(int((param_end - param_start) / param_step) + 1)
    ]

    print("=" * 60)
    print(f"PyCont Parallel {step_name}")
    print("=" * 60)
    print(f"Base configuration file: {config_file}")
    print(f"Step type: {step_name}")
    print(f"Parameters to step: {[f'{p:.2f}' for p in parameter_list]}")
    print(f"Number of runs: {len(parameter_list)}")
    print(f"Parallel processes: {num_processes}")
    print("=" * 60)

    # Load base configuration (YAML)
    base_config = load_config(config_file)

    # Prepare function with partial application
    worker_func = partial(
        run_single_parameter,
        base_config=base_config,
        base_filename=base_filename,
        step_type=step_type,
    )

    # Run parallel processing
    print(f"Starting parallel execution with {num_processes} processes...\n")

    try:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(worker_func, parameter_list)
    except KeyboardInterrupt:
        print(f"\n\n{step_name} interrupted by user (Ctrl+C)")
        return
    except Exception as e:
        print(f"\nError in parallel processing: {e}")
        return

    # Process results
    successful_runs = []
    failed_runs = []
    timesim_successful = []
    timesim_failed = []

    for param, filename, solver_success, timesim_success in results:
        if solver_success:
            successful_runs.append((param, filename))
            if timesim_success:
                timesim_successful.append((param, filename))
            else:
                timesim_failed.append((param, filename))
        else:
            failed_runs.append((param, filename))

    # Print comprehensive summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Solver: {len(successful_runs)}/{len(parameter_list)} successful")
    print(f"Timesim: {len(timesim_successful)}/{len(successful_runs)} successful")

    if failed_runs:
        print(f"\nFailed {step_type}s: {[param for param, _ in failed_runs]}")

    if timesim_failed and len(timesim_failed) < len(successful_runs):
        print(f"Timesim failed: {[param for param, _ in timesim_failed]}")

    print(f"\nOutput files: {len(successful_runs)} h5 files generated")

    # Clean up any leftover temporary config files
    cleanup_temp_configs(step_type)

    print("=" * 50)


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("runscript.py"):
        print("Error: runscript.py not found in current directory.")
        print("Please run this script from the beam_spring example directory.")
        sys.exit(1)

    if not os.path.exists("parameters.yaml"):
        print("Error: parameters.yaml not found in current directory.")
        print("Please ensure the YAML configuration file exists.")
        sys.exit(1)

    if not os.path.exists("generate_time_histories.py"):
        print("Error: generate_time_histories.py not found in current directory.")
        print("Please ensure generate_time_histories.py exists in the beam_spring directory.")
        sys.exit(1)

    parameter_step()
