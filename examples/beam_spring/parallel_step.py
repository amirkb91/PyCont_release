#!/usr/bin/env python3
"""
Parameter Step Script for PyCont Beam Spring Model

This script runs the continuation solver for different types of parameter steps.
It supports S-curve (frequency step) and Forced Response Curve (amplitude step).

Usage:
    python parallel_step.py [S|F]
    
    S: S-curve (frequency step) - Computes S-Curves by stepping through a range 
    of frequencies.
    F: Forced Response Curve (amplitude step) - Computes Forced Response Curves 
    by stepping through a range of forcing amplitudes.

    If S is selected, step parameters become frequency steps, and if F is selected, 
    step parameters become forcing amplitude steps.

    Note, timesim_branch is called automatically for all solution files. Default
    behaviour for timesim_branch can be changed in run_timesim_branch below.
    
    Once completed, the full response surface can be visualised by running:
    python ../../postprocess/FRC_Scurve_3D.py *.h5

Configuration:
    Modify the configuration variables below as needed.

"""

# =============================================================================
num_processes = 6  # Number of parallel processes

# Step parameters (Universal for F or S)
param_start = 10
param_end = 25
param_step = 0.5
# =============================================================================

import json
import subprocess
import sys
import os
import multiprocessing as mp
from functools import partial


def load_config(config_file):
    """Load JSON configuration file"""
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_file}': {e}")
        sys.exit(1)


def save_config(config, config_file):
    """Save JSON configuration file"""
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def cleanup_temp_configs(step_type):
    """Remove any leftover temporary configuration files"""
    import glob

    temp_files = glob.glob(f"contparameters_{step_type}_*.json")
    if temp_files:
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass  # Don't fail if cleanup doesn't work


def run_solver(config_file):
    """Run the solver with the given configuration file"""
    try:
        result = subprocess.run(
            [sys.executable, "runscript.py", config_file],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        return False
    except Exception as e:
        return False


def run_timesim_branch(h5_filename):
    """Run timesim_branch.py on the generated h5 file"""
    try:
        # Prepare input for timesim_branch.py (n for run_bif and y for store_physical)
        input_data = "n\ny\n"
        result = subprocess.run(
            [sys.executable, "timesim_branch.py", f"{h5_filename}.h5", "-i"],
            input=input_data,
            text=True,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        return False
    except Exception as e:
        return False


def run_single_parameter(param_value, base_config, base_filename, step_type):
    """Run solver for a single parameter value"""
    # Create unique config file for this parameter
    config_filename = f"contparameters_{step_type}_{param_value:.3f}.json"
    unique_filename = f"{base_filename}_{step_type}_{param_value:.3f}"

    # Create a copy of the base configuration
    config = base_config.copy()

    # Update the appropriate parameter based on step type
    if step_type == "frequency":
        config["forcing"]["frequency"] = param_value
        config["continuation"]["continuation_parameter"] = "amplitude"
    elif step_type == "amplitude":
        config["forcing"]["amplitude"] = param_value
        config["continuation"]["continuation_parameter"] = "frequency"
    else:
        raise ValueError(f"Unknown step type: {step_type}")

    config["Logger"]["file_name"] = unique_filename
    config["Logger"]["plot"] = False

    # Save configuration to unique file
    try:
        save_config(config, config_filename)

        # Print parameter being processed
        print(f"[{step_type.title()} {param_value:.1f}]")

        # Run solver
        solver_success = run_solver(config_filename)

        # Clean up config file
        try:
            os.remove(config_filename)
        except:
            pass  # Don't fail if cleanup doesn't work

        if solver_success:
            # Run timesim_branch
            timesim_success = run_timesim_branch(unique_filename)
            return (param_value, unique_filename, True, timesim_success)
        else:
            return (param_value, unique_filename, False, False)

    except Exception as e:
        # Clean up config file in case of error
        try:
            os.remove(config_filename)
        except:
            pass
        return (param_value, unique_filename, False, False)


def parameter_step():
    """Main function to perform parameter step with parallel processing"""

    # Determine step type from command line argument
    if len(sys.argv) < 2:
        print("Error: Missing argument")
        print("Usage: python frequency_sweep.py [S|F]")
        print("  S: S-curve (frequency step)")
        print("  F: Forced Response Curve (amplitude step)")
        sys.exit(1)

    arg = sys.argv[1].upper()
    if arg == "S":
        step_type = "frequency"
        step_name = "S-curve"
    elif arg == "F":
        step_type = "amplitude"
        step_name = "Forced Response Curve"
    else:
        print("Error: Invalid argument. Use 'S' or 'F'")
        print("Usage: python frequency_sweep.py [S|F]")
        print("  S: S-curve (frequency step)")
        print("  F: Forced Response Curve (amplitude step)")
        sys.exit(1)

    # Configuration
    config_file = "contparameters.json"
    base_filename = f"{step_type}_step"

    parameter_list = [
        param_start + i * param_step for i in range(int((param_end - param_start) / param_step) + 1)
    ]

    print("=" * 60)
    print(f"PyCont Parallel {step_name}")
    print("=" * 60)
    print(f"Configuration file: {config_file}")
    print(f"Step type: {step_name}")
    print(f"Parameters to step: {[f'{p:.2f}' for p in parameter_list]}")
    print(f"Number of runs: {len(parameter_list)}")
    print(f"Parallel processes: {num_processes}")
    print("=" * 60)

    # Load base configuration
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

    if not os.path.exists("contparameters.json"):
        print("Error: contparameters.json not found in current directory.")
        print("Please ensure the configuration file exists.")
        sys.exit(1)

    if not os.path.exists("timesim_branch.py"):
        print("Error: timesim_branch.py not found in current directory.")
        print("Please ensure timesim_branch.py exists in the beam_spring directory.")
        sys.exit(1)

    parameter_step()
