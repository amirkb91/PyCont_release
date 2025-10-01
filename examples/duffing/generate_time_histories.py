import h5py
import yaml
from alive_progress import alive_bar
import sys, shutil
import numpy as np
from duffing import Duffing
from postprocess.bifurcation import bifurcation_functions

""" 
Computes time-domain simulation data for all solution points along a continuation branch
and appends the results to the HDF5 solution file for subsequent postprocessing.
"""

run_bif = input("Compute bifurcation functions (y/n)? ")
if run_bif not in ("y", "n"):
    raise Exception("Input not valid")

# Read solution file
file = sys.argv[1]
inplace = sys.argv[-1]
if inplace == "-i":
    inplace = True
else:
    inplace = False

if not file.endswith(".h5"):
    file += ".h5"

data = h5py.File(str(file), "r")
inc = data["Config/INC"][:]
vel = data["Config/VEL"][:]
T = data["T"][:]
F = data["Force_Amp"][:]
par = data["Parameters"]
par = yaml.safe_load(par[()])
data.close()

# Create new file to store time histories or append inplace
if inplace:
    new_file = file
else:
    new_file = file.strip(".h5") + "_withtime.h5"
    shutil.copy(file, new_file)

n_solpoints = len(T)
nsteps = par["shooting"]["steps_per_period"]
n_dof_store = np.shape(inc)[0]
inc_time = np.zeros([n_dof_store, nsteps + 1, n_solpoints])
vel_time = np.zeros([n_dof_store, nsteps + 1, n_solpoints])
acc_time = np.zeros([n_dof_store, nsteps + 1, n_solpoints])
time = np.zeros([n_solpoints, nsteps + 1])

# Update model and run simulations
Duffing.update_model(par)
if run_bif == "y":
    Floquet = np.zeros([2, n_solpoints], dtype=np.complex128)
    Stability = np.zeros(n_solpoints)
    Fold = np.zeros(n_solpoints)
    Flip = np.zeros(n_solpoints)
    Neimark_Sacker = np.zeros(n_solpoints)

with alive_bar(n_solpoints) as bar:
    for i in range(n_solpoints):
        # Prepare initial conditions: concatenate pose and velocity
        X = np.concatenate([inc[:, i], vel[:, i]])

        if run_bif == "n":
            # no need for monodromy
            _inc, _vel, _acc = Duffing.time_simulate(F[i], T[i], X, par)
        elif run_bif == "y":
            _inc, _vel, _acc, M = Duffing.time_simulate_with_monodromy(F[i], T[i], X, par)

        # Store the time series data
        inc_time[:, :, i] = _inc.T
        vel_time[:, :, i] = _vel.T
        acc_time[:, :, i] = _acc.T

        if run_bif == "y":
            # Compute bifurcation functions
            bifurcation_out = bifurcation_functions(M)
            Floquet[:, i] = bifurcation_out[0]
            Stability[i] = bifurcation_out[1]
            Fold[i] = bifurcation_out[2]
            Flip[i] = bifurcation_out[3]
            Neimark_Sacker[i] = bifurcation_out[4]
        time[i, :] = np.linspace(0, T[i], nsteps + 1)
        bar()

# Write time series data to file
with h5py.File(new_file, "a") as time_data:
    # Create or overwrite Config_Time group
    if "Config_Time" in time_data:
        del time_data["Config_Time"]
    time_group = time_data.create_group("Config_Time")
    time_group.create_dataset("INC", data=inc_time)
    time_group.create_dataset("VELOCITY", data=vel_time)
    time_group.create_dataset("ACCELERATION", data=acc_time)
    time_group.create_dataset("Time", data=time)

    # Write bifurcation data if computed
    if "Bifurcation" in time_data:
        del time_data["Bifurcation"]
    if run_bif == "y":
        bif_group = time_data.create_group("Bifurcation")
        bif_group.create_dataset("Floquet", data=Floquet)
        bif_group.create_dataset("Stability", data=Stability)
        bif_group.create_dataset("Fold", data=Fold)
        bif_group.create_dataset("Flip", data=Flip)
        bif_group.create_dataset("Neimark_Sacker", data=Neimark_Sacker)

print(f"Time simulation complete. Results saved to {new_file}")
