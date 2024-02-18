import json
import h5py
import shutil
import numpy as np
from alive_progress import alive_bar

from duffing import Duffing
from runscript import run

def update_data(file='FRF1', inplace=True):
    """Run time simulations to store acceleration data

    Args:
        file (str, optional): File with Continuation results. Defaults to 'FRF1'.
        inplace (bool, optional): Modify file or create new file. Defaults to True.
    """
    if not file.endswith(".h5"):
        file += ".h5"

    data = h5py.File(str(file), "r")
    pose = data["/Config/POSE"][:]
    vel = data["/Config/VELOCITY"][:]
    T = data["/T"][:]
    par = data["/Parameters"]
    par = json.loads(par[()])
    data.close()

    # Create new file to store time histories or append inplace
    if inplace:
        new_file = file
    else:
        new_file = file.strip(".h5") + "_withtime.h5"
        shutil.copy(file, new_file)

    n_solpoints = len(T)
    nsteps = par["shooting"]["single"]["nsteps_per_period"]
    pose_time = np.zeros([np.shape(pose)[0], nsteps + 1, n_solpoints])
    vel_time = np.zeros([np.shape(vel)[0], nsteps + 1, n_solpoints])
    acc_time = np.zeros([np.shape(vel)[0], nsteps + 1, n_solpoints])
    time = np.zeros([n_solpoints, nsteps + 1])

    # Run sims
    Duffing.forcing_parameters(par)
    with alive_bar(n_solpoints) as bar:
        for i in range(n_solpoints):
            X = np.array([0.0, vel[0, i]])
            [_, J, pose_time[:, :, i], vel_time[:, :, i], _, _] = Duffing.time_solve(
                1.0, T[i], X, pose[0, i], par
            )

            time[i, :] = np.linspace(0, T[i], nsteps + 1)
            # Acceleration
            acc_time[:, :, i] = (
                Duffing.F * np.cos((2 * np.pi / T[i]) * time[i, :] + Duffing.phi)
                - Duffing.delta * vel_time[:, :, i]
                - Duffing.alpha * pose_time[:, :, i]
                - Duffing.beta * pose_time[:, :, i] ** 3
            )
            bar()

    # Write to file
    time_data = h5py.File(new_file, "a")
    if "/Config_Time/POSE" in time_data.keys():
        del time_data["/Config_Time/POSE"]
    if "/Config_Time/VELOCITY" in time_data.keys():
        del time_data["/Config_Time/VELOCITY"]
    if "/Config_Time/ACCELERATION" in time_data.keys():
        del time_data["/Config_Time/ACCELERATION"]
    if "/Config_Time/Time" in time_data.keys():
        del time_data["/Config_Time/Time"]
    time_data["/Config_Time/POSE"] = pose_time
    time_data["/Config_Time/VELOCITY"] = vel_time
    time_data["/Config_Time/ACCELERATION"] = acc_time
    time_data["/Config_Time/Time"] = time
    time_data.close()


def generate_data(file_name='contparameters.json', min_force_amp=0.1, max_force_amp=1.0, step=0.1, phase_ratio=0.5, damping=0.05):
    """Data generator

    Args:
        file_name (str, optional): File with Continuation parameters. Defaults to 'contparameters.json'.
        min_force_amp (float, optional): Defaults to 0.1.
        max_force_amp (float, optional): Defaults to 1.0.
        step (float, optional): Defaults to 0.1.
        phase_ratio (float, optional): Defaults to 0.5.
        damping (float, optional): Defaults to 0.05.
    """
    # range only works with integers
    min_force_amp = int(min_force_amp*10)
    max_force_amp = int(max_force_amp*10)
    step = int(step*10)

    for i in range(min_force_amp, max_force_amp+1, step):
        # Open contparameters.json
        with open(file_name, 'r') as file:
            data = json.load(file)
            # Fixed parameters to be set
            data['forcing']['phase_ratio'] = phase_ratio
            data['forcing']['tau0'] = damping
        
            # Modify forcing amplitude
            data['forcing']['amplitude'] = 0.1*i
            # Save file
            data['Logger']['file_name'] = f'FRF{i}'
        
        # Modify contparameters.json
        with open(file_name, 'w') as file:
            json.dump(data, file, indent = 2)
            
        # Run continuation
        run()
        
        # Add acceleration
        update_data(f'FRF{i}')
    