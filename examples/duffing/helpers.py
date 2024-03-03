import json
import h5py
import shutil
import pickle
import numpy as np
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    force_time = np.zeros([np.shape(vel)[0], nsteps + 1, n_solpoints])
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
            # Force
            force_time[:, :, i] = (
                Duffing.F * np.cos((2 * np.pi / T[i]) * time[i, :] + Duffing.phi)
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
    if "/Config_Time/Force" in time_data.keys():
        del time_data["/Config_Time/Force"]
    if "/Config_Time/Time" in time_data.keys():
        del time_data["/Config_Time/Time"]
    time_data["/Config_Time/POSE"] = pose_time
    time_data["/Config_Time/VELOCITY"] = vel_time
    time_data["/Config_Time/ACCELERATION"] = acc_time
    time_data["/Config_Time/Force"] = force_time
    time_data["/Config_Time/Time"] = time.T
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
    # Range only works with integers
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
        
        
def plot_sols(file='FRF1'):
    """Plot periodic solutions & return data (if needed)

    Args:
        file (str, optional): File to plot. Defaults to 'FRF1'.
    """
    # Get data
    if not file.endswith(".h5"):
            file += ".h5"
    data = h5py.File(str(file), "r")
    
    # Position, Velocity, Acceleration, Force = F*cos(2pi/T*t + phi), Time
    ## Note: 
    # >>> COL -> Number of Periodic Solutions
    # >>> ROW -> Number of Solution Points
    pose = data["/Config_Time/POSE"][:].squeeze()
    vel = data["/Config_Time/VELOCITY"][:].squeeze()
    acc = data["/Config_Time/ACCELERATION"][:].squeeze()
    force = data["/Config_Time/Force"][:].squeeze()
    time = data["/Config_Time/Time"][:].squeeze()
    # Period (Frequency) for Periodic Solution
    T = data["/T"][:]
    n_solpoints = len(T)
    
    # Plot figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Family of Periodic Solutions in Phase Space
    ax[0].set_title('Family of Periodic Solutions in Phase Space', fontsize=12)
    ax[0].plot(pose[:, ::30], vel[:, ::30])
    ax[0].set_xlabel(r'$x$', fontsize=12)
    ax[0].set_ylabel(r'$\dot{x}$', fontsize=12)

    # Plot NLFR
    ax[1].set_title(f'Frequency-Response Curve', fontsize=10)
    ax[1].plot(1/T, np.max(pose, axis=0))
    ax[1].set_xlabel(r'Period, $T$', fontsize=12)
    ax[1].set_ylabel(r'Amplitude, $x$', fontsize=12)
    
    d = {}
    d['pose'] = pose
    d['vel'] = vel
    d['acc'] = acc
    d['force'] = force
    d['time'] = time
    d['T'] = T
    return d


def save_to_file(num_files=10, filename='FRF'):
    """Store simulation data

    Args:
        num_files (int, optional): Files with Continuation results. Defaults to 10.
        filename (str, optional): File to save results. Defaults to 'FRF'.
    """
    # Store ML data
    d = {}
    
    # Loop over all files in directory
    for i in range(1, num_files+1):
        # Open new file
        file = filename
        file += f"{i}"
    
        if not file.endswith(".h5"):
            file += ".h5"

        # Access data
        data = h5py.File(str(file), "r")
        pose = data["/Config_Time/POSE"][:]
        vel = data["/Config_Time/VELOCITY"][:]
        acc = data["/Config_Time/ACCELERATION"][:]
        time = data["/Config_Time/Time"][:]
        force = data["/Config_Time/Force"][:]
        T = data["/T"][:]
        # Close file
        data.close()
        
        # Add to dict
        d[file.strip(".h5")] = {}
        d[file.strip(".h5")]['pose'] = pose
        d[file.strip(".h5")]['vel'] = vel
        d[file.strip(".h5")]['acc'] = acc
        d[file.strip(".h5")]['time'] = time
        d[file.strip(".h5")]['force'] = force
        d[file.strip(".h5")]['T'] = T
        
    # Save dict to file
    with open('data.pkl', 'wb') as fp:
        pickle.dump(d, fp)
    
    
def train_test_data(num_files=10, filename='FRF'):
    """Create & Split simulation data

    Args:
        num_files (int, optional): Files with Continuation results. Defaults to 10.
        filename (str, optional): File to save results. Defaults to 'FRF'.
    """
    # Access data files as dict
    data = save_to_file(num_files)
    
    # Store ML data
    x = np.array([])
    dx = np.array([])
    ddx = np.array([])
    t = np.array([])
    f = np.array([])
    period = np.array([])  
    # Empty dict to store current dile data
    d = {}
    
    # Loop over all files in directory
    for i in range(1, num_files+1):
        # Open new file
        file = filename
        file += f"{i}"
    
        if not file.endswith(".h5"):
            file += ".h5"

        # Access data
        data = h5py.File(str(file), "r")
        pose = data["/Config_Time/POSE"][:]
        vel = data["/Config_Time/VELOCITY"][:]
        acc = data["/Config_Time/ACCELERATION"][:]
        time = data["/Config_Time/Time"][:]
        force = data["/Config_Time/Force"][:]
        T = data["/T"][:]
        
        # Add to dict
        d[file.strip(".h5")] = {}
        d[file.strip(".h5")]['pose'] = pose
        d[file.strip(".h5")]['vel'] = vel
        d[file.strip(".h5")]['acc'] = acc
        d[file.strip(".h5")]['time'] = time
        d[file.strip(".h5")]['force'] = force
        d[file.strip(".h5")]['T'] = T
    
        # Close file
        data.close()
        
    # Save to file
    return d
        
        # for i, val in enumerate(data["/Config_Time"]):
            
        
        # d[file]
        
        # np.savez('save_to')
        
        
        
    #     x = np.append(x, pose)
    #     dx = np.append(dx, vel)
    #     ddx = np.append(ddx, acc)
    #     t = np.append(t, time)
    #     f = np.append(f, force)
    #     period = np.append(period, T)
        
    # # Convert to numpy array
    # x = x.reshape(301, -1)
    # dx = dx.reshape(301, -1)
    # ddx = ddx.reshape(301, -1)
    # t = t.reshape(301, -1)
    # f = f.reshape(301, -1)
    # period = period.reshape(-1)
    
    return x, dx, ddx, t, f, period
        
    # Save to file
    # np.savez(file, x, dx, ddx, t, f, period)