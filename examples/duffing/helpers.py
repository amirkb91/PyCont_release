import json
import h5py
import os
import shutil
import pickle
import numpy as np
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from duffing import Duffing
from duffing_lnn import Duffing_LNN
from runscript import run


def generate_data(file_name='contparameters.json', min_force_amp=0.1, max_force_amp=1.0, num=10, phase_ratio=0.5, damping=0.05, isLNN=False, predict_acc=None, pred_energy=None, path='./data'):
    """Data generator

    Args:
        file_name (str, optional): File with Continuation parameters. Defaults to 'contparameters.json'.
        min_force_amp (float, optional): Defaults to 0.1.
        max_force_amp (float, optional): Defaults to 1.0.
        num (int, optional): Defaults to 10.
        phase_ratio (float, optional): Defaults to 0.5.
        damping (float, optional): Defaults to 0.05.
    """
    # Range only works with integers
    min_force_amp = int(min_force_amp*10)
    max_force_amp = int(max_force_amp*10)
    step = (max_force_amp - min_force_amp)/(num-1)
    
    for i in range(1, num+1):
        # Open contparameters.json
        with open(file_name, 'r') as file:
            data = json.load(file)
            # Fixed parameters to be set
            data['forcing']['phase_ratio'] = phase_ratio
            data['forcing']['tau0'] = damping
        
            # Modify forcing amplitude
            data['forcing']['amplitude'] = min_force_amp + step
            # Save file
            data['Logger']['file_name'] = f'FRF{i}'
        
        # Modify contparameters.json
        with open(file_name, 'w') as file:
            json.dump(data, file, indent = 2)
            
        # Analytical vs LNN class
        model = Duffing_LNN if isLNN else Duffing
                   
        # Run continuation
        run(model=model, predict_acc=predict_acc, pred_energy=pred_energy)
        
        # Add acceleration & forcing
        info = update_data(file=f'FRF{i}', isLNN=isLNN)
        
    # Save results to single file 
    save_to_file(num_files=num, path=path)
    
    return info

def update_data(file='FRF1', inplace=True, isLNN=False):
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
    
    # Analytical vs LNN class
    model = Duffing_LNN if isLNN else Duffing

    # Run sims
    model.forcing_parameters(par)
    with alive_bar(n_solpoints) as bar:
        for i in range(n_solpoints):
            X = np.array([0.0, vel[0, i]])
            [_, J, pose_time[:, :, i], vel_time[:, :, i], _, _] = model.time_solve(
                1.0, T[i], X, pose[0, i], par
            )

            time[i, :] = np.linspace(0, T[i], nsteps + 1)
            # REVIEW: Acceleration 
            acc_time[:, :, i] = (
                model.F * np.cos((2 * np.pi / T[i]) * time[i, :] + model.phi)
                - model.delta * vel_time[:, :, i]
                - model.alpha * pose_time[:, :, i]
                - model.beta * pose_time[:, :, i] ** 3
            )
            # Force
            force_time[:, :, i] = (
                model.F * np.cos((2 * np.pi / T[i]) * time[i, :] + model.phi)
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
    
    # Physical System
    info = {}
    info['delta'] = model.delta
    info['alpha'] = model.alpha
    info['beta'] = model.beta
    info['M'] = 1.0
    info['K'] = model.alpha * info['M']
    info['C'] = model.delta * info['M']
    info['NL'] = model.beta * info['M']
    
    return info


def save_to_file(num_files=10, filename='FRF', path='./data'):
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
        pose = data["/Config_Time/POSE"][:].squeeze()
        vel = data["/Config_Time/VELOCITY"][:].squeeze()
        acc = data["/Config_Time/ACCELERATION"][:].squeeze()
        time = data["/Config_Time/Time"][:].squeeze()
        force = data["/Config_Time/Force"][:].squeeze()
        T = data["/T"][:].squeeze()
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
        
        # Clean directory
        if i==1:
            try:
                os.mkdir(path)
            except FileExistsError:
                shutil.rmtree(path)
                os.mkdir(path)
        shutil.move(file, f"{path}/{file}")
        
    # Save dict to file
    with open(f'{path}/data.pkl', 'wb') as fp:
        pickle.dump(d, fp)
        

def train_test_data_old(
    save_file='data/data.pkl', 
    split_size=0.20, 
    file_name='contparameters.json', 
    min_force_amp=0.1, 
    max_force_amp=1.0, 
    num=10, 
    phase_ratio=0.5, 
    damping=0.05
):
    """Create & Split simulation data

    Args:
        save_file (str, optional): Results. Defaults to 'data.pkl'.
        split_size (float, optional): Trainig/Test split. Defaults to 20%.
    """
    # Generate training data
    info = generate_data(
        file_name=file_name, 
        min_force_amp=min_force_amp, 
        max_force_amp=max_force_amp, 
        num=num, 
        phase_ratio=phase_ratio, 
        damping=damping
    )
    
    # Read data
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    
    # Store ML data
    x = np.array([])
    dx = np.array([])
    ddx = np.array([])
    t = np.array([])
    f = np.array([])
    period = np.array([])
    
    # Loop over data
    for k, v in data.items():
        pose = data[k]["pose"]
        vel = data[k]["vel"]
        acc = data[k]["acc"]
        time = data[k]["time"]
        force = data[k]["force"]
        T = data[k]["T"]
         
        x = np.append(x, pose)
        dx = np.append(dx, vel)
        ddx = np.append(ddx, acc)
        t = np.append(t, time)
        f = np.append(f, force)
        period = np.append(period, T)
        
    # Reshape array
    x = x.reshape(pose.shape[0], -1)
    dx = dx.reshape(vel.shape[0], -1)
    ddx = ddx.reshape(acc.shape[0], -1)
    t = t.reshape(time.shape[0], -1)
    f = f.reshape(force.shape[0], -1)
    period = period.reshape(-1)
    
    # Create train & test split
    x_train, x_test, dx_train, dx_test, ddx_train, ddx_test, t_train, t_test, f_train, f_test = train_test_split(x, dx, ddx, t, f, test_size=split_size, random_state=42, shuffle=True)
    
    train_dataset, test_dataset = {}, {}    
    train_dataset['x'] = x_train
    train_dataset['dx'] = dx_train
    train_dataset['ddx'] = ddx_train
    train_dataset['t'] = t_train
    train_dataset['f'] = f_train
    test_dataset['x'] = x_test
    test_dataset['dx'] = dx_test
    test_dataset['ddx'] = ddx_test
    test_dataset['t'] = t_test
    test_dataset['f'] = f_test
    
    # Add relevant info
    info['train_n_datapoints'] = train_dataset['x'].shape[0] * train_dataset['x'].shape[-1]
    info['test_n_datapoints'] = test_dataset['x'].shape[0] * test_dataset['x'].shape[-1]
    info['qmax'] = train_dataset['x'][:, 0].max()
    info['qdmax'] = train_dataset['dx'][:, 0].max()
    info['qddmax'] = train_dataset['ddx'][:, 0].max()
    info['t'] = train_dataset['t'][:, 0].max()
    info['fmax'] = train_dataset['f'].max()
    
    return train_dataset, test_dataset, info
    
    
def train_test_data(
    save_file='data/data.pkl', 
    split_size=0.20, 
    file_name='contparameters.json', 
    min_force_amp=0.1, 
    max_force_amp=1.0, 
    num=10, 
    phase_ratio=0.5, 
    damping=0.05
):
    """Create & Split simulation data

    Args:
        save_file (str, optional): Results. Defaults to 'data.pkl'.
        split_size (float, optional): Trainig/Test split. Defaults to 20%.
    """
    # Generate training data
    info = generate_data(
        file_name=file_name, 
        min_force_amp=min_force_amp, 
        max_force_amp=max_force_amp, 
        num=num, 
        phase_ratio=phase_ratio, 
        damping=damping
    )
    
    # Read data
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    
    # Store ML data
    x_train = np.array([])
    dx_train = np.array([])
    ddx_train = np.array([])
    t_train = np.array([])
    f_train = np.array([])
    x_test = np.array([])
    dx_test = np.array([])
    ddx_test = np.array([])
    t_test = np.array([])
    f_test = np.array([])

    # Loop over data
    for k, v in data.items():
        pose = data[k]["pose"].flatten()
        vel = data[k]["vel"].flatten()
        acc = data[k]["acc"].flatten()
        time = data[k]["time"].flatten()
        force = data[k]["force"].flatten()
        T = data[k]["T"].flatten()
        
        # Create train & test split with equal split for each forcing amplitude
        pose_train, pose_test, vel_train, vel_test, acc_train, acc_test, time_train, time_test, force_train, force_test = train_test_split(pose, vel, acc, time, force, test_size=0.2, random_state=42, shuffle=True)
        
        # Collect
        x_train = np.append(x_train, pose_train)
        dx_train = np.append(dx_train, vel_train)
        ddx_train = np.append(ddx_train, acc_train)
        t_train = np.append(t_train, time_train)
        f_train = np.append(f_train, force_train)
        x_test = np.append(x_test, pose_test)
        dx_test = np.append(dx_test, vel_test)
        ddx_test = np.append(ddx_test, acc_test)
        t_test = np.append(t_test, time_test)
        f_test = np.append(f_test, force_test)
        
    train_dataset, test_dataset = {}, {}    
    train_dataset['x'] = x_train
    train_dataset['dx'] = dx_train
    train_dataset['ddx'] = ddx_train
    train_dataset['t'] = t_train
    train_dataset['f'] = f_train
    test_dataset['x'] = x_test
    test_dataset['dx'] = dx_test
    test_dataset['ddx'] = ddx_test
    test_dataset['t'] = t_test
    test_dataset['f'] = f_test
    
    # Add relevant info
    info['train_n_datapoints'] = train_dataset['x'].shape[0]
    info['test_n_datapoints'] = test_dataset['x'].shape[0]
    info['qmax'] = train_dataset['x'][:].max()
    info['qdmax'] = train_dataset['dx'][:].max()
    info['qddmax'] = train_dataset['ddx'][:].max()
    info['t'] = train_dataset['t'][:].max()
    info['fmax'] = train_dataset['f'].max()
    
    return train_dataset, test_dataset, info


def format_to_LNN(old_train_dataset, old_test_dataset, info):
    """_summary_

    Args:
        old_train_dataset (dict): Training dataset with keys ['x', 'dx', 'ddx', 't', 'f'] 
        old_test_dataset (dict): Test dataset with keys ['x', 'dx', 'ddx', 't', 'f'] 
        info (dict): Information about physical system
    """
    
    # Position, velocity & total forcing conditions
    train_x = np.vstack((old_train_dataset['x'], old_train_dataset['dx'])).T
    train_f = old_train_dataset['f']
    train_dx = np.vstack((old_train_dataset['dx'], old_train_dataset['ddx'])).T

    test_x = np.vstack((old_test_dataset['x'], old_test_dataset['dx'])).T
    test_f = old_test_dataset['f']
    test_dx = np.vstack((old_test_dataset['dx'], old_test_dataset['ddx'])).T

    train_dataset = train_x, train_f, train_dx
    test_dataset = test_x, test_f, test_dx
    
    return train_dataset, test_dataset, info

# -------------------------- VISUALIZATION FUNCTIONS ------------------------- #

def compare_sols(anal_file='data/FRF1', lnn_file='data_LNN/FRF1'):
    """Compare periodic solutions

    Args:
        anal_file (str, optional): File with analytical results.
        lnn_file (str, optional): File with LNN results.
    """
    title = anal_file.split("/")[1]
    
    # Get data
    if not anal_file.endswith(".h5"):
            anal_file += ".h5"
    anal_data = h5py.File(str(anal_file), "r")
    
    if not lnn_file.endswith(".h5"):
            lnn_file += ".h5"
    lnn_data = h5py.File(str(lnn_file), "r")
    
    # Position, Velocity, Acceleration, Force, Time
    ## Note: 
    # >>> COL -> Number of Periodic Solutions
    # >>> ROW -> Number of Solution Points
    anal_pose = anal_data["/Config_Time/POSE"][:].squeeze()
    anal_vel = anal_data["/Config_Time/VELOCITY"][:].squeeze()
    anal_acc = anal_data["/Config_Time/ACCELERATION"][:].squeeze()
    anal_force = anal_data["/Config_Time/Force"][:].squeeze()
    anal_time = anal_data["/Config_Time/Time"][:].squeeze()
    # Period (Frequency) for Periodic Solution
    anal_T = anal_data["/T"][:]
    
    lnn_pose = lnn_data["/Config_Time/POSE"][:].squeeze()
    lnn_vel = lnn_data["/Config_Time/VELOCITY"][:].squeeze()
    lnn_acc = lnn_data["/Config_Time/ACCELERATION"][:].squeeze()
    lnn_force = lnn_data["/Config_Time/Force"][:].squeeze()
    lnn_time = lnn_data["/Config_Time/Time"][:].squeeze()
    # Period (Frequency) for Periodic Solution
    lnn_T = lnn_data["/T"][:]
    
    # Plot figures
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Plot NLFR
    ax[0, 0].set_title(f'{title}', fontsize=14)
    ax[0, 0].plot(1/anal_T, np.max(anal_pose, axis=0), label='Analytical')
    ax[0, 0].plot(1/lnn_T, np.max(lnn_pose, axis=0), '--', label='LNN')
    ax[0, 0].set_xlabel(r'Frequency, $1/T$', fontsize=12)
    ax[0, 0].set_ylabel(r'Amplitude, $x$', fontsize=12)
    
    # Plot position, velocity & time
    ax[0, 1].set_title('Position', fontsize=14)
    ax[0, 1].plot(anal_time[:, 0], anal_pose[:, 0], label='Analytical')
    ax[0, 1].plot(lnn_time[:, 0], lnn_pose[:, 0], '--', label='LNN')
    ax[0, 1].set_xlabel(r'$t$', fontsize=12)
    ax[0, 1].set_ylabel(r'${x}$', fontsize=12)

    ax[1, 0].set_title('Velocity', fontsize=14)
    ax[1, 0].plot(anal_time[:, 0], anal_vel[:, 0], label='Analytical')
    ax[1, 0].plot(lnn_time[:, 0], lnn_vel[:, 0], '--', label='LNN')
    ax[1, 0].set_xlabel(r'$t$', fontsize=12)
    ax[1, 0].set_ylabel(r'$\dot{x}$', fontsize=12)

    ax[1, 1].set_title('Acceleration', fontsize=14)
    ax[1, 1].plot(anal_time[:, 0], anal_acc[:, 0], label='Analytical')
    ax[1, 1].plot(lnn_time[:, 0], lnn_acc[:, 0], '--', label='LNN')
    ax[1, 1].set_xlabel(r'$t$', fontsize=12)
    ax[1, 1].set_ylabel(r'$\ddot{x}$', fontsize=12)

    ax[0,0].legend()
    fig.tight_layout()
    
def plot_sols(file='data/FRF1'):
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
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    # Plot Family of Periodic Solutions in Phase Space
    ax[0, 0].set_title('Family of Periodic Solutions in Phase Space', fontsize=12)
    ax[0, 0].plot(pose[:, ::30], vel[:, ::30])
    ax[0, 0].set_xlabel(r'$x$', fontsize=12)
    ax[0, 0].set_ylabel(r'$\dot{x}$', fontsize=12)

    # Plot NLFR
    ax[0, 1].set_title(f'Frequency-Response Curve', fontsize=10)
    ax[0, 1].plot(1/T, np.max(pose, axis=0))
    ax[0, 1].set_xlabel(r'Period, $T$', fontsize=12)
    ax[0, 1].set_ylabel(r'Amplitude, $x$', fontsize=12)
    
    # Plot position, velocity & time
    ax[1, 0].set_title('Position', fontsize=10)
    ax[1, 0].plot(time[:, 0], pose[:, 0])
    ax[1, 0].set_xlabel(r'$t$')
    ax[1, 0].set_ylabel(r'${x}$')

    ax[1, 1].set_title('Velocity', fontsize=10)
    ax[1, 1].plot(time[:, 0], vel[:, 0])
    ax[1, 1].set_xlabel(r'$t$')
    ax[1, 1].set_ylabel(r'$\dot{x}$')

    ax[2, 0].set_title('Acceleration', fontsize=10)
    ax[2, 0].plot(time[:, 0], acc[:, 0])
    ax[2, 0].set_xlabel(r'$t$')
    ax[2, 0].set_ylabel(r'$\ddot{x}$')

    fig.tight_layout()