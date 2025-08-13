import json
import h5py
import os
import shutil
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def save_to_file(filename='frequency_step_frequency_', path='./results/'):
    """Store simulation data for ML models
    """
    # Store ML data
    data = {}

    # Loop over all files in directory
    for i in np.arange(10.0, 24.1, 0.2):
        # Open new file
        file = f"{filename}{i}00.h5"

        # Access data
        data = h5py.File(str(file), "r")
        pose = data["/Config_Time/POSE"][:].squeeze()
        vel = data["/Config_Time/VELOCITY"][:].squeeze()
        acc = data["/Config_Time/ACCELERATION"][:].squeeze()
        time = data["/Config_Time/Time"][:].squeeze()
        F = data["/Force_Amp"][:].squeeze()
        T = data["/T"][:].squeeze()
        # Close file
        data.close()

        # Add to dict
        data[file.strip(".h5")] = {}
        data[file.strip(".h5")]['pose'] = pose
        data[file.strip(".h5")]['vel'] = vel
        data[file.strip(".h5")]['acc'] = acc
        data[file.strip(".h5")]['time'] = time
        data[file.strip(".h5")]['force'] = F
        data[file.strip(".h5")]['T'] = T

    # Save dict to file
    with open(f'{path}/data.pkl', 'wb') as fp:
        pickle.dump(data, fp)
