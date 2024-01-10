import h5py
import json
from alive_progress import alive_bar
import sys, shutil
import numpy as np
from scipy.integrate import odeint
from duffing import Duffing
from postprocess.bifurcation import bifurcation_functions

""" Run time simulations for all points on solution branch and store """

run_bif = input("Compute bifurcation functions? ")

# read solution file
file = sys.argv[1]
inplace = sys.argv[-1]
if inplace == "-i":
    inplace = True
else:
    inplace = False

if not file.endswith(".h5"):
    file += ".h5"

data = h5py.File(str(file), "r")
pose = data["/Config/POSE"][:]
vel = data["/Config/VELOCITY"][:]
T = data["/T"][:]
par = data["/Parameters"]
par = json.loads(par[()])
data.close()

# create new file to store time histories or append inplace
if inplace:
    new_file = file
else:
    new_file = file.strip(".h5") + "_withtime.h5"
    shutil.copy(file, new_file)

n_solpoints = len(T)
nsteps = par["shooting"]["single"]["nsteps_per_period"]
pose_time = np.zeros([np.shape(pose)[0], nsteps + 1, n_solpoints])
vel_time = np.zeros([np.shape(vel)[0], nsteps + 1, n_solpoints])
time = np.zeros([n_solpoints, nsteps + 1])

# run sims
Duffing.forcing_parameters(par)
if run_bif:
    Floquet = np.zeros([2, n_solpoints], dtype=np.complex128)
    Stability = np.zeros(n_solpoints)
    Fold = np.zeros(n_solpoints)
    Flip = np.zeros(n_solpoints)
    Neimark_Sacker = np.zeros(n_solpoints)

with alive_bar(n_solpoints) as bar:
    for i in range(n_solpoints):
        X = np.array([0.0, vel[0, i]])
        [_, J, pose_time[:, :, i], vel_time[:, :, i], _, _] = Duffing.time_solve(
            1.0, T[i], X, pose[0, i], par
        )
        if run_bif:
            M = J[:, :-1] + np.eye(2)
            bifurcation_out = bifurcation_functions(M)
            Floquet[:, i] = bifurcation_out[0]
            Stability[i] = bifurcation_out[1]
            Fold[i] = bifurcation_out[2]
            Flip[i] = bifurcation_out[3]
            Neimark_Sacker[i] = bifurcation_out[4]
        time[i, :] = np.linspace(0, T[i], nsteps + 1)
        bar()

# write to file
time_data = h5py.File(new_file, "a")
if "/Config_Time/POSE" in time_data.keys():
    del time_data["/Config_Time/POSE"]
if "/Config_Time/VELOCITY" in time_data.keys():
    del time_data["/Config_Time/VELOCITY"]
if "/Config_Time/Time" in time_data.keys():
    del time_data["/Config_Time/Time"]
time_data["/Config_Time/POSE"] = pose_time
time_data["/Config_Time/VELOCITY"] = vel_time
time_data["/Config_Time/Time"] = time
if run_bif:
    if "/Bifurcation/Floquet" in time_data.keys():
        del time_data["/Bifurcation/Floquet"]
    if "/Bifurcation/Stability" in time_data.keys():
        del time_data["/Bifurcation/Stability"]
    if "/Bifurcation/Fold" in time_data.keys():
        del time_data["/Bifurcation/Fold"]
    if "/Bifurcation/Flip" in time_data.keys():
        del time_data["/Bifurcation/Flip"]
    if "/Bifurcation/Neimark_Sacker" in time_data.keys():
        del time_data["/Bifurcation/Neimark_Sacker"]
    time_data["/Bifurcation/Floquet"] = Floquet
    time_data["/Bifurcation/Stability"] = Stability
    time_data["/Bifurcation/Fold"] = Fold
    time_data["/Bifurcation/Flip"] = Flip
    time_data["/Bifurcation/Neimark_Sacker"] = Neimark_Sacker
time_data.close()
