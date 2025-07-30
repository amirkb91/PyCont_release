import h5py
import sys
import json
import numpy as np
from scipy.integrate import odeint
from duffing import Duffing
import matplotlib.pyplot as plt

""" Run time simulations for single point solution branch and plot """

# inputs
solno = int(input("Solution Index: "))

# read solution file
file = sys.argv[-1]
if not file.endswith(".h5"):
    file += ".h5"
data = h5py.File(str(file), "r")
pose = data["/Config/POSE"][:, solno]
vel = data["/Config/VELOCITY"][:, solno]
T = data["/T"][solno]
F = data["/Force_Amp"][solno]
par = data["/Parameters"]
par = json.loads(par[()])
data.close()

# do time simulation
nperiod = par["shooting"]["single"]["nperiod"]
nsteps = par["shooting"]["single"]["nsteps_per_period"]

Duffing.forcing_parameters(par)
t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
X = np.concatenate([pose, vel])
timesol = np.array(odeint(Duffing.model_ode, X, t, args=(T * nperiod, F), rtol=1e-8, tfirst=True))
pose_time = timesol[:, 0]
vel_time = timesol[:, 1]

# plot
f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10, 4))
f.suptitle(f"Frequency = {1 / T:.3f} Hz, Force Amplitude = {F:.3f}")
a1.set(xlabel="Time (s)", ylabel="Position (m)")
a1.plot(t, pose_time, "-")
a2.set(xlabel="Time (s)", ylabel="Velocity (m/s)")
a2.plot(t, vel_time, "-g")
a3.set(xlabel="Position (m)", ylabel="Velocity (m/s)")
a3.plot(pose_time, vel_time, "-k")

plt.tight_layout()
plt.show()
