import h5py
import sys
import json
import numpy as np
from scipy.integrate import odeint
from examples.cubic_spring.cubic_spring import Cubic_Spring
import matplotlib.pyplot as plt

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

# do time simulation
par = data["/Parameters"]
par = json.loads(par[()])
method = par["shooting"]["method"]
if method == "single":
    nperiod = par["shooting"]["single"]["nperiod"]
    nsteps = par["shooting"]["single"]["nsteps_per_period"]
    t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
    X = np.concatenate([pose, vel])
    timesol = np.array(odeint(Cubic_Spring.system_ode, X, t, rtol=1e-8, tfirst=True))
    pose_time = timesol[:, :2]
    vel_time = timesol[:, 2:]

# plot
f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10, 4))
f.suptitle(f"Frequency = {1 / T:.3f} Hz")
a1.set(xlabel="Time (s)", ylabel="Position (m)")
a1.plot(t, pose_time[:, 0], "-", label="DoF 1")
a1.plot(t, pose_time[:, 1], "--", label="DoF 2")
a1.legend()
a2.set(xlabel="Position DoF 1 (m)", ylabel="Position DoF 2 (m)")
a2.plot(pose_time[:, 0], pose_time[:, 1], "-")
a3.set(xlabel="Time (s)", ylabel="Velocity (m/s)")
a3.plot(t, vel_time[:, 0], "-", label="DoF 1")
a3.plot(t, vel_time[:, 1], "--", label="DoF 2")
a3.legend()

plt.tight_layout()
plt.show()
