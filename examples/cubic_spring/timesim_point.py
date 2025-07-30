import h5py
import sys
import json
import numpy as np
from scipy.integrate import odeint
from cubic_spring import Cubic_Spring
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

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

# do time simulation
par = data["/Parameters"]
par = json.loads(par[()])
method = par["shooting"]["method"]
if method == "single":
    nperiod = par["shooting"]["single"]["nperiod"]
    nsteps = par["shooting"]["single"]["nsteps_per_period"]
    t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
    X = np.concatenate([pose, vel])
    timesol = np.array(
        odeint(Cubic_Spring.model_ode, X, t, args=(T * nperiod, F), rtol=1e-8, tfirst=True)
    )
    pose_time = timesol[:, :2]
    vel_time = timesol[:, 2:]
elif method == "multiple":
    npartition = par["shooting"]["multiple"]["npartition"]
    nsteps = par["shooting"]["multiple"]["nsteps_per_partition"]
    delta_S = 1 / npartition
    pose = pose.reshape(2, npartition, order="F")
    vel = vel.reshape(2, npartition, order="F")
    print("Partition Timestamps (s):")
    t = np.zeros([nsteps + 1, 1, npartition])
    pose_time = np.zeros([nsteps + 1, 2, npartition])
    vel_time = np.zeros([nsteps + 1, 2, npartition])
    for ipart in range(npartition):
        partition_starttime = ipart * T * delta_S
        print(f"{partition_starttime:.3f}")
        t_part = np.linspace(0, T * delta_S, nsteps + 1)
        t[:, :, ipart] = t_part.reshape(-1, 1) + partition_starttime
        X = np.concatenate([pose[:, ipart], vel[:, ipart]])
        timesol = np.array(odeint(Cubic_Spring.model_ode, X, t_part, rtol=1e-8, tfirst=True))
        pose_time[:, :, ipart] = timesol[:, :2]
        vel_time[:, :, ipart] = timesol[:, 2:]

    # t = t.flatten(order="F")
    # pose_time = np.reshape(np.transpose(pose_time, (2, 0, 1)), (-1, 2))
    # vel_time = np.reshape(np.transpose(vel_time, (2, 0, 1)), (-1, 2))

# Plot
f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5))
f.suptitle(r"Frequency = " + f"{(1 / T):.3f} Hz", fontsize=16, weight="bold")

# Common settings for all axes
for ax in (a1, a2, a3):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=12)

a1.set_xlabel(r"Time (s)", fontsize=14)
a1.set_ylabel(r"Position (m)", fontsize=14)
a2.set_xlabel(r"Position DoF 1 (m)", fontsize=14)
a2.set_ylabel(r"Position DoF 2 (m)", fontsize=14)
a3.set_xlabel(r"Time (s)", fontsize=14)
a3.set_ylabel(r"Velocity (m/s)", fontsize=14)

if method == "single":
    a1.plot(t, pose_time[:, 0], "-", label=r"DoF 1", linewidth=1.5)
    a1.plot(t, pose_time[:, 1], "--", label=r"DoF 2", linewidth=1.5)
    a2.plot(pose_time[:, 0], pose_time[:, 1], "-", linewidth=1.5)
    a3.plot(t, vel_time[:, 0], "-", label=r"DoF 1", linewidth=1.5)
    a3.plot(t, vel_time[:, 1], "--", label=r"DoF 2", linewidth=1.5)
elif method == "multiple":
    for ipart in range(npartition):
        partition_starttime = ipart * T * delta_S
        color = plt.cm.turbo(ipart / npartition)
        a1.plot(t[:, 0, ipart], pose_time[:, 0, ipart], "-", color=color, linewidth=1.5)
        a1.plot(t[:, 0, ipart], pose_time[:, 1, ipart], "--", color=color, linewidth=1.5)
        a2.plot(pose_time[:, 0, ipart], pose_time[:, 1, ipart], "-", color=color, linewidth=1.5)
        a3.plot(t[:, 0, ipart], vel_time[:, 0, ipart], "-", color=color, linewidth=1.5)
        a3.plot(t[:, 0, ipart], vel_time[:, 1, ipart], "--", color=color, linewidth=1.5)
        # np.savetxt(f"partition_{ipart}_t.txt", t[:, 0, ipart])
        # np.savetxt(f"partition_{ipart}_pose_time.txt", pose_time[:, :, ipart])
        # np.savetxt(f"partition_{ipart}_vel_time.txt", vel_time[:, :, ipart])


a1.legend([r"DoF 1", r"DoF 2"], loc="upper left", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to leave space for the suptitle

# Save the plot in vector format
# plt.savefig('plot_vector_format.pdf', format='pdf', dpi=300)

plt.show()
