import h5py
import sys
import json
import numpy as np
from beam_spring import Beam_Spring
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
nsteps = par["shooting"]["single"]["nsteps_per_period"]
t = np.linspace(0, T, nsteps + 1)
X = np.concatenate([np.zeros(2), vel])
[_, _, pose_time, vel_time, _, _, _] = Beam_Spring.time_solve(
    1.0, F, T, X, pose, par, fulltime=True
)

# Plot
f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5))
f.suptitle(r"Frequency = " + f"{(1 / T):.3f} Hz", fontsize=16, weight="bold")

# Common settings for all axes
for ax in (a1, a2, a3):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=12)

a1.set_xlabel(r"Time (s)", fontsize=14)
a1.set_ylabel(r"Modal Coordinate (m)", fontsize=14)
a2.set_xlabel(r"Modal Coordinate 1 (m)", fontsize=14)
a2.set_ylabel(r"Modal Coordinate 2 (m)", fontsize=14)
a3.set_xlabel(r"Time (s)", fontsize=14)
a3.set_ylabel(r"Modal Velocity (m/s)", fontsize=14)

a1.plot(t, pose_time[:, 0], "-", label=r"Mode 1", linewidth=1.5)
a1.plot(t, pose_time[:, 1], "--", label=r"Mode 2", linewidth=1.5)
a2.plot(pose_time[:, 0], pose_time[:, 1], "-", linewidth=1.5)
a3.plot(t, vel_time[:, 0], "-", label=r"Mode 1", linewidth=1.5)
a3.plot(t, vel_time[:, 1], "--", label=r"Mode 2", linewidth=1.5)

a1.legend([r"Mode 1", r"Mode 2"], loc="upper left", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Convert modal coordinates to physical displacement using mode shapes
# Physical displacement at tip: u_tip = phi_L @ q = phi_L[0,0]*q1 + phi_L[0,1]*q2
phi_L = Beam_Spring.phi_L  # Mode shapes at tip location
physical_disp = phi_L @ pose_time.T
physical_disp = physical_disp.flatten()

# Physical velocity at tip: u_tip_dot = phi_L @ q_dot
physical_vel = phi_L @ vel_time.T
physical_vel = physical_vel.flatten()

# New figure for physical displacement
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(r"Physical Displacement at Beam Tip", fontsize=16, weight="bold")

# Common settings for all axes
for ax in (ax1, ax2, ax3):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=12)

# Time history of physical displacement
ax1.set_xlabel(r"Time (s)", fontsize=14)
ax1.set_ylabel(r"Displacement (m)", fontsize=14)
ax1.plot(t, physical_disp, "b-", linewidth=2)

# Phase portrait: physical displacement vs velocity
ax2.set_xlabel(r"Displacement (m)", fontsize=14)
ax2.set_ylabel(r"Velocity (m/s)", fontsize=14)
ax2.plot(physical_disp, physical_vel, "r-", linewidth=1.5)
ax2.plot(physical_disp[0], physical_vel[0], "go", markersize=8, label="Start")
ax2.plot(physical_disp[-1], physical_vel[-1], "ro", markersize=8, label="End")
ax2.legend(fontsize=12)

# Modal contribution to physical displacement
ax3.set_xlabel(r"Time (s)", fontsize=14)
ax3.set_ylabel(r"Modal Contributions (m)", fontsize=14)
mode1_contribution = phi_L[0, 0] * pose_time[:, 0]  # phi_L[0,0] * q1
mode2_contribution = phi_L[0, 1] * pose_time[:, 1]  # phi_L[0,1] * q2
ax3.plot(t, mode1_contribution, "-", label=f"Mode 1", linewidth=1.5)
ax3.plot(t, mode2_contribution, "--", label=f"Mode 2", linewidth=1.5)
ax3.plot(t, physical_disp, "k-", label="Total", linewidth=2, alpha=0.8)
ax3.legend(fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
