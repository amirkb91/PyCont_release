import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py
import json
import numpy as np


config2plot = 0
normalise_freq = 1.0
normalise_force = 1.0
normalise_amp = 1.0

files = sys.argv[1:]
for i, file in enumerate(files):
    if not file.endswith(".h5"):
        files[i] += ".h5"

# Create 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("Forcing Frequency (Hz)")
ax.set_ylabel("Forcing Amplitude")
ax.set_zlabel("Max Position Amplitude")

# Define colors for different continuation types
frc_color = "blue"  # Frequency continuation (FRC)
scurve_color = "orange"  # Amplitude continuation (S-curves)

# Plot solutions from all files
for file in files:
    data = h5py.File(str(file), "r")
    pose_time = data["/Config_Time/POSE"][:]

    par = data["/Parameters"]
    par = json.loads(par[()])
    forced = par["continuation"]["forced"]
    continuation_parameter = par["continuation"]["continuation_parameter"]

    if continuation_parameter == "frequency":
        color = frc_color
    elif continuation_parameter == "amplitude":
        color = scurve_color

    # Check if we have both T and Force_Amp data
    if "/T" in data.keys():
        T = data["/T"][:]
        freq = 1 / (T * normalise_freq)  # Convert period to frequency
    else:
        print(f"Warning: No /T data found in {file}")
        continue

    if "/Force_Amp" in data.keys():
        F = data["/Force_Amp"][:]
    else:
        print(f"Warning: No /Force_Amp data found in {file}")
        continue

    n_solpoints = len(F)
    amp = np.zeros(n_solpoints)
    for i in range(n_solpoints):
        amp[i] = np.max(np.abs(pose_time[config2plot, :, i])) / normalise_amp

    # Plot the 3D curve (solid lines only, no stability info)
    ax.plot(freq, F / normalise_force, amp, color=color, linestyle="-", linewidth=2)

    data.close()

# Customize the plot
ax.grid(True, alpha=0.3)

# Set viewing angle for better visualization
ax.view_init(elev=20, azim=-70)

plt.title("3D Response Surface")
plt.tight_layout()
plt.show()
