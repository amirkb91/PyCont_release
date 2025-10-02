import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py
import yaml
import numpy as np


dof_to_plot = 0
normalise_freq = 1.0
normalise_force = 1.0
normalise_amp = 1.0

files = sys.argv[1:]
if not files:
    print("Usage: python FRC_Scurve_3D.py <file1.h5> [file2.h5] ...")
    sys.exit(1)

# Ensure files have .h5 extension
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

    # Check if Config_Time group exists
    if "Config_Time" not in data:
        data.close()
        print(f"Error: Config_Time data not found in {file}")
        print(
            "Please run 'python timesim_branch.py <solution_file>' first to generate time-domain data"
        )
        print("This postprocessing script requires time simulation data to calculate amplitudes")
        sys.exit(1)

    inc_time = data["Config_Time/INC"][:]

    par = data["Parameters"]
    par = yaml.safe_load(par[()])
    continuation_parameter = par["continuation"]["parameter"]

    if continuation_parameter in ["force_freq", "period"]:
        color = frc_color
    elif continuation_parameter == "force_amp":
        color = scurve_color

    T = data["T"][:]
    freq = 1 / (T * normalise_freq)  # Convert period to frequency
    F = data["Force_Amp"][:]

    # Calculate amplitude
    amp = np.max(np.abs(inc_time[dof_to_plot, :, :]), axis=0) / normalise_amp

    # Pre-calculate normalized data for plotting
    freq_norm = freq
    force_norm = F / normalise_force

    # Plot the 3D curve (solid lines only, no stability info)
    ax.plot(
        freq_norm,
        force_norm,
        amp,
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{file.split('.h5')[0]} ({continuation_parameter})",
    )

    data.close()

# Customize the plot
ax.grid(True, alpha=0.3)

# Set viewing angle for better visualization
ax.view_init(elev=20, azim=-70)

# Add title and improve layout
plt.title("3D Response Surface: Frequency vs Force vs Amplitude", fontsize=14, pad=20)
plt.tight_layout()

print(f"Plotted data from {len(files)} file(s)")
print("Blue: Frequency continuation (FRC), Orange: Amplitude continuation (S-curves)")
plt.show()
