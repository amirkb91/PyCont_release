import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import sys
import h5py

dof_to_plot = 0
normalise_freq = 1.0
normalise_amp = 1.0
normalise_force = 1.0

if len(sys.argv) < 2:
    print("Usage: python Floquet.py <file.h5>")
    sys.exit(1)

file = sys.argv[1]
if not file.endswith(".h5"):
    file += ".h5"

# Create figure with 3D subplot
f = plt.figure(figsize=(15, 7))
a1 = f.add_subplot(1, 2, 1)
a2 = f.add_subplot(1, 2, 2, projection="3d")
f.subplots_adjust(left=0.10, right=0.95, wspace=0.2)

# Set up Floquet multiplier plot (left)
a1.set(xlabel="Real", ylabel="Imaginary")
a1.axis("square")
a1.grid(True, alpha=0.3)

# Draw unit circle for stability reference
theta = np.linspace(0, 2 * np.pi, 1000)
a1.plot(np.cos(theta), np.sin(theta), "-", color="black", alpha=0.5, linewidth=1)
a1.set_xlim([-2, 2])
a1.set_ylim([-2, 2])

# Set up 3D response surface plot (right)
a2.set_xlabel("Frequency [Hz]")
a2.set_ylabel("Force Amplitude")
a2.set_zlabel("Response Amplitude")
a2.grid(True, alpha=0.3)

# Load data with proper error handling
try:
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

    # Check for required datasets
    if "Config_Time/INC" not in data:
        print(f"Error: Config_Time/INC dataset not found in {file}")
        sys.exit(1)
    if "T" not in data:
        print(f"Error: T dataset not found in {file}")
        sys.exit(1)
    if "Bifurcation/Floquet" not in data:
        print(f"Error: Bifurcation/Floquet dataset not found in {file}")
        sys.exit(1)

    inc_time = data["Config_Time/INC"][:]
    T = data["T"][:] * normalise_freq
    floquet = data["Bifurcation/Floquet"][:]

    # Check for Force_Amp data for 3D plot
    if "Force_Amp" in data:
        F = data["Force_Amp"][:]
    else:
        print("Warning: No Force_Amp data found, using dummy values")
        F = np.ones_like(T)  # Dummy force amplitude data

except FileNotFoundError:
    print(f"Error: File {file} not found")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data from {file}: {e}")
    sys.exit(1)

n_solpoints = len(T)

# Optimized amplitude calculation using vectorized operations
amp = np.max(np.abs(inc_time[dof_to_plot, :, :]), axis=0) / normalise_amp

# Calculate frequency from period
freq = 1 / T

# Plot 3D response surface
a2.plot(
    freq,
    F / normalise_force,
    amp,
    color="green",
    linestyle="-",
    linewidth=2,
    alpha=0.8,
    label="Response Surface",
)

# Initialize Floquet multiplier points with proper stability coloring
floq_points = []
for i in range(floquet.shape[0]):
    # Determine initial color based on stability
    initial_color = "red" if np.abs(floquet[i, 0]) > 1 else "green"
    (point,) = a1.plot(
        floquet.real[i, 0],
        floquet.imag[i, 0],
        "o",
        markersize=6,
        markerfacecolor=initial_color,
        markeredgecolor=initial_color,
        alpha=0.8,
    )
    floq_points.append(point)

# Initialize 3D response point (start with first solution)
(famp_points,) = a2.plot(
    [freq[0]], [F[0] / normalise_force], [amp[0]], "o", markersize=8, color="red", alpha=0.9
)

# Create slider with better positioning
ax = f.add_axes([0.05, 0.15, 0.025, 0.63])
slider = Slider(
    ax=ax,
    label="Sol no.",
    valmin=1,
    valmax=n_solpoints,
    valinit=1,
    valstep=1,
    orientation="vertical",
    color="lightblue",
)


# Optimized update function
def update(val):
    sol_no_display = int(slider.val)  # 1-based for display
    sol_no_index = sol_no_display - 1  # 0-based for array indexing

    # Update Floquet multipliers with efficient vectorized operations
    current_floquet = floquet[:, sol_no_index]
    magnitudes = np.abs(current_floquet)

    for i, point in enumerate(floq_points):
        # Use set_data for single point updates (expects sequences)
        point.set_data([current_floquet.real[i]], [current_floquet.imag[i]])
        color = "red" if magnitudes[i] > 1 else "green"
        point.set_markerfacecolor(color)
        point.set_markeredgecolor(color)

    # Update 3D response point
    famp_points.set_data_3d(
        [freq[sol_no_index]], [F[sol_no_index] / normalise_force], [amp[sol_no_index]]
    )

    # Update plot title with current solution info
    a1.set_title(f"Floquet Multipliers (Sol {sol_no_display}/{n_solpoints})", fontsize=12)

    f.canvas.draw_idle()


slider.on_changed(update)

# Add plot titles for better understanding
a1.set_title("Floquet Multipliers (Unit Circle)", fontsize=12)
a2.set_title("3D Response Surface", fontsize=12)
f.suptitle(f"Floquet Analysis: {file.split('.h5')[0]}", fontsize=14)

# Set optimal viewing angle for 3D plot
a2.view_init(elev=20, azim=-70)
a2.legend(loc="upper left")

print(f"Loaded Floquet analysis from: {file}")
print(f"Number of solution points: {n_solpoints}")
print("Left plot - Floquet Multipliers: Green=Stable, Red=Unstable")
print("Right plot - 3D Response Surface: Frequency vs Force vs Amplitude")
print("Use the slider to navigate through solution points")
print("Red dot shows current solution point on 3D surface")

try:
    plt.show()
finally:
    # Ensure proper cleanup
    data.close()
