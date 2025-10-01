import h5py
import sys
import yaml
import numpy as np
from duffing import Duffing
import matplotlib.pyplot as plt

"""
Visualises the time-domain dynamics of a single solution point from a continuation branch,
displaying position, velocity, and acceleration as interactive plots.
"""

plt.rcParams["text.usetex"] = True

# Get solution index from user input
solno = int(input("Solution Index: "))

# Read solution file
if len(sys.argv) < 2:
    print("Usage: python timesim_point.py <solution_file>")
    sys.exit(1)

file = sys.argv[-1]
if not file.endswith(".h5"):
    file += ".h5"

try:
    data = h5py.File(str(file), "r")

    # Check if the solution index exists
    max_solutions = data["T"].shape[0]
    if solno >= max_solutions:
        print(
            f"Error: Solution index {solno} out of range. Available solutions: 0 to {max_solutions-1}"
        )
        data.close()
        sys.exit(1)

    # Read solution data
    inc = data["Config/INC"][:, solno]
    vel = data["Config/VEL"][:, solno]
    T = data["T"][solno]
    F = data["Force_Amp"][solno]

    print(f"Loaded solution {solno}: T={T:.4f}s, F={F:.4f}N, Frequency={1/T:.4f}Hz")

except (FileNotFoundError, KeyError) as e:
    print(f"Error reading file {file}: {e}")
    sys.exit(1)

# Extract parameters and do time simulation
par = data["Parameters"]
par = yaml.safe_load(par[()])

# Update the model for forced simulation if needed
Duffing.update_model(par)

# Call time simulation
X = np.concatenate([inc, vel])
_inc, _vel, _acc = Duffing.time_simulate(F, T, X, par)
t = np.linspace(0, T, _inc.shape[0])

# Close the HDF5 file
data.close()

# Plot results
f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5))
f.suptitle(
    r"Time Simulation - Frequency = " + f"{(1 / T):.3f} Hz, " + f"Force = {F:.3f} N",
    fontsize=16,
    weight="bold",
)

# Common settings for all axes
for ax in (a1, a2, a3):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=12)

a1.set(xlabel="Time (s)", ylabel="Position (m)")
a1.plot(t, _inc, "-")
a2.set(xlabel="Time (s)", ylabel="Velocity (m/s)")
a2.plot(t, _vel, "-g")
a3.set(xlabel="Position (m)", ylabel="Velocity (m/s)")
a3.plot(_inc, _vel, "-k")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to leave space for the suptitle

# Save the plot in vector format (uncomment if needed)
# plt.savefig(f'timesim_sol{solno}_freq{1/T:.3f}Hz.pdf', format='pdf', dpi=300)

plt.show()
