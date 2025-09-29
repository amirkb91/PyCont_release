import h5py
import sys
import yaml
import numpy as np
from scipy.integrate import odeint
from cubic_spring import Cubic_Spring
import matplotlib.pyplot as plt

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

# do time simulation
par = data["Parameters"]
par = yaml.safe_load(par[()])

# Extract parameters
nperiod = 3  # Number of periods to simulate
nsteps = par["shooting"]["steps_per_period"]
t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
# Prepare initial conditions and perform time simulation
X = np.concatenate([inc, vel])

# Update the model for forced simulation if needed
Cubic_Spring.update_model(par)

# Perform time integration using the cubic spring model
rtol = par["shooting"]["integration_tolerance"]
timesol = np.array(odeint(Cubic_Spring.model_ode, X, t, args=(T, F), rtol=rtol, tfirst=True))
inc_time = timesol[:, :2]
vel_time = timesol[:, 2:]

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

a1.set_xlabel(r"Time (s)", fontsize=14)
a1.set_ylabel(r"Position (m)", fontsize=14)
a2.set_xlabel(r"Position DoF 1 (m)", fontsize=14)
a2.set_ylabel(r"Position DoF 2 (m)", fontsize=14)
a3.set_xlabel(r"Time (s)", fontsize=14)
a3.set_ylabel(r"Velocity (m/s)", fontsize=14)


a1.plot(t, inc_time[:, 0], "-", label=r"DoF 1", linewidth=1.5)
a1.plot(t, inc_time[:, 1], "--", label=r"DoF 2", linewidth=1.5)
a2.plot(inc_time[:, 0], inc_time[:, 1], "-", linewidth=1.5)
a3.plot(t, vel_time[:, 0], "-", label=r"DoF 1", linewidth=1.5)
a3.plot(t, vel_time[:, 1], "--", label=r"DoF 2", linewidth=1.5)


a1.legend([r"DoF 1", r"DoF 2"], loc="upper left", fontsize=12)
a3.legend([r"DoF 1", r"DoF 2"], loc="upper left", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to leave space for the suptitle

# Print simulation info
print(f"Simulated {nperiod} periods with {nsteps} steps per period")
print(f"Total simulation time: {T * nperiod:.4f} seconds")

# Save the plot in vector format (uncomment if needed)
# plt.savefig(f'timesim_sol{solno}_freq{1/T:.3f}Hz.pdf', format='pdf', dpi=300)

plt.show()
