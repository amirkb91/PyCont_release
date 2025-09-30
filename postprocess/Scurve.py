import matplotlib.pyplot as plt
import sys
import h5py
import yaml
import numpy as np
import mplcursors


# show point data on figure
def show_annotation(sel, offsets):
    ind = int(sel.index)
    global_index = ind + offsets[sel.artist]
    sel.annotation.set_text(f"index:{global_index}")


dof_to_plot = 0
normalise_force = 1.0
normalise_amp = 1.0

files = sys.argv[1:]
if not files:
    print("Usage: python Scurve.py <file1.h5> [file2.h5] ...")
    sys.exit(1)

# Ensure files have .h5 extension
for i, file in enumerate(files):
    if not file.endswith(".h5"):
        files[i] += ".h5"

plt.style.use("ggplot")
f, a = plt.subplots(figsize=(10, 7))
a.set(xlabel="Forcing Amplitude", ylabel="Max Position Amplitude")

# Optimized color cycle for different files
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = (colors * ((len(files) // len(colors)) + 1))[: len(files)]

# Plot solutions
line_objects = []  # Collect all line objects for cursor interaction
offsets = {}  # Store offsets for each segment
total_points = 0  # Total number of points plotted

for file, color in zip(files, color_cycle):
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
    F = data["Force_Amp"][:]
    par = data["Parameters"]
    par = yaml.safe_load(par[()])
    forced = "force" in par["continuation"]["parameter"]

    # Optimized amplitude calculation using vectorized operations
    n_solpoints = len(F)
    amp = np.max(np.abs(inc_time[dof_to_plot, :, :]), axis=0) / normalise_amp

    if forced and "Bifurcation/Stability" in data:
        stability = data["Bifurcation/Stability"][:]
        stable_index = np.where(np.diff(stability))[0] + 1

        # Create segments based on stability transitions
        if len(stable_index) == 0:
            segments = [np.arange(len(F))]
        else:
            stable_index = stable_index[stable_index < len(F)]
            segments = np.split(np.arange(len(F)), stable_index)

        for i, seg in enumerate(segments):
            linestyle = "solid" if stability[seg[0]] else "dashed"
            if i > 0 and len(seg) > 0:
                # Include overlap point for continuity
                seg = np.insert(seg, 0, seg[0] - 1)

            force_data = F[seg] / normalise_force
            amp_data = amp[seg]

            (line,) = a.plot(
                force_data,
                amp_data,
                marker="none",
                linestyle=linestyle,
                color=color,
                label=file.split(".h5")[0] if i == 0 else "",
            )
            offsets[line] = total_points
            total_points += len(seg) - (1 if i > 0 else 0)
            line_objects.append(line)

    else:
        force_data = F / normalise_force
        (line,) = a.plot(
            force_data,
            amp,
            marker="none",
            linestyle="solid",
            color=color,
            label=file.split(".h5")[0],
        )
        offsets[line] = total_points
        total_points += len(F)
        line_objects.append(line)

    data.close()

a.legend()

# Set up the cursor for all line objects
cursor = mplcursors.cursor(line_objects, hover=False)
cursor.connect("add", lambda sel: show_annotation(sel, offsets))

plt.draw()
plt.show()
