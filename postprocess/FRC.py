import matplotlib.pyplot as plt
import sys
import h5py
import json
import numpy as np
import mplcursors


# Show point data on figure
def show_annotation(sel, offsets):
    ind = int(sel.index)
    global_index = ind + offsets[sel.artist]
    sel.annotation.set_text(f"index:{global_index}")


config2plot = 0
normalise_freq = 1.0
normalise_amp = 1.0

files = sys.argv[1:]
for i, file in enumerate(files):
    if not file.endswith(".h5"):
        files[i] += ".h5"

plt.style.use("ggplot")
f, a = plt.subplots(figsize=(10, 7))
a.set(xlabel="F/\u03C9\u2099", ylabel="Normalised Position")

# Color cycle for different files
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = color_cycle * ((len(files) // len(color_cycle)) + 1)

# Plot solutions
line_objects = []  # Collect all line objects for cursor interaction
offsets = {}  # Store offsets for each segment
total_points = 0  # Total number of points plotted

for file, color in zip(files, color_cycle):
    data = h5py.File(str(file), "r")
    pose_time = data["/Config_Time/POSE"][:]
    T = data["/T"][:]
    par = data["/Parameters"]
    par = json.loads(par[()])
    forced = par["continuation"]["forced"]

    n_solpoints = len(T)
    amp = np.zeros(n_solpoints)
    for i in range(n_solpoints):
        amp[i] = np.max(np.abs(pose_time[config2plot, :, i])) / normalise_amp

    if forced and "/Bifurcation/Stability" in data.keys():
        stability = data["/Bifurcation/Stability"][:]
        stable_index = np.argwhere(np.diff(stability)).squeeze() + 1

        # Handle the case where there are no stability transitions
        if stable_index.size == 0:
            segments = [np.arange(len(T))]
        else:
            # Ensure the segments are joined properly
            stable_index = stable_index[stable_index < len(T)]
            segments = np.split(np.arange(len(T)), stable_index)

        for i, seg in enumerate(segments):
            linestyle = "solid" if stability[seg[0]] else "dashed"
            if i > 0:
                # Include the overlap point between segments
                seg = np.insert(seg, 0, seg[0] - 1)
            (line,) = a.plot(
                1 / (T[seg] * normalise_freq),
                amp[seg],
                marker="none",
                linestyle=linestyle,
                color=color,
                label=file.split(".h5")[0] if i == 0 else "",
            )
            offsets[line] = total_points
            total_points += len(seg) - 1  # Correcting for the overlap point
            line_objects.append(line)

    else:
        (line,) = a.plot(
            1 / (T * normalise_freq),
            amp,
            marker="none",
            linestyle="solid",
            color=color,
            label=file.split(".h5")[0],
        )
        offsets[line] = total_points
        total_points += len(T)
        line_objects.append(line)

a.legend()

# Set up the cursor for all line objects
cursor = mplcursors.cursor(line_objects, hover=False)
cursor.connect("add", lambda sel: show_annotation(sel, offsets))

plt.draw()
plt.show()
