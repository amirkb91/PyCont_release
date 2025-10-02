import matplotlib.pyplot as plt
import mplcursors
import sys
import h5py
import numpy as np


# Show detailed point data on figure
def show_annotation(sel):
    ind = int(sel.index)
    # Get the line object to extract data
    line = sel.artist
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    if ind < len(x_data) and ind < len(y_data):
        energy_val = x_data[ind]
        freq_val = y_data[ind]
        sel.annotation.set_text(f"Point {ind}\nEnergy: {energy_val:.2e} J\nFreq: {freq_val:.3f} Hz")
    else:
        sel.annotation.set_text(f"Point {ind}")


# Usage validation
if len(sys.argv) < 2:
    print("Usage: python FEP.py <file1.h5> [file2.h5] ... [y]")
    print("  Add 'y' as last argument to save plot as FEP.pdf")
    sys.exit(1)

plt.style.use("ggplot")
files = sys.argv[1:]

# Check for save flag
if files and files[-1] == "y":
    saveflag = True
    files = files[:-1]
else:
    saveflag = False

# Ensure files have .h5 extension
for i, file in enumerate(files):
    if not file.endswith(".h5"):
        files[i] += ".h5"

# Figure properties with improved layout
f, (a1, a2) = plt.subplots(1, 2, figsize=(12, 7))
f.subplots_adjust(left=0.1, right=0.95, wspace=0.3)

# Enhanced plot setup
a1.set(xlabel="Energy (J)", ylabel="Frequency (Hz)", xscale="log")
a1.grid(True, alpha=0.3)
a1.set_title("Frequency-Energy Plot (FEP)", fontsize=12)

a2.set(xlabel="Continuation step", ylabel="beta (deg)", ylim=(-5, 185))
a2.grid(True, alpha=0.3)
a2.set_title("Angle between Tangent Vectors", fontsize=12)

# Plot solutions with robust error handling
line_objects_a1 = []  # Collect all line objects for cursor interaction in a1
successful_files = []

for file in files:
    try:
        # Load data with proper error handling
        with h5py.File(str(file), "r") as data:
            # Check for required datasets
            if "T" not in data:
                print(f"Warning: T dataset not found in {file}, skipping")
                continue
            if "Energy" not in data:
                print(f"Warning: Energy dataset not found in {file}, skipping")
                continue

            T = data["T"][:]
            Energy = data["Energy"][:]
            beta = []

            # Check for optional beta data
            if "beta" in data:
                beta = data["beta"][:].T
            else:
                print(f"Note: No beta data found in {file}")

            # Calculate frequency from period
            freq = 1 / T

            # Plot FEP with enhanced styling
            file_label = file.split(".h5")[0]
            (line_a1,) = a1.plot(
                Energy,
                freq,
                marker=".",
                fillstyle="none",
                label=file_label,
                linewidth=1.5,
                markersize=4,
            )

            # Mark starting point
            a1.plot(
                Energy[0],
                freq[0],
                marker="x",
                fillstyle="full",
                markersize=8,
                markeredgewidth=2,
                color=line_a1.get_color(),
            )

            line_objects_a1.append(line_a1)
            successful_files.append(file_label)

            # Plot beta if available
            if len(beta) > 0:
                a2.plot(
                    range(len(beta)),
                    beta,
                    marker=".",
                    fillstyle="none",
                    linewidth=1.5,
                    markersize=3,
                    label=file_label,
                    color=line_a1.get_color(),
                )

    except FileNotFoundError:
        print(f"Error: File {file} not found, skipping")
        continue
    except Exception as e:
        print(f"Error loading data from {file}: {e}, skipping")
        continue

# Add legends and final formatting
if line_objects_a1:
    a1.legend(loc="best")

    # Set up the cursor for all line objects in a1
    cursor_a1 = mplcursors.cursor(line_objects_a1, hover=False)
    cursor_a1.connect("add", show_annotation)

    # Add legend to a2 if there are beta plots
    if a2.get_lines():
        a2.legend(loc="best")
else:
    print("Warning: No valid data found in any files")

# Add overall title
f.suptitle("Frequency-Energy Plot Analysis", fontsize=14)

# Console output
print(f"Successfully processed {len(successful_files)} file(s): {', '.join(successful_files)}")
if line_objects_a1:
    print("Interactive features: Click on FEP curves to see data point indices")
if saveflag:
    print("Saving plot as FEP.pdf")

# Save and display
if saveflag:
    plt.savefig("FEP.pdf", dpi=300, bbox_inches="tight")

plt.tight_layout()
plt.show()
