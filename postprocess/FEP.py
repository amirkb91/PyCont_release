import matplotlib.pyplot as plt
import mplcursors
import sys
import h5py


# Show point data on figure
def show_annotation(sel):
    ind = int(sel.index)
    sel.annotation.set_text(f"index:{ind}")


plt.style.use("ggplot")
files = sys.argv[1:]
if files[-1] == "y":
    saveflag = True
    files = files[:-1]
else:
    saveflag = False
for i, file in enumerate(files):
    if not file.endswith(".h5"):
        files[i] += ".h5"

# Figure properties
f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 7))
a1.set(xlabel="Energy (J)", ylabel="Frequency (Hz)", xscale="log")
a2.set(xlabel="Continuation step", ylabel="beta (deg)", ylim=(-5, 185))

# Plot solutions
line_objects_a1 = []  # Collect all line objects for cursor interaction in a1

for file in files:
    # Load data
    data = h5py.File(str(file), "r")
    T = data["/T"][:]
    Energy = data["/Energy"][:]
    beta = []
    if "beta" in data:
        beta = data["/beta"][:].T

    # Plot FEP
    (line_a1,) = a1.plot(Energy, 1 / T, marker=".", fillstyle="none", label=file.split(".h5")[0])
    a1.plot(Energy[0], 1 / T[0], marker="x", fillstyle="full")
    line_objects_a1.append(line_a1)

    # Plot beta
    a2.plot(range(len(beta)), beta, marker=".", fillstyle="none")

a1.legend()

# Set up the cursor for all line objects in a1
cursor_a1 = mplcursors.cursor(line_objects_a1, hover=False)
cursor_a1.connect("add", show_annotation)

if saveflag:
    plt.savefig("FEP.pdf")
plt.draw()
plt.show()
