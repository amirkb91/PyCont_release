import matplotlib.pyplot as plt
import mplcursors
import sys
import h5py


# show point data on figure
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

# figure properties
f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 7))
a1.set(xlabel="Energy (J)", ylabel="Frequency (Hz)", xscale="log")
a2.set(xlabel="Continuation step", ylabel="beta (deg)", ylim=(-5, 185))

# plot sols
line = []
for file in files:
    # load data
    data = h5py.File(str(file), "r")
    T = data["/T"][:]
    Energy = data["/Energy"][:]
    beta = []
    if "beta" in data:
        beta = data["/beta"][:].T

    # plot FEP and beta
    line.append(a1.plot(Energy, 1 / T, marker=".", fillstyle="none", label=file.split(".h5")[0]))
    a1.plot(Energy[0], 1 / T[0], marker="x", fillstyle="full")
    a2.plot(range(len(beta)), beta, marker=".", fillstyle="none")
a1.legend()

# Cursor and plt
cursor = mplcursors.cursor(line[0], hover=False)
cursor.connect("add", show_annotation)
if saveflag:
    plt.savefig("FEP.pdf")
plt.draw()
plt.show()
