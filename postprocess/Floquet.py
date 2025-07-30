import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import sys
import h5py

config2plot = 0
normalise_freq = 1.0
normalise_amp = 1.0

file = sys.argv[1]
if not file.endswith(".h5"):
    file += ".h5"

f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 7))
f.subplots_adjust(left=0.15, right=0.95, wspace=0.3)
a1.set(xlabel="Real", ylabel="Imaginary")
a2.set(xlabel="F/\u03C9\u2099", ylabel="Normalised Amplitude")
a1.axis("square")
a1.grid("on")
a2.grid("on")
a1.plot(np.cos(np.linspace(0, 2 * np.pi, 1000)), np.sin(np.linspace(0, 2 * np.pi, 1000)), "-")
a1.set_xlim([-1.5, 1.5])
a1.set_ylim([-1.5, 1.5])

data = h5py.File(str(file), "r")
pose_time = data["/Config_Time/POSE"][:]
T = data["/T"][:] * normalise_freq
floquet = data["/Bifurcation/Floquet"][:]

n_solpoints = len(T)
amp = np.zeros(n_solpoints)
for i in range(n_solpoints):
    amp[i] = np.max(np.abs(pose_time[config2plot, :, i])) / normalise_amp
a2.plot(1 / T, amp, marker="none", fillstyle="none", color="green")

floq_points = []
for i in range(floquet.shape[0]):
    (point,) = a1.plot(
        floquet.real[i, 0],
        floquet.imag[i, 0],
        "o",
        markersize=5,
        markerfacecolor="green",
        markeredgecolor="green",
    )
    floq_points.append(point)
(famp_points,) = a2.plot(1 / T[0], amp[0], "o", markersize=5, color="black")

ax = f.add_axes([0.05, 0.15, 0.0225, 0.63])
slider = Slider(
    ax=ax,
    label="Sol no.",
    valmin=0,
    valmax=n_solpoints - 1,
    valinit=0,
    valstep=1,
    orientation="vertical",
    color="orange",
)


def update(val):
    sol_no = int(slider.val)
    for i, point in enumerate(floq_points):
        point.set_xdata(floquet.real[i, sol_no])
        point.set_ydata(floquet.imag[i, sol_no])
        color = "red" if np.abs(floquet[i, sol_no]) > 1 else "green"
        point.set_markerfacecolor(color)
        point.set_markeredgecolor(color)

    famp_points.set_xdata(1 / T[sol_no])
    famp_points.set_ydata(amp[sol_no])

    f.canvas.draw_idle()


slider.on_changed(update)
plt.show()
