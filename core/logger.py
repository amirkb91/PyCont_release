import numpy as np
import h5py
import json
import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker


class Logger:
    def __init__(self, prob):
        self.prob = prob
        self.store_index = 0
        self.sol_X = []
        self.sol_T = []
        self.sol_tgt = []
        self.sol_pose = []
        self.sol_vel = []
        self.sol_energy = []
        self.sol_beta = []
        self.sol_itercorrect = []
        self.sol_step = []
        self.plot = False
        self.betaplot = False

        if prob.cont_params["Logger"]["plot"]:
            self.plot = True
            self.fig = plt.figure(figsize=(11, 9))
            self.gs = GridSpec(2, 2)
            self.ax = np.array([])
            self.ln = []
        if self.prob.cont_params["continuation"]["method"] == "psa":
            self.betaplot = True
        if self.prob.cont_params["shooting"]["method"] == "single":
            self.npartition = 1
        elif self.prob.cont_params["shooting"]["method"] == "multiple":
            self.npartition = self.prob.cont_params["shooting"]["multiple"]["npartition"]

    def store(self, **sol_data):
        self.store_index += 1
        for key, _value in sol_data.items():
            value = copy.copy(_value)
            if key == "sol_pose":
                self.sol_pose.append(value.flatten(order="F"))
            elif key == "sol_vel":
                self.sol_vel.append(value.flatten(order="F"))
            elif key == "sol_T":
                self.sol_T.append(value)
            elif key == "sol_tgt":
                self.sol_tgt.append(value)
            elif key == "sol_energy":
                self.sol_energy.append(value)
            elif key == "sol_beta":
                self.sol_beta.append(value)
            elif key == "sol_itercorrect":
                self.sol_itercorrect.append(value)
            elif key == "sol_step":
                self.sol_step.append(value)

        # save to disk and plot if required
        self.savetodisk()
        if self.plot:
            self.solplot()

    def screenout(self, **screen_data):
        width = 14
        screen = dict.fromkeys(
            ["Iter Cont", "Iter Corr", "Residual", "Freq", "Energy", "Step", "Beta"],
            " ".ljust(width),
        )
        header = list(screen.keys())
        printborder = False
        iterprinted = None

        for key, value in screen_data.items():
            if key == "iter":
                screen["Iter Cont"] = f"{value}".ljust(width)
                itercont = value
            elif key == "correct":
                screen["Iter Corr"] = f"{value}".ljust(width)
                itercorr = value
            elif key == "res":
                screen["Residual"] = f"{value:.4e}".ljust(width)
            elif key == "freq":
                screen["Freq"] = f"{value:.4f}".ljust(width)
            elif key == "energy":
                screen["Energy"] = f"{value:.4e}".ljust(width)
            elif key == "step":
                screen["Step"] = f"{value:.3e}".ljust(width)
            elif key == "beta":
                screen["Beta"] = f"{value:.4f}".ljust(width)
                printborder = True

        if np.mod(itercont, 20) == 0 and itercorr == 0 and itercont != iterprinted:
            itercont = iterprinted
            print("\n")
            print(*[f"{x}".ljust(width) for x in header], sep="")
            print("=" * len(header) * width)

        screen_vals = list(screen.values())
        print(*screen_vals, sep="")
        if printborder:
            print("-" * len(header) * width)

    def savetodisk(self):
        savefile = h5py.File(self.prob.cont_params["Logger"]["file_name"] + ".h5", "w")
        savefile["/Config/POSE"] = np.asarray(self.sol_pose).T
        savefile["/Config/VELOCITY"] = np.asarray(self.sol_vel).T
        savefile["/T"] = np.asarray(self.sol_T).T
        savefile["/Tangent"] = np.asarray(self.sol_tgt).T
        savefile["/Energy"] = np.asarray(self.sol_energy).T
        savefile["/beta"] = np.asarray(self.sol_beta).T
        savefile["/itercorrect"] = np.asarray(self.sol_itercorrect).T
        savefile["/step"] = np.asarray(self.sol_step).T
        savefile["/Parameters"] = json.dumps(self.prob.cont_params)
        savefile.close()

    def solplot(self):
        Energy = np.asarray(self.sol_energy)
        T = np.asarray(self.sol_T)
        beta = np.asarray(self.sol_beta)
        beta_xaxis = 10

        if not self.ax.any():
            if self.betaplot:
                self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[:, 0]))
                self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[0, 1]))
                self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[1, 1]))
            else:
                self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[:, 0]))
                self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[0, 1]))
            # Frequency energy plot
            self.ax[0].grid()
            self.ax[0].set_xscale("log")
            self.ax[0].set_xlabel("Energy (J)")
            self.ax[0].set_ylabel("Frequency (Hz)")
            self.ax[0].ticklabel_format(useOffset=False, axis="y")
            self.ax[0].set_xlim(1e-4, self.prob.cont_params["continuation"]["Emax"])
            self.ax[0].set_ylim(
                self.prob.cont_params["continuation"]["fmin"],
                self.prob.cont_params["continuation"]["fmax"],
            )
            self.ln.append(self.ax[0].plot(Energy, 1 / T, marker=".", fillstyle="none"))
            # Frequency energy plot zoom
            self.ax[1].grid()
            self.ax[1].set_xscale("log")
            self.ax[1].set_xlabel("Energy (J)")
            self.ax[1].set_ylabel("Frequency (Hz)")
            self.ax[1].ticklabel_format(useOffset=False, axis="y")
            self.ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
            self.ax[1].xaxis.set_minor_formatter(ticker.ScalarFormatter())
            self.ax[1].xaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:.1f}"))
            self.ax[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
            self.ln.append(
                self.ax[1].plot(Energy, 1 / T, marker=".", fillstyle="none", color="green")
            )
            # beta plot
            if self.betaplot:
                self.ax[2].grid()
                self.ax[2].set_xlabel("Continuation Step")
                self.ax[2].set_ylabel("beta (deg)")
                self.ax[2].set_xlim(1, beta_xaxis)
                self.ax[2].set_ylim(-5, 185)
                self.ax[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
                self.ln.append(
                    self.ax[2].plot(
                        range(1,
                              len(beta) + 1), beta, marker=".", fillstyle="none", color="red"
                    )
                )
            plt.pause(0.01)
        else:
            self.ln[0][0].set_data(Energy, 1 / T)
            self.ln[1][0].set_data(Energy[-10:], 1 / T[-10:])
            self.ax[1].relim()
            self.ax[1].autoscale()
            if self.betaplot:
                self.ln[2][0].set_data(range(1, len(beta) + 1), beta)
                self.ax[2].set_xlim(1, beta_xaxis * np.ceil(len(beta) / beta_xaxis))
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
