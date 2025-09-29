import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
import os


class Logger:
    def __init__(self, parameters):
        self.parameters = parameters
        self.store_index = 0
        self.linewidth = 14

        # Use dictionaries for cleaner data storage with dynamic keys
        self.solution_data = {
            "X": [],
            "T": [],
            "F": [],
            "tgt": [],
            "energy": [],
            "beta": [],
            "itercorrect": [],
            "step": [],
        }

        # Plotting setup
        self.plot = parameters["logger"]["enable_live_plot"]
        self.betaplot = parameters["continuation"]["method"] == "pseudo_arclength"

        if self.plot:
            self._setup_plotting()

        # HDF5 file setup - initialise file and datasets
        self.h5_filename = f"{parameters['logger']['output_file_name']}.h5"
        self._initialise_h5_file()

        # Screen output tracking
        self._last_header_iter = -1

    def _setup_plotting(self):
        """Initialise matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(11, 9))
        self.gs = GridSpec(2, 2)
        self.ax = np.array([])
        self.ln = []

    def _initialise_h5_file(self):
        """Initialise HDF5 file with proper structure and metadata."""
        # Remove existing file if it exists
        if os.path.exists(self.h5_filename):
            os.remove(self.h5_filename)

        with h5py.File(self.h5_filename, "w") as f:
            # Store parameters immediately (they don't change during continuation)
            f.create_dataset("Parameters", data=yaml.dump(self.parameters, sort_keys=False))

            # Flag to track if datasets have been created
            self._h5_datasets_created = False

    def store(self, **sol_data):
        """Store solution data efficiently."""
        self.store_index += 1

        # Store data in memory using mapping
        data_mapping = {
            "sol_X": "X",
            "sol_T": "T",
            "sol_F": "F",
            "sol_tgt": "tgt",
            "sol_energy": "energy",
            "sol_beta": "beta",
            "sol_itercorrect": "itercorrect",
            "sol_step": "step",
        }

        for key, value in sol_data.items():
            if key in data_mapping:
                # Avoid unnecessary copying - store reference directly
                self.solution_data[data_mapping[key]].append(value)

        # Append to HDF5 file efficiently
        self._append_to_h5()

        # Update plots if enabled
        if self.plot:
            self.solplot()

    def _append_to_h5(self):
        """Dynamically append current solution point to HDF5 file."""
        with h5py.File(self.h5_filename, "a") as f:
            current_idx = self.store_index - 1  # 0-based indexing

            # Create datasets on first call when we know the data dimensions
            if not self._h5_datasets_created:
                self._create_h5_datasets(f)
            else:
                # Resize all datasets to accommodate the new data point
                new_size = self.store_index

                # Resize scalar datasets
                for dataset_name in ["T", "Force_Amp", "Energy", "beta", "itercorrect", "step"]:
                    if dataset_name in f:
                        f[dataset_name].resize((new_size,))

                # Resize array datasets
                if "Config" in f:
                    for config_dataset in f["Config"].keys():
                        current_shape = f[f"Config/{config_dataset}"].shape
                        f[f"Config/{config_dataset}"].resize((current_shape[0], new_size))

                if "Tangent" in f:
                    current_shape = f["Tangent"].shape
                    f["Tangent"].resize((current_shape[0], new_size))

            # Append data
            if self.solution_data["X"]:
                X = self.solution_data["X"][-1]  # Latest data point
                n = len(X) // 2
                f["Config/INC"][:, current_idx] = X[:n]
                f["Config/VEL"][:, current_idx] = X[n:]

            # Scalar data
            if self.solution_data["T"]:
                f["T"][current_idx] = self.solution_data["T"][-1]
            if self.solution_data["F"]:
                f["Force_Amp"][current_idx] = self.solution_data["F"][-1]
            if self.solution_data["energy"]:
                f["Energy"][current_idx] = self.solution_data["energy"][-1]
            if self.solution_data["beta"]:
                f["beta"][current_idx] = self.solution_data["beta"][-1]
            if self.solution_data["itercorrect"]:
                f["itercorrect"][current_idx] = self.solution_data["itercorrect"][-1]
            if self.solution_data["step"]:
                f["step"][current_idx] = self.solution_data["step"][-1]

            # Array data
            if self.solution_data["tgt"]:
                f["Tangent"][:, current_idx] = self.solution_data["tgt"][-1]

    def _create_h5_datasets(self, f):
        """Create HDF5 datasets starting with size 1 and unlimited maxshape."""
        if not self.solution_data["X"]:
            return

        X = self.solution_data["X"][0]
        n_dof = len(X) // 2

        # Tangent vector always has size X plus 1 continuation parameter
        tgt_size = len(X) + 1

        # Create extensible datasets starting with size 1
        config_group = f.create_group("Config")

        # Configuration arrays (n_dof x 1) with unlimited growth
        config_group.create_dataset("INC", (n_dof, 1), maxshape=(n_dof, None), dtype=np.float64)
        config_group.create_dataset("VEL", (n_dof, 1), maxshape=(n_dof, None), dtype=np.float64)

        # Tangent array (tgt_size x 1) with unlimited growth
        f.create_dataset("Tangent", (tgt_size, 1), maxshape=(tgt_size, None), dtype=np.float64)

        # Scalar datasets (size 1) with unlimited growth
        f.create_dataset("T", (1,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("Force_Amp", (1,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("Energy", (1,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("beta", (1,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("itercorrect", (1,), maxshape=(None,), dtype=np.int32)
        f.create_dataset("step", (1,), maxshape=(None,), dtype=np.float64)

        self._h5_datasets_created = True

    def screenout(self, **screen_data):
        """Optimised screen output with cleaner formatting."""
        # Define column mappings and formatting
        columns = {
            "Iter Cont": ("iter", "{}"),
            "Iter Corr": ("correct", "{}"),
            "Residual": ("res", "{:.4e}"),
            "Freq": ("freq", "{:.4f}"),
            "Amp": ("amp", "{:.4f}"),
            "Energy": ("energy", "{:.4e}"),
            "Step": ("step", "{:.3e}"),
            "Beta": ("beta", "{:.4f}"),
        }

        # Build formatted row
        row_data = {}
        itercont = screen_data.get("iter", 0)
        itercorr = screen_data.get("correct", 0)

        for col_name, (key, fmt) in columns.items():
            if key in screen_data:
                row_data[col_name] = fmt.format(screen_data[key]).ljust(self.linewidth)
            else:
                row_data[col_name] = " ".ljust(self.linewidth)

        # Print header every 20 continuation iterations
        if itercont % 20 == 0 and itercorr == 0 and itercont != self._last_header_iter:
            print("\n")
            print(*[col.ljust(self.linewidth) for col in columns.keys()], sep="")
            self._print_line("=")
            self._last_header_iter = itercont

        # Print data row
        print(*row_data.values(), sep="")

    def _print_line(self, char: str):
        """Print separator line."""
        print(char * 8 * self.linewidth)

    def screenline(self, char: str):
        """Print separator line (public interface)."""
        self._print_line(char)

    def solplot(self):
        """Optimised real-time plotting with better performance."""
        if not self.solution_data["energy"]:
            return

        # Get data arrays efficiently
        Energy = np.array(self.solution_data["energy"])
        T = np.array(self.solution_data["T"]) if self.solution_data["T"] else np.array([])
        Amp = np.array(self.solution_data["F"]) if self.solution_data["F"] else np.array([])
        beta = np.array(self.solution_data["beta"]) if self.solution_data["beta"] else np.array([])

        # Determine continuation type
        is_amplitude_continuation = self.parameters["continuation"]["parameter"] == "force_amp"

        # Initialise plots on first call
        if not self.ax.any():
            self._initialise_plots(Energy, T, Amp, beta, is_amplitude_continuation)
        else:
            self._update_plots(Energy, T, Amp, beta, is_amplitude_continuation)

    def _initialise_plots(self, Energy, T, Amp, beta, is_amplitude_continuation):
        """Initialise matplotlib plots with proper setup."""
        # Create subplots
        if self.betaplot:
            self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[:, 0]))
            self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[0, 1]))
            self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[1, 1]))
        else:
            self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[:, 0]))
            self.ax = np.append(self.ax, self.fig.add_subplot(self.gs[0, 1]))

        # Configure main plot
        self._setup_main_plot(Energy, T, Amp, is_amplitude_continuation)

        # Configure zoom plot
        self._setup_zoom_plot(Energy, T, Amp, is_amplitude_continuation)

        # Configure beta plot if needed
        if self.betaplot and len(beta) > 0:
            self._setup_beta_plot(beta)

        plt.pause(0.01)

    def _setup_main_plot(self, Energy, T, Amp, is_amplitude_continuation):
        """Setup main energy plot."""
        ax = self.ax[0]
        ax.grid()
        ax.set_xscale("log")
        ax.set_xlabel("Energy (J)")
        ax.ticklabel_format(useOffset=False, axis="y")
        ax.set_xlim(1e-4, 1e6)
        ax.set_ylim(
            self.parameters["continuation"]["min_parameter_value"],
            self.parameters["continuation"]["max_parameter_value"],
        )

        if is_amplitude_continuation and len(Amp) > 0:
            ax.set_ylabel("Forcing Amplitude (N)")
            line = ax.plot(Energy, Amp, marker=".", fillstyle="none")[0]
        elif len(T) > 0:
            ax.set_ylabel("Frequency (Hz)")
            line = ax.plot(Energy, 1 / T, marker=".", fillstyle="none")[0]
        else:
            return

        self.ln.append([line])

    def _setup_zoom_plot(self, Energy, T, Amp, is_amplitude_continuation):
        """Setup zoomed energy plot."""
        ax = self.ax[1]
        ax.grid()
        ax.set_xscale("log")
        ax.set_xlabel("Energy (J)")
        ax.ticklabel_format(useOffset=False, axis="y")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

        if is_amplitude_continuation and len(Amp) > 0:
            ax.set_ylabel("Forcing Amplitude (N)")
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2e}"))
            line = ax.plot(Energy[-10:], Amp[-10:], marker=".", fillstyle="none", color="green")[0]
        elif len(T) > 0:
            ax.set_ylabel("Frequency (Hz)")
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
            line = ax.plot(Energy[-10:], 1 / T[-10:], marker=".", fillstyle="none", color="green")[
                0
            ]
        else:
            return

        self.ln.append([line])

    def _setup_beta_plot(self, beta):
        """Setup beta angle plot."""
        ax = self.ax[2]
        ax.grid()
        ax.set_xlabel("Continuation Step")
        ax.set_ylabel("beta (deg)")
        ax.set_xlim(1, 10)
        ax.set_ylim(-5, 185)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

        line = ax.plot(range(1, len(beta) + 1), beta, marker=".", fillstyle="none", color="red")[0]
        self.ln.append([line])

    def _update_plots(self, Energy, T, Amp, beta, is_amplitude_continuation):
        """Update existing plots with new data."""
        if len(self.ln) == 0:
            return

        # Update main plot
        if is_amplitude_continuation and len(Amp) > 0:
            self.ln[0][0].set_data(Energy, Amp)
        elif len(T) > 0:
            self.ln[0][0].set_data(Energy, 1 / T)

        # Update zoom plot
        if len(self.ln) > 1:
            if is_amplitude_continuation and len(Amp) > 0:
                self.ln[1][0].set_data(Energy[-10:], Amp[-10:])
            elif len(T) > 0:
                self.ln[1][0].set_data(Energy[-10:], 1 / T[-10:])

            self.ax[1].relim()
            self.ax[1].autoscale()

        # Update beta plot
        if self.betaplot and len(self.ln) > 2 and len(beta) > 0:
            self.ln[2][0].set_data(range(1, len(beta) + 1), beta)
            beta_xaxis = 10
            self.ax[2].set_xlim(1, beta_xaxis * np.ceil(len(beta) / beta_xaxis))

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
