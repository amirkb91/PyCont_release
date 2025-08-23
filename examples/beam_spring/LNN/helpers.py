import h5py
import pickle

import numpy as np
import scipy.linalg as spl

import matplotlib.pyplot as plt


def check_data(pose, vel, acc, force):
    # Beam paremeters for 2 mode system
    k_nl = 4250000
    w_1 = 91.734505484821950
    w_2 = 3.066194429903638e02
    zeta_1 = 0.03
    zeta_2 = 0.09
    phi_L = np.array([[-7.382136522799137, 7.360826867549465]])

    # Modal Matrices
    M = np.eye(2)
    C = np.array([[2 * zeta_1 * w_1, 0], [0, 2 * zeta_2 * w_2]])
    K = np.array([[w_1**2, 0], [0, w_2**2]])
    Minv = spl.inv(M)

    # Calculate acceleration
    calc_acc = Minv @ (-K @ pose.reshape(2, -1, order="F") - C @ vel.reshape(2, -1, order="F") -
                       k_nl * phi_L.T @ ((phi_L @ pose.reshape(2, -1, order="F"))**3) + force.reshape(2, -1, order="F"))

    assert np.abs(acc.reshape(2, -1, order="F") - calc_acc).all(
    ) < 1e-6, "Calculated acceleration does not match provided acceleration data."


def save_to_file(filename='frequency_step_frequency_', path='results/phys', start=10.0, stop=24.0, step=0.2, check=True):
    """Store simulation data for ML models
    """
    # Store ML data
    ml_data = {}

    # Loop over all files in directory
    for i in np.arange(10.0, 24.1, 0.2):
        # Open new file
        file = f"{path}/{filename}{i:.03f}.h5"

        # -------------------------- READ CONTINUATION FILES
        # NOTE: COL -> Number of Periodic Solutions, ROW -> Number of Solution Points
        data = h5py.File(str(file), "r")
        pose = data["/Config_Time/POSE"][:].squeeze()
        vel = data["/Config_Time/VELOCITY"][:].squeeze()
        acc = data["/Config_Time/ACCELERATION"][:].squeeze()
        time = data["/Config_Time/Time"][:].T
        F = data["/Force_Amp"][:]
        T = data["/T"][:]
        # Compute total force
        _force = F * np.sin(2 * np.pi / T * time)
        force = np.concatenate(
            (_force[np.newaxis, :, :], np.zeros_like(_force[np.newaxis, :, :])), axis=0)
        # Close file
        data.close()

        # Check data
        if check:
            check_data(pose, vel, acc, force)

        # ------------------------------- COLLECT DATA
        ml_data[f"{i:.03f}"] = {
            "pose": pose,
            "vel": vel,
            "acc": acc,
            "time": time,
            "F": F,
            "T": T,
            "force": force if check else _force
        }

    # --------------------------------- SAVE DATA
    with open(f"{path}/data.pkl", "wb") as f:
        pickle.dump(ml_data, f)

    # ------------------------------- STATS
    print(f"Data saved to {path}/data.pkl")
    print(f"Number of files: {len(ml_data)}\n")
    print("---EXAMPLE SHAPES---")
    print(f"pose: {pose.shape}, vel: {vel.shape}, acc: {acc.shape}")
    print("If MODAL: 2 Modes, 301 time steps per 39 points along curve")
    print("If PHYSICAL: 301 time steps per 39 points along curve")
    print(
        f"time: {time.shape}, F: {F.shape}, T: {T.shape}, force: {force.shape if check else _force.shape}")

    return ml_data


def create_phys_training_data(ml_data, path, split=0.2, seed=42):
    with open(f"{path}/data.pkl", "rb") as f:
        ml_data = pickle.load(f)

        # Collect data shapes
        n_curves = len(ml_data.keys())
        x_train, dx_train, ddx_train, force_train = [], [], [], []
        x_test, dx_test, ddx_test, force_test = [], [], [], []

        # Create datasets
        for k, v in ml_data.items():
            assert v["pose"].shape == v["vel"].shape == v["acc"].shape == v[
                "force"].shape, f"Data shape mismatch for key {k}"

            n_time_steps_per_point = v["time"].shape[0]
            n_points_per_curve = v["time"].shape[1]

            # Split data into training and testing sets
            rng = np.random.default_rng(seed=seed)
            shuffle = rng.permutation(n_points_per_curve)
            seed += 1  # Increment seed for next shuffle

            train_indices = shuffle[:int(n_points_per_curve * (1 - split))]
            test_indices = shuffle[int(n_points_per_curve * (1 - split)):]

            _x_train = v["pose"][:, train_indices]
            _dx_train = v["vel"][:, train_indices]
            _ddx_train = v["acc"][:, train_indices]
            _force_train = v["force"][:, train_indices]

            _x_test = v["pose"][:, test_indices]
            _dx_test = v["vel"][:, test_indices]
            _ddx_test = v["acc"][:, test_indices]
            _force_test = v["force"][:, test_indices]

            # Append to training data
            # Reshape: Full time series per point in order
            x_train.append(_x_train.flatten(order="F"))
            dx_train.append(_dx_train.flatten(order="F"))
            ddx_train.append(_ddx_train.flatten(order="F"))
            force_train.append(_force_train.flatten(order="F"))

            # Append to testing data
            # Reshape: Full time series per point in order
            x_test.append(_x_test.flatten(order="F"))
            dx_test.append(_dx_test.flatten(order="F"))
            ddx_test.append(_ddx_test.flatten(order="F"))
            force_test.append(_force_test.flatten(order="F"))

        # Convert lists to numpy arrays
        x_train = np.concatenate(x_train, axis=-1)
        dx_train = np.concatenate(dx_train, axis=-1)
        ddx_train = np.concatenate(ddx_train, axis=-1)
        force_train = np.concatenate(force_train, axis=-1)
        x_test = np.concatenate(x_test, axis=-1)
        dx_test = np.concatenate(dx_test, axis=-1)
        ddx_test = np.concatenate(ddx_test, axis=-1)
        force_test = np.concatenate(force_test, axis=-1)

        # Collect data
        train_data = np.concatenate(
            (x_train[:, np.newaxis], dx_train[:, np.newaxis],
             ddx_train[:, np.newaxis], force_train[:, np.newaxis]), axis=1
        )
        test_data = np.concatenate(
            (x_test[:, np.newaxis], dx_test[:, np.newaxis],
             ddx_test[:, np.newaxis], force_test[:, np.newaxis]), axis=1
        )

        info = {
            "train_data_shape": train_data.shape,
            "test_data_shape": test_data.shape,
            "n_curves": n_curves,
            "Shapes": "[Samples, Features: x, dx, ddx, t, F, T]",
            "qmax": x_train.max(),
            "qmin": x_train.min(),
            "qdmax": dx_train.max(),
            "qdmin": dx_train.min(),
            "qddmax": ddx_train.max(),
            "qddmin": ddx_train.min(),
        }

        # Print Stats
        print(
            f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
        print("Samples, [x, dx, ddx, force]")
        print(
            f"x_train.shape: {x_train.shape}, dx_train.shape: {dx_train.shape}, ddx_train.shape: {ddx_train.shape}, force_train.shape: {force_train.shape}")
        print(
            f"x_test.shape: {x_test.shape}, dx_test.shape: {dx_test.shape}, ddx_test.shape: {ddx_test.shape}, force_test.shape: {force_test.shape}")

    return train_data, test_data, info


def create_modal_training_data(ml_data, path, split=0.2, seed=42):
    with open(f"{path}/data.pkl", "rb") as f:
        ml_data = pickle.load(f)

        # Collect data shapes
        n_curves = len(ml_data.keys())
        x_train, dx_train, ddx_train, force_train = [], [], [], []
        x_test, dx_test, ddx_test, force_test = [], [], [], []

        # Create datasets
        for k, v in ml_data.items():
            assert v["pose"].shape == v["vel"].shape == v["acc"].shape == v[
                "force"].shape, f"Data shape mismatch for key {k}"

            n_time_steps_per_point = v["time"].shape[0]
            n_points_per_curve = v["time"].shape[1]

            # Split data into training and testing sets
            rng = np.random.default_rng(seed=seed)
            shuffle = rng.permutation(n_points_per_curve)
            seed += 1  # Increment seed for next shuffle

            train_indices = shuffle[:int(n_points_per_curve * (1 - split))]
            test_indices = shuffle[int(n_points_per_curve * (1 - split)):]

            _x_train = v["pose"][:, :, train_indices]
            _dx_train = v["vel"][:, :, train_indices]
            _ddx_train = v["acc"][:, :, train_indices]
            _force_train = v["force"][:, :, train_indices]

            _x_test = v["pose"][:, :, test_indices]
            _dx_test = v["vel"][:, :, test_indices]
            _ddx_test = v["acc"][:, :, test_indices]
            _force_test = v["force"][:, :, test_indices]

            # Append to training data
            # Reshape: Full time series per point in order
            x_train.append(_x_train.reshape(2, -1, order="F"))
            dx_train.append(_dx_train.reshape(2, -1, order="F"))
            ddx_train.append(_ddx_train.reshape(2, -1, order="F"))
            force_train.append(_force_train.reshape(2, -1, order="F"))

            # Append to testing data
            # Reshape: Full time series per point in order
            x_test.append(_x_test.reshape(2, -1, order="F"))
            dx_test.append(_dx_test.reshape(2, -1, order="F"))
            ddx_test.append(_ddx_test.reshape(2, -1, order="F"))
            force_test.append(_force_test.reshape(2, -1, order="F"))

        # Convert lists to numpy arrays
        x_train = np.concatenate(x_train, axis=-1).swapaxes(0, 1)
        dx_train = np.concatenate(dx_train, axis=-1).swapaxes(0, 1)
        ddx_train = np.concatenate(ddx_train, axis=-1).swapaxes(0, 1)
        force_train = np.concatenate(force_train, axis=-1).swapaxes(0, 1)
        x_test = np.concatenate(x_test, axis=-1).swapaxes(0, 1)
        dx_test = np.concatenate(dx_test, axis=-1).swapaxes(0, 1)
        ddx_test = np.concatenate(ddx_test, axis=-1).swapaxes(0, 1)
        force_test = np.concatenate(force_test, axis=-1).swapaxes(0, 1)

        # Collect data
        train_data = np.concatenate(
            (x_train[:, :, np.newaxis], dx_train[:, :, np.newaxis],
             ddx_train[:, :, np.newaxis], force_train[:, :, np.newaxis]), axis=-1
        )
        test_data = np.concatenate(
            (x_test[:, :, np.newaxis], dx_test[:, :, np.newaxis],
             ddx_test[:, :, np.newaxis], force_test[:, :, np.newaxis]), axis=-1
        )

        info = {
            "train_data_shape": train_data.shape,
            "test_data_shape": test_data.shape,
            "n_curves": n_curves,
            "Shapes": "[Samples, # of Modes, Features: x, dx, ddx, force]",
            "q1max": x_train[:, 0].max(),
            "q1min": x_train[:, 0].min(),
            "qd1max": dx_train[:, 0].max(),
            "qd1min": dx_train[:, 0].min(),
            "q2max": x_train[:, 1].max(),
            "q2min": x_train[:, 1].min(),
            "qd2max": dx_train[:, 1].max(),
            "qd2min": dx_train[:, 1].min(),
        }

        # Print Stats
        print(
            f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
        print("Samples, # of Modes, [x, dx, ddx, force]")
        print(
            f"x_train.shape: {x_train.shape}, dx_train.shape: {dx_train.shape}, ddx_train.shape: {ddx_train.shape}, force_train.shape: {force_train.shape}")
        print(
            f"x_test.shape: {x_test.shape}, dx_test.shape: {dx_test.shape}, ddx_test.shape: {ddx_test.shape}, force_test.shape: {force_test.shape}")

    return train_data, test_data, info


def plot_S_curves(ml_data, modal=True):
    f, a = plt.subplots(figsize=(10, 10))
    a.set(xlabel="Forcing Amplitude", ylabel="Max Position Amplitude")

    # Color cycle for different files
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = color_cycle * ((len(ml_data.keys()) // len(color_cycle)) + 1)

    # Plot solutions
    line_objects = []  # Collect all line objects for cursor interaction
    offsets = {}  # Store offsets for each segment
    total_points = 0  # Total number of points plotted

    for data, color in zip(ml_data.items(), color_cycle):
        k, v = data
        pose_time = v["pose"]
        if not modal:
            pose_time = pose_time[np.newaxis, :, :]
        F = v["F"]

        n_solpoints = len(F)
        amp = np.zeros(n_solpoints)
        for i in range(n_solpoints):
            amp[i] = np.max(np.abs(pose_time[0, :, i])) / 1.0

        (line,) = a.plot(
            F / 1.0,
            amp,
            marker="none",
            linestyle="solid",
            color=color,
            label=k,
        )
        offsets[line] = total_points
        total_points += len(F)
        line_objects.append(line)

    a.legend(ncols=2, bbox_to_anchor=(1.0, 1.0))


def plot_3DS_curves(ml_data, modal=True):
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Forcing Frequency (Hz)")
    ax.set_ylabel("Forcing Amplitude")
    ax.set_zlabel("Max Position Amplitude")

    # Plot solutions from all files
    for k, v in ml_data.items():
        pose_time = v["pose"]
        if not modal:
            pose_time = pose_time[np.newaxis, :, :]

        T = v["T"]
        freq = 1 / (T * 1.0)  # Convert period to frequency
        F = v["F"]

        n_solpoints = len(F)
        amp = np.zeros(n_solpoints)
        for i in range(n_solpoints):
            amp[i] = np.max(np.abs(pose_time[0, :, i])) / 1.0

        # Plot the 3D curve (solid lines only, no stability info)
        ax.plot(freq, F / 1.0, amp, linestyle="-", linewidth=2)

    # Customize the plot
    ax.grid(True, alpha=0.3)

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=-70)

    plt.title("3D Response Surface")
    plt.tight_layout()
