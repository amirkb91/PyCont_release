import sys
import h5py
import json
from alive_progress import alive_bar
import shutil
import numpy as np

from core.problem import Prob
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint
from beam_spring_lnn import Beam_Spring

from postprocess.bifurcation import bifurcation_functions


def run(config_file="contparameters.json", pred_acc=None):
    Beam_Spring.acc(pred_acc)

    # Problem
    prob = Prob()
    prob.read_contparams(config_file)
    prob.add_doffunction(Beam_Spring.get_fe_data)
    prob.add_icfunction(Beam_Spring.eigen_solve)
    prob.add_zerofunction(Beam_Spring.time_solve)

    # Initialise forcing parameters if continuation is forced
    Beam_Spring.forcing_parameters(prob.cont_params)

    # Continuation starting point
    start = StartingPoint(prob)
    start.get_startingpoint()

    # Logger
    log = Logger(prob)

    # Solve continuation on problem
    con = ConX(prob, start, log)
    con.solve()


def time_sim_branch(file, inplace="-i", run_bif="n", store_physical="n"):
    """ Run time simulations for all points on solution branch and store """

    # run_bif = input("Compute bifurcation functions (y/n)? ")
    if run_bif not in ("y", "n"):
        raise Exception("Input not valid")

    # store_physical = input(
    #     "Store physical displacement at tip (y) or modal coordinates (n)? ")
    if store_physical not in ("y", "n"):
        raise Exception("Input not valid")

    # read solution file
    # file = sys.argv[1]
    if inplace == "-i":
        inplace = True
    else:
        inplace = False

    if not file.endswith(".h5"):
        file += ".h5"

    data = h5py.File(str(file), "r")
    pose = data["/Config/POSE"][:]
    vel = data["/Config/VELOCITY"][:]
    T = data["/T"][:]
    F = data["/Force_Amp"][:]
    par = data["/Parameters"]
    par = json.loads(par[()])
    data.close()

    # create new file to store time histories or append inplace
    if inplace:
        new_file = file
    else:
        new_file = file.strip(".h5") + "_withtime.h5"
        shutil.copy(file, new_file)

    n_solpoints = len(T)
    nsteps = par["shooting"]["single"]["nsteps_per_period"]

    # Determine storage dimensions based on choice
    if store_physical == "y":
        # For physical displacement: 1 DOF (tip displacement)
        n_dof_store = 1
        storage_label = "Physical Tip Displacement"
    else:
        # For modal coordinates: 2 DOF (modal coordinates)
        n_dof_store = np.shape(pose)[0]
        storage_label = "Modal Coordinates"

    pose_time = np.zeros([n_dof_store, nsteps + 1, n_solpoints])
    vel_time = np.zeros([n_dof_store, nsteps + 1, n_solpoints])
    acc_time = np.zeros([n_dof_store, nsteps + 1, n_solpoints])
    time = np.zeros([n_solpoints, nsteps + 1])

    print(f"Storage configuration: {storage_label}")

    # run sims
    Beam_Spring.forcing_parameters(par)
    if run_bif == "y":
        Floquet = np.zeros([4, n_solpoints], dtype=np.complex128)
        Stability = np.zeros(n_solpoints)
        Fold = np.zeros(n_solpoints)
        Flip = np.zeros(n_solpoints)
        Neimark_Sacker = np.zeros(n_solpoints)

    with alive_bar(n_solpoints) as bar:
        for i in range(n_solpoints):
            # Initial conditions for 2-DOF system: [pos1, pos2, vel1, vel2]
            X = np.array([0.0, 0.0, vel[0, i], vel[1, i]])
            [_, J, pose_time_series, vel_time_series, acc_time_series, _, _] = Beam_Spring.time_solve(
                1.0, F[i], T[i], X, pose[:, i], par, fulltime=True
            )

            if store_physical == "y":
                # Convert modal coordinates to physical displacement at tip
                phi_L = Beam_Spring.phi_L  # Mode shapes at tip location

                # Transform modal coordinates to physical displacement: u_tip = phi_L @ q
                # Shape: (1, nsteps+1)
                physical_pose = phi_L @ pose_time_series.T
                # Shape: (1, nsteps+1)
                physical_vel = phi_L @ vel_time_series.T

                # For acceleration: a_tip = phi_L @ q_ddot
                # Shape: (1, nsteps+1)
                physical_acc = phi_L @ acc_time_series.T

                # Store physical displacement data
                pose_time[:, :, i] = physical_pose
                vel_time[:, :, i] = physical_vel
                acc_time[:, :, i] = physical_acc
            else:
                # Store modal coordinates directly
                pose_time[:, :, i] = pose_time_series.T
                vel_time[:, :, i] = vel_time_series.T
                acc_time[:, :, i] = acc_time_series.T

            if run_bif == "y":
                M = J[:, :-1] + np.eye(4)
                bifurcation_out = bifurcation_functions(M)
                Floquet[:, i] = bifurcation_out[0]
                Stability[i] = bifurcation_out[1]
                Fold[i] = bifurcation_out[2]
                Flip[i] = bifurcation_out[3]
                Neimark_Sacker[i] = bifurcation_out[4]
            time[i, :] = np.linspace(0, T[i], nsteps + 1)
            bar()

    # write to file
    time_data = h5py.File(new_file, "a")

    # Add metadata about coordinate system
    if "/Config_Time/Coordinate_System" in time_data.keys():
        del time_data["/Config_Time/Coordinate_System"]
    if store_physical == "y":
        coordinate_info = "Physical displacement at beam tip"
    else:
        coordinate_info = "Modal coordinates"
    time_data["/Config_Time/Coordinate_System"] = coordinate_info

    if "/Config_Time/POSE" in time_data.keys():
        del time_data["/Config_Time/POSE"]
    if "/Config_Time/VELOCITY" in time_data.keys():
        del time_data["/Config_Time/VELOCITY"]
    if "/Config_Time/ACCELERATION" in time_data.keys():
        del time_data["/Config_Time/ACCELERATION"]
    if "/Config_Time/Time" in time_data.keys():
        del time_data["/Config_Time/Time"]
    time_data["/Config_Time/POSE"] = pose_time
    time_data["/Config_Time/VELOCITY"] = vel_time
    time_data["/Config_Time/ACCELERATION"] = acc_time
    time_data["/Config_Time/Time"] = time
    if run_bif == "y":
        if "/Bifurcation/Floquet" in time_data.keys():
            del time_data["/Bifurcation/Floquet"]
        if "/Bifurcation/Stability" in time_data.keys():
            del time_data["/Bifurcation/Stability"]
        if "/Bifurcation/Fold" in time_data.keys():
            del time_data["/Bifurcation/Fold"]
        if "/Bifurcation/Flip" in time_data.keys():
            del time_data["/Bifurcation/Flip"]
        if "/Bifurcation/Neimark_Sacker" in time_data.keys():
            del time_data["/Bifurcation/Neimark_Sacker"]
        time_data["/Bifurcation/Floquet"] = Floquet
        time_data["/Bifurcation/Stability"] = Stability
        time_data["/Bifurcation/Fold"] = Fold
        time_data["/Bifurcation/Flip"] = Flip
        time_data["/Bifurcation/Neimark_Sacker"] = Neimark_Sacker
    time_data.close()

    print(f"\nCompleted! Data stored in: {new_file}")
    print("=" * 60)
