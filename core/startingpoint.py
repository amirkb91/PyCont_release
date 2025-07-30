import h5py
import numpy as np
import json


class StartingPoint:
    def __init__(self, prob):
        self.prob = prob
        self.X0 = None
        self.T0 = None
        self.F0 = None
        self.pose0 = None
        self.pose_ref = None
        self.tgt0 = None
        self.eig = None
        self.frq = None

    def get_startingpoint(self):
        # eigen solution is always required for dof data
        self.eig, self.frq, self.pose0 = self.prob.icfunction()
        self.pose_ref = np.copy(self.pose0)

        if self.prob.cont_params["first_point"]["from_eig"]:
            self.eig_start()
        else:
            self.restart()

    def eig_start(self):
        nnm = self.prob.cont_params["first_point"]["eig_start"]["NNM"]
        scale = self.prob.cont_params["first_point"]["eig_start"]["scale"]
        dofdata = self.prob.doffunction()
        x0 = scale * self.eig[:, nnm - 1]
        x0 = x0[dofdata["free_dof"]]
        v0 = np.zeros_like(x0)
        self.X0 = np.concatenate([x0, v0])
        self.T0 = 1 / self.frq[nnm - 1, 0]

        if self.prob.cont_params["continuation"]["forced"]:
            self.T0 = 1 / self.prob.cont_params["forcing"]["frequency"]
            self.F0 = self.prob.cont_params["forcing"]["amplitude"]
        else:
            self.F0 = 0.0

    def restart(self):
        restartsol = h5py.File(
            self.prob.cont_params["first_point"]["restart"]["file_name"] + ".h5", "r+"
        )
        index = self.prob.cont_params["first_point"]["restart"]["index"]
        dofdata = self.prob.doffunction()
        N = dofdata["ndof_free"]

        self.T0 = restartsol["/T"][index]
        self.pose0 = restartsol["/Config/POSE"][:, index]
        vel = restartsol["/Config/VELOCITY"][:, index]
        restartsol_parameters = json.loads(restartsol["/Parameters"][()])
        restartsol_shooting_method = restartsol_parameters["shooting"]["method"]

        # restart solution could be single or multiple shooting
        if restartsol_shooting_method == "single":
            v0 = vel[dofdata["free_dof"]]
            x0 = np.zeros_like(v0)
            self.X0 = np.concatenate([x0, v0])
        elif restartsol_shooting_method == "multiple":
            npartition = restartsol_parameters["shooting"]["multiple"]["npartition"]
            self.pose0 = np.reshape(self.pose0, (-1, npartition), order="F")
            vel = np.reshape(vel, (-1, npartition), order="F")
            v0 = vel[dofdata["free_dof"], :]
            self.X0 = np.concatenate((np.zeros((N, npartition)), v0))
            self.X0 = np.reshape(self.X0, (-1), order="F")

        try:
            self.tgt0 = restartsol["/Tangent"][:, index]
        except:
            self.tgt0 = None
