import h5py
import numpy as np


class StartingPoint:
    def __init__(self, prob):
        self.prob = prob
        self.X0 = None
        self.omega = None
        self.tau = None
        self.eig = None
        self.frq = None
        self.pose0 = None
        self.tgt0 = None

    def get_startingpoint(self):
        # eigen solution is always required for dof data
        self.eig, self.frq, self.pose0 = self.prob.icfunction()

        if "eig_start" in self.prob.cont_params["first_point"].keys():
            self.eig_start()
        elif "restart" in self.prob.cont_params["first_point"].keys():
            self.restart()

    def eig_start(self):
        nnm = self.prob.cont_params["first_point"]["eig_start"]["NNM"]
        scale = self.prob.cont_params["first_point"]["eig_start"]["scale"]
        dofdata = self.prob.doffunction()
        x0 = scale * self.eig[:, nnm - 1]
        x0 = x0[dofdata["free_dof"]]
        v0 = np.zeros_like(x0)
        self.X0 = np.concatenate([x0, v0])
        T0 = 1 / self.frq[nnm - 1, 0]

        if self.prob.cont_params["continuation"]["forced"]:
            freq_scale = self.prob.cont_params["forcing"]["starting_freq_scale"]
            T0 /= freq_scale

        if self.prob.cont_params["shooting"]["scaling"]:
            self.omega = 1 / T0
            self.tau = 1.0
        else:
            self.omega = 1.0
            self.tau = T0

    def restart(self):
        restartsol = h5py.File(
            self.prob.cont_params["first_point"]["restart"]["file_name"] + ".h5", "r+"
        )
        index = self.prob.cont_params["first_point"]["restart"]["index"]
        T = restartsol["/T"][index]
        self.pose0 = restartsol["/Config/POSE"][:, index]
        vel = restartsol["/Config/VELOCITY"][:, index]
        try:
            self.tgt0 = restartsol["/Tangent"][:, index]
        except:
            self.tft0 = None

        dofdata = self.prob.doffunction()
        N = dofdata["ndof_free"]
        v = vel[dofdata["free_dof"]]
        self.X0 = np.concatenate([np.zeros(N), v])

        if self.prob.cont_params["shooting"]["scaling"] == True:
            nnm = self.prob.cont_params["first_point"]["eig_start"]["NNM"]
            T0 = 1 / self.frq[nnm - 1, 0]
            self.omega = 1 / T0  # omega fixed using period of linear mode
            self.tau = self.omega * T
            self.X0[N:] *= 1 / self.omega  # scale velocities from X to Xtilde
        else:
            self.omega = 1.0
            self.tau = T

        # # If different frequency is specified
        # if self.prob.cont_params["first_point"]["restart"]["fixF"]:
        #     self.T0 = np.float64(1 / self.prob.cont_params["first_point"]["restart"]["F"])
