import numpy as np
from scipy.integrate import odeint, simps


class Duffing:
    """
    fixed parameters of the model
    xddot + delta*xdot + alpha*x + beta*x^3 = F*sin(2pi/T*t)
    """

    alpha, beta = 1.0, 4.0
    delta = 0.0

    # finite element data, 1 dof model
    free_dof = np.array([0])
    ndof_all = 1
    ndof_fix = 0
    ndof_free = 1
    nnodes_all = 1
    config_per_node = 1

    @classmethod
    def forcing_parameters(cls, cont_params):
        """
        update parameters if continuation is forced
        """
        if cont_params["continuation"]["forced"]:
            cls.delta = cont_params["forcing"]["tau0"]

    @classmethod
    def model_ode(cls, t, X, T, F):
        """
        ODE of model: Xdot(t) = g(X(t))
        """
        x = X[0]
        xdot = X[1]
        f = cls.delta * xdot + cls.alpha * x + cls.beta * x**3
        force = F * np.sin(2 * np.pi / T * t)
        Xdot = np.array([xdot, -f + force])
        return Xdot

    @classmethod
    def model_sens_ode(cls, t, ic, T, F):
        """
        Augmented ODE of model + sensitivities, to be solved together
        System: Xdot(t) = g(X(t))
        Monodromy: dXdX0dot = dg(X)dX . dXdX0
        Time sens: dXdTdot = dg(X)dX . dXdT + dgdT
        Force sens: dXdFdot = dg(X)dX . dXdF + dgdF
        """
        X0, dXdX0, dXdT, dXdF = ic[:2], ic[2:6], ic[6:8], ic[8:]
        x = X0[0]
        xdot = X0[1]
        f = cls.delta * xdot + cls.alpha * x + cls.beta * x**3
        force = F * np.sin(2 * np.pi / T * t)
        dforce_dT = F * np.cos(2 * np.pi / T * t) * (-2 * np.pi * t / T**2)
        dforce_dF = np.sin(2 * np.pi / T * t)
        Xdot = np.array([xdot, -f + force])
        dgdX = np.array([[0, 1], [-cls.alpha - 3 * cls.beta * x**2, -cls.delta]])
        dgdT = np.array([0, dforce_dT])
        dgdF = np.array([0, dforce_dF])
        dXdX0dot = dgdX @ dXdX0.reshape(2, 2)
        dXdTdot = dgdX @ dXdT.reshape(2, 1) + dgdT.reshape(-1, 1)
        dXdFdot = dgdX @ dXdF.reshape(2, 1) + dgdF.reshape(-1, 1)

        return np.concatenate([Xdot, dXdX0dot.flatten(), dXdTdot.flatten(), dXdFdot.flatten()])

    @classmethod
    def eigen_solve(cls):
        """
        Eigenvalue and eigenvector of the model
        """
        frq = np.array([[cls.alpha]])  # natural frequency
        frq = np.sqrt(frq) / (2 * np.pi)
        eig = np.array([[1.0]])

        # initial position taken as zero
        pose0 = 0.0

        return eig, frq, pose0

    @classmethod
    def time_solve(cls, omega, F, T, X, pose_base, cont_params, sensitivity=True, fulltime=False):
        """
        Time simulation of the model and sensitivity analysis of periodicity function
        """
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        continuation_parameter = cont_params["continuation"]["continuation_parameter"]

        # Add position to increment and do time sim to get solution and sensitivities
        X0 = X + np.array([pose_base, 0])
        t = np.linspace(0, T, nsteps + 1)
        # Initial conditions for the augmented system, eye for monodromy and zero for time and force sens
        all_ic = np.concatenate((X0, np.eye(2).flatten(), np.zeros(2), np.zeros(2)))
        sol = odeint(cls.model_sens_ode, all_ic, t, args=(T, F), rtol=rel_tol, tfirst=True)
        # unpack solution
        Xsol, M, dXdT, dXdF = sol[:, :2], sol[-1, 2:6].reshape(2, 2), sol[-1, 6:8], sol[-1, 8:]

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)

        # Jacobian construction depends on continuation parameter
        dHdX0 = M - np.eye(2)
        gX_T = cls.model_ode(T, Xsol[-1, :], T, F)

        if continuation_parameter == "frequency":
            # For frequency continuation, include period sensitivity (dH/dT)
            dHdT = gX_T + dXdT
            J = np.concatenate((dHdX0, dHdT.reshape(-1, 1)), axis=1)
        elif continuation_parameter == "amplitude":
            # For amplitude continuation, include force amplitude sensitivity (dH/dF)
            dHdF = dXdF
            J = np.concatenate((dHdX0, dHdF.reshape(-1, 1)), axis=1)

        # solution pose and vel at time 0
        pose = Xsol[0, 0]
        vel = Xsol[0, 1]

        # Energy
        E0 = (
            0.5 * (Xsol[:, 1] ** 2 + cls.alpha * Xsol[:, 0] ** 2)
            + 0.25 * cls.beta * Xsol[:, 0] ** 4
        )
        force_vel = F * np.sin(2 * np.pi / T * t) * Xsol[:, 1]
        damping_vel = cls.delta * Xsol[:, 1] ** 2
        E1 = np.array(
            [simps(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1]) for i in range(len(t))]
        )
        E = E0 + E1
        energy = np.max(E)

        # Acceleration
        Xddot = (
            F * np.sin(2 * np.pi / T * t)
            - cls.delta * Xsol[:, 1]
            - cls.alpha * Xsol[:, 0]
            - cls.beta * Xsol[:, 0] ** 3
        )

        # # Lagrangian
        # L = (
        #     0.5 * Xsol[:, 1]**2 - 0.5 * cls.alpha * Xsol[:, 0]**2 - 0.25 * cls.beta * Xsol[:, 0]**4
        # )

        if not fulltime:
            return H, J, pose, vel, energy, True
        else:
            return H, J, Xsol[:, 0], Xsol[:, 1], Xddot, energy, True

    @classmethod
    def get_fe_data(cls):
        """
        Get finite element data
        """
        return {
            "free_dof": cls.free_dof,
            "ndof_all": cls.ndof_all,
            "ndof_fix": cls.ndof_fix,
            "ndof_free": cls.ndof_free,
            "nnodes_all": cls.nnodes_all,
            "config_per_node": cls.config_per_node,
            "dof_per_node": cls.ndof_free,
            "n_dim": 1,
            "SEbeam": False,
        }

    # Central Difference methods can be used to validate the values from ode
    # @classmethod
    # def monodromy_centdiff(cls, t, X0, T, F):
    #     eps = 1e-8
    #     M = np.zeros([2, 2])
    #     for i in range(2):
    #         X0plus = X0.copy()
    #         X0plus[i] += eps
    #         XTplus = np.array(odeint(cls.model_ode, X0plus, t, args=(T, F), tfirst=True))[-1, :]
    #         X0mins = X0.copy()
    #         X0mins[i] -= eps
    #         XTmins = np.array(odeint(cls.model_ode, X0mins, t, args=(T, F), tfirst=True))[-1, :]
    #         m = (XTplus - XTmins) / (2 * eps)
    #         M[:, i] = m
    #     return M

    # @classmethod
    # def periodsens_centdiff(cls, X0, T, F):
    #     eps = 1e-8
    #     Tplus = T + eps
    #     t = np.linspace(0, Tplus, 301)
    #     XTplus = np.array(odeint(cls.model_ode, X0, t, args=(Tplus, F), tfirst=True))[-1, :]
    #     Tmins = T - eps
    #     t = np.linspace(0, Tmins, 301)
    #     XTmins = np.array(odeint(cls.model_ode, X0, t, args=(Tmins, F), tfirst=True))[-1, :]
    #     dXdT = (XTplus - XTmins) / (2 * eps)
    #     return dXdT

    # @classmethod
    # def forcesens_centdiff(cls, X0, t, F):
    #     T = t[-1]
    #     eps = 1e-8
    #     Fplus = F + eps
    #     XTplus = np.array(odeint(cls.model_ode, X0, t, args=(T, Fplus), tfirst=True))[-1, :]
    #     Fmins = F - eps
    #     XTmins = np.array(odeint(cls.model_ode, X0, t, args=(T, Fmins), tfirst=True))[-1, :]
    #     dXdF = (XTplus - XTmins) / (2 * eps)
    #     return dXdF
