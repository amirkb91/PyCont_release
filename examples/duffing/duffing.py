import numpy as np
from scipy.integrate import odeint, simps


class Duffing:
    # fixed parameters of the unforced model EoM
    # xddot + delta*xdot + alpha*x + beta*x^3 = F*cos(2pi/T*t + phi)
    alpha, beta = 1.0, 1.0
    delta, F, phi = 0.0, 0.0, 0.0

    # finite element data, 1 dof model
    free_dof = np.array([0])
    ndof_all = 1
    ndof_fix = 0
    ndof_free = 1

    @classmethod
    def forcing_parameters(cls, cont_params):
        # update parameters if continuation is forced
        if cont_params["continuation"]["forced"]:
            cls.F = cont_params["forcing"]["amplitude"]
            cls.delta = cont_params["forcing"]["tau0"]
            cls.phi = cont_params["forcing"]["phase_ratio"] * np.pi

    @classmethod
    def model_ode(cls, t, X, T):
        # ODE of model: Xdot(t) = g(X(t))
        x = X[0]
        xdot = X[1]
        f = cls.delta * xdot + cls.alpha * x + cls.beta * x**3
        force = cls.F * np.cos(2 * np.pi / T * t + cls.phi)
        Xdot = np.array([xdot, -f + force])
        return Xdot

    @classmethod
    def model_sens_ode(cls, t, ic, T):
        # Augemented ODE of model + sensitivities, to be solved together
        # System: Xdot(t) = g(X(t))
        # Monodromy: dXdX0dot = dg(X)dX . dXdX0
        # Time sens: dXdTdot = dg(X)dX . dXdT + dgdT
        X0, dXdX0, dXdT = ic[:2], ic[2:6], ic[6:]
        x = X0[0]
        xdot = X0[1]
        f = cls.delta * xdot + cls.alpha * x + cls.beta * x**3
        force = cls.F * np.cos(2 * np.pi / T * t + cls.phi)
        force_der = -cls.F * np.sin(2 * np.pi / T * t + cls.phi) * 2 * np.pi * t / T**2
        Xdot = np.array([xdot, -f + force])
        dgdX = np.array([[0, 1], [-cls.alpha - 3 * cls.beta * x**2, -cls.delta]])
        dgdT = np.array([0, force_der])
        dXdX0dot = dgdX @ dXdX0.reshape(2, 2)
        dXdTdot = dgdX @ dXdT.reshape(2, 1) + dgdT.reshape(-1, 1)

        return np.concatenate([Xdot, dXdX0dot.flatten(), dXdTdot.flatten()])

    @classmethod
    def eigen_solve(cls):
        frq = np.array([[cls.alpha]])  # natural frequency
        frq = np.sqrt(frq) / (2 * np.pi)
        eig = np.array([[1.0]])

        # initial position taken as zero
        pose0 = 0.0

        return eig, frq, pose0

    @classmethod
    def time_solve(cls, omega, T, X, pose_base, cont_params, sensitivity=True):
        nperiod = cont_params["shooting"]["single"]["nperiod"]
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]

        # Add position to increment and do time sim to get solution and sensitivities
        X0 = X.copy()
        X0[0] += pose_base
        t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
        all_ic = np.concatenate((X0, np.eye(2).flatten(), np.zeros(2)))
        sol = odeint(cls.model_sens_ode, all_ic, t, args=(T * nperiod,), rtol=rel_tol, tfirst=True)
        Xsol, M, dXdT = sol[:, :2], sol[-1, 2:6].reshape(2, 2), sol[-1, 6:]

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)

        # Jacobian (dHdX0 and dHdt)
        dHdX0 = M - np.eye(2)
        gX_T = cls.model_ode(T * nperiod, Xsol[-1, :], T * nperiod) * nperiod

        # *** Time sensitivity is not working, do finite difference for now ***
        if cont_params["continuation"]["forced"]:
            # dHdt = gX_T + dXdT
            dHdT = cls.timesens_centdiff(X0, T)
        else:
            dHdT = gX_T

        J = np.concatenate((dHdX0, dHdT.reshape(-1, 1)), axis=1)

        # solution pose and vel over time
        pose_time = Xsol[:, 0].reshape(1, -1)
        vel_time = Xsol[:, 1].reshape(1, -1)

        # Energy
        E0 = (
            0.5 * (Xsol[:, 1] ** 2 + cls.alpha * Xsol[:, 0] ** 2)
            + 0.25 * cls.beta * Xsol[:, 0] ** 4
        )
        force_vel = cls.F * np.cos(2 * np.pi / T * t + cls.phi) * Xsol[:, 1]
        damping_vel = cls.delta * Xsol[:, 1] ** 2
        E1 = np.array(
            [simps(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1]) for i in range(len(t))]
        )
        E = E0 + E1
        energy = np.max(E)

        cvg = True
        return H, J, pose_time, vel_time, energy, cvg

    @classmethod
    def get_fe_data(cls):
        return {
            "free_dof": cls.free_dof,
            "ndof_all": cls.ndof_all,
            "ndof_fix": cls.ndof_fix,
            "ndof_free": cls.ndof_free,
        }

    @classmethod
    def monodromy_centdiff(cls, t, X0, T):
        # central difference calculation of the monodromy matrix
        # can be used to check values from ode
        eps = 1e-8
        M = np.zeros([2, 2])
        for i in range(2):
            X0plus = X0.copy()
            X0plus[i] += eps
            XTplus = np.array(odeint(cls.model_ode, X0plus, t, args=(T,), tfirst=True))[-1, :]
            X0mins = X0.copy()
            X0mins[i] -= eps
            XTmins = np.array(odeint(cls.model_ode, X0mins, t, args=(T,), tfirst=True))[-1, :]
            m = (XTplus - XTmins) / (2 * eps)
            M[:, i] = m
        return M

    @classmethod
    def timesens_centdiff(cls, X0, T):
        # central difference calculation of the time sensitivity matrix
        # can be used to check values from ode
        eps = 1e-8
        Tplus = T + eps
        t = np.linspace(0, Tplus, 301)
        XTplus = np.array(odeint(cls.model_ode, X0, t, args=(Tplus,), tfirst=True))[-1, :]
        Tmins = T - eps
        t = np.linspace(0, Tmins, 301)
        XTmins = np.array(odeint(cls.model_ode, X0, t, args=(Tmins,), tfirst=True))[-1, :]
        dXdT = (XTplus - XTmins) / (2 * eps)
        return dXdT
