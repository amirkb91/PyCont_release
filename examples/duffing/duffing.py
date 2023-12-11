import numpy as np
from scipy.integrate import odeint


class Duffing:
    # parameters of the model EoM
    # xddot + delta*xdot + alpha*x + beta*x^3 = 0
    delta = 0.0
    alpha = 1.0
    beta = 1.0

    # finite element data, 1 dof model
    free_dof = np.array([0])
    ndof_all = 1
    ndof_fix = 0
    ndof_free = 1

    @classmethod
    def system_ode(cls, t, X):
        # ODE of model: Xdot(t) = g(X(t))
        x = X[0]
        xdot = X[1]
        f = cls.delta * xdot + cls.alpha * x + cls.beta * x**3
        Xdot = np.array([xdot, -f])
        return Xdot

    @classmethod
    def augsystem_ode(cls, t, X_aug):
        # Augemented ODE of model + Monodromy, to be solved together
        # System: Xdot(t) = g(X(t))
        # Monodromy: dXdX0dot = dg(X)dX . dXdX0
        X, dXdX0 = X_aug[:2], X_aug[2:]
        x = X[0]
        xdot = X[1]
        f = cls.delta * xdot + cls.alpha * x + cls.beta * x**3
        Xdot = np.array([xdot, -f])
        dgdX = np.array([[0, 1], [-cls.alpha - 3 * cls.beta * x**2, -cls.delta]])
        dXdX0dot = dgdX @ dXdX0.reshape(2, 2)

        return np.concatenate([Xdot, dXdX0dot.flatten()])

    @classmethod
    def eigen_solve(cls):
        frq = np.array([[cls.alpha]])  # natural frequency
        frq = np.sqrt(frq) / (2 * np.pi)
        eig = np.array([[1.0]])

        # initial position taken as zero
        pose0 = 0.0

        return eig, frq, pose0

    @classmethod
    def time_solve(
        cls, omega, tau, X, pose_base, cont_params, return_time=False, sensitivity=True
    ):
        nperiod = cont_params["shooting"]["single"]["nperiod"]
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        T = tau / omega

        # Add position to increment and do time sim to get solution and Monodromy M
        X_total = X.copy()
        X_total[0] += pose_base
        t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
        initial_cond_aug = np.concatenate((X_total, np.eye(2).flatten()))
        Xsol_aug = np.array(
            odeint(cls.augsystem_ode, initial_cond_aug, t, rtol=rel_tol, tfirst=True)
        )
        Xsol, M = Xsol_aug[:, :2], Xsol_aug[-1, 2:].reshape(2, 2)

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)

        # Augmented Jacobian (dHdX0 and dHdt)
        dHdX0 = M - np.eye(2)
        dHdt = cls.system_ode(None, Xsol[-1, :]) * nperiod
        J = np.concatenate((dHdX0, dHdt.reshape(-1, 1)), axis=1)

        # solution pose and vel taken from time 0
        pose = Xsol[0, 0]
        vel = Xsol[0, 1]

        # Energy, conservative model so take mean of all time
        E = np.zeros(nsteps * nperiod + 1)
        for i in range(nsteps * nperiod + 1):
            x = Xsol[i, 0]
            xdot = Xsol[i, 1]
            Fnl = 0.25 * cls.beta * x**4
            E[i] = 0.5 * (xdot**2 + cls.alpha * x**2) + Fnl
        energy = np.mean(E)

        cvg = True
        return H, J, M, pose, vel, energy, cvg

    @classmethod
    def get_fe_data(cls):
        return {
            "free_dof": cls.free_dof,
            "ndof_all": cls.ndof_all,
            "ndof_fix": cls.ndof_fix,
            "ndof_free": cls.ndof_free,
        }
