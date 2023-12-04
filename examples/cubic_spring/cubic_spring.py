import numpy as np
from scipy.integrate import odeint
import scipy.linalg as spl
import scipy.interpolate as spi


class Cubic_Spring:
    # parameters of nonlinear system EoM    MX'' + KX + fnl = 0
    M = np.eye(2)
    K = np.array([[2, -1], [-1, 2]])
    Knl = 0.5
    Minv = spl.inv(M)

    # finite element data, 2 dof system
    free_dof = np.array([0, 1])
    ndof_all = 2
    ndof_fix = 0
    ndof_free = 2

    @classmethod
    def system_ode(cls, t, X):
        # State equation of the mass spring system. Xdot(t) = g(X(t))
        x = X[:cls.ndof_free]
        xdot = X[cls.ndof_free:]
        KX = cls.K @ x
        fnl = np.array([cls.Knl * x[0]**3, 0])
        Xdot = np.concatenate((xdot, -cls.Minv @ (KX + fnl)))
        return Xdot

    @classmethod
    def augsystem_ode(cls, t, X_aug):
        # Augemented ODE of system + Monodromy, to be solved together
        # System: Xdot(t) = g(X(t))
        # Monodromy: dXdX0dot = dg(X)dX . dXdX0
        M = cls.M
        Minv = cls.Minv
        K = cls.K
        knl = cls.Knl
        N = cls.ndof_free
        twoN = 2 * N

        X, dXdX0 = X_aug[:twoN], X_aug[twoN:]
        x = X[:N]
        xdot = X[N:]
        KX = K @ x
        fnl = np.array([cls.Knl * x[0]**3, 0])
        Xdot = np.concatenate((xdot, -Minv @ (KX + fnl)))
        dgdz = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-1 / M[0, 0] * (K[0, 0] + knl * 3 * x[0]**2), -1 / M[0, 0] * K[0, 1], 0, 0],
                [-1 / M[1, 1] * K[1, 0], -1 / M[1, 1] * K[1, 1], 0, 0],
            ]
        )
        dXdX0dot = dgdz @ dXdX0.reshape(4, 4)

        return np.concatenate([Xdot, dXdX0dot.flatten()])

    @classmethod
    def eigen_solve(cls):
        # Continuation variables initial guess from eigenvalues
        frq, eig = spl.eigh(cls.K, cls.M)
        frq = np.sqrt(frq) / (2 * np.pi)
        frq = frq.reshape(-1, 1)

        # initial position taken as zero for both dofs.
        pose0 = np.zeros(cls.ndof_free)

        return eig, frq, pose0

    @classmethod
    def time_solve(
        cls, omega, tau, X, pose_base, cont_params, return_time=False, sensitivity=True
    ):
        nperiod = cont_params["shooting"]["single"]["nperiod"]
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        N = cls.ndof_free
        twoN = 2 * N
        T = tau / omega

        # Add increment onto pose and do time sim
        X_total = X.copy()
        X_total[:N] += pose_base.flatten().copy()
        t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
        initial_cond_aug = np.concatenate((X_total, np.eye(4).flatten()))
        Xsol_aug = np.array(
            odeint(cls.augsystem_ode, initial_cond_aug, t, rtol=rel_tol, tfirst=True)
        )
        Xsol, M = Xsol_aug[:, :twoN], Xsol_aug[-1, twoN:].reshape(twoN, twoN)

        # Monodromy and augmented Jacobian
        M -= np.eye(twoN)
        dHdt = cls.system_ode(None, Xsol[-1, :]) * nperiod
        J = np.concatenate((M, dHdt.reshape(-1, 1)), axis=1)

        # solution pose and vel taken from time 0
        pose = Xsol[0, :N].T
        vel = Xsol[0, N:].T

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)

        # Energy, conservative system so take mean of all time
        E = np.zeros(nsteps * nperiod + 1)
        for i in range(nsteps * nperiod + 1):
            x = Xsol[i, :N]
            xdot = Xsol[i, N:]
            Fnl = 0.25 * cls.Knl * x[0]**4
            E[i] = 0.5 * (xdot.T @ cls.M @ xdot + x.T @ cls.K @ x) + Fnl
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

    # @classmethod
    # def monodromy_centdiff(cls, t, X0):
    #     # central difference calculation of the monodromy matrix
    #     # can be used to check values from ode
    #     eps = 1e-8
    #     M = np.zeros([4, 4])
    #     for i in range(4):
    #         X0plus = X0.copy()
    #         X0plus[i] += eps
    #         XTplus = np.array(odeint(cls.system_ode, X0plus, t, tfirst=True))[-1, :]
    #         X0mins = X0.copy()
    #         X0mins[i] -= eps
    #         XTmins = np.array(odeint(cls.system_ode, X0mins, t, tfirst=True))[-1, :]
    #         m = (XTplus - XTmins) / (2 * eps)
    #         M[:, i] = m
    #     return M
