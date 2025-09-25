import numpy as np
from scipy.integrate import odeint, simpson
import scipy.linalg as spl


class Cubic_Spring:
    # parameters of nonlinear system EoM    MX'' + CX' + KX + fnl = F*sin(2pi/T*t)
    M = np.eye(2)
    Minv = np.eye(2)
    C = np.zeros((2, 2))
    K = np.array([[2, -1], [-1, 2]])
    Knl = 0.5

    # finite element data, 2 dof system
    free_dof = np.array([0, 1])
    ndof_all = 2
    ndof_fix = 0
    ndof_free = 2
    nnodes_all = 2
    config_per_node = 1

    @classmethod
    def update_model(cls, parameters):
        # update model definition depending on parameters
        if "force" in parameters["continuation"]["parameter"]:
            # forced continuation, update damping matrix
            cls.C = 0.05 * cls.M + 0.01 * cls.K

    @classmethod
    def eigen_solve(cls):
        # Compute eigenvectors and natural frequencies of the model
        frq, eig = spl.eigh(cls.K, cls.M)
        frq = np.sqrt(frq) / (2 * np.pi)  # Hz

        # select mode and scaling
        mode = 1
        scale = 1e-5
        X0 = scale * eig[:, mode - 1]
        T0 = 1 / frq[mode - 1]

        return X0, T0

    @classmethod
    def model_ode(cls, t, X, T, F):
        # State equation of the mass spring system. Xdot(t) = g(X(t))
        x = X[: cls.ndof_free]
        xdot = X[cls.ndof_free :]
        KX = cls.K @ x
        CXdot = cls.C @ xdot
        force = np.array([F * np.sin(2 * np.pi / T * t), 0])
        fnl = np.array([cls.Knl * x[0] ** 3, 0])
        Xdot = np.concatenate((xdot, cls.Minv @ (-KX - CXdot - fnl + force)))
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
        Minv = cls.Minv
        K = cls.K
        C = cls.C
        knl = cls.Knl
        N = cls.ndof_free
        twoN = 2 * N

        # Unpack initial conditions: X0, monodromy sensitivities, time sensitivities, force sensitivities
        X0, dXdX0, dXdT, dXdF = (
            ic[:twoN],
            ic[twoN : twoN + twoN**2],
            ic[twoN + twoN**2 : twoN + twoN**2 + twoN],
            ic[twoN + twoN**2 + twoN :],
        )

        x = X0[:N]
        xdot = X0[N:]
        KX = K @ x
        CXdot = C @ xdot
        force = np.array([F * np.sin(2 * np.pi / T * t), 0])
        fnl = np.array([cls.Knl * x[0] ** 3, 0])

        # Force derivatives
        dforce_dT = np.array([F * np.cos(2 * np.pi / T * t) * (-2 * np.pi * t / T**2), 0])
        dforce_dF = np.array([np.sin(2 * np.pi / T * t), 0])

        # System dynamics
        Xdot = np.concatenate((xdot, cls.Minv @ (-KX - CXdot - fnl + force)))

        # Jacobian of system dynamics with respect to state
        dgdX = np.zeros((twoN, twoN))
        dgdX[:N, N:] = np.eye(N)  # dx/dt = xdot
        dgdX[N:, :N] = -Minv @ (K + np.array([[knl * 3 * x[0] ** 2, 0], [0, 0]]))  # d(xddot)/dx
        dgdX[N:, N:] = -Minv @ C  # d(xddot)/d(xdot)

        # Partial derivatives with respect to parameters
        dgdT = np.concatenate([np.zeros(N), Minv @ dforce_dT])
        dgdF = np.concatenate([np.zeros(N), Minv @ dforce_dF])

        # Sensitivity calculations
        dXdX0dot = dgdX @ dXdX0.reshape(twoN, twoN)
        dXdTdot = dgdX @ dXdT.reshape(twoN, 1) + dgdT.reshape(-1, 1)
        dXdFdot = dgdX @ dXdF.reshape(twoN, 1) + dgdF.reshape(-1, 1)

        return np.concatenate([Xdot, dXdX0dot.flatten(), dXdTdot.flatten(), dXdFdot.flatten()])

    @classmethod
    def time_solve(cls, omega, F, T, X, pose_base, parameters, sensitivity=True, fulltime=False):
        nperiod = parameters["shooting"]["single"]["nperiod"]
        nsteps = parameters["shooting"]["single"]["nsteps_per_period"]
        rel_tol = parameters["shooting"]["rel_tol"]
        continuation_parameter = parameters["continuation"]["continuation_parameter"]
        N = cls.ndof_free
        twoN = 2 * N

        # Add position to increment and do time sim to get solution and sensitivities
        X0 = X + np.concatenate((pose_base, np.zeros(N)))
        t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
        # Initial conditions for the augmented system: X0, eye for monodromy, zeros for time and force sens
        all_ic = np.concatenate((X0, np.eye(twoN).flatten(), np.zeros(twoN), np.zeros(twoN)))
        sol = np.array(
            odeint(cls.model_sens_ode, all_ic, t, args=(T, F), rtol=rel_tol, tfirst=True)
        )
        # unpack solution
        Xsol, M, dXdT, dXdF = (
            sol[:, :twoN],
            sol[-1, twoN : twoN + twoN**2].reshape(twoN, twoN),
            sol[-1, twoN + twoN**2 : twoN + twoN**2 + twoN],
            sol[-1, twoN + twoN**2 + twoN :],
        )

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)

        # Jacobian construction depends on continuation parameter
        dHdX0 = M - np.eye(twoN)
        gX_T = cls.model_ode(T * nperiod, Xsol[-1, :], T, F) * nperiod

        if continuation_parameter == "frequency":
            # For frequency continuation, include period sensitivity (dH/dT)
            dHdT = gX_T + dXdT
            J = np.concatenate((dHdX0, dHdT.reshape(-1, 1)), axis=1)
        elif continuation_parameter == "amplitude":
            # For amplitude continuation, include force amplitude sensitivity (dH/dF)
            dHdF = dXdF
            J = np.concatenate((dHdX0, dHdF.reshape(-1, 1)), axis=1)

        # solution pose and vel at time 0
        pose = Xsol[0, :N]
        vel = Xsol[0, N:]

        # Energy
        E0 = (
            0.5 * np.einsum("ij,ij->i", Xsol[:, N:], np.dot(cls.M, Xsol[:, N:].T).T)
            + 0.5 * np.einsum("ij,ij->i", Xsol[:, :N], np.dot(cls.K, Xsol[:, :N].T).T)
            + 0.25 * cls.Knl * Xsol[:, 0] ** 4
        )
        force_vec = np.array([F * np.sin(2 * np.pi / T * t), np.zeros_like(t)])
        force_vel = force_vec[0] * Xsol[:, N]
        damping_vel = np.einsum("ij,ij->i", Xsol[:, N:], (cls.C @ Xsol[:, N:].T).T)
        E1 = np.array(
            [simpson(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1]) for i in range(len(t))]
        )
        E = E0 + E1
        energy = np.max(E)

        # Calculate acceleration for full time output
        acc_time = np.zeros_like(Xsol[:, :N])
        for i in range(len(t)):
            x_i = Xsol[i, :N]
            xdot_i = Xsol[i, N:]
            KX = cls.K @ x_i
            CXdot = cls.C @ xdot_i
            force_i = np.array([F * np.sin(2 * np.pi / T * t[i]), 0])
            fnl_i = np.array([cls.Knl * x_i[0] ** 3, 0])
            acc_time[i, :] = cls.Minv @ (-KX - CXdot - fnl_i + force_i)

        if not fulltime:
            return H, J, pose, vel, energy, True
        else:
            return H, J, Xsol[:, :N], Xsol[:, N:], acc_time, energy, True

    @classmethod
    def get_fe_data(cls):
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
