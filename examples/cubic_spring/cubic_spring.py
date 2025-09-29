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
        scale = 0.1
        X0 = scale * eig[:, mode - 1]
        T0 = 1 / frq[mode - 1]

        # add zeros for velocity degrees of freedom
        X0 = np.append(X0, np.zeros_like(X0))

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
    def time_solve(cls, F, T, X, parameters, fulltime=False):
        nsteps = parameters["shooting"]["steps_per_period"]
        rel_tol = parameters["shooting"]["integration_tolerance"]
        continuation_parameter = parameters["continuation"]["parameter"]

        n_dof = np.size(X)
        n_dof_2 = n_dof // 2

        # Run time simulation
        t = np.linspace(0, T, nsteps + 1)
        # Initial conditions for the augmented system: X0, eye for monodromy, zeros for time and force sens
        all_ic = np.concatenate((X, np.eye(n_dof).flatten(), np.zeros(n_dof), np.zeros(n_dof)))
        sol = np.array(
            odeint(cls.model_sens_ode, all_ic, t, args=(T, F), rtol=rel_tol, tfirst=True)
        )
        # unpack solution
        Xsol, M, dXdT, dXdF = (
            sol[:, :n_dof],
            sol[-1, n_dof : n_dof + n_dof**2].reshape(n_dof, n_dof),
            sol[-1, n_dof + n_dof**2 : n_dof + n_dof**2 + n_dof],
            sol[-1, n_dof + n_dof**2 + n_dof :],
        )

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]

        # Jacobian construction
        dHdX0 = M - np.eye(n_dof)
        gX_T = cls.model_ode(T, Xsol[-1, :], T, F)

        if continuation_parameter == "force_freq" or F == 0:
            # For frequency continuation, include period sensitivity (dH/dT)
            # Unforced continuation still has gX_T, but dXdT will be zero.
            dHdT = gX_T + dXdT
            J = np.concatenate((dHdX0, dHdT.reshape(-1, 1)), axis=1)
        elif continuation_parameter == "force_amp":
            # For amplitude continuation, include force amplitude sensitivity (dH/dF)
            dHdF = dXdF
            J = np.concatenate((dHdX0, dHdF.reshape(-1, 1)), axis=1)

        # Energy
        E0 = (
            0.5 * np.einsum("ij,ij->i", Xsol[:, n_dof_2:], np.dot(cls.M, Xsol[:, n_dof_2:].T).T)
            + 0.5 * np.einsum("ij,ij->i", Xsol[:, :n_dof_2], np.dot(cls.K, Xsol[:, :n_dof_2].T).T)
            + 0.25 * cls.Knl * Xsol[:, 0] ** 4
        )
        force_vec = np.array([F * np.sin(2 * np.pi / T * t), np.zeros_like(t)])
        force_vel = force_vec[0] * Xsol[:, n_dof_2]
        damping_vel = np.einsum("ij,ij->i", Xsol[:, n_dof_2:], (cls.C @ Xsol[:, n_dof_2:].T).T)
        E1 = np.array(
            [simpson(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1]) for i in range(len(t))]
        )
        E = E0 + E1
        energy = np.max(E)

        # Calculate acceleration for full time output
        accel = np.zeros_like(Xsol[:, :n_dof_2])
        for i in range(len(t)):
            x_i = Xsol[i, :n_dof_2]
            xdot_i = Xsol[i, n_dof_2:]
            KX = cls.K @ x_i
            CXdot = cls.C @ xdot_i
            force_i = np.array([F * np.sin(2 * np.pi / T * t[i]), 0])
            fnl_i = np.array([cls.Knl * x_i[0] ** 3, 0])
            accel[i, :] = cls.Minv @ (-KX - CXdot - fnl_i + force_i)

        if not fulltime:
            return H, J, energy
        else:
            return H, J, Xsol[:, :n_dof_2], Xsol[:, n_dof_2:], accel, energy

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
