import numpy as np
from scipy.integrate import odeint, simps
import scipy.linalg as spl


class Cubic_Spring:
    # parameters of nonlinear system EoM    MX'' + CX' + KX + fnl = F*sin(2pi/T*t)
    M = np.eye(2)
    C = np.zeros((2, 2))
    K = np.array([[2, -1], [-1, 2]])
    Knl = 0.5
    Minv = spl.inv(M)

    # finite element data, 2 dof system
    free_dof = np.array([0, 1])
    ndof_all = 2
    ndof_fix = 0
    ndof_free = 2
    nnodes_all = 2
    config_per_node = 1

    @classmethod
    def forcing_parameters(cls, cont_params):
        """
        update parameters if continuation is forced
        """
        if cont_params["continuation"]["forced"]:
            tau0 = cont_params["forcing"]["tau0"]
            tau1 = cont_params["forcing"]["tau1"]
            cls.C = tau0 * cls.M + tau1 * cls.K

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
        Augemented ODE of model + sensitivities, to be solved together
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
    def eigen_solve(cls):
        # Continuation variables initial guess from eigenvalues
        frq, eig = spl.eigh(cls.K, cls.M)
        frq = np.sqrt(frq) / (2 * np.pi)
        frq = frq.reshape(-1, 1)

        # initial position taken as zero for both dofs
        pose0 = np.zeros(cls.ndof_free)

        return eig, frq, pose0

    @classmethod
    def time_solve(cls, omega, F, T, X, pose_base, cont_params, sensitivity=True, fulltime=False):
        nperiod = cont_params["shooting"]["single"]["nperiod"]
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        continuation_parameter = cont_params["continuation"]["continuation_parameter"]
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
            [simps(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1]) for i in range(len(t))]
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
    def time_solve_multiple(
        cls, omega, F, T, X, pose_base, cont_params, sensitivity=True, fulltime=False
    ):
        npartition = cont_params["shooting"]["multiple"]["npartition"]
        nsteps = cont_params["shooting"]["multiple"]["nsteps_per_partition"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        N = cls.ndof_free
        twoN = 2 * N
        delta_S = 1 / npartition

        # Precomputations
        partition_extremeties = np.arange(npartition + 1) * (nsteps + 1)
        indices_start = partition_extremeties[:npartition]
        indices_end = indices_start - 1
        block_order = (np.arange(npartition) + 1) % npartition

        # Initialisations
        t = np.linspace(0, T * delta_S, nsteps + 1)
        eye_flat = np.eye(4).flatten()
        J = np.zeros((npartition * twoN, npartition * twoN + 1))
        pose_time = np.zeros((cls.ndof_all, (nsteps + 1) * npartition))
        vel_time = np.zeros((cls.ndof_all, (nsteps + 1) * npartition))
        E = np.zeros([nsteps + 1, npartition])
        energy = 0

        # Main loop for each partition
        for ipart in range(npartition):
            i0, i1 = ipart * twoN, (ipart + 1) * twoN
            j0, j1 = (ipart + 1) % npartition * twoN, ((ipart + 1) % npartition + 1) * twoN
            p0, p1 = partition_extremeties[ipart], partition_extremeties[ipart + 1]

            # Compute initial conditions add increment to pose
            X0 = X[i0:i1] + np.concatenate((pose_base[:, ipart], np.zeros(N)))
            all_ic = np.concatenate((X0, eye_flat))

            # Solve
            sol = np.array(odeint(cls.model_sens_ode, all_ic, t, rtol=rel_tol, tfirst=True))
            Xsol, M = sol[:, :twoN], sol[-1, twoN:].reshape(twoN, twoN)
            pose_time[:, p0:p1] = Xsol[:, :N].T
            vel_time[:, p0:p1] = Xsol[:, N:].T

            # Monodromy and augmented Jacobian
            dHdT = cls.model_ode(None, Xsol[-1, :]) * delta_S
            J[i0:i1, i0:i1] = M
            J[i0:i1, j0:j1] -= np.eye(twoN)
            J[i0:i1, -1] = dHdT

            # Energy
            E[:, ipart] = (
                0.5 * np.einsum("ij,ij->i", Xsol[:, N:], np.dot(cls.M, Xsol[:, N:].T).T)
                + 0.5 * np.einsum("ij,ij->i", Xsol[:, :N], np.dot(cls.K, Xsol[:, :N].T).T)
                + 0.25 * cls.Knl * Xsol[:, 0] ** 4
            )
            energy = np.max([energy, np.max(E)])

        # Periodicity condition for all partitions
        H1 = (
            pose_time[cls.free_dof][:, indices_end[block_order]]
            - pose_time[cls.free_dof][:, indices_start[block_order]]
        )
        H2 = (
            vel_time[cls.free_dof][:, indices_end[block_order]]
            - vel_time[cls.free_dof][:, indices_start[block_order]]
        )
        H = np.reshape(np.concatenate([H1, H2]), (-1, 1), order="F")

        # solution pose and vel at time 0 for each partition
        pose = pose_time[:, indices_start]
        vel = vel_time[:, indices_start]

        return H, J, pose, vel, energy, True

    @classmethod
    def partition_singleshooting_solution(cls, T, X, pose_base, cont_params):
        npartition = cont_params["shooting"]["multiple"]["npartition"]
        nsteps = cont_params["shooting"]["multiple"]["nsteps_per_partition"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        N = cls.ndof_free
        slicing_index = nsteps * np.arange(npartition)

        # do time integration along whole orbit before slicing
        X0 = X + np.concatenate((pose_base, np.zeros(N)))
        t = np.linspace(0, T, nsteps * npartition + 1)
        Xsol = np.array(odeint(cls.model_ode, X0, t, rtol=rel_tol, tfirst=True))
        pose_time = Xsol[:, :N].T
        vel_time = Xsol[:, N:].T
        pose = pose_time[:, slicing_index]
        vel = vel_time[:, slicing_index]
        # set inc to zero as solution stored in pose, keep velocity
        Xsol = np.concatenate((np.zeros((N, npartition)), vel))
        Xsol = np.reshape(Xsol, (-1), order="F")
        return Xsol, pose

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
