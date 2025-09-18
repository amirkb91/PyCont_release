import numpy as np
from scipy.integrate import odeint, simpson, solve_ivp
import scipy.linalg as spl


class Beam_Spring:
    # Nonlinear beam model with tip springs

    # Beam paremeters for 2 mode system
    k_nl = 4250000
    w_1 = 91.734505484821950
    w_2 = 3.066194429903638e02
    zeta_1 = 0.0
    zeta_2 = 0.0
    phi_L = np.array([[-7.382136522799137, 7.360826867549465]])

    # Modal Matrices
    M = np.eye(2)
    C = np.array([[2 * zeta_1 * w_1, 0], [0, 2 * zeta_2 * w_2]])
    K = np.array([[w_1**2, 0], [0, w_2**2]])
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
            zeta_1 = 0.03
            zeta_2 = 0.09
            cls.C = np.array([[2 * zeta_1 * cls.w_1, 0],
                             [0, 2 * zeta_2 * cls.w_2]])

    @classmethod
    def model_ode(cls, t, X, T, F):
        # State equation of the mass spring system. Xdot(t) = g(X(t))
        x = X[: cls.ndof_free]
        xdot = X[cls.ndof_free:]
        KX = cls.K @ x
        CXdot = cls.C @ xdot
        force = np.array([F * np.sin(2 * np.pi / T * t), 0])
        phi_x = cls.phi_L @ x  # Physical displacement at nonlinear location
        fnl = cls.k_nl * cls.phi_L.T @ (phi_x**3)
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
        k_nl = cls.k_nl
        phi_L = cls.phi_L
        N = cls.ndof_free
        twoN = 2 * N

        # Unpack initial conditions: X0, monodromy sensitivities, time sensitivities, force sensitivities
        X0, dXdX0, dXdT, dXdF = (
            ic[:twoN],
            ic[twoN: twoN + twoN**2],
            ic[twoN + twoN**2: twoN + twoN**2 + twoN],
            ic[twoN + twoN**2 + twoN:],
        )

        x = X0[:N]
        xdot = X0[N:]
        KX = K @ x
        CXdot = C @ xdot
        force = np.array([F * np.sin(2 * np.pi / T * t), 0])
        phi_x = phi_L @ x  # Physical displacement at nonlinear location
        fnl = k_nl * phi_L.T @ (phi_x**3)

        # Force derivatives
        dforce_dT = np.array(
            [F * np.cos(2 * np.pi / T * t) * (-2 * np.pi * t / T**2), 0])
        dforce_dF = np.array([np.sin(2 * np.pi / T * t), 0])

        # System dynamics
        Xdot = np.concatenate((xdot, Minv @ (-KX - CXdot - fnl + force)))

        # Jacobian of nonlinear force with respect to modal coordinates
        # fnl = k_nl * phi_L.T @ (phi_L @ x)^3
        # dfnl/dx = k_nl * phi_L.T @ diag(3*(phi_L @ x)^2) @ phi_L
        dfnl_dx = k_nl * phi_L.T @ (3 * phi_x**2 * phi_L)

        # Jacobian of system dynamics with respect to state
        dgdX = np.zeros((twoN, twoN))
        dgdX[:N, N:] = np.eye(N)  # dx/dt = xdot
        dgdX[N:, :N] = -Minv @ (K + dfnl_dx)  # d(xddot)/dx
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
        """
        For modal model: frequencies are already known, eigenvectors are identity
        since we're working in modal coordinates
        """
        # Natural frequencies are already defined as class parameters, convert to Hz
        frq = np.array([[cls.w_1 / (2 * np.pi)], [cls.w_2 / (2 * np.pi)]])
        eig = np.eye(cls.ndof_free)

        # Initial position taken as zero for both modal coordinates
        pose0 = np.zeros(cls.ndof_free)

        return eig, frq, pose0

    @classmethod
    def time_solve(cls, omega, F, T, X, pose_base, cont_params, sensitivity=True, fulltime=False):
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        continuation_parameter = cont_params["continuation"]["continuation_parameter"]
        N = cls.ndof_free
        twoN = 2 * N

        # Add position to increment and do time sim to get solution and sensitivities
        X0 = X + np.concatenate((pose_base, np.zeros(N)))
        t = np.linspace(0, T, nsteps + 1)
        # Initial conditions for the augmented system: X0, eye for monodromy, zeros for time and force sens
        all_ic = np.concatenate(
            (X0, np.eye(twoN).flatten(), np.zeros(twoN), np.zeros(twoN)))
        sol = np.array(
            odeint(cls.model_sens_ode, all_ic, t, args=(
                T, F), rtol=rel_tol, tfirst=True)
        )
        # unpack solution
        Xsol, M, dXdT, dXdF = (
            sol[:, :twoN],
            sol[-1, twoN: twoN + twoN**2].reshape(twoN, twoN),
            sol[-1, twoN + twoN**2: twoN + twoN**2 + twoN],
            sol[-1, twoN + twoN**2 + twoN:],
        )

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)
        # print(f"H: {H}")

        # Jacobian construction depends on continuation parameter
        dHdX0 = M - np.eye(twoN)
        # print(f"dHdX0: {dHdX0}, {dHdX0.shape}")
        gX_T = cls.model_ode(T, Xsol[-1, :], T, F)

        if continuation_parameter == "frequency":
            # For frequency continuation, include period sensitivity (dH/dT)
            dHdT = gX_T + dXdT
            J = np.concatenate((dHdX0, dHdT.reshape(-1, 1)), axis=1)
        elif continuation_parameter == "amplitude":
            # For amplitude continuation, include force amplitude sensitivity (dH/dF)
            dHdF = dXdF
            # print(f"dHdF: {dHdF}, {dHdF.shape}")
            J = np.concatenate((dHdX0, dHdF.reshape(-1, 1)), axis=1)
        # print(f"J: {J.shape}")
        # print(f"J: {J}")

        # solution pose and vel at time 0
        pose = Xsol[0, :N]
        vel = Xsol[0, N:]

        # Energy calculation
        # Kinetic energy: 0.5 * xdot^T * M * xdot
        # Potential energy: 0.5 * x^T * K * x + (1/4) * k_nl * (phi_L @ x)^4

        # The nonlinear force at the tip is a physical force: F_tip = k_nl * (phi_L @ x)**3.
        # In the equations of motion, this physical force must be projected into modal coordinates
        # as a vector using phi_L.T, so each mode receives a share of the tip force.
        # However, the potential energy stored in the nonlinear spring is a scalar, and depends only
        # on the physical tip displacement (u = phi_L @ x), not on its projection.
        # Therefore, for energy, we use (1/4) * k_nl * (phi_L @ x)**4, without any projection.

        phi_x_sol = cls.phi_L @ Xsol[:, :N].T
        E0 = (
            0.5 * np.einsum("ij,ij->i", Xsol[:, N:],
                            np.dot(cls.M, Xsol[:, N:].T).T)
            + 0.5 * np.einsum("ij,ij->i",
                              Xsol[:, :N], np.dot(cls.K, Xsol[:, :N].T).T)
            + 0.25 * cls.k_nl * phi_x_sol[0, :] ** 4
        )
        force_vec = np.array([F * np.sin(2 * np.pi / T * t), np.zeros_like(t)])
        force_vel = force_vec[0] * Xsol[:, N]
        damping_vel = np.einsum(
            "ij,ij->i", Xsol[:, N:], (cls.C @ Xsol[:, N:].T).T)
        E1 = np.array(
            [simpson(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1])
             for i in range(len(t))]
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
            phi_x_i = cls.phi_L @ x_i
            fnl_i = cls.k_nl * cls.phi_L.T @ (phi_x_i**3)
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
