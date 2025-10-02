import numpy as np
from scipy.integrate import solve_ivp, simpson
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
    def update_model(cls, parameters):
        """
        update parameters if continuation is forced
        """
        if "force" in parameters["continuation"]["parameter"]:
            zeta_1 = 0.03
            zeta_2 = 0.09
            cls.C = np.array([[2 * zeta_1 * cls.w_1, 0], [0, 2 * zeta_2 * cls.w_2]])

    @classmethod
    def model_ode(cls, t, X, T, F):
        # State equation of the mass spring system. Xdot(t) = g(X(t))
        x = X[: cls.ndof_free]
        xdot = X[cls.ndof_free :]
        KX = cls.K @ x
        CXdot = cls.C @ xdot
        force = np.array([F * np.sin(2 * np.pi / T * t), 0])
        phi_x = cls.phi_L @ x  # Physical displacement at nonlinear location
        fnl = cls.k_nl * cls.phi_L.T @ (phi_x**3)
        Xdot = np.concatenate((xdot, cls.Minv @ (-KX - CXdot - fnl + force)))
        return Xdot

    @classmethod
    def model_ode_with_sens(cls, t, ic, T, F):
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
            ic[twoN : twoN + twoN**2],
            ic[twoN + twoN**2 : twoN + twoN**2 + twoN],
            ic[twoN + twoN**2 + twoN :],
        )

        x = X0[:N]
        xdot = X0[N:]
        KX = K @ x
        CXdot = C @ xdot
        force = np.array([F * np.sin(2 * np.pi / T * t), 0])
        phi_x = phi_L @ x  # Physical displacement at nonlinear location
        fnl = k_nl * phi_L.T @ (phi_x**3)

        # Force derivatives
        dforce_dT = np.array([F * np.cos(2 * np.pi / T * t) * (-2 * np.pi * t / T**2), 0])
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
    def eigen(cls):
        """
        For modal model: frequencies are already known, eigenvectors are identity
        since we're working in modal coordinates
        """
        # Natural frequencies are already defined as class parameters, convert to Hz
        frq = np.array([[cls.w_1 / (2 * np.pi)], [cls.w_2 / (2 * np.pi)]])
        eig = np.eye(cls.ndof_free)

        # select mode and scaling
        mode = 1
        scale = 1e-5
        X0 = scale * eig[:, mode - 1]
        T0 = 1 / frq[mode - 1]

        # add zeros for velocity degrees of freedom
        X0 = np.append(X0, np.zeros_like(X0))

        return X0, T0

    @classmethod
    def periodicity(cls, F, T, X, parameters):
        # Compute periodicity function and it's sensitivity
        nsteps = parameters["shooting"]["steps_per_period"]
        rel_tol = parameters["shooting"]["integration_tolerance"]
        continuation_parameter = parameters["continuation"]["parameter"]

        n_dof = np.size(X)
        n_dof_2 = n_dof // 2

        # Run time simulation
        t = np.linspace(0, T, nsteps + 1)
        # Initial conditions for the augmented system: X0, eye for monodromy, zeros for time and force sens
        all_ic = np.concatenate((X, np.eye(n_dof).flatten(), np.zeros(n_dof), np.zeros(n_dof)))

        # solve_ivp integration
        result = solve_ivp(
            cls.model_ode_with_sens,
            t_span=[0, T],
            y0=all_ic,
            t_eval=t,
            args=(T, F),
            rtol=rel_tol,
            method="RK45",
        )
        sol = result.y.T
        # unpack solution
        Xsol, M, dXdT, dXdF = (
            sol[:, :n_dof],
            sol[-1, n_dof : n_dof + n_dof**2].reshape(n_dof, n_dof),
            sol[-1, n_dof + n_dof**2 : n_dof + n_dof**2 + n_dof],
            sol[-1, n_dof + n_dof**2 + n_dof :],
        )

        # Periodicity condition
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

        # Energy calculation
        # Kinetic energy: 0.5 * xdot^T * M * xdot
        # Potential energy: 0.5 * x^T * K * x + (1/4) * k_nl * (phi_L @ x)^4

        # The nonlinear force at the tip is a physical force: F_tip = k_nl * (phi_L @ x)**3.
        # In the equations of motion, this physical force must be projected into modal coordinates
        # as a vector using phi_L.T, so each mode receives a share of the tip force.
        # However, the potential energy stored in the nonlinear spring is a scalar, and depends only
        # on the physical tip displacement (u = phi_L @ x), not on its projection.
        # Therefore, for energy, we use (1/4) * k_nl * (phi_L @ x)**4, without any projection.

        phi_x_sol = cls.phi_L @ Xsol[:, :n_dof_2].T
        E0 = (
            0.5 * np.einsum("ij,ij->i", Xsol[:, n_dof_2:], np.dot(cls.M, Xsol[:, n_dof_2:].T).T)
            + 0.5 * np.einsum("ij,ij->i", Xsol[:, :n_dof_2], np.dot(cls.K, Xsol[:, :n_dof_2].T).T)
            + 0.25 * cls.k_nl * phi_x_sol[0, :] ** 4
        )
        force_vec = np.array([F * np.sin(2 * np.pi / T * t), np.zeros_like(t)])
        force_vel = force_vec[0] * Xsol[:, n_dof_2]
        damping_vel = np.einsum("ij,ij->i", Xsol[:, n_dof_2:], (cls.C @ Xsol[:, n_dof_2:].T).T)
        E1 = np.array(
            [simpson(force_vel[: i + 1] - damping_vel[: i + 1], t[: i + 1]) for i in range(len(t))]
        )
        E = E0 + E1
        energy = np.max(E)

        return H, J, energy

    @classmethod
    def time_simulate(cls, F, T, X, parameters):
        """
        Perform time simulation only, returning position increment, velocity, and acceleration.
        """
        nsteps = parameters["shooting"]["steps_per_period"]
        rel_tol = parameters["shooting"]["integration_tolerance"]

        n_dof = np.size(X)
        n_dof_2 = n_dof // 2

        # Run time simulation
        t = np.linspace(0, T, nsteps + 1)

        # solve_ivp integration for system dynamics only
        result = solve_ivp(
            cls.model_ode, t_span=[0, T], y0=X, t_eval=t, args=(T, F), rtol=rel_tol, method="RK45"
        )
        Xsol = result.y.T

        # Split into position and velocity
        increment = Xsol[:, :n_dof_2]
        velocity = Xsol[:, n_dof_2:]

        # Calculate acceleration for all time steps
        acceleration = np.zeros_like(increment)
        for i in range(len(t)):
            x_i = increment[i, :]
            xdot_i = velocity[i, :]
            KX = cls.K @ x_i
            CXdot = cls.C @ xdot_i
            force_i = np.array([F * np.sin(2 * np.pi / T * t[i]), 0])
            phi_x_i = cls.phi_L @ x_i
            fnl_i = cls.k_nl * cls.phi_L.T @ (phi_x_i**3)
            acceleration[i, :] = cls.Minv @ (-KX - CXdot - fnl_i + force_i)

        return increment, velocity, acceleration

    @classmethod
    def time_simulate_with_monodromy(cls, F, T, X, parameters):
        """
        Perform time simulation with monodromy matrix calculation.
        """
        nsteps = parameters["shooting"]["steps_per_period"]
        rel_tol = parameters["shooting"]["integration_tolerance"]

        n_dof = np.size(X)
        n_dof_2 = n_dof // 2

        # Run time simulation with monodromy matrix calculation
        t = np.linspace(0, T, nsteps + 1)
        # Initial conditions: X0 + eye for monodromy + zeros for parameter sensitivities
        all_ic = np.concatenate((X, np.eye(n_dof).flatten(), np.zeros(n_dof), np.zeros(n_dof)))

        # solve_ivp integration with monodromy (using existing model_ode_with_sens)
        result = solve_ivp(
            cls.model_ode_with_sens,
            t_span=[0, T],
            y0=all_ic,
            t_eval=t,
            args=(T, F),
            rtol=rel_tol,
            method="RK45",
        )
        sol = result.y.T

        # Unpack solution and monodromy matrix
        Xsol = sol[:, :n_dof]
        M = sol[-1, n_dof : n_dof + n_dof**2].reshape(n_dof, n_dof)

        # Split into position and velocity
        increment = Xsol[:, :n_dof_2]
        velocity = Xsol[:, n_dof_2:]

        # Calculate acceleration for all time steps
        acceleration = np.zeros_like(increment)
        for i in range(len(t)):
            x_i = increment[i, :]
            xdot_i = velocity[i, :]
            KX = cls.K @ x_i
            CXdot = cls.C @ xdot_i
            force_i = np.array([F * np.sin(2 * np.pi / T * t[i]), 0])
            phi_x_i = cls.phi_L @ x_i
            fnl_i = cls.k_nl * cls.phi_L.T @ (phi_x_i**3)
            acceleration[i, :] = cls.Minv @ (-KX - CXdot - fnl_i + force_i)

        return increment, velocity, acceleration, M
