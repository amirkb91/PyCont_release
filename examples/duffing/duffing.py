import numpy as np
from scipy.integrate import solve_ivp, simpson
import scipy.linalg as spl


class Duffing:
    """
    fixed parameters of the model
    xddot + delta*xdot + alpha*x + beta*x^3 = F*sin(2pi/T*t)
    """

    alpha, beta = 1.0, 4.0
    delta = 0.0

    @classmethod
    def update_model(cls, parameters):
        # update model definition depending on parameters
        if "force" in parameters["continuation"]["parameter"]:
            # forced continuation, update damping matrix
            cls.delta = 0.1

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
    def model_ode_with_sens(cls, t, ic, T, F):
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
    def eigen(cls):
        """
        Eigenvalue and eigenvector of the model
        """
        frq = np.float64(cls.alpha)  # natural frequency
        frq = np.sqrt(frq) / (2 * np.pi)
        eig = np.array([1.0])

        # select  scaling
        scale = 0.01
        X0 = scale * eig
        T0 = 1 / frq

        # add zeros for velocity degrees of freedom
        X0 = np.append(X0, np.zeros_like(X0))

        return X0, T0

    @classmethod
    def periodicity(cls, F, T, X, parameters):
        """
        Compute periodicity function and it's sensitivity
        """
        nsteps = parameters["shooting"]["steps_per_period"]
        rel_tol = parameters["shooting"]["integration_tolerance"]
        continuation_parameter = parameters["continuation"]["parameter"]

        # Run time simulation
        t = np.linspace(0, T, nsteps + 1)
        # Initial conditions for the augmented system, eye for monodromy and zero for time and force sens
        all_ic = np.concatenate((X, np.eye(2).flatten(), np.zeros(2), np.zeros(2)))

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
        Xsol, M, dXdT, dXdF = sol[:, :2], sol[-1, 2:6].reshape(2, 2), sol[-1, 6:8], sol[-1, 8:]

        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]

        # Jacobian construction
        dHdX0 = M - np.eye(2)
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
            0.5 * (Xsol[:, 1] ** 2 + cls.alpha * Xsol[:, 0] ** 2)
            + 0.25 * cls.beta * Xsol[:, 0] ** 4
        )
        force_vel = F * np.sin(2 * np.pi / T * t) * Xsol[:, 1]
        damping_vel = cls.delta * Xsol[:, 1] ** 2
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

        # Time grid over one period
        t = np.linspace(0, T, nsteps + 1)

        # Integrate the system dynamics
        result = solve_ivp(
            cls.model_ode,
            t_span=[0, T],
            y0=X,
            t_eval=t,
            args=(T, F),
            rtol=rel_tol,
            method="RK45",
        )
        Xsol = result.y.T

        # Split into position and velocity
        increment = Xsol[:, :1]
        velocity = Xsol[:, 1:]

        # Calculate acceleration for all time steps (xddot = -f + force)
        acceleration = np.zeros_like(increment)
        for i in range(len(t)):
            x_i = increment[i, 0]
            xdot_i = velocity[i, 0]
            f_i = cls.delta * xdot_i + cls.alpha * x_i + cls.beta * x_i**3
            force_i = F * np.sin(2 * np.pi / T * t[i])
            acceleration[i, 0] = -f_i + force_i

        # Lagrangian
        # L = (
        #     0.5 * Xsol[:, 1]**2 - 0.5 * cls.alpha * Xsol[:, 0]**2 - 0.25 * cls.beta * Xsol[:, 0]**4
        # )

        return increment, velocity, acceleration

    @classmethod
    def time_simulate_with_monodromy(cls, F, T, X, parameters):
        """
        Perform time simulation with monodromy matrix calculation.
        """
        nsteps = parameters["shooting"]["steps_per_period"]
        rel_tol = parameters["shooting"]["integration_tolerance"]

        # Time grid over one period
        t = np.linspace(0, T, nsteps + 1)

        # Initial conditions: state + identity for monodromy + zeros for parameter sensitivities
        all_ic = np.concatenate((X, np.eye(2).flatten(), np.zeros(2), np.zeros(2)))

        # Integrate the augmented system to obtain monodromy
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
        Xsol = sol[:, :2]
        M = sol[-1, 2:6].reshape(2, 2)

        # Split into position and velocity
        increment = Xsol[:, :1]
        velocity = Xsol[:, 1:]

        # Calculate acceleration for all time steps (xddot = -f + force)
        acceleration = np.zeros_like(increment)
        for i in range(len(t)):
            x_i = increment[i, 0]
            xdot_i = velocity[i, 0]
            f_i = cls.delta * xdot_i + cls.alpha * x_i + cls.beta * x_i**3
            force_i = F * np.sin(2 * np.pi / T * t[i])
            acceleration[i, 0] = -f_i + force_i

        return increment, velocity, acceleration, M

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
