import numpy as np
from scipy.integrate import odeint, simpson
import scipy.linalg as spl


# parameters of nonlinear system EoM    MX'' + CX' + KX + fnl = F*sin(2pi/T*t)
M = np.eye(2)
K = np.array([[2, -1], [-1, 2]])
C = 0.05 * M + 0.01 * K
Knl = 0.5
Minv = spl.inv(M)


def model_ode(t, X, T, F):
    # State equation of the mass spring system. Xdot(t) = g(X(t))
    x = X[:2]
    xdot = X[2:]
    KX = K @ x
    CXdot = C @ xdot
    force = np.array([F * np.sin(2 * np.pi / T * t), 0])
    fnl = np.array([Knl * x[0] ** 3, 0])
    Xdot = np.concatenate((xdot, Minv @ (-KX - CXdot - fnl + force)))
    return Xdot


def model_sens_ode(t, ic, T, F):
    """
    Augmented ODE of model + sensitivities, to be solved together
    System: Xdot(t) = g(X(t))
    Monodromy: dXdX0dot = dg(X)dX . dXdX0
    Time sens: dXdTdot = dg(X)dX . dXdT + dgdT
    Force sens: dXdFdot = dg(X)dX . dXdF + dgdF
    """
    N = 2
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
    fnl = np.array([Knl * x[0] ** 3, 0])

    # Force derivatives
    dforce_dT = np.array([F * np.cos(2 * np.pi / T * t) * (-2 * np.pi * t / T**2), 0])
    dforce_dF = np.array([np.sin(2 * np.pi / T * t), 0])

    # System dynamics
    Xdot = np.concatenate((xdot, Minv @ (-KX - CXdot - fnl + force)))

    # Jacobian of system dynamics with respect to state
    dgdX = np.zeros((twoN, twoN))
    dgdX[:N, N:] = np.eye(N)  # dx/dt = xdot
    dgdX[N:, :N] = -Minv @ (K + np.array([[Knl * 3 * x[0] ** 2, 0], [0, 0]]))  # d(xddot)/dx
    dgdX[N:, N:] = -Minv @ C  # d(xddot)/d(xdot)

    # Partial derivatives with respect to parameters
    dgdT = np.concatenate([np.zeros(N), Minv @ dforce_dT])
    dgdF = np.concatenate([np.zeros(N), Minv @ dforce_dF])

    # Sensitivity calculations
    dXdX0dot = dgdX @ dXdX0.reshape(twoN, twoN)
    dXdTdot = dgdX @ dXdT.reshape(twoN, 1) + dgdT.reshape(-1, 1)
    dXdFdot = dgdX @ dXdF.reshape(twoN, 1) + dgdF.reshape(-1, 1)

    return np.concatenate([Xdot, dXdX0dot.flatten(), dXdTdot.flatten(), dXdFdot.flatten()])


def time_solve(X, T, F):

    N = 2
    twoN = 2 * N

    # Add position to increment and do time sim to get solution and sensitivities
    X0 = X
    t = np.linspace(0, T, 301)
    # Initial conditions for the augmented system: X0, eye for monodromy, zeros for time and force sens
    all_ic = np.concatenate((X0, np.eye(twoN).flatten(), np.zeros(twoN), np.zeros(twoN)))
    sol = np.array(odeint(model_sens_ode, all_ic, t, args=(T, F), rtol=1e-6, tfirst=True))
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
    gX_T = model_ode(T, Xsol[-1, :], T, F)

    # For amplitude continuation, include force amplitude sensitivity (dH/dF)
    dHdF = dXdF
    J = np.concatenate((dHdX0, dHdF.reshape(-1, 1)), axis=1)

    return t, Xsol, Xsol[-1, :], J


import matplotlib.pyplot as plt

y0 = np.array([-3.880e-1, 1.768e-2, 1.392e1, -2.025e0])
T, F = 2.0, 1.397e1

ts, ys, yT, dYdY0 = time_solve(y0, T, F)
print("H =", yT - y0)
print("J =\n", dYdY0)

plt.plot(ts, ys[:, 0], label="x1")
plt.plot(ts, ys[:, 1], label="x2")
plt.plot(ts, ys[:, 2], "--", label="v1")
plt.plot(ts, ys[:, 3], "--", label="v2")
plt.xlabel("t")
plt.ylabel("state")
plt.legend()
plt.show()
