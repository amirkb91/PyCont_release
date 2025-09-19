import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg as spl
import matplotlib.pyplot as plt

phi_L = np.array([[-7.38213652279914, 7.36082686754947]])


def model_ode(t, X, F):
    """
    2-DOF nonlinear beam system ODE
    Modal equation: M*q_ddot + C*q_dot + K*q + f_nl(q) = f_ext
    State vector X = [q1, q2, q1_dot, q2_dot]
    """

    # Beam parameters for two mode system
    k_nl = 4_250_000
    w_1 = 91.734505484821950
    w_2 = 3.066194429903638e02
    zeta_1 = 0.03
    zeta_2 = 0.09

    # Modal Matrices
    M = np.eye(2)
    C = np.array([[2 * zeta_1 * w_1, 0], [0, 2 * zeta_2 * w_2]])
    K = np.array([[w_1**2, 0], [0, w_2**2]])
    Minv = spl.inv(M)

    # State equation: Xdot(t) = [q_dot; M^(-1)*(-K*q - C*q_dot - f_nl + f_ext)]
    x = X[:2]  # Modal coordinates [q1, q2]
    xdot = X[2:]  # Modal velocities [q1_dot, q2_dot]

    # Linear forces
    KX = K @ x
    CXdot = C @ xdot

    # External forcing (applied to first mode only)
    force = np.array([F * np.cos(2 * np.pi * 10 * t), 0])

    # Nonlinear force: f_nl = k_nl * phi_L^T * (phi_L * q)^3
    fnl = k_nl * phi_L.T @ (phi_L @ x) ** 3

    # State derivative
    Xdot = np.concatenate((xdot, Minv @ (-KX - CXdot - fnl + force)))
    return Xdot


# Simulation parameters
F = 20  # Forcing amplitude
t = np.linspace(0, 1, 1000)

# Initial conditions: [q1, q2, q1_dot, q2_dot]
# all_ic = np.array([-2.6e-4, -3e-4, -0.066, -0.011])
all_ic = np.array([0, 0, 0, 0])

# Solve the ODE
print("Solving 2-DOF nonlinear beam system...")
sol = solve_ivp(model_ode, (t[0], t[-1]), all_ic, method="RK45", t_eval=t, args=(F,))
sol = sol.y
print(f"Time span: {t[0]:.2f} to {t[-1]:.2f} seconds")

# Extract solution (modal and physical)
q1 = sol[0, :]  # First modal coordinate
q2 = sol[1, :]  # Second modal coordinate
q1_dot = sol[2, :]  # First modal velocity
q2_dot = sol[3, :]  # Second modal velocity
disp = phi_L @ sol[:2, :]
vel = phi_L @ sol[2:, :]

# Create plots
plt.style.use("default")

# Modal coordinates
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Beam Modal Time Response", fontsize=16)

axes[0, 0].plot(t, q1, "b-", linewidth=1.5, label="q₁(t)")
axes[0, 0].set_xlabel("Time [s]")
axes[0, 0].set_ylabel("Modal Coordinate q₁")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[0, 1].plot(t, q1_dot, "b-", linewidth=1.5, label="q̇₁(t)")
axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].set_ylabel("Modal Velocity q̇₁")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1, 0].plot(t, q2, "r-", linewidth=1.5, label="q₂(t)")
axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].set_ylabel("Modal Coordinate q₂")
axes[1, 0].grid(True, alpha=0.3)
axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1, 1].plot(t, q2_dot, "r-", linewidth=1.5, label="q̇₂(t)")
axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].set_ylabel("Modal Velocity q̇₂")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[0, 2].plot(q1, q1_dot, "b-", linewidth=1.5, alpha=0.7)
axes[0, 2].plot(q1[0], q1_dot[0], "go", markersize=8, label="Start")
axes[0, 2].plot(q1[-1], q1_dot[-1], "ro", markersize=8, label="End")
axes[0, 2].set_xlabel("q₁")
axes[0, 2].set_ylabel("q̇₁")
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()
axes[0, 2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

axes[1, 2].plot(q2, q2_dot, "r-", linewidth=1.5, alpha=0.7)
axes[1, 2].plot(q2[0], q2_dot[0], "go", markersize=8, label="Start")
axes[1, 2].plot(q2[-1], q2_dot[-1], "ro", markersize=8, label="End")
axes[1, 2].set_xlabel("q₂")
axes[1, 2].set_ylabel("q̇₂")
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()
axes[1, 2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

# Physical coordinates
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Beam Tip Physical Time Response", fontsize=16)

axes[0].plot(t, disp[0, :], "-", linewidth=1.5, color="tab:green")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Disp")
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1].plot(t, vel[0, :], "-", linewidth=1.5, color="tab:purple")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity")
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[2].plot(disp[0, :], vel[0, :], "-", linewidth=1.5, alpha=0.7)
axes[2].plot(disp[0, 0], vel[0, 0], "go", markersize=8, label="Start")
axes[2].plot(disp[0, -1], vel[0, -1], "ro", markersize=8, label="End")
axes[2].set_xlabel("Disp")
axes[2].set_ylabel("Velocity")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

plt.tight_layout()
plt.show()
