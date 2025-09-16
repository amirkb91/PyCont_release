import numpy as np
import scipy.linalg as spl

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController, Dopri5


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
    def acc(cls, pred_acc):
        cls.pred_acc = jax.jit(pred_acc)

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
    def model_ode(cls, t, X, args):
        T, F = args
        # State equation of the mass spring system. Xdot(t) = g(X(t))
        x = X[: cls.ndof_free]
        xdot = X[cls.ndof_free:]

        force = jnp.array([F * jnp.sin(2 * jnp.pi / T * t), 0])
        _X = jnp.concatenate(
            (x[None, :, None], xdot[None, :, None]), axis=-1)
        _force = force[None, :, None]
        xddot = cls.pred_acc(_X, _force)

        Xdot = jnp.concatenate((xdot, xddot[0]))
        return Xdot

    @classmethod
    def time_solve(cls, omega, F, T, X, pose_base, cont_params, sensitivity=True, fulltime=False):
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]
        continuation_parameter = cont_params["continuation"]["continuation_parameter"]
        N = cls.ndof_free
        twoN = 2 * N

        # Add position to increment and do time sim to get solution
        X0 = X + np.concatenate((pose_base, np.zeros(N)))
        t = np.linspace(0, T, nsteps + 1)

        # ODE Solver setup
        term = ODETerm(cls.model_ode)
        solver = Tsit5()
        saveat = SaveAt(ts=jnp.linspace(0, T, nsteps + 1))

        sol = diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=X0, args=(
            T, F), saveat=saveat, stepsize_controller=PIDController(rtol=rel_tol, atol=rel_tol), max_steps=1000000)
        Xsol = sol.ys

        # Periodicity
        def periodicity(X0, T, F):
            t = np.linspace(0, T, nsteps + 1)
            Xsol = diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1]-t[0], y0=X0, args=(
                T, F), saveat=saveat, stepsize_controller=PIDController(rtol=rel_tol, atol=rel_tol), max_steps=1000000)
            H = Xsol.ys[-1, :] - Xsol.ys[0, :]
            return H.reshape(-1, 1)

        H = periodicity(X0, T, F)
        # print(f"H: {H}")

        # Monodromy
        dHdX0 = jax.jacrev(periodicity, argnums=0)(X0, T, F)
        dHdX0 = dHdX0.squeeze()
        # print(f"dHdX0: {dHdX0}, {dHdX0.shape}")

        if continuation_parameter == "frequency":
            # For frequency continuation, include period sensitivity (dH/dT)
            dHdT = jax.jacrev(periodicity, argnums=1)(X0, T, F)
            J = np.concatenate((dHdX0, dHdT), axis=1)
        elif continuation_parameter == "amplitude":
            # For amplitude continuation, include force amplitude sensitivity (dH/dF)
            dHdF = jax.jacrev(periodicity, argnums=2)(X0, T, F)
            # print(f"dHdF: {dHdF}, {dHdF.shape}")
            J = np.concatenate((dHdX0, dHdF), axis=1)
        # print(f"J: {J.shape}")
        # print(f"J: {J}")

        # solution pose and vel at time 0
        pose = Xsol[0, :N]
        vel = Xsol[0, N:]

        # Energy calculation
        energy = 0.0

        # Calculate acceleration for full time output
        acc_time = np.zeros_like(Xsol[:, :N])
        for i in range(len(t)):
            x_i = Xsol[i, :N]
            xdot_i = Xsol[i, N:]
            force_i = np.array([F * np.sin(2 * np.pi / T * t[i]), 0])
            _Xsol_i = np.concatenate(
                (x_i[None, :, None], xdot_i[None, :, None]), axis=-1)
            acc_time[i, :] = cls.pred_acc(
                _Xsol_i, force_i[None, :, None])[0, :]

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
