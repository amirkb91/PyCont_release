import numpy as np
import scipy.linalg as spl
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx


class Cubic_Spring_jax:
    # parameters of nonlinear system EoM    MX'' + CX' + KX + fnl = F*sin(2pi/T*t)
    M = np.eye(2)
    Minv = np.eye(2)
    C = np.zeros((2, 2))
    K = np.array([[2, -1], [-1, 2]])
    Knl = 0.5

    @classmethod
    def update_model(cls, parameters):
        # update model definition depending on parameters
        if "force" in parameters["continuation"]["parameter"]:
            # forced continuation, update damping matrix
            cls.C = 0.05 * cls.M + 0.01 * cls.K

    @classmethod
    def model_ode(cls, t, X, args):
        # State equation of the mass spring system. Xdot(t) = g(X(t))
        T, F = args
        x = X[:2]
        xdot = X[2:]
        KX = cls.K @ x
        CXdot = cls.C @ xdot
        force = jnp.array([F * jnp.sin(2 * jnp.pi / T * t), 0.0])
        fnl = jnp.array([cls.Knl * x[0] ** 3, 0.0])
        Xdot = jnp.concatenate((xdot, cls.Minv @ (-KX - CXdot - fnl + force)))
        return Xdot

    @classmethod
    def eigen(cls):
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
    def periodicity(cls, F, T, X0, parameters):
        """
        Periodicity function and Jacobian Computed using JAX
        Returns:
        H : (4,)  = y(T) - y0 periodicity function
        J : (4,5) = dH/d[y0, p], where p is T or F per `continuation_parameter`.
                    Columns 0..3: dH/dy0
                    Column    4 : dH/dp
        """
        # Read parameters
        nsteps = int(parameters["shooting"]["steps_per_period"])
        active_is_T = parameters["continuation"]["parameter"] in ["force_freq", "period"]

        # Build solver pieces (static across calls for a fixed max_steps)
        term = dfx.ODETerm(cls.model_ode)
        solver = dfx.Tsit5()
        ctrl = dfx.ConstantStepSize()
        adj = dfx.RecursiveCheckpointAdjoint()
        max_steps = nsteps

        @jax.jit
        def _periodicity(z):
            # Single periodicity function on a concatenated input z = [y0(4), p(1)]
            # This lets us get the full Jacobian in one jacrev.
            y0_ = z[:4]
            p = z[4]
            T_ = jnp.where(active_is_T, p, T)
            F_ = jnp.where(active_is_T, F, p)
            dt_ = T_ / nsteps

            sol = dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=T_,
                dt0=dt_,
                y0=y0_,
                args=(T_, F_),
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=ctrl,
                adjoint=adj,
                max_steps=max_steps,
            )
            yT = jnp.squeeze(sol.ys)
            return yT - y0_  # H(y0, T, F)

        @jax.jit
        def _energy(z):
            # Compute energy value only (no gradients needed by caller)
            y0_ = z[:4]
            p = z[4]
            T_ = jnp.where(active_is_T, p, T)
            F_ = jnp.where(active_is_T, F, p)

            # Fixed grid save for vectorised energy computation
            ts = jnp.linspace(0.0, T_, nsteps + 1)
            dt_ = T_ / nsteps

            sol = dfx.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=T_,
                dt0=dt_,
                y0=y0_,
                args=(T_, F_),
                saveat=dfx.SaveAt(ts=ts),
                stepsize_controller=ctrl,
                adjoint=adj,
                max_steps=max_steps,
            )

            ys = jnp.asarray(sol.ys)  # (nsteps+1, 4)
            x = ys[:, :2]
            v = ys[:, 2:]

            M = jnp.asarray(cls.M)
            K = jnp.asarray(cls.K)
            C = jnp.asarray(cls.C)

            # Instantaneous energy E0
            kin = 0.5 * jnp.einsum("ij,ij->i", v, (M @ v.T).T)
            pot = 0.5 * jnp.einsum("ij,ij->i", x, (K @ x.T).T) + 0.25 * cls.Knl * (x[:, 0] ** 4)
            E0 = kin + pot

            # Work - dissipation integral via trapezoid rule
            f0 = F_ * jnp.sin(2 * jnp.pi * ts / T_)
            force_vel = f0 * v[:, 0]
            damping = jnp.einsum("ij,ij->i", v, (C @ v.T).T)
            integrand = force_vel - damping
            ymid = 0.5 * (integrand[1:] + integrand[:-1])
            E1 = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(ymid) * dt_])

            E = E0 + E1
            energy = jnp.max(E)
            # Do not propagate gradients of energy back to callers
            return jax.lax.stop_gradient(energy)

        # Pack inputs: z = [y0, p] where p is the active parameter
        p0 = T if active_is_T else F
        z0 = jnp.concatenate([X0, jnp.array([p0])])

        # JIT the Jacobian
        jac_periodicity = jax.jit(jax.jacrev(_periodicity))

        # Get outputs
        H = _periodicity(z0)
        J = jac_periodicity(z0)
        energy = _energy(z0)
        return H, J, energy
