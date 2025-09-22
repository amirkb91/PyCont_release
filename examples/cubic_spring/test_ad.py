import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx
import logging

logging.getLogger("jax").setLevel(logging.ERROR)

K = jnp.array([[2.0, -1.0], [-1.0, 2.0]])
C = 0.05 * jnp.eye(2) + 0.01 * K
Knl = 0.5


def cubic_spring_rhs(t, y, args):
    T, F = args
    x, v = y[:2], y[2:]
    Kx = K @ x
    Cv = C @ v
    fnl = jnp.array([Knl * x[0] ** 3, 0.0])
    omega = 2.0 * jnp.pi / T
    force = jnp.array([F * jnp.sin(omega * t), 0.0])
    xdot = v
    vdot = -(Kx + Cv + fnl) + force
    return jnp.concatenate([xdot, vdot])


def perfxn_and_jac(y0, T, F, N, continuation_parameter):
    """
    Periodicity function and Jacobian
    Returns:
      H : (4,)  = y(T) - y0 periodicity function
      J : (4,5) = dH/d[y0, p], where p is T or F per `continuation_parameter`.
                 Columns 0..3: dH/dy0
                 Column    4 : dH/dp
    """
    # Build solver pieces (static across calls for a fixed N)
    term = dfx.ODETerm(cubic_spring_rhs)
    solver = dfx.Tsit5()
    ctrl = dfx.ConstantStepSize()
    adj = dfx.RecursiveCheckpointAdjoint()
    max_steps = int(N)

    active_is_T = continuation_parameter.upper() == "T"

    @jax.jit
    def _periodicity(z):
        # Single periodicity function on a concatenated input z = [y0(4), p(1)]
        # This lets us get the full Jacobian in one jacrev.
        y0_ = z[:4]
        p = z[4]
        T_ = jnp.where(active_is_T, p, T)
        F_ = jnp.where(active_is_T, F, p)
        dt_ = T_ / N

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

    # JIT the Jacobian
    jac_periodicity = jax.jit(jax.jacrev(_periodicity))

    # Pack inputs: z = [y0, p] where p is the active parameter
    p0 = T if active_is_T else F
    z0 = jnp.concatenate([y0, jnp.array([p0])])

    H = _periodicity(z0)
    J = jac_periodicity(z0)
    return H, J


# example use
y0 = jnp.array([-3.880e-1, 1.768e-2, 1.392e1, -2.025e0])
T, F = 2.0, 1.397e1

H, J = perfxn_and_jac(y0, T, F, N=300, continuation_parameter="F")
print("H =", H)
print("J =\n", J)

# import matplotlib.pyplot as plt
# plt.plot(ts, ys[:,0], label="x1")
# plt.plot(ts, ys[:,1], label="x2")
# plt.plot(ts, ys[:,2], "--", label="v1")
# plt.plot(ts, ys[:,3], "--", label="v2")
# plt.xlabel("t"); plt.ylabel("state"); plt.legend(); plt.show()
