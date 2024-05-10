import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
import cmath

class Duffing_LNN:
    # Physical System to Predict:
    ## xddot + delta*xdot + alpha*x + beta*x^3 = F*cos(2pi/T*t + phi)
    # NOTE: LNN EoM is given as
    ## d/dt(dL/dxdot) + dD/dxdot - dL/dx = F*cos(2pi/T*t + phi)
    ## where L = T - V & D
    ## L, D -> Each predicted by separate NNs
    alpha, beta = 1.0, 1.0
    delta, F, phi = 0.0, 0.0, 0.0
    
    # finite element data, 1 dof model
    free_dof = np.array([0])
    ndof_all = 1
    ndof_fix = 0
    ndof_free = 1
    
    @classmethod
    def LNN_acceleration(cls, pred_acc):
        # Acceleration predicted by LNN
        cls.pred_acc = pred_acc
        
    @classmethod
    def LNN_energy(cls, pred_energy):
        # LNN & DNN
        cls.pred_energy = pred_energy

    @classmethod
    def forcing_parameters(cls, cont_params):
        # Parameters for forced continuation
        if cont_params["continuation"]["forced"]:
            cls.F = cont_params["forcing"]["amplitude"]
            cls.delta = cont_params["forcing"]["tau0"]
            cls.phi = cont_params["forcing"]["phase_ratio"] * np.pi
    
    @classmethod
    def model_ode(cls, t, X, T):
        """ODE in Euler-Lagrange Form
        Args:
            t: Time step
            X: State (disp. & vel.)
            T: Period associated with periodic motion
            L: Lagrangian Network
            D: Dissipation Network
        Returns:
            EoM in State Space Form --> Xdot(t) = g(X(t))
        """
        x = X[0]
        xdot = X[1]
        
        # Predict Acceleration
        force = cls.F * np.cos(2 * np.pi / T * t + cls.phi)
        # Repeat along axis for vmap to work
        X_arr = jnp.tile(X, (1, 1))
        force_arr = jnp.tile(force, (1, 1))
        xddot_arr = jax.jit(cls.pred_acc)(X_arr, force_arr)
        xddot = xddot_arr[0, -1]
   
        # State-space Vector for EoM
        Xdot = np.array([xdot, xddot])
        
        return Xdot

    @classmethod
    def model_sens_ode(cls, t, ic, T):
        """ODE in Euler-Lagrange Form, 
        This is the augemented ODE of model + sensitivities, to be solved together:
            # System: Xdot(t) = g(X(t))
            # Monodromy: dXdX0dot = dg(X)dX . dXdX0
            # Time sens: dXdTdot = dg(X)dX . dXdT + dgdT
        Args:
            t: Time step
            ic: Initial State
            T: Period associated with periodic motion
            L: Lagrangian Network
            D: Dissipation Network
        """
        X0, dXdX0, dXdT = ic[:2], ic[2:6], ic[6:]
        x = X0[0]
        xdot = X0[1]
        # REVIEW: Predict Acceleration
        def force_func(t, T):
            return cls.F * jnp.cos(2 * jnp.pi / T * t + cls.phi)
        force = force_func(t, T)
        
        # Repeat along axis for vmap to work
        X0_arr = jnp.tile(X0, (1, 1))
        force_arr = jnp.tile(force, (1, 1))
        
        xddot_arr = jax.jit(cls.pred_acc)(X0_arr, force_arr)
        xddot = xddot_arr[0, -1]
        
        force_der = jax.jacrev(force_func, argnums=1)(t, T)
        Xdot = np.array([xdot, xddot])
        
        dgdX = np.array([jax.jacrev(cls.pred_acc, argnums=0)(X0_arr, force_arr)[0, 0, 0, :], jax.jacrev(cls.pred_acc, argnums=0)(X0_arr, force_arr)[-1, -1, -1, :]])
        
        dgdT = np.array([0, force_der])
        dXdX0dot = dgdX @ dXdX0.reshape(2, 2)
        dXdTdot = dgdX @ dXdT.reshape(2, 1) + dgdT.reshape(-1, 1)

        return np.concatenate([Xdot, dXdX0dot.flatten(), dXdTdot.flatten()])

    @classmethod
    def eigen_solve(cls):
        frq = np.array([[cls.alpha]])
        frq = np.sqrt(frq) / (2 * np.pi)
        eig = np.array([[1.0]])

        # REVIEW: Initial position taken as zero
        omega = np.sqrt(frq)
        pose0 = cls.F * jnp.cos(cls.phi) / (-omega**2 + complex(0, omega*cls.delta) + cls.alpha)
        pose0 = pose0.real

        return eig, frq, pose0

    @classmethod
    def time_solve(cls, omega, T, X, pose_base, cont_params, sensitivity=True):
        nperiod = cont_params["shooting"]["single"]["nperiod"]
        nsteps = cont_params["shooting"]["single"]["nsteps_per_period"]
        rel_tol = cont_params["shooting"]["rel_tol"]

        # Add position to increment and do time sim to get solution and sensitivities
        X0 = X.copy()
        X0[0] += pose_base
        t = np.linspace(0, T * nperiod, nsteps * nperiod + 1)
        all_ic = np.concatenate((X0, np.eye(2).flatten(), np.zeros(2)))
        sol = odeint(cls.model_sens_ode, all_ic, t, args=(T * nperiod,), rtol=rel_tol, tfirst=True)
        Xsol, M, dXdT = sol[:, :2], sol[-1, 2:6].reshape(2, 2), sol[-1, 6:]
        
        # periodicity condition
        H = Xsol[-1, :] - Xsol[0, :]
        H = H.reshape(-1, 1)

        # Jacobian (dHdX0 and dHdt)
        dHdX0 = M - np.eye(2)
        gX_T = cls.model_ode(T * nperiod, Xsol[-1, :], T * nperiod) * nperiod

        # *** Time sensitivity is not working, do finite difference for now ***
        if cont_params["continuation"]["forced"]:
            # dHdt = gX_T + dXdT
            dHdT = cls.timesens_centdiff(X0, T)
        else:
            dHdT = gX_T

        J = np.concatenate((dHdX0, dHdT.reshape(-1, 1)), axis=1)

        # solution pose and vel over time
        pose_time = Xsol[:, 0].reshape(1, -1)
        vel_time = Xsol[:, 1].reshape(1, -1)

        # REVIEW: Energy: Hamiltonian from Lagrangian
        # dL/ddq * dq - L: https://physics.stackexchange.com/questions/190471/constructing-lagrangian-from-the-hamiltonian#:~:text=Given%20the%20Lagrangian%20L%20for,we%20need%20to%20know%20L.
        Lnn, Dnn = cls.pred_energy(Xsol[:, 0].reshape(-1, 1), Xsol[:, 1].reshape(-1, 1))
        dLddq = jax.jacobian(cls.pred_energy, 1)(Xsol[:, 0].reshape(-1, 1), Xsol[:, 1].reshape(-1, 1))[0]
        # TODO: Check if Lnn or ham is scalar; add statistics to plot
        ham = jnp.dot(Xsol[:, 1], dLddq) - Lnn
        energy = np.max(ham)
        
        # M = jax.hessian(cls.pred_energy, 1)(X[0], X[1])[0]
        # K = jax.jacobian(cls.pred_energy, 0)(X[0], X[1])[0]
        # D = jax.jacobian(cls.pred_energy, 1)(X[0], X[1])[1]
        # GC1 = jax.jacfwd(jax.jacrev(cls.pred_energy, 1), 0)(X[0], X[1])[0]
        # GCQ = jnp.dot(GC1, X[1])

        cvg = True
        return H, J, pose_time, vel_time, energy, cvg

    @classmethod
    def get_fe_data(cls):
        return {
            "free_dof": cls.free_dof,
            "ndof_all": cls.ndof_all,
            "ndof_fix": cls.ndof_fix,
            "ndof_free": cls.ndof_free,
        }

    @classmethod
    def timesens_centdiff(cls, X0, T):
        # central difference calculation of the time sensitivity matrix
        # can be used to check values from ode
        eps = 1e-8
        Tplus = T + eps
        t = np.linspace(0, Tplus, 301)
        XTplus = np.array(odeint(cls.model_ode, X0, t, args=(Tplus,), tfirst=True))[-1, :]
        Tmins = T - eps
        t = np.linspace(0, Tmins, 301)
        XTmins = np.array(odeint(cls.model_ode, X0, t, args=(Tmins,), tfirst=True))[-1, :]
        dXdT = (XTplus - XTmins) / (2 * eps)
        return dXdT
