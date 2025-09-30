import numpy as np
import scipy.linalg as spl


def psacont(self):
    """
    Pseudo-arc length continuation method.

    This method performs continuation by correcting both the solution variables X,
    and the continuation parameter (T or F) at predicted values, with corrections
    made orthogonal to the tangent vector.
    """
    # Starting point solution
    X = self.X0
    T = self.T0
    F = self.F0
    tgt = self.tgt0

    # Read parameters
    parameters = self.prob.parameters
    continuation_parameter = parameters["continuation"]["parameter"]
    step_size = parameters["continuation"]["initial_step_size"]
    max_iterations = parameters["continuation"]["max_iterations"]
    min_iterations = parameters["continuation"]["min_iterations"]
    tolerance = parameters["continuation"]["corrections_tolerance"]
    tangent_predictor = parameters["continuation"]["tangent_predictor"]
    forced = "force" in parameters["continuation"]["parameter"]

    # Continuation limits
    max_points = parameters["continuation"]["num_points"]
    min_param = parameters["continuation"]["min_parameter_value"]
    max_param = parameters["continuation"]["max_parameter_value"]

    # Set up continuation parameter abstraction and direction
    # fmt: off
    if continuation_parameter == "force_freq" or continuation_parameter == "period":
        param_current = T
        def get_param_value(): return T
        def set_param_value(val): nonlocal T; T = val
        def get_period(): return T
        def get_amplitude(): return F
        def get_cont_param_for_bounds(): return 1 / T  # para file bounds are frequency
        direction = parameters["continuation"]["direction"] * -1
    elif continuation_parameter == "force_amp":
        param_current = F
        def get_param_value(): return F
        def set_param_value(val): nonlocal F; F = val
        def get_period(): return T
        def get_amplitude(): return F
        def get_cont_param_for_bounds(): return F
        direction = parameters["continuation"]["direction"]
    # fmt: on

    # --- MAIN CONTINUATION LOOP
    itercont = 1
    while itercont <= max_points:
        # Prediction step along tangent
        param_pred = param_current + tgt[-1] * step_size * direction
        X_pred = X + tgt[:-1] * step_size * direction
        set_param_value(param_pred)

        # Check bounds before doing corrections
        if not (min_param <= get_cont_param_for_bounds() <= max_param):
            print(
                f"Continuation parameter {get_cont_param_for_bounds():.2e} outside specified bounds [{min_param:.2e}, {max_param:.2e}]."
            )
            break

        # Correction iterations
        itercorrect = 0
        while True:
            try:
                H, J, energy = self.prob.zero_function(
                    get_amplitude(), get_period(), X_pred, parameters
                )
            except Exception as e:
                print(f"Error evaluating zero function: {e}")
                converged_now = False
                break

            residual = spl.norm(H) / max(spl.norm(X_pred), 1e-12)

            # Check convergence criteria
            converged_now = residual < tolerance and itercorrect >= min_iterations
            if converged_now:
                break

            # Check divergence criteria
            if residual > 1e10:
                converged_now = False
                break

            # Maximum iterations reached
            if itercorrect >= max_iterations:
                converged_now = False
                break

            self.log.screenout(
                iter=itercont,
                correct=itercorrect,
                res=residual,
                freq=1 / get_period(),
                amp=get_amplitude(),
                energy=energy,
                step=direction * step_size,
            )

            # Compute corrections orthogonal to tangent
            # Augment the Jacobian with phase condition and tangent
            J_corr, h = self.add_phase_condition(J)
            J_corr = np.vstack([J_corr, tgt])

            Z = np.concatenate([H, h @ X_pred, np.zeros(1)])
            if not forced:
                dxt = spl.lstsq(J_corr, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            elif forced:
                dxt = spl.solve(J_corr, -Z, check_finite=False)

            # Apply correction
            X_pred += dxt[:-1]
            param_new = get_param_value() + dxt[-1]
            set_param_value(param_new)
            itercorrect += 1

        if converged_now:
            # Compute new tangent with converged solution
            if tangent_predictor == "secant":
                tgt_next = np.concatenate((X_pred - X, [get_param_value() - param_current]))
                tgt_next /= spl.norm(tgt_next)
                tgt_inner = np.dot(tgt_next, tgt)
                direction = np.sign(direction * tgt_inner)

            elif tangent_predictor == "nullspace_previous":
                # Augment the Jacobian with phase condition and tangent
                # Approximate the null space of the Jacobian using the previous tangent (Keller et al.)
                J_tgt, _ = self.add_phase_condition(J)
                J_tgt = np.vstack([J_tgt, tgt])

                Z = np.zeros((J_tgt.shape[0], 1))
                Z[-1] = 1.0
                if not forced:
                    tgt_next = spl.lstsq(
                        J_tgt, Z, cond=None, check_finite=False, lapack_driver="gelsd"
                    )[0][:, 0]
                elif forced:
                    tgt_next = spl.solve(J_tgt, Z, check_finite=False)[:, 0]
                tgt_next /= spl.norm(tgt_next)
                tgt_inner = np.dot(tgt_next, tgt)
                # Safeguard: if numerical error yields a negative dot product, flip the tangent
                if tgt_inner < 0:
                    tgt_next = -tgt_next

            elif tangent_predictor == "nullspace_pinned":
                # Augment the Jacobian with phase condition and tangent
                # Approximate the null space of the Jacobian while constraining the continuation
                # parameter component to 1  (Peeters et al.)
                J_tgt, _ = self.add_phase_condition(J)
                J_tgt = np.vstack([J_tgt, np.zeros((1, J_tgt.shape[1]))])
                J_tgt[-1, -1] = 1

                Z = np.zeros((J_tgt.shape[0], 1))
                Z[-1] = 1
                if not forced:
                    tgt_next = spl.lstsq(
                        J_tgt, Z, cond=None, check_finite=False, lapack_driver="gelsd"
                    )[0][:, 0]
                elif forced:
                    tgt_next = spl.solve(J_tgt, Z, check_finite=False)[:, 0]
                tgt_next /= spl.norm(tgt_next)
                tgt_inner = np.dot(tgt_next, tgt)
                direction = np.sign(direction * tgt_inner)

            # calculate angle between tangents
            beta = np.rad2deg(np.arccos(np.clip(tgt_inner, -1.0, 1.0)))

            self.log.screenout(
                iter=itercont,
                correct=itercorrect,
                res=residual,
                freq=1 / get_period(),
                amp=get_amplitude(),
                energy=energy,
                step=direction * step_size,
                beta=beta,
            )  # need final screenout to print beta
            self.log.store(
                sol_X=X_pred,
                sol_T=get_period(),
                sol_F=get_amplitude(),
                sol_tgt=tgt_next,
                sol_energy=energy,
                sol_beta=beta,
                sol_itercorrect=itercorrect,
                sol_step=direction * step_size,
            )

            # Accept the corrected solution
            param_current = get_param_value()
            X = X_pred
            tgt = tgt_next
            itercont += 1

        # Adaptive step size for next point
        if itercont > parameters["continuation"]["adaptive_step_start"] or not converged_now:
            step_size = self.adapt_stepsize(step_size, itercorrect, converged_now)

        # Early exit if step size becomes too small
        if abs(step_size) < parameters["continuation"]["min_step_size"]:
            print("Step size too small. Terminating continuation.")
            break

        self.log.screenline("-")
