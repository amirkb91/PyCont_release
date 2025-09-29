import numpy as np
import scipy.linalg as spl


def seqcont(self):
    """
    Sequential continuation method.

    This method performs continuation by correcting only the solution variables X,
    while keeping the continuation parameter (T or F) at predicted values.
    """
    # Starting point solution
    X = self.X0
    T = self.T0
    F = self.F0

    # Read parameters
    parameters = self.prob.parameters
    continuation_parameter = parameters["continuation"]["parameter"]
    step_size = parameters["continuation"]["initial_step_size"]

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
        # Predict next continuation parameter value
        param_pred = param_current + step_size * direction
        set_param_value(param_pred)

        # Check bounds before doing corrections
        if not (min_param <= get_cont_param_for_bounds() <= max_param):
            print(
                f"Continuation parameter {get_cont_param_for_bounds():.2e} outside specified bounds [{min_param:.2e}, {max_param:.2e}]."
            )
            break

        # Correction iterations
        X_corrected = X.copy()
        converged, itercorrect, residual, energy = _perform_corrections(
            self,
            X_corrected,
            get_period(),
            get_amplitude(),
            parameters,
            itercont,
            step_size,
        )

        # Handle convergence results
        if converged:
            self.log.store(
                sol_X=X_corrected,
                sol_T=get_period(),
                sol_F=get_amplitude(),
                sol_energy=energy,
                sol_itercorrect=itercorrect,
                sol_step=step_size,
            )

            # Accept the corrected solution
            param_current = get_param_value()
            X = X_corrected
            itercont += 1
        else:
            print(f"Correction failed at iteration {itercont}. Residual: {residual:.2e}")

        # Adaptive step size for next point
        if itercont > parameters["continuation"]["adaptive_step_start"] or not converged:
            step_size = self.adapt_stepsize(step_size, itercorrect, converged)

        # Early exit if step size becomes too small
        if abs(step_size) < parameters["continuation"]["min_step_size"]:
            print("Step size too small. Terminating continuation.")
            break

        self.log.screenline("-")


def _perform_corrections(self, X_corrected, period, amplitude, parameters, itercont, step_size):
    """
    Perform Newton-Raphson corrections for sequential continuation.

    Args:
        X_corrected: Solution vector to correct (modified in-place)
        period: Current period value
        amplitude: Current amplitude value
        parameters: Problem parameters
        itercont: Current continuation iteration number
        step_size: Current step size

    Returns:
        tuple: (converged, itercorrect, residual, energy)
    """
    itercorrect = 0
    max_iterations = parameters["continuation"]["max_iterations"]
    min_iterations = parameters["continuation"]["min_iterations"]
    tolerance = parameters["continuation"]["corrections_tolerance"]
    forced = "force" in parameters["continuation"]["parameter"]

    while True:
        # Evaluate zero function and Jacobian
        try:
            H, J, energy = self.prob.zero_function(amplitude, period, X_corrected, parameters)
        except Exception as e:
            print(f"Error evaluating zero function: {e}")
            return False, itercorrect, 1e10, 0.0

        residual = spl.norm(H) / max(spl.norm(X_corrected), 1e-12)

        self.log.screenout(
            iter=itercont,
            correct=itercorrect,
            res=residual,
            freq=1 / period,
            amp=amplitude,
            energy=energy,
            step=step_size,
        )

        # Check convergence criteria
        converged_now = residual < tolerance and itercorrect >= min_iterations

        # Return if converged
        if converged_now:
            return True, itercorrect, residual, energy

        # Check divergence criteria
        if residual > 1e10:
            return False, itercorrect, residual, energy

        # Maximum iterations reached
        if itercorrect >= max_iterations:
            return False, itercorrect, residual, energy

        # Compute correction step
        try:
            # Augment Jacobian with phase condition (remove last column as parameter not corrected)
            J_corr = self.add_phase_condition(J[:, :-1])

            Z = np.concatenate([H, np.zeros(self.num_phase_constraints)])
            if not forced:
                dx = spl.lstsq(J_corr, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            else:
                dx = spl.solve(J_corr, -Z, check_finite=False)

            # Apply correction
            X_corrected += dx
            itercorrect += 1

        except (spl.LinAlgError, np.linalg.LinAlgError) as e:
            print(f"Linear algebra error during correction: {e}")
            return False, itercorrect, 1e10, energy
