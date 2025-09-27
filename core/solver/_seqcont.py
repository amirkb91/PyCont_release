import numpy as np
import scipy.linalg as spl


def seqcont(self):
    parameters = self.prob.parameters
    parameters_cont = parameters["continuation"]
    forced = parameters_cont["forced"]
    dofdata = self.prob.doffunction()
    N = dofdata["ndof_free"]
    twoN = 2 * N

    # first point solution
    X = self.X0
    pose_base = self.pose
    pose_ref = self.pose_ref  # undeformed pose
    tau = self.T0
    amp = self.F0

    # Set up parameter continuation abstraction
    # fmt: off
    cont_parameter = parameters["continuation"]["continuation_parameter"]
    if cont_parameter == "frequency":
        param_current = tau
        def get_param_value(): return tau
        def set_param_value(val): nonlocal tau; tau = val
        def get_period(): return tau
        def get_amplitude(): return amp
        def get_cont_param_for_bounds(): return 1 / tau  # actual frequency
    elif cont_parameter == "amplitude":
        param_current = amp
        def get_param_value(): return amp
        def set_param_value(val): nonlocal amp; amp = val
        def get_period(): return tau
        def get_amplitude(): return amp
        def get_cont_param_for_bounds(): return amp
    # fmt: on

    # continuation parameters
    step = parameters_cont["s0"]
    direction = parameters_cont["dir"] * (-1 if cont_parameter == "frequency" else 1)

    # boolean mask to select inc from X (has no effect on single shooting)
    inc_mask = np.mod(np.arange(X.size), twoN) < N

    # --- MAIN CONTINUATION LOOP
    itercont = 1
    while True:
        # increment continuation parameter
        param_pred = param_current + step * direction
        X_pred = X.copy()
        set_param_value(param_pred)

        if (
            get_cont_param_for_bounds() > parameters_cont["ContParMax"]
            or get_cont_param_for_bounds() < parameters_cont["ContParMin"]
        ):
            print(
                f"Continuation Parameter {get_cont_param_for_bounds():.2e} outside of specified boundary."
            )
            break

        # correction step
        itercorrect = 0
        while True:

            [H, Jsim, pose, vel, energy, cvg_zerof] = self.prob.zerofunction(
                1.0, amp, tau, X_pred, pose_base, parameters
            )
            if not cvg_zerof:
                cvg_cont = False
                print(f"Zero function failed to converge with step = {step:.3e}.")
                break

            residual = spl.norm(H)
            # residual = normalise_residual(residual, pose_base, pose_ref, dofdata)

            J = np.block([[Jsim[:, :-1]], [self.h]])

            if residual < parameters_cont["tol"] and itercorrect >= parameters_cont["itermin"]:
                cvg_cont = True
                break
            elif itercorrect > parameters_cont["itermax"] or residual > 1e10:
                cvg_cont = False
                break

            self.log.screenout(
                iter=itercont,
                correct=itercorrect,
                res=residual,
                freq=1 / get_period(),
                amp=get_amplitude(),
                energy=energy,
                step=step,
            )

            # correction
            itercorrect += 1
            hx = self.h @ X_pred
            Z = np.vstack([H, hx.reshape(-1, 1)])
            if not forced:
                dx = spl.lstsq(J, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            elif forced:
                dx = spl.solve(J, -Z, check_finite=False)
            X_pred[:] += dx[:, 0]

        if cvg_cont:
            self.log.screenout(
                iter=itercont,
                correct=itercorrect,
                res=residual,
                freq=1 / get_period(),
                amp=get_amplitude(),
                energy=energy,
                step=step,
                beta=0.0,
            )
            self.log.store(
                sol_pose=pose,
                sol_vel=vel,
                sol_T=get_period(),
                sol_F=get_amplitude(),
                sol_energy=energy,
                sol_itercorrect=itercorrect,
                sol_step=step,
            )

            param_current = get_param_value()
            X = X_pred.copy()
            # update pose_base and set inc to zero, pose will have included inc from current sol
            pose_base = pose.copy()
            X[inc_mask] = 0.0
            itercont += 1

        # adaptive step size for next point
        if itercont > parameters_cont["nadapt"] or not cvg_cont:
            step = self.adapt_stepsize(self, step, itercorrect, cvg_cont)

        if itercont > parameters_cont["npts"]:
            print("Maximum number of continuation points reached.")
            break
        self.log.screenline("-")


# def normalise_residual(residual, pose_base, pose_ref, dofdata):
#     ndof_all = dofdata["ndof_all"]
#     n_nodes = dofdata["nnodes_all"]
#     config_per_node = dofdata["config_per_node"]
#     inc_from_ref = np.zeros((ndof_all))
#     pose_base = pose_base.flatten(order="F")
#     inc_from_ref = pose_base[: n_nodes * config_per_node] - pose_ref
#     return residual / spl.norm(inc_from_ref)
