import numpy as np
import scipy.linalg as spl
from collections import namedtuple
from ._cont_step import cont_step
from core.math.Frame import Frame

import warnings

warnings.filterwarnings("ignore", category=spl.LinAlgWarning)


def psacont(self):
    cont_params = self.prob.cont_params
    cont_params_cont = cont_params["continuation"]
    frml = cont_params_cont["tangent"].lower()
    forced = cont_params_cont["forced"]
    dofdata = self.prob.doffunction()
    N = dofdata["ndof_free"]
    twoN = 2 * N
    cvg_sol = namedtuple("converged_solution", "X, PAR, H, J")

    # first point converged solution
    X = self.X0
    pose_base = self.pose
    pose_ref = self.pose_ref  # undeformed pose
    tgt = self.tgt0
    omega = 1.0
    tau = self.T0
    amp = self.F0
    if cont_params["shooting"]["scaling"]:
        omega = 1 / self.T0
        tau = 1.0

    # Set up parameter continuation abstraction
    # fmt: off
    cont_parameter = cont_params["continuation"]["continuation_parameter"]
    if cont_parameter == "frequency":
        param_current = tau
        param_name = "frequency"
        def get_param_value(): return tau
        def set_param_value(val): nonlocal tau; tau = val
        def get_period(): return tau
        def get_amplitude(): return amp
        def get_cont_param_for_bounds(): return 1 / tau  # actual frequency
    elif cont_parameter == "amplitude":
        param_current = amp
        param_name = "amplitude"
        def get_param_value(): return amp
        def set_param_value(val): nonlocal amp; amp = val
        def get_period(): return tau
        def get_amplitude(): return amp
        def get_cont_param_for_bounds(): return amp
    # fmt: on

    # continuation parameters
    step = cont_params_cont["s0"]
    direction = (
        cont_params_cont["dir"] * np.sign(tgt[-1]) * (-1 if cont_parameter == "frequency" else 1)
    )

    # boolean masks to select inc and vel from X (has no effect on single shooting)
    inc_mask = np.mod(np.arange(X.size), twoN) < N
    vel_mask = ~inc_mask

    # --- MAIN CONTINUATION LOOP
    itercont = 1
    while True:
        # prediction step along tangent
        param_pred = param_current + tgt[-1] * step * direction
        X_pred = X + tgt[:-1] * step * direction
        set_param_value(param_pred)

        if (
            get_cont_param_for_bounds() > cont_params_cont["ContParMax"]
            or get_cont_param_for_bounds() < cont_params_cont["ContParMin"]
        ):
            print(
                f"Continuation Parameter {get_cont_param_for_bounds():.2e} outside of specified boundary."
            )
            break

        # correction step
        itercorrect = 0
        while True:
            if itercorrect % cont_params_cont["iterjac"] == 0:
                sensitivity = True
            else:
                sensitivity = False

            [H, Jsim, pose, vel, energy, cvg_zerof] = self.prob.zerofunction(
                omega, amp, tau, X_pred, pose_base, cont_params, sensitivity=sensitivity
            )
            if not cvg_zerof:
                cvg_cont = False
                print(f"Zero function failed to converge with step = {step:.3e}.")
                break

            residual = spl.norm(H)
            residual = normalise_residual(residual, pose_base, pose_ref, dofdata)

            if not sensitivity:
                # Broyden's Jacobian update
                deltaX = (
                    np.append(X_pred, get_param_value() / omega) - np.append(soldata.X, soldata.PAR)
                ).reshape(-1, 1)
                deltaf = H - soldata.H
                Jsim = soldata.J + 1 / spl.norm(deltaX) * (deltaf - soldata.J @ deltaX) @ deltaX.T

            J = np.block([[Jsim], [self.h, np.zeros((self.nphase, 1))], [tgt]])
            soldata = cvg_sol(X_pred.copy(), get_param_value() / omega, H.copy(), Jsim.copy())

            if residual < cont_params_cont["tol"] and itercorrect >= cont_params_cont["itermin"]:
                cvg_cont = True
                break
            elif itercorrect > cont_params_cont["itermax"] or residual > 1e10:
                cvg_cont = False
                break

            self.log.screenout(
                iter=itercont,
                correct=itercorrect,
                res=residual,
                freq=omega / get_period(),
                amp=get_amplitude(),
                energy=energy,
                step=direction * step,
            )

            # apply corrections orthogonal to tangent
            itercorrect += 1
            Jcr = J.copy()
            # Jcr[-1, twoN:-1] = 0.0  # ortho only 1st partition and T (no effect single shooting)
            hx = self.h @ X_pred
            Z = np.vstack([H, hx.reshape(-1, 1), np.zeros(1)])
            if not forced:
                dxt = spl.lstsq(Jcr, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            elif forced:
                dxt = spl.solve(Jcr, -Z, check_finite=False)
            param_new = get_param_value() + dxt[-1, 0]
            set_param_value(param_new)
            dx = dxt[:-1, 0]
            X_pred += dx

        if cvg_cont:
            # find new tangent with converged solution
            if frml == "secant":
                # As X[inc_mask] = 0 after convergence, X_pred[inc_mask] is already equal to
                # the difference between current and previous solutions
                tgt_next = np.concatenate(
                    (X_pred[inc_mask], (X_pred - X)[vel_mask], [get_param_value() - param_current])
                )
                tgt_next /= spl.norm(tgt_next)
                tgt_inner = np.dot(tgt_next, tgt)
                direction = np.sign(direction * tgt_inner)
            elif frml == "keller":
                # we already have J[-1, :] = tgt
                Z = np.zeros((J.shape[0], 1))
                Z[-1] = 1.0
                if not forced:
                    tgt_next = spl.lstsq(
                        J, Z, cond=None, check_finite=False, lapack_driver="gelsd"
                    )[0][:, 0]
                elif forced:
                    tgt_next = spl.solve(J, Z, check_finite=False)[:, 0]
                tgt_next /= spl.norm(tgt_next)
                tgt_inner = np.dot(tgt_next, tgt)
                # safeguard: if numerical error yields a negative dot product, flip the tangent
                if tgt_inner < 0:
                    tgt_next = -tgt_next
            elif frml == "peeters":
                J[-1, :] = 0.0
                J[-1, -1] = 1.0
                Z = np.zeros((J.shape[0], 1))
                Z[-1] = 1.0
                if not forced:
                    tgt_next = spl.lstsq(
                        J, Z, cond=None, check_finite=False, lapack_driver="gelsd"
                    )[0][:, 0]
                elif forced:
                    tgt_next = spl.solve(J, Z, check_finite=False)[:, 0]
                tgt_next /= spl.norm(tgt_next)
                tgt_inner = np.dot(tgt_next, tgt)
                direction = np.sign(direction * tgt_inner)

            # calculate beta and check against betamax if requested, fail convergence if check fails
            beta = np.rad2deg(np.arccos(np.clip(tgt_inner, -1.0, 1.0)))
            # beta below found using tangent of first partition + T only
            # beta = np.rad2deg(
            #     np.arccos(np.dot(tgt_next[np.r_[0:twoN, -1]], tgt[np.r_[0:twoN, -1]]))
            # )

            if cont_params_cont["betacontrol"] and beta > cont_params_cont["betamax"]:
                print("Beta exceeds maximum angle.")
                cvg_cont = False

            else:
                self.log.screenout(
                    iter=itercont,
                    correct=itercorrect,
                    res=residual,
                    freq=omega / get_period(),
                    amp=get_amplitude(),
                    energy=energy,
                    step=direction * step,
                    beta=beta,
                )
                self.log.store(
                    sol_pose=pose,
                    sol_vel=vel,
                    sol_T=get_period() / omega,
                    sol_amp=get_amplitude(),
                    sol_tgt=tgt_next,
                    sol_energy=energy,
                    sol_beta=beta,
                    sol_itercorrect=itercorrect,
                    sol_step=direction * step,
                )

                param_current = get_param_value()
                X = X_pred.copy()
                tgt = tgt_next.copy()
                # update pose_base and set inc to zero, pose will have included inc from current sol
                pose_base = pose.copy()
                X[inc_mask] = 0.0
                itercont += 1

                # if cont_params["shooting"]["scaling"]:
                #     # reset tau to 1.0
                #     omega = omega / tau
                #     tau = 1.0

        # adaptive step size for next point
        if itercont > cont_params_cont["nadapt"] or not cvg_cont:
            step = cont_step(self, step, itercorrect, cvg_cont)

        if itercont > cont_params_cont["npts"]:
            print("Maximum number of continuation points reached.")
            break
        if cvg_cont and energy and energy > cont_params_cont["Emax"]:
            print(f"Energy {energy:.5e} exceeds Emax.")
            break
        self.log.screenline("-")


def normalise_residual(residual, pose_base, pose_ref, dofdata):
    ndof_all = dofdata["ndof_all"]
    n_nodes = dofdata["nnodes_all"]
    config_per_node = dofdata["config_per_node"]
    dof_per_node = dofdata["dof_per_node"]
    n_dim = dofdata["n_dim"]
    SEbeam = dofdata["SEbeam"]
    inc_from_ref = np.zeros((ndof_all))
    # in multiple shooting, effectively takes pose_base of first partition only
    pose_base = pose_base.flatten(order="F")

    if SEbeam:
        for k in range(n_nodes):
            f = Frame.relative_frame(
                n_dim,
                pose_ref[k * config_per_node : (k + 1) * config_per_node],
                pose_base[k * config_per_node : (k + 1) * config_per_node],
            )
            inc_from_ref[k * dof_per_node : (k + 1) * dof_per_node] = (
                Frame.get_parameters_from_frame(n_dim, f)
            )
    else:
        inc_from_ref = pose_base[: n_nodes * config_per_node] - pose_ref
    return residual / spl.norm(inc_from_ref)


# def qrlinearsolver(A, b):
#     Q, R, P = spl.qr(A, pivoting=True, mode="economic")
#     Qt_b = np.dot(Q.T, b)
#     x_temp = spl.solve_triangular(R, Qt_b[:R.shape[0]])
#     x = np.zeros_like(x_temp)
#     x[P] = x_temp
#     return x
