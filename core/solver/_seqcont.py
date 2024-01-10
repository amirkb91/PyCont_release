import numpy as np
import scipy.linalg as spl
from ._cont_step import cont_step


def seqcont(self):
    forced = self.prob.cont_params["continuation"]["forced"]
    dofdata = self.prob.doffunction()
    N = dofdata["ndof_free"]
    twoN = 2 * N

    # first point solution
    X = self.X0.copy()
    pose_base = self.pose.copy()
    omega = self.omega
    tau = self.tau

    # continuation parameters
    step = self.prob.cont_params["continuation"]["s0"]
    direction = self.prob.cont_params["continuation"]["dir"]
    stepsign = -1 * direction  # corrections are always added

    # continuation loop
    itercont = 1
    while True:
        # increment period
        tau_pred = tau + step * stepsign
        X_pred = X.copy()

        if (
            omega / tau_pred > self.prob.cont_params["continuation"]["fmax"]
            or omega / tau_pred < self.prob.cont_params["continuation"]["fmin"]
        ):
            print(f"Frequency {omega / tau_pred:.2e} Hz outside of specified boundary.")
            break

        # correction step
        itercorrect = 0
        while True:
            if itercorrect % self.prob.cont_params["continuation"]["iterjac"] == 0:
                sensitivity = True
            else:
                sensitivity = False

            [H, Jsim, pose_time, vel_time, energy, cvg_zerof] = self.prob.zerofunction(
                omega, tau_pred, X_pred, pose_base, self.prob.cont_params, sensitivity=sensitivity
            )
            if not cvg_zerof:
                cvg_cont = False
                print(f"Zero function failed to converge with step = {step:.3e}.")
                break

            residual = spl.norm(H)

            if sensitivity:
                J = np.block([[Jsim[:, :-1]], [self.h]])

            if (
                residual < self.prob.cont_params["continuation"]["tol"]
                and itercorrect >= self.prob.cont_params["continuation"]["itermin"]
            ):
                cvg_cont = True
                break
            elif itercorrect > self.prob.cont_params["continuation"]["itermax"] or residual > 1e10:
                cvg_cont = False
                break

            self.log.screenout(
                iter=itercont,
                correct=itercorrect,
                res=residual,
                freq=omega / tau_pred,
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
                freq=omega / tau_pred,
                energy=energy,
                step=step,
                beta=0.0,
            )
            self.log.store(
                sol_pose=pose_time[:, 0],
                sol_vel=vel_time[:, 0],
                sol_T=tau_pred / omega,
                sol_energy=energy,
                sol_itercorrect=itercorrect,
                sol_step=step,
            )

            itercont += 1
            tau = tau_pred
            X = X_pred.copy()
            # update pose_base and set inc to zero (slice 0:N on each partition)
            # pose_time[:, 0] will have included inc from current sol
            pose_base = pose_time[:, 0].copy()
            X[np.mod(np.arange(X.size), twoN) < N] = 0.0

            # if self.prob.cont_params["shooting"]["scaling"]:
            #     # reset tau to 1.0
            #     omega = omega / tau
            #     tau = 1.0

        # adaptive step size for next point
        if itercont > self.prob.cont_params["continuation"]["nadapt"] or not cvg_cont:
            step = cont_step(self, step, itercorrect, cvg_cont)

        if itercont > self.prob.cont_params["continuation"]["npts"]:
            print("Maximum number of continuation points reached.")
            break
        if cvg_cont and energy and energy > self.prob.cont_params["continuation"]["Emax"]:
            print(f"Energy {energy:.5e} exceeds Emax.")
            break
