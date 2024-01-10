import numpy as np
import scipy.linalg as spl
from collections import namedtuple
from ._cont_step import cont_step

# import warnings
# warnings.filterwarnings("ignore", category=spl.LinAlgWarning)


def psacont(self):
    frml = self.prob.cont_params["continuation"]["tangent"].lower()
    forced = self.prob.cont_params["continuation"]["forced"]
    dofdata = self.prob.doffunction()
    N = dofdata["ndof_free"]
    twoN = 2 * N
    cvg_sol = namedtuple("converged_solution", "X, T, H, J")

    # first point solution
    X = self.X0.copy()
    pose_base = self.pose.copy()
    tgt = self.tgt0.copy()
    omega = self.omega
    tau = self.tau

    # continuation parameters
    step = self.prob.cont_params["continuation"]["s0"]
    direction = self.prob.cont_params["continuation"]["dir"]
    stepsign = -1 * direction * np.sign(tgt[-1])  # corrections are always added

    # continuation loop
    itercont = 1
    while True:
        # prediction step along tangent
        tau_pred = tau + tgt[-1] * step * stepsign
        X_pred = X + tgt[:-1] * step * stepsign

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
                J = np.block([[Jsim], [self.h, np.zeros((self.nphase, 1))], [tgt]])
                soldata = cvg_sol(X_pred.copy(), tau_pred / omega, H.copy(), Jsim.copy())
            else:
                # Broyden's Jacobian update
                deltaX = (
                    np.append(X_pred, tau_pred / omega) - np.append(soldata.X, soldata.T)
                ).reshape(-1, 1)
                deltaf = H - soldata.H
                Jsim = soldata.J + 1 / spl.norm(deltaX) * (deltaf - soldata.J @ deltaX) @ deltaX.T
                J = np.block([[Jsim], [self.h, np.zeros((self.nphase, 1))], [tgt]])
                soldata = cvg_sol(X_pred.copy(), tau_pred / omega, H.copy(), Jsim.copy())

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
                step=stepsign * step,
            )

            # apply corrections orthogonal to tangent
            itercorrect += 1
            Jcr = J.copy()
            # orthogonality only on first partition and period: has no effect on single shooting
            Jcr[-1, twoN:-1] = 0.0
            hx = self.h @ X_pred
            Z = np.vstack([H, hx.reshape(-1, 1), np.zeros(1)])
            if not forced:
                dxt = spl.lstsq(Jcr, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            elif forced:
                dxt = spl.solve(Jcr, -Z, check_finite=False)
            tau_pred += dxt[-1, 0]
            dx = dxt[:-1, 0]
            X_pred += dx

        if cvg_cont:
            # find new tangent with converged solution
            if frml == "secant":
                # X_pred[:N]==INC and technically is already the diff between POSE for previous two sols
                tgt_next = np.concatenate((X_pred[:N], X_pred[N:] - X[N:], [tau_pred - tau]))
                # tgt_next = np.concatenate((X_pred - X, [tau_pred - tau]))
            else:
                if frml == "peeters":
                    # remove tgt from Jacobian and fix period component to 1
                    J[-1, :] = np.zeros(np.shape(J)[1])
                    J[-1, -1] = 1
                Z = np.zeros((np.shape(J)[0], 1))
                Z[-1] = 1
                if not forced:
                    tgt_next = spl.lstsq(
                        J, Z, cond=None, check_finite=False, lapack_driver="gelsd"
                    )[0][:, 0]
                elif forced:
                    tgt_next = spl.solve(J, Z, check_finite=False)[:, 0]
            tgt_next /= spl.norm(tgt_next)

            # calculate beta and check against betamax if requested, fail convergence if check fails
            # performed using tangent of first partition + T only (mask has no effect on single shooting)
            mask = np.ones(np.shape(tgt), dtype=bool)
            mask[twoN:-1] = False
            beta = np.degrees(
                np.arccos(
                    (tgt_next[mask].T @ tgt[mask])
                    / (spl.norm(tgt[mask]) * spl.norm(tgt_next[mask]))
                )
            )
            if (
                self.prob.cont_params["continuation"]["betacontrol"]
                and beta > self.prob.cont_params["continuation"]["betamax"]
            ):
                print("Beta exceeds maximum angle.")
                cvg_cont = False
            else:
                # passed check, store and update for next step
                self.log.screenout(
                    iter=itercont,
                    correct=itercorrect,
                    res=residual,
                    freq=omega / tau_pred,
                    energy=energy,
                    step=stepsign * step,
                    beta=beta,
                )
                self.log.store(
                    sol_pose=pose_time[:, 0],
                    sol_vel=vel_time[:, 0],
                    sol_T=tau_pred / omega,
                    sol_tgt=tgt_next,
                    sol_energy=energy,
                    sol_beta=beta,
                    sol_itercorrect=itercorrect,
                    sol_step=stepsign * step,
                )

                itercont += 1
                if frml in ("peeters", "secant"):  # and beta >= 90:
                    # stepsign = np.sign(stepsign * tgt_next[mask].T @ tgt[mask])
                    stepsign = np.sign(stepsign * np.cos(np.radians(beta)))
                tau = tau_pred
                X = X_pred.copy()
                tgt = tgt_next.copy()
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
