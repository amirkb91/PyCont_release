import numpy as np
import scipy.linalg as spl
from ._phase_condition import phase_condition


def first_point(self):
    cont_params = self.prob.cont_params
    eig_start = cont_params["first_point"]["from_eig"]
    restart = not eig_start
    forced = cont_params["continuation"]["forced"]
    shooting_method = cont_params["shooting"]["method"]
    dofdata = self.prob.doffunction()
    N = dofdata["ndof_free"]
    iter_firstpoint = 0

    if eig_start and not forced:
        linearsol = self.X0.copy()
        while True:
            if iter_firstpoint > cont_params["first_point"]["itermax"]:
                raise Exception("Max number of iterations reached without convergence.")

            [H, J, self.pose, vel, energy, cvg_zerof] = self.prob.zerofunction_firstpoint(
                1.0, self.F0, self.T0, self.X0, self.pose0, cont_params
            )
            if not cvg_zerof:
                raise Exception("Zero function failed.")

            residual = spl.norm(H)
            # orthogonality to linear solution to avoid trivial 0 solution
            J = np.block([[J], [self.h, np.zeros((self.nphase, 1))], [linearsol, np.zeros(1)]])

            self.log.screenout(
                iter=0,
                correct=iter_firstpoint,
                res=residual,
                freq=1 / self.T0,
                energy=energy,
            )

            if residual < cont_params["continuation"]["tol"] and iter_firstpoint > 0:
                break

            # correct X0 and T0
            iter_firstpoint += 1
            hx = self.h @ self.X0
            Z = np.vstack([H, hx.reshape(-1, 1), np.zeros(1)])
            dxt = spl.lstsq(J, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            self.T0 += dxt[-1, 0]
            dx = dxt[:-1, 0]
            self.X0 += dx

        # set inc to zero as pose will have included inc
        self.X0[:N] = 0.0

        # Compute Tangent
        if shooting_method == "single":
            J[-1, :] = np.zeros(np.shape(J)[1])
        elif shooting_method == "multiple":
            # partition solution
            self.X0, self.pose = self.prob.partitionfunction(
                self.T0, self.X0, self.pose, cont_params
            )
            # size of X0 has changed so reconfigure phase condition matrix
            phase_condition(self)
            # override Jacobian with new Jacobian for all partitions, zerofunction is multiple shooting
            [_, J, _, vel, _, _] = self.prob.zerofunction(
                1.0, self.T0, self.X0, self.pose, cont_params
            )
            J = np.block([[J], [self.h, np.zeros((self.nphase, 1))], [np.zeros(np.shape(J)[1])]])

        J[-1, -1] = 1
        Z = np.zeros((np.shape(J)[0], 1))
        Z[-1] = 1
        self.tgt0 = spl.lstsq(J, Z, cond=None, check_finite=False, lapack_driver="gelsd")[0][:, 0]
        self.tgt0 /= spl.norm(self.tgt0)

    elif restart and not forced:
        if shooting_method == "single":
            # run sim to get data for storing solution
            [H, J, self.pose, vel, energy, _] = self.prob.zerofunction_firstpoint(
                1.0, self.F0, self.T0, self.X0, self.pose0, cont_params
            )
            residual = spl.norm(H)
            J = np.block([[J], [self.h, np.zeros((self.nphase, 1))], [np.zeros(np.shape(J)[1])]])

        elif shooting_method == "multiple":
            if self.pose0.ndim == 1:
                # if we are restarting from a single shooting solution, we need to partition it
                self.X0, self.pose0 = self.prob.partitionfunction(
                    self.T0, self.X0, self.pose0, cont_params
                )
                # we also have to recompute tangent regardless of user input
                cont_params["first_point"]["restart"]["recompute_tangent"] = True

            # phase condition matrix has to be reconfigured for multiple shooting
            phase_condition(self)

            # run sim to get data for storing solution, zerofunction is multiple shooting
            [H, J, self.pose, vel, energy, _] = self.prob.zerofunction(
                1.0, self.T0, self.X0, self.pose0, cont_params
            )
            residual = spl.norm(H)
            J = np.block([[J], [self.h, np.zeros((self.nphase, 1))], [np.zeros(np.shape(J)[1])]])

        if cont_params["first_point"]["restart"]["recompute_tangent"]:
            J[-1, -1] = 1
            Z = np.zeros((np.shape(J)[0], 1))
            Z[-1] = 1
            self.tgt0 = spl.lstsq(J, Z, cond=None, check_finite=False, lapack_driver="gelsd")[0][:, 0]  # fmt: skip
            self.tgt0 /= spl.norm(self.tgt0)

        self.log.screenout(iter=0, correct=0, res=residual, freq=1 / self.T0, energy=energy)

    elif forced:
        while True:
            if iter_firstpoint > cont_params["first_point"]["itermax"]:
                raise Exception("Max number of iterations reached without convergence.")

            [H, J, self.pose, vel, energy, cvg_zerof] = self.prob.zerofunction_firstpoint(
                1.0, self.F0, self.T0, self.X0, self.pose0, cont_params
            )
            if not cvg_zerof:
                raise Exception("Zero function failed.")

            residual = spl.norm(H)  # Note this is different to psacont residual not normalised

            self.log.screenout(
                iter=0,
                correct=iter_firstpoint,
                res=residual,
                freq=1 / self.T0,
                amp=self.F0,
                energy=energy,
            )

            if residual < cont_params["continuation"]["tol"] or restart:
                break

            # correct only X0
            iter_firstpoint += 1
            Z = H
            dx = spl.solve(J[:, :-1], -Z)
            self.X0 += dx[:, 0]

        # set inc to zero as pose will have included inc
        self.X0[:N] = 0.0

        # Compute Tangent
        J = np.block([[J], [np.zeros(np.shape(J)[1])]])
        J[-1, -1] = 1
        Z = np.zeros((np.shape(J)[0], 1))
        Z[-1] = 1
        self.tgt0 = spl.solve(J, Z)[:, 0]
        self.tgt0 /= spl.norm(self.tgt0)

    self.log.store(
        sol_pose=self.pose,
        sol_vel=vel,
        sol_T=self.T0,
        sol_amp=self.F0,
        sol_tgt=self.tgt0,
        sol_energy=energy,
        sol_itercorrect=iter_firstpoint,
        sol_step=0,
    )
    self.log.screenline("-")
