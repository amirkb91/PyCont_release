import numpy as np
import scipy.linalg as spl
from ._phase_condition import phase_condition


def first_point(self):
    parameters = self.prob.parameters
    eig_start = parameters["first_point"]["from_eig"]
    restart = not eig_start
    forced = parameters["continuation"]["forced"]
    dofdata = self.prob.doffunction()
    N = dofdata["ndof_free"]
    iter_firstpoint = 0

    if eig_start and not forced:
        linearsol = self.X0.copy()
        while True:
            if iter_firstpoint > parameters["first_point"]["itermax"]:
                raise Exception("Max number of iterations reached without convergence.")

            [H, J, self.pose, vel, energy, cvg_zerof] = self.prob.zerofunction_firstpoint(
                1.0, self.F0, self.T0, self.X0, self.pose0, parameters
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

            if residual < parameters["continuation"]["tol"] and iter_firstpoint > 0:
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
        J[-1, :] = np.zeros(np.shape(J)[1])
        J[-1, -1] = 1
        Z = np.zeros((np.shape(J)[0], 1))
        Z[-1] = 1
        self.tgt0 = spl.lstsq(J, Z, cond=None, check_finite=False, lapack_driver="gelsd")[0][:, 0]
        self.tgt0 /= spl.norm(self.tgt0)

    elif restart and not forced:
        # run sim to get data for storing solution
        [H, J, self.pose, vel, energy, _] = self.prob.zerofunction_firstpoint(
            1.0, self.F0, self.T0, self.X0, self.pose0, parameters
        )
        residual = spl.norm(H)
        J = np.block([[J], [self.h, np.zeros((self.nphase, 1))], [np.zeros(np.shape(J)[1])]])

        if parameters["first_point"]["restart"]["recompute_tangent"]:
            J[-1, -1] = 1
            Z = np.zeros((np.shape(J)[0], 1))
            Z[-1] = 1
            self.tgt0 = spl.lstsq(J, Z, cond=None, check_finite=False, lapack_driver="gelsd")[0][:, 0]  # fmt: skip
            self.tgt0 /= spl.norm(self.tgt0)

        self.log.screenout(iter=0, correct=0, res=residual, freq=1 / self.T0, energy=energy)

    elif forced:
        while True:
            if iter_firstpoint > parameters["first_point"]["itermax"]:
                raise Exception("Max number of iterations reached without convergence.")

            [H, J, self.pose, vel, energy, cvg_zerof] = self.prob.zerofunction_firstpoint(
                1.0, self.F0, self.T0, self.X0, self.pose0, parameters
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

            if residual < parameters["continuation"]["tol"] or restart:
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
