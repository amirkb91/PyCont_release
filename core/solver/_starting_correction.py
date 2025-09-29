import numpy as np
import scipy.linalg as spl


def correct_starting_point(self):
    parameters = self.prob.parameters
    func_start = parameters["starting_point"]["source"] == "function"
    file_start = not func_start
    forced = "force" in parameters["continuation"]["parameter"]

    if func_start and not forced:
        itercorrect = 0
        while True:
            H, J, energy = self.prob.zero_function(self.F0, self.T0, self.X0, parameters)

            residual = spl.norm(H) / spl.norm(self.X0)

            self.log.screenout(
                iter=0,
                correct=itercorrect,
                res=residual,
                freq=1 / self.T0,
                energy=energy,
            )

            if residual < parameters["continuation"]["corrections_tolerance"] and itercorrect > 0:
                break

            # Compute corrections
            # augment the Jacobian with the phase condition
            J_corr = self.add_phase_condition(J)

            # Make corrections orthogonal to X0 to avoid X0 being driven to zero (which
            # still qualifies as a periodic solution.)
            J_corr = np.vstack([J_corr, np.append(self.X0, 0)])

            # correct X0 and T0
            itercorrect += 1
            Z = np.concatenate([H, np.zeros(self.num_phase_constraints), np.zeros(1)])
            dxt = spl.lstsq(J_corr, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            self.X0 += dxt[:-1]
            self.T0 += dxt[-1]

        # Compute tangent vector
        # This is done by solving for the nullspace of the Jacobian, while constraining the period
        # component to 1 (ref. Peeters et al.)
        J_tgt = self.add_phase_condition(J)
        J_tgt = np.vstack([J_tgt, np.zeros((1, J_tgt.shape[1]))])
        J_tgt[-1, -1] = 1
        Z = np.zeros((J_tgt.shape[0], 1))
        Z[-1] = 1
        self.tgt0 = spl.lstsq(J_tgt, Z, cond=None, check_finite=False, lapack_driver="gelsd")[0][
            :, 0
        ]
        self.tgt0 /= spl.norm(self.tgt0)

    elif file_start and not forced:
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
            if itercorrect > parameters["first_point"]["itermax"]:
                raise Exception("Max number of iterations reached without convergence.")

            [H, J, self.pose, vel, energy, cvg_zerof] = self.prob.zerofunction_firstpoint(
                1.0, self.F0, self.T0, self.X0, self.pose0, parameters
            )
            if not cvg_zerof:
                raise Exception("Zero function failed.")

            residual = spl.norm(H)  # Note this is different to psacont residual not normalised

            self.log.screenout(
                iter=0,
                correct=itercorrect,
                res=residual,
                freq=1 / self.T0,
                amp=self.F0,
                energy=energy,
            )

            if residual < parameters["continuation"]["tol"] or file_start:
                break

            # correct only X0
            itercorrect += 1
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
        sol_X=self.X0,
        sol_T=self.T0,
        sol_F=self.F0,
        sol_tgt=self.tgt0,
        sol_energy=energy,
        sol_itercorrect=itercorrect,
        sol_step=0,
    )
    self.log.screenline("-")
