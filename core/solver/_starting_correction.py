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

            residual = spl.norm(H) / max(spl.norm(self.X0), 1e-12)

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
            # Augment the Jacobian with phase condition
            J_corr, h = self.add_phase_condition(J)

            # Make corrections orthogonal to X0 to avoid X0 being driven to zero (which
            # still qualifies as a periodic solution.)
            J_corr = np.vstack([J_corr, np.append(self.X0, 0)])

            # correct X0 and T0
            Z = np.concatenate([H, h @ self.X0, np.zeros(1)])
            dxt = spl.lstsq(J_corr, -Z, cond=None, check_finite=False, lapack_driver="gelsd")[0]
            self.X0 += dxt[:-1]
            self.T0 += dxt[-1]
            itercorrect += 1

        # Compute tangent vector
        # This is done by solving for the nullspace of the Jacobian, while constraining the period
        # component to 1 (Peeters et al.)
        J_tgt, _ = self.add_phase_condition(J)  # Ignore h matrix for now
        J_tgt = np.vstack([J_tgt, np.zeros((1, J_tgt.shape[1]))])
        J_tgt[-1, -1] = 1
        Z = np.zeros((J_tgt.shape[0], 1))
        Z[-1] = 1
        self.tgt0 = spl.lstsq(J_tgt, Z, cond=None, check_finite=False, lapack_driver="gelsd")[0][
            :, 0
        ]
        self.tgt0 /= spl.norm(self.tgt0)

    elif func_start and forced:
        itercorrect = 0
        while True:
            H, J, energy = self.prob.zero_function(self.F0, self.T0, self.X0, parameters)

            residual = spl.norm(H) / max(spl.norm(self.X0), 1e-12)

            self.log.screenout(
                iter=0,
                correct=itercorrect,
                res=residual,
                freq=1 / self.T0,
                amp=self.F0,
                energy=energy,
            )

            if residual < parameters["continuation"]["corrections_tolerance"] and itercorrect > 0:
                break

            # Compute corrections
            # Don't need phase condition augmentation for forced system
            # Only correct X0 and keep F and T fixed
            dx = spl.solve(J[:, :-1], -H)
            self.X0 += dx
            itercorrect += 1

        # Compute tangent vector
        # This is done by solving for the nullspace of the Jacobian, while constraining the period
        # component to 1 (Peeters et al.)
        J = np.vstack([J, np.zeros((1, J.shape[1]))])
        J[-1, -1] = 1
        Z = np.zeros((J.shape[0], 1))
        Z[-1] = 1
        self.tgt0 = spl.solve(J, Z)[:, 0]
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
