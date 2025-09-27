from ._starting_correction import correct_starting_point
from ._seqcont import seqcont
from ._psacont import psacont
from ._phase_condition import _add_phase_condition


class ConX:
    def __init__(self, prob, start, log):
        self.prob = prob
        self.X0 = start.X0
        self.T0 = start.T0
        self.F0 = start.F0
        self.tgt0 = start.tgt0
        self.log = log
        self.num_phase_constraints = None

    def add_phase_condition(self, J):
        # Augment the Jacobian with phase condition constraints.
        return _add_phase_condition(self, J)

    def run(self):
        # correct starting solution
        correct_starting_point(self)

        # begin continuation
        if self.prob.parameters["continuation"]["method"] == "sequential":
            seqcont(self)
        elif self.prob.parameters["continuation"]["method"] == "pseudo_arclength":
            psacont(self)
