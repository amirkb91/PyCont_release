from ._phase_condition import add_phase_condition
from ._starting_correction import correct_starting_point
from ._seqcont import seqcont
from ._psacont import psacont
from ._cont_step import adapt_stepsize


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
        """
        Augments the Jacobian with phase condition constraints.
        Implementation is imported from _phase_condition module.
        """
        return add_phase_condition(self, J)

    def correct_starting_point(self):
        """
        Corrects the starting point for continuation.
        Implementation is imported from _starting_correction module.
        """
        return correct_starting_point(self)

    def seqcont(self):
        """
        Performs sequential continuation.
        Implementation is imported from _seqcont module.
        """
        return seqcont(self)

    def psacont(self):
        """
        Performs pseudo-arclength continuation.
        Implementation is imported from _psacont module.
        """
        return psacont(self)

    def adapt_stepsize(self, *args, **kwargs):
        """
        Performs continuation step adaptation.
        Implementation is imported from _cont_step module.
        """
        return adapt_stepsize(self, *args, **kwargs)

    def run(self):
        # correct starting solution
        self.correct_starting_point()

        # perform continuation
        if self.prob.parameters["continuation"]["method"] == "sequential":
            self.seqcont()
        elif self.prob.parameters["continuation"]["method"] == "pseudo_arclength":
            self.psacont()
