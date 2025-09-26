from ._first_point import first_point
from ._seqcont import seqcont
from ._psacont import psacont


class ConX:
    def __init__(self, prob, start, log):
        self.prob = prob
        self.X0 = start.X0
        self.T0 = start.T0
        self.F0 = start.F0
        self.tgt0 = start.tgt0
        self.log = log

    def run(self):
        # correct starting solution
        first_point(self)

        # begin continuation
        if self.prob.parameters["continuation"]["method"] == "sequential":
            seqcont(self)
        elif self.prob.parameters["continuation"]["method"] == "pseudo_arclength":
            psacont(self)
