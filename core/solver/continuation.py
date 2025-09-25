from ._phase_condition import phase_condition
from ._first_point import first_point
from ._seqcont import seqcont
from ._psacont import psacont


class ConX:
    def __init__(self, prob, start, log):
        self.h = None
        self.nphase = None
        self.prob = prob
        self.X0 = start.X0
        self.T0 = start.T0
        self.F0 = start.F0
        self.pose0 = start.pose0
        self.pose_ref = start.pose_ref
        self.tgt0 = start.tgt0
        self.pose = None
        self.log = log

    def run(self):
        # calculate phase condition matrix h
        phase_condition(self)
        # correct starting solution
        first_point(self)
        if self.prob.parameters["continuation"]["method"] == "seq":
            # sequential continuation
            seqcont(self)
        elif self.prob.parameters["continuation"]["method"] == "psa":
            psacont(self)
