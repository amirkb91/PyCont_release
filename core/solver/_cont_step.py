import numpy as np


def adapt_stepsize(self, step, itercorrect, cvg):
    if cvg:
        if itercorrect == 0 or itercorrect == self.prob.parameters["continuation"]["iteropt"]:
            step *= np.sqrt(2)
        else:
            step *= self.prob.parameters["continuation"]["iteropt"] / itercorrect
        step = max(step, self.prob.parameters["continuation"]["smin"])
        step = min(step, self.prob.parameters["continuation"]["smax"])
    else:
        step /= 2
        if step < self.prob.parameters["continuation"]["smin"]:
            raise Exception("Step size below smin, continuation cannot proceed.")
    return step
