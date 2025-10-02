import numpy as np


def adapt_stepsize(self, step_size, itercorrect, converged):
    if converged:
        if itercorrect == 0:
            step_size *= np.sqrt(2)
        else:
            step_size *= self.prob.parameters["continuation"]["optimal_iterations"] / itercorrect
        # keep step size within specified bounds
        step_size = max(step_size, self.prob.parameters["continuation"]["min_step_size"])
        step_size = min(step_size, self.prob.parameters["continuation"]["max_step_size"])
    else:
        step_size /= 2
        if step_size < self.prob.parameters["continuation"]["min_step_size"]:
            raise Exception("Step size falls below minimum allowable step size.")
    return step_size
