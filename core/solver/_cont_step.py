import numpy as np


def cont_step(self, step, itercorrect, cvg):
    if cvg:
        if itercorrect == 0 or itercorrect == self.prob.cont_params["continuation"]["iteropt"]:
            step *= np.sqrt(2)
        else:
            step *= self.prob.cont_params["continuation"]["iteropt"] / itercorrect
        # if itercorrect < self.prob.cont_params["continuation"]["iteropt"]:
        #     step *= np.cbrt(2)
        # elif itercorrect > self.prob.cont_params["continuation"]["iteropt"]:
        #     step /= np.sqrt(2)
        step = max(step, self.prob.cont_params["continuation"]["smin"])
        step = min(step, self.prob.cont_params["continuation"]["smax"])
    else:
        step /= 2
        if step < self.prob.cont_params["continuation"]["smin"]:
            raise Exception("Step size below smin, continuation cannot proceed.")
    return step
