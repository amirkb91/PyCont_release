from core.problem import Prob
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint

from duffing import Duffing

# Problem
prob = Prob()
prob.configure_parameters("contparameters.json")
prob.add_doffunction(Duffing.get_fe_data)
prob.set_starting_function(Duffing.eigen_solve)

prob.set_zero_function(Duffing.time_solve)

# Initialise forcing parameters if continuation is forced
Duffing.forcing_parameters(prob.parameters)

# Continuation starting point
start = StartingPoint(prob)
start.compute_starting_values()

# Logger
log = Logger(prob)

# Solve continuation on problem
con = ConX(prob, start, log)
con.run()
