from core.problem import Problem
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint

from duffing import Duffing

# Problem
prob = Problem()
prob.configure_parameters("contparameters.json")
prob.add_doffunction(Duffing.get_fe_data)
prob.set_starting_function(Duffing.eigen_solve)

prob.set_zero_function(Duffing.time_solve)

# Initialise forcing parameters if continuation is forced
Duffing.forcing_parameters(prob.parameters)

# Continuation starting point
start = StartingPoint(prob)
start.starting_values_from_function()

# Logger
log = Logger(prob)

# Solve continuation on problem
con = ConX(prob, start, log)
con.run()
