from core.problem import Problem
from core.startingpoint import StartingPoint
from core.logger import Logger
from core.solver.continuation import ConX
from duffing import Duffing

# Problem
prob = Problem()
prob.configure_parameters("parameters.yaml")
prob.set_zero_function(Duffing.periodicity)

# Update model based on parameters if system is forced
Duffing.update_model(prob.parameters)

# Starting point for continuation
start = StartingPoint(prob.parameters)
start.set_starting_function(Duffing.eigen)
start.get_starting_values()

# Logger to log and store solution
log = Logger(prob.parameters)

# Continuation
con = ConX(prob, start, log)

# Run continuation on problem
con.run()
