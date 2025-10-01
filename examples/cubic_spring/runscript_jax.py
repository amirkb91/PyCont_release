from core.problem import Problem
from core.startingpoint import StartingPoint
from core.logger import Logger
from core.solver.continuation import ConX
from cubic_spring_jax import Cubic_Spring_jax

# Problem
prob = Problem()
prob.configure_parameters("parameters.yaml")
prob.set_zero_function(Cubic_Spring_jax.periodicity)

# Update model based on parameters if system is forced
Cubic_Spring_jax.update_model(prob.parameters)

# Starting point for continuation
start = StartingPoint(prob.parameters)
start.set_starting_function(Cubic_Spring_jax.eigen)
start.get_starting_values()

# Logger to log and store solution
log = Logger(prob.parameters)

# Continuation
con = ConX(prob, start, log)

# Run continuation on problem
con.run()
