from core.problem import Prob
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint
from cubic_spring import Cubic_Spring

# Problem object
prob = Prob()
prob.configure_parameters("parameters.yaml")
prob.set_starting_function(Cubic_Spring.eigen_solve)
prob.set_zero_function(Cubic_Spring.time_solve)

# Starting point for continuation
start = StartingPoint(prob)
start.compute_starting_values()

# Logger to log and store solution
log = Logger(prob)

# Continuation
con = ConX(prob, start, log)

# Run continuation on problem
con.run()
