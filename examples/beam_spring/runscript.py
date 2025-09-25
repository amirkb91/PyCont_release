import sys
from core.problem import Problem
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint

from beam_spring import Beam_Spring

# Check command line arguments
if len(sys.argv) != 2:
    config_file = "contparameters.json"
else:
    config_file = sys.argv[1]

# Problem
prob = Problem()
prob.configure_parameters(config_file)
prob.add_doffunction(Beam_Spring.get_fe_data)
prob.set_starting_function(Beam_Spring.eigen_solve)
prob.set_zero_function(Beam_Spring.time_solve)

# Initialise forcing parameters if continuation is forced
Beam_Spring.forcing_parameters(prob.parameters)

# Continuation starting point
start = StartingPoint(prob)
start.starting_values_from_function()

# Logger
log = Logger(prob)

# Solve continuation on problem
con = ConX(prob, start, log)
con.run()
