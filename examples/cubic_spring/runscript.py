from core.problem import Prob
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint

from cubic_spring import Cubic_Spring

# Problem
prob = Prob()
prob.read_contparams("contparameters.json")
prob.add_doffunction(Cubic_Spring.get_fe_data)
prob.add_icfunction(Cubic_Spring.eigen_solve)

prob.add_zerofunction(Cubic_Spring.time_solve)

# Initialise forcing parameters if continuation is forced
Cubic_Spring.forcing_parameters(prob.cont_params)

# Continuation starting point
start = StartingPoint(prob)
start.get_startingpoint()

# Logger
log = Logger(prob)

# Solve continuation on problem
con = ConX(prob, start, log)
con.solve()
