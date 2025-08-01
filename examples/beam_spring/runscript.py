from core.problem import Prob
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint

from beam_spring import Beam_Spring

# Problem
prob = Prob()
prob.read_contparams("contparameters.json")
prob.add_doffunction(Beam_Spring.get_fe_data)
prob.add_icfunction(Beam_Spring.eigen_solve)
prob.add_zerofunction(Beam_Spring.time_solve)

# Initialise forcing parameters if continuation is forced
Beam_Spring.forcing_parameters(prob.cont_params)

# Continuation starting point
start = StartingPoint(prob)
start.get_startingpoint()

# Logger
log = Logger(prob)

# Solve continuation on problem
con = ConX(prob, start, log)
con.solve()
