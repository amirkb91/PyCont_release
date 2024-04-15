from core.problem import Prob
from core.logger import Logger
from core.solver.continuation import ConX
from core.startingpoint import StartingPoint

from duffing import Duffing
from duffing_lnn import Duffing_LNN

def run(model=Duffing, predict_acc=None, pred_energy=None):
    # Instantiate object with trained LNN, if applicable
    Duffing_LNN.LNN_acceleration(pred_acc=predict_acc)
    Duffing_LNN.LNN_energy(pred_energy=pred_energy)
    
    # Problem
    prob = Prob()
    prob.read_contparams("contparameters.json")
    prob.add_doffunction(model.get_fe_data)
    prob.add_icfunction(model.eigen_solve)
    if prob.cont_params["shooting"]["method"] == "single":
        prob.add_zerofunction(model.time_solve)
    elif prob.cont_params["shooting"]["method"] == "multiple":
        prob.add_zerofunction(model.time_solve_multiple, model.time_solve)
        prob.add_partitionfunction(model.partition_singleshooting_solution)

    # Initialise forcing parameters if continuation is forced
    model.forcing_parameters(prob.cont_params)

    # Continuation starting point
    start = StartingPoint(prob)
    start.get_startingpoint()

    # Logger
    log = Logger(prob)

    # Solve continuation on problem
    con = ConX(prob, start, log)
    con.solve()
