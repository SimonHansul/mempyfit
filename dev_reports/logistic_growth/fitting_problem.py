from .data import *
from .parameters import *
from .simulation import *

prob = FittingProblem()
prob.data = data
prob.parameters = parameters
prob.simulator = simulator
