from functools import partial
from mempyfit import FittingProblem, ScipyBackend
from .data import *
from .parameters import parameters
from .simulation import simulator, GUTS_SD_const
from copy import deepcopy


class GUTSFit(FittingProblem):

    def __init__(self, model, data, parameters, simulator):
        super().__init__()
        self.model = model
        self.data = data
        self.parameters = parameters
        self.simulator = partial(simulator, model=model)
        self.define_loss()
        self.initial_parameters = deepcopy(parameters)

    def reset_params(self):
        self.parameters = deepcopy(self.initial_parameters)

    # TODO: this could live directly in the `FittingProblem` class 
    def solve(self, Backend = ScipyBackend, method = 'Nelder-Mead', verbose=False, **kwargs):
        
        self.define_loss() # always update the loss
        backend = Backend(self)

        # TODO: replace conditional with multiple dispatch
        if isinstance(backend, ScipyBackend):
            backend.run(method=method, **kwargs)
        else:
            backend.run(**kwargs)

        return backend
    

prob = GUTSFit(
    model=GUTS_SD_const, 
    data=data, 
    parameters=parameters,
    simulator=simulator
    )