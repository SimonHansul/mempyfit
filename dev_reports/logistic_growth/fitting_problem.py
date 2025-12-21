from .data import *
from .parameters import *
from .simulation import *

class LogisticFit(FittingProblem):
    """
    This defines the logistic fitting problem as a custom class, 
    which inherits from the generic `FittingProblem`. 
    Instantiating it will assign the data defined in `data.py`, 
    the parameters defined in `parameters.py` 
    and the simulator defined in `simulator.py`. 

    We can instatiate it like so:
    
    ```Python
    prob = LogisticFit()
    ```

    To solve the problem, we can do

    ```Python 
    sol = prob.solve()
    ```

    
    """

    def __init__(self):
        super().__init__()
        self.data = data
        self.parameters = parameters
        self.simulator = simulator
        self.define_loss()


    def solve(self, Backend = ScipyBackend, method = 'Nelder-Mead', **kwargs):
        backend = Backend(self)
        backend.run(method = method, **kwargs)

        return backend
    

prob = LogisticFit()
