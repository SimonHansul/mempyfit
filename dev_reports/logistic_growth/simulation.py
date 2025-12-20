from .data import *
from .parameters import *
from scipy.integrate import solve_ivp
from mempyfit import Dataset
  
def dNdt(t, y, p: dict):
    """
    Definition of the log-logistic ODE. 

    - t: Time point. In most cases this is not used, but scipy expects it to be part of the ODE definition.
    - y: State variables. These are always stored in an array, but can be "unpacked" at the beginning of the ODE definition.
    - p: Parameters. The syntax below implies that parameters are stored in a dictionary, which improves readability compared to a list or array.
    """
    
    N = y[0] # unpacking state variables

    return p['r']*N*(1-(N/p['K'])) 

tmin = np.min(data['t-OD'][:,0])
tmax = np.max(data['t-OD'][:,0])+1


def simulator(parameters):

    """
    Simulate the logistic growth model.

    args:
    -p: a dictionary with proposed parameter values

    kwargs:
    -t_eval: time-points to be evaluated by the ODE solver. By default, the unique time-points in a globally defined data frame called `data`
    """
    
    sim_dataset = data.empty_like()
    y0 = [data['t-OD'][0,1]] # initial conditions for the ODE

    sim = solve_ivp(    
        dNdt,
        (tmin, tmax), 
        y0,
        args = (parameters,),
        t_eval = data['t-OD'][:,0] 
    )

    sim_array = np.array([sim.t, sim.y[0]]).transpose()
    
    # Return as Dataset to match expected format
    sim_dataset['t-OD'] = sim_array

    return sim_dataset
