from .data import *
from .parameters import *
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from numbers import Real

#### ---- constants ---- ####

tested_concentrations = np.unique(data['tS'][:,1])
data_shape = np.shape(data['tS'])
n_c = data_shape[1]

observed_timepoints = np.sort(np.unique(data['tS'][:,0]))
tmin = np.min(observed_timepoints)
tmax = np.max(observed_timepoints)

sim_dataset = data.empty_like()

#### ---- ODE definition ---- ####

def GUTS_SD_const(
        t, 
        y, 
        p: Parameters, 
        C_W: Real
        ):
    """ 
    Definition of the reduced GUTS-SD ODE with constant exposure. 
    """

    D,S = y # unpacking states

    dD = p['k_d']*(C_W - D) # derivative of scaled damage
    h = p['b'] * max(0, y[0] - p['z']) + p['h_b'] # hazard rate, including background mortality
    dS = -h*S # derivative of survival probability

    return dD,dS

# helper function to collect results as numpy array 
# TODO: this is generic and should live somewhere else, e.g. a SimulationUtils package that may be imported via mempyfit  

def append_result(result, row, sim, group):
    """
    Write ODEsolution to preallocated result array, mutating the results array.
    """

    n_t = sim.t.size
    idx = slice(row, row + n_t)

    result[idx, 0] = sim.t
    result[idx, 1] = group
    result[idx, 2] = sim.y[1, :]

    return row + n_t

def simulator(
        p: Parameters,
        model = GUTS_SD_const,
        simulated_concentrations = tested_concentrations,  
        t_eval = observed_timepoints
        ):

    y0 = [0,1]
    tspan = (0,tmax)

    # creating a copy of the parameters and forcings, so I can update the forcings

    # pre-allocating simulation result
    n_c = len(simulated_concentrations)
    n_t = len(t_eval)
    tS = np.empty((n_c * n_t, 3))

    row = 0

    # for each treatment
    for C_W in simulated_concentrations:

        # solve the model with the updated input
        sim = solve_ivp(
            model,
            tspan,
            y0,
            args = (p,C_W),
            t_eval = t_eval
        )
        
        row = append_result(tS, row, sim, C_W)

    sim_dataset['tS'] = tS

    return sim_dataset