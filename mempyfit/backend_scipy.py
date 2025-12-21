from scipy import optimize
import numpy as np
from scipy.optimize import minimize
from .fitting_problem import FittingProblem
import matplotlib.pyplot as plt

from pprint import pp


class ScipyBackend:
    def __init__(self, prob: FittingProblem):

        fitted_param_names, fitted_param_values = zip(
            *[(n, v) for f, n, v in zip(prob.parameters.free, prob.parameters.names, prob.parameters.values) if f]
        )

        self.fitted_param_names  = list(fitted_param_names) # extract names of fitted parameters
        self.k = len(fitted_param_names)
        self.intguess = list(fitted_param_values) # use currently assigned parameter values as initial guess
        self.estimates = None # estimates will be assigned once we solved the problem
        self.prob = prob # store a reference to the fitting problem
        self.bounds = optimize.Bounds(lb = np.zeros(self.k)) # by default, assume parameters to positive

        #### ---- Define the objective function to be compatible with scipy ---- ####

        def objective_function(parameter_vector):
            # assign parameters´
            prob.parameters.assign(fitted_param_names, parameter_vector)

            # run simulation
            simulation = prob.simulate()

            # call the loss function   
            return prob.loss(simulation, prob.data)
        
        self.objective_function = objective_function

    def run(self, method = 'Nelder-Mead', **kwargs):

        if method != 'differential_evolution':

            opt = minimize(
                self.objective_function, # objective function 
                self.intguess, # initial guesses
                method = method, # optimization method to use
                bounds = self.bounds,
                **kwargs
                )
        else:

            if (np.inf in self.bounds.ub) | (np.inf in self.bounds.lb):
                raise(ValueError("Need non-finite bounds for `differential_evolution method`."))

            opt = optimize.differential_evolution(
                self.objective_function, 
                bounds=self.bounds
                )
                        
        print(f"Fitted model using {method} method.")

        self.estimates = opt.x
        self.opt_result = opt

    
    def get_fitted_sim(self):

        self.prob.parameters.assign(self.fitted_param_names, self.estimates)
        sim = self.prob.simulate()

        return sim
    
    def plot_fitted_sim(self, fig_kwargs = {'figsize' : (6, 4)}): 
            
        data = self.prob.data
        fitted_sim = self.get_fitted_sim()

        num_entries = len(data.names)
        ncols = np.minimum(num_entries, 4)
        nrows = int(np.ceil(num_entries/4))

        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, **fig_kwargs)
        ax = np.ravel(ax)

        for (i,name) in enumerate(data.names):

            data.plot(name, ax = ax[i])
            fitted_sim.plot(name, ax = ax[i], kind = 'simulation')

        return fig, ax
    

    # TODO: add optional output_dir to store all results
    def report(self, fig_kwargs = {'figsize' : (6, 4)}): 
        print()
        print('#### ---- Estimated parameters ---- ####')
        print()
        estimates = dict(zip(self.fitted_param_names, self.estimates))
        pp(estimates)
        print()

        print('### ---- Visual check ---- ####')

        fig, ax = self.plot_fitted_sim(fig_kwargs=fig_kwargs)

        report = {'estimates' : estimates, 'figure' : (fig,ax)}

        return report





            

