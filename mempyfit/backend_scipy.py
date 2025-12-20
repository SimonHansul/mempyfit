from scipy.optimize import minimize
from .fitting_problem import FittingProblem
from multipledispatch import dispatch


class ScipyBackend:
    def __init__(self, prob: FittingProblem):

        fitted_param_names, fitted_param_values = zip(
            *[(n, v) for f, n, v in zip(prob.parameters.free, prob.parameters.names, prob.parameters.values) if f]
        )

        self.fitted_param_names  = list(fitted_param_names) # extract names of fitted parameters
        self.intguess = list(fitted_param_values) # use currently assigned parameter values as initial guess
        self.estimates = None # estimates will be assigned once we solved the problem
        self.prob = prob # store a reference to the fitting problem

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

        opt = minimize(
            self.objective_function, # objective function 
            self.intguess, # initial guesses
            method = method, # optimization method to use
            **kwargs
            )
            
        print(f"Fitted model using {method} method.")

        self.estimates = opt.x
        self.opt_result = opt

    
    def get_fitted_sim(self):

        self.prob.parameters.assign(self.fitted_param_names, self.estimates)
        sim = self.prob.simulate()

        return sim



            

