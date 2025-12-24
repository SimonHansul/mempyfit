import pyabc
from .fitting_problem import FittingProblem
from .backend_abstract import FittingBackend
from .dataset import *
import matplotlib.pyplot as plt
import numpy as np
from numbers import Real
import os
import tempfile

class pyABCBackend(FittingBackend):

    def __init__(
            self, 
            prob: FittingProblem, 
            priors = None, 
            prior_sigma = 1, 
            scaling_function = get_max
            ):

        self.prob = prob

        # if priors were given, assign them
        if priors:
            self.priors = priors
        # if no priors were given, construct log-normal priors around initial values with fixed sigma
        else:
            self.define_lognorm_prior(
                prob,
                sigma = prior_sigma
                )
            
        # data and simulations need to be wrapped into a dict to accomodate pyabc
        data_abc = dict(zip(prob.data.names, prob.data.values))

        def sim_abc(p): 
            self.prob.parameters.assign(p)
            sim = self.prob.simulator(self.prob.parameters)
            return dict(zip(sim.names, sim.values))
        
        # it should not be necessary to change the simulator
        self.sim_abc = sim_abc
        self.data_abc = data_abc
        self.scaling_function = scaling_function
        self.distance_functions = [] # distance functions for different parts of the data
        self.distance_function = None # distance function for the whole data
        self.dist_abc = self.define_eculidean_distance()
        self.scaling_factors = []
        self.estimates = None
        self.abc_history = None 
        self.accepted = None
        self.weights = None
        self.retrodictions = None

    def define_eculidean_distance(self, scaling_function = get_max):
        
        distance_functions = []
        scaling_factors = []
        k = np.sum(self.prob.parameters.free)

        # iterate over all error models
        for val in self.prob.data.values:
            scale = scaling_function(val)

            def dist(sim, obs):
                return euclidean(sim, obs, scale)
            
            distance_functions.append(dist)
            scaling_factors.append(scale)
            
        def distfun(sim: Dataset, obs: Dataset):
            distval = 0
            # TODO: add weight functionality
            for (val,key) in obs.items():
                distval += distance_functions[i](sim[key], val)
            return distval
        
        self.distance_function = distfun
        self.distance_functions = distance_functions
        self.scaling_factors = scaling_factors
        self.k = k

    def plot_priors(self, **kwargs):
        """
        Plot pdfs of the prior distributions. Kwargs are passed down to the plot command.
        """
        
        nrows = int(np.ceil(len(self.priors.keys())/3))
        ncols = np.minimum(3, len(self.priors.keys()))

        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (12,6*nrows))
        ax = np.ravel(ax)

        for i,p in enumerate(self.priors.keys()):

            xrange = np.geomspace(self.priors[p].ppf(0.0001),self.priors[p].ppf(0.9999), 10000)
            ax[i].plot(xrange, self.priors[p].pdf(xrange), **kwargs)
            ax[i].set(xlabel = p)

        ax[0].set(ylabel = "Prior density")
        sns.despine()

        return fig, ax
    
    def prior_sample(self):
        """
        Draw a sample from the priors.
        """

        samples = [self.priors[p].rvs() for p in self.priors.keys()]
        return dict(zip(self.priors.keys(), samples))
    
    def define_lognorm_prior(self, prob: FittingProblem, sigma = 1.):
        """
        Define log-normal priors with median equal to initial guess and constant sigma (SD of log values).
        """

        fitted_param_names, fitted_param_values = zip(
            *[(n, v) for f, n, v in zip(prob.parameters.free, prob.parameters.names, prob.parameters.values) if f]
        )

        # construct a log-normal distribution for each of the free parameters
        prior_dists = [pyabc.RV("lognorm", sigma, 0, val) for val in fitted_param_values]
        
        # pack the distributions into a `pyabc.Disribution` object
        self.priors = pyabc.Distribution(dict(zip(fitted_param_names, prior_dists)))

    def prior_predictive_check(self, n = 100):
        """
        Evaluates n prior samples. 
        """
        
        self.prior_predictions = []

        for i in range(n): 
            sim = self.simulator(self.prior_sample())
            self.prior_predictions.append(sim)

    def run(
            self, 
            popsize = 1000,
            max_total_nr_simulations = 10_000, 
            max_nr_populations = 10,
            temp_database = "data.db"
            ):
        """
        Apply Bayesian inference, using Sequential Monte Carlo Approximate Bayesian Computation (SMC-ABC) 
        from the `pyABC` package.
        """

        # setting things up
        abc = pyabc.ABCSMC( 
            self.sim_abc, 
            self.priors, 
            self.dist_abc, 
            population_size=popsize, 
            )
         
        db_path = os.path.join(tempfile.gettempdir(), temp_database) # pyABC stores some information in a temporary file, this is set up here
        abc.new("sqlite:///" + db_path, self.data_abc) # the data is defined as a database entry
        history = abc.run( # running the SMC-ABC
            max_total_nr_simulations = max_total_nr_simulations, # we set a limit on the maximum number of simulations to run
            max_nr_populations = max_nr_populations, # and a limit on the maximum number of populations, i.e. successive updates of the probability distributions
            )
        
        # constructing a data frame with accepted parameter values and weights
        accepted, weights = history.get_distribution()
        accepted = accepted.reset_index().assign(weight = weights)

        print("Conducted Bayesian inference using SMC-ABC. Results are in `abc_history` and `accepted`")

        self.abc_history = history
        self.accepted = accepted

    def posterior_sample(self):
        """ 
        Draw a posterior sample from accepted particles.
        """

        sample_ar = self.accepted.sample(weights = 'weight')[list(self.priors.keys())].iloc[0]
        return dict(zip(self.priors.keys(), sample_ar))


    def retrodict(self, n = 100):
        """ 
        Generate retrodictions based on `n` posterior samples.
        """

        self.retrodictions = []

        for i in range(n): 
            sim = self.simulator(self.posterior_sample())
            self.retrodictions.append(sim)

    def extract_point_estimate(self):
        """
        Extract point estimate from pyabc results.
        """

        return dict(zip(
            list(self.priors.keys()),
            np.array(self.accepted.loc[self.accepted.weight.argmax()]) 
        ))

    