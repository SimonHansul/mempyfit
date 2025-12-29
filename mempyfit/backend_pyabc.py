import pyabc
from .fitting_problem import FittingProblem
from .backend_abstract import FittingBackend
from .error_models import euclidean
from .dataset import *
from .dataset import as_dataset
import matplotlib.pyplot as plt
import numpy as np
from numbers import Real
import os
import tempfile
import pandas as pd

class pyABCBackend(FittingBackend):

    def __init__(
            self, 
            prob: FittingProblem, 
            priors = None, 
            prior_sigma = 1, 
            scaling_function = get_max
            ):

        self.prob = prob

        # TODO: this could live in the generic FittingBackend class
        fitted_param_names, fitted_param_values = zip(
            *[(n, v) for f, n, v in zip(prob.parameters.free, prob.parameters.names, prob.parameters.values) if f]
        )

        self.fitted_param_names = fitted_param_names
        self.fitted_param_values = fitted_param_values


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
        self.data_abc = dict(zip(prob.data.names, prob.data.values))

        def sim_abc(p): 
            self.prob.parameters.assign(p)
            sim = self.prob.simulator(self.prob.parameters)
            return dict(zip(sim.names, sim.values))
        
        self.sim_abc = sim_abc
        self.scaling_function = scaling_function
        self.distance_functions = [] # distance functions for different parts of the data
        self.distance_function = None # distance function for the whole data
        self.define_eculidean_distance()
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
            for (i,key) in enumerate(obs.keys()):
                distval += distance_functions[i](sim[key], obs[key])
            return distval
        
        self.dist_abc = distfun
        self.distance_functions = distance_functions
        self.scaling_factors = scaling_factors
        self.k = k

    def plot_priors(self, linecolor = 'black', linestyle = 'solid', **kwargs):
        """
        Plot pdfs of the prior distributions. Kwargs are passed down to the plot command.
        """
        
        nrows = int(np.ceil(len(self.priors.keys())/3))
        ncols = np.minimum(3, len(self.priors.keys()))

        fig, ax = plt.subplots(nrows = nrows, ncols = ncols,  **kwargs)
        ax = np.ravel(ax)

        for i,p in enumerate(self.priors.keys()):

            xrange = np.geomspace(self.priors[p].ppf(0.0001),self.priors[p].ppf(0.9999), 10000)
            ax[i].plot(xrange, self.priors[p].pdf(xrange), color=linecolor, linestyle=linestyle)
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

        # construct a log-normal distribution for each of the free parameters
        prior_dists = [pyabc.RV("lognorm", sigma, 0, val) for val in self.fitted_param_values]
        
        # pack the distributions into a `pyabc.Disribution` object
        self.priors = pyabc.Distribution(dict(zip(self.fitted_param_names, prior_dists)))

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
            sim = self.sim_abc(self.posterior_sample())
            self.retrodictions.append(sim)

    def extract_point_estimate(self):
        """
        Extract point estimate from pyabc results.
        """

        return dict(zip(
            list(self.priors.keys()),
            np.array(self.accepted.loc[self.accepted.weight.argmax()]) 
        ))
    
    def report(
            self, 
            figkwargs_marginaldists = {'figsize' : (6, 4)}, 
            figkwargs_vrc = {'figsize' : (6,4)},
            n_retrodict = 100,
            ): 
        print()
        print('#### ---- Posterior distributions ---- ####')
        print()
        #marginaldists = self.plot_priors(**figkwargs_marginaldists)
        #fig, ax = marginaldists
        #
        #sns.histplot(self.accepted, x =  'r', weights = 'weight', ax = ax[0], kde = True)
        #sns.histplot(self.accepted, x =  'K', weights = 'weight', ax = ax[1], kde = True)
        #
        #plt.show()
        print()

        marginaldists = self.plot_priors(label = "Prior", linecolor = "black", linestyle = "--", **figkwargs_marginaldists)
        fig, ax = marginaldists 

        for t in range(self.abc_history.max_t + 1):
            df, w = self.abc_history.get_distribution(m=0, t=t)

            for (i,par) in enumerate(self.priors.keys()):

                xmin = self.priors[i].ppf(0.0001)
                xmax = self.priors[i].ppf(0.9999)

                pyabc.visualization.plot_kde_1d(
                    df,
                    w,
                    xmin=xmin,
                    xmax=xmax,
                    x=par,
                    xname=par,
                    ax=ax[i],
                    label=f"Posterior at step {t}",
                )

        ax[0].legend()
        plt.show()

        print('')
        print('### --- Posterior summary --- ###')

        medians = []
        p05s = []
        p25s = []
        p75s = []
        p95s =  []
 
        for param in self.priors.keys():

            median = pyabc.weighted_median(self.accepted[param], weights = self.accepted.weight)

            p05 = pyabc.weighted_quantile(self.accepted[param], weights = self.accepted.weight, alpha = 0.05)
            p25 = pyabc.weighted_quantile(self.accepted[param], weights = self.accepted.weight, alpha = 0.25)
            p75 = pyabc.weighted_quantile(self.accepted[param], weights = self.accepted.weight, alpha = 0.75)
            p95 = pyabc.weighted_quantile(self.accepted[param], weights = self.accepted.weight, alpha = 0.95) 
            
            medians.append(median)
            p05s.append(p05)
            p25s.append(p25)
            p75s.append(p75)
            p95s.append(p95)

        posterior_summary = pd.DataFrame({
            'param'  : self.priors.keys(),
            'median' : medians, 
            'p05' : p05s, 
            'p25' : p25s, 
            'p75' : p75s, 
            'p95' : p95s
        })

        print(posterior_summary)
        print('')

        print('### ---- Visual check ---- ####')

        self.retrodict(n_retrodict)

        num_entries = len(self.prob.data.names)
        ncols = np.minimum(num_entries, 4)
        nrows = int(np.ceil(num_entries/4))

        VPC = plt.subplots(ncols=ncols, nrows=nrows, **figkwargs_vrc)
        fig, ax = VPC
        ax = np.ravel(ax)

        for (i,name) in enumerate(self.prob.data.names):
            for r in self.retrodictions:
                self.prob.data.plot(name, ax = ax[i])
                as_dataset(r, self.prob.data).plot(name, ax = ax[i], kind = 'simulation', color = 'gray', alpha = 0.1)
        #fig, ax = self.plot_fitted_sim(fig_kwargs=fig_kwargs)
        report = {'marginaldists' : marginaldists, 'VPC' : (VPC), 'posterior_summary' : posterior_summary}

        return report


    