# Defines the FittingProblem class

import numpy as np
from scipy.optimize import minimize
import pyabc
import tempfile
import os

import matplotlib.pyplot as plt
import seaborn as sns

from .error_models import sumofsquares
from .dataset import Dataset

class FittingProblem:

    #### ---- Initialization of a generic FittingProblem object --- ####

    def __init__(self):

        self.data: dict = None
        self.simulator: function = None
        self.loss: function = None
        self.prior = None
        self.intguess: dict = None
        self.defaultparams: dict = None
        
        self.optimization_result = None
        self.abc_history = None
        self.accepted = None
    
    #### ---- Definition of complete loss / likelihood functions ---- ###
    
    def define_loss(self):
        
        error_models = self.data.error_models
        error_models_closured = []
        k = np.sum(self.parameters.free)

        # iterate over all error models
        for error_model in error_models:
            
            # check if we need to encapsulate additional input arguments
             
            if error_model == sumofsquares:
                error_models_closured.append(error_model)

            elif error_model == negloglike:
                def errmod_close(sim, obs):
                    return negloglike(sim, obs, k)
                error_models_closured.append(errmod_close)

            else: 
                raise(ValueError(f'Error model not implemented for automatic loss generation: {error_model}'))
            

        def lossfun(sim: Dataset, obs: Dataset):

            lossval = 0
            # TODO: add weight functionality
            for (i,nm) in enumerate(obs.names):
                lossval += error_models_closured[i](sim[nm], obs[nm])
            return lossval

        self.loss = lossfun

    def simulate(self):
        return self.simulator(self.parameters)

    #### ---- Generic plotting of priors ---- ####
    # TODO: this should live somewhere else, e.g. an pyABCFit subclass

    def plot_priors(self, **kwargs):
        """
        Plot pdfs of the prior distributions. Kwargs are passed down to the plot command.
        """
        
        nrows = int(np.ceil(len(self.prior.keys())/3))
        ncols = np.minimum(3, len(self.prior.keys()))

        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (12,6*nrows))
        ax = np.ravel(ax)

        for i,p in enumerate(self.prior.keys()):

            xrange = np.geomspace(self.prior[p].ppf(0.0001),self.prior[p].ppf(0.9999), 10000)
            ax[i].plot(xrange, self.prior[p].pdf(xrange), **kwargs)
            ax[i].set(xlabel = p)

        ax[0].set(ylabel = "Prior density")
        sns.despine()

        return fig, ax
    

    
    def prior_sample(self):
        """
        Draw a sample from the priors.
        """

        samples = [self.prior[p].rvs() for p in self.prior.keys()]
        return dict(zip(self.prior.keys(), samples))
    

    def define_lognorm_prior(self, p_int = None, sigma = 1):
        """
        Define log-normal priors with median equal to initial guess and constant sigma (SD of log values).
        """

        if not p_int:
            p_int = self.intguess

        self.prior = pyabc.Distribution()

        for (par,val) in zip(p_int.keys(), p_int.values()):
            self.prior[par] = pyabc.RV("lognorm", sigma, 0, val)
    
    def prior_predictive_check(self, n = 100):
        """
        Evaluates n prior samples. 
        """
        
        self.prior_predictions = []

        for i in range(n): 
            sim = self.simulator(self.prior_sample())
            self.prior_predictions.append(sim)

    def run_bayesian_inference(
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
        
        # pyabc expects the data in dict format 
        # if the data is not already given in dict format, 
        # we define a dict with a single entry, 
        # and define a wrapper around the loss function
         
        if type(self.data)!=dict: 
            data_smc = {'data' : self.data}

            def loss_SMC(predicted, data):
                return self.loss(predicted, data["data"])
            
        # if the data is already given in dict format, we assume that we can use it as is, 
        # same for the loss function
        else:
            data_smc = self.data
            loss_SMC = self.loss

        # setting things up
        abc = pyabc.ABCSMC( 
            self.simulator, 
            self.prior, 
            loss_SMC, 
            population_size=popsize
            )
         
        db_path = os.path.join(tempfile.gettempdir(), temp_database) # pyABC stores some information in a temporary file, this is set up here
        abc.new("sqlite:///" + db_path, data_smc) # the data is defined as a database entry
        history = abc.run( # running the SMC-ABC
            max_total_nr_simulations = max_total_nr_simulations, # we set a limit on the maximum number of simulations to run
            max_nr_populations = max_nr_populations # and a limit on the maximum number of populations, i.e. successive updates of the probability distributions
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

        sample_ar = self.accepted.sample(weights = 'weight')[list(self.prior.keys())].iloc[0]
        return dict(zip(self.prior.keys(), sample_ar))


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
            list(self.prior.keys()),
            np.array(self.accepted.loc[self.accepted.weight.argmax()]) 
        ))

    
    def __repr__(self):
        return f"FittingProblem(data={self.data}, simulator={self.simulator}, prior={self.prior}, intguess={self.intguess})"

def SSQ(D, P):

    return np.sum((D - P)**2)


def logMSE(D, P):
    """
    Mean squared error of log-transformed values.
    """

    return np.sum(((np.log(D + 1) - np.log(P + 1))**2)/len(D)) 


def logSSQ(D, P):      
    """
    Sum of squared error of log-transformed values.
    """

    return np.sum(((np.log(D + 1) - np.log(P + 1))**2)) 
