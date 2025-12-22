# Defines the FittingProblem class

import numpy as np
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
