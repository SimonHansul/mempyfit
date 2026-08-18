# Defines the FittingProblem class

import numpy as np
import tempfile
import os

import matplotlib.pyplot as plt
import seaborn as sns

from .error_models import sumofsquares
from .dataset import Dataset

class FittingProblem:
    """Generic fitting problem container.

    Holds observed data, a simulator, loss function, and optimization results.

    Example:
        >>> problem = FittingProblem()
        >>> problem.simulator = lambda params: ...
    """

    #### ---- Initialization of a generic FittingProblem object --- ####

    def __init__(self):
        """Initialize empty fitting problem fields."""

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
        """Construct a complete loss function from dataset error models.

        This method wraps each dataset-specific error model into a unified
        loss function that can be evaluated on simulated and observed datasets.

        Example:
            >>> problem.define_loss()
        """
        
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
        """Run the simulator with current parameter values.

        Returns:
            Dataset: Simulation output from the model.

        Example:
            >>> sim = problem.simulate()
        """
        return self.simulator(self.parameters)    

    def __repr__(self):
        return f"FittingProblem(data={self.data}, simulator={self.simulator}, prior={self.prior}, intguess={self.intguess})"

def SSQ(D, P):

    """Compute the sum of squared errors between two arrays.

    Args:
        D: Observed values.
        P: Predicted values.

    Returns:
        float: Sum of squared differences.

    Example:
        >>> SSQ(np.array([1,2]), np.array([1,3]))
    """

    return np.sum((D - P)**2)


def logMSE(D, P):
    """
    Mean squared error of log-transformed values.

    Args:
        D: Observed values.
        P: Predicted values.

    Returns:
        float: Mean squared error on log-transformed data.

    Example:
        >>> logMSE(np.array([1,2]), np.array([1,3]))
    """

    return np.sum(((np.log(D + 1) - np.log(P + 1))**2)/len(D)) 


def logSSQ(D, P):      
    """
    Sum of squared error of log-transformed values.

    Args:
        D: Observed values.
        P: Predicted values.

    Returns:
        float: Sum of squared log errors.

    Example:
        >>> logSSQ(np.array([1,2]), np.array([1,3]))
    """

    return np.sum(((np.log(D + 1) - np.log(P + 1))**2)) 
