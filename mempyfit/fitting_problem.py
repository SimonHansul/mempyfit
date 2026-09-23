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
        grouping_vars = self.data.grouping_vars
        error_models_closured = []
        #k = np.sum(self.parameters.free)

        # iterate over all error models
        for (error_model, gvars) in zip(error_models, grouping_vars):
         
            # if there are no grouping vars, there is no more work to do here
            if len(gvars)==0:
                error_models_closured.append(error_model)
            else:
                # in case we have grouping variables to consider: 
                # create a closure that captures the groupings
                # this is relevant for error models that consider temporal dependency 
                # (e.g. multinomial likelihood)
                def error_model_closured(sim: np.ndarray, data: np.ndarray):
                    l = 0
                    for gvar in grouping_vars:
                        levels =  np.unique(data[:,gvar])
                        for gval in levels:
                            idxs = np.ravel(sim[:,gvar] == gval)
                            sim_sub = sim[idxs,:]
                            data_sub = data[idxs,:]
                            l += error_model(sim_sub, data_sub)

                    return l

                error_models_closured.append(error_model_closured)

        # assemble loss function for the entire dataset

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
