from mempyfit import Dataset, nll_multinomial
import pandas as pd 
import numpy as np

import os

wd = os.getcwd()

data = Dataset()

tS = pd.read_csv(
    "survival/longispina_nickel_survival.csv", 
    comment = "#"
    )[['tday', 'Ni_nM', 'fraction_surviving']]

tS = tS.sort_values(by=["Ni_nM", "tday"])

data.add(
    name = 'tS', 
    value = tS.to_numpy(), 
    units = ['d', 'nM Ni$^{2+}$', '-'],
    labels = ['time', 'treatment', 'survival'], 
    title = 'Survival over time and Ni exposure',
    grouping_vars = 1, # index of the column of the grouping variable (here: treatment)
)


