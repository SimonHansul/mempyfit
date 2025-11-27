# test_dataset.py
import pytest
import numpy as np
from dataset import Dataset, DimensionalityType  # assuming your code is in dataset.py


# Celsius to Kelvin conversion helper
def C2K(temp_c):
    return temp_c + 273.15