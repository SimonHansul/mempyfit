from dataclasses import dataclass, field
from multipledispatch import dispatch
import numpy as np

@dataclass
class Parameters:
    names: list[str]
    values: list[float]
    free: list[bool]
    units: list[str]
    labels: list[str]

    def __init__(self, p: dict):

        self.names = []
        self.values = []
        self.free = []
        self.units = []
        self.labels = []

        for (name,info) in p.items():
            self.names.append(name)
            self.values.append(info['value'])
            self.free.append(info['free'])
            self.units.append(info['unit'])
            self.labels.append(info['label'])

        # lookup table for faster indexing
        self._index = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, name):
        return self.values[self._index[name]]
    
    def __setitem__(self, name, value):
        self.values[self._index[name]] = value

    @dispatch(list, list)
    def assign(self, names, values):
        for (name,value) in zip(names, values):
            self[name] = value

    @dispatch(tuple, np.ndarray)
    def assign(self, names, values):
        for (name,value) in zip(names, values):
            self[name] = value
        
    @dispatch(list, np.ndarray)
    def assign(self, names, values):
        for (name,value) in zip(names, values):
            self[name] = value

    @dispatch(dict)
    def assign(self, p): 
        for (name,value) in p.items():
            self[name] = value
        

        