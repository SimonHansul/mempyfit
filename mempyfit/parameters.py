from dataclasses import dataclass, field
from multipledispatch import dispatch
import numpy as np

@dataclass
class Parameters:
    """Container for model parameter definitions.

    Stores names, values, free flags, units, and labels for parameters.

    Example:
        >>> params = Parameters({'k': {'value':1.0, 'free':True, 'unit':'1/h', 'label':'rate'}})
    """
    names: list[str]
    values: list[float]
    free: list[bool]
    units: list[str]
    labels: list[str]

    def __init__(self, p: dict):
        """Build parameter metadata from a dictionary specification.

        Args:
            p (dict): Mapping of parameter names to dictionaries with keys 'value', 'free', 'unit', 'label'.

        Example:
            >>> params = Parameters({'k': {'value':1.0, 'free':True, 'unit':'1/h', 'label':'rate'}})
        """

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

    def as_dict(self):
        """Return parameter values as a dictionary.

        Returns:
            dict: Mapping of parameter names to values.

        Example:
            >>> params.as_dict()
            {'k': 1.0}
        """
        return dict(zip(self.names, self.values))

    def __getitem__(self, name):
        """Return the value associated with a parameter name.

        Args:
            name (str): Parameter name.

        Returns:
            float: Parameter value.

        Example:
            >>> params['k']
            1.0
        """
        return self.values[self._index[name]]
    
    def __setitem__(self, name, value):
        """Assign a new value for an existing parameter.

        Args:
            name (str): Parameter name.
            value: New parameter value.

        Returns:
            None

        Example:
            >>> params['k'] = 1.2
        """
        self.values[self._index[name]] = value

    @dispatch(list, list)
    def assign(self, names, values):
        """Assign multiple parameter values by name.

        Args:
            names (list or tuple): Parameter names.
            values (list or np.ndarray): Corresponding parameter values.

        Returns:
            None

        Example:
            >>> params.assign(['k'], [1.2])
        """
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
        """Assign parameter values from a dictionary.

        Args:
            p (dict): Mapping of parameter names to values.

        Returns:
            None

        Example:
            >>> params.assign({'k': 1.2})
        """
        for (name,value) in p.items():
            self[name] = value
        

        