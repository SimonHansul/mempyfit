from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
import numpy as np
import warnings
from multipledispatch import dispatch

import matplotlib.pyplot as plt
import seaborn as sns

from .error_models import sumofsquares


class AbstractDataset:
    """Abstract base class for dataset containers."""
    pass


class AbstractDataset:
    pass

class DimensionalityType(Enum):
    ZEROVARIATE  = auto()
    UNIVARIATE   = auto()
    MULTIVARIATE = auto()

@dispatch(np.ndarray)
def get_n(ar):
    return np.shape(ar)[0]

@dispatch(np.ndarray)
def get_max(ar):
    return np.max(ar[:,-1])

@dispatch(float)
def get_max(x):
    return x

@dispatch(int)
def get_max (x):
    return x

@dispatch(np.ndarray)
def make_empty_like(value):
    """Create empty array with same shape and dtype."""
    return np.empty_like(value)

@dispatch((int, float, np.number))
def make_empty_like(value):
    """Return NaN for scalar values."""
    return np.nan

@dispatch(object)
def make_empty_like(value):
    """Fallback for other types - return as is."""
    return value

@dataclass
class Dataset(AbstractDataset):
    metadata: dict = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    values: list = field(default_factory=list)  # could be numbers or np.ndarray
    units: list[list[str]] = field(default_factory=list)
    labels: list[list[str]] = field(default_factory=list)
    error_models: list[callable] = field(default_factory=list)
    titles: list[list[str]] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    temperature_units: list[str] = field(default_factory=list)
    dimensionality_types: list[DimensionalityType] = field(default_factory=list)
    bibkeys: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)

    def add(
        self,
        name: str,
        value,
        units = None,
        labels = None,
        error_model: callable = sumofsquares, 
        title: str = "",
        temperature: float = np.nan,
        temperature_unit: str = "K",
        dimensionality_type: DimensionalityType | None = None,
        bibkey: str = "",
        comment: str = "",
    ) -> None:
        """
        Add a data entry to a dataset. 
        Minimum information needed are 
        - name 
        - value
        - units 
        - labels
        """

        # Check temperature consistency
        
        if not np.isnan(temperature) and (temperature_unit == "K") and (temperature < 200):
            raise ValueError(f"Implausible temperature {temperature} K given for {name}")

        # Infer dimensionality
        if dimensionality_type is None:
            if np.isscalar(value):
                dimensionality_type = DimensionalityType.ZEROVARIATE
            elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 2:
                dimensionality_type = DimensionalityType.UNIVARIATE
            else:
                dimensionality_type = DimensionalityType.MULTIVARIATE

        # Normalize units and labels
        if isinstance(units, str):
            units = [units]
        if isinstance(labels, str):
            labels = [labels]

        # Push to dataset
        self.names.append(name)
        self.values.append(value)
        self.units.append(units)
        self.error_models.append(error_model)
        self.labels.append(labels)
        self.titles.append(title)
        self.temperatures.append(temperature)
        self.temperature_units.append(temperature_unit)
        self.dimensionality_types.append(dimensionality_type)
        self.bibkeys.append(bibkey)
        self.comments.append(comment)

    def __getitem__(self, name: str):
        if name not in self.names:
            raise KeyError(f"Entry '{name}' not found in dataset.")
        idx = self.names.index(name)
        return self.values[idx]

    def __setitem__(self, name: str, value):
        if name not in self.names:
            raise KeyError(f"Entry '{name}' not found in dataset.")
        idx = self.names.index(name)
        self.values[idx] = value

    def empty_like(self):
        """
        Create a new Dataset with the same structure but with empty arrays.
        Scalar values remain as NaN, arrays are replaced with empty_like versions.
        """
        new_dataset = Dataset(metadata=self.metadata.copy())
        
        for i, name in enumerate(self.names):
            value = self.values[i]
            
            # Create empty version using dispatched function
            empty_value = make_empty_like(value)
            
            new_dataset.add(
                name=name,
                value=empty_value,
                units=self.units[i].copy() if isinstance(self.units[i], list) else self.units[i],
                labels=self.labels[i].copy() if isinstance(self.labels[i], list) else self.labels[i],
                error_model=self.error_models[i],
                title=self.titles[i],
                temperature=self.temperatures[i],
                temperature_unit=self.temperature_units[i],
                dimensionality_type=self.dimensionality_types[i],
                bibkey=self.bibkeys[i],
                comment=self.comments[i]
            )
        
        return new_dataset

    def dump(self, container):
        """
        Create an independent deep copy of the Dataset with all current values.
        Useful for storing snapshots of simulation results.
        """
        new_dataset = Dataset(metadata=self.metadata.copy())
        
        for i, name in enumerate(self.names):
            value = self.values[i]
            
            # Create a copy of the value
            if isinstance(value, np.ndarray):
                copied_value = value.copy()
            else:
                copied_value = value  # scalars don't need copying
            
            new_dataset.add(
                name=name,
                value=copied_value,
                units=self.units[i].copy() if isinstance(self.units[i], list) else self.units[i],
                labels=self.labels[i].copy() if isinstance(self.labels[i], list) else self.labels[i],
                error_model=self.error_models[i],
                title=self.titles[i],
                temperature=self.temperatures[i],
                temperature_unit=self.temperature_units[i],
                dimensionality_type=self.dimensionality_types[i],
                bibkey=self.bibkeys[i],
                comment=self.comments[i]
            )
        
        container.add(new_dataset)

    def getinfo(self, name: str) -> OrderedDict:
        if name not in self.names:
            raise KeyError(f"Entry '{name}' not found in dataset.")
        idx = self.names.index(name)
        return OrderedDict([
            ("name", self.names[idx]),
            ("value", self.values[idx]),
            ("units", self.units[idx]),
            ("labels", self.labels[idx]),
            ("title", self.titles[idx]),
            ("temperature", self.temperatures[idx]),
            ("temperature_units", self.temperature_units[idx]),
            ("dimensionality_type", self.dimensionality_types[idx].name),
            ("bibkey", self.bibkeys[idx]),
            ("comment", self.comments[idx]),
        ])

 
    def __repr__(self):
        out = [f"<Dataset with {len(self.names)} entries>"]
        for i, name in enumerate(self.names):
            out.append(
                f"  {i+1}. {name} ({self.dimensionality_types[i].name}) "
                f"[{self.units[i]}] @ {self.temperatures[i]} {self.temperature_units[i]}"
            )
        return "\n".join(out)
    
    def plot(
            self, 
            name, 
            ax = None, 
            kind = 'observation', 
            palette = None, 
            **kwargs
            ):

        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = None

        info = self.getinfo(name)
        value = info['value']

        # zerovariate data will not be plotted by default
        if is_zerovariate(value):
           if fig:
               return fig, ax 
           else:
               return None    
           
        # univariate data: we assume that the first column is the independent variable
        if is_univariate(value):
            if kind=='observation': 
                ax.scatter(value[:,0], value[:,1], **kwargs)
            elif kind=='simulation':
                ax.plot(value[:,0], value[:,1], **kwargs)
            else:
                raise(ValueError('Unknown kind {kind}. Allowed kinds are "observation" or "simulation".'))
            ax.set(
                xlabel = f"{info['labels'][0]} [{info['units'][0]}]",
                ylabel = f"{info['labels'][1]} [{info['units'][1]}]",
                title = info['title']
                )
            
            if fig:
                return fig, ax
        
        # bivariate data: we assume that the first column is independent and continuos (x-axis, e.g. time), 
        # the second column is independent and categorical (grouping variable, e.g. treatment), 
        # the third column is the response variable

        if is_bivariate(value):
            # if no palette was given, construct viridis palette 
            if not palette:
                palette = sns.color_palette('viridis', len(np.unique(value[:,1])))
            
            for (j,group) in enumerate(np.unique(value[:,1])):

                v_group = value[value[:,1]==group,:]
                
                if kind=='observation':
                    ax.scatter(v_group[:,0], v_group[:,2], label = group, color = palette[j])
                elif kind=='simulation':
                    ax.plot(v_group[:,0], v_group[:,2], label = group, color = palette[j])
                else:
                    raise(ValueError('Unknown kind {kind}. Allowed kinds are "observation" or "simulation".'))
            
            ax.legend(title=f"{info['labels'][1]} [{info['units'][1]}]")
            ax.set(
                xlabel = f"{info['labels'][0]} [{info['units'][0]}]",
                ylabel = f"{info['labels'][1]} [{info['units'][1]}]",
                title = info['title']
                )
            
            if fig:
                return fig, ax

            


@dataclass  
class Container:

    """
    A simple Container class to store multiple datasets together.
    Essentially a list of datasets. 
    Useful to store simulation results.  

    Initialize with `container = Container()`.
    
    Use 
    
    `dataset.dump(container)`

    to dump a copy of a dataset into the container.
    """

    datasets: list[Dataset] = field(default_factory=list)

    def add(self, dataset):
        self.datasets.append(dataset)


@dispatch(dict, Dataset)
def as_dataset(d: dict, parent: Dataset):
    data = Dataset()
    
    for (key,val) in d.items():
        info = parent.getinfo(key)
        data.add(
            name=key, 
            value=val,
            units=info['units'], 
            labels=info['labels']
            )

    return data

#### ---- methods to check dimensionality of data ---- ####

from numbers import Real

@dispatch(Real)
def is_zerovariate(x):
    return True

@dispatch(Real)
def is_univariate(x):
    return False

@dispatch(Real)
def is_bivariate(x):
    return False

@dispatch(np.ndarray)
def is_zerovariate(ar):
    return False

@dispatch(np.ndarray)
def is_univariate(ar):
    return np.shape(ar)[1] == 2

@dispatch(np.ndarray)
def is_bivariate(ar):
    return np.shape(ar)[1] == 3
