from dataclasses import dataclass, field
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
        self._index = {name: i for i, name in enumerate(names)}

    def __getitem__(self, name):
        return self.values[self._index[name]]
        