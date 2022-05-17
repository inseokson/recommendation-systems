import pathlib
from pathlib import Path


class Dataset(object):
    def __init__(self, path):
        self.path = path

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, str):
            self._path = Path(value)
        elif isinstance(value, pathlib.PosixPath):
            self._path = value
        else:
            raise ValueError("`path` must be one of string or PosixPath")

    def load(self):
        dataset = {}
        for name in dir(self):
            has_prefix = name.startswith("_get")

            attribute = getattr(self, name)
            is_method = callable(attribute)

            if has_prefix and is_method:
                k = name[5:]
                v = attribute(self._path)
                dataset[k] = v

        return dataset
