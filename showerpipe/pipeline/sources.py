from functools import reduce
from typing import List

import graphicle as gcl

from showerpipe._base import GeneratorAdapter
from showerpipe.generator import PythiaGenerator
from showerpipe.pipeline._base import DataSubject, DataSink


def composite_fn(*func):
    def compose(f, g):
        return lambda x: f(g(x))
    return reduce(compose, func, lambda x: x)


class ShowerSource(DataSubject):
    """Provides an extensible interface to automated data generation and
    post-processing.
    """
    def __init__(self, data_generator: GeneratorAdapter):
        self.__generator = data_generator
        self.__observers: List[DataSink] = []
        self.data = None

    def attach(self, observer):
        """Attaches an observer to handle the data once generated."""
        if observer not in self.__observers:
            self.__observers.insert(0, observer)

    def detach(self, observer):
        """Removes an observer from the pipeline."""
        try:
            self.__observers.remove(observer)
        except ValueError:
            pass

    def notify(self):
        """Notifies all observers that new data is available."""
        observers = self.__observers.copy()
        if not isinstance(self.__observers[0], DataSink):
            raise ValueError(
                "The last element in the pipeline must be a sink."
            )
        else:
            sink = observers.pop(0)
        filters = [observer.apply for observer in observers]
        pipe = composite_fn(*filters)
        return sink.flush(pipe(self.data))

    def terminate(self):
        observers = self.__observers.copy()
        if not isinstance(self.__observers[0], DataSink):
            raise ValueError(
                "The last element in the pipeline must be a sink."
            )
        else:
            sink = observers.pop(0)
        sink.close()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__generator)

    def __next__(self):
        try:
            data = next(self.__generator)
            self.data = gcl.Graphicle.from_numpy(
                edges=data.edges,
                pmu=data.pmu,
                pdg=data.pdg,
                color=data.color,
                final=data.final,
            )
            return self
        except StopIteration as e:
            self.terminate()
            raise e
