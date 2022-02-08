from functools import reduce
from typing import List, Any, Union

import graphicle as gcl

from showerpipe._base import GeneratorAdapter
from ._base import DataSubject, DataSink, DataFilter, PipeBase


def _composite_fn(*func):
    def compose(f, g):
        return lambda x: f(g(x))
    return reduce(compose, func, lambda x: x)


class PipeSequence(PipeBase):
    def __init__(self):
        self.__filters: List[DataFilter] = []

    @property
    def end(self) -> Union[PipeBase, DataSink]:
        return self.__end

    @end.setter
    def end(self, end: Union[PipeBase, DataSink]) -> None:
        self.__end = end

    @property
    def is_final(self) -> bool:
        return isinstance(self.__end, DataSink)

    def add(self, component: DataFilter) -> None:
        self.__filters.insert(0, component)

    def remove(self, component: DataFilter) -> None:
        """Removes a component from the pipeline."""
        self.__filters.remove(component)

    def execute(self, data: gcl.Graphicle) -> Any:
        filters = self.__filters.copy()
        filter_funcs = [filter_.apply for filter_ in filters]
        pipe = _composite_fn(*filter_funcs)
        if self.is_final:
            return self.__end.flush(pipe(data))  # type: ignore
        else:
            return self.__end.execute(pipe(data))  # type: ignore

    def close(self) -> None:
        if self.is_final is True:
            self.__end.close()  # type: ignore


class PipeJunction(PipeBase):
    def __init__(self):
        self.__branches: List[PipeSequence] = []

    def add(self, branch: PipeSequence) -> None:
        self.__branches.append(branch)

    def remove(self, branch: PipeSequence) -> None:
        self.__branches.remove(branch)

    def execute(self, data: gcl.Graphicle) -> None:
        branches = self.__branches.copy()
        for branch in branches:
            branch.execute(data)

    def close(self) -> None:
        for branch in self.__branches:
            branch.close()


class ShowerSource(DataSubject):
    """Provides an extensible interface to automated data generation and
    post-processing.
    """
    def __init__(self, data_generator: GeneratorAdapter):
        self.__generator = data_generator
        self.__observers: List[PipeBase] = []
        self.data = None

    def attach(self, observer) -> None:
        """Attaches an observer to handle the data once generated."""
        if observer not in self.__observers:
            self.__observers.append(observer)

    def detach(self, observer) -> None:
        """Removes an observer from the pipeline."""
        try:
            self.__observers.remove(observer)
        except ValueError:
            pass

    def notify(self) -> None:
        """Notifies all observers that new data is available."""
        observers = self.__observers.copy()
        for observer in observers:
            observer.execute(self.data)

    def terminate(self):
        observers = self.__observers.copy()
        for observer in observers:
            observer.close()

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
                helicity=data.helicity,
                status=data.status,
                final=data.final,
            )
            return self
        except StopIteration as e:
            self.terminate()
            raise e
