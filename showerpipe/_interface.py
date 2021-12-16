from abc import ABC, abstractmethod


class GeneratorAdapter(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @property
    @abstractmethod
    def edges(self):
        pass

    @property
    @abstractmethod
    def pmu(self):
        pass

    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def pdg(self):
        pass

    @property
    @abstractmethod
    def final(self):
        pass
