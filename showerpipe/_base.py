from abc import ABC, abstractmethod


class GeneratorAdapter(ABC):
    """Adapter pattern interface for data generators."""

    @abstractmethod
    def __iter__(self):
        """Enforce the objects as iterables."""

    @abstractmethod
    def __next__(self):
        """Update underlying data to next iteration and return self."""

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
