from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterable, Any
from collections.abc import Sized, Iterator

import numpy.typing as npt
import numpy as np


__all__ = [
    "BoolVector",
    "IntVector",
    "HalfIntVector",
    "AnyVector",
    "EventAdapter",
    "GeneratorAdapter",
]


E = TypeVar("E", bound=Iterable[Any])
Self = TypeVar("Self")
BoolVector = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
AnyVector = npt.NDArray[Any]


class EventAdapter(ABC, Sized, Generic[E]):
    @property
    @abstractmethod
    def edges(self) -> AnyVector:
        pass

    @property
    @abstractmethod
    def pmu(self) -> AnyVector:
        pass

    @property
    @abstractmethod
    def color(self) -> AnyVector:
        pass

    @property
    @abstractmethod
    def pdg(self) -> IntVector:
        pass

    @property
    @abstractmethod
    def final(self) -> BoolVector:
        pass

    @property
    @abstractmethod
    def helicity(self) -> HalfIntVector:
        pass

    @property
    @abstractmethod
    def status(self) -> HalfIntVector:
        pass

    @abstractmethod
    def copy(self: Self) -> Self:
        """Returns a copy of the event."""


class GeneratorAdapter(ABC, Sized, Iterator[EventAdapter]):
    """Adapter pattern interface for data generators."""
