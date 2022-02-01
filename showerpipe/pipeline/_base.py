from abc import ABC, abstractmethod
from typing import Union, Any


# observer pattern interface for UIs to data generation and post-processing
class DataSubject(ABC):
    @abstractmethod
    def attach(self, observer):
        """Appends observer instance to subject's list of observers."""

    @abstractmethod
    def detach(self, observer):
        """Removes observer instance from subject's list of observers.
        """

    @abstractmethod
    def notify(self):
        """Calls update on all observer instances in subject's list."""

    @abstractmethod
    def terminate(self):
        """Sends a signal to all observer instances that data generation
        is over.
        """


class DataFilter(ABC):
    @abstractmethod
    def apply(self, data):
        pass


class DataSink(ABC):
    @abstractmethod
    def flush(self, data):
        pass

    @abstractmethod
    def close(self):
        pass


class DataObserver(ABC):
    @abstractmethod
    def add_filter(self, filt: DataFilter):
        pass

    @abstractmethod
    def add_sink(self, sink: DataSink):
        pass

    @abstractmethod
    def update(self, subject_data):
        """Performs some operation on the data contained within the
        subject instance.
        """

    @abstractmethod
    def close(self):
        """Cleanup behaviour once the data generation is over."""


class PipeBase(ABC):  # composite pattern interface
    @abstractmethod
    def add(self, component) -> None:
        pass

    @abstractmethod
    def remove(self, component) -> None:
        pass

    @abstractmethod
    def execute(self, data) -> Any:
        pass
