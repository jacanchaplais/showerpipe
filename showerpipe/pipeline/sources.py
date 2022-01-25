from typing import List

from showerpipe._base import GeneratorAdapter
from showerpipe.generator import PythiaGenerator
from showerpipe.pipeline._base import DataSubject, DataSink


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
            self.__observers.append(observer)

    def detach(self, observer):
        """Removes an observer from the pipeline."""
        try:
            self.__observers.remove(observer)
        except ValueError:
            pass

    def notify(self):
        """Notifies all observers that new data is available."""
        for observer in self.__observers:
            observer.flush(self.data)

    def terminate(self):
        for observer in self.__observers:
            observer.close()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__generator)

    def __next__(self):
        try:
            self.data = next(self.__generator)
            return self
        except StopIteration as e:
            self.terminate()
            raise e
