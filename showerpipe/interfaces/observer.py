from contextlib import ExitStack

from showerpipe.interfaces._base import DataObserver


class HdfStorage(DataObserver):
    def __init__(self):
        from heparchy.write.hdf import HdfWriter
        self.__stack = ExitStack()
        self.__file_obj = self.__stack.enter_context(
                HdfWriter(path='/home/jlc1n20/messy/silly.hdf5'))
        self.__process = self.__stack.enter_context(
                self.__file_obj.new_process(name='top'))

    def update(self, data):
        with self.__process.new_event() as event:
            event.set_pmu(data.pmu)
            event.set_edges(data.edges)

    def close(self):
        self.__stack.close()
