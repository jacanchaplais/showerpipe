from contextlib import ExitStack

from showerpipe.interfaces._base import DataObserver


class HdfStorage(DataObserver):
    def __init__(self, path: str):
        from heparchy.write.hdf import HdfWriter
        self.__stack = ExitStack()
        self.__file_obj = self.__stack.enter_context(
                HdfWriter(path=path))
        self.__process = self.__stack.enter_context(
                self.__file_obj.new_process(name='default'))

    def update(self, data):
        with self.__process.new_event() as event:
            event.set_pmu(data.pmu)
            event.set_color(data.color)
            event.set_edges(data.edges)
            event.set_pdg(data.pdg)

    def close(self):
        self.__stack.close()
