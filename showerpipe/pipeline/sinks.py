from typing import Optional
from contextlib import ExitStack

# from showerpipe.pipeline._base import DataObserver
from showerpipe.pipeline._base import DataSink


class HdfSink(DataSink):
    def __init__(
            self,
            path: str,
            process_name: str,
            strict_edges: bool = True,
            rank: Optional[int] = None,
    ):
        from heparchy.write.hdf import HdfWriter  # type: ignore
        # TODO: remove line below
        self.process_name = process_name
        if rank is not None:
            path_list = path.split('.')
            path_list[-2] = path_list[-2] + f'-{rank}'
            path = '.'.join(path_list)
        self.__stack = ExitStack()
        self.__file_obj = self.__stack.enter_context(
                HdfWriter(path=path))
        self.__process = self.__stack.enter_context(
                self.__file_obj.new_process(name=process_name))
        self.strict_edges = strict_edges

    def flush(self, data):
        with self.__process.new_event() as event:
            event.set_pmu(data.pmu.data)
            event.set_color(data.color.data)
            event.set_pdg(data.pdg.data)
            event.set_status(data.status.data)
            event.set_helicity(data.helicity.data)
            event.set_mask('final', data.final.data)
            edge_kwargs = {
                'data': data.edges,
                'strict_size': self.strict_edges
                }
            if len(data.adj.weights) > 0:
                edge_kwargs['weights'] = data.adj.weights
            event.set_edges(**edge_kwargs)

    def close(self):
        self.__stack.close()
