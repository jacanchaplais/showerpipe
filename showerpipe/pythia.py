import os
from functools import cached_property

import numpy as np
from heparchy import TYPE, REAL_TYPE

from showerpipe._interface import GeneratorAdapter
import showerpipe._dataframe as dataframe


class PythiaGenerator(GeneratorAdapter):
    import pythia8 as __pythia_lib
    import pandas as __pd


    def __init__(self, config_file: str, me_file: str=None, rng_seed: int=1):
        self.xml_dir = os.environ['PYTHIA8DATA']
        pythia = self.__pythia_lib.Pythia(
                xmlDir=self.xml_dir, printBanner=False)
        pythia.readFile(config_file)
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {rng_seed}")
        if me_file is not None:
            pythia.readString("Beams:frameType = 4")
            pythia.readString(f"Beams:LHEF = {me_file}")
        pythia.readString("Print:quiet = on")
        pythia.init()
        self.__pythia = pythia
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.__pythia == None:
            raise RuntimeError("Pythia generator not initialised.")
        is_next = self.__pythia.next()
        if not is_next:
            raise StopIteration("No more events left to be showered.")
        if self.__event_df is not None:
            del self.__event_df
        return self

    @cached_property
    def __event_df(self) -> __pd.DataFrame:
        def sorted_tuple(iterable):
            list_object = list(iterable)
            list_object.sort()
            return tuple(list_object)
        event_df = self.__pd.DataFrame(
            map(lambda pcl: {
                    'index': pcl.index(),
                    'pdg': pcl.id(),
                    'final': pcl.isFinal(),
                    'x': pcl.px(),
                    'y': pcl.py(),
                    'z': pcl.pz(),
                    'e': pcl.e(),
                    'color': pcl.col(),
                    'anticolor': pcl.acol(),
                    'parents': sorted_tuple(pcl.motherList())
                }, self.__pythia.event),
            )
        event_df = event_df.set_index('index')
        event_df = event_df[event_df['pdg'] != 90]
        vertex_df = dataframe.vertex_df(event_df)
        event_df = dataframe.add_edge_cols(event_df, vertex_df)
        event_df = event_df.drop(columns=['parents'])
        event_df = event_df.astype({
                'pdg': TYPE['int'],
                'final': TYPE['bool'],
                'x': REAL_TYPE,
                'y': REAL_TYPE,
                'z': REAL_TYPE,
                'e': REAL_TYPE,
                'color': TYPE['int'],
                'anticolor': TYPE['int'],
                'in': TYPE['int'],
                'out': TYPE['int'],
                },
            copy=False,
            )
        event_df['out'] *= -1
        event_df['in'] *= -1
        return event_df

    @property
    def edges(self) -> np.ndarray:
        return dataframe.df_to_struc(self.__event_df[['in', 'out']])

    @property
    def pmu(self) -> np.ndarray:
        return dataframe.df_to_struc(self.__event_df[['x', 'y', 'z', 'e']])

    @property
    def color(self) -> np.ndarray:
        return dataframe.df_to_struc(self.__event_df[['color', 'anticolor']])

    @property
    def pdg(self) -> np.ndarray:
        return self.__event_df['pdg'].values

    @property
    def final(self) -> np.ndarray:
        return self.__event_df['final'].values
