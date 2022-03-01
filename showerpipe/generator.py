"""
``showerpipe.generator``
=====================

The ShowerPipe Generator module provides a standardised Pythonic
interface to showering and hadronisation programs.

Data is generated using Python iterator objects, and provided in
NumPy arrays.

Notes
-----
The classes provided here are concrete implementations of the abstract
GeneratorAdapter class. Currently only PythiaGenerator has been
implemented, however this module may be extended with additional
concrete implementations of GeneratorAdapter. Future versions are
planned to include HerwigGenerator and AriadneGenerator interfaces.
"""

import os
import tempfile
import shutil
from functools import cached_property
from typing import Optional

import numpy as np
from typicle import Types  # type: ignore

from showerpipe._base import GeneratorAdapter
from showerpipe import _dataframe
from showerpipe.lhe import count_events, source_adapter, _LHE_STORAGE


class PythiaGenerator(GeneratorAdapter):
    """Wrapper of Pythia8 generator. Provides an iterator over
    successive showered events, whose properties expose the data
    generated via NumPy arrays.

    Parameters
    ----------
    config_file : str
        Path to Pythia .cmnd configuration file.
    me_file : Pathlike, string, or bytes
        The variable or filepath containing the LHE data. May be a path,
        string, or bytes object. If file, may be compressed with gzip.
    rng_seed : int
        Seed passed to the random number generator used by Pythia.
    types : typicle.Types
        Data container defining the types of the output physics data.

    Returns
    -------
    out : iterator
        Upon iteration a new particle shower is triggered, whose data
        is accessible via the following properties:
            edges : ndarray
                Edge list representing generation ancestry of the event
                as a directed acyclic graph.
                Provided in a structured array, with fields 'in', 'out'.
            pmu : ndarray
                Four momentum provided in a structured array, with
                fields 'x', 'y', 'z', 'e'.
            pdg : ndarray
                Particle Data Group identity codes for each particle.
            color : ndarray
                Color / anticolor pairs for each particle, provided
                in a structured array with fields 'color', 'anticolor'.
            final : ndarray
                Mask over the particle list, to extract only those in
                their final state.
    """

    import pythia8 as __pythia_lib
    import pandas as __pd

    def __init__(
            self,
            config_file: str,
            me_file: Optional[_LHE_STORAGE] = None,
            rng_seed: int = 1,
            types: Types = Types()
    ):
        self.xml_dir = os.environ['PYTHIA8DATA']
        pythia = self.__pythia_lib.Pythia(
                xmlDir=self.xml_dir, printBanner=False)
        pythia.readFile(config_file)
        pythia.readString("Print:quiet = on")
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {rng_seed}")
        if me_file is not None:
            self.__num_events = count_events(me_file)
            with source_adapter(me_file) as lhe_file:
                self.temp_me_file = tempfile.NamedTemporaryFile()
                shutil.copyfileobj(lhe_file, self.temp_me_file)
                self.temp_me_file.seek(0)
                me_path = self.temp_me_file.name
            pythia.readString("Beams:frameType = 4")
            pythia.readString(f"Beams:LHEF = {me_path}")
        pythia.init()
        pmu_type = types.pmu[0][1]
        color_type = types.color[0][1]
        edge_type = types.edge[0][1]
        self.__types = {
                'pdg': types.pdg,
                'final': types.final,
                'x': pmu_type,
                'y': pmu_type,
                'z': pmu_type,
                'e': pmu_type,
                'color': color_type,
                'anticolor': color_type,
                'in': edge_type,
                'out': edge_type,
                'status': types.h_int,
                'helicity': types.h_int,
                }
        self.__pythia = pythia
    
    def __iter__(self):
        return self

    def __len__(self):
        try:
            return self.__num_events
        except AttributeError:
            raise NotImplementedError(
                    'Length only defined when initialised with LHE file.')

    def __next__(self):
        if self.__pythia is None:
            raise RuntimeError("Pythia generator not initialised.")
        is_next = self.__pythia.next()
        if not is_next:
            if hasattr(self, 'temp_me_file'):
                self.temp_me_file.close()
            raise StopIteration("No more events left to be showered.")
        if self.__event_df is not None:
            del self.__event_df
        if self.count is not None:
            del self.count
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
                    'status': pcl.status(),
                    'helicity': pcl.pol(),
                    'anticolor': pcl.acol(),
                    'parents': sorted_tuple(pcl.motherList()),
                }, self.__pythia.event),
            )
        event_df = event_df.set_index('index')
        event_df = event_df[event_df['pdg'] != 90]
        vertex_df = _dataframe.vertex_df(event_df)
        event_df = _dataframe.add_edge_cols(event_df, vertex_df)
        event_df = event_df.drop(columns=['parents'])
        event_df = event_df.astype(self.__types, copy=False)
        event_df['out'] *= -1
        event_df['in'] *= -1
        return event_df

    @cached_property
    def count(self) -> int:
        """The number of particles in the event."""
        return len(self.__event_df)

    @property
    def edges(self) -> np.ndarray:
        return _dataframe.df_to_struc(self.__event_df[['in', 'out']])

    @property
    def pmu(self) -> np.ndarray:
        return _dataframe.df_to_struc(self.__event_df[['x', 'y', 'z', 'e']])

    @property
    def color(self) -> np.ndarray:
        return _dataframe.df_to_struc(self.__event_df[['color', 'anticolor']])

    @property
    def pdg(self) -> np.ndarray:
        return self.__event_df['pdg'].values

    @property
    def final(self) -> np.ndarray:
        return self.__event_df['final'].values

    @property
    def helicity(self) -> np.ndarray:
        """The polarisation of the particles. Particles without a
        polarisation (or where it is unknown) have a value of 9.
        """
        return self.__event_df['helicity'].values

    @property
    def status(self) -> np.ndarray:
        """Pythia-specific status codes. See link for details.
        https://pythia.org/latest-manual/ParticleProperties.html
        """
        return self.__event_df['status'].values
