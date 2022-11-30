"""
``showerpipe.generator``
========================

The ShowerPipe Generator module provides a standardised Pythonic
interface to showering and hadronisation programs. Currently only Pythia
is supported.

Data is generated using Python iterator objects, and provided in NumPy
arrays.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import (
    Optional,
    Any,
    TypeVar,
    Generic,
    Iterable,
    Dict,
    Tuple,
    Union,
)
from collections import OrderedDict
from collections.abc import Sized
from dataclasses import dataclass
import operator as op
import random
from functools import cached_property
from copy import deepcopy

import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.typing as npt
import pandas as pd
import pythia8 as _pythia8

from showerpipe._base import GeneratorAdapter
from showerpipe.lhe import count_events, source_adapter, _LHE_STORAGE


BoolVector = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
AnyVector = npt.NDArray[Any]
E = TypeVar("E", bound=Iterable[Any])


def _vertex_df(event_df: pd.DataFrame) -> pd.DataFrame:
    vertex_df = event_df.reset_index()
    vertex_df = vertex_df.pivot_table(
        index="parents",
        values=["index"],
        aggfunc=lambda x: tuple(x.to_list()),  # type: ignore
    )
    vertex_df = vertex_df.reset_index()
    vertex_df = vertex_df.rename(columns={"parents": "in", "index": "out"})
    vertex_df.index.name = "id"
    return vertex_df


def _unpack(vtx_df: pd.DataFrame, direction: str) -> pd.DataFrame:
    vtx_col: pd.DataFrame = vtx_df[direction].reset_index()  # type: ignore
    vtx_col = vtx_col.explode(direction)
    vtx_col = vtx_col.set_index(direction)  # type: ignore
    vertex_to_edge = {"in": "out", "out": "in"}
    vtx_col.index.name = "index"
    vtx_col = vtx_col.rename(columns={"id": vertex_to_edge[direction]})
    return vtx_col


def _add_edge_cols(
    event_df: pd.DataFrame, vertex_df: pd.DataFrame
) -> pd.DataFrame:
    edge_out = _unpack(vertex_df, "out")
    edge_in = _unpack(vertex_df, "in")
    shower_df = event_df.join(edge_out)
    shower_df = shower_df.join(edge_in)
    edge_df = shower_df[["in", "out"]]
    if edge_df is None:
        raise ValueError
    if shower_df is None:
        raise ValueError
    max_id = edge_df.stack().max()
    isna: pd.Series = shower_df["out"].isna()  # type: ignore
    final: pd.Series = shower_df["final"]  # type: ignore
    num_final = np.sum(final)
    final_ids = -1 * np.arange(max_id + 1, max_id + num_final + 1)
    if not shower_df[isna]["final"].all():  # type: ignore
        raise RuntimeError(
            "Failed to add edges! Some outgoing vertices are not defined. "
            "Please report this to maintainers."
        )
    shower_df.loc[(final, "out")] = final_ids
    shower_df["out"] = shower_df["out"].astype("<i4")  # type: ignore
    return shower_df


@dataclass
class PythiaEvent(Generic[E], Sized):
    """Interface wrapping the Pythia8 events, providing access to the
    event data via numpy arrays.

    Attributes
    ----------
    pdg : ndarray[int32]
        Particle Data Group identification codes for the particle set.
    pmu : structured ndarray[float64] with "x", "y", "z", "e" fields
        Four-momenta of the particle set.
    color : structured ndarray[int32] with "color", "anticolor" fields
        Colour codes assigned to each generated particle.
    helicity : ndarray[int16]
        Helicity eigenvalues for the particle set. Pythia uses a value
        of 9 as a sentinel to identify no eigenvalue.
    status : ndarray[int16]
        Status codes annotating each particle with a description of
        its method of creation and purpose.
        See https://pythia.org/latest-manual/ParticleProperties.html.
    final : ndarray[bool_]
        Boolean mask over the particle set, identifying final resulting
        particles at the end of the simulation. The leaves of the DAG
        representation.
    edges : structured ndarray[int32] with "in", "out" fields
        Describes the heritage of the generate particle set with a DAG,
        formatted as a COO adjacency list.

    Methods
    -------
    copy()
        Produces a new event with identical particle records.
    """

    _event: E

    @cached_property
    def _pcls(self) -> Tuple[Any, ...]:
        return tuple(filter(lambda pcl: pcl.id() != 90, self._event))

    def __len__(self) -> int:
        """The number of particles in the event."""
        size = self._event.size()  # type: ignore
        if size > 0:
            size = size - 1
        return size

    def _prop_map(self, prp: str) -> Iterable[Tuple[Any, ...]]:
        return map(op.methodcaller(prp), self._pcls)

    def _extract_struc(
        self, schema: OrderedDict[str, Tuple[str, npt.DTypeLike]]
    ) -> AnyVector:
        dtype = np.dtype(list(schema.values()))
        return np.fromiter(zip(*map(self._prop_map, schema.keys())), dtype)

    @property
    def edges(self) -> AnyVector:
        edge_df = pd.DataFrame(
            self._extract_struc(
                OrderedDict(
                    {"index": ("index", "<i4"), "isFinal": ("final", "<?")}
                )
            )
        )
        edge_df["parents"] = tuple(
            map(lambda x: tuple(sorted(x)), self._prop_map("motherList"))
        )
        edge_df = edge_df.set_index("index")
        vertex_df = _vertex_df(edge_df)  # type: ignore
        edge_df = _add_edge_cols(edge_df, vertex_df)  # type: ignore
        edge_df = edge_df.drop(columns=["parents", "final"])
        edge_df["out"] *= -1  # type: ignore
        edge_df["in"] *= -1  # type: ignore
        return rfn.unstructured_to_structured(
            edge_df[["in", "out"]].values,  # type: ignore
            dtype=np.dtype([("in", "<i4"), ("out", "<i4")]),
        )

    @property
    def pmu(self) -> AnyVector:
        return self._extract_struc(
            OrderedDict(
                {
                    "px": ("x", "<f8"),
                    "py": ("y", "<f8"),
                    "pz": ("z", "<f8"),
                    "e": ("e", "<f8"),
                }
            )
        )

    @property
    def color(self) -> AnyVector:
        return self._extract_struc(
            OrderedDict(
                {
                    "col": ("color", "<i4"),
                    "acol": ("anticolor", "<i4"),
                }
            )
        )

    @property
    def pdg(self) -> IntVector:
        return np.fromiter(self._prop_map("id"), np.int32)

    @property
    def final(self) -> BoolVector:
        return np.fromiter(self._prop_map("isFinal"), np.bool_)

    @property
    def helicity(self) -> HalfIntVector:
        return np.fromiter(self._prop_map("pol"), np.int16)

    @property
    def status(self) -> HalfIntVector:
        return np.fromiter(self._prop_map("status"), np.int16)

    def copy(self) -> "PythiaEvent[E]":
        """Returns a copy of the event."""
        return self.__class__(_pythia8.Event(self._event))


class PythiaGenerator(GeneratorAdapter):
    """Wrapper of Pythia8 generator. Provides an iterator over
    successive showered events in a PythiaEvent instance, whose
    properties expose the data via NumPy arrays.

    Parameters
    ----------
    config_file : pathlib.Path | str
        Path to Pythia cmnd configuration file.
    lhe_file : pathlib.Path | str | bytes, optional
        The variable or filepath containing the LHE data. May be a path,
        string, or bytes object. If file, may be compressed with gzip.
    rng_seed : int
        Seed passed to the random number generator used by Pythia.

    Attributes
    ----------
    config : dict[str, dict[str, str]]
        Settings flags passed to Pythia, formatted as a dict of dicts.
    xml_dir : pathlib.Path
        Path of the Pythia XML data directory.

    Methods
    -------
    overwrite_event(new_event)
        Overwrites the current event with ``new_event``.

    Raises
    ------
    AttributeError
        If length of iterator is accessed, without a LHE file passed as
        a parameter.
    """

    def __init__(
        self,
        config_file: Union[str, Path],
        lhe_file: Optional[_LHE_STORAGE] = None,
        rng_seed: Optional[int] = -1,
    ) -> None:
        if rng_seed is None:
            rng_seed = random.randint(1, 900_000_000)
        elif rng_seed < -1:
            raise ValueError("rng_seed must be between -1 and 900_000_000.")
        xml_dir = os.environ["PYTHIA8DATA"]
        pythia = _pythia8.Pythia(xmlDir=xml_dir, printBanner=False)
        config: Dict[str, Dict[str, str]] = {
            "Print": {"quiet": "on"},
            "Random": {"setSeed": "on", "seed": str(rng_seed)},
            "Beams": {"frameType": "4"},
        }
        with open(config_file) as f:
            for line in f:
                key, val = line.partition("=")[::2]
                sup_key, sub_key = map(lambda s: s.strip(), key.split(":"))
                config.setdefault(sup_key, dict())
                config[sup_key][sub_key] = val.strip()
        if lhe_file is not None:
            self._num_events = count_events(lhe_file)
            with source_adapter(lhe_file) as f:
                self.temp_lhe_file = tempfile.NamedTemporaryFile()
                shutil.copyfileobj(f, self.temp_lhe_file)
                self.temp_lhe_file.seek(0)
                me_path = self.temp_lhe_file.name
            config["Beams"]["LHEF"] = me_path
        for group_key, group_val in config.items():
            for key, val in group_val.items():
                pythia.readString(f"{group_key}:{key} = {val}")
        pythia.init()
        self.config = config
        self.xml_dir = xml_dir
        self._pythia = pythia
        self._event = PythiaEvent(pythia.event)
        self._fresh_event = True

    def __next__(self) -> PythiaEvent:
        if self._pythia is None:
            raise RuntimeError("Pythia generator not initialised.")
        if hasattr(self._event, "_pcls"):
            del self._event._pcls
        is_next = self._pythia.next()
        self._fresh_event = True
        if not is_next:
            if hasattr(self, "temp_lhe_file"):
                self.temp_lhe_file.close()
            raise StopIteration
        return self._event

    def __len__(self):
        try:
            return self._num_events
        except AttributeError:
            raise NotImplementedError(
                "Length only defined when initialised with LHE file."
            )

    def overwrite_event(self, new_event: PythiaEvent) -> None:
        """Replaces contents of the current event in the generator.

        Parameters
        ----------
        new_event : PythiaEvent
            The event whose contents will overwrite the current event.
        """
        if hasattr(self._event, "_pcls"):
            del self._event._pcls
        self._event._event.clear()
        for pcl in new_event._event:
            self._event._event.append(pcl)


def repeat_hadronize(
    gen: PythiaGenerator, reps: Optional[int] = None, copy: bool = True
) -> Iterable[PythiaEvent]:
    """Takes a PythiaGenerator instance with an unhadronised event
    already generated, and repeatedly hadronises the current event.

    Parameters
    ----------
    gen : PythiaGenerator
        Instance of ``PythiaGenerator`` which has already generated at
        least one event, either by calling ``next(gen)``, or iterating
        with a for-loop.
    reps : int, optional
        Number of repetitions to queue in the iterator. If ``None``,
        will produce an infinite iterator.
    copy : bool
        Whether to return copies of the events, orphaned from ``gen``,
        the passed ``PythiaGenerator`` instance. Setting this to
        ``False`` may improve speed and reduce memory, but will have a
        side-effect on ``gen``, such that its internal current event
        will be set to the most recent event yielded by this generator.
        Default is ``True``.

    Yields
    ------
    event : PythiaEvent
        The current event with new hadronisation.

    Raises
    ------
    StopIteration
        Either when the number of rehadronised events is equal to
        ``reps``, or if this generator is invalidated by iterating
        the underlying ``PythiaGenerator`` instance passed to it.
    RuntimeError
        If ``gen`` is passed without a current event to rehadronise.
    KeyError
        If the cmnd file used to initialise ``gen`` contains flags which
        conflict with each other.
    ValueError
        If the cmnd file used to initialise ``gen`` does not set
        'HadronLevel:all = off'. See notes for more information.

    Notes
    -----
    This function wraps the first method of rehadronisation described in
    https://pythia.org/latest-manual/HadronLevelStandalone.html. In
    order to use it, 'HadronLevel:all' must be set to 'off' in the
    cmnd settings file that ``gen`` was initialised with.

    If ``reps`` is set so the generator has a finite length, the
    side-effects on ``gen`` will be reversed when the generator
    yields its last element.

    If ``reps`` is set to ``None`` and ``copy`` is set to ``False``,
    the side-effects on ``gen`` will persist, until ``gen`` is iterated
    either by passing it to ``next()`` or implicit iteration within a
    for loop.
    """
    if len(gen._event) == 0:
        raise RuntimeError(
            "The passed PythiaGenerator instance must have at least one "
            "event already generated to rehadronise. Please call "
            "next(gen) and try again."
        )
    hadron_key = "HadronLevel"
    conf_copy = deepcopy(gen.config)
    if hadron_key not in conf_copy:
        hadron_level = "on"
    else:
        hadron_level = conf_copy[hadron_key].pop("all", "on")
        if len(conf_copy[hadron_key]) > 0:
            raise KeyError(
                f"Conflicting settings for {hadron_key} provided, "
                f"including {tuple(conf_copy[hadron_key].keys())}. "
                f"Please remove all other {hadron_key} flags, apart from "
                "'all', from the cmnd file and try again."
            )
    if hadron_level == "on":
        raise ValueError(
            "In order to perform repeated hadronisation, hadronisation "
            "must be switched off in the settings cmnd file. "
            "Try initialising a new PythiaGenerator, setting the flag "
            "'HadronLevel:all = off'."
        )
    event = gen._event
    event_copy = event.copy()
    gen._fresh_event = False
    i = 0
    while (reps is None) or (i < reps):
        if gen._fresh_event is True:  # stop if entirely new event generated
            break
        gen._pythia.forceHadronLevel()
        event_out = gen._event
        if copy is True:
            event_out = event_out.copy()
        yield event_out
        gen.overwrite_event(event_copy)
        i = i + 1
