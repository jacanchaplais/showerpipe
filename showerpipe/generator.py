"""
``showerpipe.generator``
========================

The ShowerPipe Generator module provides a standardised Pythonic
interface to showering and hadronisation programs. Currently only Pythia
is supported.

Data is generated using Python iterator objects, and provided in NumPy
arrays.
"""
import collections as cl
import itertools as it
import math
import operator as op
import os
import random
import shutil
import tempfile as tf
import typing as ty
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pythia8 as _pythia8
from rich.console import Console
from rich.tree import Tree

from showerpipe import base, lhe

__all__ = ["PythiaEvent", "PythiaGenerator", "repeat_hadronize"]


@dataclass
class PythiaEvent(base.EventAdapter):
    """Interface wrapping Pythia8 events, providing access to the event
    data via NumPy arrays.

    :group: Pythia
    """

    _event: _pythia8.Event

    @cached_property
    def _pcls(self) -> ty.Tuple[ty.Any, ...]:
        return tuple(filter(lambda pcl: pcl.id() != 90, self._event))

    def __len__(self) -> int:
        """The number of particles in the event."""
        size = self._event.size()  # type: ignore
        if size > 0:
            size = size - 1
        return size

    def __str__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(len={len(self)})"

    def __rich__(self) -> str:
        return str(self)

    def _prop_map(self, prp: str) -> ty.Iterable[ty.Tuple[ty.Any, ...]]:
        return map(op.methodcaller(prp), self._pcls)

    def _extract_struc(
        self, schema: ty.OrderedDict[str, ty.Tuple[str, npt.DTypeLike]]
    ) -> base.AnyVector:
        dtype = np.dtype(list(schema.values()))
        return np.fromiter(
            zip(*map(self._prop_map, schema.keys())), dtype, count=len(self)
        )

    @property
    def edges(self) -> base.AnyVector:
        """Describes the heritage of the particle set as a DAG,
        formatted as a COO adjacency list.

        Notes
        -----
        The edges' source and destination nodes obey the following
        labeling convention: the root node has an ID of 0, the leaf
        nodes have positive IDs, and all other nodes have negative IDs.
        All absolute numerical values of non-root nodes are arbitrary.
        """
        pcls = self._pcls
        parents = tuple(map(op.methodcaller("index"), pcls))
        children_groups = it.starmap(
            lambda i, x: tuple(sorted(x)) if x else (-i,),
            enumerate(map(op.methodcaller("daughterList"), pcls), start=1),
        )
        rooted = map(op.not_, map(op.methodcaller("motherList"), pcls))
        rooted_ids = []
        vertices = cl.defaultdict(list)
        for children, parent, root in zip(children_groups, parents, rooted):
            if root:
                rooted_ids.append(parent)
            vertices[children].append(parent)
        vertices[tuple(rooted_ids)].append(0)
        incoming_dict = {}
        outgoing_dict = {}
        # to the children, the current vertex is src, to parents it's dst
        for vtx_id, (inc, outg) in enumerate(vertices.items(), start=1):
            if outg[0] == 0:
                vtx_id = 0
            else:
                vtx_id = math.copysign(vtx_id, inc[0])
            for edge_id in inc:
                incoming_dict[edge_id] = vtx_id
            for edge_id in outg:
                outgoing_dict[edge_id] = vtx_id
        edge_select = op.itemgetter(*parents)
        coo_zip = zip(edge_select(incoming_dict), edge_select(outgoing_dict))
        coo_edges = -np.fromiter(
            coo_zip, dtype=np.dtype(("<i4", 2)), count=len(parents)
        )
        out = coo_edges.view(np.dtype([("in", "<i4"), ("out", "<i4")]))
        return out.reshape(-1)

    @property
    def pmu(self) -> base.AnyVector:
        """Four-momenta of the particle set."""
        return self._extract_struc(
            cl.OrderedDict(
                {
                    "px": ("x", "<f8"),
                    "py": ("y", "<f8"),
                    "pz": ("z", "<f8"),
                    "e": ("e", "<f8"),
                }
            )
        )

    @property
    def color(self) -> base.AnyVector:
        """Colour codes assigned to each generated particle."""
        return self._extract_struc(
            cl.OrderedDict(
                {
                    "col": ("color", "<i4"),
                    "acol": ("anticolor", "<i4"),
                }
            )
        )

    @property
    def pdg(self) -> base.IntVector:
        """Particle Data Group id codes for the particle set."""
        return np.fromiter(self._prop_map("id"), np.int32, count=len(self))

    @property
    def final(self) -> base.BoolVector:
        """Boolean mask over the particle set, identifying final
        resulting particles at the end of the simulation. The leaves of
        the DAG representation.
        """
        return np.fromiter(
            self._prop_map("isFinal"), np.bool_, count=len(self)
        )

    @property
    def helicity(self) -> base.HalfIntVector:
        """Helicity eigenvalues for the particle set.

        Notes
        -----
        Pythia uses the value 9 as a sentinel to identify no eigenvalue.
        """
        return np.fromiter(self._prop_map("pol"), np.int16, count=len(self))

    @property
    def status(self) -> base.HalfIntVector:
        """Status codes annotating each particle with a description of
        its method of creation and purpose.
        See https://pythia.org/latest-manual/ParticleProperties.html.
        """
        return np.fromiter(self._prop_map("status"), np.int16, count=len(self))

    def copy(self) -> "PythiaEvent":
        """Returns a copy of the event."""
        return self.__class__(_pythia8.Event(self._event))


class PythiaGenerator(base.GeneratorAdapter):
    """Wrapper of Pythia8 generator. Provides an iterator over
    successive showered events in a PythiaEvent instance, whose
    properties expose the data via NumPy arrays.

    :group: Pythia

    Parameters
    ----------
    config_file : pathlib.Path | str
        Path to Pythia cmnd configuration file.
    lhe_file : pathlib.Path | str | bytes, optional
        The variable or filepath containing the LHE data. May be a path,
        string, or bytes object. If file, may be compressed with gzip.
    rng_seed : int
        Seed passed to the random number generator used by Pythia.
    quiet : bool
        Whether to quieten ``pythia8`` during data generation. Default
        is ``True``.

    Attributes
    ----------
    config : dict[str, dict[str, str]]
        Settings flags passed to Pythia, formatted as a dict of dicts.
    xml_dir : pathlib.Path
        Path of the Pythia XML data directory.

    Raises
    ------
    NotImplementedError
        If length of iterator is accessed, without a LHE file passed
        during initialisation.
    ValueError
        If ``rng_seed`` passed during initialisation is less than -1.
    RuntimeError
        If Pythia fails to initialise.
    """

    def __init__(
        self,
        config_file: ty.Union[str, Path],
        lhe_file: ty.Optional[lhe._LHE_STORAGE] = None,
        rng_seed: ty.Optional[int] = -1,
        quiet: bool = True,
    ) -> None:
        if rng_seed is None:
            rng_seed = random.randint(1, 900_000_000)
        elif rng_seed < -1:
            raise ValueError("rng_seed must be between -1 and 900_000_000.")
        xml_dir = os.environ["PYTHIA8DATA"]
        pythia = _pythia8.Pythia(xmlDir=xml_dir, printBanner=False)
        config: ty.Dict[str, ty.Dict[str, str]] = {
            "Print": {"quiet": "on" if quiet else "off"},
            "Random": {"setSeed": "on", "seed": str(rng_seed)},
            "Beams": {"frameType": "4"},
        }
        with open(config_file) as f:
            for line in f:
                key, val = line.partition("=")[::2]
                sup_key, sub_key = map(lambda s: s.strip(), key.split(":"))
                if sup_key.startswith("#"):
                    continue
                config.setdefault(sup_key, dict())
                config[sup_key][sub_key] = val.strip()
        if lhe_file is not None:
            self._num_events = lhe.count_events(lhe_file)
            with lhe.source_adapter(lhe_file) as f:
                self.temp_lhe_file = tf.NamedTemporaryFile()
                shutil.copyfileobj(f, self.temp_lhe_file)
                self.temp_lhe_file.seek(0)
                me_path = self.temp_lhe_file.name
            config["Beams"]["LHEF"] = me_path
        for group_key, group_val in config.items():
            for key, val in group_val.items():
                pythia.readString(f"{group_key}:{key} = {val}")
        pythia.init()
        self.config = config
        self.xml_dir = Path(xml_dir)
        self._pythia = pythia
        self._event = PythiaEvent(pythia.event)
        self._fresh_event = True

    def __rich__(self) -> Tree:
        name = self.__class__.__name__
        tree = Tree(f"{name}(xml_dir=[yellow]'{self.xml_dir}'[default])")
        for class_key, group in self.config.items():
            class_tree = tree.add(f"[blue]{class_key}")
            for name_key, value in group.items():
                class_tree.add(f"[red]{name_key} [default]= [green]{value}")
        return tree

    def __repr__(self) -> str:
        console = Console(color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()

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
    gen: PythiaGenerator, reps: ty.Optional[int] = None, copy: bool = True
) -> ty.Generator[PythiaEvent, None, None]:
    """Takes a PythiaGenerator instance with an unhadronised event
    already generated, and repeatedly hadronises the current event.

    :group: Pythia

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
        the passed ``PythiaGenerator`` instance.Setting this to
        ``False`` may improve speed and reduce memory consumption, but
        will have side-effects on ``gen``, such that its internal
        current event will be set to the most recent event yielded by
        this generator (until it terminates).
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
    UserWarning
        If Pythia fails to perform the hadronization process for a given
        event, this iterator will be empty.

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
        hadron_level = conf_copy[hadron_key].pop("all", None)
        if hadron_level is None:
            hadron_level = conf_copy[hadron_key].pop("Hadronize", "on")
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
        success: bool = gen._pythia.forceHadronLevel()
        if success is False:
            gen.overwrite_event(event_copy)
            warnings.warn(
                "Pythia cannot hadronize event. Iterator empty.", UserWarning
            )
            break
        event_out = gen._event
        if copy is True:
            event_out = event_out.copy()
        yield event_out
        gen.overwrite_event(event_copy)
        i = i + 1
