import tempfile as tf
import contextlib as ctx
from pathlib import Path
import operator as op
import itertools as it
import typing as ty

from hypothesis import given, settings, strategies as st
import numpy as np
import numpy.lib.recfunctions as rfn

import showerpipe as shp


TEST_DIR = Path(__file__).parent.resolve()
DATA_PATH = TEST_DIR / "tt_bb_100.lhe.gz"


def all_equal(iterable: ty.Iterable[ty.Any]) -> bool:
    """Returns True if all the elements are equal to each other."""
    g = it.groupby(iterable)
    return next(g, True) and not next(g, False)


@ctx.contextmanager
def config_file(
    isr: bool = True, fsr: bool = True, mpi: bool = True, hadron: bool = True
) -> ty.Generator[Path, None, None]:
    """Context manager creates a temporary config file, and yields the
    path, for the purpose of instantiating a new ``PythiaGenerator``.

    Parameters
    ----------
    isr : bool
        Initial state radiation. Default is ``True``.
    fsr : bool
        Final state radiation. Default is ``True``.
    mpi : bool
        Multi-parton interaction. Default is ``True``.
    hadron : bool
        Hadronization. Default is ``True``.
    """
    f = tf.NamedTemporaryFile("w")
    switch = {True: "on", False: "off"}
    f.write(f"PartonLevel:ISR = {switch[isr]}\n")
    f.write(f"PartonLevel:FSR = {switch[fsr]}\n")
    f.write(f"PartonLevel:MPI = {switch[mpi]}\n")
    if hadron is True:
        f.write("HadronLevel:Hadronize = on\n")
    else:
        f.write("HadronLevel:all = off\n")
    f.seek(0)
    try:
        yield Path(f.name)
    finally:
        f.close()


@st.composite
def generators(
    draw: st.DrawFn,
    min_seed: int = 1,
    max_seed: int = 10_000,
    **config_kwargs: bool,
) -> shp.generator.PythiaGenerator:
    """Custom strategy providing a ``PythiaGenerator`` with a random
    seed.
    """
    seed = draw(st.integers(min_seed, max_seed))
    with config_file(**config_kwargs) as conf_path:
        return shp.generator.PythiaGenerator(conf_path, DATA_PATH, seed, False)


@given(generators())
@settings(max_examples=5, deadline=None)
def test_gen_len(gen: shp.generator.PythiaGenerator) -> None:
    """Tests that the generator length matches the number of events."""
    assert len(gen) == 100


@given(generators())
@settings(max_examples=5, deadline=None)
def test_event_len(gen: shp.generator.PythiaGenerator) -> None:
    """Tests that the event length and data lengths match."""
    event = next(gen)
    event_len = len(event)
    prop_names = (
        "edges",
        "pmu",
        "color",
        "pdg",
        "final",
        "helicity",
        "status",
    )
    props = (prop(event) for prop in map(op.attrgetter, prop_names))
    lens = tuple(map(len, props))
    assert event_len != 0 and all_equal(lens) and lens[0] == event_len


@st.composite
def hadron_repeater(
    draw: st.DrawFn,
    min_seed: int = 1,
    max_seed: int = 10_000,
    reps: int = 100,
) -> ty.Iterable[shp.generator.PythiaEvent]:
    """Custom strategy, providing a repeated hadronization generator
    with a random seed, and a user-defined number of repetitions.
    """
    gen = draw(generators(min_seed, max_seed, hadron=False))
    _ = next(gen)
    return shp.generator.repeat_hadronize(gen, reps=reps)


@given(gen=hadron_repeater())
@settings(max_examples=20, deadline=None)
def test_rep_hadron_uniq(gen: ty.Iterable[shp.generator.PythiaEvent]) -> None:
    """Tests if the events following repeated hadronization are unique."""
    data = sorted(map(op.attrgetter("pdg"), gen), key=len)
    for _, data_arrays in it.groupby(data, len):
        data_pairs = it.combinations(data_arrays, 2)
        assert any(it.starmap(np.array_equal, data_pairs)) is False


def rep_hadron_colorless(gen: ty.Iterable[shp.generator.PythiaEvent]) -> bool:
    """Given a rehadronization generator, return ``False`` if colored
    particles make it to the final state.
    """
    for event in gen:
        final_colors = rfn.structured_to_unstructured(event.color[event.final])
        if not np.all(final_colors == 0):
            return False
    return True


@given(gen=generators(hadron=False))
@settings(max_examples=5, deadline=None)
def test_rep_hadron_colorless(gen: shp.generator.PythiaGenerator) -> None:
    """Tests if colored particles make it to final state after
    rehadronization. Repeats for successive hard events.
    """
    for _ in it.islice(gen, 50):
        hadron_gen = shp.generator.repeat_hadronize(gen, 10, False)
        assert rep_hadron_colorless(hadron_gen) is True
