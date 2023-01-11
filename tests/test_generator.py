import tempfile as tf
import contextlib as ctx
from pathlib import Path
import operator as op
import itertools as it
import typing as ty

import showerpipe as shp


TEST_DIR = Path(__file__).parent.resolve()
DATA_PATH = TEST_DIR / "tt_bb_100.lhe.gz"


def all_equal(iterable: ty.Iterable[ty.Any]) -> bool:
    """Returns True if all the elements are equal to each other."""
    g = it.groupby(iterable)
    return next(g, True) and not next(g, False)


@ctx.contextmanager
def config_file(
    isr: bool = True, fsr: bool = True, mpi: bool = False, hadron: bool = True
) -> ty.Generator[Path, None, None]:
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


def test_gen_len() -> None:
    with config_file() as conf_path:
        gen = shp.generator.PythiaGenerator(conf_path, DATA_PATH, 1)
    assert len(gen) == 100


def test_event_len() -> None:
    with config_file() as conf_path:
        gen = shp.generator.PythiaGenerator(conf_path, DATA_PATH, 1)
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
