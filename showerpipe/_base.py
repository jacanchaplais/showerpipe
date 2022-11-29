from abc import ABC
from collections.abc import Sized, Iterator


class GeneratorAdapter(ABC, Sized, Iterator):
    """Adapter pattern interface for data generators."""
