"""
``showerpipe``
==============

Provides adapter and facade interfaces for high energy physics collision
event generation programs for Pythonic usage, and exposing the data via
NumPy arrays.
"""
from . import generator
from . import lhe
from ._version import __version__


__all__ = ["generator", "lhe", "__version__"]
