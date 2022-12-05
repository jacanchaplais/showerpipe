"""
``showerpipe``
==============

Provides adapter and facade interfaces for high energy physics collision
event generation programs for Pythonic usage, and exposing the data via
NumPy arrays.
"""
from ._version import __version__
from . import generator
from . import lhe


__all__ = ["__version__", "generator", "lhe"]
