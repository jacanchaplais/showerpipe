"""A simple plugin loader."""

from typing import List
import importlib


class PluginInterface:
    """A plugin has a single function called initialise."""

    @staticmethod
    def initialise() -> None:
        """Initialise the plugin."""

def import_module(name: str) -> PluginInterface:
    return importlib.import_module(name) # type: ignore

def load_plugins(plugins: List[str]) -> None:
    """Load the plugins defined in the plugins list."""
    for plugin_name in plugins:
        plugin = import_module(plugin_name)
        plugin.initialise()
