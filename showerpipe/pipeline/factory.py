from enum import Enum
from typing import Dict, Callable, Any

from showerpipe.pipeline._base import DataSink


class PipeType(Enum):
    FILTER: str = 'filter'
    SINK: str = 'sink'


create_funcs: Dict[str, Callable[..., DataSink]] = {}


def register(sink_name: str, creation_func: Callable[..., DataSink]):
    """Register a new data sink."""
    create_funcs[sink_name] = creation_func


def unregister(sink_name: str):
    """Unregister a data sink."""
    create_funcs.pop(sink_name, None)


def create(arguments: Dict[str, Any]) -> DataSink:
    """Create a data sink of a specific type, given a dictionary of
    arguments.
    """
    args_copy = arguments.copy()
    sink_name = args_copy.pop('type')
    try:
        creation_func = create_funcs[sink_name]
        return creation_func(**args_copy)
    except KeyError:
        raise ValueError(f'Unknown sink of name {sink_name}.') from None
