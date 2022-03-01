import inspect
from enum import Enum
from typing import Dict, Callable, Any, Union, Optional

from showerpipe.pipeline._base import DataSink, DataFilter


PipePiece = Union[DataFilter, DataSink]
create_funcs: Dict[str, Callable[..., PipePiece]] = {}


def register(name: str, creation_func: Callable[..., PipePiece]):
    """Register a new data filter or sink."""
    create_funcs[name] = creation_func


def unregister(name: str):
    """Unregister a data filter or sink."""
    create_funcs.pop(name, None)


def create(arguments: Dict[str, Any], rank: Optional[int] = None) -> PipePiece:
    """Create a data filter orsink of a specific type, given a
    dictionary of arguments.
    """
    args_copy = arguments.copy()
    name = args_copy.pop('type')
    try:
        creation_func = create_funcs[name]
        creation_func_spec = inspect.getfullargspec(creation_func)
        creation_func_args = creation_func_spec.args
        if 'rank' in creation_func_args:
            args_copy['rank'] = rank
        return creation_func(**args_copy)
    except KeyError:
        raise ValueError(f'Unknown filter or sink of name {name}.') from None
