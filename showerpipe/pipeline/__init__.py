from typing import Optional
from .sources import PipeJunction, PipeSequence
from . import factory
from ._base import DataSink, DataFilter


def construct_pipeline(
        tree_dict: dict,
        rank: Optional[int] = None,
) -> PipeJunction:
    tree = PipeJunction()
    for branch_dict in tree_dict:
        sequence = PipeSequence()
        for item in branch_dict['branch']:
            if 'branch' in item:
                sequence.end = construct_pipeline(item)
            else:
                node = factory.create(item, rank)
                if isinstance(node, DataFilter):
                    sequence.add(node)
                elif isinstance(node, DataSink):
                    sequence.end = node
                else:
                    raise ValueError(f"Unknown pipe piece type: {item}.")
        tree.add(sequence)
    return tree
