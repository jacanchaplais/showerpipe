"""
``showerpipe.lhe``
==================

The ShowerPipe Les Houches functions utilise xml parsing techniques to 
redistribute and repeat hard events, outputting valid lhe files.
"""

import gzip
from copy import deepcopy
from itertools import chain
from functools import cached_property
from xml.sax.saxutils import unescape

from lxml import etree


class LheData:
    """Container for the Les Houches file content."""
    def __init__(self, content: bytes):
        self.__root = etree.fromstring(content)

    @property
    def content(self) -> bytes:
        """The LHE file contents in bytes."""
        return _root_to_bytes(self.__root)

    @content.setter
    def content(self, data):
        del self.num_events
        self.__root = etree.fromstring(data)

    @cached_property
    def num_events(self):
        return len(self.__root.findall('event'))

    @property
    def __event_iter(self):
        return self.__root.iter('event')

    def repeat(self, repeats: int, inplace: bool=False) -> bytes:
        """Modifies LHE content, repeating each event the number of
        times given by repeats.

        Parameters
        ----------
        repeats : int
            The number of times to repeat the events.
        inplace : bool
            If True, modifies content inplace and returns None.

        Returns
        -------
        content : bytes
            Bytestring representation of the LHE file.

        Notes
        -----
        Repeat means, eg. repeat([A, B, C], 2) => [A, A, B, B, C, C]

        Modifying inplace is more computationally efficient than setting
        the content explicitly to bytestring output, but may lead to
        unexpected side effects. Should be used with caution.
        """
        return self.__event_duplicator(
                repeats, inplace, dup_strat=self.__repeat_order)

    def tile(self, repeats: int, inplace: bool=False) -> bytes:
        """Modifies LHE content, tile repeating all events the
        number of times given by repeats.

        Parameters
        ----------
        repeats : int
            The number of times to repeat the events.
        inplace : bool
            If True, modifies content inplace and returns None.

        Returns
        -------
        content : bytes
            Bytestring representation of the LHE file.

        Notes
        -----
        Tile repeat means, eg. tile([A, B, C], 2) => [A, B, C, A, B, C].

        Modifying inplace is more computationally efficient than setting
        the content explicitly to bytestring output, but may lead to
        unexpected side effects. Should be used with caution.
        """
        return self.__event_duplicator(
                repeats, inplace, dup_strat=self.__tile_order)

    def __tile_order(self, x): return x
    def __repeat_order(self, x): return zip(*x)

    def __build_root(self, event_iter):
        root = deepcopy(self.__root)
        for event in root.findall('event'):
            root.remove(event)
        for event in event_iter:
            root.append(event)
        return root
    
    def __event_duplicator(self, repeats: int, inplace: bool, dup_strat):
        tiled_lists = (self.__event_iter for i in range(repeats))
        dup_events = chain.from_iterable(dup_strat(tiled_lists))
        dup_event_copies = map(deepcopy, dup_events)
        root = self.__build_root(dup_event_copies)
        if inplace == True:
            self.__root = root
            del self.num_events
            return None
        else:
            return _root_to_bytes(root)

def load_lhe(path: str) -> LheData:
    """Load Les Houches file into a LheData object, allowing access
    to the file content, and methods for data manipulation.
    """
    parser = etree.XMLParser(
            resolve_entities=False, remove_comments=False,
            strip_cdata=False)
    try:
        with open(path, 'rb') as file_in:
            tree = etree.parse(file_in, parser=parser)
    except etree.XMLSyntaxError:
        with gzip.open(path, 'rb') as file_in:
            tree = etree.parse(file_in, parser=parser)
    root = tree.getroot()
    content = _root_to_bytes(root)
    return LheData(content)

def _root_to_bytes(root):
    content_invalid = etree.tostring(root)
    unescape_bytes = lambda x: unescape(x.decode()).encode()
    content = unescape_bytes(content_invalid)
    return content

def split(data: LheData, stride: int=1) -> list:
    """Split Les Houches file represented by LheData object into
    multiple LheData objects, each containing a portion of the events.

    Parameters
    ----------
    data : LheData
        LheData object whose content is to be redistributed.
    stride : int
        Number of events per output LheData object, before populating
        the next.

    Returns
    -------
    splits : generator of LheData objects
        Generator object, which can be iterated to obtain each portion
        of the split dataset.
    """
    root = etree.fromstring(data.content)
    root_empty = deepcopy(root)
    for event in root_empty.iter('event'):
        root_empty.remove(event)
    root_new = deepcopy(root_empty)
    for i, event in enumerate(root.iter('event')):
        root_new.append(event)
        if (i + 1) % stride == 0:
            yield LheData(etree.tostring(root_new))
            root_new = deepcopy(root_empty)
