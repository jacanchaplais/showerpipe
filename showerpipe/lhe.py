"""
``showerpipe.lhe``
==================

The ShowerPipe Les Houches functions utilise xml parsing techniques to
redistribute and repeat hard events, outputting valid lhe files.
"""

from typing import Union, BinaryIO, Iterator, Optional
import io
import re
import gzip
import requests  # type: ignore
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy
from itertools import chain
from functools import cached_property
from xml.sax.saxutils import unescape
from urllib.parse import urlparse

from lxml import etree  # type: ignore
from lxml.etree import ElementBase


__all__ = ["source_adapter", "load_lhe", "count_events", "split", "LheData"]

_LHE_STORAGE = Union[Path, str, bytes]

_parse_kwargs = dict(
    resolve_entities=False,
    remove_comments=False,
    strip_cdata=False,
)


@contextmanager
def source_adapter(source: _LHE_STORAGE) -> Iterator[BinaryIO]:
    """Context manager to provide a consistent adapter interface for LHE
    data stored in various formats.

    :group: LesHouches

    Parameters
    ----------
    source : Pathlike, string, or bytes
        The variable or filepath containing the LHE data. May be a path,
        url, string, or bytes object. Gzip compression is allowed.

    Returns
    -------
    LHE data : io.BytesIO, io.BufferedReader
        File-like object containing the Les Houches data. Interface is
        io.BytesIO if source is string or bytestring, and
        io.BufferedReader if source is filepath.
    """
    is_bytes = isinstance(source, bytes)
    is_str = isinstance(source, str)
    is_path = (is_str or isinstance(source, Path)) and Path(source).exists()
    is_url = False
    if is_str:
        is_url = bool(urlparse(source).netloc)  # type: ignore
    if is_path:  # provide a file-object referring to the actual file
        path = Path(source)  # type: ignore
        try:
            with open(path, "r") as lhe_filecheck:
                lhe_filecheck.read(1)
            lhe_file = open(path, "rb")
            yield lhe_file
        except UnicodeDecodeError:
            lhe_file = gzip.open(path, "rb")  # type: ignore
            lhe_file.read(1)
            lhe_file.seek(0)
            yield lhe_file  # type: ignore
        finally:
            lhe_file.close()  # type: ignore
    elif is_str or is_bytes or is_url:  # create a BytesIO file-object
        if is_url:
            lhe_request = requests.get(source)  # type: ignore
            lhe_content = lhe_request.content
            try:
                xml_bytes = gzip.decompress(lhe_content)
            except gzip.BadGzipFile:
                xml_bytes = lhe_content
        elif is_str:
            xml_bytes = source.encode()
        elif is_bytes:
            xml_bytes = source
        out_io = io.BytesIO(xml_bytes)
        try:
            yield out_io
        finally:
            out_io.close()
    else:
        raise NotImplementedError


def load_lhe(path: Union[Path, str]) -> bytes:
    """Load Les Houches file into a bytestring object.

    :group: LesHouches
    """
    parser = etree.XMLParser(**_parse_kwargs)
    with source_adapter(path) as file_in:
        tree = etree.parse(file_in, parser=parser)
    root = tree.getroot()
    content = _root_to_bytes(root)
    return content


def _get_mg_info(header_element: ElementBase) -> ElementBase:
    return header_element.find("MGGenerationInfo")


def _read_num_events(mg_info: ElementBase) -> int:
    re_num = re.findall("\d+", mg_info.text)
    num_events = int(re_num[0])
    return num_events


def _update_num_events(mg_info: ElementBase, new_num: int) -> ElementBase:
    mg_info = deepcopy(mg_info)
    prev_num = _read_num_events(mg_info)
    mg_info_list = mg_info.text.split("\n")
    mg_info_list[1] = mg_info_list[1].replace(str(prev_num), str(new_num))
    mg_info.text = "\n".join(mg_info_list)
    return mg_info


def count_events(source: _LHE_STORAGE) -> int:
    """Returns the number of LHE events stored in ``source``.

    :group: LesHouches

    Parameters
    ----------
    source : pathlib.Path | str | bytes
        File or string object containing Les Houches data.

    Returns
    -------
    count : int
        The number of LHE events.
    """
    with source_adapter(source) as xml_source:
        event_parser = etree.iterparse(
            source=xml_source, tag=("event",), **_parse_kwargs
        )
        num_events = 0
        for _, event in event_parser:
            num_events = num_events + 1
            event.clear()
    return num_events


def split(source: _LHE_STORAGE, stride: int):
    """Generator, splitting LHE file content into separate bytestrings
    representing LHE files, with maximum number of events per bytestring
    equal to stride.

    :group: LesHouches

    Parameters
    ----------
    source : Pathlike, string, or bytes
        The variable or filepath containing the LHE data. May be a path,
        string, or bytes object. The path may be compressed with gzip.

    Returns
    -------
    lhe_split : bytes
        Bytestring, which may be written out as a LHE file, or input
        to a LheData object.

    Notes
    -----
    Particularly useful for large LHE files, which cannot fit in memory.
    """
    with source_adapter(source) as xml_source:
        lhe_root_tagname = "LesHouchesEvents"
        lhe_root_parser = etree.iterparse(
            source=xml_source,
            events=("start",),
            tag=(lhe_root_tagname,),
            **_parse_kwargs
        )
        _, lhe_root_meta = next(lhe_root_parser)
        lhe_root_template = etree.Element(lhe_root_tagname)
        lhe_root_template.set("version", lhe_root_meta.get("version"))

        xml_source.seek(0)
        header_parser = etree.iterparse(
            source=xml_source, tag=("header", "init"), **_parse_kwargs
        )
        _, header = next(header_parser)
        _, init = next(header_parser)
        lhe_root_template.append(header)
        lhe_root_template.append(init)

        mg_info = _get_mg_info(header)
        total_events = _read_num_events(mg_info)

        splits = [stride for _ in range(total_events // stride)]
        split_leftover = total_events % stride
        if split_leftover > 0:
            splits = splits + [split_leftover]

        xml_source.seek(0)
        event_parser = etree.iterparse(
            source=xml_source, tag=("event",), **_parse_kwargs
        )

        for split in splits:
            lhe_root = deepcopy(lhe_root_template)
            split_header = lhe_root[0]
            split_mg_info = _get_mg_info(split_header)
            split_header.replace(
                split_mg_info,
                _update_num_events(split_mg_info, new_num=split),
            )
            for _ in range(split):
                _, event = next(event_parser)
                lhe_root.append(deepcopy(event))
                event.clear(keep_tail=True)
            yield etree.tostring(lhe_root)


def _root_to_bytes(root):
    content_invalid = etree.tostring(root)

    def unescape_bytes(x):
        return unescape(x.decode()).encode()

    content = unescape_bytes(content_invalid)
    return content


class LheData:
    """Container for the Les Houches file content.

    :group: LesHouches

    Attributes
    ----------
    content : bytes
        Bytestring containing the text of the stored file.
    num_events : int
        Number of events stored.

    Methods
    -------
    repeat(repeats, inplace)
        Returns bytes content, with additional events by repetition.
    tile(repeats, inplace)
        Returns bytes content, with additional events by tiling.
    """

    def __init__(self, content: bytes):
        self._root: ElementBase = etree.fromstring(content)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "LheData":
        return cls(etree.fromstring(load_lhe(path)))

    @property
    def content(self) -> bytes:
        """The LHE file contents in bytes."""
        return _root_to_bytes(self._root)

    @content.setter
    def content(self, data):
        del self.num_events
        self._root = etree.fromstring(data)

    @cached_property
    def num_events(self):
        return len(self._root.findall("event"))

    @property
    def _event_iter(self):
        return self._root.iter("event")

    def repeat(self, repeats: int, inplace: bool = False) -> Optional[bytes]:
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
        return self._event_duplicator(
            repeats, inplace, dup_strat=self._repeat_order
        )

    def tile(self, repeats: int, inplace: bool = False) -> Optional[bytes]:
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
        return self._event_duplicator(
            repeats, inplace, dup_strat=self._tile_order
        )

    def _tile_order(self, x):
        return x

    def _repeat_order(self, x):
        return zip(*x)

    def _build_root(self, event_iter):
        root = deepcopy(self._root)
        for event in root.findall("event"):
            root.remove(event)
        for event in event_iter:
            root.append(event)
        return root

    def _event_duplicator(
        self, repeats: int, inplace: bool, dup_strat
    ) -> Optional[bytes]:
        tiled_lists = (self._event_iter for _ in range(repeats))
        dup_events = chain.from_iterable(dup_strat(tiled_lists))
        dup_event_copies = map(deepcopy, dup_events)
        root = self._build_root(dup_event_copies)
        header = root[0]
        mg_info = _get_mg_info(header_element=header)
        prev_num = self.num_events
        new_num = prev_num * repeats
        header.replace(mg_info, _update_num_events(mg_info, new_num))
        if inplace is True:
            self._root = root
            if self.num_events is not None:
                del self.num_events
            return None
        else:
            return _root_to_bytes(root)
